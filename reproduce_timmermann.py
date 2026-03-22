"""
reproduce_timmermann.py
=======================
Reproduces the main EEG results from Figure 4 of:

    Timmermann C., Roseman L., Haridas S., Rosas F., Luan L., Kettner H.,
    Martell J., Errtizoe D., Tagliazucchi E., Pallavacini C., Girn M.,
    Alamia A., Leech R., Nutt D., Carhart-Harris R. (2023)
    "Human Brain Effects of DMT assessed via fMRI-EEG." PNAS.

Figure 4 panels reproduced:
    (A) Topomaps of DMT-induced spectral power changes and signal diversity.
    (B) Whole-brain power spectra (DMT vs PCB) and LZc comparison (boxplots).
    (D) Temporally resolved delta, alpha, and LZc timecourses with
        cluster-corrected significance shading.
    (E) Forward/backward traveling wave power analysis (DMT vs PCB).

Panels (C) and (F) require subjective ratings data, which is not available
in this dataset and are therefore omitted.

Outputs saved to the Results/ directory:
    - fig4A_topomaps.png
    - fig4B_spectra_lzc.png
    - fig4D_timecourses.png
    - fig4E_traveling_waves.png
    - results_mne_objects.pkl  (dict of MNE EvokedArray objects and arrays)

Author: Kiret Dhindsa (kiretd@gmail.com)
"""

import os
import pickle
import warnings

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
from mne.stats import permutation_cluster_test as pct

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===========================================================================
# Configuration
# ===========================================================================

EEG_FOLDER        = "EEG"          # path to subject data folders
RESULTS_DIR       = "Results"      # where figures and saved objects are written
SAVED_OUTPUTS_DIR = os.path.join(RESULTS_DIR, "saved_outputs")  # pkl data outputs
SR           = 250             # sampling rate (Hz)

# Subjects to exclude (poor data quality, per README and paper)
REMOVE_LIST  = ['01', '07', '11', '16', '19', '25']

# Epoch time boundaries (minutes)
BOLUS_ONSET_MIN  = 8.0   # injection at 8 min
ANALYSIS_END_MIN = 16.0  # analyse up to 16 min post-session-start

# Frequency bands (Hz)
DELTA_BAND  = (1,  4)
THETA_BAND  = (4,  8)
ALPHA_BAND  = (8,  13)
BETA_BAND   = (13, 30)
GAMMA_BAND  = (30, 45)

# Number of permutations for cluster statistics
N_PERMS = 7500

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_OUTPUTS_DIR, exist_ok=True)

# ===========================================================================
# 1. Data Loading
# ===========================================================================

def load_sessions(eeg_folder, remove_list):
    """
    Load DMT and placebo (PCB) EEG sessions from FieldTrip MATLAB files.

    Parameters
    ----------
    eeg_folder : str
        Root directory containing one sub-folder per subject.
    remove_list : list of str
        Two-digit subject IDs to exclude.

    Returns
    -------
    dmt_sessions : list of dict
        One entry per included subject; each dict is the 'dataref' structure.
    pcb_sessions : list of dict
        As above for the placebo session.
    subject_ids : list of str
        Two-digit IDs of the included subjects (same order as the lists above).
    """
    dmt_sessions, pcb_sessions, subject_ids = [], [], []

    for subject_folder in sorted(os.listdir(eeg_folder)):
        subject_path = os.path.join(eeg_folder, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        subject_id = subject_folder[-2:]
        if subject_id in remove_list:
            continue

        dmt_path = os.path.join(subject_path, "ses_DMT", "dataref.mat")
        pcb_path = os.path.join(subject_path, "ses_PCB", "dataref.mat")

        if not (os.path.exists(dmt_path) and os.path.exists(pcb_path)):
            print(f"  [SKIP] sub_{subject_id}: missing session file(s)")
            continue

        dmt_data = scipy.io.loadmat(dmt_path, simplify_cells=True)
        pcb_data = scipy.io.loadmat(pcb_path, simplify_cells=True)

        dmt_sessions.append(dmt_data['dataref'])
        pcb_sessions.append(pcb_data['dataref'])
        subject_ids.append(subject_id)
        print(f"  [OK]   sub_{subject_id}")

    print(f"\nLoaded {len(subject_ids)} subjects: {subject_ids}")
    return dmt_sessions, pcb_sessions, subject_ids


# ===========================================================================
# 2. MNE Setup
# ===========================================================================

def create_mne_info(channel_labels, sfreq=250):
    """
    Build an MNE Info object with a standard 10-20 montage.

    Parameters
    ----------
    channel_labels : array-like of str
        Channel names from the FieldTrip structure.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    info : mne.Info
    """
    labels = [
        lbl.strip() if isinstance(lbl, str) else lbl[0].strip()
        for lbl in channel_labels
    ]
    misc_ch = {'EOG', 'ECG1', 'ECG2'}
    ch_types = ['misc' if lbl in misc_ch else 'eeg' for lbl in labels]

    info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=ch_types)

    full_montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos = {
        ch: pos
        for ch, pos in full_montage.get_positions()['ch_pos'].items()
        if ch in labels
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage, match_case=False)
    return info


def find_bolus_onset_idx(session, bolus_onset_sec=480.0):
    """Return the epoch index at which the bolus injection occurred."""
    times = np.vstack(session['time'])
    return int(np.argmax(times[:, 0] >= bolus_onset_sec))


def create_epochs(session, info, tmin=0.0):
    """
    Create an MNE EpochsArray from a FieldTrip session dict.

    Events are labelled:
        0 = 'baseline'    (pre-bolus)
        1 = 'post_bolus'  (post-bolus)

    Parameters
    ----------
    session : dict
        FieldTrip 'dataref' structure loaded via scipy.io.
    info : mne.Info
    tmin : float
        Start time of each epoch relative to its onset (seconds).

    Returns
    -------
    epochs : mne.EpochsArray
    """
    data      = np.stack(session['trial'], axis=0)   # (n_epochs, n_ch, n_times)
    n         = data.shape[0]
    bolus_idx = find_bolus_onset_idx(session)

    events       = np.column_stack([np.arange(n), np.zeros(n, int), np.ones(n, int)])
    events[:bolus_idx, 2] = 0  # mark pre-bolus epochs as event 0

    epochs = mne.EpochsArray(
        data, info,
        events=events,
        event_id={'baseline': 0, 'post_bolus': 1},
        tmin=tmin,
        verbose=False
    )
    return epochs


def build_epochs(sessions, label=''):
    """Build MNE Info and EpochsArray for every session in a list."""
    infos  = [create_mne_info(sess['label']) for sess in sessions]
    epochs = [create_epochs(sess, info) for sess, info in zip(sessions, infos)]
    print(f"  Built {len(epochs)} {label} epoch objects")
    return epochs


# ===========================================================================
# 3. Shared time-axis helpers
# ===========================================================================

def build_time_index(dmt_sessions, pcb_sessions):
    """
    Build a shared time axis (in minutes) that spans all sessions.

    Returns
    -------
    all_times : list of float
        Sorted unique epoch start times in minutes.
    tidx : dict
        Mapping from time value (float) to integer index.
    tbase : int
        Index of the last baseline epoch (just before bolus onset).
    tend : int
        Index of the epoch at the analysis end time.
    """
    all_times_set = set()
    for sess in dmt_sessions + pcb_sessions:
        for t in sess['time']:
            all_times_set.add(t[0] / 60.0)

    all_times = sorted(all_times_set)
    tidx      = {t: i for i, t in enumerate(all_times)}
    tbase     = next(v for k, v in tidx.items() if k >= BOLUS_ONSET_MIN) - 1
    tend      = next(v for k, v in tidx.items() if k >= ANALYSIS_END_MIN) - 1
    return all_times, tidx, tbase, tend


def align_power_to_time(sessions, metric, tidx, per_channel=False):
    """
    Map per-epoch power values to a shared time axis.

    Parameters
    ----------
    sessions : list of dict
        FieldTrip session dicts (used for epoch start times).
    metric : list of ndarray
        Power values; shape (n_epochs,) if per_channel=False,
        or (n_epochs, n_channels) if per_channel=True.
    tidx : dict
        Time -> index mapping returned by build_time_index().
    per_channel : bool
        Whether metric contains channel-wise values.

    Returns
    -------
    curves : ndarray
        Shape (n_subjects, n_times) or (n_subjects, n_times, n_channels).
    """
    n_times = len(tidx)
    aligned = []

    for sess, pdata in zip(sessions, metric):
        times_min = [t[0] / 60.0 for t in sess['time']]

        if per_channel:
            n_ch  = pdata.shape[-1]
            row   = np.full((n_times, n_ch), np.nan)
            for t_val, y_val in zip(times_min, pdata):
                row[tidx[t_val], :] = y_val
        else:
            y   = pdata.mean(axis=tuple(range(1, pdata.ndim)))
            row = np.full(n_times, np.nan)
            for t_val, y_val in zip(times_min, y):
                row[tidx[t_val]] = y_val

        aligned.append(row)

    return np.stack(aligned, axis=0)


def align_channels(data_arrays, ch_lists):
    """
    Align per-subject arrays to a common channel set, inserting NaN for
    channels not present in a given subject.

    Parameters
    ----------
    data_arrays : list of ndarray, shape (n_epochs, n_channels_i)
    ch_lists : list of list of str

    Returns
    -------
    aligned : list of ndarray, shape (n_epochs, n_all_channels)
    all_chans : list of str
    """
    all_chans  = sorted(set(ch for names in ch_lists for ch in names))
    ch_index   = {ch: i for i, ch in enumerate(all_chans)}
    aligned    = []

    for data, ch_names in zip(data_arrays, ch_lists):
        out = np.full((data.shape[0], len(all_chans)), np.nan)
        for i, ch in enumerate(ch_names):
            out[:, ch_index[ch]] = data[:, i]
        aligned.append(out)

    return aligned, all_chans


def baseline_correct(curves, tbase):
    """
    Subtract the mean of the pre-bolus period from each subject's curve.

    Parameters
    ----------
    curves : ndarray, shape (n_subjects, n_times[, n_channels])
    tbase : int
        Index of the last baseline time-point (exclusive upper bound).

    Returns
    -------
    corrected : ndarray, same shape as curves.
    """
    if curves.ndim == 2:
        baseline = np.nanmean(curves[:, :tbase], axis=1, keepdims=True)
    else:
        baseline = np.nanmean(curves[:, :tbase, :], axis=1, keepdims=True)
    return curves - baseline


def eeg_ch_names(ep):
    """Return the names of EEG-type channels in an Epochs object."""
    return [ch for ch, kind in zip(ep.info.ch_names, ep.get_channel_types())
            if kind == 'eeg']


def mean_band_power(epochs_list, fmin, fmax):
    """
    Compute mean band power (dB) per epoch for every subject.

    Only EEG channels are included; misc channels (EOG/ECG) are excluded.

    Returns a list of ndarrays, each shape (n_epochs, n_eeg_channels).
    """
    out = []
    for ep in epochs_list:
        psd = ep.compute_psd(fmin=fmin, fmax=fmax, picks='eeg', verbose=False)
        # average over frequency, convert to dB, shape → (n_epochs, n_eeg_channels)
        power_db = 10 * np.log10(np.nanmean(psd.get_data(), axis=-1))
        out.append(power_db)
    return out


# ===========================================================================
# 4. LZc computation
# ===========================================================================

def _lz76(binary_seq):
    """
    Lempel-Ziv 1976 complexity of a binary sequence (faithful implementation).
    """
    n    = len(binary_seq)
    i    = 0
    c    = 1
    u    = 1
    v    = 1
    vmax = 1

    while u + v <= n:
        if binary_seq[i + v - 1] == binary_seq[u + v - 1]:
            v += 1
        else:
            if v > vmax:
                vmax = v
            i += 1
            if i == u:
                c    += 1
                u    += vmax
                v    = 1
                i    = 0
                vmax = 1
            else:
                v = 1

    if v != 1:
        c += 1

    return c


def compute_lzc(epochs):
    """
    Compute Lempel-Ziv complexity (LZc) for every EEG channel in every epoch.

    Only EEG-type channels are included; misc channels (EOG/ECG) are excluded.
    Each channel's signal is binarised using its own mean as threshold.

    Parameters
    ----------
    epochs : mne.EpochsArray

    Returns
    -------
    lzc : ndarray, shape (n_epochs, n_eeg_channels)
    """
    data                    = epochs.get_data(picks='eeg')
    n_epochs, n_ch, n_times = data.shape
    lzc                     = np.zeros((n_epochs, n_ch))

    for i in range(n_epochs):
        for j in range(n_ch):
            sig        = data[i, j]
            binary_seq = (sig > sig.mean()).tolist()
            lzc[i, j]  = _lz76(binary_seq)

    return lzc


# ===========================================================================
# 5. Figure 4A — Topomaps
# ===========================================================================

def fig4A_topomaps(epochs_dmt, epochs_pcb, dmt_sessions, pcb_sessions,
                   tidx, tbase, tend, lzc_dmt, lzc_pcb):
    """
    Figure 4A: Topographic maps of DMT-induced changes.

    For spectral bands (delta, alpha, gamma) and LZc, compute the
    post-bolus minus baseline difference, then subtract PCB from DMT.
    Renders one topomap per measure.

    Saves: Results/fig4A_topomaps.png
    Returns: dict of difference arrays keyed by band name.
    """
    print("\n--- Figure 4A: Topomaps ---")
    bands = {
        'Delta (1-4 Hz)' : DELTA_BAND,
        'Alpha (8-13 Hz)': ALPHA_BAND,
        'Gamma (30-45 Hz)': GAMMA_BAND,
    }

    info_ref = epochs_dmt[0].copy().pick('eeg').info

    # Helper: compute mean post-bolus vs baseline difference per channel per subject
    def band_diff(epochs_list, sessions, fmin, fmax):
        pows = mean_band_power(epochs_list, fmin, fmax)
        ch_lists = [eeg_ch_names(ep) for ep in epochs_list]
        aligned, all_chans = align_channels(pows, ch_lists)

        curves = align_power_to_time(sessions, aligned, tidx, per_channel=True)
        corrected = baseline_correct(curves, tbase)
        # mean over post-bolus window and subjects
        return np.nanmean(corrected[:, tbase:tend, :], axis=(0, 1))  # (n_channels,)

    # Build difference maps
    diff_maps = {}
    for band_label, (fmin, fmax) in bands.items():
        dmt_diff = band_diff(epochs_dmt, dmt_sessions, fmin, fmax)
        pcb_diff = band_diff(epochs_pcb, pcb_sessions, fmin, fmax)
        diff_maps[band_label] = dmt_diff - pcb_diff
        print(f"  {band_label}: done")

    # LZc difference
    ch_lists_dmt = [eeg_ch_names(ep) for ep in epochs_dmt]
    ch_lists_pcb = [eeg_ch_names(ep) for ep in epochs_pcb]

    lzc_dmt_al, all_chans_dmt = align_channels(lzc_dmt, ch_lists_dmt)
    lzc_pcb_al, all_chans_pcb = align_channels(lzc_pcb, ch_lists_pcb)

    dmt_lzc_curves = align_power_to_time(dmt_sessions, lzc_dmt_al, tidx, per_channel=True)
    pcb_lzc_curves = align_power_to_time(pcb_sessions, lzc_pcb_al, tidx, per_channel=True)

    dmt_lzc_bc = baseline_correct(dmt_lzc_curves, tbase)
    pcb_lzc_bc = baseline_correct(pcb_lzc_curves, tbase)

    lzc_diff = (np.nanmean(dmt_lzc_bc[:, tbase:tend, :], axis=(0, 1)) -
                np.nanmean(pcb_lzc_bc[:, tbase:tend, :], axis=(0, 1)))
    diff_maps['LZc'] = lzc_diff
    print("  LZc: done")

    # Build a union channel list (sorted, EEG only) that was used in align_channels
    all_eeg_ch = sorted(set(
        ch for ep in epochs_dmt + epochs_pcb for ch in eeg_ch_names(ep)
    ))
    ch_to_idx = {ch: i for i, ch in enumerate(all_eeg_ch)}

    # Channels present in info_ref (montage channels only)
    ref_ch = info_ref.ch_names   # already EEG-only after .pick('eeg')

    # --- Plot ---
    n_maps  = len(diff_maps)
    fig, axes = plt.subplots(1, n_maps, figsize=(4 * n_maps, 4))
    fig.suptitle('Fig 4A — DMT-induced EEG changes (DMT − PCB, post-bolus vs baseline)',
                 fontsize=13, fontweight='bold')

    for ax, (label, diff) in zip(axes, diff_maps.items()):
        # Map diff values (union-channel order) into info_ref channel order
        topo_vals = np.array([
            diff[ch_to_idx[ch]] if ch in ch_to_idx else np.nan
            for ch in ref_ch
        ])
        im, _ = mne.viz.plot_topomap(
            topo_vals,
            info_ref,
            axes=ax,
            show=False,
            sensors=True,
            contours=6,
        )
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'fig4A_topomaps.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return diff_maps


# ===========================================================================
# 6. Figure 4B — Whole-brain spectra & LZc boxplots
# ===========================================================================

def fig4B_spectra_lzc(epochs_dmt, epochs_pcb, lzc_dmt, lzc_pcb):
    """
    Figure 4B: Group-average power spectra and LZc boxplots.

    Left panel:  Mean ± SD PSD (2–30 Hz) for DMT vs PCB (all EEG channels).
    Right panel: LZc distributions (DMT vs PCB) as boxplots.

    Saves: Results/fig4B_spectra_lzc.png
    """
    print("\n--- Figure 4B: Spectra & LZc ---")

    # --- Compute PSD ---
    def group_psd(epochs_list, fmin=2, fmax=30):
        psds = [ep.compute_psd(fmin=fmin, fmax=fmax, picks='eeg', verbose=False).average()
                for ep in epochs_list]
        psd_data  = np.array([p.get_data().mean(axis=0) for p in psds])  # (n_subj, n_freqs)
        freqs     = psds[0].freqs
        return psd_data, freqs

    psd_dmt, freqs = group_psd(epochs_dmt)
    psd_pcb, _     = group_psd(epochs_pcb)

    mean_dmt = psd_dmt.mean(axis=0)
    std_dmt  = psd_dmt.std(axis=0)
    mean_pcb = psd_pcb.mean(axis=0)
    std_pcb  = psd_pcb.std(axis=0)

    # --- LZc pooled across epochs and channels, per subject ---
    lzc_dmt_flat = [arr.mean() for arr in lzc_dmt]
    lzc_pcb_flat = [arr.mean() for arr in lzc_pcb]

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fig 4B — Whole-brain power spectra and LZc',
                 fontsize=13, fontweight='bold')

    # Power spectra
    ax1.semilogy(freqs, mean_dmt, color='crimson', label='DMT', linewidth=2)
    ax1.fill_between(freqs, mean_dmt - std_dmt, mean_dmt + std_dmt,
                     color='crimson', alpha=0.25)
    ax1.semilogy(freqs, mean_pcb, color='steelblue', label='Placebo', linewidth=2)
    ax1.fill_between(freqs, mean_pcb - std_pcb, mean_pcb + std_pcb,
                     color='steelblue', alpha=0.25)
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Power (µV²/Hz)', fontsize=11)
    ax1.set_title('Group-mean PSD (all EEG channels)', fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    # Shade frequency bands
    for (flo, fhi), colour, name in [
        (DELTA_BAND,  '#AAEEFF', 'δ'),
        (ALPHA_BAND,  '#FFDDAA', 'α'),
        (GAMMA_BAND,  '#EECCFF', 'γ'),
    ]:
        ax1.axvspan(flo, fhi, color=colour, alpha=0.3, label=name)
    ax1.legend(fontsize=9)

    # LZc boxplot
    bp = ax2.boxplot(
        [lzc_dmt_flat, lzc_pcb_flat],
        labels=['DMT', 'Placebo'],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='black', linewidth=2),
    )
    bp['boxes'][0].set_facecolor('crimson')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('steelblue')
    bp['boxes'][1].set_alpha(0.6)

    # Overlay individual subject points
    x_jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(lzc_dmt_flat))
    ax2.scatter(1 + x_jitter, lzc_dmt_flat,  color='crimson',   alpha=0.7, zorder=3, s=30)
    ax2.scatter(2 + x_jitter, lzc_pcb_flat,  color='steelblue', alpha=0.7, zorder=3, s=30)

    ax2.set_ylabel('Mean LZc', fontsize=11)
    ax2.set_title('Signal complexity (LZc)', fontsize=10)
    ax2.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'fig4B_spectra_lzc.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


# ===========================================================================
# 7. Figure 4D — Timecourses with cluster significance
# ===========================================================================

def _cluster_significance_mask(x1, x2, n_perms=N_PERMS, adjacency=None):
    """
    Run a spatio-temporal paired cluster test on (x1 - x2).

    Parameters
    ----------
    x1, x2 : ndarray, shape (n_subjects, n_times[, n_channels])
    n_perms : int
    adjacency : sparse matrix or None

    Returns
    -------
    sig_mask : ndarray of bool, shape (n_times,)
        True where at least one significant cluster exists.
    """
    diff = x1 - x2

    if diff.ndim == 3:
        T_obs, clusters, cluster_p, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            diff, adjacency=adjacency, n_permutations=n_perms, verbose=False
        )
        sig_mask = np.zeros(diff.shape[1], dtype=bool)
        for clus, p in zip(clusters, cluster_p):
            if p < 0.05:
                sig_mask[clus[0]] = True
    else:
        T_obs, clusters, cluster_p, _ = mne.stats.permutation_cluster_1samp_test(
            diff, n_permutations=n_perms, verbose=False
        )
        sig_mask = np.zeros(diff.shape[1], dtype=bool)
        for clus, p in zip(clusters, cluster_p):
            if p < 0.05:
                sig_mask[clus[0]] = True

    return sig_mask


def _plot_timecourse(ax, all_times, dmt_curves, pcb_curves, sig_mask,
                     tbase, label='Power', colour_dmt='crimson',
                     colour_pcb='steelblue'):
    """Plot a single timecourse panel with significance shading."""
    mean_d = np.nanmean(dmt_curves, axis=0)
    std_d  = np.nanstd(dmt_curves, axis=0)
    mean_p = np.nanmean(pcb_curves, axis=0)
    std_p  = np.nanstd(pcb_curves, axis=0)

    t = np.array(all_times)

    ax.plot(t, gaussian_filter1d(mean_d, sigma=1), color=colour_dmt, label='DMT', linewidth=2)
    ax.fill_between(t,
                    gaussian_filter1d(mean_d - std_d, sigma=1),
                    gaussian_filter1d(mean_d + std_d, sigma=1),
                    color=colour_dmt, alpha=0.2)

    ax.plot(t, gaussian_filter1d(mean_p, sigma=1), color=colour_pcb, label='Placebo', linewidth=2)
    ax.fill_between(t,
                    gaussian_filter1d(mean_p - std_p, sigma=1),
                    gaussian_filter1d(mean_p + std_p, sigma=1),
                    color=colour_pcb, alpha=0.2)

    # Significance shading
    if sig_mask is not None and sig_mask.any():
        y_lo, y_hi = ax.get_ylim()
        sig_t = t[sig_mask]
        ax.fill_between(t, y_lo, y_hi,
                        where=sig_mask,
                        color='grey', alpha=0.25, label='p<0.05 (cluster)')

    ax.axvline(x=BOLUS_ONSET_MIN, color='black', linestyle='--', linewidth=1.2)
    ymax = ax.get_ylim()[1]
    ax.text(BOLUS_ONSET_MIN + 0.1, ymax * 0.97, 'Bolus', fontsize=8,
            color='black', va='top')

    ax.set_ylabel(label, fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.4)


def fig4D_timecourses(epochs_dmt, epochs_pcb, dmt_sessions, pcb_sessions,
                      lzc_dmt, lzc_pcb, tidx, tbase, tend, all_times, adjacency):
    """
    Figure 4D: Temporally resolved delta, alpha, and LZc timecourses.

    Includes cluster-corrected significance shading where DMT ≠ Placebo.

    Saves: Results/fig4D_timecourses.png
    """
    print("\n--- Figure 4D: Timecourses ---")

    def get_timecourse(epochs_list, sessions, fmin, fmax):
        pows = mean_band_power(epochs_list, fmin, fmax)
        ch_lists = [eeg_ch_names(ep) for ep in epochs_list]
        aligned, _ = align_channels(pows, ch_lists)
        curves = align_power_to_time(sessions, aligned, tidx, per_channel=True)
        return baseline_correct(curves, tbase)

    print("  Computing delta timecourses ...")
    dmt_delta = get_timecourse(epochs_dmt, dmt_sessions, *DELTA_BAND)
    pcb_delta = get_timecourse(epochs_pcb, pcb_sessions, *DELTA_BAND)

    print("  Computing alpha timecourses ...")
    dmt_alpha = get_timecourse(epochs_dmt, dmt_sessions, *ALPHA_BAND)
    pcb_alpha = get_timecourse(epochs_pcb, pcb_sessions, *ALPHA_BAND)

    print("  Computing LZc timecourses ...")
    ch_lists_dmt = [eeg_ch_names(ep) for ep in epochs_dmt]
    ch_lists_pcb = [eeg_ch_names(ep) for ep in epochs_pcb]
    lzc_dmt_al, _ = align_channels(lzc_dmt, ch_lists_dmt)
    lzc_pcb_al, _ = align_channels(lzc_pcb, ch_lists_pcb)
    dmt_lzc = baseline_correct(
        align_power_to_time(dmt_sessions, lzc_dmt_al, tidx, per_channel=True), tbase)
    pcb_lzc = baseline_correct(
        align_power_to_time(pcb_sessions, lzc_pcb_al, tidx, per_channel=True), tbase)

    # Cluster tests on the post-bolus window only
    print("  Running cluster permutation tests ...")
    window = slice(tbase, tend)

    try:
        sig_delta = _cluster_significance_mask(
            dmt_delta[:, window, :], pcb_delta[:, window, :],
            n_perms=N_PERMS, adjacency=adjacency)
        # Pad back to full time axis
        full_sig_delta = np.zeros(len(all_times), bool)
        full_sig_delta[window] = sig_delta
    except Exception as e:
        print(f"    Delta cluster test failed: {e}")
        full_sig_delta = None

    try:
        sig_alpha = _cluster_significance_mask(
            dmt_alpha[:, window, :], pcb_alpha[:, window, :],
            n_perms=N_PERMS, adjacency=adjacency)
        full_sig_alpha = np.zeros(len(all_times), bool)
        full_sig_alpha[window] = sig_alpha
    except Exception as e:
        print(f"    Alpha cluster test failed: {e}")
        full_sig_alpha = None

    try:
        sig_lzc = _cluster_significance_mask(
            dmt_lzc[:, window, :], pcb_lzc[:, window, :],
            n_perms=N_PERMS)
        full_sig_lzc = np.zeros(len(all_times), bool)
        full_sig_lzc[window] = sig_lzc
    except Exception as e:
        print(f"    LZc cluster test failed: {e}")
        full_sig_lzc = None

    # Average across channels for plotting
    dmt_delta_mn = np.nanmean(dmt_delta, axis=2)
    pcb_delta_mn = np.nanmean(pcb_delta, axis=2)
    dmt_alpha_mn = np.nanmean(dmt_alpha, axis=2)
    pcb_alpha_mn = np.nanmean(pcb_alpha, axis=2)
    dmt_lzc_mn   = np.nanmean(dmt_lzc, axis=2)
    pcb_lzc_mn   = np.nanmean(pcb_lzc, axis=2)

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Fig 4D — Temporally resolved DMT effects (baseline corrected)',
                 fontsize=13, fontweight='bold')

    _plot_timecourse(axes[0], all_times, dmt_delta_mn, pcb_delta_mn,
                     full_sig_delta, tbase, label='Delta power (dB)')
    axes[0].set_title('Delta (1–4 Hz)', fontsize=10)

    _plot_timecourse(axes[1], all_times, dmt_alpha_mn, pcb_alpha_mn,
                     full_sig_alpha, tbase, label='Alpha power (dB)')
    axes[1].set_title('Alpha (8–13 Hz)', fontsize=10)

    _plot_timecourse(axes[2], all_times, dmt_lzc_mn, pcb_lzc_mn,
                     full_sig_lzc, tbase, label='LZc')
    axes[2].set_title('Signal diversity (LZc)', fontsize=10)
    axes[2].set_xlabel('Time (min)', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'fig4D_timecourses.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        'dmt_delta': dmt_delta, 'pcb_delta': pcb_delta,
        'dmt_alpha': dmt_alpha, 'pcb_alpha': pcb_alpha,
        'dmt_lzc':   dmt_lzc,   'pcb_lzc':   pcb_lzc,
    }


# ===========================================================================
# 8. Figure 4E — Traveling wave analysis
# ===========================================================================

def fig4E_traveling_waves(epochs_dmt, epochs_pcb, dmt_sessions, pcb_sessions,
                           tidx, tbase, tend):
    """
    Figure 4E: Forward (FW) and backward (BW) traveling wave power.

    Traveling wave direction is estimated from the spatial gradient of
    instantaneous phase across electrodes arranged along an anterior-to-
    posterior axis. Epochs are classified as predominantly forward (FW,
    front → back) or backward (BW, back → front) based on the sign of
    this phase gradient, and mean power is compared between directions
    and conditions.

    Saves: Results/fig4E_traveling_waves.png
    """
    print("\n--- Figure 4E: Traveling Waves ---")

    # Anterior-to-posterior electrode ordering (standard 10-20 subset)
    AP_ORDER = [
        'Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'Oz', 'O2',
    ]

    def compute_tw_power(epochs_list, sessions, fmin=1, fmax=30):
        """
        Compute mean forward and backward wave power per epoch per subject.

        Returns
        -------
        fw_power : list of ndarray, shape (n_epochs,)
        bw_power : list of ndarray, shape (n_epochs,)
        """
        from scipy.signal import hilbert

        fw_list, bw_list = [], []

        for ep in epochs_list:
            # Select only AP-ordered EEG channels present in this subject
            ch_names = ep.info.ch_names
            ap_chans = [ch for ch in AP_ORDER if ch in ch_names]
            if len(ap_chans) < 4:
                fw_list.append(np.array([np.nan]))
                bw_list.append(np.array([np.nan]))
                continue

            data = ep.copy().pick(ap_chans).get_data()   # (n_ep, n_ch, n_times)
            n_ep, n_ch, n_t = data.shape

            fw_ep = np.zeros(n_ep)
            bw_ep = np.zeros(n_ep)

            for i in range(n_ep):
                # Analytic signal → instantaneous phase for each channel
                phase = np.angle(hilbert(data[i], axis=-1))  # (n_ch, n_times)

                # Spatial gradient along AP axis (finite differences)
                phase_grad = np.diff(phase, axis=0)           # (n_ch-1, n_times)

                # Mean gradient sign per time sample
                mean_grad = np.mean(phase_grad, axis=0)       # (n_times,)

                # FW: propagation from front to back → positive gradient
                fw_mask = mean_grad > 0
                bw_mask = mean_grad < 0

                ep_power = (data[i] ** 2).mean(axis=0)        # (n_times,)
                fw_ep[i] = ep_power[fw_mask].mean() if fw_mask.any() else np.nan
                bw_ep[i] = ep_power[bw_mask].mean() if bw_mask.any() else np.nan

            fw_list.append(fw_ep)
            bw_list.append(bw_ep)

        return fw_list, bw_list

    print("  Computing traveling wave power for DMT ...")
    fw_dmt, bw_dmt = compute_tw_power(epochs_dmt, dmt_sessions)
    print("  Computing traveling wave power for PCB ...")
    fw_pcb, bw_pcb = compute_tw_power(epochs_pcb, pcb_sessions)

    # Align to shared time axis and baseline correct
    def tc(power_list, sessions):
        curves = align_power_to_time(sessions, power_list, tidx, per_channel=False)
        return baseline_correct(curves, tbase)

    fw_dmt_bc = tc(fw_dmt, dmt_sessions)
    bw_dmt_bc = tc(bw_dmt, dmt_sessions)
    fw_pcb_bc = tc(fw_pcb, pcb_sessions)
    bw_pcb_bc = tc(bw_pcb, pcb_sessions)

    all_t = np.array(sorted(tidx.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Fig 4E — Forward / Backward Traveling Wave Power (baseline corrected)',
                 fontsize=13, fontweight='bold')

    for ax, (dmt_bc, pcb_bc, title) in zip(axes, [
        (fw_dmt_bc, fw_pcb_bc, 'Forward Wave (FW)'),
        (bw_dmt_bc, bw_pcb_bc, 'Backward Wave (BW)'),
    ]):
        mean_d = gaussian_filter1d(np.nanmean(dmt_bc, axis=0), sigma=1)
        std_d  = gaussian_filter1d(np.nanstd(dmt_bc,  axis=0), sigma=1)
        mean_p = gaussian_filter1d(np.nanmean(pcb_bc, axis=0), sigma=1)
        std_p  = gaussian_filter1d(np.nanstd(pcb_bc,  axis=0), sigma=1)

        ax.plot(all_t, mean_d, color='crimson',   label='DMT',     linewidth=2)
        ax.fill_between(all_t, mean_d - std_d, mean_d + std_d,
                        color='crimson', alpha=0.2)
        ax.plot(all_t, mean_p, color='steelblue', label='Placebo', linewidth=2)
        ax.fill_between(all_t, mean_p - std_p, mean_p + std_p,
                        color='steelblue', alpha=0.2)

        ax.axvline(x=BOLUS_ONSET_MIN, color='black', linestyle='--', linewidth=1.2)
        ax.set_xlabel('Time (min)', fontsize=11)
        ax.set_ylabel('Power (µV²)', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'fig4E_traveling_waves.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    return {
        'fw_dmt': fw_dmt_bc, 'bw_dmt': bw_dmt_bc,
        'fw_pcb': fw_pcb_bc, 'bw_pcb': bw_pcb_bc,
    }


# ===========================================================================
# 9. Save MNE data objects
# ===========================================================================

def save_results(epochs_dmt, epochs_pcb, timecourse_results, tw_results,
                 diff_maps, subject_ids):
    """
    Save key analysis results to disk as a pickle dictionary.

    Contents of the saved file (Results/results_mne_objects.pkl):
        'epochs_dmt'        : list of mne.EpochsArray (one per subject, DMT)
        'epochs_pcb'        : list of mne.EpochsArray (one per subject, PCB)
        'subject_ids'       : list of str
        'timecourses'       : dict of ndarray (from fig4D)
        'traveling_waves'   : dict of ndarray (from fig4E)
        'topomap_diffs'     : dict of ndarray (from fig4A)
    """
    out = {
        'epochs_dmt':      epochs_dmt,
        'epochs_pcb':      epochs_pcb,
        'subject_ids':     subject_ids,
        'timecourses':     timecourse_results,
        'traveling_waves': tw_results,
        'topomap_diffs':   diff_maps,
    }
    out_path = os.path.join(SAVED_OUTPUTS_DIR, 'results_mne_objects.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"\n  Saved MNE result objects: {out_path}")


# ===========================================================================
# 10. Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  DMT EEG Analysis — Timmermann et al. 2023 Figure 4")
    print("=" * 60)

    # --- Load data ---
    print("\n[1] Loading EEG data ...")
    dmt_sessions, pcb_sessions, subject_ids = load_sessions(EEG_FOLDER, REMOVE_LIST)

    if len(subject_ids) == 0:
        raise RuntimeError("No subjects loaded. Check EEG_FOLDER path and data.")

    # --- Build MNE epoch objects ---
    print("\n[2] Building MNE epoch objects ...")
    epochs_dmt = build_epochs(dmt_sessions, label='DMT')
    epochs_pcb = build_epochs(pcb_sessions, label='PCB')

    # --- Shared time axis ---
    print("\n[3] Building shared time axis ...")
    all_times, tidx, tbase, tend = build_time_index(dmt_sessions, pcb_sessions)
    print(f"  Time axis: {len(all_times)} time points | "
          f"Baseline index: {tbase} | Analysis end index: {tend}")

    # --- Adjacency matrix (for spatial cluster tests) ---
    print("\n[4] Computing channel adjacency ...")
    adjacency, _ = mne.channels.find_ch_adjacency(epochs_dmt[0].info, ch_type='eeg')

    # --- Compute LZc for all subjects ---
    print("\n[5] Computing LZc (this may take several minutes) ...")
    lzc_dmt = [compute_lzc(ep) for ep in epochs_dmt]
    lzc_pcb = [compute_lzc(ep) for ep in epochs_pcb]
    print(f"  Done. Example shape: {lzc_dmt[0].shape}")

    # --- Figure 4A ---
    print("\n[6] Generating Figure 4A ...")
    diff_maps = fig4A_topomaps(
        epochs_dmt, epochs_pcb,
        dmt_sessions, pcb_sessions,
        tidx, tbase, tend,
        lzc_dmt, lzc_pcb
    )

    # --- Figure 4B ---
    print("\n[7] Generating Figure 4B ...")
    fig4B_spectra_lzc(epochs_dmt, epochs_pcb, lzc_dmt, lzc_pcb)

    # --- Figure 4D ---
    print("\n[8] Generating Figure 4D (cluster tests — may take a while) ...")
    tc_results = fig4D_timecourses(
        epochs_dmt, epochs_pcb,
        dmt_sessions, pcb_sessions,
        lzc_dmt, lzc_pcb,
        tidx, tbase, tend,
        all_times, adjacency
    )

    # --- Figure 4E ---
    print("\n[9] Generating Figure 4E ...")
    tw_results = fig4E_traveling_waves(
        epochs_dmt, epochs_pcb,
        dmt_sessions, pcb_sessions,
        tidx, tbase, tend
    )

    # --- Save results ---
    print("\n[10] Saving results ...")
    save_results(epochs_dmt, epochs_pcb, tc_results, tw_results,
                 diff_maps, subject_ids)

    print("\n" + "=" * 60)
    print("  All done. Figures saved to:", os.path.abspath(RESULTS_DIR))
    print("=" * 60)
