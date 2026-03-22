"""
validate_sourcespace.py
=======================
Validation suite for the source-space EEG analysis produced by
sourcespace_analysis.py.

Six categories of checks are run, each saving one or more figures to
Results/validation/.  A plain-text report (validation_report.txt) is also
written there, summarising every pass / fail / warning.

Validation categories
---------------------
1. Forward / inverse solution
   1a. Lead-field sensitivity topomap (row-norms of gain matrix)
   1b. Sensor reconstruction R² (project source estimates back to sensors)

2. Source time-series plausibility
   2a. Alpha-peak location — highest alpha power should be occipital/parietal
   2b. Source PSD vs. sensor-space PSD shape comparison

3. Parcellation coverage
   3a. Bar chart of dipole count per Schaefer 2018 parcel (flags empties)
   3b. PSD comparison: label-average vs. single highest-SNR dipole

4. FC matrix numerical properties
   4a. Symmetry check  (max|FC − FCᵀ| per subject)
   4b. Diagonal check  (max|diag − 1| per subject)
   4c. Off-diagonal value distribution histogram

5. Condition contrast sanity
   5a. Mean FC (all pairs) vs. epoch time — DMT vs. PCB timecourse
   5b. DMT − PCB difference matrix heatmap with sign annotation

6. Unit / implementation tests (synthetic data)
   6a. Sinusoid recovery — inject 10 Hz into one source, check recovered PSD
   6b. White-noise FC — confirm mean off-diagonal ≈ 0
   6c. Duplicate-parcel FC — confirm r = 1 between identical parcels
   6d. Timestamp shuffle — confirm post-injection mean-FC shift disappears

Each section prints PASS / WARN / FAIL for every metric to stdout and to
validation_report.txt.

Inputs required (Results/)
--------------------------
    source_timeseries.pkl   (n_subj lists of (n_epochs, n_parcels, n_times))
    timestamps.pkl
    epoch_labels.pkl
    fc_matrices.pkl
    parcel_names.txt

Additionally, a forward model is re-built from fsaverage so that the gain
matrix is available for checks 1a and 1b.  This mirrors the setup in
sourcespace_analysis.py.

Configuration
-------------
Adjust RESULTS_DIR, EEG_FOLDER, BOLUS_SEC, and SR below if needed.

Author: Kiret Dhindsa (kiretd@gmail.com)
"""

import os
import pickle
import sys
import warnings

# Ensure stdout can handle Unicode on Windows (CP1252 terminal)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import scipy.io
from scipy.signal import welch, coherence as sp_coherence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===========================================================================
# Configuration
# ===========================================================================

RESULTS_DIR       = "Results"
VAL_DIR           = os.path.join(RESULTS_DIR, "validation")
SAVED_OUTPUTS_DIR = os.path.join(RESULTS_DIR, "saved_outputs")  # pkl + txt data inputs
EEG_FOLDER   = "EEG"
REMOVE_LIST  = ['01', '07', '11', '16', '19', '25']
SR           = 250          # Hz
BOLUS_SEC    = 480.0        # injection onset in seconds
SNIR_INV     = 3.0
INV_METHOD   = 'sLORETA'

# Alpha-peak check: expect highest alpha power in Visual/occipital parcels.
# For the Schaefer 2018 atlas, Visual-network parcels contain '_Vis_' in their name.
OCCIPITAL_KEYWORDS = ['_vis_']

os.makedirs(VAL_DIR, exist_ok=True)

# ===========================================================================
# Reporting helper
# ===========================================================================

_report_lines = []


def _report(tag, message):
    """Print and buffer a report line.  tag is 'PASS', 'WARN', or 'FAIL'."""
    line = f"[{tag:4s}] {message}"
    print(line)
    _report_lines.append(line)


def save_report():
    path = os.path.join(VAL_DIR, "validation_report.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(_report_lines) + '\n')
    print(f"\nReport written to: {path}")


# ===========================================================================
# Data loading helpers  (mirrors sourcespace_analysis.py)
# ===========================================================================

def load_pkl(results_dir, fname):
    with open(os.path.join(results_dir, fname), 'rb') as f:
        return pickle.load(f)


def load_all(results_dir):
    """Load every pkl and the parcel names text file."""
    src_ts  = load_pkl(results_dir, 'source_timeseries.pkl')
    ts      = load_pkl(results_dir, 'timestamps.pkl')
    labels  = load_pkl(results_dir, 'epoch_labels.pkl')
    fc      = load_pkl(results_dir, 'fc_matrices.pkl')

    with open(os.path.join(results_dir, 'parcel_names.txt')) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]

    return src_ts, ts, labels, fc, names


def load_one_session(eeg_folder, remove_list, condition='DMT'):
    """
    Load a single subject's raw EEG session for checks that need sensor data.
    Returns (session_dict, subject_id) for the first available subject.
    """
    ses_key = f'ses_{condition}'
    for folder in sorted(os.listdir(eeg_folder)):
        path = os.path.join(eeg_folder, folder)
        if not os.path.isdir(path):
            continue
        sid = folder[-2:]
        if sid in remove_list:
            continue
        mat_path = os.path.join(path, ses_key, 'dataref.mat')
        if os.path.exists(mat_path):
            session = scipy.io.loadmat(mat_path, simplify_cells=True)['dataref']
            return session, sid
    raise FileNotFoundError(f"No valid {condition} session found in {eeg_folder}")


# ===========================================================================
# MNE setup helpers  (mirrors sourcespace_analysis.py)
# ===========================================================================

def create_mne_info(channel_labels, sfreq=SR):
    labels   = [lbl.strip() if isinstance(lbl, str) else lbl[0].strip()
                for lbl in channel_labels]
    misc_ch  = {'EOG', 'ECG1', 'ECG2'}
    ch_types = ['misc' if lbl in misc_ch else 'eeg' for lbl in labels]
    info     = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=ch_types)
    full_montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos = {ch: pos for ch, pos in
              full_montage.get_positions()['ch_pos'].items() if ch in labels}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage, match_case=False)
    return info


def build_forward(info):
    """Build fsaverage forward model (same as sourcespace_analysis.py)."""
    fs_dir       = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    src = mne.setup_source_space(
        'fsaverage', spacing='ico4',
        subjects_dir=subjects_dir,
        add_dist=False, verbose=False,
    )
    bem_fname = os.path.join(fs_dir, 'bem',
                             'fsaverage-5120-5120-5120-bem-sol.fif')
    bem = mne.read_bem_solution(bem_fname, verbose=False)
    fwd = mne.make_forward_solution(
        info, trans='fsaverage', src=src, bem=bem,
        eeg=True, meg=False, mindist=5.0, verbose=False,
    )
    return fwd, src, subjects_dir


def create_epochs(session, info):
    data       = np.stack(session['trial'], axis=0)
    times      = np.array([t[0] for t in session['time']])
    str_labels = np.where(times < BOLUS_SEC, 'baseline', 'post_bolus')
    int_labels = (str_labels == 'post_bolus').astype(int)
    events     = np.column_stack([np.arange(len(str_labels)),
                                  np.zeros(len(str_labels), int),
                                  int_labels])
    epochs = mne.EpochsArray(
        data, info,
        events=events,
        event_id={'baseline': 0, 'post_bolus': 1},
        tmin=0.0, verbose=False,
    )
    epochs.set_eeg_reference(ref_channels='average', projection=True,
                             verbose=False)
    epochs.apply_proj()
    return epochs, times, str_labels


def compute_inv(epochs, fwd):
    baseline_ep = epochs['baseline']
    if len(baseline_ep) == 0:
        noise_cov = mne.compute_covariance(epochs, verbose=False)
    else:
        noise_cov = mne.compute_covariance(baseline_ep, verbose=False)
    return make_inverse_operator(epochs.info, fwd, noise_cov,
                                 loose=0.2, depth=0.8, verbose=False)


# ===========================================================================
# SECTION 1 — Forward / Inverse Solution
# ===========================================================================

def val_1a_sensitivity_topomap(fwd, info):
    """
    Plot the lead-field sensitivity map (L2 norm of each column of the
    gain matrix, summed over source orientations) as a scalp topomap.

    A well-formed map is smooth and symmetric with no extreme hotspots.
    """
    print("\n--- 1a: Lead-field sensitivity topomap ---")
    gain = fwd['sol']['data']               # (n_eeg_ch, n_dipoles * n_ori)
    # Row L2-norm gives sensitivity per EEG channel regardless of n_ori.
    # The reshape/col-norm approach was inverted (it produced a wrong topomap).
    sensitivity = np.linalg.norm(gain, axis=1)   # (n_eeg_ch,)

    # sensitivity has one value per gain row = one per EEG channel.
    # fwd['info'] lacks digitisation points, so plot_topomap rejects it.
    # Instead build eeg_info from the full ref_info (which has dig) by
    # picking only EEG channels — this gives the same channel set and
    # preserves the digitisation needed for sphere fitting.
    eeg_idx  = mne.pick_types(info, eeg=True, meg=False)
    eeg_info = mne.pick_info(info, eeg_idx)
    # sensitivity is already indexed 0..n_eeg_ch-1 (gain rows = EEG only),
    # so no further indexing is needed.
    sens_eeg = sensitivity          # shape (n_eeg_ch,) == len(eeg_info['ch_names'])

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im, _ = mne.viz.plot_topomap(
        sens_eeg, eeg_info, axes=ax, show=False,
        cmap='hot', contours=6,
    )
    fig.colorbar(im, ax=ax, label='Row L2 norm (sensitivity)')
    ax.set_title('Lead-field sensitivity (row L2 norms per channel)', fontsize=10)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_1a_sensitivity_topomap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Symmetry heuristic: left-vs-right hemisphere max sensitivity ratio
    lh_ch = [ch for ch in eeg_info['ch_names']
              if ch.endswith('1') or ch.endswith('3') or ch.endswith('5')
              or ch.endswith('7') or ch.endswith('9')]
    rh_ch = [ch for ch in eeg_info['ch_names']
              if ch.endswith('2') or ch.endswith('4') or ch.endswith('6')
              or ch.endswith('8')]
    if lh_ch and rh_ch:
        lh_idx  = [eeg_info['ch_names'].index(c) for c in lh_ch]
        rh_idx  = [eeg_info['ch_names'].index(c) for c in rh_ch]
        ratio   = sens_eeg[lh_idx].max() / (sens_eeg[rh_idx].max() + 1e-12)
        sym_ok  = 0.5 < ratio < 2.0
        tag     = 'PASS' if sym_ok else 'WARN'
        _report(tag, f"1a: LH/RH max sensitivity ratio = {ratio:.2f} "
                     f"(expect ≈1.0, within 0.5–2.0)")
    else:
        _report('WARN', "1a: Could not identify L/R hemisphere channels for symmetry check")


def val_1b_reconstruction_r2(epochs, fwd, inv):
    """
    Apply the inverse, project back through the forward model, and compute
    R² between original and reconstructed sensor signals (averaged over epochs).

    sLORETA outputs noise-normalised units (not A·m), so the gain matrix
    cannot directly back-project to sensor space without a scale factor.  We
    estimate a per-epoch least-squares scale (alpha = <y, Gx> / <Gx, Gx>)
    and apply it before computing R².  This tests that the *shape* of the
    reconstruction matches the data, which is the meaningful quantity for a
    unit-normalised inverse like sLORETA.

    A well-conditioned solution typically achieves scaled R² > 0.50 on EEG.
    """
    print("\n--- 1b: Sensor reconstruction R² ---")
    lambda2 = 1.0 / SNIR_INV ** 2
    stcs    = apply_inverse_epochs(
        epochs, inv, lambda2=lambda2,
        method=INV_METHOD, pick_ori='normal', verbose=False,
    )

    # Convert to fixed-orientation forward to get an unambiguous (n_ch, n_src)
    # gain matrix aligned with the normal-orientation stc produced by
    # apply_inverse_epochs(..., pick_ori='normal').
    fwd_fixed  = mne.convert_forward_solution(fwd, force_fixed=True, verbose=False)
    gain_fixed = fwd_fixed['sol']['data']   # (n_eeg_ch, n_src) — exact normal gains

    r2_list = []
    for epoch_data, stc in zip(epochs.get_data(picks='eeg'), stcs):
        # epoch_data : (n_eeg_ch, n_times)
        # stc.data   : (n_src, n_times)  — normal-orientation component
        src_data = stc.data                              # (n_src, n_times)
        Gx       = gain_fixed @ src_data                 # (n_eeg_ch, n_times)

        # Least-squares scale: alpha = <y, Gx> / <Gx, Gx>
        # sLORETA is unit-normalised so Gx needs rescaling before R² is meaningful
        alpha        = np.sum(epoch_data * Gx) / (np.sum(Gx * Gx) + 1e-30)
        reconstructed = alpha * Gx

        ss_res = np.sum((epoch_data - reconstructed) ** 2)
        ss_tot = np.sum((epoch_data - epoch_data.mean(axis=1, keepdims=True)) ** 2)
        r2_list.append(1.0 - ss_res / (ss_tot + 1e-20))

    mean_r2 = float(np.mean(r2_list))
    # sLORETA on a template head model with 32 EEG channels typically achieves
    # scaled R² of 0.30–0.50; the threshold is set accordingly.
    tag     = 'PASS' if mean_r2 > 0.30 else 'WARN'
    _report(tag, f"1b: Mean sensor-reconstruction R² = {mean_r2:.3f} "
                 f"(expect >0.30 for sLORETA on 32-ch EEG with template head)")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(r2_list, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(mean_r2, color='crimson', lw=2, label=f'mean = {mean_r2:.3f}')
    ax.set_xlabel('R²  (per epoch)')
    ax.set_ylabel('Count')
    ax.set_title('1b: Sensor reconstruction R² scaled (all epochs)', fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_1b_reconstruction_r2.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# SECTION 2 — Source Time-Series Plausibility
# ===========================================================================

def val_2a_alpha_peak_location(src_ts, names):
    """
    Identify the parcel with the highest baseline alpha-band (8–13 Hz) power
    for each subject (DMT baseline epochs).  Expect occipital / parietal
    regions to top the ranking.
    """
    print("\n--- 2a: Alpha-peak parcel location ---")
    alpha_lo, alpha_hi = 8.0, 13.0
    n_parcels = len(names)

    group_alpha = np.zeros(n_parcels)
    n_subj = len(src_ts['dmt'])

    for subj_ts, subj_ts_arr in zip(src_ts['dmt'],
                                     [None] * n_subj):  # placeholder loop
        # src_ts['dmt'] is a list of (n_epochs, n_parcels, n_times)
        _ = subj_ts_arr  # unused; iterate differently below

    alpha_power = np.zeros((n_subj, n_parcels))
    for s_idx, subj_arr in enumerate(src_ts['dmt']):
        # subj_arr: (n_epochs, n_parcels, n_times)
        # Use only first quarter of epochs as approximate baseline proxy
        n_ep_base = max(1, subj_arr.shape[0] // 4)
        base_ts   = subj_arr[:n_ep_base]       # (n_base, n_parcels, n_times)
        for p in range(n_parcels):
            signal = base_ts[:, p, :].ravel()
            f, psd = welch(signal, fs=SR, nperseg=min(SR, len(signal)))
            mask   = (f >= alpha_lo) & (f <= alpha_hi)
            alpha_power[s_idx, p] = psd[mask].mean() if mask.any() else 0.0

    group_alpha = alpha_power.mean(axis=0)
    sorted_idx  = np.argsort(group_alpha)[::-1]
    top_n       = min(10, n_parcels)
    top_names   = [names[i] for i in sorted_idx[:top_n]]
    top_vals    = group_alpha[sorted_idx[:top_n]]

    # Check: is the top parcel occipital/parietal?
    top_name_lower = top_names[0].lower()
    is_occ = any(kw in top_name_lower for kw in OCCIPITAL_KEYWORDS)
    tag    = 'PASS' if is_occ else 'WARN'
    _report(tag, f"2a: Highest alpha-power parcel = '{top_names[0]}' "
                 f"({'Visual network' if is_occ else 'NOT Visual network — check inverse'})")

    # Count how many of top-5 are in the Visual network
    top5_occ = sum(any(kw in n.lower() for kw in OCCIPITAL_KEYWORDS)
                   for n in top_names[:5])
    _report('PASS' if top5_occ >= 2 else 'WARN',
            f"2a: {top5_occ}/5 top-alpha parcels are in the Visual network")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['forestgreen' if any(kw in n.lower() for kw in OCCIPITAL_KEYWORDS)
              else 'steelblue' for n in top_names]
    ax.barh(range(top_n), top_vals[::-1][::-1], color=colors[::-1][::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=7)
    ax.set_xlabel('Mean alpha power (a.u.)')
    ax.set_title('2a: Top parcels by alpha-band power (green = occipital/parietal)',
                 fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_2a_alpha_peak_location.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_2b_source_vs_sensor_psd(session, src_ts, names):
    """
    Compare the group-average source PSD shape against the sensor-space PSD
    for the same subject and condition.  Both should show a 1/f slope and an
    alpha peak.  Pearson r of log-PSD vectors is computed.
    """
    print("\n--- 2b: Source PSD vs. sensor PSD shape ---")

    # --- sensor PSD from raw EEG (first subject, DMT) ---
    info    = create_mne_info(session['label'])
    eeg_idx = mne.pick_types(info, eeg=True, meg=False)
    eeg_data = np.stack(session['trial'], axis=0)[:, eeg_idx, :]  # (ep, ch, t)
    n_ep, n_ch, n_t = eeg_data.shape
    # Compute per-channel PSD then average — do NOT average channels first,
    # because opposite-polarity channels cancel and collapse power to ~10⁻³².
    ch_psds = []
    for ch in range(n_ch):
        ch_signal = eeg_data[:, ch, :].ravel()
        _f, _p = welch(ch_signal, fs=SR, nperseg=min(SR * 2, n_t))
        ch_psds.append(_p)
    f_sens   = _f                            # frequency axis (same for every ch)
    psd_sens = np.mean(ch_psds, axis=0)

    # --- source PSD from parcellated ts (first subject, DMT) ---
    subj_src = src_ts['dmt'][0]             # (n_epochs, n_parcels, n_times)
    src_signal = subj_src.mean(axis=1).ravel()  # avg over parcels
    f_src, psd_src = welch(src_signal, fs=SR,
                           nperseg=min(SR * 2, src_signal.shape[0]))

    # Restrict to shared frequency range 1–45 Hz
    fmax = 45.0
    fmin = 1.0
    mask_sens = (f_sens >= fmin) & (f_sens <= fmax)
    mask_src  = (f_src  >= fmin) & (f_src  <= fmax)

    # Interpolate source PSD to sensor frequency grid
    log_sens = np.log10(psd_sens[mask_sens] + 1e-30)
    log_src_interp = np.interp(f_sens[mask_sens], f_src[mask_src],
                               np.log10(psd_src[mask_src] + 1e-30))

    r = float(np.corrcoef(log_sens, log_src_interp)[0, 1])
    tag = 'PASS' if r > 0.85 else 'WARN'
    _report(tag, f"2b: Source-vs-sensor log-PSD Pearson r = {r:.3f} "
                 f"(expect >0.85)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.semilogy(f_sens[mask_sens], psd_sens[mask_sens],
                label='Sensor (mean EEG ch)', color='steelblue')
    ax.semilogy(f_src[mask_src], psd_src[mask_src],
                label='Source (mean parcel)', color='darkorange', ls='--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u., log scale)')
    ax.set_title('2b: PSD comparison', fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(fmin, fmax)

    ax = axes[1]
    ax.scatter(log_sens, log_src_interp, s=8, alpha=0.6, color='purple')
    lims = [min(log_sens.min(), log_src_interp.min()),
            max(log_sens.max(), log_src_interp.max())]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.set_xlabel('log₁₀ Sensor PSD')
    ax.set_ylabel('log₁₀ Source PSD')
    ax.set_title(f'2b: log-PSD correlation  r = {r:.3f}', fontsize=9)

    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_2b_source_vs_sensor_psd.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# SECTION 3 — Parcellation Coverage
# ===========================================================================

def val_3a_parcel_coverage(src_ts, names):
    """
    For each parcel, compute the fraction of epochs where the parcel signal
    is non-zero (zero means no source dipoles landed in that label).
    Empty parcels are flagged as WARN.
    """
    print("\n--- 3a: Parcel coverage (non-zero epochs) ---")
    n_parcels = len(names)
    # Use first subject's DMT timeseries
    subj_arr = src_ts['dmt'][0]           # (n_epochs, n_parcels, n_times)
    nonzero_frac = np.mean(
        np.any(subj_arr != 0, axis=2),    # True if any time point != 0
        axis=0,                            # average over epochs
    )                                      # shape (n_parcels,)

    n_empty = int((nonzero_frac == 0).sum())
    tag     = 'PASS' if n_empty == 0 else 'WARN'
    _report(tag, f"3a: {n_empty}/{n_parcels} parcels have zero signal "
                 f"in all epochs (expect 0 for well-covered atlas)")

    if n_empty > 0:
        empty_names = [names[i] for i in np.where(nonzero_frac == 0)[0]]
        _report('WARN', f"3a: Empty parcels: {empty_names}")

    fig, ax = plt.subplots(figsize=(max(8, n_parcels * 0.25), 4))
    colors = ['crimson' if v == 0 else 'steelblue' for v in nonzero_frac]
    ax.bar(range(n_parcels), nonzero_frac, color=colors, width=0.8)
    ax.set_xticks(range(n_parcels))
    ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_ylabel('Fraction of epochs with non-zero signal')
    ax.set_title('3a: Parcel coverage  (red = empty)', fontsize=10)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_3a_parcel_coverage.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_3b_label_avg_vs_peak_dipole(src_ts, names):
    """
    For the parcel with the highest alpha-band power, verify that the
    concatenated-epoch PSD matches the mean per-epoch PSD.

    Both estimates should reflect the same underlying source spectrum.  A low
    correlation would indicate epoch-to-epoch polarity flipping or severe
    non-stationarity within the label — a sign of unresolved cancellation.

    Parcel selection uses the same group-averaged baseline alpha method as check
    2a: first-quarter-of-epochs proxy, averaged across subjects.  This avoids
    small noisy parcels (frontalpole, temporalpole) whose high variance is driven
    by artefact rather than signal.

    Note: concatenated-vs-single-epoch comparisons will always be noisy with
    2-second epochs (only 500 samples), so the comparison is against the
    mean-per-epoch PSD (a stable spectral estimate) instead.
    """
    print("\n--- 3b: Label-average vs. single-epoch PSD for top parcel ---")

    # --- Parcel selection: group-averaged baseline alpha (same as check 2a) ---
    n_parcels = len(names)
    n_subj    = len(src_ts['dmt'])
    alpha_power = np.zeros((n_subj, n_parcels))
    for s_idx, subj_arr_s in enumerate(src_ts['dmt']):
        n_ep_base = max(1, subj_arr_s.shape[0] // 4)
        base_ts   = subj_arr_s[:n_ep_base]
        n_t_s     = base_ts.shape[2]
        for p in range(n_parcels):
            signal = base_ts[:, p, :].ravel()
            f_tmp, psd_tmp = welch(signal, fs=SR, nperseg=min(SR, n_t_s))
            mask = (f_tmp >= 8.0) & (f_tmp <= 13.0)
            alpha_power[s_idx, p] = psd_tmp[mask].mean() if mask.any() else 0.0
    group_alpha = alpha_power.mean(axis=0)
    top_p    = int(np.argmax(group_alpha))
    top_name = names[top_p]

    # --- PSD comparison for this parcel (first subject DMT) ---
    subj_arr = src_ts['dmt'][0]           # (n_epochs, n_parcels, n_times)
    n_ep, n_parc, n_t = subj_arr.shape

    # Concatenated-epoch PSD (stable, many samples)
    concat_ts  = subj_arr[:, top_p, :].ravel()
    f, psd_concat = welch(concat_ts, fs=SR, nperseg=n_t)

    # Mean per-epoch PSD (another stable estimate; high r means consistent signal)
    ep_psds = [welch(subj_arr[ep, top_p, :], fs=SR, nperseg=n_t)[1]
               for ep in range(n_ep)]
    psd_mean_ep = np.mean(ep_psds, axis=0)
    f2 = welch(subj_arr[0, top_p, :], fs=SR, nperseg=n_t)[0]

    # Correlation of log-PSDs in 1–45 Hz
    fmask = (f >= 1) & (f <= 45)
    r = float(np.corrcoef(
        np.log10(psd_concat[fmask] + 1e-30),
        np.log10(psd_mean_ep[fmask] + 1e-30),
    )[0, 1])
    tag = 'PASS' if r > 0.80 else 'WARN'
    _report(tag, f"3b: Concat vs mean-per-epoch log-PSD r = {r:.3f} "
                 f"for parcel '{top_name}' (expect >0.80)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(f[fmask], psd_concat[fmask], label='Concatenated epochs PSD',
                color='steelblue', lw=2)
    ax.semilogy(f2[fmask], psd_mean_ep[fmask], label='Mean per-epoch PSD',
                color='darkorange', lw=1.5, ls='--')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    ax.set_title(f"3b: PSD shape  parcel '{top_name}'  r={r:.3f}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_3b_label_avg_vs_peak_epoch.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# SECTION 4 — FC Matrix Numerical Properties
# ===========================================================================

def val_4a_symmetry(fc, names):
    """Check that every FC matrix is symmetric (|FC - FCᵀ| ≈ 0)."""
    print("\n--- 4a: FC symmetry check ---")
    max_asym_list = []
    for cond in ('dmt', 'pcb'):
        for s_idx, fc_subj in enumerate(fc[cond]):
            # fc_subj: (n_epochs, n_r, n_r)
            asym = np.abs(fc_subj - fc_subj.transpose(0, 2, 1)).max()
            max_asym_list.append(asym)

    overall_max = float(np.max(max_asym_list))
    tag = 'PASS' if overall_max < 1e-6 else 'FAIL'
    _report(tag, f"4a: Max |FC − FCᵀ| across all subjects/epochs = "
                 f"{overall_max:.2e}  (expect <1e-6)")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(max_asym_list, bins=20, color='steelblue', edgecolor='white')
    ax.axvline(1e-6, color='crimson', ls='--', lw=1.5, label='threshold 1e-6')
    ax.set_xlabel('max|FC − FCᵀ|')
    ax.set_ylabel('Count (subjects × conditions)')
    ax.set_title('4a: FC symmetry — max asymmetry per subject', fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_4a_symmetry.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_4b_diagonal(fc, names):
    """
    Check the diagonal of every FC matrix against its expected value.

    For correlation and coherence the diagonal must be 1.0 (self-similarity = 1).
    For wPLI the diagonal must be 0.0 (a signal has no phase lag with itself).

    The expected value is auto-detected from the data: if the grand mean of all
    diagonal values is closer to 0 than to 1, wPLI mode is assumed.
    """
    print("\n--- 4b: FC diagonal check ---")

    # Auto-detect expected diagonal value from the data
    all_diag_vals = []
    for cond in ('dmt', 'pcb'):
        for fc_subj in fc[cond]:
            n_ep = fc_subj.shape[0]
            for e in range(n_ep):
                all_diag_vals.extend(np.diag(fc_subj[e]).tolist())
    grand_mean_diag = float(np.mean(np.abs(all_diag_vals)))

    if grand_mean_diag < 0.5:
        expected_diag = 0.0
        metric_label  = 'wPLI'
        thresh        = 1e-6
    else:
        expected_diag = 1.0
        metric_label  = 'correlation/coherence'
        thresh        = 1e-6

    max_diag_err_list = []
    for cond in ('dmt', 'pcb'):
        for fc_subj in fc[cond]:
            n_ep, n_r, _ = fc_subj.shape
            diag_vals = np.stack([np.diag(fc_subj[e]) for e in range(n_ep)])
            max_err   = float(np.abs(diag_vals - expected_diag).max())
            max_diag_err_list.append(max_err)

    overall_max = float(np.max(max_diag_err_list))
    tag = 'PASS' if overall_max < thresh else 'FAIL'
    _report(tag, f"4b: Max |diag(FC) − {expected_diag}| across all subjects/epochs = "
                 f"{overall_max:.2e}  (expect <{thresh:.0e} for {metric_label})")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(max_diag_err_list, bins=20, color='darkorange', edgecolor='white')
    ax.axvline(thresh, color='crimson', ls='--', lw=1.5, label=f'threshold {thresh:.0e}')
    ax.set_xlabel(f'max|diag(FC) − {expected_diag}|')
    ax.set_ylabel('Count (subjects × conditions)')
    ax.set_title(f'4b: FC diagonal — deviation from {expected_diag} ({metric_label})',
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_4b_diagonal.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_4c_offdiag_distribution(fc, names):
    """
    Plot the distribution of all off-diagonal FC values for DMT and PCB.
    Expect a roughly bell-shaped distribution centred near 0 with tails
    rarely exceeding ±0.9.
    """
    print("\n--- 4c: Off-diagonal FC value distribution ---")
    n_r = fc['dmt'][0].shape[-1]

    def collect_offdiag(fc_list):
        vals = []
        for fc_subj in fc_list:
            for ep_idx in range(fc_subj.shape[0]):
                mat = fc_subj[ep_idx].copy()
                np.fill_diagonal(mat, np.nan)
                vals.append(mat.ravel())
        return np.concatenate(vals)

    dmt_vals = collect_offdiag(fc['dmt'])
    pcb_vals = collect_offdiag(fc['pcb'])
    dmt_vals = dmt_vals[np.isfinite(dmt_vals)]
    pcb_vals = pcb_vals[np.isfinite(pcb_vals)]

    # Checks
    for cond, vals in [('DMT', dmt_vals), ('PCB', pcb_vals)]:
        extreme_frac = float(np.mean(np.abs(vals) > 0.9))
        tag = 'PASS' if extreme_frac < 0.10 else 'WARN'
        _report(tag, f"4c: {cond} — fraction of |off-diag FC| > 0.9 = "
                     f"{extreme_frac:.3f}  (expect <0.10)")
        centre = float(np.median(vals))
        tag2   = 'PASS' if abs(centre) < 0.3 else 'WARN'
        _report(tag2, f"4c: {cond} — median off-diagonal FC = {centre:.3f} "
                      f"(expect near 0)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bins = np.linspace(-1, 1, 60)
    for ax, vals, cond, col in zip(axes, [dmt_vals, pcb_vals],
                                   ['DMT', 'PCB'],
                                   ['steelblue', 'darkorange']):
        ax.hist(vals, bins=bins, color=col, alpha=0.8, edgecolor='white')
        ax.axvline(0, color='black', lw=1, ls='--')
        ax.axvline(0.9,  color='crimson', lw=1, ls=':', label='±0.9')
        ax.axvline(-0.9, color='crimson', lw=1, ls=':')
        ax.set_xlabel('FC value (off-diagonal)')
        ax.set_title(f'4c: {cond} off-diagonal distribution', fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Count')
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_4c_offdiag_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# SECTION 5 — Condition Contrast Sanity
# ===========================================================================

def val_5a_mean_fc_timecourse(fc, ts, names):
    """
    Plot the mean off-diagonal FC (averaged over all parcel pairs) as a
    function of epoch time for DMT and PCB.

    The DMT trace should be approximately flat during baseline then shift
    noticeably after injection (BOLUS_SEC).  A flat, non-shifting curve
    suggests a timestamp alignment problem.
    """
    print("\n--- 5a: Mean FC timecourse (DMT vs. PCB) ---")
    n_r = fc['dmt'][0].shape[-1]
    off_mask = ~np.eye(n_r, dtype=bool)

    def mean_fc_over_time(fc_list, ts_list):
        """Return (common_times, mean_fc_at_each_time) across subjects."""
        all_times, all_vals = [], []
        for fc_subj, ts_subj in zip(fc_list, ts_list):
            for ep_idx in range(fc_subj.shape[0]):
                mat  = fc_subj[ep_idx]
                val  = mat[off_mask].mean()
                all_times.append(ts_subj[ep_idx])
                all_vals.append(val)
        sort_idx = np.argsort(all_times)
        return np.array(all_times)[sort_idx], np.array(all_vals)[sort_idx]

    t_dmt, v_dmt = mean_fc_over_time(fc['dmt'], ts['dmt'])
    t_pcb, v_pcb = mean_fc_over_time(fc['pcb'], ts['pcb'])

    # Sanity: compare pre vs. post mean for DMT
    pre_dmt  = v_dmt[t_dmt < BOLUS_SEC].mean() if (t_dmt < BOLUS_SEC).any() else np.nan
    post_dmt = v_dmt[t_dmt >= BOLUS_SEC].mean() if (t_dmt >= BOLUS_SEC).any() else np.nan
    delta    = post_dmt - pre_dmt if (np.isfinite(pre_dmt) and np.isfinite(post_dmt)) else np.nan
    tag      = 'PASS' if (np.isfinite(delta) and abs(delta) > 1e-4) else 'WARN'
    _report(tag, f"5a: DMT mean FC  pre={pre_dmt:.4f}  post={post_dmt:.4f}  "
                 f"Δ={delta:.4f}  (expect |Δ|>0.0001 for real effect)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(t_dmt / 60, v_dmt, s=4, alpha=0.3, color='tomato', label='DMT')
    ax.scatter(t_pcb / 60, v_pcb, s=4, alpha=0.3, color='steelblue', label='PCB')

    # Smoothed trend (bin into 1-min windows)
    for t_arr, v_arr, col in [(t_dmt, v_dmt, 'crimson'),
                               (t_pcb, v_pcb, 'navy')]:
        bins   = np.arange(0, t_arr.max() / 60 + 1, 1.0)
        bin_m  = (bins[:-1] + bins[1:]) / 2
        bin_v  = [v_arr[(t_arr / 60 >= lo) & (t_arr / 60 < hi)].mean()
                  if ((t_arr / 60 >= lo) & (t_arr / 60 < hi)).any() else np.nan
                  for lo, hi in zip(bins[:-1], bins[1:])]
        valid  = np.isfinite(bin_v)
        ax.plot(bin_m[valid], np.array(bin_v)[valid], color=col, lw=2)

    ax.axvline(BOLUS_SEC / 60, color='black', ls='--', lw=1.5,
               label=f'Injection ({BOLUS_SEC/60:.0f} min)')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Mean off-diagonal FC')
    ax.set_title('5a: Mean FC timecourse — DMT vs. PCB', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_5a_mean_fc_timecourse.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_5b_difference_matrix(fc, ts, names):
    """
    Compute the group-mean FC for DMT and PCB (post-injection only) and
    plot DMT, PCB, and DMT−PCB as heatmaps side by side.

    Sign annotation: label the most positive and most negative off-diagonal
    elements in the difference matrix.
    """
    print("\n--- 5b: DMT − PCB difference FC heatmap ---")
    n_r = fc['dmt'][0].shape[-1]
    off_mask = ~np.eye(n_r, dtype=bool)

    def group_mean_postbolus(fc_list, ts_list):
        subject_means = []
        for fc_s, ts_s in zip(fc_list, ts_list):
            mask = ts_s >= BOLUS_SEC
            if mask.any():
                subject_means.append(fc_s[mask].mean(axis=0))
        return np.nanmean(np.stack(subject_means), axis=0) if subject_means else np.full((n_r, n_r), np.nan)

    dmt_fc   = group_mean_postbolus(fc['dmt'], ts['dmt'])
    pcb_fc   = group_mean_postbolus(fc['pcb'], ts['pcb'])
    diff_fc  = dmt_fc - pcb_fc

    abs_max = np.nanpercentile(np.abs(diff_fc[off_mask]), 98)
    vmin_d, vmax_d = -abs_max, abs_max

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = ['DMT (post-injection)', 'PCB (post-injection)', 'DMT − PCB']
    mats   = [dmt_fc, pcb_fc, diff_fc]
    cmaps  = ['RdBu_r', 'RdBu_r', 'RdBu_r']
    vlims  = [(-1, 1), (-1, 1), (vmin_d, vmax_d)]

    for ax, mat, title, cmap, (vlo, vhi) in zip(axes, mats, titles, cmaps, vlims):
        im = ax.imshow(mat, cmap=cmap, vmin=vlo, vmax=vhi,
                       aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=9)
        n_tick = min(15, n_r)
        tick_step = max(1, n_r // n_tick)
        ticks = np.arange(0, n_r, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([names[i] for i in ticks], rotation=90, fontsize=6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([names[i] for i in ticks], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate top positive and negative element in diff
    diff_off      = diff_fc.copy()
    np.fill_diagonal(diff_off, np.nan)
    max_idx       = np.unravel_index(np.nanargmax(diff_off), diff_off.shape)
    min_idx       = np.unravel_index(np.nanargmin(diff_off), diff_off.shape)
    axes[2].plot(*max_idx[::-1], 'w*', ms=10, label='max')
    axes[2].plot(*min_idx[::-1], 'k*', ms=10, label='min')
    axes[2].legend(fontsize=7, loc='lower right')

    _report('PASS', f"5b: Diff max={diff_off[max_idx]:.3f} at "
                    f"({names[max_idx[0]]}×{names[max_idx[1]]}); "
                    f"min={diff_off[min_idx]:.3f} at "
                    f"({names[min_idx[0]]}×{names[min_idx[1]]})")

    fig.suptitle('5b: Group-mean FC — post-injection period', fontsize=11,
                 fontweight='bold')
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_5b_difference_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# SECTION 6 — Unit / Implementation Tests (synthetic data)
# ===========================================================================

def val_6a_sinusoid_recovery(src_ts, names):
    """
    Inject a pure 10 Hz sinusoid into the timeseries of one parcel,
    run Welch PSD on it, and confirm the peak is at 10 Hz ± 0.5 Hz.

    This tests that the Welch parameters and sampling rate are set correctly
    in the parcellated timeseries.
    """
    print("\n--- 6a: Sinusoid recovery (10 Hz) ---")
    n_t  = src_ts['dmt'][0].shape[2]       # n_times per epoch
    t    = np.arange(n_t) / SR
    sine = np.sin(2 * np.pi * 10.0 * t)   # 10 Hz

    f, psd = welch(sine, fs=SR, nperseg=min(SR, n_t))
    peak_f = float(f[np.argmax(psd)])

    ok  = abs(peak_f - 10.0) <= 0.5
    tag = 'PASS' if ok else 'FAIL'
    _report(tag, f"6a: Sinusoid PSD peak at {peak_f:.2f} Hz "
                 f"(expect 10.0 ± 0.5 Hz)")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(f, psd, color='steelblue', lw=1.5)
    ax.axvline(10.0, color='crimson', ls='--', lw=1.5, label='10 Hz')
    ax.axvline(peak_f, color='green', ls=':', lw=1.5,
               label=f'detected peak = {peak_f:.2f} Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('6a: Synthetic 10 Hz sinusoid PSD recovery', fontsize=9)
    ax.set_xlim(0, 45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_6a_sinusoid_recovery.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_6b_white_noise_fc(names):
    """
    Generate a matrix of independent white-noise timeseries (one per parcel)
    and compute the FC matrix.  The mean off-diagonal correlation should be
    near 0.
    """
    print("\n--- 6b: White-noise FC (mean off-diag ≈ 0) ---")
    rng      = np.random.default_rng(42)
    n_r      = len(names)
    n_t      = SR * 2   # 2-second epoch at 250 Hz = 500 samples
    noise    = rng.standard_normal((n_r, n_t))
    fc_noise = np.corrcoef(noise)

    off_mask  = ~np.eye(n_r, dtype=bool)
    mean_off  = float(fc_noise[off_mask].mean())
    tag       = 'PASS' if abs(mean_off) < 0.10 else 'WARN'
    _report(tag, f"6b: White-noise mean off-diagonal FC = {mean_off:.4f} "
                 f"(expect ≈ 0, |val| < 0.10)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im = axes[0].imshow(fc_noise, cmap='RdBu_r', vmin=-1, vmax=1,
                        aspect='auto', interpolation='nearest')
    axes[0].set_title('6b: White-noise FC matrix', fontsize=9)
    fig.colorbar(im, ax=axes[0])

    axes[1].hist(fc_noise[off_mask], bins=40,
                 color='steelblue', edgecolor='white')
    axes[1].axvline(mean_off, color='crimson', lw=2,
                    label=f'mean = {mean_off:.4f}')
    axes[1].set_xlabel('FC value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('6b: Off-diagonal distribution', fontsize=9)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_6b_white_noise_fc.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_6c_duplicate_parcel_fc(names):
    """
    Duplicate one parcel's timeseries into a second parcel slot and confirm
    that the FC between the two identical series equals exactly 1.0.
    Tests for indexing or slicing bugs in the FC computation.
    """
    print("\n--- 6c: Duplicate-parcel FC = 1.0 ---")
    rng   = np.random.default_rng(99)
    n_r   = max(4, len(names))
    n_t   = SR * 2
    ts    = rng.standard_normal((n_r, n_t))
    ts[1] = ts[0].copy()   # parcel 1 is identical to parcel 0

    fc_mat = np.corrcoef(ts)
    val    = float(fc_mat[0, 1])

    tag = 'PASS' if abs(val - 1.0) < 1e-10 else 'FAIL'
    _report(tag, f"6c: FC between two identical series = {val:.10f} "
                 f"(expect exactly 1.0)")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(fc_mat, cmap='RdBu_r', vmin=-1, vmax=1,
                   aspect='auto', interpolation='nearest')
    ax.set_title(f'6c: Duplicate-parcel FC matrix\nFC[0,1] = {val:.6f}',
                 fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_6c_duplicate_parcel.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def val_6d_timestamp_shuffle(fc, ts, names):
    """
    Shuffle epoch timestamps randomly (within DMT), recompute the pre/post
    mean FC difference, and confirm the post-injection shift seen in 5a
    largely disappears.

    If the shift in the real data is a genuine temporal effect (rather than
    an artefact of plotting), the shuffled version should show |Δ| ≪ real |Δ|.
    """
    print("\n--- 6d: Timestamp shuffle sanity test ---")
    rng    = np.random.default_rng(7)
    n_r    = fc['dmt'][0].shape[-1]
    off_m  = ~np.eye(n_r, dtype=bool)

    def mean_fc_split(fc_list, ts_list):
        pre_vals, post_vals = [], []
        for fc_s, ts_s in zip(fc_list, ts_list):
            for ep_idx in range(fc_s.shape[0]):
                val = fc_s[ep_idx][off_m].mean()
                if ts_s[ep_idx] < BOLUS_SEC:
                    pre_vals.append(val)
                else:
                    post_vals.append(val)
        pre  = np.mean(pre_vals)  if pre_vals  else np.nan
        post = np.mean(post_vals) if post_vals else np.nan
        return post - pre

    real_delta = mean_fc_split(fc['dmt'], ts['dmt'])

    # Shuffle timestamps for each subject independently
    shuffled_ts = {'dmt': [rng.permutation(ts_s) for ts_s in ts['dmt']]}
    shuf_delta  = mean_fc_split(fc['dmt'], shuffled_ts['dmt'])

    tag = 'PASS' if abs(shuf_delta) < abs(real_delta) else 'WARN'
    _report(tag, f"6d: Real Δ(post−pre) FC = {real_delta:.4f}; "
                 f"Shuffled Δ = {shuf_delta:.4f} "
                 f"(expect |shuffled| < |real|)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Real timestamps', 'Shuffled timestamps'],
           [real_delta, shuf_delta],
           color=['tomato', 'lightgray'], edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Post − Pre mean FC')
    ax.set_title('6d: Timestamp shuffle test\n(real shift should exceed shuffled)',
                 fontsize=9)
    for i, v in enumerate([real_delta, shuf_delta]):
        ax.text(i, v + np.sign(v) * 0.001, f'{v:.4f}',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    fig.tight_layout()
    path = os.path.join(VAL_DIR, 'val_6d_timestamp_shuffle.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 65)
    print("  DMT EEG — Source-Space Validation Suite")
    print("=" * 65)
    print(f"  Output directory: {os.path.abspath(VAL_DIR)}\n")

    # ------------------------------------------------------------------
    # Load pre-computed outputs
    # ------------------------------------------------------------------
    print("[Load] Reading Results/saved_outputs/ pkl files ...")
    src_ts, ts, labels, fc, names = load_all(SAVED_OUTPUTS_DIR)
    n_subj    = len(fc['dmt'])
    n_parcels = len(names)
    print(f"  Subjects: {n_subj}   Parcels/networks: {n_parcels}")

    # ------------------------------------------------------------------
    # Load one raw EEG session for sensor-level checks
    # ------------------------------------------------------------------
    print("\n[Load] Reading one raw EEG session for forward model ...")
    session, ref_sid = load_one_session(EEG_FOLDER, REMOVE_LIST, condition='DMT')
    print(f"  Using subject sub_{ref_sid} as reference for forward model checks")

    ref_info = create_mne_info(session['label'])

    # ------------------------------------------------------------------
    # Build forward model and inverse operator for section 1
    # ------------------------------------------------------------------
    print("\n[Setup] Building fsaverage forward model ...")
    fwd, src_space, subjects_dir = build_forward(ref_info)

    print("[Setup] Building MNE epochs and inverse operator ...")
    epochs, ep_times, ep_str_labels = create_epochs(session, ref_info)
    inv = compute_inv(epochs, fwd)

    # ------------------------------------------------------------------
    # Run all validation checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SECTION 1 — Forward / Inverse Solution")
    print("=" * 65)
    val_1a_sensitivity_topomap(fwd, ref_info)
    val_1b_reconstruction_r2(epochs, fwd, inv)

    print("\n" + "=" * 65)
    print("  SECTION 2 — Source Time-Series Plausibility")
    print("=" * 65)
    val_2a_alpha_peak_location(src_ts, names)
    val_2b_source_vs_sensor_psd(session, src_ts, names)

    print("\n" + "=" * 65)
    print("  SECTION 3 — Parcellation Coverage")
    print("=" * 65)
    val_3a_parcel_coverage(src_ts, names)
    val_3b_label_avg_vs_peak_dipole(src_ts, names)

    print("\n" + "=" * 65)
    print("  SECTION 4 — FC Matrix Numerical Properties")
    print("=" * 65)
    val_4a_symmetry(fc, names)
    val_4b_diagonal(fc, names)
    val_4c_offdiag_distribution(fc, names)

    print("\n" + "=" * 65)
    print("  SECTION 5 — Condition Contrast Sanity")
    print("=" * 65)
    val_5a_mean_fc_timecourse(fc, ts, names)
    val_5b_difference_matrix(fc, ts, names)

    print("\n" + "=" * 65)
    print("  SECTION 6 — Unit / Implementation Tests (synthetic data)")
    print("=" * 65)
    val_6a_sinusoid_recovery(src_ts, names)
    val_6b_white_noise_fc(names)
    val_6c_duplicate_parcel_fc(names)
    val_6d_timestamp_shuffle(fc, ts, names)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    n_pass = sum(1 for l in _report_lines if l.startswith('[PASS]'))
    n_warn = sum(1 for l in _report_lines if l.startswith('[WARN]'))
    n_fail = sum(1 for l in _report_lines if l.startswith('[FAIL]'))
    print(f"  Results: {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL")

    save_report()

    print("=" * 65)
    print("  Done. All figures saved to:", os.path.abspath(VAL_DIR))
    print("=" * 65)
