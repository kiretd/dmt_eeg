"""
sourcespace_effects.py
======================
Maps the sensor-space DMT effects from Figures 4B and 4D of Timmermann et al.
(2023) into source space, identifying which brain regions and networks drive
the observed changes in spectral power and signal complexity.

Requires pre-computed outputs from sourcespace_analysis.py:
    Results/source_timeseries.pkl
    Results/timestamps.pkl
    Results/epoch_labels.pkl
    Results/parcel_names.txt

Three analysis sections
-----------------------
Section 1 — Spectral analysis (maps Fig 4B left panel)
    Welch PSD and IRASA oscillatory/aperiodic decomposition per parcel,
    post-bolus epochs only.  Outputs band-power bar plots, parcel heatmap,
    and group-mean PSD curves per network.

Section 2 — LZc analysis (maps Fig 4B right panel)
    Lempel-Ziv complexity per parcel, post-bolus epochs only.  Outputs
    network-level bar chart and parcel-level bar chart colour-coded by network.

Section 3 — Timecourses (maps Fig 4D)
    Temporally resolved delta power, alpha power, and LZc per network,
    baseline-corrected and cluster-corrected.  Outputs a 7x3 panel grid
    and a summary heatmap.

Network definitions
-------------------
Seven networks from the Schaefer 2018 (100 parcels) cortical atlas are used,
corresponding to the Yeo 7-network parcellation:
    Default Mode, Visual, Dorsal Attention, Somatomotor,
    Frontoparietal, Salience/Ventral Attention, Limbic

Author: Kiret Dhindsa (kiretd@gmail.com)
"""

import os
import sys
import pickle
import warnings

# Windows CP1252 terminal fix — must come before any print calls
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')

import numpy as np
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

import mne
from mne.stats import permutation_cluster_1samp_test

from irasa.IRASA import IRASA

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===========================================================================
# Configuration
# ===========================================================================

RESULTS_DIR       = "Results"
SAVED_OUTPUTS_DIR = os.path.join(RESULTS_DIR, "saved_outputs")  # pkl + txt data inputs
SR               = 250          # sampling rate (Hz)
EPOCH_SAMPLES    = 500          # samples per 2-s epoch at 250 Hz
BOLUS_SEC        = 480.0        # bolus onset in seconds (8 min)
BOLUS_ONSET_MIN  = 8.0          # bolus onset in minutes
ANALYSIS_END_MIN = 16.0         # end of analysis window in minutes

# Frequency bands (Hz)
DELTA_BAND = (1,  4)
ALPHA_BAND = (8,  13)
BETA_BAND  = (13, 30)

# Permutations for cluster statistics
N_PERMS = 7500

# IRASA frequency range
IRASA_FREQS = np.arange(2, 31)  # 2–30 Hz, step 1

# Schaefer 2018 network key → display name mapping.
# Network key is always: label_name.split('-')[0].split('_')[2]
SCHAEFER_NETWORK_NAMES = {
    'Default':     'Default Mode Network',
    'Vis':         'Visual Network',
    'DorsAttn':    'Dorsal Attention Network',
    'SomMot':      'Somatomotor Network',
    'Cont':        'Frontoparietal Network',
    'SalVentAttn': 'Salience Network',
    'Limbic':      'Limbic Network',
}

# Short labels for plot axes
NETWORK_SHORT = {
    'Default Mode Network'      : 'DMN',
    'Visual Network'            : 'VIS',
    'Dorsal Attention Network'  : 'DAN',
    'Somatomotor Network'       : 'SMN',
    'Frontoparietal Network'    : 'FPN',
    'Salience Network'          : 'SAL',
    'Limbic Network'            : 'LIM',
}

# Colour palette — one colour per network
NETWORK_COLOURS = {
    'Default Mode Network'      : '#E63946',
    'Visual Network'            : '#8338EC',
    'Dorsal Attention Network'  : '#06D6A0',
    'Somatomotor Network'       : '#457B9D',
    'Frontoparietal Network'    : '#2A9D8F',
    'Salience Network'          : '#F4A261',
    'Limbic Network'            : '#FFB703',
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================================================================
# 1. Helpers
# ===========================================================================

def load_data():
    """
    Load pre-computed source-space data from Results/.

    Returns
    -------
    ts : dict
        'dmt', 'pcb' : list of ndarray, shape (n_epochs, 100, 500)
        'subject_ids' : list of str
    stamps : dict
        'dmt', 'pcb' : list of ndarray (n_epochs,) — epoch start times in s
    labels : dict
        'dmt', 'pcb' : list of ndarray (n_epochs,) — 'baseline'/'post_bolus'
    parcel_names : list of str
        100 Schaefer 2018 parcel names
    """
    print("Loading source-space data ...")
    with open(os.path.join(SAVED_OUTPUTS_DIR, 'source_timeseries.pkl'), 'rb') as f:
        ts = pickle.load(f)
    with open(os.path.join(SAVED_OUTPUTS_DIR, 'timestamps.pkl'), 'rb') as f:
        stamps = pickle.load(f)
    with open(os.path.join(SAVED_OUTPUTS_DIR, 'epoch_labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    with open(os.path.join(SAVED_OUTPUTS_DIR, 'parcel_names.txt')) as f:
        parcel_names = [line.strip() for line in f if line.strip()]

    n_sub = len(ts['subject_ids'])
    print(f"  {n_sub} subjects: {ts['subject_ids']}")
    print(f"  {len(parcel_names)} parcels, e.g. {parcel_names[0]} … {parcel_names[-1]}")
    return ts, stamps, labels, parcel_names


def assign_networks(parcel_names):
    """
    Map Schaefer 2018 parcel names to Yeo 7-network display names.

    The network key is always the third underscore-delimited field of the
    stem (before the hemisphere suffix), e.g.:
        '7Networks_LH_Default_PFC_1-lh'  ->  key='Default'

    Parcels not matching a known key are placed in 'Other'.

    Parameters
    ----------
    parcel_names : list of str

    Returns
    -------
    network_map : dict
        {network_display_name: [parcel_indices]}  — only non-empty networks
    network_order : list of str
        Ordered list of network display names matching SCHAEFER_NETWORK_NAMES
    parcel_to_network : dict
        {parcel_name: network_display_name}
    """
    network_map       = {name: [] for name in SCHAEFER_NETWORK_NAMES.values()}
    network_map['Other'] = []
    parcel_to_network = {}

    for idx, name in enumerate(parcel_names):
        stem    = name.split('-')[0]          # strip '-lh' / '-rh'
        key     = stem.split('_')[2]          # e.g. 'Default', 'Vis', ...
        display = SCHAEFER_NETWORK_NAMES.get(key, 'Other')
        network_map[display].append(idx)
        parcel_to_network[name] = display

    # Remove empty entries (e.g. 'Other' if every parcel matched)
    network_map   = {k: v for k, v in network_map.items() if v}
    network_order = [n for n in list(SCHAEFER_NETWORK_NAMES.values()) + ['Other']
                     if n in network_map]

    # Report mapping
    print("  Network parcel counts:")
    for net in network_order:
        print(f"    {net}: {len(network_map[net])} parcels")

    return network_map, network_order, parcel_to_network


# ---------------------------------------------------------------------------
# LZc helpers — identical logic to reproduce_timmermann.py
# ---------------------------------------------------------------------------

def _lz76(binary_seq):
    """
    Lempel-Ziv 1976 complexity of a binary sequence.

    Parameters
    ----------
    binary_seq : list of bool / int (0 or 1)

    Returns
    -------
    c : int
        Raw complexity count.
    """
    n    = len(binary_seq)
    i, c, u, v, vmax = 0, 1, 1, 1, 1

    while u + v <= n:
        if binary_seq[i + v - 1] == binary_seq[u + v - 1]:
            v += 1
        else:
            if v > vmax:
                vmax = v
            i += 1
            if i == u:
                c   += 1
                u   += vmax
                v, i, vmax = 1, 0, 1
            else:
                v = 1

    if v != 1:
        c += 1

    return c


def compute_lzc_parcel(data_3d):
    """
    Compute LZc per parcel per epoch.

    Parameters
    ----------
    data_3d : ndarray, shape (n_epochs, n_parcels, n_samples)

    Returns
    -------
    lzc : ndarray, shape (n_epochs, n_parcels)
    """
    n_epochs, n_parcels, _ = data_3d.shape
    lzc = np.zeros((n_epochs, n_parcels))

    for i in range(n_epochs):
        for j in range(n_parcels):
            sig = data_3d[i, j]
            binary_seq = (sig > sig.mean()).tolist()
            lzc[i, j] = _lz76(binary_seq)

    return lzc


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def compute_welch_psd(data_3d, sr=SR, nperseg=EPOCH_SAMPLES):
    """
    Compute Welch PSD per parcel per epoch.

    Uses the full epoch length as nperseg (no windowing sub-division) so that
    frequency resolution matches 1/T = 0.5 Hz for 2-second epochs.

    Parameters
    ----------
    data_3d : ndarray, shape (n_epochs, n_parcels, n_samples)
    sr      : int, sampling rate in Hz
    nperseg : int, samples per Welch segment

    Returns
    -------
    psds  : ndarray, shape (n_epochs, n_parcels, n_freqs)
    freqs : ndarray, shape (n_freqs,)
    """
    n_epochs, n_parcels, _ = data_3d.shape
    freqs = welch(data_3d[0, 0], fs=sr, nperseg=nperseg)[0]
    psds  = np.zeros((n_epochs, n_parcels, len(freqs)))

    for i in range(n_epochs):
        for j in range(n_parcels):
            _, pxx = welch(data_3d[i, j], fs=sr, nperseg=nperseg)
            psds[i, j] = pxx

    return psds, freqs


def band_power_db(psd, freqs, fmin, fmax):
    """
    Mean power in a frequency band, converted to dB (10*log10).

    Parameters
    ----------
    psd   : ndarray, shape (..., n_freqs)
    freqs : ndarray, shape (n_freqs,)
    fmin  : float
    fmax  : float

    Returns
    -------
    power_db : ndarray, shape (...,)  [last axis collapsed]
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    return 10.0 * np.log10(np.nanmean(psd[..., mask], axis=-1))


def aggregate_to_network(data_parcels, network_map, network_order):
    """
    Average parcel-level data into network-level data.

    Parameters
    ----------
    data_parcels : ndarray, shape (..., n_parcels)
        Any shape where the last axis indexes parcels.
    network_map  : dict {network_name: [parcel_indices]}
    network_order : list of str

    Returns
    -------
    data_networks : ndarray, shape (..., n_networks)
    """
    n_nets = len(network_order)
    out_shape = data_parcels.shape[:-1] + (n_nets,)
    data_networks = np.full(out_shape, np.nan)

    for k, net in enumerate(network_order):
        idx = network_map[net]
        if idx:
            data_networks[..., k] = np.nanmean(data_parcels[..., idx], axis=-1)

    return data_networks


def select_post_bolus(data_3d, epoch_labels):
    """
    Return only the post-bolus epochs from a subject's source array.

    Parameters
    ----------
    data_3d      : ndarray, shape (n_epochs, n_parcels, n_samples)
    epoch_labels : ndarray of str, shape (n_epochs,)

    Returns
    -------
    ndarray, shape (n_post, n_parcels, n_samples)
    """
    mask = epoch_labels == 'post_bolus'
    return data_3d[mask]


# ---------------------------------------------------------------------------
# Time axis helpers
# ---------------------------------------------------------------------------

def build_source_time_index(stamps_dmt, stamps_pcb):
    """
    Build a shared time axis (in minutes) spanning all subjects and conditions.

    Epoch timestamps in stamps_* are in seconds; we convert to minutes and
    round to the nearest 2-second epoch boundary (i.e. nearest 1/30 minute)
    to avoid floating-point mismatches across subjects.

    Parameters
    ----------
    stamps_dmt : list of ndarray  — per-subject epoch start times in seconds
    stamps_pcb : list of ndarray

    Returns
    -------
    all_times : ndarray of float  — sorted unique times in minutes
    tidx      : dict {time_min (float): index}
    tbase     : int  — index of last pre-bolus time point
    tend      : int  — index of first time point at/after ANALYSIS_END_MIN
    """
    all_times_set = set()
    for arr in stamps_dmt + stamps_pcb:
        for t_sec in arr:
            t_min = round(t_sec / 60.0, 6)   # convert s → min, stabilise float
            all_times_set.add(t_min)

    all_times = np.array(sorted(all_times_set))
    tidx      = {float(t): i for i, t in enumerate(all_times)}

    # Last index before bolus onset
    pre_bolus = all_times[all_times < BOLUS_ONSET_MIN]
    tbase     = len(pre_bolus) - 1 if len(pre_bolus) > 0 else 0

    # First index at/after analysis end
    post_end  = all_times[all_times >= ANALYSIS_END_MIN]
    tend      = tidx[float(post_end[0])] if len(post_end) > 0 else len(all_times) - 1

    return all_times, tidx, tbase, tend


def align_metric_to_time(stamps, metric_per_epoch, tidx):
    """
    Map per-epoch parcel metric values to a shared time axis.

    Parameters
    ----------
    stamps           : ndarray (n_epochs,) — epoch start times in seconds
    metric_per_epoch : ndarray (n_epochs, n_parcels)
    tidx             : dict {time_min: index}

    Returns
    -------
    row : ndarray (n_times, n_parcels)  — NaN where no epoch at that time
    """
    n_times   = len(tidx)
    n_parcels = metric_per_epoch.shape[1]
    row       = np.full((n_times, n_parcels), np.nan)

    for t_sec, values in zip(stamps, metric_per_epoch):
        t_min = round(t_sec / 60.0, 6)
        if t_min in tidx:
            row[tidx[t_min]] = values

    return row


def baseline_correct_tc(curves, tbase):
    """
    Subtract the mean of the pre-bolus time points (index 0..tbase inclusive).

    Parameters
    ----------
    curves : ndarray (n_subjects, n_times, n_networks)
    tbase  : int  — last pre-bolus time index

    Returns
    -------
    ndarray, same shape as curves
    """
    baseline_mean = np.nanmean(curves[:, :tbase + 1, :], axis=1, keepdims=True)
    return curves - baseline_mean


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def permutation_cluster_test_1d(diffs, n_perms=N_PERMS):
    """
    One-sample permutation cluster test on a 1-D difference timecourse.

    Parameters
    ----------
    diffs   : ndarray (n_subjects, n_times) — paired DMT-PCB difference
    n_perms : int

    Returns
    -------
    sig_mask : ndarray of bool (n_times,) — True at significant cluster time points
    T_obs    : ndarray (n_times,)
    cluster_p_values : list of float
    """
    try:
        T_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
            diffs, n_permutations=n_perms, verbose=False
        )
        sig_mask = np.zeros(diffs.shape[1], dtype=bool)
        for clus, p in zip(clusters, cluster_p):
            if p < 0.05:
                sig_mask[clus[0]] = True
        return sig_mask, T_obs, list(cluster_p)
    except Exception as e:
        print(f"    Cluster test failed: {e}")
        return np.zeros(diffs.shape[1], dtype=bool), None, []


# ===========================================================================
# 2. Section 1 — Source-space spectral analysis
# ===========================================================================

def section1_spectral(ts, stamps, labels, parcel_names,
                      network_map, network_order, parcel_to_network):
    """
    Source-space spectral analysis mapping Fig 4B (left panel) into source space.

    Computes Welch PSD and IRASA oscillatory/aperiodic decomposition per
    parcel using post-bolus epochs only.  Results are aggregated into six
    predefined cortical networks and visualised as:
        - Bar plot of DMT-PCB band power per network (3 bands)
        - Parcel-level heatmap sorted by network
        - Group-mean PSD curves per network (total + oscillatory overlay)

    Parameters
    ----------
    ts, stamps, labels : dicts from load_data()
    parcel_names       : list of str (68 parcels)
    network_map        : dict {network_name: [parcel_indices]}
    network_order      : list of str
    parcel_to_network  : dict {parcel_name: network_name}

    Returns
    -------
    results : dict with keys 'band_diff_parcels', 'irasa_osc_diff', etc.
              for use in the summary report.
    """
    print("\n" + "=" * 60)
    print("SECTION 1: Source-space spectral analysis")
    print("=" * 60)

    n_subjects = len(ts['subject_ids'])
    n_parcels  = len(parcel_names)

    # -------------------------------------------------------------------
    # 1a. Welch PSD — average over post-bolus epochs per subject
    # -------------------------------------------------------------------
    print("\n  Computing Welch PSD per subject ...")

    # We need a reference frequency axis: compute from the first subject/epoch
    _sample_epoch = ts['dmt'][0][0, 0, :]          # (500,)
    freqs_welch   = welch(_sample_epoch, fs=SR, nperseg=EPOCH_SAMPLES)[0]
    n_freqs       = len(freqs_welch)

    # Arrays: (n_subjects, n_parcels, n_freqs)
    psd_dmt_subj = np.zeros((n_subjects, n_parcels, n_freqs))
    psd_pcb_subj = np.zeros((n_subjects, n_parcels, n_freqs))

    for s_idx in range(n_subjects):
        sub_id = ts['subject_ids'][s_idx]

        for cond, arr_key, psd_arr in [
            ('DMT', 'dmt', psd_dmt_subj),
            ('PCB', 'pcb', psd_pcb_subj),
        ]:
            data_all   = ts[arr_key][s_idx]        # (n_epochs, 68, 500)
            lab        = labels[arr_key][s_idx]    # (n_epochs,)
            data_post  = select_post_bolus(data_all, lab)
            n_post     = data_post.shape[0]

            if n_post == 0:
                print(f"    WARNING: sub {sub_id} {cond} has no post-bolus epochs")
                psd_arr[s_idx] = np.nan
                continue

            # Accumulate mean PSD across post-bolus epochs
            psd_sum = np.zeros((n_parcels, n_freqs))
            for ep_idx in range(n_post):
                for p_idx in range(n_parcels):
                    _, pxx = welch(data_post[ep_idx, p_idx],
                                   fs=SR, nperseg=EPOCH_SAMPLES)
                    psd_sum[p_idx] += pxx
            psd_arr[s_idx] = psd_sum / n_post

        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == n_subjects:
            print(f"    [{s_idx + 1}/{n_subjects}] done")

    # -------------------------------------------------------------------
    # 1b. IRASA — oscillatory component per subject
    # -------------------------------------------------------------------
    print("\n  Running IRASA per subject ...")

    # Arrays: (n_subjects, n_parcels, n_irasa_freqs)
    n_irasa  = len(IRASA_FREQS)
    osc_dmt  = np.zeros((n_subjects, n_parcels, n_irasa))
    osc_pcb  = np.zeros((n_subjects, n_parcels, n_irasa))
    apc_dmt  = np.zeros((n_subjects, n_parcels, n_irasa))
    apc_pcb  = np.zeros((n_subjects, n_parcels, n_irasa))

    for s_idx in range(n_subjects):
        sub_id = ts['subject_ids'][s_idx]

        for cond, arr_key, osc_arr, apc_arr in [
            ('DMT', 'dmt', osc_dmt, apc_dmt),
            ('PCB', 'pcb', osc_pcb, apc_pcb),
        ]:
            data_all  = ts[arr_key][s_idx]          # (n_epochs, 68, 500)
            lab       = labels[arr_key][s_idx]
            data_post = select_post_bolus(data_all, lab)  # (n_post, 68, 500)

            if data_post.shape[0] == 0:
                osc_arr[s_idx] = np.nan
                apc_arr[s_idx] = np.nan
                continue

            # IRASA expects sig with last axis = time; shape: (n_post, 68, 500)
            try:
                ir = IRASA(data_post, freqs=IRASA_FREQS, samplerate=SR)
                # ir.mixed   shape: (n_post, 68, n_irasa_freqs)
                # ir.fractal shape: (n_post, 68, n_irasa_freqs)
                # Oscillatory = mixed - fractal (in linear power space)
                # Average over post-bolus epochs (axis=0)
                mixed_mean   = np.nanmean(ir.mixed,   axis=0)  # (68, n_freqs)
                fractal_mean = np.nanmean(ir.fractal, axis=0)

                # Clamp to avoid log(0)
                mixed_mean   = np.maximum(mixed_mean,   1e-30)
                fractal_mean = np.maximum(fractal_mean, 1e-30)

                # Oscillatory in log space (same as reproduce_EEG.py line 194)
                osc_arr[s_idx] = (np.log10(mixed_mean)
                                  - np.log10(fractal_mean))
                apc_arr[s_idx] = fractal_mean

            except Exception as e:
                print(f"    IRASA failed for sub {sub_id} {cond}: {e}")
                osc_arr[s_idx] = np.nan
                apc_arr[s_idx] = np.nan

        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == n_subjects:
            print(f"    [{s_idx + 1}/{n_subjects}] IRASA done")

    # -------------------------------------------------------------------
    # 1c. Band power differences (Welch)
    # -------------------------------------------------------------------
    print("\n  Computing band power differences ...")

    bands = {
        'Delta (1-4 Hz)' : DELTA_BAND,
        'Alpha (8-13 Hz)': ALPHA_BAND,
        'Beta (13-30 Hz)': BETA_BAND,
    }
    band_labels = list(bands.keys())

    # Per-subject DMT-PCB band power diff: (n_subjects, n_parcels, n_bands)
    band_diff_parcels = np.zeros((n_subjects, n_parcels, len(bands)))
    for b_idx, (_, (fmin, fmax)) in enumerate(bands.items()):
        dmt_bp = band_power_db(psd_dmt_subj, freqs_welch, fmin, fmax)  # (n_sub, 68)
        pcb_bp = band_power_db(psd_pcb_subj, freqs_welch, fmin, fmax)
        band_diff_parcels[:, :, b_idx] = dmt_bp - pcb_bp

    # Network-level: (n_subjects, n_networks, n_bands)
    band_diff_nets = np.zeros((n_subjects, len(network_order), len(bands)))
    for b_idx in range(len(bands)):
        band_diff_nets[:, :, b_idx] = aggregate_to_network(
            band_diff_parcels[:, :, b_idx], network_map, network_order
        )

    # -------------------------------------------------------------------
    # 1d. IRASA oscillatory band differences
    # -------------------------------------------------------------------
    irasa_freqs = IRASA_FREQS  # integer array 2..30

    irasa_band_diff = np.zeros((n_subjects, n_parcels, len(bands)))
    for b_idx, (_, (fmin, fmax)) in enumerate(bands.items()):
        # fmin/fmax clipped to IRASA range (2-30 Hz)
        fmin_clip = max(fmin, irasa_freqs[0])
        fmax_clip = min(fmax, irasa_freqs[-1])
        if fmin_clip >= fmax_clip:
            irasa_band_diff[:, :, b_idx] = np.nan
            continue
        mask = (irasa_freqs >= fmin_clip) & (irasa_freqs <= fmax_clip)
        dmt_osc_bp = np.nanmean(osc_dmt[:, :, mask], axis=-1)  # (n_sub, 68)
        pcb_osc_bp = np.nanmean(osc_pcb[:, :, mask], axis=-1)
        irasa_band_diff[:, :, b_idx] = dmt_osc_bp - pcb_osc_bp

    irasa_band_diff_nets = np.zeros((n_subjects, len(network_order), len(bands)))
    for b_idx in range(len(bands)):
        irasa_band_diff_nets[:, :, b_idx] = aggregate_to_network(
            irasa_band_diff[:, :, b_idx], network_map, network_order
        )

    # -------------------------------------------------------------------
    # 1e. Plots
    # -------------------------------------------------------------------

    # --- Figure 1A: Band power bar plot per network ---
    print("\n  Plotting band power per network ...")
    _plot_band_power_bars(band_diff_nets, band_labels, network_order,
                          title='Source-space spectral power change (DMT vs PCB)',
                          fname='source_fig4B_network_bandpower.png')

    # --- Figure 1B: Parcel heatmap sorted by network ---
    print("  Plotting parcel heatmap ...")
    _plot_parcel_heatmap(band_diff_parcels, band_labels, parcel_names,
                         parcel_to_network, network_order,
                         fname='source_fig4B_parcel_heatmap.png')

    # --- Figure 1C: PSD curves per network ---
    print("  Plotting PSD curves per network ...")
    _plot_network_psd_curves(psd_dmt_subj, psd_pcb_subj, freqs_welch,
                             osc_dmt, osc_pcb, irasa_freqs,
                             network_map, network_order,
                             fname='source_fig4B_network_psd.png')

    # --- Figure 1D: IRASA oscillatory band power bar plot ---
    print("  Plotting IRASA oscillatory band power ...")
    _plot_band_power_bars(irasa_band_diff_nets, band_labels, network_order,
                          title='Source-space oscillatory power change (IRASA; DMT vs PCB)',
                          fname='source_fig4B_network_irasa.png',
                          ylabel='DMT - PCB oscillatory power (log10 ratio)')

    print("  Section 1 complete.")

    return {
        'band_diff_parcels'     : band_diff_parcels,
        'band_diff_nets'        : band_diff_nets,
        'irasa_band_diff_nets'  : irasa_band_diff_nets,
        'band_labels'           : band_labels,
        'psd_dmt'               : psd_dmt_subj,
        'psd_pcb'               : psd_pcb_subj,
        'freqs_welch'           : freqs_welch,
        'osc_dmt'               : osc_dmt,
        'osc_pcb'               : osc_pcb,
        'irasa_freqs'           : irasa_freqs,
    }


# ---------------------------------------------------------------------------
# Section 1 plotting helpers
# ---------------------------------------------------------------------------

def _significance_stars(p):
    """Return a star string for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def _plot_band_power_bars(band_diff_nets, band_labels, network_order,
                          title, fname,
                          ylabel='DMT - PCB band power (dB)'):
    """
    Bar plot of DMT-PCB band power difference per network, one bar group
    per network, three bars per group (one per band).  Error bars = SEM.
    Significance stars from one-sample t-test vs zero.

    Parameters
    ----------
    band_diff_nets : ndarray (n_subjects, n_networks, n_bands)
    band_labels    : list of str
    network_order  : list of str
    title, fname, ylabel : str
    """
    from scipy.stats import ttest_1samp

    n_nets  = len(network_order)
    n_bands = len(band_labels)
    short   = [NETWORK_SHORT.get(n, n[:3]) for n in network_order]

    band_colours = ['#4575B4', '#D73027', '#FDAE61']  # delta, alpha, beta

    means = np.nanmean(band_diff_nets, axis=0)   # (n_nets, n_bands)
    sems  = np.nanstd(band_diff_nets, axis=0) / np.sqrt(
        np.sum(~np.isnan(band_diff_nets[:, :, 0]), axis=0))[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    x        = np.arange(n_nets)
    width    = 0.22
    offsets  = np.linspace(-(n_bands - 1) * width / 2,
                            (n_bands - 1) * width / 2, n_bands)

    for b_idx, (blabel, col, offset) in enumerate(
            zip(band_labels, band_colours, offsets)):
        bars = ax.bar(x + offset, means[:, b_idx], width=width,
                      color=col, alpha=0.75, label=blabel,
                      yerr=sems[:, b_idx], capsize=3,
                      error_kw=dict(elinewidth=1))

        # Significance stars
        for n_idx in range(n_nets):
            vals = band_diff_nets[:, n_idx, b_idx]
            vals = vals[~np.isnan(vals)]
            if len(vals) >= 3:
                _, p = ttest_1samp(vals, 0)
                stars = _significance_stars(p)
                if stars:
                    bar_top = means[n_idx, b_idx] + sems[n_idx, b_idx]
                    ax.text(x[n_idx] + offset, bar_top + 0.02 * abs(bar_top) + 0.01,
                            stars, ha='center', va='bottom', fontsize=9,
                            color=col)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Network', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Saved: {out}")


def _plot_parcel_heatmap(band_diff_parcels, band_labels, parcel_names,
                         parcel_to_network, network_order, fname):
    """
    Heatmap of DMT-PCB band power difference for all 68 parcels, sorted by
    network.  Rows = parcels, columns = frequency bands.

    Parameters
    ----------
    band_diff_parcels : ndarray (n_subjects, n_parcels, n_bands)
    band_labels       : list of str
    parcel_names      : list of str
    parcel_to_network : dict
    network_order     : list of str
    fname             : str
    """
    # Mean across subjects
    mean_diff = np.nanmean(band_diff_parcels, axis=0)  # (68, n_bands)

    # Sort parcels: by network order, then alphabetically within network
    ordered_indices = []
    network_boundaries = []
    tick_labels = []

    cursor = 0
    for net in network_order:
        net_parcels = [i for i, p in enumerate(parcel_names)
                       if parcel_to_network.get(p) == net]
        net_parcels.sort(key=lambda i: parcel_names[i])
        ordered_indices.extend(net_parcels)
        network_boundaries.append((cursor, cursor + len(net_parcels)))
        cursor += len(net_parcels)
        short = NETWORK_SHORT.get(net, net[:3])
        tick_labels.append((cursor - len(net_parcels) + (len(net_parcels) - 1) / 2,
                             short))

    # Also include 'Other' parcels if any
    other_parcels = [i for i, p in enumerate(parcel_names)
                     if parcel_to_network.get(p) == 'Other']
    if other_parcels:
        ordered_indices.extend(sorted(other_parcels))
        network_boundaries.append((cursor, cursor + len(other_parcels)))
        tick_labels.append((cursor + (len(other_parcels) - 1) / 2, 'Oth'))

    data_sorted = mean_diff[ordered_indices, :]   # (68, n_bands)
    parcel_sorted = [parcel_names[i] for i in ordered_indices]

    vmax = np.nanpercentile(np.abs(data_sorted), 95)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(6, 14))
    im = ax.imshow(data_sorted, aspect='auto', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    # Network boundary lines
    for start, end in network_boundaries:
        ax.axhline(start - 0.5, color='black', linewidth=1.2)

    # Y-axis: parcel names (small font) and network labels on the right
    ax.set_yticks(range(len(parcel_sorted)))
    ax.set_yticklabels(parcel_sorted, fontsize=5.5)
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_title('Parcel-level DMT-PCB spectral power\n(sorted by network)',
                 fontsize=11, fontweight='bold')

    # Network labels on right axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([pos for pos, _ in tick_labels])
    ax2.set_yticklabels([lbl for _, lbl in tick_labels], fontsize=8,
                        fontweight='bold')

    # Colour network separators
    for k, (net, (start, end)) in enumerate(zip(network_order, network_boundaries)):
        col = NETWORK_COLOURS.get(net, 'grey')
        ax.annotate('', xy=(-0.5, end - 0.5), xytext=(-0.5, start - 0.5),
                    xycoords=('axes fraction', 'data'),
                    textcoords=('axes fraction', 'data'),
                    arrowprops=dict(arrowstyle='-', color=col, lw=3))

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('DMT - PCB (dB)', fontsize=9)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def _plot_network_psd_curves(psd_dmt, psd_pcb, freqs_welch,
                              osc_dmt, osc_pcb, irasa_freqs,
                              network_map, network_order, fname):
    """
    One panel per network: group-mean Welch PSD (DMT vs PCB) with shaded SD,
    plus IRASA oscillatory overlay (dashed lines) on a secondary y-axis.

    Parameters
    ----------
    psd_dmt, psd_pcb   : ndarray (n_subjects, n_parcels, n_freqs)
    freqs_welch        : ndarray
    osc_dmt, osc_pcb   : ndarray (n_subjects, n_parcels, n_irasa_freqs)
    irasa_freqs        : ndarray
    network_map        : dict
    network_order      : list of str
    fname              : str
    """
    n_nets  = len(network_order)
    n_cols  = 3
    n_rows  = int(np.ceil(n_nets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), sharex=False)
    axes_flat = axes.flatten()

    for k, net in enumerate(network_order):
        ax = axes_flat[k]
        idx = network_map[net]

        # Welch PSD: average over network parcels → (n_subjects, n_freqs)
        psd_d = psd_dmt[:, idx, :].mean(axis=1)  # (n_sub, n_freqs)
        psd_p = psd_pcb[:, idx, :].mean(axis=1)

        # Restrict to 2–30 Hz for display
        fmask = (freqs_welch >= 2) & (freqs_welch <= 30)
        f_plot = freqs_welch[fmask]
        pd_mn = np.nanmean(psd_d[:, fmask], axis=0)
        pd_sd = np.nanstd(psd_d[:, fmask], axis=0)
        pp_mn = np.nanmean(psd_p[:, fmask], axis=0)
        pp_sd = np.nanstd(psd_p[:, fmask], axis=0)

        ax.semilogy(f_plot, pd_mn, color='crimson', label='DMT', linewidth=1.5)
        ax.fill_between(f_plot, pd_mn - pd_sd, pd_mn + pd_sd,
                        color='crimson', alpha=0.2)
        ax.semilogy(f_plot, pp_mn, color='steelblue', label='Placebo', linewidth=1.5)
        ax.fill_between(f_plot, pp_mn - pp_sd, pp_mn + pp_sd,
                        color='steelblue', alpha=0.2)

        # IRASA oscillatory overlay on twin axis
        ax2 = ax.twinx()
        osc_d_net = osc_dmt[:, idx, :].mean(axis=1)  # (n_sub, n_irasa_freqs)
        osc_p_net = osc_pcb[:, idx, :].mean(axis=1)
        osc_d_mn = np.nanmean(osc_d_net, axis=0)
        osc_p_mn = np.nanmean(osc_p_net, axis=0)
        ax2.plot(irasa_freqs, osc_d_mn, color='crimson',
                 linestyle='--', linewidth=1.0, alpha=0.6, label='DMT osc')
        ax2.plot(irasa_freqs, osc_p_mn, color='steelblue',
                 linestyle='--', linewidth=1.0, alpha=0.6, label='PCB osc')
        ax2.axhline(0, color='grey', linewidth=0.5, linestyle=':')
        ax2.set_ylabel('Osc. power (log10)', fontsize=7, color='grey')
        ax2.tick_params(axis='y', labelsize=7, labelcolor='grey')

        short = NETWORK_SHORT.get(net, net)
        ax.set_title(short, fontsize=10, fontweight='bold',
                     color=NETWORK_COLOURS.get(net, 'black'))
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        ax.set_ylabel('Power', fontsize=8)
        ax.grid(True, which='both', linestyle=':', alpha=0.3)

        # Shade frequency bands
        for (flo, fhi), col_b in [(DELTA_BAND, '#AAEEFF'),
                                   (ALPHA_BAND, '#FFDDAA'),
                                   (BETA_BAND,  '#EECCFF')]:
            ax.axvspan(max(flo, 2), min(fhi, 30), color=col_b,
                       alpha=0.2, zorder=0)

        if k == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Hide any unused panels in the last row
    for idx in range(n_nets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle('Source-space PSD per network (DMT vs PCB)\n'
                 'Solid = Welch total; Dashed = IRASA oscillatory',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Saved: {out}")


# ===========================================================================
# 3. Section 2 — Source-space LZc analysis
# ===========================================================================

def section2_lzc(ts, labels, parcel_names, network_map, network_order,
                 parcel_to_network):
    """
    Source-space Lempel-Ziv complexity analysis mapping Fig 4B (right panel).

    Computes LZc per parcel per post-bolus epoch, averages across epochs,
    and tests the DMT-PCB difference at network level.

    Parameters
    ----------
    ts, labels         : dicts from load_data()
    parcel_names       : list of str
    network_map        : dict
    network_order      : list of str
    parcel_to_network  : dict

    Returns
    -------
    results : dict with keys 'lzc_diff_parcels', 'lzc_diff_nets', etc.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: Source-space LZc analysis")
    print("=" * 60)

    n_subjects = len(ts['subject_ids'])
    n_parcels  = len(parcel_names)

    # (n_subjects, n_parcels) — mean LZc over post-bolus epochs
    lzc_dmt_mean = np.full((n_subjects, n_parcels), np.nan)
    lzc_pcb_mean = np.full((n_subjects, n_parcels), np.nan)

    print("\n  Computing LZc per parcel per subject ...")
    for s_idx in range(n_subjects):
        sub_id = ts['subject_ids'][s_idx]

        for cond, arr_key, lzc_arr in [
            ('DMT', 'dmt', lzc_dmt_mean),
            ('PCB', 'pcb', lzc_pcb_mean),
        ]:
            data_all  = ts[arr_key][s_idx]         # (n_epochs, 68, 500)
            lab       = labels[arr_key][s_idx]
            data_post = select_post_bolus(data_all, lab)

            if data_post.shape[0] == 0:
                print(f"    WARNING: sub {sub_id} {cond} no post-bolus epochs")
                continue

            # compute_lzc_parcel → (n_post, 68); mean over epochs
            lzc_epochs = compute_lzc_parcel(data_post)      # (n_post, 68)
            lzc_arr[s_idx] = np.nanmean(lzc_epochs, axis=0)  # (68,)

        if (s_idx + 1) % 5 == 0 or (s_idx + 1) == n_subjects:
            print(f"    [{s_idx + 1}/{n_subjects}] done")

    # DMT-PCB difference
    lzc_diff_parcels = lzc_dmt_mean - lzc_pcb_mean   # (n_subjects, n_parcels)

    # Network-level aggregation
    lzc_diff_nets = aggregate_to_network(lzc_diff_parcels,
                                         network_map, network_order)
    # (n_subjects, n_networks)

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------
    print("\n  Plotting LZc per network ...")
    _plot_lzc_network_bars(lzc_diff_nets, network_order,
                           fname='source_fig4B_network_lzc.png')

    print("  Plotting LZc parcel-level bar chart ...")
    _plot_lzc_parcel_bars(lzc_diff_parcels, parcel_names,
                          parcel_to_network, network_order,
                          fname='source_fig4B_parcel_lzc.png')

    print("  Section 2 complete.")

    return {
        'lzc_dmt_mean'    : lzc_dmt_mean,
        'lzc_pcb_mean'    : lzc_pcb_mean,
        'lzc_diff_parcels': lzc_diff_parcels,
        'lzc_diff_nets'   : lzc_diff_nets,
    }


# ---------------------------------------------------------------------------
# Section 2 plotting helpers
# ---------------------------------------------------------------------------

def _plot_lzc_network_bars(lzc_diff_nets, network_order,
                            fname='source_fig4B_network_lzc.png'):
    """
    Bar plot of DMT-PCB LZc difference per network, with SEM error bars and
    one-sample t-test significance stars.

    Parameters
    ----------
    lzc_diff_nets : ndarray (n_subjects, n_networks)
    network_order : list of str
    fname         : str
    """
    from scipy.stats import ttest_1samp

    short  = [NETWORK_SHORT.get(n, n[:3]) for n in network_order]
    means  = np.nanmean(lzc_diff_nets, axis=0)
    n_obs  = np.sum(~np.isnan(lzc_diff_nets), axis=0)
    sems   = np.nanstd(lzc_diff_nets, axis=0) / np.sqrt(np.maximum(n_obs, 1))
    colours = [NETWORK_COLOURS.get(net, 'grey') for net in network_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Source-space LZc change per network (DMT vs PCB)',
                 fontsize=13, fontweight='bold')

    x = np.arange(len(network_order))
    bars = ax.bar(x, means, color=colours, alpha=0.75,
                  yerr=sems, capsize=4,
                  error_kw=dict(elinewidth=1.2, ecolor='black'))

    # Overlay individual subject points
    rng = np.random.default_rng(42)
    for n_idx in range(len(network_order)):
        vals = lzc_diff_nets[:, n_idx]
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(x[n_idx] + jitter, vals, color='black', s=18,
                   alpha=0.55, zorder=4)

        # Significance stars
        clean = vals[~np.isnan(vals)]
        if len(clean) >= 3:
            _, p = ttest_1samp(clean, 0)
            stars = _significance_stars(p)
            if stars:
                bar_top = means[n_idx] + sems[n_idx]
                ax.text(x[n_idx], bar_top + 0.005, stars,
                        ha='center', va='bottom', fontsize=11)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=11)
    ax.set_ylabel('DMT - PCB mean LZc', fontsize=10)
    ax.set_xlabel('Network', fontsize=10)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Saved: {out}")


def _plot_lzc_parcel_bars(lzc_diff_parcels, parcel_names,
                           parcel_to_network, network_order,
                           fname='source_fig4B_parcel_lzc.png'):
    """
    Horizontal bar chart of DMT-PCB LZc difference for all 68 parcels,
    bars coloured by network assignment, sorted by network then by value.

    Parameters
    ----------
    lzc_diff_parcels : ndarray (n_subjects, n_parcels)
    parcel_names     : list of str
    parcel_to_network : dict
    network_order    : list of str
    fname            : str
    """
    means = np.nanmean(lzc_diff_parcels, axis=0)  # (68,)
    sems  = (np.nanstd(lzc_diff_parcels, axis=0)
             / np.sqrt(np.sum(~np.isnan(lzc_diff_parcels), axis=0)))

    # Build ordered list: by network then by mean LZc within network
    ordered = []
    for net in network_order:
        parcels_in_net = [(i, p) for i, p in enumerate(parcel_names)
                          if parcel_to_network.get(p) == net]
        parcels_in_net.sort(key=lambda x: means[x[0]])
        ordered.extend(parcels_in_net)

    # 'Other' at the end
    other = [(i, p) for i, p in enumerate(parcel_names)
             if parcel_to_network.get(p) == 'Other']
    other.sort(key=lambda x: means[x[0]])
    ordered.extend(other)

    indices  = [i for i, _ in ordered]
    p_labels = [p for _, p in ordered]
    m_sorted = means[indices]
    s_sorted = sems[indices]
    colours  = [NETWORK_COLOURS.get(parcel_to_network.get(parcel_names[i], 'Other'),
                                     'grey')
                for i in indices]

    fig, ax = plt.subplots(figsize=(8, 16))
    y = np.arange(len(ordered))
    ax.barh(y, m_sorted, xerr=s_sorted, color=colours, alpha=0.75,
            capsize=2, error_kw=dict(elinewidth=0.8))
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(p_labels, fontsize=5.5)
    ax.set_xlabel('DMT - PCB mean LZc', fontsize=10)
    ax.set_title('Parcel-level LZc change (DMT vs PCB)\n(colour = network)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, axis='x', linestyle=':', alpha=0.4)

    # Legend patches for networks
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=NETWORK_COLOURS.get(net, 'grey'),
                               label=NETWORK_SHORT.get(net, net))
               for net in network_order]
    ax.legend(handles=patches, fontsize=7, loc='lower right')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


# ===========================================================================
# 4. Section 3 — Source-space timecourses
# ===========================================================================

def section3_timecourses(ts, stamps, labels, parcel_names,
                         network_map, network_order):
    """
    Source-space timecourse analysis mapping Fig 4D into source space.

    Computes per-epoch delta power, alpha power, and LZc for each parcel,
    maps results to a shared time axis, baseline-corrects, aggregates into
    networks, and runs permutation cluster tests on the post-bolus window.

    Outputs a 6x3 panel figure (networks x metrics) with significance
    shading, and a summary heatmap of DMT-PCB differences over time.

    Parameters
    ----------
    ts, stamps, labels : dicts from load_data()
    parcel_names       : list of str
    network_map        : dict
    network_order      : list of str

    Returns
    -------
    results : dict
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Source-space timecourses")
    print("=" * 60)

    n_subjects = len(ts['subject_ids'])
    n_parcels  = len(parcel_names)

    # -------------------------------------------------------------------
    # 3a. Build shared time axis
    # -------------------------------------------------------------------
    print("\n  Building shared time axis ...")
    all_times, tidx, tbase, tend = build_source_time_index(
        stamps['dmt'], stamps['pcb']
    )
    n_times = len(all_times)
    print(f"  Time axis: {n_times} points, {all_times[0]:.2f}–{all_times[-1]:.2f} min")
    print(f"  tbase={tbase} ({all_times[tbase]:.2f} min), "
          f"tend={tend} ({all_times[tend]:.2f} min)")

    # -------------------------------------------------------------------
    # 3b. Per-epoch metrics → shared time axis
    #     Shapes: (n_subjects, n_times, n_parcels)
    # -------------------------------------------------------------------
    print("\n  Computing per-epoch delta power, alpha power, LZc ...")

    # Pre-allocate
    delta_dmt = np.full((n_subjects, n_times, n_parcels), np.nan)
    delta_pcb = np.full((n_subjects, n_times, n_parcels), np.nan)
    alpha_dmt = np.full((n_subjects, n_times, n_parcels), np.nan)
    alpha_pcb = np.full((n_subjects, n_times, n_parcels), np.nan)
    lzc_dmt   = np.full((n_subjects, n_times, n_parcels), np.nan)
    lzc_pcb   = np.full((n_subjects, n_times, n_parcels), np.nan)

    # Welch frequency axis (computed once)
    _sample = ts['dmt'][0][0, 0, :]
    freqs_w = welch(_sample, fs=SR, nperseg=EPOCH_SAMPLES)[0]

    for s_idx in range(n_subjects):
        sub_id = ts['subject_ids'][s_idx]

        for (cond, arr_key,
             d_arr, a_arr, z_arr) in [
            ('DMT', 'dmt', delta_dmt, alpha_dmt, lzc_dmt),
            ('PCB', 'pcb', delta_pcb, alpha_pcb, lzc_pcb),
        ]:
            data_all    = ts[arr_key][s_idx]         # (n_epochs, 68, 500)
            t_stamps    = stamps[arr_key][s_idx]     # (n_epochs,) in seconds
            n_epochs_s  = data_all.shape[0]

            # Per-epoch metrics: shape (n_epochs, n_parcels)
            ep_delta = np.zeros((n_epochs_s, n_parcels))
            ep_alpha = np.zeros((n_epochs_s, n_parcels))
            ep_lzc   = np.zeros((n_epochs_s, n_parcels))

            for ep_idx in range(n_epochs_s):
                epoch_data = data_all[ep_idx]    # (68, 500)

                # Welch PSD per parcel
                psd_ep = np.zeros((n_parcels, len(freqs_w)))
                for p_idx in range(n_parcels):
                    _, pxx = welch(epoch_data[p_idx], fs=SR,
                                   nperseg=EPOCH_SAMPLES)
                    psd_ep[p_idx] = pxx

                ep_delta[ep_idx] = band_power_db(psd_ep, freqs_w, *DELTA_BAND)
                ep_alpha[ep_idx] = band_power_db(psd_ep, freqs_w, *ALPHA_BAND)

                # LZc per parcel
                for p_idx in range(n_parcels):
                    sig = epoch_data[p_idx]
                    ep_lzc[ep_idx, p_idx] = _lz76(
                        (sig > sig.mean()).tolist()
                    )

            # Map to shared time axis
            d_arr[s_idx] = align_metric_to_time(t_stamps, ep_delta, tidx)
            a_arr[s_idx] = align_metric_to_time(t_stamps, ep_alpha, tidx)
            z_arr[s_idx] = align_metric_to_time(t_stamps, ep_lzc,   tidx)

        if (s_idx + 1) % 3 == 0 or (s_idx + 1) == n_subjects:
            print(f"    [{s_idx + 1}/{n_subjects}] done")

    # -------------------------------------------------------------------
    # 3c. Aggregate parcels → networks
    #     Shapes: (n_subjects, n_times, n_networks)
    # -------------------------------------------------------------------
    print("\n  Aggregating into networks ...")
    n_nets = len(network_order)

    delta_dmt_net = aggregate_to_network(delta_dmt, network_map, network_order)
    delta_pcb_net = aggregate_to_network(delta_pcb, network_map, network_order)
    alpha_dmt_net = aggregate_to_network(alpha_dmt, network_map, network_order)
    alpha_pcb_net = aggregate_to_network(alpha_pcb, network_map, network_order)
    lzc_dmt_net   = aggregate_to_network(lzc_dmt,   network_map, network_order)
    lzc_pcb_net   = aggregate_to_network(lzc_pcb,   network_map, network_order)

    # -------------------------------------------------------------------
    # 3d. Baseline correction
    # -------------------------------------------------------------------
    print("  Baseline correcting ...")
    delta_dmt_bc = baseline_correct_tc(delta_dmt_net, tbase)
    delta_pcb_bc = baseline_correct_tc(delta_pcb_net, tbase)
    alpha_dmt_bc = baseline_correct_tc(alpha_dmt_net, tbase)
    alpha_pcb_bc = baseline_correct_tc(alpha_pcb_net, tbase)
    lzc_dmt_bc   = baseline_correct_tc(lzc_dmt_net,   tbase)
    lzc_pcb_bc   = baseline_correct_tc(lzc_pcb_net,   tbase)

    # -------------------------------------------------------------------
    # 3e. Cluster permutation tests on post-bolus window, per network
    # -------------------------------------------------------------------
    print("\n  Running cluster permutation tests ...")

    # Restrict to bolus-onset → analysis-end time window for the test
    post_slice = slice(tbase, tend + 1)

    sig_delta = np.zeros((n_nets, n_times), dtype=bool)
    sig_alpha = np.zeros((n_nets, n_times), dtype=bool)
    sig_lzc   = np.zeros((n_nets, n_times), dtype=bool)

    cluster_results = {}   # for the report

    for k, net in enumerate(network_order):
        short = NETWORK_SHORT.get(net, net)
        for metric_name, dmt_bc, pcb_bc, sig_arr in [
            ('delta', delta_dmt_bc, delta_pcb_bc, sig_delta),
            ('alpha', alpha_dmt_bc, alpha_pcb_bc, sig_alpha),
            ('lzc',   lzc_dmt_bc,   lzc_pcb_bc,   sig_lzc),
        ]:
            diff_post = (dmt_bc[:, post_slice, k]
                         - pcb_bc[:, post_slice, k])   # (n_sub, n_post_times)

            # Drop subjects with all-NaN in post-bolus window
            valid = ~np.all(np.isnan(diff_post), axis=1)
            if valid.sum() < 3:
                print(f"    {short} {metric_name}: <3 valid subjects, skip")
                cluster_results[f'{short}_{metric_name}'] = []
                continue

            diff_clean = diff_post[valid]
            # Replace remaining NaNs with 0 for the permutation test
            diff_clean = np.where(np.isnan(diff_clean), 0.0, diff_clean)

            mask_post, _, pvals = permutation_cluster_test_1d(
                diff_clean, n_perms=N_PERMS
            )
            # Expand back to full time axis
            sig_arr[k, post_slice] = mask_post
            cluster_results[f'{short}_{metric_name}'] = pvals

            n_sig = int(mask_post.sum())
            print(f"    {short} {metric_name}: {n_sig} significant time-points "
                  f"({len(pvals)} clusters, "
                  f"min_p={min(pvals):.3f if pvals else 'N/A'})")

    # -------------------------------------------------------------------
    # 3f. Plots
    # -------------------------------------------------------------------
    print("\n  Plotting 6x3 timecourse grid ...")
    _plot_timecourse_grid(
        all_times, tbase,
        delta_dmt_bc, delta_pcb_bc,
        alpha_dmt_bc, alpha_pcb_bc,
        lzc_dmt_bc,   lzc_pcb_bc,
        sig_delta, sig_alpha, sig_lzc,
        network_order,
        fname='source_fig4D_network_timecourses.png'
    )

    print("  Plotting summary heatmaps ...")
    _plot_summary_heatmap(
        all_times, tbase, tend,
        delta_dmt_bc - delta_pcb_bc,
        alpha_dmt_bc - alpha_pcb_bc,
        lzc_dmt_bc   - lzc_pcb_bc,
        sig_delta, sig_alpha, sig_lzc,
        network_order,
        fname='source_fig4D_summary_heatmap.png'
    )

    print("  Section 3 complete.")

    return {
        'all_times'     : all_times,
        'tbase'         : tbase,
        'tend'          : tend,
        'delta_dmt_bc'  : delta_dmt_bc,
        'delta_pcb_bc'  : delta_pcb_bc,
        'alpha_dmt_bc'  : alpha_dmt_bc,
        'alpha_pcb_bc'  : alpha_pcb_bc,
        'lzc_dmt_bc'    : lzc_dmt_bc,
        'lzc_pcb_bc'    : lzc_pcb_bc,
        'sig_delta'     : sig_delta,
        'sig_alpha'     : sig_alpha,
        'sig_lzc'       : sig_lzc,
        'cluster_results': cluster_results,
    }


# ---------------------------------------------------------------------------
# Section 3 plotting helpers
# ---------------------------------------------------------------------------

def _plot_timecourse_grid(all_times, tbase,
                          delta_dmt, delta_pcb,
                          alpha_dmt, alpha_pcb,
                          lzc_dmt,   lzc_pcb,
                          sig_delta, sig_alpha, sig_lzc,
                          network_order,
                          fname):
    """
    7x3 panel grid: rows = networks, columns = delta / alpha / LZc.

    Each panel shows group-mean ± SD for DMT (crimson) and PCB (steelblue),
    Gaussian-smoothed (sigma=1 time-point), with grey shading where the
    cluster test is significant (p < 0.05).

    Parameters
    ----------
    all_times : ndarray (n_times,)
    tbase     : int  — index of last pre-bolus time point
    delta_dmt, delta_pcb, ... : ndarray (n_subjects, n_times, n_networks)
    sig_delta, sig_alpha, sig_lzc : ndarray bool (n_networks, n_times)
    network_order : list of str
    fname : str
    """
    n_nets   = len(network_order)
    t_arr    = np.array(all_times)
    metrics  = [
        ('Delta power (dB)',  delta_dmt, delta_pcb, sig_delta),
        ('Alpha power (dB)',  alpha_dmt, alpha_pcb, sig_alpha),
        ('LZc',               lzc_dmt,   lzc_pcb,   sig_lzc),
    ]

    fig, axes = plt.subplots(n_nets, 3, figsize=(15, 3 * n_nets),
                             sharex=True)
    fig.suptitle(
        'Source-space timecourses per network (DMT vs PCB, baseline corrected)',
        fontsize=13, fontweight='bold'
    )

    for row, net in enumerate(network_order):
        net_col = NETWORK_COLOURS.get(net, 'black')
        short   = NETWORK_SHORT.get(net, net)

        for col, (metric_label, dmt_tc, pcb_tc, sig_arr) in enumerate(metrics):
            ax = axes[row, col]

            # Network timecourse for all subjects: (n_sub, n_times)
            d = dmt_tc[:, :, row]
            p = pcb_tc[:, :, row]

            d_mn = np.nanmean(d, axis=0)
            d_sd = np.nanstd(d, axis=0)
            p_mn = np.nanmean(p, axis=0)
            p_sd = np.nanstd(p, axis=0)

            # Smooth
            d_mn_s = gaussian_filter1d(np.nan_to_num(d_mn), sigma=1)
            p_mn_s = gaussian_filter1d(np.nan_to_num(p_mn), sigma=1)

            ax.plot(t_arr, d_mn_s, color='crimson',   linewidth=1.5, label='DMT')
            ax.fill_between(t_arr,
                            gaussian_filter1d(np.nan_to_num(d_mn - d_sd), 1),
                            gaussian_filter1d(np.nan_to_num(d_mn + d_sd), 1),
                            color='crimson', alpha=0.15)
            ax.plot(t_arr, p_mn_s, color='steelblue', linewidth=1.5, label='PCB')
            ax.fill_between(t_arr,
                            gaussian_filter1d(np.nan_to_num(p_mn - p_sd), 1),
                            gaussian_filter1d(np.nan_to_num(p_mn + p_sd), 1),
                            color='steelblue', alpha=0.15)

            # Significance shading
            sig = sig_arr[row]
            if sig.any():
                ylims = ax.get_ylim()
                ax.fill_between(t_arr, ylims[0], ylims[1],
                                where=sig, color='grey', alpha=0.25,
                                label='p<0.05')

            # Bolus onset marker
            ax.axvline(x=BOLUS_ONSET_MIN, color='black',
                       linestyle='--', linewidth=0.8)

            ax.grid(True, linestyle=':', alpha=0.35)

            # Row label (left column only)
            if col == 0:
                ax.set_ylabel(short, fontsize=9, fontweight='bold',
                              color=net_col, rotation=0, labelpad=30,
                              va='center')
            # Column header (top row only)
            if row == 0:
                ax.set_title(metric_label, fontsize=10, fontweight='bold')
                ax.legend(fontsize=7, loc='upper left')

            # x-label (bottom row only)
            if row == n_nets - 1:
                ax.set_xlabel('Time (min)', fontsize=9)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


def _plot_summary_heatmap(all_times, tbase, tend,
                          diff_delta, diff_alpha, diff_lzc,
                          sig_delta, sig_alpha, sig_lzc,
                          network_order, fname):
    """
    Three side-by-side heatmaps (delta, alpha, LZc): rows = networks,
    columns = time.  Colour = group-mean DMT-PCB difference.  Significant
    time points are marked with an asterisk contour.

    Parameters
    ----------
    all_times : ndarray (n_times,)
    tbase, tend : int
    diff_delta, diff_alpha, diff_lzc : ndarray (n_subjects, n_times, n_networks)
    sig_delta, sig_alpha, sig_lzc : ndarray bool (n_networks, n_times)
    network_order : list of str
    fname : str
    """
    t_arr  = np.array(all_times)
    shorts = [NETWORK_SHORT.get(n, n[:3]) for n in network_order]

    metrics = [
        ('Delta power (dB)',  diff_delta, sig_delta),
        ('Alpha power (dB)',  diff_alpha, sig_alpha),
        ('LZc',               diff_lzc,   sig_lzc),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4),
                              gridspec_kw=dict(wspace=0.35))
    fig.suptitle(
        'Source-space DMT - PCB difference over time (group mean)\n'
        '* = significant cluster (p < 0.05)',
        fontsize=12, fontweight='bold'
    )

    for ax, (metric_label, diff_arr, sig_arr) in zip(axes, metrics):
        # Group mean: (n_times, n_networks) → transpose → (n_networks, n_times)
        gm = np.nanmean(diff_arr, axis=0).T   # (n_networks, n_times)

        vmax = np.nanpercentile(np.abs(gm), 95)
        vmin = -vmax
        im = ax.imshow(gm, aspect='auto', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, interpolation='nearest',
                       extent=[t_arr[0], t_arr[-1],
                               len(network_order) - 0.5, -0.5])

        # Mark significant time points per network
        for k in range(len(network_order)):
            sig_t = t_arr[sig_arr[k]]
            if len(sig_t):
                ax.scatter(sig_t,
                           np.full(len(sig_t), k),
                           marker='*', color='black', s=20, zorder=5,
                           linewidths=0)

        # Bolus line
        ax.axvline(x=BOLUS_ONSET_MIN, color='black',
                   linestyle='--', linewidth=1.0)

        ax.set_yticks(range(len(network_order)))
        ax.set_yticklabels(shorts, fontsize=9)
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_title(metric_label, fontsize=10, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out}")


# ===========================================================================
# 5. Summary report
# ===========================================================================

def write_report(s1, s2, s3, network_order, subject_ids):
    """
    Write a plain-text summary report to Results/sourcespace_effects_report.txt.

    Parameters
    ----------
    s1 : dict  — Section 1 results
    s2 : dict  — Section 2 results
    s3 : dict  — Section 3 results
    network_order : list of str
    subject_ids   : list of str
    """
    from scipy.stats import ttest_1samp

    out_path = os.path.join(RESULTS_DIR, 'sourcespace_effects_report.txt')
    print(f"\n  Writing report to {out_path} ...")

    lines = []
    lines.append("=" * 70)
    lines.append("SOURCE-SPACE EFFECTS REPORT")
    lines.append("Author: Kiret Dhindsa (kiretd@gmail.com)")
    lines.append("=" * 70)
    lines.append(f"Subjects analysed ({len(subject_ids)}): {', '.join(subject_ids)}")
    lines.append(f"Post-bolus epochs only (>{BOLUS_SEC/60:.0f} min).")
    lines.append(f"N permutations for cluster tests: {N_PERMS}")
    lines.append("")

    # -------------------------------------------------------------------
    # Section 1 summary
    # -------------------------------------------------------------------
    lines.append("-" * 70)
    lines.append("SECTION 1: Source-space spectral power (Welch PSD)")
    lines.append("-" * 70)
    lines.append("DMT - PCB band power difference (dB), mean +/- SEM across subjects")
    lines.append(f"{'Network':<28} {'Delta':<18} {'Alpha':<18} {'Beta':<18}")
    lines.append("-" * 82)

    for k, net in enumerate(network_order):
        short = NETWORK_SHORT.get(net, net)
        row_parts = [f"{short:<28}"]
        for b_idx, band_label in enumerate(s1['band_labels']):
            vals  = s1['band_diff_nets'][:, k, b_idx]
            clean = vals[~np.isnan(vals)]
            if len(clean) < 2:
                row_parts.append(f"{'N/A':<18}")
                continue
            m    = clean.mean()
            sem  = clean.std() / np.sqrt(len(clean))
            _, p = ttest_1samp(clean, 0) if len(clean) >= 3 else (None, 1.0)
            star = _significance_stars(p)
            row_parts.append(f"{m:+.3f} +/- {sem:.3f}{star:<6}")
        lines.append("".join(row_parts))

    lines.append("")
    lines.append("IRASA oscillatory power change (log10 ratio), mean +/- SEM")
    lines.append(f"{'Network':<28} {'Delta':<18} {'Alpha':<18} {'Beta':<18}")
    lines.append("-" * 82)

    for k, net in enumerate(network_order):
        short = NETWORK_SHORT.get(net, net)
        row_parts = [f"{short:<28}"]
        for b_idx in range(len(s1['band_labels'])):
            vals  = s1['irasa_band_diff_nets'][:, k, b_idx]
            clean = vals[~np.isnan(vals)]
            if len(clean) < 2:
                row_parts.append(f"{'N/A':<18}")
                continue
            m    = clean.mean()
            sem  = clean.std() / np.sqrt(len(clean))
            _, p = ttest_1samp(clean, 0) if len(clean) >= 3 else (None, 1.0)
            star = _significance_stars(p)
            row_parts.append(f"{m:+.4f}+/-{sem:.4f}{star:<5}")
        lines.append("".join(row_parts))

    lines.append("")

    # -------------------------------------------------------------------
    # Section 2 summary
    # -------------------------------------------------------------------
    lines.append("-" * 70)
    lines.append("SECTION 2: Source-space LZc")
    lines.append("-" * 70)
    lines.append("DMT - PCB mean LZc difference, mean +/- SEM across subjects")
    lines.append(f"{'Network':<28} {'Mean diff':<14} {'SEM':<10} {'p (1-samp t)':<16}")
    lines.append("-" * 68)

    for k, net in enumerate(network_order):
        short = NETWORK_SHORT.get(net, net)
        vals  = s2['lzc_diff_nets'][:, k]
        clean = vals[~np.isnan(vals)]
        if len(clean) < 2:
            lines.append(f"{short:<28} {'N/A'}")
            continue
        m    = clean.mean()
        sem  = clean.std() / np.sqrt(len(clean))
        _, p = ttest_1samp(clean, 0) if len(clean) >= 3 else (None, 1.0)
        star = _significance_stars(p)
        lines.append(f"{short:<28} {m:+.4f}       {sem:.4f}     {p:.4f}  {star}")

    lines.append("")

    # -------------------------------------------------------------------
    # Section 3 summary
    # -------------------------------------------------------------------
    lines.append("-" * 70)
    lines.append("SECTION 3: Source-space timecourses (cluster-corrected)")
    lines.append("-" * 70)
    lines.append("Cluster p-values from permutation test on post-bolus window")
    lines.append(f"Analysis window: {BOLUS_ONSET_MIN:.1f} - {ANALYSIS_END_MIN:.1f} min")
    lines.append("")

    for net in network_order:
        short = NETWORK_SHORT.get(net, net)
        lines.append(f"  {short}:")
        for metric in ['delta', 'alpha', 'lzc']:
            key = f'{short}_{metric}'
            pvals = s3['cluster_results'].get(key, [])
            if not pvals:
                lines.append(f"    {metric:<8}: no significant clusters (or test failed)")
            else:
                sig_clusters = [p for p in pvals if p < 0.05]
                all_p_str = ', '.join(f'{p:.3f}' for p in sorted(pvals))
                lines.append(
                    f"    {metric:<8}: {len(sig_clusters)} sig. cluster(s) "
                    f"(p={all_p_str})"
                )
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"    Report saved: {out_path}")


# ===========================================================================
# 6. Main
# ===========================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("sourcespace_effects.py")
    print("Mapping DMT effects into source space")
    print("=" * 60)

    # --- Load data ---
    ts, stamps, labels, parcel_names = load_data()

    # --- Network assignment ---
    print("\nAssigning parcels to networks ...")
    network_map, network_order, parcel_to_network = assign_networks(parcel_names)

    # --- Section 1: Spectral analysis ---
    s1 = section1_spectral(ts, stamps, labels, parcel_names,
                            network_map, network_order, parcel_to_network)

    # --- Section 2: LZc analysis ---
    s2 = section2_lzc(ts, labels, parcel_names,
                      network_map, network_order, parcel_to_network)

    # --- Section 3: Timecourses ---
    s3 = section3_timecourses(ts, stamps, labels, parcel_names,
                               network_map, network_order)

    # --- Summary report ---
    write_report(s1, s2, s3, network_order, ts['subject_ids'])

    print("\n" + "=" * 60)
    print("Done. All outputs saved to Results/")
    print("=" * 60)
