"""
sourcespace_analysis.py
=======================
Source-space EEG analysis for the DMT dataset.

Performs source localisation using MNE-Python with the standard fsaverage
template head model, projects activity into the Schaefer 2018 cortical
atlas (100 parcels, 7 Yeo networks), and computes dynamic functional
connectivity (FC) matrices.

Overview
--------
1. Discover valid subject folders (no data loaded at this stage).
2. Build a forward model using the fsaverage template MRI and a standard
   10-20 electrode montage.
3. For each subject, load one session at a time (DMT then PCB):
   a. Compute a minimum-norm (sLORETA) inverse operator.
   b. Apply the inverse operator to every 2-second epoch, producing source
      time-series on the fsaverage surface.
   c. Parcellate source activity into Schaefer 2018 atlas parcels using
      mean_flip mode to avoid polarity cancellation.
   d. Optionally aggregate parcels into major brain networks.
   e. Optionally compute a pairwise FC matrix per epoch.
   f. Save the subject's results to a temp file, then free all large arrays.
4. After all subjects are processed, assemble the temp files into the final
   output pkl files (same format as before) and clean up.

Memory-efficient design
-----------------------
Only one subject's raw EEG session is resident in memory at a time.  Large
intermediate arrays (SourceEstimate list, raw session dict) are explicitly
deleted and gc.collect() is called between subjects, preventing cumulative
RAM growth that could crash the machine when processing all 14 subjects.

Configuration
-------------
All user-adjustable parameters are defined in the "Configuration" block
below. The most important ones are:

    USE_NETWORKS    : bool
        If True, average parcels into the 7 Yeo networks defined in
        SCHAEFER_NETWORK_NAMES instead of returning all 100 individual parcels.

    SAVE_FC         : bool
        If True, compute and save dynamic FC matrices for every epoch.

    CONNECTIVITY_METRIC : str
        'correlation'  – Pearson correlation of parcel timeseries.
        'coherence'    – Mean magnitude-squared coherence across frequencies.
        'wpli'         – Weighted Phase Lag Index (Vinck et al. 2011);
                         insensitive to zero-lag/volume-conduction artefacts.

Outputs (saved to Results/saved_outputs/)
-----------------------------------------
    source_timeseries.pkl
        Dict with keys 'dmt' and 'pcb'. Each value is a list (one entry per
        subject) of ndarray, shape (n_epochs, n_parcels, n_times).

    timestamps.pkl
        Dict with keys 'dmt' and 'pcb'. Each value is a list (one per
        subject) of ndarray, shape (n_epochs,), giving the start time of
        each epoch in seconds from session start.

    epoch_labels.pkl
        Dict with keys 'dmt' and 'pcb'. Each value is a list (one per
        subject) of ndarray, shape (n_epochs,), with string labels
        'baseline' or 'post_bolus'.

    fc_matrices.pkl  (only if SAVE_FC = True)
        Dict with keys 'dmt' and 'pcb'. Each value is a list (one per
        subject) of ndarray, shape (n_epochs, n_parcels, n_parcels).

    parcel_names.txt
        Plain-text list of parcel / network names in the order they appear
        as rows/columns of the FC matrices.

Author: Kiret Dhindsa (kiretd@gmail.com)
"""

import gc
import os
import pickle
import shutil
import sys
import warnings

# Ensure stdout can handle Unicode on Windows (CP1252 terminal)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===========================================================================
# Configuration
# ===========================================================================

EEG_FOLDER        = "EEG"                                    # root folder containing per-subject subfolders
RESULTS_DIR       = "Results"                               # output directory (figures, plots)
SAVED_OUTPUTS_DIR = os.path.join(RESULTS_DIR, "saved_outputs")  # pkl + txt data outputs

# Subjects to exclude (per README and Timmermann et al.)
REMOVE_LIST  = ['01', '07', '11', '16', '19', '25']

SR           = 250         # sampling rate (Hz)
BOLUS_SEC    = 480.0       # bolus onset in seconds (8 min)

# Source-space options
USE_NETWORKS = False
"""
If True, parcel timeseries are averaged into the 7 Yeo networks defined in
SCHAEFER_NETWORK_NAMES. If False, individual Schaefer 2018 parcels
(100 parcels across both hemispheres) are used.

NOTE: Use False for validation — the validate_sourcespace.py suite checks
100-region parcel names and coverage. Network-level output (7 regions) will
cause confusing results in checks 3a, 3b, and 5b.
"""

SAVE_FC      = True
"""
If True, compute and save a dynamic FC matrix for every epoch.
This significantly increases computation time and file size.

NOTE on FC stability: 2-second epochs (500 samples at 250 Hz) produce noisy
correlation estimates, especially for 100×100 matrices. If the off-diagonal FC
distribution looks nearly uniform across −1 to 1, consider either using
longer epochs or switching to a regularised measure (e.g. coherence).
"""

CONNECTIVITY_METRIC = 'correlation'
"""
Functional connectivity metric to use for each epoch.
Options:
    'correlation'  – fast Pearson correlation matrix (n_parcels × n_parcels)
    'coherence'    – mean magnitude-squared coherence across 1–45 Hz
    'wpli'         – weighted Phase Lag Index (Vinck et al. 2011);
                     insensitive to volume conduction / zero-lag artefacts;
                     values in [0, 1], diagonal = 0.0
"""

# Inverse operator regularisation
SNIR_INV   = 3.0           # SNR assumed for inverse solution (lambda2 = 1/SNR²)
INV_METHOD = 'sLORETA'     # 'MNE', 'dSPM', or 'sLORETA'

# Schaefer 2018 atlas parcellation string (100 parcels, 7 Yeo networks)
SCHAEFER_PARC = 'Schaefer2018_100Parcels_7Networks_order'

# Mapping from Schaefer network key (field 3 of label name) to display name.
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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVED_OUTPUTS_DIR, exist_ok=True)

# ===========================================================================
# 1. Data Loading
# ===========================================================================

def discover_subjects(eeg_folder, remove_list):
    """
    Discover valid subject IDs and their session file paths without loading
    any data into memory.

    Parameters
    ----------
    eeg_folder : str
        Root directory containing one subdirectory per subject.
    remove_list : list of str
        Two-digit subject IDs to exclude.

    Returns
    -------
    subject_list : list of (str, str, str)
        Each entry is (subject_id, dmt_path, pcb_path) for every included
        subject with both session files present.
    """
    subject_list = []

    for folder in sorted(os.listdir(eeg_folder)):
        subject_path = os.path.join(eeg_folder, folder)
        if not os.path.isdir(subject_path):
            continue

        sid = folder[-2:]
        if sid in remove_list:
            continue

        dmt_path = os.path.join(subject_path, "ses_DMT", "dataref.mat")
        pcb_path = os.path.join(subject_path, "ses_PCB", "dataref.mat")

        if not (os.path.exists(dmt_path) and os.path.exists(pcb_path)):
            print(f"  [SKIP] sub_{sid}: missing file(s)")
            continue

        subject_list.append((sid, dmt_path, pcb_path))
        print(f"  [FOUND] sub_{sid}")

    print(f"\nFound {len(subject_list)} subjects.")
    return subject_list


def load_single_session(mat_path):
    """
    Load one FieldTrip session from a .mat file.

    Parameters
    ----------
    mat_path : str
        Full path to a dataref.mat file.

    Returns
    -------
    session : dict
        The 'dataref' structure from the .mat file.
    """
    return scipy.io.loadmat(mat_path, simplify_cells=True)['dataref']


# ===========================================================================
# 2. MNE Setup (Info + Epochs)
# ===========================================================================

def create_mne_info(channel_labels, sfreq=250):
    """
    Build an MNE Info object with a standard 10-20 montage.

    Parameters
    ----------
    channel_labels : array-like of str
    sfreq : float

    Returns
    -------
    info : mne.Info
    """
    labels   = [
        lbl.strip() if isinstance(lbl, str) else lbl[0].strip()
        for lbl in channel_labels
    ]
    misc_ch  = {'EOG', 'ECG1', 'ECG2'}
    ch_types = ['misc' if lbl in misc_ch else 'eeg' for lbl in labels]

    info         = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=ch_types)
    full_montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos       = {
        ch: pos
        for ch, pos in full_montage.get_positions()['ch_pos'].items()
        if ch in labels
    }
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage, match_case=False)
    return info


def build_epoch_labels(session, bolus_sec=BOLUS_SEC):
    """
    Generate string labels and timestamps for each epoch in a session.

    Parameters
    ----------
    session : dict
        FieldTrip 'dataref' structure.
    bolus_sec : float
        Bolus onset time in seconds.

    Returns
    -------
    timestamps : ndarray, shape (n_epochs,)
        Start time of each epoch in seconds from the start of the session.
    labels : ndarray of str, shape (n_epochs,)
        'baseline' for epochs before bolus_sec, 'post_bolus' after.
    """
    times      = np.array([t[0] for t in session['time']])   # seconds
    labels     = np.where(times < bolus_sec, 'baseline', 'post_bolus')
    return times, labels


def create_epochs(session, info, tmin=0.0):
    """
    Create an MNE EpochsArray from a FieldTrip session dict.

    Returns
    -------
    epochs : mne.EpochsArray
    timestamps : ndarray
    labels : ndarray of str
    """
    data        = np.stack(session['trial'], axis=0)
    timestamps, str_labels = build_epoch_labels(session)

    int_labels  = (str_labels == 'post_bolus').astype(int)
    events      = np.column_stack([np.arange(len(str_labels)),
                                   np.zeros(len(str_labels), int),
                                   int_labels])
    epochs = mne.EpochsArray(
        data, info,
        events=events,
        event_id={'baseline': 0, 'post_bolus': 1},
        tmin=tmin,
        verbose=False,
    )
    return epochs, timestamps, str_labels


# ===========================================================================
# 3. Source Localisation (fsaverage template)
# ===========================================================================

def build_forward_model(info, subjects_dir=None):
    """
    Build a forward model using the MNE fsaverage template.

    Uses a boundary element model (BEM) with EEG electrodes mapped to the
    fsaverage surface via the standard 10-20 montage.

    Parameters
    ----------
    info : mne.Info
        Must include channel positions (set_montage already called).
    subjects_dir : str or None
        Path to the FreeSurfer subjects directory. If None, uses MNE's
        built-in fsaverage data.

    Returns
    -------
    fwd : mne.Forward
    src : mne.SourceSpaces
    subjects_dir : str
    """
    # Fetch fsaverage data if not already present
    fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)

    subject = 'fsaverage'

    # Source space (ico-4 → ~2562 dipoles per hemisphere)
    src = mne.setup_source_space(
        subject, spacing='ico4',
        subjects_dir=subjects_dir,
        add_dist=False, verbose=False,
    )

    # BEM solution (pre-computed for fsaverage in MNE)
    bem_fname = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    bem       = mne.read_bem_solution(bem_fname, verbose=False)

    # Transformation: sensor → MRI space (identity for fsaverage standard)
    trans = 'fsaverage'

    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem,
        eeg=True, meg=False,
        mindist=5.0, verbose=False,
    )
    print(f"  Forward model: {fwd['nchan']} channels, "
          f"{fwd['nsource']} sources")
    return fwd, src, subjects_dir


def compute_inverse_operator(epochs, fwd):
    """
    Compute a noise-normalised inverse operator from sample covariance.

    Parameters
    ----------
    epochs : mne.EpochsArray
    fwd : mne.Forward

    Returns
    -------
    inv : mne.minimum_norm.InverseOperator
    """
    # Estimate noise covariance from baseline epochs
    baseline_epochs = epochs['baseline']
    if len(baseline_epochs) == 0:
        # Fall back to using all epochs if no explicit baseline label
        noise_cov = mne.compute_covariance(epochs, verbose=False)
    else:
        noise_cov = mne.compute_covariance(baseline_epochs, verbose=False)

    inv = make_inverse_operator(
        epochs.info, fwd, noise_cov,
        loose=0.2, depth=0.8, verbose=False,
    )
    return inv


# ===========================================================================
# 4. Parcellation
# ===========================================================================

def fetch_schaefer_labels(subjects_dir, subject='fsaverage'):
    """
    Return the list of Schaefer 2018 (100 parcels, 7 Yeo networks) label
    objects and their source indices on the fsaverage surface.

    The annotation files are expected to be pre-installed at:
        <subjects_dir>/<subject>/label/
            lh.Schaefer2018_100Parcels_7Networks_order.annot
            rh.Schaefer2018_100Parcels_7Networks_order.annot

    Parameters
    ----------
    subjects_dir : str
    subject : str

    Returns
    -------
    labels : list of mne.Label
        All 100 cortical Schaefer parcels (left + right hemisphere).
    """
    labels = mne.read_labels_from_annot(
        subject, parc=SCHAEFER_PARC,
        subjects_dir=subjects_dir,
        verbose=False,
    )
    # Exclude catch-all labels
    labels = [lbl for lbl in labels
              if 'unknown' not in lbl.name.lower()
              and 'background' not in lbl.name.lower()]
    print(f"  Schaefer 2018 atlas: {len(labels)} parcels loaded")
    return labels


def assign_networks(parcel_names):
    """
    Map Schaefer 2018 parcel names to Yeo 7-network display names.

    The network key is always the third underscore-delimited field of the
    stem (before the hemisphere suffix), e.g.:
        '7Networks_LH_Default_PFC_1-lh'  ->  key='Default'

    Parcels not matching a known key are placed into 'Other'.

    Parameters
    ----------
    parcel_names : list of str
        Parcel names as returned by the Schaefer atlas labels.

    Returns
    -------
    network_map : dict
        {network_display_name: [parcel_indices]}
    network_order : list of str
        Ordered list of network display names (for consistent matrix columns).
    """
    network_map = {name: [] for name in SCHAEFER_NETWORK_NAMES.values()}
    network_map['Other'] = []

    for idx, name in enumerate(parcel_names):
        # stem is everything before the '-lh'/'-rh' suffix
        stem = name.split('-')[0]
        key  = stem.split('_')[2]          # e.g. 'Default', 'Vis', ...
        display = SCHAEFER_NETWORK_NAMES.get(key, 'Other')
        network_map[display].append(idx)

    # Remove empty entries (e.g. 'Other' if every parcel matched)
    network_map   = {k: v for k, v in network_map.items() if v}
    network_order = list(network_map.keys())
    return network_map, network_order


def parcellate_stcs(stcs, labels, src):
    """
    Extract mean source activity for each Schaefer 2018 parcel using
    mean_flip mode to avoid polarity cancellation within labels.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate
        One per epoch; each has shape (n_sources, n_times).
    labels : list of mne.Label
    src : mne.SourceSpaces

    Returns
    -------
    parcel_ts : ndarray, shape (n_epochs, n_parcels, n_times)
    parcel_names : list of str
    """
    parcel_names = [lbl.name for lbl in labels]

    # mean_flip flips each dipole to a consistent sign before averaging,
    # eliminating polarity cancellation within labels that causes near-zero
    # signals when dipoles point in opposing directions.
    label_tc = mne.extract_label_time_course(
        stcs, labels, src, mode='mean_flip', verbose=False,
    )  # returns (n_epochs, n_labels, n_times)

    parcel_ts = np.array(label_tc)   # ensure ndarray; already correct shape
    return parcel_ts, parcel_names


def aggregate_to_networks(parcel_ts, network_map, network_order):
    """
    Average parcel timeseries into network-level timeseries.

    Parameters
    ----------
    parcel_ts : ndarray, shape (n_epochs, n_parcels, n_times)
    network_map : dict {network_name: [parcel_indices]}
    network_order : list of str

    Returns
    -------
    network_ts : ndarray, shape (n_epochs, n_networks, n_times)
    """
    n_epochs   = parcel_ts.shape[0]
    n_times    = parcel_ts.shape[2]
    n_nets     = len(network_order)
    network_ts = np.zeros((n_epochs, n_nets, n_times))

    for net_idx, net_name in enumerate(network_order):
        indices = network_map[net_name]
        if indices:
            network_ts[:, net_idx, :] = parcel_ts[:, indices, :].mean(axis=1)

    return network_ts


# ===========================================================================
# 5. Functional Connectivity
# ===========================================================================

def compute_fc_correlation(ts):
    """
    Pearson correlation FC matrix for a single epoch.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_times)

    Returns
    -------
    fc : ndarray, shape (n_regions, n_regions)
    """
    return np.corrcoef(ts)


def compute_fc_coherence(ts, sfreq=SR, fmin=1.0, fmax=45.0):
    """
    Mean magnitude-squared coherence FC matrix for a single epoch.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_times)
    sfreq : float
    fmin, fmax : float
        Frequency range to average over.

    Returns
    -------
    fc : ndarray, shape (n_regions, n_regions)
    """
    from scipy.signal import coherence as sp_coherence

    n_regions = ts.shape[0]
    fc        = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(i, n_regions):
            f, coh = sp_coherence(ts[i], ts[j], fs=sfreq, nperseg=min(128, ts.shape[1]))
            mask   = (f >= fmin) & (f <= fmax)
            val    = coh[mask].mean() if mask.any() else 0.0
            fc[i, j] = val
            fc[j, i] = val

    np.fill_diagonal(fc, 1.0)
    return fc


def compute_fc_wpli(ts, sfreq=SR, fmin=1.0, fmax=45.0):
    """
    Weighted Phase Lag Index (wPLI) FC matrix for a single epoch.

    Implements the wPLI estimator from Vinck et al. (2011, NeuroImage).
    Uses STFT to obtain per-segment cross-spectra, then computes:

        wPLI_f = |E[imag(Sxy_f)]| / E[|imag(Sxy_f)|]

    where E[·] averages across STFT segments, and the result is averaged
    over the frequency band [fmin, fmax].

    Unlike correlation and coherence, wPLI is insensitive to zero-lag
    interactions (volume conduction / common-source artefacts). The
    diagonal is 0.0 (a signal has no phase lag with itself).

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_times)
    sfreq : float
        Sampling frequency in Hz.
    fmin, fmax : float
        Frequency band to average over (Hz).

    Returns
    -------
    fc : ndarray, shape (n_regions, n_regions)
        Symmetric wPLI matrix with values in [0, 1]. Diagonal is 0.0.

    References
    ----------
    Vinck M., Oostenveld R., van Wingerden M., Battaglia F., Pennartz C.M.A.
    (2011) "An improved index of phase-synchronization for electrophysiological
    data in the presence of volume-conduction, noise and sample-size bias."
    NeuroImage, 55(4), 1548-1565.
    """
    from scipy.signal import stft

    n_regions = ts.shape[0]
    n_times   = ts.shape[1]
    nperseg   = min(128, n_times)

    # Pre-compute STFT for every region
    # f shape: (n_freq,); Zxx shape: (n_freq, n_segments)
    f, _, Zxx_all = stft(ts[0], fs=sfreq, nperseg=nperseg)
    n_freq, n_seg = Zxx_all.shape
    stfts = np.empty((n_regions, n_freq, n_seg), dtype=complex)
    stfts[0] = Zxx_all
    for i in range(1, n_regions):
        _, _, Zxx_all = stft(ts[i], fs=sfreq, nperseg=nperseg)
        stfts[i] = Zxx_all

    freq_mask = (f >= fmin) & (f <= fmax)
    if not freq_mask.any():
        return np.zeros((n_regions, n_regions))

    fc = np.zeros((n_regions, n_regions))

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            # Cross-spectrum per segment: shape (n_freq_band, n_segments)
            csd_band = (stfts[i][freq_mask, :] *
                        np.conj(stfts[j][freq_mask, :]))

            imag_csd = np.imag(csd_band)   # (n_freq_band, n_segments)

            # Per frequency bin: |mean_over_segments| / mean_over_segments(|·|)
            numerator   = np.abs(imag_csd.mean(axis=1))   # (n_freq_band,)
            denominator = np.abs(imag_csd).mean(axis=1)   # (n_freq_band,)

            with np.errstate(divide='ignore', invalid='ignore'):
                wpli_per_freq = np.where(denominator > 0,
                                         numerator / denominator, 0.0)

            val = float(wpli_per_freq.mean())
            fc[i, j] = val
            fc[j, i] = val

    # Diagonal is 0.0: a signal has no phase lag with itself
    return fc


def compute_dynamic_fc(parcel_ts, metric='correlation', sfreq=SR):
    """
    Compute a functional connectivity matrix for each epoch.

    Parameters
    ----------
    parcel_ts : ndarray, shape (n_epochs, n_regions, n_times)
    metric : str
        'correlation', 'coherence', or 'wpli'
    sfreq : float

    Returns
    -------
    fc_matrices : ndarray, shape (n_epochs, n_regions, n_regions)
    """
    n_epochs   = parcel_ts.shape[0]
    n_regions  = parcel_ts.shape[1]
    fc_matrices = np.zeros((n_epochs, n_regions, n_regions))

    for ep_idx in range(n_epochs):
        ts = parcel_ts[ep_idx]  # (n_regions, n_times)

        if metric == 'correlation':
            fc_matrices[ep_idx] = compute_fc_correlation(ts)
        elif metric == 'coherence':
            fc_matrices[ep_idx] = compute_fc_coherence(ts, sfreq=sfreq)
        elif metric == 'wpli':
            fc_matrices[ep_idx] = compute_fc_wpli(ts, sfreq=sfreq)
        else:
            raise ValueError(f"Unknown connectivity metric: '{metric}'. "
                             "Choose 'correlation', 'coherence', or 'wpli'.")

    return fc_matrices


# ===========================================================================
# 6. Per-subject pipeline
# ===========================================================================

def process_subject(session, subject_id, fwd, labels, subjects_dir,
                    use_networks=USE_NETWORKS, save_fc=SAVE_FC,
                    metric=CONNECTIVITY_METRIC):
    """
    Run the full source-space pipeline for a single session.

    Steps:
        1. Build MNE epochs.
        2. Compute inverse operator.
        3. Apply inverse to get source estimates.
        4. Parcellate into Desikan-Killany regions (or networks).
        5. Optionally compute dynamic FC matrices.

    Parameters
    ----------
    session : dict
        FieldTrip 'dataref' structure.
    subject_id : str
    fwd : mne.Forward
    labels : list of mne.Label
    subjects_dir : str
    use_networks : bool
    save_fc : bool
    metric : str

    Returns
    -------
    result : dict
        Keys: 'parcel_ts', 'timestamps', 'epoch_labels', 'fc_matrices' (optional)
    """
    info = create_mne_info(session['label'])
    epochs, timestamps, str_labels = create_epochs(session, info)

    # MNE requires an average EEG reference projector for source modelling
    epochs.set_eeg_reference(ref_channels='average', projection=True, verbose=False)
    epochs.apply_proj()

    # Compute inverse operator
    inv = compute_inverse_operator(epochs, fwd)

    # Apply inverse: produce one SourceEstimate per epoch
    lambda2 = 1.0 / SNIR_INV ** 2
    stcs    = apply_inverse_epochs(
        epochs, inv, lambda2=lambda2,
        method=INV_METHOD,
        pick_ori='normal',
        verbose=False,
    )
    print(f"    sub_{subject_id}: {len(stcs)} source estimates computed")

    # Parcellate
    src = fwd['src']
    parcel_ts, parcel_names = parcellate_stcs(stcs, labels, src)
    print(f"    sub_{subject_id}: parcellated -> {parcel_ts.shape}")

    # Optionally aggregate into networks
    if use_networks:
        network_map, network_order = assign_networks(parcel_names)
        parcel_ts    = aggregate_to_networks(parcel_ts, network_map, network_order)
        parcel_names = network_order
        print(f"    sub_{subject_id}: aggregated to {len(network_order)} networks")

    result = {
        'parcel_ts':    parcel_ts,     # (n_epochs, n_regions, n_times)
        'timestamps':   timestamps,    # (n_epochs,)
        'epoch_labels': str_labels,    # (n_epochs,)
        'parcel_names': parcel_names,
    }

    if save_fc:
        print(f"    sub_{subject_id}: computing FC matrices ({metric}) ...")
        fc = compute_dynamic_fc(parcel_ts, metric=metric)
        result['fc_matrices'] = fc     # (n_epochs, n_regions, n_regions)

    return result


# ===========================================================================
# 7. Save outputs
# ===========================================================================

def save_outputs(dmt_results, pcb_results, subject_ids):
    """
    Save all analysis outputs to Results/saved_outputs/.

    Files written:
        saved_outputs/source_timeseries.pkl
        saved_outputs/timestamps.pkl
        saved_outputs/epoch_labels.pkl
        saved_outputs/fc_matrices.pkl  (if SAVE_FC is True)
        saved_outputs/parcel_names.txt
    """
    # Collect per-subject arrays into session-level dicts
    dmt_ts     = [r['parcel_ts']    for r in dmt_results]
    pcb_ts     = [r['parcel_ts']    for r in pcb_results]
    dmt_tst    = [r['timestamps']   for r in dmt_results]
    pcb_tst    = [r['timestamps']   for r in pcb_results]
    dmt_lbl    = [r['epoch_labels'] for r in dmt_results]
    pcb_lbl    = [r['epoch_labels'] for r in pcb_results]

    with open(os.path.join(SAVED_OUTPUTS_DIR, 'source_timeseries.pkl'), 'wb') as f:
        pickle.dump({'dmt': dmt_ts, 'pcb': pcb_ts,
                     'subject_ids': subject_ids}, f)

    with open(os.path.join(SAVED_OUTPUTS_DIR, 'timestamps.pkl'), 'wb') as f:
        pickle.dump({'dmt': dmt_tst, 'pcb': pcb_tst,
                     'subject_ids': subject_ids}, f)

    with open(os.path.join(SAVED_OUTPUTS_DIR, 'epoch_labels.pkl'), 'wb') as f:
        pickle.dump({'dmt': dmt_lbl, 'pcb': pcb_lbl,
                     'subject_ids': subject_ids}, f)

    if SAVE_FC:
        dmt_fc = [r['fc_matrices'] for r in dmt_results]
        pcb_fc = [r['fc_matrices'] for r in pcb_results]
        with open(os.path.join(SAVED_OUTPUTS_DIR, 'fc_matrices.pkl'), 'wb') as f:
            pickle.dump({'dmt': dmt_fc, 'pcb': pcb_fc,
                         'subject_ids': subject_ids}, f)
        print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'fc_matrices.pkl')}")

    # Parcel names (use first subject's list)
    parcel_names = dmt_results[0]['parcel_names']
    pnames_path  = os.path.join(SAVED_OUTPUTS_DIR, 'parcel_names.txt')
    with open(pnames_path, 'w') as f:
        f.write('\n'.join(parcel_names))

    print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'source_timeseries.pkl')}")
    print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'timestamps.pkl')}")
    print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'epoch_labels.pkl')}")
    print(f"  Saved: {pnames_path}  ({len(parcel_names)} regions)")


# ===========================================================================
# 8. Incremental save / assemble helpers
# ===========================================================================

TEMP_DIR = os.path.join(RESULTS_DIR, '_temp')


def save_subject_temp(sid, dmt_result, pcb_result):
    """
    Pickle one subject's DMT and PCB results to a temporary file.

    The file is written to Results/_temp/sub_<sid>.pkl and contains a dict
    with keys 'dmt' and 'pcb', each being the result dict returned by
    process_subject().

    Parameters
    ----------
    sid : str
        Two-digit subject ID.
    dmt_result : dict
        Output of process_subject() for the DMT session.
    pcb_result : dict
        Output of process_subject() for the PCB session.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f'sub_{sid}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'dmt': dmt_result, 'pcb': pcb_result}, f)
    print(f"    sub_{sid}: temp results saved -> {path}")


def assemble_final_outputs(subject_ids):
    """
    Load per-subject temp pkl files one at a time and assemble the final
    output pkl files in Results/.

    The output format is identical to the old save_outputs() — lists of
    per-subject arrays — so downstream scripts require no changes.

    Temp files are removed after assembly.

    Parameters
    ----------
    subject_ids : list of str
        Two-digit subject IDs, in order.
    """
    # Containers for the final list-of-arrays structures
    dmt_ts,  pcb_ts  = [], []
    dmt_tst, pcb_tst = [], []
    dmt_lbl, pcb_lbl = [], []
    dmt_fc,  pcb_fc  = [], []
    parcel_names = None

    print("\n  Assembling final output files ...")

    for sid in subject_ids:
        path = os.path.join(TEMP_DIR, f'sub_{sid}.pkl')
        with open(path, 'rb') as f:
            subj = pickle.load(f)

        dmt_r = subj['dmt']
        pcb_r = subj['pcb']

        dmt_ts.append(dmt_r['parcel_ts'])
        pcb_ts.append(pcb_r['parcel_ts'])
        dmt_tst.append(dmt_r['timestamps'])
        pcb_tst.append(pcb_r['timestamps'])
        dmt_lbl.append(dmt_r['epoch_labels'])
        pcb_lbl.append(pcb_r['epoch_labels'])

        if SAVE_FC:
            dmt_fc.append(dmt_r['fc_matrices'])
            pcb_fc.append(pcb_r['fc_matrices'])

        if parcel_names is None:
            parcel_names = dmt_r['parcel_names']

        # Free the loaded dict immediately to keep memory flat
        del subj, dmt_r, pcb_r

    if parcel_names is None:
        raise RuntimeError("No subjects processed — parcel_names is unset.")

    # Write final pkl files
    with open(os.path.join(SAVED_OUTPUTS_DIR, 'source_timeseries.pkl'), 'wb') as f:
        pickle.dump({'dmt': dmt_ts, 'pcb': pcb_ts,
                     'subject_ids': subject_ids}, f)
    print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'source_timeseries.pkl')}")

    with open(os.path.join(SAVED_OUTPUTS_DIR, 'timestamps.pkl'), 'wb') as f:
        pickle.dump({'dmt': dmt_tst, 'pcb': pcb_tst,
                     'subject_ids': subject_ids}, f)
    print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'timestamps.pkl')}")

    with open(os.path.join(SAVED_OUTPUTS_DIR, 'epoch_labels.pkl'), 'wb') as f:
        pickle.dump({'dmt': dmt_lbl, 'pcb': pcb_lbl,
                     'subject_ids': subject_ids}, f)
    print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'epoch_labels.pkl')}")

    if SAVE_FC:
        with open(os.path.join(SAVED_OUTPUTS_DIR, 'fc_matrices.pkl'), 'wb') as f:
            pickle.dump({'dmt': dmt_fc, 'pcb': pcb_fc,
                         'subject_ids': subject_ids}, f)
        print(f"  Saved: {os.path.join(SAVED_OUTPUTS_DIR, 'fc_matrices.pkl')}")

    pnames_path = os.path.join(SAVED_OUTPUTS_DIR, 'parcel_names.txt')
    with open(pnames_path, 'w') as f:
        f.write('\n'.join(parcel_names))
    print(f"  Saved: {pnames_path}  ({len(parcel_names)} regions)")

    # Remove temp directory
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("  Temp files cleaned up.")


# ===========================================================================
# 9. Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  DMT EEG — Source-Space Functional Connectivity")
    print("=" * 60)
    print(f"\n  Configuration:")
    print(f"    USE_NETWORKS        = {USE_NETWORKS}")
    print(f"    SAVE_FC             = {SAVE_FC}")
    print(f"    CONNECTIVITY_METRIC = '{CONNECTIVITY_METRIC}'")
    print(f"    INV_METHOD          = '{INV_METHOD}'")

    # ------------------------------------------------------------------
    # [1] Discover subjects (no data loaded yet)
    # ------------------------------------------------------------------
    print("\n[1] Discovering subjects ...")
    subject_list = discover_subjects(EEG_FOLDER, REMOVE_LIST)
    # subject_list : list of (sid, dmt_path, pcb_path)

    if len(subject_list) == 0:
        raise RuntimeError("No subjects found. Check EEG_FOLDER path.")

    subject_ids = [s[0] for s in subject_list]

    # ------------------------------------------------------------------
    # [2] Build shared forward model
    #     Load the first subject's DMT session temporarily just to get
    #     the channel layout, then free it immediately.
    # ------------------------------------------------------------------
    print("\n[2] Building forward model (fsaverage template) ...")
    _first_session = load_single_session(subject_list[0][1])
    ref_info = create_mne_info(_first_session['label'])
    del _first_session          # free immediately — ~hundreds of MB
    gc.collect()

    fwd, src, subjects_dir = build_forward_model(ref_info)

    # ------------------------------------------------------------------
    # [3] Fetch Schaefer 2018 atlas labels
    # ------------------------------------------------------------------
    print("\n[3] Loading Schaefer 2018 atlas ...")
    labels = fetch_schaefer_labels(subjects_dir)

    # ------------------------------------------------------------------
    # [4] Process each subject one at a time
    #     Each session is loaded, processed, saved to a temp file, then
    #     freed before the next session is loaded.
    # ------------------------------------------------------------------
    print("\n[4] Running source-space pipeline per subject ...")

    for i, (sid, dmt_path, pcb_path) in enumerate(subject_list):
        print(f"\n  Subject {i+1}/{len(subject_list)}: sub_{sid}")

        # --- DMT session ---
        print(f"    Loading DMT session ...")
        dmt_session = load_single_session(dmt_path)

        print(f"    Processing DMT session ...")
        dmt_res = process_subject(
            dmt_session, sid, fwd, labels, subjects_dir,
            use_networks=USE_NETWORKS, save_fc=SAVE_FC,
            metric=CONNECTIVITY_METRIC,
        )
        del dmt_session          # raw EEG no longer needed
        gc.collect()

        # --- PCB session ---
        print(f"    Loading PCB session ...")
        pcb_session = load_single_session(pcb_path)

        print(f"    Processing PCB session ...")
        pcb_res = process_subject(
            pcb_session, sid, fwd, labels, subjects_dir,
            use_networks=USE_NETWORKS, save_fc=SAVE_FC,
            metric=CONNECTIVITY_METRIC,
        )
        del pcb_session          # raw EEG no longer needed
        gc.collect()

        # --- Save this subject's results to disk and free them ---
        save_subject_temp(sid, dmt_res, pcb_res)
        del dmt_res, pcb_res
        gc.collect()

    # ------------------------------------------------------------------
    # [5] Assemble final output files from per-subject temp pkls
    # ------------------------------------------------------------------
    print("\n[5] Assembling and saving final outputs ...")
    assemble_final_outputs(subject_ids)

    print("\n" + "=" * 60)
    print("  All done. pkl outputs saved to:", os.path.abspath(SAVED_OUTPUTS_DIR))
    print("=" * 60)
