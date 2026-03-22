"""
sourcespace_plots.py
====================
Visualises functional connectivity (FC) matrices produced by
sourcespace_analysis.py.

For both the region-level (Schaefer 2018, 100 parcels) and network-level outputs,
three 4-panel (2×2) figures are generated:

    Figure 1 — DMT: average FC over the full session, baseline period,
               first 10 minutes post-injection, and remainder.
    Figure 2 — PCB: same windows as Figure 1.
    Figure 3 — DMT minus PCB: difference FC for the same four windows.

Layout
------
Panels are arranged in a 2×2 grid:
    Top-left:     Full session        Top-right:    Pre-injection (baseline)
    Bottom-left:  Post-injection 0–10 min   Bottom-right: Post-injection >10 min

Axis labels (region/network names) are shown on:
    - Y-axis: left column only (top-left and bottom-left panels)
    - X-axis: bottom row only (bottom-left and bottom-right panels)

All region/network names are printed in full — no truncation or skipping.
Colour scale is fixed at -1 to 1 for all panels and all figures.

Expected inputs (Results/ directory)
-------------------------------------
    fc_matrices.pkl      FC matrices for region-level or network-level data.
                         Dict with keys 'dmt', 'pcb', 'subject_ids'.
                         'dmt' and 'pcb' are lists (one per subject) of
                         ndarray, shape (n_epochs, n_regions, n_regions).

    timestamps.pkl       Epoch start times in seconds.
                         Dict with keys 'dmt', 'pcb', 'subject_ids'.
                         Each is a list (one per subject) of ndarray (n_epochs,).

    epoch_labels.pkl     String labels per epoch ('baseline' or 'post_bolus').
                         Same structure as timestamps.pkl.

    parcel_names.txt     One region/network name per line, in matrix row order.

Configuration
-------------
Adjust RESULTS_DIR, BOLUS_SEC, POST1_END_SEC, and COLORMAP below as needed.

Author: Kiret Dhindsa (kiretd@gmail.com)
"""

import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# Configuration
# ===========================================================================

RESULTS_DIR       = "Results"           # directory for figure outputs
SAVED_OUTPUTS_DIR = os.path.join(RESULTS_DIR, "saved_outputs")  # pkl + txt data inputs
BOLUS_SEC     = 480.0               # injection time in seconds (8 min)
POST1_END_SEC = BOLUS_SEC + 600.0   # end of "first 10 min post-injection" (18 min)
COLORMAP      = 'RdBu_r'           # diverging colourmap; well suited for correlation

# Fixed colour scale applied to every panel in every figure
VMIN = -1.0
VMAX =  1.0

# ===========================================================================
# 1. Data Loading
# ===========================================================================

def load_data(results_dir):
    """
    Load FC matrices, timestamps, epoch labels, and parcel names from disk.

    Parameters
    ----------
    results_dir : str

    Returns
    -------
    fc       : dict  {'dmt': list of (n_ep, n_r, n_r), 'pcb': ..., 'subject_ids': ...}
    ts       : dict  {'dmt': list of (n_ep,), 'pcb': ...}
    labels   : dict  {'dmt': list of (n_ep,) str, 'pcb': ...}
    names    : list of str   region / network names
    """
    def _load(fname):
        with open(os.path.join(results_dir, fname), 'rb') as f:
            return pickle.load(f)

    fc     = _load('fc_matrices.pkl')
    ts     = _load('timestamps.pkl')
    labels = _load('epoch_labels.pkl')

    names_path = os.path.join(results_dir, 'parcel_names.txt')
    with open(names_path) as f:
        names = [ln.strip() for ln in f if ln.strip()]

    n_subj = len(fc['dmt'])
    print(f"Loaded FC data: {n_subj} subjects, {len(names)} regions")
    print(f"Subject IDs: {fc['subject_ids']}")
    return fc, ts, labels, names


# ===========================================================================
# 2. Time-window helpers
# ===========================================================================

def select_epochs(fc_subj, ts_subj, t_min, t_max):
    """
    Return FC matrices for epochs whose start time falls in [t_min, t_max).

    Parameters
    ----------
    fc_subj  : ndarray, shape (n_epochs, n_r, n_r)
    ts_subj  : ndarray, shape (n_epochs,)   — epoch start times in seconds
    t_min    : float or None   (None = no lower bound)
    t_max    : float or None   (None = no upper bound)

    Returns
    -------
    selected : ndarray, shape (n_selected, n_r, n_r)  or None if no epochs
    """
    mask = np.ones(len(ts_subj), dtype=bool)
    if t_min is not None:
        mask &= ts_subj >= t_min
    if t_max is not None:
        mask &= ts_subj < t_max
    if not mask.any():
        return None
    return fc_subj[mask]


def group_mean_fc(fc_list, ts_list, t_min, t_max):
    """
    Compute the group-mean FC matrix for a given time window.

    Averages first across epochs within each subject, then across subjects.

    Parameters
    ----------
    fc_list  : list of ndarray, shape (n_epochs_i, n_r, n_r)
    ts_list  : list of ndarray, shape (n_epochs_i,)
    t_min    : float or None
    t_max    : float or None

    Returns
    -------
    mean_fc  : ndarray, shape (n_r, n_r)
    n_valid  : int   number of subjects contributing to the mean
    """
    subject_means = []
    for fc_s, ts_s in zip(fc_list, ts_list):
        sel = select_epochs(fc_s, ts_s, t_min, t_max)
        if sel is not None and len(sel) > 0:
            subject_means.append(sel.mean(axis=0))

    if not subject_means:
        n_r = fc_list[0].shape[-1]
        return np.full((n_r, n_r), np.nan), 0

    return np.nanmean(np.stack(subject_means, axis=0), axis=0), len(subject_means)


# ===========================================================================
# 3. Define the four time windows
# ===========================================================================

def build_windows(bolus_sec, post1_end_sec, total_end_sec):
    """
    Return an ordered list of (label, t_min, t_max) tuples.

    Parameters
    ----------
    bolus_sec      : float   injection onset (seconds)
    post1_end_sec  : float   end of first post-injection window (seconds)
    total_end_sec  : float   end of session (seconds); set to None for open end

    Returns
    -------
    windows : list of (str, float|None, float|None)
    """
    return [
        ('Full session',              None,          None),
        ('Pre-injection (baseline)',  None,          bolus_sec),
        ('Post-injection 0–10 min',   bolus_sec,     post1_end_sec),
        ('Post-injection >10 min',    post1_end_sec, None),
    ]


def get_session_end(ts_list):
    """Estimate the last epoch start time across all subjects."""
    return max(ts_s.max() for ts_s in ts_list if len(ts_s) > 0)


# ===========================================================================
# 4. Plotting helpers
# ===========================================================================

def plot_fc_panel(ax, fc_matrix, title, names,
                  show_ylabels=True, show_xlabels=True,
                  vmin=VMIN, vmax=VMAX, cmap=COLORMAP, n_valid=None):
    """
    Draw a single FC heatmap on the given Axes.

    Parameters
    ----------
    ax            : matplotlib.axes.Axes
    fc_matrix     : ndarray, shape (n_r, n_r)
    title         : str
    names         : list of str   — printed in full, no truncation
    show_ylabels  : bool          — draw y-axis tick labels (left column only)
    show_xlabels  : bool          — draw x-axis tick labels (bottom row only)
    vmin/vmax     : float         — fixed colour scale limits
    cmap          : str
    n_valid       : int or None   — number of subjects; shown in subtitle

    Returns
    -------
    im : matplotlib.image.AxesImage
    """
    n_r   = fc_matrix.shape[0]
    ticks = np.arange(n_r)

    im = ax.imshow(fc_matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')

    # Y-axis: labels only on left column
    ax.set_yticks(ticks)
    if show_ylabels:
        ax.set_yticklabels(names, fontsize=6)
    else:
        ax.set_yticklabels([])

    # X-axis: labels only on bottom row
    ax.set_xticks(ticks)
    if show_xlabels:
        ax.set_xticklabels(names, rotation=90, fontsize=6)
    else:
        ax.set_xticklabels([])

    subtitle = title
    if n_valid is not None:
        subtitle += f'\n(n={n_valid} subjects)'
    ax.set_title(subtitle, fontsize=9, pad=4)

    return im


def make_figure(fc_panels, titles, names, suptitle, cmap=COLORMAP):
    """
    Produce a 2x2 figure of FC heatmaps with a single shared colourbar.

    Panel order (matches ``titles`` and ``fc_panels``):
        [0] top-left     Full session
        [1] top-right    Pre-injection (baseline)
        [2] bottom-left  Post-injection 0-10 min
        [3] bottom-right Post-injection >10 min

    Colour scale is fixed at VMIN / VMAX for every panel.
    Y-axis labels appear on the left column only; X-axis labels on the
    bottom row only.  All region/network names are printed in full.

    Parameters
    ----------
    fc_panels : list of (ndarray, int)   [(fc_matrix, n_valid), ...], length 4
    titles    : list of str              length 4
    names     : list of str              region / network names
    suptitle  : str
    cmap      : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Leave a narrow strip on the right for the shared colourbar
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 12),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.05,
                     'right': 0.88},
    )
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]   # (row, col) for each panel
    last_im   = None

    for idx, ((row, col), (fc_mat, n_valid), title) in enumerate(
            zip(positions, fc_panels, titles)):
        ax = axes[row, col]
        im = plot_fc_panel(
            ax, fc_mat, title, names,
            show_ylabels=(col == 0),
            show_xlabels=(row == 1),
            vmin=VMIN, vmax=VMAX,
            cmap=cmap, n_valid=n_valid,
        )
        last_im = im

    # Single shared colourbar in a dedicated axes to the right of the grid
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cb = fig.colorbar(last_im, cax=cbar_ax)
    cb.set_label('Functional connectivity', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    return fig


# ===========================================================================
# 5. Main plotting routine
# ===========================================================================

def plot_fc_figures(fc, ts, names, results_dir,
                    file_tag='', cmap=COLORMAP):
    """
    Generate and save the three 4-panel FC figures for one dataset
    (region-level or network-level).

    Parameters
    ----------
    fc          : dict {'dmt': [...], 'pcb': [...]}
    ts          : dict {'dmt': [...], 'pcb': [...]}
    names       : list of str
    results_dir : str
    file_tag    : str   appended to output filenames (e.g. '_networks')
    cmap        : str
    """
    total_end = get_session_end(ts['dmt'] + ts['pcb'])
    windows   = build_windows(BOLUS_SEC, POST1_END_SEC, total_end)
    win_labels = [w[0] for w in windows]

    print(f"\n  Time windows:")
    for lbl, tlo, thi in windows:
        lo_str = f"{tlo/60:.1f} min" if tlo is not None else "start"
        hi_str = f"{thi/60:.1f} min" if thi is not None else "end"
        print(f"    {lbl}: {lo_str} -> {hi_str}")

    # Compute group-mean FC for each condition × window
    print("\n  Computing group means ...")
    dmt_panels, pcb_panels, diff_panels = [], [], []

    for lbl, t_min, t_max in windows:
        dmt_fc, nd = group_mean_fc(fc['dmt'], ts['dmt'], t_min, t_max)
        pcb_fc, np_ = group_mean_fc(fc['pcb'], ts['pcb'], t_min, t_max)
        diff_fc     = dmt_fc - pcb_fc

        dmt_panels.append((dmt_fc,  nd))
        pcb_panels.append((pcb_fc,  np_))
        diff_panels.append((diff_fc, min(nd, np_)))

        print(f"    {lbl}: DMT n={nd}, PCB n={np_}")

    # --- Figure 1: DMT ---
    fig1 = make_figure(
        dmt_panels, win_labels, names,
        suptitle='DMT — Group-mean functional connectivity',
        cmap=cmap,
    )
    path1 = os.path.join(results_dir, f'fc_DMT{file_tag}.png')
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"\n  Saved: {path1}")

    # --- Figure 2: PCB ---
    fig2 = make_figure(
        pcb_panels, win_labels, names,
        suptitle='Placebo — Group-mean functional connectivity',
        cmap=cmap,
    )
    path2 = os.path.join(results_dir, f'fc_PCB{file_tag}.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # --- Figure 3: DMT − PCB ---
    fig3 = make_figure(
        diff_panels, win_labels, names,
        suptitle='DMT − Placebo — Functional connectivity difference',
        cmap=cmap,
    )
    path3 = os.path.join(results_dir, f'fc_DMT_minus_PCB{file_tag}.png')
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: {path3}")


# ===========================================================================
# 6. Main
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  DMT EEG — Source-Space FC Plots")
    print("=" * 60)

    # --- Load data ---
    print(f"\n[1] Loading data from '{SAVED_OUTPUTS_DIR}/' ...")
    fc, ts, labels, names = load_data(SAVED_OUTPUTS_DIR)

    n_regions = len(names)
    print(f"  Regions/networks: {n_regions}")

    # Determine whether this is a region-level or network-level dataset
    # by checking whether the number of names is small (networks) or large
    # (Schaefer 2018 parcels, 100 regions).
    is_network = n_regions <= 15
    file_tag   = '_networks' if is_network else '_regions'
    level_str  = 'network' if is_network else 'region (Schaefer 2018)'
    print(f"  Detected level: {level_str}")

    # --- Generate figures ---
    print(f"\n[2] Generating FC figures ({level_str} level) ...")
    plot_fc_figures(fc, ts, names, RESULTS_DIR, file_tag=file_tag)

    print("\n" + "=" * 60)
    print("  Done. Figures saved to:", os.path.abspath(RESULTS_DIR))
    print("=" * 60)
