# DMT EEG Analysis

**Author:** Kiret Dhindsa (kiretd@gmail.com)

Reproduces and extends the EEG analysis from:

> Timmermann C., Roseman L., Haridas S., Rosas F., Luan L., Kettner H., Martell J.,
> Erritzoe D., Tagliazucchi E., Pallavacini C., Girn M., Alamia A., Leech R., Nutt D.,
> Carhart-Harris R. (2023) "Human Brain Effects of DMT assessed via fMRI-EEG." *PNAS.*

The sensor-space reproduction (Figures 4A/4B/4D/4E) is implemented first, then extended
into source space to identify which cortical regions and networks drive the observed
DMT-vs-placebo effects.

---

## Environment


All scripts are run from the project root (`\DMT_EEG\`). Adjust the directory structure in the configurations as needed (e.g., line 16 in `reproduce_EEG.py`). 

**Dependencies** (see `requirements.txt`):
- `numpy`, `scipy`, `matplotlib`
- `mne >= 1.6`, `mne-connectivity >= 0.6`
- `h5py` (for .mat file I/O)
- `irasa` (IRASA spectral decomposition — install from source)

**fsaverage atlas data** required at:
```
\mne_data\MNE-fsaverage-data\fsaverage\label\
    lh.Schaefer2018_100Parcels_7Networks_order.annot
    rh.Schaefer2018_100Parcels_7Networks_order.annot
```
The location can be adjusted in the code, but this version is recommended. These are the FreeSurfer surface-space annotation files from the
[Schaefer 2018 parcellation](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label)
(same parcellation as the MNI NIfTI releases; surface format is used here because
source localisation operates on the fsaverage cortical mesh).

---

## 1. Code Files

### Execution order

```
reproduce_timmermann.py        # Step 1 — sensor-space reproduction
sourcespace_analysis.py        # Step 2 — source localisation + FC computation
sourcespace_plots.py           # Step 3 — visualise FC matrices
validate_sourcespace.py        # Step 4 — validate Step 2 outputs
sourcespace_effects.py         # Step 5 — source-space effects analysis
```

---

### `reproduce_timmermann.py`

**Purpose:** Reproduces Figure 4 panels (A, B, D, E) of Timmermann et al. (2023) from
the raw EEG `.mat` files.

**Run:**
```
python reproduce_timmermann.py
```

**Expected output** (`Results/reproduce_timmermann/`):

| File | Contents |
|---|---|
| `fig4A_topomaps.png` | Topomaps of DMT-induced spectral power changes and LZc |
| `fig4B_spectra_lzc.png` | Whole-brain power spectra (DMT vs PCB) and LZc boxplots |
| `fig4D_timecourses.png` | Temporally resolved delta power, alpha power, and LZc with cluster-corrected significance shading |
| `fig4E_traveling_waves.png` | Forward/backward traveling wave power analysis |
| `results_mne_objects.pkl` | MNE EvokedArray objects and group-mean arrays for downstream use (~2.9 GB) — saved to `Results/saved_outputs/` |

Panels C and F from the paper require subjective ratings data (not available) and are
omitted.

---

### `sourcespace_analysis.py`

**Purpose:** Source localisation and dynamic functional connectivity computation for all
14 subjects. Processes one subject at a time (memory-efficient streaming design) to
prevent RAM exhaustion.

**Key configuration flags** (edit near top of file):

| Flag | Default | Effect |
|---|---|---|
| `USE_NETWORKS` | `False` | If `True`, averages 100 parcels into 7 Yeo network signals before saving |
| `SAVE_FC` | `True` | If `True`, computes and saves a pairwise FC matrix per epoch (metric set by `CONNECTIVITY_METRIC`) |
| `CONNECTIVITY_METRIC` | `'correlation'` | `'correlation'`, `'coherence'` or `'wpli'` |
| `INV_METHOD` | `'sLORETA'` | Inverse method: `'MNE'`, `'dSPM'`, or `'sLORETA'` |

**Run:**
```
python sourcespace_analysis.py
```

**Pipeline steps:**
1. Discover valid subject folders (excludes subjects `01, 07, 11, 16, 19, 25`).
2. Build a shared fsaverage forward model (32-channel 10-20 montage, 5124 sources).
3. Load Schaefer 2018 atlas (100 parcels, 7 Yeo networks) from `.annot` files.
4. For each subject, process DMT then PCB sessions:
   - Apply sLORETA inverse operator to every 2-second epoch.
   - Parcellate source activity into 100 Schaefer parcels (mean-flip mode).
   - Compute pairwise FC matrix per epoch using the selected `CONNECTIVITY_METRIC` (`'correlation'`, `'coherence'`, or `'wpli'`).
   - Save subject temp file; free all large arrays before loading the next subject.
5. Assemble temp files into final output pkl files; delete temp files.

**Expected output** (`Results/saved_outputs/`):

| File | Shape / Contents |
|---|---|
| `source_timeseries.pkl` | Dict: `'dmt'`/`'pcb'` → list of `(n_epochs, 100, 500)` arrays per subject |
| `fc_matrices.pkl` | Dict: `'dmt'`/`'pcb'` → list of `(n_epochs, 100, 100)` arrays per subject (~1.7 GB) |
| `timestamps.pkl` | Dict: `'dmt'`/`'pcb'` → list of `(n_epochs,)` epoch start times in seconds |
| `epoch_labels.pkl` | Dict: `'dmt'`/`'pcb'` → list of `(n_epochs,)` string arrays (`'baseline'` / `'post_bolus'`) |
| `parcel_names.txt` | 100 Schaefer parcel names, one per line |

---

### `sourcespace_plots.py`

**Purpose:** Visualises the functional connectivity matrices produced by
`sourcespace_analysis.py` as heatmaps across four time windows.

**Run:**
```
python sourcespace_plots.py
```

**Expected output** (`Results/`):

| File | Contents |
|---|---|
| `fc_DMT_regions.png` | 2×2 grid: group-mean FC matrix for DMT across four time windows |
| `fc_PCB_regions.png` | Same for placebo |
| `fc_DMT_minus_PCB_regions.png` | DMT − PCB difference FC for the same four windows |

Each panel covers one time window: full session, pre-injection baseline,
post-injection 0–10 min, and post-injection >10 min. Colour scale is fixed at −1 to 1.

---

### `validate_sourcespace.py`

**Purpose:** Automated validation suite for the source-space pipeline. Runs 19 checks
across 6 categories and writes a pass/fail report.

**Run:**
```
python validate_sourcespace.py
```

**Expected output** (`Results/validation/`): 14 figures + 1 text report (see
[Section 4](#4-validation-of-source-space-analysis) for details).

---

### `sourcespace_effects.py`

**Purpose:** Maps the sensor-space DMT effects (Figs 4B and 4D) into source space,
identifying which Schaefer parcels and Yeo networks drive the changes.

**Requires:** Outputs from `sourcespace_analysis.py` to be present in `Results/`.

**Run:**
```
python sourcespace_effects.py
```

**Three analysis sections:**

| Section | Maps | Output figures |
|---|---|---|
| 1 — Spectral | Fig 4B (left) | Band-power bar plots, parcel heatmap, network PSD curves |
| 2 — LZc | Fig 4B (right) | Network-level and parcel-level LZc bar charts |
| 3 — Timecourses | Fig 4D | 7×3 network timecourse grid, summary heatmap |

---

## 2. Timmermann Paper Reproduction

### What is reproduced

The script `reproduce_timmermann.py` reproduces four of the six panels of Figure 4 from
Timmermann et al. (2023), which characterise the whole-brain EEG effects of intravenous
DMT versus placebo (PCB) in a double-blind crossover design (N = 20 subjects; 6 excluded
for poor data quality, leaving 14 analysed).

| Panel | What it shows |
|---|---|
| **4A** | Sensor topomaps of DMT-induced changes in delta power, alpha power, broadband power, and Lempel-Ziv complexity (LZc) |
| **4B** | Group-mean power spectra (DMT vs PCB, whole-scalp average) and LZc comparison boxplot |
| **4D** | Epoch-by-epoch timecourses of delta power, alpha power, and LZc from session start through 16 min, with cluster-corrected significance bands |
| **4E** | Forward vs backward traveling wave power ratio (DMT vs PCB) |

Panels C and F require subjective drug-experience ratings, which are not available in
the data release and are not reproduced.

### Analysis methods

- **Data:** 32-channel EEG, 250 Hz, recorded concurrently with fMRI. Epochs are
  non-overlapping 2-second segments. Bolus injection at t = 8 min; analysis runs to
  t = 16 min (covers the primary drug effect window).
- **Spectral power:** Welch PSD, per-electrode, then averaged across channels.
  Band power extracted for delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz),
  beta (13–30 Hz), gamma (30–45 Hz).
- **Signal complexity:** Lempel-Ziv complexity (LZc, 1976 definition) on binarised EEG
  (median-split threshold), averaged across electrodes.
- **Traveling waves:** Forward/backward spatial gradient analysis per epoch.
- **Statistics:** Non-parametric permutation cluster test (7,500 permutations) for
  timecourse significance. Sensor topomaps use paired t-statistics.

### Key findings reproduced

- DMT increases delta power (1–4 Hz) relative to baseline and placebo.
- Alpha power (8–13 Hz) is suppressed under DMT.
- LZc is elevated under DMT, indicating increased signal complexity/diversity.
- Effects peak approximately 2–4 minutes post-injection and partially recover by 16 min.

---

## 3. Source-Space Analysis

### Goal

Identify *which* cortical regions and networks drive the sensor-space effects seen in
Figure 4B and 4D — i.e., where in the brain the DMT-induced delta increase, alpha
suppression, and LZc elevation originate.

### Atlas

**Schaefer 2018, 100 parcels, Yeo 7 networks** (surface-space FreeSurfer annotations
on fsaverage). This is the same parcellation as the volumetric MNI releases from the
same lab; the `.annot` surface format is used here because MNE source estimates live on
the fsaverage cortical mesh.

The 7 networks and their parcel counts:

| Network key | Display name | Parcels |
|---|---|---|
| `Default` | Default Mode Network | 24 |
| `Vis` | Visual Network | 17 |
| `DorsAttn` | Dorsal Attention Network | 15 |
| `SomMot` | Somatomotor Network | 14 |
| `Cont` | Frontoparietal Network | 13 |
| `SalVentAttn` | Salience Network | 12 |
| `Limbic` | Limbic Network | 5 |

### Forward model

- Template: **fsaverage** (MNI standard head, no individual MRIs required).
- Electrodes: standard 10-20 32-channel montage.
- Sources: 5,124 fixed-orientation dipoles on the cortical surface.
- Coregistration: MNE default fsaverage coregistration.

### Inverse solution

**sLORETA** (standardised low-resolution electromagnetic tomography), SNR = 3.0
(λ² = 1/9). Applied to every 2-second epoch independently.

sLORETA was chosen over MNE/dSPM because it produces depth-normalised estimates with
zero localisation error for single dipoles in noiseless conditions, making it better
suited for comparing activity across cortical regions.

### Parcellation

Source time-series are parcellated using `mne.extract_label_time_course` with
`mode='mean_flip'`. Mean-flip mode computes the mean of all dipoles within a label,
with sign flips to avoid polarity cancellation (dipoles on opposite walls of a sulcus
can point in opposing directions).

Each processed epoch produces a `(100, 500)` array — 100 parcels × 500 time points
(2 s at 250 Hz).

### Functional connectivity

Pairwise **Pearson correlation** of the 100 parcel timeseries per epoch, yielding a
`(100, 100)` symmetric FC matrix with diagonal exactly 1.0.

Dynamic FC is tracked by computing one matrix per 2-second epoch, producing a
time-resolved FC tensor of shape `(n_epochs, 100, 100)` per subject per session.

### Subjects and data

14 subjects (IDs: 02, 03, 06, 08, 09, 10, 12, 13, 14, 15, 17, 18, 22, 23).
Excluded: 01, 07, 11, 16, 19, 25 (poor data quality, consistent with the paper).
Each subject has two sessions: DMT and PCB (placebo).

---

## 4. Validation of Source-Space Analysis

`validate_sourcespace.py` runs 19 automated checks across 6 categories.
Current result: **19 PASS / 0 WARN / 0 FAIL**.

### Category 1 — Forward / inverse solution

| Check | Figure | Result |
|---|---|---|
| **1a** Lead-field sensitivity topomap. Row-norms of the gain matrix should be bilaterally symmetric (LH/RH ratio ≈ 1.0). | `val_1a_sensitivity_topomap.png` | PASS — ratio = 1.00 |
| **1b** Sensor reconstruction R². Source estimates are projected back to sensor space and compared with the original epochs; R² should exceed 0.30 for a template head model. | `val_1b_reconstruction_r2.png` | PASS — mean R² = 0.413 |

### Category 2 — Source time-series plausibility

| Check | Figure | Result |
|---|---|---|
| **2a** Alpha-peak parcel location. The parcel with the highest group-mean alpha power (8–13 Hz) should lie in the Visual network (occipital cortex). | `val_2a_alpha_peak_location.png` | PASS — top parcel = `7Networks_RH_Vis_2-rh`; 5/5 top-alpha parcels are Visual |
| **2b** Source vs sensor PSD shape. Log-PSD of source-space mean should correlate strongly (r > 0.85) with sensor-space mean PSD. | `val_2b_source_vs_sensor_psd.png` | PASS — r = 0.980 |

### Category 3 — Parcellation coverage

| Check | Figure | Result |
|---|---|---|
| **3a** Dipole count per parcel. Any parcel containing zero dipoles would produce a flat timeseries and is flagged as a failure. | `val_3a_parcel_coverage.png` | PASS — 0/100 parcels empty |
| **3b** Label-average vs single-epoch PSD. Confirms that the mean-flip averaging is consistent across different epoch aggregation strategies (r > 0.80). | `val_3b_label_avg_vs_peak_epoch.png` | PASS — r = 0.996 |

### Category 4 — FC matrix numerical properties

| Check | Figure | Result |
|---|---|---|
| **4a** Symmetry. Max \|FC − FCᵀ\| should be < 1e-6 (Pearson correlation is symmetric by construction). | `val_4a_symmetry.png` | PASS — 2.22e-16 (machine epsilon) |
| **4b** Diagonal. Max \|diag(FC) − 1\| should be < 1e-6. | `val_4b_diagonal.png` | PASS — 2.22e-16 |
| **4c** Off-diagonal distribution. Fraction of \|off-diag\| > 0.9 should be < 0.10; median should be near 0. | `val_4c_offdiag_distribution.png` | PASS — DMT: 0.6% > 0.9, median = −0.006 |

### Category 5 — Condition contrast sanity

| Check | Figure | Result |
|---|---|---|
| **5a** Mean FC timecourse. DMT mean FC should show a detectable change post-injection (|Δ| > 0.0001). | `val_5a_mean_fc_timecourse.png` | PASS — Δ = 0.0002 |
| **5b** DMT − PCB difference matrix. Visualises the strongest connections that differ between conditions; confirms the difference matrix is non-trivial. | `val_5b_difference_matrix.png` | PASS — max = 0.344 (Salience), min = −0.358 (Frontoparietal × Somatomotor) |

### Category 6 — Unit / implementation tests (synthetic data)

| Check | Figure | Result |
|---|---|---|
| **6a** Sinusoid recovery. A 10 Hz sinusoid is injected into one source; the recovered parcel PSD peak should be at 10.0 ± 0.5 Hz. | `val_6a_sinusoid_recovery.png` | PASS — peak at 10.00 Hz |
| **6b** White-noise FC. FC of independent white-noise parcels should have mean off-diagonal ≈ 0 (\|val\| < 0.10). | `val_6b_white_noise_fc.png` | PASS — mean = −0.0009 |
| **6c** Duplicate-parcel FC. FC between two identical timeseries should be exactly 1.0. | `val_6c_duplicate_parcel.png` | PASS — r = 1.0000000000 |
| **6d** Timestamp shuffle. Shuffling epoch timestamps should eliminate the pre/post FC difference seen with real timestamps. | `val_6d_timestamp_shuffle.png` | PASS — real Δ = 0.0002; shuffled Δ = 0.0000 |
