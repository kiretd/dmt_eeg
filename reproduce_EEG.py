import os
import numpy as np
import pickle
from itertools import chain
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import tfr_morlet
from mne.stats import permutation_cluster_test as pct
from mne.stats import permutation_t_test as ptt
from irasa.IRASA import IRASA

# Define the folder containing the EEG data
eeg_folder = "EEG"
sr = 250 # Sampling rate 


# %% Load Data
# Initialize lists to store EEG data matrices and subject IDs
dmt_sessions = []
pcb_sessions = []
subject_ids = []

remove_list = ['01','07','11','16','19','25'] # matlab code removes 07 and 25 too

# Iterate through each subject folder in the EEG directory
for subject_folder in os.listdir(eeg_folder):
    subject_path = os.path.join(eeg_folder, subject_folder)
    
    if os.path.isdir(subject_path):
        # Extract the last two characters of the subject folder name as the subject ID
        subject_id = subject_folder[-2:]        
        if subject_id not in remove_list:
            subject_ids.append(subject_id)

            # Paths for DMT and PCB sessions
            dmt_path = os.path.join(subject_path, "ses_DMT", "dataref.mat")
            pcb_path = os.path.join(subject_path, "ses_PCB", "dataref.mat")
            
            # Load DMT session data if it exists, otherwise insert an empty matrix
            if os.path.exists(dmt_path):
                dmt_data = scipy.io.loadmat(dmt_path, simplify_cells=True)
                dmt_sessions.append(dmt_data['dataref'])
                            
            # Load PCB session data if it exists, otherwise insert an empty matrix
            if os.path.exists(pcb_path):
                pcb_data = scipy.io.loadmat(pcb_path, simplify_cells=True)
                pcb_sessions.append(pcb_data['dataref'])
            

# %% MNE setup for analysis
def create_mne_info(channel_labels, sfreq=250):
    """
    Creates a standard 10-20 montage for a given list of EEG channel labels.

    Parameters:
    - labels: list of channel name strings (e.g., ['Fp1', 'Fp2', ..., 'O2'])

    Returns:
    - montage: an MNE DigMontage object
    """
    # Ensure labels are flat strings (not nested arrays)
    labels = [label.strip() if isinstance(label, str) else label[0].strip() for label in channel_labels]
    misc_types = ['EOG','EOG','ECG1','ECG2'] 
    ch_types = ['misc' if lab in misc_types else 'eeg' for lab in labels]
    
    info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=ch_types)
    
    # Load standard montage
    full_montage = mne.channels.make_standard_montage('standard_1020')

    # Filter montage to only use available channels
    ch_pos = {ch: pos for ch, pos in full_montage.get_positions()['ch_pos'].items() if ch in labels}
    
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    info.set_montage(montage, match_case=False)
    return info

# info = create_mne_info(dmt_sessions[1]['label'], sfreq=250)


def find_bolus_onset(start_times):
    bolus_onset_idx = np.argmax(start_times >= 480.)
    return bolus_onset_idx

def create_epochs(session, info, event_id=1, tmin=0.0):
    times = np.vstack(session['time'])
    data = np.stack(session['trial'], axis=0)
    n = data.shape[0]
    
    bolus_idx = find_bolus_onset(times[:,0])
    events = np.array([[i, 0, event_id] for i in range(n)])
    events[:bolus_idx,2] = 0
    
    epochs = mne.EpochsArray(data, info, events=events, event_id={'baseline': 0, 'post_bolus': 1}, 
                             tmin=tmin, verbose=False)
    return epochs

# epochs = create_epochs(dmt_sessions[1], info)


# Get epochs for all subjects
infos_dmt = [create_mne_info(sess['label']) for sess in dmt_sessions]# if isdict(sess) else []]
epochs_dmt = [create_epochs(sess, info) for sess, info in zip(dmt_sessions, infos_dmt)]

infos_pcb = [create_mne_info(sess['label']) for sess in pcb_sessions]
epochs_pcb = [create_epochs(sess, info) for sess, info in zip(pcb_sessions, infos_pcb)]


# check full channel list and montage -- THESE ARE IN THE SAME ORDER
ch_lists = [ep.info.get_montage().ch_names for ep in epochs_dmt]
all_chans_dmt = set(ch for names in ch_lists for ch in names)

ch_lists = [ep.info.get_montage().ch_names for ep in epochs_pcb]
all_chans_pcb = set(ch for names in ch_lists for ch in names)

# Adjacency matrix
ADJ = mne.channels.find_ch_adjacency(epochs_dmt[0].info, ch_type='eeg')


# Get IRASA power estimates
# irasa_dmt = [IRASA(ep.get_data(picks='eeg'), freqs=np.arange(1,30)) for ep in epochs_dmt]
# irasa_pcb = [IRASA(ep.get_data(picks='eeg'), freqs=np.arange(1,30)) for ep in epochs_pcb]


# with open("irasa_dmt.pkl", "wb") as f:
#     pickle.dump(irasa_dmt, f)
# with open("irasa_pcb.pkl", "wb") as f:
#     pickle.dump(irasa_pcb, f)
    
with open("irasa_dmt.pkl", "rb") as f:
    irasa_dmt = pickle.load(f)
with open("irasa_pcb.pkl", "rb") as f:
    irasa_pcb = pickle.load(f)


# %%
# get power spectra for each subject
psd_dmt = [ep.compute_psd(fmin=2,fmax=30).average() for ep in epochs_dmt]
psd_pcb = [ep.compute_psd(fmin=2,fmax=30).average() for ep in epochs_pcb]

mean_psd_dmt = np.mean(np.array([psd.get_data() for psd in psd_dmt]), axis=0)
mean_psd_pcb = np.mean(np.array([psd.get_data() for psd in psd_pcb]), axis=0)
freqs = psd_dmt[0].freqs

ch_names = psd_dmt[0].ch_names

# Plot each channel's average PSD
plt.figure(figsize=(10, 6))
for ch_idx, ch_name in enumerate(ch_names):
    plt.plot(freqs, mean_psd_dmt[ch_idx], label=ch_name)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density ()')
plt.title('Average PSD per Channel (DMT, 8–13 Hz)')
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for ch_idx, ch_name in enumerate(ch_names):
    plt.plot(freqs, mean_psd_pcb[ch_idx], label=ch_name)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density ()')
plt.title('Average PSD per Channel (PCB, 8–13 Hz)')
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.show()



# %% IRASA test


# irasa_dmt = IRASA(epochs_dmt[0]['post_bolus'].get_data(), freqs=np.arange(2,30))
# irasa_pcb = IRASA(epochs_pcb[0]['post_bolus'].get_data(), freqs=np.arange(2,30))

plt.figure(figsize=(10,4))
plt.subplot(121)
irasa_dmt.psdplot(fit=True)
plt.subplot(122)
irasa_dmt.loglogplot(fit=True)


plt.figure(figsize=(10,4))
plt.subplot(121)
irasa_pcb.psdplot(fit=True)
plt.subplot(122)
irasa_pcb.loglogplot(fit=True)



osc_dmt = np.log10(np.mean(irasa_dmt.mixed, axis=0)) - np.log10(np.mean(irasa_dmt.fractal, axis=0))
osc_pcb = np.log10(np.mean(irasa_pcb.mixed, axis=0)) - np.log10(np.mean(irasa_pcb.fractal, axis=0))

plt.figure()
plt.plot(irasa_pcb.freqs, osc_dmt.mean(0), label='DMT')
plt.plot(irasa_pcb.freqs, osc_pcb.mean(0), label='PCB')
plt.axhline(0, c='k')
plt.legend()
plt.ylabel('Oscillatory power')
plt.xlabel("frequency (Hz)")


# %% Delta, Alpha and LZc over time plots 
def mean_band_power(epochs, fmin, fmax):
    psd = [10 * np.log10(ep.compute_psd(fmin=fmin, fmax=fmax)) for ep in epochs]
    # mean_psd = np.mean(np.array([10*np.log10(p.get_data()) for p in psd]), axis=0)
    return psd

def align_chans(data_arrays, ch_lists):
    all_chans = set(ch for names in ch_lists for ch in names)
    ch_index = {ch: i for i, ch in enumerate(all_chans)}
    
    aligned_data = []
    for data, ch_names in zip(data_arrays, ch_lists):
        aligned = np.full((data.shape[0], len(all_chans)), np.nan)
        for i, ch in enumerate(ch_names):
            aligned[:, ch_index[ch]] = data[:, i]
        aligned_data.append(aligned)
    return aligned_data, all_chans

def align_power_to_time(sessions, metric, time, nomean=False):
    aligned_curve = []
    for sess, pdata in zip(sessions, metric):
        times = [t[0]/60 for t in sess['time']]
        
        if nomean:
            row = np.full((len(time), pdata.shape[-1]), np.nan)
            for t_val, y_val in zip(times, pdata):
                idx = time[t_val]
                row[idx,:] = y_val
        else:
            if len(metric[0].shape) == 3:
                y = pdata.mean(axis=(1,2))
            elif len(metric[0].shape) == 2:
                y = pdata.mean(axis=1)
            row = np.full(len(time), np.nan)
            for t_val, y_val in zip(times, y):
                idx = time[t_val]
                row[idx] = y_val
            
        aligned_curve.append(row) 
    curves = np.stack(aligned_curve, axis=0)
    return curves

def baseline_correct(x, tbase):
    baseline = np.nanmean(x[:,:tbase], axis=1)
    corrected = np.array([x[i,:] - baseline[i] for i in range(len(baseline))])
    return corrected

def plot_time_curves(taxis, dmt_curves, pcb_curves, label='Alpha'):
    """
    Plot the mean curves with a shaded region representing ±1 standard deviation,
    and add a vertical line at x=8 labeled "Bolus Onset".
    """
    plt.figure(figsize=(10, 5))
    mean_vals = np.nanmean(dmt_curves, axis=(0))
    std_vals = np.nanstd(dmt_curves, axis=(0))
    plt.plot(taxis, mean_vals, label='DMT', color='r')
    plt.fill_between(taxis, mean_vals - std_vals, mean_vals + std_vals,
                         color='r', alpha=0.3, label='±1 Std Dev')
    
    mean_vals = np.nanmean(pcb_curves, axis=(0))
    std_vals = np.nanstd(pcb_curves, axis=(0))
    plt.plot(taxis, np.nanmean(pcb_curves, axis=(0)), label='PCB', color='b')
    plt.fill_between(taxis, mean_vals - std_vals, mean_vals + std_vals,
                         color='b', alpha=0.3, label='±1 Std Dev')
    
    # Mark stimulus onset
    plt.axvline(x=8, color='black', linestyle='--')
    plt.text(8, plt.ylim()[1]*0.95, 'Bolus Onset', color='black', ha='right', va='top')
    plt.xlabel('Time')
    plt.ylabel('{} Power (dB)'.format(label))
    plt.title('{} Power Over Time'.format(label))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return None

def chan_curves(data, epochs, sessions, tidx, nomean):
    ch_lists = [ep.info.ch_names for ep in epochs]
    data_aligned, all_chans = align_chans(data, ch_lists)
    return align_power_to_time(sessions, data_aligned, tidx, nomean)

def fill_nan_with_sample_mean(x):
    """
    Replaces NaNs in x (samples, channels, time) with the mean across samples
    at each (channel, time) location.
    """
    x_filled = x.copy()
    
    # Compute mean over samples, ignoring NaNs
    mean_over_samples = np.nanmean(x, axis=0)  # shape: (channels, time)
    
    # Broadcast and fill
    for i in range(x.shape[0]):  # for each sample
        nan_mask = np.isnan(x_filled[i])
        x_filled[i][nan_mask] = mean_over_samples[nan_mask]
    
    return x_filled

def plot_power_over_time(f1, f2, label='no label'):
    freqs = irasa_dmt[0].freqs
    inds = np.where((freqs >= f1) & (freqs <= f2))[0]
    # dmt_pows = [np.log10(np.mean(ir.mixed[:,:,inds], axis=-1)) - np.log10(np.mean(ir.fractal[:,:,inds], axis=-1)) for ir in irasa_dmt]
    # pcb_pows = [np.log10(np.mean(ir.mixed[:,:,inds], axis=-1)) - np.log10(np.mean(ir.fractal[:,:,inds], axis=-1)) for ir in irasa_pcb]
    
    # dmt_pows = [np.log10(np.mean(ir.mixed[:,:,inds], axis=-1)) for ir in irasa_dmt]
    # pcb_pows = [np.log10(np.mean(ir.mixed[:,:,inds], axis=-1)) for ir in irasa_pcb]
    
    # dmt_pows = [np.log10(np.mean(ir.fractal[:,:,inds], axis=-1)) for ir in irasa_dmt]
    # pcb_pows = [np.log10(np.mean(ir.fractal[:,:,inds], axis=-1)) for ir in irasa_pcb]
    
    dmt_pows = mean_band_power(epochs_dmt, f1, f2)
    pcb_pows = mean_band_power(epochs_pcb, f1, f2)
    dmt_pows = [np.nanmean(p, axis=-1) for p in dmt_pows]
    pcb_pows = [np.nanmean(p, axis=-1) for p in pcb_pows]
    
    # Align channels and time dimensions across subjects
    # dmt_chan_curves = chan_curves(dmt_pows, epochs_dmt, dmt_sessions, tidx, nomean=True)
    # pcb_chan_curves = chan_curves(pcb_pows, epochs_pcb, pcb_sessions, tidx, nomean=True)
    dmt_chan_curves = align_power_to_time(dmt_sessions, dmt_pows, tidx, nomean=True)
    pcb_chan_curves = align_power_to_time(pcb_sessions, pcb_pows, tidx, nomean=True)

    
    # Baseline correction
    dmt_corrected_curves = baseline_correct(dmt_chan_curves, tbase)
    pcb_corrected_curves = baseline_correct(pcb_chan_curves, tbase)
    


###############################################
    # Perform permutatino cluster test
    # Get the 8s windows pre- and post-bolus
    x1 = dmt_corrected_curves[:,tbase:tend,:]
    x2 = pcb_corrected_curves[:,tbase:tend,:]
    # T_obs, clusters, cluster_p, H0 = pct([x1, x2],
    #                                       # stat_fun=mne.stats.ttest_ind_no_p,
    #                                      n_permutations=7500)
    # T_obs, clusters, cluster_p, H0 = mne.stats.spatio_temporal_cluster_test([x1, x2],
    #                                       # stat_fun=mne.stats.ttest_ind_no_p,
    #                                      n_permutations=7500)
    
    
    # x1 = np.nanmean(dmt_corrected_curves[:,tbase:tend,:], axis=1)
    # x2 = np.nanmean(pcb_corrected_curves[:,tbase:tend,:], axis=1)
    # T_obs, pvals, H0 = ptt([x1, x2], n_permutations=7500)
    
    # x1 = dmt_corrected_curves[:,tbase:tend,:]
    # x2 = pcb_corrected_curves[:,tbase:tend,:]
    # x1 = fill_nan_with_sample_mean(x1)
    # x2 = fill_nan_with_sample_mean(x2)
    # T_obs, clusters, cluster_p, H0 = pct([x1, x2], stat_fun=mne.stats.ttest_ind_no_p, n_permutations=7500)
    
    # dep samples t-test
    T_obs, clusters, cluster_p, H0 = mne.stats.spatio_temporal_cluster_1samp_test(x1-x2,
                                         adjacency=ADJ[0],
                                         n_permutations=7500)
  ##############################################################  
    # get the significant channels
    sig_chans = list(np.unique([ch for clus in clusters for ch in clus[1]]))
    
    # Plot the time curves using only significant channels to average power
    dmt_curves = np.nanmean(dmt_corrected_curves[:,:,sig_chans], axis=2)
    pcb_curves = np.nanmean(pcb_corrected_curves[:,:,sig_chans], axis=2)
    
    plot_time_curves(all_times, 
                     gaussian_filter1d(dmt_curves, sigma=1, axis=1), 
                     gaussian_filter1d(pcb_curves, sigma=1, axis=1), 
                     label=label)
    return None
    

all_times = sorted(set(t[0]/60 for sess in dmt_sessions + pcb_sessions for t in sess['time']))
tidx = {t: i for i, t in enumerate(all_times)}
tbase = next((v for k, v in tidx.items() if k >= 8), None)-1
tend = next((v for k, v in tidx.items() if k >= 16), None)-1

    
 # Alpha curves  


plot_power_over_time(1, 4, label='Delta') 
plot_power_over_time(8, 13, label='Alpha')


# %%
def plot_MNEpower_over_time(f1, f2, label='no label', do_ttest=False):
    dmt_pows = mean_band_power(epochs_dmt, f1, f2)
    pcb_pows = mean_band_power(epochs_pcb, f1, f2)
    
    dmt_pows = [np.nanmean(p, axis=-1) for p in dmt_pows]
    pcb_pows = [np.nanmean(p, axis=-1) for p in pcb_pows]
    
    # Align channels and time dimensions across subjects
    dmt_chan_curves = align_power_to_time(dmt_sessions, dmt_pows, tidx, nomean=True)
    pcb_chan_curves = align_power_to_time(pcb_sessions, pcb_pows, tidx, nomean=True)
    
    # Baseline correction
    dmt_corrected_curves = baseline_correct(dmt_chan_curves, tbase)
    pcb_corrected_curves = baseline_correct(pcb_chan_curves, tbase)
    

    if do_ttest:
        ###############################################
        # Perform permutatino cluster test
        # Get the 8s windows pre- and post-bolus
        x1 = dmt_corrected_curves[:,tbase:tend,:]
        x2 = pcb_corrected_curves[:,tbase:tend,:]
        T_obs, clusters, cluster_p, H0 = mne.stats.spatio_temporal_cluster_1samp_test(x1-x2,
                                              adjacency=ADJ[0],
                                              n_permutations=7500)
        # get the significant channels
        sig_chans = list(np.unique([ch for clus in clusters for ch in clus[1]]))
        ##############################################################  
    else:
        sig_chans = list(range(dmt_chan_curves.shape[-1]))
    
    # Plot the time curves using only significant channels to average power
    dmt_curves = np.nanmean(dmt_corrected_curves[:,:,sig_chans], axis=2)
    pcb_curves = np.nanmean(pcb_corrected_curves[:,:,sig_chans], axis=2)
    
    plot_time_curves(all_times, 
                     gaussian_filter1d(dmt_curves, sigma=1, axis=1), 
                     gaussian_filter1d(pcb_curves, sigma=1, axis=1), 
                     label=label)
    return None

plot_MNEpower_over_time(1, 4, label='Delta', do_ttest=False)
plot_MNEpower_over_time(8, 13, label='Alpha', do_ttest=False)


# %% Difference Spectrograms
def get_pow_diff(f1, f2):
    freqs = irasa_dmt[0].freqs
    inds = np.where((freqs >= 1) & (freqs <= 30))[0]
    dmt_pows = [np.nanmean((np.log10(ir.mixed) - np.log10(ir.fractal))[:,:,inds], axis=-1) for ir in irasa_dmt]
    pcb_pows = [np.nanmean((np.log10(ir.mixed) - np.log10(ir.fractal))[:,:,inds], axis=-1) for ir in irasa_pcb]
    # dmt_pows = [np.log10(np.mean(ir.mixed[:,:,inds], axis=-1)) - np.log10(np.mean(ir.fractal[:,:,inds], axis=-1)) for ir in irasa_dmt]
    # pcb_pows = [np.log10(np.mean(ir.mixed[:,:,inds], axis=-1)) - np.log10(np.mean(ir.fractal[:,:,inds], axis=-1)) for ir in irasa_pcb]
    
    dmt_chan_curves = align_power_to_time(dmt_sessions, dmt_pows, tidx, nomean=True)
    pcb_chan_curves = align_power_to_time(pcb_sessions, pcb_pows, tidx, nomean=True)
    
    # Baseline correction
    dmt_corrected_curves = baseline_correct(dmt_chan_curves, tbase)
    pcb_corrected_curves = baseline_correct(pcb_chan_curves, tbase)

    diff_pow = np.nanmean(dmt_corrected_curves[:,tbase:tend,:], axis=(0,1)) - np.nanmean(pcb_corrected_curves[:,tbase:tend,:])
    return diff_pow

alpha_diff = get_pow_diff(1,4)
mne.viz.plot_topomap(alpha_diff, epochs_dmt[0].info, sensors=True)
alpha_diff = get_pow_diff(4,8)
mne.viz.plot_topomap(alpha_diff, epochs_dmt[0].info, sensors=True)
alpha_diff = get_pow_diff(8,13)
mne.viz.plot_topomap(alpha_diff, epochs_dmt[0].info, sensors=True)
alpha_diff = get_pow_diff(13,30)
mne.viz.plot_topomap(alpha_diff, epochs_dmt[0].info, sensors=True)


# %% LZc
def compute_lz76_complexity(signal):
    """
    Computes Lempel-Ziv 1976 complexity of a binary sequence.
    """
    i, k, l = 0, 1, 1
    c = 1
    n = len(signal)
    while True:
        if i + k == n:
            c += 1
            break
        if signal[i + k] == signal[l + k]:
            k += 1
            if l + k == n:
                c += 1
                break
        else:
            if k > 0:
                i += 1
                k = 1
            else:
                c += 1
                l += 1
                i = 0
                k = 1
    return c

def compute_lz_for_epochs(epochs):
    """
    Computes Lempel-Ziv complexity (LZs) for each channel in each epoch.
    Binarizes each signal using the mean of the epoch (per channel).

    Parameters:
    - epochs: MNE Epochs object (n_epochs, n_channels, n_times)

    Returns:
    - lz_matrix: ndarray of shape (n_epochs, n_channels) with LZs values
    """
    import numpy as np

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape
    lz_matrix = np.zeros((n_epochs, n_channels))

    for i in range(n_epochs):
        for j in range(n_channels):
            signal = data[i, j, :]
            threshold = signal.mean()
            binary_seq = (signal > threshold).astype(int)
            lz_matrix[i, j] = compute_lz76_complexity(binary_seq)

    return lz_matrix

lz_matrix = compute_lz_for_epochs(epochs_dmt[0])

# %% LZc time curves
def compute_lz76_complexity(binary_seq):
    """
    Faithful implementation of the Lempel-Ziv 1976 complexity algorithm
    for binary sequences, based on standard pseudocode.
    """
    n = len(binary_seq)
    i = 0
    c = 1
    u = 1
    v = 1
    vmax = v

    while u + v <= n:
        if binary_seq[i + v - 1] == binary_seq[u + v - 1]:
            v += 1
        else:
            if v > vmax:
                vmax = v
            i += 1
            if i == u:
                c += 1
                u += vmax
                v = 1
                i = 0
                vmax = v
            else:
                v = 1

    if v != 1:
        c += 1

    return c

def compute_lz_for_epochs(epochs):
    """
    Computes Lempel-Ziv complexity (LZs) for each channel in each epoch.
    Binarizes each signal using the mean of the epoch (per channel).

    Parameters:
    - epochs: MNE Epochs object (n_epochs, n_channels, n_times)

    Returns:
    - lz_matrix: ndarray of shape (n_epochs, n_channels) with LZs values
    """
    import numpy as np

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape
    lz_matrix = np.zeros((n_epochs, n_channels))

    for i in range(n_epochs):
        for j in range(n_channels):
            signal = data[i, j, :]
            threshold = signal.mean()
            binary_seq = (signal > threshold).astype(int)
            lz_matrix[i, j] = compute_lz76_complexity(binary_seq.tolist())

    return lz_matrix




LZc_dmt = [compute_lz_for_epochs(ep) for ep in epochs_dmt]
LZc_pcb = [compute_lz_for_epochs(ep) for ep in epochs_pcb]

# Align epochs across subjects
dmt_curves = align_power_to_time(dmt_sessions, LZc_dmt, tidx)
pcb_curves = align_power_to_time(pcb_sessions, LZc_pcb, tidx)

# Baseline correction
tbase = next((v for k, v in tidx.items() if k >= 8), None)-1
tend = next((v for k, v in tidx.items() if k >= 16), None)-1
dmt_corrected = baseline_correct(dmt_curves, tbase)
pcb_corrected = baseline_correct(pcb_curves, tbase)

# Plot aligned curves
plot_time_curves(all_times, 
                 gaussian_filter1d(dmt_corrected, sigma=1, axis=1), 
                 gaussian_filter1d(pcb_corrected, sigma=1, axis=1), 
                 label='LZc')

# get channel-wise curves to find significant channels
ch_lists = [ep.info.ch_names for ep in epochs_dmt]
LZc_dmt_aligned, all_chans_dmt = align_chans(LZc_dmt, ch_lists)
dmt_curves_chans = align_power_to_time(dmt_sessions, LZc_dmt_aligned, tidx, nomean=True)

ch_lists = [ep.info.ch_names for ep in epochs_pcb]
LZc_pcb_aligned, all_chans_pcb = align_chans(LZc_pcb, ch_lists)
pcb_curves_chans = align_power_to_time(pcb_sessions, LZc_pcb_aligned, tidx, nomean=True)

# Baseline correct
dmt_corrected_chans = baseline_correct(dmt_curves_chans, tbase)
pcb_corrected_chans = baseline_correct(pcb_curves_chans, tbase)

# Perform permutation-based clustering test
x1 = dmt_corrected_chans[:,tbase:tend,:]
x2 = pcb_corrected_chans[:,tbase:tend,:]

# x1 = np.nanmean(x1, axis=1)
# x2 = np.nanmean(x2, axis=1)

T_obs, clusters, cluster_p, H0 = pct([x1, x2],
                                     # stat_fun=mne.stats.ttest_ind_no_p,
                                     n_permutations=7500)

sig_chans = list(np.unique([ch for clus in clusters for ch in clus[1]]))


# LZc boxplots
curve_dmt = np.nanmean(x1, axis=1)
curve_pcb = np.nanmean(x2, axis=1)

plt.figure()
# plt.boxplot([np.hstack([arr.ravel() for arr in LZc_dmt]), np.hstack([arr.ravel() for arr in LZc_pcb])], 
            # labels=['DMT','PCB'],patch_artist=True)
x1 = np.concatenate(curve_dmt)
x2 = np.concatenate(curve_pcb)
plt.boxplot([x1[~np.isnan(x1)], x2[~np.isnan(x2)]], 
            labels=['DMT','PCB'],patch_artist=True)
plt.ylabel('')
plt.title('LZc value comparison')
plt.tight_layout()


