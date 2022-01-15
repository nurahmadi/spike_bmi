import numpy as np
from scipy.signal import boxcar, gaussian
from scipy.special import gamma

def kernel_smoother(spikes, t, M, std=None, window='boxcar', density=False):
    """
    Extract spike rates from spike trains using kernel smoothing technique based on boxcar/rectangular or gaussian window.

    Parameters
    ----------
    spikes : ndarray
        The spike event times.
    t : ndarray
        The time at which the firing rate is estimated.
    M : int
        The number of points in the output window. If zero or less, an empty array is returned.
    std : float
        The standard deviation (sigma) for Gaussian window. Default None.
    window : str
        The window type ('boxcar' or 'gaussian'). Default 'boxcar'.
    density : bool, optional
        If True, the spike rate is normalized such that the integral over the range is 1; otherwise the result will be the number of spikes in each time bin.

    Returns
    -------
    spike_rate : ndarray
        The estimated spike rate.
    """
    
    dt = np.diff(t).mean()
    if window=='gaussian':
        wdw = gaussian(M, std)
    else: # window=='boxcar'
        wdw = boxcar(M)
    t_hist = np.concatenate((t-dt/2, np.array([t[-1]+dt/2])))
    y_hist = np.histogram(spikes, t_hist)[0]
    spike_rate = np.convolve(y_hist, wdw, mode='same')
    if density:
        spike_rate = spike_rate / np.trapz(spike_rate, t)
    return spike_rate

def baks(spikes, t, a, b, density=False):
    """
    Bayesian Adaptive Kernel Smoother (BAKS) is a method for estimating firing rate from spike train data that uses 
    kernel smoothing technique with adaptive bandwidth determined using a Bayesian approach.
    
    Parameters
    ----------
    spikes : ndarray
        The spike event times.
    t : ndarray
        The time at which the firing rate is estimated.
    a : float
        The shape parameter (alpha).
    b : float
        The scale parameter (beta).
    density : bool, optional
        If True, the spike rate is normalized such that the integral over the range is 1; otherwise the result will be the number of spikes in each time bin.

    Returns
    -------
    spike_rate : ndarray
        The estimated spike rate.
    bw : ndarray
        The adaptive bandwidth.
 
    More information, please refer to: Ahmadi, N., Constandinou, T. G., & Bouganis, C. S. (2018). 
    Estimation of neuronal firing rate using Bayesian Adaptive Kernel Smoother (BAKS). Plos one, 13(11), e0206794.
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0206794
    """

    N = len(spikes)
    sumnum = 0
    sumdenum = 0
    for i in range(N):
        numerator = (((t - spikes[i])**2)/2 + 1/b)**(-a)
        denumerator = (((t - spikes[i])**2)/2 + 1/b)**(-a-0.5)
        sumnum += numerator
        sumdenum += denumerator
        
    bw = (gamma(a) / gamma(a+0.5)) * (sumnum/sumdenum)
    
    spike_rate = np.zeros(len(t))
    for i in range(N):
        K = (1/(np.sqrt(2*np.pi)*bw)) * np.exp(-((t - spikes[i])**2) / (2*bw**2))
        spike_rate += K  
    if density:
        spike_rate = spike_rate / np.trapz(spike_rate, t)
    return spike_rate, bw

def extract(spikes, t, nperseg, noverlap, task=None, method='binning', **kwargs):
    """
    Extract spike rates from spike train data.

    Parameters
    ----------
    spikes : ndarray
        The spike event times.
    t : ndarray
        The time at which the firing rate is estimated.
    nperseg : int
        The length of each segment.
    noverlap: int
        The number of points to overlap between segments.
    task : ndarray
        The task data associated with spike data. Default None.
    method : str
        The method to extract spike rate: 'binning', 'gaussian', 'boxcar', or 'baks'.
    **kwargs : str, float
        The arguments that will be passed when calling spike rate estimation function.
    
    Returns
    -------
    spike_rate : ndarray
        The estimated spike rate.
    """
    X_rate = []
    if task is not None:
        y_task = []
        # check length X_in equals to y_in
        assert len(t) == len(task), "Both data length should be equal"
    for i in range(len(t)):
        start_idx = i * (nperseg - noverlap)
        end_idx = start_idx + nperseg
        if end_idx > len(t)-1:
            break # break if index exceeds the data length
        idx = np.where((spikes >= t[start_idx]) & (spikes < t[end_idx]))[0] 
        spikes_seg = spikes[idx] # spikes per segment
        t_seg = t[start_idx:end_idx] # time per segment
        n_spikes = len(spikes_seg)
        if n_spikes == 0:
            spike_rate = np.zeros(len(t_seg))
        else:
            if method == 'binning':
                spike_rate = np.full(len(t_seg), n_spikes)
            elif method == 'boxcar':
                spike_rate = kernel_smoother(spikes_seg, t_seg, nperseg, **kwargs)
            elif method == 'gaussian':
                spike_rate = kernel_smoother(spikes_seg, t_seg, nperseg, **kwargs)
            elif method == 'baks':
                spike_rate, _ = baks(spikes_seg, t_seg, b=n_spikes**0.8, **kwargs)
            else:
                print(f"{method} method is not supported. Select other method!")
        
        X_rate.append(spike_rate[nperseg//2])
        if task is not None:
            y_task.append(task[end_idx])
    if task is not None:
        return np.asarray(X_rate), np.asarray(y_task)
    else:
        return np.asarray(X_rate)