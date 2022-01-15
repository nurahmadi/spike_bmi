"""
List of preprocessing functions
"""

# import packages
import numpy as np
from scipy import signal
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import _num_samples

def filter(x, fc, fs, order, btype="lowpass", zero_phase=True):
    """
    Filter data with a Butterworth filter.

    Parameters
    ----------
    x : ndarray
        The data to be filtered.
    fc : float or list
        The critical frequency.
    fs : float
        The sampling frequency.
    order : int
        The order of the filter.
    btype : {'lowpass', 'highpass', 'bandpass', 'bandstop'}, default 'lowpass'
        The type of filter.
    zero_phase : bool, default True.
        Zero phase (forward backward) filter.

    Returns
    ----------
    y : ndarray
        The filtered data.
    """
    fnyq = fs/2 # Nyquist frequency
    if btype in ['bandpass', 'bandstop']:
        assert len(fc)==2, f"for {btype}, you should provide a sequence of two frequencies (low and high)"
        Wn = [f/fnyq for f in fc]
    else:
        Wn = fc/fnyq
    
    b, a = signal.butter(order, Wn, btype=btype)
    if zero_phase:
        y = signal.filtfilt(b, a, x)
    else:
        y = signal.lfilter(b, a, x)
    return y
    
def downsample(x, k):
    """
    Downsample data.

    Parameters
    ----------
    x : ndarray
        The data to be downsampled.
    k : int
        The downsample factor.

    Returns
    ----------
    y : ndarray
        The downsampled data.

    """
    N = len(x)
    idx = np.arange(0, N, k)
    y = x[idx]
    return y

class TimeSeriesSplitCustom(TimeSeriesSplit):
    """
    Create time-series data cross-validation
    Ref: https://stackoverflow.com/questions/62210221/walk-forward-with-validation-window-for-time-series-data-cross-validation

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. 
    max_train_size : int, default=None
        Maximum size for a single training set.
    test_size : int, default=1
        Used to limit the size of the test set.
    min_train_size : int, default=1
        Minimum size of the training set.

    Returns
    ----------
    Indices of training and testing data.
    """
    def __init__(self, n_splits=5, max_train_size=None, test_size=1, min_train_size=1):
        super().__init__(n_splits=n_splits, max_train_size=max_train_size)
        self.test_size = test_size
        self.min_train_size = min_train_size

    def overlapping_split(self, X, y=None, groups=None):
        min_train_size = self.min_train_size
        test_size = self.test_size

        n_splits = self.n_splits
        n_samples = _num_samples(X)

        if (n_samples - min_train_size) / test_size >= n_splits:
            print('(n_samples -  min_train_size) / test_size >= n_splits')
            print('default TimeSeriesSplit.split() used')
            yield from super().split(X)

        else:
            shift = int(np.floor((n_samples - test_size - min_train_size) / (n_splits - 1)))
            start_test = n_samples - (n_splits * shift + test_size - shift)
            test_starts = range(start_test, n_samples - test_size + 1, shift)

            if start_test < min_train_size:
                raise ValueError(("The start of the testing : {0} is smaller"
                                  " than the minimum training samples: {1}.").format(start_test, min_train_size))

            indices = np.arange(n_samples)

            for test_start in test_starts:
                if self.max_train_size and self.max_train_size < test_start:
                    yield (indices[test_start - self.max_train_size:test_start],
                           indices[test_start:test_start + test_size])
                else:
                    yield (indices[:test_start],
                           indices[test_start:test_start + test_size])

def transform_data(X, y, timesteps):
    """
    Transform data into sequence data with timesteps

    Parameters
    ----------
    X : ndarray
        The nput data 
    y : ndarray
        The utput (target) data
    timesteps: int
        The umber of input steps to predict next step

    Returns
    ----------
    X_seq : ndarray
        The transformed input sequence data
    y_seq : ndarray
        The transformed ouput (target) sequence data
    """
    X_seq = []
    y_seq = []
    # check length X_in equals to y_in
    assert len(X) == len(y), "Both input data length must be equal"
    for i in range(len(X)):
        end_idx = i + timesteps
        if end_idx > len(X)-1:
            break # break if index exceeds the data length
        # get input and output sequence
        X_seq.append(X[i:end_idx,:])
        y_seq.append(y[end_idx-1,:])
    return np.asarray(X_seq), np.asarray(y_seq)