import numpy as np

def rebin_data(times,
               signal,
               resolution,
               bin_duration: float = None, 
               binning: np.array = None):

    '''
    Auxiliary method for rebining the light curve
    '''
    if binning is None:
        N_new_bins = int(np.floor((times[-1] - times[0] + resolution)/bin_duration)+1)
        binning = times[0] - resolution/2 + np.linspace(0, N_new_bins*bin_duration, num = N_new_bins+1)
        
    new_times = binning[:-1] + bin_duration/2
    # fill the time error array
    new_times_err = np.ones_like(new_times)*bin_duration/2
    # determine the number of counts in each bin
    new_counts = np.histogram(times, bins=binning)[0]
    # determine the signal in each bin
    new_signal = np.histogram(times, bins=binning, weights=signal)[0]
    # determine the signal error in each bin
    new_signal_err = np.sqrt(np.histogram(times, bins=binning, weights=signal**2)[0])
    # determine the signal error in each bin
    new_signal_err = new_signal_err/np.sqrt(new_counts)
    # return the binned data
    return new_times[:-1], new_times_err[:-1], new_signal[:-1], new_signal_err[:-1], binning

def exclude_time_interval(times: np.array, signal: np.array, intervals):
    if type(intervals[0]) == tuple:
        mask = None
        for interval in intervals:
            if mask is None:
                mask = (times < interval[0]) | (times > interval[1])
            else:
                mask = mask & ((times < interval[0]) | (times > interval[1]))
    else:
        mask = (times < intervals[0]) | (times > intervals[1])

    return times[mask], signal[mask]

def limit_to_time_interval(times: np.array, signal: np.array, intervals):
    if type(intervals[0]) == tuple:
        mask = None
        for interval in intervals:
            if mask is None:
                mask = (times >= interval[0]) & (times <= interval[1])
            else:
                mask = mask | ((times >= interval[0]) & (times <= interval[1]))
    else:
        mask = (times >= intervals[0]) & (times <= intervals[1])

    return times[mask], signal[mask]

def get_integral_curve(signal: np.array, times: np.array = None, params: np.array = None):
    '''
    Calculates the integral of a signal over time.

    Args:
        signal (np.array): A numpy array containing the signal to integrate.
        times (np.array, optional): A numpy array containing the time values corresponding to the signal. Defaults to None.
        params (np.array, optional): A numpy array containing the polynomial coefficients to subtract from the signal. Defaults to None.
    '''
    if times is None:
        return np.cumsum(signal)
    else:
        return np.cumsum(signal - np.polyval(params, times))