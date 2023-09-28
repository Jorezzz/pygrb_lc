import numpy as np

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