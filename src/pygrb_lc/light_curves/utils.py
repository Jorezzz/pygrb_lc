import numpy as np
import matplotlib.pyplot as plt

from .light_curves import GBMLightCurve
from ..utils import get_first_intersection, is_iterable
from ..config import GBM_DETECTORS

def plot_gbm_all_detectors(center_time: str, duration: float, binning: float = 0.5, axs = None, **kwargs):
    detector_list = [x for x in GBM_DETECTORS if x[0] == 'n']
    data = {}

    if axs is None:
        _, ax = plt.subplots(4,3,figsize=(30,30))
        ax = ax.reshape(-1,)
    else:
        ax = axs
        
    for i, detector in enumerate(detector_list):
        lc = GBMLightCurve(center_time, [detector], duration = duration, **kwargs)
        lc.rebin(binning).plot(ax=ax[i], label = detector)
        signal = lc.signal
        ax[i].axhline(np.mean(signal),color = 'blue')
        ax[i].axhline(np.mean(signal)+3*np.std(signal),color = 'green')
        ax[i].axhline(np.mean(signal)-3*np.std(signal),color = 'green')
        ax[i].legend()
        data[detector] = lc
    plt.suptitle(center_time)
    return ax, data

def calculate_t_90(times: np.array, intergal_curve: np.array, left_interval, right_interval, plot: str = None):
    '''
    Calculates T90 using integral curve according to the method described in the work Koshut et al. 1996.
    Args:
        times (np.array): array of times
        intergal_curve (np.array): array of itegrated counts
        left_interval (float/int/Iterable): interval of calculating low counts level. 
                                            If number - uses all interval up to this number, 
                                            othewise uses provided interval
        right_interval (float/int/Iterable): interval of calculating high counts level. 
                                             If number - uses all interval from this number, 
                                             othewise uses provided interval
        plot (str): plotting options, if None - without plot,
                                      'full' - all intervals and intersections, 
                                      'main' - only levels responsible for duration
    '''
    left_interval = left_interval if is_iterable(left_interval) else (times[0],left_interval)
    right_interval = right_interval if is_iterable(right_interval) else (right_interval,times[-1])

    mask = (times >= left_interval[0]) & (times <= left_interval[1])
    level_low = np.polyfit(times[mask],intergal_curve[mask],0)[0]
    d_low = np.sqrt(np.var(intergal_curve[mask] - level_low))

    mask = (times >= right_interval[0]) & (times <= right_interval[1])
    level_high = np.polyfit(times[mask],intergal_curve[mask],0)[0]
    d_high = np.sqrt(np.var(intergal_curve[mask] - level_high))

    delta_low = np.sqrt((0.95 * d_low)**2 + (0.05 * d_high)**2)
    delta_high = np.sqrt((0.05 * d_low)**2 + (0.95 * d_high)**2)

    level_5 = level_low + (level_high - level_low)*0.05
    level_95 = level_low + (level_high - level_low)*0.95

    t_5 = get_first_intersection(times, intergal_curve, level_5)
    t_95 = get_first_intersection(times, intergal_curve, level_95)

    t_5_low = get_first_intersection(times, intergal_curve, level_5 - delta_low)
    t_5_high = get_first_intersection(times, intergal_curve, level_5 + delta_low)
    t_95_low = get_first_intersection(times, intergal_curve, level_95 - delta_high)
    t_95_high = get_first_intersection(times, intergal_curve, level_95 + delta_high)

    t_90 = (t_95 - t_5)

    positive_err = np.sqrt((t_95_high - t_95)**2 + (t_5-  t_5_low)**2)
    negative_err = np.sqrt((t_95 - t_95_low)**2 + (t_5_high - t_5)**2)
    if plot is not None:
        plt.step(times, intergal_curve, where='mid')

        plt.axhline(level_5,color='orange')
        plt.axhline(level_95,color='orange')

        plt.axvline(t_5,color='orange')
        plt.axvline(t_95,color='orange')

    if plot == 'full':
        plt.axhline(level_low)
        plt.axhline(level_high)

        plt.axhline(level_5-delta_low)
        plt.axhline(level_5+delta_low)
        plt.axhline(level_95-delta_high)
        plt.axhline(level_95+delta_high)
        plt.axvline(t_5_low)
        plt.axvline(t_5_high)
        plt.axvline(t_95_low)
        plt.axvline(t_95_high)

    return t_90, (-negative_err, positive_err)
