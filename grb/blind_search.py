import numpy as np
from .config import DATA_PATH, logging
from .utils import Chi2_polyval
import matplotlib.pyplot as plt
from .light_curves import exclude_time_interval, LightCurve

def filter_missing_data_and_flares(times, signal):
    '''
    Load gaps in ligth curve and exclude from data
    '''
    # load hand-filtered ligth curve and extract gaps
    breaks_data = np.loadtxt(f'{DATA_PATH}spi-acs_filtered.txt')
    breaks = []
    avg_gap = np.median(breaks_data[:,1])
    for i in range(breaks_data.shape[0]-1):
        if (breaks_data[i + 1, 0] - breaks_data[i,0]) > 3 * avg_gap:
            breaks.append([breaks_data[i, 0],breaks_data[i + 1, 0]])

    # iterate through data and mark gaps with zeros
    for gap in breaks:
        signal[(times >= gap[0]) & (times <= gap[1])] = 0
    
    return times[signal != 0], signal[signal != 0]

def process_time_window(sub_time,sub_counts,center_bin,threshold, plot = False):
    '''
    For given time window (sub_time,sub_counts) find event in center_bin
    '''
    bkg_time = np.delete(sub_time,[center_bin])
    bkg_counts = np.delete(sub_counts,[center_bin])
    param = np.polyfit(bkg_time,bkg_counts,3)
    event_flux = sub_counts[center_bin] - np.polyval(param,sub_time[center_bin])
    overpuasson = np.var(bkg_counts)/np.mean(bkg_counts)
    sigma = np.sqrt(np.var(bkg_counts))
    if (event_flux/sigma) > threshold:
        event_time = sub_time[center_bin]
        chi_2 = Chi2_polyval(bkg_time,bkg_counts,param)
        logging.debug(f'{event_time=}, sigma = {round(event_flux/sigma,1)}, chi2 bkg = {round(chi_2,1)}')
        if plot:
            fig=plt.figure()
            plt.errorbar(bkg_time,bkg_counts,yerr=np.sqrt(bkg_counts*overpuasson),fmt='o',label='background')
            plt.errorbar(sub_time[center_bin],sub_counts[center_bin],yerr=sigma,fmt='o',c='red',label=f'event, {round(event_flux/sigma,1)} sigma')
            plt.plot(sub_time,np.polyval(param,sub_time),label=f'chi2 bkg={round(chi_2,1)}')
            plt.legend()
            plt.show()
        return event_time,round(event_flux/sigma,1),round(chi_2,1)
    else:
        return None,None,None

def find_event(times, times_err, signal, signal_err, 
               event_times: tuple = None, 
               bkg_polynom_degree: int = 3):
    logging.debug(f'Finding event for {event_times=}')
    if event_times is None:
        param = np.polyfit(times, signal, bkg_polynom_degree)
    else:
        param = np.polyfit(*exclude_time_interval(times, signal, event_times), bkg_polynom_degree)

    significances=((signal - np.polyval(param,times)) / signal_err)
    logging.debug(f'{significances=}')
    logging.debug(f'{times=}')
    logging.debug(f'{significances[(times >= times[0]/2)&(times <= times[-1]/2)]}')
    if event_times is None:
        peak = np.argmax(significances == np.max(significances[(times >= times[0]/2)&(times <= times[-1]/2)]))
    else:
        peak = np.argmax(significances == np.max(significances[(times >= event_times[0])&(times <= event_times[1])]))
        
    left_idx = right_idx = peak
    logging.debug(f'{peak=}')
    if significances[peak] < 1:
        return event_times[0], event_times[1]

    while significances[left_idx - 1] > 1:
        left_idx -= 1
        if left_idx == 0:
            break
    try:
        while significances[right_idx + 1] > 1 and right_idx < significances.shape[0]-1:
            right_idx += 1
    except IndexError:
        pass

    logging.debug(f'Found smth {times[left_idx] - times_err[left_idx]} {times[right_idx] + times_err[right_idx]}')
    return times[left_idx] - times_err[left_idx], times[right_idx] + times_err[right_idx]

def recursive_event_search(lc: LightCurve,
                           current_resolution: float,
                           event_times: tuple = None,
                           bkg_polynom_degree: int = 3,
                           stoping_resolution: float = 2,
                           stoping_size: int = 10,
                           filter_values: float = -10):
    logging.info(f'Stranted search in {lc=} in {current_resolution=}, previous results = {event_times=}')
    lc.rebin(current_resolution).filter_peaks(filter_values)
    times, times_err, signal, signal_err = lc.times, lc.times_err, lc.signal, lc.signal_err

    # Run 3 times to ensure good background substration
    if event_times is None:
        times_left, times_right = find_event(times, times_err, signal, signal_err)
    else:
        times_left, times_right = find_event(times, times_err, signal, signal_err,event_times)
    times_left, times_right = find_event(times, times_err, signal, signal_err, (times_left, times_right))
    times_left, times_right = find_event(times, times_err, signal, signal_err, (times_left, times_right))

    logging.info(f'Found event in {times_left=} {times_right=}')
    new_resolution = current_resolution / 2
    if new_resolution < lc.original_resolution:
        return times_left, times_right, current_resolution
    if np.sum((times > times_left)&(times < times_right)) > stoping_size or new_resolution < stoping_resolution:
        return times_left, times_right, current_resolution
    else:
        return recursive_event_search(lc,new_resolution,(times_left, times_right),bkg_polynom_degree,stoping_resolution,stoping_size,filter_values)