import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from .light_curves import LightCurve
from scipy.stats import chi2
from collections.abc import Callable, Iterable


def make_pds(signal, time_step, total_counts = None, pad_size = None):
    '''
    Get Power Density Spectrum from signal
    Args:
        signal (np.array): input signal
        time_step (float): time step
        total_counts (int): total counts in signal, if None uses np.sum(signal)
        pad_size (int): number of bins to pad, if None then doesn't pad
    '''
    mean = np.mean(signal)
    if pad_size is not None:
        signal = np.pad(signal, (pad_size - signal.shape[0], 0), 'constant')
        
    if total_counts is None:
        total_counts = np.sum(signal)

    freqs = np.fft.fftfreq(signal.shape[0], time_step)
    ps = 2*np.abs(np.fft.fft(signal - mean))**(2)/total_counts
    mask = (freqs>0)
    
    return freqs[mask], ps[mask]

def group_log_bins(freqs: np.array, ps: np.array, N_bins: int = 30, step: float = None, log_scale: np.array = None):
    '''
    Group bins in log scale
    Args:
        freqs (np.array): input frquencies
        ps (np.array): input power spectrum
        N_bins (int, optional): number of bins, defaults to 30.
    '''
    time=[]
    time_err=[]
    flux=[]
    flux_err=[]
    log_x = np.log10(freqs)

    if step is None:
        step = (log_x[-1] - log_x[0])/(2 * N_bins)
    if log_scale is None:
        log_scale = np.linspace(log_x[0] + step, log_x[-1] - step,N_bins)

    for i in range(0, N_bins):
        mask1=tuple([(log_x >= log_scale[i] - step) & (log_x < log_scale[i] + step)])
        time.append(np.mean(freqs[mask1]) if len(ps[mask1]) != 0 else 10**log_scale[i])
        time_err.append((10**(log_scale[i] + step) - 10**(log_scale[i]) + 10**(log_scale[i]) - 10**(log_scale[i] - step))/2)
        flux.append(np.mean(ps[mask1]) if len(ps[mask1])!=0 else 0)
        flux_err.append(chi2.ppf(0.67, 2*len(ps[mask1]))/len(ps[mask1]) if len(ps[mask1]) != 0 else 1)
    
    return np.array(time), np.array(time_err), np.array(flux), np.array(flux_err)

class FurieLightCurve():
    def __init__(self, light_curve: LightCurve, 
                       interval_t90: Iterable = None,
                       bkg_substraction_resolution: float = 10,
                       bkg_polynom_degree: int = 3,
                       bkg_intervals: Iterable = None,
                       pad_size: int = None,
                       window: Callable = None
                       ):

        '''
        Args:
            light_curve (LightCurve): light curve object
            interval_t90 (tuple, optional): time interval for t90. If None, uses 2 and 3 quartile of time
            bkg_substraction_resolution (float, optional): background substraction resolution, defaults to 10
            bkg_polynom_degree (int, optional): background polynom degree, defaults to 3
            pad_size (int, optional): number of bins to pad, if None then doesn't pad
            window (function, optional): window function, that applies to signal, defaults to None
        '''
        self.light_curve = light_curve

        if interval_t90 is None:
            interval_t90 = (self.light_curve.times[0]/2,self.light_curve.times[-1]/2)
        
        if bkg_intervals is None:
            bkg_intervals = [(self.light_curve.times[0]-self.light_curve.resolution,interval_t90[0]),
                         (interval_t90[1],self.light_curve.times[-1]+self.light_curve.resolution)]
        else:
            bkg_intervals = bkg_intervals
        
        self.bkg_intervals = bkg_intervals
        self.interval_t90 = interval_t90

        rebined_param = np.polyfit(self.light_curve.rebin(bkg_substraction_resolution).set_intervals(*bkg_intervals).times,
                                   self.light_curve.rebin(bkg_substraction_resolution).set_intervals(*bkg_intervals).signal,
                                   bkg_polynom_degree)
        rebined_param = rebined_param * (self.light_curve.original_resolution/self.light_curve.resolution)
        self.rebined_param = rebined_param
        self.N = np.sum(self.light_curve.rebin().set_intervals(*bkg_intervals).signal)

        signal = self.light_curve.rebin().substract_polynom(rebined_param).set_intervals(interval_t90).signal
        

        if window is not None:
            signal = signal * window(signal.shape[0])

        self.freqs, self.ps =  make_pds(signal, self.light_curve.original_resolution, self.N, pad_size)
        self.freqs_err, self.ps_err = np.full(self.freqs.shape[0], 0), np.sqrt(self.ps)

    def plot(self, kind: str = 'scatter', subtract_poisson = False, logx: bool = True, logy: bool = True, N_bins: int = 30, ax: mpl.axes.Axes = None, **kwargs):
        '''
        Plot PDS
        Args:
            kind (str, optional): plotting method
            logx (bool, optional): log x axis, defaults to True
            logy (bool, optional): log y axis, defaults to True
            N_bins (int, optional): number of bins, if None - plots without grouping
            ax (mpl.axes.Axes, optional): axes object to plot on
            kwargs (dict, optional): keyword arguments for plotting used by matplotlib.pyplot.plot
        '''
        if N_bins is None:
            x,x_err,y,y_err = self.freqs, self.freqs_err, self.ps, self.ps_err
        else:
            x,x_err,y,y_err = group_log_bins(self.freqs, self.ps, N_bins)

        if subtract_poisson:
            y = y - 2
            
        if ax is None:
            if kind == 'plot':
                plt.plot(x, y, **kwargs)
            elif kind == 'errorbar':
                plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt = 'o', **kwargs)
            elif kind == 'scatter':
                plt.scatter(x, y, **kwargs)
            elif kind == 'step':
                plt.step(x, y, **kwargs)
            else:
                raise NotImplementedError(f"Plotting method '{kind}' not supported")
            
            if logx:
                plt.xscale('log')
            if logy:
                plt.yscale('log')
        else:
            if kind == 'plot':
                ax.plot(x, y, **kwargs)
            elif kind == 'errorbar':
                ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt = 'o', **kwargs)
            elif kind == 'scatter':
                ax.scatter(x, y, **kwargs)
            elif kind == 'step':
                ax.step(x, y, **kwargs)
            else:
                raise NotImplementedError(f"Plotting method '{kind}' not supported")
            
            if logx:
                ax.set_xscale('log')
            if logy:
                ax.set_yscale('log')