import ftplib
import pickle
import datetime
import logging

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import pandas as pd

from .transformations import rebin_data
from ..config import LIGHT_CURVE_SAVE
from ..time import change_fermi_seconds, change_utc
from ..utils import retry, parse_html_table, get_overlaping_intersection


logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


class LightCurve():
    '''
    Base class for light curves from different sources
    '''
    def __init__(self, event_time: str = None, duration: float = None, data: np.array = None):
        '''
        Args:
            event_time (str, optional): time of the event in format 'YYYY-MM-DD HH:MM:SS'
            duration (int, optional): duration in seconds
            data (np.array, optional): data of the light curve, having 2 or 4 columns (time, time_err, signal, signal_err) in seconds.
        '''
        self.event_time = event_time
        self.duration = duration

        if data is not None:
            self.__get_light_curve_from_data(data)
        else:
            self.times = None
            self.times_err = None
            self.signal = None
            self.signal_err = None
            self.resolution = None

            self.original_times,self.original_signal = None, None
            self.original_resolution = None

    def __repr__(self):
        return f'{self.__class__.__name__}(event_time={self.event_time}, duration={self.duration}, original resolution={self.original_resolution})'

    def __eq__(self, other: object):
        return self.__class__ == other.__class__ and self.event_time == other.event_time and self.duration == other.duration and self.original_resolution == other.original_resolution

    def plot(self, kind: str = 'plot',
             logx: bool = None,
             logy: bool = None, 
             ax: mpl.axes.Axes = None, 
             **kwargs):
        '''
        Plot the light curve
        Args:
            kind (str, optional): plotting method
            logx (bool, optional): log x axis
            logy (bool, optional): log y axis
            kwargs (dict, optional): keyword arguments for plotting used by matplotlib.pyplot.plot
        '''
        if ax is None:
            if kind == 'plot':
                plt.plot(self.times, self.signal, **kwargs)
            elif kind == 'errorbar':
                plt.errorbar(self.times, self.signal, xerr=self.times_err, yerr=self.signal_err, fmt = 'o', **kwargs)
            elif kind == 'scatter':
                plt.scatter(self.times, self.signal, **kwargs)
            elif kind == 'step':
                plt.step(self.times, self.signal, **kwargs)
            else:
                raise NotImplementedError(f'Plotting method {kind} not supported')

            if logx:
                plt.xscale('log')
            if logy:
                plt.yscale('log')
        else:
            if kind == 'plot':
                ax.plot(self.times, self.signal, **kwargs)
            elif kind == 'errorbar':
                ax.errorbar(self.times, self.signal, xerr=self.times_err, yerr=self.signal_err, fmt = 'o', **kwargs)
            elif kind == 'scatter':
                ax.scatter(self.times, self.signal, **kwargs)
            elif kind == 'step':
                ax.step(self.times, self.signal, **kwargs)
            else:
                raise NotImplementedError(f'Plotting method {kind} not supported')

            if logx:
                ax.set_xscale('log')
            if logy:
                ax.set_yscale('log')

    def rebin(self, bin_duration: float = None, reset: bool = True):
        '''
        Rebin light curve from original time resolution
        Args:
            bin_duration (float): new bin duration in seconds, if None, return to original resolution
            reset (bool, optional): if True, rebin the light curve using original resolution, result is same as self.rebin().rebin(bin_duration)
        Returns:
            self
        '''
        if bin_duration is None:
            self._reset_light_curve()

            return self

        if reset:
            bined_times, bined_times_err, bined_signal, bined_signal_err, _ = rebin_data(self.original_times, self.original_signal, self.original_resolution, bin_duration)
        else:
            bined_times, bined_times_err, bined_signal, bined_signal_err, _ = rebin_data(self.times, self.signal, self.resolution, bin_duration)
        
        self.signal = bined_signal
        self.signal_err = bined_signal_err
        self.times = bined_times
        self.times_err = bined_times_err
        self.resolution = bin_duration

        return self

    def set_intervals(self,*intervals):
        """
        Returns light curve in intervals
        Args:
            *intervals: tuples (start_time,end_time),(start_time,end_time),...
        Returns:
            self
        """
        if len(intervals)==0:
            return self.times,self.times_err,self.signal,self.signal_err
        else:
            temp_time=[]
            temp_time_err=[]
            temp_signal=[]
            temp_signal_err=[]
            for interval in intervals:
                for i,time in enumerate(self.times):
                    if interval[0] < time < interval[1]:
                        temp_time.append(time)
                        temp_time_err.append(self.times_err[i])
                        temp_signal.append(self.signal[i])
                        temp_signal_err.append(self.signal_err[i])

            self.times= np.asarray(temp_time)
            self.times_err = np.asarray(temp_time_err)
            self.signal = np.asarray(temp_signal)
            self.signal_err = np.asarray(temp_signal_err)

            return self
        
    def substract_polynom(self,polynom_params: np.array):
        '''
        Substract polynom from light curve
        Args:
            polynom_params (np.array): parameters for polynom to be subtracted, from np.polyfit
        Returns:
            self
        '''
        self.signal = self.signal - np.polyval(polynom_params,self.times)
        return self

    def filter_peaks(self,peak_threshold: float = -3):
        '''
        Filter light curve to remove peaks
        Args:
            peak_threshold (float, optional): threshold for peak removal, if negative - remove points bellow median, if positive - remove points above median, if zero - nothing happens
        Returns:
            self
        '''
        median_flux = np.median(self.signal)
        indexes_to_remove = []

        if peak_threshold < 0:
            indexes_to_remove = (self.signal - median_flux)/np.where(np.nan_to_num(self.signal_err,nan=1)==0,1,np.nan_to_num(self.signal_err,nan=1)) < peak_threshold
        elif peak_threshold > 0:
            indexes_to_remove = (self.signal - median_flux)/np.where(np.nan_to_num(self.signal_err,nan=1)==0,1,np.nan_to_num(self.signal_err,nan=1)) >= peak_threshold

        self.signal = np.delete(self.signal, indexes_to_remove)
        self.signal_err = np.delete(self.signal_err, indexes_to_remove)
        self.times = np.delete(self.times, indexes_to_remove)
        self.times_err = np.delete(self.times_err, indexes_to_remove)
        
        return self

    def _reset_light_curve(self):
        '''
        Reset light curve to original state
        '''
        self.times = self.original_times
        self.times_err = np.full(self.original_times.shape[0], self.original_resolution)
        self.signal = self.original_signal
        self.signal_err = np.sqrt(self.original_signal)
        self.resolution = self.original_resolution

    def __get_light_curve_from_data(self, data):
        '''
        Appends data from data to light curve object
        Args:
            data (np.array): data array with shape (n_samples,n_channels), where n_channels is 2 or 4
        '''
        if data.shape[1] == 2:
            self.original_times = data[:,0]
            self.original_signal = data[:,1]
            
            self.original_resolution = round(np.median(self.original_times[1:] - self.original_times[:-1]),3) # determine size of time window
            self._reset_light_curve()
        elif data.shape[1] == 4:
            self.original_times = data[:,0]
            self.original_signal = data[:,2]
            
            self.times = self.original_times
            self.original_resolution = round(np.median(self.times[1:] - self.times[:-1]),3) # determine size of time window
            self.times_err = data[:,1]
            self.signal = self.original_signal
            self.signal_err = data[:,3]
            self.resolution = self.original_resolution

    @classmethod
    def load(cls, 
             filename: str = None, 
             event_time: str = None, 
             duration: int = None, 
             resolution: float = None):
        '''
        Load light curve from file, either filename or event_time, duration and resolution should be provided
        Args:
            filename (str, optional): filename where to load data from, 
        '''
        if (not filename) or (not (event_time and duration and resolution)):
            raise ValueError("Either filename or lc parameters should be provided")
        
        filename = filename if filename else f'{event_time.replace(":","_")}_{int(duration)}_{resolution}.txt'
        data = np.loadtxt(f'{LIGHT_CURVE_SAVE}{filename}')
        return cls(data = data)

    def save(self, filename: str = None):
        '''
        Save light curve to file
        Args:
            filename (str, optional): file name without extension
        '''
        filename = filename if filename else f'{self.event_time.replace(":","_")}_{int(self.duration)}_{self.original_resolution}.txt'
        np.savetxt(f'{LIGHT_CURVE_SAVE}{filename}',
                   np.hstack((self.original_times.reshape(-1, 1), 
                              self.original_signal.reshape(-1, 1))))



class SPI_ACSLightCurve(LightCurve):
    '''
    Class for light curves from SPI-ACS/INTEGRAL
    '''
    def __init__(self, 
                 event_time: str,
                 duration: int, 
                 loading_method: str = 'web',
                 scale: str = 'utc',
                 acs_path = 'E:/ACS/',
                 **kwargs):
        '''
        Args:
            event_time (str): date and time of event in ISO 8601 format
            duration (int): duration of light curve in seconds
            loading_method (str, optional): 'local' or 'web'
            scale (str, optional): 'utc' or 'ijd'
        '''
        super().__init__(event_time, duration, **kwargs)
        self.event_time = self.event_time[:10] + ' ' + self.event_time[11:19]
        
        if self.original_times is None:
            if loading_method == 'local':
                self.original_times,self.original_signal = self.__get_light_curve_from_file(scale = scale, acs_path = acs_path)
            elif loading_method =='web':
                self.original_times,self.original_signal = self.__get_light_curve_from_web(scale = scale)
            else:
                raise NotImplementedError(f'Loading method {loading_method} not supported')

            self.original_resolution = round(np.mean(self.original_times[1:] - self.original_times[:-1]), 3) # determine size of time window
            self._reset_light_curve()

    def __get_light_curve_from_file(self,scale = 'utc', acs_path = 'E:/ACS/'):
        '''
        Load a light curve from raw scw files
        Args:
            scale (str, optional): scale of light curve, 'utc' (seconds from trigger) or 'ijd' (days from J2000)
        '''
        with open(f'{acs_path}swg_infoc.dat','r') as f:
            acs_scw_df = [line.split() for line in f]

        acs_scw_df = pd.DataFrame(acs_scw_df,columns=['scw_id','obt_start','obt_finish','ijd_start','ijd_finish','scw_duration','x','y','z','ra','dec'])
        acs_scw_df['scw_id'] = acs_scw_df['scw_id'].astype(str)
        acs_scw_df['ijd_start'] = acs_scw_df['ijd_start'].astype(float)
        acs_scw_df['ijd_finish'] = acs_scw_df['ijd_finish'].astype(float)

        center_time = datetime.datetime.strptime(self.event_time,'%Y-%m-%d %H:%M:%S')
        left_time = float(change_utc((center_time - datetime.timedelta(seconds=self.duration)).strftime('%Y-%m-%d %H:%M:%S'), 'ijd'))
        right_time = float(change_utc((center_time + datetime.timedelta(seconds=self.duration)).strftime('%Y-%m-%d %H:%M:%S'), 'ijd'))
        scw_needed = acs_scw_df[get_overlaping_intersection(left_time, right_time, acs_scw_df['ijd_start'], acs_scw_df['ijd_finish'])]
        if scw_needed.shape[0]==0:
            raise ValueError(f'No data found for {self.event_time}')

        current_data = []
        for _,row in scw_needed.iterrows():
            for i in range(5):
                try:
                    with open(f'{acs_path}0.05s/{row["scw_id"][:4]}/{row["scw_id"]}.00{i}.dat','r') as f:
                        for line in f:
                            line = line.split()
                            if float(line[0]) > 0:
                                current_data.append([row['ijd_start']+float(line[0])/(24*60*60),float(line[2])])
                    break
                except FileNotFoundError:
                    continue
            
        current_data = np.asarray(current_data)
        
        current_data[:,0] = (current_data[:,0] - (left_time+self.duration/(24*60*60)))*(24*60*60)
        current_data = current_data[np.abs(current_data[:,0])<self.duration]

        if scale == 'ijd':
            current_data[:,0] = current_data[:,0]/(24*60*60) + (left_time+self.duration/(24*60*60))

        return current_data[:,0],current_data[:,1]

    def __get_light_curve_from_web(self,scale = 'utc'):
        '''
        Download light curve from isdc (unavailable in Russia)
        Args:
            scale (str, optional): scale of light curve, 'utc' (seconds from trigger) or 'ijd' (days from J2000)
        '''
        url = f'https://www.isdc.unige.ch/~savchenk/spiacs-online/spiacs.pl?requeststring={self.event_time[0:10]}T{self.event_time[11:13]}%3A{self.event_time[14:16]}%3A{self.event_time[17:19]}+{self.duration}&generate=ipnlc&submit=Submit'
        data = []
        r = requests.get(url, timeout=30)
        for line in r.text.split('<br>\n')[2:-2]:
            try:
                data.append([float(line.split()[0]),int(float(line.split()[1]))])
            except:
                continue
        data = np.asarray(data)

        times=data[:,0]
        signal=data[:,1]
        gap=0.05
        temp_times=[]
        temp_signal=[]
        temp_times.append(times[0])
        temp_signal.append(signal[0] if np.isfinite(signal[0]) else 0)
        
        # Fill missing values
        counter=1
        while counter<len(times):
            if times[counter]>1.5*gap+temp_times[counter-1]:
                temp_times.append(temp_times[counter-1]+gap)
                temp_signal.append(0)
            else:
                temp_times.append(times[counter])
                temp_signal.append(signal[counter] if np.isfinite(signal[counter]) else 0)
                counter += 1

        temp_times = np.asarray(temp_times)
        temp_signal = np.asarray(temp_signal)
        if scale == 'ijd':
            event_time_ijd = change_utc(self.event_time, 'ijd')
            temp_times = temp_times/(24*60*60) + event_time_ijd

        return temp_times,temp_signal
    




class GBMLightCurve(LightCurve):
    '''
    Class for light curves from GBM/Fermi
    '''
    def __init__(self,code: str, 
                 lumined_detectors: list[str],
                 redshift: float = None, 
                 original_resolution: float = None,
                 loading_method: str='web', 
                 scale = 'utc',
                 apply_redshift: bool = True,
                 filter_energy: dict = None,
                 save_mid_process = False,
                 data = None,
                 **kwargs):
        '''
        Args:
            code (str): GBM code, starting with 'bn', for example 'bn220101215', same value is assigned to event_time
            lumined_detectors (list[str]): List of detectors that should be used in the light curve
            redshift (float, optional): Cosmological redshift of GRB, if known
            original_resolution (float, optional): Starting binning of GRB, default to 0.01 seconds
            loading_method (str, optional): Method of obtaining the light curve, can be 'web' or 'local'
            scale (str, optional): Scale of the light curve, can be 'utc' or 'ijd', default to 'utc'
            apply_redshift (bool, optional): Apply redshift to the light curve, default to True
            filter_energy (dict, optional): Apply energy filter to the light curve, dict with low and high energy shoud be provided, otherwise None
            save_mid_process (bool): whether to save fits files for futher usage, uses disk space
        '''
        if data is not None:
            self.photon_data = data
            self.original_times, _, self.original_signal, _, _ = rebin_data(data[:,0], np.full(data.shape[0],1), 0, original_resolution, None)
            super().__init__(code, **kwargs)
        else:
            super().__init__(code, data = data, **kwargs)
            
        self.code = code
        self.redshift = redshift if redshift else 0
        self.photon_data = None
        self.original_resolution = original_resolution if original_resolution else 0.01

        if self.original_times is None:
            if loading_method == 'web':
                if self.code[:2] == 'bn':
                    self.original_times,self.original_signal = self.__get_light_curve_from_web(lumined_detectors, 
                                                                                               apply_redshift, 
                                                                                               filter_energy, 
                                                                                               scale = scale,
                                                                                               save = save_mid_process)
                else:
                    if self.duration is None:
                        raise ValueError('Duration of GRB is not provided, please provide it in the constructor')
                    
                    self.original_times,self.original_signal = self.__get_light_curve_from_web(lumined_detectors, 
                                                                                               apply_redshift, 
                                                                                               filter_energy, 
                                                                                               scale = scale,
                                                                                               save = save_mid_process,
                                                                                               load_daily = True)
            else:
                NotImplementedError(f'Loading method {loading_method} not implemented')

            self._reset_light_curve()
            
    def __get_light_curve_from_web(self, 
                                   lumined_detectors: list[str], 
                                   apply_redshift: bool, 
                                   filter_energy: bool, 
                                   scale = 'utc', 
                                   load_daily: bool = False,
                                   save = True):
        '''
        Binds the light curve from individual photons in lumined detectors
        Args:
            lumined_detectors (str): list of GBM detectors e.g. ['n0','n1','n2','n3','n4','n5','n6','n7']
            apply_redshift (bool, optional): Whether to apply the redshift to the light curve
            filter_energy (bool, optional): Whether to filter the light curve by energy
            scale (str, optional): scale of light curve, 'utc' (seconds from trigger) or 'ijd' (days from J2000)
        '''
        binning = None
        self.photon_data = None
        times_array = []
        signal_array = []

        if load_daily:
            logging.debug('Loading daily data')
            tzero = change_utc(self.event_time, 'fermi_seconds')
            left = pd.to_datetime(change_fermi_seconds(tzero - self.duration, 'utc')).to_pydatetime()
            right = pd.to_datetime(change_fermi_seconds(tzero + self.duration, 'utc')).to_pydatetime()

            left_new = datetime.datetime(left.year, left.month, left.day, left.hour)
            right_new = datetime.datetime(right.year, right.month, right.day, right.hour)

        for detector in lumined_detectors:
            logging.debug(f'Loading data from {detector}')
            if load_daily:
                data = None
                for date in pd.date_range(left_new, right_new, freq='1H'):
                    url = f'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/daily/{str(date.year).zfill(2)}/{str(date.month).zfill(2)}/{str(date.day).zfill(2)}/current/glg_tte_{detector}_{date.to_pydatetime().strftime("%y%m%d")}_{str(date.hour).zfill(2)}z'
                    hdul = self.load_fits(url, file_extension = 'fit.gz', save = save)
                    ebounds = {line[0]:np.sqrt(line[1]*line[2]) for line in hdul[1].data}
                    day_data = np.array(hdul[2].data.tolist())
                    day_data[:, 1] = [ebounds[x] for x in day_data[:, 1]]
                    left_bound, right_bound = change_utc(str(date),'fermi_seconds'), change_utc(str(date + datetime.timedelta(hours=1)),'fermi_seconds')
                    day_data = day_data[(day_data[:, 0] >= left_bound)&(day_data[:, 0] <= right_bound)]
                    data = np.concatenate((data, day_data)) if data is not None else day_data
                data[:, 0] = data[:, 0] - tzero
                data = data[np.where((data[:, 0] > -self.duration) & (data[:, 0] < self.duration))]
            else:
                url = f'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/20{self.code[2]}{self.code[3]}/{self.code}/current/glg_tte_{detector}_{self.code}'
                hdul = self.load_fits(url, file_extension = 'fit', save = save)
                ebounds = {line[0]:np.sqrt(line[1]*line[2]) for line in hdul[1].data}
                tzero = float(hdul[2].header['TZERO1'])
                data = np.array(hdul[2].data.tolist())
                data[:, 0] = data[:, 0] - tzero
                data[:, 1] = [ebounds[x] for x in data[:, 1]]
                

            if apply_redshift:
                data[:,0], data[:,1] = self.apply_redshift(data[:,0], data[:,1], self.redshift)

            if filter_energy is not None:
                if isinstance(filter_energy,dict):
                    data = self.filter_energy(data, detector = detector, **filter_energy)
                else:
                    data = self.filter_energy(data, detector = detector)

            if scale == 'ijd':
                data[:,0] = data[:,0]/(24 * 60 * 60) + change_fermi_seconds(tzero, 'ijd')

            self.photon_data = np.concatenate((self.photon_data,data)) if self.photon_data is not None else data

            bined_times,_,bined_signal,_,binning = rebin_data(data[:,0], np.full(data.shape[0],1), 0, self.original_resolution, binning)
            times_array.append(bined_times)
            signal_array.append(bined_signal)
            
        times = np.mean(times_array,axis=0)[1:-2]
        signal = np.sum(signal_array,axis=0)[1:-2]
            
        return times,signal
    
    @staticmethod
    def load_fits(url: str, file_extension: str = 'fit', save = True):
        '''
        Loads fits file from heasarc server
        Args:
            url (str): url to search fits for, without file extension and _vXX suffix
            file_extension (str, optional): file extension, default is 'fit', can be 'fit.gz'
        Returns:
            astropy.io.fits
        '''
        for i in range(5):
            logging.debug(f'Loading {url}_v0{i}.{file_extension}, try {i}')
            try:
                with open(f"{LIGHT_CURVE_SAVE}{url.split('/')[-1]}_v0{i}.pkl", "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                try:
                    data = retry(fits.open(f"{url}_v0{i}.{file_extension}"))
                    if save:
                        with open(f"{LIGHT_CURVE_SAVE}{url.split('/')[-1]}_v0{i}.pkl", "wb") as f:
                            pickle.dump(data, f)
                    return data
                except requests.exceptions.HTTPError:
                    pass
        raise ValueError(f'No data found for {url}_v0{i}.{file_extension}') 
        
    @staticmethod
    def filter_energy(data: np.ndarray, low_en: float = None, high_en: float = None, detector: str = 'n0'):
        '''
        Filter the energy of the light curve
        Args:
            data (np.ndarray): time of photons and threir energy
            low_en (float, optional): Low energy threshold
            high_en (float, optional): High energy threshold
            detector (str, optional): GBM detector, first letter defines type: BGO or NaI
        '''
        if detector[0] == 'b':
            low_en = low_en if low_en is not None else 200
            high_en = high_en if high_en is not None else 6500
        elif detector[0] == 'n':
            low_en = low_en if low_en is not None else  6
            high_en = high_en if high_en is not None else 850
            
        return data[(data[:,1]>low_en)&(data[:,1]<high_en)]
            
    @staticmethod
    def apply_redshift(times: np.array, energy: np.array, redshift: float):
        '''
        Apply redshift to data
        Args:
            times (np.array): time of photons and their energy
            energy (np.array): energy of photon
            redshift (float): Cosmological redshift
        '''
        times = times / (1 + redshift)
        energy = energy * (1 + redshift)
        return times, energy

    def rebin(self, bin_duration: float = None, reset: bool = True):
        '''
        Overrides LightCurve.rebin() method, uses photon data when reset is True
        Args:
            bin_duration (float): new bin duration in seconds, if None, return to original resolution
            reset (bool, optional): if True, rebin the light curve using photon data, result is same as self.rebin().rebin(bin_duration)
        Returns:
            self
        '''
        if (bin_duration is None) or (reset is False):
            return super().rebin(bin_duration, reset)
        else:
            bined_times, bined_times_err, bined_signal, bined_signal_err, _ = rebin_data(self.photon_data[:,0], np.full(self.photon_data.shape[0],1), 0, bin_duration)

            self.signal = bined_signal
            self.signal_err = bined_signal_err
            self.times = bined_times
            self.times_err = bined_times_err
            self.resolution = bin_duration

            return self
        
    def save(self, filename: str = None):
        '''
        Save light curve to file
        Args:
            filename (str, optional): filename without extension
        '''
        filename = filename if filename else f'{self.event_time.replace(":","_")}_{int(self.duration)}.txt'
        np.savetxt(f'{LIGHT_CURVE_SAVE}{filename}',
                   self.photon_data)
        


class BATLightCurve(LightCurve):
    '''
    Class for light curves from BAT/Swift
    '''
    def __init__(self, event_time: str, 
                 duration: float, 
                 energy_range: list = [(50,100),(100,350)], 
                 loading_method: str = 'web', 
                 scale: str = 'utc', 
                 **kwargs):
        '''
        Args:
            event_time (str): date and time of event in ISO 8601 format
            duration (int): duration of light curve in seconds
            energy_range (tuple, optional): energy range of light curve, available ranges are (15,25),(25,50),(50,100),(100,350)
            loading_method (str, optional): 'local' or 'web'
            scale (str, optional): 'utc' or 'ijd'
        '''
        super().__init__(event_time, duration, **kwargs)
        self.event_time = self.event_time[:10] + ' ' + self.event_time[11:19]
        self.energy_range = energy_range
        
        if self.original_times is None:
            if loading_method =='web':
                self.original_times,self.original_signal = self.__get_light_curve_from_web(scale = scale)
            else:
                raise NotImplementedError(f'Loading method {loading_method} not supported')

        self.original_resolution = 0.064 # 64 ms
        self._reset_light_curve()

    def __get_light_curve_from_web(self, scale: str):
        event_datetime = pd.to_datetime(self.event_time)
        left = (event_datetime - datetime.timedelta(seconds = self.duration))
        right = (event_datetime + datetime.timedelta(seconds = self.duration))

        columns = ['Begin','End','Target ID','Seg.','Target Name','R.A.','Dec.','Roll','XRT Mode','UVOT Mode','Merit','Time (s)','Date']
        table = None
        for date in pd.date_range(left.date(), right.date(), freq='1D'):
            request = requests.get(f'https://www.swift.psu.edu/operations/obsSchedule.php?newdate={date.year}%2F{str(date.month).zfill(2)}%2F{str(date.day).zfill(2)}&type=1').text
            add = parse_html_table(request)
            add['Date'] = date.date()
            table = pd.concat((table, add)).reset_index(drop=True)
        table.columns = columns
        table['obsid'] = table.apply(lambda x: f"{int(x['Target ID']):08d}{int(x['Seg.']):03d}", axis=1)
        table['Begin'] = pd.to_datetime(table['Begin'])
        table['End'] = pd.to_datetime(table['End'])

        relevant = table[get_overlaping_intersection(left, right, table['Begin'], table['End'])][['Date','obsid']].drop_duplicates()

        total = None
        for _, row in relevant.iterrows():
            try:
                res = self.__download_ftp(row['Date'], row['obsid'])
            except ftplib.error_perm:
                res = self.__download_nasa(row['obsid'])
            if res is None:
                logging.error(f"No data for {row['Date']} {row['obsid']}")
                continue
            data = np.array([(float(x[0]),float(x[1][0]),float(x[1][1]),float(x[1][2]),float(x[1][3])) for x in res[1].data])
            data[:,0] = data[:,0] - change_utc(self.event_time, 'fermi_seconds')
            total = np.concatenate((total, data)) if total is not None else data
        total = total[total[:,0].argsort()]
        total = total[(total[:,0] >= -self.duration)&(total[:,0] <= self.duration)]

        if scale == 'ijd':
            event_time_ijd = change_utc(self.event_time, 'ijd')
            total[:,0] = total[:,0]/(24*60*60) + event_time_ijd

        columns = self.__determine_energy_range()

        return total[:,0], total[:,columns[0]:(columns[-1]+1)].sum(axis=1)

    def __determine_energy_range(self):
        energy_ranges = [(15,25),(25,50),(50,100),(100,350)]
        columns = []
        for i, en_range in enumerate(energy_ranges):
            if en_range in self.energy_range:
                columns.append(i + 1)
        return columns

    def __download_nasa(self, obsid):
        for idx in range(25):
            url = f'https://swift.gsfc.nasa.gov/data/swift/.original/sw{obsid}.{idx:03d}/data/bat/rate/sw{obsid}brtms.lc.gz'
            try:
                return fits.open(url)
            except Exception as e:
                pass
        return None
    
    def __download_ftp(self, date: str, obsid):
        ftp = ftplib.FTP_TLS('heasarc.gsfc.nasa.gov')
        ftp.login()
        ftp.prot_p()

        ftp_dir = f"swift/data/obs/{str(date)[0:4]}_{str(date)[5:7]}/{obsid}/bat/rate"
        ftp.cwd(ftp_dir)
        ftp.retrbinary("RETR " + ftp.nlst("*brtms.lc.gz")[0], open('BATLightCurve.lc.gz','wb').write)

        return fits.open('BATLightCurve.lc.gz')
    


class IREM_LightCurve(LightCurve):
    '''
    Class for light curves from IREM/INTEGRAL
    '''
    def __init__(self,center_time,duration, loading_method: str='local', scale: str='utc', *args, **kwargs):
        #todo
        pass