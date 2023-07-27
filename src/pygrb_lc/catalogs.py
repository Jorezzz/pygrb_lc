from .crossmatching import find_closest_event
from .utils import parse_html_table
from bs4 import BeautifulSoup
import pickle
import requests
import pandas as pd
import re

class Catalog(pd.DataFrame):
    def __init__(self, *args, event_column = None, **kwargs):
        '''
        Args:
            event_column (int, str, optional): column with datetime of event, this columns should be of type datetime
        '''
        super().__init__(*args, **kwargs)
        self.event_column = event_column if event_column is not None else self.columns[0]
        self.param_columns = [_ for _ in self.columns if _ != self.event_column]

    def find_closest(self, event_time):
        idx = find_closest_event(event_time, self[self.event_column])
        return self.iloc[idx]

    def find_event(self, event_time, precision : int = 5):
        '''
        Args:
            precision (int, optional): precision of crossmatching in seconds
        '''
        closest = self.find_closest(event_time)
        if abs(closest[self.event_column] - event_time).total_seconds() <= precision:
            return closest
        else:
            return None

    def crossmatch(self, other, precision: int = 5):
        '''
        Args:
            other (Catalog): instance of other catalog to match with
            precision (int, optional): precision of crossmatching in seconds
        '''
        for i,event in self.iterrows():
            idx = other.find_event(event[self.event_column], precision)
            if idx is not None:
                for column in other.param_columns:
                    column_new = column if column not in self.param_columns else column + '_other'
                    self.loc[self.index==i,[column_new]] = other.loc[other.index==idx.name,column].values
        return self
    
    def save(self, path):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)

class Greiner_Catalog(Catalog):
    '''
    Parses Greiner online table
    '''
    def __init__(self, **kwargs):
        data, columns = self.__parse_greiner_table()
        super().__init__(data, columns=columns, **kwargs)

    def __parse_greiner_table(self):
        r = requests.get('https://www.mpe.mpg.de/~jcg/grbgen.html')
        soup = BeautifulSoup(r.text, 'lxml')
        df = parse_html_table(str(soup.find_all('table')[0]))
        return df.values, df.columns

    def update(self, **kwargs):
        self.__init__(**kwargs)        

class GBM_Catalog(Greiner_Catalog):
    '''
    Utilizes GBM online table
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self['gbm_code'] = self['GRBa'].astype(str).apply(lambda x: x if x[-1]!='S' else x[:-1]).apply(self.__find_gbm_code)
        self.event_column = 'gbm_code'
        self = self.dropna().reset_index(drop=True)

    def __find_gbm_code(self, grb_name):
        r = requests.get(f'https://www.mpe.mpg.de/~jcg/grb{grb_name}.html').text
        regexs = re.findall(r'bn\d{9}', r)
        if len(regexs) == 0:
            return None
        else:
            return regexs[0]