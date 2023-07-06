import numpy as np
import bs4
import pandas as pd
from .config import GBM_DETECTOR_CODES, logging

def Chi2_polyval(x: np.array, y: np.array, param: np.array):
    """
    Returns Chi square functional for np.polyfit approximation
    """
    approximation = np.polyval(param,x)
    squared_error = np.square((np.asarray(y) - approximation))/approximation

    return np.sum(squared_error)/(len(x)-len(param))

def get_first_intersection(x: np.array, y: np.array, value: np.array):
    '''
    Finds the first intersection of y = value and curve y = f(x)
    '''
    return x[np.argmax(y > value)]

def is_iterable(x):
    '''
    Checks if the given element x is iterable
    '''
    try:
        iterator = iter(x)
        return True
    except TypeError:
        return False
    
def extract_number(string: str):
    '''
    Tries to extract number from string
    Args:
        string: string to extract from
    '''
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            return string

def parse_html_table(html_string: str):
    '''
    Function to parse html table into pandas dataframe
    Args:
        html_string: html as string
    '''
    doc = bs4.BeautifulSoup(html_string, 'html.parser')
    tables = doc.find_all('table')
    if tables is None or tables == []:
        return None
    
    data = []
    for table in tables:
        all_th = table.find_all('th')
        all_heads = [th.get_text() for th in all_th]
        for tr in table.find_all('tr'):
            all_th = tr.find_all('th')
            if all_th:
                continue
            all_td = tr.find_all('td')
            data.append([extract_number(td.get_text().strip()) for td in all_td])
    if len(all_heads) > 1:
        return pd.DataFrame(data, columns = all_heads)
    else:
        return pd.DataFrame(data[1:], columns = data[0])

def retry(expr, tries = 5):
    '''
    Function to retry expression if it fails
    Args:
        expr: expression to retry
        tries: number of tries
    '''
    for i in range(tries):
        logging.debug(f'Attempt {i+1} of {tries}')
        try:
            return expr
        except Exception as e:
            logging.error(f'Attempt {i+1} of {tries} failed, {e}')
            if i == tries - 1:
                raise e
            
def get_detectors_from_GBM_mask(detector_mask: str):
    '''
    Using GBM_DETECTOR_CODES converts string of 0 and 1 to list om most luminant detectors
    Args:
        detector_mask: string of 0 and 1
    Returns:
        list of most luminant detectors (list[str])
    '''
    return [GBM_DETECTOR_CODES[i] for i,value in enumerate(list(detector_mask)) if value == '1']