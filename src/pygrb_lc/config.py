import logging
import os
from pathlib import Path # TODO

ROOT_PATH = './'

IMAGE_PATH = f'{ROOT_PATH}pics/'
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

LIGHT_CURVE_SAVE = f'{ROOT_PATH}light_curves/'
if not os.path.exists(LIGHT_CURVE_SAVE):
    os.makedirs(LIGHT_CURVE_SAVE)

DATA_PATH = f'{ROOT_PATH}data/'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

LOGS_PATH = f'{ROOT_PATH}logs/'
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
]

GBM_DETECTOR_CODES = {0:'n0',1:'n1',2:'n2',3:'n3',4:'n4',5:'n5',6:'n6',7:'n7',8:'n8',9:'n9',10:'na',11:'nb',12:'b0',13:'b1'}
GBM_DETECTORS = [item for item in GBM_DETECTOR_CODES.values()]

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', filename=f'{LOGS_PATH}log.log',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')