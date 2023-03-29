import logging
import os

ROOT_PATH = './'

IMAGE_PATH = f'{ROOT_PATH}pics/'
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

LIGHT_CURVE_SAVE = f'{ROOT_PATH}light_curves/'
if not os.path.exists(LIGHT_CURVE_SAVE):
    os.makedirs(LIGHT_CURVE_SAVE)

ACS_DATA_PATH = 'E:/ACS/'

DATA_PATH = f'{ROOT_PATH}data/'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

LOGS_PATH = f'{ROOT_PATH}logs/'
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)

GBM_DETECTOR_CODES = {0:'n0',1:'n1',2:'n2',3:'n3',4:'n4',5:'n5',6:'n6',7:'n7',8:'n8',9:'n9',10:'na',11:'nb',12:'b0',13:'b1'}

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', filename=f'{LOGS_PATH}log.log',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')