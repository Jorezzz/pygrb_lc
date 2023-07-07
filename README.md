# pygrb_lc
Package to load and handle GRB (Gamma-ray bursts) light curves, their furie transformations and various other functions.

To install run the following command

```bash
pip install --upgrade pygrb_lc
```
or
```bash
pip3 install --upgrade pygrb_lc
```

## Light curves

Main object in this package is LightCurve object and its children. Start by creating one
```python
from pygrb_lc.light_curves import LightCurve # base class for ligth curve

lc = LightCurve()
print(lc)
```
It is created but empty, semplest way to provide data is by ```data``` argument, that requires ```numpy.ndarray``` with 2 (time, signal) or 4 (time, time_err, signal, signal_err) columns
```python
import numpy as np

data = np.loadtxt('test.txt')
lc = LightCurve(data = data)
print(lc)
```

Base class can rebin data, subtract polynomial function, filter peaks, set intervals and load and save data to ```LIGHT_CURVE_SAVE``` folder in pickle format.

There are specific classes for INTERAL/SPI-ACS and Fermi/GBM instruments. They are able to load actual data from web, you need to specify it in loading_method parameter
```python
from pygrb_lc.light_curves import SPI_ACS_LightCurve, GBM_LightCurve

lc1 = SPI_ACS_LightCurve('2020-01-01 00:00:00', 500, loading_method = 'web')
lc2 = GBM_LightCurve('2020-01-01 00:00:00', ['na'], duration = 500, loading_method = 'web')
```
## Furie transformations

Furie transformation is performed by FurieLightCurve class. It requires LightCurve object as an argument
```python
from pygrb_lc.furie import FurieLightCurve

lc = LightCurve(data = np.loadtxt('test.txt'))
flc = FurieLightCurve(lc, interval_t90 = (0, 10))

flc.plot()
```

All classes have plot method, you can provide matplotlib.pyplot.Axes object as an argument to plot on existing plot
```python
fig,(ax1,ax2) = plt.subplots(2,1)
lc.plot(ax = ax1)
flc.plot(ax = ax2)
```

# Catalog

```Catalog```  class is an extension of pandas.DataFrame. It provides simplicity of manipulations and visualization and extend it with crossmatching functionality. Easiest way to start using it is to create it from existing tabular data. It is important to provide `event_column` argument, it is the name of column that will be used for crossmatching and comparison

```python
from pygrb_lc.catalogs import Catalog
import pandas as pd
import numpy as np

data = np.loadtxt('test.txt')
cat = Catalog(data, columns = ['datetime','duration'], event_column = 'datetime')
cat.find_event(pd.Timestamp('2023-01-01 00:00:00'), precision = 10)
```

`precision` (in seconds) controls uncertainity that is allowed for the event. If there is no such event method will return `None`.
There is a method of crossmatching two catalogs `crossmatch`, it works like inner join for SQL tables, but compares only corresponding `event_column` columns and allows uncertainity via `precision`. `precision = 0` equals to exact match.


# Roadmap
Add support of `LightCurve` class in `Catalog`
Add support of spectra (based on current `photon_data` in `GBM_LightCurve` class)
Add typical functions for approximation: Band function, power law, etc. and their interaction with `LightCurve` and `FurieLightCurve` classes.