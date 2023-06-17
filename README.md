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

Main object in this package is LightCurve object and its children. Start by creating one
```python
from pygrb_lc.light_curves import LightCurve # base class for ligth curve

lc = LightCurve()
print(lc)
```
It is created but empty, semplest way to provide data is by data argument, that requires numpy.ndarray with 2 (time, signal) or 4 (time, time_err, signal, signal_err) columns
```python
import numpy as np

data = np.loadtxt('test.txt')
lc = LightCurve(data = data)
print(lc)
```

There are specific classes for INTERAL/SPI-ACS and Fermi/GBM instruments. They are able to load actual data from web, you need to specify it in loading_method parameter
```python
from pygrb_lc.light_curves import SPI_ACS_LightCurve, GBM_LightCurve

lc1 = SPI_ACS_LightCurve('2020-01-01 00:00:00', 500, loading_method = 'web')
lc2 = GBM_LightCurve('2020-01-01 00:00:00', ['na'], duration = 500, loading_method = 'web')
```
