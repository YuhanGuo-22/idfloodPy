# idfloodPy

`idfloodPy` is a Python package for flood event separation.

Input: runoff time series and catchment area for baseflow separation
Output: separated flood events named with the start of each flood event (.csv file)

## Installation

```bash
pip install idfloodPy
```

## Usage

The flood_separate() needs users to provide:  

`filePath`, `savePath`, `catchmentID`, `catchment area`, `runoff time series`,

Apart from that, several optional variables are available:

1. `yarly_check`: if `True`, it will generate yearly runoff data figures, 
    with flood event start and end points as well as peaks threshold.
    
2. `peak_height`: The default value of `peak_height` is set to be the 90th 
    percentile value of the entire runoff series, which can be customized as needed.

3. `calculate_baseflow`: if `True`, the baseflow will be calculated, and this can 
                         be muted to `False` by users. If user can provide baseflow, 
                         please ensure the column name is set to be `baseFlow` and contains no nan value.  

Variables for flood separation settings:                    
1. `qb_threshold` is the threshold of average baseflow proportion of runoff with default value 0.5. 
   If baseflow is smaller than this threshold, the flood event valley point will be calculated using `Qobs=Qbase` 
   method (Tarasova, L.,2018). Otherwise, it will find valley points using `find_peaks` method in `scipy.signal `
   
2. `Qdiff_threshold` is the differences of Qobs and Qbase, with default value to be 0.005, 
    which can be customized as needed.

3. `peaks_diff_threshold` is the multiple between two adjacent peaks with default value to be 2.5 which can be customized as needed. 
    If the multiple of the adjacent peaks is greater than this threshold, 
     the two floods will be merged into one.
     
4. `peak_interval_threshold` if interval of two adjacent peaks with default value to be 14 which can be customized as needed. if 
    two adjacent peaks interval is very close, this two peaks will be merged. 
    When `peaks_diff_threshold` > 2.5 and `peak_interval_threshold` < 14, two adjacent peaks will be merged. 


References: 

L. Tarasova, S. Basso, M. Zink, R. Merz, Exploring Controls on Rainfall-Runoff Events: 1. Time Series-Based Event 
Separation and Temporal Dynamics of Event Runoff Response in Germany. Water Resources Research 54, 7711-7732 (2018).

S. Zhang et al., Reconciling disagreement on global river flood changes in a warming climate. 
Nature Climate Change 12, 1160-1167 (2022).


```python
from idfloodPy.idFlood import flood_separate
import pandas as pd

filePath = "path/to/data"
savePath = "path/to/save"
catchmentID = "Canada_02XA003"
area = 4473.3
data = pd.read_csv(filePath + catchmentID + ".csv", index_col=0)

flood_separate(filePath, savePath, catchmentID, area, data, 
               yarly_check=True, peak_height=None, calculate_baseflow=True,
               qb_threshold=0.5, Qdiff_threshold=0.005, peaks_diff_threshold=2.5, peak_interval_threshold=14)
```
