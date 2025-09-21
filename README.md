# idfloodPy

`idfloodPy` is a Python package for flood event separation.

Input: runoff time series and catchment area for baseflow separation  
Output: separated flood events named with the start of each flood event (.csv file)

## Installation

```bash
pip install idfloodPy == 0.1.1
```

### System Requirements

- Python 3.7 or higher
- pip package manager

### Dependencies

#### Core Dependencies

| Package | Version  | Purpose |
|---------|----------|---------|
| pandas | >=1.3.0  | Data manipulation and analysis |
| numpy | >=1.20.0 | Numerical computing |
| scipy | >=1.7.0  | Scientific computing (signal processing) |
| matplotlib | >=3.3.0  | Data visualization and plotting |
| baseflow | <= 0.0.8 | Baseflow separation algorithms |

#### Built-in Modules (No Installation Required)

- `os` - Operating system interface
- `math` - Mathematical functions
- `bisect` - Binary search algorithms
- `warnings` - Warning control


## Usage

### 1. Flood identification

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


```python
from idfloodPy.idFlood import flood_separate
import pandas as pd

filePath = "path/to/data"
savePath = "path/to/save"
# Canada_02XA003.csv is the example csv file that can be downloaded from https://github.com/YuhanGuo-22/idfloodPy
catchmentID = "Canada_02XA003"
area = 4473.3
data = pd.read_csv(filePath + catchmentID + ".csv", index_col=0)

flood_separate(filePath, savePath, catchmentID, area, data, 
               yarly_check=True, peak_height=None, calculate_baseflow=True,
               qb_threshold=0.5, Qdiff_threshold=0.005, peaks_diff_threshold=2.5, peak_interval_threshold=14)
```

### 2.Flood classification

#### Update 2025-9-21 flood_classificaiton_v2.py

Improvements in flood_classification_v2.py

- 🚀 **Exclude snowfall**: Removed **snowfall from ERA5-Land** and used **rainfall intensity** to compare with **snowmelt**.  
- 🚀 **Snow conditions**: Added **snow depth** and **snow cover fraction** during flood periods → more realistic detection of **rain-on-snow events**.  
- 🚀 **Mean vs. peak dominance**:  
  - **Snowmelt mean > rainfall mean** **but** **rainfall peak > snowmelt peak** → use **rainfall intensity ratio** to check if **extreme rainfall dominates**.  
  - **Rainfall mean > snowmelt mean** **but** **snowmelt peak > rainfall peak** → use **snowmelt intensity ratio** to test for **short-term intensive melt**.  

➡️ These three refinements improve the **mechanistic classification** of **snowmelt floods**, **rain-on-snow floods**, and **rainfall floods** in cold regions.  

**Parameters**: optimized with **SCE-UA** (3 rounds × 600 iterations). <br>
Initial ranges = **5th–95th percentile** of all events. <br>
Data source = **ERA5-Land**.  

#### Update 2025-2-24 flood_classificaiton.py 

flood_classificaiton.py is a simple flood classification code used to recognize the 
snowmelt-induced flood, rain-on-snow flood and rainfall-induced flood, which used optimal result from Zhang(2022)  

PS: Canada_02XA003_for_classification.csv attached is an example for flood classification

## How to Cite

If you use the code from this repository, please cite the following article:

Guo, Y., Yang, Y., Yang, D., Zhang, L., Zheng, H., Xiong, J., Ruan, F., Han, J., & Liu, Z. (2025).  
**Warming leads to both earlier and later snowmelt floods over the past 70 years.**  
*Nature Communications*, 16, Article 3663.  
[https://doi.org/10.1038/s41467-025-58832-0](https://doi.org/10.1038/s41467-025-58832-0)

## References: 

L. Tarasova, S. Basso, M. Zink, R. Merz, Exploring Controls on Rainfall-Runoff Events: 1. Time Series-Based Event 
Separation and Temporal Dynamics of Event Runoff Response in Germany. Water Resources Research 54, 7711-7732 (2018).

S. Zhang et al., Reconciling disagreement on global river flood changes in a warming climate. 
Nature Climate Change 12, 1160-1167 (2022).


