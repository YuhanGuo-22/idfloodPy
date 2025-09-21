#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
------------------------------------
@Project: Hydrolib
@File: flood_classification_v2.py
@Author: HannahG
@Date: 2025/2/24 9:13
@Desc: Advanced flood classification system based on decision tree logic
       and multi-parameter threshold determination
-------------------------------------
'''

import pandas as pd
import numpy as np
import math
import warnings

def safe_divide_with_logic(numerator, denominator):
    """
    Safe division function that handles division by zero based on physical meaning
    - Denominator = 0 and numerator > 0: return 999 (pure snowmelt flood)
    - Denominator = 0 and numerator = 0: return NaN (suspicious data)
    - Denominator > 0: normal calculation
    """
    if denominator == 0:
        if numerator > 0:
            return 999  # Pure snowmelt flood
        else:
            return np.nan  # Suspicious data
    return numerator / denominator


def Tr_DMCA(rain_array, flow_array, max_window):
    """
    Calculate catchment response time
    """
    rain_int = np.nancumsum(rain_array, axis=0)
    flow_int = np.nancumsum(flow_array, axis=0)
    T = len(rain_array)
    n = 0
    rho = np.empty(int(max_window / 2)) * np.nan

    for window in range(3, max_window, 2):
        rain_mean = np.convolve(rain_int, np.ones(window), 'valid') / window
        flow_mean = np.convolve(flow_int, np.ones(window), 'valid') / window
        flutt_rain = rain_int[int(float(window) / 2 + 0.5 - 1):int(len(rain_int) - float(window) / 2 + 0.5)] - rain_mean
        F_rain = (1 / float(T - window + 1)) * np.nansum((np.square(flutt_rain)))
        flutt_flow = flow_int[int(float(window) / 2 + 0.5 - 1):int(len(flow_int) - float(window) / 2 + 0.5)] - flow_mean
        F_flow = (1 / float(T - window + 1)) * np.nansum((np.square(flutt_flow)))
        F_rain_flow = (1 / float(T - window + 1)) * np.nansum(np.multiply(flutt_rain, flutt_flow))
        if np.logical_or(F_rain == 0, F_rain == 0):
            rho[n] = np.nan
        else:
            rho[n] = F_rain_flow / (math.sqrt(F_rain) * math.sqrt(F_flow))
        n = n + 1

    position_minimum = np.where(rho == np.nanmin(rho))
    catchment_response_time = float(position_minimum[0][0]) + 1
    return catchment_response_time


def classify_flood_v2(smlt_mean, smlt_max_3day, pcp_mean, pcp_max_3day, snowc_mean, snow_depth_mean, P_mean, params):
    """
    Advanced flood classification method
    1: snowmelt-induced flood
    2: rain-on-snow flood  
    3: rainfall-induced flood
    
    Parameters:
    - smlt_mean: average snowmelt
    - smlt_max_3day: maximum 3-day snowmelt
    - pcp_mean: average precipitation
    - pcp_max_3day: maximum 3-day precipitation
    - snowc_mean: average snow cover
    - snow_depth_mean: average snow depth
    - P_mean: long-term average precipitation
    - params: [smlt_pcp_ratio, smlt_pcp_max_ratio, snowc_threshold, sd_threshold, pcp_intensity_ratio, smlt_intensity_ratio]
    """
    smlt_pcp_ratio = params[0]
    smlt_pcp_max_ratio = params[1]
    snowc_threshold = params[2]
    sd_threshold = params[3]
    pcp_intensity_ratio = params[4]
    smlt_intensity_ratio = params[5]

    # Calculate key ratios
    smlt_P_ratio = safe_divide_with_logic(smlt_mean, P_mean)
    pcp_P_ratio = safe_divide_with_logic(pcp_mean, P_mean)
    smlt_pcp_ratio_val = safe_divide_with_logic(smlt_mean, pcp_mean)
    smlt_pcp_max_ratio_val = safe_divide_with_logic(smlt_max_3day, pcp_max_3day)
    
    # If key ratios are NaN, return NaN
    if np.isnan(smlt_P_ratio) or np.isnan(pcp_P_ratio) or np.isnan(smlt_pcp_ratio_val) or np.isnan(smlt_pcp_max_ratio_val):
        return np.nan

    if smlt_P_ratio > 0:                        # Has snowmelt
        if pcp_P_ratio > 0:                     # Also has rainfall
            # ---- Three combinations dominated by snowmelt ----
            if smlt_pcp_ratio_val > smlt_pcp_ratio and smlt_pcp_max_ratio_val > smlt_pcp_max_ratio:
                return 1  # snowmelt-induced flood
            # Distinguish "snowmelt mean advantage, rainfall peak advantage" cases
            elif smlt_pcp_ratio_val > smlt_pcp_ratio and smlt_pcp_max_ratio_val < smlt_pcp_max_ratio:
                pcp_intensity_ratio_val = safe_divide_with_logic(pcp_max_3day, pcp_mean)
                if np.isnan(pcp_intensity_ratio_val):
                    return np.nan
                if pcp_intensity_ratio_val > pcp_intensity_ratio:
                    return 3  # rainfall-induced flood
                else:
                    return 1  # snowmelt-induced flood
            elif smlt_pcp_ratio_val < smlt_pcp_ratio and smlt_pcp_max_ratio_val > smlt_pcp_max_ratio:
                smlt_intensity_ratio_val = safe_divide_with_logic(smlt_max_3day, smlt_mean)
                if np.isnan(smlt_intensity_ratio_val):
                    return np.nan
                if smlt_intensity_ratio_val > smlt_intensity_ratio:
                    return 1  # snowmelt-induced flood
                else:
                    return 3  # rainfall-induced flood
            # ---- Low snowmelt proportion: classify as 2 / 3 based on snow conditions ----
            elif smlt_pcp_ratio_val < smlt_pcp_ratio and smlt_pcp_max_ratio_val < smlt_pcp_max_ratio:
                if snowc_mean > snowc_threshold and snow_depth_mean > sd_threshold:
                    return 2  # rain-on-snow flood
                else:
                    return 3  # rainfall-induced flood
        else:
            return 1  # Only snowmelt → snowmelt-induced flood
    else:
        return 3      # No snowmelt → rainfall-induced flood


def main():
    """
    Main function: Demonstrate advanced flood classification method
    """
    print("=" * 60)
    print("Flood Classification System v2.0 - Advanced Decision Tree Method")
    print("=" * 60)
    
    # Ignore warnings
    warnings.filterwarnings("ignore")
    
    # File path configuration
    filePath = "path/to/time-series data"
    flood_event_path = "path/to/flood event data"
    
    print("\n1. Data Loading and Preprocessing")
    print("-" * 40)
    
    # Load catchment time series data
    data = pd.read_csv(filePath + "Canada_02XA003_for_classification.csv", index_col=0)
    print(f"✓ Loaded catchment data: {data.shape[0]} records")
    
    # Load flood event data (example: 1979-04-20 flood)
    flood_data = pd.read_csv(flood_event_path + "19790420.csv", index_col=0)
    print(f"✓ Loaded flood event data: {flood_data.shape[0]} records")
    
    # Calculate rainfall (total precipitation - snowfall)
    data["rainfall"] = data.apply(lambda row: row["pcp_era5"] - row["sf_era5"], axis=1)
    
    # Convert time index
    data.index = pd.to_datetime(data.index)
    flood_data.index = pd.to_datetime(flood_data.index)
    
    print("\n2. Catchment Response Time Calculation")
    print("-" * 40)
    
    # Determine flood event time range
    first_index = flood_data.index[0]
    last_index = flood_data.index[-1]
    preDays = 30
    first_index_adj = first_index - pd.Timedelta(days=preDays)
    
    # Extract data for analysis period
    filtered_data = data.loc[first_index_adj:last_index]
    
    # Calculate catchment response time
    flow_array = np.array(filtered_data["Runoff_mmd"])
    rain_array = np.array(filtered_data["pcp_era5"])
    smlt_array = np.array(filtered_data["smlt_era5"])
    
    max_window = 20
    rain_response_time = 0
    smlt_response_time = 0
    
    if rain_array.sum() >= len(rain_array) * 0.01:
        rain_response_time = Tr_DMCA(rain_array, flow_array, max_window)
    
    if smlt_array.sum() >= len(smlt_array) * 0.01:
        smlt_response_time = Tr_DMCA(smlt_array, flow_array, max_window)
    
    catchment_response_time = max(rain_response_time, smlt_response_time)
    print(f"✓ Catchment response time: {catchment_response_time} days")
    
    print("\n3. Flood Characteristics Extraction")
    print("-" * 40)
    
    # Define analysis period
    antecedent_time = 10
    total_extend = int(catchment_response_time + antecedent_time)
    flood_event_extend = filtered_data.iloc[int(preDays - total_extend):]
    
    # Identify flood rising phase
    rising_peak = flood_data["runoff_mmd"].argmax()
    flood_rising = flood_event_extend.iloc[10: (rising_peak + total_extend) + 1]
    
    # Calculate flood characteristics
    pcp_mean = flood_rising["rainfall"].mean()
    smlt_mean = flood_rising["smlt_era5"].mean()
    
    # Calculate 3-day rolling averages
    pcp_rolling_avg = flood_rising["rainfall"].rolling(3).mean()
    pcp_max_3day = pcp_rolling_avg.max()
    smlt_rolling_avg = flood_rising["smlt_era5"].rolling(3).mean()
    smlt_max_3day = smlt_rolling_avg.max()
    
    # Calculate long-term precipitation mean
    P_use = np.array(data["pcp_era5"].dropna())
    P_mean = np.mean(P_use)
    
    # Calculate snow-related indicators (if available)
    snowc_mean = flood_rising["snowc_era5"].mean() if "snowc_era5" in flood_rising.columns else 0
    snow_depth_mean = flood_rising["sd_era5"].mean() if "sd_era5" in flood_rising.columns else 0
    
    print(f"✓ Average rainfall: {pcp_mean:.3f} mm/day")
    print(f"✓ Average snowmelt: {smlt_mean:.3f} mm/day")
    print(f"✓ Maximum 3-day rainfall: {pcp_max_3day:.3f} mm/day")
    print(f"✓ Maximum 3-day snowmelt: {smlt_max_3day:.3f} mm/day")
    print(f"✓ Long-term average precipitation: {P_mean:.3f} mm/day")
    print(f"✓ Average snow cover: {snowc_mean:.3f}%")
    print(f"✓ Average snow depth: {snow_depth_mean:.3f} m")
    
    print("\n4. Advanced Flood Classification")
    print("-" * 40)
    
    # Use optimal parameters for classification
    optimal_params = [1.12, 1.65, 42.86, 0.06, 3.75, 3.40]
    print(f"✓ Classification parameters: {optimal_params}")
    
    # Apply advanced classification method
    flood_type = classify_flood_v2(
        smlt_mean, smlt_max_3day, pcp_mean, pcp_max_3day, 
        snowc_mean, snow_depth_mean, P_mean, optimal_params
    )
    
    # Output classification results
    flood_type_names = {1: "Snowmelt-induced", 2: "Rain-on-snow", 3: "Rainfall-induced"}
    
    print(f"✓ Classification result: {flood_type} ({flood_type_names.get(flood_type, 'Unknown type')})")
    
    print("\n" + "=" * 60)
    print("Flood Classification Analysis Complete!")
    print("=" * 60)
    
    return {
        'flood_type': flood_type,
        'characteristics': {
            'pcp_mean': pcp_mean,
            'smlt_mean': smlt_mean,
            'pcp_max_3day': pcp_max_3day,
            'smlt_max_3day': smlt_max_3day,
            'P_mean': P_mean,
            'snowc_mean': snowc_mean,
            'snow_depth_mean': snow_depth_mean
        },
        'catchment_response_time': catchment_response_time
    }


if __name__ == '__main__':
    result = main()
