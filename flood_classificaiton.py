#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
------------------------------------
@Project ：007-Hydrolib 
@File    ：flood_classification.py
@Author  ：HannahG
@Date    ：2025/2/24 9:13
@Desc:
-------------------------------------
'''

import pandas as pd
import numpy as np
import math
import warnings


def Tr_DMCA(rain_array, flow_array, max_window):
    # ROUTINE
    rain_int = np.nancumsum(rain_array, axis=0)  # cumulating rainfall timeseries (Eq.1)
    # print(rain_int)
    flow_int = np.nancumsum(flow_array, axis=0)  # cumulating streamflow timeseries (Eq.2)
    T = len(rain_array)  # length of the timeseries
    n = 0  # counter for the windows tested
    rho = np.empty(int(max_window / 2)) * np.nan  # initializing rho coefficient as a NaN vector

    for window in range(3, max_window, 2):
        rain_mean = np.convolve(rain_int, np.ones(window),
                                'valid') / window  # moving average on the integrated rainfall timeseries (Eq.5)
        flow_mean = np.convolve(flow_int, np.ones(window),
                                'valid') / window  # moving average on the integrated streamflow timeseries (Eq.6)
        flutt_rain = rain_int[int(float(window) / 2 + 0.5 - 1):int(len(rain_int) - float(window) / 2 + 0.5)] - rain_mean
        F_rain = (1 / float(T - window + 1)) * np.nansum(
            (np.square(flutt_rain)))  # Squared rainfall fluctuations (Eq.3)
        flutt_flow = flow_int[int(float(window) / 2 + 0.5 - 1):int(len(flow_int) - float(window) / 2 + 0.5)] - flow_mean
        F_flow = (1 / float(T - window + 1)) * np.nansum(
            (np.square(flutt_flow)))  # Squared streamflow fluctuations (Eq.4)
        F_rain_flow = (1 / float(T - window + 1)) * np.nansum(
            np.multiply(flutt_rain, flutt_flow))  # Bivariate rainfall-streamflow fluctuations (Eq.7)
        if np.logical_or(F_rain == 0, F_rain == 0):
            rho[n] = np.nan  # avoiding division by 0
        else:
            rho[n] = F_rain_flow / (math.sqrt(F_rain) * math.sqrt(F_flow))  # DMCA-based correlation coefficent (Eq.8)
        n = n + 1

    # OUTPUT
    position_minimum = np.where(rho == np.nanmin(rho))
    catchment_response_time = float(position_minimum[0][0]) + 1
    return catchment_response_time


if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")

    filePath = "H:/007-Hydrolib/2024-data-全球径流数据更新/flood_identification/"
    flood_event_path = "H:/007-Hydrolib/2024-data-全球径流数据更新/flood_identification/flood_events/Canada_02XA003/"
    # savePath = "G:/3-data analysis/"

    # Load the catchment time series
    data = pd.read_csv(filePath + "Canada_02XA003_for_classification.csv", index_col=0)

    # Load flood event data
    flood_data = pd.read_csv(flood_event_path + "19790420.csv", index_col=0)

    # Calculate rainfall as the difference between total precipitation and snowfall
    data["rainfall"] = data.apply(lambda row: row["pcp_era5"] - row["sf_era5"], axis=1)

    # Convert index to DatetimeIndex to ensure time-based operations
    data.index = pd.to_datetime(data.index)
    flood_data.index = pd.to_datetime(flood_data.index)

    # Identify the first and last timestamps of the flood event
    first_index = flood_data.index[0]
    last_index = flood_data.index[-1]

    # print("First index:", first_index)
    # print("Last index:", last_index)

    # Define a pre-flood analysis period
    preDays = 30
    first_index_adj = first_index - pd.Timedelta(days=preDays)  # Adjust start date
    index_type = "datetime"
    # print("Adjusted first index:", first_index_adj)

    # Extract the relevant data period from the main dataset
    filtered_data = data.loc[first_index_adj:last_index]
    # print(filtered_data)

    # Convert relevant columns to NumPy arrays for further analysis
    flow_array = np.array(filtered_data["Runoff_mmd"])
    rain_array = np.array(filtered_data["pcp_era5"])
    smlt_array = np.array(filtered_data["smlt_era5"])

    # Define analysis parameters
    max_window = 20
    rain_response_time = 0
    smlt_response_time = 0

    # Compute rain response time if rainfall is significant
    if rain_array.sum() >= len(rain_array) * 0.01:
        rain_response_time = Tr_DMCA(rain_array, flow_array, max_window)

    # Compute snowmelt response time if snowmelt is significant
    if smlt_array.sum() >= len(smlt_array) * 0.01:
        smlt_response_time = Tr_DMCA(smlt_array, flow_array, max_window)

    # Determine the overall catchment response time
    catchment_response_time = max(rain_response_time, smlt_response_time)
    print("Catchment response time:", catchment_response_time)

    # Define the total time extension for event analysis
    antecedent_time = 10
    total_extend = int(catchment_response_time + antecedent_time)

    # Extract the flood event and antecedent period
    flood_event_extend = filtered_data.iloc[int(preDays - total_extend):]
    flood_antecedent = flood_event_extend.iloc[:11]  # First 10 days + 1

    # Identify the flood rising phase based on the peak runoff
    rising_peak = flood_data["runoff_mmd"].argmax()
    flood_rising = flood_event_extend.iloc[10: (rising_peak + total_extend) + 1]

    # Compute rainfall and snowmelt characteristics
    pcp_mean = flood_rising["rainfall"].mean()
    smlt_mean = flood_rising["smlt_era5"].mean()

    # Calculate the rolling 3-day average
    pcp_rolling_avg = flood_rising["rainfall"].rolling(3).mean()
    pcp_max_3day = pcp_rolling_avg.max()
    smlt_rolling_avg = flood_rising["smlt_era5"].rolling(3).mean()
    smlt_max_3day = smlt_rolling_avg.max()

    # Compute the long-term precipitation mean
    P_use = np.array(data["pcp_era5"].dropna())
    P_mean = np.mean(P_use)

    # Compute classification indices
    Ism = smlt_mean / P_mean
    Ism_pd = smlt_mean / pcp_mean
    Ism_max_pd = smlt_max_3day / pcp_mean
    Ipd = pcp_mean / P_mean
    Ipmax = pcp_max_3day / P_mean


    # Function to classify flood type based on computed indices
    def determine_flood_type(Ism, Ism_pd, Ism_max_pd, Ipd, Ipmax):
        if Ism < 0.28:
            return 3  # Rainfall-induced flood
        elif Ism >= 0.28:
            if Ism_pd >= 1.45 or Ism_max_pd >= 5.14:
                return 1  # Snowmelt-induced flood
            elif Ipd < 2.32 or Ipmax < 6.26:
                return 1  # Snowmelt-induced flood
            elif Ism_pd < 1.45 or Ism_max_pd < 5.14 or Ipd >= 2.32 or Ipmax >= 6.26:
                return 2  # Rain-on-snow flood
        return None  # Default case if no condition matches


    # Determine the flood type based on the computed indices
    flood_type = determine_flood_type(Ism, Ism_pd, Ism_max_pd, Ipd, Ipmax)
    print("Flood Type:", flood_type)






