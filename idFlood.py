import pandas as pd
import numpy as np
import os
import baseflow
from scipy.signal import find_peaks, savgol_filter
import bisect
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def assign_local_water_year(row, start_month):
    if row.month >= start_month:
        return row.year
    else:
        return row.year - 1


def refine_peak_qb90(qobs, peaks_obs_merge, valley_obs):
    # The peak value must be greater than the average value,
    peaks_obs_fil = []
    for k in range(len(peaks_obs_merge)):
        peaks_to_mean = qobs[peaks_obs_merge[k]] - np.mean(qobs)
        if peaks_to_mean <= 0:
            continue
        # peaks_to_mean_perc = (qobs[peaks_obs[k]] - np.mean(qobs)) / np.mean(qobs)
        # if peaks_to_mean_perc < 2:
        #     continue
        peaks_obs_fil.append(peaks_obs_merge[k])
    # print(peaks_obs_fil)

    point_less_all = []
    point_more_all = []
    # print(valley_obs)
    for k in range(len(peaks_obs_fil)):
        # Use the binary method to find the two nearest valley points corresponding to the flood peak
        index = bisect.bisect_left(valley_obs, peaks_obs_fil[k])
        if index == 0:
            point_less = valley_obs[index]
            point_more = valley_obs[index + 1]
        elif index == len(valley_obs):
            point_less = valley_obs[index - 2]
            point_more = valley_obs[index - 1]
        else:
            point_less = valley_obs[index - 1]
            point_more = valley_obs[index]
        point_less_all.append(point_less)
        point_more_all.append(point_more)
    # print(point_less_all)
    # print(point_more_all)

    valley_point = pd.DataFrame({"point_less": point_less_all, "point_more": point_more_all})
    # print(valley_point)
    # 1. There is no repeated peak at the valley point
    valley_point_notdup_index = valley_point[~valley_point.duplicated(keep=False)]
    peaks_obs_merge_not_dup = []
    if not valley_point_notdup_index.empty:
        for k in range(len(valley_point_notdup_index)):
            peaks_obs_merge_not_dup.append(peaks_obs_fil[valley_point_notdup_index.index[k]])
        # print(peaks_obs_merge_not_dup)

    # 2. There are repeated peaks in the valley points,
    # and the largest peak is selected as the peak between the valley points.
    valley_point_dup_index = valley_point[valley_point.duplicated(keep=False)]
    peaks_obs_merge_dup = []
    if not valley_point_dup_index.empty:
        valley_point_dup_index = valley_point_dup_index.groupby(list(valley_point_dup_index)).apply(
            lambda x: tuple(x.index)).tolist()
        # print(valley_point_dup_index)

        for k in range(len(valley_point_dup_index)):
            peaks_max = 0
            peaks_index_final = peaks_obs_fil[valley_point_dup_index[k][0]]
            for j in range(len(valley_point_dup_index[k])):
                peaks_temp = qobs[peaks_obs_fil[valley_point_dup_index[k][j]]]
                if peaks_temp > peaks_max:
                    peaks_max = peaks_temp
                    peaks_index_final = peaks_obs_fil[valley_point_dup_index[k][j]]
            peaks_obs_merge_dup.append(peaks_index_final)
    # print(peaks_obs_merge_not_dup)
    # print(peaks_obs_merge_dup)
    peaks_obs_merge = peaks_obs_merge_not_dup + peaks_obs_merge_dup

    return peaks_obs_merge


def refine_valley_qb90(qobs, peaks_obs_merge, valley_obs, peaks_diff_threshold=2.5, peak_interval_threshold=14):
    point_less_final_all = []
    point_more_final_all = []
    diff_peak_all = []
    multi_peak_all = []

    for k in range(len(peaks_obs_merge)):
        index = bisect.bisect_left(valley_obs, peaks_obs_merge[k])
        if index == 0:
            point_less_temp = valley_obs[index]
            point_more_temp = valley_obs[index + 1]
        elif index == len(valley_obs):
            point_less_temp = valley_obs[index - 2]
            point_more_temp = valley_obs[index - 1]
        else:
            point_less_temp = valley_obs[index - 1]
            point_more_temp = valley_obs[index]

        point_more_final_all.append(point_more_temp)
        point_less_final_all.append(point_less_temp)

        if k == 0:
            diff_peak = 0
            multi_peak = 0
        else:
            diff_peak = qobs[peaks_obs_merge[k - 1]] - qobs[peaks_obs_merge[k]]
            multi_peak = qobs[peaks_obs_merge[k - 1]] / qobs[peaks_obs_merge[k]]
        diff_peak_all.append(diff_peak)
        multi_peak_all.append(multi_peak)

    df_point = pd.DataFrame({'less': point_less_final_all, 'more': point_more_final_all,
                             "diff": diff_peak_all, "multi": multi_peak_all})

    for pindex, row in df_point.iterrows():
        if pindex >= 1 and pindex < len(df_point):
            prev_peak_index = peaks_obs_merge[pindex - 1]
            current_peak_index = peaks_obs_merge[pindex]
            peak_interval = current_peak_index - prev_peak_index

            if (row['less'] <= df_point.loc[pindex - 1, 'more']
                    and row["diff"] > 0
                    and row['multi'] > peaks_diff_threshold  # Customizable threshold for multi_peak
                    and peak_interval <= peak_interval_threshold):  # Customizable threshold for peak_interval
                df_point.loc[pindex - 1, 'more'] = np.nan
                df_point.loc[pindex, 'less'] = np.nan

    point_less_final_update = df_point["less"].values.tolist()
    point_more_final_update = df_point["more"].values.tolist()

    mask_less = np.isnan(point_less_final_update)
    mask_more = np.isnan(point_more_final_update)
    point_less_final_update = list(np.array(point_less_final_update)[~mask_less])
    point_more_final_update = list(np.array(point_more_final_update)[~mask_more])

    return point_less_final_update, point_more_final_update


def mkdir(path):
    path = path.strip().rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path)


def find_valleys(qobs, baseFlow, peaks_obs, Qdiff_threshold=0.005):
    point_less_ini = peaks_obs - 1
    point_more_ini = peaks_obs + 1

    diff_arr1 = abs(qobs[point_less_ini] - baseFlow[point_less_ini])
    diff_arr2 = abs(qobs[point_more_ini] - baseFlow[point_more_ini])

    point_less_final = point_less_ini
    while point_less_final > 0:
        if diff_arr1 > Qdiff_threshold:
            point_less_final = point_less_final - 1
            diff_arr1 = abs(qobs[point_less_final] - baseFlow[point_less_final])
        else:
            break

    point_more_final = point_more_ini
    while point_more_final < len(qobs):
        if diff_arr2 > Qdiff_threshold:
            diff_arr2 = abs(qobs[point_more_final] - baseFlow[point_more_final])
            point_more_final = point_more_final + 1
        else:
            break

    return point_less_final, point_more_final


def refine_peak(qobs, baseFlow, peaks_obs):
    # First filter condition: The peak value must be greater than the average value,
    peaks_obs_fil = []
    for k in range(len(peaks_obs)):
        peaks_to_mean = qobs[peaks_obs[k]] - np.mean(qobs)
        if peaks_to_mean <= 0:
            continue
        # peaks_to_mean_perc = (qobs[peaks_obs[k]] - np.mean(qobs)) / np.mean(qobs)
        # if peaks_to_mean_perc < 2:
        #     continue
        peaks_obs_fil.append(peaks_obs[k])

    # Second filter condition: merge peaks between the same valley values
    point_less_all = []
    point_more_all = []
    for k in range(len(peaks_obs_fil)):
        point_less, point_more = find_valleys(qobs, baseFlow, peaks_obs_fil[k])
        point_less_all.append(point_less)
        point_more_all.append(point_more)
    # print(point_less_all)
    # print(point_more_all)


    valley_point = pd.DataFrame({"point_less": point_less_all, "point_more": point_more_all})
    # print(valley_point)
    # 1. There is no repeated peak at the valley point
    valley_point_notdup_index = valley_point[~valley_point.duplicated(keep=False)]
    peaks_obs_merge_not_dup = []
    if not valley_point_notdup_index.empty:
        for k in range(len(valley_point_notdup_index)):
            peaks_obs_merge_not_dup.append(peaks_obs_fil[valley_point_notdup_index.index[k]])
        # print(peaks_obs_merge_not_dup)
    # 2. There are repeated peaks in the valley points,
    # and the largest peak is selected as the peak between the valley points.
    valley_point_dup_index = valley_point[valley_point.duplicated(keep=False)]
    peaks_obs_merge_dup = []
    if not valley_point_dup_index.empty:
        valley_point_dup_index = valley_point_dup_index.groupby(list(valley_point_dup_index)).apply(
            lambda x: tuple(x.index)).tolist()
        # print(valley_point_dup_index)

        for k in range(len(valley_point_dup_index)):
            peaks_max = 0
            peaks_index_final = peaks_obs_fil[valley_point_dup_index[k][0]]
            for j in range(len(valley_point_dup_index[k])):
                peaks_temp = qobs[peaks_obs_fil[valley_point_dup_index[k][j]]]
                if peaks_temp > peaks_max:
                    peaks_max = peaks_temp
                    peaks_index_final = peaks_obs_fil[valley_point_dup_index[k][j]]
            peaks_obs_merge_dup.append(peaks_index_final)
    # print(peaks_obs_merge_not_dup)
    # print(peaks_obs_merge_dup)

    peaks_obs_merge = peaks_obs_merge_not_dup + peaks_obs_merge_dup
    return peaks_obs_merge


def refine_valley(qobs, baseFlow, peaks_obs_merge, peaks_diff_threshold=2.5, peak_interval_threshold=14):
    point_less_final_all = []
    point_more_final_all = []
    diff_peak_all = []
    multi_peak_all = []

    for k in range(len(peaks_obs_merge)):
        point_less_temp, point_more_temp = find_valleys(qobs, baseFlow, peaks_obs_merge[k])
        point_more_final_all.append(point_more_temp)
        point_less_final_all.append(point_less_temp)
        if k == 0:
            diff_peak = 0
            multi_peak = 0
        else:
            diff_peak = qobs[peaks_obs_merge[k - 1]] - qobs[peaks_obs_merge[k]]
            multi_peak = qobs[peaks_obs_merge[k - 1]] / qobs[peaks_obs_merge[k]]
        diff_peak_all.append(diff_peak)
        multi_peak_all.append(multi_peak)

    df_point = pd.DataFrame({'less': point_less_final_all, 'more': point_more_final_all,
                             "diff": diff_peak_all, "multi": multi_peak_all})

    for index, row in df_point.iterrows():
        if index >= 1 and index < len(df_point):
            prev_peak_index = peaks_obs_merge[index - 1]
            current_peak_index = peaks_obs_merge[index]
            peak_interval = current_peak_index - prev_peak_index

            if (row['less'] <= df_point.loc[index - 1, 'more']
                    and row["diff"] > 0
                    and row['multi'] > peaks_diff_threshold  # Customizable threshold for multi_peak
                    and peak_interval <= peak_interval_threshold):  # Customizable threshold for peak_interval
                df_point.loc[index - 1, 'more'] = np.nan
                df_point.loc[index, 'less'] = np.nan

    point_less_final_update = df_point["less"].values.tolist()
    point_more_final_update = df_point["more"].values.tolist()

    mask_less = np.isnan(point_less_final_update)
    mask_more = np.isnan(point_more_final_update)
    point_less_final_update = list(np.array(point_less_final_update)[~mask_less])
    point_more_final_update = list(np.array(point_more_final_update)[~mask_more])

    return point_less_final_update, point_more_final_update


def flood_separate(filePath, savePath, catName, area_cat, data,
                   yarly_check=True, peak_height=None, calculate_baseflow=True,
                   qb_threshold=0.5, Qdiff_threshold=0.005, peaks_diff_threshold=2.5, peak_interval_threshold=14):
    savefile = savePath + catName + "/"
    mkdir(savefile)
    error_log = []

    data.index = pd.to_datetime(data.index)
    data["date"] = data.index

    # Extract year, month, and compute the water year based on lowest flow month
    data['Year'] = data['date'].dt.year
    data['Month'] = data['date'].dt.month
    monthly_avg_runoff = data.groupby('Month')['runoff_mmd'].mean()
    start_month = monthly_avg_runoff.idxmin()
    data['water_year'] = data['date'].apply(lambda x: assign_local_water_year(x, start_month))
    print(f"The start month for the water year of this catchment is {start_month}.")

    # Filter years with fewer missing values
    missing_counts = data.groupby('water_year')['runoff_mmd'].apply(lambda x: x.isna().sum())
    valid_years = missing_counts[missing_counts <= 10].index
    for year in valid_years:
        year_data_index = data['water_year'] == year
        data.loc[year_data_index, 'runoff_mmd'] = data.loc[year_data_index, 'runoff_mmd'].interpolate()

    data_fil = data[data['water_year'].isin(valid_years)]
    baseflow_successful = False

    Q = np.array(data_fil["runoff_mmd"])
    date = data_fil.index.tolist()
    # Default peak height to 90th percentile if not provided
    if peak_height is None:
        peak_height = np.percentile(Q, 90)

    # If baseflow calculation is enabled by the user
    if calculate_baseflow:
        # Check if baseFlow column exists and if all values are non-null
        if 'baseFlow' in data_fil.columns and data_fil['baseFlow'].notna().all():
            print("BaseFlow column exists and has no missing values. Skipping baseflow calculation.")
            baseflow_successful = True
        else:
            print("BaseFlow column missing or contains null values. Performing baseflow separation.")

            try:
                test_b, test_KGEs = baseflow.separation(Q, date, area=area_cat)
                final_b, final_KGEs = baseflow.separation(Q, date, area=area_cat,
                                                          method=test_b.dtype.names[test_KGEs.argmax()])
                final_b = final_b.view('<f8')
                data_fil["baseFlow"] = final_b
                baseflow_successful = True
            except Exception as e:
                error_log.append((catName, str(e)))
                data_fil["baseFlow"] = np.nan
    else:
        print("Baseflow calculation is disabled by the user.")
        if 'baseFlow' in data_fil.columns and data_fil['baseFlow'].notna().all():
            baseflow_successful = True

    data_grouped = data_fil.groupby(data_fil['water_year'])

    for year, data_each_year in data_grouped:
        try:
            qobs = np.array(data_each_year["runoff_mmd"])
            date = np.array(data_each_year["date"])

            if baseflow_successful and "baseFlow" in data_each_year.columns:
                baseFlow = np.array(data_each_year["baseFlow"])
                data_each_year["QbRatio"] = data_each_year["baseFlow"] / data_each_year["runoff_mmd"]
                QbRatio_mean = data_each_year["QbRatio"].mean()
                if QbRatio_mean > qb_threshold:  # Use custom threshold
                    qobs_smooth = savgol_filter(qobs, 9, 2)
                    peaks_obs, _ = find_peaks(qobs, height=peak_height, distance=5)
                    valley_obs, _ = find_peaks(-qobs_smooth, height=np.percentile(-qobs_smooth, 50), distance=5)
                    valley_obs = valley_obs.tolist()
                    valley_obs.append(0)
                    valley_obs.append(len(qobs) - 1)
                    valley_obs.sort()

                    peaks_obs_merge = refine_peak_qb90(qobs, peaks_obs, valley_obs)
                    point_less_final_update, point_more_final_update = refine_valley_qb90(qobs, peaks_obs_merge,
                                                                                          valley_obs,
                                                                                          peaks_diff_threshold,
                                                                                          peak_interval_threshold)
                    point_less_final_update = [int(item) for item in point_less_final_update]
                    point_more_final_update = [int(item) for item in point_more_final_update]
                else:
                    peaks_obs, _ = find_peaks(qobs, height=peak_height, distance=5)
                    peaks_obs_merge = refine_peak(qobs, baseFlow, peaks_obs)
                    point_less_final_update, point_more_final_update = refine_valley(qobs, baseFlow,
                                                                                     peaks_obs_merge,
                                                                                     peaks_diff_threshold,
                                                                                     peak_interval_threshold)
            else:
                qobs_smooth = savgol_filter(qobs, 9, 2)
                peaks_obs, _ = find_peaks(qobs, height=peak_height, distance=5)
                valley_obs, _ = find_peaks(-qobs_smooth, height=np.percentile(-qobs_smooth, 50), distance=5)
                valley_obs = valley_obs.tolist()
                valley_obs.append(0)
                valley_obs.append(len(qobs) - 1)
                valley_obs.sort()

                peaks_obs_merge = refine_peak_qb90(qobs, peaks_obs, valley_obs)
                point_less_final_update, point_more_final_update = refine_valley_qb90(qobs, peaks_obs_merge,
                                                                                      valley_obs,
                                                                                      peaks_diff_threshold,
                                                                                      peak_interval_threshold)
                point_less_final_update = [int(item) for item in point_less_final_update]
                point_more_final_update = [int(item) for item in point_more_final_update]

            def plot_check():
                plt.title(year)
                plt.plot(date, qobs, color="gray")
                plt.plot(date[point_less_final_update], qobs[point_less_final_update], "*", color="yellow", ms=10)
                plt.plot(date[point_more_final_update], qobs[point_more_final_update], "*", color="orange", ms=10)
                plt.axhline(y=peak_height, color='blue', linestyle='-')
                # plt.axhline(y=np.percentile(qobs, 95), color='cyan', linestyle='-')
                plt.savefig(savefile + str(year) + ".jpg")
                plt.show()

            if yarly_check == True:
                plot_check()

            for k in range(len(point_less_final_update)):
                point_less_final = int(point_less_final_update[k])
                point_more_final = int(point_more_final_update[k])

                flood_ini = data_each_year.iloc[point_less_final: point_more_final]
                if len(flood_ini) <= 3:
                    continue

                flood_number = flood_ini["date"][0].strftime("%Y%m%d")
                flood_ini.to_csv(savefile + flood_number + ".csv")

        except Exception as e:
            error_log.append(f"Error processing year {year} for catchment {catName}: {str(e)}")
            continue

    if error_log:
        with open(filePath + "error_log.txt", "w") as f:
            for log in error_log:
                f.write(log + "\n")

    print("Flood event processing completed. Check error logs for any issues.")