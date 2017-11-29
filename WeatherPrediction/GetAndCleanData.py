import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import itertools
import numpy as np
import warnings
import math


def removeOutliers(SomeDF):
    """
    Cleans the input dataframe by calculating mean and standard deviations and removes rows which lie outside 1 std.
    :param SomeDF:
    :return:Pandas series with frequency of one day
    """
    paramColumnName = SomeDF.columns[5]
    SomeDF_mean = pd.DataFrame({'mean_t': SomeDF.groupby(['Year', 'Month'])[paramColumnName].mean()}).reset_index()
    SomeDF_std = pd.DataFrame({'std_t': SomeDF.groupby(['Year', 'Month'])[paramColumnName].std()}).reset_index()
    SomeDF_mean_std = pd.merge(SomeDF_mean, SomeDF_std, how='inner', left_on=['Year', 'Month'], right_on=['Year', 'Month'])
    SomeDF_final = pd.merge(SomeDF, SomeDF_mean_std, how='left', left_on=['Year', 'Month'], right_on=['Year', 'Month'])

    def funk(row):
        # pdb.set_trace()
        if (float(row.iloc[5]) > (row.mean_t - row.std_t)) & (float(row.iloc[5]) < (row.mean_t + row.std_t)):
            return 1
        else:
            return 0

    SomeDF_final['bool'] = SomeDF_final.apply(lambda row: funk(row), axis=1)
    SomeDF_final = SomeDF_final[SomeDF_final["bool"] == 1]

    SomeDF_final['Date'] = SomeDF_final['Year'].astype(str) + '-' + SomeDF_final['Month'].astype(str) + '-' + SomeDF_final['Day'].astype(str)
    SomeDF_final['Date'] = pd.to_datetime(SomeDF_final['Date'])
    del SomeDF_final['Year']
    del SomeDF_final['Month']
    del SomeDF_final['Day']
    del SomeDF_final['Hour']
    del SomeDF_final['Minute']
    del SomeDF_final['mean_t']
    del SomeDF_final['std_t']
    del SomeDF_final['bool']
    SomeDF_final = SomeDF_final.set_index('Date')
    SomeSeries = SomeDF_final.groupby(['Date'])[paramColumnName].mean()
    SomeSeries.index = add_freq(SomeSeries, freq='D')

    return SomeSeries


def add_freq(idx, freq=None):
    """Add a frequency attribute to idx, through inference or directly.

    Returns a copy.  If `freq` is None, it is inferred.
    """

    idx = idx.copy()
    if freq is None:
        if idx.index.freq is None:
            freq = pd.infer_freq(idx.index)
        else:
            return idx
    idx.index.freq = pd.tseries.frequencies.to_offset(freq)
    if idx.index.freq is None:
        raise AttributeError('no discernible frequency found to `idx`.  Specify'
                             ' a frequency string with `freq`.')
    return idx.index


def fetchDataAndCleanData():
    """
    Fetch the downloaded data from file storage and clean by calling removeOutliers().
    :return:
    List[ghiSeries, temperatureSeries, windSpeedSeries]
    """
    list_ = []
    for i in range(2001, 2011):
        tempDataframe = pd.read_csv("Datasets/" + str(i) + ".csv", skiprows=2, usecols=[0, 1, 2, 3, 4, 7, 9, 14])
        list_.append(tempDataframe)
    df = pd.concat(list_)

    # Splitting into 3 dataframes ghiDF, temperatureDF, windSpeedDF

    ghiDF = df.iloc[:,0:6]
    ghiDF = ghiDF[ghiDF.GHI!=0]
    ghiSeries = removeOutliers(ghiDF)

    temperatureDF = df.iloc[:, 0:7]
    del temperatureDF['GHI']
    temperatureSeries = removeOutliers(temperatureDF)

    windSpeedDF = df.iloc[:,0:8]
    del windSpeedDF['GHI']
    del windSpeedDF['Temperature']
    windSpeedSeries = removeOutliers(windSpeedDF)

    return [ghiSeries, temperatureSeries, windSpeedSeries]


def fetchDataNOAA():

    #TODO: API call to NOAA or NREL will go here

    return
