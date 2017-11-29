import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import itertools
import numpy as np
import warnings
import math
import pdb

def removeOutliers(SomeDF):
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
    # print(SomeDF_final)
    SomeSeries = SomeDF_final.groupby(['Date'])[paramColumnName].mean()
    # print (SomeSeries.index.freq)
    # print(SomeDF_final.describe())
    # SomeSeries = SomeDF_final.groupby(['Date'])['GHI'].mean()
    # print(SomeSeries.describe() )
    # SomeSeriesIndexed = SomeSeries.set_index('Date')
    SomeSeries.index = add_freq(SomeSeries, freq = 'D')
    # print(SomeSeries.index.freq)
    # pdb.set_trace()

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
def getParamspdq(SomeSeries) :
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    warnings.filterwarnings("ignore")
    aic = 10000000

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(SomeSeries,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                if(aic > results.aic):
                    aic = results.aic
                    params = (param,param_seasonal)
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

            except:
                continue
    return params
def get_model(params, SomeSeries, colString):
    SomeSeries.reindex()
    print('___________________######################_________________')
    print(SomeSeries.index.freq)
    print('___________________######################_________________')

    mod = sm.tsa.statespace.SARIMAX(SomeSeries,
                                        order=params[0],
                                        seasonal_order=params[1],
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)

    results = mod.fit()
    print(type(results))

    results.save('test')
    # results.save(SomeSeries.name+'.pkl')
    # print(results.summary().tables[1])

    # results.plot_diagnostics(figsize=(15,12))
    # plt.show()

    pred = results.get_prediction(start=pd.to_datetime('2010-01-01'), dynamic=False)
    # pred_ci = pred.conf_int()
    #
    # ax = SomeSeries['2005':].plot(label='observed')
    # pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
    #
    # ax.fill_between(pred_ci.index,
    #                 pred_ci.iloc[:, 0],
    #                 pred_ci.iloc[:, 1], color='k', alpha=.2)
    #
    # ax.set_xlabel('Date')
    # ax.set_ylabel(colString)
    # plt.legend()
    #
    # plt.show()
    # pdb.set_trace()
    forecasted = pred.predicted_mean
    truth = SomeSeries['2010-01-01':]

    # print(truth)
    # print(forecasted)
    # pred_uc = results.get_forecast(steps=100)
    # print(results.summary())
    # pdb.set_trace()
    # pred_uc = results.get_forecast(SomeSeries.last_valid_index() + pd.DateOffset(1))
    pred_uc = results.get_forecast(steps=365)
    #
    # print(forecasted.head())
    # print(SomeSeries['GHI'].head())
    # pred_uc = results.get_forecast('2014-10-12')
    # pred_ci = pred_uc.conf_int()
    # ax = SomeSeries.plot(label='observed', figsize=(20, 15))
    # pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
    # ax.fill_between(pred_ci.index,
    #                 pred_ci.iloc[:, 0],
    #                 pred_ci.iloc[:, 1], color='k', alpha=.25)
    # ax.set_xlabel('Date')
    # ax.set_ylabel('GHI')
    #
    # plt.legend()
    # plt.show()

    # Compute the mean square error
    rmse = math.sqrt(((forecasted - truth) ** 2).mean())
    return rmse

# load dataset from multiple csv file into one dataframe
df = pd.DataFrame()
list_ = []
for i in range(2001,2011):
    tempDataframe = pd.read_csv("Datasets/"+ str(i) + ".csv", skiprows = 2, usecols = [0,1,2,3,4,7,9,14])
    list_.append(tempDataframe)
df = pd.concat(list_)

#Splitting into 3 dataframes ghiDF, temperatureDF, windSpeedDF

# ghiDF = df.iloc[:,0:6]
# ghiDF = ghiDF[ghiDF.GHI!=0]

# ghiSeries = removeOutliers(ghiDF)
# print(ghiSeries.head())
temperatureDF = df.iloc[:,0:7]
del temperatureDF['GHI']
temperatureSeries = removeOutliers(temperatureDF)
print(temperatureSeries.last_valid_index() + pd.DateOffset(1))
# print(temperatureSeries.name)
# print(temperatureSeries.head())
# windSpeedDF = df.iloc[:,0:8]
# del windSpeedDF['GHI']
# del windSpeedDF['Temperature']
# print(windSpeedDF.head())
#Model prediction for GHI
# params_GHI = getParamspdq(ghiSeries)
# print(params_GHI)
# rmse = get_model(params_GHI,ghiSeries,"GHI")
# rmse = get_model(((1, 1, 1), (0, 1, 1, 12)),ghiSeries,"GHI")
# rmse = get_model(((1, 1, 1), (0, 1, 1, 12)),ghiSeries,ghiSeies.name)
# print('The Root Mean Squared Error of our forecasts for GHI with current model is {}'.format(round(rmse, 2)))
# #
# # #Model prediction for Temperature
# # params_Temperature = getParamspdq(temperatureSeries)
# # rmse = get_model(params_Temperature,temperatureSeries,"Temperature")
rmse = get_model(((1, 0, 1), (0, 1, 1, 12)),temperatureSeries,"Temperature")
print(rmse)
#
# # Model prediction for Wind Speed
# # params_WindSpeed = getParamspdq(windSpeedSeries)
# # print(params_WindSpeed)
# #rmse = get_model(params_WindSpeed,windSpeedSeries,"WindSpeed")
# rmse = get_model(((1, 0, 1), (0, 1, 1, 12)),windSpeedSeries,"WindSpeed")
# print('The Root Mean Squared Error of our forecasts for Temperature with current model is {}'.format(round(rmse, 2)))
#
#

#print(ghiSeries.head(20))
# line plot of dataset
#ghiSeries.plot()
#plt.legend()
# temperatureSeries.plot()
# windSpeedSeries.plot()
#plt.show()
