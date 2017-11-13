import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import itertools
import numpy as np
import warnings
import math


# load dataset from multiple csv file into one dataframe
df = pd.DataFrame()
list_ = []
for i in range(2001,2011):
	tempDataframe = pd.read_csv("Datasets/"+ str(i) + ".csv", skiprows = 2, usecols = [0,1,2,3,4,7,9,14])
	list_.append(tempDataframe)
df = pd.concat(list_)

#Splitting into 3 dataframes ghiDF, temperatureDF, windSpeedDF
ghiDF = df.iloc[:,0:6]

temperatureDF = df.iloc[:,0:7]
del temperatureDF['GHI']

windSpeedDF = df.iloc[:,0:8]
del windSpeedDF['GHI']
del windSpeedDF['Temperature']

#Removing rows from ghiDF with 0 GHI value
ghiDF = ghiDF[ghiDF.GHI!=0]

#Concatenating Year, Month and Day columns to form one single column called Date
ghiDF['Date'] = ghiDF['Year'].astype(str) + '-' + ghiDF['Month'].astype(str) + '-' + ghiDF['Day'].astype(str)
ghiDF['Date'] = pd.to_datetime(ghiDF['Date'])
del ghiDF['Year']
del ghiDF['Month']
del ghiDF['Day']
del ghiDF['Hour']
del ghiDF['Minute']

#calculating mean of GHI for each day and creating a panda series
ghiSeries = ghiDF.groupby(['Date'])['GHI'].mean()

#Model prediction
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# warnings.filterwarnings("ignore")
# aic = 10000000
#
# for param in pdq:
# 	for param_seasonal in seasonal_pdq:
# 		try:
# 			mod = sm.tsa.statespace.SARIMAX(ghiSeries,
# 											order=param,
# 											seasonal_order=param_seasonal,
# 											enforce_stationarity=False,
# 											enforce_invertibility=False)
#
# 			results = mod.fit()
# 			if(aic > results.aic):
# 				aic = results.aic
# 				params = (param,param_seasonal)
# 			print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#
# 		except:
# 			continue
# print(params)

mod = sm.tsa.statespace.SARIMAX(ghiSeries,
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

#print(results.summary().tables[1])

# results.plot_diagnostics(figsize=(15,12))
# plt.show()

pred = results.get_prediction(start=pd.to_datetime('2010-01-01'), dynamic=False)
pred_ci = pred.conf_int()


ax = ghiSeries['2005':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('GHI')
plt.legend()

#plt.show()

GHI_forecasted = pred.predicted_mean
GHI_truth = ghiSeries['2010-01-01':]

# Compute the mean square error
mse = math.sqrt(((GHI_forecasted - GHI_truth) ** 2).mean())
print('The Root Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


pred_uc = results.get_forecast(steps = 500)
pred_ci = pred_uc.conf_int()


ax = ghiSeries.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('GHI')

plt.legend()
plt.show()
#print(ghiSeries.head(20))
# line plot of dataset
#ghiSeries.plot()
#pyplot.show()