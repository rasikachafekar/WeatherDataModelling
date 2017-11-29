import GetAndCleanData as gc
import ARIMAModelling as am


# Generate Time Series
TimeSeries = gc.fetchDataAndCleanData()

# Obtain individual time series
ghiSeries = TimeSeries[0]
temperatureSeries = TimeSeries[1]
windSeries = TimeSeries[2]

# Generate optimal parameters for individual series
ghiParams = am.get_params_pdq(ghiSeries)
temperatureParams = am.get_params_pdq(temperatureSeries)
windParams = am.get_params_pdq(windSeries)

# After this persist these values in the database for future retrieval. For now saving it in CSV
am.save_params([(ghiSeries.name, ghiParams), (temperatureSeries.name, temperatureParams), (windSeries.name, windParams)])
print(am.get_params(ghiSeries))
# Generate the model and predict the weather

ghiPred = am.predict_values(ghiParams, ghiSeries)
tempPred = am.predict_values(temperatureParams, temperatureSeries)
windPred = am.predict_values(windParams, windSeries)

print(ghiPred.head())
print(tempPred.head())
print(windPred.head())

