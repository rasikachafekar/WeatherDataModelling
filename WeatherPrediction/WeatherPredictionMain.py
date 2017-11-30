import GetAndCleanData as gc
import ARIMAModelling as am
import pdb

# Generate Time Series
# TimeSeries = gc.fetchDataAndCleanData()
#
# Obtain individual time series
# ghiSeries = TimeSeries[0]
# pdb.set_trace()
# temperatureSeries = TimeSeries[1]
# windSeries = TimeSeries[2]

# Generate optimal parameters for individual series
# ghiParams = am.get_params_pdq(ghiSeries)
# temperatureParams = am.get_params_pdq(temperatureSeries)
# windParams = am.get_params_pdq(windSeries)

# After this persist these values in the database for future retrieval. For now saving it in CSV
# am.save_params([(ghiSeries.name, ghiParams), (temperatureSeries.name, temperatureParams), (windSeries.name, windParams)])
# print(am.get_params(ghiSeries))
# Generate the model and predict the weather

# pdb.set_trace()
# ghiParams = am.get_params(ghiSeries)
# temperatureParams = am.get_params(temperatureSeries)
# windParams = am.get_params(windSeries)
#
# ghiPred = am.predict_values(ghiParams, ghiSeries)

# tempPred = am.predict_values(temperatureParams, temperatureSeries)
# windPred = am.predict_values(windParams, windSeries)

ghiPred = am.get_predicted_values('GHI')
print(ghiPred)
# print(tempPred)
# print(windPred)


