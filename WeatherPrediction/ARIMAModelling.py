import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import itertools
import numpy as np
import warnings



def get_params_pdq(SomeSeries) :
    """
    Generates optimal p,d,q values for the input series.
    :param SomeSeries:
    :return: p,d,q parameters for ARIMA model
    """
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


def predict_values(params, SomeSeries):
    """
    Currently unable to save the generated model to file. (https://github.com/statsmodels/statsmodels/issues/3401).
    Alternate plan: Save the optimal params to DB. Fetch the params when required and predict the values. Also, this
    method will generate a predicted time series. This can also be persisted in a file.

    return: panda.series of predicted values.
    """
    mod = sm.tsa.statespace.SARIMAX(SomeSeries,
                                    order=params[0],
                                    seasonal_order=params[1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    results = mod.fit()

    pred_uc = results.get_forecast(SomeSeries.last_valid_index() + pd.DateOffset(1))

    # Not sure if we need to calculate rmse.
    # rmse = math.sqrt(((forecasted - truth) ** 2).mean())
    return pred_uc.predicted_mean


def save_params(params):
    df = pd.DataFrame(params)
    df.to_csv('test.csv', index=False, header=False)


def get_params(SomeSeries):
    df = pd.read_csv('test.csv', header=None)
    return df.loc[df[0] == SomeSeries.name].iloc[0][1]