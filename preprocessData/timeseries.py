
#Check if the model is stationaty using the Dickey-Fuller test 
#NEED A STATIONARY TIME SERIES TO USE ARMA

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd 
import matplotlib.pyplot as plt  
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss



#(H0): Time series is non-stationary (p< crit. value to reject)

def DickeyFuller(dfseries):
    
    
    #need to get the data series to test (LPA)
    adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(dfseries.values) 

    print('ADF test statistic:', adf)
    print('ADF p-values:', pval)
    print('ADF number of lags used:', usedlag)
    print('ADF number of observations:', nobs)
    print('ADF critical values:', crit_vals)
    print('ADF best information criterion:', icbest)
    
    if pval < 0.05:
        if adf < crit_vals['1%']:
            print('H0 rejected with a significance level of less than 1%')
            print('i.e. Time series is stationary')
        elif adf < crit_vals['5%'] and adf > crit_vals['1%']:
            print('H0 rejected with a significance level of less than 5%')
            print('i.e. Time series is stationary')
        elif adf < crit_vals['10%'] and adf > crit_vals['5%']:
            print('H0 rejected with a significance level of less than 10%')
            print('i.e. Time series is stationary')
        else:
            pass
    else:
        print('Time series is non-stationary')


def kpss_test(dfseries,**kw):
    statistic, p_value, n_lags, critical_values = kpss(dfseries, **kw)
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critical Values:')
    for key, value in critical_values.items():
        print(f'    {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')


def SeasonalityCheck(dfseries):
    result =  seasonal_decompose(dfseries, model='additive',period=300)

    # show the results
    result.plot()
    plt.show()


#ARIMA MODEL

def FitARIMA(dfseries,plot=False):
    # should both return stationary and proves that d=1
    kpss_test(dfseries.diff().dropna())
    DickeyFuller(dfseries.diff().dropna())

    model = ARIMA(dfseries, order=(1,1,3))
    modelfit = model.fit()
    # print(modelfit.summary())  

    # Plotting Residual Errors  
    #There is a bias in the prediction (a non-zero mean in the residuals)
    myresiduals = pd.DataFrame(modelfit.resid) 
    if plot:
        # shows what p and q values we should use for ARIMA model
        plot_acf(dfseries.diff().dropna())
        plt.show()
        plot_pacf(dfseries.diff().dropna())
        plt.show()
        
        fig, ax = plt.subplots(1,2)  
        myresiduals.plot(title = "Residuals", ax = ax[0])  
        myresiduals.plot(kind = 'kde', title = 'Density', ax = ax[1])  
        plt.show()  
    
    # Actual vs Fitted
    model_values = modelfit.predict()
    
    return model_values,myresiduals
    
    
    
    


