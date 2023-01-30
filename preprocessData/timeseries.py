
#Check if the model is stationaty using the Dickey-Fuller test 
#NEED A STATIONARY TIME SERIES TO USE ARMA

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd 
import matplotlib.pyplot as plt  
from statsmodels.tsa.seasonal import seasonal_decompose



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
    
    if pval < crit_vals[0]:
        print('H0 rejected with a significance level of less than 1%')
    elif pval < crit_vals[1] and pval > crit_vals[0]:
        print('H0 rejected with a significance level of less than 5%')
    elif pval < crit_vals[2] and pval > crit_vals[1]:
        print('H0 rejected with a significance level of less than 10%')
    else:
        print('Time series is non-stationary')


def SeasonalityCheck(dfseries):
    result =  seasonal_decompose(dfseries, model='additive',period=300)

    # show the results
    result.plot()
    plt.show()


#ARIMA MODEL

def FitARIMA(dfseries,plot=False):
    model = ARIMA(dfseries, order=(1,1,1))
    modelfit = model.fit()
    print(modelfit.summary())  
    
    # Plotting Residual Errors  
    #There is a bias in the prediction (a non-zero mean in the residuals)
    myresiduals = pd.DataFrame(modelfit.resid) 
    if plot:
        fig, ax = plt.subplots(1,2)  
        myresiduals.plot(title = "Residuals", ax = ax[0])  
        myresiduals.plot(kind = 'kde', title = 'Density', ax = ax[1])  
        plt.show()  
    
    # Actual vs Fitted  
    # modelfit.plot_predict(dynamic=True, plot_insample=False) # AttributeError: 'ARIMAResults' object has no attribute 'plot_predict'
    # plt.show()
         
    return myresiduals
    
    
    
    


