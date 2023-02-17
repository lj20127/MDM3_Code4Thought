from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Literal, get_args, get_origin
from mlxtend.preprocessing import minmax_scaling
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss
from scipy.signal import savgol_filter
from operator import attrgetter
from statistics import median
from functools import reduce
from sys import _getframe
import matplotlib.pyplot as plt
import scipy.stats as stats
import ruptures as rpt
import random as rnd
import pandas as pd
import numpy as np
import pickle
import os

import warnings
warnings.filterwarnings('ignore')


def mape(y_test, y_pred):
    return np.mean(np.abs((y_test-y_pred)/y_test)*100)


def calculate_confidence_intervals(random_walk_data, num_samples, confidence_level=0.05):
    N, T = random_walk_data.shape
    lower_bounds = np.zeros((N, T))
    upper_bounds = np.zeros((N, T))
    for t in range(T):
        endpoint = random_walk_data[:, t]
        endpoint_mean = np.mean(endpoint)
        endpoint_var = np.var(endpoint)
        endpoint_stddev = np.sqrt(endpoint_var)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_bound = endpoint_mean - z_score * endpoint_stddev
        upper_bound = endpoint_mean + z_score * endpoint_stddev
        lower_bounds[:, t] = lower_bound
        upper_bounds[:, t] = upper_bound
    return lower_bounds, upper_bounds


def enforce_literals(function):
    kwargs = _getframe(1).f_locals
    for name, type_ in function.__annotations__.items():
        value = kwargs.get(name)
        options = get_args(type_)
        if get_origin(type_) is Literal and name in kwargs and value not in options:
            raise AssertionError(f"'{value}' is not in {options} for '{name}'")


def load_project_variable(path, project_type="full", variable_type: Literal["model", "train", "merged_train", "test", "pred", "error", "mean_error", "rw_model", "arima_model"] = "rw_model", version="latest"):
    enforce_literals(load_project_variable)
    is_model = True if (variable_type in ["model", "rw_model", "arima_model"]) else False
    is_model_bool = (lambda f: "model" in f) if is_model else (lambda f: "model" not in f)
    versions = [int(file.split(".")[0].split("_")[-1]) for file in os.listdir(path) if (file.endswith(".pkl") and file.startswith(project_type) and (is_model_bool(file)))]
    if (type(version) == int) and (version not in versions):
        print(f"Version: {version} does not exist!\nPlease specify an existing verison.")
        return
    
    latest = max(versions) if len(versions)!=0 else 0
    latest = latest if (type(version) != int) else version

    file_path = f"{project_type}_{variable_type}_{latest}.pkl" if len(project_type) else f"{variable_type}_{latest}.pkl"
    with open(os.path.join(path, file_path), "rb") as file:
        out = pickle.load(file)

    return out



def detect_change_pts(model,n):
    rpt_model = rpt.Binseg(model="rbf").fit(model)
    change_pts = rpt_model.predict(n_bkps=n)
    return change_pts

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
    if p_value > 0.05:
        print(f"Result: The series is stationary")
    else:
        print(f'Result: The series is not stationary')


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
    print(modelfit.summary())
    print(modelfit.conf_int())

    # Plotting Residual Errors  
    #There is a bias in the prediction (a non-zero mean in the residuals)
    myresiduals = pd.DataFrame(modelfit.resid) 
    if plot:
        # shows what p and q values we should use for ARIMA model
        fig, ax = plt.subplots(2,1)  
        plot_acf(dfseries.diff().dropna(),ax=ax[0])
        ax[0].set_title("Autocorrelation",fontdict={'fontsize': 25})
        plot_pacf(dfseries.diff().dropna(),ax=ax[1])
        ax[1].set_title("Partial Autocorrelation",fontdict={'fontsize': 25})
        plt.show()
        
        fig, ax = plt.subplots(1,2) 
        myresiduals.plot(title = "Residuals", ax = ax[0])  
        myresiduals.plot(kind = 'kde', title = 'Density', ax = ax[1])  
        plt.show()  
    
    # Actual vs Fitted
    model_values = modelfit.predict()
    
    return model_values,myresiduals


def merge_dataframes(dfs):
    # creates a list for lpa for every json file
    lpa_lists = [[] for i in range(0,len(dfs))]
    first_week = [[] for i in range(0,len(dfs))]
    i = 0
    for df in dfs:
        # filters dataframe
        filtered_df = filter_dataframe(df)
        # gets lines per author per week 
        new_df = lines_per_author(filtered_df)

        lpa_lists[i] = np.array(new_df["week_linesperauthor"])
        first_week[i] = lpa_lists[i][0]
        i+=1

    # removes any lpa lists that are empty
    lpa_lists = [lst for lst in lpa_lists if lst != []]

    # merges data
    merged,ci,num_weeks = merge_lists(lpa_lists)
    weeks = np.linspace(0,num_weeks,num_weeks)

    # filtered data using Savitzky-Golay filter
    merged_filtered = savgol_filter(x=merged,window_length=50,polyorder=2)

    # converts model to dataframe for use with FitARIMA
    model_df = pd.DataFrame(data={'model':merged_filtered})
    
    return model_df


def merge_lists(lists): 
    maxlen = max([len(lst) for lst in lists])
    model = np.linspace(0,1,maxlen)
    interped_lists = [[] for lst in lists]
    i=0
    for lst in lists:
        interped_lists[i] = np.interp(model,np.linspace(0, 1, len(lst)),lst)
        i+=1

    # finds confidence interval of each week
    confidence_interval = [0.8 * np.std(i) / np.sqrt(len(i)) for i in zip(*interped_lists)]
    # finds median of each week
    merged = [median(i) for i in zip(*interped_lists)]
    return merged,confidence_interval,maxlen


def calc_IQR(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    return IQR


def scale_dataframe(dfs):
    lpa_dfs = dfs.copy()
    if "week_linesperauthor" not in dfs[0].columns:
        lpa_dfs = [lines_per_author(df) for df in lpa_dfs]
    
    longest_project = max([df.shape[0] for df in lpa_dfs])
    
    lpa_dfs = [pd.concat([df, pd.DataFrame(np.full((longest_project - df.shape[0], 2), np.nan), columns=['week', 'week_linesperauthor'])], ignore_index=True) for df in lpa_dfs if df.shape[0] < longest_project]
    
    for df in lpa_dfs:
        df["week"] = df.index.to_list()
        df["week_linesperauthor"].iloc[-1] = df["week_linesperauthor"].quantile(0.05)
        df.interpolate(method="linear", inplace=True)
        df["week_linesperauthor_scaled"] = minmax_scaling(df, columns="week_linesperauthor")
    
    variances = [df["week_linesperauthor_scaled"].var() for df in lpa_dfs]
    weights = np.array([1/var for var in variances]) / sum(np.array([1/var for var in variances]))
    combined = [lpa_dfs[i].week_linesperauthor_scaled*weights[i] for i in range(len(lpa_dfs))]
    combined = reduce(lambda a, b: a.add(b, fill_value=0), combined)
    
    original_IQR = np.median([calc_IQR(df.week_linesperauthor) for df in lpa_dfs])
    original_median = np.median([df.week_linesperauthor.median() for df in lpa_dfs])

    combined_median = combined.median()
    combined_IQR = calc_IQR(combined)

    combined_scaled = (((combined - combined_median)/(combined_IQR)) * original_IQR)+original_median
    
    return combined_scaled


def calc_probs(dfs):
    lpa_dfs = [lines_per_author(df, timeframe="week") for df in dfs]
    lpa_dfs_probs = lpa_dfs.copy()
    for df in lpa_dfs_probs:
        if not "lpa_diff" in df.columns:
            df["lpa_diff"] = df["week_linesperauthor"]
            df.loc[df["lpa_diff"].notna(), "lpa_diff"] = df[df.week_linesperauthor.notna()].diff(periods=1)
        conditions = [
            (df['lpa_diff'] < 0),
            (df['lpa_diff'] > 0),
            (df['lpa_diff'] == 0),
            (df['lpa_diff'].isna())
        ]
        values = [-1, 1, 0, np.nan]
        df['updown'] = np.select(conditions, values)
        
    return lpa_dfs_probs

    
def get_probs_combined(dfs):
    
    scaled_df = scale_dataframe(dfs)
    
    probs_df = calc_probs(dfs)
    
    updown_dfs = [pd.DataFrame({"updown":df.updown, "updown_size":df.updown.abs()}) for df in probs_df]
    updown_total_df = reduce(lambda a, b: a.add(b, fill_value=0), updown_dfs)
    updown_total_df = updown_total_df.loc[:updown_total_df.loc[updown_total_df["updown_size"] < 2].iloc[0].name - 1]
    
    updown_total_df["prob"] = updown_total_df.updown / updown_total_df.updown_size
    updown_total_df["prob"] = updown_total_df["prob"].apply(lambda x: map_range(x, -1, 1, 1, 0))
    
    scaled_df = scaled_df.iloc[:updown_total_df.shape[0]]
    
    combined_scaled = pd.DataFrame({"week":scaled_df.index, "lpa":scaled_df.values, "updown":updown_total_df.updown, "updown_size":updown_total_df.updown_size, "prob":updown_total_df.prob})
    
    combined_scaled["lpa_diff"] = scaled_df.diff(periods=1).abs()
    
    return combined_scaled


def read_data_to_dataframes(file_path, read_n_files=None, columns_to_keep=["date", "day", "month", "year", "author", "addedloc", "deletedloc"]):
    file_extension = os.path.splitext(os.listdir(file_path)[0])[1]
    if file_extension not in [".json", ".parquet"]:
        print("Please specify the correct path to the folder containing the data with file extensions of either '.json' or '.parquet'.")
        return
        
    file_paths = [os.path.join(file_path, name) for name in os.listdir(file_path)]
    file_paths = file_paths if (read_n_files is None) else rnd.sample(file_paths, read_n_files)

    method = pd.read_json if (file_extension == ".json") else pd.read_parquet
    dataframes = [method(path) for path in file_paths]
    return dataframes


def clean_dataframe(dataframe):
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    dataframe["day"] = (dataframe.date - dataframe.date.min()).dt.days
    dataframe["week"] = (
        dataframe.date.dt.to_period("W") - dataframe.date.dt.to_period("W").min()
        ).apply(attrgetter('n'))
    dataframe["weekday"] = dataframe.date.dt.weekday
    dataframe["cat_weekday"] = pd.Categorical(
        dataframe.date.dt.strftime("%A"), 
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    dataframe["month"] = (
        dataframe.date.dt.to_period("M") - dataframe.date.dt.to_period("M").min()
        ).apply(attrgetter('n'))
    dataframe["dayofmonth"] = dataframe.date.dt.day
    dataframe["weekofyear"] = dataframe.date.dt.week
    dataframe["monthofyear"] = dataframe.date.dt.month

    dataframe["combinedpath"] = [''.join(l) for l in dataframe.path]
    dataframe.loc[dataframe.combinedpath.str.contains("=>"), ["addedloc", "deletedloc"]] = 0
    dataframe.sort_values(by="date", inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe


def filter_dataframe(dataframe, technology="java"):
    dataframe = dataframe[dataframe.technology==technology]
    return dataframe


def lines_per_author(dataframe, timeframe="week", interp_method="linear"):
    if timeframe not in ["day", "week", "month"]:
        print("Please choose one of the following timeframes: 'day', 'week' or 'month'.")
        return
    email_group = dataframe.groupby(timeframe)["email"]
    unique_emails = email_group.unique().apply(lambda x: len(x))

    lines_added_group = dataframe.groupby(timeframe)["addedloc"]
    lines_per_timeframe = lines_added_group.sum()

    column_name = timeframe + "_linesperauthor"
    
    lines_per_author_dataframe = (lines_per_timeframe / unique_emails).to_frame().reset_index()
    lines_per_author_dataframe.columns = [timeframe, column_name]
    lines_per_author_dataframe.index = lines_per_author_dataframe[timeframe].tolist()
    lines_per_author_dataframe = lines_per_author_dataframe.reindex(np.arange(lines_per_author_dataframe[timeframe].min(), lines_per_author_dataframe[timeframe].max() + 1))
    lines_per_author_dataframe[timeframe].mask(lines_per_author_dataframe[timeframe] == 0, lines_per_author_dataframe.index.to_series(), inplace=True)
    if interp_method:
        lines_per_author_dataframe[column_name] = lines_per_author_dataframe[column_name].interpolate(interp_method)
    lines_per_author_dataframe[timeframe] = lines_per_author_dataframe.index
    return lines_per_author_dataframe


def normalize_column(dataframe, column_name):
    dataframe[column_name + "_norm"] = (
        (dataframe[column_name] - dataframe[column_name].min()) / 
        (dataframe[column_name].max() - dataframe[column_name].min())
    )
    return dataframe


def map_range(x, oldMin, oldMax, newMin, newMax):
    oldRange = (oldMax - oldMin)
    newRange = (newMax - newMin)
    newValue = (((x - oldMin) * newRange) / oldRange) + newMin
    return newValue


def fit_arima(dfseries, p, d, q):
    model = ARIMA(dfseries, order=(p,d,q))
    modelfit = model.fit()
    print(modelfit.summary())  
    
    # Plotting Residual Errors  
    #There is a bias in the prediction (a non-zero mean in the residuals)
    myresiduals = pd.DataFrame(modelfit.resid)  
    fig, ax = plt.subplots(1,2)  
    myresiduals.plot(title = "Residuals", ax = ax[0])  
    myresiduals.plot(kind = 'kde', title = 'Density', ax = ax[1])  
    plt.show()  
    
    # Actual vs Fitted  
    prediction_result = modelfit.get_prediction()
    predictions = modelfit.predict()
    forecast_result = modelfit.get_forecast()
    forecast = modelfit.forecast()
    return (prediction_result, predictions, forecast_result, forecast)


def map_range(x, oldMin, oldMax, newMin, newMax):
    oldRange = (oldMax - oldMin)
    newRange = (newMax - newMin)
    newValue = (((x - oldMin) * newRange) / oldRange) + newMin
    return newValue


def split_dataframes(dataframes, quartiles=2):
    lengths = sorted([(df.date.max()-df.date.min()).days for df in dataframes])
    intervals = pd.qcut(lengths,quartiles).categories
    splits = tuple([[df for df in dataframes if (((df.date.max()-df.date.min()).days > intervals[i].left) and ((df.date.max()-df.date.min()).days <= intervals[i].right))] for i in range(quartiles)])
    return (intervals, splits)


def median_norm(lpa_dfs):
    return np.median([df.week_linesperauthor.median() for df in lpa_dfs if str(df.week_linesperauthor.median()) != "nan"])


def combine_dataframes(lpa_dfs):
    lpa_dfs = calc_probs(lpa_dfs)
    
    updown_dfs = [pd.DataFrame({"updown":df.updown, "updown_size":df.updown.abs()}) for df in lpa_dfs]
    updown_total_df = reduce(lambda a, b: a.add(b, fill_value=0), updown_dfs)

    diff_total_df = pd.concat(([df.lpa_diff for df in lpa_dfs]), axis=1).median(axis=1)

    total_df = pd.concat([updown_total_df, diff_total_df], axis=1)
    total_df.rename({0: "lpa_diff"}, inplace=True, axis=1)
    total_df.drop(total_df[total_df.updown_size < 2][total_df[total_df.updown_size < 2].index>0].index, axis=0, inplace=True)

    total_df["prob"] = total_df.updown / total_df.updown_size
    total_df["prob"] = total_df["prob"].apply(lambda x: map_range(x, -1, 1, 1, 0))
    
    total_df = total_df.reset_index(drop=True)
    
    total_df = total_df.iloc[:total_df.updown_size.last_valid_index() + 1]
    total_df.lpa_diff = total_df.lpa_diff.interpolate()
    
    return total_df


def n_weeks_lpa(split, n):
    max_weeks = [len(s.week.unique()) for s in split]
    if min(max_weeks) < n:
        print("Uh oh! 'n' must be less than the maximum number of unique weeks committed for the projects in 'split'")
        return
    return [(s.groupby("week").addedloc.sum()/s.groupby("week").email.unique().apply(lambda x: len(x))).to_list()[:n] for s in split]


def longest_n_projects(split, n):
    max_weeks = [s.week.max() for s in split]
    max_n_weeks = sorted([s.week.max() for s in split])[len(split)-n:]
    longest_n_dfs = [split[max_weeks.index(week)] for week in max_n_weeks]
    return (longest_n_dfs, max_n_weeks)


def get_y0s(split, nproj_pct=0.1, nweek_pct=0.1):
    if ((nproj_pct<0) or (nproj_pct>1) or (nweek_pct<0) or (nweek_pct>1)):
        print("Percentage must be in the range: [0-1]")
        return
    n_projects = int(len(split)*nproj_pct)
    (longest_n_dfs, max_n_weeks) = longest_n_projects(split, n_projects)
    n_weeks = int(max(max_n_weeks)*nweek_pct)
    y0s = n_weeks_lpa(longest_n_dfs, n_weeks)
    return (y0s, np.mean(y0s))