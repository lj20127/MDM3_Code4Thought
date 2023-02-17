from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import minmax_scaling
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import savgol_filter
from operator import attrgetter
from dtools.functions import *
from statistics import median
from functools import reduce
import matplotlib.pyplot as plt
import random as rnd
import pandas as pd
import numpy as np
import os
import ruptures as rpt


def detect_change_pts(model,n):
    rpt_model = rpt.Binseg(model="rbf").fit(model)
    change_pts = rpt_model.predict(n_bkps=n)
    return change_pts


def best_distribution(dfs, N):
    preds_arr, errors_arr, mean_errors_arr, trains_arr, tests_arr, merged_trains_arr = [], [], [], [], [], []
    
    for _ in range(N):
        train, test = train_test_split(dfs, test_size=0.2)
        trains_arr.extend((train,))
        merged_dfs = merge_dataframes(train)
        train = np.array(merged_dfs.model.to_list())
        merged_trains_arr.extend((train,))

        pred = np.array(merged_dfs.model.to_list())
        test = [lines_per_author(df) for df in test]
        t = np.array(merged_dfs.index.to_list())

        test_array = [np.array(df.week_linesperauthor.to_list()) for df in test]
        tests_arr.extend((test_array,))

        trains = [train.copy() for i in range(len(test_array))]
        trains = [np.array(tr[:len(te)]) for te, tr in zip(test_array, trains)]

        preds = [pred.copy() for i in range(len(test_array))]
        preds = [np.array(pr[:len(te)]) for te, pr in zip(test_array, preds)]

        error = calc_errors(trains, preds, test_array)
        mean_error = np.mean(error)

        preds_arr.extend((pred,))
        errors_arr.extend((error,))
        mean_errors_arr.extend((mean_error,))
        
    return (merged_trains_arr, trains_arr, preds_arr, tests_arr, errors_arr, mean_errors_arr)




def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))


def mase(y_train, y_pred, y_test):
    # mean absolute scaled error
    # y: test data
    # y_hat: model predicted data (e.g. random walk)
    # y_train: data used to train model
    ## Naive in-sample Forecast
    naive_y_pred = y_train[:-1]
    naive_y = y_train[1:]

    ## Calculate MAE (in sample)
    mae_in_sample = np.mean(np.abs(naive_y - naive_y_pred))
    if len(y_test)>len(y_pred):
        diff = len(y_test) - len(y_pred)
        y_test = y_test[:len(y_test) - diff]

    mae = np.mean(np.abs(y_test - y_pred))

    return mae/mae_in_sample


def prepare_for_walk(all_dfs):
    lpa_probs_dfs = calc_probs(all_dfs)
    lpa_probs_combined = combine_probs_dfs(lpa_probs_dfs)
    merged_dfs = merge_dataframes(all_dfs)
    merged_probs_dfs = pd.DataFrame({"week":merged_dfs.index, "lpa":merged_dfs.model, "lpa_diff":merged_dfs.model.diff().abs(), "updown":lpa_probs_combined.updown, "updown_size":lpa_probs_combined.updown_size, "prob":lpa_probs_combined.prob})
    return merged_probs_dfs
    
    
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
#     print(lpa_lists)
    # merges data
    merged,ci,num_weeks = merge_lists(lpa_lists)
    weeks = np.linspace(0,num_weeks,num_weeks)

    # filtered data using Savitzky-Golay filter
    merged_filtered = savgol_filter(x=merged,window_length=49,polyorder=2)

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


def calc_probs(all_dfs):
    lpa_probs_dfs = []
    for df in all_dfs:   
        lpa_df = lines_per_author(df)
        lpa_df["lpa_diff"] = lpa_df.week_linesperauthor.diff()
        conditions = [
            (lpa_df['lpa_diff'] < 0),
            (lpa_df['lpa_diff'] > 0),
            (lpa_df['lpa_diff'] == 0)
        ]

        values = [-1, 1, 0]
        lpa_df['updown'] = np.select(conditions, values)
        lpa_df["updown_size"] = lpa_df.updown.abs()
        lpa_probs_dfs.append(lpa_df)
    return lpa_probs_dfs


def combine_probs_dfs(lpa_dfs):
    probs_df = reduce(lambda a, b: a[["updown", "updown_size"]].add(b, fill_value=0), lpa_dfs)
    probs_df["prob"] = probs_df.updown / probs_df.updown_size
    probs_df["prob"] = probs_df["prob"].apply(lambda x: map_range(x, -1, 1, 1, 0))
    probs_df.drop(probs_df[probs_df.updown_size < 2][probs_df[probs_df.updown_size < 2].index>0].index, axis=0, inplace=True)
    return probs_df

def random_walk(total_df, y0, N=10000, dstep=0.1):
    t = np.array(total_df.index.tolist())
    ys = []
    errors = []
    mean_errors = []
    for i in range(N):
        y, error = step_walk(total_df, y0, dstep)
        mean_error = np.mean(error)
        ys.append(y)
        errors.append(error)
        mean_errors.append(mean_error)
    
    smallest_error_idx = np.argmin(mean_errors)
    
    best_y = ys[smallest_error_idx]
    best_y_error = errors[smallest_error_idx]
    best_y_mean_error = mean_errors[smallest_error_idx]
    
    mean_y = sum(ys)/len(ys)
    
    walk_data = {
        "time":t,
        "all_preds":{
            "preds":ys,
            "errors":errors,
            "mean_errors":mean_errors
        }, 
        "best_pred":{
            "pred":best_y,
            "error":best_y_error,
            "mean_error":best_y_mean_error
        },
        "mean_pred":{
            "pred":mean_y,
            "error":np.mean(errors, axis=1),
            "mean_error":[np.mean(errors, axis=0), np.mean(mean_errors)]
        }
    }
    return walk_data


def step_walk(total_df, y0, dstep=0.1, tol=1e-1):
    t = np.array(total_df.index.tolist())
    
    
    y = np.zeros(shape=t.shape)
    errors = np.zeros(shape=t.shape)
    
    y[0] = y0
    errors[0] = mse(total_df.lpa[0], y[0])
    
    ps = total_df.prob.tolist()
    # sets step up or down given probability p
    steps = total_df.lpa_diff.tolist()
    
    random_nums = np.random.uniform(0, 1, size=t.size)
    
    for i in range(1,t.size):
        num = random_nums[i]
        
        if num < ps[i]:
            y[i] = y[i-1]-steps[i]
            errors[i] = mse(total_df.lpa[i], y[i])
                    
        else:
            y[i] = y[i-1]+steps[i]
            errors[i] = mse(total_df.lpa[i], y[i])
            
    return (y, errors)


def calc_errors(y_trains, y_preds, y_tests):
    return [mase(y_trains[i], y_preds[i], y_tests[i]) for i in range(len(y_trains))]


def best_train_set(dfs, N):
    models = []
    errors = []
    trains_arr = []
    hashes = []
    
    
    for _ in range(N):
        train, test = train_test_split(dfs, test_size=0.2)
        trains_arr.extend((train,))

        total_df = prepare_for_walk(train)

        model = random_walk(total_df, total_df.lpa[0], N=1000)

        train = np.array(total_df.lpa.to_list())
        pred = model["best_pred"]["pred"]
        test = [lines_per_author(df) for df in test]
        t = model["time"]

        test_array = [np.array(df.week_linesperauthor.to_list()) for df in test]

        trains = [train.copy() for i in range(len(test_array))]
        trains = [np.array(tr[:len(te)]) for te, tr in zip(test_array, trains)]

        preds = [pred.copy() for i in range(len(test_array))]
        preds = [np.array(pr[:len(te)]) for te, pr in zip(test_array, preds)]

        error = np.mean(calc_errors(trains, preds, test_array))

        models.extend((model,))
        errors.extend((error,))
        
    return (models, errors, trains_arr)