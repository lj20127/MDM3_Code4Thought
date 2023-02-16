from operator import attrgetter
from functools import reduce
import random as rnd
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')


def read_data_to_dataframes(file_path, read_n_files=None, columns_to_keep=["date", "day", "month", "year", "author", "addedloc", "deletedloc"]):
    """
    Reads JSON/PARQUET files and returns a list of pandas dataframes.
    Optionally filters specified columns from the dataframes.

    Parameters:
        - file_path (str): String of path to where the data is stored.
        - read_n_files (int, optional): Integer specifying number of files to read.
        - columns_to_filter (List[str], optional): List of columns to filter from the dataframes. Default: None
    
    Returns:
        - List[pd.DataFrame]: list of dataframes
    """
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
    # Interpolates for NaN values
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


def combine_dataframes(dataframes, column_to_combine, columns_to_keep=None):
    columns_to_keep = columns_to_keep if columns_to_keep else dataframes[0].columns.to_list()
    timeframe = "date"
    if len(dataframes[0].columns.to_list()) == 2:
        timeframe = dataframes[0].columns.to_list()[len(dataframes[0].columns.to_list()) - dataframes[0].columns.to_list().index(column_to_combine) - 1]
    combined_dataframe = dataframes[[len(dataframe) for dataframe in dataframes].index(max([len(dataframe) for dataframe in dataframes]))][columns_to_keep]

    combined_column = pd.concat([dataframe[column_to_combine] for dataframe in dataframes], axis=1)
    combined_column_sum = combined_column.sum(axis=1)

    combined_dataframe[column_to_combine] = combined_column_sum

    max_timeframes = sorted([dataframe[timeframe].max() for dataframe in dataframes])
    for i in range(len(max_timeframes)):
        div = len(max_timeframes) - i
        if i > 0:
            combined_dataframe.loc[(combined_dataframe[timeframe] > max_timeframes[i-1]) & (combined_dataframe[timeframe] <= max_timeframes[i]), column_to_combine] /= div
        else:
            combined_dataframe.loc[combined_dataframe[timeframe] <= max_timeframes[i], column_to_combine] /= div

    return combined_dataframe


def map_range(x, oldMin, oldMax, newMin, newMax):
    oldRange = (oldMax - oldMin)
    newRange = (newMax - newMin)
    newValue = (((x - oldMin) * newRange) / oldRange) + newMin
    return newValue


def dynamic_walk(linesperauthor_dataframes, y0, N=1000):
    """
    Creates a dynamic version of a binary walk by calculating specific probabilities
    of next data point being greater or less than the last one.

    Parameters:
        - dataframe (Pandas DataFrame): DataFrame containing the data.
        - y0 (int): Integer for the initial starting point of the algorithm.
        - N (int, optional): The number of iterations the algorithm should compute

    Returns:
        - (t, ys, minmaxys, minyidx, maxyidx, avgys) List[pd.DataFrame]: list of dataframes
    """
    for dataframe in linesperauthor_dataframes:
        linesperauthor_column = [column for column in dataframe.columns.to_list() if "linesperauthor" in column][0]
        dataframe["linesperauthor_diff"] = dataframe.diff(periods=1)[linesperauthor_column]
        conditions = [
            (dataframe['linesperauthor_diff'] < 0),
            (dataframe['linesperauthor_diff'] > 0),
            (dataframe['linesperauthor_diff'] == 0)
        ]
        values = [-1, 1, 0]
        dataframe['updown'] = np.select(conditions, values)

    updown_dataframes = [dataframe.updown for dataframe in linesperauthor_dataframes]
    diff_dataframes = [dataframe.linesperauthor_diff for dataframe in linesperauthor_dataframes]
    updown_total_dataframes = []
    for updown_dataframe in updown_dataframes:
        concat_dataframe = pd.concat([updown_dataframe, updown_dataframe.apply(lambda x: abs(x))], axis=1)
        concat_dataframe.columns = ["updown", "updown_size"]
        updown_total_dataframes.append(concat_dataframe)

    updown_total_dataframe = reduce(lambda a, b: a.add(b, fill_value=0), updown_total_dataframes)
    diff_total_dataframe = reduce(lambda a, b: a.add(b, fill_value=0), diff_dataframes)
    total_dataframe = pd.concat([updown_total_dataframe, diff_total_dataframe], axis=1)

    total_dataframe["prob"] = total_dataframe.updown / total_dataframe.updown_size
    total_dataframe = total_dataframe.fillna(0)
    
    min_prob = -1
    max_prob = 1
    total_dataframe.prob = total_dataframe.prob.apply(lambda x: map_range(x, min_prob, max_prob, 1, 0))
    
    linesperauthor_diff_min = total_dataframe.linesperauthor_diff.min()
    linesperauthor_diff_max = total_dataframe.linesperauthor_diff.max()
    total_dataframe.linesperauthor_diff = total_dataframe.linesperauthor_diff.apply(lambda x: map_range(abs(x), linesperauthor_diff_min, linesperauthor_diff_max, 0, abs(total_dataframe.linesperauthor_diff.mean())*2))

    t = np.array(total_dataframe.index.tolist())
    ys = []
    for _ in range(N):
        # initalises lines added (y axis) arrays
        y = np.zeros(shape=t.shape)

        # inital lines added at time 0 (week 0)
        y[0] = y0

        # sets probability that the lines added will go down
        ps = total_dataframe.prob.tolist()

        # sets step up or down given probability p
        steps = total_dataframe.linesperauthor_diff.tolist()
        for i in range(1,t.size):
            num = rnd.random()
            if num < ps[i]:
                y[i] = y[i-1]-steps[i]
            else:
                y[i] = y[i-1]+steps[i]

        ys.append(y)

    miny = ys[0]
    minyval = ys[0][-1]
    minyidx = 0
    maxy = ys[0]
    maxyval = ys[0][-1]
    maxyidx = 0
    for i in range(len(ys)):
        if ys[i][-1] < minyval:
            miny = ys[i]
            minyval=ys[i][-1]
            minyidx = i
        if ys[i][-1] > maxyval:
            maxy = ys[i]
            maxyval=ys[i][-1]
            maxyidx = i

    minmaxys = [miny,maxy]
    avgys = sum(ys)/len(ys)
    avgys = np.where(avgys<0, 0, avgys)
    return (t, ys, minmaxys, minyidx, maxyidx, avgys)