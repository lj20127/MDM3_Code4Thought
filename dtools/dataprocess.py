from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from operator import attrgetter
from functools import reduce
import matplotlib.pyplot as plt
import random as rnd
import pandas as pd
import numpy as np
import inspect
import os

import warnings
warnings.filterwarnings('ignore')


class DataProcessor():
    def __init__(self, path, splits=2):
        self.data_dir = path
        self.data = self.read_data_to_dataframes(self.data_dir)
        self.filtered_data = [self.filter_dataframe(d) for d in self.data if (self.filter_dataframe(d).date.max()-self.filter_dataframe(d).date.min()).days != 0]
        self.lpa_data = [self.lines_per_author(d, timeframe="week", interp_method=False) for d in self.filtered_data]
        self.intervals, self.splits = self.split_dataframes(self.filtered_data, splits)
        self.combined_data = [self.combine_dataframes(split) for split in self.splits]

        
    def __call__(self):
        attributes = inspect.getmembers(data, lambda a:not(inspect.isroutine(a)))
        length = len([a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))])
        names = [[a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))][i][0] for i in range(length)]
        values = [[a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))][0][i] for i in range(length)]
        print(
            f"""
            Attributes: {names}
            
            Values: {values}
            """
        )
    def read_data_to_dataframes(self, file_path, read_n_files=None, columns_to_keep=["date", "day", "month", "year", "author", "addedloc", "deletedloc"]):
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
    
    
    def filter_dataframe(self, dataframe, technology="java"):
        dataframe = dataframe[dataframe.technology==technology]
        return dataframe
    
    
    def split_dataframes(self, dataframes, quartiles=2):
        lengths = sorted([(df.date.max()-df.date.min()).days for df in dataframes])
        
        intervals = pd.qcut(lengths,quartiles, duplicates="drop").categories
        
        splits = tuple([[df for df in dataframes if (((df.date.max()-df.date.min()).days > intervals[i].left) and ((df.date.max()-df.date.min()).days <= intervals[i].right))] for i in range(quartiles)])
        
        return (intervals, splits)


    def combine_dataframes(self, lpa_dfs):
        lpa_dfs = self.calc_probs(lpa_dfs)

        updown_dfs = [pd.DataFrame({"updown":df.updown, "updown_size":df.updown.abs()}) for df in lpa_dfs]
        updown_total_df = reduce(lambda a, b: a.add(b, fill_value=0), updown_dfs)

        diff_total_df = pd.concat(([df.lpa_diff for df in lpa_dfs]), axis=1).median(axis=1)

        total_df = pd.concat([updown_total_df, diff_total_df], axis=1)
        total_df.rename({0: "lpa_diff"}, inplace=True, axis=1)
        total_df.drop(total_df[total_df.updown_size < 2][total_df[total_df.updown_size < 2].index>0].index, axis=0, inplace=True)

        total_df["prob"] = total_df.updown / total_df.updown_size
        total_df["prob"] = total_df["prob"].apply(lambda x: self.map_range(x, -1, 1, 1, 0))

        total_df = total_df.reset_index(drop=True)

        total_df = total_df.iloc[:total_df.updown_size.last_valid_index() + 1]
        total_df.lpa_diff = total_df.lpa_diff.interpolate()

        return total_df


    def n_weeks_lpa(self, split, n):
        max_weeks = [len(s.week.unique()) for s in split]
        if min(max_weeks) < n:
            print("Uh oh! 'n' must be less than the maximum number of unique weeks committed for the projects in 'split'")
            return
        return [(s.groupby("week").addedloc.sum()/s.groupby("week").email.unique().apply(lambda x: len(x))).to_list()[:n] for s in split]

    
    def longest_n_projects(self, split, n):
        max_weeks = [s.week.max() for s in split]
        max_n_weeks = sorted([s.week.max() for s in split])[len(split)-n:]
        longest_n_dfs = [split[max_weeks.index(week)] for week in max_n_weeks]
        return (longest_n_dfs, max_n_weeks)
    
    
    def get_y0s(self, split, nproj_pct=0.1, nweek_pct=0.1):
        if ((nproj_pct<0) or (nproj_pct>1) or (nweek_pct<0) or (nweek_pct>1)):
            print("Percentage must be in the range: [0-1]")
            return
        n_projects = int(len(split)*nproj_pct)
        (longest_n_dfs, max_n_weeks) = longest_n_projects(split, n_projects)
        n_weeks = int(max(max_n_weeks)*nweek_pct)
        y0s = n_weeks_lpa(longest_n_dfs, n_weeks)
        return (y0s, np.mean(y0s))
    
    
    def lines_per_author(self, dataframe, timeframe="week", interp_method="linear"):
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
    
    
    
    def fit_arima(self, dfseries):
        model = ARIMA(dfseries, order=(1,1,1))
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


    def map_range(self, x, oldMin, oldMax, newMin, newMax):
        oldRange = (oldMax - oldMin)
        newRange = (newMax - newMin)
        newValue = (((x - oldMin) * newRange) / oldRange) + newMin
        return newValue


    def median_norm(self,lpa_dfs):
        return np.median([df.week_linesperauthor.median() for df in lpa_dfs if str(df.week_linesperauthor.median()) != "nan"])


    def calc_probs(self, dfs):
        lpa_dfs = [self.lines_per_author(df, "week", False) for df in dfs]
        average_norm = self.median_norm(lpa_dfs)
        lpa_dfs_probs = lpa_dfs.copy()
        for df in lpa_dfs_probs:
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

            lpa_diff_max = df.lpa_diff.max()
            lpa_diff_min = df.lpa_diff.min()
            df["lpa_diff"] = df.lpa_diff.apply(lambda x: self.map_range(abs(x), lpa_diff_min, lpa_diff_max, 0, 2*abs(average_norm)))
        return lpa_dfs_probs



    def dynamic_walk(self, total_df, y0, N=10000):
        t = np.array(total_df.index.tolist())
        ys = []

        for _ in range(N):
            y = np.zeros(shape=t.shape)
            # inital lines added at time 0 (week 0)
            y[0] = y0
            # sets probability that the lines added will go down
            ps = total_df.prob.tolist()
            # sets step up or down given probability p
            steps = total_df.lpa_diff.tolist()

            for i in range(1,t.size):
                num = rnd.random()
                if num < ps[i]:
                    y[i] = y[i-1]-steps[i]
                else:
                    y[i] = y[i-1]+steps[i]

            ys.append(y)

        avgys = sum(ys)/len(ys)
        avgys = np.where(avgys<0, 0, avgys)

        return (t, ys, avgys)