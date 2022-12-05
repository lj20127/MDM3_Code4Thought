import os
import warnings
import pandas as pd
import random as rnd
warnings.filterwarnings('ignore')

# Function to extract the weeknum for that date
def weekNum(x, start):
    daysDiff = (x-start).days
    weekDiff = daysDiff // 7
    return weekDiff

# Function to convert the first 'n' json files to dataframes from a 'filePath'
# If you want to convert all the json files to dataframes then don't specify a value for 'n'
def extractJson(filePath, n=None):
    filepaths  = [os.path.join(filePath, name) for name in os.listdir(filePath)]
    # creates list of n random file paths
    if n==None:
        n_filepaths = rnd.sample(filepaths,n)
    else:
        n_filepaths = filepaths
    # variable all_dfs ->
    all_dfs = [pd.read_json(path) for path in n_filepaths]
    # Columns of interest
    columns = ["date", "day", "month", "year", "author", "addedloc", "deletedloc"]
    for df in all_dfs:
        df.sort_values(by="date", inplace=True)
        new_df = df[columns].copy()
        
        # Start date of the project
        start = new_df.date.min()
        
        # 'weeknum' by applying weekNum() to the date column
        df["weeknum"] = new_df.date.apply(lambda x: weekNum(x, start))
        df["combinedpath"] = [''.join(l) for l in df.path]
        
        # 'trueaddedloc' is the same as 'addedloc' without subjects including "=>" (i.e. renamed files)
        df["trueaddedloc"] = df["addedloc"]
        df["truedeletedloc"] = df["deletedloc"]
        df.loc[df.combinedpath.str.contains("=>"), "trueaddedloc"] = 0
        df.loc[df.combinedpath.str.contains("=>"), "truedeletedloc"] = 0
    return all_dfs
