import json
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from merging_lists import merge_lists
from scipy.stats import shapiro 
from scipy.stats import kstest


def project_duration(file):
    """ This function takes a .json file and returns the duration of the project in weeks"""
    with open(file,'r') as json_file:
        data = json.load(json_file)[0:]
    
    dates = []
    for commit in data:
        date = commit['date']
        tech = commit['technology']
        if tech != 'java':
            continue 
        if date not in dates:
            dates.append(date)
 
    dates = pd.to_datetime(list(dates), dayfirst=True) 
    df = pd.DataFrame({'date':  dates})
    df['date'] = dates - pd.to_timedelta(7, unit='d')
    df = df.sort_values(by= 'date', ascending=True)
    dfg = df.groupby([pd.Grouper(key='date', freq='W')])
    duration = len(dfg)
    
    return duration



# Find all json files in current path and find duration for each
path= '/Users/vickysmith/Documents/3RD YEAR/MDM3/Project 3 - Code4Thought/'
json_files = [n for n in os.listdir(path) if '.json' in n]
durations = [project_duration(file) for file in json_files]

# Visualise results in a histogram
plt.figure()
plt.hist(durations, bins=20)
plt.xlabel('Project duration in weeks')
plt.ylabel('Frequency')
plt.savefig('Project_Duration_Histogram.png')
plt.show()


mean = np.mean(durations)
mean_years = int(mean/52)

print('mean in weeks:', mean)
print('mean in years:', mean_years)


shapiro(durations)











