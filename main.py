import os
import pandas as pd
import numpy as np
from merging_lists import merge_lists
import matplotlib.pyplot as plt
from change_pts import detect_change_pts, unknown_n_bkps
from tools import read_data_to_dataframes, filter_dataframe, lines_per_author
from timeseries import FitARIMA
from scipy.signal import savgol_filter

TITLE_FONT_SIZE = 25
AXIS_FONT_SIZE = 20

def project_length(repo): # function to calculate the length of projects
    length = float(max(repo['year']))-float(min(repo['year']))
    return length

def split_repos(all_dfs):
    # Splitting the data into two:
    
    project_length_list = []  # project length list...

    for repo in all_dfs:
        project_length_list.append(project_length(repo)) # loop adds project lengths into a list 

    print(sorted(project_length_list))

    sorted_project_list = sorted(project_length_list)
    argsorted_project_lengths = np.argsort(project_length_list) # project lengths are 'argsorted' into order

    boundary = 9

    short_repo_index = []
    long_repo_index = []

    for i in range(0,len(sorted_project_list)):
        if sorted_project_list[i] <= boundary:
            short_repo_index.append(argsorted_project_lengths[i])
        else:
            long_repo_index.append(argsorted_project_lengths[i])

    short_repos = []
    long_repos = []
    # here the repos are sorted into two different splits based on length of project time
    for index in short_repo_index:
        short_repos.append(all_dfs[index])

    for index in long_repo_index:
        long_repos.append(all_dfs[index])

    print(f"short repos: {len(short_repos)}")
    print(f"long repos: {len(long_repos)}")

    return short_repos, long_repos

# Creates a model for lines per author per week for a given group of repos (dfs)
def model(dfs,txt,find_change_pts=False):
    # creates a list for lpa for every json file
    lpa_lists = [[] for i in range(0,len(dfs))]
    first_week = [[] for i in range(0,len(dfs))]
    i=0
    for df in dfs:
        # filters dataframe
        filtered_df = filter_dataframe(df)
        # gets lines per author per week
        new_df = lines_per_author(filtered_df)

        lpa_lists[i] = np.array(new_df["week_linesperauthor"])
        first_week[i] = lpa_lists[i][0]
        i+=1

    # removes any lpa lists that are less than 1 i.e. removes any projects that are 1 week in duration
    lpa_lists = [lst for lst in lpa_lists if len(lst) > 1]
    plt.hist(first_week,bins=len(dfs))
    plt.title("Histogram showing the distribution of lines per author for the first week",fontdict={'fontsize': TITLE_FONT_SIZE})
    plt.xlabel("Lines per author in the first week",fontdict={'fontsize': AXIS_FONT_SIZE})
    plt.ylabel("Frequency",fontdict={'fontsize': AXIS_FONT_SIZE})
    plt.show()

    # merges data
    median_model,ci,num_weeks = merge_lists(lpa_lists)
    weeks = np.linspace(0,num_weeks,num_weeks)

    # filtered data using Savitzky-Golay filter
    median_model_filtered = savgol_filter(x=median_model,window_length=num_weeks//50,polyorder=3)

    # converts model to dataframe for use with FitARIMA
    median_model_df = pd.DataFrame(data={'model':median_model_filtered})
    # fits ARIMA model to filtered data
    arima_model,residuals = FitARIMA(dfseries=median_model_df,plot=True)

    # creates lpa vs weeks plot
    # plots main lpa
    plt.plot(weeks,median_model,label="Unfiltered")
    plt.plot(weeks,median_model_filtered,label="Filtered",linewidth=3)
    # plots confidence interval
    # plt.fill_between(weeks,(np.array(merged)-np.array(ci)),(np.array(merged)+np.array(ci)),color='red',alpha=0.3)
    plt.title(f"Lines Per Author for {txt}",fontdict={'fontsize': TITLE_FONT_SIZE})
    plt.xlabel("Weeks",fontdict={'fontsize': AXIS_FONT_SIZE})
    plt.ylabel("Median Lines Per Author",fontdict={'fontsize': AXIS_FONT_SIZE})
    plt.legend()
    plt.show()

    # plots arima model
    plt.plot(weeks[1:],arima_model[1:])
    plt.fill_between(weeks[1:],arima_model[1:],(arima_model[1:]+residuals[0][1:]),color='red',alpha=0.3)
    plt.title(f"ARIMA Model fitted to filtered data for {txt}",fontdict={'fontsize': TITLE_FONT_SIZE})
    plt.xlabel("Weeks",fontdict={'fontsize': AXIS_FONT_SIZE})
    plt.ylabel("Median Lines Per Author",fontdict={'fontsize': AXIS_FONT_SIZE})
    plt.show()


    # finds 2 change points
    if find_change_pts:
        change_pts = detect_change_pts(np.array(arima_model[1:]),2)
        # unknown_n_bkps(np.array(arima_model[1:]))
        fig, ax = plt.subplots()
        ax.plot(arima_model[1:])
        ax.axvspan(0, change_pts[0], alpha=0.25, color='green')
        ax.axvspan(change_pts[0], change_pts[1], alpha=0.25, color='orange')
        ax.axvspan(change_pts[1], change_pts[2], alpha=0.25, color='red')
        plt.title(f"Change Point Analysis for {txt}",fontdict={'fontsize': TITLE_FONT_SIZE})
        plt.xlabel("Weeks",fontdict={'fontsize': AXIS_FONT_SIZE})
        plt.ylabel("Median Lines Per Author",fontdict={'fontsize': AXIS_FONT_SIZE})
        plt.show()
        print(f"Change Points at weeks {change_pts[0]} and {change_pts[1]}")


def main():
    # Specify folder path that contains the json files
    path = os.path.dirname(os.path.realpath(__file__))+"/systems/"
    
    # Gets repo info from either parquet or json files
    all_dfs = read_data_to_dataframes(path) # use 'all_dfs = read_data_to_dataframes(path,n)' for n random files # there are more parquet files than json files??

    # Splits repos into short and long term projects
    short_repos,long_repos = split_repos(all_dfs) 

    model(dfs=all_dfs,txt="All repos",find_change_pts=True)
    # model(dfs=short_repos,txt="Short Repos",find_change_pts=True)
    # model(dfs=long_repos,txt="Long Repos",find_change_pts=True)

if __name__ == "__main__":
    main()