from extractJson import extractJson
from lpa import lpa
import os
import numpy as np
from merging_lists import merge_lists
import matplotlib.pyplot as plt
from change_pts import detect_change_pts

# Specify folder path that contains the json files
path = os.path.dirname(os.path.realpath(__file__))+"/systemsUpdated/"

# List containing all the json files as panda dataframes
all_dfs = extractJson(path,5) # use 'all_dfs = extractJson(path,n)' for n random json files 

# Splitting the data into two:

def project_length(df): # function to calculate the length of projects
    length = float(max(repo['year']))-float(min(repo['year']))
    return length
    
project_length_list = []  # project length list...

for repo in all_dfs:
    project_length_list.append(project_length(repo)) # loop adds project lengths into a list 

print(sorted(project_length_list))

sorted_project_list = sorted(project_length_list)
argsorted_project_lengths = np.argsort(project_length_list) # project lengths are 'argsorted' into order

boundary = 8

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

    
# creates a list for lpa for every json file
lists = [[] for i in range(0,len(all_dfs))]
i=0
# sets (maximum) threshold for lpa
threshold = 3000
for df in all_dfs:
    java_df = df[df.technology == "java"]
    new_df = lpa(java_df)
    # removes any lpa over threshold
    new_df.loc[new_df.lpa > threshold] = 0
    # removes any lpa that's 0
    if len(np.array(new_df["lpa"])) > 0:
        lists[i] = np.array(new_df["lpa"])
    i+=1

# removes any lpa lists that are empty
lists = [lst for lst in lists if lst != []]

# creates lpa vs weeks plot
output = merge_lists(lists)
merged=output[0]
ci=output[1]
num_weeks=output[2]
weeks = np.linspace(0,num_weeks,num_weeks)

# plots main lpa
plt.plot(weeks,merged)

# plots confidence interval
plt.fill_between(weeks,(np.array(merged)-np.array(ci)),(np.array(merged)+np.array(ci)),color='red',alpha=0.3)

plt.show()

# finds 2 change points
change_pts = detect_change_pts(np.array(merged),2)