from extractJson import extractJson
from lpa import lpa
import os
import numpy as np
from merging_lists import merge_lists
import matplotlib.pyplot as plt
from change_pts import detect_change_pts

# Specify folder path that contains the json files
path = os.path.dirname(os.path.realpath(__file__))+"/systems/"

# List containing all the json files as panda dataframes
all_dfs = extractJson(path) # use 'all_dfs = extractJson(path,n)' for n random json files 

# creates a list for lpa for every json file
lists = [[] for i in range(0,len(all_dfs))]
i=0
# sets (maximum) threshold for lpa
threshold = 1000
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
plt.plot(weeks,merged)
plt.fill_between(weeks,(np.array(merged)-np.array(ci)),(np.array(merged)+np.array(ci)),color='red',alpha=0.3)
plt.show()

# finds 2 change points
change_pts = detect_change_pts(np.array(merged),2)