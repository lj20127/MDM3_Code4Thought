from extractJson import extractJson
from lpa import lpa
import os
import numpy as np
from merging_lists import merge_lists
import matplotlib.pyplot as plt

# Specify folder path that contains the json files
path = os.path.dirname(os.path.realpath(__file__))+"/systems/"

# List containing all the json files as panda dataframes
all_dfs = extractJson(path)

# creates a list for lpa for every json file
lists = [[] for i in range(0,len(all_dfs))]
i=0
threshold = 5000
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

# creats lpa vs weeks plot
output = merge_lists(lists)
weeks = np.linspace(0,output[1],output[1])
plt.plot(weeks,output[0])
plt.show()