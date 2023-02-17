
import json
from datetime import datetime
import random
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from merging_lists import merge_lists

def lines_per_author(file):
    """ This function takes a .json file and returns the total number of lines
    added per author in a dataframe 'lpa' """
    with open(file,'r') as json_file:
        data = json.load(json_file)[0:]
    
    authors = []
    lines_per_author = {}
    for commit in data:
        author = commit['email']
        added = commit['addedloc']
        deleted = commit['deletedloc']
        tech = commit['technology']
        subject = commit['subject']
        if tech != 'java':
            continue 
        if subject == 'Rename' or subject =='=>'or subject =='Restructure':
            continue 
        if added == deleted:
            continue
        if author not in authors:
            authors.append(author)
        
        if author not in lines_per_author:
            lines_per_author[author] = added
        else:
            lines_per_author[author] += added

  
    lines = [x for x in lines_per_author.values()]
    lpa = pd.DataFrame({'author':  authors,
                       'lines added': lines})
    
    return lpa



path= '/Users/vickysmith/Documents/3RD YEAR/MDM3/Project 3 - Code4Thought/'
json_files = os.listdir(path)

dataframes = []
for file in json_files:
    if '.json' in file:
        lpa = lines_per_author(file)
        dataframes.append(lpa)
dataframes = pd.concat(dataframes)


# Lines per author histogram
plt.figure()
plt.hist(dataframes['lines added'], bins=5000)
plt.xlabel('Lines per author')
plt.ylabel('Frequency')
plt.show()




