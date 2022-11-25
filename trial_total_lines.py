#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:22:33 2022

@author: vickysmith
"""

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read json file - makes it into a list of dictionaries
with open('arthas.json','r') as json_file:
    data = json.load(json_file)[0:]


# Dictionary of author and total lines added and deletes in the form:
# {'author':{'addedloc':total lines added, 'deletedloc': total lines deleted}}    
authors_totals = {}
for commit in data:
    date = commit['date']
    year = commit['year']
    author = commit['author']
    
    lines = {}
    lines['addedloc'] = commit['addedloc']
    lines['deletedloc'] = commit['deletedloc']

    
    if author not in authors_totals:
       authors_totals[author] = lines
    
    else:
       authors_totals[author]['addedloc']+= lines['addedloc']
       authors_totals[author]['deletedloc']+= lines['deletedloc']

    


print(authors_totals['BlueT'])


# 2 dictionaries of author and total lines addes / deleted
authors_addedloc = {}
authors_deletedloc= {}

if author not in authors_addedloc:
    authors_addedloc[author] = commit['addedloc']
    authors_deletedloc[author] = commit['deletedloc']
   
else:
    authors_addedloc[author] += commit['addedloc']
    authors_deletedloc[author] += commit['deletedloc']















