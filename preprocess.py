import pandas as pd
import numpy as np

# Data frame containing the first dataset as df:
df = pd.read_json('firstdataset.json')

# Splitting the data into relevant variables:
author = df['author']  # simply shows all corresponding authors to each commit

# DATES - day/month/year
day = df['day']
month = df['month']
year = df['year']

# Lines - will show lines added and lines deleted respectively
lines_added = df['addedloc']  # Assuming that the headers mean adding and deleting lines
lines_deleted = df['deletedloc']

# Array of all the authors that have commited
authors_unique = np.unique(np.array(author))

# Finds the lines commited per author
lines_per_author = np.zeros(shape=authors_unique.shape)
i = 0
for a in author:
    for auth in authors_unique:
        if a == auth:
            idx = np.where(authors_unique==auth)
            lines_per_author[idx] += lines_added[i]
    i += 1