import pandas as pd
import numpy as np

# Data frame containing the first dataset as df:
df = pd.read_json('firstdataset.json')

# REMOVING ALL TECHNOLOGIES APART FROM JAVA:
java_df = []
java_index = []
for i in range(len(df)):
    if df.iloc[i]['technology'] == 'java' and 'Rename' not in df.iloc[i]['subject']:
        java_index.append(i)
        java_df.append(df.iloc[i])

# Splitting the data into relevant variables:
author = java_df['author']  # simply shows all corresponding authors to each commit

# DATES - day/month/year
day = java_df['day']
month = java_df['month']
year = java_df['year']

# Lines - will show lines added and lines deleted respectively
lines_added = java_df['addedloc']  # Assuming that the headers mean adding and deleting lines
lines_deleted = java_df['deletedloc']


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


