import pandas as pd
import numpy as np

# Data frame containing the first dataset as df:
df = pd.read_json('firstdataset.json')

def filter(df, tech):
    df_idx1 = df[df['technology'] == tech].index
    df.drop(index=df[df['technology'] != tech].index, axis=0, inplace=True)
    df_idx2 = df[df['subject'].str.contains('Rename') == False].index
    df.drop(index=df[df['subject'].str.contains('Rename')].index, axis=0, inplace=True)
    df_idx3 = df[df['subject'].str.contains('=>') == False].index
    df.drop(index=df[df['subject'].str.contains('=>')].index, axis=0, inplace=True)

# removing all technologies apart from java
tech = 'java'
df_idx = df[df['technology'] == tech].index
df.drop(index=df[df['technology'] != tech].index, axis=0, inplace=True)

# removing subjects that contain 'Rename'
df_idx2 = df[df['subject'].str.contains('Rename') == False].index
df.drop(index=df[df['subject'].str.contains('Rename')].index, axis=0, inplace=True)

# removing subjects that contain '=>' 
df_idx3 = df[df['subject'].str.contains('=>') == False].index
df.drop(index=df[df['subject'].str.contains('=>')].index, axis=0, inplace=True)

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
            lines_per_author[idx] += lines_added[df_idx[i]]
    i += 1