import pandas as pd
import numpy as np
from zipfile import ZipFile

################################################
# Import all files from zip file using code below - UNCOMMENT, USE ONCE AND COMMENT OUT:
# file_name = 'systems.zip'
# # opening the zip file in READ mode
# with ZipFile(file_name, 'r') as zip:
#     # printing all the contents of the zip file
#     zip.printdir()
#
#     # extracting all the files
#     print('Extracting all the files now...')
#     zip.extractall()
#     print('Done!')
#################################################
# Data frame containing the first dataset as df:
df = pd.read_json('firstdataset.json')

# removing all technologies apart from java
java_idx = df[df['technology'] == 'java'].index
df.drop(index=df[df['technology'] != 'java'].index, axis=0, inplace=True)

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
            lines_per_author[idx] += lines_added[java_idx[i]]
    i += 1