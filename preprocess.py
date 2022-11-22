import pandas as pd

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
