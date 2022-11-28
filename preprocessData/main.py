from extractJson import extractJson
from lpa import lpa

# Specify folder path that contains the json files
path = r""

# List containing all the json files as panda dataframes
all_dfs = extractJson(path, 2)
# extracting first df
first_df = all_dfs[0]
# extracting only java entries
java_df = first_df[first_df.technology == "java"]

# create new dataframe with 'lpa' column
new_df = lpa(java_df)
print(new_df)
