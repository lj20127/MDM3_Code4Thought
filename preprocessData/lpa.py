## Function to calculate 'Lines per author'
def lpa(df):
    # Grouping by 'weeknum' and selecting "email" column
    email_group = df.groupby("week")["email"]
    # Grouping by 'weeknum' and selecting "trueaddedloc" column
    addedloc_group = df.groupby("week")["addedloc"]
    
    addedloc_group_sum = addedloc_group.sum()
    unique_emails = email_group.unique().apply(lambda x: len(x))
    # lpa variable ->
    lpa = addedloc_group_sum / unique_emails
    '''
    How it works...
    1) Sums the addedloc_group for each week to get total lines committed for that week:
    addedloc_group_sum -> weeknum
                            0       4904
                            3      30466
                            4      15643
                            ...
    2) Finds the number unique emails that committed code for that week:
    unique_emails -> weeknum
                        0      1
                        3      1
                        4      1
                        ...
    3) And divides the sum of addedloc_group by the number of unique emails:
    addedloc_group_sum/unique_emails -> weeknum
                                            0       4904 / 1
                                            3      30466 / 1
                                            4      15643 / 1
                                            ...
    '''
    # Creates a new dataframe using the 'lpa' values
    new_df = lpa.to_frame().rename({0:"week_linesperauthor"}, axis=1).reset_index()
    return new_df




