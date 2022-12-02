from statistics import mean
import numpy as np

## Merges lists of different lenghts. Finds 'time series'
def merge_lists(lists): 
    maxlen = max([len(lst) for lst in lists])
    model = np.linspace(0,1,maxlen)
    interped_lists = [[] for lst in lists]
    i=0
    for lst in lists:
        interped_lists[i] = np.interp(model,np.linspace(0, 1, len(lst)),lst)
        i+=1
    merged = [mean(i) for i in zip(*interped_lists)]
    return merged

## To test the function
if __name__ == "__main__":
    lists = [[1,2,3,4,5,7,8], [5,2,3], [1,3,5,4,6]]
    merged = merge_lists(lists)
    print(merged)