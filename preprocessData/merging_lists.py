from statistics import median
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

## Merges lists of different lengths. Finds average 'time series'.
# Takes an array of lists of different lengths as an input, returns merged list (average time series), confidence_interval and maximum length (length of longest list).
def merge_lists(lists): 
    maxlen = max([len(lst) for lst in lists])
    model = np.linspace(0,1,maxlen)
    interped_lists = [[] for lst in lists]
    i=0
    for lst in lists:
        interped_lists[i] = np.interp(model,np.linspace(0, 1, len(lst)),lst)
        i+=1

    # finds confidence interval of each week
    confidence_interval = [0.8 * np.std(i) / sqrt(len(i)) for i in zip(*interped_lists)]
    # finds median of each week
    merged = [median(i) for i in zip(*interped_lists)]
    return merged,confidence_interval,maxlen

## To test the function
if __name__ == "__main__":
    lists = [[1,5,4,2,2,3,4,5], [5,6,4,2,1,3,5]]
    output = merge_lists(lists)
    print(output)
    plt.plot(output[0])
    plt.fill_between(np.linspace(0,output[2]-1,output[2]),(np.array(output[0])-np.array(output[1])), (np.array(output[0])+np.array(output[1])), color='red', alpha=0.3)
    plt.show()