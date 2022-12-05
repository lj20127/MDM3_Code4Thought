import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def detect_change_pts(lst,n):
    rpt_list = rpt.Binseg(model="l2").fit(lst)
    change_pts = rpt_list.predict(n_bkps=n)
    fig = rpt.display(lst,change_pts)
    plt.show()
    return change_pts

if __name__ == "__main__":
    lst = np.array([2,2,2,2,2,2,1,1,1,1,1,3,3,3,3])
    change_pts = detect_change_pts(lst,2)
    print(change_pts)