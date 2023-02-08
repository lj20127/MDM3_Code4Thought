import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def detect_change_pts(model,n):
    rpt_list = rpt.Binseg(model="l2").fit(model)
    change_pts = rpt_list.predict(n_bkps=n)
    return change_pts

if __name__ == "__main__":
    lst=np.concatenate([np.random.rand(100)+5,
                       np.random.rand(100)+10,
                       np.random.rand(100)+5])
    change_pts = detect_change_pts(lst,2)
    print(change_pts)