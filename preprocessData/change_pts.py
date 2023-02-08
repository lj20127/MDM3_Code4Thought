import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

def unknown_n_bkps(model): # finds number of change points for different penalty values
    rpt_model = rpt.Pelt(model="rbf").fit(model)
    num_change_pts = []
    for i in range(0,20):
        change_pts = rpt_model.predict(pen=i*np.log(len(model)))
        num_change_pts.append(len(change_pts)-1)
    plt.plot(num_change_pts)
    plt.title("Distsribution of change points for different penalties using PELT")
    plt.show()

def detect_change_pts(model,n):
    rpt_model = rpt.Binseg(model="rbf").fit(model)
    change_pts = rpt_model.predict(n_bkps=n)
    return change_pts

if __name__ == "__main__":
    lst=np.concatenate([np.random.rand(100)+5,
                       np.random.rand(100)+10,
                       np.random.rand(100)+5])
    change_pts = detect_change_pts(lst,2)
    print(change_pts)