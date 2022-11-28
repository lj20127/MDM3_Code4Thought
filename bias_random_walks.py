import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# initalises time (x axis) and lines added (y axis) arrays
t = np.linspace(0,1000,1000)
y = np.zeros(shape=t.shape)

# inital lines added at time 0 (week 0)
y[0] = 1000
# sets probability that the lines added will go down
p = 0.75
# sets step up or down given probability p
step = 1.8

# computes random walk
for i in range(1,t.size):
    num = rnd.random()
    if num < p:
        y[i] = y[i-1]-step
    else:
        y[i] = y[i-1]+step

# creates plot
plt.plot(t,y)
plt.ylim(0,1500)
plt.grid(True)
plt.xlabel("Week")
plt.ylabel("Lines Added per Author")
plt.title("Bias Random Walk")
plt.show()