import matplotlib.pyplot as plt
import math 
import numpy as np

fig = plt.figure()
plt.title("Test")
X = np.random.rand(100)
Y = np.random.rand(100)
plt.scatter(X, Y)
fig.savefig("test.png")
