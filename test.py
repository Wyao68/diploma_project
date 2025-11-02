import numpy as np

x = np.array([0,0,1,2,3])         
derivative = np.ones_like(x)
derivative[x <= 0] = 0
print(x)
print(derivative) 