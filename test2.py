import numpy as np

a = np.arange(5)
print(a)
rng = np.random.RandomState(6)  
rng.shuffle(a) 
print(a)