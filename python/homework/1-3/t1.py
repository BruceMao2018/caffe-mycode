import numpy as np
import matplotlib.pyplot as plt
 
np.random.seed(1)
x=np.random.rand(10)
y=np.random.rand(10)
 
colors=np.random.rand(10)
area=(30*np.random.rand(10))**2
 
plt.scatter(x,y,s=area,c=colors,alpha=0.5)
plt.show()
