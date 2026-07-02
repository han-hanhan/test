import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
y1 = np.array([1,2,3,4,5,6,7,8,9,10])
plt.plot(y1,label = 'y1')
plt.title('y1')
plt.subplot(1, 2, 2)
y2 = np.array([1,2,3,4,5,6,7,8,9,10][::-1])
plt.plot(y2,label = 'y2')
plt.title('y2')
plt.legend(loc = 'best')
plt.show()
print(y2.shape)
print(y2.dtype)
print(y2)
print(y2.ndim)

#11111111