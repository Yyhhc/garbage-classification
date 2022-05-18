import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(16).reshape(4, 4)
plt.xticks(np.arange(4), ['x', 'y', 'z', 'm'])
plt.yticks(np.arange(4), ['x', 'y', 'z', 'm'])
plt.imshow(x, cmap=plt.cm.hot, vmin=0, vmax=1)
plt.title('classification heatmap')
plt.colorbar()
plt.show()
