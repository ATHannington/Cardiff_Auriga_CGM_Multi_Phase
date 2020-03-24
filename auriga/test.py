import numpy as np
import matplotlib.pyplot as plt

x = np.array([float(i) for i in range(0,11)])
y = np.array([x**2 for x in x])

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(x,y)
ax1.set_xlabel(r'$x \alpha$')
ax1.text(1.,0.,r"M$_\odot$")
ax2.plot(x,y)
ax2.set_xlabel(r'$x \alpha$')
ax2.text(1.,0.,r"M$_\odot$")
plt.show()
