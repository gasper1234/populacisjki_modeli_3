from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

z,l = np.meshgrid(np.linspace(0,2.5,50),np.linspace(0,2.5,50))

p = 0.7

u = p*z * (np.ones(len(z))-l)
v = l/p * (z-np.ones(len(z)))
speed = np.sqrt(u**2 + v**2)
lw = speed/speed.max()*3

plt.streamplot(z,l,u,v, density=1, color='k', linewidth=lw)
plt.xlim(-0.05, 2.5)
plt.ylim(-0.05, 2.5)
plt.xlabel('zajci')
plt.ylabel('lisice')
plt.show()