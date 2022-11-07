from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

r = 0.6
p = 1

x_lim = 2
y_lim = x_lim

F, A = np.meshgrid(np.linspace(0,x_lim,100),np.linspace(0,y_lim,100))

u = F/p * (A-np.ones(len(A)))
v = - p*A * (F+np.ones(len(A))) + r*np.ones(len(A))
speed = np.sqrt(u**2 + v**2)
lw = speed/speed.max()*3

plt.streamplot(F,A,u,v, density=x_lim, color='k', linewidth=lw)
plt.xlim(-0.05, x_lim)
plt.ylim(-0.05, y_lim)
plt.plot([0], [r/p], 'ro')
plt.plot([r/p-1], [1], 'ro')
plt.xlabel('F')
plt.ylabel('A')
plt.show()