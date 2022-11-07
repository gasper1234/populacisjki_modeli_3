from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


p = 2

l_0 = 1.5

init_0 = [0.0001, 0.0001]

def gen_0(t, state):
	print(p)
	dydt = np.zeros_like(state)

	dydt[0] = p*state[0] * (1-state[1])
	dydt[1] = state[1]/p * (state[0]-1)

	return dydt

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_pand(res):
	fig, ax = plt.subplots(2, 1)
	
	Time = np.array(res.t)

	x = np.array(res.y[0])
	y = np.array(res.y[1])

	points = np.array([x, y]).T.reshape(-1, 1, 2)
	segments = np.concatenate([points[:-1], points[1:]], axis=1)

	norm = plt.Normalize(Time.min(), Time.max())
	lc = LineCollection(segments, cmap='jet', norm=norm)
	lc.set_array(Time)
	line = ax[0].add_collection(lc)
	fig.colorbar(line, ax=ax[0], label='t')

	ax[0].set_xlabel('zajci')
	ax[0].set_ylabel('lisice')
	ax[0].axis('equal')
	ax[1].plot(res.t, res.y[0], color='grey', label='zajci')
	ax[1].plot(res.t, res.y[1], color='orange', label='lisice')
	ax[1].legend()
	ax[1].set_ylabel('populacija')
	ax[1].set_xlabel('t')
	plt.show()

T = 50

t_evaluate = np.linspace(0, T, T*100)

res = solve_ivp(gen_0, (0, T), init_0, t_eval=t_evaluate, method='DOP853',rtol=1e-12,atol=1e-12)
plot_pand(res)