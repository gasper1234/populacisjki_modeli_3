from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

r = 1.1
p = 1

def gen_0(t, state):
	dydt = np.zeros_like(state)

	dydt[0] = state[0]/p * (state[1]-1)
	dydt[1] = r - p*state[1] * (state[0]+1)

	return dydt

def plot_pand(res):
	plt.plot(res.t, res.y[0], label='F')
	plt.plot(res.t, res.y[1], label='A')

T = 40
t_evaluate = np.linspace(0, T, T*100)

init_0 = [2, 2]
for p in [0.3, 0.6, 1.3]:
	for r in [0.3, 0.6, 1.3]:
		res = solve_ivp(gen_0, (0, T), init_0, t_eval=t_evaluate, method='DOP853',rtol=1e-12,atol=1e-12)
		plt.plot(res.y[0], res.y[1], label='p='+str(p)+', r='+str(r))

plt.xlabel('A')
plt.ylabel('F')
plt.legend()
plt.show()
