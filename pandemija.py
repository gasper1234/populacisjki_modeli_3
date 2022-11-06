from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


#solve_ivp(gen_ivp, [10**(-5), 1], iconds,method='DOP853',rtol=1e-12,atol=1e-12)
#return res1.y[0][-1]

alpha = 0.4
beta = 0.1
cep = 0.01
cep_s = [0.01*i for i in range(10)]

init_0 = [0.99, 0.01, 0]

def gen_0(t, state):
	dydt = np.zeros_like(state)
	dydt[0] = -alpha * state[0] * state[1]
	dydt[1] = alpha * state[0] * state[1] - beta * state[1]
	dydt[2] = beta * state[1]

	return dydt

def gen_cep(t, state):
	dydt = np.zeros_like(state)
	dydt[0] = -alpha * state[0] * state[1] - cep * state[0]
	dydt[1] = alpha * state[0] * state[1] - beta * state[1]
	dydt[2] = beta * state[1] + cep * state[0]

	return dydt

def gen_cep_1(t, state):
	dydt = np.zeros_like(state)
	dydt[0] = -alpha * state[0] * state[1] - cep
	dydt[1] = alpha * state[0] * state[1] - beta * state[1]
	dydt[2] = beta * state[1] + cep
	return dydt

def plot_pand(res):
	plt.plot(res.t, res.y[0], color='limegreen')
	plt.plot(res.t, res.y[1], color='red')
	plt.plot(res.t, res.y[2], color='cyan')
	plt.xlabel('t')
	plt.ylabel('%')
	plt.show()


t_evaluate = np.linspace(0, 100, 100)


fig, ax = plt.subplots(1, 1)

colormap = plt.cm.cool
num_plots = len(cep_s)
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.cool(np.linspace(0, 1, num_plots))))

for cep in cep_s:
	res = solve_ivp(gen_cep, (0, 100), init_0, t_eval=t_evaluate)
	plt.plot(res.t, res.y[1], label=str(cep))

plt.xlabel('t')
plt.ylabel('%')
plt.legend(title='c')
plt.show()