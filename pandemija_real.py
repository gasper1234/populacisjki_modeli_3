from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

alpha = 0.4
beta = 0.2
gama = 0.06
#karantena
delta = 0.03
eps = 0.05
zeta = 0.01
cep_s = [0.01*i for i in range(11)]

init_0 = [0.99, 0.01, 0., 0., 0.]

def gen_0(t, state):
	dydt = np.zeros_like(state)

	dydt[0] = -alpha * state[0] * state[2] + zeta * state[4]
	dydt[1] = alpha * state[0] * state[2] - beta * state[1]
	dydt[2] = beta * state[1] - gama * state[2] - delta * state[2]
	dydt[3] = delta * state[2] - eps * state[3]
	dydt[4] = gama * state[2] + eps * state[3] - zeta * state[4]
	
	return dydt

def plot_pand(res):
	plt.plot(res.t, res.y[0], color='limegreen', label='zdravi')
	plt.plot(res.t, res.y[1], color='red', label='okuženi')
	plt.plot(res.t, res.y[2], color='black', label='bolni')
	plt.plot(res.t, res.y[3], color='orange', label='karantena')
	plt.plot(res.t, res.y[4], color='cyan', label='imuni')
	plt.xlabel('t')
	plt.ylabel('%')
	plt.legend()
	plt.show()


t_evaluate = np.linspace(0, 300, 100)

res = solve_ivp(gen_0, (0, 300), init_0, t_eval=t_evaluate)
plot_pand(res)

'''
colormap = plt.cm.cool
num_plots = len(cep_s)
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.cool(np.linspace(0, 1, num_plots))))

for delta in cep_s:
	res = solve_ivp(gen_0, (0, 100), init_0, t_eval=t_evaluate)
	plt.plot(res.t, res.y[2], label=str(delta))

plt.xlabel('t')
plt.ylabel('%')
plt.legend()
plt.show()'''
