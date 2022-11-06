from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


p = 2

l_0 = 3

init_0 = [1, l_0]

def gen_0(t, state):
	dydt = np.zeros_like(state)

	dydt[0] = p*state[0] * (1-state[1])
	dydt[1] = state[1]/p * (state[0]-1)

	return dydt

def plot_pand(res):
	plt.plot(res.t, res.y[0], color='grey', label='z')
	plt.plot(res.t, res.y[1], color='orange', label='l')
	plt.xlabel('t')
	plt.ylabel('populacija')
	plt.legend()
	plt.show()

T = 20

t_evaluate = np.linspace(0, T, 100)

def find_t_0(res):
	t_0 = 0
	trig = 0
	for i in range(1, len(res.t)):
		if trig == 0 and res.y[0][i] > 1:
			trig = 1
		if trig == 1 and res.y[0][i] < 1:
			t_0 = res.t[i-1] + (res.t[i]-res.t[i-1]) * abs(res.y[0][i-1]-1) / abs(res.y[0][i-1]-res.y[0][i])
			break
	if t_0 == 0:
		return 10**2
	return t_0

def find_t_1(res):
	t_0 = 0
	trig = 0
	for i in range(1, len(res.t)):
		if trig == 0 and res.y[0][i] < 1:
			trig = 1
		if trig == 1 and res.y[0][i] > 1:
			t_0 = res.t[i-1] + (res.t[i]-res.t[i-1]) * abs(res.y[0][i-1]-1) / abs(res.y[0][i-1]-res.y[0][i])
			break
	if t_0 == 0:
		return 10**2
	return t_0

for p in [0.5, 1, 1.5]:
	t_0_s = []
	T_0_s = [1.01+i/20 for i in range(80)]
	for l_0 in T_0_s:
		res = solve_ivp(gen_0, (0, T), (1, l_0), t_eval=t_evaluate, method='DOP853',rtol=1e-12,atol=1e-12)
		t_0_s.append(find_t_0(res))

	#preglej kaj se dogaja z obhodnim časom
	#naplotaj še nekaj običajnih faznih diagramo za večje čase in preglej od filipa!
	t_1_s = []
	T_1_s = [1-2*i/80 for i in range(1, 39)]
	for l_0 in T_1_s:
		res = solve_ivp(gen_0, (0, T), (1, l_0), t_eval=t_evaluate, method='DOP853',rtol=1e-12,atol=1e-12)
		t_1_s.append(find_t_1(res))
	T_1_s.reverse()
	t_1_s.reverse()
	plt.plot(T_1_s + T_0_s, t_1_s + t_0_s, label='p: '+str(p))
plt.ylabel('T')
plt.xlabel(r'$l_0$')
plt.legend()
plt.show()


