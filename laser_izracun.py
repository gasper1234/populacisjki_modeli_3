from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

r = 2
p = 0.4

def gen_0(t, state):
	dydt = np.zeros_like(state)

	dydt[0] = state[0]/p * (state[1]-1)
	dydt[1] = r - p*state[1] * (state[0]+1)

	return dydt

def plot_pand(res):
	plt.plot(res.t, res.y[0], label='F')
	plt.plot(res.t, res.y[1], label='A')

def char_time(T, f):
	max_val = f.max()
	end_val = f[-1]
	diff = abs(max_val-end_val)
	for i in range(1, len(f)):
		if abs(f[-i]-end_val) > diff / 10:
			return T[-i]
	return None

def oscilation(T, f):
	if f[1] > f[0]:
		for i in range(2, len(f)-2):
			if f[i-2] > f[i] and f[i+2] > f[i]:
				return T[i]
	else:
		for i in range(2, len(f)-2):
			if f[i-2] < f[i] and f[i+2] < f[i]:
				return T[i]
	return None

T = 50
t_evaluate = np.linspace(0, T, T*100)

init_0 = [3, 3]

r_s = [1+i/10 for i in range(1, 100)]
rel_s = []
osc_s = []
rel_s_1 = []
osc_s_1 = []
for r in r_s:
	res = solve_ivp(gen_0, (0, T), init_0, t_eval=t_evaluate, method='DOP853',rtol=1e-12,atol=1e-12)
	rel_s.append(char_time(res.t, np.array(res.y[0])))
	osc_s.append(oscilation(res.t, res.y[0]))
	rel_s_1.append(char_time(res.t, np.array(res.y[1])))
	osc_s_1.append(oscilation(res.t, res.y[1]))
	#plt.plot(res.t, res.y[1], label=str(round(rel_s[-1],2))+'  '+str(round(osc_s[-1],2)))
	#plt.legend()
	#plt.show()

plt.plot(r_s, rel_s, label=r'$F: t_r$')
plt.plot(r_s, osc_s, label=r'$F: t_0$')
plt.plot(r_s, rel_s_1, label=r'$A: t_r$')
plt.plot(r_s, osc_s_1, label=r'$A: t_0$')

plt.xlabel('r')
plt.ylabel('A')
plt.legend()
plt.show()
