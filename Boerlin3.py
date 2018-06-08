import numpy as np
from scipy.stats import norm
from fsave import fsave
import h5py

import matplotlib.pyplot as plt

################# DEBUG ZONE #########################
# - spike fired at t == 0
# - record slow synapses
######################################################

N = 20

t_end = 20
dt = 0.0001

lambda_d = 10
lambda_v = 20
sigma_v = 1e-3

g_scale = 0.158

mu = 1e-6
nu = 1e-5

s_amplitude = 50
sigma_s = 0.3*s_amplitude*0.01
t_off = 2


def s(t):
    if t < 0.2:
        return s_amplitude
    else:
        return 0


def neural_noise(sigma):
    try:
        return np.random.normal(scale=sigma)
    except ValueError:
        return 0

gamma = np.ones(N)*g_scale
gamma[int(N/2):] *= -1

omega_f = np.outer(gamma.T, gamma) + mu*lambda_d**2*np.identity(N)
omega_s = lambda_d*np.outer(gamma.T, gamma)
threshold = 0.5*(nu*lambda_d + mu*lambda_d**2 + gamma**2)

times = np.arange(0, t_end, dt)
order = np.random.permutation(N)
x = np.zeros(times.shape[0])
v = np.zeros((times.shape[0], N))
slow_synapses = np.zeros(N)

spike_times = [[] for i in range(N)]

slow_record = np.zeros((times.shape[0], N))
rates = np.zeros((times.shape[0], N))
x_hat = np.zeros((times.shape[0]))

state = np.zeros(N)

for k, t in enumerate(times[:-1]):
    if k%100 == 0:
        print('{}/{} timesteps complete'.format(k, times.shape[0]))

    # get c
    if sigma_s == 0 or t > t_off:
        c_noise = 0
    else:
        c_noise = np.random.normal(scale=sigma_s)/np.sqrt(dt)  # we divide by sqrt(dt) so we can integrate with dt later
    c = s(t) + c_noise
    # get x
    x[k + 1] = x[k] + c*dt

    # update slow synapses
    slow_synapses -= lambda_d*slow_synapses*dt
    slow_record[k] = slow_synapses

    # rates
    rates[k + 1] = rates[k] - lambda_d*rates[k]*dt

    # x_hat
    x_hat[k + 1] = x_hat[k] - lambda_d*x_hat[k]*dt

    # update v
    order = np.random.permutation(N)
    state[:] = v[k]
    for idx, i in enumerate(order):
        state[i] += (-lambda_v*v[k, i] + gamma[i]*c + omega_s[i, :]@slow_synapses)*dt \
                    + neural_noise(sigma_v)*np.sqrt(dt)
        # fast synapses
        if state[i] > threshold[i]:
            state[:] -= omega_f[i, :]

            slow_synapses[i] += 1
            rates[k + 1, i] += 1
            x_hat[k + 1] += gamma[i]

            spike_times[i].append(t)
    v[k + 1] = state[:]

for i in range(N):
    plt.scatter(spike_times[i], np.ones(len(spike_times[i]))*i*0.1, color='k', s=1)
plt.show()
plt.plot(times, v[:, 0])
plt.axhline(threshold[0], ls=':')
plt.show()

plt.plot(times, np.mean(rates[:, :int(N/2)], axis=1))
plt.plot(times, np.mean(rates[:, int(N/2):], axis=1))
plt.show()

plt.plot(times, x_hat)
plt.plot(times, x)
plt.show()

pspint = np.trapz(v[:, 0], dx=dt)
plt.plot(times[1:], v[1:, 0])
plt.axhline(0, ls=':', color='k')
plt.title('PSP, $\int PSP={0: .5f}$'.format(pspint))
plt.xlabel('time(s)')
plt.ylabel('voltage(V)')
plt.show()