import numpy as np
from scipy.stats import norm
from fsave import fsave
import h5py
import os

import matplotlib.pyplot as plt

func = 's_pos'
N = 2

t_end = 200
dt = 0.0001

lambda_d = 10
lambda_v = 0
sigma_v = 1e-2

g_scale = 0.1

mu = 0
nu = 0

s_amplitude = 50
sigma_s = 0  # 0.3*s_amplitude*0.01
t_off = 1.2


def s(t):
    if t < 0.2:
        return s_amplitude
    else:
        return 0

def s2(t):
    if 0.0 < t <0.2:
        return 0
    elif 0.2 <= t < 0.5:
        return s_amplitude
    elif 0.5 <= t < 0.7:
        return 0
    elif 0.7 <= t < 1.0:
        return -2*s_amplitude
    else:
        return 0
def s3(t):
    return np.sin(t)*s_amplitude


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

v_state = np.zeros(N)
slow_state = np.zeros(N)

spike_times = [[] for i in range(N)]

rates = np.zeros((times.shape[0], N))
x_hat = np.zeros((times.shape[0]))
x = np.zeros(times.shape[0])

for k, t in enumerate(times[:-1]):
    if k%100 == 0:
        print('{}/{} timesteps complete'.format(k, times.shape[0]))
    # get c
    if sigma_s == 0 or t > t_off:
        c_noise = 0
    else:
        c_noise = np.random.normal(scale=sigma_s) / np.sqrt(dt)  # we divide by sqrt(dt) so we can integrate with dt later
    c = s(t) + c_noise

    rates[k + 1] = rates[k] - lambda_d * rates[k] * dt
    x_hat[k + 1] = x_hat[k] - lambda_d * x_hat[k] * dt
    x[k + 1] = x[k] + c * dt

    spiking_inds = np.where(v_state > threshold)
    if spiking_inds[0].size > 0:
        spiking_ind = np.random.choice(spiking_inds[0])

        v_state -= omega_f[:, spiking_ind]

        slow_state[spiking_ind] += 1
        rates[k + 1, spiking_ind] += lambda_d
        x_hat[k + 1] += gamma[spiking_ind]

        spike_times[spiking_ind].append(t)

    v_state += (-lambda_v*v_state + gamma*c + omega_s@slow_state)*dt + neural_noise(sigma_v)*np.sqrt(dt)
    slow_state -= lambda_d * slow_state * dt

n_select = np.min((50, int(N/2)))
pos_n = np.random.choice(int(N/2), size=n_select, replace=False)
neg_n = np.random.choice(int(N/2), size=n_select, replace=False) + int(N/2)
scatter_inds = np.concatenate((pos_n, neg_n))

fig, axes = plt.subplots(2, sharex='col', gridspec_kw={'height_ratios': [3, 1]})

axes[0].plot(times, x_hat, label='$\hat{x}$', color='r', lw=2)
axes[0].plot(times, x, label='$x$', lw=1)
x_max = np.max(x)
x_min = np.min(x)
for k, i in enumerate(scatter_inds):
    axes[0].scatter(spike_times[i], np.ones(len(spike_times[i]))*((x_max-x_min)*k/(2*n_select)) + x_min, color='k', s=1, alpha=0.7)
axes[0].legend(frameon=False)
axes[0].set_title('true and approximated signal, spike trains')

axes[1].plot(times, np.mean(rates[:, :int(N/2)], axis=1), label='pos. kernel')
axes[1].plot(times, np.mean(rates[:, int(N/2):], axis=1), label='neg. kernel')
axes[1].legend(frameon=False)
axes[1].set_title('mean population firing rates')

plt.tight_layout()
plt.show()

dir = 'new_plots/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}'.format(N, lambda_v, mu, nu, sigma_v, t_end, func)
try:
    os.mkdir(dir)
except FileExistsError:
    pass

fig.savefig('new_plots/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}/full.png'.format(N, lambda_v, mu, nu, sigma_v, t_end, func))

fig2, axes = plt.subplots(2, sharex='col', gridspec_kw={'height_ratios': [3, 1]})

tt = int(5/dt)
axes[0].plot(times[:tt], x_hat[:tt], label='$\hat{x}$', color='r', lw=2)
axes[0].plot(times[:tt], x[:tt], label='$x$', lw=1)
x_max = np.max(x)
x_min = np.min(x)
for k, i in enumerate(scatter_inds):
    to_plot = [t for t in spike_times[i] if t < 5]
    axes[0].scatter(to_plot, np.ones(len(to_plot))*((x_max-x_min)*k/(2*n_select)) + x_min, color='k', s=1, alpha=0.7)
axes[0].legend(frameon=False)
axes[0].set_title('true and approximated signal, spike trains')

axes[1].plot(times[:tt], np.mean(rates[:, :int(N/2)], axis=1)[:tt], label='pos. kernel')
axes[1].plot(times[:tt], np.mean(rates[:, int(N/2):], axis=1)[:tt], label='neg. kernel')
axes[1].legend(frameon=False)
axes[1].set_title('mean population firing rates')

plt.tight_layout()
plt.show()

fig2.savefig('new_plots/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}/start.png'.format(N, lambda_v, mu, nu, sigma_v, t_end, func))
