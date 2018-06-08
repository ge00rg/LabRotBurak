import numpy as np
from scipy.stats import norm
from fsave import fsave
import h5py
import os

import matplotlib.pyplot as plt

func = 's_pos'
N = 1

t_end = 200
dt = 0.0001

lambda_d = 10
lambda_v = 0
sigma_v = 1e-3

g_scale = 0.1

mu = 0
nu = 0

s_amplitude = 100
sigma_s = 0  # 0.3*s_amplitude*0.01
t_off = 0.2


def s(t):
    if t < t_off:
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

times = np.arange(0, t_end, dt)
def network():
    gamma = np.ones(N)*g_scale

    omega_f = np.outer(gamma.T, gamma) + mu*lambda_d**2*np.identity(N)
    omega_s = lambda_d*np.outer(gamma.T, gamma)
    threshold = 0.5*(nu*lambda_d + mu*lambda_d**2 + gamma**2)


    v_state = np.zeros(N)
    slow_state = np.zeros(N)

    spike_times = [[] for i in range(N)]

    rates = np.zeros((times.shape[0], N))
    x_hat = np.zeros((times.shape[0]))
    x = np.zeros(times.shape[0])

    for k, t in enumerate(times[:-1]):
        # if k%100 == 0:
        #     print('{}/{} timesteps complete'.format(k, times.shape[0]))
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
    return x, x_hat, rates, spike_times

if True:
    n_sims = 200
    n_off = int(t_off/ dt)
    x_tot = np.zeros((n_sims, times.shape[0]))
    for i in range(n_sims):
        print('iteration {}'.format(i))
        x, x_hat, rates, spike_times = network()
        x_tot[i, :] = x_hat
    dir = 'new_plots_autapse/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}'.format(N, lambda_v, mu, nu, sigma_v, t_end, func)
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass
    f = h5py.File('new_plots_autapse/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}/x.hdf5'.format(N, lambda_v, mu, nu, sigma_v, t_end, func), 'w')
    f['x_hat'] = x_tot
    f['x'] = x
    f['times'] = times
    f['t_off'] = t_off
    f['dt'] = dt
    f['n_off'] = n_off
    f.close()
    x_mns = np.mean(x_tot, axis=0)
    x_vars = np.mean((x_tot - x_mns) ** 2, axis=0)
    plt.plot(times[n_off:], x_mns, label='mean')
    plt.plot(times[n_off:], x_vars, label='variance')
    plt.plot(times[n_off:], x_vars / (2 * times[n_off:]), label='diffusion coeff.')
    plt.legend()
    plt.show()

if False:
    fig, axes = plt.subplots(2, sharex='col', gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(times, x_hat, label='$\hat{x}$', color='r', lw=2)
    axes[0].plot(times, x, label='$x$', lw=1)

    axes[0].scatter(spike_times[0], np.zeros(len(spike_times[0])), color='k', s=1, alpha=0.7)
    axes[0].legend(frameon=False)
    axes[0].set_title('true and approximated signal, spike trains')

    axes[1].plot(times, np.mean(rates[:, :], axis=1), label='pos. kernel')
    axes[1].plot(times, np.mean(rates[:, :], axis=1), label='neg. kernel')
    axes[1].legend(frameon=False)
    axes[1].set_title('mean population firing rates')

    plt.tight_layout()
    plt.show()

    dir = 'new_plots_autapse/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}'.format(N, lambda_v, mu, nu, sigma_v, t_end, func)
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

    fig.savefig('new_plots_autapse/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}/full.png'.format(N, lambda_v, mu, nu, sigma_v, t_end, func))

    fig2, axes = plt.subplots(2, sharex='col', gridspec_kw={'height_ratios': [3, 1]})

    tt = int(5/dt)
    axes[0].plot(times[:tt], x_hat[:tt], label='$\hat{x}$', color='r', lw=2)
    axes[0].plot(times[:tt], x[:tt], label='$x$', lw=1)
    x_max = np.max(x)
    x_min = np.min(x)
    to_plot = [t for t in spike_times[0] if t < 5]
    axes[0].scatter(to_plot, np.zeros(len(to_plot)), color='k', s=1, alpha=0.7)
    axes[0].legend(frameon=False)
    axes[0].set_title('true and approximated signal, spike trains')

    axes[1].plot(times[:tt], np.mean(rates[:, :], axis=1)[:tt], label='pos. kernel')
    axes[1].plot(times[:tt], np.mean(rates[:, :], axis=1)[:tt], label='neg. kernel')
    axes[1].legend(frameon=False)
    axes[1].set_title('mean population firing rates')

    plt.tight_layout()
    plt.show()

    fig2.savefig('new_plots_autapse/N={}_l_v={}_mu={}_nu={}_sig_v={}_t_end={}_s={}/start.png'.format(N, lambda_v, mu, nu, sigma_v, t_end, func))
