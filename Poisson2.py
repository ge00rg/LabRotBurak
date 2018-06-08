import numpy as np
import matplotlib.pyplot as plt

N = 400

gamma_scale = 0.1
gamma = np.ones(N)*gamma_scale
gamma[int(N/2):] *= -1

lambda_d = 10

sigma_s = 0
t_off = 1.0

dt = 0.01
t_end = 100

times = np.arange(0, t_end, dt)

s_amplitude = 50

omega_s = lambda_d*np.outer(gamma.T, gamma)

def s(t):
    if t < 0.2:
        return s_amplitude
    else:
        return 0

def s2(t):
    if 0.0 < t < 0.2:
        return 0
    elif 0.2 <= t < 0.5:
        return s_amplitude
    elif 0.5 <= t < 0.7:
        return 0
    elif 0.7 <= t < 1.0:
        return -3*s_amplitude
    else:
        return 0

def poisson_net(throw=True):
    rates = np.zeros((times.shape[0], N))
    spike_times = [[] for i in range(N)]
    x_hat = np.zeros((times.shape[0]))
    x = np.zeros((times.shape[0]))
    for k, t in enumerate(times[:-1]):
        if k%100 == 0:
            print('{}/{} timesteps complete'.format(k, times.shape[0]))

        if sigma_s == 0 or t > t_off:
            c_noise = 0
        else:
            c_noise = np.random.normal(scale=sigma_s)/np.sqrt(dt)  # we divide by sqrt(dt) so we can integrate with dt later
        c = s2(t) + c_noise

        x[k + 1] = x[k] + c * dt
        x_hat[k + 1] = x_hat[k] - lambda_d*x_hat[k]*dt
        rates[k + 1] = rates[k] - lambda_d*rates[k]*dt
        rates_inst = (2/(N*gamma[0]**2))*(gamma*c + (1/lambda_d)*omega_s@rates[k]).clip(min=0)
        for i in range(N):
            if np.random.uniform() <= rates_inst[i]*dt:
                spike_times[i].append(t)
                x_hat[k + 1] += gamma[i]
                rates[k + 1, i] += lambda_d
        if throw and t > t_off and np.all(rates[k + 1] < 1e-250):
            raise ValueError('rates hit zero')
    return x, x_hat, spike_times, rates

if False:
    x, x_hat, spike_times, rates = poisson_net(throw=True)

    # ISI
    isi_max = 0
    tot_isi = np.array([])
    for j in range(int(N/2), N):
        isi = np.diff([i for i in spike_times[j] if i >= t_off])
        tot_isi = np.concatenate((tot_isi, isi), axis=0)

    _, bins, _ = plt.hist(tot_isi, bins=40, density=True, color='k')
    plt.yscale('log', nonposy='clip')
    plt.show()


    inds = np.random.choice(int(N/2), size=4) + int(N/2)
    fig, axes = plt.subplots(2,2, sharex='all', sharey='all')
    for k, ind in enumerate(inds):
        isi = np.diff([i for i in spike_times[ind] if i >= t_off])
        axes[k % 2, k // 2].hist(isi, density=True, bins=40, range=(0, bins[-1]), color='k')
        axes[k % 2, k // 2].set_yscale('log', nonposy='clip')
    plt.show()
    # this is not perfect(bins)
    # + flattened


    fig, axes = plt.subplots(2, sharex='col', gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(times, x, label='$x$')
    axes[0].plot(times, x_hat, label='$\hat{x}$', color='r')
    x_max = np.max(x)
    x_min = np.min(x)
    for i in range(N):
        axes[0].scatter(spike_times[i], np.ones(len(spike_times[i]))*((x_max-x_min)*i/N) + x_min, color='k', s=1)
    axes[0].legend(frameon=False)
    axes[0].set_title('true and approximated signal, spike trains')

    axes[1].plot(times, np.mean(rates[:, :int(N/2)], axis=1), label='pos. kernel')
    axes[1].plot(times, np.mean(rates[:, int(N/2):], axis=1), label='neg. kernel')
    axes[1].legend(frameon=False)
    axes[1].set_title('mean population firing rates')

    plt.tight_layout()
    plt.show()

if True:
    n_off = int(t_off/dt)
    x_tot = np.zeros((100, times[n_off:].shape[0]))
    for l in range(100):
        print('iteration {}'.format(l))
        while True:
            try:
                x, x_hat, spike_times, rates = poisson_net(throw=True)
                break
            except ValueError:
                continue
        x_tot[l, :] = x_hat[n_off:]

    x_mns = np.mean(x_tot, axis=0)
    x_vars = np.mean((x_tot - x_mns)**2, axis=0)
    plt.plot(times[n_off:], x_mns, label='mean')
    plt.plot(times[n_off: ], x_vars, label='variance')
    plt.plot(times[n_off: ], x_vars/(2*times[n_off: ]), label='diffusion coeff.')
    plt.legend()
    plt.show()

# do not forget if rates hit 0
