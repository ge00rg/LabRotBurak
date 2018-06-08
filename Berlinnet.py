import numpy as np
from scipy.stats import norm
from fsave import fsave
import h5py

import matplotlib.pyplot as plt

# define parameters
J = 1                                   # dimension of x
N = 400                                 # number of neurons
unit_scale = 1
t_0 = 0                                 # time at start of integration
t_end = 20                               # time at end of integration [s]
dt = 0.0001                             # integration time-step [s]
lamb_d = 10                             # readout decay rate
lamb_V = 20                             # voltage leak constant
g_scale = 0.158*np.sqrt(unit_scale)                           # scale of values in decoding kernel
g_structure = 'polarized-ordered'       # structure of decoding kernel
mu = 1e-6*unit_scale                               # weight of quadratic penalty on rates
nu = 1e-5*unit_scale                               # weight of linear penalty on rates
sigma_V = 1e-1*unit_scale                          # standard deviation of noise in voltage
sigma_s = 0.01                          # standard deviation of noise in stimulus
lamb_s = 0
ampli = 100*np.sqrt(unit_scale)
print(ampli)
cap = False

x_init = 'zero'     # value to which x is initialized. 'zero' initializes at zero, 'normal' - random normal

m = norm.pdf(np.linspace(-2, 2, J))      # value of memory
t_on = 20
t_off = 50


def c(t, x, m, t_on, t_off):
    """
    Command signal function c(t)
    :param t: time
    :param x: dynamic variable, dimension J
    :param m: memory, dimension J
    :param t_on: time at which memory is presented
    :param t_off: time at which it is switched off
    :return: command signal at time t
    """
    if t_on <= t < t_off:
        return m - x
    else:
        return np.zeros(x.shape[0])
# possibly add noise


def c(t, noise=True, amplitude=50, sigmamult=1):
    if noise:
        rnd = np.random.normal(scale=sigma_s, size=J)
    else:
        rnd = 0
    # print(rnd)
    if 0.0 <= t <0.2:
        return np.zeros(J) + rnd * sigmamult
    elif 0.2 < t < 0.5:
        return np.ones(J)*amplitude + rnd*sigmamult
    elif 0.5 <= t < 0.7:
        return np.zeros(J) + rnd*sigmamult
    elif 0.7 <= t < 1.0:
        return np.ones(J)*(-2)*amplitude + rnd*sigmamult
    elif 1.0 <= t < 1.2:
        return np.zeros(J) + rnd * sigmamult
    else:
        return np.zeros(J)

# def c(t):
#     if t < 0.5:
#         return np.ones(J)*50
#     else:
#         return np.zeros(J)


def make_Gkernel(J, N, scale, structure='uniform'):
    """
    :param scale: avegrage magnitude of weights
    :param structure: 'uniform' or 'polarized' or 'polarized-ordered'
    :return: matrix of output weights
    """
    if structure == 'uniform':
        return np.ones((J, N)) * scale
    if structure == 'polarized':
        Gkernel = np.ones((J, N)) * scale
        cols_to_invert = np.random.choice(N, size=int(N/2), replace=False)
        Gkernel[:, cols_to_invert] *= -1
        return Gkernel
    if structure == 'polarized-ordered':
        Gkernel = np.ones((J, N)) * scale
        Gkernel[:, int(N/2):] *= -1
        return Gkernel
    else:
        raise ValueError('invalid structure parameter')


def make_A(J, mode='zero', lamb_s=None):
    """

    :param J: dimension of x
    :param mode: at the moment only zero-mode exists
    :return: state-transition matrix A
    """
    if mode == 'zero':
        return np.zeros((J, J))
    if mode == 'leaky':
        if lamb_s == None:
            raise ValueError('lamb_s has to be set in this mode')
        return -np.identity(J)*lamb_s
    else:
        raise ValueError('incorrect mode')


def h_d(tau, lamb_d=10):
    """
    :param tau: time
    :param lamb_d: decay parameter
    :return: impact of spike after t
    """
    return np.exp(-lamb_d * tau)


def make_Omega_f(mu, G, lamb_d=10):
    """
    :param mu: quadratic penalty from error expression
    :param G: output weight Kernel
    :param lamb_d: decay parameter
    :return: fast connection matrix
    """
    return G.T@G + mu * lamb_d**2 * np.identity(N)


def make_Omega_s(G, A, lamb_d=10):
    """
    :param G: output weight Kernel
    :param A: dynamics matrix
    :param lamb_d: decay parameter
    :return: slow connection matrix
    """
    return G.T@(A + lamb_d * np.identity(J))@G


def make_T(G):
    """
    :param G: decoder kernel
    :return: thresholds T
    """
    return (np.sum(G**2, axis=0) + nu*lamb_d + mu*lamb_d**2)/2

def test_matrices(J, N, scale, structure, mu):
    G = make_Gkernel(J, N, scale, structure=structure)
    A = make_A(J)
    omega_s = make_Omega_s(G, A)
    omega_f = make_Omega_f(mu, G)
    f, axes = plt.subplots(2, 2)
    Aim = axes[0, 0].imshow(A)
    Gim = axes[0, 1].imshow(G)
    omfim = axes[1, 0].imshow(omega_f)
    omsim = axes[1, 1].imshow(omega_s)
    f.colorbar(Aim, ax=axes[0,0])
    f.colorbar(Gim, ax=axes[0,1])
    f.colorbar(omfim, ax=axes[1,0])
    f.colorbar(omsim, ax=axes[1,1])
    axes[0, 0].set_title('A')
    axes[0, 1].set_title('G')
    axes[1, 0].set_title('$\Omega_f$')
    axes[1, 1].set_title('$\Omega_s$')
    plt.tight_layout()
    plt.show()


def set_spikes(spikes, t_0, t_end, n_spikes):
    for i in range(N):
        spike_times = np.random.choice(np.arange(t_0, t_end), size=n_spikes, replace=False)
        spikes[spike_times, i] = 1


def plot_spike_trains(spikes):
    f, axes = plt.subplots(N)
    for i in range(N):
        axes[i].plot(spikes[:, i])
        axes[i].set_axis_off()
    plt.show()


def plot_x(x, t_on, t_off):
    for i in range(J):
        plt.plot(times, x[:, i])
        plt.axvline(t_on, linestyle='--', color='k', alpha=0.5)
        plt.axvline(t_off, linestyle='--', color='k', alpha=0.5)
    plt.show()


def plot_slow_currents(slow_current):
    f, axes = plt.subplots(N, 1, sharex=True, sharey=True)
    for i in range(N):
        axes[i].plot(slow_current[:, i])
        axes[i].set_axis_off()
    plt.show()


def plot_network(x_hat, x, times, spikes, r, N, c_scale=0.1, n_neurons=4, save=False):
    f, axes = plt.subplots(3, 2, sharex='col', sharey='row')
    axes[0 ,0].plot(times, x_hat, label='$\hat{x}$', linewidth=5)
    axes[0, 0].plot(times, x, label='$x$')
    # axes[0, 0].plot(times, c_scale*np.vectorize(c)(times), label='${}c(t)$'.format(c_scale))
    axes[0, 1].plot(times, x_hat, label='$\hat{x}$', linewidth=5)
    axes[0, 1].plot(times, x, label='$x$')
    # axes[0, 1].plot(times, c_scale * np.vectorize(c)(times), label='${}c(t)$'.format(c_scale))
    axes[0, 1].legend()

    num_pos = np.random.choice(int(N/2), size=n_neurons, replace=False)
    num_neg = np.random.choice(int(N/2), size=n_neurons, replace=False) + int(N/2)
    for k, i in enumerate(num_pos):
        spike_times = times[spikes[:, i].astype(bool)]
        axes[1, 0].scatter(spike_times, np.ones(spike_times.shape[0])*0.2*k, s=1, c='r')
        axes[1, 1].scatter(spike_times, np.ones(spike_times.shape[0])*0.2*k, s=1, c='r')
    for k, i in enumerate(num_neg):
        spike_times = times[spikes[:, i].astype(bool)]
        axes[1, 0].scatter(spike_times, np.ones(spike_times.shape[0])*0.2*k+n_neurons*0.2, s=1, c='b')
        axes[1, 1].scatter(spike_times, np.ones(spike_times.shape[0])*0.2*k+n_neurons*0.2, s=1, c='b')
    axes[2, 0].plot(times, np.mean(r[:, :int(N/2)], axis=1), color='r')
    axes[2, 0].plot(times, np.mean(r[:, int(N/2):], axis=1), color='b')
    axes[2, 1].plot(times, np.mean(r[:, :int(N / 2)], axis=1), color='r')
    axes[2, 1].plot(times, np.mean(r[:, int(N / 2):], axis=1), color='b')

    for i in range(3):
        axes[i, 0].set_xlim(0, t_end)
        axes[i, 1].set_xlim(t_end-2, t_end)
    if save:
        name = 'N={}_t-end={}_dt={}_lambD={}_lambV={}_mu={}_nu={}_sigV={}_sigS={}_lambS={}_mu={}_nu={}'.format(
            N, t_end, dt, lamb_d, lamb_V, mu, nu, sigma_V, sigma_s, lamb_s, mu, nu)
        f.savefig(name+'.png', dpi=400)
        fsave(f, name)
    plt.show()

    g = plt.figure()
    plt.plot(times, x_hat, label='$\hat{x}$')
    plt.plot(times, x, label='$x$')
    plt.legend()
    if save:
        name = 'complete_N={}_t-end={}_dt={}_lambD={}_lambV={}_mu={}_nu={}_sigV={}_sigS={}_lambS={}_mu={}_nu={}'.format(
            N, t_end, dt, lamb_d, lamb_V, mu, nu, sigma_V, sigma_s, lamb_s, mu, nu)
        g.savefig(name+'.png', dpi=400)
        fsave(g, name)
    plt.show()


# create system objects and dependent variables
N_t = np.round((t_end - t_0) / dt, decimals=9)
hd_0 = h_d(0, lamb_d=lamb_d)        # should always be one
if not N_t.is_integer():
    raise ValueError('(t_end - t_0)/dt has to be a whole number')
else:
    N_t = int(N_t)

times = np.linspace(t_0, t_end - dt, N_t)

t_diff = np.round(times[1] - times[0], decimals=9)

# assert(np.equal(times[1] - times[0], dt))

A = make_A(J)                                           # state transition matrix
G = make_Gkernel(J, N, g_scale, structure=g_structure)  # decoder kernel
T = make_T(G)                                           # thresholds
omega_s = make_Omega_s(G, A, lamb_d=lamb_d)             # slow connection matrix
omega_f = make_Omega_f(mu, G, lamb_d=lamb_d)            # fast connection matrix
x = np.zeros((N_t, J))                                  # will contain x across time and dimension
V = np.zeros((N_t, N))                                  # will contain voltages across times and N
spikes = np.zeros((N_t, N))                             # will contain spikes, encoded as ones
slow_current = np.zeros((N_t, N))                       # will contain slow currents. Possibly only save current state
current_spikes = np.zeros((N, N))                       # spikes during one cycle from neuron i to neuron i
# let V start at 0 for now
print(omega_f[:,0])

# test_matrices(J, N, g_scale, g_structure, mu)

# testing
# set_spikes(spikes, int(N_t/3), int(2*N_t/3), 1)
# plot_spike_trains(spikes)

# snapshot = np.zeros(N)

if x_init == 'normal':
    x[0] = np.random.normal(size=J)


np.random.seed(0)
#c_ints = np.zeros(N_t)
#c_noises = np.zeros(N_t)
#neural_noises = np.zeros((N_t, N))
#for i, t in enumerate(times):
#    c_ints[i] = c(t, amplitude=ampli, noise=False)
#    if t < 1.2:
#        c_noises[i] = np.random.normal(0, sigma_s, size=J)*ampli
#    else:
#        c_noises[i] = np.zeros(J)
#    neural_noises[i] = np.random.normal(0, sigma_V, size=N)
# euler-integrator
order = np.arange(N)
state = np.zeros(N)
for i, t in enumerate(times[:-1]):
    if i%100 == 0:
        print('{}/{} timesteps complete'.format(i, N_t))
    # debugger

    t_nonoise = 0.004
    c_int = c(t, amplitude=ampli, noise=False)
    if t >= t_nonoise:
        c_noise = np.zeros(J)
    else:
        c_noise = np.random.normal(0, sigma_s, size=J)*ampli*2
    #c_int = np.array([c_ints[i]])
    #c_noise = np.array([c_noises[i]])
    # c_noise = np.array([0])
    # c_int = np.array([0])

    # update x
    # x[i + 1] = x[i] + (A@x[i] + c(t)) * dt
    x[i + 1] = x[i] + (A @ x[i] + c_int) * dt + c_noise * np.sqrt(dt)
    slow_current[i + 1] = slow_current[i] - lamb_d*slow_current[i]*dt + spikes[i]*hd_0

    # order = np.random.permutation(N)
    state[:] = V[i]
    count = 0
    for k in order:
        neural_noise = np.random.normal(0, sigma_V)
        #neural_noise = neural_noises[i, k]
        # neural_noise = 0

        # fast_current = omega_f[:, k] @ current_spikes[:, k]
        fast_current = np.zeros(J)
        leakval = V[i, k]
        # leakval = snapshot[k]

        # V[i + 1, k] = leakval + omega_s[:, k] @ slow_current[i + 1] * dt - fast_current + \
        #     -lamb_d * leakval * dt + G[:, k].T @ c_int * dt + G[:, k].T @ c_noise * np.sqrt(dt) + \
        #     neural_noise * np.sqrt(dt)
        state[k] += omega_s[k, :] @ slow_current[i + 1] * dt - fast_current + \
            -lamb_V * leakval * dt + G[:, k].T @ c_int * dt + G[:, k].T @ c_noise * np.sqrt(dt) + \
            neural_noise * np.sqrt(dt)
        # snapshot[k] = V[i + 1, k]

        current_spikes[:, k] = 0
        # spikeval = int(V[i + 1, k] >= T[k])
        spikeval = int(state[k] >= T[k])
        # new test
        # if t == 0 and k == 0:
        #     spikeval = 1
        spikes[i + 1, k] = spikeval
        if spikeval == 1:
            count += 1
            if cap:
                V[i + 1, k] = T[k]
            current_spikes[k, :] = spikeval
            state[:] -= omega_f[k, :]
            spikeval = 0
        # slow_current[i + 1, k] = slow_current[i, k] - lamb_d * slow_current[i, k] * dt + spikes[i, k] * hd_0
    V[i + 1] = state
        # does the dt belong with the slow current? Probably yes, because otherwise it would not trail behind fast.

# decoder
r = np.zeros((N_t, N))

for i in range(N_t - 1):
    r[i + 1] = r[i] - lamb_d*r[i]*dt + lamb_d*spikes[i]
x_hat = 1/lamb_d * np.sum(G*r, axis=1)

fle = h5py.File('first_save.hdf5', 'w')
fle['x'] = x
fle['x_hat'] = x_hat
fle['r'] = r
fle['spikes'] = spikes
fle['V'] = V
fle.close()



test_matrices(J, N, g_scale, g_structure, mu)

f, axes = plt.subplots(4, 2, sharex='col')
a = np.random.choice(int(N/2), size=4, replace=False)
b = np.random.choice(int(N/2), size=4, replace=False) + int(N/2)
for i in range(4):
    axes[i, 0].plot(V[:, a[i]])
    axes[i, 1].plot(V[:, b[i]])
    axes[i, 0].axhline(T[0], linestyle='--', color='r')
    axes[i, 1].axhline(T[0], linestyle='--', color='r')
plt.show()

plt.plot(times, x, label='$x$')
plt.plot(times, x_hat, label='$\hat{x}$')
plt.legend()
plt.show()

plot_network(x_hat, x, times, spikes, r, N, n_neurons=int(3/8*N), save=True)
