import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

# define parameters
J = 10                                  # dimension of x
N = 20                                  # number of neurons
t_0 = 0                                 # time at start of integration
t_end = 100                             # time at end of integration
dt = 0.01                                # integration time-step
lamb_d = 10                             # readout decay rate
lamb_V = 20                             # voltage leak constant
g_scale = 0.1                           # scale of values in decoding kernel
g_structure = 'polarized-ordered'       # structure of decoding kernel
mu = 1e-6                               # weight of quadratic penalty on rates
nu = 1e-5                               # weight of linear penalty on rates
sigma_V = 1e-3                          # standard deviation of noise in voltage

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


def make_A(J, mode='zero'):
    """

    :param J: dimension of x
    :param mode: at the moment only zero-mode exists
    :return: state-transition matrix A
    """
    if mode == 'zero':
        return np.zeros((J, J))
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


# create system objects and dependent variables
N_t = (t_end - t_0)/dt
hd_0 = h_d(0, lamb_d=lamb_d)        # should always be one
if not N_t.is_integer():
    raise ValueError('(t_end - t_0)/dt has to be a whole number')
else:
    N_t = int(N_t)

times = np.linspace(t_0, t_end - dt, N_t)

assert(times[1] - times[0] == dt)

A = make_A(J)                                           # state transition matrix
G = make_Gkernel(J, N, g_scale, structure=g_structure)  # decoder kernel
T = make_T(G)                                           # thresholds
omega_s = make_Omega_s(G, A)                            # slow connection matrix
omega_f = make_Omega_f(mu, G)                           # fast connection matrix
x = np.zeros((N_t, J))                                  # will contain x across time and dimension
V = np.zeros((N_t, N))                                  # will contain voltages across times and N
spikes = np.zeros((N_t, N))                             # will contain spikes, encoded as ones
slow_current = np.zeros((N_t, N))                       # will contain slow currents. Possibly only save current state
# let V start at 0 for now

# testing
#set_spikes(spikes, int(N_t/3), int(2*N_t/3), 10)
#plot_spike_trains(spikes)


if x_init == 'normal':
    x[0] = np.random.normal(size=J)

# euler-integrator
for i, t in enumerate(times[:-1]):
    # update x
    x[i + 1] = x[i] + (A@x[i] + c(t, x[i], m, t_on, t_off)) * dt
    slow_current[i + 1] = slow_current[i] - lamb_d*slow_current[i]*dt + spikes[i]*hd_0

    spikes[i] = (V[i] >= T).astype(int)

    V[i + 1] = V[i] + omega_s@slow_current[i]*dt - omega_f@spikes[i] + \
        (-lamb_d*V[i] + G.T@c(t, x[i], m, t_on, t_off) + np.random.normal(0, sigma_V, size=N))*dt

    # does the dt belong with the slow current? Probably yes, because otherwise it would not trail behind fast.
    # check if voltage does what is necessary. If not, see question marks
    # later feedback xhat
    # possibly tweak lamb_d

# testing
#test_matrices(J, N, g_scale, g_structure, mu)
#plot_x(x, t_on, t_off)
plot_spike_trains(spikes)
plot_slow_currents(slow_current)
plt.plot(V[:, 0])
plt.show()
