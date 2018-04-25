import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

# define parameters
J = 10              # dimension of x
N = 20              # number of neurons
t_0 = 0             # time at start of integration
t_end = 100         # time at end of integration
dt = 0.1            # integration time-step

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
        return 0


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

# create system objects and dependent variables
N_t = (t_end - t_0)/dt
if not N_t.is_integer():
    raise ValueError('(t_end - t_0)/dt has to be a whole number')
else:
    N_t = int(N_t)

times = np.linspace(t_0, t_end - dt, N_t)

assert(times[1] - times[0] == dt)

A = make_A(J)               # state transition matrix
x = np.zeros((N_t, J))      # will contain x across time and dimension
if x_init == 'normal':
    x[0] = np.random.normal(size=J)

# euler-integrator
for i, t in enumerate(times[:-1]):
    # update x
    x[i + 1] = x[i] + (A@x[i] + c(t, x[i], m, t_on, t_off)) * dt
    # later feedback xhat


# testing
for i in range(J):
    plt.plot(times, x[:, i])
    plt.axvline(t_on, linestyle='--', color='k', alpha=0.5)
    plt.axvline(t_off, linestyle='--', color='k', alpha=0.5)
plt.show()
