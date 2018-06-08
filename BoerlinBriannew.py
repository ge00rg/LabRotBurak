import numpy as np
import matplotlib.pyplot as plt

from brian2 import *
# set_device('cpp_standalone')

N = 40

mu = 1e-5
nu = 1e-5

gamma_scale = 0.1
gamma = np.ones(N)*gamma_scale
gamma[int(N/2):] *= -1

lambda_d = 10
lambda_d_time = 10*Hz

lambda_v = 0*Hz
sigma_v = 1*Hz

sigma_s = 0.01*50
t_off = 1.0

dt = 0.01
t_end = 200*second

t_off = 1.2

# times = np.arange(0, t_end, dt)

s_amplitude = 70

omega_s = lambda_d*np.outer(gamma.T, gamma)
omega_f = np.outer(gamma.T, gamma) + mu*lambda_d**2*np.identity(N)

threshold = 0.5*(nu*lambda_d + mu*lambda_d**2 + gamma**2)


def s2(t):
    if 0.0 < t <0.2:
        return 0
    elif 0.2 <= t < 0.5:
        return s_amplitude
    elif 0.5 <= t < 0.7:
        return 0
    elif 0.7 <= t < 1.0:
        return -1.5*s_amplitude
    else:
        return 0

def s1(t):
    if t < 0.2:
        return s_amplitude
    else:
        return 0

def s3(t):
    return np.sin(t)

t_recorded = arange(int(t_end/defaultclock.dt))*defaultclock.dt
s = TimedArray(np.vectorize(s2)(t_recorded/second), dt=defaultclock.dt)

c_noise_prep = np.random.normal(scale=sigma_s, size=t_recorded.shape[0])/np.sqrt(defaultclock.dt)
n_off = int(t_off/defaultclock.dt)
c_noise_prep[n_off:] = 0
c_noise = TimedArray(c_noise_prep*second**0.5, dt=defaultclock.dt)

eqs = '''
dv/dt = -lambda_v*v + gamma*s(t)/second + gamma*c_noise(t)/second + summed_slow  + sigma_v*second**0.5*xi: 1
gamma : 1
th : 1
dslow/dt = -lambda_d*slow/second : 1
dr/dt = -lambda_d*r/second : 1
summed_slow : Hz
should_spike : boolean
'''

reset_eqs = '''
slow += 1
r += lambda_d
should_spike = False
'''

G = NeuronGroup(N, eqs, threshold='should_spike == True', reset=reset_eqs, method='euler')
G.gamma = gamma
G.th = threshold
G.should_spike = [False for i in range(N)]

@network_operation(when='before_thresholds')
def single_neuron_threshold():
    could_spike = G.v_ > G.th_  # underscores to not use units, a bit faster
    if np.any(could_spike):
        indic, = np.nonzero(could_spike)
        # Choose one of the neurons randomly (but you could also use the neuron with maximum v, for example)
        G.should_spike[indic[np.random.randint(len(indic))]] = True

# @network_operation(when='after_thresholds')
# def reset_neuron():
#     G.should_spike = [False for i in range(N)]

H = NeuronGroup(1, 'dxhat/dt = -lambda_d*xhat/second : 1', method='euler')
I = NeuronGroup(1, 'dx/dt = s(t)/second + c_noise(t)/second : 1', method='euler')

S_fast = Synapses(G, G, 'w : 1', on_pre='v_post -= w')
S_fast.connect()

S_readout = Synapses(G, H, 'w : 1', on_pre='xhat_post += w')
S_readout.connect()
for i in range(N):
    S_readout.w[i] = gamma[i]

sum_eqs = '''
slow_w : 1
summed_slow_post = slow_w*slow_pre : Hz (summed)
'''
sum_operation = Synapses(G, G, sum_eqs)
sum_operation.connect()

for i in range(N):
    for j in range(N):
        S_fast.w[i, j] = omega_f[i, j]
        sum_operation.slow_w[i, j] = omega_s[i, j]

GM = StateMonitor(G,  variables=True, record=True, dt=0.1*second)
HM = StateMonitor(H,  variables=True, record=True, dt=0.1*second)
IM = StateMonitor(I,  variables=True, record=True, dt=0.1*second)
spikemon = SpikeMonitor(G)
run(t_end)

voltages = GM.v
spiketrains0 = spikemon.spike_trains()[0]/defaultclock.dt
spiketrains1 = spikemon.spike_trains()[1]/defaultclock.dt

f, axes = plt.subplots(5, sharex='col')
axes[0].plot(GM.t, GM.v[0], alpha=0.5)
# axes[1].plot(spikemon.t, spikemon.i, '.k', lw='.1')
axes[1].scatter(spikemon.t, spikemon.i, color='k', s=1)
axes[2].plot(GM.t, GM.v[1], alpha=0.5)
# axes[3].plot(GM.t, np.mean(GM.slow[:int(N/2)], axis=0))
# axes[3].plot(GM.t, np.mean(GM.slow[int(N/2):], axis=0))
axes[3].plot(GM.t, np.mean(GM.r[:int(N/2)], axis=0))
axes[3].plot(GM.t, np.mean(GM.r[int(N/2):], axis=0))
axes[4].plot(HM.t, HM.xhat[0])
axes[4].plot(IM.t, IM.x[0])
plt.show()