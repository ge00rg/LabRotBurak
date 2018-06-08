from brian2 import *
import matplotlib.pyplot as plt


def visualize_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')

# notes
# default dt is used. Change accordingly later.

start_scope()
N = 4
t_end = 2*second
t_off = 1.2*second

A = 10
sigma_s = 0.01*A   # why?
tau = 1*second

lambda_v = 20*Hz
lambda_d = 10*Hz
sigma_v = 1e-3*volt/second

gamma_scale = 0.1

mu = 1e-5*volt*second**2
nu = 1e-5*volt*second

Gamma = np.ones(N)*gamma_scale*volt**0.5
Gamma[int(N/2):] *= -1
# create s(t). Noise is added during integration.
def s_func(t):
    if t < 0.2:
        return 0
    elif 0.2 <= t < 0.5:
        return A
    elif 0.5 <= t < 0.7:
        return 0
    elif 0.7 <= t < 1.0:
        return -2*A
    else:
        return 0
t_recorded = arange(int(t_end/defaultclock.dt))*defaultclock.dt
s = TimedArray(np.vectorize(s_func)(t_recorded/second)*volt**0.5, dt=defaultclock.dt)
noisevals = np.random.normal(0, sigma_s, size=t_recorded.shape)
t_off_idx = np.where(t_recorded >= t_off)[0][0]
noisevals[t_off_idx:] = 0
c_noise = TimedArray(noisevals*volt**0.5, dt=defaultclock.dt)

T = (nu*lambda_d + mu*lambda_d**2 + Gamma**2)/2
omega_f = np.outer(Gamma.T, Gamma)*volt + mu*lambda_d**2*np.identity(N)
omega_s = np.outer(Gamma.T, Gamma)*lambda_d*volt

c_eqs = '''
dx/dt = Hz*(s(t) + c_noise(t)*dt**-0.5*second**0.5) : volt**0.5
'''
network_eqs = '''
dv/dt = -lambda_v*v + Hz*gamma*(s(t) + c_noise(t)*dt**-0.5*second**0.5) + slow_cur + sigma_v*xi*second**0.5 : volt
gamma : volt**0.5
th : volt
res : volt
slow_cur : volt/second
should_spike : boolean
'''

G = NeuronGroup(N, network_eqs, threshold='should_spike == True', method='euler')
G.gamma = Gamma
G.th = T
G.res = np.diag(omega_f)

@network_operation(when='before_thresholds')
def single_neuron_threshold():
    could_spike = G.v_ > G.th_  # underscores to not use units, a bit faster
    if np.any(could_spike):
        indic, = np.nonzero(could_spike)
        # Choose one of the neurons randomly (but you could also use the neuron with maximum v, for example)
        G.should_spike[indic[np.random.randint(len(indic))]] = True

slow_eqs = '''
dsc/dt = -lambda_d*sc : 1
w : volt/second
slow_cur_post = w*sc : volt/second (summed)
'''

S_slow = Synapses(G, G, slow_eqs, on_pre='sc += 1')
S_slow.connect()

S_fast = Synapses(G, G, 'w : volt', on_pre='v_post -= w')
S_fast.connect()
for i in range(N):
    for j in range(N):
        S_fast.w[i, j] = omega_f[i, j]
        S_slow.w[i, j] = omega_s[i, j]

H = NeuronGroup(1, 'dxhat/dt = -lambda_d*xhat : 1')
S_approx = Synapses(G, H, 'w : 1', on_pre='xhat_post += w')
S_approx.connect()
S_approx.w = Gamma/volt**0.5

X = NeuronGroup(1, c_eqs, method='euler')
NoiseTest = NeuronGroup(1, 'dx/dt = Hz*(c_noise(t)*dt**-0.5*second**0.5) : volt**0.5', method='euler')

XM = StateMonitor(X, variables=True, record=True)
GM = StateMonitor(G,  variables=True, record=True)
XHATM = StateMonitor(H, variables=True, record=True)
NoiseTestM = StateMonitor(NoiseTest, variables=True, record=True)
spikemon = SpikeMonitor(G)
run(t_end)

print('noise mean: {}  noise variance: {}'.format(np.mean(np.diff(NoiseTestM.x[0][:t_off_idx])),
                                                  np.var(np.diff(NoiseTestM.x[0][:t_off_idx]))))


plot(spikemon.t, spikemon.i, '.k')
plt.show()

plt.plot(GM.t, GM.slow_cur[0])
plt.show()

plt.plot(GM.t, GM.v[0], alpha=0.5)
plt.plot(GM.t, GM.v[1], alpha=0.5)
plt.axhline(T[0]/volt, ls=':')
plt.show()

plt.plot(XM.t, XM.x[0])
plt.plot(XHATM.t, XHATM.xhat[0])
plt.show()