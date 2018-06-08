import numpy as np
import matplotlib.pyplot as plt

N = 400

gamma_scale = 0.1
gamma = np.ones(N)*gamma_scale
gamma[int(N/2):] *= -1

lambda_d = 20

sigma_s = 0
t_off = 2

dt = 0.001
t_end = 1.2

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
        return -2*s_amplitude
    else:
        return 0

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
    x_hat[k + 1] = x_hat[k] -lambda_d*x_hat[k]*dt
    rates[k+1] = (2/(N*gamma[0]**2))*(gamma*c*dt + (1/lambda_d)*omega_s@rates[k]).clip(min=0)
    for i in range(N):
        if np.random.uniform() <= rates[k+1, i]*dt*lambda_d:
            spike_times[i].append(t)
            x_hat[k + 1] += gamma[i]

plt.plot(np.mean(rates[:, :int(N/2)], axis=1))
plt.plot(np.mean(rates[:, int(N/2):], axis=1))
plt.show()

#for i in range(N):
#    plt.scatter(spike_times[i], np.ones(len(spike_times[i]))*i, color='k', s=1)
#plt.show()

plt.plot(times, x)
plt.plot(times, x_hat)
x_max = np.max(x)
x_min = np.min(x_hat)
for i in range(N):
    plt.scatter(spike_times[i], np.ones(len(spike_times[i]))*((x_max-x_min)*i/N) + x_min, color='k', s=1)
plt.show()