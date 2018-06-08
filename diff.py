import numpy as np
import matplotlib.pyplot as plt

import h5py

fname = 'new_plots_autapse/N=1_l_v=0_mu=0_nu=0_sig_v=0.01_t_end=400_s=s_pos/x.hdf5'

f = h5py.File(fname, 'r')

x_tot = f['x_hat'][:, ::10]
x_mns = np.mean(x_tot, axis=0)
x_vars = np.mean((x_tot - x_mns) ** 2, axis=0)
plt.plot(x_mns, label='mean')
plt.plot(x_vars, label='variance')
#plt.plot(times[n_off:], x_vars / (2 * times[n_off:]), label='diffusion coeff.')
plt.legend()
plt.show()
f.close()