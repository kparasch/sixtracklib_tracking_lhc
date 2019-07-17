import pickle
import pysixtrack
import numpy as np
import sys
sys.path.append('..')
import helpers as hp
import footprint


epsn_x = 3.5e-6
epsn_y = 3.5e-6
r_max_sigma = 6.
N_r_footp = 10.
N_theta_footp = 10.
n_turns_beta = 150

fLine = 'line_from_mad_with_bbCO.pkl'
fParticleCO = 'particle_on_CO_mad_line.pkl'
fOptics = 'optics_mad.pkl'


with open(fLine, 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

with open(fParticleCO, 'rb') as fid:
    partCO = pickle.load(fid)

with open(fOptics, 'rb') as fid:
    optics = pickle.load(fid)

part = pysixtrack.Particles(**partCO)

beta_x = optics['betx']
beta_y = optics['bety']

sigmax = np.sqrt(beta_x * epsn_x / part.beta0 / part.gamma0)
sigmay = np.sqrt(beta_y * epsn_y / part.beta0 / part.gamma0)

xy_norm = footprint.initial_xy_polar(r_min=1e-2, r_max=r_max_sigma, r_N=N_r_footp + 1,
                                     theta_min=np.pi / 100, theta_max=np.pi / 2 - np.pi / 100,
                                     theta_N=N_theta_footp)

DpxDpy_wrt_CO = np.zeros_like(xy_norm)

for ii in range(xy_norm.shape[0]):
    for jj in range(xy_norm.shape[1]):

        DpxDpy_wrt_CO[ii, jj, 0] = xy_norm[ii, jj, 0] * np.sqrt(epsn_x / part.beta0 / part.gamma0 / beta_x)
        DpxDpy_wrt_CO[ii, jj, 1] = xy_norm[ii, jj, 1] * np.sqrt(epsn_y / part.beta0 / part.gamma0 / beta_y)

with open('DpxDpy_for_footprint.pkl', 'wb') as fid:
    pickle.dump({
                'DpxDpy_wrt_CO': DpxDpy_wrt_CO,
                'xy_norm': xy_norm,
                }, fid)

