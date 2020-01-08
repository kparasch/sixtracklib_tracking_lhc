import pickle
import pysixtrack
import numpy as np
import NAFFlib

import sys 
sys.path.append('..')
import myfilemanager_sixtracklib as mfm 
import helpers as hp
import footprint
import matplotlib.pyplot as plt 

import os
import time

### initial coordinates
import footprint

i=int(sys.argv[1])

fLine = 'line_from_mad_with_bbCO.pkl'
fParticleCO = 'particle_on_CO_mad_line.pkl'
fOptics = 'optics_mad.pkl'

with open(fOptics, 'rb') as fid:
    optics_dict = pickle.load(fid)
    optics_dict['epsn_x'] = 2.e-6
    optics_dict['epsn_y'] = 2.e-6

init_delta = 0.
r_max_sigma = 10.
r_min_sigma = 0.05
r_step = 0.05
N_r = int(round((r_max_sigma-r_min_sigma)/r_step)) + 1
theta_step = (3./180.)*np.pi
theta_min_rad = (i)*theta_step
theta_max_rad = (i+1)*theta_step
N_theta = int(round((theta_max_rad-theta_min_rad)/theta_step))

epsg_x = optics_dict['epsn_x']/(optics_dict['beta0']*optics_dict['gamma0'])
epsg_y = optics_dict['epsn_y']/(optics_dict['beta0']*optics_dict['gamma0'])

xy_norm = footprint.initial_xy_polar(r_min=r_min_sigma, r_max=r_max_sigma, r_N=N_r,
                                     theta_min=theta_min_rad, theta_max=theta_max_rad,
                                     theta_N=N_theta)

A1 = xy_norm[0,:,0] * np.sqrt(epsg_x)
A2 = xy_norm[0,:,1] * np.sqrt(epsg_y)
phi1 = 0.
phi2 = 0.

norm_delta = init_delta*optics_dict['invW'][5,5]

init_normalized_6D = np.empty([len(A1),6])
init_normalized_6D[:,0] = A1 * np.cos(phi1)
init_normalized_6D[:,1] = A1 * np.sin(phi1)
init_normalized_6D[:,2] = A2 * np.cos(phi2)
init_normalized_6D[:,3] = A2 * np.sin(phi2)
init_normalized_6D[:,4] = 0.
init_normalized_6D[:,5] = norm_delta

init_denorm = np.tensordot(optics_dict['W'], init_normalized_6D, [1,1]).T

input_data = {'init_x' :    init_denorm[:,0],
              'init_px' :   init_denorm[:,1],
              'init_y' :    init_denorm[:,2],
              'init_py' :   init_denorm[:,3],
              'init_zeta' : init_denorm[:,4],
              'init_delta' :init_denorm[:,5],
              'init_delta_wrt_CO' : init_delta,
              'A1' : A1,
              'A2' : A2
             }

with open(fLine, 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

with open(fParticleCO, 'rb') as fid:
    partCO = pickle.load(fid)
part = pysixtrack.Particles(**partCO)




print('N_r = %d'%N_r)
print('N_theta = %d'%N_theta)
print('Number of particles: %d'%(xy_norm.shape[0]*xy_norm.shape[1]))
###################################################################################################

# track_with = 'PySixtrack'
# track_with = 'Sixtrack'
track_with = 'Sixtracklib'
#device = 'opencl:0.0'
device = None

disable_BB = False
n_turns = 1000000



if len(sys.argv) > 2:
    out_fname = 'dynap_sixtracklib.'+sys.argv[2]+'.'+sys.argv[1]+'.h5'
else:
    out_fname = 'dynap_sixtracklib.h5'

with open('line_from_mad_with_bbCO.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))
line.remove_inactive_multipoles(inplace=True)
line.remove_zero_length_drifts(inplace=True)
line.merge_consecutive_drifts(inplace=True)

if disable_BB:
    line.disable_beambeam()

if track_with == 'Sixtracklib':
    output_dict = hp.track_particle_sixtracklib_firstlast(
        line=line, partCO=partCO, Dx_wrt_CO_m     = init_denorm[:,0],
                                  Dpx_wrt_CO_rad  = init_denorm[:,1],
                                  Dy_wrt_CO_m     = init_denorm[:,2], 
                                  Dpy_wrt_CO_rad  = init_denorm[:,3],
                                  Dsigma_wrt_CO_m = init_denorm[:,4], 
                                  Ddelta_wrt_CO   = init_denorm[:,5], 
        n_turns=n_turns, device=device)
    info = track_with
    if device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%device
else:
    raise ValueError('What?!')


mfm.dict_to_h5(output_dict, out_fname, group='output', readwrite_opts='w')
mfm.dict_to_h5(input_data, out_fname, group='input', readwrite_opts='a')
