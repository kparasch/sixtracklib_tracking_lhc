import pickle
import pysixtrack
import numpy as np
import NAFFlib

import sys
sys.path.append('..')
import myfilemanager_sixtracklib as mfm
import helpers as hp
import random_hypersphere
import normalization
#import footprint
import matplotlib.pyplot as plt


import os
import time

track_with = 'Sixtracklib'
device = sys.argv[5]
#device = 'opencl:0.0' 
#device = None

n_sigmas=4
n_particles=10000

seed = 0
if len(sys.argv) > 4:
    seed = int(sys.argv[4])

disable_BB = False
init_delta=0.
if len(sys.argv) > 3:
    init_delta = int(sys.argv[3])*0.000032
n_turns = int(sys.argv[2])

with open('optics_mad.pkl', 'rb') as fid:
    optics_dict = pickle.load(fid)
optics_dict['epsn_x'] = 3.5e-6
optics_dict['epsn_y'] = 3.5e-6
init_normalized_coordinates_sigma = random_hypersphere.random_hypersphere_reject(n_sigmas, n_particles, dim=4, seed=seed)
init_normalized_coordinates = np.empty_like(init_normalized_coordinates_sigma)
init_normalized_coordinates[:,0] = init_normalized_coordinates_sigma[:,0] * np.sqrt( optics_dict['epsn_x']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['betx'])
init_normalized_coordinates[:,1] = init_normalized_coordinates_sigma[:,1] * np.sqrt( optics_dict['epsn_x']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['betx'])
init_normalized_coordinates[:,2] = init_normalized_coordinates_sigma[:,2] * np.sqrt( optics_dict['epsn_y']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['bety'])
init_normalized_coordinates[:,3] = init_normalized_coordinates_sigma[:,3] * np.sqrt( optics_dict['epsn_y']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['bety'])
init_x, init_px, init_y, init_py = normalization.denormalize_4d_uncoupled(init_normalized_coordinates, optics_dict)

#n_turns = 10000

#out_fname = 'dynap_sixtracklib_nturns_%d_delta_%.2e.h5'%(n_turns,init_delta_wrt_CO)
out_fname = sys.argv[1]+'.h5'

with open('line_from_mad_with_bbCO.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))
line.remove_inactive_multipoles(inplace=True)
line.remove_zero_length_drifts(inplace=True)
line.merge_consecutive_drifts(inplace=True)

if disable_BB:
    line.disable_beambeam()

with open('particle_on_CO_mad_line.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

print('hello')
if track_with == 'Sixtracklib':
    output_dict = hp.track_particle_sixtracklib_long(
        line=line, partCO=partCO, Dx_wrt_CO_m=init_x, Dpx_wrt_CO_rad=init_px,
        Dy_wrt_CO_m=init_y, Dpy_wrt_CO_rad=init_py,
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=init_delta, n_turns=n_turns, device=device)
    info = track_with
    if device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%device
else:
    raise ValueError('What?!')

input_data = {'init_x' : init_x,
              'init_px' : init_px,
              'init_y' : init_y,
              'init_py' : init_py,
              'init_delta' : init_delta
             }


del partCO['partid']
del partCO['turn']                                                                                                                       
del partCO['state'] 


start_compression = time.time()
mfm.dict_to_h5(output_dict, out_fname, group='output', readwrite_opts='w')
mfm.dict_to_h5(input_data, out_fname, group='input', readwrite_opts='a')
mfm.dict_to_h5(partCO, out_fname, group='closed-orbit', readwrite_opts='a')
mfm.dict_to_h5(optics_dict, out_fname, group='beam-optics', readwrite_opts='a')
#mfm.dict_to_h5_compressed(save_dict, out_fname, compression_opts=int(sys.argv[1]))
end_compression = time.time()
print('Tracking time: %f mins'%output_dict['tracking_time_mins'])
print('Compression time: %f mins'%((end_compression-start_compression)/60.))
