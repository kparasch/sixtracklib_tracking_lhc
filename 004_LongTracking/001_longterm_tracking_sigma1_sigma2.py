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
import simulation_parameters as pp
#import footprint
import matplotlib.pyplot as plt


import os
import time

track_with = 'Sixtracklib'

with open(pp.optics_filename, 'rb') as fid:
    optics_dict = pickle.load(fid)
optics_dict['epsn_x'] = 2.e-6
optics_dict['epsn_y'] = 2.e-6

epsg_x = optics_dict['epsn_x']/(optics_dict['beta0']*optics_dict['gamma0'])
epsg_y = optics_dict['epsn_y']/(optics_dict['beta0']*optics_dict['gamma0'])

#init_normalized_coordinates_sigma = random_hypersphere.random_hypersphere_reject(pp.n_sigmas, pp.n_particles, dim=4, seed=pp.seed)
init_normalized_coordinates_sigma = random_hypersphere.random_hypersphere_reject2(pp.n_sigma1, pp.n_sigma2, pp.n_particles, dim=4, seed=pp.seed)

normalized_delta = pp.init_delta_wrt_CO*optics_dict['invW'][5,5] 

init_normalized_6D = np.empty([pp.n_particles, 6])
init_normalized_6D[:,0] = init_normalized_coordinates_sigma[:,0] * np.sqrt(epsg_x)
init_normalized_6D[:,1] = init_normalized_coordinates_sigma[:,1] * np.sqrt(epsg_x)
init_normalized_6D[:,2] = init_normalized_coordinates_sigma[:,2] * np.sqrt(epsg_y)
init_normalized_6D[:,3] = init_normalized_coordinates_sigma[:,3] * np.sqrt(epsg_y)
init_normalized_6D[:,4] = 0.
init_normalized_6D[:,5] = normalized_delta

init_denormalized_coordinates = np.tensordot(optics_dict['W'], init_normalized_6D, [1,1]).T

with open(pp.CO_filename, 'rb') as fid:
    partCO = pickle.load(fid)

out_fname = pp.output_filename

with open(pp.line_filename, 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

if pp.simplify_line:
    line.remove_inactive_multipoles(inplace=True)
    line.remove_zero_length_drifts(inplace=True)
    line.merge_consecutive_drifts(inplace=True)

if pp.disable_BB:
    line.disable_beambeam()

if track_with == 'Sixtracklib':
    output_dict = hp.track_particle_sixtracklib_long(
            line=line, partCO=partCO, Dx_wrt_CO_m = init_denormalized_coordinates[:,0],
                                   Dpx_wrt_CO_rad = init_denormalized_coordinates[:,1],
                                      Dy_wrt_CO_m = init_denormalized_coordinates[:,2],
                                   Dpy_wrt_CO_rad = init_denormalized_coordinates[:,3],
                                   Dzeta_wrt_CO_m = init_denormalized_coordinates[:,4],
                                    Ddelta_wrt_CO = init_denormalized_coordinates[:,5],
                                n_turns=pp.n_turns, device=pp.device)
    info = track_with
    if pp.device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%pp.device
else:
    raise ValueError('What?!')

input_data = {'init_x' :    init_denormalized_coordinates[:,0],
              'init_px' :   init_denormalized_coordinates[:,1],
              'init_y' :    init_denormalized_coordinates[:,2],
              'init_py' :   init_denormalized_coordinates[:,3],
              'init_zeta' : init_denormalized_coordinates[:,4],
              'init_delta' :init_denormalized_coordinates[:,5],
              'init_delta_wrt_CO' : pp.init_delta_wrt_CO
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
