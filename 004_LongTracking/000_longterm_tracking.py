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
optics_dict['epsn_x'] = 3.5e-6
optics_dict['epsn_y'] = 3.5e-6

init_normalized_coordinates_sigma = random_hypersphere.random_hypersphere_reject(pp.n_sigmas, pp.n_particles, dim=4, seed=pp.seed)
init_normalized_coordinates = np.empty_like(init_normalized_coordinates_sigma)
init_normalized_coordinates[:,0] = init_normalized_coordinates_sigma[:,0] * np.sqrt( optics_dict['epsn_x']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['betx'])
init_normalized_coordinates[:,1] = init_normalized_coordinates_sigma[:,1] * np.sqrt( optics_dict['epsn_x']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['betx'])
init_normalized_coordinates[:,2] = init_normalized_coordinates_sigma[:,2] * np.sqrt( optics_dict['epsn_y']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['bety'])
init_normalized_coordinates[:,3] = init_normalized_coordinates_sigma[:,3] * np.sqrt( optics_dict['epsn_y']/optics_dict['beta0']/optics_dict['gamma0']/optics_dict['bety'])
init_x, init_px, init_y, init_py = normalization.denormalize_4d_uncoupled(init_normalized_coordinates, optics_dict)


with open(pp.CO_filename, 'rb') as fid:
    partCO = pickle.load(fid)

init_delta = pp.init_delta_wrt_CO + partCO['delta']

#remove dispersion
init_x += optics_dict['disp_x']*init_delta
init_px += optics_dict['disp_px']*init_delta
init_y += optics_dict['disp_y']*init_delta
init_py += optics_dict['disp_py']*init_delta

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
        line=line, partCO=partCO, Dx_wrt_CO_m=init_x, Dpx_wrt_CO_rad=init_px,
        Dy_wrt_CO_m=init_y, Dpy_wrt_CO_rad=init_py,
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=pp.init_delta_wrt_CO, n_turns=pp.n_turns, device=pp.device)
    info = track_with
    if pp.device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%pp.device
else:
    raise ValueError('What?!')

input_data = {'init_x' : init_x,
              'init_px' : init_px,
              'init_y' : init_y,
              'init_py' : init_py,
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
