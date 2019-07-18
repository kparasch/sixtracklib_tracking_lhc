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

# track_with = 'PySixtrack'
# track_with = 'Sixtrack'
track_with = 'Sixtracklib'
device = 'opencl:0.3'
#device = None

disable_BB = False
init_delta_wrt_CO=2.7e-4
n_turns = 1000000

out_fname = 'dynap_sixtracklib_nturns_%d_delta_%.2e.h5'%(n_turns,init_delta_wrt_CO)

with open('line_from_mad_with_bbCO.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))
line.remove_inactive_multipoles(inplace=True)
line.remove_zero_length_drifts(inplace=True)
line.merge_consecutive_drifts(inplace=True)

if disable_BB:
    line.disable_beambeam()

with open('particle_on_CO_mad_line.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

with open('DpxDpy_for_DA.pkl', 'rb') as fid:
    input_data = pickle.load(fid)

xy_norm = input_data['xy_norm']
DpxDpy_wrt_CO = input_data['DpxDpy_wrt_CO']

if track_with == 'Sixtracklib':
    output_dict = hp.track_particle_sixtracklib_firstlast(
        line=line, partCO=partCO, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 0].flatten(),
        Dy_wrt_CO_m=0., Dpy_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 1].flatten(),
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=init_delta_wrt_CO, n_turns=n_turns, device=device)
    info = track_with
    if device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%device
else:
    raise ValueError('What?!')

save_dict = {**output_dict, **input_data}
mfm.dict_to_h5(save_dict, out_fname)
print('Tracking time: %f mins'%save_dict['tracking_time_mins'])
