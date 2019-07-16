import pickle
import pysixtrack
import numpy as np
import NAFFlib

import sys
sys.path.append('..')

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

n_turns = 1000000

with open('line_from_mad_with_bbCO.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

line.remove_inactive_multipoles(inplace=True)
line.remove_zero_length_drifts(inplace=True)
line.merge_consecutive_drifts(inplace=True)

if disable_BB:
    line.disable_beambeam()

with open('particle_on_CO_mad_line.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

with open('DpxDpy_for_footprint_for_mad_line.pkl', 'rb') as fid:
    temp_data = pickle.load(fid)

xy_norm = temp_data['xy_norm']
DpxDpy_wrt_CO = temp_data['DpxDpy_wrt_CO']


def track_particle_sixtracklib_state(
                            line, partCO, Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                            Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                            Dsigma_wrt_CO_m, Ddelta_wrt_CO, n_turns,
                            device=None):

    Dx_wrt_CO_m, Dpx_wrt_CO_rad,\
        Dy_wrt_CO_m, Dpy_wrt_CO_rad,\
        Dsigma_wrt_CO_m, Ddelta_wrt_CO = hp.vectorize_all_coords(
                             Dx_wrt_CO_m, Dpx_wrt_CO_rad,
                             Dy_wrt_CO_m, Dpy_wrt_CO_rad,
                             Dsigma_wrt_CO_m, Ddelta_wrt_CO)

    part = pysixtrack.Particles(**partCO)

    n_turns_to_store=10

    import sixtracklib
    elements=sixtracklib.Elements()
    #elements.BeamMonitor(num_stores=n_turns)
    elements.BeamMonitor(num_stores=n_turns_to_store)
    elements.append_line(line)

    n_part = len(Dx_wrt_CO_m)

    # Build PyST particle

    ps = sixtracklib.ParticlesSet()
    p = ps.Particles(num_particles=n_part)

    for i_part in range(n_part):

        part = pysixtrack.Particles(**partCO)
        part.x += Dx_wrt_CO_m[i_part]
        part.px += Dpx_wrt_CO_rad[i_part]
        part.y += Dy_wrt_CO_m[i_part]
        part.py += Dpy_wrt_CO_rad[i_part]
        part.sigma += Dsigma_wrt_CO_m[i_part]
        part.delta += Ddelta_wrt_CO[i_part]

        part.partid = i_part
        part.state = 1
        part.elemid = 0
        part.turn = 0

        p.from_pysixtrack(part, i_part)

    if device is None:
        job = sixtracklib.TrackJob(elements, ps)
    else:
        job = sixtracklib.TrackJob(elements, ps, device=device)

    job.track(n_turns)
    job.collect()

    res = job.output

#    print(res.particles[0])
#    print(res.particles[0].at_turn.shape)
#    print(res.particles[0].state.reshape(n_turns,n_part))
#    print(n_turns*n_part)
    
    state_tbt = res.particles[0].state.reshape(n_turns_to_store, n_part)[-1,:]

    print('Done loading!')

    return state_tbt

tt = time.time()
if track_with == 'Sixtracklib':
    state_tbt = track_particle_sixtracklib_state(
        line=line, partCO=partCO, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 0].flatten(),
        Dy_wrt_CO_m=0., Dpy_wrt_CO_rad=DpxDpy_wrt_CO[:, :, 1].flatten(),
        Dsigma_wrt_CO_m=0., Ddelta_wrt_CO=0., n_turns=n_turns, device=device)
    info = track_with
    if device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%device
else:
    raise ValueError('What?!')

tracking_time = time.time() - tt

#n_part = x_tbt.shape[1]

plt.close('all')

fig3 = plt.figure(3)
ax1 = fig3.add_subplot(111)
ax1.plot(xy_norm[:,:,0], xy_norm[:,:,1],'o',c=plt.cm.jet(state_tbt))

data = {}
data['x'] = xy_norm[:,:,0]
data['y'] = xy_norm[:,:,1]
data['state'] = state_tbt[:]

with open('DA_data_turns%d.pkl'%n_turns, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#fig3.savefig('fig3.png')
#fig4.savefig('fig4.png')
print('Tracking time: %f minutes'%(tracking_time/60.))
plt.show()
