import pickle
import pysixtrack
import numpy as np
import NAFFlib
import helpers as hp
import footprint
import matplotlib.pyplot as plt
from cpymad.madx import Madx

from scipy.constants import c

import os

# track_with = 'PySixtrack'
# track_with = 'Sixtrack'
track_with = 'Sixtracklib'
#device = 'opencl:1.0'
device = None

disable_BB = True
n_turns = 8000

with open('line.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

if disable_BB:
    line.disable_beambeam()

with open('particle_on_CO.pkl', 'rb') as fid:
    partCO = pickle.load(fid)

with open('DpxDpy_for_footprint.pkl', 'rb') as fid:
    temp_data = pickle.load(fid)

mad = Madx()
mad.options.echo = False
mad.options.warn = False
mad.options.info = False
mad.call('lhcwbb_fortracking.seq')
mad.use('lhcb1')
twi = mad.twiss()


beta0 = mad.sequence['lhcb1'].beam.beta
E = mad.sequence['lhcb1'].beam.energy*1.e9 #eV
alfa = twi.alfa[0]
V0 = 12.e6 #V
gamma0 = mad.sequence['lhcb1'].beam.gamma
q = max(twi.harmon)
freq = max(twi.freq)*1.e+6
psi0 = np.pi #above transition

synchrotron_theory_tune = np.sqrt(-V0*q*np.cos(psi0)/2./np.pi/beta0**2/E*(alfa-1./gamma0**2))

x_bucket = np.linspace(-2*np.pi, 2*np.pi,1000)
y_bucket1 = np.sqrt(-beta0*beta0*V0*E/np.pi/q/(alfa-1./(gamma0**2))*(np.cos(psi0+x_bucket) + np.cos(psi0)+ (2*psi0+x_bucket-np.pi)*np.sin(psi0))  )
y_bucket2 = -np.sqrt(-beta0*beta0*V0*E/np.pi/q/(alfa-1./(gamma0**2))*(np.cos(psi0+x_bucket) + np.cos(psi0)+ (2*psi0+x_bucket-np.pi)*np.sin(psi0))  )

y_bucket1 *= 1./beta0**2 / E
y_bucket2 *= 1./beta0**2 / E
x_bucket *= -beta0*c/2./np.pi/freq

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(x_bucket, y_bucket1, 'r')
ax1.plot(x_bucket, y_bucket2, 'r')
ax1.set_xlabel('sigma')
ax1.set_ylabel('delta')


xy_norm = temp_data['xy_norm']
DpxDpy_wrt_CO = temp_data['DpxDpy_wrt_CO']

#Dsigma_wrt_CO = [0.1,0.2,0.36,0.37,0,0.]
#Ddelta_wrt_CO = [0.,0.,0.,0.,0.00033,0.00031]
Dsigma_wrt_CO = 0.
Ddelta_wrt_CO = np.linspace(0.000000001, 0.00033, 20)

if track_with == 'PySixtrack':

    part = pysixtrack.Particles(**partCO)

    x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_pysixtrack(
        line, part=part, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=0.,
        Dy_wrt_CO_m=0, Dpy_wrt_CO_rad=0.,
        Dsigma_wrt_CO_m=Dsigma_wrt_CO, Ddelta_wrt_CO=Ddelta_wrt_CO, n_turns=n_turns, verbose=True)

    info = track_with

elif track_with == 'Sixtrack':
    os.chdir('sixtrack')
    x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_sixtrack(
        partCO=partCO, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=0.,
        Dy_wrt_CO_m=0, Dpy_wrt_CO_rad=0.,
        Dsigma_wrt_CO_m=Dsigma_wrt_CO, Ddelta_wrt_CO=Ddelta_wrt_CO, n_turns=n_turns)
    os.chdir('..')
    info = track_with

elif track_with == 'Sixtracklib':
    x_tbt, px_tbt, y_tbt, py_tbt, sigma_tbt, delta_tbt = hp.track_particle_sixtracklib(
        line=line, partCO=partCO, Dx_wrt_CO_m=0., Dpx_wrt_CO_rad=0.,
        Dy_wrt_CO_m=0., Dpy_wrt_CO_rad=0.,
        Dsigma_wrt_CO_m=Dsigma_wrt_CO, Ddelta_wrt_CO=Ddelta_wrt_CO, n_turns=n_turns, device=device)
    info = track_with
    if device is None:
    	info += ' (CPU)'
    else:
    	info += ' (GPU %s)'%device
else:
    raise ValueError('What?!')

n_part = x_tbt.shape[1]
for i_part in range(n_part):
    ax1.plot(sigma_tbt[:,i_part], delta_tbt[:, i_part])
Qsigma = np.zeros(n_part)
for i_part in range(n_part):
    Qsigma[i_part] = NAFFlib.get_tune(sigma_tbt[:, i_part])

#plt.close('all')
#
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(Ddelta_wrt_CO,Qsigma,'b',label='Tunes from tracking')
ax2.plot([0.,3e-4],[synchrotron_theory_tune, synchrotron_theory_tune],'k-',label='Theoretical tune = %e'%synchrotron_theory_tune)
ax2.set_ylabel('Qsigma')
ax2.set_xlabel('delta0')
ax2.legend()
#footprint.draw_footprint(xy_norm, axis_object=axcoord, linewidth = 1)
#axcoord.set_xlim(right=np.max(xy_norm[:, :, 0]))
#axcoord.set_ylim(top=np.max(xy_norm[:, :, 1]))
#
#fig4 = plt.figure(4)
#axFP = fig4.add_subplot(1, 1, 1)
#footprint.draw_footprint(Qxy_fp, axis_object=axFP, linewidth = 1)
## axFP.set_xlim(right=np.max(Qxy_fp[:, :, 0]))
## axFP.set_ylim(top=np.max(Qxy_fp[:, :, 1]))
#fig4.suptitle(info)
plt.show(False)
