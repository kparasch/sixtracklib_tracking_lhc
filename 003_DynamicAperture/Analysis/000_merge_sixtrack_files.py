import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
import myfilemanager_sixtracklib as mfm
import footprint
import analysis_dynamic_aperture as ada
plt.style.use('kostas')

def get_dynap(h5_filename, sigmax, sigmay, nturns, i):
    ob = mfm.h5_to_dict(h5_filename)

    #Ax = ob['horizontal_amp_mm']/sigmax
    #Ay = ob['vertical_amp_mm']/sigmay
    last_turn = ob['survived_turns']
    
    da_index = np.argmax(last_turn < nturns)
    
    r_max_sigma = 10.
    r_min_sigma = 0.1
    r_step = 0.1
    N_r = int(round((r_max_sigma-r_min_sigma)/r_step)) + 1
    theta_step = (3./180.)*np.pi
    theta_min_rad = (i)*theta_step
    theta_max_rad = (i+1)*theta_step
    N_theta = int(round((theta_max_rad-theta_min_rad)/theta_step))

    xy_norm = footprint.initial_xy_polar(r_min=r_min_sigma, r_max=r_max_sigma, r_N=N_r,
                                     theta_min=theta_min_rad, theta_max=theta_max_rad,
                                     theta_N=N_theta)


    #return Ax[da_index], Ay[da_index]
    r = da_index*r_step + r_min_sigma
    theta = theta_min_rad
    return r*np.cos(theta), r*np.sin(theta)

nturns = 1000000
NN = 31
epsn_x = 3.5e-6
epsn_y = 3.5e-6

optics = pickle.load( open('optics_mad.pkl', 'rb') )

sigmax = np.sqrt((optics['betx']*epsn_x)/(optics['beta0']*optics['gamma0']))*1e+3 # mm
sigmay = np.sqrt((optics['bety']*epsn_y)/(optics['beta0']*optics['gamma0']))*1e+3 # mm

x_dynap = np.zeros([NN])
y_dynap = np.zeros([NN])

for i in range(NN):
    #fname = 'sixtrack_data/dynap_sixtrack.3652935.%d.h5'%i
    fname = 'sixtrack_data/dynap_sixtrack.3655470.%d.h5'%i
    x_dynap[i], y_dynap[i] = get_dynap(fname, sigmax, sigmay, nturns, i)

x_dynap2 = np.zeros([NN])
y_dynap2 = np.zeros([NN])

for i in range(NN):
    #fname = 'sixtrack_data/dynap_sixtrack.3652935.%d.h5'%i
    fname = 'sixtrack_data/dynap_sixtrack.3657433.%d.h5'%i
    x_dynap2[i], y_dynap2[i] = get_dynap(fname, sigmax, sigmay, nturns, i)

x_da1_stl, y_da1_stl = ada.dynamic_aperture_contour('data/dynap_sixtracklib_turns_1000000_delta_0.00000.h5')
x_da2_stl, y_da2_stl = ada.dynamic_aperture_contour('data/dynap_sixtracklib_turns_1000000_delta_0.00027.h5')

fig = plt.figure(1,figsize=[9,9])
ax1 = fig.add_subplot(111)
fig.suptitle('Dynamic Aperture')
ax1.plot(x_dynap, y_dynap, 'r-', label='SixTrack')
ax1.plot(x_dynap2, y_dynap2, 'r-', alpha=0.5,label='SixTrack Off-Mom.')
ax1.plot(x_da1_stl, y_da1_stl, 'b-', label='sixtracklib')
ax1.plot(x_da2_stl, y_da2_stl, 'b-',alpha=0.5, label='sixtracklib Off-Mom.')
ax1.legend()
ax1.set_aspect('equal')
ax1.set_xlim(0,7.5)
ax1.set_ylim(0,7.5)
ax1.set_xlabel('$A_x\ [\sigma_x]$')
ax1.set_ylabel('$A_y\ [\sigma_y]$')
plt.show()
