import numpy as np
import sys
sys.path.append('..')
import myfilemanager_sixtracklib as mfm
import matplotlib.pyplot as plt
import pickle


def dynamic_aperture_contour(h5_filename):
    
    ###############
    # input files #
    ###############
    
    #mydict = mfm.h5_to_dict('dynap_sixtracklib_turns_1000000_delta_0.00009.h5')
    mydict = mfm.h5_to_dict(h5_filename)
    optics = pickle.load( open('optics_mad.pkl', 'rb') )
    co = pickle.load( open('particle_on_CO_mad_line.pkl', 'rb') )
    
    nturns = 1e6
    delta = mydict['init_delta_wrt_CO']
    
    ########################
    # non-linear amplitude #
    ########################
    
    
    #initialization
    epsn_x = 3.5e-6
    epsn_y = 3.5e-6
    
    betx = optics['betx']
    bety = optics['bety']
    alfx = optics['alfx']
    alfy = optics['alfy']
    
    beta0 = optics['beta0']
    gamma0 = optics['gamma0']
    
    gammax = (1 + alfx**(2.)) / betx
    gammay = (1 + alfy**(2.)) / bety
    
    # coordinates w.r.t closed orbit co
    x_co = co['x']
    y_co = co['y']
    px_co = co['px']
    py_co = co['py']
    
    x = mydict['x_tbt_first'] - x_co
    y = mydict['y_tbt_first'] - y_co
    
    px = mydict['px_tbt_first'] - px_co
    py = mydict['py_tbt_first'] - py_co
    
    # calculation of amplitudes Ax [sigmax] & Ay [sigmay]
    
    Jx = 0.5 * (gammax * np.multiply(x, x) + 2 * alfx * np.multiply(x, px) + betx * np.multiply(px, px))
    Jy = 0.5 * (gammay * np.multiply(y, y) + 2 * alfy * np.multiply(y, py) + bety * np.multiply(py, py))
    
    Jx_avg = np.mean(Jx[:100], axis=0)
    Jy_avg = np.mean(Jy[:100], axis=0)
    
    sigmax = np.sqrt((betx*epsn_x)/(beta0*gamma0))
    sigmay = np.sqrt((bety*epsn_y)/(beta0*gamma0))
    
    #Ax = np.sqrt(2*betx*Jx_avg) / sigmax
    #Ay = np.sqrt(2*bety*Jy_avg) / sigmay
    
    Ax = np.sqrt(2*betx*Jx[0,:,:]) / sigmax
    Ay = np.sqrt(2*bety*Jy[0,:,:]) / sigmay
    
    
    #mydict['xy_norm'][:,:,0] = Ax[:,:]
    #mydict['xy_norm'][:,:,1] = Ay[:,:]
    
    ############
    # tracking #
    ############
    
    # if the particle's last turn < nturns
    # then it has been lost
    
    #i_max = mydict['xy_norm'].shape[0]
    #j_max = mydict['xy_norm'].shape[1]
    k_max = mydict['at_turn_tbt_last'].shape[0]
    i_max = mydict['at_turn_tbt_last'].shape[1]
    j_max = mydict['at_turn_tbt_last'].shape[2]
    
    
    ############# find the boundary of the dynap #############
    
    index_of_last_turn = np.argmax(mydict['at_turn_tbt_last'],axis=0)
    last_turn = np.max(mydict['at_turn_tbt_last'],axis=0)
    did_not_survive = last_turn < nturns - 1
    boundary = np.argmax(did_not_survive,axis=1)
        
    x_boundary = Ax[range(0,i_max), boundary]
    y_boundary = Ay[range(0,i_max), boundary]
    
    return x_boundary,y_boundary
    #i_boundary = np.array([])
    #j_boundary = np.array([])
    
    #for i in range(i_max):
    #    for j in range(j_max):
    #        if last_turn[i, j] < nturns - 1:
    #            i_boundary = np.append(i_boundary, i)
    #            j_boundary = np.append(j_boundary, j)
    #            break
    
    ############# find which particles are lost #############
    
    #i_lost = np.array([])
    #j_lost = np.array([])
    #
    #i_kept = np.array([])
    #j_kept = np.array([])
    #
    #for i in range(i_max):
    #    for j in range(j_max):
    #        if mydict['at_turn_tbt_last'][-1, i, j] < nturns - 1:
    #            i_lost = np.append(i_lost, i)
    #            j_lost = np.append(j_lost, j)
    #        else:
    #            i_kept = np.append(i_kept, i)
    #            j_kept = np.append(j_kept, j)
    #
    #i_boundary = i_boundary.astype(int)
    #j_boundary = j_boundary.astype(int)
    #
    #i_lost = i_lost.astype(int)
    #j_lost = j_lost.astype(int)
    #
    #i_kept = i_kept.astype(int)
    #j_kept = j_kept.astype(int)




#
#########
## plot #
#########
#
############## boundary x vs. y ##############
#
#f1 = plt.figure(1)
#
#
##x_boundary = mydict['xy_norm'][i_boundary, j_boundary, 0]
##y_boundary = mydict['xy_norm'][i_boundary, j_boundary, 1]
#
#plt.plot(x_boundary, y_boundary, '-r')
##label = 'dynamic aperture boundary after %d turns' % int(nturns))
#
## plt.plot(mydict['xy_norm'][:, j_max-1, 0], mydict['xy_norm'][:, j_max-1, 1], '-b', label = 'beamsize')
#
#plt.xlabel(r'$x [\sigma_x]$')
#plt.ylabel(r'$y [\sigma_y]$')
#
#plt.title('Dynamic Aperture Analysis, $\delta_{0} =$ %r' % delta)
#
#plt.legend(loc='upper right')
#
############## which particles are lost x vs. y #############
#
#f2 = plt.figure(2)
#x_lost = mydict['xy_norm'][i_lost, j_lost, 0]
#y_lost = mydict['xy_norm'][i_lost, j_lost, 1]
#
#x_kept = mydict['xy_norm'][i_kept, j_kept, 0]
#y_kept = mydict['xy_norm'][i_kept, j_kept, 1]
#
#
#plt.plot(Ax, Ay, 'o', color='blue', label = 'tracked particles')
#plt.plot(x_lost, y_lost, '.', color='red', label = 'lost particles')
#
#plt.xlabel(r'$x [\sigma_x]$')
#plt.ylabel(r'$y [\sigma_y]$')
#
#plt.title('Lost particles after %r  turns' % nturns)
#
#plt.legend(loc='upper right')
#
############## boundary r vs. theta #############
#
#f3 = plt.figure(3)
#
#r_boundary = np.sqrt( np.multiply(x_boundary, x_boundary) + np.multiply(y_boundary, y_boundary) )
#theta_boundary = np.arctan(np.divide(y_boundary, x_boundary))
#
#plt.plot(theta_boundary, r_boundary, '-r')
#
#plt.xlabel(r'$\theta [\sigma_{\theta}]$')
#plt.ylabel(r'$r [\sigma_r]$')
#
#plt.title(r'Dynamic Aperture Analysis, $\delta_{0} =$ %r' % delta)
#
#plt.legend(loc='upper right')
#
#
#
#plt.style.use('kostas')
#
#plt.show()
