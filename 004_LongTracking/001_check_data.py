import pickle
import pysixtrack
import numpy as np


import sys
sys.path.append('..')
import myfilemanager_sixtracklib as mfm
import helpers as hp
import random_hypersphere
import normalization
import matplotlib.pyplot as plt
sys.path.append('000_PrepareLine')
from a003c_pickle_line_and_closed_orbit_for_pysixtrack import *


cnaf0_input = mfm.h5_to_dict('data/losses_sixtracklib.12.0.h5', group = 'input')
# cnaf1_input = mfm.h5_to_dict('data/losses_sixtracklib.12.1.h5', group = 'input')
# cnaf2_input = mfm.h5_to_dict('data/losses_sixtracklib.12.2.h5', group = 'input')
# cnaf3_input = mfm.h5_to_dict('data/losses_sixtracklib.12.3.h5', group = 'input')

cnaf0_output = mfm.h5_to_dict('data/losses_sixtracklib.12.0.h5', group = 'output')
# cnaf1_output = mfm.h5_to_dict('data/losses_sixtracklib.12.1.h5', group = 'output')
# cnaf2_output = mfm.h5_to_dict('data/losses_sixtracklib.12.2.h5', group = 'output')
# cnaf3_output = mfm.h5_to_dict('data/losses_sixtracklib.12.3.h5', group = 'output')



# condor0_input = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'input')
# condor1_input = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'input')
# condor2_input = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'input')
# condor3_input = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'input')
# condor4_input = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'input')
# condor5_input = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'input')
# condor6_input = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'input')
 
# condor0_output = mfm.h5_to_dict('data/losses_sixtracklib.3672504.0.h5', group = 'output') 
# condor1_output = mfm.h5_to_dict('data/losses_sixtracklib.3672504.1.h5', group = 'output') 
# condor2_output = mfm.h5_to_dict('data/losses_sixtracklib.3672504.2.h5', group = 'output') 
# condor3_output = mfm.h5_to_dict('data/losses_sixtracklib.3672504.3.h5', group = 'output') 
# condor4_output = mfm.h5_to_dict('data/losses_sixtracklib.3672504.4.h5', group = 'output') 
# condor5_output = mfm.h5_to_dict('data/losses_sixtracklib.3672504.5.h5', group = 'output') 
# condor6_output = mfm.h5_to_dict('data/losses_sixtracklib.3672504.6.h5', group = 'output') 



with open('../000_PrepareLine/toolbox_pysixtrack.pkl', 'rb') as fid:
    toolbox = pickle.load(fid)

with open('../000_PrepareLine/bb_toolbox_pysixtrack.pkl', 'rb') as fid:
    bb_toolbox = pickle.load(fid)


n_part = 10000
n_turns = 1000


X0 = np.array([cnaf0_input['init_x'], cnaf0_input['init_px'], 
               cnaf0_input['init_y'], cnaf0_input['init_py'],
               np.zeros([10000,]),    np.zeros([10000,])   ])


X = np.array([cnaf0_output['x_tbt_first'],    cnaf0_output['x_tbt_first'],
              cnaf0_output['y_tbt_first'],    cnaf0_output['py_tbt_first'],
              cnaf0_output['zeta_tbt_first'], cnaf0_output['delta_tbt_first']]).T


# I'll just look at a couple of individual X vectors instead of the whole thing

X10   = X[9,   ::10, :] #particle nr. 10, every 10 turns
X100  = X[99,  ::10, :] #particle nr. 10, every 10 turns
X1000 = X[999, ::10, :] #particle nr. 10, every 10 turns

#normalize

bb_R = bb_toolbox['R']
def bb_normalize(X, n_turns, n_part, bb_R):
    print('Getting W...')
    W = bb_toolbox['W']
    X_norm = np.zeros([n_part, n_turns, 6])
    print('Now normalizing X...')
    for n in range(n_turns):
        for i in range(n_part):
            X_norm[i, n, :] = np.matmul(np.linalg.inv(W), X[i, n, :])
        print('%s out of %s turns' %(n, n_turns))
    return X_norm


# plot

def plot():
    plt.plot(X10[:,0],   X10[:, 1],   'bo', label = 'part nr. 10'  )
    plt.plot(X100[:,0],  X100[:, 1],  'ro', label = 'part nr. 100' )
    plt.plot(X1000[:,0], X1000[:, 1], 'mo', label = 'part nr. 1000')
    plt.title('x vs. px')
    plt.legend(loc = 'upper left')
    plt.show()
    return 

def bb_plot_norm():
    plt.plot(bb_X_norm[::100, ::10 ,0],   bb_X_norm[::100, ::10 , 1],   'bo'  )
    plt.title('with bb norm x vs. px')
    # plt.legend(loc = 'upper left')
    plt.show()
    return

def plot_norm():
    plt.plot(X_norm[::100, ::10 ,0],   X_norm[::100, ::10 , 1],   'bo'  )
    plt.title('without bb norm x vs. px')
    # plt.legend(loc = 'upper left')
    plt.show()
    return

