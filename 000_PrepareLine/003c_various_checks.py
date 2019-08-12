import matplotlib.pyplot as plt
import pysixtrack
import pickle
import numpy as np

from 003c_pickle_line_and_closed_orbit_for_pysixtrack import *

###############################################################################
######################## HIGHER ORDER FINITE DIFFERENCE #######################
###############################################################################

# function definitions

# functions denoted by _k are used to calculate higher order approximation
def get_init_particles_k(part, d):
    new_part = pysixtrack.Particles()
    new_part.p0c = part.p0c
    new_part.x = part.x
    new_part.px = part.px
    new_part.y = part.y
    new_part.py = part.py
    new_part.zeta = part.zeta
    new_part.delta = part.delta

    new_part.x     += np.array([0., 1.*d, 0., 0., 0., 0., 0.,   -1.*d, 0., 0., 0., 0., 0. ])
    new_part.px    += np.array([0., 0., 1.*d, 0., 0., 0., 0.,    0., -1.*d, 0., 0., 0., 0 ])
    new_part.y     += np.array([0., 0., 0., 1.*d, 0., 0., 0.,    0., 0., -1.*d, 0., 0., 0.])
    new_part.py    += np.array([0., 0., 0., 0., 1.*d, 0., 0.,    0., 0., 0., -1.*d, 0., 0.])
    new_part.zeta  += np.array([0., 0., 0., 0., 0., 1.*d, 0.,    0., 0., 0., 0., -1.*d, 0.])
    new_part.delta += np.array([0., 0., 0., 0., 0., 0., 1.*d,    0., 0., 0., 0., 0., -1.*d])

    return new_part

def get_R_m_k(part, line, d):
    init_part = get_init_particles_k(part, d)
    fin_part = get_init_particles_k(part, d)
    line.track(fin_part)

    X_init = np.empty([6,13])
    X_fin = np.empty([6,13])
    X_init[0,:] = init_part.x
    X_init[1,:] = init_part.px
    X_init[2,:] = init_part.y
    X_init[3,:] = init_part.py
    X_init[4,:] = init_part.zeta
    X_init[5,:] = init_part.delta

    X_fin[0,:] = fin_part.x
    X_fin[1,:] = fin_part.px
    X_fin[2,:] = fin_part.y
    X_fin[3,:] = fin_part.py
    X_fin[4,:] = fin_part.zeta
    X_fin[5,:] = fin_part.delta

    m = X_fin[:, 0] - X_init[:, 0]
    R = np.empty([6, 6])
    for j in range(6):
        R[:,j] = (X_fin[:,j + 1] - X_fin[:,j + 7]) / (2*d)

    X_CO = X_init[:,0] + np.matmul(np.linalg.inv(np.identity(6) - R), m.T)

    part_CO = pysixtrack.Particles(p0c = part.p0c)

    part_CO.x = X_CO[0]
    part_CO.px = X_CO[1]
    part_CO.y = X_CO[2]
    part_CO.py = X_CO[3]
    part_CO.zeta = X_CO[4]
    part_CO.delta = X_CO[5]

    return R, m, part_CO

def get_part_CO_k(part, line, d, ii):
    for i in range(ii):
        R_k, m_k, part_k = get_R_m_k(part, line, d)
        i += 1
        print ("turn %s of %s"%(i,ii))
    return R_k, part_k


# let's start the fun

dd =  [1e-5, 1e-6,  1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

iii = [10,   10,    10,   10,   10,   8,     7,     2,     2,     2,     2]


RR_k = np.zeros([11, 6, 6])
RR = np.zeros([11, 6, 6])

for beep in range(11):
    print('R_k for d = %s' %(dd[beep]))
    RR_k[beep, :, :] = get_part_CO_k(part, line, dd[beep], iii[beep])[0]


for beep in range(11):
    print('R for d = %s' %(dd[beep]))
    RR[beep, :, :] = get_part_CO(part, line, dd[beep], iii[beep])[0]


##### non-symplectified #####
import matplotlib.pyplot as plt
plt.style.use("kostas")

plt.plot(-np.log10(dd), abs(1 - np.linalg.det(RR_k)), '-or', label = '2nd order')
plt.plot(-np.log10(dd), abs(1 - np.linalg.det(RR)),   '-og', label = '1st order')
plt.xlabel(r'$-\log{d}$')
plt.ylabel(r'$|det(R) - 1|$')
plt.legend(loc='upper left')
plt.title('Non-symplectified')

plt.show()


dragt_RR_k = np.zeros([len(dd), 6, 6])
dragt_RR = np.zeros([len(dd), 6, 6])
healy_RR_k = np.zeros([len(dd), 6, 6])
healy_RR = np.zeros([len(dd), 6, 6])



for boop in range(11):
    dragt_RR_k[boop, :, :] = dragt_symplectify(RR_k[boop, :, :])
    dragt_RR[boop, :, :]   = dragt_symplectify(RR[boop, :, :])
    healy_RR_k[boop, :, :] = healy_symplectify(RR_k[boop, :, :])
    healy_RR[boop, :, :]   = healy_symplectify(RR[boop, :, :])


##### Dragt symplectified #####



diff_dragt_k = np.sum(np.sum(
               np.divide(abs(dragt_RR_k - RR_k), RR_k),
               axis = 2), axis =1)         

diff_dragt   = np.sum(np.sum(
               np.divide(abs(dragt_RR - RR), RR),
               axis = 2), axis =1)


plt.plot(-np.log10(dd), diff_dragt_k, '-or', label = '2nd order')
plt.plot(-np.log10(dd), diff_dragt, '-og', label = '1st order')
plt.xlabel(r'$-\log{d}$')
plt.ylabel(r'$\sum_{i, j} |R_{ij} - R^s_{ij}|/R_{ij}$')
plt.legend(loc='upper left')
plt.title('Dragt symplectification')


plt.show()

##### Healy symplectified #####

diff_healy_k = np.sum(np.sum(
               np.divide(abs(healy_RR_k - RR_k), RR_k),
               axis = 2), axis =1)

diff_healy   = np.sum(np.sum(               
               np.divide(abs(healy_RR - RR), RR),
               axis = 2), axis =1)

plt.plot(-np.log10(dd), diff_healy_k, '-or', label = '2nd order')
plt.plot(-np.log10(dd), diff_healy, '-og', label = '1st order')
plt.xlabel('-log(d)')
plt.legend(loc='upper left')
plt.ylabel(r'$\sum_{i, j} |R_{ij} - R^s_{ij}|/R_{ij}$')
plt.title('Healy symplectification')


plt.show()

###############################################################################
################################  RANDOM CHECKS  ##############################
###############################################################################

re = np.zeros(6, 6)

re[0,:] = [twiss_table.re11[0], twiss_table.re12[0], 
           twiss_table.re13[0], twiss_table.re14[0],
           twiss_table.re15[0], twiss_table.re16[0]]

re[1,:] = [twiss_table.re21[0], twiss_table.re22[0],
           twiss_table.re23[0], twiss_table.re24[0],
           twiss_table.re25[0], twiss_table.re26[0]]                                                 

re[2,:] = [twiss_table.re31[0], twiss_table.re32[0],
           twiss_table.re33[0], twiss_table.re34[0], 
           twiss_table.re35[0], twiss_table.re36[0]]

re[3,:] = [twiss_table.re41[0], twiss_table.re42[0],
           twiss_table.re43[0], twiss_table.re44[0],
           twiss_table.re45[0], twiss_table.re46[0]]

re[4,:] = [twiss_table.re51[0], twiss_table.re52[0],
           twiss_table.re53[0], twiss_table.re54[0],
           twiss_table.re55[0], twiss_table.re56[0]]

re[5,:] = [twiss_table.re61[0], twiss_table.re62[0],
           twiss_table.re63[0], twiss_table.re64[0],
           twiss_table.re65[0], twiss_table.re66[0]]








