from cpymad.madx import Madx
import pysixtrack
import pickle
import numpy as np

########################################################
#                  Search closed orbit                 #
########################################################
seq = 'lhcb1'

mad = Madx()
mad.options.echo = False
mad.options.warn = False
mad.options.info = False

mad.call('lhcwbb_fortracking.seq')
mad.use(seq)

beta0 = mad.sequence[seq].beam.beta
gamma0 = mad.sequence[seq].beam.gamma
p0c_eV = mad.sequence[seq].beam.pc*1.e9

twiss_table = mad.twiss()

# Load line
with open('line_from_mad.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

# Disable BB elements
line.disable_beambeam()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

######################################################
################### Initialization ###################
######################################################

part = pysixtrack.Particles(p0c = p0c_eV)

# get an initial particle to use for finding the closed orbit
def get_init_particles(part, d):
    new_part = pysixtrack.Particles()
    new_part.p0c = part.p0c
    new_part.x = part.x
    new_part.px = part.px
    new_part.y = part.y
    new_part.py = part.py
    new_part.zeta = part.zeta
    new_part.delta = part.delta
    
    new_part.x     += np.array([0., 1.*d, 0., 0., 0., 0., 0.])
    new_part.px    += np.array([0., 0., 1.*d, 0., 0., 0., 0.])
    new_part.y     += np.array([0., 0., 0., 1.*d, 0., 0., 0.])
    new_part.py    += np.array([0., 0., 0., 0., 1.*d, 0., 0.])
    new_part.zeta  += np.array([0., 0., 0., 0., 0., 1.*d, 0.])
    new_part.delta += np.array([0., 0., 0., 0., 0., 0., 1.*d])

    return new_part

# Get the matrix M and vector m after 1 turn
def get_M_m(part, line, d):
    init_part = get_init_particles(part, d)
    fin_part = get_init_particles(part, d)
    line.track(fin_part)
    
    X_init = np.empty([6,7])
    X_fin = np.empty([6,7])
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
    M = np.empty([6, 6])
    for j in range(6):
        M[:,j] = (X_fin[:,j+1] - X_fin[:,0]) / d
    
    X_CO = X_init[:,0] + np.matmul(np.linalg.inv(np.identity(6) - M), m.T)
    
    part_CO = pysixtrack.Particles(p0c = part.p0c)
    
    part_CO.x = X_CO[0]
    part_CO.px = X_CO[1]
    part_CO.y = X_CO[2]
    part_CO.py = X_CO[3]
    part_CO.zeta = X_CO[4]
    part_CO.delta = X_CO[5]

    return M, m, part_CO


######################################################
### Get transfer matrix M and closed orbit part_CO ###
######################################################

print(' ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~')
print(' ')
print('Tracking particle over multiple turns to get the closed orbit...')

#iterate over ii turns to get a stable approximation
def get_part_CO(part, line, d, ii):
    for i in range(ii):
        M, m, part = get_M_m(part, line, d)
        i += 1
        print ("turn %s of %s"%(i,ii))
    return M, part

M, part_CO = get_part_CO(part, line, 1e-10, 10)

######################################################
##### Symplectify the almost-symplectic matrix M #####
######################################################

J = np.array([[0., 1., 0., 0., 0., 0.],
              [-1., 0., 0., 0., 0., 0.],
              [ 0., 0., 0., 1., 0., 0.],
              [ 0., 0.,-1., 0., 0., 0.],
              [ 0., 0., 0., 0., 0., 1.],
              [ 0., 0., 0., 0.,-1., 0.]])

def dragt_symplectify(M): 
    import scipy
    N = np.matmul(np.matmul(M , J), np.matmul(M.T, J.T))
    Q = scipy.linalg.expm(0.5 * scipy.linalg.logm(N))
    Ms = np.matmul(np.linalg.inv(Q), M)
    return Ms

def healy_symplectify(M):
    I = np.identity(6)

    V = np.matmul(J, np.matmul(I - M, np.linalg.inv(I + M)))
    W = (V + V.T)/2
    if np.linalg.det(I - np.matmul(J, W)) != 0:
        M_new = np.matmul(I + np.matmul(J, W), np.linalg.inv(I - np.matmul(J, W)))
    else:
        V_else = np.matmul(J, np.matmul(I + M, np.linalg.inv(I - M)))
        W_else = (V_else + V_else.T)/2
        M_new = -np.matmul(I + np.matmul(J, W_else), np.linalg(I - np.matmul(J, W_else)))
    return M_new

print(' ')
print('Symplectifying M...')

Ms = dragt_symplectify(M)
# Ms = healy_symplectify(M)

######################################################
### Implement Normalization of fully coupled motion ##
######################################################

def get_W_sigma(Ms):
    
    w0, v0 = np.linalg.eig(Ms)
    
    a0 = np.real(v0)
    b0 = np.imag(v0)
    
    # remove the vectors corresponding to eigenvalues that are
    # complex conjugates with negative Im(v)
    
    a = np.array([a0[:,0], a0[:,2], a0[:,4]])   
    b = np.array([b0[:,0], b0[:,2], b0[:,4]])
    v = v = a + b*1j
    w = np.array([w0[0], w0[2], w0[4]])
    mu = np.imag(np.log(w))
    
    # SIGMA
    sigma = np.zeros([6,6])
    for i in range(6):
        sigma[i,:] = v0[i, :] * np.conj(v0[i, :]).T
    
    # R
    R = np.array([[ np.cos(mu[0]), np.sin(mu[0]), 0, 0, 0, 0],
                  [-np.sin(mu[0]), np.cos(mu[0]), 0, 0, 0, 0],
    
                  [ 0, 0, np.cos(mu[1]), np.sin(mu[1]), 0, 0],
                  [ 0, 0,-np.sin(mu[1]), np.cos(mu[1]), 0, 0],
    
                  [ 0, 0, 0, 0, np.cos(mu[2]), np.sin(mu[2])],
                  [ 0, 0, 0, 0,-np.sin(mu[2]), np.cos(mu[2])]])
    
    # W
    n = np.zeros([3,]) 
    aa = np.zeros_like(a)
    bb = np.zeros_like(b)
    for i in range(a.shape[0]): 
        n[i] = np.sqrt(abs(1 / (np.matmul(a[i,:].T, J) @ b[i,:]))) 
        aa[i] = -n[i] * a[i, :]
        bb[i] = -n[i] * b[i, :]
    W = np.array([aa[0,:], bb[0,:],
                  aa[1,:], bb[1,:], 
                  aa[2,:], bb[2,:]]).T

    return W, R, sigma


def normalize(X, n_turns, n_part, Ms):
    print('Getting W...')
    W = get_W_sigma(Ms)[0]
    inv_W = np.linalg.inv(W)
    X_norm = np.zeros([n_part, n_turns, 6])
    print('Now normalizing X...')
    for n in range(n_turns):
        for i in range(n_part):
            X_norm[i, n, :] = np.matmul(inv_W, X[i, n, :] - np.array([part_CO.x, 
                                                                      part_CO.px,
                                                                      part_CO.y,
                                                                      part_CO.py,
                                                                      part_CO.zeta,
                                                                      part_CO.delta]))
        if n % 100 == 0:
            print('%s out of %s turns' %(n, n_turns))
    return X_norm

print(' ')
print('Now getting the W matrix for normalization...')

W, R, sigma = get_W_sigma(Ms)
inv_W = np.linalg.inv(W)


######################################################
###### Track n. particles  for n. turns to check #####
######################################################


def track_multi_particles(n_turns, n_part, part_CO, Ms):
    X = np.zeros([n_part, n_turns, 6])
    for i in range(n_part):
        X[i, 0, 0] = part_CO.x + i
    X[:, 0, 1] = part_CO.px
    X[:, 0, 2] = part_CO.y
    X[:, 0, 3] = part_CO.py
    X[:, 0, 4] = part_CO.zeta
    X[:, 0, 5] = part_CO.delta
    print("Tracking the %s particles for %s turns"%(n_part, n_turns))
    for n in range(n_turns - 1):
        for i in range(n_part):
            X[i, n + 1,:] = np.matmul(Ms, X[i, n,:])
        if n % 100 == 0:
            print("turn %s of %s" % (n, n_turns))
    
    return X

def return_plot(X, n_turns, n_part, Ms):
    
    import matplotlib.pyplot as plt
    plt.style.use("kostas")
    
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(X[0, :, 0], X[0, :, 1], '.b')
    axs[0, 0].plot(X[1, :, 0], X[1, :, 1], '.c')
    axs[0, 0].plot(X[2, :, 0], X[2, :, 1], '.m')
    axs[0, 0].plot(X[3, :, 0], X[3, :, 1], '.r')
    axs[0, 0].plot(X[4, :, 0], X[4, :, 1], '.y')
   # plt.xlabel('$x$', axes = axs[0,0])
   # plt.ylabel(r'$p_x$', axes = axs[0,0])
    axs[0, 0].set_xlim([-10.3, 10.3])
    axs[0, 0].set_ylim([-0.230, 0.230])
    
    axs[0, 0].set_title(r'$x, p_x$')

    axs[1, 0].plot(X[0, :, 2], X[0, :, 3], '.b')
    axs[1, 0].plot(X[1, :, 2], X[1, :, 3], '.c')
    axs[1, 0].plot(X[2, :, 2], X[2, :, 3], '.m')
    axs[1, 0].plot(X[3, :, 2], X[3, :, 3], '.r')
    axs[1, 0].plot(X[4, :, 2], X[4, :, 3], '.y')
   # plt.xlabel('$y$', axes=axs[1,0])
   # plt.ylabel(r'$p_y$', axes=axs[1,0])
    axs[1, 0].set_xlim([-10.3, 10.3])
    axs[1, 0].set_ylim([-0.230, 0.230])

    axs[1, 0].set_title(r'$y, p_y$')

    X_norm = normalize(X, 100000, 5, Ms)
    
    axs[0, 1].plot(X_norm[0, :, 0], X_norm[0, :, 1], '.b')
    axs[0, 1].plot(X_norm[1, :, 0], X_norm[1, :, 1], '.c')
    axs[0, 1].plot(X_norm[2, :, 0], X_norm[2, :, 1], '.m')
    axs[0, 1].plot(X_norm[3, :, 0], X_norm[3, :, 1], '.r')
    axs[0, 1].plot(X_norm[4, :, 0], X_norm[4, :, 1], '.y')
   # plt.xlabel('$x$ norm', axes = axs[0,1])
   # plt.ylabel(r'$p_x$ norm', axes = axs[0,1])
    axs[0, 1].set_xlim([-1.3, 1.3])
    axs[0, 1].set_ylim([-1.3, 1.3])

    axs[0, 1].set_title(r'$x, p_x$ norm')

    axs[1, 1].plot(X_norm[0, :, 2], X_norm[0, :, 3], '.b')
    axs[1, 1].plot(X_norm[1, :, 2], X_norm[1, :, 3], '.c')
    axs[1, 1].plot(X_norm[2, :, 2], X_norm[2, :, 3], '.m')
    axs[1, 1].plot(X_norm[3, :, 2], X_norm[3, :, 3], '.r')
    axs[1, 1].plot(X_norm[4, :, 2], X_norm[4, :, 3], '.y')
   # plt.xlabel('$y$ norm', axes = axs[1,1])
   # plt.ylabel(r'$p_y norm$', axes = axs[1,1])
    axs[1, 1].set_xlim([-1.3, 1.3])
    axs[1, 1].set_ylim([-1.3, 1.3])

    axs[1, 1].set_title(r'$y, p_y$ norm')

    plt.show()
    
    return

######################################################
####### Save some useful tools as dictionary #########
######################################################

optics_dict = {'betx'      : twiss_table.betx[0],
               'bety'      : twiss_table.bety[0],
               'alfx'      : twiss_table.alfx[0],
               'alfy'      : twiss_table.alfy[0],
               'qx'        : mad.table['summ']['q1'][0],
               'qy'        : mad.table['summ']['q2'][0],
               'dqx'       : mad.table['summ']['dq1'][0],
               'dqy'       : mad.table['summ']['dq2'][0],
               'rf_volt'   : sum(twiss_table.volt),
               'rf_freq'   : max(twiss_table.freq),
               'rf_harmon' : max(twiss_table.harmon),
               'rf_lag'    : twiss_table.lag[twiss_table.harmon != 0][0],
               'length'    : twiss_table.s[-1],
               'alfa'      : twiss_table.alfa[0], 
               'beta0'     : mad.sequence[seq].beam.beta,
               'gamma0'    : mad.sequence[seq].beam.gamma,
               'p0c_eV'    : mad.sequence[seq].beam.pc*1.e9,
               'M'         : M,                             # Transfer matrix M
               'Ms'        : Ms,                            # Symplectified M
               'R'         : R,                             # Rotation matrix
               'W'         : W,
               'inv_W'     : inv_W,
               'part_CO'   : np.array([part_CO.x,
                                       part_CO.px,
                                       part_CO.y,
                                       part_CO.py,
                                       part_CO.zeta,
                                       part_CO.delta])
              }

with open('optics_mad.pkl', 'wb') as fid:
    pickle.dump(optics_dict , fid)

print('~       ~       ~       ~       ~       ~')

print(' ')

print(optics_dict)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

'''

#convert tau, pt to sigma,delta
sigma_CO = beta0 * t_CO
delta_CO = ((pt_CO**2 + 2*pt_CO/beta0) + 1)**0.5 - 1

mad_CO = [x_CO, px_CO, y_CO, py_CO, sigma_CO, delta_CO]

#Put closed orbit
part_on_CO = line.find_closed_orbit(
        guess=mad_CO, method='get_guess', p0c=p0c_eV)


print('Closed orbit at start machine:')
print('x px y py sigma delta:')
print(part_on_CO)

#######################################################
#  Store closed orbit and dipole kicks at BB elements #
#######################################################

line.beambeam_store_closed_orbit_and_dipolar_kicks(
        part_on_CO,
        separation_given_wrt_closed_orbit_4D = True,
        separation_given_wrt_closed_orbit_6D = True)

#################################
# Save machine in pyblep format #
#################################

with open('line_from_mad_with_bbCO.pkl', 'wb') as fid:
    pickle.dump(line.to_dict(keepextra=True), fid)
  
#########################################
# Save particle on closed orbit as dict #
#########################################

with open('particle_on_CO_mad_line.pkl', 'wb') as fid:
    pickle.dump(part_on_CO.to_dict(), fid)

'''
