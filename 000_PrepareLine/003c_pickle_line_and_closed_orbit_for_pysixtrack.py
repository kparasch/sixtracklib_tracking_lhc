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
x_CO  = twiss_table.x[0]
px_CO = twiss_table.px[0]
y_CO  = twiss_table.y[0]
py_CO = twiss_table.py[0]
t_CO  = twiss_table.t[0]
pt_CO = twiss_table.pt[0]

# Load line
with open('line_from_mad.pkl', 'rb') as fid:
    line = pysixtrack.Line.from_dict(pickle.load(fid))

# Disable BB elements
line.disable_beambeam()


twiss_table = mad.twiss()

##############

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

# Get the matrix R and vector m after 1 turn
def get_R_m(part, line, d):
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
    R = np.empty([6, 6])
    for j in range(6):
        R[:,j] = (X_fin[:,j+1] - X_fin[:,0]) / d
    
    X_CO = X_init[:,0] + np.matmul(np.linalg.inv(np.identity(6) - R), m.T)
    
    part_CO = pysixtrack.Particles(p0c = part.p0c)
    
    part_CO.x = X_CO[0]
    part_CO.px = X_CO[1]
    part_CO.y = X_CO[2]
    part_CO.py = X_CO[3]
    part_CO.zeta = X_CO[4]
    part_CO.delta = X_CO[5]

    return R, m, part_CO

#iterate over ii turns to get a stable approximation
def get_part_CO(part, line, d, ii):
    for i in range(ii):
        R, m, part = get_R_m(part, line, d)
        i += 1
        print ("turn %s of %s"%(i,ii))
    return R, part

# Symplectify R using Alex J. Dragt's algorithm
def dragt_symplectify(M): 
    import scipy
    J = np.array([[0., 1., 0., 0., 0., 0.],
                 [-1., 0., 0., 0., 0., 0.],
                 [ 0., 0., 0., 1., 0., 0.],
                 [ 0., 0.,-1., 0., 0., 0.],
                 [ 0., 0., 0., 0., 0., 1.],
                 [ 0., 0., 0., 0.,-1., 0.]])
    N = np.matmul(np.matmul(M , J), np.matmul(M.T, J.T))
    Q = scipy.linalg.expm(0.5 * scipy.linalg.logm(N))
    M_new = np.matmul(np.linalg.inv(Q), M)
#    print(np.linalg.det(M_new))
    return M_new

# re = np.zeros(6, 6)
# re[0,:] = [twiss_table.re11[0], twiss_table.re12[0], 
#            twiss_table.re13[0], twiss_table.re14[0],
#            twiss_table.re15[0], twiss_table.re16[0]]
# re[1,:] = [twiss_table.re21[0], twiss_table.re22[0],
#            twiss_table.re23[0], twiss_table.re24[0],
#            twiss_table.re25[0], twiss_table.re26[0]]                                                 
# re[2,:] = [twiss_table.re31[0], twiss_table.re32[0],
#            twiss_table.re33[0], twiss_table.re34[0], 
#            twiss_table.re35[0], twiss_table.re36[0]]
# 
# re[3,:] = [twiss_table.re41[0], twiss_table.re42[0],
#            twiss_table.re43[0], twiss_table.re44[0],
#            twiss_table.re45[0], twiss_table.re46[0]]
# 
# re[4,:] = [twiss_table.re51[0], twiss_table.re52[0],
#            twiss_table.re53[0], twiss_table.re54[0],
#            twiss_table.re55[0], twiss_table.re56[0]]
# 
# re[5,:] = [twiss_table.re61[0], twiss_table.re62[0],
#            twiss_table.re63[0], twiss_table.re64[0],
#            twiss_table.re65[0], twiss_table.re66[0]]


# Implement Normalization of fully coupled motion
def get_W(R):
    M = dragt_symplectify(R)
    
    w0, v0 = np.linalg.eig(M)
    
    a0 = np.real(v0)
    b0 = np.imag(v0)
    
    # remove the vectors corresponding to eigenvalues that are
    # complex conjugates with negative Im(v)
    
    a = np.array([a0[:,0], a0[:,2], a0[:,4]])   
    b = np.array([b0[:,0], b0[:,2], b0[:,4]])
    w = np.array([w0[0], w0[2], w0[4]])
    mu = np.imag(np.log(w))
    
    R_mu = np.array([[ np.cos(mu[0]), np.sin(mu[0]), 0, 0, 0, 0],
                  [-np.sin(mu[0]), np.cos(mu[0]), 0, 0, 0, 0],
    
                  [ 0, 0, np.cos(mu[1]), np.sin(mu[1]), 0, 0],
                  [ 0, 0,-np.sin(mu[1]), np.cos(mu[1]), 0, 0],
    
                  [ 0, 0, 0, 0, np.cos(mu[2]), np.sin(mu[2])],
                  [ 0, 0, 0, 0,-np.sin(mu[2]), np.cos(mu[2])]])
    
    W = np.array([a[0,:], b[0,:],
                  a[1,:], b[1,:], 
                  a[2,:], b[2,:]]).T

    return W, R_mu


def normalize(X, turns, nr_part, R):
    W, idc = get_W(R)
    X_norm = np.zeros([nr_part, turns, 6])
    for n in range(turns):
        for i in range(nr_part):
            X_norm[i, n, :] = np.matmul(np.linalg.inv(W), X[i, n, :])
    return X_norm

def track_multi_particles(turns, nr_part, part_CO, R):
    X = np.zeros([nr_part, turns, 6])
    for i in range(nr_part):
        X[i, 0, 0] = part_CO.x + i
    X[:, 0, 1] = part_CO.px
    X[:, 0, 2] = part_CO.y
    X[:, 0, 3] = part_CO.py
    X[:, 0, 4] = part_CO.zeta
    X[:, 0, 5] = part_CO.delta
    print("Tracking the %s particles for %s turns"%(nr_part, turns))
    for n in range(turns - 1):
        for i in range(nr_part):
            X[i, n + 1,:] = np.matmul(R, X[i, n,:])
        if n % 100 == 0:
            print("turn %s of %s" % (n, turns))
    
    return X



def return_plot(X, turns, nr_part, R):
    
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

    X_norm = normalize(X, 100000, 5, R)
    
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



################

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

#########################################
# Save some optics as dict              #
#########################################

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
               'p0c_eV'    : mad.sequence[seq].beam.pc*1.e9
              }

with open('optics_mad.pkl', 'wb') as fid:
    pickle.dump(optics_dict , fid)

