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

# Symplectify R
# use dragt_simplectify
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
    print(np.linalg.det(M_new))
    return M_new

def furman_symplectify(M):
    J = np.array([[0., 0., 0., 1., 0., 0.],
                 [ 0., 0., 0., 0., 1., 0.],
                 [ 0., 0., 0., 0., 0., 1.],
                 [-1., 0., 0., 0., 0., 0.],
                 [ 0.,-1., 0., 0., 0., 0.],
                 [ 0., 0.,-1., 0., 0., 0.]])

    C = 0.5 * (3 + np.matmul(np.matmul(M , J), np.matmul(M.T, J)))
    return np.matmul(C, M)

def healy_symplectify(M):
    J = np.array([[0., 1., 0., 0., 0., 0.],
                  [-1., 0., 0., 0., 0., 0.],
                  [ 0., 0., 0., 1., 0., 0.],
                  [ 0., 0.,-1., 0., 0., 0.],
                  [ 0., 0., 0., 0., 0., 1.],
                  [ 0., 0., 0., 0.,-1., 0.]])

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


# Implement Normalization of fully coupled motion
def normalization_fully_coupled(R):
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
    
    R = np.array([[ np.cos(mu[0]), np.sin(mu[0]), 0, 0, 0, 0],
                  [-np.sin(mu[0]), np.cos(mu[0]), 0, 0, 0, 0],
    
                  [ 0, 0, np.cos(mu[1]), np.sin(mu[1]), 0, 0],
                  [ 0, 0,-np.sin(mu[1]), np.cos(mu[1]), 0, 0],
    
                  [ 0, 0, 0, 0, np.cos(mu[2]), np.sin(mu[2])],
                  [ 0, 0, 0, 0,-np.sin(mu[2]), np.cos(mu[2])]])
    
    W = np.array([a[0,:], b[0,:], 
                  a[1,:], b[1,:], 
                  a[2,:], b[2,:]]).T

    return W, R

#convert tau, pt to sigma,delta
sigma_CO = beta0 * t_CO
delta_CO = ((pt_CO**2 + 2*pt_CO/beta0) + 1)**0.5 - 1

mad_CO = [x_CO, px_CO, y_CO, py_CO, sigma_CO, delta_CO]

#Put closed orbit
part_on_CO = line.find_closed_orbit(
        guess=mad_CO, method='get_guess', p0c=p0c_eV)

################

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

