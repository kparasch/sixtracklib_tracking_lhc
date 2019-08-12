import numpy as np
import pysixtrack
import scipy

S = np.array([[0., 1., 0., 0., 0., 0.],
              [-1., 0., 0., 0., 0., 0.],
              [ 0., 0., 0., 1., 0., 0.],
              [ 0., 0.,-1., 0., 0., 0.],
              [ 0., 0., 0., 0., 0., 1.],
              [ 0., 0., 0., 0.,-1., 0.]])

def normalize_4d_uncoupled(A, optics):
    assert(A.shape[1] == 4)
    
    betx = optics['betx']
    alfx = optics['alfx']
    bety = optics['bety']
    alfy = optics['alfy']

    sbetx = np.sqrt(betx)
    sbety = np.sqrt(bety)

    x = 1./sbetx * A[:,0]
    px = alfx/sbetx * A[:,0] + sbetx * A[:,1]
    y = 1./sbety * A[:,2]
    py = alfy/sbety * A[:,2] + sbety * A[:,3]

    return x, px, y, py

def denormalize_4d_uncoupled(A, optics):
    assert(A.shape[1] == 4)
    
    betx = optics['betx']
    alfx = optics['alfx']
    bety = optics['bety']
    alfy = optics['alfy']

    sbetx = np.sqrt(betx)
    sbety = np.sqrt(bety)

    x = sbetx * A[:,0]
    px = -alfx/sbetx * A[:,0] + 1./sbetx * A[:,1]
    y = sbety * A[:,2]
    py = -alfy/sbety * A[:,2] + 1./sbety * A[:,3]

    return x, px, y, py

def get_init_particles_for_linear_map(part, d):
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
def linearize_around_closed_orbit(part, line, d):
    init_part = get_init_particles_for_linear_map(part, d)
    fin_part = init_part.copy()
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

#iterate over ii turns to get a stable approximation
def get_CO_and_linear_map(part, line, d, tol):
    ii = 20
    for i in range(ii):
        M, m, part_new = linearize_around_closed_orbit(part, line, d)
        H = 0.
        H += abs(part_new.x - part.x)
        H += abs(part_new.px - part.px)
        H += abs(part_new.y - part.y)
        H += abs(part_new.py - part.py)
        H += abs(part_new.zeta - part.zeta)
        H += abs(part_new.delta - part.delta)
        if H < tol:
            print('Converged with distance: {}'.format(H))
            return part_new, M
        print ('Closed orbit search iteration: {}'.format(i))
        part = part_new
    print('Search did not converge, distance: {}'.format(H))
    return part, M

def dragt_symplectify(M): 
    #symplectic polar decomposition
    N = np.matmul(np.matmul(M , S), np.matmul(M.T, S.T))
    Q = scipy.linalg.expm(0.5 * scipy.linalg.logm(N))
    Ms = np.matmul(np.linalg.inv(Q), M)
    return Ms

def healy_symplectify(M):
    I = np.identity(6)

    V = np.matmul(S, np.matmul(I - M, np.linalg.inv(I + M)))
    W = (V + V.T)/2
    if np.linalg.det(I - np.matmul(S, W)) != 0:
        M_new = np.matmul(I + np.matmul(S, W), np.linalg.inv(I - np.matmul(S, W)))
    else:
        V_else = np.matmul(S, np.matmul(I + M, np.linalg.inv(I - M)))
        W_else = (V_else + V_else.T)/2
        M_new = -np.matmul(I + np.matmul(S, W_else), np.linalg(I - np.matmul(S, W_else)))
    return M_new

def Rot2D(mu):
    return np.array([[ np.cos(mu), np.sin(mu)],
                     [-np.sin(mu), np.cos(mu)]])

######################################################
### Implement Normalization of fully coupled motion ##
######################################################

def linear_normal_form(M):
    w0, v0 = np.linalg.eig(M)
    
    a0 = np.real(v0)
    b0 = np.imag(v0)
    
    index_list = [0,5,1,2,3,4]
    
    ##### Sort modes in pairs of conjugate modes #####
    
    conj_modes = np.zeros([3,2], dtype=np.int)
    for j in [0,1]:
        conj_modes[j,0] = index_list[0]
        del index_list[0]
        
        min_index = 0 
        min_diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[min_index]]))
        for i in range(1,len(index_list)):
            diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[i]]))
            if min_diff > diff:
                min_diff = diff
                min_index = i 
        
        conj_modes[j,1] = index_list[min_index]
        del index_list[min_index]
    
    conj_modes[2,0] = index_list[0]
    conj_modes[2,1] = index_list[1]
    
    ##################################################
    #### Select mode from pairs with positive (real @ S @ imag) #####
    
    modes = np.empty(3, dtype=np.int)
    for ii,ind in enumerate(conj_modes):
        if np.matmul(np.matmul(a0[:,ind[0]], S), b0[:,ind[0]]) > 0:
            modes[ii] = ind[0]
        else:
            modes[ii] = ind[1]
    
    ##################################################
    #### Sort modes such that (1,2,3) is close to (x,y,zeta) ####
    for i in [1,2]:
        if abs(v0[:,modes[0]])[0] < abs(v0[:,modes[i]])[0]:
            modes[0], modes[i] = modes[i], modes[0]
    
    if abs(v0[:,modes[1]])[2] < abs(v0[:,modes[2]])[2]:
        modes[2], modes[1] = modes[1], modes[2]
    
    ##################################################
    #### Rotate eigenvectors to the Courant-Snyder parameterization ####
    phase0 = np.log(v0[0,modes[0]]).imag
    phase1 = np.log(v0[2,modes[1]]).imag
    phase2 = np.log(v0[4,modes[2]]).imag
    
    v0[:,modes[0]] *= np.exp(-1.j*phase0)
    v0[:,modes[1]] *= np.exp(-1.j*phase1)
    v0[:,modes[2]] *= np.exp(-1.j*phase2)
    
    ##################################################
    #### Construct W #################################
    
    a1 = v0[:,modes[0]].real
    a2 = v0[:,modes[1]].real
    a3 = v0[:,modes[2]].real
    b1 = v0[:,modes[0]].imag
    b2 = v0[:,modes[1]].imag
    b3 = v0[:,modes[2]].imag
    
    n1 = 1./np.sqrt(np.matmul(np.matmul(a1, S), b1))
    n2 = 1./np.sqrt(np.matmul(np.matmul(a2, S), b2))
    n3 = 1./np.sqrt(np.matmul(np.matmul(a3, S), b3))
    
    a1 *= n1
    a2 *= n2
    a3 *= n3
    
    b1 *= n1
    b2 *= n2
    b3 *= n3
    
    W = np.array([a1,b1,a2,b2,a3,b3]).T
    W[abs(W) < 1.e-14] = 0. # Set very small numbers to zero.
    invW = np.matmul(np.matmul(S.T, W.T), S)
    
    ##################################################
    #### Get tunes and rotation matrix in the normalized coordinates ####
    
    mu1 = np.log(w0[modes[0]]).imag
    mu2 = np.log(w0[modes[1]]).imag
    mu3 = np.log(w0[modes[2]]).imag
    
    q1 = mu1/(2.*np.pi)
    q2 = mu2/(2.*np.pi)
    q3 = mu3/(2.*np.pi)
    
    R = np.zeros_like(W)
    R[0:2,0:2] = Rot2D(mu1)
    R[2:4,2:4] = Rot2D(mu2)
    R[4:6,4:6] = Rot2D(mu3)
    ##################################################    
    
    return W, invW, R
