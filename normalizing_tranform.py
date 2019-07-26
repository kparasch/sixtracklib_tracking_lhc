from cpymad.madx import Madx
import numpy as np
import scipy.linalg

def symplectify_matrix(M):

    S = np.array([[0,1,0,0,0,0],
                  [-1,0,0,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,-1,0,0,0],
                  [0,0,0,0,0,1],
                  [0,0,0,0,-1,0]
                 ]
                )
    N = np.matmul( np.matmul(M, S), np.matmul(M.T, S.T) )
    Q = scipy.linalg.expm( 0.5* scipy.linalg.logm(N) )
    R = np.matmul(scipy.linalg.inv(Q) , M)
    return R

def get_normalizing_transform(R):

    S = np.array([[0,1,0,0,0,0],
                  [-1,0,0,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,-1,0,0,0],
                  [0,0,0,0,0,1],
                  [0,0,0,0,-1,0]
                 ]
                )

    w,a = scipy.linalg.eig(R)

    c1 = ((a[:,0] @ S @ a[:,1].T)*(-1.j))
    c2 = ((a[:,0] @ S @ a[:,1].T)*(-1.j))
    c3 = ((a[:,0] @ S @ a[:,1].T)*(-1.j))

    a[:,0:2] /= np.sqrt(c1)
    a[:,2:4] /= np.sqrt(c2)
    a[:,4:6] /= np.sqrt(c3)

    Q = 1./np.sqrt(2)*np.array([[1,1,0,0,0,0],
                                [-1.j,1.j,0,0,0,0],
                                [0,0,1,1,0,0],
                                [0,0,-1.j,1.j,0,0],
                                [0,0,0,0,1,1],
                                [0,0,0,0,-1.j,1.j]
                               ]
                              )
    N = Q @ scipy.linalg.inv(a)
seq = 'lhcb1'

mad = Madx()
mad.options.echo = False
mad.options.warn = False
mad.options.info = False

mad.call('000_PrepareLine/lhcwbb_fortracking.seq')
mad.use(seq)
twi = mad.twiss(rmatrix=True)

R = np.zeros([6,6])
for i in range(6):
    for j in range(6):
        R[i,j] = twi['re%d%d'%(i+1,j+1)][0]



#I = np.eye(6)
#SM2 = np.matmul(S,R)/2.
#tanhSM2 = scipy.linalg.tanhm(SM2)
#
#F = np.matmul(I+tanhSM2, scipy.linalg.inv(I - tanh(SM2)))


