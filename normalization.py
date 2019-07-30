import numpy as np

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
