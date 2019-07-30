import numpy as np

def random_hypersphere_reject(sigma, nr_particles, dim, seed = 0):
    np.random.seed(seed)
    np.random.uniform((1,10000))
    i = 0 
    sphere = np.zeros((nr_particles, dim)) 
    while i < nr_particles:
        point = np.random.uniform(low=0, high=sigma, size=(dim,))
        if np.linalg.norm(point) <= sigma:
            sphere[i] = point
            i += 1
    return sphere


