import h5py
import numpy

def dict_to_h5(dict_save, filename):
    with h5py.File(filename, 'w') as fid:
        for kk in dict_save.keys():
            fid[kk] = dict_save[kk]

def h5_to_dict(filename):
    f = h5py.File(filename, 'r')
    mydict={} 
    for key in f.keys(): 
        mydict[key] = f[key][()] 
    return mydict
         

def dict_to_h5_gzip_compressed(dict_save, filename, compression_opts=4):
    with h5py.File(filename, 'w') as fid:
        for kk in dict_save.keys():
            if isinstance(dict_save[kk], numpy.ndarray):
                print('Compressing '+kk)
                dset = fid.create_dataset(kk, dict_save[kk].shape, compression='gzip', compression_opts=compression_opts)
                dset[...] = dict_save[kk]
            else:
                fid[kk] = dict_save[kk]

