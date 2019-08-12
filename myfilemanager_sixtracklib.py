import h5py
import numpy

#def dict_to_h5(dict_save, filename):
#    with h5py.File(filename, 'w') as fid:
#        for kk in dict_save.keys():
#            fid[kk] = dict_save[kk]

def h5_to_dict(filename, group=None):
    fid = h5py.File(filename, 'r')
    if group == None :
        grp = fid
    else:
        grp = fid[group]
    mydict={}
    for key in grp.keys():
        mydict[key] = grp[key][()]
    return mydict


def dict_to_h5(dict_save, filename, compression_opts=4, group=None, readwrite_opts='w'):
    with h5py.File(filename, readwrite_opts) as fid:

        if group == None :
            grp = fid
        else:
            grp = fid.create_group(group)

        for kk in dict_save.keys():
            if isinstance(dict_save[kk], numpy.ndarray):
                print('Compressing '+kk)
                dset = grp.create_dataset(kk, shape=dict_save[kk].shape, compression='gzip', compression_opts=compression_opts)
                dset[...] = dict_save[kk]
            else:
                grp[kk] = dict_save[kk]
