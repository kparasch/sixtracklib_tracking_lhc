import h5py

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
         
