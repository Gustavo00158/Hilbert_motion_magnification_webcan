import h5py
import numpy as np


def save_matrix(matrix, name):
    hf = h5py.File('%s.h5' % name, 'w')
    hf.create_dataset('dataset', data=matrix, compression="gzip", compression_opts=9)
    hf.close()


def read_matrix(name):
    hf = h5py.File('%s.h5' % name, 'r')
    dataset = hf.get('dataset')
    matrix = np.array(dataset)
    hf.close()
    return matrix

def Vetor_save(vetor, name, path):
    with h5py.File(path + 'vetor_coord'+name+'.h5', 'w') as f:
        f.create_dataset('dataset', data=vetor)
    return "sucessful salve vector!"