import os
import h5py
import numpy as np
from utils import extract_features, extract_midi


def generate_data(data_path, index_begin, index_end):
    """
    Create dataset from musicnet data.
    """
    
    X, Y = list(), list()

    with h5py.File(data_path, 'r') as f:
        IDs = list(f.keys())[index_begin:index_end]

        for ID in sorted(IDs):
            print('Reading', ID)
            signal = f[ID]['data']
            labels = f[ID]['labels']

            print('Extracting features..')
            x = extract_features(signal)
            print('Extracting notes..')
            y = extract_midi(labels, len(x))

            X.extend(x)
            Y.extend(y)

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

if __name__ == '__main__':
    data_range = 20

    for index_begin in range(0, 330, data_range):
        index_end = index_begin + data_range
        X, Y = generate_data('../../musicnet.h5', index_begin, index_end)

        output_folder = './data/{}-{}/'.format(index_begin, index_end)
        os.mkdir(output_folder)

        np.save(os.path.join(output_folder, 'X'), X)
        np.save(os.path.join(output_folder, 'Y'), Y)
