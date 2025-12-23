import numpy as np
from .simulate import *

def sample_quarter(size, inp, sigma=0.05, d=2):
    i, o = inp #angle, offset
    f = lambda x: (np.cos(x), np.sin(x))
    z = np.stack(f(np.pi/2 * i + np.pi/2 * np.random.uniform(size=size)), axis = 1)
    if d > 2:
        x = np.zeros((z.shape[0], d))
        x[:,:2] = z
        z = x
    ep = sigma * np.random.normal(size=(size, d))
    offset = np.pad(np.array(o), (0, d-2), 'constant')[None, :]
    return z + ep + offset

def process_fake_time_data(size, d):
    sizes = [size, size, size]
    angles = [0, 1, 2]
    offsets = [[0,0], [0,0], [0,0]]
    inputs = list(zip(angles, offsets))
    return process_fake_data(sizes, inputs, d, sample_quarter)

def process_fake_time_data_reweighted(size, d):
    sizes = [size//20, size//10, size]
    angles = [0, 1, 2]
    offsets = [[0,0], [0,0], [0,0]]
    inputs = list(zip(angles, offsets))
    return process_fake_data(sizes, inputs, d, sample_quarter)

def process_s_data(size, d):
    sizes = [size, size, size, size, size, size]
    angles = [0, 1, 2, 0, 3, 2]
    offsets = [[0,1], [0,1], [0,1], [0, -1], [0, -1], [0, -1]]
    inputs = list(zip(angles, offsets))
    return process_fake_data(sizes, inputs, d, sample_quarter)

def process_s_outlier_data(size, d):
    sizes = [size, size, size, size, size, size, 1]
    angles = [0, 1, 2, 0, 3, 2, 0]
    offsets = [[0,1], [0,1], [0,1], [0, -1], [0, -1], [0, -1], [3, 3]]
    inputs = list(zip(angles, offsets))
    return process_fake_data(sizes, inputs, d, sample_quarter)

def process_imbalanced_s_data(size, d):
    sizes = [size//20, size//10, size, size//20, size, size//20]
    angles = [0, 1, 2, 0, 3, 2]
    offsets = [[0,1], [0,1], [0,1], [0, -1], [0, -1], [0, -1]]
    inputs = list(zip(angles, offsets))
    return process_fake_data(sizes, inputs, d, sample_quarter)
