import numpy as np
from .simulate import *

def sample_sphere(size, inp, sigma=0.0, d=2):

    z = np.random.normal(size=(size, d))
    z /= np.linalg.norm(z, axis = 1, keepdims=True)

    # z[:,0] = inp * np.abs(z[:,0]) #two hemispheres based on first coordiante
    
    ep = sigma * np.random.normal(size=(size, d))
    return z + ep

def sample_cylinder(size, inp, sigma=0.02, d=2):

    z = np.random.normal(size=(size, d))
    z[:,:2] /= np.linalg.norm(z[:,:2], axis = 1, keepdims=True)

    z[:,0] = inp * np.abs(z[:,0]) #two hemispheres based on first coordiante
    
    ep = sigma * np.random.normal(size=(size, d))
    return z + ep

def sample_ellipse(size, inp, sigma=0.0, d=2):

    z = np.random.normal(size=(size, d))
    z /= np.linalg.norm(z, axis = 1, keepdims=True)

    z[:,0] *= 10

    # z[:,0] = inp * np.abs(z[:,0]) #two hemispheres based on first coordiante
    
    ep = sigma * np.random.normal(size=(size, d))
    return z + ep

def sample_barbell(size, inp, sigma = 0.05, d=2):

    width = 30

    size0 = int(9/20*size)
    size2 = int(9/20*size)
    size1 = size - (size0 + size2)

    z0 = np.random.normal(size=(size0, d))
    z0[:,0] -= width

    x = np.random.uniform(low = -width, high = width, size=size1)
    z1 = sigma * np.random.normal(size=(size1, d))
    z1[:,0] += x
    z1[:,1] += np.sin(x/5)
    
    z2 = np.random.normal(size=(size2, d))
    z2[:,0] += width

    return np.concatenate([z0, z1, z2], axis=0)

def process_sphere_data(size, d, normalize=False):
    sizes = [size, size]
    signs = [1, -1]
    inputs = signs
    return process_fake_data(sizes, inputs, d, sample_sphere, normalize=normalize)

def process_cylinder_data(size, d, normalize=False):
    sizes = [size, size]
    signs = [1, -1]
    inputs = signs
    return process_fake_data(sizes, inputs, d, sample_cylinder, normalize=normalize)

def process_ellipse_data(size, d, normalize=False):
    sizes = [size, size]
    signs = [1, -1]
    inputs = signs
    return process_fake_data(sizes, inputs, d, sample_ellipse, normalize=normalize)

def process_barbell_data(size, d, normalize=True):
    sizes = [size, size]
    inputs = [0,1]
    return process_fake_data(sizes, inputs, d, sample_barbell, normalize=normalize)