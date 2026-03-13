# library functions

import numpy as np

def gaussian_kernel(sigma, radius=None):
    
    if radius is None:
        radius = int(3 * sigma)   # typical cutoff
    
    x = np.arange(-radius, radius + 1)
    
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    
    return kernel

def first_derivative_kernel(dt=0.01):
    """
    Central difference first derivative kernel.
    """
    return np.array([-1, 0, 1]) / (2 * dt)

def second_derivative_kernel(dt=0.01):
    """
    Central difference second derivative kernel.
    """
    return np.array([1, -2, 1]) / (dt**2)

def smooth_1d(x, kernel):
    k = np.asarray(kernel, dtype=float)
    return np.convolve(x, k, mode='same')


def smooth_3d(x, kernel):
    """
    x: (N,3)
    smooth each coordinate separately
    """
    k = np.asarray(kernel, dtype=float)
    out = np.zeros_like(x)
    for j in range(3):
        out[:, j] = np.convolve(x[:, j], kernel, mode='same')
    return out