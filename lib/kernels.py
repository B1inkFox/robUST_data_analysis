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