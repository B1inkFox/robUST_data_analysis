# library functions

import numpy as np

def gaussian_kernel(sigma, radius=None):
    
    if radius is None:
        radius = int(3 * sigma)   # typical cutoff
    
    x = np.arange(-radius, radius + 1)
    
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    
    return kernel