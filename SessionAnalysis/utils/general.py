import numpy as np
from scipy.optimize import leastsq



def zero_phase_kernel(x, x_center):
    """ Zero pads the 1D kernel x, so that it is aligned with the current element
        of x located at x_center.  This ensures that convolution with the kernel
        x will be zero phase with respect to x_center.
    """

    kernel_offset = x.size - 2 * x_center - 1 # -1 To center ON the x_center index
    kernel_size = np.abs(kernel_offset) + x.size
    if kernel_size // 2 == 0: # Kernel must be odd
        kernel_size -= 1
        kernel_offset -= 1
    kernel = np.zeros(kernel_size)
    if kernel_offset > 0:
        kernel_slice = slice(kernel_offset, kernel.size)
    elif kernel_offset < 0:
        kernel_slice = slice(0, kernel.size + kernel_offset)
    else:
        kernel_slice = slice(0, kernel.size)
    kernel[kernel_slice] = x

    return kernel


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.sqrt(x**2 + y**2)
    
    return theta, rho


def compute_empirical_cdf(x):

    x = np.array(x)
    cdf = [0, 0]
    unique_vals, unique_counts = np.unique(x, return_counts=True)
    cdf[0] = unique_vals
    cdf[1] = np.cumsum(unique_counts) / x.size

    return cdf


def central_difference(x, y):

    cdiffs = np.zeros(x.size)
    cdiffs[1:-1] = (y[2:] - y[0:-2]) / 2.
    cdiffs[0] = cdiffs[1]
    cdiffs[-1] = cdiffs[-2]

    return cdiffs


def bin_x_func_y(x, y, bin_edges, y_func=np.mean, y_func_args=[], y_func_kwargs={}):
    """ Bins data in y according to values in x as in a histogram, but instead
        of counts this function calls the function 'y_func' on the binned values
        of y. Returns the center point of the binned x values and the result of
        the function y_func(binned y data). """

    n_bins = len(bin_edges) - 1
    x_out = np.empty(n_bins)
    y_out = np.empty(n_bins)
    for edge in range(0, n_bins):
        x_out[edge] = bin_edges[edge] + (bin_edges[edge+1] - bin_edges[edge]) / 2
        x_index = np.logical_and(x >= bin_edges[edge], x < bin_edges[edge+1])
        y_out[edge] = y_func(y[x_index], *y_func_args, **y_func_kwargs)

    return x_out, y_out


def fit_cos_fixed_freq(x_val, y_val):
    """ Fit cosine function to x and y data assuming fixed frequency of 1. """
    y_mean = np.mean(y_val)
    y_amp = (np.amax(y_val) - np.amin(y_val)) / 2
    optimize_func = lambda x: (x[0] * (np.cos(x_val + x[1])) + x[2]) - y_val
    amp, phase, offset = leastsq(optimize_func, [y_amp, 0, y_mean])[0]

    if amp < 0:
        phase += np.pi
        amp *= -1

    return amp, phase, offset
