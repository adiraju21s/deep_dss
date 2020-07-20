import numpy as np
import healpy as hp
import pandas as pd
from numba import jit
from sklearn.utils import shuffle

from deepsphere import experiment_helper


# Constants


def set_constants():
    """
    Sets constants for future functions (especially accelerated ones)
    :return: NSIDE, NPIX, PIXEL_AREA (in arcmin^2), ORDER, BIAS, DENSITY_M, DENSITY_KG and ELLIP_SIGMA
    """
    nside = 1024
    npix = hp.nside2npix(nside)
    pixel_area = hp.nside2pixarea(nside, degrees=True) * 3600
    order = 2
    bias = 1.54
    density_m = 0.04377
    density_kg = 10
    ellip_sigma = 0.25
    return nside, npix, pixel_area, order, bias, density_m, density_kg, ellip_sigma


(NSIDE, NPIX, PIXEL_AREA, ORDER, BIAS, DENSITY_M, DENSITY_KG, ELLIP_SIGMA) = set_constants()


# C(l) helper functions

def path_to_cl(sigma8, name="f1z1f1z1", path_to_input="../data/flask/input/"):
    """
    Returns relative path to FLASK input C(l) generated by trough_lenser
    :param sigma8:Value of $\\sigma_8$ used to generate the C(l)s
    :param name: Name of the C(l)
    :param path_to_input: Path to flask input directory, ending in / (default assumes data folder in repo)
    :return: relative path string to the appropriate C(l) file
    """
    return path_to_input + "dss-20-0.28-{0}-1.54Cl-{1}.dat".format(round(sigma8, 3), name)


def load_cl_from_path(path, lmax=9999):
    """
    Generate pandas dataframe for a given input C(l) file
    :param path: path to C(l) file
    :param lmax: maximum l value in C(l) file
    :return: data frame containing vector of ls and corresponding C(l) values
    """
    data = pd.read_csv(path, sep=' ')
    data.columns = ['L', 'CL']
    data.index = np.arange(lmax + 1)
    return data


def load_cl_from_val(sigma8, lmax=9999, name="f1z1f1z1", path_to_input="../data/flask/input/"):
    """
    Wrapper function to return pandas data frame for a specified C(l)
    :param sigma8: Value of $\\sigma_8$ used to generate the C(l)s
    :param lmax: maximum l value in C(l) file
    :param name: Name of the C(l)
    :param path_to_input: Path to flask input directory, ending in / (default assumes data folder in repo)
    :return: data frame containing vector of ls and corresponding C(l) values
    """
    return load_cl_from_path(path_to_cl(sigma8, name=name, path_to_input=path_to_input), lmax=lmax)


# Descriptions of different data sets

def full_cosmologies_list():
    """
    Return the full list of $\\sigma_8$ values in the simulated data
    :return: A numpy array covering all 101 $\\sigma_8$ values in the flat prior
    """
    return np.linspace(0.5, 1.2, num=101)


def q1_cosmologies_list():
    """
    Return the list of $\\sigma_8$ values used in training Q1
    :return: A numpy array of 20 $\\sigma_8$ values
    """
    return np.array([1.165, 0.766, 1.095, 0.976, 0.99, 0.773, 0.57, 0.64, 0.563,
                     1.193, 0.584, 0.542, 1.109, 0.969, 0.983, 0.675, 1.039, 0.927,
                     1.032, 1.06])


def q2_cosmologies_list():
    """
    Return the list of $\\sigma_8$ values used in training Q2
    :return: A numpy array of 20 $\\sigma_8$ values
    """
    return np.array([0.843, 0.857, 0.535, 1.186, 1.144, 0.906, 0.962, 1.067, 0.815,
                     0.822, 0.717, 0.808, 1.13, 1.004, 0.626, 1.123, 0.724, 0.913,
                     0.696, 0.745])


def q3_cosmologies_list():
    """
    Return the list of $\\sigma_8$ values used in training Q3
    :return: A numpy array of 20 $\\sigma_8$ values
    """
    return np.array([0.829, 0.605, 0.647, 1.088, 0.864, 0.92, 0.661, 0.997, 0.955,
                     1.053, 0.759, 0.703, 0.934, 0.738, 0.752, 1.018, 0.794, 0.619,
                     0.892, 1.116])


def q4_cosmologies_list():
    """
    Return the list of $\\sigma_8$ values used in training Q4
    :return: A numpy array of 20 $\\sigma_8$ values
    """
    return np.array([0.787, 0.5, 0.836, 0.577, 1.179, 0.899, 0.598, 0.78, 0.941,
                     0.528, 1.2, 1.081, 0.948, 0.507, 0.633, 0.85, 1.137, 0.689,
                     1.074, 0.521])


def test_cosmologies_list():
    """
    Return the list of $\\sigma_8$ values used in testing
    :return: A numpy array of 21 $\\sigma_8$ values
    """
    return np.array([0.682, 1.102, 0.514, 0.885, 1.025, 1.158, 0.612, 1.011, 0.878,
                     1.172, 0.871, 1.151, 1.046, 0.591, 0.549, 0.71, 0.654, 0.668,
                     0.731, 0.556, 0.801])


def cosmologies_list(dataset):
    """
    Returns list of $\\sigma_8$ values for an input data set
    :param dataset: Name of data set
    :return: Numpy array of 20, 21, or 101 values
    """
    if dataset == "Q1":
        return q1_cosmologies_list()
    if dataset == "Q2":
        return q2_cosmologies_list()
    if dataset == "Q3":
        return q3_cosmologies_list()
    if dataset == "Q4":
        return q4_cosmologies_list()
    if dataset == "TEST":
        return test_cosmologies_list()
    if dataset == "FULL":
        full_cosmologies_list()
    print("Invalid data set specification. Please try again")


def dataset_names(val=False):
    """
    Returns list of data set names
    :param val: Whether or not to include the validation set
    :return: List of strings
    """
    if val:
        return ["Q1", "Q2", "Q3", "Q4", "TEST", "VAL"]
    return ["Q1", "Q2", "Q3", "Q4", "TEST"]


# Map loading functions

def path_to_map(sigma8, name="map-f1z1.fits.gz", path_to_output="../data/flask/output/"):
    """
    Return relative path to Healpix map given $\\sigma_8$
    :param sigma8: Value of $\\sigma_8$$ from which the map was generated
    :param name: Name of the map file
    :param path_to_output: Relative path to the FLASK output directory
    :return: String with path
    """
    return path_to_output + "dss-20-0.28-{0}-1.54/{1}".format(round(sigma8, 3), name)


def load_map_by_path(path, field=0, nest=True):
    """
    Returns HEALPIX map located at a given path
    :param path: relative path to the map
    :param field: field of the map (for lensing maps with multiple fields)
    :param nest: True for NEST pixellization, False for RING
    :return: Numpy array with map
    """
    return hp.read_map(path, field=field, nest=nest)


def load_map_by_val(sigma8, name="map-f1z1.fits.gz", path_to_output="../data/flask/output/", field=0, nest=True):
    """
    Returns HEALPIX map for FLASK realization of a given $\\sigma_8$ value
    :param sigma8: Value of $\\sigma_8$
    :param name: name of the map
    :param path_to_output: relative path to the FLASK output directory
    :param field: field of the map (for lensing maps with multiple fields)
    :param nest: True for NEST pixelization, False for RING
    :return: Numpy array with map
    """
    return load_map_by_path(path_to_map(sigma8, name=name, path_to_output=path_to_output), field=field, nest=nest)


@jit(nopython=True)
def accelerated_poissonian_shot_noise(m, npix=NPIX, pixarea=PIXEL_AREA,
                                      density=DENSITY_M, density_0=DENSITY_M, multiplier=250.0, bias=BIAS,
                                      normalize=True):
    """
    Returns new version of input map with a specified level of Poissonian shot noise applied.
    :param m: FLASK output map of galaxy density contrast, $\\delta_g$
    :param npix: Number of pixels in map
    :param pixarea: Area of each pixel, in arcmin^2
    :param nest: True if "NEST" pixelization, False if "RING"
    :param density: Tracer galaxy density, in arcmin^2, to use for noise application
    :param density_0: Baseline galaxy density, in arcmin^2, to scale distribution by
    :param multiplier: Scale factor used to amplify noise distribution
    :param bias: Linear galaxy-matter bias
    :param normalize: True if resulting noise should be made to reflect a linear galaxy-matter bias of 1
    :return: A noisy Poisson-sampled output map
    """
    x = np.empty(m.shape)
    for i in range(npix):
        if normalize:
            x[i] = multiplier * (density_0 / density) * np.random.poisson(density * pixarea * (1 + m[i] / bias))
        else:
            x[i] = multiplier * (density_0 / density) * np.random.poisson(density * pixarea * (1 + m[i]))
    return x


def poisson_map_by_val(sigma8, name="map-f1z1.fits.gz", path_to_output="../data/flask/output/", field=0, nest=True,
                       npix=NPIX, pixarea=PIXEL_AREA, density=DENSITY_M, density_0=DENSITY_M, multiplier=250.0,
                       bias=BIAS, normalize=True):
    """
    Loads galaxy density contrast map for a given $\\sigma_8$ and applies Poissonian shot noise
    :param sigma8: Value of $\\sigma_8$
    :param name: name of the map
    :param path_to_output: relative path to the FLASK output directory
    :param field: field of the map (for lensing maps with multiple fields)
    :param nest: True for NEST pixellization, False for RING
    :return: Numpy array with map
    :param npix: Number of pixels in map
    :param pixarea: Area of each pixel, in arcmin^2
    :param nest: True if "NEST" pixelization, False if "RING"
    :param density: Tracer galaxy density, in arcmin^2, to use for noise application
    :param density_0: Baseline galaxy density, in arcmin^2, to scale distribution by
    :param multiplier: Scale factor used to amplify noise distribution
    :param bias: Linear galaxy-matter bias
    :param normalize: True if resulting noise should be made to reflect a linear galaxy-matter bias of 1
    :return: A noisy Poisson-sampled output map
    """
    return accelerated_poissonian_shot_noise(
        load_map_by_val(sigma8, name=name, path_to_output=path_to_output, field=field, nest=nest),
        npix=npix, pixarea=pixarea, density=density, density_0=density_0, multiplier=multiplier, bias=bias,
        normalize=normalize)


def split_map(m, order=ORDER, nest=True):
    """
    Returns Numpy array of partial-sky Healpix realizations split from an input full-sky map
    :param m: Full-sky Healpix map
    :param order: ORDER giving the number of maps to split into (12*ORDER**2)
    :param nest: True if "NEST" pixelization, False if "RING"
    :return: Numpy array of split maps
    """
    return experiment_helper.hp_split(m, order, nest)


def split_poisson_map_by_val(sigma8, name="map-f1z1.fits.gz", path_to_output="../data/flask/output/", field=0,
                             nest=True, npix=NPIX, pixarea=PIXEL_AREA, density=DENSITY_M, density_0=DENSITY_M,
                             multiplier=250.0,
                             bias=BIAS, normalize=True, order=ORDER):
    """
    Generates partial-sky maps with applied Poissonian shot noise for a given $\\sigma_8$
    :param sigma8: Value of $\\sigma_8$
    :param name: name of the map
    :param path_to_output: relative path to the FLASK output directory
    :param field: field of the map (for lensing maps with multiple fields)
    :param nest: True for NEST pixellization, False for RING
    :return: Numpy array with map
    :param npix: Number of pixels in map
    :param pixarea: Area of each pixel, in arcmin^2
    :param nest: True if "NEST" pixelization, False if "RING"
    :param density: Tracer galaxy density, in arcmin^2, to use for noise application
    :param density_0: Baseline galaxy density, in arcmin^2, to scale distribution by
    :param multiplier: Scale factor used to amplify noise distribution
    :param bias: Linear galaxy-matter bias
    :param normalize: True if resulting noise should be made to reflect a linear galaxy-matter bias of 1
    :param order: ORDER giving the number of maps to split into (12*ORDER**2)
    :return: Numpy array of split, (rescaled) Poisson-sampled maps
    """
    return split_map(poisson_map_by_val(sigma8, name=name, path_to_output=path_to_output, field=field, nest=nest,
                                        npix=npix, pixarea=pixarea, density=density, density_0=density_0,
                                        multiplier=multiplier,
                                        bias=bias, normalize=normalize), order=order, nest=nest)


def split_poisson_maps_by_vals(sigma8s, name="map-f1z1.fits.gz", path_to_output="../data/flask/output/", field=0,
                               nest=True, npix=NPIX, pixarea=PIXEL_AREA, density=DENSITY_M, density_0=DENSITY_M,
                               multiplier=250.0,
                               bias=BIAS, normalize=True, order=ORDER, scramble=False, ground_truths=True):
    """
    Generates stacked array of partial-sky Poisson-sampled maps for a list of $\\sigma_8$ values
    :param sigma8s: List of $\\sigma_8$ values
    :param name: name of the map
    :param path_to_output: relative path to the FLASK output directory
    :param field: field of the map (for lensing maps with multiple fields)
    :param nest: True for NEST pixellization, False for RING
    :return: Numpy array with map
    :param npix: Number of pixels in map
    :param pixarea: Area of each pixel, in arcmin^2
    :param nest: True if "NEST" pixelization, False if "RING"
    :param density: Tracer galaxy density, in arcmin^2, to use for noise application
    :param density_0: Baseline galaxy density, in arcmin^2, to scale distribution by
    :param multiplier: Scale factor used to amplify noise distribution
    :param bias: Linear galaxy-matter bias
    :param normalize: True if resulting noise should be made to reflect a linear galaxy-matter bias of 1
    :param order: ORDER giving the number of maps to split into (12*ORDER**2)
    :param scramble: If True, randomly scrambles the maps out of order
    :param ground_truths: True if corresponding labels should be returned as well
    :return: Dictionary of maps and labels if ground_truths=True,
        stacked Numpy array of split, (rescaled) Poisson-sampled maps otherwise
    """
    x = np.empty((0, npix // (12 * order * order)))
    for sigma8 in sigma8s:
        m = split_poisson_map_by_val(sigma8, name=name, path_to_output=path_to_output, field=field,
                                     nest=nest, npix=npix, pixarea=pixarea, density=density, density_0=density_0,
                                     multiplier=multiplier,
                                     bias=bias, normalize=normalize, order=order)
        x = np.vstack((x, m))
    x = np.reshape(x, (len(sigma8s) * 12 * order * order, npix // (12 * order * order), 1))
    if ground_truths:
        y = np.zeros(len(sigma8s) * 12 * order * order)
        for i in range(len(sigma8s)):
            y[i * 12 * order * order:(i * 12 * order * order + 12 * order * order)] = sigma8s[i]
        y = np.reshape(y, (len(sigma8s) * 12 * order * order, 1))
        if scramble:
            (x, y) = shuffle(x, y, random_state=0)
        return {"x": x, "y": y}
    if scramble:
        x = shuffle(x, random_state=0)
    return x


def split_poisson_maps_by_dataset(dataset, name="map-f1z1.fits.gz", path_to_output="../data/flask/output/", field=0,
                                  nest=True, npix=NPIX, pixarea=PIXEL_AREA, density=DENSITY_M, density_0=DENSITY_M,
                                  multiplier=250.0,
                                  bias=BIAS, normalize=True, order=ORDER, scramble=False, ground_truths=True):
    """
    Generates stacked array of partial-sky Poisson-sampled maps for a given data set
    :param dataset: String name of data-set to be used
    :param name: name of the map
    :param path_to_output: relative path to the FLASK output directory
    :param field: field of the map (for lensing maps with multiple fields)
    :param nest: True for NEST pixelization, False for RING
    :return: Numpy array with map
    :param npix: Number of pixels in map
    :param pixarea: Area of each pixel, in arcmin^2
    :param nest: True if "NEST" pixelization, False if "RING"
    :param density: Tracer galaxy density, in arcmin^2, to use for noise application
    :param density_0: Baseline galaxy density, in arcmin^2, to scale distribution by
    :param multiplier: Scale factor used to amplify noise distribution
    :param bias: Linear galaxy-matter bias
    :param normalize: True if resulting noise should be made to reflect a linear galaxy-matter bias of 1
    :param order: ORDER giving the number of maps to split into (12*ORDER**2)
    :param scramble: If True, randomly scrambles the maps out of order
    :param ground_truths: True if corresponding labels should be returned as well
    :return: Stacked Numpy array of split, (rescaled) Poisson-sampled maps
    """
    return split_poisson_maps_by_vals(cosmologies_list(dataset), name=name, path_to_output=path_to_output, field=field,
                                      nest=nest, npix=npix, pixarea=pixarea, density=density, density_0=density_0,
                                      multiplier=multiplier,
                                      bias=bias, normalize=normalize, order=order, scramble=scramble,
                                      ground_truths=ground_truths)


def split_poisson_maps_by_datasets(val=False, name="map-f1z1.fits.gz", path_to_output="../data/flask/output/",
                                   field=0,
                                   nest=True, npix=NPIX, pixarea=PIXEL_AREA, density=DENSITY_M, density_0=DENSITY_M,
                                   multiplier=250.0,
                                   bias=BIAS, normalize=True, order=ORDER, scramble=False, ground_truths=True):
    """
    Returns a data dictionary containing Poisson-sampled maps for each data-set
    :param val: If True, validation set is included in dataset_names()
    :param name: name of the map
    :param path_to_output: relative path to the FLASK output directory
    :param field: field of the map (for lensing maps with multiple fields)
    :param nest: True for NEST pixelization, False for RING
    :return: Numpy array with map
    :param npix: Number of pixels in map
    :param pixarea: Area of each pixel, in arcmin^2
    :param nest: True if "NEST" pixelization, False if "RING"
    :param density: Tracer galaxy density, in arcmin^2, to use for noise application
    :param density_0: Baseline galaxy density, in arcmin^2, to scale distribution by
    :param multiplier: Scale factor used to amplify noise distribution
    :param bias: Linear galaxy-matter bias
    :param normalize: True if resulting noise should be made to reflect a linear galaxy-matter bias of 1
    :param order: ORDER giving the number of maps to split into (12*ORDER**2)
    :param scramble: If True, randomly scrambles the maps out of order
    :param ground_truths: True if corresponding labels should be returned as well
    :return: Dictionary, each value of which is a stacked Numpy array of split, (rescaled) Poisson-sampled maps
    """
    data = {}
    for dataset in dataset_names(val=val):
        data[dataset] = split_poisson_maps_by_dataset(dataset=dataset, name=name, path_to_output=path_to_output,
                                                      field=field,
                                                      nest=nest, npix=npix, pixarea=pixarea, density=density,
                                                      density_0=density_0, multiplier=multiplier,
                                                      bias=bias, normalize=normalize, order=order,
                                                      scramble=scramble,
                                                      ground_truths=ground_truths)
    return data
