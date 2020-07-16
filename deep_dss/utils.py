import numpy as np
import healpy as hp
import pandas as pd
from numba import jit


# C(l) helper functions

def path_to_cl(sigma8, name="f1z1f1z1", path_to_input="../data/flask/"):
    """
    Returns relative path to FLASK input C(l) generated by trough_lenser
    :param sigma8:Value of $\\sigma_8$ used to generate the C(l)s
    :param name: Name of the C(l)
    :param path_to_input: Path to flask input directory, ending in / (default assumes data folder in repo)
    :return: relative path string to the appropriate C(l) file
    """
    return path_to_input + "input/dss-20-0.28-{0}-1.54Cl-{1}.dat".format(round(sigma8, 3), name)


def load_cl_from_path(path, lmax=9999):
    """
    Generate pandas dataframe for a given input C(l) file
    :param path: path to C(l) file
    :param lmax: maximum l value in C(l) file
    :return: data frame containing vector of ls and corresponding C(l) values
    """
    data = pd.read_csv(path, sep=' ')
    data.columns = ['L', 'CL']
    data.index = np.arange(1, lmax + 1)
    return data


def load_cl_from_val(sigma8, lmax=9999, name="f1z1f1z1", path_to_input="../data/flask"):
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
    :param nest: True for NEST pixellization, False for RING
    :return: Numpy array with map
    """
    return load_map_by_path(path_to_map(sigma8, name=name, path_to_output=path_to_output), field=field, nest=nest)

# @jit(nopython=True)
# def accelerated_poissonian_shot_noise(map, nside=1024, npix=12*1024*1024, pixarea=
#                           nest=True, density=0.04377, density_0 = 0.04377, bias=1.54, normalize=True):
#     x = np.empty(map.shape)
#     for i in range(hp.nside2npix(nside)):
#         x[i] = np.random.poisson()
