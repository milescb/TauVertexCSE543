import numpy as np

def get_quantile_width(arr, cl=0.68):
    """Get width of `arr` at `cl`%. Default is 68% CL"""

    q1 = (1. - cl) / 2.
    q2 = 1. - q1
    y = np.quantile(arr, [q1, q2])
    width = (y[1] - y[0]) / 2.
    return width

def makeBins(bmin, bmax, nbins):
    """Make tuples to extract data between bins. 

    Parameters:
    ----------

    bmin : float
        Smallest value of bin
    bmax : float
        Largest value of bins
    nbins : int
        Number of desired bins

    Returns: 
    -------

    List of tuples containing bins
    """

    returnBins = []
    stepsize = (bmax - bmin) / nbins
    for i in range(nbins):
        returnBins.append((bmin + i*stepsize, bmin + (i+1)*stepsize))
    return returnBins

def response_curve(res, var, bins, cl=0.68):
    """Prepare data fot plotting the response and resolution curve

    Parameters:
    ----------

    res : vector
        Data to be prepared
    var : vector of floats
        Variable to be plotted against
    bins : int
        Number of bins to be used in computation
    cl=0.68 : float
        Confidence level to be used. Default is 68%

    Returns: 
    -------

    - bin centers : center value of each bin
    - bin errors : distance to nearest bin edge
    - means : mean of distribution within each bin
    - means statistical error : stat error of distribution within each bin
    - resolution : quantile width at cl% of distribution
    """

    _bin_centers = []
    _bin_errors = []
    _means = []
    _mean_stat_err = []
    _resol = []
    if np.any(np.isnan(res)):
        print('Data contains nans! Removing nans')
        nan_indices = np.isnan(res)
        res = res[~nan_indices]
        var = var[~nan_indices]
    for _bin in bins:
        a = res[(var > _bin[0]) & (var < _bin[1])]
        if len(a) == 0:
            print('Bin was empty! Moving on to next bin')
            continue
        if np.any(np.isnan(a)):
            print(f'Bin {_bin} contains nans! Moving on to next bin')
            continue
        if np.any(np.isinf(a)):
            print(f'Bin {_bin} contains infs! Moving on to next bin')
            continue
        _means += [np.mean(a)]
        _mean_stat_err += [np.std(a, ddof=1) / np.sqrt(np.size(a))]
        _resol += [get_quantile_width(a, cl=cl)]
        _bin_centers += [_bin[0] + (_bin[1] - _bin[0]) / 2]
        _bin_errors += [(_bin[1] - _bin[0]) / 2]
    return np.array(_bin_centers), np.array(_bin_errors), np.array(_means), np.array(_mean_stat_err), np.array(_resol)