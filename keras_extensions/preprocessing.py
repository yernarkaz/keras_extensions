from __future__ import division
import numpy as np
import warnings

def standardize(X, axis=0, ddof=0):
    """Standardize input data to have zero mean and unit variance along feature dimensions.
    
    Similar to scikit-learn.processing.scale(), but returns means and variances in addition to 
    standarized input, and has ddof parameter which affects computation of variances.

    :param X:       Input data, typically nframes-by-nfeat matrix.
    :param axis:    Axis over which to standardize (i.e. 0 standardize each features independently, 
                    1 standardizes each sample over all features).
    :param ddof:    Degrees of freedom for variance calculation (i.e. 0 for maximum likelihood estimate of variance,
                    1 for unbiased estimation of variance).
    """

    # Modified from scikit-learn.preprocessing.scale()!

    #X = np.asarray(X)
    X = np.asarray(X, dtype=np.float)       # XXX: what about dtype? convert to float64? for higher precision? let client decide?
    Xr = np.rollaxis(X, axis)   # view on X to enable broadcasting on the axis we are interested in
    
    mean_ = Xr.mean(axis=0)
    std_  = Xr.std(axis=0, ddof=ddof)
    std_[std_ == 0.0] = 1.0     # avoid NaNs due to div/zero

    # center mean on zero
    Xr -= mean_

    # Verify that mean_1 is 'close to zero'. If X contains very
    # large values, mean_1 can also be very large, due to a lack of
    # precision of mean_. In this case, a pre-scaling of the
    # concerned feature is efficient, for instance by its mean or
    # maximum.
    mean_1 = Xr.mean(axis=0)
    if not np.allclose(mean_1, 0.0):
        warnings.warn("Numerical issues were encountered "
                      "when centering the data "
                      "and might not be solved. Dataset may "
                      "contain too large values. You may need "
                      "to prescale your features.")
        Xr -= mean_1
        mean_ += mean_1

    # scale to unit variance
    Xr /= std_

    # If mean_2 is not 'close to zero', it comes from the fact that
    # std_ is very small so that mean_2 = mean_1/std_ > 0, even if
    # mean_1 was close to zero. The problem is thus essentially due
    # to the lack of precision of mean_. A solution is then to
    # substract the mean again.
    mean_2 = Xr.mean(axis=0)
    if not np.allclose(mean_2, 0.0):
        warnings.warn("Numerical issues were encountered "
                      "when scaling the data "
                      "and might not be solved. The standard "
                      "deviation of the data is probably "
                      "very close to 0.")
        Xr -= mean_2
        mean_ += mean_2

    # Additional check if variances are 'close to one'
    std_1 = Xr.std(axis=0, ddof=ddof)
    if not np.allclose(std_1, 1.0):
        warnings.warn("Numerical issues were encountered "
                      "when scaling the data "
                      "and might not be solved. Standard deviation "
                      "not close to one after scaling.")

    return X, mean_, std_
