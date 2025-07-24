import numpy as np
from scipy.stats import norm

__all__ = ['probability_of_improvement']

def probability_of_improvement(mu, sigma, y_max, xi=0.0):
    """
    Compute the Probability of Improvement (PoI) acquisition values.

    Parameters
    ----------
    mu : array-like or float
        Predicted mean of the objective at query points.
    sigma : array-like or float
        Predicted standard deviation (uncertainty) of the objective at query points.
    y_max : float
        Current best (max) observed target value.
    xi : float, optional (default=0.0)
        Exploration parameter. Higher xi encourages exploration.

    Returns
    -------
    poi : ndarray or float
        Probability of Improvement at each query point.

    Notes
    -----
    PoI is defined as:
        PoI(x) = P(f(x) >= y_max + xi)
               = Phi((mu - y_max - xi) / sigma)
    where Phi is the CDF of the standard normal distribution.
    """
    # mu = np.asarray(mu)
    # sigma = np.asarray(sigma)
    # # Prevent division by zero
    # sigma = np.maximum(sigma, 1e-9)

    # # Compute standardized improvement
    # z = (mu - y_max - xi) / sigma

    # # Standard normal CDF
    # poi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    # return poi

    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    # Prevent division by zero or negative
    sigma = np.maximum(sigma, 1e-9)

    # Standardized improvement
    z = (mu - y_max - xi) / sigma

    # Use SciPy to compute standard normal CDF
    poi = norm.cdf(z)
    return poi

