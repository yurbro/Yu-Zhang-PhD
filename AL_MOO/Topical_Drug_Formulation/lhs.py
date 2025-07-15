# requirements: pip install pyDOE2 numpy

import numpy as np
from pyDOE2 import lhs

def generate_lhs_samples(bounds, n_samples, criterion='maximin'):
    """
    Generate samples using Latin Hypercube Sampling (LHS) and map them to the specified bounds.
    
    Parameters
    ----------
    bounds : ndarray of shape (d, 2)
        Lower and upper bounds for each variable [[min1, max1], [min2, max2], ..., [mind, maxd]].
    n_samples : int
        Number of samples to generate.
    criterion : str
        Optimization criterion for LHS, options include 'center', 'maximin', 'centermaximin', 'correlation', etc.

    Returns
    -------
    samples : ndarray of shape (n_samples, d)
        Sampling results, each row is a sample, each column corresponds to a variable.
    """
    bounds = np.asarray(bounds)
    d = bounds.shape[0]
    # 1. Generate LHS points in [0,1]^d
    unit_lhs = lhs(d, samples=n_samples, criterion=criterion)
    # 2. Map to actual bounds
    samples = np.zeros_like(unit_lhs)
    for i in range(d):
        lo, hi = bounds[i]
        samples[:, i] = lo + unit_lhs[:, i] * (hi - lo)
    return samples

if __name__ == "__main__":
    # Example: 3 variables, ranges are
    # Poloxamer407: [20, 30], Ethanol: [10, 20], PG: [10, 20]
    bounds = np.array([
        [20.0, 30.0],
        [10.0, 20.0],
        [10.0, 20.0],
    ])
    n_samples = 500  # Generate 500 samples
    lhs_samples = generate_lhs_samples(bounds, n_samples)

    # Print the first 20 samples for inspection
    print(lhs_samples[:20])
