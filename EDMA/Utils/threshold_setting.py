

from scipy.stats import norm
"""
    The threshold of the expected improvement based on the history data.
"""

def threshold_setting(mu, sigma, y_best):
    """
    Calculate the threshold of the expected improvement based on the history data.
    
    Parameters:
    mu (float): The mean of the predicted permeation values
    sigma (float): The standard deviation of the predicted permeation values
    y_best (float): The best permeation value
    
    Returns:
    float: The threshold of the expected improvement
    """
    # Calculate the threshold of the expected improvement
    z_score = (y_best - mu) / sigma
    probability = 1 - norm.cdf(z_score)
    threshold = probability * sigma
    return threshold