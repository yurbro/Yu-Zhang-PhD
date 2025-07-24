
import numpy as np
from scipy.stats import norm
from scipy import stats


# Assumption: given mean and variance of the permeation value, we can calculate the probability of exceeding a better permeation value.

# Calculate the current_best permeation value
def calculate_current_best(y_best, phi):
    """
    Calculate the current best permeation value based on the given y_best and phi.

    Parameters
    ----------
        y_test (np.array): test data of the permeation value.
        phi (float): a parameter to adjust the current best value.

    Returns
    -------
        float: The best permeation effect.
    """
    # current_best = np.max(np.mean(y_best, axis=1))

    current_best = y_best * (1 - phi)       

    return current_best

def method2_probability(mu_matrix, sigma_matrix, y_best, phi=0.0):
    mu_F = mu_matrix
    y_best = calculate_current_best(y_best, phi)
    sigma_F = np.sqrt(np.sum(sigma_matrix ** 2) / (mu_matrix.size))
    z = (y_best - mu_F) / sigma_F
    P_F2 = 1 - stats.norm.cdf(z)
    # P_F2 = norm.cdf(z)
    return P_F2, y_best, mu_F, sigma_F


# next is another method to calculate the probability of reaching a better permeation value
def PoE_decision_v2(mu, sigma, y_max, phi, threshold=0.5):

    # Calculate the probability of reaching a better permeation value
    # prob_to_better, current_best = probability_of_prediction_m2(mu, sigma, y_max)
    prob_to_better, current_best, mu_F, sigma_F = method2_probability(mu, sigma, y_max, phi)

    # calculate the totally probability
    # total_prob = np.mean(prob_to_better)

    # # make decision based on the threshold value
    # if total_prob >= threshold:                         # this threshold value should be concerned.
    #     decision = True
    # else:
    #     decision = False

    # calculate the EI value
    ei_m2 = np.mean((mu_F - current_best) * norm.cdf((mu_F - current_best) / sigma_F) + sigma_F * norm.pdf((mu_F - current_best) / sigma_F))

    return prob_to_better, current_best, ei_m2

# Probability of Improvement (PoI) acquisition function
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
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    # Prevent division by zero
    sigma = np.maximum(sigma, 1e-9)

    # Compute standardized improvement
    z = (mu - y_max - xi) / sigma

    # Standard normal CDF
    poi = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
    return poi
