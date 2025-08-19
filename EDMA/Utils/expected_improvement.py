
from Utils.probability_of_reaching_better_permeation import probability_of_prediction
import numpy as np
from scipy.stats import norm


def custom_expected_improvement(X, gp, y_max, xi):
    """
    Custom utility function to evaluate the potential of new points.

    Arguments:
        X: The points to evaluate (unscaled).
        gp: The current Gaussian process model.
        X_train_scaled: Scaled training inputs.
        y_train_scaled: Scaled training outputs.
        scaler_y: Scaler for the outputs.

    Returns:
        Utility value based on the probability of exceeding the best observed results.
    """
    # Scale the new points
    # X_scaled = scaler_X.transform(X)

    # Predict the mean and std deviation of the new points
    mu, sigma = gp.predict(X, return_std=True)
    # mu_rescaled = scaler_y.inverse_transform(mu)

    # Calculate the expected improvement
    with np.errstate(divide='warn'):
        imp = mu - y_max - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0


    return ei
