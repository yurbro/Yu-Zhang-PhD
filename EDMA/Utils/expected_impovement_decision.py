
import numpy as np
from scipy.stats import norm
from Utils.probability_of_reaching_better_permeation import probability_of_prediction, probability_of_prediction_m2, calculate_current_best, method1_probability, method2_probability
from Utils.difference_analysis import analyze_data, decide_method, calculate_probability
from icecream import ic


# expected improvement and decision-making
def ei_decision(mu, sigma, y_max, phi, threshold=0.5):

    # Calculate the probability of reaching a better permeation value
    # prob_to_better, current_best = probability_of_prediction(mu, sigma, y_max)
    prob_to_better, current_best, mu_each, sigma_each = method1_probability(mu, sigma, y_max, phi)

    # calculate the totally probability
    total_prob = np.mean(prob_to_better)

    # make decision based on the threshold value
    if total_prob >= threshold:                         # this threshold value should be concerned.
        decision = True
    else:
        decision = False

    # calculate the EI value
    ei_m1 = np.mean((mu_each - current_best) * norm.cdf((mu_each - current_best) / sigma_each) + sigma_each * norm.pdf((mu_each - current_best) / sigma_each))


    return prob_to_better, total_prob, decision, current_best, ei_m1

# next is another method to calculate the probability of reaching a better permeation value
def ei_decision_v2(mu, sigma, y_max, phi, threshold=0.5):

    # Calculate the probability of reaching a better permeation value
    # prob_to_better, current_best = probability_of_prediction_m2(mu, sigma, y_max)
    prob_to_better, current_best, mu_F, sigma_F = method2_probability(mu, sigma, y_max, phi)

    # calculate the totally probability
    total_prob = np.mean(prob_to_better)

    # make decision based on the threshold value
    if total_prob >= threshold:                         # this threshold value should be concerned.
        decision = True
    else:
        decision = False

    # calculate the EI value
    ei_m2 = np.mean((mu_F - current_best) * norm.cdf((mu_F - current_best) / sigma_F) + sigma_F * norm.pdf((mu_F - current_best) / sigma_F))

    return total_prob, decision, current_best, ei_m2

# automatic choose the method to calculate the probability of reaching a better permeation value
def ei_decision_auto(mu, sigma, y_max, threshold=0.5):

    # analyze the difference between groups
    analysis_results = analyze_data(mu, sigma)

    # decide which method to use
    chosen_method = decide_method(analysis_results)

    if chosen_method == 'method1':
        print('\033[90mDue to the large difference between groups, method1 is chosen.\033[m')
    else:
        print('\033[90mDue to the small difference between groups, method2 is chosen.\033[m')
    
    # calculate the probability of reaching a better permeation value
    # current_best = calculate_current_best(y_max)
    total_prob, current_best = calculate_probability(chosen_method, y_max, mu, sigma)

    # make decision based on the threshold value
    if total_prob >= threshold:                         # this threshold value should be concerned.
        decision = True
    else:
        decision = False
    
    # calculate the EI value
    ei = np.mean((mu - current_best) * norm.cdf((mu - current_best) / sigma) + sigma * norm.pdf((mu - current_best) / sigma))

    return total_prob, decision, current_best, ei