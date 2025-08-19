
import numpy as np
from icecream import ic

def q_uncertainty(pred_mean, pred_sigma, y_test, confidence_level=1.96):

    # Calculate the upper and lower bounds of the confidence interval
    confidence_interval_upper = pred_mean + confidence_level * pred_sigma
    confidence_interval_lower = pred_mean - confidence_level * pred_sigma

    # Initialize the quantification of uncertainty Q(x)
    Q_x_matrix = np.zeros_like(y_test, dtype=float)

    # 检查 y_test 中的每个值是否在对应的置信区间内
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if y_test[i, j] >= confidence_interval_lower[i, j] and y_test[i, j] <= confidence_interval_upper[i, j]:
                Q_x_matrix[i, j] = 1
            else:
                Q_x_matrix[i, j] = 0
    
    # Calculate the average quantification of uncertainty
    average_Q_x = np.mean(Q_x_matrix)

    return Q_x_matrix, average_Q_x



    # # Compare the predicted values with the actual values
    # for i in range(len(y_test)):
    #     actual_value =y_test[i]
    #     lower_bound = confidence_interval_lower[i]
    #     upper_bound = confidence_interval_upper[i]
    #     if actual_value >= lower_bound and actual_value <= upper_bound:
    #         Q_x[i] = 1  # The predicted value is within the confidence interval
    #     else:
    #         Q_x[i] = 0  # The predicted value is outside the confidence interval
