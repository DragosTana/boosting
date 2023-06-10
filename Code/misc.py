import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind

def simulatedData1(n = 10, seed = None):
    """
    This function simulates the data foloowing the example 3.10 of the Elements of Statistical Learning in the Boosting section.
    """
    np.random.seed(seed)
    X = pow(np.random.randn(n, 10), 2) 
    Y = [1 if np.sum(X[i,:]) > 9.34 else -1 for i in range(n)]
    Y = np.array(Y)
    Y = Y.reshape((n,1))
    Y = Y.astype(int)
    Y.flatten()
    return X, Y

def simulatedData2(n: int = 100, noise: float = 1):
    """
    This function generates a simulated dataset for regression
    using the function 7 * sin(2x) * exp(-0.1x) + noise. (dumped sine wave)
    """
    X = np.random.uniform(0, 15, n)
    X = np.sort(X)
    Y = [7 * np.sin(2*x) * np.exp(-0.1*x) + noise * np.random.normal(0, 1) for x in X]
    Y_true = [7 * np.sin(2*x) * np.exp(-0.1*x) for x in X]
    
    return X, Y, Y_true
    
def simulatedData3(n = 100, seed = None, noise = 1):
    """
    Simulated data that follow parabolic function
    adding an outlier.
    """
    
    X = np.random.uniform(-10, 10, n)
    X = np.sort(X)
    Y = [pow(x, 2) + noise * np.random.normal(0, 1) for x in X]
    Y_true = [pow(x, 2) for x in X]
    
    return X, Y, Y_true
    
def simulatedDataInteraction(n = 1000, seed = None, noise = 1, interaction = 2):
    """
    This function generates a simulated dataset with interaction terms for regression.
    """
    X = np.random.randn(n, 10)
    Y = - 1 * X[:,0] + 4 * X[:,1] +  3 * X[:,2] - 5 * X[:,3] + 0.5 * X[:,4] + 0.1 * X[:,5] - 5 * X[:,6] + 7 * X[:,7] + 0.5 * X[:,8] + 4 * X[:,9] + noise * np.random.randn(n)
    #add interaction terms
    if interaction == 2:
        Y = 2 * X[:,0] * X[:,1] +  X[:,4] * X[:,5] - 3 * X[:,6] * X[:,7] 
        if interaction == 3:
            Y = X[:,2] *  X[:,4] * X[:,5] +  X[:,7] * X[:,8] * X[:,9]
            if interaction == 4:
                Y = 0.5 * X[:,0] * X[:,1] * X[:,2] * X[:,3] +  X[:,8] * X[:,9] * X[:,0] * X[:,1]
    
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def tTest():
    path = os.getcwd()
    gb_scores = np.genfromtxt(path + "/gb_scores.csv", delimiter=",")
    rf_scores = np.genfromtxt(path + "/rf_scores.csv", delimiter=",")
    
    accuracy_gb = gb_scores[:,0]
    accuracy_rf = rf_scores[:,0]
    
    precision_gb = gb_scores[:,1]
    precision_rf = rf_scores[:,1]
 
    recall_gb = gb_scores[:,2]
    recall_rf = rf_scores[:,2]
    
    
    print("Accuracy test:")
    test(accuracy_gb, accuracy_rf)
    print("\nPrecision test:")
    test(precision_gb, precision_rf)
    print("\nRecall test:")
    test(recall_gb, recall_rf)
    
def test(score_gb, score_rf):
    """
    This function performs a t-test to compare the performance of two models.
    """
    # Perform independent t-test
    t_stat, p_value = ttest_ind(score_gb, score_rf)

    # Compare p-value to significance level
    alpha = 0.05  # Example significance level
    if p_value < alpha:
        print("There is a significant difference between the models.")
        if np.mean(score_gb) > np.mean(score_rf):
            print("The Gradient Boosting model performs better.")
        else:
            print("The Random Forest model performs better.")
    else:
        print("There is no significant difference between the models.")
        
    print("t-statistic: {0:.3f}, p-value: {1:.3f}".format(t_stat, p_value))
          

if __name__ == "__main__":
    tTest()