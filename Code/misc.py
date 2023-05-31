import numpy as np

def simulatedData1(n = 10, seed = 1):
    """
    This function simulates the data foloowing the example 3.10 of the Elements of Statistical Learning in the Boosting section.
    """
    X = pow(np.random.randn(n, 10), 2) 
    Y = [1 if np.sum(X[i,:]) > 9.34 else -1 for i in range(n)]
    Y = np.array(Y)
    Y = Y.reshape((n,1))
    Y = Y.astype(int)
    return X, Y
    
if __name__ == "__main__":
    simulatedData1()