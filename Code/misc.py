import numpy as np

def simulatedData1(n = 10, seed = None):
    """
    This function simulates the data foloowing the example 3.10 of the Elements of Statistical Learning in the Boosting section.
    """
    X = pow(np.random.randn(n, 10), 2) 
    Y = [1 if np.sum(X[i,:]) > 9.34 else -1 for i in range(n)]
    Y = np.array(Y)
    Y = Y.reshape((n,1))
    Y = Y.astype(int)
    return X, Y

def simulatedData2(n = 100, seed = None, noise = 1):
    X = np.random.uniform(0, 15, n)
    X = np.sort(X)
    Y = [7 * np.sin(2*x) * np.exp(-0.1*x) + noise * np.random.normal(0, 1) for x in X]
    Y_true = [7 * np.sin(2*x) * np.exp(-0.1*x) for x in X]
    
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


    
if __name__ == "__main__":
    X, Y = SimulatedDataInteraction(n = 10)
    print(X)
    print(Y)