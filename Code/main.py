import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import AdaBoost as ab
import misc as ms

def main():
    np.random.seed(42)
    X, y = ms.simulatedData1(n = 1000)
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    test_error = []
    trainig_error = []
    n = 500
    for i in tqdm.tqdm(range(1, n)):
        ada = ab.AdaBoost(n_estimators=i, depth=1, learning_rate=1.0)
        ada.fit(x_train, y_train)
        y_pred = ada.predict(x_test)
        test_error.append(1 - accuracy_score(y_test, y_pred))
        y_pred = ada.predict(x_train)
        trainig_error.append(1 - accuracy_score(y_train, y_pred))
        
    plt.plot(range(1, n), test_error, label="Test Error")
    plt.plot(range(1, n), trainig_error, label="Training Error")
    plt.plot(range(1, n), ada.error_rates, label="Error")
    plt.legend()
    plt.show()
        
    
    
main()
    
    