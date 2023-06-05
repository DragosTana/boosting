import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

import AdaBoost as ab
import misc as ms


    
    
    
def main():
    np.random.seed(42)
    #X, y = ms.simulatedData1(n = 1000)
    X, y = datasets.make_classification(n_samples=10000, n_features=5, n_classes=2)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    test_error = []
    trainig_error = []
    n = 10

    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(x_train, y_train)
    print("Decision tree error: ", 1 - accuracy_score(y_test, tree.predict(x_test)))
    
    
    ada = AdaBoostClassifier(n_estimators = n, learning_rate=0.1)
    ada.fit(x_train, y_train)
    print("AdaBoost error: ", 1 - ada.score(x_test, y_test))
    
    estimators = ada.estimators_
    weights = ada.estimator_weights_
    errors = ada.estimator_errors_
    print("estimator weights: ", weights)
    print("estimator errors: ", errors)
        
    
    
    
    #for i in tqdm.tqdm(range(1, n)):
    #    tree = DecisionTreeClassifier(max_depth=1)
    #    ada = AdaBoostClassifier(n_estimators=i, learning_rate=1.0, estimator=tree)
    #    ada.fit(x_train, y_train)
    #    test_error.append(1 - ada.score(x_test, y_test))
    #    trainig_error.append(1 - ada.score(x_train, y_train))
        
    #plt.plot(range(1, n), test_error, label="Test Error")
    #plt.plot(range(1, n), trainig_error, label="Training Error")
    #plt.legend()
    #plt.show()
main()
    
    