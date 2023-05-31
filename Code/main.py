import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import AdaBoost as ab
import misc as ms

def main():
    X, Y = ms.simulatedData1(10000)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(x_train, y_train)
    tree_pred = tree.predict(x_test)

    error_tree = 1 - accuracy_score(y_test, tree_pred)
    
    ada = ab.AdaBoost(n_estimators=500)
    ada.fit(x_train, y_train)
    ada_pred = ada.predict(x_test)
    ada_error = 1 - accuracy_score(y_test, ada_pred)
    print("Error rate of a single decision tree: %.4f" % error_tree)
    print("Error rate of AdaBoost: %.4f" % ada_error)
    
    
    
    
    
main()
    
    