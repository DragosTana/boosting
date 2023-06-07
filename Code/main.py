from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import time

import AdaBoost as ab
import GradientBoosting as gb
import misc as ms

#TODO: run compareGB() 
    
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
        
    for i in range(1, n):


        y_pred = np.sign(y_pred)

        
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

def interactionTest():
    """
    Function to test the effect of the deapth of the trees on the performance of the Gradient Boosting algorithm in 
    the case of interaction terms.
    """
   
    sim = 100
    score2, score3, score4 = [], [], []
   
    for _ in tqdm.tqdm(range(sim)):
        X, Y = ms.simulatedDataInteraction(n = 1000, interaction = 4, noise = 5)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        gb2_grid = GridSearchCV(estimator=gb.GradientBoostingRegressor(max_depth=2), param_grid={'learning_rate': np.arange(0.001, 0.1, 0.01), "n_estimators": range(10, 200, 5)}, cv=5, n_jobs=-1)
        #gb3_grid = GridSearchCV(estimator=gb.GradientBoostingRegressor(max_depth=3), param_grid={'learning_rate': np.arange(0.001, 0.1, 0.01), "n_estimators": range(10, 200, 5)}, cv=5, n_jobs=-1)
        gb4_grid = GridSearchCV(estimator=gb.GradientBoostingRegressor(max_depth=4), param_grid={'learning_rate': np.arange(0.001, 0.1, 0.01), "n_estimators": range(10, 200, 5)}, cv=5, n_jobs=-1)
        
        gb2_grid.fit(x_train, y_train)
        #gb3_grid.fit(x_train, y_train)
        gb4_grid.fit(x_train, y_train)
        
        gb2_grid.best_estimator_.fit(x_train, y_train)
        #gb3_grid.best_estimator_.fit(x_train, y_train)
        gb4_grid.best_estimator_.fit(x_train, y_train)
        
        score2.append(gb2_grid.best_estimator_.score(x_test, y_test))
        #score3.append(gb3_grid.best_estimator_.score(x_test, y_test))
        score4.append(gb4_grid.best_estimator_.score(x_test, y_test))
        
    print("Score 2: ", np.mean(score2))
    #print("Score 3: ", np.mean(score3))
    print("Score 4: ", np.mean(score4))
              
def compareGB():
    """
    Function to compare my implementation of the Gradient Boosting algorithm with the one from sklearn.
    """
    n_samples = [100, 500, 1000, 5000, 10000]
    
    score_my, score_sk = [], []
    time_my, time_sk = [], []
    for n in n_samples:    
        tmp_my, tmp_sk = [], []
        tmp_t_my, tmp_t_sk = [], []
        
        for i in tqdm.tqdm(range(100)):
            X, Y = datasets.make_regression(n_samples = n, n_features=10, n_informative = 6, noise=5)
            #X, Y = ms.SimulatedDataInteraction(n = n, interaction = 3, noise = 5)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

            my_gb = gb.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, loss='ls', verbose = False)
            sk_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, loss='squared_error', verbose = False)

            start_time = time.time()
            my_gb.fit(x_train, y_train)
            end_time = time.time()
            tmp_t_my.append(end_time - start_time)

            start_time = time.time()
            sk_gb.fit(x_train, y_train)
            end_time = time.time()
            tmp_t_sk.append(end_time - start_time)


            tmp_my.append(my_gb.score(x_test, y_test))
            tmp_sk.append(sk_gb.score(x_test, y_test))
        
        score_my.append(np.mean(tmp_my))
        score_sk.append(np.mean(tmp_sk))
        time_my.append(np.mean(tmp_t_my))
        time_sk.append(np.mean(tmp_t_sk))
        
    plt.plot(n_samples, score_my, label="My implementation")
    plt.plot(n_samples, score_sk, label="Sklearn implementation")
    plt.legend()
    plt.show()
    
    plt.plot(n_samples, time_my, label="My implementation")
    plt.plot(n_samples, time_sk, label="Sklearn implementation")
    plt.legend()
    plt.show()
    
def learningVisualization():
    """
    Function to visualize the learning process of the Gradient Boosting algorithm.
    """
    estimators = 200
    size = 500
    X, Y, Y_true = ms.simulatedData2(n = size, seed = None, noise=2)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((size, 1))
    Y = Y.reshape((size, 1))
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    values = np.linspace(0, 15, 1000)
    values = values.reshape((1000, 1))
    print(values)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(X, Y_true, label="True function")
    ax[0].plot(x_train, y_train, '.', label="Training Data", alpha=0.5)
    ax[1].set_xlim(0, estimators)
    ax[1].set_ylim(0, 1)
    
    training_error = []
    test_error = []
    
    plt.pause(5)
    for i in range(1, estimators):
        np.random.seed(1)
        mygb = GradientBoostingRegressor(n_estimators=i, learning_rate=0.1, max_depth=3, loss='squared_error', verbose = False)
        mygb.fit(x_train, y_train)
        training_error.append(mygb.score(x_train, y_train))
        test_error.append(mygb.score(x_test, y_test))
        
        line = ax[0].plot(values, mygb.predict(values), color='r', label="Prediction")
        test = ax[1].plot(range(0, i), test_error, color='r', label="Test error")
        train = ax[1].plot(range(0, i), training_error, color='g', label="Training error")
        ax[1].legend()
        ax[0].legend()
        
        plt.pause(0.5)
        for l in line:
            l.remove()
        for t in test:
            t.remove()
        for t in train:
            t.remove()
            
    plt.show()
       
def learningVisualization2():
    """
    Function to visualize the learning process of the Gradient Boosting algorithm.
    """
    estimators = 200
    size = 1000
    X, Y, Y_true = ms.simulatedData2(n = size, seed = None, noise=1)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape((size, 1))
    Y = Y.reshape((size, 1))
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    values = np.linspace(0, 15, 1000)
    values = values.reshape((1000, 1))
    print(values)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(X, Y_true, label="True function")
    ax.plot(x_train, y_train, '.', label="Training Data", alpha=0.5)
    
    plt.pause(5)
    for i in range(1, estimators):
        np.random.seed(1)

        mygb = GradientBoostingRegressor(n_estimators=i, learning_rate=0.1, max_depth=3, loss='squared_error', verbose = False)
        mygb.fit(x_train, y_train)

        line = ax.plot(values, mygb.predict(values), color='r', label="Prediction")

        ax.legend()
        plt.pause(0.3)
        for l in line:
            l.remove()
            
    plt.show()
    
interactionTest()
        
        
    
    


    
    