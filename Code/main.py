from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import time

import AdaBoost as ab
import GradientBoosting as gb
import misc as ms
    
def main() -> None:
    montecarlo()
    
def montecarlo():
    sim = 100
    tree = 501
    shrink_error, no_shrink_error = [], []
    
    for i in tqdm.tqdm(range(sim)):
        score_shrink, score_no_shrink = shrikageTest(trees=tree, shrinkage=0.1, n_samples=1000)
        shrink_error.append(score_shrink)
        no_shrink_error.append(score_no_shrink)
        
    shrink_error = np.array(shrink_error)
    no_shrink_error = np.array(no_shrink_error)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(1, tree), np.mean(shrink_error, axis=0), label="Shrinkage at 0.1", linewidth=2)
    ax.plot(range(1, tree), np.mean(no_shrink_error, axis=0), label="No shrinkage", linewidth=2)
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Test Error")
    plt.legend()
    plt.show()
    
        

     
def compareHyperparameter(trees: int = 1001, subsampling: float = 0.5, shrinkage: float = 0.1, n_samples: int = 500)-> None:
    """
    Function to compare the effect of the shrinkage and subsampling parameters on the performance of the 
    Gradient Boosting algorithm both in the case of classification and regression.
    """
    score_noSub_noShrink, score_noSub_shrink, score_sub_noShrink, score_sub_shrink = [], [], [], []
    
    X, Y = datasets.make_classification(n_samples = n_samples, n_features=10, n_informative = 6)
    #X, Y = datasets.make_regression(n_samples = n_samples, n_features=10, n_informative = 6, noise=5)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    gb_noSub_noShrink = GradientBoostingClassifier(n_estimators=trees, learning_rate=1, max_depth=3, loss='exponential', verbose = False)
    gb_noSub_shrink = GradientBoostingClassifier(n_estimators=trees, learning_rate=shrinkage, max_depth=3, loss='exponential', verbose = False, subsample=subsampling)
    gb_sub_noShrink = GradientBoostingClassifier(n_estimators=trees, learning_rate=1, max_depth=3, loss='exponential', verbose = False)
    gb_sub_shrink = GradientBoostingClassifier(n_estimators=trees, learning_rate=shrinkage, max_depth=3, loss='exponential', verbose = False, subsample=subsampling)
    
    #gb_noSub_noShrink = GradientBoostingRegressor(n_estimators=trees, learning_rate=1, max_depth=3, loss='least_square', verbose = False)
    #gb_noSub_shrink = GradientBoostingRegressor(n_estimators=trees, learning_rate=shrinkage, max_depth=3, loss='least_square', verbose = False, subsample=subsampling)
    #gb_sub_noShrink = GradientBoostingRegressor(n_estimators=trees, learning_rate=1, max_depth=3, loss='least_square', verbose = False)
    #gb_sub_shrink = GradientBoostingRegressor(n_estimators=trees, learning_rate=shrinkage, max_depth=3, loss='least_square', verbose = False, subsample=subsampling)
    
    gb_noSub_noShrink.fit(x_train, y_train), gb_noSub_shrink.fit(x_train, y_train), gb_sub_noShrink.fit(x_train, y_train), gb_sub_shrink.fit(x_train, y_train)
    
    for i in range(1, trees):
        pred_noSub_noShrink = np.array(list(gb_noSub_noShrink.staged_predict(x_test))[i])
        pred_noSub_shrink = np.array(list(gb_noSub_shrink.staged_predict(x_test))[i])
        pred_sub_noShrink = np.array(list(gb_sub_noShrink.staged_predict(x_test))[i])
        pred_sub_shrink = np.array(list(gb_sub_shrink.staged_predict(x_test))[i])
        
        score_noSub_noShrink.append(1 - accuracy_score(y_test, pred_noSub_noShrink))
        score_noSub_shrink.append(1 - accuracy_score(y_test, pred_noSub_shrink))
        score_sub_noShrink.append(1 - accuracy_score(y_test, pred_sub_noShrink))
        score_sub_shrink.append(1 - accuracy_score(y_test, pred_sub_shrink))
        
        #score_noSub_noShrink.append(np.mean((y_test - pred_noSub_noShrink)**2))
        #score_noSub_shrink.append(np.mean((y_test - pred_noSub_shrink)**2))
        #score_sub_noShrink.append(np.mean((y_test - pred_sub_noShrink)**2))
        #score_sub_shrink.append(np.mean((y_test - pred_sub_shrink)**2))
        
    #fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #ax.plot(range(1, trees), score_noSub_noShrink, label="No subsampling, no shrinkage", linewidth=2)
    #ax.plot(range(1, trees), score_noSub_shrink, label="No subsampling, shrinkage", linewidth=2)
    #ax.plot(range(1, trees), score_sub_noShrink, label="Subsampling, no shrinkage", linewidth=2)
    #ax.plot(range(1, trees), score_sub_shrink, label="Subsampling, shrinkage", linewidth=2)
    #
    #plt.xlabel("Boosting Iterations")
    #plt.ylabel("Test Error")
    #plt.legend()
    #plt.show()
    
    return score_noSub_noShrink, score_noSub_shrink, score_sub_noShrink, score_sub_shrink
    
def SubsamplingTest(trees: int = 1001, subsampling: float = 0.5, n_samples: int = 1000) -> None:
    """
    Function to test the effect of the subsampling parameter on the performance of the Gradient Boosting algorithm
    """
    t = trees
    time_sub, time_no_sub = [], []
    
    #X, Y = datasets.make_classification(n_samples = n_samples, n_features=10, n_informative = 6)
    X, Y = ms.simulatedData1(n = n_samples, seed = None)
    Y = Y.flatten()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    start = time.time()
    gb_sub = GradientBoostingClassifier(n_estimators=t, learning_rate=0.1, max_depth=3, loss='exponential', verbose = False, subsample = subsampling)
    end = time.time()
    time_sub.append(end - start)
    start = time.time()
    gb_no_sub = GradientBoostingClassifier(n_estimators=t, learning_rate=0.1, max_depth=3, loss='exponential', verbose = False)
    end = time.time()
    time_no_sub.append(end - start)
    
    gb_sub.fit(x_train, y_train), gb_no_sub.fit(x_train, y_train)
    
    score_sub, score_no_sub = [], []
    
    
    for i in range(1, t):
        pred_sub = np.array(list(gb_sub.staged_predict(x_test))[i])
        pred_no_sub = np.array(list(gb_no_sub.staged_predict(x_test))[i])
        
        score_sub.append(1 - accuracy_score(y_test, pred_sub))
        score_no_sub.append(1 - accuracy_score(y_test, pred_no_sub))
    
    #fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #ax.plot(range(1, trees), score_sub, label="Subsampling at " + str(subsampling), linewidth=2)
    #ax.plot(range(1, trees), score_no_sub, label="No subsampling", linewidth=2)
    #plt.xlabel("Boosting Iterations")
    #plt.ylabel("Test Error")
    #plt.legend()
    #plt.show()
        
    return score_sub, score_no_sub, time_sub, time_no_sub
        
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
    
def Test1():
    """
    To test the hypothesis that the Gradient Boosting algorithm performs with higher max_depth
    fixing all the other parameters.
    """
    
    sim = 100
    score1, score2, score3, score4, score5 = [], [], [], [], []
    
    for _ in tqdm.tqdm(range(sim)):
        #X, Y = ms.simulatedDataInteraction(n = 1000, interaction = 4, noise = 5)
        X, Y = datasets.make_regression(n_samples = 1000, n_features=10, n_informative = 6, noise=5)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        gb1 = GradientBoostingRegressor(max_depth=1)
        gb2 = GradientBoostingRegressor(max_depth=2)
        gb3 = GradientBoostingRegressor(max_depth=3)
        gb4 = GradientBoostingRegressor(max_depth=4)
        gb5 = GradientBoostingRegressor(max_depth=5)
        
        gb1.fit(x_train, y_train)
        gb2.fit(x_train, y_train)
        gb3.fit(x_train, y_train)
        gb4.fit(x_train, y_train)
        gb5.fit(x_train, y_train)
        
        score1.append(gb1.score(x_test, y_test))
        score2.append(gb2.score(x_test, y_test))
        score3.append(gb3.score(x_test, y_test))
        score4.append(gb4.score(x_test, y_test))
        score5.append(gb5.score(x_test, y_test))
        
    print("Test score with max tree depth 1: ", np.mean(score1))
    print("Test score with max tree depth 2: ", np.mean(score2))
    print("Test score with max tree depth 3: ", np.mean(score3))
    print("Test score with max tree depth 4: ", np.mean(score4))
    print("Test score with max tree depth 5: ", np.mean(score5))
    
def MSEandNumberofTrees():
    """
    Function to test the effect of the number of trees on the performance of the Gradient Boosting algorithm.
    """   
    trees = 1001
    mse_train, mse_test = [], []
    
    mygb = GradientBoostingClassifier(n_estimators=1001, learning_rate=0.1, max_depth=3, loss='exponential', verbose = False)
    X, Y = datasets.make_classification(n_samples = 1000, n_features=10, n_informative = 6)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    mygb.fit(x_train, y_train)
    
    for i in tqdm.tqdm(range(1, trees)):
        y_pred_train = np.array(list(mygb.staged_predict(x_train))[i])
        y_pred_test = np.array(list(mygb.staged_predict(x_test))[i])
        
        mse_test.append(1 - accuracy_score(y_test, y_pred_test))
        mse_train.append(1 - accuracy_score(y_train, y_pred_train))
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(1, trees), mse_train, label="Training error", linewidth=3)
    ax.plot(range(1, trees), mse_test, label="Test error", linewidth=3)
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Test Error")
    plt.legend()
    plt.show()
             
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
        
    plt.plot(n_samples, score_my, label="My implementation", linewidth=3)
    plt.plot(n_samples, score_sk, label="Sklearn implementation", linewidth=3)
    plt.ylabel("R2 score")
    plt.xlabel("Number of samples")
    plt.legend()
    plt.show()
    
    plt.plot(n_samples, time_my, label="My implementation", linewidth=3)
    plt.plot(n_samples, time_sk, label="Sklearn implementation", linewidth=3)
    plt.ylabel("Time in seconds")
    plt.xlabel("Number of samples")
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
    
def compareLoss():
    """
    Function to compare the difference between the different loss functions
    when dealing with outliers.
    """
    size = 500
    x, y, Y_true = ms.simulatedData3(n = size, seed = None, noise=5)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x, Y_true, label="True function")
    ax.plot(x, y, '.', label="Training Data", alpha=0.5)
    ax.plot([0, -7.5, 7.5], [40, 20, 20], "o", color = "r", label="Outlier")
    ax.legend()
    
    X = np.append(x, [0, -7.5, 7.5])
    Y = np.append(y, [40, 20, 20])
    
    values = np.linspace(-10, 10, 1000)
    
    X = X.reshape((size + 3, 1))
    Y = Y.reshape((size + 3, 1))
    values = values.reshape((1000, 1))
    
    #plt.pause(5)
    for i in tqdm.tqdm(range(1, 200)):
        np.random.seed(1)
        
        #NOTE: change the loss function here to test the different ones
        mygb = GradientBoostingRegressor(n_estimators=i, learning_rate=0.1, max_depth=3, loss='huber', verbose = False)
        mygb.fit(X, Y)
        
        line = ax.plot(values, mygb.predict(values), color='r', label="Prediction")
        plt.pause(0.01)
        for l in line:
            l.remove()
    
    line = ax.plot(values, mygb.predict(values), color='r', label="Prediction")
    plt.show()
    
    
def shrikageTest(trees: int = 1001, shrinkage: float = 0.1, n_samples: int = 1000) -> None:
    """
    Function to test the effect of the shrinkage parameter on the performance of the Gradient Boosting algorithm
    and the trade-off between the number of trees and the shrinkage parameter.
    """
    t = trees
    shrink_error, no_shrink_error = [], []
    
    #X, Y = datasets.make_classification(n_samples = n_samples, n_features=10, n_informative = 6)
    X, Y = ms.simulatedData1(n = n_samples, seed = None)
    Y = Y.flatten()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    gb_shrink = GradientBoostingClassifier(n_estimators=t, learning_rate=shrinkage, max_depth=3, loss='exponential', verbose = False)
    gb_no_shrink = GradientBoostingClassifier(n_estimators=t, learning_rate=1, max_depth=3, loss='exponential', verbose = False)
    
    gb_shrink.fit(x_train, y_train), gb_no_shrink.fit(x_train, y_train)
    
    for i in range(1, t):
        y_pred_shrink = np.array(list(gb_shrink.staged_predict(x_test))[i])
        y_pred_no_shrink = np.array(list(gb_no_shrink.staged_predict(x_test))[i])
        
        shrink_error.append(1 - accuracy_score(y_test, y_pred_shrink))
        no_shrink_error.append(1 - accuracy_score(y_test, y_pred_no_shrink))
        
    #fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #ax.plot(range(1, trees), shrink_error, label="Shrinkage at 0.1", linewidth=2)
    #ax.plot(range(1, trees), no_shrink_error, label="No shrinkage", linewidth=2)
    #ax.set_xlabel("Boosting Iterations")
    #ax.set_ylabel("Test Error")
    #ax.legend()
    #
    #plt.show()
    
    return shrink_error, no_shrink_error
    
main()


    
    