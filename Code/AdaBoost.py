from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import tqdm

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []
        self.error_rates = []

    def fit(self, X, y):
        sample_weights = np.ones(len(X)) / len(X)
        y = y.flatten()
        
        for _ in tqdm.tqdm(range(self.n_estimators)):
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=sample_weights)
            y_pred = estimator.predict(X)
            error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)
            estimator_weight = np.log((1 - error) / error)
            sample_weights *= np.exp(estimator_weight * (y_pred != y))
            sample_weights /= np.sum(sample_weights)
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            self.error_rates.append(error)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for estimator, estimator_weight in zip(self.estimators, self.estimator_weights):
            y_pred += estimator_weight * estimator.predict(X)

        y_pred = np.sign(y_pred)

        return y_pred




