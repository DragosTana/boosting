from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import tqdm

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class AdaBoost(BaseEstimator, ClassifierMixin):
    """
    AdaBoost algorithm for binary classification.
    """

    def __init__(self, n_estimators=50, depth=1, learning_rate=1.0, verbose=False):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []
        self.error_rates = []
        self.depth = depth
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoost":
        y = y.flatten()
        X, y = check_X_y(X, y)
        sample_weights = np.ones(len(X)) / len(X)
        
        iterator = range(self.n_estimators)
        if self.verbose:
            iterator = tqdm.tqdm(iterator)

        for _ in iterator:
            estimator = DecisionTreeClassifier(max_depth=self.depth)
            estimator.fit(X, y, sample_weight=sample_weights)
            y_pred = estimator.predict(X)
            error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)
            estimator_weight = self.learning_rate * np.log((1 - error) / error)
            sample_weights *= np.exp(estimator_weight * (y_pred != y))
            sample_weights /= np.sum(sample_weights)
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
            self.error_rates.append(error)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        y_pred = np.zeros(len(X))
        for estimator, estimator_weight in zip(self.estimators, self.estimator_weights):
            y_pred += estimator_weight * estimator.predict(X)

        y_pred = np.sign(y_pred)

        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        check_is_fitted(self)
        y = y.reshape((-1, 1))
        X = check_array(X)
        y = check_array(y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y.flatten())
        return accuracy



