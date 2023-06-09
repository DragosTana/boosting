import numpy as np
import tqdm
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting for regression.

    This estimator builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. 
    In each stage a regression tree is fit on the negative gradient of the given loss function.
    
    ## Parameters
    
    n_estimators : int, default=100
    The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
    
    learning_rate : float, default=0.1
    The learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
    
    max_depth : int, default=3
    The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
    
    loss : {‘ls’, ‘lad’, ‘huber’}, default=’ls’
    loss function to be optimized. ‘ls’ refers to least squares regression. ‘lad’ (least absolute deviation) is a highly robust loss function solely based on order information of the input variables. ‘huber’ is a combination of the two.
    
    verbose : bool, default=False
    Allows to print the progress of the boosting process.
    
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss='ls', verbose = False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.estimators = []
        self.verbose = verbose
        
    def fit(self, X: np.array, y: np.array) -> "GradientBoostingRegressor":
        self.estimators = []
        self.init_prediction = np.mean(y)
        #self.prediction = self.init_prediction * np.ones(len(X))
        
        iterator = range(self.n_estimators)
        if self.verbose:
            iterator = tqdm.tqdm(iterator)
        
        for _ in iterator:
            residuals = self._compute_residuals(y, self.predict(X))
            estimator = DecisionTreeRegressor(max_depth=self.max_depth)
            estimator.fit(X, residuals)
            self.estimators.append(estimator)
            #self.prediction += self.learning_rate * estimator.predict(X)
        
        return self

    def predict(self, X: np.array) -> np.array:
        prediction = self.init_prediction * np.ones(len(X))
        for estimator in self.estimators:
            prediction += self.learning_rate * estimator.predict(X) 
        
        return prediction

    def _compute_residuals(self, y: np.array, prediction: np.array) -> np.array:
        if self.loss == 'ls':  # Least Squares Loss
            return y - prediction
        elif self.loss == 'lad':  # Least Absolute Deviation Loss
            return np.sign(y - prediction)
        elif self.loss == 'huber':  # Huber Loss
            delta = 1.0
            residuals = y - prediction
            mask = np.abs(residuals) <= delta
            residuals[mask] *= 1.0
            residuals[~mask] *= delta
            return residuals
        else:
            raise ValueError("Invalid loss function. Choose 'ls', 'lad', or 'huber'.")
        
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient Boosting for classification.

    This estimator builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. 
    In each stage a regression tree is fit on the negative gradient of the given loss function.
    
    ## Parameters
    
    n_estimators : int, default=100
    The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
    
    learning_rate : float, default=0.1
    The learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
    
    max_depth : int, default=3
    The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
    
    loss : {‘exponential’, ‘logistic’}, default=’exponential’
    Loss function to be optimized. ‘exponential’ refers to the AdaBoost exponential loss function. ‘logistic’ refers to the logistic regression loss function.
    
    verbose : bool, default=False
    Allows to print the progress of the boosting process.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, loss_function='exponential', verbose = False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss_function = loss_function
        self.estimators = []
        self.verbose = verbose
        
    def fit(self, X, y):
        self.estimators = []
        self.init_prediction = np.mean(y)
        
        iterator = range(self.n_estimators)
        if self.verbose:
            iterator = tqdm.tqdm(iterator)
        
        for _ in iterator:
            residuals = self._compute_residuals(y, self.predict(X))
            estimator = DecisionTreeClassifier(max_depth=self.max_depth)
            estimator.fit(X, residuals)
            
            self.estimators.append(estimator)
            #self.init_prediction += self.learning_rate * estimator.predict(X)
        
        return self

    def predict(self, X):
        prediction = self.init_prediction * np.ones(len(X))
        for estimator in self.estimators:
            prediction += self.learning_rate * estimator.predict(X)
        
        return np.sign(prediction)

    def predict_proba(self, X):
        prediction = self.init_prediction.copy()
        for estimator in self.estimators:
            prediction += self.learning_rate * estimator.predict(X)
            
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        proba = sigmoid(prediction)
        
        return np.column_stack((1 - proba, proba))

    def _compute_residuals(self, y, prediction):
        if self.loss_function == 'exponential':
            return y * np.exp(-y * prediction)
        elif self.loss_function == 'logistic':
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            return y - sigmoid(prediction)
        else:
            raise ValueError("Invalid loss function. Choose 'exponential' or 'logistic'.")
