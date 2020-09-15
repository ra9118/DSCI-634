import pandas as pd
import numpy as np
from collections import Counter
from pdb import set_trace

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        key_values = {key: set(X[key]) for key in X}
        self.P_y = Counter(y)
        self.P = {}
        for label in self.classes_:
            indices = np.where(np.array(y)==label)[0]
            self.P[label] = {}
            for key in X:
                stats = Counter(X[key][indices])
                self.P[label][key] = {value: (stats[value]+self.alpha)/(self.P_y[label]+self.alpha*len(key_values[key])) for value in key_values[key]}
        return

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # write your code below
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # write your code below
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions





