import numpy as np
from scipy.linalg import svd
from copy import deepcopy
from collections import Counter
from pdb import set_trace

def pca(X, n_components = 5):
    #  Use svd to perform PCA on X
    #  Inputs:
    #     X: input matrix
    #     n_components: number of principal components to keep
    #  Output:
    #     principal_components: the top n_components principal_components
    #     X_pca = X.dot(principal_components)

    U, s, Vh = svd(X)
    col = np.min((n_components,len(s)))
    principal_components = Vh.transpose()[:,:col]
    return principal_components

def vector_norm(x, norm="Min-Max"):
    # Calculate the normalized vector
    if norm == "Min-Max":
        upper = np.max(x)
        lower = np.min(x)
        x_norm = (x - lower) / (upper - lower)
    elif norm == "L1":
        x_norm = x / np.sum(np.abs(x))
    elif norm == "L2":
        x_norm = x / np.sqrt(np.sum(x**2))
    elif norm == "Standard_Score":
        x_norm = (x - np.mean(x)) / np.std(x)
    else:
        raise Exception("Unknown normlization.")
    return x_norm

def normalize(X, norm="Min-Max", axis = 1):
    #  Inputs:
    #     X: input matrix
    #     norm = {"L1", "L2", "Min-Max", "Standard_Score"}
    #     axis = 0: normalize rows
    #     axis = 1: normalize columns
    #  Output:
    #     X_norm: normalized matrix (numpy.array)

    X_norm = deepcopy(np.asarray(X))
    m, n = X_norm.shape
    if axis == 1:
        for col in range(n):
            X_norm[:,col] = vector_norm(X_norm[:,col], norm=norm)
    elif axis == 0:
        X_norm = np.array([vector_norm(X_norm[i], norm=norm) for i in range(m)])
    else:
        raise Exception("Unknown axis.")
    return X_norm

def stratified_sampling(y, ratio, replace = True):
    #  Inputs:
    #     y: class labels
    #     0 < ratio < 1: number of samples = len(y) * ratio
    #     replace = True: sample with replacement
    #     replace = False: sample without replacement
    #  Output:
    #     sample: indices of stratified sampled points
    #             (ratio is the same across each class, 
    #             samples for each class = int(np.ceil(ratio * # data in each class)) )

    if ratio<=0 or ratio>=1:
        raise Exception("ratio must be 0 < ratio < 1.")
    y_array = np.asarray(y)
    stats_y = Counter(y_array)
    sample = []
    for key in stats_y:
        sample = np.append(sample,np.random.choice(np.where(y_array == key)[0], int(np.ceil(ratio*stats_y[key])), replace = replace))
    return sample.astype(int)
