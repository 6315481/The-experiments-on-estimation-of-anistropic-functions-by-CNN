import numpy as np


def generate_AR_poly(n, dim, p):

    X = np.random.uniform(0, 1, size = (n, dim))

    a = []
    for i in range(dim):
        a.append((i+1) ** (-p))
    
    y = np.dot(X, np.array(a))

    return np.reshape(X, (n, 1, dim)), y





