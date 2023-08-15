import numpy as np

#pass in the svd of the cov matrix, important because the svd is only calculated once 
def multivariate_normal(mean,A):

    x = (np.random.standard_normal(mean.shape))
    x = np.dot(x, A)
    x += mean
    x.shape = tuple(mean.shape)

    return x



