# dual_perceptron/gram_matrix.py
import numpy as np

def compute_gram_matrix(X, kernel_fn, **kernel_kwargs):
    '''
    Computes the Gram matrix for a given kernel function
    
    Parameters:
    X : ndarray of shape (n_samples, n_features)
        Input data matrix
    kernel_fn : callable
        Kernel function (e.g., linear_kernel, rbf_kernel)
    **kernel_kwargs : dict
        Additional keyword arguments for the kernel function
        
    Returns:
    gram : ndarray of shape (n_samples, n_samples)
        Computed Gram matrix
    '''
    n = X.shape[0]
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gram[i, j] = kernel_fn(X[i], X[j], **kernel_kwargs)
    return gram