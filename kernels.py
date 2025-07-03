# dual_perceptron/kernels.py
import numpy as np

def linear_kernel(x1, x2, **kwargs):
    """Linear kernel: <x1, x2>"""
    return np.dot(x1, x2)

def poly_kernel(x1, x2, d=2, c=1, **kwargs):
    """Polynomial kernel: (<x1, x2> + c)^d"""
    return (np.dot(x1, x2) + c) ** d

def rbf_kernel(x1, x2, gamma=0.5, **kwargs):
    """RBF/Gaussian kernel: exp(-gamma * ||x1-x2||^2)"""
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
   
KERNELS = {
    'linear': linear_kernel,
    'poly': poly_kernel,
    'rbf': rbf_kernel
}

def get_kernel(kernel_name):
    """Factory function to retrieve kernels"""
    if kernel_name not in KERNELS:
        raise ValueError(f"Unsupported kernel: {kernel_name}. Choose from {list(KERNELS.keys())}")
    return KERNELS[kernel_name]