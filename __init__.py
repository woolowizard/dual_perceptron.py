from .dual_perceptron import DualPerceptron
from .kernels import get_kernel
from .gram_matrix import compute_gram_matrix
from .params_manager import ModelParams, ParamsManager

__version__ = '1.0.0'
__author__ = 'wolowizard'
__all__ = ["get_kernel", "compute_gram_matrix", "ModelParams", "ParamsManager", "DualPerceptron"]