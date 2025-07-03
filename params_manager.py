# dual_perceptron/params_manager.py
import joblib
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class ModelParams:
    alpha: np.ndarray
    bias: float
    kernel: str
    X_train: np.ndarray
    y_train: np.ndarray
    kernel_params: Dict[str, Any]

class ParamsManager:
    @staticmethod
    def get_params(model) -> ModelParams:
        """Extracts and structures all model parameters"""
        kernel_params = model.kernel_kwargs

        return ModelParams(
            alpha=model.alpha,
            bias=model.b,
            kernel=model.kernel_name,
            X_train=model.X_train,
            y_train=model.y_train,
            kernel_params=kernel_params
        )

    @staticmethod
    def save_params(params: ModelParams, filename: str = None):
        """Saves parameters to disk"""
        if not filename:
            filename = f"model_params_{params.kernel_name}.pkl"
        joblib.dump(params, filename)