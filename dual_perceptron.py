# dual_perceptron/dual_perceptron.py
from .kernels import get_kernel
from .gram_matrix import compute_gram_matrix
from .params_manager import ModelParams, ParamsManager
import numpy as np
from tqdm import tqdm
import time
import joblib

class DualPerceptron:

    def __init__(self, kernel, epochs=1000, patience=10, time=None, **kernel_kwargs):
        self.kernel_fn = get_kernel(kernel) # select kernel function
        self.kernel_name = kernel
        self.kernel_kwargs = kernel_kwargs  # Parametri specifici del kernel (gamma, d, c, etc.)
        self.epochs = epochs
        self.patience = patience
        self.alpha = None
        self.b = 0
        self.gram_matrix = None
        self.accuracy = None
        self.time = time
        
    def fit(self, X_train, y_train, X_val, y_val, **params):
        r2 = np.round(np.max(np.linalg.norm(X_train, axis=1))**2, 1)
        self.X_train, self.y_train = X_train, y_train
        self.n = len(X_train)
        self.alpha = np.zeros(self.n)
        self.gram_matrix = self._compute_gram_matrix(X_train)
        train_accuracies = []
        validation_accuracies = []
        best_validation_accuracy = -float('inf')
        best_accuracy = -float('inf')
        patience_counter = 0
        previous_accuracy = 0
        start_time = time.time()

        for epoch in tqdm(range(self.epochs), desc=f'Training model with {self.kernel_name} kernel'):
            errors = 0
            for i in range(self.n):
                w = np.sum(self.alpha * self.y_train * self.gram_matrix[i]) + self.b
                if w * y_train[i] <= 0: 
                    self.alpha[i] += 1
                    self.b += y_train[i]*r2
                    errors += 1
            epoch_accuracy = 1-(errors / self.n) # Accuratezza nel train

            # Check for early stopping using patience
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                patience_counter = 0  # Reset patience if accuracy improves
            else:
                patience_counter += 1

            previous_accuracy = epoch_accuracy

            # Prediction on validation set
            self.predict(X_val, y_val)
            validation_accuracy = self.get_accuracy()
            validation_accuracies.append(validation_accuracy)
            train_accuracies.append(epoch_accuracy)

            if validation_accuracy > best_validation_accuracy:
                best_model_params = self.get_params()

            # Stop if the accuracy doesn't improve for `patience` consecutive epochs
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in accuracy.")
                break
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Training time: {elapsed_time} seconds')

        return {'train_accuracies': train_accuracies, 
                'validation_accuracies': validation_accuracies,
                'train_accuracy': max(train_accuracies),
                'validation_accuracy': max(validation_accuracies),
                '_model_params': best_model_params
        }

    def predict(self, X_test, y_test):
        self.predictions = np.array([self._predict_point(x) for x in X_test])
        self.accuracy = np.mean(self.predictions == y_test)
        return self.predictions

    def _predict_point(self, x):
        w = sum(self.alpha[i] * self.y_train[i] * self._kernel(self.X_train[i], x) for i in range(self.n)) + self.b
        return np.sign(w)

    def get_accuracy(self):
        return self.accuracy

    ''' Wrappers '''

    def _kernel(self, x1, x2):
        '''Wrapper for kernel parameters application'''
        return self.kernel_fn(x1, x2, **self.kernel_kwargs)

    def _compute_gram_matrix(self, X):
        '''Wrapper method for the gram matrix computation'''
        return compute_gram_matrix(X, self.kernel_fn, **self.kernel_kwargs)

    def get_params(self):
        '''Expose params through the manager'''
        return ParamsManager.get_params(self)

    def save_params(self, params_dict=None):
        '''Save params through the manager'''
        params = params_dict or self.get_params()
        ParamsManager.save_params(params)