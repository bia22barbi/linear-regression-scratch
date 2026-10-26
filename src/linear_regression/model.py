"""Implementação do estimador LinearRegression."""

import numpy as np


class LinearRegression:
    """Regressão Linear implementada com NumPy puro."""

    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000) -> None:
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina o modelo usando Batch Gradient Descent."""
        m_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            residuals = y_pred - y
            mse = (1 / (2 * m_samples)) * np.sum(residuals**2)
            self.loss_history.append(float(mse))

            dw = (1 / m_samples) * np.dot(X.T, residuals)
            db = (1 / m_samples) * np.sum(residuals)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 200 == 0 or epoch == self.n_iterations - 1:
                print(f"Época {epoch:04d} | Custo J(w,b): {mse:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calcula inferências lineares para a matriz de entrada."""
        if self.weights is None:
            raise ValueError("O modelo precisa ser treinado com .fit() antes de prever.")
        return np.dot(X, self.weights) + self.bias
