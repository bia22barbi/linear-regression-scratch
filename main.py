"""Módulo de execução e validação da Regressão Linear do zero."""

import numpy as np
from metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    train_test_split,
)


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
            raise ValueError(
                "O modelo precisa ser treinado com .fit() antes de prever."
            )
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    np.random.seed(42)

    TRUE_W = np.array([[3.5]])
    TRUE_B = 12.0
    M_SAMPLES = 250

    print("--- 1. Gerando Dados Sintéticos com Ruído ---")
    X = np.random.uniform(low=0.0, high=10.0, size=(M_SAMPLES, 1))
    noise = np.random.normal(loc=0.0, scale=1.0, size=(M_SAMPLES, 1))
    y = np.dot(X, TRUE_W) + TRUE_B + noise

    # Particionamento: 80% treino / 20% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    print(f"Dataset total: {X.shape[0]} amostras")
    print(f"Conjunto Treino: {X_train.shape[0]} | Conjunto Teste: {X_test.shape[0]}\n")

    print("--- 2. Treinando o Modelo ---")
    model = LinearRegression(learning_rate=0.02, n_iterations=1000)
    model.fit(X_train, y_train)

    print("\n--- 3. Avaliação no Conjunto de Teste (Dados Inéditos) ---")
    y_test_pred = model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"MSE  (Erro Quadrático Médio): {test_mse:.4f}")
    print(f"RMSE (Raiz do Erro Quadrático): {test_rmse:.4f}")
    print(f"MAE  (Erro Absoluto Médio):     {test_mae:.4f}")
    print(
        f"R²   (Coef. de Determinação):   {test_r2:.4f} ({test_r2 * 100:.2f}% da variância explicada)"
    )
