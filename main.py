import numpy as np


class LinearRegression:
    """Regressão Linear implementada do zero utilizando NumPy.

    Attributes:
        learning_rate (float): Taxa de aprendizado para o gradiente descendente.
        n_iterations (int): Número de épocas/iterações do treino.
        weights (np.ndarray): Vetor de pesos (w).
        bias (float): Termo independente / intercepto (b).
        loss_history (list[float]): Registro do MSE ao longo das épocas.
    """

    def __init__(self, learning_rate: float = 0.05, n_iterations: int = 1000) -> None:
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Treina o modelo usando Batch Gradient Descent.

        Args:
            X: Matriz de features com shape (m, n).
            y: Vetor de labels com shape (m, 1).
        """
        m_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_iterations):
            # 1. Hipótese Vetorizada: y_hat = Xw + b
            y_pred = np.dot(X, self.weights) + self.bias

            # 2. Função de Custo (MSE): J = (1 / 2m) * sum((y_hat - y)^2)
            residuals = y_pred - y
            mse = (1 / (2 * m_samples)) * np.sum(residuals**2)
            self.loss_history.append(float(mse))

            # 3. Gradientes: dJ/dw e dJ/db
            dw = (1 / m_samples) * np.dot(X.T, residuals)
            db = (1 / m_samples) * np.sum(residuals)

            # 4. Atualização de Parâmetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 100 == 0 or epoch == self.n_iterations - 1:
                print(f"Época {epoch:04d} | MSE: {mse:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calcula inferências para novos dados."""
        if self.weights is None:
            raise ValueError(
                "O modelo precisa ser treinado com .fit() antes de prever."
            )
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    np.random.seed(42)

    # Parâmetros reais (Ground Truth)
    TRUE_W = np.array([[3.5]])
    TRUE_B = 12.0
    M_SAMPLES = 200

    print("--- 1. Gerando Dados Sintéticos ---")
    # Features X distribuídas uniformemente entre 0 e 10
    X = np.random.uniform(low=0.0, high=10.0, size=(M_SAMPLES, 1))

    # Ruído Gaussiano: N(0, 1)
    noise = np.random.normal(loc=0.0, scale=1.0, size=(M_SAMPLES, 1))

    # y = Xw + b + ruído
    y = np.dot(X, TRUE_W) + TRUE_B + noise

    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"Valores reais esperados -> w: {TRUE_W[0][0]:.4f} | b: {TRUE_B:.4f}\n")

    print("--- 2. Treinando o Modelo ---")
    model = LinearRegression(learning_rate=0.02, n_iterations=1000)
    model.fit(X, y)

    learned_w = model.weights[0][0]
    learned_b = model.bias

    print("\n--- 3. Validação e Convergência ---")
    print(f"Peso real (w):    {TRUE_W[0][0]:.4f} | Peso aprendido:    {learned_w:.4f}")
    print(f"Bias real (b):    {TRUE_B:.4f} | Bias aprendido:    {learned_b:.4f}")
    print(f"MSE Inicial:      {model.loss_history[0]:.6f}")
    print(f"MSE Final:        {model.loss_history[-1]:.6f}")
