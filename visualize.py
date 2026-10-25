import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate: float = 0.02, n_iterations: int = 1000) -> None:
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        m_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            residuals = y_pred - y
            mse = (1 / (2 * m_samples)) * np.sum(residuals**2)
            self.loss_history.append(float(mse))

            dw = (1 / m_samples) * np.dot(X.T, residuals)
            db = (1 / m_samples) * np.sum(residuals)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


if __name__ == "__main__":
    np.random.seed(42)

    TRUE_W = 3.5
    TRUE_B = 12.0
    M_SAMPLES = 200

    X = np.random.uniform(0.0, 10.0, size=(M_SAMPLES, 1))
    noise = np.random.normal(0.0, 1.0, size=(M_SAMPLES, 1))
    y = X * TRUE_W + TRUE_B + noise

    model = LinearRegression(learning_rate=0.02, n_iterations=1000)
    model.fit(X, y)

    x_line = np.linspace(0, 10, 100).reshape(-1, 1)
    y_pred_line = model.predict(x_line)
    y_true_line = x_line * TRUE_W + TRUE_B

    # Configuração de layout com GridSpec para incluir o bloco de texto explicativo
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.2], hspace=0.35, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    # Painel 1: Ajuste da Reta
    ax1.scatter(X, y, color="#2b5c8f", alpha=0.5, label="Amostras reais (com ruído)")
    ax1.plot(
        x_line,
        y_true_line,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Ground Truth (y = {TRUE_W}x + {TRUE_B})",
    )
    ax1.plot(
        x_line,
        y_pred_line,
        color="#d9381e",
        linewidth=2.2,
        label=f"Hipótese Aprendida (ŷ = {model.weights[0][0]:.2f}x + {model.bias:.2f})",
    )
    ax1.set_title("1. Ajuste da Hipótese: ŷ = Xw + b", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Feature X (Variável de Entrada)", fontsize=9)
    ax1.set_ylabel("Target y (Valor a Estimar)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Painel 2: Curva de Custo (MSE)
    ax2.plot(
        range(len(model.loss_history)), model.loss_history, color="#1b7837", linewidth=2
    )
    ax2.set_title(
        "2. Convergência da Função de Custo J(w, b)", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Épocas / Iterações (Passos do Gradiente)", fontsize=9)
    ax2.set_ylabel("Custo J(w, b) [MSE]", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2.annotate(
        f"MSE Final: {model.loss_history[-1]:.4f}",
        xy=(model.n_iterations - 1, model.loss_history[-1]),
        xytext=(model.n_iterations - 450, model.loss_history[150] * 2.0),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        fontsize=9,
        fontweight="bold",
    )

    # Painel Inferior: Glossário e Interpretação dos Gráficos
    ax_info.axis("off")
    guia_texto = (
        "GUIA DE LEITURA DOS GRÁFICOS E GLOSSÁRIO DE SIGLAS\n\n"
        "• Painel 1 (Ajuste da Reta):\n"
        "   - X (Feature): Dado de entrada fornecido (ex: metragem quadrada).\n"
        "   - y (Target Real): Valor real observado contendo variações aleatórias naturais (ruído gaussiano).\n"
        "   - ŷ (y-chapéu / Hipótese): Predição gerada pelo modelo calculada por ŷ = Xw + b.\n"
        "   - w (Weights / Peso): Inclinação da reta. Define o impacto de cada unidade de X sobre o resultado.\n"
        "   - b (Bias / Intercepto): Ponto de partida onde a reta cruza o eixo vertical quando X = 0.\n"
        "   - Ground Truth: A função teórica perfeita que gerou a base sintética original.\n\n"
        "• Painel 2 (Convergência do Treino):\n"
        "   - Épocas / Iterações: Quantidade de vezes que o algoritmo recalculou as derivadas e atualizou w e b.\n"
        "   - J(w, b) / MSE (Mean Squared Error): Média da distância ao quadrado entre o ponto real (y) e a reta (ŷ).\n"
        "   - Curva Descendente: Demonstra o Gradiente Descendente minimizando o erro gradualmente até convergir para o mínimo global."
    )
    ax_info.text(
        0.01,
        0.95,
        guia_texto,
        transform=ax_info.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f5f5f5",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    plt.show()
