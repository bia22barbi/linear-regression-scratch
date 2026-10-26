"""Visualização gráfica com Matplotlib para convergência e resíduos."""

import matplotlib.pyplot as plt
import numpy as np

from linear_regression import LinearRegression

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
    residuals = y - model.predict(X)

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.1], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[1, :])

    # Painel 1: Ajuste
    ax1.scatter(X, y, color="#2b5c8f", alpha=0.5, label="Amostras reais")
    ax1.plot(x_line, y_true_line, "k--", linewidth=1.5, label=f"Ideal (y={TRUE_W}x+{TRUE_B})")
    ax1.plot(
        x_line,
        y_pred_line,
        color="#d9381e",
        linewidth=2.2,
        label=f"Modelo (ŷ={model.weights[0][0]:.2f}x+{model.bias:.2f})",
    )
    ax1.set_title("1. Ajuste da Hipótese: ŷ = Xw + b", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Feature X", fontsize=9)
    ax1.set_ylabel("Target y", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Painel 2: Custo
    ax2.plot(range(len(model.loss_history)), model.loss_history, color="#1b7837", linewidth=2)
    ax2.set_title("2. Convergência da Função de Custo J(w,b)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Épocas", fontsize=9)
    ax2.set_ylabel("Custo [MSE]", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Painel 3: Histograma
    ax3.hist(residuals, bins=15, color="#4a7c59", edgecolor="#222222", alpha=0.75, density=True)
    res_std = np.std(residuals)
    res_mean = np.mean(residuals)
    x_axis = np.linspace(residuals.min() - 0.5, residuals.max() + 0.5, 100)
    pdf = (1 / (res_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - res_mean) / res_std) ** 2)
    ax3.plot(x_axis, pdf, "r--", linewidth=1.8, label=f"Normal (μ={res_mean:.2f})")
    ax3.set_title("3. Distribuição dos Resíduos (e = y - ŷ)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Erro Residual", fontsize=9)
    ax3.set_ylabel("Densidade", fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Informações Técnicas
    ax_info.axis("off")
    guia_texto = (
        "DIAGNÓSTICO E PREMISSAS DO MODELO:\n\n"
        "1. Ajuste: Avalia a linearidade entre a variável independente X e a resposta y.\n"
        "2. Convergência: Demonstra decaimento monótono da função de perda por época.\n"
        "3. Resíduos: Devem aproximar a curva normal centrada em zero (μ ≈ 0) "
        "sem assimetrias."
    )
    ax_info.text(
        0.01,
        0.95,
        guia_texto,
        transform=ax_info.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#cccccc", alpha=0.9),
    )

    plt.tight_layout()
    plt.show()
