# Linear Regression from Scratch (NumPy)

Implementação modular e vetorizada do algoritmo de Regressão Linear com descida do gradiente em lote (Batch Gradient Descent) construída do zero, utilizando estritamente Python e NumPy.

## Fundamentação Matemática

### 1. Hipótese Vetorizada
$$\hat{y} = Xw + b$$
- $X \in \mathbb{R}^{m \times n}$: Matriz de features ($m$ amostras, $n$ variáveis).
- $w \in \mathbb{R}^{n \times 1}$: Vetor de pesos (inclinação / taxa de variação).
- $b \in \mathbb{R}$: Bias (intercepto linear).
- $\hat{y} \in \mathbb{R}^{m \times 1}$: Vetor de predições.

### 2. Função de Custo (Mean Squared Error - MSE)
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

### 3. Gradientes Analíticos
$$\nabla_w J(w, b) = \frac{1}{m} X^T (\hat{y} - y)$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

### 4. Regra de Atualização
$$w := w - \alpha \nabla_w J(w, b)$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$
Onde $\alpha$ representa a taxa de aprendizado (learning rate).

---

## Execução do Projeto

### Pré-requisitos
- Python 3.12+ (testado em Python 3.14)
- NumPy e Matplotlib

### Instalação e Execução
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib

# Treinamento e validação no console:
python main.py

# Visualização gráfica dos dados e convergência:
python visualize.py
