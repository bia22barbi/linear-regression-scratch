"""Módulo de métricas de avaliação e particionamento de dados."""

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide matrizes em partições aleatórias de treino e teste."""
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X e y devem ter a mesma quantidade de linhas: {X.shape[0]} != {y.shape[0]}"
        )

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size deve ser um valor flutuante entre 0.0 e 1.0.")

    m_samples = X.shape[0]
    indices = np.arange(m_samples)

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    test_count = int(m_samples * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Erro Quadrático Médio (MSE)."""
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a Raiz do Erro Quadrático Médio (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Erro Médio Absoluto (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Coeficiente de Determinação (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0

    return float(1.0 - (ss_res / ss_tot))
