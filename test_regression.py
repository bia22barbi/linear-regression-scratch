"""Suite de testes unitarios de alta precisao para a Regressao Linear e metricas."""

import re

import numpy as np
import pytest

from linear_regression import (
    LinearRegression,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    train_test_split,
)


def test_linear_regression_shapes_and_training() -> None:
    """Valida se pesos e bias assumem shapes e valores coerentes apos treino."""
    np.random.seed(42)
    m_samples, n_features = 100, 2
    X = np.random.uniform(0.0, 5.0, size=(m_samples, n_features))
    true_w = np.array([[2.0], [-1.5]])
    true_b = 5.0
    y = np.dot(X, true_w) + true_b

    model = LinearRegression(learning_rate=0.03, n_iterations=1200)
    model.fit(X, y)

    assert model.weights is not None
    assert model.weights.shape == (n_features, 1)
    assert isinstance(model.bias, float)
    np.testing.assert_allclose(model.weights, true_w, atol=1e-1)
    np.testing.assert_allclose(model.bias, true_b, atol=1e-1)


def test_cost_reduction_monotonic() -> None:
    """Verifica se o gradiente descendente diminui o erro continuamente."""
    np.random.seed(42)
    X = np.random.uniform(1.0, 10.0, size=(80, 1))
    y = 3.0 * X + 2.0

    model = LinearRegression(learning_rate=0.01, n_iterations=300)
    model.fit(X, y)

    assert len(model.loss_history) == 300
    assert model.loss_history[-1] < model.loss_history[0]
    assert model.loss_history[-1] < 0.1


def test_predict_before_fit_raises_error() -> None:
    """Garante falha controlada caso .predict() seja chamado antes de .fit()."""
    model = LinearRegression()
    dummy_x = np.array([[1.0, 2.0]])
    expected_msg = "O modelo precisa ser treinado com .fit() antes de prever."
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        model.predict(dummy_x)


def test_train_test_split_ratios_and_integrity() -> None:
    """Valida dimensoes e proporcoes geradas pelo particionamento."""
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100).reshape(100, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )

    assert X_train.shape == (75, 2)
    assert X_test.shape == (25, 2)
    assert y_train.shape == (75, 1)
    assert y_test.shape == (25, 1)


def test_metrics_analytical_exactness() -> None:
    """Verifica exatidao analitica das funcoes de metrica com valores conhecidos."""
    y_true = np.array([[10.0], [20.0], [30.0]])
    y_pred = np.array([[12.0], [18.0], [32.0]])

    assert mean_squared_error(y_true, y_pred) == pytest.approx(4.0)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(2.0)
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(2.0)
    assert r2_score(y_true, y_pred) > 0.90
