"""Módulo de execução e validação da Regressão Linear do zero."""

import numpy as np

from linear_regression import (
    LinearRegression,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
    train_test_split,
)

if __name__ == "__main__":
    np.random.seed(42)

    TRUE_W = np.array([[3.5]])
    TRUE_B = 12.0
    M_SAMPLES = 250

    print("--- 1. Gerando Dados Sintéticos com Ruído ---")
    X = np.random.uniform(low=0.0, high=10.0, size=(M_SAMPLES, 1))
    noise = np.random.normal(loc=0.0, scale=1.0, size=(M_SAMPLES, 1))
    y = np.dot(X, TRUE_W) + TRUE_B + noise

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    print(f"Dataset total: {X.shape[0]} amostras")
    print(f"Conjunto Treino: {X_train.shape[0]} | Conjunto Teste: {X_test.shape[0]}\n")

    print("--- 2. Treinando o Modelo ---")
    model = LinearRegression(learning_rate=0.02, n_iterations=1000)
    model.fit(X_train, y_train)

    print("\n--- 3. Avaliação no Conjunto de Teste ---")
    y_test_pred = model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"MSE  (Erro Quadrático Médio): {test_mse:.4f}")
    print(f"RMSE (Raiz do Erro Quadrático): {test_rmse:.4f}")
    print(f"MAE  (Erro Absoluto Médio):     {test_mae:.4f}")
    print(f"R²   (Coef. de Determinação):   {test_r2:.4f}")
