import pandas as pd
import numpy as np
import re
import warnings
from itertools              import product
from sklearn                import metrics as mt
from sklearn.base           import clone
from sklearn.exceptions     import ConvergenceWarning

def get_descriptive_statistics(X):
    # Central Tendency - mean, median
    X.describe().T
    ct1 = pd.DataFrame(X.mean()).T
    ct2 = pd.DataFrame(X.median()).T

    # Dispersion - sdt, min, max, range, skew, kurtosis
    d1 = pd.DataFrame(X.std()).T
    d2 = pd.DataFrame(X.min()).T
    d3 = pd.DataFrame(X.max()).T
    d4 = pd.DataFrame(X.apply(lambda x: x.max() - x.min())).T
    d5 = pd.DataFrame(X.skew()).T
    d6 = pd.DataFrame(X.kurtosis()).T

    # concatenate
    m = pd.concat([d2, d3, d4, ct1, ct2, d1, d5, d6]).T.reset_index()
    m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
    return m

def grid_search(model, param_grid, X_train, y_train, X_val, y_val, classification=True):
    """
    Realiza uma busca em grade (grid search) para encontrar os melhores hiperparâmetros para um modelo.

    Args:
        model (sklearn.base.BaseEstimator): Instância do modelo a ser ajustado.
        param_grid (dict): Dicionário de hiperparâmetros a serem testados.
        X_train (pd.DataFrame or np.ndarray): Dados de treinamento (features).
        y_train (pd.Series or np.ndarray): Rótulos de treinamento.
        X_val (pd.DataFrame or np.ndarray): Dados de validação (features).
        y_val (pd.Series or np.ndarray): Rótulos de validação.
        classification (bool): Indica se o problema é de classificação (True) ou regressão (False).

    Returns:
        tuple: Contendo o melhor modelo, os melhores hiperparâmetros e o melhor score.
    """

    # Gera todas as combinações de hiperparâmetros
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    best_score = -np.inf if classification else np.inf
    best_params = None
    best_model = None

    # Loop sobre todas as combinações de hiperparâmetros
    for params in param_combinations:
        # Cria um dicionário de hiperparâmetros para o modelo
        param_dict = {param_names[i]: params[i] for i in range(len(params))}
        
        # Instanciar e treinar o modelo
        model_instance = clone(model)
        model_instance.set_params(**param_dict)

        # Ignorar warnings temporariamente
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model_instance.fit(X_train, y_train)

        # Avaliar o modelo nos dados de validação
        y_val_pred = model_instance.predict(X_val)
        if classification:
            score = mt.f1_score(y_val, y_val_pred)
            if score > best_score:
                best_score = score
                best_params = param_dict
                best_model = model_instance
        else:
            score = np.sqrt(mt.mean_squared_error(y_val, y_val_pred))
            if score < best_score:
                best_score = score
                best_params = param_dict
                best_model = model_instance

    # Inicializar o best_model novamente para garantir que não há parâmetros internos remanescentes
    best_model = clone(model)
    best_model.set_params(**best_params)

    # Retreinar o melhor modelo encontrado com a união dos dados de treino e validação
    X_combined = np.concatenate((X_train, X_val), axis=0)
    y_combined = np.concatenate((y_train, y_val), axis=0)
    best_model.fit(X_combined, y_combined)

    return best_model, best_params, best_score

def get_ml_performance(model, X, y, classification=True):
    """
    Realiza uma busca em grade (grid search) para encontrar os melhores hiperparâmetros para um modelo.

    Args:
        model (sklearn.base.BaseEstimator): Instância do modelo a ser ajustado.
        param_grid (dict): Dicionário com hiperparâmetros e listas de valores para testar.
        X_train (array-like): Dados de treinamento.
        y_train (array-like): Rótulos de treinamento.
        X_val (array-like): Dados de validação.
        y_val (array-like): Rótulos de validação.
        classification (bool): Se True, usa métricas de classificação; se False, usa métricas de regressão.

    Returns:
        best_model: O melhor modelo treinado com os melhores hiperparâmetros.
        best_params: Os melhores hiperparâmetros encontrados.
        best_score: A melhor pontuação obtida com os hiperparâmetros.
    """
    # Obter o nome do modelo formatado
    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', model.__class__.__name__)).split()
    model_name = " ".join(n for n in splitted)

    # Prever os valores
    y_hat = model.predict(X)

    if classification:
        # Calcular as métricas
        accuracy = mt.accuracy_score(y, y_hat)
        precision = mt.precision_score(y, y_hat)
        recall = mt.recall_score(y, y_hat)
        f1 = mt.f1_score(y, y_hat)

        return pd.DataFrame({'Algoritmo': model_name,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1}, index=[0]).round(3)

    else:
        # Calcular as métricas
        r2 = mt.r2_score(y, y_hat)
        mse = mt.mean_squared_error(y, y_hat)
        rmse = mt.root_mean_squared_error(y, y_hat)
        mae = mt.mean_absolute_error(y, y_hat)
        mape = mt.mean_absolute_percentage_error(y, y_hat)

        return pd.DataFrame({'Algoritmo': model_name,
                             'R2': r2,
                             'MSE': mse,
                             'RMSE': rmse,
                             'MAE': mae,
                             'MAPE': mape}, index=[0]).round(3)