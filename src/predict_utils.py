import numpy as np

def get_rul_status(predicted_rul: float) -> (str, str, str):
    if predicted_rul > 14:
        return "НОРМАЛЬНО", "normal", "зеленый"
    elif predicted_rul > 7:
        return "ТРЕБУЕТ ОБСЛУЖИВАНИЯ", "warning", "желтый"
    else:
        return "КРИТИЧЕСКОЕ СОСТОЯНИЕ", "critical", "красный"

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    y_true_stable = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true_stable, epsilon))) * 100
    return float(mape)