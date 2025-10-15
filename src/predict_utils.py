import numpy as np

def get_rul_status(predicted_rul: float) -> (str, str, str):
    if predicted_rul > 14:
        return "НОРМАЛЬНО", "normal", "зеленый"
    elif predicted_rul > 7:
        return "ТРЕБУЕТ ОБСЛУЖИВАНИЯ", "warning", "желтый"
    else:
        return "КРИТИЧЕСКОЕ СОСТОЯНИЕ", "critical", "красный"

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    # 1. Защита от деления на ноль и слишком малых значений в y_true
    #    При клиппировании, любые y_true < min_val будут считаться min_val.
    #    Это помогает предотвратить очень большие MAPE, когда истинное RUL = 1-2 цикла.
    min_mape_true_val = 1.0 # Например, считать, что RUL ниже 1 цикла для MAPE - это RUL 1
    y_true_stable = np.maximum(y_true, min_mape_true_val) # Clamp y_true for MAPE calc
                                                            # If y_true is 0 or 0.x, it will be 1.0

    # Также клиппируем y_pred, чтобы не получать отрицательные проценты,
    # и не давать отрицательные RUL
    y_pred_clipped = np.maximum(y_pred, 0)
    
    mape = np.mean(np.abs((y_true - y_pred_clipped) / y_true_stable)) * 100
    
    return float(mape)