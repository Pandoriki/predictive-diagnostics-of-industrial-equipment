def get_rul_status(predicted_rul: float) -> (str, str, str):
    """
    Преобразует численное предсказание RUL в строковый статус и цвет для отображения.
    Использует русские наименования для соответствия ТЗ фронтенда.

    Args:
        predicted_rul: Прогнозируемое значение RUL.

    Returns:
        Кортеж из (Статус на русском, Краткий статус для кодирования, Цвет).
    """
    if predicted_rul > 14: 
        return "НОРМАЛЬНО", "normal", "зеленый" 
    elif predicted_rul > 7:
        return "ТРЕБУЕТ ОБСЛУЖИВАНИЯ", "warning", "желтый"
    else:
        return "КРИТИЧЕСКОЕ СОСТОЯНИЕ", "critical", "красный"