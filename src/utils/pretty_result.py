from IPython.display import display
import json
import pandas as pd

"""
Утилита для красивого вывода результатов обучения или оценки модели.
Поддерживает различные структуры данных, такие как словарь, DataFrame и список.
"""


def pretty_results(res, title: str = "Результаты"):
    """
    Форматирует и отображает результаты обучения/оценки модели в удобочитаемом виде.

    Поддерживаемые типы данных:
    - Строки (с автоматической попыткой парсинга JSON)
    - Pandas DataFrame
    - Словари (с превращением в таблицу)
    - Списки (с превращением в таблицу)

    Parameters:
        res (any): Результаты, которые необходимо отформатировать и отобразить

    Behavior:
        - Для строк: пытается распарсить как JSON, иначе выводит как есть
        - Для DataFrame: отображает таблицу
        - Для словарей: преобразует в таблицу, автоматически транспонируя при необходимости
        - Для списков: пытается преобразовать в таблицу
        - Для остальных типов: выводит как есть

    Notes:
        Использует pd.json_normalize() для нормализации вложенных структур.
        При транспонировании словарей предполагается, что ключи представляют собой модели,
        а значения - их метрики. В этом случае индекс таблицы будет называться 'model'.
    """
    # Попытка декодировать строку JSON
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            print(res)
            return

    # Pandas DataFrame
    if isinstance(res, pd.DataFrame):
        display(res)
        return

    # Словарный тип -> создать читаемый DataFrame
    if isinstance(res, dict):
        try:
            df = pd.json_normalize(res, sep="_")
            # Если ориентация - модели -> метрики, предпочтительнее использовать индекс=models
            if df.shape[0] == 1 and df.shape[1] > 1:
                df = df.T
            df.index.name = "model"
            display(df)
            return
        except Exception:
            display(pd.DataFrame.from_dict(res, orient="index"))
            return

    # Список -> DataFrame при возможности
    if isinstance(res, list):
        try:
            display(pd.DataFrame(res))
            return
        except Exception:
            print(res)
            return

    # fallback для остальных типов
    print(res)
