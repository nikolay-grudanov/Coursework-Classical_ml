"""Wrapper script expected by Makefile to run feature engineering step."""

from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[1]))

from data.load_data import load_data, save_data
from data.preprocess import clean_data, create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path="configs/model_config.yaml"):
    """
    Загружает конфигурационный файл в формате YAML.

    Parameters:
        config_path (str): Путь к YAML-файлу конфигурации.
                           По умолчанию: "configs/model_config.yaml".

    Returns:
        dict: Словарь, содержащий данные из загруженного YAML-файла.

    Raises:
        FileNotFoundError: Если указанный файл не существует.
        yaml.YAMLError: Если произошла ошибка парсинга YAML-файла.
    """
    with open(config_path, "r") as f:
        import yaml as _yaml

        return _yaml.safe_load(f)


def main():
    """
    Основная функция для обработки данных согласно конфигурации.

    Выполняет следующие шаги:
    1. Загружает конфигурацию из YAML-файла.
    2. Резолвит пути в конфиге, поддерживая переменные окружения (форматы ${VAR:-default} и $VAR).
    3. Загружает сырые данные из указанного пути.
    4. Очищает данные.
    5. Создаёт признаки (feature engineering).
    6. Сохраняет обработанные данные.
    7. Логирует завершение процесса.

    Поддерживает расширение путей через os.path.expanduser/expandvars и обработку
    шаблонов переменных окружения в конфиге.
    """
    cfg = load_config()
    # Resolve environment-like values in config (support ${VAR:-default} and $VAR)
    import os

    def resolve_path(val):
        """
        Вспомогательная функция для резолвинга путей с подстановкой переменных окружения.

        Parameters:
            val (any): Значение, которое может содержать путь с переменными окружения.

        Returns:
            any: Обработанное значение (путь с подставленными переменными или исходное значение).
        """
        if not isinstance(val, str):
            return val
        if val.startswith("${") and val.endswith("}") and ":-" in val:
            inner = val[2:-1]
            var, default = inner.split(":-", 1)
            return os.environ.get(var, default)
        return os.path.expanduser(os.path.expandvars(val))

    data = load_data(resolve_path(cfg.get("data_path")))
    data = clean_data(data)
    data = create_features(data)
    save_data(data, resolve_path(cfg.get("processed_data_path")))
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
