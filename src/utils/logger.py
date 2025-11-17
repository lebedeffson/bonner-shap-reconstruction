"""
Утилита для логирования в проекте
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "anfis_shap",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Настройка логгера для проекта
    
    Args:
        name: Имя логгера
        level: Уровень логирования (logging.INFO, logging.DEBUG и т.д.)
        log_file: Путь к файлу для сохранения логов (опционально)
        format_string: Кастомный формат строки (опционально)
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    logger = logging.getLogger(name)
    
    # Если логгер уже настроен, возвращаем его
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Формат по умолчанию
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Консольный handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый handler (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "anfis_shap") -> logging.Logger:
    """
    Получить логгер (создает новый, если не существует)
    
    Args:
        name: Имя логгера
        
    Returns:
        logging.Logger: Логгер
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Если логгер не настроен, используем базовую настройку
        setup_logger(name)
    return logger

