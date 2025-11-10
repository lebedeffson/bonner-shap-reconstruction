"""
Загрузка конфигурации из YAML файла
"""

import yaml
import os


def load_config(config_path):
    """
    Загрузка конфигурации из YAML файла
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        dict: Словарь конфигурации
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Добавляем абсолютные пути
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Пути к данным
    if 'dataset' in config:
        if 'train_data' in config['dataset']:
            config['dataset']['train_data'] = os.path.join(
                base_dir, config['dataset']['train_data']
            )
        if 'validation_data' in config['dataset']:
            config['dataset']['validation_data'] = os.path.join(
                base_dir, config['dataset']['validation_data']
            )
    
    # Путь к результатам
    if 'output' in config and 'results_dir' in config['output']:
        results_dir = os.path.join(base_dir, config['output']['results_dir'])
        config['output']['results_dir'] = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    return config

