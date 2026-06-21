"""
日志工具
提供统一的日志配置和 get_logger 工厂函数
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from factor_engine.config import get_config


# 避免重复配置
_logging_initialized = False


def setup_logging(level=None, log_dir=None):
    """
    配置根日志记录器
    同时输出到控制台和文件
    """
    global _logging_initialized
    if _logging_initialized:
        return
    
    cfg = get_config()
    
    # 日志级别
    if level is None:
        level_name = cfg.get('logging.level', 'INFO')
        level = getattr(logging, level_name.upper(), logging.INFO)
    
    # 日志格式
    log_format = cfg.get(
        'logging.format',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    date_format = cfg.get('logging.date_format', '%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # 根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除已有 handler，避免重复
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件 handler
    if log_dir is None:
        log_dir = cfg.get_path('log_dir')
    
    if log_dir:
        today = datetime.now().strftime('%Y%m%d')
        log_path = Path(log_dir) / today
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f"pipeline_{datetime.now().strftime('%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file}")
    
    _logging_initialized = True


def get_logger(name):
    """获取命名日志记录器"""
    if not _logging_initialized:
        setup_logging()
    return logging.getLogger(name)


def reset_logging():
    """重置日志配置（主要用于测试）"""
    global _logging_initialized
    _logging_initialized = False
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
