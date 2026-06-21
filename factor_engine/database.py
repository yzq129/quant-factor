"""
MySQL数据库操作封装（兼容旧代码）
现已迁移到 factor_engine.data.db，此文件保留用于向后兼容
"""
from factor_engine.data.db import (
    get_engine,
    get_connection,
    save_dataframe,
    read_sql,
    get_trade_dates,
    get_latest_date,
    execute,
)

__all__ = [
    'get_engine',
    'get_connection',
    'save_dataframe',
    'read_sql',
    'get_trade_dates',
    'get_latest_date',
    'execute',
]
