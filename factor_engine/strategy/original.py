"""
原始多因子策略 (V1)
使用 29 个基础因子 + 滚动 LightGBM
"""
import os
import sys

import pandas as pd

from factor_engine.strategy.base import BaseStrategy
from factor_engine.data.db import read_sql
from factor_engine.ic_analysis import FACTOR_NAMES


class OriginalStrategy(BaseStrategy):
    """原始多因子策略 V1"""
    
    def __init__(self):
        super().__init__(name='Original')
    
    def get_factor_names(self):
        """返回 29 个基础因子"""
        return list(FACTOR_NAMES)
    
    def get_table_suffix(self):
        return ''
    
    def use_ml(self):
        return True
