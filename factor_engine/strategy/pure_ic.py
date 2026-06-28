"""
纯 IC 加权策略
不使用 LightGBM，仅使用 29 个基础因子的 IC 权重打分，用于剥离因子 Alpha
"""
from factor_engine.strategy.base import BaseStrategy
from factor_engine.ic_analysis import FACTOR_NAMES


class PureICStrategy(BaseStrategy):
    """纯 IC 加权策略"""
    
    def __init__(self):
        super().__init__(name='Pure_IC')
    
    def get_factor_names(self):
        """返回 29 个基础因子"""
        return list(FACTOR_NAMES)
    
    def get_table_suffix(self):
        return '_pure_ic'
    
    def use_ml(self):
        return False
