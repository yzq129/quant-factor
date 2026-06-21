"""
挖掘增强多因子策略 (V2)
基础 15 个因子 + factor_mining_v2 挖掘的因子 + LightGBM
"""
import pandas as pd

from factor_engine.strategy.base import BaseStrategy
from factor_engine.data.db import read_sql, save_dataframe
from factor_engine.ic_analysis import FACTOR_NAMES, FACTOR_CATEGORIES
from factor_engine.preprocess import mad_winsorize, zscore_normalize


MINED_FACTOR_CATEGORIES = {
    'asset_ln_trend20': 'mined_ts',
    'asset_ln_trend5': 'mined_ts',
    'netincome_chg1y_std5': 'mined_ts',
    'revenues_ln_trend20': 'mined_ts',
    'netincome_chg1y_trend5': 'mined_ts',
    'op_profit_chg1y_trend20': 'mined_ts',
    'price_chg180d_trend20': 'mined_ts',
    'netincome_chg1y_std20': 'mined_ts',
    'capex2sales_trend5': 'mined_ts',
    'revenues_ln_std20': 'mined_ts',
    'fcfp_ttm_x_price_chg180d': 'mined_interact',
    'fcfp_ttm_x_price_chg120d': 'mined_interact',
}


class MinedStrategy(BaseStrategy):
    """挖掘增强多因子策略 V2"""
    
    def __init__(self):
        super().__init__(name='Mined')
        self.mined_factor_names = []
    
    def get_factor_names(self):
        """返回基础 + 挖掘因子"""
        return list(FACTOR_NAMES) + self.mined_factor_names
    
    def get_table_suffix(self):
        return '_v2'
    
    def use_ml(self):
        return True
    
    def get_factor_categories(self):
        """扩展类别映射，包含挖掘因子"""
        cats = FACTOR_CATEGORIES.copy()
        for fac in self.get_factor_names():
            if fac not in cats:
                cats[fac] = MINED_FACTOR_CATEGORIES.get(fac, 'mined_other')
        return cats
    
    def load_data(self):
        """加载已处理基础因子 + 挖掘因子"""
        self.logger.info("Loading processed base factors and mined factors...")
        
        # 读取已处理的基础因子
        df_proc = read_sql("SELECT * FROM factor_processed_daily ORDER BY trade_date, code")
        if df_proc.empty:
            raise ValueError("factor_processed_daily is empty. Run Original strategy first.")
        
        # 读取未来收益（用于 IC 计算）
        df_ret = read_sql("SELECT trade_date, code, future_5d_return FROM factor_raw_daily ORDER BY trade_date, code")
        
        # 读取挖掘因子
        df_mined = read_sql("SELECT * FROM factor_mined_daily ORDER BY trade_date, code")
        if df_mined.empty:
            raise ValueError("factor_mined_daily is empty. Run factor_mining_v2 first.")
        
        # 统一 trade_date 格式，避免不同表类型不一致导致 merge 为空
        for df_ in [df_proc, df_ret, df_mined]:
            if 'trade_date' in df_.columns:
                df_['trade_date'] = pd.to_datetime(df_['trade_date']).dt.date
        
        df_proc = df_proc.merge(df_ret, on=['trade_date', 'code'], how='left')
        
        df_mined_proc = self._preprocess_mined_factors(df_mined)
        self.mined_factor_names = [c.replace('_neu', '') for c in df_mined_proc.columns
                                   if c not in ['trade_date', 'code']]
        
        # 合并（内连接，保留有 mining 数据的所有日期）
        self.df_raw = df_proc.merge(df_mined_proc, on=['trade_date', 'code'], how='inner')
        self.logger.info(f"  Merged data: {len(self.df_raw)} rows, {self.df_raw['trade_date'].nunique()} dates")
        self.logger.info(f"  Base factors: {len(FACTOR_NAMES)}, Mined factors: {len(self.mined_factor_names)}")
    
    def preprocess(self):
        """V2 中基础因子已在 factor_processed_daily 处理好，
        挖掘因子在 load_data 中已完成 MAD+ZScore，因此这里只做清洗和保存。"""
        self.logger.info("Preprocessing merged V2 data...")
        self.df_proc = self.df_raw.copy()
        
        # 删除全空列
        self.df_proc = self.df_proc.dropna(axis=1, how='all')
        
        # 保存合并后的处理后数据（不包含 future_5d_return，避免与已有表结构冲突）
        # 所有因子列都已以 _neu 结尾，无需再单独添加 mined_cols
        neu_cols = [c for c in self.df_proc.columns if c.endswith('_neu')]
        save_cols = ['trade_date', 'code', 'market_cap'] + neu_cols
        save_cols = [c for c in save_cols if c in self.df_proc.columns]
        save_dataframe(self.df_proc[save_cols], self.get_processed_table(), if_exists='replace')
    
    def _preprocess_mined_factors(self, df_mined):
        """对挖掘因子做 MAD + Z-Score，返回 *_neu 列"""
        df = df_mined.copy()
        factor_cols = [c for c in df.columns
                       if c not in ['trade_date', 'code', 'future_5d_return']]
        neu_cols = []
        for col in factor_cols:
            neu_col = f'{col}_neu'
            df[col] = df.groupby('trade_date')[col].transform(lambda x: mad_winsorize(x, n=5))
            # 对 MAD 后的列做 Z-Score
            df[neu_col] = df.groupby('trade_date')[col].transform(zscore_normalize)
            neu_cols.append(neu_col)
        return df[['trade_date', 'code'] + neu_cols]
