"""
基准计算
提供中证500等权基准等对比指标
"""
import os
import zipfile

import numpy as np
import pandas as pd

from factor_engine.config import get_config
from factor_engine.utils.logger import get_logger


class BenchmarkEngine:
    """基准回测引擎"""
    
    def __init__(self, name='CSI500_EW'):
        self.name = name
        self.config = get_config()
        self.logger = get_logger(f"backtest.{name}")
        self.close_df = None
    
    def load_close_prices(self, codes, zip_paths=None):
        """加载收盘价"""
        if zip_paths is None:
            zip_paths = self.config.get('data.local_zip_paths', [
                '全A日K/2024.zip', '全A日K/2025.zip', '全A日K/2026.zip'
            ])
        
        self.logger.info(f"Loading close prices for benchmark [{self.name}]...")
        codes = set(codes)
        all_data = []
        
        for zip_path in zip_paths:
            if not os.path.exists(zip_path):
                self.logger.warning(f"  {zip_path} not found, skip")
                continue
            with zipfile.ZipFile(zip_path, 'r') as z:
                target_files = []
                for fname in z.namelist():
                    if not fname.endswith('.csv') or fname.startswith('__MACOSX'):
                        continue
                    code = fname.split('/')[-1].replace('.csv', '')
                    if code in codes:
                        target_files.append((fname, code))
                
                for fname, code in target_files:
                    with z.open(fname) as f:
                        df = pd.read_csv(f)
                    df = df[['datetime', 'close']].copy()
                    df['code'] = code
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    all_data.append(df)
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.rename(columns={'datetime': 'trade_date'})
        df = df.sort_values(['code', 'trade_date'])
        self.close_df = df
        self.logger.info(f"  Loaded {len(df)} price records, {df['code'].nunique()} stocks")
        return df
    
    def run_equal_weight(self, score_df, rebalance_dates=None, rebalance_freq=5):
        """
        基于 score_df 中的成分股计算等权基准
        
        Parameters
        ----------
        score_df : DataFrame
            包含 trade_date, code 的得分数据
        rebalance_dates : list, optional
            调仓日期列表，如果为 None 则按 rebalance_freq 生成
        rebalance_freq : int
            调仓频率
        """
        score_df = score_df.copy()
        score_df['trade_date'] = pd.to_datetime(score_df['trade_date'])
        
        if rebalance_dates is None:
            all_dates = sorted(score_df['trade_date'].unique())
            rebalance_dates = all_dates[::rebalance_freq]
        else:
            rebalance_dates = sorted(pd.to_datetime(rebalance_dates))
        
        if self.close_df is None:
            self.load_close_prices(score_df['code'].unique())
        
        price_pivot = self.close_df.pivot(index='trade_date', columns='code', values='close')
        
        records = []
        nav = 1.0
        
        for i, reb_date in enumerate(rebalance_dates):
            universe = score_df[score_df['trade_date'] == reb_date]['code'].unique().tolist()
            if i + 1 < len(rebalance_dates):
                next_reb_date = rebalance_dates[i + 1]
            else:
                next_reb_date = sorted(score_df['trade_date'].unique())[-1]
            
            valid_rets = []
            for code in universe:
                if code in price_pivot.columns:
                    buy = price_pivot.loc[reb_date, code] if reb_date in price_pivot.index else np.nan
                    sell = price_pivot.loc[next_reb_date, code] if next_reb_date in price_pivot.index else np.nan
                    if pd.notna(buy) and pd.notna(sell) and buy > 0:
                        valid_rets.append(sell / buy - 1)
            
            if valid_rets:
                ret = np.mean(valid_rets)
                nav *= (1 + ret)
                records.append({
                    'rebalance_date': reb_date,
                    'next_date': next_reb_date,
                    'n_holdings': len(valid_rets),
                    'gross_return': ret,
                    'net_return': ret,
                    'nav': nav
                })
        
        return pd.DataFrame(records)
