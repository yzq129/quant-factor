"""
回测引擎
提供事件驱动的简单回测框架
"""
import os
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from factor_engine.config import get_config
from factor_engine.utils.logger import get_logger
from factor_engine.data.db import read_sql

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class BacktestEngine:
    """简单事件驱动回测引擎"""
    
    def __init__(self, strategy_name, score_df=None, score_table=None,
                 top_n=10, rebalance_freq=5, costs=None):
        self.strategy_name = strategy_name
        self.config = get_config()
        self.logger = get_logger(f"backtest.{strategy_name}")
        
        # 回测参数
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self.costs = costs or self._default_costs()
        
        # 数据
        if score_df is not None:
            self.score_df = score_df.copy()
        elif score_table:
            self.score_df = read_sql(f"SELECT trade_date, code, score, rank_in_pool FROM {score_table} ORDER BY trade_date, rank_in_pool")
        else:
            raise ValueError("Must provide score_df or score_table")
        
        self.score_df['trade_date'] = pd.to_datetime(self.score_df['trade_date'])
        self.close_df = None
        self.records = None
        self.metrics = None
    
    def _default_costs(self):
        return {
            'commission': self.config.get('backtest.commission_rate', 0.0003),
            'stamp_tax': self.config.get('backtest.stamp_tax_rate', 0.001),
            'slippage': self.config.get('backtest.slippage_rate', 0.0005),
        }
    
    @property
    def buy_cost(self):
        return self.costs['commission'] + self.costs['slippage']
    
    @property
    def sell_cost(self):
        return self.costs['commission'] + self.costs['stamp_tax'] + self.costs['slippage']
    
    def load_close_prices(self, codes=None, zip_paths=None):
        """从本地 zip 读取收盘价"""
        if codes is None:
            codes = set(self.score_df['code'].unique())
        else:
            codes = set(codes)
        
        if zip_paths is None:
            zip_paths = self.config.get('data.local_zip_paths', [
                '全A日K/2024.zip', '全A日K/2025.zip', '全A日K/2026.zip'
            ])
        
        self.logger.info("Loading close prices from local zips...")
        all_data = []
        
        for zip_path in zip_paths:
            if not os.path.exists(zip_path):
                self.logger.warning(f"  {zip_path} not found, skip")
                continue
            self.logger.info(f"  Reading {zip_path}...")
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
        
        if not all_data:
            raise ValueError("No close price data loaded")
        
        df = pd.concat(all_data, ignore_index=True)
        df = df.rename(columns={'datetime': 'trade_date'})
        df = df.sort_values(['code', 'trade_date'])
        self.close_df = df
        self.logger.info(f"  Loaded {len(df)} price records, {df['code'].nunique()} stocks")
        return df
    
    def run(self):
        """运行回测"""
        if self.close_df is None:
            self.load_close_prices()
        
        self.logger.info(f"Running backtest [{self.strategy_name}]...")
        self.logger.info(f"  Top N: {self.top_n}, Rebalance freq: {self.rebalance_freq} days")
        self.logger.info(f"  Buy cost: {self.buy_cost:.4%}, Sell cost: {self.sell_cost:.4%}")
        
        score_df = self.score_df.copy()
        close_df = self.close_df.copy()
        
        all_dates = sorted(score_df['trade_date'].unique())
        rebalance_dates = all_dates[::self.rebalance_freq]
        
        self.logger.info(f"  Total dates: {len(all_dates)}, Rebalance dates: {len(rebalance_dates)}")
        
        price_pivot = close_df.pivot(index='trade_date', columns='code', values='close')
        
        records = []
        nav = 1.0
        prev_holdings = set()
        
        for i, reb_date in enumerate(rebalance_dates):
            current_top = score_df[
                (score_df['trade_date'] == reb_date) &
                (score_df['rank_in_pool'] <= self.top_n)
            ]['code'].tolist()
            
            if not current_top:
                self.logger.warning(f"  No holdings on {reb_date}")
                continue
            
            if i + 1 < len(rebalance_dates):
                next_reb_date = rebalance_dates[i + 1]
            else:
                next_reb_date = all_dates[-1]
            
            # 换手率
            if prev_holdings:
                sold = prev_holdings - set(current_top)
                bought = set(current_top) - prev_holdings
                turnover = (len(sold) + len(bought)) / (2 * self.top_n)
            else:
                turnover = 0.5
            
            trade_cost = turnover * (self.buy_cost + self.sell_cost)
            
            valid_codes = []
            period_returns = []
            for code in current_top:
                if code in price_pivot.columns:
                    buy_price = price_pivot.loc[reb_date, code] if reb_date in price_pivot.index else np.nan
                    sell_price = price_pivot.loc[next_reb_date, code] if next_reb_date in price_pivot.index else np.nan
                    if pd.notna(buy_price) and pd.notna(sell_price) and buy_price > 0:
                        ret = sell_price / buy_price - 1
                        valid_codes.append(code)
                        period_returns.append(ret)
            
            if not valid_codes:
                self.logger.warning(f"  No valid prices for rebalance {reb_date} -> {next_reb_date}")
                continue
            
            gross_return = np.mean(period_returns)
            net_return = gross_return - trade_cost
            nav *= (1 + net_return)
            
            records.append({
                'rebalance_date': reb_date,
                'next_date': next_reb_date,
                'holdings': ','.join(valid_codes),
                'n_holdings': len(valid_codes),
                'turnover': turnover,
                'trade_cost': trade_cost,
                'gross_return': gross_return,
                'net_return': net_return,
                'nav': nav
            })
            
            prev_holdings = set(valid_codes)
        
        self.records = pd.DataFrame(records)
        if not self.records.empty:
            self.metrics = self.calc_metrics(self.records['nav'])
        
        self.logger.info(f"  Final NAV: {nav:.4f}")
        return self.records
    
    @staticmethod
    def calc_metrics(nav_series, freq_per_year=50):
        """计算回测指标"""
        nav_series = nav_series.reset_index(drop=True)
        returns = nav_series.pct_change().dropna()
        
        total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        annual_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (freq_per_year / len(nav_series)) - 1
        annual_vol = returns.std() * np.sqrt(freq_per_year)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan
        
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_dd = drawdown.min()
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan
        
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_ret,
            'annual_return': annual_ret,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'periods': len(nav_series)
        }
    
    def save(self, output_dir='backtest_results'):
        """保存回测记录和指标"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.records is not None and not self.records.empty:
            path = os.path.join(output_dir, f'{self.strategy_name}_records.csv')
            self.records.to_csv(path, index=False)
            self.logger.info(f"  Saved {path}")
        
        if self.metrics:
            metrics_df = pd.DataFrame([self.metrics], index=[self.strategy_name]).T
            path = os.path.join(output_dir, f'{self.strategy_name}_metrics.csv')
            metrics_df.to_csv(path)
            self.logger.info(f"  Saved {path}")
        
        return self
