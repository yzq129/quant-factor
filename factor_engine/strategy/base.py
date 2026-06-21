"""
策略基类
定义多因子策略的统一流程
"""
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import numpy as np

from factor_engine.config import get_config
from factor_engine.utils.logger import get_logger
from factor_engine.data.db import save_dataframe, read_sql
from factor_engine.preprocess import preprocess_all
from factor_engine.ic_analysis import (
    calc_ic_series, calc_ic_stats, select_factors,
    calc_factor_correlation, FACTOR_CATEGORIES
)
from factor_engine.ml_model import prepare_ml_data, train_model, train_rolling_models, calc_score
from factor_engine.visualization.charts import plot_ic_analysis


class BaseStrategy(ABC):
    """多因子策略基类"""
    
    def __init__(self, name):
        self.name = name
        self.config = get_config()
        self.logger = get_logger(f"strategy.{name}")
        
        # 运行时数据
        self.df_raw = None
        self.df_proc = None
        self.selected = None
        self.weights = None
        self.ic_means = None
        self.model = None
        self.models = {}
        self.ml_weights = None
        self.final_weights = None
        self.df_score = None
    
    @abstractmethod
    def get_factor_names(self):
        """返回该策略使用的因子列表"""
        pass
    
    @abstractmethod
    def get_table_suffix(self):
        """返回输出表名后缀，例如 '' 或 '_v2' 或 '_pure_ic'"""
        pass
    
    @abstractmethod
    def use_ml(self):
        """是否使用 LightGBM"""
        return True
    
    def get_factor_categories(self):
        """返回因子到类别的映射，子类可覆盖"""
        return FACTOR_CATEGORIES
    
    def get_processed_table(self):
        """处理后数据表名"""
        suffix = self.get_table_suffix()
        return f'factor_processed_daily{suffix}'
    
    def get_ic_table(self):
        """IC 结果表名"""
        suffix = self.get_table_suffix()
        return f'factor_ic_monthly{suffix}'
    
    def get_selected_table(self):
        """入选因子表名"""
        suffix = self.get_table_suffix()
        return f'factor_selected{suffix}'
    
    def get_score_table(self):
        """股票得分表名"""
        suffix = self.get_table_suffix()
        return f'stock_score_daily{suffix}'
    
    def get_charts_dir(self):
        """图表输出目录"""
        base_dir = self.config.get_path('charts_dir') or 'charts'
        suffix = self.get_table_suffix()
        if suffix:
            return os.path.join(base_dir, f'charts{suffix}')
        return base_dir
    
    def load_data(self):
        """从数据库读取原始数据"""
        self.logger.info("Loading raw data from database...")
        self.df_raw = read_sql("SELECT * FROM factor_raw_daily ORDER BY trade_date, code")
        self.logger.info(f"  Raw data: {len(self.df_raw)} rows, {self.df_raw['trade_date'].nunique()} dates")
        
        if self.df_raw.empty:
            raise ValueError("No raw data available")
    
    def preprocess(self):
        """预处理数据"""
        self.logger.info("Preprocessing data (MAD -> market-cap neutralize -> ZScore)...")
        self.df_proc = preprocess_all(self.df_raw)
        self.logger.info(f"  Processed data: {len(self.df_proc)} rows")
        
        # 保存处理后数据
        neu_cols = [c for c in self.df_proc.columns if c.endswith('_neu')]
        save_cols = ['trade_date', 'code', 'market_cap'] + neu_cols
        save_dataframe(self.df_proc[save_cols], self.get_processed_table(), if_exists='replace')
    
    def select_factors(self):
        """IC 计算与因子筛选"""
        self.logger.info("IC analysis and factor selection...")
        corr_thresh = self.config.get('factors.corr_thresh', 0.7)
        vif_thresh = self.config.get('factors.vif_thresh', 5.0)
        min_ic_months = self.config.get('factors.min_ic_months', 2)
        
        self.selected, self.weights, self.ic_means = select_factors(
            self.df_proc,
            corr_thresh=corr_thresh,
            vif_thresh=vif_thresh,
            min_ic_months=min_ic_months,
            factor_names=self.get_factor_names()
        )
        
        if not self.selected:
            factor_names = self.get_factor_names()
            self.selected = [f for f in factor_names if f'{f}_neu' in self.df_proc.columns]
            self.weights = {f: 1.0 / len(self.selected) for f in self.selected}
            self.ic_means = {f: 0.0 for f in self.selected}
            self.logger.warning("No factors selected, using all available factors")
        
        self.logger.info(f"Selected {len(self.selected)} factors: {self.selected}")
    
    def align_factor_directions(self):
        """根据 IC 方向翻转负 IC 因子"""
        self.logger.info("Aligning factor directions (IC < 0 -> flip)...")
        for fac in self.selected:
            if self.ic_means.get(fac, 0) < 0:
                neu_col = f'{fac}_neu'
                if neu_col in self.df_proc.columns:
                    self.df_proc[neu_col] = -self.df_proc[neu_col]
                    self.logger.info(f"  Flipped {fac} (IC={self.ic_means[fac]:.4f})")
    
    def train_ml_model(self):
        """训练 LightGBM 模型（滚动训练，避免未来函数）"""
        if not self.use_ml():
            self.model = None
            self.ml_weights = None
            self.models = {}
            self.final_weights = self.weights.copy()
            self.logger.info("ML disabled, using IC weights directly")
            return
        
        min_train_days = self.config.get('ml.min_train_days', 60)
        retrain_freq = self.config.get('ml.retrain_freq', 5)
        
        self.logger.info("Training rolling LightGBM models...")
        df_ml, feature_cols = prepare_ml_data(self.df_proc, self.selected)
        self.models = train_rolling_models(
            df_ml, feature_cols,
            min_train_days=min_train_days,
            retrain_freq=retrain_freq
        )
        
        # 用最后一个有效模型作为 representative（用于图表和日志）
        if self.models:
            last_date = sorted(self.models.keys())[-1]
            self.model, self.ml_weights = self.models[last_date]
            self.logger.info(f"  Trained {len(self.models)} rolling models, last: {last_date}")
        else:
            self.model = None
            self.ml_weights = None
        
        # 合并权重（仅用于日志展示，实际打分仍按日期对应模型）
        self.final_weights = {}
        for fac in self.selected:
            neu_col = f'{fac}_neu'
            w_ic = self.weights.get(fac, 0)
            w_ml = self.ml_weights.get(neu_col, 0) if self.ml_weights else 0
            self.final_weights[fac] = 0.5 * w_ic + 0.5 * w_ml if self.ml_weights else w_ic
        
        # 归一化
        total_w = sum(self.final_weights.values())
        if total_w > 0:
            self.final_weights = {k: v / total_w for k, v in self.final_weights.items()}
        
        self.logger.info(f"  Representative final weights: {self.final_weights}")
    
    def calc_scores(self):
        """计算股票得分（按日期使用对应滚动模型）"""
        self.logger.info("Calculating stock scores...")
        
        # 统一日期格式
        self.df_proc['_date_str'] = pd.to_datetime(self.df_proc['trade_date']).dt.strftime('%Y-%m-%d')
        
        score_results = []
        for date_str, g in self.df_proc.groupby('_date_str'):
            # 找到该日期可用的最新模型
            model = None
            if self.use_ml() and self.models:
                available_dates = [d for d in self.models.keys() if d <= date_str]
                if available_dates:
                    model = self.models[max(available_dates)][0]
            
            if model is not None:
                g_scored = calc_score(g, self.selected, self.final_weights, use_ml=True, model=model)
            else:
                # 数据不足时回退到 IC 线性权重
                g_scored = calc_score(g, self.selected, self.weights, use_ml=False, model=None)
            
            score_results.append(g_scored)
        
        self.df_score = pd.concat(score_results, ignore_index=True)
        self.df_score = self.df_score.drop(columns=['_date_str'], errors='ignore')
        
        # 按日期分组排名
        rank_results = []
        for date, g in self.df_score.groupby('trade_date'):
            g = g.sort_values('score', ascending=False).reset_index(drop=True)
            g['rank_in_pool'] = range(1, len(g) + 1)
            rank_results.append(g[['trade_date', 'code', 'score', 'rank_in_pool']])
        
        self.df_score = pd.concat(rank_results, ignore_index=True)
    
    def save_results(self):
        """保存 IC、入选因子、得分结果"""
        # 保存 IC 结果
        ic_records = []
        for fac in self.selected:
            ic_df = calc_ic_series(self.df_proc, fac)
            for _, row in ic_df.iterrows():
                ic_records.append({
                    'month_end': row['trade_date'],
                    'factor_name': fac,
                    'ic_value': row['ic'],
                    'p_value': row['pval']
                })
        if ic_records:
            save_dataframe(pd.DataFrame(ic_records), self.get_ic_table(), if_exists='replace')
        
        # 保存入选因子
        categories = self.get_factor_categories()
        sel_records = []
        for fac in self.selected:
            sel_records.append({
                'update_date': self.df_proc['trade_date'].max(),
                'factor_name': fac,
                'weight': self.weights.get(fac, 0),
                'category': categories.get(fac, 'other')
            })
        save_dataframe(pd.DataFrame(sel_records), self.get_selected_table(), if_exists='replace')
        
        # 保存得分
        save_dataframe(self.df_score, self.get_score_table(), if_exists='replace')
        
        self.logger.info(f"Saved results to {self.get_score_table()}")
        self.logger.info(f"  Total stocks: {self.df_score['code'].nunique()}")
        self.logger.info(f"  Total dates: {self.df_score['trade_date'].nunique()}")
    
    def print_top10(self):
        """打印最新一期 Top10"""
        latest_date = self.df_score['trade_date'].max()
        top10 = self.df_score[self.df_score['trade_date'] == latest_date].head(10)
        self.logger.info(f"\nLatest date {latest_date} Top10:")
        for _, row in top10.iterrows():
            self.logger.info(f"  {row['code']}: score={row['score']:.4f}, rank={row['rank_in_pool']}")
    
    def plot(self):
        """绘制图表"""
        self.logger.info("Generating charts...")
        output_dir = self.get_charts_dir()
        plot_ic_analysis(
            self.df_proc,
            self.df_score,
            self.selected,
            self.weights,
            self.final_weights,
            ml_weights=self.ml_weights if self.use_ml() else None,
            output_dir=output_dir,
            title_prefix=self.name
        )
    
    def run(self):
        """运行完整流程"""
        self.logger.info("=" * 60)
        self.logger.info(f"Strategy [{self.name}] pipeline started")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            self.load_data()
            self.preprocess()
            self.select_factors()
            self.align_factor_directions()
            self.train_ml_model()
            self.calc_scores()
            self.save_results()
            self.print_top10()
            self.plot()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info("=" * 60)
            self.logger.info(f"Strategy [{self.name}] completed in {elapsed:.1f}s")
            self.logger.info("=" * 60)
            
            return self.df_score
        except Exception as e:
            self.logger.exception(f"Strategy [{self.name}] failed: {e}")
            raise
