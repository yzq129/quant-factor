"""
机器学习动态加权模块
使用LightGBM进行因子加权打分
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

FACTOR_NAMES = [
    'bp_lr', 'ep_deducted_ttm', 'fcfp_ttm', 'ocfp_ttm',
    'amount_mean_20d', 'asset_ln', 'revenues_ln',
    'currentratio', 'ocf_to_operating_profit',
    'price_chg1200d', 'price_chg120d', 'price_chg180d',
    'capex2sales', 'netincome_chg1y', 'op_profit_chg1y'
]


def prepare_ml_data(df_processed, selected_factors, return_col='future_5d_return'):
    """准备ML训练数据"""
    feature_cols = [f'{f}_neu' for f in selected_factors if f'{f}_neu' in df_processed.columns]
    df = df_processed[feature_cols + [return_col, 'code', 'trade_date']].copy()
    df = df.dropna()
    return df, feature_cols


def train_model(df_train, feature_cols, label_col='future_5d_return'):
    """训练LightGBM回归模型"""
    X = df_train[feature_cols]
    y = df_train[label_col]
    
    if len(X) < 100:
        print("[WARN] Too few samples for ML training, using linear weight fallback")
        return None, None
    
    # 分位数标签（用于稳定性）
    y_q = pd.qcut(y, q=5, labels=False, duplicates='drop')
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]
    )
    
    # 特征重要性作为动态权重
    importance = model.feature_importance(importance_type='gain')
    weights = importance / importance.sum() if importance.sum() > 0 else np.ones(len(feature_cols)) / len(feature_cols)
    weight_dict = {f: float(w) for f, w in zip(feature_cols, weights)}
    
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f"[INFO] LightGBM validation R2: {r2:.4f}")
    
    return model, weight_dict


def calc_score(df, selected_factors, weights, use_ml=True, model=None):
    """计算股票综合得分"""
    feature_cols = [f'{f}_neu' for f in selected_factors if f'{f}_neu' in df.columns]
    
    if use_ml and model is not None:
        # 使用ML模型预测得分
        X = df[feature_cols].fillna(0)
        scores = model.predict(X)
    else:
        # 使用线性加权
        scores = np.zeros(len(df))
        for fac in selected_factors:
            col = f'{fac}_neu'
            if col in df.columns and fac in weights:
                scores += df[col].fillna(0) * weights.get(fac, 0)
    
    df = df.copy()
    df['score'] = scores
    return df
