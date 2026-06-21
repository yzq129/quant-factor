"""
机器学习动态加权模块
使用LightGBM进行因子加权打分
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
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
    """训练LightGBM回归模型（按时间顺序切分训练/验证集，避免未来函数）"""
    X = df_train[feature_cols]
    y = df_train[label_col]
    dates = df_train['trade_date'].values

    if len(X) < 100:
        print("[WARN] Too few samples for ML training, using linear weight fallback")
        return None, None

    # 按时间顺序排序后切分：前80%训练，后20%验证
    df_train_sorted = df_train.sort_values('trade_date').reset_index(drop=True)
    n = len(df_train_sorted)
    split_idx = int(n * 0.8)

    X_train = df_train_sorted.loc[:split_idx-1, feature_cols]
    y_train = df_train_sorted.loc[:split_idx-1, label_col]
    X_val = df_train_sorted.loc[split_idx:, feature_cols]
    y_val = df_train_sorted.loc[split_idx:, label_col]

    if len(X_val) < 10:
        print("[WARN] Validation set too small, using linear weight fallback")
        return None, None

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


def train_rolling_models(df_ml, feature_cols, label_col='future_5d_return',
                         min_train_days=60, retrain_freq=5):
    """
    滚动训练：每个调仓日用其之前的历史数据训练一个模型。
    返回 {trade_date_str: (model, weight_dict)}
    """
    df_ml = df_ml.copy()
    df_ml['trade_date'] = pd.to_datetime(df_ml['trade_date']).dt.strftime('%Y-%m-%d')
    df_ml = df_ml.sort_values('trade_date')

    all_dates = sorted(df_ml['trade_date'].unique())
    retrain_dates = all_dates[::retrain_freq]

    models = {}
    for pred_date in retrain_dates:
        idx = all_dates.index(pred_date)
        if idx < min_train_days:
            continue

        train_dates = all_dates[:idx]
        df_train = df_ml[df_ml['trade_date'].isin(train_dates)]

        if len(df_train) < 100:
            print(f"[WARN] Not enough training data for {pred_date}, skip")
            continue

        print(f"[INFO] Training model for {pred_date} using {len(df_train)} samples from {train_dates[0]} to {train_dates[-1]}")
        model, weight_dict = train_model(df_train, feature_cols, label_col)
        if model is not None:
            models[pred_date] = (model, weight_dict)

    return models


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
