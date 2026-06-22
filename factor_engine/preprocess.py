"""
因子预处理模块：去极值、标准化、市值中性化
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FACTOR_NAMES = [
    'bp_lr', 'ep_deducted_ttm', 'fcfp_ttm', 'ocfp_ttm',
    'amount_mean_20d', 'asset_ln', 'revenues_ln',
    'currentratio', 'ocf_to_operating_profit',
    'price_chg20d', 'price_chg60d', 'price_chg120d',
    'price_chg180d', 'price_chg1200d',
    'capex2sales', 'netincome_chg1y', 'op_profit_chg1y',
    'volatility_20d', 'volatility_120d',
    'turnover_mean_20d', 'turnover_std_20d', 'turnover_ratio_20d_120d',
    'rsi_14', 'price_bias_20d', 'amihud_20d',
    'roe', 'roa', 'grossprofit_margin', 'assets_yoy'
]


def mad_winsorize(series, n=5):
    """MAD去极值：中位数 ± n * MAD"""
    median = series.median()
    if pd.isna(median):
        return series
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return series
    upper = median + n * mad
    lower = median - n * mad
    return series.clip(lower, upper)


def zscore_normalize(series):
    """Z-Score标准化"""
    if series.isna().all():
        return series
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std) or pd.isna(mean):
        return series * 0
    return (series - mean) / std


def market_cap_neutralize(df, factor_col, cap_col='market_cap'):
    """市值中性化：对数市值回归取残差"""
    sub = df[[factor_col, cap_col]].dropna()
    if len(sub) < 10:
        return pd.Series(np.nan, index=df.index)
    
    X = np.log(sub[cap_col].replace(0, np.nan)).dropna()
    y = sub.loc[X.index, factor_col]
    
    # 对齐
    valid = X.notna() & y.notna()
    X = X[valid].values.reshape(-1, 1)
    y = y[valid].values
    
    if len(y) < 10:
        return pd.Series(np.nan, index=df.index)
    
    model = LinearRegression()
    model.fit(X, y)
    residual = y - model.predict(X)
    
    result = pd.Series(np.nan, index=df.index)
    result.loc[valid.index] = residual
    return result


def preprocess_daily(df, mad_n=5):
    """
    对单日截面数据进行预处理
    df: 包含多只股票某日数据的DataFrame
    """
    df = df.copy()
    
    for fac in FACTOR_NAMES:
        if fac not in df.columns:
            continue
        
        # 1. MAD去极值
        df[fac] = mad_winsorize(df[fac], n=mad_n)
        
        # 2. 市值中性化
        df[f'{fac}_neu'] = market_cap_neutralize(df, fac)
        
        # 3. Z-Score标准化（对中性化后的因子）
        df[f'{fac}_neu'] = zscore_normalize(df[f'{fac}_neu'])
    
    return df


def preprocess_all(df_raw):
    """
    对全部历史数据进行逐日截面预处理
    df_raw: 包含多只股票多日数据的DataFrame
    """
    results = []
    for date, g in df_raw.groupby('trade_date'):
        g_proc = preprocess_daily(g)
        results.append(g_proc)
    return pd.concat(results, ignore_index=True)
