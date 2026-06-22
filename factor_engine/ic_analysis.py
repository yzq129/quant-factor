"""
IC计算、因子筛选、稳定性检验
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

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

FACTOR_CATEGORIES = {
    'bp_lr': 'value', 'ep_deducted_ttm': 'value', 'fcfp_ttm': 'value', 'ocfp_ttm': 'value',
    'amount_mean_20d': 'tech', 'asset_ln': 'tech', 'revenues_ln': 'tech',
    'currentratio': 'quality', 'ocf_to_operating_profit': 'quality',
    'price_chg20d': 'momentum', 'price_chg60d': 'momentum', 'price_chg120d': 'momentum',
    'price_chg180d': 'momentum', 'price_chg1200d': 'momentum',
    'capex2sales': 'growth', 'netincome_chg1y': 'growth', 'op_profit_chg1y': 'growth'
}


def calc_rank_ic(df, factor_col, return_col='future_5d_return'):
    """计算单日截面的Rank IC (Spearman)"""
    sub = df[[factor_col, return_col]].dropna()
    if len(sub) < 10:
        return np.nan, np.nan
    ic, pval = stats.spearmanr(sub[factor_col], sub[return_col])
    return ic, pval


def calc_ic_series(df_processed, factor_col, return_col='future_5d_return'):
    """计算某因子在全时间段的IC序列"""
    ics = []
    for date, g in df_processed.groupby('trade_date'):
        ic, pval = calc_rank_ic(g, f'{factor_col}_neu', return_col)
        ics.append({'trade_date': date, 'ic': ic, 'pval': pval})
    return pd.DataFrame(ics)


def calc_ic_stats(ic_df):
    """计算IC统计量：均值、标准差、IR、胜率、t检验"""
    ics = ic_df['ic'].dropna()
    if len(ics) < 2:
        return {'ic_mean': np.nan, 'ic_std': np.nan, 'ir': np.nan,
                'win_rate': np.nan, 't_stat': np.nan, 'p_value': np.nan}
    
    ic_mean = ics.mean()
    ic_std = ics.std()
    ir = ic_mean / ic_std if ic_std != 0 else np.nan
    win_rate = (ics > 0).mean()
    t_stat, p_value = stats.ttest_1samp(ics, 0)
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ir': ir,
        'win_rate': win_rate,
        't_stat': t_stat,
        'p_value': p_value / 2 if t_stat > 0 else 1 - p_value / 2  # 单侧p值
    }


def calc_factor_correlation(df_processed):
    """计算因子间相关系数矩阵"""
    neu_cols = [f'{f}_neu' for f in FACTOR_NAMES if f'{f}_neu' in df_processed.columns]
    if len(neu_cols) < 2:
        return pd.DataFrame()
    return df_processed[neu_cols].corr()


def calc_vif(df_processed):
    """计算VIF"""
    neu_cols = [f'{f}_neu' for f in FACTOR_NAMES if f'{f}_neu' in df_processed.columns]
    df = df_processed[neu_cols].dropna()
    if df.empty or len(df.columns) < 2:
        return pd.Series()
    
    # 去常数
    df = df.loc[:, df.std() > 0]
    if df.shape[1] < 2:
        return pd.Series()
    
    X = add_constant(df, has_constant='add')
    vif_data = pd.Series(index=df.columns, dtype=float)
    for i, col in enumerate(df.columns):
        try:
            vif_data[col] = variance_inflation_factor(X.values, i+1)
        except Exception:
            vif_data[col] = np.nan
    return vif_data


def select_factors(df_processed, corr_thresh=0.7, vif_thresh=5.0, min_ic_months=2, factor_names=None):
    """
    因子筛选：共线性剔除 + 稳定性检验
    返回：入选因子列表、权重字典
    
    Parameters
    ----------
    factor_names : list, optional
        待筛选的因子列表。默认为 None，使用 ic_analysis.FACTOR_NAMES。
    """
    if factor_names is None:
        factor_names = FACTOR_NAMES
    
    # 1. 计算IC统计
    ic_stats = {}
    for fac in factor_names:
        col = f'{fac}_neu'
        if col not in df_processed.columns:
            continue
        ic_df = calc_ic_series(df_processed, fac)
        if len(ic_df.dropna()) < min_ic_months:
            continue
        stats_dict = calc_ic_stats(ic_df)
        stats_dict['ic_df'] = ic_df
        ic_stats[fac] = stats_dict
    
    if not ic_stats:
        print("[WARN] No factors passed IC minimum period filter")
        return [], {}, {}
    
    # 2. 按IR排序
    sorted_factors = sorted(ic_stats.keys(), key=lambda x: abs(ic_stats[x]['ir']), reverse=True)
    
    # 3. 共线性剔除（贪婪算法）
    corr_mat = calc_factor_correlation(df_processed)
    selected = []
    category_count = {}
    
    for fac in sorted_factors:
        cat = FACTOR_CATEGORIES.get(fac, 'other')
        
        # 检查与已选因子的相关性
        too_corr = False
        for sel in selected:
            c1 = f'{fac}_neu'
            c2 = f'{sel}_neu'
            if c1 in corr_mat.columns and c2 in corr_mat.columns:
                corr_val = abs(corr_mat.loc[c1, c2])
                if corr_val > corr_thresh:
                    too_corr = True
                    break
        
        if too_corr:
            continue
        
        selected.append(fac)
        category_count[cat] = category_count.get(cat, 0) + 1
    
    # 4. 确保每类至少1个（如果该类有因子）
    available_cats = set(FACTOR_CATEGORIES.get(f, 'other') for f in ic_stats.keys())
    for cat in available_cats:
        if category_count.get(cat, 0) == 0:
            # 从该类选一个IR最高的
            cat_factors = [f for f in sorted_factors if FACTOR_CATEGORIES.get(f, 'other') == cat and f not in selected]
            if cat_factors:
                selected.append(cat_factors[0])
                category_count[cat] = category_count.get(cat, 0) + 1
    
    # 5. 计算权重（基于IR的softmax）
    irs = np.array([abs(ic_stats[f]['ir']) for f in selected if not np.isnan(ic_stats[f]['ir'])])
    if len(irs) == 0:
        weights = {f: 1.0 / len(selected) for f in selected} if selected else {}
    else:
        # 防止IR全为0或nan
        irs = np.nan_to_num(irs, nan=0.0)
        exp_ir = np.exp(irs - np.max(irs))
        w = exp_ir / exp_ir.sum()
        weights = {f: float(w[i]) for i, f in enumerate(selected)}
    
    print(f"[INFO] Selected {len(selected)} factors: {selected}")
    ic_means = {f: ic_stats[f]['ic_mean'] for f in selected}
    return selected, weights, ic_means


def calc_factor_turnover(df_processed, factor_col):
    """计算因子换手率：相邻两期因子排名的Spearman相关性"""
    dates = sorted(df_processed['trade_date'].unique())
    corrs = []
    for i in range(1, len(dates)):
        d1 = df_processed[df_processed['trade_date'] == dates[i-1]][['code', f'{factor_col}_neu']].dropna()
        d2 = df_processed[df_processed['trade_date'] == dates[i]][['code', f'{factor_col}_neu']].dropna()
        merged = d1.merge(d2, on='code', suffixes=('_prev', '_curr'))
        if len(merged) < 10:
            continue
        corr, _ = stats.spearmanr(merged[f'{factor_col}_neu_prev'], merged[f'{factor_col}_neu_curr'])
        corrs.append(corr)
    return np.mean(corrs) if corrs else np.nan
