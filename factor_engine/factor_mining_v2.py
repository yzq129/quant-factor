"""
自动因子挖掘模块 V2（扩大搜索空间 + 性能优化）
1. 全部 29 个基础因子可两两交互（默认关闭，避免未来函数风险）
2. 时序特征：5日/20日均值、标准差、趋势
3. 行业内市值中性化后再做交互
4. XGBoost 残差学习提取非线性信号

优化：先用快速 IC 粗筛，再对通过者做完整中性化。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import xgboost as xgb

from factor_engine.database import read_sql, save_dataframe, get_connection
from factor_engine.preprocess import mad_winsorize

BASE_FACTORS = [
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


def safe_numeric(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace({None: np.nan}).astype(float)
    return df


def add_industry(df):
    if not os.path.exists('stock_basic.csv'):
        df['industry'] = 'unknown'
        return df
    sb = pd.read_csv('stock_basic.csv')
    if 'industry' not in sb.columns:
        df['industry'] = 'unknown'
        return df
    industry_map = dict(zip(sb['ts_code'], sb['industry']))
    df['industry'] = df['code'].map(industry_map).fillna('unknown')
    return df


def fast_neutralize(y, cap):
    """快速市值中性化：y ~ log(cap) 的 OLS 残差（numpy实现）"""
    y = np.asarray(y, dtype=float)
    cap = np.asarray(cap, dtype=float)
    valid = np.isfinite(y) & np.isfinite(cap) & (cap > 0)
    if valid.sum() < 10:
        return np.full_like(y, np.nan)
    
    x = np.log(cap[valid])
    X = np.column_stack([np.ones(x.shape[0]), x])
    yv = y[valid]
    
    # 最小二乘
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    
    out = np.full_like(y, np.nan, dtype=float)
    out[valid] = resid
    return out


def neutralize_all_factors(df_full, factor_cols, by_group=None):
    """对全时段多因子做 MAD + 市值中性化"""
    out_dfs = []
    for date, g in df_full.groupby('trade_date'):
        g = g.copy()
        if by_group and by_group in g.columns:
            for _, sub in g.groupby(by_group):
                sub = sub.copy()
                for col in factor_cols:
                    if col in sub.columns:
                        sub[col] = mad_winsorize(sub[col], n=5)
                        sub[col] = fast_neutralize(sub[col].values, sub['market_cap'].values)
                out_dfs.append(sub)
        else:
            for col in factor_cols:
                if col in g.columns:
                    g[col] = mad_winsorize(g[col], n=5)
                    g[col] = fast_neutralize(g[col].values, g['market_cap'].values)
            out_dfs.append(g)
    return pd.concat(out_dfs, ignore_index=True)


def calc_ic_stats(df_full, factor_col, label_col='future_5d_return'):
    """快速计算 IC（Spearman rank IC）"""
    if factor_col not in df_full.columns:
        return np.nan, np.nan, np.nan
    ics = []
    for date, g in df_full.groupby('trade_date'):
        x = pd.to_numeric(g[factor_col], errors='coerce').values
        y = g[label_col].values
        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[valid], y[valid]
        n = len(xv)
        if n < 10:
            ics.append(np.nan)
            continue
        # Spearman = Pearson of ranks
        rx = np.argsort(np.argsort(xv)) + 1
        ry = np.argsort(np.argsort(yv)) + 1
        ic = np.corrcoef(rx, ry)[0, 1]
        ics.append(ic)
    ics = pd.Series(ics).dropna()
    if len(ics) < 10:
        return np.nan, np.nan, np.nan
    return ics.mean(), ics.std(), ics.mean() / ics.std()


def calc_ic_stats_fast(df, factor_col):
    """不做中性化，快速计算 IC"""
    return calc_ic_stats(df, factor_col)


def generate_interaction_candidates(df_neu, top_k=10):
    """高 IC 因子两两交互（默认前10个，控制计算量）"""
    # 先计算基线 IC，选高 IC 因子
    ic_records = []
    for f in BASE_FACTORS:
        ic_mean, _, _ = calc_ic_stats(df_neu, f)
        if pd.notna(ic_mean):
            ic_records.append((f, abs(ic_mean)))
    ic_records.sort(key=lambda x: x[1], reverse=True)
    top_factors = [x[0] for x in ic_records[:top_k]]
    print(f"  Using top-{top_k} factors for interactions: {top_factors}")
    
    candidates = {}
    for f1, f2 in combinations(top_factors, 2):
        v1 = df_neu[f1].values
        v2 = df_neu[f2].values
        candidates[f'{f1}_x_{f2}'] = v1 * v2
        denom = np.where(v2 == 0, np.nan, v2)
        candidates[f'{f1}_div_{f2}'] = v1 / denom
    
    cand_df = pd.DataFrame(candidates)
    for f in BASE_FACTORS:
        cand_df[f] = df_neu[f].values
    for c in ['trade_date', 'code', 'market_cap', 'future_5d_return']:
        cand_df[c] = df_neu[c].values
    return cand_df


def generate_ts_candidates(df_neu):
    """时序特征"""
    df = df_neu.copy()
    df = df.sort_values(['code', 'trade_date'])
    
    for f in BASE_FACTORS:
        df[f'{f}_ma5'] = df.groupby('code')[f].transform(lambda x: x.rolling(5, min_periods=3).mean())
        df[f'{f}_ma20'] = df.groupby('code')[f].transform(lambda x: x.rolling(20, min_periods=10).mean())
        df[f'{f}_std5'] = df.groupby('code')[f].transform(lambda x: x.rolling(5, min_periods=3).std())
        df[f'{f}_std20'] = df.groupby('code')[f].transform(lambda x: x.rolling(20, min_periods=10).std())
        df[f'{f}_trend5'] = df.groupby('code')[f].transform(lambda x: x - x.shift(5))
        df[f'{f}_trend20'] = df.groupby('code')[f].transform(lambda x: x - x.shift(20))
    
    return df


def generate_sector_candidates(df_neu, min_stocks_per_sector=30):
    """行业内中性化后的因子交互（只在大行业做，减少计算量）"""
    if 'industry' not in df_neu.columns:
        return pd.DataFrame()
    
    # 统计各行业股票数，只保留大行业
    sector_counts = df_neu.groupby('industry')['code'].nunique()
    large_sectors = sector_counts[sector_counts >= min_stocks_per_sector].index.tolist()
    print(f"  Using {len(large_sectors)} large sectors (>= {min_stocks_per_sector} stocks)")
    
    df_large = df_neu[df_neu['industry'].isin(large_sectors)].copy()
    if len(df_large) < 1000:
        return pd.DataFrame()
    
    print("  Neutralizing within industries...")
    df_sec = neutralize_all_factors(df_large, BASE_FACTORS, by_group='industry')
    
    candidates = {}
    ic_records = []
    for f in BASE_FACTORS:
        ic_mean, _, _ = calc_ic_stats(df_sec, f)
        if pd.notna(ic_mean):
            ic_records.append((f, abs(ic_mean)))
    ic_records.sort(key=lambda x: x[1], reverse=True)
    top_factors = [x[0] for x in ic_records[:6]]
    
    for f1, f2 in combinations(top_factors, 2):
        v1 = df_sec[f1].values
        v2 = df_sec[f2].values
        candidates[f'sec_{f1}_x_{f2}'] = v1 * v2
    
    cand_df = pd.DataFrame(candidates)
    for f in BASE_FACTORS:
        cand_df[f] = df_large[f].values
    for c in ['trade_date', 'code', 'market_cap', 'future_5d_return']:
        cand_df[c] = df_sec[c].values
    return cand_df


def xgboost_residual_factor(df_neu):
    """XGBoost 残差学习"""
    feature_cols = [f for f in BASE_FACTORS if f in df_neu.columns]
    df_ml = df_neu[feature_cols + ['future_5d_return', 'trade_date', 'code']].dropna()
    if len(df_ml) < 1000:
        return None
    
    X = df_ml[feature_cols].values
    y = df_ml['future_5d_return'].values
    
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred_lr = lr.predict(X)
    residual = y - y_pred_lr
    
    dtrain = xgb.DMatrix(X, label=residual)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'eta': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'seed': 42,
        'verbosity': 0
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    residual_pred = model.predict(dtrain)
    
    df_out = df_ml[['trade_date', 'code', 'market_cap', 'future_5d_return'] + feature_cols].copy()
    df_out['xgboost_residual'] = residual_pred
    
    importance = model.get_score(importance_type='gain')
    total = sum(importance.values())
    weights = {k: v / total for k, v in importance.items()}
    print(f"  Linear model R2: {r2_score(y, y_pred_lr):.4f}")
    print(f"  XGBoost top features by gain:")
    for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]:
        idx = int(k[1:]) if k.startswith('f') else None
        name = BASE_FACTORS[idx] if idx is not None and idx < len(BASE_FACTORS) else k
        print(f"    {name}: {v:.3f}")
    
    return df_out


def fast_screen(cand_df, base_factors, min_ic=0.010, top_k=80):
    """快速 IC 粗筛"""
    candidate_cols = [c for c in cand_df.columns 
                      if c not in ['trade_date', 'code', 'market_cap', 'future_5d_return'] + base_factors]
    print(f"  Total candidates: {len(candidate_cols)}")
    
    results = []
    for idx, col in enumerate(candidate_cols):
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(candidate_cols)} rough screened")
        ic_mean, _, _ = calc_ic_stats_fast(cand_df, col)
        if pd.notna(ic_mean) and abs(ic_mean) >= min_ic:
            results.append((col, abs(ic_mean)))
    
    results.sort(key=lambda x: x[1], reverse=True)
    selected = [x[0] for x in results[:top_k]]
    print(f"  {len(selected)} candidates passed rough IC >= {min_ic}")
    return selected


def neutralize_candidates(cand_df, cols):
    """对候选因子逐日 MAD+市值中性化"""
    out = []
    for date, g in cand_df.groupby('trade_date'):
        g = g.copy()
        for col in cols:
            if col in g.columns:
                g[col] = mad_winsorize(g[col], n=5)
                g[col] = fast_neutralize(g[col].values, g['market_cap'].values)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def final_screen(cand_df_neu, base_factors, top_n=30, min_ic=0.015, corr_thresh=0.85):
    """最终 IC 筛选 + 相关性去冗余"""
    candidate_cols = [c for c in cand_df_neu.columns 
                      if c not in ['trade_date', 'code', 'market_cap', 'future_5d_return'] + base_factors]
    print(f"  Final screening {len(candidate_cols)} candidates...")
    
    results = []
    for idx, col in enumerate(candidate_cols):
        if (idx + 1) % 10 == 0:
            print(f"    {idx+1}/{len(candidate_cols)} computed")
        ic_mean, ic_std, ir = calc_ic_stats(cand_df_neu, col)
        if pd.notna(ic_mean) and abs(ic_mean) >= min_ic:
            results.append({'factor': col, 'ic_mean': ic_mean, 'ic_std': ic_std, 'ir': ir})
    
    if not results:
        return []
    
    res_df = pd.DataFrame(results)
    res_df['abs_ir'] = res_df['ir'].abs()
    res_df = res_df.sort_values('abs_ir', ascending=False).reset_index(drop=True)
    print(f"  {len(res_df)} passed min_ic={min_ic}")
    
    selected = []
    check_against = base_factors.copy()
    
    for _, row in res_df.iterrows():
        col = row['factor']
        if len(selected) >= top_n:
            break
        
        too_corr = False
        for other in check_against:
            corr_vals = []
            for date, g in cand_df_neu.groupby('trade_date'):
                sub = g[[col, other]].dropna()
                if len(sub) > 10:
                    x, y = sub[col].values, sub[other].values
                    if np.std(x) > 0 and np.std(y) > 0:
                        corr_vals.append(abs(np.corrcoef(x, y)[0, 1]))
            if corr_vals and np.mean(corr_vals) > corr_thresh:
                too_corr = True
                break
        
        if not too_corr:
            selected.append(row.to_dict())
            check_against.append(col)
    
    print(f"  Selected {len(selected)} factors:")
    for s in selected[:15]:
        print(f"    {s['factor']:40s} IC={s['ic_mean']:+.4f}  IR={s['ir']:.4f}")
    return [s['factor'] for s in selected]


def save_mined_factors(merged_df, selected_cols):
    if not selected_cols:
        return
    
    df_save = merged_df[['trade_date', 'code'] + selected_cols].copy()
    df_save = df_save.replace({np.nan: None})
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS factor_mined_daily")
            col_defs = ["`trade_date` DATE", "`code` VARCHAR(20)"]
            for col in selected_cols:
                col_defs.append(f"`{col}` DOUBLE")
            create_sql = f"CREATE TABLE factor_mined_daily ({', '.join(col_defs)}) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
            cur.execute(create_sql)
            conn.commit()
    finally:
        conn.close()
    
    save_dataframe(df_save, 'factor_mined_daily', if_exists='append')
    print(f"[OK] Saved {len(selected_cols)} mined factors to factor_mined_daily")


def main():
    print("Loading raw data...")
    df_raw = read_sql("SELECT * FROM factor_raw_daily ORDER BY trade_date, code")
    print(f"Loaded {len(df_raw)} rows")
    df_raw = safe_numeric(df_raw, BASE_FACTORS + ['market_cap', 'future_5d_return'])
    df_raw = add_industry(df_raw)
    
    print("\n[Step 1] Neutralizing base factors by market cap...")
    df_neu = neutralize_all_factors(df_raw, BASE_FACTORS)
    
    print("\n[Baseline] IC of neutralized existing factors:")
    for f in BASE_FACTORS:
        ic_mean, ic_std, ir = calc_ic_stats(df_neu, f)
        print(f"  {f:25s} IC={ic_mean:+.4f}  IR={ir:.4f}")
    
    all_selected = []
    merged = df_neu[['trade_date', 'code', 'market_cap', 'future_5d_return']].copy()
    
    # 1. 两两交互
    print("\n[Step 2] Pairwise interactions...")
    cand_inter = generate_interaction_candidates(df_neu, top_k=10)
    rough_inter = fast_screen(cand_inter, BASE_FACTORS, min_ic=0.010, top_k=80)
    if rough_inter:
        cand_inter_neu = neutralize_candidates(cand_inter, rough_inter)
        selected_inter = final_screen(cand_inter_neu, BASE_FACTORS, top_n=15, min_ic=0.015, corr_thresh=0.85)
        all_selected.extend(selected_inter)
        for col in selected_inter:
            merged[col] = cand_inter_neu[col].values
    
    # 2. 时序特征
    print("\n[Step 3] Time-series features...")
    cand_ts = generate_ts_candidates(df_neu)
    ts_base = BASE_FACTORS.copy()
    rough_ts = fast_screen(cand_ts, ts_base, min_ic=0.008, top_k=60)
    if rough_ts:
        cand_ts_neu = neutralize_candidates(cand_ts, rough_ts)
        selected_ts = final_screen(cand_ts_neu, ts_base, top_n=10, min_ic=0.012, corr_thresh=0.85)
        all_selected.extend(selected_ts)
        for col in selected_ts:
            merged[col] = cand_ts_neu[col].values
    
    # 3. 行业内交互（可选，计算量大）
    RUN_SECTOR = False
    if RUN_SECTOR:
        print("\n[Step 4] Sector-neutralized interactions...")
        cand_sec = generate_sector_candidates(df_neu)
        selected_sec = []
        if not cand_sec.empty:
            rough_sec = fast_screen(cand_sec, BASE_FACTORS, min_ic=0.010, top_k=40)
            if rough_sec:
                cand_sec_neu = neutralize_candidates(cand_sec, rough_sec)
                selected_sec = final_screen(cand_sec_neu, BASE_FACTORS, top_n=8, min_ic=0.012, corr_thresh=0.85)
                all_selected.extend(selected_sec)
                for col in selected_sec:
                    merged[col] = cand_sec_neu[col].values
    
    # 4. XGBoost 残差因子（可选）
    RUN_XGB = False
    if RUN_XGB:
        print("\n[Step 5] XGBoost residual learning...")
        df_xgb = xgboost_residual_factor(df_neu)
        selected_xgb = []
        if df_xgb is not None:
            cand_xgb_neu = neutralize_candidates(df_xgb, ['xgboost_residual'])
            selected_xgb = final_screen(cand_xgb_neu, BASE_FACTORS, top_n=2, min_ic=0.010, corr_thresh=0.90)
            all_selected.extend(selected_xgb)
            for col in selected_xgb:
                merged = merged.merge(cand_xgb_neu[['trade_date', 'code', col]], on=['trade_date', 'code'], how='left')
    
    print(f"\n[Step 6] Total selected mined factors: {len(all_selected)}")
    if not all_selected:
        print("[WARN] No factors selected")
        return
    
    save_mined_factors(merged, all_selected)
    print("\nDone!")


if __name__ == "__main__":
    main()
