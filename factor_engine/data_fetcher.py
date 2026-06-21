"""
Tushare数据获取模块
替代聚宽，支持更长历史数据
"""
import pandas as pd
import numpy as np
import tushare as ts
import os
import pickle
import time

TS_TOKEN = "8e84ac848657f57f4abba76b7d7053f04d546923a1f43414017a95a3"
pro = ts.pro_api(TS_TOKEN)

# 缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def get_csi500_stocks(trade_date):
    """获取中证500成分股，使用最近一期调整日数据"""
    # 获取所有历史成分股调整记录
    df_all = pro.index_weight(index_code='000905.SH')
    if df_all.empty:
        return []
    
    # 找到trade_date之前的最新调整日
    df_all['trade_date_int'] = df_all['trade_date'].astype(int)
    target_int = int(trade_date.replace('-', ''))
    df_valid = df_all[df_all['trade_date_int'] <= target_int]
    if df_valid.empty:
        return []
    
    latest_date = df_valid['trade_date'].max()
    df = df_valid[df_valid['trade_date'] == latest_date]
    stocks = df['con_code'].unique().tolist()
    print(f"[{trade_date}] CSI500 stocks (using {latest_date} weight): {len(stocks)}")
    return stocks


def get_report_period(trade_date):
    """根据交易日期推断最新财报报告期"""
    year = int(trade_date[:4])
    month = int(trade_date[4:6]) if len(trade_date) == 8 else int(trade_date[5:7])
    
    if month >= 5:
        return f"{year}0331"  # 一季报
    else:
        return f"{year-1}0930"  # 三季报


def fetch_fina_indicator_batch(stocks, period):
    """批量获取财务指标，每次100只"""
    if not stocks:
        return pd.DataFrame()
    
    results = []
    for i in range(0, len(stocks), 100):
        batch = stocks[i:i+100]
        ts_codes = ','.join(batch)
        try:
            df = pro.fina_indicator(ts_code=ts_codes, period=period)
            if df is not None and not df.empty:
                results.append(df)
        except Exception as e:
            print(f"  fina_indicator batch {i} error: {e}")
        time.sleep(0.05)
    
    valid_results = [r for r in results if r is not None and not r.empty]
    return pd.concat(valid_results, ignore_index=True) if valid_results else pd.DataFrame()


def fetch_daily_basic(trade_date):
    """获取全部A股日行情基础指标（PE/PB/市值）"""
    date_str = trade_date.replace('-', '') if '-' in trade_date else trade_date
    try:
        df = pro.daily_basic(trade_date=date_str)
        return df
    except Exception as e:
        print(f"  daily_basic error: {e}")
        return pd.DataFrame()


def fetch_daily_batch(stocks, start_date, end_date):
    """批量获取日行情，每次50只"""
    if not stocks:
        return pd.DataFrame()
    
    results = []
    for i in range(0, len(stocks), 50):
        batch = stocks[i:i+50]
        ts_codes = ','.join(batch)
        try:
            df = pro.daily(ts_code=ts_codes, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                results.append(df)
        except Exception as e:
            print(f"  daily batch {i} error: {e}")
        time.sleep(0.05)
    
    valid_results = [r for r in results if r is not None and not r.empty]
    return pd.concat(valid_results, ignore_index=True) if valid_results else pd.DataFrame()


def fetch_financial_statements(stocks, period):
    """获取资产负债表/利润表/现金流量表，带本地缓存"""
    cache_file = os.path.join(CACHE_DIR, f'fin_stmt_{period}.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"  Fetching financial statements for {period} ({len(stocks)} stocks)...")
    bal_list, inc_list, cf_list = [], [], []
    
    for i, ts_code in enumerate(stocks):
        try:
            bal = pro.balancesheet(ts_code=ts_code, period=period)
            if bal is not None and not bal.empty:
                bal_list.append(bal)
        except:
            pass
        
        try:
            inc = pro.income(ts_code=ts_code, period=period)
            if inc is not None and not inc.empty:
                inc_list.append(inc)
        except:
            pass
        
        try:
            cf = pro.cashflow(ts_code=ts_code, period=period)
            if cf is not None and not cf.empty:
                cf_list.append(cf)
        except:
            pass
        
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(stocks)} done")
            time.sleep(0.5)
    
    bal_df = pd.concat([b for b in bal_list if not b.empty], ignore_index=True) if bal_list else pd.DataFrame()
    inc_df = pd.concat([i for i in inc_list if not i.empty], ignore_index=True) if inc_list else pd.DataFrame()
    cf_df = pd.concat([c for c in cf_list if not c.empty], ignore_index=True) if cf_list else pd.DataFrame()
    
    result = {'balancesheet': bal_df, 'income': inc_df, 'cashflow': cf_df}
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"  Cached to {cache_file}")
    return result


def calc_momentum_factors(price_df, trade_date):
    """从价格数据计算动量因子"""
    if price_df.empty:
        return pd.DataFrame()
    
    results = []
    for code, g in price_df.groupby('ts_code'):
        g = g.sort_values('trade_date').reset_index(drop=True)
        if len(g) < 5:
            continue
        latest = g.iloc[-1]['close']
        
        # 成交额20日均值 (amount单位是千元，转为元)
        amount_20d = g['amount'].tail(20).mean() * 1000 if len(g) >= 20 else np.nan
        
        # 收益率
        ret_120d = (latest / g.iloc[-120]['close'] - 1) if len(g) >= 120 else np.nan
        ret_180d = (latest / g.iloc[-180]['close'] - 1) if len(g) >= 180 else np.nan
        ret_1200d = (latest / g.iloc[-1200]['close'] - 1) if len(g) >= 1200 else np.nan
        
        # 未来5日收益率（用于IC和ML标签）
        future_5d = (g.iloc[-1]['close'] / g.iloc[-6]['close'] - 1) if len(g) >= 6 else np.nan
        
        results.append({
            'code': code,
            'amount_mean_20d': amount_20d,
            'price_chg120d': ret_120d,
            'price_chg180d': ret_180d,
            'price_chg1200d': ret_1200d,
            'future_5d_return': future_5d
        })
    return pd.DataFrame(results)


def calc_factors(trade_date):
    """计算某交易日的全部因子"""
    date_str = trade_date.replace('-', '') if '-' in trade_date else trade_date
    
    # 1. 获取股票池
    stocks = get_csi500_stocks(trade_date)
    if not stocks:
        print(f"[{trade_date}] No CSI500 stocks found")
        return pd.DataFrame()
    
    # 2. 获取日行情基础指标 (PE/PB/市值)
    print(f"  Fetching daily_basic...")
    df_basic = fetch_daily_basic(date_str)
    if df_basic.empty:
        print(f"  No daily_basic data")
        return pd.DataFrame()
    
    # 过滤中证500成分股
    df_basic = df_basic[df_basic['ts_code'].isin(stocks)].copy()
    if df_basic.empty:
        print(f"  No CSI500 stocks in daily_basic")
        return pd.DataFrame()
    
    # 3. 获取财务指标
    period = get_report_period(date_str)
    print(f"  Fetching fina_indicator (period={period})...")
    df_fina = fetch_fina_indicator_batch(stocks, period)
    
    # 4. 获取历史行情（用于动量因子和成交额）
    # 需要至少1200日历史
    start_dt = pd.to_datetime(trade_date) - pd.Timedelta(days=1800)
    start_str = start_dt.strftime('%Y%m%d')
    print(f"  Fetching daily price history ({start_str} to {date_str})...")
    price_df = fetch_daily_batch(stocks, start_str, date_str)
    df_price = calc_momentum_factors(price_df, trade_date)
    
    # 5. 获取财务报表（用于资产、营收、现金流、Capex）
    print(f"  Fetching financial statements...")
    fin_stmt = fetch_financial_statements(stocks, period)
    df_bal = fin_stmt['balancesheet']
    df_inc = fin_stmt['income']
    df_cf = fin_stmt['cashflow']
    
    # 6. 合并数据
    df = df_basic[['ts_code', 'total_mv', 'pe_ttm', 'pb']].copy()
    df.rename(columns={'ts_code': 'code', 'total_mv': 'market_cap'}, inplace=True)
    # total_mv单位是万元，转为亿元
    df['market_cap'] = df['market_cap'] / 10000.0
    
    # 合并fina_indicator
    if not df_fina.empty:
        fina_cols = ['ts_code', 'netprofit_yoy', 'op_yoy', 'current_ratio', 
                     'grossprofit_margin', 'assets_yoy']
        fina_avail = [c for c in fina_cols if c in df_fina.columns]
        if len(fina_avail) > 1:
            df_fina = df_fina[fina_avail].copy()
            df_fina.rename(columns={'ts_code': 'code'}, inplace=True)
            df_fina = df_fina.drop_duplicates(subset=['code'], keep='first')
            df = df.merge(df_fina, on='code', how='left')
    
    # 合并price
    if not df_price.empty:
        df = df.merge(df_price, on='code', how='left')
    
    # 合并balancesheet
    if not df_bal.empty:
        bal_cols = ['ts_code', 'total_assets', 'total_cur_assets', 'total_cur_liab']
        bal_avail = [c for c in bal_cols if c in df_bal.columns]
        if len(bal_avail) > 1:
            df_bal = df_bal[bal_avail].copy()
            df_bal.rename(columns={'ts_code': 'code'}, inplace=True)
            df_bal = df_bal.drop_duplicates(subset=['code'], keep='first')
            df = df.merge(df_bal, on='code', how='left')
    
    # 合并income
    if not df_inc.empty:
        inc_cols = ['ts_code', 'revenue', 'operate_profit']
        inc_avail = [c for c in inc_cols if c in df_inc.columns]
        if len(inc_avail) > 1:
            df_inc = df_inc[inc_avail].copy()
            df_inc.rename(columns={'ts_code': 'code'}, inplace=True)
            df_inc = df_inc.drop_duplicates(subset=['code'], keep='first')
            df = df.merge(df_inc, on='code', how='left')
    
    # 合并cashflow
    if not df_cf.empty:
        cf_cols = ['ts_code', 'n_cashflow_act', 'c_pay_acq_const_fiolta', 'free_cashflow']
        cf_avail = [c for c in cf_cols if c in df_cf.columns]
        if len(cf_avail) > 1:
            df_cf = df_cf[cf_avail].copy()
            df_cf.rename(columns={'ts_code': 'code'}, inplace=True)
            df_cf = df_cf.drop_duplicates(subset=['code'], keep='first')
            df = df.merge(df_cf, on='code', how='left')
    
    # 7. 计算最终因子值
    # 价值
    df['bp_lr'] = 1.0 / df['pb'].replace(0, np.nan)
    df['ep_deducted_ttm'] = 1.0 / df['pe_ttm'].replace(0, np.nan)
    # fcfp_ttm: free_cashflow / market_cap (free_cashflow单位是元，market_cap是亿元)
    df['fcfp_ttm'] = df['free_cashflow'] / (df['market_cap'].replace(0, np.nan) * 1e8)
    df['ocfp_ttm'] = df['n_cashflow_act'] / (df['market_cap'].replace(0, np.nan) * 1e8)
    
    # 技术
    df['asset_ln'] = np.log(df['total_assets'].replace(0, np.nan))
    df['revenues_ln'] = np.log(df['revenue'].replace(0, np.nan))
    
    # 质量
    df['currentratio'] = df['total_cur_assets'] / df['total_cur_liab'].replace(0, np.nan)
    # ocf_to_operating_profit
    df['ocf_to_operating_profit'] = df['n_cashflow_act'] / df['operate_profit'].replace(0, np.nan)
    
    # 动量已在price中计算
    
    # 成长
    df['capex2sales'] = df['c_pay_acq_const_fiolta'] / df['revenue'].replace(0, np.nan)
    df['netincome_chg1y'] = df.get('netprofit_yoy', np.nan) / 100.0
    df['op_profit_chg1y'] = df.get('op_yoy', np.nan) / 100.0
    
    # 统一列名
    df['trade_date'] = trade_date
    factor_cols = [
        'trade_date', 'code', 'market_cap', 'future_5d_return',
        'bp_lr', 'ep_deducted_ttm', 'fcfp_ttm', 'ocfp_ttm',
        'amount_mean_20d', 'asset_ln', 'revenues_ln',
        'currentratio', 'ocf_to_operating_profit',
        'price_chg1200d', 'price_chg120d', 'price_chg180d',
        'capex2sales', 'netincome_chg1y', 'op_profit_chg1y'
    ]
    
    avail_cols = [c for c in factor_cols if c in df.columns]
    result = df[avail_cols].copy()
    print(f"  Result: {len(result)} stocks with {len(avail_cols)} columns")
    return result
