"""
每日增量更新：从 Tushare 获取最新行情，结合本地历史数据计算因子
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zipfile
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta

from factor_engine.database import save_dataframe, read_sql, get_latest_date
from factor_engine.preprocess import FACTOR_NAMES

TS_TOKEN = "8e84ac848657f57f4abba76b7d7053f04d546923a1f43414017a95a3"
pro = ts.pro_api(TS_TOKEN, timeout=30)


def get_latest_trade_date():
    """获取 Tushare 最新有行情的交易日"""
    today = datetime.now().strftime('%Y%m%d')
    start = (datetime.now() - timedelta(days=15)).strftime('%Y%m%d')
    df = pro.trade_cal(exchange='SSE', start_date=start, end_date=today)
    df = df[df['is_open'] == 1].sort_values('cal_date')
    for date in reversed(df['cal_date'].tolist()):
        try:
            sample = pro.daily(trade_date=date)
            if not sample.empty:
                return date
        except Exception:
            continue
    return None


def get_csi500_stocks(trade_date):
    """获取中证500成分股，最新日期没有则回退"""
    for d in [trade_date, '20260529']:
        df = pro.index_weight(index_code='000905.SH', trade_date=d)
        if not df.empty:
            codes = df['con_code'].unique().tolist()
            print(f"CSI500 stocks on {d}: {len(codes)}")
            return codes
    return []


def load_stock_basic():
    """加载股票基础信息"""
    if os.path.exists('stock_basic.csv'):
        df = pd.read_csv('stock_basic.csv')
    else:
        df = pro.stock_basic(exchange='', list_status='L')
        df.to_csv('stock_basic.csv', index=False)
    df['is_st'] = df['name'].str.contains(r'ST|退', na=False, case=False)
    return df


def fetch_tushare_data(trade_date, csi500_codes):
    """从 Tushare 获取指定日期的行情和市值数据（按日期查全市场，再过滤成分股）"""
    print(f"\nFetching Tushare data for {trade_date}...")
    code_set = set(csi500_codes)
    
    # 日线行情
    try:
        daily_df = pro.daily(trade_date=trade_date)
        daily_df = daily_df[daily_df['ts_code'].isin(code_set)].copy()
    except Exception as e:
        print(f"  daily error: {e}")
        return None, None
    
    if daily_df.empty:
        return None, None
    
    # 日线基本指标（市值）
    try:
        basic_df = pro.daily_basic(trade_date=trade_date)
        basic_df = basic_df[basic_df['ts_code'].isin(code_set)].copy()
    except Exception as e:
        print(f"  daily_basic error: {e}")
        basic_df = None
    
    print(f"  Fetched {len(daily_df)} daily rows, {len(basic_df) if basic_df is not None else 0} basic rows")
    return daily_df, basic_df


def load_historical_prices(csi500_codes):
    """从本地 zip 加载历史价格数据"""
    print("\nLoading historical prices from local zips...")
    code_set = set(csi500_codes)
    all_data = []
    for zip_path in ['全A日K/2024.zip', '全A日K/2025.zip', '全A日K/2026.zip']:
        if not os.path.exists(zip_path):
            continue
        with zipfile.ZipFile(zip_path, 'r') as z:
            target_files = []
            for fname in z.namelist():
                if not fname.endswith('.csv') or fname.startswith('__MACOSX'):
                    continue
                code = fname.split('/')[-1].replace('.csv', '')
                if code in code_set:
                    target_files.append(fname)
            for fname in target_files:
                with z.open(fname) as f:
                    df = pd.read_csv(f)
                df = df[['datetime', 'close', 'volume', 'amount']].copy()
                df['code'] = fname.split('/')[-1].replace('.csv', '')
                df['datetime'] = pd.to_datetime(df['datetime'])
                all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
    df = pd.concat(all_data, ignore_index=True)
    df = df.rename(columns={'datetime': 'trade_date'})
    df = df.sort_values(['code', 'trade_date'])
    print(f"  Loaded {len(df)} historical price records")
    return df


def calc_factor_for_latest(hist_df, latest_df, latest_fina):
    """计算最新一日的因子"""
    latest_date = latest_df['trade_date'].iloc[0]
    print(f"\nCalculating factors for {latest_date.strftime('%Y-%m-%d')}...")
    
    results = []
    for code in latest_df['code'].unique():
        row = {'code': code, 'trade_date': latest_date}
        
        # 最新行情
        latest = latest_df[latest_df['code'] == code].iloc[0]
        close = latest['close']
        
        # 市值
        row['market_cap'] = latest.get('total_mv', np.nan) * 10000  # total_mv 单位是万元
        
        # 历史价格序列
        hist = hist_df[hist_df['code'] == code].sort_values('trade_date')
        closes = hist['close'].values
        amounts = hist['amount'].values
        
        # 价格变化因子
        row['price_chg120d'] = close / closes[-120] - 1 if len(closes) >= 120 else np.nan
        row['price_chg180d'] = close / closes[-180] - 1 if len(closes) >= 180 else np.nan
        row['price_chg1200d'] = close / closes[-1200] - 1 if len(closes) >= 1200 else np.nan
        row['amount_mean_20d'] = amounts[-20:].mean() if len(amounts) >= 20 else np.nan
        
        # 财务因子前向填充
        for col in ['bp_lr', 'ep_deducted_ttm', 'fcfp_ttm', 'ocfp_ttm', 'asset_ln', 'revenues_ln',
                    'currentratio', 'ocf_to_operating_profit', 'capex2sales',
                    'netincome_chg1y', 'op_profit_chg1y']:
            row[col] = latest_fina.get(col, {}).get(code, np.nan)
        
        row['future_5d_return'] = np.nan
        results.append(row)
    
    return pd.DataFrame(results)


def get_latest_fina_factors():
    """从数据库获取每只股票最新的财务因子"""
    df = read_sql("""
    SELECT t1.* FROM factor_raw_daily t1
    INNER JOIN (
        SELECT code, MAX(trade_date) as max_date FROM factor_raw_daily
        GROUP BY code
    ) t2 ON t1.code = t2.code AND t1.trade_date = t2.max_date
    """)
    fina_cols = ['bp_lr', 'ep_deducted_ttm', 'fcfp_ttm', 'ocfp_ttm', 'asset_ln', 'revenues_ln',
                 'currentratio', 'ocf_to_operating_profit', 'capex2sales',
                 'netincome_chg1y', 'op_profit_chg1y']
    latest = {}
    for col in fina_cols:
        latest[col] = dict(zip(df['code'], df[col]))
    return latest


def main():
    print("=" * 60)
    print("Daily Incremental Update")
    print("=" * 60)
    
    # 1. 数据库最新日期
    db_latest = get_latest_date('factor_raw_daily')
    print(f"\nDatabase latest: {db_latest}")
    
    # 2. Tushare 最新交易日
    ts_latest = get_latest_trade_date()
    print(f"Tushare latest: {ts_latest}")
    
    # 也检查挖掘因子是否已更新到最新
    mined_latest = get_latest_date('factor_mined_daily')
    print(f"Mined factors latest: {mined_latest}")
    
    if db_latest and db_latest >= pd.to_datetime(ts_latest).strftime('%Y-%m-%d') and \
       mined_latest and mined_latest >= pd.to_datetime(ts_latest).strftime('%Y-%m-%d'):
        print("[INFO] Data is up to date, no update needed")
        return
    
    # 如果行情已更新但挖掘因子没更新，跳过行情更新步骤
    skip_price_update = bool(db_latest and db_latest >= pd.to_datetime(ts_latest).strftime('%Y-%m-%d'))
    
    # 3. 获取最新成分股
    csi500_codes = get_csi500_stocks(ts_latest)
    if not csi500_codes:
        print("[ERROR] No CSI500 stocks")
        return
    
    # 4. 批量更新从 db_latest 后一个交易日到 ts_latest 的所有日期
    if not skip_price_update:
        trade_cal = pro.trade_cal(exchange='SSE', start_date=db_latest.replace('-', ''), end_date=ts_latest)
        trade_cal = trade_cal[trade_cal['is_open'] == 1].sort_values('cal_date')
        dates_to_update = [d for d in trade_cal['cal_date'].tolist() if d > db_latest.replace('-', '')]
        print(f"\nDates to update: {dates_to_update}")
        
        # 5. 加载历史价格（一次加载）
        hist_df = load_historical_prices(csi500_codes)
        if hist_df.empty:
            print("[ERROR] No historical prices")
            return
        
        # 6. 逐日更新
        for date_str in dates_to_update:
            print(f"\n{'='*60}")
            print(f"Updating {date_str}")
            print(f"{'='*60}")
            
            daily_df, basic_df = fetch_tushare_data(date_str, csi500_codes)
            if daily_df is None or daily_df.empty:
                print(f"[WARN] Skip {date_str}: no daily data")
                continue
            
            # 合并 daily 和 basic
            latest_df = daily_df.rename(columns={
                'ts_code': 'code',
                'trade_date': 'trade_date_str',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'pre_close': 'pre_close',
                'change': 'change',
                'pct_chg': 'pct_chg',
                'vol': 'volume',
                'amount': 'amount'
            })
            latest_df['trade_date'] = pd.to_datetime(latest_df['trade_date_str'], format='%Y%m%d')
            
            if basic_df is not None and not basic_df.empty:
                basic_df = basic_df.rename(columns={
                    'ts_code': 'code',
                    'trade_date': 'trade_date_str',
                    'total_mv': 'total_mv',
                    'circ_mv': 'circ_mv'
                })
                basic_df = basic_df[['code', 'total_mv', 'circ_mv']]
                latest_df = latest_df.merge(basic_df, on='code', how='left')
            
            # 过滤 ST 和停牌
            stock_basic = load_stock_basic()
            st_set = set(stock_basic[stock_basic['is_st']]['ts_code'].tolist())
            latest_df = latest_df[~latest_df['code'].isin(st_set)]
            latest_df = latest_df[(latest_df['volume'] > 0) & (latest_df['amount'] > 0)]
            
            print(f"  After filter: {len(latest_df)} stocks")
            
            # 获取最新财务因子
            latest_fina = get_latest_fina_factors()
            
            # 计算最新日因子
            df_factors = calc_factor_for_latest(hist_df, latest_df, latest_fina)
            
            # 保存
            save_cols = ['trade_date', 'code'] + FACTOR_NAMES + ['market_cap', 'future_5d_return']
            save_cols = [c for c in save_cols if c in df_factors.columns]
            save_dataframe(df_factors[save_cols], 'factor_raw_daily', if_exists='append')
            
            print(f"[OK] Updated factor_raw_daily to {date_str}")
    
    # 7. 更新挖掘因子
    print("\n" + "=" * 60)
    print("Running factor mining V2 to update mined factors...")
    print("=" * 60)
    os.system(f'"{sys.executable}" factor_engine/factor_mining_v2.py')
    
    # 8. 运行策略
    print("\n" + "=" * 60)
    print("Running strategies with new framework...")
    print("=" * 60)
    os.system(f'"{sys.executable}" scripts/run_pipeline.py all')
    
    print("\n" + "=" * 60)
    print("Update complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
