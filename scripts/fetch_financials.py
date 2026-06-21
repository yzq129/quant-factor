"""
批量获取历史 CSI500 成分股的财务数据并缓存
包括：fina_indicator、balancesheet、income、cashflow
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import tushare as ts

from factor_engine.config import get_config

TS_TOKEN = "8e84ac848657f57f4abba76b7d7053f04d546923a1f43414017a95a3"
pro = ts.pro_api(TS_TOKEN, timeout=60)


def load_universe_codes():
    """从历史成分股文件中加载所有出现过的股票代码"""
    df = pd.read_csv('config/csi500_universe_history.csv')
    codes = sorted(df['code'].unique().tolist())
    return codes


def fetch_with_retry(func, **kwargs):
    """带重试的 Tushare 调用"""
    max_retries = 3
    for i in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            print(f"    Retry {i+1}/{max_retries}: {e}")
            time.sleep(2)
    return None


def main():
    codes = load_universe_codes()
    print(f"Total unique codes: {len(codes)}")
    
    cache_file = 'cache/financials_all.pkl'
    if os.path.exists(cache_file):
        print(f"Loading existing cache from {cache_file}")
        cache = pd.read_pickle(cache_file)
    else:
        cache = {'fina_indicator': {}, 'balancesheet': {}, 'income': {}, 'cashflow': {}}
    
    for idx, code in enumerate(codes):
        print(f"\n[{idx+1}/{len(codes)}] Fetching {code}...")
        
        # fina_indicator
        if code not in cache['fina_indicator'] or cache['fina_indicator'][code] is None:
            time.sleep(0.5)
            df = fetch_with_retry(pro.fina_indicator, ts_code=code)
            if df is not None:
                cache['fina_indicator'][code] = df
                print(f"  fina_indicator: {len(df)} rows")
        
        # balancesheet
        if code not in cache['balancesheet'] or cache['balancesheet'][code] is None:
            time.sleep(0.5)
            df = fetch_with_retry(pro.balancesheet, ts_code=code)
            if df is not None:
                cache['balancesheet'][code] = df
                print(f"  balancesheet: {len(df)} rows")
        
        # income
        if code not in cache['income'] or cache['income'][code] is None:
            time.sleep(0.5)
            df = fetch_with_retry(pro.income, ts_code=code)
            if df is not None:
                cache['income'][code] = df
                print(f"  income: {len(df)} rows")
        
        # cashflow
        if code not in cache['cashflow'] or cache['cashflow'][code] is None:
            time.sleep(0.5)
            df = fetch_with_retry(pro.cashflow, ts_code=code)
            if df is not None:
                cache['cashflow'][code] = df
                print(f"  cashflow: {len(df)} rows")
        
        # 每 10 只股票保存一次缓存
        if (idx + 1) % 10 == 0:
            pd.to_pickle(cache, cache_file)
            print(f"  Cache saved ({idx+1}/{len(codes)})")
    
    pd.to_pickle(cache, cache_file)
    print(f"\nAll done. Cache saved to {cache_file}")
    
    # 统计
    for key in cache:
        n = sum(1 for v in cache[key].values() if v is not None)
        print(f"  {key}: {n}/{len(codes)} codes fetched")


if __name__ == "__main__":
    main()
