"""
中证500成分股动态调整工具
从 Tushare 获取指定日期的成分股，并与当前数据库中的股票池对比
"""
import sys
import os
import argparse
from datetime import datetime, timedelta

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from factor_engine.config import get_config
from factor_engine.data.db import read_sql, get_latest_date, execute


def get_tushare_pro():
    """获取 Tushare Pro 接口"""
    import tushare as ts
    token = get_config().get_tushare_token()
    pro = ts.pro_api(token)
    return pro


def get_csi500_stocks(trade_date, fallback_dates=None):
    """
    获取指定日期的中证500成分股
    如果指定日期没有数据，尝试 fallback_dates
    """
    pro = get_tushare_pro()
    date_str = trade_date.replace('-', '')
    
    candidates = [date_str]
    if fallback_dates:
        candidates.extend([d.replace('-', '') for d in fallback_dates])
    
    for d in candidates:
        df = pro.index_weight(index_code='000905.SH', trade_date=d)
        if not df.empty:
            print(f"[OK] Got CSI500 constituents for {d}: {len(df)} stocks")
            return df['con_code'].unique().tolist()
        print(f"[WARN] No CSI500 data for {d}, try next")
    
    return []


def get_current_universe():
    """获取当前数据库中的股票池"""
    latest = get_latest_date('factor_raw_daily')
    query = f"""
    SELECT DISTINCT code FROM factor_raw_daily
    WHERE trade_date = '{latest}'
    """
    df = read_sql(query)
    return set(df['code'].tolist()), latest


def main():
    parser = argparse.ArgumentParser(description='Update CSI500 universe')
    parser.add_argument('--date', type=str, default=None,
                        help='Target date (YYYY-MM-DD). Default: latest date in database')
    args = parser.parse_args()
    
    if args.date:
        target_date = args.date
    else:
        target_date = get_latest_date('factor_raw_daily')
    
    print(f"Target date: {target_date}")
    
    # 获取最新成分股
    # 通常成分股调整在每月第二个周五的下一个交易日生效
    # 如果目标日期没有数据，尝试往前推几个交易日
    fallback = [(pd.to_datetime(target_date) - timedelta(days=i)).strftime('%Y-%m-%d')
                for i in range(1, 15)]
    new_universe = set(get_csi500_stocks(target_date, fallback_dates=fallback))
    
    if not new_universe:
        print("[ERROR] Failed to fetch CSI500 constituents")
        return
    
    # 获取当前股票池
    current_universe, current_date = get_current_universe()
    
    print(f"\nCurrent universe date: {current_date}")
    print(f"Current universe size: {len(current_universe)}")
    print(f"New universe size: {len(new_universe)}")
    
    added = new_universe - current_universe
    removed = current_universe - new_universe
    unchanged = new_universe & current_universe
    
    print(f"\nUnchanged: {len(unchanged)}")
    print(f"Added: {len(added)}")
    if added:
        print("  " + ", ".join(sorted(added)[:20]) + (" ..." if len(added) > 20 else ""))
    
    print(f"Removed: {len(removed)}")
    if removed:
        print("  " + ", ".join(sorted(removed)[:20]) + (" ..." if len(removed) > 20 else ""))
    
    # 保存新股票池到临时文件
    output_dir = 'config'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'csi500_universe_{target_date}.txt')
    with open(output_file, 'w') as f:
        for code in sorted(new_universe):
            f.write(code + '\n')
    print(f"\nSaved new universe to {output_file}")


if __name__ == '__main__':
    main()
