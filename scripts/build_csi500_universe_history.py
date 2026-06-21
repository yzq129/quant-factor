"""
构建历史中证500成分股动态股票池
从 Tushare 获取每月末/调整日的 CSI500 成分股
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tushare as ts
from datetime import datetime, timedelta

from factor_engine.config import get_config

TS_TOKEN = "8e84ac848657f57f4abba76b7d7053f04d546923a1f43414017a95a3"
pro = ts.pro_api(TS_TOKEN, timeout=30)


def get_last_trade_date(year, month):
    """获取某月最后一个交易日"""
    start = f"{year}{month:02d}01"
    # 下月第一天
    if month == 12:
        next_month = f"{year+1}0101"
    else:
        next_month = f"{year}{month+1:02d}01"
    
    df = pro.trade_cal(exchange='SSE', start_date=start, end_date=next_month)
    df = df[df['is_open'] == 1].sort_values('cal_date')
    if df.empty:
        return None
    # 取小于 next_month 的最大日期
    last = df[df['cal_date'] < next_month]['cal_date'].max()
    return last


def get_csi500_for_date(trade_date):
    """查询某日期的 CSI500 成分股，返回 list 或 None"""
    try:
        df = pro.index_weight(index_code='000905.SH', trade_date=trade_date)
        if df.empty:
            return None
        return df['con_code'].unique().tolist()
    except Exception as e:
        print(f"  Error querying {trade_date}: {e}")
        return None


def build_universe_history(start_year=2024, start_month=6, end_year=2026, end_month=6):
    """构建历史成分股数据"""
    records = []
    
    # 生成月份列表
    months = []
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    
    for y, m in months:
        trade_date = get_last_trade_date(y, m)
        if trade_date is None:
            continue
        
        print(f"Querying CSI500 for {y}-{m:02d} (trade_date={trade_date})...")
        codes = get_csi500_for_date(trade_date)
        
        if codes is None:
            print(f"  No data for {trade_date}")
            continue
        
        print(f"  Got {len(codes)} stocks")
        for code in codes:
            records.append({
                'trade_date': trade_date,
                'code': code
            })
    
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = build_universe_history()
    output_path = 'config/csi500_universe_history.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} records to {output_path}")
    print("Available dates:")
    print(df.groupby('trade_date').size())
