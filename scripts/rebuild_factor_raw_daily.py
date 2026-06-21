"""
用动态 CSI500 股票池 + 本地日 K + 历史财报重建 factor_raw_daily。
彻底消除股票池/财务数据未来函数。
"""
import os
import sys
import io
import zipfile
import pickle
from datetime import datetime, date

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factor_engine.database import save_dataframe, execute, read_sql

UNIVERSE_CSV = 'config/csi500_universe_history.csv'
FINANCIALS_PKL = 'cache/financials_all.pkl'
ZIP_PATHS = ['全A日K/2024.zip', '全A日K/2025.zip', '全A日K/2026.zip']

# 财报字段
FINA_COLS_NEED = {
    'fina_indicator': ['netprofit_yoy', 'op_yoy', 'current_ratio'],
    'balancesheet': ['total_assets', 'total_cur_assets', 'total_cur_liab'],
    'income': ['revenue', 'operate_profit'],
    'cashflow': ['n_cashflow_act', 'c_pay_acq_const_fiolta', 'free_cashflow'],
}

ALL_FIN_COLS = ['code'] + [c for cols in FINA_COLS_NEED.values() for c in cols]

OUTPUT_COLS = [
    'trade_date', 'code', 'market_cap', 'future_5d_return',
    'bp_lr', 'ep_deducted_ttm', 'fcfp_ttm', 'ocfp_ttm',
    'amount_mean_20d', 'asset_ln', 'revenues_ln', 'currentratio',
    'ocf_to_operating_profit', 'price_chg1200d', 'price_chg120d', 'price_chg180d',
    'capex2sales', 'netincome_chg1y', 'op_profit_chg1y'
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_universe():
    """加载历史 CSI500 成分股月末截面。"""
    df = pd.read_csv(UNIVERSE_CSV)
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
    return df[['trade_date', 'code']]


def get_universe_for_date(universe_df, t):
    """交易日 t 使用 <= t 的最新月末截面（避免用未来月份成分股）。"""
    avail = universe_df[universe_df['trade_date'] <= t]
    if avail.empty:
        return []
    latest = avail['trade_date'].max()
    return avail[avail['trade_date'] == latest]['code'].unique().tolist()


def load_prices(codes_set):
    """从本地 zip 读取指定股票池的全部日 K。"""
    records = []
    for zpath in ZIP_PATHS:
        if not os.path.exists(zpath):
            log(f"Skip missing zip: {zpath}")
            continue
        with zipfile.ZipFile(zpath, 'r') as z:
            # 构建文件名 -> code 映射
            mapping = {}
            for fname in z.namelist():
                if not fname.endswith('.csv') or '__MACOSX' in fname:
                    continue
                code = os.path.basename(fname).replace('.csv', '')
                if code in codes_set:
                    mapping[code] = fname
            log(f"{zpath}: found {len(mapping)} target files")
            for code, fname in mapping.items():
                raw = z.read(fname)
                try:
                    text = raw.decode('utf-8-sig')
                except UnicodeDecodeError:
                    text = raw.decode('utf-8-sig', errors='replace')
                df = pd.read_csv(io.StringIO(text))
                df['code'] = code
                df['trade_date'] = pd.to_datetime(df['datetime'])
                keep = ['trade_date', 'code', 'close', 'amount', 'total_mv', 'pe_ttm', 'pb']
                df = df[[c for c in keep if c in df.columns]].copy()
                records.append(df)
    return pd.concat(records, ignore_index=True)


def _to_date(s):
    """将 20240101 / '20240101' / Timestamp 统一转为 date。"""
    if pd.isna(s):
        return pd.NaT
    if isinstance(s, date):
        return s
    if isinstance(s, pd.Timestamp):
        return s.date()
    s = str(int(s)) if isinstance(s, (int, float, np.integer, np.floating)) else str(s)
    return datetime.strptime(s, '%Y%m%d').date()


def load_financials():
    """加载财务缓存并合并为按表的大表。"""
    with open(FINANCIALS_PKL, 'rb') as f:
        data = pickle.load(f)

    result = {}
    for table, need_cols in FINA_COLS_NEED.items():
        rows = []
        for code, df in data[table].items():
            if df is None or df.empty:
                continue
            df = df.copy()
            df['code'] = code
            rows.append(df)
        if not rows:
            result[table] = pd.DataFrame(columns=['code'] + need_cols)
            continue
        big = pd.concat(rows, ignore_index=True)
        # 选择需要的列 + 公告日期列
        date_cols = ['ann_date']
        if 'f_ann_date' in big.columns:
            date_cols.append('f_ann_date')
        avail = [c for c in ['code'] + date_cols + need_cols if c in big.columns]
        big = big[avail].copy()
        for c in date_cols:
            big[c] = big[c].apply(_to_date)
        # 剔除公告日期缺失的记录，否则 merge_asof 会报错
        big = big.dropna(subset=date_cols)
        # 按公告日期去重，保留 end_date 最大的一条
        if 'end_date' in big.columns:
            big = big.sort_values(['code'] + date_cols + ['end_date'])
        dedup_by = ['code'] + date_cols
        big = big.drop_duplicates(subset=dedup_by, keep='last')
        result[table] = big
    return result


def expand_financials_to_daily(fin_dfs, trade_dates):
    """把财报展开到每个交易日：取公告日 <= 交易日的最新一条。"""
    all_dates = pd.DataFrame({'trade_date': pd.to_datetime(trade_dates)})
    daily_list = []

    for table, need_cols in FINA_COLS_NEED.items():
        df = fin_dfs[table]
        if df.empty:
            continue
        # 使用 f_ann_date 优先，否则 ann_date
        date_col = 'f_ann_date' if 'f_ann_date' in df.columns else 'ann_date'
        df = df.copy()
        df['ann_dt'] = pd.to_datetime(df[date_col])
        df = df.sort_values(['code', 'ann_dt'])

        pieces = []
        for code, g in df.groupby('code'):
            g = g.sort_values('ann_dt')
            merged = pd.merge_asof(
                all_dates, g,
                left_on='trade_date', right_on='ann_dt',
                direction='backward'
            )
            merged['code'] = code
            keep = ['trade_date', 'code'] + need_cols
            pieces.append(merged[[c for c in keep if c in merged.columns]])

        if pieces:
            daily = pd.concat(pieces, ignore_index=True)
            daily['trade_date'] = daily['trade_date'].dt.date
            daily_list.append(daily)

    if not daily_list:
        return pd.DataFrame(columns=['trade_date', 'code'] + ALL_FIN_COLS[1:])

    merged = daily_list[0]
    for d in daily_list[1:]:
        merged = merged.merge(d, on=['trade_date', 'code'], how='outer')
    return merged


def calc_factors(price_df, fin_daily, universe_df):
    """逐日计算因子。"""
    price_df = price_df.copy()
    price_df['trade_date_d'] = price_df['trade_date'].dt.date

    # 预先把价格数据按股票分组，加速逐股查询
    price_groups = {
        code: g.sort_values('trade_date').reset_index(drop=True)
        for code, g in price_df.groupby('code')
    }

    trade_dates = sorted(price_df['trade_date_d'].unique())
    # 从第一个有 universe 的交易日开始；最后一个 universe 之后的交易日复用最后一期截面
    start_u = universe_df['trade_date'].min()
    trade_dates = [d for d in trade_dates if d >= start_u]
    log(f"Calc factors for {len(trade_dates)} trade dates")

    # 预先把 fin_daily 按 trade_date 分组
    fin_by_date = {d: g for d, g in fin_daily.groupby('trade_date')}

    results = []
    for i, t in enumerate(trade_dates, 1):
        codes = get_universe_for_date(universe_df, t)
        if not codes:
            continue

        today = price_df[(price_df['trade_date_d'] == t) & (price_df['code'].isin(codes))]
        if today.empty:
            continue

        # 合并当日财务数据
        fin_t = fin_by_date.get(t)
        if fin_t is not None:
            today = today.merge(fin_t, on='code', how='left')

        # 每只股票分别计算动量/未来收益
        day_rows = []
        for _, row in today.iterrows():
            code = row['code']
            sub = price_groups.get(code)
            if sub is None:
                continue
            pos = sub[sub['trade_date_d'] == t].index
            if len(pos) == 0:
                continue
            pos = pos[0]

            close = row['close']
            market_cap = row.get('total_mv', np.nan)
            if pd.notna(market_cap):
                market_cap = float(market_cap) / 10000.0

            r = {
                'trade_date': t,
                'code': code,
                'market_cap': market_cap,
                'bp_lr': 1.0 / row['pb'] if pd.notna(row.get('pb')) and row['pb'] != 0 else np.nan,
                'ep_deducted_ttm': 1.0 / row['pe_ttm'] if pd.notna(row.get('pe_ttm')) and row['pe_ttm'] != 0 else np.nan,
            }

            closes = sub['close'].values
            amounts = sub['amount'].values
            r['price_chg120d'] = close / closes[pos - 120] - 1 if pos >= 120 else np.nan
            r['price_chg180d'] = close / closes[pos - 180] - 1 if pos >= 180 else np.nan
            r['price_chg1200d'] = close / closes[pos - 1200] - 1 if pos >= 1200 else np.nan
            r['amount_mean_20d'] = float(np.mean(amounts[max(0, pos - 19):pos + 1])) if pos >= 0 else np.nan

            if pos + 5 < len(closes):
                r['future_5d_return'] = closes[pos + 5] / close - 1
            else:
                r['future_5d_return'] = np.nan

            # 财务字段
            for c in ALL_FIN_COLS[1:]:
                r[c] = row.get(c, np.nan)

            day_rows.append(r)

        if not day_rows:
            continue
        results.append(pd.DataFrame(day_rows))
        if i % 20 == 0 or i == len(trade_dates):
            log(f"  {i}/{len(trade_dates)} {t} -> {len(day_rows)} rows")

    df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return df


def compute_derived_factors(df):
    """计算价值/质量/成长等衍生因子。"""
    df = df.copy()
    df['fcfp_ttm'] = df['free_cashflow'] / (df['market_cap'].replace(0, np.nan) * 1e8)
    df['ocfp_ttm'] = df['n_cashflow_act'] / (df['market_cap'].replace(0, np.nan) * 1e8)
    df['asset_ln'] = np.log(df['total_assets'].replace(0, np.nan))
    df['revenues_ln'] = np.log(df['revenue'].replace(0, np.nan))
    df['currentratio'] = df['total_cur_assets'] / df['total_cur_liab'].replace(0, np.nan)
    df['ocf_to_operating_profit'] = df['n_cashflow_act'] / df['operate_profit'].replace(0, np.nan)
    df['capex2sales'] = df['c_pay_acq_const_fiolta'] / df['revenue'].replace(0, np.nan)
    df['netincome_chg1y'] = df['netprofit_yoy'] / 100.0
    df['op_profit_chg1y'] = df['op_yoy'] / 100.0
    return df


def main():
    log("Loading CSI500 universe history...")
    universe_df = load_universe()
    log(f"  universe dates: {universe_df['trade_date'].min()} ~ {universe_df['trade_date'].max()}")

    all_codes = set(universe_df['code'].unique())
    log(f"  unique codes: {len(all_codes)}")

    log("Loading local price data...")
    price_df = load_prices(all_codes)
    log(f"  price records: {len(price_df)}, codes: {price_df['code'].nunique()}, dates: {price_df['trade_date'].dt.date.nunique()}")

    log("Loading financial statements...")
    fin_dfs = load_financials()

    trade_dates = sorted(price_df['trade_date'].dt.date.unique())
    log("Expanding financials to daily...")
    fin_daily = expand_financials_to_daily(fin_dfs, trade_dates)
    log(f"  fin_daily: {len(fin_daily)} rows")

    log("Calculating factors...")
    df = calc_factors(price_df, fin_daily, universe_df)
    log(f"  raw factors: {len(df)} rows, {df['trade_date'].nunique()} dates")

    log("Computing derived factors...")
    df = compute_derived_factors(df)

    out = df[[c for c in OUTPUT_COLS if c in df.columns]].copy()
    out['trade_date'] = pd.to_datetime(out['trade_date']).dt.date

    log("Saving to database...")
    execute("DROP TABLE IF EXISTS factor_raw_daily")
    save_dataframe(out, 'factor_raw_daily', if_exists='replace')
    log(f"[OK] Rebuilt factor_raw_daily: {len(out)} rows, {out['trade_date'].nunique()} dates")


if __name__ == '__main__':
    main()
