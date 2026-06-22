"""
统一回测入口
示例：
    python scripts/run_backtest.py --strategy all
    python scripts/run_backtest.py --strategy original --top_n 10 --freq 5
    python scripts/run_backtest.py --strategy pure_ic --output_dir my_results
"""
import sys
import os
import argparse

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from factor_engine.config import get_config
from factor_engine.data.db import read_sql
from backtest.engine import BacktestEngine
from backtest.benchmark import BenchmarkEngine

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


STRATEGY_TABLES = {
    'original': 'stock_score_daily',
    'mined': 'stock_score_daily_v2',
    'pure_ic': 'stock_score_daily_pure_ic',
}


def run_single(strategy_name, top_n, freq, output_dir):
    """运行单个策略回测"""
    config = get_config()
    table = STRATEGY_TABLES[strategy_name]
    
    engine = BacktestEngine(
        strategy_name=strategy_name,
        score_table=table,
        top_n=top_n,
        rebalance_freq=freq
    )
    engine.run()
    engine.save(output_dir=output_dir)
    return engine


def run_all(top_n, freq, output_dir):
    """运行所有策略回测并生成对比图"""
    config = get_config()
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载得分
    scores = {}
    for name, table in STRATEGY_TABLES.items():
        df = read_sql(f"SELECT trade_date, code, score, rank_in_pool FROM {table} ORDER BY trade_date, rank_in_pool")
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        scores[name] = df
    
    # 收集所有代码用于加载价格
    all_codes = set()
    for df in scores.values():
        all_codes.update(df['code'].unique())
    
    # 运行策略回测
    results = {}
    metrics = {}
    for name in STRATEGY_TABLES.keys():
        print(f"\n### Running backtest: {name} ###")
        engine = BacktestEngine(
            strategy_name=name,
            score_df=scores[name],
            top_n=top_n,
            rebalance_freq=freq
        )
        engine.load_close_prices(codes=all_codes)
        engine.run()
        engine.save(output_dir=output_dir)
        results[name] = engine.records
        metrics[name] = engine.metrics
    
    # 计算 CSI500 等权基准
    print("\n### Calculating CSI500 equal-weight benchmark ###")
    bench_engine = BenchmarkEngine(name='CSI500_EW')
    # 使用第一个策略的得分日期作为基准日期
    first_name = list(STRATEGY_TABLES.keys())[0]
    bench_records = bench_engine.run_equal_weight(
        scores[first_name],
        rebalance_dates=results[first_name]['rebalance_date'].tolist(),
        rebalance_freq=freq
    )
    bench_records.to_csv(os.path.join(output_dir, 'csi500_ew_records.csv'), index=False)
    metrics['csi500_ew'] = BacktestEngine.calc_metrics(bench_records['nav'])
    
    # 汇总指标
    metrics_df = pd.DataFrame(metrics).T
    metrics_df = metrics_df[['total_return', 'annual_return', 'annual_volatility',
                             'sharpe_ratio', 'max_drawdown', 'calmar_ratio',
                             'win_rate', 'periods']]
    metrics_df.to_csv(os.path.join(output_dir, 'backtest_metrics_summary.csv'))
    print("\n" + metrics_df.to_string())
    
    # 绘制对比图
    plot_comparison(results, bench_records, output_dir)
    
    return results, metrics_df


def _merge_records(results, col):
    """把各策略的 records 按 rebalance_date 对齐合并。"""
    merged = None
    for name, records in results.items():
        sub = records[['rebalance_date', col]].rename(columns={col: name})
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on='rebalance_date', how='outer')
    merged = merged.sort_values('rebalance_date').reset_index(drop=True)
    return merged


def plot_comparison(results, bench_records, output_dir):
    """绘制多策略对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 净值曲线
    ax = axes[0, 0]
    for name, records in results.items():
        ax.plot(records['rebalance_date'], records['nav'], label=name, linewidth=1.5)
    ax.plot(bench_records['rebalance_date'], bench_records['nav'],
            label='CSI500 EW', linewidth=1.5, linestyle='--', color='black')
    ax.set_title('Strategy NAV Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('NAV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 每期收益（按共同日期对齐）
    ax = axes[0, 1]
    merged = _merge_records(results, 'net_return')
    names = [c for c in merged.columns if c != 'rebalance_date']
    width = 0.8 / len(names)
    x = range(len(merged))
    for i, name in enumerate(names):
        offset = (i - len(names) / 2) * width
        ax.bar([j + offset for j in x], merged[name] * 100,
               alpha=0.6, label=name, width=width)
    ax.set_title('Period Net Return (%)')
    ax.set_xlabel('Period')
    ax.set_ylabel('Return (%)')
    tick_step = max(1, len(merged) // 10)
    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([d.strftime('%m-%d') for d in merged['rebalance_date'][::tick_step]],
                       rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 回撤
    ax = axes[1, 0]
    for name, records in results.items():
        nav = records['nav']
        cummax = nav.cummax()
        dd = (nav - cummax) / cummax * 100
        ax.plot(records['rebalance_date'], dd, label=name, linewidth=1.5)
    bench_nav = bench_records['nav']
    bench_cummax = bench_nav.cummax()
    bench_dd = (bench_nav - bench_cummax) / bench_cummax * 100
    ax.plot(bench_records['rebalance_date'], bench_dd,
            label='CSI500 EW', linewidth=1.5, linestyle='--', color='black')
    ax.set_title('Drawdown (%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 换手率（按共同日期对齐）
    ax = axes[1, 1]
    merged_turn = _merge_records(results, 'turnover')
    names_turn = [c for c in merged_turn.columns if c != 'rebalance_date']
    for name in names_turn:
        ax.plot(merged_turn['rebalance_date'], merged_turn[name] * 100,
                label=name, linewidth=1.5, marker='o', markersize=2)
    ax.set_title('Turnover per Rebalance (%)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Turnover (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'backtest_comparison.png'), dpi=150)
    plt.close()
    print(f"  Comparison chart saved to {output_dir}/backtest_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Unified backtest runner')
    parser.add_argument('--strategy', type=str, default='all',
                        choices=list(STRATEGY_TABLES.keys()) + ['all'],
                        help='Strategy to backtest')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top stocks to hold')
    parser.add_argument('--freq', type=int, default=5,
                        help='Rebalance frequency in days')
    parser.add_argument('--output_dir', type=str, default='backtest_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    if args.strategy == 'all':
        run_all(args.top_n, args.freq, args.output_dir)
    else:
        engine = run_single(args.strategy, args.top_n, args.freq, args.output_dir)
        print("\nMetrics:")
        for k, v in engine.metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()
