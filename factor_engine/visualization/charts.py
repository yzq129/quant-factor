"""
可视化图表工具
统一封装因子分析相关的图表绘制
"""
import os
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from factor_engine.ic_analysis import calc_ic_series, calc_ic_stats, calc_factor_correlation

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_ic_analysis(df_proc, df_score, selected, ic_weights, final_weights,
                     ml_weights=None, output_dir='charts', title_prefix=''):
    """
    绘制 IC 分析全套图表
    
    Parameters
    ----------
    df_proc : DataFrame
        预处理后数据，包含 _neu 列
    df_score : DataFrame
        股票得分数据
    selected : list
        入选因子列表
    ic_weights : dict
        IC 权重
    final_weights : dict
        最终权重
    ml_weights : dict, optional
        LightGBM 权重，如果为 None 则不绘制 ML 饼图
    output_dir : str
        输出目录
    title_prefix : str
        图表标题前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{title_prefix}_" if title_prefix else ""
    
    # 每次更新前清理该策略目录下的旧图
    for old_path in glob.glob(os.path.join(output_dir, '*.png')):
        os.remove(old_path)
    
    # 1. IC 均值柱状图
    print("  -> ic_mean_bar.png")
    ic_stats_list = []
    for fac in selected:
        ic_df = calc_ic_series(df_proc, fac)
        stats = calc_ic_stats(ic_df)
        ic_stats_list.append({'factor': fac, 'ic_mean': stats['ic_mean'], 'ir': stats['ir']})
    ic_stats_df = pd.DataFrame(ic_stats_list).dropna(subset=['ic_mean'])
    
    fig, ax = plt.subplots(figsize=(12, max(5, len(selected) * 0.4)))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in ic_stats_df['ic_mean']]
    ax.barh(ic_stats_df['factor'], ic_stats_df['ic_mean'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean Rank IC')
    ax.set_title(f'{title_prefix} Factor Mean IC (Rank)'.strip())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}ic_mean_bar.png'), dpi=150)
    plt.close()
    
    # 2. IC 累积曲线
    print("  -> ic_cumsum_line.png")
    nrows = (len(selected) + 2) // 3
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(14, 3 * nrows))
    axes = axes.flatten() if len(selected) > 1 else [axes]
    for idx, fac in enumerate(selected):
        ic_df = calc_ic_series(df_proc, fac)
        ic_df['trade_date'] = pd.to_datetime(ic_df['trade_date'])
        ic_df = ic_df.sort_values('trade_date')
        ic_df['ic_cumsum'] = ic_df['ic'].fillna(0).cumsum()
        ax = axes[idx]
        ax.plot(ic_df['trade_date'], ic_df['ic_cumsum'], linewidth=1.2)
        ax.set_title(fac, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=30, labelsize=7)
    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}ic_cumsum_line.png'), dpi=150)
    plt.close()
    
    # 3. 因子权重饼图
    print("  -> factor_weight_pie.png")
    if ml_weights:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # IC权重
        ic_w = {f: ic_weights.get(f, 0) for f in selected}
        axes[0].pie(ic_w.values(), labels=ic_w.keys(), autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 7})
        axes[0].set_title('IC Weight (Softmax)', fontsize=10)
        
        # ML权重
        ml_w_clean = {f: ml_weights.get(f'{f}_neu', 0) for f in selected}
        axes[1].pie(ml_w_clean.values(), labels=ml_w_clean.keys(), autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 7})
        axes[1].set_title('LightGBM Weight', fontsize=10)
        
        # 最终权重
        axes[2].pie(final_weights.values(), labels=final_weights.keys(), autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 7})
        axes[2].set_title('Final Combined Weight', fontsize=10)
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ic_w = {f: ic_weights.get(f, 0) for f in selected}
        ax.pie(ic_w.values(), labels=ic_w.keys(), autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9})
        ax.set_title(f'{title_prefix} IC Weight (Softmax)'.strip(), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}factor_weight_pie.png'), dpi=150)
    plt.close()
    
    # 4. 最新一期得分分布
    print("  -> score_dist_hist.png")
    latest_date = df_score['trade_date'].max()
    latest_scores = df_score[df_score['trade_date'] == latest_date]['score']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(latest_scores, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(latest_scores.mean(), color='red', linestyle='--',
               label=f'Mean={latest_scores.mean():.4f}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_title(f'{title_prefix} Score Distribution ({latest_date})'.strip())
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}score_dist_hist.png'), dpi=150)
    plt.close()
    
    # 5. Top10 股票得分条形图（带股票名称）
    print("  -> top10_score_bar.png")
    top10 = df_score[df_score['trade_date'] == latest_date].head(10)
    
    name_map = {}
    stock_basic_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'stock_basic.csv')
    if os.path.exists(stock_basic_path):
        sb = pd.read_csv(stock_basic_path)
        name_map = dict(zip(sb['ts_code'], sb['name']))
    
    labels = []
    for code in top10['code']:
        name = name_map.get(code, '')
        labels.append(f"{code}  {name}" if name else code)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors_top10 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))[::-1]
    ax.barh(labels[::-1], top10['score'][::-1], color=colors_top10)
    ax.set_xlabel('Score')
    ax.set_title(f'{title_prefix} Top 10 Stocks ({latest_date})'.strip())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}top10_score_bar.png'), dpi=150)
    plt.close()
    
    # 6. 因子相关系数热力图
    print("  -> corr_heatmap.png")
    corr_mat = calc_factor_correlation(df_proc)
    if not corr_mat.empty:
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'{title_prefix} Factor Correlation Matrix (Neutralized)'.strip())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}corr_heatmap.png'), dpi=150)
        plt.close()
    
    print(f"  All charts saved to: {output_dir}")
