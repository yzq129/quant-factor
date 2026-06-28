# 使用说明

## 1. 环境准备

### 1.1 Python 环境

项目使用 `vnpy` 自带的 Python 解释器：

```bash
D:\vnstudio2026\python.exe --version
```

### 1.2 数据库

MySQL 配置保存在 `config/secrets.yaml`（默认未提交，请手动创建）：

```yaml
db:
  host: localhost
  port: 3306
  user: root
  password: your_password
  database: vnpy

tushare:
  token: your_tushare_token
```

项目数据库为 `vnpy`，主要表包括：

- `factor_raw_daily` —— 原始日频因子
- `factor_processed_daily` / `_v2` / `_pure_ic` —— 预处理后因子
- `factor_ic_monthly` / `_v2` / `_pure_ic` —— 月度 IC
- `factor_selected` / `_v2` / `_pure_ic` —— 入选因子
- `stock_score_daily` / `_v2` / `_pure_ic` —— 股票得分
- `factor_mined_daily` —— 挖掘因子

## 2. 运行策略流水线

### 2.1 运行单个策略

```bash
D:\vnstudio2026\python.exe scripts/run_pipeline.py original
D:\vnstudio2026\python.exe scripts/run_pipeline.py mined
D:\vnstudio2026\python.exe scripts/run_pipeline.py pure_ic
```

### 2.2 运行全部策略

```bash
D:\vnstudio2026\python.exe scripts/run_pipeline.py all
```

> 注意：`mined` 策略依赖 `factor_processed_daily` 和 `factor_mined_daily`，请先确保 `original` 策略和因子挖掘已生成数据。`factor_mining_v2` 仅生成历史时序特征，不再基于全历史 IC 筛选因子。

## 3. 运行回测

### 3.1 回测全部策略（含 CSI500 等权基准）

```bash
D:\vnstudio2026\python.exe scripts/run_backtest.py --strategy all
```

### 3.2 回测单个策略

```bash
D:\vnstudio2026\python.exe scripts/run_backtest.py --strategy original --top_n 10 --freq 5
```

### 3.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--strategy` | str | `all` | 策略名：`original`, `mined`, `pure_ic`, `all` |
| `--top_n` | int | 10 | 每期持仓股票数量 |
| `--freq` | int | 5 | 调仓频率（天） |
| `--output_dir` | str | `backtest_results` | 结果输出目录 |

### 3.4 回测输出

```
backtest_results/
├── original_records.csv
├── original_metrics.csv
├── mined_records.csv
├── mined_metrics.csv
├── pure_ic_records.csv
├── pure_ic_metrics.csv
├── csi500_ew_records.csv
└── backtest_metrics_summary.csv
└── backtest_comparison.png
```

## 4. 配置说明

### 4.1 `config/config.yaml`

```yaml
db:
  host: localhost
  port: 3306
  database: vnpy

tushare:
  token: ""  # 在 secrets.yaml 中覆盖

factors:
  corr_thresh: 0.7
  vif_thresh: 5.0
  min_ic_months: 2

backtest:
  commission_rate: 0.0003
  stamp_tax_rate: 0.001
  slippage_rate: 0.0005

data:
  local_zip_paths:
    - 全A日K/2024.zip
    - 全A日K/2025.zip
    - 全A日K/2026.zip

paths:
  charts_dir: charts
  logs_dir: logs
```

### 4.2 `config/secrets.yaml`

敏感配置单独存放，**不要提交到版本控制**。

```yaml
db:
  user: root
  password: your_password

tushare:
  token: your_tushare_token
```

## 5. 扩展新策略

继承 `factor_engine.strategy.base.BaseStrategy`，覆盖以下方法：

```python
from factor_engine.strategy.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name='MyStrategy')
    
    def get_factor_names(self):
        return ['factor_a', 'factor_b']
    
    def get_table_suffix(self):
        return '_my'
    
    def use_ml(self):
        return True
```

然后在 `scripts/run_pipeline.py` 的 `STRATEGIES` 字典中注册即可。

## 6. 常见问题

### Q1: 运行 `mined` 策略时报 `factor_mined_daily is empty`

需要先运行因子挖掘脚本：

```bash
D:\vnstudio2026\python.exe factor_engine/factor_mining_v2.py
```

### Q2: 回测收益能否直接外推？

本项目已修复主要未来函数：动态 CSI500 成分股池、滚动 LightGBM 训练、时序挖掘因子仅使用历史窗口。但 `Pure_IC` 与 `Mined` 的因子筛选仍基于全历史 IC 权重；且本版本未做行业中性化，组合存在行业 beta 暴露。回测结果仅供参考，详情请参阅 `docs/architecture.md` 的“已知限制”章节。

### Q3: 如何查看日志？

日志默认输出到 `logs/` 目录，按模块分类：

```
logs/
├── strategy.original.log
├── strategy.mined.log
├── strategy.pure_ic.log
├── backtest.original.log
└── ...
```
