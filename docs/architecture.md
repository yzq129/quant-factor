# 项目架构说明

## 1. 目录结构

```
聚宽/
├── config/                      # 配置管理
│   ├── config.yaml              # 项目主配置（不含敏感信息）
│   ├── secrets.yaml             # 敏感凭据
│   └── __init__.py
├── factor_engine/               # 因子研究与策略实现
│   ├── config.py                # 配置加载器
│   ├── factor_calc.py           # 因子计算（29个基础因子）
│   ├── factor_mining_v2.py      # 因子挖掘
│   ├── preprocess.py            # 数据预处理
│   ├── ic_analysis.py           # IC 计算与因子筛选
│   ├── ml_model.py              # LightGBM 训练与打分
│   ├── database.py              # 原 MySQL 工具（保留兼容）
│   ├── data/                    # 重构后的数据层
│   │   ├── db.py                # 新数据库抽象
│   │   └── __init__.py
│   ├── strategy/                # 策略层
│   │   ├── base.py              # 策略基类
│   │   ├── original.py          # V1 原始策略
│   │   ├── mined.py             # V2 挖掘增强策略
│   │   ├── pure_ic.py           # 纯 IC 加权策略
│   │   └── __init__.py
│   ├── utils/                   # 工具层
│   │   ├── logger.py            # 统一日志
│   │   └── __init__.py
│   └── visualization/           # 可视化
│       ├── charts.py            # 统一图表绘制
│       └── __init__.py
├── backtest/                    # 回测引擎
│   ├── engine.py                # 回测引擎
│   ├── benchmark.py             # 基准计算
│   └── __init__.py
├── scripts/                     # 统一执行脚本
│   ├── run_pipeline.py          # 运行策略流水线
│   ├── run_backtest.py          # 运行回测
│   └── __init__.py
├── logs/                        # 日志输出
├── docs/                        # 文档
│   ├── architecture.md          # 本文件
│   └── usage.md                 # 使用说明
└── README.md                    # 项目总览
```

## 2. 数据流

```
原始数据（Tushare / 本地 zip）
    │
    ▼
factor_raw_daily  ──►  factor_engine.data.db.read_sql()
    │
    ▼
预处理（MAD 去极值 → 市值中性化 → Z-Score；本版本未做行业中性化）
    │
    ▼
factor_processed_daily / factor_processed_daily_v2
    │
    ▼
IC 分析 + 相关性过滤 + VIF
    │
    ▼
入选因子 + 权重
    │
    ├──────► LightGBM 训练（Original / Mined）
    │              │
    │              ▼
    │        ML 权重
    │              │
    ▼              ▼
最终权重（IC 50% + ML 50%） / IC 权重（Pure IC）
    │
    ▼
股票得分 stock_score_daily / stock_score_daily_v2 / stock_score_daily_pure_ic
    │
    ▼
回测引擎 backtest.engine.BacktestEngine
    │
    ▼
回测结果 CSV + 图表
```

## 3. 核心模块职责

### `factor_engine.config`
- 加载 `config/config.yaml` 和 `config/secrets.yaml`
- 支持点号路径访问：`config.get('db.host')`
- 敏感信息与主配置分离

### `factor_engine.data.db`
- 提供 `save_dataframe()` / `read_sql()` / `get_trade_dates()` 等
- 基于 PyMySQL + SQLAlchemy
- 自动建表、批量插入、NaN 处理

### `factor_engine.strategy.base`
- 定义策略统一生命周期：
  1. `load_data()` —— 加载数据
  2. `preprocess()` —— 预处理
  3. `select_factors()` —— IC 筛选
  4. `align_factor_directions()` —— 负 IC 翻转
  5. `train_ml_model()` —— LightGBM（可选）
  6. `calc_scores()` —— 计算得分
  7. `save_results()` —— 保存结果
  8. `plot()` —— 绘制图表
- 子类只需覆盖少量方法即可定义新策略

### `factor_engine.strategy.original / mined / pure_ic`
- `OriginalStrategy`: 29 个基础因子 + 滚动 LightGBM，表后缀 `''`
- `MinedStrategy`: 29 个基础因子 + 时序挖掘因子 + 滚动 LightGBM，表后缀 `_v2`
- `PureICStrategy`: 29 个基础因子，仅 IC 权重，表后缀 `_pure_ic`

### `backtest.engine`
- `BacktestEngine`: 事件驱动回测
- 支持 Top N 等权、自定义调仓频率、交易成本
- 输出净值曲线、回撤、换手率等

### `backtest.benchmark`
- `BenchmarkEngine`: 基于 score_df 中的成分股计算等权基准
- 默认计算 CSI500 等权基准

## 4. 已知限制

- 本项目已尽量避免主要未来函数：`factor_raw_daily` 使用动态 CSI500 成分股池，`Original` / `Mined` 的 LightGBM 使用滚动训练，`factor_mining_v2` 仅基于历史窗口生成时序特征。
- **纯 IC 策略** 仍使用全历史 IC 权重估计，严格样本外可进一步改为滚动窗口 IC。
- 本版本 **未做行业中性化**，组合存在行业偏置，回测收益中包含行业 beta 暴露。
- 回测结果仅供参考，不代表实盘可复现结果。
