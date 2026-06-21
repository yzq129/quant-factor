# 中证500多因子量化策略

本项目是一个围绕中证500成分股的多因子选股研究与回测框架，包含因子计算、因子挖掘、IC分析、机器学习选股、事件驱动回测等完整链路。

## 快速开始

```bash
# 运行策略流水线
D:\vnstudio2026\python.exe scripts/run_pipeline.py all

# 运行回测
D:\vnstudio2026\python.exe scripts/run_backtest.py --strategy all
```

更多使用细节请参见 [docs/usage.md](docs/usage.md)。

## 项目结构

```
聚宽/
├── config/              # 配置与凭据
├── factor_engine/       # 因子研究与策略
├── backtest/            # 回测引擎与基准
├── scripts/             # 统一执行入口
├── docs/                # 文档
└── logs/                # 运行日志
```

架构说明请参见 [docs/architecture.md](docs/architecture.md)。

## 当前策略

| 策略 | 说明 | 输出表 |
|------|------|--------|
| Original (V1) | 15 个基础因子 + LightGBM | `stock_score_daily` |
| Mined (V2) | 15 个基础因子 + 挖掘因子 + LightGBM | `stock_score_daily_v2` |
| Pure IC | 15 个基础因子，仅 IC 加权 | `stock_score_daily_pure_ic` |

## 数据

- **本地行情**：`全A日K/2024.zip`、`2025.zip`、`2026.zip`
- **财务/市值数据**：Tushare Pro API
- **因子数据**：MySQL `vnpy` 数据库

## 重要提示

> **当前版本存在未来函数问题**：LightGBM 在全样本上训练，CSI500 成分股使用固定日期（2026-05-29）。这导致 Original 和 Mined 策略的回测收益显著偏高，**不代表实盘可复现收益**。该问题已记录在案，计划在下一阶段通过滚动训练和动态成分股解决。

Pure IC 策略用于剥离 LightGBM 的贡献，但仍使用全样本 IC 权重和固定股票池，仅作对比参考。

## 依赖

- Python 3.13.8（vnpy 4.3.0 环境）
- pandas / numpy / matplotlib / seaborn
- lightgbm / xgboost（可选，用于模型训练）
- pymysql / sqlalchemy
- tushare

## 许可证

仅供研究与学习使用。
