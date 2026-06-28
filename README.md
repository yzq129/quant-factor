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
| Original | 29 个基础因子 + 滚动 LightGBM | `stock_score_daily` |
| Mined (V2) | 29 个基础因子 + 时序挖掘因子 + 滚动 LightGBM | `stock_score_daily_v2` |
| Pure IC | 29 个基础因子，仅 IC 加权 | `stock_score_daily_pure_ic` |

## 数据

- **本地行情**：`全A日K/2024.zip`、`2025.zip`、`2026.zip`
- **财务/市值数据**：Tushare Pro API
- **因子数据**：MySQL `vnpy` 数据库

## 重要提示

本项目在因子构建与模型训练环节已尽量避免未来函数：

- **动态股票池**：`factor_raw_daily` 基于月末 CSI500 成分股截面动态复用，避免使用固定未来成分股。
- **滚动 LightGBM**：`Original` 与 `Mined` 策略在每个交易日使用过去 N 个交易日数据训练模型，杜绝全样本训练。
- **时序挖掘因子**：`factor_mining_v2` 仅基于历史滚动窗口生成均值/标准差/趋势等时序特征，不再基于全历史 IC 筛选因子。

> **本分支为未行业中性化版本**：预处理阶段未按行业分组做市值中性化，组合在行业上存在偏置（例如银行、证券占比可能较高），回测收益中包含较多行业 beta 暴露。若需查看行业中性化版本，可在 `industry-neutralized` 标签/分支中对比。

> **仍需注意**：`Pure_IC` 策略的因子权重仍基于全历史 IC/IR 估计；`Mined` 策略端在因子筛选时同样使用全历史 IC。回测结果仅供参考，不代表实盘可复现收益。

## 依赖

- Python 3.13.8（vnpy 4.3.0 环境）
- pandas / numpy / matplotlib / seaborn
- lightgbm / xgboost（可选，用于模型训练）
- pymysql / sqlalchemy
- tushare

## 许可证

仅供研究与学习使用。
