# AI Bubble Quantitative Assessment Framework

AI 泡沫量化评估与做空策略框架

## 项目结构

```
ai-bubble-quant/
├── data/              # 数据获取与存储
│   ├── fetcher.py     # 外部数据拉取（股价、财报、宏观指标、情绪数据）
│   ├── storage.py     # 数据缓存与本地存储（SQLite/Parquet）
│   └── sources.py     # 数据源配置（YFinance, FRED, Alpha Vantage 等）
├── analysis/          # 量化分析引擎
│   ├── bubble_metrics.py    # 泡沫指标（Shiller PE, Buffett Indicator, NVRRI 等）
│   ├── technical.py         # 技术分析（RSI, MACD, 布林带, 动量）
│   ├── sentiment.py         # 市场情绪分析（VIX, 新闻情绪, 搜索指数）
│   ├── fundamental.py       # 基本面分析（估值比率, 盈利质量, 收入增速）
│   └── composite.py         # 综合评分模型
├── strategy/          # 交易决策引擎
│   ├── signals.py     # 做空信号生成
│   ├── sizing.py      # 仓位管理与风险计算
│   ├── portfolio.py   # 组合构建（标的筛选, 权重分配）
│   └── risk.py        # 风控模块（止损, 最大回撤, 杠杆限制）
├── backtest/          # 回测引擎
│   ├── engine.py      # 回测执行器
│   ├── metrics.py     # 绩效评估（Sharpe, Sortino, Max Drawdown, Win Rate）
│   └── visualization.py  # 结果可视化
├── config/            # 配置文件
│   ├── settings.yaml  # 全局配置
│   └── tickers.yaml   # 监控标的列表
├── notebooks/         # Jupyter 探索性分析
├── main.py            # 入口脚本
└── requirements.txt   # 依赖
```

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整流程
python main.py

# 仅获取数据
python main.py --step fetch

# 仅运行分析
python main.py --step analyze

# 仅运行回测
python main.py --step backtest
```

## 开发路线图

- [ ] Phase 1: 数据基础设施（价格、财报、宏观数据）
- [ ] Phase 2: 泡沫指标体系搭建
- [ ] Phase 3: 做空信号生成逻辑
- [ ] Phase 4: 回测引擎与绩效评估
- [ ] Phase 5: 实时监控与预警
- [ ] Phase 6: 情绪面数据集成（新闻、社交媒体）
