# Google Play 数据质量处理与智能体分群实验

用同一份 Google Play 历史数据，对比**最低必要处理、固定规则、智能体选择策略**三种方案，观察清洗方式如何影响记录保留和 K-Means 分群。智能体只选择受约束的处理策略，数据解析、清洗和聚类均由本地 Python 工具执行。

本仓库是项目的精简复现版，包含主实验、三次真实模型补充实验、已完成的 12 项人工复核，以及最终课程报告与 7 分钟汇报 PPT。

## 汇报 PPT

[![Google Play 数据处理智能体 7 分钟汇报 PPT 封面](docs/course/presentation-preview.png)](docs/course/GooglePlay数据处理智能体PPT.pptx)

点击封面可查看或下载当前正式 PPT。

## 汇报前更正（2026-11-02）

- [四次模型调用费用表](docs/audit/model_costs.csv) · [费用与估算依据](docs/audit/费用依据_1102.md)：已补账户截图：主实验<¥0.01，三次补充合计<¥0.01；精确额未披露，美元估算另列。
- [新增理由一致性规则检查与澄清](docs/audit/理由一致性增强_1102.md)：已接入新实验入口；历史四次仍保留原始过程，规则不等于完整语义验证。
- [模型理由矛盾的人工检查说明](docs/audit/reason_consistency_review.md)：补充04按JSON执行；历史实验仅校验结构；增强版增加有限规则检查。
- [已更正的12项复核](docs/review/README.md)：日期统一为2026-10-27，保留更正对照；清洗数值、去留与聚类结果未改。
- [新版问答](docs/汇报/数据挖掘课程汇报问答_完整版_优化补充版_1103.docx)；项目报告第7、9页同步补充说明，仍为10页。

## 从这里开始

| 你想了解什么 | 阅读入口 |
| --- | --- |
| 项目做了什么、智能体如何工作 | [项目方法与质量判断](docs/PROJECT.md) |
| 安装、离线复现、重新实验 | [运行指南](docs/REPRODUCE.md) |
| 主实验结果、清洗表、图表 | [主实验结果索引](results/final/README.md) |
| 多次调用的策略与指标是否稳定 | [三次补充实验](results/stability/README.md) |
| 人工复核的依据与结论 | [12 项三组复核](docs/review/README.md) |
| 最终报告、7 分钟 PPT、实际贡献 | [课程材料](docs/course/README.md) · [贡献说明](docs/CONTRIBUTIONS.md) |
| 数据来源与整理时保留了什么 | [数据说明](data/README.md) · [版本与来源](docs/SOURCES.md) |

## 一分钟了解结果

原始数据共 **10,841 条、13 列**。主实验隔离 1 条错位记录；固定规则和智能体组均移除 483 条完全重复的多余记录。`cleaned` 是可追溯的清洗档案，只有特征完整的记录进入聚类。

| 主实验方案 | 清洗档案数 | 聚类记录数 | 聚类保留率¹ | 簇数 k | 轮廓系数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最低必要处理 | 10,840 | 7,729 | 71.29% | 3 | 0.3483 |
| 固定规则 | 10,357 | 10,357 | 95.54% | 3 | 0.4178 |
| 智能体 | 10,357 | 8,892 | 82.02% | 3 | 0.3665 |

¹ 分母均为原始 10,841 条。数据来源：[comparison.csv](results/final/comparison.csv)。各组原生样本不同，不能只按轮廓系数排名。共同的 7,424 条完整记录固定 `k=4` 后，组间 ARI 均为 1；这不能证明填补值接近真实值。

智能体主实验选择“去重 + 评分缺失排除聚类 + 大小按类别中位数填补”。三次补充调用中，去重和评分策略均一致，大小策略为 `drop / drop / category_median`，说明**大小处理决策存在波动**。人工复核 12 条均判为合理，但不代表全表正确率；当前尚无带真值或遮蔽验证的填补误差实验。

## 快速复现：无需 API 密钥

需要 Python 3.12。在仓库根目录执行：

```bash
python -m venv .venv
```

激活环境：Windows PowerShell 用 `.\.venv\Scripts\Activate.ps1`；macOS/Linux 用 `source .venv/bin/activate`。如果 PowerShell 限制脚本执行，直接用 `.\.venv\Scripts\python.exe` 代替下列 `python`。

```bash
python -m pip install -r requirements.lock.txt
python -m unittest discover -s tests -v
python run.py --replay
```

成功时输出 `status: verified`、`csvs: 26`、`jsons: 8`、`new_api_calls: 0`。回放从原始 CSV 和真实模型已记录的策略重新计算，逐项核对结果；耗时字段除外。新结果写入 `results/replay/`，不会覆盖正式记录。详见[补充实验回放和真实模型运行](docs/REPRODUCE.md)。

## 目录

```text
Data-Mining-sol/
├── README.md                  项目首页
├── run.py                     唯一命令入口
├── requirements*.txt          依赖范围与锁定版本
├── src/                       数据质量、智能体、聚类和流程
├── scripts/                   离线回放、结果指纹、模型启动辅助
├── utils/                     图表样式与运行清单
├── tests/                     规则、调用证据和结果校验
├── data/                      原始 CSV、来源与数据许可
├── results/
│   ├── final/                 主实验完整数值结果和图表
│   └── stability/             三次补充统计、调用证据和复算指纹
└── docs/                      方法、运行、贡献、复核与课程材料
```

核心六份 `src/*.py` 和原始 CSV 保持实验记录对应的字节版本。补充实验的大表可通过回放再生成，仓库只保留核验指纹及阅读必需的摘要。重复 ZIP、历史交接材料、空白复核模板、虚拟环境和新运行缓存均不纳入版本管理。
