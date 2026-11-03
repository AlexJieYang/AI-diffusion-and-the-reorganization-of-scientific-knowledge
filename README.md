# AI 方法扩散、发表来源与知识特征

本项目基于本地 SciSciNet2 数据生成 Journal of Informetrics 研究所用的描述表、五张主图和可选 R 条件关联模型。总体为 1980—2024 年具有有效 `AIProb` 的全部核心论文，主趋势从 1990 年展示。`AI-positive` 为严格 `AIProb > 0.5`，`Other-scored` 为有效评分不超过 0.5。

论文按唯一 `paper_id` 计数，发表来源使用核心表已有的 `source_id`。论文是否进入总体只由评分和年份决定。程序不读取论文—Fields 关系表，不生成六领域分组或标签集合。

## 从头运行

解压完整代码包到新的代码目录，在代码根目录运行。先检查 `config.json` 中的 `research_dir`；默认值为 `C:/OpenAlex/SciSciNet2_research`，输出为独立的 `results_scored_sources`。

```bash
python -m pip install -r requirements.txt
python run_from_scratch.py
```

该入口从输入 Parquet 重建母表，依次生成总体描述、国家信用、source 组合、知识特征比较、引用队列及五张主图。每次完整运行只清理由本程序创建的专用输出目录。配置和必需输入通过检查后才开始清理。

可选的 R 模型：

```bash
Rscript setup_R.R
python run_from_scratch.py --with-r
```

R 需要已安装且能够从命令行调用。依赖为 `fixest`、`arrow`、`data.table`、`jsonlite`。R 使用本轮 Python 导出的唯一论文输入。

使用 notebook 时先执行 `python setup_kernel.py`，再按 00、01、02、03、04、06 顺序执行。需要 R 模型时在 04 与 06 之间执行 05；05 使用 R 内核，其余使用 `Python - JOI` 内核。00 开始一轮新分析，01—06 不清理其他阶段的结果。所有 notebook 交付时均为空输出，需从头执行。

```bash
python run_from_scratch.py --config config.json
python -m unittest discover -s tests -v
```

## 使用本地已有指标

| 本地输入，相对 research 目录 | 用途 |
| --- | --- |
| `core/papers.parquet` | 论文、年份、`source_id`、团队、参考年龄和已有引用指标 |
| `extended/AI_papers.parquet` | `paper_id` 与 `AIProb` |
| `extended/country_members.parquet` | 论文—国家归属 |
| `metrics/interdisciplinarity.parquet` | 预计算的参考学科多样性及其覆盖率 |
| `metrics/embedding_paper.parquet` | 预计算的参考语义距离及其覆盖率 |
| `metrics/embedding_field_history.parquet` | 预计算的历史学科语义偏离 |
| `stage/edges/data.parquet` | 现成引用边，仅汇总引用年份与队列 |

可选读取 `metrics/team_affiliation_diversity.parquet` 的国家归属覆盖指标。设置 `edges_enabled=false` 可以关闭引用队列；相应图内保留原面板并标示无数据。

预计算指标按 `paper_id` 原值连接，不重新计算其参考总体、学科定义、标准化、覆盖率或缺失值。参考学科多样性与历史学科距离作为论文层面的既有结果指标保留；它们不用于六领域筛选或分组。`c5_norm` 等既有标准化指标也保留原定义。C5 比较、R 输入及相关描述只使用截至 2019 年的成熟队列；母表保存输入原值。

## Source 分析与示例来源

每篇核心论文只有一个发表 source，因此来源分组合计与唯一论文总体一致。国家信用在每篇论文的不同已知国家之间均分，总计一单位；country × source 单元同样守恒。没有国家信息的论文保留在论文总体中，国家统计使用可归属论文，并输出归属覆盖。

国家 RCA、组合熵、国家对 Jensen–Shannon 距离及 AI 比例变化分解使用全部 source。矩阵使用稀疏存储，RCA 只导出观察到的 country × source 单元，避免将全部国家与全部来源展开为笛卡尔积。AI 计数为零且分母可识别的单元保留为观察零值。来源熵描述发表场所分布，不解释为研究内容或学科多样性。

为保持紧凑的图布局，图1d、图2a/b/e、图4d使用同一组最多六个示例来源。自动选择规则为：在 `comparison_years`（默认 2020—2024）内两组分别至少 `min_source_group_n=20` 篇，然后按该时期全部评分论文量降序排列，ID 用于并列排序。选择不使用多样性差异、回归系数或显著性。来源不足六个时只展示实际合格来源，不填造来源。示例不限制总体、国家组合、source 调整或 R 模型的样本，也不代表所有来源的随机样本。

`source_catalog.csv` 保存所有来源的真实 ID、两组样本数、资格及展示顺序。本地准备后的 source 表不含名称，因此默认图签使用 `S` 加真实 ID。可通过配置为已核实的 ID 指定短名称，或用 `representative_source_ids` 固定最多六个有足够支持的来源；列表中的 ID 仍需满足同一支持规则。`source_labels` 的键使用不带 S 前缀的十进制 ID 字符串。

## 比较与支持范围

知识特征的 source 调整使用每个指标、每一年中两组各至少 `min_profile_cell_n=5` 篇可测论文的全部来源。先计算各 source 内的原单位均值差，再按两组合并的可测论文数加权。每篇论文只出现在其自身 source 中一次；不要求六个示例同时存在。表 S11 记录支持来源数、ID、两组样本数、保留比例与支持集合变化。趋势线仅连接支持来源集合相同的相邻年份。

图3e 和表 S13 使用同一份 2024 年国家横截面。国家 source 调整要求该 country × source 在每个配置窗口均有最低评分信用、在目标年有正分母且基准期有正世界来源权重。原始比例与调整比例在同一保留 source 支持上计算；另报国家总体唯一论文 AI 比例。空支持保持为空，不插补、不放宽门槛。

## 五张主图

五张图保持固定的画布尺寸、子图位置、网格比例、面板字母及三维视角，输出 PNG、PDF、SVG 和图注。

| 主图 | 内容 |
| --- | --- |
| 图1 | AI 规模、总体趋势、示例来源、评分分布及国家分布 |
| 图2 | 全部来源的国家专业化；热图和流图展示固定示例来源 |
| 图3 | 来源组合熵、国家间距离、变化分解及共同来源标准化 |
| 图4 | 既有参考学科多样性与语义指标；source 调整差异和示例来源内比较；保留原参考多样性曲面 |
| 图5 | 引用队列、参考年龄、国家集中度及 AI 比例 |

图1b继续使用上下独立的线性计数尺度。图4c的 Simpson 与余弦距离保持各自的原单位。图5d与表7、图3e与表S13共用完整精度来源表。

R 论文模型以 publishing source × year 为固定效应，使用原有预计算指标作为结果变量，来源数充足时按 source 聚类；国家模型使用 country 与 source × year 固定效应。这些是条件描述性关联。

## 输出

`data` 保存母表、国家/source 汇总及 R 输入；`tables` 保存统计表与绘图来源；`figures` 保存五张主图；`r` 保存可选模型；`status` 保存运行所需的配置和样本一致性信息。代码包不含本地原始数据或研究结果。测试使用明确标识的合成数据，核验完整流程、指标原值、计数守恒、支持变化、表图一致性与布局。
