# 数据格式与指标来源

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

