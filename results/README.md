# 实验记录

仅记录真实执行数据、失败证据和适用范围；运行命令与指标定义维护在[验证文档](../docs/validation.md)。

| 文件 | 状态与范围 |
| --- | --- |
| [2026-10-13_dense-2b_fp16.md](2026-10-13_dense-2b_fp16.md) | 历史实测：RTX 2080 Ti / Dense 2B / FP16 Eager，旧代码提交和 dirty 工作树 |
| [qwen35_dense_2b.json](qwen35_dense_2b.json) / [unit JSON](qwen35_dense_2b_unit.json) | 上述实测原始 JSON |
| [TEMPLATE.md](TEMPLATE.md) | Parity、Prefix、Baseline、Profile 和 Graph 共用新实验模板 |

记录应包括时间、Git commit / dirty 状态、模型 revision、GPU 与依赖、输入与参数、原始日志、统计口径、结果和局限。状态分为 PENDING（未执行）、PASS（已通过指定用例）、FAIL（有失败）、BLOCKED（环境阻塞）；SKIP 不能归类为 PASS。

**当前 HEAD 的完整 MoE 对齐及性能测量尚无此目录中的实测证据。** 未执行的新设备测试只需复制模板，不预先维护多份空白报告。
