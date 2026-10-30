# 交给 Codex 的启动说明

将本目录中的完整方案与 YAML 模板放到目标 PRS 仓库可读取的位置，然后把下面这段任务交给 Codex：

> 请完整阅读 `PRS_Prototype_Distillation_Codex_Plan_v1.md`，在当前仓库增量实现其中的 Phase A 核心实验。
>
> 先检查 AGENTS.md、README、现有 PRS manifest、数据划分、Teacher 配置与评估脚本，自动定位能够明确核实的路径和字段，生成 discovery_report 和 resolved_config。不要猜测 PRS 来源、校正方式、标签或路径；缺少真实数据时继续完成软件实现与测试，并准确报告 blocker。
>
> 严格保留模型：logit p=b(age,A,C)+exp(u(age))*sum_k softmax(WA+c)_k*Z_k。每个已有 PRS 就是固定原型；不增加 r_k、v_kj、新原型、MLP gate、额外 Teacher 输入或 SPEX。
>
> 实现并测试：先内部划分再生成 Teacher OOF 软标签；同结构直接监督、纯蒸馏与混合蒸馏；严格的作用域隔离、缓存 provenance、selection lock、refit、official-validation 上的直接诊断，以及精确的个体分项输出。Teacher 与 Student 使用同一输入信息，不把最新版默认 TabPFN 当作已固定的历史版本。
>
> 首先跑测试与一个 trait 的 smoke，再运行核心实验。若数据和权限齐备，完成预定核心模型与两个轻量消融；不要自行增加模型家族。原有 Stage-2 正式比较独立按 Phase B 执行，未执行时不要将直接诊断称为历史官方协议结果。
>
> 最后报告实际修改文件、实际执行命令、测试与隔离审计、三类学生和 Teacher 的比较、权重稳定性、产物路径以及未完成事项。不得伪造模型结果或自动访问 blind holdout 标签。

`prs_distillation_v1.example.yaml` 是待解析并适配的配置规格，不是已存在软件的可直接运行命令。命令接口由 Codex 结合现有仓库实现。
