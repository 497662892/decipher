# All of Us 与 HPC 对接：可行性核查

核查日期：2026-10-12（美国东部时间；运行记录为 UTC）。本次完成政策、软件和公开端点核查，未登录研究账户、读取参与者数据或开通云计算资源。

## 结论

可以在 HPC 开发、测试和封存本项目的代码与公开 PRS 权重，再把符合授权要求的输入迁移到 All of Us 的受控云环境执行。标准访问路径不支持把 All of Us 个体基因型、表型或逐人 PRS 下载到本地 Slurm 节点计算。官方要求个体数据留在项目提供的分析环境中。[官方数据下载说明](https://www.researchallofus.org/faq/what-data-is-available-for-download-in-the-researcher-workbench/)

这里的关键限制是数据使用范围，安装 gcloud、挂载云盘或打通网络本身不能解决。2026 年官方术语允许 Researcher Workbench 以及由 All of Us 提供的其他分析环境；这并不等于本机构的普通 HPC 自动获得授权。基因组研究通常需要 Controlled Tier；CT+ 另有申请要求，不能将不同层级的凭证互换。[官方术语与权限定义](https://support.researchallofus.org/hc/en-us/articles/51716409164308-All-of-Us-Agreements-and-Policy-Glossary)

## 当前平台与本地检查

官方现行 Getting Started 文档指向由 Verily Pre 提供的 Researcher Workbench，支持 Git 仓库、JupyterLab、工作流及 GPU 环境。实现时应使用当前平台说明，避免直接套用旧 Terra 平台的认证与环境变量教程。[现行入门指南](https://support.researchallofus.org/hc/en-us/articles/41981050556564-Researcher-Workbench-Getting-Started-Guide)

| 项目 | 本次证据 | 含义 |
|---|---|---|
| Slurm、Python/uv、PLINK2、GPU | 本项目已执行 CPU 计分和 GPU TabPFN 作业 | 本地开发与复现条件具备 |
| 云工具 | 当前 PATH 无 gcloud/gsutil；module 目录提供 gcloud/5.0、google-cloud 环境 | 可进一步配置客户端；模块名不是已验证的 SDK 版本 |
| 容器工具 | module 目录提供 Apptainer 1.4.5 和多个 Singularity 版本 | 可以准备可迁移的软件环境；本次未构建云镜像 |
| 公开登录端点 | HPC HEAD 请求：researchallofus 登录地址返回 403，workbench.verily.com 返回 200 | 说明收到 HTTP 响应；403 不能算登录成功，也未验证受控 API 或数据权限 |
| 机构与个人权限 | 未核查 DURA、Controlled Tier、个人培训/身份验证和计费项目 | 真实运行前仍须在账户中确认 |

公开端点回执保存在忽略目录 `outputs/external_pgs/holdout_wrapper_20261012/aou_network_preflight.json`。未读取或导出本地凭证。

## 适合本项目的执行路径

1. 在 HPC 封存 Git commit、公开权重校验和、匹配规则、参考频率、模型配置和验证方案；用合成数据验证可移植性。
2. 在有 Controlled Tier 权限的 Workbench workspace 中配置项目说明、数据版本和计费。机构 DURA、个人身份验证、培训及 Data User Code of Conduct 均属于正式访问条件。[官方 Workbench 访问说明](https://www.researchallofus.org/data-tools/workbench/)
3. 上传代码和可合法复用的公开资源。在受控云内定义 CAD、T2D 和女性 BC 队列，完成基因型协调、PRS 计分和模型推理。可以采用平台支持的 Nextflow/Cromwell 等工作流；实际执行器位于云端，不是本地 Slurm。[官方工作流说明](https://support.researchallofus.org/hc/en-us/articles/4811899197076-Workflows-in-the-All-of-Us-Researcher-Workbench-Nextflow-and-Cromwell)
4. 只取回符合政策的汇总指标与图表。逐人预测、逐人 PRS 与表型留在受控环境中。小样本亚组发布须遵循当前披露规则；正式 Policy Statement 禁止直接发布或间接推导 1–20 人的计数，0 允许。[Data and Statistics Dissemination Policy](https://support.researchallofus.org/hc/en-us/articles/22346276580372-Data-and-Statistics-Dissemination-Policy)

## 当前 TabPFN 方案特有的问题

当前代码在预测时向 TabPFN 提供有标签的 NYU 训练 context，包括逐人 PRS、年龄、性别、PC 和结局。这是本地实现的直接事实，不能把这批输入视为只含公共参数的普通模型文件。若要在 AoU 完整复现 NYU 训练的 TabPFN＋wrapper，需要先确认这批去标识化 context 的外部使用与上传授权，并符合 AoU 对外部数据的要求。官方允许合规的外部资源进入环境，同时限制可识别个人的信息与逐行数据链接。[官方外部数据说明](https://www.researchallofus.org/data-tools/workbench/)

公开 PRS 权重与通用 TabPFN checkpoint 可以独立准备，但它们不能替代这批训练 context。若 context 不能上传，蒸馏成参数模型是另一项需要验证的研究工作；或者在 AoU 内重新训练整合器，但那将属于 AoU 内开发，不能再称为“完整模型从未在 AoU fit”。只有冻结的 SNP 权重仍可保留外部来源这一属性。

严格外部验证还需解决特征对齐：本项目使用提供的标准化 PC。AoU 独立计算的 PC1–PC10 不应直接假定与 NYU 的轴、符号和尺度一致。应确认两边都能投影到同一冻结参考坐标，并使用同一年龄编码、PRS 等位基因及标准化参数。这是迁移设计中尚未完成的工作，不是本次网络检查能够证明的事项。

因此，当前可行路线是“HPC 准备与封存 → AoU 受控云执行 → 合规汇总回传”。是否能够原样迁移当前 TabPFN 教师，还取决于 NYU context 的授权与共同参考特征空间；本次未宣称已完成 AoU 接入。
