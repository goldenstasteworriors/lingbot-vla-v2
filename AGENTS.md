# 项目约束

## 动作监督来源

- 所有后续训练、微调及新建 robot config 中，模型输出的动作序列必须使用未来的 `observation.state` 作为监督目标。
- 禁止使用 `wbc.action`、`action.wbc`、数据集 `action` 或其他控制指令列作为模型输出的监督目标。
- 对 LeRobot 时刻 `t`，目标动作 chunk 必须由后续实测状态构造；在本项目的 robot config 中使用 `convert_from_state: true`，并保留 episode 边界的 padding mask。
- 输出的物理维度及顺序必须与对应的 `observation.state` 一致；数据 schema 发生变化时，必须先核验状态切片和归一化统计再开始训练。
