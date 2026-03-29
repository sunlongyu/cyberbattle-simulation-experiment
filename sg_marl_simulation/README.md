# SG MARL Simulation

本目录汇总多智能体强化学习仿真实验的核心代码与最终结果。

目录说明：

- `code/`：实验环境、模块化脚本与运行入口代码快照。
- `module1_convergence/`：收敛性与训练稳定性结果。
- `module2_policy/`：策略演化与信号机制结果。
- `module3_effectiveness/`：算法对比与消融结果。

每个结果模块目录中保留：

- `figures/`：正式图表
- `tables/`：统计表
- `csv/`：原始结果数据
- `configs/`：实验配置
- `results_summary.csv`：汇总结果
- `README.md`：模块说明

说明：

- 本次上传未包含训练检查点和中间日志 `logs/`。
- 如需复现实验，可结合 `code/` 目录中的脚本与配置文件重新运行。
