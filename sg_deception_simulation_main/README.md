# Chapter 3 Signaling-Game Simulation Main

本目录是第三章多阶段信号博弈的正式实验结果目录，用于保存论文侧采用的 Chapter 3 代码、JSON/CSV 结果和图表。

## 目录说明

- `cyber_simulation_core/`: 核心模型、策略和实验逻辑
- `run_experiments.py`: 生成实验结果 JSON 和 CSV
- `plot_results.py`: 根据结果文件生成 PNG 图片
- `results/`: 已生成的实验结果与图片

## 正式实验

1. 四类策略的防御者累积效用对比。
2. 四类策略的终局信念对比。
3. 参数敏感性分析，包含初始信念、收益/成本参数与折扣因子。

## 正式结果文件

- `experiment1_payoff_comparison.json`
- `experiment1_payoff_summary.csv`
- `experiment1_stage_paths.csv`
- `experiment2_terminal_belief_comparison.json`
- `experiment2_terminal_beliefs.csv`
- `experiment2_terminal_state_composition.csv`
- `experiment2_terminal_uncertainty_summary.csv`
- `experiment3_sensitivity_analysis.json`
- `experiment3_sensitivity_analysis.csv`
- `experiment3_sensitivity_trajectories.csv`
- `experiment3_discount_factor_sensitivity.json`
- `experiment3_discount_factor_mechanism.csv`
- `experiment3_discount_factor_horizon_paths.csv`

## 运行方式

在仓库根目录执行：

```bash
python sg_deception_simulation_main/run_experiments.py
python sg_deception_simulation_main/plot_results.py
```

维护规则：本目录是第三章正式实验的唯一主入口；旧 `sg_deception_simulation/` 已移除，历史内容可通过 Git 历史回溯。
