# module2_policy: strategy distribution probability

## Description
第四章模块二正式结果，对应策略分布概率实验。当前仓库仅保留 `T` 维度正式结果，用于支撑策略分布、攻击概率与信号机制图表。

## Shared Protocol
- 基础环境：`N=5`, `T=50`, `beta=0.50`, `psi0=0.50`
- 模块二主学习率：`7e-5`
- 参数敏感性训练步数：`120 episodes`
- `T` 主扫描训练步数：`160 episodes`
- 评估 episode 数：`20` 或 `24`
- 图表语言：中文

## Shared Payoff Setting
- `g_a=5.0`
- `c_a=0.5`
- `l_a=3.0`
- `l_i=5.0`
- `g_i=4.0`
- `g_c=2.0`
- `eta_c=0.5`
- `kappa_d=1.5`
- `kappa_a=2.5`
- `c_theta1=1.0`
- `c_theta2=1.0`

## Core Files
- `tables/tab_4_3_T_policy_metrics.csv`
- `csv/module2_T_formal_tuned_evaluation_records.csv`
- `csv/episode_log_T_*.csv`
- `configs/module2_T_formal_tuned.json`
- `results_summary.csv`

## Key Figure Files
- `figures/fig_4_5_policy_distribution_T100.0.png`
- `figures/fig_4_5_attack_probability_T100.0.png`
- `figures/fig_4_5_T_signal_effect.png`

## Snapshot
- `T=100` 时策略分流最明显：
  - `real_host_attack_rate = 0.2807`
  - `honeypot_hit_rate = 0.1270`
  - `deception_success_rate = 0.3333`
  - `signal_effect = 0.3485`
