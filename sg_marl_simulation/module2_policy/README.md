# module2_policy: complete sensitivity bundle

## Description
第四章模块二完整结果，对应 4.5.3“策略演化与均衡一致性分析”。
当前已经完成以下五组参数扫描：
- `T in {10,25,50,100}`
- `N in {5,7,10}`
- `psi0 in {0.3,0.5,0.7}`
- `c_theta1 in {0.6,1.0,1.4}`
- `c_theta2 in {0.6,1.0,1.4}`

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
- `tables/tab_4_3_parameter_sensitivity_overview.csv`
- `csv/module2_parameter_sensitivity_overview.csv`
- `csv/module2_T_formal_tuned_evaluation_records.csv`
- `csv/module2_N_formal_tuned_results_summary.csv`
- `csv/module2_psi0_formal_tuned_results_summary.csv`
- `csv/module2_ctheta1_formal_tuned_results_summary.csv`
- `csv/module2_ctheta2_formal_tuned_results_summary.csv`

## Key Figure Files
- `figures/fig_4_5_policy_distribution_T100.0.png`
- `figures/fig_4_5_attack_probability_T100.0.png`
- `figures/fig_4_5_T_signal_effect.png`
- `figures/fig_4_5_N_signal_effect.png`
- `figures/fig_4_5_psi0_signal_effect.png`
- `figures/fig_4_5_c_theta1_signal_effect.png`
- `figures/fig_4_5_c_theta2_signal_effect.png`

## Snapshot
- `T=100` 时策略分流最明显：
  - `real_host_attack_rate = 0.2807`
  - `honeypot_hit_rate = 0.1270`
  - `deception_success_rate = 0.3333`
  - `signal_effect = 0.3485`
- `N=10` 时蜜罐命中率和 deception success 明显上升
- `psi0=0.3` 与 `psi0=0.7` 都比 `psi0=0.5` 更能诱发可分析的选择性攻击
- `c_theta1` 增大时防御侧伪装意愿与攻击分流能力下降
- `c_theta2=1.4` 时整体 signal effect 明显增强
