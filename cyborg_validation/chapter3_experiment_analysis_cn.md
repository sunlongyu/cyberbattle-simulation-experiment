# 第三章实验效果分析稿

## 1. 分析目标

本节关注的问题不是“策略是否在所有意义下都优于基线”，而是更严格的两个问题：

1. PBNE 伪装策略是否能在更真实的攻防环境中有效降低攻击成功率与关键资产风险；
2. 实验结果是否足以为策略合理性提供证据支撑。

本节结论基于以下结果文件：

- 理论层：[feasible_scenarios.json](/Users/SL/Documents/expproject/sg_deception_simulation/results/feasible_scenarios.json)
- 环境层：[chapter3_formal_experiments_newutility.json](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/chapter3_formal_experiments_newutility.json)

## 2. 主实验图示

蓝方回报主图：
![chapter3_primary_blue_rewards](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/figures/chapter3_primary_blue_rewards.png)

安全效果指标图：
![chapter3_primary_security_metrics](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/figures/chapter3_primary_security_metrics.png)

失陷结构图：
![chapter3_compromise_breakdown](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/figures/chapter3_compromise_breakdown.png)

理论与环境方向一致性图：
![chapter3_theory_environment_alignment](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/figures/chapter3_theory_environment_alignment.png)

## 3. 能够明确支持的结论

### 3.1 PBNE 策略在环境层有效

从主实验可以直接看出，PBNE 策略在两个场景下都优于真实披露基线：

- 场景 A 中，`PBNE-1` 使蓝方平均回报提升 `106.142`
- 场景 B 中，`PBNE-2` 使蓝方平均回报提升 `69.083`

这说明在 CybORG 环境中，基于信号伪装的策略不是“理论上成立、环境中失效”，而是在更高保真环境中仍能产生可观测的防守收益。

### 3.2 PBNE 策略的收益来源是“压攻击链”，而不是简单增加蜜罐命中

图中最重要的一点是：PBNE 策略带来的收益主要来自以下三项下降：

- `attack-like actions` 减少
- `production compromise rate` 下降
- `critical host compromise rate` 下降

以场景 A 为例，`PBNE-1` 将：

- 攻击样动作均值从 `18.000` 降到 `15.083`
- 生产主机失陷率从 `0.917` 降到 `0.667`
- 关键主机失陷率从 `0.917` 降到 `0.583`

这说明 PBNE 策略的核心机制是延缓和扰乱攻击链推进，而不是简单追求更高的蜜罐接管率。

### 3.3 策略合理性得到了“方向性证明”

理论层中，改进效用函数下：

- 场景 A：防御者期望效用由 `-7.779` 提升至 `-5.672`
- 场景 B：防御者期望效用由 `-4.359` 提升至 `-3.409`

环境层中，蓝方回报也同步提升。因此可以说：

> 理论推导出的 PBNE 伪装策略，在更真实的网络攻防环境中仍然表现出明确的防守优势。

这已经足以支撑“策略具有现实合理性”的论文结论。

## 4. 不能过度宣称的结论

### 4.1 不能写成“理论效用值与环境奖励严格一致”

当前结果不支持这一表述。原因是：

- 理论层是有效单阶段信号博弈；
- 环境层是多阶段攻击链仿真；
- 环境收益更敏感于攻击链延缓、关键资产保全和攻击压力下降。

因此，理论效用和环境奖励在数值上不应被表述为“严格一一对应”，只能写成“方向一致、机制相符”。

### 4.2 不能写成“PBNE 一定提高蜜罐命中率”

图上已经表明，这并不是稳定结论。某些情况下，PBNE 策略同时降低了生产主机失陷率和蜜罐主机失陷率。这恰恰说明策略的主要作用不是“强行把攻击者导入蜜罐”，而是通过扰乱信号判断降低整体攻击推进效率。

所以论文里更稳的表述应是：

> PBNE 伪装策略通过压制攻击推进、降低关键节点暴露风险来提升防守收益；蜜罐交互增减只是次级表现，而非唯一目标。

## 5. 敏感性分析能说明什么

先验概率敏感性图：
![chapter3_sensitivity_prior_theta1](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/figures/chapter3_sensitivity_prior_theta1.png)

`beta` 敏感性图：
![chapter3_sensitivity_beta](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/figures/chapter3_sensitivity_beta.png)

### 5.1 `prior_theta1` 是最关键的影响因素

当 `prior_theta1` 位于 `0.4-0.6` 区间时，PBNE 策略的环境回报提升最明显。这说明：

- 当真实系统与欺骗系统的先验不确定性较强时，信号伪装最容易发挥作用；
- 当类型结构过于偏向单一类型时，攻击者更容易形成稳定预期，从而削弱伪装收益。

### 5.2 `beta` 会影响效果，但不是主导因素

`beta` 的变化会影响最终收益和攻击动作减少幅度，但整体波动小于 `prior_theta1`。因此在论文中可以写成：

> 信念折扣因子会影响策略表现，但其作用强度弱于类型先验分布。

## 6. 论文中建议采用的分析口径

如果要让实验结果真正为策略合理性作证，建议在论文中采用以下口径：

1. 先证明理论层 PBNE 策略在改进效用函数下提升防御者期望效用。
2. 再说明 CybORG 环境实验中，PBNE 策略进一步降低了攻击推进强度与关键资产失陷率。
3. 最后总结：理论策略不是仅在抽象模型中成立，而是在真实攻防环境中也具有防守优势。

## 7. 可直接使用的结论段

可在论文中使用如下表述：

> 实验结果表明，基于改进效用函数推导得到的 PBNE 伪装策略在理论层与环境层均表现出稳定优势。在理论数值仿真中，PBNE-1 与 PBNE-2 均提高了防御者期望效用并降低了攻击者收益；在 CybORG 环境实验中，PBNE 策略进一步显著降低了攻击样动作数量、生产主机失陷率与关键主机失陷率，说明该策略的主要作用机制在于压制攻击链推进并降低高价值资产暴露风险。因此，可以认为 PBNE 伪装策略不仅具有博弈论上的可解释性，而且具备较强的现实防御有效性。

## 8. 当前证据的边界

本组实验已经足以支撑“策略合理且有效”这一结论，但仍需保持以下边界：

- 可以证明“PBNE 优于真实披露基线”
- 可以证明“PBNE 降低攻击推进与关键资产风险”
- 不应宣称“理论效用与环境奖励严格等价”
- 不应宣称“所有收益都来自蜜罐命中增加”

在毕业论文中，按上述边界写作是稳妥且可信的。
