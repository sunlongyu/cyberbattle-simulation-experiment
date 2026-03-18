# 第三章 基于改进效用函数的多阶段信号博弈建模与实验验证

## 3.1 问题描述

在工业互联网攻防场景中，攻击者往往并非基于一次性静态观察完成决策，而是在多轮侦察、利用、提权与横向移动过程中逐步修正对目标系统属性的判断。若仍采用仅刻画“生产系统被攻击”与“蜜罐被攻击”终局结果的静态效用函数，则难以准确反映以下三类现实因素：

1. 攻击压力本身会导致防御方额外处置开销；
2. 欺骗交互本身具有情报收益，而不仅仅表现为“蜜罐最终是否被完全接管”；
3. 理论策略在真实攻防环境中是否仍然有效，需要借助更高保真度的环境框架进行验证。

因此，本章保留原有信号博弈框架，但引入更合理的有效单阶段效用函数，并使用 CybORG 构建环境层实验，对理论推导出的 PBNE 伪装策略进行验证。

## 3.2 博弈模型

### 3.2.1 参与方与类型空间

设博弈双方为防御者 `D` 与攻击者 `A`。防御者私有类型集合为：

```text
Theta = {theta1, theta2}
```

其中：

- `theta1` 表示真实生产系统；
- `theta2` 表示蜜罐/欺骗系统。

攻击者关于防御者类型的先验信念为：

```text
Pr(theta = theta1) = p,   Pr(theta = theta2) = 1 - p,   0 < p < 1
```

### 3.2.2 信号与动作空间

防御者发送信号：

```text
Sigma = {sigma1, sigma2}
```

其中 `sigma1` 表示生产系统外观信号，`sigma2` 表示伪装/欺骗信号。

攻击者观察信号后选择动作：

```text
M = {attack, retreat}
```

## 3.3 改进效用函数

### 3.3.1 参数定义

记：

- `g_a`：攻击者成功攻击生产系统时获得的收益；
- `c_a`：攻击者发起攻击的直接成本；
- `l_a`：生产系统被攻击时防御者承受的核心损失；
- `eta_d`：攻击发生时防御方额外响应压力成本；
- `g_i`：攻击者与欺骗系统交互时防御者的基础情报收益；
- `kappa_i`：欺骗交互带来的额外情报收益；
- `l_i`：攻击者误攻欺骗系统时承受的基础损失；
- `kappa_a`：攻击者陷入欺骗环境时承受的额外惩罚；
- `c_theta1`：生产系统发送欺骗信号的伪装成本；
- `c_theta2`：欺骗系统发送生产信号的伪装成本。

伪装成本函数定义为：

```text
c(theta1, sigma2) = c_theta1
c(theta2, sigma1) = c_theta2
c(theta1, sigma1) = 0
c(theta2, sigma2) = 0
```

### 3.3.2 有效单阶段效用

若攻击者选择撤退，则：

```text
U_D(theta, retreat | sigma) = -c(theta, sigma)
U_A(theta, retreat | sigma) = 0
```

若攻击者选择攻击，则：

```text
U_D(theta1, attack | sigma) = -(l_a + eta_d) - c(theta1, sigma)
U_A(theta1, attack | sigma) = g_a - c_a
```

```text
U_D(theta2, attack | sigma) = g_i + kappa_i - c(theta2, sigma)
U_A(theta2, attack | sigma) = -(c_a + l_i + kappa_a)
```

与原模型相比，该效用函数将“攻击压力”和“欺骗附加价值”显式写入收益结构，因此更适合描述真实网络攻防中的诱骗收益与响应负担。

## 3.4 完美贝叶斯均衡分析

### 3.4.1 基准分离策略

首先考虑真实披露基线：

```text
theta1 -> sigma1
theta2 -> sigma2
```

攻击者在 `sigma1` 后攻击，在 `sigma2` 后撤退。该策略可作为后续 PBNE 伪装策略的对照基线。

### 3.4.2 PBNE-1：生产系统伪装均衡

设候选均衡满足：

```text
theta1: sigma1 的概率为 1 - lambda_d^*，sigma2 的概率为 lambda_d^*
theta2: sigma2 的概率为 1
```

攻击者策略为：

```text
sigma1 后必然攻击；
sigma2 后以概率 lambda_a^* 攻击。
```

在该结构下：

```text
mu(theta1 | sigma1) = 1
mu(theta1 | sigma2) = p lambda_d^* / (p lambda_d^* + 1 - p)
```

攻击者在 `sigma2` 后无差异，可得：

```text
lambda_d^* = ((1 - p)(c_a + l_i + kappa_a)) / (p(g_a - c_a))
```

`theta1` 型防御者在 `sigma1` 与 `sigma2` 间无差异，可得：

```text
lambda_a^* = 1 - c_theta1 / (l_a + eta_d)
```

因此，当

```text
0 < ((1 - p)(c_a + l_i + kappa_a)) / (p(g_a - c_a)) < 1
0 < c_theta1 < l_a + eta_d
g_a > c_a
```

时，PBNE-1 存在。

### 3.4.3 PBNE-2：欺骗系统伪装均衡

设候选均衡满足：

```text
theta1: sigma1 的概率为 1
theta2: sigma1 的概率为 lambda_d'，sigma2 的概率为 1 - lambda_d'
```

攻击者策略为：

```text
sigma1 后以概率 1 - lambda_a' 攻击；
sigma2 后必然撤退。
```

在该结构下：

```text
mu(theta1 | sigma1) = p / (p + (1 - p)lambda_d')
mu(theta1 | sigma2) = 0
```

攻击者在 `sigma1` 后无差异，可得：

```text
lambda_d' = p(g_a - c_a) / ((1 - p)(c_a + l_i + kappa_a))
```

`theta2` 型防御者在 `sigma1` 与 `sigma2` 间无差异，可得：

```text
lambda_a' = (g_i + kappa_i - c_theta2) / (g_i + kappa_i)
```

因此，当

```text
0 < p(g_a - c_a) / ((1 - p)(c_a + l_i + kappa_a)) < 1
0 < c_theta2 < g_i + kappa_i
g_a > c_a
```

时，PBNE-2 存在。

## 3.5 理论数值实验设计

### 3.5.1 参数设置

当前理论层默认参数为：

```text
g_a = 12, c_a = 2, l_a = 10, eta_d = 2
g_i = 8,  l_i = 8, kappa_i = 2, kappa_a = 2
c_theta1 = 2.5, c_theta2 = 1.5
```

并采用两组可行场景：

- 场景 A：`p = 0.65`，比较 truthful baseline 与 `PBNE-1`
- 场景 B：`p = 0.35`，比较 truthful baseline 与 `PBNE-2`

理论实验结果文件位于 [feasible_scenarios.json](/Users/SL/Documents/expproject/sg_deception_simulation/results/feasible_scenarios.json)。

### 3.5.2 理论结果

表 3-1 给出两组可行场景下的理论结果。

| 场景 | 策略 | 防御者期望效用 | 攻击者期望效用 | 关键均衡参数 |
|---|---|---:|---:|---|
| A (`p=0.65`) | truthful baseline | -7.779 | 6.482 | - |
| A (`p=0.65`) | PBNE-1 | -5.672 | 2.338 | `lambda_d^*=0.6462`, `lambda_a^*=0.7917` |
| B (`p=0.35`) | truthful baseline | -4.359 | 3.632 | - |
| B (`p=0.35`) | PBNE-2 | -3.409 | 1.238 | `lambda_d'=0.4487`, `lambda_a'=0.8500` |

从理论结果可见，两类 PBNE 伪装策略均降低了攻击者期望效用，并提升了防御者期望效用，说明改进后的效用函数下，伪装策略仍具有明确博弈优势。

## 3.6 CybORG 环境实验设计

### 3.6.1 环境构建

为提高实验环境保真度，本章进一步引入 CybORG 作为网络攻防环境层。与理论模型对应，本研究构建了 thesis-specific 场景，其中：

- `Prod_Enterprise` 与 `Prod_Operational` 对应 `theta1` 类主机；
- `Honey_Enterprise` 与 `Honey_Operational` 对应 `theta2` 类主机；
- `User1` 与 `User2` 作为攻击链跳板；
- 红方通过用户区、企业区、业务区逐步推进攻击。

实验脚本位于 [run_chapter3_formal_experiments.py](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/run_chapter3_formal_experiments.py)，结果文件位于 [chapter3_formal_experiments_newutility.json](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/chapter3_formal_experiments_newutility.json)。

### 3.6.2 环境主实验结果

本研究采用 `12 episodes / 30 steps` 作为主实验设置，结果如表 3-2 所示。

| 场景 | 策略 | 蓝方平均回报 | 攻击样动作均值 | 生产主机失陷率 | 蜜罐主机失陷率 | 关键主机失陷率 |
|---|---|---:|---:|---:|---:|---:|
| A (`p=0.65`) | truthful baseline | -326.758 | 18.000 | 0.917 | 0.750 | 0.917 |
| A (`p=0.65`) | PBNE-1 | -220.617 | 15.083 | 0.667 | 0.417 | 0.583 |
| B (`p=0.35`) | truthful baseline | -357.725 | 18.000 | 0.917 | 0.250 | 0.917 |
| B (`p=0.35`) | PBNE-2 | -288.642 | 15.833 | 0.750 | 0.333 | 0.750 |

对应比较结果为：

- 场景 A：`PBNE-1` 相比真实披露基线，蓝方平均回报提升 `106.142`，攻击样动作均值减少 `2.917`
- 场景 B：`PBNE-2` 相比真实披露基线，蓝方平均回报提升 `69.083`，攻击样动作均值减少 `2.167`

这表明理论推导出的伪装策略在更真实的攻防环境中仍然能够显著降低攻击压力，并减少高价值主机失陷率。

## 3.7 敏感性分析

### 3.7.1 理论敏感性分析

理论层敏感性分析结果位于 [sensitivity_analysis.json](/Users/SL/Documents/expproject/sg_deception_simulation/results/sensitivity_analysis.json)。当前可得如下结论：

1. `prior_theta1` 升高时，`PBNE-1` 的防御者期望效用逐步下降，说明在生产系统占比过高时，伪装收益会受到约束。
2. `c_theta1` 增大时，`PBNE-1` 的防御者期望效用持续下降，攻击比例随之下降，但伪装成本上升削弱了总体收益。
3. `c_theta2` 增大时，`PBNE-2` 的防御者期望效用下降，说明欺骗系统伪装为生产系统的成本过高会显著压缩策略收益。
4. `eta_d` 增大时，`PBNE-1` 的攻击混合概率 `lambda_a^*` 上升，说明攻击压力越大，生产系统越有动力采用伪装策略。
5. `kappa_a` 增大时，`PBNE-2` 的攻击比例下降，说明更强的欺骗惩罚能够有效抑制攻击者对 `sigma1` 的攻击倾向。

### 3.7.2 CybORG 环境敏感性分析

环境层敏感性分析基于 [chapter3_formal_experiments_newutility.json](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/results/chapter3_formal_experiments_newutility.json)，可归纳为：

1. `prior_theta1` 在 `0.4-0.6` 区间时，PBNE 策略带来的蓝方收益提升最明显，表明此时类型不确定性最有利于信号伪装发挥作用。
2. `beta` 变化对环境收益有一定影响，但其影响强度低于先验概率，说明信念折扣对环境表现的调节作用次于类型先验。
3. `c_theta1` 与 `c_theta2` 在当前环境映射中对环境回报的边际影响较小，原因在于 CybORG 环境的主要收益变化仍然来自攻击链压制效果，而非理论层显式的伪装成本项。

## 3.8 本章结论

本章在保留信号博弈基本结构的前提下，提出了一个包含攻击压力成本与欺骗附加收益的改进效用函数，并在该效用函数下重新推导出两类 PBNE 伪装策略。理论数值实验表明，`PBNE-1` 与 `PBNE-2` 均能提升防御者期望效用并压制攻击者收益。进一步地，CybORG 环境实验表明，这些策略在真实攻防链路中仍然能够显著降低关键主机失陷率与攻击样动作数量，验证了改进模型的现实有效性。

需要指出的是，理论层的有效单阶段效用与环境层的多阶段动态奖励并非严格逐项等价，而是“理论给出可解释策略，环境验证其现实有效性”的关系。对于毕业论文而言，这一分层建模方式既保留了均衡证明的严谨性，也提升了实验设计的现实可信度。
