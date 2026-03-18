# 第三章理论模型重写稿（中文）

## 3.1 问题描述与建模目标

为更准确刻画高级持续性威胁场景中“攻击压力”“欺骗交互价值”与“伪装代价”对攻防双方策略的影响，本章在保留原有信号博弈结构的基础上，对效用函数进行修正。修正后的模型仍然是一个两类型、两信号、两动作的静态贝叶斯信号博弈，因此可以继续采用完美贝叶斯均衡（Perfect Bayesian Nash Equilibrium, PBNE）进行分析，并保留闭式混合均衡解。

本章不直接将 CybORG 的完整多阶段动态过程纳入理论主模型，而是将其作为环境验证层。理论层采用“有效单阶段效用”建模，实验层再利用 CybORG 验证该策略在真实攻防环境中的表现。

## 3.2 参与方、类型空间与策略空间

设博弈参与方为防御者 `D` 与攻击者 `A``。`

防御者私有类型空间为：

```text
Theta = {theta1, theta2}
```

其中：

- `theta1` 表示真实生产系统；
- `theta2` 表示蜜罐/欺骗系统。

攻击者对防御者类型的先验信念为：

```text
Pr(theta = theta1) = p,   Pr(theta = theta2) = 1 - p,   0 < p < 1
```

防御者可发送的信号集合为：

```text
Sigma = {sigma1, sigma2}
```

其中：

- `sigma1` 表示生产系统外观信号；
- `sigma2` 表示伪装/欺骗信号。

攻击者观察到信号后采取动作：

```text
M = {attack, retreat}
```

因此，防御者策略是从类型到信号的映射，攻击者策略是从信号到动作的映射。

## 3.3 修正后的效用函数

### 3.3.1 参数定义

记：

- `g_a`：攻击者成功攻击生产系统所获得的收益；
- `c_a`：攻击者实施攻击的直接成本；
- `l_a`：生产系统遭攻击时防御者承受的核心损失；
- `eta_d`：攻击发生时防御者因告警、分析、处置等产生的额外防守压力成本；
- `g_i`：攻击者攻击欺骗系统时，防御者获得的基础情报收益；
- `kappa_i`：欺骗交互带来的额外诱骗收益；
- `l_i`：攻击者攻击欺骗系统时承受的基础损失；
- `kappa_a`：攻击者因落入欺骗环境而承担的额外惩罚；
- `c_theta1`：生产系统伪装为欺骗系统时的伪装成本；
- `c_theta2`：欺骗系统伪装为生产系统时的伪装成本。

### 3.3.2 伪装成本函数

定义伪装成本函数：

```text
c(theta1, sigma2) = c_theta1
c(theta2, sigma1) = c_theta2
c(theta1, sigma1) = 0
c(theta2, sigma2) = 0
```

该定义表示：只有在类型与所发送信号不一致时，防御者才需要承担伪装成本。

### 3.3.3 有效单阶段效用

若攻击者选择 `retreat`，则双方效用分别为：

```text
U_D(theta, retreat | sigma) = -c(theta, sigma)
U_A(theta, retreat | sigma) = 0
```

若攻击者选择 `attack`，则：

当目标为真实生产系统 `theta1` 时，

```text
U_D(theta1, attack | sigma) = -(l_a + eta_d) - c(theta1, sigma)
U_A(theta1, attack | sigma) = g_a - c_a
```

当目标为欺骗系统 `theta2` 时，

```text
U_D(theta2, attack | sigma) = g_i + kappa_i - c(theta2, sigma)
U_A(theta2, attack | sigma) = -(c_a + l_i + kappa_a)
```

与原模型相比，上述修正包含两个关键增强：

1. `eta_d` 将攻击行为本身带来的告警与响应压力显式纳入防御者损失。
2. `kappa_i` 与 `kappa_a` 将欺骗交互带来的额外诱骗价值与额外攻击惩罚显式纳入双方效用。

## 3.4 基准分离策略

作为比较基线，考虑真实披露分离策略：

```text
theta1 -> sigma1
theta2 -> sigma2
```

在该情形下：

- 攻击者观察到 `sigma1` 后相信目标必为生产系统，因此选择 `attack`；
- 攻击者观察到 `sigma2` 后相信目标必为欺骗系统，因此选择 `retreat`。

这构成全文后续 PBNE 分析的基准策略。

## 3.5 PBNE-1：生产系统伪装均衡

### 3.5.1 均衡结构

考虑如下候选混合均衡：

```text
theta1: sigma1 的概率为 1 - lambda_d^*，sigma2 的概率为 lambda_d^*
theta2: sigma2 的概率为 1
```

攻击者策略为：

```text
观察到 sigma1 后必然攻击；
观察到 sigma2 后以概率 lambda_a^* 攻击，以概率 1 - lambda_a^* 撤退。
```

记该均衡为 PBNE-1。

### 3.5.2 信念更新

根据贝叶斯法则，

```text
mu(theta1 | sigma1) = 1
```

因为在该候选均衡中只有 `theta1` 会发送 `sigma1``。`

而在观察到 `sigma2` 时，有：

```text
mu(theta1 | sigma2) = p lambda_d^* / (p lambda_d^* + 1 - p)
```

### 3.5.3 攻击者对 `sigma2` 的无差异条件

攻击者在观察到 `sigma2` 后选择攻击的期望收益为：

```text
mu(theta1 | sigma2)(g_a - c_a) + (1 - mu(theta1 | sigma2))[-(c_a + l_i + kappa_a)]
```

在混合均衡中，攻击者必须在 `sigma2` 后无差异，因此令上式等于 0。利用分母消去后可得：

```text
p lambda_d^* (g_a - c_a) - (1 - p)(c_a + l_i + kappa_a) = 0
```

整理得：

```text
lambda_d^* = ((1 - p)(c_a + l_i + kappa_a)) / (p(g_a - c_a))
```

### 3.5.4 `theta1` 型防御者的无差异条件

当 `theta1` 发送 `sigma1` 时，由于攻击者必然攻击，因此其收益为：

```text
U_D(theta1, sigma1) = -(l_a + eta_d)
```

当 `theta1` 发送 `sigma2` 时，攻击者以概率 `lambda_a^*` 攻击，因此收益为：

```text
U_D(theta1, sigma2) = -lambda_a^*(l_a + eta_d) - c_theta1
```

在混合均衡中，`theta1` 对两种信号必须无差异，即：

```text
-(l_a + eta_d) = -lambda_a^*(l_a + eta_d) - c_theta1
```

故有：

```text
lambda_a^* = 1 - c_theta1 / (l_a + eta_d)
```

### 3.5.5 命题 3-1

若满足：

```text
0 < ((1 - p)(c_a + l_i + kappa_a)) / (p(g_a - c_a)) < 1
0 < c_theta1 < l_a + eta_d
g_a > c_a
```

则存在生产系统伪装混合均衡 PBNE-1，其混合概率为：

```text
lambda_d^* = ((1 - p)(c_a + l_i + kappa_a)) / (p(g_a - c_a))
lambda_a^* = 1 - c_theta1 / (l_a + eta_d)
```

### 3.5.6 证明

1. 由 `g_a > c_a` 可知在观察到 `sigma1` 时，攻击者面对确定的 `theta1` 目标，攻击收益为正，因此攻击是序贯最优的。
2. 在 `sigma2` 后，若攻击者采用混合策略，则其必须无差异，由此得到 `lambda_d^*`。
3. 对 `theta1` 而言，若其采用混合策略，则在 `sigma1` 与 `sigma2` 之间必须无差异，由此得到 `lambda_a^*`。
4. 对 `theta2` 而言，在该候选均衡中发送 `sigma2` 的收益不低于偏离至 `sigma1` 的收益，因此其最优响应为纯策略 `sigma2`。
5. 由于上述信念由贝叶斯法则在均衡路径上得到，且双方策略均为给定信念下的最优响应，因此该策略组合与信念系统构成 PBNE-1。

证毕。

## 3.6 PBNE-2：欺骗系统伪装均衡

### 3.6.1 均衡结构

考虑如下候选混合均衡：

```text
theta1: sigma1 的概率为 1
theta2: sigma1 的概率为 lambda_d'，sigma2 的概率为 1 - lambda_d'
```

攻击者策略为：

```text
观察到 sigma1 后以概率 1 - lambda_a' 攻击；
观察到 sigma2 后必然撤退。
```

记该均衡为 PBNE-2。

### 3.6.2 信念更新

根据贝叶斯法则，在观察到 `sigma1` 时，

```text
mu(theta1 | sigma1) = p / (p + (1 - p)lambda_d')
```

在观察到 `sigma2` 时，

```text
mu(theta1 | sigma2) = 0
```

因为只有 `theta2` 会发送 `sigma2`。

### 3.6.3 攻击者对 `sigma1` 的无差异条件

攻击者在 `sigma1` 后攻击的期望收益为：

```text
mu(theta1 | sigma1)(g_a - c_a) + (1 - mu(theta1 | sigma1))[-(c_a + l_i + kappa_a)]
```

在混合均衡中令其等于 0，并约去公共分母后可得：

```text
p(g_a - c_a) - (1 - p)lambda_d'(c_a + l_i + kappa_a) = 0
```

故：

```text
lambda_d' = p(g_a - c_a) / ((1 - p)(c_a + l_i + kappa_a))
```

### 3.6.4 `theta2` 型防御者的无差异条件

当 `theta2` 发送 `sigma2` 时，攻击者必然撤退，因此收益为：

```text
U_D(theta2, sigma2) = 0
```

当 `theta2` 发送 `sigma1` 时，攻击者以概率 `1 - lambda_a'` 发起攻击，因此收益为：

```text
U_D(theta2, sigma1) = (1 - lambda_a')(g_i + kappa_i) - c_theta2
```

令两者无差异：

```text
0 = (1 - lambda_a')(g_i + kappa_i) - c_theta2
```

从而：

```text
lambda_a' = (g_i + kappa_i - c_theta2) / (g_i + kappa_i)
```

### 3.6.5 命题 3-2

若满足：

```text
0 < p(g_a - c_a) / ((1 - p)(c_a + l_i + kappa_a)) < 1
0 < c_theta2 < g_i + kappa_i
g_a > c_a
```

则存在欺骗系统伪装混合均衡 PBNE-2，其混合概率为：

```text
lambda_d' = p(g_a - c_a) / ((1 - p)(c_a + l_i + kappa_a))
lambda_a' = (g_i + kappa_i - c_theta2) / (g_i + kappa_i)
```

### 3.6.6 证明

1. 在 `sigma2` 后，攻击者面对确定的 `theta2`，攻击收益为负，因此撤退是序贯最优的。
2. 若攻击者在 `sigma1` 后采用混合策略，则必须对攻击与撤退无差异，由此得到 `lambda_d'`。
3. 若 `theta2` 在 `sigma1` 与 `sigma2` 间采用混合策略，则必须无差异，由此得到 `lambda_a'`。
4. `theta1` 在该候选均衡中发送 `sigma1` 的收益不低于偏离到 `sigma2` 的收益，因此其最优响应为纯策略 `sigma1`。
5. 结合贝叶斯更新可知，该策略组合与信念系统构成 PBNE-2。

证毕。

## 3.7 模型解释与实验衔接

本章新模型的作用在于提供一个“可证明、可解释、可计算”的理论主干，而不是逐项复制真实攻防环境中的所有动态细节。CybORG 环境实验承担的角色是：

1. 检验 PBNE 推导出的信号伪装策略在真实攻防环境中是否仍具有优势；
2. 衡量攻击压力、关键主机失陷率、攻击样动作数量等环境层指标；
3. 为第四章或本章实验部分提供比纯数值仿真更强的现实支撑。

因此，理论层与环境层并非逐项一一等价，而是“理论给出可解释策略，环境验证其现实有效性”的关系。

## 3.8 写作边界说明

本稿中命题 3-1 与命题 3-2 的闭式解、无差异条件与可行性条件，均已与当前代码实现保持一致。若后续论文中继续引入完全动态的折扣多阶段效用，则本章均衡证明需要重新升级为动态信号博弈分析，复杂度将显著增加，不建议在当前毕业论文阶段采用。
