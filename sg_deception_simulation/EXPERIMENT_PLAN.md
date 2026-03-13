# Chapter 3 Numerical Simulation Plan

## Alignment With Section 3.3

This simulation follows the model in Section 3.3:

- target type set: `Theta = {theta1, theta2}`
- signal set: `Sigma = {sigma1, sigma2}`
- attacker action set: `A = {attack, retreat}`
- finite horizon: `T`
- attacker belief on each target: `psi_n^t(theta1)`

Here:

- `theta1`: production system
- `theta2`: honeypot
- `sigma1`: normal signal
- `sigma2`: honeypot-like signal

## Utilities

The simulation uses the utility logic described in the thesis text:

- if `theta1` is attacked, attacker gains `g_a - c_a`, defender loses `l_a`
- if `theta2` is attacked, attacker loses `c_a + l_i`, defender gains `g_i`
- if a system sends a deceptive signal, defender pays camouflage cost
- if the attacker retreats, attacker utility is `0`

Camouflage costs are applied only when the sent signal does not match the true type:

- `theta1 -> sigma2`: cost `c_theta1`
- `theta2 -> sigma1`: cost `c_theta2`

## Belief Update

The thesis gives a discounted Bayesian update. To preserve the intended "recent signals matter more" behavior in simulation, belief is updated by discounted log-likelihood accumulation:

`score_t(theta) = log P(theta) + sum_{tau=1}^t w(t,tau) log lambda_D(sigma_tau | theta)`

`w(t,tau) = exp(-beta * (t - tau))`

Then normalize the two scores to obtain `psi_n^t(theta1)` and `psi_n^t(theta2)`.

## Three Strategy Regimes

### 1. Truthful Baseline

- `theta1` always sends `sigma1`
- `theta2` always sends `sigma2`
- attacker attacks after `sigma1`, retreats after `sigma2`

This is the no-camouflage benchmark.

### 2. PBNE-1: Production Camouflage Equilibrium

This matches Theorem 3.1, equilibrium pair `(lambda_D*, lambda_A*)`.

- `theta1` sends `sigma2` with probability
  `lambda_D* = ((1 - p)(c_a + l_i)) / (p(g_a - c_a))`
- `theta2` sends `sigma2` with probability `1`
- after observing `sigma2`, attacker attacks with probability
  `lambda_A* = 1 - c_theta1 / (2 l_a)`
- after observing `sigma1`, attacker attacks with probability `1`

### 3. PBNE-2: Honeypot Camouflage Equilibrium

This matches Theorem 3.1, equilibrium pair `(lambda_D', lambda_A')`.

- `theta1` sends `sigma1` with probability `1`
- `theta2` sends `sigma1` with probability
  `lambda_D' = p(g_a - c_a) / ((1 - p)(c_a + l_i))`
- after observing `sigma1`, attacker retreats with probability
  `lambda_A' = (g_i - c_theta2) / g_i`
- equivalently, after observing `sigma1`, attacker attacks with probability
  `1 - lambda_A'`
- after observing `sigma2`, attacker retreats with probability `1`

All equilibrium probabilities are clipped into `[epsilon, 1 - epsilon]` for numerical stability and to respect mixed-strategy feasibility.

## Experiment 1: Strategy Comparison

Because the two PBNEs are feasible in different prior regions, the strategy comparison should be reported as two sub-scenarios instead of forcing all three strategies into one identical interior setting.

### Scenario A

- higher probability that a target is a production system
- compare `truthful baseline` with `PBNE-1`

### Scenario B

- lower probability that a target is a production system
- compare `truthful baseline` with `PBNE-2`

### Outputs

- defender expected utility bar chart
- attacker expected utility bar chart
- belief trajectory `psi_n^t(theta1)` over stages
- final belief distribution
- attack probability under each strategy

### Claim To Verify

Compared with truthful disclosure, the PBNE camouflage strategy that matches the current prior regime can alter attacker belief dynamics and improve defender-side outcomes.

## Experiment 2: Sensitivity Analysis

Analyze how the equilibrium behavior changes when key parameters vary.

### Recommended parameters

- prior probability `p = P(theta1)`
- time discount `beta`
- camouflage cost `c_theta1` for PBNE-1
- camouflage cost `c_theta2` for PBNE-2

### Outputs

- defender expected utility versus parameter
- attack probability versus parameter
- final belief versus parameter
- equilibrium mixing probability versus parameter

### Claim To Verify

The effectiveness of the camouflage strategy depends on the prior environment structure, defender camouflage cost, and attacker memory discount.

## Current Deliverables In This Project

- simulation core
- strategy-comparison experiment
- sensitivity-analysis experiment
- machine-readable result files for later plotting
