# Chapter 3 Rewrite Draft

## Positioning

This rewrite keeps the Chapter 3 signaling-game structure, but replaces the original terminal utility with an effective one-shot utility that better captures:

- defender-side response burden under attack
- defensive value from deception engagement
- attacker-side penalty when attacking deceptive targets

This is the recommended theory layer to pair with the CybORG validation layer.

## 1. Game structure

Let the defender type be `theta in {theta1, theta2}`:

- `theta1`: production system
- `theta2`: honeypot/deceptive system

The defender sends signal `sigma in {sigma1, sigma2}`:

- `sigma1`: truthful production-like disclosure
- `sigma2`: camouflage/deceptive disclosure

The attacker observes the signal and chooses `a in {attack, retreat}`.

The prior belief is:

```text
Pr(theta = theta1) = p,   Pr(theta = theta2) = 1 - p
```

## 2. Revised utility

Define:

- `g_a`: attacker gain from successfully attacking a production target
- `c_a`: attacker attack cost
- `l_a`: defender loss when a production target is attacked
- `eta_d`: defender response burden caused by attack pressure
- `g_i`: defender intelligence gain from deception engagement
- `kappa_i`: extra defender deception bonus
- `l_i`: attacker loss when attacking a deceptive target
- `kappa_a`: extra attacker deception penalty
- `c(theta, sigma)`: signaling cost

The signaling cost is:

```text
c(theta1, sigma2) = c_theta1,   c(theta2, sigma1) = c_theta2
```

and zero otherwise.

Then the effective one-shot utilities are:

```text
U_D(theta1, attack | sigma) = -(l_a + eta_d) - c(theta1, sigma)
U_D(theta2, attack | sigma) = g_i + kappa_i - c(theta2, sigma)
U_D(theta, retreat | sigma) = -c(theta, sigma)
```

```text
U_A(theta1, attack | sigma) = g_a - c_a
U_A(theta2, attack | sigma) = -(c_a + l_i + kappa_a)
U_A(theta, retreat | sigma) = 0
```

## 3. Truthful baseline

In the truthful baseline:

```text
theta1 -> sigma1,   theta2 -> sigma2
```

and the attacker attacks after `sigma1`, retreats after `sigma2`.

This gives a separating benchmark.

## 4. PBNE-1: production-system camouflage

Consider the mixed regime:

```text
theta1: sigma1 with probability 1 - lambda_d^*, sigma2 with probability lambda_d^*
theta2: sigma2 with probability 1
```

and the attacker:

```text
after sigma1: attack
after sigma2: attack with probability lambda_a^*
```

### 4.1 Attacker indifference after sigma2

Conditioning on `sigma2`, the attacker is indifferent when:

```text
p * lambda_d^* (g_a - c_a) + (1 - p) * [-(c_a + l_i + kappa_a)] = 0
```

which gives:

```text
lambda_d^* = ((1 - p)(c_a + l_i + kappa_a)) / (p(g_a - c_a))
```

### 4.2 Defender indifference for theta1

Type `theta1` is indifferent between `sigma1` and `sigma2` when:

```text
-(l_a + eta_d) = -lambda_a^*(l_a + eta_d) - c_theta1
```

thus:

```text
lambda_a^* = 1 - c_theta1 / (l_a + eta_d)
```

### 4.3 Feasibility conditions

To keep a valid mixed equilibrium:

```text
0 < lambda_d^* < 1,   0 < lambda_a^* < 1
```

which implies:

```text
0 < ((1 - p)(c_a + l_i + kappa_a)) / (p(g_a - c_a)) < 1
0 < c_theta1 < l_a + eta_d
```

## 5. PBNE-2: honeypot camouflage

Consider the mixed regime:

```text
theta1: sigma1 with probability 1
theta2: sigma1 with probability lambda_d', sigma2 with probability 1 - lambda_d'
```

and the attacker:

```text
after sigma1: attack with probability 1 - lambda_a'
after sigma2: retreat
```

### 5.1 Attacker indifference after sigma1

The attacker is indifferent when:

```text
p(g_a - c_a) + (1 - p)lambda_d'[-(c_a + l_i + kappa_a)] = 0
```

so:

```text
lambda_d' = p(g_a - c_a) / ((1 - p)(c_a + l_i + kappa_a))
```

### 5.2 Defender indifference for theta2

Type `theta2` is indifferent between `sigma2` and `sigma1` when:

```text
0 = (1 - lambda_a')(g_i + kappa_i) - c_theta2
```

therefore:

```text
1 - lambda_a' = c_theta2 / (g_i + kappa_i)
```

equivalently:

```text
lambda_a' = (g_i + kappa_i - c_theta2) / (g_i + kappa_i)
```

### 5.3 Feasibility conditions

The mixed equilibrium requires:

```text
0 < lambda_d' < 1,   0 < lambda_a' < 1
```

which implies:

```text
0 < p(g_a - c_a) / ((1 - p)(c_a + l_i + kappa_a)) < 1
0 < c_theta2 < g_i + kappa_i
```

## 6. Why this rewrite is preferable

Compared with the old Chapter 3 utility, this rewrite has three advantages:

1. It preserves analytical tractability and closed-form mixing probabilities.
2. It explicitly prices attack pressure and deception value.
3. It is much easier to align with CybORG than the original terminal-only utility.

## 7. Recommended writing strategy for the thesis

Use this chapter structure:

1. Present the revised utility as an "effective stage utility".
2. Derive PBNE-1 and PBNE-2 with the closed forms above.
3. State feasibility conditions as propositions.
4. Use CybORG as the environment-level validation layer rather than forcing the theory to fully mirror the environment dynamics.
