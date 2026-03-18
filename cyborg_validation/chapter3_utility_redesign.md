# Chapter 3 Utility Redesign Notes

## Why the original utility is no longer sufficient

The CybORG experiments show a stable pattern:

- Environment-side blue reward improves under PBNE-inspired camouflage.
- Legacy Chapter 3 utility proxies can still move in the opposite direction.

This mismatch means the original utility only captures the terminal outcome of "production compromised" versus "honeypot compromised", but misses three effects that are important in a realistic multi-stage environment:

1. Attack pressure:
   repeated exploit/escalate/impact actions consume attacker resources and force defender response overhead even before final compromise.
2. Delay value:
   a compromise that occurs late in the episode is less harmful than one that occurs early.
3. Deception engagement:
   interacting with decoy hosts/services yields defensive value even when the attacker does not fully "commit" to a honeypot terminal state.

## Recommended revised utility

For stage `t = 1, 2, ..., T`, define:

- `rho in (0,1]`: temporal discount factor
- `I_t^prod`: indicator that a new production target is compromised at stage `t`
- `I_t^honeypot`: indicator that the attacker engages a deceptive target at stage `t`
- `I_t^attack`: indicator that an attack-like action occurs at stage `t`
- `c(theta, sigma_t)`: signaling/camouflage cost
- `eta_d`: defender-side attack pressure coefficient

Then use:

```text
U_D = sum_{t=1}^T rho^(t-1) [ g_i * I_t^honeypot - l_a * I_t^prod - eta_d * I_t^attack ] - sum_{t=1}^T c(theta, sigma_t)
```

```text
U_A = sum_{t=1}^T rho^(t-1) [ g_a * I_t^prod - l_i * I_t^honeypot - c_a * I_t^attack ]
```

## Interpretation

- `I_t^prod` is not "the attacker touched any host"; it means a new high-value production target was actually reached.
- `I_t^honeypot` should include deceptive engagement, not only terminal honeypot ownership.
- `I_t^attack` prices repeated exploit pressure, which the original one-shot utility ignored.
- `rho^(t-1)` makes late compromise less damaging and late intelligence less valuable, which fits multi-stage cyber confrontation better than a static one-shot payoff.

## Mapping to the current CybORG implementation

The current code in [run_thesis_scenario_validation.py](/Users/SL/Documents/expproject/cyberbattle-simulation-experiment/cyborg_validation/run_thesis_scenario_validation.py) already computes an intermediate "refined" utility approximation with:

- temporal discount
- new target compromise events
- per-stage camouflage cost
- attack-like pressure cost

This should be treated as a transition version of the revised theory utility, not the final paper formula.

## Suggested paper revision

If you revise Chapter 3, the cleanest change is:

1. Keep the signaling-game structure and PBNE derivation.
2. Replace the one-shot terminal utility with the discounted stage utility above.
3. Explain that in realistic APT environments, attack attempts, compromise delay, and deception engagement all have utility consequences.
4. Use the CybORG experiment section as empirical support for this revision.
