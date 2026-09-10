# Volatility target contract

All model families must use the same causal input and rolling-60 volatility
target.

For forecast origin `t`:

```text
input:      r[t-59 : t+1]
target(h):  std(r[t+h-59 : t+h+1], ddof=0)
VaR return: r[t+1]
```

The target is `vol60[t+h]`, with the slice endpoint written as an exclusive
Python index. Therefore:

- `h=1` uses `r[t-58 : t+2]`: 59 observed returns plus `r[t+1]`;
- `h=3` uses rolling `std(60)` ending at `r[t+3]`;
- all horizons use exactly 60 returns;
- future returns may appear in the supervised label, but never in the input
  features at origin `t`.

The same target definition must be used during training, validation, test
export, and metric evaluation. `h=1` is not the one-observation standard
deviation proxy from the former implementation.
