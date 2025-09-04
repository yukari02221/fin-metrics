
# Sharpe (Non-Annualized) API — Friendly Guide

> Goal: deterministically (reproducibly) compute the **non-annualized Sharpe Ratio** from investment returns `R` and the risk‑free rate `R_f`.

$$
\mathrm{SR} = \frac{\mathbb{E}[R - R_f]}{\mathrm{SD}[R - R_f]}
$$

- **Numerator**: the average of excess returns \(R - R_f\)  
- **Denominator**: the standard deviation of excess returns \(R - R_f\) (default is sample SD, `ddof = 1`)  
- **Non-annualized**: SR per period. If you want annualized SR, apply `sr * sqrt(periods_per_year)` on the caller side.

---

## Public module

From external projects, use the **crate-root** public API:

```rust
use your_crate_name::sharpe::{
    SharpeCfg, SharpeOut,
    sharpe_nonannualized_const,
    sharpe_nonannualized_from_pair,
};
```

> Inside this crate’s docs/tests, prefer `use crate::sharpe::{...}`.

---

## Provided functions

### `sharpe_nonannualized_const(rets, rf_const, cfg) -> SharpeOut`

- **Use when** the risk‑free rate is **constant**.  
- **Numerator**: `E[R - rf_const]`  
- **Denominator**: `SD[R]` (subtracting a constant doesn’t change variance)

```rust
pub fn sharpe_nonannualized_const(
    rets: &[f64],
    rf_const: f64,
    cfg: SharpeCfg
) -> SharpeOut
```

### `sharpe_nonannualized_from_pair(rets, rfs, cfg) -> SharpeOut`

- **Use when** the risk‑free rate is **time‑varying** (different each period).  
- **Numerator**: `E[R_t - R_{f,t}]`  
- **Denominator**: `SD[R_t - R_{f,t}]` (build the per‑period differences and then compute SD)

```rust
pub fn sharpe_nonannualized_from_pair(
    rets: &[f64],
    rfs: &[f64],
    cfg: SharpeCfg
) -> SharpeOut
```

---

## Configuration (`SharpeCfg`)

```rust
pub struct SharpeCfg {
    pub sum_cfg: SumCfg,        // Numerator (mean) settings
    pub std_cfg: StdCfg,        // Denominator (std dev) settings, e.g., ddof
    pub pair_policy: PairPolicy // Handling of non-finite values in (r, rf) pairs
}
```

Defaults (`SharpeCfg::default()`):
- `sum_cfg.non_finite = Drop`
- `std_cfg.non_finite = Drop`, `ddof = 1`, `use_fma = true`, `clamp_eps = 0.0`
- `pair_policy = Drop`

### `PairPolicy` (how to handle non‑finite values)

| Policy        | Behavior                                                         |
|---------------|------------------------------------------------------------------|
| `Propagate`   | On NaN/±Inf, **propagate immediately** (final SR becomes NaN)    |
| `Drop`        | **Exclude** pairs containing non‑finite values                   |
| `TreatAsZero` | **Replace** non‑finite values with `0.0`                         |

> Internally, `PairPolicy` is mapped to `NonFinitePolicy` so numerator and denominator behave **consistently**.

---

## Return type (`SharpeOut`)

```rust
pub struct SharpeOut {
    pub sr: f64,                    // Non-annualized Sharpe Ratio
    pub mean_excess: MeanExcessOut, // Numerator diagnostics (mean, counts, flags)
    pub std: StdOut,                // Denominator diagnostics (SD, counts, flags)
    pub flags: SharpeFlags,         // Additional flags
}
```

### `SharpeFlags`

| Flag                   | Meaning                                                          |
|------------------------|------------------------------------------------------------------|
| `zero_std_dev`         | Denominator SD is 0 or ≤ `clamp_eps` → SR is **undefined (NaN)** |
| `len_mismatch`         | Detected `rets.len() != rfs.len()` (only in `from_pair`)         |
| `propagated_non_finite`| Non‑finite values **propagated** leading to NaN                  |

---

## Key points about non‑finite values & time‑varying `R_f`

- **Constant `R_f`**: the denominator uses `SD[R]` (variance unchanged by subtracting a constant).  
- **Time‑varying `R_f`**: always build the difference series `(r_i - rf_i)` under the **same `PairPolicy`**, then compute SD.  
  This ensures numerator and denominator are computed from the **same population**, avoiding distortion of SR.

---

## Typical edge cases

- With `PairPolicy::Propagate`, encountering a non‑finite value → SR **NaN** (`propagated_non_finite = true`).  
- Denominator SD is 0/very small → SR **NaN** (`zero_std_dev = true`).  
- In `from_pair`, different series lengths → `len_mismatch = true` (computation proceeds deterministically with `min(len)`).

---

## Usage examples

### 1) Constant `R_f`

```rust
use your_crate_name::sharpe::{SharpeCfg, sharpe_nonannualized_const};
use your_crate_name::mean_excess::PairPolicy;

fn main() {
    let rets = [0.01, 0.02, 0.03];
    let rf = 0.0;

    let mut cfg = SharpeCfg::default();
    cfg.pair_policy = PairPolicy::Drop; // drop non-finite pairs

    let out = sharpe_nonannualized_const(&rets, rf, cfg);
    println!("SR = {}", out.sr);
    // Diagnostics you can inspect:
    // out.mean_excess.mean, out.mean_excess.n_used, out.std.sd, out.flags.zero_std_dev, ...
}
```

### 2) Time‑varying `R_f`

```rust
use your_crate_name::sharpe::{SharpeCfg, sharpe_nonannualized_from_pair};

fn main() {
    let rets = [0.02, 0.02, 0.02];
    let rfs  = [0.01, 0.015, 0.005]; // risk-free rate differs by period

    let out = sharpe_nonannualized_from_pair(&rets, &rfs, SharpeCfg::default());
    println!("SR = {}", out.sr);
}
```

---

## FAQ

**Q. What is `ddof`?**  
A. The degrees of freedom for standard deviation. Default is `1` (sample SD). Set `std_cfg.ddof = 0` if you want the population SD.

**Q. How are non‑finite values (NaN/±Inf) handled?**  
A. Controlled by `pair_policy`. `Drop` excludes, `TreatAsZero` replaces with `0.0`, `Propagate` yields NaN immediately. The behavior is consistent across numerator and denominator.

**Q. How do I annualize SR?**  
A. This API returns a **non‑annualized** SR. To annualize, apply `sr * sqrt(periods_per_year)` at call‑site (e.g., 252 for daily, 52 for weekly).

---

## Design policies (at a glance)

- **Determinism**: left‑to‑right streaming, KBN‑based, reproducible given the same config.  
- **Consistency**: identical non‑finite policy and population for numerator and denominator.  
- **Observability**: returns diagnostics for both numerator and denominator (counts, flags).

---

## License

Dual-licensed under MIT / Apache-2.0.
