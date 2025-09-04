
# fin-metrics

`fin-metrics` is a Rust crate providing **deterministic, numerically-stable** implementations of financial performance metrics.
Current focus: **non-annualized Sharpe Ratio** with transparent handling of non-finite values.

---

## Highlights

- **Deterministic**: left-to-right streaming, Kahan/Neumaier–style compensated accumulation (KBN)
- **Numerically stable**: reproducible results in `f64`
- **Clear non-finite policies**: `Propagate` / `Drop` / `TreatAsZero`
- **Fixed conventions**:
  - Sample standard deviation (`ddof = 1`) by default
  - Non-annualized Sharpe Ratio (per-period)
- **Transparent diagnostics**: return structs include counts and flags (dropped / propagated / length mismatch)

---

## Quickstart

Add to your `Cargo.toml`:

```toml
[dependencies]
fin-metrics = "<latest>"
```

Then import from the **crate-root public API**:

```rust
use fin_metrics::sharpe::{
    SharpeCfg, SharpeOut,
    sharpe_nonannualized_const,
    sharpe_nonannualized_from_pair,
};
use fin_metrics::mean_excess::PairPolicy;

fn main() {
    // Example 1: constant risk-free rate
    let rets = [0.01, 0.02, 0.03];
    let mut cfg = SharpeCfg::default();
    cfg.pair_policy = PairPolicy::Drop;

    let out_const: SharpeOut = sharpe_nonannualized_const(&rets, 0.0, cfg);
    println!("SR(const) = {}", out_const.sr);

    // Example 2: time-varying risk-free rate
    let rfs = [0.005, 0.007, 0.006];
    let out_pair: SharpeOut = sharpe_nonannualized_from_pair(&rets, &rfs, SharpeCfg::default());
    println!("SR(pair) = {}", out_pair.sr);
}
```

> Note: In Rust imports, the hyphenated crate name `fin-metrics` becomes `fin_metrics`.

---

## API Summary

### Public (crate root)

- Module: `fin_metrics::sharpe` (re-exported public API)
  - `SharpeCfg`
  - `SharpeFlags`
  - `SharpeOut`
  - `sharpe_nonannualized_const(rets, rf_const, cfg) -> SharpeOut`
  - `sharpe_nonannualized_from_pair(rets, rfs, cfg) -> SharpeOut`

### Internals
- `metrics::mean_excess` — mean of excess returns (`mean_excess_const`, `mean_excess_from_pair`), `PairPolicy`
- `metrics::std_dev` — standard deviation (`StdCfg`, `StdOut`, `std_dev_all`)
- `metrics::sum::kbn_ext` — KBN-based accumulation (`SumCfg`, `SumFlags`, `NonFinitePolicy`)

---

## How SR is computed

$$
\mathrm{SR} = \frac{\mathbb{E}[R - R_f]}{\mathrm{SD}[R - R_f]}
$$

- If **risk-free is constant**: numerator uses `E[R - rf_const]`, denominator uses `SD[R]` (variance unchanged by subtracting a constant).
- If **risk-free is time-varying**: both numerator and denominator are computed on the **same per-period differences** `(r_i - rf_i)`, using the **same policy** for non-finite values.

### Non-finite handling (policies)

- `Propagate`: any NaN/±Inf → **propagate to NaN** and stop
- `Drop`: exclude non-finite pairs
- `TreatAsZero`: replace non-finite with `0.0`

We map `PairPolicy` → `NonFinitePolicy` so numerator/denominator behave consistently.

---

## Docs

- JP: `docs/sharpe_nonannualized_api.md`
- EN: `docs/sharpe_nonannualized_api_en.md`

---

## License

Licensed under either of
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

at your option.
