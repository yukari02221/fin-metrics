//! Module: `metrics::sharpe`
//!
//! Deterministic **non-annualized Sharpe Ratio** implementation:
//!
//! \[ \mathrm{SR} = \frac{\mathbb{E}[R - R_f]}{\mathrm{SD}[R - R_f]} \]
//!
//! - **Numerator** uses this crate's [`metrics::mean_excess`] primitives.
//! - **Denominator** uses this crate's [`metrics::std_dev`] (sample SD; `ddof=1` by default).
//! - **Determinism**: accumulation order is strictly left→right and delegates to deterministic
//!   KBN-based routines, mirroring the style of `mean_excess`/`std_dev`.
//! - **Non-finite policy**: `(NaN, ±Inf)` handling is aligned between numerator/denominator by
//!   mapping [`PairPolicy`] to [`NonFinitePolicy`].
//!
//! # Non-annualized only
//! このモジュールは **非年率** SR を提供します（1 期間あたり）。年率化は呼び出し側で
//! `sr * sqrt(periods_per_year)` 等を適用してください（将来 `sharpe_annualized` を追加する拡張は容易です）。
//!
//! # API
//! - [`SharpeCfg`]: mean/std の設定と `(r, rf)` ペアの非有限ポリシーを束ねる設定構造体。
//! - [`SharpeOut`]: SR 値に加え、分子（[`MeanExcessOut`]) と分母（[`StdOut`]) の診断を同梱。
//! - [`sharpe_nonannualized_const`]: 定数 `rf_const` を用いる SR。分母は `SD[R]` を流用。
//! - [`sharpe_nonannualized_from_pair`]: 時変 `rfs` を用いる SR。分母は `SD[R - R_f]` を組み立てて計算。
//!
//! # Examples
//! ```
//! use fin_metrics::metrics::sharpe::{sharpe_nonannualized_const, SharpeCfg};
//! use fin_metrics::metrics::mean_excess::PairPolicy;
//!
//! let rets = [0.01, 0.02, 0.03];
//! let rf = 0.0;
//! let mut cfg = SharpeCfg::default();
//! cfg.pair_policy = PairPolicy::Drop; // drop non-finite by default
//!
//! let out = sharpe_nonannualized_const(&rets, rf, cfg);
//! assert!(out.sr.is_finite());
//! assert_eq!(out.mean_excess.n_used, 3);
//! ```
//!
//! ---
//! Licensed under **MIT OR Apache-2.0** at your option.
//!

use crate::metrics::mean_excess::{PairPolicy, MeanExcessOut, mean_excess_from_pair, mean_excess_const};
use crate::metrics::std_dev::{StdCfg, StdOut, std_dev_all};
use crate::metrics::sum::kbn_ext::{NonFinitePolicy, SumCfg, SumFlags};

/// Configuration bundle for non-annualized Sharpe Ratio.
#[derive(Debug, Clone, Copy)]
pub struct SharpeCfg {
    /// Summation config for the numerator (mean excess).
    pub sum_cfg: SumCfg,
    /// Std-dev config for the denominator.
    pub std_cfg: StdCfg,
    /// Pairwise non-finite handling for `(r_i, rf_i)`.
    pub pair_policy: PairPolicy,
}

impl Default for SharpeCfg {
    fn default() -> Self {
        Self {
            sum_cfg: SumCfg::default(),  // NonFinitePolicy::Drop, clamp_eps=0.0
            std_cfg: StdCfg::default(),  // NonFinitePolicy::Drop, ddof=1, use_fma=true
            pair_policy: PairPolicy::Drop,
        }
    }
}

/// Diagnostic flags for Sharpe computation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SharpeFlags {
    /// True if denominator SD is zero (or effectively zero per `clamp_eps`).
    pub zero_std_dev: bool,
    /// True if `rets.len() != rfs.len()` in the pair variant.
    pub len_mismatch: bool,
    /// True if either numerator or denominator propagated a non-finite.
    pub propagated_non_finite: bool,
}

/// Output bundle for Sharpe computation.
#[derive(Debug, Clone, Copy)]
pub struct SharpeOut {
    /// Non-annualized Sharpe Ratio value.
    pub sr: f64,
    /// Numerator diagnostics (mean excess block).
    pub mean_excess: MeanExcessOut,
    /// Denominator diagnostics (std dev block).
    pub std: StdOut,
    /// Additional flags.
    pub flags: SharpeFlags,
}

#[inline]
fn map_pair_to_nonfinite(p: PairPolicy) -> NonFinitePolicy {
    match p {
        PairPolicy::Propagate   => NonFinitePolicy::Propagate,
        PairPolicy::Drop        => NonFinitePolicy::Drop,
        PairPolicy::TreatAsZero => NonFinitePolicy::TreatAsZero,
    }
}

/// Compute non-annualized Sharpe Ratio when the risk-free rate is **constant**.
///
/// Numerator: `E[R - rf_const]` (via [`mean_excess_const`])  
/// Denominator: `SD[R]`（定数減算は分散不変）
pub fn sharpe_nonannualized_const(rets: &[f64], rf_const: f64, mut cfg: SharpeCfg) -> SharpeOut {
    // Align non-finite policy between numerator and denominator.
    cfg.std_cfg.non_finite = map_pair_to_nonfinite(cfg.pair_policy);

    let mex = mean_excess_const(rets, rf_const, cfg.pair_policy, cfg.sum_cfg);
    let sd  = std_dev_all(rets, cfg.std_cfg);

    let mut flags = SharpeFlags::default();
    flags.propagated_non_finite =
        mex.flags.propagated_non_finite || sd.flags.propagated_non_finite;

    if sd.sd.abs() <= cfg.std_cfg.clamp_eps || sd.sd == 0.0 {
        flags.zero_std_dev = true;
    }

    let sr = if flags.zero_std_dev || mex.mean.is_nan() || sd.sd.is_nan() {
        f64::NAN
    } else {
        mex.mean / sd.sd
    };

    SharpeOut { sr, mean_excess: mex, std: sd, flags }
}

/// Compute non-annualized Sharpe Ratio when the risk-free rate is **time-varying**.
///
/// Numerator: `E[R - R_f]`（[`mean_excess_from_pair`]）  
/// Denominator: `SD[R - R_f]` を **同一ポリシー**で構築して計算
pub fn sharpe_nonannualized_from_pair(rets: &[f64], rfs: &[f64], cfg: SharpeCfg) -> SharpeOut {
    let mex = mean_excess_from_pair(rets, rfs, cfg.pair_policy, cfg.sum_cfg);

    // Early-out for propagate semantics.
    if cfg.pair_policy == PairPolicy::Propagate && mex.flags.propagated_non_finite {
        return SharpeOut {
            sr: f64::NAN,
            mean_excess: mex,
            std: StdOut { sd: f64::NAN, n_used: 0, n_dropped: 0, flags: SumFlags { dropped_non_finite: false, propagated_non_finite: false } },
            flags: SharpeFlags {
                zero_std_dev: false,
                len_mismatch: rets.len() != rfs.len() || mex.flags.len_mismatch,
                propagated_non_finite: true,
            },
        };
    }

    // Build diffs deterministically, honoring the pair policy.
    let n = rets.len().min(rfs.len());
    let mut diffs = Vec::with_capacity(n);
    for i in 0..n {
        let (mut r, mut rf) = (rets[i], rfs[i]);
        match cfg.pair_policy {
            PairPolicy::Propagate => {
                if !r.is_finite() || !rf.is_finite() {
                    // will lead to NaN SD; break to keep cost small (determinism unaffected).
                    diffs.clear();
                    break;
                }
                diffs.push(r - rf);
            }
            PairPolicy::Drop => {
                if r.is_finite() && rf.is_finite() {
                    diffs.push(r - rf);
                }
            }
            PairPolicy::TreatAsZero => {
                if !r.is_finite() { r = 0.0; }
                if !rf.is_finite() { rf = 0.0; }
                diffs.push(r - rf);
            }
        }
    }

    let mut std_cfg = cfg.std_cfg;
    std_cfg.non_finite = map_pair_to_nonfinite(cfg.pair_policy);

    let sd = if cfg.pair_policy == PairPolicy::Propagate && diffs.is_empty() {
        // Propagation path: keep diagnostics explicit.
        StdOut { sd: f64::NAN, n_used: 0, n_dropped: 0, flags: SumFlags { dropped_non_finite: false, propagated_non_finite: false } }
    } else {
        std_dev_all(&diffs, std_cfg)
    };

    let mut flags = SharpeFlags::default();
    flags.len_mismatch = rets.len() != rfs.len() || mex.flags.len_mismatch;
    flags.propagated_non_finite =
        mex.flags.propagated_non_finite || sd.flags.propagated_non_finite;
    if sd.sd.abs() <= std_cfg.clamp_eps || sd.sd == 0.0 {
        flags.zero_std_dev = true;
    }

    let sr = if flags.zero_std_dev || mex.mean.is_nan() || sd.sd.is_nan() {
        f64::NAN
    } else {
        mex.mean / sd.sd
    };

    SharpeOut { sr, mean_excess: mex, std: sd, flags }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::mean_excess::PairPolicy;

    #[test]
    fn basic_const_rf() {
        let rets = [0.01, 0.02, 0.03];
        let rf = 0.0;
        let out = sharpe_nonannualized_const(&rets, rf, SharpeCfg::default());
        assert!(out.sr.is_finite());
        assert_eq!(out.mean_excess.n_used, 3);
        assert_eq!(out.std.n_used, 3);
        assert!(!out.flags.zero_std_dev);
    }

    #[test]
    fn zero_variance_returns() {
        let rets = [0.01f64; 5];
        let out = sharpe_nonannualized_const(&rets, 0.0, SharpeCfg::default());
        assert!(out.sr.is_nan());
        assert!(out.flags.zero_std_dev);
    }

    #[test]
    fn pair_drop_nan() {
        let rets = [0.01, f64::NAN, 0.03];
        let rfs  = [0.0,   0.0,       0.0];
        let mut cfg = SharpeCfg::default();
        cfg.pair_policy = PairPolicy::Drop;
        let out = sharpe_nonannualized_from_pair(&rets, &rfs, cfg);
        // second element dropped → effectively 2 samples
        assert_eq!(out.mean_excess.n_used, 2);
        assert!(out.sr.is_finite());
    }

    #[test]
    fn pair_propagate_nan() {
        let rets = [0.01, f64::NAN, 0.03];
        let rfs  = [0.0,   0.0,       0.0];
        let mut cfg = SharpeCfg::default();
        cfg.pair_policy = PairPolicy::Propagate;
        let out = sharpe_nonannualized_from_pair(&rets, &rfs, cfg);
        assert!(out.sr.is_nan());
        assert!(out.flags.propagated_non_finite);
    }

    #[test]
    fn len_mismatch_flag() {
        let rets = [0.01, 0.02, 0.03, 0.04];
        let rfs  = [0.0,  0.0,  0.0];
        let out = sharpe_nonannualized_from_pair(&rets, &rfs, SharpeCfg::default());
        assert!(out.flags.len_mismatch);
    }

    #[test]
    fn const_vs_pair_equivalence() {
        // rf_const == all-zeros rfs → SR should match (within floating tolerance)
        let rets = [0.01, 0.02, 0.03, -0.01, 0.0];
        let rfs  = [0.0; 5];
        let cfg = SharpeCfg::default();
        let a = sharpe_nonannualized_const(&rets, 0.0, cfg);
        let b = sharpe_nonannualized_from_pair(&rets, &rfs, cfg);
        assert!((a.sr - b.sr).abs() < 1e-12);
    }
}