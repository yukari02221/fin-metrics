//! Mean Excess Return (a.k.a. mean excess over risk-free rate)
//!
//! # Overview
//! This module computes the **mean excess return**
//! \[ E[r - r_f] \], i.e., the average of per-period returns minus the
//! per-period risk-free rate. Two entry points are provided:
//!
//! - [`mean_excess_from_pair`]: takes `rets: &[f64]` and `rfs: &[f64]` and
//!   averages the element-wise differences.
//! - [`mean_excess_const`]: takes `rets: &[f64]` and a constant `rf: f64`.
//!
//! The implementation follows the crate’s **determinism** principles and
//! explicit **NaN/Inf policies**. Summation is performed via the crate’s
//! deterministic accumulation utilities governed by [`SumCfg`], so that
//! results are reproducible across platforms and runs.
//!
//! # Mathematical definition
//! Given two sequences of the same length `n > 0`,
//! `rets = (r_1, …, r_n)` and `rfs = (rf_1, …, rf_n)`,
//! the mean excess return is:
//!
//! ```text
//!   μ_excess = (1/n) * Σ_{i=1..n} (r_i - rf_i)
//! ```
//!
//! For the constant-rf variant, `rf_i = rf (const)`.
//!
//! # Non-finite handling (PairPolicy)
//! Financial time series often contain **NaN / ±Inf** due to data gaps,
//! circuit breakers, or upstream processing. We make the behavior explicit
//! via [`PairPolicy`]:
//!
//! - `Propagate`: If any used element is non-finite, **return NaN** immediately.
//!   No partial statistics are reported; `flags.propagated_non_finite = true`.
//! - `Drop`: **Skip** any pair whose `r_i` or `rf_i` is non-finite.
//!   Counts are tracked in the result (`n_dropped`).
//! - `TreatAsZero`: Replace any non-finite element with **0.0** before differencing.
//!
//! # Length mismatch
//! If `rets.len() != rfs.len()`, only the **first `min(len)`** pairs are
//! consumed and `flags.len_mismatch = true` is set. This mirrors the
//! “truncate to the shared window” convention common in time-series pipelines.
//!
//! # Determinism & numerical notes
//! - Summation uses the crate’s deterministic accumulator configured by [`SumCfg`]
//!   (e.g., compensated summation, optional `clamp_eps` to zero-out tiny residuals).
//! - No parallelism is used inside these functions. If you need parallelism,
//!   segment your data externally and combine deterministically.
//!
//! # Complexity
//! - Time: `O(n)`
//! - Memory: `O(1)` additional (streaming accumulation)
//!
//! # Returned metadata
//! Both entry points return a struct carrying:
//!
//! - `mean`: the mean excess value (`f64`)
//! - `n_used`: number of pairs contributing to the average
//! - `n_dropped`: number of pairs excluded by policy (only for `Drop`)
//! - `flags`: structured markers, including `len_mismatch` and `propagated_non_finite`
//!
//! # Examples
//! Clean input, pair variant:
//! ```rust,ignore
//! use crate::metrics::mean_excess::{mean_excess_from_pair, PairPolicy};
//! use crate::metrics::sum::{SumCfg};
//!
//! let rets = [0.10, 0.20, 0.30];
//! let rfs  = [0.05, 0.05, 0.05];
//! let out = mean_excess_from_pair(&rets, &rfs, PairPolicy::Propagate, SumCfg::default());
//! assert!((out.mean - 0.15).abs() < 1e-12);
//! assert_eq!(out.n_used, 3);
//! assert_eq!(out.n_dropped, 0);
//! assert!(!out.flags.len_mismatch);
//! assert!(!out.flags.propagated_non_finite);
//! ```
//!
//! Constant risk-free, equivalent to pairing with a constant vector:
//! ```rust,ignore
//! use crate::metrics::mean_excess::{mean_excess_const, mean_excess_from_pair, PairPolicy};
//! use crate::metrics::sum::SumCfg;
//!
//! let rets = [0.01, 0.02, 0.03, 0.04];
//! let rf   = 0.01;
//!
//! let a = mean_excess_const(&rets, rf, PairPolicy::Drop, SumCfg::default());
//! let rfs: Vec<f64> = core::iter::repeat(rf).take(rets.len()).collect();
//! let b = mean_excess_from_pair(&rets, &rfs, PairPolicy::Drop, SumCfg::default());
//! assert!((a.mean - b.mean).abs() < 1e-12);
//! assert_eq!(a.n_used, b.n_used);
//! ```
//!
//! Non-finite handling (`Drop` vs `TreatAsZero` vs `Propagate`):
//! ```rust,ignore
//! use crate::metrics::mean_excess::{mean_excess_from_pair, PairPolicy};
//! use crate::metrics::sum::SumCfg;
//!
//! let rets = [0.10, f64::NAN, 0.30];
//! let rfs  = [0.05, 0.05, 0.05];
//!
//! // Drop: skip the NaN pair → mean of [0.05, 0.25] = 0.15
//! let d = mean_excess_from_pair(&rets, &rfs, PairPolicy::Drop, SumCfg::default());
//! assert!((d.mean - 0.15).abs() < 1e-12);
//! assert_eq!(d.n_used, 2);
//! assert_eq!(d.n_dropped, 1);
//!
//! // TreatAsZero: NaN→0 → diffs=[0.05, -0.05, 0.25] → mean=1/12
//! let z = mean_excess_from_pair(&rets, &rfs, PairPolicy::TreatAsZero, SumCfg::default());
//! assert!((z.mean - (1.0/12.0)).abs() < 1e-12);
//!
//! // Propagate: any non-finite → NaN, n_used=0, flag set
//! let p = mean_excess_from_pair(&rets, &rfs, PairPolicy::Propagate, SumCfg::default());
//! assert!(p.mean.is_nan() && p.n_used == 0 && p.flags.propagated_non_finite);
//! ```
//!
//! Length mismatch (truncate to `min(len)`):
//! ```rust,ignore
//! use crate::metrics::mean_excess::{mean_excess_from_pair, PairPolicy};
//! use crate::metrics::sum::SumCfg;
//!
//! let rets = [0.10, 0.20, 0.30];
//! let rfs  = [0.05, 0.05]; // shorter
//! let out = mean_excess_from_pair(&rets, &rfs, PairPolicy::Drop, SumCfg::default());
//! assert!((out.mean - 0.10).abs() < 1e-12); // (0.05 + 0.15) / 2
//! assert!(out.flags.len_mismatch);
//! ```
//!
//! # Safety
//! - No `unsafe` code is used.
//! - Inputs are never mutated.
//!
//! # When should I choose each policy?
//! - `Propagate`: You require **strict** data cleanliness—any contamination should
//!   abort the measurement.
//! - `Drop`: You want a **robust** estimate that ignores sparse glitches,
//!   while tracking what was excluded.
//! - `TreatAsZero`: You model missing/invalid values as “no return” for that leg
//!   of the pair, which is appropriate only if this semantic is intended.
//!
//! # See also
//! - [`SumCfg`]: deterministic accumulation config (NaN/Inf policy for single-series
//!   pipelines, `clamp_eps`, etc.).
//! - Metrics built atop mean excess: `Sharpe`, `Information Ratio`, etc.
//!
//! ---
//! Licensed under **MIT OR Apache-2.0** at your option.

use crate::metrics::mean::{mean_iter, MeanOut};
use crate::metrics::sum::kbn_ext::SumCfg;
use compensated_summation::KahanBabuskaNeumaier; 

/// Policy controlling how `(r_i, rf_i)` pairs are handled when either side is non-finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairPolicy {
    /// Propagate any non-finite to the output:
    /// - return `mean = NaN`, set `flags.propagated_non_finite = true`
    /// - `n_used = 0`, `n_dropped = 0`
    Propagate,
    /// Drop any pair containing a non-finite component:
    /// - exclude it from the average
    /// - increment `n_dropped`
    Drop,
    /// Replace non-finite components with `0.0` before differencing.
    /// Use only if “no return” is a valid semantic for your pipeline.
    TreatAsZero,
}

/// Flags emitted by mean-excess computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeanExcessFlags {
    /// `true` if `PairPolicy::Propagate` encountered a non-finite value
    pub propagated_non_finite: bool,
    /// `true` if `rets.len() != rfs.len()` and we truncated to `min(len)`
    pub len_mismatch: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Result of a mean-excess computation.
pub struct MeanExcessOut {
    /// Mean of `(r_i - rf_i)` across the used pairs.
    pub mean: f64,
    /// Number of pairs used to compute the mean.
    pub n_used: usize,
    /// Number of pairs skipped due to policy (`Drop` only).
    pub n_dropped: usize,
    /// Additional diagnostic flags.
    pub flags: MeanExcessFlags,
}

impl MeanExcessOut {
    #[inline]
    fn from_mean(mean_out: MeanOut, dropped_pairs: usize, flags: MeanExcessFlags) -> Self {
        Self {
            mean: mean_out.mean,
            n_used: mean_out.n_used,
            n_dropped: dropped_pairs + mean_out.n_dropped,
            flags: MeanExcessFlags {
                propagated_non_finite: mean_out.propagated_non_finite || flags.propagated_non_finite,
                len_mismatch: flags.len_mismatch,
            },
        }
    }
}

fn mean_excess_from_pairs<I>(
    pairs: I,
    pair_policy: PairPolicy,
    sum_cfg: SumCfg,
    len_mismatch: bool,
) -> MeanExcessOut
where
    I: IntoIterator<Item = (f64, f64)>,
{
    let mut flags = MeanExcessFlags { len_mismatch, ..Default::default() };

    match pair_policy {
        PairPolicy::Propagate => {
            // 逐次で早期伝播
            let mut acc = KahanBabuskaNeumaier::<f64>::new();
            let mut used = 0usize;
            for (r, f) in pairs {
                if !r.is_finite() || !f.is_finite() {
                    flags.propagated_non_finite = true;
                    return MeanExcessOut { mean: f64::NAN, n_used: 0, n_dropped: 0, flags };
                }
                acc += r - f;
                used += 1;
            }
            let mean = if used == 0 { f64::NAN } else { acc.total() / (used as f64) };
            return MeanExcessOut { mean, n_used: used, n_dropped: 0, flags };
        }

        PairPolicy::Drop | PairPolicy::TreatAsZero => {
            // diffs をストリームで生成 → mean_iter に委譲
            let mut dropped_pairs = 0usize;
            let diffs = pairs.into_iter().filter_map(|(r, f)| {
                match pair_policy {
                    PairPolicy::Drop => {
                        if r.is_finite() && f.is_finite() { Some(r - f) }
                        else { dropped_pairs += 1; None }
                    }
                    PairPolicy::TreatAsZero => {
                        let rr = if r.is_finite() { r } else { 0.0 };
                        let ff = if f.is_finite() { f } else { 0.0 };
                        Some(rr - ff)
                    }
                    PairPolicy::Propagate => unreachable!(),
                }
            });

            let mean_out = mean_iter(diffs, sum_cfg);
            return MeanExcessOut::from_mean(mean_out, dropped_pairs, flags);
        }
    }
}

/// Computes mean excess return from paired series.
///
/// See module-level docs for policies, determinism, and examples.
///
/// ## Parameters
/// - `rets`: per-period returns
/// - `rfs`: per-period risk-free rates (paired with `rets`)
/// - `policy`: non-finite handling policy (see [`PairPolicy`])
/// - `sum_cfg`: deterministic accumulation config (see [`SumCfg`])
///
/// ## Returns
/// A [`MeanExcessOut`] carrying the mean, counts, and flags.
///
/// ## Panics
/// Never panics.
///
/// ## Notes
/// - If `rets.len() != rfs.len()`, only the first `min(len)` pairs are consumed
///   and `flags.len_mismatch = true` is set.
/// - When `policy = PairPolicy::Propagate`, the function returns `mean = NaN` on
///   first non-finite and sets `flags.propagated_non_finite = true`.
pub fn mean_excess_from_pair(
    rets: &[f64],
    rfs: &[f64],
    pair_policy: PairPolicy,
    sum_cfg: SumCfg
) -> MeanExcessOut {
    let len_mismatch = rets.len() != rfs.len();
    let n = core::cmp::min(rets.len(), rfs.len());
    let pairs = (0..n).map(move |i| (rets[i], rfs[i]));
    mean_excess_from_pairs(pairs, pair_policy, sum_cfg, len_mismatch)
}

/// Computes mean excess return with a **constant** risk-free rate.
///
/// This is equivalent to pairing `rets` with a constant vector `rf`, but avoids
/// materializing that vector and may be slightly faster.
///
/// See [`mean_excess_from_pair`] for behavior details.
///
/// ## Parameters
/// - `rets`: per-period returns
/// - `rf`: constant per-period risk-free rate
/// - `policy`: non-finite handling policy (see [`PairPolicy`])
/// - `sum_cfg`: deterministic accumulation config (see [`SumCfg`])
///
/// ## Returns
/// A [`MeanExcessOut`] carrying the mean, counts, and flags.
///
/// ## Panics
/// Never panics.
pub fn mean_excess_const(
    rets: &[f64],
    rf_const: f64,
    pair_policy: PairPolicy,
    sum_cfg: SumCfg
) -> MeanExcessOut {
    let pairs = rets.iter().copied().map(move |r| (r, rf_const));
    mean_excess_from_pairs(pairs, pair_policy, sum_cfg, /*len_mismatch=*/false)
}

pub fn mean_excess_iter<R, F>(
    rets: R,
    rfs:  F,
    pair_policy: PairPolicy,
    sum_cfg: SumCfg,
) -> MeanExcessOut
where
    R: IntoIterator<Item = f64>,
    F: IntoIterator<Item = f64>,
{
    let mut it_r = rets.into_iter();
    let mut it_f = rfs.into_iter();
    let mut ended_even = true;

    let pairs = core::iter::from_fn(move || {
        match (it_r.next(), it_f.next()) {
            (Some(r), Some(f)) => Some((r, f)),
            (None, None) => None,
            _ => { ended_even = false; None }
        }
    });

    mean_excess_from_pairs(pairs, pair_policy, sum_cfg, /*len_mismatch=*/!ended_even)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn approx_eq(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "abs diff={} > tol={}, a={}, b={}",
            (a - b).abs(),
            tol,
            a,
            b
        );
    }

    fn cfg() -> SumCfg { SumCfg::default() }

    #[test]
    fn propagate_policy_clean_input() {
        let rets = [0.10, 0.20, 0.30];
        let rfs  = [0.05, 0.05, 0.05];
        let out = mean_excess_from_pair(&rets, &rfs, PairPolicy::Propagate, cfg());
        approx_eq(out.mean, 0.15, 1e-12);
        assert_eq!(out.n_used, 3);
        assert_eq!(out.n_dropped, 0);
        assert!(!out.flags.propagated_non_finite);
        assert!(!out.flags.len_mismatch);
    }

    #[test]
    fn propagate_policy_propagates_nan() {
        let rets = [0.10, f64::NAN, 0.30];
        let rfs  = [0.05, 0.05, 0.05];
        let out = mean_excess_from_pair(&rets, &rfs, PairPolicy::Propagate, cfg());
        assert!(out.mean.is_nan());
        assert_eq!(out.n_used, 0);
        assert_eq!(out.n_dropped, 0);
        assert!(out.flags.propagated_non_finite);
        assert!(!out.flags.len_mismatch);
    }

    #[test]
    fn drop_policy_drops_non_finite_pairs() {
        let rets = [0.10, f64::NAN, 0.30];
        let rfs  = [0.05, 0.05, 0.05];
        let out = mean_excess_from_pair(&rets, &rfs, PairPolicy::Drop, cfg());
        approx_eq(out.mean, 0.15, 1e-12);
        assert_eq!(out.n_used, 2);
        assert_eq!(out.n_dropped, 1);
        assert!(!out.flags.propagated_non_finite);
        assert!(!out.flags.len_mismatch);
    }

    #[test]
    fn treat_as_zero_policy_replaces_non_finite_with_zero() {
        let rets = [0.10, f64::NAN, 0.30];
        let rfs  = [0.05, 0.05, 0.05];
        let out = mean_excess_from_pair(&rets, &rfs, PairPolicy::TreatAsZero, cfg());
        approx_eq(out.mean, 1.0/12.0, 1e-12);
        assert_eq!(out.n_used, 3);
        assert_eq!(out.n_dropped, 0);
        assert!(!out.flags.propagated_non_finite);
        assert!(!out.flags.len_mismatch);
    }

    #[test]
    fn length_mismatch_sets_flag_and_uses_min_len() {
        let rets = [0.10, 0.20, 0.30];
        let rfs  = [0.05, 0.05];
        let out = mean_excess_from_pair(&rets, &rfs, PairPolicy::Drop, cfg());
        approx_eq(out.mean, 0.10, 1e-12);
        assert_eq!(out.n_used, 2);
        assert_eq!(out.n_dropped, 0);
        assert!(out.flags.len_mismatch);
        assert!(!out.flags.propagated_non_finite);
    }

    #[test]
    fn const_rf_matches_pair_version() {
        let rets = [0.01, 0.02, 0.03, 0.04];
        let rf   = 0.01;
        let rfs: Vec<f64> = core::iter::repeat(rf).take(rets.len()).collect();

        let a = mean_excess_const(&rets, rf, PairPolicy::Drop, cfg());
        let b = mean_excess_from_pair(&rets, &rfs, PairPolicy::Drop, cfg());

        approx_eq(a.mean, b.mean, 1e-12);
        assert_eq!(a.n_used, b.n_used);
        assert_eq!(a.n_dropped, b.n_dropped);
        assert_eq!(a.flags.len_mismatch, b.flags.len_mismatch);
        assert_eq!(a.flags.propagated_non_finite, b.flags.propagated_non_finite);
    }
}
