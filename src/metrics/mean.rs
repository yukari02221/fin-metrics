//! Module: `metrics::mean`
//!
//! Deterministic arithmetic **mean** for `f64`, built on top of the Kahan–Babuška–Neumaier
//! (KBN) compensated summation from [`crate::metrics::sum::kbn_ext`] with explicit
//! **non-finite handling**.
//!
//! # Why
//! Floating-point addition is not associative. A naive `sum()` can vary with reduction order,
//! threads, or hardware. We compute the mean via a deterministic left-to-right KBN sum, then
//! divide by the **post-policy** count, so reruns over the same input order yield identical
//! results across machines.
//!
//! # Goals
//! - **Determinism:** strictly sequential accumulation; no internal parallelism.
//! - **Non-finite policy:** explicit control when `NaN` / `±Inf` are present (Drop / Propagate / TreatAsZero).
//! - **Tiny-result zeroing:** inherits `clamp_eps` semantics from the underlying sum (affects the total, hence the mean).
//!
//! # API Surface
//! - [`MeanOut`]: result bundle for the mean (value + diagnostics).
//! - [`mean`]: compute mean of a slice with a full [`SumCfg`].
//! - [`mean_with_policy`]: convenience wrapper to set only [`NonFinitePolicy`].
//! - [`mean_iter`]: iterator-based variant for streaming sources.
//!
//! # Non-finite policy semantics
//! Mirrors [`crate::metrics::sum::kbn_ext`]:
//! - **Propagate** → as soon as a non-finite is observed, the computation aborts,
//!   the mean is `NaN`, `n_used = 0`, and [`MeanOut::propagated_non_finite`] is `true`.
//! - **Drop** → non-finite inputs are removed; they are counted in [`MeanOut::n_dropped`].
//!   If **all** inputs are dropped, the mean is `NaN` with `n_used = 0`, `propagated_non_finite = false`.
//! - **TreatAsZero** → non-finite inputs are replaced by `0.0` and included in [`MeanOut::n_used`].
//!
//! # Determinism & reproducibility
//! The total is produced by deterministic KBN summation (`left→right`), then divided by
//! the deterministic `n_used`. Repeated runs with the same input order produce the same
//! bit pattern (subject to identical `SumCfg`).
//!
//! # Complexity
//! `O(n)` time, single pass. No heap allocation on the hot path for the slice variant.
//! The iterator variant does not allocate for well-behaved iterators.
//!
//! # Stability policy (SemVer)
//! - **Stable contract:** determinism, non-finite semantics, and error handling are stable.
//! - **Internal math:** we may refine compensated summation (still deterministic). Numerical
//!   differences should remain within typical `f64` tolerances while preserving the public contract.
//!
//! # Examples
//! ```rust
//! use fin_metrics::metrics::mean::{mean, mean_iter, MeanOut};
//! use fin_metrics::metrics::sum::kbn_ext::{SumCfg, NonFinitePolicy};
//!
//! // Basic mean on finite inputs
//! let xs = [1.0, 2.0, 3.0];
//! let cfg = SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() };
//! let out: MeanOut = mean(&xs, cfg);
//! assert!((out.mean - 2.0).abs() < 1e-15);
//! assert_eq!(out.n_used, 3);
//!
//! // Drop non-finite values
//! let ys = [1.0, f64::NAN, 4.0];
//! let out = mean(&ys, cfg);
//! assert!((out.mean - 2.5).abs() < 1e-15);
//! assert_eq!(out.n_dropped, 1);
//!
//! // Propagate policy
//! let zs = [1.0, f64::INFINITY, 3.0];
//! let cfg_p = SumCfg { non_finite: NonFinitePolicy::Propagate, ..Default::default() };
//! let out = mean(&zs, cfg_p);
//! assert!(out.mean.is_nan());
//! assert!(out.propagated_non_finite);
//!
//! // Iterator variant matches slice variant bit-for-bit
//! let v = vec![1.0, 2.0, 3.0];
//! let a = mean(&v, cfg).mean.to_bits();
//! let b = mean_iter(v.clone().into_iter(), cfg).mean.to_bits();
//! assert_eq!(a, b);
//! ```

use crate::metrics::sum::kbn_ext::{
    sum_kbn, sum_kbn_iter, NonFinitePolicy, SumCfg, SumOut as SumAggOut,
};

/// Result bundle returned by [`mean`], [`mean_with_policy`], and [`mean_iter`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanOut {
    /// Arithmetic mean of the inputs **after** applying the non-finite policy.
    /// `NaN` if the policy is `Propagate` or if `n_used == 0` (e.g., all inputs dropped).
    pub mean: f64,
    /// Count of values that contributed to the mean (i.e., the post-policy count).
    pub n_used: usize,
    /// Count of non-finite values that were **dropped** (policy = [`NonFinitePolicy::Drop`]).
    /// For `TreatAsZero`, these are not counted as dropped.
    pub n_dropped: usize,
    /// `true` if a non-finite was **propagated** (policy = [`NonFinitePolicy::Propagate`]).
    /// Use this to distinguish "propagated NaN" from "empty after dropping".
    pub propagated_non_finite: bool,
}

impl MeanOut {
    #[inline]
    fn from_sum(sum_out: SumAggOut) -> Self {
        let mean = if sum_out.flags.propagated_non_finite || sum_out.n_used == 0 {
            f64::NAN
        } else {
            sum_out.sum / (sum_out.n_used as f64)
        };
        Self {
            mean,
            n_used: sum_out.n_used,
            n_dropped: sum_out.n_dropped,
            propagated_non_finite: sum_out.flags.propagated_non_finite,
        }
    }
}

/// Compute the arithmetic mean of a slice using deterministic KBN summation.
///
/// - Applies the provided [`SumCfg`] (non-finite policy and `clamp_eps` for the total).
/// - Returns [`MeanOut`], including counts and propagation diagnostics.
///
/// # Determinism
/// Accumulates left-to-right via KBN (`sum_kbn`) and divides by the post-policy count.
/// Repeating this function on the same input order yields the same bit pattern.
///
/// # Returns
/// - `NaN` if `sum_cfg.non_finite == Propagate` and a non-finite was observed
///   (with `propagated_non_finite = true`), or if `n_used == 0` after dropping.
///
/// # Examples
/// ```
/// use fin_metrics::metrics::mean::{mean, MeanOut};
/// use fin_metrics::metrics::sum::kbn_ext::{SumCfg, NonFinitePolicy};
///
/// let xs = [0.5, 0.5, 1.5];
/// let cfg = SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() };
/// let out: MeanOut = mean(&xs, cfg);
/// assert!((out.mean - 0.8333333333333333).abs() < 1e-15);
/// ```
#[inline]
pub fn mean(xs: &[f64], sum_cfg: SumCfg) -> MeanOut {
    MeanOut::from_sum(sum_kbn(xs, sum_cfg))
}

/// Convenience wrapper to compute the mean by specifying only the non-finite policy.
///
/// Internally builds a [`SumCfg`] with the given [`NonFinitePolicy`] and default `clamp_eps`.
///
/// # Examples
/// ```
/// use fin_metrics::metrics::mean::mean_with_policy;
/// use fin_metrics::metrics::sum::kbn_ext::NonFinitePolicy;
///
/// let out = mean_with_policy(&[1.0, f64::NAN, 3.0], NonFinitePolicy::Drop);
/// assert!((out.mean - 2.0).abs() < 1e-15);
/// ```
pub fn mean_with_policy(xs: &[f64], non_finite: NonFinitePolicy) -> MeanOut {
    let sum_cfg = SumCfg { non_finite, ..Default::default() };
    MeanOut::from_sum(sum_kbn(xs, sum_cfg))
}

/// Iterator-based variant of [`mean`].
///
/// Accepts any `IntoIterator<Item = f64>` and applies the same deterministic
/// accumulation semantics via [`sum_kbn_iter`].
///
/// # Examples
/// ```
/// use fin_metrics::metrics::mean::mean_iter;
/// use fin_metrics::metrics::sum::kbn_ext::{SumCfg, NonFinitePolicy};
///
/// let v = vec![1.0, 2.0, 3.0];
/// let cfg = SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() };
/// let out = mean_iter(v.into_iter(), cfg);
/// assert!((out.mean - 2.0).abs() < 1e-15);
/// ```
#[inline]
pub fn mean_iter<I>(iter: I, sum_cfg: SumCfg) -> MeanOut
where 
    I: IntoIterator<Item = f64>,
{
    MeanOut::from_sum(sum_kbn_iter(iter, sum_cfg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::sum::kbn_ext::{NonFinitePolicy, SumCfg};

    #[test]
    fn mean_basic() {
        let xs = [1.0, 2.0, 3.0];
        let cfg = SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() };
        let out = mean(&xs, cfg);
        assert_eq!(out.n_used, 3);
        assert_eq!(out.n_dropped, 0);
        assert!(!out.propagated_non_finite);
        assert!((out.mean - 2.0).abs() < 1e-15);
    }

    #[test]
    fn mean_drop_non_finite() {
        let xs = [1.0, f64::NAN, 3.0, f64::INFINITY];
        let cfg = SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() };
        let out = mean(&xs, cfg);
        assert_eq!(out.n_used, 2);
        assert_eq!(out.n_dropped, 2);
        assert!(!out.propagated_non_finite);
        assert!((out.mean - 2.0).abs() < 1e-15);
    }

    #[test]
    fn mean_propagate_non_finite() {
        let xs = [1.0, f64::INFINITY, 3.0];
        let cfg = SumCfg { non_finite: NonFinitePolicy::Propagate, ..Default::default() };
        let out = mean(&xs, cfg);
        assert!(out.mean.is_nan());
        assert_eq!(out.n_used, 0);
        assert!(out.propagated_non_finite);
    }

    #[test]
    fn mean_all_dropped_results_nan() {
        let xs = [f64::NAN, f64::INFINITY];
        let cfg = SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() };
        let out = mean(&xs, cfg);
        assert!(out.mean.is_nan());
        assert_eq!(out.n_used, 0);
        assert_eq!(out.n_dropped, 2);
        assert!(!out.propagated_non_finite);
    }

    #[test]
    fn mean_iter_matches_slice_and_is_deterministic() {
        let xs = vec![1.0, 2.0, 3.0];
        let cfg = SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() };
        let a = mean(&xs, cfg);
        let b = mean_iter(xs.clone().into_iter(), cfg);
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        assert_eq!(a.n_used, b.n_used);
        assert_eq!(a.n_dropped, b.n_dropped);

        let c = mean(&xs, cfg).mean.to_bits();
        let d = mean(&xs, cfg).mean.to_bits();
        assert_eq!(c, d);
    }
}