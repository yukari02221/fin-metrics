//! Module: `metrics::sum::kbn_ext`
//!
//! Kahan–Babuška–Neumaier (KBN) compensated summation for `f64`, with a
//! **deterministic** evaluation order and explicit **non-finite policies**.
//!
//! # Why
//! Floating-point addition is not associative; a naive `sum()` depends on the
//! numeric path. KBN maintains a running compensation term to reduce
//! catastrophic cancellation while staying fast and allocation-free on the sum
//! itself. We intentionally **avoid parallelization** to guarantee the same
//! result for the same input order across machines and runs.
//!
//! # Goals
//! - **Determinism:** strictly left-to-right, single-thread accumulation.
//! - **Non-finite policy:** explicit handling when `NaN` / `±Inf` are present.
//! - **Tiny-result zeroing:** optional `clamp_eps` that rounds tiny totals to `0.0`.
//!
//! # Behavior summary
//! - Input values are **pre-filtered** according to [`NonFinitePolicy`] and
//!   then summed with KBN.
//! - If `Propagate` is selected and any non-finite value appears, the function
//!   returns `NaN` (the summation step is skipped) with
//!   `flags.propagated_non_finite = true`.
//! - With `Drop`, non-finite values are removed from the sum and tracked via
//!   `n_dropped` and `flags.dropped_non_finite = true`.
//! - With `TreatAsZero`, non-finite values contribute `0.0` to the sum.
//! - If `clamp_eps > 0.0` and `|sum| < clamp_eps`, the total is returned as `0.0`.
//!
//! # Complexity
//! `O(n)` time. A temporary `Vec<f64>` is used only for the filtered values to
//! keep the summation pass simple and deterministic. Memory is `O(k)`, where
//! `k ≤ n` is the number of used elements.
//!
//! # Panics
//! This module does not intentionally panic.
//!
//! # Examples
//! Basic cancellation robustness (classic `[-1e16, 1.0, 1.0, 1e16]`):
//! ```rust
//! use fin_metrics::metrics::sum::kbn_ext::{sum_kbn, SumCfg};
//!
//! let xs = [-1e16, 1.0, 1.0, 1e16];
//! let out = sum_kbn(&xs, SumCfg::default());
//! assert!((out.sum - 2.0).abs() < 1e-9);
//! ```
//!
//! Non-finite policies:
//! ```rust
//! use fin_metrics::metrics::sum::kbn_ext::{sum_kbn, NonFinitePolicy, SumCfg};
//!
//! let xs = [1.0, f64::NAN, 2.0, f64::INFINITY, 3.0];
//!
//! let drop_out = sum_kbn(&xs, SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() });
//! assert_eq!(drop_out.n_used, 3);
//! assert_eq!(drop_out.n_dropped, 2);
//! assert!(drop_out.flags.dropped_non_finite);
//!
//! let prop_out = sum_kbn(&xs, SumCfg { non_finite: NonFinitePolicy::Propagate, ..Default::default() });
//! assert!(prop_out.sum.is_nan());
//! assert!(prop_out.flags.propagated_non_finite);
//!
//! let zero_out = sum_kbn(&xs, SumCfg { non_finite: NonFinitePolicy::TreatAsZero, ..Default::default() });
//! assert_eq!(zero_out.n_used, 5);
//! assert_eq!(zero_out.n_dropped, 0);
//! assert!((zero_out.sum - 6.0).abs() < 1e-15);
//! ```
//!
//! Tiny-result clamping:
//! ```rust
//! use fin_metrics::metrics::sum::kbn_ext::{sum_kbn, SumCfg};
//! let xs = [1e-20, -1e-20];
//! let out = sum_kbn(&xs, SumCfg { clamp_eps: 1e-18, ..Default::default() });
//! assert_eq!(out.sum, 0.0);
//! ```
//!
//! # See also
//! - <https://docs.rs/compensated-summation>

use compensated_summation::KahanBabuskaNeumaier;

/// How to handle non-finite inputs (`NaN`, `+∞`, `-∞`) during summation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NonFinitePolicy {
    /// As soon as any non-finite value is observed, **propagate** it:
    /// the result becomes `NaN`, and the computation is aborted.
    Propagate,
    /// **Drop** all non-finite values from the sum. They are counted in
    /// [`SumOut::n_dropped`] and marked in [`SumFlags::dropped_non_finite`].
    Drop,
    /// **Treat as zero**: replace each non-finite value by `0.0` and proceed.
    TreatAsZero,
}

/// Configuration for KBN summation.
#[derive(Clone, Copy, Debug)]
pub struct SumCfg {
    /// Policy for handling `NaN`/`±Inf`.
    pub non_finite: NonFinitePolicy,
    /// If `abs(total) < clamp_eps` after summation, return `0.0`.
    /// Set to `0.0` to disable.
    pub clamp_eps: f64,
}

impl Default for SumCfg {
    fn default() -> Self { Self {non_finite: NonFinitePolicy::Drop, clamp_eps: 0.0 } }
}

/// Diagnostic flags indicating how the input was treated.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumFlags {
    /// At least one non-finite value was **dropped** (policy = [`NonFinitePolicy::Drop`]).
    pub dropped_non_finite: bool,
    /// A non-finite value was **propagated** (policy = [`NonFinitePolicy::Propagate`]).
    pub propagated_non_finite: bool,
}

/// Result bundle for a KBN summation.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumOut {
    /// The final (possibly clamped) total.
    pub sum: f64,
    /// Count of values that contributed to the sum (after policy application).
    pub n_used: usize,
    /// Count of non-finite values that were **dropped**.
    pub n_dropped: usize,
    /// Diagnostic flags for auditing the run.
    pub flags: SumFlags,
}

#[inline]
/// Internal helper used by the public wrappers.
///
/// Applies the configured non-finite policy to the incoming iterator and then
/// performs **deterministic** Kahan–Babuška–Neumaier (KBN) summation.
/// See the module-level docs for the overall design goals (determinism,
/// non-finite policy, tiny-result clamping).
///
/// # Parameters
/// - `iter`: Any `IntoIterator<Item = f64>`. The **left-to-right** order of
///   elements is preserved and defines the numeric result.
/// - `cfg`: [`SumCfg`] controlling non-finite handling and tiny-result clamping.
///
/// # Returns
/// A [`SumOut`] bundle containing the final total and auditing counters/flags.
///
/// # Behavior notes
/// - If `cfg.non_finite == NonFinitePolicy::Propagate` and any `NaN`/`±Inf` is
///   encountered during the pre-filter, the function **skips summation** and
///   returns `sum = NaN` with `flags.propagated_non_finite = true`.
/// - With `Drop`, non-finite values are removed (`n_dropped` increments).
/// - With `TreatAsZero`, non-finite values are replaced by `0.0` and counted
///   as used.
///
/// # Complexity
/// `O(n)` time; an `O(k)` temporary `Vec<f64>` is created for the filtered
/// values (`k ≤ n`).
///
/// # Panics
/// This function does not intentionally panic and does not use `unsafe`.
fn sum_kbn_core<I>(iter: I, cfg: SumCfg) -> SumOut
where
    I: IntoIterator<Item = f64>,
{
    // Diagnostic bookkeeping returned to the caller (see SumOut / SumFlags).
    let mut flags   = SumFlags::default();
    let mut used    = 0usize;   // how many elements contributed to the sum
    let mut dropped = 0usize;   // how many non-finite values were dropped

    // Pre-filter step: apply the selected policy to each element while
    // preserving the original order (determinism).
    let filtered: Vec<f64> = iter.into_iter().filter_map(|v| match (v.is_finite(), cfg.non_finite) {
        // Finite: keep as-is.
        (true, _) => { used += 1; Some(v) },

        // Non-finite + Drop: exclude from the sum, record that we dropped.
        (false, NonFinitePolicy::Drop) => { dropped += 1; flags.dropped_non_finite = true; None },

        // Non-finite + TreatAsZero: include as 0.0 in the sum.
        (false, NonFinitePolicy::TreatAsZero) => { used += 1; Some(0.0) },

        // Non-finite + Propagate: mark and exclude; summation will be skipped.
        (false, NonFinitePolicy::Propagate) => { flags.propagated_non_finite = true; None },
    }).collect();

    // If propagation was requested and triggered, return NaN and skip KBN.
    if flags.propagated_non_finite {
        return SumOut { sum: f64::NAN, n_used: 0, n_dropped: dropped, flags };
    }

    let mut sum = filtered.iter().copied().sum::<KahanBabuskaNeumaier<f64>>().total();

    // Optional tiny-result zeroing to clean up cancellation leftovers.
    if cfg.clamp_eps > 0.0 && sum.abs() < cfg.clamp_eps {
        sum = 0.0;
    }

    // Return the final total along with auditing counters and flags.
    SumOut { sum, n_used: used, n_dropped: dropped, flags }
}


/// Public slice-based wrapper.
///
/// Convenience API for callers that already have a `&[f64]`. This preserves
/// the input order and delegates to the internal core function, ensuring
/// **deterministic** results for the same slice and config.
pub fn sum_kbn(xs: &[f64], cfg: SumCfg) -> SumOut {
    // Convert the slice into a left-to-right iterator and reuse the core.
    sum_kbn_core(xs.iter().copied(), cfg)
}

/// Public iterator-based wrapper.
///
/// Accepts any `IntoIterator<Item = f64>` (e.g., an owned `Vec<f64>`,
/// an iterator adaptor chain, or a generator). The element order is preserved,
/// and the computation delegates to the internal core function.
pub fn sum_kbn_iter<I>(iter: I, cfg: SumCfg) -> SumOut
where
    I: IntoIterator<Item = f64>,
{
    // Forward the iterator to the core without materializing a slice.
    sum_kbn_core(iter, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kbn_corrects_cancellation() {
        let xs = [-1e16, 1.0, 1.0, 1e16];
        let out = sum_kbn(&xs, SumCfg::default());
        assert!((out.sum - 2.0).abs() < 1e-9, "sum={}", out.sum);
    }

    #[test]
    fn kbn_handles_decimal_representation() {
        let xs = [0.1f64; 10];
        let out = sum_kbn(&xs, SumCfg::default());
        assert!((out.sum - 1.0).abs() < 1e-15, "sum={}", out.sum);
    }

    #[test]
    fn policy_drop_vs_propagate_vs_zero() {
        let xs = [1.0, f64::NAN, 2.0, f64::INFINITY, 3.0];
        let out_drop = sum_kbn(&xs, SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() });
        assert_eq!(out_drop.n_used, 3);
        assert_eq!(out_drop.n_dropped, 2);
        assert!(out_drop.flags.dropped_non_finite);

        let out_prop = sum_kbn(&xs, SumCfg { non_finite: NonFinitePolicy::Propagate, ..Default::default() });
        assert!(out_prop.sum.is_nan());
        assert!(out_prop.flags.propagated_non_finite);

        let out_zero = sum_kbn(&xs, SumCfg { non_finite: NonFinitePolicy::TreatAsZero, ..Default::default() });
        assert_eq!(out_zero.n_used, 5);
        assert_eq!(out_zero.n_dropped, 0);
        assert!((out_zero.sum - 6.0).abs() < 1e-15);
    }

    #[test]
    fn clamp_eps_zeroing() {
        let xs = [1e-20, -1e-20];
        let out = sum_kbn(&xs, SumCfg { clamp_eps: 1e-18, ..Default::default() });
        assert_eq!(out.sum, 0.0);
    }
}