//! Module: `metrics::sum::running_kbn`
//!
//! Deterministic **online** compensated summation (Neumaier) for `f64`,
//! aligned with the policies and diagnostics used in `metrics::sum::kbn_ext`.
//!
//! # Goals
//! - **Determinism:** strictly left-to-right updates; no parallelization here.
//! - **Non-finite policies:** reuse [`SumCfg`] / [`NonFinitePolicy`] / [`SumFlags`].
//! - **Tiny-result zeroing:** optional `clamp_eps` via [`SumCfg`].
//!
//! # Notes
//! This accumulator is intended to be the *streaming* counterpart of the
//! batch functions in `kbn_ext` (`sum_kbn`, `sum_kbn_iter`). If a
//! `NonFinitePolicy::Propagate` value is observed, the accumulator becomes
//! *poisoned* and reports `NaN` from [`RunningKbn::total`], mirroring the
//! batch behavior which skips KBN and returns `NaN` immediately.
//!
use super::kbn_ext::{SumCfg, NonFinitePolicy, SumOut, SumFlags};

/// Deterministic Neumaier-style compensated **running** sum.
#[derive(Clone, Debug)]
pub struct RunningKbn {
    sum: f64,       // main sum
    c: f64,         // compensation
    n_used: usize,  // how many values contributed (after policy application)
    n_dropped: usize, // how many non-finite values were dropped
    flags: SumFlags,
    cfg: SumCfg,
    poisoned: bool, // set when `Propagate` is triggered
}

impl Default for RunningKbn {
    #[inline]
    fn default() -> Self {
        Self::with_cfg(SumCfg::default())
    }
}

impl RunningKbn {
    /// Construct with default policy (`Drop` non-finite, `clamp_eps = 0.0`).
    #[inline]
    pub fn new() -> Self { Self::default() }

    /// Construct with an explicit configuration.
    #[inline]
    pub fn with_cfg(cfg: SumCfg) -> Self {
        Self {
            sum: 0.0,
            c: 0.0,
            n_used: 0,
            n_dropped: 0,
            flags: SumFlags::default(),
            cfg,
            poisoned: false,
        }
    }

    /// Add one value according to the configured non-finite policy.
    ///
    /// * `Propagate`  -> mark poisoned, future totals are `NaN`.
    /// * `Drop`       -> skip and count as dropped.
    /// * `TreatAsZero`-> include as `0.0` (counts as used).
    #[inline]
    pub fn add(&mut self, x: f64) {
        if self.poisoned {
            return; // already poisoned: keep state as-is
        }
        match (x.is_finite(), self.cfg.non_finite) {
            (true, _) => {
                // Neumaier update
                let t = self.sum + x;
                if self.sum.abs() >= x.abs() {
                    self.c += (self.sum - t) + x;
                } else {
                    self.c += (x - t) + self.sum;
                }
                self.sum = t;
                self.n_used += 1;
            }
            (false, NonFinitePolicy::Drop) => {
                self.n_dropped += 1;
                let mut f = self.flags;
                f.dropped_non_finite = true;
                self.flags = f;
            }
            (false, NonFinitePolicy::TreatAsZero) => {
                // treat as 0.0, count as used (to mirror batch behavior)
                self.n_used += 1;
                // sum unchanged
            }
            (false, NonFinitePolicy::Propagate) => {
                self.poisoned = true;
                let mut f = self.flags;
                f.propagated_non_finite = true;
                self.flags = f;
                // To mirror the batch-return semantics (n_used=0 on propagate),
                // we invalidate any previous usage count here.
                self.n_used = 0;
            }
        }
    }

    /// Current total (sum + compensation), with optional tiny-result zeroing.
    #[inline]
    pub fn total(&self) -> f64 {
        if self.poisoned || self.flags.propagated_non_finite {
            return f64::NAN;
        }
        let mut total = self.sum + self.c;
        if self.cfg.clamp_eps > 0.0 && total.abs() < self.cfg.clamp_eps {
            total = 0.0;
        }
        total
    }

    /// Snapshot the current state using the `SumOut` diagnostic shape.
    #[inline]
    pub fn snapshot(&self) -> SumOut {
        SumOut {
            sum: self.total(),
            n_used: if self.poisoned { 0 } else { self.n_used },
            n_dropped: self.n_dropped,
            flags: self.flags,
        }
    }

    /// Reset the accumulator to the initial clean state, preserving the config.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::with_cfg(self.cfg);
    }

    /// Whether a `Propagate` non-finite value has poisoned this accumulator.
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned || self.flags.propagated_non_finite
    }

    /// Merge another partial sum into this one (optional helper).
    ///
    /// Note: to preserve determinism, merging should be used with care in
    /// contexts where the exact order matters. This method simply folds the
    /// other's `(sum, c)` into `self` via two `add()` calls.
    pub fn merge(&mut self, other: &RunningKbn) {
        if self.is_poisoned() || other.is_poisoned() {
            self.poisoned = true;
            let mut f = self.flags;
            f.propagated_non_finite = true;
            self.flags = f;
            self.n_used = 0;
            return;
        }
        // combine diagnostics
        self.n_used += other.n_used;
        self.n_dropped += other.n_dropped;
        let mut f = self.flags;
        f.dropped_non_finite |= other.flags.dropped_non_finite;
        f.propagated_non_finite |= other.flags.propagated_non_finite;
        self.flags = f;

        // fold numerical state (order: sum then compensation)
        self.add(other.sum);
        self.add(other.c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::kbn_ext::{sum_kbn_iter};

    #[test]
    fn running_kbn_basic_sum() {
        let mut acc = RunningKbn::new();
        for x in [1.0, 2.0, 3.0] {
            acc.add(x);
        }
        assert_eq!(acc.total(), 6.0);
        let snap = acc.snapshot();
        assert_eq!(snap.n_used, 3);
        assert_eq!(snap.n_dropped, 0);
        assert!(!acc.is_poisoned());
    }

    #[test]
    fn running_kbn_cancellation() {
        let xs = [-1e16, 1.0, 1.0, 1e16];
        let mut acc = RunningKbn::new();
        for &x in &xs { acc.add(x); }
        assert!((acc.total() - 2.0).abs() < 1e-9, "total={}", acc.total());
    }

    #[test]
    fn running_vs_batch_match() {
        let cfg = SumCfg::default();
        let xs: Vec<f64> = (0..10_000).map(|i| if i%2==0 { 0.1 } else { -0.1 }).collect();
        let out_batch = sum_kbn_iter(xs.iter().copied(), cfg);
        let mut acc = RunningKbn::with_cfg(cfg);
        for &x in &xs { acc.add(x); }
        let total = acc.total();
        assert!((total - out_batch.sum).abs() < 1e-12, "running={}, batch={}", total, out_batch.sum);
    }

    #[test]
    fn non_finite_policies() {
        let xs = [1.0, f64::NAN, 2.0];

        // Drop: drop NaN, count dropped
        let mut acc_drop = RunningKbn::with_cfg(SumCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() });
        for &x in &xs { acc_drop.add(x); }
        let snap_drop = acc_drop.snapshot();
        assert_eq!(snap_drop.n_used, 2);
        assert_eq!(snap_drop.n_dropped, 1);
        assert!(snap_drop.flags.dropped_non_finite);
        assert!(!acc_drop.is_poisoned());
        assert!((acc_drop.total() - 3.0).abs() < 1e-15);

        // TreatAsZero: count as used, sum unaffected by NaN
        let mut acc_zero = RunningKbn::with_cfg(SumCfg { non_finite: NonFinitePolicy::TreatAsZero, ..Default::default() });
        for &x in &xs { acc_zero.add(x); }
        let snap_zero = acc_zero.snapshot();
        assert_eq!(snap_zero.n_used, 3);
        assert_eq!(snap_zero.n_dropped, 0);
        assert!((acc_zero.total() - 3.0).abs() < 1e-15);

        // Propagate: mark poisoned, total is NaN, n_used=0 in snapshot
        let mut acc_prop = RunningKbn::with_cfg(SumCfg { non_finite: NonFinitePolicy::Propagate, ..Default::default() });
        for &x in &xs { acc_prop.add(x); }
        assert!(acc_prop.total().is_nan());
        assert!(acc_prop.is_poisoned());
        let snap_prop = acc_prop.snapshot();
        assert_eq!(snap_prop.n_used, 0);
        assert_eq!(snap_prop.n_dropped, 0);
        assert!(snap_prop.flags.propagated_non_finite);
    }

    #[test]
    fn clamp_eps_zeroing() {
        let mut acc = RunningKbn::with_cfg(SumCfg { clamp_eps: 1e-18, ..Default::default() });
        acc.add(1e-20);
        acc.add(-1e-20);
        assert_eq!(acc.total(), 0.0);
    }
}
