//! Module: `metrics::std_dev`
//!
//! Deterministic **sample standard deviation (ddof=1 by default)** for `f64`,
//! designed for use as the denominator of the Sharpe Ratio.
//!
//! ## Design goals
//! - **Determinism**: strict left-to-right evaluation; no internal parallelism.
//! - **Explicit NaN/Inf policy**: match the crate's policy knobs (Drop / Propagate / TreatAsZero).
//! - **Numeric robustness**: two-pass algorithm (mean -> squared deviations),
//!   KBN accumulation for both passes, and optional FMA (`mul_add`) in the square.
//!
//! # Examples
//! ```rust,ignore
//! use crate::metrics::std_dev::{std_dev_all, StdCfg};
//! use crate::metrics::sum::kbn_ext::NonFinitePolicy;
//!
//! let xs = [1.0, 2.0, 3.0, 4.0];
//! let out = std_dev_all(&xs, StdCfg::default());
//! assert!((out.sd - 1.2909944487358056).abs() < 1e-12); // sample sd
//! assert_eq!(out.n_used, 4);
//! ```
//!
//! Policy behaviour (Drop / TreatAsZero / Propagate):
//! ```rust,ignore
//! use crate::metrics::std_dev::{std_dev_all, StdCfg};
//! use crate::metrics::sum::kbn_ext::NonFinitePolicy;
//!
//! let xs = [1.0, f64::NAN, 3.0];
//! let drop = std_dev_all(&xs, StdCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() });
//! assert_eq!(drop.n_used, 2);
//!
//! let zero = std_dev_all(&xs, StdCfg { non_finite: NonFinitePolicy::TreatAsZero, ..Default::default() });
//! assert_eq!(zero.n_used, 3);
//!
//! let prop = std_dev_all(&xs, StdCfg { non_finite: NonFinitePolicy::Propagate, ..Default::default() });
//! assert!(prop.sd.is_nan() && prop.flags.propagated_non_finite);
//! ```
use crate::metrics::sum::kbn_ext::{sum_kbn, NonFinitePolicy, SumCfg, SumFlags};

/// Configuration for sample standard deviation.
#[derive(Clone, Copy, Debug)]
pub struct StdCfg {
    /// Policy for handling `NaN`/`±Inf`.
    pub non_finite: NonFinitePolicy,
    /// If `abs(variance) < clamp_eps`, treat as `0.0` before `sqrt`.
    /// This clamps tiny negative variances arising from rounding.
    pub clamp_eps: f64,
    /// Use fused multiply-add `mul_add` for squaring (`d.mul_add(d, 0.0)`).
    /// Disable if you want to eliminate cross-architecture LSB drift.
    pub use_fma: bool,
    /// Degrees of freedom for the denominator (default = 1 for Sharpe).
    pub ddof: usize,
}

impl Default for StdCfg {
    fn default() -> Self {
        Self {
            non_finite: NonFinitePolicy::Drop,
            clamp_eps: 0.0,
            use_fma: true,
            ddof: 1,
        }
    }
}

/// Result bundle for [`std_dev_all`].
#[derive(Clone, Copy, Debug)]
pub struct StdOut {
    /// Sample standard deviation (or `NaN` if `n_used < 2` or policy propagated).
    pub sd: f64,
    /// Count of values that contributed (post-policy).
    pub n_used: usize,
    /// Count of non-finite values that were dropped (policy = Drop).
    pub n_dropped: usize,
    /// Diagnostic flags for the run (mirrors sum::kbn_ext).
    pub flags: SumFlags,
}

impl PartialEq for StdOut {
    fn eq(&self, other: &Self) -> bool {
        // Compare floats bitwise to treat -0.0/0.0 and NaN payloads deterministically.
        self.sd.to_bits() == other.sd.to_bits()
            && self.n_used == other.n_used
            && self.n_dropped == other.n_dropped
            && self.flags.dropped_non_finite == other.flags.dropped_non_finite
            && self.flags.propagated_non_finite == other.flags.propagated_non_finite
    }
}

#[inline]
pub fn std_dev_all(xs: &[f64], cfg: StdCfg) -> StdOut {
    // 1) Apply non-finite policy and materialize filtered data to ensure that
    //    mean and variance are computed on **the exact same sample set**.
    let mut flags = SumFlags::default();
    let mut n_used = 0usize;
    let mut n_dropped = 0usize;

    let mut filtered: Vec<f64> = Vec::with_capacity(xs.len());
    for &v in xs {
        match (v.is_finite(), cfg.non_finite) {
            (true, _) => { filtered.push(v); n_used += 1; }
            (false, NonFinitePolicy::Drop) => { n_dropped += 1; flags.dropped_non_finite = true; }
            (false, NonFinitePolicy::TreatAsZero) => { filtered.push(0.0); n_used += 1; }
            (false, NonFinitePolicy::Propagate) => { flags.propagated_non_finite = true; }
        }
        if flags.propagated_non_finite {
            // Short-circuit on Propagate: no need to continue scanning.
            break;
        }
    }

    // 2) Boundary conditions.
    if flags.propagated_non_finite {
        return StdOut { sd: f64::NAN, n_used: 0, n_dropped, flags };
    }
    if n_used < 2 || cfg.ddof >= n_used {
        return StdOut { sd: f64::NAN, n_used, n_dropped, flags };
    }

    // 3) First pass: mean of the filtered data via deterministic KBN.
    //    We call sum_kbn with Drop since filtered contains no non-finite values.
    let sum_out = sum_kbn(&filtered, SumCfg { non_finite: NonFinitePolicy::Drop, clamp_eps: 0.0 });
    // sum_out.n_used == n_used and n_dropped == 0 here by construction.
    let mean = sum_out.sum / (n_used as f64);

    // 4) Second pass: sum of squared deviations using KBN and optional FMA.
    let mut squares: Vec<f64> = Vec::with_capacity(n_used);
    squares.extend(filtered.iter().map(|&v| {
        let d = v - mean;
        if cfg.use_fma { d.mul_add(d, 0.0) } else { d * d }
    }));
    let ss_out = sum_kbn(&squares, SumCfg { non_finite: NonFinitePolicy::Drop, clamp_eps: 0.0 });

    // 5) Variance -> sd, with tiny negative clamped to zero if requested.
    let mut var = ss_out.sum / ((n_used - cfg.ddof) as f64);
    if cfg.clamp_eps > 0.0 && var.abs() < cfg.clamp_eps {
        var = 0.0;
    }
    let sd = var.sqrt();

    StdOut { sd, n_used, n_dropped, flags }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::sum::kbn_ext::NonFinitePolicy;

    #[test]
    fn basic_sample_sd() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        let out = std_dev_all(&xs, StdCfg::default());
        assert!((out.sd - 1.290_994_448_735_805_6).abs() < 1e-12);
        assert_eq!(out.n_used, 4);
        assert_eq!(out.n_dropped, 0);
        assert!(!out.flags.propagated_non_finite);
    }

    #[test]
    fn policy_drop_vs_zero_vs_propagate() {
        let xs = [1.0, f64::NAN, 3.0];

        let d = std_dev_all(&xs, StdCfg { non_finite: NonFinitePolicy::Drop, ..Default::default() });
        assert_eq!(d.n_used, 2);
        assert_eq!(d.n_dropped, 1);
        assert!(!d.sd.is_nan());

        let z = std_dev_all(&xs, StdCfg { non_finite: NonFinitePolicy::TreatAsZero, ..Default::default() });
        assert_eq!(z.n_used, 3);
        assert_eq!(z.n_dropped, 0);
        assert!(!z.sd.is_nan());

        let p = std_dev_all(&xs, StdCfg { non_finite: NonFinitePolicy::Propagate, ..Default::default() });
        assert!(p.sd.is_nan());
        assert_eq!(p.n_used, 0);
        assert!(p.flags.propagated_non_finite);
    }

    #[test]
    fn invariances() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::thread_rng();
        let mut xs: Vec<f64> = (0..100).map(|_| rng.gen_range(-5.0..5.0)).collect();

        let base = std_dev_all(&xs, StdCfg::default()).sd;

        // Shift invariance
        for v in &mut xs { *v += 123.456; }
        let shifted = std_dev_all(&xs, StdCfg::default()).sd;
        assert!((base - shifted).abs() < 1e-9);

        // Scale law
        for v in &mut xs { *v *= -7.0; }
        let scaled = std_dev_all(&xs, StdCfg::default()).sd;
        assert!((scaled.abs() - base.abs() * 7.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;
    use crate::metrics::sum::kbn_ext::NonFinitePolicy;

    #[test]
    fn edge_cases() {
        let empty: &[f64] = &[];
        let out = std_dev_all(empty, StdCfg::default());
        assert!(out.sd.is_nan());
        assert_eq!(out.n_used, 0);

        let single = [42.0];
        let out = std_dev_all(&single, StdCfg::default());
        assert!(out.sd.is_nan());
        assert_eq!(out.n_used, 1);


        let identical = [5.0, 5.0, 5.0, 5.0];
        let out = std_dev_all(&identical, StdCfg::default());
        assert!((out.sd).abs() < 1e-15);
        assert_eq!(out.n_used, 4);
    }

    #[test]
    fn ddof_variations() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        
        let pop = std_dev_all(&xs, StdCfg { ddof: 0, ..Default::default() });
        
        let sample = std_dev_all(&xs, StdCfg { ddof: 1, ..Default::default() });
        
        assert!(pop.sd < sample.sd);
        
        let invalid = std_dev_all(&xs, StdCfg { ddof: 5, ..Default::default() });
        assert!(invalid.sd.is_nan());
    }

    #[test]
    fn infinite_values() {
        let with_pos_inf = [1.0, f64::INFINITY, 3.0];
        let drop = std_dev_all(&with_pos_inf, StdCfg { 
            non_finite: NonFinitePolicy::Drop, 
            ..Default::default() 
        });
        assert!(!drop.sd.is_nan());
        assert_eq!(drop.n_used, 2);
        assert_eq!(drop.n_dropped, 1);

        let with_neg_inf = [1.0, f64::NEG_INFINITY, 3.0];
        let prop = std_dev_all(&with_neg_inf, StdCfg { 
            non_finite: NonFinitePolicy::Propagate, 
            ..Default::default() 
        });
        assert!(prop.sd.is_nan());
        assert!(prop.flags.propagated_non_finite);
    }

    #[test]
    fn clamp_eps_behavior() {
        let xs = [1.0000000000000001, 1.0, 0.9999999999999999];
        
        let no_clamp = std_dev_all(&xs, StdCfg { 
            clamp_eps: 0.0, 
            ..Default::default() 
        });
        
        let with_clamp = std_dev_all(&xs, StdCfg { 
            clamp_eps: 1e-10, 
            ..Default::default() 
        });
        
        println!("No clamp: {}, With clamp: {}", no_clamp.sd, with_clamp.sd);
    }

    #[test]
    fn fma_consistency() {
        let xs: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        
        let with_fma = std_dev_all(&xs, StdCfg { 
            use_fma: true, 
            ..Default::default() 
        });
        
        let without_fma = std_dev_all(&xs, StdCfg { 
            use_fma: false, 
            ..Default::default() 
        });
        
        let diff = (with_fma.sd - without_fma.sd).abs();
        assert!(diff < 1e-12, "FMA vs non-FMA difference: {}", diff);
    }

    #[test]
    fn large_dataset_performance() {
        let large_xs: Vec<f64> = (0..10_000).map(|i| (i as f64).sin()).collect();
        let out = std_dev_all(&large_xs, StdCfg::default());
        
        assert!(!out.sd.is_nan());
        assert_eq!(out.n_used, 10_000);
        assert_eq!(out.n_dropped, 0);
        assert!(out.sd > 0.5 && out.sd < 1.0);
    }

    #[test]
    fn mixed_non_finite_values() {
        let xs = [1.0, f64::NAN, 3.0, f64::INFINITY, 5.0, f64::NEG_INFINITY];
        
        let drop = std_dev_all(&xs, StdCfg { 
            non_finite: NonFinitePolicy::Drop, 
            ..Default::default() 
        });
        assert_eq!(drop.n_used, 3);   
        assert_eq!(drop.n_dropped, 3); 
        assert!(!drop.sd.is_nan());
        
        let zero = std_dev_all(&xs, StdCfg { 
            non_finite: NonFinitePolicy::TreatAsZero, 
            ..Default::default() 
        });
        assert_eq!(zero.n_used, 6);   
        assert_eq!(zero.n_dropped, 0);
        assert!(!zero.sd.is_nan());
    }

    #[test]
    fn partial_eq_implementation() {
        let xs = [1.0, 2.0, 3.0];
        let out1 = std_dev_all(&xs, StdCfg::default());
        let out2 = std_dev_all(&xs, StdCfg::default());
        
        assert_eq!(out1, out2);
        
        let nan_xs = [f64::NAN];
        let nan_out1 = std_dev_all(&nan_xs, StdCfg { 
            non_finite: NonFinitePolicy::Propagate, 
            ..Default::default() 
        });
        let nan_out2 = std_dev_all(&nan_xs, StdCfg { 
            non_finite: NonFinitePolicy::Propagate, 
            ..Default::default() 
        });
        
        assert_eq!(nan_out1, nan_out2);
    }
}
