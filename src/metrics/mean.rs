

use crate::metrics::sum::kbn_ext::{
    sum_kbn, sum_kbn_iter, NonFinitePolicy, SumCfg, SumOut as SumAggOut,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanOut {
    pub mean: f64,
    pub n_used: usize,
    pub n_dropped: usize,
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

#[inline]
pub fn mean(xs: &[f64], sum_cfg: SumCfg) -> MeanOut {
    MeanOut::from_sum(sum_kbn(xs, sum_cfg))
}

pub fn mean_with_policy(xs: &[f64], non_finite: NonFinitePolicy) -> MeanOut {
    let sum_cfg = SumCfg { non_finite, ..Default::default() };
    MeanOut::from_sum(sum_kbn(xs, sum_cfg))
}

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