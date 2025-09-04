//! Public-facing Sharpe API (crate root re-exports).
//!
//! 他プロジェクトからは `use fin_metrics::sharpe::{...};` で直接使えるように再エクスポートします。
//! crate 内 doctest では `use crate::sharpe::{...}` を推奨します。
//!
//! # Examples
//! ```
//! use fin_metrics::sharpe::{SharpeCfg, sharpe_nonannualized_const};
//!
//! let rets = [0.01, 0.02, 0.03];
//! let out = sharpe_nonannualized_const(&rets, 0.0, SharpeCfg::default());
//! assert!(out.sr.is_finite());
//! ```
//!
pub use crate::metrics::sharpe::{
    SharpeCfg,
    SharpeFlags,
    SharpeOut,
    sharpe_nonannualized_const,
    sharpe_nonannualized_from_pair,
};