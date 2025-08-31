# fin-metrics

`fin-metrics` is a Rust crate providing **deterministic implementations** of 
financial performance metrics such as the **Sharpe Ratio** and 
the **Probabilistic Sharpe Ratio (PSR)**.

### Features

- Deterministic calculation using compensated summation (Kahan / Neumaier)
- Clear handling of non-finite values (NaN / Inf) via configurable policies
- Fixed conventions:
  - `f64` precision
  - Sample variance (`ddof = 1`)
  - Sharpe Ratio in non-annualized form
  - PSR definition consistent with López de Prado
- Utility functions for mean and standard deviation with reproducible results

### Example

## References

1. Bailey, D. H., & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*. *Journal of Risk*, 15(2), 3–44.  
   SSRN preprint available at https://ssrn.com/abstract=1821643

## License
Licensed under either of
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.