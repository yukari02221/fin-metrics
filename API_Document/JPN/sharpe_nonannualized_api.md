
# Sharpe（非年率）API 仕様

> 目的：投資リターン `R` と無リスク利回り `R_f` から **非年率 Sharpe Ratio** を決定的（再現可能）に計算する。

$$
\mathrm{SR}=\frac{\mathbb{E}[R - R_f]}{\mathrm{SD}[R - R_f]}
$$

- **分子**：超過リターン \(R - R_f\) の平均  
- **分母**：超過リターン \(R - R_f\) の標準偏差（既定は標本SD: `ddof=1`）  
- **非年率**：1期間あたりの SR。年率化は `sr * sqrt(periods_per_year)` を利用側で適用する

---

## 公開モジュール

外部プロジェクトからは **crate 直下**で利用できます。

```rust
use fin_metrics::sharpe::{
    SharpeCfg, SharpeOut,
    sharpe_nonannualized_const,
    sharpe_nonannualized_from_pair,
};
```

> crate 内部のドキュメント/テストでは `use crate::sharpe::{...}` を推奨

---

## 提供関数

### `sharpe_nonannualized_const(rets, rf_const, cfg) -> SharpeOut`

- **用途**：無リスク利回りが **定数** のとき
- **分子**：`E[R - rf_const]`
- **分母**：`SD[R]`（定数の減算は分散不変）

```rust
pub fn sharpe_nonannualized_const(
    rets: &[f64],
    rf_const: f64,
    cfg: SharpeCfg
) -> SharpeOut
```

### `sharpe_nonannualized_from_pair(rets, rfs, cfg) -> SharpeOut`

- **用途**：無リスク利回りが **時変**（期ごとに異なる）とき
- **分子**：`E[R_t - R_{f,t}]`
- **分母**：`SD[R_t - R_{f,t}]`（ペアごとの差分列を組み立ててからSDを計算）

```rust
pub fn sharpe_nonannualized_from_pair(
    rets: &[f64],
    rfs: &[f64],
    cfg: SharpeCfg
) -> SharpeOut
```

---

## 設定（`SharpeCfg`）

```rust
pub struct SharpeCfg {
    pub sum_cfg: SumCfg,        // 分子の加算（平均）設定
    pub std_cfg: StdCfg,        // 分母の標準偏差設定（ddof等）
    pub pair_policy: PairPolicy // (r, rf) ペアの非有限値の扱い
}
```

既定（`SharpeCfg::default()`）：
- `sum_cfg.non_finite = Drop`
- `std_cfg.non_finite = Drop`, `ddof = 1`, `use_fma = true`, `clamp_eps = 0.0`
- `pair_policy = Drop`

### `PairPolicy`（非有限値の扱い）

| Policy        | 動作                                                                 |
|---------------|----------------------------------------------------------------------|
| `Propagate`   | NaN/±Inf を検出したら **即伝播**（最終的に SR=NaN）                  |
| `Drop`        | 非有限を含むペアを **除外**                                         |
| `TreatAsZero` | 非有限を **0.0 に置換**                                             |

> 実装では `PairPolicy` を `NonFinitePolicy` にマッピングし、**分子/分母で一貫**した扱いにしています。

---

## 返り値（`SharpeOut`）

```rust
pub struct SharpeOut {
    pub sr: f64,             // 非年率 Sharpe Ratio
    pub mean_excess: MeanExcessOut, // 分子側の診断（平均・件数・フラグ）
    pub std: StdOut,         // 分母側の診断（SD・件数・フラグ）
    pub flags: SharpeFlags,  // 追加フラグ
}
```

### `SharpeFlags`

| フラグ名                 | 意味                                                                 |
|--------------------------|----------------------------------------------------------------------|
| `zero_std_dev`           | 分母SDが 0 または `clamp_eps` 以下 → SR は **定義不能（NaN）**       |
| `len_mismatch`           | `rets.len() != rfs.len()` を検出（`from_pair` のみ）                 |
| `propagated_non_finite`  | 非有限が **伝播** して NaN に落ちた                                 |

---

## 非有限値・時変 R_f の要点

- **定数 `R_f`**：分母は `SD[R]` を使用（分散は定数差で不変）
- **時変 `R_f`**：必ず `(r_i - rf_i)` 列を **同じ `PairPolicy`** で構築してから `SD` を取る  
  （分子と分母で**同じ母集団**を見ないと SR が歪むため）

---

## 代表的なエッジケース

- `PairPolicy::Propagate` で非有限発生 → SR は **NaN**（`propagated_non_finite = true`）
- 分母 SD が 0/極小 → SR は **NaN**（`zero_std_dev = true`）
- `from_pair` で系列長が異なる → `len_mismatch = true`（計算は `min(len)` で決定的に進行）

---

## 使用例

### 1) 定数 `R_f` のとき

```rust
use fin_metrics::sharpe::{SharpeCfg, sharpe_nonannualized_const};
use fin_metrics::mean_excess::PairPolicy;

fn main() {
    let rets = [0.01, 0.02, 0.03];
    let rf = 0.0;

    let mut cfg = SharpeCfg::default();
    cfg.pair_policy = PairPolicy::Drop; // 非有限は捨てる

    let out = sharpe_nonannualized_const(&rets, rf, cfg);
    println!("SR = {}", out.sr);
    // 診断も確認可能:
    // out.mean_excess.mean, out.mean_excess.n_used, out.std.sd, out.flags.zero_std_dev, ...
}
```

### 2) 時変 `R_f` のとき

```rust
use fin_metrics::sharpe::{SharpeCfg, sharpe_nonannualized_from_pair};

fn main() {
    let rets = [0.02, 0.02, 0.02];
    let rfs  = [0.01, 0.015, 0.005]; // 期ごとに違う無リスク利回り

    let out = sharpe_nonannualized_from_pair(&rets, &rfs, SharpeCfg::default());
    println!("SR = {}", out.sr);
}
```

---

## よくある質問（FAQ）

**Q. `ddof` って？**  
A. 標準偏差の自由度。既定は `1`（標本SD）。母集団SDにしたいなら `std_cfg.ddof = 0`。

**Q. 非有限（NaN/±Inf）は？**  
A. `pair_policy` で統一的に扱います。`Drop`=除外、`TreatAsZero`=0置換、`Propagate`=即NaN。

**Q. 年率化は？**  
A. 本APIは **非年率**。年率化したい場合は `sr * sqrt(periods_per_year)` を利用側で適用（例：日次=252、週次=52 など）。

---

## 設計ポリシー（要点）

- **決定性**：左→右の順次処理、KBNベース、設定に基づく再現性
- **一貫性**：分子/分母で同じ非有限ポリシー・同じ母集団
- **可観測性**：分子/分母の診断情報（件数・フラグ）を返す

---

## ライセンス

MIT / Apache-2.0 のデュアルライセンス
