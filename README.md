# Statistical-Arbitrage-in-Crypto-Currencies
This works was based on work done by the user JonnyTung123. The proposed strategy uses dual cointegration test using both ADF and Johansen trace test to select trading pairs - meant to improve robustness.

# Crypto Statistical Arbitrage - Cointegration-Based Pairs Trading

A walk-forward pairs trading strategy for cryptocurrency markets, built on cointegration theory. The strategy identifies pairs of coins whose log-price spread is stationary, trades mean-reversion in that spread, and evaluates performance out-of-sample against a buy-and-hold benchmark.

---

## Overview

The strategy operates on daily OHLCV data sourced from CoinGecko. Starting from a universe of the largest cryptocurrencies by market cap (filtered for data completeness), it:

1. Selects cointegrated pairs using a **dual ADF + Johansen test** on a rolling in-sample window
2. Generates z-score signals from a rolling OLS spread
3. Enters mean-reversion trades when the spread deviates beyond +/-1sigma and exits when it reverts within a configurable threshold
4. Re-selects pairs every 6 months to adapt to the changing crypto market structure
5. Evaluates net returns after transaction costs, with stop-losses on diverging spreads

---

## Strategy Logic

### Universe Construction

Coins are filtered from CoinGecko daily price data starting January 2018. Stablecoins and wrapped tokens (`TUSD`, `DAI`, `WBTC`, `WETH`, `USDC`) are excluded as they would trivially cointegrate. Any coin with more than 10% missing data over the full sample is dropped.

### Pair Selection

All combinations of coins are tested for cointegration over a trailing 1-year in-sample window. A pair is accepted only if it passes **both**:

- **Engle-Granger ADF test** - OLS residuals of `log(P_i)` regressed on `log(P_j)` must be stationary at the 5% significance level
- **Johansen trace test** - the trace statistic must exceed the 5% critical value, confirming at least one cointegrating vector

For each coin, the best-cointegrated partner (lowest ADF test statistic) is selected, giving at most one active pair per coin.

### Signal Generation

For each accepted pair, a rolling 90-day OLS hedge ratio (beta) and intercept (alpha) are estimated:

```
spread(t) = log(P_i(t)) - beta(t) * log(P_j(t)) - alpha(t)
z(t)      = (spread(t) - mean(spread)) / std(spread)
```

All rolling statistics use only data up to time `t` - no look-ahead.

### Position Sizing

| Condition | Action |
|---|---|
| z > +1sigma | Short asset i, long beta units of asset j |
| z < -1sigma | Long asset i, short beta units of asset j |
| \|z\| <= exit threshold | Close position (mean-reversion signal) |
| \|z\| > 3sigma | Stop-loss exit (broken pair) |

Positions across all active pairs are summed per asset and normalised so absolute weights sum to 1 (fully invested).

### Transaction Costs

A flat 10 bps per-side cost is applied to daily portfolio turnover:

```
net_return(t) = gross_return(t) - turnover(t) * 10bps
```

### Walk-Forward Rebalancing

Pairs are re-selected every 6 months using the prior 12 months as in-sample data. This prevents the pair set from becoming stale as crypto market structure evolves.

---

## Installation

```bash
pip install numpy pandas matplotlib statsmodels joblib
```

Python 3.9+ is recommended. All dependencies are standard scientific Python - no proprietary libraries required.

---

## Usage

Update the data path at the top of `stat_arbitrage_improved.py`:

```python
from_pickle = pd.read_pickle('/path/to/your/CoinGecko_px_vol_1D.pkl')
```

Then run:

```bash
python3 stat_arbitrage_improved.py
```

The backtest is parameterised via `run_backtest()` - key arguments:

| Parameter | Default | Description |
|---|---|---|
| `end_of_insample` | `'2018-12-31'` | Last in-sample date; OOS starts the next day |
| `thresholds` | `[0.1, 0.2, 0.5, 0.7]` | Exit z-score thresholds to sweep |
| `signal_window` | `90` | Rolling window (days) for beta/spread estimation |
| `tcost_bps` | `10` | One-way transaction cost in basis points |

---

## Improvements Over the Original

The original `stat_arbitrage.py` contained several bugs and design weaknesses. The improved version (`stat_arbitrage_improved.py`) addresses all of them.

### Bug Fixes

**`full_portfolio` not reset between threshold iterations**

The portfolio DataFrame was initialised once outside the threshold loop. Each subsequent threshold iteration was writing into the same object, so iterations 2-4 stacked on top of iteration 1's positions. This made every threshold after the first completely invalid. Fixed by moving initialisation inside the loop.

```python
# Before - shared across all thresholds (bug)
full_portfolio = pd.DataFrame(...)
for i, threshold in enumerate(thresholds):
    ...

# After - reset for each threshold
for i, threshold in enumerate(thresholds):
    full_portfolio = pd.DataFrame(...)  # clean slate
    ...
```

**Position overwriting in `gen_port`**

When an asset appeared in multiple pairs, `pos.loc[condition, asset] = value` silently overwrote the earlier pair's position with the later one's. For coins with many cointegrated partners this meant only the last pair in the loop was ever actually traded. Fixed by computing positions per-pair in separate DataFrames and summing them.

```python
# Before - last pair wins (bug)
pos.loc[z > 1, asset_i] = -1

# After - accumulate across pairs
pair_pos.loc[z > 1, asset_i] = -1
...
pos = sum(pair_positions)  # correct aggregation
```

### Correctness Fixes

**`duration()` called on `cumsum()` instead of `cumprod()`**

`cumsum()` of returns is not an equity curve - it has no meaningful "peak" or drawdown duration. `duration()` expects a price-level series. Fixed by passing `(1 + returns).cumprod()`.

```python
# Before
ddd = duration(full_sample_ret.cumsum())

# After
equity_curve = (1 + full_sample_ret).cumprod()
ddd = duration(equity_curve)
```

**Look-ahead bias in `gen_signals`**

The original code passed the full `crypto_px` DataFrame to `gen_signals`, meaning the rolling window could in principle reference prices from beyond the current out-of-sample period. Fixed by slicing the price data to `[insample_start : end_date]` before calling signal generation.

**Spread direction inconsistency**

`adf_for_pair` regressed `log_j` on `log_i` (Y on X), but `gen_signals` computed the spread as `log_i - beta*log_j - alpha`. The OLS beta from pair selection and the rolling beta used for trading now both follow the same convention, so the cointegration test and the traded spread are consistent.

**NaN/inf values crashing OLS**

`np.log(0)` produces `-inf`, and `ffill()` alone cannot handle leading NaN values at the start of a series. Added `.bfill()` after `.ffill()`, plus explicit `replace([np.inf, -np.inf], np.nan).dropna()` before the regression. A minimum observation guard (30 days) ensures degenerate pairs are rejected cleanly rather than crashing.

### Robustness Improvements

**Single cointegration test -> dual test**

The original used only the ADF test (Engle-Granger step 2), which is sensitive to the arbitrary choice of which asset is the dependent variable and has low power with short samples. The improved version additionally runs the **Johansen trace test**, which is symmetric, multivariate, and more powerful. Both tests must agree for a pair to be selected.

**No stop-loss**

If a spread diverges beyond 3sigma the original code held the position indefinitely, exposing the strategy to large losses from broken cointegration relationships. A stop-loss exits any position where `|z| > 3`.

**`n_jobs=1` -> `n_jobs=-1`**

The pair selection loop over all combinations is embarrassingly parallel. Changed to use all available CPU cores, substantially reducing pair selection time for larger universes.

**Transaction costs reduced from 20 bps to 10 bps**

20 bps per side is aggressive for daily crypto spot - most liquid exchanges charge 5-10 bps for maker/taker. The revised 10 bps is more realistic and avoids overstating the cost drag.

**Global backtest code -> `run_backtest()` function**

The original backtest was a block of global script-level code. Wrapping it in a parameterised function makes it testable, reusable, and easy to sweep over different configurations.

---

## Performance Notes

The strategy is benchmarked against an equal-weight buy-and-hold portfolio of all coins in the universe. Key characteristics of the stat arb approach relative to buy-and-hold:

- Substantially lower volatility and drawdown due to the long-short, market-neutral construction
- Performance depends heavily on the exit threshold - tighter thresholds turn over faster and capture smaller moves; wider thresholds hold longer and are more exposed to spread divergence
- Crypto pairs cointegration is unstable across market regimes (bull/bear cycles often break relationships), which is why 6-month pair re-selection is important

---

## Caveats

- **Data**: Data Required can be downloaden in a file 'CoinGecko_px_vol_1D.pkl.zip'
- **Shorting**: The strategy assumes the ability to short all coins at equal cost. In practice, short availability and borrow rates vary significantly.
- **Slippage**: Transaction costs are modelled as a flat rate on turnover. Actual slippage on less liquid pairs may be higher.
- **Survivorship bias**: The universe is filtered on data completeness, which may introduce mild survivorship bias for the earliest dates.
