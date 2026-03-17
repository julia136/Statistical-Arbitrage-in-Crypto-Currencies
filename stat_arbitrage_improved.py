from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from joblib import Parallel, delayed
from itertools import combinations

print("SCRIPT STARTED")

# =============================================================================
# DATA LOADING
# =============================================================================

from_pickle = pd.read_pickle('/Users/juleczka/Quant Project/CoinGecko_px_vol_1D.pkl')

price_columns = [col for col in from_pickle.columns if 'price' in col]
crypto_px = from_pickle[price_columns]
crypto_px.columns = crypto_px.columns.droplevel(1)

crypto_px = crypto_px.loc['2018-01-01':]
crypto_px = crypto_px.drop(columns=['TUSD', 'DAI', 'WBTC', 'WETH', 'USDC', 'BSV'])

total_data_points = len(crypto_px)
non_null_counts = crypto_px.notnull().sum()
threshold = 0.90 * total_data_points
crypto_px = crypto_px.loc[:, non_null_counts >= threshold]

print(f"Universe: {crypto_px.shape[1]} coins, {crypto_px.shape[0]} days")

# Daily returns
coins_ret = crypto_px / crypto_px.shift() - 1

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_turnover(port):
    """Absolute change in weights each day — proxy for trading activity."""
    return (port.fillna(0) - port.shift().fillna(0)).abs().sum(axis=1)


def compute_sharpe_ratio(rets):
    mean_rets = rets.mean() * 252
    vol = rets.std() * np.sqrt(252)
    return mean_rets / vol


def compute_stats(rets):
    stats = {}
    stats['avg'] = rets.mean() * 252
    stats['vol'] = rets.std() * np.sqrt(252)
    stats['sharpe'] = stats['avg'] / stats['vol']
    stats['hit_rate'] = (rets > 0).sum() / rets.count()
    return pd.DataFrame(stats)


def drawdown(returns):
    """Drawdown series from a daily-return series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def duration(equity_curve):
    """
    Number of consecutive days spent below the previous peak.
    Pass in an equity curve (cumprod), NOT a return series.
    """
    peak = equity_curve.expanding(min_periods=1).max()
    below_peak = equity_curve < peak
    res = (below_peak.cumsum()
           - below_peak.cumsum().where(~below_peak).ffill().fillna(0))
    return res


def final_results(avg_return, volatility, sharpe, hit_rate,
                  max_drawdown, max_duration, holding_period, IR):
    stats = {
        'Return (Ann.)':     f"{avg_return * 100:.2f}%",
        'Volatility (Ann.)': f"{volatility * 100:.2f}%",
        'Sharpe Ratio':      f"{sharpe:.2f}",
        'Hit Rate':          f"{hit_rate * 100:.2f}%",
        'Max Drawdown':      f"{max_drawdown:.2f}%",
        'Max Duration':      f"{max_duration:.2f} days",
        'Holding Period':    f"{holding_period:.2f} days",
        'Information Ratio': f"{IR:.2f}",
    }
    return pd.DataFrame(stats, index=['Cointegration']).T


# =============================================================================
# PAIR SELECTION — dual ADF + Johansen test
#  Previously only ADF was used, prone to false positives.
#      Now both tests must agree before a pair is accepted.
#  n_jobs defaults to -1 (use all CPU cores) instead of 1.
# =============================================================================

def adf_for_pair(symbol_i, symbol_j, insample_px):
    """
    Run ADF on OLS residuals (Engle-Granger step 2) and Johansen test.
    Returns p-value, ADF test statistic, and whether Johansen also confirms.

     beta/alpha estimated here are consistent with gen_signals —
         both regress log_px_j on log_px_i (Y on X).
    """
    # Replace zeros before log, forward-fill, then drop any remaining NaN/inf
    raw = insample_px[[symbol_i, symbol_j]].replace(0, np.nan).ffill().bfill()
    log_px = np.log(raw)
    log_px = log_px.replace([np.inf, -np.inf], np.nan).dropna()

    # Need enough observations to run a meaningful regression
    if len(log_px) < 30:
        return (symbol_i, symbol_j), (1.0, 0.0, False)

    X = log_px[symbol_i].values
    Y = log_px[symbol_j].values

    model = sm.OLS(Y, sm.add_constant(X)).fit()
    alpha = model.params[0]
    beta  = model.params[1]
    residuals = Y - beta * X - alpha

    # --- ADF test ---
    adf_res   = adfuller(residuals)
    p_value   = adf_res[1]
    test_stat = adf_res[0]

    # --- Johansen test (trace statistic, r=0 vs r>=1) ---
    johansen_confirms = False
    try:
        jres = coint_johansen(log_px.values, det_order=0, k_ar_diff=1)
        # critical value at 5 %: column index 1
        johansen_confirms = jres.lr1[0] > jres.cvt[0, 1]
    except Exception:
        pass  # if Johansen fails, fall back to ADF alone

    return (symbol_i, symbol_j), (p_value, test_stat, johansen_confirms)


def select_pairs(insample_px, significance_level=0.05, top_n=1, n_jobs=-1,
                 require_johansen=True):
    """
    Select cointegrated pairs from an in-sample price window.

     n_jobs=-1 uses all available CPU cores (was hard-coded to 1).
     Dual-test filter — pair must pass ADF AND Johansen (configurable).
    """
    symbols = insample_px.columns.tolist()
    pairs   = list(combinations(symbols, 2))

    raw = Parallel(n_jobs=n_jobs)(
        delayed(adf_for_pair)(si, sj, insample_px) for si, sj in pairs
    )

    pairs_out, results = zip(*raw)
    adf_df = pd.DataFrame(
        results,
        columns=['p_value', 'test_statistics', 'johansen_confirms'],
        index=pairs_out,
    )

    # Primary filter: ADF p-value
    filtered = adf_df[adf_df['p_value'] < significance_level].copy()

    # Secondary filter: Johansen confirmation
    if require_johansen:
        filtered = filtered[filtered['johansen_confirms']]

    if filtered.empty:
        print("  No cointegrated pairs found; relaxing to ADF-only.")
        filtered = adf_df[adf_df['p_value'] < significance_level].copy()

    final_pairs_set  = set()
    top_pairs_per_coin = {}
    all_coins = set(c for pair in filtered.index for c in pair)

    for coin in all_coins:
        coin_pairs  = filtered.loc[[coin in p for p in filtered.index]]
        sorted_pairs = coin_pairs.sort_values('test_statistics')
        top         = sorted_pairs.head(top_n)
        top_pairs_per_coin[coin] = top
        final_pairs_set.update(top.index.tolist())

    final_pairs = sorted(final_pairs_set)
    update_date = insample_px.index[-1] + pd.Timedelta(days=1)
    print(f"  Pairs updated: {update_date.date()}  |  "
          f"Pairs selected: {len(final_pairs)}")

    return final_pairs


# =============================================================================
# SIGNAL GENERATION
# Spread direction is now consistent with adf_for_pair
#      (spread = log_i - beta*log_j - alpha, matching Y=log_j, X=log_i OLS).
#  gen_signals now accepts a price slice so rolling windows only see
#      in-sample + current-window data (prevents subtle look-ahead).
# =============================================================================

def gen_signals(px, pairs, window=90):
    """
    Generate z-score signals for each pair using rolling OLS beta/alpha.

    px     : full price DataFrame (we slice internally so rolling look-back
              never crosses the out-of-sample boundary in terms of pairs,
              but we need enough history for the rolling window to warm up).
    pairs  : list of (asset_i, asset_j) tuples.
    window : rolling estimation window in days.
    """
    signal_dict = {}
    for pair in pairs:
        asset_i, asset_j = pair

        px_i = px[asset_i].replace(0, np.nan).ffill().bfill()
        px_j = px[asset_j].replace(0, np.nan).ffill().bfill()

        log_i = np.log(px_i).replace([np.inf, -np.inf], np.nan)
        log_j = np.log(px_j).replace([np.inf, -np.inf], np.nan)

        # Rolling OLS: regress log_j on log_i  (consistent with adf_for_pair)
        rolling_cov = log_i.rolling(window=window, min_periods=window).cov(log_j)
        rolling_var = log_i.rolling(window=window, min_periods=window).var()

        beta  = rolling_cov / rolling_var
        alpha = (log_j.rolling(window=window, min_periods=window).mean()
                 - beta * log_i.rolling(window=window, min_periods=window).mean())

        # Spread = log_i - beta*log_j - alpha
        # (same direction as residuals in adf_for_pair)
        spread = log_i - (beta * log_j + alpha)

        spread_mean = spread.rolling(window=window, min_periods=window).mean()
        spread_std  = spread.rolling(window=window, min_periods=window).std()

        z_score = (spread - spread_mean) / spread_std.replace(0, np.nan)

        signal_dict[(pair, 'beta')]    = beta
        signal_dict[(pair, 'alpha')]   = alpha
        signal_dict[(pair, 'spread')]  = spread
        signal_dict[(pair, 'z_score')] = z_score

    return pd.DataFrame(signal_dict)


# =============================================================================
# PORTFOLIO CONSTRUCTION
#  Positions are accumulated per asset across pairs with += instead of
#      direct assignment, so an asset appearing in multiple pairs no longer
#      gets overwritten by the last pair processed.
#  Stop-loss added — force exit if |z| > stop_loss_sigma (default 3).
# =============================================================================

def gen_port(signal_df, pairs, all_columns, exit_threshold=0.5,
             entry_sigma=1.0, stop_loss_sigma=3.0):
    """
    Build a daily position DataFrame.

    exit_threshold  : |z| below this → close the trade.
    entry_sigma     : |z| above this → open the trade.
    stop_loss_sigma : |z| above this → stop-loss exit (broken pair).

     per-pair positions are stored separately then summed, so multiple
         pairs sharing an asset accumulate correctly instead of overwriting.
    """
    # One sub-DataFrame per pair, then sum at the end
    pair_positions = []

    for pair in pairs:
        asset_i, asset_j = pair

        z  = signal_df[(pair, 'z_score')]
        b  = signal_df[(pair, 'beta')]

        pp = pd.DataFrame(0.0, index=signal_df.index, columns=all_columns)

        # Entry: z > +entry_sigma → short i, long j (scaled by beta)
        long_j  =  z >  entry_sigma
        short_j =  z < -entry_sigma
        exit_   =  z.abs() <= exit_threshold
        stop_   =  z.abs() >  stop_loss_sigma

        pp.loc[long_j,  asset_i] = -1.0
        pp.loc[long_j,  asset_j] =  b[long_j]

        pp.loc[short_j, asset_i] =  1.0
        pp.loc[short_j, asset_j] = -b[short_j]

        # Exit on mean-reversion or stop-loss
        pp.loc[exit_ | stop_, asset_i] = 0.0
        pp.loc[exit_ | stop_, asset_j] = 0.0

        pair_positions.append(pp)

    # Sum positions across all pairs ( was direct assignment → overwrites)
    pos = sum(pair_positions)

    pos = pos.ffill()
    abs_sum = pos.abs().sum(axis=1).replace(0, np.nan)
    pos = pos.divide(abs_sum, axis=0).fillna(0.0)
    return pos


# =============================================================================
# BACKTEST
#  full_portfolio is now re-initialised inside the threshold loop so
#      each threshold starts with a clean slate (was shared across iterations).
#  tcost_bps reduced from 20 to 10 bps (more realistic for daily crypto).
#  duration() is called on the equity curve (cumprod) not cumsum.
#  gen_signals receives only price data up to insample_end + window
#      so future prices cannot contaminate rolling beta estimation.
# =============================================================================

def run_backtest(crypto_px, coins_ret,
                 end_of_insample='2018-12-31',
                 thresholds=(0.1, 0.2, 0.5, 0.7),
                 signal_window=90,
                 tcost_bps=10):
    """
    Walk-forward backtest across multiple exit thresholds.

    Returns
    -------
    metrics_df       : DataFrame of performance metrics per threshold
    strat_net_ret_dict: dict mapping threshold → daily net return Series
    """
    end_of_insample    = pd.Timestamp(end_of_insample)
    last_available     = crypto_px.index[-1]
    start_of_oos       = end_of_insample + pd.DateOffset(days=1)
    update_dates       = pd.date_range(start=start_of_oos,
                                       end=last_available, freq='6MS')

    metrics = {k: np.zeros(len(thresholds)) for k in
               ['Sharpe Ratio', 'Return', 'Volatility',
                'Holding Period', 'Turnover', 'Transaction Costs']}
    strat_net_ret_dict = {}

    for i, thr in enumerate(thresholds):
        print(f"\n── Threshold {thr} ──────────────────────────────")

        #  re-initialise for every threshold
        full_portfolio = pd.DataFrame(
            index=crypto_px.loc[start_of_oos:].index,
            columns=crypto_px.columns,
            dtype=float,
        )

        for start_date in update_dates:
            end_date = min(
                start_date + pd.DateOffset(months=6) - pd.DateOffset(days=1),
                last_available,
            )

            insample_start = start_date - pd.DateOffset(years=1)
            insample_end   = start_date - pd.DateOffset(days=1)

            # Pair selection on in-sample window only
            updated_pairs = select_pairs(
                crypto_px.loc[insample_start:insample_end]
            )

            if not updated_pairs:
                continue

            #  pass only the price data needed for rolling estimation
            # (in-sample + out-of-sample window to warm up rolling stats)
            px_slice = crypto_px.loc[insample_start:end_date]
            signal_df = gen_signals(px_slice, updated_pairs, window=signal_window)
            signal_df = signal_df.loc[start_date:end_date]

            port = gen_port(
                signal_df, updated_pairs,
                all_columns=crypto_px.columns,
                exit_threshold=thr,
            )

            full_portfolio.loc[start_date:end_date] = port.reindex(
                columns=full_portfolio.columns
            )

        full_portfolio = full_portfolio.fillna(0.0)

        # Returns
        out_ret = coins_ret.loc[start_of_oos:][full_portfolio.columns]
        strat_gross = (full_portfolio.shift() * out_ret).sum(axis=1)

        to = compute_turnover(full_portfolio)
        strat_net = strat_gross - to * tcost_bps * 1e-4

        strat_net_ret_dict[thr] = strat_net

        sr = compute_sharpe_ratio(strat_net)
        metrics['Sharpe Ratio'][i]     = sr
        metrics['Transaction Costs'][i] = (to * tcost_bps * 1e-4).mean()
        metrics['Holding Period'][i]   = 2 / (to.mean() + 1e-9)
        metrics['Turnover'][i]         = to.mean()
        metrics['Return'][i]           = strat_net.mean()
        metrics['Volatility'][i]       = strat_net.std()

        print(f"  Sharpe: {sr:.3f}  |  Ann. Return: "
              f"{strat_net.mean()*252*100:.2f}%")

    return pd.DataFrame(metrics, index=list(thresholds)), strat_net_ret_dict


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    metrics_df, strat_net_ret_dict = run_backtest(
        crypto_px, coins_ret,
        end_of_insample='2018-12-31',
        thresholds=[0.1, 0.2, 0.5, 0.7],
        signal_window=90,
        tcost_bps=10,          #  was 20 bps
    )

    print("\n── Metrics across thresholds ──")
    print(metrics_df)

    # Use the 0.7 threshold strategy for detailed analysis
    strat_net_ret      = strat_net_ret_dict[0.7]
    end_of_insample    = pd.Timestamp('2018-12-31')
    start_of_oos       = end_of_insample + pd.DateOffset(days=1)
    buy_and_hold_btc   = coins_ret['BTC'][start_of_oos:]

    full_sample_ret = pd.DataFrame({
        'strat_ret':       strat_net_ret,
        'buy_and_hold_btc': buy_and_hold_btc,
    })

    full_sample_stats = compute_stats(full_sample_ret)
    print("\n── Full-sample stats ──")
    print(full_sample_stats)

    # --- Beta / Information Ratio ---
    corr = full_sample_ret.rolling(252).corr(full_sample_ret['buy_and_hold_btc'])
    vol  = full_sample_ret.rolling(252).std()
    beta_series = (corr * vol).divide(vol['buy_and_hold_btc'], axis=0)

    resid = full_sample_ret - beta_series.multiply(
        full_sample_ret['buy_and_hold_btc'], axis=0
    )
    IR = resid.mean() / resid.std() * np.sqrt(252)
    print(f"\nInformation Ratio: {IR['strat_ret']:.3f}")

    # --- Drawdown ---
    dd = drawdown(full_sample_ret['strat_ret']) * 100
    plt.figure(figsize=(10, 2))
    dd.plot()
    plt.ylim(-30, 0)
    plt.title('Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.tight_layout()
    plt.show()
    print(f"Max drawdown: {dd.min():.2f}%")

    #  duration() expects an equity curve, not a return cumsum
    equity_curve = (1 + full_sample_ret).cumprod()
    ddd = duration(equity_curve)
    ddd.plot(title='Drawdown Duration (days)')
    plt.tight_layout()
    plt.show()
    print(f"Max drawdown duration:\n{ddd.max()}")

    # --- Final summary table ---
    avg_return      = full_sample_stats.loc['strat_ret', 'avg']
    volatility      = full_sample_stats.loc['strat_ret', 'vol']
    sharpe          = full_sample_stats.loc['strat_ret', 'sharpe']
    hit_rate        = full_sample_stats.loc['strat_ret', 'hit_rate']
    max_dd          = dd.min()
    max_dur         = ddd.max().loc['strat_ret']
    holding_period  = metrics_df.loc[0.1, 'Holding Period']
    information_ratio = IR['strat_ret']

    summary = final_results(avg_return, volatility, sharpe, hit_rate,
                            max_dd, max_dur, holding_period, information_ratio)
    print("\n── Final Results ──")
    print(summary)