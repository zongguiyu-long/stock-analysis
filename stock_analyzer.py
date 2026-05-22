#!/usr/bin/env python3
"""
================================================================================
  Stock Short-Term Overbought/Oversold Analysis Tool  v1.0
  Multi-indicator weighted scoring system for short-term market timing
================================================================================

INSTALLATION (run once):
    pip install yfinance pandas numpy rich matplotlib tqdm

USAGE:
    python stock_analyzer.py                           # Interactive mode
    python stock_analyzer.py --symbols AAPL            # Single stock
    python stock_analyzer.py --symbols AAPL,MU,MSFT   # Batch comparison
    python stock_analyzer.py --symbols 0700.HK --chart # HK stock + chart
    python stock_analyzer.py --symbols 600519.SS       # A-share
    python stock_analyzer.py --test                    # Self-validation tests

SUPPORTED MARKETS:
    US  : AAPL, MSFT, MU, TSLA, SPY
    HK  : 0700.HK, 9988.HK
    A   : 600519.SS (Shanghai), 000858.SZ (Shenzhen)
    ETF : QQQ, IWM

⚠  DISCLAIMER: For technical analysis reference only. Not investment advice.
   Technical indicators may produce false signals. Invest at your own risk.
================================================================================
"""
from __future__ import annotations

import sys
import logging
import argparse
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ── Dependency check ──────────────────────────────────────────────────────────
_MISSING: List[str] = []
for _pkg in ("yfinance", "rich"):
    try:
        __import__(_pkg.replace("-", "_"))
    except ImportError:
        _MISSING.append(_pkg)

if _MISSING:
    print(f"\n⚠  Missing packages: {', '.join(_MISSING)}")
    print(f"   Install:  pip install {' '.join(_MISSING)}\n")
    sys.exit(1)

import yfinance as yf

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("StockAnalyzer")
console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_DATA_DAYS = 30
FETCH_EXTRA_DAYS = 120   # warmup buffer beyond display period

# Per-indicator weights (sum = 16.0 → normalised to ±20)
INDICATOR_WEIGHTS: Dict[str, float] = {
    "rsi_14":    2.0,
    "rsi_6":     1.5,
    "kdj":       2.0,
    "bbands":    2.0,
    "macd":      1.5,
    "cci":       1.5,
    "williams_r":1.0,
    "stoch_rsi": 1.0,
    "bias":      1.5,
    "obv":       1.0,
}
_WEIGHT_SUM = sum(INDICATOR_WEIGHTS.values())   # 16.0

# Normalised score thresholds (±20 scale)
_TH_STRONG_OB  =  10.0
_TH_OB         =   5.0
_TH_OS         =  -5.0
_TH_STRONG_OS  = -10.0


# ══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class IndicatorResult:
    """Single-indicator analysis result."""
    name:           str
    value:          str
    signal:         str    # e.g. "Overbought", "Neutral", "Oversold"
    raw_score:      float  # −2 … +2
    weighted_score: float  # raw × weight
    details:        str = ""


@dataclass
class ComprehensiveSignal:
    """Aggregated signal across all indicators."""
    symbol:            str
    timestamp:         datetime
    normalized_score:  float          # −20 … +20
    raw_weighted_score:float
    verdict:           str
    verdict_emoji:     str
    indicators:        List[IndicatorResult]
    bullish_count:     int
    bearish_count:     int
    neutral_count:     int
    summary_points:    List[str]
    trend_context:     str
    volatility_context:str


# ══════════════════════════════════════════════════════════════════════════════
#  DataFetcher
# ══════════════════════════════════════════════════════════════════════════════
class DataFetcher:
    """Fetches and validates OHLCV data from Yahoo Finance."""

    def __init__(self, period_days: int = 90) -> None:
        self.period_days = period_days

    def fetch_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Download OHLCV history for *symbol*.

        Returns DataFrame or None on failure.
        """
        try:
            total_days = self.period_days + FETCH_EXTRA_DAYS
            end   = datetime.now()
            start = end - timedelta(days=total_days)

            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            if df.empty:
                logger.warning("No data returned for %s", symbol)
                return None

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=["Close"])

            if not self.validate_data(df, symbol):
                return None
            return df

        except Exception as exc:
            logger.error("Fetch failed for %s: %s", symbol, exc)
            return None

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Return metadata dict (name, currency, sector, market cap)."""
        info: Dict[str, Any] = {
            "symbol":     symbol,
            "name":       symbol,
            "currency":   "USD",
            "sector":     "N/A",
            "market_cap": "N/A",
        }
        try:
            raw = yf.Ticker(symbol).info
            info["name"]     = raw.get("longName") or raw.get("shortName") or symbol
            info["currency"] = raw.get("currency", "USD")
            info["sector"]   = raw.get("sector", "N/A")
            mc = raw.get("marketCap")
            if mc:
                if mc >= 1e12:
                    info["market_cap"] = f"${mc/1e12:.2f}T"
                elif mc >= 1e9:
                    info["market_cap"] = f"${mc/1e9:.2f}B"
                else:
                    info["market_cap"] = f"${mc/1e6:.0f}M"
        except Exception:
            pass
        return info

    def validate_data(self, df: pd.DataFrame, symbol: str = "") -> bool:
        """Check minimum data length and quality."""
        if df is None or df.empty:
            console.print(f"[red]✗ No data for {symbol}[/red]")
            return False
        if len(df) < MIN_DATA_DAYS:
            console.print(
                f"[red]✗ Insufficient data for {symbol}: "
                f"need {MIN_DATA_DAYS} bars, got {len(df)}[/red]"
            )
            return False
        nan_pct = df["Close"].isna().mean()
        if nan_pct > 0.1:
            console.print(f"[yellow]⚠ {symbol}: {nan_pct:.0%} missing values[/yellow]")
            return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  IndicatorCalculator
# ══════════════════════════════════════════════════════════════════════════════
class IndicatorCalculator:
    """Computes all 10 technical indicators used by the analysis engine."""

    # ── Individual indicators ─────────────────────────────────────────────────

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """RSI using Wilder's EWM smoothing."""
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        return 100.0 - 100.0 / (1.0 + rs)

    def calculate_kdj(
        self,
        high:  pd.Series,
        low:   pd.Series,
        close: pd.Series,
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        KDJ stochastic indicator (Chinese variant).
        K, D initialized at 50; J = 3K − 2D.
        """
        low_n  = low.rolling(n).min()
        high_n = high.rolling(n).max()
        rsv    = (close - low_n) / (high_n - low_n + 1e-10) * 100.0

        k_vals = np.full(len(close), np.nan)
        d_vals = np.full(len(close), np.nan)
        kv, dv = 50.0, 50.0

        for i, rv in enumerate(rsv.values):
            if np.isnan(rv):
                continue
            kv = (m1 - 1) / m1 * kv + rv / m1
            dv = (m2 - 1) / m2 * dv + kv / m2
            k_vals[i] = kv
            d_vals[i] = dv

        k = pd.Series(k_vals, index=close.index)
        d = pd.Series(d_vals, index=close.index)
        j = 3.0 * k - 2.0 * d
        return k, d, j

    def calculate_bbands(
        self,
        close:   pd.Series,
        period:  int   = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands → (upper, middle, lower, %B)."""
        mid   = close.rolling(period).mean()
        sigma = close.rolling(period).std(ddof=0)
        upper = mid + std_dev * sigma
        lower = mid - std_dev * sigma
        pct_b = (close - lower) / (upper - lower + 1e-10)
        return upper, mid, lower, pct_b

    def calculate_macd(
        self,
        close:  pd.Series,
        fast:   int = 12,
        slow:   int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD → (macd_line, signal_line, histogram)."""
        ema_f    = close.ewm(span=fast,   adjust=False).mean()
        ema_s    = close.ewm(span=slow,   adjust=False).mean()
        macd     = ema_f - ema_s
        sig_line = macd.ewm(span=signal,  adjust=False).mean()
        return macd, sig_line, macd - sig_line

    def calculate_cci(
        self,
        high:   pd.Series,
        low:    pd.Series,
        close:  pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """Commodity Channel Index."""
        tp     = (high + low + close) / 3.0
        tp_ma  = tp.rolling(period).mean()
        mean_d = tp.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        return (tp - tp_ma) / (0.015 * mean_d + 1e-10)

    def calculate_williams_r(
        self,
        high:   pd.Series,
        low:    pd.Series,
        close:  pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Williams %R (range: −100 … 0)."""
        h = high.rolling(period).max()
        l = low.rolling(period).min()
        return -100.0 * (h - close) / (h - l + 1e-10)

    def calculate_stoch_rsi(
        self,
        close:        pd.Series,
        rsi_period:   int = 14,
        stoch_period: int = 14,
        smooth_k:     int = 3,
        smooth_d:     int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic RSI → (K, D) in [0, 100]."""
        rsi      = self.calculate_rsi(close, rsi_period)
        rsi_min  = rsi.rolling(stoch_period).min()
        rsi_max  = rsi.rolling(stoch_period).max()
        stoch    = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100.0
        k        = stoch.rolling(smooth_k).mean()
        d        = k.rolling(smooth_d).mean()
        return k, d

    def calculate_bias(
        self,
        close:   pd.Series,
        periods: List[int] = None,
    ) -> Dict[str, pd.Series]:
        """Percent deviation from rolling mean (乖离率)."""
        if periods is None:
            periods = [20, 60]
        result: Dict[str, pd.Series] = {}
        for p in periods:
            ma = close.rolling(p).mean()
            result[f"bias_{p}"] = (close - ma) / (ma + 1e-10) * 100.0
        return result

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    def calculate_atr(
        self,
        high:   pd.Series,
        low:    pd.Series,
        close:  pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range (Wilder EWM)."""
        prev  = close.shift(1)
        tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, min_periods=period).mean()

    # ── Divergence helpers ────────────────────────────────────────────────────

    def detect_divergence(
        self,
        price:    pd.Series,
        osc:      pd.Series,
        lookback: int = 20,
    ) -> str:
        """
        Detect bearish or bullish divergence between price and an oscillator.

        Returns: 'bearish_divergence' | 'bullish_divergence' | 'none'
        """
        if len(price) < lookback + 5:
            return "none"
        p = price.dropna().tail(lookback)
        o = osc.dropna().tail(lookback)
        if len(p) < lookback or len(o) < lookback:
            return "none"

        # Bearish: price making new high, oscillator failing to match
        price_new_high = p.iloc[-1] >= p.iloc[:-3].max() * 0.98
        osc_lower      = o.iloc[-1]  <  o.iloc[:-3].max() * 0.95
        if price_new_high and osc_lower:
            return "bearish_divergence"

        # Bullish: price making new low, oscillator holding higher
        price_new_low  = p.iloc[-1] <= p.iloc[:-3].min() * 1.02
        osc_higher     = o.iloc[-1]  >  o.iloc[:-3].min() * 1.05
        if price_new_low and osc_higher:
            return "bullish_divergence"

        return "none"

    def _obv_divergence(self, close: pd.Series, obv: pd.Series, lookback: int = 20) -> str:
        if len(close) < lookback:
            return "none"
        pc = (close.tail(lookback).iloc[-1] - close.tail(lookback).iloc[0]) / (abs(close.tail(lookback).iloc[0]) + 1e-10)
        po = (obv.tail(lookback).iloc[-1]   - obv.tail(lookback).iloc[0])   / (abs(obv.tail(lookback).iloc[0])   + 1e-10)
        if pc > 0.03 and po < -0.03:
            return "bearish_divergence"
        if pc < -0.03 and po > 0.03:
            return "bullish_divergence"
        return "none"

    # ── Main entry point ──────────────────────────────────────────────────────

    def calculate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute all indicators from a DataFrame and return a flat dict of
        latest scalar values plus a '_series' sub-dict for charting.
        """
        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        rsi14 = self.calculate_rsi(close, 14)
        rsi6  = self.calculate_rsi(close, 6)
        rsi_div = self.detect_divergence(close, rsi14)

        k, d, j = self.calculate_kdj(high, low, close)

        bb_up, bb_mid, bb_lo, pct_b = self.calculate_bbands(close)

        macd_line, macd_sig, macd_hist = self.calculate_macd(close)
        macd_cross = "none"
        if len(macd_hist) >= 3:
            h = macd_hist.dropna()
            if len(h) >= 2:
                if h.iloc[-2] < 0 < h.iloc[-1]:
                    macd_cross = "golden_cross"
                elif h.iloc[-2] > 0 > h.iloc[-1]:
                    macd_cross = "death_cross"

        cci = self.calculate_cci(high, low, close)
        wr  = self.calculate_williams_r(high, low, close)

        srsi_k, srsi_d = self.calculate_stoch_rsi(close)

        bias_map = self.calculate_bias(close, [20, 60])

        obv    = self.calculate_obv(close, volume)
        obv_ma = obv.rolling(20).mean()

        vol_ma5   = volume.rolling(5).mean()
        vol_ratio = float(volume.iloc[-1] / (vol_ma5.iloc[-1] + 1e-10))

        atr       = self.calculate_atr(high, low, close)
        atr_ratio = float(atr.iloc[-1] / (close.iloc[-1] + 1e-10) * 100.0)

        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        def _last(s: pd.Series) -> float:
            v = s.dropna()
            return float(v.iloc[-1]) if len(v) else float("nan")

        return {
            "rsi_14":      _last(rsi14),
            "rsi_6":       _last(rsi6),
            "rsi_div":     rsi_div,
            "kdj_k":       _last(k),
            "kdj_d":       _last(d),
            "kdj_j":       _last(j),
            "bb_upper":    _last(bb_up),
            "bb_mid":      _last(bb_mid),
            "bb_lower":    _last(bb_lo),
            "bb_pct_b":    _last(pct_b),
            "macd_line":   _last(macd_line),
            "macd_signal": _last(macd_sig),
            "macd_hist":   _last(macd_hist),
            "macd_cross":  macd_cross,
            "cci":         _last(cci),
            "williams_r":  _last(wr),
            "srsi_k":      _last(srsi_k),
            "srsi_d":      _last(srsi_d),
            "bias_20":     _last(bias_map["bias_20"]),
            "bias_60":     _last(bias_map["bias_60"]),
            "obv_trend":   "up" if _last(obv) > _last(obv_ma) else "down",
            "obv_div":     self._obv_divergence(close, obv),
            "vol_ratio":   vol_ratio,
            "atr":         _last(atr),
            "atr_ratio":   atr_ratio,
            "close":       float(close.iloc[-1]),
            "prev_close":  float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1]),
            "ma50":        _last(ma50)  if not np.isnan(_last(ma50))  else None,
            "ma200":       _last(ma200) if not np.isnan(_last(ma200)) else None,
            "_series": {
                "close":       close,
                "high":        high,
                "low":         low,
                "open":        df["Open"],
                "volume":      volume,
                "rsi_14":      rsi14,
                "rsi_6":       rsi6,
                "kdj_k":       k,
                "kdj_d":       d,
                "kdj_j":       j,
                "bb_upper":    bb_up,
                "bb_mid":      bb_mid,
                "bb_lower":    bb_lo,
                "macd_line":   macd_line,
                "macd_signal": macd_sig,
                "macd_hist":   macd_hist,
                "cci":         cci,
                "williams_r":  wr,
                "obv":         obv,
                "ma50":        ma50,
                "ma200":       ma200,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SignalAnalyzer
# ══════════════════════════════════════════════════════════════════════════════
class SignalAnalyzer:
    """Converts raw indicator values into IndicatorResult objects and a final ComprehensiveSignal."""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _signal_label(score: float) -> str:
        if score >= 1.8:  return "Extreme Overbought"
        if score >= 0.9:  return "Overbought"
        if score <= -1.8: return "Extreme Oversold"
        if score <= -0.9: return "Oversold"
        return "Neutral"

    @staticmethod
    def _clamp(v: float, lo: float = -2.0, hi: float = 2.0) -> float:
        return max(lo, min(hi, v))

    # ── Per-indicator analyzers ───────────────────────────────────────────────

    def analyze_rsi14(self, rsi: float, divergence: str = "none") -> IndicatorResult:
        if   rsi >= 80: score = 2.0
        elif rsi >= 70: score = 1.0
        elif rsi >= 60: score = 0.4
        elif rsi <= 20: score = -2.0
        elif rsi <= 30: score = -1.0
        elif rsi <= 40: score = -0.4
        else:           score = (rsi - 50.0) / 20.0 * 0.4

        detail = ""
        if divergence == "bearish_divergence":
            score   = self._clamp(score + 0.5)
            detail  = "⚠ 顶背离"
        elif divergence == "bullish_divergence":
            score   = self._clamp(score - 0.5)
            detail  = "⚠ 底背离"

        return IndicatorResult(
            "RSI(14)", f"{rsi:.1f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["rsi_14"], detail,
        )

    def analyze_rsi6(self, rsi: float) -> IndicatorResult:
        if   rsi >= 85: score = 2.0
        elif rsi >= 75: score = 1.0
        elif rsi >= 65: score = 0.4
        elif rsi <= 15: score = -2.0
        elif rsi <= 25: score = -1.0
        elif rsi <= 35: score = -0.4
        else:           score = (rsi - 50.0) / 25.0 * 0.4

        return IndicatorResult(
            "RSI(6)", f"{rsi:.1f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["rsi_6"],
        )

    def analyze_kdj(self, kv: float, dv: float, jv: float) -> IndicatorResult:
        kd = (kv + dv) / 2.0
        if   kd >= 85 or jv > 100: score = 2.0
        elif kd >= 75:             score = 1.0
        elif kd >= 60:             score = 0.4
        elif kd <= 15 or jv < 0:  score = -2.0
        elif kd <= 25:             score = -1.0
        elif kd <= 40:             score = -0.4
        else:                      score = (kd - 50.0) / 25.0 * 0.4

        # J extremes amplify
        if jv > 110:  score = self._clamp(score + 0.3)
        elif jv < -10: score = self._clamp(score - 0.3)

        detail = f"K={kv:.1f} D={dv:.1f} J={jv:.1f}"
        if jv > 100: detail += "  [J极端超买]"
        elif jv < 0: detail += "  [J极端超卖]"

        return IndicatorResult(
            "KDJ", f"K:{kv:.1f}/D:{dv:.1f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["kdj"], detail,
        )

    def analyze_bbands(
        self, close: float, upper: float, lower: float, pct_b: float
    ) -> IndicatorResult:
        if   pct_b >= 1.1:  score = 2.0
        elif pct_b >= 0.95: score = 1.5
        elif pct_b >= 0.80: score = 1.0
        elif pct_b >= 0.65: score = 0.4
        elif pct_b <= -0.1: score = -2.0
        elif pct_b <=  0.05: score = -1.5
        elif pct_b <=  0.20: score = -1.0
        elif pct_b <=  0.35: score = -0.4
        else:               score = (pct_b - 0.5) / 0.15 * 0.4

        score  = self._clamp(score)
        detail = f"%B={pct_b:.3f}"
        if   close > upper: detail += "  [突破上轨]"
        elif close < lower: detail += "  [跌破下轨]"

        return IndicatorResult(
            "BBands %B", f"{pct_b:.3f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["bbands"], detail,
        )

    def analyze_macd(
        self,
        macd:       float,
        sig:        float,
        hist:       float,
        hist_series: Optional[pd.Series] = None,
        cross:      str = "none",
    ) -> IndicatorResult:
        # Histogram momentum (acceleration)
        momentum = 0.0
        if hist_series is not None:
            recent = hist_series.dropna().tail(5)
            if len(recent) >= 3:
                momentum = float(recent.diff().dropna().mean())

        if hist > 0:
            score = 1.5 if momentum > 0 else 0.7
        elif hist < 0:
            score = -1.5 if momentum < 0 else -0.7
        else:
            score = 0.0

        detail = ""
        if   cross == "golden_cross": score = self._clamp(score + 0.5); detail = "金叉"
        elif cross == "death_cross":  score = self._clamp(score - 0.5); detail = "死叉"

        return IndicatorResult(
            "MACD Hist", f"{hist:+.4f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["macd"], detail,
        )

    def analyze_cci(self, cci: float) -> IndicatorResult:
        if   cci >= 200:  score = 2.0
        elif cci >= 100:  score = 1.0
        elif cci >=  50:  score = 0.4
        elif cci <= -200: score = -2.0
        elif cci <= -100: score = -1.0
        elif cci <=  -50: score = -0.4
        else:             score = cci / 100.0 * 0.4

        return IndicatorResult(
            "CCI(20)", f"{cci:+.1f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["cci"],
        )

    def analyze_williams_r(self, wr: float) -> IndicatorResult:
        # WR range: −100 (oversold) … 0 (overbought)
        if   wr >= -10:  score = 2.0
        elif wr >= -20:  score = 1.0
        elif wr >= -35:  score = 0.4
        elif wr <= -90:  score = -2.0
        elif wr <= -80:  score = -1.0
        elif wr <= -65:  score = -0.4
        else:            score = (-wr - 50.0) / 30.0 * -0.4

        return IndicatorResult(
            "Williams %R", f"{wr:.1f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["williams_r"],
        )

    def analyze_stoch_rsi(self, sk: float, sd: float) -> IndicatorResult:
        avg = (sk + sd) / 2.0
        if   avg >= 90: score = 2.0
        elif avg >= 80: score = 1.0
        elif avg >= 65: score = 0.4
        elif avg <= 10: score = -2.0
        elif avg <= 20: score = -1.0
        elif avg <= 35: score = -0.4
        else:           score = (avg - 50.0) / 30.0 * 0.4

        return IndicatorResult(
            "Stoch RSI", f"K:{sk:.1f}/D:{sd:.1f}",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["stoch_rsi"],
        )

    def analyze_bias(
        self,
        bias20: float,
        bias60: float,
        atr_ratio: float = 2.0,
    ) -> IndicatorResult:
        # Dynamic threshold: wider for volatile stocks
        thresh = max(7.0, min(15.0, atr_ratio * 4.5))

        if   bias20 >=  thresh * 1.5: score = 2.0
        elif bias20 >=  thresh:       score = 1.0
        elif bias20 >=  thresh * 0.5: score = 0.4
        elif bias20 <= -thresh * 1.5: score = -2.0
        elif bias20 <= -thresh:       score = -1.0
        elif bias20 <= -thresh * 0.5: score = -0.4
        else:                         score = bias20 / thresh * 0.4

        # Bias_60 amplifies if same direction
        if bias60 * bias20 > 0 and abs(bias60) > thresh * 0.6:
            score = self._clamp(score * 1.15)

        score  = self._clamp(score)
        detail = f"MA20:{bias20:+.1f}%  MA60:{bias60:+.1f}%  thresh:{thresh:.1f}%"

        return IndicatorResult(
            "Bias(MA20/60)", f"{bias20:+.1f}%/{bias60:+.1f}%",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["bias"], detail,
        )

    def analyze_obv(
        self,
        obv_trend:  str,
        divergence: str,
        vol_ratio:  float,
    ) -> IndicatorResult:
        score = 0.0
        parts: List[str] = []

        if   divergence == "bearish_divergence": score =  1.0; parts.append("价格/OBV顶背离")
        elif divergence == "bullish_divergence": score = -1.0; parts.append("价格/OBV底背离")

        if vol_ratio >= 2.5:  parts.append(f"极端放量 {vol_ratio:.1f}×")
        elif vol_ratio >= 2.0: parts.append(f"异常放量 {vol_ratio:.1f}×")

        trend_arrow = "↑" if obv_trend == "up" else "↓"
        return IndicatorResult(
            "OBV/Volume",
            f"OBV{trend_arrow}  Vol:{vol_ratio:.1f}×",
            self._signal_label(score), score,
            score * INDICATOR_WEIGHTS["obv"],
            " | ".join(parts),
        )

    # ── Trend & volatility context ────────────────────────────────────────────

    @staticmethod
    def get_trend_context(
        close: float,
        ma50:  Optional[float],
        ma200: Optional[float],
    ) -> str:
        if ma200 is None and ma50 is None:
            return "数据不足，无法判断趋势"
        if ma200 is None:
            return "上升趋势 (价格 > MA50)" if close > ma50 else "下降趋势 (价格 < MA50)"
        above50  = ma50  is None or close > ma50
        above200 = close > ma200
        if above200 and above50:
            return "长期上升趋势 (>MA50 & >MA200) — 超买信号反转风险较高"
        if above200 and not above50:
            return "中期整理 (>MA200 但 <MA50) — 趋势混沌"
        if not above200 and not above50:
            return "长期下降趋势 (<MA50 & <MA200) — 超卖反弹可能是短暂的"
        return "弱势 (<MA200) — 谨慎对待反弹信号"

    # ── Verdict ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_verdict(score: float) -> Tuple[str, str]:
        if   score >  _TH_STRONG_OB: return "STRONG OVERBOUGHT",  "🔴"
        elif score >  _TH_OB:        return "OVERBOUGHT",          "🟠"
        elif score >= _TH_OS:        return "NEUTRAL",             "🟢"
        elif score >= _TH_STRONG_OS: return "OVERSOLD",            "🟡"
        else:                        return "STRONG OVERSOLD",     "🔵"

    # ── Summary bullets ───────────────────────────────────────────────────────

    def _build_summary(
        self,
        ind:      Dict[str, Any],
        results:  List[IndicatorResult],
        verdict:  str,
    ) -> List[str]:
        pts: List[str] = []
        ob_cnt = sum(1 for r in results if r.raw_score > 0.5)
        os_cnt = sum(1 for r in results if r.raw_score < -0.5)
        n      = len(results)

        if "OVERBOUGHT" in verdict:
            pts.append(f"{ob_cnt}/{n} 项指标指向超买，多指标共振确认短期超买状态")
        elif "OVERSOLD" in verdict:
            pts.append(f"{os_cnt}/{n} 项指标指向超卖，多指标共振确认短期超卖状态")
        else:
            pts.append(f"信号分散：{ob_cnt} 偏多 / {n-ob_cnt-os_cnt} 中性 / {os_cnt} 偏空，市场处于整理状态")

        rsi = ind.get("rsi_14", 50.0)
        if rsi > 70:
            pts.append(f"RSI(14)={rsi:.1f}，进入超买区间，历史上易触发回调")
        elif rsi < 30:
            pts.append(f"RSI(14)={rsi:.1f}，进入超卖区间，可关注反弹机会")

        div = ind.get("rsi_div", "none")
        if   div == "bearish_divergence": pts.append("RSI 与价格出现顶背离，需警惕趋势反转")
        elif div == "bullish_divergence": pts.append("RSI 与价格出现底背离，可能存在反弹动力")

        b20 = ind.get("bias_20", 0.0)
        if abs(b20) > 8:
            word = "高于" if b20 > 0 else "低于"
            pts.append(
                f"价格偏离 MA20 达 {b20:+.1f}%，大幅{word}均线"
                f"{'，均值回归压力大' if b20 > 0 else '，超跌反弹概率上升'}"
            )

        jv = ind.get("kdj_j", 50.0)
        if   jv > 100: pts.append(f"KDJ-J={jv:.1f}，超过100，进入极端超买区域")
        elif jv <   0: pts.append(f"KDJ-J={jv:.1f}，低于0，进入极端超卖区域")

        vr = ind.get("vol_ratio", 1.0)
        if vr >= 2.0:
            pts.append(f"成交量为5日均量的 {vr:.1f} 倍，异常放量，需注意资金动向")

        od = ind.get("obv_div", "none")
        if   od == "bearish_divergence": pts.append("OBV 与价格出现顶背离，量价关系异常")
        elif od == "bullish_divergence": pts.append("OBV 与价格出现底背离，量能支撑反弹")

        cross = ind.get("macd_cross", "none")
        if   cross == "golden_cross": pts.append("MACD 金叉刚出现，短期动能转强")
        elif cross == "death_cross":  pts.append("MACD 死叉刚出现，短期动能转弱")

        # Action suggestion
        if   "STRONG OVERBOUGHT" in verdict: pts.append("⚠ 建议：避免追高，可考虑分批减仓，设置止盈位")
        elif "OVERBOUGHT"        in verdict: pts.append("⚠ 建议：注意控仓，留意回调风险，止盈保护")
        elif "STRONG OVERSOLD"   in verdict: pts.append("💡 建议：可关注超跌反弹，需结合基本面与量能确认")
        elif "OVERSOLD"          in verdict: pts.append("💡 建议：关注技术性反弹，轻仓试探，严设止损")
        else:                                pts.append("建议：耐心等待方向性信号明确，避免追涨杀跌")

        return pts

    # ── Main analysis method ──────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        ind:    Dict[str, Any],
    ) -> ComprehensiveSignal:
        """
        Run full multi-indicator analysis and return a ComprehensiveSignal.
        Gracefully skips indicators whose values are NaN.
        """
        hist_series = ind["_series"].get("macd_hist")

        candidates = [
            self.analyze_rsi14(ind["rsi_14"], ind.get("rsi_div", "none")),
            self.analyze_rsi6(ind["rsi_6"]),
            self.analyze_kdj(ind["kdj_k"], ind["kdj_d"], ind["kdj_j"]),
            self.analyze_bbands(ind["close"], ind["bb_upper"], ind["bb_lower"], ind["bb_pct_b"]),
            self.analyze_macd(
                ind["macd_line"], ind["macd_signal"], ind["macd_hist"],
                hist_series, ind.get("macd_cross", "none"),
            ),
            self.analyze_cci(ind["cci"]),
            self.analyze_williams_r(ind["williams_r"]),
            self.analyze_stoch_rsi(ind["srsi_k"], ind["srsi_d"]),
            self.analyze_bias(ind["bias_20"], ind["bias_60"], ind.get("atr_ratio", 2.0)),
            self.analyze_obv(ind["obv_trend"], ind.get("obv_div", "none"), ind.get("vol_ratio", 1.0)),
        ]

        # Drop any NaN results
        results = [r for r in candidates if not (np.isnan(r.raw_score) or np.isnan(r.weighted_score))]

        total_wt = sum(r.weighted_score for r in results)
        max_wt   = sum(INDICATOR_WEIGHTS[k] * 2.0 for k in INDICATOR_WEIGHTS)
        norm     = (total_wt / max_wt) * 20.0

        verdict, emoji = self.get_verdict(norm)

        bullish = sum(1 for r in results if r.raw_score >  0.3)
        bearish = sum(1 for r in results if r.raw_score < -0.3)
        neutral = len(results) - bullish - bearish

        trend_ctx  = self.get_trend_context(ind["close"], ind.get("ma50"), ind.get("ma200"))
        volat_ctx  = f"ATR(14)={ind['atr']:.2f}  ({ind['atr_ratio']:.1f}% of price)"
        summary    = self._build_summary(ind, results, verdict)

        return ComprehensiveSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            normalized_score=round(norm, 2),
            raw_weighted_score=round(total_wt, 3),
            verdict=verdict,
            verdict_emoji=emoji,
            indicators=results,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            summary_points=summary,
            trend_context=trend_ctx,
            volatility_context=volat_ctx,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Display
# ══════════════════════════════════════════════════════════════════════════════
class Display:
    """Renders analysis results to terminal (rich) and optional chart (matplotlib)."""

    def __init__(self) -> None:
        self.console = Console()

    # ── Color helpers ─────────────────────────────────────────────────────────

    def _sig_color(self, signal: str) -> str:
        return {
            "Extreme Overbought": "bright_red",
            "Overbought":         "red",
            "Neutral":            "green",
            "Oversold":           "yellow",
            "Extreme Oversold":   "bright_blue",
        }.get(signal, "white")

    def _score_color(self, s: float) -> str:
        if   s >=  1.5: return "bright_red"
        elif s >=  0.5: return "red"
        elif s <= -1.5: return "bright_blue"
        elif s <= -0.5: return "yellow"
        return "green"

    def _verdict_color(self, v: str) -> str:
        if   "STRONG OVERBOUGHT" in v: return "bright_red"
        elif "OVERBOUGHT"        in v: return "red"
        elif "STRONG OVERSOLD"   in v: return "bright_blue"
        elif "OVERSOLD"          in v: return "yellow"
        return "bright_green"

    # ── Main print method ─────────────────────────────────────────────────────

    def print_analysis(
        self,
        info:   Dict[str, Any],
        signal: ComprehensiveSignal,
        df:     pd.DataFrame,
    ) -> None:
        close     = float(df["Close"].iloc[-1])
        prev      = float(df["Close"].iloc[-2]) if len(df) >= 2 else close
        chg       = close - prev
        chg_pct   = chg / (prev + 1e-10) * 100.0
        currency  = info.get("currency", "USD")
        chg_color = "green" if chg >= 0 else "red"

        # ── Header ────────────────────────────────────────────────────────────
        self.console.print(Panel(
            f"[bold cyan]📊 Stock Overbought/Oversold Analysis[/bold cyan]\n"
            f"[white]Symbol  : [bold]{signal.symbol}[/bold]  —  {info.get('name', signal.symbol)}[/white]\n"
            f"[white]Date    : {signal.timestamp.strftime('%Y-%m-%d %H:%M')}[/white]\n"
            f"[white]Close   : [bold]{currency} {close:.2f}[/bold]  "
            f"[{chg_color}]({'+'if chg>=0 else ''}{chg:.2f} / {'+'if chg_pct>=0 else ''}{chg_pct:.2f}%)[/{chg_color}][/white]\n"
            f"[white]Sector  : {info.get('sector','N/A')}   MCap: {info.get('market_cap','N/A')}[/white]",
            border_style="cyan", padding=(0, 2),
        ))

        # ── Indicators table ──────────────────────────────────────────────────
        tbl = Table(
            title="[bold]Technical Indicators[/bold]",
            box=box.ROUNDED,
            header_style="bold magenta",
            border_style="bright_blue",
            padding=(0, 1),
        )
        tbl.add_column("Indicator",  style="bold white",  min_width=15)
        tbl.add_column("Value",      justify="right",     min_width=14)
        tbl.add_column("Signal",                          min_width=18)
        tbl.add_column("Raw",        justify="center",    min_width=7)
        tbl.add_column("Weighted",   justify="center",    min_width=9)
        tbl.add_column("Details",                         min_width=26)

        for r in signal.indicators:
            sc = self._score_color(r.raw_score)
            tbl.add_row(
                r.name,
                r.value,
                f"[{self._sig_color(r.signal)}]{r.signal}[/]",
                f"[{sc}]{r.raw_score:+.1f}[/]",
                f"[{sc}]{r.weighted_score:+.2f}[/]",
                f"[dim]{r.details}[/dim]" if r.details else "",
            )
        self.console.print(tbl)

        # ── Score summary ─────────────────────────────────────────────────────
        max_wt    = sum(INDICATOR_WEIGHTS[k] * 2.0 for k in INDICATOR_WEIGHTS)
        vc        = self._verdict_color(signal.verdict)
        direction = "Bullish" if signal.normalized_score > 0 else "Bearish"

        self.console.print(Panel(
            f"[white]Weighted Score  : [bold]{signal.raw_weighted_score:+.3f}[/bold]  / ±{max_wt:.1f}[/white]\n"
            f"[white]Normalised Score: [bold]{signal.normalized_score:+.2f}[/bold]  / ±20.0"
            f"  ({direction} {abs(signal.normalized_score/20*100):.0f}%)[/white]\n"
            f"[white]Signal Alignment: "
            f"[red]{signal.bullish_count} OB[/red]  |  "
            f"[green]{signal.neutral_count} Neutral[/green]  |  "
            f"[bright_blue]{signal.bearish_count} OS[/bright_blue]"
            f"  (out of {len(signal.indicators)} indicators)[/white]\n\n"
            f"[bold {vc}]Final Verdict: {signal.verdict_emoji}  {signal.verdict}[/bold {vc}]",
            title="[bold]Comprehensive Score[/bold]",
            border_style="cyan",
        ))

        # ── Summary bullets ───────────────────────────────────────────────────
        bullets = "\n".join(f"  • {p}" for p in signal.summary_points)
        self.console.print(Panel(
            f"[white]{bullets}[/white]\n\n"
            f"[dim]📈 Trend    : {signal.trend_context}[/dim]\n"
            f"[dim]📊 Volatility: {signal.volatility_context}[/dim]",
            title="[bold]💡 Analysis Summary[/bold]",
            border_style="yellow",
        ))

        # ── Disclaimer ────────────────────────────────────────────────────────
        self.console.print(
            "[dim yellow]\n⚠  本工具仅供技术分析参考，不构成投资建议。"
            "技术指标可能产生假信号，投资有风险，决策需谨慎。[/dim yellow]\n"
        )

    # ── Batch comparison table ────────────────────────────────────────────────

    def print_batch_table(
        self,
        rows: List[Tuple[Dict[str, Any], ComprehensiveSignal]],
    ) -> None:
        self.console.print(Rule("[bold cyan]Multi-Stock Comparison[/bold cyan]"))

        # Sort by score descending (most overbought → most oversold)
        rows = sorted(rows, key=lambda x: x[1].normalized_score, reverse=True)

        tbl = Table(
            title="[bold]Overbought / Oversold Rankings[/bold]",
            box=box.DOUBLE_EDGE,
            header_style="bold cyan",
        )
        tbl.add_column("Symbol",    style="bold white")
        tbl.add_column("Name",      max_width=22)
        tbl.add_column("Close",     justify="right")
        tbl.add_column("Chg%",      justify="right")
        tbl.add_column("Score",     justify="center")
        tbl.add_column("Verdict")
        tbl.add_column("OB|N|OS",   justify="center")

        for info, sig in rows:
            vc = self._verdict_color(sig.verdict)
            cc = self._score_color(sig.normalized_score / 10.0)
            chg_pct = info.get("change_pct", 0.0)
            tbl.add_row(
                sig.symbol,
                info.get("name", sig.symbol)[:22],
                str(info.get("close", "N/A")),
                f"[{'green' if chg_pct >= 0 else 'red'}]{chg_pct:+.2f}%[/]",
                f"[{cc}]{sig.normalized_score:+.1f}[/]",
                f"[{vc}]{sig.verdict_emoji} {sig.verdict}[/{vc}]",
                f"[red]{sig.bullish_count}[/] | [green]{sig.neutral_count}[/] | [bright_blue]{sig.bearish_count}[/]",
            )

        self.console.print(tbl)
        self.console.print(
            "[dim yellow]⚠  本工具仅供技术分析参考，不构成投资建议。[/dim yellow]\n"
        )

    # ── Chart ─────────────────────────────────────────────────────────────────

    def plot_chart(
        self,
        symbol:    str,
        df:        pd.DataFrame,
        ind:       Dict[str, Any],
        signal:    ComprehensiveSignal,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a 5-panel dark-theme technical chart and save as PNG."""
        if not _MATPLOTLIB:
            self.console.print("[yellow]⚠ matplotlib not installed — chart skipped[/yellow]")
            return None

        N = 60
        disp = df.tail(N).copy()
        s    = ind["_series"]
        dates = list(range(N))   # integer x-axis

        def _tail(key: str) -> np.ndarray:
            return s[key].tail(N).values

        bg   = "#0d0d1f"
        axbg = "#111122"
        UP   = "#00e676"
        DN   = "#ff1744"
        GRID = "#1e1e3a"
        TXT  = "#cfd8dc"

        plt.rcParams.update({
            "figure.facecolor":  bg,
            "axes.facecolor":    axbg,
            "axes.edgecolor":    "#2a2a4a",
            "text.color":        TXT,
            "xtick.color":       TXT,
            "ytick.color":       TXT,
            "grid.color":        GRID,
        })

        fig = plt.figure(figsize=(17, 14), facecolor=bg)
        gs  = gridspec.GridSpec(
            5, 1, figure=fig,
            height_ratios=[3.2, 1.1, 1.1, 1.1, 1.1],
            hspace=0.04,
        )

        # ── Panel 1: Candlestick + BBands + MA50 ──────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(axbg)

        o = disp["Open"].values
        h = disp["High"].values
        l = disp["Low"].values
        c = disp["Close"].values

        for i in range(N):
            color = UP if c[i] >= o[i] else DN
            ax1.plot([i, i], [l[i], h[i]], color=color, lw=0.8, alpha=0.9)
            rect_h = max(abs(c[i] - o[i]), c[i] * 0.002)
            ax1.add_patch(plt.Rectangle(
                (i - 0.35, min(o[i], c[i])), 0.7, rect_h,
                color=color, alpha=0.9, zorder=3,
            ))

        bb_u = _tail("bb_upper")
        bb_m = _tail("bb_mid")
        bb_l = _tail("bb_lower")
        ax1.plot(dates, bb_u, color="#ff9800", lw=1.0, ls="--", alpha=0.7, label="BB Upper")
        ax1.plot(dates, bb_m, color="#29b6f6", lw=1.0, alpha=0.8,           label="MA20")
        ax1.plot(dates, bb_l, color="#ff9800", lw=1.0, ls="--", alpha=0.7, label="BB Lower")
        ax1.fill_between(dates, bb_u, bb_l, alpha=0.04, color="#ff9800")

        ma50v = _tail("ma50")
        ax1.plot(dates, ma50v, color="#e040fb", lw=1.2, alpha=0.8, label="MA50")

        vc_hex = {
            "STRONG OVERBOUGHT": "#ff1744",
            "OVERBOUGHT":        "#ff6d00",
            "NEUTRAL":           "#00c853",
            "OVERSOLD":          "#ffd600",
            "STRONG OVERSOLD":   "#2979ff",
        }.get(signal.verdict, "#ffffff")

        ax1.set_title(
            f"{symbol}  ·  {signal.verdict_emoji} {signal.verdict}  "
            f"(Score: {signal.normalized_score:+.1f} / ±20)",
            color=vc_hex, fontsize=11, fontweight="bold", pad=6,
        )
        ax1.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor="#2a2a4a")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1, N)
        ax1.set_xticks([])

        # Volume overlay on ax1
        ax1v = ax1.twinx()
        vcols = [UP if c[i] >= o[i] else DN for i in range(N)]
        ax1v.bar(dates, _tail("volume"), color=vcols, alpha=0.18, width=0.8)
        ax1v.set_ylabel("Volume", color="#607d8b", fontsize=7)
        ax1v.tick_params(axis="y", labelsize=6, colors="#607d8b")
        ax1v.set_xlim(-1, N)

        # ── Panel 2: RSI ──────────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.set_facecolor(axbg)
        r14 = _tail("rsi_14")
        r6  = _tail("rsi_6")
        ax2.plot(dates, r14, color="#29b6f6", lw=1.3, label="RSI(14)")
        ax2.plot(dates, r6,  color="#f06292", lw=1.0, alpha=0.7, label="RSI(6)")
        for level, col, ls in [(80,"#ff1744",":"), (70,"#ff5252","--"), (30,"#69f0ae","--"), (20,"#2979ff",":")]:
            ax2.axhline(level, color=col, lw=0.7, ls=ls, alpha=0.6)
        ax2.fill_between(dates, 70, r14, where=r14 > 70, alpha=0.12, color="#ff1744")
        ax2.fill_between(dates, r14, 30, where=r14 < 30, alpha=0.12, color="#2979ff")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", color=TXT, fontsize=8)
        ax2.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor="#2a2a4a")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([])

        # ── Panel 3: MACD ─────────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.set_facecolor(axbg)
        ml  = _tail("macd_line")
        msi = _tail("macd_signal")
        mh  = _tail("macd_hist")
        ax3.plot(dates, ml,  color="#29b6f6", lw=1.2, label="MACD")
        ax3.plot(dates, msi, color="#f06292", lw=1.0, label="Signal")
        bar_c = [UP if v >= 0 else DN for v in mh]
        ax3.bar(dates, mh, color=bar_c, alpha=0.65, width=0.8)
        ax3.axhline(0, color="#546e7a", lw=0.7, alpha=0.6)
        ax3.set_ylabel("MACD", color=TXT, fontsize=8)
        ax3.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor="#2a2a4a")
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks([])

        # ── Panel 4: KDJ ─────────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.set_facecolor(axbg)
        kk = _tail("kdj_k")
        kd = _tail("kdj_d")
        kj = _tail("kdj_j")
        ax4.plot(dates, kk, color="#29b6f6", lw=1.2, label="K")
        ax4.plot(dates, kd, color="#ff9800", lw=1.0, label="D")
        ax4.plot(dates, kj, color="#f06292", lw=0.8, alpha=0.7, label="J")
        for level, col in [(80, "#ff5252"), (20, "#69f0ae")]:
            ax4.axhline(level, color=col, lw=0.8, ls="--", alpha=0.6)
        ax4.set_ylabel("KDJ", color=TXT, fontsize=8)
        ax4.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor="#2a2a4a")
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks([])

        # ── Panel 5: CCI ─────────────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        ax5.set_facecolor(axbg)
        cv = _tail("cci")
        ax5.plot(dates, cv, color="#ffd600", lw=1.2, label="CCI(20)")
        for level, col, ls in [(200,"#ff1744",":"), (100,"#ff5252","--"), (-100,"#69f0ae","--"), (-200,"#2979ff",":")]:
            ax5.axhline(level, color=col, lw=0.7, ls=ls, alpha=0.6)
        ax5.fill_between(dates, 100, cv, where=cv >  100, alpha=0.12, color="#ff1744")
        ax5.fill_between(dates, cv, -100, where=cv < -100, alpha=0.12, color="#2979ff")
        ax5.set_ylabel("CCI", color=TXT, fontsize=8)
        ax5.legend(loc="upper left", fontsize=7, facecolor=bg, edgecolor="#2a2a4a")
        ax5.grid(True, alpha=0.3)

        # X-axis date labels on bottom panel
        step = max(1, N // 10)
        ticks = list(range(0, N, step))
        date_idx = disp.index
        ax5.set_xticks(ticks)
        ax5.set_xticklabels(
            [date_idx[t].strftime("%m/%d") for t in ticks if t < len(date_idx)],
            rotation=30, ha="right", fontsize=7,
        )

        plt.suptitle(
            f"Stock Technical Analysis  ·  {symbol}  [{signal.timestamp.strftime('%Y-%m-%d')}]",
            color=TXT, fontsize=12, fontweight="bold", y=1.0,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.999])

        if save_path is None:
            save_path = f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        return save_path


# ══════════════════════════════════════════════════════════════════════════════
#  StockAnalyzer  (orchestrator)
# ══════════════════════════════════════════════════════════════════════════════
class StockAnalyzer:
    """High-level orchestrator: fetch → compute → analyse → display."""

    def __init__(self, period_days: int = 90) -> None:
        self.fetcher    = DataFetcher(period_days)
        self.calculator = IndicatorCalculator()
        self.analyzer   = SignalAnalyzer()
        self.display    = Display()

    def _analyse_one(
        self,
        symbol:     str,
        show_chart: bool = False,
        chart_path: Optional[str] = None,
        print_full: bool = True,
    ) -> Optional[Tuple[Dict[str, Any], ComprehensiveSignal, pd.DataFrame]]:
        """
        Full pipeline for one symbol.

        Returns (info, signal, df) or None on failure.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.display.console,
            transient=True,
        ) as prog:
            task = prog.add_task(f"Analysing [cyan]{symbol}[/cyan]…", total=3)

            df = self.fetcher.fetch_stock_data(symbol)
            if df is None:
                self.display.console.print(
                    f"[red]✗ Cannot fetch data for '{symbol}'. "
                    "Check symbol and network.[/red]"
                )
                return None
            prog.advance(task)

            prog.update(task, description=f"Fetching metadata [{symbol}]…")
            info = self.fetcher.get_stock_info(symbol)
            info["close"]      = f"{df['Close'].iloc[-1]:.2f}"
            info["change_pct"] = (
                (df["Close"].iloc[-1] - df["Close"].iloc[-2])
                / (df["Close"].iloc[-2] + 1e-10) * 100.0
            )
            prog.advance(task)

            prog.update(task, description=f"Computing indicators [{symbol}]…")
            try:
                indicators = self.calculator.calculate_all(df)
            except Exception as exc:
                logger.error("Indicator error for %s: %s", symbol, exc, exc_info=True)
                self.display.console.print(f"[red]✗ Indicator error: {exc}[/red]")
                return None
            prog.advance(task)

        try:
            signal = self.analyzer.analyze(symbol, indicators)
        except Exception as exc:
            logger.error("Analysis error for %s: %s", symbol, exc, exc_info=True)
            self.display.console.print(f"[red]✗ Analysis error: {exc}[/red]")
            return None

        if print_full:
            self.display.print_analysis(info, signal, df)

        if show_chart:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.display.console,
                transient=True,
            ) as prog:
                t = prog.add_task("Generating chart…", total=1)
                path = self.display.plot_chart(symbol, df, indicators, signal, chart_path)
                prog.advance(t)
            if path:
                self.display.console.print(
                    f"[green]✓ Chart saved → [bold]{path}[/bold][/green]"
                )

        return info, signal, df

    def analyse_symbol(
        self,
        symbol:     str,
        show_chart: bool = False,
        chart_path: Optional[str] = None,
    ) -> Optional[ComprehensiveSignal]:
        result = self._analyse_one(symbol, show_chart, chart_path, print_full=True)
        return result[1] if result else None

    def analyse_batch(
        self,
        symbols:    List[str],
        show_chart: bool = False,
    ) -> List[Tuple[Dict[str, Any], ComprehensiveSignal]]:
        """Analyse multiple symbols, print each, then show comparison table."""
        self.display.console.print(
            f"\n[bold cyan]Analysing {len(symbols)} symbol(s)…[/bold cyan]\n"
        )
        rows: List[Tuple[Dict[str, Any], ComprehensiveSignal]] = []

        for sym in symbols:
            sym = sym.strip().upper()
            self.display.console.print(Rule(f"[dim]{sym}[/dim]"))
            result = self._analyse_one(sym, show_chart, print_full=True)
            if result:
                info, sig, _ = result
                rows.append((info, sig))

        if len(rows) > 1:
            self.display.console.print()
            self.display.print_batch_table(rows)

        return rows

    def run_interactive(self) -> None:
        """Interactive CLI mode."""
        self.display.console.print(Panel(
            "[bold cyan]📊 Stock Overbought / Oversold Analysis Tool[/bold cyan]\n"
            "[white]Multi-indicator weighted scoring system[/white]\n"
            "[dim]Supports US (AAPL) · HK (0700.HK) · A-share (600519.SS) · ETF (QQQ)[/dim]",
            border_style="cyan",
            padding=(1, 3),
        ))

        while True:
            self.display.console.print(
                "[bold]Enter symbol(s)[/bold] "
                "[dim](comma-separated, 'q' quit, 'test' demo):[/dim]"
            )
            try:
                raw = input("→ ").strip()
            except (EOFError, KeyboardInterrupt):
                self.display.console.print("\n[dim]Goodbye![/dim]")
                break

            if not raw:
                continue
            if raw.lower() in ("q", "quit", "exit"):
                self.display.console.print("[dim]Goodbye![/dim]")
                break
            if raw.lower() == "test":
                run_self_tests()
                continue

            symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
            if not symbols:
                continue

            try:
                chart_ans = input("Generate chart? (y/[n]): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                chart_ans = "n"
            show_chart = chart_ans in ("y", "yes")

            if len(symbols) == 1:
                self.analyse_symbol(symbols[0], show_chart=show_chart)
            else:
                self.analyse_batch(symbols, show_chart=show_chart)


# ══════════════════════════════════════════════════════════════════════════════
#  Self-Validation Tests
# ══════════════════════════════════════════════════════════════════════════════
def run_self_tests() -> None:
    """Built-in validation suite — no network required for most tests."""
    console.print(Rule("[bold yellow]Self-Validation Test Suite[/bold yellow]"))

    calc     = IndicatorCalculator()
    analyzer = SignalAnalyzer()
    fetcher  = DataFetcher()

    passed = failed = 0
    dates  = pd.date_range("2024-01-01", periods=120, freq="B")

    def ok(name: str, cond: bool, note: str = "") -> None:
        nonlocal passed, failed
        if cond:
            console.print(f"  [green]✓[/green] {name}")
            passed += 1
        else:
            console.print(f"  [red]✗[/red] {name}  [dim]{note}[/dim]")
            failed += 1

    # ── 1. RSI ────────────────────────────────────────────────────────────────
    console.print("\n[bold]1. RSI[/bold]")
    up   = pd.Series(np.linspace(100, 200, 120), index=dates, dtype=float)
    dn   = pd.Series(np.linspace(200, 100, 120), index=dates, dtype=float)
    flat = pd.Series([100.0] * 120, index=dates)

    rsi_up   = calc.calculate_rsi(up,   14)
    rsi_dn   = calc.calculate_rsi(dn,   14)
    rsi_flat = calc.calculate_rsi(flat, 14)

    ok("RSI trending-up last value > 70",  rsi_up.iloc[-1]   > 70,  f"got {rsi_up.iloc[-1]:.1f}")
    ok("RSI trending-dn last value < 30",  rsi_dn.iloc[-1]   < 30,  f"got {rsi_dn.iloc[-1]:.1f}")
    ok("RSI values in [0, 100]",
       rsi_up.dropna().between(0, 100).all() and rsi_dn.dropna().between(0, 100).all())
    ok("RSI flat → no NaN at end",         not np.isnan(rsi_flat.iloc[-1]))

    # ── 2. KDJ ────────────────────────────────────────────────────────────────
    console.print("\n[bold]2. KDJ[/bold]")
    np.random.seed(0)
    p = 100 + np.cumsum(np.random.randn(120) * 0.8)
    cs = pd.Series(p, index=dates)
    hs = cs + np.abs(np.random.randn(120) * 0.5)
    ls = cs - np.abs(np.random.randn(120) * 0.5)

    k, d, j = calc.calculate_kdj(hs, ls, cs)
    ok("K no NaN at end",         not np.isnan(k.iloc[-1]))
    ok("D no NaN at end",         not np.isnan(d.iloc[-1]))
    ok("K in [0, 100]",           0 <= k.iloc[-1] <= 100,  f"K={k.iloc[-1]:.2f}")
    ok("D in [0, 100]",           0 <= d.iloc[-1] <= 100,  f"D={d.iloc[-1]:.2f}")

    # ── 3. Bollinger Bands ────────────────────────────────────────────────────
    console.print("\n[bold]3. Bollinger Bands[/bold]")
    bu, bm, bl, pb = calc.calculate_bbands(cs)
    ok("Upper > Middle always",   (bu.dropna() > bm.dropna()).all())
    ok("Lower < Middle always",   (bl.dropna() < bm.dropna()).all())
    # Sine-wave: one complete 20-period cycle ends at exactly the mean → %B ≈ 0.5
    _sine = pd.Series(
        [100 + 10 * np.sin(2 * np.pi * i / 20) for i in range(119)] + [100.0],
        index=dates[:120],
    )
    _, _bm_s, _, _pb_s = calc.calculate_bbands(_sine)
    ok("Price at MA → %B ≈ 0.5",
       abs(float(_pb_s.iloc[-1]) - 0.5) < 0.12,
       f"got {float(_pb_s.iloc[-1]):.3f}")

    # ── 4. MACD ───────────────────────────────────────────────────────────────
    console.print("\n[bold]4. MACD[/bold]")
    ml, ms, mh = calc.calculate_macd(cs)
    ok("MACD line no NaN",        not np.isnan(ml.iloc[-1]))
    ok("Histogram = line - signal",
       abs(mh.iloc[-1] - (ml.iloc[-1] - ms.iloc[-1])) < 1e-9)

    # ── 5. CCI ────────────────────────────────────────────────────────────────
    console.print("\n[bold]5. CCI[/bold]")
    cci = calc.calculate_cci(hs, ls, cs)
    ok("CCI no NaN at end",       not np.isnan(cci.iloc[-1]))
    ok("CCI absolute value < 500", cci.dropna().abs().max() < 500,
       f"max={cci.dropna().abs().max():.1f}")

    # ── 6. Williams %R ────────────────────────────────────────────────────────
    console.print("\n[bold]6. Williams %R[/bold]")
    wr = calc.calculate_williams_r(hs, ls, cs)
    ok("WR in [−100, 0]", wr.dropna().between(-100, 0).all(),
       f"range: [{wr.dropna().min():.1f}, {wr.dropna().max():.1f}]")

    # ── 7. Stochastic RSI ─────────────────────────────────────────────────────
    console.print("\n[bold]7. Stochastic RSI[/bold]")
    sk, sd = calc.calculate_stoch_rsi(cs)
    ok("Stoch RSI K in [0, 100]",
       sk.dropna().between(0, 100).all(), f"range: [{sk.dropna().min():.1f}, {sk.dropna().max():.1f}]")

    # ── 8. Signal analyzer scoring ────────────────────────────────────────────
    console.print("\n[bold]8. Signal Analyzer — scoring logic[/bold]")
    r80 = analyzer.analyze_rsi14(80.0)
    ok("RSI 80 → score = +2",       r80.raw_score == 2.0, f"got {r80.raw_score}")
    r18 = analyzer.analyze_rsi14(18.0)
    ok("RSI 18 → score = −2",       r18.raw_score == -2.0, f"got {r18.raw_score}")
    r50 = analyzer.analyze_rsi14(50.0)
    ok("RSI 50 → near-neutral",     abs(r50.raw_score) < 0.5, f"got {r50.raw_score:.2f}")
    rwr_ob = analyzer.analyze_williams_r(-5.0)
    ok("WR −5 → score ≥ +1",       rwr_ob.raw_score >= 1.0, f"got {rwr_ob.raw_score}")
    rwr_os = analyzer.analyze_williams_r(-85.0)
    ok("WR −85 → score ≤ −1",      rwr_os.raw_score <= -1.0, f"got {rwr_os.raw_score}")
    rcc_ob = analyzer.analyze_cci(210.0)
    ok("CCI +210 → score = +2",    rcc_ob.raw_score == 2.0, f"got {rcc_ob.raw_score}")
    rcc_os = analyzer.analyze_cci(-210.0)
    ok("CCI −210 → score = −2",    rcc_os.raw_score == -2.0, f"got {rcc_os.raw_score}")

    # ── 9. Weighted score normalisation ──────────────────────────────────────
    console.print("\n[bold]9. Normalisation & Verdict[/bold]")
    all_ob_ind = {
        "rsi_14": 82.0, "rsi_6": 88.0, "rsi_div": "none",
        "kdj_k": 88.0, "kdj_d": 85.0, "kdj_j": 110.0,
        "bb_upper": 110.0, "bb_lower": 90.0, "bb_pct_b": 1.15,
        "macd_line": 2.0, "macd_signal": 1.0, "macd_hist": 0.5,
        "macd_cross": "none",
        "cci": 210.0, "williams_r": -5.0,
        "srsi_k": 92.0, "srsi_d": 90.0,
        "bias_20": 18.0, "bias_60": 14.0,
        "obv_trend": "up", "obv_div": "bearish_divergence",
        "vol_ratio": 1.0, "atr": 2.0, "atr_ratio": 1.5,
        "close": 150.0, "prev_close": 145.0, "ma50": 140.0, "ma200": 130.0,
        "_series": {"macd_hist": pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])},
    }
    sig_ob = analyzer.analyze("TEST", all_ob_ind)
    ok("All-overbought → score > +10",  sig_ob.normalized_score > 10.0,
       f"got {sig_ob.normalized_score:.2f}")
    ok("All-overbought verdict correct", "OVERBOUGHT" in sig_ob.verdict,
       f"got '{sig_ob.verdict}'")

    # ── 10. Error handling ────────────────────────────────────────────────────
    console.print("\n[bold]10. Error Handling[/bold]")
    df_inv = fetcher.fetch_stock_data("XXXXINVALID999ZZZ")
    ok("Invalid symbol → None",         df_inv is None or (df_inv is not None and df_inv.empty))
    short = pd.DataFrame({
        "Close":  [100.0]*10, "Open": [99.0]*10,
        "High":   [101.0]*10, "Low":  [98.0]*10, "Volume": [1e6]*10,
    }, index=pd.date_range("2024-01-01", periods=10))
    ok("Short data fails validation",    not fetcher.validate_data(short, "SHORT"))
    ok("Empty df fails validation",      not fetcher.validate_data(pd.DataFrame(), "EMPTY"))

    # ── 11. Real data (AAPL) ─────────────────────────────────────────────────
    console.print("\n[bold]11. Real Data — AAPL (requires network)[/bold]")
    try:
        df_aapl = fetcher.fetch_stock_data("AAPL")
        if df_aapl is not None and not df_aapl.empty:
            ok("AAPL fetch succeeded",      True)
            ind = calc.calculate_all(df_aapl)
            ok("AAPL RSI(14) in [0, 100]",  0 <= ind["rsi_14"] <= 100, f"{ind['rsi_14']:.1f}")
            ok("AAPL KDJ K in [0, 100]",    0 <= ind["kdj_k"]  <= 100, f"{ind['kdj_k']:.1f}")
            ok("AAPL %B computed",          not np.isnan(ind["bb_pct_b"]), f"{ind['bb_pct_b']:.3f}")
            sig = analyzer.analyze("AAPL", ind)
            ok("AAPL score in [−20, +20]",  -20 <= sig.normalized_score <= 20,
               f"{sig.normalized_score:.2f}")
        else:
            console.print("  [yellow]⚠ AAPL data unavailable (network?)[/yellow]")
    except Exception as exc:
        console.print(f"  [yellow]⚠ AAPL test skipped: {exc}[/yellow]")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    console.print(f"\n[bold]Results: [green]{passed}[/green] / {total} passed"
                  + (f",  [red]{failed} failed[/red]" if failed else "") + "[/bold]")
    if failed == 0:
        console.print("[bold green]✓ All tests passed![/bold green]\n")
    else:
        console.print(f"[bold yellow]⚠ {failed} test(s) failed — see above[/bold yellow]\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Parse CLI arguments and run the appropriate mode."""
    parser = argparse.ArgumentParser(
        description="Stock Overbought/Oversold Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stock_analyzer.py                            # Interactive
  python stock_analyzer.py --symbols AAPL             # Single stock
  python stock_analyzer.py --symbols AAPL,MU,0700.HK # Batch
  python stock_analyzer.py --symbols TSLA --chart     # With chart
  python stock_analyzer.py --test                     # Self-tests
        """,
    )
    parser.add_argument("--symbols", "-s", type=str,
                        help="Comma-separated ticker symbols")
    parser.add_argument("--period",  "-p", type=int,  default=90,
                        help="Analysis period in days (default: 90)")
    parser.add_argument("--chart",   "-c", action="store_true",
                        help="Save PNG chart(s)")
    parser.add_argument("--test",          action="store_true",
                        help="Run self-validation tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tool = StockAnalyzer(period_days=args.period)

    if args.test:
        run_self_tests()
        return

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if len(syms) == 1:
            tool.analyse_symbol(syms[0], show_chart=args.chart)
        else:
            tool.analyse_batch(syms, show_chart=args.chart)
    else:
        tool.run_interactive()


if __name__ == "__main__":
    main()
