# multi_timeframe.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

class MultiTimeframeAnalyzer:
    def __init__(self, base_timeframe: str = "1m"):
        self.base_timeframe = base_timeframe
        self.timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

    def derive_higher_timeframes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                try:
                    df = df.set_index("timestamp")
                except Exception as e:
                    logging.warning(f"Failed to set timestamp index: {e}")
                    return {}
            elif "date" in df.columns:
                try:
                    df = df.set_index("date")
                except Exception as e:
                    logging.warning(f"Failed to set date index: {e}")
                    return {}
            else:
                logging.warning("No timestamp column for multi-timeframe analysis")
                return {}

        results = {self.base_timeframe: df}

        rules = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D"
        }

        target_timeframes = [tf for tf in self.timeframes if tf != self.base_timeframe]

        for tf in target_timeframes:
            if tf not in rules:
                continue

            rule = rules[tf]
            try:
                resampled = df.resample(rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })

                resampled = resampled.dropna()

                if len(resampled) >= 10:
                    results[tf] = resampled
                    logging.info(f"Generated {tf} timeframe with {len(resampled)} bars")
            except Exception as e:
                logging.warning(f"Failed to resample to {tf}: {e}")

        return results

    def analyze_multi_timeframe(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        if not timeframe_data:
            return {"valid": False, "message": "No timeframe data available"}

        trends = {}

        for tf, df in timeframe_data.items():
            if len(df) < 3:
                continue

            last_3_closes = df['Close'].iloc[-3:].values

            if last_3_closes[2] > last_3_closes[0]:
                trend = "bullish"
            elif last_3_closes[2] < last_3_closes[0]:
                trend = "bearish"
            else:
                trend = "neutral"

            momentum = ((last_3_closes[2] / last_3_closes[0]) - 1) * 100

            trends[tf] = {
                "trend": trend,
                "momentum": momentum
            }

        if not trends:
            return {"valid": False, "message": "No valid trends calculated"}

        bullish_count = sum(1 for t in trends.values() if t["trend"] == "bullish")
        bearish_count = sum(1 for t in trends.values() if t["trend"] == "bearish")
        neutral_count = sum(1 for t in trends.values() if t["trend"] == "neutral")
        total_timeframes = len(trends)

        if total_timeframes > 0:
            bullish_alignment = bullish_count / total_timeframes
            bearish_alignment = bearish_count / total_timeframes

            if bullish_alignment > 0.5:
                alignment = "bullish"
                alignment_strength = bullish_alignment
            elif bearish_alignment > 0.5:
                alignment = "bearish"
                alignment_strength = bearish_alignment
            else:
                alignment = "mixed"
                alignment_strength = 0.5
        else:
            alignment = "unknown"
            alignment_strength = 0.0

        lower_tfs = ["1m", "5m", "15m"]
        higher_tfs = ["1h", "4h", "1d"]

        lower_bullish = any(trends.get(tf, {}).get("trend") == "bullish" for tf in lower_tfs if tf in trends)
        lower_bearish = any(trends.get(tf, {}).get("trend") == "bearish" for tf in lower_tfs if tf in trends)

        higher_bullish = any(trends.get(tf, {}).get("trend") == "bullish" for tf in higher_tfs if tf in trends)
        higher_bearish = any(trends.get(tf, {}).get("trend") == "bearish" for tf in higher_tfs if tf in trends)

        bullish_divergence = lower_bullish and higher_bearish
        bearish_divergence = lower_bearish and higher_bullish

        divergence = None
        if bullish_divergence:
            divergence = "bullish_divergence"
        elif bearish_divergence:
            divergence = "bearish_divergence"

        return {
            "valid": True,
            "timeframe_trends": trends,
            "alignment": alignment,
            "alignment_strength": alignment_strength,
            "divergence": divergence,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count
        }
