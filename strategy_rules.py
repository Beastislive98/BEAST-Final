# strategy_rules.py - Complete Enhanced Version with All Strategies

import logging
import numpy as np
from typing import Dict, Any, Union, List, Optional
from functools import lru_cache
import time

# Performance tracking for dynamic thresholds
STRATEGY_PERFORMANCE = {}

# Cache for expensive calculations
@lru_cache(maxsize=128)
def get_cached_market_volatility(symbol: str, timestamp: int) -> float:
    """Cached volatility calculation to avoid recalculation"""
    # This would normally connect to your volatility calculation
    # For now, return a default that can be overridden
    return 0.015  # 1.5% default volatility

def update_strategy_performance(strategy_name: str, success: bool, return_pct: float):
    """Update strategy performance metrics for dynamic adaptation"""
    if strategy_name not in STRATEGY_PERFORMANCE:
        STRATEGY_PERFORMANCE[strategy_name] = {
            "trades": 0, "wins": 0, "avg_return": 0.0, "win_rate": 0.0
        }
    
    stats = STRATEGY_PERFORMANCE[strategy_name]
    stats["trades"] += 1
    if success:
        stats["wins"] += 1
    
    # Update moving averages
    stats["win_rate"] = stats["wins"] / stats["trades"]
    stats["avg_return"] = ((stats["avg_return"] * (stats["trades"] - 1)) + return_pct) / stats["trades"]

# Enhanced helper functions with market regime awareness
def is_valid(bundle: Dict[str, Any]) -> bool:
    """Check if the bundle is valid and contains data"""
    return isinstance(bundle, dict) and len(bundle.keys()) > 0

def get_safe(bundle: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    """Safely retrieve a nested value from a dictionary using a path list"""
    temp = bundle
    for key in path:
        if isinstance(temp, dict) and key in temp:
            temp = temp[key]
        else:
            return default
    return temp

def get_market_regime(bundle: Dict[str, Any]) -> str:
    """Get current market regime"""
    return get_safe(bundle, ['market_regime', 'regime_type'], 'unknown')

def get_regime_confidence(bundle: Dict[str, Any]) -> float:
    """Get market regime confidence"""
    return get_safe(bundle, ['market_regime', 'confidence'], 0.0)

def get_dynamic_threshold(base_threshold: float, bundle: Dict[str, Any], strategy_name: str = None) -> float:
    """Calculate dynamic threshold based on market conditions and strategy performance"""
    
    # Get market volatility
    volatility = get_market_volatility(bundle)
    if volatility is None:
        volatility = 0.015  # Default 1.5%
    
    # Adjust threshold based on volatility
    volatility_multiplier = max(0.5, min(2.0, volatility / 0.015))  # Scale around 1.5% baseline
    adjusted_threshold = base_threshold * volatility_multiplier
    
    # Market regime adjustments
    regime = get_market_regime(bundle)
    regime_confidence = get_regime_confidence(bundle)
    
    if regime == 'high_volatility' and regime_confidence > 0.7:
        adjusted_threshold *= 1.3  # Higher bar in volatile markets
    elif regime == 'ranging' and regime_confidence > 0.7:
        adjusted_threshold *= 0.8  # Lower bar in ranging markets
    
    # Strategy performance adjustments
    if strategy_name and strategy_name in STRATEGY_PERFORMANCE:
        perf = STRATEGY_PERFORMANCE[strategy_name]
        if perf["trades"] >= 10:  # Enough data for adjustment
            if perf["win_rate"] > 0.7:
                adjusted_threshold *= 0.9  # Lower bar for successful strategies
            elif perf["win_rate"] < 0.4:
                adjusted_threshold *= 1.2  # Higher bar for poor performers
    
    return adjusted_threshold

def get_rsi(bundle: Dict[str, Any]) -> Optional[float]:
    """Helper function to get RSI value safely"""
    rsi = get_safe(bundle, ['indicator_data', 'RSI_14'])
    if hasattr(rsi, 'iloc') and len(rsi) > 0:
        return rsi.iloc[-1]
    return None

def get_price_trend(bundle: Dict[str, Any], periods: int = 5) -> Optional[str]:
    """Helper function to determine price trend over n periods"""
    try:
        closes = get_safe(bundle, ['indicator_data', 'Close'])
        if hasattr(closes, 'iloc') and len(closes) >= periods:
            last_n = closes.iloc[-periods:].values
            avg_diff = np.mean(np.diff(last_n))
            if avg_diff > 0:
                return "up"
            elif avg_diff < 0:
                return "down"
            return "sideways"
    except:
        pass
    return None

def get_pattern_confidence(bundle: Dict[str, Any]) -> float:
    """Helper function to get pattern confidence"""
    return get_safe(bundle, ['pattern_signal', 'confidence'], 0.0)

def get_market_volatility(bundle: Dict[str, Any]) -> Optional[float]:
    """Helper function to get ATR-based volatility"""
    atr = get_safe(bundle, ['indicator_data', 'ATR_14'])
    price = get_safe(bundle, ['indicator_data', 'Close'])
    if hasattr(atr, 'iloc') and hasattr(price, 'iloc') and len(atr) > 0 and len(price) > 0:
        return atr.iloc[-1] / price.iloc[-1]
    return None

def get_funding_rate_signal(bundle: Dict[str, Any]) -> Optional[float]:
    """Get funding rate for crypto perpetual futures"""
    return get_safe(bundle, ['market_data', 'funding_rate'], 0.0)

def get_open_interest_change(bundle: Dict[str, Any]) -> Optional[float]:
    """Get open interest change percentage"""
    return get_safe(bundle, ['market_data', 'oi_change_pct'], 0.0)

def get_cross_exchange_spread(bundle: Dict[str, Any]) -> Optional[float]:
    """Get price spread across exchanges"""
    return get_safe(bundle, ['market_data', 'cross_exchange_spread'], 0.0)

def has_whale_activity(bundle: Dict[str, Any]) -> bool:
    """Check for whale activity"""
    return get_safe(bundle, ['whale_flags', 'whale_present'], False)

def get_order_flow_pressure(bundle: Dict[str, Any]) -> float:
    """Get order flow pressure from order book analysis"""
    return get_safe(bundle, ['order_flow', 'pressure_strength'], 0.0)

def get_multi_timeframe_alignment(bundle: Dict[str, Any]) -> float:
    """Get multi-timeframe trend alignment strength"""
    return get_safe(bundle, ['mtf_analysis', 'alignment_strength'], 0.0)

# ==================== NEW CRYPTO-SPECIFIC STRATEGIES ====================

def funding_rate_arbitrage_logic(bundle: Dict[str, Any]) -> bool:
    """Funding Rate Arbitrage: exploit funding rate imbalances"""
    if not is_valid(bundle):
        return False
    
    funding_rate = get_funding_rate_signal(bundle)
    if funding_rate is None:
        return False
    
    # Dynamic threshold based on market conditions
    base_threshold = 0.01  # 1% funding rate
    threshold = get_dynamic_threshold(base_threshold, bundle, "funding_rate_arbitrage")
    
    # High funding rates indicate over-leveraged longs
    return abs(funding_rate) > threshold

def perpetual_futures_momentum_logic(bundle: Dict[str, Any]) -> bool:
    """Perpetual Futures Momentum: OI change + price momentum"""
    if not is_valid(bundle):
        return False
    
    oi_change = get_open_interest_change(bundle)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    if oi_change is None:
        return False
    
    base_threshold = 0.05  # 5% slope
    slope_threshold = get_dynamic_threshold(base_threshold, bundle, "perpetual_momentum")
    
    # Strong price movement with increasing open interest
    return abs(slope) > slope_threshold and oi_change > 10  # 10% OI increase

def cross_exchange_arbitrage_logic(bundle: Dict[str, Any]) -> bool:
    """Cross-Exchange Arbitrage: price differences across exchanges"""
    if not is_valid(bundle):
        return False
    
    spread = get_cross_exchange_spread(bundle)
    if spread is None:
        return False
    
    base_threshold = 0.002  # 0.2% spread
    threshold = get_dynamic_threshold(base_threshold, bundle, "cross_exchange_arb")
    
    return abs(spread) > threshold

def defi_yield_strategy_logic(bundle: Dict[str, Any]) -> bool:
    """DeFi Yield Strategy: governance token + yield opportunities"""
    if not is_valid(bundle):
        return False
    
    # Check if this is a DeFi/governance token (simplified)
    symbol = bundle.get("symbol", "")
    defi_tokens = ["UNI", "SUSHI", "COMP", "AAVE", "MKR", "CRV", "1INCH"]
    
    if not any(token in symbol for token in defi_tokens):
        return False
    
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "defi_yield")
    
    return sentiment > 0.3 and slope > threshold

def spot_futures_arbitrage_logic(bundle: Dict[str, Any]) -> bool:
    """Spot-Futures Arbitrage: basis trading"""
    if not is_valid(bundle):
        return False
    
    # This would compare spot vs futures prices
    spot_price = get_safe(bundle, ['market_data', 'spot_price'], 0)
    futures_price = get_safe(bundle, ['market_data', 'price'], 0)
    
    if not spot_price or not futures_price:
        return False
    
    basis = (futures_price - spot_price) / spot_price
    base_threshold = 0.001  # 0.1% basis
    threshold = get_dynamic_threshold(base_threshold, bundle, "spot_futures_arb")
    
    return abs(basis) > threshold

def crypto_momentum_breakout_logic(bundle: Dict[str, Any]) -> bool:
    """Crypto Momentum Breakout: high volume + whale activity + breakout"""
    if not is_valid(bundle):
        return False
    
    whale_activity = has_whale_activity(bundle)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    confidence = get_pattern_confidence(bundle)
    
    base_threshold = 0.06
    slope_threshold = get_dynamic_threshold(base_threshold, bundle, "crypto_momentum")
    
    # Multi-timeframe confirmation
    mtf_alignment = get_multi_timeframe_alignment(bundle)
    
    return (whale_activity and 
            abs(slope) > slope_threshold and 
            confidence > 0.7 and
            mtf_alignment > 0.6)

def altcoin_rotation_logic(bundle: Dict[str, Any]) -> bool:
    """Altcoin Rotation: sector rotation based on market cap"""
    if not is_valid(bundle):
        return False
    
    # Check if this is an altcoin (not BTC/ETH)
    symbol = bundle.get("symbol", "")
    major_coins = ["BTC", "ETH"]
    
    if any(coin in symbol for coin in major_coins):
        return False
    
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    regime = get_market_regime(bundle)
    
    base_threshold = 0.04
    threshold = get_dynamic_threshold(base_threshold, bundle, "altcoin_rotation")
    
    # Altcoins perform well in bull markets with positive sentiment
    return (regime == "bull_trend" and 
            sentiment > 0.4 and 
            slope > threshold)

def adaptive_momentum_logic(bundle: Dict[str, Any]) -> bool:
    """Adaptive Momentum: self-adjusting based on recent performance"""
    if not is_valid(bundle):
        return False
    
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    confidence = get_pattern_confidence(bundle)
    
    # Base thresholds
    base_slope = 0.05
    base_confidence = 0.7
    
    # Get performance-adjusted thresholds
    slope_threshold = get_dynamic_threshold(base_slope, bundle, "adaptive_momentum")
    confidence_threshold = get_dynamic_threshold(base_confidence, bundle, "adaptive_momentum")
    
    # Multi-factor confirmation
    whale_activity = has_whale_activity(bundle)
    mtf_alignment = get_multi_timeframe_alignment(bundle)
    
    return (abs(slope) > slope_threshold and 
            confidence > confidence_threshold and
            (whale_activity or mtf_alignment > 0.6))

# ==================== ENHANCED TRADITIONAL STRATEGIES ====================

def long_call_logic(bundle: Dict[str, Any]) -> bool:
    """Enhanced Long Call: bullish outlook with regime awareness"""
    if not is_valid(bundle): 
        return False
    
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    regime = get_market_regime(bundle)
    
    # Dynamic thresholds based on market conditions
    base_slope_threshold = 0.05
    base_sentiment_threshold = 0.3
    
    slope_threshold = get_dynamic_threshold(base_slope_threshold, bundle, "long_call")
    sentiment_threshold = get_dynamic_threshold(base_sentiment_threshold, bundle, "long_call")
    
    # Enhanced conditions with regime awareness
    rsi = get_rsi(bundle)
    price_trend = get_price_trend(bundle)
    
    basic_condition = slope > slope_threshold and sentiment > sentiment_threshold
    
    # Regime-specific adjustments
    if regime == "bear_trend":
        # Higher bar in bear markets
        basic_condition = basic_condition and sentiment > 0.5
    elif regime == "bull_trend":
        # Lower bar in bull markets
        sentiment_threshold *= 0.8
        basic_condition = slope > slope_threshold and sentiment > sentiment_threshold
    
    # Additional confirmations if available
    if rsi is not None and price_trend:
        return basic_condition and rsi < 70 and price_trend == "up"
    
    return basic_condition

def long_put_logic(bundle: Dict[str, Any]) -> bool:
    """Enhanced Long Put: bearish outlook with regime awareness"""
    if not is_valid(bundle): 
        return False
    
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    regime = get_market_regime(bundle)
    
    # Dynamic thresholds
    base_slope_threshold = -0.05
    base_sentiment_threshold = -0.3
    
    slope_threshold = get_dynamic_threshold(abs(base_slope_threshold), bundle, "long_put")
    sentiment_threshold = get_dynamic_threshold(abs(base_sentiment_threshold), bundle, "long_put")
    
    # Enhanced conditions
    rsi = get_rsi(bundle)
    price_trend = get_price_trend(bundle)
    
    basic_condition = slope < -slope_threshold and sentiment < -sentiment_threshold
    
    # Regime adjustments
    if regime == "bull_trend":
        # Higher bar in bull markets for bearish trades
        basic_condition = basic_condition and sentiment < -0.5
    elif regime == "bear_trend":
        # Lower bar in bear markets
        basic_condition = slope < -slope_threshold * 0.8 and sentiment < -sentiment_threshold * 0.8
    
    # Additional confirmations
    if rsi is not None and price_trend:
        return basic_condition and rsi > 30 and price_trend == "down"
    
    return basic_condition

def bull_call_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Bull Call Spread: moderately bullish with RSI not overbought"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    rsi = get_rsi(bundle)
    
    base_threshold = 0.04
    threshold = get_dynamic_threshold(base_threshold, bundle, "bull_call_spread")
    
    return slope > threshold and rsi is not None and rsi < 70

def bear_put_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Bear Put Spread: moderately bearish with RSI not oversold"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    rsi = get_rsi(bundle)
    
    base_threshold = 0.04
    threshold = get_dynamic_threshold(base_threshold, bundle, "bear_put_spread")
    
    return slope < -threshold and rsi is not None and rsi > 30

def debit_call_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Debit Call Spread: bullish with strong pattern confidence"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    confidence = get_pattern_confidence(bundle)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "debit_call_spread")
    
    return slope > threshold and confidence > 0.7

def debit_put_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Debit Put Spread: bearish with strong pattern confidence"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    confidence = get_pattern_confidence(bundle)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "debit_put_spread")
    
    return slope < -threshold and confidence > 0.7

def long_straddle_logic(bundle: Dict[str, Any]) -> bool:
    """Enhanced Long Straddle: volatility play with regime awareness"""
    if not is_valid(bundle): 
        return False
    
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    volatility = get_market_volatility(bundle)
    regime = get_market_regime(bundle)
    
    base_threshold = 0.08
    slope_threshold = get_dynamic_threshold(base_threshold, bundle, "long_straddle")
    
    basic_condition = abs(slope) > slope_threshold and abs(sentiment) > 0.5
    
    # Regime adjustments
    if regime == "high_volatility":
        # Even better conditions for straddles in volatile markets
        basic_condition = abs(slope) > slope_threshold * 0.8 and abs(sentiment) > 0.4
    
    # Enhanced volatility confirmation
    if volatility is not None:
        vol_threshold = get_dynamic_threshold(0.02, bundle, "long_straddle")
        return basic_condition and volatility > vol_threshold
    
    return basic_condition

def long_strangle_logic(bundle: Dict[str, Any]) -> bool:
    """Long Strangle: volatility play with whale activity"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    whale_present = get_safe(bundle, ['whale_flags', 'whale_present'], False)
    
    base_threshold = 0.07
    threshold = get_dynamic_threshold(base_threshold, bundle, "long_strangle")
    
    return abs(slope) > threshold and whale_present

def long_call_butterfly_logic(bundle: Dict[str, Any]) -> bool:
    """Long Call Butterfly: defined range with bullish sentiment"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    return rsi is not None and rsi > 60 and sentiment > 0.2

def long_put_butterfly_logic(bundle: Dict[str, Any]) -> bool:
    """Long Put Butterfly: defined range with bearish sentiment"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    return rsi is not None and rsi < 40 and sentiment < -0.2

def long_call_condor_logic(bundle: Dict[str, Any]) -> bool:
    """Long Call Condor: slightly bullish with pattern recognition"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    pattern = get_safe(bundle, ['pattern_signal', 'pattern'])
    
    base_threshold = 0.02
    threshold = get_dynamic_threshold(base_threshold, bundle, "long_call_condor")
    
    return slope > threshold and pattern is not None

def long_put_condor_logic(bundle: Dict[str, Any]) -> bool:
    """Long Put Condor: slightly bearish with pattern recognition"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    pattern = get_safe(bundle, ['pattern_signal', 'pattern'])
    
    base_threshold = 0.02
    threshold = get_dynamic_threshold(base_threshold, bundle, "long_put_condor")
    
    return slope < -threshold and pattern is not None

def ratio_call_backspread_logic(bundle: Dict[str, Any]) -> bool:
    """Ratio Call Backspread: strongly bullish with positive sentiment"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.06
    threshold = get_dynamic_threshold(base_threshold, bundle, "ratio_call_backspread")
    
    return slope > threshold and sentiment > 0.4

def ratio_put_backspread_logic(bundle: Dict[str, Any]) -> bool:
    """Ratio Put Backspread: strongly bearish with negative sentiment"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.06
    threshold = get_dynamic_threshold(base_threshold, bundle, "ratio_put_backspread")
    
    return slope < -threshold and sentiment < -0.4

def calendar_call_logic(bundle: Dict[str, Any]) -> bool:
    """Calendar Call: slightly bullish with moderate RSI"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return rsi is not None and rsi < 60 and slope > 0

def calendar_put_logic(bundle: Dict[str, Any]) -> bool:
    """Calendar Put: slightly bearish with moderate RSI"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return rsi is not None and rsi > 40 and slope < 0

def diagonal_call_logic(bundle: Dict[str, Any]) -> bool:
    """Diagonal Call: positive sentiment and bullish trend"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "diagonal_call")
    
    return sentiment > 0.3 and slope > threshold

def diagonal_put_logic(bundle: Dict[str, Any]) -> bool:
    """Diagonal Put: negative sentiment and bearish trend"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "diagonal_put")
    
    return sentiment < -0.3 and slope < -threshold

def short_call_logic(bundle: Dict[str, Any]) -> bool:
    """Short Call: overbought RSI"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    
    return rsi is not None and rsi > 70

def short_put_logic(bundle: Dict[str, Any]) -> bool:
    """Short Put: oversold RSI"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    
    return rsi is not None and rsi < 30

def bear_call_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Bear Call Spread: bearish trend with negative sentiment"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "bear_call_spread")
    
    return slope < -threshold and sentiment < 0

def bull_put_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Bull Put Spread: bullish trend with positive sentiment"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "bull_put_spread")
    
    return slope > threshold and sentiment > 0

def credit_call_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Credit Call Spread: high RSI (overbought)"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    
    return rsi is not None and rsi > 65

def credit_put_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Credit Put Spread: low RSI (oversold)"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    
    return rsi is not None and rsi < 35

def short_straddle_logic(bundle: Dict[str, Any]) -> bool:
    """Short Straddle: low volatility expectation"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    volatility = get_market_volatility(bundle)
    
    basic_condition = abs(sentiment) < 0.2 and abs(slope) < 0.02
    
    # If we have volatility data, expect low volatility
    if volatility is not None:
        return basic_condition and volatility < 0.01
    
    return basic_condition

def short_strangle_logic(bundle: Dict[str, Any]) -> bool:
    """Short Strangle: low-medium volatility expectation"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return abs(sentiment) < 0.2 and abs(slope) < 0.03

def iron_butterfly_logic(bundle: Dict[str, Any]) -> bool:
    """Iron Butterfly: very low volatility"""
    # Reuse short straddle logic
    return short_straddle_logic(bundle)

def iron_condor_logic(bundle: Dict[str, Any]) -> bool:
    """Iron Condor: low-medium volatility"""
    # Reuse short strangle logic
    return short_strangle_logic(bundle)

def jade_lizard_logic(bundle: Dict[str, Any]) -> bool:
    """Jade Lizard: moderate RSI with positive bias"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    
    return rsi is not None and rsi > 50

def gamma_scalping_logic(bundle: Dict[str, Any]) -> bool:
    """Enhanced Gamma Scalping: with order flow and multi-timeframe"""
    if not is_valid(bundle): 
        return False
    
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    confidence = get_pattern_confidence(bundle)
    order_flow = get_order_flow_pressure(bundle)
    mtf_alignment = get_multi_timeframe_alignment(bundle)
    
    base_slope_threshold = 0.08
    slope_threshold = get_dynamic_threshold(base_slope_threshold, bundle, "gamma_scalping")
    
    base_confidence_threshold = 0.85
    confidence_threshold = get_dynamic_threshold(base_confidence_threshold, bundle, "gamma_scalping")
    
    # Enhanced conditions with order flow
    return (abs(slope) > slope_threshold and 
            confidence > confidence_threshold and
            order_flow > 0.6 and  # Strong order flow pressure
            mtf_alignment > 0.7)   # Multi-timeframe alignment

def event_sniper_logic(bundle: Dict[str, Any]) -> bool:
    """Event Sniper: very strong sentiment with high pattern confidence"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    confidence = get_pattern_confidence(bundle)
    
    base_sentiment_threshold = 0.7
    sentiment_threshold = get_dynamic_threshold(base_sentiment_threshold, bundle, "event_sniper")
    
    base_confidence_threshold = 0.85
    confidence_threshold = get_dynamic_threshold(base_confidence_threshold, bundle, "event_sniper")
    
    return sentiment > sentiment_threshold and confidence > confidence_threshold

def statistical_arb_logic(bundle: Dict[str, Any]) -> bool:
    """Enhanced Statistical Arbitrage: mean reversion with regime awareness"""
    if not is_valid(bundle): 
        return False
    
    rsi = get_rsi(bundle)
    regime = get_market_regime(bundle)
    volatility = get_market_volatility(bundle)
    
    if rsi is None:
        return False
    
    # Dynamic RSI bounds based on regime
    if regime == "ranging":
        # Tighter bounds in ranging markets
        return 47 < rsi < 53
    elif regime == "high_volatility":
        # Wider bounds in volatile markets
        return 40 < rsi < 60
    else:
        # Default bounds
        return 45 < rsi < 55

def latency_arbitrage_logic(bundle: Dict[str, Any]) -> bool:
    """Latency Arbitrage: whale present with bullish forecast"""
    if not is_valid(bundle): 
        return False
    whale_present = get_safe(bundle, ['whale_flags', 'whale_present'], False)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.05
    threshold = get_dynamic_threshold(base_threshold, bundle, "latency_arbitrage")
    
    return whale_present and slope > threshold

def delta_hedge_logic(bundle: Dict[str, Any]) -> bool:
    """Delta Hedge: very flat forecast"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return abs(slope) < 0.01

def volatility_arb_logic(bundle: Dict[str, Any]) -> bool:
    """Enhanced Volatility Arbitrage: regime-aware volatility trading"""
    if not is_valid(bundle): 
        return False
    
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    volatility = get_market_volatility(bundle)
    regime = get_market_regime(bundle)
    regime_confidence = get_regime_confidence(bundle)
    
    base_threshold = 0.06
    slope_threshold = get_dynamic_threshold(base_threshold, bundle, "volatility_arb")
    
    basic_condition = abs(slope) > slope_threshold
    
    # Enhanced volatility analysis with regime awareness
    if volatility is not None and regime_confidence > 0.7:
        expected_move = abs(slope)
        implied_move = volatility * 2  # rough approximation
        
        # Regime-specific adjustments
        if regime == "high_volatility":
            # Expect larger divergences in volatile regimes
            divergence_threshold = 0.03
        else:
            # Smaller divergences in stable regimes
            divergence_threshold = 0.02
        
        return abs(expected_move - implied_move) > divergence_threshold
    
    return basic_condition

def machine_learning_options_logic(bundle: Dict[str, Any]) -> bool:
    """ML Options: extremely high confidence with bullish forecast"""
    if not is_valid(bundle): 
        return False
    confidence = get_pattern_confidence(bundle)
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_confidence_threshold = 0.9
    confidence_threshold = get_dynamic_threshold(base_confidence_threshold, bundle, "machine_learning_options")
    
    base_slope_threshold = 0.05
    slope_threshold = get_dynamic_threshold(base_slope_threshold, bundle, "machine_learning_options")
    
    return confidence > confidence_threshold and slope > slope_threshold

def exotic_option_combo_logic(bundle: Dict[str, Any]) -> bool:
    """Exotic Option Combo: positive sentiment with moderate RSI"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    rsi = get_rsi(bundle)
    
    return sentiment > 0.3 and rsi is not None and rsi < 60

def collar_protective_logic(bundle: Dict[str, Any]) -> bool:
    """Collar Protective: bearish forecast with negative sentiment"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "collar_protective")
    
    return slope < -threshold and sentiment < -0.2

def box_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Box Spread: flat forecast with sufficient pattern confidence"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    confidence = get_pattern_confidence(bundle)
    
    return abs(slope) < 0.01 and confidence > 0.5

def ladder_call_logic(bundle: Dict[str, Any]) -> bool:
    """Ladder Call: strong bullish forecast"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.05
    threshold = get_dynamic_threshold(base_threshold, bundle, "ladder_call")
    
    return slope > threshold

def ladder_put_logic(bundle: Dict[str, Any]) -> bool:
    """Ladder Put: strong bearish forecast"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.05
    threshold = get_dynamic_threshold(base_threshold, bundle, "ladder_put")
    
    return slope < -threshold

def earnings_play_logic(bundle: Dict[str, Any]) -> bool:
    """Earnings Play: strong sentiment with whale activity"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    whale_present = get_safe(bundle, ['whale_flags', 'whale_present'], False)
    
    base_threshold = 0.7
    threshold = get_dynamic_threshold(base_threshold, bundle, "earnings_play")
    
    return sentiment > threshold and whale_present

def index_arbitrage_logic(bundle: Dict[str, Any]) -> bool:
    """Index Arbitrage: moderate bullish forecast"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.04
    threshold = get_dynamic_threshold(base_threshold, bundle, "index_arbitrage")
    
    return slope > threshold

def barrier_option_strategy_logic(bundle: Dict[str, Any]) -> bool:
    """Barrier Option Strategy: mild bullish forecast with decent confidence"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    confidence = get_pattern_confidence(bundle)
    
    base_threshold = 0.02
    threshold = get_dynamic_threshold(base_threshold, bundle, "barrier_option_strategy")
    
    return slope > threshold and confidence > 0.6

def digital_option_strategy_logic(bundle: Dict[str, Any]) -> bool:
    """Digital Option Strategy: strong positive sentiment"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.5
    threshold = get_dynamic_threshold(base_threshold, bundle, "digital_option_strategy")
    
    return sentiment > threshold

def lookback_option_strategy_logic(bundle: Dict[str, Any]) -> bool:
    """Lookback Option Strategy: bullish with not overbought"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    rsi = get_rsi(bundle)
    
    base_threshold = 0.03
    threshold = get_dynamic_threshold(base_threshold, bundle, "lookback_option_strategy")
    
    return slope > threshold and rsi is not None and rsi < 60

def asian_option_strategy_logic(bundle: Dict[str, Any]) -> bool:
    """Asian Option Strategy: moderate directional move expected"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    base_threshold = 0.04
    threshold = get_dynamic_threshold(base_threshold, bundle, "asian_option_strategy")
    
    return abs(slope) > threshold

def stat_arbitrage_hft_logic(bundle: Dict[str, Any]) -> bool:
    """Statistical Arbitrage HFT: strong directional move with whale activity"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    whale_present = get_safe(bundle, ['whale_flags', 'whale_present'], False)
    
    base_threshold = 0.07
    threshold = get_dynamic_threshold(base_threshold, bundle, "stat_arbitrage_hft")
    
    return abs(slope) > threshold and whale_present

def rebate_arbitrage_logic(bundle: Dict[str, Any]) -> bool:
    """Rebate Arbitrage: bullish forecast with positive sentiment"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.06
    threshold = get_dynamic_threshold(base_threshold, bundle, "rebate_arbitrage")
    
    return slope > threshold and sentiment > 0.2

def market_making_hft_logic(bundle: Dict[str, Any]) -> bool:
    """Market Making HFT: flat market with low sentiment"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    volatility = get_market_volatility(bundle)
    
    basic_condition = abs(slope) < 0.02 and abs(sentiment) < 0.1
    
    # If we have volatility, check for optimal conditions
    if volatility is not None:
        # For market making, we want some volatility but not too much
        return basic_condition and 0.005 < volatility < 0.02
    
    return basic_condition

def order_flow_strategy_logic(bundle: Dict[str, Any]) -> bool:
    """Order Flow Strategy: whale activity detected"""
    if not is_valid(bundle): 
        return False
    whale_present = get_safe(bundle, ['whale_flags', 'whale_present'], False)
    
    return whale_present

def tick_arbitrage_logic(bundle: Dict[str, Any]) -> bool:
    """Tick Arbitrage: very slight bullish trend"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return slope > 0.01 and slope < 0.03

def event_arb_hft_logic(bundle: Dict[str, Any]) -> bool:
    """Event Arbitrage HFT: strong sentiment signal"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.6
    threshold = get_dynamic_threshold(base_threshold, bundle, "event_arb_hft")
    
    return abs(sentiment) > threshold

def value_at_risk_optimization_logic(bundle: Dict[str, Any]) -> bool:
    """VaR Optimization: decent pattern confidence"""
    if not is_valid(bundle): 
        return False
    confidence = get_pattern_confidence(bundle)
    
    base_threshold = 0.6
    threshold = get_dynamic_threshold(base_threshold, bundle, "value_at_risk_optimization")
    
    return confidence > threshold

def dynamic_hedging_logic(bundle: Dict[str, Any]) -> bool:
    """Dynamic Hedging: slight bearish bias"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return -0.03 < slope < 0

def synthetic_stock_strategy_logic(bundle: Dict[str, Any]) -> bool:
    """Synthetic Stock Strategy: any pattern detected"""
    if not is_valid(bundle): 
        return False
    pattern = get_safe(bundle, ['pattern_signal', 'pattern'])
    
    return pattern is not None

def calendarized_condor_logic(bundle: Dict[str, Any]) -> bool:
    """Calendarized Condor: above average pattern confidence"""
    if not is_valid(bundle): 
        return False
    confidence = get_pattern_confidence(bundle)
    
    base_threshold = 0.65
    threshold = get_dynamic_threshold(base_threshold, bundle, "calendarized_condor")
    
    return confidence > threshold

def iron_condor_adjustment_logic(bundle: Dict[str, Any]) -> bool:
    """Iron Condor Adjustment: RSI above 50"""
    if not is_valid(bundle): 
        return False
    rsi = get_rsi(bundle)
    
    return rsi is not None and rsi > 50

def vega_hedging_logic(bundle: Dict[str, Any]) -> bool:
    """Vega Hedging: significant sentiment in either direction"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.4
    threshold = get_dynamic_threshold(base_threshold, bundle, "vega_hedging")
    
    return abs(sentiment) > threshold

def theta_harvest_logic(bundle: Dict[str, Any]) -> bool:
    """Theta Harvest: flat market expected"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return abs(slope) < 0.01

def back_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Back Spread: slight bearish forecast"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return slope < -0.02 and slope > -0.05

def front_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Front Spread: slight bullish forecast"""
    if not is_valid(bundle): 
        return False
    slope = get_safe(bundle, ['forecast', 'slope'], 0)
    
    return slope > 0.02 and slope < 0.05

def skew_spread_logic(bundle: Dict[str, Any]) -> bool:
    """Skew Spread: moderate sentiment in either direction"""
    if not is_valid(bundle): 
        return False
    sentiment = get_safe(bundle, ['sentiment', 'sentiment_score'], 0)
    
    base_threshold = 0.25
    threshold = get_dynamic_threshold(base_threshold, bundle, "skew_spread")
    
    return abs(sentiment) > threshold