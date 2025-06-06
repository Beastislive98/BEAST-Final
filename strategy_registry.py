# strategy_registry.py - Enhanced with Crypto-Specific Strategies

from strategy_rules import *

# Enhanced Registry with Crypto-Specific Strategies and Performance Tracking
# New format includes performance metrics and regime preferences

STRATEGY_REGISTRY = [
    # === NEW CRYPTO-SPECIFIC STRATEGIES ===
    {
        "name": "funding_rate_arbitrage",
        "type": "crypto_arbitrage",
        "risk_profile": "moderate",
        "trade_type": "arbitrage",
        "min_confidence": 0.6,
        "logic": funding_rate_arbitrage_logic,
        "preferred_regimes": ["any"],
        "crypto_native": True,
        "requires_funding_data": True
    },
    {
        "name": "perpetual_futures_momentum",
        "type": "crypto_momentum",
        "risk_profile": "aggressive",
        "trade_type": "swing",
        "min_confidence": 0.7,
        "logic": perpetual_futures_momentum_logic,
        "preferred_regimes": ["bull_trend", "high_volatility"],
        "crypto_native": True,
        "requires_oi_data": True
    },
    {
        "name": "cross_exchange_arbitrage",
        "type": "crypto_arbitrage",
        "risk_profile": "conservative",
        "trade_type": "arbitrage",
        "min_confidence": 0.8,
        "logic": cross_exchange_arbitrage_logic,
        "preferred_regimes": ["any"],
        "crypto_native": True,
        "requires_multi_exchange": True
    },
    {
        "name": "defi_yield_strategy",
        "type": "crypto_defi",
        "risk_profile": "moderate",
        "trade_type": "yield",
        "min_confidence": 0.65,
        "logic": defi_yield_strategy_logic,
        "preferred_regimes": ["bull_trend"],
        "crypto_native": True,
        "token_specific": True
    },
    {
        "name": "spot_futures_arbitrage",
        "type": "crypto_arbitrage",
        "risk_profile": "conservative",
        "trade_type": "arbitrage",
        "min_confidence": 0.75,
        "logic": spot_futures_arbitrage_logic,
        "preferred_regimes": ["any"],
        "crypto_native": True,
        "requires_spot_data": True
    },
    {
        "name": "crypto_momentum_breakout",
        "type": "crypto_momentum",
        "risk_profile": "aggressive",
        "trade_type": "breakout",
        "min_confidence": 0.8,
        "logic": crypto_momentum_breakout_logic,
        "preferred_regimes": ["bull_trend", "high_volatility"],
        "crypto_native": True,
        "requires_whale_data": True
    },
    {
        "name": "altcoin_rotation",
        "type": "crypto_rotation",
        "risk_profile": "moderate",
        "trade_type": "swing",
        "min_confidence": 0.65,
        "logic": altcoin_rotation_logic,
        "preferred_regimes": ["bull_trend"],
        "crypto_native": True,
        "exclude_majors": True
    },
    {
        "name": "adaptive_momentum",
        "type": "ai_adaptive",
        "risk_profile": "moderate",
        "trade_type": "swing",
        "min_confidence": 0.7,
        "logic": adaptive_momentum_logic,
        "preferred_regimes": ["any"],
        "self_optimizing": True
    },
    
    # === ENHANCED TRADITIONAL STRATEGIES ===
    # (Existing strategies with enhanced logic)
    
    # === Options Strategies ===
    {
        "name": "long_call", 
        "type": "options", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.65, 
        "logic": long_call_logic,
        "preferred_regimes": ["bull_trend", "ranging"],
        "regime_aware": True
    },
    {
        "name": "long_put", 
        "type": "options", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.65, 
        "logic": long_put_logic,
        "preferred_regimes": ["bear_trend", "ranging"]
    },
    {
        "name": "short_call", 
        "type": "options", 
        "risk_profile": "moderate", 
        "trade_type": "income", 
        "min_confidence": 0.65, 
        "logic": short_call_logic,
        "preferred_regimes": ["bear_trend", "ranging"]
    },
    {
        "name": "short_put", 
        "type": "options", 
        "risk_profile": "moderate", 
        "trade_type": "income", 
        "min_confidence": 0.65, 
        "logic": short_put_logic,
        "preferred_regimes": ["bull_trend", "ranging"]
    },
    
    # === Spread Strategies ===
    {
        "name": "bull_call_spread", 
        "type": "spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": bull_call_spread_logic,
        "preferred_regimes": ["bull_trend"]
    },
    {
        "name": "bear_put_spread", 
        "type": "spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": bear_put_spread_logic,
        "preferred_regimes": ["bear_trend"]
    },
    {
        "name": "debit_call_spread", 
        "type": "spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": debit_call_spread_logic,
        "preferred_regimes": ["bull_trend"]
    },
    {
        "name": "debit_put_spread", 
        "type": "spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": debit_put_spread_logic,
        "preferred_regimes": ["bear_trend"]
    },
    {
        "name": "ratio_call_backspread", 
        "type": "spread", 
        "risk_profile": "aggressive", 
        "trade_type": "swing", 
        "min_confidence": 0.75, 
        "logic": ratio_call_backspread_logic,
        "preferred_regimes": ["bull_trend", "high_volatility"]
    },
    {
        "name": "ratio_put_backspread", 
        "type": "spread", 
        "risk_profile": "aggressive", 
        "trade_type": "swing", 
        "min_confidence": 0.75, 
        "logic": ratio_put_backspread_logic,
        "preferred_regimes": ["bear_trend", "high_volatility"]
    },
    {
        "name": "front_spread", 
        "type": "spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": front_spread_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "back_spread", 
        "type": "spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": back_spread_logic,
        "preferred_regimes": ["ranging"]
    },
    
    # === Credit Spread Strategies ===
    {
        "name": "bear_call_spread", 
        "type": "credit_spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": bear_call_spread_logic,
        "preferred_regimes": ["bear_trend", "ranging"]
    },
    {
        "name": "bull_put_spread", 
        "type": "credit_spread", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": bull_put_spread_logic,
        "preferred_regimes": ["bull_trend", "ranging"]
    },
    {
        "name": "credit_call_spread", 
        "type": "credit", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": credit_call_spread_logic,
        "preferred_regimes": ["bear_trend", "ranging"]
    },
    {
        "name": "credit_put_spread", 
        "type": "credit", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": credit_put_spread_logic,
        "preferred_regimes": ["bull_trend", "ranging"]
    },
    
    # === Enhanced Volatility Strategies ===
    {
        "name": "long_straddle", 
        "type": "volatility", 
        "risk_profile": "aggressive", 
        "trade_type": "swing", 
        "min_confidence": 0.75, 
        "logic": long_straddle_logic,
        "preferred_regimes": ["high_volatility"],
        "regime_aware": True
    },
    {
        "name": "long_strangle", 
        "type": "volatility", 
        "risk_profile": "aggressive", 
        "trade_type": "swing", 
        "min_confidence": 0.75, 
        "logic": long_strangle_logic,
        "preferred_regimes": ["high_volatility"]
    },
    {
        "name": "short_straddle", 
        "type": "volatility", 
        "risk_profile": "aggressive", 
        "trade_type": "income", 
        "min_confidence": 0.75, 
        "logic": short_straddle_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "short_strangle", 
        "type": "volatility", 
        "risk_profile": "aggressive", 
        "trade_type": "income", 
        "min_confidence": 0.75, 
        "logic": short_strangle_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "skew_spread", 
        "type": "volatility", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": skew_spread_logic,
        "preferred_regimes": ["any"]
    },
    
    # === Complex Option Structures ===
    {
        "name": "long_call_butterfly", 
        "type": "butterfly", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": long_call_butterfly_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "long_put_butterfly", 
        "type": "butterfly", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": long_put_butterfly_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "long_call_condor", 
        "type": "condor", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": long_call_condor_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "long_put_condor", 
        "type": "condor", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": long_put_condor_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "iron_butterfly", 
        "type": "iron", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": iron_butterfly_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "iron_condor", 
        "type": "iron", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": iron_condor_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "iron_condor_adjustment", 
        "type": "iron", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": iron_condor_adjustment_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "jade_lizard", 
        "type": "complex", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": jade_lizard_logic,
        "preferred_regimes": ["bull_trend", "ranging"]
    },
    {
        "name": "calendarized_condor", 
        "type": "calendar", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": calendarized_condor_logic,
        "preferred_regimes": ["ranging"]
    },
    
    # === Calendar and Diagonal Strategies ===
    {
        "name": "calendar_call", 
        "type": "calendar", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": calendar_call_logic,
        "preferred_regimes": ["ranging", "bull_trend"]
    },
    {
        "name": "calendar_put", 
        "type": "calendar", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": calendar_put_logic,
        "preferred_regimes": ["ranging", "bear_trend"]
    },
    {
        "name": "diagonal_call", 
        "type": "diagonal", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": diagonal_call_logic,
        "preferred_regimes": ["bull_trend"]
    },
    {
        "name": "diagonal_put", 
        "type": "diagonal", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": diagonal_put_logic,
        "preferred_regimes": ["bear_trend"]
    },
    
    # === Exotic Strategies ===
    {
        "name": "exotic_option_combo", 
        "type": "exotic", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.7, 
        "logic": exotic_option_combo_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "barrier_option_strategy", 
        "type": "exotic", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.7, 
        "logic": barrier_option_strategy_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "digital_option_strategy", 
        "type": "exotic", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.7, 
        "logic": digital_option_strategy_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "lookback_option_strategy", 
        "type": "exotic", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.7, 
        "logic": lookback_option_strategy_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "asian_option_strategy", 
        "type": "exotic", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.7, 
        "logic": asian_option_strategy_logic,
        "preferred_regimes": ["any"]
    },
    
    # === Hedge Strategies ===
    {
        "name": "collar_protective", 
        "type": "hedge", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": collar_protective_logic,
        "preferred_regimes": ["bear_trend", "high_volatility"]
    },
    {
        "name": "delta_hedge", 
        "type": "hedge", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.6, 
        "logic": delta_hedge_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "vega_hedging", 
        "type": "hedge", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": vega_hedging_logic,
        "preferred_regimes": ["high_volatility"]
    },
    {
        "name": "dynamic_hedging", 
        "type": "hedge", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": dynamic_hedging_logic,
        "preferred_regimes": ["any"]
    },
    
    # === Income Strategies ===
    {
        "name": "theta_harvest", 
        "type": "income", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": theta_harvest_logic,
        "preferred_regimes": ["ranging"]
    },
    
    # === Enhanced HFT/Nano Trading Strategies ===
    {
        "name": "gamma_scalping", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.8, 
        "logic": gamma_scalping_logic,
        "preferred_regimes": ["high_volatility"],
        "requires_order_flow": True,
        "regime_aware": True
    },
    {
        "name": "latency_arbitrage", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.78, 
        "logic": latency_arbitrage_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "stat_arbitrage_hft", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.78, 
        "logic": stat_arbitrage_hft_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "rebate_arbitrage", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.78, 
        "logic": rebate_arbitrage_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "market_making_hft", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.78, 
        "logic": market_making_hft_logic,
        "preferred_regimes": ["ranging"]
    },
    {
        "name": "order_flow_strategy", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.78, 
        "logic": order_flow_strategy_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "tick_arbitrage", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.78, 
        "logic": tick_arbitrage_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "event_arb_hft", 
        "type": "hft", 
        "risk_profile": "aggressive", 
        "trade_type": "nano", 
        "min_confidence": 0.78, 
        "logic": event_arb_hft_logic,
        "preferred_regimes": ["high_volatility"]
    },
    
    # === Event-Driven Strategies ===
    {
        "name": "event_sniper", 
        "type": "event", 
        "risk_profile": "aggressive", 
        "trade_type": "sniper", 
        "min_confidence": 0.85, 
        "logic": event_sniper_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "earnings_play", 
        "type": "event", 
        "risk_profile": "aggressive", 
        "trade_type": "sniper", 
        "min_confidence": 0.85, 
        "logic": earnings_play_logic,
        "preferred_regimes": ["high_volatility"]
    },
    
    # === Enhanced Quantitative Strategies ===
    {
        "name": "statistical_arb", 
        "type": "quant", 
        "risk_profile": "moderate", 
        "trade_type": "quant_arb", 
        "min_confidence": 0.7, 
        "logic": statistical_arb_logic,
        "preferred_regimes": ["ranging"],
        "regime_aware": True
    },
    {
        "name": "index_arbitrage", 
        "type": "quant", 
        "risk_profile": "moderate", 
        "trade_type": "quant_arb", 
        "min_confidence": 0.75, 
        "logic": index_arbitrage_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "volatility_arb", 
        "type": "arbitrage", 
        "risk_profile": "moderate", 
        "trade_type": "quant_arb", 
        "min_confidence": 0.75, 
        "logic": volatility_arb_logic,
        "preferred_regimes": ["any"],
        "regime_aware": True
    },
    {
        "name": "box_spread", 
        "type": "arbitrage", 
        "risk_profile": "moderate", 
        "trade_type": "quant_arb", 
        "min_confidence": 0.7, 
        "logic": box_spread_logic,
        "preferred_regimes": ["ranging"]
    },
    
    # === AI/ML-Based Strategies ===
    {
        "name": "machine_learning_options", 
        "type": "ai", 
        "risk_profile": "aggressive", 
        "trade_type": "sniper", 
        "min_confidence": 0.85, 
        "logic": machine_learning_options_logic,
        "preferred_regimes": ["any"]
    },
    
    # === Risk Management Strategies ===
    {
        "name": "value_at_risk_optimization", 
        "type": "risk", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": value_at_risk_optimization_logic,
        "preferred_regimes": ["any"]
    },
    
    # === Other Strategies ===
    {
        "name": "synthetic_stock_strategy", 
        "type": "synthetic", 
        "risk_profile": "moderate", 
        "trade_type": "hedge", 
        "min_confidence": 0.7, 
        "logic": synthetic_stock_strategy_logic,
        "preferred_regimes": ["any"]
    },
    {
        "name": "ladder_call", 
        "type": "ladder", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.7, 
        "logic": ladder_call_logic,
        "preferred_regimes": ["bull_trend"]
    },
    {
        "name": "ladder_put", 
        "type": "ladder", 
        "risk_profile": "moderate", 
        "trade_type": "swing", 
        "min_confidence": 0.7, 
        "logic": ladder_put_logic,
        "preferred_regimes": ["bear_trend"]
    }
]

# === STRATEGY SELECTION HELPERS ===

def get_strategies_by_regime(regime_type: str) -> list:
    """Get strategies that prefer a specific market regime"""
    return [
        strategy for strategy in STRATEGY_REGISTRY 
        if regime_type in strategy.get("preferred_regimes", ["any"]) or 
           "any" in strategy.get("preferred_regimes", ["any"])
    ]

def get_crypto_native_strategies() -> list:
    """Get strategies specifically designed for crypto markets"""
    return [
        strategy for strategy in STRATEGY_REGISTRY 
        if strategy.get("crypto_native", False)
    ]

def get_regime_aware_strategies() -> list:
    """Get strategies that adapt to market regimes"""
    return [
        strategy for strategy in STRATEGY_REGISTRY 
        if strategy.get("regime_aware", False)
    ]

def get_self_optimizing_strategies() -> list:
    """Get strategies that self-optimize based on performance"""
    return [
        strategy for strategy in STRATEGY_REGISTRY 
        if strategy.get("self_optimizing", False)
    ]

def filter_strategies_by_requirements(bundle: dict) -> list:
    """Filter strategies based on available data in bundle"""
    available_strategies = []
    
    for strategy in STRATEGY_REGISTRY:
        # Check data requirements
        requirements_met = True
        
        if strategy.get("requires_funding_data", False):
            if not bundle.get("market_data", {}).get("funding_rate"):
                requirements_met = False
        
        if strategy.get("requires_oi_data", False):
            if not bundle.get("market_data", {}).get("oi_change_pct"):
                requirements_met = False
        
        if strategy.get("requires_multi_exchange", False):
            if not bundle.get("market_data", {}).get("cross_exchange_spread"):
                requirements_met = False
        
        if strategy.get("requires_spot_data", False):
            if not bundle.get("market_data", {}).get("spot_price"):
                requirements_met = False
        
        if strategy.get("requires_whale_data", False):
            if not bundle.get("whale_flags", {}).get("whale_present"):
                requirements_met = False
        
        if strategy.get("requires_order_flow", False):
            if not bundle.get("order_flow", {}).get("pressure_strength"):
                requirements_met = False
        
        # Token-specific filtering
        if strategy.get("token_specific", False):
            symbol = bundle.get("symbol", "")
            defi_tokens = ["UNI", "SUSHI", "COMP", "AAVE", "MKR", "CRV", "1INCH"]
            if not any(token in symbol for token in defi_tokens):
                requirements_met = False
        
        # Exclude majors filtering
        if strategy.get("exclude_majors", False):
            symbol = bundle.get("symbol", "")
            major_coins = ["BTC", "ETH"]
            if any(coin in symbol for coin in major_coins):
                requirements_met = False
        
        if requirements_met:
            available_strategies.append(strategy)
    
    return available_strategies

# === STRATEGY PERFORMANCE TRACKING ===

STRATEGY_METRICS = {
    "funding_rate_arbitrage": {"expected_win_rate": 0.75, "expected_return": 0.008},
    "perpetual_futures_momentum": {"expected_win_rate": 0.65, "expected_return": 0.025},
    "cross_exchange_arbitrage": {"expected_win_rate": 0.85, "expected_return": 0.005},
    "crypto_momentum_breakout": {"expected_win_rate": 0.70, "expected_return": 0.035},
    "altcoin_rotation": {"expected_win_rate": 0.60, "expected_return": 0.020},
    # Add more as strategies are tested and optimized
}

def get_strategy_expected_performance(strategy_name: str) -> dict:
    """Get expected performance metrics for a strategy"""
    return STRATEGY_METRICS.get(strategy_name, {"expected_win_rate": 0.60, "expected_return": 0.015})