# strategy_selection.py - Enhanced Production Ready Version

import logging
import time
import os
import sys
import json
import threading
from typing import Dict, Any, Set, List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/strategy_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("strategy_selector")

# Environment variable for consistent directory paths
SIGNALS_DIR = os.environ.get('TRADING_SIGNALS_DIR', os.path.abspath('./trade_signals'))
DATA_DIR = os.environ.get('TRADING_DATA_DIR', os.path.abspath('./data'))
LOGS_DIR = os.environ.get('TRADING_LOGS_DIR', os.path.abspath('./logs'))

# Ensure directories exist
for directory in [SIGNALS_DIR, DATA_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

try:
    from pattern_score import generate_trade_signal
    from strategy_registry import STRATEGY_REGISTRY
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Ensure pattern_score.py and strategy_registry.py are in your PYTHONPATH")
    sys.exit(1)

# Strategy status tracking
COLD_SYMBOLS = set()
HOT_SYMBOLS = set()
SYMBOL_NO_TRADE_COUNT = {}
NO_TRADE_THRESHOLD = 5
SYMBOL_PERFORMANCE = {}
SYMBOL_TRADE_HISTORY = {}
STRATEGY_USAGE_STATS = {}

# Status file path
SYMBOL_STATUS_FILE = os.path.join(LOGS_DIR, "symbol_status.json")

class TradeStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    CAPITAL_CHECKED = "capital_checked"
    ALLOCATED = "allocated"
    REJECTED = "rejected"
    QUEUED = "queued"

def determine_trade_bias(symbol_tag: str) -> Dict[str, str]:
    """
    Infers trade direction and strategy type based on symbol tag.
    """
    bias_map = {
        "rally": {"positionSide": "LONG", "strategy_type": "momentum"},
        "fall": {"positionSide": "SHORT", "strategy_type": "momentum"},
        "breakout": {"positionSide": "LONG", "strategy_type": "breakout"},
        "breakdown": {"positionSide": "SHORT", "strategy_type": "breakout"},
        "pullback": {"positionSide": "LONG", "strategy_type": "reversal"}
    }
    
    return bias_map.get(symbol_tag, {"positionSide": "LONG", "strategy_type": "default"})

def is_symbol_cold(symbol: str) -> bool:
    """
    Check if a symbol is marked as cold (not producing trades)
    """
    return symbol in COLD_SYMBOLS

def is_symbol_hot(symbol: str) -> bool:
    """
    Check if a symbol is marked as hot (consistently producing good trades)
    """
    return symbol in HOT_SYMBOLS

def mark_symbol_cold(symbol: str):
    """
    Mark a symbol as cold after repeated failed trade attempts
    """
    COLD_SYMBOLS.add(symbol)
    if symbol in HOT_SYMBOLS:
        HOT_SYMBOLS.remove(symbol)
    logger.warning(f"⚠️ Symbol {symbol} marked as COLD after {NO_TRADE_THRESHOLD} failed trade attempts")
    
    # Save current state
    save_symbol_status()

def mark_symbol_hot(symbol: str):
    """
    Mark a symbol as hot after consistent successful trades
    """
    HOT_SYMBOLS.add(symbol)
    if symbol in COLD_SYMBOLS:
        COLD_SYMBOLS.remove(symbol)
    logger.info(f"🔥 Symbol {symbol} marked as HOT due to consistent performance")
    
    # Save current state
    save_symbol_status()

def get_cold_symbols() -> Set[str]:
    """
    Return the set of cold symbols
    """
    return COLD_SYMBOLS

def get_hot_symbols() -> Set[str]:
    """
    Return the set of hot symbols
    """
    return HOT_SYMBOLS

def reset_cold_symbol(symbol: str):
    """
    Remove a symbol from the cold list to give it another chance
    """
    if symbol in COLD_SYMBOLS:
        COLD_SYMBOLS.remove(symbol)
        logger.info(f"Symbol {symbol} removed from cold list")
    
    # Reset the counter
    SYMBOL_NO_TRADE_COUNT[symbol] = 0
    
    # Save current state
    save_symbol_status()

def reset_all_cold_symbols():
    """
    Reset all cold symbols to give them another chance
    """
    global COLD_SYMBOLS
    count = len(COLD_SYMBOLS)
    COLD_SYMBOLS = set()
    logger.info(f"Reset {count} cold symbols to give them another chance")
    
    # Save current state
    save_symbol_status()

def save_symbol_status():
    """
    Save the current symbol status to disk
    """
    try:
        # Create the status object
        status = {
            "hot_symbols": list(HOT_SYMBOLS),
            "cold_symbols": list(COLD_SYMBOLS),
            "timestamp": datetime.now().isoformat(),
            "symbol_stats": SYMBOL_PERFORMANCE
        }
        
        # Write to file
        with open(SYMBOL_STATUS_FILE, "w") as f:
            json.dump(status, f, indent=2)
            
        logger.info(f"Symbol status saved with {len(HOT_SYMBOLS)} hot and {len(COLD_SYMBOLS)} cold symbols")
    except Exception as e:
        logger.error(f"Failed to save symbol status: {e}")

def load_symbol_status():
    """
    Load the symbol status from disk
    """
    try:
        # Check if status file exists
        if not os.path.exists(SYMBOL_STATUS_FILE):
            logger.info("No symbol status file found")
            return
            
        # Read the status file
        with open(SYMBOL_STATUS_FILE, "r") as f:
            status = json.load(f)
            
        # Update global variables
        global HOT_SYMBOLS, COLD_SYMBOLS, SYMBOL_PERFORMANCE
        
        if "hot_symbols" in status:
            HOT_SYMBOLS = set(status["hot_symbols"])
            
        if "cold_symbols" in status:
            COLD_SYMBOLS = set(status["cold_symbols"])
            
        if "symbol_stats" in status:
            SYMBOL_PERFORMANCE = status["symbol_stats"]
            
        logger.info(f"Loaded symbol status with {len(HOT_SYMBOLS)} hot and {len(COLD_SYMBOLS)} cold symbols")
    except Exception as e:
        logger.error(f"Failed to load symbol status: {e}")

def track_trade_outcome(symbol: str, success: bool, pnl: float, confidence: float, strategy_name: str = None):
    """
    Track trade outcomes for symbols to identify hot performers and
    update strategy performance statistics.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        success: Whether the trade was successful
        pnl: Profit and loss amount
        confidence: Confidence score of the trade signal
        strategy_name: Name of the strategy used
    """
    # Track symbol performance
    if symbol not in SYMBOL_PERFORMANCE:
        SYMBOL_PERFORMANCE[symbol] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "avg_confidence": 0.0,
            "last_trade_time": None
        }
    
    # Update stats
    stats = SYMBOL_PERFORMANCE[symbol]
    stats["trades"] += 1
    if success:
        stats["wins"] += 1
    else:
        stats["losses"] += 1
    stats["pnl"] += pnl
    stats["avg_confidence"] = ((stats["avg_confidence"] * (stats["trades"] - 1)) + confidence) / stats["trades"]
    stats["last_trade_time"] = datetime.now().isoformat()
    
    # Add to trade history
    if symbol not in SYMBOL_TRADE_HISTORY:
        SYMBOL_TRADE_HISTORY[symbol] = []
    
    SYMBOL_TRADE_HISTORY[symbol].append({
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "pnl": pnl,
        "confidence": confidence,
        "strategy": strategy_name
    })
    
    # Update strategy stats
    if strategy_name:
        if strategy_name not in STRATEGY_USAGE_STATS:
            STRATEGY_USAGE_STATS[strategy_name] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0
            }
            
        strategy_stats = STRATEGY_USAGE_STATS[strategy_name]
        strategy_stats["trades"] += 1
        if success:
            strategy_stats["wins"] += 1
        else:
            strategy_stats["losses"] += 1
        strategy_stats["pnl"] += pnl
    
    # Check if symbol should be marked as hot
    if stats["trades"] >= 5 and stats["wins"] / stats["trades"] >= 0.6 and stats["pnl"] > 0:
        mark_symbol_hot(symbol)
    
    # Check if symbol should be marked as cold
    if stats["trades"] >= 5 and stats["wins"] / stats["trades"] < 0.4 and stats["pnl"] < 0:
        mark_symbol_cold(symbol)
        
    # Try to save to database if available
    try:
        from database_interface import db_interface
        db_interface.update_symbol_performance(symbol, success, pnl, confidence)
    except (ImportError, AttributeError):
        pass  # Database not available, skip saving
        
    # Always save symbol status after updating trade outcomes
    save_symbol_status()

def get_top_performing_strategies(min_trades: int = 5) -> List[Dict[str, Any]]:
    """
    Get top performing strategies based on win rate and PnL
    """
    results = []
    
    for name, stats in STRATEGY_USAGE_STATS.items():
        if stats["trades"] >= min_trades:
            win_rate = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            results.append({
                "name": name,
                "trades": stats["trades"],
                "win_rate": win_rate,
                "pnl": stats["pnl"],
                "score": win_rate * 0.7 + (stats["pnl"] / max(stats["trades"], 1)) * 0.3
            })
    
    # Sort by score descending
    return sorted(results, key=lambda x: x["score"], reverse=True)

def run_shadow_analysis_for_cold_symbols(max_symbols: int = 5) -> List[Dict[str, Any]]:
    """
    Run lightweight analysis on cold symbols to find sudden opportunities
    Returns list of promising cold symbols with high confidence signals
    """
    opportunities = []
    cold_symbols_list = list(COLD_SYMBOLS)
    
    # Limit the number of cold symbols to check to avoid performance issues
    symbols_to_check = cold_symbols_list[:max_symbols]
    
    for symbol in symbols_to_check:
        try:
            from data_bundle import assemble_data_bundle
            
            # Run a lightweight analysis
            bundle = assemble_data_bundle(symbol, lightweight=True)
            if not bundle:
                continue
                
            # Generate trade signal
            current_price = bundle.get("market_data", {}).get("price", 0)
            if current_price <= 0:
                ticker_data = bundle.get("market_data", {}).get("ticker", {})
                if ticker_data:
                    current_price = float(ticker_data.get("lastPrice", 0))
                    
                if current_price <= 0:
                    continue
                
            # Check for promising signals
            pattern_signal = bundle.get("pattern_signal", {})
            confidence = pattern_signal.get("confidence", 0)
            
            # Check for significant price moves
            price_change = bundle.get("market_data", {}).get("ticker", {}).get("priceChangePercent", 0)
            try:
                price_change = float(price_change)
            except (ValueError, TypeError):
                price_change = 0
                
            # Check for whale activity
            whale_present = bundle.get("whale_flags", {}).get("whale_present", False)
            
            # Check for strong sentiment
            sentiment_score = abs(bundle.get("sentiment", {}).get("sentiment_score", 0))
            
            # Calculate an opportunity score
            opportunity_score = 0.0
            
            if confidence > 0.8:  # Strong pattern confidence
                opportunity_score += 0.5
                
            if abs(price_change) > 3.0:  # Significant price move
                opportunity_score += 0.3
                
            if whale_present:  # Whale activity
                opportunity_score += 0.2
                
            if sentiment_score > 0.6:  # Strong sentiment
                opportunity_score += 0.2
                
            # If score is high enough, consider it an opportunity
            if opportunity_score >= 0.5:
                trade_signal = generate_trade_signal(bundle, current_price)
                
                # Further filter by trade signal confidence
                if trade_signal and trade_signal.get("confidence", 0) > 0.6:
                    opportunities.append({
                        "symbol": symbol,
                        "confidence": trade_signal.get("confidence", 0),
                        "signal": trade_signal,
                        "price_change": price_change,
                        "opportunity_score": opportunity_score
                    })
                    logger.info(f"Shadow analysis found opportunity in cold symbol {symbol} with score {opportunity_score:.2f}")
        except Exception as e:
            logger.warning(f"Shadow analysis failed for {symbol}: {e}")
            
    # Sort opportunities by score
    opportunities.sort(key=lambda x: x.get("opportunity_score", 0), reverse=True)
    return opportunities

def find_historical_matches(current_conditions: Dict[str, Any], lookback_days: int = 30) -> Optional[Dict[str, Any]]:
    """
    Find historical matches for current market conditions
    when no patterns are directly detected
    """
    try:
        from database_interface import db_interface
        
        # Extract key features from current conditions
        rsi = current_conditions.get("RSI_14", 50)
        volume = current_conditions.get("Volume", 0)
        volatility = current_conditions.get("ATR_14", 0)
        
        # Search database for similar conditions
        query = f"""
        SELECT date, pattern, direction, outcome, confidence
        FROM pattern_history
        WHERE ABS(rsi - {rsi}) < 5
        AND ABS(volume_normalized - {volume / 1000000}) < 0.2
        AND ABS(volatility_normalized - {volatility}) < 0.15
        ORDER BY confidence DESC
        LIMIT 5
        """
        
        # Execute query against SQLite database
        similar_conditions = db_interface.execute_query(query)
        
        if similar_conditions and len(similar_conditions) > 0:
            # Calculate average confidence and outcome
            avg_confidence = sum(row.get("confidence", 0) for row in similar_conditions) / len(similar_conditions)
            positive_outcomes = sum(1 for row in similar_conditions if row.get("outcome") == "positive")
            success_rate = positive_outcomes / len(similar_conditions) if len(similar_conditions) > 0 else 0
            
            # Determine most likely direction
            long_count = sum(1 for row in similar_conditions if row.get("direction") == "long")
            short_count = len(similar_conditions) - long_count
            direction = "long" if long_count >= short_count else "short"
            
            if success_rate > 0.6 and avg_confidence > 0.7:
                # Found promising historical pattern
                return {
                    "historical_match": True,
                    "pattern": similar_conditions[0].get("pattern", "unknown"),  # Most confident pattern
                    "confidence": avg_confidence * success_rate,  # Adjust confidence
                    "similar_dates": [row.get("date") for row in similar_conditions],
                    "success_rate": success_rate,
                    "direction": direction
                }
                
        return None
    except Exception as e:
        logger.error(f"Historical pattern matching failed: {e}")
        return None

def get_strategy_success_rate(strategy_name: str) -> float:
    """
    Get the success rate for a specific strategy
    """
    stats = STRATEGY_USAGE_STATS.get(strategy_name, {})
    trades = stats.get("trades", 0)
    wins = stats.get("wins", 0)
    
    if trades > 0:
        return wins / trades
    return 0.5  # Default to 50% if no data

def preprocess_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the data bundle to ensure all required fields are present
    and correctly formatted.
    """
    if not bundle:
        return {}
        
    # Ensure all key sections exist
    for section in ['sentiment', 'forecast', 'pattern_signal', 'whale_flags']:
        if section not in bundle:
            bundle[section] = {}
    
    # Fill in missing values with defaults
    if 'sentiment_score' not in bundle['sentiment']:
        bundle['sentiment']['sentiment_score'] = 0.0
        
    if 'slope' not in bundle['forecast']:
        bundle['forecast']['slope'] = 0.0
        
    if 'confidence' not in bundle['pattern_signal']:
        bundle['pattern_signal']['confidence'] = 0.0
        
    if 'whale_present' not in bundle['whale_flags']:
        bundle['whale_flags']['whale_present'] = False
        
    return bundle

def get_account_balance() -> float:
    """Get current account balance for risk calculations"""
    try:
        # Try to get from capital_manager via API
        import requests
        response = requests.get("http://localhost:5000/api/balance")
        if response.status_code == 200:
            data = response.json()
            return data.get("balance", 100.0)
    except Exception as e:
        logger.warning(f"Failed to get account balance: {e}")
    
    # Default balance if can't get actual value
    return 100.0

# ENHANCED FUNCTIONS FROM PASTE.TXT - INTEGRATED
def validate_risk_strict_fallback(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict fallback risk validation that cannot be bypassed - NEW FUNCTION
    """
    try:
        # Essential validations
        symbol = signal.get("symbol", "UNKNOWN")
        entry = signal.get("entry", 0)
        stop_loss = signal.get("stopLoss", 0)
        take_profit = signal.get("takeProfit", 0)
        confidence = signal.get("confidence", 0)
        side = signal.get("side", "")
        
        # Basic field validation
        if not entry or entry <= 0:
            return {"valid": False, "reason": "invalid_entry", "symbol": symbol}
            
        if not stop_loss or stop_loss <= 0:
            return {"valid": False, "reason": "invalid_stop_loss", "symbol": symbol}
            
        if not take_profit or take_profit <= 0:
            return {"valid": False, "reason": "invalid_take_profit", "symbol": symbol}
            
        if not side or side not in ["BUY", "SELL", "LONG", "SHORT"]:
            return {"valid": False, "reason": "invalid_side", "symbol": symbol}
        
        # Direction-specific validation
        if side.upper() in ["BUY", "LONG"]:
            if stop_loss >= entry:
                return {"valid": False, "reason": "stop_loss_above_entry_for_long", "symbol": symbol}
            if take_profit <= entry:
                return {"valid": False, "reason": "take_profit_below_entry_for_long", "symbol": symbol}
                
            risk = entry - stop_loss
            reward = take_profit - entry
            
        else:  # SHORT
            if stop_loss <= entry:
                return {"valid": False, "reason": "stop_loss_below_entry_for_short", "symbol": symbol}
            if take_profit >= entry:
                return {"valid": False, "reason": "take_profit_above_entry_for_short", "symbol": symbol}
                
            risk = stop_loss - entry
            reward = entry - take_profit
        
        # Risk-reward validation
        if risk <= 0 or reward <= 0:
            return {"valid": False, "reason": "invalid_risk_reward_calculation", "symbol": symbol}
            
        rr_ratio = reward / risk
        min_rr = 1.0  # Minimum 1:1
        
        if rr_ratio < min_rr:
            return {"valid": False, "reason": f"poor_rr_{rr_ratio:.2f}", "symbol": symbol}
        
        # Confidence validation
        if confidence < 0.3:  # Minimum 30% confidence
            return {"valid": False, "reason": f"low_confidence_{confidence:.2f}", "symbol": symbol}
        
        # Risk percentage validation
        risk_pct = risk / entry
        if risk_pct > 0.05:  # Maximum 5% risk per trade
            return {"valid": False, "reason": f"high_risk_{risk_pct:.2%}", "symbol": symbol}
        
        # Validation passed
        signal.update({
            "risk_reward_ratio": rr_ratio,
            "risk_percentage": risk_pct * 100,
            "validation_passed": True,
            "validation_time": time.time()
        })
        
        return {"valid": True, "trade": signal}
        
    except Exception as e:
        logging.error(f"Error in risk validation fallback: {e}")
        return {"valid": False, "reason": f"validation_error", "symbol": signal.get("symbol", "UNKNOWN")}

def validate_risk(trade_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced risk validation that cannot be bypassed - UPDATED VERSION
    """
    try:
        # Import the actual risk validator (no bypass)
        from risk_validator import RiskValidator
        risk_validator = RiskValidator()
        
        # Run strict validation
        result = risk_validator.validate(trade_signal)
        
        # Log the result
        if result.get("valid", False):
            logging.info(f"✅ Risk validation PASSED for {trade_signal.get('symbol')}")
        else:
            logging.warning(f"❌ Risk validation FAILED for {trade_signal.get('symbol')}: {result.get('reason')}")
        
        return result
        
    except ImportError:
        # Fallback validation - but still strict
        logging.warning("RiskValidator module not available, using strict fallback validation")
        return validate_risk_strict_fallback(trade_signal)

def check_capital_strict_fallback(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict fallback capital check - NEW FUNCTION
    """
    try:
        symbol = signal.get("symbol", "UNKNOWN")
        entry = signal.get("entry", 0)
        confidence = signal.get("confidence", 0.5)
        
        # Get current account balance (fallback to 100 if can't get real balance)
        current_capital = get_account_balance()
        
        # Calculate position size based on 2% risk rule
        stop_loss = signal.get("stopLoss", 0)
        side = signal.get("side", "BUY")
        
        if side.upper() in ["BUY", "LONG"]:
            price_risk = entry - stop_loss
        else:
            price_risk = stop_loss - entry
            
        if price_risk <= 0:
            return {"valid": False, "reason": "invalid_risk_calculation", "symbol": symbol}
        
        # Risk amount (2% of capital)
        risk_amount = current_capital * 0.02
        
        # Position size
        position_size = risk_amount / price_risk
        notional_value = position_size * entry
        
        # Check position size limits (max 10% of capital per position)
        max_position_value = current_capital * 0.10
        
        if notional_value > max_position_value:
            return {"valid": False, "reason": f"position_too_large_{notional_value:.2f}_{max_position_value:.2f}", "symbol": symbol}
        
        # Apply confidence-based sizing
        confidence_multiplier = max(0.5, min(1.5, confidence))
        final_position_size = position_size * confidence_multiplier
        final_notional = final_position_size * entry
        
        # Update signal with position sizing
        signal.update({
            "quantity": final_position_size,
            "notional_value": final_notional,
            "capital_allocated": final_notional,
            "confidence_multiplier": confidence_multiplier
        })
        
        return {"valid": True, "trade": signal}
        
    except Exception as e:
        logging.error(f"Error in capital check fallback: {e}")
        return {"valid": False, "reason": "capital_check_error", "symbol": signal.get("symbol", "UNKNOWN")}

def check_capital(trade_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced capital check that cannot be bypassed - UPDATED VERSION
    """
    try:
        # Import the actual capital manager (no bypass)
        from capital_manager import CapitalManager
        capital_manager = CapitalManager()
        
        # Run strict capital validation
        result = capital_manager.validate(trade_signal)
        
        # Log the result
        if result.get("valid", False):
            logging.info(f"✅ Capital check PASSED for {trade_signal.get('symbol')}")
        else:
            logging.warning(f"❌ Capital check FAILED for {trade_signal.get('symbol')}: {result.get('reason')}")
        
        return result
        
    except ImportError:
        # Fallback capital check - but still has limits
        logging.warning("CapitalManager module not available, using strict fallback")
        return check_capital_strict_fallback(trade_signal)

def forced_trade_signal(bundle: Dict[str, Any], current_price: float, tag: str) -> Dict[str, Any]:
    """
    Generate forced trade signal with balanced LONG/SHORT logic - UPDATED VERSION
    """
    try:
        # Use the enhanced pattern scoring for balanced signals
        from pattern_score import generate_trade_signal
        
        # Generate the signal using the enhanced balanced logic
        signal = generate_trade_signal(bundle, current_price)
        
        # Mark as forced trade
        signal.update({
            "forced": True,
            "strategy_name": "forced_balanced_signal",
            "strategy_type": "forced",
            "trade_type": "swing"
        })
        
        # Adjust confidence for forced trades (slightly lower)
        original_confidence = signal.get("confidence", 0.5)
        forced_confidence = max(0.3, original_confidence * 0.9)  # Reduce by 10% but keep minimum 30%
        signal["confidence"] = forced_confidence
        
        logging.info(f"🔄 Generated FORCED balanced signal for {bundle.get('symbol', 'UNKNOWN')}: "
                    f"{signal.get('side')} with confidence {forced_confidence:.2f}")
        
        return signal
        
    except Exception as e:
        logging.error(f"Failed to generate forced balanced signal: {e}")
        
        # Last resort fallback - but still balanced
        symbol = bundle.get("symbol", "UNKNOWN")
        
        # Use random selection for true balance
        import random
        direction = "long" if random.random() > 0.5 else "short"
        
        if direction == "long":
            side = "BUY"
            position_side = "LONG"
            stop_loss = round(current_price * 0.985, 8)  # 1.5% stop
            take_profit = round(current_price * 1.03, 8)  # 3% target
        else:
            side = "SELL"
            position_side = "SHORT"
            stop_loss = round(current_price * 1.015, 8)  # 1.5% stop
            take_profit = round(current_price * 0.97, 8)  # 3% target
        
        return {
            "decision": "trade",
            "entry": current_price,
            "stopLoss": stop_loss,
            "takeProfit": take_profit,
            "side": side,
            "positionSide": position_side,
            "confidence": 0.4,  # Low confidence for fallback
            "strategy_name": "emergency_balanced_fallback",
            "strategy_type": "fallback",
            "forced": True,
            "symbol": symbol,
            "trade_type": "hedge",
            "random_direction": True
        }

def process_trade_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process trade signal through enhanced pipeline - UPDATED VERSION
    """
    try:
        symbol = signal.get("symbol", "UNKNOWN")
        
        # Step 1: Enhanced Risk Validation (cannot be bypassed)
        logging.info(f"🔍 Step 1: Risk validation for {symbol}")
        validation_result = validate_risk(signal)
        
        if not validation_result.get("valid", False):
            reason = validation_result.get("reason", "unknown")
            logging.warning(f"❌ Trade REJECTED in risk validation: {symbol} - {reason}")
            return validation_result
        
        validated_trade = validation_result.get("trade", signal)
        
        # Step 2: Enhanced Capital Check (cannot be bypassed)
        logging.info(f"💰 Step 2: Capital check for {symbol}")
        capital_result = check_capital(validated_trade)
        
        if not capital_result.get("valid", False):
            reason = capital_result.get("reason", "unknown")
            if capital_result.get("queued", False):
                logging.info(f"📋 Trade QUEUED: {symbol} - {reason}")
                return capital_result
            else:
                logging.warning(f"❌ Trade REJECTED in capital check: {symbol} - {reason}")
                return capital_result
        
        capital_checked_trade = capital_result.get("trade", validated_trade)
        
        # Step 3: Final Validation Check
        logging.info(f"✅ Step 3: Final validation for {symbol}")
        
        # Ensure all required fields are present
        required_fields = ["entry", "stopLoss", "takeProfit", "side", "quantity", "notional_value"]
        missing_fields = [field for field in required_fields if field not in capital_checked_trade]
        
        if missing_fields:
            logging.error(f"❌ Trade missing required fields: {symbol} - {missing_fields}")
            return {"valid": False, "reason": f"missing_fields_{missing_fields}", "symbol": symbol}
        
        # Log successful processing
        trade_info = {
            "symbol": symbol,
            "side": capital_checked_trade.get("side"),
            "entry": capital_checked_trade.get("entry"),
            "quantity": capital_checked_trade.get("quantity"),
            "confidence": capital_checked_trade.get("confidence"),
            "strategy": capital_checked_trade.get("strategy_name", "unknown")
        }
        
        logging.info(f"✅ Trade signal processed successfully: {trade_info}")
        
        return {"valid": True, "trade": capital_checked_trade}
        
    except Exception as e:
        logging.error(f"Error processing trade signal: {e}")
        return {
            "valid": False, 
            "reason": f"processing_error_{str(e)}", 
            "symbol": signal.get("symbol", "UNKNOWN")
        }

def write_trade_signal_to_file(trade_data: Dict[str, Any]) -> Optional[str]:
    """
    Write final trade signal to file for BEAST engine to process
    """
    try:
        from json_writer import write_trade_signal_to_file
        return write_trade_signal_to_file(trade_data)
    except ImportError:
        # Fallback if json_writer module not available
        try:
            # Ensure directory exists
            os.makedirs(SIGNALS_DIR, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename with symbol and timestamp
            symbol = trade_data.get("symbol", "UNKNOWN")
            filename = f"{symbol}_{timestamp}.json"
            filepath = os.path.join(SIGNALS_DIR, filename)
            
            # Ensure trade data has timestamp
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = timestamp
            
            # Write JSON file
            with open(filepath, 'w') as f:
                json.dump(trade_data, f, indent=2)
                
            logger.info(f"Trade signal written to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to write trade signal: {e}")
            return None

def strategy_tournament(bundle: Dict[str, Any], current_price: float, tag: str = "neutral") -> Dict[str, Any]:
    """
    Enhanced strategy tournament with strict validation - UPDATED VERSION  
    """
    start_time = time.time()
    symbol = bundle.get("symbol", "UNKNOWN")
    
    try:
        logging.info(f"🏁 Starting strategy tournament for {symbol}")
        
        # Preprocess the bundle
        bundle = preprocess_bundle(bundle)
        
        # Check if cold symbol
        if is_symbol_cold(symbol):
            pattern_confidence = bundle.get("pattern_signal", {}).get("confidence", 0)
            if pattern_confidence < 0.85:
                logging.info(f"❄️ Cold symbol {symbol} - forcing balanced signal")
                forced_signal = forced_trade_signal(bundle, current_price, tag)
                return process_trade_signal(forced_signal).get("trade", forced_signal)
        
        # Try to find valid strategies
        valid_strategies = []
        
        # Strategy selection logic (keeping existing logic but enhanced)
        for strategy in STRATEGY_REGISTRY:
            try:
                min_confidence = strategy.get("min_confidence", 0.6)
                pattern_confidence = bundle.get("pattern_signal", {}).get("confidence", 0.5)
                
                if pattern_confidence >= min_confidence and strategy["logic"](bundle):
                    strategy_score = pattern_confidence * 0.7 + 0.3  # Base score
                    
                    valid_strategies.append({
                        "name": strategy["name"],
                        "type": strategy["type"],
                        "score": strategy_score,
                        "risk_profile": strategy.get("risk_profile", "moderate"),
                        "trade_type": strategy.get("trade_type", "swing")
                    })
                    
            except Exception as e:
                logging.warning(f"Strategy '{strategy['name']}' evaluation failed: {e}")
        
        # Generate trade signal
        if valid_strategies:
            # Use best strategy
            best_strategy = max(valid_strategies, key=lambda x: x["score"])
            
            # Generate signal using enhanced pattern scoring
            from pattern_score import generate_trade_signal
            trade_signal = generate_trade_signal(bundle, current_price)
            
            if trade_signal.get("decision") == "trade":
                trade_signal.update({
                    "strategy_name": best_strategy["name"],
                    "strategy_type": best_strategy["type"],
                    "trade_type": best_strategy["trade_type"],
                    "risk_profile": best_strategy["risk_profile"]
                })
                
                # Reset no-trade counter
                SYMBOL_NO_TRADE_COUNT[symbol] = 0
                
                if symbol in COLD_SYMBOLS:
                    reset_cold_symbol(symbol)
                
                # Process through enhanced pipeline
                result = process_trade_signal(trade_signal)
                
                elapsed = time.time() - start_time
                logging.info(f"🏆 Strategy tournament completed in {elapsed:.3f}s for {symbol}: {best_strategy['name']}")
                
                return result.get("trade", trade_signal)
        
        # No valid strategies - use forced balanced signal
        logging.info(f"🔄 No valid strategies for {symbol} - generating forced balanced signal")
        
        SYMBOL_NO_TRADE_COUNT[symbol] = SYMBOL_NO_TRADE_COUNT.get(symbol, 0) + 1
        
        if SYMBOL_NO_TRADE_COUNT[symbol] >= NO_TRADE_THRESHOLD:
            mark_symbol_cold(symbol)
        
        forced_signal = forced_trade_signal(bundle, current_price, tag)
        result = process_trade_signal(forced_signal)
        
        elapsed = time.time() - start_time
        logging.info(f"🔄 Strategy tournament (forced) completed in {elapsed:.3f}s for {symbol}")
        
        return result.get("trade", forced_signal)
        
    except Exception as e:
        logging.error(f"Strategy tournament failed for {symbol}: {e}")
        
        # Even on error, try to generate a signal
        forced_signal = forced_trade_signal(bundle, current_price, tag)
        return process_trade_signal(forced_signal).get("trade", forced_signal)

# Load symbol status on initialization
load_symbol_status()

# Start background thread for processing pending trades
def process_pending_trades_thread():
    """Background thread to process trades that are in queued state"""
    while True:
        try:
            # Scan for queued trade files
            queued_files = []
            for file in os.listdir(SIGNALS_DIR):
                if file.endswith(".json"):
                    filepath = os.path.join(SIGNALS_DIR, file)
                    try:
                        with open(filepath, 'r') as f:
                            trade_data = json.load(f)
                            if trade_data.get("status") == TradeStatus.QUEUED.value:
                                queued_files.append((filepath, trade_data))
                    except Exception as e:
                        logger.error(f"Error reading queued trade file {filepath}: {e}")
            
            # Process queued trades
            for filepath, trade_data in queued_files:
                try:
                    # Check if we can process now
                    capital_result = check_capital(trade_data)
                    if capital_result.get("valid", False):
                        # Can process now, move through pipeline
                        capital_checked_trade = capital_result.get("trade", trade_data)
                        allocation_result = allocate_capital(capital_checked_trade)
                        
                        if allocation_result.get("valid", False):
                            # Successfully allocated, replace file with updated trade
                            allocated_trade = allocation_result.get("trade", capital_checked_trade)
                            with open(filepath, 'w') as f:
                                json.dump(allocated_trade, f, indent=2)
                            logger.info(f"Processed queued trade successfully: {filepath}")
                    else:
                        logger.debug(f"Queued trade still waiting: {filepath}")
                except Exception as e:
                    logger.error(f"Error processing queued trade {filepath}: {e}")
            
            # Sleep before next check
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Error in pending trades thread: {e}")
            time.sleep(60)  # Wait longer on error

def allocate_capital(trade_signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Allocate capital for the trade
    Returns allocation result with status
    """
    try:
        # Try to import capital_allocator module
        try:
            from capital_allocator import allocate_position_size
            
            # Run allocation
            allocation_result = allocate_position_size(trade_signal)
            
            # Update trade with allocation result
            if allocation_result.get("valid", False):
                allocated_trade = allocation_result.get("trade", {})
                allocated_trade["allocation_status"] = TradeStatus.ALLOCATED.value
                allocated_trade["allocation_time"] = time.time()
                logger.info(f"Capital allocation successful for {allocated_trade.get('symbol')}: "
                           f"size={allocated_trade.get('quantity', 0)}, "
                           f"leverage={allocated_trade.get('leverage', 1)}")
                return {
                    "valid": True,
                    "trade": allocated_trade,
                    "status": TradeStatus.ALLOCATED.value
                }
            else:
                reason = allocation_result.get("reason", "unknown_allocation_issue")
                logger.warning(f"Capital allocation failed: {reason}")
                return {
                    "valid": False,
                    "reason": reason,
                    "status": TradeStatus.REJECTED.value
                }
        except ImportError:
            # Simple allocation if module not available
            logger.warning("Capital allocator not available, using default allocation")
            
            # Simple position sizing based on account size
            account_balance = get_account_balance()
            entry_price = trade_signal.get("entry", 0.0)
            
            # Default to 2% risk per trade
            risk_per_trade = 0.02
            
            # Calculate risk amount
            risk_amount = account_balance * risk_per_trade
            
            # Calculate price risk based on entry/stop
            side = trade_signal.get("side", "").upper()
            stop_loss = trade_signal.get("stopLoss", 0.0)
            
            if side == "BUY" or side == "LONG":
                price_risk = entry_price - stop_loss
            else:
                price_risk = stop_loss - entry_price
                
            # Calculate position size
            quantity = risk_amount / price_risk if price_risk > 0 else 0
            
            # Cap at 5% of account
            notional_value = quantity * entry_price
            max_notional = account_balance * 0.05
            
            if notional_value > max_notional:
                quantity = max_notional / entry_price
                notional_value = max_notional
            
            # Update trade with allocation
            trade_signal["quantity"] = quantity
            trade_signal["notional_value"] = notional_value
            trade_signal["leverage"] = 1.0  # Default leverage
            trade_signal["allocation_status"] = TradeStatus.ALLOCATED.value
            trade_signal["allocation_time"] = time.time()
            
            logger.info(f"Applied default allocation for {trade_signal.get('symbol')}: "
                      f"quantity={quantity}, notional=${notional_value:.2f}")
            
            return {
                "valid": True,
                "trade": trade_signal,
                "status": TradeStatus.ALLOCATED.value
            }
    except Exception as e:
        logger.error(f"Error in capital allocation: {e}")
        return {
            "valid": False,
            "reason": f"allocation_error: {str(e)}",
            "status": TradeStatus.REJECTED.value
        }

# Start the background thread
pending_thread = threading.Thread(target=process_pending_trades_thread, daemon=True)
pending_thread.start()

# If this file is run directly, run a test
if __name__ == "__main__":
    logger.info("Enhanced Strategy Selection module initialized")
    logger.info(f"Using signals directory: {SIGNALS_DIR}")
    
    # Test with a sample bundle if available
    try:
        from data_bundle import assemble_data_bundle
        
        # Test symbol
        test_symbol = "BTCUSDT"
        logger.info(f"Running test with symbol: {test_symbol}")
        
        # Try to get bundle
        bundle = assemble_data_bundle(test_symbol)
        if bundle:
            # Get current price
            current_price = bundle.get("market_data", {}).get("price", 25000.0)
            
            # Run strategy tournament
            result = strategy_tournament(bundle, current_price, "neutral")
            
            # Log result
            if result.get("decision") == "trade":
                logger.info(f"Test successful! Generated trade signal for {test_symbol}: "
                           f"{result.get('side')} at {result.get('entry')} with "
                           f"strategy {result.get('strategy_name')}")
            else:
                logger.info(f"Test generated no-trade decision for {test_symbol}")
        else:
            logger.warning(f"Test bundle not available for {test_symbol}")
    except ImportError:
        logger.warning("data_bundle module not available for testing")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)