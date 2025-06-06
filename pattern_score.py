import logging
import random
from typing import List, Dict, Any

def generate_trade_signal(market_data: Dict[str, Any], price: float, leverage: int = 10) -> Dict[str, Any]:
    """
    Enhanced trade signal generation with balanced LONG/SHORT consideration
    """
    try:
        if price <= 0:
            logging.warning("Invalid price provided.")
            return {"decision": "no_trade", "reason": "invalid_price"}

        # Extract data for decision making
        symbol = market_data.get("symbol", "UNKNOWN")
        pattern = market_data.get("pattern_signal", {})
        sentiment = market_data.get("sentiment", {}).get("sentiment_score", 0)
        forecast = market_data.get("forecast", {})
        whale_flags = market_data.get("whale_flags", {})
        
        # ENHANCED DIRECTION DETERMINATION - NO BIAS
        
        # Score-based approach for balanced signals
        long_score = 0
        short_score = 0
        confidence_factors = []
        
        # 1. PATTERN ANALYSIS (Strongest signal - weight 40%)
        pattern_type = pattern.get("type", "neutral")
        pattern_confidence = pattern.get("confidence", 0.5)
        confidence_factors.append(pattern_confidence)
        
        if pattern_type == "bullish":
            long_score += 4
            logging.debug(f"{symbol}: Bullish pattern detected (+4 long)")
        elif pattern_type == "bearish":
            short_score += 4
            logging.debug(f"{symbol}: Bearish pattern detected (+4 short)")
        elif pattern_type == "neutral":
            # For neutral patterns, add slight randomness
            if random.random() > 0.5:
                long_score += 1
            else:
                short_score += 1
        
        # 2. SENTIMENT ANALYSIS (Weight 25%)
        if sentiment > 0.3:
            long_score += 3
            logging.debug(f"{symbol}: Strong positive sentiment (+3 long)")
        elif sentiment > 0.1:
            long_score += 2
            logging.debug(f"{symbol}: Positive sentiment (+2 long)")
        elif sentiment < -0.3:
            short_score += 3
            logging.debug(f"{symbol}: Strong negative sentiment (+3 short)")
        elif sentiment < -0.1:
            short_score += 2
            logging.debug(f"{symbol}: Negative sentiment (+2 short)")
        elif sentiment > 0:
            long_score += 1
        elif sentiment < 0:
            short_score += 1
        else:
            # Neutral sentiment - random
            if random.random() > 0.5:
                long_score += 1
            else:
                short_score += 1
        
        # 3. FORECAST ANALYSIS (Weight 20%)
        slope = forecast.get("slope", 0)
        if slope > 0.05:
            long_score += 2
            logging.debug(f"{symbol}: Strong upward forecast (+2 long)")
        elif slope > 0.02:
            long_score += 1
            logging.debug(f"{symbol}: Upward forecast (+1 long)")
        elif slope < -0.05:
            short_score += 2
            logging.debug(f"{symbol}: Strong downward forecast (+2 short)")
        elif slope < -0.02:
            short_score += 1
            logging.debug(f"{symbol}: Downward forecast (+1 short)")
        elif slope > 0:
            long_score += 1
        elif slope < 0:
            short_score += 1
        else:
            # Flat forecast - random
            if random.random() > 0.5:
                long_score += 1
            else:
                short_score += 1
        
        # 4. WHALE ACTIVITY ANALYSIS (Weight 15%)
        whale_present = whale_flags.get("whale_present", False)
        large_bid_wall = whale_flags.get("large_bid_wall", False)
        large_ask_wall = whale_flags.get("large_ask_wall", False)
        
        if whale_present:
            if large_bid_wall and not large_ask_wall:
                long_score += 2
                logging.debug(f"{symbol}: Large bid wall detected (+2 long)")
            elif large_ask_wall and not large_bid_wall:
                short_score += 2
                logging.debug(f"{symbol}: Large ask wall detected (+2 short)")
            elif large_bid_wall and large_ask_wall:
                # Both walls - use sentiment to decide
                if sentiment >= 0:
                    long_score += 1
                else:
                    short_score += 1
            else:
                # Generic whale activity - use sentiment
                if sentiment > 0:
                    long_score += 1
                elif sentiment < 0:
                    short_score += 1
                else:
                    # Random if sentiment is neutral
                    if random.random() > 0.5:
                        long_score += 1
                    else:
                        short_score += 1
        
        # 5. FINAL DIRECTION DETERMINATION
        total_signals = long_score + short_score
        
        if long_score > short_score:
            direction = "long"
            side = "BUY"
            position_side = "LONG"
            winning_score = long_score
        elif short_score > long_score:
            direction = "short"
            side = "SELL"
            position_side = "SHORT"
            winning_score = short_score
        else:
            # Perfect tie - use pure randomization (50/50)
            if random.random() >= 0.5:
                direction = "long"
                side = "BUY"
                position_side = "LONG"
                winning_score = long_score
            else:
                direction = "short"
                side = "SELL"
                position_side = "SHORT"
                winning_score = short_score
        
        # 6. CONFIDENCE CALCULATION
        # Base confidence from signal strength
        if total_signals > 0:
            signal_confidence = min(0.9, 0.4 + (winning_score / total_signals) * 0.5)
        else:
            signal_confidence = 0.5
        
        # Combine with pattern confidence
        final_confidence = (signal_confidence + pattern_confidence) / 2
        
        # Boost confidence for strong signal agreement
        if winning_score >= 6:  # Very strong signals
            final_confidence = min(0.95, final_confidence * 1.1)
        elif winning_score <= 2:  # Weak signals
            final_confidence = max(0.3, final_confidence * 0.9)
        
        # 7. DYNAMIC RISK/REWARD CALCULATION
        # Get volatility if available
        volatility = market_data.get("volatility", 0.02)
        
        # Adjust risk based on confidence and volatility
        base_risk_pct = 0.015  # 1.5% base risk
        
        # Higher confidence = can take slightly more risk
        confidence_risk_adj = 0.5 + (final_confidence * 0.5)  # 0.5x to 1x
        
        # Higher volatility = reduce risk
        volatility_risk_adj = max(0.5, 1.0 - (volatility * 5))  # Reduce risk in high volatility
        
        final_risk_pct = base_risk_pct * confidence_risk_adj * volatility_risk_adj
        
        # Calculate reward multiplier (dynamic R:R ratio)
        base_reward_multiplier = 2.0  # 2:1 base R:R
        
        # Higher confidence = higher reward target
        confidence_reward_adj = 1.0 + (final_confidence * 0.5)  # 1x to 1.5x
        
        final_reward_multiplier = base_reward_multiplier * confidence_reward_adj
        
        # 8. CALCULATE STOP LOSS AND TAKE PROFIT
        if direction == "long":
            stop_loss = round(price * (1 - final_risk_pct), 8)
            take_profit = round(price * (1 + (final_risk_pct * final_reward_multiplier)), 8)
        else:
            stop_loss = round(price * (1 + final_risk_pct), 8)
            take_profit = round(price * (1 - (final_risk_pct * final_reward_multiplier)), 8)
        
        # 9. CALCULATE DYNAMIC LEVERAGE
        base_leverage = 3.0
        confidence_leverage_adj = 0.7 + (final_confidence * 0.6)  # 0.7x to 1.3x
        volatility_leverage_adj = max(0.5, 1.0 - (volatility * 3))  # Reduce leverage in high volatility
        
        calculated_leverage = base_leverage * confidence_leverage_adj * volatility_leverage_adj
        final_leverage = max(1.0, min(10.0, calculated_leverage))  # Cap between 1-10x
        
        # 10. BUILD FINAL SIGNAL
        signal = {
            "decision": "trade",
            "entry": price,
            "stopLoss": stop_loss,
            "takeProfit": take_profit,
            "confidence": round(final_confidence, 3),
            "side": side,
            "positionSide": position_side,
            "leverage": round(final_leverage, 2),
            "symbol": symbol,
            
            # Analysis details for debugging
            "analysis": {
                "long_score": long_score,
                "short_score": short_score,
                "total_signals": total_signals,
                "winning_score": winning_score,
                "pattern_type": pattern_type,
                "pattern_confidence": pattern_confidence,
                "sentiment_score": sentiment,
                "forecast_slope": slope,
                "whale_present": whale_present,
                "final_risk_pct": final_risk_pct,
                "reward_multiplier": final_reward_multiplier,
                "volatility": volatility
            }
        }
        
        logging.info(f"📊 Signal Generated for {symbol}: {direction.upper()} | "
                    f"Confidence: {final_confidence:.2f} | "
                    f"Signals: L={long_score} S={short_score} | "
                    f"Leverage: {final_leverage:.1f}x")
        
        return signal

    except Exception as e:
        logging.error(f"Pattern scoring failed for {market_data.get('symbol', 'UNKNOWN')}: {e}")
        
        # Conservative fallback
        return {
            "decision": "trade",
            "entry": price,
            "stopLoss": round(price * 0.985, 8),  # 1.5% stop loss
            "takeProfit": round(price * 1.03, 8),  # 3% take profit
            "confidence": 0.4,
            "side": "BUY" if random.random() > 0.5 else "SELL",  # Random fallback
            "positionSide": "LONG" if random.random() > 0.5 else "SHORT",
            "leverage": 2.0,
            "symbol": market_data.get("symbol", "UNKNOWN"),
            "fallback": True
        }