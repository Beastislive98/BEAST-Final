# adaptive_sizing.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class AdaptivePositionSizer:
    def __init__(self, base_risk: float = 0.02, max_risk: float = 0.05):
        self.base_risk = base_risk  # Base risk percentage of account
        self.max_risk = max_risk    # Maximum risk percentage
        
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float, 
                              stop_loss: float,
                              market_data: Dict[str, Any],
                              confidence: float = 0.7) -> Dict[str, Any]:
        """
        Calculate optimal position size based on volatility and confidence
        
        Args:
            account_balance: Current account balance
            entry_price: Planned entry price
            stop_loss: Planned stop loss price
            market_data: Dict with market data including volatility metrics
            confidence: Trade confidence score
            
        Returns:
            Dict with position size and risk metrics
        """
        # Extract volatility data if available
        volatility = self._get_volatility(market_data)
        
        # Calculate standard risk in currency units
        risk_pct = self.base_risk
        
        # Adjust risk based on volatility
        if volatility:
            # Normalize volatility (ATR as % of price)
            norm_volatility = volatility / entry_price
            
            # Reduce risk for high volatility assets
            if norm_volatility > 0.03:  # 3% daily ATR is considered high
                risk_pct = self.base_risk * 0.7
            elif norm_volatility < 0.01:  # 1% daily ATR is considered low
                risk_pct = self.base_risk * 1.3
        
        # Adjust risk based on confidence
        risk_pct = risk_pct * (0.5 + confidence / 2)  # Scale from 50% to 150% of base risk
        
        # Ensure risk stays within limits
        risk_pct = min(risk_pct, self.max_risk)
        
        # Calculate risk amount in currency
        risk_amount = account_balance * risk_pct
        
        # Calculate stop loss distance
        if entry_price > stop_loss:  # Long position
            stop_distance = entry_price - stop_loss
            position_side = "LONG"
            side = "BUY"
        else:  # Short position
            stop_distance = stop_loss - entry_price
            position_side = "SHORT" 
            side = "SELL"
            
        # Avoid division by zero
        if stop_distance <= 0:
            return {
                "position_size": 0,
                "risk_amount": 0,
                "risk_pct": 0,
                "notional_value": 0,
                "error": "Invalid stop distance"
            }
            
        # Calculate position size
        position_size = risk_amount / stop_distance
        
        # Calculate notional value
        notional_value = position_size * entry_price
        
        # Calculate optimal leverage
        optimal_leverage = min(10, round(notional_value / (account_balance * 0.1)))
        
        return {
            "position_size": position_size,
            "position_size_rounded": self._round_size_for_symbol(position_size, entry_price),
            "risk_amount": risk_amount,
            "risk_pct": risk_pct,
            "notional_value": notional_value,
            "leverage": optimal_leverage,
            "side": side,
            "positionSide": position_side
        }
    
    def _get_volatility(self, market_data: Dict[str, Any]) -> Optional[float]:
        """Extract volatility metric from market data"""
        # Try to get ATR if available
        if "indicator_data" in market_data:
            df = market_data["indicator_data"]
            atr_col = None
            
            # Find ATR column (might be named differently)
            for col in df.columns:
                if "ATR" in col:
                    atr_col = col
                    break
                    
            if atr_col and len(df[atr_col]) > 0:
                return df[atr_col].iloc[-1]
        
        # Fallback to high-low range
        if "market_data" in market_data and "ticker" in market_data["market_data"]:
            ticker = market_data["market_data"]["ticker"]
            if "highPrice" in ticker and "lowPrice" in ticker:
                try:
                    high = float(ticker["highPrice"])
                    low = float(ticker["lowPrice"])
                    return high - low
                except (ValueError, TypeError):
                    pass
                    
        # Fallback to a simple volatility calculation from OHLC data
        if "ohlcv" in market_data:
            df = market_data["ohlcv"]
            if len(df) >= 20:
                # Calculate simple volatility as average of daily ranges
                daily_ranges = (df['High'] - df['Low']).rolling(window=20).mean()
                if len(daily_ranges) > 0:
                    return daily_ranges.iloc[-1]
                    
        return None
    
    def _round_size_for_symbol(self, size: float, price: float) -> float:
        """Round position size based on symbol price range"""
        if price >= 10000:  # BTC
            return round(size, 3)
        elif price >= 1000:  # ETH etc
            return round(size, 2)
        elif price >= 100:
            return round(size, 1)
        elif price >= 10:
            return round(size, 0)
        elif price >= 1:
            return round(size, 0)
        else:  # Very low priced assets
            return round(size, 0)