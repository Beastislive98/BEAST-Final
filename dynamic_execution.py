# dynamic_execution.py
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import time
import logging

class DynamicExecutionAlgorithm:
    def __init__(self):
        self.execution_strategies = {
            "twap": self._execute_twap,
            "immediate": self._execute_immediate,
            "iceberg": self._execute_iceberg,
            "smart": self._execute_smart
        }
    
    def optimize_execution(self, trade: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize trade execution based on market conditions
        
        Args:
            trade: Trade parameters
            market_data: Current market data
            
        Returns:
            Dict with optimized execution parameters
        """
        # Extract key parameters
        symbol = trade.get("symbol", "")
        side = trade.get("side", "")
        quantity = trade.get("quantity", 0.0)
        
        if not symbol or not side or quantity <= 0:
            return {"success": False, "message": "Invalid trade parameters"}
        
        # Analyze market conditions to select best execution strategy
        strategy = self._select_execution_strategy(trade, market_data)
        
        # Execute the selected strategy
        return self.execution_strategies[strategy](trade, market_data)
    
    def _select_execution_strategy(self, trade: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Select the best execution strategy based on market conditions"""
        # Extract relevant metrics
        volatility = self._get_volatility(market_data)
        liquidity = self._get_liquidity(market_data)
        spread = self._get_spread(market_data)
        trade_size = trade.get("notional_value", 0.0)
        
        # Default to immediate for small orders in liquid markets
        if trade_size < 100 and liquidity > 0.7:
            return "immediate"
        
        # Use TWAP for larger orders in stable markets
        if trade_size >= 100 and volatility < 0.01:
            return "twap"
        
        # Use iceberg for large orders in volatile markets
        if trade_size >= 500 and volatility >= 0.02:
            return "iceberg"
        
        # Default to smart for everything else
        return "smart"
    
    def _execute_immediate(self, trade: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entire order immediately at market price"""
        return {
            "success": True,
            "strategy": "immediate",
            "params": {
                "type": "MARKET",
                "quantity": trade.get("quantity", 0.0),
                "reduceOnly": False,
                "timeInForce": "GTC"
            }
        }
    
    def _execute_twap(self, trade: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order using Time-Weighted Average Price algorithm"""
        quantity = trade.get("quantity", 0.0)
        
        # Calculate TWAP parameters
        duration_minutes = 15  # 15 minute execution
        num_slices = 5
        slice_size = quantity / num_slices
        
        return {
            "success": True,
            "strategy": "twap",
            "params": {
                "type": "LIMIT",
                "quantity": quantity,
                "reduceOnly": False,
                "timeInForce": "GTC",
                "execution_algorithm": "twap",
                "slices": num_slices,
                "slice_size": slice_size,
                "duration_minutes": duration_minutes,
                "interval_seconds": duration_minutes * 60 / num_slices
            }
        }
    
    def _execute_iceberg(self, trade: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order using Iceberg algorithm to hide true order size"""
        quantity = trade.get("quantity", 0.0)
        
        # Calculate iceberg parameters
        visible_size = min(quantity * 0.1, 1.0)  # 10% of total size, max 1.0 units
        
        return {
            "success": True,
            "strategy": "iceberg",
            "params": {
                "type": "LIMIT",
                "quantity": quantity,
                "reduceOnly": False,
                "timeInForce": "GTC",
                "execution_algorithm": "iceberg",
                "visible_size": visible_size
            }
        }
    
    def _execute_smart(self, trade: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute order using smart algorithm that adapts to market conditions"""
        quantity = trade.get("quantity", 0.0)
        
        # Get order book and volatility
        volatility = self._get_volatility(market_data)
        spread = self._get_spread(market_data)
        
        # Calculate smart execution parameters
        if volatility > 0.02:
            # High volatility: be more passive
            price_offset = spread * 0.3  # 30% of spread from best price
            time_limit_seconds = 60  # Wait up to 60 seconds
        else:
            # Low volatility: be more aggressive
            price_offset = spread * 0.1  # 10% of spread from best price
            time_limit_seconds = 30  # Wait up to 30 seconds
        
        return {
            "success": True,
            "strategy": "smart",
            "params": {
                "type": "LIMIT",
                "quantity": quantity,
                "reduceOnly": False,
                "timeInForce": "GTT",  # Good Till Time
                "execution_algorithm": "smart",
                "price_offset": price_offset,
                "time_limit_seconds": time_limit_seconds,
                "adaptive": True
            }
        }
    
    def _get_volatility(self, market_data: Dict[str, Any]) -> float:
        """Extract volatility from market data"""
        # Try to get from indicator data
        if "indicator_data" in market_data:
            df = market_data["indicator_data"]
            if "ATR_14" in df.columns and len(df["ATR_14"]) > 0:
                atr = df["ATR_14"].iloc[-1]
                price = df["Close"].iloc[-1]
                return atr / price  # ATR as percentage of price
        
        # Fallback to simpler calculation
        if "ohlcv" in market_data:
            df = market_data["ohlcv"]
            if len(df) >= 20:
                returns = df["Close"].pct_change().dropna()
                return returns.std()
        
        return 0.01  # Default volatility estimate
    
    def _get_liquidity(self, market_data: Dict[str, Any]) -> float:
        """Estimate market liquidity from order book"""
        if "market_data" in market_data and "depth" in market_data["market_data"]:
            depth = market_data["market_data"]["depth"]
            
            bids = depth.get("bids", [])
            asks = depth.get("asks", [])
            
            if not bids or not asks:
                return 0.5  # Default liquidity
            
            try:
                # Calculate total order book depth
                bid_depth = sum(float(bid[1]) for bid in bids[:10])
                ask_depth = sum(float(ask[1]) for ask in asks[:10])
                
                total_depth = bid_depth + ask_depth
                
                # Normalize to 0-1 scale (higher is more liquid)
                # This is a simplified metric - you might want to adjust the scaling
                normalized_liquidity = min(1.0, total_depth / 1000)
                
                return normalized_liquidity
            except (IndexError, ValueError):
                pass
        
        return 0.5  # Default liquidity
    
    def _get_spread(self, market_data: Dict[str, Any]) -> float:
        """Get bid-ask spread from market data"""
        if "market_data" in market_data and "depth" in market_data["market_data"]:
            depth = market_data["market_data"]["depth"]
            
            bids = depth.get("bids", [])
            asks = depth.get("asks", [])
            
            if bids and asks:
                try:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    
                    return best_ask - best_bid
                except (IndexError, ValueError):
                    pass
        
        return 0.0  # Default spread