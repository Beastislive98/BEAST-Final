# order_flow.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import time

class OrderFlowAnalyzer:
    def __init__(self):
        self.snapshots = []
        self.last_update = 0
        self.snapshot_interval = 1.0  # seconds
        
    def analyze_order_book(self, depth_data: Dict[str, List]) -> Dict[str, Any]:
        """
        Analyze order book for imbalances and pressure
        
        Args:
            depth_data: Order book depth data from Binance
            
        Returns:
            Dict with order flow metrics
        """
        current_time = time.time()
        
        # Extract bid and ask data
        bids = depth_data.get("bids", [])
        asks = depth_data.get("asks", [])
        
        if not bids or not asks:
            return {"valid": False, "message": "Insufficient order book data"}
            
        # Convert to numpy arrays for faster processing
        try:
            # Format: [price, quantity]
            bid_prices = np.array([float(bid[0]) for bid in bids])
            bid_quantities = np.array([float(bid[1]) for bid in bids])
            
            ask_prices = np.array([float(ask[0]) for ask in asks])
            ask_quantities = np.array([float(ask[1]) for ask in asks])
        except (IndexError, ValueError) as e:
            return {"valid": False, "message": f"Error parsing order book: {e}"}
            
        # Calculate mid price
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        
        # Calculate bid-ask spread
        spread = ask_prices[0] - bid_prices[0]
        spread_pct = spread / mid_price * 100
        
        # Calculate order book imbalance
        total_bid_value = np.sum(bid_prices * bid_quantities)
        total_ask_value = np.sum(ask_prices * ask_quantities)
        
        if total_ask_value > 0:
            book_imbalance = total_bid_value / total_ask_value - 1.0
        else:
            book_imbalance = 0.0
            
        # Calculate pressure within 0.5% of mid price
        near_bid_mask = bid_prices >= (mid_price * 0.995)
        near_ask_mask = ask_prices <= (mid_price * 1.005)
        
        near_bid_value = np.sum(bid_prices[near_bid_mask] * bid_quantities[near_bid_mask])
        near_ask_value = np.sum(ask_prices[near_ask_mask] * ask_quantities[near_ask_mask])
        
        if near_ask_value > 0:
            pressure_ratio = near_bid_value / near_ask_value
        else:
            pressure_ratio = 1.0
            
        # Detect large orders (walls)
        largest_bid_idx = np.argmax(bid_quantities)
        largest_ask_idx = np.argmax(ask_quantities)
        
        largest_bid = {
            "price": float(bid_prices[largest_bid_idx]),
            "quantity": float(bid_quantities[largest_bid_idx])
        }
        
        largest_ask = {
            "price": float(ask_prices[largest_ask_idx]),
            "quantity": float(ask_quantities[largest_ask_idx])
        }
        
        # Take a snapshot for time-series analysis if enough time has passed
        if current_time - self.last_update >= self.snapshot_interval:
            self.snapshots.append({
                "timestamp": current_time,
                "mid_price": mid_price,
                "spread": spread,
                "book_imbalance": book_imbalance,
                "pressure_ratio": pressure_ratio
            })
            
            # Keep only the most recent 60 snapshots (1 minute at 1 snapshot/sec)
            if len(self.snapshots) > 60:
                self.snapshots = self.snapshots[-60:]
                
            self.last_update = current_time
        
        # Calculate rate of change for metrics if we have enough snapshots
        imbalance_change = 0.0
        pressure_change = 0.0
        
        if len(self.snapshots) >= 2:
            imbalance_change = book_imbalance - self.snapshots[-2]["book_imbalance"]
            pressure_change = pressure_ratio - self.snapshots[-2]["pressure_ratio"]
        
        # Determine directional pressure
        if pressure_ratio > 1.2 and imbalance_change > 0:
            direction = "bullish"
            pressure_strength = min(1.0, (pressure_ratio - 1.0) / 2)
        elif pressure_ratio < 0.8 and imbalance_change < 0:
            direction = "bearish"
            pressure_strength = min(1.0, (1.0 - pressure_ratio) / 2)
        else:
            direction = "neutral"
            pressure_strength = 0.0
        
        return {
            "valid": True,
            "mid_price": mid_price,
            "spread": spread,
            "spread_pct": spread_pct,
            "book_imbalance": book_imbalance,
            "imbalance_change": imbalance_change,
            "pressure_ratio": pressure_ratio,
            "pressure_change": pressure_change,
            "direction": direction,
            "pressure_strength": pressure_strength,
            "largest_bid": largest_bid,
            "largest_ask": largest_ask
        }