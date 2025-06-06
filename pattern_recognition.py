# pattern_recognition.py

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from pattern_registry import PATTERN_REGISTRY

def find_peaks_and_troughs(prices: np.ndarray, window: int = 5) -> Tuple[List[int], List[int]]:
    """
    Optimized method to identify peaks (local maxima) and troughs (local minima) in price data
    with vectorized operations for performance
    
    Args:
        prices: Array of price values
        window: Window size for peak/trough detection
        
    Returns:
        Tuple of (peak_indices, trough_indices)
    """
    if len(prices) < 2*window + 1:
        # Not enough data points
        return [], []
    
    # Use numpy operations for better performance
    peaks = []
    troughs = []
    
    # Vectorized approach for large datasets
    if len(prices) > 1000:
        # Create rolling windows for comparison
        max_left = np.zeros_like(prices)
        max_right = np.zeros_like(prices)
        min_left = np.zeros_like(prices)
        min_right = np.zeros_like(prices)
        
        # Calculate rolling max/min for each window
        for i in range(window, len(prices) - window):
            max_left[i] = np.max(prices[i-window:i])
            max_right[i] = np.max(prices[i+1:i+window+1])
            min_left[i] = np.min(prices[i-window:i])
            min_right[i] = np.min(prices[i+1:i+window+1])
        
        # Find peaks and troughs in vectorized way
        for i in range(window, len(prices) - window):
            # Peak if current value > all values in window
            if prices[i] > max_left[i] and prices[i] > max_right[i]:
                peaks.append(i)
            
            # Trough if current value < all values in window
            if prices[i] < min_left[i] and prices[i] < min_right[i]:
                troughs.append(i)
    else:
        # Original approach for smaller datasets
        for i in range(window, len(prices) - window):
            # Check if current point is a peak
            if all(prices[i] > prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, window+1)):
                peaks.append(i)
                
            # Check if current point is a trough
            if all(prices[i] < prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, window+1)):
                troughs.append(i)
            
    return peaks, troughs

def fit_trendline(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit a linear trendline (y = mx + b) to data points with improved performance
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        
    Returns:
        Tuple of (slope, intercept)
    """
    if len(x) < 2 or len(y) < 2:
        return 0, 0
        
    # Use numpy's polyfit for better performance
    if len(x) > 1000:
        # For large datasets, use a random sample for better performance
        sample_size = min(1000, len(x))
        indices = np.random.choice(len(x), sample_size, replace=False)
        x_sample = x[indices]
        y_sample = y[indices]
        m, b = np.polyfit(x_sample, y_sample, 1)
    else:
        # For smaller datasets, use all points
        m, b = np.polyfit(x, y, 1)
        
    return m, b

def is_head_and_shoulders(df: pd.DataFrame) -> bool:
    """
    Detect head and shoulders pattern: three peaks with the middle one higher
    Optimized implementation with better peak detection
    """
    if len(df) < 20:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        
        # Find peaks using optimized function
        peaks, _ = find_peaks_and_troughs(prices)
        
        if len(peaks) < 3:
            return False
            
        # Look for 3 peaks with specific characteristics
        # Calculate distances between consecutive peaks
        peak_distances = []
        for i in range(len(peaks) - 1):
            peak_distances.append(peaks[i+1] - peaks[i])
        
        # Find candidate triplets
        for i in range(len(peaks) - 2):
            # Get three consecutive peaks
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            
            # Extract peak heights
            h1, h2, h3 = prices[p1], prices[p2], prices[p3]
            
            # Check if middle peak is highest
            if h2 > h1 and h2 > h3:
                # Check if shoulders are at similar heights (within 10%)
                shoulder_diff_pct = abs(h1 - h3) / h1
                if shoulder_diff_pct < 0.10:
                    # Check spacing between peaks (should be somewhat even)
                    spacing_ratio = abs((p3 - p2) / (p2 - p1) - 1)
                    if spacing_ratio < 0.5:  # Peaks should be somewhat evenly spaced
                        # Find troughs between peaks
                        left_trough = min(prices[p1:p2])
                        right_trough = min(prices[p2:p3])
                        
                        # Troughs should be at similar levels
                        trough_diff_pct = abs(left_trough - right_trough) / left_trough
                        if trough_diff_pct < 0.10:
                            # Current price should be near or below the neckline
                            neckline = (left_trough + right_trough) / 2
                            if prices[-1] <= neckline * 1.02:  # Within 2% of neckline
                                return True
        
        return False
    except Exception as e:
        logging.warning(f"Head and shoulders detection error: {e}")
        return False

def recognize_patterns(df, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced pattern recognition that combines candlestick and chart patterns
    for higher confidence signals
    """
    try:
        if df is None or df.empty:
            logging.warning(f"Empty dataframe for {symbol}, cannot recognize patterns")
            return None
            
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing required columns for {symbol}: {missing_cols}")
            return None
        
        # Separate candlestick and chart patterns
        candlestick_patterns = {name: rule_set for name, rule_set in PATTERN_REGISTRY.items() 
                                if rule_set.get("type") == "candlestick"}
        
        chart_patterns = {name: rule_set for name, rule_set in PATTERN_REGISTRY.items() 
                           if rule_set.get("type") == "chart"}
        
        # Track all matched patterns for potential combination
        all_matched_patterns = []
        
        # Check for candlestick patterns first (usually smaller windows)
        best_candlestick_match = None
        for name, rule_set in candlestick_patterns.items():
            window = rule_set["window"]
            if len(df) < window:
                continue
                
            data_slice = df.tail(window)
            rule_func = rule_set["rule"]

            try:
                if rule_func(data_slice):
                    pattern_type = get_pattern_type(name)
                    pattern_match = {
                        "pattern": name,
                        "confidence": rule_set["confidence"],
                        "type": pattern_type,
                        "method": "candlestick"
                    }
                    all_matched_patterns.append(pattern_match)
                    
                    # Track the best candlestick match
                    if not best_candlestick_match or rule_set["confidence"] > best_candlestick_match["confidence"]:
                        best_candlestick_match = pattern_match
            except Exception as pattern_err:
                logging.warning(f"Error checking {name} candlestick pattern for {symbol}: {pattern_err}")
        
        # Next check for chart patterns
        best_chart_match = None
        for name, rule_set in chart_patterns.items():
            window = rule_set["window"]
            if len(df) < window:
                continue
                
            data_slice = df.tail(window)
            rule_func = rule_set["rule"]

            try:
                if rule_func(data_slice):
                    pattern_type = get_pattern_type(name)
                    pattern_match = {
                        "pattern": name,
                        "confidence": rule_set["confidence"],
                        "type": pattern_type,
                        "method": "chart"
                    }
                    all_matched_patterns.append(pattern_match)
                    
                    # Track the best chart match
                    if not best_chart_match or rule_set["confidence"] > best_chart_match["confidence"]:
                        best_chart_match = pattern_match
            except Exception as pattern_err:
                logging.warning(f"Error checking {name} chart pattern for {symbol}: {pattern_err}")
        
        # Log the total pattern matches found
        if all_matched_patterns:
            logging.info(f"Found {len(all_matched_patterns)} pattern matches for {symbol}")
            for match in all_matched_patterns:
                logging.debug(f"  - {match['method']}: {match['pattern']} ({match['type']}) with confidence {match['confidence']}")
        
        # If no patterns found
        if not best_candlestick_match and not best_chart_match:
            logging.info(f"[PATTERN] No pattern matched for {symbol}")
            
            # Try to find similar historical pattern if available
            historical_match = find_historical_pattern_match(df)
            if historical_match:
                logging.info(f"[PATTERN] Found historical match for {symbol}: {historical_match['pattern']}")
                return historical_match
                
            return {
                "pattern": None,
                "confidence": 0.0,
                "type": None,
                "method": None
            }
        
        # If both candlestick and chart patterns found, combine them
        if best_candlestick_match and best_chart_match:
            # Check if patterns agree on direction
            candlestick_type = best_candlestick_match["type"]
            chart_type = best_chart_match["type"]
            
            # If patterns confirm each other (both bullish or both bearish), boost confidence
            if (candlestick_type == "bullish" and chart_type == "bullish") or \
               (candlestick_type == "bearish" and chart_type == "bearish"):
                combined_confidence = min(0.98, best_candlestick_match["confidence"] * 0.6 + 
                                         best_chart_match["confidence"] * 0.5)
                pattern_type = candlestick_type  # They agree, so use either one
            else:
                # Patterns conflict, use the one with higher confidence but reduce overall confidence
                if best_candlestick_match["confidence"] > best_chart_match["confidence"]:
                    pattern_type = candlestick_type
                    combined_confidence = best_candlestick_match["confidence"] * 0.85
                else:
                    pattern_type = chart_type
                    combined_confidence = best_chart_match["confidence"] * 0.85
            
            # Create combined pattern result
            combined_result = {
                "pattern": f"{best_candlestick_match['pattern']}+{best_chart_match['pattern']}",
                "confidence": round(combined_confidence, 3),
                "type": pattern_type,
                "method": "combined",
                "candlestick_pattern": best_candlestick_match["pattern"],
                "chart_pattern": best_chart_match["pattern"],
                "candlestick_confidence": best_candlestick_match["confidence"],
                "chart_confidence": best_chart_match["confidence"],
                "pattern_id": hash_patterns(best_candlestick_match["pattern"], best_chart_match["pattern"])
            }
            
            logging.info(f"[PATTERN] Combined patterns found for {symbol}: {combined_result['pattern']}")
            return combined_result
        
        # If only one pattern type found, return that one
        result = best_candlestick_match or best_chart_match
        
        # Add pattern_id for tracking
        if result:
            result["pattern_id"] = hash_pattern(result["pattern"])
        
        logging.info(f"[PATTERN] Found {result['method']} pattern {result['pattern']} for {symbol}")
        return result

    except Exception as e:
        logging.exception(f"Pattern recognition failed for {symbol}: {e}")
        return None

def determine_pattern_type(pattern_name: str) -> str:
    """
    Determine if a pattern is bullish, bearish, or neutral based on its name
    """
    pattern_name = pattern_name.lower()
    
    # Check for bullish patterns
    bullish_keywords = ['bullish', 'bottom', 'support', 'ascending', 'hammer', 'morning', 'engulfing_bull', 
                        'three_white', 'piercing', 'three_inside_up', 'three_outside_up']
    
    bearish_keywords = ['bearish', 'top', 'resistance', 'descending', 'hanging', 'evening', 'engulfing_bear',
                       'three_black', 'dark_cloud', 'three_inside_down', 'three_outside_down'] 
    
    # First check exact matches
    if 'bullish' in pattern_name:
        return "bullish"
    if 'bearish' in pattern_name:
        return "bearish"
        
    # Then check keyword matches
    for keyword in bullish_keywords:
        if keyword in pattern_name:
            return "bullish"
            
    for keyword in bearish_keywords:
        if keyword in pattern_name:
            return "bearish"
    
    # Some specific patterns
    if pattern_name == 'hammer' or pattern_name == 'inverse_head_and_shoulders' or pattern_name == 'double_bottom':
        return "bullish"
    if pattern_name == 'shooting_star' or pattern_name == 'head_and_shoulders' or pattern_name == 'double_top':
        return "bearish"
    
    return "neutral"

def get_pattern_type(pattern_name: str) -> str:
    """
    Wrapper for determine_pattern_type with additional logic
    """
    return determine_pattern_type(pattern_name)

def hash_pattern(pattern_name: str) -> int:
    """
    Create a unique integer hash for a pattern name for tracking
    """
    # Simple hash function - take string hash and make positive
    return abs(hash(pattern_name)) % 1000000

def hash_patterns(pattern1: str, pattern2: str) -> int:
    """
    Create a unique integer hash for a combined pattern
    """
    # Combine patterns and hash them
    combined = f"{pattern1}_{pattern2}"
    return abs(hash(combined)) % 1000000

def get_key_levels(df: pd.DataFrame, n_levels: int = 3) -> List[float]:
    """
    Find key support/resistance levels from historical data
    Used for ATM/ITM/OTM assessment
    """
    try:
        # Locate significant highs and lows
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find local maximums and minimums using the optimized find_peaks_and_troughs
        peaks, troughs = find_peaks_and_troughs(highs, window=5)
        
        # Get the values at the peak and trough indices
        peak_values = [highs[p] for p in peaks]
        trough_values = [lows[t] for t in troughs]
        
        # Combine all levels
        all_levels = sorted(peak_values + trough_values)
        if not all_levels:
            return []
            
        # Define a proximity threshold (e.g., 0.5% of price range)
        price_range = max(highs) - min(lows)
        proximity_threshold = price_range * 0.005
        
        # Group levels that are close together
        grouped_levels = []
        if all_levels:
            current_group = [all_levels[0]]
            
            for level in all_levels[1:]:
                # If this level is close to the previous one, add to current group
                if abs(level - current_group[-1]) < proximity_threshold:
                    current_group.append(level)
                else:
                    # Otherwise, finalize the current group and start a new one
                    grouped_levels.append(sum(current_group) / len(current_group))
                    current_group = [level]
            
            # Add the last group
            if current_group:
                grouped_levels.append(sum(current_group) / len(current_group))
        
        # Take the top N most significant levels (those that occur most frequently)
        return sorted(grouped_levels, reverse=True)[:n_levels]
        
    except Exception as e:
        logging.error(f"Error finding key levels: {e}")
        return []

def find_historical_pattern_match(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    When no pattern is directly recognized, try to find similar historical patterns
    """
    try:
        # This function would ideally query a database of historical patterns
        # For now, implement a simple version based on recent price action
        
        # Calculate some basic metrics over the last N bars
        window = 10
        if len(df) < window:
            return None
            
        recent_df = df.tail(window)
        
        # Calculate trend direction
        price_change = recent_df['Close'].iloc[-1] / recent_df['Close'].iloc[0] - 1
        
        # Calculate volatility
        volatility = recent_df['High'].max() / recent_df['Low'].min() - 1
        
        # Calculate volume trend
        if 'Volume' in recent_df.columns:
            volume_change = recent_df['Volume'].iloc[-1] / recent_df['Volume'].iloc[0] - 1
        else:
            volume_change = 0
            
        # Simple trend detection
        if price_change > 0.02 and volume_change > 0:  # 2% up with increasing volume
            return {
                "pattern": "historical_uptrend",
                "confidence": 0.65,
                "type": "bullish",
                "method": "historical_analysis", 
                "pattern_id": hash_pattern("historical_uptrend")
            }
        elif price_change < -0.02 and volume_change > 0:  # 2% down with increasing volume
            return {
                "pattern": "historical_downtrend",
                "confidence": 0.65,
                "type": "bearish",
                "method": "historical_analysis",
                "pattern_id": hash_pattern("historical_downtrend")
            }
        elif volatility > 0.03:  # High volatility
            return {
                "pattern": "historical_volatility",
                "confidence": 0.6,
                "type": "neutral",
                "method": "historical_analysis",
                "pattern_id": hash_pattern("historical_volatility")
            }
        
        # No clear pattern
        return None
            
    except Exception as e:
        logging.error(f"Historical pattern matching failed: {e}")
        return None

def assess_moneyness(current_price: float, key_levels: List[float], direction: str = "long") -> str:
    """
    Assess moneyness of a trade similar to options (ATM/ITM/OTM)
    For crypto trading, this means position relative to key levels
    
    Args:
        current_price: Current price of the asset
        key_levels: List of key support/resistance levels
        direction: Trade direction ('long' or 'short')
        
    Returns:
        String indicating moneyness ('ATM', 'ITM', 'OTM')
    """
    if not key_levels:
        return "unknown"
    
    # Find the closest key level
    closest_level = min(key_levels, key=lambda x: abs(x - current_price))
    pct_diff = (current_price - closest_level) / closest_level * 100
    
    if direction.lower() == "long":
        if abs(pct_diff) < 0.5:
            return "ATM"  # At the money (near key level)
        elif pct_diff > 0:
            return "ITM"  # In the money (above key level)
        else:
            return "OTM"  # Out of the money (below key level)
    else:  # Short direction
        if abs(pct_diff) < 0.5:
            return "ATM"
        elif pct_diff < 0:
            return "ITM"  # For shorts, in the money means below key level
        else:
            return "OTM"