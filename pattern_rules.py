# pattern_rules.py

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any, Union

# === Candlestick Pattern Detection Implementations ===

def is_hammer(df: pd.DataFrame) -> bool:
    """
    Detect hammer candlestick pattern: small body, long lower shadow, little/no upper shadow
    """
    if len(df) < 1:
        return False
        
    row = df.iloc[-1]
    
    # Calculate parts of the candle
    body_size = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    
    # Avoid division by zero
    if total_range == 0:
        return False
        
    body_percentage = body_size / total_range
    
    # A hammer has a small body (less than 1/3 of the total range)
    if body_percentage > 0.3:
        return False
        
    # Get lower shadow length and upper shadow length
    if row['Close'] >= row['Open']:  # Bullish candle
        upper_shadow = row['High'] - row['Close']
        lower_shadow = row['Open'] - row['Low']
    else:  # Bearish candle
        upper_shadow = row['High'] - row['Open']
        lower_shadow = row['Close'] - row['Low']
        
    # Upper shadow should be very small or non-existent
    if upper_shadow / total_range > 0.1:
        return False
        
    # Lower shadow should be at least 2 times the body size
    if lower_shadow < (2 * body_size):
        return False
        
    # Lower shadow should be at least 60% of the total range
    if lower_shadow / total_range < 0.6:
        return False
        
    return True

def is_inverted_hammer(df: pd.DataFrame) -> bool:
    """
    Detect inverted hammer: small body, long upper shadow, little/no lower shadow
    """
    if len(df) < 1:
        return False
        
    row = df.iloc[-1]
    
    # Calculate parts of the candle
    body_size = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    
    # Avoid division by zero
    if total_range == 0:
        return False
        
    body_percentage = body_size / total_range
    
    # An inverted hammer has a small body (less than 1/3 of the total range)
    if body_percentage > 0.3:
        return False
        
    # Get lower shadow length and upper shadow length
    if row['Close'] >= row['Open']:  # Bullish candle
        upper_shadow = row['High'] - row['Close']
        lower_shadow = row['Open'] - row['Low']
    else:  # Bearish candle
        upper_shadow = row['High'] - row['Open']
        lower_shadow = row['Close'] - row['Low']
        
    # Lower shadow should be very small or non-existent
    if lower_shadow / total_range > 0.1:
        return False
        
    # Upper shadow should be at least 2 times the body size
    if upper_shadow < (2 * body_size):
        return False
        
    # Upper shadow should be at least 60% of the total range
    if upper_shadow / total_range < 0.6:
        return False
        
    return True

def is_bullish_marubozu(df: pd.DataFrame) -> bool:
    """
    Detect bullish marubozu: long bullish body with no or very small shadows
    """
    if len(df) < 1:
        return False
        
    row = df.iloc[-1]
    
    # Must be a bullish candle
    if row['Close'] <= row['Open']:
        return False
        
    # Calculate parts of the candle
    body_size = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    
    # Avoid division by zero
    if total_range == 0:
        return False
        
    body_percentage = body_size / total_range
    
    # Body should be at least 90% of the total range
    if body_percentage < 0.9:
        return False
        
    # Upper shadow
    upper_shadow = row['High'] - row['Close']
    
    # Lower shadow
    lower_shadow = row['Open'] - row['Low']
    
    # Both shadows should be less than 5% of the total range
    if (upper_shadow / total_range > 0.05) or (lower_shadow / total_range > 0.05):
        return False
        
    return True

def is_bearish_marubozu(df: pd.DataFrame) -> bool:
    """
    Detect bearish marubozu: long bearish body with no or very small shadows
    """
    if len(df) < 1:
        return False
        
    row = df.iloc[-1]
    
    # Must be a bearish candle
    if row['Close'] >= row['Open']:
        return False
        
    # Calculate parts of the candle
    body_size = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    
    # Avoid division by zero
    if total_range == 0:
        return False
        
    body_percentage = body_size / total_range
    
    # Body should be at least 90% of the total range
    if body_percentage < 0.9:
        return False
        
    # Upper shadow
    upper_shadow = row['High'] - row['Open']
    
    # Lower shadow
    lower_shadow = row['Close'] - row['Low']
    
    # Both shadows should be less than 5% of the total range
    if (upper_shadow / total_range > 0.05) or (lower_shadow / total_range > 0.05):
        return False
        
    return True

def is_doji(df: pd.DataFrame) -> bool:
    """
    Detect doji: opening and closing prices are very close
    """
    if len(df) < 1:
        return False
        
    row = df.iloc[-1]
    
    # Calculate parts of the candle
    body_size = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    
    # Avoid division by zero
    if total_range == 0:
        return False
        
    body_percentage = body_size / total_range
    
    # Body should be very small (<5% of total range)
    if body_percentage > 0.05:
        return False
        
    # Should have significant total range
    price_avg = (row['High'] + row['Low']) / 2
    if total_range / price_avg < 0.001:  # Less than 0.1% of price
        return False
        
    return True

def is_spinning_top(df: pd.DataFrame) -> bool:
    """
    Detect spinning top: small body with upper and lower shadows of similar length
    """
    if len(df) < 1:
        return False
        
    row = df.iloc[-1]
    
    # Calculate parts of the candle
    body_size = abs(row['Close'] - row['Open'])
    total_range = row['High'] - row['Low']
    
    # Avoid division by zero
    if total_range == 0:
        return False
        
    body_percentage = body_size / total_range
    
    # Body should be small (between 10% and 30% of total range)
    if body_percentage < 0.1 or body_percentage > 0.3:
        return False
        
    # Get upper and lower shadow lengths
    if row['Close'] >= row['Open']:  # Bullish candle
        upper_shadow = row['High'] - row['Close']
        lower_shadow = row['Open'] - row['Low']
    else:  # Bearish candle
        upper_shadow = row['High'] - row['Open']
        lower_shadow = row['Close'] - row['Low']
        
    # Both shadows should be significant
    if upper_shadow / total_range < 0.3 or lower_shadow / total_range < 0.3:
        return False
        
    # Shadows should be roughly equal (within 30% difference)
    if lower_shadow == 0:  # Avoid division by zero
        return False
        
    shadow_ratio = upper_shadow / lower_shadow
    if shadow_ratio < 0.7 or shadow_ratio > 1.3:
        return False
        
    return True

def is_hanging_man(df: pd.DataFrame) -> bool:
    """
    Detect hanging man: hammer pattern occurring at the end of an uptrend
    """
    if len(df) < 5:  # Need at least 5 candles to check the trend
        return False
        
    # Check if it's a hammer pattern
    if not is_hammer(df.iloc[-1:]):
        return False
        
    # Check if it occurs at the end of an uptrend (at least 3 of 5 prior candles should be bullish)
    prior_candles = df.iloc[-6:-1]  # Get the 5 candles before the current one
    bullish_count = sum(1 for i in range(len(prior_candles)) if prior_candles.iloc[i]['Close'] > prior_candles.iloc[i]['Open'])
    
    # Check if the prior trend was up
    if bullish_count < 3:
        return False
    
    # Check if price has been rising
    if df.iloc[-6]['Close'] >= df.iloc[-2]['Close']:
        return False
        
    return True

def is_shooting_star(df: pd.DataFrame) -> bool:
    """
    Detect shooting star: inverted hammer occurring at the end of an uptrend
    """
    if len(df) < 5:  # Need at least 5 candles to check the trend
        return False
        
    # Check if it's an inverted hammer pattern
    if not is_inverted_hammer(df.iloc[-1:]):
        return False
        
    # Check if it occurs at the end of an uptrend (at least 3 of 5 prior candles should be bullish)
    prior_candles = df.iloc[-6:-1]  # Get the 5 candles before the current one
    bullish_count = sum(1 for i in range(len(prior_candles)) if prior_candles.iloc[i]['Close'] > prior_candles.iloc[i]['Open'])
    
    # Check if the prior trend was up
    if bullish_count < 3:
        return False
    
    # Check if price has been rising
    if df.iloc[-6]['Close'] >= df.iloc[-2]['Close']:
        return False
        
    return True

def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    """
    Detect bullish engulfing pattern: a bearish candle followed by a larger bullish candle
    """
    if len(df) < 2:
        return False
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Previous candle must be bearish
    if prev['Close'] >= prev['Open']:
        return False
        
    # Current candle must be bullish
    if curr['Close'] <= curr['Open']:
        return False
        
    # Current candle's body must engulf previous candle's body
    if not (curr['Open'] <= prev['Close'] and curr['Close'] >= prev['Open']):
        return False
        
    # Additional confirmation: current candle's body should be significantly larger
    prev_body = abs(prev['Close'] - prev['Open'])
    curr_body = abs(curr['Close'] - curr['Open'])
    
    if curr_body <= prev_body:
        return False
        
    return True

def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    """
    Detect bearish engulfing pattern: a bullish candle followed by a larger bearish candle
    """
    if len(df) < 2:
        return False
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Previous candle must be bullish
    if prev['Close'] <= prev['Open']:
        return False
        
    # Current candle must be bearish
    if curr['Close'] >= curr['Open']:
        return False
        
    # Current candle's body must engulf previous candle's body
    if not (curr['Open'] >= prev['Close'] and curr['Close'] <= prev['Open']):
        return False
        
    # Additional confirmation: current candle's body should be significantly larger
    prev_body = abs(prev['Close'] - prev['Open'])
    curr_body = abs(curr['Close'] - curr['Open'])
    
    if curr_body <= prev_body:
        return False
        
    return True

def is_bullish_harami(df: pd.DataFrame) -> bool:
    """
    Detect bullish harami pattern: a large bearish candle followed by a smaller bullish candle
    contained within the body of the first candle
    """
    if len(df) < 2:
        return False
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Previous candle must be bearish
    if prev['Close'] >= prev['Open']:
        return False
        
    # Current candle must be bullish
    if curr['Close'] <= curr['Open']:
        return False
        
    # Current candle's body must be contained within previous candle's body
    if not (curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']):
        return False
        
    # Additional confirmation: current candle's body should be significantly smaller
    prev_body = abs(prev['Close'] - prev['Open'])
    curr_body = abs(curr['Close'] - curr['Open'])
    
    if curr_body > prev_body * 0.7:  # Current body should be at most 70% of previous
        return False
        
    return True

def is_bearish_harami(df: pd.DataFrame) -> bool:
    """
    Detect bearish harami pattern: a large bullish candle followed by a smaller bearish candle
    contained within the body of the first candle
    """
    if len(df) < 2:
        return False
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Previous candle must be bullish
    if prev['Close'] <= prev['Open']:
        return False
        
    # Current candle must be bearish
    if curr['Close'] >= curr['Open']:
        return False
        
    # Current candle's body must be contained within previous candle's body
    if not (curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']):
        return False
        
    # Additional confirmation: current candle's body should be significantly smaller
    prev_body = abs(prev['Close'] - prev['Open'])
    curr_body = abs(curr['Close'] - curr['Open'])
    
    if curr_body > prev_body * 0.7:  # Current body should be at most 70% of previous
        return False
        
    return True

def is_piercing_line(df: pd.DataFrame) -> bool:
    """
    Detect piercing line pattern: a bearish candle followed by a bullish candle that
    closes above the midpoint of the previous candle's body
    """
    if len(df) < 2:
        return False
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Previous candle must be bearish
    if prev['Close'] >= prev['Open']:
        return False
        
    # Current candle must be bullish
    if curr['Close'] <= curr['Open']:
        return False
        
    # Current candle must open below previous close
    if curr['Open'] >= prev['Close']:
        return False
        
    # Current candle must close above midpoint of previous candle's body
    prev_midpoint = (prev['Open'] + prev['Close']) / 2
    if curr['Close'] <= prev_midpoint:
        return False
        
    # Current candle must close below previous open
    if curr['Close'] >= prev['Open']:
        return False
        
    return True

def is_dark_cloud_cover(df: pd.DataFrame) -> bool:
    """
    Detect dark cloud cover pattern: a bullish candle followed by a bearish candle that
    opens above the previous high and closes below the midpoint of the previous candle's body
    """
    if len(df) < 2:
        return False
        
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    # Previous candle must be bullish
    if prev['Close'] <= prev['Open']:
        return False
        
    # Current candle must be bearish
    if curr['Close'] >= curr['Open']:
        return False
        
    # Current candle must open above previous high
    if curr['Open'] <= prev['High']:
        return False
        
    # Current candle must close below midpoint of previous candle's body
    prev_midpoint = (prev['Open'] + prev['Close']) / 2
    if curr['Close'] >= prev_midpoint:
        return False
        
    return True

def is_morning_star(df: pd.DataFrame) -> bool:
    """
    Detect morning star pattern: a bearish candle, followed by a small-bodied candle 
    that gaps lower, followed by a bullish candle
    """
    if len(df) < 3:
        return False
        
    first = df.iloc[-3]
    middle = df.iloc[-2]
    last = df.iloc[-1]
    
    # First candle must be bearish
    if first['Close'] >= first['Open']:
        return False
        
    # Last candle must be bullish
    if last['Close'] <= last['Open']:
        return False
        
    # Middle candle must have a small body
    middle_body = abs(middle['Close'] - middle['Open'])
    middle_range = middle['High'] - middle['Low']
    
    if middle_range == 0:  # Avoid division by zero
        return False
    
    if middle_body / middle_range > 0.3:  # Body should be less than 30% of the range
        return False
        
    # Gap between first and middle candle
    if middle['High'] >= first['Low']:
        return False
        
    # Last candle must close above the midpoint of the first candle's body
    first_midpoint = (first['Open'] + first['Close']) / 2
    if last['Close'] <= first_midpoint:
        return False
        
    return True

def is_evening_star(df: pd.DataFrame) -> bool:
    """
    Detect evening star pattern: a bullish candle, followed by a small-bodied candle 
    that gaps higher, followed by a bearish candle
    """
    if len(df) < 3:
        return False
        
    first = df.iloc[-3]
    middle = df.iloc[-2]
    last = df.iloc[-1]
    
    # First candle must be bullish
    if first['Close'] <= first['Open']:
        return False
        
    # Last candle must be bearish
    if last['Close'] >= last['Open']:
        return False
        
    # Middle candle must have a small body
    middle_body = abs(middle['Close'] - middle['Open'])
    middle_range = middle['High'] - middle['Low']
    
    if middle_range == 0:  # Avoid division by zero
        return False
    
    if middle_body / middle_range > 0.3:  # Body should be less than 30% of the range
        return False
        
    # Gap between first and middle candle
    if middle['Low'] <= first['High']:
        return False
        
    # Last candle must close below the midpoint of the first candle's body
    first_midpoint = (first['Open'] + first['Close']) / 2
    if last['Close'] >= first_midpoint:
        return False
        
    return True

def is_three_white_soldiers(df: pd.DataFrame) -> bool:
    """
    Detect three white soldiers pattern: three consecutive bullish candles, each 
    opening within the previous candle's body and closing higher than the previous close
    """
    if len(df) < 3:
        return False
        
    first = df.iloc[-3]
    middle = df.iloc[-2]
    last = df.iloc[-1]
    
    # All three candles must be bullish
    if first['Close'] <= first['Open'] or middle['Close'] <= middle['Open'] or last['Close'] <= last['Open']:
        return False
        
    # Each candle should open within the body of the previous candle
    if middle['Open'] < first['Open'] or middle['Open'] > first['Close']:
        return False
        
    if last['Open'] < middle['Open'] or last['Open'] > middle['Close']:
        return False
        
    # Each candle should close higher than the previous candle
    if middle['Close'] <= first['Close'] or last['Close'] <= middle['Close']:
        return False
        
    # Each candle should have a similar size (within 50% difference)
    first_body = first['Close'] - first['Open']
    middle_body = middle['Close'] - middle['Open']
    last_body = last['Close'] - last['Open']
    
    if not (0.5 * first_body <= middle_body <= 1.5 * first_body and 
            0.5 * middle_body <= last_body <= 1.5 * middle_body):
        return False
        
    return True

def is_three_black_crows(df: pd.DataFrame) -> bool:
    """
    Detect three black crows pattern: three consecutive bearish candles, each 
    opening within the previous candle's body and closing lower than the previous close
    """
    if len(df) < 3:
        return False
        
    first = df.iloc[-3]
    middle = df.iloc[-2]
    last = df.iloc[-1]
    
    # All three candles must be bearish
    if first['Close'] >= first['Open'] or middle['Close'] >= middle['Open'] or last['Close'] >= last['Open']:
        return False
        
    # Each candle should open within the body of the previous candle
    if middle['Open'] > first['Open'] or middle['Open'] < first['Close']:
        return False
        
    if last['Open'] > middle['Open'] or last['Open'] < middle['Close']:
        return False
        
    # Each candle should close lower than the previous candle
    if middle['Close'] >= first['Close'] or last['Close'] >= middle['Close']:
        return False
        
    # Each candle should have a similar size (within 50% difference)
    first_body = first['Open'] - first['Close']
    middle_body = middle['Open'] - middle['Close']
    last_body = last['Open'] - last['Close']
    
    if not (0.5 * first_body <= middle_body <= 1.5 * first_body and 
            0.5 * middle_body <= last_body <= 1.5 * middle_body):
        return False
        
    return True

def is_three_inside_up(df: pd.DataFrame) -> bool:
    """
    Detect three inside up pattern: bearish candle, followed by bullish harami,
    followed by a third bullish candle that closes above the first candle's close
    """
    if len(df) < 3:
        return False
        
    # Check if the first two candles form a bullish harami
    if not is_bullish_harami(df.iloc[-3:-1]):
        return False
        
    # The third candle must be bullish
    if df.iloc[-1]['Close'] <= df.iloc[-1]['Open']:
        return False
        
    # The third candle must close above the first candle's open
    if df.iloc[-1]['Close'] <= df.iloc[-3]['Open']:
        return False
        
    return True

def is_three_inside_down(df: pd.DataFrame) -> bool:
    """
    Detect three inside down pattern: bullish candle, followed by bearish harami,
    followed by a third bearish candle that closes below the first candle's close
    """
    if len(df) < 3:
        return False
        
    # Check if the first two candles form a bearish harami
    if not is_bearish_harami(df.iloc[-3:-1]):
        return False
        
    # The third candle must be bearish
    if df.iloc[-1]['Close'] >= df.iloc[-1]['Open']:
        return False
        
    # The third candle must close below the first candle's open
    if df.iloc[-1]['Close'] >= df.iloc[-3]['Open']:
        return False
        
    return True

def is_three_outside_up(df: pd.DataFrame) -> bool:
    """
    Detect three outside up pattern: bearish candle, followed by bullish engulfing,
    followed by a third bullish candle
    """
    if len(df) < 3:
        return False
        
    # Check if the first two candles form a bullish engulfing
    if not is_bullish_engulfing(df.iloc[-3:-1]):
        return False
        
    # The third candle must be bullish
    if df.iloc[-1]['Close'] <= df.iloc[-1]['Open']:
        return False
        
    # The third candle must close higher than the second
    if df.iloc[-1]['Close'] <= df.iloc[-2]['Close']:
        return False
        
    return True

def is_three_outside_down(df: pd.DataFrame) -> bool:
    """
    Detect three outside down pattern: bullish candle, followed by bearish engulfing,
    followed by a third bearish candle
    """
    if len(df) < 3:
        return False
        
    # Check if the first two candles form a bearish engulfing
    if not is_bearish_engulfing(df.iloc[-3:-1]):
        return False
        
    # The third candle must be bearish
    if df.iloc[-1]['Close'] >= df.iloc[-1]['Open']:
        return False
        
    # The third candle must close lower than the second
    if df.iloc[-1]['Close'] >= df.iloc[-2]['Close']:
        return False
        
    return True

def is_abandoned_baby_bullish(df: pd.DataFrame) -> bool:
    """
    Detect bullish abandoned baby: bearish candle, followed by a doji that gaps down,
    followed by a bullish candle that gaps up
    """
    if len(df) < 3:
        return False
        
    first = df.iloc[-3]
    middle = df.iloc[-2]
    last = df.iloc[-1]
    
    # First candle must be bearish
    if first['Close'] >= first['Open']:
        return False
        
    # Middle candle must be a doji
    if not is_doji(df.iloc[-2:-1]):
        return False
        
    # Last candle must be bullish
    if last['Close'] <= last['Open']:
        return False
        
    # Gap down between first and middle
    if middle['High'] >= first['Low']:
        return False
        
    # Gap up between middle and last
    if last['Low'] <= middle['High']:
        return False
        
    return True

def is_abandoned_baby_bearish(df: pd.DataFrame) -> bool:
    """
    Detect bearish abandoned baby: bullish candle, followed by a doji that gaps up,
    followed by a bearish candle that gaps down
    """
    if len(df) < 3:
        return False
        
    first = df.iloc[-3]
    middle = df.iloc[-2]
    last = df.iloc[-1]
    
    # First candle must be bullish
    if first['Close'] <= first['Open']:
        return False
        
    # Middle candle must be a doji
    if not is_doji(df.iloc[-2:-1]):
        return False
        
    # Last candle must be bearish
    if last['Close'] >= last['Open']:
        return False
        
    # Gap up between first and middle
    if middle['Low'] <= first['High']:
        return False
        
    # Gap down between middle and last
    if last['High'] >= middle['Low']:
        return False
        
    return True

def is_rising_three_methods(df: pd.DataFrame) -> bool:
    """
    Detect rising three methods: bullish candle followed by three small bearish candles
    contained within the range of the first, followed by a bullish candle
    """
    if len(df) < 5:
        return False
        
    first = df.iloc[-5]
    middle1 = df.iloc[-4]
    middle2 = df.iloc[-3]
    middle3 = df.iloc[-2]
    last = df.iloc[-1]
    
    # First candle must be bullish
    if first['Close'] <= first['Open']:
        return False
        
    # Last candle must be bullish
    if last['Close'] <= last['Open']:
        return False
        
    # Middle candles must be bearish
    if (middle1['Close'] >= middle1['Open'] or 
        middle2['Close'] >= middle2['Open'] or 
        middle3['Close'] >= middle3['Open']):
        return False
        
    # Middle candles must be contained within the range of the first
    if (middle1['High'] > first['High'] or middle1['Low'] < first['Low'] or
        middle2['High'] > first['High'] or middle2['Low'] < first['Low'] or
        middle3['High'] > first['High'] or middle3['Low'] < first['Low']):
        return False
        
    # Last candle must close above first candle's close
    if last['Close'] <= first['Close']:
        return False
        
    return True

def is_falling_three_methods(df: pd.DataFrame) -> bool:
    """
    Detect falling three methods: bearish candle followed by three small bullish candles
    contained within the range of the first, followed by a bearish candle
    """
    if len(df) < 5:
        return False
        
    first = df.iloc[-5]
    middle1 = df.iloc[-4]
    middle2 = df.iloc[-3]
    middle3 = df.iloc[-2]
    last = df.iloc[-1]
    
    # First candle must be bearish
    if first['Close'] >= first['Open']:
        return False
        
    # Last candle must be bearish
    if last['Close'] >= last['Open']:
        return False
        
    # Middle candles must be bullish
    if (middle1['Close'] <= middle1['Open'] or 
        middle2['Close'] <= middle2['Open'] or 
        middle3['Close'] <= middle3['Open']):
        return False
        
    # Middle candles must be contained within the range of the first
    if (middle1['High'] > first['High'] or middle1['Low'] < first['Low'] or
        middle2['High'] > first['High'] or middle2['Low'] < first['Low'] or
        middle3['High'] > first['High'] or middle3['Low'] < first['Low']):
        return False
        
    # Last candle must close below first candle's close
    if last['Close'] >= first['Close']:
        return False
        
    return True

# === Chart Pattern Detection Implementations ===

def find_peaks_and_troughs(prices: np.ndarray, window: int = 5) -> Tuple[List[int], List[int]]:
    """
    Identifies peaks (local maxima) and troughs (local minima) in price data
    
    Args:
        prices: Array of price values
        window: Window size for peak/trough detection
        
    Returns:
        Tuple of (peak_indices, trough_indices)
    """
    peaks = []
    troughs = []
    
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

def is_head_and_shoulders(df: pd.DataFrame) -> bool:
    """
    Detect head and shoulders pattern: three peaks with the middle one higher
    """
    if len(df) < 20:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        
        # Find peaks
        peaks, _ = find_peaks_and_troughs(prices)
        
        if len(peaks) < 3:
            return False
            
        # Look at last 3 peaks
        if len(peaks) >= 3:
            last_three_peaks = sorted(peaks[-3:])
            
            # Extract peak heights
            peak_heights = [prices[i] for i in last_three_peaks]
            
            # Check for head and shoulders pattern
            # Middle peak (head) should be higher than both shoulders
            if peak_heights[1] > peak_heights[0] and peak_heights[1] > peak_heights[2]:
                # Shoulders should be at approximately the same height (within 10%)
                shoulder_diff_pct = abs(peak_heights[0] - peak_heights[2]) / peak_heights[0]
                if shoulder_diff_pct < 0.10:
                    # Should be enough distance between peaks
                    if last_three_peaks[1] - last_three_peaks[0] >= 3 and last_three_peaks[2] - last_three_peaks[1] >= 3:
                        return True
        
        return False
    except Exception as e:
        logging.warning(f"Head and shoulders detection error: {e}")
        return False

def is_inverse_head_and_shoulders(df: pd.DataFrame) -> bool:
    """
    Detect inverse head and shoulders pattern: three troughs with the middle one lower
    """
    if len(df) < 20:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        
        # Find troughs
        _, troughs = find_peaks_and_troughs(prices)
        
        if len(troughs) < 3:
            return False
            
        # Look at last 3 troughs
        if len(troughs) >= 3:
            last_three_troughs = sorted(troughs[-3:])
            
            # Extract trough depths
            trough_depths = [prices[i] for i in last_three_troughs]
            
            # Check for inverse head and shoulders pattern
            # Middle trough (head) should be lower than both shoulders
            if trough_depths[1] < trough_depths[0] and trough_depths[1] < trough_depths[2]:
                # Shoulders should be at approximately the same height (within 10%)
                shoulder_diff_pct = abs(trough_depths[0] - trough_depths[2]) / trough_depths[0]
                if shoulder_diff_pct < 0.10:
                    # Should be enough distance between troughs
                    if last_three_troughs[1] - last_three_troughs[0] >= 3 and last_three_troughs[2] - last_three_troughs[1] >= 3:
                        return True
        
        return False
    except Exception as e:
        logging.warning(f"Inverse head and shoulders detection error: {e}")
        return False

def is_double_top(df: pd.DataFrame) -> bool:
    """
    Detect double top pattern: two peaks at similar price levels with a trough in between
    """
    if len(df) < 15:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        if len(peaks) < 2 or len(troughs) < 1:
            return False
            
        # Look at last 2 peaks
        if len(peaks) >= 2:
            last_two_peaks = sorted(peaks[-2:])
            
            # Extract peak heights
            peak_heights = [prices[i] for i in last_two_peaks]
            
            # Check if peaks are at similar height (within 3%)
            peak_diff_pct = abs(peak_heights[0] - peak_heights[1]) / peak_heights[0]
            if peak_diff_pct > 0.03:
                return False
            
            # Find troughs between the peaks
            mid_troughs = [t for t in troughs if last_two_peaks[0] < t < last_two_peaks[1]]
            if not mid_troughs:
                return False
                
            # Get the lowest trough between peaks
            mid_trough_idx = mid_troughs[np.argmin([prices[t] for t in mid_troughs])]
            mid_trough_val = prices[mid_trough_idx]
            
            # Measure the depth of the trough relative to the peaks
            avg_peak_height = (peak_heights[0] + peak_heights[1]) / 2
            trough_depth_pct = (avg_peak_height - mid_trough_val) / avg_peak_height
            
            # Trough should be at least 5% below the peaks
            if trough_depth_pct < 0.05:
                return False
                
            # Should be enough distance between peaks (at least 5 candles)
            if last_two_peaks[1] - last_two_peaks[0] < 5:
                return False
                
            return True
        
        return False
    except Exception as e:
        logging.warning(f"Double top detection error: {e}")
        return False

def is_double_bottom(df: pd.DataFrame) -> bool:
    """
    Detect double bottom pattern: two troughs at similar price levels with a peak in between
    """
    if len(df) < 15:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        if len(troughs) < 2 or len(peaks) < 1:
            return False
            
        # Look at last 2 troughs
        if len(troughs) >= 2:
            last_two_troughs = sorted(troughs[-2:])
            
            # Extract trough depths
            trough_depths = [prices[i] for i in last_two_troughs]
            
            # Check if troughs are at similar height (within 3%)
            trough_diff_pct = abs(trough_depths[0] - trough_depths[1]) / trough_depths[0]
            if trough_diff_pct > 0.03:
                return False
            
            # Find peaks between the troughs
            mid_peaks = [p for p in peaks if last_two_troughs[0] < p < last_two_troughs[1]]
            if not mid_peaks:
                return False
                
            # Get the highest peak between troughs
            mid_peak_idx = mid_peaks[np.argmax([prices[p] for p in mid_peaks])]
            mid_peak_val = prices[mid_peak_idx]
            
            # Measure the height of the peak relative to the troughs
            avg_trough_depth = (trough_depths[0] + trough_depths[1]) / 2
            peak_height_pct = (mid_peak_val - avg_trough_depth) / avg_trough_depth
            
            # Peak should be at least 5% above the troughs
            if peak_height_pct < 0.05:
                return False
                
            # Should be enough distance between troughs (at least 5 candles)
            if last_two_troughs[1] - last_two_troughs[0] < 5:
                return False
                
            return True
        
        return False
    except Exception as e:
        logging.warning(f"Double bottom detection error: {e}")
        return False

def fit_trendline(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit a linear trendline (y = mx + b) to data points
    
    Args:
        x: Array of x-coordinates
        y: Array of y-coordinates
        
    Returns:
        Tuple of (slope, intercept)
    """
    if len(x) < 2:
        return 0, 0
        
    # Calculate trendline using polyfit
    m, b = np.polyfit(x, y, 1)
    return m, b

def is_ascending_channel(df: pd.DataFrame) -> bool:
    """
    Detect ascending channel: rising support and resistance lines with parallel slopes
    """
    if len(df) < 12:
        return False
        
    try:
        # Get high and low prices
        highs = df['High'].values
        lows = df['Low'].values
        x = np.arange(len(highs))
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(df['Close'].values)
        
        # Need at least 2 peaks and 2 troughs to form channel
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # Use the peaks to create resistance line
        peak_x = np.array([peaks[-2], peaks[-1]])
        peak_y = np.array([highs[peaks[-2]], highs[peaks[-1]]])
        res_slope, res_intercept = fit_trendline(peak_x, peak_y)
        
        # Use the troughs to create support line
        trough_x = np.array([troughs[-2], troughs[-1]])
        trough_y = np.array([lows[troughs[-2]], lows[troughs[-1]]])
        sup_slope, sup_intercept = fit_trendline(trough_x, trough_y)
        
        # For ascending channel:
        # 1. Both slopes must be positive
        if res_slope <= 0 or sup_slope <= 0:
            return False
            
        # 2. Slopes should be similar (parallel lines)
        slope_diff_pct = abs(res_slope - sup_slope) / sup_slope
        if slope_diff_pct > 0.3:  # Allow up to 30% difference
            return False
            
        # 3. Channel should be wide enough
        channel_width_pct = (res_intercept - sup_intercept) / sup_intercept
        if channel_width_pct < 0.02:  # At least 2% wide
            return False
            
        # 4. Price should be touching or near the trendlines at multiple points
        resistance_line = res_slope * x + res_intercept
        support_line = sup_slope * x + sup_intercept
        
        high_near_resistance = sum(1 for i in range(len(highs)) if abs(highs[i] - resistance_line[i]) / resistance_line[i] < 0.01)
        low_near_support = sum(1 for i in range(len(lows)) if abs(lows[i] - support_line[i]) / support_line[i] < 0.01)
        
        if high_near_resistance < 2 or low_near_support < 2:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Ascending channel detection error: {e}")
        return False

def is_descending_channel(df: pd.DataFrame) -> bool:
    """
    Detect descending channel: falling support and resistance lines with parallel slopes
    """
    if len(df) < 12:
        return False
        
    try:
        # Get high and low prices
        highs = df['High'].values
        lows = df['Low'].values
        x = np.arange(len(highs))
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(df['Close'].values)
        
        # Need at least 2 peaks and 2 troughs to form channel
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # Use the peaks to create resistance line
        peak_x = np.array([peaks[-2], peaks[-1]])
        peak_y = np.array([highs[peaks[-2]], highs[peaks[-1]]])
        res_slope, res_intercept = fit_trendline(peak_x, peak_y)
        
        # Use the troughs to create support line
        trough_x = np.array([troughs[-2], troughs[-1]])
        trough_y = np.array([lows[troughs[-2]], lows[troughs[-1]]])
        sup_slope, sup_intercept = fit_trendline(trough_x, trough_y)
        
        # For descending channel:
        # 1. Both slopes must be negative
        if res_slope >= 0 or sup_slope >= 0:
            return False
            
        # 2. Slopes should be similar (parallel lines)
        slope_diff_pct = abs(res_slope - sup_slope) / abs(sup_slope)
        if slope_diff_pct > 0.3:  # Allow up to 30% difference
            return False
            
        # 3. Channel should be wide enough
        channel_width_pct = (res_intercept - sup_intercept) / sup_intercept
        if channel_width_pct < 0.02:  # At least 2% wide
            return False
            
        # 4. Price should be touching or near the trendlines at multiple points
        resistance_line = res_slope * x + res_intercept
        support_line = sup_slope * x + sup_intercept
        
        high_near_resistance = sum(1 for i in range(len(highs)) if abs(highs[i] - resistance_line[i]) / resistance_line[i] < 0.01)
        low_near_support = sum(1 for i in range(len(lows)) if abs(lows[i] - support_line[i]) / support_line[i] < 0.01)
        
        if high_near_resistance < 2 or low_near_support < 2:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Descending channel detection error: {e}")
        return False

def is_horizontal_channel(df: pd.DataFrame) -> bool:
    """
    Detect horizontal channel: price moving between horizontal support and resistance
    """
    if len(df) < 12:
        return False
        
    try:
        # Get high and low prices
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(df['Close'].values)
        
        # Need at least 2 peaks and 2 troughs to form channel
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # Get recent peak heights
        peak_heights = [highs[i] for i in peaks[-3:]] if len(peaks) >= 3 else [highs[i] for i in peaks]
        
        # Get recent trough depths
        trough_depths = [lows[i] for i in troughs[-3:]] if len(troughs) >= 3 else [lows[i] for i in troughs]
        
        # Calculate average and std deviation for both
        avg_peak = np.mean(peak_heights)
        avg_trough = np.mean(trough_depths)
        std_peak = np.std(peak_heights)
        std_trough = np.std(trough_depths)
        
        # For a horizontal channel:
        # 1. Peaks and troughs should have low standard deviation (horizontal lines)
        if std_peak / avg_peak > 0.05 or std_trough / avg_trough > 0.05:
            return False
            
        # 2. Channel height should be significant
        channel_height = avg_peak - avg_trough
        channel_height_pct = channel_height / avg_trough
        
        if channel_height_pct < 0.02:  # At least 2% high
            return False
            
        # 3. Price should have touched both boundaries multiple times
        high_near_resistance = sum(1 for h in highs if abs(h - avg_peak) / avg_peak < 0.01)
        low_near_support = sum(1 for l in lows if abs(l - avg_trough) / avg_trough < 0.01)
        
        if high_near_resistance < 2 or low_near_support < 2:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Horizontal channel detection error: {e}")
        return False

def is_rounding_top(df: pd.DataFrame) -> bool:
    """
    Detect rounding top (saucer top): gradually rising then falling price forming an inverted 'U'
    """
    if len(df) < 20:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        x = np.arange(len(prices))
        
        # Fit a quadratic curve to the data (y = ax^2 + bx + c)
        coeffs = np.polyfit(x, prices, 2)
        a, b, c = coeffs
        
        # For a rounding top, 'a' coefficient must be negative (concave down)
        if a >= 0:
            return False
            
        # Calculate the fitted curve
        fitted = a * x**2 + b * x + c
        
        # Calculate the mean squared error
        mse = np.mean((prices - fitted)**2)
        
        # Calculate the total variation
        total_var = np.var(prices)
        
        # Calculate the R-squared value: 1 - (mse / total_var)
        r_squared = 1 - (mse / total_var)
        
        # R-squared must be high enough to indicate a good fit
        if r_squared < 0.7:
            return False
            
        # Check if the peak is near the middle
        peak_x = -b / (2 * a)
        if peak_x < 0.3 * len(prices) or peak_x > 0.7 * len(prices):
            return False
            
        # Check if current price is significantly below the peak
        peak_price = a * peak_x**2 + b * peak_x + c
        current_price = prices[-1]
        
        if current_price > peak_price * 0.95:  # Should be at least 5% below peak
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Rounding top detection error: {e}")
        return False

def is_rounding_bottom(df: pd.DataFrame) -> bool:
    """
    Detect rounding bottom (saucer bottom): gradually falling then rising price forming a 'U'
    """
    if len(df) < 20:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        x = np.arange(len(prices))
        
        # Fit a quadratic curve to the data (y = ax^2 + bx + c)
        coeffs = np.polyfit(x, prices, 2)
        a, b, c = coeffs
        
        # For a rounding bottom, 'a' coefficient must be positive (concave up)
        if a <= 0:
            return False
            
        # Calculate the fitted curve
        fitted = a * x**2 + b * x + c
        
        # Calculate the mean squared error
        mse = np.mean((prices - fitted)**2)
        
        # Calculate the total variation
        total_var = np.var(prices)
        
        # Calculate the R-squared value: 1 - (mse / total_var)
        r_squared = 1 - (mse / total_var)
        
        # R-squared must be high enough to indicate a good fit
        if r_squared < 0.7:
            return False
            
        # Check if the trough is near the middle
        trough_x = -b / (2 * a)
        if trough_x < 0.3 * len(prices) or trough_x > 0.7 * len(prices):
            return False
            
        # Check if current price is significantly above the trough
        trough_price = a * trough_x**2 + b * trough_x + c
        current_price = prices[-1]
        
        if current_price < trough_price * 1.05:  # Should be at least 5% above trough
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Rounding bottom detection error: {e}")
        return False

def is_v_bottom(df: pd.DataFrame) -> bool:
    """
    Detect V-bottom: sharp reversal from downtrend to uptrend
    """
    if len(df) < 10:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        
        # Split the data into two halves
        mid_point = len(prices) // 2
        first_half = prices[:mid_point]
        second_half = prices[mid_point:]
        
        # Check if first half is falling and second half is rising
        first_half_trend = np.polyfit(np.arange(len(first_half)), first_half, 1)[0]
        second_half_trend = np.polyfit(np.arange(len(second_half)), second_half, 1)[0]
        
        # For a V-bottom, first half must be falling and second half must be rising
        if first_half_trend >= 0 or second_half_trend <= 0:
            return False
            
        # Check if the lowest point is near the middle
        lowest_idx = np.argmin(prices)
        if abs(lowest_idx - mid_point) > 2:  # Allow some flexibility
            return False
            
        # Check if the reversal is sharp (steep enough slopes)
        if abs(first_half_trend) < 0.005 or abs(second_half_trend) < 0.005:
            return False
            
        # Check if price has recovered significantly
        recovery_pct = (prices[-1] - prices[lowest_idx]) / prices[lowest_idx]
        if recovery_pct < 0.05:  # Should recover by at least 5%
            return False
            
        return True
    except Exception as e:
        logging.warning(f"V-bottom detection error: {e}")
        return False

def is_inverted_v_top(df: pd.DataFrame) -> bool:
    """
    Detect inverted V-top: sharp reversal from uptrend to downtrend
    """
    if len(df) < 10:
        return False
        
    try:
        # Get closing prices
        prices = df['Close'].values
        
        # Split the data into two halves
        mid_point = len(prices) // 2
        first_half = prices[:mid_point]
        second_half = prices[mid_point:]
        
        # Check if first half is rising and second half is falling
        first_half_trend = np.polyfit(np.arange(len(first_half)), first_half, 1)[0]
        second_half_trend = np.polyfit(np.arange(len(second_half)), second_half, 1)[0]
        
        # For an inverted V-top, first half must be rising and second half must be falling
        if first_half_trend <= 0 or second_half_trend >= 0:
            return False
            
        # Check if the highest point is near the middle
        highest_idx = np.argmax(prices)
        if abs(highest_idx - mid_point) > 2:  # Allow some flexibility
            return False
            
        # Check if the reversal is sharp (steep enough slopes)
        if abs(first_half_trend) < 0.005 or abs(second_half_trend) < 0.005:
            return False
            
        # Check if price has declined significantly
        decline_pct = (prices[highest_idx] - prices[-1]) / prices[highest_idx]
        if decline_pct < 0.05:  # Should decline by at least 5%
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Inverted V-top detection error: {e}")
        return False

def is_key_reversal(df: pd.DataFrame) -> bool:
    """
    Detect key reversal: single candlestick that marks a turning point in trend
    """
    if len(df) < 2:
        return False
        
    try:
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # For bullish key reversal:
        if prev['Close'] < prev['Open']:  # Previous candle is bearish
            # Current candle opens below previous low but closes above previous close
            if curr['Open'] < prev['Low'] and curr['Close'] > prev['Close']:
                return True
                
        # For bearish key reversal:
        if prev['Close'] > prev['Open']:  # Previous candle is bullish
            # Current candle opens above previous high but closes below previous close
            if curr['Open'] > prev['High'] and curr['Close'] < prev['Close']:
                return True
                
        return False
    except Exception as e:
        logging.warning(f"Key reversal detection error: {e}")
        return False

def is_island_reversal(df: pd.DataFrame) -> bool:
    """
    Detect island reversal: gap in one direction followed by sideways movement then gap in opposite direction
    """
    if len(df) < 5:
        return False
        
    try:
        # Check for bullish island reversal
        # Gap down
        gap_down = df.iloc[-5]['Low'] > df.iloc[-4]['High']
        
        # Sideways movement (contained within a narrow range)
        middle_high = max(df.iloc[-4:-1]['High'].values)
        middle_low = min(df.iloc[-4:-1]['Low'].values)
        sideways = (middle_high - middle_low) / middle_low < 0.03  # Less than 3% range
        
        # Gap up
        gap_up = df.iloc[-1]['Low'] > middle_high
        
        if gap_down and sideways and gap_up:
            return True
            
        # Check for bearish island reversal
        # Gap up
        gap_up = df.iloc[-5]['High'] < df.iloc[-4]['Low']
        
        # Sideways movement
        middle_high = max(df.iloc[-4:-1]['High'].values)
        middle_low = min(df.iloc[-4:-1]['Low'].values)
        sideways = (middle_high - middle_low) / middle_low < 0.03  # Less than 3% range
        
        # Gap down
        gap_down = df.iloc[-1]['High'] < middle_low
        
        if gap_up and sideways and gap_down:
            return True
            
        return False
    except Exception as e:
        logging.warning(f"Island reversal detection error: {e}")
        return False

def is_diamond_top(df: pd.DataFrame) -> bool:
    """
    Detect diamond top: price action forms a diamond shape at the top of an uptrend
    """
    if len(df) < 15:
        return False
        
    try:
        # Get high and low prices
        highs = df['High'].values
        lows = df['Low'].values
        
        # Calculate price range for each candle
        ranges = highs - lows
        
        # For a diamond pattern:
        # 1. Ranges should increase, then decrease
        mid_point = len(ranges) // 2
        first_half_ranges = ranges[:mid_point]
        second_half_ranges = ranges[mid_point:]
        
        # Check if ranges are expanding in first half
        x_first = np.arange(len(first_half_ranges))
        first_half_trend = np.polyfit(x_first, first_half_ranges, 1)[0]
        
        # Check if ranges are contracting in second half
        x_second = np.arange(len(second_half_ranges))
        second_half_trend = np.polyfit(x_second, second_half_ranges, 1)[0]
        
        # For a diamond top, ranges should expand then contract
        if first_half_trend <= 0 or second_half_trend >= 0:
            return False
            
        # 2. Price should be trending up before the pattern
        if len(df) >= 20:
            prior_trend = np.polyfit(np.arange(5), df.iloc[-20:-15]['Close'].values, 1)[0]
            if prior_trend <= 0:
                return False
        else:
            return False
            
        # 3. Check if current price is below the average of the pattern
        avg_price = np.mean(df.iloc[-15:]['Close'].values)
        if df.iloc[-1]['Close'] >= avg_price:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Diamond top detection error: {e}")
        return False

# Fix for is_diamond_bottom function in pattern_recognition.py
def is_diamond_bottom(df: pd.DataFrame) -> bool:
    """
    Detect diamond bottom: price action forms a diamond shape at the bottom of a downtrend
    """
    if len(df) < 15:
        return False
        
    try:
        # Get high and low prices
        highs = df['High'].values
        lows = df['Low'].values
        
        # Calculate price range for each candle
        ranges = highs - lows
        
        # For a diamond pattern:
        # 1. Ranges should increase, then decrease
        mid_point = len(ranges) // 2
        first_half_ranges = ranges[:mid_point]
        second_half_ranges = ranges[mid_point:]
        
        # Check if ranges are expanding in first half
        x_first = np.arange(len(first_half_ranges))
        first_half_trend = np.polyfit(x_first, first_half_ranges, 1)[0]
        
        # Check if ranges are contracting in second half
        x_second = np.arange(len(second_half_ranges))
        second_half_trend = np.polyfit(x_second, second_half_ranges, 1)[0]
        
        # For a diamond bottom, ranges should expand then contract
        if first_half_trend <= 0 or second_half_trend >= 0:
            return False
            
        # 2. Price should be trending down before the pattern
        if len(df) >= 20:
            prior_trend = np.polyfit(np.arange(5), df.iloc[-20:-15]['Close'].values, 1)[0]
            if prior_trend >= 0:
                return False
        else:
            return False
            
        # 3. Check if current price is above the average of the pattern
        avg_price = np.mean(df.iloc[-15:]['Close'].values)
        if df.iloc[-1]['Close'] <= avg_price:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Diamond bottom detection error: {e}")
        return False

def is_coil_pattern(df: pd.DataFrame) -> bool:
    """
    Detect coil pattern (symmetrical triangle): converging trendlines with price narrowing
    """
    if len(df) < 10:
        return False
        
    try:
        # Get high and low prices
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find local peaks and troughs in the last n candles
        peaks, troughs = find_peaks_and_troughs(df['Close'].values)
        
        # Need at least 2 peaks and 2 troughs to form trendlines
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # Create upper trendline using peaks
        peak_x = np.array(peaks[-2:])
        peak_y = np.array([highs[i] for i in peaks[-2:]])
        upper_slope, upper_intercept = fit_trendline(peak_x, peak_y)
        
        # Create lower trendline using troughs
        trough_x = np.array(troughs[-2:])
        trough_y = np.array([lows[i] for i in troughs[-2:]])
        lower_slope, lower_intercept = fit_trendline(trough_x, trough_y)
        
        # For a coil pattern:
        # 1. Upper slope should be negative and lower slope should be positive
        if upper_slope >= 0 or lower_slope <= 0:
            return False
            
        # 2. Trendlines should be converging
        # Calculate the x-coordinate of the intersection point
        intersection_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)
        
        # Intersection should be ahead of current position
        if intersection_x <= len(df) - 1:
            return False
            
        # 3. Intersection should not be too far in the future (3-4x the current length at max)
        if intersection_x > 4 * len(df):
            return False
            
        # 4. Price range should be narrowing
        first_half_range = np.mean(highs[:len(highs)//2] - lows[:len(lows)//2])
        second_half_range = np.mean(highs[len(highs)//2:] - lows[len(lows)//2:])
        
        if second_half_range >= first_half_range:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Coil pattern detection error: {e}")
        return False

def is_gartley_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Gartley harmonic pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 2 peaks and 2 troughs
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # For a bullish Gartley pattern:
        # Point X (start) - trough
        # Point A (peak) - peak
        # Point B (higher trough than X) - trough
        # Point C (lower peak than A) - peak
        # Point D (higher trough than B, but lower than A) - trough
        
        # We need to identify these points
        # First, try to identify point A (should be a significant peak)
        if len(peaks) < 2 or len(troughs) < 3:
            return False
            
        # Last peak should be point C
        point_c_idx = peaks[-1]
        point_c = prices[point_c_idx]
        
        # Last trough should be point D
        point_d_idx = troughs[-1]
        point_d = prices[point_d_idx]
        
        # Point A should be the peak before C
        point_a_idx = peaks[-2]
        point_a = prices[point_a_idx]
        
        # Point B should be the trough before C
        point_b_idx = [t for t in troughs if t > point_a_idx and t < point_c_idx]
        if not point_b_idx:
            return False
        point_b_idx = point_b_idx[-1]
        point_b = prices[point_b_idx]
        
        # Point X should be the trough before A
        point_x_idx = [t for t in troughs if t < point_a_idx]
        if not point_x_idx:
            return False
        point_x_idx = point_x_idx[-1]
        point_x = prices[point_x_idx]
        
        # Now check the Gartley ratios
        # AB/XA should be around 0.618
        xa = point_a - point_x
        ab = point_a - point_b
        if xa == 0:  # Avoid division by zero
            return False
        ab_xa_ratio = ab / xa
        if abs(ab_xa_ratio - 0.618) > 0.1:  # Allow some flexibility
            return False
            
        # BC/AB should be around 0.382 or 0.886
        bc = point_c - point_b
        if ab == 0:  # Avoid division by zero
            return False
        bc_ab_ratio = bc / ab
        if not (abs(bc_ab_ratio - 0.382) < 0.1 or abs(bc_ab_ratio - 0.886) < 0.1):
            return False
            
        # CD/BC should be around 1.272 or 1.618
        cd = point_c - point_d
        if bc == 0:  # Avoid division by zero
            return False
        cd_bc_ratio = cd / bc
        if not (abs(cd_bc_ratio - 1.272) < 0.1 or abs(cd_bc_ratio - 1.618) < 0.1):
            return False
            
        # XA/AD should be around 0.786
        ad = point_a - point_d
        if ad == 0:  # Avoid division by zero
            return False
        xa_ad_ratio = xa / ad
        if abs(xa_ad_ratio - 0.786) > 0.1:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Gartley pattern detection error: {e}")
        return False

def is_bat_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Bat harmonic pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 2 peaks and 2 troughs
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # For a bullish Bat pattern:
        # Point X (start) - trough
        # Point A (peak) - peak
        # Point B (higher trough than X) - trough
        # Point C (lower peak than A) - peak
        # Point D (higher trough than B, but lower than A) - trough
        
        # We need to identify these points
        # First, try to identify point A (should be a significant peak)
        if len(peaks) < 2 or len(troughs) < 3:
            return False
            
        # Last peak should be point C
        point_c_idx = peaks[-1]
        point_c = prices[point_c_idx]
        
        # Last trough should be point D
        point_d_idx = troughs[-1]
        point_d = prices[point_d_idx]
        
        # Point A should be the peak before C
        point_a_idx = peaks[-2]
        point_a = prices[point_a_idx]
        
        # Point B should be the trough before C
        point_b_idx = [t for t in troughs if t > point_a_idx and t < point_c_idx]
        if not point_b_idx:
            return False
        point_b_idx = point_b_idx[-1]
        point_b = prices[point_b_idx]
        
        # Point X should be the trough before A
        point_x_idx = [t for t in troughs if t < point_a_idx]
        if not point_x_idx:
            return False
        point_x_idx = point_x_idx[-1]
        point_x = prices[point_x_idx]
        
        # Now check the Bat ratios
        # AB/XA should be around 0.382-0.5
        xa = point_a - point_x
        ab = point_a - point_b
        if xa == 0:  # Avoid division by zero
            return False
        ab_xa_ratio = ab / xa
        if not (0.382 - 0.1 < ab_xa_ratio < 0.5 + 0.1):  # Allow some flexibility
            return False
            
        # BC/AB should be around 0.382-0.886
        bc = point_c - point_b
        if ab == 0:  # Avoid division by zero
            return False
        bc_ab_ratio = bc / ab
        if not (0.382 - 0.1 < bc_ab_ratio < 0.886 + 0.1):
            return False
            
        # CD/BC should be around 1.618-2.618
        cd = point_c - point_d
        if bc == 0:  # Avoid division by zero
            return False
        cd_bc_ratio = cd / bc
        if not (1.618 - 0.1 < cd_bc_ratio < 2.618 + 0.1):
            return False
            
        # AD/XA should be around 0.886
        ad = point_a - point_d
        if xa == 0:  # Avoid division by zero
            return False
        ad_xa_ratio = ad / xa
        if abs(ad_xa_ratio - 0.886) > 0.1:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Bat pattern detection error: {e}")
        return False

def is_crab_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Crab harmonic pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 2 peaks and 2 troughs
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # For a bullish Crab pattern:
        # Point X (start) - trough
        # Point A (peak) - peak
        # Point B (higher trough than X) - trough
        # Point C (lower peak than A) - peak
        # Point D (higher trough than B, but lower than A) - trough
        
        # We need to identify these points
        # First, try to identify point A (should be a significant peak)
        if len(peaks) < 2 or len(troughs) < 3:
            return False
            
        # Last peak should be point C
        point_c_idx = peaks[-1]
        point_c = prices[point_c_idx]
        
        # Last trough should be point D
        point_d_idx = troughs[-1]
        point_d = prices[point_d_idx]
        
        # Point A should be the peak before C
        point_a_idx = peaks[-2]
        point_a = prices[point_a_idx]
        
        # Point B should be the trough before C
        point_b_idx = [t for t in troughs if t > point_a_idx and t < point_c_idx]
        if not point_b_idx:
            return False
        point_b_idx = point_b_idx[-1]
        point_b = prices[point_b_idx]
        
        # Point X should be the trough before A
        point_x_idx = [t for t in troughs if t < point_a_idx]
        if not point_x_idx:
            return False
        point_x_idx = point_x_idx[-1]
        point_x = prices[point_x_idx]
        
        # Now check the Crab ratios
        # AB/XA should be around 0.382-0.618
        xa = point_a - point_x
        ab = point_a - point_b
        if xa == 0:  # Avoid division by zero
            return False
        ab_xa_ratio = ab / xa
        if not (0.382 - 0.1 < ab_xa_ratio < 0.618 + 0.1):  # Allow some flexibility
            return False
            
        # BC/AB should be around 0.382-0.886
        bc = point_c - point_b
        if ab == 0:  # Avoid division by zero
            return False
        bc_ab_ratio = bc / ab
        if not (0.382 - 0.1 < bc_ab_ratio < 0.886 + 0.1):
            return False
            
        # CD/BC should be around 2.24-3.618
        cd = point_c - point_d
        if bc == 0:  # Avoid division by zero
            return False
        cd_bc_ratio = cd / bc
        if not (2.24 - 0.1 < cd_bc_ratio < 3.618 + 0.1):
            return False
            
        # AD/XA should be around 1.618
        ad = point_a - point_d
        if xa == 0:  # Avoid division by zero
            return False
        ad_xa_ratio = ad / xa
        if abs(ad_xa_ratio - 1.618) > 0.1:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Crab pattern detection error: {e}")
        return False

def is_butterfly_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Butterfly harmonic pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 2 peaks and 2 troughs
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # For a bullish Butterfly pattern:
        # Point X (start) - trough
        # Point A (peak) - peak
        # Point B (higher trough than X) - trough
        # Point C (lower peak than A) - peak
        # Point D (higher trough than B, but lower than A) - trough
        
        # We need to identify these points
        # First, try to identify point A (should be a significant peak)
        if len(peaks) < 2 or len(troughs) < 3:
            return False
            
        # Last peak should be point C
        point_c_idx = peaks[-1]
        point_c = prices[point_c_idx]
        
        # Last trough should be point D
        point_d_idx = troughs[-1]
        point_d = prices[point_d_idx]
        
        # Point A should be the peak before C
        point_a_idx = peaks[-2]
        point_a = prices[point_a_idx]
        
        # Point B should be the trough before C
        point_b_idx = [t for t in troughs if t > point_a_idx and t < point_c_idx]
        if not point_b_idx:
            return False
        point_b_idx = point_b_idx[-1]
        point_b = prices[point_b_idx]
        
        # Point X should be the trough before A
        point_x_idx = [t for t in troughs if t < point_a_idx]
        if not point_x_idx:
            return False
        point_x_idx = point_x_idx[-1]
        point_x = prices[point_x_idx]
        
        # Now check the Butterfly ratios
        # AB/XA should be around 0.786
        xa = point_a - point_x
        ab = point_a - point_b
        if xa == 0:  # Avoid division by zero
            return False
        ab_xa_ratio = ab / xa
        if abs(ab_xa_ratio - 0.786) > 0.1:  # Allow some flexibility
            return False
            
        # BC/AB should be around 0.382-0.886
        bc = point_c - point_b
        if ab == 0:  # Avoid division by zero
            return False
        bc_ab_ratio = bc / ab
        if not (0.382 - 0.1 < bc_ab_ratio < 0.886 + 0.1):
            return False
            
        # CD/BC should be around 1.618-2.24
        cd = point_c - point_d
        if bc == 0:  # Avoid division by zero
            return False
        cd_bc_ratio = cd / bc
        if not (1.618 - 0.1 < cd_bc_ratio < 2.24 + 0.1):
            return False
            
        # AD/XA should be around 1.27-1.618
        ad = point_a - point_d
        if xa == 0:  # Avoid division by zero
            return False
        ad_xa_ratio = ad / xa
        if not (1.27 - 0.1 < ad_xa_ratio < 1.618 + 0.1):
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Butterfly pattern detection error: {e}")
        return False

def is_shark_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Shark harmonic pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 2 peaks and 2 troughs
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # For a Shark pattern:
        # Point O (start) - origin
        # Point X (first extreme) - peak/trough
        # Point A (retracement of OX) - trough/peak
        # Point B (extension of XA) - peak/trough
        # Point C (key reversal) - trough/peak
        
        # We need to identify these points
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # For simplicity, assume bullish pattern (we can later add bearish check)
        # Point O should be a significant trough
        point_o_idx = troughs[0] if troughs[0] < peaks[0] else 0
        point_o = prices[point_o_idx]
        
        # Point X should be a peak after O
        point_x_idx = [p for p in peaks if p > point_o_idx]
        if not point_x_idx:
            return False
        point_x_idx = point_x_idx[0]
        point_x = prices[point_x_idx]
        
        # Point A should be a trough after X
        point_a_idx = [t for t in troughs if t > point_x_idx]
        if not point_a_idx:
            return False
        point_a_idx = point_a_idx[0]
        point_a = prices[point_a_idx]
        
        # Point B should be a peak after A
        point_b_idx = [p for p in peaks if p > point_a_idx]
        if not point_b_idx:
            return False
        point_b_idx = point_b_idx[0]
        point_b = prices[point_b_idx]
        
        # Point C should be the current price (or last trough)
        point_c_idx = troughs[-1] if troughs[-1] > point_b_idx else len(prices) - 1
        point_c = prices[point_c_idx]
        
        # Now check the Shark ratios
        # XA/OX should be around 0.382-0.618
        ox = point_x - point_o
        xa = point_x - point_a
        if ox == 0:  # Avoid division by zero
            return False
        xa_ox_ratio = xa / ox
        if not (0.382 - 0.1 < xa_ox_ratio < 0.618 + 0.1):  # Allow some flexibility
            return False
            
        # AB/XA should be around 1.13-1.618
        ab = point_b - point_a
        if xa == 0:  # Avoid division by zero
            return False
        ab_xa_ratio = ab / xa
        if not (1.13 - 0.1 < ab_xa_ratio < 1.618 + 0.1):
            return False
            
        # BC/AB should be around 0.886-1.13
        bc = point_b - point_c
        if ab == 0:  # Avoid division by zero
            return False
        bc_ab_ratio = bc / ab
        if not (0.886 - 0.1 < bc_ab_ratio < 1.13 + 0.1):
            return False
            
        # OC/OX should be around 0.5-0.886
        oc = point_o - point_c
        if ox == 0:  # Avoid division by zero
            return False
        oc_ox_ratio = oc / ox
        if not (0.5 - 0.1 < oc_ox_ratio < 0.886 + 0.1):
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Shark pattern detection error: {e}")
        return False

def is_cypher_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Cypher harmonic pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 2 peaks and 2 troughs
        if len(peaks) < 2 or len(troughs) < 2:
            return False
            
        # For a bullish Cypher pattern:
        # Point X (start) - trough
        # Point A (peak) - peak
        # Point B (higher trough than X) - trough
        # Point C (lower peak than A) - peak
        # Point D (higher trough than B, but lower than A) - trough
        
        # We need to identify these points
        # First, try to identify point A (should be a significant peak)
        if len(peaks) < 2 or len(troughs) < 3:
            return False
            
        # Last peak should be point C
        point_c_idx = peaks[-1]
        point_c = prices[point_c_idx]
        
        # Last trough should be point D
        point_d_idx = troughs[-1]
        point_d = prices[point_d_idx]
        
        # Point A should be the peak before C
        point_a_idx = peaks[-2]
        point_a = prices[point_a_idx]
        
        # Point B should be the trough before C
        point_b_idx = [t for t in troughs if t > point_a_idx and t < point_c_idx]
        if not point_b_idx:
            return False
        point_b_idx = point_b_idx[-1]
        point_b = prices[point_b_idx]
        
        # Point X should be the trough before A
        point_x_idx = [t for t in troughs if t < point_a_idx]
        if not point_x_idx:
            return False
        point_x_idx = point_x_idx[-1]
        point_x = prices[point_x_idx]
        
        # Now check the Cypher ratios
        # XA should be significant move
        xa = point_a - point_x
        
        # AB/XA should be around 0.382-0.618
        ab = point_a - point_b
        if xa == 0:  # Avoid division by zero
            return False
        ab_xa_ratio = ab / xa
        if not (0.382 - 0.1 < ab_xa_ratio < 0.618 + 0.1):  # Allow some flexibility
            return False
            
        # BC/AB should be around 1.272-1.414
        bc = point_c - point_b
        if ab == 0:  # Avoid division by zero
            return False
        bc_ab_ratio = bc / ab
        if not (1.272 - 0.1 < bc_ab_ratio < 1.414 + 0.1):
            return False
            
        # CD/BC should be around 0.786
        cd = point_c - point_d
        if bc == 0:  # Avoid division by zero
            return False
        cd_bc_ratio = cd / bc
        if abs(cd_bc_ratio - 0.786) > 0.1:
            return False
            
        # XD/XC should be around 0.786
        xc = point_c - point_x
        xd = point_d - point_x
        if xc == 0:  # Avoid division by zero
            return False
        xd_xc_ratio = xd / xc
        if abs(xd_xc_ratio - 0.786) > 0.1:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Cypher pattern detection error: {e}")
        return False

def is_three_drives_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Three Drives harmonic pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 3 peaks and 3 troughs
        if len(peaks) < 3 or len(troughs) < 3:
            return False
            
        # For a bullish Three Drives pattern:
        # Point 0 (start) - peak
        # Point A1 (first drive) - trough
        # Point B1 (first correction) - peak
        # Point A2 (second drive) - trough
        # Point B2 (second correction) - peak
        # Point A3 (third drive) - trough
        
        # For simplicity, use the most recent points
        point_a3_idx = troughs[-1]
        point_a3 = prices[point_a3_idx]
        
        point_b2_idx = [p for p in peaks if p < point_a3_idx]
        if not point_b2_idx:
            return False
        point_b2_idx = point_b2_idx[-1]
        point_b2 = prices[point_b2_idx]
        
        point_a2_idx = [t for t in troughs if t < point_b2_idx]
        if not point_a2_idx:
            return False
        point_a2_idx = point_a2_idx[-1]
        point_a2 = prices[point_a2_idx]
        
        point_b1_idx = [p for p in peaks if p < point_a2_idx]
        if not point_b1_idx:
            return False
        point_b1_idx = point_b1_idx[-1]
        point_b1 = prices[point_b1_idx]
        
        point_a1_idx = [t for t in troughs if t < point_b1_idx]
        if not point_a1_idx:
            return False
        point_a1_idx = point_a1_idx[-1]
        point_a1 = prices[point_a1_idx]
        
        point_0_idx = [p for p in peaks if p < point_a1_idx]
        if not point_0_idx:
            return False
        point_0_idx = point_0_idx[-1]
        point_0 = prices[point_0_idx]
        
        # Now check the Three Drives ratios
        # A1/0 should have a significant move
        a1_0 = point_0 - point_a1
        
        # B1/A1 should be around 0.618
        b1_a1 = point_b1 - point_a1
        if a1_0 == 0:  # Avoid division by zero
            return False
        b1_a1_ratio = b1_a1 / a1_0
        if abs(b1_a1_ratio - 0.618) > 0.1:  # Allow some flexibility
            return False
            
        # A2/B1 should be around 1.272
        a2_b1 = point_b1 - point_a2
        if b1_a1 == 0:  # Avoid division by zero
            return False
        a2_b1_ratio = a2_b1 / b1_a1
        if abs(a2_b1_ratio - 1.272) > 0.1:
            return False
            
        # B2/A2 should be around 0.618
        b2_a2 = point_b2 - point_a2
        if a2_b1 == 0:  # Avoid division by zero
            return False
        b2_a2_ratio = b2_a2 / a2_b1
        if abs(b2_a2_ratio - 0.618) > 0.1:
            return False
            
        # A3/B2 should be around 1.272
        a3_b2 = point_b2 - point_a3
        if b2_a2 == 0:  # Avoid division by zero
            return False
        a3_b2_ratio = a3_b2 / b2_a2
        if abs(a3_b2_ratio - 1.272) > 0.1:
            return False
            
        # Check that drives get progressively deeper
        if a1_0 >= a2_b1 or a2_b1 >= a3_b2:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Three Drives pattern detection error: {e}")
        return False

def is_impulse_wave(df: pd.DataFrame) -> bool:
    """
    Detect Elliott Wave Impulse Wave
    """
    if len(df) < 10:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 3 peaks and 2 troughs for an impulse wave
        if len(peaks) < 3 or len(troughs) < 2:
            return False
            
        # Get the most recent points
        # Assuming the sequence: trough (start) -> peak (1) -> trough (2) -> peak (3) -> trough (4) -> peak (5)
        
        # For a bullish impulse wave
        last_idx = len(peaks) - 1
        if last_idx < 2:
            return False
            
        # Point 5 (end of impulse wave)
        point_5_idx = peaks[last_idx]
        point_5 = prices[point_5_idx]
        
        # Point 4
        point_4_idx = [t for t in troughs if t < point_5_idx]
        if not point_4_idx:
            return False
        point_4_idx = point_4_idx[-1]
        point_4 = prices[point_4_idx]
        
        # Point 3
        point_3_idx = [p for p in peaks if p < point_4_idx]
        if not point_3_idx:
            return False
        point_3_idx = point_3_idx[-1]
        point_3 = prices[point_3_idx]
        
        # Point 2
        point_2_idx = [t for t in troughs if t < point_3_idx]
        if not point_2_idx:
            return False
        point_2_idx = point_2_idx[-1]
        point_2 = prices[point_2_idx]
        
        # Point 1
        point_1_idx = [p for p in peaks if p < point_2_idx]
        if not point_1_idx:
            return False
        point_1_idx = point_1_idx[-1]
        point_1 = prices[point_1_idx]
        
        # Point 0 (start)
        point_0_idx = [t for t in troughs if t < point_1_idx]
        if not point_0_idx:
            return False
        point_0_idx = point_0_idx[-1]
        point_0 = prices[point_0_idx]
        
        # Check the Elliott Wave rules
        # Rule 1: Wave 2 cannot retrace more than 100% of wave 1
        wave_1 = point_1 - point_0
        wave_2_retracement = point_1 - point_2
        if wave_2_retracement >= wave_1:
            return False
            
        # Rule 2: Wave 3 must travel beyond the end of wave 1
        if point_3 <= point_1:
            return False
            
        # Rule 3: Wave 4 cannot overlap wave 1
        if point_4 <= point_1:
            return False
            
        # Rule 4: Wave 3 cannot be the shortest among waves 1, 3, and 5
        wave_3 = point_3 - point_2
        wave_5 = point_5 - point_4
        if wave_3 <= wave_1 and wave_3 <= wave_5:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Impulse wave detection error: {e}")
        return False

def is_corrective_wave(df: pd.DataFrame) -> bool:
    """
    Detect Elliott Wave Corrective Wave
    """
    if len(df) < 10:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 2 peaks and 1 trough for a simple corrective wave (A-B-C)
        if len(peaks) < 1 or len(troughs) < 2:
            return False
            
        # For a bearish corrective wave
        # Point 0 (start) - peak
        # Point A - trough
        # Point B - peak
        # Point C - trough
        
        # Get the most recent points
        point_c_idx = troughs[-1]
        point_c = prices[point_c_idx]
        
        point_b_idx = [p for p in peaks if p < point_c_idx]
        if not point_b_idx:
            return False
        point_b_idx = point_b_idx[-1]
        point_b = prices[point_b_idx]
        
        point_a_idx = [t for t in troughs if t < point_b_idx]
        if not point_a_idx:
            return False
        point_a_idx = point_a_idx[-1]
        point_a = prices[point_a_idx]
        
        point_0_idx = [p for p in peaks if p < point_a_idx]
        if not point_0_idx:
            return False
        point_0_idx = point_0_idx[-1]
        point_0 = prices[point_0_idx]
        
        # Check the corrective wave rules
        # Rule 1: Wave B should not exceed the start of wave A
        if point_b >= point_0:
            return False
            
        # Rule 2: Wave C should move beyond the end of wave A
        if point_c >= point_a:
            return False
            
        # Rule 3: Wave A should be a clear impulse or have 5 sub-waves
        # This is complex to check, so we'll use a simpler check
        wave_a = point_0 - point_a
        if wave_a <= 0:
            return False
            
        # Calculate common Fibonacci relationships
        wave_b_retracement = (point_b - point_a) / wave_a
        
        # Wave B often retraces 50% or 61.8% of wave A
        if not (0.38 <= wave_b_retracement <= 0.618 + 0.1):
            # Allow more flexibility since corrective waves can vary
            if not (0.23 <= wave_b_retracement <= 0.786 + 0.1):
                return False
            
        # Wave C is often equal to wave A or 1.618 times wave A
        wave_c = point_b - point_c
        wave_c_relation = wave_c / wave_a
        
        if not (0.618 - 0.1 <= wave_c_relation <= 1.618 + 0.1):
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Corrective wave detection error: {e}")
        return False

def is_bump_and_run(df: pd.DataFrame) -> bool:
    """
    Detect Bump and Run Reversal pattern
    """
    if len(df) < 20:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        x = np.arange(len(prices))
        
        # For a Bump and Run pattern, we need:
        # 1. A lead-in phase with a trendline
        # 2. A bump phase with steeper slope
        # 3. A run phase with price decline
        
        # Split the data into three equal segments
        segment_size = len(prices) // 3
        lead_in = prices[:segment_size]
        bump = prices[segment_size:2*segment_size]
        run = prices[2*segment_size:]
        
        # Calculate the slope of each segment
        lead_in_slope, _ = fit_trendline(np.arange(len(lead_in)), lead_in)
        bump_slope, _ = fit_trendline(np.arange(len(bump)), bump)
        run_slope, _ = fit_trendline(np.arange(len(run)), run)
        
        # For a bullish Bump and Run:
        # 1. Lead-in should have positive slope
        if lead_in_slope <= 0:
            return False
            
        # 2. Bump should have steeper positive slope
        if bump_slope <= lead_in_slope:
            return False
            
        # 3. Run should have negative slope
        if run_slope >= 0:
            return False
            
        # 4. The bump phase should be at least 2x the lead-in slope
        if bump_slope < 2 * lead_in_slope:
            return False
            
        # Check volume pattern if available (optional)
        #if 'Volume' in df.columns:
        #    volume = df['Volume'].values
        #    lead_in_vol = volume[:segment_size]
        #    bump_vol = volume[segment_size:2*segment_size]
        #    run_vol = volume[2*segment_size:]
        #    
        #    # Volume should increase during the bump phase
        #    if np.mean(bump_vol) <= np.mean(lead_in_vol):
        #        return False
            
        return True
    except Exception as e:
        logging.warning(f"Bump and Run detection error: {e}")
        return False

def is_scallop_pattern(df: pd.DataFrame) -> bool:
    """
    Detect Scallop pattern
    """
    if len(df) < 15:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 1 peak and 2 troughs for a scallop pattern
        if len(peaks) < 1 or len(troughs) < 2:
            return False
            
        # For a bullish scallop pattern:
        # 1. Initial trough
        # 2. Peak
        # 3. Final trough (higher than initial trough)
        
        # Get the most recent points
        final_trough_idx = troughs[-1]
        final_trough = prices[final_trough_idx]
        
        peak_idx = [p for p in peaks if p < final_trough_idx]
        if not peak_idx:
            return False
        peak_idx = peak_idx[-1]
        peak = prices[peak_idx]
        
        initial_trough_idx = [t for t in troughs if t < peak_idx]
        if not initial_trough_idx:
            return False
        initial_trough_idx = initial_trough_idx[-1]
        initial_trough = prices[initial_trough_idx]
        
        # Check scallop pattern rules
        # 1. Final trough should be higher than initial trough
        if final_trough <= initial_trough:
            return False
            
        # 2. Peak should be significantly higher than both troughs
        if peak <= initial_trough or peak <= final_trough:
            return False
            
        # 3. Price should form a rounded shape
        # Fit a quadratic curve to the data
        subset_indices = list(range(initial_trough_idx, final_trough_idx + 1))
        subset_prices = prices[initial_trough_idx:final_trough_idx + 1]
        
        if len(subset_indices) < 3:
            return False
            
        # Fit a quadratic curve to the data (y = ax^2 + bx + c)
        coeffs = np.polyfit(subset_indices, subset_prices, 2)
        a, b, c = coeffs
        
        # For a scallop pattern, 'a' coefficient should be negative (concave down)
        if a >= 0:
            return False
            
        # Calculate the fitted curve
        fitted = a * np.array(subset_indices)**2 + b * np.array(subset_indices) + c
        
        # Calculate the mean squared error
        mse = np.mean((subset_prices - fitted)**2)
        
        # Calculate the total variation
        total_var = np.var(subset_prices)
        
        # Calculate the R-squared value: 1 - (mse / total_var)
        r_squared = 1 - (mse / total_var)
        
        # R-squared must be high enough to indicate a good fit
        if r_squared < 0.7:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Scallop pattern detection error: {e}")
        return False

def is_ascending_staircase(df: pd.DataFrame) -> bool:
    """
    Detect Ascending Staircase pattern
    """
    if len(df) < 12:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 3 peaks and 3 troughs for a staircase pattern
        if len(peaks) < 3 or len(troughs) < 3:
            return False
            
        # For an ascending staircase:
        # 1. Each peak should be higher than the previous peak
        # 2. Each trough should be higher than the previous trough
        
        # Check last 3 peaks
        last_three_peaks = sorted(peaks[-3:])
        peak_heights = [prices[i] for i in last_three_peaks]
        
        if not (peak_heights[0] < peak_heights[1] < peak_heights[2]):
            return False
            
        # Check last 3 troughs
        last_three_troughs = sorted(troughs[-3:])
        trough_depths = [prices[i] for i in last_three_troughs]
        
        if not (trough_depths[0] < trough_depths[1] < trough_depths[2]):
            return False
            
        # Check that the pattern has significant steps
        avg_peak_step = (peak_heights[2] - peak_heights[0]) / 2
        avg_trough_step = (trough_depths[2] - trough_depths[0]) / 2
        
        if avg_peak_step / peak_heights[0] < 0.01 or avg_trough_step / trough_depths[0] < 0.01:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Ascending staircase detection error: {e}")
        return False

def is_descending_staircase(df: pd.DataFrame) -> bool:
    """
    Detect Descending Staircase pattern
    """
    if len(df) < 12:
        return False
        
    try:
        # Get price data
        prices = df['Close'].values
        
        # Find peaks and troughs
        peaks, troughs = find_peaks_and_troughs(prices)
        
        # Need at least 3 peaks and 3 troughs for a staircase pattern
        if len(peaks) < 3 or len(troughs) < 3:
            return False
            
        # For a descending staircase:
        # 1. Each peak should be lower than the previous peak
        # 2. Each trough should be lower than the previous trough
        
        # Check last 3 peaks
        last_three_peaks = sorted(peaks[-3:])
        peak_heights = [prices[i] for i in last_three_peaks]
        
        if not (peak_heights[0] > peak_heights[1] > peak_heights[2]):
            return False
            
        # Check last 3 troughs
        last_three_troughs = sorted(troughs[-3:])
        trough_depths = [prices[i] for i in last_three_troughs]
        
        if not (trough_depths[0] > trough_depths[1] > trough_depths[2]):
            return False
            
        # Check that the pattern has significant steps
        avg_peak_step = (peak_heights[0] - peak_heights[2]) / 2
        avg_trough_step = (trough_depths[0] - trough_depths[2]) / 2
        
        if avg_peak_step / peak_heights[0] < 0.01 or avg_trough_step / trough_depths[0] < 0.01:
            return False
            
        return True
    except Exception as e:
        logging.warning(f"Descending staircase detection error: {e}")
        return False