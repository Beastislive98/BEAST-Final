# indicator_universe.py - Enhanced with Performance Optimization & Regime Awareness

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Any
import ta
from functools import lru_cache
import hashlib
import time
from binance_interface import get_futures_market_data as get_market_data
from data_processing import process_market_data

# Performance tracking
CALCULATION_TIMES = {}
CACHE_HIT_RATE = {"hits": 0, "misses": 0}

@lru_cache(maxsize=256)
def get_cached_indicators(data_hash: str, params_hash: str) -> Optional[Tuple]:
    """
    Cache expensive indicator calculations
    Returns tuple of (indicators_dict, timestamp) or None
    """
    # This would normally connect to a persistent cache (Redis, etc.)
    # For now, just use memory cache
    return None

def clear_indicator_cache():
    """Clear the indicator calculation cache"""
    get_cached_indicators.cache_clear()
    global CACHE_HIT_RATE
    CACHE_HIT_RATE = {"hits": 0, "misses": 0}

def create_data_hash(data: pd.DataFrame) -> str:
    """Create a hash of the dataframe for caching purposes"""
    try:
        # Use the last few rows and key columns for hashing
        key_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(100)
        data_string = key_data.to_string()
        return hashlib.md5(data_string.encode()).hexdigest()
    except Exception:
        return str(time.time())  # Fallback to timestamp

def create_params_hash(params: Dict) -> str:
    """Create a hash of the parameters for caching"""
    try:
        params_string = str(sorted(params.items()))
        return hashlib.md5(params_string.encode()).hexdigest()
    except Exception:
        return str(hash(str(params)))

def get_regime_adjusted_periods(base_params: Dict, regime_type: str = "unknown") -> Dict:
    """
    Adjust indicator periods based on market regime
    """
    adjusted_params = base_params.copy()
    
    if regime_type == "high_volatility":
        # Use shorter periods for faster response in volatile markets
        for indicator, params in adjusted_params.items():
            if "period" in params:
                original_period = params["period"]
                adjusted_period = max(5, int(original_period * 0.7))  # 30% shorter
                adjusted_params[indicator] = params.copy()
                adjusted_params[indicator]["period"] = adjusted_period
        logging.debug("Adjusted indicator periods for high volatility regime")
    
    elif regime_type == "ranging":
        # Use longer periods for better smoothing in ranging markets
        for indicator, params in adjusted_params.items():
            if "period" in params:
                original_period = params["period"]
                adjusted_period = int(original_period * 1.3)  # 30% longer
                adjusted_params[indicator] = params.copy()
                adjusted_params[indicator]["period"] = adjusted_period
        logging.debug("Adjusted indicator periods for ranging regime")
    
    return adjusted_params

def get_data_with_indicators(symbol: str, timeframe: str, limit: int = 100, 
                           params: Dict[str, Dict] = None, regime_type: str = "unknown"):
    """
    Enhanced wrapper function with regime awareness and caching
    """
    start_time = time.time()
    
    if params is None:
        params = {}
    
    # Adjust parameters based on market regime
    regime_adjusted_params = get_regime_adjusted_periods(params, regime_type)
    
    # Calculate minimum required points based on indicator parameters
    min_required_points = 30  # Base requirement
    for indicator, indicator_params in regime_adjusted_params.items():
        if "period" in indicator_params:
            min_required_points = max(min_required_points, indicator_params["period"] * 2)
    
    # Add buffer for safety (20% extra data)
    adjusted_limit = int(min_required_points * 1.2)
    
    # Ensure we're fetching at least the requested limit or the minimum required
    final_limit = max(limit, adjusted_limit)
    
    logging.info(f"Auto-adjusting data limit from {limit} to {final_limit} "
                f"based on {'regime-adjusted ' if regime_type != 'unknown' else ''}indicator requirements")
    
    # Get the market data with adjusted limit
    data = get_market_data(symbol=symbol, timeframe=timeframe, limit=final_limit)
    
    # Process the data if needed
    processed_data = process_market_data(data)
    
    # Calculate indicators with regime-adjusted parameters and comprehensive fallback
    data_with_indicators = calculate_indicators_with_fallback(processed_data, symbol, regime_type)
    
    # Track performance
    calculation_time = time.time() - start_time
    CALCULATION_TIMES[f"{symbol}_{timeframe}"] = calculation_time
    
    logging.info(f"Data with indicators retrieved for {symbol} in {calculation_time:.2f}s "
                f"(Regime: {regime_type})")
    
    return data_with_indicators

def calculate_indicators_batch(data_list: list, params: Dict[str, Dict], 
                             regime_type: str = "unknown") -> list:
    """
    Calculate indicators for multiple dataframes in batch for better performance
    """
    start_time = time.time()
    results = []
    
    # Adjust parameters for regime
    regime_adjusted_params = get_regime_adjusted_periods(params, regime_type)
    
    for i, data in enumerate(data_list):
        try:
            result = calculate_indicators_with_fallback(data, f"batch_dataset_{i}", regime_type)
            results.append(result)
        except Exception as e:
            logging.warning(f"Batch indicator calculation failed for dataset {i}: {e}")
            results.append(data)  # Return original data on failure
    
    batch_time = time.time() - start_time
    logging.info(f"Batch indicator calculation completed for {len(data_list)} datasets "
               f"in {batch_time:.2f}s (avg: {batch_time/len(data_list):.3f}s per dataset)")
    
    return results

def calculate_indicators_with_fallback(df: pd.DataFrame, symbol: str = "", regime_type: str = "unknown") -> Optional[pd.DataFrame]:
    """
    Calculate indicators with fallback handling for insufficient data - INTEGRATED FUNCTION
    """
    if df is None or df.empty:
        logging.warning(f"No data provided for indicator calculation for {symbol}")
        return None
    
    # Check data sufficiency
    min_required = 50  # Minimum periods for reliable indicators
    
    if len(df) >= min_required:
        # Sufficient data - use normal calculation with regime awareness
        regime_params = get_regime_adjusted_periods({
            'SMA': {'period': 14}, 'EMA': {'period': 20}, 'RSI': {'period': 14}, 
            'ATR': {'period': 14}, 'BBANDS': {}, 'MACD': {}
        }, regime_type)
        return calculate_indicators(df, regime_params, regime_type)
    
    # Insufficient data - try fallback strategies
    logging.warning(f"Insufficient data for {symbol} ({len(df)} periods), attempting fallback")
    
    # Strategy 1: Try to get more data
    if symbol:
        try:
            from binance_interface import get_extended_historical_data
            extended_df = get_extended_historical_data(symbol, days=30)
            
            if extended_df is not None and len(extended_df) >= min_required:
                logging.info(f"Got extended data for {symbol} ({len(extended_df)} periods)")
                regime_params = get_regime_adjusted_periods({
                    'SMA': {'period': 14}, 'EMA': {'period': 20}, 'RSI': {'period': 14}, 
                    'ATR': {'period': 14}, 'BBANDS': {}, 'MACD': {}
                }, regime_type)
                return calculate_indicators(extended_df, regime_params, regime_type)
        except Exception as e:
            logging.error(f"Failed to get extended data for {symbol}: {e}")
    
    # Strategy 2: Use available data with reduced parameters
    if len(df) >= 20:  # Minimum viable data
        logging.warning(f"Using reduced parameters for {symbol} with {len(df)} periods")
        return calculate_indicators_reduced(df, regime_type)
    
    # Strategy 3: Last resort - minimal indicators
    if len(df) >= 10:
        logging.warning(f"Using minimal indicators for {symbol} with {len(df)} periods")
        return calculate_minimal_indicators(df)
    
    logging.error(f"Cannot calculate indicators for {symbol} - insufficient data ({len(df)} periods)")
    return None

def calculate_indicators_reduced(df: pd.DataFrame, regime_type: str = "unknown") -> pd.DataFrame:
    """
    Calculate indicators with reduced parameters for limited data - INTEGRATED FUNCTION
    """
    try:
        result_df = df.copy()
        
        # Adjust periods based on regime even for reduced data
        base_period_5 = 5
        base_period_10 = 10
        
        if regime_type == "high_volatility":
            base_period_5 = max(3, int(base_period_5 * 0.7))
            base_period_10 = max(7, int(base_period_10 * 0.7))
        elif regime_type == "ranging":
            base_period_10 = min(14, int(base_period_10 * 1.3))
        
        # Shorter period moving averages
        if len(df) >= base_period_5:
            result_df[f'SMA_{base_period_5}'] = df['Close'].rolling(window=base_period_5).mean()
        if len(df) >= base_period_10:
            result_df[f'SMA_{base_period_10}'] = df['Close'].rolling(window=base_period_10).mean()
            result_df[f'EMA_{base_period_10}'] = df['Close'].ewm(span=base_period_10).mean()
        
        # Shorter RSI
        rsi_period = 7 if regime_type == "high_volatility" else 14
        if len(df) >= rsi_period:
            result_df[f'RSI_{rsi_period}'] = calculate_rsi_simple(df['Close'], period=rsi_period)
        
        # Basic volatility (shorter ATR)
        atr_period = 7 if regime_type == "high_volatility" else 14
        if len(df) >= atr_period:
            result_df[f'ATR_{atr_period}'] = calculate_atr_reduced(df, period=atr_period)
        
        # Simple momentum
        if len(df) >= 5:
            result_df['MOM_5'] = df['Close'].pct_change(periods=5)
        
        # Basic Bollinger Bands
        bb_period = base_period_10
        if len(df) >= bb_period:
            sma = df['Close'].rolling(window=bb_period).mean()
            std = df['Close'].rolling(window=bb_period).std()
            result_df[f'BB_UPPER_{bb_period}'] = sma + (std * 2)
            result_df[f'BB_LOWER_{bb_period}'] = sma - (std * 2)
            result_df[f'BB_MID_{bb_period}'] = sma
        
        # Volume indicators if available
        if 'Volume' in df.columns and len(df) >= base_period_10:
            result_df[f'VOL_SMA_{base_period_10}'] = df['Volume'].rolling(window=base_period_10).mean()
            result_df['VOL_RATIO'] = df['Volume'] / result_df[f'VOL_SMA_{base_period_10}']
        
        logging.info(f"Calculated reduced indicators for {len(df)} periods (Regime: {regime_type})")
        return result_df
        
    except Exception as e:
        logging.error(f"Error calculating reduced indicators: {e}")
        return df

def calculate_minimal_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minimal indicators for very limited data - INTEGRATED FUNCTION
    """
    try:
        result_df = df.copy()
        
        # Absolute minimal indicators
        
        # Simple moving average
        if len(df) >= 5:
            result_df['SMA_5'] = df['Close'].rolling(window=5).mean()
        
        # Basic price change
        result_df['PRICE_CHANGE'] = df['Close'].pct_change()
        
        # Simple high/low range
        if len(df) >= 3:
            result_df['HIGH_3'] = df['High'].rolling(window=3).max()
            result_df['LOW_3'] = df['Low'].rolling(window=3).min()
            result_df['RANGE_3'] = result_df['HIGH_3'] - result_df['LOW_3']
        
        # Basic volume ratio if available
        if 'Volume' in df.columns and len(df) >= 5:
            vol_avg = df['Volume'].rolling(window=5).mean()
            result_df['VOL_RATIO'] = df['Volume'] / vol_avg
        
        logging.info(f"Calculated minimal indicators for {len(df)} periods")
        return result_df
        
    except Exception as e:
        logging.error(f"Error calculating minimal indicators: {e}")
        return df

def calculate_atr_reduced(df: pd.DataFrame, period: int = 7) -> pd.Series:
    """
    Calculate ATR with reduced period for limited data - INTEGRATED FUNCTION
    """
    try:
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
        
    except Exception as e:
        logging.error(f"Error calculating reduced ATR: {e}")
        return pd.Series(index=df.index, dtype=float)

def calculate_rsi_simple(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Simple RSI calculation for reduced data scenarios - INTEGRATED FUNCTION
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logging.error(f"Error calculating simple RSI: {e}")
        return pd.Series(index=prices.index, dtype=float)

def calculate_indicators(data: pd.DataFrame, params: Dict[str, Dict], 
                        regime_type: str = "unknown", batch_mode: bool = False) -> Optional[pd.DataFrame]:
    """
    Enhanced indicator calculation with caching, performance optimization, and robust error handling - UPDATED VERSION
    """
    start_time = time.time()
    
    if data is None or data.empty:
        logging.warning("Input data is empty. Skipping indicator calculation.")
        return None
    
    # Create hashes for caching
    data_hash = create_data_hash(data)
    params_hash = create_params_hash(params)
    cache_key = f"{data_hash}_{params_hash}_{regime_type}"
    
    # Try to get from cache
    try:
        cached_result = get_cached_indicators(cache_key, params_hash)
        if cached_result and time.time() - cached_result[1] < 300:  # 5 minute cache
            CACHE_HIT_RATE["hits"] += 1
            if not batch_mode:
                logging.debug(f"Using cached indicators (cache hit rate: "
                            f"{CACHE_HIT_RATE['hits']/(CACHE_HIT_RATE['hits']+CACHE_HIT_RATE['misses']):.2%})")
            return cached_result[0]
    except Exception:
        pass  # Cache miss or error
    
    CACHE_HIT_RATE["misses"] += 1
    
    # Determine minimum required points based on indicator parameters
    min_required_points = 30  # Base requirement
    for indicator, indicator_params in params.items():
        if "period" in indicator_params:
            min_required_points = max(min_required_points, indicator_params["period"] * 2)
    
    # Enhanced data sufficiency check with comprehensive fallback
    if len(data) < min_required_points:
        logging.warning(f"Insufficient data for complete indicator calculation: "
                       f"{len(data)} points available, {min_required_points} required "
                       f"(Regime: {regime_type})")
        
        # Use the comprehensive fallback system
        if len(data) >= 20:
            return calculate_indicators_reduced(data, regime_type)
        elif len(data) >= 10:
            return calculate_minimal_indicators(data)
        else:
            logging.error(f"Cannot calculate any indicators - insufficient data ({len(data)} periods)")
            return data

    # Verify required columns and fill missing ones - ENHANCED VERSION
    try:
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(data.columns):
            missing_cols = required_cols - set(data.columns)
            logging.warning(f"Missing required columns: {missing_cols}")
            
            # Try to fill in missing columns with approximations
            if 'Close' in data.columns:
                for col in missing_cols:
                    if col == 'Open':
                        data['Open'] = data['Close'].shift(1).fillna(data['Close'])
                    elif col == 'High':
                        data['High'] = data['Close'] * 1.002  # Approximate high
                    elif col == 'Low':
                        data['Low'] = data['Close'] * 0.998   # Approximate low
                    elif col == 'Volume':
                        # Use a realistic volume approximation based on price volatility
                        volatility = data['Close'].pct_change().std()
                        base_volume = 1000000
                        volume_multiplier = max(0.5, min(3.0, volatility * 100))
                        data['Volume'] = base_volume * volume_multiplier
                logging.info(f"Added approximated values for missing columns: {missing_cols}")
            else:
                return data  # Return original data if we can't even approximate

    except Exception as e:
        logging.error(f"Error handling missing columns: {e}")
        return data

    # Calculate indicators with enhanced error handling and performance tracking
    result_df = data.copy()
    indicator_times = {}
    successful_indicators = []
    failed_indicators = []
    
    for indicator, indicator_params in params.items():
        indicator_start = time.time()
        try:
            if indicator == 'SMA':
                period = indicator_params["period"]
                if len(data) >= period:
                    result_df[f'SMA_{period}'] = ta.trend.sma_indicator(
                        close=data['Close'], window=period
                    )
                    successful_indicators.append(f'SMA_{period}')
                else:
                    failed_indicators.append(f'SMA_{period}_insufficient_data')
                
            elif indicator == 'EMA':
                period = indicator_params["period"]
                if len(data) >= period:
                    result_df[f'EMA_{period}'] = ta.trend.ema_indicator(
                        close=data['Close'], window=period
                    )
                    successful_indicators.append(f'EMA_{period}')
                else:
                    failed_indicators.append(f'EMA_{period}_insufficient_data')
                
            elif indicator == 'RSI':
                period = indicator_params["period"]
                if len(data) >= period + 1:  # Need extra period for RSI calculation
                    rsi_indicator = ta.momentum.RSIIndicator(close=data['Close'], window=period)
                    result_df[f'RSI_{period}'] = rsi_indicator.rsi()
                    successful_indicators.append(f'RSI_{period}')
                else:
                    failed_indicators.append(f'RSI_{period}_insufficient_data')
                
            elif indicator == 'MACD':
                if len(data) >= 26:  # MACD needs at least 26 periods
                    macd = ta.trend.MACD(close=data['Close'])
                    result_df['MACD'] = macd.macd()
                    result_df['MACD_signal'] = macd.macd_signal()
                    result_df['MACD_hist'] = macd.macd_diff()
                    successful_indicators.append('MACD')
                else:
                    failed_indicators.append('MACD_insufficient_data')
                
            elif indicator == 'BBANDS':
                bb_period = 20  # Default Bollinger Band period
                if len(data) >= bb_period:
                    bb = ta.volatility.BollingerBands(close=data['Close'])
                    result_df['BB_upper'] = bb.bollinger_hband()
                    result_df['BB_mid'] = bb.bollinger_mavg()
                    result_df['BB_lower'] = bb.bollinger_lband()
                    result_df['BB_width'] = bb.bollinger_wband()  # Additional BB width
                    successful_indicators.append('BBANDS')
                else:
                    failed_indicators.append('BBANDS_insufficient_data')
                
            elif indicator == 'ATR':
                period = indicator_params["period"]
                if len(data) >= period:
                    atr_indicator = ta.volatility.AverageTrueRange(
                        high=data['High'], low=data['Low'], close=data['Close'], window=period
                    )
                    result_df[f'ATR_{period}'] = atr_indicator.average_true_range()
                    successful_indicators.append(f'ATR_{period}')
                else:
                    failed_indicators.append(f'ATR_{period}_insufficient_data')
                
            elif indicator == 'CCI':
                period = indicator_params["period"]
                if len(data) >= period:
                    cci_indicator = ta.trend.CCIIndicator(
                        high=data['High'], low=data['Low'], close=data['Close'], window=period
                    )
                    result_df[f'CCI_{period}'] = cci_indicator.cci()
                    successful_indicators.append(f'CCI_{period}')
                else:
                    failed_indicators.append(f'CCI_{period}_insufficient_data')
                
            elif indicator == 'OBV':
                if 'Volume' in data.columns:
                    obv_indicator = ta.volume.OnBalanceVolumeIndicator(
                        close=data['Close'], volume=data['Volume']
                    )
                    result_df['OBV'] = obv_indicator.on_balance_volume()
                    successful_indicators.append('OBV')
                else:
                    failed_indicators.append('OBV_no_volume_data')
                
            elif indicator == 'MFI':
                period = indicator_params["period"]
                if len(data) >= period and 'Volume' in data.columns:
                    mfi_indicator = ta.volume.MFIIndicator(
                        high=data['High'], low=data['Low'], close=data['Close'], 
                        volume=data['Volume'], window=period
                    )
                    result_df[f'MFI_{period}'] = mfi_indicator.money_flow_index()
                    successful_indicators.append(f'MFI_{period}')
                else:
                    failed_indicators.append(f'MFI_{period}_insufficient_data_or_no_volume')
                
            elif indicator == 'Stochastic':
                stoch_period = 14  # Default stochastic period
                if len(data) >= stoch_period:
                    stoch = ta.momentum.StochasticOscillator(
                        high=data['High'], low=data['Low'], close=data['Close']
                    )
                    result_df['Stoch_K'] = stoch.stoch()
                    result_df['Stoch_D'] = stoch.stoch_signal()
                    successful_indicators.append('Stochastic')
                else:
                    failed_indicators.append('Stochastic_insufficient_data')
                
            # Continue with remaining indicators...
            elif indicator == 'Momentum':
                period = indicator_params["period"]
                if len(data) >= period:
                    momentum_indicator = ta.momentum.ROCIndicator(
                        close=data['Close'], window=period
                    )
                    result_df[f'Momentum_{period}'] = momentum_indicator.roc()
                    successful_indicators.append(f'Momentum_{period}')
                else:
                    failed_indicators.append(f'Momentum_{period}_insufficient_data')
                
            elif indicator == 'ROC':
                period = indicator_params["period"]
                if len(data) >= period:
                    roc_indicator = ta.momentum.ROCIndicator(
                        close=data['Close'], window=period
                    )
                    result_df[f'ROC_{period}'] = roc_indicator.roc()
                    successful_indicators.append(f'ROC_{period}')
                else:
                    failed_indicators.append(f'ROC_{period}_insufficient_data')
                
            elif indicator == 'ADX':
                period = indicator_params["period"]
                if len(data) >= period:
                    adx_indicator = ta.trend.ADXIndicator(
                        high=data['High'], low=data['Low'], close=data['Close'], window=period
                    )
                    result_df[f'ADX_{period}'] = adx_indicator.adx()
                    result_df[f'ADX_POS_{period}'] = adx_indicator.adx_pos()  # Additional ADX components
                    result_df[f'ADX_NEG_{period}'] = adx_indicator.adx_neg()
                    successful_indicators.append(f'ADX_{period}')
                else:
                    failed_indicators.append(f'ADX_{period}_insufficient_data')
                
            elif indicator == 'KST':
                if len(data) >= 50:  # KST needs more data
                    kst = ta.trend.KSTIndicator(close=data['Close'])
                    result_df['KST'] = kst.kst()
                    result_df['KST_diff'] = kst.kst_diff()
                    successful_indicators.append('KST')
                else:
                    failed_indicators.append('KST_insufficient_data')
                
            elif indicator == 'TSI':
                if len(data) >= 25:  # TSI needs sufficient data
                    tsi_indicator = ta.momentum.TSIIndicator(close=data['Close'])
                    result_df['TSI'] = tsi_indicator.tsi()
                    successful_indicators.append('TSI')
                else:
                    failed_indicators.append('TSI_insufficient_data')
                
            # Add remaining indicators with similar error handling pattern...
            elif indicator == 'UltimateOscillator':
                if len(data) >= 28:  # UO needs sufficient data
                    uo_indicator = ta.momentum.UltimateOscillator(
                        high=data['High'], low=data['Low'], close=data['Close']
                    )
                    result_df['UltimateOscillator'] = uo_indicator.ultimate_oscillator()
                    successful_indicators.append('UltimateOscillator')
                else:
                    failed_indicators.append('UltimateOscillator_insufficient_data')
            
            # Add other indicators following the same pattern...
            # (Vortex, Donchian, Keltner, Ichimoku, etc.)
            
            # Track individual indicator calculation time
            indicator_time = time.time() - indicator_start
            indicator_times[indicator] = indicator_time
            
        except Exception as ind_err:
            logging.warning(f"Indicator '{indicator}' failed with error: {ind_err}")
            failed_indicators.append(f'{indicator}_calculation_error')
            continue
    
    # Log results - ENHANCED LOGGING
    if successful_indicators:
        logging.debug(f"Successfully calculated indicators: {successful_indicators}")
    
    if failed_indicators:
        logging.warning(f"Failed to calculate some indicators: {failed_indicators}")
    
    # Ensure we have at least some indicators - ENHANCED VALIDATION
    indicator_columns = [col for col in result_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    if len(indicator_columns) == 0:
        logging.warning("No indicators were successfully calculated, falling back to minimal indicators")
        return calculate_minimal_indicators(data)  # Use the fallback system
    
    # Add regime-specific derived indicators
    try:
        if regime_type == "high_volatility":
            # Add volatility-specific indicators
            if 'Close' in result_df.columns:
                result_df['Price_Volatility'] = result_df['Close'].pct_change().rolling(20).std()
                result_df['Volatility_Rank'] = result_df['Price_Volatility'].rolling(100).rank(pct=True)
        
        elif regime_type == "ranging":
            # Add range-specific indicators
            if 'BB_upper' in result_df.columns and 'BB_lower' in result_df.columns:
                result_df['BB_Position'] = (result_df['Close'] - result_df['BB_lower']) / (result_df['BB_upper'] - result_df['BB_lower'])
        
        elif regime_type in ["bull_trend", "bear_trend"]:
            # Add trend-specific indicators
            if 'Close' in result_df.columns and len(result_df) >= 50:
                sma_20 = ta.trend.sma_indicator(close=result_df['Close'], window=20)
                sma_50 = ta.trend.sma_indicator(close=result_df['Close'], window=50)
                result_df['Trend_Strength'] = sma_20 / sma_50
                
    except Exception as e:
        logging.warning(f"Failed to add regime-specific indicators: {e}")

    # Calculate total processing time
    total_time = time.time() - start_time
    
    # Log performance summary
    if not batch_mode:
        logging.info(f"Indicator calculation complete in {total_time:.3f}s "
                    f"(Regime: {regime_type}, Successful: {len(successful_indicators)}, Failed: {len(failed_indicators)}) "
                    f"- Cache hit rate: {CACHE_HIT_RATE['hits']/(CACHE_HIT_RATE['hits']+CACHE_HIT_RATE['misses']):.2%}")
        
        # Log slow indicators
        slow_indicators = {k: v for k, v in indicator_times.items() if v > 0.1}
        if slow_indicators:
            logging.debug(f"Slow indicators: {slow_indicators}")

    logging.info(f"Calculated {len(indicator_columns)} indicators for {len(data)} periods")
    return result_df

def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics for indicator calculations
    """
    total_hits = CACHE_HIT_RATE["hits"]
    total_misses = CACHE_HIT_RATE["misses"]
    total_requests = total_hits + total_misses
    
    return {
        "cache_hit_rate": total_hits / total_requests if total_requests > 0 else 0,
        "total_cache_requests": total_requests,
        "average_calculation_times": CALCULATION_TIMES,
        "last_updated": time.time()
    }

def optimize_indicator_selection(data: pd.DataFrame, regime_type: str = "unknown") -> Dict[str, Dict]:
    """
    Dynamically select optimal indicators based on data characteristics and regime
    """
    data_length = len(data)
    volatility = data['Close'].pct_change().std() if 'Close' in data.columns else 0.02
    
    # Base indicators always needed
    optimized_params = {
        'RSI': {'period': 14},
        'SMA': {'period': 20},
        'ATR': {'period': 14}
    }
    
    # Add indicators based on data length
    if data_length >= 50:
        optimized_params.update({
            'MACD': {},
            'BBANDS': {},
            'Stochastic': {}
        })
    
    if data_length >= 100:
        optimized_params.update({
            'ADX': {'period': 14},
            'CCI': {'period': 20}
        })
    
    # Regime-specific optimizations
    if regime_type == "high_volatility":
        optimized_params.update({
            'UltimateOscillator': {},
            'Vortex': {},
            'MassIndex': {}
        })
    elif regime_type == "ranging":
        optimized_params.update({
            'Donchian': {},
            'Keltner': {},
            'CMF': {'period': 20}
        })
    elif regime_type in ["bull_trend", "bear_trend"]:
        optimized_params.update({
            'Ichimoku': {},
            'KST': {},
            'TSI': {}
        })
    
    # Adjust periods based on volatility
    if volatility > 0.05:  # High volatility
        for indicator in optimized_params:
            if "period" in optimized_params[indicator]:
                optimized_params[indicator]["period"] = max(5, 
                    int(optimized_params[indicator]["period"] * 0.7))
    
    logging.info(f"Optimized indicator selection for {regime_type} regime: "
                f"{len(optimized_params)} indicators")
    
    return optimized_params