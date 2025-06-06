import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

# FIXED: Data validation functions
def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """FIXED: Comprehensive OHLCV data validation"""
    if df is None or df.empty:
        return False
        
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_columns):
        return False
        
    # Check for sufficient data points
    if len(df) < 5:  # FIXED: Reduced minimum from 10 to 5
        logging.debug("Insufficient data points for analysis")
        return False
        
    # Check for data quality issues
    if df[required_columns].isnull().all().any():
        logging.debug("Column(s) with all NaN values detected")
        return False
        
    # Check for negative prices or volumes
    price_cols = ["Open", "High", "Low", "Close"]
    if (df[price_cols] <= 0).any().any():
        logging.debug("Negative or zero prices detected")
        return False
        
    if (df["Volume"] < 0).any():
        logging.debug("Negative volume detected")
        return False
        
    return True

def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """FIXED: Clean and validate OHLCV data with gentler cleaning"""
    if df is None or df.empty:
        return df
        
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    
    # Ensure all required columns exist
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    # FIXED: More careful numeric conversion
    original_length = len(df)
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # FIXED: Only remove rows where ALL OHLC values are NaN (less aggressive)
    price_cols = ["Open", "High", "Low", "Close"]
    df = df.dropna(subset=price_cols, how='all')
    
    # FIXED: Handle individual NaN values by forward/backward fill
    for col in price_cols:
        if df[col].isnull().any():
            # Forward fill first, then backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Handle Volume NaNs separately (can be 0)
    if df['Volume'].isnull().any():
        df['Volume'] = df['Volume'].fillna(0)
    
    # FIXED: Only remove rows with invalid prices (more lenient)
    df = df[(df[price_cols] > 0).any(axis=1)]  # At least one price > 0
    df = df[df["Volume"] >= 0]  # Volume can be 0 but not negative
    
    # FIXED: Fix invalid OHLC relationships but don't remove rows
    invalid_mask = (
        (df["High"] < df["Low"]) |
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"]) |
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"])
    )
    
    if invalid_mask.any():
        logging.debug(f"Fixing {invalid_mask.sum()} invalid OHLC relationships")
        # For invalid rows, set High = max(O,H,L,C) and Low = min(O,H,L,C)
        df.loc[invalid_mask, "High"] = df.loc[invalid_mask, ["Open", "High", "Low", "Close"]].max(axis=1)
        df.loc[invalid_mask, "Low"] = df.loc[invalid_mask, ["Open", "High", "Low", "Close"]].min(axis=1)
    
    # FIXED: Less aggressive outlier removal
    if len(df) > 10:  # Only remove outliers if we have enough data
        price_change = df["Close"].pct_change().abs()
        outlier_mask = price_change > 0.8  # FIXED: Increased threshold from 0.5 to 0.8
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0 and outlier_count < len(df) * 0.1:  # Only if < 10% of data
            logging.debug(f"Removing {outlier_count} extreme outlier candles")
            df = df[~outlier_mask]
    
    # Sort by timestamp if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    
    cleaned_length = len(df)
    if cleaned_length < original_length * 0.5:  # Lost more than 50% of data
        logging.warning(f"Aggressive cleaning removed {original_length - cleaned_length} rows, may have been too harsh")
    
    return df

def process_market_data(market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    FIXED: Process market data with better fallback handling and less aggressive cleaning
    """
    try:
        # Check if we have any market data at all
        if not market_data:
            logging.debug("Market data is empty or None")
            return None

        # Log available keys for debugging
        logging.debug(f"Available keys in market_data: {list(market_data.keys())}")
        
        df = None
        
        # Case 1: OHLCV data is already present
        if "ohlcv" in market_data:
            ohlcv_data = market_data["ohlcv"]
            
            if isinstance(ohlcv_data, pd.DataFrame):
                df = ohlcv_data.copy()
            elif isinstance(ohlcv_data, dict):
                try:
                    df = pd.DataFrame(ohlcv_data)
                except Exception as e:
                    logging.debug(f"Failed to convert OHLCV dict to DataFrame: {e}")
                    df = None
            else:
                logging.debug(f"OHLCV data is in unexpected format: {type(ohlcv_data)}")
                df = None
        
        # Case 2: Try to construct OHLCV from klines
        elif "klines" in market_data:
            try:
                klines = market_data["klines"]
                
                # FIXED: Better type checking and handling
                if isinstance(klines, pd.DataFrame):
                    required_cols = ["Open", "High", "Low", "Close", "Volume"]
                    if all(col in klines.columns for col in required_cols):
                        df = klines[required_cols].copy()
                    else:
                        logging.debug("Klines DataFrame missing required OHLCV columns")
                        df = None
                
                elif isinstance(klines, list) and len(klines) > 0:
                    # FIXED: More robust klines processing
                    if isinstance(klines[0], (list, tuple)) and len(klines[0]) >= 6:
                        try:
                            df = pd.DataFrame(klines, columns=[
                                "timestamp", "Open", "High", "Low", "Close", "Volume", 
                                "close_time", "quote_volume", "trades", "taker_buy_base", 
                                "taker_buy_quote", "ignored"
                            ])
                            
                            # Convert numeric columns with error handling
                            numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
                            for col in numeric_cols:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Handle timestamp conversion
                            if "timestamp" in df.columns:
                                try:
                                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors='coerce')
                                    # Only set index if conversion successful
                                    if not df["timestamp"].isnull().all():
                                        df.set_index("timestamp", inplace=True)
                                except Exception as e:
                                    logging.debug(f"Timestamp conversion failed: {e}")
                                    # Continue without timestamp index
                                    pass
                                    
                            # Keep only OHLCV columns
                            df = df[["Open", "High", "Low", "Close", "Volume"]]
                            
                        except Exception as e:
                            logging.debug(f"Failed to process klines list: {e}")
                            df = None
                    else:
                        logging.debug("Klines list format not recognized")
                        df = None
                else:
                    logging.debug(f"Klines data in unexpected format: {type(klines)}")
                    df = None
                    
            except Exception as e:
                logging.debug(f"Failed to construct OHLCV from klines: {e}")
                df = None
        
        # Case 3: FIXED: Better ticker fallback
        elif "ticker" in market_data:
            try:
                ticker = market_data["ticker"]
                
                # Try to get OHLC from ticker
                current_price = float(ticker.get("lastPrice", 0))
                high_price = float(ticker.get("highPrice", current_price))
                low_price = float(ticker.get("lowPrice", current_price))
                open_price = float(ticker.get("openPrice", current_price))
                volume = float(ticker.get("volume", 0))
                
                if current_price > 0:
                    # Create a single-row DataFrame with realistic OHLCV
                    df = pd.DataFrame({
                        "Open": [open_price if open_price > 0 else current_price],
                        "High": [max(high_price, current_price) if high_price > 0 else current_price],
                        "Low": [min(low_price, current_price) if low_price > 0 else current_price], 
                        "Close": [current_price],
                        "Volume": [volume]
                    })
                    
                    # Set timestamp index
                    try:
                        df.index = pd.to_datetime([pd.Timestamp.now()])
                    except:
                        pass  # Continue without timestamp index
                    
                    logging.debug("Created OHLCV from ticker data")
                else:
                    logging.debug("No valid price in ticker data")
                    df = None
            except Exception as e:
                logging.debug(f"Failed to construct OHLCV from ticker: {e}")
                df = None
        
        # Case 4: FIXED: Try to get basic price from market_data
        elif "price" in market_data:
            try:
                price = float(market_data["price"])
                if price > 0:
                    # Create minimal OHLCV with current price
                    df = pd.DataFrame({
                        "Open": [price],
                        "High": [price],
                        "Low": [price],
                        "Close": [price],
                        "Volume": [market_data.get("volume", 0)]
                    })
                    logging.debug("Created minimal OHLCV from price data")
                else:
                    df = None
            except Exception as e:
                logging.debug(f"Failed to create OHLCV from price: {e}")
                df = None
        else:
            logging.debug("No usable price data found in market_data")
            return None

        # Validate we have a DataFrame
        if df is None or df.empty:
            logging.debug("No DataFrame created from market data")
            return None

        # Verify required columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            logging.debug(f"Missing required OHLCV columns: {missing}")
            return None

        # FIXED: Use gentler cleaning
        original_len = len(df)
        df = clean_ohlcv_data(df)
        
        if df.empty:
            logging.debug("OHLCV data is empty after cleaning")
            return None

        # FIXED: Less strict validation
        if not validate_ohlcv_data(df):
            # If validation fails but we have some data, try to salvage it
            if len(df) >= 1:
                logging.debug("Data failed validation but attempting to salvage")
                # Just ensure we have the basic columns
                df = df[required_columns].copy()
            else:
                logging.debug("OHLCV data failed validation and cannot be salvaged")
                return None

        # Ensure proper column order
        df = df[required_columns].copy()

        # FIXED: Add technical features with error handling
        try:
            # Only add features if we have enough data
            if len(df) >= 2:
                df['Pct_Change'] = df['Close'].pct_change()
            
            # Add moving averages only if we have sufficient data
            if len(df) >= 5:
                df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            
            if len(df) >= 10:
                df['MA_10'] = df['Close'].rolling(window=10, min_periods=5).mean()
            
            if len(df) >= 20:
                df['MA_20'] = df['Close'].rolling(window=20, min_periods=10).mean()
            
            # Add volume moving average
            if len(df) >= 5:
                df['Volume_MA'] = df['Volume'].rolling(window=5, min_periods=1).mean()
                
            # FIXED: Use ffill() instead of fillna(method='ffill')
            df = df.ffill()
            
            # Drop any remaining NaNs, but be gentle
            initial_len = len(df)
            df.dropna(inplace=True)
            
            # If we lost too much data, be more permissive
            if len(df) < initial_len * 0.7 and initial_len > 5:
                logging.debug("Too much data lost in final cleanup, being more permissive")
                # Restore from before dropna and just fill remaining NaNs
                df = df.ffill().bfill()
                # Only drop rows where Close is NaN (most critical)
                df = df[df['Close'].notna()]
            
        except Exception as e:
            logging.debug(f"Failed to add technical features: {e}")
            # Continue with basic OHLCV data
            pass

        # Final check
        if df.empty:
            logging.debug("OHLCV data is empty after processing")
            return None

        logging.debug(f"Market data successfully processed: {len(df)} candles (from {original_len} original)")
        return df

    except Exception as e:
        logging.error(f"Error processing market data: {e}")
        return None

def resample_data(df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
    """
    FIXED: Resample OHLCV data with better error handling
    """
    try:
        if df is None or df.empty:
            logging.debug("Cannot resample empty DataFrame")
            return df
            
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.debug("DataFrame index is not DatetimeIndex, cannot resample")
            return df
            
        # Updated frequency mapping
        freq_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D',
            '1w': '1W',
        }
        
        freq = freq_map.get(timeframe, '1h')
        
        # FIXED: Check if we have enough data for resampling
        if len(df) < 2:
            logging.debug(f"Insufficient data for resampling to {timeframe}")
            return df
        
        # Required columns for resampling
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 4:  # FIXED: Need at least OHLC
            logging.debug(f"Missing too many columns for resampling: {set(required_cols) - set(available_cols)}")
            return df
        
        # Build aggregation dict for available columns
        agg_dict = {}
        if 'Open' in available_cols:
            agg_dict['Open'] = 'first'
        if 'High' in available_cols:
            agg_dict['High'] = 'max'
        if 'Low' in available_cols:
            agg_dict['Low'] = 'min'
        if 'Close' in available_cols:
            agg_dict['Close'] = 'last'
        if 'Volume' in available_cols:
            agg_dict['Volume'] = 'sum'
        
        # FIXED: Resample with error handling
        try:
            resampled = df.resample(freq).agg(agg_dict)
            
            # Remove completely empty rows
            resampled = resampled.dropna(how='all')
            
            if resampled.empty:
                logging.debug(f"Resampling to {timeframe} resulted in empty DataFrame")
                return df
                
            logging.debug(f"Successfully resampled data from {len(df)} to {len(resampled)} candles")
            return resampled
        except Exception as e:
            logging.debug(f"Resampling operation failed: {e}")
            return df
        
    except Exception as e:
        logging.error(f"Resampling failed: {e}")
        return df

def create_multi_timeframe_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    FIXED: Create multiple timeframe datasets with better error handling
    """
    timeframes = {}
    
    try:
        if df is None or df.empty:
            logging.debug("Cannot create multi-timeframe data from empty DataFrame")
            return timeframes
            
        # Define timeframes
        tf_configs = {
            '1m': '1min',
            '5m': '5min', 
            '15m': '15min',
            '1h': '1h',
            '4h': '4h',
            '1d': '1D'
        }
        
        # Check if we have datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.debug("DataFrame index is not DatetimeIndex, returning original data")
            timeframes['original'] = df
            return timeframes
        
        # Calculate time span
        try:
            time_span = df.index.max() - df.index.min()
            time_span_hours = time_span.total_seconds() / 3600
        except:
            time_span_hours = 0
        
        for tf_name, freq in tf_configs.items():
            try:
                # FIXED: Skip timeframes that would result in very few data points
                min_hours_needed = {
                    '1d': 24,
                    '4h': 8,
                    '1h': 2,
                    '15m': 0.5,
                    '5m': 0.1,
                    '1m': 0.01
                }
                
                if time_span_hours < min_hours_needed.get(tf_name, 0):
                    logging.debug(f"Insufficient time span for {tf_name} timeframe")
                    continue
                
                # Required columns check
                required_cols = ['Open', 'High', 'Low', 'Close']
                if not all(col in df.columns for col in required_cols):
                    logging.debug(f"Missing required columns for {tf_name} timeframe")
                    continue
                
                # Build agg dict
                agg_dict = {
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last'
                }
                
                if 'Volume' in df.columns:
                    agg_dict['Volume'] = 'sum'
                
                # FIXED: Resample with error handling
                try:
                    resampled = df.resample(freq).agg(agg_dict)
                    resampled = resampled.dropna(how='all')
                    
                    if len(resampled) > 0:
                        timeframes[tf_name] = resampled
                        logging.debug(f"Created {tf_name} timeframe with {len(resampled)} candles")
                except Exception as e:
                    logging.debug(f"Failed to resample {tf_name}: {e}")
                    continue
                    
            except Exception as e:
                logging.debug(f"Failed to create {tf_name} timeframe: {e}")
                continue
        
        # If no timeframes were created, return original data
        if not timeframes:
            timeframes['original'] = df
                
        return timeframes
        
    except Exception as e:
        logging.error(f"Multi-timeframe creation failed: {e}")
        return {'original': df}

def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
    """
    FIXED: Calculate volume profile with better error handling
    """
    try:
        if df is None or df.empty:
            logging.debug("Cannot calculate volume profile from empty DataFrame")
            return {}
            
        # Check required columns
        required_cols = ['High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.debug(f"Missing required columns for volume profile: {missing_cols}")
            return {}
        
        # Use the data as-is if not enough for resampling
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 24:
            try:
                hourly_data = df.resample('1h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min', 
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna(how='all')
            except Exception as e:
                logging.debug(f"Failed to resample for volume profile: {e}")
                hourly_data = df
        else:
            hourly_data = df
            
        if len(hourly_data) == 0:
            logging.debug("No data available after resampling for volume profile")
            return {}
            
        # Calculate VWAP and other volume metrics
        try:
            typical_price = (hourly_data['High'] + hourly_data['Low'] + hourly_data['Close']) / 3
            volume_price = typical_price * hourly_data['Volume']
            
            # Avoid division by zero
            total_volume = hourly_data['Volume'].sum()
            if total_volume > 0:
                vwap = volume_price.sum() / total_volume
            else:
                vwap = hourly_data['Close'].iloc[-1] if len(hourly_data) > 0 else 0
        except Exception as e:
            logging.debug(f"Failed to calculate VWAP: {e}")
            vwap = hourly_data['Close'].iloc[-1] if len(hourly_data) > 0 else 0
        
        # Create price bins
        try:
            price_min = hourly_data['Low'].min()
            price_max = hourly_data['High'].max()
            
            if price_max <= price_min or not np.isfinite(price_min) or not np.isfinite(price_max):
                logging.debug("Invalid price range for volume profile")
                return {'vwap': vwap, 'total_volume': total_volume}
                
            price_range = price_max - price_min
            bin_size = price_range / bins
            
            volume_profile = {}
            for i in range(bins):
                price_level = price_min + (i * bin_size)
                
                # Find volume at this price level
                mask = (
                    (hourly_data['Low'] <= price_level) & 
                    (hourly_data['High'] >= price_level)
                )
                
                volume_at_level = hourly_data.loc[mask, 'Volume'].sum()
                volume_profile[f'{price_level:.4f}'] = volume_at_level
            
            return {
                'volume_profile': volume_profile,
                'vwap': vwap,
                'total_volume': total_volume,
                'price_range': {
                    'min': price_min,
                    'max': price_max
                }
            }
        except Exception as e:
            logging.debug(f"Failed to create volume profile bins: {e}")
            return {
                'vwap': vwap,
                'total_volume': total_volume
            }
        
    except Exception as e:
        logging.error(f"Volume profile calculation failed: {e}")
        return {}

def detect_market_sessions(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    FIXED: Detect different market sessions with better error handling
    """
    try:
        sessions = {}
        
        if df is None or df.empty:
            logging.debug("Cannot detect market sessions from empty DataFrame")
            return sessions
        
        # Check if we have datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.debug("DataFrame index is not DatetimeIndex, cannot detect market sessions")
            return sessions
        
        # Required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logging.debug(f"Missing required columns for session detection: {missing_cols}")
            return sessions
        
        # Use the data as-is if not enough for hourly resampling
        if len(df) > 24:
            try:
                hourly_data = df.resample('1h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last', 
                    'Volume': 'sum' if 'Volume' in df.columns else 'first'
                }).dropna(how='all')
            except Exception as e:
                logging.debug(f"Failed to resample for session detection: {e}")
                hourly_data = df
        else:
            hourly_data = df
        
        if len(hourly_data) == 0:
            return sessions
        
        # Define session hours (UTC)
        session_definitions = {
            'asian': (0, 8),
            'london': (8, 16),
            'new_york': (13, 21),
            'sydney': (21, 5)
        }
        
        for session_name, (start_hour, end_hour) in session_definitions.items():
            try:
                if start_hour < end_hour:
                    session_mask = (
                        (hourly_data.index.hour >= start_hour) & 
                        (hourly_data.index.hour < end_hour)
                    )
                else:  # Crosses midnight
                    session_mask = (
                        (hourly_data.index.hour >= start_hour) | 
                        (hourly_data.index.hour < end_hour)
                    )
                
                session_data = hourly_data.loc[session_mask]
                if len(session_data) > 0:
                    sessions[session_name] = session_data
                    logging.debug(f"Detected {session_name} session with {len(session_data)} candles")
                    
            except Exception as e:
                logging.debug(f"Failed to detect {session_name} session: {e}")
                continue
        
        return sessions
        
    except Exception as e:
        logging.error(f"Market session detection failed: {e}")
        return {}

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """FIXED: Get a summary of the processed data for debugging"""
    if df is None or df.empty:
        return {"status": "empty", "rows": 0, "columns": []}
    
    try:
        summary = {
            "status": "valid",
            "rows": len(df),
            "columns": list(df.columns),
            "data_quality": "good" if validate_ohlcv_data(df) else "fair",
            "has_nan": df.isnull().any().any(),
            "nan_count": df.isnull().sum().sum()
        }
        
        # Add time range if datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            summary["time_range"] = {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            }
        else:
            summary["time_range"] = "N/A - no datetime index"
            
        # Add price range if Close column exists
        if 'Close' in df.columns:
            summary["price_range"] = {
                "min": float(df['Close'].min()),
                "max": float(df['Close'].max())
            }
        
        return summary
    except Exception as e:
        return {"status": "error", "error": str(e)}