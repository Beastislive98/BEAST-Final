import os
import certifi
import subprocess
import json
import logging
import time
import random
import requests
import pandas as pd
from typing import Dict, Any, Optional, List
import threading

BASE_URL = "https://fapi.binance.com"
os.environ["SSL_CERT_FILE"] = certifi.where()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants for retry logic
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2
RATE_LIMIT_STATUS = 429

# FIXED: Server time synchronization to prevent timestamp errors
_server_time_offset = 0
_last_time_sync = 0
TIME_SYNC_INTERVAL = 1800  # Sync every 30 minutes
_sync_lock = threading.Lock()

# FIXED: Connection pooling for better performance
_session = None
_session_lock = threading.Lock()

def get_session():
    """Get or create a requests session with connection pooling"""
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
            _session.headers.update({
                'User-Agent': 'BEAST-Bot/2.0',
                'Connection': 'keep-alive'
            })
            # Connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=0  # We handle retries manually
            )
            _session.mount('https://', adapter)
            _session.mount('http://', adapter)
        return _session

def sync_server_time():
    """FIXED: Synchronize with Binance server time to prevent timestamp errors"""
    global _server_time_offset, _last_time_sync
    
    current_time = time.time()
    with _sync_lock:
        if current_time - _last_time_sync < TIME_SYNC_INTERVAL:
            return  # Skip if recently synced
    
    try:
        session = get_session()
        response = session.get(f"{BASE_URL}/fapi/v1/time", timeout=10)
        if response.status_code == 200:
            server_time = response.json()["serverTime"]
            local_time = int(time.time() * 1000)
            
            with _sync_lock:
                _server_time_offset = server_time - local_time
                _last_time_sync = current_time
                
            logging.info(f"Time synchronized with Binance server. Offset: {_server_time_offset}ms")
        else:
            logging.warning(f"Failed to sync time: HTTP {response.status_code}")
    except Exception as e:
        logging.warning(f"Time sync failed: {e}")
        # Try NTP as fallback
        try:
            import ntplib
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request('pool.ntp.org', version=3, timeout=5)
            ntp_time = int(response.tx_time * 1000)
            local_time = int(time.time() * 1000)
            
            with _sync_lock:
                _server_time_offset = ntp_time - local_time
                _last_time_sync = current_time
                
            logging.info(f"Time synchronized with NTP. Offset: {_server_time_offset}ms")
        except Exception as ntp_error:
            logging.debug(f"NTP sync also failed: {ntp_error}")

def get_server_time() -> int:
    """Get current time adjusted for server offset"""
    sync_server_time()
    with _sync_lock:
        return int(time.time() * 1000) + _server_time_offset

def safe_curl_json(url: str, max_retries: int = MAX_RETRIES) -> dict:
    """
    Uses system `curl` to fetch JSON from Binance with improved retry logic.
    """
    for attempt in range(max_retries):
        try:
            # Add a small random delay between requests
            time.sleep(random.uniform(0.1, 0.5))
            
            # FIXED: Increased timeouts for better reliability
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-f",
                    "-k",
                    "-H", "User-Agent: BEAST-Bot",
                    "-H", "Cache-Control: no-cache",
                    "-m", "30",  # FIXED: Increased from 15 to 30 seconds
                    "--retry", "3",
                    "--retry-delay", "2",
                    "--connect-timeout", "15",  # FIXED: Increased connection timeout
                    url
                ],
                capture_output=True,
                text=True,
                timeout=40  # FIXED: Increased total timeout
            )
            
            if result.returncode == 0 and result.stdout:
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    logging.warning(f"[CURL] Invalid JSON response from: {url}")
            else:
                logging.warning(f"[CURL] Failed for: {url} - Return code: {result.returncode}")
                if result.stderr:
                    logging.warning(f"[CURL] Error: {result.stderr.strip()}")
        
        except subprocess.TimeoutExpired:
            logging.warning(f"[CURL] Timeout for: {url}")
        except Exception as e:
            logging.warning(f"[CURL] Exception for {url}: {str(e)}")
        
        # Exponential backoff with jitter
        wait_time = (RETRY_DELAY_BASE ** attempt) + random.uniform(0.5, 2.0)
        if attempt < max_retries - 1:
            logging.info(f"Retrying in {wait_time:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)
    
    return {}

def safe_request_json(url: str, max_retries: int = MAX_RETRIES) -> dict:
    """
    FIXED: Improved requests method with better error handling and timeouts
    """
    session = get_session()
    
    for attempt in range(max_retries):
        try:
            # Add delay between requests
            time.sleep(random.uniform(0.1, 0.5))
            
            # FIXED: Increased timeout
            response = session.get(url, timeout=30, verify=True)
            
            # Handle rate limiting
            if response.status_code == RATE_LIMIT_STATUS:
                retry_after = int(response.headers.get('Retry-After', RETRY_DELAY_BASE * 2))
                logging.warning(f"Rate limited. Waiting for {retry_after} seconds")
                time.sleep(retry_after)
                continue
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.warning(f"[REQUESTS] HTTP {response.status_code} for: {url}")
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        logging.error(f"API Error: {error_data}")
                    except:
                        logging.error(f"API Error: {response.text}")
        
        except requests.exceptions.Timeout:
            logging.warning(f"[REQUESTS] Timeout for {url} (attempt {attempt+1}/{max_retries})")
        except requests.exceptions.ConnectionError as e:
            logging.warning(f"[REQUESTS] Connection error for {url}: {str(e)}")
        except requests.RequestException as e:
            logging.warning(f"[REQUESTS] Exception for {url}: {str(e)}")
        except json.JSONDecodeError:
            logging.warning(f"[REQUESTS] Invalid JSON response from: {url}")
        
        # Exponential backoff with jitter
        if attempt < max_retries - 1:
            wait_time = (RETRY_DELAY_BASE ** attempt) + random.uniform(0.5, 2.0)
            logging.info(f"Retrying in {wait_time:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(wait_time)
    
    return {}

def enhanced_request_with_backoff(url: str, max_retries: int = MAX_RETRIES, 
                                 headers: Optional[Dict[str, str]] = None, 
                                 params: Optional[Dict[str, Any]] = None, 
                                 timeout: int = 30) -> Dict[str, Any]:  # FIXED: Increased default timeout
    """
    FIXED: Enhanced request method with better error handling
    """
    session = get_session()
    
    # Merge headers
    request_headers = session.headers.copy()
    if headers:
        request_headers.update(headers)
    
    for attempt in range(max_retries):
        try:
            # Add jitter to prevent thundering herd
            if attempt > 0:
                jitter = random.uniform(0.1, 0.5)
                delay = (RETRY_DELAY_BASE ** attempt) + jitter
                logging.info(f"Retry attempt {attempt+1}/{max_retries} with {delay:.2f}s delay for URL: {url}")
                time.sleep(delay)
            
            response = session.get(
                url,
                headers=request_headers,
                params=params,
                timeout=timeout,
                verify=True
            )
            
            # Handle rate limiting
            if response.status_code == RATE_LIMIT_STATUS:
                retry_after = int(response.headers.get('Retry-After', RETRY_DELAY_BASE * 2))
                actual_wait = retry_after * 1.2  # Add 20% buffer
                logging.warning(f"Rate limited by Binance API. Waiting for {actual_wait:.2f} seconds")
                time.sleep(actual_wait)
                continue
            
            if response.status_code == 200:
                return response.json()
            elif 500 <= response.status_code < 600:
                logging.warning(f"Binance server error {response.status_code}, will retry")
            else:
                logging.warning(f"HTTP {response.status_code} for: {url}")
                if 400 <= response.status_code < 500 and response.status_code != RATE_LIMIT_STATUS:
                    try:
                        error_data = response.json()
                        logging.error(f"API error: {error_data}")
                    except:
                        logging.error(f"API error: {response.text}")
                    break  # Don't retry 4xx errors
        
        except requests.exceptions.Timeout:
            logging.warning(f"Request timeout for {url} (attempt {attempt+1}/{max_retries})")
        except requests.exceptions.ConnectionError as e:
            logging.warning(f"Connection error for {url} (attempt {attempt+1}/{max_retries}): {e}")
        except requests.RequestException as e:
            logging.warning(f"Request exception for {url}: {str(e)}")
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON response from: {url}")
    
    return {}

def fetch_data(url: str) -> dict:
    """
    Try both methods to increase chances of success.
    """
    data = enhanced_request_with_backoff(url)
    
    if not data:
        logging.info(f"Enhanced requests failed, trying curl fallback for: {url}")
        data = safe_curl_json(url)
    
    return data

def get_historical_klines(symbol: str, days: int = 7, interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    FIXED: Get historical klines with server time synchronization
    """
    try:
        # FIXED: Use server time instead of local time
        end_time = get_server_time()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1500
        }
        
        # FIXED: Use enhanced request with better timeout handling
        data = enhanced_request_with_backoff(
            url,
            params=params,
            timeout=30,  # Increased timeout
            max_retries=3
        )
        
        if not data:
            logging.warning(f"No kline data returned for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Clean and format data
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.dropna()
        
        if len(df) > 0:
            logging.debug(f"Got {len(df)} klines for {symbol} ({days}d, {interval})")
            return df
        else:
            logging.warning(f"No valid kline data after cleanup for {symbol}")
            return None
                
    except Exception as e:
        logging.error(f"Unexpected error getting klines for {symbol}: {e}")
        return None

def get_extended_historical_data(symbol: str, days: int = 30, interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    Get extended historical data with fallback periods
    """
    fallback_periods = [days, days * 2, days * 3, 60]
    fallback_intervals = [interval, "5m", "15m", "1h"]
    
    for period in fallback_periods:
        for intv in fallback_intervals:
            try:
                logging.info(f"Attempting to fetch {period} days of {intv} data for {symbol}")
                
                df = get_historical_klines(symbol, period, intv)
                
                if df is not None and len(df) >= 50:
                    logging.info(f"Successfully got {len(df)} records for {symbol} with {period}d/{intv}")
                    return df
                    
                elif df is not None:
                    logging.debug(f"Got {len(df)} records for {symbol} with {period}d/{intv} - insufficient")
                    
            except Exception as e:
                logging.warning(f"Failed to get {period}d/{intv} data for {symbol}: {e}")
                continue
    
    logging.error(f"Could not get sufficient historical data for {symbol} after all fallback attempts")
    return None

# FIXED: Cache for symbol validation to reduce API calls
_symbol_cache = {}
_symbol_cache_lock = threading.Lock()
_symbol_cache_ttl = 3600  # 1 hour cache

def validate_symbol_active(symbol: str) -> bool:
    """
    FIXED: Check if a symbol is active with caching to reduce API calls
    """
    current_time = time.time()
    
    # Check cache first
    with _symbol_cache_lock:
        if symbol in _symbol_cache:
            cached_time, is_active = _symbol_cache[symbol]
            if current_time - cached_time < _symbol_cache_ttl:
                return is_active
    
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        
        # FIXED: Use enhanced request with better timeout
        data = enhanced_request_with_backoff(url, timeout=30, max_retries=3)
        
        if not data:
            logging.error("Failed to get exchange info")
            return False
            
        symbols = data.get('symbols', [])
        
        for s in symbols:
            if s.get('symbol') == symbol:
                status = s.get('status', '')
                contract_type = s.get('contractType', '')
                
                is_active = status == 'TRADING' and contract_type == 'PERPETUAL'
                
                # Cache the result
                with _symbol_cache_lock:
                    _symbol_cache[symbol] = (current_time, is_active)
                
                if not is_active:
                    logging.warning(f"Symbol {symbol} status: {status}, contract: {contract_type}")
                
                return is_active
        
        # Symbol not found
        with _symbol_cache_lock:
            _symbol_cache[symbol] = (current_time, False)
        
        logging.warning(f"Symbol {symbol} not found in exchange info")
        return False
            
    except Exception as e:
        logging.error(f"Error validating symbol {symbol}: {e}")
        return False

def get_active_futures_symbols() -> List[str]:
    """
    FIXED: Get active symbols with better error handling
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        
        # FIXED: Use enhanced request method
        data = enhanced_request_with_backoff(url, timeout=30, max_retries=3)
        
        if not data:
            logging.error("Failed to get exchange info")
            return []
        
        symbols = data.get('symbols', [])
        
        active_symbols = []
        for symbol_info in symbols:
            symbol = symbol_info.get('symbol', '')
            status = symbol_info.get('status', '')
            contract_type = symbol_info.get('contractType', '')
            
            if (status == 'TRADING' and 
                contract_type == 'PERPETUAL' and 
                symbol.endswith('USDT')):
                active_symbols.append(symbol)
        
        logging.info(f"Found {len(active_symbols)} active USDT perpetual futures")
        return active_symbols
        
    except Exception as e:
        logging.error(f"Error getting active symbols: {e}")
        return []

# FIXED: Circuit breaker for problematic symbols
_circuit_breakers = {}
_circuit_breaker_lock = threading.Lock()

def _is_circuit_open(symbol: str) -> bool:
    """Check if circuit breaker is open for a symbol"""
    with _circuit_breaker_lock:
        if symbol in _circuit_breakers:
            failure_count, last_failure = _circuit_breakers[symbol]
            # Reset after 10 minutes
            if time.time() - last_failure > 600:
                del _circuit_breakers[symbol]
                return False
            return failure_count >= 5
        return False

def _record_failure(symbol: str):
    """Record a failure for circuit breaker"""
    with _circuit_breaker_lock:
        if symbol in _circuit_breakers:
            failure_count, _ = _circuit_breakers[symbol]
            _circuit_breakers[symbol] = (failure_count + 1, time.time())
        else:
            _circuit_breakers[symbol] = (1, time.time())

def _record_success(symbol: str):
    """Record a success for circuit breaker"""
    with _circuit_breaker_lock:
        if symbol in _circuit_breakers:
            del _circuit_breakers[symbol]

def get_futures_market_data(symbol: str, circuit_breaker: bool = True) -> Optional[Dict[str, Any]]:
    """
    FIXED: Get comprehensive futures market data with circuit breaker
    """
    # Check circuit breaker
    if circuit_breaker and _is_circuit_open(symbol):
        logging.debug(f"Circuit breaker open for {symbol}, skipping")
        return None
    
    try:
        # Validate symbol is active
        if not validate_symbol_active(symbol):
            logging.error(f"Symbol {symbol} is not active or tradeable")
            if circuit_breaker:
                _record_failure(symbol)
            return None
        
        # Get basic ticker data
        ticker_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        params = {"symbol": symbol}
        
        # FIXED: Use enhanced request with better timeout
        ticker_data = enhanced_request_with_backoff(
            ticker_url,
            params=params,
            timeout=20,  # Increased timeout
            max_retries=3
        )
        
        if not ticker_data:
            logging.error(f"Failed to get ticker data for {symbol}")
            if circuit_breaker:
                _record_failure(symbol)
            return None
        
        # Get order book data
        depth_url = "https://fapi.binance.com/fapi/v1/depth"
        depth_params = {"symbol": symbol, "limit": 20}
        
        depth_data = enhanced_request_with_backoff(
            depth_url,
            params=depth_params,
            timeout=15,
            max_retries=2
        )
        
        if not depth_data:
            logging.warning(f"Could not get depth data for {symbol}")
            depth_data = {"bids": [], "asks": []}
        
        # Get historical klines
        klines_df = get_extended_historical_data(symbol, days=7, interval="1m")
        
        # Build market data
        market_data = {
            "symbol": symbol,
            "ticker": ticker_data,
            "price": float(ticker_data.get("lastPrice", 0)),
            "volume": float(ticker_data.get("volume", 0)),
            "price_change_pct": float(ticker_data.get("priceChangePercent", 0)),
            "bid_depth": depth_data.get("bids", []),
            "ask_depth": depth_data.get("asks", []),
            "klines": klines_df,
            "timestamp": get_server_time(),
            "valid_endpoints": 2 if klines_df is not None else 1
        }
        
        # Record success
        if circuit_breaker:
            _record_success(symbol)
        
        logging.debug(f"Successfully assembled market data for {symbol}")
        return market_data
        
    except Exception as e:
        logging.error(f"Error getting futures market data for {symbol}: {e}")
        if circuit_breaker:
            _record_failure(symbol)
        return None

def classify_symbol_behavior(market_data: Dict[str, Any]) -> str:
    """
    Classify trading behavior based on market data with improved error handling.
    """
    try:
        ticker = market_data.get("ticker", {})
        
        # Handle potential missing or invalid data
        try:
            volume = float(ticker.get("quoteVolume", 0))
        except (ValueError, TypeError):
            volume = 0
            
        try:
            price_change = float(ticker.get("priceChangePercent", 0))
        except (ValueError, TypeError):
            price_change = 0
            
        try:
            high_price = float(ticker.get("highPrice", 0))
            low_price = float(ticker.get("lowPrice", 0))
            last_price = float(ticker.get("lastPrice", 0))
        except (ValueError, TypeError):
            high_price = low_price = last_price = 0

        # Avoid division by zero
        near_high = abs(last_price - high_price) / high_price < 0.01 if high_price > 0 else False
        near_low = abs(last_price - low_price) / low_price < 0.01 if low_price > 0 else False

        if price_change > 5 and volume > 2_000_000:
            return "rally"
        elif price_change < -5 and volume > 2_000_000:
            return "fall"
        elif near_high:
            return "breakout"
        elif near_low:
            return "breakdown"
        elif 1 < price_change < 3:
            return "pullback"
        else:
            return "neutral"
    except Exception as e:
        logging.error(f"Behavior classification failed: {e}")
        return "unknown"

def get_futures_account_info(api_key: str, secret_key: str) -> Optional[Dict[str, Any]]:
    """
    Placeholder for future authenticated queries.
    """
    return None