import logging
import requests
import time
import hmac
import hashlib
import urllib.parse
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import pytz
import os
from dotenv import load_dotenv
import json
import threading
import atexit

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_system.log"),
        logging.StreamHandler()
    ]
)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
BASE_URL = "https://fapi.binance.com"

# FIXED: Server time synchronization
_server_time_offset = 0
_last_time_sync = 0
TIME_SYNC_INTERVAL = 1800  # Sync every 30 minutes
_sync_lock = threading.Lock()

def sync_server_time():
    """FIXED: Synchronize with Binance server time to prevent timestamp errors"""
    global _server_time_offset, _last_time_sync
    
    current_time = time.time()
    with _sync_lock:
        if current_time - _last_time_sync < TIME_SYNC_INTERVAL:
            return
    
    try:
        response = requests.get(f"{BASE_URL}/fapi/v1/time", timeout=15)
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

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, max_calls: int = 10, time_frame: int = 60):
        self.max_calls = max_calls
        self.time_frame = time_frame  # in seconds
        self.calls = []
        self.lock = threading.Lock()

    def can_call(self) -> bool:
        """Check if we can make an API call without exceeding rate limits"""
        current_time = time.time()
        with self.lock:
            # Remove calls older than the time frame
            self.calls = [t for t in self.calls if current_time - t < self.time_frame]
            # Check if we're under the limit
            if len(self.calls) < self.max_calls:
                self.calls.append(current_time)
                return True
            return False

    def wait_if_needed(self):
        """Wait until we can make a call if rate limited"""
        while not self.can_call():
            time.sleep(1)

class CapitalManager:
    def __init__(self, max_exposure_pct: float = 90.0, max_open_trades: int = 25, max_drawdown_pct: float = 10.0, initial_capital: float = None):
        self.max_exposure_pct = max_exposure_pct / 100
        self.max_open_trades = max_open_trades
        self.max_drawdown_pct = max_drawdown_pct / 100
        self.open_trades = []
        self.pending_trades = []  # Queue for trades that couldn't be executed
        self.realized_loss = 0.0
        self.realized_profit = 0.0
        
        # Enhanced initialization from enhancement code
        if initial_capital:
            self.current_capital = initial_capital
        else:
            self.current_capital = self._load_initial_capital()
        
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_total_exposure = 0.20  # 20% max total exposure
        self.active_positions = {}
        self.position_count = 0
        
        # Thread safety locks
        self.trades_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.capital_lock = threading.Lock()
        
        # Rate limiters for API calls
        self.balance_limiter = RateLimiter(max_calls=5, time_frame=60)  # FIXED: Reduced to 5 calls per minute
        
        # Track capital dynamically
        self.starting_day_capital = self.current_capital
        
        self.last_reset_day = self.get_ist_day()
        self.last_balance_check = 0
        self.daily_profit = 0.0
        self.daily_target_pct = 0.02  # 2% daily target
        self.risk_profile = "aggressive"  # Start aggressive
        self.balance_cache_ttl = 60  # FIXED: Increased cache TTL to 60 seconds
        
        # FIXED: Connection pooling for better reliability
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CapitalManager/2.0',
            'Connection': 'keep-alive'
        })
        
        # Connection pooling adapter
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=0  # We handle retries manually
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        # Dependencies check - warn but don't fail if missing
        self._check_dependencies()
        
        # Set up background threads for periodic tasks
        self.running = True
        self.start_background_threads()
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        logging.info(f"CapitalManager initialized with ${self.starting_day_capital} and {max_exposure_pct}% max exposure")

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import pytz
        except ImportError:
            logging.warning("pytz module not found. Timezone functionality may not work correctly.")
            
        # Try to set up price fetcher
        try:
            self.get_price_fetcher()
        except Exception as e:
            logging.warning(f"Could not initialize price fetcher: {e}")

    def start_background_threads(self):
        """Start background threads for periodic tasks"""
        # Thread for balance updates
        self.balance_thread = threading.Thread(target=self._balance_update_loop, name="BalanceUpdater")
        self.balance_thread.daemon = True
        self.balance_thread.start()
        
        # Thread for processing pending trades
        self.queue_thread = threading.Thread(target=self._process_pending_trades, name="QueueProcessor")
        self.queue_thread.daemon = True
        self.queue_thread.start()
        
        logging.info("Started background threads for balance updates and trade processing")

    def shutdown(self):
        """Safely shut down the capital manager"""
        logging.info("Shutting down CapitalManager...")
        self.running = False
        
        try:
            self.balance_thread.join(timeout=5)
            self.queue_thread.join(timeout=5)
        except Exception as e:
            logging.error(f"Error during thread shutdown: {e}")
            
        self._save_capital_state()  # Save final state
        self.session.close()  # FIXED: Close session properly
        logging.info("CapitalManager shutdown complete")

    def _balance_update_loop(self):
        """FIXED: Periodically update balance with better error handling"""
        while self.running:
            try:
                # FIXED: Only update if enough time has passed and rate limiter allows
                if self.balance_limiter.can_call():
                    self.get_current_balance(force_refresh=True)
                time.sleep(120)  # FIXED: Check every 2 minutes instead of 30 seconds
            except Exception as e:
                logging.error(f"Balance update error: {e}")
                time.sleep(300)  # Wait 5 minutes after error

    def _load_initial_capital(self) -> float:
        """
        Load the initial capital from config or use default of 32.0
        """
        try:
            # Check if a capital history file exists
            if os.path.exists("logs/capital_history.json"):
                with open("logs/capital_history.json", "r") as f:
                    data = json.load(f)
                    if "current_capital" in data:
                        return data["current_capital"]
        except Exception as e:
            logging.warning(f"Could not load initial capital: {e}")
            
        # Default initial capital if not found
        return 32.0
        
    def _save_capital_state(self):
        """
        Save current capital state to disk for persistence
        """
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            with self.capital_lock:
                current_capital = self.current_capital
                starting_day_capital = self.starting_day_capital
                daily_profit = self.daily_profit
                realized_profit = self.realized_profit
                realized_loss = self.realized_loss
                risk_profile = self.risk_profile
            
            with self.trades_lock:
                open_trades_count = len(self.open_trades)
            
            with self.pending_lock:
                pending_trades_count = len(self.pending_trades)
            
            # Save current state
            data = {
                "current_capital": current_capital,
                "starting_day_capital": starting_day_capital,
                "daily_profit": daily_profit,
                "realized_profit": realized_profit,
                "realized_loss": realized_loss,
                "timestamp": datetime.now().isoformat(),
                "risk_profile": risk_profile,
                "open_trades_count": open_trades_count,
                "pending_trades_count": pending_trades_count
            }
            
            with open("logs/capital_history.json", "w") as f:
                json.dump(data, f, indent=2)
                
            logging.debug(f"Saved capital state: ${current_capital:.2f}")
            
        except Exception as e:
            logging.error(f"Failed to save capital state: {e}")

    def get_ist_day(self):
        """Get current day in Indian Standard Time"""
        try:
            return datetime.now(pytz.timezone('Asia/Kolkata')).date()
        except:
            # Fallback to UTC if timezone fails
            return datetime.now().date()

    def register_trade(self, trade: Dict[str, Any]):
        """Register a new trade in the system"""
        with self.trades_lock:
            self.open_trades.append(trade)
            trades_count = len(self.open_trades)
            
        logging.info(f"Trade registered. Total open trades: {trades_count}")

    def close_trade(self, trade_id: str, pnl: float):
        """
        Enhanced close trade method that handles both old and new systems
        """
        found = False
        
        # Handle both open_trades and active_positions
        with self.trades_lock:
            for i, trade in enumerate(self.open_trades):
                if trade.get('trade_id') == trade_id:
                    self.open_trades.pop(i)
                    found = True
                    break
        
        # Also check active_positions (from enhancement code)
        if trade_id in self.active_positions:
            del self.active_positions[trade_id]
            found = True
            logging.info(f"Closed trade {trade_id} with PnL: ${pnl:.2f}")
        
        if not found:
            logging.warning(f"Trade {trade_id} not found in open trades or active positions")
        
        # Update capital based on PnL
        with self.capital_lock:
            self.current_capital += pnl
            
            if pnl >= 0:
                self.realized_profit += pnl
                self.daily_profit += pnl
            else:
                self.realized_loss += abs(pnl)
                self.daily_profit += pnl  # Can be negative
                
            # Check if we've hit daily profit target
            if self.daily_profit >= self.starting_day_capital * self.daily_target_pct:
                self.risk_profile = "moderate"  # Switch to moderate after target
                logging.info(f"Daily profit target achieved: ${self.daily_profit:.2f}. Switching to {self.risk_profile} risk profile")
            
            current_capital = self.current_capital
        
        # Save trade result to database for analysis
        self._save_trade_result(trade_id, pnl)
        
        # Save updated capital state
        self._save_capital_state()
        
        # Process pending trades after a closure
        self._process_pending_trades_now()
        
        logging.info(f"Trade {trade_id} closed with PnL: ${pnl:.2f}. Current capital: ${current_capital:.2f}")
        return current_capital

    def _save_trade_result(self, trade_id: str, pnl: float):
        """Save trade result to database for future analysis"""
        try:
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            
            with self.capital_lock:
                capital_after = self.current_capital
                risk_profile = self.risk_profile
            
            # This could be expanded to save to SQLite or other DB
            with open("logs/trade_results.json", "a") as f:
                result = {
                    "trade_id": trade_id,
                    "pnl": pnl,
                    "timestamp": datetime.now().isoformat(),
                    "capital_after": capital_after,
                    "risk_profile": risk_profile
                }
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            logging.error(f"Failed to save trade result: {e}")

    def get_price_fetcher(self):
        """Get or create a price fetcher for market prices"""
        if not hasattr(self, "price_fetcher"):
            try:
                from price_fetcher import PriceFetcher
                self.price_fetcher = PriceFetcher()
                logging.info("Using PriceFetcher module for price data")
            except ImportError:
                # Simple fallback price fetcher that uses Binance API
                class SimplePriceFetcher:
                    def __init__(self, session):
                        self.price_cache = {}
                        self.cache_time = {}
                        self.cache_ttl = 30  # FIXED: Increased cache TTL to 30 seconds
                        self.rate_limiter = RateLimiter(max_calls=20, time_frame=60)
                        self.session = session  # FIXED: Use shared session
                        
                    def get_latest_price(self, symbol: str) -> float:
                        current_time = time.time()
                        
                        # Return cached price if fresh
                        if symbol in self.price_cache and (current_time - self.cache_time.get(symbol, 0)) < self.cache_ttl:
                            return self.price_cache[symbol]
                            
                        try:
                            # Wait if we're rate limited
                            self.rate_limiter.wait_if_needed()
                            
                            url = f"{BASE_URL}/fapi/v1/ticker/price?symbol={symbol}"
                            # FIXED: Use session and increased timeout
                            response = self.session.get(url, timeout=20)
                            
                            if response.status_code == 200:
                                data = response.json()
                                price = float(data.get("price", 0))
                                
                                # Update cache
                                self.price_cache[symbol] = price
                                self.cache_time[symbol] = current_time
                                
                                return price
                            else:
                                logging.warning(f"Price fetch HTTP error {response.status_code} for {symbol}")
                            return 0
                        except Exception as e:
                            logging.error(f"Price fetch error for {symbol}: {e}")
                            return 0
                
                self.price_fetcher = SimplePriceFetcher(self.session)
                logging.info("Using SimplePriceFetcher fallback for price data")
        
        return self.price_fetcher

    def get_current_balance(self, force_refresh=False) -> float:
        """
        FIXED: Enhanced get current balance method with better error handling
        """
        current_time = time.time()
        
        with self.capital_lock:
            # Return cached value if it's fresh enough
            if not force_refresh and (current_time - self.last_balance_check) < self.balance_cache_ttl:
                return self.current_capital
                
        # Try to get actual balance from Binance
        try:
            account_info = self._get_futures_account_info()
            if account_info and 'totalWalletBalance' in account_info:
                with self.capital_lock:
                    self.current_capital = float(account_info['totalWalletBalance'])
                    self.last_balance_check = current_time
                    current_capital = self.current_capital
                
                # Save updated capital state
                self._save_capital_state()
                
                logging.debug(f"Updated capital from Binance: ${current_capital:.2f}")
            else:
                logging.warning("Failed to get account balance from Binance, using local tracking")
                with self.capital_lock:
                    current_capital = self.current_capital
        except Exception as e:
            logging.error(f"Error getting account balance: {e}")
            with self.capital_lock:
                current_capital = self.current_capital
            
        return current_capital

    def _get_futures_account_info(self) -> Optional[Dict[str, Any]]:
        """FIXED: Get account info with proper timestamp handling and error recovery"""
        if not API_KEY or not API_SECRET:
            logging.debug("Missing API credentials. Could not fetch account info.")
            return None
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # FIXED: Use server time instead of local time
                timestamp = get_server_time()
                
                endpoint = "/fapi/v2/account"
                params = {
                    'timestamp': timestamp,
                    'recvWindow': 20000  # FIXED: Increased receive window to 20 seconds
                }
                
                query_string = urllib.parse.urlencode(params)
                signature = hmac.new(
                    API_SECRET.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                params['signature'] = signature
                headers = {
                    'X-MBX-APIKEY': API_KEY,
                    'User-Agent': 'CapitalManager/2.0'
                }
                
                url = f"{BASE_URL}{endpoint}"
                # FIXED: Use session and increased timeout
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=30  # FIXED: Increased timeout from 15 to 30 seconds
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('msg', 'Unknown error')
                        
                        if 'recvWindow' in error_msg or 'Timestamp' in error_msg:
                            logging.warning(f"Timestamp error on attempt {attempt + 1}: {error_msg}")
                            # Force time sync and retry
                            global _last_time_sync
                            with _sync_lock:
                                _last_time_sync = 0  # Force resync
                            sync_server_time()
                            
                            if attempt < max_retries - 1:
                                time.sleep(2)  # Wait before retry
                                continue
                        else:
                            logging.error(f"API error: {error_msg}")
                            break
                    except:
                        logging.error(f"API error: {response.text}")
                        break
                else:
                    logging.error(f"API error: {response.status_code} - {response.text}")
                    break
                    
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout getting account info (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait before retry
                    continue
            except requests.exceptions.ConnectionError:
                logging.warning(f"Connection error getting account info (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(10)  # Wait longer for connection issues
                    continue
            except Exception as e:
                logging.error(f"Error getting account info (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                    
        return None

    def can_open_new_trade(self, new_trade_capital: float) -> bool:
        """
        Check if a new trade can be opened based on capital and risk limits
        """
        current_day = self.get_ist_day()
        
        with self.capital_lock:
            if current_day != self.last_reset_day:
                self.realized_loss = 0.0
                self.realized_profit = 0.0
                self.daily_profit = 0.0
                self.risk_profile = "aggressive"  # Reset to aggressive each day
                self.last_reset_day = current_day
                logging.info("Daily stats reset")

        # Check open trade limit
        with self.trades_lock:
            open_trades_count = len(self.open_trades)
            if open_trades_count >= self.max_open_trades:
                logging.warning(f"Max open trades reached: {open_trades_count}/{self.max_open_trades}")
                return False
        
        # Get current balance - FIXED: don't force refresh too often
        current_balance = self.get_current_balance(force_refresh=False)
        
        # Calculate used capital with thread safety
        with self.trades_lock:
            used_capital = sum(t.get('capital_used', 0.0) for t in self.open_trades)
            
        # Detailed logging for debugging
        logging.debug(f"current_balance=${current_balance:.2f}")
        logging.debug(f"used_capital=${used_capital:.2f} ({open_trades_count} trades)")
        logging.debug(f"new_trade_capital=${new_trade_capital:.2f}")
        
        exposure_pct = (used_capital + new_trade_capital) / current_balance if current_balance > 0 else float('inf')
        logging.debug(f"exposure_pct={exposure_pct*100:.1f}% vs limit {self.max_exposure_pct*100:.1f}%")
        
        # Check exposure limit
        if exposure_pct > self.max_exposure_pct:
            logging.warning(f"Capital exposure limit exceeded: {exposure_pct*100:.1f}% > {self.max_exposure_pct*100:.1f}%")
            return False

        # Check drawdown limit
        with self.capital_lock:
            drawdown_pct = self.realized_loss / self.starting_day_capital if self.starting_day_capital > 0 else float('inf')
            
        if drawdown_pct > self.max_drawdown_pct:
            logging.warning(f"Daily drawdown limit hit: {drawdown_pct*100:.1f}% > {self.max_drawdown_pct*100:.1f}%")
            return False

        return True

    def calculate_dynamic_leverage(self, confidence: float, volatility: float) -> float:
        """Calculate dynamic leverage based on confidence and volatility"""
        
        # Base leverage (conservative approach)
        base_leverage = 3.0
        
        # Adjust for confidence (0.5x to 2x multiplier)
        confidence_multiplier = 0.5 + (confidence * 1.5)
        
        # Adjust for volatility (inverse relationship)
        volatility_multiplier = max(0.3, 1.0 - (volatility * 10))
        
        # Calculate final leverage
        final_leverage = base_leverage * confidence_multiplier * volatility_multiplier
        
        # Cap leverage between 1 and 10
        return max(1.0, min(10.0, final_leverage))

    def get_risk_parameters(self) -> Dict[str, Any]:
        """
        Enhanced get risk parameters combining both old and new approaches
        """
        with self.capital_lock:
            # Calculate today's profit %
            profit_pct = self.daily_profit / self.starting_day_capital * 100 if self.starting_day_capital > 0 else 0
            risk_profile = self.risk_profile
            current_capital = self.current_capital
        
        # Base parameters
        if risk_profile == "aggressive":
            risk_per_trade = 0.03  # 3% risk per trade
            max_leverage = 10     # Up to 10x leverage
            position_sizing = 0.15  # Use up to 15% of capital per trade
        elif risk_profile == "conservative":
            risk_per_trade = 0.01  # 1% risk per trade
            max_leverage = 3      # Up to 3x leverage
            position_sizing = 0.05  # Use up to 5% of capital per trade
        else:  # moderate
            risk_per_trade = 0.02  # 2% risk per trade
            max_leverage = 5      # Up to 5x leverage
            position_sizing = 0.10  # Use up to 10% of capital per trade
            
        return {
            "risk_profile": risk_profile,
            "risk_per_trade": risk_per_trade,
            "max_leverage": max_leverage,
            "position_sizing": position_sizing,
            "daily_profit_pct": profit_pct,
            "current_capital": current_capital,
            "active_positions": len(self.active_positions),
            "total_exposure": sum(pos.get("notional", 0) for pos in self.active_positions.values())
        }

    def _scale_down_position(self, trade_dict: Dict[str, Any], available_capital: float) -> Dict[str, Any]:
        """Scale down a position to fit within available capital"""
        original_capital = trade_dict.get("capital_used", trade_dict.get("notional_value", 0.0))
        
        if original_capital <= 0 or available_capital <= 0:
            return trade_dict
        
        scaling_factor = available_capital / original_capital
        
        # Don't scale below 20% of original size
        if scaling_factor < 0.2:
            logging.warning(f"Cannot scale down trade below 20% of original size")
            return trade_dict
            
        logging.info(f"Scaling down trade {trade_dict.get('trade_id')} to {scaling_factor*100:.1f}% of original size")
        
        # Create a copy to avoid modifying the original
        scaled_trade = trade_dict.copy()
        
        # Scale down the position
        scaled_trade["capital_used"] = available_capital
        scaled_trade["quantity"] = scaled_trade.get("quantity", 0) * scaling_factor
        scaled_trade["notional_value"] = available_capital
        scaled_trade["original_size"] = original_capital  # Keep track of original intended size
        scaled_trade["scaled_down"] = True
        
        return scaled_trade

    def _process_pending_trades_now(self):
        """Process pending trades immediately (called after trade closure)"""
        try:
            with self.pending_lock:
                pending_count = len(self.pending_trades)
                
            if pending_count > 0:
                logging.info(f"Processing {pending_count} pending trades after capital change")
                threading.Thread(target=self._process_trade_queue, name="QueueProcessorImmediate").start()
        except Exception as e:
            logging.error(f"Error processing pending trades: {e}")

    def _process_pending_trades(self):
        """Background thread that periodically processes the trade queue"""
        while self.running:
            try:
                self._process_trade_queue()
                time.sleep(10)  # FIXED: Check every 10 seconds instead of 5
            except Exception as e:
                logging.error(f"Error in trade queue processing: {e}")
                time.sleep(30)  # Wait longer after error

    def _process_trade_queue(self):
        """Process the pending trades queue"""
        # Skip if no pending trades
        with self.pending_lock:
            if not self.pending_trades:
                return
                
        current_time = time.time()
        price_fetcher = self.get_price_fetcher()
        
        # Process trades with thread safety
        with self.pending_lock:
            # Use a copy of the list to avoid modification issues during iteration
            pending_trades = self.pending_trades.copy()
        
        processed_trades = []
        
        for trade in pending_trades:
            # Check if trade has expired
            expiry_time = trade.get("expiration_time", float('inf'))
            if current_time > expiry_time:
                processed_trades.append((trade, "expired"))
                logging.info(f"Trade {trade.get('trade_id')} expired after {int(current_time - trade.get('queue_time', current_time))} seconds")
                continue
                
            # Update entry price if possible
            symbol = trade.get("symbol")
            if symbol:
                try:
                    current_price = price_fetcher.get_latest_price(symbol)
                    if current_price > 0:
                        # Store the original entry
                        if "original_entry" not in trade:
                            trade["original_entry"] = trade.get("entry")
                            
                        # Update the price
                        trade["entry"] = current_price
                        
                        # Update stop loss to maintain risk ratio
                        if "stopLoss" in trade:
                            original_stop = trade.get("stopLoss")
                            original_entry = trade.get("original_entry")
                            side = trade.get("side", "").upper()
                            
                            # Calculate and update stop loss
                            if side == "BUY" or side == "LONG":
                                # For long positions
                                stop_distance_pct = (original_entry - original_stop) / original_entry
                                trade["stopLoss"] = current_price * (1 - stop_distance_pct)
                            else:
                                # For short positions
                                stop_distance_pct = (original_stop - original_entry) / original_entry
                                trade["stopLoss"] = current_price * (1 + stop_distance_pct)
                except Exception as e:
                    logging.warning(f"Failed to update price for {symbol}: {e}")
            
            # Try to allocate the trade now
            capital_used = trade.get("capital_used", trade.get("notional_value", 0.0))
            
            # Check if we can open the trade now
            if self.can_open_new_trade(capital_used):
                processed_trades.append((trade, "executed"))
                self.register_trade(trade)
                logging.info(f"Executed queued trade {trade.get('trade_id')} after {int(current_time - trade.get('queue_time', current_time))} seconds")
            
            # If we failed but still have available capital, try scaling down
            elif self.can_open_new_trade(0):  # We have at least some capital
                current_balance = self.get_current_balance()
                
                with self.trades_lock:
                    used_capital = sum(t.get('capital_used', 0.0) for t in self.open_trades)
                    
                available_capital = (current_balance * self.max_exposure_pct) - used_capital
                
                if available_capital > 0:
                    # Try with scaled down position
                    scaled_trade = self._scale_down_position(trade, available_capital)
                    
                    if scaled_trade.get("scaled_down", False):
                        processed_trades.append((trade, "scaled"))
                        self.register_trade(scaled_trade)
                        logging.info(f"Executed scaled-down queued trade {scaled_trade.get('trade_id')} at {scaled_trade.get('quantity')} units")
        
        # Remove processed trades from the queue
        if processed_trades:
            with self.pending_lock:
                for trade, status in processed_trades:
                    if trade in self.pending_trades:
                        self.pending_trades.remove(trade)

    def validate(self, trade: Dict[str, Any], capital: float = None) -> Dict[str, Any]:
        """Enhanced trade validation that cannot be bypassed"""
        
        if capital:
            with self.capital_lock:
                self.current_capital = capital
            
        entry = trade.get("entry", 0)
        stop_loss = trade.get("stopLoss", 0)
        confidence = trade.get("confidence", 0)
        side = trade.get("side", "BUY")
        symbol = trade.get("symbol", "UNKNOWN")
        strategy_name = trade.get("strategy_name", "unknown")
        trade_type = trade.get("trade_type", "swing")
        
        logging.debug(f"CapitalManager validating trade: strategy={strategy_name}, capital_used=${entry:.2f}")

        if not isinstance(trade, dict):
            logging.warning("Invalid trade format - not a dictionary")
            return {"valid": False, "reason": "invalid_trade_format"}
        
        # Calculate trade risk
        if side in ["BUY", "LONG"]:
            price_risk = entry - stop_loss
        else:
            price_risk = stop_loss - entry
            
        if price_risk <= 0:
            return {"valid": False, "reason": "invalid_risk_calculation", "trade": trade}
            
        risk_amount = self.current_capital * self.max_risk_per_trade
        
        # Calculate position size based on risk
        position_size = risk_amount / price_risk
        
        # Calculate notional value
        notional_value = position_size * entry
        
        # Check if notional exceeds capital limits (max 10% per position)
        max_position_size = self.current_capital * 0.1
        if notional_value > max_position_size:
            # Reduce position size to fit limits
            position_size = max_position_size / entry
            notional_value = max_position_size
            logging.warning(f"Reduced position size for {symbol} to fit capital limits")
        
        # Check total exposure
        total_exposure = sum(pos.get("notional", 0) for pos in self.active_positions.values())
        max_total_exposure_amount = self.current_capital * self.max_total_exposure
        
        if (total_exposure + notional_value) > max_total_exposure_amount:
            return {
                "valid": False, 
                "reason": f"max_exposure_exceeded_{total_exposure + notional_value:.2f}_{max_total_exposure_amount:.2f}",
                "trade": trade
            }
        
        # Check maximum number of positions (max 10 concurrent)
        if len(self.active_positions) >= 10:
            return {"valid": False, "reason": "max_positions_reached", "trade": trade}
        
        # Confidence-based position sizing
        confidence_multiplier = max(0.3, min(1.5, confidence))  # 30% to 150% of calculated size
        final_position_size = position_size * confidence_multiplier
        final_notional = final_position_size * entry
        
        # Calculate dynamic leverage based on confidence and volatility
        base_leverage = self.calculate_dynamic_leverage(confidence, trade.get("volatility", 0.02))
        
        # Set expiration based on strategy type
        if trade_type == "nano":
            expiration = 30  # 30 seconds for HFT
        elif trade_type == "sniper":
            expiration = 300  # 5 minutes for event-based
        else:
            expiration = 1800  # 30 minutes for swing
            
        trade.update({
            "queue_time": time.time(),
            "expiration_time": time.time() + expiration
        })

        # Try normal allocation first
        capital_used = final_notional
        if self.can_open_new_trade(capital_used):
            # Update trade with proper sizing
            trade.update({
                "quantity": final_position_size,
                "notional_value": final_notional,
                "risk_amount": price_risk * final_position_size,
                "capital_allocated": final_notional,
                "capital_used": final_notional,
                "leverage": base_leverage,
                "confidence_multiplier": confidence_multiplier
            })
            
            # Track this position
            trade_id = trade.get("trade_id", f"trade_{int(time.time())}_{symbol}")
            self.active_positions[trade_id] = {
                "symbol": symbol,
                "notional": final_notional,
                "risk": price_risk * final_position_size,
                "timestamp": time.time()
            }

            # Add risk profile data to the trade
            risk_params = self.get_risk_parameters()
            trade.update({
                "risk_profile": risk_params["risk_profile"],
                "max_leverage": risk_params["max_leverage"]
            })

            # Register the trade if validated
            self.register_trade(trade)

            logging.info(f"Capital validation passed for {symbol}: size={final_position_size:.4f}, notional=${final_notional:.2f}")
            return {"valid": True, "trade": trade}
            
        # Try scaling down the position
        current_balance = self.get_current_balance(force_refresh=False)  # FIXED: Don't force refresh
        
        with self.trades_lock:
            used_capital = sum(t.get('capital_used', 0.0) for t in self.open_trades)
            
        available_capital = (current_balance * self.max_exposure_pct) - used_capital
        
        if available_capital > 0:
            # Try with a scaled down position
            scaled_trade = self._scale_down_position(trade, available_capital)
            
            if scaled_trade.get("scaled_down", False):
                # Add risk profile data to the trade
                risk_params = self.get_risk_parameters()
                scaled_trade.update({
                    "risk_profile": risk_params["risk_profile"],
                    "max_leverage": risk_params["max_leverage"]
                })
                
                # Register the scaled trade
                self.register_trade(scaled_trade)
                
                logging.info(f"CapitalManager registered scaled-down trade for {strategy_name}")
                return {"valid": True, "trade": scaled_trade, "scaled": True}
                
        # If we can't allocate now, add to pending queue
        # Create a copy to avoid reference issues
        queue_trade = trade.copy()
        
        with self.pending_lock:
            self.pending_trades.append(queue_trade)
        
        logging.info(f"CapitalManager queued trade for {strategy_name}: over limits")
        return {"valid": False, "reason": "capital_limits_exceeded", "queued": True}