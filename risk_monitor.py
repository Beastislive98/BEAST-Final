import numpy as np
import pandas as pd
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import threading
import logging
import json
import os
import atexit
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/risk_monitor.log"),
        logging.StreamHandler()
    ]
)

class RiskMonitor:
    """
    Monitors and tracks risk metrics for active trades.
    Designed to integrate with strategy_tournament output format.
    """
    def __init__(self, max_trades=25, max_exposure_pct=80.0, max_drawdown_pct=10.0):
        # Risk parameters
        self.max_trades = max_trades
        self.max_exposure_pct = max_exposure_pct / 100
        self.max_drawdown_pct = max_drawdown_pct / 100
        
        # Thread safety locks
        self.trade_lock = threading.Lock()
        self.capital_lock = threading.Lock()
        self.price_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Trade tracking
        self.active_trades = []
        self.pending_trades = []
        self.completed_trades = []  # Store recently completed trades for analysis
        self.max_completed_history = 100  # Maximum number of completed trades to keep
        
        # Strategy performance tracking
        self.strategy_performance = {}
        
        # Financial tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.starting_capital = 0.0
        self.current_capital = 0.0
        self.drawdown = 0.0
        self.peak_capital = 0.0
        
        # Price tracking
        self.price_cache = {}  # Symbol -> (timestamp, price)
        self.price_cache_ttl = 5  # seconds
        self.rate_limiter = RateLimiter(max_calls=20, time_frame=60)  # 20 calls per minute
        self.last_prices_update = 0
        self.update_interval = 1.0  # seconds
        
        # Status tracking
        self.last_alert_time = {}  # Track when alerts were last sent
        self.alert_cooldown = 300  # 5 minutes between repeat alerts
        self.last_state_save = 0
        self.save_interval = 60  # Save state every 60 seconds
        
        # Metrics history
        self.metrics_history = deque(maxlen=1440)  # Store last 24 hours (at 1-min intervals)
        self.last_metrics_save = 0
        
        # Callback to strategy_selection for outcome tracking
        self.strategy_feedback_callback = None
        
        # Background threads
        self.running = True
        
        # Ensure directories exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Check dependencies
        self.binance_interface_available = self._check_binance_interface()
        
        # Load previous state
        self._load_state()
        
        # Start background threads
        self._start_background_threads()
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        logging.info(f"RiskMonitor initialized with max_trades={max_trades}, "
                     f"max_exposure_pct={max_exposure_pct}%, max_drawdown_pct={max_drawdown_pct}%")
    
    def register_strategy_feedback(self, callback_function):
        """Register a callback to send trade outcomes back to the strategy selector"""
        self.strategy_feedback_callback = callback_function
        logging.info("Strategy feedback callback registered")
    
    def _check_binance_interface(self) -> bool:
        """Check if Binance interface is available"""
        try:
            from binance_interface import get_futures_market_data
            logging.info("Binance interface module available")
            return True
        except ImportError:
            logging.warning("Binance interface module not available, using fallback price fetching")
            return False
    
    def _start_background_threads(self):
        """Start background monitoring threads"""
        # Thread for updating prices and unrealized PnL
        self.price_update_thread = threading.Thread(
            target=self._price_update_loop, 
            name="PriceUpdateThread",
            daemon=True
        )
        self.price_update_thread.start()
        
        # Thread for metrics calculation and alerts
        self.metrics_thread = threading.Thread(
            target=self._metrics_loop, 
            name="MetricsThread",
            daemon=True
        )
        self.metrics_thread.start()
        
        logging.info("Started background monitoring threads")
    
    def _price_update_loop(self):
        """Background thread that updates prices and unrealized PnL"""
        while self.running:
            try:
                self._update_all_prices()
                time.sleep(1)  # Check every second
            except Exception as e:
                logging.error(f"Error in price update loop: {e}")
                time.sleep(5)  # Wait longer after error
    
    def _metrics_loop(self):
        """Background thread that calculates metrics and generates alerts"""
        while self.running:
            try:
                # Get and save metrics
                metrics = self.get_risk_metrics()
                
                # Save metrics history periodically
                current_time = time.time()
                if current_time - self.last_metrics_save >= 60:  # Every minute
                    with self.metrics_lock:
                        self.metrics_history.append((current_time, metrics))
                    self.last_metrics_save = current_time
                
                # Check for alerts
                self._check_alert_conditions(metrics)
                
                # Save state periodically
                if current_time - self.last_state_save >= self.save_interval:
                    self._save_state()
                    self.last_state_save = current_time
                
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error in metrics loop: {e}")
                time.sleep(5)  # Wait longer after error
    
    def _check_alert_conditions(self, metrics: Dict[str, Any]):
        """Check if any alert conditions are met"""
        current_time = time.time()
        risk_status = metrics["risk_status"]
        
        # Alert based on risk status
        if risk_status in ["critical", "high"]:
            alert_key = f"risk_status_{risk_status}"
            if self._should_send_alert(alert_key, current_time):
                logging.warning(f"ALERT: Risk status is {risk_status.upper()}! "
                                f"Exposure: {metrics['exposure_pct']:.1f}%, "
                                f"Drawdown: {metrics['drawdown']:.1f}%")
                self.last_alert_time[alert_key] = current_time
        
        # Alert on high drawdown
        if metrics["drawdown"] >= self.max_drawdown_pct * 80:  # 80% of max drawdown
            alert_key = "high_drawdown"
            if self._should_send_alert(alert_key, current_time):
                logging.warning(f"ALERT: High drawdown detected: {metrics['drawdown']:.1f}% "
                                f"(max: {self.max_drawdown_pct*100:.1f}%)")
                self.last_alert_time[alert_key] = current_time
        
        # Alert on high exposure
        if metrics["exposure_pct"] >= self.max_exposure_pct * 90:  # 90% of max exposure
            alert_key = "high_exposure"
            if self._should_send_alert(alert_key, current_time):
                logging.warning(f"ALERT: High capital exposure: {metrics['exposure_pct']:.1f}% "
                                f"(max: {self.max_exposure_pct*100:.1f}%)")
                self.last_alert_time[alert_key] = current_time
                
        # Alert on strategy performance
        for strategy, stats in self.strategy_performance.items():
            if stats.get("trades", 0) >= 5:
                win_rate = stats.get("wins", 0) / stats.get("trades", 1)
                
                # Alert on poor performing strategies
                if win_rate < 0.3 and stats.get("pnl", 0) < 0:
                    alert_key = f"poor_strategy_{strategy}"
                    if self._should_send_alert(alert_key, current_time):
                        logging.warning(f"ALERT: Strategy {strategy} performing poorly: "
                                      f"Win rate {win_rate:.1%}, PnL ${stats.get('pnl', 0):.2f}")
                        self.last_alert_time[alert_key] = current_time
    
    def _should_send_alert(self, alert_key: str, current_time: float) -> bool:
        """Check if we should send an alert based on cooldown period"""
        last_time = self.last_alert_time.get(alert_key, 0)
        return (current_time - last_time) >= self.alert_cooldown
        
    def shutdown(self):
        """Safely shut down the monitor"""
        logging.info("Shutting down RiskMonitor...")
        self.running = False
        
        # Save final state
        self._save_state()
        
        # Save metrics history
        self._save_metrics_history()
        
        logging.info("RiskMonitor shutdown complete")
    
    def _load_state(self):
        """Load previous state"""
        try:
            if os.path.exists("data/risk_monitor_state.json"):
                with open("data/risk_monitor_state.json", "r") as f:
                    state = json.load(f)
                    
                    with self.capital_lock:
                        self.realized_pnl = state.get("realized_pnl", 0.0)
                        self.current_capital = state.get("current_capital", 0.0)
                        self.peak_capital = state.get("peak_capital", 0.0)
                        self.starting_capital = state.get("starting_capital", self.current_capital)
                    
                    # Load strategy performance
                    if "strategy_performance" in state:
                        self.strategy_performance = state["strategy_performance"]
                    
                    # Don't load active trades directly since they may have changed
                    # but keep them for reference if needed
                    prev_active_trades = state.get("active_trades", [])
                    
                    logging.info(f"Loaded previous state: capital=${self.current_capital:.2f}, "
                                 f"peak=${self.peak_capital:.2f}, "
                                 f"previous active trades: {len(prev_active_trades)}")
        except Exception as e:
            logging.warning(f"Could not load previous state: {e}")
    
    def _save_state(self):
        """Save current state"""
        try:
            with self.capital_lock:
                capital_data = {
                    "realized_pnl": self.realized_pnl,
                    "unrealized_pnl": self.unrealized_pnl,
                    "current_capital": self.current_capital,
                    "peak_capital": self.peak_capital,
                    "starting_capital": self.starting_capital,
                    "drawdown": self.drawdown
                }
            
            with self.trade_lock:
                trade_data = {
                    "active_trades": self.active_trades,
                    "pending_trades": self.pending_trades,
                    "active_count": len(self.active_trades),
                    "pending_count": len(self.pending_trades)
                }
            
            strategy_data = {"strategy_performance": self.strategy_performance}
            
            state = {
                **capital_data,
                **trade_data,
                **strategy_data,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat()
            }
            
            with open("data/risk_monitor_state.json", "w") as f:
                json.dump(state, f, indent=2)
                
            logging.debug(f"Saved state: capital=${self.current_capital:.2f}, "
                          f"active trades: {len(self.active_trades)}")
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
    
    def _save_metrics_history(self):
        """Save metrics history to file"""
        try:
            with self.metrics_lock:
                history = list(self.metrics_history)
                
            if history:
                with open("data/risk_metrics_history.json", "w") as f:
                    json.dump(history, f, indent=2)
                    
                logging.info(f"Saved metrics history: {len(history)} entries")
        except Exception as e:
            logging.error(f"Failed to save metrics history: {e}")
        
    def set_capital(self, capital: float):
        """Set the current capital and update peak if needed"""
        with self.capital_lock:
            if self.starting_capital == 0:
                self.starting_capital = capital
                
            self.current_capital = capital
            
            if capital > self.peak_capital:
                self.peak_capital = capital
                
            # Update drawdown
            if self.peak_capital > 0:
                self.drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        
        # Process pending trades after capital change
        self._process_pending_trades()
        
    def add_trade(self, trade: Dict[str, Any]) -> bool:
        """Add a new trade to monitor from the strategy tournament output"""
        # Validate minimum required fields
        if not self._validate_trade_structure(trade):
            logging.warning(f"Invalid trade structure: {trade.get('trade_id', 'unknown')}")
            return False
            
        # Set trade start time if not present
        if "start_time" not in trade:
            trade["start_time"] = time.time()
            
        # Normalize position side if needed
        side = trade.get("side", "")
        if "positionSide" not in trade and side:
            if "BUY" in side.upper():
                trade["positionSide"] = "LONG"
            elif "SELL" in side.upper():
                trade["positionSide"] = "SHORT"
                
        # Get current price if not already in trade
        if "current_price" not in trade and "entry" in trade:
            trade["current_price"] = trade["entry"]
        
        with self.trade_lock:
            self.active_trades.append(trade)
            trades_count = len(self.active_trades)
            
        logging.info(f"Trade added: {trade.get('symbol')} {trade.get('strategy_name')} "
                     f"confidence={trade.get('confidence', 0):.2f}. "
                     f"Total active trades: {trades_count}")
        return True
    
    def _validate_trade_structure(self, trade: Dict[str, Any]) -> bool:
        """Validate if trade has minimum required fields expected from strategy tournament"""
        required_fields = ["symbol", "entry", "side", "stopLoss", "takeProfit", "strategy_name"]
        for field in required_fields:
            if field not in trade:
                logging.warning(f"Trade missing required field: {field}")
                return False
                
        # Generate trade_id if missing
        if "trade_id" not in trade:
            trade["trade_id"] = f"trade_{int(time.time())}_{trade.get('symbol', 'UNKNOWN')}"
            
        # Ensure numeric values
        numeric_fields = ["entry", "stopLoss", "takeProfit", "confidence"]
        for field in numeric_fields:
            if field in trade:
                try:
                    trade[field] = float(trade[field])
                except (ValueError, TypeError):
                    logging.warning(f"Trade has invalid {field} value: {trade.get(field)}")
                    return False
        
        return True
    
    def close_trade(self, trade_id: str, pnl: float, exit_price: float = None) -> bool:
        """
        Close a trade and update realized PnL.
        Works with trades from strategy tournament.
        """
        found_trade = None
        
        with self.trade_lock:
            for i, trade in enumerate(self.active_trades):
                if trade.get("trade_id") == trade_id:
                    found_trade = self.active_trades.pop(i)
                    break
        
        if not found_trade:
            logging.warning(f"Trade {trade_id} not found in active trades")
            return False
            
        # Update completed trades history with trade result
        if found_trade:
            # Calculate success/failure
            success = pnl > 0
            
            # Set exit details
            found_trade["exit_time"] = time.time()
            found_trade["pnl"] = pnl
            found_trade["exit_price"] = exit_price if exit_price else found_trade.get("current_price")
            found_trade["success"] = success
            
            # Calculate trade duration
            start_time = found_trade.get("start_time", found_trade.get("timestamp", 0))
            if start_time:
                duration_seconds = time.time() - start_time
                found_trade["duration_seconds"] = duration_seconds
            
            with self.trade_lock:
                self.completed_trades.append(found_trade)
                # Maintain max history size
                if len(self.completed_trades) > self.max_completed_history:
                    self.completed_trades = self.completed_trades[-self.max_completed_history:]
        
        # Update capital based on PnL
        with self.capital_lock:
            self.realized_pnl += pnl
            
        # Update strategy performance stats
        strategy_name = found_trade.get("strategy_name", "unknown")
        confidence = found_trade.get("confidence", 0.5)
        self._update_strategy_performance(strategy_name, success, pnl, confidence)
        
        # Provide feedback to strategy selector if callback is registered
        if self.strategy_feedback_callback:
            try:
                symbol = found_trade.get("symbol", "UNKNOWN")
                strategy_name = found_trade.get("strategy_name", "unknown")
                self.strategy_feedback_callback(symbol, success, pnl, confidence, strategy_name)
                logging.debug(f"Sent trade outcome feedback for {symbol} to strategy selector")
            except Exception as e:
                logging.error(f"Failed to send strategy feedback: {e}")
        
        logging.info(f"Trade {trade_id} closed with PnL: ${pnl:.2f}, "
                     f"Strategy: {strategy_name}, Success: {success}")
        
        # Process any pending trades that might now fit
        self._process_pending_trades()
        
        return True
    
    def _update_strategy_performance(self, strategy_name: str, success: bool, pnl: float, confidence: float):
        """Update performance statistics for strategies"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "pnl": 0.0,
                "avg_confidence": 0.0
            }
            
        stats = self.strategy_performance[strategy_name]
        stats["trades"] += 1
        if success:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["pnl"] += pnl
        
        # Update average confidence
        total_conf = stats["avg_confidence"] * (stats["trades"] - 1) + confidence
        stats["avg_confidence"] = total_conf / stats["trades"]
    
    def _process_pending_trades(self):
        """Process trades that were previously pending"""
        processed_trades = []
        
        with self.trade_lock:
            pending_copy = self.pending_trades.copy()
        
        for trade in pending_copy:
            result = self.can_accept_trade(trade)
            if result["accepted"]:
                self.add_trade(trade)
                processed_trades.append(trade)
                logging.info(f"Accepted previously pending trade: {trade.get('symbol')} {trade.get('strategy_name')}")
        
        # Remove processed trades from pending list
        if processed_trades:
            with self.trade_lock:
                for trade in processed_trades:
                    if trade in self.pending_trades:
                        self.pending_trades.remove(trade)
    
    def _update_all_prices(self):
        """Update prices for all symbols in active trades"""
        # Get unique symbols from active trades
        with self.trade_lock:
            symbols = set(trade.get("symbol") for trade in self.active_trades)
        
        # Update price for each symbol
        for symbol in symbols:
            if symbol:
                self.get_current_price(symbol, force_refresh=True)
        
        # Update unrealized PnL after price updates
        self.update_unrealized_pnl()
    
    def get_current_price(self, symbol: str, force_refresh: bool = False) -> float:
        """Get current price for a symbol with caching"""
        current_time = time.time()
        
        # Check cache first if not forcing refresh
        if not force_refresh:
            with self.price_lock:
                if symbol in self.price_cache:
                    cache_time, price = self.price_cache[symbol]
                    if current_time - cache_time < self.price_cache_ttl:
                        return price
        
        # Get fresh price
        price = 0.0
        
        # Try Binance interface first if available
        if self.binance_interface_available:
            try:
                from binance_interface import get_futures_market_data
                data = get_futures_market_data(symbol)
                if data and "ticker" in data:
                    ticker = data["ticker"]
                    price = float(ticker.get("lastPrice", 0))
            except Exception as e:
                logging.warning(f"Binance interface error for {symbol}: {e}")
        
        # Fallback to direct API if needed
        if price <= 0:
            try:
                # Use rate limiter to avoid API throttling
                self.rate_limiter.wait_if_needed()
                
                # Futures API for futures symbols
                url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    price = float(data.get("price", 0))
                else:
                    # Try spot API as fallback
                    self.rate_limiter.wait_if_needed()
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        price = float(data.get("price", 0))
            except Exception as e:
                logging.warning(f"API price fetch failed for {symbol}: {e}")
        
        # Update cache if we got a valid price
        if price > 0:
            with self.price_lock:
                self.price_cache[symbol] = (current_time, price)
        else:
            # Fallback to last known price
            with self.price_lock:
                if symbol in self.price_cache:
                    price = self.price_cache[symbol][1]
            
        return price
    
    def update_unrealized_pnl(self):
        """Update unrealized PnL for all active trades"""
        unrealized = 0.0
        
        # Make a thread-safe copy of active trades
        with self.trade_lock:
            trades_copy = self.active_trades.copy()
        
        for trade in trades_copy:
            symbol = trade.get("symbol")
            entry = trade.get("entry", 0.0)
            position_side = trade.get("positionSide", trade.get("side", "LONG"))
            quantity = trade.get("quantity", 1.0)  # Default to 1 if not specified
            
            # Normalize position side
            if isinstance(position_side, str):
                position_side = position_side.upper()
                if "BUY" in position_side or "LONG" in position_side:
                    position_side = "LONG"
                else:
                    position_side = "SHORT"
            
            current_price = self.get_current_price(symbol)
            
            if current_price > 0 and entry > 0:
                if position_side == "LONG":
                    trade_pnl = (current_price - entry) * quantity
                else:  # SHORT
                    trade_pnl = (entry - current_price) * quantity
                    
                # Update trade's unrealized PnL
                trade["unrealized_pnl"] = trade_pnl
                trade["current_price"] = current_price
                trade["last_update"] = time.time()
                    
                unrealized += trade_pnl
                
                # Check for stop loss/take profit hits
                self._check_exit_conditions(trade, current_price)
        
        with self.capital_lock:
            self.unrealized_pnl = unrealized
            
            # Update drawdown with unrealized P&L included
            equity = self.current_capital + unrealized
            if equity > self.peak_capital:
                self.peak_capital = equity
            
            if self.peak_capital > 0:
                self.drawdown = max(0, (self.peak_capital - equity) / self.peak_capital)
    
    def _check_exit_conditions(self, trade: Dict[str, Any], current_price: float):
        """Check if a trade has hit its stop loss or take profit"""
        # Skip if already processing exits
        if trade.get("exiting", False) or not current_price:
            return
            
        trade_id = trade.get("trade_id")
        symbol = trade.get("symbol")
        stop_loss = trade.get("stopLoss", 0.0)
        take_profit = trade.get("takeProfit", 0.0)
        position_side = trade.get("positionSide", "LONG").upper()
        entry = trade.get("entry", 0.0)
        quantity = trade.get("quantity", 1.0)
        
        # Skip if no stop loss or take profit
        if not stop_loss or not take_profit:
            return
            
        # Check for stop loss hit
        sl_hit = False
        tp_hit = False
        
        if "LONG" in position_side:
            sl_hit = current_price <= stop_loss
            tp_hit = current_price >= take_profit
        else:  # SHORT
            sl_hit = current_price >= stop_loss
            tp_hit = current_price <= take_profit
            
        # If either condition hit, exit the trade
        if sl_hit or tp_hit:
            # Mark as exiting to prevent duplicate exits
            trade["exiting"] = True
            
            # Calculate PnL
            if sl_hit:
                exit_price = stop_loss
                reason = "stop_loss"
            else:
                exit_price = take_profit
                reason = "take_profit"
                
            if "LONG" in position_side:
                pnl = (exit_price - entry) * quantity
            else:
                pnl = (entry - exit_price) * quantity
                
            # Log the event
            hit_type = "STOP LOSS" if sl_hit else "TAKE PROFIT"
            logging.info(f"{hit_type} HIT for {symbol}: {trade_id} at price {current_price}")
            
            # Close the trade
            trade["exit_reason"] = reason
            trade["auto_exit"] = True
            
            # Use a separate thread to avoid blocking the price update loop
            threading.Thread(
                target=self.close_trade,
                args=(trade_id, pnl, exit_price),
                name=f"ExitHandler-{trade_id}"
            ).start()
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        # Calculate with thread safety
        with self.capital_lock:
            current_capital = self.current_capital
            unrealized_pnl = self.unrealized_pnl
            realized_pnl = self.realized_pnl
            peak_capital = self.peak_capital
            drawdown = self.drawdown
        
        with self.trade_lock:
            active_trades = len(self.active_trades)
            pending_trades = len(self.pending_trades)
            # Calculate total exposure
            total_exposure = sum(trade.get("notional_value", trade.get("entry", 0.0) * trade.get("quantity", 1.0)) 
                                for trade in self.active_trades)
        
        # Calculate exposure percentage
        exposure_pct = total_exposure / current_capital if current_capital > 0 else 0.0
        
        # Calculate current equity with unrealized PnL
        equity = current_capital + unrealized_pnl
        
        # Get top strategies by PnL
        top_strategies = []
        for name, stats in self.strategy_performance.items():
            if stats.get("trades", 0) >= 3:  # Minimum 3 trades
                win_rate = stats.get("wins", 0) / stats.get("trades", 1)
                top_strategies.append({
                    "name": name,
                    "trades": stats.get("trades", 0),
                    "win_rate": win_rate,
                    "pnl": stats.get("pnl", 0.0)
                })
                
        # Sort by PnL
        top_strategies.sort(key=lambda x: x["pnl"], reverse=True)
        top_strategies = top_strategies[:5]  # Top 5
        
        return {
            "active_trades": active_trades,
            "pending_trades": pending_trades,
            "exposure": total_exposure,
            "exposure_pct": exposure_pct * 100,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "current_capital": current_capital,
            "equity": equity,
            "drawdown": drawdown * 100,
            "max_trades": self.max_trades,
            "max_exposure_pct": self.max_exposure_pct * 100,
            "max_drawdown_pct": self.max_drawdown_pct * 100,
            "trades_available": max(0, self.max_trades - active_trades),
            "exposure_available_pct": max(0, (self.max_exposure_pct - exposure_pct) * 100),
            "exposure_available": max(0, (self.max_exposure_pct * current_capital) - total_exposure),
            "risk_status": self._get_risk_status(exposure_pct, drawdown),
            "top_strategies": top_strategies,
            "timestamp": time.time()
        }
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get historical metrics"""
        with self.metrics_lock:
            history = list(self.metrics_history)
        return history
    
    def _get_risk_status(self, exposure_pct: float, drawdown: float) -> str:
        """Determine current risk status"""
        if drawdown >= self.max_drawdown_pct:
            return "critical"  # Reached max drawdown
        elif exposure_pct >= self.max_exposure_pct:
            return "high"      # Reached max exposure
        elif exposure_pct >= self.max_exposure_pct * 0.8:
            return "elevated"  # Near max exposure
        elif drawdown >= self.max_drawdown_pct * 0.8:
            return "caution"   # Near max drawdown
        else:
            return "normal"    # Normal operations
    
    def can_accept_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if a trade from strategy tournament can be accepted
        under current risk constraints
        """
        # Validate trade structure first
        if not self._validate_trade_structure(trade):
            return {
                "accepted": False,
                "reason": "invalid_trade_structure",
                "symbol": trade.get("symbol", "UNKNOWN")
            }
            
        # Get current risk metrics
        metrics = self.get_risk_metrics()
        
        # Get trade value - use notional_value if available, or calculate from entry and quantity
        notional_value = trade.get("notional_value", 0.0)
        if notional_value <= 0:
            entry = trade.get("entry", 0.0)
            quantity = trade.get("quantity", 1.0)
            notional_value = entry * quantity
            
        symbol = trade.get("symbol", "UNKNOWN")
        strategy_name = trade.get("strategy_name", "unknown")
        
        # Check if we're at max trades
        if metrics["active_trades"] >= self.max_trades:
            return {
                "accepted": False, 
                "reason": "max_trades_reached",
                "max_trades": self.max_trades,
                "symbol": symbol
            }
            
        # Check if this would exceed max exposure
        new_exposure_pct = (metrics["exposure"] + notional_value) / metrics["current_capital"] if metrics["current_capital"] > 0 else float('inf')
        if new_exposure_pct > self.max_exposure_pct:
            return {
                "accepted": False, 
                "reason": "max_exposure_reached",
                "current": new_exposure_pct * 100,
                "max": self.max_exposure_pct * 100,
                "symbol": symbol
            }
            
        # Check drawdown limit
        if metrics["drawdown"] >= self.max_drawdown_pct * 100:
            return {
                "accepted": False, 
                "reason": "max_drawdown_reached",
                "current": metrics["drawdown"],
                "max": self.max_drawdown_pct * 100,
                "symbol": symbol
            }
            
        # Check strategy performance - reject if strategy has poor performance
        if strategy_name in self.strategy_performance:
            stats = self.strategy_performance[strategy_name]
            if stats.get("trades", 0) >= 5 and stats.get("wins", 0) / stats.get("trades", 1) < 0.3:
                return {
                    "accepted": False,
                    "reason": "poor_strategy_performance",
                    "win_rate": stats.get("wins", 0) / stats.get("trades", 1),
                    "symbol": symbol
                }
            
        # Trade accepted
        return {
            "accepted": True,
            "symbol": symbol,
            "strategy_name": strategy_name,
            "metrics": metrics
        }
        
    def queue_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a trade that cannot be accepted now due to risk limits"""
        # Validate first
        if not self._validate_trade_structure(trade):
            return {
                "queued": False,
                "reason": "invalid_trade_structure",
                "symbol": trade.get("symbol", "UNKNOWN")
            }
            
        # Set queue time
        trade["queue_time"] = time.time()
        
        with self.trade_lock:
            self.pending_trades.append(trade)
            pending_count = len(self.pending_trades)
            
        symbol = trade.get("symbol", "UNKNOWN")
        strategy_name = trade.get("strategy_name", "unknown")
        logging.info(f"Trade queued: {symbol} {strategy_name}. Total pending: {pending_count}")
        
        return {
            "queued": True,
            "symbol": symbol,
            "strategy_name": strategy_name,
            "queue_position": pending_count
        }


class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, max_calls: int, time_frame: int):
        self.max_calls = max_calls
        self.time_frame = time_frame  # in seconds
        self.calls = []
        self.lock = threading.Lock()
        
    def can_call(self) -> bool:
        """Check if we can make an API call without hitting limits"""
        with self.lock:
            current_time = time.time()
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
            time.sleep(0.1)  # Small sleep to avoid tight loop