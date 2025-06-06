# pipeline.py

import logging
import time
import signal
import sys
import os
import random
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid

# Import and initialize RiskMonitor
from risk_monitor import RiskMonitor
risk_monitor = RiskMonitor()

# Create trade_signals directory if it doesn't exist
SIGNALS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_signals")
os.makedirs(SIGNALS_DIR, exist_ok=True)

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Global trade queue for managing multiple trades
trade_queue = queue.PriorityQueue()
active_trades = []
max_concurrent_trades = 25  # Allow up to 25 concurrent trades
trade_lock = threading.Lock()
shadow_queue = queue.PriorityQueue()  # Separate queue for shadow trades on cold symbols

# Import modules with proper error handling
try:
    from data_bundle import assemble_data_bundle
except ImportError:
    logging.critical("Critical module data_bundle not found!")
    raise

try:
    from strategy_selection import strategy_tournament, run_shadow_analysis_for_cold_symbols
except ImportError:
    logging.critical("Critical module strategy_selection not found!")
    raise

try:
    from capital_allocator import CapitalAllocator
    capital_allocator = CapitalAllocator()
except ImportError:
    logging.warning("capital_allocator module not found, using fallback implementation")
    # Inline fallback implementation
    class FallbackCapitalAllocator:
        def __init__(self):
            self.current_capital = 32.0  # Initial capital
            
        def allocate(self, signal):
            # Simple fallback that adds a fixed quantity
            price = signal.get("entry", 0.0)
            symbol = signal.get("symbol", "UNKNOWN")
            
            # Determine appropriate position size based on price
            if price > 10000:  # BTC-like
                quantity = 0.001
            elif price > 1000:  # ETH-like
                quantity = 0.01
            elif price > 100:
                quantity = 0.1
            elif price > 10:
                quantity = 1.0
            elif price > 1:
                quantity = 10.0
            else:
                quantity = 100.0
                
            # Add quantity to the signal
            result = signal.copy()
            result["quantity"] = quantity
            result["capital_allocated"] = price * quantity
            return result
            
    capital_allocator = FallbackCapitalAllocator()
    logging.info("Using fallback capital allocator")

try:
    from capital_manager import CapitalManager
    capital_manager = CapitalManager()
    # Connect capital manager to capital allocator for dynamic capital tracking
    capital_allocator.capital_manager = capital_manager
except ImportError:
    logging.warning("capital_manager module not found, using fallback implementation")
    # Inline fallback implementation
    class FallbackCapitalManager:
        def validate(self, trade, capital=32.0):
            # Always approve trades in fallback mode
            return {"valid": True, "trade": trade}
            
        def get_current_balance(self):
            return 32.0
            
        def get_risk_parameters(self):
            return {
                "risk_profile": "moderate",
                "risk_per_trade": 0.02,
                "max_leverage": 5,
                "position_sizing": 0.1,
                "current_capital": 32.0
            }
            
    capital_manager = FallbackCapitalManager()
    logging.info("Using fallback capital manager")
    # Connect fallback capital manager to capital allocator
    capital_allocator.capital_manager = capital_manager

try:
    from risk_validator import RiskValidator
    risk_validator = RiskValidator()
except ImportError:
    logging.warning("risk_validator module not found, using fallback implementation")
    # Inline fallback implementation
    class FallbackRiskValidator:
        def validate(self, trade):
            # Always approve trades in fallback mode
            return {"valid": True, "trade": trade}
    risk_validator = FallbackRiskValidator()
    logging.info("Using fallback risk validator")

# Attempt to import specialized modules
try:
    from whale_detector import detect_whale_activity
    whale_detector_available = True
except ImportError:
    logging.warning("whale_detector module not found, this feature will be disabled")
    whale_detector_available = False
    detect_whale_activity = lambda data: {"whale_present": False, "large_bid_wall": False, "large_ask_wall": False}

try:
    from sentiment_analyzer import extract_sentiment_score
    sentiment_analyzer_available = True
except ImportError:
    logging.warning("sentiment_analyzer module not found, this feature will be disabled")
    sentiment_analyzer_available = False
    extract_sentiment_score = lambda symbol: {"sentiment_score": 0.0, "headline_count": 0}

try:
    from json_writer import write_trade_signal_to_file
    json_writer_available = True
except ImportError:
    logging.warning("json_writer module not found, using fallback implementation")
    json_writer_available = False
    
    def write_trade_signal_to_file(trade_data: Dict[str, Any]) -> Optional[str]:
        """
        Fallback function for writing trade signals.
        """
        try:
            if not trade_data or not isinstance(trade_data, dict):
                logging.error("Invalid trade data provided")
                return None
                    
            # Ensure symbol exists
            symbol = trade_data.get("symbol")
            if not symbol:
                logging.error("Trade data missing symbol")
                return None
                    
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
            # Ensure trade data has timestamp
            if "timestamp" not in trade_data:
                trade_data["timestamp"] = timestamp
                    
            # Create filename with symbol and timestamp
            filename = f"{symbol}_{timestamp}.json"
            filepath = os.path.join(SIGNALS_DIR, filename)
                
            # Write JSON file
            with open(filepath, "w") as f:
                json.dump(trade_data, f, indent=2)
                    
            logging.info(f"📝 Trade signal written to {filepath}")
            return filepath
                
        except Exception as e:
            logging.exception(f"Failed to write trade signal: {e}")
            return None

def queue_processor_thread():
    """
    Background thread that processes the trade queue
    """
    logging.info("Trade queue processor thread started")
    
    while True:
        try:
            # Get current number of active trades
            with trade_lock:
                current_active = len(active_trades)
                
            # If we have room for more trades, process from queue
            if current_active < max_concurrent_trades and not trade_queue.empty():
                # Get the highest priority trade
                priority, trade_data = trade_queue.get(block=False)
                
                # Execute final validation
                try:
                    final_check = risk_validator.validate(trade_data)
                    if final_check.get("valid", False):
                        # Write the trade signal
                        if json_writer_available:
                            write_trade_signal_to_file(trade_data)
                        else:
                            write_trade_signal_to_file(trade_data)
                            
                        # Add to active trades
                        with trade_lock:
                            active_trades.append(trade_data)
                            
                        logging.info(f"Executed trade for {trade_data.get('symbol')} with priority {-priority}")
                    else:
                        logging.warning(f"Trade validation failed: {final_check.get('reason')}")
                except Exception as e:
                    logging.error(f"Trade execution failed: {e}")
                    
                # Mark task as done
                trade_queue.task_done()
            
            # Sleep briefly to avoid CPU spinning
            time.sleep(0.1)
            
        except queue.Empty:
            # Queue is empty, sleep longer
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Error in queue processor: {e}")
            time.sleep(1)  # Sleep longer on error

def shadow_processor_thread():
    """
    Background thread that processes the shadow queue for cold symbols
    This runs parallel to the main queue processor but doesn't execute trades
    unless they have high confidence
    """
    logging.info("Shadow queue processor thread started")
    
    while True:
        try:
            # Only proceed if there are items in the shadow queue
            if shadow_queue.empty():
                time.sleep(1)
                continue
                
            # Get the highest priority shadow trade
            priority, shadow_data = shadow_queue.get(block=False)
            
            # Log this shadow opportunity
            symbol = shadow_data.get("symbol", "UNKNOWN")
            confidence = shadow_data.get("confidence", 0)
            logging.info(f"Processing shadow opportunity for {symbol} with confidence {confidence:.2f}")
            
            # Check if this opportunity has high enough confidence to execute
            if confidence >= 0.8:  # Only execute very high confidence shadow trades
                logging.info(f"Shadow opportunity for {symbol} has high confidence, promoting to main queue")
                
                # Promote to main queue with high priority
                add_trade_to_queue(shadow_data, priority=9.0)
            else:
                logging.info(f"Shadow opportunity for {symbol} has insufficient confidence, keeping as shadow")
                
                # Record this shadow opportunity for analysis
                try:
                    shadow_log_file = os.path.join(LOGS_DIR, "shadow_opportunities.json")
                    shadow_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "confidence": confidence,
                        "pattern": shadow_data.get("pattern", "unknown"),
                        "side": shadow_data.get("side", "unknown"),
                        "price": shadow_data.get("entry", 0)
                    }
                    
                    with open(shadow_log_file, "a") as f:
                        f.write(json.dumps(shadow_entry) + "\n")
                except Exception as e:
                    logging.error(f"Failed to log shadow opportunity: {e}")
            
            # Mark task as done
            shadow_queue.task_done()
            
            # Sleep briefly between processing shadow opportunities
            time.sleep(0.2)
            
        except queue.Empty:
            # Shadow queue is empty, sleep longer
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error in shadow processor: {e}")
            time.sleep(1)  # Sleep longer on error

def add_trade_to_queue(trade_signal: Dict[str, Any], priority: float = 0.0):
    """
    Add a trade to the priority queue
    Higher priority values get processed first
    """
    try:
        # Ensure trade has a unique ID
        if "trade_id" not in trade_signal:
            trade_signal["trade_id"] = f"trade_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
        # Negate priority because queue is a min-heap
        trade_queue.put((-priority, trade_signal))
        logging.info(f"Added trade for {trade_signal.get('symbol')} to queue with priority {priority}")
    except Exception as e:
        logging.error(f"Failed to add trade to queue: {e}")

def add_to_shadow_queue(trade_signal: Dict[str, Any], priority: float = 0.0):
    """
    Add a trade to the shadow queue for cold symbols
    """
    try:
        # Ensure trade has a unique ID
        if "trade_id" not in trade_signal:
            trade_signal["trade_id"] = f"shadow_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
        # Mark as shadow trade
        trade_signal["shadow"] = True
            
        # Negate priority because queue is a min-heap
        shadow_queue.put((-priority, trade_signal))
        logging.info(f"Added shadow trade for {trade_signal.get('symbol')} with priority {priority}")
    except Exception as e:
        logging.error(f"Failed to add to shadow queue: {e}")

def mark_trade_complete(trade_id: str, success: bool = True, pnl: float = 0.0):
    """
    Mark a trade as complete and remove from active trades
    """
    try:
        with trade_lock:
            # Find the trade by ID
            for i, trade in enumerate(active_trades):
                if trade.get("trade_id") == trade_id:
                    # Remove from active trades
                    completed_trade = active_trades.pop(i)
                    
                    # Update risk monitor
                    risk_monitor.remove_trade(trade_id, pnl)
                    
                    # Update capital manager with the result
                    if hasattr(capital_manager, "close_trade"):
                        capital_manager.close_trade(trade_id, pnl)
                        # Update risk monitor with new capital
                        risk_monitor.set_capital(capital_manager.get_current_balance())
                    
                    # Update trade tracking
                    from strategy_selection import track_trade_outcome
                    symbol = completed_trade.get("symbol", "UNKNOWN")
                    confidence = completed_trade.get("confidence", 0.5)
                    strategy_name = completed_trade.get("strategy_name", "unknown")
                    track_trade_outcome(symbol, success, pnl, confidence, strategy_name)
                    
                    logging.info(f"Trade {trade_id} completed with PnL: ${pnl:.2f}")
                    return True
                    
            logging.warning(f"Trade {trade_id} not found in active trades")
            return False
    except Exception as e:
        logging.error(f"Error marking trade complete: {e}")
        return False

def process_cold_symbols():
    """
    Process cold symbols to check for new opportunities
    """
    try:
        # Run shadow analysis on cold symbols
        opportunities = run_shadow_analysis_for_cold_symbols(max_symbols=5)
        
        if opportunities:
            logging.info(f"Found {len(opportunities)} opportunities in cold symbols")
            
            # Process each opportunity
            for opportunity in opportunities:
                symbol = opportunity.get("symbol")
                signal = opportunity.get("signal")
                
                if not signal:
                    continue
                    
                # Run through capital allocation and validation
                try:
                    allocated = capital_allocator.allocate(signal)
                    
                    # Add to shadow queue to be potentially executed if conditions are right
                    add_to_shadow_queue(allocated, priority=opportunity.get("opportunity_score", 0.5) * 10)
                    
                    # Log this opportunity
                    logging.info(f"Added cold symbol {symbol} to shadow queue with score {opportunity.get('opportunity_score', 0.5):.2f}")
                    
                except Exception as e:
                    logging.error(f"Failed to process cold symbol opportunity for {symbol}: {e}")
                    
        return len(opportunities)
    except Exception as e:
        logging.error(f"Error processing cold symbols: {e}")
        return 0

def run_enhanced_pipeline(symbol: str, lightweight: bool = False):
    """
    Enhanced trading pipeline with support for multiple concurrent trades
    
    Args:
        symbol: Symbol to analyze
        lightweight: If True, run a lightweight analysis (for cold symbols)
    """
    try:
        # Step 1: Get market data and create bundle
        bundle = assemble_data_bundle(symbol, lightweight=lightweight)
        if not bundle:
            logging.warning(f"Failed to create data bundle for {symbol}")
            return None
            
        # Step 2: Analyze with specialized detectors
        # Whale activity detection
        if whale_detector_available:
            whale_flags = detect_whale_activity(bundle.get("market_data", {}))
            bundle["whale_flags"] = whale_flags
        
        # Sentiment analysis
        if sentiment_analyzer_available:
            sentiment = extract_sentiment_score(symbol)
            bundle["sentiment"] = sentiment
        
        # Current price is needed for signal generation
        current_price = bundle.get("market_data", {}).get("price", 0)
        if not current_price:
            ticker_data = bundle.get("market_data", {}).get("ticker", {})
            if ticker_data:
                current_price = float(ticker_data.get("lastPrice", 0))
            
        if current_price <= 0:
            logging.warning(f"Invalid price ({current_price}) for {symbol}, skipping")
            return None
        
        # Load the RL agent
        try:
            from rl_agent import TradingAgent, create_state_from_bundle, action_to_trade_signal
            
            # Create agent (with saved model)
            agent = TradingAgent(state_size=20, action_size=5)  # Adjust sizes as needed
            agent.load("models/rl_agent.h5")  # Load pretrained model
            
            # Create state from bundle
            state = create_state_from_bundle(bundle)
            
            # Get action from agent
            action = agent.act(state, training=False)
            
            # Convert action to trade signal
            rl_signal = action_to_trade_signal(action, bundle, current_price)
            
            # Set default signal to None for evaluation
            trade_signal = None
            
            # Either use RL signal or continue with existing strategy tournament
            if rl_signal.get("decision") == "trade" and rl_signal.get("confidence", 0) > 0.7:
                # Use RL agent's trade decision
                trade_signal = rl_signal
                logging.info(f"Using RL agent trade signal for {symbol} with confidence {rl_signal.get('confidence', 0):.2f}")
            
        except ImportError:
            logging.warning("rl_agent module not found, falling back to strategy tournament")
            trade_signal = None
        except Exception as e:
            logging.warning(f"RL agent evaluation failed: {e}, falling back to strategy tournament")
            trade_signal = None
        
        # Step 3: Strategy Selection (if RL agent didn't provide a valid signal)
        symbol_tag = bundle.get("symbol_tag", "neutral")
        if not trade_signal:
            trade_signal = strategy_tournament(bundle, current_price, symbol_tag)
        
        # Always generate a trade signal if possible (force a trade)
        if not trade_signal or trade_signal.get("confidence", 0) <= 0:
            logging.info(f"No valid trade signal for {symbol}, will force a trade")
            
            # Import the forced trade generator function from strategy_selection
            from strategy_selection import forced_trade_signal
            trade_signal = forced_trade_signal(bundle, current_price, symbol_tag)
            
            # If still no signal, return None
            if not trade_signal:
                logging.warning(f"Could not generate forced trade for {symbol}")
                return None
            
        # Step 4: Capital Allocation
        try:
            allocated = capital_allocator.allocate(trade_signal)
        except Exception as e:
            logging.warning(f"Capital allocation failed for {symbol}: {e}, using fallback allocation")
            # Simple fallback allocation
            allocated = trade_signal.copy()
            price = trade_signal.get("entry", current_price)
            # Determine position size based on price
            if price > 10000:
                quantity = 0.001
            elif price > 1000:
                quantity = 0.01
            elif price > 100:
                quantity = 0.1
            elif price > 10:
                quantity = 1
            else:
                quantity = 10
            allocated["quantity"] = quantity
            allocated["capital_allocated"] = price * quantity
                
        # Step 5: Capital Management Validation
        try:
            current_capital = capital_allocator.get_current_capital()
            validated = capital_manager.validate(allocated)
            if not validated.get("valid", False):
                logging.info(f"Capital Manager rejected trade for {symbol}: {validated.get('reason')}")
                return None
                
            trade_data = validated.get("trade", allocated)
        except Exception as e:
            logging.warning(f"Capital validation failed for {symbol}: {e}, using unvalidated allocation")
            trade_data = allocated
        
        # Calculate trade priority based on confidence and other factors
        confidence = trade_data.get("confidence", 0.5)
        priority = confidence * 10  # Base priority from confidence
        
        # Boost priority for special situations
        if trade_data.get("method") == "combined":
            priority += 2  # Boost for combined patterns
            
        trade_type = trade_data.get("trade_type", "swing")
        if trade_type == "nano":
            priority += 1  # Boost for HFT trades
        elif trade_type == "sniper":
            priority += 1.5  # Boost for event-driven trades
            
        # Boost for whale presence
        if bundle.get("whale_flags", {}).get("whale_present", False):
            priority += 1
            
        # Boost for strong sentiment
        sentiment_score = bundle.get("sentiment", {}).get("sentiment_score", 0)
        if abs(sentiment_score) > 0.5:
            priority += 0.5
            
        # Add to queue with calculated priority
        add_trade_to_queue(trade_data, priority)
        
        return trade_data
        
    except Exception as e:
        logging.exception(f"Enhanced pipeline failed for {symbol}: {e}")
        return None

def run_pipeline(symbol: str, max_retries: int = 2, retry_delay: int = 5):
    """
    Main entry point for the trading pipeline - enhanced version
    
    Args:
        symbol: Trading pair to analyze
        max_retries: Number of times to retry in case of failure
        retry_delay: Seconds to wait between retries
    """
    logging.info(f"Running pipeline for {symbol}")
    
    for attempt in range(max_retries + 1):
        try:
            # Run the enhanced pipeline
            result = run_enhanced_pipeline(symbol)
            
            if result:
                return result
            else:
                if attempt < max_retries:
                    logging.warning(f"No result for {symbol}, retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logging.warning(f"No result for {symbol} after {max_retries} attempts, skipping")
                    return None
                    
        except Exception as e:
            if attempt < max_retries:
                logging.warning(f"Pipeline error for {symbol}: {e}, retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logging.exception(f"Pipeline failed for {symbol} after {max_retries} attempts: {e}")
                return None

# Start the queue processor thread
queue_processor = threading.Thread(target=queue_processor_thread, daemon=True)
queue_processor.start()

# Start the shadow processor thread
shadow_processor = threading.Thread(target=shadow_processor_thread, daemon=True)
shadow_processor.start()

# Log startup
logging.info("Pipeline initialized - queue and shadow processors active")