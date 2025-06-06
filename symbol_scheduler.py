import logging
import time
import signal
import sys
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from typing import List, Dict, Set, Optional
from datetime import datetime, timezone
import json
from dotenv import load_dotenv
from pipeline import run_pipeline, process_cold_symbols
from binance_interface import get_futures_market_data, classify_symbol_behavior

# Define the missing get_cold_symbols function here:
def get_cold_symbols(symbol_data=None) -> List[str]:
    """
    Identify cold symbols with missing or insufficient data.
    If symbol_data is None, returns empty list to avoid breaking code.
    """
    if symbol_data is None:
        # Defensive: no data given, return empty list
        return []

    cold_symbols = []
    for symbol, data in symbol_data.items():
        if not data or data.get("valid_endpoints", 0) < 5:
            cold_symbols.append(symbol)
    return cold_symbols


# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# UPDATED - Enhanced logging configuration with encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/beast_output.log", encoding='utf-8'),  # Added encoding
        logging.StreamHandler()
    ]
)

load_dotenv()

# Configuration parameters
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", 10))
ROTATION_INTERVAL = int(os.getenv("ROTATION_INTERVAL", 600))
INTERVAL_SECONDS = int(os.getenv("SYMBOL_INTERVAL", 60))
MAX_THREADS = int(os.getenv("SYMBOL_THREADS", 8))
COLD_SYMBOL_CHECK_INTERVAL = int(os.getenv("COLD_SYMBOL_INTERVAL", 300))  # Check cold symbols every 5 minutes

# State variables
symbol_state: Dict[str, Dict] = {}
selected_symbols: List[str] = []
hot_symbols: Set[str] = set()
processed_symbols_count = 0

# Initialize scheduler with misfire handling
scheduler = BackgroundScheduler(
    job_defaults={
        'misfire_grace_time': 10,  # Allow jobs to be 10 seconds late
        'coalesce': True,  # Combine missed executions
        'max_instances': 1  # Prevent overlapping executions of the same job
    }
)

executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

def get_valid_futures_symbols() -> List[str]:
    """
    Fetch all valid Binance Futures USDT contracts
    """
    try:
        # Try to use the imported function
        from binance_interface import get_valid_futures_symbols as binance_get_symbols
        symbols = binance_get_symbols()
        if symbols:
            return symbols
    except ImportError:
        logging.warning("Using fallback method to get symbols")
        
    # Fallback implementation
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        import requests
        
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logging.error(f"Failed to get exchangeInfo: {response.status_code}")
            return selected_symbols if selected_symbols else []
            
        info = response.json()
        
        symbols = [
            s["symbol"]
            for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL" and 
               s.get("status") == "TRADING" and 
               s.get("symbol", "").endswith("USDT")
        ]
        
        logging.info(f"Found {len(symbols)} valid futures symbols")
        return symbols
    except Exception as e:
        logging.error(f"Failed to get symbols: {e}")
        return selected_symbols if selected_symbols else []

def load_symbol_status() -> Dict[str, Dict]:
    """
    Load saved symbol status from disk if available
    """
    try:
        filepath = os.path.join(log_dir, "symbol_status.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.error(f"Failed to load symbol status: {e}")
        return {}

def save_symbol_status():
    """
    Save current symbol status to disk
    """
    try:
        filepath = os.path.join(log_dir, "symbol_status.json")
        
        # Create a dictionary with current status
        status = {
            "hot_symbols": list(hot_symbols),
            "cold_symbols": list(get_cold_symbols(symbol_state)),  # Pass symbol_state here
            "selected_symbols": selected_symbols,
            "timestamp": datetime.now().isoformat(),
            "symbol_stats": symbol_state
        }
        
        with open(filepath, 'w') as f:
            json.dump(status, f, indent=2)
            
        logging.info("Symbol status saved to disk")
    except Exception as e:
        logging.error(f"Failed to save symbol status: {e}")

def classify_and_rank_symbols(symbols: List[str]) -> List[str]:
    """
    Classify and rank symbols with improved efficiency - UPDATED VERSION
    """
    # Define high priority symbols that should always be considered  
    high_priority = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT"
    ]
    
    # Get cold symbols
    from strategy_selection import get_cold_symbols, get_hot_symbols
    cold_symbols = get_cold_symbols()
    
    global hot_symbols
    hot_symbols = get_hot_symbols()
    
    # Combine high priority and hot symbols, ensure uniqueness
    priority_symbols = list(set(high_priority + list(hot_symbols)))
    
    # Filter out cold symbols
    priority_symbols = [s for s in priority_symbols if s not in cold_symbols]
    
    # Log cold symbols if any (UPDATED - no emoji)
    if cold_symbols:
        logging.info(f"COLD symbols being excluded from selection: {list(cold_symbols)}")
        
    # Log hot symbols (UPDATED - no emoji)  
    if hot_symbols:
        logging.info(f"HOT symbols prioritized in selection: {list(hot_symbols)}")
    
    # If we need more symbols, add some from remaining non-cold symbols
    remaining_symbols = [s for s in symbols if s not in priority_symbols and s not in cold_symbols]
    if len(priority_symbols) < MAX_SYMBOLS and remaining_symbols:
        additional_needed = min(MAX_SYMBOLS - len(priority_symbols), len(remaining_symbols))
        additional_symbols = random.sample(remaining_symbols, additional_needed)
        priority_symbols.extend(additional_symbols)
    
    # Limit to max symbols
    priority_symbols = priority_symbols[:MAX_SYMBOLS]
    
    # Process only these priority symbols
    scored = []
    processed_count = 0
    
    logging.info(f"Processing {len(priority_symbols)} priority symbols")
    
    for symbol in priority_symbols:
        try:
            processed_count += 1  
            logging.info(f"Processing symbol {processed_count}/{len(priority_symbols)}: {symbol}")
            
            data = get_futures_market_data(symbol)
            if not data:
                logging.warning(f"No data returned for {symbol}, skipping")
                continue
                
            tag = classify_symbol_behavior(data)
            score = {
                "rally": 5,
                "breakout": 4,  
                "pullback": 3,
                "fall": 2,
                "breakdown": 1,
                "neutral": 0
            }.get(tag, 0)
            
            # Boost score for hot symbols
            if symbol in hot_symbols:
                score += 2
                
            scored.append((symbol, score))
            symbol_state[symbol] = {
                "tag": tag,
                "score": score,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
                
        except Exception as e:
            logging.warning(f"[Scheduler] Error for {symbol}: {e}")
            
        # Add a small delay between requests to avoid rate limits
        time.sleep(random.uniform(0.2, 0.5))
    
    # Sort by score and return top symbols
    sorted_symbols = sorted(scored, key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in sorted_symbols]
    logging.info(f"Selected top {len(selected)} symbols from {len(scored)} processed symbols")
    
    # Save symbol status to disk
    save_symbol_status()
    
    return selected

def dispatch_pipeline(symbol: str):
    """
    Run the pipeline for a symbol with better error handling
    """
    global processed_symbols_count
    
    try:
        logging.info(f"Dispatching pipeline for {symbol}")
        start_time = time.time()
        run_pipeline(symbol=symbol)
        elapsed = time.time() - start_time
        processed_symbols_count += 1
        logging.info(f"Pipeline for {symbol} completed in {elapsed:.2f} seconds. Total processed: {processed_symbols_count}")
    except Exception as e:
        logging.error(f"Pipeline failed for {symbol}: {e}")

def check_cold_symbols():
    """
    Periodically check cold symbols for new opportunities
    """
    try:
        logging.info("Running cold symbol analysis")
        start_time = time.time()
        
        # Process cold symbols
        opportunities_count = process_cold_symbols()
        
        elapsed = time.time() - start_time
        logging.info(f"Cold symbol analysis completed in {elapsed:.2f} seconds. Found {opportunities_count} opportunities")
    except Exception as e:
        logging.error(f"Cold symbol analysis failed: {e}")

def rotate_symbols():
    """
    Rotate and select symbols - UPDATED VERSION (no emojis)
    """
    global selected_symbols
    
    start_time = time.time()
    try:
        # Log the current selected symbols before rotation
        if selected_symbols:
            logging.info(f"Current symbols before rotation: {selected_symbols}")
        
        # Get all valid symbols
        all_symbols = get_valid_futures_symbols()
        if not all_symbols:
            logging.error("No valid symbols found. Keeping existing selection.")
            return
        
        # Get the cold symbols list from strategy_selection
        try:
            from strategy_selection import get_cold_symbols, reset_all_cold_symbols
            cold_symbols = get_cold_symbols()
            
            # Every 5 rotations, give cold symbols another chance
            if hasattr(rotate_symbols, 'counter'):
                rotate_symbols.counter += 1
            else:
                rotate_symbols.counter = 1
                
            if rotate_symbols.counter >= 5:
                reset_all_cold_symbols()
                rotate_symbols.counter = 0
                # UPDATED - no emoji
                logging.info("RESET: Cold symbol reset triggered - giving all symbols a fresh chance")
                cold_symbols = set()  # Now empty after reset
                
        except ImportError:
            logging.warning("Cold symbol tracking not available in strategy_selection module")
            cold_symbols = set()
        
        # Select and rank symbols
        new_symbols = classify_and_rank_symbols(all_symbols)
        
        if new_symbols:
            # Check for changes in the selected symbols
            if set(new_symbols) != set(selected_symbols):
                added = [s for s in new_symbols if s not in selected_symbols]
                removed = [s for s in selected_symbols if s not in new_symbols]
                
                if added or removed:
                    logging.info(f"Symbol changes - Added: {added} | Removed: {removed}")
            
            selected_symbols = new_symbols
            logging.info(f"Top {len(selected_symbols)} symbols selected: {selected_symbols}")
            
            # Save selected symbols to disk
            symbol_status_file = os.path.join(log_dir, "selected_symbols.json")
            with open(symbol_status_file, 'w') as f:
                json.dump({
                    "symbols": selected_symbols,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
        else:
            logging.warning("Symbol rotation failed to find new symbols. Keeping existing selection.")
    except Exception as e:
        logging.error(f"Symbol rotation failed: {e}")
        
    elapsed = time.time() - start_time
    logging.info(f"Symbol rotation completed in {elapsed:.2f} seconds")

def scheduled_execution():
    """
    Execute pipelines for selected symbols with improved timing tracking
    """
    start_time = time.time()
    
    if not selected_symbols:
        logging.warning("No symbols selected for execution")
        return
    
    logging.info(f"Starting execution cycle for {len(selected_symbols)} symbols")
        
    # Submit all symbols to the executor
    for symbol in selected_symbols:
        executor.submit(dispatch_pipeline, symbol)
    
    elapsed = time.time() - start_time
    logging.info(f"Scheduled execution submission completed in {elapsed:.2f} seconds")

def graceful_shutdown(signum, frame):
    """
    Handle graceful shutdown
    """
    logging.info("Shutting down scheduler and executor...")
    
    # Save status before shutting down
    save_symbol_status()
    
    scheduler.shutdown()
    executor.shutdown(wait=False)
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# Initialize and start scheduler
def main():
    try:
        # Create required directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("trade_signals", exist_ok=True)
        
        # Load previously saved status if available
        status = load_symbol_status()
        if status:
            global hot_symbols, selected_symbols, symbol_state
            hot_symbols = set(status.get("hot_symbols", []))
            if "selected_symbols" in status:
                selected_symbols = status["selected_symbols"]
            if "symbol_stats" in status:
                symbol_state = status["symbol_stats"]
        
        logging.info("Starting symbol rotation...")
        rotate_symbols()
        
        # Add jobs with more descriptive IDs and logging
        symbol_rotation_job = scheduler.add_job(
            rotate_symbols, 
            IntervalTrigger(seconds=ROTATION_INTERVAL), 
            id="symbol_rotation",
            name="Symbol Rotation Job"
        )
        logging.info(f"Added symbol rotation job with ID: {symbol_rotation_job.id}")
        
        execution_job = scheduler.add_job(
            scheduled_execution, 
            IntervalTrigger(seconds=INTERVAL_SECONDS), 
            id="symbol_execution",
            name="Symbol Execution Job"
        )
        logging.info(f"Added execution job with ID: {execution_job.id}")
        
        # Add job for cold symbol analysis
        cold_symbol_job = scheduler.add_job(
            check_cold_symbols,
            IntervalTrigger(seconds=COLD_SYMBOL_CHECK_INTERVAL),
            id="cold_symbol_analysis",
            name="Cold Symbol Analysis Job"
        )
        logging.info(f"Added cold symbol analysis job with ID: {cold_symbol_job.id}")
        
        scheduler.start()
        
        logging.info("BEAST scheduler is running...")
        logging.info(f"Processing up to {MAX_SYMBOLS} symbols with {MAX_THREADS} threads")
        logging.info(f"Symbol rotation interval: {ROTATION_INTERVAL}s, Execution interval: {INTERVAL_SECONDS}s")
        logging.info(f"Cold symbol analysis interval: {COLD_SYMBOL_CHECK_INTERVAL}s")
        
        # Print current scheduler state
        for job in scheduler.get_jobs():
            logging.info(f"Job: {job.name} (ID: {job.id})")

        # Keep the main thread alive
        while True:
            time.sleep(5)
            
    except Exception as e:
        logging.critical(f"Fatal error in scheduler: {e}")
        graceful_shutdown(None, None)

if __name__ == "__main__":
    main()