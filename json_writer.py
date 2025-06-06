# json_writer.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Create trade_signals directory if it doesn't exist
SIGNALS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_signals")
os.makedirs(SIGNALS_DIR, exist_ok=True)

class JsonWriter:
    """
    Handles writing trade signals to JSON files in a structured format.
    """
    def __init__(self, signals_dir: str = SIGNALS_DIR):
        self.signals_dir = signals_dir
        os.makedirs(self.signals_dir, exist_ok=True)
        logging.info(f"JsonWriter initialized with signals directory: {self.signals_dir}")
    
    def write_trade_signal(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """
        Writes trade signal data to a JSON file with timestamp and unique identifier.
        
        Args:
            trade_data: Dictionary containing the trade signal data
            
        Returns:
            Path to the created file or None if writing failed
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
            filepath = os.path.join(self.signals_dir, filename)
            
            # Write JSON file
            with open(filepath, 'w') as f:
                json.dump(trade_data, f, indent=2)
                
            logging.info(f"Trade signal written to {filepath}")
            return filepath
            
        except Exception as e:
            logging.exception(f"Failed to write trade signal: {e}")
            return None
            
    def read_trade_signal(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Reads a trade signal from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Dictionary containing the trade signal data or None if reading failed
        """
        try:
            with open(filepath, 'r') as f:
                trade_data = json.load(f)
            return trade_data
        except Exception as e:
            logging.exception(f"Failed to read trade signal from {filepath}: {e}")
            return None


# Create a global instance for convenience
json_writer = JsonWriter()

def write_trade_signal_to_file(trade_data: Dict[str, Any]) -> Optional[str]:
    """
    Convenience function for writing trade signals.
    Uses the global JsonWriter instance.
    
    Args:
        trade_data: Dictionary containing the trade signal data
        
    Returns:
        Path to the created file or None if writing failed
    """
    return json_writer.write_trade_signal(trade_data)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test the JsonWriter with a sample trade signal
    sample_trade = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "positionSide": "LONG",
        "entry": 25000.0,
        "stopLoss": 24500.0,
        "takeProfit": 26000.0,
        "quantity": 0.1,
        "leverage": 10,
        "strategy_type": "momentum",
        "strategy_name": "breakout_momentum",
        "confidence": 0.85
    }
    
    filepath = write_trade_signal_to_file(sample_trade)
    if filepath:
        print(f"Test trade signal written to {filepath}")
    else:
        print("Failed to write test trade signal")