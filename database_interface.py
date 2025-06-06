# database_interface.py

import sqlite3
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

DB_PATH = "./beast.db"

class DatabaseInterface:
    def __init__(self):
        # Create the database directory if it doesn't exist
        db_dir = os.path.dirname(DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        # Trades table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT,
                strategy TEXT,
                pattern_id INTEGER,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                capital REAL,
                risk_profile TEXT,
                trade_type TEXT,
                signal_strength REAL,
                timestamp TEXT,
                side TEXT,
                confidence REAL
            )
        """)
        
        # Pattern feedback table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER,
                success INTEGER,
                confidence REAL,
                strategy TEXT,
                timestamp TEXT,
                pnl REAL
            )
        """)
        
        # Pattern history table for historical matching
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                pattern TEXT,
                direction TEXT,
                outcome TEXT,
                confidence REAL,
                symbol TEXT,
                rsi REAL,
                volume_normalized REAL,
                volatility_normalized REAL,
                price REAL
            )
        """)
        
        # Symbol performance table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbol_performance (
                symbol TEXT PRIMARY KEY,
                trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                pnl REAL,
                avg_confidence REAL,
                last_trade_time TEXT,
                status TEXT
            )
        """)
        
        self.conn.commit()

    def log_trade(self, trade: dict):
        """Log a new trade to the database"""
        try:
            now = datetime.utcnow().isoformat()
            
            # Generate trade_id if not present
            trade_id = trade.get("trade_id")
            if not trade_id:
                trade_id = f"trade_{now}_{trade.get('symbol', 'unknown')}"
                
            self.cursor.execute("""
                INSERT INTO trades (
                    trade_id, symbol, strategy, pattern_id, entry_price, stop_loss, take_profit, 
                    capital, risk_profile, trade_type, signal_strength, timestamp, side, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                trade.get("symbol"),
                trade.get("strategy_name"),
                trade.get("pattern_id"),
                trade.get("entry"),
                trade.get("stopLoss"),
                trade.get("takeProfit"),
                trade.get("capital_allocated"),
                trade.get("risk_profile"),
                trade.get("trade_type"),
                trade.get("confidence"),
                now,
                trade.get("side"),
                trade.get("confidence", 0.5)
            ))
            self.conn.commit()
            logging.info(f"[DB] Trade logged successfully. ID: {trade_id}")
            return trade_id
        except Exception as e:
            logging.exception(f"[DB ERROR] Failed to log trade: {e}")
            return None

    def log_pattern_feedback(self, pattern_id: int, success: bool, confidence: float, strategy: str, pnl: float = 0.0):
        """Log feedback on pattern performance"""
        try:
            now = datetime.utcnow().isoformat()
            self.cursor.execute("""
                INSERT INTO pattern_feedback (pattern_id, success, confidence, strategy, timestamp, pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (pattern_id, int(success), confidence, strategy, now, pnl))
            self.conn.commit()
            logging.info(f"[DB] Pattern feedback logged for ID {pattern_id}")
        except Exception as e:
            logging.exception(f"[DB ERROR] Failed to log feedback: {e}")

    def log_pattern_history(self, pattern: Dict[str, Any], market_data: Dict[str, Any], outcome: str = "unknown"):
        """Log pattern for historical matching"""
        try:
            now = datetime.utcnow().isoformat()
            
            # Extract pattern details
            pattern_name = pattern.get("pattern", "unknown")
            direction = pattern.get("type", "neutral")
            confidence = pattern.get("confidence", 0.5)
            symbol = market_data.get("symbol", "unknown")
            
            # Extract market conditions
            indicator_data = market_data.get("indicator_data", {})
            if not isinstance(indicator_data, dict):
                # If it's a DataFrame, extract latest values
                try:
                    rsi = float(indicator_data.get("RSI_14", {}).iloc[-1])
                    volume = float(indicator_data.get("Volume", {}).iloc[-1])
                    atr = float(indicator_data.get("ATR_14", {}).iloc[-1])
                    price = float(indicator_data.get("Close", {}).iloc[-1])
                except:
                    rsi = 50.0
                    volume = 1000000.0
                    atr = 100.0
                    price = 0.0
            else:
                # Default values if not available
                rsi = 50.0
                volume = 1000000.0
                atr = 100.0
                price = 0.0
            
            # Normalize volume and volatility for better comparison
            volume_normalized = volume / 1000000.0  # Normalize to millions
            volatility_normalized = atr / price if price > 0 else 0.0  # Normalize to percentage of price
            
            self.cursor.execute("""
                INSERT INTO pattern_history (
                    date, pattern, direction, outcome, confidence, symbol, 
                    rsi, volume_normalized, volatility_normalized, price
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                pattern_name,
                direction,
                outcome,
                confidence,
                symbol,
                rsi,
                volume_normalized,
                volatility_normalized,
                price
            ))
            self.conn.commit()
            logging.info(f"[DB] Pattern history logged: {pattern_name} ({direction}) for {symbol}")
        except Exception as e:
            logging.exception(f"[DB ERROR] Failed to log pattern history: {e}")

    def update_symbol_performance(self, symbol: str, success: bool, pnl: float, confidence: float):
        """Update symbol performance metrics"""
        try:
            now = datetime.utcnow().isoformat()
            
            # Get current stats
            self.cursor.execute("""
                SELECT trades, wins, losses, pnl, avg_confidence FROM symbol_performance 
                WHERE symbol = ?
            """, (symbol,))
            
            row = self.cursor.fetchone()
            
            if row:
                # Update existing record
                trades, wins, losses, total_pnl, avg_confidence = row
                trades += 1
                wins = wins + 1 if success else wins
                losses = losses + 1 if not success else losses
                total_pnl += pnl
                avg_confidence = ((avg_confidence * (trades - 1)) + confidence) / trades
                
                self.cursor.execute("""
                    UPDATE symbol_performance SET 
                    trades = ?, wins = ?, losses = ?, pnl = ?, avg_confidence = ?, 
                    last_trade_time = ?, status = ?
                    WHERE symbol = ?
                """, (
                    trades, wins, losses, total_pnl, avg_confidence, now, 
                    "hot" if (wins/trades >= 0.6) else ("cold" if (wins/trades < 0.4) else "neutral"),
                    symbol
                ))
            else:
                # Insert new record
                self.cursor.execute("""
                    INSERT INTO symbol_performance (
                        symbol, trades, wins, losses, pnl, avg_confidence, last_trade_time, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, 1, 1 if success else 0, 0 if success else 1, pnl, confidence, now,
                    "neutral"  # Start as neutral until we have more data
                ))
                
            self.conn.commit()
            logging.info(f"[DB] Symbol performance updated for {symbol}")
        except Exception as e:
            logging.exception(f"[DB ERROR] Failed to update symbol performance: {e}")

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a custom query and return results as a list of dictionaries"""
        try:
            self.cursor.execute(query, params)
            columns = [description[0] for description in self.cursor.description]
            results = []
            
            for row in self.cursor.fetchall():
                result = {}
                for i, column in enumerate(columns):
                    result[column] = row[i]
                results.append(result)
                
            return results
        except Exception as e:
            logging.exception(f"[DB ERROR] Failed to execute query: {e}")
            return []

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

# Singleton instance
db_interface = DatabaseInterface()

def find_historical_matches(current_conditions: Dict[str, Any], max_matches: int = 5) -> List[Dict[str, Any]]:
    """
    Find historical matches for current market conditions
    
    Args:
        current_conditions: Dictionary with current market indicators
        max_matches: Maximum number of matches to return
        
    Returns:
        List of historical matches with similar conditions
    """
    try:
        # Extract key conditions for matching
        rsi = current_conditions.get("RSI_14", 50)
        volume = current_conditions.get("Volume", 0) / 1000000.0  # Normalize to millions
        volatility = current_conditions.get("ATR_14", 0)
        price = current_conditions.get("Close", 100)
        
        # Normalize volatility
        volatility_normalized = volatility / price if price > 0 else 0.0
        
        # Search database for similar conditions
        query = f"""
        SELECT date, pattern, direction, outcome, confidence
        FROM pattern_history
        WHERE ABS(rsi - ?) < 5
        AND ABS(volume_normalized - ?) < 0.2
        AND ABS(volatility_normalized - ?) < 0.15
        ORDER BY confidence DESC
        LIMIT ?
        """
        
        params = (rsi, volume, volatility_normalized, max_matches)
        matches = db_interface.execute_query(query, params)
        
        logging.info(f"Found {len(matches)} historical matches")
        return matches
        
    except Exception as e:
        logging.error(f"Failed to find historical matches: {e}")
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_trade = {
        "symbol": "BTCUSDT",
        "strategy_name": "gamma_scalping",
        "pattern_id": 42,
        "entry": 24500,
        "stopLoss": 24300,
        "takeProfit": 24800,
        "capital_allocated": 50,
        "risk_profile": "aggressive",
        "trade_type": "nano",
        "signal_strength": 0.91,
        "side": "BUY",
        "confidence": 0.95
    }
    db_interface.log_trade(test_trade)
    db_interface.log_pattern_feedback(42, True, 0.91, "gamma_scalping", 300.0)