# pattern_feedback.py

import logging
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

# Direct import of bayesian_scorer singleton from bayesian_memory
from bayesian_memory import bayesian_scorer

class PatternFeedbackLogger:
    def __init__(self, log_file: str = "logs/pattern_feedback.json"):
        self.logs = []
        self.log_file = log_file
        self._create_log_dir()
        self._load_logs()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Store pattern stats in memory for quick access
        self.pattern_stats = {}
        self._initialize_pattern_stats()
        
        logging.info("PatternFeedbackLogger initialized and connected to Bayesian scorer")

    def _create_log_dir(self):
        """Create log directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def _load_logs(self):
        """Load existing logs if available"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    self.logs = json.load(f)
                logging.info(f"Loaded {len(self.logs)} pattern feedback logs")
        except Exception as e:
            logging.error(f"Failed to load pattern feedback logs: {e}")
            self.logs = []
    
    def _initialize_pattern_stats(self):
        """Initialize pattern stats from logs and bayesian memory"""
        try:
            # Group logs by pattern_id
            for log in self.logs:
                pattern_id = log.get("pattern_id")
                if pattern_id:
                    if pattern_id not in self.pattern_stats:
                        self.pattern_stats[pattern_id] = {
                            "total": 0,
                            "wins": 0,
                            "losses": 0,
                            "pnl": 0.0,
                            "bayesian_score": 0.5  # Default score
                        }
                    
                    stats = self.pattern_stats[pattern_id]
                    stats["total"] += 1
                    if log.get("success", False):
                        stats["wins"] += 1
                    else:
                        stats["losses"] += 1
                    stats["pnl"] += log.get("pnl", 0.0)
            
            # Update Bayesian scores
            for pattern_id in self.pattern_stats:
                bayesian_score = bayesian_scorer.compute_score(pattern_id)
                self.pattern_stats[pattern_id]["bayesian_score"] = bayesian_score
                
            logging.info(f"Initialized stats for {len(self.pattern_stats)} patterns")
        except Exception as e:
            logging.error(f"Error initializing pattern stats: {e}")

    def _save_logs(self):
        """Save logs to disk"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save pattern feedback logs: {e}")

    def fetch_and_log_outcome(self, pattern_id: int, trade_result: Optional[Dict[str, Any]] = None):
        """
        Looks up trade result and updates pattern reliability using Bayesian scorer.
        
        Args:
            pattern_id: The ID of the pattern to update
            trade_result: Optional trade result data. If None, will attempt to get from order_manager
        """
        try:
            # If trade_result not provided, try to get from order_manager
            if trade_result is None:
                try:
                    from order_manager import get_trade_result_by_pattern_id
                    trade_result = get_trade_result_by_pattern_id(pattern_id)
                except (ImportError, AttributeError):
                    logging.warning(f"Order manager not available, cannot get trade result for pattern ID {pattern_id}")
            
            # If we don't have a real result, create a simulated one for testing
            if trade_result is None:
                logging.info(f"[SIMULATED] No trade data for pattern ID {pattern_id} — using dummy success=True")
                outcome = True
                pnl = 0.0
            else:
                # Extract outcome and PnL from the result
                outcome = trade_result.get("success")
                pnl = trade_result.get("pnl", 0.0)
                
                if outcome is None:
                    logging.warning(f"Missing outcome for pattern ID {pattern_id}")
                    return

            # Update the Bayesian scorer - INTEGRATED CONNECTION POINT
            bayesian_scorer.update(pattern_id, outcome, pnl)

            # Update local pattern stats
            with self.lock:
                if pattern_id not in self.pattern_stats:
                    self.pattern_stats[pattern_id] = {
                        "total": 0,
                        "wins": 0,
                        "losses": 0,
                        "pnl": 0.0,
                        "bayesian_score": 0.5
                    }
                
                stats = self.pattern_stats[pattern_id]
                stats["total"] += 1
                if outcome:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                stats["pnl"] += pnl
                
                # Update Bayesian score
                stats["bayesian_score"] = bayesian_scorer.compute_score(pattern_id)

            # Create log entry
            log_entry = {
                "pattern_id": pattern_id,
                "pattern": trade_result.get("pattern", "unknown") if trade_result else "unknown",
                "symbol": trade_result.get("symbol", "unknown") if trade_result else "unknown",
                "strategy": trade_result.get("strategy_name", "unknown") if trade_result else "unknown",
                "side": trade_result.get("positionSide", "LONG") if trade_result else "LONG",
                "confidence": trade_result.get("confidence", 0.5) if trade_result else 0.5,
                "success": outcome,
                "pnl": pnl,
                "timestamp": trade_result.get("timestamp", datetime.now().isoformat()) if trade_result else datetime.now().isoformat(),
                "bayesian_score": stats["bayesian_score"]  # Include Bayesian score in log
            }

            # Append to logs with thread safety
            with self.lock:
                self.logs.append(log_entry)
                self._save_logs()
            
            # Try to update the database if available
            try:
                from database_interface import db_interface
                db_interface.log_pattern_feedback(
                    pattern_id, 
                    outcome, 
                    log_entry["confidence"], 
                    log_entry["strategy"],
                    pnl
                )
            except (ImportError, AttributeError):
                pass  # Database not available, skip

            # Log the outcome
            logging.info(f"Logged outcome for pattern {pattern_id}: {'WIN' if outcome else 'LOSS'} with PnL ${pnl:.2f}, Bayesian score: {stats['bayesian_score']:.3f}")

        except Exception as e:
            logging.error(f"Feedback logging failed for pattern ID {pattern_id}: {e}")
    
    def batch_score_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score multiple patterns using the Bayesian scorer - INTEGRATED CONNECTION POINT
        
        Args:
            patterns: List of pattern dictionaries with pattern_id
            
        Returns:
            Same list with bayesian_score added
        """
        try:
            # Delegate directly to Bayesian scorer
            return bayesian_scorer.batch_score(patterns)
        except Exception as e:
            logging.error(f"Error batch scoring patterns: {e}")
            # Fallback to adding scores manually
            for pattern in patterns:
                pattern_id = pattern.get("pattern_id")
                if pattern_id:
                    score = bayesian_scorer.compute_score(pattern_id)
                    pattern["bayesian_score"] = score
            return patterns
    
    def get_pattern_statistics(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get statistics about pattern performance based on feedback logs
        
        Args:
            limit: Maximum number of patterns to include in the statistics
            
        Returns:
            Dictionary with pattern performance statistics
        """
        try:
            # First get top patterns from Bayesian scorer - INTEGRATED CONNECTION POINT
            bayesian_top_patterns = bayesian_scorer.get_top_patterns(min_trades=3, limit=limit)
            
            # Combine with local statistics
            pattern_stats = {}
            
            # First add all patterns from Bayesian scorer
            for pattern in bayesian_top_patterns:
                pattern_id = pattern.get("pattern_id")
                pattern_stats[pattern_id] = {
                    "pattern_id": pattern_id,
                    "total": pattern.get("total", 0),
                    "wins": pattern.get("success", 0),
                    "losses": pattern.get("total", 0) - pattern.get("success", 0),
                    "pnl": pattern.get("pnl", 0.0),
                    "bayesian_score": pattern.get("bayesian_score", 0.5)
                }
            
            # Then add any other patterns from local logs
            for pattern_id, stats in self.pattern_stats.items():
                if pattern_id not in pattern_stats and stats["total"] >= 3:
                    pattern_stats[pattern_id] = {
                        "pattern_id": pattern_id,
                        "total": stats["total"],
                        "wins": stats["wins"],
                        "losses": stats["losses"],
                        "pnl": stats["pnl"],
                        "bayesian_score": stats["bayesian_score"]
                    }
            
            # Calculate additional metrics for all patterns
            results = []
            for pattern_id, stats in pattern_stats.items():
                if stats["total"] > 0:
                    win_rate = stats["wins"] / stats["total"]
                    avg_pnl = stats["pnl"] / stats["total"]
                    
                    # Try to get pattern name from logs
                    pattern_name = "unknown"
                    for log in self.logs:
                        if log.get("pattern_id") == pattern_id:
                            pattern_name = log.get("pattern", "unknown")
                            break
                    
                    results.append({
                        "pattern_id": pattern_id,
                        "pattern": pattern_name,
                        "total": stats["total"],
                        "wins": stats["wins"],
                        "losses": stats["losses"],
                        "win_rate": win_rate,
                        "pnl": stats["pnl"],
                        "avg_pnl": avg_pnl,
                        "bayesian_score": stats["bayesian_score"],
                        "combined_score": (win_rate * 0.4) + (stats["bayesian_score"] * 0.4) + (avg_pnl / max(1, abs(avg_pnl)) * 0.2)  # Combined score using all metrics
                    })
            
            # Sort by combined score (bayesian + win rate + pnl contribution)
            results.sort(key=lambda x: x["combined_score"], reverse=True)
            
            return {
                "total_patterns": len(results),
                "total_feedback_entries": len(self.logs),
                "top_patterns": results[:limit]
            }
        except Exception as e:
            logging.error(f"Failed to get pattern statistics: {e}")
            return {"error": str(e)}
    
    def get_pattern_details(self, pattern_id: int) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific pattern
        
        Args:
            pattern_id: The pattern ID to query
            
        Returns:
            Dictionary with detailed pattern statistics
        """
        try:
            # Get Bayesian stats - INTEGRATED CONNECTION POINT
            bayesian_stats = bayesian_scorer.get_pattern_stats(pattern_id)
            
            # If pattern not found in Bayesian scorer, check local stats
            if not bayesian_stats and pattern_id in self.pattern_stats:
                with self.lock:
                    stats = self.pattern_stats[pattern_id]
                    bayesian_stats = {
                        "pattern_id": pattern_id,
                        "success": stats["wins"],
                        "total": stats["total"],
                        "pnl": stats["pnl"],
                        "bayesian_score": stats["bayesian_score"]
                    }
            
            if not bayesian_stats:
                return {"error": f"Pattern ID {pattern_id} not found"}
            
            # Get related trades from logs
            pattern_trades = []
            for log in self.logs:
                if log.get("pattern_id") == pattern_id:
                    pattern_trades.append(log)
            
            # Sort by timestamp descending
            pattern_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Calculate additional statistics
            if bayesian_stats["total"] > 0:
                win_rate = bayesian_stats["success"] / bayesian_stats["total"]
                avg_pnl = bayesian_stats["pnl"] / bayesian_stats["total"]
            else:
                win_rate = 0
                avg_pnl = 0
            
            return {
                "pattern_id": pattern_id,
                "total_trades": bayesian_stats["total"],
                "wins": bayesian_stats["success"],
                "losses": bayesian_stats["total"] - bayesian_stats["success"],
                "win_rate": win_rate,
                "pnl": bayesian_stats["pnl"],
                "avg_pnl": avg_pnl,
                "bayesian_score": bayesian_stats["bayesian_score"],
                "recent_trades": pattern_trades[:10]  # Last 10 trades
            }
            
        except Exception as e:
            logging.error(f"Failed to get pattern details: {e}")
            return {"error": str(e)}

# Create a singleton instance
pattern_feedback_logger = PatternFeedbackLogger()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with simulated trade results
    test_result = {
        "pattern": "bullish_engulfing",
        "symbol": "BTCUSDT",
        "strategy_name": "momentum",
        "positionSide": "LONG",
        "confidence": 0.85,
        "success": True,
        "pnl": 25.50,
        "timestamp": datetime.now().isoformat()
    }
    
    pattern_feedback_logger.fetch_and_log_outcome(42, test_result)
    
    # Add another test result
    test_result2 = {
        "pattern": "hammer",
        "symbol": "ETHUSDT",
        "strategy_name": "reversal",
        "positionSide": "LONG",
        "confidence": 0.75,
        "success": False,
        "pnl": -15.25,
        "timestamp": datetime.now().isoformat()
    }
    
    pattern_feedback_logger.fetch_and_log_outcome(43, test_result2)
    
    # Test getting statistics
    stats = pattern_feedback_logger.get_pattern_statistics()
    print("Pattern statistics:", json.dumps(stats, indent=2))
    
    # Test batch scoring
    patterns = [
        {"pattern_id": 42, "pattern": "bullish_engulfing"}, 
        {"pattern_id": 43, "pattern": "hammer"}
    ]
    scored_patterns = pattern_feedback_logger.batch_score_patterns(patterns)
    print("Scored patterns:", json.dumps(scored_patterns, indent=2))