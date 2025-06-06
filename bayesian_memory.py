# bayesian_memory.py

import logging
import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

class BayesianScorer:
    def __init__(self, persistence_file: str = "logs/bayesian_scores.json"):
        self.history = {}  # pattern_id: {"success": int, "total": int}
        self.persistence_file = persistence_file
        self.last_save_time = datetime.now()
        self.save_interval = 300  # Save every 5 minutes
        self._load_history()
        logging.info("BayesianScorer initialized with persistence")

    def update(self, pattern_id: int, outcome: bool, pnl: float = 0.0):
        """
        Update the Bayesian history for a pattern with a trade outcome
        
        Args:
            pattern_id: Unique identifier for the pattern
            outcome: True if trade was successful, False otherwise
            pnl: Profit/loss amount from the trade
        """
        if pattern_id not in self.history:
            self.history[pattern_id] = {"success": 0, "total": 0, "pnl": 0.0, "last_updated": None}

        self.history[pattern_id]["total"] += 1
        if outcome:
            self.history[pattern_id]["success"] += 1
        
        # Track PnL
        self.history[pattern_id]["pnl"] = self.history[pattern_id].get("pnl", 0.0) + pnl
        self.history[pattern_id]["last_updated"] = datetime.now().isoformat()
        
        # Log the update
        success_rate = self.history[pattern_id]["success"] / self.history[pattern_id]["total"]
        logging.info(f"Updated pattern {pattern_id}: success rate {success_rate:.2f} ({self.history[pattern_id]['success']}/{self.history[pattern_id]['total']}), PnL: ${self.history[pattern_id]['pnl']:.2f}")
        
        # Periodically save to disk
        current_time = datetime.now()
        if (current_time - self.last_save_time).total_seconds() > self.save_interval:
            self._save_history()
            self.last_save_time = current_time

    def compute_score(self, pattern_id: int, prior: float = 0.5, strength: int = 5) -> float:
        """
        Compute Bayesian score for a pattern based on historical performance
        
        Args:
            pattern_id: Unique identifier for the pattern
            prior: Prior probability (default: 0.5)
            strength: Strength of prior (higher = more weight to prior)
            
        Returns:
            Bayesian score between 0 and 1
        """
        if pattern_id not in self.history:
            return prior  # no data → return prior

        data = self.history[pattern_id]
        alpha = data["success"] + prior * strength
        beta = data["total"] - data["success"] + (1 - prior) * strength

        score = alpha / (alpha + beta)
        return round(score, 4)

    def batch_score(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score multiple pattern matches at once
        
        Args:
            matches: List of pattern match dictionaries containing pattern_id
            
        Returns:
            Same list with bayesian_score added to each match
        """
        for match in matches:
            pattern_id = match.get("pattern_id")
            score = self.compute_score(pattern_id)
            match["bayesian_score"] = score
            
            # Add additional metadata if available
            if pattern_id in self.history:
                match["trade_count"] = self.history[pattern_id]["total"]
                match["success_rate"] = self.history[pattern_id]["success"] / self.history[pattern_id]["total"] \
                    if self.history[pattern_id]["total"] > 0 else 0.5
                match["pattern_pnl"] = self.history[pattern_id].get("pnl", 0.0)
        
        return matches
    
    def get_pattern_stats(self, pattern_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed stats for a specific pattern
        
        Args:
            pattern_id: Unique identifier for the pattern
            
        Returns:
            Dictionary with pattern statistics or None if pattern not found
        """
        if pattern_id not in self.history:
            return None
            
        data = self.history[pattern_id]
        return {
            "pattern_id": pattern_id,
            "success": data["success"],
            "total": data["total"],
            "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
            "pnl": data.get("pnl", 0.0),
            "bayesian_score": self.compute_score(pattern_id),
            "last_updated": data.get("last_updated")
        }
    
    def get_top_patterns(self, min_trades: int = 5, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top performing patterns based on Bayesian score
        
        Args:
            min_trades: Minimum number of trades required
            limit: Maximum number of patterns to return
            
        Returns:
            List of pattern stats sorted by Bayesian score
        """
        top_patterns = []
        
        for pattern_id, data in self.history.items():
            if data["total"] >= min_trades:
                stats = self.get_pattern_stats(pattern_id)
                if stats:
                    top_patterns.append(stats)
        
        # Sort by Bayesian score (highest first)
        top_patterns.sort(key=lambda x: x["bayesian_score"], reverse=True)
        
        return top_patterns[:limit]

    def _save_history(self):
        """Save Bayesian history to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            
            with open(self.persistence_file, 'w') as f:
                json.dump(self.history, f, indent=2)
                
            logging.info(f"Saved Bayesian history with {len(self.history)} patterns")
        except Exception as e:
            logging.error(f"Failed to save Bayesian history: {e}")

    def _load_history(self):
        """Load Bayesian history from disk if available"""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r') as f:
                    self.history = json.load(f)
                logging.info(f"Loaded Bayesian history with {len(self.history)} patterns")
        except Exception as e:
            logging.error(f"Failed to load Bayesian history: {e}")
            self.history = {}


# Singleton for reuse
bayesian_scorer = BayesianScorer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bayesian_scorer.update(1, True, 5.0)
    bayesian_scorer.update(1, False, -2.0)
    bayesian_scorer.update(1, True, 3.5)
    print("Score for ID 1:", bayesian_scorer.compute_score(1))
    
    # Test batch scoring
    test_matches = [
        {"pattern_id": 1, "pattern": "bullish_engulfing"},
        {"pattern_id": 2, "pattern": "hammer"}
    ]
    scored_matches = bayesian_scorer.batch_score(test_matches)
    print("Scored matches:", scored_matches)
    
    # Test getting top patterns
    bayesian_scorer.update(2, True, 7.0)
    bayesian_scorer.update(2, True, 4.0)
    bayesian_scorer.update(3, False, -3.0)
    bayesian_scorer.update(3, False, -2.0)
    
    top_patterns = bayesian_scorer.get_top_patterns(min_trades=2)
    print("Top patterns:", top_patterns)