import logging
import threading
import time
import json
import os
from typing import Dict, Any, Optional, Tuple
from typing import List, Dict, Any
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/risk_validator.log"),
        logging.StreamHandler()
    ]
)

class RiskValidator:
    """
    Validates trades against risk management rules before execution.
    Designed to work with signals from the strategy_tournament.
    """
    def __init__(self, config_path: str = "config/risk_rules.json"):
        self.lock = threading.Lock()
        self.validation_stats = {
            "total_validated": 0,
            "passed": 0,
            "rejected": 0,
            "rejection_reasons": {}
        }
        
        # Load risk rules from config
        self.risk_rules = self._load_config(config_path)
        
        # Validation history for analytics
        self.validation_history = []
        self.max_history_size = 1000
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        logging.info("RiskValidator initialized with dynamic risk assessment")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load risk configuration from file or use defaults"""
        default_config = {
            "risk_reward_min": 1.0,
            "profiles": {
                "aggressive": {
                    "max_risk_pct": 0.15,
                    "max_notional_factor": 0.15,
                    "max_leverage": 20.0,
                    "max_position_pct": 0.15
                },
                "moderate": {
                    "max_risk_pct": 0.10,
                    "max_notional_factor": 0.10,
                    "max_leverage": 10.0,
                    "max_position_pct": 0.10
                },
                "conservative": {
                    "max_risk_pct": 0.05,
                    "max_notional_factor": 0.05,
                    "max_leverage": 5.0,
                    "max_position_pct": 0.05
                }
            },
            # Strategy-specific overrides
            "strategy_overrides": {
                "volatility_breakout": {"max_risk_pct": 0.12},
                "scalping_strategy": {"max_notional_factor": 0.20},
                "hft_strategy": {"max_risk_pct": 0.02}
            },
            # Trade type minimum confidence thresholds
            "trade_type_confidence": {
                "nano": 0.75,
                "sniper": 0.65,
                "swing": 0.60,
                "position": 0.55
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    logging.info(f"Loaded risk rules from {config_path}")
                    return config
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")
            
        # If we reach here, use default config
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        try:
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)
                logging.info(f"Created default risk rules at {config_path}")
        except Exception as e:
            logging.warning(f"Could not save default config to {config_path}: {e}")
            
        return default_config
        
    def shutdown(self):
        """Safely shut down the validator"""
        logging.info("Shutting down RiskValidator...")
        self._save_stats()
        logging.info("RiskValidator shutdown complete")
        
    def _save_stats(self):
        """Save validation statistics"""
        try:
            with self.lock:
                stats = self.validation_stats.copy()
                
            with open("logs/risk_validation_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
                
            # Also save a sample of validation history for analysis
            history_snapshot = self.validation_history[-100:] if len(self.validation_history) > 100 else self.validation_history
            with open("logs/validation_history_sample.json", "w") as f:
                json.dump(history_snapshot, f, indent=2)
                
            logging.info(f"Saved validation stats: {stats['passed']}/{stats['total_validated']} passed")
        except Exception as e:
            logging.error(f"Failed to save validation stats: {e}")

    def validate(self, trade_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced validation with strict checks that cannot be bypassed
        """
        with self.lock:
            # Update stats counter
            self.validation_stats["total_validated"] += 1
            
            try:
                # Extract trade parameters
                symbol = trade_dict.get("symbol", "UNKNOWN")
                strategy_name = trade_dict.get("strategy_name", "unknown")
                decision = trade_dict.get("decision", "")
                side = trade_dict.get("side", "")
                position_side = trade_dict.get("positionSide", side)
                entry = trade_dict.get("entry", 0.0)
                stop_loss = trade_dict.get("stopLoss", 0.0)
                take_profit = trade_dict.get("takeProfit", 0.0)
                confidence = trade_dict.get("confidence", 0.0)
                trade_type = trade_dict.get("trade_type", "swing")
                forced = trade_dict.get("forced", False)

                logging.info(f"RiskValidator: {symbol} {strategy_name}, entry={entry}, SL={stop_loss}, "
                             f"TP={take_profit}, confidence={confidence}")

                # STRICT VALIDATION - NO BYPASSING
                
                # 1. Required fields validation
                if not decision or decision != "trade":
                    return self._reject_trade("invalid_decision", symbol)
                
                if not side or side not in ["BUY", "SELL", "LONG", "SHORT"]:
                    return self._reject_trade("invalid_side", symbol)
                
                if not isinstance(entry, (int, float)) or entry <= 0:
                    return self._reject_trade("invalid_entry_price", symbol)
                
                if not isinstance(stop_loss, (int, float)) or stop_loss <= 0:
                    return self._reject_trade("invalid_stop_loss", symbol)
                    
                if not isinstance(take_profit, (int, float)) or take_profit <= 0:
                    return self._reject_trade("invalid_take_profit", symbol)
                
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                    return self._reject_trade("invalid_confidence", symbol)

                # 2. Direction-specific price validation
                if side.upper() in ["BUY", "LONG"]:
                    if stop_loss >= entry:
                        return self._reject_trade("stop_loss_above_entry_for_long", symbol)
                    if take_profit <= entry:
                        return self._reject_trade("take_profit_below_entry_for_long", symbol)
                    
                    risk = entry - stop_loss
                    reward = take_profit - entry
                    
                elif side.upper() in ["SELL", "SHORT"]:
                    if stop_loss <= entry:
                        return self._reject_trade("stop_loss_below_entry_for_short", symbol)
                    if take_profit >= entry:
                        return self._reject_trade("take_profit_above_entry_for_short", symbol)
                        
                    risk = stop_loss - entry
                    reward = entry - take_profit
                else:
                    return self._reject_trade("unrecognized_side", symbol)

                # 3. Risk-reward ratio validation
                if risk <= 0 or reward <= 0:
                    return self._reject_trade("invalid_risk_reward_calculation", symbol)
                    
                risk_reward_ratio = reward / risk
                min_rr_ratio = self.risk_rules.get("risk_reward_min", 1.0)
                
                # Slightly more lenient for high-confidence forced trades
                if forced and confidence > 0.7:
                    min_rr_ratio = max(0.8, min_rr_ratio - 0.2)
                
                if risk_reward_ratio < min_rr_ratio:
                    logging.warning(f"Poor risk-reward ratio: {risk_reward_ratio:.2f} < {min_rr_ratio}")
                    return self._reject_trade(f"poor_risk_reward_{risk_reward_ratio:.2f}", symbol)

                # 4. Confidence threshold validation
                trade_type_confidence = self.risk_rules.get("trade_type_confidence", {})
                min_confidence = trade_type_confidence.get(trade_type, 0.5)
                
                # More lenient for forced trades
                if forced:
                    min_confidence = max(0.3, min_confidence - 0.2)
                
                if confidence < min_confidence:
                    return self._reject_trade(f"low_confidence_{confidence:.2f}_for_{trade_type}", symbol)

                # 5. Risk percentage validation
                risk_pct = abs(entry - stop_loss) / entry
                
                # Get profile-specific risk limits
                risk_profile = trade_dict.get("risk_profile", "moderate")
                profile_rules = self.risk_rules.get("profiles", {}).get(risk_profile, {})
                strategy_overrides = self.risk_rules.get("strategy_overrides", {}).get(strategy_name, {})
                
                max_risk = strategy_overrides.get("max_risk_pct", profile_rules.get("max_risk_pct", 0.10))

                if risk_pct > max_risk:
                    logging.warning(f"Risk too high: {risk_pct:.2%} > {max_risk:.2%}")
                    return self._reject_trade(f"high_risk_pct_{risk_pct:.2%}", symbol)

                # 6. Leverage validation (if present)
                leverage = trade_dict.get("leverage", 1.0)
                max_leverage = profile_rules.get("max_leverage", 10.0)
                
                if leverage > max_leverage:
                    # Cap leverage instead of rejecting
                    trade_dict["leverage"] = max_leverage
                    logging.warning(f"Capped leverage from {leverage} to {max_leverage} for {symbol}")

                # 7. Position size validation (if present)
                notional_value = trade_dict.get("notional_value", 0)
                if notional_value > 0:
                    # Check if position is too large relative to account
                    account_balance = trade_dict.get("account_balance", 100.0)
                    position_pct = notional_value / account_balance
                    
                    max_position_pct = profile_rules.get("max_position_pct", 0.1)  # 10% max per position
                    
                    if position_pct > max_position_pct:
                        return self._reject_trade(f"position_too_large_{position_pct:.2%}", symbol)

                # VALIDATION PASSED
                self.validation_stats["passed"] += 1
                
                # Store validation in history
                validation_record = {
                    "timestamp": time.time(),
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "confidence": confidence,
                    "risk_reward": risk_reward_ratio,
                    "risk_pct": risk_pct,
                    "forced": forced,
                    "side": side,
                    "valid": True
                }
                self._add_to_history(validation_record)
                
                # Return enhanced trade dict with validation info
                validated_trade = trade_dict.copy()
                validated_trade.update({
                    "validation_passed": True,
                    "validation_time": time.time(),
                    "risk_profile": risk_profile,
                    "risk_reward_ratio": risk_reward_ratio,
                    "risk_percentage": risk_pct * 100,
                    "next_component": "capital_manager"
                })
                
                logging.info(f"✅ RiskValidator PASSED: {symbol} {side} - R:R {risk_reward_ratio:.2f}, Conf {confidence:.2f}")
                return {
                    "valid": True, 
                    "trade": validated_trade, 
                    "validation_time": time.time(),
                    "next": "capital_manager"
                }

            except Exception as e:
                logging.error(f"RiskValidator error for {trade_dict.get('symbol', 'UNKNOWN')}: {e}")
                return self._reject_trade(f"exception_{str(e)}", trade_dict.get('symbol', 'UNKNOWN'))

    def _reject_trade(self, reason: str, symbol: str = "UNKNOWN") -> Dict[str, Any]:
        """Enhanced rejection tracking with detailed logging"""
        with self.lock:
            self.validation_stats["rejected"] += 1
            if reason in self.validation_stats["rejection_reasons"]:
                self.validation_stats["rejection_reasons"][reason] += 1
            else:
                self.validation_stats["rejection_reasons"][reason] = 1
                
            # Store rejection in history
            rejection_record = {
                "timestamp": time.time(),
                "symbol": symbol,
                "reason": reason,
                "valid": False
            }
            self._add_to_history(rejection_record)
                
        logging.warning(f"❌ RiskValidator REJECTED: {symbol} - {reason}")
        return {
            "valid": False, 
            "reason": reason, 
            "validation_time": time.time(), 
            "symbol": symbol,
            "rejected_by": "risk_validator"
        }
    
    def _add_to_history(self, record: Dict[str, Any]):
        """Add validation record to history with size limiting"""
        self.validation_history.append(record)
        if len(self.validation_history) > self.max_history_size:
            self.validation_history = self.validation_history[-self.max_history_size:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current validation statistics"""
        with self.lock:
            return self.validation_stats.copy()
    
    def get_recent_validations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get most recent validation results"""
        with self.lock:
            return self.validation_history[-limit:]