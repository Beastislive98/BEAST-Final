import logging
from typing import Dict, Any, Optional, List
import uuid
import time
import threading
import atexit

# Ensure shared logging configuration with CapitalManager
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/trading_system.log"),
            logging.StreamHandler()
        ]
    )

class CapitalAllocator:
    """
    Allocates capital for trading signals based on risk management rules.
    Always fetches current capital from capital_manager instead of using static values.
    """
    def __init__(self, capital_manager=None):
        self.capital_manager = capital_manager
        self.lock = threading.Lock()  # Thread safety for allocations
        
        # Check dependencies
        self._check_dependencies()
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        logging.info(f"CapitalAllocator initialized with dynamic capital tracking")
    
    def _check_dependencies(self):
        """Check if optional dependencies are available"""
        # Check for adaptive_sizing module
        try:
            from adaptive_sizing import AdaptivePositionSizer
            logging.info("AdaptivePositionSizer module is available")
        except ImportError:
            logging.warning("AdaptivePositionSizer module not found, will use fallback sizing")
            
        # Check for pattern_recognition module
        try:
            from pattern_recognition import assess_moneyness
            logging.info("Pattern recognition module is available")
        except ImportError:
            logging.warning("Pattern recognition module not found, will use simple moneyness assessment")
    
    def shutdown(self):
        """Clean shutdown of the allocator"""
        logging.info("Shutting down CapitalAllocator...")
        # Any cleanup tasks would go here
        logging.info("CapitalAllocator shutdown complete")
        
    def get_current_capital(self) -> float:
        """
        Get current capital from the capital manager
        """
        if self.capital_manager:
            return self.capital_manager.get_current_balance()
        # Default fallback if capital manager not available
        return 32.0
        
    def allocate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate capital for a trade signal based on risk parameters.
        Always retrieves current capital for each allocation.
        """
        # Use a lock to ensure thread safety
        with self.lock:
            try:
                # Generate unique trade ID for tracking
                trade_id = f"trade_{uuid.uuid4().hex[:8]}_{int(time.time())}"
                
                symbol = signal.get("symbol", "UNKNOWN")
                entry_price = signal.get("entry", 0.0)
                
                # Handle missing or invalid entry price
                if entry_price <= 0:
                    logging.warning(f"Invalid entry price for {symbol}: {entry_price}")
                    return self._create_fallback_allocation(signal, trade_id=trade_id)
                
                # Dynamically get current capital from capital manager
                self.current_capital = self.get_current_capital()
                logging.info(f"Current capital for allocation: ${self.current_capital:.2f}")
                
                # Get the signal parameters
                stop_loss = signal.get("stopLoss", 0.0)
                confidence = signal.get("confidence", 0.7)
                
                # Try using the adaptive position sizer
                try:
                    from adaptive_sizing import AdaptivePositionSizer
                    sizer = AdaptivePositionSizer()
                    
                    position_info = sizer.calculate_position_size(
                        account_balance=self.current_capital,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        market_data=signal.get("market_data", {}),
                        confidence=confidence
                    )
                    
                    # Create allocation result with adaptive sizing
                    result = signal.copy()
                    result.update({
                        "trade_id": trade_id,
                        "quantity": position_info["position_size_rounded"],
                        "risk_amount": position_info["risk_amount"],
                        "risk_percent": position_info["risk_pct"] * 100,
                        "capital_allocated": position_info["notional_value"],
                        "capital_used": position_info["notional_value"],
                        "leverage": position_info["leverage"],
                        "timestamp": int(time.time() * 1000),
                        "side": position_info.get("side", result.get("side")),
                        "positionSide": position_info.get("positionSide", result.get("positionSide"))
                    })
                    
                    # Preserve risk profile if in original signal
                    if "risk_profile" in signal:
                        result["risk_profile"] = signal["risk_profile"]
                    else:
                        # Get risk profile from capital manager or use defaults
                        if self.capital_manager:
                            risk_params = self.capital_manager.get_risk_parameters()
                            result["risk_profile"] = risk_params["risk_profile"]
                        else:
                            result["risk_profile"] = "moderate"
                    
                    logging.info(f"Used adaptive position sizer for {symbol}: {position_info['position_size_rounded']} units (${position_info['notional_value']:.2f})")
                    
                    # Assess moneyness using dedicated method
                    result = self._assess_position_moneyness(result)
                    
                    # CRITICAL FIX: Validate the allocation against capital limits
                    if self.capital_manager:
                        validation_result = self.capital_manager.validate(result)
                        if not validation_result.get("valid", False) and not validation_result.get("queued", False):
                            logging.warning(f"Allocation rejected by capital manager: {validation_result.get('reason')}")
                            # Create a rejection response
                            result["rejected"] = True
                            result["rejection_reason"] = validation_result.get("reason")
                            return result
                        elif validation_result.get("queued", False):
                            # Trade was queued
                            result["queued"] = True
                            result["queue_reason"] = validation_result.get("reason")
                            return result
                        elif validation_result.get("scaled", False):
                            # Trade was scaled down
                            return validation_result.get("trade", result)
                        else:
                            # Trade was accepted as-is
                            return validation_result.get("trade", result)
                    
                    return result
                    
                except (ImportError, Exception) as e:
                    logging.warning(f"Adaptive position sizing failed for {symbol}: {e}, falling back to original method")
                    # Fall back to original implementation if adaptive sizer fails
                
                # Get risk profile from capital manager or use defaults
                if self.capital_manager:
                    risk_params = self.capital_manager.get_risk_parameters()
                    risk_per_trade = risk_params["risk_per_trade"]
                    max_position_pct = risk_params["position_sizing"]
                    max_leverage = risk_params["max_leverage"]
                    risk_profile = risk_params["risk_profile"]
                else:
                    # Default values if capital manager not available
                    risk_per_trade = 0.02  # 2% risk per trade
                    max_position_pct = 0.15  # Max 15% of capital per trade
                    max_leverage = 10      # Max 10x leverage
                    risk_profile = "moderate"
                
                # Handle missing stop loss - we need this for proper risk management
                stop_loss = signal.get("stopLoss")
                if not stop_loss:
                    # Try to calculate a conservative stop loss based on pattern type if available
                    pattern = signal.get("pattern", "")
                    side = signal.get("side", "").upper()
                    
                    if side == "BUY" or side == "LONG":
                        # For long positions, set a 2% stop loss if not provided
                        stop_loss = entry_price * 0.98
                        signal["stopLoss"] = stop_loss
                        logging.warning(f"Added automatic stop loss for {symbol} at {stop_loss:.4f} (2% below entry)")
                    elif side == "SELL" or side == "SHORT":
                        # For short positions, set a 2% stop loss if not provided
                        stop_loss = entry_price * 1.02
                        signal["stopLoss"] = stop_loss
                        logging.warning(f"Added automatic stop loss for {symbol} at {stop_loss:.4f} (2% above entry)")
                    else:
                        logging.warning(f"Cannot allocate for {symbol}: Missing side and stop loss")
                        return self._create_fallback_allocation(signal, trade_id=trade_id)
                
                # Calculate risk per trade in currency terms
                risk_amount = self.current_capital * risk_per_trade
                
                # Calculate dynamic max position value based on account size and risk profile
                max_position_value = self.current_capital * max_position_pct
                
                # Calculate risk in price terms (difference between entry and stop-loss)
                price_risk = abs(entry_price - stop_loss)
                risk_percent = price_risk / entry_price
                
                # Calculate position size based on risk
                if price_risk > 0:
                    position_size = risk_amount / price_risk
                else:
                    position_size = 0
                    
                # Apply position limits based on capital
                max_position = max_position_value / entry_price
                position_size = min(position_size, max_position)
                
                # Calculate notional value
                notional_value = position_size * entry_price
                
                # Make sure we have a minimum notional value based on account size
                min_notional = max(3.0, self.current_capital * 0.01)  # Minimum $3 or 1% of capital
                if notional_value < min_notional:
                    # Adjust position size to meet minimum notional
                    position_size = min_notional / entry_price
                    notional_value = position_size * entry_price
                
                # Calculate leverage based on confidence and strategy
                confidence = signal.get("confidence", 0.7)
                trade_type = signal.get("trade_type", "swing")
                
                # Adjust max leverage based on trade type
                if trade_type == "nano":
                    allowed_leverage = min(20, max_leverage + 5)  # HFT can use more leverage
                elif trade_type == "sniper":
                    allowed_leverage = min(15, max_leverage + 3)  # Event-based can use more leverage
                else:
                    allowed_leverage = max_leverage
                    
                # Calculate actual leverage - scales with confidence
                confidence_factor = max(0.5, min(1.0, confidence))
                leverage = max(1, min(allowed_leverage, round(confidence_factor * allowed_leverage)))
                
                # Round position size based on asset precision (adjusted for specific assets)
                position_size = self._round_position_size(position_size, entry_price)
                
                # Ensure minimum position size
                min_position_size = 0.001  # Smallest possible position
                position_size = max(position_size, min_position_size)
                
                # Update notional value after rounding
                notional_value = position_size * entry_price
                
                # Create allocation result
                result = signal.copy()
                result.update({
                    "trade_id": trade_id,
                    "quantity": position_size,
                    "leverage": leverage,
                    "notional_value": notional_value,
                    "risk_amount": risk_amount,
                    "risk_percent": risk_percent * 100,
                    "capital_allocated": notional_value,
                    "capital_used": notional_value,
                    "timestamp": int(time.time() * 1000),
                    "risk_profile": risk_profile
                })
                
                # Assess moneyness using dedicated method
                result = self._assess_position_moneyness(result)
                
                logging.info(f"Allocated {position_size} {symbol} (${notional_value:.2f}) with {leverage}x leverage based on ${risk_amount:.2f} risk")
                
                # CRITICAL FIX: Validate the allocation against capital limits
                if self.capital_manager:
                    validation_result = self.capital_manager.validate(result)
                    if not validation_result.get("valid", False) and not validation_result.get("queued", False):
                        logging.warning(f"Allocation rejected by capital manager: {validation_result.get('reason')}")
                        # Create a rejection response
                        result["rejected"] = True
                        result["rejection_reason"] = validation_result.get("reason")
                        return result
                    elif validation_result.get("queued", False):
                        # Trade was queued
                        result["queued"] = True
                        result["queue_reason"] = validation_result.get("reason")
                        return result
                    elif validation_result.get("scaled", False):
                        # Trade was scaled down
                        return validation_result.get("trade", result)
                    else:
                        # Trade was accepted as-is
                        return validation_result.get("trade", result)
                
                return result
                
            except Exception as e:
                logging.error(f"Allocation error for {signal.get('symbol', 'UNKNOWN')}: {e}")
                return self._create_fallback_allocation(signal, valid=False, trade_id=trade_id)
    
    def _round_position_size(self, position_size: float, entry_price: float) -> float:
        """Round position size based on asset price range"""
        if entry_price > 10000:  # BTC-like
            return round(position_size, 3)
        elif entry_price > 1000:  # ETH-like
            return round(position_size, 3)
        elif entry_price > 100:  # Mid-priced assets
            return round(position_size, 2)
        elif entry_price > 10:  # Lower-priced assets
            return round(position_size, 1)
        elif entry_price > 1:  # Very low-priced assets
            return round(position_size, 0)
        else:  # Ultra-low priced
            return round(position_size, 0)
                
    def _assess_position_moneyness(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if a position is in-the-money, at-the-money, or out-of-the-money"""
        entry_price = result.get("entry", 0)
        key_level = result.get("key_level", 0)
        key_levels = result.get("key_levels", [])
        side = result.get("side", "").upper()
        direction = "long" if side == "BUY" else "short"
        
        # Use key_levels for more accurate moneyness assessment if available
        if key_levels and len(key_levels) > 0:
            try:
                from pattern_recognition import assess_moneyness
                moneyness = assess_moneyness(entry_price, key_levels, direction)
                result["moneyness"] = moneyness
            except Exception as e:
                logging.warning(f"Failed to assess moneyness using key_levels: {e}")
                # Fall back to single key level
                result = self._assess_moneyness_simple(result, entry_price, key_level, direction)
        elif key_level > 0:
            # Fallback to single key level if available
            result = self._assess_moneyness_simple(result, entry_price, key_level, direction)
            
        return result
        
    def _assess_moneyness_simple(self, result: Dict[str, Any], entry_price: float, key_level: float, direction: str) -> Dict[str, Any]:
        """Simple moneyness assessment using a single key level"""
        if direction == "long":
            if entry_price > key_level * 1.01:  # 1% above
                result["moneyness"] = "ITM"
            elif entry_price < key_level * 0.99:  # 1% below
                result["moneyness"] = "OTM"
            else:
                result["moneyness"] = "ATM"
        else:  # short
            if entry_price < key_level * 0.99:  # 1% below
                result["moneyness"] = "ITM"
            elif entry_price > key_level * 1.01:  # 1% above
                result["moneyness"] = "OTM"
            else:
                result["moneyness"] = "ATM"
                
        return result
                
    def _create_fallback_allocation(self, signal: Dict[str, Any], valid: bool = True, trade_id: str = None) -> Dict[str, Any]:
        """
        Create a fallback allocation when normal allocation fails.
        """
        if trade_id is None:
            trade_id = f"trade_{uuid.uuid4().hex[:8]}_{int(time.time())}"
                
        symbol = signal.get("symbol", "UNKNOWN")
        entry_price = signal.get("entry", 100.0)  # Use default if not available
        
        # Always get current capital to ensure accurate allocation
        current_capital = self.get_current_capital()
        
        # For a small account, use very conservative fallback values
        if entry_price > 10000:  # BTC-like
            position_size = 0.001
        elif entry_price > 1000:  # ETH-like
            position_size = 0.002
        elif entry_price > 100:
            position_size = 0.01
        elif entry_price > 10:
            position_size = 0.1
        elif entry_price > 1:
            position_size = 1.0
        else:
            position_size = 5.0
                
        # Cap notional value at 5% of capital for fallbacks
        notional_value = position_size * entry_price
        max_notional = current_capital * 0.05
        
        if notional_value > max_notional and max_notional > 0:
            position_size = max_notional / entry_price
            notional_value = position_size * entry_price
        
        result = signal.copy()
        result.update({
            "trade_id": trade_id,
            "quantity": position_size,
            "leverage": 1,  # Conservative leverage for fallback
            "notional_value": notional_value,
            "risk_amount": notional_value * 0.01,  # Assume 1% risk
            "risk_percent": 1.0,
            "capital_allocated": notional_value,
            "capital_used": notional_value,
            "timestamp": int(time.time() * 1000),
            "risk_profile": "conservative"  # Conservative profile for fallbacks
        })
        
        if valid:
            logging.info(f"Fallback allocation: {position_size} {symbol} (${notional_value:.2f})")
        else:
            logging.warning(f"Invalid fallback allocation for {symbol}")
            
        # CRITICAL FIX: Validate fallback allocation against capital limits too
        if self.capital_manager and valid:
            validation_result = self.capital_manager.validate(result)
            if not validation_result.get("valid", False) and not validation_result.get("queued", False):
                logging.warning(f"Fallback allocation rejected by capital manager: {validation_result.get('reason')}")
                # Create a rejection response
                result["rejected"] = True
                result["rejection_reason"] = validation_result.get("reason")
                return result
            elif validation_result.get("queued", False):
                # Trade was queued
                result["queued"] = True
                result["queue_reason"] = validation_result.get("reason")
                return result
            elif validation_result.get("scaled", False):
                # Trade was scaled down
                return validation_result.get("trade", result)
            else:
                # If valid, return the validated allocation
                return validation_result.get("trade", result)
                
        return result