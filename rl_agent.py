# rl_agent.py
import os
import numpy as np
import logging
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import random
from collections import deque
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/rl_agent.log"),
        logging.StreamHandler()
    ]
)

class TradingAgent:
    def __init__(self, state_size: int, action_size: int, batch_size: int = 64, memory_size: int = 10000):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Check TensorFlow availability first
        self.tf_available = self._check_tensorflow_availability()
        
        if self.tf_available:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model()
        else:
            logging.warning("TensorFlow not available, RL agent will use fallback mode")
            self.model = None
            self.target_model = None
    
    def _check_tensorflow_availability(self) -> bool:
        """Check if TensorFlow is available and working properly"""
        try:
            import tensorflow as tf
            # Test basic TensorFlow functionality
            test_tensor = tf.constant([1, 2, 3])
            _ = tf.reduce_mean(test_tensor)
            return True
        except ImportError:
            logging.warning("TensorFlow not installed")
            return False
        except Exception as e:
            logging.warning(f"TensorFlow available but has issues: {e}")
            return False
    
    def _build_model(self):
        """Neural Net for Deep-Q learning Model - FIXED VERSION"""
        if not self.tf_available:
            return None
            
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Clear any existing session
            tf.keras.backend.clear_session()
            
            model = keras.Sequential([
                keras.layers.Input(shape=(self.state_size,)),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.action_size, activation='linear')
            ])
            
            # FIXED: Use full function name instead of 'mse'
            model.compile(
                loss='mean_squared_error',  # Fixed: was 'mse'
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Failed to build model: {e}")
            self.tf_available = False
            return None
    
    def ensure_model_exists(self, filepath: str):
        """Create a default model if none exists"""
        if not self.tf_available:
            return False
            
        try:
            if not os.path.exists(filepath):
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Create and save default model
                model = self._build_model()
                if model is not None:
                    # Train on dummy data to initialize
                    dummy_x = np.random.random((32, self.state_size))
                    dummy_y = np.random.random((32, self.action_size))
                    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
                    
                    model.save(filepath, save_format='h5')
                    logging.info(f"Created default RL model at {filepath}")
                    return True
                
            return True
        except Exception as e:
            logging.error(f"Failed to create default model: {e}")
            return False
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        if self.model and self.target_model:
            self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training: bool = True):
        """Return action based on current state"""
        if not self.tf_available or self.model is None:
            # Fallback to random action
            return random.randrange(self.action_size)
            
        try:
            if training and np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            
            # Ensure state is properly shaped
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
                
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])
            
        except Exception as e:
            logging.error(f"Error in act method: {e}")
            return random.randrange(self.action_size)
    
    def replay(self, batch_size: int):
        """Train model on batch of experiences"""
        if not self.tf_available or self.model is None or len(self.memory) < batch_size:
            return
            
        try:
            minibatch = random.sample(self.memory, batch_size)
            
            states = np.array([e[0].flatten() for e in minibatch])
            targets = self.model.predict(states, verbose=0)
            
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                target = reward
                if not done:
                    next_state_reshaped = next_state.reshape(1, -1)
                    target = (reward + self.gamma * 
                              np.amax(self.target_model.predict(next_state_reshaped, verbose=0)[0]))
                targets[i][action] = target
                
            self.model.fit(states, targets, epochs=1, verbose=0)
                
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            logging.error(f"Error in replay method: {e}")
    
    def save(self, filepath: str):
        """Save model to disk"""
        if not self.tf_available or self.model is None:
            logging.warning("Cannot save model - TensorFlow not available or model not initialized")
            return False
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath, save_format='h5')
            logging.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            return False
    
    def load(self, filepath: str):
        """Load model with comprehensive error handling - FIXED VERSION"""
        if not self.tf_available:
            logging.warning("Cannot load model - TensorFlow not available")
            return False
            
        try:
            # Ensure model file exists
            if not self.ensure_model_exists(filepath):
                return False
                
            import tensorflow as tf
            from tensorflow import keras
            
            # Clear session
            tf.keras.backend.clear_session()
            
            # FIXED: Handle custom objects and loss function issues
            custom_objects = {
                'mse': tf.keras.losses.MeanSquaredError(),
                'mean_squared_error': tf.keras.losses.MeanSquaredError()
            }
            
            try:
                # Try loading with custom objects first
                self.model = keras.models.load_model(filepath, custom_objects=custom_objects)
            except Exception:
                try:
                    # Try loading without custom objects
                    self.model = keras.models.load_model(filepath)
                except Exception:
                    # Create new model if loading fails
                    logging.warning("Model loading failed, creating new model")
                    self.model = self._build_model()
                    if self.model is None:
                        return False
            
            # Create target model
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            
            # Recompile with current settings to avoid loss function issues
            self.model.compile(
                loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                metrics=['mae']
            )
            
            self.target_model.compile(
                loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                metrics=['mae']
            )
            
            logging.info(f"Successfully loaded RL model from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load model from {filepath}: {e}")
            
            # Final fallback - create new model
            try:
                self.model = self._build_model()
                self.target_model = self._build_model()
                if self.model and self.target_model:
                    self.update_target_model()
                    self.save(filepath)  # Save new model
                    logging.info("Created new model as fallback")
                    return True
            except Exception:
                pass
                
            return False

def create_state_from_bundle(bundle: Dict[str, Any]) -> np.ndarray:
    """Convert a data bundle to state vector for the RL agent - IMPROVED VERSION"""
    
    features = []
    
    try:
        # 1. Price data features (10 features)
        df = bundle.get("indicator_data")
        if df is not None and not df.empty and 'Close' in df.columns:
            close_prices = df['Close'].values
            # Remove NaN and infinite values
            close_prices = close_prices[np.isfinite(close_prices)]
            
            if len(close_prices) >= 10:
                recent_closes = close_prices[-10:]
                if recent_closes[0] > 0:
                    normalized_closes = recent_closes / recent_closes[0]
                    features.extend(normalized_closes.tolist())
                else:
                    features.extend([1.0] * 10)
            else:
                if len(close_prices) > 0 and close_prices[0] > 0:
                    normalized_closes = close_prices / close_prices[0]
                    padded = np.pad(normalized_closes, (0, 10 - len(normalized_closes)), 'constant', constant_values=1.0)
                    features.extend(padded.tolist())
                else:
                    features.extend([1.0] * 10)
        else:
            features.extend([1.0] * 10)
        
        # 2. RSI feature (1 feature)
        if df is not None and not df.empty:
            rsi_cols = [col for col in df.columns if 'RSI' in col]
            if rsi_cols and len(df[rsi_cols[0]]) > 0:
                rsi_value = df[rsi_cols[0]].iloc[-1]
                if np.isfinite(rsi_value):
                    features.append(np.clip(rsi_value / 100.0, 0, 1))
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # 3. ATR feature (1 feature)
        if df is not None and not df.empty:
            atr_cols = [col for col in df.columns if 'ATR' in col]
            if atr_cols and len(df[atr_cols[0]]) > 0 and 'Close' in df.columns:
                atr_value = df[atr_cols[0]].iloc[-1]
                close_value = df['Close'].iloc[-1]
                if np.isfinite(atr_value) and np.isfinite(close_value) and close_value > 0:
                    features.append(np.clip(atr_value / close_value, 0, 0.1))
                else:
                    features.append(0.02)
            else:
                features.append(0.02)
        else:
            features.append(0.02)
        
        # 4. Sentiment feature (1 feature)
        sentiment = bundle.get("sentiment", {}).get("sentiment_score", 0)
        features.append(np.clip((sentiment + 1) / 2, 0, 1))
        
        # 5. Whale activity feature (1 feature)
        whale_present = bundle.get("whale_flags", {}).get("whale_present", False)
        features.append(1.0 if whale_present else 0.0)
        
        # 6. Pattern confidence feature (1 feature)
        pattern_confidence = bundle.get("pattern_signal", {}).get("confidence", 0.5)
        features.append(np.clip(pattern_confidence, 0, 1))
        
        # 7. Forecast slope feature (1 feature)
        slope = bundle.get("forecast", {}).get("slope", 0)
        normalized_slope = np.clip((slope + 0.1) / 0.2, 0, 1)
        features.append(normalized_slope)
        
        # 8. Volume feature (1 feature) 
        if df is not None and not df.empty and 'Volume' in df.columns:
            volumes = df['Volume'].values
            volumes = volumes[np.isfinite(volumes)]
            if len(volumes) >= 2:
                current_vol = volumes[-1]
                avg_vol = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
                if avg_vol > 0:
                    vol_ratio = np.clip(current_vol / avg_vol, 0, 3) / 3.0
                    features.append(vol_ratio)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # 9. Market regime feature (1 feature)
        regime_type = bundle.get("regime_type", "unknown")
        regime_mapping = {
            "bull_trend": 0.8,
            "bear_trend": 0.2,
            "high_volatility": 0.6,
            "low_volatility": 0.4,
            "unknown": 0.5
        }
        features.append(regime_mapping.get(regime_type, 0.5))
        
        # 10. Pattern type feature (1 feature)
        pattern_type = bundle.get("pattern_signal", {}).get("type", "neutral")
        type_mapping = {
            "bullish": 0.8,
            "bearish": 0.2,
            "neutral": 0.5
        }
        features.append(type_mapping.get(pattern_type, 0.5))
        
        # Ensure exactly 20 features
        while len(features) < 20:
            features.append(0.5)
        features = features[:20]
        
        # Convert to numpy array and validate
        state_array = np.array(features, dtype=np.float32)
        
        # Check for invalid values
        if not np.all(np.isfinite(state_array)):
            logging.warning("Invalid values in state array, using neutral state")
            state_array = np.array([0.5] * 20, dtype=np.float32)
        
        return state_array.reshape(1, -1)
        
    except Exception as e:
        logging.error(f"Error creating state from bundle: {e}")
        neutral_state = np.array([0.5] * 20, dtype=np.float32)
        return neutral_state.reshape(1, -1)

def action_to_trade_signal(action: int, bundle: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """Convert RL agent action to a trade signal"""
    # Actions: 0=No trade, 1=Long tight SL, 2=Long wide SL, 3=Short tight SL, 4=Short wide SL
    
    if action == 0 or current_price <= 0:
        return {"decision": "no_trade"}
    
    # Handle long positions
    if action in [1, 2]:
        side = "BUY"
        position_side = "LONG"
        
        if action == 1:  # Tight stop loss
            stop_loss = current_price * 0.99
        else:  # Wide stop loss
            stop_loss = current_price * 0.97
            
        take_profit = current_price * (1 + 2 * (current_price - stop_loss) / current_price)
    
    # Handle short positions
    else:
        side = "SELL"
        position_side = "SHORT"
        
        if action == 3:  # Tight stop loss
            stop_loss = current_price * 1.01
        else:  # Wide stop loss
            stop_loss = current_price * 1.03
            
        take_profit = current_price * (1 - 2 * (stop_loss - current_price) / current_price)
    
    return {
        "decision": "trade",
        "entry": current_price,
        "stopLoss": round(stop_loss, 8),
        "takeProfit": round(take_profit, 8),
        "side": side,
        "positionSide": position_side,
        "confidence": 0.7,
        "strategy_name": "rl_agent",
        "strategy_type": "ai",
        "trade_type": "adaptive"
    }

def evaluate_with_rl_agent(bundle: Dict[str, Any], current_price: float, fallback_handler=None) -> Dict[str, Any]:
    """
    Wrapper function to safely evaluate trades using RL agent with proper fallback
    """
    try:
        # Create models directory
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, "rl_agent.h5")
        
        # Create agent
        agent = TradingAgent(state_size=20, action_size=5)
        
        # Check TensorFlow availability
        if not agent.tf_available:
            logging.debug("TensorFlow not available, using fallback")
            if fallback_handler:
                return fallback_handler(bundle, current_price)
            else:
                return {"decision": "no_trade", "reason": "tensorflow_unavailable"}
        
        # Load or create model
        if not agent.load(model_path):
            logging.debug("Model loading failed, using fallback")
            if fallback_handler:
                return fallback_handler(bundle, current_price)
            else:
                return {"decision": "no_trade", "reason": "model_load_failed"}
        
        # Create state and get action
        state = create_state_from_bundle(bundle)
        action = agent.act(state, training=False)
        
        # Convert to trade signal
        return action_to_trade_signal(action, bundle, current_price)
        
    except Exception as e:
        logging.error(f"RL agent evaluation failed: {e}")
        if fallback_handler:
            return fallback_handler(bundle, current_price)
        else:
            return {"decision": "no_trade", "reason": "rl_agent_error"}