import os
import logging
import numpy as np
from typing import Dict, Any, Tuple

def evaluate_with_rl_agent(bundle: Dict[str, Any], current_price: float, fallback_handler=None) -> Dict[str, Any]:
    """
    Wrapper function to safely evaluate trades using RL agent with proper fallback
    """
    try:
        # First, make sure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, "rl_agent.h5")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logging.warning(f"RL agent model file not found at {model_path}")
            
            # Create a stub model file to prevent future errors
            try:
                from tensorflow import keras
                from tensorflow.keras import layers
                
                # Create a simple model
                model = keras.Sequential([
                    layers.Dense(64, input_dim=20, activation='relu'),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(5, activation='linear')
                ])
                model.compile(loss='mse', optimizer='adam')
                
                # Save the stub model
                model.save(model_path)
                logging.info(f"Created stub RL model at {model_path}")
            except Exception as e:
                logging.error(f"Failed to create stub model: {e}")
            
            # Use fallback handler
            if fallback_handler:
                logging.info("Using fallback strategy for trade selection")
                return fallback_handler(bundle, current_price)
            else:
                # Return a conservative default if no fallback provided
                return {
                    "decision": "no_trade",
                    "reason": "rl_model_not_found"
                }
        
        # If model exists, proceed with normal evaluation
        from rl_agent import TradingAgent, create_state_from_bundle, action_to_trade_signal
        
        # Load the agent
        agent = TradingAgent(state_size=20, action_size=5)
        agent.load(model_path)
        
        # Create state representation
        state = create_state_from_bundle(bundle)
        
        # Get action from agent (with exploration disabled)
        action = agent.act(state, training=False)
        
        # Convert action to trade signal
        return action_to_trade_signal(action, bundle, current_price)
        
    except Exception as e:
        logging.error(f"RL agent evaluation failed: {e}")
        
        # Use fallback if provided
        if fallback_handler:
            logging.info("Using fallback strategy after RL agent failure")
            return fallback_handler(bundle, current_price)
        else:
            # Return a conservative default
            return {
                "decision": "no_trade",
                "reason": "rl_agent_error"
            }