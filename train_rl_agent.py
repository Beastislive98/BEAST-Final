import os
import numpy as np
from rl_agent import TradingAgent, create_state_from_bundle
from pipeline import get_latest_market_data, get_next_market_data

def get_initial_data_bundle():
    """
    Fetch the latest market data bundle at the start of an episode.
    """
    bundle = get_latest_market_data()
    if bundle is None:
        print("Warning: get_latest_market_data() returned None")
    return bundle

def get_next_data_bundle():
    """
    Fetch the next market data bundle after taking an action.
    """
    bundle = get_next_market_data()
    if bundle is None:
        print("Warning: get_next_market_data() returned None")
    return bundle

def compute_reward(action, state, next_state):
    """
    Reward based on price change following action.
    Assumes last feature in state array is price.
    """
    try:
        current_price = state[0][-1]
        next_price = next_state[0][-1]
        
        if action in [1, 2]:  # Long
            return next_price - current_price
        elif action in [3, 4]:  # Short
            return current_price - next_price
        else:  # No trade
            return 0.0
    except Exception as e:
        print(f"Error in compute_reward: {e}")
        return 0.0

def check_done_condition(step_count, max_steps=100):
    """
    End episode after max_steps
    """
    return step_count >= max_steps

def train_rl_agent(episodes=1000, max_steps_per_episode=100):
    state_size = 20  # Must match create_state_from_bundle output length
    action_size = 5  # As defined in rl_agent.py action space
    
    agent = TradingAgent(state_size, action_size)
    
    for episode in range(episodes):
        step_count = 0
        total_reward = 0.0
        
        bundle = get_initial_data_bundle()
        if bundle is None:
            print("No initial data available, stopping training.")
            break
        
        state = create_state_from_bundle(bundle)
        done = False
        
        while not done:
            action = agent.act(state)
            
            next_bundle = get_next_data_bundle()
            if next_bundle is None:
                print("No next data available, ending episode.")
                break
            
            next_state = create_state_from_bundle(next_bundle)
            reward = compute_reward(action, state, next_state)
            
            step_count += 1
            done = check_done_condition(step_count, max_steps_per_episode)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay(agent.batch_size)
        
        if episode % 10 == 0:
            agent.update_target_model()
            print(f"Episode {episode + 1}/{episodes} - Total reward: {total_reward:.2f}")
    
    os.makedirs("models", exist_ok=True)
    agent.save("models/rl_agent.h5")
    print("Training complete. Model saved to 'models/rl_agent.h5'")

if __name__ == "__main__":
    train_rl_agent()
