# market_regime.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Dict, Any, List, Tuple

class MarketRegimeDetector:
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42)
        self.regime_properties = {}
        self.is_fitted = False
        
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract regime-indicative features from price data"""
        # Price action features
        volatility = df['High'].values - df['Low'].values
        returns = df['Close'].pct_change().values
        
        # Trend features
        sma_20 = df['Close'].rolling(window=20).mean().values
        sma_50 = df['Close'].rolling(window=50).mean().values
        trend = (sma_20 - sma_50) / sma_50
        
        # Momentum features
        rsi = compute_rsi(df['Close'].values, 14)
        
        # Volume features
        volume_ma = df['Volume'].rolling(window=20).mean().values
        vol_change = df['Volume'] / volume_ma
        
        features = np.column_stack((
            volatility[-30:], 
            returns[-30:],
            trend[-30:],
            rsi[-30:],
            vol_change[-30:]
        ))
        
        return np.nan_to_num(features)
    
    def fit(self, market_data_dict: Dict[str, pd.DataFrame]):
        """Fit the regime detection model on historical data"""
        all_features = []
        
        for symbol, df in market_data_dict.items():
            if len(df) > 100:  # Ensure enough data
                features = self.extract_features(df)
                all_features.append(features)
        
        # Combine all features and fit KMeans
        combined_features = np.vstack(all_features)
        self.model.fit(combined_features)
        
        # Characterize the regimes
        self._characterize_regimes(combined_features)
        self.is_fitted = True
        
    def _characterize_regimes(self, features: np.ndarray):
        """Determine the characteristics of each regime"""
        labels = self.model.predict(features)
        
        for i in range(self.n_regimes):
            regime_features = features[labels == i]
            
            # Calculate regime properties
            volatility = np.mean(regime_features[:, 0])
            returns = np.mean(regime_features[:, 1])
            trend = np.mean(regime_features[:, 2])
            
            # Classify the regime
            if trend > 0.01 and volatility < np.median(features[:, 0]):
                regime_type = "bull_trend"
            elif trend < -0.01 and volatility < np.median(features[:, 0]):
                regime_type = "bear_trend"
            elif volatility > np.percentile(features[:, 0], 75):
                regime_type = "high_volatility"
            else:
                regime_type = "ranging"
                
            self.regime_properties[i] = {
                "type": regime_type,
                "avg_volatility": float(volatility),
                "avg_returns": float(returns),
                "avg_trend": float(trend)
            }
    
    def predict_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict the current market regime from recent data"""
        if not self.is_fitted:
            return {"regime": "unknown", "confidence": 0.0}
            
        features = self.extract_features(df)
        regime_id = int(self.model.predict(features[-1].reshape(1, -1))[0])
        
        # Calculate confidence based on distance to cluster center
        distances = self.model.transform(features[-1].reshape(1, -1))[0]
        total_distance = np.sum(distances)
        if total_distance > 0:
            confidence = 1.0 - (distances[regime_id] / total_distance)
        else:
            confidence = 1.0
            
        return {
            "regime_id": regime_id,
            "regime_type": self.regime_properties[regime_id]["type"],
            "confidence": float(confidence),
            "properties": self.regime_properties[regime_id]
        }

def compute_rsi(prices, window=14):
    """Compute RSI indicator"""
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down if down != 0 else np.inf
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta
            
        up = (up * (window-1) + upval) / window
        down = (down * (window-1) + downval) / window
        rs = up/down if down != 0 else np.inf
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi