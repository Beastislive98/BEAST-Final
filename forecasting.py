import logging
import pandas as pd
from typing import Optional, Dict
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

def forecast_price_direction(df: pd.DataFrame, forecast_horizon: int = 3) -> Optional[Dict[str, float]]:
    """
    Forecasts short-term price direction using XGBoost regression on technical indicators.

    Args:
        df (pd.DataFrame): DataFrame including 'Close' and technical indicators as features.
        forecast_horizon (int): Number of future steps to predict (used to target next-step price).

    Returns:
        Dict[str, float]: Predicted price and direction slope.
    """
    try:
        if df is None or 'Close' not in df.columns or len(df) < forecast_horizon + 10:
            logging.warning("Insufficient data for XGBoost forecasting.")
            return None

        df = df.dropna().copy()

        # Target: future close price
        df['target'] = df['Close'].shift(-forecast_horizon)
        df.dropna(inplace=True)

        # Features: drop non-indicator columns
        drop_cols = ['Open', 'High', 'Low', 'Volume', 'target']
        features = [col for col in df.columns if col not in drop_cols and df[col].dtype.kind == 'f']

        X = df[features].values
        y = df['target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBRegressor(n_estimators=50, max_depth=3, random_state=42, verbosity=0)
        model.fit(X_scaled[:-1], y[:-1])  # train on all but last

        predicted = model.predict(X_scaled[-1].reshape(1, -1))[0]
        current_price = df['Close'].iloc[-1]
        slope = predicted - current_price

        # ⛔ Reject obviously broken predictions
        if predicted <= 0:
            logging.warning(f"Rejected forecast: predicted price ≤ 0 | Current: {current_price}, Forecast: {predicted}")
            return None

        # ⚠ Warn if predicted price is way off (e.g. >±99%)
        pct_dev = abs((predicted - current_price) / current_price)
        if pct_dev > 0.99:
            logging.warning(f"Forecast outlier for model sanity: {predicted} vs {current_price} ({pct_dev:.2%})")

        result = {
            "forecast_price": round(predicted, 4),
            "slope": round(slope, 6)
        }

        logging.info(f"XGBoost forecast result: {result}")
        return result

    except Exception as e:
        logging.exception(f"XGBoost forecasting error: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import yfinance as yf

    df = yf.download("BTC-USD", period="1d", interval="1m")
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
    from indicator_universe import calculate_indicators, INDICATOR_PARAMS

    df = calculate_indicators(df, INDICATOR_PARAMS)
    print(forecast_price_direction(df))
