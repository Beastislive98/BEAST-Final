# visual_transformer.py

import numpy as np
import pandas as pd
import cv2
import logging
from typing import Optional


def chart_to_image(df: pd.DataFrame, image_size: int = 64) -> Optional[np.ndarray]:
    """
    Converts OHLC data into a grayscale image representing the price pattern.

    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close'
        image_size (int): Size of output image (e.g., 64x64)

    Returns:
        np.ndarray: Grayscale image matrix
    """
    try:
        if df is None or len(df) < 10 or not set(['Open', 'High', 'Low', 'Close']).issubset(df.columns):
            logging.warning("Insufficient or invalid data for visual transformation.")
            return None

        # Normalize prices for plotting
        prices = df[['Open', 'High', 'Low', 'Close']].copy()
        normalized = (prices - prices.min()) / (prices.max() - prices.min())

        # Create a blank white canvas
        img = np.ones((image_size, image_size), dtype=np.uint8) * 255

        # Plot lines (e.g., close prices)
        points = np.linspace(0, image_size - 1, len(normalized)).astype(int)
        close_scaled = (1.0 - normalized['Close']) * (image_size - 1)

        for i in range(len(points) - 1):
            pt1 = (points[i], int(close_scaled.iloc[i]))
            pt2 = (points[i + 1], int(close_scaled.iloc[i + 1]))
            cv2.line(img, pt1, pt2, color=0, thickness=1)

        return img

    except Exception as e:
        logging.exception(f"Visual transformation failed: {e}")
        return None


if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.INFO)
    df = yf.download("BTC-USD", period="1d", interval="1m")
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close"})
    img = chart_to_image(df.tail(64))

    if img is not None:
        plt.imshow(img, cmap="gray")
        plt.title("Visual Transformer Output")
        plt.axis("off")
        plt.show()
