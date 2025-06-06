# data_storage.py

import os
import logging
import pandas as pd
from datetime import datetime

DATA_DIR = "./snapshots"
os.makedirs(DATA_DIR, exist_ok=True)

class DataStorage:
    def __init__(self):
        self.snapshots = []

    def save_session_snapshot(self, capital: float, trades: int, wins: int, losses: int, pnl: float):
        try:
            date_str = datetime.now().strftime("%Y-%m-%d")
            snapshot = {
                "date": date_str,
                "capital": capital,
                "trades": trades,
                "wins": wins,
                "losses": losses,
                "pnl": pnl,
                "accuracy": round(wins / trades, 2) if trades > 0 else 0.0
            }
            self.snapshots.append(snapshot)
            df = pd.DataFrame(self.snapshots)
            filepath = os.path.join(DATA_DIR, f"session_{date_str}.csv")
            df.to_csv(filepath, index=False)
            logging.info(f"[STORAGE] Session snapshot saved: {filepath}")

        except Exception as e:
            logging.exception("[STORAGE ERROR] Failed to save session snapshot")


# Singleton instance
data_storage = DataStorage()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_storage.save_session_snapshot(
        capital=1234.56,
        trades=15,
        wins=10,
        losses=5,
        pnl=234.56
    )
