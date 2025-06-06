# whale_detector.py

import logging
from typing import Dict, List, Union


def _extract_volume_levels(depth: List[List[Union[str, float]]], levels: int = 10) -> List[float]:
    """
    Extracts volume from order book depth entries safely.

    Args:
        depth (List): List of [price, volume] pairs.
        levels (int): Number of top levels to check.

    Returns:
        List of float volumes.
    """
    try:
        return [float(level[1]) for level in depth[:levels] if len(level) == 2]
    except (ValueError, IndexError, TypeError) as e:
        logging.warning(f"Failed to extract volume levels from depth data: {e}")
        return []


def detect_whale_activity(order_book: Dict, volume_threshold: float = 5.0) -> Dict[str, Union[bool, List[float]]]:
    """
    Detects large volume walls ("whale" activity) in bid/ask order book data.

    Args:
        order_book (Dict): Order book data containing 'bid_depth' and 'ask_depth'.
        volume_threshold (float): Volume value to define a "whale".

    Returns:
        Dict[str, Union[bool, List[float]]]: Flags and volumes indicating whale presence.
    """
    try:
        bids = order_book.get("bid_depth", [])
        asks = order_book.get("ask_depth", [])

        bid_volumes = _extract_volume_levels(bids)
        ask_volumes = _extract_volume_levels(asks)

        large_bid = any(vol > volume_threshold for vol in bid_volumes)
        large_ask = any(vol > volume_threshold for vol in ask_volumes)

        result = {
            "large_bid_wall": large_bid,
            "large_ask_wall": large_ask,
            "whale_present": large_bid or large_ask,
            "bid_volumes_top10": bid_volumes,
            "ask_volumes_top10": ask_volumes
        }

        logging.info(f"Whale detection result: {result}")
        return result

    except Exception as e:
        logging.exception(f"Error during whale detection: {e}")
        return {
            "large_bid_wall": False,
            "large_ask_wall": False,
            "whale_present": False,
            "bid_volumes_top10": [],
            "ask_volumes_top10": []
        }


if __name__ == "__main__":
    import sys
    from binance_interface import get_market_data

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    symbol = "BTCUSDT"
    market_data = get_market_data(symbol)

    if market_data:
        whale_result = detect_whale_activity(market_data)
        print(f"Whale Detection for {symbol}:\n", whale_result)
    else:
        print(f"No data available for {symbol}.")
