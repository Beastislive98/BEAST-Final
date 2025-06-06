# pattern_registry.py

from pattern_rules import *

# Registry format:
# "pattern_name": {
#    "type": "candlestick" or "chart" (pattern type),
#    "window": int (number of candles needed),
#    "rule": function (detection function),
#    "confidence": float (base confidence for pattern)
# }

PATTERN_REGISTRY = {
    # === Single-Candle Patterns ===
    "hammer": {"type": "candlestick", "window": 1, "rule": is_hammer, "confidence": 0.9},
    "inverted_hammer": {"type": "candlestick", "window": 1, "rule": is_inverted_hammer, "confidence": 0.9},
    "bullish_marubozu": {"type": "candlestick", "window": 1, "rule": is_bullish_marubozu, "confidence": 0.9},
    "bearish_marubozu": {"type": "candlestick", "window": 1, "rule": is_bearish_marubozu, "confidence": 0.9},
    "doji": {"type": "candlestick", "window": 1, "rule": is_doji, "confidence": 0.85},
    "spinning_top": {"type": "candlestick", "window": 1, "rule": is_spinning_top, "confidence": 0.85},
    "hanging_man": {"type": "candlestick", "window": 1, "rule": is_hanging_man, "confidence": 0.9},
    "shooting_star": {"type": "candlestick", "window": 1, "rule": is_shooting_star, "confidence": 0.9},

    # === Two-Candle Patterns ===
    "bullish_engulfing": {"type": "candlestick", "window": 2, "rule": is_bullish_engulfing, "confidence": 0.95},
    "bearish_engulfing": {"type": "candlestick", "window": 2, "rule": is_bearish_engulfing, "confidence": 0.95},
    "bullish_harami": {"type": "candlestick", "window": 2, "rule": is_bullish_harami, "confidence": 0.9},
    "bearish_harami": {"type": "candlestick", "window": 2, "rule": is_bearish_harami, "confidence": 0.9},
    "piercing_line": {"type": "candlestick", "window": 2, "rule": is_piercing_line, "confidence": 0.9},
    "dark_cloud_cover": {"type": "candlestick", "window": 2, "rule": is_dark_cloud_cover, "confidence": 0.9},
    "key_reversal": {"type": "chart", "window": 2, "rule": is_key_reversal, "confidence": 0.85},

    # === Three-Candle Patterns ===
    "morning_star": {"type": "candlestick", "window": 3, "rule": is_morning_star, "confidence": 0.92},
    "evening_star": {"type": "candlestick", "window": 3, "rule": is_evening_star, "confidence": 0.92},
    "three_white_soldiers": {"type": "candlestick", "window": 3, "rule": is_three_white_soldiers, "confidence": 0.93},
    "three_black_crows": {"type": "candlestick", "window": 3, "rule": is_three_black_crows, "confidence": 0.93},
    "three_inside_up": {"type": "candlestick", "window": 3, "rule": is_three_inside_up, "confidence": 0.9},
    "three_inside_down": {"type": "candlestick", "window": 3, "rule": is_three_inside_down, "confidence": 0.9},
    "three_outside_up": {"type": "candlestick", "window": 3, "rule": is_three_outside_up, "confidence": 0.9},
    "three_outside_down": {"type": "candlestick", "window": 3, "rule": is_three_outside_down, "confidence": 0.9},
    "abandoned_baby_bullish": {"type": "candlestick", "window": 3, "rule": is_abandoned_baby_bullish, "confidence": 0.94},
    "abandoned_baby_bearish": {"type": "candlestick", "window": 3, "rule": is_abandoned_baby_bearish, "confidence": 0.94},
    "rising_three_methods": {"type": "candlestick", "window": 3, "rule": is_rising_three_methods, "confidence": 0.9},
    "falling_three_methods": {"type": "candlestick", "window": 3, "rule": is_falling_three_methods, "confidence": 0.9},

    # === Reversal Chart Patterns ===
    "head_and_shoulders": {"type": "chart", "window": 20, "rule": is_head_and_shoulders, "confidence": 0.87},
    "inverse_head_and_shoulders": {"type": "chart", "window": 20, "rule": is_inverse_head_and_shoulders, "confidence": 0.87},
    "double_top": {"type": "chart", "window": 15, "rule": is_double_top, "confidence": 0.86},
    "double_bottom": {"type": "chart", "window": 15, "rule": is_double_bottom, "confidence": 0.86},
    "rounding_top": {"type": "chart", "window": 20, "rule": is_rounding_top, "confidence": 0.87},
    "rounding_bottom": {"type": "chart", "window": 20, "rule": is_rounding_bottom, "confidence": 0.87},
    "v_bottom": {"type": "chart", "window": 10, "rule": is_v_bottom, "confidence": 0.86},
    "inverted_v_top": {"type": "chart", "window": 10, "rule": is_inverted_v_top, "confidence": 0.86},
    "island_reversal": {"type": "chart", "window": 5, "rule": is_island_reversal, "confidence": 0.9},
    "diamond_top": {"type": "chart", "window": 15, "rule": is_diamond_top, "confidence": 0.88},
    "diamond_bottom": {"type": "chart", "window": 15, "rule": is_diamond_bottom, "confidence": 0.88},
    "bump_and_run": {"type": "chart", "window": 20, "rule": is_bump_and_run, "confidence": 0.88},

    # === Continuation Chart Patterns ===
    "ascending_channel": {"type": "chart", "window": 12, "rule": is_ascending_channel, "confidence": 0.85},
    "descending_channel": {"type": "chart", "window": 12, "rule": is_descending_channel, "confidence": 0.85},
    "horizontal_channel": {"type": "chart", "window": 12, "rule": is_horizontal_channel, "confidence": 0.85},
    "ascending_staircase": {"type": "chart", "window": 12, "rule": is_ascending_staircase, "confidence": 0.85},
    "descending_staircase": {"type": "chart", "window": 12, "rule": is_descending_staircase, "confidence": 0.85},
    "scallop": {"type": "chart", "window": 15, "rule": is_scallop_pattern, "confidence": 0.87},
    "coil": {"type": "chart", "window": 10, "rule": is_coil_pattern, "confidence": 0.84},

    # === Harmonic Patterns ===
    "gartley": {"type": "chart", "window": 20, "rule": is_gartley_pattern, "confidence": 0.9},
    "bat": {"type": "chart", "window": 20, "rule": is_bat_pattern, "confidence": 0.9},
    "crab": {"type": "chart", "window": 20, "rule": is_crab_pattern, "confidence": 0.9},
    "butterfly": {"type": "chart", "window": 20, "rule": is_butterfly_pattern, "confidence": 0.9},
    "shark": {"type": "chart", "window": 20, "rule": is_shark_pattern, "confidence": 0.9},
    "cypher": {"type": "chart", "window": 20, "rule": is_cypher_pattern, "confidence": 0.9},
    "three_drives": {"type": "chart", "window": 20, "rule": is_three_drives_pattern, "confidence": 0.9},

    # === Elliott Wave Patterns ===
    "impulse_wave": {"type": "chart", "window": 10, "rule": is_impulse_wave, "confidence": 0.88},
    "corrective_wave": {"type": "chart", "window": 10, "rule": is_corrective_wave, "confidence": 0.88},
}