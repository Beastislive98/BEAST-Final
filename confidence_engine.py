# confidence_engine.py

import logging
from typing import Dict, Any


def compute_confidence(match: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fuses pattern similarity, bayesian reliability, and market context into a final confidence score.

    Args:
        match (Dict[str, Any]): One matched pattern result from FAISS, including 'distance' and 'bayesian_score'.
        context (Dict[str, Any]): Includes whale_flags, sentiment, volatility, etc.

    Returns:
        Dict[str, Any]: Augmented match dict with final 'confidence_score'
    """
    try:
        base_score = 1.0 - match.get("distance", 0.5)  # Lower distance = better match
        bayes = match.get("bayesian_score", 0.5)

        whale_boost = 0.05 if context.get("whale_flags", {}).get("whale_present") else 0
        sentiment_score = context.get("sentiment", {}).get("sentiment_score", 0.0)
        sentiment_adjustment = sentiment_score * 0.1

        adjusted = (base_score * 0.5 + bayes * 0.4 + whale_boost + sentiment_adjustment)
        adjusted = max(0.0, min(round(adjusted, 4), 1.0))

        match["confidence_score"] = adjusted
        return match

    except Exception as e:
        logging.exception(f"Confidence fusion failed: {e}")
        match["confidence_score"] = 0.0
        return match


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    faiss_match = {"distance": 0.2, "bayesian_score": 0.85, "pattern": "bullish_engulfing"}
    market_context = {
        "whale_flags": {"whale_present": True},
        "sentiment": {"sentiment_score": 0.6}
    }

    result = compute_confidence(faiss_match, market_context)
    print("Confidence Scored Match:", result)
