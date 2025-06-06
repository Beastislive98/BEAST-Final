# data_bundle.py - Enhanced with Market Regime Integration & Crypto Data

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import time
from functools import lru_cache

# Import dependencies with fallback handling
try:
    from binance_interface import get_futures_market_data as get_market_data
except ImportError:
    logging.error("binance_interface module not found! Critical dependency missing.")
    def get_market_data(symbol): return None

try:
    from data_processing import process_market_data
except ImportError:
    logging.error("data_processing module not found! Critical dependency missing.")
    def process_market_data(data): return None

# Enhanced imports for crypto-specific data
try:
    from crypto_data_enhancer import (
        get_funding_rate, get_open_interest_data, 
        get_cross_exchange_spread, get_spot_price
    )
    CRYPTO_DATA_AVAILABLE = True
except ImportError:
    logging.warning("crypto_data_enhancer not available, crypto-specific strategies may be limited")
    CRYPTO_DATA_AVAILABLE = False
    def get_funding_rate(symbol): return None
    def get_open_interest_data(symbol): return None
    def get_cross_exchange_spread(symbol): return None
    def get_spot_price(symbol): return None

# Market regime detection - Enhanced integration
try:
    from market_regime import MarketRegimeDetector
    REGIME_DETECTOR = MarketRegimeDetector()
    REGIME_DETECTION_AVAILABLE = True
except ImportError:
    logging.warning("market_regime module not found, regime detection disabled")
    REGIME_DETECTION_AVAILABLE = False
    REGIME_DETECTOR = None

# Optional components with fallback
try:
    from indicator_universe import calculate_indicators
    INDICATOR_PARAMS = {
        'SMA': {'period': 14}, 'EMA': {'period': 20}, 'RSI': {'period': 14}, 'MACD': {},
        'BBANDS': {}, 'ATR': {'period': 14}, 'CCI': {'period': 20}, 'OBV': {}, 'MFI': {'period': 14},
        'Stochastic': {}, 'Momentum': {'period': 10}, 'ROC': {'period': 12}, 'ADX': {'period': 14},
        'KST': {}, 'TSI': {}, 'UltimateOscillator': {}, 'Vortex': {}, 'Donchian': {},
        'Keltner': {}, 'Ichimoku': {}, 'MassIndex': {}, 'ChandelierExit': {},
        'CMF': {'period': 20}, 'EOM': {'period': 20}, 'NVI': {}, 'PVI': {}
    }
except ImportError:
    logging.warning("indicator_universe module not found, skipping indicator calculation")
    calculate_indicators = None
    INDICATOR_PARAMS = {}

# Import remaining modules with fallbacks
for module_name in [
    "whale_detector", "sentiment_analyzer", "forecasting", "visual_transformer", 
    "pattern_recognition", "faiss_search", "bayesian_memory", "confidence_engine"
]:
    try:
        globals()[module_name] = __import__(module_name)
        logging.debug(f"Successfully imported {module_name}")
    except ImportError:
        globals()[module_name] = None
        logging.warning(f"{module_name} module not found, related features will be skipped")

# Enhanced caching for performance optimization
@lru_cache(maxsize=100)
def get_cached_regime_data(symbol: str, timeframe: str, data_hash: str) -> Optional[Dict[str, Any]]:
    """Cache regime detection results to avoid recalculation"""
    # This would normally store the regime detection result
    # The data_hash ensures cache invalidation when data changes
    return None

def clear_regime_cache():
    """Clear the regime detection cache"""
    get_cached_regime_data.cache_clear()

def transform_klines_to_ohlcv(market_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Transforms klines data to OHLCV format if not already present.
    """
    if "ohlcv" in market_data:
        return market_data["ohlcv"]
        
    if "klines" not in market_data:
        return None
        
    try:
        klines = market_data["klines"]
        
        # Binance klines format: [open_time, open, high, low, close, volume, ...]
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", 
            "close_time", "quote_volume", "trades", "taker_buy_base", 
            "taker_buy_quote", "ignored"
        ])
        
        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamp to datetime for easier handling
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Rename columns to match expected format
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        
        return df
    except Exception as e:
        logging.error(f"Failed to transform klines to OHLCV: {e}")
        return None

def enhance_market_data_with_crypto_metrics(market_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Enhance market data with crypto-specific metrics
    """
    if not CRYPTO_DATA_AVAILABLE:
        return market_data
    
    try:
        # Add funding rate for perpetual futures
        funding_rate = get_funding_rate(symbol)
        if funding_rate is not None:
            market_data["funding_rate"] = funding_rate
            logging.debug(f"Added funding rate for {symbol}: {funding_rate}")
        
        # Add open interest data
        oi_data = get_open_interest_data(symbol)
        if oi_data:
            market_data.update(oi_data)  # Includes oi_change_pct, oi_value, etc.
            logging.debug(f"Added OI data for {symbol}")
        
        # Add cross-exchange spread
        spread = get_cross_exchange_spread(symbol)
        if spread is not None:
            market_data["cross_exchange_spread"] = spread
            logging.debug(f"Added cross-exchange spread for {symbol}: {spread}")
        
        # Add spot price for futures symbols
        if "USDT" in symbol:  # Likely a futures symbol
            spot_price = get_spot_price(symbol)
            if spot_price:
                market_data["spot_price"] = spot_price
                # Calculate basis
                futures_price = market_data.get("price", 0)
                if futures_price and spot_price:
                    basis = (futures_price - spot_price) / spot_price
                    market_data["basis"] = basis
                    logging.debug(f"Added spot price and basis for {symbol}")
        
    except Exception as e:
        logging.warning(f"Failed to enhance market data with crypto metrics for {symbol}: {e}")
    
    return market_data

def detect_market_regime_enhanced(df: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Enhanced market regime detection with caching and error handling
    """
    if not REGIME_DETECTION_AVAILABLE or df is None or len(df) < 100:
        return {"regime_type": "unknown", "confidence": 0.0}
    
    try:
        # Create a hash of the recent data for caching
        recent_data = df.tail(50)
        data_hash = str(hash(str(recent_data['Close'].tolist())))
        
        # Try to get cached result
        cached_result = get_cached_regime_data(symbol, "1h", data_hash)
        if cached_result:
            return cached_result
        
        # Perform regime detection
        regime_result = REGIME_DETECTOR.predict_regime(df)
        
        # Enhance with additional context
        if regime_result.get("regime_type") != "unknown":
            # Add volatility context
            volatility = df['Close'].pct_change().std()
            regime_result["current_volatility"] = volatility
            
            # Add trend strength
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else sma_20
            trend_strength = abs((sma_20 - sma_50) / sma_50) if sma_50 != 0 else 0
            regime_result["trend_strength"] = trend_strength
            
            logging.info(f"Regime detected for {symbol}: {regime_result['regime_type']} "
                        f"(confidence: {regime_result['confidence']:.2f}, "
                        f"volatility: {volatility:.4f})")
        
        return regime_result
        
    except Exception as e:
        logging.warning(f"Enhanced regime detection failed for {symbol}: {e}")
        return {"regime_type": "unknown", "confidence": 0.0}

def assemble_data_bundle(symbol: str, lightweight: bool = False) -> Optional[Dict[str, Any]]:
    """
    Enhanced data bundle assembly with market regime integration and crypto-specific data
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        lightweight: If True, perform a lighter analysis (for cold symbols)
        
    Returns:
        Dictionary containing all data needed for decision making or None if critical data missing
    """
    start_time = time.time()
    logging.info(f"Assembling {'lightweight' if lightweight else 'full'} enhanced data bundle for {symbol}")
    
    try:
        # Step 1: Get market data from Binance
        market_data = get_market_data(symbol)
        if not market_data:
            logging.warning(f"No market data received for {symbol}")
            return None
        
        # Step 1.5: Enhance with crypto-specific metrics
        market_data = enhance_market_data_with_crypto_metrics(market_data, symbol)
        
        # Step 2: Transform klines to OHLCV if needed
        if "ohlcv" not in market_data:
            ohlcv_df = transform_klines_to_ohlcv(market_data)
            if ohlcv_df is not None:
                market_data["ohlcv"] = ohlcv_df
                logging.info(f"Successfully transformed klines to OHLCV format for {symbol}")
            else:
                logging.warning(f"Failed to transform klines to OHLCV for {symbol}")
        
        # Create basic bundle with required data
        bundle = {
            "symbol": symbol,
            "market_data": market_data,
            "symbol_tag": "neutral",  # Default value, will be updated if available
            "assembly_time": time.time(),
            "data_quality": "basic"
        }
        
        # Step 3: Process market data to create clean DataFrame
        df = process_market_data(market_data)
        if df is None:
            logging.warning(f"Data processing failed for {symbol}")
            return bundle  # Return basic bundle even if processing fails
        
        # Verify DataFrame has all required columns
        required_columns = ['Volume', 'High', 'Close', 'Open', 'Low']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns for {symbol}: {missing_cols}")
            return bundle
        
        # Step 3.5: ENHANCED - Market Regime Detection (Early)
        regime_info = detect_market_regime_enhanced(df, symbol)
        bundle["market_regime"] = regime_info
        bundle["regime_type"] = regime_info.get("regime_type", "unknown")
        
        if regime_info.get("regime_type") != "unknown":
            bundle["data_quality"] = "enhanced"
            logging.info(f"Market regime for {symbol}: {regime_info['regime_type']} "
                        f"(confidence: {regime_info.get('confidence', 0):.2f})")
        
        # Step 4: Calculate technical indicators (regime-aware)
        indicator_data = None
        if calculate_indicators and not lightweight:
            try:
                # Dynamically adjust indicator parameters based on regime
                adjusted_params = INDICATOR_PARAMS.copy()
                
                # In high volatility regimes, use shorter periods for faster response
                if regime_info.get("regime_type") == "high_volatility":
                    for indicator in adjusted_params:
                        if "period" in adjusted_params[indicator]:
                            adjusted_params[indicator]["period"] = max(7, 
                                adjusted_params[indicator]["period"] // 2)
                    logging.debug(f"Adjusted indicator periods for high volatility regime")
                
                indicator_data = calculate_indicators(df.copy(), adjusted_params)
                if indicator_data is None or len(indicator_data) < 64:
                    logging.warning(f"Insufficient data for indicator calculation for {symbol}")
                else:
                    bundle["indicator_data"] = indicator_data
                    bundle["data_quality"] = "full"
            except Exception as e:
                logging.warning(f"Failed to calculate indicators for {symbol}: {e}")
        
        # For lightweight analysis, skip some steps but include regime info
        if lightweight:
            # Still need basic indicator data for pattern detection
            try:
                minimal_params = {'RSI': {'period': 14}, 'SMA': {'period': 20}, 'ATR': {'period': 14}}
                indicator_data = calculate_indicators(df.copy(), minimal_params)
                if indicator_data is not None:
                    bundle["indicator_data"] = indicator_data
            except Exception as e:
                logging.warning(f"Failed to calculate minimal indicators for {symbol}: {e}")
            
            # Extract key levels for ATM/ITM/OTM assessment
            try:
                from pattern_recognition import get_key_levels
                key_levels = get_key_levels(df)
                if key_levels:
                    bundle["key_levels"] = key_levels
            except Exception as e:
                logging.warning(f"Failed to calculate key levels for {symbol}: {e}")
                
            # Do basic pattern recognition
            if "pattern_recognition" in globals() and pattern_recognition and indicator_data is not None:
                try:
                    pattern_signal = pattern_recognition.recognize_patterns(indicator_data, symbol)
                    if pattern_signal:
                        bundle["pattern_signal"] = pattern_signal
                except Exception as e:
                    logging.warning(f"Pattern recognition failed for {symbol}: {e}")
                
            # Skip the rest of the heavy processing
            processing_time = time.time() - start_time
            bundle["processing_time"] = processing_time
            logging.info(f"Lightweight enhanced data bundle assembled for {symbol} in {processing_time:.2f} seconds")
            return bundle
        
        # Step 5: Detect whale activity (if available)
        if "whale_detector" in globals() and whale_detector:
            try:
                whale_flags = whale_detector.detect_whale_activity(market_data)
                bundle["whale_flags"] = whale_flags
                logging.info(f"Whale detection result for {symbol}: {whale_flags}")
            except Exception as e:
                logging.warning(f"Whale detection failed for {symbol}: {e}")
                bundle["whale_flags"] = {"whale_present": False}
        
        # Step 5.5: Enhanced order flow analysis with regime context
        if not lightweight and "market_data" in bundle and "depth" in bundle["market_data"]:
            try:
                from order_flow import OrderFlowAnalyzer
                analyzer = OrderFlowAnalyzer()
                order_flow = analyzer.analyze_order_book(bundle["market_data"]["depth"])
                
                # Enhance with regime context
                if order_flow.get("valid", False) and regime_info.get("regime_type") != "unknown":
                    regime_type = regime_info["regime_type"]
                    
                    # Adjust pressure interpretation based on regime
                    if regime_type == "high_volatility":
                        order_flow["regime_adjusted_pressure"] = order_flow["pressure_strength"] * 1.2
                    elif regime_type == "ranging":
                        order_flow["regime_adjusted_pressure"] = order_flow["pressure_strength"] * 0.8
                    else:
                        order_flow["regime_adjusted_pressure"] = order_flow["pressure_strength"]
                
                bundle["order_flow"] = order_flow
                
                if order_flow.get("valid", False):
                    logging.info(f"Order flow analysis for {symbol}: {order_flow['direction']} "
                               f"with {order_flow['pressure_strength']:.2f} strength "
                               f"(regime-adjusted: {order_flow.get('regime_adjusted_pressure', 0):.2f})")
            except Exception as e:
                logging.warning(f"Order flow analysis failed for {symbol}: {e}")
        
        # Step 5.75: Enhanced multi-timeframe analysis with regime correlation
        if not lightweight and "ohlcv" in market_data:
            try:
                from multi_timeframe import MultiTimeframeAnalyzer
                mtf_analyzer = MultiTimeframeAnalyzer()
                
                # Derive higher timeframes
                timeframe_data = mtf_analyzer.derive_higher_timeframes(market_data["ohlcv"])
                
                # Analyze multi-timeframe trends with regime awareness
                mtf_analysis = mtf_analyzer.analyze_multi_timeframe(timeframe_data)
                
                # Enhance with regime correlation
                if mtf_analysis.get("valid", False) and regime_info.get("regime_type") != "unknown":
                    regime_type = regime_info["regime_type"]
                    
                    # Calculate regime-timeframe alignment score
                    if regime_type == "bull_trend" and mtf_analysis["alignment"] == "bullish":
                        mtf_analysis["regime_alignment_bonus"] = 0.2
                    elif regime_type == "bear_trend" and mtf_analysis["alignment"] == "bearish":
                        mtf_analysis["regime_alignment_bonus"] = 0.2
                    else:
                        mtf_analysis["regime_alignment_bonus"] = 0.0
                    
                    # Adjust alignment strength
                    original_strength = mtf_analysis["alignment_strength"]
                    bonus = mtf_analysis["regime_alignment_bonus"]
                    mtf_analysis["regime_adjusted_strength"] = min(1.0, original_strength + bonus)
                
                if mtf_analysis.get("valid", False):
                    bundle["timeframe_data"] = timeframe_data
                    bundle["mtf_analysis"] = mtf_analysis
                    
                    logging.info(f"Multi-timeframe analysis for {symbol}: {mtf_analysis['alignment']} "
                               f"with {mtf_analysis['alignment_strength']:.2f} strength "
                               f"(regime-adjusted: {mtf_analysis.get('regime_adjusted_strength', 0):.2f})")
            except Exception as e:
                logging.warning(f"Multi-timeframe analysis failed for {symbol}: {e}")
        
        # Step 6: Sentiment analysis (if available)
        if "sentiment_analyzer" in globals() and sentiment_analyzer:
            try:
                sentiment = sentiment_analyzer.extract_sentiment_score(symbol)
                bundle["sentiment"] = sentiment
                logging.info(f"Sentiment for {symbol}: {sentiment}")
            except Exception as e:
                logging.warning(f"Sentiment analysis failed for {symbol}: {e}")
                bundle["sentiment"] = {"sentiment_score": 0, "headline_count": 0}
        
        # Step 7: Price forecasting with regime context (if available)
        if "forecasting" in globals() and forecasting and indicator_data is not None:
            try:
                forecast_result = forecasting.forecast_price_direction(indicator_data)
                
                # Enhance forecast with regime context
                if forecast_result and regime_info.get("regime_type") != "unknown":
                    regime_type = regime_info["regime_type"]
                    original_slope = forecast_result.get("slope", 0)
                    
                    # Adjust forecast confidence based on regime alignment
                    if regime_type == "bull_trend" and original_slope > 0:
                        forecast_result["regime_confidence_boost"] = 0.1
                    elif regime_type == "bear_trend" and original_slope < 0:
                        forecast_result["regime_confidence_boost"] = 0.1
                    elif regime_type == "ranging":
                        forecast_result["regime_confidence_boost"] = -0.05  # Lower confidence in ranging
                    else:
                        forecast_result["regime_confidence_boost"] = 0.0
                
                if forecast_result:
                    bundle["forecast"] = forecast_result
            except Exception as e:
                logging.warning(f"Forecasting failed for {symbol}: {e}")
                bundle["forecast"] = {"forecast_price": 0, "slope": 0}
        
        # Step 8: Pattern recognition (if available)
        if "pattern_recognition" in globals() and pattern_recognition and indicator_data is not None:
            try:
                pattern_signal = pattern_recognition.recognize_patterns(indicator_data, symbol)
                if pattern_signal:
                    bundle["pattern_signal"] = pattern_signal
            except Exception as e:
                logging.warning(f"Pattern recognition failed for {symbol}: {e}")
                bundle["pattern_signal"] = {"pattern": None, "confidence": 0}
        
        # Step 9: Visual embedding (if available)
        if "visual_transformer" in globals() and visual_transformer and indicator_data is not None:
            try:
                embedding = visual_transformer.chart_to_image(indicator_data.tail(64))
                if embedding is not None:
                    embedding_vector = embedding.flatten().astype(np.float32)
                    bundle["embedding"] = embedding_vector
            except Exception as e:
                logging.warning(f"Visual embedding failed for {symbol}: {e}")
        
        # Step 10: Pattern matching from memory (if available)
        if "faiss_search" in globals() and faiss_search and "embedding" in bundle:
            try:
                pattern_bank = getattr(faiss_search, "pattern_memory_bank", None)
                if pattern_bank:
                    faiss_result = pattern_bank.query_pattern(bundle["embedding"])
                    if faiss_result and "matches" in faiss_result:
                        bundle["pattern_matches"] = []
                        
                        # Score matches with Bayesian scorer (if available)
                        if "bayesian_memory" in globals() and bayesian_memory:
                            try:
                                bayesian_scorer = getattr(bayesian_memory, "bayesian_scorer", None)
                                if bayesian_scorer:
                                    scored = bayesian_scorer.batch_score(faiss_result["matches"])
                                    
                                    # Add confidence calculation (if available)
                                    if "confidence_engine" in globals() and confidence_engine:
                                        try:
                                            compute_confidence = getattr(confidence_engine, "compute_confidence", None)
                                            if compute_confidence:
                                                context = {
                                                    "whale_flags": bundle.get("whale_flags", {}),
                                                    "sentiment": bundle.get("sentiment", {}),
                                                    "market_regime": bundle.get("market_regime", {})  # Enhanced context
                                                }
                                                bundle["pattern_matches"] = [
                                                    compute_confidence(m, context) for m in scored
                                                ]
                                        except Exception as e:
                                            logging.warning(f"Confidence calculation failed for {symbol}: {e}")
                                            bundle["pattern_matches"] = scored
                                    else:
                                        bundle["pattern_matches"] = scored
                            except Exception as e:
                                logging.warning(f"Bayesian scoring failed for {symbol}: {e}")
            except Exception as e:
                logging.warning(f"Pattern matching failed for {symbol}: {e}")
        
        # Step 11: Extract key price levels for ATM/ITM/OTM assessment
        try:
            from pattern_recognition import get_key_levels
            key_levels = get_key_levels(df)
            if key_levels:
                bundle["key_levels"] = key_levels
        except Exception as e:
            logging.warning(f"Failed to calculate key levels for {symbol}: {e}")
        
        # Step 12: Classify symbol behavior with regime context
        if "classify_symbol_behavior" in globals():
            try:
                from binance_interface import classify_symbol_behavior
                tag = classify_symbol_behavior(market_data)
                
                # Enhance tag with regime information
                regime_type = regime_info.get("regime_type", "unknown")
                if regime_type != "unknown":
                    tag = f"{tag}_{regime_type}"
                
                bundle["symbol_tag"] = tag
            except Exception as e:
                logging.warning(f"Symbol behavior classification failed for {symbol}: {e}")
        
        # Step 13: Add crypto-specific analysis summary
        crypto_summary = {
            "has_funding_data": "funding_rate" in market_data,
            "has_oi_data": "oi_change_pct" in market_data,
            "has_cross_exchange_data": "cross_exchange_spread" in market_data,
            "has_spot_data": "spot_price" in market_data,
            "regime_detected": regime_info.get("regime_type") != "unknown",
            "whale_activity": bundle.get("whale_flags", {}).get("whale_present", False)
        }
        bundle["crypto_analysis_summary"] = crypto_summary
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        bundle["processing_time"] = processing_time
        
        # Final data quality assessment
        quality_score = 0
        if bundle.get("indicator_data") is not None:
            quality_score += 30
        if regime_info.get("regime_type") != "unknown":
            quality_score += 25
        if bundle.get("whale_flags", {}).get("whale_present"):
            quality_score += 15
        if bundle.get("order_flow", {}).get("valid"):
            quality_score += 15
        if crypto_summary["has_funding_data"]:
            quality_score += 15
        
        bundle["data_quality_score"] = quality_score
        bundle["data_quality"] = "premium" if quality_score >= 80 else "enhanced" if quality_score >= 50 else "basic"
        
        logging.info(f"Enhanced data bundle assembled for {symbol} in {processing_time:.2f} seconds "
                    f"(Quality: {bundle['data_quality']} - {quality_score}%)")
        return bundle

    except Exception as e:
        logging.exception(f"Failed to assemble enhanced data bundle for {symbol}: {e}")
        return None

# Enhanced bundle validation
def validate_bundle_for_strategy(bundle: Dict[str, Any], strategy_requirements: Dict[str, Any]) -> bool:
    """
    Validate that a bundle meets the requirements for a specific strategy
    """
    if not bundle or not strategy_requirements:
        return False
    
    # Check regime requirements
    if "preferred_regimes" in strategy_requirements:
        preferred_regimes = strategy_requirements["preferred_regimes"]
        if "any" not in preferred_regimes:
            current_regime = bundle.get("regime_type", "unknown")
            if current_regime not in preferred_regimes:
                return False
    
    # Check crypto-specific requirements
    crypto_summary = bundle.get("crypto_analysis_summary", {})
    
    if strategy_requirements.get("requires_funding_data", False):
        if not crypto_summary.get("has_funding_data", False):
            return False
    
    if strategy_requirements.get("requires_oi_data", False):
        if not crypto_summary.get("has_oi_data", False):
            return False
    
    # Check data quality requirements
    min_quality_score = strategy_requirements.get("min_quality_score", 0)
    if bundle.get("data_quality_score", 0) < min_quality_score:
        return False
    
    return True