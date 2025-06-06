import logging
from typing import Dict, List, any
import requests
import time
import threading
from collections import defaultdict
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# API Configuration
CRYPTO_PANIC_API_KEY = "a14d2e9d7fa2a38b48ff0533059f58d388467a24"
CRYPTO_PANIC_URL = "https://cryptopanic.com/api/v1/posts/"

# FIXED: Improved caching and rate limiting
CACHE_DURATION = 1800  # 30 minutes cache (increased from 15)
headline_cache = {}
cache_lock = threading.Lock()

# FIXED: Proper rate limiting implementation
class RateLimiter:
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def can_call(self) -> bool:
        current_time = time.time()
        with self.lock:
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls 
                         if current_time - call_time < self.time_window]
            
            if len(self.calls) < self.max_calls:
                self.calls.append(current_time)
                return True
            return False
    
    def wait_until_ready(self) -> float:
        """Wait until we can make a call, return wait time"""
        while not self.can_call():
            with self.lock:
                if self.calls:
                    oldest_call = min(self.calls)
                    wait_time = self.time_window - (time.time() - oldest_call) + 1
                    if wait_time > 0:
                        logging.debug(f"Rate limited, waiting {wait_time:.1f} seconds")
                        time.sleep(min(wait_time, 60))  # Cap wait time at 60 seconds
                    else:
                        break
                else:
                    break
        return 0

# FIXED: Global rate limiter
api_rate_limiter = RateLimiter(max_calls=8, time_window=60)  # 8 calls per minute (conservative)

# FIXED: TextBlob availability check
def check_textblob_available() -> bool:
    """Check if TextBlob is available"""
    try:
        from textblob import TextBlob
        # Test basic functionality
        test_blob = TextBlob("test")
        _ = test_blob.sentiment.polarity
        return True
    except ImportError:
        logging.warning("TextBlob not installed, using fallback sentiment analysis")
        return False
    except Exception as e:
        logging.warning(f"TextBlob available but has issues: {e}")
        return False

# Check availability at module level
TEXTBLOB_AVAILABLE = check_textblob_available()

def load_cache():
    """FIXED: Load cache from file"""
    global headline_cache
    try:
        cache_file = "data/sentiment_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
                # Only load recent cache entries
                current_time = time.time()
                valid_cache = {}
                
                for key, cache_entry in data.items():
                    if isinstance(cache_entry, dict) and 'timestamp' in cache_entry:
                        if current_time - cache_entry['timestamp'] < CACHE_DURATION:
                            valid_cache[key] = cache_entry
                    elif isinstance(cache_entry, (list, tuple)) and len(cache_entry) >= 2:
                        # Old format compatibility
                        timestamp, headlines = cache_entry[0], cache_entry[1]
                        if current_time - timestamp < CACHE_DURATION:
                            valid_cache[key] = {
                                'timestamp': timestamp,
                                'headlines': headlines,
                                'sentiment': None
                            }
                
                with cache_lock:
                    headline_cache = valid_cache
                
                logging.info(f"Loaded {len(valid_cache)} cached sentiment entries")
    except Exception as e:
        logging.debug(f"Failed to load sentiment cache: {e}")

def save_cache():
    """FIXED: Save cache to file"""
    try:
        os.makedirs("data", exist_ok=True)
        cache_file = "data/sentiment_cache.json"
        
        with cache_lock:
            cache_data = headline_cache.copy()
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
            
        logging.debug("Sentiment cache saved")
    except Exception as e:
        logging.debug(f"Failed to save sentiment cache: {e}")

# Load cache at startup
load_cache()

def fetch_latest_headlines(symbol: str, max_retries: int = 3) -> List[str]:
    """FIXED: Fetch latest headlines with proper rate limiting and error handling"""
    try:
        now = time.time()
        cache_key = symbol.upper()
        
        # Check cache first
        with cache_lock:
            if cache_key in headline_cache:
                cache_entry = headline_cache[cache_key]
                if isinstance(cache_entry, dict):
                    timestamp = cache_entry.get('timestamp', 0)
                    headlines = cache_entry.get('headlines', [])
                else:
                    # Old format compatibility
                    timestamp, headlines = cache_entry[0], cache_entry[1]
                
                if now - timestamp < CACHE_DURATION:
                    logging.debug(f"Using cached headlines for {symbol}")
                    return headlines

        # Clean symbol for API (remove USDT/BUSD suffix)
        clean_symbol = symbol.replace("USDT", "").replace("BUSD", "").replace("USDC", "")
        
        params = {
            "auth_token": CRYPTO_PANIC_API_KEY,
            "currencies": clean_symbol,
            "kind": "news",
            "filter": "hot",
            "page_size": 10  # Limit to reduce response size
        }

        for attempt in range(max_retries):
            try:
                # FIXED: Use global rate limiter
                api_rate_limiter.wait_until_ready()
                
                response = requests.get(
                    CRYPTO_PANIC_URL, 
                    params=params, 
                    timeout=20,  # Increased timeout
                    headers={
                        'User-Agent': 'SentimentBot/2.0',
                        'Accept': 'application/json'
                    }
                )
                
                # FIXED: Better status code handling
                if response.status_code == 429:
                    # Rate limited by server
                    retry_after = int(response.headers.get('Retry-After', 120))
                    logging.warning(f"Server rate limit hit for {symbol}, waiting {retry_after}s")
                    time.sleep(min(retry_after, 300))  # Cap at 5 minutes
                    continue
                elif response.status_code == 401:
                    logging.error("Invalid API key for CryptoPanic")
                    return []
                elif response.status_code == 403:
                    logging.error("API access forbidden for CryptoPanic")
                    return []
                elif response.status_code == 200:
                    try:
                        data = response.json()
                        articles = data.get("results", [])
                        headlines = []
                        
                        for article in articles:
                            title = article.get("title", "")
                            # FIXED: Better title filtering
                            if title and len(title.strip()) > 15 and not title.lower().startswith("ad:"):
                                headlines.append(title.strip())
                        
                        # Cache the results
                        cache_entry = {
                            'timestamp': now,
                            'headlines': headlines,
                            'sentiment': None
                        }
                        
                        with cache_lock:
                            headline_cache[cache_key] = cache_entry
                        
                        # Save cache periodically
                        if len(headline_cache) % 5 == 0:
                            save_cache()
                        
                        logging.debug(f"Fetched {len(headlines)} headlines for {symbol}")
                        return headlines
                        
                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON response for {symbol}: {e}")
                        break
                    except Exception as e:
                        logging.warning(f"Error parsing response for {symbol}: {e}")
                        break
                else:
                    logging.warning(f"API error {response.status_code} for {symbol}: {response.text[:200]}")
                    if response.status_code >= 500:
                        # Server error, can retry
                        pass
                    else:
                        # Client error, don't retry
                        break
                    
            except requests.exceptions.Timeout:
                logging.debug(f"Timeout fetching headlines for {symbol} (attempt {attempt+1})")
            except requests.exceptions.ConnectionError as e:
                logging.debug(f"Connection error for {symbol} (attempt {attempt+1}): {e}")
            except Exception as e:
                logging.debug(f"Attempt {attempt+1} failed for {symbol}: {e}")
            
            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = min(30, (2 ** attempt) + random.uniform(1, 3))
                time.sleep(wait_time)

        # If all attempts failed, return empty list
        logging.debug(f"Failed to fetch headlines for {symbol} after {max_retries} attempts")
        return []

    except Exception as e:
        logging.error(f"Failed to fetch headlines for {symbol}: {e}")
        return []

def analyze_sentiment(text: str) -> float:
    """FIXED: Analyze sentiment with better fallback handling"""
    if not text or len(text.strip()) < 3:
        return 0.0
    
    # Primary method: TextBlob
    if TEXTBLOB_AVAILABLE:
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            # Ensure polarity is within expected range
            return max(-1.0, min(1.0, polarity))
        except Exception as e:
            logging.debug(f"TextBlob sentiment analysis failed: {e}")
    
    # Fallback method: Enhanced keyword-based sentiment
    return _enhanced_sentiment_analysis(text)

def _enhanced_sentiment_analysis(text: str) -> float:
    """FIXED: Enhanced keyword-based sentiment analysis as fallback"""
    try:
        text_lower = text.lower()
        
        # Enhanced positive keywords with weights
        positive_words = {
            # Strong positive
            'moon': 2.0, 'bull': 1.5, 'bullish': 1.5, 'surge': 1.5, 'pump': 1.5,
            'breakthrough': 1.5, 'rally': 1.5, 'explosive': 1.5, 'soar': 1.5,
            # Medium positive
            'up': 1.0, 'rise': 1.0, 'gain': 1.0, 'profit': 1.0, 'positive': 1.0,
            'good': 1.0, 'great': 1.0, 'excellent': 1.0, 'strong': 1.0,
            'adoption': 1.0, 'partnership': 1.0, 'upgrade': 1.0, 'launch': 1.0,
            'success': 1.0, 'win': 1.0, 'growth': 1.0, 'buy': 0.8, 'long': 0.8,
            # Mild positive
            'optimistic': 0.7, 'confident': 0.7, 'bullish': 0.7, 'upward': 0.7
        }
        
        # Enhanced negative keywords with weights
        negative_words = {
            # Strong negative
            'crash': 2.0, 'dump': 2.0, 'bear': 1.5, 'bearish': 1.5, 'panic': 1.5,
            'scam': 2.0, 'fraud': 2.0, 'hack': 1.5, 'ban': 1.5, 'crisis': 1.5,
            # Medium negative
            'down': 1.0, 'fall': 1.0, 'loss': 1.0, 'negative': 1.0, 'bad': 1.0,
            'terrible': 1.0, 'weak': 1.0, 'fear': 1.0, 'concern': 1.0, 'risk': 1.0,
            'regulation': 1.0, 'problem': 1.0, 'issue': 1.0, 'sell': 0.8, 'short': 0.8,
            # Mild negative
            'uncertain': 0.7, 'doubt': 0.7, 'worry': 0.7, 'decline': 0.7
        }
        
        # Tokenize and analyze
        words = text_lower.split()
        total_score = 0.0
        word_count = 0
        
        for word in words:
            # Remove punctuation
            clean_word = word.strip('.,!?;:"()[]{}')
            
            if clean_word in positive_words:
                total_score += positive_words[clean_word]
                word_count += 1
            elif clean_word in negative_words:
                total_score -= negative_words[clean_word]
                word_count += 1
        
        # Calculate average sentiment
        if word_count == 0:
            return 0.0
        
        avg_sentiment = total_score / len(words)  # Normalize by total words, not just sentiment words
        
        # Apply sigmoid-like function to keep result in [-1, 1] range
        normalized_sentiment = 2 / (1 + math.exp(-3 * avg_sentiment)) - 1
        
        return max(-1.0, min(1.0, normalized_sentiment))
        
    except Exception as e:
        logging.debug(f"Enhanced sentiment analysis failed: {e}")
        return 0.0

def extract_sentiment_score(symbol: str, use_cache: bool = True) -> Dict[str, float]:
    """FIXED: Extract sentiment score with comprehensive error handling and caching"""
    try:
        cache_key = symbol.upper()
        
        # Check cache for complete sentiment result
        if use_cache:
            with cache_lock:
                if cache_key in headline_cache:
                    cache_entry = headline_cache[cache_key]
                    if isinstance(cache_entry, dict):
                        cached_sentiment = cache_entry.get('sentiment')
                        timestamp = cache_entry.get('timestamp', 0)
                        
                        if cached_sentiment is not None and time.time() - timestamp < CACHE_DURATION:
                            logging.debug(f"Using cached sentiment result for {symbol}")
                            return cached_sentiment

        # Fetch fresh headlines
        headlines = fetch_latest_headlines(symbol)
        
        if not headlines:
            # Return neutral sentiment if no headlines
            result = {
                "sentiment_score": 0.0,
                "headline_count": 0,
                "confidence": 0.0,
                "source": "no_data"
            }
            logging.debug(f"No headlines found for {symbol}, returning neutral sentiment")
            return result

        # Analyze sentiment for each headline
        sentiments = []
        valid_headlines = 0
        
        for headline in headlines:
            try:
                if len(headline.strip()) >= 10:  # Only analyze substantial headlines
                    sentiment = analyze_sentiment(headline)
                    if sentiment is not None and abs(sentiment) <= 1.0:  # Validate sentiment range
                        sentiments.append(sentiment)
                        valid_headlines += 1
            except Exception as e:
                logging.debug(f"Failed to analyze headline sentiment: {e}")
                continue

        # Calculate weighted average sentiment
        if sentiments:
            # Weight recent sentiments more heavily if we have timestamps
            # For now, use simple average
            average_sentiment = sum(sentiments) / len(sentiments)
            
            # Calculate confidence based on number of headlines and consistency
            confidence = min(1.0, valid_headlines / 8.0)  # Max confidence at 8+ headlines
            
            # Reduce confidence for very mixed sentiments
            if len(sentiments) > 1:
                sentiment_variance = sum((s - average_sentiment) ** 2 for s in sentiments) / len(sentiments)
                sentiment_consistency = max(0.3, 1.0 - (sentiment_variance * 2))
                confidence *= sentiment_consistency
        else:
            average_sentiment = 0.0
            confidence = 0.0

        result = {
            "sentiment_score": round(average_sentiment, 4),
            "headline_count": len(headlines),
            "valid_headlines": valid_headlines,
            "confidence": round(confidence, 3),
            "source": "textblob" if TEXTBLOB_AVAILABLE else "keyword_based",
            "sample_headlines": headlines[:3] if headlines else []  # Include samples for debugging
        }

        # Cache the complete result
        with cache_lock:
            if cache_key in headline_cache:
                headline_cache[cache_key]['sentiment'] = result
            else:
                headline_cache[cache_key] = {
                    'timestamp': time.time(),
                    'headlines': headlines,
                    'sentiment': result
                }

        # Periodic cache save
        if len(headline_cache) % 3 == 0:
            save_cache()

        logging.debug(f"Sentiment for {symbol}: {result['sentiment_score']:.3f} "
                     f"(confidence: {result['confidence']:.3f}, headlines: {result['headline_count']})")
        return result

    except Exception as e:
        logging.error(f"Sentiment extraction failed for {symbol}: {e}")
        return {
            "sentiment_score": 0.0,
            "headline_count": 0,
            "valid_headlines": 0,
            "confidence": 0.0,
            "source": "error",
            "error": str(e)
        }

def get_market_sentiment(symbols: List[str] = None) -> Dict[str, Any]:
    """FIXED: Get overall market sentiment from major symbols"""
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']
    
    sentiments = []
    total_headlines = 0
    total_confidence = 0
    symbol_results = {}
    
    for symbol in symbols:
        try:
            result = extract_sentiment_score(symbol)
            if result['headline_count'] > 0:
                # Weight by confidence and number of headlines
                weight = result['confidence'] * min(1.0, result['headline_count'] / 5.0)
                weighted_sentiment = result['sentiment_score'] * weight
                
                sentiments.append(weighted_sentiment)
                total_headlines += result['headline_count']
                total_confidence += result['confidence']
                symbol_results[symbol] = result
            else:
                symbol_results[symbol] = result
        except Exception as e:
            logging.debug(f"Failed to get sentiment for {symbol}: {e}")
            continue
    
    if sentiments:
        market_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        avg_confidence = total_confidence / len(symbols) if symbols else 0.0
    else:
        market_sentiment = 0.0
        avg_confidence = 0.0
    
    return {
        "market_sentiment": round(market_sentiment, 4),
        "confidence": round(avg_confidence, 3),
        "total_headlines": total_headlines,
        "symbols_analyzed": len([s for s in symbol_results.values() if s['headline_count'] > 0]),
        "symbol_results": symbol_results,
        "timestamp": time.time()
    }

def cleanup_old_cache():
    """FIXED: Clean up old cache entries"""
    try:
        current_time = time.time()
        
        with cache_lock:
            old_keys = []
            for key, cache_entry in headline_cache.items():
                if isinstance(cache_entry, dict):
                    timestamp = cache_entry.get('timestamp', 0)
                else:
                    # Old format compatibility
                    timestamp = cache_entry[0] if isinstance(cache_entry, (list, tuple)) else 0
                
                if current_time - timestamp > CACHE_DURATION * 3:  # Remove entries older than 3x cache duration
                    old_keys.append(key)
            
            for key in old_keys:
                del headline_cache[key]
        
        if old_keys:
            logging.info(f"Cleaned up {len(old_keys)} old cache entries")
            save_cache()
            
    except Exception as e:
        logging.error(f"Cache cleanup failed: {e}")

# FIXED: Background cache cleanup
_cleanup_thread = None
_cleanup_running = False

def start_cache_cleanup_thread():
    """Start background thread for cache cleanup"""
    global _cleanup_thread, _cleanup_running
    
    if _cleanup_running:
        return
    
    def cleanup_loop():
        global _cleanup_running
        _cleanup_running = True
        while _cleanup_running:
            try:
                time.sleep(3600)  # Clean up every hour
                cleanup_old_cache()
            except Exception as e:
                logging.error(f"Cache cleanup thread error: {e}")
                time.sleep(600)  # Wait 10 minutes before retry
    
    _cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True, name="SentimentCacheCleanup")
    _cleanup_thread.start()
    logging.info("Started sentiment cache cleanup thread")

def stop_cache_cleanup_thread():
    """Stop background cache cleanup thread"""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread and _cleanup_thread.is_alive():
        _cleanup_thread.join(timeout=5)

# Import required modules at the end to avoid circular imports
import random
import math

if __name__ == "__main__":
    # Test the sentiment analyzer
    logging.basicConfig(level=logging.DEBUG)
    
    # Start cache cleanup thread
    start_cache_cleanup_thread()
    
    # Test sentiment analysis
    test_symbols = ["BTCUSDT", "ETHUSDT"]
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}:")
        result = extract_sentiment_score(symbol)
        print(f"  Sentiment: {result['sentiment_score']}")
        print(f"  Headlines: {result['headline_count']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Source: {result['source']}")
        if result.get('sample_headlines'):
            print(f"  Sample headlines: {result['sample_headlines'][:2]}")
    
    # Test market sentiment
    print("\nMarket Sentiment:")
    market = get_market_sentiment()
    print(f"  Overall: {market['market_sentiment']}")
    print(f"  Confidence: {market['confidence']}")
    print(f"  Headlines: {market['total_headlines']}")
    print(f"  Symbols analyzed: {market['symbols_analyzed']}")
    
    # Save cache before exit
    save_cache()
    stop_cache_cleanup_thread()