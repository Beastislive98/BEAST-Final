import streamlit as st
import pandas as pd
import os
import json
import time
import subprocess
import sys
import threading
import requests
from datetime import datetime
from glob import glob
from flask import Flask, request, jsonify

st.set_page_config(page_title="BEAST Futures Dashboard", layout="wide")
st.title("📈 BEAST Binance Futures Dashboard")

# Paths
TRADE_LOG_DIR = "trade_signals"
ERROR_LOG = "./logs/error_log.txt"
CLASSIFIED_SYMBOLS_JSON = "./logs/classified_symbols.json"
BEAST_SCRIPT = "symbol_scheduler.py"

# Global state
if 'beast_process' not in st.session_state:
    st.session_state.beast_process = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'log_thread' not in st.session_state:
    st.session_state.log_thread = None
if 'beast_log' not in st.session_state:
    st.session_state.beast_log = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'api_server' not in st.session_state:
    st.session_state.api_server = None

# Create Flask app for API endpoints
app = Flask(__name__)
API_KEY = os.getenv("BEAST_API_KEY", "your_secure_api_key")
API_PORT = int(os.getenv("API_PORT", 5000))

@app.route('/api/start', methods=['POST'])
def start_beast_api():
    """API endpoint to start BEAST engine"""
    api_key = request.json.get('api_key')
    
    # Validate API key
    if api_key != API_KEY:
        return jsonify({"status": "error", "message": "Invalid API key"}), 401
        
    # Start BEAST engine
    success = start_beast_engine()
    if success:
        return jsonify({"status": "success", "message": "BEAST engine started"})
    else:
        return jsonify({"status": "error", "message": "Failed to start BEAST engine"}), 500

@app.route('/api/stop', methods=['POST'])
def stop_beast_api():
    """API endpoint to stop BEAST engine"""
    api_key = request.json.get('api_key')
    
    # Validate API key
    if api_key != API_KEY:
        return jsonify({"status": "error", "message": "Invalid API key"}), 401
        
    # Stop BEAST engine
    success = stop_beast_engine()
    if success:
        return jsonify({"status": "success", "message": "BEAST engine stopped"})
    else:
        return jsonify({"status": "error", "message": "Failed to stop BEAST engine"}), 500

@app.route('/api/status', methods=['GET'])
def beast_status_api():
    """API endpoint to get BEAST status"""
    api_key = request.args.get('api_key')
    
    # Validate API key
    if api_key != API_KEY:
        return jsonify({"status": "error", "message": "Invalid API key"}), 401
        
    # Get BEAST status
    running = is_beast_running()
    return jsonify({
        "status": "success",
        "running": running,
        "state": "running" if running else "stopped",
        "active_since": st.session_state.start_time.isoformat() if running and st.session_state.start_time else None
    })

@app.route('/api/trades', methods=['GET'])
def get_trades_api():
    """API endpoint to get recent trades"""
    api_key = request.args.get('api_key')
    
    # Validate API key
    if api_key != API_KEY:
        return jsonify({"status": "error", "message": "Invalid API key"}), 401
        
    # Get recent trades
    trades = load_trade_signals()
    return jsonify({
        "status": "success",
        "count": len(trades),
        "trades": trades
    })

def run_api_server():
    """Run the Flask API server"""
    app.run(host='0.0.0.0', port=API_PORT, threaded=True)

@st.cache_data(ttl=5)
def load_trade_signals():
    if not os.path.exists(TRADE_LOG_DIR):
        return []
    files = sorted(glob(f"{TRADE_LOG_DIR}/*.json"), reverse=True)[:50]
    entries = []
    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                entries.append(data)
        except Exception:
            continue
    return entries

@st.cache_data(ttl=5)
def load_errors():
    if not os.path.exists(ERROR_LOG):
        return []
    with open(ERROR_LOG, "r") as f:
        raw = f.read().split("\n\n")
    return [r.strip() for r in raw if r.strip()]

# Function to start the BEAST engine
def start_beast_engine():
    if st.session_state.beast_process is None or st.session_state.beast_process.poll() is not None:
        try:
            # Get the absolute path to the script
            script_path = os.path.abspath(BEAST_SCRIPT)
            
            # Use Python executable from current environment
            python_path = sys.executable
            
            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)
            os.makedirs("trade_signals", exist_ok=True)
            
            # Create the full command
            cmd = [python_path, script_path]
            
            # Start the process without capturing output
            st.session_state.beast_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Record start time
            st.session_state.start_time = datetime.now()
            
            # Give it a moment to start
            time.sleep(1)
            
            # Check if it's still running
            if st.session_state.beast_process.poll() is None:
                return True
            else:
                # Process exited immediately
                return_code = st.session_state.beast_process.poll()
                output, _ = st.session_state.beast_process.communicate()
                st.error(f"BEAST Engine exited with code {return_code}. Output: {output}")
                st.session_state.beast_process = None
                return False
        except Exception as e:
            st.error(f"⚠️ Failed to start BEAST Engine: {e}")
            return False
    else:
        st.info("ℹ️ BEAST Engine is already running")
        return True

# Function to stop the BEAST engine
def stop_beast_engine():
    if st.session_state.beast_process is not None:
        try:
            # Try sending SIGTERM first
            st.session_state.beast_process.terminate()
            
            # Give it a moment to shut down gracefully
            for _ in range(10):  # Wait up to 10 seconds
                if st.session_state.beast_process.poll() is not None:
                    st.session_state.beast_process = None
                    st.session_state.start_time = None
                    return True
                time.sleep(1)
            
            # If still running, force kill
            st.session_state.beast_process.kill()
            st.session_state.beast_process.wait(timeout=5)
            st.session_state.beast_process = None
            st.session_state.start_time = None
            return True
        except Exception as e:
            st.error(f"⚠️ Failed to stop BEAST Engine: {e}")
            # Force reset the process state
            st.session_state.beast_process = None
            st.session_state.start_time = None
            return False
    else:
        st.info("ℹ️ BEAST Engine is not running")
        return True

# Function to monitor logs
def monitor_logs():
    while True:
        try:
            if os.path.exists("logs/beast_output.log"):
                with open("logs/beast_output.log", "r") as f:
                    lines = f.readlines()
                    st.session_state.beast_log = lines[-100:] if len(lines) > 100 else lines
        except Exception as e:
            print(f"Error reading logs: {e}")
        time.sleep(2)

# Function to determine if BEAST process is running
def is_beast_running():
    return (st.session_state.beast_process is not None and 
            st.session_state.beast_process.poll() is None)

# Start the API server if not already running
if st.session_state.api_server is None:
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    st.session_state.api_server = api_thread

# Sidebar
st.sidebar.header("🔎 Monitoring")
view = st.sidebar.radio("Select View", [
    "Trading Dashboard", "Trade Log", "Beast Log", "Error Log", 
    "Symbol Tags", "Performance Metrics", "API Documentation"
])

# Beast control
st.sidebar.header("🤖 BEAST Control")
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🚀 Start BEAST", key="start_btn"):
        if start_beast_engine():
            st.success("✅ BEAST Engine started!")

with col2:
    if st.button("🛑 Stop BEAST", key="stop_btn"):
        if stop_beast_engine():
            st.success("✅ BEAST Engine stopped!")

# Display status
beast_status = "🟢 Running" if is_beast_running() else "🔴 Stopped"
uptime = ""
if is_beast_running() and st.session_state.start_time:
    uptime_seconds = (datetime.now() - st.session_state.start_time).total_seconds()
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

st.sidebar.metric("BEAST Status", beast_status)
if uptime:
    st.sidebar.metric("Uptime", uptime)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

# Set refresh rate
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 
                              min_value=5, max_value=60, value=10, step=5)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Display views
if view == "Trading Dashboard":
    st.subheader("📊 BEAST Trading Dashboard")
    
    # Display summary stats
    trades = load_trade_signals()
    
    if not trades:
        st.info("No trades found.")
    else:
        # Summary metrics
        st.subheader("Trading Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        # Count total trades
        col1.metric("Total Trades", len(trades))
        
        # Count by side
        if "side" in trades[0]:
            buy_count = sum(1 for t in trades if t.get("side") == "BUY")
            sell_count = sum(1 for t in trades if t.get("side") == "SELL")
            col2.metric("Buy Orders", buy_count)
            col3.metric("Sell Orders", sell_count)
        
        # Calculate success rate if available
        success_rate = ""
        if os.path.exists("logs/trade_results.json"):
            try:
                with open("logs/trade_results.json", "r") as f:
                    results = [json.loads(line) for line in f if line.strip()]
                
                if results:
                    wins = sum(1 for r in results if r.get("pnl", 0) > 0)
                    success_rate = f"{wins/len(results):.1%}"
            except:
                pass
        
        if success_rate:
            col4.metric("Win Rate", success_rate)
        
        # Recent trades table
        st.subheader("Recent Trades")
        if trades:
            df = pd.DataFrame(trades[:10])
            wanted_cols = ["timestamp", "symbol", "side", "positionSide", "leverage",
                          "entry", "stopLoss", "takeProfit", "confidence", "strategy_name"]
            existing_cols = [col for col in wanted_cols if col in df.columns]
            
            if "timestamp" in existing_cols:
                df = df[existing_cols].sort_values(by="timestamp", ascending=False)
            else:
                df = df[existing_cols]
                
            st.dataframe(df, use_container_width=True)
        
        # Active trades
        st.subheader("Active Trades")
        try:
            from pipeline import active_trades
            if active_trades:
                active_df = pd.DataFrame(active_trades)
                active_cols = [col for col in wanted_cols if col in active_df.columns]
                st.dataframe(active_df[active_cols], use_container_width=True)
            else:
                st.info("No active trades at the moment")
        except:
            st.info("Cannot retrieve active trades information")

elif view == "Trade Log":
    st.subheader("🟢 Recent Trade Signals")
    trades = load_trade_signals()
    if not trades:
        st.info("No trades found.")
    else:
        df = pd.DataFrame(trades)
        wanted_cols = ["timestamp", "symbol", "side", "positionSide", "leverage",
                       "entry", "stopLoss", "takeProfit", "confidence", "strategy_name"]
        existing_cols = [col for col in wanted_cols if col in df.columns]
        missing = list(set(wanted_cols) - set(df.columns))
        
        if missing:
            st.warning(f"Missing columns in trade data: {missing}")

        if "timestamp" in existing_cols:
            df = df[existing_cols].sort_values(by="timestamp", ascending=False)
        else:
            df = df[existing_cols]

        st.dataframe(df, use_container_width=True)
        
        # Add summary metrics
        if len(df) > 0:
            st.subheader("Trade Summary")
            cols = st.columns(4)
            
            # Count by symbol
            if "symbol" in df.columns:
                symbol_counts = df["symbol"].value_counts()
                most_traded = symbol_counts.index[0] if len(symbol_counts) > 0 else "None"
                cols[0].metric("Most Traded", most_traded)
            
            # Count by side
            if "side" in df.columns:
                buy_count = len(df[df["side"] == "BUY"])
                sell_count = len(df[df["side"] == "SELL"])
                cols[1].metric("Buy Orders", buy_count)
                cols[2].metric("Sell Orders", sell_count)
            
            # Total trades
            cols[3].metric("Total Trades", len(df))

elif view == "Beast Log":
    st.subheader("🔍 BEAST Engine Log")
    if st.session_state.beast_log:
        log_text = "".join(st.session_state.beast_log)
        st.text_area("Latest Output", log_text, height=400)
    else:
        st.info("No BEAST log data available. Start the BEAST engine to see logs.")
        
elif view == "Error Log":
    st.subheader("❌ Recent Errors")
    errors = load_errors()
    if not errors:
        st.info("No errors found.")
    else:
        for err in errors[-20:][::-1]:
            st.code(err, language="text")

elif view == "Symbol Tags":
    st.subheader("📊 Symbol Classifications")
    if os.path.exists("logs/symbol_status.json"):
        try:
            with open("logs/symbol_status.json", "r") as f:
                status_data = json.load(f)
                
            if "symbol_stats" in status_data:
                symbol_stats = status_data["symbol_stats"]
                
                # Convert to dataframe
                data = []
                for symbol, stats in symbol_stats.items():
                    data.append({
                        "symbol": symbol,
                        "tag": stats.get("tag", "unknown"),
                        "score": stats.get("score", 0),
                        "timestamp": stats.get("timestamp", "")
                    })
                
                df = pd.DataFrame(data)
                
                # Add hot/cold status
                hot_symbols = set(status_data.get("hot_symbols", []))
                cold_symbols = set(status_data.get("cold_symbols", []))
                
                df["status"] = df["symbol"].apply(
                    lambda s: "🔥 Hot" if s in hot_symbols else (
                        "❄️ Cold" if s in cold_symbols else "Neutral"
                    )
                )
                
                st.dataframe(df.sort_values("score", ascending=False), use_container_width=True)
                
                # Show counts by tag
                st.subheader("Tag Distribution")
                tag_counts = df["tag"].value_counts()
                st.bar_chart(tag_counts)
                
                # Show hot/cold distribution
                st.subheader("Symbol Status")
                status_counts = df["status"].value_counts()
                st.bar_chart(status_counts)
            else:
                st.warning("No symbol stats available in status file.")
        except Exception as e:
            st.error(f"Error loading symbol data: {e}")
    else:
        st.warning("No classification data available.")

elif view == "Performance Metrics":
    st.subheader("📈 Performance Metrics")
    
    # Try to load trade results
    if os.path.exists("logs/trade_results.json"):
        try:
            results = []
            with open("logs/trade_results.json", "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
                        
            if results:
                # Convert to dataframe
                df = pd.DataFrame(results)
                
                # Calculate cumulative PnL
                if "pnl" in df.columns:
                    df["cumulative_pnl"] = df["pnl"].cumsum()
                    
                    # Add timestamp column if not exists
                    if "timestamp" not in df.columns and "trade_id" in df.columns:
                        # Extract timestamp from trade_id (format: trade_XXXXX_TIMESTAMP)
                        df["timestamp"] = df["trade_id"].apply(
                            lambda x: x.split("_")[-1] if isinstance(x, str) and len(x.split("_")) > 2 else ""
                        )
                    
                    # Sort by timestamp
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.sort_values("timestamp")
                    
                    # Create PnL chart
                    st.subheader("Cumulative Profit/Loss")
                    st.line_chart(df["cumulative_pnl"])
                    
                    # Calculate win/loss stats
                    wins = len(df[df["pnl"] > 0])
                    losses = len(df[df["pnl"] <= 0])
                    total = wins + losses
                    
                    if total > 0:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Win Rate", f"{wins/total:.1%}")
                        col2.metric("Total Trades", total)
                        col3.metric("Wins", wins)
                        col4.metric("Losses", losses)
                    # Calculate current capital
                        if "capital_after" in df.columns:
                            current_capital = df["capital_after"].iloc[-1]
                            initial_capital = df["capital_after"].iloc[0] - df["pnl"].iloc[0]
                            gain = (current_capital - initial_capital) / initial_capital
                            
                            st.metric("Current Capital", f"${current_capital:.2f}", 
                                     f"{gain:.2%} from initial ${initial_capital:.2f}")
                        
                        # Calculate average win/loss
                        if wins > 0:
                            avg_win = df[df["pnl"] > 0]["pnl"].mean()
                            st.metric("Average Win", f"${avg_win:.2f}")
                        
                        if losses > 0:
                            avg_loss = df[df["pnl"] <= 0]["pnl"].mean()
                            st.metric("Average Loss", f"${avg_loss:.2f}")
                        
                        # Calculate profit factor
                        if losses > 0 and wins > 0:
                            total_wins = df[df["pnl"] > 0]["pnl"].sum()
                            total_losses = abs(df[df["pnl"] <= 0]["pnl"].sum())
                            if total_losses > 0:
                                profit_factor = total_wins / total_losses
                                st.metric("Profit Factor", f"{profit_factor:.2f}")
                    
                    # Group by risk_profile if available
                    if "risk_profile" in df.columns:
                        st.subheader("Performance by Risk Profile")
                        grouped = df.groupby("risk_profile").agg({
                            "pnl": ["sum", "mean", "count"],
                            "trade_id": "count"
                        })
                        
                        st.dataframe(grouped)
            else:
                st.info("No trade results available yet.")
        except Exception as e:
            st.error(f"Error loading trade results: {e}")
    else:
        st.info("No performance data available yet. Complete some trades first.")

elif view == "API Documentation":
    st.subheader("🔌 BEAST API Documentation")
    
    st.write("""
    The BEAST engine provides a REST API for remote control and monitoring. 
    The API is available at `http://your-server-ip:5000/api/`.
    
    ### Authentication
    
    All API endpoints require authentication using an API key. The API key should be included:
    - In the request body as `api_key` for POST requests
    - In the query parameters as `api_key` for GET requests
    
    ### Available Endpoints
    
    #### Start BEAST Engine
    ```
    POST /api/start
    {
        "api_key": "your_secure_api_key"
    }
    ```
    
    #### Stop BEAST Engine
    ```
    POST /api/stop
    {
        "api_key": "your_secure_api_key"
    }
    ```
    
    #### Get BEAST Status
    ```
    GET /api/status?api_key=your_secure_api_key
    ```
    
    #### Get Recent Trades
    ```
    GET /api/trades?api_key=your_secure_api_key
    ```
    
    ### Example Usage (Python)
    ```python
    import requests
    
    API_KEY = "your_secure_api_key"
    BASE_URL = "http://your-server-ip:5000/api"
    
    # Start BEAST
    response = requests.post(f"{BASE_URL}/start", json={"api_key": API_KEY})
    print(response.json())
    
    # Get status
    response = requests.get(f"{BASE_URL}/status", params={"api_key": API_KEY})
    print(response.json())
    ```
    """)
    
    st.info(f"The API is currently running on port {API_PORT}. The API key is defined in your environment variables or defaults to 'your_secure_api_key'.")

# Auto-refresh the app
if auto_refresh:
    time_diff = (datetime.now() - st.session_state.last_update).total_seconds()
    if time_diff >= refresh_rate:
        st.session_state.last_update = datetime.now()
        st.rerun()