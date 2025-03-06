import yfinance as yf
import pandas as pd
import numpy as np
import talib
import asyncio
import smtplib
import schedule
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import requests

# ---------------------------
# 0. config
# ---------------------------
EMAIL_CONFIG = {
    'sender': 'your_email@gmail.com',  # to
    'password': 'your_app_password',  # gmail app password
    'receiver': 'their_email@gmail.com',  # from
    'smtp_server': 'smtp.gmail.com',  # SMTP server
    'smtp_port': 587  # port
}

INDICATOR_WEIGHTS = {
    'bollinger': 2.0,
    'rsi': 1.5,
    'macd': 1.2,
    'stochastic': 1.0,
    'ichimoku': 2.5,
    'bias': 1.0,
    'adx': 1.5,
    'sar': 1.8,
    'obv': 1.0,
    'ma_cross': 2.0,
    'market_cap': 2e9,
    'history_day': 52,
    'run_interval': 120
}

# ---------------------------
# 1. logger config
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_scanner.log'),
        logging.StreamHandler()
    ]
)

# ---------------------------
# 2. Fetch Nasdaq stock list
# ---------------------------


def fetch_nasdaq_symbols(retries=3):
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=false&exchange=NASDAQ&download=true"
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        'Host': 'api.nasdaq.com',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logging.error(f"HTTP error: {response.status_code}")
                continue
            json_data = response.json()
            rows = json_data['data']['rows']
            symbols = [{'symbol': row['symbol'], 'name': row['name'], 'lastsale': row['lastsale'],
                        'netchange': row['netchange'], 'pctchange': row['pctchange'], 'volume': row['volume'],
                        'marketCap': row['marketCap'], 'country': row['country'], 'ipoyear': row['ipoyear'],
                        'sector': row['sector'], 'industry': row['industry'], 'url': row['url']}
                       for row in rows if row['marketCap'] and float(row['marketCap']) > INDICATOR_WEIGHTS['market_cap']]
            logging.info(f"Fetched {len(symbols)} Nasdaq Symbols Successfully")

            # Save symbols to CSV
            df = pd.DataFrame(symbols)
            df.to_csv('nasdaq_symbols.csv', index=False)

            return symbols
        except Exception as e:
            logging.error(
                f"Fetch Nasdaq Symbols Failed (Attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return []

# ---------------------------
# 3. Asynchronous Stock Data Fetching (Optimized Retry Mechanism)
# ---------------------------


async def get_stock_data(symbol_dict, retries=3):
    for attempt in range(retries):
        try:
            symbol = symbol_dict['symbol']
            market_cap = float(symbol_dict['marketCap'])
            if market_cap < INDICATOR_WEIGHTS['market_cap']:
                return None
            history = yf.download(symbol, period='60d',
                                  interval='1d', progress=False)
            if len(history) < INDICATOR_WEIGHTS['history_day']:
                return None
            return {'symbol': symbol, 'history': history, 'market_cap': market_cap}
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff retry
                continue
            logging.error(f"Failed to fetch {symbol}: {str(e)}")
            return None

# ---------------------------
# 4. Enhanced Indicator Calculation
# ---------------------------


def safe_get(arr, index=-1, default=np.nan):
    """Safely get array element, avoid out-of-bounds"""
    if len(arr) == 0:
        return default
    adjusted_index = index if index >= 0 else len(arr) + index
    if adjusted_index < 0 or adjusted_index >= len(arr):
        return default
    return arr[adjusted_index]


def calculate_ichimoku(high, low, close):
    """
    Calculate Ichimoku Cloud indicators
    Returns: (tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span)
    """
    # Conversion Line (Tenkan-sen): Average of the highest and lowest prices over the last 9 days
    tenkan_high = talib.MAX(high, 9)
    tenkan_low = talib.MIN(low, 9)
    tenkan_sen = (tenkan_high + tenkan_low) / 2

    # Base Line (Kijun-sen): Average of the highest and lowest prices over the last 26 days
    kijun_high = talib.MAX(high, 26)
    kijun_low = talib.MIN(low, 26)
    kijun_sen = (kijun_high + kijun_low) / 2

    # Leading Span A (Senkou Span A): Average of the Conversion Line and Base Line, shifted forward 26 days
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_a = senkou_a.ffill()

    # Leading Span B (Senkou Span B): Average of the highest and lowest prices over the last 52 days, shifted forward 26 days
    senkou_high = talib.MAX(high, 52)
    senkou_low = talib.MIN(low, 52)
    senkou_b = ((senkou_high + senkou_low) / 2).shift(26)
    senkou_b = senkou_b.ffill()

    # Lagging Span (Chikou Span): Closing price shifted back 26 days
    chikou_span = close.shift(-26)
    chikou_span = chikou_span.ffill()

    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span


def calculate_enhanced_indicators(close, high, low, volume):
    # Core indicator calculations
    indicators = {}

    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)

    # Moving Average System
    indicators['ma5'] = talib.SMA(close, 5)
    indicators['ma20'] = talib.SMA(close, 20)
    indicators['ma60'] = talib.SMA(close, 60)

    # Trend indicators
    indicators['macd'], indicators['macd_signal'], _ = talib.MACD(close)
    indicators['adx'] = talib.ADX(high, low, close, 14)
    indicators['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    # Momentum Indicators
    indicators['rsi'] = talib.RSI(close, 14)
    indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close,
                                                               fastk_period=14,
                                                               slowk_period=3,
                                                               slowd_period=3)
    # Volatility Indicators
    indicators['upper_bb'], indicators['middle_bb'], indicators['lower_bb'] = talib.BBANDS(
        close, 20, 2)
    indicators['atr'] = talib.ATR(high, low, close, 14)

    # Volume Indicators
    indicators['obv'] = talib.OBV(close, volume)

    # Manually calculate Ichimoku
    (indicators['ichi_tenkan'],
     indicators['ichi_kijun'],
     indicators['ichi_senkou_a'],
     indicators['ichi_senkou_b'],
     indicators['ichi_chikou']) = calculate_ichimoku(
         high=pd.Series(high),
         low=pd.Series(low),
         close=pd.Series(close)
    )
    # Bias (BIAS)
    indicators['ma6'] = talib.SMA(close, 6)  # 6-day moving average
    indicators['ma12'] = talib.SMA(close, 12)  # 12-day moving average
    indicators['bias_6'] = (close - indicators['ma6']) / \
        indicators['ma6'] * 100
    indicators['bias_12'] = (close - indicators['ma12']
                             ) / indicators['ma12'] * 100

    return indicators

# ---------------------------
# 5. Dynamic Scoring Engine
# ---------------------------


def calculate_dynamic_score(data):
    try:
        history = data['history']
        if len(history) < INDICATOR_WEIGHTS['history_day']:
            return None

        # Convert to NumPy arrays
        close = history['Close'].values.flatten().astype(np.float64)
        high = history['High'].values.flatten().astype(np.float64)
        low = history['Low'].values.flatten().astype(np.float64)
        volume = history['Volume'].values.flatten().astype(np.float64)

        # Calculate all indicators
        indicators = calculate_enhanced_indicators(close, high, low, volume)

        # Safely get indicator values
        latest = {
            'close': safe_get(close, -1),
            'rsi': safe_get(indicators['rsi'], -1),
            'macd_line': safe_get(indicators['macd'], -1),
            'macd_signal': safe_get(indicators['macd_signal'], -1),
            'lower_bb': safe_get(indicators['lower_bb'], -1),
            'stoch_k': safe_get(indicators['stoch_k'], -1),
            'stoch_d': safe_get(indicators['stoch_d'], -1),
            'bias_12': safe_get(indicators['bias_12'], -1),
            'adx': safe_get(indicators['adx'], -1),
            'obv': safe_get(indicators['obv'], -1),
            'ma5': safe_get(indicators['ma5'], -1),
            'ma20': safe_get(indicators['ma20'], -1),
            'ma60': safe_get(indicators['ma60'], -1),
            'sar': safe_get(indicators['sar'], -1),
            'obv': safe_get(indicators['obv'], -1),
            'ichi_senkou_a': safe_get(indicators['ichi_senkou_a'], -1),
            'ichi_senkou_b': safe_get(indicators['ichi_senkou_b'], -1)
        }

        prev = {
            'rsi': safe_get(indicators['rsi'], -2),
            'macd_line': safe_get(indicators['macd'], -2),
            'macd_signal': safe_get(indicators['macd_signal'], -2),
            'stoch_k': safe_get(indicators['stoch_k'], -2),
            'stoch_d': safe_get(indicators['stoch_d'], -2),
            'obv': safe_get(indicators['obv'], -2)
        }

        score = 0

        # 1. Bollinger Bands Rule
        if latest['close'] < latest['lower_bb']:
            score += INDICATOR_WEIGHTS['bollinger']

        # 2. RSI Rule
        if latest['rsi'] < 30 and latest['rsi'] > prev['rsi']:
            score += INDICATOR_WEIGHTS['rsi']
        elif latest['rsi'] > 70 and latest['rsi'] < prev['rsi']:
            score -= INDICATOR_WEIGHTS['rsi'] * 0.5

        # 3. MACD Rule
        if (latest['macd_line'] > latest['macd_signal']) and (prev['macd_line'] <= prev['macd_signal']):
            score += INDICATOR_WEIGHTS['macd']

        # 4. Stochastic Indicator Rule
        if latest['stoch_k'] > latest['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']:
            score += INDICATOR_WEIGHTS['stochastic']

        # 5. Ichimoku Cloud Rule
        if latest['close'] > max(latest['ichi_senkou_a'], latest['ichi_senkou_b']):
            score += INDICATOR_WEIGHTS['ichimoku']
        elif latest['close'] < min(latest['ichi_senkou_a'], latest['ichi_senkou_b']):
            score -= INDICATOR_WEIGHTS['ichimoku'] * 0.5

        # 6. Bias Rule
        if latest['bias_12'] < -8:
            score += INDICATOR_WEIGHTS['bias']

        # 7. ADX Trend Strength Rule
        if latest['adx'] > 25:
            score += INDICATOR_WEIGHTS['adx']

        # 8. Parabolic SAR Rule
        if latest['close'] > latest['sar']:
            score += INDICATOR_WEIGHTS['sar']

        # 9. OBV Volume Verification
        if latest['obv'] > prev['obv'] and latest['close'] > close[-2]:
            score += INDICATOR_WEIGHTS['obv']

        # 10. Moving Average Crossover System
        if latest['ma5'] > latest['ma20'] and latest['ma20'] > latest['ma60']:
            score += INDICATOR_WEIGHTS['ma_cross']

        result = {
            'Symbol': data['symbol'],
            'Score': round(score, 2),
            'Price': latest['close'],
            'RSI': latest['rsi'],
            'MACD': latest['macd_line'] - latest['macd_signal'],
            'ADX': latest['adx'],
            'Market_Cap': f"${data['market_cap']/1e9:.1f}B"
        }
        logging.info(f"Result: {result}")
        return result
    except Exception as e:
        logging.error(f"Enhanced scoring error: {str(e)}")
        return None

# ---------------------------
# 6. Email Notification Module
# ---------------------------


def send_email(df):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['receiver']
        msg['Subject'] = f"Stock Screening Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # 构建HTML内容
        html = f"""<h3>Top 10 Recommended Stocks:</h3>
        {df.to_html(index=False, float_format='%.2f')}
        <p>Screening Criteria: Market Cap > 2 Billion + Technical Indicator Composite Score</p>
        <p>Program Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        """
        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
            server.send_message(msg)
        logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")

# ---------------------------
# 7. Main Task (Integrate All Modules)
# ---------------------------


async def main_task():
    logging.info("Task execution started...")
    try:
        # Fetch stock symbols
        symbols = fetch_nasdaq_symbols()
        if not symbols:
            logging.error("Unable to fetch stock symbols, using fallback list")
            symbols = [{'symbol': sym, 'marketCap': '10000000001'} for sym in [
                'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC']]

        # Asynchronously fetch data
        tasks = [get_stock_data(sym) for sym in symbols[:]]
        stocks_data = await asyncio.gather(*tasks)

        # Process data
        valid_data = [data for data in stocks_data if data is not None]
        scored_stocks = [calculate_dynamic_score(data) for data in valid_data]
        scored_stocks = [s for s in scored_stocks if s is not None]

        # Generate report
        df = pd.DataFrame(scored_stocks)
        if not df.empty:
            df.sort_values('Score', ascending=False, inplace=True)
            top_10 = df.head(10)
            send_email(top_10)
        else:
            logging.warning("No qualifying stocks found")
            send_email(pd.DataFrame(columns=['Symbol', 'Score']))
    except Exception as e:
        logging.error(f"Main task exception: {str(e)}")

# ---------------------------
# 8. Scheduled Task
# ---------------------------


def job():
    asyncio.run(main_task())


def run_scheduler():
    job()
    schedule.every(INDICATOR_WEIGHTS['run_interval']).minutes.do(job)
    logging.info(
        f"Scheduled task started, running every {INDICATOR_WEIGHTS['run_interval']} minutes")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    run_scheduler()
