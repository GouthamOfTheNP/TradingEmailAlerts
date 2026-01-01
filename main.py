#!/usr/bin/env python3
import time
import io
import os
import logging
import smtplib
import pytz
import schedule
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from email.message import EmailMessage
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

STOCKS = ["AAPL", "GOOG", "BAC", "JPM", "CSCO"]
ETFS = ["VOO", "IEFA", "RSST"]
COMMODITIES = ["GLD", "SLV"]
ENERGY = ["XOM"]

TICKERS = STOCKS + ETFS + COMMODITIES + ENERGY

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SIGNIFICANT_PRICE_CHANGE_PCT = 1.5
SIGNIFICANT_SCORE_CHANGE = 2
VOLUME_SPIKE_THRESHOLD = 2.0
VOLUME_CHANGE_THRESHOLD = 1.5

RSI_LOW, RSI_HIGH = 30, 70
STOCH_LOW, STOCH_HIGH = 20, 80
ADX_TREND = 25

hourly_baseline = {}
hourly_email_queue = {}


def get_timeframe_params(ticker):
    if ticker in STOCKS: return "3mo", "1d"
    elif ticker in ETFS: return "6mo", "1d"
    elif ticker in COMMODITIES: return "1y", "1d"
    elif ticker in ENERGY: return "6mo", "1d"
    return "3mo", "1d"


def get_prediction_label(score):
    if score >= 5: return "STRONG BUY", "#27ae60"
    elif score >= 3: return "BUY", "#2ecc71"
    elif score >= 1: return "WEAK BUY", "#abd5bb"
    elif score <= -5: return "STRONG SELL", "#c0392b"
    elif score <= -3: return "SELL", "#e74c3c"
    elif score <= -1: return "WEAK SELL", "#e6b0aa"
    return "NEUTRAL", "#95a5a6"


def generate_ai_summary(ticker_data):
    if not GEMINI_API_KEY:
        logger.warning("No Gemini API Key found. Skipping AI summary.")
        return "AI Summary unavailable (No Key provided)."

    logger.info("Generating AI summary with Gemini...")
    client = genai.Client(api_key=GEMINI_API_KEY)

    context = f"Analysis of {len(ticker_data)} securities showing SIGNIFICANT HOURLY SHIFTS:\n\n"
    for ticker, alerts, metrics, pred, color, score in ticker_data:
        context += f"{ticker} (${metrics['price']:.2f}) - {pred} (Strength: {score}):\n"
        for alert in alerts:
            context += f"  - {alert}\n"
        context += "\n"

    prompt = f"""You are a financial analyst. These stocks have shifted significantly compared to one hour ago.
    Data:
    {context}

    Provide a concise HTML summary (no markdown code blocks, just raw HTML tags like <p>, <ul>, <b>).
    1. Why these specific stocks are moving right now based on the indicators.
    2. The collective sentiment (Bullish/Bearish).
    Keep it under 150 words.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text if response.text else "AI analysis returned empty content."
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return f"AI Summary Error: {str(e)}"


def compute_all_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is not None:
        df["BB_Lower"] = bb.iloc[:, 0]
        df["BB_Upper"] = bb.iloc[:, 2]

    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None:
        df["ADX"] = adx.iloc[:, 0]
        df["DI_plus"] = adx.iloc[:, 1]
        df["DI_minus"] = adx.iloc[:, 2]

    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"].replace(0, np.nan)
    return df


def analyze_ticker(df, ticker):
    alerts = []
    signal_strength = 0
    df = compute_all_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    metrics = {
        'price': last["Close"],
        'rsi': last.get("RSI", np.nan),
        'vol_ratio': last.get("Volume_Ratio", np.nan)
    }

    if pd.notna(last.get("RSI")):
        if last["RSI"] < RSI_LOW:
            alerts.append(f"RSI Oversold ({last['RSI']:.1f})")
            signal_strength += 2
        elif last["RSI"] > RSI_HIGH:
            alerts.append(f"RSI Overbought ({last['RSI']:.1f})")
            signal_strength -= 2

    if pd.notna(last.get("Stoch_K")):
        if last["Stoch_K"] < STOCH_LOW and last["Stoch_D"] < STOCH_LOW:
            alerts.append("Stoch Oversold")
            signal_strength += 1
        elif last["Stoch_K"] > STOCH_HIGH and last["Stoch_D"] > STOCH_HIGH:
            alerts.append("Stoch Overbought")
            signal_strength -= 1

    if pd.notna(last.get("MACD")):
        if last["MACD"] > last["Signal"] and prev["MACD"] <= prev["Signal"]:
            alerts.append("MACD Bullish Cross")
            signal_strength += 2
        elif last["MACD"] < last["Signal"] and prev["MACD"] >= prev["Signal"]:
            alerts.append("MACD Bearish Cross")
            signal_strength -= 2

    if pd.notna(last.get("ADX")) and last["ADX"] > ADX_TREND:
        if last.get("DI_plus", 0) > last.get("DI_minus", 0):
            alerts.append(f"Strong Uptrend (ADX:{last['ADX']:.1f})")
            signal_strength += 2
        else:
            alerts.append(f"Strong Downtrend (ADX:{last['ADX']:.1f})")
            signal_strength -= 2

    if pd.notna(last.get("EMA20")) and pd.notna(last.get("EMA50")):
        if last["EMA20"] > last["EMA50"] and prev["EMA20"] <= prev["EMA50"]:
            alerts.append("Golden Cross")
            signal_strength += 3
        elif last["EMA20"] < last["EMA50"] and prev["EMA20"] >= prev["EMA50"]:
            alerts.append("Death Cross")
            signal_strength -= 3

    prediction, color = get_prediction_label(signal_strength)

    return alerts, metrics, signal_strength, prediction, color


def is_significant_shift(ticker, current_score, current_price, current_pred, current_vol_ratio):
    if ticker not in hourly_baseline:
        return False

    base = hourly_baseline[ticker]

    if current_pred != base['prediction']:
        logger.info(f"{ticker}: Prediction flip detected ({base['prediction']} -> {current_pred})")
        return True

    if abs(current_score - base['score']) >= SIGNIFICANT_SCORE_CHANGE:
        logger.info(f"{ticker}: Significant score shift ({base['score']} -> {current_score})")
        return True

    price_change_pct = abs((current_price - base['price']) / base['price']) * 100
    if price_change_pct >= SIGNIFICANT_PRICE_CHANGE_PCT:
        logger.info(f"{ticker}: Price moved {price_change_pct:.2f}%")
        return True

    base_vol = base.get('vol_ratio', 1.0)
    if current_vol_ratio > VOLUME_SPIKE_THRESHOLD:
        vol_increase = current_vol_ratio / base_vol if base_vol > 0 else current_vol_ratio
        if vol_increase >= VOLUME_CHANGE_THRESHOLD:
            logger.info(f"{ticker}: Volume spike (Baseline: {base_vol:.2f}x, Now: {current_vol_ratio:.2f}x)")
            return True

    return False


def draw_chart(df, ticker):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#f8f9fa')

    ax1.plot(df.index, df["Close"], label="Price", color="#2c3e50", linewidth=1.5)
    if "BB_Upper" in df.columns:
        ax1.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], alpha=0.1, color='purple', label="Bollinger Bands")

    ax1.set_title(f"{ticker} - Analysis", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.plot(df.index, df["RSI"], color="#e74c3c", label="RSI", linewidth=1.5)
    ax2.axhline(70, linestyle="--", alpha=0.5, color='gray')
    ax2.axhline(30, linestyle="--", alpha=0.5, color='gray')
    ax2.fill_between(df.index, 70, 30, color='gray', alpha=0.1)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches='tight')
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


def send_email(subject, ticker_data, timestamp, images, ai_summary):
    if not EMAIL_USER or not EMAIL_PASS:
        logger.error("Email credentials missing. Skipping email.")
        return

    logger.info(f"Preparing email for {len(ticker_data)} tickers...")

    html_body = f"""
    <html>
    <body style="font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px;">
        <div style="max-width: 650px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">

            <div style="background-color: #2c3e50; padding: 20px; text-align: center;">
                <h2 style="color: #ffffff; margin: 0; font-size: 22px;">Market Shift Alert</h2>
                <p style="color: #bdc3c7; margin: 5px 0 0; font-size: 14px;">{timestamp}</p>
            </div>

            <div style="background-color: #e8f4f9; border-left: 5px solid #3498db; margin: 20px; padding: 15px; border-radius: 4px;">
                <h3 style="color: #2980b9; margin-top: 0; font-size: 16px;">🤖 AI Market Analysis</h3>
                <div style="color: #34495e; font-size: 14px; line-height: 1.5;">
                    {ai_summary}
                </div>
            </div>

            <div style="padding: 0 20px 20px;">
    """

    for ticker, alerts, metrics, pred, color, score in ticker_data:
        alerts_html = "".join([f'<li style="margin-bottom: 5px;">{a}</li>' for a in alerts])
        if not alerts_html: alerts_html = "<li>No specific technical alerts (Volume/Price shift only)</li>"

        html_body += f"""
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 20px; overflow: hidden;">
            <div style="background-color: {color}; padding: 10px 15px; display: flex; justify-content: space-between; align-items: center;">
                <span style="color: white; font-weight: bold; font-size: 18px;">{ticker}</span>
                <span style="background-color: rgba(255,255,255,0.2); color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                    {pred}
                </span>
            </div>
            <div style="padding: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 10px;">
                    <span style="color: #7f8c8d; font-size: 12px;">PRICE: <b style="color: #2c3e50; font-size: 14px;">${metrics['price']:.2f}</b></span>
                    <span style="color: #7f8c8d; font-size: 12px;">RSI: <b style="color: #2c3e50; font-size: 14px;">{metrics['rsi']:.1f}</b></span>
                    <span style="color: #7f8c8d; font-size: 12px;">VOL RATIO: <b style="color: #2c3e50; font-size: 14px;">{metrics['vol_ratio']:.1f}x</b></span>
                </div>
                <ul style="color: #555; font-size: 13px; padding-left: 20px; margin: 0;">
                    {alerts_html}
                </ul>
                <div style="margin-top: 15px; text-align: center;">
                    <img src="cid:{ticker}.png" style="width: 100%; max-width: 580px; border-radius: 4px; border: 1px solid #eee;">
                </div>
            </div>
        </div>
        """

    html_body += """
            </div>
            <div style="text-align: center; padding: 20px; color: #95a5a6; font-size: 11px;">
                Automated Python Market Monitor
            </div>
        </div>
    </body>
    </html>
    """

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = TO_EMAIL
    msg.set_content("This email requires an HTML-compatible client.")
    html_part = msg.add_alternative(html_body, subtype="html")

    for filename, img_bytes in images:
        html_part.add_related(img_bytes, maintype='image', subtype='png', cid=f"<{filename}>")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


def check_market_vs_baseline():
    if not is_market_open():
        logger.info("Market is closed. Skipping check.")
        return

    global hourly_email_queue, hourly_baseline
    logger.info("Running market check cycle...")

    for ticker in TICKERS:
        try:
            period, interval = get_timeframe_params(ticker)
            df = yf.download(ticker, period=period, interval=interval, progress=False, multi_level_index=False)

            if df is None or df.empty or len(df) < 50:
                logger.debug(f"Insufficient data for {ticker}")
                continue

            df.dropna(subset=["Close"], inplace=True)

            alerts, metrics, score, prediction, color = analyze_ticker(df, ticker)

            if ticker not in hourly_baseline:
                hourly_baseline[ticker] = {
                    'score': score,
                    'price': metrics['price'],
                    'prediction': prediction,
                    'vol_ratio': metrics.get('vol_ratio', 1.0),
                    'last_df': df.copy()
                }
                continue

            vol = metrics.get('vol_ratio', 1.0)
            if is_significant_shift(ticker, score, metrics['price'], prediction, vol):
                hourly_email_queue[ticker] = (alerts, metrics, df, prediction, color, score)

            hourly_baseline[ticker]['last_df'] = df.copy()

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")


def process_hourly_cycle():
    if not is_market_open():
        logger.info("Market is closed. Skipping hourly cycle.")
        return

    global hourly_email_queue, hourly_baseline

    logger.info(f"Processing hourly cycle. Queue size: {len(hourly_email_queue)}")

    if hourly_email_queue:
        ticker_data = []
        images = []

        for ticker, (alerts, metrics, df, prediction, color, score) in hourly_email_queue.items():
            ticker_data.append((ticker, alerts, metrics, prediction, color, score))
            images.append((f"{ticker}.png", draw_chart(df, ticker)))

        ai_summary = generate_ai_summary(ticker_data)
        subject = f"Market Alert: {', '.join(list(hourly_email_queue.keys())[:3])}" + ("..." if len(hourly_email_queue)>3 else "")
        timestamp = datetime.now().strftime("%I:%M %p")

        send_email(subject, ticker_data, timestamp, images, ai_summary)
    else:
        logger.info("No significant shifts detected this hour.")

    logger.info("Resetting baselines...")
    for ticker in TICKERS:
        if ticker in hourly_baseline and 'last_df' in hourly_baseline[ticker]:
            try:
                df = hourly_baseline[ticker]['last_df']
                _, metrics, score, prediction, _ = analyze_ticker(df, ticker)

                hourly_baseline[ticker] = {
                    'score': score,
                    'price': metrics['price'],
                    'prediction': prediction,
                    'vol_ratio': metrics.get('vol_ratio', 1.0),
                    'last_df': df
                }
            except Exception as e:
                logger.error(f"Failed to reset baseline for {ticker}: {e}")

    hourly_email_queue.clear()
    logger.info("Cycle complete. Waiting for next schedule.")


def is_market_open():
    """Check if US stock market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    if now.weekday() >= 5:
        return False

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now < market_close


def process_weekend_summary():
    """Summarizes the week's performance every Saturday morning."""
    logger.info("Generating Weekend Weekly Summary...")
    weekly_data = []

    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False, multi_level_index=False)
            if df.empty: continue

            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            weekly_change = ((end_price - start_price) / start_price) * 100

            weekly_data.append({
                'ticker': ticker,
                'change': weekly_change,
                'price': end_price
            })
        except Exception as e:
            logger.error(f"Weekly summary failed for {ticker}: {e}")

    if weekly_data:
        weekly_data.sort(key=lambda x: x['change'], reverse=True)
        send_weekend_email(weekly_data)


def send_weekend_email(weekly_data):
    """Sends a simplified text/html table of the week's winners and losers."""
    msg = EmailMessage()
    msg["Subject"] = f"Weekly Market Wrap: {datetime.now().strftime('%Y-W%U')}"
    msg["From"], msg["To"] = EMAIL_USER, TO_EMAIL

    rows = ""
    for d in weekly_data:
        color = "#27ae60" if d['change'] >= 0 else "#c0392b"
        rows += f"<tr><td><b>{d['ticker']}</b></td><td>${d['price']:.2f}</td><td style='color:{color};'>{d['change']:.2f}%</td></tr>"

    html = f"""
    <html><body>
        <h2>Weekly Performance Summary</h2>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <thead><tr><th>Ticker</th><th>Close</th><th>Weekly %</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </body></html>
    """
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)


logger.info("Starting Market Monitor Service...")
logger.info(f"Monitored Tickers: {TICKERS}")

logger.info("Initializing baselines (First Run)...")
check_market_vs_baseline()

schedule.every(1).minutes.do(check_market_vs_baseline)
schedule.every().hour.at(":00").do(process_hourly_cycle)
schedule.every().saturday.at("09:00").do(process_weekend_summary)

while True:
    schedule.run_pending()
    time.sleep(1)
