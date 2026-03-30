# Trading Email Alerts

An automated Python service that monitors stocks, ETFs, commodities, and others in real-time. It combines technical analysis on data from YFinance with Generative AI (Google Gemini 2.0 Flash) to detect significant market shifts and send detailed HTML email alerts.

## Features

* **Real-Time Monitoring:** Checks market data every minute during US market hours (9:30 AM - 4:00 PM ET).
* **Smart Filtering:** Only alerts on *significant* events (e.g., Prediction flips, price moves >1.5%, Volume spikes >2.0x).
* **Technical Analysis Engine:**
    * **RSI & Stochastic:** Overbought/Oversold detection.
    * **MACD:** Bullish/Bearish crossovers.
    * **ADX:** Trend strength analysis.
    * **Moving Averages:** EMA 20/50 Golden & Death Crosses.
    * **Bollinger Bands:** Volatility visualization.
* **AI Analyst Integration:** Uses **Google Gemini 2.0 Flash** to generate a narrative summary of *why* the stocks are moving and the collective sentiment.
* **Rich Email Alerts:** Sends HTML emails with:
    * AI-written summary.
    * Color-coded signals (Strong Buy to Strong Sell).
    * Embedded Matplotlib charts (Price + RSI + Bollinger Bands).
* **Weekly Summaries:** Automatically sends a "Weekly Market Wrap" every Saturday morning.

## Prerequisites

* Python 3.8+
* A Gmail account with an **App Password** (for SMTP).
* A Google Cloud Project with the **Gemini API** enabled.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GouthamOfTheNP/TradingEmailAlerts.git
    cd TradingEmailAlerts
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    You must export the following variables in your terminal or add them to a `.env` file (if using `python-dotenv`):

    ```bash
    export EMAIL_USER="your_email@gmail.com"
    export EMAIL_PASS="your_16_char_app_password"
    export GEMINI_API_KEY="your_google_gemini_api_key"
    ```

4.  **Set up the Email Recipient List:** You must set up the email recipient list in order for this software to email people. Set it up in `emails.txt`.

5.  **Run:**
    Once done with setting everything up, feel free to run the program with:
    ```bash
    python main.py
    ```

## Configuration

You can modify the `STOCKS`, `ETFS`, and `COMMODITIES` lists at the top of the script to track your preferred assets:

```python
STOCKS = ["AAPL", "GOOG", "BAC", "JPM", "CSCO"]
ETFS = ["VOO", "IEFA", "RSST"]
COMMODITIES = ["GLD", "SLV"]
```

or you can add different categories and update them in `get_timeframe_params` (example below):

```python
TECH = ["APPL", "GOOG"]
FUNDS = ["BRK.B"]
```

```python
def get_timeframe_params(ticker):
    ...

    elif ticker in TECH: return "1mo", "1d"
    elif ticker in FUNDS: return "2y", "1d"
```

You can, as mentioned above, add emails you want as recipients in `emails.txt`, either manually or through a client/script.

---
Script for easy configuration will come soon.