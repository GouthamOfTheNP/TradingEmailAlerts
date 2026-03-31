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

* Python 3.12+
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

4.  **Configure Script** Use `config.py` to configure the email list and tickers

5.  **Run:**
    Once done with setting everything up, feel free to run the program with:
    ```bash
    python main.py
    ```

## Configuration

Ticker config is stored in `config.json` with the following structure:
```json
{
    "tickers": [
        "AAPL",
        "MSFT",
        "TSLA"
    ],
    "ticker_map": {
        "AAPL": [
            "5d",
            "5m"
        ],
        "MSFT": [
            "5d",
            "5m"
        ]
    },
    "defaults": [
        "5d",
        "5m"
    ]
}
```

| Field | Type | Description |
|---|---|---|
| `tickers` | `list[str]` | All tracked ticker symbols |
| `ticker_map` | `dict[str, list[str]]` | Per-ticker timeframe overrides |
| `defaults` | `list[str]` | Fallback timeframes for tickers not in `ticker_map` |

Email list is stored in `emails.txt`