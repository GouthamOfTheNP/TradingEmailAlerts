#!/usr/bin/env python3
import json
import os

CONFIG_FILE = "config.json"
EMAIL_FILE = "emails.txt"


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


def configure_tickers():
    config = load_config()

    tickers_input = input("Enter tickers separated by commas: ").upper()
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    config["tickers"] = tickers

    save_config(config)
    print(f"Tickers set: {tickers}")


def add_tickers():
    config = load_config()

    existing = set(config.get("tickers", []))

    tickers_input = input("Enter tickers to ADD (comma-separated): ").upper()
    new_tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    updated = list(existing.union(new_tickers))
    config["tickers"] = updated

    save_config(config)
    print(f"Updated tickers: {updated}")


def remove_tickers():
    config = load_config()

    existing = set(config.get("tickers", []))

    tickers_input = input("Enter tickers to REMOVE (comma-separated): ").upper()
    remove_list = [t.strip() for t in tickers_input.split(",") if t.strip()]

    updated = [t for t in existing if t not in remove_list]
    config["tickers"] = updated

    save_config(config)
    print(f"Updated tickers: {updated}")


def configure_ticker_settings():
    """Map a ticker to a specific list of timeframes in ticker_map."""
    config = load_config()

    ticker = input("Enter ticker symbol: ").upper().strip()

    timeframes_input = input("Enter timeframes (comma-separated, e.g. 5d,5m): ")
    timeframes = [t.strip() for t in timeframes_input.split(",") if t.strip()]

    if "ticker_map" not in config:
        config["ticker_map"] = {}

    config["ticker_map"][ticker] = timeframes

    config.setdefault("tickers", [])
    if ticker not in config["tickers"]:
        config["tickers"].append(ticker)

    save_config(config)
    print(f"Timeframes saved for {ticker}: {timeframes}")


def configure_defaults():
    """Set the default timeframes list."""
    config = load_config()

    timeframes_input = input("Enter default timeframes (comma-separated, e.g. 5d,5m): ")
    timeframes = [t.strip() for t in timeframes_input.split(",") if t.strip()]

    config["defaults"] = timeframes

    save_config(config)
    print(f"Default timeframes set: {timeframes}")


def get_ticker_timeframes(ticker):
    """Return timeframes for a ticker, falling back to defaults."""
    config = load_config()

    ticker_map = config.get("ticker_map", {})
    defaults = config.get("defaults", [])

    return ticker_map.get(ticker, defaults)


def configure_emails():
    overwrite = input("Overwrite email list? (Y/N): ").lower()

    emails_input = input("Enter email addresses separated by commas: ")
    emails = [e.strip() for e in emails_input.split(",") if e.strip()]

    if overwrite in ["y", "yes"]:
        with open(EMAIL_FILE, "w") as f:
            f.write("\n".join(emails))
        print("Email list overwritten.")
    else:
        existing = []
        if os.path.exists(EMAIL_FILE):
            with open(EMAIL_FILE, "r") as f:
                existing = [line.strip() for line in f if line.strip()]

        combined = list(set(existing + emails))

        with open(EMAIL_FILE, "w") as f:
            f.write("\n".join(combined))

        print("Email list updated.")


def main():
    print("\n=== CONFIG MENU ===")
    print("1. Set tickers")
    print("2. Add tickers")
    print("3. Remove tickers")
    print("4. Configure email list")
    print("5. Configure ticker timeframes")
    print("6. Configure default timeframes")
    print("7. Exit")

    choice = input("Select an option: ").strip()

    if choice == "1":
        configure_tickers()
    elif choice == "2":
        add_tickers()
    elif choice == "3":
        remove_tickers()
    elif choice == "4":
        configure_emails()
    elif choice == "5":
        configure_ticker_settings()
    elif choice == "6":
        configure_defaults()
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()