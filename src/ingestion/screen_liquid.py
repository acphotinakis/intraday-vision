import pandas as pd
from finvizfinance.screener.overview import Overview


def get_liquid_tickers(count=30):
    print(f"Screening for top {count} liquid stocks...")
    f_overview = Overview()

    # Filters: Large Cap + High Volume
    filters_dict = {
        "Market Cap.": "Mega ($200bln and more)",
        "Average Volume": "Over 1M",
    }
    f_overview.set_filter(filters_dict=filters_dict)

    # Get screener results
    df_screener = f_overview.screener_view()

    # Check if screener returned results
    if df_screener is None or df_screener.empty:
        raise ValueError("Finviz screener returned no data. Check your filters.")

    # Clean Volume column: convert to string first, then remove commas
    if "Volume" not in df_screener.columns:
        raise ValueError("Volume column not found in screener results.")

    df_screener["Volume"] = pd.to_numeric(
        df_screener["Volume"].astype(str).str.replace(",", ""), errors="coerce"
    )

    # Drop rows where Volume could not be parsed
    df_screener = df_screener.dropna(subset=["Volume"])

    # Sort by Volume descending and take top N
    top_stocks = df_screener.sort_values(by="Volume", ascending=False).head(count)

    return top_stocks["Ticker"].tolist()


# --- EXECUTION ---
if __name__ == "__main__":
    try:
        liquid_tickers = get_liquid_tickers(30)
        print(f"Top Liquid Tickers: {liquid_tickers}")
    except Exception as e:
        print(f"An error occurred: {e}")
