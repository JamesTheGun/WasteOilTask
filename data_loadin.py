import json
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MONTH_FILES = ["AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER"]


def load_month_data(months: list[str] | None = None) -> pd.DataFrame:
    """
    Load one or more monthly waste oil JSON files into a combined DataFrame.

    Args:
        months: List of month names to load (e.g. ["AUGUST", "SEPTEMBER"]).
                Defaults to all available months: AUGUST, SEPTEMBER, OCTOBER, NOVEMBER.

    Returns:
        A DataFrame with columns:
            facility, date, timeStart, timeEnd, supplierCode,
            suppliedM3, recoveredM3, month
    """
    if months is None:
        months = MONTH_FILES

    frames = []
    for month in months:
        filepath = os.path.join(DATA_DIR, f"{month.upper()}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        with open(filepath, "r") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["month"] = month.capitalize()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    combined["date"] = pd.to_datetime(combined["date"], format="%b %d, %Y")
    combined["timeStart"] = pd.to_datetime(
        combined["date"].dt.strftime("%Y-%m-%d") + " " + combined["timeStart"],
        format="%Y-%m-%d %I:%M:%S %p",
    )
    combined["timeEnd"] = pd.to_datetime(
        combined["date"].dt.strftime("%Y-%m-%d") + " " + combined["timeEnd"],
        format="%Y-%m-%d %I:%M:%S %p",
    )

    return combined
