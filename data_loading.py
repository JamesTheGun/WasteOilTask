import json
import os
from datetime import timedelta
from typing import Tuple
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MONTH_FILES = ["AUGUST", "OCTOBER", "NOVEMBER"]

SUPPLIER_CODE_TO_NAME = {
    "dic": "Dick Tracey",
    "har": "Harry Houdini",
    "tom": "Tom Hanks",
}
SUPPLIER_NAME_TO_CODE = {v: k for k, v in SUPPLIER_CODE_TO_NAME.items()}


ALL_SUPPLIERS = [
    "Dick Tracey",
    "Harry Houdini",
    "Tom Hanks",
    "Mary NO_LAST_NAME",
    "Mary Anne",
    "Mary Jane",
    "Mary Therese",
]


def _resolve_supplier(record: dict) -> str | None:
    code = record.get("supplierCode")
    if code and code in SUPPLIER_CODE_TO_NAME:
        return SUPPLIER_CODE_TO_NAME[code]

    name = record.get("supplier")
    if name and name != "N/A":
        return name

    return None


def _parse_newcastle_record(record: dict) -> dict:
    date = pd.to_datetime(record["date"], format="%b %d, %Y")
    time_start = pd.to_datetime(
        date.strftime("%Y-%m-%d") + " " + record["timeStart"],
        format="%Y-%m-%d %I:%M:%S %p",
    )
    time_end = pd.to_datetime(
        date.strftime("%Y-%m-%d") + " " + record["timeEnd"],
        format="%Y-%m-%d %I:%M:%S %p",
    )
    return {
        "facility": record["facility"],
        "date": date.strftime("%Y-%m-%d"),
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "process_time_mins": round((time_end - time_start).total_seconds() / 60, 2),
        "supplier": _resolve_supplier(record),
        "supplied_m3": record["suppliedM3"],
        "recovered_m3": record["recoveredM3"],
    }


def _parse_bundaberg_record(record: dict) -> dict:
    time_start = pd.to_datetime(record["timeStart"], format="%m/%d/%y %I:%M %p")
    h, m = map(int, record["processTime"].split(":"))
    time_end = time_start + timedelta(hours=h, minutes=m)
    return {
        "facility": record["facility"],
        "date": time_start.strftime("%Y-%m-%d"),
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "process_time_mins": round(h * 60 + m, 2),
        "supplier": _resolve_supplier(record),
        "supplied_m3": record["suppliedM3"],
        "recovered_m3": record["recoveredM3"],
    }


def standardise_month_jsons(months: list[str] | None = None) -> None:
    if months is None:
        months = MONTH_FILES

    for month in months:
        filepath = os.path.join(DATA_DIR, f"{month.upper()}.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        with open(filepath, "r") as f:
            records = json.load(f)

        standardised = []
        for record in records:
            if "processTime" in record:
                standardised.append(_parse_bundaberg_record(record))
            else:
                standardised.append(_parse_newcastle_record(record))

        out_path = os.path.join(DATA_DIR, f"{month.upper()}_standardised.json")
        with open(out_path, "w") as f:
            json.dump(standardised, f, indent=4)
        print(f"Written: {out_path}")


def load_month_data(months: list[str] | None = None) -> pd.DataFrame:
    if months is None:
        months = MONTH_FILES

    frames = []
    for month in months:
        filepath = os.path.join(DATA_DIR, f"{month.upper()}_standardised.json")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Standardised file not found: {filepath}\n"
                "Run standardise_month_jsons() first."
            )
        with open(filepath, "r") as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        df["month"] = month.capitalize()
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined["time_start"] = pd.to_numeric(pd.to_datetime(combined["time_start"]))
    combined["time_start_index"] = pd.to_datetime(combined["time_start"])
    combined = combined.sort_values("time_start").set_index(
        "time_start_index", drop=False
    )

    return combined


def generate_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if "recovered_m3" in df.columns and "supplied_m3" in df.columns:
        df["recovery_ratio"] = df["recovered_m3"] / df["supplied_m3"]

    if "time_start_index" in df.columns:
        df["time_start_hour_minute"] = (
            df["time_start_index"].dt.hour + df["time_start_index"].dt.minute / 60
        )

    return df


def load_data_with_derived_features(months: list[str] | None = None) -> pd.DataFrame:
    data = load_month_data(months)
    return generate_derived_features(data)


def load_scheduled_data() -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, "SCHEDUALS.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scheduled file not found: {filepath}")

    with open(filepath, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df = df.rename(columns={"volumeM3": "supplied_m3"})

    df["time_parsed"] = pd.to_datetime(df["time"], format="%I:%M:%S %p")
    df["time_start_hour_minute"] = (
        df["time_parsed"].dt.hour + df["time_parsed"].dt.minute / 60
    )

    df = df[["facility", "date", "supplier", "supplied_m3", "time_start_hour_minute"]]

    return df
