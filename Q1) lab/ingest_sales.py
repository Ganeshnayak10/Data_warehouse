from pathlib import Path
import pandas as pd
import re
import hashlib
import shutil

SOURCE = Path("data/sales")
OUTPUT = Path("staged_sales")

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

all_data = []

for file in SOURCE.glob("*.csv"):
    name = file.name

    match = re.match(
        r"SALES_(S\d{2})_(\d{8})(?:__R\d+)?\.csv$",
        name
    )

    if not match:
        continue

    store_id = match.group(1)

    business_date = pd.to_datetime(
        match.group(2),
        format="%Y%m%d"
    )

    if store_id in ["S01", "S02", "S03", "S04", "S05"]:

        df = pd.read_csv(file)

    elif store_id in ["S06", "S07", "S08", "S09"]:

        df = pd.read_csv(file, sep=";")

        df = df.rename(columns={
            "item_code": "product_code",
            "quantity": "qty",
            "rate": "unit_price",
            "type": "line_type",
            "txn_time": "ts"
        })

    else:

        df = pd.read_csv(
            file,
            encoding="utf-8-sig"
        )

        df["ts"] = pd.to_datetime(
            df["ts"],
            unit="s",
            utc=True
        )

    df["store_id"] = store_id
    df["business_date"] = business_date

    df = df[
        [
            "bill_no",
            "line_no",
            "product_code",
            "qty",
            "unit_price",
            "line_type",
            "ts",
            "store_id",
            "business_date"
        ]
    ]

    all_data.append(df)

print(f"Files processed: {len(all_data)}")

sales = pd.concat(
    all_data,
    ignore_index=True
)

print(
    f"Rows before deduplication: "
    f"{len(sales):,}"
)

sales = sales.drop_duplicates(
    subset=["bill_no", "line_no"],
    keep="first"
)

print(
    f"Rows after deduplication: "
    f"{len(sales):,}"
)

sales["line_no"] = pd.to_numeric(
    sales["line_no"],
    errors="coerce"
).astype("Int64")

sales["qty"] = pd.to_numeric(
    sales["qty"],
    errors="coerce"
)

sales["unit_price"] = pd.to_numeric(
    sales["unit_price"],
    errors="coerce"
)

sales["business_date"] = pd.to_datetime(
    sales["business_date"]
)

sales["year"] = sales["business_date"].dt.year
sales["month"] = sales["business_date"].dt.month

sales["ts"] = sales["ts"].astype(str)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

sales.to_parquet(
    OUTPUT,
    engine="pyarrow",
    index=False,
    partition_cols=[
        "year",
        "month",
        "store_id"
    ]
)

check_data = sales.sort_values(
    ["bill_no", "line_no"]
)[
    [
        "bill_no",
        "line_no",
        "product_code",
        "qty",
        "unit_price",
        "line_type"
    ]
]

checksum = hashlib.sha256(
    pd.util.hash_pandas_object(
        check_data,
        index=False
    ).values.tobytes()
).hexdigest()

print()
print("===== INGESTION COMPLETE =====")
print(f"Unique rows : {len(sales):,}")
print(f"Checksum    : {checksum}")
print(f"Output      : {OUTPUT.resolve()}")