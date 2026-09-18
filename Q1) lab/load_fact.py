import pandas as pd
import glob, re, psycopg2

files = glob.glob(r".\staged_sales\**\*.parquet", recursive=True)

dfs = []
for f in files:
    m = re.search(r"store_id=([^\\]+)", f)
    df = pd.read_parquet(f)
    df["store_id"] = m.group(1)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df = df[df["line_type"].isin(["SALE","RETURN","DISCOUNT","VOID"])].copy()
df["revenue_inr"] = df["qty"] * df["unit_price"]

conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="annapurna",
    user="admin", password="admin123"
)
cur = conn.cursor()

for _, r in df.iterrows():
    cur.execute("""
        INSERT INTO analytics.fact_sales_line
        (bill_no,line_no,store_id,product_sk,date_key,qty,unit_price,line_type,revenue_inr)
        SELECT %s,%s,%s,p.product_sk,%s,%s,%s,%s,%s
        FROM products p
        WHERE p.product_code=%s
          AND %s::date BETWEEN p.valid_from AND p.valid_to
        LIMIT 1
        ON CONFLICT (bill_no,line_no) DO NOTHING
    """, (
        r.bill_no, int(r.line_no), r.store_id,
        r.business_date, r.qty, r.unit_price,
        r.line_type, r.revenue_inr,
        r.product_code, r.business_date
    ))

conn.commit()

cur.execute("SELECT COUNT(*) FROM analytics.fact_sales_line")
print("Fact rows:", cur.fetchone()[0])

cur.close()
conn.close()