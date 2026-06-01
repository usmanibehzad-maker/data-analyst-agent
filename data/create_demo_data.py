"""
Creates synthetic demo data for showcasing the agent.
Run this once: python data/create_demo_data.py
"""

import sqlite3
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── SQLite Demo Database ────────────────────────────────────────────────────────
db_path = os.path.join(os.path.dirname(__file__), "sample.sqlite")
conn = sqlite3.connect(db_path)

# Customers table
customers = pd.DataFrame({
    "customer_id": range(1, 501),
    "name": [f"Customer_{i}" for i in range(1, 501)],
    "segment": np.random.choice(["Enterprise", "SMB", "Startup"], 500, p=[0.2, 0.5, 0.3]),
    "country": np.random.choice(["Poland", "Germany", "UK", "France", "Netherlands"], 500),
    "created_at": pd.date_range("2024-01-01", periods=500, freq="12h").strftime("%Y-%m-%d"),
})
customers.to_sql("customers", conn, if_exists="replace", index=False)

# Products table
products = pd.DataFrame({
    "product_id": range(1, 21),
    "name": [f"Product_{chr(65+i)}" for i in range(20)],
    "category": np.random.choice(["SaaS", "Consulting", "Support", "Training"], 20),
    "unit_price": np.round(np.random.uniform(50, 2000, 20), 2),
    "cost": np.round(np.random.uniform(10, 800, 20), 2),
})
products.to_sql("products", conn, if_exists="replace", index=False)

# Orders table — with a deliberate Q1 dip and March anomaly
months = pd.date_range("2025-01-01", "2026-03-31", freq="D")
orders = []
order_id = 1
for date in months:
    month = date.month
    # March 2025: revenue dip (simulate a bad month)
    if date.year == 2025 and month == 3:
        daily_orders = np.random.randint(2, 6)
    elif date.year == 2025 and month in [11, 12]:
        daily_orders = np.random.randint(12, 20)  # holiday spike
    else:
        daily_orders = np.random.randint(5, 15)

    for _ in range(daily_orders):
        customer_id = np.random.randint(1, 501)
        product_id = np.random.randint(1, 21)
        quantity = np.random.randint(1, 10)
        unit_price = products.loc[product_id - 1, "unit_price"]
        total = round(unit_price * quantity, 2)
        status = np.random.choice(["completed", "completed", "completed", "refunded", "pending"], p=[0.75, 0.1, 0.05, 0.05, 0.05])
        orders.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "product_id": product_id,
            "quantity": quantity,
            "total": total,
            "status": status,
            "created_at": date.strftime("%Y-%m-%d"),
        })
        order_id += 1

orders_df = pd.DataFrame(orders)
orders_df.to_sql("orders", conn, if_exists="replace", index=False)

conn.close()
print(f"SQLite database created: {db_path}")
print(f"  customers: {len(customers)} rows")
print(f"  products: {len(products)} rows")
print(f"  orders: {len(orders_df)} rows")

# ── CSV Demo File ───────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "sales_messy.csv")

# Simulate a messy sales file with nulls, mixed types
sales = pd.DataFrame({
    "order_id": range(1001, 2001),
    "customer_name": [f"Client {i}" if i % 20 != 0 else None for i in range(1000)],
    "email": [f"client{i}@example.com" if i % 7 != 0 else None for i in range(1000)],
    "product": np.random.choice(["Product_A", "Product_B", "Product_C", "Product_D"], 1000),
    "region": np.random.choice(["North", "South", "East", "West", None], 1000, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    "revenue": np.round(np.random.exponential(500, 1000), 2),
    "units_sold": np.random.randint(1, 50, 1000),
    "sale_date": pd.date_range("2025-01-01", periods=1000, freq="8h").strftime("%Y-%m-%d"),
    "discount_pct": np.round(np.random.uniform(0, 30, 1000), 1),
    "sales_rep": np.random.choice(["Alice", "Bob", "Carlos", "Diana", "Ethan"], 1000),
})

# Inject some outliers for the agent to find
sales.loc[42, "revenue"] = 99999.99
sales.loc[777, "revenue"] = 0.01
sales.loc[500, "units_sold"] = 999

sales.to_csv(csv_path, index=False)
print(f"\nCSV file created: {csv_path}")
print(f"  rows: {len(sales)}")
print(f"  columns: {len(sales.columns)}")
print(f"  nulls injected: customer_name, email, region")
print(f"  outliers injected: rows 42, 500, 777")
