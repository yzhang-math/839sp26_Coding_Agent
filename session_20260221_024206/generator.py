import csv
import random
from datetime import datetime, timedelta

products = ["Widget", "Gadget", "Gizmo", "Tool"]
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

def generate_sales_data():
    with open("sales_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Product", "Price", "Quantity"])
        
        for _ in range(100):
            random_days = random.randint(0, (end_date - start_date).days)
            date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
            product = random.choice(products)
            price = round(random.uniform(10, 100), 2)
            quantity = random.randint(1, 10)
            writer.writerow([date, product, price, quantity])