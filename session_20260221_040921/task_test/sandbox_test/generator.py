#!/usr/bin/env python3
"""Generate mock sales data CSV file."""

import csv
import random
from datetime import datetime, timedelta

PRODUCTS = ['Widget A', 'Widget B', 'Gadget X', 'Gadget Y']

def random_date_2024():
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    days_range = (end - start).days
    random_days = random.randint(0, days_range)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')

def generate_sales_data(rows=100):
    return [
        {
            'Date': random_date_2024(),
            'Product': random.choice(PRODUCTS),
            'Price': random.randint(10, 100),
            'Quantity': random.randint(1, 20)
        }
        for _ in range(rows)
    ]

def write_csv(filename='sales_data.csv', rows=100):
    data = generate_sales_data(rows)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Date', 'Product', 'Price', 'Quantity'])
        writer.writeheader()
        writer.writerows(data)
    print(f"Generated {filename} with {rows} rows of sales data.")

if __name__ == '__main__':
    write_csv()