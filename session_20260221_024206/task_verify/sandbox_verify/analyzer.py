import csv

def calculate_total_revenue():
    total_revenue = 0.0

    with open('sales_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            price = float(row['Price'])
            quantity = int(row['Quantity'])
            total_revenue += price * quantity

    print(f"Total Revenue: {total_revenue}")