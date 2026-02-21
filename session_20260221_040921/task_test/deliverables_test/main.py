import csv

# Generate sales_data.csv
data = [
    ["product", "quantity", "price"],
    ["Widget", 10, 25.00],
    ["Gadget", 5, 50.00],
    ["Gizmo", 3, 30.00]
]

with open("sales_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Calculate total revenue
total_revenue = sum(row[1] * row[2] for row in data[1:])
print(f"Total revenue: ${total_revenue:.2f}")