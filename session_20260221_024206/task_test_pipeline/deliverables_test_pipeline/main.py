"""Main script that runs the data processing pipeline sequentially."""

import generator
import analyzer

if __name__ == "__main__":
    print("Running data processing pipeline...")
    print("-" * 40)
    
    # Step 1: Generate the CSV data
    print("Step 1: Generating sales data...")
    generator.generate_sales_data()
    print("Generated sales_data.csv with 100 rows.")
    print("-" * 40)
    
    # Step 2: Analyze the CSV data
    print("Step 2: Analyzing sales data...")
    analyzer.calculate_total_revenue()
    print("-" * 40)
    print("Pipeline complete!")