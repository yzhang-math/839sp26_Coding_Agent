#!/usr/bin/env python3
"""Main script to generate and analyze sales data."""

import generator
import analyzer

if __name__ == '__main__':
    generator.write_csv()
    analyzer.analyze_sales_data()