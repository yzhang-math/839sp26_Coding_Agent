#!/usr/bin/env python3
"""Simple calculator that evaluates basic math expressions."""

import operator
import re

OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}

def evaluate(expression: str) -> float:
    """Evaluate a simple math expression like '2 + 3'."""
    # Find operator
    for op in OPERATORS:
        if op in expression:
            parts = expression.split(op)
            if len(parts) == 2:
                a, b = map(float, parts)
                return OPERATORS[op](a, b)
    raise ValueError("Invalid expression")

if __name__ == "__main__":
    expr = input("Enter expression: ")
    result = evaluate(expr)
    print(result)