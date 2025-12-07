#!/usr/bin/env python3
"""
Simple script to generate banking data
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bank_data_generator import main

if __name__ == "__main__":
    # Generate small dataset for testing
    sys.argv = ['generate_data.py', '--customers', '100', '--seed', '42']
    main()