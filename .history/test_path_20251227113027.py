#!/usr/bin/env python3
"""Test script to verify FER2013 data path"""

import os
import sys

# Add src to path
sys.path.append('src')

from src.emotion_data import EmotionDataLoader

def test_path():
    """Test if the data path is correct"""
    loader = EmotionDataLoader()
    print(f"Data path: {loader.data_path}")
    print(f"Path exists: {os.path.exists(loader.data_path)}")

    if os.path.exists(loader.data_path):
        try:
            data = loader.load_data()
            print(f"Data loaded successfully: {len(data)} samples")
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print("Data path does not exist!")

if __name__ == "__main__":
    test_path()
