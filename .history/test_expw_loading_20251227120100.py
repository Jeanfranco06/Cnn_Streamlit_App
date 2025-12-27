#!/usr/bin/env python3
"""Test script for ExpW dataset loading"""

import sys
import os

# Add src to path
sys.path.append('src')

from src.emotion_data import EmotionDataLoader

def test_expw_loading():
    """Test ExpW dataset loading"""
    print("Testing ExpW dataset loading...")
    print("=" * 50)

    loader = EmotionDataLoader(dataset='expw')
    print(f'Data path: {loader.data_path}')

    try:
        # Load raw data
        data = loader.load_data()
        print(f'✅ Successfully loaded {len(data)} samples from ExpW dataset')
        print(f'Sample data:')
        print(data.head())

        # Test preprocessing with small sample
        processed = loader.preprocess_data(max_samples=20)  # Small sample for testing
        print(f'✅ Data preprocessing completed')
        print(f'Train shape: {processed["X_train"].shape}')
        print(f'Test shape: {processed["X_test"].shape}')
        print(f'Validation shape: {processed["X_val"].shape}')

        # Get dataset info
        info = loader.get_dataset_info()
        print("\nDataset info:")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Image shape: {info['image_shape']}")

        print("\n✅ ExpW dataset integration works correctly!")
        return True

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_expw_loading()
    sys.exit(0 if success else 1)
