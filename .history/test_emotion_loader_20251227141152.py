import sys
sys.path.append('src')

from src.emotion_data import EmotionDataLoader

print("Testing EmotionDataLoader...")

try:
    loader = EmotionDataLoader()
    print(f"Data path: {loader.data_path}")
    print(f"Path exists: {loader.data_path}")
    data = loader.load_data()
    print(f"Data loaded: {len(data)} samples")
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
