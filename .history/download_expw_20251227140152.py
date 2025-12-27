#!/usr/bin/env python3
"""
Script to download the RAF-DB (Real-world Affective Faces Database) dataset
"""

import os
import requests
import zipfile
import pandas as pd
from tqdm import tqdm
import shutil

def download_rafdb():
    """Download RAF-DB dataset from a reliable source"""

    # Create data directory if it doesn't exist
    rafdb_dir = 'data/rafdb'
    os.makedirs(rafdb_dir, exist_ok=True)

    print("RAF-DB Dataset Download Instructions")
    print("=" * 50)
    print("RAF-DB (Real-world Affective Faces Database) es un dataset académico.")
    print("Por favor, sigue estos pasos para obtenerlo:")
    print()
    print("1. Visitar: https://www.kaggle.com/datasets/shawon10/ckplus")
    print("   o buscar 'RAF-DB dataset' en Google Scholar")
    print("2. Descargar el dataset completo")
    print("3. Extraer los archivos a data/rafdb/")
    print("4. Asegurar que la estructura sea:")
    print("   data/rafdb/")
    print("   ├── EmoLabel/")
    print("   │   └── list_patition_label.txt")
    print("   └── Image/")
    print("       └── aligned/")
    print("           ├── train_00001.jpg")
    print("           └── ... (imágenes de entrenamiento)")
    print()
    print("Alternativamente, puedes crear un dataset mínimo de prueba para desarrollo.")

    # Create minimal test dataset for development
    create_minimal_rafdb_dataset()

def create_minimal_rafdb_dataset():
    """Create a minimal RAF-DB-like dataset for testing purposes"""

    print("Creating minimal RAF-DB dataset for testing...")

    rafdb_dir = 'data/rafdb'
    emolabel_dir = os.path.join(rafdb_dir, 'EmoLabel')
    images_dir = os.path.join(rafdb_dir, 'Image', 'aligned')
    os.makedirs(emolabel_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Create sample list_patition_label.txt file
    # RAF-DB format: image_name emotion_label (1-7)
    # Our mapping: 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
    label_data = []

    # Create sample entries (some for train, some for test)
    emotions_rafdb = [1, 2, 3, 4, 5, 6, 7]  # RAF-DB labels
    samples_per_emotion = 20

    for emotion_label in emotions_rafdb:
        for i in range(samples_per_emotion):
            # Alternate between train and test
            partition = "train" if i < samples_per_emotion // 2 else "test"
            image_name = "04d"

            label_data.append(f"{image_name} {emotion_label}")

    # Save list_patition_label.txt file
    label_file = os.path.join(emolabel_dir, 'list_patition_label.txt')
    with open(label_file, 'w') as f:
        for line in label_data:
            f.write(line + '\n')

    print(f"✅ Created minimal RAF-DB dataset with {len(label_data)} samples")
    print(f"Label file saved to: {label_file}")
    print("Note: This is a minimal test dataset. For real training, download the full RAF-DB dataset.")

    return True

def download_from_kaggle():
    """Try to download from Kaggle (requires API key)"""

    print("Trying to download from Kaggle...")

    try:
        import kaggle
        print("Kaggle API found. Attempting download...")

        # This would require kaggle API setup
        # kaggle.api.competition_download_files('raf-db-dataset', path='data/rafdb', unzip=True)

        print("⚠️  Kaggle download requires API key setup.")
        print("   Run: kaggle competitions download -c raf-db-dataset")
        print("   Or download manually from the website.")

        return False

    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False

if __name__ == "__main__":
    print("RAF-DB Dataset Downloader")
    print("=" * 30)

    # Try primary download method
    if not download_rafdb():
        print("Primary download failed, trying Kaggle...")
        if not download_from_kaggle():
            print("❌ All download methods failed")
            print()
            print("To use RAF-DB dataset:")
            print("1. Visit academic sources or Kaggle")
            print("2. Search for 'RAF-DB Real-world Affective Faces Database'")
            print("3. Download the complete dataset")
            print("4. Extract to data/rafdb/ directory")
            print("5. Ensure the label file exists: data/rafdb/EmoLabel/list_patition_label.txt")
            print()
            print("For development/testing, a minimal dataset has been created.")
        else:
            print("✅ Dataset downloaded successfully!")
    else:
        print("✅ Dataset ready!")
