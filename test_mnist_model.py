#!/usr/bin/env python3
"""
Script para probar la carga y predicción del modelo MNIST
"""

import sys
import os
import numpy as np

# Agregar src al path
sys.path.append('src')

from src.data import MNISTDataLoader
from src.model import MNISTCNN

def test_mnist_model():
    """Prueba la carga y predicción del modelo MNIST"""

    print("Cargando datos MNIST...")
    data_loader = MNISTDataLoader(validation_split=0.1, random_state=42)
    data = data_loader.load_data()

    print(f"Datos cargados. Forma de X_test: {data['X_test'].shape}")
    print(f"Forma de una imagen: {data['X_test'][0].shape}")
    print(f"Tipo de datos: {data['X_test'].dtype}")
    print(f"Rango de valores: {data['X_test'].min()} - {data['X_test'].max()}")

    # Probar cargar modelo básico
    print("\nProbando modelo básico...")
    model_path_basic = "models/mnist/basic_trained.keras"
    if os.path.exists(model_path_basic):
        try:
            cnn_basic = MNISTCNN()
            cnn_basic.load_model(model_path_basic)
            print("Modelo básico cargado exitosamente")

            # Probar predicción
            test_image = data['X_test'][0:1]  # Una imagen
            print(f"Forma de imagen de prueba: {test_image.shape}")

            predictions = cnn_basic.model.predict(test_image, verbose=0)
            print(f"Predicciones: {predictions}")
            print(f"Forma de predicciones: {predictions.shape}")
            print(f"Predicción más alta: {np.argmax(predictions[0])}")

        except Exception as e:
            print(f"Error con modelo básico: {e}")
    else:
        print(f"Modelo básico no encontrado en {model_path_basic}")

    # Probar cargar modelo avanzado
    print("\nProbando modelo avanzado...")
    model_path_advanced = "models/mnist/advanced_trained.keras"
    if os.path.exists(model_path_advanced):
        try:
            cnn_advanced = MNISTCNN()
            cnn_advanced.load_model(model_path_advanced)
            print("Modelo avanzado cargado exitosamente")

            # Probar predicción
            test_image = data['X_test'][0:1]  # Una imagen
            print(f"Forma de imagen de prueba: {test_image.shape}")

            predictions = cnn_advanced.model.predict(test_image, verbose=0)
            print(f"Predicciones: {predictions}")
            print(f"Forma de predicciones: {predictions.shape}")
            print(f"Predicción más alta: {np.argmax(predictions[0])}")

        except Exception as e:
            print(f"Error con modelo avanzado: {e}")
    else:
        print(f"Modelo avanzado no encontrado en {model_path_advanced}")

    # Probar crear y usar modelo desde cero
    print("\nProbando crear modelo desde cero...")
    try:
        cnn_new = MNISTCNN()
        model = cnn_new.build_model('basic')
        print("Modelo creado exitosamente")

        # Probar predicción sin entrenar
        test_image = data['X_test'][0:1]
        predictions = model.predict(test_image, verbose=0)
        print(f"Predicciones modelo sin entrenar: {predictions[0][:5]}...")  # Solo primeros 5

    except Exception as e:
        print(f"Error creando modelo: {e}")

if __name__ == "__main__":
    test_mnist_model()
