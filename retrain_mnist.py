#!/usr/bin/env python3
"""
Script para reentrenar los modelos MNIST con mejores parámetros
para mejorar el reconocimiento de dígitos escritos a mano.
"""

import sys
import os
sys.path.append('src')

from src.data import MNISTDataLoader
from src.model import MNISTCNN
from src.evaluation import ModelEvaluator
import numpy as np

def retrain_mnist_models():
    """Reentrena los modelos MNIST con mejores parámetros"""

    print("=" * 80)
    print("🔄 REENTRENAMIENTO DE MODELOS MNIST")
    print("=" * 80)
    print("Entrenando modelos con mejores parámetros para mejorar")
    print("el reconocimiento de dígitos escritos a mano.")
    print("=" * 80)

    # Cargar datos
    print("\n📥 Cargando datos MNIST...")
    data_loader = MNISTDataLoader(validation_split=0.1, random_state=42)
    data = data_loader.load_data()

    # Configuraciones de entrenamiento mejoradas
    training_configs = {
        'basic': {
            'epochs': 20,
            'batch_size': 64,
            'learning_rate': 0.001,
            'model_params': {'filters': [32, 64], 'dropout_rate': 0.25}
        },
        'advanced': {
            'epochs': 25,
            'batch_size': 64,
            'learning_rate': 0.001,
            'model_params': {'filters': [32, 64, 128], 'dropout_rate': 0.3}
        }
    }

    evaluator = ModelEvaluator(class_names=data['class_names'])

    for model_type, config in training_configs.items():
        print(f"\n🚀 Entrenando modelo {model_type.upper()}...")
        print(f"Configuración: {config}")

        # Crear y construir modelo
        cnn = MNISTCNN()
        model = cnn.build_model(model_type, **config['model_params'])

        # Entrenar modelo
        history = cnn.train(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            data_augmentation=True,
            save_path=f"models/mnist/{model_type}_trained.keras"
        )

        # Evaluar modelo
        print(f"\n📊 Evaluando modelo {model_type.upper()}...")
        results = evaluator.evaluate_model(
            cnn.model, data['X_test'], data['y_test'],
            data['X_train'], data['y_train']
        )

        print(f"✅ Modelo {model_type} reentrenado exitosamente!")
        print(f"   Accuracy final: {results['accuracy']:.4f}")
        print(f"   Precision: {results['precision']:.4f}")
        print(f"   Recall: {results['recall']:.4f}")
        print(f"   F1-Score: {results['f1_score']:.4f}")

    print("\n" + "=" * 80)
    print("🎉 ¡REENTRENAMIENTO COMPLETADO!")
    print("=" * 80)
    print("Los modelos han sido reentrenados con mejores parámetros.")
    print("Ahora deberían reconocer mejor los dígitos escritos a mano.")
    print("Reinicia la aplicación Streamlit para usar los nuevos modelos.")
    print("=" * 80)

if __name__ == "__main__":
    retrain_mnist_models()
