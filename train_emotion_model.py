import sys
sys.path.append('src')
from src.emotion_data import EmotionDataLoader
from src.emotion_model import EmotionClassifier
import numpy as np
import os

def train_improved_emotion_model():
    """Entrenar un modelo mejorado de emociones"""
    print("🚀 Entrenando modelo mejorado de emociones...")

    # Cargar datos
    data_loader = EmotionDataLoader()
    data = data_loader.preprocess_data()

    # Crear modelo mejorado
    classifier = EmotionClassifier()
    classifier.model = classifier.create_model(model_type='basic')  # Usar arquitectura mejorada
    classifier.compile_model(learning_rate=1e-3)  # Learning rate más alto inicialmente

    # Calcular pesos de clase para datos desbalanceados
    class_weights = data_loader.get_class_weights()
    print(f"Pesos de clase: {class_weights}")

    # Entrenar modelo
    history = classifier.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        epochs=50,
        batch_size=64,
        class_weights=class_weights,
        save_path='models/emotion/emotion_model_improved.h5'
    )

    # Evaluar modelo
    results = classifier.evaluate(data['X_test'], data['y_test'])
    print("\n📊 Resultados de evaluación:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")

    # Guardar modelo final
    classifier.save_model('models/emotion/emotion_model_improved.h5')
    print("✅ Modelo mejorado guardado!")

    return classifier, results

if __name__ == "__main__":
    train_improved_emotion_model()
