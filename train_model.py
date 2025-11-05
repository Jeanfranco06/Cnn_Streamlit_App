"""
Script de entrenamiento para el modelo CNN CIFAR-10
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Agregar el directorio src al path
sys.path.append('src')

from data import CIFAR10DataLoader
from model import CIFAR10CNN
from evaluation import ModelEvaluator
from utils import ExperimentTracker, ModelInspector, DataVisualizer

def main():
    """
    Función principal para entrenar el modelo CNN
    """
    parser = argparse.ArgumentParser(description='Entrenar modelo CNN para CIFAR-10')
    parser.add_argument('--model_type', type=str, default='advanced',
                       choices=['basic', 'advanced', 'residual'],
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Tamaño del batch')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Tasa de aprendizaje')
    parser.add_argument('--data_augmentation', action='store_true', default=True,
                       help='Usar aumento de datos')
    parser.add_argument('--experiment_name', type=str, default='cnn_cifar10',
                       help='Nombre del experimento')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Guardar gráficos generados')

    args = parser.parse_args()

    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO CNN CIFAR-10")
    print("=" * 60)
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tipo de modelo: {args.model_type}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Data augmentation: {args.data_augmentation}")
    print("=" * 60)

    # 1. Cargar y preparar datos
    print("\n1. CARGANDO DATOS...")
    data_loader = CIFAR10DataLoader(validation_split=0.1, random_state=42)
    data = data_loader.load_data()

    # Información del dataset
    info = data_loader.get_dataset_info()
    print(f"Dataset CIFAR-10 cargado:")
    print(f"  - Clases: {info['num_classes']}")
    print(f"  - Entrenamiento: {info['train_samples']} muestras")
    print(f"  - Validación: {info['val_samples']} muestras")
    print(f"  - Prueba: {info['test_samples']} muestras")
    print(f"  - Dimensiones: {info['image_shape']}")

    # 2. Inicializar experimento
    print("\n2. INICIALIZANDO EXPERIMENTO...")
    tracker = ExperimentTracker()

    # Configuración del experimento
    config = {
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'data_augmentation': args.data_augmentation,
        'dataset_info': info,
        'timestamp': datetime.now().isoformat()
    }

    experiment_id = tracker.start_experiment(args.experiment_name, config)

    # 3. Crear y configurar el modelo
    print(f"\n3. CONSTRUYENDO MODELO {args.model_type.upper()}...")

    if args.model_type == 'basic':
        model_config = {
            'filters': [32, 64, 128],
            'dropout_rate': 0.5,
            'learning_rate': args.learning_rate
        }
    elif args.model_type == 'advanced':
        model_config = {
            'filters': [64, 128, 256, 512],
            'dropout_rate': 0.3,
            'learning_rate': args.learning_rate
        }
    elif args.model_type == 'residual':
        model_config = {
            'num_blocks': 3,
            'filters': 64,
            'learning_rate': args.learning_rate
        }

    cnn = CIFAR10CNN()
    model = cnn.build_model(args.model_type, **model_config)

    # Mostrar resumen del modelo
    print("\nResumen del modelo:")
    print(cnn.get_model_summary())

    # Información adicional del modelo
    model_info = ModelInspector.get_model_info(model)
    print(f"\nInformación del modelo:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # 4. Entrenar el modelo
    print(f"\n4. ENTRENANDO MODELO ({args.epochs} ÉPOCAS)...")

    history = cnn.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_augmentation=args.data_augmentation,
        save_path=os.path.join("models", f"cifar10_{args.model_type}_model.keras")
    )

    # Guardar historial de entrenamiento
    tracker.save_history(history)

    # 5. Evaluar el modelo
    print("\n5. EVALUANDO MODELO...")

    evaluator = ModelEvaluator(class_names=data['class_names'])

    # Evaluar en datos de prueba
    evaluation_results = evaluator.evaluate_model(
        model, data['X_test'], data['y_test'],
        data['X_train'], data['y_train']
    )

    # Guardar resultados de evaluación
    tracker.save_results(evaluation_results)

    # 6. Generar visualizaciones
    if args.save_plots:
        print("\n6. GENERANDO VISUALIZACIONES...")

        # Crear directorio para gráficos
        plots_dir = os.path.join("experiments", experiment_id, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Curvas de aprendizaje
        cnn.plot_training_history(save_path=os.path.join(plots_dir, "learning_curves.png"))

        # Matriz de confusión
        evaluator.plot_confusion_matrix(
            evaluation_results['confusion_matrix'],
            save_path=os.path.join(plots_dir, "confusion_matrix.png")
        )

        # Matriz de confusión normalizada
        evaluator.plot_confusion_matrix(
            evaluation_results['confusion_matrix'],
            normalize=True,
            save_path=os.path.join(plots_dir, "confusion_matrix_normalized.png")
        )

        # Curvas ROC
        if 'roc_curves' in evaluation_results:
            evaluator.plot_roc_curves(
                evaluation_results['roc_curves'],
                save_path=os.path.join(plots_dir, "roc_curves.png")
            )

        # Curvas Precision-Recall
        if 'precision_recall_curves' in evaluation_results:
            evaluator.plot_precision_recall_curves(
                evaluation_results['precision_recall_curves'],
                save_path=os.path.join(plots_dir, "precision_recall_curves.png")
            )

        # Visualizar algunas predicciones
        num_samples = 20
        indices = np.random.choice(len(data['X_test']), num_samples, replace=False)
        sample_images = data['X_test'][indices]
        true_labels = data['y_test'][indices]
        pred_labels = evaluation_results['predictions'][indices]

        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.ravel()

        for i in range(num_samples):
            axes[i].imshow(sample_images[i])
            true_class = data['class_names'][true_labels[i]]
            pred_class = data['class_names'][pred_labels[i]]
            color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            axes[i].set_title(f'Real: {true_class}\nPred: {pred_class}',
                            color=color, fontsize=9)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "predictions_sample.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Arquitectura del modelo (si es posible)
        try:
            ModelInspector.plot_model_architecture(
                model,
                save_path=os.path.join(plots_dir, "model_architecture.png")
            )
        except:
            print("No se pudo generar el diagrama de arquitectura (graphviz no disponible)")

    # 7. Generar reporte de evaluación
    print("\n7. GENERANDO REPORTE...")

    report = evaluator.generate_evaluation_report(
        evaluation_results,
        save_path=os.path.join("experiments", experiment_id, "evaluation_report.txt")
    )

    print("\n" + "="*60)
    print("REPORTE DE EVALUACIÓN")
    print("="*60)
    print(report)

    # 8. Resumen final
    print("\n8. RESUMEN FINAL")
    print("="*60)
    print(f"Experimento completado: {experiment_id}")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Precision: {evaluation_results['precision']:.4f}")
    print(f"Recall: {evaluation_results['recall']:.4f}")
    print(f"F1-Score: {evaluation_results['f1_score']:.4f}")
    print(f"Modelo guardado en: models/cifar10_{args.model_type}_model.keras")
    print(f"Resultados guardados en: experiments/{experiment_id}/")

    # Finalizar experimento
    tracker.end_experiment()

    print("\n¡Entrenamiento completado exitosamente!")
    print("="*60)

    return experiment_id, evaluation_results


if __name__ == "__main__":
    # Ejecutar entrenamiento
    experiment_id, results = main()

    # Mostrar resultados finales
    print("\nResultados finales:")
    print(f"- Accuracy: {results['accuracy']:.4f}")
    print(f"- Precision: {results['precision']:.4f}")
    print(f"- Recall: {results['recall']:.4f}")
    print(f"- F1-Score: {results['f1_score']:.4f}")
    print(f"\nExperimento: {experiment_id}")
