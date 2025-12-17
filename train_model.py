"""
Script de entrenamiento para el modelo CNN MNIST
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

from data import MNISTDataLoader
from model import MNISTCNN
from evaluation import ModelEvaluator
from utils import ExperimentTracker, ModelInspector, DataVisualizer

def main():
    """
    Función principal para entrenar el modelo CNN
    """
    parser = argparse.ArgumentParser(description='Entrenar modelo CNN para MNIST')
    parser.add_argument('--model_type', type=str, default='advanced',
                       choices=['basic', 'advanced', 'residual'],
                       help='Tipo de modelo a entrenar')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Tamaño del batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Tasa de aprendizaje')
    parser.add_argument('--data_augmentation', action='store_true', default=True,
                       help='Usar aumento de datos')
    parser.add_argument('--experiment_name', type=str, default='cnn_mnist',
                       help='Nombre del experimento')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Guardar gráficos generados')
    parser.add_argument('--cross_validation', action='store_true', default=False,
                       help='Realizar validación cruzada')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Número de folds para validación cruzada')
    parser.add_argument('--grid_search', action='store_true', default=False,
                       help='Realizar búsqueda de hiperparámetros con GridSearchCV')
    parser.add_argument('--optimize_hyperparams', action='store_true', default=False,
                       help='Optimizar hiperparámetros antes del entrenamiento final')

    args = parser.parse_args()

    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO CNN MNIST")
    print("=" * 60)
    print(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tipo de modelo: {args.model_type}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Data augmentation: {args.data_augmentation}")
    print(f"Cross validation: {args.cross_validation}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Grid search: {args.grid_search}")
    print(f"Optimize hyperparameters: {args.optimize_hyperparams}")
    print("=" * 60)

    # 1. Cargar y preparar datos
    print("\n1. CARGANDO DATOS...")
    data_loader = MNISTDataLoader(validation_split=0.1, random_state=42)
    data = data_loader.load_data()

    # Información del dataset
    info = data_loader.get_dataset_info()
    print(f"Dataset MNIST cargado:")
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
        'cross_validation': args.cross_validation,
        'cv_folds': args.cv_folds,
        'grid_search': args.grid_search,
        'optimize_hyperparams': args.optimize_hyperparams,
        'dataset_info': info,
        'timestamp': datetime.now().isoformat()
    }

    experiment_id = tracker.start_experiment(args.experiment_name, config)

    # 3. Crear y configurar el modelo
    print(f"\n3. CONSTRUYENDO MODELO {args.model_type.upper()}...")

    if args.model_type == 'basic':
        model_config = {
            'filters': [32, 64],
            'dropout_rate': 0.25,
            'learning_rate': args.learning_rate
        }
    elif args.model_type == 'advanced':
        model_config = {
            'filters': [32, 64, 128],
            'dropout_rate': 0.3,
            'learning_rate': args.learning_rate
        }
    elif args.model_type == 'residual':
        model_config = {
            'num_blocks': 2,
            'filters': 32,
            'learning_rate': args.learning_rate
        }

    cnn = MNISTCNN()
    model = cnn.build_model(args.model_type, **model_config)

    # Inicializar evaluador temprano para CV y GridSearch
    evaluator = ModelEvaluator(class_names=data['class_names'])


    # Mostrar resumen del modelo
    print("\nResumen del modelo:")
    print(cnn.get_model_summary())

    # Información adicional del modelo
    model_info = ModelInspector.get_model_info(model)
    print(f"\nInformación del modelo:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # 4. Optimización de hiperparámetros (si se solicita)
    best_params = None
    if args.optimize_hyperparams or args.grid_search:
        print(f"\n4. OPTIMIZANDO HIPERPARÁMETROS...")

        # Definir espacio de búsqueda para GridSearchCV
        param_grid = {
            'learning_rate': [1e-4, 1e-3, 1e-2],
            'batch_size': [32, 64, 128],
            'epochs': [10, 20, 30]
        }

        # Función constructora del modelo para GridSearchCV
        def create_model_for_cv(learning_rate=1e-3, dropout_rate=0.3):
            """Función para crear modelo compatible con GridSearchCV"""
            model_cv = cnn.build_model(args.model_type,
                                     filters=model_config['filters'],
                                     dropout_rate=dropout_rate,
                                     learning_rate=learning_rate)
            return model_cv

        # Realizar GridSearchCV
        grid_results = evaluator.perform_grid_search_cv(
            model_builder_func=create_model_for_cv,
            X=np.concatenate([data['X_train'], data['X_val']]),
            y=np.concatenate([data['y_train'], data['y_val']]),
            param_grid=param_grid,
            cv_folds=args.cv_folds,
            epochs=10,  # Épocas reducidas para CV
            batch_size=64,
            scoring='accuracy'
        )

        best_params = grid_results['best_params']
        print(f"✓ Mejores parámetros encontrados: {best_params}")

        # Actualizar configuración con mejores parámetros
        args.learning_rate = best_params.get('learning_rate', args.learning_rate)
        args.batch_size = best_params.get('batch_size', args.batch_size)
        args.epochs = best_params.get('epochs', args.epochs)

        # Reconstruir modelo con mejores parámetros
        model_config['learning_rate'] = args.learning_rate
        model = cnn.build_model(args.model_type, **model_config)

        # Guardar resultados de GridSearchCV
        tracker.save_results({'grid_search_results': grid_results})

    # 5. Validación cruzada (si se solicita)
    cv_results = None
    if args.cross_validation:
        print(f"\n5. REALIZANDO VALIDACIÓN CRUZADA...")

        # Combinar datos de entrenamiento y validación para CV
        X_cv = np.concatenate([data['X_train'], data['X_val']])
        y_cv = np.concatenate([data['y_train'], data['y_val']])

        cv_results = evaluator.perform_cross_validation(
            model_builder_func=lambda **kwargs: cnn.build_model(args.model_type, **{**model_config, **kwargs}),
            X=X_cv,
            y=y_cv,
            cv_folds=args.cv_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            scoring='accuracy'
        )

        print(f"✓ CV completado - Accuracy promedio: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

        # Guardar resultados de CV
        tracker.save_results({'cv_results': cv_results})

        # Visualizar resultados de CV si hay gráficos habilitados
        if args.save_plots:
            plots_dir = os.path.join("experiments", experiment_id, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            evaluator.plot_cross_validation_results(
                cv_results,
                save_path=os.path.join(plots_dir, "cross_validation_results.png")
            )

    # 6. Entrenar el modelo
    step_num = 6 if (args.optimize_hyperparams or args.grid_search or args.cross_validation) else 4
    print(f"\n{step_num}. ENTRENANDO MODELO ({args.epochs} ÉPOCAS)...")

    history = cnn.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_augmentation=args.data_augmentation,
        save_path=os.path.join("models", f"mnist_{args.model_type}_model.keras")
    )

    # Guardar historial de entrenamiento
    tracker.save_history(history)

    # 7. Evaluar el modelo
    eval_step = 7 if (args.optimize_hyperparams or args.grid_search or args.cross_validation) else 5
    print(f"\n{eval_step}. EVALUANDO MODELO...")

    evaluator = ModelEvaluator(class_names=data['class_names'])

    # Evaluar en datos de prueba
    evaluation_results = evaluator.evaluate_model(
        model, data['X_test'], data['y_test'],
        data['X_train'], data['y_train']
    )

    # Guardar resultados de evaluación
    tracker.save_results(evaluation_results)

    # 8. Generar visualizaciones
    if args.save_plots:
        viz_step = 8 if (args.optimize_hyperparams or args.grid_search or args.cross_validation) else 6
        print(f"\n{viz_step}. GENERANDO VISUALIZACIONES...")

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

    # 9. Generar reporte de evaluación
    report_step = 9 if (args.optimize_hyperparams or args.grid_search or args.cross_validation) else 7
    print(f"\n{report_step}. GENERANDO REPORTE...")

    report = evaluator.generate_evaluation_report(
        evaluation_results,
        save_path=os.path.join("experiments", experiment_id, "evaluation_report.txt")
    )

    print("\n" + "="*60)
    print("REPORTE DE EVALUACIÓN")
    print("="*60)
    print(report)

    # 10. Resumen final
    summary_step = 10 if (args.optimize_hyperparams or args.grid_search or args.cross_validation) else 8
    print(f"\n{summary_step}. RESUMEN FINAL")
    print("="*60)
    print(f"Experimento completado: {experiment_id}")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Precision: {evaluation_results['precision']:.4f}")
    print(f"Recall: {evaluation_results['recall']:.4f}")
    print(f"F1-Score: {evaluation_results['f1_score']:.4f}")
    print(f"Modelo guardado en: models/mnist_{args.model_type}_model.keras")
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
