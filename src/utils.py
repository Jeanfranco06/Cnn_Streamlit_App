"""
Módulo de utilidades para el proyecto CNN CIFAR-10
"""

import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import tensorflow as tf


class ExperimentTracker:
    """
    Clase para rastrear experimentos y guardar configuraciones
    """

    def __init__(self, base_dir: str = "experiments"):
        """
        Inicializa el rastreador de experimentos

        Args:
            base_dir: Directorio base para guardar experimentos
        """
        self.base_dir = base_dir
        self.current_experiment = None
        os.makedirs(base_dir, exist_ok=True)

    def start_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Inicia un nuevo experimento

        Args:
            experiment_name: Nombre del experimento
            config: Configuración del experimento

        Returns:
            ID único del experimento
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"

        experiment_dir = os.path.join(self.base_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)

        self.current_experiment = experiment_id

        # Guardar configuración
        config_path = os.path.join(experiment_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"Experimento '{experiment_id}' iniciado")
        print(f"Directorio: {experiment_dir}")

        return experiment_id

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Registra métricas del experimento

        Args:
            metrics: Diccionario con métricas
            step: Paso/época actual (opcional)
        """
        if self.current_experiment is None:
            raise ValueError("No hay experimento activo. Use start_experiment() primero")

        experiment_dir = os.path.join(self.base_dir, self.current_experiment)
        metrics_file = os.path.join(experiment_dir, "metrics.json")

        # Cargar métricas existentes si las hay
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r', encoding='utf-8') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        # Agregar timestamp y step
        entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }

        # Usar step como key si está disponible, sino usar timestamp
        key = str(step) if step is not None else datetime.now().strftime("%H%M%S")
        all_metrics[key] = entry

        # Guardar métricas actualizadas
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    def save_model(self, model, model_name: str = "model"):
        """
        Guarda un modelo en el directorio del experimento actual

        Args:
            model: Modelo a guardar
            model_name: Nombre del archivo del modelo
        """
        if self.current_experiment is None:
            raise ValueError("No hay experimento activo")

        experiment_dir = os.path.join(self.base_dir, self.current_experiment)
        model_path = os.path.join(experiment_dir, f"{model_name}.keras")

        model.save(model_path)
        print(f"Modelo guardado en: {model_path}")

        return model_path

    def save_history(self, history: Dict[str, Any], filename: str = "training_history.pkl"):
        """
        Guarda el historial de entrenamiento

        Args:
            history: Historial de entrenamiento
            filename: Nombre del archivo
        """
        if self.current_experiment is None:
            raise ValueError("No hay experimento activo")

        experiment_dir = os.path.join(self.base_dir, self.current_experiment)
        history_path = os.path.join(experiment_dir, filename)

        with open(history_path, 'wb') as f:
            pickle.dump(history, f)

        print(f"Historial guardado en: {history_path}")

    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.pkl"):
        """
        Guarda resultados de evaluación

        Args:
            results: Resultados de evaluación
            filename: Nombre del archivo
        """
        if self.current_experiment is None:
            raise ValueError("No hay experimento activo")

        experiment_dir = os.path.join(self.base_dir, self.current_experiment)
        results_path = os.path.join(experiment_dir, filename)

        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        print(f"Resultados guardados en: {results_path}")

    def end_experiment(self):
        """
        Finaliza el experimento actual
        """
        if self.current_experiment:
            print(f"Experimento '{self.current_experiment}' finalizado")
            self.current_experiment = None
        else:
            print("No hay experimento activo para finalizar")

    def list_experiments(self) -> List[str]:
        """
        Lista todos los experimentos guardados

        Returns:
            Lista de IDs de experimentos
        """
        if not os.path.exists(self.base_dir):
            return []

        return [d for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))]

    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Carga la configuración y resultados de un experimento

        Args:
            experiment_id: ID del experimento

        Returns:
            Diccionario con la información del experimento
        """
        experiment_dir = os.path.join(self.base_dir, experiment_id)

        if not os.path.exists(experiment_dir):
            raise ValueError(f"Experimento '{experiment_id}' no encontrado")

        experiment_data = {'experiment_id': experiment_id}

        # Cargar configuración
        config_path = os.path.join(experiment_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                experiment_data['config'] = json.load(f)

        # Cargar métricas
        metrics_path = os.path.join(experiment_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                experiment_data['metrics'] = json.load(f)

        # Cargar historial si existe
        history_path = os.path.join(experiment_dir, "training_history.pkl")
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                experiment_data['history'] = pickle.load(f)

        # Cargar resultados si existen
        results_path = os.path.join(experiment_dir, "evaluation_results.pkl")
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                experiment_data['results'] = pickle.load(f)

        return experiment_data


class ModelInspector:
    """
    Clase para inspeccionar y analizar modelos
    """

    @staticmethod
    def get_model_info(model) -> Dict[str, Any]:
        """
        Obtiene información detallada del modelo

        Args:
            model: Modelo de TensorFlow/Keras

        Returns:
            Diccionario con información del modelo
        """
        info = {
            'Tipo de Modelo': type(model).__name__,
            'Forma de Entrada': str(model.input_shape) if hasattr(model, 'input_shape') else 'N/A',
            'Forma de Salida': str(model.output_shape) if hasattr(model, 'output_shape') else 'N/A',
            'Parámetros Entrenables': 0,
            'Parámetros No Entrenables': 0,
            'Total de Parámetros': 0
        }

        if hasattr(model, 'count_params'):
            info['Total de Parámetros'] = int(model.count_params())

        if hasattr(model, 'trainable_variables'):
            info['Parámetros Entrenables'] = int(sum([tf.size(var).numpy() for var in model.trainable_variables]))
            info['Parámetros No Entrenables'] = int(sum([tf.size(var).numpy() for var in model.non_trainable_variables]))

        return info

    @staticmethod
    def analyze_layer_activations(model, X_sample: np.ndarray, layer_names: List[str] = None) -> Dict[str, Any]:
        """
        Analiza las activaciones de capas específicas

        Args:
            model: Modelo entrenado
            X_sample: Muestra de datos de entrada
            layer_names: Nombres de capas a analizar (opcional)

        Returns:
            Diccionario con análisis de activaciones
        """
        activations = {}

        # Crear modelo intermediario para obtener activaciones
        if layer_names is None:
            # Obtener todas las capas convolucionales
            layer_names = [layer.name for layer in model.layers
                          if isinstance(layer, tf.keras.layers.Conv2D)]

        for layer_name in layer_names:
            try:
                intermediate_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.get_layer(layer_name).output
                )
                layer_output = intermediate_model.predict(X_sample, verbose=0)

                activations[layer_name] = {
                    'shape': layer_output.shape,
                    'mean_activation': np.mean(layer_output),
                    'std_activation': np.std(layer_output),
                    'min_activation': np.min(layer_output),
                    'max_activation': np.max(layer_output)
                }

            except Exception as e:
                print(f"Error analizando capa {layer_name}: {e}")
                continue

        return activations

    @staticmethod
    def plot_model_architecture(model, save_path: Optional[str] = None):
        """
        Visualiza la arquitectura del modelo

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=save_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                dpi=96
            )
            if save_path:
                print(f"Arquitectura del modelo guardada en: {save_path}")
        except ImportError:
            print("Graphviz no está instalado. No se puede generar el diagrama de arquitectura.")
        except Exception as e:
            print(f"Error generando diagrama de arquitectura: {e}")


class DataVisualizer:
    """
    Clase para visualizaciones avanzadas de datos
    """

    @staticmethod
    def plot_image_grid(images: np.ndarray, labels: np.ndarray,
                       class_names: List[str], grid_size: tuple = (5, 5),
                       save_path: Optional[str] = None):
        """
        Visualiza un grid de imágenes con sus etiquetas

        Args:
            images: Array de imágenes
            labels: Array de etiquetas
            class_names: Nombres de las clases
            grid_size: Tamaño del grid (filas, columnas)
            save_path: Ruta para guardar la imagen (opcional)
        """
        fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 15))
        axes = axes.ravel()

        num_images = min(len(images), grid_size[0] * grid_size[1])

        for i in range(num_images):
            axes[i].imshow(images[i])
            axes[i].set_title(f'{class_names[labels[i]]}', fontsize=10)
            axes[i].axis('off')

        # Ocultar axes no utilizados
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grid de imágenes guardado en: {save_path}")

        plt.show()

    @staticmethod
    def plot_feature_maps(activations: np.ndarray, layer_name: str,
                         num_filters: int = 8, save_path: Optional[str] = None):
        """
        Visualiza los feature maps de una capa convolucional

        Args:
            activations: Activaciones de la capa
            layer_name: Nombre de la capa
            num_filters: Número de filtros a mostrar
            save_path: Ruta para guardar la imagen (opcional)
        """
        # Tomar la primera imagen del batch
        if len(activations.shape) == 4:
            activations = activations[0]

        num_filters = min(num_filters, activations.shape[-1])

        # Calcular grid size
        grid_cols = int(np.ceil(np.sqrt(num_filters)))
        grid_rows = int(np.ceil(num_filters / grid_cols))

        fig, axes = plt.subplots(grid_rows, grid_cols,
                                figsize=(3*grid_cols, 3*grid_rows))

        if grid_rows == 1:
            axes = axes.reshape(1, -1)
        elif grid_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_filters):
            row = i // grid_cols
            col = i % grid_cols

            if grid_rows == 1:
                ax = axes[0, col]
            elif grid_cols == 1:
                ax = axes[row, 0]
            else:
                ax = axes[row, col]

            feature_map = activations[:, :, i]
            ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Filter {i+1}', fontsize=10)
            ax.axis('off')

        # Ocultar axes no utilizados
        total_subplots = grid_rows * grid_cols
        for i in range(num_filters, total_subplots):
            row = i // grid_cols
            col = i % grid_cols
            if grid_rows == 1:
                axes[0, col].axis('off')
            elif grid_cols == 1:
                axes[row, 0].axis('off')
            else:
                axes[row, col].axis('off')

        plt.suptitle(f'Feature Maps - {layer_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature maps guardados en: {save_path}")

        plt.show()


class PerformanceAnalyzer:
    """
    Clase para analizar el rendimiento del modelo
    """

    @staticmethod
    def analyze_prediction_confidence(predictions_proba: np.ndarray,
                                    true_labels: np.ndarray) -> Dict[str, Any]:
        """
        Analiza la confianza de las predicciones

        Args:
            predictions_proba: Probabilidades predichas
            true_labels: Etiquetas verdaderas

        Returns:
            Diccionario con análisis de confianza
        """
        # Obtener la probabilidad máxima para cada predicción
        max_probs = np.max(predictions_proba, axis=1)
        predictions = np.argmax(predictions_proba, axis=1)

        # Separar predicciones correctas e incorrectas
        correct_mask = predictions == true_labels
        incorrect_mask = ~correct_mask

        analysis = {
            'overall_confidence': {
                'mean': np.mean(max_probs),
                'std': np.std(max_probs),
                'min': np.min(max_probs),
                'max': np.max(max_probs)
            },
            'correct_predictions': {
                'mean_confidence': np.mean(max_probs[correct_mask]),
                'count': np.sum(correct_mask)
            },
            'incorrect_predictions': {
                'mean_confidence': np.mean(max_probs[incorrect_mask]),
                'count': np.sum(incorrect_mask)
            }
        }

        return analysis

    @staticmethod
    def find_hard_examples(predictions_proba: np.ndarray, true_labels: np.ndarray,
                          num_examples: int = 10) -> Dict[str, Any]:
        """
        Encuentra los ejemplos más difíciles de clasificar

        Args:
            predictions_proba: Probabilidades predichas
            true_labels: Etiquetas verdaderas
            num_examples: Número de ejemplos a retornar

        Returns:
            Diccionario con ejemplos difíciles
        """
        predictions = np.argmax(predictions_proba, axis=1)
        max_probs = np.max(predictions_proba, axis=1)

        # Encontrar predicciones incorrectas con alta confianza
        incorrect_mask = predictions != true_labels
        incorrect_indices = np.where(incorrect_mask)[0]
        incorrect_confidences = max_probs[incorrect_mask]

        # Ordenar por confianza descendente
        sorted_indices = incorrect_indices[np.argsort(incorrect_confidences)[::-1]]

        hard_examples = {
            'indices': sorted_indices[:num_examples],
            'confidences': max_probs[sorted_indices[:num_examples]],
            'true_labels': true_labels[sorted_indices[:num_examples]],
            'predicted_labels': predictions[sorted_indices[:num_examples]]
        }

        return hard_examples


def save_experiment_summary(experiment_data: Dict[str, Any], save_path: str):
    """
    Guarda un resumen del experimento en formato markdown

    Args:
        experiment_data: Datos del experimento
        save_path: Ruta donde guardar el resumen
    """
    summary = []
    summary.append("# Resumen del Experimento")
    summary.append(f"**ID:** {experiment_data['experiment_id']}")
    summary.append("")

    if 'config' in experiment_data:
        summary.append("## Configuración")
        config = experiment_data['config']
        for key, value in config.items():
            if isinstance(value, (list, dict)):
                summary.append(f"- **{key}:** {json.dumps(value, indent=2, ensure_ascii=False)}")
            else:
                summary.append(f"- **{key}:** {value}")
        summary.append("")

    if 'results' in experiment_data:
        summary.append("## Resultados de Evaluación")
        results = experiment_data['results']
        summary.append(f"- **Accuracy:** {results.get('accuracy', 'N/A'):.4f}")
        summary.append(f"- **Precision:** {results.get('precision', 'N/A'):.4f}")
        summary.append(f"- **Recall:** {results.get('recall', 'N/A'):.4f}")
        summary.append(f"- **F1-Score:** {results.get('f1_score', 'N/A'):.4f}")
        summary.append("")

    summary_str = "\n".join(summary)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(summary_str)

    print(f"Resumen del experimento guardado en: {save_path}")


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de utilidades para CNN CIFAR-10")
    print("Funciones disponibles:")
    print("- ExperimentTracker: Para rastrear experimentos")
    print("- ModelInspector: Para inspeccionar modelos")
    print("- DataVisualizer: Para visualizaciones avanzadas")
    print("- PerformanceAnalyzer: Para análisis de rendimiento")

    # Crear rastreador de experimentos
    tracker = ExperimentTracker()

    print(f"Experimentos existentes: {tracker.list_experiments()}")
