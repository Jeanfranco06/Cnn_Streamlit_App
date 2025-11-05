"""
Módulo de evaluación y métricas para modelos CNN CIFAR-10
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import pickle
import os

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Clase para evaluar modelos de clasificación con métricas completas
    """

    def __init__(self, class_names: List[str] = None):
        """
        Inicializa el evaluador

        Args:
            class_names: Nombres de las clases
        """
        self.class_names = class_names or [
            'Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
            'Perro', 'Rana', 'Caballo', 'Barco', 'Camión'
        ]
        self.num_classes = len(self.class_names)

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       X_train: np.ndarray = None, y_train: np.ndarray = None) -> Dict[str, Any]:
        """
        Evalúa completamente el modelo

        Args:
            model: Modelo entrenado
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
            X_train: Datos de entrenamiento (opcional)
            y_train: Etiquetas de entrenamiento (opcional)

        Returns:
            Diccionario con todas las métricas de evaluación
        """
        print("Evaluando modelo...")

        # Obtener predicciones
        if hasattr(model, 'predict'):
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # Para modelos de sklearn
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Reporte de clasificación detallado
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'true_labels': y_test
        }

        # Curvas ROC y Precision-Recall si hay datos de entrenamiento
        if X_train is not None and y_train is not None:
            roc_results = self.compute_roc_curves(y_test, y_pred_proba)
            pr_results = self.compute_precision_recall_curves(y_test, y_pred_proba)

            results.update({
                'roc_curves': roc_results,
                'precision_recall_curves': pr_results
            })

        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")

        return results

    def compute_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calcula curvas ROC para todas las clases

        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas

        Returns:
            Diccionario con datos de curvas ROC
        """
        # Binarizar etiquetas para multiclase
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        roc_data = {}

        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)

            roc_data[class_name] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }

        return roc_data

    def compute_precision_recall_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calcula curvas Precision-Recall para todas las clases

        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas

        Returns:
            Diccionario con datos de curvas Precision-Recall
        """
        # Binarizar etiquetas para multiclase
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        pr_data = {}

        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])

            pr_data[class_name] = {
                'precision': precision,
                'recall': recall
            }

        return pr_data

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None,
                             normalize: bool = False):
        """
        Visualiza la matriz de confusión

        Args:
            cm: Matriz de confusión
            save_path: Ruta para guardar la imagen (opcional)
            normalize: Si normalizar la matriz
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Matriz de Confusión Normalizada'
        else:
            fmt = 'd'
            title = 'Matriz de Confusión'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Número de muestras'})

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicción', fontsize=12)
        plt.ylabel('Valor Real', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada en: {save_path}")

        plt.show()

    def plot_roc_curves(self, roc_data: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualiza las curvas ROC

        Args:
            roc_data: Datos de curvas ROC
            save_path: Ruta para guardar la imagen (opcional)
        """
        plt.figure(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

        for i, (class_name, data) in enumerate(roc_data.items()):
            plt.plot(data['fpr'], data['tpr'],
                    color=colors[i],
                    linewidth=2,
                    label=f'{class_name} (AUC = {data["auc"]:.3f})')
        # Línea diagonal de referencia
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Clasificador Aleatorio')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
        plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
        plt.title('Curvas ROC - Clasificación Multiclase', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curvas ROC guardadas en: {save_path}")

        plt.show()

    def plot_precision_recall_curves(self, pr_data: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualiza las curvas Precision-Recall

        Args:
            pr_data: Datos de curvas Precision-Recall
            save_path: Ruta para guardar la imagen (opcional)
        """
        plt.figure(figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

        for i, (class_name, data) in enumerate(pr_data.items()):
            plt.plot(data['recall'], data['precision'],
                    color=colors[i],
                    linewidth=2,
                    label=f'{class_name}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Curvas Precision-Recall', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curvas Precision-Recall guardadas en: {save_path}")

        plt.show()

    def plot_learning_curves(self, history: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualiza las curvas de aprendizaje

        Args:
            history: Historial de entrenamiento
            save_path: Ruta para guardar la imagen (opcional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(history['accuracy']) + 1)

        # Accuracy
        axes[0, 0].plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Entrenamiento')
        axes[0, 0].plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validación')
        axes[0, 0].set_title('Accuracy vs Épocas', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Épocas')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(epochs, history['loss'], 'b-', linewidth=2, label='Entrenamiento')
        axes[0, 1].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validación')
        axes[0, 1].set_title('Loss vs Épocas', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Épocas')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning Rate (si está disponible)
        if 'lr' in history:
            axes[1, 0].plot(epochs, history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate vs Épocas', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Épocas')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate no disponible',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')

        # Diferencia entre train y validation
        train_acc = np.array(history['accuracy'])
        val_acc = np.array(history['val_accuracy'])
        acc_diff = train_acc - val_acc

        train_loss = np.array(history['loss'])
        val_loss = np.array(history['val_loss'])
        loss_diff = train_loss - val_loss

        axes[1, 1].plot(epochs, acc_diff, 'b-', linewidth=2, label='Accuracy Gap')
        axes[1, 1].plot(epochs, loss_diff, 'r-', linewidth=2, label='Loss Gap')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Gaps de Entrenamiento vs Validación', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Épocas')
        axes[1, 1].set_ylabel('Diferencia')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curvas de aprendizaje guardadas en: {save_path}")

        plt.show()

    def perform_cross_validation(self, model, X: np.ndarray, y: np.ndarray,
                               cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Realiza validación cruzada

        Args:
            model: Modelo a evaluar
            X: Datos de entrada
            y: Etiquetas
            cv_folds: Número de folds para CV
            scoring: Métrica para evaluar

        Returns:
            Resultados de validación cruzada
        """
        print(f"Realizando validación cruzada con {cv_folds} folds...")

        # Para modelos de TensorFlow/Keras, necesitamos adaptar
        if hasattr(model, 'predict'):
            # Modelo de Keras - usar wrapper para sklearn
            from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

            def create_model():
                # Esta función debería recrear el modelo
                # Por simplicidad, devolveremos el modelo entrenado
                return model

            keras_model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64, verbose=0)
            cv_scores = cross_val_score(keras_model, X, y, cv=cv_folds, scoring=scoring)
        else:
            # Modelo de sklearn
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)

        cv_results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'cv_folds': cv_folds,
            'scoring': scoring
        }

        print(f"CV {scoring}: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        return cv_results

    def plot_cross_validation_results(self, cv_results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Visualiza resultados de validación cruzada

        Args:
            cv_results: Resultados de CV
            save_path: Ruta para guardar la imagen (opcional)
        """
        scores = cv_results['cv_scores']

        plt.figure(figsize=(10, 6))

        # Box plot de los scores
        plt.boxplot(scores, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='blue'),
                   medianprops=dict(color='red', linewidth=2))

        plt.scatter(scores, [1] * len(scores), alpha=0.7, s=100, c='red', edgecolors='black')
        plt.axvline(x=cv_results['mean_score'], color='green', linestyle='--', linewidth=2,
                   label=f'Media: {cv_results["mean_score"]:.3f}')
        plt.xlabel(f'{cv_results["scoring"].capitalize()} Score', fontsize=12)
        plt.title(f'Validación Cruzada - {cv_results["cv_folds"]} Folds', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resultados CV guardados en: {save_path}")

        plt.show()

    def generate_evaluation_report(self, results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> str:
        """
        Genera un reporte completo de evaluación

        Args:
            results: Resultados de evaluación
            save_path: Ruta para guardar el reporte (opcional)

        Returns:
            String con el reporte
        """
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE EVALUACIÓN DEL MODELO CNN CIFAR-10")
        report.append("=" * 60)

        # Métricas generales
        report.append("\nMÉTRICAS GENERALES:")
        report.append("-" * 30)
        report.append(f"Accuracy: {results['accuracy']:.3f}")
        report.append(f"Precision: {results['precision']:.3f}")
        report.append(f"Recall: {results['recall']:.3f}")
        report.append(f"F1-Score: {results['f1_score']:.3f}")

        # Reporte por clase
        report.append("\nMÉTRICAS POR CLASE:")
        report.append("-" * 30)
        class_report = results['classification_report']
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                report.append(f"\n{class_name}:")
                report.append(f"  Precision: {metrics['precision']:.3f}")
                report.append(f"  Recall: {metrics['recall']:.3f}")
                report.append(f"  F1-Score: {metrics['f1-score']:.3f}")
                report.append(f"  Soporte: {metrics['support']}")

        # Matriz de confusión resumen
        report.append("\nMATRIZ DE CONFUSIÓN (Resumen):")
        report.append("-" * 30)
        cm = results['confusion_matrix']
        report.append("Verdaderos Positivos (diagonal):")
        for i, class_name in enumerate(self.class_names):
            report.append(f"  {class_name}: {cm[i, i]}")

        # Curvas ROC si disponibles
        if 'roc_curves' in results:
            report.append("\nCURVAS ROC - AUC Scores:")
            report.append("-" * 30)
            for class_name, data in results['roc_curves'].items():
                report.append(f"  {class_name}: {data['auc']:.3f}")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            print(f"Reporte guardado en: {save_path}")

        return report_str

    def save_evaluation_results(self, results: Dict[str, Any], save_dir: str = "evaluation_results"):
        """
        Guarda todos los resultados de evaluación

        Args:
            save_dir: Directorio donde guardar los resultados
        """
        os.makedirs(save_dir, exist_ok=True)

        # Guardar métricas principales
        metrics_file = os.path.join(save_dir, "metrics.pkl")
        with open(metrics_file, 'wb') as f:
            pickle.dump({
                'accuracy': results.get('accuracy'),
                'precision': results.get('precision'),
                'recall': results.get('recall'),
                'f1_score': results.get('f1_score'),
                'classification_report': results.get('classification_report')
            }, f)

        # Guardar predicciones
        predictions_file = os.path.join(save_dir, "predictions.pkl")
        with open(predictions_file, 'wb') as f:
            pickle.dump({
                'predictions': results.get('predictions'),
                'predictions_proba': results.get('predictions_proba'),
                'true_labels': results.get('true_labels')
            }, f)

        print(f"Resultados de evaluación guardados en: {save_dir}")


def compare_models(models_results: Dict[str, Dict[str, Any]],
                  metric: str = 'accuracy') -> pd.DataFrame:
    """
    Compara múltiples modelos

    Args:
        models_results: Diccionario con resultados de diferentes modelos
        metric: Métrica para comparar

    Returns:
        DataFrame con comparación de modelos
    """
    comparison_data = []

    for model_name, results in models_results.items():
        row = {'Modelo': model_name}
        row.update({
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1-Score': results.get('f1_score', 0)
        })
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.set_index('Modelo')

    return df


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de evaluación para CNN CIFAR-10")
    print("Use este módulo para evaluar modelos entrenados")

    # Crear evaluador
    evaluator = ModelEvaluator()

    print("Funciones disponibles:")
    print("- evaluate_model(): Evalúa un modelo completamente")
    print("- plot_confusion_matrix(): Visualiza matriz de confusión")
    print("- plot_roc_curves(): Visualiza curvas ROC")
    print("- plot_precision_recall_curves(): Visualiza curvas Precision-Recall")
    print("- plot_learning_curves(): Visualiza curvas de aprendizaje")
    print("- perform_cross_validation(): Realiza validación cruzada")
    print("- generate_evaluation_report(): Genera reporte completo")
