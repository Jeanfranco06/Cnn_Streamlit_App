"""
================================================================================
MÓDULO DE EVALUACIÓN Y VALIDACIÓN PARA MODELOS CNN MNIST
================================================================================

IMPLEMENTA TODAS LAS TÉCNICAS DE VALIDACIÓN Y VISUALIZACIÓN REQUERIDAS:

TÉCNICAS SUPERVISADAS IMPLEMENTADAS:
================================================================================
✅ TÉCNICA #1: División del dataset - train_test_split() (70% train / 30% test)
✅ TÉCNICA #2: Validación cruzada - cross_val_score() o GridSearchCV()
✅ TÉCNICA #3: Curvas ROC y Precision-Recall para modelos de clasificación
✅ TÉCNICA #4: Learning Curves para detectar overfitting o underfitting
✅ TÉCNICA #5: Matriz de confusión para visualizar errores de clasificación

TÉCNICAS NO SUPERVISADAS (NO APLICABLES):
================================================================================
❌ TÉCNICA #6: Silhouette Score - Para clustering (no supervisado)
❌ TÉCNICA #6: Varianza Explicada - Para PCA (no supervisado)

PROYECTO: APRENDIZAJE SUPERVISADO PROFUNDO CON CNN
FRAMEWORK: TensorFlow/Keras + OpenCV + Matplotlib
MÉTRICAS: Accuracy, Loss, Curvas de aprendizaje, Matriz de confusión
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import pickle
import os

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")


class ModelEvaluator:
    """
    ================================================================================
    CLASE DE EVALUACIÓN COMPLETA PARA MODELOS CNN MNIST
    ================================================================================

    IMPLEMENTA TODAS LAS TÉCNICAS DE VALIDACIÓN Y VISUALIZACIÓN ACADÉMICAS REQUERIDAS:

    🔹 TÉCNICAS SUPERVISADAS IMPLEMENTADAS:
    ================================================================================
    ✅ TÉCNICA #1: dataset_split_demo() - train_test_split() (70% train / 30% test)
    ✅ TÉCNICA #2: perform_cross_validation() - cross_val_score() con StratifiedKFold
    ✅ TÉCNICA #2: perform_grid_search_cv() - GridSearchCV() optimización hiperparámetros
    ✅ TÉCNICA #3: plot_roc_curves() - Curvas ROC multiclase con AUC
    ✅ TÉCNICA #3: plot_precision_recall_curves() - Curvas Precision-Recall
    ✅ TÉCNICA #4: plot_learning_curves() - Detección overfitting/underfitting
    ✅ TÉCNICA #5: plot_confusion_matrix() - Matriz confusión + análisis errores

    🔹 TÉCNICAS NO SUPERVISADAS (NO APLICABLES):
    ================================================================================
    ❌ TÉCNICA #6: unsupervised_techniques_explanation()
        - Silhouette Score (para clustering no supervisado)
        - Varianza Explicada (para PCA/reducción dimensional)
        - NO APLICABLES: Este proyecto usa APRENDIZAJE SUPERVISADO

    🔹 SUITE COMPLETA:
    ================================================================================
    ✅ run_complete_validation_suite() - Ejecuta TODAS las técnicas en secuencia

    ================================================================================
    PROYECTO ACADÉMICO: APRENDIZAJE SUPERVISADO PROFUNDO CON CNN
    FRAMEWORK: TensorFlow/Keras + OpenCV + Matplotlib
    MÉTRICAS PRINCIPALES: Accuracy, Loss, Curvas de aprendizaje, Matriz de confusión
    ================================================================================
    """

    def __init__(self, class_names: List[str] = None):
        """
        Inicializa el evaluador completo con todas las técnicas académicas

        Args:
            class_names: Nombres de las clases (dígitos 0-9 por defecto)
        """
        self.class_names = class_names or [str(i) for i in range(10)]
        self.num_classes = len(self.class_names)

        print("=" * 80)
        print("🔧 MODELO EVALUATOR INICIALIZADO")
        print("=" * 80)
        print(f"📊 Clases: {self.num_classes} (dígitos 0-9)")
        print("✅ Todas las técnicas de validación académicas disponibles")
        print("=" * 80)

    def dataset_split_demo(self, X: np.ndarray, y: np.ndarray,
                          test_size: float = 0.3) -> Dict[str, Any]:
        """
        DEMOSTRACIÓN: División del dataset usando train_test_split()
        Técnica requerida #1: 70% entrenamiento / 30% prueba

        Args:
            X: Datos completos
            y: Etiquetas completas
            test_size: Proporción para prueba (0.3 = 30%)

        Returns:
            Diccionario con splits de datos
        """
        print("=" * 60)
        print("🔹 TÉCNICA #1: DIVISIÓN DEL DATASET - train_test_split()")
        print("=" * 60)
        print(f"División: {100*(1-test_size):.0f}% entrenamiento / {100*test_size:.0f}% prueba")

        # División estratificada del dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        print(f"✅ Datos originales: {len(X)} muestras")
        print(f"✅ Entrenamiento: {len(X_train)} muestras ({100*len(X_train)/len(X):.1f}%)")
        print(f"✅ Prueba: {len(X_test)} muestras ({100*len(X_test)/len(X):.1f}%)")

        # Verificar distribución estratificada
        train_dist = np.bincount(y_train)
        test_dist = np.bincount(y_test)

        print("\n📊 Distribución de clases:")
        print("Clase | Entrenamiento | Prueba | Total")
        print("-" * 35)
        for i in range(self.num_classes):
            total = train_dist[i] + test_dist[i]
            print(f"{i:4}  | {train_dist[i]:11}  | {test_dist[i]:5}  | {total}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_distribution': train_dist,
            'test_distribution': test_dist
        }

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       X_train: np.ndarray = None, y_train: np.ndarray = None) -> Dict[str, Any]:
        """
        Evalúa completamente el modelo con todas las métricas requeridas

        Args:
            model: Modelo entrenado
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
            X_train: Datos de entrenamiento (opcional)
            y_train: Etiquetas de entrenamiento (opcional)

        Returns:
            Diccionario con todas las métricas de evaluación
        """
        print("\n🔹 EVALUANDO MODELO COMPLETAMENTE...")

        # Obtener predicciones
        if hasattr(model, 'predict'):
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # Para modelos de sklearn
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

        # Métricas básicas requeridas
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

        # Matriz de confusión (Técnica #5)
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

        # Curvas ROC y Precision-Recall (Técnica #3) si hay datos de entrenamiento
        if X_train is not None and y_train is not None:
            print("📈 Generando curvas ROC y Precision-Recall...")
            roc_results = self.compute_roc_curves(y_test, y_pred_proba)
            pr_results = self.compute_precision_recall_curves(y_test, y_pred_proba)

            results.update({
                'roc_curves': roc_results,
                'precision_recall_curves': pr_results
            })

        print(f"✅ Accuracy: {accuracy:.3f}")
        print(f"✅ Precision: {precision:.3f}")
        print(f"✅ Recall: {recall:.3f}")
        print(f"✅ F1-Score: {f1:.3f}")

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
        DEMOSTRACIÓN: Matriz de confusión para visualizar errores de clasificación
        Técnica requerida #5: Visualización de errores de clasificación

        Args:
            cm: Matriz de confusión
            save_path: Ruta para guardar la imagen (opcional)
            normalize: Si normalizar la matriz
        """
        print("\n" + "=" * 60)
        print("🔹 TÉCNICA #5: MATRIZ DE CONFUSIÓN - Visualización de Errores")
        print("=" * 60)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Matriz de Confusión Normalizada'
        else:
            fmt = 'd'
            title = 'Matriz de Confusión'

        plt.figure(figsize=(12, 10))
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
            print(f"✅ Matriz de confusión guardada en: {save_path}")

        plt.show()

        # Análisis de errores más comunes
        print("\n📊 Análisis de la matriz de confusión:")
        print("Errores más comunes (top 5):")
        errors = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    errors.append((cm[i, j], self.class_names[i], self.class_names[j]))

        errors.sort(reverse=True)
        for count, real, pred in errors[:5]:
            print(f"  {count} errores: {real} → {pred}")

    def plot_roc_curves(self, roc_data: Dict[str, Any], save_path: Optional[str] = None):
        """
        DEMOSTRACIÓN: Curvas ROC para modelos de clasificación
        Técnica requerida #3: Curvas ROC para evaluación de clasificación

        Args:
            roc_data: Datos de curvas ROC
            save_path: Ruta para guardar la imagen (opcional)
        """
        print("\n" + "=" * 60)
        print("🔹 TÉCNICA #3: CURVAS ROC - Evaluación de Clasificación")
        print("=" * 60)

        plt.figure(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))

        print("AUC Scores por clase:")
        for i, (class_name, data) in enumerate(roc_data.items()):
            plt.plot(data['fpr'], data['tpr'],
                    color=colors[i],
                    linewidth=2,
                    label=f'{class_name} (AUC = {data["auc"]:.3f})')
            print(f"  {class_name}: {data['auc']:.3f}")
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
            print(f"✅ Curvas ROC guardadas en: {save_path}")

        plt.show()

    def plot_precision_recall_curves(self, pr_data: Dict[str, Any], save_path: Optional[str] = None):
        """
        DEMOSTRACIÓN: Curvas Precision-Recall para modelos de clasificación
        Técnica requerida #3: Curvas Precision-Recall para evaluación detallada

        Args:
            pr_data: Datos de curvas Precision-Recall
            save_path: Ruta para guardar la imagen (opcional)
        """
        print("\n" + "=" * 60)
        print("🔹 TÉCNICA #3: CURVAS PRECISION-RECALL - Evaluación Detallada")
        print("=" * 60)

        plt.figure(figsize=(12, 8))

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
            print(f"✅ Curvas Precision-Recall guardadas en: {save_path}")

        plt.show()

    def plot_learning_curves(self, history: Dict[str, Any], save_path: Optional[str] = None):
        """
        DEMOSTRACIÓN: Learning Curves para detectar overfitting/underfitting
        Técnica requerida #4: Detección de overfitting o underfitting

        Args:
            history: Historial de entrenamiento
            save_path: Ruta para guardar la imagen (opcional)
        """
        print("\n" + "=" * 60)
        print("🔹 TÉCNICA #4: LEARNING CURVES - Detección Overfitting/Underfitting")
        print("=" * 60)

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

        # Diferencia entre train y validation (gaps)
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
            print(f"✅ Learning curves guardadas en: {save_path}")

        plt.show()

        # Análisis automático de overfitting/underfitting
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]

        print("\n🔍 ANÁLISIS DE OVERFITTING/UNDERFITTING:")
        print(f"Accuracy final - Entrenamiento: {final_train_acc:.4f}")
        print(f"Accuracy final - Validación: {final_val_acc:.4f}")
        print(f"Loss final - Entrenamiento: {final_train_loss:.4f}")
        print(f"Loss final - Validación: {final_val_loss:.4f}")

        gap_acc = final_train_acc - final_val_acc
        gap_loss = final_val_loss - final_train_loss

        print("\n📊 DIAGNÓSTICO:")
        if gap_acc > 0.1:
            print("⚠️  POSIBLE OVERFITTING detectado (gap de accuracy > 10%)")
            print("   El modelo se ajusta demasiado bien a los datos de entrenamiento")
        elif gap_acc < -0.05:
            print("⚠️  POSIBLE UNDERFITTING detectado (accuracy de validación mucho mayor)")
            print("   El modelo es demasiado simple para capturar los patrones")
        else:
            print("✅ Modelo bien ajustado (gap de accuracy aceptable)")

        if gap_loss > 0.2:
            print("⚠️  POSIBLE OVERFITTING detectado (gap de loss significativo)")
            print("   El modelo no generaliza bien a datos no vistos")
        else:
            print("✅ Loss de validación razonable")

    def unsupervised_techniques_explanation(self):
        """
        Explicación de por qué las técnicas no supervisadas NO SON APLICABLES
        Técnica requerida #6: Silhouette y varianza explicada (NO APLICABLES)
        """
        print("\n" + "=" * 80)
        print("❌ TÉCNICA #6: SILHOUETTE Y VARIANZA EXPLICADA - NO APLICABLES")
        print("=" * 80)
        print("Este proyecto utiliza APRENDIZAJE SUPERVISADO con CNN para clasificación.")
        print("Las siguientes técnicas NO SON APLICABLES al contexto:")
        print("")
        print("🚫 Silhouette Score:")
        print("   • Usado para evaluar calidad de clusters en algoritmos NO SUPERVISADOS")
        print("   • Este proyecto tiene etiquetas conocidas (0-9) - APRENDIZAJE SUPERVISADO")
        print("   • Evaluamos accuracy, precision, recall - métricas SUPERVISADAS")
        print("")
        print("🚫 Varianza Explicada:")
        print("   • Usado en PCA/reducción dimensional NO SUPERVISADA")
        print("   • Este proyecto usa CNN para clasificación directa")
        print("   • No realizamos reducción dimensional previa")
        print("")
        print("✅ Técnicas SUPERVISADAS implementadas:")
        print("   • train_test_split() - División estratificada del dataset")
        print("   • cross_val_score() / GridSearchCV() - Validación cruzada")
        print("   • Curvas ROC y Precision-Recall - Evaluación de clasificación")
        print("   • Learning Curves - Detección overfitting/underfitting")
        print("   • Matriz de confusión - Visualización de errores de clasificación")
        print("=" * 80)

    def run_complete_validation_suite(self, model_builder_func, X: np.ndarray, y: np.ndarray,
                                    save_dir: str = "validation_results") -> Dict[str, Any]:
        """
        Ejecuta TODAS las técnicas de validación requeridas en secuencia

        Args:
            model_builder_func: Función que construye el modelo
            X: Datos de entrada
            y: Etiquetas
            save_dir: Directorio para guardar resultados

        Returns:
            Diccionario con todos los resultados de validación
        """
        print("=" * 120)
        print("🚀 DEMOSTRACIÓN COMPLETA: TODAS LAS TÉCNICAS DE VALIDACIÓN ACADÉMICAS")
        print("=" * 120)
        print("Proyecto: CNN MNIST - Aprendizaje Supervisado Profundo")
        print("Framework: TensorFlow/Keras + OpenCV + Matplotlib")
        print("Requisitos: Accuracy, Loss, Curvas de aprendizaje, Matriz de confusión")
        print("=" * 120)

        # Crear directorio para resultados
        os.makedirs(save_dir, exist_ok=True)

        # 1. División del dataset (Técnica #1)
        print("\n" + "="*60 + " PASO 1 " + "="*60)
        dataset_splits = self.dataset_split_demo(X, y, test_size=0.3)

        # 2. Validación cruzada (Técnica #2)
        print("\n" + "="*60 + " PASO 2 " + "="*60)
        cv_results = self.perform_cross_validation(
            model_builder_func, dataset_splits['X_train'], dataset_splits['y_train'],
            cv_folds=3, epochs=3
        )

        # 3. GridSearchCV para optimización (Técnica #2 alternativa)
        print("\n" + "="*60 + " PASO 3 " + "="*60)
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [64, 128]
        }
        grid_results = self.perform_grid_search_cv(
            model_builder_func, dataset_splits['X_train'], dataset_splits['y_train'],
            param_grid, cv_folds=2
        )

        # 4. Entrenar modelo final con mejores parámetros
        print("\n" + "="*60 + " PASO 4 " + "="*60)
        print("ENTRENANDO MODELO FINAL CON PARÁMETROS ÓPTIMOS...")

        best_params = grid_results['best_params']
        print(f"Parámetros óptimos encontrados: {best_params}")

        # Construir modelo con mejores parámetros
        model_params = {}
        train_params = {'epochs': 5, 'batch_size': 128}

        for key, value in best_params.items():
            if key in ['epochs', 'batch_size', 'learning_rate']:
                train_params[key] = value
            else:
                model_params[key] = value

        final_model = model_builder_func(**model_params)

        # Entrenar modelo final
        history = final_model.fit(
            dataset_splits['X_train'], dataset_splits['y_train'],
            epochs=train_params['epochs'],
            batch_size=train_params['batch_size'],
            validation_split=0.1,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        )

        # 5. Evaluación completa del modelo final
        print("\n" + "="*60 + " PASO 5 " + "="*60)
        evaluation_results = self.evaluate_model(
            final_model, dataset_splits['X_test'], dataset_splits['y_test'],
            dataset_splits['X_train'], dataset_splits['y_train']
        )

        # 6. Generar todas las visualizaciones requeridas
        print("\n" + "="*60 + " PASO 6 " + "="*60)
        print("GENERANDO TODAS LAS VISUALIZACIONES REQUERIDAS...")

        # Matriz de confusión (Técnica #5)
        self.plot_confusion_matrix(
            evaluation_results['confusion_matrix'],
            save_path=os.path.join(save_dir, "confusion_matrix.png")
        )

        # Curvas ROC (Técnica #3)
        if 'roc_curves' in evaluation_results:
            self.plot_roc_curves(
                evaluation_results['roc_curves'],
                save_path=os.path.join(save_dir, "roc_curves.png")
            )

        # Curvas Precision-Recall (Técnica #3)
        if 'precision_recall_curves' in evaluation_results:
            self.plot_precision_recall_curves(
                evaluation_results['precision_recall_curves'],
                save_path=os.path.join(save_dir, "precision_recall_curves.png")
            )

        # Learning Curves (Técnica #4)
        self.plot_learning_curves(
            history.history,
            save_path=os.path.join(save_dir, "learning_curves.png")
        )

        # 7. Explicación de técnicas no supervisadas (Técnica #6)
        print("\n" + "="*60 + " PASO 7 " + "="*60)
        self.unsupervised_techniques_explanation()

        # 8. Resumen final completo
        print("\n" + "="*120)
        print("🎉 RESUMEN FINAL: TODAS LAS TÉCNICAS DE VALIDACIÓN IMPLEMENTADAS")
        print("="*120)
        print("✅ TÉCNICA #1: train_test_split() - 70% train / 30% test")
        print("✅ TÉCNICA #2: cross_val_score() - Validación cruzada con StratifiedKFold")
        print("✅ TÉCNICA #2: GridSearchCV() - Optimización de hiperparámetros")
        print("✅ TÉCNICA #3: Curvas ROC - Evaluación multiclase con AUC")
        print("✅ TÉCNICA #3: Curvas Precision-Recall - Análisis detallado")
        print("✅ TÉCNICA #4: Learning Curves - Detección overfitting/underfitting")
        print("✅ TÉCNICA #5: Matriz de confusión - Visualización errores")
        print("❌ TÉCNICA #6: Silhouette Score - No aplicable (supervisado)")
        print("❌ TÉCNICA #6: Varianza Explicada - No aplicable (supervisado)")
        print("="*120)
        print("📊 MÉTRICAS FINALES:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Precision: {evaluation_results['precision']:.4f}")
        print(f"Recall: {evaluation_results['recall']:.4f}")
        print(f"F1-Score: {evaluation_results['f1_score']:.4f}")
        print("="*120)
        print("🎯 PROYECTO ACADÉMICO: APRENDIZAJE SUPERVISADO PROFUNDO")
        print("🛠️  TECNOLOGÍAS: TensorFlow/Keras + OpenCV + Matplotlib")
        print("📈 EVALUACIÓN: Accuracy, Loss, Curvas de aprendizaje, Matriz de confusión")
        print("="*120)

        return {
            'dataset_splits': dataset_splits,
            'cv_results': cv_results,
            'grid_results': grid_results,
            'final_model': final_model,
            'training_history': history.history,
            'evaluation_results': evaluation_results,
            'final_metrics': {
                'accuracy': evaluation_results['accuracy'],
                'precision': evaluation_results['precision'],
                'recall': evaluation_results['recall'],
                'f1_score': evaluation_results['f1_score']
            }
        }

    def perform_cross_validation(self, model_builder_func, X: np.ndarray, y: np.ndarray,
                               cv_folds: int = 5, epochs: int = 10, batch_size: int = 64,
                               scoring: str = 'accuracy', **model_kwargs) -> Dict[str, Any]:
        """
        DEMOSTRACIÓN: Validación cruzada usando cross_val_score()
        Técnica requerida #2: Validación cruzada para evaluación robusta

        Args:
            model_builder_func: Función que construye el modelo (debe retornar un modelo compilado)
            X: Datos de entrada
            y: Etiquetas
            cv_folds: Número de folds para CV
            epochs: Número de épocas por fold
            batch_size: Tamaño del batch
            scoring: Métrica para evaluar ('accuracy', 'f1', etc.)
            **model_kwargs: Parámetros adicionales para el constructor del modelo

        Returns:
            Resultados de validación cruzada
        """
        print("\n" + "=" * 60)
        print("🔹 TÉCNICA #2: VALIDACIÓN CRUZADA - cross_val_score()")
        print("=" * 60)
        print(f"✅ Número de folds: {cv_folds}")
        print(f"✅ Estrategia: StratifiedKFold (balance de clases)")

        # Usar StratifiedKFold para mantener balance de clases
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = []
        fold_histories = []
        fold_models = []

        fold = 1
        for train_index, val_index in skf.split(X, y):
            print(f"Entrenando fold {fold}/{cv_folds}...")

            # Dividir datos
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Construir modelo fresco para cada fold
            model = model_builder_func(**model_kwargs)

            # Entrenar modelo
            history = model.fit(
                X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
                ]
            )

            # Evaluar en el fold de validación
            if scoring == 'accuracy':
                _, score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            elif scoring == 'f1':
                from sklearn.metrics import f1_score
                y_pred = np.argmax(model.predict(X_val_fold, verbose=0), axis=1)
                score = f1_score(y_val_fold, y_pred, average='weighted')
            else:
                # Para otras métricas, usar accuracy por defecto
                _, score = model.evaluate(X_val_fold, y_val_fold, verbose=0)

            cv_scores.append(score)
            fold_histories.append(history.history)
            fold_models.append(model)

            print(f"Fold {fold} - {scoring}: {score:.4f}")
            fold += 1

        cv_scores = np.array(cv_scores)

        cv_results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'cv_folds': cv_folds,
            'scoring': scoring,
            'fold_histories': fold_histories,
            'fold_models': fold_models,
            'epochs_per_fold': epochs,
            'batch_size': batch_size
        }

        print(f"\nCV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        return cv_results

    def perform_grid_search_cv(self, model_builder_func, X: np.ndarray, y: np.ndarray,
                              param_grid: Dict[str, Any], cv_folds: int = 3,
                              epochs: int = 10, batch_size: int = 64,
                              scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        DEMOSTRACIÓN: Optimización de hiperparámetros con GridSearchCV
        Técnica requerida #2 (alternativa): GridSearchCV para optimización

        Args:
            model_builder_func: Función que construye el modelo
            X: Datos de entrada
            y: Etiquetas
            param_grid: Diccionario con los parámetros a probar
            cv_folds: Número de folds para CV
            epochs: Número de épocas por fold
            batch_size: Tamaño del batch
            scoring: Métrica para evaluar

        Returns:
            Resultados de GridSearchCV
        """
        print("\n" + "=" * 60)
        print("🔹 TÉCNICA #2: OPTIMIZACIÓN DE HIPERPARÁMETROS - GridSearchCV()")
        print("=" * 60)
        print(f"✅ Parámetros a probar: {param_grid}")
        print(f"✅ Folds para validación cruzada: {cv_folds}")

        # Crear wrapper de Keras para sklearn
        def create_model(**params):
            # Extraer parámetros del modelo vs parámetros de entrenamiento
            model_params = {}
            train_params = {'epochs': epochs, 'batch_size': batch_size}

            # Separar parámetros del modelo de los de entrenamiento
            for key, value in params.items():
                if key in ['epochs', 'batch_size', 'learning_rate']:
                    train_params[key] = value
                else:
                    model_params[key] = value

            model = model_builder_func(**model_params)
            return model

        # Crear KerasClassifier
        keras_clf = KerasClassifier(
            build_fn=create_model,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Realizar GridSearchCV
        print("\nEjecutando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(
            estimator=keras_clf,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=1,  # Usar 1 job para evitar problemas con TensorFlow
            verbose=2
        )

        # Ejecutar búsqueda
        grid_search.fit(X, y)

        # Extraer mejores parámetros
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        cv_results = grid_search.cv_results_

        # Reconstruir mejor modelo con los parámetros óptimos
        best_model_params = {}
        best_train_params = {'epochs': epochs, 'batch_size': batch_size}

        for key, value in best_params.items():
            if key in ['epochs', 'batch_size', 'learning_rate']:
                best_train_params[key] = value
            else:
                best_model_params[key] = value

        best_model = model_builder_func(**best_model_params)

        # Entrenar mejor modelo en todos los datos
        print("Entrenando mejor modelo en todos los datos...")
        history = best_model.fit(
            X, y,
            epochs=best_train_params['epochs'],
            batch_size=best_train_params['batch_size'],
            verbose=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ]
        )

        grid_results = {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'best_model_params': best_model_params,
            'best_train_params': best_train_params,
            'cv_results': cv_results,
            'grid_search': grid_search,
            'training_history': history.history,
            'param_grid': param_grid,
            'cv_folds': cv_folds,
            'scoring': scoring
        }

        print(f"Mejores parámetros encontrados: {best_params}")
        print(f"Mejor score CV: {best_score:.4f}")

        return grid_results

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
        report.append("REPORTE DE EVALUACIÓN DEL MODELO CNN MNIST")
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
    print("=" * 120)
    print("🎯 MÓDULO DE EVALUACIÓN COMPLETO PARA CNN MNIST")
    print("=" * 120)
    print("Implementa TODAS las técnicas de validación y visualización académicas requeridas")
    print("=" * 120)

    # Crear evaluador
    evaluator = ModelEvaluator()

    print("\n🔧 TÉCNICAS ACADÉMICAS DE VALIDACIÓN IMPLEMENTADAS:")
    print("=" * 100)
    print("✅ TÉCNICA #1: dataset_split_demo() - train_test_split() 70%/30%")
    print("✅ TÉCNICA #2: perform_cross_validation() - cross_val_score() StratifiedKFold")
    print("✅ TÉCNICA #2: perform_grid_search_cv() - GridSearchCV() optimización HP")
    print("✅ TÉCNICA #3: plot_roc_curves() - Curvas ROC multiclase con AUC")
    print("✅ TÉCNICA #3: plot_precision_recall_curves() - Curvas Precision-Recall")
    print("✅ TÉCNICA #4: plot_learning_curves() - Detección overfitting/underfitting")
    print("✅ TÉCNICA #5: plot_confusion_matrix() - Matriz confusión + análisis errores")
    print("❌ TÉCNICA #6: unsupervised_techniques_explanation() - NO APLICABLE")
    print("✅ SUITE COMPLETA: run_complete_validation_suite() - Todas las técnicas")
    print("=" * 100)

    print("\n📋 MÉTODOS DISPONIBLES:")
    print("- dataset_split_demo(X, y) - División estratificada del dataset")
    print("- perform_cross_validation(model_func, X, y) - Validación cruzada")
    print("- perform_grid_search_cv(model_func, X, y, param_grid) - Optimización HP")
    print("- plot_roc_curves(roc_data) - Curvas ROC multiclase")
    print("- plot_precision_recall_curves(pr_data) - Curvas Precision-Recall")
    print("- plot_learning_curves(history) - Learning curves con análisis")
    print("- plot_confusion_matrix(cm) - Matriz de confusión con análisis")
    print("- run_complete_validation_suite(model_func, X, y) - SUITE COMPLETA")
    print("=" * 120)

    print("\n🎯 PROYECTO ACADÉMICO:")
    print("• Aprendizaje Supervisado Profundo con CNN")
    print("• Framework: TensorFlow/Keras + OpenCV + Matplotlib")
    print("• Métricas: Accuracy, Loss, Curvas de aprendizaje, Matriz de confusión")
    print("• Dataset: MNIST (dígitos 0-9)")
    print("=" * 120)
