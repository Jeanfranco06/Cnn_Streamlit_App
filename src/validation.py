"""
Módulo de validación y visualización para modelos CNN MNIST
Implementa todas las técnicas requeridas de validación y visualización
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


class ValidationTools:
    """
    Clase que implementa todas las técnicas de validación y visualización requeridas
    para el proyecto CNN MNIST
    """

    def __init__(self, class_names: List[str] = None):
        """
        Inicializa las herramientas de validación

        Args:
            class_names: Nombres de las clases
        """
        self.class_names = class_names or [str(i) for i in range(10)]
        self.num_classes = len(self.class_names)

    def dataset_division_demo(self, X: np.ndarray, y: np.ndarray,
                            test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """
        Demuestra la división del dataset usando train_test_split()
        70% entrenamiento / 30% prueba

        Args:
            X: Datos de entrada
            y: Etiquetas
            test_size: Proporción para prueba
            random_state: Semilla para reproducibilidad

        Returns:
            Diccionario con los splits de datos
        """
        print("=" * 60)
        print("1. DIVISIÓN DEL DATASET: train_test_split()")
        print("=" * 60)
        print(f"División: {100*(1-test_size):.0f}% entrenamiento / {100*test_size:.0f}% prueba")

        # División estratificada del dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        print(f"Datos originales: {len(X)} muestras")
        print(f"Entrenamiento: {len(X_train)} muestras ({100*len(X_train)/len(X):.1f}%)")
        print(f"Prueba: {len(X_test)} muestras ({100*len(X_test)/len(X):.1f}%)")

        # Verificar distribución de clases
        train_dist = np.bincount(y_train)
        test_dist = np.bincount(y_test)

        print("\nDistribución de clases:")
        print("Clase | Entrenamiento | Prueba | Total")
        print("-" * 35)
        for i in range(self.num_classes):
            total = train_dist[i] + test_dist[i]
            print("4")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_distribution': train_dist,
            'test_distribution': test_dist
        }

    def cross_validation_demo(self, model_builder_func, X: np.ndarray, y: np.ndarray,
                            cv_folds: int = 5, epochs: int = 5) -> Dict[str, Any]:
        """
        Demuestra validación cruzada usando cross_val_score()

        Args:
            model_builder_func: Función que construye el modelo
            X: Datos de entrada
            y: Etiquetas
            cv_folds: Número de folds
            epochs: Épocas por fold

        Returns:
            Resultados de validación cruzada
        """
        print("\n" + "=" * 60)
        print("2. VALIDACIÓN CRUZADA: cross_val_score()")
        print("=" * 60)
        print(f"Número de folds: {cv_folds}")

        # Usar StratifiedKFold para mantener balance de clases
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = []
        fold_details = []

        print("\nEntrenando folds...")
        fold = 1
        for train_index, val_index in skf.split(X, y):
            print(f"Fold {fold}/{cv_folds}...")

            # Dividir datos
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            # Construir modelo fresco
            model = model_builder_func()

            # Entrenar modelo
            model.fit(
                X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=128,
                validation_data=(X_val_fold, y_val_fold),
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
            )

            # Evaluar
            _, score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
            cv_scores.append(score)

            fold_details.append({
                'fold': fold,
                'score': score,
                'train_samples': len(X_train_fold),
                'val_samples': len(X_val_fold)
            })

            print(".4f")
            fold += 1

        cv_scores = np.array(cv_scores)

        print("Resultados finales:")
        print(".4f")
        print(".4f")
        print(".4f")

        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'fold_details': fold_details,
            'cv_folds': cv_folds
        }

    def grid_search_demo(self, model_builder_func, X: np.ndarray, y: np.ndarray,
                        param_grid: Dict[str, Any], cv_folds: int = 3) -> Dict[str, Any]:
        """
        Demuestra optimización de hiperparámetros usando GridSearchCV()

        Args:
            model_builder_func: Función que construye el modelo
            X: Datos de entrada
            y: Etiquetas
            param_grid: Parámetros a probar
            cv_folds: Número de folds para CV

        Returns:
            Resultados de GridSearchCV
        """
        print("\n" + "=" * 60)
        print("3. OPTIMIZACIÓN DE HIPERPARÁMETROS: GridSearchCV()")
        print("=" * 60)
        print(f"Parámetros a probar: {param_grid}")
        print(f"Folds para CV: {cv_folds}")

        # Crear wrapper de Keras para sklearn
        def create_model(**params):
            model_params = {}
            train_params = {'epochs': 5, 'batch_size': 128}

            for key, value in params.items():
                if key in ['epochs', 'batch_size', 'learning_rate']:
                    train_params[key] = value
                else:
                    model_params[key] = value

            model = model_builder_func(**model_params)
            return model

        keras_clf = KerasClassifier(
            build_fn=create_model,
            epochs=5,
            batch_size=128,
            verbose=0
        )

        # Ejecutar GridSearchCV
        print("\nEjecutando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(
            estimator=keras_clf,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X, y)

        print("Mejores parámetros encontrados:")
        print(f"  {grid_search.best_params_}")
        print(".4f")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'grid_search': grid_search
        }

    def roc_curves_demo(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       save_path: Optional[str] = None):
        """
        Demuestra curvas ROC para modelos de clasificación

        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas
            save_path: Ruta para guardar la imagen
        """
        print("\n" + "=" * 60)
        print("4. CURVAS ROC PARA MODELOS DE CLASIFICACIÓN")
        print("=" * 60)

        # Binarizar etiquetas para multiclase
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(12, 8))

        # Calcular AUC para cada clase
        auc_scores = {}

        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[class_name] = roc_auc

            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')

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

        print("AUC Scores por clase:")
        for class_name, auc_score in auc_scores.items():
            print(".3f")

    def precision_recall_curves_demo(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   save_path: Optional[str] = None):
        """
        Demuestra curvas Precision-Recall para modelos de clasificación

        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas
            save_path: Ruta para guardar la imagen
        """
        print("\n" + "=" * 60)
        print("5. CURVAS PRECISION-RECALL PARA MODELOS DE CLASIFICACIÓN")
        print("=" * 60)

        # Binarizar etiquetas para multiclase
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(12, 8))

        for i, class_name in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])

            plt.plot(recall, precision, linewidth=2, label=f'{class_name}')

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

    def learning_curves_demo(self, history: Dict[str, Any], save_path: Optional[str] = None):
        """
        Demuestra curvas de aprendizaje para detectar overfitting/underfitting

        Args:
            history: Historial de entrenamiento
            save_path: Ruta para guardar la imagen
        """
        print("\n" + "=" * 60)
        print("6. LEARNING CURVES PARA DETECTAR OVERFITTING/UNDERFITTING")
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
            print(f"Learning curves guardadas en: {save_path}")

        plt.show()

        # Análisis de overfitting/underfitting
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]

        print("Análisis de Overfitting/Underfitting:")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

        gap_acc = final_train_acc - final_val_acc
        gap_loss = final_val_loss - final_train_loss

        if gap_acc > 0.1:
            print("⚠️  Posible OVERFITTING detectado (gap de accuracy > 10%)")
        elif gap_acc < -0.05:
            print("⚠️  Posible UNDERFITTING detectado (accuracy de validación mucho mayor)")
        else:
            print("✅ Modelo bien ajustado (gap de accuracy aceptable)")

        if gap_loss > 0.2:
            print("⚠️  Posible OVERFITTING detectado (gap de loss significativo)")
        else:
            print("✅ Loss de validación razonable")

    def confusion_matrix_demo(self, y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[str] = None, normalize: bool = False):
        """
        Demuestra matriz de confusión para visualizar errores de clasificación

        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            save_path: Ruta para guardar la imagen
            normalize: Si normalizar la matriz
        """
        print("\n" + "=" * 60)
        print("7. MATRIZ DE CONFUSIÓN PARA VISUALIZAR ERRORES DE CLASIFICACIÓN")
        print("=" * 60)

        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

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
            print(f"Matriz de confusión guardada en: {save_path}")

        plt.show()

        # Análisis de la matriz de confusión
        print("Análisis de la matriz de confusión:")
        print("Verdaderos Positivos (diagonal principal):")
        for i, class_name in enumerate(self.class_names):
            vp = cm[i, i]
            total_real = cm[i, :].sum()
            precision = vp / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
            recall = vp / total_real if total_real > 0 else 0
            print("2d")

        # Errores más comunes
        print("Errores más comunes:")
        errors = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    errors.append((cm[i, j], self.class_names[i], self.class_names[j]))

        errors.sort(reverse=True)
        for count, real, pred in errors[:5]:  # Top 5 errores
            print("2d")

    def unsupervised_note(self):
        """
        Nota sobre técnicas no supervisadas (no aplicables a este proyecto supervisado)
        """
        print("\n" + "=" * 60)
        print("8. NOTA SOBRE TÉCNICAS NO SUPERVISADAS")
        print("=" * 60)
        print("Este proyecto utiliza APRENDIZAJE SUPERVISADO con CNN para clasificación.")
        print("Las siguientes técnicas NO SON APLICABLES:")
        print("• Silhouette Score - Para clustering (no supervisado)")
        print("• Varianza Explicada - Para PCA/reducción dimensional (no supervisado)")
        print("\nRazones:")
        print("• El dataset MNIST tiene etiquetas (y) - aprendizaje SUPERVISADO")
        print("• Usamos CNN para clasificar dígitos 0-9 con etiquetas conocidas")
        print("• Evaluamos accuracy, precision, recall, F1-score (métricas supervisadas)")
        print("• No realizamos clustering ni reducción dimensional no supervisada")

    def run_complete_validation_demo(self, model_builder_func, X: np.ndarray, y: np.ndarray,
                                   save_dir: str = "validation_demo"):
        """
        Ejecuta una demostración completa de todas las técnicas de validación

        Args:
            model_builder_func: Función que construye el modelo
            X: Datos de entrada
            y: Etiquetas
            save_dir: Directorio para guardar resultados
        """
        print("=" * 80)
        print("DEMONSTRACIÓN COMPLETA DE TÉCNICAS DE VALIDACIÓN Y VISUALIZACIÓN")
        print("=" * 80)
        print("Proyecto: CNN MNIST - Aprendizaje Supervisado Profundo")
        print("Framework: TensorFlow/Keras, OpenCV, Matplotlib")
        print("=" * 80)

        # Crear directorio para guardar resultados
        os.makedirs(save_dir, exist_ok=True)

        # 1. División del dataset
        splits = self.dataset_division_demo(X, y)

        # 2. Validación cruzada
        cv_results = self.cross_validation_demo(
            model_builder_func, splits['X_train'], splits['y_train'], cv_folds=3, epochs=3
        )

        # 3. GridSearchCV para optimización de hiperparámetros
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [64, 128]
        }
        grid_results = self.grid_search_demo(
            model_builder_func, splits['X_train'], splits['y_train'], param_grid, cv_folds=2
        )

        # 4. Entrenar modelo final con mejores parámetros
        print("\n" + "=" * 60)
        print("ENTRENANDO MODELO FINAL CON MEJORES PARÁMETROS")
        print("=" * 60)

        best_params = grid_results['best_params']
        print(f"Usando parámetros óptimos: {best_params}")

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
            splits['X_train'], splits['y_train'],
            epochs=train_params['epochs'],
            batch_size=train_params['batch_size'],
            validation_split=0.1,
            verbose=1,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        )

        # 5. Evaluación final
        print("\n" + "=" * 60)
        print("EVALUACIÓN FINAL DEL MODELO")
        print("=" * 60)

        # Predicciones
        y_pred_proba = final_model.predict(splits['X_test'])
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Métricas
        accuracy = accuracy_score(splits['y_test'], y_pred)
        precision = precision_score(splits['y_test'], y_pred, average='weighted')
        recall = recall_score(splits['y_test'], y_pred, average='weighted')
        f1 = f1_score(splits['y_test'], y_pred, average='weighted')

        print("Métricas finales:")
        print(".4f")
        print(".4f")
        print(".4f")
        print(".4f")

        # 6. Visualizaciones
        self.roc_curves_demo(splits['y_test'], y_pred_proba,
                           save_path=os.path.join(save_dir, "roc_curves.png"))

        self.precision_recall_curves_demo(splits['y_test'], y_pred_proba,
                                        save_path=os.path.join(save_dir, "precision_recall_curves.png"))

        self.learning_curves_demo(history.history,
                                save_path=os.path.join(save_dir, "learning_curves.png"))

        self.confusion_matrix_demo(splits['y_test'], y_pred,
                                 save_path=os.path.join(save_dir, "confusion_matrix.png"))

        self.confusion_matrix_demo(splits['y_test'], y_pred, normalize=True,
                                 save_path=os.path.join(save_dir, "confusion_matrix_normalized.png"))

        # 7. Nota sobre técnicas no supervisadas
        self.unsupervised_note()

        # 8. Resumen final
        print("\n" + "=" * 80)
        print("RESUMEN FINAL - TÉCNICAS DE VALIDACIÓN IMPLEMENTADAS")
        print("=" * 80)
        print("✅ División del dataset: train_test_split() (70% train / 30% test)")
        print("✅ Validación cruzada: cross_val_score() con StratifiedKFold")
        print("✅ Optimización de hiperparámetros: GridSearchCV()")
        print("✅ Curvas ROC: Para evaluación de clasificación multiclase")
        print("✅ Curvas Precision-Recall: Para clases desbalanceadas")
        print("✅ Learning Curves: Detección de overfitting/underfitting")
        print("✅ Matriz de confusión: Visualización de errores de clasificación")
        print("❌ Silhouette Score: No aplicable (aprendizaje supervisado)")
        print("❌ Varianza Explicada: No aplicable (aprendizaje supervisado)")
        print("=" * 80)
        print("Proyecto CNN MNIST: APRENDIZAJE SUPERVISADO PROFUNDO")
        print("Framework: TensorFlow/Keras + OpenCV + Matplotlib")
        print(".4f")
        print("=" * 80)

        return {
            'splits': splits,
            'cv_results': cv_results,
            'grid_results': grid_results,
            'final_model': final_model,
            'history': history.history,
            'final_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'predictions': {
                'y_true': splits['y_test'],
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }


if __name__ == "__main__":
    print("Módulo de validación para CNN MNIST")
    print("Implementa todas las técnicas requeridas de validación y visualización")
    print("\nTécnicas disponibles:")
    print("1. División del dataset: train_test_split()")
    print("2. Validación cruzada: cross_val_score()")
    print("3. Optimización de hiperparámetros: GridSearchCV()")
    print("4. Curvas ROC para clasificación")
    print("5. Curvas Precision-Recall")
    print("6. Learning Curves (overfitting/underfitting)")
    print("7. Matriz de confusión")
    print("8. Nota sobre técnicas no supervisadas (no aplicables)")
