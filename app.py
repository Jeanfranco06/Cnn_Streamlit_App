"""
Aplicación Streamlit para demostración de modelos CNN
CIFAR-10 y MNIST
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
import sys
from datetime import datetime
import pandas as pd

# Agregar el directorio src al path
sys.path.append('src')

from src.data import CIFAR10DataLoader, MNISTDataLoader
from src.model import CIFAR10CNN, MNISTCNN
from src.evaluation import ModelEvaluator
from src.utils import ExperimentTracker, ModelInspector, DataVisualizer

# Configurar página
st.set_page_config(
    page_title="CNN Demos - Grupo 9",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

# Título principal
st.title("🧠 Clasificación de Imágenes con Redes Neuronales Convolucionales")
st.markdown("---")

# Sidebar con información del proyecto
with st.sidebar:
    st.header("📋 Información del Proyecto")

    st.markdown("""
    **Tema:** Redes Neuronales Convolucionales (CNN)

    **Objetivo:** Demostrar el funcionamiento de modelos CNN para clasificación de imágenes

    **Aplicación:** Interfaz interactiva para explorar datasets, entrenar modelos y realizar predicciones

    **Características:**
    - Exploración de datasets CIFAR-10 y MNIST
    - Arquitecturas CNN: Básica, Avanzada y Residual
    - Entrenamiento en tiempo real
    - Evaluación de rendimiento
    - Predicciones interactivas
    """)

    st.markdown("---")

    # Información técnica
    st.markdown("### 🔧 Configuración Técnica")
    st.markdown("""
    - **Framework:** TensorFlow/Keras
    - **Lenguaje:** Python
    - **Interfaz:** Streamlit
    - **Datasets:** CIFAR-10 (60,000 imágenes), MNIST (70,000 dígitos)
    """)

# Crear pestañas principales
tab1, tab2 = st.tabs(["🎨 CIFAR-10 (Clasificación de Objetos)", "🔢 MNIST (Reconocimiento de Dígitos)"])

# Función para cargar datos CIFAR-10
@st.cache_data
def load_cifar10_data():
    """Carga un subconjunto reducido de CIFAR-10 para Streamlit Cloud"""
    try:
        # Liberar memoria antes de cargar
        import gc
        gc.collect()

        # Limpiar cualquier dato de MNIST que pueda estar en memoria
        if 'mnist_data_loaded' in st.session_state and st.session_state['mnist_data_loaded']:
            if 'mnist_data_loader' in st.session_state:
                del st.session_state['mnist_data_loader']
            if 'mnist_data' in st.session_state:
                del st.session_state['mnist_data']
            st.session_state['mnist_data_loaded'] = False
            gc.collect()

        st.warning("⚠️ **CIFAR-10 en Streamlit Cloud:** Se cargará solo un subconjunto reducido de datos para evitar problemas de memoria.")

        # Cargar datos completos primero
        data_loader = CIFAR10DataLoader(validation_split=0.1, random_state=42)
        full_data = data_loader.load_data()

        # Reducir drásticamente el tamaño del dataset para Streamlit Cloud
        # Usar solo 10,000 muestras de entrenamiento, 1,000 de validación y 2,000 de test
        train_subset = 10000
        val_subset = 1000
        test_subset = 2000

        reduced_data = {
            'X_train': full_data['X_train'][:train_subset],
            'y_train': full_data['y_train'][:train_subset],
            'X_val': full_data['X_val'][:val_subset],
            'y_val': full_data['y_val'][:val_subset],
            'X_test': full_data['X_test'][:test_subset],
            'y_test': full_data['y_test'][:test_subset],
            'class_names': full_data['class_names']
        }

        # Convertir a tipos más eficientes para memoria, pero mantener compatibilidad con matplotlib
        for key in ['X_train', 'X_val', 'X_test']:
            if key in reduced_data:
                # Usar float32 en lugar de float16 para compatibilidad con matplotlib
                reduced_data[key] = reduced_data[key].astype('float32')

        # Liberar memoria
        del full_data
        gc.collect()

        st.success(f"✅ CIFAR-10 reducido cargado: {train_subset} train, {val_subset} val, {test_subset} test muestras")

        return data_loader, reduced_data
    except Exception as e:
        st.error(f"Error al cargar los datos de CIFAR-10: {e}")
        st.error("💡 **Solución:** Intenta recargar la página (F5) y usar solo MNIST, que requiere menos memoria.")
        return None, None

# Función para cargar datos MNIST
@st.cache_data
def load_mnist_data():
    """Carga los datos de MNIST"""
    try:
        data_loader = MNISTDataLoader(validation_split=0.1, random_state=42)
        data = data_loader.load_data()

        # Convertir a tipos más eficientes
        for key in ['X_train', 'X_val', 'X_test']:
            if key in data:
                data[key] = data[key].astype('float32')

        return data_loader, data
    except Exception as e:
        st.error(f"Error al cargar los datos de MNIST: {e}")
        return None, None

# Función para cargar modelo
@st.cache_resource
def load_model(model_path):
    """Carga un modelo entrenado"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Modelo no encontrado en: {model_path}")
            return None

        # Determinar el tipo de modelo basado en el nombre del archivo
        if 'mnist' in model_path.lower():
            cnn = MNISTCNN()
        else:
            cnn = CIFAR10CNN()

        # Intentar cargar el modelo
        cnn.load_model(model_path)

        # Verificar que el modelo se cargó correctamente
        if cnn.model is None:
            st.error(f"Error: El modelo se cargó pero es None")
            return None

        # Verificar que el modelo tenga la estructura correcta
        if not hasattr(cnn.model, 'predict'):
            st.error(f"Error: El modelo cargado no tiene método predict")
            return None

        return cnn

    except Exception as e:
        st.error(f"Error al cargar el modelo desde {model_path}: {str(e)}")
        return None

# Función para mostrar sección de dataset
def show_dataset_section(data_loader, data, dataset_name):
    """Muestra la sección de exploración del dataset"""
    st.header(f"📊 Exploración del Dataset {dataset_name}")

    if data_loader is not None and data is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Información General")
            info = data_loader.get_dataset_info()

            st.metric("Total de Imágenes", f"{info['total_samples']:,}")
            st.metric("Clases", info['num_classes'])
            st.metric("Dimensiones", f"{info['image_shape']}")

            st.markdown("#### Distribución por Split:")
            splits_data = {
                'Conjunto': ['Entrenamiento', 'Validación', 'Prueba'],
                'Muestras': [info['train_samples'], info['val_samples'], info['test_samples']],
                'Porcentaje': [f"{info['train_split']:.1%}",
                             f"{info['val_split']:.1%}",
                             f"{info['test_split']:.1%}"]
            }
            st.table(pd.DataFrame(splits_data))

        with col2:
            st.markdown("### 📊 Distribución de Clases")

            # Gráfico de distribución
            fig, ax = plt.subplots(figsize=(8, 6))
            counts = np.bincount(data['y_train'])
            bars = ax.bar(data['class_names'], counts, color='skyblue', alpha=0.7)
            ax.set_title(f'Distribución de Clases - Entrenamiento', fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de muestras')
            ax.tick_params(axis='x', rotation=45)

            # Agregar valores encima de las barras
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       f'{count}', ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown(f"### 🖼️ Muestras del Dataset {dataset_name}")

        if st.button(f"🔄 Generar Nuevas Muestras - {dataset_name}"):
            st.rerun()

        # Mostrar muestras aleatorias
        num_samples = 20
        indices = np.random.choice(len(data['X_train']), num_samples, replace=False)

        cols = st.columns(5)
        for i, idx in enumerate(indices):
            with cols[i % 5]:
                image = data['X_train'][idx]
                label = data['class_names'][data['y_train'][idx]]

                # Convertir a formato PIL Image
                if dataset_name == "MNIST":
                    # Para MNIST, remover dimensión de canal y usar colormap gray
                    image = image.squeeze()
                    # Convertir float16 a float32 si es necesario para procesamiento
                    if image.dtype == 'float16':
                        image = image.astype('float32')
                    img_array = (image * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_array, mode='L')
                else:
                    # Para CIFAR-10
                    # Convertir float16 a float32 si es necesario para procesamiento
                    if image.dtype == 'float16':
                        image = image.astype('float32')
                    img_array = (image * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_array)

                st.image(pil_image, caption=f"{label}", width=100)

    else:
        st.error(f"No se pudieron cargar los datos del dataset {dataset_name}.")

# Función para mostrar sección de modelo
def show_model_section(cnn_class, dataset_name, input_shape):
    """Muestra la sección de arquitectura del modelo"""
    st.header(f"🧠 Arquitectura del Modelo CNN - {dataset_name}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Configuración del Modelo")

        model_type = st.selectbox(
            "Tipo de Modelo",
            ["basic", "advanced", "residual"],
            key=f"model_type_{dataset_name.lower()}"
        )

        if model_type == "basic":
            if dataset_name == "CIFAR-10":
                st.markdown("""
                **Modelo Básico:**
                - 3 capas convolucionales
                - Filtros: [32, 64, 128]
                - Dropout: 50%
                - Capa densa: 512 neuronas
                """)
            else:  # MNIST
                st.markdown("""
                **Modelo Básico:**
                - 2 capas convolucionales
                - Filtros: [32, 64]
                - Dropout: 25%
                - Capa densa: 128 neuronas
                """)
        elif model_type == "advanced":
            if dataset_name == "CIFAR-10":
                st.markdown("""
                **Modelo Avanzado:**
                - 4 capas convolucionales
                - Filtros: [64, 128, 256, 512]
                - Batch Normalization
                - Regularización L2
                - Dropout: 30%
                """)
            else:  # MNIST
                st.markdown("""
                **Modelo Avanzado:**
                - 3 capas convolucionales
                - Filtros: [32, 64, 128]
                - Batch Normalization
                - Regularización L2
                - Dropout: 30%
                """)
        elif model_type == "residual":
            if dataset_name == "CIFAR-10":
                st.markdown("""
                **Modelo Residual:**
                - Bloques residuales
                - Conexiones skip
                - 3 bloques convolucionales
                - Global Average Pooling
                """)
            else:  # MNIST
                st.markdown("""
                **Modelo Residual:**
                - Bloques residuales simplificados
                - 2 bloques convolucionales
                - Global Average Pooling
                """)

        # Parámetros del modelo
        st.markdown("### 🔧 Hiperparámetros")

        epochs = st.slider("Épocas", 10, 100, 50, key=f"epochs_{dataset_name.lower()}")
        batch_size = st.slider("Tamaño del Batch", 16, 128, 64, key=f"batch_{dataset_name.lower()}")
        learning_rate = st.select_slider(
            "Tasa de Aprendizaje",
            options=[1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-4,
            key=f"lr_{dataset_name.lower()}"
        )

        data_augmentation = st.checkbox("Aumento de Datos", value=True, key=f"aug_{dataset_name.lower()}")

    with col2:
        st.markdown("### 📋 Resumen de la Arquitectura")

        # Crear modelo para mostrar resumen
        try:
            if dataset_name == "CIFAR-10":
                cnn = CIFAR10CNN(input_shape=input_shape)
            else:
                cnn = MNISTCNN(input_shape=input_shape)

            if model_type == "basic":
                if dataset_name == "CIFAR-10":
                    model_config = {
                        'filters': [32, 64, 128],
                        'dropout_rate': 0.5,
                        'learning_rate': learning_rate
                    }
                else:
                    model_config = {
                        'filters': [32, 64],
                        'dropout_rate': 0.25,
                        'learning_rate': learning_rate
                    }
            elif model_type == "advanced":
                if dataset_name == "CIFAR-10":
                    model_config = {
                        'filters': [64, 128, 256, 512],
                        'dropout_rate': 0.3,
                        'learning_rate': learning_rate
                    }
                else:
                    model_config = {
                        'filters': [32, 64, 128],
                        'dropout_rate': 0.3,
                        'learning_rate': learning_rate
                    }
            elif model_type == "residual":
                if dataset_name == "CIFAR-10":
                    model_config = {
                        'num_blocks': 3,
                        'filters': 64,
                        'learning_rate': learning_rate
                    }
                else:
                    model_config = {
                        'num_blocks': 2,
                        'filters': 32,
                        'learning_rate': learning_rate
                    }

            model = cnn.build_model(model_type, **model_config)

            # Mostrar resumen
            summary_text = cnn.get_model_summary()
            st.code(summary_text, language="text")

        # Información del modelo
            model_info = ModelInspector.get_model_info(model)
            st.markdown("### 📊 Información del Modelo")
            info_df = pd.DataFrame(list(model_info.items()),
                                 columns=['Parámetro', 'Valor'])
            # Convertir valores a strings para evitar problemas de serialización
            info_df['Valor'] = info_df['Valor'].astype(str)
            st.table(info_df)

        except Exception as e:
            st.error(f"Error al crear el modelo: {e}")

# Función para mostrar sección de entrenamiento
def show_training_section(cnn_class, data_loader, data, dataset_name, input_shape):
    """Muestra la sección de entrenamiento del modelo"""
    st.header(f"🚀 Entrenamiento del Modelo - {dataset_name}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎯 Configuración de Entrenamiento")

        model_type = st.selectbox("Tipo de Modelo", ["basic", "advanced", "residual"],
                                key=f"train_model_select_{dataset_name.lower()}")
        epochs = st.slider("Épocas", 5, 50, 10, key=f"train_epochs_{dataset_name.lower()}")
        batch_size = st.slider("Batch Size", 16, 128, 64, key=f"train_batch_{dataset_name.lower()}")

        train_button_key = f"train_button_{dataset_name.lower()}"
        if st.button(f"🚀 Iniciar Entrenamiento - {dataset_name}", type="primary", key=train_button_key):
            st.session_state[f'training_started_{dataset_name.lower()}'] = True
            # Store training parameters in session state with different keys
            st.session_state[f'training_model_type_{dataset_name.lower()}'] = model_type
            st.session_state[f'training_epochs_{dataset_name.lower()}'] = epochs
            st.session_state[f'training_batch_size_{dataset_name.lower()}'] = batch_size
            st.rerun()

    with col2:
        training_key = f'training_started_{dataset_name.lower()}'
        if training_key in st.session_state and st.session_state[training_key]:
            st.markdown("### 📈 Progreso del Entrenamiento")

            # Placeholder para progreso
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Cargando datos...")
                progress_bar.progress(10)

                status_text.text("Construyendo modelo...")
                # Configurar modelo
                model_type = st.session_state[f'training_model_type_{dataset_name.lower()}']

                if dataset_name == "CIFAR-10":
                    cnn = CIFAR10CNN(input_shape=input_shape)
                    if model_type == "basic":
                        model_config = {'filters': [32, 64, 128], 'dropout_rate': 0.5, 'learning_rate': 1e-4}
                    elif model_type == "advanced":
                        model_config = {'filters': [64, 128, 256, 512], 'dropout_rate': 0.3, 'learning_rate': 1e-4}
                    else:  # residual
                        model_config = {'num_blocks': 3, 'filters': 64, 'learning_rate': 1e-4}
                else:  # MNIST
                    cnn = MNISTCNN(input_shape=input_shape)
                    if model_type == "basic":
                        model_config = {'filters': [32, 64], 'dropout_rate': 0.25, 'learning_rate': 1e-4}
                    elif model_type == "advanced":
                        model_config = {'filters': [32, 64, 128], 'dropout_rate': 0.3, 'learning_rate': 1e-4}
                    else:  # residual
                        model_config = {'num_blocks': 2, 'filters': 32, 'learning_rate': 1e-4}

                model = cnn.build_model(model_type, **model_config)
                progress_bar.progress(30)

                status_text.text("Iniciando entrenamiento...")
                # Entrenar modelo
                # Mapear nombres de dataset a nombres de directorio
                dataset_dir_map = {
                    "CIFAR-10": "cifar10",
                    "MNIST": "mnist"
                }
                dataset_dir_name = dataset_dir_map.get(dataset_name, dataset_name.lower().replace("-", ""))
                save_path = os.path.join("models", dataset_dir_name, f"{model_type}_trained.keras")

                history = cnn.train(
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_val=data['X_val'],
                    y_val=data['y_val'],
                    epochs=st.session_state[f'training_epochs_{dataset_name.lower()}'],
                    batch_size=st.session_state[f'training_batch_size_{dataset_name.lower()}'],
                    data_augmentation=True,
                    save_path=save_path
                )

                progress_bar.progress(100)
                status_text.text("¡Entrenamiento completado!")

                # Mostrar resultados
                st.success(f"Modelo {dataset_name} entrenado exitosamente!")

                # Métricas finales
                final_acc = history['val_accuracy'][-1]
                final_loss = history['val_loss'][-1]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy Final", f"{final_acc:.4f}")
                with col2:
                    st.metric("Loss Final", f"{final_loss:.4f}")

                # Gráfico de curvas de aprendizaje
                st.markdown("### 📊 Curvas de Aprendizaje")
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                axes[0].plot(history['accuracy'], label='Entrenamiento')
                axes[0].plot(history['val_accuracy'], label='Validación')
                axes[0].set_title('Accuracy vs Épocas')
                axes[0].set_xlabel('Épocas')
                axes[0].set_ylabel('Accuracy')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(history['loss'], label='Entrenamiento')
                axes[1].plot(history['val_loss'], label='Validación')
                axes[1].set_title('Loss vs Épocas')
                axes[1].set_xlabel('Épocas')
                axes[1].set_ylabel('Loss')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error durante el entrenamiento: {e}")
                progress_bar.empty()
                status_text.empty()

        else:
            st.info("Configura los parámetros y haz clic en 'Iniciar Entrenamiento'")

# Función para mostrar sección de evaluación
def show_evaluation_section(data_loader, data, dataset_name):
    """Muestra la sección de evaluación del modelo"""
    st.header(f"📊 Evaluación del Modelo - {dataset_name}")

    # Seleccionar tipo de modelo para evaluación
    model_options = ["basic", "advanced", "residual"]
    selected_model_type = st.selectbox(
        "Selecciona el tipo de modelo para evaluar:",
        model_options,
        key=f"eval_model_type_{dataset_name.lower()}"
    )

    # Explicación de las diferencias entre modelos
    st.markdown("### 🔍 Diferencias entre Tipos de Modelo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🟢 Modelo Básico:**
        - Arquitectura simple
        - Menos parámetros
        - Entrenamiento rápido
        - Menor precisión
        """)

    with col2:
        st.markdown("""
        **🟡 Modelo Avanzado:**
        - Capas Batch Normalization
        - Regularización L2
        - Mayor precisión
        - Entrenamiento moderado
        """)

    with col3:
        st.markdown("""
        **🔴 Modelo Residual:**
        - Conexiones residuales (skip)
        - Mejor para datasets grandes
        - Mayor precisión potencial
        - Más parámetros y tiempo
        """)

    # Buscar el modelo específico seleccionado
    # Mapear nombres de dataset a nombres de directorio
    dataset_dir_map = {
        "CIFAR-10": "cifar10",
        "MNIST": "mnist"
    }
    dataset_dir_name = dataset_dir_map.get(dataset_name, dataset_name.lower().replace("-", ""))
    dataset_models_dir = os.path.join("models", dataset_dir_name)

    model_path = None
    if os.path.exists(dataset_models_dir):
        # Buscar modelo entrenado del tipo seleccionado
        trained_model = f"{selected_model_type}_trained.keras"
        trained_path = os.path.join(dataset_models_dir, trained_model)

        if os.path.exists(trained_path):
            model_path = trained_path
        else:
            # Fallback a modelo pre-entrenado
            fallback_model = f"{selected_model_type}_model.keras"
            fallback_path = os.path.join(dataset_models_dir, fallback_model)
            if os.path.exists(fallback_path):
                model_path = fallback_path
                st.warning(f"No se encontró modelo {selected_model_type} entrenado. Usando modelo pre-entrenado.")
            else:
                st.error(f"No se encontró modelo {selected_model_type} para {dataset_name}.")
    else:
        st.error(f"Directorio de modelos para {dataset_name} no encontrado.")

    if model_path is not None and os.path.exists(model_path):
        cnn = load_model(model_path)

        if cnn is not None:
            st.markdown("### 🎯 Métricas de Evaluación")

            # Evaluar modelo
            evaluator = ModelEvaluator(class_names=data['class_names'])
            results = evaluator.evaluate_model(
                cnn.model, data['X_test'], data['y_test']
            )

            # Mostrar métricas principales
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{results['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{results['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{results['f1_score']:.4f}")

            st.markdown("---")

            # Matriz de confusión
            st.markdown("### 📋 Matriz de Confusión")

            tab1, tab2 = st.tabs(["Matriz Normal", "Matriz Normalizada"])

            with tab1:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d',
                          xticklabels=data['class_names'],
                          yticklabels=data['class_names'],
                          cmap='Blues', ax=ax)
                ax.set_title('Matriz de Confusión')
                ax.set_xlabel('Predicción')
                ax.set_ylabel('Valor Real')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

            with tab2:
                cm_normalized = results['confusion_matrix'].astype('float') / results['confusion_matrix'].sum(axis=1)[:, np.newaxis]
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm_normalized, annot=True, fmt='.2f',
                          xticklabels=data['class_names'],
                          yticklabels=data['class_names'],
                          cmap='Blues', ax=ax)
                ax.set_title('Matriz de Confusión Normalizada')
                ax.set_xlabel('Predicción')
                ax.set_ylabel('Valor Real')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

            # Reporte por clase
            st.markdown("### 📊 Métricas por Clase")
            class_report = results['classification_report']

            # Convertir a DataFrame para mejor visualización
            class_data = []
            for class_name in data['class_names']:
                if class_name in class_report:
                    metrics = class_report[class_name]
                    class_data.append({
                        'Clase': class_name,
                        'Precision': float(metrics['precision']),
                        'Recall': float(metrics['recall']),
                        'F1-Score': float(metrics['f1-score']),
                        'Soporte': int(metrics['support'])
                    })

            df_class = pd.DataFrame(class_data)
            st.dataframe(df_class.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}'
            }))

        else:
            st.error("No se pudo cargar el modelo.")
    else:
        st.warning(f"No se encontró un modelo entrenado para {dataset_name}. Entrena un modelo primero en la sección 'Entrenamiento'.")

# Función para mostrar sección de predicciones
def show_predictions_section(data_loader, data, dataset_name, input_shape):
    """Muestra la sección de predicciones"""
    st.header(f"🔮 Realizar Predicciones - {dataset_name}")

    # Seleccionar tipo de modelo
    model_options = ["basic", "advanced", "residual"]
    selected_model_type = st.selectbox(
        "Selecciona el tipo de modelo para predicción:",
        model_options,
        key=f"predict_model_type_{dataset_name.lower()}"
    )

    st.markdown("### 📸 Cargar Imagen para Predicción")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Opciones de entrada
        input_method = st.radio(
            "Método de entrada:",
            ["Imagen del dataset", "Subir imagen"],
            key=f"input_method_{dataset_name.lower()}"
        )

        if input_method == "Imagen del dataset":
            # Seleccionar imagen aleatoria del test set
            if st.button(f"🎲 Seleccionar Imagen Aleatoria - {dataset_name}",
                       key=f"random_button_{dataset_name.lower()}"):
                st.session_state[f'selected_image_idx_{dataset_name.lower()}'] = np.random.randint(len(data['X_test']))
                st.rerun()

            # Slider para seleccionar imagen específica
            image_idx = st.slider(
                f"Seleccionar imagen del test set - {dataset_name}:",
                0, len(data['X_test'])-1,
                st.session_state.get(f'selected_image_idx_{dataset_name.lower()}', 0),
                key=f"slider_{dataset_name.lower()}"
            )

            selected_image = data['X_test'][image_idx]
            true_label = data['class_names'][data['y_test'][image_idx]]

        else:  # Subir imagen
            if dataset_name == "CIFAR-10":
                uploaded_file = st.file_uploader(
                    "Sube una imagen (32x32, formato RGB)",
                    type=['png', 'jpg', 'jpeg'],
                    key=f"uploader_{dataset_name.lower()}"
                )
            else:  # MNIST
                uploaded_file = st.file_uploader(
                    "Sube una imagen de dígito (28x28, escala de grises)",
                    type=['png', 'jpg', 'jpeg'],
                    key=f"uploader_{dataset_name.lower()}"
                )

            if uploaded_file is not None:
                try:
                    # Procesar imagen subida
                    image = Image.open(uploaded_file)

                    if dataset_name == "CIFAR-10":
                        image = image.resize((32, 32))
                        image_array = np.array(image) / 255.0

                        # Asegurar que tenga 3 canales
                        if len(image_array.shape) == 2:
                            image_array = np.stack([image_array] * 3, axis=-1)
                        elif image_array.shape[-1] == 4:
                            image_array = image_array[:, :, :3]
                    else:  # MNIST
                        # Mejor preprocesamiento para MNIST
                        image = image.resize((28, 28)).convert('L')

                        # Convertir a array y normalizar
                        image_array = np.array(image, dtype=np.float32) / 255.0

                        # Invertir colores si es necesario (fondo blanco -> fondo negro)
                        if image_array.mean() > 0.5:  # Si la imagen es mayormente clara
                            image_array = 1.0 - image_array

                        # Asegurar que tenga la forma correcta (28, 28, 1)
                        if image_array.ndim == 2:
                            image_array = np.expand_dims(image_array, axis=-1)

                        # Verificar dimensiones
                        if image_array.shape != (28, 28, 1):
                            st.error(f"Error: La imagen procesada tiene forma {image_array.shape}, se esperaba (28, 28, 1)")
                            selected_image = None
                            true_label = None
                        else:
                            selected_image = image_array
                            true_label = "Desconocido (imagen subida)"
                except Exception as e:
                    st.error(f"Error al procesar la imagen subida: {str(e)}")
                    st.error("Por favor, asegúrate de subir un archivo de imagen válido (PNG, JPG, JPEG).")
                    selected_image = None
                    true_label = None
            else:
                selected_image = None
                true_label = None

    with col2:
        if selected_image is not None:
            # Mostrar imagen
            fig, ax = plt.subplots(figsize=(6, 6))

            # Convertir imagen a formato compatible con matplotlib
            display_image = selected_image.copy()
            if display_image.dtype == 'float16':
                display_image = display_image.astype('float32')

            if dataset_name == "MNIST":
                ax.imshow(display_image.squeeze(), cmap='gray')
            else:
                ax.imshow(display_image)
            ax.set_title(f"Imagen Seleccionada\nEtiqueta real: {true_label}",
                       fontsize=14, fontweight='bold')
            ax.axis('off')
            st.pyplot(fig)

            # Botón para predecir
            if st.button(f"🔮 Realizar Predicción - {dataset_name}", type="primary",
                       key=f"predict_button_{dataset_name.lower()}"):
                try:
                    with st.spinner("Cargando modelo..."):
                        # Buscar el modelo correspondiente solo cuando se hace clic
                        dataset_dir_map = {
                            "CIFAR-10": "cifar10",
                            "MNIST": "mnist"
                        }
                        dataset_dir_name = dataset_dir_map.get(dataset_name, dataset_name.lower().replace("-", ""))
                        dataset_models_dir = os.path.join("models", dataset_dir_name)
                        model_path = None

                        if os.path.exists(dataset_models_dir):
                            # Buscar modelo entrenado del tipo seleccionado
                            trained_model = f"{selected_model_type}_trained.keras"
                            trained_path = os.path.join(dataset_models_dir, trained_model)

                            if os.path.exists(trained_path):
                                model_path = trained_path
                            else:
                                # Fallback a modelo pre-entrenado
                                fallback_model = f"{selected_model_type}_model.keras"
                                fallback_path = os.path.join(dataset_models_dir, fallback_model)
                                if os.path.exists(fallback_path):
                                    model_path = fallback_path
                                else:
                                    # Si no existe modelo residual, usar advanced como fallback
                                    if selected_model_type == "residual":
                                        advanced_fallback = "advanced_model.keras"
                                        advanced_path = os.path.join(dataset_models_dir, advanced_fallback)
                                        if os.path.exists(advanced_path):
                                            model_path = advanced_path
                                            st.warning(f"No se encontró modelo residual entrenado. Usando modelo avanzado como alternativa.")
                                        else:
                                            st.error(f"No se encontraron modelos para {dataset_name}.")
                                    else:
                                        st.error(f"No se encontró modelo {selected_model_type} para {dataset_name}.")
                        else:
                            st.error(f"Directorio de modelos para {dataset_name} no encontrado.")

                        if model_path is not None and os.path.exists(model_path):
                            cnn = load_model(model_path)

                            if cnn is not None:
                                # Realizar predicción
                                input_image = np.expand_dims(selected_image, axis=0)

                                # Verificar que la imagen tenga la forma correcta
                                expected_shape = (1,) + input_shape
                                if input_image.shape != expected_shape:
                                    st.error(f"Error: La imagen procesada tiene forma {input_image.shape}, se esperaba {expected_shape}")
                                    st.stop()

                                # Realizar predicción con manejo de errores
                                try:
                                    predictions = cnn.model.predict(input_image, verbose=0)[0]
                                except Exception as pred_error:
                                    st.error(f"Error durante la predicción: {str(pred_error)}")
                                    st.stop()

                                # Verificar que las predicciones sean válidas
                                if not isinstance(predictions, np.ndarray) or len(predictions) == 0:
                                    st.error("Error: Las predicciones no son válidas")
                                    st.stop()

                                # Obtener top 3 predicciones
                                try:
                                    top_3_indices = np.argsort(predictions)[-3:][::-1]
                                    top_3_probs = predictions[top_3_indices]
                                    top_3_classes = [data['class_names'][i] for i in top_3_indices]
                                except Exception as sort_error:
                                    st.error(f"Error procesando predicciones: {str(sort_error)}")
                                    st.stop()

                                # Mostrar resultados
                                st.success("¡Predicción completada!")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("### 🏆 Top 3 Predicciones")

                                    for i, (class_name, prob) in enumerate(zip(top_3_classes, top_3_probs)):
                                        if i == 0:
                                            st.metric(f"🥇 {class_name}", f"{prob:.4f}")
                                        elif i == 1:
                                            st.metric(f"🥈 {class_name}", f"{prob:.4f}")
                                        else:
                                            st.metric(f"🥉 {class_name}", f"{prob:.4f}")

                                with col2:
                                    st.markdown("### 📊 Probabilidades")

                                    try:
                                        # Gráfico de barras
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        bars = ax.barh(range(len(top_3_classes)), top_3_probs,
                                                     color=['gold', 'silver', '#CD7F32'])
                                        ax.set_yticks(range(len(top_3_classes)))
                                        ax.set_yticklabels(top_3_classes)
                                        ax.set_xlabel('Probabilidad')
                                        ax.set_title('Top 3 Predicciones')

                                        # Agregar valores en las barras
                                        for bar, prob in zip(bars, top_3_probs):
                                            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                                   f'{prob:.3f}', va='center', fontsize=10)

                                        st.pyplot(fig)
                                    except Exception as plot_error:
                                        st.error(f"Error al generar el gráfico: {str(plot_error)}")
                                        # Mostrar resultados en texto como fallback
                                        st.markdown("**Resultados en texto:**")
                                        for i, (class_name, prob) in enumerate(zip(top_3_classes, top_3_probs)):
                                            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                                            st.write(f"{medal} {class_name}: {prob:.4f}")

                                # Comparación con etiqueta real (si aplica)
                                if input_method == "Imagen del dataset":
                                    pred_class = top_3_classes[0]
                                    is_correct = pred_class == true_label

                                    if is_correct:
                                        st.success(f"✅ ¡Predicción correcta! El modelo acertó.")
                                    else:
                                        st.error(f"❌ Predicción incorrecta. El modelo predijo '{pred_class}' pero la etiqueta real es '{true_label}'.")
                            else:
                                st.error("No se pudo cargar el modelo.")
                        else:
                            st.warning(f"No se encontró un modelo entrenado para {dataset_name}. Entrena un modelo primero.")

                except Exception as e:
                    st.error(f"Error inesperado durante la predicción: {str(e)}")
                    st.error("Por favor, intenta con otra imagen o verifica que el modelo esté cargado correctamente.")

        else:
            st.info("Selecciona o sube una imagen para realizar una predicción.")

# Función para mostrar contenido de pestaña con carga lazy
def show_tab_content(dataset_name, cnn_class, input_shape):
    """Muestra el contenido de una pestaña con carga lazy de datos"""
    if dataset_name == "CIFAR-10":
        emoji = "🎨"
        description = "Dataset con 60,000 imágenes de 32x32 píxeles en 10 categorías diferentes."
    else:
        emoji = "🔢"
        description = "Dataset con 70,000 imágenes de dígitos escritos a mano (0-9)."

    st.markdown(f"## {emoji} {dataset_name}: {description.split(':')[0]}")
    st.markdown(description)

    # Sub-pestañas
    tabs = st.tabs(["📊 Dataset", "🧠 Modelo", "🚀 Entrenamiento", "📊 Evaluación", "🔮 Predicciones"])

    # Estado de carga de datos
    data_key = f"{dataset_name.lower()}_data_loaded"
    if data_key not in st.session_state:
        st.session_state[data_key] = False

    # Botón para cargar datos
    if not st.session_state[data_key]:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(f"📥 Cargar Datos {dataset_name}", type="primary", key=f"load_{dataset_name.lower()}_btn"):
                with st.spinner(f"Cargando datos de {dataset_name}..."):
                    try:
                        if dataset_name == "CIFAR-10":
                            data_loader, data = load_cifar10_data()
                        else:
                            data_loader, data = load_mnist_data()

                        if data_loader and data:
                            st.session_state[f"{dataset_name.lower()}_data_loader"] = data_loader
                            st.session_state[f"{dataset_name.lower()}_data"] = data
                            st.session_state[data_key] = True
                            st.success(f"✅ Datos de {dataset_name} cargados exitosamente!")
                            st.rerun()
                        else:
                            st.error(f"❌ Error al cargar los datos de {dataset_name}")
                    except Exception as e:
                        st.error(f"❌ Error al cargar datos: {str(e)}")
            else:
                st.info(f"💡 Haz clic en 'Cargar Datos {dataset_name}' para comenzar")
                return

    # Si los datos están cargados, mostrar las pestañas
    if st.session_state[data_key]:
        data_loader = st.session_state[f"{dataset_name.lower()}_data_loader"]
        data = st.session_state[f"{dataset_name.lower()}_data"]

        with tabs[0]:  # Dataset
            show_dataset_section(data_loader, data, dataset_name)

        with tabs[1]:  # Modelo
            show_model_section(cnn_class, dataset_name, input_shape)

        with tabs[2]:  # Entrenamiento
            show_training_section(cnn_class, data_loader, data, dataset_name, input_shape)

        with tabs[3]:  # Evaluación
            # Lazy loading - only run evaluation when tab is active
            eval_key = f"{dataset_name.lower()}_eval_active"
            if st.session_state.get(eval_key, False) or st.button(f"🔍 Ejecutar Evaluación {dataset_name}", key=f"{dataset_name.lower()}_eval_btn"):
                st.session_state[eval_key] = True
                show_evaluation_section(data_loader, data, dataset_name)
            else:
                st.info("Haz clic en 'Ejecutar Evaluación' para ver las métricas del modelo.")

        with tabs[4]:  # Predicciones
            show_predictions_section(data_loader, data, dataset_name, input_shape)

# Contenido de la pestaña CIFAR-10
with tab1:
    show_tab_content("CIFAR-10", CIFAR10CNN, (32, 32, 3))

# Contenido de la pestaña MNIST
with tab2:
    show_tab_content("MNIST", MNISTCNN, (28, 28, 1))

# Footer
st.markdown("---")
st.markdown("### 📚 Tecnologías Utilizadas")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Datasets:**
    - CIFAR-10
    - MNIST
    """)

with col2:
    st.markdown("""
    **Framework:**
    - TensorFlow/Keras
    """)

with col3:
    st.markdown("""
    **Interfaz:**
    - Streamlit
    """)

st.markdown("---")
st.markdown("**© 2025 - Aplicación de Demostración CNN**")
