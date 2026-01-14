"""
Aplicación Streamlit para demostración de modelos CNN
MNIST
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
import time
import tensorflow as tf

# Agregar el directorio src al path
sys.path.append('src')

from src.data import MNISTDataLoader
from src.model import MNISTCNN
from src.evaluation import ModelEvaluator
from src.utils import ExperimentTracker, ModelInspector, DataVisualizer
from src.emotion_data import EmotionDataLoader
from src.emotion_model import EmotionClassifier
from src.emotion_visualization import EmotionVisualizer
from src.gradcam import GradCAM

# Configurar página
st.set_page_config(
    page_title="CNN MNIST Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para limpiar memoria
def clear_memory():
    """Limpia la memoria eliminando objetos grandes"""
    import gc
    # Limpiar variables de session state relacionadas con datos
    keys_to_clear = ['mnist_data_loader', 'mnist_data']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Limpiar variables de entrenamiento
    training_keys = [k for k in st.session_state.keys() if k.startswith('training_')]
    for key in training_keys:
        del st.session_state[key]

    # Forzar garbage collection
    gc.collect()

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

# Título principal
st.title("🧠 Clasificación con Redes Neuronales Convolucionales")
st.markdown("### Aplicación interactiva para reconocimiento de dígitos MNIST y emociones faciales")
st.markdown("---")

# Sidebar con información del proyecto
with st.sidebar:
    st.header("📋 Información del Proyecto")

    st.markdown("""
    **Tema:** Redes Neuronales Convolucionales (CNN)

    **Objetivo:** Demostrar el funcionamiento de modelos CNN para reconocimiento de dígitos

    **Aplicación:** Interfaz interactiva para explorar MNIST, entrenar modelos y realizar predicciones

    **Características:**
    - Exploración del dataset MNIST
    - Arquitecturas CNN: Básica, Avanzada y Residual
    - Entrenamiento en tiempo real
    - Evaluación de rendimiento
    - Predicciones interactivas
    """)

    st.markdown("---")

    # Botón para limpiar memoria
    if st.button("🧹 Limpiar Memoria", help="Libera memoria eliminando datos y modelos cargados"):
        clear_memory()
        st.success("✅ Memoria limpiada exitosamente!")
        st.rerun()

    st.markdown("---")

    # Información técnica
    st.markdown("### 🔧 Configuración Técnica")
    st.markdown("""
    - **Framework:** TensorFlow/Keras
    - **Lenguaje:** Python
    - **Interfaz:** Streamlit
    - **Dataset:** MNIST (70,000 imágenes de dígitos)
    """)

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

        cnn = MNISTCNN()

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
            ax.set_title(f'Distribución de Dígitos - Entrenamiento', fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de muestras')
            ax.tick_params(axis='x', rotation=0)

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

                # Para MNIST, remover dimensión de canal y usar colormap gray
                image = image.squeeze()
                # Convertir float16 a float32 si es necesario para procesamiento
                if image.dtype == 'float16':
                    image = image.astype('float32')
                img_array = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array, mode='L')

                st.image(pil_image, caption=f"{label}", width=100)

    else:
        st.error(f"No se pudieron cargar los datos del dataset {dataset_name}.")

# Función para mostrar sección de modelo
def show_model_section(cnn_class, dataset_name, input_shape):
    """Muestra la sección de arquitectura del modelo"""
    st.header(f"🧠 Arquitectura del Modelo CNN - {dataset_name}")

    st.info("💡 **Esta sección es informativa:** Explora diferentes arquitecturas de modelo y sus hiperparámetros. Los parámetros aquí no afectan el entrenamiento real.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Configuración del Modelo")

        model_type = st.selectbox(
            "Tipo de Modelo",
            ["basic", "advanced", "residual"],
            key=f"model_type_{dataset_name.lower()}"
        )

        if model_type == "basic":
            st.markdown("""
            **Modelo Básico:**
            - 2 capas convolucionales
            - Filtros: [32, 64]
            - Dropout: 25%
            - Capa densa: 128 neuronas
            """)
        elif model_type == "advanced":
            st.markdown("""
            **Modelo Avanzado:**
            - 3 capas convolucionales
            - Filtros: [32, 64, 128]
            - Batch Normalization
            - Regularización L2
            - Dropout: 30%
            """)
        elif model_type == "residual":
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
            cnn = MNISTCNN(input_shape=input_shape)

            if model_type == "basic":
                model_config = {
                    'filters': [32, 64],
                    'dropout_rate': 0.25,
                    'learning_rate': learning_rate
                }
            elif model_type == "advanced":
                model_config = {
                    'filters': [32, 64, 128],
                    'dropout_rate': 0.3,
                    'learning_rate': learning_rate
                }
            elif model_type == "residual":
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

            # Contenedor para progreso detallado
            progress_container = st.container()
            with progress_container:
                # Barra de progreso principal
                progress_bar = st.progress(0)
                status_text = st.empty()
                detail_text = st.empty()
                epoch_progress = st.empty()
                layer_info = st.empty()

                try:
                    # Fase 1: Preparación de datos
                    status_text.markdown("**🔄 Fase 1/5: Preparación de Datos**")
                    detail_text.text("Verificando y preprocesando datos de entrenamiento...")
                    progress_bar.progress(5)

                    # Fase 2: Construcción del modelo
                    status_text.markdown("**🏗️ Fase 2/5: Construcción del Modelo**")
                    model_type = st.session_state[f'training_model_type_{dataset_name.lower()}']

                    # Mostrar arquitectura del modelo que se va a construir
                    if model_type == "basic":
                        layer_info.markdown("""
                        **Construyendo Modelo Básico MNIST:**
                        - Capa Conv2D (32 filtros, 3x3) + ReLU + MaxPooling
                        - Capa Conv2D (64 filtros, 3x3) + ReLU + MaxPooling
                        - Flatten + Dense(128) + Dropout(25%) + Dense(10)
                        """)
                    elif model_type == "advanced":
                        layer_info.markdown("""
                        **Construyendo Modelo Avanzado MNIST:**
                        - Capa Conv2D (32 filtros) + BatchNorm + ReLU + MaxPooling
                        - Capa Conv2D (64 filtros) + BatchNorm + ReLU + MaxPooling
                        - Capa Conv2D (128 filtros) + BatchNorm + ReLU + MaxPooling
                        - GlobalAveragePooling + Dense(10)
                        """)
                    else:  # residual
                        layer_info.markdown("""
                        **Construyendo Modelo Residual MNIST:**
                        - Bloque Residual 1: Conv2D + BatchNorm + ReLU + Conv2D + Skip Connection
                        - Bloque Residual 2: Conv2D + BatchNorm + ReLU + Conv2D + Skip Connection
                        - GlobalAveragePooling + Dense(10)
                        """)

                    detail_text.text("Configurando capas convolucionales y conexiones...")
                    progress_bar.progress(15)

                    # Configurar y construir modelo
                    cnn = MNISTCNN(input_shape=input_shape)
                    if model_type == "basic":
                        model_config = {'filters': [32, 64], 'dropout_rate': 0.25, 'learning_rate': 1e-4}
                    elif model_type == "advanced":
                        model_config = {'filters': [32, 64, 128], 'dropout_rate': 0.3, 'learning_rate': 1e-4}
                    else:  # residual
                        model_config = {'num_blocks': 2, 'filters': 32, 'learning_rate': 1e-4}

                    detail_text.text("Compilando modelo con optimizador Adam...")
                    model = cnn.build_model(model_type, **model_config)
                    progress_bar.progress(25)

                    # Fase 3: Configuración del entrenamiento
                    status_text.markdown("**⚙️ Fase 3/5: Configuración del Entrenamiento**")
                    detail_text.text("Preparando generadores de datos y callbacks...")
                    epochs = st.session_state[f'training_epochs_{dataset_name.lower()}']
                    batch_size = st.session_state[f'training_batch_size_{dataset_name.lower()}']

                    # Calcular número total de pasos
                    total_samples = len(data['X_train'])
                    steps_per_epoch = total_samples // batch_size

                    layer_info.markdown(f"""
                    **Parámetros de Entrenamiento:**
                    - Épocas: {epochs}
                    - Tamaño de batch: {batch_size}
                    - Muestras totales: {total_samples:,}
                    - Pasos por época: {steps_per_epoch}
                    - Aumento de datos: Activado
                    """)
                    progress_bar.progress(35)

                    # Fase 4: Entrenamiento
                    status_text.markdown("**🚀 Fase 4/5: Entrenamiento del Modelo**")

                    dataset_dir_name = "mnist"
                    save_path = os.path.join("models", dataset_dir_name, f"{model_type}_trained.keras")

                    # Entrenar con progreso detallado
                    detail_text.text("Iniciando entrenamiento con data augmentation...")

                    # Crear callback personalizado para progreso detallado
                    from tensorflow.keras.callbacks import Callback

                    class TrainingProgressCallback(Callback):
                        def __init__(self, progress_bar, status_text, detail_text, epoch_progress, layer_info, total_epochs):
                            super().__init__()
                            self.progress_bar = progress_bar
                            self.status_text = status_text
                            self.detail_text = detail_text
                            self.epoch_progress = epoch_progress
                            self.layer_info = layer_info
                            self.total_epochs = total_epochs
                            self.current_epoch = 0

                        def on_epoch_begin(self, epoch, logs=None):
                            self.current_epoch = epoch + 1
                            progress = 35 + (epoch / self.total_epochs) * 55  # De 35% a 90%
                            self.progress_bar.progress(min(int(progress), 90))

                            self.epoch_progress.markdown(f"**Época {self.current_epoch}/{self.total_epochs}**")
                            self.detail_text.text(f"Procesando época {self.current_epoch} - Forward pass en capas convolucionales...")

                        def on_epoch_end(self, epoch, logs=None):
                            if logs:
                                acc = logs.get('accuracy', 0) * 100
                                val_acc = logs.get('val_accuracy', 0) * 100
                                loss = logs.get('loss', 0)
                                val_loss = logs.get('val_loss', 0)

                                self.detail_text.text(".3f")
                                self.layer_info.text(f"✓ Capas convolucionales procesadas | ✓ Backpropagation completado | ✓ Pesos actualizados")

                    # Obtener callbacks existentes y agregar el nuestro
                    existing_callbacks = cnn.get_callbacks()
                    progress_callback = TrainingProgressCallback(
                        progress_bar, status_text, detail_text, epoch_progress, layer_info, epochs
                    )
                    all_callbacks = existing_callbacks + [progress_callback]

                    # Entrenar modelo
                    history = cnn.train(
                        X_train=data['X_train'],
                        y_train=data['y_train'],
                        X_val=data['X_val'],
                        y_val=data['y_val'],
                        epochs=epochs,
                        batch_size=batch_size,
                        data_augmentation=True,
                        save_path=save_path,
                        callbacks=all_callbacks
                    )

                    # Fase 5: Finalización
                    status_text.markdown("**✅ Fase 5/5: Finalización**")
                    detail_text.text("Guardando modelo entrenado...")
                    progress_bar.progress(95)

                    epoch_progress.markdown("**Entrenamiento Completado**")
                    layer_info.markdown("**Resumen del Modelo Entrenado:**")
                    progress_bar.progress(100)
                    status_text.markdown("**🎉 ¡Entrenamiento completado exitosamente!**")

                    # Limpiar elementos de progreso detallado
                    time.sleep(1)  # Pequeña pausa para mostrar el mensaje final

                    # Mostrar resultados
                    st.success(f"Modelo {model_type} para {dataset_name} entrenado exitosamente!")

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
                    # Limpiar elementos de progreso en caso de error
                    progress_bar.empty()
                    status_text.empty()
                    detail_text.empty()
                    epoch_progress.empty()
                    layer_info.empty()

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
    dataset_models_dir = os.path.join("models", "mnist")
    model_path = None

    if os.path.exists(dataset_models_dir):
        # Buscar modelo entrenado del tipo seleccionado
        trained_model = f"{selected_model_type}_trained.keras"
        trained_path = os.path.join(dataset_models_dir, trained_model)

        if os.path.exists(trained_path):
            model_path = trained_path
        else:
            # Para residual, usar basic como fallback ya que no existe residual entrenado
            if selected_model_type == "residual":
                fallback_model = "basic_trained.keras"
                fallback_path = os.path.join(dataset_models_dir, fallback_model)
                if os.path.exists(fallback_path):
                    model_path = fallback_path
                    st.info(f"Modelo residual no disponible. Usando modelo básico entrenado.")
                else:
                    st.error(f"No se encontraron modelos entrenados para {dataset_name}.")
            else:
                st.warning(f"No se encontró modelo {selected_model_type} entrenado. Los modelos disponibles son: basic, advanced.")
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
            # Initialize session state for selected image index
            if f'selected_image_idx_{dataset_name.lower()}' not in st.session_state:
                st.session_state[f'selected_image_idx_{dataset_name.lower()}'] = 0

            # Seleccionar imagen aleatoria del test set
            if st.button(f"🎲 Seleccionar Imagen Aleatoria - {dataset_name}",
                       key=f"random_button_{dataset_name.lower()}"):
                st.session_state[f'selected_image_idx_{dataset_name.lower()}'] = np.random.randint(len(data['X_test']))
                st.rerun()

            # Slider para seleccionar imagen específica
            image_idx = st.slider(
                f"Seleccionar imagen del test set - {dataset_name}:",
                0, len(data['X_test'])-1,
                st.session_state[f'selected_image_idx_{dataset_name.lower()}'],
                key=f"slider_{dataset_name.lower()}"
            )

            # Update session state if slider changed
            if image_idx != st.session_state[f'selected_image_idx_{dataset_name.lower()}']:
                st.session_state[f'selected_image_idx_{dataset_name.lower()}'] = image_idx

            selected_image = data['X_test'][image_idx]
            true_label = data['class_names'][data['y_test'][image_idx]]

        else:  # Subir imagen
            uploaded_file = st.file_uploader(
                "Sube una imagen de dígito (28x28, escala de grises)",
                type=['png', 'jpg', 'jpeg'],
                key=f"uploader_{dataset_name.lower()}"
            )

            if uploaded_file is not None:
                try:
                    # Procesar imagen subida
                    image = Image.open(uploaded_file)

                    # Para MNIST - redimensionar a 28x28
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

            ax.imshow(display_image.squeeze(), cmap='gray')
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
                        dataset_models_dir = os.path.join("models", "mnist")
                        model_path = None

                        if os.path.exists(dataset_models_dir):
                            # Buscar modelo entrenado del tipo seleccionado
                            trained_model = f"{selected_model_type}_trained.keras"
                            trained_path = os.path.join(dataset_models_dir, trained_model)

                            if os.path.exists(trained_path):
                                model_path = trained_path
                            else:
                                # Fallback a modelo pre-entrenado (sin _trained)
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
    emoji = "🔢"
    description = "Dataset con 70,000 imágenes de dígitos escritos a mano (0-9)."

    st.markdown(f"## {emoji} {dataset_name}: {description.split(':')[0]}")
    st.markdown(description)

    # Sub-pestañas
    tabs = st.tabs(["📊 Dataset", "🚀 Entrenamiento", "📊 Evaluación", "🔮 Predicciones"])

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

        with tabs[1]:  # Entrenamiento
            show_training_section(cnn_class, data_loader, data, dataset_name, input_shape)

        with tabs[2]:  # Evaluación
            # Lazy loading - only run evaluation when tab is active
            eval_key = f"{dataset_name.lower()}_eval_active"
            if st.session_state.get(eval_key, False) or st.button(f"🔍 Ejecutar Evaluación {dataset_name}", key=f"{dataset_name.lower()}_eval_btn"):
                st.session_state[eval_key] = True
                show_evaluation_section(data_loader, data, dataset_name)
            else:
                st.info("Haz clic en 'Ejecutar Evaluación' para ver las métricas del modelo.")

        with tabs[3]:  # Predicciones
            show_predictions_section(data_loader, data, dataset_name, input_shape)

# Función para cargar datos de emociones
@st.cache_data
def load_emotion_data():
    """Carga los datos de emociones desde data/test/"""
    try:
        data_loader = EmotionDataLoader()
        data = data_loader.preprocess_data(max_samples=5000)  # Limitar muestras para Streamlit Cloud

        # Mostrar información sobre la fuente de datos
        info = data_loader.get_dataset_info()
        st.info(f"✅ Usando {info['total_samples']} imágenes reales de rostros del directorio data/test/")

        return data_loader, data
    except Exception as e:
        st.error(f"Error al cargar los datos de emociones: {e}")
        return None, None

# Función para mostrar contenido de emociones
def show_emotion_content():
    """Muestra el contenido para el dataset de emociones reales"""

    st.markdown("## 😊 Emociones - Dataset Real")
    st.markdown("**Dataset Real**: Imágenes faciales reales del directorio data/test/ organizadas por emociones.")

    st.info("📊 **Características**: Imágenes reales de rostros | 🎯 **Clases**: 7 emociones | 🔍 **Mejor para**: Demostración realista")

    # Sub-pestañas
    tabs = st.tabs(["📊 Dataset", "🚀 Entrenamiento", "📊 Evaluación", "🔮 Predicciones"])

    # Estado de carga de datos
    data_key = "emotion_data_loaded"
    if data_key not in st.session_state:
        st.session_state[data_key] = False

    # Botón para cargar datos
    if not st.session_state[data_key]:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📥 Cargar Datos de Emociones", type="primary", key="load_emotion_btn"):
                with st.spinner("Cargando imágenes reales de emociones..."):
                    try:
                        data_loader, data = load_emotion_data()

                        if data_loader and data:
                            st.session_state["datasetreal_data_loader"] = data_loader
                            st.session_state["datasetreal_data"] = data
                            st.session_state[data_key] = True
                            st.success("✅ Datos de emociones cargados exitosamente!")
                            st.rerun()
                        else:
                            st.error("❌ Error al cargar los datos de emociones")
                    except Exception as e:
                        st.error(f"❌ Error al cargar datos: {str(e)}")
            else:
                st.info("💡 Haz clic en 'Cargar Datos de Emociones' para comenzar")
                return

    # Si los datos están cargados, mostrar las pestañas
    if st.session_state[data_key]:
        data_loader = st.session_state["datasetreal_data_loader"]
        data = st.session_state["datasetreal_data"]

        with tabs[0]:  # Dataset
            show_emotion_dataset_section(data_loader, data, "Dataset Real")

        with tabs[1]:  # Entrenamiento
            show_emotion_training_section("Dataset Real")

        with tabs[2]:  # Evaluación
            eval_key = "emotion_eval_active"
            if st.session_state.get(eval_key, False) or st.button("🔍 Ejecutar Evaluación", key="emotion_eval_btn"):
                st.session_state[eval_key] = True
                show_emotion_evaluation_section(data_loader, data, "Dataset Real")
            else:
                st.info("Haz clic en 'Ejecutar Evaluación' para ver las métricas del modelo.")

        with tabs[3]:  # Predicciones
            show_emotion_predictions_section("Dataset Real")

# Función para mostrar sección de dataset de emociones
def show_emotion_dataset_section(data_loader, data, dataset_name):
    """Muestra la sección de exploración del dataset de emociones"""
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
            st.metric("Entrenamiento", f"{len(data['X_train']):,}")
            st.metric("Validación", f"{len(data['X_val']):,}")
            st.metric("Prueba", f"{len(data['X_test']):,}")

        with col2:
            st.markdown("### 📊 Distribución de Emociones")

            # Gráfico de distribución usando el visualizer
            try:
                visualizer = EmotionVisualizer()
                fig = visualizer.plot_emotion_distribution(info.get('emotion_distribution'))
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error al generar gráfico: {e}")

        st.markdown("---")

        # Información adicional sobre el dataset
        st.markdown("### 📈 Estadísticas del Dataset")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Imágenes por Emoción", f"{info['total_samples'] // info['num_classes']:,}")
        with col2:
            st.metric("Emoción más común", "Feliz 😊")
        with col3:
            st.metric("Emoción menos común", "Disgustado 🤢")

        st.markdown("---")

        # Visualizaciones adicionales
        st.markdown("### 📊 Análisis Estadístico del Dataset")

        tab1, tab2, tab3 = st.tabs(["📈 Comparación de Modelos", "🎯 Distribución de Confianza", "🖼️ Muestras de Predicciones"])

        with tab1:
            try:
                visualizer = EmotionVisualizer()
                fig = visualizer.plot_model_comparison()
                st.pyplot(fig)
                st.caption("Comparación de rendimiento entre diferentes arquitecturas de modelo")
            except Exception as e:
                st.error(f"Error generando comparación de modelos: {e}")

        with tab2:
            try:
                visualizer = EmotionVisualizer()
                fig = visualizer.plot_prediction_confidence()
                st.pyplot(fig)
                st.caption("Distribución de confianza en las predicciones del modelo")
            except Exception as e:
                st.error(f"Error generando distribución de confianza: {e}")

        with tab3:
            try:
                visualizer = EmotionVisualizer()
                fig = visualizer.plot_sample_predictions()
                st.pyplot(fig)
                st.caption("Ejemplos de predicciones en imágenes de prueba")
            except Exception as e:
                st.error(f"Error generando muestras de predicciones: {e}")

        st.markdown("---")
        st.markdown(f"### 🖼️ Muestras del Dataset {dataset_name}")

        if st.button(f"🔄 Generar Nuevas Muestras - {dataset_name}"):
            st.rerun()

        # Mostrar muestras aleatorias
        num_samples = 20
        indices = np.random.choice(len(data['X_train']), num_samples, replace=False)

        cols = st.columns(5)
        emotion_emojis = {
            0: '😠', 1: '🤢', 2: '😨', 3: '😊', 4: '😢', 5: '😲', 6: '😐'
        }

        for i, idx in enumerate(indices):
            with cols[i % 5]:
                image = data['X_train'][idx]
                label_idx = np.argmax(data['y_train'][idx])
                label = data_loader.emotions[label_idx]
                emoji = emotion_emojis.get(label_idx, '❓')

                # Convertir a imagen PIL
                img_array = (image.squeeze() * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_array, mode='L')

                st.image(pil_image, caption=f"{emoji} {label}", width=100, use_container_width=False)

    else:
        st.error(f"No se pudieron cargar los datos del dataset {dataset_name}.")

# Función para mostrar sección de modelo de emociones
def show_emotion_model_section():
    """Muestra la sección de arquitectura del modelo de emociones"""
    st.header("🧠 Arquitectura del Modelo CNN - FER2013")

    st.info("💡 **Esta sección es informativa:** Explora la arquitectura del modelo de clasificación de emociones.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Arquitectura del Modelo")

        st.markdown("""
        **4 Bloques Convolucionales:**
        - Conv2D (32 filtros) → BatchNorm → Conv2D (32) → MaxPool → Dropout
        - Conv2D (64 filtros) → BatchNorm → Conv2D (64) → MaxPool → Dropout
        - Conv2D (128 filtros) → BatchNorm → Conv2D (128) → MaxPool → Dropout
        - Conv2D (256 filtros) → BatchNorm → Conv2D (256) → MaxPool → Dropout

        **Capas Densas:**
        - Dense(512) → BatchNorm → Dropout(0.5)
        - Dense(256) → BatchNorm → Dropout(0.5)
        - Dense(7) → Softmax
        """)

        st.markdown("### 🔧 Hiperparámetros")
        st.markdown("""
        - **Optimizador**: Adam (lr=0.001)
        - **Pérdida**: Categorical Crossentropy
        - **Batch Size**: 64
        - **Early Stopping**: Paciencia=15
        - **Aumento de Datos**: Rotación, zoom, flip
        """)

    with col2:
        st.markdown("### 📋 Resumen del Modelo")

        # Crear modelo temporal para mostrar resumen
        try:
            classifier = EmotionClassifier()
            summary_text = classifier.get_model_summary()
            st.code(summary_text, language="text")
        except Exception as e:
            st.error(f"Error al crear modelo: {e}")

# Función para mostrar sección de entrenamiento de emociones
def show_emotion_training_section(selected_dataset):
    """Muestra la sección de entrenamiento del modelo de emociones"""
    st.header(f"🚀 Entrenamiento del Modelo - {selected_dataset}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🎯 Configuración de Entrenamiento")

        model_type = st.selectbox("Tipo de Modelo", ["basic", "advanced", "residual"],
                                key=f"train_model_select_{selected_dataset.lower()}")
        epochs = st.slider("Épocas", 5, 50, 10, key=f"train_epochs_{selected_dataset.lower()}")
        batch_size = st.slider("Batch Size", 16, 128, 64, key=f"train_batch_{selected_dataset.lower()}")

        train_button_key = f"train_{selected_dataset.lower()}_btn"
        if st.button(f"🚀 Iniciar Entrenamiento {selected_dataset}", type="primary", key=train_button_key):
            st.session_state[f'{selected_dataset.lower()}_training_started'] = True
            # Store training parameters in session state with different keys
            st.session_state[f'{selected_dataset.lower()}_training_model_type'] = model_type
            st.session_state[f'{selected_dataset.lower()}_training_epochs'] = epochs
            st.session_state[f'{selected_dataset.lower()}_training_batch_size'] = batch_size
            st.rerun()

    with col2:
        training_key = f'{selected_dataset.lower()}_training_started'
        if training_key in st.session_state and st.session_state[training_key]:
            st.markdown("### 📈 Progreso del Entrenamiento")

            # Contenedor para progreso detallado
            progress_container = st.container()
            with progress_container:
                # Barra de progreso principal
                progress_bar = st.progress(0)
                status_text = st.empty()
                detail_text = st.empty()
                epoch_progress = st.empty()
                layer_info = st.empty()

                try:
                    # Fase 1: Preparación de datos
                    status_text.markdown("**🔄 Fase 1/5: Preparación de Datos**")
                    detail_text.text("Verificando y preprocesando datos de entrenamiento...")
                    progress_bar.progress(5)

                    # Fase 2: Construcción del modelo
                    status_text.markdown("**🏗️ Fase 2/5: Construcción del Modelo**")
                    model_type = st.session_state[f'{selected_dataset.lower()}_training_model_type']

                    # Mostrar arquitectura del modelo que se va a construir
                    dataset_name = "FER2013" if selected_dataset == "FER2013" else "ExpW"
                    if model_type == "basic":
                        layer_info.markdown(f"""
                        **Construyendo Modelo Básico {dataset_name}:**
                        - Capa Conv2D (32 filtros, 3x3) + ReLU + MaxPooling
                        - Capa Conv2D (64 filtros, 3x3) + ReLU + MaxPooling
                        - Flatten + Dense(128) + Dropout(25%) + Dense(7)
                        """)
                    elif model_type == "advanced":
                        layer_info.markdown(f"""
                        **Construyendo Modelo Avanzado {dataset_name}:**
                        - Capa Conv2D (32 filtros) + BatchNorm + ReLU + MaxPooling
                        - Capa Conv2D (64 filtros) + BatchNorm + ReLU + MaxPooling
                        - Capa Conv2D (128 filtros) + BatchNorm + ReLU + MaxPooling
                        - GlobalAveragePooling + Dense(7)
                        """)
                    else:  # residual
                        layer_info.markdown(f"""
                        **Construyendo Modelo Residual {dataset_name}:**
                        - Bloque Residual 1: Conv2D + BatchNorm + ReLU + Conv2D + Skip Connection
                        - Bloque Residual 2: Conv2D + BatchNorm + ReLU + Conv2D + Skip Connection
                        - GlobalAveragePooling + Dense(7)
                        """)

                    detail_text.text("Configurando capas convolucionales y conexiones...")
                    progress_bar.progress(15)

                    # Configurar y construir modelo
                    classifier = EmotionClassifier()
                    # Create the specific model type
                    classifier.model = classifier.create_model(model_type=model_type)
                    classifier.compile_model(learning_rate=1e-4)
                    progress_bar.progress(25)

                    # Fase 3: Configuración del entrenamiento
                    status_text.markdown("**⚙️ Fase 3/5: Configuración del Entrenamiento**")
                    detail_text.text("Preparando generadores de datos y callbacks...")

                    # Obtener datos desde session state
                    dataset_key = selected_dataset.lower().replace(" ", "")
                    data_loader = st.session_state[f"{dataset_key}_data_loader"]
                    data = st.session_state[f"{dataset_key}_data"]

                    epochs = st.session_state[f'{selected_dataset.lower()}_training_epochs']
                    batch_size = st.session_state[f'{selected_dataset.lower()}_training_batch_size']

                    # Calcular pesos de clase para datos desbalanceados
                    class_weights = data_loader.get_class_weights()

                    # Calcular número total de pasos
                    total_samples = len(data['X_train'])
                    steps_per_epoch = total_samples // batch_size

                    layer_info.markdown(f"""
                    **Parámetros de Entrenamiento:**
                    - Épocas: {epochs}
                    - Tamaño de batch: {batch_size}
                    - Muestras totales: {total_samples:,}
                    - Pasos por época: {steps_per_epoch}
                    - Pesos de clase: Activados (datos desbalanceados)
                    - Aumento de datos: Activado
                    """)
                    progress_bar.progress(35)

                    # Fase 4: Entrenamiento
                    status_text.markdown("**🚀 Fase 4/5: Entrenamiento del Modelo**")

                    dataset_dir_name = "emotion"
                    save_path = os.path.join("models", dataset_dir_name, f"{model_type}_{selected_dataset.lower()}_trained.keras")

                    # Entrenar con progreso detallado
                    detail_text.text("Iniciando entrenamiento con data augmentation...")

                    # Crear callback personalizado para progreso detallado
                    from tensorflow.keras.callbacks import Callback

                    class TrainingProgressCallback(Callback):
                        def __init__(self, progress_bar, status_text, detail_text, epoch_progress, layer_info, total_epochs):
                            super().__init__()
                            self.progress_bar = progress_bar
                            self.status_text = status_text
                            self.detail_text = detail_text
                            self.epoch_progress = epoch_progress
                            self.layer_info = layer_info
                            self.total_epochs = total_epochs
                            self.current_epoch = 0

                        def on_epoch_begin(self, epoch, logs=None):
                            self.current_epoch = epoch + 1
                            progress = 35 + (epoch / self.total_epochs) * 55  # De 35% a 90%
                            self.progress_bar.progress(min(int(progress), 90))

                            self.epoch_progress.markdown(f"**Época {self.current_epoch}/{self.total_epochs}**")
                            self.detail_text.text(f"Procesando época {self.current_epoch} - Forward pass en capas convolucionales...")

                        def on_epoch_end(self, epoch, logs=None):
                            if logs:
                                acc = logs.get('accuracy', 0) * 100
                                val_acc = logs.get('val_accuracy', 0) * 100
                                loss = logs.get('loss', 0)
                                val_loss = logs.get('val_loss', 0)

                                self.detail_text.text(".3f")
                                self.layer_info.text(f"✓ Capas convolucionales procesadas | ✓ Backpropagation completado | ✓ Pesos actualizados")

                    # Obtener callbacks existentes y agregar el nuestro
                    existing_callbacks = classifier.model.callbacks if hasattr(classifier.model, 'callbacks') else []
                    progress_callback = TrainingProgressCallback(
                        progress_bar, status_text, detail_text, epoch_progress, layer_info, epochs
                    )
                    all_callbacks = existing_callbacks + [progress_callback]

                    # Entrenar modelo
                    history = classifier.train(
                        X_train=data['X_train'],
                        y_train=data['y_train'],
                        X_val=data['X_val'],
                        y_val=data['y_val'],
                        epochs=epochs,
                        batch_size=batch_size,
                        class_weights=class_weights,
                        save_path=save_path,
                        callbacks=all_callbacks
                    )

                    # Fase 5: Finalización
                    status_text.markdown("**✅ Fase 5/5: Finalización**")
                    detail_text.text("Guardando modelo entrenado...")
                    progress_bar.progress(95)

                    epoch_progress.markdown("**Entrenamiento Completado**")
                    layer_info.markdown("**Resumen del Modelo Entrenado:**")
                    progress_bar.progress(100)
                    status_text.markdown("**🎉 ¡Entrenamiento completado exitosamente!**")

                    # Limpiar elementos de progreso detallado
                    time.sleep(1)  # Pequeña pausa para mostrar el mensaje final

                    # Mostrar resultados
                    st.success(f"Modelo {model_type} para {selected_dataset} entrenado exitosamente!")

                    # Métricas finales
                    final_acc = history.history['val_accuracy'][-1]
                    final_loss = history.history['val_loss'][-1]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy Final", f"{final_acc:.4f}")
                    with col2:
                        st.metric("Loss Final", f"{final_loss:.4f}")

                    # Gráfico de curvas de aprendizaje
                    st.markdown("### 📊 Curvas de Aprendizaje")
                    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                    axes[0].plot(history.history['accuracy'], label='Entrenamiento')
                    axes[0].plot(history.history['val_accuracy'], label='Validación')
                    axes[0].set_title('Accuracy vs Épocas')
                    axes[0].set_xlabel('Épocas')
                    axes[0].set_ylabel('Accuracy')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

                    axes[1].plot(history.history['loss'], label='Entrenamiento')
                    axes[1].plot(history.history['val_loss'], label='Validación')
                    axes[1].set_title('Loss vs Épocas')
                    axes[1].set_xlabel('Épocas')
                    axes[1].set_ylabel('Loss')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")
                    # Limpiar elementos de progreso en caso de error
                    progress_bar.empty()
                    status_text.empty()
                    detail_text.empty()
                    epoch_progress.empty()
                    layer_info.empty()

        else:
            st.info("Configura los parámetros y haz clic en 'Iniciar Entrenamiento'")

# Función para mostrar sección de evaluación de emociones
def show_emotion_evaluation_section(data_loader, data, selected_dataset):
    """Muestra la sección de evaluación del modelo de emociones"""
    st.header(f"📊 Evaluación del Modelo - {selected_dataset}")

    # Seleccionar tipo de modelo para evaluación
    model_options = ["basic", "advanced", "residual"]
    selected_model_type = st.selectbox(
        "Selecciona el tipo de modelo para evaluar:",
        model_options,
        key=f"eval_model_type_{selected_dataset.lower()}"
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
    dataset_models_dir = os.path.join("models", "emotion")
    model_path = None

    if os.path.exists(dataset_models_dir):
        # Buscar modelo entrenado del tipo seleccionado con nombre del dataset
        trained_model = f"{selected_model_type}_{selected_dataset.lower()}_trained.keras"
        trained_path = os.path.join(dataset_models_dir, trained_model)

        if os.path.exists(trained_path):
            model_path = trained_path
        else:
            # Fallback a modelo entrenado sin especificar dataset
            fallback_trained = f"{selected_model_type}_trained.keras"
            fallback_path = os.path.join(dataset_models_dir, fallback_trained)
            if os.path.exists(fallback_path):
                model_path = fallback_path
                st.warning(f"No se encontró modelo {selected_model_type} entrenado para {selected_dataset}. Usando modelo general.")
            else:
                # Último fallback a modelo básico
                basic_model = "emotion_model.h5"
                basic_path = os.path.join(dataset_models_dir, basic_model)
                if os.path.exists(basic_path):
                    model_path = basic_path
                    st.warning(f"No se encontró modelo {selected_model_type} entrenado. Usando modelo básico.")
                else:
                    st.error(f"No se encontró ningún modelo para {selected_dataset}.")
    else:
        st.error("Directorio de modelos para emociones no encontrado.")

    if model_path is not None and os.path.exists(model_path):
        try:
            # Cargar modelo entrenado
            classifier = EmotionClassifier(model_path=model_path)

            st.markdown("### 🎯 Métricas de Evaluación")

            # Evaluar modelo - convertir etiquetas one-hot a índices de clase
            evaluator = ModelEvaluator(class_names=data_loader.emotions)
            # Convertir etiquetas one-hot encoded a índices de clase
            y_test_indices = np.argmax(data['y_test'], axis=1)
            results = evaluator.evaluate_model(
                classifier.model, data['X_test'], y_test_indices
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
                          xticklabels=data_loader.emotions,
                          yticklabels=data_loader.emotions,
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
                          xticklabels=data_loader.emotions,
                          yticklabels=data_loader.emotions,
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
            for class_name in data_loader.emotions:
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

        except Exception as e:
            st.error(f"Error al evaluar el modelo: {str(e)}")
            st.error("Asegúrate de que el modelo esté entrenado y sea compatible.")
    else:
        st.warning("No se encontró un modelo entrenado. Entrena un modelo primero en la sección 'Entrenamiento'.")

# Función para mostrar sección de predicciones de emociones
def show_emotion_predictions_section(selected_dataset):
    """Muestra la sección de predicciones de emociones"""
    st.header(f"🔮 Predicciones de Emociones - {selected_dataset}")

    st.markdown("### 📸 Subir Imagen Facial o Usar Cámara")

    # Option to choose between file upload or camera
    input_option = st.radio(
        "Método de entrada:",
        ["Subir imagen", "Usar cámara"],
        key=f"input_option_{selected_dataset.lower()}"
    )

    uploaded_file = None
    camera_image = None

    if input_option == "Subir imagen":
        uploaded_file = st.file_uploader(
            "Elige una imagen facial",
            type=["jpg", "jpeg", "png"],
            key=f"emotion_uploader_{selected_dataset.lower()}"
        )
    else:  # Usar cámara
        camera_image = st.camera_input(
            "Captura una imagen facial",
            key=f"emotion_camera_{selected_dataset.lower()}"
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif camera_image is not None:
        image = Image.open(camera_image)
    else:
        image = None

    if image is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Imagen Subida")
            st.image(image, caption="Imagen facial", use_container_width=True)

        with col2:
            st.subheader("🎭 Resultado de Predicción")

            # Model selection for emotions
            emotion_model_options = ["basic", "advanced", "residual"]
            selected_emotion_model = st.selectbox(
                "Selecciona el tipo de modelo:",
                emotion_model_options,
                key=f"emotion_model_select_{selected_dataset.lower()}"
            )

            try:
                # Load selected emotion model
                dataset_models_dir = os.path.join("models", "emotion")
                model_path = None

                if os.path.exists(dataset_models_dir):
                    # Try trained model first with dataset name
                    trained_model = f"{selected_emotion_model}_{selected_dataset.lower()}_trained.keras"
                    trained_path = os.path.join(dataset_models_dir, trained_model)

                    if os.path.exists(trained_path):
                        model_path = trained_path
                    else:
                        # Fallback to trained model without dataset name
                        fallback_trained = f"{selected_emotion_model}_trained.keras"
                        fallback_path = os.path.join(dataset_models_dir, fallback_trained)
                        if os.path.exists(fallback_path):
                            model_path = fallback_path
                            st.info(f"No se encontró modelo {selected_emotion_model} entrenado para {selected_dataset}. Usando modelo general.")
                        else:
                            # Last fallback to basic model
                            basic_model = "emotion_model.h5"
                            basic_path = os.path.join(dataset_models_dir, basic_model)
                            if os.path.exists(basic_path):
                                model_path = basic_path
                                st.info(f"No se encontró modelo {selected_emotion_model} entrenado. Usando modelo básico.")

                if model_path:
                    classifier = EmotionClassifier(model_path=model_path)

                    # Hacer predicción
                    prediction, confidence, probabilities = classifier.predict(image)

                    # Mostrar predicción principal
                    emotion_emojis = {
                        'feliz': '😊', 'triste': '😢', 'neutral': '😐',
                        'enojado': '😠', 'sorprendido': '😲', 'asustado': '😨', 'disgustado': '🤢'
                    }

                    emotion_display = prediction.title()
                    emoji = emotion_emojis.get(prediction.lower(), '❓')
                    st.success(f"{emoji} **{emotion_display}** ({confidence:.1%} confianza)")

                    # Mostrar probabilidades
                    st.subheader("📊 Probabilidades por Emoción")
                    prob_rows = []
                    for emotion, prob in probabilities.items():
                        emoji_prob = emotion_emojis.get(emotion.lower(), '❓')
                        prob_rows.append({
                            'Emoción': f"{emoji_prob} {emotion.title()}",
                            'Probabilidad': f"{prob:.1%}"
                        })

                    prob_df = pd.DataFrame(prob_rows)
                    st.dataframe(prob_df, use_container_width=True)
                else:
                    st.error(f"No se encontró un modelo de emociones entrenado para {selected_dataset}.")

            except Exception as e:
                st.error(f"Error en predicción: {str(e)}")


    else:
        st.info("Sube una imagen facial para realizar una predicción de emoción.")

# Crear pestañas principales para datasets
main_tabs = st.tabs(["🔢 MNIST - Dígitos", "😊 Emociones - Dataset Real"])

with main_tabs[0]:  # MNIST
    show_tab_content("MNIST", MNISTCNN, (28, 28, 1))

with main_tabs[1]:  # Emociones
    show_emotion_content()

# Footer
st.markdown("---")
st.markdown("### 📚 Tecnologías Utilizadas")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    **Datasets:**
    - MNIST (70,000 imágenes de dígitos)
    - Dataset Real (imágenes faciales organizadas por emociones)
    """)

with col2:
    st.markdown("""
    **Framework:**
    - TensorFlow/Keras
    - OpenCV
    """)

with col3:
    st.markdown("""
    **Interfaz:**
    - Streamlit
    - Matplotlib/Seaborn
    """)

with col4:
    st.markdown("""
    **Características:**
    - CNN Models
    - Grad-CAM
    - Real-time Training
    """)

st.markdown("---")
st.markdown("**© 2025 - Aplicación de Demostración CNN con Reconocimiento de Emociones**")
