# CNN Streamlit App - Grupo 9

Aplicación web interactiva para demostración de modelos de Redes Neuronales Convolucionales (CNN) con datasets CIFAR-10 y MNIST.

## 🚀 Despliegue en Streamlit Cloud

Esta aplicación está preparada para ser desplegada en Streamlit Cloud.

### Requisitos del Sistema

- Python 3.9
- TensorFlow 2.15.0+
- Keras 3.0.0+
- Streamlit 1.28+

### Instalación Local

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd cnn-streamlit-app
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación:
```bash
streamlit run app.py
```

### Despliegue en Streamlit Cloud

1. **Sube el código a GitHub**: Asegúrate de que todo el código esté en un repositorio de GitHub público.

2. **Archivos necesarios para el despliegue**:
   - `app.py` (archivo principal)
   - `requirements.txt` (dependencias)
   - `src/` (directorio con módulos)
   - `models/` (directorio con modelos entrenados)

3. **Ve a [Streamlit Cloud](https://streamlit.io/cloud)** y conecta tu repositorio de GitHub.

4. **Configura el despliegue**:
   - **Main file path**: `app.py`
   - El resto de configuraciones usarán los valores por defecto

5. **Haz clic en "Deploy"**

### ⚠️ Consideraciones para el Despliegue

- **Tamaño de la aplicación**: La aplicación incluye modelos de ~50-100MB, lo que puede requerir tiempo para cargar inicialmente
- **Límites de Streamlit Cloud**: El plan gratuito tiene límites de recursos. Para uso intensivo, considera actualizar a un plan pago
- **Tiempo de carga inicial**: La primera carga puede tomar tiempo debido al tamaño de TensorFlow/Keras
- **Almacenamiento**: Los modelos están incluidos en el repositorio. Asegúrate de que no excedan los límites de GitHub

## 📁 Estructura del Proyecto

```
cnn-streamlit-app/
├── app.py                    # Archivo principal de Streamlit
├── requirements.txt          # Dependencias del proyecto
├── src/                      # Módulos de Python
│   ├── __init__.py
│   ├── data.py              # Carga de datasets
│   ├── model.py             # Definición de modelos CNN
│   ├── evaluation.py        # Evaluación de modelos
│   └── utils.py             # Utilidades
├── models/                   # Modelos entrenados
│   ├── cifar10/
│   │   ├── basic_trained.keras
│   │   └── advanced_model.keras
│   └── mnist/
│       ├── basic_trained.keras
│       └── advanced_trained.keras
├── training_history/         # Historial de entrenamiento
└── README.md                # Este archivo
```

## 🎯 Funcionalidades

### Datasets Disponibles
- **CIFAR-10**: 60,000 imágenes de 32x32 píxeles en 10 categorías
- **MNIST**: 70,000 imágenes de dígitos escritos a mano (0-9)

### Modelos Disponibles
- **Básico**: Arquitectura CNN simple
- **Avanzado**: CNN con Batch Normalization y regularización
- **Residual**: Arquitectura con bloques residuales (solo CIFAR-10)

### Secciones de la Aplicación
1. **📊 Dataset**: Exploración y visualización de datos
2. **🧠 Modelo**: Arquitectura y configuración de modelos
3. **🚀 Entrenamiento**: Entrenamiento de modelos desde cero
4. **📊 Evaluación**: Métricas de rendimiento y matrices de confusión
5. **🔮 Predicciones**: Clasificación de imágenes en tiempo real

## 🔧 Tecnologías Utilizadas

- **TensorFlow/Keras**: Framework de deep learning
- **Streamlit**: Framework web para aplicaciones de datos
- **NumPy**: Computación numérica
- **Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización de datos
- **PIL**: Procesamiento de imágenes
- **Scikit-learn**: Métricas de evaluación

## 📈 Rendimiento de Modelos

### CIFAR-10
- **Modelo Básico**: ~75-80% accuracy
- **Modelo Avanzado**: ~85-90% accuracy
- **Modelo Residual**: ~87-92% accuracy

### MNIST
- **Modelo Básico**: ~98-99% accuracy
- **Modelo Avanzado**: ~99%+ accuracy

## 🤝 Contribuidores

- Grupo 9 - Algoritmos de Machine Learning

## 📄 Licencia

Este proyecto es parte de un trabajo académico del Grupo 9.
