"""
Módulo de carga y preprocesamiento de datos para el proyecto CNN CIFAR-10
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import pickle
import os

# Configurar estilo de gráficos
plt.style.use('default')
sns.set_palette("husl")

class CIFAR10DataLoader:
    """
    Clase para cargar y preprocesar el dataset CIFAR-10
    """

    def __init__(self, validation_split: float = 0.1, random_state: int = 42):
        """
        Inicializa el cargador de datos

        Args:
            validation_split: Proporción de datos para validación
            random_state: Semilla para reproducibilidad
        """
        self.validation_split = validation_split
        self.random_state = random_state
        self.class_names = [
            'Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
            'Perro', 'Rana', 'Caballo', 'Barco', 'Camión'
        ]
        self.scaler = StandardScaler()

        # Variables para almacenar datos
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def load_data(self) -> Dict[str, Any]:
        """
        Carga el dataset CIFAR-10 y realiza preprocesamiento básico

        Returns:
            Diccionario con los datos divididos
        """
        print("Cargando dataset CIFAR-10...")

        # Cargar datos
        (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

        # Convertir etiquetas a formato adecuado
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()

        # Dividir datos de entrenamiento en train y validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=y_train_full
        )

        # Normalizar imágenes (0-255 -> 0-1)
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Almacenar datos
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        print(f"Datos cargados:")
        print(f"  Train: {X_train.shape[0]} muestras")
        print(f"  Validation: {X_val.shape[0]} muestras")
        print(f"  Test: {X_test.shape[0]} muestras")
        print(f"  Dimensiones de imagen: {X_train.shape[1:]}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_names': self.class_names
        }

    def get_data_augmentation(self):
        """
        Retorna un generador de aumento de datos para entrenamiento

        Returns:
            ImageDataGenerator configurado
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )

        return datagen

    def visualize_samples(self, num_samples: int = 10, save_path: str = None):
        """
        Visualiza muestras aleatorias del dataset

        Args:
            num_samples: Número de muestras a mostrar
            save_path: Ruta para guardar la imagen (opcional)
        """
        if self.X_train is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        # Seleccionar índices aleatorios
        indices = np.random.choice(len(self.X_train), num_samples, replace=False)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i, idx in enumerate(indices):
            if i >= 10:  # Máximo 10 muestras
                break

            image = self.X_train[idx]
            label = self.class_names[self.y_train[idx]]

            axes[i].imshow(image)
            axes[i].set_title(f'Clase: {label}', fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def plot_class_distribution(self, save_path: str = None):
        """
        Visualiza la distribución de clases en el dataset

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        if self.y_train is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Distribución en conjunto de entrenamiento
        train_counts = np.bincount(self.y_train)
        axes[0].bar(self.class_names, train_counts, color='skyblue', alpha=0.7)
        axes[0].set_title('Distribución de Clases - Entrenamiento')
        axes[0].set_ylabel('Número de muestras')
        axes[0].tick_params(axis='x', rotation=45)

        # Distribución en conjunto de validación
        val_counts = np.bincount(self.y_val)
        axes[1].bar(self.class_names, val_counts, color='lightgreen', alpha=0.7)
        axes[1].set_title('Distribución de Clases - Validación')
        axes[1].set_ylabel('Número de muestras')
        axes[1].tick_params(axis='x', rotation=45)

        # Distribución en conjunto de prueba
        test_counts = np.bincount(self.y_test)
        axes[2].bar(self.class_names, test_counts, color='salmon', alpha=0.7)
        axes[2].set_title('Distribución de Clases - Prueba')
        axes[2].set_ylabel('Número de muestras')
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Retorna información detallada del dataset

        Returns:
            Diccionario con información del dataset
        """
        if self.X_train is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        info = {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'image_shape': self.X_train.shape[1:],
            'total_samples': len(self.X_train) + len(self.X_val) + len(self.X_test),
            'train_split': len(self.X_train) / (len(self.X_train) + len(self.X_val) + len(self.X_test)),
            'val_split': len(self.X_val) / (len(self.X_train) + len(self.X_val) + len(self.X_test)),
            'test_split': len(self.X_test) / (len(self.X_train) + len(self.X_val) + len(self.X_test))
        }

        return info


class MNISTDataLoader:
    """
    Clase para cargar y preprocesar el dataset MNIST
    """

    def __init__(self, validation_split: float = 0.1, random_state: int = 42):
        """
        Inicializa el cargador de datos

        Args:
            validation_split: Proporción de datos para validación
            random_state: Semilla para reproducibilidad
        """
        self.validation_split = validation_split
        self.random_state = random_state
        self.class_names = [str(i) for i in range(10)]  # Dígitos del 0 al 9
        self.scaler = StandardScaler()

        # Variables para almacenar datos
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def load_data(self) -> Dict[str, Any]:
        """
        Carga el dataset MNIST y realiza preprocesamiento básico

        Returns:
            Diccionario con los datos divididos
        """
        print("Cargando dataset MNIST...")

        # Cargar datos
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

        # Dividir datos de entrenamiento en train y validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=y_train_full
        )

        # Normalizar imágenes (0-255 -> 0-1)
        X_train = X_train.astype('float32') / 255.0
        X_val = X_val.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Expandir dimensiones para formato de canal (28, 28) -> (28, 28, 1)
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        # Almacenar datos
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        print(f"Datos cargados:")
        print(f"  Train: {X_train.shape[0]} muestras")
        print(f"  Validation: {X_val.shape[0]} muestras")
        print(f"  Test: {X_test.shape[0]} muestras")
        print(f"  Dimensiones de imagen: {X_train.shape[1:]}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'class_names': self.class_names
        }

    def get_data_augmentation(self):
        """
        Retorna un generador de aumento de datos para entrenamiento

        Returns:
            ImageDataGenerator configurado
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )

        return datagen

    def visualize_samples(self, num_samples: int = 10, save_path: str = None):
        """
        Visualiza muestras aleatorias del dataset

        Args:
            num_samples: Número de muestras a mostrar
            save_path: Ruta para guardar la imagen (opcional)
        """
        if self.X_train is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        # Seleccionar índices aleatorios
        indices = np.random.choice(len(self.X_train), num_samples, replace=False)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()

        for i, idx in enumerate(indices):
            if i >= 10:  # Máximo 10 muestras
                break

            image = self.X_train[idx].squeeze()  # Remover dimensión de canal para visualización
            label = self.class_names[self.y_train[idx]]

            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Dígito: {label}', fontsize=10)
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def plot_class_distribution(self, save_path: str = None):
        """
        Visualiza la distribución de clases en el dataset

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        if self.y_train is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Distribución en conjunto de entrenamiento
        train_counts = np.bincount(self.y_train)
        axes[0].bar(self.class_names, train_counts, color='skyblue', alpha=0.7)
        axes[0].set_title('Distribución de Dígitos - Entrenamiento')
        axes[0].set_ylabel('Número de muestras')
        axes[0].tick_params(axis='x', rotation=0)

        # Distribución en conjunto de validación
        val_counts = np.bincount(self.y_val)
        axes[1].bar(self.class_names, val_counts, color='lightgreen', alpha=0.7)
        axes[1].set_title('Distribución de Dígitos - Validación')
        axes[1].set_ylabel('Número de muestras')
        axes[1].tick_params(axis='x', rotation=0)

        # Distribución en conjunto de prueba
        test_counts = np.bincount(self.y_test)
        axes[2].bar(self.class_names, test_counts, color='salmon', alpha=0.7)
        axes[2].set_title('Distribución de Dígitos - Prueba')
        axes[2].set_ylabel('Número de muestras')
        axes[2].tick_params(axis='x', rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Imagen guardada en: {save_path}")

        plt.show()

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Retorna información detallada del dataset

        Returns:
            Diccionario con información del dataset
        """
        if self.X_train is None:
            raise ValueError("Debe cargar los datos primero usando load_data()")

        info = {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'image_shape': self.X_train.shape[1:],
            'total_samples': len(self.X_train) + len(self.X_val) + len(self.X_test),
            'train_split': len(self.X_train) / (len(self.X_train) + len(self.X_val) + len(self.X_test)),
            'val_split': len(self.X_val) / (len(self.X_train) + len(self.X_val) + len(self.X_test)),
            'test_split': len(self.X_test) / (len(self.X_train) + len(self.X_val) + len(self.X_test))
        }

        return info


def save_data_splits(data_dict: Dict[str, Any], save_dir: str = "data_splits"):
    """
    Guarda los splits de datos en archivos pickle

    Args:
        data_dict: Diccionario con los datos
        save_dir: Directorio donde guardar los archivos
    """
    os.makedirs(save_dir, exist_ok=True)

    for key, value in data_dict.items():
        if key.startswith(('X_', 'y_')):
            filepath = os.path.join(save_dir, f"{key}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)
            print(f"Datos guardados en: {filepath}")


def load_data_splits(save_dir: str = "data_splits") -> Dict[str, Any]:
    """
    Carga los splits de datos desde archivos pickle

    Args:
        save_dir: Directorio desde donde cargar los archivos

    Returns:
        Diccionario con los datos cargados
    """
    data_dict = {}

    for filename in os.listdir(save_dir):
        if filename.endswith('.pkl'):
            key = filename[:-4]  # Remover extensión .pkl
            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'rb') as f:
                data_dict[key] = pickle.load(f)
            print(f"Datos cargados desde: {filepath}")

    return data_dict


if __name__ == "__main__":
    # Ejemplo de uso
    loader = CIFAR10DataLoader()
    data = loader.load_data()

    # Mostrar información del dataset
    info = loader.get_dataset_info()
    print("\nInformación del Dataset:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Visualizar muestras
    loader.visualize_samples(num_samples=10, save_path="samples.png")

    # Visualizar distribución de clases
    loader.plot_class_distribution(save_path="class_distribution.png")
