"""
Módulo de definición de la arquitectura CNN para clasificación de CIFAR-10
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional


class CIFAR10CNN:
    """
    Clase para crear y gestionar modelos CNN para CIFAR-10
    """

    def __init__(self, input_shape: tuple = (32, 32, 3), num_classes: int = 10):
        """
        Inicializa la clase CNN

        Args:
            input_shape: Dimensiones de entrada de las imágenes
            num_classes: Número de clases para clasificación
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self, model_type: str = 'basic', **kwargs) -> Model:
        """
        Construye el modelo CNN según el tipo especificado

        Args:
            model_type: Tipo de modelo ('basic', 'advanced', 'residual')
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            Modelo compilado
        """
        if model_type == 'basic':
            self.model = self._build_basic_cnn(**kwargs)
        elif model_type == 'advanced':
            self.model = self._build_advanced_cnn(**kwargs)
        elif model_type == 'residual':
            self.model = self._build_residual_cnn(**kwargs)
        else:
            raise ValueError(f"Tipo de modelo '{model_type}' no reconocido")

        return self.model

    def _build_basic_cnn(self, filters: list = [32, 64, 128],
                        kernel_size: tuple = (3, 3),
                        dropout_rate: float = 0.5,
                        learning_rate: float = 0.001) -> Model:
        """
        Construye un modelo CNN básico

        Args:
            filters: Lista con número de filtros por capa convolucional
            kernel_size: Tamaño del kernel para convoluciones
            dropout_rate: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo compilado
        """
        model = Sequential(name='Basic_CNN')

        # Primera capa convolucional
        model.add(Conv2D(filters[0], kernel_size, activation='relu',
                        input_shape=self.input_shape, padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Segunda capa convolucional
        model.add(Conv2D(filters[1], kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Tercera capa convolucional
        model.add(Conv2D(filters[2], kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Capas fully connected
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _build_advanced_cnn(self, filters: list = [64, 128, 256, 512],
                           kernel_size: tuple = (3, 3),
                           dropout_rate: float = 0.3,
                           learning_rate: float = 0.0001) -> Model:
        """
        Construye un modelo CNN avanzado con Batch Normalization y regularización

        Args:
            filters: Lista con número de filtros por capa convolucional
            kernel_size: Tamaño del kernel para convoluciones
            dropout_rate: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo compilado
        """
        model = Sequential(name='Advanced_CNN')

        # Primera capa convolucional con Batch Normalization
        model.add(Conv2D(filters[0], kernel_size, input_shape=self.input_shape,
                        padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Segunda capa convolucional
        model.add(Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Tercera capa convolucional
        model.add(Conv2D(filters[2], kernel_size, padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Cuarta capa convolucional
        model.add(Conv2D(filters[3], kernel_size, padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Capas fully connected
        model.add(Flatten())
        model.add(Dense(1024, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _build_residual_cnn(self, num_blocks: int = 3,
                           filters: int = 64,
                           learning_rate: float = 0.001) -> Model:
        """
        Construye un modelo CNN con bloques residuales simplificados

        Args:
            num_blocks: Número de bloques residuales
            filters: Número base de filtros
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo compilado
        """
        inputs = Input(shape=self.input_shape)

        # Primera capa convolucional
        x = Conv2D(filters, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Bloques residuales
        for i in range(num_blocks):
            # Guardar entrada del bloque
            shortcut = x

            # Primera convolución del bloque
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            # Segunda convolución del bloque
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)

            # Conexión residual
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)

            # Max pooling cada 2 bloques
            if (i + 1) % 2 == 0:
                x = MaxPooling2D((2, 2))(x)

        # Capas finales
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Residual_CNN')

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_callbacks(self, patience: int = 10, factor: float = 0.5,
                     min_lr: float = 1e-6) -> list:
        """
        Retorna los callbacks para el entrenamiento

        Args:
            patience: Paciencia para Early Stopping
            factor: Factor de reducción de learning rate
            min_lr: Learning rate mínimo

        Returns:
            Lista de callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=factor,
                patience=patience//2,
                min_lr=min_lr,
                verbose=1
            )
        ]

        return callbacks

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 64,
              data_augmentation: bool = True,
              save_path: Optional[str] = None,
              callbacks: Optional[list] = None) -> Dict[str, Any]:
        """
        Entrena el modelo

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación
            y_val: Etiquetas de validación
            epochs: Número máximo de épocas
            batch_size: Tamaño del batch
            data_augmentation: Si usar aumento de datos
            save_path: Ruta para guardar el modelo

        Returns:
            Historial de entrenamiento
        """
        if self.model is None:
            raise ValueError("Debe construir el modelo primero usando build_model()")

        # Usar callbacks proporcionados o los predeterminados
        if callbacks is None:
            callbacks = self.get_callbacks()

        if data_augmentation:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode='nearest'
            )

            # Ajustar el generador a los datos de entrenamiento
            datagen.fit(X_train)

            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

        self.history = history.history

        # Guardar modelo si se especifica
        if save_path:
            self.save_model(save_path)

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo en datos de prueba

        Args:
            X_test: Datos de prueba
            y_test: Etiquetas de prueba

        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("Debe construir y entrenar el modelo primero")

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        return {
            'test_loss': loss,
            'test_accuracy': accuracy
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo

        Args:
            X: Datos para predecir

        Returns:
            Predicciones del modelo
        """
        if self.model is None:
            raise ValueError("Debe construir y entrenar el modelo primero")

        return self.model.predict(X, verbose=0)

    def save_model(self, filepath: str):
        """
        Guarda el modelo en disco

        Args:
            filepath: Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")

        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath: str):
        """
        Carga un modelo desde disco

        Args:
            filepath: Ruta del modelo a cargar
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Modelo cargado desde: {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Visualiza el historial de entrenamiento

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        if self.history is None:
            raise ValueError("No hay historial de entrenamiento disponible")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Gráfico de accuracy
        axes[0].plot(self.history['accuracy'], label='Entrenamiento', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Validación', linewidth=2)
        axes[0].set_title('Evolución del Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Gráfico de loss
        axes[1].plot(self.history['loss'], label='Entrenamiento', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Validación', linewidth=2)
        axes[1].set_title('Evolución del Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Historial guardado en: {save_path}")

        plt.show()

    def get_model_summary(self) -> str:
        """
        Retorna un resumen del modelo

        Returns:
            String con el resumen del modelo
        """
        if self.model is None:
            return "No hay modelo construido"

        # Capturar el output del summary
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        self.model.summary()

        sys.stdout = old_stdout
        return buffer.getvalue()


def create_model_comparison(models_config: list) -> Dict[str, CIFAR10CNN]:
    """
    Crea múltiples modelos para comparación

    Args:
        models_config: Lista de configuraciones de modelos

    Returns:
        Diccionario con los modelos creados
    """
    models = {}

    for config in models_config:
        name = config.get('name', f"model_{len(models)}")
        model_type = config.get('type', 'basic')
        params = config.get('params', {})

        cnn = CIFAR10CNN()
        cnn.build_model(model_type, **params)
        models[name] = cnn

    return models


class MNISTCNN:
    """
    Clase para crear y gestionar modelos CNN para reconocimiento de dígitos MNIST
    """

    def __init__(self, input_shape: tuple = (28, 28, 1), num_classes: int = 10):
        """
        Inicializa la clase CNN

        Args:
            input_shape: Dimensiones de entrada de las imágenes
            num_classes: Número de clases para clasificación
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self, model_type: str = 'basic', **kwargs) -> Model:
        """
        Construye el modelo CNN según el tipo especificado

        Args:
            model_type: Tipo de modelo ('basic', 'advanced', 'residual')
            **kwargs: Parámetros adicionales para el modelo

        Returns:
            Modelo compilado
        """
        if model_type == 'basic':
            self.model = self._build_basic_cnn(**kwargs)
        elif model_type == 'advanced':
            self.model = self._build_advanced_cnn(**kwargs)
        elif model_type == 'residual':
            self.model = self._build_residual_cnn(**kwargs)
        else:
            raise ValueError(f"Tipo de modelo '{model_type}' no reconocido")

        return self.model

    def _build_basic_cnn(self, filters: list = [32, 64],
                        kernel_size: tuple = (3, 3),
                        dropout_rate: float = 0.25,
                        learning_rate: float = 0.001) -> Model:
        """
        Construye un modelo CNN básico para MNIST

        Args:
            filters: Lista con número de filtros por capa convolucional
            kernel_size: Tamaño del kernel para convoluciones
            dropout_rate: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo compilado
        """
        model = Sequential(name='Basic_MNIST_CNN')

        # Primera capa convolucional
        model.add(Conv2D(filters[0], kernel_size, activation='relu',
                        input_shape=self.input_shape, padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Segunda capa convolucional
        model.add(Conv2D(filters[1], kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Capas fully connected
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _build_advanced_cnn(self, filters: list = [32, 64, 128],
                           kernel_size: tuple = (3, 3),
                           dropout_rate: float = 0.3,
                           learning_rate: float = 0.001) -> Model:
        """
        Construye un modelo CNN avanzado para MNIST con Batch Normalization

        Args:
            filters: Lista con número de filtros por capa convolucional
            kernel_size: Tamaño del kernel para convoluciones
            dropout_rate: Tasa de dropout
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo compilado
        """
        model = Sequential(name='Advanced_MNIST_CNN')

        # Primera capa convolucional con Batch Normalization
        model.add(Conv2D(filters[0], kernel_size, input_shape=self.input_shape,
                        padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Segunda capa convolucional
        model.add(Conv2D(filters[1], kernel_size, padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Tercera capa convolucional
        model.add(Conv2D(filters[2], kernel_size, padding='same', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout_rate))

        # Capas fully connected
        model.add(Flatten())
        model.add(Dense(256, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _build_residual_cnn(self, num_blocks: int = 2,
                           filters: int = 32,
                           learning_rate: float = 0.001) -> Model:
        """
        Construye un modelo CNN con bloques residuales simplificados para MNIST

        Args:
            num_blocks: Número de bloques residuales
            filters: Número base de filtros
            learning_rate: Tasa de aprendizaje

        Returns:
            Modelo compilado
        """
        inputs = Input(shape=self.input_shape)

        # Primera capa convolucional
        x = Conv2D(filters, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Bloques residuales
        for i in range(num_blocks):
            # Guardar entrada del bloque
            shortcut = x

            # Primera convolución del bloque
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)

            # Segunda convolución del bloque
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)

            # Conexión residual
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.Activation('relu')(x)

            # Max pooling cada bloque
            x = MaxPooling2D((2, 2))(x)

        # Capas finales
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='Residual_MNIST_CNN')

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def get_callbacks(self, patience: int = 10, factor: float = 0.5,
                     min_lr: float = 1e-6) -> list:
        """
        Retorna los callbacks para el entrenamiento

        Args:
            patience: Paciencia para Early Stopping
            factor: Factor de reducción de learning rate
            min_lr: Learning rate mínimo

        Returns:
            Lista de callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=factor,
                patience=patience//2,
                min_lr=min_lr,
                verbose=1
            )
        ]

        return callbacks

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 20, batch_size: int = 128,
              data_augmentation: bool = True,
              save_path: Optional[str] = None,
              callbacks: Optional[list] = None) -> Dict[str, Any]:
        """
        Entrena el modelo

        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación
            y_val: Etiquetas de validación
            epochs: Número máximo de épocas
            batch_size: Tamaño del batch
            data_augmentation: Si usar aumento de datos
            save_path: Ruta para guardar el modelo

        Returns:
            Historial de entrenamiento
        """
        if self.model is None:
            raise ValueError("Debe construir el modelo primero usando build_model()")

        # Usar callbacks proporcionados o los predeterminados
        if callbacks is None:
            callbacks = self.get_callbacks()

        if data_augmentation:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                fill_mode='nearest'
            )

            # Ajustar el generador a los datos de entrenamiento
            datagen.fit(X_train)

            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

        self.history = history.history

        # Guardar modelo si se especifica
        if save_path:
            self.save_model(save_path)

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evalúa el modelo en datos de prueba

        Args:
            X_test: Datos de prueba
            y_test: Etiquetas de prueba

        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("Debe construir y entrenar el modelo primero")

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        return {
            'test_loss': loss,
            'test_accuracy': accuracy
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo

        Args:
            X: Datos para predecir

        Returns:
            Predicciones del modelo
        """
        if self.model is None:
            raise ValueError("Debe construir y entrenar el modelo primero")

        return self.model.predict(X, verbose=0)

    def save_model(self, filepath: str):
        """
        Guarda el modelo en disco

        Args:
            filepath: Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")

        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath: str):
        """
        Carga un modelo desde disco

        Args:
            filepath: Ruta del modelo a cargar
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Modelo cargado desde: {filepath}")

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Visualiza el historial de entrenamiento

        Args:
            save_path: Ruta para guardar la imagen (opcional)
        """
        if self.history is None:
            raise ValueError("No hay historial de entrenamiento disponible")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Gráfico de accuracy
        axes[0].plot(self.history['accuracy'], label='Entrenamiento', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Validación', linewidth=2)
        axes[0].set_title('Evolución del Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Gráfico de loss
        axes[1].plot(self.history['loss'], label='Entrenamiento', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Validación', linewidth=2)
        axes[1].set_title('Evolución del Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Historial guardado en: {save_path}")

        plt.show()

    def get_model_summary(self) -> str:
        """
        Retorna un resumen del modelo

        Returns:
            String con el resumen del modelo
        """
        if self.model is None:
            return "No hay modelo construido"

        # Capturar el output del summary
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        self.model.summary()

        sys.stdout = old_stdout
        return buffer.getvalue()


if __name__ == "__main__":
    # Ejemplo de uso
    print("Creando modelo CNN básico...")

    # Crear modelo básico
    cnn = CIFAR10CNN()
    model = cnn.build_model('basic')

    print("Resumen del modelo:")
    print(cnn.get_model_summary())

    print("\nCreando modelo CNN avanzado...")
    cnn_advanced = CIFAR10CNN()
    model_advanced = cnn_advanced.build_model('advanced')

    print("Resumen del modelo avanzado:")
    print(cnn_advanced.get_model_summary())
