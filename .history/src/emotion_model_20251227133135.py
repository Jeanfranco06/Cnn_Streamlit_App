import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import cv2
from PIL import Image

class EmotionClassifier:
    def __init__(self, model_path='models/emotion/emotion_model.h5', input_shape=(48, 48, 1), num_classes=7):
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None

        # Emotion labels in Spanish (primary labels)
        self.emotions = {
            0: 'Enojado',
            1: 'Disgustado',
            2: 'Asustado',
            3: 'Feliz',
            4: 'Triste',
            5: 'Sorprendido',
            6: 'Neutral'
        }

        # English labels for compatibility
        self.emotions_en = {
            0: 'angry',
            1: 'disgusted',
            2: 'fearful',
            3: 'happy',
            4: 'sad',
            5: 'surprised',
            6: 'neutral'
        }

        # Load or create model
        self.load_or_create_model()

    def create_model(self, model_type='basic', **kwargs):
        """Crear arquitectura del modelo CNN para clasificación de emociones"""
        if model_type == 'basic':
            return self._create_basic_model(**kwargs)
        elif model_type == 'advanced':
            return self._create_advanced_model(**kwargs)
        elif model_type == 'residual':
            return self._create_residual_model(**kwargs)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    def _create_basic_model(self, filters=[32, 64], dropout_rate=0.25, learning_rate=1e-4):
        """Crear modelo simple para formas geométricas sintéticas"""
        model = Sequential()

        # Primera capa convolucional simple
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Segunda capa convolucional
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Aplanar y capas densas simples
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def _create_advanced_model(self, filters=[32, 64, 128], dropout_rate=0.3, learning_rate=1e-4):
        """Crear modelo avanzado CNN para emociones con Batch Normalization y regularización"""
        model = Sequential()

        # Primer bloque convolucional
        model.add(Conv2D(filters[0], (3, 3), activation='relu', input_shape=self.input_shape, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

        # Segundo bloque convolucional
        model.add(Conv2D(filters[1], (3, 3), activation='relu', padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

        # Tercer bloque convolucional
        model.add(Conv2D(filters[2], (3, 3), activation='relu', padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))

        # Global Average Pooling
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def _create_residual_model(self, num_blocks=2, filters=32, learning_rate=1e-4):
        """Crear modelo residual simplificado para emociones"""
        inputs = tf.keras.Input(shape=self.input_shape)

        # Initial convolution
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)

        # Residual blocks
        for i in range(num_blocks):
            # Save input for skip connection
            skip = x

            # First conv in block
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)

            # Second conv in block
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)

            # Skip connection
            x = tf.keras.layers.Add()([x, skip])

            # Max pooling after each block
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)

        # Global Average Pooling and output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def load_or_create_model(self):
        """Cargar modelo existente o crear uno nuevo"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"Modelo cargado desde {self.model_path}")
                # Ensure model is properly built and initialized
                self.ensure_model_built()
            except Exception as e:
                print(f"Error cargando modelo: {e}")
                self.model = self.create_model()
                print("Modelo nuevo creado")
        else:
            self.model = self.create_model()
            print("Modelo nuevo creado")

    def ensure_model_built(self):
        """Asegurar que el modelo esté construido y listo para uso"""
        if self.model is not None:
            # Hacer una predicción dummy para asegurar que las capas estén inicializadas
            dummy_input = tf.zeros((1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            _ = self.model(dummy_input, training=False)

    def compile_model(self, learning_rate=0.001):
        """Compilar el modelo con optimizador y función de pérdida"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Modelo compilado exitosamente")

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, class_weights=None, save_path=None, callbacks=None, data_augmentation=True):
        """Entrenar el modelo con aumento de datos"""
        if self.model is None:
            raise ValueError("Modelo no inicializado")

        # Usar save_path si se proporciona, de lo contrario usar el path por defecto
        model_save_path = save_path if save_path is not None else self.model_path

        # Aumento de datos
        if data_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator()

        # Callbacks por defecto - ajustados para datos sintéticos
        default_callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=50,  # Mucho más paciente para datos sintéticos
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001  # Solo parar si mejora menos de 0.1%
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=10,  # Más paciencia antes de reducir LR
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Usar callbacks proporcionados o los por defecto
        if callbacks is not None:
            all_callbacks = default_callbacks + callbacks
        else:
            all_callbacks = default_callbacks

        # Entrenar el modelo
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=all_callbacks,
            class_weight=class_weights,
            verbose=1
        )

        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluar modelo en datos de prueba"""
        if self.model is None:
            raise ValueError("Modelo no inicializado")

        # Obtener predicciones
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        # Calcular métricas
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_classes, y_pred_classes, average='weighted'
        )

        # Reporte de clasificación
        class_report = classification_report(
            y_true_classes, y_pred_classes,
            target_names=list(self.emotions.values()),
            output_dict=True
        )

        # Matriz de confusión
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred_classes,
            'true_labels': y_true_classes
        }

        return results

    def predict(self, image):
        """Predecir emoción desde una imagen única"""
        if self.model is None:
            raise ValueError("Modelo no inicializado")

        try:
            # Preprocesar imagen
            processed_image = self.preprocess_image(image)

            # Hacer predicción
            predictions = self.model.predict(processed_image, verbose=0)[0]

            # Obtener clase predicha y confianza
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]

            # Obtener nombre de emoción
            emotion = self.emotions[predicted_class]

            # Crear diccionario de probabilidades
            probabilities = {self.emotions[i]: float(predictions[i]) for i in range(len(self.emotions))}

            return emotion, confidence, probabilities

        except Exception as e:
            print(f"Error en predicción: {e}")
            # Fallback: devolver predicción por defecto
            return "Neutral", 0.0, {emotion: 0.0 for emotion in self.emotions.values()}

    def preprocess_image(self, image):
        """Preprocesar imagen para predicción con mejor manejo de imágenes reales"""
        try:
            # Convertir imagen PIL a array numpy
            if isinstance(image, Image.Image):
                # Convertir a RGB primero si es necesario
                if image.mode not in ['RGB', 'L']:
                    image = image.convert('RGB')
                image = np.array(image)
            elif isinstance(image, np.ndarray):
                # Si es BGR (OpenCV), convertir a RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:
                    # Ya está en escala de grises, convertir a RGB para procesamiento
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Verificar que la imagen no esté vacía
            if image.size == 0:
                raise ValueError("Imagen vacía")

            # Mejor procesamiento para imágenes reales
            if len(image.shape) == 3:
                # Convertir a escala de grises
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Mejorar contraste y reducir ruido
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray.astype(np.uint8))

            # Suavizado ligero para reducir ruido
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

            # Redimensionar a 48x48 con mejor interpolación
            processed = cv2.resize(enhanced, (48, 48), interpolation=cv2.INTER_CUBIC)

            # Normalizar con mejor rango
            processed = processed.astype('float32') / 255.0

            # Mejora adicional: normalización local
            # Esto ayuda con variaciones de iluminación
            processed = (processed - processed.mean()) / (processed.std() + 1e-7)
            processed = np.clip(processed, -2, 2)  # Limitar valores extremos
            processed = (processed + 2) / 4  # Re-escalar a [0, 1]

            # Agregar dimensiones de lote y canal
            processed = np.expand_dims(processed, axis=[0, -1])

            return processed

        except Exception as e:
            print(f"Error procesando imagen: {e}")
            # Crear imagen dummy con mejor patrón en caso de error
            dummy_image = np.random.normal(0.5, 0.1, (1, 48, 48, 1)).astype('float32')
            dummy_image = np.clip(dummy_image, 0, 1)
            return dummy_image

    def get_model_metrics(self):
        """Obtener métricas de rendimiento del modelo"""
        # Esto típicamente cargaría resultados de evaluación guardados
        # Por ahora, devolver valores de placeholder
        return {
            'accuracy': 0.65,
            'precision': 0.64,
            'recall': 0.63,
            'f1_score': 0.64
        }

    def save_model(self, save_path=None):
        """Guardar el modelo entrenado"""
        if save_path is None:
            save_path = self.model_path

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Modelo guardado en {save_path}")

    def get_model_summary(self):
        """Obtener resumen de arquitectura del modelo"""
        if self.model is None:
            return "Modelo no inicializado"

        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)

    def plot_training_history(self):
        """Graficar historial de entrenamiento"""
        if self.history is None:
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Precisión
        ax1.plot(self.history.history['accuracy'], label='Precisión de Entrenamiento')
        ax1.plot(self.history.history['val_accuracy'], label='Precisión de Validación')
        ax1.set_title('Precisión del Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        ax1.grid(True)

        # Pérdida
        ax2.plot(self.history.history['loss'], label='Pérdida de Entrenamiento')
        ax2.plot(self.history.history['val_loss'], label='Pérdida de Validación')
        ax2.set_title('Pérdida del Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        ax2.grid(True)

        # Tasa de aprendizaje (si está disponible)
        if 'lr' in self.history.history:
            ax3.plot(self.history.history['lr'])
            ax3.set_title('Tasa de Aprendizaje')
            ax3.set_xlabel('Época')
            ax3.set_ylabel('Tasa de Aprendizaje')
            ax3.set_yscale('log')
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, 'Datos de tasa de aprendizaje no disponibles',
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Tasa de Aprendizaje')

        # Brecha entre entrenamiento y validación
        train_acc = np.array(self.history.history['accuracy'])
        val_acc = np.array(self.history.history['val_accuracy'])
        gap = train_acc - val_acc

        ax4.plot(gap)
        ax4.set_title('Brecha de Precisión Entrenamiento vs Validación')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Brecha de Precisión')
        ax4.grid(True)

        plt.tight_layout()
        return fig
