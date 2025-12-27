import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class EmotionDataLoader:
    def __init__(self):
        # Emotion mappings
        self.emotions = {
            0: 'angry',
            1: 'disgusted',
            2: 'fearful',
            3: 'happy',
            4: 'sad',
            5: 'surprised',
            6: 'neutral'
        }
        # Etiquetas en español para la interfaz
        self.emotion_labels_es = {
            0: 'Enojado',
            1: 'Disgustado',
            2: 'Asustado',
            3: 'Feliz',
            4: 'Triste',
            5: 'Sorprendido',
            6: 'Neutral'
        }
        self.emotion_labels = list(self.emotion_labels_es.values())
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def preprocess_data(self, test_size=0.2, validation_split=0.1, max_samples=None):
        """Cargar y preprocesar imágenes reales del directorio data/test/"""
        print("Cargando imágenes reales de emociones de data/test/...")

        # Mapeo de nombres de directorios a índices de emoción
        emotion_map = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }

        X = []
        labels = []

        # Directorio base de test
        script_dir = os.path.dirname(__file__)
        test_base_dir = os.path.join(script_dir, '..', 'data', 'test')
        test_base_dir = os.path.abspath(test_base_dir)

        print(f"Directorio de imágenes: {test_base_dir}")

        # Procesar cada directorio de emoción
        for emotion_name, emotion_idx in emotion_map.items():
            emotion_dir = os.path.join(test_base_dir, emotion_name)
            if not os.path.exists(emotion_dir):
                print(f"Directorio no encontrado: {emotion_dir}")
                continue

            print(f"Procesando emoción {emotion_name} (índice {emotion_idx})...")

            # Obtener lista de archivos de imagen
            image_files = [f for f in os.listdir(emotion_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if len(image_files) == 0:
                print(f"No se encontraron imágenes en {emotion_dir}")
                continue

            # Limitar imágenes por emoción si se especifica max_samples
            if max_samples:
                max_per_emotion = max_samples // len(emotion_map)
                if len(image_files) > max_per_emotion:
                    image_files = image_files[:max_per_emotion]

            # Procesar cada imagen
            for image_file in image_files:
                try:
                    img_path = os.path.join(emotion_dir, image_file)
                    image = cv2.imread(img_path)

                    if image is None:
                        continue

                    # Convertir a escala de grises
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Redimensionar a 48x48
                    processed = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_CUBIC)

                    # Normalizar
                    processed = processed.astype('float32') / 255.0
                    processed = np.expand_dims(processed, axis=-1)

                    X.append(processed)
                    labels.append(emotion_idx)

                except Exception as e:
                    print(f"Error procesando {image_file}: {e}")
                    continue

        if len(X) == 0:
            raise ValueError("❌ No se pudieron cargar imágenes reales del directorio data/test/")

        X = np.array(X)
        y = np.array(labels)

        print(f"✅ Imágenes reales cargadas: {len(X)} muestras")
        print(f"Distribución por emoción: {np.bincount(y)}")

        # Convertir etiquetas a categóricas
        y = to_categorical(y, num_classes=len(self.emotions))

        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=labels
        )

        # Dividir aún más el entrenamiento en entrenamiento y validación
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=validation_split, random_state=42
        )

        print(f"Muestras de entrenamiento: {len(self.X_train)}")
        print(f"Muestras de validación: {len(self.X_val)}")
        print(f"Muestras de prueba: {len(self.X_test)}")

        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'class_names': self.emotion_labels
        }

    def get_dataset_info(self):
        """Obtener información básica sobre el conjunto de datos"""
        return {
            'total_samples': len(self.X_train) + len(self.X_val) + len(self.X_test) if self.X_train is not None else 0,
            'train_samples': len(self.X_train) if self.X_train is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'validation_samples': len(self.X_val) if hasattr(self, 'X_val') and self.X_val is not None else 0,
            'num_classes': len(self.emotions),
            'image_shape': (48, 48, 1),
        }

    def get_class_weights(self):
        """Calcular pesos de clase para conjunto de datos desbalanceado"""
        if self.X_train is None:
            raise ValueError("Los datos no han sido cargados. Llama a preprocess_data() primero.")

        # Convertir etiquetas one-hot encoded a índices de clase
        y_train_indices = np.argmax(self.y_train, axis=1)

        # Calcular distribución de clases
        unique_classes, class_counts = np.unique(y_train_indices, return_counts=True)

        # Calcular pesos de clase (inversamente proporcional a la frecuencia)
        total_samples = len(y_train_indices)
        class_weights = {}

        for class_idx, count in zip(unique_classes, class_counts):
            # Peso = total_samples / (num_classes * count_class)
            weight = total_samples / (len(unique_classes) * count)
            class_weights[int(class_idx)] = weight

        return class_weights
