import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class EmotionDataLoader:
    def __init__(self, data_path=None):
        self.dataset = 'rafdb'

        if data_path is None:
            script_dir = os.path.dirname(__file__)
            data_path = os.path.join(script_dir, '..', 'data', 'rafdb', 'EmoLabel', 'list_patition_label.txt')
        self.data_path = os.path.abspath(data_path)

        # Emotion mappings (same for both datasets)
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
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        """Cargar conjunto de datos desde archivo"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Archivo de conjunto de datos no encontrado: {self.data_path}")

        if self.dataset == 'fer2013':
            self.data = pd.read_csv(self.data_path)
            print(f"FER2013 dataset cargado: {len(self.data)} muestras")
        elif self.dataset == 'rafdb':
            self.data = self._load_rafdb_data()
            print(f"RAF-DB dataset cargado: {len(self.data)} muestras")

        return self.data

    def _load_expw_data(self):
        """Cargar datos del dataset ExpW"""
        # ExpW usa un archivo label.lst con formato específico
        # Formato: #imagen etiqueta bbox_x bbox_y bbox_width bbox_height landmarks...
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 6:  # imagen, etiqueta, bbox coords
                        image_path = parts[0]
                        emotion_label = int(parts[1])
                        bbox = [int(x) for x in parts[2:6]]  # x, y, width, height

                        data.append({
                            'image_path': image_path,
                            'emotion': emotion_label,
                            'bbox': bbox
                        })

        return pd.DataFrame(data)

    def _load_rafdb_data(self):
        """Cargar datos del dataset RAF-DB"""
        # RAF-DB usa un archivo list_patition_label.txt con formato específico
        # Formato: partition/image_name.jpg emotion_label
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_path = parts[0]  # e.g., "train_00001.jpg"
                        emotion_label = int(parts[1]) - 1  # RAF-DB labels are 1-7, convert to 0-6

                        # Convert emotion labels from RAF-DB format to our format
                        # RAF-DB: 1=Surprise, 2=Fear, 3=Disgust, 4=Happiness, 5=Sadness, 6=Anger, 7=Neutral
                        # Our format: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
                        rafdb_to_our = {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6}
                        our_emotion_label = rafdb_to_our.get(emotion_label, emotion_label)

                        data.append({
                            'image_path': image_path,
                            'emotion': our_emotion_label
                        })

        return pd.DataFrame(data)

    def preprocess_data(self, test_size=0.2, validation_split=0.1, max_samples=None):
        """Preprocesar los datos para entrenamiento"""
        if self.data is None:
            self.load_data()

        if self.dataset == 'fer2013':
            return self._preprocess_fer2013(test_size, validation_split, max_samples)
        elif self.dataset == 'rafdb':
            return self._preprocess_rafdb(test_size, validation_split, max_samples)

    def _preprocess_fer2013(self, test_size=0.2, validation_split=0.1, max_samples=None):
        """Preprocesar datos FER2013"""
        # Extraer píxeles y etiquetas
        pixels = self.data['pixels'].tolist()
        labels = self.data['emotion'].tolist()

        # Limitar muestras si se especifica
        if max_samples and len(pixels) > max_samples:
            indices = np.random.choice(len(pixels), max_samples, replace=False)
            pixels = [pixels[i] for i in indices]
            labels = [labels[i] for i in indices]

        # Convertir píxeles a arrays numpy
        X = []
        for pixel_sequence in pixels:
            # Dividir la cadena y convertir a float, filtrando strings vacías
            pixel_values = [int(pixel) for pixel in pixel_sequence.split(' ') if pixel.strip()]
            # Redimensionar a 48x48
            pixel_array = np.array(pixel_values).reshape(48, 48, 1)
            # Normalizar a [0, 1]
            pixel_array = pixel_array.astype('float32') / 255.0
            X.append(pixel_array)

        X = np.array(X)
        y = np.array(labels)

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

    def _load_from_test_directory(self, test_size=0.2, validation_split=0.1, max_samples=None):
        """Cargar imágenes reales del directorio data/test/"""
        print("Cargando imágenes reales de data/test/...")

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
        processed_count = 0

        # Directorio base de test
        test_base_dir = os.path.join(os.path.dirname(self.data_path), '..', 'test')
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
                    processed_count += 1

                except Exception as e:
                    print(f"Error procesando {image_file}: {e}")
                    continue

        if len(X) == 0:
            print("❌ No se pudieron cargar imágenes reales. Usando datos sintéticos...")
            return self._create_synthetic_dataset(test_size, validation_split)

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
