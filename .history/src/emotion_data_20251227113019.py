import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class EmotionDataLoader:
    def __init__(self, data_path=None):
        if data_path is None:
            # Path relative to this script's location
            script_dir = os.path.dirname(__file__)
            data_path = os.path.join(script_dir, '..', 'data', 'fer2013.csv')
        self.data_path = os.path.abspath(data_path)
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
        """Cargar conjunto de datos FER2013 desde archivo CSV"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Archivo de conjunto de datos no encontrado: {self.data_path}")

        self.data = pd.read_csv(self.data_path)
        print(f"Conjunto de datos cargado: {len(self.data)} muestras")

        return self.data

    def preprocess_data(self, test_size=0.2, validation_split=0.1):
        """Preprocesar los datos para entrenamiento"""
        if self.data is None:
            self.load_data()

        # Extraer píxeles y etiquetas
        pixels = self.data['pixels'].tolist()
        labels = self.data['emotion'].tolist()

        # Convertir píxeles a arrays numpy
        X = []
        for pixel_sequence in pixels:
            # Dividir la cadena y convertir a float
            pixel_values = [int(pixel) for pixel in pixel_sequence.split(' ')]
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

    def get_dataset_info(self):
        """Obtener información básica sobre el conjunto de datos"""
        if self.data is None:
            self.load_data()

        # Contar muestras por emoción
        emotion_counts = self.data['emotion'].value_counts().sort_index()

        info = {
            'total_samples': len(self.data),
            'train_samples': len(self.X_train) if self.X_train is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0,
            'validation_samples': len(self.X_val) if hasattr(self, 'X_val') and self.X_val is not None else 0,
            'num_classes': len(self.emotions),
            'image_shape': (48, 48, 1),
            'emotion_distribution': {
                self.emotions[i]: count for i, count in emotion_counts.items()
            }
        }

        return info

    def augment_data(self, X, y, augmentation_factor=2):
        """Aplicar aumento de datos para incrementar el tamaño del conjunto de datos"""
        augmented_X = []
        augmented_y = []

        # Datos originales
        augmented_X.extend(X)
        augmented_y.extend(y)

        # Aumento de datos
        for i in range(len(X)):
            img = X[i].reshape(48, 48)

            # Volteo horizontal
            flipped = cv2.flip(img, 1)
            augmented_X.append(flipped.reshape(48, 48, 1))
            augmented_y.append(y[i])

            # Rotación aleatoria
            angle = np.random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((24, 24), angle, 1)
            rotated = cv2.warpAffine(img, M, (48, 48))
            augmented_X.append(rotated.reshape(48, 48, 1))
            augmented_y.append(y[i])

            # Brillo/contraste aleatorio
            alpha = np.random.uniform(0.8, 1.2)  # contraste
            beta = np.random.randint(-20, 20)    # brillo
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            augmented_X.append(adjusted.reshape(48, 48, 1))
            augmented_y.append(y[i])

        return np.array(augmented_X), np.array(augmented_y)

    def visualize_samples(self, num_samples=5):
        """Visualizar muestras aleatorias del conjunto de datos"""
        if self.data is None:
            self.load_data()

        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

        for i in range(num_samples):
            # Muestra aleatoria
            idx = np.random.randint(len(self.data))
            pixels = self.data.iloc[idx]['pixels']
            emotion = self.data.iloc[idx]['emotion']

            # Convertir píxeles a imagen
            pixel_values = [int(pixel) for pixel in pixels.split(' ')]
            img = np.array(pixel_values).reshape(48, 48)

            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"{self.emotions[emotion]}")
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def get_class_weights(self):
        """Calcular pesos de clase para conjunto de datos desbalanceado"""
        if self.data is None:
            self.load_data()

        from sklearn.utils.class_weight import compute_class_weight
        labels = self.data['emotion'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)

        return dict(enumerate(class_weights))

    def save_processed_data(self, save_path='data/processed/'):
        """Guardar datos procesados para carga más rápida"""
        os.makedirs(save_path, exist_ok=True)

        if self.X_train is not None:
            np.save(os.path.join(save_path, 'X_train.npy'), self.X_train)
            np.save(os.path.join(save_path, 'y_train.npy'), self.y_train)
            np.save(os.path.join(save_path, 'X_val.npy'), self.X_val)
            np.save(os.path.join(save_path, 'y_val.npy'), self.y_val)
            np.save(os.path.join(save_path, 'X_test.npy'), self.X_test)
            np.save(os.path.join(save_path, 'y_test.npy'), self.y_test)

    def load_processed_data(self, load_path='data/processed/'):
        """Cargar datos preprocesados"""
        try:
            self.X_train = np.load(os.path.join(load_path, 'X_train.npy'))
            self.y_train = np.load(os.path.join(load_path, 'y_train.npy'))
            self.X_val = np.load(os.path.join(load_path, 'X_val.npy'))
            self.y_val = np.load(os.path.join(load_path, 'y_val.npy'))
            self.X_test = np.load(os.path.join(load_path, 'X_test.npy'))
            self.y_test = np.load(os.path.join(load_path, 'y_test.npy'))

            return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
        except FileNotFoundError:
            print("Datos procesados no encontrados. Por favor preprocese los datos primero.")
            return None
