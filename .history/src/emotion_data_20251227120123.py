import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class EmotionDataLoader:
    def __init__(self, dataset='fer2013', data_path=None):
        self.dataset = dataset

        if data_path is None:
            script_dir = os.path.dirname(__file__)
            if dataset == 'fer2013':
                data_path = os.path.join(script_dir, '..', 'data', 'fer2013.csv')
            elif dataset == 'expw':
                data_path = os.path.join(script_dir, '..', 'data', 'expw', 'label.lst')
            else:
                raise ValueError(f"Dataset '{dataset}' not supported. Use 'fer2013' or 'expw'")
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
        elif self.dataset == 'expw':
            self.data = self._load_expw_data()
            print(f"ExpW dataset cargado: {len(self.data)} muestras")

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

    def preprocess_data(self, test_size=0.2, validation_split=0.1, max_samples=None):
        """Preprocesar los datos para entrenamiento"""
        if self.data is None:
            self.load_data()

        if self.dataset == 'fer2013':
            return self._preprocess_fer2013(test_size, validation_split, max_samples)
        elif self.dataset == 'expw':
            return self._preprocess_expw(test_size, validation_split, max_samples)

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

    def _preprocess_expw(self, test_size=0.2, validation_split=0.1, max_samples=None):
        """Preprocesar datos ExpW"""
        print("Procesando imágenes ExpW...")

        # Directorio base de imágenes ExpW
        expw_base_dir = os.path.dirname(self.data_path)

        X = []
        labels = []
        processed_count = 0
        missing_images = 0

        # Procesar cada imagen
        for idx, row in self.data.iterrows():
            if max_samples and idx >= max_samples:
                break

            try:
                # Cargar imagen
                img_path = os.path.join(expw_base_dir, row['image_path'])
                if not os.path.exists(img_path):
                    missing_images += 1
                    if missing_images <= 3:  # Solo mostrar primeros errores
                        print(f"Imagen no encontrada: {img_path}")
                    elif missing_images == 4:
                        print("... (y más imágenes faltantes)")
                    continue

                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Convertir a escala de grises
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Extraer cara usando bounding box
                bbox = row['bbox']
                x, y, w, h = bbox

                # Asegurar que bbox esté dentro de la imagen
                x = max(0, x)
                y = max(0, y)
                w = min(w, gray.shape[1] - x)
                h = min(h, gray.shape[0] - y)

                if w <= 0 or h <= 0:
                    continue

                face = gray[y:y+h, x:x+w]

                # Redimensionar a 48x48
                face_resized = cv2.resize(face, (48, 48))

                # Normalizar
                face_normalized = face_resized.astype('float32') / 255.0
                face_normalized = np.expand_dims(face_normalized, axis=-1)

                X.append(face_normalized)
                labels.append(row['emotion'])
                processed_count += 1

            except Exception as e:
                print(f"Error procesando imagen {row['image_path']}: {e}")
                continue

        if len(X) == 0:
            print("❌ No se encontraron imágenes válidas en el dataset ExpW.")
            print("💡 Para usar ExpW, descarga el dataset completo desde:")
            print("   https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/")
            print("   Extrae los archivos a data/expw/")
            print("   Asegúrate de que la estructura sea:")
            print("   data/expw/")
            print("   ├── WIDER_train/images/")
            print("   ├── WIDER_val/images/")
            print("   └── label.lst")
            print("\n🔄 Usando FER2013 como alternativa...")
            # Fallback to FER2013
            return self._preprocess_fer2013(test_size, validation_split, max_samples)

        if missing_images > 0:
            print(f"⚠️  {missing_images} imágenes no encontradas, {processed_count} procesadas correctamente")

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

        print(f"ExpW - Muestras procesadas: {len(X)}")
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
