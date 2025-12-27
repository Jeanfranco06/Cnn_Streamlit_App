import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class EmotionDataLoader:
    def __init__(self, data_path=None):
        self.dataset = 'expw'

        if data_path is None:
            script_dir = os.path.dirname(__file__)
            data_path = os.path.join(script_dir, '..', 'data', 'expw', 'label.lst')
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
            print("\n💡 Actualmente usando datos de prueba. Para mejores resultados, descarga el dataset completo.")

            # Create a minimal test dataset with dummy images for demonstration
            print("📝 Creando dataset mínimo de prueba...")

            # Create a highly realistic synthetic face dataset
            emotions = list(self.emotions.keys())
            X = []
            labels = []

            samples_per_emotion = 100  # Even more samples for better generalization

            for emotion_idx in emotions:
                for i in range(samples_per_emotion):
                    # Create base face structure - much more realistic
                    img = self._generate_realistic_face_base()

                    # Add emotion-specific modifications
                    img = self._add_emotion_features(img, emotion_idx)

                    # Add realistic facial texture and variations
                    img = self._add_facial_texture(img)

                    # Random lighting variations
                    img = self._apply_lighting_variations(img)

                    # Normalize to [0, 1] and expand dimensions
                    img_normalized = img.astype('float32') / 255.0
                    img_normalized = np.expand_dims(img_normalized, axis=-1)

                    X.append(img_normalized)
                    labels.append(emotion_idx)

    def _generate_realistic_face_base(self):
        """Generate a realistic base face structure"""
        img = np.full((48, 48), 120, dtype=np.uint8)  # Base skin tone

        # Add oval face shape
        y_center, x_center = 24, 24
        for y in range(48):
            for x in range(48):
                # Distance from center
                dist = np.sqrt((y - y_center)**2 + (x - x_center)**2)
                # Face boundary (oval shape)
                if dist > 20:
                    # Fade to background
                    fade_factor = max(0, 1 - (dist - 20) / 5)
                    img[y, x] = int(120 * fade_factor + 80 * (1 - fade_factor))
                elif dist < 18:
                    # Face interior with slight variations
                    img[y, x] = np.random.randint(110, 140)

        # Add forehead
        img[2:12, 12:36] = np.random.randint(130, 150, (10, 24))

        # Add cheeks
        img[20:35, 5:15] = np.random.randint(125, 145, (15, 10))   # Left cheek
        img[20:35, 33:43] = np.random.randint(125, 145, (15, 10))  # Right cheek

        # Add nose bridge
        for x in range(20, 28):
            img[15:25, x] = np.random.randint(100, 120, (10,))

        return img

    def _add_emotion_features(self, img, emotion_idx):
        """Add emotion-specific facial features"""

        if emotion_idx == 0:  # Angry - furrowed brows, tense mouth
            # Furrowed brows (lowered and closer)
            img[6:10, 8:18] = np.random.randint(80, 100, (4, 10))   # Left brow
            img[6:10, 30:40] = np.random.randint(80, 100, (4, 10))  # Right brow
            # Vertical frown lines between brows
            img[8:18, 22:26] = np.random.randint(70, 90, (10, 4))
            # Tense mouth - pressed lips
            img[32:36, 18:30] = np.random.randint(90, 110, (4, 12))
            # Jaw tension lines
            img[35:40, 15:33] = np.random.randint(85, 105, (5, 18))

        elif emotion_idx == 1:  # Disgust - wrinkled nose, raised upper lip
            # Wrinkled nose area
            for y in range(20, 30, 2):
                for x in range(18, 30, 2):
                    if np.random.random() > 0.3:
                        img[y + np.random.randint(-1, 2), x + np.random.randint(-1, 2)] = np.random.randint(95, 115)
            # Raised upper lip
            img[28:32, 20:28] = np.random.randint(85, 105, (4, 8))
            # Narrowed eyes
            img[10:16, 10:16] = np.random.randint(100, 120, (6, 6))  # Left eye narrower
            img[10:16, 32:38] = np.random.randint(100, 120, (6, 6))  # Right eye narrower

        elif emotion_idx == 2:  # Fear - wide eyes, raised brows, open mouth
            # Wide open eyes
            img[8:18, 8:18] = np.random.randint(180, 220, (10, 10))   # Left eye wide
            img[8:18, 30:40] = np.random.randint(180, 220, (10, 10))  # Right eye wide
            # Very raised eyebrows
            for x in range(5, 43):
                height_variation = np.random.randint(2, 6)
                img[3 + np.random.randint(0, height_variation), x] = np.random.randint(140, 170)
            # Open mouth with tension
            img[30:40, 16:32] = np.random.randint(60, 90, (10, 16))
            # Wide mouth corners
            img[32:36, 14:18] = np.random.randint(70, 100, (4, 4))  # Left corner
            img[32:36, 30:34] = np.random.randint(70, 100, (4, 4))  # Right corner

        elif emotion_idx == 3:  # Happy - smiling eyes, upturned mouth, raised cheeks
            # Smiling eyes - slight crinkle at corners
            img[12:15, 6:10] = np.random.randint(130, 150, (3, 4))   # Left eye crinkle
            img[12:15, 38:42] = np.random.randint(130, 150, (3, 4))  # Right eye crinkle
            # Big smile curve
            for x in range(15, 33):
                y_offset = int(6 * np.sin((x - 15) * np.pi / 18)) + np.random.randint(-1, 2)
                img[28 + y_offset, x] = np.random.randint(170, 210)
                # Add smile thickness
                if np.random.random() > 0.4:
                    img[29 + y_offset, x] = np.random.randint(150, 190)
            # Raised cheeks
            img[25:32, 8:15] = np.random.randint(145, 165, (7, 7))   # Left cheek
            img[25:32, 33:40] = np.random.randint(145, 165, (7, 7))  # Right cheek

        elif emotion_idx == 4:  # Sad - downturned mouth, drooping eyes, lowered brows
            # Downturned mouth
            for x in range(15, 33):
                y_offset = int(-4 * np.sin((x - 15) * np.pi / 18)) + np.random.randint(-1, 2)
                img[34 + y_offset, x] = np.random.randint(120, 150)
            # Drooping eyes
            img[14:20, 10:16] = np.random.randint(100, 125, (6, 6))  # Left eye droop
            img[14:20, 32:38] = np.random.randint(100, 125, (6, 6))  # Right eye droop
            # Lowered brows
            img[10:14, 8:18] = np.random.randint(90, 110, (4, 10))   # Left brow
            img[10:14, 30:40] = np.random.randint(90, 110, (4, 10))  # Right brow
            # Sad mouth corners
            img[30:34, 12:16] = np.random.randint(100, 120, (4, 4))  # Left corner down
            img[30:34, 32:36] = np.random.randint(100, 120, (4, 4))  # Right corner down

        elif emotion_idx == 5:  # Surprise - wide open eyes, round mouth, raised brows
            # Maximum wide eyes
            img[6:20, 6:18] = np.random.randint(200, 240, (14, 12))   # Left eye very wide
            img[6:20, 30:42] = np.random.randint(200, 240, (14, 12))  # Right eye very wide
            # Round open mouth
            center_y, center_x = 36, 24
            for y in range(28, 44):
                for x in range(16, 32):
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist <= 8:
                        img[y, x] = np.random.randint(40, 80)
            # Maximally raised brows
            for x in range(3, 45):
                height_variation = np.random.randint(4, 8)
                img[2 + np.random.randint(0, height_variation), x] = np.random.randint(150, 180)

        elif emotion_idx == 6:  # Neutral - balanced, relaxed features
            # Balanced eyes
            img[10:16, 10:16] = np.random.randint(150, 175, (6, 6))  # Left eye
            img[10:16, 32:38] = np.random.randint(150, 175, (6, 6))  # Right eye
            # Neutral mouth - slight natural curve
            for x in range(18, 30):
                y_offset = int(2 * np.sin((x - 18) * np.pi / 12))
                img[32 + y_offset, x] = np.random.randint(130, 155)
            # Balanced brows
            img[6:10, 8:18] = np.random.randint(120, 140, (4, 10))   # Left brow
            img[6:10, 30:40] = np.random.randint(120, 140, (4, 10))  # Right brow

        return img

    def _add_facial_texture(self, img):
        """Add realistic facial texture and skin variations"""
        # Add skin texture variations
        for _ in range(np.random.randint(30, 60)):
            y, x = np.random.randint(5, 43, 2)
            # Create small skin texture variations
            texture_size = np.random.randint(1, 4)
            for dy in range(-texture_size, texture_size + 1):
                for dx in range(-texture_size, texture_size + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 48 and 0 <= nx < 48:
                        variation = np.random.randint(-15, 16)
                        img[ny, nx] = np.clip(img[ny, nx] + variation, 50, 200)

        # Add subtle facial hair shadows for some samples (20% chance)
        if np.random.random() > 0.8:
            # Light shadow along jawline
            for x in range(10, 38):
                if np.random.random() > 0.6:
                    img[38 + np.random.randint(-2, 3), x] = np.random.randint(90, 110)

        return img

    def _apply_lighting_variations(self, img):
        """Apply realistic lighting variations"""
        # Random overall brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)

        # Add directional lighting (left or right side brighter)
        if np.random.random() > 0.5:
            # Left side lighting
            for x in range(24):
                lighting_factor = 0.9 + (x / 24) * 0.2
                img[:, x] = np.clip(img[:, x] * lighting_factor, 0, 255)
        else:
            # Right side lighting
            for x in range(24, 48):
                lighting_factor = 0.9 + ((47 - x) / 24) * 0.2
                img[:, x] = np.clip(img[:, x] * lighting_factor, 0, 255)

        # Add subtle shadows under features
        # Under eyes
        img[16:20, 8:20] = np.clip(img[16:20, 8:20] * 0.9, 0, 255)   # Left
        img[16:20, 28:40] = np.clip(img[16:20, 28:40] * 0.9, 0, 255) # Right

        # Under nose
        img[25:30, 20:28] = np.clip(img[25:30, 20:28] * 0.85, 0, 255)

        return img.astype(np.uint8)

            X = np.array(X)
            y = np.array(labels)

            # Convert to categorical
            y = to_categorical(y, num_classes=len(self.emotions))

            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=labels
            )

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X_train, self.y_train, test_size=validation_split, random_state=42
            )

            print(f"✅ Dataset sintético creado: {len(X)} muestras")
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
