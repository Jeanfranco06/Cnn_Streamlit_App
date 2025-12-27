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

            # Create a simple but safe synthetic face dataset
            emotions = list(self.emotions.keys())
            X = []
            labels = []

            samples_per_emotion = 200  # More samples for better learning

            for emotion_idx in emotions:
                for i in range(samples_per_emotion):
                    # Create highly distinct patterns for each emotion
                    img = np.full((48, 48), 128, dtype=np.uint8)  # Neutral gray base

                    if emotion_idx == 0:  # Angry - Triangle pointing up (anger)
                        # Draw upward triangle
                        for y in range(24):
                            start_x = 24 - y
                            end_x = 24 + y
                            if start_x < 0: start_x = 0
                            if end_x > 47: end_x = 47
                            img[y, start_x:end_x+1] = 30

                    elif emotion_idx == 1:  # Disgust - Horizontal zigzag
                        # Create horizontal zigzag pattern
                        for y in range(0, 48, 8):
                            for x in range(48):
                                if (x // 4) % 2 == (y // 8) % 2:
                                    img[y:y+4, x] = 45

                    elif emotion_idx == 2:  # Fear - Diamond pattern
                        # Draw diamond shape
                        center_y, center_x = 24, 24
                        for y in range(48):
                            for x in range(48):
                                dist_from_center = abs(y - center_y) + abs(x - center_x)
                                if dist_from_center <= 16 and dist_from_center % 4 == 0:
                                    img[y, x] = 35

                    elif emotion_idx == 3:  # Happy - Vertical waves
                        # Create vertical wave pattern
                        for x in range(0, 48, 6):
                            for y in range(48):
                                wave_offset = int(3 * np.sin(y * 0.15))
                                wave_x = x + wave_offset
                                if 0 <= wave_x < 48:
                                    img[y, wave_x] = 220

                    elif emotion_idx == 4:  # Sad - Downward arrows
                        # Draw downward pointing arrows
                        for arrow_x in range(8, 41, 8):
                            # Arrow shaft
                            img[10:35, arrow_x-1:arrow_x+2] = 25
                            # Arrow head (pointing down)
                            for offset in range(6):
                                y_pos = 35 + offset
                                x_start = arrow_x - (5 - offset)
                                x_end = arrow_x + (6 - offset)
                                if 0 <= y_pos < 48 and x_start >= 0 and x_end <= 47:
                                    img[y_pos, x_start:x_end] = 25

                    elif emotion_idx == 5:  # Surprise - Five-pointed star
                        # Draw star pattern
                        center_y, center_x = 24, 24
                        for angle in range(0, 360, 72):  # 5 points
                            rad_angle = np.radians(angle)
                            # Outer point
                            outer_y = int(center_y + 15 * np.sin(rad_angle))
                            outer_x = int(center_x + 15 * np.cos(rad_angle))
                            if 0 <= outer_y < 48 and 0 <= outer_x < 48:
                                img[outer_y, outer_x] = 240
                            # Inner point
                            inner_y = int(center_y + 6 * np.sin(rad_angle + np.pi/5))
                            inner_x = int(center_x + 6 * np.cos(rad_angle + np.pi/5))
                            if 0 <= inner_y < 48 and 0 <= inner_x < 48:
                                img[inner_y, inner_x] = 240

                    elif emotion_idx == 6:  # Neutral - Checkerboard
                        # Create checkerboard pattern
                        for y in range(0, 48, 6):
                            for x in range(0, 48, 6):
                                if (x//6 + y//6) % 2 == 0:
                                    img[y:y+6, x:x+6] = 160

                    # Ensure final image is valid uint8
                    img = np.clip(img, 0, 255).astype(np.uint8)

                    # Normalize to [0, 1] and expand dimensions
                    img_normalized = img.astype('float32') / 255.0
                    img_normalized = np.expand_dims(img_normalized, axis=-1)

                    X.append(img_normalized)
                    labels.append(emotion_idx)

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

    def _generate_realistic_face_base(self):
        """Generate a realistic base face structure"""
        # Create base image with safe uint8 values
        img = np.full((48, 48), 120, dtype=np.uint8)

        # Add oval face shape with safe calculations
        y_center, x_center = 24, 24
        for y in range(48):
            for x in range(48):
                # Distance from center
                dist = np.sqrt((y - y_center)**2 + (x - x_center)**2)
                # Face boundary (oval shape)
                if dist > 20:
                    # Fade to background - ensure non-negative result
                    fade_factor = max(0.0, min(1.0, 1 - (dist - 20) / 5))
                    new_value = int(max(0, 120 * fade_factor + 80 * (1 - fade_factor)))
                    img[y, x] = np.uint8(min(255, max(0, new_value)))
                elif dist < 18:
                    # Face interior with slight variations
                    random_val = np.random.randint(110, 140)
                    img[y, x] = np.uint8(min(255, max(0, random_val)))

        # Add forehead with safe random values
        forehead_vals = np.random.randint(130, 150, (10, 24))
        img[2:12, 12:36] = np.uint8(np.clip(forehead_vals, 0, 255))

        # Add cheeks with safe random values
        left_cheek_vals = np.random.randint(125, 145, (15, 10))
        img[20:35, 5:15] = np.uint8(np.clip(left_cheek_vals, 0, 255))

        right_cheek_vals = np.random.randint(125, 145, (15, 10))
        img[20:35, 33:43] = np.uint8(np.clip(right_cheek_vals, 0, 255))

        # Add nose bridge with safe random values
        for x in range(20, 28):
            nose_vals = np.random.randint(100, 120, (10,))
            img[15:25, x] = np.uint8(np.clip(nose_vals, 0, 255))

        # Final safety check
        return np.uint8(np.clip(img, 0, 255))

    def _add_emotion_features(self, img, emotion_idx):
        """Add emotion-specific facial features"""

        if emotion_idx == 0:  # Angry - furrowed brows, tense mouth
            # Furrowed brows (lowered and closer)
            img[6:10, 8:18] = np.clip(np.random.randint(80, 100, (4, 10)), 0, 255)   # Left brow
            img[6:10, 30:40] = np.clip(np.random.randint(80, 100, (4, 10)), 0, 255)  # Right brow
            # Vertical frown lines between brows
            img[8:18, 22:26] = np.clip(np.random.randint(70, 90, (10, 4)), 0, 255)
            # Tense mouth - pressed lips
            img[32:36, 18:30] = np.clip(np.random.randint(90, 110, (4, 12)), 0, 255)
            # Jaw tension lines
            img[35:40, 15:33] = np.clip(np.random.randint(85, 105, (5, 18)), 0, 255)

        elif emotion_idx == 1:  # Disgust - wrinkled nose, raised upper lip
            # Wrinkled nose area
            for y in range(20, 30, 2):
                for x in range(18, 30, 2):
                    if np.random.random() > 0.3:
                        ny, nx = y + np.random.randint(-1, 2), x + np.random.randint(-1, 2)
                        if 0 <= ny < 48 and 0 <= nx < 48:
                            img[ny, nx] = np.clip(np.random.randint(95, 115), 0, 255)
            # Raised upper lip
            img[28:32, 20:28] = np.clip(np.random.randint(85, 105, (4, 8)), 0, 255)
            # Narrowed eyes
            img[10:16, 10:16] = np.clip(np.random.randint(100, 120, (6, 6)), 0, 255)  # Left eye narrower
            img[10:16, 32:38] = np.clip(np.random.randint(100, 120, (6, 6)), 0, 255)  # Right eye narrower

        elif emotion_idx == 2:  # Fear - wide eyes, raised brows, open mouth
            # Wide open eyes
            img[8:18, 8:18] = np.clip(np.random.randint(180, 220, (10, 10)), 0, 255)   # Left eye wide
            img[8:18, 30:40] = np.clip(np.random.randint(180, 220, (10, 10)), 0, 255)  # Right eye wide
            # Very raised eyebrows
            for x in range(5, 43):
                height_variation = np.random.randint(2, 6)
                ny = 3 + np.random.randint(0, height_variation)
                if 0 <= ny < 48:
                    img[ny, x] = np.clip(np.random.randint(140, 170), 0, 255)
            # Open mouth with tension
            img[30:40, 16:32] = np.clip(np.random.randint(60, 90, (10, 16)), 0, 255)
            # Wide mouth corners
            img[32:36, 14:18] = np.clip(np.random.randint(70, 100, (4, 4)), 0, 255)  # Left corner
            img[32:36, 30:34] = np.clip(np.random.randint(70, 100, (4, 4)), 0, 255)  # Right corner

        elif emotion_idx == 3:  # Happy - smiling eyes, upturned mouth, raised cheeks
            # Smiling eyes - slight crinkle at corners
            img[12:15, 6:10] = np.clip(np.random.randint(130, 150, (3, 4)), 0, 255)   # Left eye crinkle
            img[12:15, 38:42] = np.clip(np.random.randint(130, 150, (3, 4)), 0, 255)  # Right eye crinkle
            # Big smile curve
            for x in range(15, 33):
                y_offset = int(6 * np.sin((x - 15) * np.pi / 18)) + np.random.randint(-1, 2)
                ny = 28 + y_offset
                if 0 <= ny < 48:
                    img[ny, x] = np.clip(np.random.randint(170, 210), 0, 255)
                    # Add smile thickness
                    if np.random.random() > 0.4:
                        ny2 = 29 + y_offset
                        if 0 <= ny2 < 48:
                            img[ny2, x] = np.clip(np.random.randint(150, 190), 0, 255)
            # Raised cheeks
            img[25:32, 8:15] = np.clip(np.random.randint(145, 165, (7, 7)), 0, 255)   # Left cheek
            img[25:32, 33:40] = np.clip(np.random.randint(145, 165, (7, 7)), 0, 255)  # Right cheek

        elif emotion_idx == 4:  # Sad - downturned mouth, drooping eyes, lowered brows
            # Downturned mouth
            for x in range(15, 33):
                y_offset = int(-4 * np.sin((x - 15) * np.pi / 18)) + np.random.randint(-1, 2)
                ny = 34 + y_offset
                if 0 <= ny < 48:
                    img[ny, x] = np.clip(np.random.randint(120, 150), 0, 255)
            # Drooping eyes
            img[14:20, 10:16] = np.clip(np.random.randint(100, 125, (6, 6)), 0, 255)  # Left eye droop
            img[14:20, 32:38] = np.clip(np.random.randint(100, 125, (6, 6)), 0, 255)  # Right eye droop
            # Lowered brows
            img[10:14, 8:18] = np.clip(np.random.randint(90, 110, (4, 10)), 0, 255)   # Left brow
            img[10:14, 30:40] = np.clip(np.random.randint(90, 110, (4, 10)), 0, 255)  # Right brow
            # Sad mouth corners
            img[30:34, 12:16] = np.clip(np.random.randint(100, 120, (4, 4)), 0, 255)  # Left corner down
            img[30:34, 32:36] = np.clip(np.random.randint(100, 120, (4, 4)), 0, 255)  # Right corner down

        elif emotion_idx == 5:  # Surprise - wide open eyes, round mouth, raised brows
            # Maximum wide eyes
            img[6:20, 6:18] = np.clip(np.random.randint(200, 240, (14, 12)), 0, 255)   # Left eye very wide
            img[6:20, 30:42] = np.clip(np.random.randint(200, 240, (14, 12)), 0, 255)  # Right eye very wide
            # Round open mouth
            center_y, center_x = 36, 24
            for y in range(28, 44):
                for x in range(16, 32):
                    dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                    if dist <= 8:
                        img[y, x] = np.clip(np.random.randint(40, 80), 0, 255)
            # Maximally raised brows
            for x in range(3, 45):
                height_variation = np.random.randint(4, 8)
                ny = 2 + np.random.randint(0, height_variation)
                if 0 <= ny < 48:
                    img[ny, x] = np.clip(np.random.randint(150, 180), 0, 255)

        elif emotion_idx == 6:  # Neutral - balanced, relaxed features
            # Balanced eyes
            img[10:16, 10:16] = np.clip(np.random.randint(150, 175, (6, 6)), 0, 255)  # Left eye
            img[10:16, 32:38] = np.clip(np.random.randint(150, 175, (6, 6)), 0, 255)  # Right eye
            # Neutral mouth - slight natural curve
            for x in range(18, 30):
                y_offset = int(2 * np.sin((x - 18) * np.pi / 12))
                ny = 32 + y_offset
                if 0 <= ny < 48:
                    img[ny, x] = np.clip(np.random.randint(130, 155), 0, 255)
            # Balanced brows
            img[6:10, 8:18] = np.clip(np.random.randint(120, 140, (4, 10)), 0, 255)   # Left brow
            img[6:10, 30:40] = np.clip(np.random.randint(120, 140, (4, 10)), 0, 255)  # Right brow

        return img

    def _add_facial_texture(self, img):
        """Add realistic facial texture and skin variations"""
        # Ensure img is in valid range before modifications
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Add skin texture variations
        for _ in range(np.random.randint(30, 60)):
            y, x = np.random.randint(5, 43, 2)
            # Create small skin texture variations
            texture_size = np.random.randint(1, 4)
            for dy in range(-texture_size, texture_size + 1):
                for dx in range(-texture_size, texture_size + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 48 and 0 <= nx < 48:
                        variation = np.random.randint(-10, 11)  # Reduced range
                        new_value = img[ny, nx] + variation
                        img[ny, nx] = np.clip(new_value, 60, 220)  # Keep within safe range

        # Add subtle facial hair shadows for some samples (20% chance)
        if np.random.random() > 0.8:
            # Light shadow along jawline
            for x in range(10, 38):
                if np.random.random() > 0.6:
                    jaw_y = 38 + np.random.randint(-2, 3)
                    if 0 <= jaw_y < 48:
                        img[jaw_y, x] = np.random.randint(90, 110)

        return np.clip(img, 0, 255).astype(np.uint8)

    def _apply_lighting_variations(self, img):
        """Apply realistic lighting variations"""
        # Ensure img is float32 for calculations
        img = img.astype(np.float32)

        # Random overall brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * brightness_factor, 0, 255)

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

        # Ensure final result is in valid uint8 range
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

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

    def _preprocess_rafdb(self, test_size=0.2, validation_split=0.1, max_samples=None):
        """Preprocesar datos RAF-DB"""
        print("Procesando imágenes RAF-DB...")

        # Directorio base de imágenes RAF-DB
        rafdb_base_dir = os.path.dirname(self.data_path)

        X = []
        labels = []

        # Limitar muestras si se especifica
        data_to_process = self.data
        if max_samples and len(data_to_process) > max_samples:
            data_to_process = data_to_process.sample(max_samples, random_state=42)

        # Procesar cada imagen
        for idx, row in data_to_process.iterrows():
            try:
                # Construir ruta de imagen (RAF-DB tiene estructura train/test)
                image_path = row['image_path']
                if image_path.startswith('train_'):
                    img_full_path = os.path.join(rafdb_base_dir, 'Image', 'aligned', image_path)
                else:  # test images
                    img_full_path = os.path.join(rafdb_base_dir, 'Image', 'aligned', image_path)

                if not os.path.exists(img_full_path):
                    continue

                # Cargar imagen
                image = cv2.imread(img_full_path)
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
                labels.append(row['emotion'])

            except Exception as e:
                print(f"Error procesando imagen {row['image_path']}: {e}")
                continue

        if len(X) == 0:
            print("❌ No se encontraron imágenes válidas en el dataset RAF-DB.")
            print("🔄 Usando imágenes reales disponibles en data/test/ para entrenamiento...")

            # Usar imágenes reales del directorio data/test/ como alternativa
            return self._load_from_test_directory(test_size, validation_split, max_samples)

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

        print(f"RAF-DB - Muestras procesadas: {len(X)}")
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

    def _create_synthetic_dataset(self, test_size=0.2, validation_split=0.1):
        """Crear dataset sintético cuando no hay imágenes reales"""
        print("📝 Creando dataset sintético de prueba...")

        # Dataset sintético simple
        emotions = list(self.emotions.keys())
        X = []
        labels = []

        samples_per_emotion = 100

        for emotion_idx in emotions:
            for i in range(samples_per_emotion):
                # Crear patrón simple para cada emoción
                img = np.full((48, 48), 128, dtype=np.uint8)

                if emotion_idx == 0:  # Angry - líneas diagonales
                    for j in range(48):
                        if j < 48:
                            img[j, j] = 50
                elif emotion_idx == 1:  # Disgust - patrón de ajedrez
                    img[::4, ::4] = 60
                elif emotion_idx == 2:  # Fear - círculos
                    cv2.circle(img, (24, 24), 15, 70, -1)
                elif emotion_idx == 3:  # Happy - sonrisa
                    cv2.ellipse(img, (24, 32), (12, 8), 0, 0, 180, 200, -1)
                elif emotion_idx == 4:  # Sad - líneas horizontales bajas
                    img[35:40, :] = 40
                elif emotion_idx == 5:  # Surprise - estrella
                    cv2.drawMarker(img, (24, 24), 220, cv2.MARKER_STAR, 20, 2)
                elif emotion_idx == 6:  # Neutral - cruz
                    cv2.line(img, (24, 10), (24, 38), 180, 2)
                    cv2.line(img, (10, 24), (38, 24), 180, 2)

                # Normalizar
                img_normalized = img.astype('float32') / 255.0
                img_normalized = np.expand_dims(img_normalized, axis=-1)

                X.append(img_normalized)
                labels.append(emotion_idx)

        X = np.array(X)
        y = np.array(labels)
        y = to_categorical(y, num_classes=len(self.emotions))

        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=labels
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=validation_split, random_state=42
        )

        print(f"✅ Dataset sintético creado: {len(X)} muestras")

        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_val': self.X_val,
            'y_val': self.y_val,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'class_names': self.emotion_labels
        }
