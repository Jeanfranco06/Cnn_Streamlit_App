import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Implementación de GradCAM para clasificación de emociones

        Args:
            model: Modelo Keras entrenado
            layer_name: Nombre de la capa convolucional a usar para GradCAM.
                       Si es None, intentará encontrar la última capa convolucional.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()

        # Create GradCAM model
        self.gradcam_model = self._create_gradcam_model()

    def _find_last_conv_layer(self):
        """Encontrar la última capa convolucional en el modelo"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No se encontró capa convolucional en el modelo")

    def _create_gradcam_model(self):
        """Crear un modelo que produce tanto la última capa conv como las predicciones finales"""
        # Get the last convolutional layer
        last_conv_layer = self.model.get_layer(self.layer_name)

        # Create a new model that outputs both the conv layer and final predictions
        # We need to build this dynamically during computation to avoid initialization issues
        return None

    def compute_heatmap(self, image, class_idx=None):
        """
        Calcular mapa de calor GradCAM para una imagen dada

        Args:
            image: Array de imagen preprocesada de forma (1, 48, 48, 1)
            class_idx: Índice de clase para calcular mapa de calor. Si es None, usa clase predicha.

        Returns:
            heatmap: Array de mapa de calor de forma (48, 48)
        """
        # Get the last convolutional layer
        last_conv_layer = self.model.get_layer(self.layer_name)

        # Create a submodel for just the conv layer
        conv_model = Model(inputs=self.model.inputs, outputs=last_conv_layer.output)

        with tf.GradientTape() as tape:
            # Forward pass through conv model
            conv_outputs = conv_model(image)

            # Forward pass through full model for predictions
            predictions = self.model(image)

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            # Get the score for the target class
            class_score = predictions[:, class_idx]

        # Compute gradients with respect to conv outputs
        grads = tape.gradient(class_score, conv_outputs)

        if grads is None:
            # Fallback: use conv output magnitude as heatmap
            heatmap = tf.reduce_mean(tf.abs(conv_outputs[0]), axis=-1)
        else:
            # Global average pooling of gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight the conv outputs with gradients
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.reduce_sum(heatmap, axis=-1)

        # Apply ReLU and normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap.numpy()

        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        return heatmap

    def generate_heatmap(self, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Generar visualización GradCAM superpuesta en la imagen original

        Args:
            image: Imagen PIL o array numpy
            alpha: Transparencia de la superposición del mapa de calor
            colormap: Mapa de colores OpenCV a usar

        Returns:
            figura matplotlib con la visualización
        """
        # Preprocess image for model
        if isinstance(image, Image.Image):
            # Convert to grayscale if needed
            if image.mode != 'L':
                display_image = np.array(image.convert('L'))
            else:
                display_image = np.array(image)
        else:
            display_image = image.copy()

        # Prepare image for model input
        processed_image = self._preprocess_for_model(image)

        # Compute heatmap
        heatmap = self.compute_heatmap(processed_image)

        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (display_image.shape[1], display_image.shape[0]))

        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(display_image[:, :, np.newaxis] if display_image.ndim == 2
                                else display_image, 1 - alpha, heatmap, alpha, 0)

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        ax1.imshow(display_image, cmap='gray' if display_image.ndim == 2 else None)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Heatmap
        ax2.imshow(heatmap)
        ax2.set_title('GradCAM Heatmap', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Overlay
        ax3.imshow(overlay)
        ax3.set_title('Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')

        plt.tight_layout()
        return fig

    def _preprocess_for_model(self, image):
        """Preprocesar imagen para entrada del modelo"""
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            image = np.array(image)
        elif isinstance(image, np.ndarray):
            # If RGB, convert to grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Resize to 48x48
        image = cv2.resize(image, (48, 48))

        # Normalize
        image = image.astype('float32') / 255.0

        # Add batch and channel dimensions
        image = np.expand_dims(image, axis=[0, -1])

        return image

    def get_important_regions(self, image, threshold=0.5):
        """
        Obtener las regiones más importantes basadas en el mapa de calor GradCAM

        Args:
            image: Imagen de entrada
            threshold: Umbral para considerar una región como importante

        Returns:
            mask: Máscara binaria de regiones importantes
            regions: Lista de cajas delimitadoras para regiones importantes
        """
        processed_image = self._preprocess_for_model(image)
        heatmap = self.compute_heatmap(processed_image)

        # Resize heatmap to original image size
        if isinstance(image, Image.Image):
            orig_size = image.size[::-1]  # PIL size is (width, height)
        else:
            orig_size = image.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (orig_size[1], orig_size[0]))

        # Create binary mask
        mask = (heatmap_resized > threshold).astype(np.uint8)

        # Find contours (important regions)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, x+w, y+h))

        return mask, regions

    def analyze_attention_patterns(self, images, labels=None):
        """
        Analizar patrones de atención a través de múltiples imágenes

        Args:
            images: Lista de imágenes
            labels: Etiquetas correspondientes (opcional)

        Returns:
            analysis: Diccionario con resultados de análisis de atención
        """
        heatmaps = []
        attention_centers = []

        for image in images:
            processed_image = self._preprocess_for_model(image)
            heatmap = self.compute_heatmap(processed_image)

            heatmaps.append(heatmap)

            # Calculate attention center (weighted center of mass)
            y_coords, x_coords = np.indices(heatmap.shape)
            total_weight = np.sum(heatmap)

            if total_weight > 0:
                center_y = np.sum(y_coords * heatmap) / total_weight
                center_x = np.sum(x_coords * heatmap) / total_weight
                attention_centers.append((center_x, center_y))
            else:
                attention_centers.append((24, 24))  # Center of 48x48 image

        # Convert to numpy arrays
        heatmaps = np.array(heatmaps)
        attention_centers = np.array(attention_centers)

        analysis = {
            'heatmaps': heatmaps,
            'attention_centers': attention_centers,
            'mean_attention_center': np.mean(attention_centers, axis=0),
            'attention_spread': np.std(attention_centers, axis=0),
            'average_heatmap': np.mean(heatmaps, axis=0)
        }

        if labels is not None:
            # Analyze attention patterns by emotion
            unique_labels = np.unique(labels)
            emotion_attention = {}

            for label in unique_labels:
                mask = labels == label
                emotion_heatmaps = heatmaps[mask]
                emotion_centers = attention_centers[mask]

                if len(emotion_heatmaps) > 0:
                    emotion_attention[label] = {
                        'mean_heatmap': np.mean(emotion_heatmaps, axis=0),
                        'mean_center': np.mean(emotion_centers, axis=0),
                        'center_spread': np.std(emotion_centers, axis=0),
                        'sample_count': len(emotion_heatmaps)
                    }

            analysis['emotion_attention'] = emotion_attention

        return analysis

    def visualize_attention_analysis(self, analysis):
        """
        Visualizar resultados de análisis de atención

        Args:
            analysis: Salida de analyze_attention_patterns

        Returns:
            figura matplotlib
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Average heatmap
        avg_heatmap = analysis['average_heatmap']
        im1 = ax1.imshow(avg_heatmap, cmap='jet')
        ax1.set_title('Average Attention Heatmap', fontsize=14, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # Attention centers scatter plot
        centers = analysis['attention_centers']
        ax2.scatter(centers[:, 0], centers[:, 1], alpha=0.6, s=50, c='blue', edgecolors='black')
        ax2.set_title('Attention Centers Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.set_xlim(0, 48)
        ax2.set_ylim(0, 48)
        ax2.grid(True, alpha=0.3)

        # Mark mean center
        mean_center = analysis['mean_attention_center']
        ax2.scatter(mean_center[0], mean_center[1], c='red', s=200, marker='*',
                   edgecolors='black', linewidth=2, label='Mean Center')
        ax2.legend()

        # Attention spread
        emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

        if 'emotion_attention' in analysis:
            emotion_names = []
            spreads = []

            for emotion_idx, data in analysis['emotion_attention'].items():
                emotion_names.append(emotions[emotion_idx])
                spreads.append(np.linalg.norm(data['center_spread']))

            bars = ax3.bar(emotion_names, spreads, color='skyblue', alpha=0.7)
            ax3.set_title('Attention Center Spread by Emotion', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Spread (pixels)')
            ax3.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, spread in zip(bars, spreads):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{spread:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'No emotion-specific data available',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Attention Center Spread by Emotion', fontsize=14, fontweight='bold')

        # Statistics text
        stats_text = f"""
        Overall Statistics:
        Mean Attention Center: ({mean_center[0]:.1f}, {mean_center[1]:.1f})
        Attention Spread: ({analysis['attention_spread'][0]:.2f}, {analysis['attention_spread'][1]:.2f})
        Total Samples: {len(centers)}
        """

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_title('Attention Analysis Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')

        plt.suptitle('GradCAM Attention Pattern Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        return fig
