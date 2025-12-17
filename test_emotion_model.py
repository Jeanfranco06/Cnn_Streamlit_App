import sys
sys.path.append('src')
from src.emotion_model import EmotionClassifier
import tensorflow as tf
import numpy as np

# Test current model loading and GradCAM
try:
    classifier = EmotionClassifier()
    print('Model loaded successfully')

    # Test dummy prediction to initialize layers
    dummy_input = tf.zeros((1, 48, 48, 1))
    result = classifier.model(dummy_input, training=False)
    print('Dummy prediction successful')

    # Test GradCAM
    from src.gradcam import GradCAM
    gradcam = GradCAM(classifier.model)
    print('GradCAM initialized successfully')

    # Test heatmap generation
    heatmap = gradcam.compute_heatmap(dummy_input)
    print('Heatmap computation successful')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
