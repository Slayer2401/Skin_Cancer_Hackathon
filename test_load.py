import tensorflow as tf
import traceback
import sys

print('TF Version:', tf.__version__)
try:
    model = tf.keras.models.load_model('skin_cancer_model_v2_perfect.h5', compile=False)
    print('Loaded successfully!')
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
