import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Load and preprocess the image
img_path = 'images/61DMgCXRRGL._SY879_.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)
decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions)

# Print the top predictions
for _, label, score in decoded_predictions[0]:
    print(f"{label}: {score:.2f}")