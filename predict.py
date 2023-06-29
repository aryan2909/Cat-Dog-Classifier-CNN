import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the pretrained  model
#I loaded pretrained model through transfer learning because  my model needed more data set to be accurate
model = tf.keras.applications.VGG16(weights='imagenet')
# model = tf.keras.models.load_model('cat_dog_classifier.h5')


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict_image_class(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    preds_decoded = decode_predictions(preds, top=1)[0]
    return preds_decoded[0][1]


cat_image_path = 'train/cats/951.jpg'
dog_image_path = 'train/dogs/911.jpg'

# Predict the class of the cat image
cat_class = predict_image_class(cat_image_path)
print("Cat image prediction:", cat_class)

# Predict the class of the dog image
dog_class = predict_image_class(dog_image_path)
print("Dog image prediction:", dog_class)
