import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


model = tf.keras.applications.VGG16(weights='imagenet')


base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)
gap_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
model_cam = tf.keras.Model(inputs=base_model.input, outputs=gap_layer)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_image_class(img_path):
    img = preprocess_image(img_path)
    

    feature_maps = base_model.predict(img)
    

    cam_weights = np.mean(feature_maps, axis=(1, 2))
    

    cam_weights = np.reshape(cam_weights, (1, 1, -1))
    

    preds = model.predict(img)
    preds_decoded = decode_predictions(preds, top=1)[0]
    

    cam_output = np.dot(feature_maps, np.squeeze(cam_weights, axis=(0, 1)))
    
    return preds_decoded[0][1], cam_output

cat_image_path = 'train/cats/951.jpg'
dog_image_path = 'train/dogs/911.jpg'
output_folder = 'cam_outputs' 


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


cat_class, cat_cam_output = predict_image_class(cat_image_path)
print("Cat image prediction:", cat_class)


dog_class, dog_cam_output = predict_image_class(dog_image_path)
print("Dog image prediction:", dog_class)

# Saving the CAM outputs to files inside the output folder
cat_output_file = os.path.join(output_folder, 'cat_cam_output.npy')
dog_output_file = os.path.join(output_folder, 'dog_cam_output.npy')
np.save(cat_output_file, cat_cam_output)
np.save(dog_output_file, dog_cam_output)
