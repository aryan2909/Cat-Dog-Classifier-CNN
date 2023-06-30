# Cat and Dog Classifier

This repository contains scripts for training and predicting with a cat and dog classifier using transfer learning with the VGG16 model and Convolutional Neural Networks (CNN).

## Components

### `train.py`

- **Importing Libraries**: The necessary libraries are imported, including TensorFlow, Keras, and the VGG16 model.

- **Data Preparation**: The script assumes that I have organized my data into separate folders for cats and dogs. It uses the `ImageDataGenerator` class from Keras to load and preprocess the image data. The images are resized to (224, 224) to match the input size expected by the VGG16 model.

- **Model Creation**: The VGG16 model is imported, excluding the top classification layers. A new classification head is added on top of the pre-trained base model, which includes Convolutional Neural Networks (CNN) layers. CNN layers are responsible for extracting hierarchical features from images.

- **Model Compilation**: The model is compiled with the binary cross-entropy loss function and the Adam optimizer.

- **Model Training**: The model is trained using the `fit` method. The training data is passed to the `fit` method using the `train_generator` object. The number of epochs can be adjusted according to the desired training duration.

- **Model Evaluation**: After training, the model's performance is evaluated using the test data. The accuracy and loss are printed to assess the model's performance.

- **Model Saving**: The trained model is saved to a file named `cat_dog_classifier.h5` using the `save` method.

### `predict.py`

- **Importing Libraries**: The necessary libraries are imported, including TensorFlow, Keras, and the VGG16 model.

- **Loading the Model**: The trained model is loaded from the saved file using the `load_model` function.

- **Preprocessing the Image**: The input image is loaded and preprocessed using the `preprocess_image` function. The image is resized to (224, 224) and preprocessed to match the input format expected by the VGG16 model.

- **Making Predictions**: The preprocessed image is passed to the loaded model's `predict` method to obtain the prediction probabilities. A threshold of 0.5 is used to classify the image as either a cat or a dog.

- **Displaying the Prediction**: The prediction result is printed, indicating whether the image is classified as a cat or a dog.

## Class Activation Mapping (CAM)

The repository also includes the implementation of Class Activation Mapping (CAM) to visualize the important regions of the input image that contribute to the model's prediction. The CAM outputs are saved as numpy files.

To use the CAM functionality, follow these steps:
1. Ensure that you have trained the model using `train.py` and have the trained model file (`cat_dog_classifier.h5`) available.
2. Use the `predict_cam.py` script to predict an image and obtain the CAM output.
3. The CAM output will be saved as a numpy file in the `cam_outputs` directory.


## Why the Model Works and Aspects Worked Upon

- **Transfer Learning**: The model utilizes transfer learning by leveraging the pre-trained VGG16 model. The VGG16 model is trained on a large dataset and has learned to extract meaningful features from images. By using the pre-trained weights and freezing the layers of the base model, we benefit from the learned representations, enabling us to train a new classification head specifically for cat and dog classification.

- **Convolutional Neural Networks (CNN)**: The model includes CNN layers that are responsible for learning and extracting hierarchical features from images. These layers apply filters to the input image, capturing local patterns and spatial relationships, which are crucial for distinguishing between cats and dogs.

- **Image Manipulation**: The images are manipulated using the `ImageDataGenerator` class, which performs data augmentation techniques like resizing, rescaling, and applying random transformations. These techniques help to increase the diversity and robustness of the training data, reducing overfitting and improving the model's generalization capabilities.

- **Model Improvement**: Several aspects can be further explored to improve the model's performance:
  - Hyperparameter Tuning: Experimenting with different hyperparameters such as learning rate, batch size, and number of layers can impact the model's performance.
  - Data Augmentation: Increasing the variety of data augmentation techniques, such as rotation, zooming, and flipping, can further enhance the model's ability to generalize.
  - Regularization Techniques: Employing regularization techniques like dropout or L2 regularization can help prevent overfitting and improve the model's ability to generalize to unseen data.
  - Different Pre-trained Models: Besides VGG16, other pre-trained models like ResNet, Inception, or EfficientNet can be explored to see if they provide better performance for the cat and dog classification task.

Overall, the combination of transfer learning, CNN layers, data augmentation, and careful model tuning contribute to the effectiveness of the model in classifying cats and dogs. However, it's important to note that model performance can vary depending on the quality and diversity of the training data, as well as the complexity of the classification task. Further experimentation and fine-tuning may be required to achieve optimal results.

