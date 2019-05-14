from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Flatten, Dense
import numpy as np
import tensorflow as tf


class FeatureExtractor:
    def __init__(self):
        model = VGG16(weights='imagenet', include_top=True)
        self.model = Model(inputs=model.input, outputs=model.get_layer('fc1').output)
        self.graph = tf.get_default_graph()

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        with self.graph.as_default():
            feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
            return feature / np.linalg.norm(feature)  # Normalize