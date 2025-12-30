import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load saved CNN model
model = tf.keras.models.load_model("../models/pneumonia_cnn_final.h5")

def predict_xray(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        print("PNEUMONIA DETECTED")
    else:
        print("NORMAL LUNGS")

# Example test
predict_xray(r"D:\final_project\data\chest_xray\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg")
