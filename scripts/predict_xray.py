import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model(r"D:\final_project\models\pneumonia_cnn_final.h5")

IMG_SIZE = 224   # MUST MATCH TRAINING

def predict_xray(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred >= 0.5:
        print("🦠 PNEUMONIA DETECTED")
    else:
        print("🫁 NORMAL LUNGS")

if __name__ == "__main__":
    predict_xray(r"D:\final_project\data\chest_xray\chest_xray\test\NORMAL\IM-0001-0001.jpeg")
