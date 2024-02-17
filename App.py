import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

def load_model():
    model = keras.models.load_model('my_model')
    return model

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_image_class(model, img_array):
    preds = model.predict(img_array)
    preds = np.argmax(preds, axis=1)
    class_labels = ['diseased cotton leaf', 'diseased cotton plant', 'fresh cotton leaf', 'fresh cotton plant']
    return class_labels[preds[0]]

def main():
    st.title('Cotton Disease Detection')

    page = st.sidebar.selectbox("Choose a page", ["CNN Explanation", "Image Inference"])

    if page == "CNN Explanation":
        st.header("CNN for Cotton Disease Detection")
        st.write("Explanation of the CNN model used for cotton disease detection goes here.")

    elif page == "Image Inference":
        st.header("Image Inference")
        st.write("Upload an image of a cotton leaf or plant to detect if it's diseased or fresh.")

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction
            model = load_model()
            img_array = preprocess_image(uploaded_file)
            prediction = predict_image_class(model, img_array)
            st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
