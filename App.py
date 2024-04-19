import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_class(model, img_array):
    preds = model(img_array)
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
            model_path = 'path/to/saved_model.pb'  # Update this with the correct path
            model = load_model(model_path)
            img_array = preprocess_image(uploaded_file)
            prediction = predict_image_class(model, img_array)
            st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
