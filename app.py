import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load model
model = load_model('fashion_cnn_model.keras')
class_names = ['Accessories', 'Apparel', 'Footwear', 'Personal Care']

# Page config
st.set_page_config(
    page_title="E-Commerce Product Classifier",
    page_icon="🛍️",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-bag.png", width=80)
    st.title("🛍️ About This App")
    st.markdown("""
    This app uses a **Convolutional Neural Network (CNN)** 
    trained on **44,288 real e-commerce products** 
    from Myntra to classify product images.
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("Test Accuracy", "93.85%")
    st.metric("Total Images Trained", "44,288")
    st.metric("Categories", "4")
    
    st.markdown("---")
    st.markdown("### 🏷️ Categories")
    st.markdown("👜 Accessories")
    st.markdown("👕 Apparel")
    st.markdown("👟 Footwear")
    st.markdown("🧴 Personal Care")
    
    st.markdown("---")
    st.markdown("### 🧠 CNN Architecture")
    st.markdown("""
    - 3 Convolutional Blocks
    - Batch Normalization
    - MaxPooling Layers
    - Dropout (0.5)
    - Data Augmentation
    - Early Stopping
    """)
    
    st.markdown("---")
    st.markdown("Built by **[Your Name]**")
    st.markdown("Deep Learning Final Project")

# Main content
st.title("🛍️ E-Commerce Product Classifier")
st.markdown("### Powered by CNN — Deep Learning")
st.markdown("Upload a product image and the AI will classify it!")
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("Upload a product image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    with st.spinner("Classifying..."):
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    with col2:
        st.markdown("## 🎯 Prediction Result")
        st.markdown("---")
        st.metric("Category", predicted_class)
        st.metric("Confidence", f"{confidence:.2f}%")
        
        st.markdown("### Confidence for all categories:")
        for i, cls in enumerate(class_names):
            st.progress(float(prediction[0][i]), 
                       text=f"{cls}: {prediction[0][i]*100:.1f}%")

st.markdown("---")
st.markdown("Built with TensorFlow & Streamlit | Deep Learning Final Project")