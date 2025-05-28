import streamlit as st
import requests
from PIL import Image
import io
import json

# Set page config
st.set_page_config(
    page_title="Cat/Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# Title and description
st.title("🐱 Cat/Dog Image Classifier 🐶")
st.markdown("""
Upload an image of a cat or dog, and our AI model will predict which one it is!
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to bytes for API request
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Create files dict for API request
    files = {"file_upload": ("image.jpg", img_byte_arr, "image/jpeg")}
    
    # Make API request
    try:
        with st.spinner("Analyzing image..."):
            response = requests.post("http://localhost:8080/catdog_classification/predict", files=files)
            
        if response.status_code == 200:
            result = response.json()
            
            # Create two columns for displaying results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Results")
                # Display class prediction
                prediction = "🐱 Cat" if result["predicted_id"] == 0 else "🐶 Dog"
                st.metric("Predicted Class", prediction)
                
                # Display class ID
                st.metric("Class ID", result["predicted_id"])
                
            with col2:
                st.subheader("Probabilities")
                # Display probabilities

                cat_prob = result["probs"][0] * 100
                dog_prob = result["probs"][1] * 100
                
                # Create progress bars for probabilities
                st.metric("Cat Probability", f"{cat_prob:.2f}%")
                st.progress(cat_prob/100)
                
                st.metric("Dog Probability", f"{dog_prob:.2f}%")
                st.progress(dog_prob/100)
                
        else:
            st.error("Error occurred while processing the image. Please try again.")
            
    except Exception as e:
        st.error(f"Error connecting to the server: {str(e)}")
        st.info("Make sure the FastAPI server is running on http://localhost:8080") 