import streamlit as st
from src.inference import SentimentPredictor

# Page config

st.set_page_config(
    page_title = "Sentiment Analysis",
    layout = "centered"
)

# Load Inference Model

@st.cache_resource
def load_predictor():
    """
    Load the inference-only sentiment predictor.
    Cached to avoid reloading on every interaction.
    """
    return SentimentPredictor()

predictor = load_predictor()

# UI components

st.title("Sentiment Analysis - Inference Only")
st.write(
    "This application performs sentiment prediction using a pre-trained"
    "TF-IDP + Logistic Regression model. The model is Trained offline and "
    "used here only for inference."
)

user_input = st.text_area(
    "Enter a review or sentence",
    height = 150,
    placeholder = "Type your text here..."
)

# Prediction logic

if st.button("Predict Sentiment"):
    if not user_input():
        st.warning("Please enter some text for prediction.")
    else:
        result = predictor.predict(user_input)

        label = result["label"]
        confidence = result["confidence"]

        if label == "positive":
            st.success(f"Positve Sentiment (Confidence: {confidence:.2%})")
        else:
            st.error(f"Negative Sentiment (Confidence: {confidence:.2%})")