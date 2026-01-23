# 📊 Scalable Sentiment Analysis Pipeline

## Overview
This project implements an end-to-end **sentiment analysis pipeline** for classifying textual input as *positive* or *negative*.  
The emphasis is on **correct machine learning workflow design**, including offline training, reproducible evaluation, and inference-only deployment.

The system is structured to clearly separate:
- Training
- Evaluation
- Inference
- Presentation (UI)

This mirrors real-world ML system design and avoids data leakage or unintended retraining.

---

## Problem Statement
Given unstructured textual data (e.g., user reviews), predict sentiment while ensuring:
- fair evaluation on unseen data
- reproducibility
- reusable inference logic

---

## Dataset
- **IMDb Movie Reviews Dataset**
- Binary sentiment labels:
  - `0` → Negative
  - `1` → Positive

The dataset is accessed programmatically using the Hugging Face `datasets` library.

---

## Core Approach

### Feature Engineering
- TF-IDF Vectorization
  - Vocabulary size capped to control dimensionality
  - Unigrams and bigrams to capture contextual sentiment (e.g., negation)
  - Stop-word removal to reduce noise

### Model
- Logistic Regression
  - Strong linear baseline for high-dimensional sparse text data
  - Interpretable and computationally efficient
  - Suitable for academic benchmarking

---

## Project Structure
```
├── data/
│ └── README.md # Notes on dataset handling (no raw data stored)
│
├── src/
│ ├── scalability/
│ │ └── spark_pipeline.py # Conceptual PySpark scalability extension
│ │
│ ├── train.py # Offline model training and artifact persistence
│ ├── evaluate.py # Reproducible evaluation using saved artifacts
│ └── inference.py # Inference-only prediction interface
│
├── app.py # Streamlit UI (presentation layer only)
├── model.pkl # Trained Logistic Regression model
├── vectorizer.pkl # Trained TF-IDF vectorizer
│
├── requirements.txt # Python dependencies
├── runtime.txt # Runtime specification for deployment
├── README.md # Project overview and documentation
└── .gitignore
```
---

## Training (`train.py`)
- Loads the IMDb dataset
- Performs a stratified train/test split to preserve class balance
- Fits the TF-IDF vectorizer **only on training data**
- Trains a Logistic Regression classifier
- Evaluates performance on unseen test data
- Saves trained artifacts for reuse

**Key design decision:**  
Training is performed offline and only once.

---

## Evaluation (`evaluate.py`)
- Reloads saved artifacts
- Reconstructs the test split using the same random seed and stratification
- Evaluates performance without retraining

This ensures reproducibility and prevents data leakage.

**Sample performance:**  
- Accuracy: ~90%  
- Balanced precision and recall across classes

---

## Inference (`inference.py`)
Defines a reusable inference interface that:
- loads trained artifacts once
- accepts raw text input
- returns predicted sentiment and model confidence

This module is independent of any UI and can be reused in APIs or batch pipelines.

---

## Streamlit Application (`app.py`)
The Streamlit app acts purely as a **presentation layer**:
- no training
- no evaluation
- no feature fitting

It simply collects user input, calls the inference module, and displays predictions.

---

## Scalability Considerations (Conceptual)
A conceptual PySpark pipeline is included under `scalability/` to illustrate how the same logic could be adapted for distributed environments.

**Important:**
- This code is not executed as part of the current setup
- No performance claims are made
- The reproducible implementation uses scikit-learn

---

## Key Takeaways
- Clear separation of ML lifecycle stages
- Reproducible experimentation
- Honest, defensible performance reporting
- Design aligned with real-world ML systems

---

## Future Work
- Probability calibration
- Threshold tuning
- Multi-class sentiment classification
- REST API deployment

---

## Author
**Kayan Sirsat**  
Bachelor of Engineering – Computer Engineering  
Aspiring Master’s student in Computer Science (AI/ML)

