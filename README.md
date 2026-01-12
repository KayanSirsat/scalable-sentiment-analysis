# 📊 Scalable Real-Time Sentiment Analysis Pipeline

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Apache%20Spark-PySpark-orange)](https://spark.apache.org/)

### 🚀 Live Demo
**[Click here to try the Live Web App](https://sentiment-demo-ultctfrfumn29rkdjy9uug.streamlit.app/)**

---

# 📊 Scalable Customer Sentiment Analysis Pipeline

## Overview
This project implements an end-to-end sentiment analysis pipeline for customer-like textual feedback.  
The focus is on **robust baseline modeling, proper evaluation and low-latency inference**, rather than model complexity.

A standard benchmark dataset (IMDb Reviews) is used as a **proxy for customer sentiment classification** to ensure reliable training and evaluation.

---

## Architecture Design

The system follows a two-layer architecture to separate model training from real-time inference.
This design mirrors how large-scale ML systems are typically deployed, while keeping the implementation simple and reproducible.

### Offline Training Layer
- **Purpose:** Model training and feature extraction
- **Tech:** Apache Spark (PySpark), Spark MLlib
- **Role:** Implements tokenization, TF-IDF / HashingTF feature extraction and Logistic Regression training.
- **Note:** In this repository, training is demonstrated on a benchmark dataset for clarity, though the pipeline is designed to scale using distributed processing.

**File:** `spark_pipeline.py`

### Online Inference Layer
- **Purpose:** Low-latency sentiment prediction for user input
- **Tech:** Scikit-Learn, Streamlit
- **Role:** Loads trained artifacts and performs inference in milliseconds.
- **Design Choice:** Inference is kept lightweight and inference-only; retraining is handled offline to avoid data leakage.

**File:** `app.py`

---

## 🛠️ Tech Stack
* **Big Data Engine:** Apache Spark (PySpark), Spark MLlib
* **Machine Learning:** Logistic Regression, HashingTF / CountVectorizer, NLP
* **Web Framework:** Streamlit (Python)
* **Language:** Python 3.x
* **Version Control:** Git & GitHub

---

## 📂 Project Structure
```text
├── app.py                  # The Live Web Application (Streamlit + Scikit-Learn)
├── spark_pipeline.py       # The Distributed Training Logic (PySpark)
├── requirements.txt        # Dependencies for the cloud deployment
└── README.md               # Documentation
