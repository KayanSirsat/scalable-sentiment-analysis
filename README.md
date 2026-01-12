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


### 🏗️ Architecture Design
To ensure both **scalability** for big data and **low latency** for end-users, this project utilizes a hybrid approach:

1.  **Offline Training Layer (The "Kitchen"):**
    * **Tech:** Apache Spark (PySpark)
    * **Role:** Handles massive datasets (GBs/TBs) using distributed computing. It performs tokenization, HashingTF, and logistic regression training on a cluster.
    * **File:** `spark_pipeline.py`
    
2.  **Online Inference Layer (The "Delivery"):**
    * **Tech:** Scikit-Learn & Streamlit
    * **Role:** A lightweight, low-latency web application that mimics the trained logic to serve predictions to users in milliseconds.
    * **File:** `app.py`

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
