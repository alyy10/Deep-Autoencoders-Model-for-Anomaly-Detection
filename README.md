# Deep Autoencoders for Anomaly Detection with Flask Deployment

This project demonstrates how to build and deploy a **Deep Autoencoders** model for **Anomaly Detection** in transaction data. The model is deployed using **Flask** and served as an API for real-time predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Approach](#approach)
- [Tools](#tools)
- [Project Structure](#project-structure)


---

## Project Overview

The goal of this project is to build a **deep learning model** based on **Autoencoders** for **Anomaly Detection**. The model is trained on a transaction dataset and can identify fraudulent transactions by detecting anomalies in the data. After training the model, it is deployed as a Flask API for real-time anomaly detection.

### What You Will Learn

- **Autoencoders**: Their architecture and how they work for anomaly detection.
- **Deep Learning with Keras**: Build a model using Keras and TensorFlow.
- **Flask Deployment**: Deploying a trained model as an API endpoint with Flask.
- **Data Preprocessing**: Handling missing values, normalization, and exploratory data analysis (EDA).

---

## Tech Stack

- **Programming Language**: Python
- **Deep Learning Framework**: Keras, TensorFlow
- **Web Framework**: Flask (for deploying the model as an API)
- **Database/Storage**: N/A (Using local file storage for datasets)
- **Data Manipulation and Analysis**: Pandas, Numpy
- **Data Visualization**: Matplotlib
- **Server**: Gunicorn (for production deployment)

---

## Approach

### Step-by-Step Process

1. **Business Objective Understanding**:
   - The goal is to identify fraudulent transactions using an unsupervised deep learning approach.
   
2. **Exploratory Data Analysis (EDA)**:
   - Analyze the dataset to understand its structure and key features.
   - Visualize the distributions and handle missing values.

3. **Data Cleaning and Normalization**:
   - Impute missing values and normalize the data using Min-Max scaling to ensure consistent input to the model.

4. **Autoencoder Model Development**:
   - Build the Autoencoder model with Keras. Start with a basic architecture and experiment with different configurations for improved performance.

5. **Model Tuning**:
   - Tune the hyperparameters such as the number of layers, neurons, activation functions, and optimizer settings.

6. **Training the Model**:
   - Train the model on the transaction data and compute the reconstruction error.

7. **Anomaly Detection**:
   - Define a threshold for the reconstruction error to classify normal and anomalous transactions.

8. **Flask API Deployment**:
   - Serve the trained Autoencoder model as an API endpoint using Flask.
   - Set up Gunicorn for running Flask in production.

9. **Real-Time Predictions**:
   - Use the deployed API to make real-time predictions on incoming transaction data.

---

## Tools

- **Keras**: A deep learning library that simplifies model creation and training.
- **TensorFlow**: Backend engine for Keras, used for training deep learning models.
- **Flask**: Web framework for building and serving the trained model as an API.
- **Gunicorn**: WSGI server used for running Flask in production environments.
- **Pandas**: Data manipulation and analysis library for handling datasets.
- **Numpy**: Used for numerical operations, especially for data processing.
- **Matplotlib**: Data visualization library used for plotting graphs and distributions.

---

## Project Structure

This repository contains the following main directories and files:

├── modular_code/ # Contains the core project code
│ ├── input/ # Input datasets (e.g., final_cred_data.csv, test_data.csv)
│ ├── lib/ # Notebooks and additional libraries
│ │ ├── Deep-Autoencoder.ipynb # Notebook for training the autoencoder model
│ │ └── Model_Api.ipynb # Notebook for setting up the Flask API
│ ├── output/ # Model and output files (e.g., pickle files, model weights)
│ ├── requirements.txt # List of required Python packages
│ └── src/ # Source code for the project
│ ├── init.py # Package initialization
│ ├── Engine.py # Main engine code for training and serving model
│ └── ML_Pipeline/ # Machine learning pipeline for model training and predictions
│ ├── Preprocess.py # Data preprocessing functions
│ ├── Utils.py # Utility functions for model saving/loading
│ └── Train_Model.py # Model training code
└── README.md # Project overview and setup instructions


---

