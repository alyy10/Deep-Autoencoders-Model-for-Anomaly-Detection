# Deep Autoencoders for Anomaly Detection with Flask Deployment

This project focuses on building and deploying a **Deep Autoencoders** model for **Anomaly Detection** in transaction data. The deep learning model is trained using **Keras** and **TensorFlow** and deployed as a **Flask API** for real-time predictions. The system leverages **autoencoders**, a powerful unsupervised learning technique, to detect fraudulent transactions by learning compressed representations of normal data and identifying anomalies based on reconstruction error.

## Overview
Autoencoders are neural networks that learn efficient encodings of data, typically used for dimensionality reduction or anomaly detection. The model consists of an **encoder** and a **decoder**, where the encoder compresses the data into a lower-dimensional representation, and the decoder attempts to reconstruct the original data. This model is trained to minimize the reconstruction error between the input and the output, allowing it to detect anomalous data points that deviate significantly from the learned patterns.

In this project, we apply **autoencoders** to a **transaction dataset** containing over 100K records, with features representing various attributes of credit card transactions. Anomalies are identified based on the reconstruction error, where transactions with high error values are considered anomalies (e.g., fraudulent activities).

## Learning Outcomes
- **Understanding Autoencoders**: Learn how autoencoders work, their architecture, and applications in anomaly detection.
- **Deep Learning with Keras**: Build and train an autoencoder using Keras and TensorFlow.
- **Flask for API Deployment**: Serve the trained model via Flask as an API, enabling real-time predictions.
- **Data Preprocessing**: Handling missing values, normalization, and exploratory data analysis (EDA).

## Tech Stack
- **Programming Language**: Python
- **Deep Learning Framework**: Keras (with TensorFlow backend)
- **API Service**: Flask (for deploying the model as a RESTful API)
- **Web Server**: Gunicorn (for production deployment)
- **Data Analysis & Manipulation**: Pandas, Numpy
- **Data Visualization**: Matplotlib

## Approach
The project follows a systematic approach to build the autoencoder model and deploy it using Flask:

1. **Business Objective Understanding**: The goal is to use an unsupervised deep learning model to identify fraudulent transactions from a set of credit card transactions.
2. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand its structure, perform data visualizations, and inspect for missing values or outliers.
3. **Data Cleaning and Normalization**: Handle missing values using imputation, and normalize the features using Min-Max scaling to prepare the data for training.
4. **Autoencoder Theory**: Autoencoders consist of an encoder and a decoder. The encoder reduces the dimensionality of the input, and the decoder attempts to reconstruct the original data from this compressed representation.
5. **Model Building**: Build an autoencoder model using Keras with an architecture consisting of several dense layers. The model is trained to minimize the reconstruction error.
6. **Model Tuning**: Experiment with different architectures, activation functions, and hyperparameters (e.g., number of layers, neurons, learning rate) to optimize performance.
7. **Training the Model**: Train the model on the transaction data and compute the reconstruction error.
8. **Anomaly Detection**: Define a threshold for the reconstruction error to classify normal and anomalous transactions.
9. **Flask API Deployment**: Deploy the trained model using Flask. Set up Gunicorn for running Flask in production.
10. **Real-Time Predictions**: Once the API is running, it can be used to make predictions on new data by sending HTTP POST requests with transaction features.

## Project Structure
The project is organized into the following key folders:

### Folder Descriptions:

- **`modular_code/`**: Contains the core project files, including input datasets, source code, and model-related files.
  - **`input/`**: Directory where datasets like `final_cred_data.csv` and `test_data.csv` are stored.
  - **`lib/`**: Contains Jupyter notebooks and additional libraries required for model training and API setup.
  - **`output/`**: Stores output files, such as trained model weights and pickle files.
  - **`requirements.txt`**: Specifies the Python packages required to run the project.
  - **`src/`**: Source code for the project, including the engine and machine learning pipeline.
    - **`Engine.py`**: Core code for training and serving the model.
    - **`ML_Pipeline/`**: Contains modules for data preprocessing, utility functions, and training the model.
  
- **`README.md`**: This file, providing an overview of the project, setup instructions, and usage.
