# **Customer Segmentation App 🛒**
The app is available on My Streamlit cloud account go check it out!!
## Overview

This project involves customer segmentation for an e-commerce company using machine learning techniques. The goal is to categorize customers into different segments based on their demographic and behavioral data. The project uses a **K-Means clustering** algorithm to segment customers and a **Streamlit** app to deploy the model and make predictions interactively. 🤖💡

## Project Description

The customer segmentation model was developed using various customer attributes such as:

* **Demographics**: Age, Gender, City, Membership Type 👩‍🦳👨‍🦳🌆
* **Purchase Behavior**: Items Purchased, Purchase Amount, Discount Applied 🛍️💳💸
* **Satisfaction**: Satisfaction Level 😀😐😞

The **K-Means clustering** model is trained to group customers into clusters based on similar characteristics. A Streamlit app is then built to allow users to input new customer data and predict the cluster to which the customer belongs. 🖥️🔮

## Features

* **Predictive Modeling**: K-Means clustering model to predict the segment of a new customer based on input data. 🎯
* **Streamlit Interface**: User-friendly web interface built using Streamlit to input new customer data and receive predictions. 💻📊
* **Model Deployment**: The trained model is deployed via Streamlit, allowing you to easily predict customer segments. 🚀

## Setup Instructions ⚙️

Follow the steps below to set up and run the app locally:

### Prerequisites 🛠️

1. **Python 3.6+**
2. Install the necessary dependencies. You can install the required packages by using the following:

   ```bash
   pip install -r requirements.txt
   ```

### Dependencies 📦

The project uses the following key libraries:

* **Pandas**: Data manipulation and analysis. 📊
* **Scikit-learn**: Machine learning tools for clustering and data preprocessing. 🤖
* **Streamlit**: Framework to create the web app for model deployment. 🌐
* **Pickle**: Serialization of models, encoders, and scalers for reuse. 💾

### Running the Streamlit App 🚀

1. Navigate to the project directory.

2. Run the following command to start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. The app will open in your default web browser. 🌍

### Example of App Interface 📱

When you run the app, the following features are available to users:

* **Customer Inputs**: Users can select gender, age, city, membership type, number of items purchased, purchase amount, satisfaction level, and discount applied. ✍️
* **Predict Button**: After entering the data, users can press the "Predict" button to display the predicted customer segment (cluster). 🔮
* **Cluster Characteristics**: Once the prediction is made, the app also displays the characteristics of the predicted cluster, including typical attributes of customers in that cluster. 📈

---

## File Structure 📂

Here is an overview of the project file structure:

```
/project-directory
├── app.py                   # Streamlit app for customer segmentation
├── kmeans_model.pkl         # Saved KMeans model
├── label_encoders.pkl       # Saved LabelEncoders
├── scaler.pkl               # Saved StandardScaler
├── one_hot_columns.pkl      # List of one-hot encoded columns
├── customer_data.csv        # Sample customer data for model training
├── requirements.txt         # List of dependencies
└── README.md                # Project description
```

---

## How to Save the Model and Scalers 💾

After training the model, save the following components to disk:

* **KMeans Model**: The trained KMeans model for clustering customer segments. 🤖
* **LabelEncoders**: Encoders used to transform categorical variables (e.g., 'Satisfaction Level', 'Membership Type'). 🔄
* **StandardScaler**: Scaler used to normalize the features before feeding them into the KMeans model. 📊
* **One-Hot Columns**: The columns used for one-hot encoding to ensure consistency in input data. 🔠

These components are saved in `.pkl` files for later use. 📁

---

## Conclusion 🎉
This project provides an easy-to-use tool for customer segmentation using machine learning. By using this Streamlit app, businesses can predict customer segments in real-time and personalize their marketing strategies accordingly. 📈💡
