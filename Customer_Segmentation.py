import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model, encoders, and scaler
with open("customer_segmentation_kmeans.pkl", "rb") as f:
    kmeans_model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("one_hot_columns.pkl", "rb") as f:
    one_hot_columns = pickle.load(f)

# Streamlit App Configuration
st.set_page_config(page_title="Customer Segmentation App", page_icon="🛒")
st.title("🛒 Customer Segmentation Prediction")
st.write("Predict the customer segment for a new customer based on their demographic and behavioral data.")

# Input Fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
city = st.selectbox("City", ["New York", "Los Angeles", "Chicago", "San Francisco", "Miami", "Houston"])
membership_type = st.selectbox("Membership Type", ["Gold", "Silver", "Bronze"])
items_purchased = st.number_input("Items Purchased", min_value=1, max_value=500, value=10)
purchase_amount = st.number_input("Purchase Amount ($)", min_value=1, max_value=5000, value=500)
discount_applied = st.selectbox("Discount Applied", ["Yes", "No"])
satisfaction_level = st.selectbox("Satisfaction Level", ["Satisfied", "Neutral", "Unsatisfied"])

# Prepare the input data
new_customer = pd.DataFrame([[gender, age, city, membership_type, items_purchased, purchase_amount, discount_applied == "Yes", satisfaction_level]],
                            columns=["Gender", "Age", "City", "Membership Type", "Items Purchased", "Purchase Amount", "Discount Applied", "Satisfaction Level"])

# Apply the label encoders for the ordered categories
label_cols = ['Membership Type', 'Satisfaction Level']
for col in label_cols:
    le = label_encoders[col]
    new_customer[col] = le.transform(new_customer[col])

# Convert Boolean column to integer
new_customer['Discount Applied'] = new_customer['Discount Applied'].astype(int)

# One-hot encode the nominal categories
new_customer_encoded = pd.get_dummies(new_customer, columns=['Gender', 'City'], drop_first=True)

# Ensure the columns match with the training data
missing_cols = set(one_hot_columns) - set(new_customer_encoded.columns)
for col in missing_cols:
    new_customer_encoded[col] = 0  # Add missing columns as zeros

# Reorder the columns to match the training set
new_customer_encoded = new_customer_encoded[one_hot_columns]

# Scale the data
new_customer_scaled = scaler.transform(new_customer_encoded)

# Predict the cluster when the "Predict" button is pressed
if st.button("Predict Customer Type"):
    cluster_prediction = kmeans_model.predict(new_customer_scaled)

    # Show the predicted cluster
    st.write(f"### 🚀 Predicted Customer Segment: **Customer Type-{cluster_prediction[0]+1}**")

    # Show some characteristics of the predicted cluster
    st.write(f"### Customer Characteristics:")

    # Define some cluster descriptions (based on your analysis of cluster centers)
    cluster_descriptions = {
        0: "Customer Type-1: Primarily young customers who purchase small amounts and rarely use discounts. They have a neutral satisfaction level.",
        1: "Customer Type-2: Older customers who make larger purchases, are more likely to use discounts, and are generally satisfied.",
        2: "Customer Type-3: Young customers with high purchase frequency but low satisfaction, tend to buy many items at a time.",
        3: "Customer Type-4: Customers with high satisfaction, mostly from the 'Gold' membership tier, who purchase in moderate amounts.",
        4: "Customer Type-5: Customers who make purchases infrequently, rarely use discounts, and tend to be in the 'Bronze' membership group.",
        5: "Customer Type-6: Customers with frequent small purchases, but use discounts often and show moderate satisfaction."
    }

    # Show description based on the predicted cluster
    cluster_index = cluster_prediction[0]
    st.write(cluster_descriptions.get(cluster_index, "No description available for this cluster."))

