import streamlit as st
import numpy as np
import pickle
import random
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt")

def load_artifacts():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("food_rating_model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_artifacts()

# Lightweight text generator using Markov-like chaining
def lightweight_text_generator(product_ratings):
    phrases = [
        ["Product", "{best_product}", "is highly praised for", "nutritional value."],
        ["Considering its", "shelf life", "and", "sustainability,", "Product", "{best_product}", "stands out."],
        ["Among the options,", "Product", "{best_product}", "excels", "in customer satisfaction."],
        ["For affordability and", "health benefits,", "Product", "{best_product}", "is a great choice."],
    ]
    random.shuffle(phrases)
    sentence = " ".join(random.choice(phrases))
    additional_details = (
        f"The ratings are: Product 1 - {round(product_ratings[0], 2)}, Product 2 - {round(product_ratings[1], 2)}, "
        f"Product 3 - {round(product_ratings[2], 2)}."
    )
    return additional_details + " " + sentence.format(best_product=np.argmax(product_ratings) + 1)

st.title("Food Rating Comparison and Recommendation")
st.write("Compare ratings of three food products and get a detailed recommendation.")

def get_product_inputs(product_id, default_values):
    with st.expander(f"Product {product_id} Details"):
        st.write(f"Calories (Product {product_id}): {default_values[0]} kcal")
        st.write(f"Fats (g) (Product {product_id}): {default_values[1]} g")
        st.write(f"Proteins (g) (Product {product_id}): {default_values[2]} g")
        st.write(f"Carbs (g) (Product {product_id}): {default_values[3]} g")
        st.write(f"Temperature (°C) (Product {product_id}): {default_values[4]} °C")
        st.write(f"Humidity (%) (Product {product_id}): {default_values[5]} %")
        st.write(f"Shelf Life (days) (Product {product_id}): {default_values[6]} days")
        st.write(f"Cost ($) (Product {product_id}): ${default_values[7]}")
        st.write(f"Sustainability Factor (Product {product_id}): {default_values[8]}")
        st.write(f"Potassium (mg) (Product {product_id}): {default_values[9]} mg")
        st.write(f"Sodium (mg) (Product {product_id}): {default_values[10]} mg")
    return default_values

default_values_1 = [90, 2.2, 2.5, 17, 190, 65, 5, 35, 5, 40, 150]
default_values_2 = [92, 2.5, 2.3, 16, 190, 60, 5, 40, 7, 30, 150]
default_values_3 = [10, 1.9, 2.1, 16, 180, 55, 4, 45, 5, 25, 225]

st.header("Product 1")
product_1 = get_product_inputs(1, default_values_1)

st.header("Product 2")
product_2 = get_product_inputs(2, default_values_2)

st.header("Product 3")
product_3 = get_product_inputs(3, default_values_3)

if st.button("Compare Ratings and Generate Recommendation"):

    input_data = np.array([product_1, product_2, product_3])
    input_scaled = scaler.transform(input_data)
    ratings = model.predict(input_scaled)

    st.subheader("Product Ratings and Details")
    for idx, (rating, product) in enumerate(zip(ratings, [product_1, product_2, product_3]), start=1):
        st.write(f"*Product {idx} Composition:*")
        st.write(f"Calories: {product[0]} kcal, Fats: {product[1]} g, Proteins: {product[2]} g, "
                 f"Carbs: {product[3]} g, Temperature: {product[4]} °C, Humidity: {product[5]} %, "
                 f"Shelf Life: {product[6]} days, Cost: ${product[7]}, Sustainability Factor: {product[8]}, "
                 f"Potassium: {product[9]} mg, Sodium: {product[10]} mg")
        st.write(f"*Product {idx} Rating:* {round(rating, 2)}")

    best_product = np.argmax(ratings) + 1
    st.success(f"Recommended Product: Product {best_product}")

    detailed_recommendation = lightweight_text_generator(ratings)
    st.subheader("Recommendation Summary")
    st.text_area("Detailed Recommendation", value=detailed_recommendation, height=200, disabled=True)

st.markdown("---")
st.write("Powered by Streamlit and Python")