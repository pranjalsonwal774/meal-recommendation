import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Set Streamlit page config
st.set_page_config(page_title="Recipe Recommender", layout="wide")

# Load dataset
data = pd.read_csv("recipeex001.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['Ingredients'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['TotalTimeInMins']])

# Combine Features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

# Recommend function
def recommend_recipes(TotalTimeInMins, Ingredients):
    input_features_scaled = scaler.transform(np.array([[TotalTimeInMins]]))
    input_ingredients_transformed = vectorizer.transform([Ingredients])
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data.iloc[indices[0]]
    return recommendations[['Name', 'Ingredients', 'img_url']].head(5)

# Truncate function
def truncate(text, length=100):
    return text[:length] + "..." if len(text) > length else text

# Streamlit UI
st.title("🍽️ Recipe Recommender")
st.write("Enter ingredients and desired total time to get similar recipe suggestions!")

with st.form("recipe_form"):
    total_time = st.number_input("Total Time in Minutes", min_value=1, max_value=1000, step=1)
    ingredients = st.text_area("Ingredients", placeholder="e.g., tomato, onion, garlic")

    submitted = st.form_submit_button("Get Recommendations")

if submitted:
    recommendations = recommend_recipes(total_time, ingredients)

    st.subheader("Recommended Recipes:")
    for idx, row in recommendations.iterrows():
        st.markdown(f"### {row['Name']}")
        if pd.notna(row['img_url']) and row['img_url'].startswith('http'):
            st.image(row['img_url'], width=300)
        st.markdown(f"**Ingredients**: {truncate(row['Ingredients'])}")
        st.markdown("---")
