import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = 'recipeex001.csv'
recipe_df = pd.read_csv('recipeex001.csv')

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(recipe_df['Ingredients'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(recipe_df[['TotalTimeInMins']])

# Combine Features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

# Function to Recommend Recipes
def recommend_recipes(input_features):
    # Extract numerical values and convert to DataFrame with correct column names
    input_numerical = pd.DataFrame([input_features[:1]], columns=['TotalTimeInMins'])
    input_features_scaled = scaler.transform(input_numerical)

    # Transform ingredient text
    input_ingredients_transformed = vectorizer.transform([input_features[1]])

    # Combine numerical and ingredient features
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])

    # Get nearest neighbors
    distances, indices = knn.kneighbors(input_combined)
    recommendations = recipe_df.iloc[indices[0]]
    
    return recommendations[['Name', 'Ingredients', 'img_url']]

# Example Input
input_features = [10, 'Rice flour, jaggery, ghee, vegetable oil, elachi']
recommendations = recommend_recipes(input_features)
print(recommendations)
