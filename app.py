from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load dataset
data = pd.read_csv("recipeex001.csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['Ingredients'])

# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[['TotalTimeInMins']])  # Fixed trailing comma

# Combine Features
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

def recommend_recipes(TotalTimeInMins,Ingredients):
     input_features_scaled = scaler.transform(np.array([[TotalTimeInMins]]))
     input_ingredients_transformed = vectorizer.transform([Ingredients])
     input_combined=np.hstack([input_features_scaled,input_ingredients_transformed.toarray()])
     distances, indices = knn.kneighbors(input_combined)
     recommendations = data.iloc[indices[0]]
    
     return recommendations[['Name', 'Ingredients', 'img_url']].head(5)

# Function to truncate product name
def truncate(text, length):
    if len(text)>length:
        return text[:length] +"....."
    else:
        return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
            TotalTimeInMins = float(request.form['TotalTimeInMins'])
            Ingredients = request.form['Ingredients']

            input_features = [TotalTimeInMins, Ingredients]
            recommendations = recommend_recipes(TotalTimeInMins,Ingredients)

            return render_template('index.html', recommendations=recommendations.to_dict(orient='records'), truncate= truncate)

    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)
