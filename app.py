from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

# Load the Netflix dataset
def load_data():
    try:
        df = pd.read_csv(r'C:\Users\DELL\NETFLIX\net\netflix_titles.csv')
        return df[['title', 'description', 'listed_in', 'cast', 'director', 'rating', 'duration', 'type', 'country', 'release_year']]
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return None

# Load the data at startup
df = load_data()

# Check if df is loaded correctly
if df is None:
    print("Error: Dataframe is not loaded. Exiting the application.")
    exit(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_history = request.form['user_history'].split(',')
    user_history = [title.strip() for title in user_history]

    recommendations, naive_bayes_accuracy = get_recommendations(user_history, df)

    if not recommendations:
        recommendations = ["No recommendations found. Please check your input titles."]

    return render_template('results.html', recommendations=recommendations, 
                           naive_bayes_accuracy=naive_bayes_accuracy)

@app.route('/predict_country_content', methods=['POST'])
def predict_country_content():
    country_input = request.form['country']
    predicted_content = get_country_specific_prediction(country_input)
    return render_template('country_content_results.html', country=country_input, predicted_content=predicted_content)

@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    title_input = request.form['title']
    predicted_rating, actual_rating, accuracy = get_rating_prediction(title_input)
    return render_template('rating_results.html', title=title_input, 
                           predicted_rating=predicted_rating, 
                           actual_rating=actual_rating, accuracy=accuracy)

@app.route('/predict_genre', methods=['POST'])
def predict_genre():
    title_input = request.form['title']
    predicted_genres = get_genre_prediction(title_input)
    return render_template('genre_results.html', title=title_input, predicted_genres=predicted_genres)

@app.route('/analyze_trends', methods=['GET'])
def analyze_trends():
    trend_graph = generate_trend_analysis()
    future_predictions = generate_future_predictions()
    return render_template('trend_analysis.html', trend_graph=trend_graph, future_predictions=future_predictions)

# Recommendation logic
def get_recommendations(user_history, df):
    user_history = [title.strip().lower() for title in user_history]
    
    df.fillna('', inplace=True)
    df['combined_features'] = (df['description'] + ' ' + df['listed_in'] + ' ' + df['cast'] + ' ' + df['director'])
    
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
    recommended_titles = []

    for title in user_history:
        if title in indices:
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:16]  # Get top 15 recommendations
            
            movie_indices = [i[0] for i in sim_scores]
            recommended_titles.extend(df['title'].iloc[movie_indices].tolist())
        else:
            print(f"'{title}' not found in dataset.")

    naive_bayes_accuracy = train_naive_bayes(df)
    return list(set(recommended_titles)), naive_bayes_accuracy

# Function to get country-specific content prediction
def get_country_specific_prediction(country):
    country_df = df[df['country'].str.contains(country, na=False)]

    if country_df.empty:
        return "No data available for this country."

    # Analyze the trends based on genres
    genre_counts = country_df['listed_in'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
    
    # Return top 5 genres as prediction
    return genre_counts.head(5).index.tolist()

# Function to get rating predictions
def get_rating_prediction(title):
    df.fillna('', inplace=True)

    # Drop rows where rating is NaN
    df.dropna(subset=['rating'], inplace=True)

    if title not in df['title'].values:
        return "Title not found.", None, None  # Return if title not found

    # Create a mapping of ratings to numerical values
    rating_mapping = {
        'G': 1,
        'PG': 2,
        'PG-13': 3,
        'R': 4,
        'TV-MA': 5,
        'TV-14': 6,
        'TV-G': 7,
        'TV-PG': 8,
        'NR': 0  # Not Rated
    }

    # Map ratings to numeric values for training
    df['numeric_rating'] = df['rating'].map(rating_mapping)

    # Check for NaN values after mapping
    df.dropna(subset=['numeric_rating'], inplace=True)

    # Predict ratings using Linear Regression
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    model = LinearRegression()
    model.fit(tfidf_matrix, df['numeric_rating'])  # Train on numeric ratings

    title_vector = tfidf.transform([df[df['title'] == title]['description'].values[0]])
    predicted_numeric_rating = model.predict(title_vector).item()

    # Reverse mapping to find the closest rating
    actual_rating = df[df['title'] == title]['rating'].values[0]
    predicted_rating = list(rating_mapping.keys())[list(rating_mapping.values()).index(round(predicted_numeric_rating))]

    accuracy = (predicted_rating == actual_rating)

    return predicted_rating, actual_rating, accuracy

# Genre Prediction Logic
def get_genre_prediction(title):
    if title not in df['title'].values:
        return "Title not found."

    # Use the 'listed_in' column (genre) for prediction
    predicted_genres = df[df['title'] == title]['listed_in'].values[0]
    return predicted_genres

# Function to train Naive Bayes model and get accuracy
def train_naive_bayes(df):
    df['is_popular'] = df['rating'].apply(lambda x: 1 if x in ['PG', 'PG-13', 'R'] else 0)
    X = df['description']
    y = df['is_popular']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return accuracy

# Generate trend analysis graph
def generate_trend_analysis():
    # Prepare data for trend analysis
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
    trends = df.groupby(['release_year', 'type']).size().unstack().fillna(0)

    # Plotting the trends
    plt.figure(figsize=(12, 6))
    trends.plot(kind='bar', stacked=True)
    plt.title('Trends in Netflix Shows/Movies Over the Years')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Titles')
    plt.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot to a bytes buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Function to generate future predictions based on trends
def generate_future_predictions():
    # Sample future prediction logic (this can be improved with a proper forecasting model)
    future_years = list(range(2025, 2031))
    # For simplicity, assume a linear increase in content production based on the last known values
    last_year = df['release_year'].max()
    last_values = df[df['release_year'] == last_year]['type'].value_counts()
    future_values = {type_: last_values.get(type_, 0) + (i * 5) for i, type_ in enumerate(last_values.index, start=1)}

    return future_values

if __name__ == '__main__':
    app.run(debug=True)
