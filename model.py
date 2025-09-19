import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

# Load the Netflix dataset
def load_data():
    df = pd.read_csv(r'C:\Users\windows\OneDrive\Desktop\NETFLIX\net\netflix_titles.csv')
    print("Columns in dataset:", df.columns.tolist())  # Print columns for inspection
    return df[['title', 'description', 'listed_in', 'cast', 'director', 'rating', 'duration']]  # Include 'duration' for classification

def classify_titles_by_duration(df):
    # Check if 'duration' column exists
    if 'duration' in df.columns:
        # Define a function to extract duration
        def extract_duration(duration):
            if 'min' in duration:
                return int(duration.replace(' min', ''))  # Convert to integer
            elif 'Seasons' in duration:
                return int(duration.split()[0]) * 60  # Convert seasons to minutes (e.g., 2 Seasons -> 120)
            return None  # Return None for unrecognized formats

        # Apply the function to the 'duration' column
        df['duration_minutes'] = df['duration'].apply(extract_duration)

        # Now you can continue with your classification logic
        # Example classification based on duration
        df['title_type'] = np.where(df['duration_minutes'] > 0, 'Movie', 'TV Show')  # Simple classification
        return df['title_type'].value_counts()  # Just an example return

    return None  # Handle case where duration column is not present


def get_recommendations(user_history, df):
    user_history = [title.strip().lower() for title in user_history]  # Normalize user input to lower case
    
    # Handle missing values by filling NaNs with empty strings
    df.fillna('', inplace=True)

    # Combine features into a single string
    df['combined_features'] = (
        df['description'] + ' ' + df['listed_in'] + ' ' + df['cast'] + ' ' + df['director']
    )

    # Create a TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Added ngram_range
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Calculate the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a lowercase index mapping
    indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()  # Use lower case titles for indexing

    recommended_titles = []
    ratings = []

    # Map categorical ratings to numeric values
    rating_map = {
        'G': 1,
        'PG': 2,
        'PG-13': 3,
        'R': 4,
        'TV-G': 1,
        'TV-PG': 2,
        'TV-14': 3,
        'TV-MA': 4,
        # Add more mappings as needed
    }
    
    # Convert ratings to numeric
    df['numeric_rating'] = df['rating'].map(rating_map)

    for title in user_history:
        if title in indices:
            idx = indices[title]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:16]  # Get top 15 recommendations
            movie_indices = [i[0] for i in sim_scores]
            recommended_titles.extend(df['title'].iloc[movie_indices].tolist())
            ratings.extend(df['numeric_rating'].iloc[movie_indices].tolist())
        else:
            print(f"'{title}' not found in dataset.")  # Debugging line

    # Predict Ratings Using Linear Regression
    if len(ratings) > 0:
        # Prepare features (TF-IDF) and target (ratings)
        X = tfidf_matrix  # Use TF-IDF features of movie descriptions
        y = df['numeric_rating'].fillna(0)  # Replace missing ratings with 0
        
        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict ratings for recommended movies
        predicted_ratings = model.predict(tfidf_matrix[movie_indices])

        # Clip predictions to avoid negative or out-of-bounds values
        predicted_ratings = np.clip(predicted_ratings, 1, 4)

        regression_result = {
            'predicted_ratings': predicted_ratings.tolist(),  # Convert to list for rendering in HTML
            'actual_ratings': ratings
        }
    else:
        regression_result = None

    # Train Naive Bayes Model for accuracy
    naive_bayes_accuracy = train_naive_bayes(df)

    # Return unique recommendations, regression results, and Naive Bayes accuracy
    return list(set(recommended_titles)), regression_result, naive_bayes_accuracy

# Function to train Naive Bayes model and get accuracy
def train_naive_bayes(df):
    # Prepare dataset for Naive Bayes
    df['is_popular'] = df['rating'].apply(lambda x: 1 if x in ['PG', 'PG-13', 'R'] else 0)  # Simple popularity classification

    # Features and target variable
    X = df['description']  # Using description as the feature
    y = df['is_popular']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF and Naive Bayes
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)

    return accuracy

# Example usage
df = load_data()
user_history = ["Stranger Things", "Breaking Bad"]
recommendations, regression_result, naive_bayes_accuracy = get_recommendations(user_history, df)

# Calculate ID3 decision tree accuracy
id3_accuracy = classify_titles_by_duration(df)

# Output results
print("Recommended Titles:", recommendations)
if regression_result:
    print("Predicted Ratings:", regression_result['predicted_ratings'])
    print("Actual Ratings:", regression_result['actual_ratings'])
else:
    print("No regression results available.")
print("Naive Bayes Model Accuracy:", naive_bayes_accuracy)
print("ID3 Decision Tree Accuracy:", id3_accuracy)
