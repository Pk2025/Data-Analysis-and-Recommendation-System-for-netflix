# Data-Analysis-and-Recommendation-System-for-netflix

# Data Analysis and Recommendation System for Netflix

## 📌 Project Overview

This project implements a data analysis and recommendation system for Netflix using machine learning techniques. It leverages user interaction data to build models that predict user preferences, enabling personalized content recommendations.

## 🧪 Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation and analysis
  - `scikit-learn` – Machine learning algorithms
  - `matplotlib`, `seaborn` – Data visualization
  - `pickle` – Model serialization
  - `Flask` – Web framework for serving the model
- **Data**: User interaction datasets (ratings, watch history, etc.)

## ⚙️ Project Structure

├── app.py # Flask application to serve the model
├── model.pkl # Trained recommendation model
├── model_genre.pkl # Genre-based recommendation model
├── model_type.pkl # Type-based recommendation model
├── requirements.txt # Project dependencies
├── static/ # Static files (CSS, JS)
└── templates/ # HTML templates for the web interface

bash
Copy code

## 🚀 Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Pk2025/Data-Analysis-and-Recommendation-System-for-netflix.git
   cd Data-Analysis-and-Recommendation-System-for-netflix
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Download Pre-trained Models:

The project utilizes large pre-trained model files. Due to GitHub's file size limitations, these models are not included in the repository. You can download them from the following Google Drive links:

Download model.pkl

Download model_genre.pkl

Download model_type.pkl

Place the downloaded files in the project root directory.

Run the Application:

bash
Copy code
python app.py
The application will be accessible at http://127.0.0.1:5000/ in your web browser.

🧠 How It Works
Data Analysis: The system analyzes user interaction data to understand viewing patterns and preferences.

Model Training: Machine learning models are trained to predict user preferences based on historical data.

Recommendation Generation: The trained models generate personalized content recommendations for users.

📄 License
This project is licensed under the MIT License
