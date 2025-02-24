import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MovieRecommender:
    """A simple content-based movie recommendation system."""
    
    def __init__(self, csv_path: str):
        """
        Load movie dataset and get everything we need for movie recommending
        
        Args:
            csv_path (str): Path to the CSV file with movie data
        """
        # Load dataset
        self.df = pd.read_csv(csv_path)
        
        # Set up TF-IDF vectorizer (removing common words and only keeping top 1000 features)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000
        )
        
        # Create TF-IDF matrix from movie descriptions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['description'].fillna(''))
    
    def get_recommendations(self, query: str, n: int = 5) -> list:
        """
        Get movie recommendations based on a text query.
        
        Args:
            query (str): description of movie type user is looking for 
            n (int): Number of recommendations to return
            
        Returns:
            list: Top N movie recommendations with titles, genres, and similarity scores
        """
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top N most similar movies
        top_indices = similarities.argsort()[-n:][::-1]
        
        # Prepare recommendations for displaying
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'title': self.df.iloc[idx]['title'],
                'similarity': similarities[idx],
                'genre': self.df.iloc[idx]['genre']
            })
        
        return recommendations

def main():
    """Main function to run the recommendation system."""
    try:
        # Initialize recommender
        recommender = MovieRecommender("TOP 100 IMDB MOVIES.csv")
        
        # Get user input
        print("\nMovie Recommendation System")
        print("---------------------------")
        query = input("Describe what kind of movie you're looking for: ")
        
        # Get and display recommendations
        recommendations = recommender.get_recommendations(query)
        
        print("\nTop Recommendations:")
        print("-------------------")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Genre: {rec['genre']}")
            print(f"   Similarity Score: {rec['similarity']:.3f}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()