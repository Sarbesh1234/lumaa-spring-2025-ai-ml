# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Deadline**: Sunday, Feb 23th 11:59 pm PST

---

## Movie Recommendation System

A simple content-based movie recommendeation system that suggests movies based on user text descriptions

### Dataset

This project uses a dataset from kaggle that has the top 100 IMBD movies, and contains information about top-related movies including their descriptions, genres, rating, and other metadata.  The file is already in the repository! Kaggle Link: https://www.kaggle.com/datasets/mayurkadam9833/top-100-imdb-movies?resource=download

### Dataset Structure

- 100 movies
- Features inlcude: rank, title, description, genre, rating, year

---

## Setup

### Requirements
   - Python 3.8+ 
   - pandas
   - scikit-learn
   - numpy

### Installation

1. **Clone the repository:**  
      ```bash
      git clone https://github.com/Sarbesh1234/lumaa-spring-2025-ai-ml.git
      cd movie-recommender
      ```

2. **Create and activate a virtual environment:**  
   ```bash
   python -m venv venv
   python3 -m venv venv #Or this line here depending on your version of python installed
   source venv/bin/activate  #On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

## Running the System

Run the recommendation system using:

*Make sure you are using the correct Python Interpreter in VS Code
```
python script.py
```

The system will prompt you to enter a description of the type of movie you're looking for.

### Example Usage
Input:

```
I love thrilling action movies set in space, with a comedic twist"
```

Sample Output:
```
Top Recommendations:
-------------------
1. Aliens
   Genre: ['Action', 'Adventure', 'Sci-Fi']
   Similarity Score: 0.195

2. The Lord of the Rings: The Fellowship of the Ring
   Genre: ['Action', 'Adventure', 'Drama']
   Similarity Score: 0.151

3. Interstellar
   Genre: ['Adventure', 'Drama', 'Sci-Fi']
   Similarity Score: 0.141

4. Saving Private Ryan
   Genre: ['Drama', 'War']
   Similarity Score: 0.139

5. Toy Story
   Genre: ['Animation', 'Adventure', 'Comedy']
   Similarity Score: 0.135
```

## Implementation Details

The system uses:
- TF-IDF(Term Frequency-Inverse Document Frequency) vectorization for text processing
- Cosine similarity for finding matches
- Simple command-line interface for interaction

## Salary Expectations

3000 per month

## Video Demo

https://drive.google.com/file/d/15Y88XilFwcRwcrzDATZ1ItxeY1eu1byb/view?usp=share_link

