# 🎬 MovieMood – Detailed Project Documentation

Website Link : https://movie-recommendation-system-using-nlp-tf-idf-fastapi-eqknjxeel.streamlit.app/?view=home

## 📌 Introduction

MovieMood is a **content-based movie recommendation system** built using **Natural Language Processing (NLP)** techniques. The system analyzes movie metadata such as **overview, genres, and taglines** to recommend similar movies.

Unlike collaborative filtering, this system does not depend on user history. Instead, it focuses entirely on **content similarity**, making it effective even for new users.

---

## 🎯 Objectives

* Build an intelligent movie recommendation system using NLP
* Implement **TF-IDF vectorization** for text representation
* Use **cosine similarity** to measure similarity between movies
* Develop a scalable architecture using **FastAPI (backend)** and **Streamlit (frontend)**
* Integrate real-time movie data using **TMDB API**

---

## ⚙️ System Overview

The system follows a **three-layer architecture**:

```
User → Streamlit Frontend → FastAPI Backend → ML Model + TMDB API
```

### Key Components:

* **Frontend:** Streamlit (User Interface)
* **Backend:** FastAPI (API + ML logic)
* **Model:** TF-IDF + Cosine Similarity
* **Data Source:** TMDB API + Movie Dataset

---

<img width="1024" height="1536" alt="System Diagram" src="https://github.com/user-attachments/assets/69ac07c4-5bee-4874-90d7-ae339797242c" />

## 📂 Dataset Description

* Dataset: `movies_metadata.csv`
* Contains:

  * Movie titles
  * Overviews (descriptions)
  * Genres
  * Taglines
  * Ratings and popularity

---

## 🔬 Machine Learning Pipeline

### 1. Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv('movies_metadata.csv')
```

---

### 2. Data Exploration

Performed to understand dataset structure:

* `df.head()` → View sample data
* `df.info()` → Data types and memory usage
* `df.columns` → Column names
* `df.isnull().sum()` → Missing values

---

### 3. Data Cleaning

Steps:

* Remove duplicate rows
* Select relevant columns:

  * title
  * overview
  * genres
  * tagline
  * vote_average
  * popularity

```python
df = df.drop_duplicates().reset_index(drop=True)
```

---

### 4. Handling Missing Values

* Remove rows with missing titles
* Fill missing text fields with empty strings

```python
df = df.dropna(subset=['title'])
df['overview'] = df['overview'].fillna('')
df['tagline'] = df['tagline'].fillna('')
```

---

### 5. Genre Extraction

Genres are stored as JSON-like strings. Using AST, they are converted into readable format.

Example:

```
[{'id': 28, 'name': 'Action'}]
```

Converted to:

```
Action
```

---

### 6. Feature Engineering

Combine text features into a single column:

```python
df['tags'] = df['overview'] + " " + df['genres'] + " " + df['tagline']
```

---

### 7. Text Preprocessing

Steps:

* Convert text to lowercase
* Remove punctuation
* Remove stopwords (using NLTK)

```python
import nltk
nltk.download('stopwords')
```

---

### 8. TF-IDF Vectorization

Convert text into numerical vectors:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1,2),
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(df['tags'])
```

---

### 9. Similarity Calculation

Compute similarity using cosine similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)
```

---

### 10. Recommendation Function

```python
def recommend(title, top_n=5):
    idx = indices[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return [df.iloc[i[0]].title for i in scores]
```

---

### 11. Model Saving

Save processed data using pickle:

```python
import pickle

pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))
pickle.dump(indices, open('indices.pkl', 'wb'))
```

---

## ⚡ FastAPI Backend

### Purpose

The backend handles:

* Recommendation logic
* TMDB API communication
* Data processing

---

### Key Features

* High performance (async support)
* Scalable architecture
* Secure API key handling

---

### Important Components

#### 1. Load Environment Variables

```python
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
```

---

#### 2. Load Model Files

* `df.pkl`
* `tfidf.pkl`
* `tfidf_matrix.pkl`
* `indices.pkl`

Loaded at startup to improve performance.

---

#### 3. API Endpoints

| Endpoint           | Description                 |
| ------------------ | --------------------------- |
| `/health`          | Check API status            |
| `/home`            | Trending movies             |
| `/tmdb/search`     | Search movies               |
| `/movie/id/{id}`   | Movie details               |
| `/recommend/tfidf` | NLP recommendations         |
| `/recommend/genre` | Genre-based recommendations |
| `/movie/search`    | Combined response           |

---

### TMDB Integration

Used to fetch:

* Posters
* Movie details
* Genres
* Ratings

---

## 🎨 Streamlit Frontend

### Features

* Movie search with autocomplete
* Movie details page
* Recommendation display
* Trending movie feed

---

### UI Components

* Search bar
* Poster grid layout
* Sidebar navigation
* Details view

---

### Key Functionalities

#### 1. API Communication

```python
def api_get_json(path, params=None):
```

* Uses caching
* Reduces API calls

---

#### 2. Navigation

* Home page
* Movie details page

---

#### 3. Recommendation Display

* TF-IDF recommendations
* Genre-based recommendations

---

## 🌐 Deployment Architecture

```
GitHub → Render (Backend) → Streamlit Cloud (Frontend)
```

---

### Deployment Platforms

| Component | Platform        |
| --------- | --------------- |
| Backend   | Render          |
| Frontend  | Streamlit Cloud |
| Code      | GitHub          |

---

## 🔐 Security

* API keys stored in `.env`
* Not exposed in frontend
* Secure backend communication

---

## 🔄 Data Flow

Example:

```
User searches "Interstellar"
↓
Frontend sends request
↓
Backend fetches TMDB data
↓
TF-IDF model computes similarity
↓
Results returned
↓
Frontend displays movies
```

---

## 🧠 Advantages of the System

* No need for user history
* Fast recommendations
* Scalable API architecture
* Real-time movie data integration

---

## 📈 Future Improvements

* Deep learning embeddings (BERT)
* Collaborative filtering
* User authentication
* Watchlist feature
* Redis caching
* Personalized recommendations

---

## 🎯 Conclusion

MovieMood successfully demonstrates how **Natural Language Processing** can be applied to build an intelligent recommendation system.

It combines:

* **Machine Learning**
* **Web Development**
* **API Integration**

to deliver a **scalable, real-world application**.

---

## ⭐ Final Output

* AI-powered movie recommendations
* Clean and interactive UI
* Real-time movie data
* Modular and scalable architecture

---

**End of Documentation**
