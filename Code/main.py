import pandas as pd
import numpy as np
import easyocr
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text 
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, LinearRegression

# 1. Initialize Components
reader = easyocr.Reader(['en'], gpu=False) # Set to True if using Colab GPU
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text_data):
    tokens = nltk.word_tokenize(str(text_data).lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)

# 2. Load Dataset from your GitHub folder structure
# Updated path to match: /Dataset/nanoparticle_data.csv
try:
    data = pd.read_csv('Dataset/nanoparticle_data.csv')
    # Pre-process descriptions for the model
    # Assuming 'description' is the column name from your report
    data['processed_desc'] = data['description'].apply(preprocess_text)
    print("✅ Dataset loaded from /Dataset/ folder successfully!")
except FileNotFoundError:
    print("❌ Error: Ensure 'nanoparticle_data.csv' is inside the 'Dataset' folder.")

# 3. Vectorization (TF-IDF)
tfidf = text.TfidfVectorizer()
X_tfidf = tfidf.fit_transform(data['processed_desc'])

# 4. Training Models (Logic from Page 18 of your report)
# Classification: Toxic (1) vs Non-Toxic (0)
clf = LogisticRegression()
clf.fit(X_tfidf, data['label'])

# Regression: Predicting Dosage-based Toxicity Score
reg = LinearRegression()
reg.fit(X_tfidf, data['toxicity_score'])

# 5. Updated Inference Function
def analyze_label(image_path):
    result = reader.readtext(image_path, detail=0)
    raw_text = " ".join(result)
    
    clean_text = preprocess_text(raw_text)
    vec = tfidf.transform([clean_text])
    
    # Model Predictions
    prediction = clf.predict(vec)[0]
    tox_score = reg.predict(vec)[0]
    
    # KNN Similarity Search (Logic from Page 17)
    knn = NearestNeighbors(n_neighbors=1, metric='cosine')
    knn.fit(X_tfidf)
    distance, index = knn.kneighbors(vec)
    
    return {
        "Status": "TOXIC" if prediction == 1 else "SAFE",
        "Predicted Score": tox_score,
        "Closest Dataset Match": data.iloc[index[0][0]]['description'],
        "Cosine Distance": distance[0][0] # 0.000 means perfect match
    }
  
