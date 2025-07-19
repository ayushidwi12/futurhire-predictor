import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import joblib
import os

# Load dataset
df = pd.read_csv("dataset.csv")
print("Columns in dataset:", df.columns.tolist())

# Combine text features
df['combined_text'] = df['resume_text'].fillna('') + " " + df['required_skills'].fillna('')

# Encode target variable
label_encoder = LabelEncoder()
df['company_encoded'] = label_encoder.fit_transform(df['company_name'])

# TF-IDF on text
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df['combined_text'])

# Numerical features: CGPA and Aptitude Score
num_features = ['cgpa', 'aptitude_score']
scaler = StandardScaler()
X_num = scaler.fit_transform(df[num_features])
X_num = csr_matrix(X_num)  # Make sparse for hstack

# One-hot encode college tier
college_tier_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_array = college_tier_ohe.fit_transform(df[['college_tier']])
X_cat = csr_matrix(X_cat_array)

# Combine all features
X_combined = hstack([X_text, X_num, X_cat])
y = df['company_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/hiring_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")
joblib.dump(scaler, "model/feature_scaler.pkl")
joblib.dump(college_tier_ohe, "model/college_tier_encoder.pkl")

print("✅ Model training complete. All files saved in 'model/'")
