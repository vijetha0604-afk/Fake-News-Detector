import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text.lower()

true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

true_df['label'] = 0  # Real
fake_df['label'] = 1  # Fake

df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)
df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=15000)),
    ('clf', LogisticRegression(max_iter=300))
])

print("Training...")
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds) * 100)

joblib.dump(model, 'fake_news_model.pkl')
print("✅ Saved model as fake_news_model.pkl")