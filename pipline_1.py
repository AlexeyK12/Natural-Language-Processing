import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import joblib

def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('/', ' ')
    return text

train['description'] = train['description'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=10000)
model = LGBMClassifier()

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(train['description'], train['is_bad'], test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)


y_pred = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC_AUC: {roc_auc}')

joblib.dump(pipeline, 'text_class.joblib')