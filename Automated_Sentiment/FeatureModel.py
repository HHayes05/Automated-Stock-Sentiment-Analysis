import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import re

# Load the data
df = pd.read_csv('tech_stock_sentiment_data.csv')

# Convert sentiment to binary (0 for negative/neutral, 1 for positive)
df['sentiment_binary'] = (df['sentiment'] == 'positive').astype(int)

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Simple tokenization
    tokens = text.split()
    return ' '.join(tokens)

# Combine cleaned_title and cleaned_content, then apply preprocessing
df['processed_text'] = (df['cleaned_title'] + ' ' + df['cleaned_content']).apply(preprocess_text)

# Feature Extraction
def extract_features(df):
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_features = tfidf.fit_transform(df['processed_text'])

    # Create financial domain-specific features
    df['title_length'] = df['title'].apply(len)
    df['contains_number'] = df['title'].str.contains('\d').astype(int)
    df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
    df['unique_word_count'] = df['processed_text'].apply(lambda x: len(set(x.split())))

    # Sentiment score based on predefined financial terms
    positive_terms = ['surge', 'gain', 'uptick', 'bullish', 'outperform']
    negative_terms = ['slump', 'decline', 'downturn', 'bearish', 'underperform']

    df['positive_score'] = df['processed_text'].apply(lambda x: sum(term in x for term in positive_terms))
    df['negative_score'] = df['processed_text'].apply(lambda x: sum(term in x for term in negative_terms))

    # Combine TF-IDF features with domain-specific features
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)

    features = pd.concat([tfidf_df, df[
        ['title_length', 'contains_number', 'word_count', 'unique_word_count', 'positive_score', 'negative_score']]],
                         axis=1)

    # Scale the features
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    return features, tfidf, scaler

# Model Development
def train_and_evaluate_models(X, y, tfidf_vectorizer, scaler):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(probability=True, random_state=42)
    }

    results = {}

    for name, model in models.items():
        if name == 'Naive Bayes':
            # For Naive Bayes, use unscaled features
            X_train_model = tfidf_vectorizer.transform(df.loc[X_train.index, 'processed_text'])
            X_test_model = tfidf_vectorizer.transform(df.loc[X_test.index, 'processed_text'])
        else:
            # For other models, use scaled features
            X_train_model = X_train
            X_test_model = X_test

        # Train model
        model.fit(X_train_model, y_train)

        # Predictions
        y_pred = model.predict(X_test_model)
        y_pred_proba = model.predict_proba(X_test_model)[:, 1]

        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=1)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation (use unscaled features for Naive Bayes)
        if name == 'Naive Bayes':
            cv_scores = cross_val_score(model, tfidf_vectorizer.transform(df['processed_text']), y, cv=5)
        else:
            cv_scores = cross_val_score(model, X, y, cv=5)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'auc': auc,
            'cv_scores': cv_scores
        }

    return results

# Main execution
if __name__ == "__main__":
    # Extract features
    X, tfidf_vectorizer, scaler = extract_features(df)
    y = df['sentiment_binary']

    # Train and evaluate models
    results = train_and_evaluate_models(X, y, tfidf_vectorizer, scaler)

    # Print results
    for name, result in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"AUC: {result['auc']:.4f}")
        print(f"Cross-validation scores: {result['cv_scores'].mean():.4f} (+/- {result['cv_scores'].std() * 2:.4f})")
        print("\nClassification Report:")
        print(result['report'])
        print("\nConfusion Matrix:")
        print(result['confusion_matrix'])

    # Select the best model (based on accuracy)
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']

    # Save the best model, vectorizer, and scaler
    joblib.dump(best_model, 'best_sentiment_model.joblib')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')

    print(f"\nBest model ({best_model_name}), vectorizer, and scaler saved as 'best_sentiment_model.joblib', 'tfidf_vectorizer.joblib', and 'feature_scaler.joblib'")