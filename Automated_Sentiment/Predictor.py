import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

model = joblib.load('best_sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('feature_scaler.joblib')

positive_terms = [
    'surge', 'gain', 'uptick', 'bullish', 'outperform', 'record', 'breakthrough', 'optimistic',
    'revolutionize', 'jump', 'soar', 'rally', 'boost', 'upgrade', 'beat', 'exceed', 'positive',
    'growth', 'profit', 'success', 'innovation', 'partnership', 'launch', 'expansion', 'achievement',
    'advancement', 'advantage', 'boom', 'bright', 'brilliance', 'climb', 'commend', 'commendation',
    'conquer', 'delight', 'dominate', 'double', 'earnings', 'elevate', 'enhance', 'enthusiasm',
    'excel', 'excellence', 'explosive', 'favorable', 'flourish', 'fortune', 'gains', 'glory',
    'great', 'headway', 'high', 'improvement', 'increase', 'inflation', 'inspire', 'leap', 'leverage',
    'lift', 'lucrative', 'maximize', 'milestone', 'miracle', 'noteworthy', 'opportunity', 'peak',
    'phenomenal', 'prosper', 'prosperity', 'ramp', 'recovery', 'remarkable', 'reward', 'satisfaction',
    'shine', 'skyrocket', 'stability', 'strength', 'superior', 'support', 'surpass', 'thrill', 'triumph',
    'unbeatable', 'unprecedented', 'upward', 'victory', 'vigorous', 'well', 'win', 'zeal'
]


negative_terms = [
    'slump', 'decline', 'downturn', 'bearish', 'underperform', 'uncertainty', 'investigation',
    'drop', 'fall', 'plunge', 'sink', 'tumble', 'downgrade', 'miss', 'negative', 'loss',
    'lawsuit', 'crisis', 'recall', 'bankruptcy', 'debt', 'cut', 'risk', 'warning', 'adverse',
    'anxiety', 'backlash', 'bad', 'bailout', 'bankrupt', 'bleak', 'collapse', 'conflict', 'crash',
    'danger', 'dark', 'decline', 'deficit', 'deflation', 'degradation', 'delayed', 'deteriorate',
    'disaster', 'disappoint', 'dismal', 'distress', 'down', 'drastic', 'drought', 'economic',
    'emergency', 'endanger', 'fail', 'falter', 'fear', 'foreclosure', 'fraud', 'grim', 'halt',
    'hardship', 'harm', 'hit', 'hurdle', 'instability', 'insufficient', 'jeopardize', 'layoff',
    'liability', 'low', 'lower', 'meltdown', 'misstep', 'obstacle', 'overdue', 'panic', 'pitfall',
    'poor', 'pressure', 'problem', 'protest', 'recession', 'reduce', 'reduction', 'regret', 'repercussion',
    'risk', 'ruin', 'scandal', 'setback', 'shortfall', 'shrink', 'slack', 'slow', 'struggle', 'threat',
    'trouble', 'turmoil', 'uncertain', 'unfavorable', 'unrest', 'vulnerable', 'weak', 'weaken', 'worry'
]

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer, scaler):
        self.vectorizer = vectorizer
        self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_list = []
        for text in X:
            processed_text = preprocess_text(text)
            features = self._extract_features(processed_text, text)
            features_list.append(features)
        return pd.DataFrame(features_list, columns=self.scaler.feature_names_in_)

    def _extract_features(self, processed_text, original_text):
        tfidf_features = self.vectorizer.transform([processed_text])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=self.vectorizer.get_feature_names_out())

        additional_features = {
            'title_length': len(original_text),
            'contains_number': int(bool(re.search(r'\d', original_text))),
            'word_count': len(processed_text.split()),
            'unique_word_count': len(set(processed_text.split())),
            'positive_score': sum(term in processed_text for term in positive_terms),
            'negative_score': sum(term in processed_text for term in negative_terms)
        }

        features = pd.concat([tfidf_df, pd.DataFrame([additional_features])], axis=1)
        features = features.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
        return features.iloc[0]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

feature_extractor = FeatureExtractor(vectorizer, scaler)

def predict_sentiment(text):
    features = feature_extractor.transform([text])
    scaled_features = scaler.transform(features)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0]

    # Use keyword-based adjustment
    lower_text = text.lower()
    positive_score = sum(term in lower_text for term in positive_terms)
    negative_score = sum(term in lower_text for term in negative_terms)

    if positive_score > negative_score:
        sentiment = "positive"
        confidence = max(probability[1], 0.6)  # Ensure at least 60% confidence for positive
    elif negative_score > positive_score:
        sentiment = "negative/neutral"
        confidence = max(probability[0], 0.6)  # Ensure at least 60% confidence for negative
    else:
        sentiment = "positive" if prediction[0] == 1 else "negative/neutral"
        confidence = probability[1] if sentiment == "positive" else probability[0]

    return sentiment, confidence

if __name__ == "__main__":

    #some testing headlines to show model performance
    headlines = [
        "Tech Giant XYZ Reports Setback in Q2",
        "Market Uncertainty Leads to Sharp Decline in Tech Stocks",
        "New AI Breakthrough Could Revolutionize Healthcare Industry",
        "Tech Company ABC Faces Antitrust Investigation",
        "Investors Optimistic About Future of Electric Vehicle Market",
        "Why Nvidia Stock Is Jumping Today",
        "Pharmaceutical Company Reports Positive Results from Clinical Trial",
        "Major Bank Announces Layoffs Amid Economic Downturn",
        "Startup Secures $100 Million in Funding for Expansion",
        "Consumer Spending Drops as Inflation Concerns Grow"
    ]

    for headline in headlines:
        sentiment, confidence = predict_sentiment(headline)
        print(f"Headline: {headline}")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}")
        print()