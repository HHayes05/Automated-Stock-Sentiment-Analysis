import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import learning_curve
import re

# Load the best model, vectorizer, and scaler
best_model = joblib.load('best_sentiment_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('feature_scaler.joblib')

# Load the data
df = pd.read_csv('tech_stock_sentiment_data.csv')

# Preprocess the data
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

# Combine cleaned_title and cleaned_content, then apply preprocessing
df['processed_text'] = (df['cleaned_title'] + ' ' + df['cleaned_content']).apply(preprocess_text)

# Extract features
tfidf_features = tfidf_vectorizer.transform(df['processed_text'])
X_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Create additional features
df['title_length'] = df['title'].apply(len)
df['contains_number'] = df['title'].str.contains('\d').astype(int)
df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
df['unique_word_count'] = df['processed_text'].apply(lambda x: len(set(x.split())))

# Sentiment score based on predefined financial terms
positive_terms = ['surge', 'gain', 'uptick', 'bullish', 'outperform']
negative_terms = ['slump', 'decline', 'downturn', 'bearish', 'underperform']

df['positive_score'] = df['processed_text'].apply(lambda x: sum(term in x for term in positive_terms))
df['negative_score'] = df['processed_text'].apply(lambda x: sum(term in x for term in negative_terms))

# Combine TF-IDF features with additional features
X = pd.concat([X_tfidf, df[['title_length', 'contains_number', 'word_count', 'unique_word_count', 'positive_score', 'negative_score']]], axis=1)

# Scale features
X = pd.DataFrame(scaler.transform(X), columns=X.columns)

y = (df['sentiment'] == 'positive').astype(int)

# Make predictions
y_pred = best_model.predict(X)
y_pred_proba = best_model.predict_proba(X)[:, 1]

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative/Neutral', 'Positive'],
                yticklabels=['Negative/Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


# Function to plot learning curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.show()


# Evaluate the model
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Plot confusion matrix
plot_confusion_matrix(y, y_pred)

# Plot ROC curve
plot_roc_curve(y, y_pred_proba)

# Plot Precision-Recall curve
plot_precision_recall_curve(y, y_pred_proba)

# Plot learning curve
plot_learning_curve(best_model, X, y)

# Feature importance (if the model supports it)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.show()
elif hasattr(best_model, 'coef_'):
    # For models like Logistic Regression
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(best_model.coef_[0])
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.show()
else:
    print("Feature importance visualization not applicable for this model.")

# Error analysis
errors = y != y_pred
error_df = df[errors].copy()
error_df['predicted'] = y_pred[errors]
error_df['true'] = y[errors]
print("\nSample of Misclassified Instances:")
print(error_df[['title', 'sentiment', 'predicted', 'true']].sample(min(10, len(error_df))))

# Class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Class Distribution')
plt.show()

print("\nModel Evaluation Complete")