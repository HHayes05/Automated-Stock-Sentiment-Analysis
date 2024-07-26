import sys
from Data_Collector import collect_and_preprocess_data, clean_text, assign_sentiment
from Predictor import predict_sentiment
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def collect_data_for_ticker(ticker):
    data = collect_and_preprocess_data(ticker)

    if not data:
        print(f"No articles found for {ticker}")
        return None

    df = pd.DataFrame(data)

    # Add ML sentiment prediction
    df['ml_sentiment'], df['confidence'] = zip(*df['title'].apply(predict_sentiment))

    return df

def generate_report(df):
    print("\nSentiment Analysis Report")
    print("========================")
    print(f"Ticker: {df['ticker'].iloc[0]}")
    print(f"Number of articles analyzed: {len(df)}")
    print("\nSentiment Distribution:")
    print(df['ml_sentiment'].value_counts(normalize=True).to_string())
    print("\nTop 5 Most Recent Articles:")

    # Convert 'date' to datetime and sort
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_sorted = df.sort_values('date', ascending=False, na_position='last')

    for _, row in df_sorted.head().iterrows():
        print(f"\nDate: {row['date']}")
        print(f"Title: {row['title']}")
        print(f"Sentiment: {row['ml_sentiment']} (Confidence: {row['confidence']:.2f})")
        print(f"Source: {row['source']}")


def plot_sentiment_distribution(df):
    sentiment_counts = df['ml_sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar')
    plt.title(f"Sentiment Distribution for {df['ticker'].iloc[0]}")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_confidence_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=20, edgecolor='black')
    plt.title(f"Confidence Distribution for {df['ticker'].iloc[0]}")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_sentiment_vs_confidence(df):
    fig, ax = plt.subplots(figsize=(12, 6))

    sentiments = df['ml_sentiment'].unique()
    colors = ['#ff9999', '#66b3ff']

    for sentiment, color in zip(sentiments, colors):
        subset = df[df['ml_sentiment'] == sentiment]
        ax.scatter(subset['confidence'], np.random.normal(0, 0.05, size=len(subset)),
                   alpha=0.6, c=color, label=sentiment)

    ax.set_title(f"Sentiment vs Confidence for {df['ticker'].iloc[0]}")
    ax.set_xlabel("Confidence")
    ax.set_yticks([])
    ax.set_ylabel("Sentiment")

    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    while True:
        ticker = input("Enter a stock ticker (or 'quit' to exit): ").upper()
        if ticker == 'QUIT':
            break

        df = collect_data_for_ticker(ticker)
        if df is not None and not df.empty:
            generate_report(df)
            plot_sentiment_distribution(df)
            plot_confidence_distribution(df)
            plot_sentiment_vs_confidence(df)
        else:
            print("No relevant data found for the given ticker. Please try another.")


if __name__ == "__main__":
    main()