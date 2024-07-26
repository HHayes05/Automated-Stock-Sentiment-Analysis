from Data_Collector import collect_and_preprocess_data

def analyze_stock(ticker):
    data = collect_and_preprocess_data(ticker)
    if data:
        print(f"\nAnalysis for {ticker}:")
        for article in data[:5]:  # Print first 5 articles
            print(f"\nTitle: {article['title']}")
            print(f"Sentiment: {article['sentiment']}")
            print(f"Source: {article['source']}")
            print(f"URL: {article['url']}")
    else:
        print("No relevant data found.")

if __name__ == "__main__":
    ticker = input("Enter a stock ticker: ").upper()
    analyze_stock(ticker)