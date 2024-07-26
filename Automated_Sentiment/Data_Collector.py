import requests
from newsapi import NewsApiClient
import time
from datetime import datetime, timedelta
import re
import pandas as pd

FINNHUB_API_KEY = 'cqg56ehr01qhf3d1bg50cqg56ehr01qhf3d1bg5g' #My Api Keys
NEWS_API_KEY = "5a086b0dba64474db00843bc56d421eb"

TOP_TECH_STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA",
    "TSM", "AVGO", "ADBE", "CSCO", "ORCL", "CRM",
    "INTC", "AMD", "IBM", "PYPL", "NFLX", "UBER", "ABNB"
]


def fetch_finnhub_news(ticker):
    current_date = datetime.now().strftime('%Y-%m-%d')
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2024-06-01&to={current_date}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data[:20]:
            articles.append({
                'date': item['datetime'],
                'title': item['headline'],
                'content': item.get('summary', ''),
                'source': 'Finnhub',
                'url': item.get('url', '')
            })

        return articles
    except Exception as e:
        print(f"Error fetching Finnhub news for {ticker}: {str(e)}")
        return []


def fetch_newsapi_news(ticker):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    try:
        query = f'"{ticker}" AND (stock OR shares OR earnings OR finance OR market)'

        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        response = newsapi.get_everything(q=query,
                                          language='en',
                                          sort_by='publishedAt',
                                          from_param=from_date,
                                          to=to_date,
                                          page_size=20)

        articles = []
        for item in response['articles']:
            title = item['title'].lower()
            content = item.get('description', '').lower()

            # Check if the article is relevant
            if (ticker.lower() in title or ticker.lower() in content) and \
                    any(keyword in title or keyword in content for keyword in
                        ['stock', 'share', 'earning', 'finance', 'market']):
                articles.append({
                    'date': item['publishedAt'],
                    'title': item['title'],
                    'content': item.get('description', ''),
                    'source': 'NewsAPI',
                    'url': item.get('url', '')
                })

            if len(articles) == 20:
                break

        return articles
    except Exception as e:
        print(f"Error fetching NewsAPI news for {ticker}: {str(e)}")
        return []
def clean_text(text):
    """Clean and preprocess the text"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text


def assign_sentiment(text):
    positive_words = ['surge', 'gain', 'uptick', 'bullish', 'outperform', 'record', 'breakthrough', 'optimistic',
                      'revolutionize', 'soar', 'rally', 'boost', 'upgrade', 'beat', 'exceed', 'growth', 'profit',
                      'success', 'innovation', 'partnership', 'launch', 'expansion', 'momentum', 'thrive', 'prosper',
                      'upswing', 'rebound', 'recover', 'advance', 'climb', 'rise', 'improve', 'flourish', 'upbeat',
                      'promising', 'robust', 'stellar', 'stronghold', 'uptrend', 'lucrative', 'great', 'boom']

    negative_words = ['slump', 'decline', 'downturn', 'bearish', 'underperform', 'uncertainty', 'investigation',
                      'drop', 'fall', 'plunge', 'sink', 'tumble', 'downgrade', 'miss', 'loss', 'lawsuit', 'crisis',
                      'recall', 'bankruptcy', 'debt', 'cut', 'risk', 'warning', 'crash', 'plummet', 'struggle',
                      'concern', 'worry', 'weak', 'volatile', 'setback', 'scandal', 'disappointing', 'gloomy',
                      'pessimistic', 'shrink', 'slowing', 'sluggish', 'erode', 'threat', 'poor', 'false']

    text = text.lower()
    positive_count = sum(word in text for word in positive_words)
    negative_count = sum(word in text for word in negative_words)

    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'


def is_relevant(title, content, ticker):
    title = str(title) if title is not None else ""
    content = str(content) if content is not None else ""
    text = (title + " " + content).lower()
    ticker = ticker.lower()
    return ticker in text or 'stock' in text or 'share' in text or 'market' in text


def collect_and_preprocess_data(ticker):
    all_data = []
    print(f"Fetching news for {ticker}...")
    newsapi_articles = fetch_newsapi_news(ticker)
    finnhub_articles = fetch_finnhub_news(ticker)
    all_articles = newsapi_articles + finnhub_articles
    if not all_articles:
        print(f"No articles found for {ticker}")
        return []
    for article in all_articles:
        title = article.get('title')
        content = article.get('content')
        date = article.get('date')
        source = article.get('source')
        url = article.get('url')

        # Skip articles with missing essential data
        if not all([title, content, date, source, url]):
            print(f"Skipping article with missing data for {ticker}")
            continue

        if not is_relevant(title, content, ticker):
            continue
        cleaned_title = clean_text(title)
        cleaned_content = clean_text(content)
        sentiment = assign_sentiment(cleaned_title + " " + cleaned_content)
        all_data.append({
            'ticker': ticker,
            'date': date,
            'title': title,
            'content': content,
            'cleaned_title': cleaned_title,
            'cleaned_content': cleaned_content,
            'sentiment': sentiment,
            'source': source,
            'url': url
        })
        # Add a small delay to avoid overwhelming servers
        time.sleep(0.1)
    return all_data


def collect_data_for_all_stocks():
    all_stock_data = []
    for ticker in TOP_TECH_STOCKS:
        stock_data = collect_and_preprocess_data(ticker)
        all_stock_data.extend(stock_data)
        time.sleep(1)  # Add a delay between stocks to respect API rate limits
    return all_stock_data


if __name__ == "__main__":
    data = collect_data_for_all_stocks()
    df = pd.DataFrame(data)
    df.to_csv('tech_stock_sentiment_data.csv', index=False)
    print(f"Data collected and saved. Total records: {len(df)}")
