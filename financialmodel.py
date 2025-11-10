

import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from newsapi import NewsApiClient
from transformers import pipeline

NEWS_API_KEY = "ec064ce719114fe78bd3affdd71e5db8"  # Replace with your actual API key

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Lazy-initialize sentiment analysis using BERT to avoid heavy model load on import
sentiment_analysis = None

def get_sentiment_pipeline():
    """Lazy create and cache the transformers sentiment pipeline."""
    global sentiment_analysis
    if sentiment_analysis is None:
        sentiment_analysis = pipeline("sentiment-analysis")
    return sentiment_analysis

def fetch_data(ticker):
    """Fetch 20 years of historical stock data using yfinance."""
    t = yf.Ticker(ticker)
    historical_data = t.history(period="20y")
    historical_data.reset_index(inplace=True)
    historical_data['Date'] = historical_data['Date'].dt.tz_localize(None)
    return historical_data


def fetch_news(ticker):
    """Fetch recent news articles for the given stock ticker."""
    today = datetime.today().date()

    # Set the max allowed date (from the NewsAPI error message)
    max_allowed_date = datetime(2024, 12, 30).date()

    # Adjust date range dynamically to avoid API errors
    from_date = max(max_allowed_date, today - timedelta(days=7))
    to_date = today

    print(f"Fetching news from {from_date} to {to_date}...")

    all_articles = newsapi.get_everything(
        q=ticker,
        from_param=from_date.strftime('%Y-%m-%d'),
        to=to_date.strftime('%Y-%m-%d'),
        language='en',
        sort_by='relevancy'
    )

    news_data = all_articles.get('articles', [])

    if not news_data:
        print(f"No articles found for {ticker}")
        return pd.DataFrame(columns=['Date', 'title', 'description', 'content'])

    news_df = pd.DataFrame(news_data)
    news_df['Date'] = pd.to_datetime(news_df['publishedAt']).dt.tz_localize(None)

    return news_df[['Date', 'title', 'description', 'content']]


def extract_sentiment(news_df):
    """Extract sentiment scores using BERT sentiment analysis."""
    if news_df.empty:
        return news_df

    # Combine text fields (title, description, content)
    news_df['combined_text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('') + ' ' + news_df['content'].fillna('')

    # Use a lazily-initialized pipeline so importing this module doesn't load the model
    sp = get_sentiment_pipeline()
    news_df['sentiment_score'] = news_df['combined_text'].apply(lambda x: sp(x[:512])[0]['score'])  # Limit to 512 characters
    news_df['sentiment_label'] = news_df['combined_text'].apply(lambda x: sp(x[:512])[0]['label'])

    # Convert labels into numerical values
    news_df['title_sentiment'] = news_df.apply(
        lambda x: x['sentiment_score'] if x['sentiment_label'] == 'POSITIVE' else 
                  (-x['sentiment_score'] if x['sentiment_label'] == 'NEGATIVE' else 0),
        axis=1
    )

    return news_df[['Date', 'title', 'title_sentiment', 'combined_text']]


def merge_news_with_data(data, news):
    """Merge stock data with news sentiment scores."""
    if news.empty:
        data['title_sentiment'] = 0
        return data

    merged_data = data.merge(news, on='Date', how='left')
    merged_data.fillna({'title_sentiment': 0}, inplace=True)

    # Weight sentiment by volume for impact
    merged_data['weighted_sentiment'] = merged_data['title_sentiment'] * merged_data['Volume']
    
    return merged_data

def process_data(data, look_back=60):
    """Create lag features and prepare data for model training."""
    
    # Ensure dataset has enough rows before applying lags
    if len(data) < look_back + 1:
        print(f"⚠️ Warning: Dataset has only {len(data)} rows. Reducing look_back to {max(1, len(data) - 1)}.")
        look_back = max(1, len(data) - 1)  # Adjust look_back dynamically

    # Create lag features
    for i in range(1, look_back + 1):
        data[f"lag_{i}"] = data['Close'].shift(i)

    # Drop NaNs created by lag features
    data.dropna(inplace=True)

    # Ensure we still have data after dropping NaNs
    if data.empty:
        raise ValueError(f"⚠️ Error: After applying lag features, no rows remain. Reduce `look_back` or check dataset. Available rows: {len(data)}")

    # Define columns to exclude from training
    columns_to_drop = ['Close', 'Date', 'title', 'combined_text']
    X = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')
    y = data['Close']

    # Ensure X is not empty before scaling
    if X.empty:
        raise ValueError("⚠️ Error: No valid feature data remaining after preprocessing. Check dataset integrity.")

    # Scale data
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    return pd.DataFrame(X_scaled, columns=X.columns), pd.Series(y_scaled.ravel()), scaler_x, scaler_y
def plot_feature_importance(model, X):
    """Plot feature importance of the trained model."""
    importance = model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.barh([features[i] for i in sorted_idx[-10:]], importance[sorted_idx[-10:]])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Important Features")
    plt.show()


# NOTE: the earlier non-debug `execute` run was removed so importing this module
# doesn't immediately fetch data, download models, or run long blocking tasks.

def execute(ticker, target_date, look_back=60):
    """Execute the stock prediction pipeline with debugging information."""

    print(f"\n🚀 Fetching stock data for {ticker}...")
    data = fetch_data(ticker)
    print(f"✅ Stock data retrieved: {len(data)} rows\n")

    print(f"📰 Fetching news for {ticker}...")
    news = fetch_news(ticker)
    print(f"✅ News data retrieved: {len(news)} articles\n")

    print("🔍 Extracting sentiment from news articles...")
    news = extract_sentiment(news)
    print(f"✅ Sentiment analysis applied: {news.shape[0]} rows\n")

    print("🔗 Merging stock data with news sentiment...")
    data = merge_news_with_data(data, news)
    print(f"✅ Data after merging: {data.shape}\n")

    print("📊 Preview of merged data:")
    print(data.head())  # Show first few rows

    print("\n⏳ Processing data with lag features...")
    try:
        X, y, scaler_x, scaler_y = process_data(data, look_back)
    except ValueError as e:
        print(f"\n🚨 Error in process_data(): {e}")
        print("📊 Data before failing:")
        print(data.head())  # Show first few rows before error
        return

    print(f"✅ Processed data: {X.shape} features, {y.shape} targets\n")
    print("📊 Preview of processed features:")
    print(X.head())  # Show first few processed rows

    print("🛠 Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✅ Model training complete!\n")

    print("📈 Predicting future price...")
    target_data = X.iloc[-1].values.reshape(1, -1)
    pred_scaled = model.predict(target_data)
    pred = scaler_y.inverse_transform([[pred_scaled[0]]])[0, 0]

    print(f"\n📊 **Predicted closing price for {ticker} on {target_date}: ${pred:.2f}**\n")

    last_known_price = scaler_y.inverse_transform([[y.iloc[-1]]])[0, 0]
    if pred > last_known_price:
        print(f"📊 **Recommendation:** Buy a **CALL** option for {ticker} expiring on {target_date}.")
    elif pred < last_known_price:
        print(f"📉 **Recommendation:** Buy a **PUT** option for {ticker} expiring on {target_date}.")
    else:
        print(f"⚖️ No clear direction for {ticker} on {target_date}.")

    # Display feature importance
    plot_feature_importance(model, X)

    # Show top 3 positive news articles
    positive_news = news[news['title_sentiment'] > 0].sort_values(by="Date", ascending=False).head(3)
    print("\n🔍 **Top 3 Positive News Articles:**")
    for _, row in positive_news.iterrows():
        print(f"\n📌 **Title:** {row['title']}")
        print(f"📝 **Content:** {row['combined_text'][:200]}...")  # Display first 200 characters


# Run with debug mode (only when executed as a script)
if __name__ == "__main__":
    execute('TSLA', '2025-12-26')