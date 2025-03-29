import requests
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from datetime import datetime, timedelta

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# API Key (Replace with your own)
NEWS_API_KEY = "daf72e77eb8944a0ab152034bc533538"

# Step 1: Fetch financial news (Last 7-14 Days)
def fetch_financial_news(days=14, max_articles=20):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    url = f"https://newsapi.org/v2/everything?q=finance OR stock market OR policy&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&sortBy=popularity&language=en&apiKey={NEWS_API_KEY}"
    
    response = requests.get(url).json()
    articles = response.get('articles', [])

    if not articles:
        print("⚠️ No news articles found. Please try again later.")
        return []

    return [(article.get('title', ''), article.get('description', '')) for article in articles[:max_articles]]

# Step 2: Text Preprocessing
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = re.sub(r'\W+', ' ', text.lower())  
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Step 3: Sentiment Analysis (VADER + TextBlob only)
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    vader_score = sia.polarity_scores(text)['compound']
    textblob_score = TextBlob(text).sentiment.polarity

    # Convert scores into sentiment categories
    def get_sentiment_label(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    vader_sentiment = get_sentiment_label(vader_score)
    textblob_sentiment = get_sentiment_label(textblob_score)

    return vader_score, vader_sentiment, textblob_score, textblob_sentiment

# Step 4: Process News Articles & Get Sentiment Scores
news_data = fetch_financial_news(days=14)

if news_data:
    processed_data = []
    all_text = ""

    for title, desc in news_data:
        if title and desc:
            full_text = preprocess_text(title + " " + desc)
            if full_text:
                vader, vader_sentiment, textblob, textblob_sentiment = analyze_sentiment(full_text)
                processed_data.append([title, desc, vader, vader_sentiment, textblob, textblob_sentiment])
                all_text += " " + full_text  

    # Convert to DataFrame with additional sentiment columns
    df = pd.DataFrame(processed_data, columns=['Title', 'Description', 'VADER', 'VADER Sentiment', 'TextBlob', 'TextBlob Sentiment'])

    # Save results to CSV
    df.to_csv("financial_sentiment.csv", index=False)
    pd.set_option('display.max_columns', None)
    print("✅ Sentiment Analysis Completed! CSV file saved.")
    print(df.head())

else:
    print("⚠️ No articles found. Exiting script.")


# Mean Sentiment Score
mean_vader_score = df['VADER'].mean()
mean_textblob_score = df['TextBlob'].mean()

print(f"Mean VADER Sentiment Score: {mean_vader_score}")
print(f"Mean TextBlob Sentiment Score: {mean_textblob_score}")
# Sentiment Distribution
vader_sentiment_counts = df['VADER Sentiment'].value_counts()
textblob_sentiment_counts = df['TextBlob Sentiment'].value_counts()

print("VADER Sentiment Distribution:")
print(vader_sentiment_counts)
print("\nTextBlob Sentiment Distribution:")
print(textblob_sentiment_counts)

# Add a Date Column (if available in the dataset)
df['Date'] = pd.to_datetime(df['Date'])  # Adjust to match the actual date column

# Group by Date and Calculate Average Sentiment
sentiment_trend = df.groupby(df['Date'].dt.date)['VADER'].mean()

# Plot the Trend
import matplotlib.pyplot as plt

sentiment_trend.plot(kind='line', title="Sentiment Trend Over Time", ylabel="Average VADER Score", xlabel="Date")
plt.show()
