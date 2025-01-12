import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Download necessary NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Initialize NLTK Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Create get_sentiment function
def get_sentiment(text):
    if isinstance(text, str):
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    return None

# Specify input file path
input_file = 'C:/Users/prana/Downloads/college/Reddit mi/r_maleinfertility_comments_SA.csv'  # Full file path

# Ensure the input file exists
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Input file '{input_file}' not found.")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Convert 'Date' column to datetime format if necessary
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')

# Apply get_sentiment function to the desired column (replace 'cleaned_comment_text' with your column name)
df['sentiment_post'] = df['cleaned_post_body'].apply(get_sentiment)

# Group by date and calculate the mean sentiment score for line plot
sentiment_trend = df.groupby('Date')['sentiment_post'].mean()

# Line plot of sentiment scores vs date
plt.figure(figsize=(12, 6))
plt.plot(sentiment_trend.index, sentiment_trend, color='b', marker='o', label='Sentiment Score')
plt.title('Sentiment Trend Over Time for r/maleinfertility')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
