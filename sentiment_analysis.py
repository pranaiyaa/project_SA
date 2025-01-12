import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download necessary NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Initialize NLTK Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Create get_sentiment function
def get_sentiment(text):
    if isinstance(text, str):  # Ensure the input is a string
        text = text.strip()  # Remove leading/trailing whitespaces
        if text:  # Only process non-empty strings
            scores = analyzer.polarity_scores(text)
            # Classify sentiment based on compound score
            if scores['compound'] > 0:
                return 1  # Positive sentiment
            elif scores['compound'] == 0:
                return 0  # Neutral sentiment
            else:
                return -1  # Negative sentiment
    return None  # Return None for empty or invalid strings

# Specify input file path and output directory
input_file = 'C:/Users/prana/Downloads/college/Reddit mi/r_maleinfertility_comments_cleaned.csv'  # Full file path
output_dir = 'C:/Users/prana/Downloads/college/Reddit mi/'  # Output directory

# Ensure the input file exists
if not os.path.isfile(input_file):
    raise FileNotFoundError(f"Input file '{input_file}' not found.")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Check for any missing values in the 'cleaned_post_body' column
missing_data_count = df['Body'].isnull().sum()
if missing_data_count > 0:
    print(f"Warning: There are {missing_data_count} posts with missing 'cleaned_comment_body' data.")

# Apply get_sentiment function to the desired column (replace 'cleaned_post_body' with your column name)
df['sentiment_post'] = df['Body'].apply(get_sentiment)

# Check for any rows that still have 'None' as sentiment score
missing_sentiment_count = df['sentiment_comment'].isnull().sum()
if missing_sentiment_count > 0:
    print(f"Warning: {missing_sentiment_count} posts did not receive a sentiment score.")

# Save the updated DataFrame to a new CSV file
output_csv = os.path.join(output_dir, 'r_maleinfertility_comments_SA.csv')
df.to_csv(output_csv, index=False)

print(f"Sentiment analysis complete. Results saved to {output_csv}")
