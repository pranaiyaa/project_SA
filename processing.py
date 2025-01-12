import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources from NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess text
def preprocess_text(text):
    if pd.isnull(text):  # Handle missing or null values
        return ""

    # Noise removal: Remove URLs, non-ASCII characters, and punctuation
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters (e.g., emojis)
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stop word removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string after processing
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# Load the JSONL file
input_file_path = "C:/Users/prana/Downloads/college/Reddit mi/r_maleinfertility_posts.jsonl"  # Replace with your actual file path

# Read the JSONL file into a DataFrame
df = pd.read_json(input_file_path, lines=True)

# Preprocess the 'post_title' column (or other text columns)
df['cleaned_post_title'] = df['post_title'].apply(preprocess_text)

# Save the cleaned data to a new CSV file
output_file_path = "C:/Users/prana/Downloads/college/Reddit mi/r_maleinfertility_posts_cleaned.jsonl"  # Update with your desired output path
df.to_csv(output_file_path, index=False)

# Show the first few rows of the cleaned data to verify
print(df.head())
