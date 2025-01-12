import pandas as pd
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load your data
df = pd.read_csv('C:/Users/prana/Downloads/college/Reddit mi/r_maleinfertility_comments_cleaned.csv')  # Replace with your file path
documents = df['cleaned_comment_body'].tolist()  # Replace 'text_column' with the name of your text column

# Preprocess the text
stop_words = set(stopwords.words('english'))
preprocessed_docs = []

for doc in documents:
    if isinstance(doc, str):  # Ensure the document is a string
        tokens = word_tokenize(doc.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        preprocessed_docs.append(tokens)
    else:
        # Handle non-string entries (e.g., NaN or numeric values)
        preprocessed_docs.append([])

# Create a dictionary and corpus
dictionary = corpora.Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

# Apply LDA
lda_model = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)

# Define output file path
output_file = 'C:/Users/prana/Downloads/college/Reddit mi/topic_modeling/mi_comments.csv'

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Write topics to the output file
with open(output_file, 'w') as f:
    for idx, topic in lda_model.print_topics(-1):
        f.write(f'Topic: {idx}\nWords: {topic}\n\n')

print(f"Topic modeling results saved to {output_file}")
