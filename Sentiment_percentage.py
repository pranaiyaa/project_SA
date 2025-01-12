import pandas as pd

# Load the CSV file
file_path = "C:/Users/prana/Downloads/college/Reddit mi/r_maleinfertility_comments_SA.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Define the columns you want to analyze (replace with your actual column names)
columns_to_analyze = [ 'sentiment_post']

# Iterate through each column and calculate frequency and percentage of each score (-1, 0, 1)
for column in columns_to_analyze:
    print(f"Sentiment score frequencies for {column}:")
    
    # Count the frequency of each sentiment score
    sentiment_counts = df[column].value_counts()
    
    # Calculate the total number of values for percentage calculation
    total_values = sentiment_counts.sum()
    
    # Print frequency and percentage of each score (-1, 0, 1)
    for score in [-1, 0, 1]:
        count = sentiment_counts.get(score, 0)
        percentage = (count / total_values) * 100
        print(f"Score {score}: {count} ({percentage:.2f}%)")
    
    print("\n")  # Add a new line for better readability between columns
