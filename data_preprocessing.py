import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase and remove non-alphabetical characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column):
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

if __name__ == "__main__":
    # Sample usage with placeholder file
    path = "sample_data.csv"  # Replace with your actual dataset path
    df = pd.read_csv(path)

    processed_df = preprocess_dataframe(df, "text")
    processed_df.to_csv("cleaned_data.csv", index=False)
    print("Preprocessing complete. Saved as cleaned_data.csv.")
