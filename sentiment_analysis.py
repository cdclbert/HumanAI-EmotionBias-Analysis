import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')

def apply_vader(df, text_column):
    sia = SentimentIntensityAnalyzer()
    def label_sentiment(text):
        score = sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    df["vader_sentiment"] = df[text_column].apply(label_sentiment)
    return df

def train_logistic_regression(df, text_column, label_column):
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df[text_column])
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    return clf, tfidf

if __name__ == "__main__":
    # Load preprocessed text
    df = pd.read_csv("cleaned_data.csv")  # Output from data_preprocessing.py
    df = apply_vader(df, "text")

    # Optional: save VADER output
    df.to_csv("sentiment_labeled.csv", index=False)

    # Train ML model using VADER labels as pseudo ground-truth
    clf, tfidf = train_logistic_regression(df, "text", "vader_sentiment")
