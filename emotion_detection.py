import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Load BERT-based emotion model (GoEmotions)
MODEL = "j-hartmann/emotion-english-distilroberta-base"

def load_emotion_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    emo_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)
    return emo_pipeline

def predict_emotions(df, text_column, emo_pipeline):
    emotion_labels = []
    scores = []

    for text in df[text_column]:
        result = emo_pipeline(text)[0]
        emotion_labels.append(result['label'])
        scores.append(result['score'])

    df["emotion"] = emotion_labels
    df["emotion_confidence"] = scores
    return df

def plot_emotion_distribution(df):
    plt.figure(figsize=(10,6))
    sns.countplot(data=df, x="emotion", order=df["emotion"].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Predicted Emotion Distribution")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("cleaned_data.csv")
    emo_pipe = load_emotion_pipeline()
    df = predict_emotions(df, "text", emo_pipe)

    df.to_csv("emotion_labeled.csv", index=False)
    print("Emotion prediction complete. Saved to emotion_labeled.csv.")

    plot_emotion_distribution(df)
