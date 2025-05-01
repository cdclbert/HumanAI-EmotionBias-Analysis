import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
from bertopic import BERTopic
import seaborn as sns
import matplotlib.pyplot as plt

def encode_user_groups(df, group_column):
    le = LabelEncoder()
    df[group_column + "_enc"] = le.fit_transform(df[group_column])
    return df, le

def topic_modeling(df, text_column):
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(df[text_column])
    df["topic"] = topics
    return df, topic_model

def check_topic_distribution(df, group_column):
    contingency = pd.crosstab(df["topic"], df[group_column])
    chi2, p, _, _ = chi2_contingency(contingency)
    print(f"Chi-squared test for topic distribution across {group_column}:")
    print(f"Chi2 = {chi2:.2f}, p = {p:.4f}")
    sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Topic vs {group_column}")
    plt.ylabel("Topic")
    plt.xlabel(group_column)
    plt.tight_layout()
    plt.show()

def disparate_impact(df, group_column, outcome_column, privileged_value):
    rates = df.groupby(group_column)[outcome_column].value_counts(normalize=True).unstack()
    if len(rates.columns) < 2:
        print("Not enough class variation in outcomes for disparate impact.")
        return
    outcome_positive = rates.columns[1]  # Assuming binary classification
    protected_groups = [g for g in rates.index if g != privileged_value]
    for group in protected_groups:
        ratio = rates.loc[group, outcome_positive] / rates.loc[privileged_value, outcome_positive]
        print(f"Disparate Impact Ratio ({group} vs {privileged_value}): {ratio:.2f}")

if __name__ == "__main__":
    # Load AI interaction data with user metadata
    df = pd.read_csv("sentiment_labeled.csv")  # Add your user metadata columns here

    # Example: simulate user groups
    import numpy as np
    np.random.seed(42)
    df["user_group"] = np.random.choice(["GroupA", "GroupB"], size=len(df))

    # Cluster AI response topics
    df, topic_model = topic_modeling(df, "text")

    # Chi-squared test of topic distribution across user groups
    check_topic_distribution(df, "user_group")

    # Disparate impact on sentiment
    disparate_impact(df, "user_group", "vader_sentiment", privileged_value="GroupA")
