
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def load_data():
    df = pd.read_csv("data/customer_churn.csv")
    return df



def run_eda():

    df = load_data()

    print("\nDataset Preview")
    print(df.head())

    print("\nDataset Information")
    df.info()

    print("\nDataset Shape")
    print(df.shape)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nChurn Distribution")
    print(df["churn"].value_counts())

    print("\nDescriptive Statistics")
    print(df.describe())

    print("\nChurn Percentage")
    print(df["churn"].value_counts(normalize=True) * 100)

    print("\nCorrelation Matrix")
    print(df.corr(numeric_only=True))

    # Correlation Heatmap
    plt.figure(figsize=(10, 7))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Customer Churn Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Churn Distribution Graph
    sns.countplot(data=df, x="churn")

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = Retained, 1 = Churned)")
    plt.ylabel("Number of Customers")

    plt.show()


if __name__ == "__main__":
    run_eda()
