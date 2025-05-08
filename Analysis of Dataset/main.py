import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv("../dataset/flowFeatures.csv")

# Basic summary
print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())

# Check for missing values
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0])

# Label distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Label")
plt.title("Label Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure(figsize=(12, 10))
corr = numeric_df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Optional: feature importance using random forest
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    X = numeric_df.copy()
    y = LabelEncoder().fit_transform(df["Label"])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Feature importance skipped:", e)
