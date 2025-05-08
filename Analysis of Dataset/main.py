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
    from sklearn.impute import SimpleImputer
    import numpy as np

    # Handle infinite and NaNs
    X = numeric_df.replace([np.inf, -np.inf], np.nan)
    imp = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imp.fit_transform(X), columns=numeric_df.columns)
    y = LabelEncoder().fit_transform(df["Label"])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.nlargest(5).index

    importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    # Boxplot of top 5 features grouped by label
    df_cleaned = df.copy()
    df_cleaned[numeric_df.columns] = X  # use imputed data
    df_melted = pd.melt(df_cleaned[top_features].assign(Label=df["Label"]), id_vars="Label")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="variable", y="value", hue="Label", data=df_melted)
    plt.title("Distribution of Top 5 Important Features by Label")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Feature importance skipped:", e)
