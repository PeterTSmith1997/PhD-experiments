import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Step 1: Load Dataset ---
df = pd.read_csv("../dataset/flowFeatures.csv")

# --- Step 2: Basic Exploration ---
print("Shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())

missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0])

# --- Step 3: Label Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Label", order=df["Label"].value_counts().index)
plt.title("Label Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Step 4: Correlation Heatmap ---
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(12, 10))
corr = numeric_df.corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# --- Step 5: Feature Importance ---
try:
    X = numeric_df.copy()
    y = LabelEncoder().fit_transform(df["Label"])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.nlargest(10).plot(kind="barh")
    plt.title("Top 10 Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Feature importance skipped:", e)

# --- Step 6: Check for Conflicting Duplicate Rows ---
feature_cols = df.columns.difference(["Label"])
conflicts = df.groupby(list(feature_cols))["Label"].nunique()
conflicting_rows = conflicts[conflicts > 1]
print(f"\n⚠️ Potentially mislabeled duplicate rows: {len(conflicting_rows)}")

# --- Step 7: Classifier Evaluation on Label Integrity ---
df_clean = df.dropna()
X = df_clean.select_dtypes(include=["number"])
y = LabelEncoder().fit_transform(df_clean["Label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=LabelEncoder().fit(df_clean["Label"]).classes_))
