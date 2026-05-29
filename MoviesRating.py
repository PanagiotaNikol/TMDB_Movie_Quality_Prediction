import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================
# LOAD DATASET
# =========================================

# Διαβάζουμε το dataset ταινιών
df = pd.read_csv("Projects For Uni\\tmdb_5000_movies.csv")

# =========================================
# CREATE TARGET VARIABLE
# =========================================

# Αν rating >= 7 -> 1 (Good Movie)
# Αλλιώς -> 0 (Not Good Movie)

df["target"] = df["vote_average"].apply(
    lambda x: 1 if x >= 7 else 0
)

# =========================================
# FEATURE SELECTION
# =========================================

features = ["budget", "runtime", "popularity", "vote_count"]

# Κρατάμε μόνο τα features + target
df_clean = df[features + ["target"]]

# =========================================
# DATA CLEANING
# =========================================

# Αφαίρεση κενών τιμών
df_clean = df_clean.dropna()

# =========================================
# DATASET INFORMATION
# =========================================

print("Πρώτες γραμμές dataset:")
print(df_clean.head())

print("\nΠληροφορίες dataset:")
print(df_clean.info())

print("\nΚατανομή target:")
print(df_clean["target"].value_counts())

# =========================================
# TARGET DISTRIBUTION GRAPH
# =========================================

target_counts = df_clean["target"].value_counts()

plt.figure(figsize=(6,5))

plt.bar(
    ["Not Good", "Good"],
    target_counts
)

plt.title("Target Distribution")
plt.ylabel("Number of Movies")

plt.show()

# =========================================
# DEFINE FEATURES AND TARGET
# =========================================

X = df_clean[features]
y = df_clean["target"]

# =========================================
# TRAIN / TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# LOGISTIC REGRESSION MODEL
# =========================================

# Δημιουργία μοντέλου
lr_model = LogisticRegression(max_iter=1000)

# Εκπαίδευση
lr_model.fit(X_train, y_train)

# Προβλέψεις
lr_pred = lr_model.predict(X_test)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)

# =========================================
# LOGISTIC REGRESSION RESULTS
# =========================================

print("===================================")
print("LOGISTIC REGRESSION RESULTS")
print("===================================")

print("Accuracy:", lr_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# =========================================
# RANDOM FOREST MODEL
# =========================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Εκπαίδευση μοντέλου
rf_model.fit(X_train, y_train)

# Προβλέψεις
rf_pred = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

# =========================================
# RANDOM FOREST RESULTS
# =========================================

print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================")

print("Accuracy:", rf_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# =========================================
# MODEL COMPARISON
# =========================================

print("\n===================================")
print("MODEL COMPARISON")
print("===================================")

print(f"Logistic Regression Accuracy: {lr_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")

# =========================================
# CONFUSION MATRIX (RANDOM FOREST)
# =========================================

cm = confusion_matrix(y_test, rf_pred)

# Μετατροπή σε ποσοστά
cm_percent = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=["Not Good", "Good"],
    yticklabels=["Not Good", "Good"]
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

cm_lr = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =========================================
# FEATURE IMPORTANCE
# =========================================

importances = rf_model.feature_importances_

plt.figure(figsize=(6, 5))

plt.bar(features, importances)

plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importance")

plt.show()

# =========================================
# MODEL COMPARISON GRAPH
# =========================================

models = ["Logistic Regression", "Random Forest"]
scores = [lr_accuracy, rf_accuracy]

plt.figure(figsize=(6, 5))

plt.bar(models, scores)

plt.title("Model Comparison")
plt.ylabel("Accuracy")

plt.show()
