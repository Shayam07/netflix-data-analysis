# =========================================================
# Predictive Modeling Using Machine Learning
# Internship Project
# =========================================================
#
# Objective:
# Build machine learning models to predict outcomes
# using supervised learning techniques.
#
# Algorithms Used:
# 1. Logistic Regression
# 2. Decision Tree Classifier
# 3. Random Forest Classifier
#
# Features:
# ✔ Data preprocessing
# ✔ Model training & testing
# ✔ Accuracy evaluation
# ✔ Confusion Matrix visualization
# ✔ ROC Curve visualization
#
# Dataset:
# Breast Cancer Dataset (built into sklearn)
#
# =========================================================

# -----------------------------
# Import Required Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# -----------------------------
# Load Dataset
# -----------------------------

data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Target column
df["target"] = data.target

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# -----------------------------
# Split Features and Target
# -----------------------------

X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 1. Logistic Regression
# =========================================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

lr_model = LogisticRegression()

# Train Model
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_pred = lr_model.predict(X_test_scaled)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"Accuracy: {lr_accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# =========================================================
# 2. Decision Tree Classifier
# =========================================================

print("\n==============================")
print("DECISION TREE CLASSIFIER")
print("==============================")

dt_model = DecisionTreeClassifier(random_state=42)

# Train Model
dt_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)

# Accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"Accuracy: {dt_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

# =========================================================
# 3. Random Forest Classifier
# =========================================================

print("\n==============================")
print("RANDOM FOREST CLASSIFIER")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Model
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Accuracy: {rf_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# =========================================================
# Compare Model Accuracies
# =========================================================

models = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracies = [lr_accuracy, dt_accuracy, rf_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)

for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.002, f"{acc:.2f}", ha='center')

plt.show()

# =========================================================
# Confusion Matrix Visualization
# =========================================================

print("\nGenerating Confusion Matrix...")

cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")
plt.show()

# =========================================================
# ROC Curve Visualization
# =========================================================

# Predict probabilities
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)

# Calculate AUC Score
auc_score = roc_auc_score(y_test, rf_probs)

# Plot ROC Curve
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")

# Random Guess Line
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")
plt.legend()

plt.show()

# =========================================================
# Feature Importance (Random Forest)
# =========================================================

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))

top_features = feature_importance.head(10)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")

plt.show()

# =========================================================
# Final Conclusion
# =========================================================

print("\n======================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("======================================")

print(f"""
Model Performance Summary:

1. Logistic Regression Accuracy : {lr_accuracy:.4f}
2. Decision Tree Accuracy       : {dt_accuracy:.4f}
3. Random Forest Accuracy       : {rf_accuracy:.4f}

Best Performing Model:
Random Forest Classifier
""")