import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load & prepare dataset
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)

# Convert to binary classification (above median = diabetic)
y = (data.target > np.median(data.target)).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# Predict & Evaluate
y_pred = dt.predict(X_test)
print(f"Accuracy : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Diabetic", "Diabetic"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Feature Importance
importance = pd.Series(dt.feature_importances_, index=data.feature_names).sort_values(ascending=False)
print("\nTop Features:")
print(importance)

# Predict for a new patient
new_patient = np.array([X_test.iloc[0]])
result = dt.predict(new_patient)
print(f"\nSample Patient Prediction: {'Diabetic' if result[0] == 1 else 'Not Diabetic'}")

# --- Plots ---

# Plot 1: Feature Importance
plt.figure(figsize=(8, 5))
importance.plot(kind='bar', color='steelblue')
plt.title('Feature Importance - Decision Tree')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.show()

# Plot 2: Decision Tree Structure
plt.figure(figsize=(18, 8))
tree.plot_tree(dt, feature_names=data.feature_names,
               class_names=["Not Diabetic", "Diabetic"],
               filled=True, fontsize=9)
plt.title('Decision Tree Structure')
plt.tight_layout()
plt.show()