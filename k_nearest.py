import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model (k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict & evaluate
y_pred = knn.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Predict a new flower
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = knn.predict(new_flower)
print(f"\nNew flower predicted as: {iris.target_names[prediction[0]]}")

# Plot accuracy for different K values
k_values = range(1, 21)
accuracies = [KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
              .score(X_test, y_test) for k in k_values]

plt.plot(k_values, accuracies, marker='o', color='green')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different K Values')
plt.grid(True)
plt.show()