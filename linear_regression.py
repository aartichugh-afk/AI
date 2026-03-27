import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data: house size (sq ft) vs price (in lakhs)
house_size = np.array([600, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000]).reshape(-1, 1)
price      = np.array([25,  35,  45,   55,   70,   85,   95,  110,  130,  160])

# Split data
X_train, X_test, y_train, y_test = train_test_split(house_size, price, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print(f"R² Score     : {r2_score(y_test, y_pred):.2f}")
print(f"MSE          : {mean_squared_error(y_test, y_pred):.2f}")
print(f"Predict 1400 sq ft: ₹{model.predict([[1400]])[0]:.1f} Lakhs")

# Plot
plt.scatter(house_size, price, color='blue', label='Actual Data')
plt.plot(house_size, model.predict(house_size), color='red', label='Regression Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price (Lakhs)')
plt.title('House Price Prediction - Linear Regression')
plt.legend()
plt.grid(True)
plt.show()