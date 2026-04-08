import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Create dataset
data = {
    'hours_studied': [1,2,3,4,5,6,7,8,9,10],
    'sleep_hours': [6,7,6,7,8,7,8,9,8,9],
    'previous_score': [40,50,55,60,65,70,75,80,85,90],
    'final_score': [42,52,57,63,67,72,78,83,88,93]
}

df = pd.DataFrame(data)

# Features & target
X = df[['hours_studied', 'sleep_hours', 'previous_score']]
y = df['final_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
error = mean_absolute_error(y_test, predictions)

print("Predictions:", predictions)
print("Actual:", y_test.values)
print("Mean Absolute Error:", error)

# Test custom input
sample = pd.DataFrame([[5, 7, 65]], columns=['hours_studied', 'sleep_hours', 'previous_score'])
predicted_score = model.predict(sample)
print("Predicted score:", predicted_score)

# Plot
plt.scatter(y_test, predictions)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Scores")
plt.show()