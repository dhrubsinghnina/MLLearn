import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("LandPrice_GradientDescent/land_price.csv")
print(data.head())

# Check null values
print("Null:")
print(data.isnull().sum())

# Data summary
print(data.describe())

# Features and target
X = data[['area_sqft']]
y = data['price_lakhs']

# Train-test split (FIXED typo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale ONLY training data (avoid data leakage) featurs only-X_scaled
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Gradient Descent Regressor
model = SGDRegressor(
    max_iter=1000,
    learning_rate="constant",
    eta0=0.01,
    random_state=42
)

# Train model
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
print("------- Model Evaluation -------")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Correct regression plot
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (lakhs)")
plt.ylabel("Predicted Price (lakhs)")
plt.title("Actual vs Predicted Land Price")

# Perfect-fit reference line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.show()
