import pandas as pd
import os

house  = pd.read_csv("data/house.csv")
output_path = "outputRegression"
os.makedirs(output_path, exist_ok=True)

#inspect dataset
print(house.head())
print(house.shape)
print(house.info())
print(house.isnull().sum())

#handles missing values
house  = house.dropna()

#split features and target
X = house.drop('medv', axis=1)
y = house['medv']

#train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

#visualization
sns.histplot(house['medv'], kde=True)
plt.title("Distribution of House Prices")
plt.savefig("outputRegression/house_price_distribution.png")
plt.close()

#correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(house.corr(),cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputRegression/heatmap.png")
plt.close()

#ANN model
import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

_, mae1 = model.evaluate(X_test, y_test)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

_, mae2 = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae2:.4f}") 
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

y_pred = model.predict(X_test)

with open("outputRegression/predictions.txt", "w") as f:
    f.write(f"MAE: {mae:.4f}\n")



pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()})
pred_df.to_csv("outputRegression/predictions.csv", index=False)


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.savefig("outputRegression/actual_vs_predicted.png")
plt.close()


errors = y_test - y_pred.flatten()
sns.histplot(errors, kde=True)
plt.title("Distribution of Prediction Errors")
plt.savefig("outputRegression/error_distribution.png")
plt.close()

with open(os.path.join(output_path, "results.txt"), "w") as f:
    f.write(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}\n")




results = [
    ["Baseline", 50, 32, "16-8", mae1],
    ["More Neurons", 50, 32, "32-16", mae2],
]

df_results = pd.DataFrame(results, columns=["Model", "Epochs", "Batch", "Layers", "MAE"])

df_results.to_csv(os.path.join(output_path, "regression_comparison.csv"), index=False)