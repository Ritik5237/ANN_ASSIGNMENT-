import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------- SETUP ----------------
output_path = "outputRegression"
os.makedirs(output_path, exist_ok=True)

# ---------------- LOAD DATA ----------------
house = pd.read_csv("data/house.csv")

print(house.head())
print(house.shape)
print(house.info())
print(house.isnull().sum())

house = house.dropna()

# ---------------- SPLIT ----------------
X = house.drop('medv', axis=1)
y = house['medv']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- VISUALIZATION ----------------

sns.histplot(house['medv'], kde=True)
plt.title("House Price Distribution")
plt.savefig(os.path.join(output_path, "price_distribution.png"))
plt.close()

plt.figure(figsize=(10,6))
sns.heatmap(house.corr(), cmap='coolwarm')
plt.title("Heatmap")
plt.savefig(os.path.join(output_path, "heatmap.png"))
plt.close()

# ---------------- FUNCTION TO BUILD MODEL ----------------
def build_model(neurons1, neurons2):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(neurons1, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dense(neurons2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    return model

# ---------------- EXPERIMENTS ----------------

# Baseline
model1 = build_model(16, 8)
model1.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
_, mae1 = model1.evaluate(X_test, y_test)

# More neurons
model2 = build_model(32, 16)
model2.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
_, mae2 = model2.evaluate(X_test, y_test)

# More epochs
model3 = build_model(32, 16)
model3.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
_, mae3 = model3.evaluate(X_test, y_test)

# Smaller batch
model4 = build_model(32, 16)
model4.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
_, mae4 = model4.evaluate(X_test, y_test)

# ---------------- FINAL MODEL ----------------
model = model2  # best model

loss, mae = model.evaluate(X_test, y_test)
print(f"Final MAE: {mae:.4f}")

# ---------------- PREDICTIONS ----------------
y_pred = model.predict(X_test)

# Save results
with open(os.path.join(output_path, "results.txt"), "w") as f:
    f.write(f"MAE: {mae:.4f}\n")

pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred.flatten()
})
pred_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

# ---------------- PLOTS ----------------

# Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig(os.path.join(output_path, "actual_vs_predicted.png"))
plt.close()

# Error distribution
errors = y_test - y_pred.flatten()
sns.histplot(errors, kde=True)
plt.title("Error Distribution")
plt.savefig(os.path.join(output_path, "error_distribution.png"))
plt.close()

# ---------------- COMPARISON ----------------

results = [
    ["Baseline", 50, 32, "16-8", mae1],
    ["More Neurons", 50, 32, "32-16", mae2],
    ["More Epochs", 100, 32, "32-16", mae3],
    ["Batch 16", 50, 16, "32-16", mae4],
]

df_results = pd.DataFrame(results, columns=["Model", "Epochs", "Batch", "Layers", "MAE"])
df_results.to_csv(os.path.join(output_path, "comparison.csv"), index=False)

# Plot comparison
plt.plot(df_results["Model"], df_results["MAE"], marker='o')
plt.title("Hyperparameter Comparison")
plt.ylabel("MAE")
plt.savefig(os.path.join(output_path, "comparison.png"))
plt.close()

# import pandas as pd
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# # SETUP
# output_path = "outputRegression"
# os.makedirs(output_path, exist_ok=True)

# # LOAD DATA
# house = pd.read_csv("data/house.csv").dropna()
# X = house.drop('medv', axis=1).values.astype(np.float32)
# y = house['medv'].values.astype(np.float32)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Convert to Tensors
# X_train_t = torch.FloatTensor(X_train)
# y_train_t = torch.FloatTensor(y_train).view(-1, 1)
# X_test_t = torch.FloatTensor(X_test)
# y_test_t = torch.FloatTensor(y_test).view(-1, 1)

# # PYTORCH REGRESSION MODEL
# class HouseNet(nn.Module):
#     def __init__(self, input_dim, h1, h2):
#         super(HouseNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, h1),
#             nn.ReLU(),
#             nn.Linear(h1, h2),
#             nn.ReLU(),
#             nn.Linear(h2, 1)
#         )
#     def forward(self, x):
#         return self.net(x)

# def train_pytorch_model(h1, h2, epochs, batch_size):
#     model = HouseNet(X_train.shape[1], h1, h2)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     # Simple batching logic
#     for epoch in range(epochs):
#         model.train()
#         permutation = torch.randperm(X_train_t.size(0))
#         for i in range(0, X_train_t.size(0), batch_size):
#             indices = permutation[i:i+batch_size]
#             batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
#             optimizer.zero_grad()
#             outputs = model(batch_x)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
            
#     model.eval()
#     with torch.no_grad():
#         preds = model(X_test_t)
#         mae = torch.abs(preds - y_test_t).mean()
#     return model, mae.item()

# # EXPERIMENTS
# _, mae1 = train_pytorch_model(16, 8, 50, 32)
# model, mae2 = train_pytorch_model(32, 16, 50, 32) # Best Model Choice
# _, mae3 = train_pytorch_model(32, 16, 100, 32)
# _, mae4 = train_pytorch_model(32, 16, 50, 16)

# # SAVE FINAL PREDICTIONS
# y_pred = model(X_test_t).detach().numpy()
# pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.flatten()})
# pred_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False)

# # PLOT Actual vs Predicted
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.savefig(os.path.join(output_path, "actual_vs_predicted.png"))
# plt.close()

# print(f"Final MAE: {mae2:.4f}")
# torch.save(model.state_dict(), "outputRegression/house_model.pth")