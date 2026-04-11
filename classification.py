import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import numpy as np


# Create output folder inside project
output_path = "output"
os.makedirs(output_path, exist_ok=True)

# Load dataset
titanic = pd.read_csv("titanic.csv")

# Inspect dataset
print(titanic.head())
print(titanic.shape)
print(titanic.info())
print(titanic.isnull().sum())

# Handle missing values
titanic['age'].fillna(titanic['age'].mean(), inplace=True)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
titanic.drop(['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], axis=1, inplace=True)

# Convert categorical → numerical
titanic = pd.get_dummies(titanic, drop_first=True)

# Split features and target
X = titanic.drop('survived', axis=1)
y = titanic['survived']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- VISUALIZATION ----------------

sns.countplot(x='survived', data=titanic)
plt.title("Survival Count")
plt.savefig("output/survival_count.png")
plt.close()

sns.histplot(titanic['age'], kde=True)
plt.title("Age Distribution")
plt.savefig("output/age_distribution.png")
plt.close()

sns.countplot(x='sex_male', hue='survived', data=titanic)
plt.title("Survival based on Gender")
plt.savefig("output/gender_survival.png")
plt.close()

plt.figure(figsize=(10,6))
sns.heatmap(titanic.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("output/heatmap.png")
plt.close()

sns.histplot(titanic['fare'], kde=True)
plt.title("Fare Distribution")
plt.savefig("output/fare_distribution.png")
plt.close()

# ---------------- ANN MODEL ----------------

model = tf.keras.models.Sequential()
Dense = tf.keras.layers.Dense

model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=16,
                    validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# ---------------- SAVE RESULTS ----------------

# Save accuracy
with open("output/results.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")

# Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)

np.savetxt("output/confusion_matrix.txt", cm, fmt="%d")

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save confusion matrix plot
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("output/confusion_matrix.png")
plt.close()

# Save predictions
pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred.flatten()
})
pred_df.to_csv("output/predictions.csv", index=False)

# Save accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(['Train', 'Validation'])
plt.savefig("output/accuracy_plot.png")
plt.close()

# Save loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(['Train', 'Validation'])
plt.savefig("output/loss_plot.png")
plt.close()

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv("output/training_history.csv", index=False)

# Save model
model.save("output/ann_model.h5")