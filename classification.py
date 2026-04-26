# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# import os
# import numpy as np


# # Create output folder inside project
# output_path = "output"
# os.makedirs(output_path, exist_ok=True)

# # Load dataset
# titanic = pd.read_csv("data/titanic.csv")

# # Inspect dataset
# print(titanic.head())
# print(titanic.shape)
# print(titanic.info())
# print(titanic.isnull().sum())

# # Handle missing values
# titanic['age'].fillna(titanic['age'].mean(), inplace=True)
# titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# # Drop unnecessary columns
# titanic.drop(['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], axis=1, inplace=True)

# # Convert categorical → numerical
# titanic = pd.get_dummies(titanic, drop_first=True)

# # Split features and target
# X = titanic.drop('survived', axis=1)
# y = titanic['survived']

# # Train-test split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Feature scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # ---------------- VISUALIZATION ----------------

# sns.countplot(x='survived', data=titanic)
# plt.title("Survival Count")
# plt.savefig("output/survival_count.png")
# plt.close()

# sns.histplot(titanic['age'], kde=True)
# plt.title("Age Distribution")
# plt.savefig("output/age_distribution.png")
# plt.close()

# sns.countplot(x='sex_male', hue='survived', data=titanic)
# plt.title("Survival based on Gender")
# plt.savefig("output/gender_survival.png")
# plt.close()

# plt.figure(figsize=(10,6))
# sns.heatmap(titanic.corr(), annot=False, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.savefig("output/heatmap.png")
# plt.close()

# sns.histplot(titanic['fare'], kde=True)
# plt.title("Fare Distribution")
# plt.savefig("output/fare_distribution.png")
# plt.close()

# # ---------------- ANN MODEL ----------------

# model = tf.keras.models.Sequential()
# Dense = tf.keras.layers.Dense

# model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Train
# history = model.fit(X_train, y_train,
#                     epochs=50,
#                     batch_size=16,
#                     validation_split=0.2)

# # Evaluate
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.4f}")

# # ---------------- SAVE RESULTS ----------------

# # Save accuracy
# with open("output/results.txt", "w") as f:
#     f.write(f"Test Accuracy: {accuracy:.4f}\n")

# # Predictions
# y_pred = model.predict(X_test)
# y_pred = (y_pred > 0.5)

# # Confusion matrix
# from sklearn.metrics import confusion_matrix, classification_report
# cm = confusion_matrix(y_test, y_pred)

# np.savetxt("output/confusion_matrix.txt", cm, fmt="%d")

# print("Confusion Matrix:\n", cm)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Save confusion matrix plot
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title("Confusion Matrix")
# plt.savefig("output/confusion_matrix.png")
# plt.close()

# # Save predictions
# pred_df = pd.DataFrame({
#     "Actual": y_test,
#     "Predicted": y_pred.flatten()
# })
# pred_df.to_csv("output/predictions.csv", index=False)

# # Save accuracy graph
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title("Accuracy")
# plt.legend(['Train', 'Validation'])
# plt.savefig("output/accuracy_plot.png")
# plt.close()

# # Save loss graph
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title("Loss")
# plt.legend(['Train', 'Validation'])
# plt.savefig("output/loss_plot.png")
# plt.close()

# # Save training history
# history_df = pd.DataFrame(history.history)
# history_df.to_csv("output/training_history.csv", index=False)

# # Save model
# model.save("output/ann_model.h5")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. Dataset Definition
# ==========================================
class TitanicDataset(Dataset):
    """Custom PyTorch Dataset for Titanic data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Model Architecture
# ==========================================
class TitanicNN(nn.Module):
    """Neural Network for Binary Classification."""
    def __init__(self, input_size):
        super(TitanicNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1) # Raw logits output
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Preprocessing & Visualization Logic
# ==========================================
def load_and_preprocess_titanic(filepath, output_dir):
    print(f"--- Loading Titanic Dataset from {filepath} ---")
    df = pd.read_csv(filepath)

    # Handle missing values
    df['age'] = df['age'].fillna(df['age'].mean())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    # Visualizations (from first code logic)
    sns.countplot(x='survived', data=df)
    plt.title("Survival Count")
    plt.savefig(os.path.join(output_dir, "survival_count.png"))
    plt.close()

    # Drop non-numeric/unnecessary columns
    cols_to_drop = ['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df.drop(existing_drops, axis=1, inplace=True)

    # Convert categorical to numerical
    df = pd.get_dummies(df, drop_first=True)

    # Split features and target
    X = df.drop('survived', axis=1).values.astype(np.float32)
    y = df['survived'].values.astype(np.float32)

    # Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.shape[1]

# ==========================================
# 4. Training Function
# ==========================================
def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    print("\n--- Starting Training ---")
    model.train()
    history = {'loss': [], 'accuracy': []}

    for epoch in range(num_epochs):
        batch_losses = []
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
            # Accuracy calculation
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_acc = correct / total
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    return history

# ==========================================
# 5. Evaluation Function
# ==========================================
def evaluate_model(model, X_test, y_test, output_dir):
    print("\n--- Evaluation Results ---")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        y_pred = (probs > 0.5).float().numpy()

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    
    # Save Report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Titanic Survival Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

# ==========================================
# 6. Main Flow
# ==========================================
def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    data_path = "data/titanic.csv" 
    
    # 1. Preprocess
    X_train, X_test, y_train, y_test, input_size = load_and_preprocess_titanic(data_path, output_dir)

    # 2. DataLoaders
    train_dataset = TitanicDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 3. Model Setup
    model = TitanicNN(input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Training
    history = train_model(model, train_loader, criterion, optimizer, num_epochs=50)

    # 5. Evaluation
    evaluate_model(model, X_test, y_test, output_dir)

    # 6. Final Plots
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    
    # Save Model Weights
    torch.save(model.state_dict(), os.path.join(output_dir, "titanic_model.pth"))
    print(f"\nTask complete. All artifacts saved in '{output_dir}'.")

if __name__ == "__main__":
    main()