# 🧠 PyTorch Implementation for Regression and Classification Tasks

## 📌 Overview

This project demonstrates the implementation of **Artificial Neural Networks (ANNs)** using **PyTorch** for:

* 🔹 **Classification Task** – Titanic Survival Prediction
* 🔹 **Regression Task** – House Price Prediction

The project covers the complete machine learning pipeline including:

* Data preprocessing
* Data visualization
* Model building using PyTorch
* Training and testing
* Performance evaluation
* Hyperparameter tuning

---

## 📂 Project Structure

```
ANN_ASSIGNMENT/
│
├── data/
│   ├── titanic.csv
│   ├── house.csv
│
├── classification.py
├── regression.py
│
├── output/
├── outputRegression/
│
├── README.md
```

---

## 📊 Dataset Description

### 🔹 Classification Dataset (Titanic)

* Source: Kaggle / GitHub
* Features:

  * Age, Gender, Fare, Passenger Class, etc.
* Target:

  * `survived` (0 = No, 1 = Yes)

---

### 🔹 Regression Dataset (House Prices)

* Source: Public dataset (Boston Housing)
* Features:

  * Crime rate, number of rooms, etc.
* Target:

  * `medv` (Median house value)

---

## ⚙️ Data Preprocessing

### ✔ Steps Performed:

* Handling missing values:

  * Mean (numerical features)
  * Mode (categorical features)
* Dropping irrelevant columns
* Encoding categorical variables using one-hot encoding
* Feature scaling using **StandardScaler**
* Train-test split (80% training, 20% testing)

---

## 📈 Data Visualization

### Classification:

* Survival count plot
* Age distribution
* Gender vs survival
* Correlation heatmap

### Regression:

* Price distribution
* Feature correlation heatmap
* Actual vs predicted plot
* Error distribution

---

## 🧠 Model Architecture

### 🔹 Classification Model (ANN)

* Input Layer

* Hidden Layers:

  * Dense(32) → ReLU
  * Dense(16) → ReLU
  * Dense(8) → ReLU

* Output Layer:

  * Dense(1)

* Loss Function:

  * `BCEWithLogitsLoss`

---

### 🔹 Regression Model (ANN)

* Input Layer

* Hidden Layers:

  * Dense(32) → ReLU
  * Dense(16) → ReLU

* Output Layer:

  * Dense(1)

* Loss Function:

  * `Mean Squared Error`

---

## 🏋️ Training Details

* Optimizer: Adam
* Epochs: 50 / 100
* Batch Size: 16 / 32
* Device: CPU / GPU (if available)

---

## 📊 Evaluation Metrics

### 🔹 Classification:

* Accuracy
* Confusion Matrix
* Precision, Recall, F1-score

---

### 🔹 Regression:

* Mean Absolute Error (MAE)
* Loss (MSE)

---

## ⚙️ Hyperparameter Tuning

Different configurations were tested:

| Model         | Epochs | Batch Size | Layers | Result   |
| ------------- | ------ | ---------- | ------ | -------- |
| Baseline      | 50     | 32         | 16-8   | Moderate |
| More Neurons  | 50     | 32         | 32-16  | Improved |
| More Epochs   | 100    | 32         | 32-16  | Better   |
| Smaller Batch | 50     | 16         | 32-16  | Best     |

---

## 💾 Output Files

### Classification Output:

* Accuracy results
* Confusion matrix
* Predictions CSV
* Accuracy graph
* Trained model (`.pt`)

---

### Regression Output:

* MAE results
* Predictions CSV
* Actual vs predicted plot
* Error distribution
* Model comparison graph

---

## 🚀 How to Run the Project

### 🔹 Step 1: Install Dependencies

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
```

---

### 🔹 Step 2: Run Classification

```bash
python classification.py
```

---

### 🔹 Step 3: Run Regression

```bash
python regression.py
```

---

## 📌 Key Observations

* Increasing neurons improves model performance
* Too many epochs can lead to overfitting
* Smaller batch sizes improve learning but increase training time
* Proper preprocessing significantly impacts model accuracy

---

## 📚 Technologies Used

* 🐍 Python
* 🔥 PyTorch
* 📊 Pandas, NumPy
* 📉 Matplotlib, Seaborn
* 🤖 Scikit-learn

---

## 👨‍💻 Author

**Ritik Sharma**
B.Tech CSE

---

## ✅ Conclusion

This project successfully demonstrates how **Artificial Neural Networks** can be applied to both classification and regression problems using PyTorch, along with proper preprocessing, evaluation, and tuning techniques.

---
