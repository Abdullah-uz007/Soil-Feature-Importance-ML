# 🌾 Crop Prediction Using Single Soil Feature (Feature Selection Project)

## 📌 Project Overview

In real-world agriculture, measuring soil nutrients can be expensive.

A farmer approached with a constraint:

> He could only afford to measure **one** soil feature out of:
>
> * Nitrogen (N)
> * Phosphorous (P)
> * Potassium (K)
> * pH value

The objective of this project was to:

🎯 Identify the **single most predictive soil feature** for classifying the optimal crop.

This problem represents a **feature selection challenge under budget constraints**.

---

## 📊 Dataset Description

The dataset `soil_measures.csv` contains:

| Feature | Description                         |
| ------- | ----------------------------------- |
| N       | Nitrogen content ratio in soil      |
| P       | Phosphorous content ratio in soil   |
| K       | Potassium content ratio in soil     |
| pH      | Soil pH value                       |
| crop    | Optimal crop type (Target Variable) |

* Multi-class classification problem
* Each row represents a field's soil measurements
* Goal: Predict `crop`

---

## 🧠 Approach

### 1️⃣ Problem Type

Multi-class Classification

### 2️⃣ Model Used

* Logistic Regression (Multinomial)

### 3️⃣ Evaluation Metric

* **Weighted F1-Score**

  * Chosen because dataset is multi-class
  * Balances precision and recall

### 4️⃣ Methodology

* Split data into training and testing sets (80/20)
* Train separate models using only **one feature at a time**
* Compare F1-scores
* Select the best-performing feature

---

## 💻 Core Implementation

```python
# Train a logistic regression model for each feature
feature_performance = {}

for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    
    f1 = metrics.f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = f1

best_predictive_feature = {
    max(feature_performance, key=feature_performance.get):
    max(feature_performance.values())
}
```

---

## 📈 Results

After evaluating each feature independently:

🏆 **Potassium (K)** achieved the highest weighted F1-score.

```python
best_predictive_feature = {"K": feature_performance["K"]}
```

This means that if the farmer can measure only one feature, **Potassium content provides the strongest predictive power**.

---

## 📊 Example Output

```
F1-score for N: 0.51
F1-score for P: 0.47
F1-score for K: 0.63
F1-score for pH: 0.39
```

Best Feature:

```
{'K': 0.63}
```

---

## 🚀 Key Learnings

* Practical Feature Selection
* Multi-class classification
* Model evaluation using F1-score
* Applying ML under real-world constraints
* Cost-aware machine learning modeling

---

## 🔮 Future Improvements

* Add Cross-Validation
* Compare with Decision Trees / Random Forest
* Perform feature scaling
* Add confusion matrix visualization
* Deploy as a simple Streamlit app

---

## 📂 How to Run

```bash
pip install pandas scikit-learn
python main.py
```

---

## 👨‍💻 Author

Abdullah
Machine Learning Enthusiast

