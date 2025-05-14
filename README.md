
# 🚢 Titanic Survival Prediction Using Naive Bayes

This project aims to build a **machine learning model** that predicts whether a passenger survived the Titanic disaster, using the **Naive Bayes classifier**. The project involves exploratory data analysis (EDA), model training, evaluation using key metrics, and hyperparameter tuning for performance optimization.

---

## 🎯 Objective

To predict **survival** on the Titanic using a classification model based on passenger attributes like age, sex, class, fare, etc.

---

## 🧾 Dataset

The dataset used is the **Titanic dataset** from [Kaggle](https://www.kaggle.com/c/titanic/data) or available through `Seaborn` or `sklearn`.

### Key Features:

* `PassengerId`
* `Pclass` (Ticket class)
* `Name`
* `Sex`
* `Age`
* `SibSp` (Number of siblings/spouses aboard)
* `Parch` (Number of parents/children aboard)
* `Ticket`
* `Fare`
* `Cabin`
* `Embarked` (Port of Embarkation)
* `Survived` (Target variable: 0 = No, 1 = Yes)

---

## 🛠️ Tools & Libraries

* Python
* Pandas, NumPy
* Matplotlib, Seaborn (for visualization)
* Scikit-learn (for modeling and evaluation)

---

## 🔍 Project Workflow

### 1. 📊 Exploratory Data Analysis (EDA)

* Understand data distributions
* Visualize survival rates across different features (e.g., gender, class, age)
* Handle missing values and encode categorical variables

### 2. 🧠 Apply Naive Bayes Classifier

* Train a **Gaussian Naive Bayes** model (or other suitable variant)
* Use features selected after preprocessing
* Split data into training and testing sets

### 3. 📈 Model Evaluation

Evaluate model performance using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

### 4. ⚙️ Hyperparameter Tuning

Although Naive Bayes has fewer hyperparameters, fine-tuning (e.g., feature selection, binning strategies) can improve performance. Techniques may include:

* Grid search
* Cross-validation
* Feature engineering and scaling experiments


## ✅ Results

After training and tuning, results will be displayed using:

* **Confusion matrix**
* **Classification report**
* **Evaluation metrics**


