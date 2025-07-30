# 📊 Customer Churn Prediction

A complete machine learning project that predicts whether a telecom customer is likely to churn based on their usage patterns, contract type, and service preferences. This project demonstrates data preprocessing, exploratory analysis, model training, evaluation, and feature importance visualization.

---

## 🔍 Features

- ✅ Data cleaning & preprocessing  
- 📊 Exploratory Data Analysis (EDA)  
- 🤖 Random Forest classifier  
- 📈 Feature importance visualization  
- 📦 Virtual environment setup  
- 🧪 Model evaluation with classification report  

---

## 📁 Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- **Format**: CSV  
- **Target Variable**: `Churn` (Yes/No)

---

## 📂 Project Structure

customer_churn_prediction/
│
├── churn_model.py # Main ML pipeline (EDA, model, visualization)
├── requirements.txt # Python dependencies
├── data/
│ └── Telco-Customer-Churn.csv
├── feature_importance.png # Bar plot of top features
└── README.md # Project overview


### 📈 Sample Output

<img width="540" height="250" alt="image" src="https://github.com/user-attachments/assets/6024c942-1fdf-42e2-881a-ae611735fe55" />

### 📌 Future Improvements

- Try XGBoost or Gradient Boosting for better accuracy
- Add hyperparameter tuning using GridSearchCV
- Deploy model using Flask + Streamlit or FastAPI
- Explain predictions using SHAP

####👨‍💻 Built by [Mitesh S.](https://github.com/Mitsu007) – Aspiring AI/Data Science Master's student | Passionate about ML, data storytelling, and AI systems.
---

## 🚀 Getting Started

### 🛠️ Clone & Setup

```bash
# Clone the repository
git clone https://github.com/Mitsu007/customer_churn_prediction.git
cd customer_churn_prediction

# Create virtual environment
pip install virtualenv
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the project
python churn_model.py





