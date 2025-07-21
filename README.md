
**Title**--**💵Employee Salary Prediction Project**


---
## 👨🏼‍💻Employee salary prediction
Predict employee salary class (<=50K or >50K) using machine learning and a user-friendly Streamlit to deploy the Project.

---

## 📊 Dataset

This project uses the **Adult** dataset, which contains information about individuals from the 1994 Census database. The goal is to predict whether an individual's income is above or below $50K.

---

## 🛠️ Techniques Used

### 1. Data Loading & Initial Exploration
- Load the dataset into a pandas DataFrame
- Inspect data shape, head, and tail

### 2. Data Cleaning & Preprocessing
- Handle missing values ('?') by replacing with NaN and dropping rows
- Remove redundant columns (e.g., 'fnlwgt')
- Identify categorical and numerical features
- Standardize numerical features with StandardScaler
- Encode categorical features with OneHotEncoder
- Use ColumnTransformer for combined preprocessing

### 3. Feature Engineering
- Create 'experience_level' by binning the 'age' column
- Create 'capital_diff' by subtracting 'capital-loss' from 'capital-gain'
- Convert 'income' to binary (0 for <=50K, 1 for >50K)

### 4. Model Training & Evaluation
- Split data into training and testing sets
- Train Logistic Regression and Random Forest Classifier
- Evaluate models using Accuracy and Classification Report (Precision, Recall, F1-score)
- Select the best model based on accuracy

### 5. Model Saving
- Save the trained pipeline (preprocessing + model) using joblib for use in the Streamlit app

### 6. Feature Importance (Tree-based Models)
- Analyze and visualize feature importance in Random Forest

### 7. Streamlit Application
- Develop a web app to showcase the model
- Upload CSV for batch predictions
- Sidebar for manual user input and single prediction
- Load the saved pipeline for preprocessing and prediction

---

## 🚀 How to Use

1. **Install dependencies:**
```bash
pip install -r requirements/requirements.txt
```
2. **Train the model:**
```bash
python scripts/train_model.py
```
3. **Run the app:**
```bash
streamlit run app/app.py
```
4. **Use the app:**
- Upload a CSV file for batch predictions
- Enter employee details in the sidebar for single prediction

---

## 🌐 Deployment

- Host on [Streamlit Cloud](https://streamlit.io/cloud) for a public URL
- Share the app link on GitHub, LinkedIn, etc.

---

