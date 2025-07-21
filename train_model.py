import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 🔹 Load dataset
df = pd.read_csv(r"adult 3.csv")
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 🔹 Feature engineering
df["experience_level"] = pd.cut(df["age"], bins=[15, 25, 35, 50, 65, 100],
                                labels=["Entry", "Junior", "Mid", "Senior", "Executive"])
df["capital_diff"] = df["capital-gain"] - df["capital-loss"]
df["income"] = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

# 🔹 Input/output split
X = df.drop(columns=["income", "fnlwgt"])
y = df["income"]

# 🔹 Feature types
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 🔹 Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# 🔹 Model options
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=42)
}

# 🔹 Train & evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_score = 0
best_model_name = ""
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"{name} accuracy: {score:.4f}")

    if score > best_score:
        best_score = score
        best_model_name = name
        best_pipeline = pipeline

# 🔹 Save pipeline
joblib.dump(pipeline, "model.pkl")
print(f"\nSaved best model '{best_model_name}' to best_model.pkl")
