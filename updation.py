import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


project_folder = Path(__file__).parent
csv_files = list(project_folder.rglob("*.csv"))
csv_path = csv_files[0]
df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print(df.head())

df.columns = df.columns.str.strip()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Shape after cleaning:", df.shape)

target_column = "Label"

X = df.drop(columns=[target_column])
y = df[target_column]

columns_to_remove = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

columns_to_remove = [
    col for col in columns_to_remove
    if col in X.columns
]

X.drop(columns=columns_to_remove, inplace=True)

X = X.apply(pd.to_numeric, errors="coerce")

X.dropna(axis=1, how="all", inplace=True)

X.fillna(X.median(numeric_only=True), inplace=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nClasses:")
print(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape)
print("Testing samples:", X_test.shape)

variance_selector = VarianceThreshold(
    threshold=0.01
)

X_train_var = variance_selector.fit_transform(X_train)
X_test_var = variance_selector.transform(X_test)

selected_variance_features = X_train.columns[
    variance_selector.get_support()
]

print(
    "\nFeatures after variance filtering:",
    len(selected_variance_features)
)

X_train_var_df = pd.DataFrame(
    X_train_var,
    columns=selected_variance_features,
    index=X_train.index
)

X_test_var_df = pd.DataFrame(
    X_test_var,
    columns=selected_variance_features,
    index=X_test.index
)

correlation_matrix = X_train_var_df.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

correlated_features = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.95)
]

print(
    "Highly correlated features removed:",
    len(correlated_features)
)

X_train_filtered = X_train_var_df.drop(
    columns=correlated_features
)

X_test_filtered = X_test_var_df.drop(
    columns=correlated_features
)

print(
    "Features after correlation filtering:",
    X_train_filtered.shape[1]
)

print("\nCalculating Mutual Information...")

mi_scores = mutual_info_classif(
    X_train_filtered,
    y_train,
    random_state=42
)

mi_scores = pd.Series(
    mi_scores,
    index=X_train_filtered.columns
)

mi_scores = mi_scores.sort_values(
    ascending=False
)

print("\nTop 20 features according to Mutual Information:")
print(mi_scores.head(20))

K = 30

top_features = mi_scores.head(K).index.tolist()

print("\nSelected features:")
print(top_features)

X_train_selected = X_train_filtered[
    top_features
]

X_test_selected = X_test_filtered[
    top_features
]

print(
    "\nFinal number of features:",
    X_train_selected.shape[1]
)

models = {

    "Logistic Regression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier",
             LogisticRegression(max_iter=2000))
        ]),
        "params": {
            "classifier__C": [0.1, 1, 10],
            "classifier__solver": ["lbfgs"]
        }
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(
            random_state=42
        ),
        "params": {
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        }
    },

    "KNN": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier",
             KNeighborsClassifier())
        ]),
        "params": {
            "classifier__n_neighbors": [3, 5, 7],
            "classifier__weights": [
                "uniform",
                "distance"
            ]
        }
    },

    "SVM": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC())
        ]),
        "params": {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": [
                "linear",
                "rbf"
            ]
        }
    },

    "XGBoost": {
        "model": XGBClassifier(
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1]
        }
    }
}

results = []
best_models = {}

for name, config in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    start_time = time.time()

    grid = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(
        X_train_selected,
        y_train
    )

    training_time = time.time() - start_time

    best_model = grid.best_estimator_

    best_models[name] = best_model

    y_pred = best_model.predict(
        X_test_selected
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (sec)": training_time,
        "Best Parameters": grid.best_params_
    })

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest CV Score:")
    print(grid.best_score_)

    print("\nTest Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("Test F1:", f1)

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n\nFINAL MODEL COMPARISON")
print("=" * 80)

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Training Time (sec)"
        ]
    ].to_string(index=False)
)

best_model_name = results_df.iloc[0]["Model"]

print("\n" + "=" * 60)
print("BEST MODEL:", best_model_name)
print("=" * 60)

print(
    best_models[best_model_name]
)