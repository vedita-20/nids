# ============================================================
# NETWORK INTRUSION DETECTION SYSTEM (NIDS)
# CICIDS2017 - Binary Classification
#
# Models:
# 1. Random Forest
# 2. KNN
# 3. SVM
# 4. XGBoost
# 5. LSTM
# 6. GRU
#
# Outputs:
# - Model comparison
# - Class-wise Precision / Recall / F1
# - Normal Error %
# - Attack Error %
# ============================================================

import pandas as pd
import numpy as np
import time
from pathlib import Path

# ============================================================
# SKLEARN
# ============================================================

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ============================================================
# XGBOOST
# ============================================================

from xgboost import XGBClassifier

# ============================================================
# TENSORFLOW / KERAS
# ============================================================

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Dropout
)

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

# Number of features after Mutual Information
TOP_K = 15

# Faster cross-validation
CV_FOLDS = 3

# Maximum samples used for LSTM / GRU training
RNN_SAMPLES = 50000

# Reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 2. FIND DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

csv_files = list(BASE_DIR.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV dataset found in the project folder."
    )

DATA_PATH = csv_files[0]

print("\nDataset found:")
print(DATA_PATH)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ============================================================
# 4. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("Number of columns:", len(df.columns))


# ============================================================
# 5. REMOVE INFINITY
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 6. CHECK TARGET
# ============================================================

if "Label" not in df.columns:

    raise ValueError(
        "Label column not found in dataset."
    )


# ============================================================
# 7. REMOVE ROWS WHERE LABEL IS MISSING
# ============================================================

df = df.dropna(
    subset=["Label"]
)


# ============================================================
# 8. CREATE BINARY LABEL
# ============================================================
#
# BENIGN = 0 = NORMAL
#
# Everything else = 1 = ATTACK
#
# ============================================================

y_original = (
    df["Label"]
    .astype(str)
    .str.strip()
)

y = np.where(
    y_original.str.upper() == "BENIGN",
    0,
    1
)

y = pd.Series(
    y,
    index=df.index
)


print("\nClass distribution:")

print(
    "Normal (0):",
    np.sum(y == 0)
)

print(
    "Attack (1):",
    np.sum(y == 1)
)


# ============================================================
# 9. REMOVE LABEL
# ============================================================

X = df.drop(
    columns=["Label"]
)


# ============================================================
# 10. REMOVE IDENTIFIER COLUMNS
# ============================================================

columns_to_remove = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp"
]

existing_columns = [
    col
    for col in columns_to_remove
    if col in X.columns
]

X = X.drop(
    columns=existing_columns
)

print("\nRemoved identifier columns:")

print(existing_columns)


# ============================================================
# 11. CONVERT FEATURES TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# 12. REMOVE INF AGAIN
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 13. REMOVE COMPLETELY EMPTY COLUMNS
# ============================================================

X = X.dropna(
    axis=1,
    how="all"
)


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nTrain shape:", X_train.shape)

print("Test shape:", X_test.shape)


# ============================================================
# 15. MEDIAN IMPUTATION
# ============================================================
#
# IMPORTANT:
# Median is calculated ONLY from training data.
#
# This avoids test-set information leaking into training.
#
# ============================================================

train_medians = X_train.median()

X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


# ============================================================
# 16. VARIANCE FILTER
# ============================================================

variance_filter = VarianceThreshold(
    threshold=0.01
)

X_train_var = variance_filter.fit_transform(
    X_train
)

X_test_var = variance_filter.transform(
    X_test
)


print(
    "\nFeatures after variance filtering:",
    X_train_var.shape[1]
)


# ============================================================
# 17. CORRELATION FILTER
# ============================================================

X_train_var_df = pd.DataFrame(
    X_train_var,

    columns=[
        f"feature_{i}"
        for i in range(
            X_train_var.shape[1]
        )
    ]
)

X_test_var_df = pd.DataFrame(
    X_test_var,

    columns=X_train_var_df.columns
)


corr_matrix = (
    X_train_var_df
    .corr()
    .abs()
)


upper_triangle = corr_matrix.where(
    np.triu(
        np.ones(
            corr_matrix.shape
        ),
        k=1
    ).astype(bool)
)


high_corr_columns = [

    column

    for column in upper_triangle.columns

    if any(
        upper_triangle[column] > 0.95
    )

]


X_train_corr = (
    X_train_var_df
    .drop(
        columns=high_corr_columns
    )
)

X_test_corr = (
    X_test_var_df
    .drop(
        columns=high_corr_columns
    )
)


print(
    "Features after correlation filtering:",
    X_train_corr.shape[1]
)


# ============================================================
# 18. MUTUAL INFORMATION
# ============================================================

print(
    "\nRunning Mutual Information..."
)


mi_scores = mutual_info_classif(

    X_train_corr,

    y_train,

    random_state=RANDOM_STATE
)


mi_df = pd.DataFrame({

    "Feature":
    X_train_corr.columns,

    "MI_Score":
    mi_scores

})


mi_df = mi_df.sort_values(

    by="MI_Score",

    ascending=False

)


actual_top_k = min(

    TOP_K,

    X_train_corr.shape[1]

)


selected_features = (

    mi_df
    .head(actual_top_k)
    ["Feature"]
    .tolist()

)


X_train_selected = (

    X_train_corr[
        selected_features
    ]

)

X_test_selected = (

    X_test_corr[
        selected_features
    ]

)


print(
    "\nFinal number of selected features:",
    len(selected_features)
)


print("\nSelected features:")

for feature in selected_features:

    print(
        feature
    )


# ============================================================
# 19. STANDARDIZATION
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train_selected
)


X_test_scaled = scaler.transform(
    X_test_selected
)


# ============================================================
# 20. RESULT STORAGE
# ============================================================

results = []

classwise_results = []


# ============================================================
# 21. CLASS-WISE METRIC FUNCTION
# ============================================================

def calculate_classwise_metrics(
    model_name,
    y_true,
    y_pred
):

    precision = precision_score(

        y_true,

        y_pred,

        labels=[0, 1],

        average=None,

        zero_division=0
    )


    recall = recall_score(

        y_true,

        y_pred,

        labels=[0, 1],

        average=None,

        zero_division=0
    )


    f1 = f1_score(

        y_true,

        y_pred,

        labels=[0, 1],

        average=None,

        zero_division=0
    )


    # Confusion matrix
    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]

    )


    TN, FP, FN, TP = cm.ravel()


    # --------------------------------------------------------
    # NORMAL ERROR
    # --------------------------------------------------------

    normal_error = (

        FP / (TN + FP) * 100

        if (TN + FP) > 0

        else 0

    )


    # --------------------------------------------------------
    # ATTACK ERROR
    # --------------------------------------------------------

    attack_error = (

        FN / (FN + TP) * 100

        if (FN + TP) > 0

        else 0

    )


    classwise_results.append({

        "Model":
        model_name,

        "Precision (Normal)":
        precision[0],

        "Precision (Attack)":
        precision[1],

        "Recall (Normal)":
        recall[0],

        "Recall (Attack)":
        recall[1],

        "F1 Score (Normal)":
        f1[0],

        "F1 Score (Attack)":
        f1[1],

        "Error % (Normal)":
        normal_error,

        "Error % (Attack)":
        attack_error

    })


# ============================================================
# 22. RANDOM FOREST
# ============================================================

print("\n")
print("=" * 70)
print("RANDOM FOREST")
print("=" * 70)


rf = RandomForestClassifier(

    random_state=RANDOM_STATE,

    class_weight="balanced",

    n_jobs=-1

)


rf_params = {

    "n_estimators":
    [100],

    "max_depth":
    [8, 12],

    "min_samples_split":
    [10],

    "min_samples_leaf":
    [5],

    "max_features":
    ["sqrt"],

    "max_samples":
    [0.8]

}


start_time = time.time()


rf_grid = GridSearchCV(

    rf,

    rf_params,

    cv=CV_FOLDS,

    scoring="f1_macro",

    n_jobs=-1

)


rf_grid.fit(

    X_train_selected,

    y_train

)


rf_time = (
    time.time()
    - start_time
)


rf_best = (
    rf_grid.best_estimator_
)


rf_train_pred = (
    rf_best.predict(
        X_train_selected
    )
)


rf_test_pred = (
    rf_best.predict(
        X_test_selected
    )
)


rf_train_acc = accuracy_score(

    y_train,

    rf_train_pred

)


rf_test_acc = accuracy_score(

    y_test,

    rf_test_pred

)


rf_precision = precision_score(

    y_test,

    rf_test_pred,

    average="macro",

    zero_division=0

)


rf_recall = recall_score(

    y_test,

    rf_test_pred,

    average="macro",

    zero_division=0

)


rf_f1 = f1_score(

    y_test,

    rf_test_pred,

    average="macro",

    zero_division=0

)


results.append({

    "Model":
    "Random Forest",

    "Train Accuracy":
    rf_train_acc,

    "Test Accuracy":
    rf_test_acc,

    "Overfitting Gap":
    rf_train_acc - rf_test_acc,

    "Precision":
    rf_precision,

    "Recall":
    rf_recall,

    "F1 Score":
    rf_f1,

    "Training Time (sec)":
    rf_time,

    "Best Parameters":
    str(
        rf_grid.best_params_
    )

})


calculate_classwise_metrics(

    "Random Forest",

    y_test,

    rf_test_pred

)


print(
    "Best parameters:",
    rf_grid.best_params_
)

print(
    "Train Accuracy:",
    rf_train_acc
)

print(
    "Test Accuracy:",
    rf_test_acc
)


# ============================================================
# 23. KNN
# ============================================================

print("\n")
print("=" * 70)
print("KNN")
print("=" * 70)


knn = KNeighborsClassifier()


knn_params = {

    "n_neighbors":
    [15, 25],

    "weights":
    ["uniform"]

}


start_time = time.time()


knn_grid = GridSearchCV(

    knn,

    knn_params,

    cv=CV_FOLDS,

    scoring="f1_macro",

    n_jobs=-1

)


knn_grid.fit(

    X_train_scaled,

    y_train

)


knn_time = (
    time.time()
    - start_time
)


knn_best = (
    knn_grid.best_estimator_
)


knn_train_pred = (
    knn_best.predict(
        X_train_scaled
    )
)


knn_test_pred = (
    knn_best.predict(
        X_test_scaled
    )
)


knn_train_acc = accuracy_score(

    y_train,

    knn_train_pred

)


knn_test_acc = accuracy_score(

    y_test,

    knn_test_pred

)


knn_precision = precision_score(

    y_test,

    knn_test_pred,

    average="macro",

    zero_division=0

)


knn_recall = recall_score(

    y_test,

    knn_test_pred,

    average="macro",

    zero_division=0

)


knn_f1 = f1_score(

    y_test,

    knn_test_pred,

    average="macro",

    zero_division=0

)


results.append({

    "Model":
    "KNN",

    "Train Accuracy":
    knn_train_acc,

    "Test Accuracy":
    knn_test_acc,

    "Overfitting Gap":
    knn_train_acc - knn_test_acc,

    "Precision":
    knn_precision,

    "Recall":
    knn_recall,

    "F1 Score":
    knn_f1,

    "Training Time (sec)":
    knn_time,

    "Best Parameters":
    str(
        knn_grid.best_params_
    )

})


calculate_classwise_metrics(

    "KNN",

    y_test,

    knn_test_pred

)


print(
    "Best parameters:",
    knn_grid.best_params_
)

print(
    "Train Accuracy:",
    knn_train_acc
)

print(
    "Test Accuracy:",
    knn_test_acc
)


# ============================================================
# 24. SVM
# ============================================================

print("\n")
print("=" * 70)
print("SVM")
print("=" * 70)


svm = SVC(

    class_weight="balanced"

)


svm_params = {

    "C":
    [0.1, 1],

    "kernel":
    ["rbf"],

    "gamma":
    ["scale"]

}


start_time = time.time()


svm_grid = GridSearchCV(

    svm,

    svm_params,

    cv=CV_FOLDS,

    scoring="f1_macro",

    n_jobs=-1

)


svm_grid.fit(

    X_train_scaled,

    y_train

)


svm_time = (
    time.time()
    - start_time
)


svm_best = (
    svm_grid.best_estimator_
)


svm_train_pred = (
    svm_best.predict(
        X_train_scaled
    )
)


svm_test_pred = (
    svm_best.predict(
        X_test_scaled
    )
)


svm_train_acc = accuracy_score(

    y_train,

    svm_train_pred

)


svm_test_acc = accuracy_score(

    y_test,

    svm_test_pred

)


svm_precision = precision_score(

    y_test,

    svm_test_pred,

    average="macro",

    zero_division=0

)


svm_recall = recall_score(

    y_test,

    svm_test_pred,

    average="macro",

    zero_division=0

)


svm_f1 = f1_score(

    y_test,

    svm_test_pred,

    average="macro",

    zero_division=0

)


results.append({

    "Model":
    "SVM",

    "Train Accuracy":
    svm_train_acc,

    "Test Accuracy":
    svm_test_acc,

    "Overfitting Gap":
    svm_train_acc - svm_test_acc,

    "Precision":
    svm_precision,

    "Recall":
    svm_recall,

    "F1 Score":
    svm_f1,

    "Training Time (sec)":
    svm_time,

    "Best Parameters":
    str(
        svm_grid.best_params_
    )

})


calculate_classwise_metrics(

    "SVM",

    y_test,

    svm_test_pred

)


print(
    "Best parameters:",
    svm_grid.best_params_
)

print(
    "Train Accuracy:",
    svm_train_acc
)

print(
    "Test Accuracy:",
    svm_test_acc
)


# ============================================================
# 25. XGBOOST
# ============================================================

print("\n")
print("=" * 70)
print("XGBOOST")
print("=" * 70)


xgb = XGBClassifier(

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1

)


xgb_params = {

    "n_estimators":
    [50, 100],

    "max_depth":
    [2, 3],

    "learning_rate":
    [0.05],

    "subsample":
    [0.7],

    "colsample_bytree":
    [0.7],

    "min_child_weight":
    [5],

    "reg_alpha":
    [0.1],

    "reg_lambda":
    [2]

}


start_time = time.time()


xgb_grid = GridSearchCV(

    xgb,

    xgb_params,

    cv=CV_FOLDS,

    scoring="f1_macro",

    n_jobs=-1

)


xgb_grid.fit(

    X_train_selected,

    y_train

)


xgb_time = (
    time.time()
    - start_time
)


xgb_best = (
    xgb_grid.best_estimator_
)


xgb_train_pred = (
    xgb_best.predict(
        X_train_selected
    )
)


xgb_test_pred = (
    xgb_best.predict(
        X_test_selected
    )
)


xgb_train_acc = accuracy_score(

    y_train,

    xgb_train_pred

)


xgb_test_acc = accuracy_score(

    y_test,

    xgb_test_pred

)


xgb_precision = precision_score(

    y_test,

    xgb_test_pred,

    average="macro",

    zero_division=0

)


xgb_recall = recall_score(

    y_test,

    xgb_test_pred,

    average="macro",

    zero_division=0

)


xgb_f1 = f1_score(

    y_test,

    xgb_test_pred,

    average="macro",

    zero_division=0

)


results.append({

    "Model":
    "XGBoost",

    "Train Accuracy":
    xgb_train_acc,

    "Test Accuracy":
    xgb_test_acc,

    "Overfitting Gap":
    xgb_train_acc - xgb_test_acc,

    "Precision":
    xgb_precision,

    "Recall":
    xgb_recall,

    "F1 Score":
    xgb_f1,

    "Training Time (sec)":
    xgb_time,

    "Best Parameters":
    str(
        xgb_grid.best_params_
    )

})


calculate_classwise_metrics(

    "XGBoost",

    y_test,

    xgb_test_pred

)


print(
    "Best parameters:",
    xgb_grid.best_params_
)

print(
    "Train Accuracy:",
    xgb_train_acc
)

print(
    "Test Accuracy:",
    xgb_test_acc
)


# ============================================================
# 26. PREPARE LSTM / GRU DATA
# ============================================================

print("\n")
print("=" * 70)
print("PREPARING LSTM / GRU DATA")
print("=" * 70)


X_train_rnn = (
    X_train_scaled.reshape(
        X_train_scaled.shape[0],
        X_train_scaled.shape[1],
        1
    )
)


X_test_rnn = (
    X_test_scaled.reshape(
        X_test_scaled.shape[0],
        X_test_scaled.shape[1],
        1
    )
)


print(
    "Full RNN training shape:",
    X_train_rnn.shape
)


# ============================================================
# 27. CREATE SMALL RNN TRAINING SET
# ============================================================

actual_rnn_samples = min(

    RNN_SAMPLES,

    len(X_train_rnn)

)


rng = np.random.RandomState(
    RANDOM_STATE
)


rnn_indices = rng.choice(

    len(X_train_rnn),

    size=actual_rnn_samples,

    replace=False

)


X_train_rnn_small = (
    X_train_rnn[
        rnn_indices
    ]
)


y_train_rnn_small = (
    np.asarray(
        y_train
    )[rnn_indices]
)


print(
    "RNN training samples:",
    actual_rnn_samples
)


# ============================================================
# 28. LSTM MODEL
# ============================================================

def build_lstm(input_shape):

    model = Sequential([

        LSTM(

            32,

            input_shape=input_shape,

            return_sequences=False,

            kernel_regularizer=l2(
                0.001
            ),

            recurrent_regularizer=l2(
                0.001
            )

        ),

        Dropout(0.4),

        Dense(

            16,

            activation="relu",

            kernel_regularizer=l2(
                0.001
            )

        ),

        Dropout(0.3),

        Dense(

            1,

            activation="sigmoid"

        )

    ])


    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=0.001

        ),

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )


    return model


# ============================================================
# 29. LSTM
# ============================================================

print("\n")
print("=" * 70)
print("LSTM")
print("=" * 70)


lstm_model = build_lstm(

    (

        X_train_rnn.shape[1],

        X_train_rnn.shape[2]

    )

)


lstm_early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


lstm_reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001

)


start_time = time.time()


lstm_history = lstm_model.fit(

    X_train_rnn_small,

    y_train_rnn_small,

    validation_split=0.20,

    epochs=15,

    batch_size=256,

    callbacks=[

        lstm_early_stopping,

        lstm_reduce_lr

    ],

    verbose=1

)


lstm_time = (
    time.time()
    - start_time
)


# ============================================================
# LSTM PREDICTIONS
# ============================================================

lstm_train_prob = (
    lstm_model.predict(

        X_train_rnn,

        batch_size=512,

        verbose=0

    )
)


lstm_test_prob = (
    lstm_model.predict(

        X_test_rnn,

        batch_size=512,

        verbose=0

    )
)


lstm_train_pred = (

    lstm_train_prob >= 0.5

).astype(int).ravel()


lstm_test_pred = (

    lstm_test_prob >= 0.5

).astype(int).ravel()


lstm_train_acc = accuracy_score(

    y_train,

    lstm_train_pred

)


lstm_test_acc = accuracy_score(

    y_test,

    lstm_test_pred

)


lstm_precision = precision_score(

    y_test,

    lstm_test_pred,

    average="macro",

    zero_division=0

)


lstm_recall = recall_score(

    y_test,

    lstm_test_pred,

    average="macro",

    zero_division=0

)


lstm_f1 = f1_score(

    y_test,

    lstm_test_pred,

    average="macro",

    zero_division=0

)


results.append({

    "Model":
    "LSTM",

    "Train Accuracy":
    lstm_train_acc,

    "Test Accuracy":
    lstm_test_acc,

    "Overfitting Gap":
    lstm_train_acc - lstm_test_acc,

    "Precision":
    lstm_precision,

    "Recall":
    lstm_recall,

    "F1 Score":
    lstm_f1,

    "Training Time (sec)":
    lstm_time,

    "Best Parameters":
    "32 units + Dropout + L2 + EarlyStopping"

})


calculate_classwise_metrics(

    "LSTM",

    y_test,

    lstm_test_pred

)


print(
    "Train Accuracy:",
    lstm_train_acc
)

print(
    "Test Accuracy:",
    lstm_test_acc
)


# ============================================================
# 30. GRU MODEL
# ============================================================

def build_gru(input_shape):

    model = Sequential([

        GRU(

            32,

            input_shape=input_shape,

            return_sequences=False,

            kernel_regularizer=l2(
                0.001
            ),

            recurrent_regularizer=l2(
                0.001
            )

        ),

        Dropout(0.4),

        Dense(

            16,

            activation="relu",

            kernel_regularizer=l2(
                0.001
            )

        ),

        Dropout(0.3),

        Dense(

            1,

            activation="sigmoid"

        )

    ])


    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=0.001

        ),

        loss="binary_crossentropy",

        metrics=["accuracy"]

    )


    return model


# ============================================================
# 31. GRU
# ============================================================

print("\n")
print("=" * 70)
print("GRU")
print("=" * 70)


gru_model = build_gru(

    (

        X_train_rnn.shape[1],

        X_train_rnn.shape[2]

    )

)


gru_early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


gru_reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001

)


start_time = time.time()


gru_history = gru_model.fit(

    X_train_rnn_small,

    y_train_rnn_small,

    validation_split=0.20,

    epochs=15,

    batch_size=256,

    callbacks=[

        gru_early_stopping,

        gru_reduce_lr

    ],

    verbose=1

)


gru_time = (
    time.time()
    - start_time
)


# ============================================================
# GRU PREDICTIONS
# ============================================================

gru_train_prob = (
    gru_model.predict(

        X_train_rnn,

        batch_size=512,

        verbose=0

    )
)


gru_test_prob = (
    gru_model.predict(

        X_test_rnn,

        batch_size=512,

        verbose=0

    )
)


gru_train_pred = (

    gru_train_prob >= 0.5

).astype(int).ravel()


gru_test_pred = (

    gru_test_prob >= 0.5

).astype(int).ravel()


gru_train_acc = accuracy_score(

    y_train,

    gru_train_pred

)


gru_test_acc = accuracy_score(

    y_test,

    gru_test_pred

)


gru_precision = precision_score(

    y_test,

    gru_test_pred,

    average="macro",

    zero_division=0

)


gru_recall = recall_score(

    y_test,

    gru_test_pred,

    average="macro",

    zero_division=0

)


gru_f1 = f1_score(

    y_test,

    gru_test_pred,

    average="macro",

    zero_division=0

)


results.append({

    "Model":
    "GRU",

    "Train Accuracy":
    gru_train_acc,

    "Test Accuracy":
    gru_test_acc,

    "Overfitting Gap":
    gru_train_acc - gru_test_acc,

    "Precision":
    gru_precision,

    "Recall":
    gru_recall,

    "F1 Score":
    gru_f1,

    "Training Time (sec)":
    gru_time,

    "Best Parameters":
    "32 units + Dropout + L2 + EarlyStopping"

})


calculate_classwise_metrics(

    "GRU",

    y_test,

    gru_test_pred

)


print(
    "Train Accuracy:",
    gru_train_acc
)

print(
    "Test Accuracy:",
    gru_test_acc
)


# ============================================================
# 32. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(

    by="F1 Score",

    ascending=False

)


print("\n")
print("=" * 110)
print("MODEL COMPARISON")
print("=" * 110)


print(

    results_df[

        [

            "Model",

            "Train Accuracy",

            "Test Accuracy",

            "Overfitting Gap",

            "Precision",

            "Recall",

            "F1 Score",

            "Training Time (sec)"

        ]

    ].to_string(index=False)

)


# ============================================================
# 33. CLASS-WISE ERROR TABLE
# ============================================================

classwise_df = pd.DataFrame(
    classwise_results
)


print("\n")
print("=" * 130)
print("CLASS-WISE PERFORMANCE AND ERROR PERCENTAGE")
print("=" * 130)


print(

    classwise_df.to_string(
        index=False
    )

)


# ============================================================
# 34. ROUND VALUES
# ============================================================

results_export = (
    results_df.copy()
)


result_numeric_columns = [

    "Train Accuracy",

    "Test Accuracy",

    "Overfitting Gap",

    "Precision",

    "Recall",

    "F1 Score",

    "Training Time (sec)"

]


for column in result_numeric_columns:

    results_export[column] = (

        results_export[column]
        .round(4)

    )


classwise_export = (
    classwise_df.copy()
)


classwise_numeric_columns = [

    "Precision (Normal)",

    "Precision (Attack)",

    "Recall (Normal)",

    "Recall (Attack)",

    "F1 Score (Normal)",

    "F1 Score (Attack)",

    "Error % (Normal)",

    "Error % (Attack)"

]


for column in classwise_numeric_columns:

    classwise_export[column] = (

        classwise_export[column]
        .round(4)

    )


# ============================================================
# 35. SAVE CSV FILES
# ============================================================

results_file = (

    BASE_DIR
    / "NIDS_Model_Comparison.csv"

)


error_file = (

    BASE_DIR
    / "NIDS_Classwise_Error_Percentage.csv"

)


results_export.to_csv(

    results_file,

    index=False

)


classwise_export.to_csv(

    error_file,

    index=False

)


# ============================================================
# 36. BEST MODEL
# ============================================================

best_model = (

    results_df.iloc[0]["Model"]

)


print("\n")
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)


print(
    "Best Model:",
    best_model
)


print("\nFiles created:")

print(
    results_file
)

print(
    error_file
)


print("\nTraining completed successfully.")