import pandas as pd
import numpy as np
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    GRU,
    LSTM
)
from tensorflow.keras.callbacks import EarlyStopping


project_folder = Path(__file__).parent
csv_files = list(project_folder.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found in {project_folder} or its subdirectories."
    )

csv_path = csv_files[0]

print("Loading dataset:", csv_path)

df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)

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


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_selected
)

X_test_scaled = scaler.transform(
    X_test_selected
)


X_train_dl = X_train_scaled.reshape(
    X_train_scaled.shape[0],
    X_train_scaled.shape[1],
    1
)

X_test_dl = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    X_test_scaled.shape[1],
    1
)

num_classes = len(
    np.unique(y_train)
)


def evaluate_model(name, model):

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    start_time = time.time()

    model.fit(
        X_train_dl,
        y_train,
        validation_split=0.20,
        epochs=20,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    training_time = time.time() - start_time

    y_pred_probs = model.predict(
        X_test_dl,
        verbose=0
    )

    y_pred = np.argmax(
        y_pred_probs,
        axis=1
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

    print("\n" + name + " RESULTS")

    print("Accuracy:", accuracy)

    print("Precision:", precision)

    print("Recall:", recall)

    print("F1 Score:", f1)

    print("Training Time:", training_time)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                str(c)
                for c in label_encoder.classes_
            ],
            zero_division=0
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (sec)": training_time
    }


cnn_model = Sequential([
    Input(
        shape=(
            X_train_dl.shape[1],
            1
        )
    ),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    BatchNormalization(),

    MaxPooling1D(
        pool_size=2
    ),

    Dropout(0.3),

    Flatten(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        num_classes,
        activation="softmax"
    )
])

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


gru_model = Sequential([
    Input(
        shape=(
            X_train_dl.shape[1],
            1
        )
    ),

    GRU(
        128,
        return_sequences=True
    ),

    Dropout(0.3),

    GRU(64),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        num_classes,
        activation="softmax"
    )
])

gru_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


lstm_model = Sequential([
    Input(
        shape=(
            X_train_dl.shape[1],
            1
        )
    ),

    LSTM(
        128,
        return_sequences=True
    ),

    Dropout(0.3),

    LSTM(64),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        num_classes,
        activation="softmax"
    )
])

lstm_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


results = []

results.append(
    evaluate_model(
        "1D-CNN",
        cnn_model
    )
)

results.append(
    evaluate_model(
        "GRU",
        gru_model
    )
)

results.append(
    evaluate_model(
        "LSTM",
        lstm_model
    )
)


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 80)
print("DEEP LEARNING MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)

best_model = results_df.iloc[0]["Model"]

print("\n" + "=" * 60)
print("BEST DEEP LEARNING MODEL:", best_model)
print("=" * 60)