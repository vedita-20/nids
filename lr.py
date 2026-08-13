from train import X_train_scaled, X_test_scaled, y_train, y_test

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train
lr_model.fit(X_train_scaled, y_train)

# Predict
lr_pred = lr_model.predict(X_test_scaled)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy: {:.2f}%".format(lr_accuracy * 100))