from train import X_train, X_test, y_train, y_test

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Decision Tree Classifier
dt_model = DecisionTreeClassifier(
    max_depth=15,
    random_state=42
)

# Train
dt_model.fit(X_train, y_train)

# Predict
dt_pred = dt_model.predict(X_test)

# Accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy: {:.2f}%".format(dt_accuracy * 100))