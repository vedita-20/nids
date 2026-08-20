from train import X_train, X_test, y_train, y_test
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
# Train
xgb_model.fit(X_train, y_train)
# Predict
xgb_pred = xgb_model.predict(X_test)
# Accuracy
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print("XGBoost Accuracy: {:.2f}%".format(xgb_accuracy * 100))