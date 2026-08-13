from train import X_train_scaled, X_test_scaled, y_train, y_test

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


svm_model = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

svm_pred = svm_model.predict(X_test_scaled)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy: {:.2f}%".format(svm_accuracy * 100))