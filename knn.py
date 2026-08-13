from train import X_train_scaled, X_test_scaled, y_train, y_test

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Use only a subset of the training data
sample_size = 50000

X_train_knn = X_train_scaled[:sample_size]
y_train_knn = y_train[:sample_size]

print("KNN training samples:", len(X_train_knn))


# KNN
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    n_jobs=-1
)

# Train
knn_model.fit(X_train_knn, y_train_knn)

# Predict on the complete test set
knn_pred = knn_model.predict(X_test_scaled)

# Accuracy
knn_accuracy = accuracy_score(y_test, knn_pred)

print("KNN Accuracy: {:.2f}%".format(knn_accuracy * 100))