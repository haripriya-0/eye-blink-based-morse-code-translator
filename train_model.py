import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'blink_classifier_model.pkl')
print("Model saved as 'blink_classifier_model.pkl'")
