import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the iris dataset from an external CSV file
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a support vector machine (SVM) classifier
clf = SVC(kernel='linear', random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
predictions = clf.predict(X_test)

# Print the accuracy of the classifier
print("Accuracy: ", np.mean(predictions == y_test))

# Print the detailed classification report
print(classification_report(y_test, predictions))

# Test the classifier with new measurements
new_measurements = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
new_predictions = clf.predict(new_measurements)
print("Prediction of Species: ", new_predictions)
