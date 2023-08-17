import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sys


def load_dataset(filename):
    # Load dataset using pandas
    dataset = pd.read_csv(filename, header=None)

    # Check if it's the Sonar dataset by examining the last column's data type
    if dataset.iloc[:, -1].dtype == object:
        # Convert 'M' to 1 and 'R' to 0 for the Sonar dataset
        dataset.iloc[:, -1] = dataset.iloc[:, -1].map({'M': 1, 'R': 0})

    return dataset


def load_and_split_data(dataset_file):
    if "sonar" in dataset_file:
        dataset = pd.read_csv(dataset_file, header=None)
        X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
        y = np.where(y == "R", 0, 1)
    elif "data_banknote_authentication" in dataset_file:
        dataset = pd.read_csv(dataset_file, header=None)
        X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
    else:
        raise ValueError("Invalid dataset")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



def split_data(X, y):
    # Split the dataset into training and testing sets using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


class MLChoice:
    def __init__(self, ML, dataset):
        self.ML = ML
        self.dataset = dataset

    def train_knn_scratch(self, X_train, y_train, k=3):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def test_knn_scratch(self, X_test, y_test):
    # Define a function to calculate Euclidean distance between two points
        def euclidean_distance(a, b):
            return np.sqrt(np.sum((a - b) ** 2))

        # Create an empty list to store predicted labels for each test point
        y_pred = []

        # Iterate over each test point
        for test_point in X_test:
            # Create an empty list to store distances between the current test point and all training points
            distances = []
            
            # Iterate over each training point and its corresponding label
            for train_point, train_label in zip(self.X_train, self.y_train):
                # Calculate Euclidean distance between the current test point and the current training point
                dist = euclidean_distance(test_point, train_point)
                
                # Append the calculated distance and the corresponding training label to the distances list
                distances.append((dist, train_label))

            # Sort the distances list in ascending order based on the distance values
            sorted_distances = sorted(distances, key=lambda x: x[0])
            
            # Select the labels of the k nearest neighbors based on the sorted distances list
            k_nearest_neighbors = [label for _, label in sorted_distances[:self.k]]

            # Predict the label for the current test point based on the majority label among the k nearest neighbors
            predicted_label = max(set(k_nearest_neighbors), key=k_nearest_neighbors.count)
            
            # Append the predicted label to the y_pred list
            y_pred.append(predicted_label)

        # Calculate the accuracy of the model by comparing the predicted labels with the true labels for the test data
        accuracy = accuracy_score(y_test, y_pred)
        
        # Return the accuracy value
        return accuracy

    
    def train_svm_scratch(self, X_train, y_train, C=1.0, learning_rate=0.001, epochs=1000):
    # Set the values of hyperparameters
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Get the number of samples and features in the training data
        n_samples, n_features = X_train.shape

        # Transform the labels to {1, -1}
        y_transformed = np.where(y_train <= 0, -1, 1)

        # Initialize the weights and bias to zero
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training using Stochastic Gradient Descent
        for epoch in range(1, epochs):
            # Iterate over each sample in the training data
            for idx, x_i in enumerate(X_train):
                # Calculate the condition for updating the weights and bias
                condition = y_transformed[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1

                # If the condition is true, update the weights using L2 regularization
                if condition:
                    self.weights -= learning_rate * (2 * (1 / epoch) * self.weights)
                # If the condition is false, update the weights and bias using L2 regularization and hinge loss
                else:
                    self.weights -= learning_rate * (2 * (1 / epoch) * self.weights - np.dot(x_i, y_transformed[idx]))
                    self.bias -= learning_rate * y_transformed[idx]



    def test_svm_scratch(self, X_test, y_test):
        y_pred = np.sign(np.dot(X_test, self.weights) - self.bias)
        y_pred_transformed = np.where(y_pred <= 0, 0, 1)

        accuracy = accuracy_score(y_test, y_pred_transformed)
        return accuracy
    
    def train_test_sklearn(self, X_train, y_train, X_test, y_test):
        if self.ML == "KNN":
            model = KNeighborsClassifier()
        elif self.ML == "SVM":
            model = SVC()
        else:
            raise ValueError("Invalid ML algorithm")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy
    

    def compare_and_predict(self, X_train, y_train, X_test, y_test, prediction_point):
    # Train the model from scratch and calculate its accuracy
        if self.ML == "KNN":
            self.train_knn_scratch(X_train, y_train)
            scratch_accuracy = self.test_knn_scratch(X_test, y_test)
        elif self.ML == "SVM":
            self.train_svm_scratch(X_train, y_train)
            scratch_accuracy = self.test_svm_scratch(X_test, y_test)
        else:
            raise ValueError("Invalid ML algorithm")

        # Train the model using scikit-learn and calculate its accuracy
        sklearn_accuracy = self.train_test_sklearn(X_train, y_train, X_test, y_test)

        # Make a prediction for the given test point using the model trained from scratch
        if self.ML == "KNN":
            # For KNN, predict the label based on the k-nearest neighbors
            prediction = max(set(self.y_train[np.argsort(np.sum((X_train - prediction_point) ** 2, axis=1))[:self.k]]),
                            key=list(self.y_train[np.argsort(np.sum((X_train - prediction_point) ** 2, axis=1))[:self.k]]).count)
        elif self.ML == "SVM":
            # For SVM, predict the label based on the sign of the decision function
            prediction = 1 if np.sign(np.dot(prediction_point, self.weights) - self.bias) > 0 else 0

        # Get the actual label for the given test point
        actual = y_test[np.where((X_test == prediction_point).all(axis=1))[0][0]]

        # Return the accuracy of the model trained from scratch, the accuracy of the model trained using scikit-learn,
        # the predicted label for the given test point, and the actual label for the given test point
        return scratch_accuracy, sklearn_accuracy, prediction, actual

    


if __name__ == "__main__":
    import sys

    # Get command line arguments
    ML = sys.argv[1]
    dataset_file = sys.argv[2]

    # Load the dataset and preprocess it
    dataset = load_dataset(dataset_file)
    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = load_and_split_data(dataset_file)

    # Create an instance of the MLChoice class and run the compare_and_predict method
    ml_choice = MLChoice(ML, dataset)
    scratch_accuracy, sklearn_accuracy, prediction, actual = ml_choice.compare_and_predict(
        X_train, y_train, X_test, y_test, X_test[0])

    # Print the results
    output_file=open('generated_outputs.txt','w')


    output_file.write("DataSet: "+dataset_file.split(".")[0]+"\n\n")
    output_file.write("Machine Learning Algorithm Chosen: "+ML+"\n\n")
    output_file.write("Accuracy of Training (Scratch): "+str(round(scratch_accuracy * 100, 2))+"%"+"\n\n")
    output_file.write("Accuracy of ScikitLearn Function: "+str(round(sklearn_accuracy * 100, 2))+"%"+"\n\n")
    output_file.write("Prediction Point: "+str(X_test[0])+"\n\n")
    output_file.write("Predicted Class: "+str(prediction)+"\n\n")
    output_file.write("Actual Class: "+str(actual)+"\n\n")



    print("DataSet:", dataset_file.split(".")[0])
    print("Machine Learning Algorithm Chosen:", ML)
    print("Accuracy of Training (Scratch):", round(scratch_accuracy * 100, 2), "%")
    print("Accuracy of ScikitLearn Function:", round(sklearn_accuracy * 100, 2), "%")
    print("Prediction Point:", X_test[0])
    print("Predicted Class:", prediction)
    print("Actual Class:", actual)
