from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import numpy as np

class MyPerceptron:
    def __init__(self, lr, max_iterations, accuracy_hurdle):
        self.lr = lr
        self.max_iterations = max_iterations
        self.accuracy_hurdle = accuracy_hurdle

    def fit(self, X, y):
        self.weight = np.zeros(X.shape[1]) # weight for features
        self.b = 0 # intercept
        for i in range(self.max_iterations):
            # calculate predicted y
            y_predict = np.dot(X, self.weight) + self.b
            # find misclassified sample
            misclassified = np.where(y * y_predict <= 0)[0]
            if len(misclassified) == 0:
                break
            else:
                # select a misclassified sample randomly and update weight, intercept
                misclassified_x = np.random.choice(misclassified)
                self.weight += self.lr * np.dot(y[misclassified_x], X[misclassified_x])
                self.b += self.lr * y[misclassified_x]
                # calculate the accuracy and set a hurdle to stop the iteration
                accuracy = self.calculateAccuracy(X, y)
                if accuracy >= self.accuracy_hurdle:
                    break
    def predict(self, X):
        return np.sign(np.dot(X, self.weight) + self.b)
    def calculateAccuracy(self, X, y):
        y_predict_label = self.predict(X)
        # use 0\1 loss
        misclassified_y = np.where(y * y_predict_label <= 0)[0]
        accuracy = 1 - len(misclassified_y) / len(y)
        return accuracy

if __name__ == "__main__":
    # load data
    data = load_breast_cancer()
    # print(data)
    # print(type(data))
    X = data.data
    y = data.target
    y = np.array([1 if i == 1 else -1 for i in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = 0.01
    iterations = 1000
    stop_hurdle = 0.9
    myPerceptron = MyPerceptron(lr, iterations, stop_hurdle)
    myPerceptron.fit(X_train, y_train)

    # using sklearn's Perceptron function to train data
    # base case: default value
    perceptron_base = Perceptron()
    perceptron_base.fit(X_train, y_train)
    # try different hyperparameter
    perceptron_1 = Perceptron(penalty='l1', alpha=0.001, eta0=0.1)
    perceptron_1.fit(X_train, y_train)

    # print the accuracy of training set
    print("The accuracy of the training set using my own perceptron is:")
    print(myPerceptron.calculateAccuracy(X_train, y_train))
    print("The accuracy of the training set using sklearn's perceptron(base case) is:")
    print(perceptron_base.score(X_train, y_train))
    print("The accuracy of the training set using sklearn's perceptron(compare case) is:")
    print(perceptron_1.score(X_train, y_train))
    # print the accuray of testing set
    print("The accuracy of the test set using my own perceptron is:")
    print(myPerceptron.calculateAccuracy(X_test, y_test))
    print("The accuracy of the test set using sklearn's perceptron(base case) is:")
    print(perceptron_base.score(X_test, y_test))
    print("The accuracy of the test set using sklearn's perceptron(compare case) is:")
    print(perceptron_1.score(X_test, y_test))

