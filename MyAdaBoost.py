import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class MyAda:
    def __init__(self, alpha, weakLearner_list, maxIterations, D_t):
        # D_t: denote the distribution of the sample
        # alpha: denote the weight for each weak learner
        self.alpha = alpha
        self.weakLearner_list = weakLearner_list
        self.maxIterations = maxIterations
        self.D_t = D_t

    def adaBoost(self, X, y):
        sample_size = len(y)
        D_t = np.ones(sample_size) / sample_size
        for iteration in range(self.maxIterations):
            # 1. train a weak learner
            h_t = Perceptron()
            h_t.fit(X, y, sample_weight=D_t)

            # 2. calculate the predict value and train error
            y_predict = h_t.predict(X)
            self.weakLearner_list.append(h_t)
            errsample = (y != y_predict).astype(np.float64)
            error = np.dot(errsample, D_t)

            # 3. calculate weight for each weak learner(alpha)
            alpha_t = 0.5 * np.log((1 - error) / error)
            self.alpha.append(alpha_t)

            # 4. update distribution of the sample
            D_t *= np.exp(-alpha_t * y * y_predict)
            D_t /= np.sum(D_t)
            self.D_t.append(D_t)

    def predict(self,a,wl,X,y):
        # use alpha and prediceted value by h to calculate f
        # eg. self.alpha = [0.3, 0.2, 0.5]
        # eg. self.predict_h = [[0,1,0,1], [0,1,0,0]]
        alpha_array = np.array(a)
        # wl_predict = wl.predict(X)
        predict_h_array = np.array(wl.predict(X))
        f_predict = np.dot(alpha_array, predict_h_array)
        return f_predict

    def calculateAccuracy(self, X, y):
        res = []
        for i in range(1,1+self.maxIterations):
            f = 0
            for alpha, weak_learner in zip(self.alpha[:i], self.weakLearner_list[:i]):
                f += self.predict(alpha, weak_learner, X, y)
            accuracy = accuracy_score(y, np.sign(f))
            # accuracy = np.sum(np.sign(f) == y) / len(y)
            res.append([i, accuracy])
        return res

    def calculateConfidenceMargin(self, X, y):
        res_margin = []
        res_f = []
        for i in range(1, 1 + self.maxIterations):
            f= 0
            for alpha, weak_learner in zip(self.alpha[:i], self.weakLearner_list[:i]):
                f += self.predict(alpha, weak_learner, X, y)
            # calculate the confidence margin as the minimum of f mutiplied by y
            res = np.min(np.abs(f))
            # margin = np.min(np.abs(f))
            res_margin.append([i, res])
            res_f.append(f)
        return res_margin

    def calculateMisclassifed(self, X, y):
        # calculate Σi=1mDi(t+1)1{yift(xi)<0} for every single iteration of weak learner
        # 1. sample uniformly from D(t+1)
        sample_size = len(y)
        res_misclassified = []
        for i, distribution, weak_learner in zip(range(1, self.maxIterations+1), self.D_t, self.weakLearner_list):
            sample_uniform = np.random.choice(sample_size, sample_size, p=distribution)
            sampled_X = X[sample_uniform]
            sampled_y = y[sample_uniform]
        # 2. calculate the misclassified percentage of points by previous model and updated model
            res = np.sum(sampled_y != weak_learner.predict(sampled_X)) / len(sampled_y)
        #     res = np.sum(y != weak_learner.predict(X)) / len(y)
            res_misclassified.append([i, res])
        return res_misclassified

if __name__ == "__main__":
    # 1. load data and seprate data into train and test subset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # scale the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # print(y)
    y = np.array([1 if i == 1 else -1 for i in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. use adaboost to train data
    maxIteration = 50
    alpha = []
    weakLearner_list = []
    D_t = []
    myAdaBoost = MyAda(alpha, weakLearner_list, maxIteration, D_t)
    myAdaBoost.adaBoost(X_train,y_train)

    # 3. calculate the accuracy of every iteration
    res_train = myAdaBoost.calculateAccuracy(X_train, y_train)
    res_test = myAdaBoost.calculateAccuracy(X_test,y_test)

    # 4. plot the graph
    x_values, y_values = zip(*res_train)
    plt.plot(x_values, y_values)
    plt.title("Accuracy Plot: Train Data")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Accuracy: Train Data")
    plt.show()

    x_values2, y_values2 = zip(*res_test)
    plt.plot(x_values2, y_values2)
    plt.title("Accuracy Plot: Test Data")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Accuracy: Test Data")
    plt.show()

    # 5. calculate the confidence margin of every iteration
    res_margin_train = myAdaBoost.calculateConfidenceMargin(X_train, y_train)
    res_margin_test = myAdaBoost.calculateConfidenceMargin(X_test,y_test)

    # 6. plot the graph of confidence margin
    x_values3, y_values3 = zip(*res_margin_train)
    plt.plot(x_values3, y_values3)
    plt.title("Confidence Margin Plot: Train Data")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Confidence Margin: Train Data")
    plt.show()

    x_values4, y_values4 = zip(*res_margin_test)
    plt.plot(x_values4, y_values4)
    plt.title("Confidence Margin Plot: Test Data")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Confidence Margin: Test Data")
    plt.show()

    # 7. plot the graph of misclassified percentage
    calculateMisclassifed_train = myAdaBoost.calculateMisclassifed(X_train, y_train)
    # calculateMisclassifed_test = myAdaBoost.calculateMisclassifed(X_test, y_test)
    x_values5, y_values5 = zip(*calculateMisclassifed_train)
    plt.plot(x_values5, y_values5)
    plt.title("Misclassified Percentage Plot: Train Data")
    plt.xlabel("Number of Iteration")
    plt.ylabel("Misclassified Percentage: Train Data")
    plt.show()

    # x_values6, y_values6 = zip(*calculateMisclassifed_test)
    # plt.plot(x_values6, y_values6)
    # plt.title("Misclassified Percentage Plot: Test Data")
    # plt.xlabel("Number of Iteration")
    # plt.ylabel("Misclassified Percentage: Test Data")
    # plt.show()