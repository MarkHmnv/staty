import numpy as np


def _sigmoid(z):
    """
    :param z: The input value or array.
    :return: The sigmoid value or array of sigmoid values.

    """
    return 1 / (1 + np.exp(-z))


def _propagate(w, b, X, Y):
    """
    Performs forward propagation and calculates the cost function and gradients.

    :param w: numpy array of shape (n, 1) representing weights
    :param b: scalar representing bias
    :param X: numpy array of shape (n, m) representing input data
    :param Y: numpy array of shape (1, m) representing true labels
    :return: dictionary containing gradients dw, db and the cost function

    """
    m_inv = 1 / X.shape[1]
    A = _sigmoid(np.dot(w.T, X) + b)
    cost = -m_inv * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = m_inv * np.dot(X, (A - Y).T)
    db = m_inv * np.sum(A - Y)
    cost = np.squeeze(np.array(cost))

    return {"dw": dw, "db": db}, cost


def _optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    Optimizes the parameters `w` and `b` using gradient descent for logistic regression.

    :param w: The weights of the logistic regression model.
    :param b: The bias of the logistic regression model.
    :param X: The input features.
    :param Y: The true labels.
    :param num_iterations: The number of iterations to perform gradient descent.
    :param learning_rate: The learning rate for gradient descent.
    :param print_cost: Whether to print the cost after every 100 iterations. Default is False.

    :return: The optimized values of `w` and `b`.
    """
    costs = []

    for i in range(num_iterations):
        grads, cost = _propagate(w, b, X, Y)

        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print(f"Cost after iteration {i}: {cost}")
    return w, b


class LogisticRegression:
    def __init__(self):
        self.w, self.b = None, None

    def fit(self, X_train, Y_train, num_iterations=2000, learning_rate=0.001, print_cost=False):
        """
        Fit the logistic regression model to the training data.

        :param X_train: The feature matrix of shape (num_features, num_examples).
        :param Y_train: The target vector of shape (1, num_examples).
        :param num_iterations: The number of iterations for training (default: 2000).
        :param learning_rate: The learning rate for training (default: 0.001).
        :param print_cost: Whether to print the cost after every 100 iterations (default: False).
        """
        w = np.zeros((X_train.shape[0], 1))
        b = 0.0

        self.w, self.b = _optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    def predict(self, X):
        """
        Predicts the labels for the given input examples.

        :param X: A numpy array of shape (m, n) containing the input examples.
        :return: A numpy array of shape (m,) containing the predicted labels for the input examples.
        """
        A = _sigmoid(np.dot(self.w.T, X) + self.b)
        return np.where(A > 0.5, 1, 0)

    @staticmethod
    def calculate_accuracy(Y_true, Y_pred):
        """Calculate the accuracy of a classification model.

        :param Y_true: The true labels.
        :param Y_pred: The predicted labels.
        :return: The accuracy of the model, calculated as the percentage of correct predictions.
        """
        return 1 - np.mean(np.abs(Y_pred - Y_true))
