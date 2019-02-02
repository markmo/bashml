import numpy as np


# noinspection PyPep8Naming
class LogisticRegression(object):

    def __init__(self, lr=0.01, n_epochs=100000, fit_intercept=False, verbose=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = None

    @staticmethod
    def __add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y, report_every=10000):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        if self.verbose:
            print('X shape:', X.shape)
            print(X)
            print(y)
        # initialize weights
        self.theta = np.zeros(X.shape[1])
        if self.verbose:
            print('theta shape:', self.theta.shape)
            print(self.theta)

        for i in range(self.n_epochs):
            z = np.dot(X, self.theta)
            if self.verbose:
                print('z shape:', z.shape)
                print(z)
            h = self.__sigmoid(z)
            if self.verbose:
                print('h shape:', h.shape)
                print(h)
            d = h - y
            if self.verbose:
                print('d shape:', d.shape)
                print(d)
            gradient = np.dot(X.T, (h - y)) / y.size
            if self.verbose:
                print('g shape:', gradient.shape)
                print(gradient)
            self.theta -= self.lr * gradient
            if self.verbose:
                print('updated theta:')
                print(self.theta)
            # if self.verbose and i % report_every == 0:
            if i % report_every == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

    def save(self):
        np.save('weights.npy', self.theta, allow_pickle=True)

    def load(self):
        self.theta = np.load('weights.npy', allow_pickle=True)
