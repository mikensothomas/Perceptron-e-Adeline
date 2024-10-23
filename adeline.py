import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    def __init__(self, input_size, learning_rate=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    def predict(self, x):
        return 1 if self.activation(x) >= 0 else 0

    def fit(self, X, y):
        mse_history = []
        for _ in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                output = self.activation(xi)
                error = target - output
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error
                total_error += error**2
            mse = total_error / len(X)
            mse_history.append(mse)
        return mse_history

def train_adaline(logic_gate, X, y):
    adaline = Adaline(input_size=2)
    mse_history = adaline.fit(X, y)
    
    plt.plot(mse_history)
    plt.title(f'Treinamento Adaline-{logic_gate}. Fecha esta janela para visualizar OR e XOR.')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.show()

def main():
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    train_adaline('AND', X_and, y_and)

    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    train_adaline('OR', X_or, y_or)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    train_adaline('XOR', X_xor, y_xor)

if __name__ == '__main__':
    main()
