import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        errors = []
        for _ in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error
                total_error += int(error != 0.0)
            errors.append(total_error)
        return errors

def train_perceptron(logic_gate, X, y):
    perceptron = Perceptron(input_size=2)
    errors = perceptron.fit(X, y)
    
    plt.plot(errors)
    plt.title(f'Treinamento Perceptron-{logic_gate}. Fecha esta janela para visualizar OR e XOR.')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    
    plt.show()

def main():
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    train_perceptron('AND', X_and, y_and)

    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    train_perceptron('OR', X_or, y_or)

    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    train_perceptron('XOR', X_xor, y_xor)

if __name__ == '__main__':
    main()
