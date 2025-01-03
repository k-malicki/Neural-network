# Przykładowa implementacja sieci neuronowej od podstaw
import numpy as np

# Funkcja aktywacji - sigmoid
def activation(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji aktywacji
def derivative_activation(x):
    fx = activation(x)
    return fx * (1 - fx)

# Funkcja straty MSE
def loss_function(y, y_pred):
    return np.mean((y - y_pred) ** 2)

class NeuralNetwork:
    def __init__(self):
        # Inicjalizacja wag i biasów
        self.weights = np.random.normal(size=6)  
        self.biases = np.random.normal(size=3)  

    def feedforward(self, x):
        # Aktywacje - Warstwa ukryta
        a0 = activation(self.weights[0] * x[0] + self.weights[2] * x[1] + self.biases[0])
        a1 = activation(self.weights[1] * x[0] + self.weights[3] * x[1] + self.biases[1])

        # Aktywacje - warstwa wyjściowa
        a2 = activation(self.weights[4] * a0 + self.weights[5] * a1 + self.biases[2])
        return a2

    def train(self, data, y_true, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            for x, y_t in zip(data, y_true):
                # Propagacja w przód
                a0 = activation(self.weights[0] * x[0] + self.weights[2] * x[1] + self.biases[0])
                a1 = activation(self.weights[1] * x[0] + self.weights[3] * x[1] + self.biases[1])
                a2 = activation(self.weights[4] * a0 + self.weights[5] * a1 + self.biases[2])
                y_pred = a2

                # Propagacja wsteczna. Obliczanie pochodnych cząstkowych
                dL_dypred = -2 * (y_t - y_pred)
                dypred_dz2 = derivative_activation(a2)

                dz2_dw4 = a0
                dz2_dw5 = a1

                # Aktualizacja wag i biasów warstwy wyjściowej
                self.weights[4] -= learning_rate * dL_dypred * dypred_dz2 * dz2_dw4
                self.weights[5] -= learning_rate * dL_dypred * dypred_dz2 * dz2_dw5
                self.biases[2] -= learning_rate * dL_dypred * dypred_dz2

                # Gradienty dla warstwy ukrytej
                da0_dz0 = derivative_activation(a0)
                da1_dz1 = derivative_activation(a1)

                dz0_dw0 = x[0]
                dz0_dw2 = x[1]

                dz1_dw1 = x[0]
                dz1_dw3 = x[1]

                dL_dz0 = dL_dypred * dypred_dz2 * self.weights[4]
                dL_dz1 = dL_dypred * dypred_dz2 * self.weights[5]

                # Aktualizacja wag i biasów warstwy ukrytej
                self.weights[0] -= learning_rate * dL_dz0 * da0_dz0 * dz0_dw0
                self.weights[2] -= learning_rate * dL_dz0 * da0_dz0 * dz0_dw2
                self.biases[0] -= learning_rate * dL_dz0 * da0_dz0

                self.weights[1] -= learning_rate * dL_dz1 * da1_dz1 * dz1_dw1
                self.weights[3] -= learning_rate * dL_dz1 * da1_dz1 * dz1_dw3
                self.biases[1] -= learning_rate * dL_dz1 * da1_dz1

            if epoch % 100 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = loss_function(y_true, y_preds)
                print(f"Epoka {epoch}, Strata: {loss}")

# Dane treningowe
data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_true = np.array([0, 0, 0, 1])

# Tworzenie sieci
nn = NeuralNetwork()
nn.train(data, y_true)

print("\n--Przewidywania sieci: ")
for x in data:
    print(f"Input: {x}, Wynik: {nn.feedforward(x):.4f}")
