import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialisiere Gewichte und Bias für Hidden Layer
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))

        # Initialisiere Gewichte und Bias für Output Layer
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))

    # Aktivierungsfunktionen
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    # Vorwärtsdurchlauf
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)  # ReLU im Hidden Layer

        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)  # Sigmoid im Output Layer

        return self.a2

    # Rückwärtsdurchlauf (Backpropagation)
    def backward(self, X, y, output, learning_rate):
        # Fehler im Output
        error = output - y
        d_z2 = error * self.sigmoid_derivative(self.a2)

        # Output-Layer-Anpassung
        d_w2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)
        self.weights2 -= learning_rate * d_w2
        self.bias2 -= learning_rate * d_b2

        # Fehler zurückpropagieren
        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * self.relu_derivative(self.z1)

        # Hidden-Layer-Anpassung
        d_w1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)
        self.weights1 -= learning_rate * d_w1
        self.bias1 -= learning_rate * d_b1

    # Training mit Logging
    def train(self, X, y, epochs, learning_rate):
        self.losses = []

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

            loss = np.mean((y - output) ** 2)
            self.losses.append(loss)

            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")

    # Klassifikation
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)
