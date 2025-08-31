from network import NeuralNetwork
from data_generator import generate_moons
import matplotlib.pyplot as plt
import numpy as np

# Daten vorbereiten
X, y = generate_moons(n_samples=300, noise=0.2)
y = y.reshape(-1, 1)  # Form anpassen

# Netzwerk initialisieren
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Trainieren
nn.train(X, y, epochs=5000, learning_rate=0.1)

# Vorhersagen
pred = nn.predict(X)

# Plotten
plt.scatter(X[:, 0], X[:, 1], c=pred.flatten(), cmap='coolwarm', alpha=0.7)
plt.title("Klassifizierte Punkte durch NN")
plt.show()

# Loss visualisieren
plt.plot(nn.losses)
plt.title("Loss während des Trainings")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
