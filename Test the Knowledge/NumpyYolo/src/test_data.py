import matplotlib.pyplot as plt
from data_generator import generate_moons

X, y = generate_moons(300, noise=0.1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title("2D Moons-Daten")
plt.show()
