import numpy as np

def generate_moons(n_samples=200, noise=0.1):
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Erster Halbmond
    theta = np.linspace(0, np.pi, n_samples_out)
    x1 = np.cos(theta)
    y1 = np.sin(theta)

    # Zweiter Halbmond (versetzt)
    x2 = 1 - np.cos(theta)
    y2 = -np.sin(theta) - 0.5

    X = np.vstack([np.stack([x1, y1], axis=1), np.stack([x2, y2], axis=1)])
    y = np.array([0] * n_samples_out + [1] * n_samples_in)

    # Rauschen hinzufügen
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y
