# 📘 SVD und PCA ohne NumPy – nur Python + Mathe
# Hinweis: Nur für kleine Matrizen praktikabel

import math

# 1️⃣ Beispielmatrix X (2D, 3 Punkte)
X = [
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9]
]

# 2️⃣ Mittelwert jeder Spalte berechnen
def mean(data):
    n = len(data)
    d = len(data[0])
    return [sum(data[i][j] for i in range(n)) / n for j in range(d)]

X_mean = mean(X)

# 3️⃣ Daten zentrieren
def center(data, mean):
    return [[x_j - mean[j] for j, x_j in enumerate(x)] for x in data]

X_centered = center(X, X_mean)

# 4️⃣ Kovarianzmatrix berechnen (manuell, 2D)
def covariance_matrix(data):
    n = len(data)
    m = len(data[0])
    cov = [[0.0 for _ in range(m)] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            cov[i][j] = sum(data[k][i] * data[k][j] for k in range(n)) / (n - 1)
    return cov

C = covariance_matrix(X_centered)

# 5️⃣ Eigenwerte und Eigenvektoren (nur 2x2 Spezialfall)
def eigen_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    trace = a + d
    det = a*d - b*c
    term = math.sqrt(trace**2 - 4*det)
    eig1 = (trace + term) / 2
    eig2 = (trace - term) / 2

    def eigenvector(mat, eig):
        a, b = mat[0]
        c, d = mat[1]
        if b != 0:
            return [eig - d, b]
        elif c != 0:
            return [c, eig - a]
        else:
            return [1, 0]

    v1 = eigenvector(matrix, eig1)
    v2 = eigenvector(matrix, eig2)

    def normalize(v):
        norm = math.sqrt(sum(x**2 for x in v))
        return [x / norm for x in v]

    return [eig1, normalize(v1)], [eig2, normalize(v2)]

eig1, eig2 = eigen_2x2(C)

# 6️⃣ Projektion auf erste Hauptkomponente
principal_axis = eig1[1]  # Vektor mit größtem Eigenwert
Z = [sum(x * v for x, v in zip(point, principal_axis)) for point in X_centered]

# 7️⃣ Rückprojektion (für Verständnis)
X_proj = [[z * v for v in principal_axis] for z in Z]

# 8️⃣ Ausgabe
print("Zentrierte Daten:")
for row in X_centered:
    print(row)

print("\nKovarianzmatrix:")
for row in C:
    print(row)

print("\nHauptkomponenten (Eigenvektoren):")
print("1:", eig1)
print("2:", eig2)

print("\nProjektion auf Hauptkomponente:")
for z in Z:
    print([z * v for v in principal_axis])
