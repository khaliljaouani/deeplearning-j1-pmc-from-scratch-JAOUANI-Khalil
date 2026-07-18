"""Phase 2 : descente de gradient à la main, courbe de loss par epoch.

Le neurone met à jour ses poids à chaque epoch grâce au gradient
de la loss BCE (qui se simplifie élégamment avec sigmoid) :
    dL/dw = (1/n) * X^T . (y_pred - y)
    dL/db = moyenne(y_pred - y)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X = np.array([[0.2, 0.1], [0.8, 0.9], [0.3, 0.7], [0.9, 0.2]])
y = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


np.random.seed(42)
w = np.random.randn(2) * 0.01   # petits poids aléatoires — briser la symétrie
b = 0.0
learning_rate = 0.1
n_epochs = 50

# Cas limite : learning_rate = 0 -> la loss ne bouge pas, les poids restent figés.
# C'est le premier réflexe de débogage si la courbe ne descend pas.
# Scénario adversarial : learning_rate = 10.0 -> la loss oscille/explose,
# on saute par-dessus le minimum à chaque pas (sweet spot dépassé).

losses = []
for epoch in range(n_epochs):
    # Forward pass
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)

    loss = compute_loss(y, y_pred)
    losses.append(loss)

    # Gradient BCE+sigmoid (résultat de la chain rule, simplifié)
    error = y_pred - y                 # shape [4,]
    dw = np.dot(X.T, error) / len(y)   # shape [2,]
    db = np.mean(error)                # scalaire

    # Descente de gradient : un pas dans la direction opposée au gradient
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w.round(3)} | b: {b:.3f}")

# Note : avec le gradient moyenné (1/n) et lr=0.1, la loss atteint ~0.59 en
# 50 epochs et continue de descendre si on prolonge. Sans le /n (gradient
# sommé), chaque pas est 4x plus grand et on atteint ~0.4 : même direction,
# taille de pas différente — c'est équivalent à lr=0.4.

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("Loss BCE")
plt.title("Convergence du neurone unique")
plt.savefig("phase2_loss_curve.png", dpi=100, bbox_inches='tight')
print(f"\nCourbe sauvegardée : phase2_loss_curve.png")
print(f"Loss finale : {losses[-1]:.4f}")
