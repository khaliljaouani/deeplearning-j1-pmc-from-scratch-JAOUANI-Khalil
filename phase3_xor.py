"""Phase 3 : XOR, échec du neurone unique, victoire de la couche cachée.

XOR n'est pas linéairement séparable : aucune droite ne sépare
(0,0),(1,1) de (0,1),(1,0). Un neurone seul (= une droite) échoue.
Un réseau 2-2-1 avec couche cachée apprend une frontière non-linéaire.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss_bce(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Architecture 2-2-1 : 2 entrées -> 2 neurones cachés -> 1 sortie
# Biais initialisés aléatoirement eux aussi : avec b=0 et cette seed, le
# réseau tombe dans un minimum local (loss ~0.35, accuracy 50%) — XOR n'a
# que 4 points, l'init compte énormément.
np.random.seed(42)
W1 = np.random.randn(2, 2) * 0.5
b1 = np.random.randn(2) * 0.5
W2 = np.random.randn(2, 1) * 0.5
b2 = np.random.randn(1) * 0.5

learning_rate = 0.5
n_epochs = 10000
n = len(y_xor)
losses = []

for epoch in range(n_epochs):
    # Forward pass couche 1 (cachée)
    z1 = X_xor @ W1 + b1        # [4, 2]
    a1 = sigmoid(z1)            # [4, 2]

    # Forward pass couche 2 (sortie)
    z2 = a1 @ W2 + b2           # [4, 1]
    a2 = sigmoid(z2)            # [4, 1]

    y_pred = a2.flatten()
    loss = compute_loss_bce(y_xor, y_pred)
    losses.append(loss)

    # Backprop couche 2 — simplification BCE+sigmoid : erreur = prédiction - cible
    error2 = (y_pred - y_xor).reshape(-1, 1)   # [4, 1]
    dW2 = a1.T @ error2 / n                    # [2, 1]
    db2 = np.mean(error2)                      # scalaire

    # Backprop couche 1 — chain rule : on remonte via W2, puis dérivée de sigmoid
    # sigma'(z1) = a1 * (1 - a1)
    error1 = (error2 @ W2.T) * a1 * (1 - a1)   # [4, 2]
    dW1 = X_xor.T @ error1 / n                 # [2, 2]
    db1 = error1.mean(axis=0)                  # [2,]

    # Mise à jour des poids
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 2000 == 0:
        acc = np.mean((y_pred > 0.5) == y_xor)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")

# Cas limite : avec 1 seul neurone caché (2-1-1), le réseau ne converge
# généralement pas à 100% : la couche cachée compresse les 2 entrées en un
# seul scalaire, il n'y a plus assez de représentations intermédiaires pour
# séparer la diagonale de l'anti-diagonale. Il faut au moins 2 neurones
# cachés pour combiner deux "demi-plans" en une région non-linéaire.

# Frontière de décision (code fourni)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
z1g = sigmoid(np.dot(grid, W1) + b1)
z2g = sigmoid(np.dot(z1g, W2) + b2).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z2g, alpha=0.4, cmap='RdBu')
plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor, s=100, cmap='RdBu', edgecolors='k')
plt.title("XOR : frontière de décision du réseau 2-2-1")
plt.savefig("phase3_xor_boundary.png", dpi=100, bbox_inches='tight')

print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred > 0.5) == y_xor):.2%}")
