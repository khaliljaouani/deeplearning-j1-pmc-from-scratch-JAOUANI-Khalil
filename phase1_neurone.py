"""Phase 1 : neurone unique, forward pass et calcul d'erreur (BCE).

Pas encore d'entraînement : les poids sont fixés a la main,
on observe juste ce que le neurone predit et combien il se trompe.
"""
import numpy as np

X = np.array([
    [0.2, 0.1],
    [0.8, 0.9],
    [0.3, 0.7],
    [0.9, 0.2],
])
y = np.array([0, 1, 1, 0])


def sigmoid(x):
    # Ecrase n'importe quel nombre entre 0 et 1 -> interpretable comme une probabilite
    return 1 / (1 + np.exp(-x))


def forward(X, w, b):
    # Somme ponderee (produit scalaire entree/poids) puis activation
    z = np.dot(X, w) + b
    return sigmoid(z)


def compute_loss(y_true, y_pred):
    # Binary Cross-Entropy : penalise fort les predictions confiantes et fausses
    # clip pour eviter log(0) = -inf
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# Poids fixes — pas encore d'entrainement dans cette phase
w = np.array([0.5, -0.3])
b = 0.1

y_pred = forward(X, w, b)
loss = compute_loss(y, y_pred)

print(f"Prédictions : {y_pred.round(3)}")
print(f"Étiquettes  : {y}")
print(f"Loss BCE    : {loss:.4f}")

# --- Cas limite : toutes les entrées à 0 -> seul le biais pilote la sortie ---
X_zeros = np.zeros((4, 2))
pred_zeros = forward(X_zeros, w, b)
print(f"\nCas limite X=0 : {pred_zeros.round(3)} (= sigmoid(b) = {sigmoid(b):.3f} partout)")

# --- Scénario adversarial : poids à zéro -> le réseau ne prédit rien (0.5) ---
# loss = -log(0.5) ≈ 0.693 : pire point de départ, mais une boucle
# d'entraînement (phase 2) permettra quand même d'apprendre.
w0, b0 = np.zeros(2), 0.0
pred_w0 = forward(X, w0, b0)
print(f"Poids à zéro   : {pred_w0.round(3)} | Loss : {compute_loss(y, pred_w0):.4f} (~ -log(0.5) = 0.693)")
