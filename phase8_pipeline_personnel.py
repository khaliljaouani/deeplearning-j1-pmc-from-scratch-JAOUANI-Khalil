"""Phase 8 : pipeline complet sur dataset personnel (Breast Cancer).

Chargement -> préprocessing -> réseau numpy from-scratch -> même réseau
en Keras -> comparaison des métriques. 569 exemples, 30 features,
classification binaire (malin / bénin).
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# ---- 1. Chargement ----
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Shape X : {X.shape} | Classes : {np.unique(y)}")

# ---- 2. Préprocessing ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# IMPORTANT : fitter le scaler sur X_train uniquement, jamais sur X_test
# (sinon data leakage : le modèle "voit" les stats du test avant l'évaluation)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X_train.shape[1]

# ---- 3. Pipeline numpy from-scratch : [30 -> 16 -> 8 -> 1] ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Initialisation He : std = sqrt(2 / n_entrées de la couche)
np.random.seed(42)
W1 = np.random.randn(n_features, 16) * np.sqrt(2 / n_features)
b1 = np.zeros(16)
W2 = np.random.randn(16, 8) * np.sqrt(2 / 16)
b2 = np.zeros(8)
W3 = np.random.randn(8, 1) * np.sqrt(2 / 8)
b3 = np.zeros(1)

lr = 0.2
n_epochs = 200
n = len(y_train)
np_losses = []

start = time.time()
for epoch in range(n_epochs):
    # Forward
    z1 = X_train @ W1 + b1; a1 = relu(z1)
    z2 = a1 @ W2 + b2;      a2 = relu(z2)
    z3 = a2 @ W3 + b3
    y_pred = sigmoid(z3).flatten()

    np_losses.append(bce_loss(y_train, y_pred))

    # Backward (même schéma que phase 4)
    err3 = (y_pred - y_train).reshape(-1, 1)
    dW3 = a2.T @ err3 / n;  db3 = err3.mean(axis=0)
    err2 = (err3 @ W3.T) * relu_grad(z2)
    dW2 = a1.T @ err2 / n;  db2 = err2.mean(axis=0)
    err1 = (err2 @ W2.T) * relu_grad(z1)
    dW1 = X_train.T @ err1 / n; db1 = err1.mean(axis=0)

    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
    W3 -= lr * dW3; b3 -= lr * db3
numpy_time = time.time() - start

# Évaluation numpy sur le test set
a1t = relu(X_test @ W1 + b1)
a2t = relu(a1t @ W2 + b2)
y_pred_test = sigmoid(a2t @ W3 + b3).flatten()
numpy_loss = bce_loss(y_test, y_pred_test)
numpy_acc = np.mean((y_pred_test > 0.5) == y_test)
print(f"\nNumpy from-scratch | Loss finale : {numpy_loss:.4f} | Test accuracy : {numpy_acc:.4f} | {numpy_time:.1f}s")

# ---- 4. Pipeline Keras : même architecture ----
tf.random.set_seed(42)
model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

start = time.time()
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.1, verbose=0)
keras_time = time.time() - start

keras_loss, keras_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Keras              | Loss finale : {keras_loss:.4f} | Test accuracy : {keras_acc:.4f} | {keras_time:.1f}s")

# ---- 5. Comparaison agrégée ----
print("\n=== COMPARAISON NUMPY vs KERAS (Breast Cancer) ===")
print(f"{'Pipeline':20s} | {'Test loss':10s} | {'Test accuracy':14s} | {'Temps (s)':10s}")
print("-" * 65)
print(f"{'Numpy from-scratch':20s} | {numpy_loss:.4f}     | {numpy_acc:.4f}         | {numpy_time:.1f}")
print(f"{'Keras':20s} | {keras_loss:.4f}     | {keras_acc:.4f}         | {keras_time:.1f}")
print(f"\nGain Keras vs Numpy : {(keras_acc - numpy_acc) * 100:+.1f} points de %")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(np_losses, label='numpy (train)')
axes[0].plot(history.history['loss'], label='keras (train)')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss BCE")
axes[0].set_title("Courbes de loss"); axes[0].legend()
axes[1].bar(['Numpy', 'Keras'], [numpy_acc, keras_acc], color=['steelblue', 'darkorange'])
axes[1].set_ylim(0.8, 1.0); axes[1].set_ylabel("Test accuracy")
axes[1].set_title("Accuracy comparée")
plt.savefig("phase8_comparaison.png", dpi=100, bbox_inches='tight')
print("Graphe sauvegardé : phase8_comparaison.png")

# Scénario adversarial : une ligne de valeurs extrêmes hors distribution
X_extreme = np.array([[99999.0] * n_features])
X_extreme_scaled = scaler.transform(X_extreme)
pred_np = sigmoid(relu(relu(X_extreme_scaled @ W1 + b1) @ W2 + b2) @ W3 + b3).flatten()[0]
pred_keras = float(model.predict(X_extreme_scaled, verbose=0).flatten()[0])
print(f"\nEntrée extrême (99999 partout) -> numpy : {pred_np:.4f} | keras : {pred_keras:.4f}")
# Observation : le modèle sort une prédiction très confiante (proche de 0 ou 1)
# sur une donnée qui n'a aucun sens physique. En production, c'est un signal
# d'alarme : un réseau ne sait pas dire "je ne sais pas" hors distribution.
