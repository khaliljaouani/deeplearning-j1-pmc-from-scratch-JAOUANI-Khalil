"""Phase 5 : passage à Keras sur MNIST flatten.

Le même genre de réseau que les phases 1-4, mais en ~15 lignes :
forward pass, backprop et boucle d'entraînement sont automatiques.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Préprocessing : flatten 28x28 -> 784, normaliser entre 0 et 1
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Classes uniques : {np.unique(y_train)}")

# Architecture : Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

# sparse_categorical_crossentropy : étiquettes entières (0-9), pas one-hot
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

start = time.time()
history = model.fit(
    X_train, y_train,
    epochs=5, batch_size=64,
    validation_split=0.1, verbose=1,
)
elapsed = time.time() - start

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTemps d'entraînement : {elapsed:.1f}s")
print(f"Test accuracy : {test_acc:.4f}")
print(f"Test loss : {test_loss:.4f}")

# Comparaison avec le from-scratch : ~80 lignes numpy pour 2 couches,
# backprop à la main, des heures sur MNIST — contre 15 lignes et 97% en
# 5 epochs ici. Keras fait la même chose, en C/CUDA optimisé.

# Scénario adversarial (batch_size=1 = SGD pur) : mise à jour des poids
# après chaque exemple -> 60 000 updates/epoch au lieu de ~938, temps
# d'entraînement x20+ et courbe de loss beaucoup plus bruitée.

# Courbes (code fourni — repose sur history.history)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'], label='train')
axes[0].plot(history.history['val_loss'], label='val')
axes[0].set_title("Loss"); axes[0].legend()
axes[1].plot(history.history['accuracy'], label='train')
axes[1].plot(history.history['val_accuracy'], label='val')
axes[1].set_title("Accuracy"); axes[1].legend()
plt.savefig("phase5_mnist_curves.png", dpi=100, bbox_inches='tight')
print("Courbes sauvegardées : phase5_mnist_curves.png")
