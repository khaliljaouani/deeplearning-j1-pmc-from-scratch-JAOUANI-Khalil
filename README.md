# Deep Learning J1 — PMC from scratch

Premier projet du mini-portfolio Deep Learning : un Perceptron Multi-Couches
codé entièrement en numpy — poids, forward pass, backpropagation et boucle
d'entraînement à la main — puis comparé à l'API Keras sur les mêmes problèmes.

## Les phases

| Fichier | Contenu | Résultat |
|---|---|---|
| `phase1_neurone.py` | Neurone unique : forward pass sigmoid + loss BCE, poids fixés | Loss 0.7519 sur 4 points |
| `phase2_gradient.py` | Descente de gradient à la main, courbe de loss par epoch | Loss 0.69 → 0.59 en 50 epochs |
| `phase3_xor.py` | XOR : réseau 2-2-1, backprop 2 couches, frontière de décision | 100% accuracy |
| `phase4_spirale.py` | Spirale 2D : réseau 2-64-64-1, ReLU + init He | 99.75% accuracy |
| `phase5_keras_mnist.py` | Le même genre de réseau en ~15 lignes de Keras, sur MNIST | 97.7% test accuracy en 5 epochs |
| `phase6_activations.py` | sigmoid vs tanh vs relu : tableau + courbes superposées | ReLU/tanh convergent dès l'epoch 3, sigmoid à l'epoch 5 |
| `phase7_learning_rate.py` | 1e-7 vs 1e-3 vs 1.0 : le sweet spot du learning rate | 1e-3 → 97.7% ; les extrêmes n'apprennent rien (~10-18%) |
| `phase8_pipeline_personnel.py` | Pipeline complet numpy + Keras sur Breast Cancer (30 features) | 97.4% des deux côtés |

Chaque script sauvegarde ses graphes en `.png` (courbes de loss, frontières
de décision, comparatifs).

## Lancer

```bash
pip install numpy matplotlib tensorflow scikit-learn
python phase1_neurone.py   # etc.
```

## Ce que j'ai observé en chemin

- **Phase 2** : le gradient moyenné `(1/n) X^T (ŷ-y)` et le gradient sommé
  ne diffèrent que par la taille de pas effective (facteur n) — même
  direction, convergence plus ou moins rapide à learning rate égal.
- **Phase 3** : avec les biais initialisés à zéro, le réseau 2-2-1 tombe
  dans un minimum local (loss ~0.35, accuracy 50%). Des biais aléatoires
  suffisent à briser la symétrie et atteindre 100%. Sur 4 points,
  l'initialisation compte énormément.
- **Phase 4** : lr=0.01 avec gradient moyenné sur 400 points → 71% en 2000
  epochs ; lr=0.3 → 99.75%. Le learning rate doit se juger relativement à
  l'échelle des gradients.
- **Phase 7** : à lr=1.0, la loss oscille autour de 2.3 = -log(1/10),
  c'est-à-dire une prédiction uniforme sur 10 classes : le réseau n'apprend
  littéralement rien.
- **Phase 8** : sur une entrée hors distribution (99999 partout), le modèle
  prédit 1.0000 en toute confiance. Un réseau ne sait pas dire « je ne sais
  pas » : signal d'alarme pour la production.
