# Rapport de Projet — Typologies de Deep Learning
**Bibliothèque de Deep Learning from Scratch en C++20**

- **Auteur :** Mohammed Ifkirne
- **Module :** Typologies de Deep Learning (Master)
- **Langage & Standard :** C++20 (Bibliothèque standard C++ uniquement, sans dépendance externe)
- **Environnement de développement :** CMake, MSVC / Clang / Ninja (CLion)

---

## 1. Architecture de la Bibliothèque `dl`

La bibliothèque est située dans le dossier `core/include/dl/` et s'articule autour des composants suivants :

### 1.1. Gestion des Tenseurs (`tensor.hpp`)
La classe `dl::Tensor<T>` représente des structures de données N-dimensionnelles.
- **Stockage contigu et drapeau `contiguous_` :** Les données sont stockées sous forme d'un tableau contigu unidimensionnel (`std::shared_ptr<T[]>`). Pour éviter tout surcoût d'allocation mémoire dynamique lors des parcours, un indicateur de contiguïté permet un accès direct `data_[i]` en $O(1)$.
- **Gestion des foulées (Strides) & Vues :** Gestion des opérations de transposition (`transpose`) et de redimensionnement (`reshape`) sans copie inutile.
- **Opérations matricielles et vectorielles :** Implémentation optimisée du produit matriciel (`matmul`) et du produit extérieur (`outer`).
- **Fonctions mathématiques et réductions :** Support des opérations élémentaires (`exp`, `log`, `sqrt`, `tanh`, `pow`, `clamp`) et des réductions (`sum`, `mean`, `max`, `argmax`).

### 1.2. Interface des Couches (`layer.hpp` et `sequential.hpp`)
- **`Layer<T>` :** Classe abstraite de base définissant l'interface avec les méthodes `forward()`, `backward()` et `set_training(bool)` pour la gestion des modes d'entraînement et d'évaluation.
- **`Sequential<T>` :** Modèle conteneur permettant l'empilement séquentiel de couches arbitraires et la propagation automatique des gradients lors de la rétropropagation.

### 1.3. Fonctions d'Activation (`activations.hpp`)
Implémentation des fonctions d'activation et de leurs dérivées analytiques :
- **`ReLU` :** Activation redressée $f(x) = \max(0, x)$.
- **`Sigmoid` :** Activation sigmoïde $f(x) = \frac{1}{1 + e^{-x}}$.
- **`Tanh` :** Tangente hyperbolique.
- **`Softmax` :** Normalisation en probabilités sur un vecteur d'entrée.

### 1.4. Optimisateurs (`optimizers.hpp`)
Gestion de la mise à jour des paramètres des couches :
- **`SGD` :** Descente de gradient stochastique classique.
- **`SGDMomentum` :** Descente de gradient avec quantité de mouvement.
- **`Adam` :** Optimisateur adaptatif combinant l'estimation des moments d'ordre 1 (moyenne) et d'ordre 2 (variance non centrée).

### 1.5. Couches Paramétrées et Spatiales (`layers.hpp`)
- **`Dense` :** Couche entièrement connectée avec initialisation de He/Kaiming ou Xavier/Glorot.
- **`Dropout` :** Régularisation par désactivation aléatoire de neurones.
- **`Reshape` & `Flatten` :** Manipulation de la forme des tenseurs entre les couches spatiales et denses.
- **`Conv2D` :** Convolution 2D avec support du stride, du zero-padding et rétropropagation complète des gradients par rapport aux filtres, aux biais et aux entrées.
- **`MaxPool2D` :** Suréchantillonnage maximum 2D avec conservation des indices maximaux pour le routage exact des gradients.
- **`Embedding` :** Table de projection d'identifiants de mots en vecteurs denses.
- **`RNNCell` :** Cellule récurrente d'Elman $h_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$.
- **`LSTMCell` :** Cellule récurrente à mémoire long/court terme à 4 portes (Forget, Input, Candidate, Output).
- **`GRUCell` :** Cellule récurrente à portes simplifiée à 3 portes (Reset, Update, Candidate).
- **`PositionalEncoding` :** Encodage positionnel sinusoïdal pour les modèles Transformer.
- **`SingleHeadAttention` :** Mécanisme d'attention simple tête (Self-Attention et Cross-Attention) avec produit scalaire itéré, masquage causal et rétropropagation complète.

---

## 2. Emplacement et Description des Modèles (`models/`)

Chaque modèle est autonome et réside dans son propre sous-dossier avec son fichier `CMakeLists.txt` et son fichier source C++ :

### 2.1. Perceptron (`models/perceptron/perceptron.cpp`)
- **Description :** Implémentation d'un perceptron linéaire simple pour vérifier la séparation de données élémentaires.

### 2.2. ANN / MLP pour MNIST (`models/ann/ann.cpp`)
- **Emplacement :** `models/ann/ann.cpp`
- **Architecture :** $784 \to \text{Dense}(128) \to \text{Sigmoid} \to \text{Dense}(64) \to \text{Sigmoid} \to \text{Dense}(10) \to \text{Softmax}$.
- **Initialisation & Entraînement :** Initialisation de LeCun normal, SGD avec $lr=0.1$, entraîné sur l'ensemble complet de 60 000 images MNIST.
- **Performance :** Atteint une précision sur l'ensemble de test d'environ **97.6%**.

### 2.3. CNN pour MNIST (`models/cnn/cnn.cpp`)
- **Emplacement :** `models/cnn/cnn.cpp`
- **Architecture :** $\text{Reshape}(1, 28, 28) \to \text{Conv2D}(1\to 8) \to \text{ReLU} \to \text{MaxPool2D}(2\times 2) \to \text{Conv2D}(8\to 16) \to \text{ReLU} \to \text{MaxPool2D}(2\times 2) \to \text{Flatten} \to \text{Dense}(784\to 128) \to \text{ReLU} \to \text{Dense}(128\to 10) \to \text{Softmax}$.
- **Optimisateur :** Adam ($lr=0.001$).
- **Performance :** Atteint une précision supérieure à **98%** en 3 à 5 époques.

### 2.4. Traducteur RNN Simple (`models/rnn/rnn.cpp`)
- **Emplacement :** `models/rnn/rnn.cpp`
- **Architecture :** Modèle Sequence-to-Sequence (Seq2Seq) Encodeur-Décodeur basé sur des cellules `RNNCell` et des couches `Embedding` pour la traduction de phrases Anglais $\to$ Français avec jeton de fin de séquence `<EOS>`.

### 2.5. Traducteur LSTM (`models/lstm/lstm.cpp`)
- **Emplacement :** `models/lstm/lstm.cpp`
- **Architecture :** Modèle Seq2Seq Encodeur-Décodeur exploitant les cellules `LSTMCell` (gestion conjointe des états cachés $h_t$ et des états cellules $c_t$) pour capturer les dépendances temporelles.

### 2.6. Traducteur GRU (`models/gru/gru.cpp`)
- **Emplacement :** `models/gru/gru.cpp`
- **Architecture :** Modèle Seq2Seq Encodeur-Décodeur utilisant des cellules `GRUCell` à 3 portes.

### 2.7. Traducteur Transformer Simple Tête (`models/transformer/transformer.cpp`)
- **Emplacement :** `models/transformer/transformer.cpp`
- **Architecture :** Modèle Transformer Seq2Seq combinant l'encodage positionnel sinusoïdal, l'auto-attention de l'encodeur, l'auto-attention causale du décodeur et l'attention croisée Encodeur-Décodeur.

---

## 3. Instructions de Compilation et d'Exécution

### 3.1. Prérequis
- Un compilateur C++ supportant le standard **C++20** (GCC 10+, Clang 11+, ou MSVC 2019+).
- **CMake** (version 3.25 ou supérieure).
- **Ninja** ou **Make**.

### 3.2. Compilation depuis la ligne de commande

1. Créer le dossier de compilation et s'y déplacer :
```bash
mkdir build
cd build
```

2. Configurer le projet avec CMake en mode Release (recommandé pour des exécutions rapides) :
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

3. Compiler l'ensemble des cibles :
```bash
cmake --build . --config Release
```

### 3.3. Exécution des Modèles

Après compilation, les exécutables se trouvent dans leurs dossiers respectifs sous le répertoire de build :

- **Exécuter le modèle ANN :**
```bash
./models/ann/ann
```
*(Sur Windows : `.\models\ann\Release\ann.exe` ou via le bouton de lancement CLion)*

- **Exécuter le modèle CNN :**
```bash
./models/cnn/cnn
```

- **Exécuter le traducteur RNN :**
```bash
./models/rnn/rnn
```

- **Exécuter le traducteur LSTM :**
```bash
./models/lstm/lstm
```

- **Exécuter le traducteur GRU :**
```bash
./models/gru/gru
```

- **Exécuter le traducteur Transformer :**
```bash
./models/transformer/transformer
```

- **Exécuter les tests unitaires :**
```bash
./tests/tensor_test
```

---

## 4. Organisation des Dossiers du Projet

```text
typologies-de-deep-learning-ttheghost/
├── CMakeLists.txt                # Configuration CMake racine
├── README.md                     # Rapport du projet
├── core/
│   ├── CMakeLists.txt            # Configuration de la bibliothèque core
│   └── include/
│       ├── dl/                   # En-têtes de la bibliothèque Deep Learning
│       │   ├── activations.hpp   # ReLU, Sigmoid, Tanh, Softmax
│       │   ├── dl.h              # Inclusions générales
│       │   ├── layer.hpp         # Interface abstraite Layer
│       │   ├── layers.hpp        # Dense, Conv2D, RNNCell, LSTMCell, GRUCell, Attention, etc.
│       │   ├── optimizers.hpp    # SGD, SGDMomentum, Adam
│       │   ├── sequential.hpp    # Modèle Séquentiel
│       │   └── tensor.hpp        # Classe Tensor N-dimensionnelle
│       └── helper/               # Utilitaires d'assistance
│           ├── mnist.hpp         # Chargeur de données MNIST
│           └── utils.hpp         # Utilitaires de chemins de fichiers
├── datasets/                     # Jeux de données (MNIST binary ubyte)
├── models/                       # Modèles d'application
│   ├── ann/                      # Modèle ANN pour MNIST
│   ├── cnn/                      # Modèle CNN pour MNIST
│   ├── gru/                      # Modèle Traducteur GRU
│   ├── lstm/                     # Modèle Traducteur LSTM
│   ├── perceptron/               # Modèle Perceptron
│   ├── rnn/                      # Modèle Traducteur RNN
│   └── transformer/              # Modèle Traducteur Transformer
└── tests/                        # Tests unitaires C++
    ├── CMakeLists.txt
    └── tensor.cpp
```
