# 🎬 Movie Recommendation System

Un moteur de recommandation de films basé sur la similarité de contenu, développé en Python.

---

## 📌 Description

Ce projet implémente un système de recommandation de films par **content-based filtering** : à partir du titre d'un film, le programme analyse son type et sa description pour suggérer 5 films au contenu similaire.

Il n'utilise pas les notes des utilisateurs ni leurs historiques — uniquement les caractéristiques intrinsèques de chaque film.

---

## ⚙️ Fonctionnement

1. **Chargement des données** — lecture d'un fichier `movies.csv` contenant titre, type et description de chaque film
2. **Nettoyage** — suppression des lignes incomplètes
3. **Feature engineering** — création d'une colonne `tags` fusionnant le type et la description
4. **Vectorisation** — transformation du texte en vecteurs numériques avec `CountVectorizer` (TF, 5000 features, sans stop words anglais)
5. **Similarité cosinus** — calcul de la similarité entre tous les films
6. **Recommandation** — tri des scores et retour des 5 films les plus proches

---

## 🛠️ Technologies utilisées

| Outil | Rôle |
|---|---|
| Python 3 | Langage principal |
| Pandas | Chargement et manipulation des données |
| Scikit-learn | Vectorisation (`CountVectorizer`) |
| Scikit-learn | Similarité cosinus (`cosine_similarity`) |

---

## 📁 Structure du projet

```
📦 movie-recommender/
├── movies.csv          # Dataset des films
├── recommender.py      # Script principal
└── README.md
```

---

## 🚀 Installation & Utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/CLAUN1/Projet.git
cd Projet
```

### 2. Installer les dépendances

```bash
pip install pandas scikit-learn
```

### 3. Lancer une recommandation

```bash
python recommender.py
```

Par défaut, le script recommande des films similaires à **Titanic**. Pour changer le film, modifie la dernière ligne du script :

```python
recommend("inception")
```

---

## 📊 Exemple de sortie

```
Films recommandés pour 'titanic' :
the notebook
a walk to remember
pearl harbor
romeo + juliet
ghost
```

---

## 📂 Format du dataset

Le fichier `movies.csv` doit contenir au minimum ces colonnes :

| Colonne | Description |
|---|---|
| `title` | Titre du film |
| `type` | Genre / type (ex: Drama, Action…) |
| `description` | Synopsis ou description du film |

---

## 💡 Améliorations possibles

- Utiliser **TF-IDF** à la place de CountVectorizer pour mieux pondérer les mots rares
- Ajouter le genre, le casting ou le réalisateur dans les `tags`
- Créer une interface web avec **Streamlit**
- Intégrer l'API **TMDB** pour enrichir les données et afficher les affiches

---

## 👤 Auteur

**Nathan Tegua**
[LinkedIn](https://linkedin.com/in/nathan-tegua) · [GitHub](https://github.com/CLAUN1)
