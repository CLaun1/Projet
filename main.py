import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# charger les données
df = pd.read_csv("movies.csv")

#nettoyer les données ( supprimer les lignes vides)
df = df.dropna()

#creer une colonne "tags" (fusion type + description)
df["tags"] = df["type"] + " " +df["description"]

#transformer le texte en vecteurs
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

#  Vérifier le résultat
print("Shape des vecteurs :", vectors.shape)  # nombre de films et nombre de mots
print("Vecteur du premier film :", vectors[0])

#similarité entre vecteur 
similarity = cosine_similarity(vectors)

def recommend(movie_name):
    movie_name = movie_name.lower().strip()
    df['title'] = df['title'].str.lower().str.strip()
    
    #verifier que le film existe bien dans notre liste
    filtered = df[df['title'] == movie_name]
    if len(filtered) == 0:
        print("film non trouvé ! Verifie le titre.")
        return
    
    #récuperer l'index du film choisi
    index = filtered.index[0]

    #creation de tuple index + valeur de similarité 
    scores = list(enumerate(similarity[index]))

    #trier par ordre décroissant (les plus similaires au moins)
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    #les films recommandés en ignorant lui même 
    recommended = sorted_scores[1:6]

    #afficher les films les plus recommandés
    print(f"Films recommandés pour '{movie_name}' :")
    for i in recommended:
        film_index = i[0]          # récupère l'index
        film_title = df.iloc[film_index]['title']  # récupère le titre
        print(film_title)

recommend("titanic")