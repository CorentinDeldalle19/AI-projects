import string
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# Importation des données
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=["Label", "Message"])

# Fonction pour le nettoyage du texte
def cleanText(text):
    text = text.lower()

    # Suppression caractères non alphanumériques
    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    # Suppression des mots vides
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    return ' '.join(words)

# Appliquer cette fonction à chaque message
df['Cleaned_Message'] = df['Message'].apply(cleanText)

# Transformation du texte en vecteurs numériques
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df['Cleaned_Message']).toarray()

# Entraînement d'un modèle Naive Bayes
# Définir la variable cible (spam ou non spam)
y = df['Label'].apply(lambda x: 1 if x == 'spam' else 0)

# Séparation des données train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Prédictions
y_pred = nb_model.predict(X_test)

# Évaluation du modèle
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")