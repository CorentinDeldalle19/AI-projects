# This model uses a supervised learning algorithm, the Random Forest Classifier, to classify the different iris flower
# species in the Iris dataset according to their characteristics.
#
# Model context:
# The Iris dataset contains measurements of flowers from three different iris species: Iris setosa, Iris versicolor,
# and Iris virginica. Each flower is described by four characteristics:
# - Sepal length
# - Sepal width
# - Petal length
# - Petal width
# The aim is to train a model capable of predicting the species of a flower based on these four characteristics.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Chargement des données
from sklearn.datasets import load_iris
data = load_iris()

# Conversion en DataFrame pandas
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = data.target

# Aperçu des premières lignes pour vérifier les données
print(df.head())

# Définition des caractéristiques et de la cible
X = df.drop('species', axis=1)  # Caractéristiques
y = df['species']               # Cible

# Normaliser les données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Prédiction et évaluation du modèle
y_pred = rf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Matrice de confusion
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="YlGnBu", fmt="d")
plt.xlabel("Prédictions")
plt.ylabel("Véritables étiquettes")
plt.title("Matrice de confusion")
plt.show()

# Importance des caractéristiques
feature_importances = pd.Series(rf.feature_importances_, index=data.feature_names)
feature_importances = feature_importances.sort_values(ascending=False)
feature_importances.plot(kind='bar', figsize=(10, 6))
plt.title("Importance des caractéristiques")
plt.xlabel("Caractéristiques")
plt.ylabel("Importance")
plt.show()