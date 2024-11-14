import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Charger les données
housingData = fetch_california_housing(as_frame=True)
df = housingData.frame

# Vérifier la présence de valeurs manquantes
# print(df.isnull().sum())

# Analyse des corrélations pour la prédiction du prix médian
# Matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Visualisation de la distribution de la variable cible (prix)
sns.histplot(df['MedHouseVal'], kde=True)
plt.title("Distribution des prix des maisons")
plt.xlabel("Prix")
plt.ylabel("Fréquence")
plt.show()

# Division des données en ensembles d’entraînement et de test
# Définition des caractéristiques (X) et de la cible (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train, y_test)

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train, X_test)

# Entraînement des modèles de régression
# Modèle de régression linéaire
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Modèle de régression avec RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluer les performances du modèle (prédictions)
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Calcul des erreurs
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

if rmse_rf < 1:
    print("Excellent : Le modèle a une précision élevé avec un RMSE > 1")
elif 1 <= rmse_rf < 2:
    print("Bon : Le modèle a une bonne précision mais il pourrait encore être optimisé")
elif rmse_rf > 2:
    print("Médiocre : Le modèle a une précision insuffisante et mérite des améliorations")