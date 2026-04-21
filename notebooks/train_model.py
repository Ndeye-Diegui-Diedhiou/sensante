import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Charger le dataset
df = pd.read_csv("data/patients_dakar.csv")

# Verifier les dimensions
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

#partie 2
# Encoder les variables categoriques en nombres
# Le modele ne comprend que des nombres !
le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])
# Definir les features (X) et la cible (y)
# Utilise exactement ces noms (ceux qui sont dans ton CSV)
feature_cols = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys', 
    'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']
print(f"Features : {X.shape}") # (500, 8)
print(f"Cible : {y.shape}")# (500,)

#partie 3
# 80% pour l'entrainement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
# 20% pour le test
random_state=42,
# Pour avoir les memes resultats a chaque fois
stratify=y
# Garder les memes proportions de diagnostics
)
print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#partie 4
from sklearn.ensemble import RandomForestClassifier
# Creer le modele
model = RandomForestClassifier(
n_estimators=100,
# 100 arbres de decision
random_state=42
# Reproductibilite
)
# Entrainer sur les donnees d'entrainement
model.fit(X_train, y_train)
print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")

#partie 5
# Predire sur les donnees de test
y_pred = model.predict(X_test)
# Comparer les 10 premieres predictions avec la realite
comparison = pd.DataFrame({
'Vrai diagnostic': y_test.values[:10],
'Prediction': y_pred[:10]
})
print(comparison)
#etape 5.2
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

#etape 5.3
from sklearn.metrics import confusion_matrix, classification_report
# Pour la visualisation (Seaborn)
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("Matrice de confusion :")
print(cm)

# 2. Préparation du dossier de sauvegarde
if not os.path.exists('figures'):
    os.makedirs('figures')

# 3. Visualisation avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)

plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()

# 4. Sauvegarde et affichage
plt.savefig('figures/confusion_matrix.png', dpi=150)
print("Figure sauvegardee dans figures/confusion_matrix.png")

plt.show()

# --- Rapport de classification ---
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

import joblib
import os
# Creer le dossier models/ s'il n'existe pas
os.makedirs("models", exist_ok=True)
# Serialiser le modele
joblib.dump(model, "models/model.pkl")
# Verifier la taille du fichier
size = os.path.getsize("models/model.pkl")
print(f"Modele sauvegarde : models/model.pkl")
print(f"Taille : {size / 1024:.1f} Ko")

# Sauvegarder les encodeurs (indispensables pour les nouvelles donnees)
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
# Sauvegarder la liste des features (pour reference)
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("Encodeurs et metadata sauvegardes.")

# Simuler ce que fera l'API en Lab 3 :
# Charger le modele DEPUIS LE FICHIER (pas depuis la memoire)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

# Un nouveau patient arrive au centre de sante de Medina
# Un nouveau patient arrive
# 1. Un nouveau patient avec les 10 infos nécessaires
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'frissons': True,   # Ajouté pour correspondre au nouveau modèle
    'nausee': False,    # Ajouté pour correspondre au nouveau modèle
    'region': 'Dakar'
}


# 2. On prépare la liste des 10 features (Vérifie bien l'ordre !)
# age, sexe_enc, temp, tension, toux, fatigue, maux_tete, frissons, nausee, region_enc
# L'ordre DOIT être : age, sexe, temp, tension, toux, fatigue, maux_tete, frissons, nausee, region
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]
features = [
    nouveau_patient['age'],                             # 28
    sexe_enc,                                           # 0 (pour F)
    nouveau_patient['temperature'],                    # 39.5
    nouveau_patient['tension_sys'],                    # 110
    int(nouveau_patient['toux']),                       # 1
    int(nouveau_patient['fatigue']),                    # 1
    int(nouveau_patient['maux_tete']),                  # 1
    int(nouveau_patient['frissons']),                   # 1
    int(nouveau_patient['nausee']),                     # 0
    region_enc                                          # Code pour Dakar
]
# 3. On lance la prédiction
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]

# L'affichage des barres devrait maintenant apparaître !
print(f"\n--- Resultat du pre-diagnostic ---")
print(f"Diagnostic : {diagnostic}")
for classe, proba in zip(model_loaded.classes_, probas):
 bar = '#' * int(proba * 30)
 print(f" {classe:8s} : {proba:.1%} {bar}")

# Prédire
diagnostic = model_loaded.predict([features])[0]

#Exercice 1
importances = model.feature_importances_
for name, imp in sorted(zip(feature_cols, importances),
    key=lambda x: x[1], reverse=True):
    print(f" {name:20s} : {imp:.3f}")
    
# 1. Liste exacte des colonnes attendues par le modèle (d'après ton log)
features_train = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys', 'toux', 
    'fatigue', 'maux_tete', 'frissons', 'nausee', 'region_encoded'
]

# 2. Création des profils avec TOUTES les colonnes
# On met des valeurs neutres (0) pour les colonnes qu'on ne teste pas spécifiquement
data_test = [
    # Patient 1 : Jeune, sain (sexe=0, tension=120, region=0)
    [19, 0, 37.0, 120, 0, 0, 0, 0, 0, 0],
    # Patient 2 : Adulte, fièvre, toux, maux de tête, frissons
    [35, 1, 39.8, 130, 1, 1, 1, 1, 0, 1],
    # Patient 3 : Âgé, légère fièvre, fatigue
    [72, 0, 37.6, 140, 1, 1, 0, 0, 1, 2]
]

# 3. Création du DataFrame avec les bons noms de colonnes
patients = pd.DataFrame(data_test, columns=features_train)

# 4. Maintenant la prédiction fonctionnera
predictions = model.predict(patients)
probabilites = model.predict_proba(patients)

print("--- Résultats des Tests ---")
for i, pred in enumerate(predictions):
    print(f"Patient {i+1} : {pred} ({np.max(probabilites[i])*100:.1f}%)")