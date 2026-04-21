# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP / UCAD

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    # Validation stricte : l'age doit être entre 0 et 120
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

# --- Chargement du modele (une seule fois au démarrage) ---
try:
    print("Chargement du modele et des encodeurs...")
    model = joblib.load("models/model.pkl")
    le_sexe = joblib.load("models/encoder_sexe.pkl")
    le_region = joblib.load("models/encoder_region.pkl")
    # feature_cols n'est pas utilisé directement ici mais peut servir à vérifier l'ordre des colonnes
    feature_cols = joblib.load("models/feature_cols.pkl")
    print(f"Modele charge avec succès. Classes : {list(model.classes_)}")
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")

# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # Encodage du Sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", 
            probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide (attendu: M/F) : {patient.sexe}"
        )

    # Encodage de la Région
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", 
            probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # Préparation des caractéristiques (Features)
    features = np.array([[
        patient.age, 
        sexe_enc, 
        patient.temperature,
        patient.tension_sys, 
        int(patient.toux),
        int(patient.fatigue), 
        int(patient.maux_tete),
        region_enc
    ]])

    # Prediction
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())
    
    # Détermination du niveau de confiance
    confiance = ("haute" if proba_max >= 0.7 
                else "moyenne" if proba_max >= 0.4 
                else "faible")

    # Messages personnalisés selon le diagnostic
    messages = {
        "palu": "Suspicion de paludisme. Consultez rapidement un centre de santé.",
        "grippe": "Suspicion de grippe. Repos et hydratation conseillés.",
        "typh": "Suspicion de typhoide. Une analyse de sang est nécessaire.",
        "sain": "Pas de pathologie majeure détectée. Restez vigilant."
    }

    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin pour un avis approfondi.")
    )