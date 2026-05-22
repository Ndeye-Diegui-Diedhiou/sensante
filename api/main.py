# api/main.py
# SenSante API - Assistant pre-diagnostic medical
# Lab 3 - Integration de Modeles IA - ESP / UCAD

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import Dict
from pathlib import Path

import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Client Groq (chargé au démarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. / explain sera desactive.")
    
# --- Configuration du Prompt Système ---
SYSTEM_PROMPT = """
Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple, comme un médecin parlerait à son patient.
Sois rassurant mais recommande toujours une consultation médicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni.
"""

# --- Modèles de données ---
class ExplainInput(BaseModel):
    sexe: str
    age: int
    region: str
    temperature: float
    diagnostic: str
    probabilite: float

class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"

# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)

# Servir le frontend comme fichier statique
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
def serve_frontend():
    """Servir la page d'accueil."""
    return FileResponse(frontend_dir / "index.html")

from fastapi.middleware.cors import CORSMiddleware
# Autoriser les requetes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # En dev : tout accepter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Route API ---
@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    """Expliquer un diagnostic en français avec un LLM."""
    
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Clé API non configurée.",
            modele_llm="aucun"
        )

    # Construction du prompt utilisateur
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, région {data.region}\n"
        f"Température : {data.temperature}°C\n"
        f"Diagnostic du modèle : {data.diagnostic} (probabilité {data.probabilite:.0%})\n"
        f"Explique ce résultat au patient."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content or "Aucune explication disponible."
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"

    return ExplainOutput(explication=explication)
# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    # ge=0 empechera l'age negatif au niveau du middleware FastAPI
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

# --- Chargement des artefacts ---
# On charge les modèles au démarrage pour plus d'efficacité
try:
    model = joblib.load("models/model.pkl")
    le_sexe = joblib.load("models/encoder_sexe.pkl")
    le_region = joblib.load("models/encoder_region.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    print("Tous les modeles et encodeurs ont été chargés.")
except Exception as e:
    print(f"Erreur de chargement : {e}")

# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}

# Exercice 1 : Informations sur le modèle
@app.get("/model-info")
def get_model_info():
    return {
        "type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", "N/A"),
        "classes": list(model.classes_),
        "n_features": len(feature_cols),
        "features_names": list(feature_cols)
    }

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    # 1. Encodage du sexe
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utilisez 'M' ou 'F'."
        )

    # 2. Encodage de la région
    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0, confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # 3. Préparation des caractéristiques
    # Créer un dictionnaire avec les données du patient
    data_dict = {
        'age': patient.age,
        'sexe': sexe_enc,
        'temperature': patient.temperature,
        'tension_sys': patient.tension_sys,
        'toux': int(patient.toux),
        'fatigue': int(patient.fatigue),
        'maux_tete': int(patient.maux_tete),
        'region': region_enc
    }
    
    # Construire le vecteur de features en utilisant l'ordre de feature_cols
    features = np.array([[data_dict.get(col, 0) for col in feature_cols]])

    # 4. Prédiction et Probabilités
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())
    
    confiance = ("haute" if proba_max >= 0.7
                else "moyenne" if proba_max >= 0.4
                else "faible")

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
        message=messages.get(diagnostic, "Consultez un medecin.")
    )
