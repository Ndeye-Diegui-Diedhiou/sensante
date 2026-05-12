# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# Charger la clé depuis .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvée dans .env")
    exit()

# Créer le client Groq
client = Groq(api_key=api_key)

# Premier appel : question simple
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": (
                "Tu es un assistant médical sénégalais. "
                "Réponds en français simple. "
                "Maximum 3 phrases."
            ),
        },
        {
            "role": "user",
            "content": "Quels sont les symptômes du paludisme ?",
        },
    ],
    max_tokens=200,
    temperature=0.3,
)

# Afficher la réponse
print("=== Réponse de Llama 3 ===")
print(response.choices[0].message.content)
if response.usage:
    print(f"\nTokens utilisés : {response.usage.total_tokens}")
    
# Test avec le format SenSante
response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": """Tu es un assistant medical senegalais.

    Tu recois un diagnostic et des donnees patient.
    Explique le resultat en francais simple,
    comme un medecin parlerait a son patient.
    Sois rassurant mais recommande une consultation.
    Maximum 3 phrases.
    Ne fais JAMAIS de diagnostic toi-meme."""
        },
        {
            "role": "user",
            "content": """Patient: Femme, 28 ans, region Dakar
    Symptomes: temperature 39.5, toux, fatigue, maux de tete
    Diagnostic du modele: paludisme (probabilite 72%)
    Explique ce resultat au patient."""
        }
    ],
    max_tokens=200,
    temperature=0.3
)
print ("=== Explication SenSante === ")
print ( response2 . choices [0]. message . content )    