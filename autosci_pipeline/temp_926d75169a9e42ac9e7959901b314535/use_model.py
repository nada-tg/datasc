# Script d'utilisation du modèle 926d7516
import json
import joblib
import pickle
import os

# Charger les métadonnées
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

print(f"Modèle: {model_info['model_name']}")
print(f"Framework: {model_info['framework']}")
print(f"Type: {model_info['model_type']}")

# Charger le modèle (adapter selon votre framework)
try:
    # Pour scikit-learn
    if os.path.exists('model.pkl'):
        model = joblib.load('model.pkl')
    elif os.path.exists('model.joblib'):
        model = joblib.load('model.joblib')
    else:
        print("Fichier modèle non trouvé")
        
    print("Modèle chargé avec succès!")
    
    # Exemple d'utilisation
    # predictions = model.predict(your_data)
    
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
