# deployment_platform.py - API de déploiement et gestion des modèles

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
import os
import pickle
import joblib
import sqlite3
from datetime import datetime, timedelta
import requests
import zipfile
import io
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

# Modèles Pydantic
class ModelDeployment(BaseModel):
    run_id: str
    model_name: str
    framework: str
    model_type: str
    metrics: Dict[str, float]
    timestamp: str

class PredictionRequest(BaseModel):
    input: Any
    format: str = "json"  # json, csv, text

class PaymentRequest(BaseModel):
    model_id: str
    purchase_type: str  # "buy" or "subscription"
    price: float
    email: str
    payment_method: Dict[str, str]

# Configuration de la base de données
def init_database():
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    # Table des modèles déployés
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deployed_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            model_name TEXT NOT NULL,
            framework TEXT,
            model_type TEXT,
            metrics TEXT,
            deployment_date TEXT,
            status TEXT DEFAULT 'active',
            model_path TEXT,
            pricing TEXT
        )
    ''')
    
    # Table des transactions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            purchase_type TEXT,
            price REAL,
            email TEXT,
            transaction_id TEXT,
            status TEXT,
            created_at TEXT,
            expires_at TEXT
        )
    ''')
    
    # Table des utilisateurs et abonnements
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            model_id TEXT,
            subscription_type TEXT,
            status TEXT,
            created_at TEXT,
            expires_at TEXT,
            usage_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_database()
    yield
    # Shutdown
    pass

app = FastAPI(
    title="AutoSci Model Deployment Platform",
    description="Plateforme de déploiement et monétisation de modèles ML",
    version="1.0.0",
    lifespan=lifespan
)

# Gestionnaire de modèles déployés
class ModelManager:
    def __init__(self):
        self.models_dir = "deployed_models"
        os.makedirs(self.models_dir, exist_ok=True)
        self.loaded_models = {}
    
    def save_model(self, run_id: str, model_data: bytes) -> str:
        """Sauvegarde un modèle déployé"""
        model_path = os.path.join(self.models_dir, f"{run_id}.zip")
        with open(model_path, 'wb') as f:
            f.write(model_data)
        return model_path
    
    def load_model(self, run_id: str):
        """Charge un modèle en mémoire"""
        if run_id in self.loaded_models:
            return self.loaded_models[run_id]
        
        model_path = os.path.join(self.models_dir, f"{run_id}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle {run_id} non trouvé")
        
        # Extraire et charger le modèle
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            temp_dir = f"temp_{run_id}"
            zip_ref.extractall(temp_dir)
            
            # Tenter de charger avec différentes méthodes
            for filename in os.listdir(temp_dir):
                if filename.endswith('.pkl') or filename.endswith('.joblib'):
                    model_file = os.path.join(temp_dir, filename)
                    try:
                        if filename.endswith('.pkl'):
                            model = pickle.load(open(model_file, 'rb'))
                        else:
                            model = joblib.load(model_file)
                        
                        self.loaded_models[run_id] = model
                        return model
                    except Exception as e:
                        print(f"Erreur chargement {filename}: {e}")
                        continue
        
        raise Exception(f"Impossible de charger le modèle {run_id}")
    
    def predict(self, run_id: str, data: Any):
        """Effectue une prédiction"""
        model = self.load_model(run_id)
        
        # Préparation des données selon le type
        if isinstance(data, dict):
            if 'input' in data:
                input_data = data['input']
                
                # Conversion en format utilisable par le modèle
                if isinstance(input_data, str):
                    # Pour les modèles de texte, utiliser un vectoriseur ou encoder
                    input_array = self._text_to_features(input_data)
                elif isinstance(input_data, list):
                    input_array = np.array(input_data).reshape(1, -1)
                else:
                    input_array = np.array([[input_data]])
                
                # Prédiction
                prediction = model.predict(input_array)
                
                # Si le modèle supporte les probabilités
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(input_array)
                        confidence = np.max(probabilities)
                    except:
                        confidence = 0.8  # Valeur par défaut
                else:
                    confidence = 0.8
                
                return {
                    "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                    "confidence": float(confidence),
                    "probabilities": probabilities.tolist() if probabilities is not None else None
                }
        
        raise ValueError("Format de données non supporté")
    
    def _text_to_features(self, text: str):
        """Conversion basique de texte en features numériques"""
        # Implémentation simplifiée - dans la réalité, utiliser un vrai vectoriseur
        features = [len(text), text.count(' '), len(set(text.lower()))]
        return np.array(features).reshape(1, -1)

model_manager = ModelManager()

# Endpoints de l'API

@app.post("/deploy")
async def deploy_model(deployment: ModelDeployment, background_tasks: BackgroundTasks):
    """Déploie un modèle sur la plateforme"""
    try:
        # Récupérer le modèle depuis MLflow
        response = requests.get(f"http://localhost:8000/models/{deployment.run_id}/download")
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Modèle non trouvé")
        
        # Sauvegarder le modèle
        model_path = model_manager.save_model(deployment.run_id, response.content)
        
        # Enregistrer en base de données
        conn = sqlite3.connect('deployment_platform.db')
        cursor = conn.cursor()
        
        pricing = json.dumps({
            "buy_price": 99.99,
            "subscription_price": 29.99
        })
        
        cursor.execute('''
            INSERT OR REPLACE INTO deployed_models 
            (run_id, model_name, framework, model_type, metrics, deployment_date, model_path, pricing)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            deployment.run_id,
            deployment.model_name,
            deployment.framework,
            deployment.model_type,
            json.dumps(deployment.metrics),
            deployment.timestamp,
            model_path,
            pricing
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "model_url": f"/model/{deployment.run_id}",
            "prediction_endpoint": f"/predict/{deployment.run_id}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{run_id}")
async def get_deployment_status(run_id: str):
    """Vérifie le statut de déploiement d'un modèle"""
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT status FROM deployed_models WHERE run_id = ?', (run_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"is_deployed": True, "status": result[0]}
    else:
        return {"is_deployed": False}

@app.get("/model/{run_id}")
async def get_model_info(run_id: str):
    """Récupère les informations d'un modèle déployé"""
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT model_name, framework, model_type, metrics, deployment_date, status, pricing
        FROM deployed_models WHERE run_id = ?
    ''', (run_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    return {
        "model_name": result[0],
        "framework": result[1],
        "model_type": result[2],
        "metrics": json.loads(result[3]) if result[3] else {},
        "deployment_date": result[4],
        "status": result[5],
        "pricing": json.loads(result[6]) if result[6] else {}
    }

@app.post("/predict/{run_id}")
async def predict(run_id: str, request: PredictionRequest):
    """Effectue une prédiction avec un modèle déployé"""
    try:
        # Vérifier l'accès (abonnement/achat)
        # access_granted = check_user_access(run_id, request.user_email)
        # if not access_granted:
        #     raise HTTPException(status_code=403, detail="Accès non autorisé")
        
        # Effectuer la prédiction
        result = model_manager.predict(run_id, {"input": request.input})
        
        # Incrémenter le compteur d'usage pour les abonnements
        # update_usage_count(run_id, request.user_email)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/purchase")
async def process_purchase(payment: PaymentRequest):
    """Traite un achat ou abonnement"""
    try:
        # Simuler le traitement de paiement
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Enregistrer la transaction
        conn = sqlite3.connect('deployment_platform.db')
        cursor = conn.cursor()
        
        expires_at = None
        if payment.purchase_type == "subscription":
            expires_at = (datetime.now() + timedelta(days=30)).isoformat()
        
        cursor.execute('''
            INSERT INTO transactions 
            (model_id, purchase_type, price, email, transaction_id, status, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            payment.model_id,
            payment.purchase_type,
            payment.price,
            payment.email,
            transaction_id,
            "completed",
            datetime.now().isoformat(),
            expires_at
        ))
        
        # Si c'est un abonnement, créer l'entrée dans subscriptions
        if payment.purchase_type == "subscription":
            cursor.execute('''
                INSERT INTO subscriptions 
                (email, model_id, subscription_type, status, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                payment.email,
                payment.model_id,
                "monthly",
                "active",
                datetime.now().isoformat(),
                expires_at
            ))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "transaction_id": transaction_id,
            "download_available": payment.purchase_type == "buy",
            "online_access": payment.purchase_type == "subscription",
            "expires_at": expires_at
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{run_id}")
async def download_purchased_model(run_id: str, email: str):
    """Télécharge un modèle acheté"""
    # Vérifier l'achat
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM transactions 
        WHERE model_id = ? AND email = ? AND purchase_type = 'buy' AND status = 'completed'
    ''', (run_id, email))
    
    purchase = cursor.fetchone()
    conn.close()
    
    if not purchase:
        raise HTTPException(status_code=403, detail="Achat non trouvé ou non autorisé")
    
    # Retourner le fichier du modèle
    model_path = os.path.join(model_manager.models_dir, f"{run_id}.zip")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Fichier modèle non trouvé")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        model_path,
        media_type='application/zip',
        filename=f"model_{run_id[:8]}.zip"
    )

@app.get("/user/access/{email}")
async def check_user_access(email: str, model_id: str):
    """Vérifie l'accès d'un utilisateur à un modèle"""
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    # Vérifier achat unique
    cursor.execute('''
        SELECT * FROM transactions 
        WHERE email = ? AND model_id = ? AND purchase_type = 'buy' AND status = 'completed'
    ''', (email, model_id))
    
    buy_access = cursor.fetchone()
    
    # Vérifier abonnement actif
    cursor.execute('''
        SELECT * FROM subscriptions 
        WHERE email = ? AND model_id = ? AND status = 'active' AND expires_at > ?
    ''', (email, model_id, datetime.now().isoformat()))
    
    subscription_access = cursor.fetchone()
    
    conn.close()
    
    return {
        "has_access": bool(buy_access or subscription_access),
        "access_type": "buy" if buy_access else "subscription" if subscription_access else None,
        "expires_at": subscription_access[6] if subscription_access else None
    }

@app.get("/models")
async def list_deployed_models():
    """Liste tous les modèles déployés"""
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT run_id, model_name, framework, model_type, metrics, deployment_date, status, pricing
        FROM deployed_models WHERE status = 'active'
    ''')
    
    models = []
    for row in cursor.fetchall():
        models.append({
            "run_id": row[0],
            "model_name": row[1],
            "framework": row[2],
            "model_type": row[3],
            "metrics": json.loads(row[4]) if row[4] else {},
            "deployment_date": row[5],
            "status": row[6],
            "pricing": json.loads(row[7]) if row[7] else {}
        })
    
    conn.close()
    return models

@app.get("/analytics/model/{run_id}")
async def get_model_analytics(run_id: str):
    """Récupère les analytics d'un modèle"""
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    # Statistiques des ventes
    cursor.execute('''
        SELECT purchase_type, COUNT(*), SUM(price)
        FROM transactions 
        WHERE model_id = ? AND status = 'completed'
        GROUP BY purchase_type
    ''', (run_id,))
    
    sales_stats = {}
    total_revenue = 0
    for row in cursor.fetchall():
        sales_stats[row[0]] = {"count": row[1], "revenue": row[2]}
        total_revenue += row[2]
    
    # Abonnements actifs
    cursor.execute('''
        SELECT COUNT(*) FROM subscriptions 
        WHERE model_id = ? AND status = 'active' AND expires_at > ?
    ''', (run_id, datetime.now().isoformat()))
    
    active_subscriptions = cursor.fetchone()[0]
    
    # Usage récent (7 derniers jours)
    cursor.execute('''
        SELECT SUM(usage_count) FROM subscriptions 
        WHERE model_id = ? AND created_at > ?
    ''', (run_id, (datetime.now() - timedelta(days=7)).isoformat()))
    
    recent_usage = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return {
        "model_id": run_id,
        "sales_statistics": sales_stats,
        "total_revenue": total_revenue,
        "active_subscriptions": active_subscriptions,
        "recent_usage": recent_usage
    }

# Tâches de maintenance
@app.post("/admin/cleanup")
async def cleanup_expired_subscriptions():
    """Nettoie les abonnements expirés"""
    conn = sqlite3.connect('deployment_platform.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE subscriptions 
        SET status = 'expired' 
        WHERE expires_at < ? AND status = 'active'
    ''', (datetime.now().isoformat(),))
    
    expired_count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return {"expired_subscriptions": expired_count}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)