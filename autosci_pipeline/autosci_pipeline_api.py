# autosci_main.py - API principale mise à jour

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
import os
import zipfile
import io
import tempfile
import shutil
from datetime import datetime
import json

# Configuration MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(title="AutoSci Pipeline API", version="1.0.0")

# Existing endpoints... (keeping your original endpoints)

"""
AutoSciML - Application principale
Gestion automatisée de données scientifiques avec ML
"""

import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any
from threading import Lock

import numpy as np
import pandas as pd
from pydantic import BaseModel

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim

# MLflow
import mlflow
from mlflow.tracking import MlflowClient

# Web Framework
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler

# Configuration de l'environnement
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configuration des chemins

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "donnees_propres")
MODELS_DIR = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Créer les dossiers nécessaires
for directory in [DATA_DIR, MODELS_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)


# Configuration MLflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("autosci_experiment")
client = MlflowClient()

# Thread safety
LOCK = Lock()


# Configuration par défaut

DEFAULT_CONFIG = {
    "domaine": "demo",
    "specialite": "general",
    "freq_minutes": 60,
    "n_samples": 1000,
    "test_size": 0.2,
    "random_state": 42
}

if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)


# Modèle PyTorch

class SimpleTorchNet(nn.Module):
    """Réseau de neurones simple pour classification binaire"""
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# Classe principale du Pipeline

class AutoSciPipeline:
    def __init__(self):
        self.config = self.load_config()
        self.last_run = {"timestamp": None, "metrics": None, "status": "idle"}
        
    def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier JSON"""
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement de la config: {e}")
            return DEFAULT_CONFIG
    
    def save_config(self, config: Dict[str, Any]):
        """Sauvegarde la configuration"""
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        self.config = config
    
    def generate_synthetic_data(self, domaine: str, specialite: str, n_samples: int = 1000) -> pd.DataFrame:
        """Génère des données synthétiques basées sur le domaine et la spécialité"""
        rng = np.random.default_rng(int(time.time()) % (2**32 - 1))
        
        # Génération des features
        data = {
            "id": np.arange(1, n_samples + 1),
            "domaine": [domaine] * n_samples,
            "specialite": [specialite] * n_samples,
            "timestamp": [datetime.utcnow().isoformat()] * n_samples,
            "feature_1": rng.integers(0, 100, size=n_samples),
            "feature_2": rng.normal(loc=50, scale=15, size=n_samples),
            "feature_3": rng.exponential(scale=10, size=n_samples),
            "feature_4": rng.uniform(0, 1, size=n_samples)
        }
        
        # Features catégorielles
        categories_a = [f"cat_A_{i%5}" for i in range(n_samples)]
        categories_b = [f"cat_B_{i%3}" for i in range(n_samples)]
        data["category_a"] = categories_a
        data["category_b"] = categories_b
        
        # Génération de la target (fonction non-linéaire des features)
        logits = (
            0.03 * data["feature_1"] +
            0.5 * data["feature_2"] / 50 +
            0.2 * data["feature_3"] / 10 +
            rng.normal(0, 0.1, n_samples)
        )
        probs = 1 / (1 + np.exp(-logits))
        data["target"] = (rng.random(n_samples) < probs).astype(int)
        
        df = pd.DataFrame(data)
        
        # Injection de valeurs manquantes (simulation données réelles)
        missing_indices = rng.choice(n_samples, size=int(0.02 * n_samples), replace=False)
        df.loc[missing_indices, "feature_2"] = np.nan
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame):
        """Prétraitement des données"""
        # Copie pour éviter les modifications
        df_processed = df.copy()
        
        # Suppression des doublons
        df_processed = df_processed.drop_duplicates(subset=["id"])
        
        # Imputation des valeurs manquantes
        numeric_features = ["feature_1", "feature_2", "feature_3", "feature_4"]
        for col in numeric_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        # Encodage des variables catégorielles
        categorical_features = ["category_a", "category_b"]
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        cat_encoded = ohe.fit_transform(df_processed[categorical_features])
        
        # Normalisation des features numériques
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(df_processed[numeric_features])
        
        # Combinaison des features
        X = np.hstack([num_scaled, cat_encoded])
        y = df_processed["target"].values
        
        metadata = {
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "n_features": X.shape[1],
            "n_samples": X.shape[0]
        }
        
        return X, y, metadata, ohe, scaler
    
    def train_sklearn_model(self, X_train, y_train, X_test, y_test, run_name: str) -> Dict[str, float]:
        """Entraîne un modèle Random Forest avec scikit-learn"""
        with mlflow.start_run(run_name=f"sklearn_{run_name}"):
            mlflow.set_tag("framework", "scikit-learn")
            mlflow.set_tag("model_type", "RandomForest")
            
            # Entraînement
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            # Log MLflow
            mlflow.log_params({
                "n_estimators": 100,
                "max_depth": 10
            })
            mlflow.log_metrics({
                "accuracy": accuracy,
                "auc": auc
            })
            
            # Sauvegarde du modèle
            model_path = os.path.join(MODELS_DIR, f"rf_{run_name}.joblib")
            joblib.dump(model, model_path)
            mlflow.log_artifact(model_path)
            
            return {"accuracy": accuracy, "auc": auc}
    
    def train_tensorflow_model(self, X_train, y_train, X_test, y_test, run_name: str) -> Dict[str, float]:
        """Entraîne un modèle avec TensorFlow"""
        tf.keras.backend.clear_session()
        
        with mlflow.start_run(run_name=f"tensorflow_{run_name}"):
            mlflow.set_tag("framework", "tensorflow")
            mlflow.set_tag("model_type", "DNN")
            
            # Construction du modèle
            model = models.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Entraînement
            history = model.fit(
                X_train.astype('float32'),
                y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Évaluation
            y_proba = model.predict(X_test.astype('float32')).ravel()
            y_pred = (y_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            # Log MLflow
            mlflow.log_metrics({
                "accuracy": accuracy,
                "auc": auc,
                "final_loss": history.history['loss'][-1]
            })
            
            # Sauvegarde du modèle
            model_path = os.path.join(MODELS_DIR, f"tf_{run_name}.keras")
            model.save(model_path)
            mlflow.log_artifact(model_path)
            
            return {"accuracy": accuracy, "auc": auc}
    
    def train_pytorch_model(self, X_train, y_train, X_test, y_test, run_name: str) -> Dict[str, float]:
        """Entraîne un modèle avec PyTorch"""
        with mlflow.start_run(run_name=f"pytorch_{run_name}"):
            mlflow.set_tag("framework", "pytorch")
            mlflow.set_tag("model_type", "DNN")
            
            # Préparation des données
            X_train_t = torch.tensor(X_train.astype('float32'))
            y_train_t = torch.tensor(y_train.astype('float32'))
            X_test_t = torch.tensor(X_test.astype('float32'))
            
            # Modèle
            model = SimpleTorchNet(X_train.shape[1], hidden_dim=64)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Entraînement
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            
            # Évaluation
            model.eval()
            with torch.no_grad():
                y_proba = model(X_test_t).numpy()
                y_pred = (y_proba > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            # Log MLflow
            mlflow.log_metrics({
                "accuracy": accuracy,
                "auc": auc,
                "final_loss": loss.item()
            })
            
            # Sauvegarde du modèle
            model_path = os.path.join(MODELS_DIR, f"torch_{run_name}.pt")
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            
            return {"accuracy": accuracy, "auc": auc}
    
    def run_pipeline(self):
        """Exécute le pipeline complet"""
        with LOCK:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.last_run["status"] = "running"
            self.last_run["timestamp"] = timestamp
            
            try:
                config = self.config
                domaine = config["domaine"]
                specialite = config["specialite"]
                n_samples = config.get("n_samples", 1000)
                
                print(f"[PIPELINE] Démarrage - {timestamp}")
                
                # Génération des données
                df = self.generate_synthetic_data(domaine, specialite, n_samples)
                
                # Sauvegarde des données brutes
                raw_path = os.path.join(DATA_DIR, f"raw_{timestamp}.csv")
                df.to_csv(raw_path, index=False)
                
                # Prétraitement
                X, y, metadata, ohe, scaler = self.preprocess_data(df)
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=config.get("test_size", 0.2),
                    random_state=config.get("random_state", 42)
                )
                
                # Entraînement des modèles
                metrics = {}
                
                # Scikit-learn
                try:
                    metrics["sklearn"] = self.train_sklearn_model(
                        X_train, y_train, X_test, y_test, timestamp
                    )
                except Exception as e:
                    print(f"Erreur sklearn: {e}")
                    metrics["sklearn"] = {"error": str(e)}
                
                # TensorFlow
                try:
                    metrics["tensorflow"] = self.train_tensorflow_model(
                        X_train, y_train, X_test, y_test, timestamp
                    )
                except Exception as e:
                    print(f"Erreur tensorflow: {e}")
                    metrics["tensorflow"] = {"error": str(e)}
                
                # PyTorch
                try:
                    metrics["pytorch"] = self.train_pytorch_model(
                        X_train, y_train, X_test, y_test, timestamp
                    )
                except Exception as e:
                    print(f"Erreur pytorch: {e}")
                    metrics["pytorch"] = {"error": str(e)}
                
                # Sauvegarde des métriques
                metrics_file = os.path.join(METRICS_DIR, f"metrics_{timestamp}.json")
                with open(metrics_file, "w") as f:
                    json.dump({
                        "timestamp": timestamp,
                        "config": config,
                        "metadata": metadata,
                        "metrics": metrics
                    }, f, indent=2)
                
                self.last_run["metrics"] = metrics
                self.last_run["status"] = "completed"
                
                print(f"[PIPELINE] Terminé - {timestamp}")
                return {"timestamp": timestamp, "metrics": metrics}
                
            except Exception as e:
                self.last_run["status"] = "error"
                self.last_run["error"] = str(e)
                print(f"[PIPELINE] Erreur: {e}")
                raise


# API FastAPI

app = FastAPI(
    title="AutoSci Pipeline API",
    description="API pour la gestion automatisée de pipelines ML",
    version="1.0.0"
)

pipeline = AutoSciPipeline()
scheduler = BackgroundScheduler()

class ConfigRequest(BaseModel):
    domaine: str
    specialite: str
    freq_minutes: Optional[int] = 60
    n_samples: Optional[int] = 1000
    test_size: Optional[float] = 0.2

@app.on_event("startup")
async def startup_event():
    """Initialise le scheduler au démarrage"""
    config = pipeline.load_config()
    freq_minutes = config.get("freq_minutes", 60)
    
    scheduler.add_job(
        pipeline.run_pipeline,
        'interval',
        minutes=freq_minutes,
        id='pipeline_job',
        replace_existing=True
    )
    scheduler.start()
    print(f"✅ Scheduler démarré - Exécution toutes les {freq_minutes} minutes")

@app.on_event("shutdown")
async def shutdown_event():
    """Arrête le scheduler proprement"""
    scheduler.shutdown()

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "AutoSci Pipeline API",
        "status": "running",
        "endpoints": {
            "config": "/config",
            "run": "/run",
            "status": "/status",
            "metrics": "/metrics",
            "models": "/models"
        }
    }

@app.get("/config")
async def get_config():
    """Récupère la configuration actuelle"""
    return pipeline.config

@app.post("/config")
async def update_config(config: ConfigRequest):
    """Met à jour la configuration"""
    new_config = config.dict()
    pipeline.save_config(new_config)
    
    # Mise à jour du scheduler
    freq_minutes = new_config.get("freq_minutes", 60)
    scheduler.reschedule_job(
        'pipeline_job',
        trigger='interval',
        minutes=freq_minutes
    )
    
    return {
        "message": "Configuration mise à jour",
        "config": new_config
    }

@app.post("/run")
async def run_now(background_tasks: BackgroundTasks):
    """Lance une exécution immédiate du pipeline"""
    background_tasks.add_task(pipeline.run_pipeline)
    return {"message": "Pipeline lancé en arrière-plan"}

@app.get("/status")
async def get_status():
    """Récupère le statut de la dernière exécution"""
    return pipeline.last_run

@app.get("/metrics")
async def get_metrics():
    """Récupère l'historique des métriques"""
    metrics_files = sorted(
        [f for f in os.listdir(METRICS_DIR) if f.endswith('.json')],
        reverse=True
    )
    
    results = []
    for file in metrics_files[:10]:  # Derniers 10 runs
        with open(os.path.join(METRICS_DIR, file), 'r') as f:
            results.append(json.load(f))
    
    return {"count": len(results), "metrics": results}

@app.get("/models")
def get_models():
    """
    Liste tous les modèles enregistrés dans MLflow
    """
    try:
        models = []
        experiments = client.search_experiments()
        
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=20
            )
            for run in runs:
                models.append({
                    "run_id": run.info.run_id,
                    "experiment_id": exp.experiment_id,
                    "tags": run.data.tags,
                    "metrics": run.data.metrics,
                    "status": run.info.status
                })
        
        return {"count": len(models), "models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-mlflow")
def test_mlflow():
    try:
        exps = client.search_experiments()
        return {"experiments": [exp.name for exp in exps]}
    except Exception as e:
        return {"error": str(e)}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/models/{run_id}/download")
async def download_model(run_id: str):
    """Télécharge un modèle MLflow sous forme de ZIP"""
    try:
        # Récupérer le modèle depuis MLflow
        client = mlflow.tracking.MlflowClient()
        
        # Vérifier que le run existe
        try:
            run = client.get_run(run_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Modèle non trouvé")
        
        # Créer un dossier temporaire
        temp_dir = tempfile.mkdtemp()
        model_temp_dir = os.path.join(temp_dir, "model")
        
        try:
            # Télécharger les artifacts du modèle
            artifacts = client.list_artifacts(run_id)
            
            # Créer la structure du modèle exporté
            os.makedirs(model_temp_dir, exist_ok=True)
            
            # Télécharger tous les artifacts
            for artifact in artifacts:
                if artifact.path.startswith('model/'):
                    local_path = client.download_artifacts(run_id, artifact.path, model_temp_dir)
            
            # Ajouter les métadonnées du modèle
            model_info = {
                "run_id": run_id,
                "model_name": run.data.tags.get("mlflow.runName", f"model_{run_id[:8]}"),
                "framework": run.data.tags.get("framework", "unknown"),
                "model_type": run.data.tags.get("model_type", "unknown"),
                "metrics": dict(run.data.metrics),
                "parameters": dict(run.data.params),
                "tags": dict(run.data.tags),
                "created_at": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                "export_date": datetime.now().isoformat()
            }
            
            # Sauvegarder les métadonnées
            with open(os.path.join(model_temp_dir, "model_info.json"), 'w') as f:
                json.dump(model_info, f, indent=2)
            
            # Créer le script d'utilisation
            usage_script = f"""# Script d'utilisation du modèle {run_id[:8]}
import json
import joblib
import pickle
import os

# Charger les métadonnées
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

print(f"Modèle: {{model_info['model_name']}}")
print(f"Framework: {{model_info['framework']}}")
print(f"Type: {{model_info['model_type']}}")

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
    print(f"Erreur lors du chargement: {{e}}")
"""
            
            with open(os.path.join(model_temp_dir, "use_model.py"), 'w') as f:
                f.write(usage_script)
            
            # Créer le README
            readme_content = f"""# Modèle {model_info['model_name']}

## Informations
- **Run ID**: {run_id}
- **Framework**: {model_info['framework']}
- **Type**: {model_info['model_type']}
- **Date d'export**: {model_info['export_date']}

## Métriques
"""
            for metric, value in model_info['metrics'].items():
                readme_content += f"- **{metric}**: {value:.4f}\n"
            
            readme_content += f"""
## Utilisation

1. Extraire le fichier ZIP
2. Installer les dépendances requises
3. Utiliser le script `use_model.py` comme exemple

```python
python use_model.py
```

## Structure des fichiers
- `model_info.json`: Métadonnées du modèle
- `use_model.py`: Script d'exemple d'utilisation
- `model/`: Dossier contenant les fichiers du modèle MLflow
- `README.md`: Ce fichier

## Support
Pour toute question, consultez la documentation AutoSci Pipeline.
"""
            
            with open(os.path.join(model_temp_dir, "README.md"), 'w') as f:
                f.write(readme_content)
            
            # Créer le fichier ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(model_temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_temp_dir)
                        zip_file.write(file_path, arcname)
            
            zip_buffer.seek(0)
            
            # Nettoyer le dossier temporaire
            shutil.rmtree(temp_dir)
            
            # Retourner le fichier ZIP
            return StreamingResponse(
                io.BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f"attachment; filename=model_{run_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                }
            )
            
        except Exception as e:
            # Nettoyer en cas d'erreur
            shutil.rmtree(temp_dir)
            raise HTTPException(status_code=500, detail=f"Erreur lors de la création du package: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{run_id}/info")
async def get_model_detailed_info(run_id: str):
    """Récupère les informations détaillées d'un modèle"""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        return {
            "run_id": run_id,
            "model_name": run.data.tags.get("mlflow.runName", f"model_{run_id[:8]}"),
            "framework": run.data.tags.get("framework", "unknown"),
            "model_type": run.data.tags.get("model_type", "unknown"),
            "status": run.info.status,
            "metrics": dict(run.data.metrics),
            "parameters": dict(run.data.params),
            "tags": dict(run.data.tags),
            "created_at": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
            "updated_at": datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
            "artifacts": [artifact.path for artifact in client.list_artifacts(run_id)]
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")

@app.post("/models/{run_id}/validate")
async def validate_model_download(run_id: str):
    """Valide qu'un modèle peut être téléchargé"""
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Vérifier la présence des artifacts essentiels
        artifacts = client.list_artifacts(run_id)
        model_artifacts = [a for a in artifacts if a.path.startswith('model/')]
        
        if not model_artifacts:
            return {
                "valid": False,
                "reason": "Aucun artifact de modèle trouvé"
            }
        
        # Estimer la taille du téléchargement
        total_size = 0
        for artifact in model_artifacts:
            artifact_path = client.download_artifacts(run_id, artifact.path)
            if os.path.isfile(artifact_path):
                total_size += os.path.getsize(artifact_path)
            elif os.path.isdir(artifact_path):
                for root, dirs, files in os.walk(artifact_path):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))
        
        return {
            "valid": True,
            "estimated_size_mb": round(total_size / (1024 * 1024), 2),
            "artifacts_count": len(model_artifacts),
            "framework": run.data.tags.get("framework", "unknown")
        }
        
    except Exception as e:
        return {
            "valid": False,
            "reason": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)