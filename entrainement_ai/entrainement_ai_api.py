# ai_training_platform.py - API complète pour l'entraînement d'IA
from tkinter import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import librosa
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
import asyncio
import threading
import time
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib
import logging
from dataclasses import dataclass, asdict
import psutil
import GPUtil

# Imports ML/DL  uvicorn ai_training_platform:app --host 0.0.0.0 --port 8008 --reload
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go

# Configuration
MODELS_DIR = "trained_ai_models"
DATA_DIR = "training_datasets"
LOGS_DIR = "training_logs"
CHECKPOINTS_DIR = "model_checkpoints"

# Modèles Pydantic  uvicorn entrainement_ai_api:app --host 0.0.0.0 --port 8025 --reload
class TrainingJobRequest(BaseModel):
    job_name: str
    user_id: str
    model_type: str  # "sklearn", "pytorch", "tensorflow", "xgboost", "lightgbm"
    algorithm: str   # "random_forest", "neural_network", "cnn", "rnn", etc.
    dataset_path: str
    target_column: Optional[str] = None
    task_type: str = "classification"  # "classification", "regression", "clustering"
    hyperparameters: Dict[str, Any] = {}
    training_config: Dict[str, Any] = {}

class ModelDeploymentRequest(BaseModel):
    model_id: str
    user_id: str
    deployment_name: str
    endpoint_config: Dict[str, Any] = {}

class TrainingJobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed", "cancelled"
    progress: float
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    metrics: Dict[str, Any] = {}
    logs: List[str] = []

@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: str = ""

class ConnectionManager:
    """Gestionnaire des connexions WebSocket pour le streaming en temps réel"""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except:
                self.disconnect(client_id)
    
    async def broadcast_to_user(self, message: dict, user_id: str):
        # Envoyer à toutes les connexions d'un utilisateur
        for client_id, websocket in self.active_connections.items():
            if client_id.startswith(user_id):
                await self.send_personal_message(message, client_id)

# Base de données
def init_ai_training_db():
    conn = sqlite3.connect('ai_training_platform.db')
    cursor = conn.cursor()
    
    # Table des jobs d'entraînement
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            job_name TEXT NOT NULL,
            user_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            task_type TEXT NOT NULL,
            dataset_path TEXT,
            target_column TEXT,
            hyperparameters TEXT,
            training_config TEXT,
            status TEXT DEFAULT 'queued',
            progress REAL DEFAULT 0.0,
            created_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            model_path TEXT,
            metrics_history TEXT,
            final_metrics TEXT,
            error_message TEXT
        )
    ''')
    
    # Table des modèles déployés
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deployed_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT UNIQUE NOT NULL,
            job_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            deployment_name TEXT NOT NULL,
            model_path TEXT NOT NULL,
            endpoint_url TEXT,
            status TEXT DEFAULT 'active',
            deployed_at TEXT,
            last_used TEXT,
            usage_count INTEGER DEFAULT 0,
            FOREIGN KEY (job_id) REFERENCES training_jobs (job_id)
        )
    ''')
    
    # Table des datasets disponibles
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS available_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            file_path TEXT NOT NULL,
            size_mb REAL,
            rows_count INTEGER,
            columns_count INTEGER,
            target_columns TEXT,
            dataset_type TEXT,
            created_at TEXT,
            is_public BOOLEAN DEFAULT 1,
            owner_id TEXT
        )
    ''')
    
    # Table des métriques en temps réel
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            epoch INTEGER,
            train_loss REAL,
            val_loss REAL,
            train_accuracy REAL,
            val_accuracy REAL,
            learning_rate REAL,
            timestamp TEXT,
            FOREIGN KEY (job_id) REFERENCES training_jobs (job_id)
        )
    ''')
    
    conn.commit()
    conn.close()

app = FastAPI(title="AI Training Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gestionnaire de connexions WebSocket
manager = ConnectionManager()

# Variables globales pour le suivi des jobs
active_training_jobs = {}
system_monitor = {}

class TrainingEngine:
    """Moteur d'entraînement principal"""
    
    def __init__(self):
        self.current_job = None
        self.training_thread = None
        self.should_stop = False
        
    async def start_training(self, job_request: TrainingJobRequest) -> str:
        """Démarre un job d'entraînement"""
        job_id = str(uuid.uuid4())
        
        # Enregistrer le job en base
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_jobs (
                job_id, job_name, user_id, model_type, algorithm, task_type,
                dataset_path, target_column, hyperparameters, training_config,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id, job_request.job_name, job_request.user_id,
            job_request.model_type, job_request.algorithm, job_request.task_type,
            job_request.dataset_path, job_request.target_column,
            json.dumps(job_request.hyperparameters),
            json.dumps(job_request.training_config),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Ajouter à la queue d'entraînement
        active_training_jobs[job_id] = {
            "request": job_request,
            "status": "queued",
            "progress": 0.0,
            "metrics": [],
            "logs": []
        }
        
        # Démarrer l'entraînement en arrière-plan
        threading.Thread(target=self._train_model, args=(job_id, job_request)).start()
        
        return job_id
    
    def _train_model(self, job_id: str, job_request: TrainingJobRequest):
        """Entraîne le modèle selon la configuration"""
        try:
            # Mettre à jour le statut
            self._update_job_status(job_id, "running", 0.0)
            
            # Charger les données
            self._log_message(job_id, "Chargement des données...")
            data = self._load_dataset(job_request.dataset_path)
            
            # Préparer les données
            self._log_message(job_id, "Préparation des données...")
            X_train, X_val, y_train, y_val = self._prepare_data(data, job_request)
            
            # Sélectionner et entraîner le modèle
            self._log_message(job_id, f"Entraînement du modèle {job_request.algorithm}...")
            
            if job_request.model_type == "sklearn":
                model = self._train_sklearn_model(job_id, X_train, X_val, y_train, y_val, job_request)
            elif job_request.model_type == "pytorch":
                model = self._train_pytorch_model(job_id, X_train, X_val, y_train, y_val, job_request)
            elif job_request.model_type == "tensorflow":
                model = self._train_tensorflow_model(job_id, X_train, X_val, y_train, y_val, job_request)
            elif job_request.model_type == "xgboost":
                model = self._train_xgboost_model(job_id, X_train, X_val, y_train, y_val, job_request)
            else:
                raise ValueError(f"Type de modèle non supporté: {job_request.model_type}")
            
            # Sauvegarder le modèle
            model_path = self._save_model(job_id, model, job_request)
            
            # Finaliser
            self._finalize_training(job_id, model_path, model, X_val, y_val, job_request)
            
        except Exception as e:
            self._handle_training_error(job_id, str(e))
    
    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Charge un dataset"""
        if dataset_path.endswith('.csv'):
            return pd.read_csv(dataset_path)
        elif dataset_path.endswith('.json'):
            return pd.read_json(dataset_path)
        elif dataset_path.endswith('.pkl'):
            return pd.read_pickle(dataset_path)
        else:
            raise ValueError("Format de dataset non supporté")
    
    def _prepare_data(self, data: pd.DataFrame, job_request: TrainingJobRequest):
        """Prépare les données pour l'entraînement"""
        if job_request.target_column:
            X = data.drop(columns=[job_request.target_column])
            y = data[job_request.target_column]
        else:
            # Pour le clustering, pas de target
            X = data
            y = None
        
        # Encoding des variables catégorielles
        for col in X.select_dtypes(include=['object']):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if y is not None:
            # Split train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            return X_train, X_val, y_train, y_val
        else:
            return X_scaled, None, None, None
    
    def _train_sklearn_model(self, job_id: str, X_train, X_val, y_train, y_val, job_request):
        """Entraîne un modèle scikit-learn"""
        algorithm = job_request.algorithm
        hyperparams = job_request.hyperparameters
        
        # Sélectionner l'algorithme
        if algorithm == "random_forest":
            model = RandomForestClassifier(**hyperparams) if job_request.task_type == "classification" else RandomForestClassifier(**hyperparams)
        elif algorithm == "logistic_regression":
            model = LogisticRegression(**hyperparams)
        elif algorithm == "svm":
            model = SVC(**hyperparams) if job_request.task_type == "classification" else SVR(**hyperparams)
        elif algorithm == "neural_network":
            model = MLPClassifier(**hyperparams) if job_request.task_type == "classification" else MLPRegressor(**hyperparams)
        elif algorithm == "gradient_boosting":
            model = GradientBoostingClassifier(**hyperparams)
        else:
            raise ValueError(f"Algorithme sklearn non supporté: {algorithm}")
        
        # Entraînement avec progress tracking
        model.fit(X_train, y_train)
        self._update_job_progress(job_id, 80.0)
        
        # Évaluation
        if X_val is not None and y_val is not None:
            predictions = model.predict(X_val)
            
            if job_request.task_type == "classification":
                accuracy = accuracy_score(y_val, predictions)
                precision = precision_score(y_val, predictions, average='weighted')
                recall = recall_score(y_val, predictions, average='weighted')
                f1 = f1_score(y_val, predictions, average='weighted')
                
                metrics = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1)
                }
            else:
                mse = mean_squared_error(y_val, predictions)
                r2 = r2_score(y_val, predictions)
                
                metrics = {
                    "mse": float(mse),
                    "rmse": float(np.sqrt(mse)),
                    "r2_score": float(r2)
                }
            
            active_training_jobs[job_id]["final_metrics"] = metrics
        
        return model
    
    def _train_pytorch_model(self, job_id: str, X_train, X_val, y_train, y_val, job_request):
        """Entraîne un modèle PyTorch"""
        
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_sizes, output_size, task_type="classification"):
                super(SimpleNN, self).__init__()
                layers = []
                
                prev_size = input_size
                for hidden_size in hidden_sizes:
                    layers.append(nn.Linear(prev_size, hidden_size))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(0.2))
                    prev_size = hidden_size
                
                layers.append(nn.Linear(prev_size, output_size))
                
                if task_type == "classification" and output_size > 1:
                    layers.append(nn.Softmax(dim=1))
                
                self.network = nn.Sequential(*layers)
                
            def forward(self, x):
                return self.network(x)
        
        # Configuration
        hyperparams = job_request.hyperparameters
        epochs = hyperparams.get("epochs", 100)
        batch_size = hyperparams.get("batch_size", 32)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        hidden_sizes = hyperparams.get("hidden_sizes", [128, 64])
        
        # Préparer les données PyTorch
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val) if X_val is not None else None
        
        if job_request.task_type == "classification":
            y_train_tensor = torch.LongTensor(y_train.values)
            y_val_tensor = torch.LongTensor(y_val.values) if y_val is not None else None
            output_size = len(np.unique(y_train))
        else:
            y_train_tensor = torch.FloatTensor(y_train.values)
            y_val_tensor = torch.FloatTensor(y_val.values) if y_val is not None else None
            output_size = 1
        
        # Créer le modèle
        model = SimpleNN(X_train.shape[1], hidden_sizes, output_size, job_request.task_type)
        
        # Loss et optimizer
        if job_request.task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Dataset et DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Entraînement
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if job_request.task_type == "classification":
                    loss = criterion(outputs, batch_y)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == batch_y).sum().item()
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total += batch_y.size(0)
            
            # Validation
            val_loss = 0.0
            val_accuracy = 0.0
            if X_val_tensor is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    if job_request.task_type == "classification":
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_accuracy = (val_predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                    else:
                        val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                model.train()
            
            # Enregistrer les métriques
            train_accuracy = correct / total if job_request.task_type == "classification" else None
            
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=total_loss / len(train_loader),
                val_loss=val_loss if X_val_tensor is not None else None,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy if X_val_tensor is not None else None,
                learning_rate=learning_rate,
                timestamp=datetime.now().isoformat()
            )
            
            self._save_epoch_metrics(job_id, metrics)
            
            # Mettre à jour le progrès
            progress = (epoch + 1) / epochs * 100
            self._update_job_progress(job_id, progress)
            
            # Log périodique
            if (epoch + 1) % 10 == 0:
                self._log_message(job_id, f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        return model
    
    def _train_tensorflow_model(self, job_id: str, X_train, X_val, y_train, y_val, job_request):
        """Entraîne un modèle TensorFlow/Keras"""
        hyperparams = job_request.hyperparameters
        epochs = hyperparams.get("epochs", 100)
        batch_size = hyperparams.get("batch_size", 32)
        learning_rate = hyperparams.get("learning_rate", 0.001)
        
        # Construire le modèle
        model = keras.Sequential()
        
        # Première couche
        model.add(keras.layers.Dense(
            hyperparams.get("first_layer_size", 128),
            activation='relu',
            input_shape=(X_train.shape[1],)
        ))
        model.add(keras.layers.Dropout(0.2))
        
        # Couches cachées
        for hidden_size in hyperparams.get("hidden_layers", [64]):
            model.add(keras.layers.Dense(hidden_size, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
        
        # Couche de sortie
        if job_request.task_type == "classification":
            num_classes = len(np.unique(y_train))
            if num_classes == 2:
                model.add(keras.layers.Dense(1, activation='sigmoid'))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                model.add(keras.layers.Dense(num_classes, activation='softmax'))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            model.add(keras.layers.Dense(1))
            loss = 'mse'
            metrics = ['mae']
        
        # Compiler le modèle
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )
        
        # Callback personnalisé pour le suivi
        class TrainingCallback(keras.callbacks.Callback):
            def __init__(self, job_id):
                self.job_id = job_id
                
            def on_epoch_end(self, epoch, logs=None):
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=logs.get('loss', 0),
                    val_loss=logs.get('val_loss'),
                    train_accuracy=logs.get('accuracy') or logs.get('mae'),
                    val_accuracy=logs.get('val_accuracy') or logs.get('val_mae'),
                    timestamp=datetime.now().isoformat()
                )
                
                training_engine._save_epoch_metrics(self.job_id, metrics)
                
                progress = (epoch + 1) / epochs * 100
                training_engine._update_job_progress(self.job_id, progress)
        
        # Entraînement
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[TrainingCallback(job_id)],
            verbose=0
        )
        
        return model
    
    def _train_xgboost_model(self, job_id: str, X_train, X_val, y_train, y_val, job_request):
        """Entraîne un modèle XGBoost"""
        hyperparams = job_request.hyperparameters
        
        if job_request.task_type == "classification":
            model = xgb.XGBClassifier(**hyperparams)
        else:
            model = xgb.XGBRegressor(**hyperparams)
        
        # Entraînement avec validation
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self._update_job_progress(job_id, 90.0)
        
        return model
    
    def _save_model(self, job_id: str, model, job_request: TrainingJobRequest) -> str:
        """Sauvegarde le modèle entraîné"""
        model_dir = os.path.join(MODELS_DIR, job_id)
        os.makedirs(model_dir, exist_ok=True)
        
        if job_request.model_type == "sklearn" or job_request.model_type == "xgboost":
            model_path = os.path.join(model_dir, "model.pkl")
            joblib.dump(model, model_path)
        elif job_request.model_type == "pytorch":
            model_path = os.path.join(model_dir, "model.pth")
            torch.save(model.state_dict(), model_path)
        elif job_request.model_type == "tensorflow":
            model_path = os.path.join(model_dir, "model.h5")
            model.save(model_path)
        
        return model_path
    
    def _finalize_training(self, job_id: str, model_path: str, model, X_val, y_val, job_request):
        """Finalise l'entraînement"""
        # Calculer les métriques finales si pas déjà fait
        final_metrics = active_training_jobs[job_id].get("final_metrics", {})
        
        # Mettre à jour la base de données
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_jobs 
            SET status = 'completed', progress = 100.0, completed_at = ?, 
                model_path = ?, final_metrics = ?
            WHERE job_id = ?
        ''', (
            datetime.now().isoformat(),
            model_path,
            json.dumps(final_metrics),
            job_id
        ))
        
        conn.commit()
        conn.close()
        
        # Mettre à jour le statut en mémoire
        active_training_jobs[job_id]["status"] = "completed"
        active_training_jobs[job_id]["progress"] = 100.0
        
        self._log_message(job_id, "Entraînement terminé avec succès!")
    
    def _handle_training_error(self, job_id: str, error_message: str):
        """Gère les erreurs d'entraînement"""
        # Mettre à jour la base
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE training_jobs 
            SET status = 'failed', error_message = ?
            WHERE job_id = ?
        ''', (error_message, job_id))
        
        conn.commit()
        conn.close()
        
        # Mettre à jour le statut en mémoire
        if job_id in active_training_jobs:
            active_training_jobs[job_id]["status"] = "failed"
            active_training_jobs[job_id]["error"] = error_message
        
        self._log_message(job_id, f"Erreur: {error_message}")
    
    def _update_job_status(self, job_id: str, status: str, progress: float):
        """Met à jour le statut d'un job"""
        if job_id in active_training_jobs:
            active_training_jobs[job_id]["status"] = status
            active_training_jobs[job_id]["progress"] = progress
    
    def _update_job_progress(self, job_id: str, progress: float):
        """Met à jour le progrès d'un job"""
        if job_id in active_training_jobs:
            active_training_jobs[job_id]["progress"] = progress
    
    def _log_message(self, job_id: str, message: str):
        """Ajoute un message de log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        if job_id in active_training_jobs:
            active_training_jobs[job_id]["logs"].append(log_entry)
    
    def _save_epoch_metrics(self, job_id: str, metrics: TrainingMetrics):
        """Sauvegarde les métriques d'une époque"""
        # Sauver en base de données
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_metrics (
                job_id, epoch, train_loss, val_loss, train_accuracy, 
                val_accuracy, learning_rate, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id, metrics.epoch, metrics.train_loss, metrics.val_loss,
            metrics.train_accuracy, metrics.val_accuracy, 
            metrics.learning_rate, metrics.timestamp
        ))
        
        conn.commit()
        conn.close()
        
        # Ajouter aux métriques en mémoire
        if job_id in active_training_jobs:
            active_training_jobs[job_id]["metrics"].append(asdict(metrics))
    async def _analyze_audio_emotion(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyse d'émotion audio"""
        try:
            # Extraire des features pour l'analyse d'émotion
            # Caractéristiques prosodiques
            pitch = librosa.yin(y, fmin=50, fmax=400)
            pitch_mean = np.nanmean(pitch)
            pitch_std = np.nanstd(pitch)
            
            # Énergie
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = np.mean(rms)
            energy_std = np.std(rms)
            
            # Zero crossing rate (indicateur d'émotion)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = np.mean(zcr)
            
            # Spectral centroid (brillance)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # Classification simple basée sur les features
            # Logique simplifiée d'analyse émotionnelle
            
            # Émotion basée sur l'énergie et le pitch
            if energy_mean > 0.02 and pitch_mean > 150:
                dominant_emotion = "excited"
                confidence = 0.75
            elif energy_mean < 0.01 and pitch_mean < 120:
                dominant_emotion = "sad"
                confidence = 0.70
            elif energy_std > 0.015:  # Variations importantes
                dominant_emotion = "angry"
                confidence = 0.65
            elif zcr_mean < 0.05:  # Voix stable
                dominant_emotion = "calm"
                confidence = 0.80
            else:
                dominant_emotion = "neutral"
                confidence = 0.60
            
            return {
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "features": {
                    "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 0.0,
                    "energy_mean": float(energy_mean),
                    "zcr_mean": float(zcr_mean),
                    "spectral_centroid": float(spectral_centroid_mean)
                }
            }
            
        except Exception as e:
            return {
                "dominant_emotion": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _analyze_music_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyse des caractéristiques musicales"""
        try:
            # Tempo et rythme
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            
            # Tonalité (approximation)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_profile = np.mean(chroma, axis=1)
            dominant_key = np.argmax(key_profile)
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Analyse harmonique
            harmonic, percussive = librosa.effects.hpss(y)
            
            # Ratio harmonique/percussive
            harmonic_strength = np.mean(np.abs(harmonic))
            percussive_strength = np.mean(np.abs(percussive))
            
            # Analyse du genre musical (approximation basique)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            rolloff_mean = np.mean(spectral_rolloff)
            
            # Classification simple de genre
            if tempo > 120 and percussive_strength > harmonic_strength:
                genre_guess = "electronic/dance"
            elif tempo < 80 and harmonic_strength > percussive_strength * 2:
                genre_guess = "classical/ambient"
            elif 90 <= tempo <= 120 and harmonic_strength > percussive_strength:
                genre_guess = "pop/rock"
            else:
                genre_guess = "unknown"
            
            return {
                "tempo": float(tempo),
                "estimated_key": key_names[dominant_key],
                "key_confidence": float(key_profile[dominant_key]),
                "harmonic_strength": float(harmonic_strength),
                "percussive_strength": float(percussive_strength),
                "spectral_rolloff_mean": float(rolloff_mean),
                "genre_estimate": genre_guess,
                "beat_count": len(beat_frames),
                "rhythmic_regularity": float(np.std(np.diff(beat_frames))) if len(beat_frames) > 1 else 0.0
            }
            
        except Exception as e:
            return {"error": str(e)}

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraction d'entités nommées"""
        try:
            # Limiter la taille du texte
            text_sample = text[:512] if len(text) > 512 else text
            
            entities = self.ner_analyzer(text_sample)
            
            processed_entities = []
            for entity in entities:
                processed_entities.append({
                    "text": entity["word"],
                    "label": entity["entity_group"],
                    "confidence": entity["score"],
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0)
                })
            
            return processed_entities
            
        except Exception as e:
            return [{"error": str(e)}]
    
    async def _analyze_semantics(self, text: str) -> Dict[str, Any]:
        """Analyse sémantique du texte"""
        try:
            # Limiter la taille
            text_sample = text[:1024] if len(text) > 1024 else text
            
            # Analyse des thèmes principaux (approche simple)
            words = text_sample.lower().split()
            word_freq = {}
            
            # Mots vides français et anglais
            stop_words = {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
                         'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
            
            for word in words:
                word = word.strip('.,!?;:"()[]')
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Top mots-clés
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Tentative de résumé automatique si le texte est long
            summary = ""
            if len(text) > 200:
                try:
                    summary_result = self.summarizer(text_sample, max_length=50, min_length=10, do_sample=False)
                    summary = summary_result[0]['summary_text']
                except Exception:
                    # Résumé simple: prendre les 2 premières phrases
                    sentences = text.split('.')[:2]
                    summary = '. '.join(sentences) + '.' if sentences else ""
            
            return {
                "top_keywords": [{"word": word, "frequency": freq} for word, freq in top_keywords],
                "word_diversity": len(set(words)) / len(words) if words else 0,
                "summary": summary,
                "dominant_topics": self._extract_topics_simple(text_sample),
                "text_complexity": self._calculate_text_complexity(text)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_topics_simple(self, text: str) -> List[str]:
        """Extraction simple de topics"""
        # Catégories basiques basées sur des mots-clés
        categories = {
            "technology": ["technology", "computer", "software", "data", "algorithm", "AI", "machine", "learning"],
            "business": ["business", "company", "market", "sales", "revenue", "profit", "customer"],
            "science": ["research", "study", "analysis", "experiment", "hypothesis", "theory", "scientific"],
            "health": ["health", "medical", "doctor", "patient", "treatment", "medicine", "hospital"],
            "education": ["education", "school", "student", "teacher", "learn", "knowledge", "study"]
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score >= 2:  # Au moins 2 mots-clés de la catégorie
                detected_topics.append(category)
        
        return detected_topics[:3]  # Top 3
    
    def _calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """Calcule la complexité du texte"""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        if not words or not sentences:
            return {"flesch_reading_ease": 0, "avg_word_length": 0, "avg_sentence_length": 0}
        
        # Approximation du score Flesch
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        flesch_score = max(0, min(100, flesch_score))  # Borner entre 0-100
        
        return {
            "flesch_reading_ease": flesch_score,
            "avg_word_length": sum(len(word) for word in words) / len(words),
            "avg_sentence_length": avg_sentence_length
        }
    
    def _count_syllables(self, word: str) -> int:
        """Compte approximatif des syllabes"""
        word = word.lower().strip('.,!?;:"()')
        if len(word) <= 3:
            return 1
        
        vowels = "aeiouy"
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Règles d'ajustement
        if word.endswith('e'):
            syllables -= 1
        if syllables == 0:
            syllables = 1
            
        return syllables
    
    def _calculate_readability(self, text: str) -> Dict[str, Any]:
        """Calcule les métriques de lisibilité"""
        complexity = self._calculate_text_complexity(text)
        
        flesch_score = complexity["flesch_reading_ease"]
        
        # Interprétation du score Flesch
        if flesch_score >= 90:
            reading_level = "très facile"
        elif flesch_score >= 80:
            reading_level = "facile"
        elif flesch_score >= 70:
            reading_level = "assez facile"
        elif flesch_score >= 60:
            reading_level = "standard"
        elif flesch_score >= 50:
            reading_level = "assez difficile"
        elif flesch_score >= 30:
            reading_level = "difficile"
        else:
            reading_level = "très difficile"
        
        return {
            "flesch_reading_ease": flesch_score,
            "reading_level": reading_level,
            "avg_word_length": complexity["avg_word_length"],
            "avg_sentence_length": complexity["avg_sentence_length"]
        }
    
    async def _extract_topics(self, text: str) -> Dict[str, Any]:
        """Extraction de topics avancée"""
        try:
            # Utiliser l'extraction simple pour l'instant
            simple_topics = self._extract_topics_simple(text)
            
            # Analyse des co-occurrences de mots
            words = [word.lower().strip('.,!?;:"()[]') for word in text.split()]
            words = [w for w in words if len(w) > 3]
            
            # Créer des bigrammes
            bigrams = []
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)
            
            # Compter les bigrammes les plus fréquents
            bigram_freq = {}
            for bigram in bigrams:
                bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
            
            top_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "main_topics": simple_topics,
                "topic_confidence": 0.7,
                "frequent_phrases": [{"phrase": phrase, "count": count} for phrase, count in top_bigrams],
                "topic_distribution": {topic: 1.0/len(simple_topics) for topic in simple_topics} if simple_topics else {}
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_aesthetics(self, image) -> Dict[str, Any]:
        """Analyse esthétique de l'image"""
        try:
            # Convertir en numpy array
            img_array = np.array(image)
            
            # Règle des tiers (approximation)
            height, width = img_array.shape[:2]
            
            # Points d'intérêt selon la règle des tiers
            third_points = [
                (width//3, height//3), (2*width//3, height//3),
                (width//3, 2*height//3), (2*width//3, 2*height//3)
            ]
            
            # Analyse de composition basique
            # Centre de masse des couleurs
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
            
            # Trouver les régions les plus lumineuses
            bright_regions = gray > np.percentile(gray, 80)
            
            # Calculer le centre de masse des régions lumineuses
            y_coords, x_coords = np.where(bright_regions)
            if len(x_coords) > 0:
                center_of_interest = (np.mean(x_coords), np.mean(y_coords))
            else:
                center_of_interest = (width//2, height//2)
            
            # Distance aux points de règle des tiers
            distances_to_thirds = [
                np.sqrt((center_of_interest[0] - p[0])**2 + (center_of_interest[1] - p[1])**2)
                for p in third_points
            ]
            
            # Score de composition (plus proche de la règle des tiers = meilleur)
            min_distance = min(distances_to_thirds)
            max_possible_distance = np.sqrt(width**2 + height**2)
            composition_score = 1.0 - (min_distance / max_possible_distance)
            
            # Analyse de la balance des couleurs
            color_balance = self._analyze_color_balance(img_array)
            
            # Score esthétique global (approximation)
            aesthetic_score = (composition_score * 0.4 + 
                             color_balance * 0.3 + 
                             self._calculate_contrast(image) / 255.0 * 0.3)
            
            return {
                "aesthetic_score": float(aesthetic_score),
                "composition_score": float(composition_score),
                "color_balance": float(color_balance),
                "rule_of_thirds_adherence": float(1.0 - min_distance / max_possible_distance),
                "center_of_interest": [float(center_of_interest[0]), float(center_of_interest[1])],
                "overall_rating": self._get_aesthetic_rating(aesthetic_score)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_color_balance(self, img_array: np.ndarray) -> float:
        """Analyse l'équilibre des couleurs"""
        try:
            if len(img_array.shape) != 3:
                return 0.5  # Image en niveaux de gris
            
            # Moyennes des canaux RGB
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            
            # Écart-type des moyennes (plus faible = plus équilibré)
            rgb_means = [r_mean, g_mean, b_mean]
            std_dev = np.std(rgb_means)
            
            # Normaliser le score (0-1, 1 étant parfaitement équilibré)
            max_std = 255 / np.sqrt(3)  # Maximum théorique
            balance_score = 1.0 - (std_dev / max_std)
            
            return max(0.0, min(1.0, balance_score))
            
        except Exception:
            return 0.5
    
    def _get_aesthetic_rating(self, score: float) -> str:
        """Convertit le score esthétique en rating"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "très bon"
        elif score >= 0.6:
            return "bon"
        elif score >= 0.5:
            return "moyen"
        elif score >= 0.4:
            return "faible"
        else:
            return "très faible"

    def _extract_exif_data(self, image_path: str) -> Dict[str, Any]:
        """Extraction des métadonnées EXIF"""
        try:
            from PIL.ExifTags import TAGS
            
            with Image.open(image_path) as img:
                exifdata = img.getexif()
                
                if not exifdata:
                    return {"exif_available": False}
                
                exif_dict = {}
                for tag_id in exifdata:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)
                    
                    # Convertir en string pour la sérialisation JSON
                    if isinstance(data, bytes):
                        try:
                            data = data.decode()
                        except:
                            data = str(data)
                    
                    exif_dict[tag] = str(data)
                
                # Extraire les informations importantes
                camera_info = {
                    "camera_make": exif_dict.get("Make", "Unknown"),
                    "camera_model": exif_dict.get("Model", "Unknown"),
                    "datetime": exif_dict.get("DateTime", "Unknown"),
                    "orientation": exif_dict.get("Orientation", "Unknown"),
                    "x_resolution": exif_dict.get("XResolution", "Unknown"),
                    "y_resolution": exif_dict.get("YResolution", "Unknown"),
                    "software": exif_dict.get("Software", "Unknown")
                }
                
                return {
                    "exif_available": True,
                    "camera_info": camera_info,
                    "full_exif": exif_dict
                }
                
        except Exception as e:
            return {"exif_available": False, "error": str(e)}

    def _detect_patterns(self, data: Dict) -> Dict[str, Any]:
        """Détection de patterns dans les données"""
        patterns = []
        
        try:
            # Analyser les patterns numériques
            numeric_data = {}
            self._extract_numeric_data(data, numeric_data)
            
            if numeric_data:
                # Détecter les tendances
                for key, values in numeric_data.items():
                    if isinstance(values, list) and len(values) > 2:
                        # Tendance croissante/décroissante
                        if self._is_increasing(values):
                            patterns.append({
                                "type": "increasing_trend",
                                "field": key,
                                "confidence": 0.8,
                                "description": f"Tendance croissante détectée dans {key}"
                            })
                        elif self._is_decreasing(values):
                            patterns.append({
                                "type": "decreasing_trend", 
                                "field": key,
                                "confidence": 0.8,
                                "description": f"Tendance décroissante détectée dans {key}"
                            })
                
                # Détecter les valeurs aberrantes
                outliers = self._detect_outliers_simple(numeric_data)
                if outliers:
                    patterns.append({
                        "type": "outliers",
                        "fields": list(outliers.keys()),
                        "confidence": 0.7,
                        "description": f"Valeurs aberrantes détectées dans {len(outliers)} champs"
                    })
            
            # Pattern de répétition dans les chaînes
            string_patterns = self._detect_string_patterns(data)
            patterns.extend(string_patterns)
            
            return {
                "patterns_found": len(patterns),
                "patterns": patterns,
                "pattern_types": list(set([p["type"] for p in patterns]))
            }
            
        except Exception as e:
            return {"error": str(e), "patterns": []}
    
    def _extract_numeric_data(self, data: Dict, result: Dict, prefix: str = ""):
        """Extrait récursivement les données numériques"""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, (int, float)):
                if full_key not in result:
                    result[full_key] = []
                result[full_key].append(value)
            elif isinstance(value, list):
                numeric_values = [v for v in value if isinstance(v, (int, float))]
                if numeric_values:
                    result[full_key] = numeric_values
            elif isinstance(value, dict):
                self._extract_numeric_data(value, result, full_key)
    
    def _is_increasing(self, values: list) -> bool:
        """Vérifie si une série est croissante"""
        if len(values) < 3:
            return False
        
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        return increases >= len(values) * 0.7  # 70% des valeurs croissantes
    
    def _is_decreasing(self, values: list) -> bool:
        """Vérifie si une série est décroissante"""
        if len(values) < 3:
            return False
        
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        return decreases >= len(values) * 0.7  # 70% des valeurs décroissantes
    
    def _detect_outliers_simple(self, numeric_data: Dict) -> Dict:
        """Détection simple d'outliers"""
        outliers = {}
        
        for key, values in numeric_data.items():
            if len(values) < 4:
                continue
                
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_values = [v for v in values if v < lower_bound or v > upper_bound]
            
            if outlier_values:
                outliers[key] = {
                    "count": len(outlier_values),
                    "percentage": len(outlier_values) / len(values) * 100,
                    "values": outlier_values[:5]  # Limiter à 5 exemples
                }
        
        return outliers
    
    def _detect_string_patterns(self, data: Dict) -> List[Dict[str, Any]]:
        """Détecte des patterns dans les chaînes de caractères"""
        patterns = []
        
        try:
            string_values = []
            self._extract_string_data(data, string_values)
            
            if len(string_values) > 1:
                # Détecter les répétitions
                from collections import Counter
                counter = Counter(string_values)
                
                # Patterns répétitifs
                repeated = {k: v for k, v in counter.items() if v > 1}
                if repeated:
                    patterns.append({
                        "type": "repeated_strings",
                        "confidence": 0.9,
                        "description": f"{len(repeated)} chaînes répétées trouvées",
                        "examples": list(repeated.keys())[:3]
                    })
                
                # Patterns de format (email, URL, etc.)
                import re
                email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
                url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
                
                emails = [s for s in string_values if email_pattern.search(s)]
                urls = [s for s in string_values if url_pattern.search(s)]
                
                if emails:
                    patterns.append({
                        "type": "email_addresses",
                        "confidence": 0.95,
                        "description": f"{len(emails)} adresses email détectées",
                        "count": len(emails)
                    })
                
                if urls:
                    patterns.append({
                        "type": "urls",
                        "confidence": 0.95,
                        "description": f"{len(urls)} URLs détectées",
                        "count": len(urls)
                    })
            
            return patterns
            
        except Exception as e:
            return [{"type": "error", "description": f"Erreur analyse patterns: {str(e)}"}]
    
    def _extract_string_data(self, data: Dict, result: List):
        """Extrait récursivement les chaînes de caractères"""
        for value in data.values():
            if isinstance(value, str) and len(value.strip()) > 0:
                result.append(value.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        result.append(item.strip())
                    elif isinstance(item, dict):
                        self._extract_string_data(item, result)
            elif isinstance(value, dict):
                self._extract_string_data(value, result)

    def _detect_anomalies(self, data: Dict) -> Dict[str, Any]:
        """Détection d'anomalies dans les données"""
        anomalies = []
        
        try:
            # Extraire les données numériques
            numeric_data = {}
            self._extract_numeric_data(data, numeric_data)
            
            # Détecter les anomalies statistiques
            for key, values in numeric_data.items():
                if len(values) < 3:
                    continue
                
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if std_val == 0:  # Toutes les valeurs identiques
                    anomalies.append({
                        "type": "constant_values",
                        "field": key,
                        "severity": "low",
                        "description": f"Toutes les valeurs sont identiques dans {key}",
                        "value": mean_val
                    })
                else:
                    # Z-score pour détecter les valeurs aberrantes
                    z_scores = [(v - mean_val) / std_val for v in values]
                    extreme_values = [(i, v, z) for i, (v, z) in enumerate(zip(values, z_scores)) if abs(z) > 3]
                    
                    if extreme_values:
                        anomalies.append({
                            "type": "statistical_outlier",
                            "field": key,
                            "severity": "medium" if len(extreme_values) > len(values) * 0.1 else "high",
                            "description": f"{len(extreme_values)} valeurs aberrantes dans {key}",
                            "outlier_count": len(extreme_values)
                        })
            
            # Détecter les structures de données anormales
            structure_anomalies = self._detect_structure_anomalies(data)
            anomalies.extend(structure_anomalies)
            
            # Score global d'anomalie
            anomaly_score = min(len(anomalies) * 0.2, 1.0)
            
            return {
                "anomaly_score": anomaly_score,
                "anomalies_detected": len(anomalies),
                "anomalies": anomalies,
                "severity_distribution": {
                    "high": len([a for a in anomalies if a.get("severity") == "high"]),
                    "medium": len([a for a in anomalies if a.get("severity") == "medium"]),
                    "low": len([a for a in anomalies if a.get("severity") == "low"])
                }
            }
            
        except Exception as e:
            return {"error": str(e), "anomalies": []}
    
    def _detect_structure_anomalies(self, data: Dict) -> List[Dict[str, Any]]:
        """Détecte les anomalies de structure"""
        anomalies = []
        
        try:
            # Vérifier la profondeur excessive
            max_depth = self._get_dict_depth(data)
            if max_depth > 10:
                anomalies.append({
                    "type": "excessive_nesting",
                    "severity": "medium",
                    "description": f"Structure trop profonde ({max_depth} niveaux)",
                    "depth": max_depth
                })
            
            # Vérifier les clés vides ou nulles
            empty_keys = self._find_empty_values(data)
            if empty_keys:
                anomalies.append({
                    "type": "empty_values",
                    "severity": "low",
                    "description": f"{len(empty_keys)} champs vides détectés",
                    "empty_fields": empty_keys[:5]  # Limiter les exemples
                })
            
            # Vérifier les types de données incohérents
            type_inconsistencies = self._detect_type_inconsistencies(data)
            if type_inconsistencies:
                anomalies.extend(type_inconsistencies)
            
            return anomalies
            
        except Exception as e:
            return [{"type": "structure_analysis_error", "description": str(e)}]
    
    def _get_dict_depth(self, d: Dict, current_depth: int = 0) -> int:
        """Calcule la profondeur maximale d'un dictionnaire"""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth + 1
        
        return max(self._get_dict_depth(v, current_depth + 1) for v in d.values())
    
    def _find_empty_values(self, data: Dict, prefix: str = "") -> List[str]:
        """Trouve les clés avec des valeurs vides"""
        empty_keys = []
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if value is None or value == "" or value == []:
                empty_keys.append(full_key)
            elif isinstance(value, dict):
                empty_keys.extend(self._find_empty_values(value, full_key))
        
        return empty_keys
    
    def _detect_type_inconsistencies(self, data: Dict) -> List[Dict[str, Any]]:
        """Détecte les incohérences de types"""
        inconsistencies = []
        
        # Analyser les listes pour vérifier la cohérence des types
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 1:
                types_in_list = set(type(item).__name__ for item in value)
                if len(types_in_list) > 1:
                    inconsistencies.append({
                        "type": "mixed_types_in_list",
                        "field": key,
                        "severity": "medium",
                        "description": f"Types mixtes dans la liste {key}: {', '.join(types_in_list)}",
                        "types_found": list(types_in_list)
                    })
            
            elif isinstance(value, dict):
                nested_inconsistencies = self._detect_type_inconsistencies(value)
                inconsistencies.extend(nested_inconsistencies)
        
        return inconsistencies

    def _analyze_correlations(self, data: Dict) -> Dict[str, Any]:
        """Analyse les corrélations entre les données"""
        try:
            # Extraire toutes les données numériques
            numeric_data = {}
            self._extract_numeric_data(data, numeric_data)
            
            correlations = {}
            correlation_matrix = {}
            
            if len(numeric_data) >= 2:
                # Calculer les corrélations entre les champs numériques
                fields = list(numeric_data.keys())
                
                for i, field1 in enumerate(fields):
                    correlation_matrix[field1] = {}
                    for j, field2 in enumerate(fields):
                        if i != j:
                            # S'assurer que les deux champs ont la même longueur
                            values1 = numeric_data[field1]
                            values2 = numeric_data[field2]
                            
                            if isinstance(values1, list) and isinstance(values2, list):
                                min_len = min(len(values1), len(values2))
                                if min_len > 1:
                                    corr_coef = np.corrcoef(values1[:min_len], values2[:min_len])[0, 1]
                                    if not np.isnan(corr_coef):
                                        correlation_matrix[field1][field2] = float(corr_coef)
                                        
                                        # Identifier les corrélations fortes
                                        if abs(corr_coef) > 0.7:
                                            correlations[f"{field1}_vs_{field2}"] = {
                                                "coefficient": float(corr_coef),
                                                "strength": "strong" if abs(corr_coef) > 0.8 else "moderate",
                                                "direction": "positive" if corr_coef > 0 else "negative"
                                            }
            
            # Analyser les patterns de corrélation
            strong_correlations = len([c for c in correlations.values() if c["strength"] == "strong"])
            
            return {
                "correlation_matrix": correlation_matrix,
                "significant_correlations": correlations,
                "strong_correlations_count": strong_correlations,
                "fields_analyzed": len(numeric_data),
                "correlation_summary": self._summarize_correlations(correlations)
            }
            
        except Exception as e:
            return {"error": str(e), "correlations": {}}
    
    def _summarize_correlations(self, correlations: Dict) -> Dict[str, Any]:
        """Résume les corrélations trouvées"""
        if not correlations:
            return {"summary": "Aucune corrélation significative trouvée"}
        
        positive_corr = len([c for c in correlations.values() if c["direction"] == "positive"])
        negative_corr = len([c for c in correlations.values() if c["direction"] == "negative"])
        
        return {
            "total_significant": len(correlations),
            "positive_correlations": positive_corr,
            "negative_correlations": negative_corr,
            "strongest_correlation": max(correlations.values(), key=lambda x: abs(x["coefficient"])) if correlations else None
        }

    def _generate_recommendations(self, data: Dict) -> List[Dict[str, Any]]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []
        
        try:
            # Recommandations basées sur la qualité des données
            numeric_data = {}
            self._extract_numeric_data(data, numeric_data)
            
            # Recommandation de normalisation
            if numeric_data:
                ranges = {}
                for key, values in numeric_data.items():
                    if isinstance(values, list) and len(values) > 1:
                        ranges[key] = max(values) - min(values)
                
                if ranges:
                    max_range = max(ranges.values())
                    min_range = min(ranges.values())
                    
                    if max_range > min_range * 100:  # Différence d'échelle importante
                        recommendations.append({
                            "type": "data_preprocessing",
                            "priority": "high",
                            "title": "Normalisation recommandée",
                            "description": "Les données ont des échelles très différentes. Considérez une normalisation.",
                            "affected_fields": list(ranges.keys())
                        })
            
            # Recommandation de nettoyage des données
            empty_count = len(self._find_empty_values(data))
            if empty_count > 0:
                recommendations.append({
                    "type": "data_quality",
                    "priority": "medium",
                    "title": "Nettoyage des données manquantes",
                    "description": f"{empty_count} champs vides détectés. Considérez l'imputation ou la suppression.",
                    "empty_fields_count": empty_count
                })
            
            # Recommandations d'analyse supplémentaire
            if len(numeric_data) >= 3:
                recommendations.append({
                    "type": "analysis",
                    "priority": "medium",
                    "title": "Analyse de clustering",
                    "description": "Avec plusieurs variables numériques, une analyse de clustering pourrait révéler des patterns.",
                    "suggested_algorithms": ["k-means", "DBSCAN"]
                })
            
            # Recommandations de visualisation
            string_fields = []
            self._extract_string_data(data, string_fields)
            
            if len(set(string_fields)) < len(string_fields) / 2:  # Beaucoup de répétitions
                recommendations.append({
                    "type": "visualization",
                    "priority": "low",
                    "title": "Graphique en secteurs",
                    "description": "Les données catégorielles répétitives se prêtent bien aux graphiques en secteurs."
                })
            
            # Recommandations de modélisation
            if len(numeric_data) >= 2:
                recommendations.append({
                    "type": "modeling",
                    "priority": "high",
                    "title": "Modélisation prédictive",
                    "description": "Les données numériques multiples permettent la création de modèles prédictifs.",
                    "suggested_models": ["régression linéaire", "forêt aléatoire", "réseau de neurones"]
                })
            
            return recommendations
            
        except Exception as e:
            return [{"type": "error", "description": f"Erreur génération recommandations: {str(e)}"}]

    def _create_correlation_heatmap(self, correlations: Dict) -> go.Figure:
        """Crée une heatmap des corrélations"""
        if not correlations.get("correlation_matrix"):
            # Créer un graphique vide
            fig = go.Figure()
            fig.add_annotation(
                text="Aucune donnée de corrélation disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            fig.update_layout(title="Matrice de Corrélation")
            return fig
        
        matrix = correlations["correlation_matrix"]
        fields = list(matrix.keys())
        
        # Construire la matrice de corrélation
        correlation_values = []
        for field1 in fields:
            row = []
            for field2 in fields:
                if field1 == field2:
                    row.append(1.0)
                else:
                    value = matrix[field1].get(field2, 0)
                    row.append(value)
            correlation_values.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_values,
            x=fields,
            y=fields,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Corrélation")
        ))
        
        fig.update_layout(
            title="Matrice de Corrélation",
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
# Instance globale du moteur d'entraînement
training_engine = TrainingEngine()

# Système de monitoring des ressources
class SystemMonitor:
    def __init__(self):
        self.start_monitoring()
    
    def start_monitoring(self):
        """Démarre le monitoring des ressources système"""
        threading.Thread(target=self._monitor_resources, daemon=True).start()
    
    def _monitor_resources(self):
        """Surveille les ressources système en continu"""
        while True:
            try:
                # CPU et RAM
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU si disponible
                gpu_info = []
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_info.append({
                            "id": gpu.id,
                            "name": gpu.name,
                            "load": gpu.load * 100,
                            "memory_used": gpu.memoryUsed,
                            "memory_total": gpu.memoryTotal,
                            "temperature": gpu.temperature
                        })
                except:
                    pass
                
                system_monitor.update({
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "gpu_info": gpu_info
                })
                
                time.sleep(5)  # Mise à jour toutes les 5 secondes
                
            except Exception as e:
                print(f"Erreur monitoring: {e}")
                time.sleep(10)

monitor = SystemMonitor()

# Endpoints API
@app.on_event("startup")
async def startup_event():
    # Créer les dossiers nécessaires
    for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialiser la base de données
    init_ai_training_db()
    
    # Créer des datasets par défaut
    await create_default_datasets()

async def create_default_datasets():
    """Crée des datasets par défaut pour tester la plateforme"""
    try:
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        # Vérifier si des datasets existent déjà
        cursor.execute('SELECT COUNT(*) FROM available_datasets')
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("Création des datasets par défaut...")
            
            # Dataset 1: Iris (classification)
            iris_data = {
                'sepal_length': np.random.normal(5.8, 0.8, 150),
                'sepal_width': np.random.normal(3.0, 0.4, 150), 
                'petal_length': np.random.normal(3.8, 1.8, 150),
                'petal_width': np.random.normal(1.2, 0.8, 150),
                'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
            }
            iris_df = pd.DataFrame(iris_data)
            iris_path = os.path.join(DATA_DIR, 'iris_dataset.csv')
            iris_df.to_csv(iris_path, index=False)
            
            iris_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO available_datasets (
                    dataset_id, name, description, file_path, size_mb,
                    rows_count, columns_count, target_columns, dataset_type,
                    created_at, is_public, owner_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                iris_id, "Iris Dataset", 
                "Données de classification des fleurs Iris - Dataset classique de machine learning",
                iris_path, 0.01, 150, 5, 
                json.dumps(["species"]), "tabular",
                datetime.now().isoformat(), True, "system"
            ))
            print(f"✅ Dataset Iris créé: {iris_path}")
            
            # Dataset 2: House Prices (regression)
            house_data = {
                'size': np.random.normal(2000, 500, 1000),
                'bedrooms': np.random.randint(1, 6, 1000),
                'bathrooms': np.random.randint(1, 4, 1000),
                'age': np.random.randint(0, 50, 1000),
                'location_score': np.random.uniform(1, 10, 1000),
                'price': np.random.normal(300000, 100000, 1000)
            }
            house_df = pd.DataFrame(house_data)
            house_df['price'] = house_df['price'].abs()  # Assurer des prix positifs
            house_path = os.path.join(DATA_DIR, 'house_prices.csv')
            house_df.to_csv(house_path, index=False)
            
            house_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO available_datasets (
                    dataset_id, name, description, file_path, size_mb,
                    rows_count, columns_count, target_columns, dataset_type,
                    created_at, is_public, owner_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                house_id, "House Prices",
                "Données de prix immobilier pour régression - Features: taille, chambres, âge, etc.",
                house_path, 0.05, 1000, 6,
                json.dumps(["price"]), "tabular", 
                datetime.now().isoformat(), True, "system"
            ))
            print(f"✅ Dataset House Prices créé: {house_path}")
            
            # Dataset 3: Customer Segmentation (clustering)
            customer_data = {
                'annual_spending': np.random.normal(50000, 15000, 500),
                'frequency_visits': np.random.poisson(20, 500),
                'avg_purchase': np.random.normal(150, 50, 500),
                'loyalty_score': np.random.uniform(0, 10, 500),
                'age': np.random.randint(18, 70, 500)
            }
            customer_df = pd.DataFrame(customer_data)
            customer_path = os.path.join(DATA_DIR, 'customer_segmentation.csv')
            customer_df.to_csv(customer_path, index=False)
            
            customer_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO available_datasets (
                    dataset_id, name, description, file_path, size_mb,
                    rows_count, columns_count, target_columns, dataset_type,
                    created_at, is_public, owner_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                customer_id, "Customer Segmentation",
                "Données clients pour clustering - Analyse comportementale et segmentation",
                customer_path, 0.02, 500, 5,
                json.dumps([]), "tabular",
                datetime.now().isoformat(), True, "system"
            ))
            print(f"✅ Dataset Customer Segmentation créé: {customer_path}")
        
        conn.commit()
        conn.close()
        print(f"✅ Initialisation datasets terminée. Total: {count + (3 if count == 0 else 0)}")
        
    except Exception as e:
        print(f"❌ Erreur création datasets par défaut: {e}")
        import traceback
        traceback.print_exc()
        # Ne pas faire échouer le démarrage pour cette erreur
        pass

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket pour streaming en temps réel"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Traiter les messages entrants si nécessaire
            message = json.loads(data)
            
            if message.get("type") == "subscribe_job":
                job_id = message.get("job_id")
                # L'utilisateur s'abonne aux mises à jour d'un job
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "job_id": job_id
                }))
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/training/start")
async def start_training_job(job_request: TrainingJobRequest, background_tasks: BackgroundTasks):
    """Démarre un nouvel entraînement"""
    try:
        # Valider la demande
        if not os.path.exists(job_request.dataset_path):
            raise HTTPException(status_code=400, detail="Dataset non trouvé")
        
        # Démarrer l'entraînement
        job_id = await training_engine.start_training(job_request)
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Entraînement démarré",
            "estimated_duration": "Variable selon le modèle et les données"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/jobs/{user_id}")
async def get_user_training_jobs(user_id: str):
    """Récupère tous les jobs d'un utilisateur"""
    try:
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT job_id, job_name, model_type, algorithm, task_type, status,
                   progress, created_at, started_at, completed_at, final_metrics
            FROM training_jobs 
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        jobs = []
        for row in cursor.fetchall():
            final_metrics = json.loads(row[10]) if row[10] else {}
            jobs.append({
                "job_id": row[0],
                "job_name": row[1],
                "model_type": row[2],
                "algorithm": row[3],
                "task_type": row[4],
                "status": row[5],
                "progress": row[6],
                "created_at": row[7],
                "started_at": row[8],
                "completed_at": row[9],
                "final_metrics": final_metrics
            })
        
        conn.close()
        return {"jobs": jobs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Récupère le statut détaillé d'un job"""
    try:
        # Vérifier d'abord en mémoire (jobs actifs)
        if job_id in active_training_jobs:
            job_data = active_training_jobs[job_id]
            return TrainingJobStatus(
                job_id=job_id,
                status=job_data["status"],
                progress=job_data["progress"],
                metrics=job_data.get("metrics", {}),
                logs=job_data.get("logs", [])
            )
        
        # Sinon, récupérer depuis la base de données
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT status, progress, final_metrics, error_message
            FROM training_jobs WHERE job_id = ?
        ''', (job_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Job non trouvé")
        
        return TrainingJobStatus(
            job_id=job_id,
            status=result[0],
            progress=result[1],
            metrics=json.loads(result[2]) if result[2] else {},
            logs=[result[3]] if result[3] else []
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/job/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Récupère l'historique des métriques d'un job"""
    try:
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT epoch, train_loss, val_loss, train_accuracy, val_accuracy,
                   learning_rate, timestamp
            FROM training_metrics 
            WHERE job_id = ?
            ORDER BY epoch
        ''', (job_id,))
        
        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                "epoch": row[0],
                "train_loss": row[1],
                "val_loss": row[2],
                "train_accuracy": row[3],
                "val_accuracy": row[4],
                "learning_rate": row[5],
                "timestamp": row[6]
            })
        
        conn.close()
        return {"metrics": metrics}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/job/{job_id}/stop")
async def stop_training_job(job_id: str, user_id: str):
    """Arrête un job d'entraînement"""
    try:
        if job_id in active_training_jobs:
            active_training_jobs[job_id]["status"] = "cancelled"
            training_engine.should_stop = True
            
            # Mettre à jour la base
            conn = sqlite3.connect('ai_training_platform.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE training_jobs SET status = 'cancelled'
                WHERE job_id = ? AND user_id = ?
            ''', (job_id, user_id))
            
            conn.commit()
            conn.close()
            
            return {"message": "Job arrêté"}
        else:
            raise HTTPException(status_code=404, detail="Job non trouvé ou déjà terminé")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/resources")
async def get_system_resources():
    """Récupère l'état des ressources système"""
    return system_monitor

@app.get("/datasets/available")
async def get_available_datasets():
    """Récupère la liste des datasets disponibles"""
    try:
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT dataset_id, name, description, file_path, size_mb, rows_count,
                   columns_count, target_columns, dataset_type, created_at, is_public
            FROM available_datasets
            WHERE is_public = 1
            ORDER BY created_at DESC
        ''')
        
        datasets = []
        for row in cursor.fetchall():
            datasets.append({
                "dataset_id": row[0],
                "name": row[1],
                "description": row[2],
                "file_path": row[3],  # S'assurer que file_path est inclus
                "size_mb": row[4],
                "rows_count": row[5],
                "columns_count": row[6],
                "target_columns": json.loads(row[7]) if row[7] else [],
                "dataset_type": row[8],
                "created_at": row[9],
                "is_public": bool(row[10])
            })
        
        conn.close()
        return {"datasets": datasets}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/datasets/upload")
async def upload_dataset(
    dataset_name: str,
    description: str,
    user_id: str,
    file_path: str,
    is_public: bool = True
):
    """Ajoute un nouveau dataset à la plateforme"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail="Fichier non trouvé")
        
        # Analyser le dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise HTTPException(status_code=400, detail="Format non supporté")
        
        dataset_id = str(uuid.uuid4())
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        # Détecter les colonnes cibles potentielles
        target_candidates = []
        for col in df.columns:
            # Colonnes avec peu de valeurs uniques peuvent être des targets
            if df[col].nunique() < len(df) * 0.1 and df[col].nunique() > 1:
                target_candidates.append(col)
        
        # Enregistrer en base
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO available_datasets (
                dataset_id, name, description, file_path, size_mb,
                rows_count, columns_count, target_columns, dataset_type,
                created_at, is_public, owner_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id, dataset_name, description, file_path, file_size,
            len(df), len(df.columns), json.dumps(target_candidates),
            "tabular", datetime.now().isoformat(), is_public, user_id
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "dataset_id": dataset_id,
            "message": "Dataset ajouté avec succès",
            "rows": len(df),
            "columns": len(df.columns),
            "potential_targets": target_candidates
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/deploy")
async def deploy_model(deployment_request: ModelDeploymentRequest):
    """Déploie un modèle entraîné"""
    try:
        # Vérifier que le modèle existe
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_path FROM training_jobs 
            WHERE job_id = ? AND user_id = ? AND status = 'completed'
        ''', (deployment_request.model_id, deployment_request.user_id))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Modèle non trouvé ou non terminé")
        
        model_path = result[0]
        
        # Créer un endpoint unique
        endpoint_url = f"/api/predict/{deployment_request.model_id}"
        deployment_id = str(uuid.uuid4())
        
        # Enregistrer le déploiement
        cursor.execute('''
            INSERT INTO deployed_models (
                model_id, job_id, user_id, deployment_name,
                model_path, endpoint_url, deployed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            deployment_id, deployment_request.model_id,
            deployment_request.user_id, deployment_request.deployment_name,
            model_path, endpoint_url, datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "deployment_id": deployment_id,
            "endpoint_url": endpoint_url,
            "status": "deployed",
            "message": "Modèle déployé avec succès"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/deployed/{user_id}")
async def get_deployed_models(user_id: str):
    """Récupère les modèles déployés d'un utilisateur"""
    try:
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT dm.model_id, dm.deployment_name, dm.endpoint_url,
                   dm.deployed_at, dm.usage_count, tj.job_name, tj.algorithm
            FROM deployed_models dm
            JOIN training_jobs tj ON dm.job_id = tj.job_id
            WHERE dm.user_id = ? AND dm.status = 'active'
            ORDER BY dm.deployed_at DESC
        ''', (user_id,))
        
        models = []
        for row in cursor.fetchall():
            models.append({
                "model_id": row[0],
                "deployment_name": row[1],
                "endpoint_url": row[2],
                "deployed_at": row[3],
                "usage_count": row[4],
                "original_job_name": row[5],
                "algorithm": row[6]
            })
        
        conn.close()
        return {"deployed_models": models}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/{model_id}")
async def predict_with_model(model_id: str, data: Dict[str, Any]):
    """Utilise un modèle déployé pour faire des prédictions"""
    try:
        # Récupérer le modèle
        conn = sqlite3.connect('ai_training_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT dm.model_path, tj.model_type, tj.algorithm
            FROM deployed_models dm
            JOIN training_jobs tj ON dm.job_id = tj.job_id
            WHERE dm.model_id = ? AND dm.status = 'active'
        ''', (model_id,))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Modèle déployé non trouvé")
        
        model_path, model_type, algorithm = result
        
        # Charger le modèle
        if model_type == "sklearn" or model_type == "xgboost":
            model = joblib.load(model_path)
        elif model_type == "tensorflow":
            model = keras.models.load_model(model_path)
        else:
            raise HTTPException(status_code=400, detail="Type de modèle non supporté pour prédiction")
        
        # Préparer les données d'entrée
        input_data = np.array(data["features"]).reshape(1, -1)
        
        # Faire la prédiction
        if model_type == "sklearn" or model_type == "xgboost":
            prediction = model.predict(input_data)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)
            else:
                probabilities = None
        elif model_type == "tensorflow":
            prediction = model.predict(input_data)
            probabilities = prediction if len(prediction[0]) > 1 else None
        
        # Mettre à jour le compteur d'utilisation
        cursor.execute('''
            UPDATE deployed_models 
            SET usage_count = usage_count + 1, last_used = ?
            WHERE model_id = ?
        ''', (datetime.now().isoformat(), model_id))
        
        conn.commit()
        conn.close()
        
        response = {
            "model_id": model_id,
            "prediction": prediction.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        if probabilities is not None:
            response["probabilities"] = probabilities.tolist()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training/algorithms")
async def get_available_algorithms():
    """Récupère la liste des algorithmes disponibles"""
    algorithms = {
        "sklearn": {
            "classification": [
                {"name": "random_forest", "display_name": "Random Forest", "hyperparams": ["n_estimators", "max_depth", "min_samples_split"]},
                {"name": "logistic_regression", "display_name": "Régression Logistique", "hyperparams": ["C", "penalty", "solver"]},
                {"name": "svm", "display_name": "SVM", "hyperparams": ["C", "kernel", "gamma"]},
                {"name": "neural_network", "display_name": "Réseau de Neurones", "hyperparams": ["hidden_layer_sizes", "learning_rate", "max_iter"]},
                {"name": "gradient_boosting", "display_name": "Gradient Boosting", "hyperparams": ["n_estimators", "learning_rate", "max_depth"]}
            ],
            "regression": [
                {"name": "linear_regression", "display_name": "Régression Linéaire", "hyperparams": []},
                {"name": "random_forest", "display_name": "Random Forest", "hyperparams": ["n_estimators", "max_depth"]},
                {"name": "svm", "display_name": "SVR", "hyperparams": ["C", "kernel", "gamma"]}
            ]
        },
        "pytorch": {
            "classification": [
                {"name": "neural_network", "display_name": "Réseau de Neurones", "hyperparams": ["hidden_sizes", "epochs", "batch_size", "learning_rate"]}
            ],
            "regression": [
                {"name": "neural_network", "display_name": "Réseau de Neurones", "hyperparams": ["hidden_sizes", "epochs", "batch_size", "learning_rate"]}
            ]
        },
        "tensorflow": {
            "classification": [
                {"name": "neural_network", "display_name": "Réseau de Neurones Dense", "hyperparams": ["hidden_layers", "epochs", "batch_size", "learning_rate"]},
                {"name": "cnn", "display_name": "Réseau Convolutionnel", "hyperparams": ["conv_layers", "dense_layers", "epochs", "batch_size"]}
            ],
            "regression": [
                {"name": "neural_network", "display_name": "Réseau de Neurones Dense", "hyperparams": ["hidden_layers", "epochs", "batch_size", "learning_rate"]}
            ]
        },
        "xgboost": {
            "classification": [
                {"name": "xgboost", "display_name": "XGBoost Classifier", "hyperparams": ["n_estimators", "max_depth", "learning_rate", "subsample"]}
            ],
            "regression": [
                {"name": "xgboost", "display_name": "XGBoost Regressor", "hyperparams": ["n_estimators", "max_depth", "learning_rate", "subsample"]}
            ]
        }
    }
    
    return algorithms

@app.get("/health")
async def health_check():
    """Vérification de santé du service"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(active_training_jobs),
        "system_load": system_monitor.get("cpu_percent", 0)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8025)
