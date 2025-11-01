# personal_data_platform.py - API de la plateforme de données personnelles

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import uuid
from contextlib import asynccontextmanager
import asyncio
import psutil
import socket
import subprocess
import platform
from pathlib import Path
import pickle
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from data_platform_app import sanitize_floats

# Modèles Pydantic  uvicorn data_platform_api:app --reload --host 0.0.0.0 --port 8022
class DataCollectionConfig(BaseModel):
    user_id: str
    collection_types: List[str] = Field(default=["network", "files", "system"])
    consent_timestamp: str
    duration_hours: int = 24
    privacy_level: str = "high"  # low, medium, high
    include_sensitive: bool = False

class AnalysisRequest(BaseModel):
    file_id: str
    analysis_type: str  # "descriptive", "exploratory", "statistical"
    user_id: str

class StudyRequest(BaseModel):
    file_id: str
    target_variable: Optional[str] = None
    problem_type: str = "auto"  # auto, classification, regression
    user_id: str

class DataSaleOffer(BaseModel):
    file_id: str
    price: float
    description: str
    anonymization_level: str = "high"
    license_type: str = "single_use"
    user_id: str

class DataDonation(BaseModel):
    file_id: str
    recipient_organization: str
    purpose: str
    anonymization_level: str = "high"
    user_id: str

class ConsentRecord(BaseModel):
    user_id: str
    data_type: str
    consent_given: bool
    timestamp: str
    expires_at: Optional[str] = None

# Configuration de la base de données
def init_personal_data_db():
    conn = sqlite3.connect('personal_data_platform.db')
    cursor = conn.cursor()
    
    # Table des configurations de collecte
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            collection_id TEXT UNIQUE NOT NULL,
            config TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            completed_at TEXT
        )
    ''')
    
    # Table des fichiers collectés
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS collected_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            file_name TEXT,
            file_path TEXT,
            data_type TEXT,
            source_type TEXT,
            size_bytes INTEGER,
            created_at TEXT,
            processed_at TEXT,
            metadata TEXT
        )
    ''')
    
    # Table des analyses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            analysis_type TEXT,
            results TEXT,
            visualizations TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    # Table des études data science
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            problem_type TEXT,
            model_performance TEXT,
            cleaned_data_path TEXT,
            model_path TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    # Table des offres de vente
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sale_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            seller_id TEXT NOT NULL,
            price REAL,
            description TEXT,
            anonymized_file_path TEXT,
            status TEXT DEFAULT 'available',
            created_at TEXT,
            sold_at TEXT,
            buyer_id TEXT
        )
    ''')
    
    # Table des dons
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            donation_id TEXT UNIQUE NOT NULL,
            file_id TEXT NOT NULL,
            donor_id TEXT NOT NULL,
            recipient_organization TEXT,
            purpose TEXT,
            anonymized_file_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            completed_at TEXT
        )
    ''')
    
    # Table des consentements
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_consents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            data_type TEXT NOT NULL,
            consent_given BOOLEAN,
            timestamp TEXT,
            expires_at TEXT,
            revoked_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_personal_data_db()
    yield
    # Shutdown
    pass

app = FastAPI(
    title="Personal Data Intelligence Platform",
    description="Plateforme de collecte et d'exploitation éthique de données personnelles",
    version="1.0.0",
    lifespan=lifespan
)

# Collecteur de données système
class PersonalDataCollector:
    def __init__(self, user_id: str, config: DataCollectionConfig):
        self.user_id = user_id
        self.config = config
        self.collected_data = {"network": [], "files": [], "system": []}
    
    async def collect_network_data(self):
        """Collecte des données réseau avec consentement"""
        if "network" not in self.config.collection_types:
            return
        
        try:
            # Connexions réseau actives
            connections = psutil.net_connections()
            network_data = []
            
            for conn in connections:
                if conn.status == 'ESTABLISHED':
                    network_data.append({
                        "local_address": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                        "remote_address": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                        "status": conn.status,
                        "protocol": "TCP" if conn.type == socket.SOCK_STREAM else "UDP",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Statistiques réseau
            net_stats = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": net_stats.bytes_sent,
                "bytes_recv": net_stats.bytes_recv,
                "packets_sent": net_stats.packets_sent,
                "packets_recv": net_stats.packets_recv,
                "timestamp": datetime.now().isoformat()
            }
            
            self.collected_data["network"] = {
                "connections": network_data[:50],  # Limiter pour la vie privée
                "statistics": network_stats
            }
            
        except Exception as e:
            print(f"Erreur collecte réseau: {e}")
    
    async def collect_system_data(self):
        """Collecte des données système avec consentement"""
        if "system" not in self.config.collection_types:
            return
        
        try:
            # Informations système générales (non sensibles)
            system_info = {
                "platform": platform.system(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor() if not self.config.include_sensitive else "masked",
                "python_version": platform.python_version(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Utilisation des ressources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resource_usage = {
                "cpu_percent": cpu_percent,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_percent": memory.percent,
                "disk_total": disk.total,
                "disk_used": disk.used,
                "disk_free": disk.free,
                "timestamp": datetime.now().isoformat()
            }
            
            self.collected_data["system"] = {
                "info": system_info,
                "resources": resource_usage
            }
            
        except Exception as e:
            print(f"Erreur collecte système: {e}")
    
    async def collect_file_metadata(self):
        """Collecte des métadonnées de fichiers (pas le contenu)"""
        if "files" not in self.config.collection_types:
            return
        
        try:
            # Seulement les métadonnées des fichiers autorisés
            allowed_extensions = ['.csv', '.json', '.txt', '.log']
            home_dir = Path.home()
            
            file_metadata = []
            
            # Parcourir uniquement les dossiers publics/autorisés
            public_dirs = [
                home_dir / "Documents",
                home_dir / "Downloads",
                home_dir / "Desktop"
            ]
            
            for directory in public_dirs:
                if directory.exists():
                    for file_path in directory.rglob("*"):
                        if (file_path.is_file() and 
                            file_path.suffix.lower() in allowed_extensions and
                            file_path.stat().st_size < 10 * 1024 * 1024):  # Max 10MB
                            
                            try:
                                stat = file_path.stat()
                                file_metadata.append({
                                    "name": file_path.name,
                                    "extension": file_path.suffix,
                                    "size_bytes": stat.st_size,
                                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                    "directory": str(file_path.parent.name)  # Pas le chemin complet
                                })
                                
                                if len(file_metadata) >= 100:  # Limiter le nombre
                                    break
                            except Exception:
                                continue
            
            self.collected_data["files"] = file_metadata
            
        except Exception as e:
            print(f"Erreur collecte fichiers: {e}")
    
    async def save_collected_data(self):
        """Sauvegarde les données collectées"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for data_type, data in self.collected_data.items():
            if data:
                file_id = str(uuid.uuid4())
                file_name = f"{data_type}_data_{timestamp}.json"
                file_path = f"collected_data/{self.user_id}/{file_name}"
                
                # Créer le dossier si nécessaire
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Sauvegarder les données
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Enregistrer en base
                conn = sqlite3.connect('personal_data_platform.db')
                cursor = conn.cursor()
                
                metadata = {
                    "collection_config": self.config.dict(),
                    "record_count": len(data) if isinstance(data, list) else len(str(data)),
                    "data_types": list(data.keys()) if isinstance(data, dict) else [type(data).__name__]
                }
                
                cursor.execute('''
                    INSERT INTO collected_files 
                    (file_id, user_id, file_name, file_path, data_type, source_type, size_bytes, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    file_id,
                    self.user_id,
                    file_name,
                    file_path,
                    data_type,
                    "system_collection",
                    os.path.getsize(file_path),
                    datetime.now().isoformat(),
                    json.dumps(metadata)
                ))
                
                conn.commit()
                conn.close()

# Analyseur de données
class DataAnalyzer:
    def __init__(self, file_path: str, analysis_type: str):
        self.file_path = file_path
        self.analysis_type = analysis_type
        self.results = {}
        self.visualizations = {}
    
    def load_data(self):
        """Charge les données depuis le fichier"""
        try:
            if self.file_path.endswith('.json'):
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            elif self.file_path.endswith('.csv'):
                return pd.read_csv(self.file_path)
            else:
                raise ValueError("Format de fichier non supporté")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur lecture fichier: {e}")
    
    def descriptive_analysis(self, data):
        """Analyse descriptive des données"""
        if isinstance(data, dict):
            # Pour les données JSON (système, réseau)
            results = {
                "data_structure": self._analyze_dict_structure(data),
                "summary": {
                    "total_keys": len(data.keys()) if isinstance(data, dict) else 0,
                    "data_types": self._get_data_types(data),
                    "size_estimation": len(str(data))
                }
            }
            
            # Créer des visualisations
            if "connections" in data:
                self._create_network_visualizations(data)
            elif "resources" in data:
                self._create_system_visualizations(data)
                
        elif isinstance(data, pd.DataFrame):
            # Pour les données CSV
            results = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.astype(str).to_dict(),
                "null_counts": data.isnull().sum().to_dict(),
                "summary_stats": data.describe().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 0 else {},
                "unique_counts": data.nunique().to_dict()
            }
            
            self._create_dataframe_visualizations(data)
        
        return results
    
    def _analyze_dict_structure(self, data, max_depth=3, current_depth=0):
        """Analyse la structure d'un dictionnaire"""
        if current_depth > max_depth:
            return "max_depth_reached"
        
        structure = {}
        for key, value in data.items():
            if isinstance(value, dict):
                structure[key] = self._analyze_dict_structure(value, max_depth, current_depth + 1)
            elif isinstance(value, list):
                structure[key] = f"list[{len(value)} items]"
            else:
                structure[key] = type(value).__name__
        
        return structure
    
    def _get_data_types(self, data):
        """Identifie les types de données"""
        types = set()
        
        def extract_types(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    extract_types(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_types(item)
            else:
                types.add(type(obj).__name__)
        
        extract_types(data)
        return list(types)
    
    def _create_network_visualizations(self, data):
        """Crée des visualisations pour les données réseau"""
        try:
            if "connections" in data and data["connections"]:
                # Graphique des connexions par protocole
                protocols = {}
                for conn in data["connections"]:
                    protocol = conn.get("protocol", "Unknown")
                    protocols[protocol] = protocols.get(protocol, 0) + 1
                
                fig = px.pie(
                    values=list(protocols.values()),
                    names=list(protocols.keys()),
                    title="Répartition des connexions par protocole"
                )
                
                self.visualizations["protocol_distribution"] = fig.to_json()
            
            if "statistics" in data:
                # Graphique des statistiques réseau
                stats = data["statistics"]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Bytes Sent", "Bytes Received", "Packets Sent", "Packets Received"],
                    y=[stats.get("bytes_sent", 0), stats.get("bytes_recv", 0),
                       stats.get("packets_sent", 0), stats.get("packets_recv", 0)],
                    name="Network Statistics"
                ))
                fig.update_layout(title="Statistiques réseau globales")
                
                self.visualizations["network_stats"] = fig.to_json()
                
        except Exception as e:
            print(f"Erreur création visualisations réseau: {e}")
    
    def _create_system_visualizations(self, data):
        """Crée des visualisations pour les données système"""
        try:
            if "resources" in data:
                resources = data["resources"]
                
                # Graphique d'utilisation des ressources
                fig = go.Figure()
                
                # CPU et Mémoire en pourcentage
                fig.add_trace(go.Bar(
                    x=["CPU Usage", "Memory Usage"],
                    y=[resources.get("cpu_percent", 0), resources.get("memory_percent", 0)],
                    name="Usage (%)",
                    marker_color=["red", "blue"]
                ))
                
                fig.update_layout(
                    title="Utilisation des ressources système",
                    yaxis_title="Pourcentage (%)"
                )
                
                self.visualizations["resource_usage"] = fig.to_json()
                
                # Graphique d'utilisation disque
                total_disk = resources.get("disk_total", 1)
                used_disk = resources.get("disk_used", 0)
                free_disk = resources.get("disk_free", 0)
                
                fig_disk = go.Figure(data=[go.Pie(
                    labels=["Utilisé", "Libre"],
                    values=[used_disk, free_disk],
                    title="Utilisation du disque"
                )])
                
                self.visualizations["disk_usage"] = fig_disk.to_json()
                
        except Exception as e:
            print(f"Erreur création visualisations système: {e}")
    
    def _create_dataframe_visualizations(self, data):
        """Crée des visualisations pour les DataFrames"""
        try:
            # Distribution des types de données
            dtype_counts = data.dtypes.value_counts()
            
            fig_dtypes = px.bar(
                x=dtype_counts.index.astype(str),
                y=dtype_counts.values,
                title="Répartition des types de données"
            )
            self.visualizations["data_types"] = fig_dtypes.to_json()
            
            # Valeurs manquantes
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                fig_nulls = px.bar(
                    x=null_counts.index,
                    y=null_counts.values,
                    title="Valeurs manquantes par colonne"
                )
                self.visualizations["missing_values"] = fig_nulls.to_json()
            
            # Corrélations pour les variables numériques
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Matrice de corrélation",
                    color_continuous_scale="RdBu"
                )
                self.visualizations["correlation_matrix"] = fig_corr.to_json()
                
        except Exception as e:
            print(f"Erreur création visualisations DataFrame: {e}")

# Data Scientist automatisé
class AutoDataScientist:
    def __init__(self, file_path: str, target_variable: Optional[str] = None):
        self.file_path = file_path
        self.target_variable = target_variable
        self.cleaned_data = None
        self.model = None
        self.results = {}
    
    def load_and_clean_data(self):
        """Charge et nettoie les données automatiquement"""
        try:
            # Charger les données
            if self.file_path.endswith('.csv'):
                data = pd.read_csv(self.file_path)
            elif self.file_path.endswith('.json'):
                # Essayer de convertir JSON en DataFrame
                with open(self.file_path, 'r') as f:
                    json_data = json.load(f)
                data = self._json_to_dataframe(json_data)
            else:
                raise ValueError("Format non supporté pour l'étude data science")
            
            # Nettoyage automatique
            cleaned_data = self._auto_clean_data(data)
            self.cleaned_data = cleaned_data
            
            return cleaned_data
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erreur traitement données: {e}")
    
    def _json_to_dataframe(self, json_data):
        """Convertit des données JSON en DataFrame"""
        if isinstance(json_data, list):
            return pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            # Essayer de trouver une structure tabulaire
            for key, value in json_data.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], dict):
                        return pd.DataFrame(value)
            
            # Si pas de structure tabulaire, créer un DataFrame simple
            return pd.DataFrame([json_data])
        else:
            raise ValueError("Structure JSON non convertible en DataFrame")
    
    def _auto_clean_data(self, data):
        """Nettoyage automatique des données"""
        cleaned = data.copy()
        
        # Supprimer les colonnes avec trop de valeurs manquantes (>50%)
        null_percentage = cleaned.isnull().sum() / len(cleaned)
        columns_to_drop = null_percentage[null_percentage > 0.5].index
        cleaned = cleaned.drop(columns=columns_to_drop)
        
        # Remplir les valeurs manquantes
        for column in cleaned.columns:
            if cleaned[column].dtype in ['object', 'string']:
                # Mode pour les variables catégorielles
                mode_value = cleaned[column].mode()
                if not mode_value.empty:
                    cleaned[column].fillna(mode_value[0], inplace=True)
            else:
                # Médiane pour les variables numériques
                median_value = cleaned[column].median()
                cleaned[column].fillna(median_value, inplace=True)
        
        # Encoder les variables catégorielles
        label_encoders = {}
        for column in cleaned.columns:
            if cleaned[column].dtype == 'object':
                le = LabelEncoder()
                cleaned[column] = le.fit_transform(cleaned[column].astype(str))
                label_encoders[column] = le
        
        # Sauvegarder les encoders pour utilisation future
        self.label_encoders = label_encoders
        
        return cleaned
    
    def auto_train_model(self, problem_type="auto"):
        """Entraîne automatiquement un modèle"""
        if self.cleaned_data is None:
            raise ValueError("Données pas encore nettoyées")
        
        data = self.cleaned_data
        
        # Déterminer le type de problème automatiquement
        if problem_type == "auto":
            if self.target_variable and self.target_variable in data.columns:
                target = data[self.target_variable]
                unique_values = target.nunique()
                
                if unique_values <= 10:
                    problem_type = "classification"
                else:
                    problem_type = "regression"
            else:
                # Pas de variable cible spécifiée, faire de l'analyse exploratoire
                problem_type = "exploratory"
        
        if problem_type == "exploratory":
            return self._exploratory_analysis()
        
        # Préparer les données pour l'entraînement
        if self.target_variable not in data.columns:
            raise ValueError(f"Variable cible '{self.target_variable}' non trouvée")
        
        X = data.drop(columns=[self.target_variable])
        y = data[self.target_variable]
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Entraînement du modèle
        if problem_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results = {
                "problem_type": "classification",
                "accuracy": accuracy,
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "feature_importance": dict(zip(X.columns, model.feature_importances_))
            }
            
        elif problem_type == "regression":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            self.results = {
                "problem_type": "regression",
                "mse": mse,
                "rmse": rmse,
                "r2_score": model.score(X_test_scaled, y_test),
                "feature_importance": dict(zip(X.columns, model.feature_importances_))
            }
        
        self.model = model
        self.scaler = scaler
        
        return self.results
    
    def _exploratory_analysis(self):
        """Analyse exploratoire automatique"""
        data = self.cleaned_data
        
        results = {
            "problem_type": "exploratory",
            "data_shape": data.shape,
            "summary_statistics": data.describe().to_dict(),
            "correlation_analysis": data.corr().to_dict() if len(data.select_dtypes(include=[np.number]).columns) > 1 else {},
            "recommendations": self._generate_recommendations(data)
        }
        
        return results
    
    def _generate_recommendations(self, data):
        """Génère des recommandations automatiques"""
        recommendations = []
        
        # Recommandations basées sur la structure des données
        if data.shape[1] > 10:
            recommendations.append("Considérer une réduction de dimensionnalité (PCA)")
        
        if data.isnull().sum().sum() > 0:
            recommendations.append("Traitement des valeurs manquantes nécessaire")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            recommendations.append("Analyse de corrélation recommandée")
        
        if data.shape[0] < 100:
            recommendations.append("Dataset petit - considérer l'augmentation de données")
        
        return recommendations
    
    def save_results(self, user_id: str, file_id: str):
        """Sauvegarde les résultats de l'étude"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les données nettoyées
        cleaned_data_path = f"processed_data/{user_id}/cleaned_{file_id}_{timestamp}.csv"
        os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
        self.cleaned_data.to_csv(cleaned_data_path, index=False)
        
        # Sauvegarder le modèle si existant
        model_path = None
        if self.model is not None:
            model_path = f"models/{user_id}/model_{file_id}_{timestamp}.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            model_package = {
                "model": self.model,
                "scaler": getattr(self, "scaler", None),
                "label_encoders": getattr(self, "label_encoders", {}),
                "feature_names": list(self.cleaned_data.columns)
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
        
        return cleaned_data_path, model_path

# Endpoints de l'API

@app.post("/data/collect")
async def start_data_collection(config: DataCollectionConfig, background_tasks: BackgroundTasks):
    """Démarre la collecte de données avec consentement explicite"""
    try:
        collection_id = str(uuid.uuid4())
        
        # Enregistrer la configuration
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO data_collections (collection_id, user_id, config, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            collection_id,
            config.user_id,
            json.dumps(config.dict()),
            "in_progress",
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Lancer la collecte en arrière-plan
        collector = PersonalDataCollector(config.user_id, config)
        
        async def collect_data():
            await collector.collect_network_data()
            await collector.collect_system_data() 
            await collector.collect_file_metadata()
            await collector.save_collected_data()
            
            # Marquer comme terminé
            conn = sqlite3.connect('personal_data_platform.db')
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE data_collections 
                SET status = 'completed', completed_at = ?
                WHERE collection_id = ?
            ''', (datetime.now().isoformat(), collection_id))
            conn.commit()
            conn.close()
        
        background_tasks.add_task(collect_data)
        
        return {
            "collection_id": collection_id,
            "status": "started",
            "message": "Collecte de données démarrée avec votre consentement"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/collections/{user_id}")
async def get_user_collections(user_id: str):
    """Récupère toutes les collectes d'un utilisateur"""
    try:
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT collection_id, config, status, created_at, completed_at
            FROM data_collections WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        collections = []
        for row in cursor.fetchall():
            collections.append({
                "collection_id": row[0],
                "config": json.loads(row[1]),
                "status": row[2],
                "created_at": row[3],
                "completed_at": row[4]
            })
        
        conn.close()
        return {"collections": collections}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/files/{user_id}")
async def get_user_files(user_id: str):
    """Récupère tous les fichiers collectés d'un utilisateur"""
    try:
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_id, file_name, data_type, source_type, size_bytes, created_at, processed_at, metadata
            FROM collected_files WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        files = []
        for row in cursor.fetchall():
            files.append({
                "file_id": row[0],
                "file_name": row[1],
                "data_type": row[2],
                "source_type": row[3],
                "size_bytes": row[4],
                "created_at": row[5],
                "processed_at": row[6],
                "metadata": json.loads(row[7]) if row[7] else {}
            })
        
        conn.close()
        return {"files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/analyze")
async def analyze_data(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Lance une analyse des données"""
    try:
        # Vérifier que le fichier appartient à l'utilisateur
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path FROM collected_files 
            WHERE file_id = ? AND user_id = ?
        ''', (request.file_id, request.user_id))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        file_path = result[0]
        analysis_id = str(uuid.uuid4())
        
        # Enregistrer l'analyse comme en cours
        cursor.execute('''
            INSERT INTO data_analyses (analysis_id, file_id, user_id, analysis_type, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            request.file_id,
            request.user_id,
            request.analysis_type,
            datetime.now().isoformat(),
            "in_progress"
        ))
        
        conn.commit()
        conn.close()
        
        # Lancer l'analyse en arrière-plan
        async def run_analysis():
            try:
                analyzer = DataAnalyzer(file_path, request.analysis_type)
                data = analyzer.load_data()
                results = analyzer.descriptive_analysis(data)
                
                # Sauvegarder les résultats
                conn = sqlite3.connect('personal_data_platform.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE data_analyses 
                    SET results = ?, visualizations = ?, status = ?
                    WHERE analysis_id = ?
                ''', (
                    json.dumps(results),
                    json.dumps(analyzer.visualizations),
                    "completed",
                    analysis_id
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                # Marquer l'analyse comme échouée
                conn = sqlite3.connect('personal_data_platform.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE data_analyses SET status = ?, results = ?
                    WHERE analysis_id = ?
                ''', ("failed", json.dumps({"error": str(e)}), analysis_id))
                conn.commit()
                conn.close()
        
        background_tasks.add_task(run_analysis)
        
        return {
            "analysis_id": analysis_id,
            "status": "started",
            "message": "Analyse des données démarrée"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/study")
async def study_data(request: StudyRequest, background_tasks: BackgroundTasks):
    """Lance une étude data science complète"""
    study_id = str(uuid.uuid4())
    try:
        # Vérifier que le fichier appartient à l'utilisateur
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path FROM collected_files 
            WHERE file_id = ? AND user_id = ?
        ''', (request.file_id, request.user_id))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        file_path = result[0]
        study_id = str(uuid.uuid4())
        
        # Enregistrer l'étude comme en cours
        cursor.execute('''
            INSERT INTO data_studies (study_id, file_id, user_id, problem_type, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            study_id,
            request.file_id,
            request.user_id,
            request.problem_type,
            datetime.now().isoformat(),
            "in_progress"
        ))
        
        conn.commit()
        conn.close()
        
        # Lancer l'étude en arrière-plan
        async def run_study():
            try:
                scientist = AutoDataScientist(file_path, request.target_variable)
                scientist.load_and_clean_data()
                results = scientist.auto_train_model(request.problem_type)
                
                # Nettoyer les floats avant de sauvegarder
                results = sanitize_floats(results)
                
                # Sauvegarder les données nettoyées et le modèle
                cleaned_data_path, model_path = scientist.save_results(
                    request.user_id, request.file_id
                )
                
                # Sauvegarder les résultats
                conn = sqlite3.connect('personal_data_platform.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE data_studies 
                    SET model_performance = ?, cleaned_data_path = ?, model_path = ?, status = ?
                    WHERE study_id = ?
                ''', (
                    json.dumps(results),
                    cleaned_data_path,
                    model_path,
                    "completed",
                    study_id
                ))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                # Marquer l'étude comme échouée
                conn = sqlite3.connect('personal_data_platform.db')
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE data_studies SET status = ?, model_performance = ?
                    WHERE study_id = ?
                ''', ("failed", json.dumps({"error": str(e)}), study_id))
                conn.commit()
                conn.close()
        
        background_tasks.add_task(run_study)
        # data = sanitize_floats(data)

        return {
            "study_id": study_id,
            "status": "started",
            "message": "Étude data science démarrée"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/analyses/{user_id}")
async def get_user_analyses(user_id: str):
    """Récupère toutes les analyses d'un utilisateur"""
    try:
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.analysis_id, a.file_id, a.analysis_type, a.results, 
                   a.visualizations, a.created_at, a.status, f.file_name
            FROM data_analyses a
            JOIN collected_files f ON a.file_id = f.file_id
            WHERE a.user_id = ?
            ORDER BY a.created_at DESC
        ''', (user_id,))
        
        analyses = []
        for row in cursor.fetchall():
            analyses.append({
                "analysis_id": row[0],
                "file_id": row[1],
                "analysis_type": row[2],
                "results": json.loads(row[3]) if row[3] else {},
                "visualizations": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
                "status": row[6],
                "file_name": row[7]
            })
        
        conn.close()
        return {"analyses": analyses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/studies/{user_id}")
async def get_user_studies(user_id: str):
    """Récupère toutes les études d'un utilisateur"""
    try:
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.study_id, s.file_id, s.problem_type, s.model_performance,
                   s.cleaned_data_path, s.model_path, s.created_at, s.status, f.file_name
            FROM data_studies s
            JOIN collected_files f ON s.file_id = f.file_id
            WHERE s.user_id = ?
            ORDER BY s.created_at DESC
        ''', (user_id,))
        
        studies = []
        for row in cursor.fetchall():
            studies.append({
                "study_id": row[0],
                "file_id": row[1],
                "problem_type": row[2],
                "model_performance": json.loads(row[3]) if row[3] else {},
                "cleaned_data_path": row[4],
                "model_path": row[5],
                "created_at": row[6],
                "status": row[7],
                "file_name": row[8]
            })
        
        conn.close()
        return {"studies": studies}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/sell")
async def create_sale_offer(offer: DataSaleOffer):
    """Crée une offre de vente de données"""
    try:
        # Vérifier que le fichier appartient à l'utilisateur
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path FROM collected_files 
            WHERE file_id = ? AND user_id = ?
        ''', (offer.file_id, offer.user_id))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        # Anonymiser les données
        original_file_path = result[0]
        anonymized_path = f"anonymized_data/{offer.user_id}/anon_{offer.file_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # TODO: Implémenter l'anonymisation réelle
        os.makedirs(os.path.dirname(anonymized_path), exist_ok=True)
        
        sale_id = str(uuid.uuid4())
        
        # Enregistrer l'offre de vente
        cursor.execute('''
            INSERT INTO data_sales (sale_id, file_id, seller_id, price, description, 
                                  anonymized_file_path, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sale_id,
            offer.file_id,
            offer.user_id,
            offer.price,
            offer.description,
            anonymized_path,
            "available",
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "sale_id": sale_id,
            "status": "listed",
            "message": "Votre offre de vente a été créée"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/donate")
async def create_donation(donation: DataDonation):
    """Crée un don de données"""
    try:
        # Vérifier que le fichier appartient à l'utilisateur
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path FROM collected_files 
            WHERE file_id = ? AND user_id = ?
        ''', (donation.file_id, donation.user_id))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Fichier non trouvé")
        
        donation_id = str(uuid.uuid4())
        
        # Enregistrer le don
        cursor.execute('''
            INSERT INTO data_donations (donation_id, file_id, donor_id, recipient_organization,
                                      purpose, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            donation_id,
            donation.file_id,
            donation.user_id,
            donation.recipient_organization,
            donation.purpose,
            "pending",
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "donation_id": donation_id,
            "status": "pending",
            "message": "Votre don de données a été enregistré"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consent/record")
async def record_consent(consent: ConsentRecord):
    """Enregistre le consentement d'un utilisateur"""
    try:
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_consents (user_id, data_type, consent_given, timestamp, expires_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            consent.user_id,
            consent.data_type,
            consent.consent_given,
            consent.timestamp,
            consent.expires_at
        ))
        
        conn.commit()
        conn.close()
        
        return {"status": "recorded", "message": "Consentement enregistré"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/consent/{user_id}")
async def get_user_consents(user_id: str):
    """Récupère tous les consentements d'un utilisateur"""
    try:
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data_type, consent_given, timestamp, expires_at, revoked_at
            FROM user_consents WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,))
        
        consents = []
        for row in cursor.fetchall():
            consents.append({
                "data_type": row[0],
                "consent_given": row[1],
                "timestamp": row[2],
                "expires_at": row[3],
                "revoked_at": row[4]
            })
        
        conn.close()
        return {"consents": consents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/marketplace/sales")
async def get_available_sales():
    """Récupère toutes les offres de vente disponibles"""
    try:
        conn = sqlite3.connect('personal_data_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.sale_id, s.price, s.description, s.created_at, f.data_type, f.size_bytes
            FROM data_sales s
            JOIN collected_files f ON s.file_id = f.file_id
            WHERE s.status = 'available'
            ORDER BY s.created_at DESC
        ''')
        
        sales = []
        for row in cursor.fetchall():
            sales.append({
                "sale_id": row[0],
                "price": row[1],
                "description": row[2],
                "created_at": row[3],
                "data_type": row[4],
                "size_bytes": row[5]
            })
        
        conn.close()
        return {"available_sales": sales}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Vérification de santé du service"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)