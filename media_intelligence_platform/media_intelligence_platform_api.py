# media_intelligence_platform.py - API pour l'analyse multimodale

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import json
import os
import sqlite3
import uuid
from datetime import datetime
import asyncio
import aiofiles
from pathlib import Path
import hashlib

# Imports pour traitement multimodal uvicorn media_intelligence_platform_api:app --reload --host 0.0.0.0 --port 8032
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import librosa
import speech_recognition as sr
import whisper
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import torchvision.transforms as transforms
from moviepy import VideoFileClip
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
UPLOAD_DIR = "uploaded_media"
PROCESSED_DIR = "processed_media_data"
MODELS_DIR = "trained_models"
MARKETPLACE_DIR = "marketplace_media"

# Modèles Pydantic
class MediaUpload(BaseModel):
    user_id: str
    media_type: str  # "image", "video", "audio", "text"   
    filename: str
    extraction_config: Dict[str, Any] = {}

class DataExtractionResult(BaseModel):
    media_id: str
    extracted_data: Dict[str, Any]
    metadata: Dict[str, Any]
    extraction_timestamp: str

class MediaAnalysisRequest(BaseModel):
    media_id: str
    user_id: str
    analysis_type: str = "comprehensive"

class MediaStudyRequest(BaseModel):
    media_id: str
    user_id: str
    target_task: str = "auto"  # classification, regression, generation, etc.

# Base de données
def init_media_db():
    conn = sqlite3.connect('media_intelligence.db')
    cursor = conn.cursor()
    
    # Table des médias uploadés
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id TEXT UNIQUE NOT NULL,
            user_id TEXT NOT NULL,
            filename TEXT,
            media_type TEXT,
            file_path TEXT,
            file_size INTEGER,
            upload_timestamp TEXT,
            processing_status TEXT DEFAULT 'pending'
        )
    ''')
    
    # Table des données extraites
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            extraction_id TEXT UNIQUE NOT NULL,
            media_id TEXT NOT NULL,
            extracted_features TEXT,
            metadata TEXT,
            data_file_path TEXT,
            extraction_timestamp TEXT,
            FOREIGN KEY (media_id) REFERENCES uploaded_media (media_id)
        )
    ''')
    
    # Table des analyses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            media_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            analysis_type TEXT,
            results TEXT,
            visualizations TEXT,
            created_at TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    # Table des études ML
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id TEXT UNIQUE NOT NULL,
            media_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            target_task TEXT,
            model_path TEXT,
            performance_metrics TEXT,
            trained_at TEXT,
            status TEXT DEFAULT 'pending'
        )
    ''')
    
    # Table marketplace médias
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_marketplace (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            listing_id TEXT UNIQUE NOT NULL,
            media_id TEXT NOT NULL,
            seller_id TEXT NOT NULL,
            media_type TEXT,
            price REAL,
            description TEXT,
            features_preview TEXT,
            status TEXT DEFAULT 'available',
            created_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

app = FastAPI(title="Media Intelligence Platform", version="1.0.0")

# Extracteurs de données multimodaux
class ImageDataExtractor:
    """Extraction complète de données depuis les images"""
    
    def __init__(self):
        # Modèles pré-entraînés
        self.feature_extractor = pipeline("image-feature-extraction", model="google/vit-base-patch16-224")
        self.object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        self.image_classifier = pipeline("image-classification", model="microsoft/resnet-50")
        self.ocr_reader = pipeline("image-to-text", model="microsoft/trocr-base-printed")
    
    async def extract_comprehensive_data(self, image_path: str) -> Dict[str, Any]:
        """Extraction complète des données d'une image"""
        image = Image.open(image_path)
        
        extracted_data = {
            "basic_properties": self._extract_basic_properties(image),
            "visual_features": await self._extract_visual_features(image),
            "objects_detected": await self._detect_objects(image),
            "text_content": await self._extract_text(image),
            "aesthetic_analysis": await self._analyze_aesthetics(image),
            "technical_metadata": self._extract_exif_data(image_path)
        }
        
        return extracted_data
    
    def _extract_basic_properties(self, image: Image) -> Dict[str, Any]:
        """Propriétés de base de l'image"""
        return {
            "dimensions": image.size,
            "format": image.format,
            "mode": image.mode,
            "color_channels": len(image.getbands()) if hasattr(image, 'getbands') else 3,
            "aspect_ratio": image.size[0] / image.size[1] if image.size[1] != 0 else 0
        }
    
    async def _extract_visual_features(self, image: Image) -> Dict[str, Any]:
        """Features visuelles avancées"""
        try:
            # Embeddings visuels
            features = self.feature_extractor(image)
            
            # Classification générale
            classifications = self.image_classifier(image)
            
            # Analyse couleurs dominantes
            color_analysis = self._analyze_colors(image)
            
            return {
                "embeddings_shape": np.array(features).shape,
                "top_classifications": classifications[:3],
                "dominant_colors": color_analysis,
                "brightness_avg": self._calculate_brightness(image),
                "contrast_score": self._calculate_contrast(image)
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _detect_objects(self, image: Image) -> List[Dict[str, Any]]:
        """Détection d'objets dans l'image"""
        try:
            detections = self.object_detector(image)
            return [{
                "label": detection["label"],
                "confidence": detection["score"],
                "bbox": detection["box"]
            } for detection in detections]
        except Exception as e:
            return [{"error": str(e)}]
    
    async def _extract_text(self, image: Image) -> Dict[str, Any]:
        """Extraction de texte (OCR)"""
        try:
            # OCR avec TrOCR
            ocr_result = self.ocr_reader(image)
            return {
                "extracted_text": ocr_result[0]["generated_text"] if ocr_result else "",
                "text_confidence": ocr_result[0].get("score", 0) if ocr_result else 0
            }
        except Exception as e:
            return {"error": str(e), "extracted_text": ""}
    
    def _analyze_colors(self, image: Image) -> Dict[str, Any]:
        """Analyse des couleurs dominantes"""
        # Convertir en numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Reshape pour k-means
        pixels = img_array.reshape(-1, 3)
        
        # K-means pour couleurs dominantes (simplifié)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        
        return {
            "dominant_colors": [{"rgb": color.tolist()} for color in colors],
            "color_count": len(colors)
        }
    
    def _calculate_brightness(self, image: Image) -> float:
        """Calcul de la luminosité moyenne"""
        grayscale = image.convert('L')
        return float(np.array(grayscale).mean())
    
    def _calculate_contrast(self, image: Image) -> float:
        """Calcul du contraste"""
        grayscale = np.array(image.convert('L'))
        return float(grayscale.std())

class VideoDataExtractor:
    """Extraction complète de données depuis les vidéos"""
    
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
    
    async def extract_comprehensive_data(self, video_path: str) -> Dict[str, Any]:
        """Extraction complète des données d'une vidéo"""
        try:
            clip = VideoFileClip(video_path)
            
            extracted_data = {
                "video_properties": self._extract_video_properties(clip),
                "frame_analysis": await self._analyze_frames(video_path),
                "audio_features": await self._extract_audio_features(clip),
                "scene_detection": await self._detect_scenes(video_path),
                "motion_analysis": await self._analyze_motion(video_path),
                "transcription": await self._transcribe_audio(video_path)
            }
            
            clip.close()
            return extracted_data
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_video_properties(self, clip: VideoFileClip) -> Dict[str, Any]:
        """Propriétés de base de la vidéo"""
        return {
            "duration": clip.duration,
            "fps": clip.fps,
            "size": clip.size,
            "aspect_ratio": clip.size[0] / clip.size[1] if clip.size[1] != 0 else 0,
            "total_frames": int(clip.duration * clip.fps) if clip.fps else 0,
            "has_audio": clip.audio is not None
        }
    
    async def _analyze_frames(self, video_path: str) -> Dict[str, Any]:
        """Analyse des frames clés"""
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Analyser quelques frames clés
        key_frames = []
        sample_frames = min(10, frame_count // 10) if frame_count > 10 else frame_count
        
        for i in range(0, frame_count, frame_count // sample_frames if sample_frames > 0 else 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Analyse basique du frame
                frame_analysis = {
                    "frame_number": i,
                    "timestamp": i / fps if fps > 0 else 0,
                    "brightness": float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()),
                    "dominant_color": self._get_dominant_color(frame)
                }
                key_frames.append(frame_analysis)
        
        cap.release()
        
        return {
            "total_frames_analyzed": len(key_frames),
            "key_frames": key_frames,
            "avg_brightness": np.mean([f["brightness"] for f in key_frames]) if key_frames else 0
        }
    
    async def _extract_audio_features(self, clip: VideoFileClip) -> Dict[str, Any]:
        """Extraction des caractéristiques audio"""
        if clip.audio is None:
            return {"error": "No audio track"}
        
        try:
            # Extraire l'audio temporairement
            temp_audio_path = f"temp_audio_{uuid.uuid4()}.wav"
            clip.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            
            # Analyse avec librosa
            y, sr = librosa.load(temp_audio_path)
            
            features = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
                "spectral_centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
                "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y).mean()),
                "mfcc_features": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
                "rms_energy": float(librosa.feature.rms(y=y).mean())
            }
            
            # Nettoyage
            os.remove(temp_audio_path)
            
            return features
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_dominant_color(self, frame) -> List[int]:
        """Couleur dominante d'un frame"""
        # Redimensionner pour performance
        small_frame = cv2.resize(frame, (50, 50))
        
        # Moyenne des couleurs
        avg_color = small_frame.mean(axis=(0, 1))
        return avg_color.astype(int).tolist()

class AudioDataExtractor:
    """Extraction complète de données depuis l'audio"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.whisper_model = whisper.load_model("base")
    
    async def extract_comprehensive_data(self, audio_path: str) -> Dict[str, Any]:
        """Extraction complète des données audio"""
        try:
            # Charger l'audio
            y, sr = librosa.load(audio_path)
            
            extracted_data = {
                "audio_properties": self._extract_audio_properties(y, sr),
                "spectral_features": self._extract_spectral_features(y, sr),
                "temporal_features": self._extract_temporal_features(y, sr),
                "transcription": await self._transcribe_audio(audio_path),
                "emotion_analysis": await self._analyze_audio_emotion(y, sr),
                "music_analysis": self._analyze_music_features(y, sr)
            }
            
            return extracted_data
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_audio_properties(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Propriétés de base de l'audio"""
        return {
            "duration": len(y) / sr,
            "sample_rate": sr,
            "channels": 1,  # librosa charge en mono par défaut
            "total_samples": len(y),
            "bit_depth": 32,  # float32 par défaut
            "dynamic_range": float(y.max() - y.min()),
            "rms_level": float(librosa.feature.rms(y=y).mean())
        }
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Caractéristiques spectrales"""
        return {
            "spectral_centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
            "spectral_rolloff": float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()),
            "spectral_bandwidth": float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()),
            "zero_crossing_rate": float(librosa.feature.zero_crossing_rate(y).mean()),
            "mfcc_features": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
            "chroma_features": librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1).tolist(),
            "mel_spectrogram": librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1).tolist()
        }
    
    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Caractéristiques temporelles"""
        return {
            "tempo": float(librosa.beat.tempo(y=y, sr=sr)[0]),
            "beat_frames": len(librosa.beat.beat_track(y=y, sr=sr)[1]),
            "onset_frames": len(librosa.onset.onset_detect(y=y, sr=sr)),
            "silence_ratio": float(np.sum(np.abs(y) < 0.01) / len(y)),
            "energy_variance": float(np.var(librosa.feature.rms(y=y)))
        }
    
    async def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcription audio vers texte"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "confidence": 0.8,  # Whisper ne fournit pas toujours la confiance
                "segments": len(result.get("segments", []))
            }
        except Exception as e:
            return {"error": str(e), "text": ""}

class TextDataExtractor:
    """Extraction complète de données depuis le texte"""
    
    def __init__(self):
        # Modèles NLP
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.ner_analyzer = pipeline("ner", aggregation_strategy="simple")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.emotion_analyzer = pipeline("text-classification", 
                                       model="j-hartmann/emotion-english-distilroberta-base")
    
    async def extract_comprehensive_data(self, text_content: str) -> Dict[str, Any]:
        """Extraction complète des données textuelles"""
        try:
            extracted_data = {
                "text_properties": self._extract_text_properties(text_content),
                "linguistic_features": self._extract_linguistic_features(text_content),
                "semantic_analysis": await self._analyze_semantics(text_content),
                "entity_extraction": await self._extract_entities(text_content),
                "sentiment_emotion": await self._analyze_sentiment_emotion(text_content),
                "readability_metrics": self._calculate_readability(text_content),
                "topic_modeling": await self._extract_topics(text_content)
            }
            
            return extracted_data
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_text_properties(self, text: str) -> Dict[str, Any]:
        """Propriétés de base du texte"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len(text.split('\n\n')),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "vocabulary_richness": len(set(words)) / len(words) if words else 0
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Caractéristiques linguistiques"""
        import re
        
        # Compter différents éléments
        uppercase_words = len(re.findall(r'\b[A-Z]+\b', text))
        numbers = len(re.findall(r'\d+', text))
        punctuation = len(re.findall(r'[!?.,;:]', text))
        urls = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        emails = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        
        return {
            "uppercase_words": uppercase_words,
            "numbers_count": numbers,
            "punctuation_count": punctuation,
            "urls_count": urls,
            "emails_count": emails,
            "special_chars": len(re.findall(r'[^a-zA-Z0-9\s]', text))
        }
    
    async def _analyze_sentiment_emotion(self, text: str) -> Dict[str, Any]:
        """Analyse sentiment et émotions"""
        try:
            # Limiter la taille du texte pour les modèles
            text_sample = text[:512] if len(text) > 512 else text
            
            sentiment = self.sentiment_analyzer(text_sample)
            emotions = self.emotion_analyzer(text_sample)
            
            return {
                "sentiment": {
                    "label": sentiment[0]["label"],
                    "confidence": sentiment[0]["score"]
                },
                "emotions": {
                    "primary_emotion": emotions[0]["label"],
                    "confidence": emotions[0]["score"],
                    "all_emotions": emotions
                }
            }
        except Exception as e:
            return {"error": str(e)}

# Endpoints API
@app.on_event("startup")
async def startup_event():
    # Créer les dossiers nécessaires
    for directory in [UPLOAD_DIR, PROCESSED_DIR, MODELS_DIR, MARKETPLACE_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialiser la base de données
    init_media_db()

@app.post("/media/upload")
async def upload_media(
    file: UploadFile = File(...),
    user_id: str = "default_user",
    background_tasks: BackgroundTasks = None
):
    """Upload et traitement d'un fichier média"""
    try:
        # Générer un ID unique
        media_id = str(uuid.uuid4())
        
        # Déterminer le type de média
        file_extension = Path(file.filename).suffix.lower()
        media_type = _determine_media_type(file_extension)
        
        if not media_type:
            raise HTTPException(status_code=400, detail="Type de fichier non supporté")
        
        # Sauvegarder le fichier
        file_path = os.path.join(UPLOAD_DIR, f"{media_id}_{file.filename}")
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Enregistrer en base de données
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO uploaded_media (media_id, user_id, filename, media_type, file_path, file_size, upload_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            media_id, user_id, file.filename, media_type, file_path,
            len(content), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Lancer l'extraction en arrière-plan
        if background_tasks:
            background_tasks.add_task(extract_media_data, media_id, file_path, media_type, user_id)
        
        return {
            "media_id": media_id,
            "status": "uploaded",
            "message": "Fichier uploadé avec succès, extraction des données en cours...",
            "media_type": media_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def extract_media_data(media_id: str, file_path: str, media_type: str, user_id: str):
    """Extraction des données en arrière-plan"""
    try:
        # Sélectionner l'extracteur approprié
        if media_type == "image":
            extractor = ImageDataExtractor()
            extracted_data = await extractor.extract_comprehensive_data(file_path)
        elif media_type == "video":
            extractor = VideoDataExtractor()
            extracted_data = await extractor.extract_comprehensive_data(file_path)
        elif media_type == "audio":
            extractor = AudioDataExtractor()
            extracted_data = await extractor.extract_comprehensive_data(file_path)
        elif media_type == "text":
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            extractor = TextDataExtractor()
            extracted_data = await extractor.extract_comprehensive_data(text_content)
        else:
            raise ValueError(f"Type de média non supporté: {media_type}")
        
        # Sauvegarder les données extraites
        extraction_id = str(uuid.uuid4())
        data_file_path = os.path.join(PROCESSED_DIR, f"extracted_{extraction_id}.json")
        
        with open(data_file_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, default=str)
        
        # Mettre à jour la base de données
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        # Marquer l'upload comme traité
        cursor.execute('''
            UPDATE uploaded_media 
            SET processing_status = 'completed'
            WHERE media_id = ?
        ''', (media_id,))
        
        # Enregistrer les données extraites
        cursor.execute('''
            INSERT INTO extracted_data (extraction_id, media_id, extracted_features, metadata, data_file_path, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            extraction_id,
            media_id,
            json.dumps(extracted_data),
            json.dumps({"user_id": user_id, "media_type": media_type}),
            data_file_path,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        # Marquer l'extraction comme échouée
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE uploaded_media 
            SET processing_status = 'failed'
            WHERE media_id = ?
        ''', (media_id,))
        conn.commit()
        conn.close()
        
        print(f"Erreur extraction {media_id}: {e}")

def _determine_media_type(file_extension: str) -> str:
    """Détermine le type de média basé sur l'extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
    text_extensions = {'.txt', '.md', '.rtf', '.doc', '.docx'}
    
    if file_extension in image_extensions:
        return "image"
    elif file_extension in video_extensions:
        return "video"
    elif file_extension in audio_extensions:
        return "audio"
    elif file_extension in text_extensions:
        return "text"
    else:
        return None

@app.get("/media/user/{user_id}")
async def get_user_media(user_id: str):
    """Récupère tous les médias d'un utilisateur"""
    try:
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.media_id, m.filename, m.media_type, m.file_size, 
                   m.upload_timestamp, m.processing_status,
                   e.extraction_id, e.data_file_path
            FROM uploaded_media m
            LEFT JOIN extracted_data e ON m.media_id = e.media_id
            WHERE m.user_id = ?
            ORDER BY m.upload_timestamp DESC
        ''', (user_id,))
        
        media_files = []
        for row in cursor.fetchall():
            media_files.append({
                "media_id": row[0],
                "filename": row[1],
                "media_type": row[2],
                "file_size": row[3],
                "upload_timestamp": row[4],
                "processing_status": row[5],
                "extraction_id": row[6],
                "has_extracted_data": row[7] is not None
            })
        
        conn.close()
        return {"media_files": media_files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media/analyze")
async def analyze_media(request: MediaAnalysisRequest, background_tasks: BackgroundTasks):
    """Lance l'analyse d'un média"""
    try:
        analysis_id = str(uuid.uuid4())
        
        # Enregistrer l'analyse comme en cours
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO media_analyses (analysis_id, media_id, user_id, analysis_type, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            request.media_id,
            request.user_id,
            request.analysis_type,
            datetime.now().isoformat(),
            "in_progress"
        ))
        
        conn.commit()
        conn.close()
        
        # Lancer l'analyse en arrière-plan
        background_tasks.add_task(run_media_analysis, analysis_id, request.media_id, request.analysis_type)
        
        return {
            "analysis_id": analysis_id,
            "status": "started",
            "message": "Analyse des données multimodales démarrée"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_media_analysis(analysis_id: str, media_id: str, analysis_type: str):
    """Exécute l'analyse multimodale en arrière-plan"""
    try:
        # Récupérer les données extraites
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data_file_path, extracted_features 
            FROM extracted_data WHERE media_id = ?
        ''', (media_id,))
        
        result = cursor.fetchone()
        if not result:
            raise Exception("Données extraites non trouvées")
        
        data_file_path = result[0]
        
        # Charger les données extraites
        with open(data_file_path, 'r') as f:
            extracted_data = json.load(f)
        
        # Effectuer l'analyse selon le type
        analyzer = MediaAnalyzer()
        analysis_results = await analyzer.comprehensive_analysis(extracted_data, analysis_type)
        
        # Créer des visualisations
        visualizations = await analyzer.create_visualizations(extracted_data, analysis_results)
        
        # Sauvegarder les résultats
        cursor.execute('''
            UPDATE media_analyses 
            SET results = ?, visualizations = ?, status = ?
            WHERE analysis_id = ?
        ''', (
            json.dumps(analysis_results),
            json.dumps(visualizations),
            "completed",
            analysis_id
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        # Marquer l'analyse comme échouée
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE media_analyses SET status = ?, results = ?
            WHERE analysis_id = ?
        ''', ("failed", json.dumps({"error": str(e)}), analysis_id))
        conn.commit()
        conn.close()

@app.post("/media/study")
async def study_media(request: MediaStudyRequest, background_tasks: BackgroundTasks):
    """Lance une étude ML sur un média"""
    try:
        study_id = str(uuid.uuid4())
        
        # Enregistrer l'étude comme en cours
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO media_studies (study_id, media_id, user_id, target_task, trained_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            study_id,
            request.media_id,
            request.user_id,
            request.target_task,
            datetime.now().isoformat(),
            "in_progress"
        ))
        
        conn.commit()
        conn.close()
        
        # Lancer l'étude en arrière-plan
        background_tasks.add_task(run_media_study, study_id, request.media_id, request.target_task, request.user_id)
        
        return {
            "study_id": study_id,
            "status": "started",
            "message": "Étude ML multimodale démarrée"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_media_study(study_id: str, media_id: str, target_task: str, user_id: str):
    """Exécute l'étude ML multimodale en arrière-plan"""
    try:
        # Charger les données extraites
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data_file_path FROM extracted_data WHERE media_id = ?
        ''', (media_id,))
        
        result = cursor.fetchone()
        if not result:
            raise Exception("Données extraites non trouvées")
        
        with open(result[0], 'r') as f:
            extracted_data = json.load(f)
        
        # Effectuer l'étude ML
        ml_researcher = MediaMLResearcher()
        study_results = await ml_researcher.auto_study(extracted_data, target_task)
        
        # Sauvegarder le modèle
        model_path = os.path.join(MODELS_DIR, f"model_{study_id}.pkl")
        ml_researcher.save_model(model_path)
        
        # Sauvegarder les résultats
        cursor.execute('''
            UPDATE media_studies 
            SET model_path = ?, performance_metrics = ?, status = ?
            WHERE study_id = ?
        ''', (
            model_path,
            json.dumps(study_results),
            "completed",
            study_id
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE media_studies SET status = ?, performance_metrics = ?
            WHERE study_id = ?
        ''', ("failed", json.dumps({"error": str(e)}), study_id))
        conn.commit()
        conn.close()

# Classes d'analyse et ML
class MediaAnalyzer:
    """Analyseur de données multimodales"""
    
    async def comprehensive_analysis(self, extracted_data: Dict, analysis_type: str) -> Dict[str, Any]:
        """Analyse complète des données multimodales"""
        results = {
            "analysis_type": analysis_type,
            "summary_statistics": self._calculate_summary_stats(extracted_data),
            "pattern_detection": self._detect_patterns(extracted_data),
            "anomaly_detection": self._detect_anomalies(extracted_data),
            "correlation_analysis": self._analyze_correlations(extracted_data),
            "recommendations": self._generate_recommendations(extracted_data)
        }
        
        return results
    
    def _calculate_summary_stats(self, data: Dict) -> Dict[str, Any]:
        """Calcule les statistiques descriptives"""
        stats = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                stats[key] = self._analyze_nested_dict(value)
            elif isinstance(value, list):
                stats[key] = self._analyze_list(value)
            elif isinstance(value, (int, float)):
                stats[key] = {"value": value, "type": "numeric"}
        
        return stats
    
    def _analyze_nested_dict(self, nested_dict: Dict) -> Dict:
        """Analyse un dictionnaire imbriqué"""
        numeric_values = []
        for k, v in nested_dict.items():
            if isinstance(v, (int, float)):
                numeric_values.append(v)
        
        if numeric_values:
            return {
                "count": len(numeric_values),
                "mean": np.mean(numeric_values),
                "std": np.std(numeric_values),
                "min": np.min(numeric_values),
                "max": np.max(numeric_values)
            }
        else:
            return {"count": len(nested_dict), "type": "non_numeric"}
    
    def _analyze_list(self, data_list: List) -> Dict:
        """Analyse une liste de données"""
        if not data_list:
            return {"count": 0}
        
        numeric_items = [item for item in data_list if isinstance(item, (int, float))]
        
        if numeric_items:
            return {
                "count": len(numeric_items),
                "mean": np.mean(numeric_items),
                "std": np.std(numeric_items),
                "distribution": "numeric"
            }
        else:
            return {
                "count": len(data_list),
                "unique_items": len(set(str(item) for item in data_list)),
                "distribution": "categorical"
            }
    
    async def create_visualizations(self, extracted_data: Dict, analysis_results: Dict) -> Dict[str, str]:
        """Crée des visualisations pour les données multimodales"""
        visualizations = {}
        
        # Graphique de distribution des features
        if "summary_statistics" in analysis_results:
            fig = self._create_feature_distribution_chart(analysis_results["summary_statistics"])
            visualizations["feature_distribution"] = fig.to_json()
        
        # Graphique de corrélations si applicable
        correlations = analysis_results.get("correlation_analysis", {})
        if correlations:
            fig = self._create_correlation_heatmap(correlations)
            visualizations["correlation_heatmap"] = fig.to_json()
        
        return visualizations
    
    def _create_feature_distribution_chart(self, stats: Dict) -> go.Figure:
        """Crée un graphique de distribution des features"""
        features = []
        values = []
        
        for feature, stat in stats.items():
            if isinstance(stat, dict) and "mean" in stat:
                features.append(feature)
                values.append(stat["mean"])
        
        fig = go.Figure(data=[go.Bar(x=features, y=values)])
        fig.update_layout(
            title="Distribution des Features Extraites",
            xaxis_title="Features",
            yaxis_title="Valeurs Moyennes"
        )
        
        return fig

class MediaMLResearcher:
    """Chercheur ML pour données multimodales"""
    
    def __init__(self):
        self.trained_model = None
        self.performance_metrics = {}
    
    async def auto_study(self, extracted_data: Dict, target_task: str) -> Dict[str, Any]:
        """Étude ML automatique sur données multimodales"""
        
        # Préparer les données pour l'ML
        features = self._prepare_ml_features(extracted_data)
        
        if target_task == "auto":
            target_task = self._determine_best_task(features)
        
        # Entraîner le modèle selon la tâche
        if target_task in ["classification", "clustering"]:
            results = await self._train_classification_model(features)
        elif target_task == "generation":
            results = await self._train_generation_model(features)
        elif target_task == "similarity":
            results = await self._train_similarity_model(features)
        else:
            results = await self._train_regression_model(features)
        
        self.performance_metrics = results
        return results
    
    def _prepare_ml_features(self, extracted_data: Dict) -> np.ndarray:
        """Prépare les features pour l'ML"""
        features = []
        
        def extract_numeric_features(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    new_prefix = f"{prefix}_{key}" if prefix else key
                    extract_numeric_features(value, new_prefix)
            elif isinstance(data, list):
                if data and isinstance(data[0], (int, float)):
                    features.extend(data)
                elif data:
                    for i, item in enumerate(data[:10]):  # Limiter à 10 éléments
                        extract_numeric_features(item, f"{prefix}_{i}")
            elif isinstance(data, (int, float)):
                features.append(data)
        
        extract_numeric_features(extracted_data)
        
        # Normaliser et reshaper
        feature_array = np.array(features).reshape(1, -1) if features else np.array([[0]])
        return feature_array
    
    def _determine_best_task(self, features: np.ndarray) -> str:
        """Détermine la meilleure tâche ML selon les données"""
        if features.shape[1] > 50:
            return "dimensionality_reduction"
        elif features.shape[1] > 10:
            return "clustering"
        else:
            return "classification"
    
    async def _train_classification_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Entraîne un modèle de classification"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Puisqu'on n'a pas de labels, faire du clustering
        if features.shape[0] == 1:
            # Pas assez de données pour l'entraînement
            return {
                "task": "classification",
                "status": "insufficient_data",
                "message": "Données insuffisantes pour l'entraînement"
            }
        
        # Clustering comme proxy
        kmeans = KMeans(n_clusters=min(3, features.shape[0]), random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        try:
            silhouette_avg = silhouette_score(features, labels)
        except:
            silhouette_avg = 0.0
        
        self.trained_model = kmeans
        
        return {
            "task": "clustering",
            "n_clusters": len(set(labels)),
            "silhouette_score": silhouette_avg,
            "inertia": kmeans.inertia_,
            "feature_importance": "Not applicable for clustering"
        }
    
    def save_model(self, model_path: str):
        """Sauvegarde le modèle entraîné"""
        if self.trained_model:
            import joblib
            joblib.dump(self.trained_model, model_path)

@app.get("/media/analyses/{user_id}")
async def get_user_analyses(user_id: str):
    """Récupère toutes les analyses d'un utilisateur"""
    try:
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.analysis_id, a.media_id, a.analysis_type, a.results, 
                   a.visualizations, a.created_at, a.status, m.filename, m.media_type
            FROM media_analyses a
            JOIN uploaded_media m ON a.media_id = m.media_id
            WHERE a.user_id = ?
            ORDER BY a.created_at DESC
        ''', (user_id,))
        
        analyses = []
        for row in cursor.fetchall():
            analyses.append({
                "analysis_id": row[0],
                "media_id": row[1],
                "analysis_type": row[2],
                "results": json.loads(row[3]) if row[3] else {},
                "visualizations": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
                "status": row[6],
                "filename": row[7],
                "media_type": row[8]
            })
        
        conn.close()
        return {"analyses": analyses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/media/studies/{user_id}")
async def get_user_studies(user_id: str):
    """Récupère toutes les études d'un utilisateur"""
    try:
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.study_id, s.media_id, s.target_task, s.model_path,
                   s.performance_metrics, s.trained_at, s.status, m.filename, m.media_type
            FROM media_studies s
            JOIN uploaded_media m ON s.media_id = m.media_id
            WHERE s.user_id = ?
            ORDER BY s.trained_at DESC
        ''', (user_id,))
        
        studies = []
        for row in cursor.fetchall():
            studies.append({
                "study_id": row[0],
                "media_id": row[1],
                "target_task": row[2],
                "model_path": row[3],
                "performance_metrics": json.loads(row[4]) if row[4] else {},
                "trained_at": row[5],
                "status": row[6],
                "filename": row[7],
                "media_type": row[8]
            })
        
        conn.close()
        return {"studies": studies}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/media/marketplace/list")
async def create_marketplace_listing(
    media_id: str,
    seller_id: str,
    price: float,
    description: str
):
    """Crée une offre sur le marketplace"""
    try:
        listing_id = str(uuid.uuid4())
        
        # Récupérer les infos du média
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT media_type FROM uploaded_media WHERE media_id = ?
        ''', (media_id,))
        
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Média non trouvé")
        
        media_type = result[0]
        
        # Créer l'offre
        cursor.execute('''
            INSERT INTO media_marketplace (listing_id, media_id, seller_id, media_type, price, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            listing_id,
            media_id,
            seller_id,
            media_type,
            price,
            description,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "listing_id": listing_id,
            "status": "listed",
            "message": "Offre créée sur le marketplace"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/media/marketplace/listings")
async def get_marketplace_listings():
    """Récupère toutes les offres du marketplace"""
    try:
        conn = sqlite3.connect('media_intelligence.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ml.listing_id, ml.media_id, ml.seller_id, ml.media_type,
                   ml.price, ml.description, ml.created_at, m.filename
            FROM media_marketplace ml
            JOIN uploaded_media m ON ml.media_id = m.media_id
            WHERE ml.status = 'available'
            ORDER BY ml.created_at DESC
        ''')
        
        listings = []
        for row in cursor.fetchall():
            listings.append({
                "listing_id": row[0],
                "media_id": row[1],
                "seller_id": row[2],
                "media_type": row[3],
                "price": row[4],
                "description": row[5],
                "created_at": row[6],
                "filename": row[7]
            })
        
        conn.close()
        return {"listings": listings}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Vérification de santé du service"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8032)