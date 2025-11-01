# universal_tokenizer_api.py - API pour Tokenizer et Corpus Universels

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import os
import sqlite3
import uuid
from datetime import datetime
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from collections import Counter, defaultdict
import re
import unicodedata

# Imports tokenization
from transformers import AutoTokenizer, AutoConfig
import sentencepiece as spm
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.models import BPE, WordLevel, WordPiece, Unigram

# Imports pour langues et corpus
import spacy
from spacy.lang.xx import MultiLanguage
# import polyglot
# from polyglot.detect import Detector
import langdetect
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import fasttext

# Configuration
TOKENIZERS_DIR = "custom_tokenizers"
CORPUS_DIR = "multilingual_corpus"
MODELS_DIR = "trained_tokenizer_models"
ANALYSIS_DIR = "tokenizer_analysis"

# Modèles Pydantic  uvicorn tokenizer_ai_api:app --host 0.0.0.0 --port 8046 --reload
class TokenizerConfig(BaseModel):
    name: str
    tokenizer_type: str = "bpe"  # bpe, wordpiece, unigram, wordlevel
    vocab_size: int = 50000
    min_frequency: int = 2
    languages: List[str] = ["multi"]
    special_tokens: List[str] = ["<unk>", "<pad>", "<s>", "</s>", "<mask>"]
    normalization: bool = True
    lowercase: bool = True
    strip_accents: bool = False

class TokenizeRequest(BaseModel):
    text: str
    tokenizer_name: Optional[str] = "default"
    return_tokens: bool = True
    return_ids: bool = True
    return_attention_mask: bool = False
    return_analysis: bool = False

class CorpusAnalysisRequest(BaseModel):
    corpus_name: str
    analysis_type: str = "comprehensive"  # basic, comprehensive, linguistic
    sample_size: Optional[int] = 10000
    languages: Optional[List[str]] = None

class TokenizerTrainingRequest(BaseModel):
    name: str
    corpus_sources: List[str]
    config: TokenizerConfig
    user_id: str

# Base de données
def init_tokenizer_db():
    conn = sqlite3.connect('universal_tokenizer.db')
    cursor = conn.cursor()
    
    # Table des tokenizers
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tokenizers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tokenizer_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            tokenizer_type TEXT NOT NULL,
            vocab_size INTEGER,
            languages TEXT,
            file_path TEXT,
            config_json TEXT,
            created_at TEXT,
            trained_by TEXT,
            status TEXT DEFAULT 'active',
            performance_metrics TEXT
        )
    ''')
    
    # Table des corpus
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS corpus_collection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            corpus_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            languages TEXT,
            file_paths TEXT,
            total_tokens BIGINT,
            total_sentences INTEGER,
            size_mb REAL,
            created_at TEXT,
            source_type TEXT,
            is_public BOOLEAN DEFAULT 1
        )
    ''')
    
    # Table des analyses
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tokenizer_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            tokenizer_id TEXT NOT NULL,
            analysis_type TEXT,
            results TEXT,
            created_at TEXT,
            FOREIGN KEY (tokenizer_id) REFERENCES tokenizers (tokenizer_id)
        )
    ''')
    
    # Table des sessions de tokenization
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tokenization_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            tokenizer_id TEXT NOT NULL,
            input_text_hash TEXT,
            results TEXT,
            processing_time REAL,
            created_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

app = FastAPI(title="Universal Tokenizer & Corpus Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UniversalTokenizer:
    """Tokenizer universel supportant toutes les langues"""
    
    def __init__(self):
        self.tokenizers = {}
        self.language_models = {}
        self.load_default_tokenizers()
        self.setup_language_detection()
    
    def load_default_tokenizers(self):
        """Charge les tokenizers par défaut"""
        try:
            # Tokenizer multilingue par défaut 
            self.tokenizers["multilingual"] = AutoTokenizer.from_pretrained("xlm-roberta-base")
            
            # Tokenizers spécialisés
            specialized_tokenizers = {
                "english": "bert-base-uncased",
                "french": "camembert-base", 
                "german": "bert-base-german-cased",
                "chinese": "bert-base-chinese",
                "arabic": "aubmindlab/bert-base-arabertv2",
                "japanese": "cl-tohoku/bert-base-japanese",
                "korean": "klue/bert-base",
                "russian": "DeepPavlov/rubert-base-cased",
                "spanish": "dccuchile/bert-base-spanish-wwm-cased",
                "hindi": "google/muril-base-cased"
            }
            
            for lang, model_name in specialized_tokenizers.items():
                try:
                    self.tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
                except Exception as e:
                    print(f"Erreur chargement tokenizer {lang}: {e}")
                    
        except Exception as e:
            print(f"Erreur chargement tokenizers par défaut: {e}")
    
    def setup_language_detection(self):
        """Configure la détection de langue"""
        try:
            # Télécharger les modèles NLTK nécessaires  
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            
            # Modèle FastText pour détection de langue
            try:
                import fasttext
                # Note: En production, télécharger le modèle de détection de langue
                # fasttext.download_model('lid.176.bin')
            except:
                pass
                
        except Exception as e:
            print(f"Erreur setup détection langue: {e}")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Détecte la langue du texte"""
        try:
            # Utiliser langdetect comme méthode principale
            detected = langdetect.detect_langs(text)
            if detected:
                return detected[0].lang, detected[0].prob
            
            return "unknown", 0.0
            
        except Exception as e:
            # Fallback: tentative avec polyglot
            try:   
                import pycld2 as cld2
                detector  = cld2.detect(text)
                # from langdetect import detect
                # detector = detect(text)
                return detector.language.code, detector.language.confidence
            except:
                return "unknown", 0.0
    
    def select_best_tokenizer(self, text: str, requested_tokenizer: str = None) -> str:
        """Sélectionne le meilleur tokenizer selon le texte et la demande"""
        if requested_tokenizer and requested_tokenizer in self.tokenizers:
            return requested_tokenizer
        
        # Détecter la langue et sélectionner le tokenizer approprié
        lang_code, confidence = self.detect_language(text)
        
        # Mapping des codes de langue vers nos tokenizers
        lang_mapping = {
            "en": "english",
            "fr": "french", 
            "de": "german",
            "zh": "chinese",
            "ar": "arabic",
            "ja": "japanese",
            "ko": "korean",
            "ru": "russian",
            "es": "spanish",
            "hi": "hindi"
        }
        
        tokenizer_name = lang_mapping.get(lang_code, "multilingual")
        
        if tokenizer_name not in self.tokenizers:
            tokenizer_name = "multilingual"
            
        return tokenizer_name
    
    def tokenize(self, text: str, tokenizer_name: str = None, **options) -> Dict[str, Any]:
        """Tokenise le texte avec analyse complète"""
        start_time = datetime.now()
        
        # Sélectionner le tokenizer
        selected_tokenizer = self.select_best_tokenizer(text, tokenizer_name)
        tokenizer = self.tokenizers.get(selected_tokenizer, self.tokenizers["multilingual"])
        
        # Détecter la langue
        detected_lang, lang_confidence = self.detect_language(text)
        
        # Tokenisation
        result = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=True,
            truncation=False,
            padding=False
        )
        
        tokens = tokenizer.tokenize(text)
        
        # Analyse du texte
        analysis = self.analyze_text(text, tokens, detected_lang)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "input_text": text,
            "tokenizer_used": selected_tokenizer,
            "detected_language": detected_lang,
            "language_confidence": lang_confidence,
            "processing_time_seconds": processing_time
        }
        
        # Ajouter les résultats demandés
        if options.get("return_tokens", True):
            response["tokens"] = tokens
            response["token_count"] = len(tokens)
            
        if options.get("return_ids", True):
            response["input_ids"] = result["input_ids"]
            
        if options.get("return_attention_mask", False):
            response["attention_mask"] = result.get("attention_mask", [])
            
        if options.get("return_analysis", False):
            response["analysis"] = analysis
        
        return response
    
    def analyze_text(self, text: str, tokens: List[str], language: str) -> Dict[str, Any]:
        """Analyse linguistique complète du texte"""
        analysis = {
            "basic_stats": self._get_basic_stats(text, tokens),
            "linguistic_features": self._get_linguistic_features(text, language),
            "token_analysis": self._analyze_tokens(tokens),
            "complexity_metrics": self._calculate_complexity(text, language)
        }
        
        return analysis
    
    def _get_basic_stats(self, text: str, tokens: List[str]) -> Dict[str, Any]:
        """Statistiques de base"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "token_count": len(tokens),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "avg_tokens_per_word": len(tokens) / len(words) if words else 0,
            "compression_ratio": len(tokens) / len(text) if text else 0
        }
    
    def _get_linguistic_features(self, text: str, language: str) -> Dict[str, Any]:
        """Caractéristiques linguistiques"""
        features = {
            "language": language,
            "script_type": self._detect_script_type(text),
            "has_punctuation": bool(re.search(r'[^\w\s]', text)),
            "has_numbers": bool(re.search(r'\d', text)),
            "has_uppercase": bool(re.search(r'[A-Z]', text)),
            "has_lowercase": bool(re.search(r'[a-z]', text)),
        }
        
        # Analyse des caractères Unicode
        char_categories = defaultdict(int)
        for char in text:
            char_categories[unicodedata.category(char)] += 1
        
        features["unicode_categories"] = dict(char_categories)
        
        return features
    
    def _detect_script_type(self, text: str) -> str:
        """Détecte le type d'écriture"""
        scripts = {
            "latin": r'[a-zA-ZÀ-ÿ]',
            "cyrillic": r'[А-я]',
            "arabic": r'[\u0600-\u06FF]',
            "chinese": r'[\u4e00-\u9fff]',
            "japanese": r'[\u3040-\u309f\u30a0-\u30ff]',
            "korean": r'[\uac00-\ud7af]',
            "devanagari": r'[\u0900-\u097f]'
        }
        
        for script_name, pattern in scripts.items():
            if re.search(pattern, text):
                return script_name
        
        return "mixed"
    
    def _analyze_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """Analyse des tokens"""
        token_lengths = [len(token) for token in tokens]
        
        # Types de tokens
        token_types = {
            "subword": sum(1 for token in tokens if token.startswith("##") or token.startswith("▁")),
            "punctuation": sum(1 for token in tokens if not token.isalnum()),
            "numeric": sum(1 for token in tokens if token.isdigit()),
            "alphabetic": sum(1 for token in tokens if token.isalpha()),
            "mixed": sum(1 for token in tokens if any(c.isalpha() for c in token) and any(c.isdigit() for c in token))
        }
        
        # Fréquence des tokens
        token_freq = Counter(tokens)
        most_frequent = token_freq.most_common(10)
        
        return {
            "avg_token_length": np.mean(token_lengths) if token_lengths else 0,
            "token_length_std": np.std(token_lengths) if token_lengths else 0,
            "min_token_length": min(token_lengths) if token_lengths else 0,
            "max_token_length": max(token_lengths) if token_lengths else 0,
            "token_types": token_types,
            "most_frequent_tokens": most_frequent,
            "unique_tokens": len(set(tokens)),
            "repetition_rate": 1 - (len(set(tokens)) / len(tokens)) if tokens else 0
        }
    
    def _calculate_complexity(self, text: str, language: str) -> Dict[str, Any]:
        """Calcul de métriques de complexité"""
        try:
            # Utiliser textstat pour les métriques de lisibilité
            complexity = {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "lix": textstat.lix(text),
                "rix": textstat.rix(text)
            }
            
            # Métriques linguistiques
            words = text.split()
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            complexity.update({
                "lexical_diversity": len(set(words)) / len(words) if words else 0,
                "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
                "sentence_length_variance": np.var([len(s.split()) for s in sentences]) if sentences else 0
            })
            
            return complexity
            
        except Exception as e:
            return {"error": str(e)}

class CorpusManager:
    """Gestionnaire de corpus multilingue"""
    
    def __init__(self):
        self.corpus_collection = {}
        self.load_default_corpus()
    
    def load_default_corpus(self):
        """Charge les corpus par défaut"""
        try:
            # Créer des corpus d'exemple multilingues
            self.create_sample_corpus()
        except Exception as e:
            print(f"Erreur chargement corpus par défaut: {e}")
    
    def create_sample_corpus(self):
        """Crée des corpus d'exemple"""
        sample_texts = {
            "multilingual_news": {
                "en": [
                    "The quick brown fox jumps over the lazy dog.",
                    "Artificial intelligence is transforming the world.",
                    "Climate change requires immediate global action."
                ],
                "fr": [
                    "Le renard brun et rapide saute par-dessus le chien paresseux.",
                    "L'intelligence artificielle transforme le monde.",
                    "Le changement climatique nécessite une action mondiale immédiate."
                ],
                "es": [
                    "El zorro marrón rápido salta sobre el perro perezoso.",
                    "La inteligencia artificial está transformando el mundo.",
                    "El cambio climático requiere acción global inmediata."
                ]
            },
            "technical_corpus": {
                "en": [
                    "Machine learning algorithms require large datasets for training.",
                    "Neural networks consist of interconnected nodes called neurons.",
                    "Deep learning models can automatically extract features from raw data."
                ],
                "fr": [
                    "Les algorithmes d'apprentissage automatique nécessitent de grands ensembles de données.",
                    "Les réseaux de neurones se composent de nœuds interconnectés appelés neurones.",
                    "Les modèles d'apprentissage profond peuvent extraire automatiquement des caractéristiques."
                ]
            }
        }
        
        self.corpus_collection = sample_texts
    
    def analyze_corpus(self, corpus_name: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyse complète d'un corpus"""
        if corpus_name not in self.corpus_collection:
            raise ValueError(f"Corpus '{corpus_name}' non trouvé")
        
        corpus_data = self.corpus_collection[corpus_name]
        
        analysis = {
            "corpus_name": corpus_name,
            "analysis_type": analysis_type,
            "languages": list(corpus_data.keys()),
            "language_analysis": {}
        }
        
        total_stats = {
            "total_texts": 0,
            "total_words": 0,
            "total_characters": 0,
            "languages_count": len(corpus_data)
        }
        
        for lang, texts in corpus_data.items():
            lang_analysis = self._analyze_language_corpus(texts, lang)
            analysis["language_analysis"][lang] = lang_analysis
            
            total_stats["total_texts"] += lang_analysis["text_count"]
            total_stats["total_words"] += lang_analysis["total_words"]
            total_stats["total_characters"] += lang_analysis["total_characters"]
        
        analysis["global_statistics"] = total_stats
        
        if analysis_type == "comprehensive":
            analysis["cross_language_analysis"] = self._cross_language_analysis(corpus_data)
        
        return analysis
    
    def _analyze_language_corpus(self, texts: List[str], language: str) -> Dict[str, Any]:
        """Analyse un corpus pour une langue spécifique"""
        all_text = " ".join(texts)
        words = all_text.split()
        
        # Statistiques de base
        stats = {
            "text_count": len(texts),
            "total_words": len(words),
            "total_characters": len(all_text),
            "unique_words": len(set(words)),
            "avg_text_length": np.mean([len(text) for text in texts]),
            "vocabulary_richness": len(set(words)) / len(words) if words else 0
        }
        
        # Analyse des fréquences
        word_freq = Counter(words)
        stats["most_frequent_words"] = word_freq.most_common(20)
        
        # Longueur des mots
        word_lengths = [len(word) for word in words]
        stats["avg_word_length"] = np.mean(word_lengths) if word_lengths else 0
        stats["word_length_distribution"] = {
            "min": min(word_lengths) if word_lengths else 0,
            "max": max(word_lengths) if word_lengths else 0,
            "std": np.std(word_lengths) if word_lengths else 0
        }
        
        return stats
    
    def _cross_language_analysis(self, corpus_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyse comparative entre langues"""
        languages = list(corpus_data.keys())
        
        # Comparaison de vocabulaire
        vocab_sizes = {}
        avg_word_lengths = {}
        
        for lang, texts in corpus_data.items():
            all_words = " ".join(texts).split()
            vocab_sizes[lang] = len(set(all_words))
            avg_word_lengths[lang] = np.mean([len(word) for word in all_words]) if all_words else 0
        
        return {
            "vocabulary_sizes": vocab_sizes,
            "avg_word_lengths": avg_word_lengths,
            "language_similarity": self._calculate_language_similarity(corpus_data),
            "coverage_analysis": self._analyze_coverage(corpus_data)
        }
    
    def _calculate_language_similarity(self, corpus_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calcule la similarité entre langues (approximation basique)"""
        languages = list(corpus_data.keys())
        similarities = {}
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages[i+1:], i+1):
                # Similarité basée sur les caractères communs (très basique)
                text1 = "".join(corpus_data[lang1]).lower()
                text2 = "".join(corpus_data[lang2]).lower()
                
                chars1 = set(text1)
                chars2 = set(text2)
                
                similarity = len(chars1 & chars2) / len(chars1 | chars2) if (chars1 | chars2) else 0
                similarities[f"{lang1}-{lang2}"] = similarity
        
        return similarities
    
    def _analyze_coverage(self, corpus_data: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyse la couverture du corpus"""
        total_unique_words = set()
        
        for texts in corpus_data.values():
            words = " ".join(texts).split()
            total_unique_words.update(words)
        
        coverage = {
            "total_unique_vocabulary": len(total_unique_words),
            "estimated_language_coverage": min(len(total_unique_words) / 100000, 1.0),  # Estimation basique
            "domain_diversity": "mixed"  # Placeholder
        }
        
        return coverage

# Instances globales
universal_tokenizer = UniversalTokenizer()
corpus_manager = CorpusManager()

# Endpoints API
@app.on_event("startup")
async def startup_event():
    # Créer les dossiers nécessaires
    for directory in [TOKENIZERS_DIR, CORPUS_DIR, MODELS_DIR, ANALYSIS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialiser la base de données
    init_tokenizer_db()
    
    print("Universal Tokenizer & Corpus Platform démarré")

@app.post("/tokenize")
async def tokenize_text(request: TokenizeRequest):
    """Tokenise un texte avec analyse complète"""
    try:
        result = universal_tokenizer.tokenize(
            text=request.text,
            tokenizer_name=request.tokenizer_name,
            return_tokens=request.return_tokens,
            return_ids=request.return_ids,
            return_attention_mask=request.return_attention_mask,
            return_analysis=request.return_analysis
        )
        
        # Sauvegarder la session
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect('universal_tokenizer.db')
        cursor = conn.cursor()
        
        text_hash = str(hash(request.text))
        
        cursor.execute('''
            INSERT INTO tokenization_sessions (session_id, tokenizer_id, input_text_hash, results, processing_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id, result["tokenizer_used"], text_hash,
            json.dumps(result, default=str), result["processing_time_seconds"],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        result["session_id"] = session_id
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tokenizers/available")
async def get_available_tokenizers():
    """Récupère la liste des tokenizers disponibles"""
    try:
        tokenizers_info = {}
        
        for name, tokenizer in universal_tokenizer.tokenizers.items():
            try:
                vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else "unknown"
                tokenizers_info[name] = {
                    "name": name,
                    "type": type(tokenizer).__name__,
                    "vocab_size": vocab_size,
                    "model_max_length": getattr(tokenizer, 'model_max_length', 'unknown'),
                    "special_tokens": getattr(tokenizer, 'special_tokens_map', {})
                }
            except Exception as e:
                tokenizers_info[name] = {
                    "name": name,
                    "error": str(e)
                }
        
        return {"tokenizers": tokenizers_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/corpus/analyze")
async def analyze_corpus(request: CorpusAnalysisRequest):
    """Analyse un corpus"""
    try:
        analysis = corpus_manager.analyze_corpus(
            request.corpus_name,
            request.analysis_type
        )
        
        # Sauvegarder l'analyse
        analysis_id = str(uuid.uuid4())
        conn = sqlite3.connect('universal_tokenizer.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tokenizer_analyses (analysis_id, tokenizer_id, analysis_type, results, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            analysis_id, "corpus_analysis", request.analysis_type,
            json.dumps(analysis, default=str), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        analysis["analysis_id"] = analysis_id
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/corpus/available")
async def get_available_corpus():
    """Récupère la liste des corpus disponibles"""
    try:
        corpus_info = {}
        
        for name, data in corpus_manager.corpus_collection.items():
            total_texts = sum(len(texts) for texts in data.values())
            total_words = sum(len(" ".join(texts).split()) for texts in data.values())
            
            corpus_info[name] = {
                "name": name,
                "languages": list(data.keys()),
                "total_texts": total_texts,
                "total_words": total_words,
                "description": f"Corpus multilingue avec {len(data)} langues"
            }
        
        return {"corpus": corpus_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenizer/train")
async def train_custom_tokenizer(request: TokenizerTrainingRequest, background_tasks: BackgroundTasks):
    """Entraîne un tokenizer personnalisé"""
    try:
        training_id = str(uuid.uuid4())
        
        # Lancer l'entraînement en arrière-plan
        background_tasks.add_task(
            train_tokenizer_background,
            training_id,
            request.name,
            request.corpus_sources,
            request.config,
            request.user_id
        )
        
        return {
            "training_id": training_id,
            "status": "started",
            "message": "Entraînement du tokenizer démarré en arrière-plan"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def train_tokenizer_background(training_id: str, name: str, corpus_sources: List[str], config: TokenizerConfig, user_id: str):
    """Entraîne un tokenizer en arrière-plan"""
    try:
        # Préparer les données d'entraînement
        training_data = []
        for corpus_name in corpus_sources:
            if corpus_name in corpus_manager.corpus_collection:
                corpus_data = corpus_manager.corpus_collection[corpus_name]
                for texts in corpus_data.values():
                    training_data.extend(texts)
        
        if not training_data:
            raise Exception("Aucune donnée d'entraînement trouvée")
        
        # Créer le tokenizer selon le type
        if config.tokenizer_type == "bpe":
            tokenizer = Tokenizer(BPE())
            trainer = trainers.BpeTrainer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency,
                special_tokens=config.special_tokens
            )
        elif config.tokenizer_type == "wordpiece":
            tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
            trainer = trainers.WordPieceTrainer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency,
                special_tokens=config.special_tokens
            )
        elif config.tokenizer_type == "unigram":
            tokenizer = Tokenizer(Unigram())
            trainer = trainers.UnigramTrainer(
                vocab_size=config.vocab_size,
                special_tokens=config.special_tokens
            )
        else:  # wordlevel
            tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
            trainer = trainers.WordLevelTrainer(
                vocab_size=config.vocab_size,
                min_frequency=config.min_frequency,
                special_tokens=config.special_tokens
            )
        
        # Configuration de normalisation et pré-tokenisation
        if config.normalization:
            normalizers_list = [NFD()]
            if config.lowercase:
                normalizers_list.append(Lowercase())
            if config.strip_accents:
                normalizers_list.append(StripAccents())
            tokenizer.normalizer = normalizers.Sequence(normalizers_list)
        
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Whitespace(),
            Punctuation()
        ])
        
        # Entraînement
        tokenizer.train_from_iterator(training_data, trainer)
        
        # Sauvegarder le tokenizer
        tokenizer_path = os.path.join(TOKENIZERS_DIR, f"{name}_{training_id}.json")
        tokenizer.save(tokenizer_path)
        
        # Évaluation des performances
        performance_metrics = evaluate_tokenizer_performance(tokenizer, training_data[:1000])
        
        # Enregistrer en base de données
        conn = sqlite3.connect('universal_tokenizer.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tokenizers (tokenizer_id, name, tokenizer_type, vocab_size, languages, file_path, config_json, created_at, trained_by, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            training_id, name, config.tokenizer_type, config.vocab_size,
            json.dumps(config.languages), tokenizer_path,
            json.dumps(config.dict()), datetime.now().isoformat(),
            user_id, json.dumps(performance_metrics, default=str)
        ))
        
        conn.commit()
        conn.close()
        
        print(f"Tokenizer '{name}' entraîné avec succès")
        
    except Exception as e:
        print(f"Erreur entraînement tokenizer: {e}")
        # Marquer comme échoué en base
        conn = sqlite3.connect('universal_tokenizer.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE tokenizers SET status = 'failed' WHERE tokenizer_id = ?
        ''', (training_id,))
        conn.commit()
        conn.close()

def evaluate_tokenizer_performance(tokenizer, test_texts: List[str]) -> Dict[str, Any]:
    """Évalue les performances d'un tokenizer"""
    if not test_texts:
        return {"error": "Aucun texte de test"}
    
    total_chars = 0
    total_tokens = 0
    compression_ratios = []
    
    for text in test_texts[:100]:  # Limiter pour performance
        if not text.strip():
            continue
            
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        
        total_chars += len(text)
        total_tokens += len(tokens)
        
        if len(text) > 0:
            compression_ratios.append(len(tokens) / len(text))
    
    return {
        "avg_compression_ratio": np.mean(compression_ratios) if compression_ratios else 0,
        "total_chars_processed": total_chars,
        "total_tokens_generated": total_tokens,
        "efficiency_score": total_chars / total_tokens if total_tokens > 0 else 0,
        "vocab_utilization": "calculated_separately"  # Nécessiterait analyse plus poussée
    }

@app.get("/tokenizer/training/{training_id}/status")
async def get_training_status(training_id: str):
    """Récupère le statut d'entraînement d'un tokenizer"""
    try:
        conn = sqlite3.connect('universal_tokenizer.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, tokenizer_type, vocab_size, status, performance_metrics, created_at
            FROM tokenizers WHERE tokenizer_id = ?
        ''', (training_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            raise HTTPException(status_code=404, detail="Training non trouvé")
        
        return {
            "training_id": training_id,
            "name": result[0],
            "tokenizer_type": result[1],
            "vocab_size": result[2],
            "status": result[3],
            "performance_metrics": json.loads(result[4]) if result[4] else {},
            "created_at": result[5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze/language")
async def analyze_language_detection(text: str):
    """Analyse avancée de détection de langue"""
    try:
        # Utiliser plusieurs méthodes de détection
        lang_code, confidence = universal_tokenizer.detect_language(text)
        
        # Analyse détaillée des caractéristiques linguistiques
        linguistic_features = universal_tokenizer._get_linguistic_features(text, lang_code)
        
        # Suggestions de tokenizers appropriés
        suggested_tokenizer = universal_tokenizer.select_best_tokenizer(text)
        
        return {
            "text_sample": text[:100] + "..." if len(text) > 100 else text,
            "detected_language": lang_code,
            "confidence": confidence,
            "linguistic_features": linguistic_features,
            "suggested_tokenizer": suggested_tokenizer,
            "available_tokenizers": list(universal_tokenizer.tokenizers.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare/tokenizers")
async def compare_tokenizers(texts: List[str], tokenizer_names: List[str]):
    """Compare plusieurs tokenizers sur les mêmes textes"""
    try:
        if not texts or not tokenizer_names:
            raise HTTPException(status_code=400, detail="Textes et noms de tokenizers requis")
        
        comparison_results = []
        
        for text in texts[:10]:  # Limiter à 10 textes pour performance
            text_results = {"text": text[:50] + "..." if len(text) > 50 else text}
            
            for tokenizer_name in tokenizer_names:
                if tokenizer_name in universal_tokenizer.tokenizers:
                    result = universal_tokenizer.tokenize(
                        text, 
                        tokenizer_name, 
                        return_analysis=True
                    )
                    
                    text_results[tokenizer_name] = {
                        "token_count": result["token_count"],
                        "compression_ratio": result["analysis"]["basic_stats"]["compression_ratio"],
                        "processing_time": result["processing_time_seconds"]
                    }
            
            comparison_results.append(text_results)
        
        # Calcul des statistiques comparatives
        summary_stats = {}
        for tokenizer_name in tokenizer_names:
            if tokenizer_name in universal_tokenizer.tokenizers:
                token_counts = [r[tokenizer_name]["token_count"] for r in comparison_results if tokenizer_name in r]
                compression_ratios = [r[tokenizer_name]["compression_ratio"] for r in comparison_results if tokenizer_name in r]
                processing_times = [r[tokenizer_name]["processing_time"] for r in comparison_results if tokenizer_name in r]
                
                summary_stats[tokenizer_name] = {
                    "avg_token_count": np.mean(token_counts) if token_counts else 0,
                    "avg_compression_ratio": np.mean(compression_ratios) if compression_ratios else 0,
                    "avg_processing_time": np.mean(processing_times) if processing_times else 0,
                    "consistency": np.std(compression_ratios) if compression_ratios else 0
                }
        
        return {
            "detailed_results": comparison_results,
            "summary_statistics": summary_stats,
            "recommendation": _recommend_best_tokenizer(summary_stats)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _recommend_best_tokenizer(stats: Dict[str, Dict]) -> Dict[str, Any]:
    """Recommande le meilleur tokenizer basé sur les statistiques"""
    if not stats:
        return {"recommendation": "Aucune donnée pour recommandation"}
    
    # Critères de sélection (pondérés)
    scores = {}
    
    for tokenizer, metrics in stats.items():
        score = 0
        
        # Efficacité de compression (plus bas = mieux)
        compression = metrics.get("avg_compression_ratio", 1)
        score += (1 - compression) * 40  # 40% du score
        
        # Rapidité (plus bas = mieux)  
        speed = metrics.get("avg_processing_time", 1)
        max_speed = max(m.get("avg_processing_time", 1) for m in stats.values())
        if max_speed > 0:
            score += (1 - speed/max_speed) * 30  # 30% du score
        
        # Consistance (plus bas = mieux)
        consistency = metrics.get("consistency", 1)
        score += max(0, 1 - consistency) * 30  # 30% du score
        
        scores[tokenizer] = score
    
    best_tokenizer = max(scores.items(), key=lambda x: x[1])
    
    return {
        "recommended_tokenizer": best_tokenizer[0],
        "confidence_score": best_tokenizer[1],
        "reasoning": "Basé sur efficacité de compression, vitesse et consistance",
        "all_scores": scores
    }

def fetch_platform_statistics():
    conn = sqlite3.connect('universal_tokenizer.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM tokenization_sessions')
    total_tokenizations = cursor.fetchone()[0]

    cursor.execute('''
        SELECT tokenizer_id, COUNT(*) as usage_count
        FROM tokenization_sessions
        GROUP BY tokenizer_id
        ORDER BY usage_count DESC
        LIMIT 10
    ''')
    popular_tokenizers = cursor.fetchall()

    cursor.execute('SELECT COUNT(*) FROM tokenizer_analyses WHERE analysis_type LIKE "%corpus%"')
    corpus_analyses = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM tokenizers WHERE trained_by != "system"')
    custom_tokenizers = cursor.fetchone()[0]

    conn.close()

    return {
        "platform_stats": {
            "total_tokenizations": total_tokenizations,
            "corpus_analyses": corpus_analyses,
            "custom_tokenizers_trained": custom_tokenizers,
            "available_tokenizers": len(universal_tokenizer.tokenizers),
            "available_corpus": len(corpus_manager.corpus_collection)
        },
        "usage_patterns": {
            "most_used_tokenizers": [
                {"tokenizer": t[0], "usage": t[1]} for t in popular_tokenizers
            ]
        }
    }

def fetch_platform_statistics():
    conn = sqlite3.connect('universal_tokenizer.db')
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM tokenization_sessions')
    total_tokenizations = cursor.fetchone()[0]

    cursor.execute('''
        SELECT tokenizer_id, COUNT(*) as usage_count
        FROM tokenization_sessions
        GROUP BY tokenizer_id
        ORDER BY usage_count DESC
        LIMIT 10
    ''')
    popular_tokenizers = cursor.fetchall()

    cursor.execute('SELECT COUNT(*) FROM tokenizer_analyses WHERE analysis_type LIKE "%corpus%"')
    corpus_analyses = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM tokenizers WHERE trained_by != "system"')
    custom_tokenizers = cursor.fetchone()[0]

    conn.close()

    return {
        "platform_stats": {
            "total_tokenizations": total_tokenizations,
            "corpus_analyses": corpus_analyses,
            "custom_tokenizers_trained": custom_tokenizers,
            "available_tokenizers": len(universal_tokenizer.tokenizers),
            "available_corpus": len(corpus_manager.corpus_collection)
        },
        "usage_patterns": {
            "most_used_tokenizers": [
                {"tokenizer": t[0], "usage": t[1]} for t in popular_tokenizers
            ]
        }
    }

@app.get("/statistics/usage")
async def get_platform_statistics():
    """Statistiques d'utilisation de la plateforme"""
    try:
        conn = sqlite3.connect('universal_tokenizer.db')
        cursor = conn.cursor()
        
        # Statistiques de tokenisation
        cursor.execute('SELECT COUNT(*) FROM tokenization_sessions')
        total_tokenizations = cursor.fetchone()[0]
        
        # Tokenizers les plus utilisés
        cursor.execute('''
            SELECT tokenizer_id, COUNT(*) as usage_count
            FROM tokenization_sessions
            GROUP BY tokenizer_id
            ORDER BY usage_count DESC
            LIMIT 10
        ''')
        popular_tokenizers = cursor.fetchall()
        
        # Statistiques de corpus
        cursor.execute('SELECT COUNT(*) FROM tokenizer_analyses WHERE analysis_type LIKE "%corpus%"')
        corpus_analyses = cursor.fetchone()[0]
        
        # Tokenizers entraînés
        cursor.execute('SELECT COUNT(*) FROM tokenizers WHERE trained_by != "system"')
        custom_tokenizers = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "platform_stats": {
                "total_tokenizations": total_tokenizations,
                "corpus_analyses": corpus_analyses,
                "custom_tokenizers_trained": custom_tokenizers,
                "available_tokenizers": len(universal_tokenizer.tokenizers),
                "available_corpus": len(corpus_manager.corpus_collection)
            },
            "usage_patterns": {
                "most_used_tokenizers": [{"tokenizer": t[0], "usage": t[1]} for t in popular_tokenizers]
            },
            "system_info": {
                "supported_languages": ["en", "fr", "de", "es", "zh", "ja", "ko", "ar", "hi", "ru", "multi"],
                "tokenizer_types": ["BPE", "WordPiece", "Unigram", "WordLevel"],
                "max_vocab_size": 100000,
                "supported_scripts": ["latin", "cyrillic", "arabic", "chinese", "japanese", "korean", "devanagari"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return fetch_platform_statistics()

@app.post("/corpus/upload")
async def upload_corpus(
    file: UploadFile = File(...),
    corpus_name: str = "uploaded_corpus",
    language: str = "auto",
    user_id: str = "anonymous"
):
    """Upload d'un corpus personnalisé"""
    try:
        # Lire le contenu du fichier
        content = await file.read()
        
        if file.filename.endswith('.txt'):
            text_content = content.decode('utf-8')
        elif file.filename.endswith('.json'):
            json_content = json.loads(content.decode('utf-8'))
            if isinstance(json_content, list):
                text_content = '\n'.join(json_content)
            else:
                text_content = str(json_content)
        else:
            raise HTTPException(status_code=400, detail="Format de fichier non supporté")
        
        # Diviser en textes
        texts = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        if not texts:
            raise HTTPException(status_code=400, detail="Aucun texte trouvé dans le fichier")
        
        # Détecter la langue si auto
        if language == "auto":
            sample_text = ' '.join(texts[:10])  # Échantillon
            language, _ = universal_tokenizer.detect_language(sample_text)
        
        # Créer l'entrée corpus
        corpus_id = str(uuid.uuid4())
        corpus_data = {language: texts}
        
        # Analyser le corpus
        analysis = corpus_manager._analyze_language_corpus(texts, language)
        
        # Sauvegarder le fichier
        corpus_file_path = os.path.join(CORPUS_DIR, f"{corpus_name}_{corpus_id}.json")
        with open(corpus_file_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        # Enregistrer en base de données
        conn = sqlite3.connect('universal_tokenizer.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO corpus_collection (corpus_id, name, description, languages, file_paths, total_tokens, total_sentences, size_mb, created_at, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            corpus_id, corpus_name, f"Corpus uploadé par {user_id}",
            json.dumps([language]), json.dumps([corpus_file_path]),
            analysis["total_words"], len(texts), len(content) / (1024*1024),
            datetime.now().isoformat(), "user_upload"
        ))
        
        conn.commit()
        conn.close()
        
        # Ajouter au gestionnaire de corpus
        corpus_manager.corpus_collection[corpus_name] = corpus_data
        
        return {
            "corpus_id": corpus_id,
            "name": corpus_name,
            "language": language,
            "stats": analysis,
            "message": "Corpus uploadé et analysé avec succès"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Vérification de santé de la plateforme"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "tokenizers_loaded": len(universal_tokenizer.tokenizers),
        "corpus_available": len(corpus_manager.corpus_collection),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8046)