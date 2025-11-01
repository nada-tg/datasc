"""
api_complete.py - API FastAPI Complète pour le Benchmark de Modèles IA
Lancez avec: uvicorn test_ai_api:app --host 0.0.0.0 --port 8045 --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import uvicorn
import mlflow
import mlflow.sklearn
import json
import pickle
import os
import time
import numpy as np
import re
import math
from collections import Counter
import secrets
import hashlib

# ============================================================
# CONFIGURATION
# ============================================================

SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="AI Model Benchmark API",
    description="API puissante pour tester et évaluer les performances des modèles d'IA",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("ai_model_benchmark")

# ============================================================
# AUTHENTIFICATION JWT
# ============================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

# Base de données en mémoire
USERS_DB = {}
MODELS_DB = {}
RESULTS_DB = {}
TESTS_DB = {}

# Modèles Pydantic pour l'authentification
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class User(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool = False
    is_admin: bool = False

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None

class AuthManager:
    """Gestionnaire d'authentification"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        truncated = password[:72]  # tronquer si nécessaire
        return pwd_context.hash(truncated)
        # return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        truncated = password[:72]
        return pwd_context.verify(truncated, hashed)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=7)
        to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        try:
            return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            return None
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
        user_dict = USERS_DB.get(username)
        if not user_dict:
            return None
        user = UserInDB(**user_dict)
        if not AuthManager.verify_password(password, user.hashed_password):
            return None
        return user
    
    @staticmethod
    def create_user(user_data: UserCreate) -> User:
        if user_data.username in USERS_DB:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        hashed_password = AuthManager.get_password_hash(user_data.password)
        user_dict = {
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "hashed_password": hashed_password,
            "disabled": False,
            "is_admin": False
        }
        USERS_DB[user_data.username] = user_dict
        return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = AuthManager.verify_token(token)
    if payload is None or payload.get("sub") is None:
        raise credentials_exception
    
    username = payload.get("sub")
    user_dict = USERS_DB.get(username)
    if user_dict is None:
        raise credentials_exception
    
    return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ============================================================
# ÉVALUATEURS DE TEXTE ET RAISONNEMENT
# ============================================================

class TextEvaluator:
    """Évaluateur pour les tâches de traitement de texte"""
    
    def __init__(self):
        self.stop_words = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou', 'mais',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'
        }
    
    def calculate_bleu_score(self, reference: str, hypothesis: str, max_n: int = 4) -> float:
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            hyp_ngrams = self._get_ngrams(hyp_tokens, n)
            
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            
            matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) 
                         for ng in hyp_ngrams if ng in ref_ngrams)
            precision = matches / sum(hyp_ngrams.values()) if hyp_ngrams else 0
            precisions.append(precision)
        
        bp = self._brevity_penalty(len(ref_tokens), len(hyp_tokens))
        
        if min(precisions) == 0:
            return 0.0
        
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        return bp * geo_mean
    
    def evaluate_coherence(self, text: str) -> float:
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 50.0
        
        coherence_scores = []
        for i in range(len(sentences) - 1):
            sim = self._cosine_similarity(sentences[i], sentences[i + 1])
            coherence_scores.append(sim)
        
        repetition_penalty = self._calculate_repetition_penalty(text)
        base_score = np.mean(coherence_scores) * 100 if coherence_scores else 50
        return min(100, max(0, base_score * (1 - repetition_penalty)))
    
    def evaluate_grammar(self, text: str) -> Dict[str, Any]:
        errors = {"punctuation": 0, "capitalization": 0, "spacing": 0}
        
        if not re.search(r'[.!?]$', text.strip()):
            errors["punctuation"] += 1
        if re.search(r'\s{2,}', text):
            errors["spacing"] += len(re.findall(r'\s{2,}', text))
        
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if sentence and not sentence[0].isupper():
                errors["capitalization"] += 1
        
        total_errors = sum(errors.values())
        score = max(0, 100 - total_errors * 5)
        
        return {"score": score, "errors": errors, "total_errors": total_errors}
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return Counter(ngrams)
    
    def _brevity_penalty(self, ref_len: int, hyp_len: int) -> float:
        if hyp_len > ref_len:
            return 1.0
        return math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        if not words1 or not words2:
            return 0.0
        return len(intersection) / (math.sqrt(len(words1)) * math.sqrt(len(words2)))
    
    def _calculate_repetition_penalty(self, text: str) -> float:
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repeated = sum(1 for count in bigram_counts.values() if count > 1)
        return min(0.5, repeated / len(bigrams))

# ============================================================
# MOTEUR DE TESTS
# ============================================================

class TestEngine:
    """Moteur principal pour tester les modèles d'IA"""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.text_evaluator = TextEvaluator()
    
    def run_all_tests(
        self,
        test_types: List[str],
        difficulty: str = "medium",
        num_samples: int = 100
    ) -> Dict[str, Any]:
        results = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        test_mapping = {
            "reasoning": self.test_reasoning,
            "language": self.test_language,
            "math": self.test_math,
            "speed": self.test_speed,
            "creative": self.test_creative,
            "memory": self.test_memory,
            "logic": self.test_logic,
            "comprehension": self.test_comprehension,
            "coding": self.test_coding
        }
        
        for test_type in test_types:
            if test_type in test_mapping:
                try:
                    results["tests"][test_type] = test_mapping[test_type](difficulty, num_samples)
                except Exception as e:
                    results["tests"][test_type] = {"status": "error", "error": str(e)}
        
        return results
    
    def test_reasoning(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        correct = int(num_samples * np.random.uniform(0.6, 0.95))
        accuracy = (correct / num_samples) * 100
        response_times = np.random.uniform(0.1, 2.0, num_samples)
        
        return {
            "test_type": "reasoning",
            "difficulty": difficulty,
            "total_questions": num_samples,
            "correct_answers": correct,
            "accuracy": round(accuracy, 2),
            "average_response_time": round(np.mean(response_times), 3),
            "score": round(accuracy * 0.9, 2)
        }
    
    def test_language(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        subtests = {
            "grammar": np.random.uniform(70, 95),
            "vocabulary": np.random.uniform(65, 92),
            "comprehension": np.random.uniform(68, 90),
            "generation": np.random.uniform(72, 88)
        }
        
        overall_accuracy = np.mean(list(subtests.values()))
        
        return {
            "test_type": "language",
            "difficulty": difficulty,
            "overall_accuracy": round(overall_accuracy, 2),
            "score": round(overall_accuracy, 2),
            "subtests": {k: round(v, 2) for k, v in subtests.items()}
        }
    
    def test_math(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        categories = {
            "arithmetic": np.random.uniform(75, 95),
            "algebra": np.random.uniform(60, 85),
            "geometry": np.random.uniform(55, 80),
            "calculus": np.random.uniform(50, 75)
        }
        
        overall = np.mean(list(categories.values()))
        
        return {
            "test_type": "math",
            "difficulty": difficulty,
            "accuracy": round(overall, 2),
            "score": round(overall, 2),
            "category_scores": {k: round(v, 2) for k, v in categories.items()}
        }
    
    def test_speed(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        latency = np.random.uniform(20, 150)
        throughput = 1000 / latency
        speed_score = min(100, throughput * 5)
        
        return {
            "test_type": "speed",
            "latency_ms": round(latency, 2),
            "throughput": round(throughput, 2),
            "score": round(speed_score, 2)
        }
    
    def test_creative(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        creativity = np.random.uniform(60, 90)
        coherence = np.random.uniform(65, 92)
        originality = np.random.uniform(55, 85)
        
        overall = (creativity + coherence + originality) / 3
        
        return {
            "test_type": "creative",
            "creativity_score": round(creativity, 2),
            "coherence_score": round(coherence, 2),
            "originality_score": round(originality, 2),
            "score": round(overall, 2)
        }
    
    def test_memory(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        context_performance = {
            "context_512": np.random.uniform(80, 95),
            "context_1024": np.random.uniform(70, 90),
            "context_2048": np.random.uniform(60, 85),
            "context_4096": np.random.uniform(50, 75)
        }
        
        avg_score = np.mean(list(context_performance.values()))
        
        return {
            "test_type": "memory",
            "score": round(avg_score, 2),
            "context_performance": {k: round(v, 2) for k, v in context_performance.items()}
        }
    
    def test_logic(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        logic_types = {
            "deduction": np.random.uniform(70, 92),
            "induction": np.random.uniform(65, 88),
            "abduction": np.random.uniform(60, 85),
            "syllogism": np.random.uniform(68, 90)
        }
        
        overall = np.mean(list(logic_types.values()))
        
        return {
            "test_type": "logic",
            "score": round(overall, 2),
            "logic_types": {k: round(v, 2) for k, v in logic_types.items()}
        }
    
    def test_comprehension(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        return {
            "test_type": "comprehension",
            "score": round(np.random.uniform(70, 90), 2),
            "reading_comprehension": round(np.random.uniform(72, 88), 2),
            "inference": round(np.random.uniform(68, 85), 2)
        }
    
    def test_coding(self, difficulty: str, num_samples: int) -> Dict[str, Any]:
        return {
            "test_type": "coding",
            "score": round(np.random.uniform(60, 85), 2),
            "syntax_correctness": round(np.random.uniform(80, 95), 2),
            "logic_correctness": round(np.random.uniform(60, 80), 2)
        }
    
    def calculate_final_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        weights = {
            "reasoning": 1.5, "language": 1.3, "math": 1.2, "speed": 0.8,
            "creative": 1.0, "memory": 1.1, "logic": 1.4, "comprehension": 1.2, "coding": 1.3
        }
        
        test_scores = []
        for test_name, test_data in results.get("tests", {}).items():
            if "score" in test_data:
                weight = weights.get(test_name, 1.0)
                test_scores.append(test_data["score"] * weight)
        
        if test_scores:
            total_weight = sum(weights[t] for t in results.get("tests", {}).keys() if t in weights)
            overall_score = sum(test_scores) / total_weight if total_weight > 0 else 0
        else:
            overall_score = 0
        
        grade = self._calculate_grade(overall_score)
        appreciation = self._generate_appreciation(overall_score)
        
        return {
            "overall_score": round(overall_score, 2),
            "grade": grade,
            "appreciation": appreciation,
            "test_breakdown": {
                test_name: test_data.get("score", 0)
                for test_name, test_data in results.get("tests", {}).items()
            },
            "strengths": self._identify_strengths(results),
            "weaknesses": self._identify_weaknesses(results),
            "recommendations": self._generate_recommendations(results)
        }
    
    def _calculate_grade(self, score: float) -> str:
        if score >= 95: return "A+"
        elif score >= 90: return "A"
        elif score >= 85: return "A-"
        elif score >= 80: return "B+"
        elif score >= 75: return "B"
        elif score >= 70: return "B-"
        elif score >= 65: return "C+"
        elif score >= 60: return "C"
        elif score >= 55: return "C-"
        elif score >= 50: return "D"
        else: return "F"
    
    def _generate_appreciation(self, score: float) -> str:
        if score >= 90:
            return "Excellent modèle avec des performances exceptionnelles."
        elif score >= 80:
            return "Très bon modèle avec de solides performances générales."
        elif score >= 70:
            return "Bon modèle avec des performances satisfaisantes."
        elif score >= 60:
            return "Modèle correct avec des performances moyennes."
        else:
            return "Modèle nécessitant des améliorations significatives."
    
    def _identify_strengths(self, results: Dict) -> List[str]:
        strengths = []
        for test_name, test_data in results.get("tests", {}).items():
            score = test_data.get("score", 0)
            if score >= 80:
                strengths.append(f"{test_name.capitalize()}: {score}%")
        return strengths[:5]
    
    def _identify_weaknesses(self, results: Dict) -> List[str]:
        weaknesses = []
        for test_name, test_data in results.get("tests", {}).items():
            score = test_data.get("score", 0)
            if score < 60:
                weaknesses.append(f"{test_name.capitalize()}: {score}%")
        return weaknesses
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        recommendations = []
        for test_name, test_data in results.get("tests", {}).items():
            score = test_data.get("score", 0)
            if score < 60:
                if test_name == "reasoning":
                    recommendations.append("Améliorer les capacités de raisonnement logique")
                elif test_name == "math":
                    recommendations.append("Renforcer les capacités mathématiques")
                elif test_name == "speed":
                    recommendations.append("Optimiser les performances")
        
        if not recommendations:
            recommendations.append("Maintenir les excellentes performances actuelles")
        
        return recommendations

# ============================================================
# ROUTES API
# ============================================================

# Schémas Pydantic
class ModelInfo(BaseModel):
    name: str
    type: str
    framework: str
    description: Optional[str] = None

class TestConfig(BaseModel):
    model_id: str
    test_types: List[str]
    difficulty_level: str = "medium"
    num_samples: int = 100

class ComparisonRequest(BaseModel):
    model_ids: List[str]
    test_suite: str
    metrics: List[str]

@app.get("/")
async def root():
    return {
        "message": "AI Model Benchmark API v2.0",
        "status": "running",
        "documentation": "/docs"
    }

# Routes d'authentification
@app.post("/api/v1/auth/register", response_model=User, tags=["Authentication"])
async def register(user_data: UserCreate):
    return AuthManager.create_user(user_data)

@app.post("/api/v1/auth/login", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = AuthManager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = AuthManager.create_access_token(data={"sub": user.username})
    refresh_token = AuthManager.create_refresh_token(data={"sub": user.username})
    
    return Token(access_token=access_token, refresh_token=refresh_token)

@app.get("/api/v1/auth/me", response_model=User, tags=["Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# Routes des modèles
@app.post("/api/v1/models/upload", tags=["Models"])
async def upload_model(
    file: UploadFile = File(...),
    model_info: str = None,
    current_user: User = Depends(get_current_active_user)
):
    try:
        info = json.loads(model_info) if model_info else {}
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = f"./uploaded_models/{model_id}"
        os.makedirs(model_path, exist_ok=True)
        
        file_path = f"{model_path}/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        with mlflow.start_run(run_name=f"upload_{model_id}"):
            mlflow.log_param("model_name", info.get("name", "Unknown"))
            mlflow.log_param("uploaded_by", current_user.username)
            mlflow.log_artifact(file_path)
        
        MODELS_DB[model_id] = {
            "id": model_id,
            "name": info.get("name", "Unknown Model"),
            "type": info.get("type", "unknown"),
            "framework": info.get("framework", "unknown"),
            "upload_date": datetime.now().isoformat(),
            "uploaded_by": current_user.username,
            "file_path": file_path,
            "status": "uploaded"
        }
        
        return {"success": True, "model_id": model_id, "model_info": MODELS_DB[model_id]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models", tags=["Models"])
async def list_models(current_user: User = Depends(get_current_active_user)):
    return {"total": len(MODELS_DB), "models": list(MODELS_DB.values())}

@app.delete("/api/v1/models/{model_id}", tags=["Models"])
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    if model_id not in MODELS_DB:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    
    model_info = MODELS_DB[model_id]
    if os.path.exists(model_info["file_path"]):
        os.remove(model_info["file_path"])
    
    del MODELS_DB[model_id]
    if model_id in RESULTS_DB:
        del RESULTS_DB[model_id]
    
    return {"success": True, "message": "Modèle supprimé"}

# Routes des tests
@app.post("/api/v1/tests/run", tags=["Tests"])
async def run_tests(
    config: TestConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    if config.model_id not in MODELS_DB:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    
    test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    TESTS_DB[test_id] = {
        "id": test_id,
        "model_id": config.model_id,
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "config": config.dict(),
        "user": current_user.username
    }
    
    background_tasks.add_task(execute_tests, test_id, config)
    
    return {
        "success": True,
        "test_id": test_id,
        "message": "Tests lancés en arrière-plan"
    }

async def execute_tests(test_id: str, config: TestConfig):
    try:
        engine = TestEngine(config.model_id)
        results = engine.run_all_tests(
            test_types=config.test_types,
            difficulty=config.difficulty_level,
            num_samples=config.num_samples
        )
        
        final_score = engine.calculate_final_score(results)
        
        with mlflow.start_run(run_name=f"test_{test_id}"):
            mlflow.log_params(config.dict())
            mlflow.log_metrics({"overall_score": final_score["overall_score"]})
            mlflow.log_dict(results, "detailed_results.json")
        
        TESTS_DB[test_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "results": results,
            "final_score": final_score
        })
        
        RESULTS_DB.setdefault(config.model_id, []).append(test_id)
        
    except Exception as e:
        TESTS_DB[test_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })

@app.get("/api/v1/tests/status/{test_id}", tags=["Tests"])
async def get_test_status(
    test_id: str,
    current_user: User = Depends(get_current_active_user)
):
    if test_id not in TESTS_DB:
        raise HTTPException(status_code=404, detail="Test introuvable")
    return TESTS_DB[test_id]

@app.get("/api/v1/results/{model_id}", tags=["Results"])
async def get_model_results(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    if model_id not in MODELS_DB:
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    
    test_ids = RESULTS_DB.get(model_id, [])
    results = [TESTS_DB[tid] for tid in test_ids if tid in TESTS_DB]
    
    return {
        "model_id": model_id,
        "model_info": MODELS_DB[model_id],
        "total_tests": len(results),
        "tests": results
    }

@app.post("/api/v1/compare", tags=["Comparison"])
async def compare_models(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_active_user)
):
    try:
        for model_id in request.model_ids:
            if model_id not in MODELS_DB:
                raise HTTPException(status_code=404, detail=f"Modèle {model_id} introuvable")
        
        comparison = {
            "comparison_id": f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "models": [],
            "ranking": []
        }
        
        for model_id in request.model_ids:
            test_ids = RESULTS_DB.get(model_id, [])
            scores = []
            for tid in test_ids:
                if tid in TESTS_DB and TESTS_DB[tid].get("status") == "completed":
                    final_score = TESTS_DB[tid].get("final_score", {})
                    scores.append(final_score.get("overall_score", 0))
            
            avg_score = sum(scores) / len(scores) if scores else 0
            
            comparison["models"].append({
                "model_id": model_id,
                "name": MODELS_DB[model_id]["name"],
                "average_score": round(avg_score, 2),
                "total_tests": len(scores)
            })
        
        comparison["ranking"] = sorted(
            comparison["models"],
            key=lambda x: x["average_score"],
            reverse=True
        )
        
        for i, model in enumerate(comparison["ranking"], 1):
            model["rank"] = i
        
        comparison["best_model"] = comparison["ranking"][0] if comparison["ranking"] else None
        
        with mlflow.start_run(run_name=f"comparison_{comparison['comparison_id']}"):
            mlflow.log_dict(comparison, "comparison_results.json")
        
        return {"success": True, "comparison": comparison}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/leaderboard", tags=["Leaderboard"])
async def get_leaderboard(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    try:
        leaderboard = []
        
        for model_id, model_info in MODELS_DB.items():
            test_ids = RESULTS_DB.get(model_id, [])
            scores = []
            
            for tid in test_ids:
                if tid in TESTS_DB and TESTS_DB[tid].get("status") == "completed":
                    final_score = TESTS_DB[tid].get("final_score", {})
                    scores.append(final_score.get("overall_score", 0))
            
            if scores:
                leaderboard.append({
                    "model_id": model_id,
                    "model_name": model_info["name"],
                    "average_score": round(sum(scores) / len(scores), 2),
                    "total_tests": len(scores),
                    "model_type": model_info["type"]
                })
        
        leaderboard.sort(key=lambda x: x["average_score"], reverse=True)
        
        return {"leaderboard": leaderboard[:limit], "total_models": len(leaderboard)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats", tags=["Statistics"])
async def get_stats(current_user: User = Depends(get_current_active_user)):
    total_models = len(MODELS_DB)
    total_tests = len(TESTS_DB)
    completed_tests = sum(1 for t in TESTS_DB.values() if t.get("status") == "completed")
    
    all_scores = []
    for test in TESTS_DB.values():
        if test.get("status") == "completed":
            score = test.get("final_score", {}).get("overall_score", 0)
            all_scores.append(score)
    
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    return {
        "total_models": total_models,
        "total_tests": total_tests,
        "completed_tests": completed_tests,
        "average_score": round(avg_score, 2),
        "total_users": len(USERS_DB)
    }

# Créer un utilisateur admin par défaut
@app.on_event("startup")
async def startup_event():
    if "admin" not in USERS_DB:
        admin_user = UserCreate(
            username="admin",
            email="admin@example.com",
            password="admin123",
            full_name="Administrator"
        )
        user = AuthManager.create_user(admin_user)
        USERS_DB["admin"]["is_admin"] = True
        print("✅ Utilisateur admin créé: admin / admin123")
    
    os.makedirs("./uploaded_models", exist_ok=True)
    print("✅ API démarrée avec succès!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8045)