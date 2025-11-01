"""
🤖 Advanced AI Decision Intelligence Platform - API FastAPI
Architecture IA • Décisions • Biais • Hallucinations • Explainabilité

Installation:
pip install fastapi uvicorn sqlalchemy pydantic python-jose passlib scikit-learn transformers torch

Lancement:
uvicorn ai_decision_platform_api:app --reload --host 0.0.0.0 --port 8048

Documentation: http://localhost:8030/docs
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
from enum import Enum
import uuid
import json

# ==================== CONFIGURATION ====================

app = FastAPI(
    title="🤖 AI Decision Intelligence API",
    description="API complète pour analyse décisions IA, biais et hallucinations",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité JWT
SECRET_KEY = "your_secret_key_change_in_production_ai_platform"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Base de données simulée
fake_db = {
    "users": {},
    "models": {},
    "decisions": [],
    "bias_tests": [],
    "hallucinations": [],
    "explainability": [],
    "datasets": {},
    "training_logs": [],
    "evaluations": []
}

# ==================== ENUMS ====================

class ModelType(str, Enum):
    TRANSFORMER = "Transformer (GPT, BERT)"
    CNN = "CNN (Vision)"
    RNN_LSTM = "RNN/LSTM (Séquences)"
    DECISION_TREE = "Decision Tree"
    RANDOM_FOREST = "Random Forest"
    NEURAL_NET = "Neural Network"
    REINFORCEMENT = "Reinforcement Learning"

class TaskType(str, Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Régression"
    GENERATION = "Génération Texte"
    TRANSLATION = "Traduction"
    QA = "Question-Answering"
    SUMMARIZATION = "Résumé"

class BiasType(str, Enum):
    SELECTION = "Biais de Sélection"
    CONFIRMATION = "Biais de Confirmation"
    SAMPLING = "Biais d'Échantillonnage"
    ALGORITHMIC = "Biais Algorithmique"
    HISTORICAL = "Biais Historique"
    DEMOGRAPHIC = "Biais Démographique"

class HallucinationType(str, Enum):
    FACTUAL = "Hallucination Factuelle"
    LOGICAL = "Hallucination Logique"
    CONTEXT = "Hallucination Contextuelle"
    TEMPORAL = "Hallucination Temporelle"

# ==================== MODELS PYDANTIC ====================

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class AIModelCreate(BaseModel):
    name: str = Field(..., example="GPT-Analyzer-1")
    model_type: ModelType
    task_type: TaskType
    parameters_millions: float = Field(..., gt=0, example=1300)
    training_data_size_gb: float = Field(..., gt=0, example=100)
    architecture_layers: int = Field(..., gt=0, example=24)
    hidden_size: int = Field(..., gt=0, example=1024)
    attention_heads: int = Field(..., gt=0, example=16)
    context_window: int = Field(..., gt=0, example=2048)

class AIModelResponse(BaseModel):
    id: str
    name: str
    model_type: str
    task_type: str
    parameters_millions: float
    complexity_score: float
    estimated_inference_ms: float
    memory_gb: float
    created_at: datetime

class DecisionRequest(BaseModel):
    model_id: str
    input_data: str
    context: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    max_tokens: int = Field(default=256, gt=0)

class DecisionResponse(BaseModel):
    decision_id: str
    model_id: str
    output: str
    confidence: float
    reasoning_steps: List[str]
    attention_weights: Dict[str, float]
    processing_time_ms: float
    timestamp: datetime

class BiasTestCreate(BaseModel):
    model_id: str
    bias_type: BiasType
    test_dataset: str
    demographic_groups: List[str]
    metrics_to_check: List[str]

class BiasTestResult(BaseModel):
    test_id: str
    model_id: str
    bias_type: str
    bias_score: float
    fairness_metrics: Dict[str, float]
    demographic_parity: float
    equal_opportunity: float
    mitigation_suggestions: List[str]
    timestamp: datetime

class HallucinationDetection(BaseModel):
    model_id: str
    generated_text: str
    source_context: Optional[str] = None
    fact_check_database: Optional[str] = None

class HallucinationResult(BaseModel):
    detection_id: str
    hallucination_detected: bool
    hallucination_type: Optional[str]
    confidence: float
    problematic_segments: List[Dict[str, Any]]
    fact_check_results: List[Dict[str, Any]]
    correction_suggestions: List[str]
    timestamp: datetime

class ExplainabilityRequest(BaseModel):
    model_id: str
    decision_id: str
    method: str = Field(default="SHAP", example="SHAP, LIME, Attention, GradCAM")

class ExplainabilityResponse(BaseModel):
    explanation_id: str
    method: str
    feature_importance: Dict[str, float]
    decision_path: List[str]
    counterfactual_examples: List[Dict[str, Any]]
    visualization_data: Dict[str, Any]
    timestamp: datetime

# ==================== FONCTIONS AUTHENTIFICATION ====================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    if username in fake_db["users"]:
        user_dict = fake_db["users"][username]
        return UserInDB(**user_dict)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ==================== FONCTIONS UTILITAIRES IA ====================

def calculate_model_complexity(params_millions: float, layers: int, hidden_size: int) -> float:
    """Calculer score complexité modèle"""
    return (params_millions / 1000) * (layers / 10) * (hidden_size / 1000)

def estimate_inference_time(params_millions: float, context_window: int) -> float:
    """Estimer temps inférence (ms)"""
    base_time = params_millions * 0.01
    context_factor = context_window / 1000
    return base_time * context_factor

def calculate_memory_usage(params_millions: float, precision_bits: int = 16) -> float:
    """Calculer utilisation mémoire (GB)"""
    bytes_per_param = precision_bits / 8
    return (params_millions * 1e6 * bytes_per_param) / 1e9

def simulate_attention_weights(input_tokens: List[str]) -> Dict[str, float]:
    """Simuler poids d'attention"""
    n_tokens = len(input_tokens)
    weights = np.random.dirichlet(np.ones(n_tokens), size=1)[0]
    return {token: float(weight) for token, weight in zip(input_tokens, weights)}

def detect_bias_in_predictions(predictions: np.ndarray, groups: List[str]) -> Dict[str, float]:
    """Détecter biais dans prédictions"""
    metrics = {}
    
    # Disparate Impact
    group_rates = {}
    for i, group in enumerate(groups):
        group_preds = predictions[predictions[:, 0] == i, 1]
        if len(group_preds) > 0:
            group_rates[group] = np.mean(group_preds)
    
    if len(group_rates) >= 2:
        rates = list(group_rates.values())
        metrics['disparate_impact'] = min(rates) / max(rates) if max(rates) > 0 else 0
    
    # Demographic Parity
    overall_rate = np.mean(predictions[:, 1])
    max_diff = 0
    for group, rate in group_rates.items():
        diff = abs(rate - overall_rate)
        max_diff = max(max_diff, diff)
    
    metrics['demographic_parity_diff'] = max_diff
    metrics['statistical_parity'] = 1 - max_diff
    
    return metrics

def check_factual_consistency(text: str, knowledge_base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Vérifier cohérence factuelle"""
    issues = []
    
    # Simuler vérification de faits
    sentences = text.split('.')
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) > 10:
            # Probabilité d'hallucination
            if np.random.random() < 0.2:
                issues.append({
                    'sentence_index': i,
                    'text': sentence.strip(),
                    'issue_type': 'potential_factual_error',
                    'confidence': float(np.random.uniform(0.6, 0.95))
                })
    
    return issues

def generate_counterfactuals(input_features: Dict[str, float], 
                            target_class: int,
                            n_examples: int = 3) -> List[Dict[str, Any]]:
    """Générer exemples contrefactuels"""
    counterfactuals = []
    
    for i in range(n_examples):
        modified_features = input_features.copy()
        
        # Modifier aléatoirement quelques features
        n_modifications = np.random.randint(1, min(4, len(input_features)))
        features_to_modify = np.random.choice(list(input_features.keys()), 
                                             size=n_modifications, 
                                             replace=False)
        
        for feature in features_to_modify:
            modified_features[feature] *= np.random.uniform(0.7, 1.3)
        
        counterfactuals.append({
            'modified_features': modified_features,
            'predicted_class': 1 - target_class,
            'distance': float(np.random.uniform(0.1, 0.5)),
            'plausibility': float(np.random.uniform(0.6, 0.95))
        })
    
    return counterfactuals

# ==================== ENDPOINTS AUTHENTIFICATION ====================

@app.post("/register", response_model=User, tags=["Authentication"])
async def register(user: UserCreate):
    """Créer nouveau compte utilisateur"""
    if user.username in fake_db["users"]:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    user_dict = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "disabled": False
    }
    fake_db["users"][user.username] = user_dict
    
    return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

@app.post("/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Connexion et obtention token JWT"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=User, tags=["Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Obtenir infos utilisateur courant"""
    return current_user

# ==================== ENDPOINTS MODÈLES IA ====================

@app.post("/models", response_model=AIModelResponse, tags=["AI Models"])
async def create_ai_model(
    model: AIModelCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Créer nouveau modèle IA"""
    model_id = str(uuid.uuid4())
    
    # Calculer métriques
    complexity = calculate_model_complexity(
        model.parameters_millions,
        model.architecture_layers,
        model.hidden_size
    )
    
    inference_time = estimate_inference_time(
        model.parameters_millions,
        model.context_window
    )
    
    memory = calculate_memory_usage(model.parameters_millions)
    
    model_data = {
        "id": model_id,
        **model.dict(),
        "complexity_score": complexity,
        "estimated_inference_ms": inference_time,
        "memory_gb": memory,
        "created_at": datetime.now(),
        "owner": current_user.username
    }
    
    fake_db["models"][model_id] = model_data
    
    return AIModelResponse(**model_data)

@app.get("/models", response_model=List[AIModelResponse], tags=["AI Models"])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[ModelType] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister tous les modèles"""
    models = list(fake_db["models"].values())
    
    if model_type:
        models = [m for m in models if m["model_type"] == model_type]
    
    return [AIModelResponse(**m) for m in models[skip:skip+limit]]

@app.get("/models/{model_id}", response_model=AIModelResponse, tags=["AI Models"])
async def get_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir détails d'un modèle"""
    if model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return AIModelResponse(**fake_db["models"][model_id])

# ==================== ENDPOINTS DÉCISIONS ====================

@app.post("/decisions", response_model=DecisionResponse, tags=["Decisions"])
async def make_decision(
    request: DecisionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Faire une prédiction/décision"""
    if request.model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = fake_db["models"][request.model_id]
    decision_id = str(uuid.uuid4())
    
    # Simuler génération
    import time
    start_time = time.time()
    
    # Tokenization simulée
    tokens = request.input_data.split()
    
    # Attention weights
    attention = simulate_attention_weights(tokens[:10])
    
    # Génération sortie (simulée)
    output = f"Réponse générée pour: {request.input_data[:50]}... [Simulation]"
    confidence = float(np.random.uniform(0.7, 0.98))
    
    # Reasoning steps
    reasoning = [
        "1. Analyse du contexte d'entrée",
        "2. Activation des couches d'attention",
        f"3. Passage par {model['architecture_layers']} couches",
        "4. Génération de tokens séquentiels",
        "5. Application des contraintes (top_p, temperature)",
        "6. Décodage et sélection finale"
    ]
    
    processing_time = (time.time() - start_time) * 1000
    
    decision_data = {
        "decision_id": decision_id,
        "model_id": request.model_id,
        "output": output,
        "confidence": confidence,
        "reasoning_steps": reasoning,
        "attention_weights": attention,
        "processing_time_ms": processing_time,
        "timestamp": datetime.now(),
        "input": request.input_data,
        "parameters": {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens
        }
    }
    
    fake_db["decisions"].append(decision_data)
    
    return DecisionResponse(**decision_data)

@app.get("/decisions", response_model=List[DecisionResponse], tags=["Decisions"])
async def list_decisions(
    skip: int = 0,
    limit: int = 100,
    model_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister toutes les décisions"""
    decisions = fake_db["decisions"]
    
    if model_id:
        decisions = [d for d in decisions if d["model_id"] == model_id]
    
    return [DecisionResponse(**d) for d in decisions[skip:skip+limit]]

# ==================== ENDPOINTS BIAIS ====================

@app.post("/bias/test", response_model=BiasTestResult, tags=["Bias Detection"])
async def test_bias(
    test: BiasTestCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Tester biais dans un modèle"""
    if test.model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    test_id = str(uuid.uuid4())
    
    # Simuler prédictions pour différents groupes
    n_samples = 1000
    n_groups = len(test.demographic_groups)
    
    predictions = np.zeros((n_samples, 2))
    predictions[:, 0] = np.random.choice(n_groups, size=n_samples)
    predictions[:, 1] = np.random.binomial(1, 0.5, size=n_samples)
    
    # Introduire un biais simulé
    for i in range(n_groups):
        mask = predictions[:, 0] == i
        bias_factor = 0.5 + (i * 0.15)
        predictions[mask, 1] = np.random.binomial(1, bias_factor, size=np.sum(mask))
    
    # Calculer métriques de fairness
    metrics = detect_bias_in_predictions(predictions, test.demographic_groups)
    
    # Calculer score de biais global
    bias_score = 1 - metrics.get('statistical_parity', 0.5)
    
    # Fairness metrics détaillées
    fairness_metrics = {
        'demographic_parity': metrics.get('statistical_parity', 0),
        'disparate_impact': metrics.get('disparate_impact', 0),
        'equal_opportunity': float(np.random.uniform(0.6, 0.9)),
        'equalized_odds': float(np.random.uniform(0.6, 0.9)),
        'calibration': float(np.random.uniform(0.7, 0.95))
    }
    
    # Suggestions de mitigation
    suggestions = []
    if bias_score > 0.3:
        suggestions.append("Rééquilibrer dataset avec oversampling/undersampling")
    if metrics.get('disparate_impact', 1) < 0.8:
        suggestions.append("Appliquer contraintes de fairness pendant entraînement")
    if fairness_metrics['equal_opportunity'] < 0.8:
        suggestions.append("Utiliser post-processing pour calibrer seuils par groupe")
    
    if not suggestions:
        suggestions.append("Biais acceptable - continuer monitoring")
    
    result_data = {
        "test_id": test_id,
        "model_id": test.model_id,
        "bias_type": test.bias_type.value,
        "bias_score": bias_score,
        "fairness_metrics": fairness_metrics,
        "demographic_parity": fairness_metrics['demographic_parity'],
        "equal_opportunity": fairness_metrics['equal_opportunity'],
        "mitigation_suggestions": suggestions,
        "timestamp": datetime.now(),
        "test_config": {
            "dataset": test.test_dataset,
            "groups": test.demographic_groups,
            "metrics": test.metrics_to_check
        }
    }
    
    fake_db["bias_tests"].append(result_data)
    
    return BiasTestResult(**result_data)

@app.get("/bias/tests", tags=["Bias Detection"])
async def list_bias_tests(
    skip: int = 0,
    limit: int = 100,
    model_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister tests de biais"""
    tests = fake_db["bias_tests"]
    
    if model_id:
        tests = [t for t in tests if t["model_id"] == model_id]
    
    return [BiasTestResult(**t) for t in tests[skip:skip+limit]]

# ==================== ENDPOINTS HALLUCINATIONS ====================

@app.post("/hallucination/detect", response_model=HallucinationResult, tags=["Hallucination"])
async def detect_hallucination(
    detection: HallucinationDetection,
    current_user: User = Depends(get_current_active_user)
):
    """Détecter hallucinations dans texte généré"""
    if detection.model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    detection_id = str(uuid.uuid4())
    
    # Vérification cohérence factuelle
    knowledge_base = {}  # Simulé
    issues = check_factual_consistency(detection.generated_text, knowledge_base)
    
    hallucination_detected = len(issues) > 0
    
    # Type d'hallucination
    hallucination_type = None
    if hallucination_detected:
        types = [HallucinationType.FACTUAL, HallucinationType.LOGICAL, 
                HallucinationType.CONTEXT, HallucinationType.TEMPORAL]
        hallucination_type = np.random.choice([t.value for t in types])
    
    # Confiance détection
    confidence = float(np.random.uniform(0.75, 0.95)) if hallucination_detected else 0.0
    
    # Segments problématiques
    problematic_segments = []
    for issue in issues:
        problematic_segments.append({
            'text': issue['text'],
            'position': issue['sentence_index'],
            'issue_type': issue['issue_type'],
            'severity': 'high' if issue['confidence'] > 0.8 else 'medium'
        })
    
    # Fact-checking
    fact_check_results = []
    for segment in problematic_segments[:3]:
        fact_check_results.append({
            'claim': segment['text'][:100],
            'verified': False,
            'contradicts_knowledge': True,
            'confidence': float(np.random.uniform(0.6, 0.9)),
            'sources': []
        })
    
    # Suggestions de correction
    corrections = []
    if hallucination_detected:
        corrections.extend([
            "Vérifier les faits avec une base de connaissances fiable",
            "Augmenter le grounding avec contexte additionnel",
            "Réduire la temperature de génération (< 0.7)",
            "Utiliser retrieval-augmented generation (RAG)",
            "Implémenter fact-checking en temps réel"
        ])
    
    result_data = {
        "detection_id": detection_id,
        "hallucination_detected": hallucination_detected,
        "hallucination_type": hallucination_type,
        "confidence": confidence,
        "problematic_segments": problematic_segments,
        "fact_check_results": fact_check_results,
        "correction_suggestions": corrections,
        "timestamp": datetime.now(),
        "analysis": {
            "total_sentences": len(detection.generated_text.split('.')),
            "problematic_sentences": len(problematic_segments),
            "hallucination_rate": len(problematic_segments) / max(1, len(detection.generated_text.split('.')))
        }
    }
    
    fake_db["hallucinations"].append(result_data)
    
    return HallucinationResult(**result_data)

@app.get("/hallucination/detections", tags=["Hallucination"])
async def list_hallucinations(
    skip: int = 0,
    limit: int = 100,
    model_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister détections d'hallucinations"""
    detections = fake_db["hallucinations"]
    
    if model_id:
        detections = [d for d in detections if d.get("model_id") == model_id]
    
    return [HallucinationResult(**d) for d in detections[skip:skip+limit]]

# ==================== ENDPOINTS EXPLAINABILITÉ ====================

@app.post("/explainability/explain", response_model=ExplainabilityResponse, tags=["Explainability"])
async def explain_decision(
    request: ExplainabilityRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Expliquer une décision IA"""
    if request.model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Trouver la décision
    decision = None
    for d in fake_db["decisions"]:
        if d["decision_id"] == request.decision_id:
            decision = d
            break
    
    if not decision:
        raise HTTPException(status_code=404, detail="Decision not found")
    
    explanation_id = str(uuid.uuid4())
    
    # Feature importance (SHAP values simulés)
    features = ['context_relevance', 'semantic_similarity', 'frequency', 
               'position', 'attention_score', 'prior_knowledge']
    
    importance = np.random.dirichlet(np.ones(len(features)), size=1)[0]
    feature_importance = {f: float(imp) for f, imp in zip(features, importance)}
    
    # Decision path
    decision_path = [
        f"Input received: {len(decision.get('input', ''))} tokens",
        f"Embedding layer: {fake_db['models'][request.model_id]['hidden_size']}d vectors",
        f"Attention mechanism: {fake_db['models'][request.model_id]['attention_heads']} heads",
        f"Key features activated: {', '.join(sorted(feature_importance.keys(), key=lambda x: feature_importance[x], reverse=True)[:3])}",
        f"Output layer: Softmax over vocabulary",
        f"Final prediction: confidence {decision.get('confidence', 0):.2%}"
    ]
    
    # Counterfactual examples
    input_features = {f: float(np.random.uniform(0, 1)) for f in features}
    counterfactuals = generate_counterfactuals(
        input_features,
        target_class=1,
        n_examples=3
    )
    
    # Visualization data
    visualization_data = {
        'method': request.method,
        'feature_importance_chart': {
            'labels': list(feature_importance.keys()),
            'values': list(feature_importance.values())
        },
        'attention_heatmap': decision.get('attention_weights', {}),
        'layer_activations': {
            f'layer_{i}': float(np.random.uniform(0.3, 0.9))
            for i in range(min(10, fake_db['models'][request.model_id]['architecture_layers']))
        }
    }
    
    result_data = {
        "explanation_id": explanation_id,
        "method": request.method,
        "feature_importance": feature_importance,
        "decision_path": decision_path,
        "counterfactual_examples": counterfactuals,
        "visualization_data": visualization_data,
        "timestamp": datetime.now(),
        "model_id": request.model_id,
        "decision_id": request.decision_id
    }
    
    fake_db["explainability"].append(result_data)
    
    return ExplainabilityResponse(**result_data)

@app.get("/explainability/explanations", tags=["Explainability"])
async def list_explanations(
    skip: int = 0,
    limit: int = 100,
    model_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister explications"""
    explanations = fake_db["explainability"]
    
    if model_id:
        explanations = [e for e in explanations if e.get("model_id") == model_id]
    
    return [ExplainabilityResponse(**e) for e in explanations[skip:skip+limit]]

# ==================== ENDPOINTS STATISTIQUES ====================

@app.get("/stats/overview", tags=["Statistics"])
async def get_overview_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Statistiques générales plateforme"""
    total_decisions = len(fake_db["decisions"])
    total_bias_tests = len(fake_db["bias_tests"])
    total_hallucinations = len(fake_db["hallucinations"])
    
    # Calculer moyennes
    avg_confidence = 0
    if total_decisions > 0:
        avg_confidence = np.mean([d.get("confidence", 0) for d in fake_db["decisions"]])
    
    avg_bias_score = 0
    if total_bias_tests > 0:
        avg_bias_score = np.mean([t.get("bias_score", 0) for t in fake_db["bias_tests"]])
    
    hallucination_rate = 0
    if total_hallucinations > 0:
        detected = sum(1 for h in fake_db["hallucinations"] if h.get("hallucination_detected", False))
        hallucination_rate = detected / total_hallucinations
    
    return {
        "models": len(fake_db["models"]),
        "decisions": total_decisions,
        "bias_tests": total_bias_tests,
        "hallucination_checks": total_hallucinations,
        "explainability_analyses": len(fake_db["explainability"]),
        "avg_confidence": float(avg_confidence),
        "avg_bias_score": float(avg_bias_score),
        "hallucination_rate": float(hallucination_rate),
        "total_parameters": sum(m.get("parameters_millions", 0) for m in fake_db["models"].values())
    }

@app.get("/stats/model/{model_id}", tags=["Statistics"])
async def get_model_stats(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Statistiques d'un modèle"""
    if model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = fake_db["models"][model_id]
    
    # Compter décisions
    model_decisions = [d for d in fake_db["decisions"] if d.get("model_id") == model_id]
    
    # Statistiques
    avg_processing_time = 0
    avg_confidence = 0
    
    if model_decisions:
        avg_processing_time = np.mean([d.get("processing_time_ms", 0) for d in model_decisions])
        avg_confidence = np.mean([d.get("confidence", 0) for d in model_decisions])
    
    return {
        "model_id": model_id,
        "model_name": model["name"],
        "n_decisions": len(model_decisions),
        "avg_processing_time_ms": float(avg_processing_time),
        "avg_confidence": float(avg_confidence),
        "total_tokens_generated": len(model_decisions) * 200,  # Estimation
        "uptime_hours": float(np.random.uniform(100, 1000))
    }

# ==================== ENDPOINTS MITIGATION ====================

@app.post("/mitigation/debias", tags=["Mitigation"])
async def apply_debiasing(
    model_id: str,
    technique: str = "adversarial_debiasing",
    target_fairness: float = 0.9,
    current_user: User = Depends(get_current_active_user)
):
    """Appliquer technique de débiaisage"""
    if model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    techniques = {
        'adversarial_debiasing': 'Entraînement adversarial pour fairness',
        'reweighting': 'Repondération des exemples d\'entraînement',
        'calibration': 'Calibration des seuils de décision',
        'preprocessing': 'Transformation des features pour éliminer corrélations',
        'postprocessing': 'Ajustement des prédictions post-hoc'
    }
    
    if technique not in techniques:
        raise HTTPException(status_code=400, detail=f"Technique invalide. Options: {list(techniques.keys())}")
    
    # Simuler application
    import time
    time.sleep(1)
    
    # Résultats
    improvement = float(np.random.uniform(0.1, 0.3))
    
    return {
        "model_id": model_id,
        "technique_applied": technique,
        "description": techniques[technique],
        "target_fairness": target_fairness,
        "fairness_before": float(np.random.uniform(0.5, 0.7)),
        "fairness_after": float(np.random.uniform(0.8, 0.95)),
        "improvement": improvement,
        "performance_impact": float(np.random.uniform(-0.05, 0.02)),
        "timestamp": datetime.now().isoformat(),
        "success": True
    }

@app.post("/mitigation/reduce-hallucination", tags=["Mitigation"])
async def reduce_hallucination(
    model_id: str,
    method: str = "retrieval_augmentation",
    current_user: User = Depends(get_current_active_user)
):
    """Réduire hallucinations"""
    if model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    methods = {
        'retrieval_augmentation': 'RAG - Retrieval Augmented Generation',
        'fact_checking': 'Vérification factuelle en temps réel',
        'temperature_tuning': 'Ajustement température génération',
        'constrained_decoding': 'Décodage contraint par connaissances',
        'knowledge_grounding': 'Ancrage dans base de connaissances',
        'confidence_thresholding': 'Filtrage par seuil de confiance'
    }
    
    if method not in methods:
        raise HTTPException(status_code=400, detail=f"Méthode invalide. Options: {list(methods.keys())}")
    
    # Simuler réduction
    reduction_rate = float(np.random.uniform(0.3, 0.7))
    
    return {
        "model_id": model_id,
        "method_applied": method,
        "description": methods[method],
        "hallucination_rate_before": float(np.random.uniform(0.2, 0.4)),
        "hallucination_rate_after": float(np.random.uniform(0.05, 0.15)),
        "reduction_rate": reduction_rate,
        "factual_accuracy_gain": float(np.random.uniform(0.15, 0.35)),
        "latency_impact_ms": float(np.random.uniform(10, 100)),
        "timestamp": datetime.now().isoformat(),
        "success": True
    }

# ==================== ENDPOINTS ARCHITECTURE ====================

@app.get("/architecture/layers/{model_id}", tags=["Architecture"])
async def get_model_architecture(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir architecture détaillée du modèle"""
    if model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = fake_db["models"][model_id]
    
    layers = []
    
    # Embedding layer
    layers.append({
        "layer_id": 0,
        "type": "Embedding",
        "input_dim": 50000,  # vocab size
        "output_dim": model["hidden_size"],
        "parameters": 50000 * model["hidden_size"]
    })
    
    # Transformer layers
    for i in range(model["architecture_layers"]):
        # Multi-head attention
        layers.append({
            "layer_id": len(layers),
            "type": "MultiHeadAttention",
            "n_heads": model["attention_heads"],
            "hidden_size": model["hidden_size"],
            "parameters": 4 * model["hidden_size"] * model["hidden_size"]
        })
        
        # Feed-forward
        layers.append({
            "layer_id": len(layers),
            "type": "FeedForward",
            "hidden_size": model["hidden_size"],
            "intermediate_size": model["hidden_size"] * 4,
            "parameters": 2 * model["hidden_size"] * (model["hidden_size"] * 4)
        })
        
        # Layer normalization
        layers.append({
            "layer_id": len(layers),
            "type": "LayerNorm",
            "hidden_size": model["hidden_size"],
            "parameters": 2 * model["hidden_size"]
        })
    
    # Output layer
    layers.append({
        "layer_id": len(layers),
        "type": "Output",
        "input_dim": model["hidden_size"],
        "output_dim": 50000,
        "parameters": model["hidden_size"] * 50000
    })
    
    total_params = sum(layer["parameters"] for layer in layers)
    
    return {
        "model_id": model_id,
        "model_name": model["name"],
        "total_layers": len(layers),
        "total_parameters": total_params,
        "layers": layers,
        "computational_graph": {
            "forward_pass_flops": total_params * 2,
            "backward_pass_flops": total_params * 4,
            "memory_footprint_gb": model["memory_gb"]
        }
    }

@app.post("/architecture/visualize/{model_id}", tags=["Architecture"])
async def visualize_architecture(
    model_id: str,
    format: str = "graph",
    current_user: User = Depends(get_current_active_user)
):
    """Générer visualisation architecture"""
    if model_id not in fake_db["models"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = fake_db["models"][model_id]
    
    # Graph structure pour visualisation
    nodes = []
    edges = []
    
    # Input node
    nodes.append({"id": "input", "label": "Input", "type": "input"})
    
    # Embedding
    nodes.append({"id": "embedding", "label": "Embedding", "type": "embedding"})
    edges.append({"from": "input", "to": "embedding"})
    
    prev_node = "embedding"
    
    # Transformer blocks
    for i in range(min(5, model["architecture_layers"])):  # Limiter pour visualisation
        block_id = f"transformer_{i}"
        nodes.append({
            "id": block_id,
            "label": f"Transformer Block {i+1}",
            "type": "transformer",
            "details": {
                "attention_heads": model["attention_heads"],
                "hidden_size": model["hidden_size"]
            }
        })
        edges.append({"from": prev_node, "to": block_id})
        prev_node = block_id
    
    # Output
    nodes.append({"id": "output", "label": "Output", "type": "output"})
    edges.append({"from": prev_node, "to": "output"})
    
    return {
        "model_id": model_id,
        "visualization_format": format,
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "metadata": {
            "total_layers": model["architecture_layers"],
            "parameters": model["parameters_millions"],
            "complexity": model["complexity_score"]
        }
    }

# ==================== ENDPOINT RACINE ====================

@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil API"""
    return {
        "message": "🤖 AI Decision Intelligence API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "authentication": "/token, /register",
            "models": "/models",
            "decisions": "/decisions",
            "bias": "/bias",
            "hallucination": "/hallucination",
            "explainability": "/explainability",
            "mitigation": "/mitigation",
            "architecture": "/architecture",
            "stats": "/stats"
        }
    }

@app.get("/health", tags=["Root"])
async def health_check():
    """Vérification santé API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",
        "models_loaded": len(fake_db["models"])
    }

# ==================== LANCEMENT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8030)