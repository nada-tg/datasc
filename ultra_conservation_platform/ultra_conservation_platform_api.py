"""
🛡️ Ultra Conservation Technologies Platform - API FastAPI
Préservation • Restauration • Archivage • Monitoring • Protection

Installation:
pip install fastapi uvicorn sqlalchemy pydantic python-jose passlib python-multipart

Lancement:
uvicorn ultra_conservation_platform_api:app --reload --host 0.0.0.0 --port 8047

Documentation: http://localhost:8040/docs
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
    title="🛡️ Ultra Conservation Technologies API",
    description="API complète pour conservation, préservation et restauration",
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
SECRET_KEY = "your_secret_key_ultra_conservation_platform"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Base de données simulée
fake_db = {
    "users": {},
    "artifacts": {},
    "materials": {},
    "preservation_plans": [],
    "restorations": [],
    "monitoring_data": [],
    "archives": {},
    "climate_logs": [],
    "degradation_analyses": [],
    "treatments": []
}

# ==================== ENUMS ====================

class ArtifactType(str, Enum):
    PAINTING = "Peinture"
    SCULPTURE = "Sculpture"
    MANUSCRIPT = "Manuscrit"
    TEXTILE = "Textile"
    CERAMIC = "Céramique"
    METAL = "Métal"
    WOOD = "Bois"
    STONE = "Pierre"
    DIGITAL = "Numérique"

class MaterialType(str, Enum):
    ORGANIC = "Organique"
    INORGANIC = "Inorganique"
    COMPOSITE = "Composite"
    SYNTHETIC = "Synthétique"

class ConservationState(str, Enum):
    EXCELLENT = "Excellent"
    GOOD = "Bon"
    FAIR = "Moyen"
    POOR = "Mauvais"
    CRITICAL = "Critique"

class DegradationType(str, Enum):
    PHYSICAL = "Physique (usure, fissures)"
    CHEMICAL = "Chimique (oxydation, acidification)"
    BIOLOGICAL = "Biologique (moisissures, insectes)"
    ENVIRONMENTAL = "Environnemental (lumière, humidité)"
    MECHANICAL = "Mécanique (chocs, vibrations)"

class TreatmentType(str, Enum):
    CLEANING = "Nettoyage"
    CONSOLIDATION = "Consolidation"
    STABILIZATION = "Stabilisation"
    RESTORATION = "Restauration"
    DIGITIZATION = "Numérisation"
    ENCAPSULATION = "Encapsulation"

# ==================== MODELS PYDANTIC ====================

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    role: Optional[str] = "conservator"

class UserInDB(User):
    hashed_password: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class ArtifactCreate(BaseModel):
    name: str = Field(..., example="La Joconde")
    artifact_type: ArtifactType
    material_type: MaterialType
    creation_date: Optional[str] = None
    origin: str = Field(..., example="Italie, Renaissance")
    dimensions: Dict[str, float] = Field(..., example={"height": 77, "width": 53, "depth": 2})
    weight_kg: Optional[float] = None
    current_location: str
    estimated_value_eur: Optional[float] = None

class ArtifactResponse(BaseModel):
    id: str
    name: str
    artifact_type: str
    material_type: str
    conservation_state: str
    risk_score: float
    last_inspection: datetime
    requires_intervention: bool
    created_at: datetime

class MaterialAnalysis(BaseModel):
    artifact_id: str
    material_composition: Dict[str, float]
    porosity: float = Field(..., ge=0, le=1)
    moisture_content: float = Field(..., ge=0, le=100)
    ph_level: Optional[float] = Field(None, ge=0, le=14)
    structural_integrity: float = Field(..., ge=0, le=100)

class MaterialAnalysisResponse(BaseModel):
    analysis_id: str
    artifact_id: str
    material_composition: Dict[str, float]
    degradation_indicators: Dict[str, float]
    recommendations: List[str]
    urgency_level: str
    timestamp: datetime

class PreservationPlan(BaseModel):
    artifact_id: str
    target_conditions: Dict[str, Any]
    treatments_planned: List[str]
    timeline_months: int
    budget_eur: float
    priority: str = Field(..., example="High, Medium, Low")

class PreservationPlanResponse(BaseModel):
    plan_id: str
    artifact_id: str
    treatments: List[Dict[str, Any]]
    estimated_duration_months: int
    estimated_cost_eur: float
    success_probability: float
    created_at: datetime

class ClimateMonitoring(BaseModel):
    artifact_id: str
    temperature_c: float = Field(..., ge=-50, le=50)
    humidity_percent: float = Field(..., ge=0, le=100)
    light_lux: float = Field(..., ge=0)
    uv_index: float = Field(..., ge=0, le=15)
    air_quality_index: Optional[float] = Field(None, ge=0, le=500)

class ClimateAlert(BaseModel):
    alert_id: str
    artifact_id: str
    parameter: str
    current_value: float
    threshold_value: float
    severity: str
    timestamp: datetime
    action_required: str

class DegradationAnalysis(BaseModel):
    artifact_id: str
    degradation_type: DegradationType
    affected_area_percent: float = Field(..., ge=0, le=100)
    progression_rate: str

class DegradationResult(BaseModel):
    analysis_id: str
    artifact_id: str
    degradation_types: List[str]
    severity_score: float
    predicted_lifespan_years: float
    intervention_urgency: str
    mitigation_strategies: List[str]
    timestamp: datetime

class TreatmentRecord(BaseModel):
    artifact_id: str
    treatment_type: TreatmentType
    description: str
    products_used: List[str]
    duration_hours: float
    cost_eur: float

class TreatmentResult(BaseModel):
    treatment_id: str
    artifact_id: str
    treatment_type: str
    before_state: str
    after_state: str
    improvement_percent: float
    side_effects: List[str]
    documentation: Dict[str, Any]
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
    return current_user

# ==================== FONCTIONS UTILITAIRES CONSERVATION ====================

def calculate_risk_score(artifact: Dict) -> float:
    """Calculer score de risque dégradation"""
    base_risk = 0.2
    
    # Facteurs de risque
    age_factor = 0.1 if artifact.get('creation_date') else 0.05
    material_risk = {
        'Organique': 0.3,
        'Inorganique': 0.1,
        'Composite': 0.2,
        'Synthétique': 0.15
    }
    
    risk = base_risk + age_factor + material_risk.get(artifact['material_type'], 0.2)
    
    # Ajouter variabilité
    risk += np.random.uniform(-0.05, 0.15)
    
    return min(1.0, max(0.0, risk))

def assess_conservation_state(risk_score: float) -> str:
    """Évaluer état conservation"""
    if risk_score < 0.2:
        return ConservationState.EXCELLENT.value
    elif risk_score < 0.4:
        return ConservationState.GOOD.value
    elif risk_score < 0.6:
        return ConservationState.FAIR.value
    elif risk_score < 0.8:
        return ConservationState.POOR.value
    else:
        return ConservationState.CRITICAL.value

def analyze_material_degradation(composition: Dict[str, float], 
                                 environmental_data: Dict) -> Dict[str, float]:
    """Analyser dégradation matériau"""
    indicators = {}
    
    # Oxydation
    if 'metal' in str(composition).lower():
        indicators['oxidation'] = float(np.random.uniform(0.1, 0.8))
    
    # Acidification
    if 'paper' in str(composition).lower() or 'cellulose' in str(composition).lower():
        indicators['acidification'] = float(np.random.uniform(0.2, 0.7))
    
    # Photodégradation
    if environmental_data.get('light_exposure', 0) > 100:
        indicators['photodegradation'] = float(np.random.uniform(0.3, 0.9))
    
    # Biodétérioration
    if environmental_data.get('humidity', 0) > 65:
        indicators['biodeterioration'] = float(np.random.uniform(0.2, 0.6))
    
    # Décoloration
    indicators['discoloration'] = float(np.random.uniform(0.1, 0.5))
    
    return indicators

def predict_lifespan(current_state: float, degradation_rate: float) -> float:
    """Prédire durée de vie restante"""
    if degradation_rate <= 0:
        return 1000  # Stable
    
    # Années jusqu'à état critique (0.9)
    years_to_critical = (0.9 - current_state) / degradation_rate
    
    return max(0, years_to_critical)

def generate_preservation_recommendations(degradation_data: Dict) -> List[str]:
    """Générer recommandations préservation"""
    recommendations = []
    
    for indicator, value in degradation_data.items():
        if value > 0.7:
            if indicator == 'oxidation':
                recommendations.append("🛡️ Appliquer couche protectrice anti-corrosion")
                recommendations.append("💨 Contrôler atmosphère (désoxygénation)")
            elif indicator == 'acidification':
                recommendations.append("⚗️ Désacidification avec solutions tampon")
                recommendations.append("📦 Stockage dans contenants alcalins")
            elif indicator == 'photodegradation':
                recommendations.append("☀️ Réduire exposition lumière (< 50 lux)")
                recommendations.append("🔆 Installer filtres UV")
            elif indicator == 'biodeterioration':
                recommendations.append("🌡️ Contrôler humidité (40-55%)")
                recommendations.append("🧪 Traitement biocide préventif")
    
    if not recommendations:
        recommendations.append("✅ Maintenir conditions actuelles")
        recommendations.append("📊 Monitoring régulier recommandé")
    
    return recommendations

# ==================== ENDPOINTS AUTHENTIFICATION ====================

@app.post("/register", response_model=User, tags=["Authentication"])
async def register(user: UserCreate):
    """Créer nouveau compte"""
    if user.username in fake_db["users"]:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    user_dict = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "role": "conservator"
    }
    fake_db["users"][user.username] = user_dict
    
    return User(**{k: v for k, v in user_dict.items() if k != "hashed_password"})

@app.post("/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Connexion"""
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
    """Infos utilisateur"""
    return current_user

# ==================== ENDPOINTS ARTIFACTS ====================

@app.post("/artifacts", response_model=ArtifactResponse, tags=["Artifacts"])
async def create_artifact(
    artifact: ArtifactCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Enregistrer nouvel artefact"""
    artifact_id = str(uuid.uuid4())
    
    risk_score = calculate_risk_score(artifact.dict())
    conservation_state = assess_conservation_state(risk_score)
    
    artifact_data = {
        "id": artifact_id,
        **artifact.dict(),
        "conservation_state": conservation_state,
        "risk_score": risk_score,
        "last_inspection": datetime.now(),
        "requires_intervention": risk_score > 0.6,
        "created_at": datetime.now(),
        "curator": current_user.username
    }
    
    fake_db["artifacts"][artifact_id] = artifact_data
    
    return ArtifactResponse(**artifact_data)

@app.get("/artifacts", response_model=List[ArtifactResponse], tags=["Artifacts"])
async def list_artifacts(
    skip: int = 0,
    limit: int = 100,
    artifact_type: Optional[ArtifactType] = None,
    conservation_state: Optional[ConservationState] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister artefacts"""
    artifacts = list(fake_db["artifacts"].values())
    
    if artifact_type:
        artifacts = [a for a in artifacts if a["artifact_type"] == artifact_type]
    
    if conservation_state:
        artifacts = [a for a in artifacts if a["conservation_state"] == conservation_state.value]
    
    return [ArtifactResponse(**a) for a in artifacts[skip:skip+limit]]

@app.get("/artifacts/{artifact_id}", response_model=ArtifactResponse, tags=["Artifacts"])
async def get_artifact(
    artifact_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Détails artefact"""
    if artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return ArtifactResponse(**fake_db["artifacts"][artifact_id])

# ==================== ENDPOINTS ANALYSIS ====================

@app.post("/analysis/material", response_model=MaterialAnalysisResponse, tags=["Analysis"])
async def analyze_material(
    analysis: MaterialAnalysis,
    current_user: User = Depends(get_current_active_user)
):
    """Analyser matériau"""
    if analysis.artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    analysis_id = str(uuid.uuid4())
    
    # Simuler analyse environnementale
    env_data = {
        'light_exposure': float(np.random.uniform(50, 300)),
        'humidity': analysis.moisture_content
    }
    
    # Analyser dégradation
    degradation = analyze_material_degradation(
        analysis.material_composition,
        env_data
    )
    
    # Générer recommandations
    recommendations = generate_preservation_recommendations(degradation)
    
    # Déterminer urgence
    max_degradation = max(degradation.values()) if degradation else 0
    if max_degradation > 0.7:
        urgency = "URGENT"
    elif max_degradation > 0.5:
        urgency = "High"
    elif max_degradation > 0.3:
        urgency = "Medium"
    else:
        urgency = "Low"
    
    result_data = {
        "analysis_id": analysis_id,
        "artifact_id": analysis.artifact_id,
        "material_composition": analysis.material_composition,
        "degradation_indicators": degradation,
        "recommendations": recommendations,
        "urgency_level": urgency,
        "timestamp": datetime.now()
    }
    
    fake_db["degradation_analyses"].append(result_data)
    
    return MaterialAnalysisResponse(**result_data)

@app.post("/analysis/degradation", response_model=DegradationResult, tags=["Analysis"])
async def analyze_degradation(
    analysis: DegradationAnalysis,
    current_user: User = Depends(get_current_active_user)
):
    """Analyser dégradation"""
    if analysis.artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    analysis_id = str(uuid.uuid4())
    artifact = fake_db["artifacts"][analysis.artifact_id]
    
    # Types de dégradation détectés
    degradation_types = [analysis.degradation_type.value]
    
    # Score sévérité
    severity = analysis.affected_area_percent / 100 * 0.7 + artifact['risk_score'] * 0.3
    
    # Prédire durée de vie
    degradation_rate = 0.05 if analysis.progression_rate == "Slow" else 0.15 if analysis.progression_rate == "Moderate" else 0.30
    lifespan = predict_lifespan(artifact['risk_score'], degradation_rate)
    
    # Urgence intervention
    if severity > 0.8 or lifespan < 5:
        urgency = "CRITICAL - Immediate action"
    elif severity > 0.6 or lifespan < 15:
        urgency = "HIGH - Action within 6 months"
    elif severity > 0.4 or lifespan < 30:
        urgency = "MEDIUM - Action within 2 years"
    else:
        urgency = "LOW - Routine monitoring"
    
    # Stratégies mitigation
    strategies = []
    if analysis.degradation_type == DegradationType.PHYSICAL:
        strategies.extend(["Consolidation structurelle", "Encadrement protecteur"])
    elif analysis.degradation_type == DegradationType.CHEMICAL:
        strategies.extend(["Neutralisation pH", "Atmosphère contrôlée"])
    elif analysis.degradation_type == DegradationType.BIOLOGICAL:
        strategies.extend(["Traitement biocide", "Contrôle humidité"])
    elif analysis.degradation_type == DegradationType.ENVIRONMENTAL:
        strategies.extend(["Filtres UV", "Contrôle climatique"])
    
    result_data = {
        "analysis_id": analysis_id,
        "artifact_id": analysis.artifact_id,
        "degradation_types": degradation_types,
        "severity_score": float(severity),
        "predicted_lifespan_years": float(lifespan),
        "intervention_urgency": urgency,
        "mitigation_strategies": strategies,
        "timestamp": datetime.now()
    }
    
    return DegradationResult(**result_data)

# ==================== ENDPOINTS PRESERVATION ====================

@app.post("/preservation/plan", response_model=PreservationPlanResponse, tags=["Preservation"])
async def create_preservation_plan(
    plan: PreservationPlan,
    current_user: User = Depends(get_current_active_user)
):
    """Créer plan de préservation"""
    if plan.artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    plan_id = str(uuid.uuid4())
    
    # Générer traitements détaillés
    treatments = []
    for treatment_name in plan.treatments_planned:
        treatments.append({
            "name": treatment_name,
            "duration_weeks": int(np.random.uniform(2, 12)),
            "cost_eur": float(np.random.uniform(500, 5000)),
            "success_rate": float(np.random.uniform(0.75, 0.98)),
            "risks": ["Minimal color alteration", "Temporary stress on material"]
        })
    
    # Calculs
    total_duration = sum(t['duration_weeks'] for t in treatments) / 4  # mois
    total_cost = sum(t['cost_eur'] for t in treatments)
    avg_success = np.mean([t['success_rate'] for t in treatments])
    
    plan_data = {
        "plan_id": plan_id,
        "artifact_id": plan.artifact_id,
        "treatments": treatments,
        "estimated_duration_months": int(total_duration),
        "estimated_cost_eur": float(total_cost),
        "success_probability": float(avg_success),
        "created_at": datetime.now(),
        "priority": plan.priority
    }
    
    fake_db["preservation_plans"].append(plan_data)
    
    return PreservationPlanResponse(**plan_data)

@app.get("/preservation/plans", tags=["Preservation"])
async def list_preservation_plans(
    skip: int = 0,
    limit: int = 100,
    artifact_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister plans préservation"""
    plans = fake_db["preservation_plans"]
    
    if artifact_id:
        plans = [p for p in plans if p["artifact_id"] == artifact_id]
    
    return [PreservationPlanResponse(**p) for p in plans[skip:skip+limit]]

# ==================== ENDPOINTS CLIMATE MONITORING ====================

@app.post("/monitoring/climate", response_model=ClimateAlert, tags=["Monitoring"])
async def monitor_climate(
    monitoring: ClimateMonitoring,
    current_user: User = Depends(get_current_active_user)
):
    """Monitorer conditions climatiques"""
    if monitoring.artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    monitoring_data = {
        **monitoring.dict(),
        "timestamp": datetime.now()
    }
    fake_db["climate_logs"].append(monitoring_data)
    
    # Vérifier seuils
    alerts = []
    
    # Température
    if not (18 <= monitoring.temperature_c <= 22):
        alerts.append({
            "parameter": "Temperature",
            "current_value": monitoring.temperature_c,
            "threshold_value": 20.0,
            "severity": "HIGH" if abs(monitoring.temperature_c - 20) > 5 else "MEDIUM"
        })
    
    # Humidité
    if not (40 <= monitoring.humidity_percent <= 55):
        alerts.append({
            "parameter": "Humidity",
            "current_value": monitoring.humidity_percent,
            "threshold_value": 50.0,
            "severity": "HIGH" if abs(monitoring.humidity_percent - 50) > 15 else "MEDIUM"
        })
    
    # Lumière
    if monitoring.light_lux > 150:
        alerts.append({
            "parameter": "Light",
            "current_value": monitoring.light_lux,
            "threshold_value": 150.0,
            "severity": "HIGH" if monitoring.light_lux > 300 else "MEDIUM"
        })
    
    # UV
    if monitoring.uv_index > 0.5:
        alerts.append({
            "parameter": "UV",
            "current_value": monitoring.uv_index,
            "threshold_value": 0.5,
            "severity": "HIGH"
        })
    
    # Retourner première alerte ou tout OK
    if alerts:
        alert = alerts[0]
        alert_id = str(uuid.uuid4())
        
        action_map = {
            "Temperature": "Ajuster HVAC system",
            "Humidity": "Activer déshumidificateur/humidificateur",
            "Light": "Réduire éclairage ou installer filtres",
            "UV": "Installer filtres UV immédiatement"
        }
        
        result = ClimateAlert(
            alert_id=alert_id,
            artifact_id=monitoring.artifact_id,
            parameter=alert["parameter"],
            current_value=alert["current_value"],
            threshold_value=alert["threshold_value"],
            severity=alert["severity"],
            timestamp=datetime.now(),
            action_required=action_map[alert["parameter"]]
        )
        
        return result
    else:
        # Pas d'alerte
        return ClimateAlert(
            alert_id=str(uuid.uuid4()),
            artifact_id=monitoring.artifact_id,
            parameter="All",
            current_value=0.0,
            threshold_value=0.0,
            severity="OK",
            timestamp=datetime.now(),
            action_required="None - conditions optimal"
        )

@app.get("/monitoring/climate/{artifact_id}", tags=["Monitoring"])
async def get_climate_history(
    artifact_id: str,
    hours: int = 24,
    current_user: User = Depends(get_current_active_user)
):
    """Historique climat"""
    if artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # Filtrer logs
    logs = [l for l in fake_db["climate_logs"] if l["artifact_id"] == artifact_id]
    
    return {
        "artifact_id": artifact_id,
        "period_hours": hours,
        "n_measurements": len(logs),
        "logs": logs[-100:]  # Dernier 100
    }

# ==================== ENDPOINTS TREATMENT ====================

@app.post("/treatment/apply", response_model=TreatmentResult, tags=["Treatment"])
async def apply_treatment(
    treatment: TreatmentRecord,
    current_user: User = Depends(get_current_active_user)
):
    """Appliquer traitement"""
    if treatment.artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    treatment_id = str(uuid.uuid4())
    artifact = fake_db["artifacts"][treatment.artifact_id]
    
    # État avant
    before_state = artifact['conservation_state']
    before_risk = artifact['risk_score']
    
    # Simuler amélioration
    improvement_map = {
        TreatmentType.CLEANING: 0.15,
        TreatmentType.CONSOLIDATION: 0.25,
        TreatmentType.STABILIZATION: 0.30,
        TreatmentType.RESTORATION: 0.40,
        TreatmentType.DIGITIZATION: 0.0,  # Pas d'amélioration physique
        TreatmentType.ENCAPSULATION: 0.20
    }
    
    improvement = improvement_map.get(treatment.treatment_type, 0.15)
    improvement += np.random.uniform(-0.05, 0.1)
    
    # Nouveau état
    new_risk = max(0.0, before_risk - improvement)
    after_state = assess_conservation_state(new_risk)
    
    # Mise à jour artifact
    artifact['conservation_state'] = after_state
    artifact['risk_score'] = new_risk
    artifact['last_inspection'] = datetime.now()
    
    # Effets secondaires possibles
    side_effects = []
    if np.random.random() < 0.1:
        side_effects.append("Légère altération de couleur")
    if np.random.random() < 0.05:
        side_effects.append("Stress temporaire du matériau")
    
    if not side_effects:
        side_effects.append("Aucun effet secondaire observé")
    
    # Documentation
    documentation = {
        "before_images": ["before_001.jpg", "before_002.jpg"],
        "after_images": ["after_001.jpg", "after_002.jpg"],
        "treatment_protocol": treatment.description,
        "environmental_conditions": {
            "temperature_c": 20,
            "humidity_percent": 50
        },
        "conservator": current_user.username,
        "approval_date": datetime.now().isoformat()
    }
    
    result_data = {
        "treatment_id": treatment_id,
        "artifact_id": treatment.artifact_id,
        "treatment_type": treatment.treatment_type.value,
        "before_state": before_state,
        "after_state": after_state,
        "improvement_percent": float(improvement * 100),
        "side_effects": side_effects,
        "documentation": documentation,
        "timestamp": datetime.now()
    }
    
    fake_db["treatments"].append(result_data)
    
    return TreatmentResult(**result_data)

@app.get("/treatment/history/{artifact_id}", tags=["Treatment"])
async def get_treatment_history(
    artifact_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Historique traitements"""
    if artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    treatments = [t for t in fake_db["treatments"] if t["artifact_id"] == artifact_id]
    
    return {
        "artifact_id": artifact_id,
        "total_treatments": len(treatments),
        "treatments": [TreatmentResult(**t) for t in treatments]
    }

# ==================== ENDPOINTS STATISTICS ====================

@app.get("/stats/overview", tags=["Statistics"])
async def get_overview_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Statistiques générales"""
    total_artifacts = len(fake_db["artifacts"])
    
    # États conservation
    states_count = {}
    for artifact in fake_db["artifacts"].values():
        state = artifact['conservation_state']
        states_count[state] = states_count.get(state, 0) + 1
    
    # Artefacts à risque
    at_risk = sum(1 for a in fake_db["artifacts"].values() if a['risk_score'] > 0.6)
    
    # Traitements
    total_treatments = len(fake_db["treatments"])
    
    # Coût total
    total_cost = sum(t.get('cost_eur', 0) for t in fake_db["treatments"])
    
    return {
        "total_artifacts": total_artifacts,
        "conservation_states": states_count,
        "artifacts_at_risk": at_risk,
        "total_treatments_applied": total_treatments,
        "total_preservation_cost_eur": float(total_cost),
        "active_preservation_plans": len(fake_db["preservation_plans"]),
        "climate_alerts_today": len([l for l in fake_db["climate_logs"] 
                                     if (datetime.now() - l.get('timestamp', datetime.now())).days == 0])
    }

@app.get("/stats/artifact/{artifact_id}", tags=["Statistics"])
async def get_artifact_stats(
    artifact_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Statistiques artefact"""
    if artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    artifact = fake_db["artifacts"][artifact_id]
    
    # Historique traitements
    treatments = [t for t in fake_db["treatments"] if t["artifact_id"] == artifact_id]
    
    # Analyses
    analyses = [a for a in fake_db["degradation_analyses"] if a.get("artifact_id") == artifact_id]
    
    # Climate logs
    climate_logs = [l for l in fake_db["climate_logs"] if l["artifact_id"] == artifact_id]
    
    return {
        "artifact_id": artifact_id,
        "artifact_name": artifact["name"],
        "current_state": artifact["conservation_state"],
        "risk_score": artifact["risk_score"],
        "total_treatments": len(treatments),
        "total_analyses": len(analyses),
        "climate_measurements": len(climate_logs),
        "last_inspection": artifact["last_inspection"].isoformat(),
        "requires_intervention": artifact["requires_intervention"]
    }

# ==================== ENDPOINTS DIGITIZATION ====================

@app.post("/digitization/scan", tags=["Digitization"])
async def digitize_artifact(
    artifact_id: str,
    resolution_dpi: int = 600,
    color_depth_bits: int = 48,
    format: str = "TIFF",
    current_user: User = Depends(get_current_active_user)
):
    """Numériser artefact"""
    if artifact_id not in fake_db["artifacts"]:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    artifact = fake_db["artifacts"][artifact_id]
    
    # Calculer taille fichier
    dimensions = artifact.get('dimensions', {'height': 30, 'width': 20})
    width_px = int(dimensions['width'] * resolution_dpi / 2.54)  # cm to inches to pixels
    height_px = int(dimensions['height'] * resolution_dpi / 2.54)
    
    file_size_mb = (width_px * height_px * color_depth_bits / 8) / (1024 * 1024)
    
    # Métadonnées
    metadata = {
        "artifact_id": artifact_id,
        "artifact_name": artifact["name"],
        "scan_date": datetime.now().isoformat(),
        "resolution_dpi": resolution_dpi,
        "dimensions_px": {"width": width_px, "height": height_px},
        "color_depth": color_depth_bits,
        "format": format,
        "file_size_mb": float(file_size_mb),
        "operator": current_user.username,
        "equipment": "High-precision flatbed scanner Model XYZ",
        "color_profile": "Adobe RGB 1998",
        "checksum_md5": "a1b2c3d4e5f6g7h8i9j0"
    }
    
    # Sauvegarder dans archives
    archive_id = str(uuid.uuid4())
    fake_db["archives"][archive_id] = {
        "archive_id": archive_id,
        "artifact_id": artifact_id,
        "type": "digital_scan",
        "metadata": metadata,
        "created_at": datetime.now()
    }
    
    return {
        "digitization_id": archive_id,
        "status": "completed",
        "metadata": metadata,
        "storage_location": f"/archives/digital/{artifact_id}/{archive_id}.{format.lower()}",
        "backup_locations": [
            "Primary NAS",
            "Cloud Storage (encrypted)",
            "Offline tape backup"
        ]
    }

@app.get("/digitization/archives", tags=["Digitization"])
async def list_digital_archives(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user)
):
    """Lister archives numériques"""
    archives = list(fake_db["archives"].values())
    
    return archives[skip:skip+limit]

# ==================== ENDPOINT RACINE ====================

@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil API"""
    return {
        "message": "🛡️ Ultra Conservation Technologies API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "authentication": "/token, /register",
            "artifacts": "/artifacts",
            "analysis": "/analysis",
            "preservation": "/preservation",
            "monitoring": "/monitoring",
            "treatment": "/treatment",
            "digitization": "/digitization",
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
        "artifacts_count": len(fake_db["artifacts"])
    }

# ==================== LANCEMENT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)