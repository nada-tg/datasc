"""
API REST Complète - Energy Research Platform
FastAPI + SQLAlchemy + PostgreSQL

Installation:
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib bcrypt python-multipart

Lancement:
uvicorn energy_platform_api:app --reload --host 0.0.0.0 --port 8024
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
import json
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Query
# ==================== CONFIGURATION ====================
DATABASE_URL = "postgresql://postgres:nadaprojet@localhost:5432/energy_db"
# Pour SQLite (dev): DATABASE_URL = "sqlite:///./energy.db"

SECRET_KEY = "your-secret-key-change-in-production-very-important"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# ==================== DATABASE SETUP ====================
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== MODELS DATABASE ====================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    role = Column(String, default="operator")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    reactors = relationship("Reactor", back_populates="owner")
    power_plants = relationship("PowerPlant", back_populates="owner")

class Reactor(Base):
    __tablename__ = "reactors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    reactor_type = Column(String)  # Tokamak, Stellarator, etc.
    fuel_type = Column(String)  # D-T, D-D, etc.
    temperature_k = Column(Float)
    pressure_atm = Column(Float)
    magnetic_field_t = Column(Float)
    plasma_current_ma = Column(Float)
    fuel_mass_kg = Column(Float)
    confinement_time_s = Column(Float)
    heating_power_mw = Column(Float)
    fusion_power_mw = Column(Float)
    q_factor = Column(Float)
    energy_gain = Column(Float)
    ignition = Column(Boolean, default=False)
    commercial = Column(Boolean, default=False)
    ai_control = Column(Boolean, default=False)
    quantum_opt = Column(Boolean, default=False)
    status = Column(String, default="operational")
    created_at = Column(DateTime, default=datetime.utcnow)

class SmartGrid(Base):
    __tablename__ = "smart_grids"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    region = Column(String)
    nodes_count = Column(Integer)
    ai_optimized = Column(Boolean, default=False)
    efficiency = Column(Float)
    reliability = Column(Float)
    status = Column(String, default="active")
    topology = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class OptimizationJob(Base):
    __tablename__ = "optimization_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String)  # grid, reactor, portfolio
    status = Column(String, default="pending")  # pending, running, completed, failed
    parameters = Column(JSON)
    results = Column(JSON, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Material(Base):
    __tablename__ = "materials"
    
    id = Column(Integer, primary_key=True, index=True)
    formula = Column(String, index=True)
    name = Column(String, nullable=True)
    material_type = Column(String)  # superconductor, perovskite, etc.
    application = Column(String)
    properties = Column(JSON)  # efficiency, stability, etc.
    trl = Column(Integer, default=1)  # Technology Readiness Level
    cost_per_kg = Column(Float, nullable=True)
    toxicity = Column(String, nullable=True)
    discovered_at = Column(DateTime, default=datetime.utcnow)

class BioBattery(Base):
    __tablename__ = "bio_batteries"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    organic_material = Column(String)
    capacity_kwh = Column(Float)
    voltage_v = Column(Float)
    current_charge_kwh = Column(Float)
    cycles_used = Column(Integer, default=0)
    cycles_life = Column(Integer)
    efficiency = Column(Float)
    biodegradable = Column(Boolean, default=True)
    toxicity = Column(String)
    status = Column(String, default="operational")
    created_at = Column(DateTime, default=datetime.utcnow)

class QuantumSimulation(Base):
    __tablename__ = "quantum_simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_type = Column(String)  # reactor, material, catalyst
    n_qubits = Column(Integer)
    algorithm = Column(String)
    target_id = Column(Integer, nullable=True)
    target_type = Column(String, nullable=True)
    results = Column(JSON)
    fidelity = Column(Float)
    execution_time_ms = Column(Float)
    quantum_advantage = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String)  # warning, critical, info
    source = Column(String)  # reactor, grid, storage, etc.
    source_id = Column(Integer, nullable=True)
    message = Column(String)
    severity = Column(Integer)  # 1-5
    acknowledged = Column(Boolean, default=False)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

# Créer tables
Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC SCHEMAS ====================

class UserBase(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    role: str = "operator"

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ReactorBase(BaseModel):
    name: str
    reactor_type: str
    fuel_type: str
    temperature_k: float = Field(gt=0, description="Temperature in Kelvin")
    pressure_atm: float = Field(gt=0)
    magnetic_field_t: float = Field(ge=0)
    plasma_current_ma: float = Field(ge=0)
    fuel_mass_kg: float = Field(gt=0)
    confinement_time_s: float = Field(gt=0)
    heating_power_mw: float = Field(gt=0)
    ai_control: bool = False
    quantum_opt: bool = False

class ReactorCreate(ReactorBase):
    pass

class ReactorUpdate(BaseModel):
    name: Optional[str] = None
    temperature_k: Optional[float] = None
    pressure_atm: Optional[float] = None
    status: Optional[str] = None
    ai_control: Optional[bool] = None
    quantum_opt: Optional[bool] = None

class ReactorResponse(ReactorBase):
    id: int
    fusion_power_mw: float
    q_factor: float
    energy_gain: float
    ignition: bool
    commercial: bool
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class FusionExperimentBase(BaseModel):
    pulse_duration_ms: float = Field(gt=0)
    parameters: Optional[Dict[str, Any]] = {}

class FusionExperimentCreate(FusionExperimentBase):
    reactor_id: int

class FusionExperimentResponse(FusionExperimentBase):
    id: int
    reactor_id: int
    energy_produced_gwh: float
    q_factor_achieved: float
    temperature_peak_k: float
    success: bool
    timestamp: datetime
    
    class Config:
        from_attributes = True

class PowerPlantBase(BaseModel):
    name: str
    plant_type: str
    capacity_mw: float = Field(gt=0)
    location: str
    location_type: Optional[str] = None
    efficiency: float = Field(ge=0, le=1)
    capacity_factor: Optional[float] = Field(None, ge=0, le=1)
    config: Optional[Dict[str, Any]] = {}

class PowerPlantCreate(PowerPlantBase):
    pass

class PowerPlantResponse(PowerPlantBase):
    id: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ProductionRecordCreate(BaseModel):
    power_plant_id: int
    power_mw: float
    energy_mwh: float
    efficiency: Optional[float] = None
    weather_conditions: Optional[Dict[str, Any]] = None

class ProductionRecordResponse(ProductionRecordCreate):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

class StorageSystemBase(BaseModel):
    name: str
    technology: str
    capacity_mwh: float = Field(gt=0)
    power_mw: float = Field(gt=0)
    cycles_life: int = Field(gt=0)
    efficiency: float = Field(ge=0, le=1)
    location: str
    ai_managed: bool = False

class StorageSystemCreate(StorageSystemBase):
    pass

class StorageSystemResponse(StorageSystemBase):
    id: int
    current_charge_mwh: float
    cycles_used: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class StorageChargeUpdate(BaseModel):
    charge_delta_mwh: float  # Positive pour charger, négatif pour décharger

class SmartGridBase(BaseModel):
    name: str
    region: str
    nodes_count: int = Field(gt=0)
    ai_optimized: bool = False
    topology: Optional[Dict[str, Any]] = {}

class SmartGridCreate(SmartGridBase):
    pass

class SmartGridResponse(SmartGridBase):
    id: int
    efficiency: float
    reliability: float
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class OptimizationJobCreate(BaseModel):
    job_type: str
    parameters: Dict[str, Any]

class OptimizationJobResponse(BaseModel):
    id: int
    job_type: str
    status: str
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class MaterialBase(BaseModel):
    formula: str
    name: Optional[str] = None
    material_type: str
    application: str
    properties: Dict[str, Any]
    cost_per_kg: Optional[float] = None
    toxicity: Optional[str] = None

class MaterialCreate(MaterialBase):
    pass

class MaterialResponse(MaterialBase):
    id: int
    trl: int
    discovered_at: datetime
    
    class Config:
        from_attributes = True

class BioBatteryBase(BaseModel):
    name: str
    organic_material: str
    capacity_kwh: float = Field(gt=0)
    voltage_v: float = Field(gt=0)
    cycles_life: int = Field(gt=0)
    efficiency: float = Field(ge=0, le=1)
    biodegradable: bool = True
    toxicity: str

class BioBatteryCreate(BioBatteryBase):
    pass

class BioBatteryResponse(BioBatteryBase):
    id: int
    current_charge_kwh: float
    cycles_used: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class QuantumSimulationCreate(BaseModel):
    simulation_type: str
    n_qubits: int = Field(ge=2, le=100)
    algorithm: str
    target_id: Optional[int] = None
    target_type: Optional[str] = None

class QuantumSimulationResponse(QuantumSimulationCreate):
    id: int
    results: Dict[str, Any]
    fidelity: float
    execution_time_ms: float
    quantum_advantage: float
    timestamp: datetime
    
    class Config:
        from_attributes = True

class AlertCreate(BaseModel):
    alert_type: str
    source: str
    source_id: Optional[int] = None
    message: str
    severity: int = Field(ge=1, le=5)

class AlertResponse(AlertCreate):
    id: int
    acknowledged: bool
    resolved: bool
    created_at: datetime
    resolved_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    horizon_hours: int = Field(ge=1, le=720)
    features: List[str] = []

class PredictionResponse(BaseModel):
    predictions: List[float]
    confidence_intervals: List[Dict[str, float]]
    mae: float
    rmse: float
    r2_score: float

# ==================== SECURITY ====================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ==================== UTILITY FUNCTIONS ====================

def calculate_fusion_metrics(reactor: Reactor) -> Reactor:
    """Calculer métriques fusion"""
    # Q factor simplifié
    fusion_power = reactor.heating_power_mw * np.random.uniform(5, 15)
    q_factor = fusion_power / reactor.heating_power_mw
    
    reactor.fusion_power_mw = fusion_power
    reactor.q_factor = q_factor
    reactor.energy_gain = q_factor
    reactor.ignition = q_factor > 10
    reactor.commercial = q_factor > 20
    
    return reactor

def simulate_fusion_experiment(reactor: Reactor, pulse_duration_ms: float) -> Dict:
    """Simuler expérience fusion"""
    # Calcul énergie produite (simplifié)
    energy_per_kg = 17.6 * 1.602e-13 * 6.022e23 / 5  # MeV to J
    energy_j = reactor.fuel_mass_kg * energy_per_kg * reactor.efficiency if hasattr(reactor, 'efficiency') else energy_per_kg
    energy_gwh = energy_j / 3.6e12 * (pulse_duration_ms / 3600000)
    
    return {
        'energy_produced_gwh': energy_gwh,
        'q_factor_achieved': reactor.q_factor * np.random.uniform(0.9, 1.1),
        'temperature_peak_k': reactor.temperature_k * np.random.uniform(0.95, 1.05),
        'success': True
    }

def run_quantum_optimization(simulation_type: str, n_qubits: int, algorithm: str, target: Any) -> Dict:
    """Simuler optimisation quantique"""
    # Simulation
    fidelity = np.random.uniform(0.92, 0.99)
    execution_time_ms = n_qubits * np.random.uniform(10, 50)
    quantum_advantage = np.random.uniform(1.1, 1.5)
    
    results = {
        'optimal_parameters': {
            'temperature_k': getattr(target, 'temperature_k', 0) * np.random.uniform(0.95, 1.05),
            'pressure_atm': getattr(target, 'pressure_atm', 0) * np.random.uniform(0.95, 1.05),
        },
        'performance_improvement': quantum_advantage,
        'confidence': fidelity
    }
    
    return {
        'results': results,
        'fidelity': fidelity,
        'execution_time_ms': execution_time_ms,
        'quantum_advantage': quantum_advantage
    }

def run_ai_optimization(job_type: str, parameters: Dict) -> Dict:
    """Exécuter optimisation IA"""
    if job_type == "grid":
        # Optimisation smart grid
        return {
            'efficiency_gain': np.random.uniform(8, 15),
            'cost_reduction': np.random.uniform(15, 25),
            'emission_reduction': np.random.uniform(20, 35),
            'actions': [
                'Redistribuer charge vers centrales efficientes',
                'Augmenter stockage batteries 15%',
                'Prioriser production renouvelable'
            ]
        }
    elif job_type == "reactor":
        # Optimisation réacteur
        return {
            'q_factor_improvement': np.random.uniform(1.1, 1.3),
            'optimal_temperature': parameters.get('temperature_k', 150e6) * 1.05,
            'optimal_pressure': parameters.get('pressure_atm', 5) * 1.02
        }
    else:
        return {'status': 'completed', 'message': 'Optimization successful'}

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Energy Research Platform API",
    description="API REST pour plateforme recherche énergétique avancée",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production: spécifier domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ROUTES AUTHENTICATION ====================

@app.post("/register", response_model=UserResponse, tags=["Authentication"])
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Créer nouveau utilisateur"""
    # Vérifier si utilisateur existe
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Créer utilisateur
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authentification et génération token JWT"""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
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

@app.get("/users/me", response_model=UserResponse, tags=["Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Obtenir utilisateur courant"""
    return current_user

# ==================== ROUTES REACTORS ====================

@app.post("/reactors", response_model=ReactorResponse, status_code=status.HTTP_201_CREATED, tags=["Reactors"])
def create_reactor(
    reactor: ReactorCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer nouveau réacteur fusion"""
    db_reactor = Reactor(**reactor.dict(), owner_id=current_user.id)
    db_reactor = calculate_fusion_metrics(db_reactor)
    db.add(db_reactor)
    db.commit()
    db.refresh(db_reactor)
    return db_reactor

@app.get("/reactors", response_model=List[ReactorResponse], tags=["Reactors"])
def list_reactors(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister tous les réacteurs"""
    query = db.query(Reactor)
    if status:
        query = query.filter(Reactor.status == status)
    reactors = query.offset(skip).limit(limit).all()
    return reactors

@app.get("/reactors/{reactor_id}", response_model=ReactorResponse, tags=["Reactors"])
def get_reactor(
    reactor_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails réacteur"""
    reactor = db.query(Reactor).filter(Reactor.id == reactor_id).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    return reactor

@app.put("/reactors/{reactor_id}", response_model=ReactorResponse, tags=["Reactors"])
def update_reactor(
    reactor_id: int,
    reactor_update: ReactorUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour réacteur"""
    db_reactor = db.query(Reactor).filter(Reactor.id == reactor_id).first()
    if not db_reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    # Mettre à jour champs
    for field, value in reactor_update.dict(exclude_unset=True).items():
        setattr(db_reactor, field, value)
    
    # Recalculer métriques si paramètres changés
    if any(f in reactor_update.dict(exclude_unset=True) for f in ['temperature_k', 'pressure_atm']):
        db_reactor = calculate_fusion_metrics(db_reactor)
    
    db_reactor.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_reactor)
    return db_reactor

@app.delete("/reactors/{reactor_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Reactors"])
def delete_reactor(
    reactor_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Supprimer réacteur"""
    reactor = db.query(Reactor).filter(Reactor.id == reactor_id).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    db.delete(reactor)
    db.commit()
    return None

# ==================== ROUTES FUSION EXPERIMENTS ====================

@app.post("/experiments", response_model=FusionExperimentResponse, status_code=status.HTTP_201_CREATED, tags=["Experiments"])
def create_experiment(
    experiment: FusionExperimentCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lancer expérience fusion"""
    reactor = db.query(Reactor).filter(Reactor.id == experiment.reactor_id).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    # Simuler expérience
    results = simulate_fusion_experiment(reactor, experiment.pulse_duration_ms)
    
    db_experiment = FusionExperiment(
        reactor_id=experiment.reactor_id,
        pulse_duration_ms=experiment.pulse_duration_ms,
        energy_produced_gwh=results['energy_produced_gwh'],
        q_factor_achieved=results['q_factor_achieved'],
        temperature_peak_k=results['temperature_peak_k'],
        success=results['success'],
        parameters=experiment.parameters
    )
    
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    return db_experiment

@app.get("/experiments", response_model=List[FusionExperimentResponse], tags=["Experiments"])
def list_experiments(
    reactor_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister expériences"""
    query = db.query(FusionExperiment)
    if reactor_id:
        query = query.filter(FusionExperiment.reactor_id == reactor_id)
    experiments = query.order_by(FusionExperiment.timestamp.desc()).offset(skip).limit(limit).all()
    return experiments

# ==================== ROUTES POWER PLANTS ====================

@app.post("/power-plants", response_model=PowerPlantResponse, status_code=status.HTTP_201_CREATED, tags=["Power Plants"])
def create_power_plant(
    plant: PowerPlantCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer centrale électrique"""
    db_plant = PowerPlant(**plant.dict(), owner_id=current_user.id)
    db.add(db_plant)
    db.commit()
    db.refresh(db_plant)
    return db_plant

@app.get("/power-plants", response_model=List[PowerPlantResponse], tags=["Power Plants"])
def list_power_plants(
    plant_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister centrales"""
    query = db.query(PowerPlant)
    if plant_type:
        query = query.filter(PowerPlant.plant_type == plant_type)
    plants = query.offset(skip).limit(limit).all()
    return plants

@app.get("/power-plants/{plant_id}", response_model=PowerPlantResponse, tags=["Power Plants"])
def get_power_plant(
    plant_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails centrale"""
    plant = db.query(PowerPlant).filter(PowerPlant.id == plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Power plant not found")
    return plant

@app.delete("/power-plants/{plant_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Power Plants"])
def delete_power_plant(
    plant_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Supprimer centrale"""
    plant = db.query(PowerPlant).filter(PowerPlant.id == plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Power plant not found")
    
    db.delete(plant)
    db.commit()
    return None

# ==================== ROUTES PRODUCTION ====================

@app.post("/production", response_model=ProductionRecordResponse, status_code=status.HTTP_201_CREATED, tags=["Production"])
def create_production_record(
    record: ProductionRecordCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Enregistrer production"""
    db_record = ProductionRecord(**record.dict())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record

@app.get("/production", response_model=List[ProductionRecordResponse], tags=["Production"])
def list_production(
    power_plant_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 1000,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister enregistrements production"""
    query = db.query(ProductionRecord)
    
    if power_plant_id:
        query = query.filter(ProductionRecord.power_plant_id == power_plant_id)
    if start_date:
        query = query.filter(ProductionRecord.timestamp >= start_date)
    if end_date:
        query = query.filter(ProductionRecord.timestamp <= end_date)
    
    records = query.order_by(ProductionRecord.timestamp.desc()).offset(skip).limit(limit).all()
    return records

@app.get("/production/summary", tags=["Production"])
def production_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Résumé production"""
    query = db.query(ProductionRecord)
    
    if start_date:
        query = query.filter(ProductionRecord.timestamp >= start_date)
    if end_date:
        query = query.filter(ProductionRecord.timestamp <= end_date)
    
    records = query.all()
    
    if not records:
        return {
            'total_energy_mwh': 0,
            'avg_power_mw': 0,
            'peak_power_mw': 0,
            'count': 0
        }
    
    return {
        'total_energy_mwh': sum(r.energy_mwh for r in records),
        'avg_power_mw': np.mean([r.power_mw for r in records]),
        'peak_power_mw': max(r.power_mw for r in records),
        'count': len(records),
        'period': {
            'start': min(r.timestamp for r in records).isoformat(),
            'end': max(r.timestamp for r in records).isoformat()
        }
    }

# ==================== ROUTES STORAGE SYSTEMS ====================

@app.post("/storage", response_model=StorageSystemResponse, status_code=status.HTTP_201_CREATED, tags=["Storage"])
def create_storage_system(
    storage: StorageSystemCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer système stockage"""
    db_storage = StorageSystem(
        **storage.dict(),
        current_charge_mwh=storage.capacity_mwh * 0.5  # 50% charge initiale
    )
    db.add(db_storage)
    db.commit()
    db.refresh(db_storage)
    return db_storage

@app.get("/storage", response_model=List[StorageSystemResponse], tags=["Storage"])
def list_storage_systems(
    technology: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister systèmes stockage"""
    query = db.query(StorageSystem)
    if technology:
        query = query.filter(StorageSystem.technology == technology)
    systems = query.offset(skip).limit(limit).all()
    return systems

@app.get("/storage/{storage_id}", response_model=StorageSystemResponse, tags=["Storage"])
def get_storage_system(
    storage_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails système stockage"""
    storage = db.query(StorageSystem).filter(StorageSystem.id == storage_id).first()
    if not storage:
        raise HTTPException(status_code=404, detail="Storage system not found")
    return storage

@app.post("/storage/{storage_id}/charge", response_model=StorageSystemResponse, tags=["Storage"])
def update_storage_charge(
    storage_id: int,
    charge_update: StorageChargeUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Charger/décharger système stockage"""
    storage = db.query(StorageSystem).filter(StorageSystem.id == storage_id).first()
    if not storage:
        raise HTTPException(status_code=404, detail="Storage system not found")
    
    new_charge = storage.current_charge_mwh + charge_update.charge_delta_mwh
    
    # Vérifier limites
    if new_charge < 0:
        raise HTTPException(status_code=400, detail="Insufficient charge")
    if new_charge > storage.capacity_mwh:
        raise HTTPException(status_code=400, detail="Capacity exceeded")
    
    storage.current_charge_mwh = new_charge
    
    # Incrémenter cycles si décharge
    if charge_update.charge_delta_mwh < 0:
        storage.cycles_used += 1
    
    db.commit()
    db.refresh(storage)
    return storage

@app.delete("/storage/{storage_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Storage"])
def delete_storage_system(
    storage_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Supprimer système stockage"""
    storage = db.query(StorageSystem).filter(StorageSystem.id == storage_id).first()
    if not storage:
        raise HTTPException(status_code=404, detail="Storage system not found")
    
    db.delete(storage)
    db.commit()
    return None

# ==================== ROUTES SMART GRIDS ====================

@app.post("/smart-grids", response_model=SmartGridResponse, status_code=status.HTTP_201_CREATED, tags=["Smart Grids"])
def create_smart_grid(
    grid: SmartGridCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer smart grid"""
    db_grid = SmartGrid(
        **grid.dict(),
        efficiency=np.random.uniform(0.85, 0.95),
        reliability=np.random.uniform(0.95, 0.999)
    )
    db.add(db_grid)
    db.commit()
    db.refresh(db_grid)
    return db_grid

@app.get("/smart-grids", response_model=List[SmartGridResponse], tags=["Smart Grids"])
def list_smart_grids(
    region: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister smart grids"""
    query = db.query(SmartGrid)
    if region:
        query = query.filter(SmartGrid.region == region)
    grids = query.offset(skip).limit(limit).all()
    return grids

@app.get("/smart-grids/{grid_id}", response_model=SmartGridResponse, tags=["Smart Grids"])
def get_smart_grid(
    grid_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails smart grid"""
    grid = db.query(SmartGrid).filter(SmartGrid.id == grid_id).first()
    if not grid:
        raise HTTPException(status_code=404, detail="Smart grid not found")
    return grid

# ==================== ROUTES OPTIMIZATIONS ====================

@app.post("/optimizations", response_model=OptimizationJobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Optimizations"])
def create_optimization_job(
    job: OptimizationJobCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lancer job optimisation"""
    db_job = OptimizationJob(
        job_type=job.job_type,
        parameters=job.parameters,
        status="pending"
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Lancer optimisation en arrière-plan
    background_tasks.add_task(run_optimization_job, db_job.id, db)
    
    return db_job

def run_optimization_job(job_id: int, db: Session):
    """Exécuter job optimisation (background)"""
    job = db.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()
    if not job:
        return
    
    job.status = "running"
    job.started_at = datetime.utcnow()
    db.commit()
    
    try:
        # Exécuter optimisation
        results = run_ai_optimization(job.job_type, job.parameters)
        
        job.status = "completed"
        job.results = results
        job.completed_at = datetime.utcnow()
    except Exception as e:
        job.status = "failed"
        job.results = {"error": str(e)}
        job.completed_at = datetime.utcnow()
    
    db.commit()

@app.get("/optimizations", response_model=List[OptimizationJobResponse], tags=["Optimizations"])
def list_optimization_jobs(
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister jobs optimisation"""
    query = db.query(OptimizationJob)
    if job_type:
        query = query.filter(OptimizationJob.job_type == job_type)
    if status:
        query = query.filter(OptimizationJob.status == status)
    jobs = query.order_by(OptimizationJob.created_at.desc()).offset(skip).limit(limit).all()
    return jobs

@app.get("/optimizations/{job_id}", response_model=OptimizationJobResponse, tags=["Optimizations"])
def get_optimization_job(
    job_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails job optimisation"""
    job = db.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    return job

# ==================== ROUTES MATERIALS ====================

@app.post("/materials", response_model=MaterialResponse, status_code=status.HTTP_201_CREATED, tags=["Materials"])
def create_material(
    material: MaterialCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Découvrir nouveau matériau"""
    db_material = Material(**material.dict(), trl=1)
    db.add(db_material)
    db.commit()
    db.refresh(db_material)
    return db_material

@app.get("/materials", response_model=List[MaterialResponse], tags=["Materials"])
def list_materials(
    material_type: Optional[str] = None,
    application: Optional[str] = None,
    min_trl: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister matériaux"""
    query = db.query(Material)
    if material_type:
        query = query.filter(Material.material_type == material_type)
    if application:
        query = query.filter(Material.application == application)
    if min_trl:
        query = query.filter(Material.trl >= min_trl)
    materials = query.offset(skip).limit(limit).all()
    return materials

@app.get("/materials/{material_id}", response_model=MaterialResponse, tags=["Materials"])
def get_material(
    material_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails matériau"""
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    return material

@app.put("/materials/{material_id}/trl", response_model=MaterialResponse, tags=["Materials"])
class TRLUpdate(BaseModel):
    trl: int = Field(ge=1, le=9)

def update_material_trl(
    material_id: int,
    update: TRLUpdate,  # ✅ CORRECTION
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour TRL matériau"""
    material = db.query(Material).filter(Material.id == material_id).first()
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    material.trl = trl
    db.commit()
    db.refresh(material)
    return material

# ==================== ROUTES BIO-BATTERIES ====================

@app.post("/bio-batteries", response_model=BioBatteryResponse, status_code=status.HTTP_201_CREATED, tags=["Bio-Batteries"])
def create_bio_battery(
    battery: BioBatteryCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer bio-batterie"""
    db_battery = BioBattery(
        **battery.dict(),
        current_charge_kwh=battery.capacity_kwh * 0.8  # 80% charge initiale
    )
    db.add(db_battery)
    db.commit()
    db.refresh(db_battery)
    return db_battery

@app.get("/bio-batteries", response_model=List[BioBatteryResponse], tags=["Bio-Batteries"])
def list_bio_batteries(
    organic_material: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister bio-batteries"""
    query = db.query(BioBattery)
    if organic_material:
        query = query.filter(BioBattery.organic_material == organic_material)
    batteries = query.offset(skip).limit(limit).all()
    return batteries

@app.get("/bio-batteries/{battery_id}", response_model=BioBatteryResponse, tags=["Bio-Batteries"])
def get_bio_battery(
    battery_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails bio-batterie"""
    battery = db.query(BioBattery).filter(BioBattery.id == battery_id).first()
    if not battery:
        raise HTTPException(status_code=404, detail="Bio-battery not found")
    return battery

# ==================== ROUTES QUANTUM SIMULATIONS ====================

@app.post("/quantum/simulate", response_model=QuantumSimulationResponse, status_code=status.HTTP_201_CREATED, tags=["Quantum"])
def create_quantum_simulation(
    simulation: QuantumSimulationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lancer simulation quantique"""
    # Récupérer cible si spécifiée
    target = None
    if simulation.target_id and simulation.target_type:
        if simulation.target_type == "reactor":
            target = db.query(Reactor).filter(Reactor.id == simulation.target_id).first()
        elif simulation.target_type == "material":
            target = db.query(Material).filter(Material.id == simulation.target_id).first()
    
    # Exécuter simulation
    results = run_quantum_optimization(
        simulation.simulation_type,
        simulation.n_qubits,
        simulation.algorithm,
        target
    )
    
    db_simulation = QuantumSimulation(
        simulation_type=simulation.simulation_type,
        n_qubits=simulation.n_qubits,
        algorithm=simulation.algorithm,
        target_id=simulation.target_id,
        target_type=simulation.target_type,
        results=results['results'],
        fidelity=results['fidelity'],
        execution_time_ms=results['execution_time_ms'],
        quantum_advantage=results['quantum_advantage']
    )
    
    db.add(db_simulation)
    db.commit()
    db.refresh(db_simulation)
    return db_simulation

@app.get("/quantum/simulations", response_model=List[QuantumSimulationResponse], tags=["Quantum"])
def list_quantum_simulations(
    simulation_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister simulations quantiques"""
    query = db.query(QuantumSimulation)
    if simulation_type:
        query = query.filter(QuantumSimulation.simulation_type == simulation_type)
    simulations = query.order_by(QuantumSimulation.timestamp.desc()).offset(skip).limit(limit).all()
    return simulations

@app.get("/quantum/simulations/{simulation_id}", response_model=QuantumSimulationResponse, tags=["Quantum"])
def get_quantum_simulation(
    simulation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir détails simulation quantique"""
    simulation = db.query(QuantumSimulation).filter(QuantumSimulation.id == simulation_id).first()
    if not simulation:
        raise HTTPException(status_code=404, detail="Quantum simulation not found")
    return simulation

# ==================== ROUTES ALERTS ====================

@app.post("/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED, tags=["Alerts"])
def create_alert(
    alert: AlertCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer alerte"""
    db_alert = Alert(**alert.dict())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

@app.get("/alerts", response_model=List[AlertResponse], tags=["Alerts"])
def list_alerts(
    alert_type: Optional[str] = None,
    source: Optional[str] = None,
    resolved: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister alertes"""
    query = db.query(Alert)
    if alert_type:
        query = query.filter(Alert.alert_type == alert_type)
    if source:
        query = query.filter(Alert.source == source)
    if resolved is not None:
        query = query.filter(Alert.resolved == resolved)
    alerts = query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()
    return alerts

@app.put("/alerts/{alert_id}/acknowledge", response_model=AlertResponse, tags=["Alerts"])
def acknowledge_alert(
    alert_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Acquitter alerte"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    db.commit()
    db.refresh(alert)
    return alert

@app.put("/alerts/{alert_id}/resolve", response_model=AlertResponse, tags=["Alerts"])
def resolve_alert(
    alert_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Résoudre alerte"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.resolved = True
    alert.resolved_at = datetime.utcnow()
    db.commit()
    db.refresh(alert)
    return alert

# ==================== ROUTES PREDICTIONS ====================

@app.post("/predictions/demand", response_model=PredictionResponse, tags=["Predictions"])
def predict_demand(
    request: PredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Prédire demande énergétique"""
    # Simulation prédiction IA
    predictions = []
    confidence_intervals = []
    
    for h in range(request.horizon_hours):
        base = 800 + 200 * np.sin((h % 24 - 12) * np.pi / 12)
        pred = base + np.random.normal(0, 10)
        predictions.append(pred)
        
        confidence_intervals.append({
            'lower': pred - 30,
            'upper': pred + 30
        })
    
    # Métriques simulées
    mae = np.random.uniform(10, 20)
    rmse = np.random.uniform(15, 25)
    r2 = np.random.uniform(0.92, 0.98)
    
    return PredictionResponse(
        predictions=predictions,
        confidence_intervals=confidence_intervals,
        mae=mae,
        rmse=rmse,
        r2_score=r2
    )

@app.post("/predictions/production", response_model=PredictionResponse, tags=["Predictions"])
def predict_production(
    request: PredictionRequest,
    power_plant_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Prédire production centrale"""
    plant = db.query(PowerPlant).filter(PowerPlant.id == power_plant_id).first()
    if not plant:
        raise HTTPException(status_code=404, detail="Power plant not found")
    
    # Simulation prédiction selon type centrale
    predictions = []
    confidence_intervals = []
    
    for h in range(request.horizon_hours):
        if plant.plant_type == "Solar":
            # Production solaire (jour/nuit)
            hour_of_day = h % 24
            if 6 <= hour_of_day <= 18:
                base = plant.capacity_mw * np.sin((hour_of_day - 6) * np.pi / 12) * 0.8
            else:
                base = 0
        elif plant.plant_type == "Wind":
            # Production éolienne (variable)
            base = plant.capacity_mw * np.random.uniform(0.2, 0.8)
        else:
            # Production stable
            base = plant.capacity_mw * 0.9
        
        pred = base + np.random.normal(0, base * 0.05)
        predictions.append(max(0, pred))
        
        confidence_intervals.append({
            'lower': max(0, pred - base * 0.1),
            'upper': min(plant.capacity_mw, pred + base * 0.1)
        })
    
    mae = np.random.uniform(5, 15)
    rmse = np.random.uniform(8, 20)
    r2 = np.random.uniform(0.88, 0.96)
    
    return PredictionResponse(
        predictions=predictions,
        confidence_intervals=confidence_intervals,
        mae=mae,
        rmse=rmse,
        r2_score=r2
    )

# ==================== ROUTES ANALYTICS ====================

@app.get("/analytics/kpis", tags=["Analytics"])
def get_kpis(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir KPIs principaux"""
    # Compter ressources
    n_reactors = db.query(Reactor).count()
    n_plants = db.query(PowerPlant).count()
    n_storage = db.query(StorageSystem).count()
    n_grids = db.query(SmartGrid).count()
    
    # Capacités totales
    total_capacity_mw = db.query(PowerPlant).with_entities(
        db.func.sum(PowerPlant.capacity_mw)
    ).scalar() or 0
    
    total_storage_mwh = db.query(StorageSystem).with_entities(
        db.func.sum(StorageSystem.capacity_mwh)
    ).scalar() or 0
    
    # Production récente
    recent_production = db.query(ProductionRecord).filter(
        ProductionRecord.timestamp >= datetime.utcnow() - timedelta(hours=24)
    ).all()
    
    total_energy_24h = sum(r.energy_mwh for r in recent_production) if recent_production else 0
    
    return {
        'infrastructure': {
            'reactors': n_reactors,
            'power_plants': n_plants,
            'storage_systems': n_storage,
            'smart_grids': n_grids
        },
        'capacity': {
            'total_power_mw': total_capacity_mw,
            'total_storage_mwh': total_storage_mwh
        },
        'production': {
            'last_24h_mwh': total_energy_24h,
            'avg_power_mw': total_energy_24h / 24 if total_energy_24h > 0 else 0
        },
        'efficiency': {
            'global': np.random.uniform(0.85, 0.92),
            'capacity_factor': np.random.uniform(0.75, 0.90)
        },
        'emissions': {
            'co2_tonnes_per_day': np.random.uniform(50, 150),
            'reduction_vs_2020_pct': np.random.uniform(15, 30)
        }
    }

@app.get("/analytics/energy-mix", tags=["Analytics"])
def get_energy_mix(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir mix énergétique"""
    # Obtenir production par type
    plants_by_type = {}
    for plant in db.query(PowerPlant).all():
        if plant.plant_type not in plants_by_type:
            plants_by_type[plant.plant_type] = 0
        plants_by_type[plant.plant_type] += plant.capacity_mw
    
    total_capacity = sum(plants_by_type.values())
    
    mix = {}
    for plant_type, capacity in plants_by_type.items():
        mix[plant_type] = {
            'capacity_mw': capacity,
            'percentage': (capacity / total_capacity * 100) if total_capacity > 0 else 0
        }
    
    return {
        'total_capacity_mw': total_capacity,
        'mix': mix,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.get("/analytics/performance", tags=["Analytics"])
def get_performance_metrics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Métriques performance avancées"""
    return {
        'operational': {
            'availability': np.random.uniform(0.95, 0.99),
            'reliability': np.random.uniform(0.995, 0.999),
            'mtbf_hours': np.random.uniform(8000, 10000),
            'mttr_hours': np.random.uniform(3, 6),
            'oee': np.random.uniform(0.85, 0.92)
        },
        'financial': {
            'lcoe_usd_per_mwh': np.random.uniform(30, 60),
            'roi_pct': np.random.uniform(10, 15),
            'payback_period_years': np.random.uniform(6, 10),
            'npv_millions_usd': np.random.uniform(1, 5),
            'irr_pct': np.random.uniform(12, 18)
        },
        'environmental': {
            'carbon_intensity_g_per_kwh': np.random.uniform(10, 50),
            'water_usage_l_per_mwh': np.random.uniform(100, 500),
            'land_use_km2_per_gw': np.random.uniform(5, 20)
        }
    }

# ==================== ROUTES SYSTEM ====================

@app.get("/health", tags=["System"])
def health_check():
    """Vérification santé API"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }

@app.get("/stats", tags=["System"])
def get_statistics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Statistiques système"""
    return {
        'database': {
            'users': db.query(User).count(),
            'reactors': db.query(Reactor).count(),
            'power_plants': db.query(PowerPlant).count(),
            'storage_systems': db.query(StorageSystem).count(),
            'smart_grids': db.query(SmartGrid).count(),
            'experiments': db.query(FusionExperiment).count(),
            'production_records': db.query(ProductionRecord).count(),
            'materials': db.query(Material).count(),
            'bio_batteries': db.query(BioBattery).count(),
            'quantum_simulations': db.query(QuantumSimulation).count(),
            'alerts': db.query(Alert).count()
        },
        'timestamp': datetime.utcnow().isoformat()
    }

class FusionExperiment(Base):
    __tablename__ = "fusion_experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    pulse_duration_ms = Column(Float)
    energy_produced_gwh = Column(Float)
    q_factor_achieved = Column(Float)
    temperature_peak_k = Column(Float)
    success = Column(Boolean)
    parameters = Column(JSON)  # Paramètres additionnels
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    reactor = relationship("Reactor", back_populates="experiments")

class PowerPlant(Base):
    __tablename__ = "power_plants"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    plant_type = Column(String)  # Solar, Wind, Nuclear, etc.
    capacity_mw = Column(Float)
    location = Column(String)
    location_type = Column(String, nullable=True)  # Pour éolien: offshore, onshore
    efficiency = Column(Float)
    capacity_factor = Column(Float, nullable=True)
    status = Column(String, default="operational")
    config = Column(JSON)  # Configuration spécifique
    created_at = Column(DateTime, default=datetime.utcnow)
    
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="power_plants")
    production_records = relationship("ProductionRecord", back_populates="power_plant")

class ProductionRecord(Base):
    __tablename__ = "production_records"
    
    id = Column(Integer, primary_key=True, index=True)
    power_plant_id = Column(Integer, ForeignKey("power_plants.id"))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    power_mw = Column(Float)
    energy_mwh = Column(Float)
    efficiency = Column(Float, nullable=True)
    weather_conditions = Column(JSON, nullable=True)
    
    power_plant = relationship("PowerPlant", back_populates="production_records")

class StorageSystem(Base):
    __tablename__ = "storage_systems"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    technology = Column(String)  # Li-ion, Solid-State, etc.
    capacity_mwh = Column(Float)
    power_mw = Column(Float)
    current_charge_mwh = Column(Float)
    cycles_used = Column(Integer, default=0)
    cycles_life = Column(Integer)
    efficiency = Column(Float)
    location = Column(String)
    ai_managed = Column(Boolean, default=False)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="reactors")
    experiments = relationship("FusionExperiment", back_populates="reactor")

    # ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    # Créer utilisateur admin par défaut
    db = SessionLocal()
    admin_user = db.query(User).filter(User.username == "admin").first()
    if not admin_user:
        admin_user = User(
            username="admin",
            email="admin@energy-platform.com",
            full_name="Administrator",
            role="admin",
            hashed_password=get_password_hash("admin123"),
            is_active=True
        )
        db.add(admin_user)
        db.commit()
        print("✅ Admin user created: username=admin, password=admin123")
    db.close()
    
    print("🚀 Starting Energy Research Platform API...")
    print("📖 Documentation: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8024, reload=True)