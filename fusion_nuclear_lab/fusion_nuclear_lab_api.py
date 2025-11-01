"""
⚛️ Nuclear Fusion Laboratory Platform - Complete API
FastAPI + SQLAlchemy + PostgreSQL + Plasma Physics

Installation:
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib[bcrypt] numpy scipy

Lancement:
uvicorn fusion_nuclear_lab_api:app --reload --host 0.0.0.0 --port 8027
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
DATABASE_URL = "postgresql://postgres:nadaprojet@localhost:5432/fusion_db"
SECRET_KEY = "fusion-nuclear-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Constantes physiques
PHYSICS_CONSTANTS = {
    'c': 299792458,
    'k_B': 1.380649e-23,
    'e': 1.602176634e-19,
    'epsilon_0': 8.8541878128e-12,
    'mu_0': 1.25663706212e-6,
    'mass_deuterium': 3.344e-27,
    'mass_tritium': 5.008e-27,
    'mass_helium': 6.646e-27,
    'mass_neutron': 1.675e-27,
    'energy_DT': 17.6,
    'lawson_criterion': 3e21,
}

# ==================== DATABASE SETUP ====================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==================== DATABASE MODELS ====================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    full_name = Column(String)
    institution = Column(String, nullable=True)
    role = Column(String, default="physicist")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Reactor(Base):
    __tablename__ = "reactors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    reactor_type = Column(String)
    fuel_type = Column(String)
    major_radius_m = Column(Float)
    minor_radius_m = Column(Float)
    aspect_ratio = Column(Float)
    toroidal_field_T = Column(Float)
    plasma_current_MA = Column(Float)
    target_density_m3 = Column(Float)
    target_temperature_keV = Column(Float)
    confinement_time_s = Column(Float)
    volume_m3 = Column(Float)
    heating_power_MW = Column(Float)
    Q_factor_est = Column(Float)
    triple_product = Column(Float)
    beta = Column(Float)
    status = Column(String, default="offline")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class PlasmaShot(Base):
    __tablename__ = "plasma_shots"
    id = Column(Integer, primary_key=True, index=True)
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    shot_number = Column(Integer)
    duration_s = Column(Float)
    ramp_up_s = Column(Float)
    flat_top_s = Column(Float)
    heating_scenario = Column(String)
    target_Q = Column(Float)
    achieved_Q = Column(Float)
    max_power_MW = Column(Float)
    total_energy_MJ = Column(Float)
    max_neutron_rate = Column(Float)
    disruption = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Diagnostic(Base):
    __tablename__ = "diagnostics"
    id = Column(Integer, primary_key=True, index=True)
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    diagnostic_type = Column(String)
    measurement = Column(String)
    data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    experiment_type = Column(String)
    duration_min = Column(Integer)
    objective = Column(Text)
    protocol = Column(Text)
    results = Column(JSON, nullable=True)
    status = Column(String, default="planned")
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

class MagneticField(Base):
    __tablename__ = "magnetic_fields"
    id = Column(Integer, primary_key=True, index=True)
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    toroidal_field_T = Column(Float)
    poloidal_field_T = Column(Float)
    q_factor = Column(Float)
    beta_N = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class HeatingSystem(Base):
    __tablename__ = "heating_systems"
    id = Column(Integer, primary_key=True, index=True)
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    method = Column(String)
    power_MW = Column(Float)
    efficiency = Column(Float)
    duration_s = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Simulation(Base):
    __tablename__ = "simulations"
    id = Column(Integer, primary_key=True, index=True)
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    simulation_type = Column(String)
    parameters = Column(JSON)
    results = Column(JSON)
    execution_time_s = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class FuelInventory(Base):
    __tablename__ = "fuel_inventory"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    fuel_type = Column(String)
    quantity_kg = Column(Float)
    location = Column(String)
    last_updated = Column(DateTime, default=datetime.utcnow)

class SafetyLog(Base):
    __tablename__ = "safety_logs"
    id = Column(Integer, primary_key=True, index=True)
    reactor_id = Column(Integer, ForeignKey("reactors.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    event_type = Column(String)
    severity = Column(String)
    description = Column(Text)
    actions_taken = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC SCHEMAS ====================
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    institution: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    institution: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ReactorCreate(BaseModel):
    name: str
    reactor_type: str
    fuel_type: str
    major_radius_m: float
    minor_radius_m: float
    toroidal_field_T: float
    plasma_current_MA: float
    target_density_m3: float
    target_temperature_keV: float
    confinement_time_s: float
    heating_power_MW: float

class ReactorResponse(BaseModel):
    id: int
    name: str
    reactor_type: str
    fuel_type: str
    major_radius_m: float
    minor_radius_m: float
    toroidal_field_T: float
    plasma_current_MA: float
    Q_factor_est: float
    triple_product: float
    status: str
    created_at: datetime
    class Config:
        from_attributes = True

class PlasmaShotCreate(BaseModel):
    reactor_id: int
    duration_s: float
    heating_scenario: str
    target_Q: float

class PlasmaShotResponse(BaseModel):
    id: int
    reactor_id: int
    shot_number: int
    duration_s: float
    achieved_Q: float
    max_power_MW: float
    total_energy_MJ: float
    disruption: bool
    timestamp: datetime
    class Config:
        from_attributes = True

class DiagnosticCreate(BaseModel):
    reactor_id: int
    diagnostic_type: str
    measurement: str

class DiagnosticResponse(BaseModel):
    id: int
    reactor_id: int
    diagnostic_type: str
    measurement: str
    timestamp: datetime
    class Config:
        from_attributes = True

class ExperimentCreate(BaseModel):
    name: str
    reactor_id: int
    experiment_type: str
    duration_min: int
    objective: str
    protocol: str

class ExperimentResponse(BaseModel):
    id: int
    name: str
    experiment_type: str
    status: str
    created_at: datetime
    class Config:
        from_attributes = True

class MagneticFieldCreate(BaseModel):
    reactor_id: int
    toroidal_field_T: float
    poloidal_field_T: float

class HeatingSystemCreate(BaseModel):
    reactor_id: int
    method: str
    power_MW: float
    duration_s: float

class SimulationCreate(BaseModel):
    reactor_id: int
    simulation_type: str
    parameters: Dict[str, Any]

class SimulationResponse(BaseModel):
    id: int
    reactor_id: int
    simulation_type: str
    execution_time_s: float
    timestamp: datetime
    class Config:
        from_attributes = True

# ==================== SECURITY ====================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# ==================== UTILITY FUNCTIONS ====================
def calculate_fusion_power(n: float, T: float, reaction: str = 'D-T') -> float:
    """Calculer puissance fusion (W/m³)"""
    T_keV = T / 1000
    
    if reaction == 'D-T':
        if T_keV < 1:
            reactivity = 1e-30
        else:
            reactivity = 1.1e-24 * (T_keV**2) / (1 + (T_keV/25)**3)
    elif reaction == 'D-D':
        reactivity = 2.33e-14 * (T_keV**(-2/3)) * np.exp(-18.76 * T_keV**(-1/3))
    else:
        reactivity = 1e-24
    
    energy_per_reaction = PHYSICS_CONSTANTS['energy_DT'] * 1.602e-13
    power_density = 0.25 * n**2 * reactivity * energy_per_reaction
    
    return power_density

def calculate_triple_product(n: float, T: float, tau: float) -> float:
    """Calculer produit triple de Lawson"""
    T_keV = T / 1000
    return n * tau * T_keV

def calculate_beta(n: float, T: float, B: float) -> float:
    """Calculer paramètre beta"""
    p_plasma = n * PHYSICS_CONSTANTS['k_B'] * T
    p_magnetic = B**2 / (2 * PHYSICS_CONSTANTS['mu_0'])
    return p_plasma / p_magnetic

def calculate_q_factor(P_fusion: float, P_heating: float) -> float:
    """Calculer facteur Q"""
    if P_heating == 0:
        return 0
    return P_fusion / P_heating

def calculate_neutron_flux(P_fusion: float, volume: float) -> float:
    """Calculer flux neutronique"""
    E_neutron = 14.1 * 1.602e-13
    n_neutrons_per_second = (0.8 * P_fusion) / E_neutron
    radius = (3 * volume / (4 * np.pi))**(1/3)
    surface = 4 * np.pi * radius**2
    return n_neutrons_per_second / surface

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="⚛️ Nuclear Fusion Laboratory API",
    description="API complète pour recherche en fusion nucléaire",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== AUTHENTICATION ====================
@app.post("/register", response_model=UserResponse, status_code=201, tags=["Auth"])
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Créer nouveau compte utilisateur"""
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        full_name=user.full_name,
        institution=user.institution
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    logger.info(f"User registered: {user.username}")
    return db_user

@app.post("/token", response_model=Token, tags=["Auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authentification JWT"""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    logger.info(f"User logged in: {user.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse, tags=["Auth"])
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Profil utilisateur"""
    return current_user

# ==================== REACTORS ====================
@app.post("/reactors", response_model=ReactorResponse, status_code=201, tags=["Reactors"])
async def create_reactor(
    reactor: ReactorCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer réacteur fusion"""
    if db.query(Reactor).filter(Reactor.name == reactor.name).first():
        raise HTTPException(status_code=400, detail="Reactor name exists")
    
    aspect_ratio = reactor.major_radius_m / reactor.minor_radius_m
    volume = 2 * np.pi**2 * reactor.major_radius_m * reactor.minor_radius_m**2
    
    P_fusion = calculate_fusion_power(
        reactor.target_density_m3, 
        reactor.target_temperature_keV * 1000, 
        reactor.fuel_type
    ) * volume
    
    Q_factor = calculate_q_factor(P_fusion, reactor.heating_power_MW * 1e6)
    triple_product = calculate_triple_product(
        reactor.target_density_m3,
        reactor.target_temperature_keV * 1000,
        reactor.confinement_time_s
    )
    beta = calculate_beta(
        reactor.target_density_m3,
        reactor.target_temperature_keV * 1000 * PHYSICS_CONSTANTS['e'],
        reactor.toroidal_field_T
    )
    
    db_reactor = Reactor(
        name=reactor.name,
        user_id=current_user.id,
        reactor_type=reactor.reactor_type,
        fuel_type=reactor.fuel_type,
        major_radius_m=reactor.major_radius_m,
        minor_radius_m=reactor.minor_radius_m,
        aspect_ratio=aspect_ratio,
        toroidal_field_T=reactor.toroidal_field_T,
        plasma_current_MA=reactor.plasma_current_MA,
        target_density_m3=reactor.target_density_m3,
        target_temperature_keV=reactor.target_temperature_keV,
        confinement_time_s=reactor.confinement_time_s,
        volume_m3=volume,
        heating_power_MW=reactor.heating_power_MW,
        Q_factor_est=Q_factor,
        triple_product=triple_product,
        beta=beta
    )
    db.add(db_reactor)
    db.commit()
    db.refresh(db_reactor)
    logger.info(f"Reactor created: {reactor.name}")
    return db_reactor

@app.get("/reactors", response_model=List[ReactorResponse], tags=["Reactors"])
async def list_reactors(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste réacteurs"""
    return db.query(Reactor).filter(Reactor.user_id == current_user.id).offset(skip).limit(limit).all()

@app.get("/reactors/{reactor_id}", response_model=ReactorResponse, tags=["Reactors"])
async def get_reactor(
    reactor_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Détails réacteur"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    return reactor

@app.put("/reactors/{reactor_id}", response_model=ReactorResponse, tags=["Reactors"])
async def update_reactor(
    reactor_id: int,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour réacteur"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    if status:
        reactor.status = status
    reactor.last_updated = datetime.utcnow()
    
    db.commit()
    db.refresh(reactor)
    return reactor

@app.delete("/reactors/{reactor_id}", status_code=204, tags=["Reactors"])
async def delete_reactor(
    reactor_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Supprimer réacteur"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    db.delete(reactor)
    db.commit()
    logger.info(f"Reactor deleted: {reactor.name}")
    return None

# ==================== PLASMA SHOTS ====================
@app.post("/plasma-shots", response_model=PlasmaShotResponse, status_code=201, tags=["Plasma"])
async def create_plasma_shot(
    shot: PlasmaShotCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lancer tir plasma"""
    reactor = db.query(Reactor).filter(
        Reactor.id == shot.reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    shot_number = db.query(PlasmaShot).filter(
        PlasmaShot.reactor_id == shot.reactor_id
    ).count() + 1
    
    ramp_up = min(shot.duration_s * 0.2, 5.0)
    flat_top = shot.duration_s - ramp_up - 2
    
    n = reactor.target_density_m3
    T = reactor.target_temperature_keV * 1000
    P_fusion = calculate_fusion_power(n, T, reactor.fuel_type) * reactor.volume_m3
    
    achieved_Q = shot.target_Q + np.random.normal(0, 0.05)
    max_power_MW = P_fusion / 1e6
    total_energy_MJ = P_fusion * flat_top / 1e6
    neutron_rate = calculate_neutron_flux(P_fusion, reactor.volume_m3)
    disruption = np.random.random() > 0.95
    
    db_shot = PlasmaShot(
        reactor_id=shot.reactor_id,
        user_id=current_user.id,
        shot_number=shot_number,
        duration_s=shot.duration_s,
        ramp_up_s=ramp_up,
        flat_top_s=flat_top,
        heating_scenario=shot.heating_scenario,
        target_Q=shot.target_Q,
        achieved_Q=achieved_Q,
        max_power_MW=max_power_MW,
        total_energy_MJ=total_energy_MJ,
        max_neutron_rate=neutron_rate,
        disruption=disruption
    )
    db.add(db_shot)
    db.commit()
    db.refresh(db_shot)
    logger.info(f"Plasma shot: #{shot_number} (Q={achieved_Q:.2f})")
    return db_shot

@app.get("/plasma-shots", response_model=List[PlasmaShotResponse], tags=["Plasma"])
async def list_plasma_shots(
    reactor_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste tirs plasma"""
    query = db.query(PlasmaShot).filter(PlasmaShot.user_id == current_user.id)
    
    if reactor_id:
        query = query.filter(PlasmaShot.reactor_id == reactor_id)
    
    return query.offset(skip).limit(limit).all()

@app.get("/plasma-shots/{shot_id}", response_model=PlasmaShotResponse, tags=["Plasma"])
async def get_plasma_shot(
    shot_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Détails tir plasma"""
    shot = db.query(PlasmaShot).filter(
        PlasmaShot.id == shot_id,
        PlasmaShot.user_id == current_user.id
    ).first()
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    return shot

# ==================== DIAGNOSTICS ====================
@app.post("/diagnostics", response_model=DiagnosticResponse, status_code=201, tags=["Diagnostics"])
async def create_diagnostic(
    diagnostic: DiagnosticCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Effectuer diagnostic"""
    reactor = db.query(Reactor).filter(
        Reactor.id == diagnostic.reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    # Simuler données diagnostic
    r = np.linspace(0, reactor.minor_radius_m, 50)
    T_e = reactor.target_temperature_keV * (1 - (r/reactor.minor_radius_m)**2)**2
    n_e = reactor.target_density_m3 * (1 - (r/reactor.minor_radius_m)**2)**1.5
    
    data = {
        'radius_m': r.tolist(),
        'T_e_keV': T_e.tolist(),
        'n_e_m3': n_e.tolist()
    }
    
    db_diagnostic = Diagnostic(
        reactor_id=diagnostic.reactor_id,
        user_id=current_user.id,
        diagnostic_type=diagnostic.diagnostic_type,
        measurement=diagnostic.measurement,
        data=data
    )
    db.add(db_diagnostic)
    db.commit()
    db.refresh(db_diagnostic)
    logger.info(f"Diagnostic: {diagnostic.diagnostic_type}")
    return db_diagnostic

@app.get("/diagnostics", response_model=List[DiagnosticResponse], tags=["Diagnostics"])
async def list_diagnostics(
    reactor_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste diagnostics"""
    query = db.query(Diagnostic).filter(Diagnostic.user_id == current_user.id)
    
    if reactor_id:
        query = query.filter(Diagnostic.reactor_id == reactor_id)
    
    return query.offset(skip).limit(limit).all()

# ==================== EXPERIMENTS ====================
@app.post("/experiments", response_model=ExperimentResponse, status_code=201, tags=["Experiments"])
async def create_experiment(
    experiment: ExperimentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer expérience"""
    reactor = db.query(Reactor).filter(
        Reactor.id == experiment.reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    db_experiment = Experiment(
        name=experiment.name,
        user_id=current_user.id,
        reactor_id=experiment.reactor_id,
        experiment_type=experiment.experiment_type,
        duration_min=experiment.duration_min,
        objective=experiment.objective,
        protocol=experiment.protocol
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    logger.info(f"Experiment created: {experiment.name}")
    return db_experiment

@app.get("/experiments", response_model=List[ExperimentResponse], tags=["Experiments"])
async def list_experiments(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste expériences"""
    return db.query(Experiment).filter(Experiment.user_id == current_user.id).offset(skip).limit(limit).all()

@app.post("/experiments/{experiment_id}/execute", response_model=ExperimentResponse, tags=["Experiments"])
async def execute_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Exécuter expérience"""
    experiment = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == current_user.id
    ).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    results = {
        "success": True,
        "measurements": int(np.random.randint(100, 1000)),
        "average_Q": float(np.random.uniform(0.5, 1.5))
    }
    
    experiment.status = "completed"
    experiment.results = results
    experiment.executed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(experiment)
    logger.info(f"Experiment executed: {experiment.name}")
    return experiment

# ==================== MAGNETIC FIELDS ====================
@app.post("/magnetic-fields", status_code=201, tags=["Magnetic Fields"])
async def configure_magnetic_field(
    mag_field: MagneticFieldCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configurer champs magnétiques"""
    reactor = db.query(Reactor).filter(
        Reactor.id == mag_field.reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    q_factor = (reactor.minor_radius_m * mag_field.toroidal_field_T) / \
               (reactor.major_radius_m * mag_field.poloidal_field_T)
    
    beta_N = calculate_beta(
        reactor.target_density_m3,
        reactor.target_temperature_keV * 1000 * PHYSICS_CONSTANTS['e'],
        mag_field.toroidal_field_T
    ) * 100 * reactor.aspect_ratio * reactor.toroidal_field_T / reactor.plasma_current_MA
    
    db_field = MagneticField(
        reactor_id=mag_field.reactor_id,
        user_id=current_user.id,
        toroidal_field_T=mag_field.toroidal_field_T,
        poloidal_field_T=mag_field.poloidal_field_T,
        q_factor=q_factor,
        beta_N=beta_N
    )
    db.add(db_field)
    db.commit()
    db.refresh(db_field)
    logger.info(f"Magnetic field configured: q={q_factor:.2f}")
    return {"q_factor": q_factor, "beta_N": beta_N}

# ==================== HEATING SYSTEMS ====================
@app.post("/heating", status_code=201, tags=["Heating"])
async def apply_heating(
    heating: HeatingSystemCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Appliquer chauffage plasma"""
    reactor = db.query(Reactor).filter(
        Reactor.id == heating.reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    efficiency = 0.4 if heating.method == "NBI" else 0.6 if heating.method == "ICRH" else 0.5
    
    db_heating = HeatingSystem(
        reactor_id=heating.reactor_id,
        user_id=current_user.id,
        method=heating.method,
        power_MW=heating.power_MW,
        efficiency=efficiency,
        duration_s=heating.duration_s
    )
    db.add(db_heating)
    db.commit()
    db.refresh(db_heating)
    logger.info(f"Heating applied: {heating.method} {heating.power_MW}MW")
    return {"method": heating.method, "efficiency": efficiency, "effective_power_MW": heating.power_MW * efficiency}

# ==================== SIMULATIONS ====================
@app.post("/simulations", response_model=SimulationResponse, status_code=201, tags=["Simulations"])
async def run_simulation(
    simulation: SimulationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Exécuter simulation"""
    reactor = db.query(Reactor).filter(
        Reactor.id == simulation.reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    import time
    start_time = time.time()
    
    # Simuler résultats
    results = {
        "plasma_evolution": "stable",
        "max_temperature_keV": float(reactor.target_temperature_keV * 1.1),
        "confinement_quality": float(np.random.uniform(0.8, 1.2))
    }
    
    execution_time = time.time() - start_time
    
    db_simulation = Simulation(
        reactor_id=simulation.reactor_id,
        user_id=current_user.id,
        simulation_type=simulation.simulation_type,
        parameters=simulation.parameters,
        results=results,
        execution_time_s=execution_time
    )
    db.add(db_simulation)
    db.commit()
    db.refresh(db_simulation)
    logger.info(f"Simulation: {simulation.simulation_type}")
    return db_simulation

@app.get("/simulations", response_model=List[SimulationResponse], tags=["Simulations"])
async def list_simulations(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste simulations"""
    return db.query(Simulation).filter(Simulation.user_id == current_user.id).offset(skip).limit(limit).all()

# ==================== ANALYTICS ====================
@app.get("/analytics/overview", tags=["Analytics"])
async def get_analytics_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Vue d'ensemble analytics"""
    reactors = db.query(Reactor).filter(Reactor.user_id == current_user.id).all()
    shots = db.query(PlasmaShot).filter(PlasmaShot.user_id == current_user.id).count()
    experiments = db.query(Experiment).filter(Experiment.user_id == current_user.id).count()
    
    avg_Q = float(np.mean([r.Q_factor_est for r in reactors])) if reactors else 0
    total_energy = db.query(PlasmaShot).filter(PlasmaShot.user_id == current_user.id).all()
    total_energy_MJ = sum([s.total_energy_MJ for s in total_energy])
    
    return {
        "reactors": len(reactors),
        "plasma_shots": shots,
        "experiments": experiments,
        "average_Q_factor": avg_Q,
        "total_energy_produced_MJ": total_energy_MJ,
        "ignition_achieved": any(r.triple_product >= PHYSICS_CONSTANTS['lawson_criterion'] for r in reactors)
    }

@app.get("/analytics/reactor/{reactor_id}/performance", tags=["Analytics"])
async def get_reactor_performance(
    reactor_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Performance réacteur"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    shots = db.query(PlasmaShot).filter(PlasmaShot.reactor_id == reactor_id).all()
    
    if not shots:
        return {
            "reactor_id": reactor_id,
            "reactor_name": reactor.name,
            "total_shots": 0,
            "message": "No shots yet"
        }
    
    avg_Q = float(np.mean([s.achieved_Q for s in shots]))
    max_Q = float(max([s.achieved_Q for s in shots]))
    total_energy = sum([s.total_energy_MJ for s in shots])
    disruption_rate = sum([1 for s in shots if s.disruption]) / len(shots)
    
    return {
        "reactor_id": reactor_id,
        "reactor_name": reactor.name,
        "total_shots": len(shots),
        "average_Q_factor": avg_Q,
        "max_Q_factor": max_Q,
        "total_energy_MJ": total_energy,
        "disruption_rate": disruption_rate,
        "lawson_product": reactor.triple_product,
        "ignition_achieved": reactor.triple_product >= PHYSICS_CONSTANTS['lawson_criterion']
    }

# ==================== PHYSICS CALCULATIONS ====================
@app.get("/physics/fusion-power", tags=["Physics"])
async def calculate_fusion_power_endpoint(
    density_m3: float,
    temperature_keV: float,
    reaction: str = "D-T",
    volume_m3: float = 1000.0
):
    """Calculer puissance fusion"""
    P_density = calculate_fusion_power(density_m3, temperature_keV * 1000, reaction)
    P_total = P_density * volume_m3
    
    return {
        "density_m3": density_m3,
        "temperature_keV": temperature_keV,
        "reaction": reaction,
        "volume_m3": volume_m3,
        "power_density_W_m3": float(P_density),
        "total_power_MW": float(P_total / 1e6),
        "neutron_flux_m2_s": float(calculate_neutron_flux(P_total, volume_m3))
    }

@app.get("/physics/lawson-criterion", tags=["Physics"])
async def check_lawson_criterion(
    density_m3: float,
    temperature_keV: float,
    confinement_time_s: float
):
    """Vérifier critère Lawson"""
    triple_product = calculate_triple_product(density_m3, temperature_keV * 1000, confinement_time_s)
    ratio = triple_product / PHYSICS_CONSTANTS['lawson_criterion']
    
    return {
        "density_m3": density_m3,
        "temperature_keV": temperature_keV,
        "confinement_time_s": confinement_time_s,
        "triple_product": float(triple_product),
        "lawson_threshold": PHYSICS_CONSTANTS['lawson_criterion'],
        "ratio": float(ratio),
        "ignition_achieved": ratio >= 1.0,
        "progress_percent": float(min(ratio * 100, 100))
    }

@app.get("/physics/beta-limit", tags=["Physics"])
async def calculate_beta_limit(
    density_m3: float,
    temperature_keV: float,
    magnetic_field_T: float,
    plasma_current_MA: float,
    aspect_ratio: float
):
    """Calculer limite beta"""
    beta = calculate_beta(density_m3, temperature_keV * 1000 * PHYSICS_CONSTANTS['e'], magnetic_field_T)
    
    # Limite Troyon: β_N < 3.5
    beta_N = beta * 100 * aspect_ratio * magnetic_field_T / plasma_current_MA
    troyon_limit = 3.5
    
    return {
        "beta": float(beta),
        "beta_percent": float(beta * 100),
        "beta_N": float(beta_N),
        "troyon_limit": troyon_limit,
        "safe": beta_N < troyon_limit,
        "margin": float(troyon_limit - beta_N)
    }

@app.get("/physics/q-factor", tags=["Physics"])
async def calculate_safety_factor(
    major_radius_m: float,
    minor_radius_m: float,
    toroidal_field_T: float,
    plasma_current_MA: float
):
    """Calculer facteur de sécurité q"""
    # q ≈ (a·B_T) / (R·B_P)
    # B_P ≈ μ₀·I_p / (2π·a)
    
    B_P = (PHYSICS_CONSTANTS['mu_0'] * plasma_current_MA * 1e6) / (2 * np.pi * minor_radius_m)
    q = (minor_radius_m * toroidal_field_T) / (major_radius_m * B_P)
    
    return {
        "major_radius_m": major_radius_m,
        "minor_radius_m": minor_radius_m,
        "aspect_ratio": major_radius_m / minor_radius_m,
        "toroidal_field_T": toroidal_field_T,
        "poloidal_field_T": float(B_P),
        "q_factor": float(q),
        "kink_stable": q > 1.0,
        "mhd_stable": q > 2.0,
        "status": "stable" if q > 2.0 else "marginally stable" if q > 1.0 else "unstable"
    }

@app.get("/physics/neutron-yield", tags=["Physics"])
async def calculate_neutron_yield(
    reactor_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Calculer production neutrons"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    P_fusion = calculate_fusion_power(
        reactor.target_density_m3,
        reactor.target_temperature_keV * 1000,
        reactor.fuel_type
    ) * reactor.volume_m3
    
    # Pour D-T: 1 neutron (14.1 MeV) par réaction
    E_neutron = 14.1 * 1.602e-13  # J
    neutron_rate = (0.8 * P_fusion) / E_neutron  # neutrons/s
    
    # Dose première paroi
    flux = calculate_neutron_flux(P_fusion, reactor.volume_m3)
    dpa_per_year = flux * 3.15e7 * 1e-28  # Displacements per atom
    
    return {
        "reactor_name": reactor.name,
        "fusion_power_MW": float(P_fusion / 1e6),
        "neutron_rate_per_s": float(neutron_rate),
        "neutron_flux_m2_s": float(flux),
        "first_wall_dpa_per_year": float(dpa_per_year),
        "tritium_breeding_required": True if reactor.fuel_type == "D-T" else False
    }

# ==================== FUEL MANAGEMENT ====================
@app.post("/fuel/inventory", status_code=201, tags=["Fuel"])
async def update_fuel_inventory(
    fuel_type: str,
    quantity_kg: float,
    location: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour inventaire combustible"""
    fuel = db.query(FuelInventory).filter(
        FuelInventory.user_id == current_user.id,
        FuelInventory.fuel_type == fuel_type
    ).first()
    
    if fuel:
        fuel.quantity_kg = quantity_kg
        fuel.location = location
        fuel.last_updated = datetime.utcnow()
    else:
        fuel = FuelInventory(
            user_id=current_user.id,
            fuel_type=fuel_type,
            quantity_kg=quantity_kg,
            location=location
        )
        db.add(fuel)
    
    db.commit()
    db.refresh(fuel)
    logger.info(f"Fuel inventory updated: {fuel_type} {quantity_kg}kg")
    return {"fuel_type": fuel_type, "quantity_kg": quantity_kg, "location": location}

@app.get("/fuel/inventory", tags=["Fuel"])
async def get_fuel_inventory(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Consulter inventaire combustible"""
    inventory = db.query(FuelInventory).filter(FuelInventory.user_id == current_user.id).all()
    
    return {
        "inventory": [
            {
                "fuel_type": item.fuel_type,
                "quantity_kg": item.quantity_kg,
                "location": item.location,
                "last_updated": item.last_updated
            }
            for item in inventory
        ],
        "total_deuterium_kg": sum([i.quantity_kg for i in inventory if i.fuel_type == "Deuterium"]),
        "total_tritium_g": sum([i.quantity_kg * 1000 for i in inventory if i.fuel_type == "Tritium"])
    }

@app.post("/fuel/consumption", tags=["Fuel"])
async def calculate_fuel_consumption(
    reactor_id: int,
    shot_duration_s: float,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Calculer consommation combustible"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    # Nombre réactions fusion
    P_fusion = calculate_fusion_power(
        reactor.target_density_m3,
        reactor.target_temperature_keV * 1000,
        reactor.fuel_type
    ) * reactor.volume_m3
    
    energy_per_reaction = PHYSICS_CONSTANTS['energy_DT'] * 1.602e-13  # J
    n_reactions = (P_fusion * shot_duration_s) / energy_per_reaction
    
    # Masse combustible (D-T)
    if reactor.fuel_type == "D-T":
        mass_D = n_reactions * PHYSICS_CONSTANTS['mass_deuterium'] * 1e6  # mg
        mass_T = n_reactions * PHYSICS_CONSTANTS['mass_tritium'] * 1e6  # mg
    else:
        mass_D = n_reactions * 2 * PHYSICS_CONSTANTS['mass_deuterium'] * 1e6
        mass_T = 0
    
    return {
        "reactor_name": reactor.name,
        "shot_duration_s": shot_duration_s,
        "fusion_reactions": float(n_reactions),
        "deuterium_consumed_mg": float(mass_D),
        "tritium_consumed_mg": float(mass_T),
        "total_consumed_mg": float(mass_D + mass_T)
    }

# ==================== SAFETY ====================
@app.post("/safety/log", status_code=201, tags=["Safety"])
async def log_safety_event(
    reactor_id: int,
    event_type: str,
    severity: str,
    description: str,
    actions_taken: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enregistrer événement sécurité"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    safety_log = SafetyLog(
        reactor_id=reactor_id,
        user_id=current_user.id,
        event_type=event_type,
        severity=severity,
        description=description,
        actions_taken=actions_taken
    )
    db.add(safety_log)
    db.commit()
    db.refresh(safety_log)
    logger.warning(f"Safety event: {severity} - {event_type}")
    return {"message": "Safety event logged", "severity": severity}

@app.get("/safety/logs", tags=["Safety"])
async def get_safety_logs(
    reactor_id: Optional[int] = None,
    severity: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Consulter logs sécurité"""
    query = db.query(SafetyLog).filter(SafetyLog.user_id == current_user.id)
    
    if reactor_id:
        query = query.filter(SafetyLog.reactor_id == reactor_id)
    if severity:
        query = query.filter(SafetyLog.severity == severity)
    
    logs = query.offset(skip).limit(limit).all()
    
    return {
        "logs": [
            {
                "id": log.id,
                "event_type": log.event_type,
                "severity": log.severity,
                "description": log.description,
                "actions_taken": log.actions_taken,
                "timestamp": log.timestamp
            }
            for log in logs
        ],
        "total": query.count()
    }

@app.get("/safety/status/{reactor_id}", tags=["Safety"])
async def get_safety_status(
    reactor_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """État sécurité réacteur"""
    reactor = db.query(Reactor).filter(
        Reactor.id == reactor_id,
        Reactor.user_id == current_user.id
    ).first()
    if not reactor:
        raise HTTPException(status_code=404, detail="Reactor not found")
    
    # Vérifications sécurité
    checks = []
    
    # Check 1: Beta limite
    beta = reactor.beta
    checks.append({
        "check": "Beta Limit",
        "status": "OK" if beta < 0.05 else "WARNING",
        "value": beta,
        "limit": 0.05
    })
    
    # Check 2: Neutron flux
    P_fusion = calculate_fusion_power(
        reactor.target_density_m3,
        reactor.target_temperature_keV * 1000,
        reactor.fuel_type
    ) * reactor.volume_m3
    
    neutron_flux = calculate_neutron_flux(P_fusion, reactor.volume_m3)
    checks.append({
        "check": "Neutron Flux",
        "status": "OK" if neutron_flux < 1e20 else "WARNING",
        "value": float(neutron_flux),
        "limit": 1e20
    })
    
    # Check 3: Q factor
    checks.append({
        "check": "Q Factor",
        "status": "OK" if reactor.Q_factor_est > 0 else "WARNING",
        "value": reactor.Q_factor_est
    })
    
    overall_status = "OK" if all(c["status"] == "OK" for c in checks) else "WARNING"
    
    return {
        "reactor_id": reactor_id,
        "reactor_name": reactor.name,
        "overall_status": overall_status,
        "checks": checks,
        "last_updated": reactor.last_updated
    }

# ==================== CONSTANTS & INFO ====================
@app.get("/constants", tags=["Info"])
async def get_constants():
    """Constantes physiques"""
    return PHYSICS_CONSTANTS

@app.get("/reactor-types", tags=["Info"])
async def get_reactor_types():
    """Types de réacteurs disponibles"""
    return {
        "Tokamak": {
            "description": "Confinement magnétique toroïdal",
            "examples": ["ITER", "JET", "SPARC"],
            "typical_Q": "0.67-10"
        },
        "Stellarator": {
            "description": "Confinement magnétique hélicoïdal",
            "examples": ["Wendelstein 7-X", "LHD"],
            "typical_Q": "0.1-1"
        },
        "Laser ICF": {
            "description": "Fusion par confinement inertiel",
            "examples": ["NIF", "LMJ"],
            "typical_Q": "1.5+"
        },
        "Z-Pinch": {
            "description": "Compression magnétique pulsée",
            "examples": ["Z Machine"],
            "typical_Q": "Variable"
        }
    }

@app.get("/fusion-reactions", tags=["Info"])
async def get_fusion_reactions():
    """Réactions de fusion disponibles"""
    return {
        "D-T": {
            "formula": "D + T → He-4 + n",
            "energy_MeV": 17.6,
            "cross_section_peak_keV": 64,
            "advantages": "Haute réactivité, basse température",
            "disadvantages": "Tritium radioactif, neutrons"
        },
        "D-D": {
            "formula": "D + D → T + p (50%) ou He-3 + n (50%)",
            "energy_MeV": 3.27,
            "cross_section_peak_keV": 1500,
            "advantages": "Deutérium abondant",
            "disadvantages": "Réactivité faible, haute température"
        },
        "D-He3": {
            "formula": "D + He-3 → He-4 + p",
            "energy_MeV": 18.3,
            "cross_section_peak_keV": 200,
            "advantages": "Aneutronique (pas de neutrons rapides)",
            "disadvantages": "He-3 rare, haute température"
        },
        "p-B11": {
            "formula": "p + B-11 → 3 He-4",
            "energy_MeV": 8.7,
            "cross_section_peak_keV": 600,
            "advantages": "Totalement aneutronique",
            "disadvantages": "Très haute température (>300 keV)"
        }
    }

# ==================== HEALTH CHECK ====================
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "⚛️ Nuclear Fusion Laboratory API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "timestamp": datetime.utcnow()
    }

# ==================== RUN ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027, reload=True)