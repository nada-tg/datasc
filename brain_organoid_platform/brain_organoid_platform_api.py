"""
🧠 Brain Organoid Computing Platform - Complete API
FastAPI + SQLAlchemy + PostgreSQL + Neuroscience Computing

Installation:
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib[bcrypt] numpy scipy

Lancement:
uvicorn brain_organoid_platform_api:app --reload --host 0.0.0.0 --port 8013
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
DATABASE_URL = "postgresql://postgres:nadaprojet@localhost:5432/organoid_db"
SECRET_KEY = "brain-organoid-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Constantes biologiques
BIO_CONSTANTS = {
    'neuron_diameter_um': 20,
    'synapse_density': 10000,
    'action_potential_mv': 100,
    'resting_potential_mv': -70,
    'firing_threshold_mv': -55,
    'refractory_period_ms': 2,
    'synaptic_delay_ms': 0.5,
    'glucose_consumption_umol': 5.5,
    'oxygen_consumption_ml': 3.5,
    'neuron_growth_rate': 0.1,
    'max_organoid_size_mm': 5,
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
    role = Column(String, default="researcher")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Organoid(Base):
    __tablename__ = "organoids"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    cell_source = Column(String)
    brain_region = Column(String)
    size_mm = Column(Float)
    neuron_count = Column(Integer)
    viability = Column(Float)
    maturation_stage = Column(String)
    culture_duration_days = Column(Integer)
    oxygen_level = Column(Float)
    growth_factors = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class Neuron(Base):
    __tablename__ = "neurons"
    id = Column(Integer, primary_key=True, index=True)
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    neuron_type = Column(String)
    x_position = Column(Float)
    y_position = Column(Float)
    z_position = Column(Float)
    resting_potential = Column(Float, default=-70)
    firing_threshold = Column(Float, default=-55)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Synapse(Base):
    __tablename__ = "synapses"
    id = Column(Integer, primary_key=True, index=True)
    pre_neuron_id = Column(Integer, ForeignKey("neurons.id"))
    post_neuron_id = Column(Integer, ForeignKey("neurons.id"))
    strength = Column(Float, default=1.0)
    neurotransmitter = Column(String)
    delay_ms = Column(Float, default=0.5)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Recording(Base):
    __tablename__ = "recordings"
    id = Column(Integer, primary_key=True, index=True)
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    recording_type = Column(String)
    duration_s = Column(Float)
    n_neurons = Column(Integer)
    total_spikes = Column(Integer)
    firing_rate = Column(Float)
    data = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    experiment_type = Column(String)
    duration_min = Column(Integer)
    hypothesis = Column(Text)
    protocol = Column(Text)
    results = Column(JSON, nullable=True)
    status = Column(String, default="planned")
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

class Stimulation(Base):
    __tablename__ = "stimulations"
    id = Column(Integer, primary_key=True, index=True)
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    stim_type = Column(String)
    pattern = Column(String)
    amplitude = Column(Float)
    duration_ms = Column(Float)
    frequency_hz = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Training(Base):
    __tablename__ = "training_sessions"
    id = Column(Integer, primary_key=True, index=True)
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    task = Column(String)
    n_epochs = Column(Integer)
    final_accuracy = Column(Float)
    final_loss = Column(Float)
    learning_rule = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Pharmacology(Base):
    __tablename__ = "pharmacology"
    id = Column(Integer, primary_key=True, index=True)
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    compound = Column(String)
    concentration_um = Column(Float)
    duration_min = Column(Integer)
    effect = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Imaging(Base):
    __tablename__ = "imaging_sessions"
    id = Column(Integer, primary_key=True, index=True)
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    imaging_type = Column(String)
    indicator = Column(String, nullable=True)
    duration_s = Column(Float)
    roi_count = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Computation(Base):
    __tablename__ = "computations"
    id = Column(Integer, primary_key=True, index=True)
    organoid_id = Column(Integer, ForeignKey("organoids.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    task = Column(String)
    accuracy = Column(Float)
    execution_time_ms = Column(Float, nullable=True)
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

class OrganoidCreate(BaseModel):
    name: str
    cell_source: str
    brain_region: str
    culture_duration_days: int = 90
    oxygen_level: float = 20.0
    growth_factors: List[str] = []

class OrganoidResponse(BaseModel):
    id: int
    name: str
    cell_source: str
    brain_region: str
    size_mm: float
    neuron_count: int
    viability: float
    maturation_stage: str
    created_at: datetime
    class Config:
        from_attributes = True

class NeuronCreate(BaseModel):
    organoid_id: int
    neuron_type: str
    x_position: float = 0.0
    y_position: float = 0.0
    z_position: float = 0.0

class NeuronResponse(BaseModel):
    id: int
    organoid_id: int
    neuron_type: str
    resting_potential: float
    firing_threshold: float
    is_active: bool
    class Config:
        from_attributes = True

class SynapseCreate(BaseModel):
    pre_neuron_id: int
    post_neuron_id: int
    neurotransmitter: str = "Glutamate"
    strength: float = 1.0

class SynapseResponse(BaseModel):
    id: int
    pre_neuron_id: int
    post_neuron_id: int
    strength: float
    neurotransmitter: str
    is_active: bool
    class Config:
        from_attributes = True

class RecordingCreate(BaseModel):
    organoid_id: int
    recording_type: str
    duration_s: float

class RecordingResponse(BaseModel):
    id: int
    organoid_id: int
    recording_type: str
    duration_s: float
    total_spikes: int
    firing_rate: float
    timestamp: datetime
    class Config:
        from_attributes = True

class ExperimentCreate(BaseModel):
    name: str
    organoid_id: int
    experiment_type: str
    duration_min: int
    hypothesis: str
    protocol: str

class ExperimentResponse(BaseModel):
    id: int
    name: str
    experiment_type: str
    status: str
    created_at: datetime
    class Config:
        from_attributes = True

class StimulationCreate(BaseModel):
    organoid_id: int
    stim_type: str
    pattern: str
    amplitude: float
    duration_ms: float
    frequency_hz: Optional[float] = None

class StimulationResponse(BaseModel):
    id: int
    organoid_id: int
    stim_type: str
    pattern: str
    amplitude: float
    timestamp: datetime
    class Config:
        from_attributes = True

class TrainingCreate(BaseModel):
    organoid_id: int
    task: str
    n_epochs: int
    learning_rule: str = "STDP"

class TrainingResponse(BaseModel):
    id: int
    organoid_id: int
    task: str
    n_epochs: int
    final_accuracy: float
    timestamp: datetime
    class Config:
        from_attributes = True

class PharmacologyCreate(BaseModel):
    organoid_id: int
    compound: str
    concentration_um: float
    duration_min: int

class PharmacologyResponse(BaseModel):
    id: int
    organoid_id: int
    compound: str
    concentration_um: float
    timestamp: datetime
    class Config:
        from_attributes = True

class ImagingCreate(BaseModel):
    organoid_id: int
    imaging_type: str
    duration_s: float
    indicator: Optional[str] = None
    roi_count: Optional[int] = None

class ImagingResponse(BaseModel):
    id: int
    organoid_id: int
    imaging_type: str
    duration_s: float
    timestamp: datetime
    class Config:
        from_attributes = True

class ComputationCreate(BaseModel):
    organoid_id: int
    task: str

class ComputationResponse(BaseModel):
    id: int
    organoid_id: int
    task: str
    accuracy: float
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
def calculate_neuron_count(organoid_size_mm: float) -> int:
    volume = (4/3) * np.pi * (organoid_size_mm/2)**3
    return int(volume * 100000)

def simulate_action_potential():
    t = np.linspace(0, 5, 1000)
    V = np.where(t < 1, -70 + 170 * (t / 1),
                 np.where((t >= 1) & (t < 3), 100 - 150 * ((t - 1) / 2),
                         np.where((t >= 3) & (t < 4), -50 - 30 * ((t - 3) / 1), -70)))
    return t.tolist(), V.tolist()

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="🧠 Brain Organoid Computing API",
    description="API complète pour recherche en organoïdes cérébraux et biocomputing",
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

# ==================== ORGANOIDS ====================
@app.post("/organoids", response_model=OrganoidResponse, status_code=201, tags=["Organoids"])
async def create_organoid(
    organoid: OrganoidCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer organoïde cérébral"""
    if db.query(Organoid).filter(Organoid.name == organoid.name).first():
        raise HTTPException(status_code=400, detail="Organoid name exists")
    
    expected_size = min(organoid.culture_duration_days * 0.01, BIO_CONSTANTS['max_organoid_size_mm'])
    neuron_count = calculate_neuron_count(expected_size)
    
    db_organoid = Organoid(
        name=organoid.name,
        user_id=current_user.id,
        cell_source=organoid.cell_source,
        brain_region=organoid.brain_region,
        size_mm=expected_size,
        neuron_count=neuron_count,
        viability=np.random.uniform(85, 98),
        maturation_stage='Early' if organoid.culture_duration_days < 60 else 'Intermediate' if organoid.culture_duration_days < 120 else 'Mature',
        culture_duration_days=organoid.culture_duration_days,
        oxygen_level=organoid.oxygen_level,
        growth_factors=organoid.growth_factors
    )
    db.add(db_organoid)
    db.commit()
    db.refresh(db_organoid)
    logger.info(f"Organoid created: {organoid.name}")
    return db_organoid

@app.get("/organoids", response_model=List[OrganoidResponse], tags=["Organoids"])
async def list_organoids(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste organoïdes"""
    return db.query(Organoid).filter(Organoid.user_id == current_user.id).offset(skip).limit(limit).all()

@app.get("/organoids/{organoid_id}", response_model=OrganoidResponse, tags=["Organoids"])
async def get_organoid(
    organoid_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Détails organoïde"""
    organoid = db.query(Organoid).filter(
        Organoid.id == organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    return organoid

@app.put("/organoids/{organoid_id}", response_model=OrganoidResponse, tags=["Organoids"])
async def update_organoid(
    organoid_id: int,
    viability: Optional[float] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour organoïde"""
    organoid = db.query(Organoid).filter(
        Organoid.id == organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    if viability is not None:
        organoid.viability = viability
    organoid.last_updated = datetime.utcnow()
    
    db.commit()
    db.refresh(organoid)
    return organoid

@app.delete("/organoids/{organoid_id}", status_code=204, tags=["Organoids"])
async def delete_organoid(
    organoid_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Supprimer organoïde"""
    organoid = db.query(Organoid).filter(
        Organoid.id == organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    db.delete(organoid)
    db.commit()
    logger.info(f"Organoid deleted: {organoid.name}")
    return None

# ==================== NEURONS ====================
@app.post("/neurons", response_model=NeuronResponse, status_code=201, tags=["Neurons"])
async def create_neuron(
    neuron: NeuronCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer neurone"""
    organoid = db.query(Organoid).filter(
        Organoid.id == neuron.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    db_neuron = Neuron(
        organoid_id=neuron.organoid_id,
        neuron_type=neuron.neuron_type,
        x_position=neuron.x_position,
        y_position=neuron.y_position,
        z_position=neuron.z_position
    )
    db.add(db_neuron)
    db.commit()
    db.refresh(db_neuron)
    logger.info(f"Neuron created: {neuron.neuron_type}")
    return db_neuron

@app.get("/neurons", response_model=List[NeuronResponse], tags=["Neurons"])
async def list_neurons(
    organoid_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste neurones d'un organoïde"""
    organoid = db.query(Organoid).filter(
        Organoid.id == organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    return db.query(Neuron).filter(Neuron.organoid_id == organoid_id).offset(skip).limit(limit).all()

@app.post("/neurons/{neuron_id}/fire", tags=["Neurons"])
async def fire_neuron(
    neuron_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simuler potentiel d'action"""
    neuron = db.query(Neuron).join(Organoid).filter(
        Neuron.id == neuron_id,
        Organoid.user_id == current_user.id
    ).first()
    if not neuron:
        raise HTTPException(status_code=404, detail="Neuron not found")
    
    t, V = simulate_action_potential()
    
    return {
        "neuron_id": neuron_id,
        "action_potential": {"time_ms": t, "voltage_mv": V},
        "duration_ms": 5,
        "amplitude_mv": 110
    }

# ==================== SYNAPSES ====================
@app.post("/synapses", response_model=SynapseResponse, status_code=201, tags=["Synapses"])
async def create_synapse(
    synapse: SynapseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer synapse"""
    # Vérifier neurones
    pre_neuron = db.query(Neuron).join(Organoid).filter(
        Neuron.id == synapse.pre_neuron_id,
        Organoid.user_id == current_user.id
    ).first()
    post_neuron = db.query(Neuron).join(Organoid).filter(
        Neuron.id == synapse.post_neuron_id,
        Organoid.user_id == current_user.id
    ).first()
    
    if not pre_neuron or not post_neuron:
        raise HTTPException(status_code=404, detail="Neuron not found")
    
    db_synapse = Synapse(
        pre_neuron_id=synapse.pre_neuron_id,
        post_neuron_id=synapse.post_neuron_id,
        neurotransmitter=synapse.neurotransmitter,
        strength=synapse.strength
    )
    db.add(db_synapse)
    db.commit()
    db.refresh(db_synapse)
    logger.info(f"Synapse created: {synapse.pre_neuron_id} -> {synapse.post_neuron_id}")
    return db_synapse

@app.get("/synapses", response_model=List[SynapseResponse], tags=["Synapses"])
async def list_synapses(
    neuron_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste synapses"""
    query = db.query(Synapse).join(Neuron, Synapse.pre_neuron_id == Neuron.id).join(Organoid).filter(
        Organoid.user_id == current_user.id
    )
    
    if neuron_id:
        query = query.filter(
            (Synapse.pre_neuron_id == neuron_id) | (Synapse.post_neuron_id == neuron_id)
        )
    
    return query.offset(skip).limit(limit).all()

@app.put("/synapses/{synapse_id}/strength", response_model=SynapseResponse, tags=["Synapses"])
async def update_synapse_strength(
    synapse_id: int,
    strength: float,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Modifier force synaptique (plasticité)"""
    synapse = db.query(Synapse).join(Neuron, Synapse.pre_neuron_id == Neuron.id).join(Organoid).filter(
        Synapse.id == synapse_id,
        Organoid.user_id == current_user.id
    ).first()
    if not synapse:
        raise HTTPException(status_code=404, detail="Synapse not found")
    
    synapse.strength = max(0, min(2, strength))  # Limiter 0-2
    db.commit()
    db.refresh(synapse)
    return synapse

# ==================== RECORDINGS ====================
@app.post("/recordings", response_model=RecordingResponse, status_code=201, tags=["Recordings"])
async def create_recording(
    recording: RecordingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enregistrer activité neuronale"""
    organoid = db.query(Organoid).filter(
        Organoid.id == recording.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    # Simuler enregistrement
    n_neurons_sample = min(100, organoid.neuron_count)
    total_spikes = int(np.random.poisson(n_neurons_sample * recording.duration_s * 5))
    firing_rate = total_spikes / (n_neurons_sample * recording.duration_s)
    
    db_recording = Recording(
        organoid_id=recording.organoid_id,
        user_id=current_user.id,
        recording_type=recording.recording_type,
        duration_s=recording.duration_s,
        n_neurons=n_neurons_sample,
        total_spikes=total_spikes,
        firing_rate=firing_rate
    )
    db.add(db_recording)
    db.commit()
    db.refresh(db_recording)
    logger.info(f"Recording created: {recording.recording_type}")
    return db_recording

@app.get("/recordings", response_model=List[RecordingResponse], tags=["Recordings"])
async def list_recordings(
    organoid_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste enregistrements"""
    query = db.query(Recording).filter(Recording.user_id == current_user.id)
    
    if organoid_id:
        query = query.filter(Recording.organoid_id == organoid_id)
    
    return query.offset(skip).limit(limit).all()

@app.get("/recordings/{recording_id}", response_model=RecordingResponse, tags=["Recordings"])
async def get_recording(
    recording_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Détails enregistrement"""
    recording = db.query(Recording).filter(
        Recording.id == recording_id,
        Recording.user_id == current_user.id
    ).first()
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    return recording

# ==================== EXPERIMENTS ====================
@app.post("/experiments", response_model=ExperimentResponse, status_code=201, tags=["Experiments"])
async def create_experiment(
    experiment: ExperimentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer expérience"""
    organoid = db.query(Organoid).filter(
        Organoid.id == experiment.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    db_experiment = Experiment(
        name=experiment.name,
        user_id=current_user.id,
        organoid_id=experiment.organoid_id,
        experiment_type=experiment.experiment_type,
        duration_min=experiment.duration_min,
        hypothesis=experiment.hypothesis,
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
    
    # Simuler résultats
    results = {
        "success": True,
        "measurements": np.random.randint(100, 1000),
        "average_response": float(np.random.uniform(50, 150))
    }
    
    experiment.status = "completed"
    experiment.results = results
    experiment.executed_at = datetime.utcnow()
    
    db.commit()
    db.refresh(experiment)
    logger.info(f"Experiment executed: {experiment.name}")
    return experiment

# ==================== STIMULATIONS ====================
@app.post("/stimulations", response_model=StimulationResponse, status_code=201, tags=["Stimulations"])
async def create_stimulation(
    stimulation: StimulationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Appliquer stimulation"""
    organoid = db.query(Organoid).filter(
        Organoid.id == stimulation.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    db_stimulation = Stimulation(
        organoid_id=stimulation.organoid_id,
        user_id=current_user.id,
        stim_type=stimulation.stim_type,
        pattern=stimulation.pattern,
        amplitude=stimulation.amplitude,
        duration_ms=stimulation.duration_ms,
        frequency_hz=stimulation.frequency_hz
    )
    db.add(db_stimulation)
    db.commit()
    db.refresh(db_stimulation)
    logger.info(f"Stimulation applied: {stimulation.stim_type}")
    return db_stimulation

@app.get("/stimulations", response_model=List[StimulationResponse], tags=["Stimulations"])
async def list_stimulations(
    organoid_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste stimulations"""
    query = db.query(Stimulation).filter(Stimulation.user_id == current_user.id)
    
    if organoid_id:
        query = query.filter(Stimulation.organoid_id == organoid_id)
    
    return query.offset(skip).limit(limit).all()

# ==================== TRAINING ====================
@app.post("/training", response_model=TrainingResponse, status_code=201, tags=["Training"])
async def create_training(
    training: TrainingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Entraîner organoïde"""
    organoid = db.query(Organoid).filter(
        Organoid.id == training.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    # Simuler entraînement
    final_accuracy = float(100 * (1 - np.exp(-training.n_epochs/50)) + np.random.normal(0, 2))
    final_accuracy = max(0, min(100, final_accuracy))
    final_loss = float(100 * np.exp(-training.n_epochs/50) + np.random.normal(0, 5))
    
    db_training = Training(
        organoid_id=training.organoid_id,
        user_id=current_user.id,
        task=training.task,
        n_epochs=training.n_epochs,
        final_accuracy=final_accuracy,
        final_loss=final_loss,
        learning_rule=training.learning_rule
    )
    db.add(db_training)
    db.commit()
    db.refresh(db_training)
    logger.info(f"Training completed: {training.task}")
    return db_training

@app.get("/training", response_model=List[TrainingResponse], tags=["Training"])
async def list_training_sessions(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste sessions entraînement"""
    return db.query(Training).filter(Training.user_id == current_user.id).offset(skip).limit(limit).all()

@app.get("/training/performance", tags=["Training"])
async def get_training_performance(
    organoid_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Performance entraînement organoïde"""
    trainings = db.query(Training).filter(
        Training.organoid_id == organoid_id,
        Training.user_id == current_user.id
    ).all()
    
    if not trainings:
        raise HTTPException(status_code=404, detail="No training sessions found")
    
    avg_accuracy = np.mean([t.final_accuracy for t in trainings])
    best_accuracy = max([t.final_accuracy for t in trainings])
    total_epochs = sum([t.n_epochs for t in trainings])
    
    return {
        "organoid_id": organoid_id,
        "total_sessions": len(trainings),
        "total_epochs": total_epochs,
        "average_accuracy": float(avg_accuracy),
        "best_accuracy": float(best_accuracy),
        "tasks": list(set([t.task for t in trainings]))
    }

# ==================== PHARMACOLOGY ====================
@app.post("/pharmacology", response_model=PharmacologyResponse, status_code=201, tags=["Pharmacology"])
async def apply_compound(
    pharmacology: PharmacologyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Appliquer composé pharmacologique"""
    organoid = db.query(Organoid).filter(
        Organoid.id == pharmacology.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    # Déterminer effet
    excitatory = ["Glutamate", "Dopamine"]
    inhibitory = ["GABA", "Bicuculline"]
    
    if any(comp in pharmacology.compound for comp in excitatory):
        effect = "+40%"
    elif any(comp in pharmacology.compound for comp in inhibitory):
        effect = "-35%"
    else:
        effect = "Modulation"
    
    db_pharmacology = Pharmacology(
        organoid_id=pharmacology.organoid_id,
        user_id=current_user.id,
        compound=pharmacology.compound,
        concentration_um=pharmacology.concentration_um,
        duration_min=pharmacology.duration_min,
        effect=effect
    )
    db.add(db_pharmacology)
    db.commit()
    db.refresh(db_pharmacology)
    logger.info(f"Compound applied: {pharmacology.compound}")
    return db_pharmacology

@app.get("/pharmacology", response_model=List[PharmacologyResponse], tags=["Pharmacology"])
async def list_pharmacology(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste applications pharmacologiques"""
    return db.query(Pharmacology).filter(Pharmacology.user_id == current_user.id).offset(skip).limit(limit).all()

# ==================== IMAGING ====================
@app.post("/imaging", response_model=ImagingResponse, status_code=201, tags=["Imaging"])
async def create_imaging_session(
    imaging: ImagingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Session d'imagerie"""
    organoid = db.query(Organoid).filter(
        Organoid.id == imaging.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    db_imaging = Imaging(
        organoid_id=imaging.organoid_id,
        user_id=current_user.id,
        imaging_type=imaging.imaging_type,
        indicator=imaging.indicator,
        duration_s=imaging.duration_s,
        roi_count=imaging.roi_count
    )
    db.add(db_imaging)
    db.commit()
    db.refresh(db_imaging)
    logger.info(f"Imaging session: {imaging.imaging_type}")
    return db_imaging

@app.get("/imaging", response_model=List[ImagingResponse], tags=["Imaging"])
async def list_imaging_sessions(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste sessions imagerie"""
    return db.query(Imaging).filter(Imaging.user_id == current_user.id).offset(skip).limit(limit).all()

# ==================== COMPUTATIONS ====================
@app.post("/computations", response_model=ComputationResponse, status_code=201, tags=["Biocomputing"])
async def perform_computation(
    computation: ComputationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Effectuer calcul biocomputing"""
    organoid = db.query(Organoid).filter(
        Organoid.id == computation.organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    # Simuler calcul
    if computation.task == "Fonction XOR":
        accuracy = float(np.random.uniform(75, 100))
    elif computation.task == "Classification Images":
        accuracy = float(np.random.uniform(70, 95))
    else:
        accuracy = float(np.random.uniform(60, 90))
    
    execution_time = float(np.random.uniform(10, 500))
    
    db_computation = Computation(
        organoid_id=computation.organoid_id,
        user_id=current_user.id,
        task=computation.task,
        accuracy=accuracy,
        execution_time_ms=execution_time
    )
    db.add(db_computation)
    db.commit()
    db.refresh(db_computation)
    logger.info(f"Computation performed: {computation.task}")
    return db_computation

@app.get("/computations", response_model=List[ComputationResponse], tags=["Biocomputing"])
async def list_computations(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Liste calculs effectués"""
    return db.query(Computation).filter(Computation.user_id == current_user.id).offset(skip).limit(limit).all()

@app.get("/computations/benchmarks", tags=["Biocomputing"])
async def get_benchmarks(
    organoid_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Benchmarks biocomputing"""
    query = db.query(Computation).filter(Computation.user_id == current_user.id)
    
    if organoid_id:
        query = query.filter(Computation.organoid_id == organoid_id)
    
    computations = query.all()
    
    if not computations:
        return {"message": "No computations found"}
    
    tasks = {}
    for comp in computations:
        if comp.task not in tasks:
            tasks[comp.task] = []
        tasks[comp.task].append(comp.accuracy)
    
    benchmarks = {}
    for task, accuracies in tasks.items():
        benchmarks[task] = {
            "count": len(accuracies),
            "average_accuracy": float(np.mean(accuracies)),
            "best_accuracy": float(max(accuracies)),
            "worst_accuracy": float(min(accuracies))
        }
    
    return {
        "total_computations": len(computations),
        "tasks": benchmarks,
        "organoid_id": organoid_id
    }

# ==================== ANALYTICS ====================
@app.get("/analytics/overview", tags=["Analytics"])
async def get_analytics_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Vue d'ensemble analytics"""
    organoids = db.query(Organoid).filter(Organoid.user_id == current_user.id).all()
    experiments = db.query(Experiment).filter(Experiment.user_id == current_user.id).count()
    recordings = db.query(Recording).filter(Recording.user_id == current_user.id).count()
    stimulations = db.query(Stimulation).filter(Stimulation.user_id == current_user.id).count()
    
    total_neurons = sum([org.neuron_count for org in organoids])
    total_synapses = total_neurons * BIO_CONSTANTS['synapse_density']
    
    return {
        "organoids": len(organoids),
        "total_neurons": total_neurons,
        "total_synapses": total_synapses,
        "experiments": experiments,
        "recordings": recordings,
        "stimulations": stimulations,
        "average_viability": float(np.mean([org.viability for org in organoids])) if organoids else 0
    }

@app.get("/analytics/organoids/comparison", tags=["Analytics"])
async def compare_organoids(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Comparer organoïdes"""
    organoids = db.query(Organoid).filter(Organoid.user_id == current_user.id).all()
    
    if len(organoids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 organoids to compare")
    
    comparison = []
    for org in organoids:
        comparison.append({
            "name": org.name,
            "size_mm": org.size_mm,
            "neuron_count": org.neuron_count,
            "viability": org.viability,
            "age_days": org.culture_duration_days,
            "maturation": org.maturation_stage
        })
    
    return {
        "organoids": comparison,
        "total": len(organoids)
    }

@app.get("/analytics/activity", tags=["Analytics"])
async def get_activity_stats(
    organoid_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Statistiques activité organoïde"""
    organoid = db.query(Organoid).filter(
        Organoid.id == organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    recordings = db.query(Recording).filter(Recording.organoid_id == organoid_id).all()
    stimulations = db.query(Stimulation).filter(Stimulation.organoid_id == organoid_id).count()
    trainings = db.query(Training).filter(Training.organoid_id == organoid_id).count()
    
    total_spikes = sum([rec.total_spikes for rec in recordings])
    avg_firing_rate = np.mean([rec.firing_rate for rec in recordings]) if recordings else 0
    
    return {
        "organoid_id": organoid_id,
        "organoid_name": organoid.name,
        "recordings": len(recordings),
        "total_spikes": total_spikes,
        "average_firing_rate": float(avg_firing_rate),
        "stimulations": stimulations,
        "training_sessions": trainings
    }

# ==================== BIOLOGY CALCULATIONS ====================
@app.get("/biology/metabolism", tags=["Biology"])
async def calculate_metabolism(
    organoid_id: int,
    firing_rate_hz: float = 5.0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Calculer métabolisme organoïde"""
    organoid = db.query(Organoid).filter(
        Organoid.id == organoid_id,
        Organoid.user_id == current_user.id
    ).first()
    if not organoid:
        raise HTTPException(status_code=404, detail="Organoid not found")
    
    n_neurons = organoid.neuron_count
    
    # Consommation basale
    glucose_base = BIO_CONSTANTS['glucose_consumption_umol'] * (n_neurons / 1e6)
    oxygen_base = BIO_CONSTANTS['oxygen_consumption_ml'] * (n_neurons / 1e6)
    
    # Augmentation avec activité
    activity_factor = 1 + (firing_rate_hz / 10)
    
    return {
        "organoid_id": organoid_id,
        "neuron_count": n_neurons,
        "firing_rate_hz": firing_rate_hz,
        "glucose_consumption_umol_min": float(glucose_base * activity_factor),
        "oxygen_consumption_ml_min": float(oxygen_base * activity_factor),
        "atp_production": float(glucose_base * activity_factor * 38),
        "heat_production_mw": float(n_neurons * firing_rate_hz * 0.01)
    }

@app.get("/biology/action-potential", tags=["Biology"])
async def get_action_potential():
    """Obtenir données potentiel d'action"""
    t, V = simulate_action_potential()
    
    return {
        "time_ms": t,
        "voltage_mv": V,
        "resting_potential": -70,
        "threshold": -55,
        "peak": 40,
        "duration_ms": 2
    }

# ==================== CONSTANTS & INFO ====================
@app.get("/constants", tags=["Info"])
async def get_constants():
    """Constantes biologiques"""
    return BIO_CONSTANTS

@app.get("/neuron-types", tags=["Info"])
async def get_neuron_types():
    """Types de neurones disponibles"""
    return {
        "Pyramidal": {
            "description": "Neurones excitateurs principaux",
            "percentage": 80,
            "neurotransmitter": "Glutamate"
        },
        "Interneuron": {
            "description": "Neurones inhibiteurs (GABA)",
            "percentage": 15,
            "neurotransmitter": "GABA"
        },
        "Dopaminergic": {
            "description": "Neurones dopaminergiques",
            "percentage": 3,
            "neurotransmitter": "Dopamine"
        },
        "Serotonergic": {
            "description": "Neurones sérotoninergiques",
            "percentage": 2,
            "neurotransmitter": "Serotonin"
        }
    }

# ==================== HEALTH CHECK ====================
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "🧠 Brain Organoid Computing API",
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
    uvicorn.run(app, host="0.0.0.0", port=8013, reload=True)