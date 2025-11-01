"""
⚛️ Quantum Physics Research Platform - Complete API
FastAPI + SQLAlchemy + PostgreSQL + Quantum Computing Libraries

Installation:
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib[bcrypt] numpy scipy

Lancement:
uvicorn quantum_physics_platform_api:app --reload --host 0.0.0.0 --port 8039
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
import json
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
DATABASE_URL = "postgresql://postgres:nadaprojet@localhost:5432/quantum_db"
SECRET_KEY = "quantum-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Constantes physiques
PHYSICAL_CONSTANTS = {
    'c': 299792458,
    'h': 6.62607015e-34,
    'hbar': 1.054571817e-34,
    'G': 6.67430e-11,
    'k_B': 1.380649e-23,
    'e': 1.602176634e-19,
    'm_e': 9.1093837015e-31,
    'm_p': 1.67262192369e-27,
    'planck_length': 1.616255e-35,
    'planck_time': 5.391247e-44,
    'planck_mass': 2.176434e-8,
    'planck_energy': 1.956e9,
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
    role = Column(String, default="researcher")
    institution = Column(String, nullable=True)
    specialization = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Qubit(Base):
    __tablename__ = "qubits"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    state_real_0 = Column(Float)
    state_imag_0 = Column(Float)
    state_real_1 = Column(Float)
    state_imag_1 = Column(Float)
    bloch_x = Column(Float)
    bloch_y = Column(Float)
    bloch_z = Column(Float)
    temperature_mk = Column(Float)
    coherence_time_ms = Column(Float)
    fidelity = Column(Float)
    status = Column(String, default="initialized")
    created_at = Column(DateTime, default=datetime.utcnow)

class EntangledPair(Base):
    __tablename__ = "entangled_pairs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    qubit1_id = Column(Integer, ForeignKey("qubits.id"))
    qubit2_id = Column(Integer, ForeignKey("qubits.id"))
    bell_state = Column(String)
    distance_km = Column(Float, default=0)
    entanglement_entropy = Column(Float)
    fidelity = Column(Float)
    status = Column(String, default="entangled")
    created_at = Column(DateTime, default=datetime.utcnow)

class QuantumCircuit(Base):
    __tablename__ = "quantum_circuits"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    n_qubits = Column(Integer)
    n_gates = Column(Integer)
    depth = Column(Integer)
    circuit_json = Column(JSON)
    results = Column(JSON, nullable=True)
    fidelity = Column(Float, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class QuantumNetwork(Base):
    __tablename__ = "quantum_networks"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    topology = Column(String)
    n_nodes = Column(Integer)
    channel_type = Column(String)
    qkd_enabled = Column(Boolean, default=False)
    network_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class BlackHole(Base):
    __tablename__ = "black_holes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    bh_class = Column(String)
    mass_solar = Column(Float)
    mass_kg = Column(Float)
    spin = Column(Float)
    charge = Column(Float, default=0)
    schwarzschild_radius = Column(Float)
    hawking_temperature = Column(Float)
    bekenstein_entropy = Column(Float)
    evaporation_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Singularity(Base):
    __tablename__ = "singularities"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    singularity_type = Column(String)
    mass_solar = Column(Float)
    schwarzschild_radius = Column(Float)
    hawking_temperature = Column(Float)
    bekenstein_entropy = Column(Float)
    quantum_corrections = Column(Boolean, default=False)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class TunnelingExperiment(Base):
    __tablename__ = "tunneling_experiments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    particle_type = Column(String)
    particle_energy_ev = Column(Float)
    barrier_height_ev = Column(Float)
    barrier_width_nm = Column(Float)
    tunneling_probability = Column(Float)
    reflection_coefficient = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    experiment_type = Column(String)
    n_qubits = Column(Integer)
    n_measurements = Column(Integer)
    fidelity_target = Column(Float)
    fidelity_achieved = Column(Float, nullable=True)
    result = Column(Text, nullable=True)
    hypothesis = Column(Text)
    data = Column(JSON, nullable=True)
    status = Column(String, default="planned")
    created_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)

class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, index=True)
    qubit_id = Column(Integer, ForeignKey("qubits.id"))
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)
    measurement_basis = Column(String)
    result = Column(Integer)
    probability_0 = Column(Float)
    probability_1 = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class QuantumGravitySimulation(Base):
    __tablename__ = "quantum_gravity_simulations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    simulation_type = Column(String)
    energy_scale = Column(Float)
    parameters = Column(JSON)
    results = Column(JSON, nullable=True)
    quantum_corrections = Column(Float, nullable=True)
    status = Column(String, default="initialized")
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC SCHEMAS ====================
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    institution: Optional[str] = None
    specialization: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class QubitCreate(BaseModel):
    name: str
    state_real_0: float = 1.0
    state_imag_0: float = 0.0
    state_real_1: float = 0.0
    state_imag_1: float = 0.0
    temperature_mk: float = 20.0

class QubitResponse(BaseModel):
    id: int
    name: str
    bloch_x: float
    bloch_y: float
    bloch_z: float
    coherence_time_ms: float
    fidelity: float
    status: str
    created_at: datetime
    class Config:
        from_attributes = True

class EntangledPairCreate(BaseModel):
    qubit1_id: int
    qubit2_id: int
    bell_state: str
    distance_km: float = 0

class EntangledPairResponse(BaseModel):
    id: int
    qubit1_id: int
    qubit2_id: int
    bell_state: str
    entanglement_entropy: float
    fidelity: float
    created_at: datetime
    class Config:
        from_attributes = True

class QuantumCircuitCreate(BaseModel):
    name: str
    n_qubits: int
    circuit_json: Dict[str, Any]

class QuantumCircuitResponse(BaseModel):
    id: int
    name: str
    n_qubits: int
    n_gates: int
    depth: int
    results: Optional[Dict[str, Any]]
    fidelity: Optional[float]
    created_at: datetime
    class Config:
        from_attributes = True

class QuantumNetworkCreate(BaseModel):
    name: str
    topology: str
    n_nodes: int
    channel_type: str
    qkd_enabled: bool = False
    network_data: Dict[str, Any]

class QuantumNetworkResponse(BaseModel):
    id: int
    name: str
    topology: str
    n_nodes: int
    created_at: datetime
    class Config:
        from_attributes = True

class BlackHoleCreate(BaseModel):
    name: str
    bh_class: str
    mass_solar: float
    spin: float = 0.0
    charge: float = 0.0

class BlackHoleResponse(BaseModel):
    id: int
    name: str
    bh_class: str
    mass_solar: float
    schwarzschild_radius: float
    hawking_temperature: float
    bekenstein_entropy: float
    created_at: datetime
    class Config:
        from_attributes = True

class SingularityCreate(BaseModel):
    singularity_type: str
    mass_solar: float
    quantum_corrections: bool = False
    parameters: Dict[str, Any] = {}

class SingularityResponse(BaseModel):
    id: int
    singularity_type: str
    mass_solar: float
    schwarzschild_radius: float
    hawking_temperature: float
    created_at: datetime
    class Config:
        from_attributes = True

class TunnelingExperimentCreate(BaseModel):
    particle_type: str
    particle_energy_ev: float
    barrier_height_ev: float
    barrier_width_nm: float

class TunnelingExperimentResponse(BaseModel):
    id: int
    particle_type: str
    tunneling_probability: float
    reflection_coefficient: float
    timestamp: datetime
    class Config:
        from_attributes = True

class ExperimentCreate(BaseModel):
    name: str
    experiment_type: str
    n_qubits: int
    n_measurements: int
    fidelity_target: float
    hypothesis: str

class ExperimentResponse(BaseModel):
    id: int
    name: str
    experiment_type: str
    status: str
    fidelity_achieved: Optional[float]
    result: Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True

class MeasurementCreate(BaseModel):
    qubit_id: int
    experiment_id: Optional[int] = None
    measurement_basis: str = "Z"

class MeasurementResponse(BaseModel):
    id: int
    qubit_id: int
    result: int
    probability_0: float
    probability_1: float
    timestamp: datetime
    class Config:
        from_attributes = True

class QuantumGravitySimCreate(BaseModel):
    simulation_type: str
    energy_scale: float
    parameters: Dict[str, Any]

class QuantumGravitySimResponse(BaseModel):
    id: int
    simulation_type: str
    energy_scale: float
    quantum_corrections: Optional[float]
    status: str
    created_at: datetime
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
def calculate_bloch_coordinates(alpha: complex, beta: complex) -> Tuple[float, float, float]:
    theta = 2 * np.arccos(np.clip(abs(alpha), -1, 1))
    phi = np.angle(beta) - np.angle(alpha)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return float(x), float(y), float(z)

def schwarzschild_radius(mass_kg: float) -> float:
    G, c = PHYSICAL_CONSTANTS['G'], PHYSICAL_CONSTANTS['c']
    return 2 * G * mass_kg / c**2

def hawking_temperature(mass_kg: float) -> float:
    hbar, c, k_B, G = PHYSICAL_CONSTANTS['hbar'], PHYSICAL_CONSTANTS['c'], PHYSICAL_CONSTANTS['k_B'], PHYSICAL_CONSTANTS['G']
    return (hbar * c**3) / (8 * np.pi * G * mass_kg * k_B)

def bekenstein_entropy(mass_kg: float) -> float:
    k_B, c, hbar, G = PHYSICAL_CONSTANTS['k_B'], PHYSICAL_CONSTANTS['c'], PHYSICAL_CONSTANTS['hbar'], PHYSICAL_CONSTANTS['G']
    r_s = schwarzschild_radius(mass_kg)
    A = 4 * np.pi * r_s**2
    return (k_B * c**3 * A) / (4 * G * hbar)

def calculate_tunneling_probability(barrier_height: float, barrier_width: float, 
                                   particle_energy: float, mass: float) -> float:
    hbar = PHYSICAL_CONSTANTS['hbar']
    if particle_energy >= barrier_height:
        return 1.0
    kappa = np.sqrt(2 * mass * (barrier_height - particle_energy)) / hbar
    return float(np.exp(-2 * kappa * barrier_width))

def calculate_entanglement_entropy(state: np.ndarray) -> float:
    rho = np.outer(state, np.conj(state))
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="⚛️ Quantum Physics Research API",
    description="API complète pour recherche en physique quantique",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== AUTHENTICATION ====================
@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Auth"])
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
        institution=user.institution,
        specialization=user.specialization
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

# ==================== QUBITS ====================
@app.post("/qubits", response_model=QubitResponse, status_code=201, tags=["Qubits"])
async def create_qubit(qubit: QubitCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer qubit"""
    if db.query(Qubit).filter(Qubit.name == qubit.name).first():
        raise HTTPException(status_code=400, detail="Qubit name exists")
    
    alpha = complex(qubit.state_real_0, qubit.state_imag_0)
    beta = complex(qubit.state_real_1, qubit.state_imag_1)
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha/norm, beta/norm
    
    x, y, z = calculate_bloch_coordinates(alpha, beta)
    
    db_qubit = Qubit(
        name=qubit.name,
        user_id=current_user.id,
        state_real_0=alpha.real,
        state_imag_0=alpha.imag,
        state_real_1=beta.real,
        state_imag_1=beta.imag,
        bloch_x=x, bloch_y=y, bloch_z=z,
        temperature_mk=qubit.temperature_mk,
        coherence_time_ms=np.random.uniform(0.5, 2.0),
        fidelity=np.random.uniform(0.95, 0.99)
    )
    db.add(db_qubit)
    db.commit()
    db.refresh(db_qubit)
    logger.info(f"Qubit created: {qubit.name}")
    return db_qubit

@app.get("/qubits", response_model=List[QubitResponse], tags=["Qubits"])
async def list_qubits(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste qubits"""
    return db.query(Qubit).filter(Qubit.user_id == current_user.id).offset(skip).limit(limit).all()

@app.get("/qubits/{qubit_id}", response_model=QubitResponse, tags=["Qubits"])
async def get_qubit(qubit_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Détails qubit"""
    qubit = db.query(Qubit).filter(Qubit.id == qubit_id, Qubit.user_id == current_user.id).first()
    if not qubit:
        raise HTTPException(status_code=404, detail="Qubit not found")
    return qubit

@app.delete("/qubits/{qubit_id}", status_code=204, tags=["Qubits"])
async def delete_qubit(qubit_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Supprimer qubit"""
    qubit = db.query(Qubit).filter(Qubit.id == qubit_id, Qubit.user_id == current_user.id).first()
    if not qubit:
        raise HTTPException(status_code=404, detail="Qubit not found")
    db.delete(qubit)
    db.commit()
    logger.info(f"Qubit deleted: {qubit.name}")
    return None

@app.post("/qubits/{qubit_id}/measure", tags=["Qubits"])
async def measure_qubit(qubit_id: int, basis: str = "Z", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Mesurer qubit"""
    qubit = db.query(Qubit).filter(Qubit.id == qubit_id, Qubit.user_id == current_user.id).first()
    if not qubit:
        raise HTTPException(status_code=404, detail="Qubit not found")
    
    alpha = complex(qubit.state_real_0, qubit.state_imag_0)
    prob_0 = abs(alpha)**2
    result = 0 if np.random.random() < prob_0 else 1
    
    if result == 0:
        qubit.state_real_0, qubit.state_imag_0, qubit.state_real_1, qubit.state_imag_1 = 1.0, 0.0, 0.0, 0.0
    else:
        qubit.state_real_0, qubit.state_imag_0, qubit.state_real_1, qubit.state_imag_1 = 0.0, 0.0, 1.0, 0.0
    
    qubit.status = "measured"
    db.commit()
    logger.info(f"Qubit measured: {qubit.name} = {result}")
    return {"result": result, "probability_0": float(prob_0), "probability_1": float(1-prob_0)}

@app.post("/qubits/{qubit_id}/gate/{gate_name}", tags=["Qubits"])
async def apply_gate(qubit_id: int, gate_name: str, angle: Optional[float] = None, 
                    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Appliquer porte quantique"""
    qubit = db.query(Qubit).filter(Qubit.id == qubit_id, Qubit.user_id == current_user.id).first()
    if not qubit:
        raise HTTPException(status_code=404, detail="Qubit not found")
    
    state = np.array([complex(qubit.state_real_0, qubit.state_imag_0), complex(qubit.state_real_1, qubit.state_imag_1)])
    
    gates = {
        "H": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.array([[1, 0], [0, -1]]),
        "S": np.array([[1, 0], [0, 1j]]),
        "T": np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
    }
    
    if gate_name.upper() not in gates:
        raise HTTPException(status_code=400, detail="Unknown gate")
    
    state = gates[gate_name.upper()] @ state
    qubit.state_real_0, qubit.state_imag_0 = state[0].real, state[0].imag
    qubit.state_real_1, qubit.state_imag_1 = state[1].real, state[1].imag
    x, y, z = calculate_bloch_coordinates(state[0], state[1])
    qubit.bloch_x, qubit.bloch_y, qubit.bloch_z = x, y, z
    db.commit()
    logger.info(f"Gate {gate_name} applied to {qubit.name}")
    return {"message": f"Gate {gate_name} applied", "new_state": [complex(state[0]), complex(state[1])]}

# ==================== ENTANGLEMENT ====================
@app.post("/entanglement", response_model=EntangledPairResponse, status_code=201, tags=["Entanglement"])
async def create_entangled_pair(pair: EntangledPairCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer paire intriquée"""
    q1 = db.query(Qubit).filter(Qubit.id == pair.qubit1_id, Qubit.user_id == current_user.id).first()
    q2 = db.query(Qubit).filter(Qubit.id == pair.qubit2_id, Qubit.user_id == current_user.id).first()
    
    if not q1 or not q2:
        raise HTTPException(status_code=404, detail="Qubit not found")
    
    bell_states = {
        "Φ+": np.array([1, 0, 0, 1]) / np.sqrt(2),
        "Φ-": np.array([1, 0, 0, -1]) / np.sqrt(2),
        "Ψ+": np.array([0, 1, 1, 0]) / np.sqrt(2),
        "Ψ-": np.array([0, 1, -1, 0]) / np.sqrt(2)
    }
    
    state = bell_states.get(pair.bell_state, bell_states["Φ+"])
    entropy = calculate_entanglement_entropy(state)
    
    db_pair = EntangledPair(
        user_id=current_user.id,
        qubit1_id=pair.qubit1_id,
        qubit2_id=pair.qubit2_id,
        bell_state=pair.bell_state,
        distance_km=pair.distance_km,
        entanglement_entropy=entropy,
        fidelity=np.random.uniform(0.92, 0.98)
    )
    db.add(db_pair)
    db.commit()
    db.refresh(db_pair)
    logger.info(f"Entangled pair created: {pair.bell_state}")
    return db_pair

@app.get("/entanglement", response_model=List[EntangledPairResponse], tags=["Entanglement"])
async def list_entangled_pairs(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste paires intriquées"""
    return db.query(EntangledPair).filter(EntangledPair.user_id == current_user.id).offset(skip).limit(limit).all()

@app.get("/entanglement/{pair_id}", response_model=EntangledPairResponse, tags=["Entanglement"])
async def get_entangled_pair(pair_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Détails paire intriquée"""
    pair = db.query(EntangledPair).filter(EntangledPair.id == pair_id, EntangledPair.user_id == current_user.id).first()
    if not pair:
        raise HTTPException(status_code=404, detail="Entangled pair not found")
    return pair

@app.post("/entanglement/{pair_id}/bell-test", tags=["Entanglement"])
async def bell_test(pair_id: int, n_measurements: int = 1000, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Test des inégalités de Bell"""
    pair = db.query(EntangledPair).filter(EntangledPair.id == pair_id, EntangledPair.user_id == current_user.id).first()
    if not pair:
        raise HTTPException(status_code=404, detail="Entangled pair not found")
    
    # Simulation test Bell (CHSH)
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    correlations = [-np.cos(angle) + np.random.normal(0, 0.05) for angle in angles]
    S = abs(correlations[0] - correlations[1]) + abs(correlations[2] + correlations[3])
    
    violation = S > 2.0
    
    logger.info(f"Bell test performed: S={S:.3f}, violation={violation}")
    return {
        "S_parameter": float(S),
        "bell_inequality_violated": violation,
        "classical_limit": 2.0,
        "quantum_limit": 2.828,
        "n_measurements": n_measurements,
        "correlations": [float(c) for c in correlations]
    }

@app.post("/entanglement/{pair_id}/teleport", tags=["Entanglement"])
async def quantum_teleportation(pair_id: int, qubit_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Téléportation quantique"""
    pair = db.query(EntangledPair).filter(EntangledPair.id == pair_id, EntangledPair.user_id == current_user.id).first()
    qubit = db.query(Qubit).filter(Qubit.id == qubit_id, Qubit.user_id == current_user.id).first()
    
    if not pair or not qubit:
        raise HTTPException(status_code=404, detail="Pair or qubit not found")
    
    fidelity_teleport = np.random.uniform(0.90, 0.98)
    
    logger.info(f"Quantum teleportation: qubit {qubit.name}")
    return {
        "message": "Quantum teleportation successful",
        "fidelity": float(fidelity_teleport),
        "classical_bits_sent": 2,
        "source_qubit_id": qubit_id,
        "destination_qubit_id": pair.qubit2_id
    }

# ==================== QUANTUM CIRCUITS ====================
@app.post("/circuits", response_model=QuantumCircuitResponse, status_code=201, tags=["Circuits"])
async def create_circuit(circuit: QuantumCircuitCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer circuit quantique"""
    gates = circuit.circuit_json.get("gates", [])
    n_gates = len(gates)
    depth = circuit.circuit_json.get("depth", 0)
    
    db_circuit = QuantumCircuit(
        name=circuit.name,
        user_id=current_user.id,
        n_qubits=circuit.n_qubits,
        n_gates=n_gates,
        depth=depth,
        circuit_json=circuit.circuit_json
    )
    db.add(db_circuit)
    db.commit()
    db.refresh(db_circuit)
    logger.info(f"Circuit created: {circuit.name}")
    return db_circuit

@app.get("/circuits", response_model=List[QuantumCircuitResponse], tags=["Circuits"])
async def list_circuits(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste circuits"""
    return db.query(QuantumCircuit).filter(QuantumCircuit.user_id == current_user.id).offset(skip).limit(limit).all()

@app.post("/circuits/{circuit_id}/execute", tags=["Circuits"])
async def execute_circuit(circuit_id: int, shots: int = 1024, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Exécuter circuit quantique"""
    circuit = db.query(QuantumCircuit).filter(QuantumCircuit.id == circuit_id, QuantumCircuit.user_id == current_user.id).first()
    if not circuit:
        raise HTTPException(status_code=404, detail="Circuit not found")
    
    # Simulation résultats
    n_qubits = circuit.n_qubits
    results = {}
    for i in range(2**n_qubits):
        bitstring = format(i, f'0{n_qubits}b')
        results[bitstring] = int(np.random.poisson(shots / (2**n_qubits)))
    
    execution_time = np.random.uniform(10, 100)
    fidelity = np.random.uniform(0.90, 0.98)
    
    circuit.results = results
    circuit.fidelity = fidelity
    circuit.execution_time_ms = execution_time
    db.commit()
    
    logger.info(f"Circuit executed: {circuit.name}")
    return {
        "results": results,
        "shots": shots,
        "fidelity": float(fidelity),
        "execution_time_ms": float(execution_time)
    }

# ==================== QUANTUM NETWORKS ====================
@app.post("/networks", response_model=QuantumNetworkResponse, status_code=201, tags=["Networks"])
async def create_network(network: QuantumNetworkCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer réseau quantique"""
    if db.query(QuantumNetwork).filter(QuantumNetwork.name == network.name).first():
        raise HTTPException(status_code=400, detail="Network name exists")
    
    db_network = QuantumNetwork(
        name=network.name,
        user_id=current_user.id,
        topology=network.topology,
        n_nodes=network.n_nodes,
        channel_type=network.channel_type,
        qkd_enabled=network.qkd_enabled,
        network_data=network.network_data
    )
    db.add(db_network)
    db.commit()
    db.refresh(db_network)
    logger.info(f"Network created: {network.name}")
    return db_network

@app.get("/networks", response_model=List[QuantumNetworkResponse], tags=["Networks"])
async def list_networks(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste réseaux"""
    return db.query(QuantumNetwork).filter(QuantumNetwork.user_id == current_user.id).offset(skip).limit(limit).all()

@app.post("/networks/{network_id}/qkd", tags=["Networks"])
async def quantum_key_distribution(network_id: int, key_length: int = 256, alice_node: int = 0, bob_node: int = 1,
                                  current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Distribution de clés quantiques (QKD)"""
    network = db.query(QuantumNetwork).filter(QuantumNetwork.id == network_id, QuantumNetwork.user_id == current_user.id).first()
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")
    
    if not network.qkd_enabled:
        raise HTTPException(status_code=400, detail="QKD not enabled for this network")
    
    # Générer clé quantique
    key = ''.join(str(np.random.randint(0, 2)) for _ in range(key_length))
    qber = np.random.uniform(0.01, 0.05)  # Quantum Bit Error Rate
    
    logger.info(f"QKD performed: {key_length} bits")
    return {
        "key": key[:64] + "..." if len(key) > 64 else key,
        "key_length": key_length,
        "qber": float(qber),
        "alice_node": alice_node,
        "bob_node": bob_node,
        "security": "Information-Theoretic",
        "protocol": "BB84"
    }

# ==================== BLACK HOLES ====================
@app.post("/black-holes", response_model=BlackHoleResponse, status_code=201, tags=["Black Holes"])
async def create_black_hole(bh: BlackHoleCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer simulation trou noir"""
    M_sun = 1.989e30
    mass_kg = bh.mass_solar * M_sun
    
    r_s = schwarzschild_radius(mass_kg)
    T_H = hawking_temperature(mass_kg)
    S_BH = bekenstein_entropy(mass_kg)
    t_evap = 2.1e67 * (mass_kg / 1e30)**3
    
    db_bh = BlackHole(
        name=bh.name,
        user_id=current_user.id,
        bh_class=bh.bh_class,
        mass_solar=bh.mass_solar,
        mass_kg=mass_kg,
        spin=bh.spin,
        charge=bh.charge,
        schwarzschild_radius=r_s,
        hawking_temperature=T_H,
        bekenstein_entropy=S_BH,
        evaporation_time=t_evap
    )
    db.add(db_bh)
    db.commit()
    db.refresh(db_bh)
    logger.info(f"Black hole created: {bh.name}")
    return db_bh

@app.get("/black-holes", response_model=List[BlackHoleResponse], tags=["Black Holes"])
async def list_black_holes(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste trous noirs"""
    return db.query(BlackHole).filter(BlackHole.user_id == current_user.id).offset(skip).limit(limit).all()

@app.get("/black-holes/{bh_id}", response_model=BlackHoleResponse, tags=["Black Holes"])
async def get_black_hole(bh_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Détails trou noir"""
    bh = db.query(BlackHole).filter(BlackHole.id == bh_id, BlackHole.user_id == current_user.id).first()
    if not bh:
        raise HTTPException(status_code=404, detail="Black hole not found")
    return bh

@app.get("/black-holes/{bh_id}/hawking-radiation", tags=["Black Holes"])
async def hawking_radiation_spectrum(bh_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Spectre radiation Hawking"""
    bh = db.query(BlackHole).filter(BlackHole.id == bh_id, BlackHole.user_id == current_user.id).first()
    if not bh:
        raise HTTPException(status_code=404, detail="Black hole not found")
    
    T_H = bh.hawking_temperature
    r_s = bh.schwarzschild_radius
    A = 4 * np.pi * r_s**2
    sigma = 5.67e-8  # Stefan-Boltzmann
    power = sigma * A * T_H**4
    
    return {
        "temperature_K": float(T_H),
        "power_W": float(power),
        "spectrum_type": "Blackbody",
        "peak_wavelength_nm": float(2.898e-3 / T_H * 1e9) if T_H > 0 else None
    }

# ==================== SINGULARITIES ====================
@app.post("/singularities", response_model=SingularityResponse, status_code=201, tags=["Singularities"])
async def create_singularity(sing: SingularityCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer singularité"""
    M_sun = 1.989e30
    mass_kg = sing.mass_solar * M_sun
    
    r_s = schwarzschild_radius(mass_kg)
    T_H = hawking_temperature(mass_kg)
    S_BH = bekenstein_entropy(mass_kg)
    
    db_sing = Singularity(
        user_id=current_user.id,
        singularity_type=sing.singularity_type,
        mass_solar=sing.mass_solar,
        schwarzschild_radius=r_s,
        hawking_temperature=T_H,
        bekenstein_entropy=S_BH,
        quantum_corrections=sing.quantum_corrections,
        parameters=sing.parameters
    )
    db.add(db_sing)
    db.commit()
    db.refresh(db_sing)
    logger.info(f"Singularity created: {sing.singularity_type}")
    return db_sing

@app.get("/singularities", response_model=List[SingularityResponse], tags=["Singularities"])
async def list_singularities(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste singularités"""
    return db.query(Singularity).filter(Singularity.user_id == current_user.id).offset(skip).limit(limit).all()

# ==================== TUNNELING ====================
@app.post("/tunneling", response_model=TunnelingExperimentResponse, status_code=201, tags=["Tunneling"])
async def create_tunneling_experiment(exp: TunnelingExperimentCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Expérience effet tunnel"""
    eV_to_J = PHYSICAL_CONSTANTS['e']
    
    masses = {
        "electron": PHYSICAL_CONSTANTS['m_e'],
        "proton": PHYSICAL_CONSTANTS['m_p'],
        "alpha": 6.64e-27
    }
    mass = masses.get(exp.particle_type.lower(), PHYSICAL_CONSTANTS['m_e'])
    
    T = calculate_tunneling_probability(
        exp.barrier_height_ev * eV_to_J,
        exp.barrier_width_nm * 1e-9,
        exp.particle_energy_ev * eV_to_J,
        mass
    )
    
    db_exp = TunnelingExperiment(
        user_id=current_user.id,
        particle_type=exp.particle_type,
        particle_energy_ev=exp.particle_energy_ev,
        barrier_height_ev=exp.barrier_height_ev,
        barrier_width_nm=exp.barrier_width_nm,
        tunneling_probability=T,
        reflection_coefficient=1-T
    )
    db.add(db_exp)
    db.commit()
    db.refresh(db_exp)
    logger.info(f"Tunneling experiment created: T={T:.2e}")
    return db_exp

@app.get("/tunneling", response_model=List[TunnelingExperimentResponse], tags=["Tunneling"])
async def list_tunneling_experiments(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste expériences tunnel"""
    return db.query(TunnelingExperiment).filter(TunnelingExperiment.user_id == current_user.id).offset(skip).limit(limit).all()

# ==================== EXPERIMENTS ====================
@app.post("/experiments", response_model=ExperimentResponse, status_code=201, tags=["Experiments"])
async def create_experiment(exp: ExperimentCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer expérience"""
    db_exp = Experiment(
        name=exp.name,
        user_id=current_user.id,
        experiment_type=exp.experiment_type,
        n_qubits=exp.n_qubits,
        n_measurements=exp.n_measurements,
        fidelity_target=exp.fidelity_target,
        hypothesis=exp.hypothesis
    )
    db.add(db_exp)
    db.commit()
    db.refresh(db_exp)
    logger.info(f"Experiment created: {exp.name}")
    return db_exp

@app.get("/experiments", response_model=List[ExperimentResponse], tags=["Experiments"])
async def list_experiments(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste expériences"""
    return db.query(Experiment).filter(Experiment.user_id == current_user.id).offset(skip).limit(limit).all()

@app.post("/experiments/{exp_id}/execute", tags=["Experiments"])
async def execute_experiment(exp_id: int, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Exécuter expérience"""
    exp = db.query(Experiment).filter(Experiment.id == exp_id, Experiment.user_id == current_user.id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Simulation résultats
    fidelity_achieved = exp.fidelity_target * np.random.uniform(0.95, 1.02)
    
    if exp.experiment_type == "Test Bell":
        S = np.random.uniform(2.3, 2.8)
        result = f"S = {S:.3f} (Bell violation!)"
    elif exp.experiment_type == "Téléportation":
        result = f"Fidelity: {fidelity_achieved:.3f}"
    else:
        result = "Success"
    
    exp.fidelity_achieved = fidelity_achieved
    exp.result = result
    exp.status = "completed"
    exp.executed_at = datetime.utcnow()
    db.commit()
    
    logger.info(f"Experiment executed: {exp.name}")
    return {
        "message": "Experiment executed",
        "fidelity_achieved": float(fidelity_achieved),
        "result": result,
        "status": "completed"
    }

# ==================== MEASUREMENTS ====================
@app.post("/measurements", response_model=MeasurementResponse, status_code=201, tags=["Measurements"])
async def create_measurement(meas: MeasurementCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Effectuer mesure"""
    qubit = db.query(Qubit).filter(Qubit.id == meas.qubit_id, Qubit.user_id == current_user.id).first()
    if not qubit:
        raise HTTPException(status_code=404, detail="Qubit not found")
    
    alpha = complex(qubit.state_real_0, qubit.state_imag_0)
    prob_0 = abs(alpha)**2
    result = 0 if np.random.random() < prob_0 else 1
    
    db_meas = Measurement(
        qubit_id=meas.qubit_id,
        experiment_id=meas.experiment_id,
        measurement_basis=meas.measurement_basis,
        result=result,
        probability_0=prob_0,
        probability_1=1-prob_0
    )
    db.add(db_meas)
    db.commit()
    db.refresh(db_meas)
    logger.info(f"Measurement performed: qubit {meas.qubit_id} = {result}")
    return db_meas

@app.get("/measurements", response_model=List[MeasurementResponse], tags=["Measurements"])
async def list_measurements(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste mesures"""
    measurements = db.query(Measurement).join(Qubit).filter(Qubit.user_id == current_user.id).offset(skip).limit(limit).all()
    return measurements

# ==================== QUANTUM GRAVITY ====================
@app.post("/quantum-gravity", response_model=QuantumGravitySimResponse, status_code=201, tags=["Quantum Gravity"])
async def create_qg_simulation(sim: QuantumGravitySimCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Créer simulation gravité quantique"""
    E_planck = PHYSICAL_CONSTANTS['planck_energy']
    quantum_correction = (sim.energy_scale / E_planck) ** 2
    
    db_sim = QuantumGravitySimulation(
        user_id=current_user.id,
        simulation_type=sim.simulation_type,
        energy_scale=sim.energy_scale,
        parameters=sim.parameters,
        quantum_corrections=quantum_correction,
        status="initialized"
    )
    db.add(db_sim)
    db.commit()
    db.refresh(db_sim)
    logger.info(f"QG simulation created: {sim.simulation_type}")
    return db_sim

@app.get("/quantum-gravity", response_model=List[QuantumGravitySimResponse], tags=["Quantum Gravity"])
async def list_qg_simulations(skip: int = 0, limit: int = 100, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Liste simulations gravité quantique"""
    return db.query(QuantumGravitySimulation).filter(QuantumGravitySimulation.user_id == current_user.id).offset(skip).limit(limit).all()

@app.post("/quantum-gravity/{sim_id}/run", tags=["Quantum Gravity"])
async def run_qg_simulation(sim_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Exécuter simulation gravité quantique"""
    sim = db.query(QuantumGravitySimulation).filter(QuantumGravitySimulation.id == sim_id, QuantumGravitySimulation.user_id == current_user.id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Résultats simulation
    results = {
        "planck_scale_effects": sim.quantum_corrections > 0.01,
        "foam_structure": sim.quantum_corrections > 0.01,
        "length_scale": float(PHYSICAL_CONSTANTS['planck_length'] * np.sqrt(1 + sim.quantum_corrections)),
        "time_scale": float(PHYSICAL_CONSTANTS['planck_time'] * np.sqrt(1 + sim.quantum_corrections))
    }
    
    sim.results = results
    sim.status = "completed"
    db.commit()
    
    logger.info(f"QG simulation executed: {sim.simulation_type}")
    return results

# ==================== PHYSICS CALCULATIONS ====================
@app.get("/constants", tags=["Physics"])
async def get_constants():
    """Constantes physiques fondamentales"""
    return PHYSICAL_CONSTANTS

@app.post("/calculate/schwarzschild", tags=["Physics"])
async def calc_schwarzschild(mass_solar: float):
    """Calculer rayon Schwarzschild"""
    M_sun = 1.989e30
    mass_kg = mass_solar * M_sun
    r_s = schwarzschild_radius(mass_kg)
    return {
        "mass_solar": mass_solar,
        "mass_kg": mass_kg,
        "schwarzschild_radius_m": float(r_s),
        "schwarzschild_radius_km": float(r_s / 1000)
    }

@app.post("/calculate/hawking-temperature", tags=["Physics"])
async def calc_hawking_temp(mass_solar: float):
    """Calculer température Hawking"""
    M_sun = 1.989e30
    mass_kg = mass_solar * M_sun
    T_H = hawking_temperature(mass_kg)
    return {
        "mass_solar": mass_solar,
        "hawking_temperature_K": float(T_H),
        "evaporation_time_years": float(2.1e67 * (mass_kg / 1e30)**3 / 3.15e7)
    }

@app.get("/planck-scale", tags=["Physics"])
async def planck_scale_info():
    """Informations échelle de Planck"""
    return {
        "planck_length_m": PHYSICAL_CONSTANTS['planck_length'],
        "planck_time_s": PHYSICAL_CONSTANTS['planck_time'],
        "planck_mass_kg": PHYSICAL_CONSTANTS['planck_mass'],
        "planck_energy_J": PHYSICAL_CONSTANTS['planck_energy'],
        "description": "Fundamental scale where quantum gravity becomes important"
    }

# ==================== STATISTICS ====================
@app.get("/statistics", tags=["Statistics"])
async def get_statistics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Statistiques utilisateur"""
    stats = {
        "user": current_user.username,
        "qubits": db.query(Qubit).filter(Qubit.user_id == current_user.id).count(),
        "entangled_pairs": db.query(EntangledPair).filter(EntangledPair.user_id == current_user.id).count(),
        "circuits": db.query(QuantumCircuit).filter(QuantumCircuit.user_id == current_user.id).count(),
        "networks": db.query(QuantumNetwork).filter(QuantumNetwork.user_id == current_user.id).count(),
        "black_holes": db.query(BlackHole).filter(BlackHole.user_id == current_user.id).count(),
        "singularities": db.query(Singularity).filter(Singularity.user_id == current_user.id).count(),
        "experiments": db.query(Experiment).filter(Experiment.user_id == current_user.id).count(),
        "measurements": db.query(Measurement).join(Qubit).filter(Qubit.user_id == current_user.id).count(),
        "timestamp": datetime.utcnow()
    }
    return stats

@app.get("/statistics/global", tags=["Statistics"])
async def get_global_statistics(db: Session = Depends(get_db)):
    """Statistiques globales plateforme"""
    return {
        "total_users": db.query(User).count(),
        "total_qubits": db.query(Qubit).count(),
        "total_entangled_pairs": db.query(EntangledPair).count(),
        "total_experiments": db.query(Experiment).count(),
        "total_measurements": db.query(Measurement).count(),
        "timestamp": datetime.utcnow()
    }

# ==================== HEALTH CHECK ====================
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "message": "⚛️ Quantum Physics Research API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc"
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
    uvicorn.run(app, host="0.0.0.0", port=8038, reload=True)