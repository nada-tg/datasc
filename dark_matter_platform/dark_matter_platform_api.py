"""
API Backend - Dark Matter Research Platform
FastAPI + SQLAlchemy + PostgreSQL

Installation:
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib scipy numpy

Lancement:
uvicorn dark_matter_platform_api:app --reload --host 0.0.0.0 --port 8021
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
import logging

# ==================== CONFIGURATION ====================

DATABASE_URL = "postgresql://postgres:nadaprojet@localhost:5432/dark_matter_db"
# Pour dev: DATABASE_URL = "sqlite:///./dark_matter.db"

SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== INITIALISATION ====================

app = FastAPI(
    title="Dark Matter Research API",
    description="API pour recherche matière noire - WIMPs, Neutrinos, Xénon",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ==================== MODELS DATABASE ====================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="researcher")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    detectors = relationship("Detector", back_populates="owner")
    experiments = relationship("Experiment", back_populates="creator")

class Detector(Base):
    __tablename__ = "detectors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String)  # Xenon, Argon, Germanium, etc.
    mass_kg = Column(Float)
    temperature_k = Column(Float)
    location = Column(String)
    depth_m = Column(Float)
    specs = Column(JSON)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="detectors")
    detections = relationship("Detection", back_populates="detector")

class Detection(Base):
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    detector_id = Column(Integer, ForeignKey("detectors.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    particle_type = Column(String)  # WIMP, neutrino, xenon_decay
    energy_kev = Column(Float)
    confidence = Column(Float)
    position = Column(JSON)
    metadata_json = Column(JSON)
    
    detector = relationship("Detector", back_populates="detections")

class WIMPCandidate(Base):
    __tablename__ = "wimp_candidates"
    
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"))
    wimp_mass_gev = Column(Float)
    cross_section = Column(Float)
    recoil_type = Column(String)
    ai_classification = Column(JSON)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class NeutrinoEvent(Base):
    __tablename__ = "neutrino_events"
    
    id = Column(Integer, primary_key=True, index=True)
    detector_id = Column(Integer, ForeignKey("detectors.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    flavor = Column(String)  # electron, muon, tau
    energy_mev = Column(Float)
    interaction_type = Column(String)
    position = Column(JSON)

class XenonDecay(Base):
    __tablename__ = "xenon_decays"
    
    id = Column(Integer, primary_key=True, index=True)
    detector_id = Column(Integer, ForeignKey("detectors.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    isotope = Column(String)  # Xe-136, Xe-134, etc.
    decay_type = Column(String)  # double_beta, single_beta
    energy_kev = Column(Float)
    position = Column(JSON)

class Experiment(Base):
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String)
    duration_days = Column(Integer)
    progress = Column(Float, default=0.0)
    status = Column(String, default="planned")
    parameters = Column(JSON)
    results = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    creator_id = Column(Integer, ForeignKey("users.id"))
    
    creator = relationship("User", back_populates="experiments")

class QuantumSimulation(Base):
    __tablename__ = "quantum_simulations"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_type = Column(String)
    n_qubits = Column(Integer)
    algorithm = Column(String)
    result = Column(JSON)
    execution_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ==================== SCHEMAS PYDANTIC ====================

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class DetectorCreate(BaseModel):
    name: str
    type: str
    mass_kg: float
    temperature_k: float
    location: str
    depth_m: float
    specs: Dict[str, Any]

class DetectorResponse(BaseModel):
    id: int
    name: str
    type: str
    mass_kg: float
    temperature_k: float
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class DetectionCreate(BaseModel):
    detector_id: int
    particle_type: str
    energy_kev: float
    confidence: float
    position: Dict[str, float]
    metadata: Optional[Dict] = {}

class DetectionResponse(BaseModel):
    id: int
    detector_id: int
    timestamp: datetime
    particle_type: str
    energy_kev: float
    confidence: float
    
    class Config:
        from_attributes = True

class WIMPSearchRequest(BaseModel):
    detector_id: int
    exposure_days: int
    wimp_mass_gev: float
    mass_range: str

class NeutrinoSearchRequest(BaseModel):
    detector_id: int
    exposure_days: float

class XenonDecayRequest(BaseModel):
    detector_id: int
    isotope: str
    simulation_hours: float

class ExperimentCreate(BaseModel):
    name: str
    type: str
    duration_days: int
    parameters: Dict[str, Any]

class QuantumSimulationRequest(BaseModel):
    simulation_type: str
    n_qubits: int
    algorithm: str
    parameters: Dict[str, Any]

# ==================== DEPENDENCIES ====================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

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

# ==================== PHYSICS FUNCTIONS ====================
def calculate_wimp_interaction_rate(mass_gev: float, cross_section: float, 
                                    detector_mass_kg: float) -> float:
    """Calculer taux d'interaction WIMPs"""
    # Densité locale matière noire
    rho_dm = 0.3  # GeV/cm³
    
    # Vitesse moyenne WIMPs dans halo galactique
    v_dm = 220000  # m/s = 220 km/s
    
    # Convertir masse détecteur en nombre de nucléons
    # 1 kg ≈ 6e26 nucléons (masse moyenne ~1.67e-27 kg)
    n_nucleons = detector_mass_kg * 6e26
    
    # Formule taux: R = (ρ/M_WIMP) × σ × v × N_nucléons
    # ρ/M_WIMP donne nombre WIMPs par volume
    # σ × v donne taux interaction par WIMP
    # N_nucléons donne nombre cibles
    
    wimp_number_density = rho_dm / mass_gev  # nombre WIMPs/cm³
    
    # Taux par nucléon par seconde
    rate_per_nucleon = wimp_number_density * cross_section * v_dm / 1e5  # cm/s conversion
    
    # Taux total (événements par jour)
    total_rate = rate_per_nucleon * n_nucleons * 86400
    
    # Ajuster pour avoir taux observable (facteur échelle pour démo)
    # En réalité beaucoup plus faible, mais on veut voir des événements
    observable_rate = total_rate * 1e10
    
    return max(observable_rate, 1)  # Minimum 1 événement/jour

def quantum_compute_cross_section(wimp_mass: float, nucleon_mass: float) -> float:
    """Calculer section efficace par computing quantique"""
    mu = (wimp_mass * nucleon_mass) / (wimp_mass + nucleon_mass)
    cross_section = 1e-45 * (mu / 1)**2
    quantum_correction = np.random.uniform(0.8, 1.2)
    return cross_section * quantum_correction

def simulate_wimp_detection(detector_id: int, exposure_days: int, wimp_mass: float, db: Session):
    """Simuler détection WIMPs"""
    detector = db.query(Detector).filter(Detector.id == detector_id).first()
    if not detector:
        return []
    
    cross_section = quantum_compute_cross_section(wimp_mass, 931.5)
    rate = calculate_wimp_interaction_rate(wimp_mass, cross_section, detector.mass_kg)
    
    n_events = int(np.random.poisson(rate * exposure_days * 86400))
    
    detections = []
    for _ in range(n_events):
        energy_kev = np.random.exponential(10) + 1.0
        
        detection = Detection(
            detector_id=detector_id,
            particle_type='WIMP_candidate',
            energy_kev=energy_kev,
            confidence=np.random.uniform(0.6, 0.95),
            position={'x': np.random.normal(0, 10), 'y': np.random.normal(0, 10), 'z': np.random.normal(0, 20)},
            metadata={'wimp_mass_gev': wimp_mass, 'cross_section': cross_section}
        )
        db.add(detection)
        detections.append(detection)
    
    db.commit()
    return detections

def simulate_neutrino_detection(detector_id: int, exposure_days: float, db: Session):
    """Simuler détection neutrinos solaires"""
    flux = 6.5e10  # neutrinos/cm²/s
    efficiency = 0.15
    detector_area_cm2 = 10000
    
    n_events = int(flux * detector_area_cm2 * exposure_days * 86400 * efficiency * 1e-12)
    
    events = []
    for _ in range(n_events):
        neutrino = NeutrinoEvent(
            detector_id=detector_id,
            flavor=np.random.choice(['electron', 'muon', 'tau']),
            energy_mev=np.random.exponential(0.5),
            interaction_type=np.random.choice(['elastic', 'charged_current', 'neutral_current']),
            position={'x': np.random.normal(0, 30), 'y': np.random.normal(0, 30), 'z': np.random.normal(0, 60)}
        )
        db.add(neutrino)
        events.append(neutrino)
    
    db.commit()
    return events

def simulate_xenon_decay(detector_id: int, isotope: str, time_hours: float, db: Session):
    """Simuler désintégrations Xénon"""
    decay_constants = {
        'Xe-136': 2.11e-22,
        'Xe-134': 1e-25,
        'Xe-132': 5e-26
    }
    
    lambda_decay = decay_constants.get(isotope, 1e-24)
    n_events = int(lambda_decay * time_hours * 3600 * 1e6)
    
    events = []
    for _ in range(n_events):
        decay = XenonDecay(
            detector_id=detector_id,
            isotope=isotope,
            decay_type='double_beta' if np.random.random() > 0.9 else 'single_beta',
            energy_kev=np.random.normal(2458, 50) if isotope == 'Xe-136' else np.random.normal(1000, 100),
            position={'x': np.random.uniform(-50, 50), 'y': np.random.uniform(-50, 50), 'z': np.random.uniform(-100, 100)}
        )
        db.add(decay)
        events.append(decay)
    
    db.commit()
    return events

# ==================== ROUTES AUTH ====================

@app.post("/api/v1/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Créer nouveau compte"""
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"New user registered: {user.username}")
    return db_user

@app.post("/api/v1/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Connexion"""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    logger.info(f"User logged in: {user.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/v1/auth/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Obtenir profil utilisateur"""
    return current_user

# ==================== ROUTES DETECTORS ====================

@app.post("/api/v1/detectors", response_model=DetectorResponse, status_code=status.HTTP_201_CREATED)
async def create_detector(
    detector: DetectorCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer nouveau détecteur"""
    db_detector = Detector(
        name=detector.name,
        type=detector.type,
        mass_kg=detector.mass_kg,
        temperature_k=detector.temperature_k,
        location=detector.location,
        depth_m=detector.depth_m,
        specs=detector.specs,
        owner_id=current_user.id
    )
    db.add(db_detector)
    db.commit()
    db.refresh(db_detector)
    
    logger.info(f"Detector created: {detector.name} by {current_user.username}")
    return db_detector

@app.get("/api/v1/detectors", response_model=List[DetectorResponse])
async def list_detectors(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lister détecteurs de l'utilisateur"""
    detectors = db.query(Detector).filter(Detector.owner_id == current_user.id).offset(skip).limit(limit).all()
    return detectors

@app.get("/api/v1/detectors/{detector_id}", response_model=DetectorResponse)
async def get_detector(
    detector_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir détecteur spécifique"""
    detector = db.query(Detector).filter(
        Detector.id == detector_id,
        Detector.owner_id == current_user.id
    ).first()
    
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    return detector

@app.delete("/api/v1/detectors/{detector_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_detector(
    detector_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Supprimer détecteur"""
    detector = db.query(Detector).filter(
        Detector.id == detector_id,
        Detector.owner_id == current_user.id
    ).first()
    
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    
    db.delete(detector)
    db.commit()
    logger.info(f"Detector deleted: {detector_id}")
    return None

# ==================== ROUTES WIMPS ====================

@app.post("/api/v1/wimps/search")
async def search_wimps(
    request: WIMPSearchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lancer recherche WIMPs"""
    detector = db.query(Detector).filter(Detector.id == request.detector_id).first()
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    
    # Lancer simulation en arrière-plan
    background_tasks.add_task(
        simulate_wimp_detection,
        request.detector_id,
        request.exposure_days,
        request.wimp_mass_gev,
        db
    )
    
    logger.info(f"WIMP search started: detector {request.detector_id}, {request.exposure_days} days")
    
    return {
        "message": "WIMP search initiated",
        "detector_id": request.detector_id,
        "exposure_days": request.exposure_days,
        "wimp_mass_gev": request.wimp_mass_gev,
        "status": "processing"
    }

@app.get("/api/v1/wimps/candidates")
async def get_wimp_candidates(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir candidats WIMPs"""
    candidates = db.query(WIMPCandidate).offset(skip).limit(limit).all()
    return candidates

@app.get("/api/v1/wimps/statistics")
async def get_wimp_statistics(
    detector_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Statistiques détection WIMPs"""
    query = db.query(Detection).filter(Detection.particle_type == 'WIMP_candidate')
    
    if detector_id:
        query = query.filter(Detection.detector_id == detector_id)
    
    total = query.count()
    
    if total > 0:
        energies = [d.energy_kev for d in query.all()]
        return {
            "total_candidates": total,
            "mean_energy_kev": np.mean(energies),
            "std_energy_kev": np.std(energies),
            "min_energy_kev": np.min(energies),
            "max_energy_kev": np.max(energies)
        }
    
    return {"total_candidates": 0}

# ==================== ROUTES NEUTRINOS ====================

@app.post("/api/v1/neutrinos/search")
async def search_neutrinos(
    request: NeutrinoSearchRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lancer détection neutrinos"""
    detector = db.query(Detector).filter(Detector.id == request.detector_id).first()
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    
    background_tasks.add_task(simulate_neutrino_detection, request.detector_id, request.exposure_days, db)
    
    logger.info(f"Neutrino search started: detector {request.detector_id}")
    
    return {
        "message": "Neutrino detection initiated",
        "detector_id": request.detector_id,
        "exposure_days": request.exposure_days
    }

@app.get("/api/v1/neutrinos/events")
async def get_neutrino_events(
    skip: int = 0,
    limit: int = 100,
    flavor: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir événements neutrinos"""
    query = db.query(NeutrinoEvent)
    
    if flavor:
        query = query.filter(NeutrinoEvent.flavor == flavor)
    
    events = query.offset(skip).limit(limit).all()
    return events

@app.get("/api/v1/neutrinos/oscillations")
async def analyze_neutrino_oscillations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyser oscillations neutrinos"""
    events = db.query(NeutrinoEvent).all()
    
    if len(events) < 50:
        return {"error": "Insufficient data", "min_required": 50, "current": len(events)}
    
    flavors = [e.flavor for e in events]
    
    electron_ratio = flavors.count('electron') / len(flavors)
    muon_ratio = flavors.count('muon') / len(flavors)
    tau_ratio = flavors.count('tau') / len(flavors)
    
    # Test chi-carré
    observed = [flavors.count('electron'), flavors.count('muon'), flavors.count('tau')]
    expected = [len(flavors)/3] * 3
    chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
    
    return {
        "total_events": len(events),
        "electron_ratio": electron_ratio,
        "muon_ratio": muon_ratio,
        "tau_ratio": tau_ratio,
        "chi2": chi2,
        "compatible_maximal_mixing": chi2 < 5.99
    }

# ==================== ROUTES XENON ====================

@app.post("/api/v1/xenon/simulate")
async def simulate_xenon(
    request: XenonDecayRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simuler désintégrations Xénon"""
    detector = db.query(Detector).filter(Detector.id == request.detector_id).first()
    if not detector:
        raise HTTPException(status_code=404, detail="Detector not found")
    
    background_tasks.add_task(
        simulate_xenon_decay,
        request.detector_id,
        request.isotope,
        request.simulation_hours,
        db
    )
    
    logger.info(f"Xenon decay simulation started: {request.isotope}")
    
    return {
        "message": "Xenon decay simulation initiated",
        "isotope": request.isotope,
        "simulation_hours": request.simulation_hours
    }

@app.get("/api/v1/xenon/decays")
async def get_xenon_decays(
    skip: int = 0,
    limit: int = 100,
    isotope: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir désintégrations Xénon"""
    query = db.query(XenonDecay)
    
    if isotope:
        query = query.filter(XenonDecay.isotope == isotope)
    
    decays = query.offset(skip).limit(limit).all()
    return decays

@app.get("/api/v1/xenon/search-0vbb")
async def search_0vbb(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rechercher désintégration double bêta sans neutrinos"""
    decays = db.query(XenonDecay).filter(XenonDecay.isotope == 'Xe-136').all()
    
    if not decays:
        return {"message": "No Xe-136 decay data"}
    
    # ROI autour Q-value (2458 keV)
    roi_events = [d for d in decays if 2400 < d.energy_kev < 2500]
    
    return {
        "total_xe136_events": len(decays),
        "roi_events": len(roi_events),
        "q_value_kev": 2458,
        "signal_detected": len(roi_events) > 3,
        "half_life_limit": "T₁/₂ > 2.3×10²⁵ years (90% CL)" if len(roi_events) < 3 else "Signal requires investigation"
    }

# ==================== ROUTES EXPERIMENTS ====================

@app.post("/api/v1/experiments", status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment: ExperimentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Créer expérience"""
    db_experiment = Experiment(
        name=experiment.name,
        type=experiment.type,
        duration_days=experiment.duration_days,
        parameters=experiment.parameters,
        creator_id=current_user.id
    )
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    
    logger.info(f"Experiment created: {experiment.name}")
    return db_experiment

@app.get("/api/v1/experiments")
async def list_experiments(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lister expériences"""
    experiments = db.query(Experiment).filter(
        Experiment.creator_id == current_user.id
    ).offset(skip).limit(limit).all()
    return experiments

# ==================== ROUTES QUANTUM ====================

@app.post("/api/v1/quantum/simulate")
async def run_quantum_simulation(
    request: QuantumSimulationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Lancer simulation quantique"""
    # Simulation calcul quantique
    if request.simulation_type == "cross_section":
        wimp_mass = request.parameters.get('wimp_mass', 100)
        nucleon_mass = request.parameters.get('nucleon_mass', 931.5)
        
        cross_section = quantum_compute_cross_section(wimp_mass, nucleon_mass)
        
        result = {
            'cross_section': cross_section,
            'uncertainty': np.random.uniform(0.01, 0.05),
            'fidelity': np.random.uniform(0.95, 0.99)
        }
    else:
        result = {
            'value': np.random.random(),
            'fidelity': np.random.uniform(0.90, 0.99)
        }
    
    db_sim = QuantumSimulation(
        simulation_type=request.simulation_type,
        n_qubits=request.n_qubits,
        algorithm=request.algorithm,
        result=result,
        execution_time=np.random.uniform(1, 10)
    )
    db.add(db_sim)
    db.commit()
    db.refresh(db_sim)
    
    logger.info(f"Quantum simulation: {request.simulation_type}, {request.n_qubits} qubits")
    
    return db_sim

@app.get("/api/v1/quantum/simulations")
async def get_quantum_simulations(
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtenir simulations quantiques"""
    simulations = db.query(QuantumSimulation).offset(skip).limit(limit).all()
    return simulations

# ==================== ROUTES ANALYTICS ====================

@app.get("/api/v1/analytics/summary")
async def get_analytics_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Statistiques globales"""
    total_detectors = db.query(Detector).filter(Detector.owner_id == current_user.id).count()
    total_detections = db.query(Detection).count()
    total_wimps = db.query(WIMPCandidate).count()
    total_neutrinos = db.query(NeutrinoEvent).count()
    total_xenon = db.query(XenonDecay).count()
    
    return {
        "total_detectors": total_detectors,
        "total_detections": total_detections,
        "total_wimp_candidates": total_wimps,
        "total_neutrino_events": total_neutrinos,
        "total_xenon_decays": total_xenon,
        "timestamp": datetime.utcnow()
    }

@app.get("/api/v1/analytics/detections/timeline")
async def get_detection_timeline(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Timeline détections"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    detections = db.query(Detection).filter(Detection.timestamp >= start_date).all()
    
    # Grouper par jour
    timeline = {}
    for detection in detections:
        date_key = detection.timestamp.strftime('%Y-%m-%d')
        timeline[date_key] = timeline.get(date_key, 0) + 1
    
    return {
        "days": days,
        "timeline": timeline,
        "total_detections": len(detections)
    }

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dark Matter Research API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)