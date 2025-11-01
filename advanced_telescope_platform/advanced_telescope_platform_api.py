"""
🔭 Advanced Space Telescope Platform - API FastAPI Complète
Backend REST API pour gestion télescopes, observations, et analyses

Installation:
pip install fastapi uvicorn sqlalchemy pydantic python-jose passlib python-multipart

Lancement:
uvicorn advanced_telescope_platform_api:app --reload --host 0.0.0.0 --port 8002

Documentation: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
from enum import Enum
import uuid

# ==================== CONFIGURATION ====================

app = FastAPI(
    title="🔭 Space Telescope API",
    description="API complète pour gestion observatoire astronomique",
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
SECRET_KEY = "votre_cle_secrete_super_securisee_changez_moi"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Base de données simulée (remplacer par vraie DB en production)
fake_db = {
    "users": {},
    "telescopes": {},
    "targets": {},
    "observations": {},
    "spectra": {},
    "exoplanets": {},
    "galaxies": {},
    "black_holes": {},
    "discoveries": {}
}

# ==================== ENUMS ====================

class TelescopeType(str, Enum):
    SPATIAL = "Spatial"
    GROUND_OPTICAL = "Sol - Optique"
    RADIO = "Radio"
    GAMMA_X = "Gamma/X"

class ObjectType(str, Enum):
    STAR = "Étoiles"
    EXOPLANET = "Exoplanètes"
    GALAXY = "Galaxies"
    NEBULA = "Nébuleuses"
    BLACK_HOLE = "Trous Noirs"

class ObservationMode(str, Enum):
    IMAGING = "Imagerie"
    SPECTROSCOPY = "Spectroscopie"
    PHOTOMETRY = "Photométrie"
    POLARIMETRY = "Polarimétrie"

# ==================== MODELS PYDANTIC ====================

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

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

class TelescopeCreate(BaseModel):
    name: str = Field(..., example="DeepSky-1")
    type: TelescopeType
    diameter_m: float = Field(..., gt=0, example=6.5)
    focal_length_m: float = Field(..., gt=0, example=20.0)
    location: str = Field(..., example="Orbite Terrestre")
    wavelength_range: List[str]
    instruments: List[str]
    detector_type: str
    field_of_view_arcmin: float
    science_goals: List[str]
    budget_millions: float

class TelescopeResponse(BaseModel):
    id: str
    name: str
    type: str
    diameter_m: float
    resolution_arcsec: float
    limiting_magnitude: float
    collecting_area_m2: float
    status: str
    created_at: datetime

class TargetCreate(BaseModel):
    name: str
    object_type: ObjectType
    ra_deg: float = Field(..., ge=0, le=360)
    dec_deg: float = Field(..., ge=-90, le=90)
    magnitude: float
    distance_mpc: float = Field(..., gt=0)
    priority: str
    notes: Optional[str] = None

class TargetResponse(BaseModel):
    id: str
    name: str
    object_type: str
    ra_deg: float
    dec_deg: float
    magnitude: float
    distance_mpc: float
    priority: str
    observations_count: int
    created_at: datetime

class ObservationCreate(BaseModel):
    telescope_id: str
    target_id: str
    mode: ObservationMode
    exposure_time_s: int = Field(..., gt=0)
    n_exposures: int = Field(..., gt=0)
    filter_band: str
    seeing_arcsec: float

class ObservationResponse(BaseModel):
    id: str
    telescope_id: str
    target_id: str
    mode: str
    exposure_time_s: int
    n_exposures: int
    snr: float
    n_sources_detected: int
    limiting_mag: float
    timestamp: datetime

class SpectrumCreate(BaseModel):
    target_id: str
    wavelength_range_nm: tuple[float, float]
    resolution: int = Field(..., gt=0)
    integration_time_s: int

class GalaxyCreate(BaseModel):
    type: str = Field(..., example="Sa")
    magnitude: float
    redshift: float = Field(..., ge=0)
    mass_msun: float = Field(..., gt=0)
    sfr_msun_per_year: float = Field(..., ge=0)

class GalaxyResponse(BaseModel):
    id: str
    type: str
    magnitude: float
    redshift: float
    distance_mpc: float
    mass_msun: float
    detected: datetime

class BlackHoleCreate(BaseModel):
    mass_solar: float = Field(..., gt=0)
    spin: float = Field(..., ge=0, le=0.998)

class BlackHoleResponse(BaseModel):
    id: str
    mass_solar: float
    spin: float
    schwarzschild_radius_km: float
    isco_rs: float
    hawking_temp_K: float
    timestamp: datetime

class ExoplanetCreate(BaseModel):
    radius_r_earth: float = Field(..., gt=0)
    period_days: float = Field(..., gt=0)
    semi_major_axis_AU: float = Field(..., gt=0)
    equilibrium_temp_K: float = Field(..., gt=0)
    detection_method: str

class ExoplanetResponse(BaseModel):
    id: str
    radius_r_earth: float
    period_days: float
    transit_depth: Optional[float]
    equilibrium_temp_K: float
    detection_method: str
    confirmed: bool
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
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ==================== FONCTIONS UTILITAIRES ASTRONOMIQUES ====================

def calculate_angular_resolution(diameter_m: float, wavelength_m: float = 550e-9) -> float:
    """Calculer résolution angulaire (critère Rayleigh)"""
    theta_rad = 1.22 * wavelength_m / diameter_m
    return theta_rad * 206265  # arcsec

def calculate_limiting_magnitude(diameter_m: float, exposure_s: float, qe: float = 0.8) -> float:
    """Calculer magnitude limite"""
    base_limit = 2.5 * np.log10(diameter_m**2) + 2.5 * np.log10(exposure_s)
    return 20 + base_limit + 2.5 * np.log10(qe)

def calculate_schwarzschild_radius(mass_solar: float) -> float:
    """Calculer rayon de Schwarzschild en km"""
    G = 6.67430e-11
    c = 299792458
    M_sun = 1.989e30
    rs_m = 2 * G * mass_solar * M_sun / c**2
    return rs_m / 1000

def calculate_isco(mass_solar: float, spin: float) -> float:
    """Calculer ISCO pour trou noir Kerr"""
    if spin < 0.01:
        return 6.0  # Schwarzschild
    Z1 = 1 + (1 - spin**2)**(1/3) * ((1+spin)**(1/3) + (1-spin)**(1/3))
    Z2 = np.sqrt(3*spin**2 + Z1**2)
    r_isco = 3 + Z2 - np.sqrt((3-Z1)*(3+Z1+2*Z2))
    return r_isco

def simulate_transit(period_days: float, duration_h: float, depth_percent: float) -> Dict:
    """Simuler courbe de transit"""
    time = np.linspace(0, period_days, 100)
    flux = np.ones(100)
    
    transit_start = period_days/2 - duration_h/48
    transit_end = period_days/2 + duration_h/48
    
    in_transit = (time >= transit_start) & (time <= transit_end)
    flux[in_transit] = 1 - depth_percent/100
    flux += np.random.normal(0, 0.001, 100)
    
    return {
        "time": time.tolist(),
        "flux": flux.tolist(),
        "depth_percent": depth_percent
    }

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

# ==================== ENDPOINTS TÉLESCOPES ====================

@app.post("/telescopes", response_model=TelescopeResponse, tags=["Telescopes"])
async def create_telescope(
    telescope: TelescopeCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Créer nouveau télescope"""
    telescope_id = str(uuid.uuid4())
    
    # Calculs performances
    f_ratio = telescope.focal_length_m / telescope.diameter_m
    resolution_arcsec = calculate_angular_resolution(telescope.diameter_m)
    limit_mag = calculate_limiting_magnitude(telescope.diameter_m, 3600)
    collecting_area = np.pi * (telescope.diameter_m/2)**2
    
    telescope_data = {
        "id": telescope_id,
        **telescope.dict(),
        "f_ratio": f_ratio,
        "resolution_arcsec": resolution_arcsec,
        "limiting_magnitude": limit_mag,
        "collecting_area_m2": collecting_area,
        "status": "operational",
        "created_at": datetime.now(),
        "owner": current_user.username
    }
    
    fake_db["telescopes"][telescope_id] = telescope_data
    
    return TelescopeResponse(**telescope_data)

@app.get("/telescopes", response_model=List[TelescopeResponse], tags=["Telescopes"])
async def list_telescopes(
    skip: int = 0,
    limit: int = 100,
    telescope_type: Optional[TelescopeType] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister tous les télescopes"""
    telescopes = list(fake_db["telescopes"].values())
    
    if telescope_type:
        telescopes = [t for t in telescopes if t["type"] == telescope_type]
    
    return [TelescopeResponse(**t) for t in telescopes[skip:skip+limit]]

@app.get("/telescopes/{telescope_id}", response_model=TelescopeResponse, tags=["Telescopes"])
async def get_telescope(
    telescope_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir détails d'un télescope"""
    if telescope_id not in fake_db["telescopes"]:
        raise HTTPException(status_code=404, detail="Telescope not found")
    
    return TelescopeResponse(**fake_db["telescopes"][telescope_id])

@app.delete("/telescopes/{telescope_id}", tags=["Telescopes"])
async def delete_telescope(
    telescope_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Supprimer un télescope"""
    if telescope_id not in fake_db["telescopes"]:
        raise HTTPException(status_code=404, detail="Telescope not found")
    
    telescope = fake_db["telescopes"][telescope_id]
    if telescope["owner"] != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    del fake_db["telescopes"][telescope_id]
    return {"message": "Telescope deleted successfully"}

# ==================== ENDPOINTS CIBLES ====================

@app.post("/targets", response_model=TargetResponse, tags=["Targets"])
async def create_target(
    target: TargetCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Créer nouvelle cible d'observation"""
    target_id = str(uuid.uuid4())
    
    target_data = {
        "id": target_id,
        **target.dict(),
        "observations_count": 0,
        "created_at": datetime.now(),
        "owner": current_user.username
    }
    
    fake_db["targets"][target_id] = target_data
    
    return TargetResponse(**target_data)

@app.get("/targets", response_model=List[TargetResponse], tags=["Targets"])
async def list_targets(
    skip: int = 0,
    limit: int = 100,
    object_type: Optional[ObjectType] = None,
    min_magnitude: Optional[float] = None,
    max_magnitude: Optional[float] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister toutes les cibles"""
    targets = list(fake_db["targets"].values())
    
    if object_type:
        targets = [t for t in targets if t["object_type"] == object_type]
    
    if min_magnitude is not None:
        targets = [t for t in targets if t["magnitude"] >= min_magnitude]
    
    if max_magnitude is not None:
        targets = [t for t in targets if t["magnitude"] <= max_magnitude]
    
    return [TargetResponse(**t) for t in targets[skip:skip+limit]]

@app.get("/targets/{target_id}", response_model=TargetResponse, tags=["Targets"])
async def get_target(
    target_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir détails d'une cible"""
    if target_id not in fake_db["targets"]:
        raise HTTPException(status_code=404, detail="Target not found")
    
    return TargetResponse(**fake_db["targets"][target_id])

@app.put("/targets/{target_id}/priority", tags=["Targets"])
async def update_target_priority(
    target_id: str,
    priority: str,
    current_user: User = Depends(get_current_active_user)
):
    """Mettre à jour priorité d'une cible"""
    if target_id not in fake_db["targets"]:
        raise HTTPException(status_code=404, detail="Target not found")
    
    fake_db["targets"][target_id]["priority"] = priority
    return {"message": "Priority updated", "new_priority": priority}

# ==================== ENDPOINTS OBSERVATIONS ====================

@app.post("/observations", response_model=ObservationResponse, tags=["Observations"])
async def create_observation(
    observation: ObservationCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Créer nouvelle observation"""
    # Vérifier télescope et cible existent
    if observation.telescope_id not in fake_db["telescopes"]:
        raise HTTPException(status_code=404, detail="Telescope not found")
    if observation.target_id not in fake_db["targets"]:
        raise HTTPException(status_code=404, detail="Target not found")
    
    telescope = fake_db["telescopes"][observation.telescope_id]
    target = fake_db["targets"][observation.target_id]
    
    observation_id = str(uuid.uuid4())
    
    # Calculer SNR (simplifié)
    snr = np.sqrt(observation.exposure_time_s * telescope["collecting_area_m2"]) * np.random.uniform(0.8, 1.2)
    
    # Nombre sources détectées
    n_sources = int(np.random.uniform(50, 500))
    
    observation_data = {
        "id": observation_id,
        **observation.dict(),
        "snr": snr,
        "n_sources_detected": n_sources,
        "limiting_mag": telescope["limiting_magnitude"],
        "timestamp": datetime.now(),
        "owner": current_user.username
    }
    
    fake_db["observations"][observation_id] = observation_data
    
    # Incrémenter compteur observations de la cible
    fake_db["targets"][observation.target_id]["observations_count"] += 1
    
    # Tâche background: analyse automatique
    background_tasks.add_task(analyze_observation, observation_id)
    
    return ObservationResponse(**observation_data)

async def analyze_observation(observation_id: str):
    """Analyse automatique observation (tâche background)"""
    # Simuler traitement
    import asyncio
    await asyncio.sleep(2)
    
    # Détection transients, etc.
    observation = fake_db["observations"][observation_id]
    observation["analyzed"] = True
    observation["transients_detected"] = np.random.randint(0, 5)

@app.get("/observations", response_model=List[ObservationResponse], tags=["Observations"])
async def list_observations(
    skip: int = 0,
    limit: int = 100,
    telescope_id: Optional[str] = None,
    target_id: Optional[str] = None,
    mode: Optional[ObservationMode] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister toutes les observations"""
    observations = list(fake_db["observations"].values())
    
    if telescope_id:
        observations = [o for o in observations if o["telescope_id"] == telescope_id]
    
    if target_id:
        observations = [o for o in observations if o["target_id"] == target_id]
    
    if mode:
        observations = [o for o in observations if o["mode"] == mode]
    
    # Trier par date (plus récent d'abord)
    observations.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return [ObservationResponse(**o) for o in observations[skip:skip+limit]]

@app.get("/observations/{observation_id}", response_model=ObservationResponse, tags=["Observations"])
async def get_observation(
    observation_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir détails d'une observation"""
    if observation_id not in fake_db["observations"]:
        raise HTTPException(status_code=404, detail="Observation not found")
    
    return ObservationResponse(**fake_db["observations"][observation_id])

@app.get("/observations/{observation_id}/data", tags=["Observations"])
async def get_observation_data(
    observation_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir données brutes observation (image simulée)"""
    if observation_id not in fake_db["observations"]:
        raise HTTPException(status_code=404, detail="Observation not found")
    
    # Générer données simulées
    size = 512
    image_data = np.random.poisson(50, (size, size)) + np.random.randn(size, size) * 10
    
    # Ajouter quelques sources
    n_sources = np.random.randint(10, 50)
    for _ in range(n_sources):
        x, y = np.random.randint(0, size, 2)
        brightness = np.random.uniform(100, 1000)
        sigma = np.random.uniform(2, 5)
        
        y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
        source = brightness * np.exp(-(x_grid**2 + y_grid**2)/(2*sigma**2))
        image_data += source
    
    return {
        "observation_id": observation_id,
        "image_shape": [size, size],
        "n_sources": n_sources,
        "mean_value": float(np.mean(image_data)),
        "std_value": float(np.std(image_data)),
        "min_value": float(np.min(image_data)),
        "max_value": float(np.max(image_data))
    }

# ==================== ENDPOINTS SPECTROSCOPIE ====================

@app.post("/spectra", tags=["Spectroscopy"])
async def create_spectrum(
    spectrum: SpectrumCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Acquérir spectre d'une cible"""
    if spectrum.target_id not in fake_db["targets"]:
        raise HTTPException(status_code=404, detail="Target not found")
    
    spectrum_id = str(uuid.uuid4())
    
    # Générer spectre simulé
    wavelengths = np.linspace(spectrum.wavelength_range_nm[0], 
                             spectrum.wavelength_range_nm[1], 
                             spectrum.resolution)
    
    # Spectre corps noir + raies
    temp = np.random.uniform(3000, 10000)  # K
    flux = np.exp(-(wavelengths - 550)**2 / (100**2))  # Simplifié
    flux += np.random.normal(0, 0.01, len(wavelengths))
    
    # SNR
    snr = np.sqrt(spectrum.integration_time_s) * 5
    
    spectrum_data = {
        "id": spectrum_id,
        "target_id": spectrum.target_id,
        "wavelengths": wavelengths.tolist(),
        "flux": flux.tolist(),
        "snr": snr,
        "temperature_K": temp,
        "resolution": spectrum.resolution,
        "timestamp": datetime.now(),
        "owner": current_user.username
    }
    
    fake_db["spectra"][spectrum_id] = spectrum_data
    
    return {
        "spectrum_id": spectrum_id,
        "snr": snr,
        "temperature_K": temp,
        "n_lines_detected": np.random.randint(5, 20)
    }

@app.get("/spectra/{spectrum_id}", tags=["Spectroscopy"])
async def get_spectrum(
    spectrum_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir spectre complet"""
    if spectrum_id not in fake_db["spectra"]:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    
    return fake_db["spectra"][spectrum_id]

@app.post("/spectra/{spectrum_id}/analyze", tags=["Spectroscopy"])
async def analyze_spectrum(
    spectrum_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Analyser raies spectrales"""
    if spectrum_id not in fake_db["spectra"]:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    
    spectrum = fake_db["spectra"][spectrum_id]
    
    # Identifier raies (simulé)
    lines_db = {
        'H-alpha': 656.3,
        'H-beta': 486.1,
        'Na D': 589.0,
        'Ca II K': 393.4,
        'Ca II H': 396.8,
        'Mg I': 518.4,
        'Fe I': 440.5
    }
    
    detected_lines = []
    for name, wavelength in lines_db.items():
        if spectrum["wavelengths"][0] <= wavelength <= spectrum["wavelengths"][-1]:
            if np.random.random() > 0.3:  # Probabilité détection
                detected_lines.append({
                    "line": name,
                    "wavelength_nm": wavelength,
                    "equivalent_width_angstrom": np.random.uniform(0.1, 2.0),
                    "snr": np.random.uniform(5, 50)
                })
    
    # Mesurer redshift
    redshift = np.random.uniform(0, 0.5)
    
    return {
        "spectrum_id": spectrum_id,
        "lines_detected": detected_lines,
        "n_lines": len(detected_lines),
        "redshift": redshift,
        "radial_velocity_km_s": redshift * 299792
    }

# ==================== ENDPOINTS EXOPLANÈTES ====================

@app.post("/exoplanets", response_model=ExoplanetResponse, tags=["Exoplanets"])
async def create_exoplanet(
    exoplanet: ExoplanetCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Enregistrer candidat exoplanète"""
    exoplanet_id = str(uuid.uuid4())
    
    # Calculer profondeur transit si méthode transit
    transit_depth = None
    if exoplanet.detection_method == "Transit":
        R_earth = 6.371e6
        R_sun = 6.96e8
        transit_depth = (exoplanet.radius_r_earth * R_earth)**2 / R_sun**2 * 100
    
    exoplanet_data = {
        "id": exoplanet_id,
        **exoplanet.dict(),
        "transit_depth": transit_depth,
        "confirmed": False,
        "timestamp": datetime.now(),
        "discoverer": current_user.username
    }
    
    fake_db["exoplanets"][exoplanet_id] = exoplanet_data
    
    return ExoplanetResponse(**exoplanet_data)

@app.get("/exoplanets", response_model=List[ExoplanetResponse], tags=["Exoplanets"])
async def list_exoplanets(
    skip: int = 0,
    limit: int = 100,
    confirmed: Optional[bool] = None,
    min_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister exoplanètes"""
    exoplanets = list(fake_db["exoplanets"].values())
    
    if confirmed is not None:
        exoplanets = [e for e in exoplanets if e["confirmed"] == confirmed]
    
    if min_radius is not None:
        exoplanets = [e for e in exoplanets if e["radius_r_earth"] >= min_radius]
    
    if max_radius is not None:
        exoplanets = [e for e in exoplanets if e["radius_r_earth"] <= max_radius]
    
    return [ExoplanetResponse(**e) for e in exoplanets[skip:skip+limit]]

@app.post("/exoplanets/{exoplanet_id}/simulate-transit", tags=["Exoplanets"])
async def simulate_exoplanet_transit(
    exoplanet_id: str,
    star_radius_rsun: float = 1.0,
    transit_duration_h: float = 3.0,
    current_user: User = Depends(get_current_active_user)
):
    """Simuler courbe de transit"""
    if exoplanet_id not in fake_db["exoplanets"]:
        raise HTTPException(status_code=404, detail="Exoplanet not found")
    
    exoplanet = fake_db["exoplanets"][exoplanet_id]
    
    # Calculer profondeur transit
    R_earth = 6.371e6
    R_sun = 6.96e8
    depth_percent = (exoplanet["radius_r_earth"] * R_earth)**2 / (star_radius_rsun * R_sun)**2 * 100
    
    # Générer courbe
    transit_data = simulate_transit(
        exoplanet["period_days"],
        transit_duration_h,
        depth_percent
    )
    
    return {
        "exoplanet_id": exoplanet_id,
        "transit_data": transit_data,
        "star_radius_rsun": star_radius_rsun
    }

@app.put("/exoplanets/{exoplanet_id}/confirm", tags=["Exoplanets"])
async def confirm_exoplanet(
    exoplanet_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Confirmer exoplanète"""
    if exoplanet_id not in fake_db["exoplanets"]:
        raise HTTPException(status_code=404, detail="Exoplanet not found")
    
    fake_db["exoplanets"][exoplanet_id]["confirmed"] = True
    fake_db["exoplanets"][exoplanet_id]["confirmation_date"] = datetime.now()
    
    return {"message": "Exoplanet confirmed!", "exoplanet_id": exoplanet_id}

# ==================== ENDPOINTS GALAXIES ====================

@app.post("/galaxies", response_model=GalaxyResponse, tags=["Galaxies"])
async def create_galaxy(
    galaxy: GalaxyCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Cataloguer nouvelle galaxie"""
    galaxy_id = str(uuid.uuid4())
    
    # Calculer distance cosmologique
    H0 = 70  # km/s/Mpc
    c = 299792.458  # km/s
    distance_mpc = c * galaxy.redshift / H0
    
    galaxy_data = {
        "id": galaxy_id,
        **galaxy.dict(),
        "distance_mpc": distance_mpc,
        "detected": datetime.now(),
        "cataloger": current_user.username
    }
    
    fake_db["galaxies"][galaxy_id] = galaxy_data
    
    return GalaxyResponse(**galaxy_data)

@app.get("/galaxies", response_model=List[GalaxyResponse], tags=["Galaxies"])
async def list_galaxies(
    skip: int = 0,
    limit: int = 100,
    min_redshift: Optional[float] = None,
    max_redshift: Optional[float] = None,
    galaxy_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister galaxies cataloguées"""
    galaxies = list(fake_db["galaxies"].values())
    
    if min_redshift is not None:
        galaxies = [g for g in galaxies if g["redshift"] >= min_redshift]
    
    if max_redshift is not None:
        galaxies = [g for g in galaxies if g["redshift"] <= max_redshift]
    
    if galaxy_type:
        galaxies = [g for g in galaxies if g["type"].startswith(galaxy_type)]
    
    return [GalaxyResponse(**g) for g in galaxies[skip:skip+limit]]

@app.get("/galaxies/hubble-diagram", tags=["Galaxies"])
async def get_hubble_diagram(
    current_user: User = Depends(get_current_active_user)
):
    """Obtenir données pour diagramme de Hubble"""
    galaxies = list(fake_db["galaxies"].values())
    
    if not galaxies:
        return {"message": "No galaxies in catalog"}
    
    redshifts = [g["redshift"] for g in galaxies]
    distances = [g["distance_mpc"] for g in galaxies]
    
    # Calculer H0 observé (régression linéaire simple)
    if len(redshifts) > 1:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(distances, redshifts)
        H0_measured = slope * 299792.458
    else:
        H0_measured = 70.0
        r_value = 0.0
    
    return {
        "n_galaxies": len(galaxies),
        "redshifts": redshifts,
        "distances_mpc": distances,
        "H0_measured_km_s_Mpc": H0_measured,
        "correlation_r2": r_value**2
    }

# ==================== ENDPOINTS TROUS NOIRS ====================

@app.post("/black-holes", response_model=BlackHoleResponse, tags=["Black Holes"])
async def create_black_hole(
    black_hole: BlackHoleCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Analyser trou noir"""
    bh_id = str(uuid.uuid4())
    
    # Calculs
    rs_km = calculate_schwarzschild_radius(black_hole.mass_solar)
    r_isco = calculate_isco(black_hole.mass_solar, black_hole.spin)
    T_hawking = 6.17e-8 / black_hole.mass_solar
    
    bh_data = {
        "id": bh_id,
        **black_hole.dict(),
        "schwarzschild_radius_km": rs_km,
        "isco_rs": r_isco,
        "hawking_temp_K": T_hawking,
        "timestamp": datetime.now(),
        "analyst": current_user.username
    }
    
    fake_db["black_holes"][bh_id] = bh_data
    
    return BlackHoleResponse(**bh_data)

@app.get("/black-holes", response_model=List[BlackHoleResponse], tags=["Black Holes"])
async def list_black_holes(
    skip: int = 0,
    limit: int = 100,
    min_mass: Optional[float] = None,
    max_mass: Optional[float] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Lister trous noirs catalogués"""
    black_holes = list(fake_db["black_holes"].values())
    
    if min_mass is not None:
        black_holes = [bh for bh in black_holes if bh["mass_solar"] >= min_mass]
    
    if max_mass is not None:
        black_holes = [bh for bh in black_holes if bh["mass_solar"] <= max_mass]
    
    return [BlackHoleResponse(**bh) for bh in black_holes[skip:skip+limit]]

@app.post("/black-holes/{bh_id}/accretion-disk", tags=["Black Holes"])
async def simulate_accretion_disk(
    bh_id: str,
    accretion_rate_msun_yr: float = 1.0,
    current_user: User = Depends(get_current_active_user)
):
    """Simuler disque d'accrétion"""
    if bh_id not in fake_db["black_holes"]:
        raise HTTPException(status_code=404, detail="Black hole not found")
    
    bh = fake_db["black_holes"][bh_id]
    
    # Efficacité accrétion
    if bh["spin"] < 0.1:
        efficiency = 0.057
    else:
        efficiency = 0.057 + 0.32 * bh["spin"]
    
    # Luminosité
    M_sun_kg = 1.989e30
    c = 299792458
    M_dot_kg_s = accretion_rate_msun_yr * M_sun_kg / (365.25 * 24 * 3600)
    L_bol = efficiency * M_dot_kg_s * c**2
    
    # Luminosité Eddington
    L_edd = 1.26e38 * bh["mass_solar"]
    eddington_ratio = L_bol / L_edd
    
    # Profil température
    r_range = np.logspace(np.log10(bh["isco_rs"]), 3, 50)
    T_profile = 3e6 * (bh["mass_solar"] / 1e8)**(-0.25) * r_range**(-0.75)
    
    return {
        "black_hole_id": bh_id,
        "accretion_rate_msun_yr": accretion_rate_msun_yr,
        "efficiency": efficiency,
        "luminosity_bol_W": L_bol,
        "luminosity_eddington_W": L_edd,
        "eddington_ratio": eddington_ratio,
        "radius_rs": r_range.tolist(),
        "temperature_K": T_profile.tolist()
    }

# ==================== ENDPOINTS STATISTIQUES ====================

@app.get("/stats/overview", tags=["Statistics"])
async def get_overview_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Statistiques générales observatoire"""
    return {
        "telescopes": len(fake_db["telescopes"]),
        "targets": len(fake_db["targets"]),
        "observations": len(fake_db["observations"]),
        "spectra": len(fake_db["spectra"]),
        "exoplanets": len(fake_db["exoplanets"]),
        "galaxies": len(fake_db["galaxies"]),
        "black_holes": len(fake_db["black_holes"]),
        "total_observing_time_hours": len(fake_db["observations"]) * 1.0,  # Simplifié
        "discoveries": len(fake_db["discoveries"])
    }

@app.get("/stats/telescope/{telescope_id}", tags=["Statistics"])
async def get_telescope_stats(
    telescope_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Statistiques d'un télescope"""
    if telescope_id not in fake_db["telescopes"]:
        raise HTTPException(status_code=404, detail="Telescope not found")
    
    telescope_obs = [o for o in fake_db["observations"].values() 
                    if o["telescope_id"] == telescope_id]
    
    total_time = sum(o["exposure_time_s"] * o["n_exposures"] for o in telescope_obs)
    
    return {
        "telescope_id": telescope_id,
        "n_observations": len(telescope_obs),
        "total_observing_time_s": total_time,
        "total_observing_time_hours": total_time / 3600,
        "avg_snr": np.mean([o["snr"] for o in telescope_obs]) if telescope_obs else 0
    }

# ==================== ENDPOINTS RECHERCHE ====================

@app.get("/search/objects", tags=["Search"])
async def search_objects(
    query: str,
    object_types: Optional[List[str]] = None,
    max_results: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """Rechercher objets célestes"""
    results = []
    
    # Rechercher dans cibles
    for target_id, target in fake_db["targets"].items():
        if query.lower() in target["name"].lower():
            if not object_types or target["object_type"] in object_types:
                results.append({
                    "type": "target",
                    "id": target_id,
                    "name": target["name"],
                    "object_type": target["object_type"],
                    "magnitude": target["magnitude"]
                })
    
    return {"query": query, "results": results[:max_results], "n_results": len(results)}

# ==================== ENDPOINT RACINE ====================

@app.get("/", tags=["Root"])
async def root():
    """Page d'accueil API"""
    return {
        "message": "🔭 Space Telescope API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "authentication": "/token, /register",
            "telescopes": "/telescopes",
            "targets": "/targets",
            "observations": "/observations",
            "spectra": "/spectra",
            "exoplanets": "/exoplanets",
            "galaxies": "/galaxies",
            "black_holes": "/black-holes",
            "stats": "/stats"
        }
    }

@app.get("/health", tags=["Root"])
async def health_check():
    """Vérification santé API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }

# ==================== LANCEMENT ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)