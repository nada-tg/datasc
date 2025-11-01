"""
API Backend Python pour Plateforme AR/VR Avancée
FastAPI + SQLAlchemy + PostgreSQL
Version: 1.0.0

Installation:
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib python-multipart redis celery

Lancement:
uvicorn arvr_platform_api:app --reload --host 0.0.0.0 --port 8008
"""

from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
from enum import Enum

# ==================== CONFIGURATION ====================

# Configuration Base de Données
DATABASE_URL = "postgresql://postgres:nadaprojet@localhost:5432/arvr_db"

# Pour SQLite (dev): DATABASE_URL = "sqlite:///./arvr.db"

# Configuration JWT
SECRET_KEY = "votre-cle-secrete-super-securisee-changez-moi"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuration Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== INITIALISATION ====================

# FastAPI App
app = FastAPI(
    title="AR/VR Platform API",
    description="API complète pour plateforme AR/VR avancée",
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

# SQLAlchemy
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ==================== ENUMS ====================

class DeviceType(str, Enum):
    VR = "VR"
    AR = "AR"
    MR = "MR"
    XR = "XR"

class UserRole(str, Enum):
    USER = "user"
    CREATOR = "creator"
    ADMIN = "admin"

class EnvironmentType(str, Enum):
    PLANET = "Planet"
    CITY = "City"
    NATURE = "Nature"
    SPACE = "Space"
    INTERIOR = "Interior"

# ==================== MODELS DATABASE ====================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default=UserRole.USER.value)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relations
    devices = relationship("Device", back_populates="owner")
    applications = relationship("Application", back_populates="creator")
    environments = relationship("Environment", back_populates="creator")

class Device(Base):
    __tablename__ = "devices"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    specs = Column(JSON)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="devices")

class Application(Base):
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String)
    type = Column(String)
    description = Column(String)
    features = Column(JSON)
    requirements = Column(JSON)
    downloads = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    active_users = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    creator_id = Column(Integer, ForeignKey("users.id"))
    
    creator = relationship("User", back_populates="applications")

class Environment(Base):
    __tablename__ = "environments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    type = Column(String)
    size_km = Column(Float)
    detail_level = Column(String)
    generation_method = Column(String)
    object_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    creator_id = Column(Integer, ForeignKey("users.id"))
    
    creator = relationship("User", back_populates="environments")

class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String)
    metric_name = Column(String)
    value = Column(Float)
    metadata_json = Column("metadata", JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Créer toutes les tables
Base.metadata.create_all(bind=engine)

# ==================== SCHEMAS PYDANTIC ====================

# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Device Schemas
class DeviceSpecs(BaseModel):
    display: Dict[str, Any]
    tracking: Dict[str, Any]
    performance: Dict[str, Any]
    features: List[str] = []

class DeviceCreate(BaseModel):
    name: str
    type: DeviceType
    specs: DeviceSpecs

class DeviceResponse(BaseModel):
    id: int
    name: str
    type: str
    specs: Dict
    status: str
    created_at: datetime
    owner_id: int
    
    class Config:
        from_attributes = True

# Application Schemas
class ApplicationCreate(BaseModel):
    name: str
    category: str
    type: str
    description: str
    features: List[str] = []
    requirements: Dict[str, Any]

class ApplicationResponse(BaseModel):
    id: int
    name: str
    category: str
    type: str
    description: str
    downloads: int
    rating: float
    active_users: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Environment Schemas
class EnvironmentCreate(BaseModel):
    name: str
    type: EnvironmentType
    size_km: float
    detail_level: str
    generation_method: str = "AI"

class EnvironmentResponse(BaseModel):
    id: int
    name: str
    type: str
    size_km: float
    detail_level: str
    object_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Analytics Schemas
class AnalyticsCreate(BaseModel):
    metric_type: str
    metric_name: str
    value: float
    metadata: Optional[Dict] = {}

class AnalyticsResponse(BaseModel):
    id: int
    metric_type: str
    metric_name: str
    value: float
    metadata: Dict
    timestamp: datetime
    
    class Config:
        from_attributes = True

# ==================== DEPENDENCIES ====================

def get_db():
    """Dépendance pour obtenir session DB"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifier password"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Créer JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Obtenir utilisateur courant depuis token"""
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

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Vérifier que l'utilisateur est actif"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ==================== ROUTES AUTH ====================

@app.post("/api/v1/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Enregistrer nouvel utilisateur"""
    # Vérifier si username existe
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Vérifier si email existe
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Créer utilisateur
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"New user registered: {user.username}")
    return db_user

@app.post("/api/v1/auth/token", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Connexion et obtention token"""
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
    
    logger.info(f"User logged in: {user.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/v1/auth/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Obtenir informations utilisateur courant"""
    return current_user

# ==================== ROUTES DEVICES ====================

@app.post("/api/v1/devices", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
async def create_device(
    device: DeviceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer nouveau device AR/VR"""
    db_device = Device(
        name=device.name,
        type=device.type.value,
        specs=device.specs.dict(),
        owner_id=current_user.id
    )
    db.add(db_device)
    db.commit()
    db.refresh(db_device)
    
    logger.info(f"Device created: {device.name} by user {current_user.username}")
    return db_device

@app.get("/api/v1/devices", response_model=List[DeviceResponse])
async def list_devices(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister devices de l'utilisateur"""
    devices = db.query(Device).filter(
        Device.owner_id == current_user.id
    ).offset(skip).limit(limit).all()
    return devices

@app.get("/api/v1/devices/{device_id}", response_model=DeviceResponse)
async def get_device(
    device_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir device spécifique"""
    device = db.query(Device).filter(
        Device.id == device_id,
        Device.owner_id == current_user.id
    ).first()
    
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    return device

@app.put("/api/v1/devices/{device_id}", response_model=DeviceResponse)
async def update_device(
    device_id: int,
    device_update: DeviceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mettre à jour device"""
    device = db.query(Device).filter(
        Device.id == device_id,
        Device.owner_id == current_user.id
    ).first()
    
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    device.name = device_update.name
    device.type = device_update.type.value
    device.specs = device_update.specs.dict()
    
    db.commit()
    db.refresh(device)
    
    logger.info(f"Device updated: {device.name}")
    return device

@app.delete("/api/v1/devices/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_device(
    device_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Supprimer device"""
    device = db.query(Device).filter(
        Device.id == device_id,
        Device.owner_id == current_user.id
    ).first()
    
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found"
        )
    
    db.delete(device)
    db.commit()
    
    logger.info(f"Device deleted: {device_id}")
    return None

# ==================== ROUTES APPLICATIONS ====================

@app.post("/api/v1/applications", response_model=ApplicationResponse, status_code=status.HTTP_201_CREATED)
async def create_application(
    app: ApplicationCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer nouvelle application VR/AR"""
    db_app = Application(
        name=app.name,
        category=app.category,
        type=app.type,
        description=app.description,
        features=app.features,
        requirements=app.requirements,
        creator_id=current_user.id
    )
    db.add(db_app)
    db.commit()
    db.refresh(db_app)
    
    logger.info(f"Application created: {app.name} by {current_user.username}")
    return db_app

@app.get("/api/v1/applications", response_model=List[ApplicationResponse])
async def list_applications(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lister toutes les applications (publiques)"""
    query = db.query(Application)
    
    if category:
        query = query.filter(Application.category == category)
    
    apps = query.offset(skip).limit(limit).all()
    return apps

@app.get("/api/v1/applications/my", response_model=List[ApplicationResponse])
async def list_my_applications(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister applications de l'utilisateur"""
    apps = db.query(Application).filter(
        Application.creator_id == current_user.id
    ).offset(skip).limit(limit).all()
    return apps

@app.get("/api/v1/applications/{app_id}", response_model=ApplicationResponse)
async def get_application(app_id: int, db: Session = Depends(get_db)):
    """Obtenir application spécifique"""
    app = db.query(Application).filter(Application.id == app_id).first()
    
    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )
    return app

@app.post("/api/v1/applications/{app_id}/download")
async def download_application(
    app_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Télécharger/Installer application"""
    app = db.query(Application).filter(Application.id == app_id).first()
    
    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )
    
    # Incrémenter compteur downloads
    app.downloads += 1
    db.commit()
    
    logger.info(f"Application {app.name} downloaded by {current_user.username}")
    return {"message": "Application download started", "app_id": app_id}

@app.post("/api/v1/applications/{app_id}/rate")
async def rate_application(
    app_id: int,
    rating: float,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Noter une application (1-5 étoiles)"""
    if rating < 1 or rating > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rating must be between 1 and 5"
        )
    
    app = db.query(Application).filter(Application.id == app_id).first()
    
    if not app:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Application not found"
        )
    
    # Mise à jour rating (simplifié - en prod: système plus complexe)
    current_total = app.rating * app.downloads
    new_total = current_total + rating
    app.rating = new_total / (app.downloads + 1)
    
    db.commit()
    
    logger.info(f"Application {app.name} rated {rating} by {current_user.username}")
    return {"message": "Rating submitted", "new_rating": app.rating}

# ==================== ROUTES ENVIRONMENTS ====================

@app.post("/api/v1/environments", response_model=EnvironmentResponse, status_code=status.HTTP_201_CREATED)
async def create_environment(
    env: EnvironmentCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Créer nouvel environnement 3D"""
    db_env = Environment(
        name=env.name,
        type=env.type.value,
        size_km=env.size_km,
        detail_level=env.detail_level,
        generation_method=env.generation_method,
        creator_id=current_user.id
    )
    db.add(db_env)
    db.commit()
    db.refresh(db_env)
    
    # Tâche en arrière-plan pour générer l'environnement
    background_tasks.add_task(generate_environment_async, db_env.id)
    
    logger.info(f"Environment created: {env.name} by {current_user.username}")
    return db_env

@app.get("/api/v1/environments", response_model=List[EnvironmentResponse])
async def list_environments(
    skip: int = 0,
    limit: int = 100,
    type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lister tous les environnements"""
    query = db.query(Environment)
    
    if type:
        query = query.filter(Environment.type == type)
    
    envs = query.offset(skip).limit(limit).all()
    return envs

@app.get("/api/v1/environments/{env_id}", response_model=EnvironmentResponse)
async def get_environment(env_id: int, db: Session = Depends(get_db)):
    """Obtenir environnement spécifique"""
    env = db.query(Environment).filter(Environment.id == env_id).first()
    
    if not env:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Environment not found"
        )
    return env

@app.post("/api/v1/environments/{env_id}/teleport")
async def teleport_to_environment(
    env_id: int,
    coordinates: Optional[List[float]] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Téléporter utilisateur vers environnement"""
    env = db.query(Environment).filter(Environment.id == env_id).first()
    
    if not env:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Environment not found"
        )
    
    # Coordonnées par défaut si non spécifiées
    if not coordinates:
        coordinates = [0, 0, 0]
    
    logger.info(f"User {current_user.username} teleported to {env.name}")
    
    return {
        "message": "Teleportation successful",
        "environment": env.name,
        "coordinates": coordinates
    }

# ==================== ROUTES ANALYTICS ====================

@app.post("/api/v1/analytics", response_model=AnalyticsResponse, status_code=status.HTTP_201_CREATED)
async def create_analytics(
    analytics: AnalyticsCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Enregistrer métrique analytics"""
    db_analytics = Analytics(
        metric_type=analytics.metric_type,
        metric_name=analytics.metric_name,
        value=analytics.value,
        metadata=analytics.metadata
    )
    db.add(db_analytics)
    db.commit()
    db.refresh(db_analytics)
    
    return db_analytics

@app.get("/api/v1/analytics/users")
async def get_user_analytics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir analytics utilisateurs"""
    query = db.query(Analytics).filter(Analytics.metric_type == "users")
    
    if start_date:
        query = query.filter(Analytics.timestamp >= start_date)
    if end_date:
        query = query.filter(Analytics.timestamp <= end_date)
    
    metrics = query.all()
    
    # Agréger données
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "metrics": [
            {
                "name": m.metric_name,
                "value": m.value,
                "timestamp": m.timestamp
            } for m in metrics
        ]
    }

@app.get("/api/v1/analytics/performance")
async def get_performance_analytics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir analytics performance"""
    metrics = db.query(Analytics).filter(
        Analytics.metric_type == "performance"
    ).order_by(Analytics.timestamp.desc()).limit(100).all()
    
    return {
        "metrics": [
            {
                "name": m.metric_name,
                "value": m.value,
                "metadata": m.metadata,
                "timestamp": m.timestamp
            } for m in metrics
        ]
    }

@app.get("/api/v1/analytics/business")
async def get_business_analytics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Obtenir analytics business"""
    # Vérifier que l'utilisateur est admin
    if current_user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    metrics = db.query(Analytics).filter(
        Analytics.metric_type == "business"
    ).order_by(Analytics.timestamp.desc()).limit(100).all()
    
    return {
        "metrics": [
            {
                "name": m.metric_name,
                "value": m.value,
                "metadata": m.metadata,
                "timestamp": m.timestamp
            } for m in metrics
        ]
    }

# ==================== ROUTES ADMIN ====================

@app.get("/api/v1/admin/users", response_model=List[UserResponse])
async def list_all_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Lister tous les utilisateurs (Admin uniquement)"""
    if current_user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@app.put("/api/v1/admin/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    new_role: UserRole,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Modifier rôle utilisateur (Admin uniquement)"""
    if current_user.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user.role = new_role.value
    db.commit()
    
    logger.info(f"User {user.username} role changed to {new_role.value}")
    return {"message": "Role updated successfully", "new_role": new_role.value}

# ==================== ROUTES STATS ====================

@app.get("/api/v1/stats/summary")
async def get_stats_summary(db: Session = Depends(get_db)):
    """Obtenir statistiques globales"""
    total_users = db.query(User).count()
    total_devices = db.query(Device).count()
    total_apps = db.query(Application).count()
    total_envs = db.query(Environment).count()
    
    return {
        "total_users": total_users,
        "total_devices": total_devices,
        "total_applications": total_apps,
        "total_environments": total_envs,
        "timestamp": datetime.utcnow()
    }

# ==================== TÂCHES EN ARRIÈRE-PLAN ====================

def generate_environment_async(env_id: int):
    """Générer environnement de manière asynchrone"""
    logger.info(f"Starting environment generation for env_id: {env_id}")
    
    # Simulation génération (en prod: vraie génération IA/procédurale)
    import time
    time.sleep(5)
    
    # Mettre à jour object_count
    db = SessionLocal()
    env = db.query(Environment).filter(Environment.id == env_id).first()
    if env:
        env.object_count = 10000  # Simulation
        db.commit()
    db.close()
    
    logger.info(f"Environment generation completed for env_id: {env_id}")

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AR/VR Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)