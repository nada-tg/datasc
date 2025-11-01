"""
🏢 Datacenter Management Platform - Complete API
FastAPI + SQLAlchemy + PostgreSQL + Redis + Celery

Installation:
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic python-jose passlib bcrypt redis celery prometheus-client

Lancement:
uvicorn datacenter_platform_api:app --reload --host 0.0.0.0 --port 8023
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, WebSocket
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import numpy as np
import json
import asyncio

# ==================== CONFIGURATION ====================
DATABASE_URL = "postgresql://postgres:nadaprojet@localhost:5432/datacenter_db"
# Pour SQLite: DATABASE_URL = "sqlite:///./datacenter.db"

SECRET_KEY = "datacenter-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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
    role = Column(String, default="operator")
    security_clearance = Column(String, default="Public")
    mfa_enabled = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Datacenter(Base):
    __tablename__ = "datacenters"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    location = Column(String)
    tier = Column(String, default="Tier 3")
    total_space_sqm = Column(Float)
    power_capacity_mw = Column(Float)
    cooling_capacity_mw = Column(Float)
    pue_target = Column(Float, default=1.5)
    status = Column(String, default="operational")
    certifications = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    
    racks = relationship("Rack", back_populates="datacenter")

class Rack(Base):
    __tablename__ = "racks"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    datacenter_id = Column(Integer, ForeignKey("datacenters.id"))
    location = Column(String)
    row = Column(String)
    position = Column(Integer)
    u_capacity = Column(Integer, default=42)
    u_used = Column(Integer, default=0)
    power_capacity_kw = Column(Float)
    power_used_kw = Column(Float, default=0)
    cooling_type = Column(String)
    security_zone = Column(String)
    temperature_c = Column(Float, default=22.0)
    humidity_pct = Column(Float, default=50.0)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    datacenter = relationship("Datacenter", back_populates="racks")
    servers = relationship("Server", back_populates="rack")

class Server(Base):
    __tablename__ = "servers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    rack_id = Column(Integer, ForeignKey("racks.id"))
    server_type = Column(String)
    manufacturer = Column(String)
    model = Column(String)
    serial_number = Column(String, unique=True)
    u_position = Column(Integer)
    u_size = Column(Integer, default=1)
    cpu_model = Column(String)
    cpu_cores = Column(Integer)
    ram_gb = Column(Integer)
    storage_tb = Column(Float)
    network_ports = Column(Integer, default=2)
    power_supply_w = Column(Integer)
    os = Column(String)
    ip_address = Column(String)
    management_ip = Column(String)
    status = Column(String, default="online")
    health_score = Column(Float, default=100.0)
    warranty_expiry = Column(DateTime)
    last_boot = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    rack = relationship("Rack", back_populates="servers")
    metrics = relationship("ServerMetric", back_populates="server")

class ServerMetric(Base):
    __tablename__ = "server_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    server_id = Column(Integer, ForeignKey("servers.id"))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_usage_pct = Column(Float)
    memory_usage_pct = Column(Float)
    disk_usage_pct = Column(Float)
    network_in_mbps = Column(Float)
    network_out_mbps = Column(Float)
    power_consumption_w = Column(Float)
    temperature_c = Column(Float)
    
    server = relationship("Server", back_populates="metrics")

class StorageSystem(Base):
    __tablename__ = "storage_systems"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    storage_type = Column(String)
    capacity_tb = Column(Float)
    used_tb = Column(Float, default=0)
    raid_level = Column(String)
    iops = Column(Integer)
    throughput_gbps = Column(Float)
    latency_ms = Column(Float)
    replication = Column(String)
    encryption = Column(Boolean, default=True)
    status = Column(String, default="online")
    health = Column(Float, default=100.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class NetworkDevice(Base):
    __tablename__ = "network_devices"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    device_type = Column(String)
    manufacturer = Column(String)
    model = Column(String)
    ports = Column(Integer)
    ports_used = Column(Integer, default=0)
    speed_gbps = Column(Integer)
    management_ip = Column(String)
    firmware_version = Column(String)
    redundancy = Column(String)
    status = Column(String, default="online")
    cpu_usage = Column(Float, default=0)
    memory_usage = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class CoolingSystem(Base):
    __tablename__ = "cooling_systems"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    cooling_type = Column(String)
    capacity_kw = Column(Float)
    current_load_kw = Column(Float, default=0)
    efficiency_pue = Column(Float)
    temperature_setpoint_c = Column(Float, default=22.0)
    humidity_setpoint_pct = Column(Float, default=50.0)
    status = Column(String, default="operational")
    power_consumption_kw = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class PowerSystem(Base):
    __tablename__ = "power_systems"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    system_type = Column(String)
    capacity_kw = Column(Float)
    current_load_kw = Column(Float, default=0)
    voltage_v = Column(Float)
    frequency_hz = Column(Float, default=50.0)
    redundancy = Column(String)
    ups_runtime_minutes = Column(Integer, default=0)
    generator_fuel_liters = Column(Float, default=0)
    status = Column(String, default="normal")
    created_at = Column(DateTime, default=datetime.utcnow)

class VirtualMachine(Base):
    __tablename__ = "virtual_machines"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    host_server_id = Column(Integer, ForeignKey("servers.id"), nullable=True)
    os = Column(String)
    cpu_cores = Column(Integer)
    ram_gb = Column(Integer)
    disk_gb = Column(Integer)
    ip_address = Column(String, nullable=True)
    vlan = Column(String, nullable=True)
    network_type = Column(String)
    status = Column(String, default="running")
    autostart = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Container(Base):
    __tablename__ = "containers"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    image = Column(String)
    tag = Column(String, default="latest")
    replicas = Column(Integer, default=1)
    cpu_limit = Column(Float)
    memory_limit_gb = Column(Float)
    port = Column(Integer)
    status = Column(String, default="running")
    created_at = Column(DateTime, default=datetime.utcnow)

class SecurityZone(Base):
    __tablename__ = "security_zones"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    level = Column(String)
    access_requirements = Column(JSON)
    monitoring_enabled = Column(Boolean, default=True)
    two_factor_required = Column(Boolean, default=True)
    biometric_required = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Incident(Base):
    __tablename__ = "incidents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    severity = Column(String)
    status = Column(String, default="open")
    affected_component = Column(String)
    affected_component_id = Column(Integer, nullable=True)
    reported_by = Column(Integer, ForeignKey("users.id"))
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolution = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class MaintenanceSchedule(Base):
    __tablename__ = "maintenance_schedules"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    maintenance_type = Column(String)
    target_component = Column(String)
    target_component_id = Column(Integer)
    scheduled_start = Column(DateTime)
    scheduled_end = Column(DateTime)
    status = Column(String, default="scheduled")
    performed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String)
    severity = Column(String)
    source = Column(String)
    source_id = Column(Integer, nullable=True)
    message = Column(String)
    details = Column(JSON, nullable=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String)
    resource_type = Column(String)
    resource_id = Column(Integer, nullable=True)
    ip_address = Column(String)
    user_agent = Column(String, nullable=True)
    status = Column(String)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class Backup(Base):
    __tablename__ = "backups"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    backup_type = Column(String)
    source_type = Column(String)
    source_id = Column(Integer)
    size_gb = Column(Float)
    location = Column(String)
    status = Column(String, default="completed")
    retention_days = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow)

# Créer tables
Base.metadata.create_all(bind=engine)

# ==================== PYDANTIC SCHEMAS ====================

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: str = "operator"
    security_clearance: str = "Public"

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    mfa_enabled: bool
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class DatacenterBase(BaseModel):
    name: str
    location: str
    tier: str = "Tier 3"
    total_space_sqm: float
    power_capacity_mw: float
    cooling_capacity_mw: float
    pue_target: float = 1.5

class DatacenterCreate(DatacenterBase):
    pass

class DatacenterResponse(DatacenterBase):
    id: int
    status: str
    certifications: List[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class RackBase(BaseModel):
    name: str
    datacenter_id: int
    location: str
    row: str
    position: int
    u_capacity: int = 42
    power_capacity_kw: float
    cooling_type: str
    security_zone: str

class RackCreate(RackBase):
    pass

class RackResponse(RackBase):
    id: int
    u_used: int
    power_used_kw: float
    temperature_c: float
    humidity_pct: float
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ServerBase(BaseModel):
    name: str
    rack_id: int
    server_type: str
    manufacturer: str
    model: str
    serial_number: str
    u_position: int
    u_size: int = 1
    cpu_model: str
    cpu_cores: int
    ram_gb: int
    storage_tb: float
    power_supply_w: int
    os: str
    ip_address: str
    management_ip: str

class ServerCreate(ServerBase):
    pass

class ServerUpdate(BaseModel):
    status: Optional[str] = None
    health_score: Optional[float] = None
    os: Optional[str] = None

class ServerResponse(ServerBase):
    id: int
    status: str
    health_score: float
    warranty_expiry: Optional[datetime]
    last_boot: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class ServerMetricCreate(BaseModel):
    server_id: int
    cpu_usage_pct: float
    memory_usage_pct: float
    disk_usage_pct: float
    network_in_mbps: float
    network_out_mbps: float
    power_consumption_w: float
    temperature_c: float

class ServerMetricResponse(ServerMetricCreate):
    id: int
    timestamp: datetime
    
    class Config:
        from_attributes = True

class StorageSystemBase(BaseModel):
    name: str
    storage_type: str
    capacity_tb: float
    raid_level: str
    iops: int
    throughput_gbps: float
    replication: str
    encryption: bool = True

class StorageSystemCreate(StorageSystemBase):
    pass

class StorageSystemResponse(StorageSystemBase):
    id: int
    used_tb: float
    latency_ms: float
    status: str
    health: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class NetworkDeviceBase(BaseModel):
    name: str
    device_type: str
    manufacturer: str
    model: str
    ports: int
    speed_gbps: int
    management_ip: str
    redundancy: str

class NetworkDeviceCreate(NetworkDeviceBase):
    pass

class NetworkDeviceResponse(NetworkDeviceBase):
    id: int
    ports_used: int
    firmware_version: Optional[str]
    status: str
    cpu_usage: float
    memory_usage: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class VirtualMachineBase(BaseModel):
    name: str
    host_server_id: Optional[int] = None
    os: str
    cpu_cores: int
    ram_gb: int
    disk_gb: int
    network_type: str
    vlan: Optional[str] = None

class VirtualMachineCreate(VirtualMachineBase):
    pass

class VirtualMachineResponse(VirtualMachineBase):
    id: int
    ip_address: Optional[str]
    status: str
    autostart: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class ContainerBase(BaseModel):
    name: str
    image: str
    tag: str = "latest"
    replicas: int = 1
    cpu_limit: float
    memory_limit_gb: float
    port: int

class ContainerCreate(ContainerBase):
    pass

class ContainerResponse(ContainerBase):
    id: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class IncidentBase(BaseModel):
    title: str
    description: str
    severity: str
    affected_component: str
    affected_component_id: Optional[int] = None

class IncidentCreate(IncidentBase):
    pass

class IncidentResponse(IncidentBase):
    id: int
    status: str
    reported_by: int
    assigned_to: Optional[int]
    resolution: Optional[str]
    created_at: datetime
    resolved_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class AlertBase(BaseModel):
    alert_type: str
    severity: str
    source: str
    source_id: Optional[int] = None
    message: str
    details: Optional[Dict[str, Any]] = None

class AlertCreate(AlertBase):
    pass

class AlertResponse(AlertBase):
    id: int
    acknowledged: bool
    acknowledged_by: Optional[int]
    resolved: bool
    resolved_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class MaintenanceScheduleBase(BaseModel):
    title: str
    description: str
    maintenance_type: str
    target_component: str
    target_component_id: int
    scheduled_start: datetime
    scheduled_end: datetime

class MaintenanceScheduleCreate(MaintenanceScheduleBase):
    pass

class MaintenanceScheduleResponse(MaintenanceScheduleBase):
    id: int
    status: str
    performed_by: Optional[int]
    notes: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class DashboardMetrics(BaseModel):
    total_racks: int
    total_servers: int
    total_vms: int
    total_containers: int
    power_consumption_kw: float
    pue: float
    avg_temperature_c: float
    capacity_utilization_pct: float
    uptime_pct: float
    active_alerts: int

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
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ==================== UTILITY FUNCTIONS ====================

def log_audit(db: Session, user_id: int, action: str, resource_type: str, 
              resource_id: Optional[int], status: str, ip_address: str = "0.0.0.0"):
    """Log audit event"""
    audit = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=ip_address,
        status=status
    )
    db.add(audit)
    db.commit()

def calculate_pue(db: Session) -> float:
    """Calculate Power Usage Effectiveness"""
    total_power = db.query(PowerSystem).with_entities(
        db.func.sum(PowerSystem.current_load_kw)
    ).scalar() or 0
    
    it_power = db.query(Server).count() * 0.3  # Simplified
    
    if it_power == 0:
        return 0
    return total_power / it_power

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Datacenter Management Platform API",
    description="Complete datacenter infrastructure management API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ROUTES AUTHENTICATION ====================

@app.post("/register", response_model=UserResponse, tags=["Authentication"])
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        security_clearance=user.security_clearance,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get JWT token"""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user.last_login = datetime.utcnow()
    db.commit()
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse, tags=["Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user"""
    return current_user

# ==================== ROUTES DATACENTERS ====================

@app.post("/datacenters", response_model=DatacenterResponse, status_code=status.HTTP_201_CREATED, tags=["Datacenters"])
def create_datacenter(
    dc: DatacenterCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create new datacenter"""
    db_dc = Datacenter(**dc.dict())
    db.add(db_dc)
    db.commit()
    db.refresh(db_dc)
    log_audit(db, current_user.id, "CREATE", "Datacenter", db_dc.id, "success")
    return db_dc

@app.get("/datacenters", response_model=List[DatacenterResponse], tags=["Datacenters"])
def list_datacenters(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all datacenters"""
    datacenters = db.query(Datacenter).offset(skip).limit(limit).all()
    return datacenters

@app.get("/datacenters/{dc_id}", response_model=DatacenterResponse, tags=["Datacenters"])
def get_datacenter(
    dc_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get datacenter details"""
    dc = db.query(Datacenter).filter(Datacenter.id == dc_id).first()
    if not dc:
        raise HTTPException(status_code=404, detail="Datacenter not found")
    return dc

# ==================== ROUTES RACKS ====================

@app.post("/racks", response_model=RackResponse, status_code=status.HTTP_201_CREATED, tags=["Racks"])
def create_rack(
    rack: RackCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create new rack"""
    db_rack = Rack(**rack.dict())
    db.add(db_rack)
    db.commit()
    db.refresh(db_rack)
    log_audit(db, current_user.id, "CREATE", "Rack", db_rack.id, "success")
    return db_rack

@app.get("/racks", response_model=List[RackResponse], tags=["Racks"])
def list_racks(
    datacenter_id: Optional[int] = None,
    security_zone: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List racks with optional filters"""
    query = db.query(Rack)
    if datacenter_id:
        query = query.filter(Rack.datacenter_id == datacenter_id)
    if security_zone:
        query = query.filter(Rack.security_zone == security_zone)
    racks = query.offset(skip).limit(limit).all()
    return racks

@app.get("/racks/{rack_id}", response_model=RackResponse, tags=["Racks"])
def get_rack(
    rack_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get rack details"""
    rack = db.query(Rack).filter(Rack.id == rack_id).first()
    if not rack:
        raise HTTPException(status_code=404, detail="Rack not found")
    return rack

@app.delete("/racks/{rack_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Racks"])
def delete_rack(
    rack_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete rack"""
    rack = db.query(Rack).filter(Rack.id == rack_id).first()
    if not rack:
        raise HTTPException(status_code=404, detail="Rack not found")
    
    # Check if rack has servers
    server_count = db.query(Server).filter(Server.rack_id == rack_id).count()
    if server_count > 0:
        raise HTTPException(status_code=400, detail=f"Cannot delete rack with {server_count} servers")
    
    db.delete(rack)
    db.commit()
    log_audit(db, current_user.id, "DELETE", "Rack", rack_id, "success")
    return None

# ==================== ROUTES SERVERS ====================

@app.post("/servers", response_model=ServerResponse, status_code=status.HTTP_201_CREATED, tags=["Servers"])
def create_server(
    server: ServerCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create new server"""
    # Check rack capacity
    rack = db.query(Rack).filter(Rack.id == server.rack_id).first()
    if not rack:
        raise HTTPException(status_code=404, detail="Rack not found")
    
    if rack.u_used + server.u_size > rack.u_capacity:
        raise HTTPException(status_code=400, detail="Insufficient U space in rack")
    
    db_server = Server(**server.dict(), last_boot=datetime.utcnow())
    db.add(db_server)
    
    # Update rack
    rack.u_used += server.u_size
    rack.power_used_kw += server.power_supply_w / 1000
    
    db.commit()
    db.refresh(db_server)
    log_audit(db, current_user.id, "CREATE", "Server", db_server.id, "success")
    return db_server

@app.get("/servers", response_model=List[ServerResponse], tags=["Servers"])
def list_servers(
    rack_id: Optional[int] = None,
    server_type: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List servers with filters"""
    query = db.query(Server)
    if rack_id:
        query = query.filter(Server.rack_id == rack_id)
    if server_type:
        query = query.filter(Server.server_type == server_type)
    if status:
        query = query.filter(Server.status == status)
    servers = query.offset(skip).limit(limit).all()
    return servers

@app.get("/servers/{server_id}", response_model=ServerResponse, tags=["Servers"])
def get_server(
    server_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get server details"""
    server = db.query(Server).filter(Server.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    return server

@app.put("/servers/{server_id}", response_model=ServerResponse, tags=["Servers"])
def update_server(
    server_id: int,
    server_update: ServerUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update server"""
    db_server = db.query(Server).filter(Server.id == server_id).first()
    if not db_server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    for field, value in server_update.dict(exclude_unset=True).items():
        setattr(db_server, field, value)
    
    db.commit()
    db.refresh(db_server)
    log_audit(db, current_user.id, "UPDATE", "Server", server_id, "success")
    return db_server

@app.post("/servers/{server_id}/reboot", tags=["Servers"])
def reboot_server(
    server_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Reboot server"""
    server = db.query(Server).filter(Server.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    server.last_boot = datetime.utcnow()
    db.commit()
    log_audit(db, current_user.id, "REBOOT", "Server", server_id, "success")
    return {"message": f"Server {server.name} rebooting"}

@app.delete("/servers/{server_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Servers"])
def delete_server(
    server_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete/decommission server"""
    server = db.query(Server).filter(Server.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # Update rack
    rack = db.query(Rack).filter(Rack.id == server.rack_id).first()
    if rack:
        rack.u_used -= server.u_size
        rack.power_used_kw -= server.power_supply_w / 1000
    
    db.delete(server)
    db.commit()
    log_audit(db, current_user.id, "DELETE", "Server", server_id, "success")
    return None

# ==================== ROUTES SERVER METRICS ====================

@app.post("/metrics/servers", response_model=ServerMetricResponse, status_code=status.HTTP_201_CREATED, tags=["Metrics"])
def create_server_metric(
    metric: ServerMetricCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Record server metric"""
    db_metric = ServerMetric(**metric.dict())
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    return db_metric

@app.get("/metrics/servers/{server_id}", response_model=List[ServerMetricResponse], tags=["Metrics"])
def get_server_metrics(
    server_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get server metrics history"""
    query = db.query(ServerMetric).filter(ServerMetric.server_id == server_id)
    
    if start_time:
        query = query.filter(ServerMetric.timestamp >= start_time)
    if end_time:
        query = query.filter(ServerMetric.timestamp <= end_time)
    
    metrics = query.order_by(ServerMetric.timestamp.desc()).limit(limit).all()
    return metrics

# ==================== ROUTES STORAGE ====================

@app.post("/storage", response_model=StorageSystemResponse, status_code=status.HTTP_201_CREATED, tags=["Storage"])
def create_storage_system(
    storage: StorageSystemCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create storage system"""
    db_storage = StorageSystem(**storage.dict(), latency_ms=np.random.uniform(0.5, 2.0))
    db.add(db_storage)
    db.commit()
    db.refresh(db_storage)
    log_audit(db, current_user.id, "CREATE", "Storage", db_storage.id, "success")
    return db_storage

@app.get("/storage", response_model=List[StorageSystemResponse], tags=["Storage"])
def list_storage_systems(
    storage_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List storage systems"""
    query = db.query(StorageSystem)
    if storage_type:
        query = query.filter(StorageSystem.storage_type == storage_type)
    storage_systems = query.offset(skip).limit(limit).all()
    return storage_systems

@app.get("/storage/{storage_id}", response_model=StorageSystemResponse, tags=["Storage"])
def get_storage_system(
    storage_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get storage system details"""
    storage = db.query(StorageSystem).filter(StorageSystem.id == storage_id).first()
    if not storage:
        raise HTTPException(status_code=404, detail="Storage system not found")
    return storage

# ==================== ROUTES NETWORK ====================

@app.post("/network/devices", response_model=NetworkDeviceResponse, status_code=status.HTTP_201_CREATED, tags=["Network"])
def create_network_device(
    device: NetworkDeviceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create network device"""
    db_device = NetworkDevice(**device.dict())
    db.add(db_device)
    db.commit()
    db.refresh(db_device)
    log_audit(db, current_user.id, "CREATE", "NetworkDevice", db_device.id, "success")
    return db_device

@app.get("/network/devices", response_model=List[NetworkDeviceResponse], tags=["Network"])
def list_network_devices(
    device_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List network devices"""
    query = db.query(NetworkDevice)
    if device_type:
        query = query.filter(NetworkDevice.device_type == device_type)
    devices = query.offset(skip).limit(limit).all()
    return devices

@app.get("/network/traffic", tags=["Network"])
def get_network_traffic(
    hours: int = 24,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get simulated network traffic data"""
    traffic_data = []
    for hour in range(hours):
        hour_of_day = hour % 24
        if 9 <= hour_of_day <= 18:
            base = 80
        elif 18 <= hour_of_day <= 23:
            base = 60
        else:
            base = 20
        
        traffic_data.append({
            'hour': hour,
            'inbound_gbps': base + np.random.uniform(-10, 10),
            'outbound_gbps': base * 0.8 + np.random.uniform(-8, 8)
        })
    
    return traffic_data

# ==================== ROUTES VIRTUAL MACHINES ====================

@app.post("/vms", response_model=VirtualMachineResponse, status_code=status.HTTP_201_CREATED, tags=["Virtual Machines"])
def create_vm(
    vm: VirtualMachineCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create virtual machine"""
    db_vm = VirtualMachine(
        **vm.dict(),
        ip_address=f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    )
    db.add(db_vm)
    db.commit()
    db.refresh(db_vm)
    log_audit(db, current_user.id, "CREATE", "VM", db_vm.id, "success")
    return db_vm

@app.get("/vms", response_model=List[VirtualMachineResponse], tags=["Virtual Machines"])
def list_vms(
    host_server_id: Optional[int] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List virtual machines"""
    query = db.query(VirtualMachine)
    if host_server_id:
        query = query.filter(VirtualMachine.host_server_id == host_server_id)
    if status:
        query = query.filter(VirtualMachine.status == status)
    vms = query.offset(skip).limit(limit).all()
    return vms

@app.post("/vms/{vm_id}/power", tags=["Virtual Machines"])
def vm_power_action(
    vm_id: int,
    action: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Power control for VM (start/stop/restart)"""
    vm = db.query(VirtualMachine).filter(VirtualMachine.id == vm_id).first()
    if not vm:
        raise HTTPException(status_code=404, detail="VM not found")
    
    if action == "start":
        vm.status = "running"
    elif action == "stop":
        vm.status = "stopped"
    elif action == "restart":
        vm.status = "running"
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    db.commit()
    log_audit(db, current_user.id, f"VM_{action.upper()}", "VM", vm_id, "success")
    return {"message": f"VM {action} successful"}

@app.delete("/vms/{vm_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Virtual Machines"])
def delete_vm(
    vm_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete virtual machine"""
    vm = db.query(VirtualMachine).filter(VirtualMachine.id == vm_id).first()
    if not vm:
        raise HTTPException(status_code=404, detail="VM not found")
    
    db.delete(vm)
    db.commit()
    log_audit(db, current_user.id, "DELETE", "VM", vm_id, "success")
    return None

# ==================== ROUTES CONTAINERS ====================

@app.post("/containers", response_model=ContainerResponse, status_code=status.HTTP_201_CREATED, tags=["Containers"])
def create_container(
    container: ContainerCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Deploy container"""
    db_container = Container(**container.dict())
    db.add(db_container)
    db.commit()
    db.refresh(db_container)
    log_audit(db, current_user.id, "CREATE", "Container", db_container.id, "success")
    return db_container

@app.get("/containers", response_model=List[ContainerResponse], tags=["Containers"])
def list_containers(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List containers"""
    query = db.query(Container)
    if status:
        query = query.filter(Container.status == status)
    containers = query.offset(skip).limit(limit).all()
    return containers

@app.post("/containers/{container_id}/scale", tags=["Containers"])
def scale_container(
    container_id: int,
    replicas: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Scale container replicas"""
    container = db.query(Container).filter(Container.id == container_id).first()
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    container.replicas = replicas
    db.commit()
    log_audit(db, current_user.id, "SCALE", "Container", container_id, "success")
    return {"message": f"Container scaled to {replicas} replicas"}

# ==================== ROUTES INCIDENTS ====================

@app.post("/incidents", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED, tags=["Incidents"])
def create_incident(
    incident: IncidentCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create incident"""
    db_incident = Incident(**incident.dict(), reported_by=current_user.id)
    db.add(db_incident)
    db.commit()
    db.refresh(db_incident)
    log_audit(db, current_user.id, "CREATE", "Incident", db_incident.id, "success")
    return db_incident

@app.get("/incidents", response_model=List[IncidentResponse], tags=["Incidents"])
def list_incidents(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List incidents"""
    query = db.query(Incident)
    if status:
        query = query.filter(Incident.status == status)
    if severity:
        query = query.filter(Incident.severity == severity)
    incidents = query.order_by(Incident.created_at.desc()).offset(skip).limit(limit).all()
    return incidents

@app.put("/incidents/{incident_id}/resolve", tags=["Incidents"])
def resolve_incident(
    incident_id: int,
    resolution: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Resolve incident"""
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    incident.status = "resolved"
    incident.resolution = resolution
    incident.resolved_at = datetime.utcnow()
    db.commit()
    log_audit(db, current_user.id, "RESOLVE", "Incident", incident_id, "success")
    return {"message": "Incident resolved"}

# ==================== ROUTES ALERTS ====================

@app.post("/alerts", response_model=AlertResponse, status_code=status.HTTP_201_CREATED, tags=["Alerts"])
def create_alert(
    alert: AlertCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create alert"""
    db_alert = Alert(**alert.dict())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

@app.get("/alerts", response_model=List[AlertResponse], tags=["Alerts"])
def list_alerts(
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List alerts"""
    query = db.query(Alert)
    if severity:
        query = query.filter(Alert.severity == severity)
    if resolved is not None:
        query = query.filter(Alert.resolved == resolved)
    alerts = query.order_by(Alert.created_at.desc()).offset(skip).limit(limit).all()
    return alerts

@app.put("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
def acknowledge_alert(
    alert_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Acknowledge alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    alert.acknowledged_by = current_user.id
    db.commit()
    return {"message": "Alert acknowledged"}

@app.put("/alerts/{alert_id}/resolve", tags=["Alerts"])
def resolve_alert(
    alert_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Resolve alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.resolved = True
    alert.resolved_at = datetime.utcnow()
    db.commit()
    return {"message": "Alert resolved"}

# ==================== ROUTES MAINTENANCE ====================

@app.post("/maintenance", response_model=MaintenanceScheduleResponse, status_code=status.HTTP_201_CREATED, tags=["Maintenance"])
def schedule_maintenance(
    maintenance: MaintenanceScheduleCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Schedule maintenance"""
    db_maintenance = MaintenanceSchedule(**maintenance.dict())
    db.add(db_maintenance)
    db.commit()
    db.refresh(db_maintenance)
    log_audit(db, current_user.id, "CREATE", "Maintenance", db_maintenance.id, "success")
    return db_maintenance

@app.get("/maintenance", response_model=List[MaintenanceScheduleResponse], tags=["Maintenance"])
def list_maintenance(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List maintenance schedules"""
    query = db.query(MaintenanceSchedule)
    if status:
        query = query.filter(MaintenanceSchedule.status == status)
    maintenance = query.order_by(MaintenanceSchedule.scheduled_start).offset(skip).limit(limit).all()
    return maintenance

# ==================== ROUTES ANALYTICS & DASHBOARD ====================

@app.get("/dashboard/metrics", response_model=DashboardMetrics, tags=["Dashboard"])
def get_dashboard_metrics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get dashboard metrics"""
    total_racks = db.query(Rack).count()
    total_servers = db.query(Server).count()
    total_vms = db.query(VirtualMachine).count()
    total_containers = db.query(Container).count()
    
    power_consumption = total_servers * 0.3  # Simplified
    pue = calculate_pue(db) or 1.5
    
    active_alerts = db.query(Alert).filter(Alert.resolved == False).count()
    
    return DashboardMetrics(
        total_racks=total_racks,
        total_servers=total_servers,
        total_vms=total_vms,
        total_containers=total_containers,
        power_consumption_kw=power_consumption,
        pue=pue,
        avg_temperature_c=23.5,
        capacity_utilization_pct=67.5,
        uptime_pct=99.97,
        active_alerts=active_alerts
    )

@app.get("/analytics/capacity", tags=["Analytics"])
def get_capacity_analytics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get capacity analytics"""
    total_u_capacity = db.query(Rack).with_entities(db.func.sum(Rack.u_capacity)).scalar() or 0
    total_u_used = db.query(Rack).with_entities(db.func.sum(Rack.u_used)).scalar() or 0
    
    total_power_capacity = db.query(Rack).with_entities(db.func.sum(Rack.power_capacity_kw)).scalar() or 0
    total_power_used = db.query(Rack).with_entities(db.func.sum(Rack.power_used_kw)).scalar() or 0
    
    return {
        'rack_space': {
            'total_u': total_u_capacity,
            'used_u': total_u_used,
            'available_u': total_u_capacity - total_u_used,
            'utilization_pct': (total_u_used / total_u_capacity * 100) if total_u_capacity > 0 else 0
        },
        'power': {
            'capacity_kw': total_power_capacity,
            'used_kw': total_power_used,
            'available_kw': total_power_capacity - total_power_used,
            'utilization_pct': (total_power_used / total_power_capacity * 100) if total_power_capacity > 0 else 0
        }
    }

@app.get("/analytics/forecast", tags=["Analytics"])
def get_capacity_forecast(
    months: int = 6,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get capacity forecast"""
    current_servers = db.query(Server).count()
    growth_rate = 0.05  # 5% monthly growth
    
    forecast = []
    for month in range(months + 1):
        forecast.append({
            'month': month,
            'servers': int(current_servers * ((1 + growth_rate) ** month)),
            'racks_needed': int(current_servers * ((1 + growth_rate) ** month) / 20)
        })
    
    return forecast

@app.get("/analytics/cost", tags=["Analytics"])
def get_cost_analytics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get cost analytics"""
    total_servers = db.query(Server).count()
    total_storage_tb = db.query(StorageSystem).with_entities(
        db.func.sum(StorageSystem.capacity_tb)).scalar() or 0
    
    # Simplified cost calculations
    compute_cost = total_servers * 150  # $150/server/month
    storage_cost = total_storage_tb * 50  # $50/TB/month
    network_cost = 5000  # Fixed
    power_cost = total_servers * 0.3 * 0.12 * 730  # Power consumption
    
    total_cost = compute_cost + storage_cost + network_cost + power_cost
    
    return {
        'monthly_cost': total_cost,
        'breakdown': {
            'compute': compute_cost,
            'storage': storage_cost,
            'network': network_cost,
            'power': power_cost
        },
        'per_server': total_cost / max(total_servers, 1)
    }

# ==================== ROUTES AUDIT ====================

@app.get("/audit/logs", tags=["Audit"])
def get_audit_logs(
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get audit logs"""
    query = db.query(AuditLog)
    
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if action:
        query = query.filter(AuditLog.action == action)
    if start_date:
        query = query.filter(AuditLog.timestamp >= start_date)
    if end_date:
        query = query.filter(AuditLog.timestamp <= end_date)
    
    logs = query.order_by(AuditLog.timestamp.desc()).offset(skip).limit(limit).all()
    
    return [{
        'id': log.id,
        'user_id': log.user_id,
        'action': log.action,
        'resource_type': log.resource_type,
        'resource_id': log.resource_id,
        'ip_address': log.ip_address,
        'status': log.status,
        'timestamp': log.timestamp.isoformat()
    } for log in logs]

# ==================== SYSTEM ROUTES ====================

@app.get("/health", tags=["System"])
def health_check():
    """Health check"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }

@app.get("/stats", tags=["System"])
def get_system_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get system statistics"""
    return {
        'database': {
            'users': db.query(User).count(),
            'datacenters': db.query(Datacenter).count(),
            'racks': db.query(Rack).count(),
            'servers': db.query(Server).count(),
            'vms': db.query(VirtualMachine).count(),
            'containers': db.query(Container).count(),
            'storage_systems': db.query(StorageSystem).count(),
            'network_devices': db.query(NetworkDevice).count(),
            'incidents': db.query(Incident).count(),
            'alerts': db.query(Alert).count(),
            'maintenance_schedules': db.query(MaintenanceSchedule).count()
        },
        'timestamp': datetime.utcnow().isoformat()
    }

# ==================== WEBSOCKET FOR REAL-TIME MONITORING ====================

@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring"""
    await websocket.accept()
    try:
        while True:
            # Simulate real-time metrics
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_usage': float(np.random.uniform(40, 80)),
                'memory_usage': float(np.random.uniform(50, 85)),
                'network_in': float(np.random.uniform(20, 60)),
                'network_out': float(np.random.uniform(15, 50)),
                'power_consumption': float(np.random.uniform(50, 90)),
                'temperature': float(np.random.uniform(22, 28))
            }
            
            await websocket.send_json(metrics)
            await asyncio.sleep(5)  # Send every 5 seconds
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    # Create default admin user
    db = SessionLocal()
    admin_user = db.query(User).filter(User.username == "admin").first()
    if not admin_user:
        admin_user = User(
            username="admin",
            email="admin@datacenter.com",
            full_name="Administrator",
            role="admin",
            security_clearance="Top Secret",
            mfa_enabled=True,
            hashed_password=get_password_hash("admin123"),
            is_active=True
        )
        db.add(admin_user)
        db.commit()
        print("✅ Admin user created: username=admin, password=admin123")
    
    # Create default datacenter
    default_dc = db.query(Datacenter).filter(Datacenter.name == "DC-01").first()
    if not default_dc:
        default_dc = Datacenter(
            name="DC-01",
            location="Primary Site",
            tier="Tier 3",
            total_space_sqm=1000.0,
            power_capacity_mw=5.0,
            cooling_capacity_mw=3.0,
            pue_target=1.5,
            status="operational",
            certifications=["ISO 27001", "SOC 2"]
        )
        db.add(default_dc)
        db.commit()
        print("✅ Default datacenter created: DC-01")
    
    db.close()
    
    print("🚀 Starting Datacenter Management Platform API...")
    print("📖 Documentation: http://localhost:8000/docs")
    print("📖 ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8023, reload=True)