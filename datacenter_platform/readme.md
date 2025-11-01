# 🏢 Datacenter Management Platform - Documentation Complète

Plateforme complète de gestion de datacenter avec monitoring temps réel, IA, automation, sécurité avancée et analytics.

## 📋 Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [API Documentation](#api-documentation)
- [Exemples](#exemples)

---

## 🌟 Vue d'ensemble

### Plateforme Enterprise de Gestion Datacenter

Cette plateforme offre une solution complète pour gérer l'infrastructure datacenter moderne avec :

- **Gestion Infrastructure** : Racks, serveurs, storage, réseau
- **Virtualisation** : VMs, containers, orchestration Kubernetes
- **Monitoring Temps Réel** : Métriques, alertes, dashboards
- **IA & Automation** : Prédictions, optimisation, auto-remediation
- **Sécurité** : Multi-niveaux, audit trail, compliance
- **Capacity Planning** : Forecasting, recommandations
- **Cost Management** : Tracking, analyse, optimisation

---

## 🚀 Fonctionnalités

### 🏗️ Infrastructure Management

#### Datacenters
- Gestion multi-datacenter
- Classification Tier (I-IV)
- Certifications (ISO 27001, SOC 2, PCI DSS)
- Métriques PUE/DCiE

#### Racks
- Tracking U-space (1-52U)
- Gestion power capacity
- Monitoring température/humidité
- Zones de sécurité
- Cooling type par rack

#### Servers
- Inventaire complet (CPU, RAM, Storage)
- Health monitoring
- Warranty tracking
- Power consumption tracking
- Multiple types: Compute, Storage, GPU, Database

### 💾 Storage Management

- **Types supportés** : SAN, NAS, Object, Block, File
- **RAID levels** : 0, 1, 5, 6, 10
- **Métriques** : IOPS, throughput, latency
- **Features** : Replication, encryption, snapshots

### 🌐 Network Management

- **Devices** : Core switches, routers, firewalls, load balancers
- **Speeds** : 1G, 10G, 25G, 40G, 100G
- **Monitoring** : Traffic, latency, packet loss
- **Security** : Threat detection, DDoS protection

### ☁️ Virtualization

#### Virtual Machines
- Multi-hypervisor support
- Resource allocation (vCPU, vRAM, vDisk)
- Live migration
- Snapshots & clones
- Auto-scaling

#### Containers
- Docker support
- Kubernetes orchestration
- Scaling automatique
- Health checks
- Rolling updates

### 📊 Monitoring & Alerting

#### Real-Time Metrics
- CPU, Memory, Disk, Network
- Power consumption
- Temperature
- Custom metrics

#### Alerting
- Multi-level severity
- Auto-acknowledgment
- Integration webhooks
- Escalation policies

### 🤖 AI Operations

#### Predictive Analytics
- Capacity forecasting (3-12 months)
- Failure prediction (servers, storage)
- Anomaly detection
- Pattern recognition

#### Auto-Optimization
- Workload balancing
- VM placement optimization
- Power efficiency
- Cooling optimization

#### Auto-Remediation
- Service restart
- VM migration
- Disk cleanup
- Auto-scaling

### 🔐 Security

#### Multi-Level Security
- 4 security zones (Public → Top Secret)
- Role-based access control (RBAC)
- 2FA/MFA support
- Biometric authentication

#### Compliance
- Audit trail complet
- Compliance reporting (ISO, SOC, PCI, HIPAA)
- Security scanning
- Vulnerability assessment

### 💰 Cost Management

- Real-time cost tracking
- Cost by resource type
- Budget alerts
- Optimization recommendations
- TCO analysis

### 📈 Capacity Planning

- Growth forecasting
- Resource recommendations
- Procurement planning
- Risk analysis

---

## 🏗️ Architecture

```
datacenter-platform/
├── frontend/
│   └── datacenter_platform.py       # Streamlit UI
├── backend/
│   └── datacenter_api.py            # FastAPI backend
├── database/
│   ├── schema.sql                   # Database schema
│   └── migrations/                  # Alembic migrations
├── docker/
│   ├── Dockerfile.frontend
│   ├── Dockerfile.backend
│   └── docker-compose.yml
├── tests/
│   ├── test_api.py
│   └── test_integration.py
├── requirements.txt
├── README.md
└── .env
```

### Stack Technique

**Frontend:**
- Streamlit 1.28+
- Plotly 5.17+ (visualizations)
- Pandas, NumPy (data processing)

**Backend:**
- FastAPI 0.104+
- SQLAlchemy 2.0+ (ORM)
- PostgreSQL 14+ (database)
- Redis 7+ (caching, queues)
- Celery 5+ (async tasks)

**Monitoring:**
- Prometheus (metrics)
- Grafana (dashboards)
- ELK Stack (logs)

**Security:**
- JWT authentication
- OAuth2 flows
- SSL/TLS encryption

---

## 📦 Installation

### Prérequis

```bash
# System requirements
- Python 3.9+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose (optional)
```

### Installation Standard

```bash
# 1. Clone repository
git clone https://github.com/your-org/datacenter-platform.git
cd datacenter-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```txt
# Web Frameworks
streamlit>=1.28.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.9
alembic>=1.12.0

# Caching & Queue
redis>=5.0.0
celery>=5.3.0

# Data Science
pandas>=2.1.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0

# Visualization
plotly>=5.17.0

# Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6

# Monitoring
prometheus-client>=0.19.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.4.0
pydantic-settings>=2.1.0
```

### Installation Docker

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## ⚙️ Configuration

### 1. Variables d'Environnement

Créer fichier `.env` :

```env
# Database
DATABASE_URL=postgresql://dcuser:password@localhost:5432/datacenter_db
# For SQLite (dev): DATABASE_URL=sqlite:///./datacenter.db

# Security
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Email (for alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=alerts@yourdomain.com
SMTP_PASSWORD=your-email-password

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
```

### 2. Database Setup

#### PostgreSQL

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
# or
brew install postgresql  # macOS

# Create database
sudo -u postgres psql
CREATE DATABASE datacenter_db;
CREATE USER dcuser WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE datacenter_db TO dcuser;
\q

# Run migrations
alembic upgrade head
```

### 3. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: datacenter_db
      POSTGRES_USER: dcuser
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dcuser"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://dcuser:secure_password@postgres:5432/datacenter_db
      REDIS_URL: redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./backend:/app

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      API_URL: http://api:8000
    depends_on:
      - api
    volumes:
      - ./frontend:/app

  celery:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    command: celery -A tasks worker --loglevel=info
    environment:
      DATABASE_URL: postgresql://dcuser:secure_password@postgres:5432/datacenter_db
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - postgres
      - redis

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

---

## 🚀 Utilisation

### Démarrage

```bash
# Terminal 1: Start API
cd backend
uvicorn datacenter_api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd frontend
streamlit run datacenter_platform.py

# Terminal 3: Start Celery Worker (optional)
celery -A tasks worker --loglevel=info
```

### Accès Applications

- **Frontend** : http://localhost:8501
- **API Documentation** : http://localhost:8000/docs
- **API ReDoc** : http://localhost:8000/redoc
- **Prometheus** : http://localhost:9090
- **Grafana** : http://localhost:3000

### Credentials par Défaut

```
Username: admin
Password: admin123
```

⚠️ **Changer le mot de passe en production !**

---

## 📡 API Documentation

### Authentication

```bash
# Register new user
curl -X POST "http://localhost:8000/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "secure_password",
    "full_name": "John Doe",
    "role": "operator"
  }'

# Login
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"

# Response:
# {"access_token": "eyJhbGc...", "token_type": "bearer"}
```

### Endpoints Summary

| Category | Endpoints | Description |
|----------|-----------|-------------|
| Auth | `/register`, `/token`, `/users/me` | Authentication |
| Datacenters | `/datacenters` | CRUD datacenters |
| Racks | `/racks` | Rack management |
| Servers | `/servers` | Server management |
| Metrics | `/metrics/servers` | Server metrics |
| Storage | `/storage` | Storage systems |
| Network | `/network/devices`, `/network/traffic` | Network management |
| VMs | `/vms` | Virtual machines |
| Containers | `/containers` | Container orchestration |
| Incidents | `/incidents` | Incident management |
| Alerts | `/alerts` | Alert management |
| Maintenance | `/maintenance` | Maintenance scheduling |
| Analytics | `/analytics/*` | Analytics & forecasting |
| Dashboard | `/dashboard/metrics` | Dashboard data |
| Audit | `/audit/logs` | Audit trail |

### Examples Détaillés

#### 1. Créer un Datacenter

```bash
curl -X POST "http://localhost:8000/datacenters" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DC-PRIMARY",
    "location": "New York, USA",
    "tier": "Tier 3",
    "total_space_sqm": 5000.0,
    "power_capacity_mw": 10.0,
    "cooling_capacity_mw": 6.0,
    "pue_target": 1.4
  }'
```

#### 2. Ajouter un Rack

```bash
curl -X POST "http://localhost:8000/racks" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "RACK-A-001",
    "datacenter_id": 1,
    "location": "Row A, Position 1",
    "row": "A",
    "position": 1,
    "u_capacity": 42,
    "power_capacity_kw": 10.0,
    "cooling_type": "Liquid Cooling",
    "security_zone": "Confidential"
  }'
```

#### 3. Déployer un Serveur

```bash
curl -X POST "http://localhost:8000/servers" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "SRV-WEB-001",
    "rack_id": 1,
    "server_type": "Compute",
    "manufacturer": "Dell",
    "model": "PowerEdge R750",
    "serial_number": "SRV123456",
    "u_position": 1,
    "u_size": 2,
    "cpu_model": "Intel Xeon Gold 6338",
    "cpu_cores": 64,
    "ram_gb": 512,
    "storage_tb": 4.0,
    "power_supply_w": 750,
    "os": "Ubuntu 22.04",
    "ip_address": "10.0.1.10",
    "management_ip": "10.0.0.10"
  }'
```

#### 4. Créer une VM

```bash
curl -X POST "http://localhost:8000/vms" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "VM-WEB-APP-01",
    "host_server_id": 1,
    "os": "Ubuntu 22.04",
    "cpu_cores": 4,
    "ram_gb": 16,
    "disk_gb": 200,
    "network_type": "Bridged",
    "vlan": "100"
  }'
```

#### 5. Obtenir Dashboard Metrics

```bash
curl -X GET "http://localhost:8000/dashboard/metrics" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### 6. Capacity Forecast

```bash
curl -X GET "http://localhost:8000/analytics/forecast?months=12" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 🎯 Use Cases & Exemples

### Use Case 1: Setup Initial Infrastructure

```python
import requests

BASE_URL = "http://localhost:8000"
TOKEN = "your_jwt_token"
headers = {"Authorization": f"Bearer {TOKEN}"}

# 1. Create datacenter
dc_data = {
    "name": "DC-01",
    "location": "Primary Site",
    "tier": "Tier 3",
    "total_space_sqm": 2000.0,
    "power_capacity_mw": 5.0,
    "cooling_capacity_mw": 3.0,
    "pue_target": 1.4
}

response = requests.post(f"{BASE_URL}/datacenters", json=dc_data, headers=headers)
dc = response.json()
print(f"Datacenter created: {dc['name']}")

# 2. Add 10 racks
for i in range(1, 11):
    rack_data = {
        "name": f"RACK-A-{i:03d}",
        "datacenter_id": dc['id'],
        "location": f"Row A, Position {i}",
        "row": "A",
        "position": i,
        "u_capacity": 42,
        "power_capacity_kw": 10.0,
        "cooling_type": "Air Cooling",
        "security_zone": "Confidential"
    }
    
    response = requests.post(f"{BASE_URL}/racks", json=rack_data, headers=headers)
    rack = response.json()
    print(f"Rack created: {rack['name']}")
```

### Use Case 2: Monitor & Alert

```python
# Get real-time dashboard metrics
response = requests.get(f"{BASE_URL}/dashboard/metrics", headers=headers)
metrics = response.json()

print(f"Total Servers: {metrics['total_servers']}")
print(f"PUE: {metrics['pue']}")
print(f"Active Alerts: {metrics['active_alerts']}")

# Check for alerts
response = requests.get(f"{BASE_URL}/alerts?resolved=false", headers=headers)
alerts = response.json()

for alert in alerts:
    if alert['severity'] == 'critical':
        print(f"🔴 CRITICAL: {alert['message']}")
        
        # Auto-acknowledge
        requests.put(
            f"{BASE_URL}/alerts/{alert['id']}/acknowledge",
            headers=headers
        )
```

### Use Case 3: Capacity Planning

```python
# Get current capacity
response = requests.get(f"{BASE_URL}/analytics/capacity", headers=headers)
capacity = response.json()

rack_util = capacity['rack_space']['utilization_pct']
print(f"Rack utilization: {rack_util:.1f}%")

if rack_util > 80:
    print("⚠️ HIGH UTILIZATION - Need to add racks")
    
    # Get forecast
    response = requests.get(f"{BASE_URL}/analytics/forecast?months=6", headers=headers)
    forecast = response.json()
    
    print(f"In 6 months, you'll need: {forecast[-1]['racks_needed']} racks")
```

---

## 🧪 Tests

### Tests Unitaires

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from datacenter_api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_datacenter():
    # Login
    response = client.post("/token",
        data={"username": "admin", "password": "admin123"})
    token = response.json()["access_token"]
    
    # Create datacenter
    dc_data = {
        "name": "Test-DC",
        "location": "Test Location",
        "tier": "Tier 3",
        "total_space_sqm": 1000.0,
        "power_capacity_mw": 2.0,
        "cooling_capacity_mw": 1.5,
        "pue_target": 1.5
    }
    
    response = client.post("/datacenters",
        json=dc_data,
        headers={"Authorization": f"Bearer {token}"})
    
    assert response.status_code == 201
    assert response.json()["name"] == "Test-DC"
```

### Lancer Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=.
```

---

## 🔒 Sécurité

### Best Practices

1. **Changer credentials par défaut**
2. **Utiliser HTTPS en production**
3. **Activer 2FA pour tous les utilisateurs**
4. **Rotation régulière des secrets**
5. **Audit logs activés**
6. **Backup réguliers**
7. **Penetration testing**

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name datacenter.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/datacenter.crt;
    ssl_certificate_key /etc/ssl/private/datacenter.key;
    
    location / {
        proxy_pass http://localhost:8501;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
    }
}
```

---

## 📊 Monitoring & Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'datacenter-api'
    static_configs:
      - targets: ['api:8000']
```

### Grafana Dashboards

Import pre-built dashboards:
- Infrastructure Overview
- Server Metrics
- Network Traffic
- Power & Cooling
- Capacity Planning

---

## 🤝 Support

- **Email**: support@datacenter-platform.com
- **Documentation**: https://docs.datacenter-platform.com
- **Issues**: https://github.com/your-org/datacenter-platform/issues
- **Slack**: datacenter-platform.slack.com

---

## 📄 Licence

MIT License - See LICENSE file for details

---

## 🎯 Roadmap

### v1.1 (Q1 2025)
- [ ] Integration DCIM tools (Nlyte, Sunbird)
- [ ] Advanced AI predictions
- [ ] Mobile app (iOS/Android)
- [ ] Multi-tenancy support

### v2.0 (Q2 2025)
- [ ] Edge computing support
- [ ] Blockchain for audit trail
- [ ] AR/VR datacenter visualization
- [ ] Advanced automation workflows

---

**🏢 Datacenter Management Platform - Enterprise Edition © 2024**
1. Frontend Complet (Streamlit)

🏠 Dashboard avec KPIs temps réel
🖥️ Infrastructure (Datacenters, Racks, Servers)
💾 Storage Management
🌐 Network Management & Traffic
☁️ Virtualization (VMs + Containers)
📊 Monitoring temps réel avec graphiques
🤖 AI Operations (prédictions, optimisation, auto-remediation)
🔐 Security (multi-niveaux, audit)
🚨 Incidents & Alerts
💰 Cost Management
📈 Capacity Planning avec forecasting
⚙️ Settings & Configuration

2. API Backend Complète (FastAPI)

15+ catégories d'endpoints
20+ modèles database (PostgreSQL)
CRUD complet pour toutes les ressources
WebSocket pour monitoring temps réel
JWT Authentication avec RBAC
Audit trail complet
Metrics & Analytics
Swagger/ReDoc documentation auto

3. Documentation Complète

Guide installation
Configuration (PostgreSQL, Redis, Docker)
Exemples d'utilisation détaillés
Use cases réels
Tests
Sécurité & Best practices
Monitoring avec Prometheus/Grafana

🚀 Démarrage Rapide
bash# Installation
pip install streamlit fastapi uvicorn sqlalchemy psycopg2-binary pandas plotly numpy redis celery

# Lancer API
uvicorn datacenter_api:app --reload

# Lancer Frontend (autre terminal)
streamlit run datacenter_platform.py
📍 URLs

Frontend: http://localhost:8501
API Docs: http://localhost:8000/docs
Credentials: admin / admin123

🎯 Fonctionnalités Clés
✅ Gestion complète infrastructure (DC, Racks, Servers, Storage, Network)
✅ Virtualisation (VMs + Containers)
✅ Monitoring temps réel avec alertes
✅ IA prédictive (pannes, capacité, coûts)
✅ Auto-remediation
✅ Sécurité multi-niveaux
✅ Capacity planning & forecasting
✅ Cost management
✅ Audit trail complet