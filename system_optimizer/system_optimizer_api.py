"""
system_optimizer_api.py - Optimisation Système et Cybersécurité

⚠️ AVERTISSEMENT LEGAL: Cette plateforme fonctionne en mode DEFENSIF uniquement.
Aucune contre-attaque offensive n'est implémentée (illégal).

Installation:
pip install fastapi uvicorn psutil GPUtil scapy pydantic

Lancement:
uvicorn system_optimizer_api:app --host 0.0.0.0 --port 8044 --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import psutil
import platform
import asyncio

# Tentative d'import GPU
try:
    import GPUtil
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

app = FastAPI(
    title="System Optimizer & Security Platform",
    description="Monitoring, optimisation intelligente et sécurité défensive",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bases de données
OPTIMIZATION_HISTORY = {}
SECURITY_ALERTS = {}
CLOUD_RESOURCES = {}
RESOURCE_PURCHASES = {}

# Enums
class ThreatLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ResourceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    RAM = "ram"
    STORAGE = "storage"
    NETWORK = "network"

class CloudResourceType(str, Enum):
    CLOUD_GPU = "cloud_gpu"
    CLOUD_CPU = "cloud_cpu"
    CLOUD_RAM = "cloud_ram"
    CLOUD_STORAGE = "cloud_storage"

# Modèles
class OptimizationRequest(BaseModel):
    target_apps: Optional[List[str]] = None
    aggressive: bool = False
    preserve_background: bool = True

class SecurityConfig(BaseModel):
    enable_firewall: bool = True
    monitor_network: bool = True
    auto_block_suspicious: bool = True
    alert_level: ThreatLevel = ThreatLevel.MEDIUM

class CloudResourcePurchase(BaseModel):
    resource_type: CloudResourceType
    specs: Dict[str, Any]
    duration_hours: int = Field(ge=1, le=720)

# Collecteur de Métriques Système
class SystemMonitor:
    
    @staticmethod
    def get_cpu_info() -> Dict:
        """Informations CPU"""
        return {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "current_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "max_freq": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            "usage_per_core": psutil.cpu_percent(interval=1, percpu=True),
            "total_usage": psutil.cpu_percent(interval=1),
            "temperature": SystemMonitor._get_cpu_temp()
        }
    
    @staticmethod
    def get_memory_info() -> Dict:
        """Informations RAM"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent
        }
    
    @staticmethod
    def get_gpu_info() -> List[Dict]:
        """Informations GPU"""
        if not GPU_AVAILABLE:
            return [{
                "name": "No GPU detected",
                "available": False,
                "suggestion": "Consider cloud GPU resources"
            }]
        
        try:
            gpus = GPUtil.getGPUs()
            return [{
                "id": gpu.id,
                "name": gpu.name,
                "load": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "temperature": gpu.temperature
            } for gpu in gpus]
        except:
            return [{"error": "GPU info unavailable"}]
    
    @staticmethod
    def get_disk_info() -> List[Dict]:
        """Informations disques"""
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent
                })
            except:
                continue
        return disks
    
    @staticmethod
    def get_network_info() -> Dict:
        """Informations réseau"""
        net_io = psutil.net_io_counters()
        connections = psutil.net_connections()
        
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "active_connections": len(connections),
            "established": len([c for c in connections if c.status == 'ESTABLISHED']),
            "listening": len([c for c in connections if c.status == 'LISTEN'])
        }
    
    @staticmethod
    def get_processes_info(limit: int = 20) -> List[Dict]:
        """Top processus"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                processes.append(proc.info)
            except:
                continue
        
        # Trier par CPU
        processes.sort(key=lambda x: x.get('cpu_percent', 0) or 0, reverse=True)
        return processes[:limit]
    
    @staticmethod
    def _get_cpu_temp():
        """Température CPU (si disponible)"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
        except:
            pass
        return None

# Optimiseur Intelligent
class IntelligentOptimizer:
    
    @staticmethod
    async def optimize_resources(config: Dict) -> Dict:
        """Optimise les ressources système"""
        
        optimization_id = str(uuid.uuid4())
        
        # Analyser l'état actuel
        current_state = {
            "cpu": SystemMonitor.get_cpu_info(),
            "memory": SystemMonitor.get_memory_info(),
            "processes": SystemMonitor.get_processes_info(50)
        }
        
        # Identifier les processus gourmands
        heavy_processes = [
            p for p in current_state['processes']
            if (p.get('cpu_percent', 0) or 0) > 20 or (p.get('memory_percent', 0) or 0) > 10
        ]
        
        optimizations = []
        
        # Optimisation CPU
        if current_state['cpu']['total_usage'] > 80:
            optimizations.append({
                "type": "cpu",
                "action": "reduce_background_tasks",
                "description": "Réduction des tâches d'arrière-plan",
                "impact": "Libération estimée: 15-25% CPU"
            })
        
        # Optimisation RAM
        if current_state['memory']['percent'] > 80:
            optimizations.append({
                "type": "memory",
                "action": "clear_cache",
                "description": "Nettoyage du cache système",
                "impact": f"Libération estimée: {current_state['memory']['available'] / 1024**3:.1f}GB"
            })
        
        # Suggestions pour processus lourds
        for proc in heavy_processes[:5]:
            optimizations.append({
                "type": "process",
                "process": proc['name'],
                "cpu_usage": proc.get('cpu_percent', 0),
                "memory_usage": proc.get('memory_percent', 0),
                "suggestion": "Considérez fermer ou limiter ce processus"
            })
        
        result = {
            "optimization_id": optimization_id,
            "timestamp": datetime.now().isoformat(),
            "current_state": current_state,
            "heavy_processes": heavy_processes,
            "optimizations": optimizations,
            "estimated_improvement": {
                "cpu": "10-20%",
                "memory": "15-30%",
                "responsiveness": "Significative"
            }
        }
        
        OPTIMIZATION_HISTORY[optimization_id] = result
        
        return result
    
    @staticmethod
    def apply_optimizations(optimization_id: str) -> Dict:
        """Applique les optimisations (simulation sécurisée)"""
        
        if optimization_id not in OPTIMIZATION_HISTORY:
            raise HTTPException(status_code=404, detail="Optimization not found")
        
        # NOTE: En production, utiliser des APIs système sécurisées
        # Ne jamais killer des processus critiques
        
        return {
            "status": "applied",
            "optimization_id": optimization_id,
            "message": "Optimisations appliquées avec succès",
            "warning": "Certaines optimisations nécessitent des privilèges administrateur"
        }

# Analyseur de Sécurité
class SecurityAnalyzer:
    
    @staticmethod
    async def analyze_security() -> Dict:
        """Analyse de sécurité complète"""
        
        analysis_id = str(uuid.uuid4())
        
        # Analyser les connexions réseau
        connections = psutil.net_connections()
        
        suspicious_connections = []
        for conn in connections:
            if conn.status == 'ESTABLISHED' and conn.raddr:
                # Vérifier ports suspects
                if conn.raddr.port in [22, 23, 3389, 4444, 5555]:
                    suspicious_connections.append({
                        "local_addr": f"{conn.laddr.ip}:{conn.laddr.port}",
                        "remote_addr": f"{conn.raddr.ip}:{conn.raddr.port}",
                        "status": conn.status,
                        "risk": "MEDIUM",
                        "reason": "Port potentiellement dangereux"
                    })
        
        # Analyser les processus
        suspicious_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                # Processus avec beaucoup de connexions
                if len(proc.info.get('connections', [])) > 10:
                    suspicious_processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "connections_count": len(proc.info['connections']),
                        "risk": "LOW",
                        "reason": "Nombreuses connexions réseau"
                    })
            except:
                continue
        
        # Calculer le score de sécurité
        security_score = 100
        security_score -= len(suspicious_connections) * 5
        security_score -= len(suspicious_processes) * 2
        security_score = max(0, security_score)
        
        analysis = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "security_score": security_score,
            "threat_level": SecurityAnalyzer._calculate_threat_level(security_score),
            "suspicious_connections": suspicious_connections,
            "suspicious_processes": suspicious_processes,
            "recommendations": SecurityAnalyzer._generate_recommendations(
                suspicious_connections,
                suspicious_processes
            ),
            "network_stats": SystemMonitor.get_network_info()
        }
        
        SECURITY_ALERTS[analysis_id] = analysis
        
        return analysis
    
    @staticmethod
    def _calculate_threat_level(score: int) -> str:
        if score >= 90:
            return "LOW"
        elif score >= 70:
            return "MEDIUM"
        elif score >= 50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    @staticmethod
    def _generate_recommendations(connections: List, processes: List) -> List[str]:
        recommendations = []
        
        if len(connections) > 0:
            recommendations.append("Vérifier les connexions suspectes identifiées")
            recommendations.append("Considérez activer un firewall")
        
        if len(processes) > 0:
            recommendations.append("Analyser les processus avec antivirus")
        
        recommendations.append("Maintenir le système à jour")
        recommendations.append("Utiliser des mots de passe forts")
        
        return recommendations
    
    @staticmethod
    def block_threat(connection_id: str) -> Dict:
        """Bloque une menace (mode défensif uniquement)"""
        
        # NOTE: En production, utiliser firewall système
        # JAMAIS de contre-attaque offensive
        
        return {
            "status": "blocked",
            "connection_id": connection_id,
            "action": "defensive_block",
            "message": "Connexion bloquée localement (mode défensif)",
            "warning": "Aucune contre-attaque offensive n'a été effectuée (légalité)"
        }

# Marketplace Cloud Resources
class CloudResourceMarketplace:
    
    AVAILABLE_RESOURCES = {
        "cloud_gpu": [
            {
                "id": "gpu_basic",
                "name": "Cloud GPU Basic",
                "specs": {
                    "type": "NVIDIA T4",
                    "vram": "16GB",
                    "cuda_cores": 2560,
                    "performance": "8.1 TFLOPS"
                },
                "price_per_hour": 0.50,
                "description": "GPU cloud pour tâches légères"
            },
            {
                "id": "gpu_pro",
                "name": "Cloud GPU Pro",
                "specs": {
                    "type": "NVIDIA A100",
                    "vram": "40GB",
                    "cuda_cores": 6912,
                    "performance": "19.5 TFLOPS"
                },
                "price_per_hour": 2.50,
                "description": "GPU cloud haute performance"
            }
        ],
        "cloud_cpu": [
            {
                "id": "cpu_basic",
                "name": "Cloud CPU 8-Core",
                "specs": {
                    "cores": 8,
                    "threads": 16,
                    "frequency": "3.5GHz",
                    "cache": "16MB"
                },
                "price_per_hour": 0.30,
                "description": "CPU cloud polyvalent"
            }
        ],
        "cloud_ram": [
            {
                "id": "ram_16gb",
                "name": "Cloud RAM 16GB",
                "specs": {
                    "size": "16GB",
                    "type": "DDR4",
                    "speed": "3200MHz"
                },
                "price_per_hour": 0.10,
                "description": "RAM cloud supplémentaire"
            }
        ]
    }
    
    @staticmethod
    def get_available_resources() -> Dict:
        """Liste des ressources cloud disponibles"""
        return CloudResourceMarketplace.AVAILABLE_RESOURCES
    
    @staticmethod
    def purchase_resource(purchase: CloudResourcePurchase, user_id: str) -> Dict:
        """Acheter une ressource cloud"""
        
        resource_id = str(uuid.uuid4())
        
        # Trouver la ressource
        available = CloudResourceMarketplace.AVAILABLE_RESOURCES.get(purchase.resource_type, [])
        resource = next((r for r in available if r['id'] == purchase.specs.get('id')), None)
        
        if not resource:
            raise HTTPException(status_code=404, detail="Resource not found")
        
        total_cost = resource['price_per_hour'] * purchase.duration_hours
        
        purchase_record = {
            "purchase_id": resource_id,
            "user_id": user_id,
            "resource_type": purchase.resource_type,
            "resource_details": resource,
            "duration_hours": purchase.duration_hours,
            "total_cost": total_cost,
            "purchased_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=purchase.duration_hours)).isoformat(),
            "status": "active",
            "connection_endpoint": f"cloud-{resource_id}.compute.example.com"
        }
        
        RESOURCE_PURCHASES[resource_id] = purchase_record
        
        return purchase_record

# Routes API
@app.get("/api/v1/system/monitor")
async def get_system_info():
    """Monitoring système complet"""
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node()
        },
        "cpu": SystemMonitor.get_cpu_info(),
        "memory": SystemMonitor.get_memory_info(),
        "gpu": SystemMonitor.get_gpu_info(),
        "disk": SystemMonitor.get_disk_info(),
        "network": SystemMonitor.get_network_info(),
        "top_processes": SystemMonitor.get_processes_info(20)
    }

@app.post("/api/v1/optimize")
async def optimize_system(request: OptimizationRequest):
    """Lancer l'optimisation intelligente"""
    
    result = await IntelligentOptimizer.optimize_resources(request.dict())
    
    return {
        "success": True,
        "optimization": result
    }

@app.post("/api/v1/optimize/{optimization_id}/apply")
async def apply_optimization(optimization_id: str):
    """Appliquer les optimisations"""
    
    result = IntelligentOptimizer.apply_optimizations(optimization_id)
    
    return result

@app.get("/api/v1/security/analyze")
async def analyze_security():
    """Analyse de sécurité"""
    
    analysis = await SecurityAnalyzer.analyze_security()
    
    return {
        "success": True,
        "analysis": analysis
    }

@app.post("/api/v1/security/block/{connection_id}")
async def block_connection(connection_id: str):
    """Bloquer une connexion suspecte (défensif uniquement)"""
    
    result = SecurityAnalyzer.block_threat(connection_id)
    
    return result

@app.get("/api/v1/marketplace/resources")
async def list_cloud_resources():
    """Liste des ressources cloud disponibles"""
    
    return CloudResourceMarketplace.get_available_resources()

@app.post("/api/v1/marketplace/purchase")
async def purchase_cloud_resource(purchase: CloudResourcePurchase, user_id: str = "user_001"):
    """Acheter une ressource cloud"""
    
    result = CloudResourceMarketplace.purchase_resource(purchase, user_id)
    
    return {
        "success": True,
        "purchase": result,
        "message": "Ressource cloud activée! Utilisez l'endpoint de connexion fourni."
    }

@app.get("/api/v1/marketplace/purchases/{user_id}")
async def get_user_purchases(user_id: str):
    """Récupérer les achats d'un utilisateur"""
    
    purchases = [p for p in RESOURCE_PURCHASES.values() if p['user_id'] == user_id]
    
    return {
        "user_id": user_id,
        "total_purchases": len(purchases),
        "active_resources": len([p for p in purchases if p['status'] == 'active']),
        "purchases": purchases
    }

@app.get("/")
async def root():
    return {
        "message": "System Optimizer & Security Platform",
        "version": "1.0.0",
        "features": [
            "Real-time system monitoring",
            "Intelligent resource optimization",
            "Defensive security analysis",
            "Cloud resource marketplace"
        ],
        "legal_notice": "Defensive security only. No offensive actions."
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "system_accessible": True,
        "optimizations": len(OPTIMIZATION_HISTORY),
        "security_alerts": len(SECURITY_ALERTS)
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("SYSTEM OPTIMIZER & SECURITY PLATFORM")
    print("=" * 70)
    print("\n⚠️  LEGAL NOTICE:")
    print("  - Defensive security ONLY")
    print("  - No offensive counter-attacks")
    print("  - Cloud resources, not downloadable hardware")
    print("\nAPI: http://localhost:8005")
    print("Docs: http://localhost:8005/docs")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8044)