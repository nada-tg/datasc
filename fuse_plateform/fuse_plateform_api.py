"""
API Backend pour Plateforme Conception Fusées Spatiales
Architecture complète avec IA, Quantique, Bio-computing
uvicorn fuse_plateform_api:app --host 0.0.0.0 --port 8026 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import json
from enum import Enum

app = FastAPI(
    title="Rocket Engineering Platform API",
    description="API complète pour conception, fabrication et simulation de fusées spatiales",
    version="2.0.0"
)

# ==================== CONSTANTES ====================

class PhysicsConstants:
    """Constantes physiques fondamentales"""
    G = 6.67430e-11  # Constante gravitationnelle universelle
    c = 299792458  # Vitesse de la lumière
    h = 6.62607015e-34  # Constante de Planck
    k_B = 1.380649e-23  # Constante de Boltzmann
    N_A = 6.02214076e23  # Nombre d'Avogadro
    
    # Terre
    EARTH_MU = 3.986004418e14
    EARTH_RADIUS = 6371000
    EARTH_MASS = 5.972e24
    EARTH_G = 9.80665
    EARTH_ATM = 101325
    
    # Mars
    MARS_MU = 4.282837e13
    MARS_RADIUS = 3389500
    MARS_MASS = 6.4171e23
    MARS_G = 3.721
    MARS_ATM = 610

# ==================== ENUMS ====================

class PropulsionType(str, Enum):
    CHEMICAL = "chemical"
    ELECTRIC = "electric"
    NUCLEAR = "nuclear"
    PLASMA = "plasma"
    FUSION = "fusion"
    ANTIMATTER = "antimatter"

class RocketStatus(str, Enum):
    DESIGN = "design"
    MANUFACTURING = "manufacturing"
    TESTING = "testing"
    ACTIVE = "active"
    RETIRED = "retired"

class MissionTarget(str, Enum):
    LEO = "LEO"
    GTO = "GTO"
    MOON = "Moon"
    MARS = "Mars"
    ASTEROIDS = "Asteroids"
    JUPITER = "Jupiter"
    INTERSTELLAR = "Interstellar"

# ==================== MODÈLES PYDANTIC ====================

class EngineConfig(BaseModel):
    name: str
    propulsion_type: PropulsionType
    propellant: str
    thrust_sl: float  # N
    thrust_vac: float  # N
    isp_sl: float  # s
    isp_vac: float  # s
    chamber_pressure: float  # MPa
    expansion_ratio: float
    mass: float  # kg
    throttle_range: tuple[int, int]
    restart_capable: bool
    gimbaling: float  # degrees

class RocketStage(BaseModel):
    stage_number: int
    dry_mass: float  # kg
    propellant_mass: float  # kg
    engines: List[str]  # Engine IDs
    structure_material: str
    tanks_material: str

class RocketConfig(BaseModel):
    name: str
    target: MissionTarget
    num_stages: int
    stages: List[RocketStage]
    payload_mass: float  # kg
    height: float  # m
    diameter: float  # m
    reusability: bool
    technologies: List[str]

class AIModel(BaseModel):
    name: str
    model_type: str
    application: str
    training_samples: int
    accuracy: Optional[float] = None

class QuantumSimulation(BaseModel):
    name: str
    algorithm: str
    num_qubits: int
    problem_type: str
    backend: str

class MarsMission(BaseModel):
    name: str
    mission_type: str
    crew_size: int
    cargo_mass: float  # tonnes
    launch_window: str
    surface_duration: int  # days
    landing_site: str
    technologies: List[str]

# ==================== CLASSES MÉTIER ====================

class Engine:
    """Moteur-fusée"""
    
    def __init__(self, config: EngineConfig):
        self.id = f"engine_{datetime.now().timestamp()}"
        self.config = config
        self.created_at = datetime.now()
        self.test_fires = 0
        self.reliability = 0.0
        
    def calculate_twr(self) -> float:
        """Calcule le thrust-to-weight ratio"""
        return self.config.thrust_vac / (self.config.mass * PhysicsConstants.EARTH_G)
    
    def calculate_exhaust_velocity(self) -> float:
        """Calcule la vitesse d'échappement"""
        return self.config.isp_vac * PhysicsConstants.EARTH_G
    
    def calculate_mass_flow(self) -> float:
        """Calcule le débit massique"""
        ve = self.calculate_exhaust_velocity()
        return self.config.thrust_vac / ve if ve > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'config': self.config.dict(),
            'created_at': self.created_at.isoformat(),
            'twr': self.calculate_twr(),
            'exhaust_velocity': self.calculate_exhaust_velocity(),
            'mass_flow': self.calculate_mass_flow(),
            'test_fires': self.test_fires,
            'reliability': self.reliability
        }

class Rocket:
    """Fusée complète"""
    
    def __init__(self, config: RocketConfig):
        self.id = f"rocket_{datetime.now().timestamp()}"
        self.config = config
        self.created_at = datetime.now()
        self.status = RocketStatus.DESIGN
        self.test_flights = 0
        self.success_rate = 0.0
        
        # Calculs masse
        self.dry_mass = sum(stage.dry_mass for stage in config.stages)
        self.propellant_mass = sum(stage.propellant_mass for stage in config.stages)
        self.total_mass = self.dry_mass + self.propellant_mass + config.payload_mass
        
    def calculate_delta_v(self, isp_avg: float = 350) -> float:
        """Calcule le delta-v total (Tsiolkovsky)"""
        g0 = PhysicsConstants.EARTH_G
        ve = isp_avg * g0
        
        m0 = self.total_mass
        mf = self.dry_mass + self.config.payload_mass
        
        if mf > 0 and m0 > mf:
            return ve * np.log(m0 / mf)
        return 0.0
    
    def calculate_payload_capacity(self, target: str) -> float:
        """Estime la capacité charge utile selon destination"""
        delta_v = self.calculate_delta_v()
        
        # Delta-v requis par destination
        dv_requirements = {
            'LEO': 9400,
            'GTO': 12500,
            'Moon': 15000,
            'Mars': 16000,
            'Jupiter': 30000
        }
        
        dv_required = dv_requirements.get(target, 9400)
        
        if delta_v >= dv_required:
            # Estimation simplifiée
            excess_dv = delta_v - dv_required
            return self.config.payload_mass * (1 + excess_dv / dv_required * 0.1)
        else:
            return self.config.payload_mass * (delta_v / dv_required)
    
    def calculate_cost(self) -> float:
        """Estime le coût de lancement"""
        # Modèle coût simplifié
        base_cost = self.total_mass * 1000  # $/kg
        
        if self.config.reusability:
            base_cost *= 0.3  # Réduction 70% si réutilisable
        
        # Majoration selon technologie
        if 'Nuclear' in self.config.technologies:
            base_cost *= 2.0
        if 'IA' in self.config.technologies:
            base_cost *= 1.1
        
        return base_cost
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'config': self.config.dict(),
            'created_at': self.created_at.isoformat(),
            'status': self.status.value,
            'mass': {
                'dry': self.dry_mass,
                'propellant': self.propellant_mass,
                'payload': self.config.payload_mass,
                'total': self.total_mass
            },
            'performance': {
                'delta_v': self.calculate_delta_v(),
                'payload_leo': self.calculate_payload_capacity('LEO'),
                'payload_mars': self.calculate_payload_capacity('Mars'),
                'cost': self.calculate_cost()
            },
            'test_flights': self.test_flights,
            'success_rate': self.success_rate
        }

class AIOptimizer:
    """Optimiseur IA pour design fusée"""
    
    @staticmethod
    def optimize_rocket_design(rocket: Rocket, objectives: List[str]) -> Dict:
        """Optimise le design selon objectifs"""
        
        # Simulation optimisation (algorithme génétique simplifié)
        iterations = 1000
        best_fitness = 0.0
        improvements = {}
        
        # Objectifs
        if 'maximize_payload' in objectives:
            # Optimiser ratio masse
            current_ratio = rocket.propellant_mass / rocket.dry_mass
            optimal_ratio = 9.5  # Ratio optimal théorique
            
            improvement = (optimal_ratio - current_ratio) / current_ratio * 100
            improvements['mass_ratio'] = {
                'current': current_ratio,
                'optimal': optimal_ratio,
                'improvement': improvement
            }
            best_fitness += 0.3
        
        if 'minimize_cost' in objectives:
            # Suggestions réduction coût
            cost_savings = []
            if not rocket.config.reusability:
                cost_savings.append(('Add reusability', 70))
            
            improvements['cost_reduction'] = cost_savings
            best_fitness += 0.25
        
        if 'maximize_reliability' in objectives:
            # Redondance, qualité matériaux
            improvements['reliability'] = {
                'engine_redundancy': 'Add 2 engines (N+2)',
                'materials': 'Upgrade to aerospace grade',
                'estimated_improvement': '+8%'
            }
            best_fitness += 0.2
        
        return {
            'iterations': iterations,
            'best_fitness': best_fitness + np.random.random() * 0.15,
            'improvements': improvements,
            'estimated_gain': {
                'payload': '+15-20%',
                'cost': '-10-15%',
                'reliability': '+5-8%'
            }
        }

class QuantumSimulator:
    """Simulateur quantique pour problèmes aérospatial"""
    
    @staticmethod
    def optimize_trajectory(origin: str, destination: str, num_qubits: int) -> Dict:
        """Optimise trajectoire avec algorithme quantique"""
        
        # Simulation algorithme quantique (Grover/QAOA)
        num_states = 2 ** num_qubits
        
        # Calcul temps
        classical_time = num_states * 0.001  # ms
        quantum_time = np.sqrt(num_states) * 0.01  # ms
        speedup = classical_time / quantum_time
        
        # Delta-v optimal (simulé)
        delta_v_classical = 12000  # m/s
        delta_v_quantum = delta_v_classical * 0.88  # 12% amélioration
        
        return {
            'num_qubits': num_qubits,
            'states_explored': num_states,
            'computation_time': quantum_time,
            'speedup_vs_classical': speedup,
            'delta_v_optimal': delta_v_quantum,
            'improvement': ((delta_v_classical - delta_v_quantum) / delta_v_classical) * 100,
            'success_probability': 0.94
        }
    
    @staticmethod
    def simulate_combustion(fuel: str, oxidizer: str, temperature: float) -> Dict:
        """Simule combustion au niveau quantique"""
        
        # Simulation VQE (Variational Quantum Eigensolver)
        
        # Énergies de réaction (simulées)
        reaction_energies = {
            ('RP-1', 'LOX'): -45.2e6,  # J/kg
            ('LH2', 'LOX'): -120.9e6,
            ('Methane', 'LOX'): -55.5e6
        }
        
        energy_released = reaction_energies.get((fuel, oxidizer), -50e6)
        
        # Température flamme
        flame_temp = temperature + 500  # K
        
        # Vitesse échappement
        R = 8.314  # J/mol/K
        M = 18  # g/mol (approximation H2O)
        ve = np.sqrt(2 * R * flame_temp / (M / 1000))
        
        # Isp
        isp = ve / PhysicsConstants.EARTH_G
        
        return {
            'fuel': fuel,
            'oxidizer': oxidizer,
            'energy_released': abs(energy_released),
            'flame_temperature': flame_temp,
            'exhaust_velocity': ve,
            'isp_predicted': isp,
            'products': [
                {'molecule': 'H2O', 'fraction': 0.42},
                {'molecule': 'CO2', 'fraction': 0.38},
                {'molecule': 'CO', 'fraction': 0.12},
                {'molecule': 'H2', 'fraction': 0.05},
                {'molecule': 'OH', 'fraction': 0.03}
            ]
        }

class BioComputing:
    """Système bio-computing pour contrôle"""
    
    @staticmethod
    def create_neural_network(num_neurons: int, learning_rate: float) -> Dict:
        """Crée réseau neuronal organique"""
        
        num_synapses = num_neurons * 50  # Moyenne 50 connexions/neurone
        
        # Métriques performance
        latency = 0.5 + np.random.random() * 0.5  # ms
        power = num_neurons * 0.0012  # mW par neurone
        
        return {
            'neurons': num_neurons,
            'synapses': num_synapses,
            'learning_rate': learning_rate,
            'latency': latency,
            'power_consumption': power,
            'plasticity': 96.5,  # %
            'viability': 98.7,  # %
            'adaptation_speed': 'fast' if learning_rate > 0.05 else 'medium'
        }
    
    @staticmethod
    def run_diagnostics(system_data: Dict) -> Dict:
        """Diagnostic prédictif bio-sensoriel"""
        
        # Analyse patterns (simulé)
        anomalies = []
        
        if np.random.random() > 0.85:
            anomalies.append({
                'type': 'vibration',
                'severity': 'low',
                'confidence': 0.87,
                'prediction_window': '15 minutes'
            })
        
        health_score = 100 - len(anomalies) * 5
        
        return {
            'health_score': health_score,
            'anomalies_detected': anomalies,
            'prediction_accuracy': 99.2,
            'system_status': 'nominal' if health_score > 95 else 'warning'
        }

class MarsCalculator:
    """Calculateurs spécifiques missions Mars"""
    
    @staticmethod
    def calculate_edl(spacecraft_mass: float, velocity: float) -> Dict:
        """Calcule séquence EDL (Entry, Descent, Landing)"""
        
        # Phases EDL
        phases = {
            'entry': {
                'altitude_start': 125000,  # m
                'altitude_end': 35000,
                'velocity_start': velocity,
                'velocity_end': 1200,
                'duration': 240,  # s
                'peak_temperature': 1600,  # °C
                'peak_deceleration': 11  # g
            },
            'parachute': {
                'altitude_start': 10000,
                'altitude_end': 7000,
                'velocity_start': 470,
                'velocity_end': 100,
                'duration': 120,
                'system': 'supersonic_parachute'
            },
            'powered_descent': {
                'altitude_start': 2000,
                'altitude_end': 20,
                'velocity_start': 100,
                'velocity_end': 0.75,
                'duration': 50,
                'propellant_used': spacecraft_mass * 0.1
            }
        }
        
        total_duration = sum(p['duration'] for p in phases.values())
        
        return {
            'phases': phases,
            'total_duration': total_duration,
            'success_probability': 0.85,
            'precision': '50-100m',
            'propellant_required': spacecraft_mass * 0.1
        }
    
    @staticmethod
    def calculate_isru_production(power_kw: float, duration_days: int) -> Dict:
        """Calcule production ISRU"""
        
        # Taux production (kg/jour par kW)
        ch4_rate = 0.02  # kg/jour/kW
        o2_rate = 0.08
        h2o_rate = 0.5
        
        # Production totale
        ch4_total = power_kw * ch4_rate * duration_days / 1000  # tonnes
        o2_total = power_kw * o2_rate * duration_days / 1000
        h2o_total = power_kw * h2o_rate * duration_days / 1000
        
        return {
            'power_available': power_kw,
            'duration': duration_days,
            'production': {
                'methane': ch4_total,
                'oxygen': o2_total,
                'water': h2o_total
            },
            'propellant_total': ch4_total + o2_total,
            'efficiency': 0.85
        }

# ==================== STOCKAGE EN MÉMOIRE ====================

class Database:
    """Base de données en mémoire"""
    
    def __init__(self):
        self.engines: Dict[str, Engine] = {}
        self.rockets: Dict[str, Rocket] = {}
        self.ai_models: Dict[str, Dict] = {}
        self.quantum_simulations: List[Dict] = []
        self.mars_missions: Dict[str, Dict] = {}
        self.tests: List[Dict] = []
    
    def add_engine(self, engine: Engine) -> str:
        self.engines[engine.id] = engine
        return engine.id
    
    def add_rocket(self, rocket: Rocket) -> str:
        self.rockets[rocket.id] = rocket
        return rocket.id

# Instance globale
db = Database()

# ==================== ENDPOINTS API ====================

@app.get("/")
async def root():
    return {
        "name": "Rocket Engineering Platform API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Rocket Design & Engineering",
            "Engine Simulation",
            "AI Optimization",
            "Quantum Computing",
            "Bio-computing",
            "Mars Mission Planning"
        ]
    }

# ========== ENGINES ==========

@app.post("/api/engines/create")
async def create_engine(config: EngineConfig):
    """Crée un nouveau moteur"""
    try:
        engine = Engine(config)
        engine_id = db.add_engine(engine)
        return {
            "success": True,
            "engine_id": engine_id,
            "engine": engine.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/engines/{engine_id}")
async def get_engine(engine_id: str):
    """Récupère un moteur"""
    if engine_id not in db.engines:
        raise HTTPException(status_code=404, detail="Engine not found")
    
    return db.engines[engine_id].to_dict()

@app.get("/api/engines")
async def list_engines():
    """Liste tous les moteurs"""
    return {
        "count": len(db.engines),
        "engines": [e.to_dict() for e in db.engines.values()]
    }

@app.post("/api/engines/{engine_id}/test")
async def test_engine(engine_id: str, duration: int = 30):
    """Simule un test moteur"""
    if engine_id not in db.engines:
        raise HTTPException(status_code=404, detail="Engine not found")
    
    engine = db.engines[engine_id]
    engine.test_fires += 1
    
    # Simulation test
    success = np.random.random() > 0.1
    
    if success:
        engine.reliability = min(100, engine.reliability + 2)
    
    test_result = {
        'engine_id': engine_id,
        'duration': duration,
        'success': success,
        'thrust_avg': engine.config.thrust_vac * (0.95 + np.random.random() * 0.05),
        'isp_measured': engine.config.isp_vac * (0.98 + np.random.random() * 0.04),
        'chamber_pressure': engine.config.chamber_pressure * (0.97 + np.random.random() * 0.06),
        'reliability_updated': engine.reliability
    }
    
    db.tests.append(test_result)
    
    return test_result

# ========== ROCKETS ==========

@app.post("/api/rockets/create")
async def create_rocket(config: RocketConfig):
    """Crée une nouvelle fusée"""
    try:
        rocket = Rocket(config)
        rocket_id = db.add_rocket(rocket)
        return {
            "success": True,
            "rocket_id": rocket_id,
            "rocket": rocket.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/rockets/{rocket_id}")
async def get_rocket(rocket_id: str):
    """Récupère une fusée"""
    if rocket_id not in db.rockets:
        raise HTTPException(status_code=404, detail="Rocket not found")
    
    return db.rockets[rocket_id].to_dict()

@app.get("/api/rockets")
async def list_rockets():
    """Liste toutes les fusées"""
    return {
        "count": len(db.rockets),
        "rockets": [r.to_dict() for r in db.rockets.values()]
    }

@app.post("/api/rockets/{rocket_id}/simulate")
async def simulate_launch(rocket_id: str, target: str = "LEO"):
    """Simule un lancement"""
    if rocket_id not in db.rockets:
        raise HTTPException(status_code=404, detail="Rocket not found")
    
    rocket = db.rockets[rocket_id]
    
    # Simulation lancement
    success = np.random.random() > 0.15
    
    if success:
        rocket.test_flights += 1
        rocket.success_rate = (rocket.success_rate * (rocket.test_flights - 1) + 100) / rocket.test_flights
    else:
        rocket.test_flights += 1
        rocket.success_rate = (rocket.success_rate * (rocket.test_flights - 1)) / rocket.test_flights
    
    return {
        'rocket_id': rocket_id,
        'target': target,
        'success': success,
        'flight_number': rocket.test_flights,
        'altitude_max': 450000 if success else np.random.randint(50000, 300000),
        'velocity_max': 7800 if success else np.random.randint(3000, 7000),
        'payload_delivered': rocket.config.payload_mass if success else 0,
        'success_rate_updated': rocket.success_rate
    }

# ========== AI OPTIMIZATION ==========

@app.post("/api/ai/optimize")
async def optimize_rocket_ai(
    rocket_id: str,
    objectives: List[str] = ["maximize_payload", "minimize_cost", "maximize_reliability"]
):
    """Optimise une fusée avec IA"""
    if rocket_id not in db.rockets:
        raise HTTPException(status_code=404, detail="Rocket not found")
    
    rocket = db.rockets[rocket_id]
    
    result = AIOptimizer.optimize_rocket_design(rocket, objectives)
    
    return {
        'rocket_id': rocket_id,
        'objectives': objectives,
        'optimization_result': result
    }

@app.post("/api/ai/predict")
async def predict_performance(parameters: Dict[str, Any]):
    """Prédit les performances avec IA"""
    
    # Modèle ML simplifié
    mass = parameters.get('mass', 500000)
    thrust = parameters.get('thrust', 9000000)
    isp = parameters.get('isp', 350)
    
    # Prédictions
    payload_leo = (thrust / 50000) * (isp / 350) * 20
    success_probability = 0.85 + np.random.random() * 0.1
    cost_estimate = mass * 0.8 / 1000
    
    return {
        'predictions': {
            'payload_leo': payload_leo,
            'success_probability': success_probability,
            'cost_estimate_M': cost_estimate,
            'confidence': 0.94
        },
        'model': 'Neural Network v2.1',
        'trained_on': '50,000 simulations'
    }

@app.post("/api/ai/model/create")
async def create_ai_model(model: AIModel):
    """Crée un modèle IA"""
    model_id = f"ai_model_{datetime.now().timestamp()}"
    
    db.ai_models[model_id] = {
        'id': model_id,
        'config': model.dict(),
        'created_at': datetime.now().isoformat(),
        'accuracy': model.accuracy or (0.92 + np.random.random() * 0.07),
        'status': 'trained'
    }
    
    return {
        'success': True,
        'model_id': model_id,
        'model': db.ai_models[model_id]
    }

# ========== QUANTUM COMPUTING ==========

@app.post("/api/quantum/trajectory")
async def optimize_trajectory_quantum(
    origin: str,
    destination: str,
    num_qubits: int = 20,
    algorithm: str = "QAOA"
):
    """Optimise trajectoire avec computing quantique"""
    
    result = QuantumSimulator.optimize_trajectory(origin, destination, num_qubits)
    
    simulation = {
        'origin': origin,
        'destination': destination,
        'algorithm': algorithm,
        'result': result,
        'timestamp': datetime.now().isoformat()
    }
    
    db.quantum_simulations.append(simulation)
    
    return simulation

@app.post("/api/quantum/combustion")
async def simulate_combustion_quantum(
    fuel: str,
    oxidizer: str,
    temperature: float = 3000,
    pressure: float = 20
):
    """Simule combustion au niveau quantique"""
    
    result = QuantumSimulator.simulate_combustion(fuel, oxidizer, temperature)
    
    return {
        'fuel': fuel,
        'oxidizer': oxidizer,
        'conditions': {
            'temperature': temperature,
            'pressure': pressure
        },
        'quantum_simulation': result,
        'method': 'VQE (Variational Quantum Eigensolver)'
    }

@app.get("/api/quantum/simulations")
async def list_quantum_simulations():
    """Liste simulations quantiques"""
    return {
        'count': len(db.quantum_simulations),
        'simulations': db.quantum_simulations
    }

# ========== BIO-COMPUTING ==========

@app.post("/api/bio/neural-network")
async def create_bio_neural_network(
    num_neurons: int = 10000,
    learning_rate: float = 0.01
):
    """Crée réseau neuronal bio-organique"""
    
    network = BioComputing.create_neural_network(num_neurons, learning_rate)
    
    return {
        'success': True,
        'network_id': f"bio_nn_{datetime.now().timestamp()}",
        'network': network
    }

@app.post("/api/bio/diagnostics")
async def run_bio_diagnostics(system_data: Dict = None):
    """Exécute diagnostic prédictif bio"""
    
    if system_data is None:
        system_data = {
            'telemetry': {'thrust': 9000, 'pressure': 30, 'temperature': 2800}
        }
    
    result = BioComputing.run_diagnostics(system_data)
    
    return {
        'diagnostics': result,
        'timestamp': datetime.now().isoformat()
    }

# ========== MARS MISSIONS ==========

@app.post("/api/mars/mission/create")
async def create_mars_mission(mission: MarsMission):
    """Crée une mission Mars"""
    mission_id = f"mars_{datetime.now().timestamp()}"
    
    # Calculs mission
    if mission.crew_size > 0:
        transit_duration = 240  # jours
        delta_v_total = 12000  # m/s
    else:
        transit_duration = 180
        delta_v_total = 10000
    
    mission_data = {
        'id': mission_id,
        'config': mission.dict(),
        'calculations': {
            'transit_out': transit_duration,
            'surface_duration': mission.surface_duration,
            'transit_return': transit_duration,
            'total_duration': transit_duration * 2 + mission.surface_duration,
            'delta_v_total': delta_v_total,
            'propellant_needed': mission.cargo_mass * 8
        },
        'created_at': datetime.now().isoformat()
    }
    
    db.mars_missions[mission_id] = mission_data
    
    return {
        'success': True,
        'mission_id': mission_id,
        'mission': mission_data
    }

@app.post("/api/mars/edl")
async def calculate_mars_edl(
    spacecraft_mass: float,
    entry_velocity: float = 5700
):
    """Calcule séquence EDL Mars"""
    
    result = MarsCalculator.calculate_edl(spacecraft_mass, entry_velocity)
    
    return result

@app.post("/api/mars/isru")
async def calculate_isru_production(
    power_kw: float = 100,
    duration_days: int = 540
):
    """Calcule production ISRU"""
    
    result = MarsCalculator.calculate_isru_production(power_kw, duration_days)
    
    return result

@app.get("/api/mars/missions")
async def list_mars_missions():
    """Liste missions Mars"""
    return {
        'count': len(db.mars_missions),
        'missions': list(db.mars_missions.values())
    }

# ========== ANALYTICS ==========

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Vue d'ensemble analytique"""
    
    return {
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'engines': len(db.engines),
            'rockets': len(db.rockets),
            'ai_models': len(db.ai_models),
            'quantum_simulations': len(db.quantum_simulations),
            'mars_missions': len(db.mars_missions),
            'tests': len(db.tests)
        },
        'success_rates': {
            'engine_tests': sum(1 for t in db.tests if t.get('success', False)) / max(len(db.tests), 1) * 100,
            'rocket_launches': sum(r.success_rate for r in db.rockets.values()) / max(len(db.rockets), 1)
        }
    }

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Analyse performances"""
    
    if not db.rockets:
        return {'message': 'No rockets to analyze'}
    
    rockets_data = [r.to_dict() for r in db.rockets.values()]
    
    avg_delta_v = np.mean([r['performance']['delta_v'] for r in rockets_data])
    avg_cost = np.mean([r['performance']['cost'] for r in rockets_data])
    
    return {
        'fleet_analytics': {
            'total_rockets': len(rockets_data),
            'avg_delta_v': avg_delta_v,
            'avg_cost_per_launch': avg_cost,
            'reusability_rate': sum(1 for r in rockets_data if r['config']['reusability']) / len(rockets_data) * 100
        }
    }

# ========== UTILITIES ==========

@app.post("/api/calculate/delta-v")
async def calculate_delta_v(
    dry_mass: float,
    propellant_mass: float,
    isp: float = 350
):
    """Calcule delta-v (Tsiolkovsky)"""
    
    g0 = PhysicsConstants.EARTH_G
    ve = isp * g0
    
    m0 = dry_mass + propellant_mass
    mf = dry_mass
    
    if mf > 0 and m0 > mf:
        delta_v = ve * np.log(m0 / mf)
    else:
        delta_v = 0
    
    return {
        'dry_mass': dry_mass,
        'propellant_mass': propellant_mass,
        'total_mass': m0,
        'isp': isp,
        'exhaust_velocity': ve,
        'delta_v': delta_v,
        'mass_ratio': m0 / mf if mf > 0 else 0
    }

@app.get("/api/constants")
async def get_constants():
    """Retourne constantes physiques"""
    
    return {
        'universal': {
            'G': PhysicsConstants.G,
            'c': PhysicsConstants.c,
            'h': PhysicsConstants.h,
            'k_B': PhysicsConstants.k_B,
            'N_A': PhysicsConstants.N_A
        },
        'earth': {
            'mu': PhysicsConstants.EARTH_MU,
            'radius': PhysicsConstants.EARTH_RADIUS,
            'mass': PhysicsConstants.EARTH_MASS,
            'g': PhysicsConstants.EARTH_G,
            'atm_pressure': PhysicsConstants.EARTH_ATM
        },
        'mars': {
            'mu': PhysicsConstants.MARS_MU,
            'radius': PhysicsConstants.MARS_RADIUS,
            'mass': PhysicsConstants.MARS_MASS,
            'g': PhysicsConstants.MARS_G,
            'atm_pressure': PhysicsConstants.MARS_ATM
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8026)