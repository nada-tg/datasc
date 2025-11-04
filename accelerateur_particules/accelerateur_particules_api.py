"""
Moteur d'Accélérateur de Particules - IA Quantique
Architecture complète pour créer, développer, fabriquer, tester et déployer
des accélérateurs de particules avec simulations avancées
uvicorn accelerateur_particules_api:app --host 0.0.0.0 --port 8001 --reload
"""

import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid
from fastapi import FastAPI
from scipy import constants

app = FastAPI()

# ==================== CONSTANTES PHYSIQUES ====================

PHYSICS_CONSTANTS = {
    'c': constants.c,  # Vitesse de la lumière (m/s)
    'e': constants.e,  # Charge électron (C)
    'm_e': constants.m_e,  # Masse électron (kg)
    'm_p': constants.m_p,  # Masse proton (kg)
    'm_n': constants.m_n,  # Masse neutron (kg)
    'h': constants.h,  # Constante de Planck (J·s)
    'hbar': constants.hbar,  # h/2π
    'epsilon_0': constants.epsilon_0,  # Permittivité du vide
    'mu_0': constants.mu_0,  # Perméabilité du vide
    'alpha': constants.alpha,  # Constante de structure fine
    'N_A': constants.N_A,  # Nombre d'Avogadro
    'k_B': constants.k,  # Constante de Boltzmann
    'G': constants.G,  # Constante gravitationnelle
}

# ==================== ENUMS ET CONSTANTES ====================

class AcceleratorType(Enum):
    LINEAR = "lineaire"
    CIRCULAR = "circulaire"
    SYNCHROTRON = "synchrotron"
    CYCLOTRON = "cyclotron"
    BETATRON = "betatron"
    SYNCHROCYCLOTRON = "synchrocyclotron"
    COLLIDER = "collisionneur"
    STORAGE_RING = "anneau_stockage"
    RFCA = "cavite_rf"
    WAKEFIELD = "champ_sillage"
    LASER_PLASMA = "laser_plasma"

class ParticleType(Enum):
    ELECTRON = "electron"
    POSITRON = "positron"
    PROTON = "proton"
    ANTIPROTON = "antiproton"
    NEUTRON = "neutron"
    MUON = "muon"
    PION = "pion"
    KAON = "kaon"
    DEUTERON = "deuteron"
    ALPHA = "alpha"
    HEAVY_ION = "ion_lourd"

class BeamType(Enum):
    CONTINUOUS = "continu"
    PULSED = "pulse"
    BUNCHED = "paquets"

class FocusingType(Enum):
    STRONG_FOCUSING = "focalisation_forte"
    WEAK_FOCUSING = "focalisation_faible"
    ALTERNATING_GRADIENT = "gradient_alternant"
    SOLENOID = "solenoide"
    QUADRUPOLE = "quadrupole"
    SEXTUPOLE = "sextupole"

class DetectorType(Enum):
    CALORIMETER = "calorimetre"
    TRACKER = "trajectographe"
    MUON_CHAMBER = "chambre_muons"
    TIME_PROJECTION = "projection_temporelle"
    SILICON_VERTEX = "vertex_silicium"
    CHERENKOV = "cherenkov"
    SCINTILLATOR = "scintillateur"

class ExperimentType(Enum):
    COLLISION = "collision"
    FIXED_TARGET = "cible_fixe"
    SPECTROSCOPY = "spectroscopie"
    IRRADIATION = "irradiation"
    MEDICAL = "medical"
    MATERIAL_SCIENCE = "science_materiaux"
    FUNDAMENTAL_PHYSICS = "physique_fondamentale"

# ==================== PARTICULES ====================

class Particle:
    """Classe représentant une particule"""
    
    PARTICLE_DATA = {
        ParticleType.ELECTRON: {
            'mass': 9.10938e-31,  # kg
            'charge': -1.602176e-19,  # C
            'spin': 0.5,
            'lifetime': float('inf'),  # stable
            'symbol': 'e⁻'
        },
        ParticleType.POSITRON: {
            'mass': 9.10938e-31,
            'charge': 1.602176e-19,
            'spin': 0.5,
            'lifetime': float('inf'),
            'symbol': 'e⁺'
        },
        ParticleType.PROTON: {
            'mass': 1.67262e-27,
            'charge': 1.602176e-19,
            'spin': 0.5,
            'lifetime': float('inf'),
            'symbol': 'p'
        },
        ParticleType.ANTIPROTON: {
            'mass': 1.67262e-27,
            'charge': -1.602176e-19,
            'spin': 0.5,
            'lifetime': float('inf'),
            'symbol': 'p̄'
        },
        ParticleType.NEUTRON: {
            'mass': 1.67493e-27,
            'charge': 0,
            'spin': 0.5,
            'lifetime': 881.5,  # secondes
            'symbol': 'n'
        },
        ParticleType.MUON: {
            'mass': 1.88353e-28,
            'charge': -1.602176e-19,
            'spin': 0.5,
            'lifetime': 2.2e-6,
            'symbol': 'μ⁻'
        }
    }
    
    def __init__(self, particle_type: ParticleType):
        self.type = particle_type
        data = self.PARTICLE_DATA[particle_type]
        self.mass = data['mass']
        self.charge = data['charge']
        self.spin = data['spin']
        self.lifetime = data['lifetime']
        self.symbol = data['symbol']
        
        # Propriétés cinématiques
        self.energy = 0.0  # Joules
        self.momentum = 0.0  # kg·m/s
        self.velocity = 0.0  # m/s
        self.gamma = 1.0  # Facteur de Lorentz
        self.beta = 0.0  # v/c
    
    def set_kinetic_energy(self, energy_ev: float):
        """Définit l'énergie cinétique en eV"""
        energy_joules = energy_ev * constants.e
        
        # Énergie totale = énergie cinétique + énergie de masse
        rest_energy = self.mass * constants.c**2
        total_energy = energy_joules + rest_energy
        
        # Facteur de Lorentz
        self.gamma = total_energy / rest_energy
        
        # Beta (v/c)
        self.beta = np.sqrt(1 - 1/self.gamma**2)
        
        # Vitesse
        self.velocity = self.beta * constants.c
        
        # Momentum
        self.momentum = self.gamma * self.mass * self.velocity
        
        self.energy = total_energy
    
    def get_relativistic_mass(self) -> float:
        """Retourne la masse relativiste"""
        return self.gamma * self.mass
    
    def get_rigidity(self) -> float:
        """Retourne la rigidité magnétique (Bρ) en T·m"""
        return self.momentum / abs(self.charge)

# ==================== FAISCEAU ====================

class Beam:
    """Classe représentant un faisceau de particules"""
    
    def __init__(self, particle_type: ParticleType, beam_type: BeamType):
        self.particle = Particle(particle_type)
        self.beam_type = beam_type
        
        # Paramètres du faisceau
        self.intensity = 0.0  # Particules par seconde
        self.current = 0.0  # Ampères
        self.energy = 0.0  # eV
        self.energy_spread = 0.0  # Dispersion en énergie (ΔE/E)
        
        # Émittance (qualité du faisceau)
        self.emittance_x = 0.0  # m·rad
        self.emittance_y = 0.0  # m·rad
        self.emittance_z = 0.0  # Longitudinale
        
        # Dimensions du faisceau
        self.beam_size_x = 0.0  # m
        self.beam_size_y = 0.0  # m
        self.bunch_length = 0.0  # m
        
        # Luminosité (pour collisionneurs)
        self.luminosity = 0.0  # cm⁻²·s⁻¹
        
        # Durée de vie du faisceau
        self.lifetime = 0.0  # secondes
    
    def calculate_current(self):
        """Calcule le courant du faisceau"""
        self.current = self.intensity * abs(self.particle.charge)
        return self.current
    
    def calculate_luminosity(self, n_bunches: int, freq: float, other_beam: 'Beam'):
        """Calcule la luminosité pour un collisionneur"""
        # Formule simplifiée
        N1 = self.intensity / (n_bunches * freq)  # Particules par paquet
        N2 = other_beam.intensity / (n_bunches * freq)
        
        sigma_x = np.sqrt(self.beam_size_x * other_beam.beam_size_x)
        sigma_y = np.sqrt(self.beam_size_y * other_beam.beam_size_y)
        
        self.luminosity = (n_bunches * freq * N1 * N2) / (4 * np.pi * sigma_x * sigma_y)
        return self.luminosity

# ==================== COMPOSANTS ====================

class RFCavity:
    """Cavité radiofréquence pour l'accélération"""
    
    def __init__(self, frequency: float, voltage: float, length: float):
        self.frequency = frequency  # Hz
        self.voltage = voltage  # Volts
        self.length = length  # mètres
        self.quality_factor = 0.0  # Q
        self.shunt_impedance = 0.0  # Ohms
        self.power = 0.0  # Watts
        self.temperature = 300.0  # Kelvin
        self.material = "copper"
    
    def calculate_energy_gain(self, particle: Particle) -> float:
        """Calcule le gain d'énergie pour une particule"""
        # Gain d'énergie en eV
        return abs(particle.charge) * self.voltage / constants.e
    
    def calculate_transit_time_factor(self, beta: float) -> float:
        """Facteur de temps de transit"""
        k = 2 * np.pi * self.frequency / (beta * constants.c)
        return np.sin(k * self.length / 2) / (k * self.length / 2)

class Magnet:
    """Aimant dipolaire ou multipolaire"""
    
    def __init__(self, magnet_type: str, field_strength: float, length: float):
        self.type = magnet_type  # dipole, quadrupole, sextupole
        self.field_strength = field_strength  # Tesla
        self.length = length  # mètres
        self.aperture = 0.0  # mètres
        self.gradient = 0.0  # T/m (pour quadrupoles)
        self.current = 0.0  # Ampères
        self.power = 0.0  # Watts
        self.cooling_system = "water"
        self.superconducting = False
    
    def calculate_bending_radius(self, particle: Particle) -> float:
        """Rayon de courbure pour un aimant dipolaire"""
        if self.type != "dipole":
            return 0.0
        
        rigidity = particle.get_rigidity()
        return rigidity / self.field_strength
    
    def calculate_focal_length(self, particle: Particle) -> float:
        """Longueur focale pour un quadrupole"""
        if self.type != "quadrupole":
            return 0.0
        
        rigidity = particle.get_rigidity()
        return rigidity / (self.gradient * self.length)

class Detector:
    """Détecteur de particules"""
    
    def __init__(self, detector_type: DetectorType, name: str):
        self.type = detector_type
        self.name = name
        self.resolution = 0.0  # Résolution en énergie
        self.efficiency = 0.0  # Efficacité de détection
        self.acceptance = 0.0  # Acceptance géométrique
        self.granularity = 0.0  # Granularité spatiale
        self.time_resolution = 0.0  # Résolution temporelle (s)
        self.energy_threshold = 0.0  # Seuil en énergie (eV)
        self.max_rate = 0.0  # Taux maximal (Hz)

# ==================== ACCÉLÉRATEUR ====================

class Accelerator:
    """Classe principale d'un accélérateur de particules"""
    
    def __init__(self, accelerator_id: str, name: str, acc_type: AcceleratorType):
        self.id = accelerator_id
        self.name = name
        self.type = acc_type
        self.created_at = datetime.now().isoformat()
        
        # Géométrie
        self.length = 0.0  # mètres (linéaire) ou circonférence (circulaire)
        self.radius = 0.0  # mètres (pour circulaire)
        self.circumference = 0.0  # mètres
        
        # Énergie
        self.energy_min = 0.0  # eV
        self.energy_max = 0.0  # eV
        self.energy_final = 0.0  # eV
        
        # Composants
        self.rf_cavities = []
        self.magnets = []
        self.detectors = []
        
        # Faisceau
        self.beams = []
        self.particle_types = []
        
        # Performance
        self.luminosity = 0.0  # cm⁻²·s⁻¹
        self.collision_rate = 0.0  # Hz
        self.beam_current = 0.0  # A
        
        # Système de vide
        self.vacuum_pressure = 0.0  # Pascal
        self.vacuum_quality = 0.0  # 0-1
        
        # Refroidissement
        self.cooling_power = 0.0  # Watts
        self.temperature = 300.0  # Kelvin
        
        # Radioprotection
        self.shielding_thickness = 0.0  # mètres
        self.radiation_level = 0.0  # Sievert/heure
        
        # Coûts
        self.construction_cost = 0.0  # USD
        self.operational_cost = 0.0  # USD/année
        self.energy_consumption = 0.0  # MWh/année
        
        # État
        self.status = 'offline'
        self.operational_hours = 0.0
        self.total_collisions = 0
        self.experiments_run = 0
        
        # Sécurité
        self.safety_systems = []
        self.interlocks = []
        self.emergency_stops = []
    
    def add_rf_cavity(self, frequency: float, voltage: float, length: float):
        """Ajoute une cavité RF"""
        cavity = RFCavity(frequency, voltage, length)
        self.rf_cavities.append(cavity)
        return cavity
    
    def add_magnet(self, magnet_type: str, field: float, length: float):
        """Ajoute un aimant"""
        magnet = Magnet(magnet_type, field, length)
        self.magnets.append(magnet)
        return magnet
    
    def add_beam(self, particle_type: ParticleType, beam_type: BeamType):
        """Ajoute un faisceau"""
        beam = Beam(particle_type, beam_type)
        self.beams.append(beam)
        return beam
    
    def add_detector(self, detector_type: DetectorType, name: str):
        """Ajoute un détecteur"""
        detector = Detector(detector_type, name)
        self.detectors.append(detector)
        return detector
    
    def calculate_total_voltage(self) -> float:
        """Calcule la tension totale d'accélération"""
        return sum(cavity.voltage for cavity in self.rf_cavities)
    
    def calculate_energy_gain_per_turn(self, particle: Particle) -> float:
        """Gain d'énergie par tour (circulaire)"""
        total_gain = 0.0
        for cavity in self.rf_cavities:
            total_gain += cavity.calculate_energy_gain(particle)
        return total_gain
    
    def calculate_number_of_turns(self, initial_energy: float, final_energy: float, particle: Particle) -> int:
        """Nombre de tours nécessaires"""
        gain_per_turn = self.calculate_energy_gain_per_turn(particle)
        if gain_per_turn == 0:
            return 0
        return int((final_energy - initial_energy) / gain_per_turn)
    
    def calculate_revolution_frequency(self, particle: Particle) -> float:
        """Fréquence de révolution"""
        if self.circumference == 0:
            return 0.0
        return particle.velocity / self.circumference
    
    def calculate_synchrotron_radiation_power(self, particle: Particle) -> float:
        """Puissance rayonnée par radiation synchrotron (électrons)"""
        if particle.type != ParticleType.ELECTRON and particle.type != ParticleType.POSITRON:
            return 0.0
        
        if self.radius == 0:
            return 0.0
        
        # Formule classique
        r_e = 2.8179e-15  # Rayon classique de l'électron (m)
        c = constants.c
        
        power = (2 * r_e * c * particle.energy**4) / (3 * (particle.mass * c**2)**4 * self.radius**2)
        return power
    
    def simulate_acceleration(self, particle_type: ParticleType, initial_energy: float, n_steps: int = 100):
        """Simule le processus d'accélération"""
        particle = Particle(particle_type)
        particle.set_kinetic_energy(initial_energy)
        
        trajectory = {
            'time': [],
            'position': [],
            'energy': [],
            'velocity': [],
            'gamma': []
        }
        
        if self.type in [AcceleratorType.LINEAR]:
            # Accélération linéaire
            total_length = sum(cavity.length for cavity in self.rf_cavities)
            
            for i in range(n_steps):
                t = i / n_steps
                position = t * total_length
                
                # Gain d'énergie proportionnel
                energy_gain = self.calculate_total_voltage() * t * abs(particle.charge) / constants.e
                current_energy = initial_energy + energy_gain
                
                particle.set_kinetic_energy(current_energy)
                
                trajectory['time'].append(t * total_length / particle.velocity)
                trajectory['position'].append(position)
                trajectory['energy'].append(current_energy)
                trajectory['velocity'].append(particle.velocity)
                trajectory['gamma'].append(particle.gamma)
        
        elif self.type in [AcceleratorType.CIRCULAR, AcceleratorType.SYNCHROTRON]:
            # Accélération circulaire
            gain_per_turn = self.calculate_energy_gain_per_turn(particle)
            n_turns = self.calculate_number_of_turns(initial_energy, self.energy_final, particle)
            
            current_energy = initial_energy
            
            for turn in range(min(n_turns, n_steps)):
                particle.set_kinetic_energy(current_energy)
                
                rev_freq = self.calculate_revolution_frequency(particle)
                time = turn / rev_freq if rev_freq > 0 else 0
                
                trajectory['time'].append(time)
                trajectory['position'].append(turn * self.circumference)
                trajectory['energy'].append(current_energy)
                trajectory['velocity'].append(particle.velocity)
                trajectory['gamma'].append(particle.gamma)
                
                current_energy += gain_per_turn
        
        return trajectory
    
    def calculate_beam_optics(self, beam: Beam):
        """Calcule les paramètres optiques du faisceau"""
        # Fonction beta (Twiss parameter)
        beta_x = 10.0  # mètres (valeur typique)
        beta_y = 10.0
        
        # Taille du faisceau
        beam.beam_size_x = np.sqrt(beam.emittance_x * beta_x)
        beam.beam_size_y = np.sqrt(beam.emittance_y * beta_y)
        
        return {
            'beta_x': beta_x,
            'beta_y': beta_y,
            'beam_size_x': beam.beam_size_x,
            'beam_size_y': beam.beam_size_y
        }
    
    def estimate_construction_cost(self):
        """Estime le coût de construction"""
        base_cost = 0
        
        # Coût basé sur le type
        type_costs = {
            AcceleratorType.LINEAR: 1e6,  # $1M per meter
            AcceleratorType.CIRCULAR: 5e6,  # $5M per meter
            AcceleratorType.SYNCHROTRON: 10e6,
            AcceleratorType.COLLIDER: 50e6,
        }
        
        base_cost = type_costs.get(self.type, 1e6) * self.length
        
        # Coût des cavités RF
        rf_cost = len(self.rf_cavities) * 500000
        
        # Coût des aimants
        magnet_cost = len(self.magnets) * 100000
        
        # Coût des détecteurs
        detector_cost = len(self.detectors) * 1000000
        
        self.construction_cost = base_cost + rf_cost + magnet_cost + detector_cost
        return self.construction_cost
    
    def get_comprehensive_status(self) -> Dict:
        """Retourne l'état complet"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type.value,
            'status': self.status,
            'geometry': {
                'length': self.length,
                'radius': self.radius,
                'circumference': self.circumference
            },
            'energy': {
                'min': self.energy_min,
                'max': self.energy_max,
                'final': self.energy_final
            },
            'components': {
                'rf_cavities': len(self.rf_cavities),
                'magnets': len(self.magnets),
                'detectors': len(self.detectors)
            },
            'beams': len(self.beams),
            'performance': {
                'luminosity': float(self.luminosity),
                'collision_rate': float(self.collision_rate),
                'beam_current': float(self.beam_current)
            },
            'costs': {
                'construction': float(self.construction_cost),
                'operational': float(self.operational_cost),
                'energy_consumption': float(self.energy_consumption)
            },
            'operational_hours': float(self.operational_hours),
            'total_collisions': self.total_collisions,
            'experiments_run': self.experiments_run
        }

# ==================== EXPÉRIENCES ====================

class Experiment:
    """Classe représentant une expérience"""
    
    def __init__(self, exp_id: str, name: str, exp_type: ExperimentType):
        self.id = exp_id
        self.name = name
        self.type = exp_type
        self.created_at = datetime.now().isoformat()
        
        # Configuration
        self.accelerator_id = None
        self.beam_energy = 0.0  # eV
        self.target_material = ""
        self.collision_energy = 0.0  # eV (centre de masse)
        
        # Objectifs
        self.objectives = []
        self.physics_goals = []
        
        # Données collectées
        self.events_recorded = 0
        self.data_volume = 0.0  # GB
        self.run_time = 0.0  # heures
        
        # Résultats
        self.results = {}
        self.discoveries = []
        self.publications = []
        
        # État
        self.status = 'planned'  # planned, running, completed, analysing
        self.progress = 0.0  # 0-1

# ==================== SIMULATIONS ====================

class ParticleCollisionSimulation:
    """Simule des collisions de particules"""
    
    def __init__(self):
        self.events = []
        self.cross_sections = {}
    
    def simulate_collision(self, beam1: Beam, beam2: Beam, n_events: int = 1000):
        """Simule des collisions entre deux faisceaux"""
        results = {
            'total_events': n_events,
            'elastic_scattering': 0,
            'inelastic_scattering': 0,
            'particle_production': [],
            'energy_distribution': [],
            'angle_distribution': []
        }
        
        # Énergie dans le centre de masse
        E_cm = np.sqrt(2 * beam1.energy * beam2.energy * (1 + beam1.particle.beta * beam2.particle.beta))
        
        for i in range(n_events):
            # Type d'événement (probabilités simplifiées)
            event_type = np.random.choice(
                ['elastic', 'inelastic', 'production'],
                p=[0.3, 0.5, 0.2]
            )
            
            if event_type == 'elastic':
                results['elastic_scattering'] += 1
                # Diffusion élastique
                theta = np.random.exponential(0.1)  # Angle de diffusion
                results['angle_distribution'].append(theta)
            
            elif event_type == 'inelastic':
                results['inelastic_scattering'] += 1
                # Perte d'énergie
                energy_loss = np.random.uniform(0.1, 0.5) * E_cm
                results['energy_distribution'].append(E_cm - energy_loss)
            
            else:
                # Production de particules
                n_particles = np.random.poisson(5)  # Nombre de particules produites
                results['particle_production'].append(n_particles)
        
        return results
    
    def calculate_cross_section(self, process: str, energy: float) -> float:
        """Calcule la section efficace pour un processus"""
        # Formules simplifiées
        if process == 'elastic':
            # Section efficace élastique (dépend de l'énergie)
            sigma = 1e-27 * (1 + 1/energy)  # m²
        elif process == 'inelastic':
            sigma = 5e-28 * np.log(energy)
        else:
            sigma = 1e-30
        
        return sigma

# ==================== GESTIONNAIRE ====================

class AcceleratorManager:
    """Gestionnaire principal"""
    
    def __init__(self):
        self.accelerators = {}
        self.experiments = {}
        self.simulations = []
    
    def create_accelerator(self, name: str, acc_type: str, config: Dict) -> str:
        """Crée un nouvel accélérateur"""
        acc_id = f"acc_{len(self.accelerators) + 1}"
        accelerator = Accelerator(acc_id, name, AcceleratorType(acc_type))
        
        # Configuration géométrique
        if 'length' in config:
            accelerator.length = config['length']
        if 'radius' in config:
            accelerator.radius = config['radius']
            accelerator.circumference = 2 * np.pi * accelerator.radius
        
        # Configuration énergétique
        accelerator.energy_min = config.get('energy_min', 1e6)
        accelerator.energy_max = config.get('energy_max', 1e9)
        accelerator.energy_final = config.get('energy_final', accelerator.energy_max)
        
        # Ajouter composants
        if 'rf_cavities' in config:
            for cavity_config in config['rf_cavities']:
                accelerator.add_rf_cavity(
                    cavity_config['frequency'],
                    cavity_config['voltage'],
                    cavity_config['length']
                )
        
        if 'magnets' in config:
            for magnet_config in config['magnets']:
                accelerator.add_magnet(
                    magnet_config['type'],
                    magnet_config['field'],
                    magnet_config['length']
                )
        
        # Estimer les coûts
        accelerator.estimate_construction_cost()
        
        self.accelerators[acc_id] = accelerator
        return acc_id
    
    def get_accelerator(self, acc_id: str) -> Optional[Accelerator]:
        """Récupère un accélérateur"""
        return self.accelerators.get(acc_id)
    
    def create_experiment(self, name: str, exp_type: str, accelerator_id: str, config: Dict) -> str:
        """Crée une nouvelle expérience"""
        exp_id = f"exp_{len(self.experiments) + 1}"
        experiment = Experiment(exp_id, name, ExperimentType(exp_type))
        
        experiment.accelerator_id = accelerator_id
        experiment.beam_energy = config.get('beam_energy', 1e9)
        experiment.target_material = config.get('target_material', 'proton')
        experiment.objectives = config.get('objectives', [])
        
        self.experiments[exp_id] = experiment
        return exp_id
    
    def run_simulation(self, accelerator_id: str, particle_type: str, initial_energy: float, n_steps: int = 100):
        """Lance une simulation"""
        accelerator = self.get_accelerator(accelerator_id)
        if not accelerator:
            return {'error': 'Accelerator not found'}
        
        trajectory = accelerator.simulate_acceleration(
            ParticleType(particle_type),
            initial_energy,
            n_steps
        )
        
        simulation_result = {
            'simulation_id': f"sim_{len(self.simulations) + 1}",
            'accelerator_id': accelerator_id,
            'particle_type': particle_type,
            'initial_energy': initial_energy,
            'final_energy': trajectory['energy'][-1] if trajectory['energy'] else 0,
            'trajectory': trajectory,
            'timestamp': datetime.now().isoformat()
        }
        
        self.simulations.append(simulation_result)
        return simulation_result

# ==================== POINT D'ENTRÉE ====================

if __name__ == "__main__":
    manager = AcceleratorManager()
    
    # Exemple: Créer un accélérateur linéaire
    acc_id = manager.create_accelerator(
        "LINAC-Alpha",
        "lineaire",
        {
            'length': 1000,  # 1 km
            'energy_min': 1e6,  # 1 MeV
            'energy_max': 10e9,  # 10 GeV
            'energy_final': 10e9,
            'rf_cavities': [
                {'frequency': 3e9, 'voltage': 10e6, 'length': 1.0}
                for _ in range(100)
            ],
            'magnets': [
                {'type': 'quadrupole', 'field': 1.0, 'length': 0.5}
                for _ in range(200)
            ]
        }
    )
    
    # Créer un faisceau
    accelerator = manager.get_accelerator(acc_id)
    beam = accelerator.add_beam(ParticleType.ELECTRON, BeamType.PULSED)
    beam.intensity = 1e12  # particules/s
    beam.energy = 10e9  # 10 GeV
    beam.emittance_x = 1e-9
    beam.emittance_y = 1e-9
    
    # Simuler l'accélération
    simulation = manager.run_simulation(acc_id, "electron", 1e6, 100)
    
    # Créer une expérience
    exp_id = manager.create_experiment(
        "High Energy Physics Experiment",
        "physique_fondamentale",
        acc_id,
        {
            'beam_energy': 10e9,
            'target_material': 'proton',
            'objectives': ['Recherche Higgs', 'Physique au-delà du modèle standard']
        }
    )
    
    print(json.dumps({
        'accelerator': accelerator.get_comprehensive_status(),
        'simulation': simulation,
        'experiment_id': exp_id
    }, indent=2, default=str))