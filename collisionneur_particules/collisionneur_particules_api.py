"""
Plateforme Complète de Physique des Particules - Collisionneurs
Architecture pour créer, développer, fabriquer, tester et analyser
tous types de collisionneurs de particules et expériences HEP
uvicorn collisionneur_particules_api:app --host 0.0.0.0 --port 8015 --reload
"""

import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid
from fastapi import FastAPI

app = FastAPI()

# ==================== CONSTANTES PHYSIQUES ====================

class PhysicsConstants:
    """Constantes fondamentales"""
    SPEED_OF_LIGHT = 299792458  # m/s
    PLANCK = 6.62607015e-34  # J·s
    ELECTRON_MASS = 9.1093837015e-31  # kg
    PROTON_MASS = 1.67262192369e-27  # kg
    NEUTRON_MASS = 1.67492749804e-27  # kg
    ELEMENTARY_CHARGE = 1.602176634e-19  # C
    FINE_STRUCTURE = 1/137.035999084  # sans dimension
    FERMI_COUPLING = 1.1663787e-5  # GeV^-2
    STRONG_COUPLING = 0.1181  # à l'échelle Z
    WEAK_MIXING_ANGLE = 0.2312  # sin²θW

# ==================== ENUMS ET TYPES ====================

class ColliderType(Enum):
    LINEAR = "lineaire"
    CIRCULAR = "circulaire"
    PLASMA = "plasma"
    MUON = "muon"
    ELECTRON_POSITRON = "electron_positron"
    PROTON_PROTON = "proton_proton"
    HEAVY_ION = "ion_lourd"
    ELECTRON_PROTON = "electron_proton"
    PHOTON_PHOTON = "photon_photon"

class ParticleType(Enum):
    ELECTRON = "electron"
    POSITRON = "positron"
    PROTON = "proton"
    ANTIPROTON = "antiproton"
    MUON = "muon"
    ANTIMUON = "antimuon"
    PHOTON = "photon"
    NEUTRINO = "neutrino"
    QUARK_UP = "quark_up"
    QUARK_DOWN = "quark_down"
    QUARK_CHARM = "quark_charm"
    QUARK_STRANGE = "quark_strange"
    QUARK_TOP = "quark_top"
    QUARK_BOTTOM = "quark_bottom"
    GLUON = "gluon"
    W_BOSON = "w_boson"
    Z_BOSON = "z_boson"
    HIGGS_BOSON = "higgs_boson"
    GRAVITON = "graviton"

class DetectorType(Enum):
    TRACKER = "trajectographe"
    CALORIMETER_EM = "calorimetrie_em"
    CALORIMETER_HAD = "calorimetrie_hadronique"
    MUON_CHAMBER = "chambre_muon"
    VERTEX_DETECTOR = "detecteur_vertex"
    TIME_OF_FLIGHT = "temps_vol"
    CHERENKOV = "cherenkov"
    TRANSITION_RADIATION = "radiation_transition"

class AcceleratorComponent(Enum):
    RF_CAVITY = "cavite_rf"
    DIPOLE_MAGNET = "dipole"
    QUADRUPOLE_MAGNET = "quadrupole"
    SEXTUPOLE_MAGNET = "sextupole"
    KLYSTRON = "klystron"
    BEAM_PIPE = "tube_faisceau"
    VACUUM_SYSTEM = "systeme_vide"
    COOLING_SYSTEM = "refroidissement"
    INJECTION_SYSTEM = "injection"
    EXTRACTION_SYSTEM = "extraction"

class ExperimentType(Enum):
    DISCOVERY = "decouverte"
    PRECISION = "precision"
    SEARCH_BSM = "recherche_bsm"
    QCD_STUDY = "etude_qcd"
    ELECTROWEAK = "electrofaible"
    HIGGS_PHYSICS = "physique_higgs"
    TOP_PHYSICS = "physique_top"
    FLAVOR_PHYSICS = "physique_saveur"
    HEAVY_ION = "ion_lourd"
    DARK_MATTER = "matiere_noire"

# ==================== PARTICULES ====================

class Particle:
    """Classe pour une particule élémentaire"""
    
    def __init__(self, particle_type: ParticleType):
        self.type = particle_type
        self.name = particle_type.value
        
        # Propriétés physiques
        self.mass = 0.0  # GeV/c²
        self.charge = 0.0  # en unités de e
        self.spin = 0.0  # en unités de ℏ
        self.lifetime = 0.0  # secondes
        self.decay_modes = []
        
        # État quantique
        self.energy = 0.0  # GeV
        self.momentum = [0.0, 0.0, 0.0]  # GeV/c
        self.position = [0.0, 0.0, 0.0]  # m
        
        # Propriétés interaction
        self.color_charge = None  # pour quarks et gluons
        self.weak_isospin = 0.0
        self.hypercharge = 0.0
        
        self._initialize_properties()
    
    def _initialize_properties(self):
        """Initialise les propriétés selon le type"""
        properties = {
            ParticleType.ELECTRON: {
                'mass': 0.000511, 'charge': -1, 'spin': 0.5, 'lifetime': float('inf')
            },
            ParticleType.POSITRON: {
                'mass': 0.000511, 'charge': 1, 'spin': 0.5, 'lifetime': float('inf')
            },
            ParticleType.PROTON: {
                'mass': 0.938272, 'charge': 1, 'spin': 0.5, 'lifetime': float('inf')
            },
            ParticleType.MUON: {
                'mass': 0.105658, 'charge': -1, 'spin': 0.5, 'lifetime': 2.2e-6
            },
            ParticleType.W_BOSON: {
                'mass': 80.379, 'charge': 1, 'spin': 1, 'lifetime': 3e-25
            },
            ParticleType.Z_BOSON: {
                'mass': 91.1876, 'charge': 0, 'spin': 1, 'lifetime': 3e-25
            },
            ParticleType.HIGGS_BOSON: {
                'mass': 125.10, 'charge': 0, 'spin': 0, 'lifetime': 1.6e-22
            },
            ParticleType.TOP_QUARK: {
                'mass': 173.0, 'charge': 2/3, 'spin': 0.5, 'lifetime': 5e-25
            }
        }
        
        if self.type in properties:
            for key, value in properties[self.type].items():
                setattr(self, key, value)
    
    def boost_to_energy(self, energy: float):
        """Boost la particule à une énergie donnée"""
        self.energy = energy
        gamma = energy / self.mass if self.mass > 0 else 1
        beta = np.sqrt(1 - 1/gamma**2) if gamma > 1 else 0
        self.momentum = [0, 0, beta * energy]

class BeamParticle:
    """Faisceau de particules"""
    
    def __init__(self, particle_type: ParticleType, n_particles: int):
        self.particle_type = particle_type
        self.n_particles = n_particles
        
        # Propriétés du faisceau
        self.energy = 0.0  # GeV
        self.intensity = 0.0  # particules/seconde
        self.emittance_x = 0.0  # m·rad
        self.emittance_y = 0.0
        self.bunch_length = 0.0  # m
        self.energy_spread = 0.0  # %
        
        # Position et direction
        self.position = [0.0, 0.0, 0.0]
        self.direction = [0.0, 0.0, 1.0]
        
        # Qualité du faisceau
        self.polarization = 0.0  # %
        self.luminosity_contribution = 0.0

# ==================== DÉTECTEURS ====================

class Detector:
    """Détecteur de particules"""
    
    def __init__(self, detector_id: str, detector_type: DetectorType, name: str):
        self.id = detector_id
        self.type = detector_type
        self.name = name
        
        # Géométrie
        self.inner_radius = 0.0  # m
        self.outer_radius = 0.0  # m
        self.length = 0.0  # m
        self.coverage = 0.0  # acceptance géométrique
        
        # Performance
        self.resolution_energy = 0.0  # %
        self.resolution_position = 0.0  # mm
        self.resolution_time = 0.0  # ps
        self.efficiency = 0.95
        
        # Électronique
        self.channels = 0
        self.readout_rate = 0.0  # MHz
        self.trigger_rate = 0.0  # kHz
        
        # Données
        self.hits = []
        self.reconstructed_tracks = []
        self.energy_deposits = []
        
        # État
        self.status = "operational"
        self.dead_channels = 0
        self.calibration_date = datetime.now()
    
    def detect_particle(self, particle: Particle) -> Dict:
        """Détecte une particule"""
        detected = np.random.random() < self.efficiency
        
        if detected:
            # Simulation de la mesure avec résolution
            energy_measured = particle.energy * (1 + np.random.randn() * self.resolution_energy / 100)
            
            hit = {
                'particle_type': particle.type.value,
                'energy': energy_measured,
                'position': particle.position.copy(),
                'time': datetime.now().timestamp(),
                'detector_id': self.id
            }
            
            self.hits.append(hit)
            return hit
        
        return None

class DetectorSystem:
    """Système complet de détection"""
    
    def __init__(self, system_id: str, name: str):
        self.id = system_id
        self.name = name
        self.detectors = []
        
        # Architecture
        self.layers = []
        self.total_channels = 0
        self.total_volume = 0.0  # m³
        self.total_weight = 0.0  # tonnes
        
        # Performance globale
        self.acceptance = 0.0  # couverture en angle solide
        self.resolution_global = {}
        
        # DAQ (Data Acquisition)
        self.daq_rate = 0.0  # GB/s
        self.trigger_system = None
        self.event_rate = 0.0  # Hz
        
        # Analyse
        self.events_recorded = 0
        self.data_volume = 0.0  # PB
    
    def add_detector(self, detector: Detector):
        """Ajoute un détecteur au système"""
        self.detectors.append(detector)
        self.total_channels += detector.channels
    
    def reconstruct_event(self, hits: List[Dict]) -> Dict:
        """Reconstruit un événement à partir des hits"""
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'n_hits': len(hits),
            'reconstructed_particles': [],
            'missing_energy': 0.0,
            'vertex_position': [0.0, 0.0, 0.0]
        }
        
        # Reconstruction simplifiée
        total_energy = sum(hit['energy'] for hit in hits)
        event['total_energy'] = total_energy
        
        return event

# ==================== ACCÉLÉRATEUR ====================

class AcceleratorSection:
    """Section d'accélérateur"""
    
    def __init__(self, section_id: str, component_type: AcceleratorComponent):
        self.id = section_id
        self.type = component_type
        
        # Géométrie
        self.length = 0.0  # m
        self.aperture = 0.0  # m
        
        # Paramètres magnétiques
        self.magnetic_field = 0.0  # T
        self.gradient = 0.0  # T/m
        
        # Paramètres RF
        self.frequency = 0.0  # MHz
        self.voltage = 0.0  # MV
        self.phase = 0.0  # degrés
        
        # Performance
        self.power_consumption = 0.0  # MW
        self.efficiency = 0.0
        
        # État
        self.status = "operational"
        self.temperature = 4.2  # K (supraconducteur)
        self.uptime = 0.0  # heures
    
    def accelerate_beam(self, beam: BeamParticle, dt: float) -> float:
        """Accélère un faisceau"""
        if self.type == AcceleratorComponent.RF_CAVITY:
            energy_gain = self.voltage * np.sin(np.radians(self.phase))
            beam.energy += energy_gain
            return energy_gain
        return 0.0

class Collider:
    """Collisionneur de particules complet"""
    
    def __init__(self, collider_id: str, name: str, collider_type: ColliderType):
        self.id = collider_id
        self.name = name
        self.type = collider_type
        self.created_at = datetime.now().isoformat()
        
        # Spécifications physiques
        self.circumference = 0.0  # km (pour circulaire)
        self.tunnel_depth = 0.0  # m
        self.beam_energy = 0.0  # TeV
        self.center_mass_energy = 0.0  # TeV
        
        # Faisceaux
        self.beam_1 = None
        self.beam_2 = None
        self.beam_particle_type_1 = None
        self.beam_particle_type_2 = None
        
        # Performance
        self.luminosity = 0.0  # cm⁻²s⁻¹
        self.peak_luminosity = 0.0
        self.integrated_luminosity = 0.0  # fb⁻¹
        self.collision_rate = 0.0  # Hz
        
        # Composants
        self.accelerator_sections = []
        self.detector_systems = []
        self.injection_system = None
        
        # Paramètres faisceau
        self.bunch_spacing = 0.0  # ns
        self.bunches_per_beam = 0
        self.particles_per_bunch = 0
        
        # Infrastructure
        self.power_consumption = 0.0  # MW
        self.cooling_capacity = 0.0  # MW
        self.cryogenic_capacity = 0.0  # kW à 4.5K
        
        # Opération
        self.status = "offline"
        self.operational_hours = 0.0
        self.efficiency = 0.0
        self.uptime = 0.0
        
        # Expériences
        self.experiments = []
        self.collisions_delivered = 0
        self.data_recorded = 0.0  # PB
        
        # Coûts
        self.construction_cost = 0.0  # millions €
        self.operational_cost_per_year = 0.0  # millions €
        
        # Physique
        self.cross_sections = {}  # processus -> section efficace (pb)
        self.event_rates = {}
    
    def calculate_luminosity(self):
        """Calcule la luminosité instantanée"""
        if self.beam_1 and self.beam_2:
            # Formule simplifiée
            N1 = self.particles_per_bunch
            N2 = self.particles_per_bunch
            nb = self.bunches_per_beam
            f_rev = PhysicsConstants.SPEED_OF_LIGHT / (self.circumference * 1000)
            
            # Luminosité (formule simplifiée)
            self.luminosity = (N1 * N2 * nb * f_rev) / (4 * np.pi * 1e-8)
            
        return self.luminosity
    
    def calculate_event_rate(self, cross_section: float) -> float:
        """Calcule le taux d'événements"""
        # Rate = Luminosity × Cross-section
        rate = self.luminosity * cross_section * 1e-36  # conversion pb
        return rate
    
    def add_detector_system(self, detector_system: DetectorSystem):
        """Ajoute un système de détection"""
        self.detector_systems.append(detector_system)
    
    def run_collision(self, duration: float) -> Dict:
        """Simule des collisions"""
        result = {
            'duration': duration,
            'collisions': 0,
            'events_recorded': 0,
            'luminosity_delivered': 0.0,
            'data_volume': 0.0
        }
        
        # Calcul du nombre de collisions
        collision_rate = self.luminosity * 1e-24 * 40e6  # taux simplifié
        result['collisions'] = int(collision_rate * duration)
        
        # Événements enregistrés (avec efficacité trigger)
        trigger_efficiency = 0.1  # 10% des événements passent le trigger
        result['events_recorded'] = int(result['collisions'] * trigger_efficiency)
        
        # Luminosité intégrée
        result['luminosity_delivered'] = self.luminosity * duration * 1e-39  # fb⁻¹
        self.integrated_luminosity += result['luminosity_delivered']
        
        # Volume de données (environ 1 MB par événement)
        result['data_volume'] = result['events_recorded'] * 1e-6  # GB
        
        return result

# ==================== SIMULATIONS PHYSIQUES ====================

class PhysicsSimulator:
    """Simulateur de processus physiques"""
    
    def __init__(self):
        self.monte_carlo_events = []
        self.cross_sections = {}
    
    def generate_event(self, process: str, energy: float) -> Dict:
        """Génère un événement Monte Carlo"""
        event = {
            'process': process,
            'energy_cm': energy,
            'particles_final': [],
            'four_momenta': [],
            'weight': 1.0
        }
        
        # Génération simplifiée selon le processus
        if process == "higgs_production":
            # H → γγ
            event['particles_final'] = [ParticleType.PHOTON, ParticleType.PHOTON]
            event['branching_ratio'] = 0.00227
            
        elif process == "top_pair":
            # tt̄ production
            event['particles_final'] = [ParticleType.TOP_QUARK, ParticleType.TOP_QUARK]
            
        elif process == "diboson":
            # WW, ZZ production
            event['particles_final'] = [ParticleType.W_BOSON, ParticleType.W_BOSON]
        
        self.monte_carlo_events.append(event)
        return event
    
    def calculate_cross_section(self, process: str, energy: float) -> float:
        """Calcule la section efficace d'un processus"""
        # Sections efficaces approximatives (pb)
        cross_sections = {
            'higgs_production': 50.0,  # à 13 TeV
            'top_pair': 830.0,
            'diboson': 120.0,
            'drell_yan': 6000.0,
            'qcd_jets': 50000.0
        }
        
        # Ajustement en fonction de l'énergie (approximatif)
        base_cross_section = cross_sections.get(process, 1.0)
        energy_factor = (energy / 13000) ** 0.3  # scaling approximatif
        
        return base_cross_section * energy_factor
    
    def simulate_detector_response(self, event: Dict, detector: Detector) -> Dict:
        """Simule la réponse du détecteur"""
        response = {
            'event_id': str(uuid.uuid4()),
            'reconstructed': True,
            'visible_particles': [],
            'missing_energy': 0.0,
            'jets': [],
            'leptons': [],
            'photons': []
        }
        
        # Simulation de la reconstruction
        for particle_type in event['particles_final']:
            if np.random.random() < detector.efficiency:
                response['visible_particles'].append(particle_type.value)
        
        return response

# ==================== ANALYSES ====================

class PhysicsAnalysis:
    """Analyse de données de physique des particules"""
    
    def __init__(self, analysis_id: str, name: str):
        self.id = analysis_id
        self.name = name
        
        # Sélection
        self.selection_cuts = []
        self.efficiency = 0.0
        self.background_rejection = 0.0
        
        # Résultats
        self.n_signal = 0
        self.n_background = 0
        self.significance = 0.0
        
        # Distributions
        self.histograms = {}
    
    def apply_cuts(self, events: List[Dict]) -> List[Dict]:
        """Applique les coupures de sélection"""
        selected = []
        
        for event in events:
            passes = True
            for cut in self.selection_cuts:
                if not self._evaluate_cut(event, cut):
                    passes = False
                    break
            
            if passes:
                selected.append(event)
        
        return selected
    
    def _evaluate_cut(self, event: Dict, cut: Dict) -> bool:
        """Évalue une coupure"""
        # Implémentation simplifiée
        return True
    
    def calculate_significance(self) -> float:
        """Calcule la significancestatistique"""
        if self.n_background > 0:
            self.significance = self.n_signal / np.sqrt(self.n_background)
        else:
            self.significance = 0.0
        
        return self.significance

# ==================== GESTIONNAIRE PRINCIPAL ====================

class ParticlePhysicsManager:
    """Gestionnaire de la plateforme de physique des particules"""
    
    def __init__(self):
        self.colliders = {}
        self.experiments = {}
        self.simulations = []
        self.analyses = {}
        self.datasets = {}
        
    def create_collider(self, name: str, collider_type: str, config: Dict) -> str:
        """Crée un nouveau collisionneur"""
        collider_id = f"collider_{len(self.colliders) + 1}"
        collider = Collider(collider_id, name, ColliderType(collider_type))
        
        # Configuration
        collider.circumference = config.get('circumference', 27.0)
        collider.beam_energy = config.get('beam_energy', 7.0)
        collider.center_mass_energy = 2 * collider.beam_energy
        
        collider.particles_per_bunch = config.get('particles_per_bunch', 1.15e11)
        collider.bunches_per_beam = config.get('bunches_per_beam', 2808)
        collider.bunch_spacing = config.get('bunch_spacing', 25.0)
        
        # Calcul luminosité
        collider.calculate_luminosity()
        
        # Coûts
        collider.construction_cost = config.get('construction_cost', 5000)
        collider.operational_cost_per_year = config.get('operational_cost', 500)
        
        self.colliders[collider_id] = collider
        
        return collider_id
    
    def get_collider(self, collider_id: str) -> Optional[Collider]:
        """Récupère un collisionneur"""
        return self.colliders.get(collider_id)

# ==================== POINT D'ENTRÉE ====================

if __name__ == "__main__":
    manager = ParticlePhysicsManager()
    
    # Exemple: Créer un collisionneur type LHC
    collider_id = manager.create_collider(
        "Large Hadron Collider",
        "circulaire",
        {
            'circumference': 27.0,  # km
            'beam_energy': 7000.0,  # GeV
            'particles_per_bunch': 1.15e11,
            'bunches_per_beam': 2808,
            'bunch_spacing': 25.0,  # ns
            'construction_cost': 5000,  # millions €
            'operational_cost': 500
        }
    )
    
    collider = manager.get_collider(collider_id)
    
    print(json.dumps({
        'collider': {
            'name': collider.name,
            'energy': f"{collider.center_mass_energy} TeV",
            'luminosity': f"{collider.luminosity:.2e} cm⁻²s⁻¹"
        }
    }, indent=2))