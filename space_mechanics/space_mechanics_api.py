"""
Plateforme Complète de Mécanique Spatiale
Architecture pour créer, développer, simuler et analyser
missions spatiales, satellites, trajectoires et systèmes orbitaux
uvicorn space_mechanics_api:app --host 0.0.0.0 --port 8041 --reload
"""

import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid
from fastapi import FastAPI

app = FastAPI()

# ==================== CONSTANTES PHYSIQUES & ASTRONOMIQUES ====================

class SpaceConstants:
    """Constantes fondamentales spatiales"""
    # Constantes universelles
    G = 6.67430e-11  # m³/kg/s² - Constante gravitationnelle
    c = 299792458  # m/s - Vitesse lumière
    AU = 1.496e11  # m - Unité Astronomique
    
    # Terre
    EARTH_MASS = 5.972e24  # kg
    EARTH_RADIUS = 6371000  # m
    EARTH_MU = 3.986004418e14  # m³/s² - Paramètre gravitationnel
    EARTH_J2 = 1.08263e-3  # Coefficient oblateness
    EARTH_ROTATION = 7.2921159e-5  # rad/s
    EARTH_SOI = 9.24e8  # m - Sphere of Influence
    
    # Lune
    MOON_MASS = 7.342e22  # kg
    MOON_RADIUS = 1737400  # m
    MOON_MU = 4.9028e12  # m³/s²
    MOON_DISTANCE = 384400000  # m
    
    # Soleil
    SUN_MASS = 1.989e30  # kg
    SUN_RADIUS = 696000000  # m
    SUN_MU = 1.32712440018e20  # m³/s²
    
    # Mars
    MARS_MASS = 6.4171e23  # kg
    MARS_RADIUS = 3389500  # m
    MARS_MU = 4.282837e13  # m³/s²
    MARS_DISTANCE = 2.279e11  # m (moyenne)
    
    # Conversions
    DEG_TO_RAD = np.pi / 180
    RAD_TO_DEG = 180 / np.pi
    KM_TO_M = 1000
    M_TO_KM = 0.001

# ==================== ENUMS ET TYPES ====================

class OrbitType(Enum):
    LEO = "orbite_basse"  # Low Earth Orbit
    MEO = "orbite_moyenne"  # Medium Earth Orbit
    GEO = "orbite_geostationnaire"  # Geostationary
    HEO = "orbite_haute_elliptique"  # Highly Elliptical
    POLAR = "orbite_polaire"  # Polar
    SSO = "orbite_heliosynchrone"  # Sun-Synchronous
    LUNAR = "orbite_lunaire"  # Lunar
    INTERPLANETARY = "interplanetaire"  # Interplanetary
    LAGRANGE = "point_lagrange"  # Lagrange Point
    HALO = "orbite_halo"  # Halo Orbit

class ManeuverType(Enum):
    HOHMANN = "transfert_hohmann"
    BI_ELLIPTIC = "bi_elliptique"
    INCLINATION_CHANGE = "changement_inclinaison"
    PHASING = "phasage"
    RENDEZVOUS = "rendez_vous"
    PLANE_CHANGE = "changement_plan"
    CIRCULARIZATION = "circularisation"
    ESCAPE = "evasion"
    CAPTURE = "capture"

class PropulsionType(Enum):
    CHEMICAL = "chimique"
    ELECTRIC = "electrique"
    NUCLEAR = "nucleaire"
    SOLAR_SAIL = "voile_solaire"
    ION = "ionique"
    HALL_EFFECT = "effet_hall"
    COLD_GAS = "gaz_froid"

class MissionType(Enum):
    EARTH_OBSERVATION = "observation_terre"
    COMMUNICATION = "communication"
    NAVIGATION = "navigation"
    SCIENTIFIC = "scientifique"
    EXPLORATION = "exploration"
    CREWED = "habitee"
    CARGO = "cargo"
    INTERPLANETARY = "interplanetaire"
    DEEP_SPACE = "espace_profond"

class CelestialBody(Enum):
    EARTH = "terre"
    MOON = "lune"
    MARS = "mars"
    VENUS = "venus"
    JUPITER = "jupiter"
    SATURN = "saturne"
    SUN = "soleil"

# ==================== ORBITES ====================

class Orbit:
    """Classe pour une orbite képlérienne"""
    
    def __init__(self, orbit_id: str, name: str):
        self.id = orbit_id
        self.name = name
        
        # Éléments orbitaux (éléments képlériens)
        self.semi_major_axis = 0.0  # m - demi-grand axe
        self.eccentricity = 0.0  # sans dimension (0=circulaire, <1=ellipse, 1=parabole, >1=hyperbole)
        self.inclination = 0.0  # rad - inclinaison
        self.raan = 0.0  # rad - Right Ascension Ascending Node
        self.argument_periapsis = 0.0  # rad - argument du périastre
        self.true_anomaly = 0.0  # rad - anomalie vraie
        
        # Corps central
        self.central_body = CelestialBody.EARTH
        self.mu = SpaceConstants.EARTH_MU
        
        # Paramètres dérivés
        self.period = 0.0  # s
        self.apoapsis = 0.0  # m
        self.periapsis = 0.0  # m
        self.velocity_periapsis = 0.0  # m/s
        self.velocity_apoapsis = 0.0  # m/s
        self.specific_energy = 0.0  # J/kg
        self.angular_momentum = 0.0  # m²/s
        
    def calculate_parameters(self):
        """Calcule les paramètres orbitaux dérivés"""
        a = self.semi_major_axis
        e = self.eccentricity
        
        # Période orbitale (3ème loi de Kepler)
        self.period = 2 * np.pi * np.sqrt(a**3 / self.mu)
        
        # Apoapside et périapside
        self.periapsis = a * (1 - e)
        self.apoapsis = a * (1 + e)
        
        # Vitesses
        self.velocity_periapsis = np.sqrt(self.mu * (1 + e) / (a * (1 - e)))
        self.velocity_apoapsis = np.sqrt(self.mu * (1 - e) / (a * (1 + e)))
        
        # Énergie spécifique
        self.specific_energy = -self.mu / (2 * a)
        
        # Moment angulaire spécifique
        self.angular_momentum = np.sqrt(self.mu * a * (1 - e**2))
        
        return self.period
    
    def position_velocity(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule position et vitesse à un instant t"""
        # Anomalie moyenne
        n = np.sqrt(self.mu / self.semi_major_axis**3)
        M = n * t
        
        # Résolution équation de Kepler (Newton-Raphson)
        E = M  # Anomalie excentrique (première approximation)
        for _ in range(10):
            E = E - (E - self.eccentricity * np.sin(E) - M) / (1 - self.eccentricity * np.cos(E))
        
        # Anomalie vraie
        nu = 2 * np.arctan2(
            np.sqrt(1 + self.eccentricity) * np.sin(E/2),
            np.sqrt(1 - self.eccentricity) * np.cos(E/2)
        )
        
        # Distance radiale
        r = self.semi_major_axis * (1 - self.eccentricity * np.cos(E))
        
        # Position dans le plan orbital
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # Vitesse dans le plan orbital
        h = self.angular_momentum
        vx_orb = -self.mu / h * np.sin(nu)
        vy_orb = self.mu / h * (self.eccentricity + np.cos(nu))
        
        # Transformation vers référentiel inertiel
        # (simplifié - matrice de rotation complète nécessaire pour cas général)
        position = np.array([x_orb, y_orb, 0.0])
        velocity = np.array([vx_orb, vy_orb, 0.0])
        
        return position, velocity
    
    def ground_track(self, duration: float, dt: float) -> List[Tuple[float, float]]:
        """Calcule la trace au sol"""
        track = []
        
        t = 0
        while t < duration:
            pos, _ = self.position_velocity(t)
            
            # Conversion en coordonnées géographiques (simplifié)
            x, y, z = pos
            r = np.sqrt(x**2 + y**2 + z**2)
            
            lat = np.arcsin(z / r) * SpaceConstants.RAD_TO_DEG
            lon = np.arctan2(y, x) * SpaceConstants.RAD_TO_DEG
            
            # Rotation Terre
            lon -= SpaceConstants.EARTH_ROTATION * t * SpaceConstants.RAD_TO_DEG
            lon = (lon + 180) % 360 - 180
            
            track.append((lat, lon))
            t += dt
        
        return track

# ==================== MANŒUVRES ORBITALES ====================

class OrbitalManeuver:
    """Manœuvre orbitale"""
    
    def __init__(self, maneuver_id: str, maneuver_type: ManeuverType):
        self.id = maneuver_id
        self.type = maneuver_type
        
        # Paramètres manœuvre
        self.delta_v = 0.0  # m/s
        self.burn_time = 0.0  # s
        self.propellant_mass = 0.0  # kg
        
        # Orbites
        self.initial_orbit = None
        self.final_orbit = None
        
    def hohmann_transfer(self, r1: float, r2: float, mu: float) -> Dict:
        """Calcule un transfert de Hohmann"""
        # Vitesses circulaires
        v1 = np.sqrt(mu / r1)
        v2 = np.sqrt(mu / r2)
        
        # Orbite de transfert
        a_transfer = (r1 + r2) / 2
        
        # Delta-v au périastre
        v_transfer_peri = np.sqrt(mu * (2/r1 - 1/a_transfer))
        dv1 = v_transfer_peri - v1
        
        # Delta-v à l'apoastre
        v_transfer_apo = np.sqrt(mu * (2/r2 - 1/a_transfer))
        dv2 = v2 - v_transfer_apo
        
        # Delta-v total
        total_dv = abs(dv1) + abs(dv2)
        
        # Temps de transfert
        transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)
        
        return {
            'delta_v_1': dv1,
            'delta_v_2': dv2,
            'total_delta_v': total_dv,
            'transfer_time': transfer_time,
            'transfer_orbit_a': a_transfer
        }
    
    def inclination_change(self, v: float, di: float) -> float:
        """Calcule delta-v pour changement d'inclinaison"""
        # Delta-v pour changement inclinaison pur
        dv = 2 * v * np.sin(di / 2)
        return dv
    
    def plane_change(self, v: float, di: float, dOmega: float) -> float:
        """Calcule delta-v pour changement de plan"""
        # Angle de changement de plan
        dphi = np.arccos(np.cos(di) * np.cos(dOmega))
        
        # Delta-v
        dv = 2 * v * np.sin(dphi / 2)
        return dv

# ==================== PROPULSION ====================

class PropulsionSystem:
    """Système de propulsion"""
    
    def __init__(self, system_id: str, prop_type: PropulsionType):
        self.id = system_id
        self.type = prop_type
        
        # Performances
        self.specific_impulse = 0.0  # s (Isp)
        self.thrust = 0.0  # N
        self.mass_flow_rate = 0.0  # kg/s
        self.efficiency = 0.0  # %
        
        # Propergol
        self.propellant_mass = 0.0  # kg
        self.propellant_type = ""
        
        # Paramètres spécifiques
        if prop_type == PropulsionType.CHEMICAL:
            self.specific_impulse = 450  # s (LOX/LH2)
            self.thrust = 1000000  # N
        elif prop_type == PropulsionType.ELECTRIC:
            self.specific_impulse = 3000  # s
            self.thrust = 0.5  # N
        elif prop_type == PropulsionType.ION:
            self.specific_impulse = 3500  # s
            self.thrust = 0.09  # N
        
        self.calculate_parameters()
    
    def calculate_parameters(self):
        """Calcule les paramètres de propulsion"""
        g0 = 9.80665  # m/s²
        
        if self.specific_impulse > 0:
            self.exhaust_velocity = self.specific_impulse * g0
            
        if self.thrust > 0 and self.exhaust_velocity > 0:
            self.mass_flow_rate = self.thrust / self.exhaust_velocity
    
    def delta_v_capability(self, dry_mass: float, propellant_mass: float) -> float:
        """Calcule le delta-v disponible (équation de Tsiolkovsky)"""
        g0 = 9.80665
        ve = self.specific_impulse * g0
        
        m0 = dry_mass + propellant_mass
        mf = dry_mass
        
        delta_v = ve * np.log(m0 / mf)
        
        return delta_v
    
    def burn_time(self, delta_v: float, spacecraft_mass: float) -> float:
        """Calcule le temps de combustion"""
        if self.thrust > 0:
            # Approximation pour faible delta-v
            burn_time = (spacecraft_mass * delta_v) / self.thrust
            return burn_time
        return 0.0

# ==================== SATELLITE ====================

class Satellite:
    """Satellite ou vaisseau spatial"""
    
    def __init__(self, sat_id: str, name: str):
        self.id = sat_id
        self.name = name
        self.created_at = datetime.now().isoformat()
        
        # Masses
        self.dry_mass = 1000.0  # kg
        self.propellant_mass = 500.0  # kg
        self.payload_mass = 200.0  # kg
        self.total_mass = 1700.0  # kg
        
        # Dimensions
        self.length = 2.0  # m
        self.width = 2.0  # m
        self.height = 3.0  # m
        self.solar_panel_area = 10.0  # m²
        
        # Propulsion
        self.propulsion_system = None
        
        # Puissance
        self.power_generation = 5000  # W
        self.battery_capacity = 50000  # Wh
        
        # Orbite actuelle
        self.orbit = None
        
        # Statut
        self.status = "inactive"  # inactive, active, maneuvering, station_keeping
        self.operational_time = 0.0  # heures
        
        # Mission
        self.mission_type = MissionType.EARTH_OBSERVATION
        self.lifetime_years = 5.0
        
    def calculate_delta_v_budget(self) -> Dict:
        """Calcule le budget delta-v"""
        budget = {
            'launch_to_orbit': 0,
            'orbit_raising': 0,
            'station_keeping': 0,
            'deorbit': 0,
            'total': 0
        }
        
        if self.propulsion_system:
            # Delta-v disponible
            total_dv = self.propulsion_system.delta_v_capability(
                self.dry_mass + self.payload_mass,
                self.propellant_mass
            )
            
            # Allocation typique pour LEO
            budget['orbit_raising'] = 100  # m/s
            budget['station_keeping'] = 50 * self.lifetime_years  # m/s/an
            budget['deorbit'] = 100  # m/s
            budget['total'] = sum([v for k, v in budget.items() if k != 'total'])
            budget['available'] = total_dv
            
        return budget

# ==================== MISSION SPATIALE ====================

class SpaceMission:
    """Mission spatiale complète"""
    
    def __init__(self, mission_id: str, name: str, mission_type: MissionType):
        self.id = mission_id
        self.name = name
        self.type = mission_type
        self.created_at = datetime.now().isoformat()
        
        # Véhicules
        self.satellites = []
        
        # Trajectoire
        self.trajectory = []
        self.maneuvers = []
        
        # Paramètres mission
        self.launch_date = None
        self.arrival_date = None
        self.mission_duration = 0.0  # jours
        
        # Objectifs
        self.primary_objectives = []
        self.secondary_objectives = []
        
        # Budget
        self.delta_v_budget = 0.0  # m/s
        self.propellant_budget = 0.0  # kg
        
        # Statut
        self.status = "planning"  # planning, ready, active, completed
        self.progress = 0.0  # %
        
        # Résultats
        self.science_data = 0.0  # GB
        self.images_captured = 0
        self.communications = 0
        
    def add_satellite(self, satellite: Satellite):
        """Ajoute un satellite à la mission"""
        self.satellites.append(satellite)
    
    def calculate_mission_parameters(self):
        """Calcule les paramètres de la mission"""
        if self.launch_date and self.arrival_date:
            delta = datetime.fromisoformat(self.arrival_date) - datetime.fromisoformat(self.launch_date)
            self.mission_duration = delta.total_seconds() / 86400  # jours

# ==================== GESTIONNAIRE PRINCIPAL ====================

class SpaceMechanicsManager:
    """Gestionnaire de la plateforme de mécanique spatiale"""
    
    def __init__(self):
        self.satellites = {}
        self.missions = {}
        self.orbits = {}
        self.maneuvers = {}
        self.simulations = []
        
    def create_satellite(self, name: str, config: Dict) -> str:
        """Crée un nouveau satellite"""
        sat_id = f"sat_{len(self.satellites) + 1}"
        satellite = Satellite(sat_id, name)
        
        # Configuration
        satellite.dry_mass = config.get('dry_mass', 1000)
        satellite.propellant_mass = config.get('propellant_mass', 500)
        satellite.payload_mass = config.get('payload_mass', 200)
        satellite.total_mass = satellite.dry_mass + satellite.propellant_mass + satellite.payload_mass
        
        satellite.power_generation = config.get('power', 5000)
        satellite.lifetime_years = config.get('lifetime', 5)
        
        # Propulsion
        prop_type = PropulsionType(config.get('propulsion', 'chimique'))
        satellite.propulsion_system = PropulsionSystem(f"prop_{sat_id}", prop_type)
        
        self.satellites[sat_id] = satellite
        
        return sat_id
    
    def create_orbit(self, name: str, orbital_elements: Dict) -> str:
        """Crée une orbite"""
        orbit_id = f"orbit_{len(self.orbits) + 1}"
        orbit = Orbit(orbit_id, name)
        
        orbit.semi_major_axis = orbital_elements.get('semi_major_axis', 7000000)
        orbit.eccentricity = orbital_elements.get('eccentricity', 0.0)
        orbit.inclination = orbital_elements.get('inclination', 0.0) * SpaceConstants.DEG_TO_RAD
        orbit.raan = orbital_elements.get('raan', 0.0) * SpaceConstants.DEG_TO_RAD
        orbit.argument_periapsis = orbital_elements.get('arg_periapsis', 0.0) * SpaceConstants.DEG_TO_RAD
        orbit.true_anomaly = orbital_elements.get('true_anomaly', 0.0) * SpaceConstants.DEG_TO_RAD
        
        orbit.calculate_parameters()
        
        self.orbits[orbit_id] = orbit
        
        return orbit_id
    
    def get_satellite(self, sat_id: str) -> Optional[Satellite]:
        """Récupère un satellite"""
        return self.satellites.get(sat_id)

# ==================== POINT D'ENTRÉE ====================

if __name__ == "__main__":
    manager = SpaceMechanicsManager()
    
    # Exemple: Créer un satellite en LEO
    sat_id = manager.create_satellite(
        "ObservationSat-1",
        {
            'dry_mass': 1500,
            'propellant_mass': 300,
            'payload_mass': 400,
            'power': 6000,
            'lifetime': 7,
            'propulsion': 'electrique'
        }
    )
    
    # Créer orbite LEO
    orbit_id = manager.create_orbit(
        "LEO 500km",
        {
            'semi_major_axis': 6871000,  # Terre + 500km
            'eccentricity': 0.0,
            'inclination': 97.8,  # Héliosynchrone
            'raan': 0.0,
            'arg_periapsis': 0.0,
            'true_anomaly': 0.0
        }
    )
    
    satellite = manager.get_satellite(sat_id)
    orbit = manager.orbits[orbit_id]
    
    print(json.dumps({
        'satellite': {
            'name': satellite.name,
            'mass': f"{satellite.total_mass} kg",
            'power': f"{satellite.power_generation} W"
        },
        'orbit': {
            'name': orbit.name,
            'altitude': f"{(orbit.semi_major_axis - SpaceConstants.EARTH_RADIUS)/1000:.1f} km",
            'period': f"{orbit.period/60:.1f} min",
            'velocity': f"{orbit.velocity_periapsis:.0f} m/s"
        }
    }, indent=2))