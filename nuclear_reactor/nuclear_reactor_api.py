"""
Plateforme Complète de Réacteurs Nucléaires
Architecture pour créer, développer, fabriquer, tester et analyser
tous types de réacteurs nucléaires et systèmes énergétiques
uvicorn nuclear_reactor_api:app --host 0.0.0.0 --port 8034 --reload
"""

import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid
from fastapi import FastAPI

app = FastAPI()

# ==================== CONSTANTES PHYSIQUES NUCLÉAIRES ====================

class NuclearConstants:
    """Constantes fondamentales nucléaires"""
    AVOGADRO = 6.02214076e23  # mol⁻¹
    ATOMIC_MASS_UNIT = 1.66053906660e-27  # kg
    ELECTRON_VOLT = 1.602176634e-19  # J
    PLANCK = 6.62607015e-34  # J·s
    BOLTZMANN = 1.380649e-23  # J/K
    SPEED_LIGHT = 299792458  # m/s
    
    # Masses atomiques (uma)
    NEUTRON_MASS = 1.008664916  # uma
    PROTON_MASS = 1.007276467  # uma
    ELECTRON_MASS = 0.000548579909  # uma
    
    # Énergies caractéristiques
    U235_FISSION_ENERGY = 200.0  # MeV par fission
    U238_FISSION_ENERGY = 205.0  # MeV
    PU239_FISSION_ENERGY = 210.0  # MeV
    
    # Sections efficaces (barns = 10⁻²⁴ cm²)
    U235_THERMAL_FISSION = 585.0  # barns
    U238_THERMAL_FISSION = 0.0  # barns (seuil)
    PU239_THERMAL_FISSION = 750.0  # barns
    
    # Neutrons par fission
    NU_U235 = 2.43
    NU_U238 = 2.47
    NU_PU239 = 2.89

# ==================== ENUMS ET TYPES ====================

class ReactorType(Enum):
    PWR = "reacteur_eau_pressurisee"  # REP
    BWR = "reacteur_eau_bouillante"  # REB
    PHWR = "reacteur_eau_lourde"  # CANDU
    GCR = "reacteur_graphite_gaz"  # Magnox
    LMFBR = "reacteur_rapide_sodium"  # Superphénix
    MSR = "reacteur_sels_fondus"  # Molten Salt
    HTR = "reacteur_haute_temperature"  # HTGR
    FUSION = "reacteur_fusion"  # ITER
    SMR = "petit_reacteur_modulaire"  # SMR
    GEN_IV = "generation_4"  # Gen IV

class FuelType(Enum):
    UO2 = "dioxyde_uranium"  # UO₂
    MOX = "oxyde_mixte"  # Mixed Oxide (U+Pu)
    METAL = "uranium_metallique"
    THORIUM = "thorium"  # Th-232
    PLUTONIUM = "plutonium"  # Pu-239
    LIQUID_SALT = "sel_fondu"
    TRITIUM = "tritium_fusion"  # Pour fusion
    DEUTERIUM = "deuterium_fusion"

class ModeratorType(Enum):
    LIGHT_WATER = "eau_legere"  # H₂O
    HEAVY_WATER = "eau_lourde"  # D₂O
    GRAPHITE = "graphite"  # C
    BERYLLIUM = "beryllium"  # Be
    NONE = "aucun"  # Réacteur rapide

class CoolantType(Enum):
    LIGHT_WATER = "eau_legere"
    HEAVY_WATER = "eau_lourde"
    LIQUID_SODIUM = "sodium_liquide"
    HELIUM = "helium"
    LEAD = "plomb"
    LEAD_BISMUTH = "plomb_bismuth"
    MOLTEN_SALT = "sel_fondu"
    CO2 = "dioxyde_carbone"

class ControlMaterial(Enum):
    BORON = "bore"  # B-10
    CADMIUM = "cadmium"  # Cd
    HAFNIUM = "hafnium"  # Hf
    SILVER_INDIUM_CADMIUM = "argent_indium_cadmium"  # AIC
    BORON_CARBIDE = "carbure_bore"  # B₄C

class SafetySystemType(Enum):
    SCRAM = "arret_urgence"
    ECCS = "refroidissement_urgence"  # Emergency Core Cooling
    CONTAINMENT = "enceinte_confinement"
    PASSIVE_COOLING = "refroidissement_passif"
    PRESSURE_RELIEF = "soupape_surpression"
    HYDROGEN_RECOMBINERS = "recombineurs_hydrogene"

# ==================== NEUTRONIQUE ====================

class Neutronics:
    """Calculs neutroniques"""
    
    def __init__(self):
        self.k_effective = 1.0
        self.flux_distribution = []
        self.power_distribution = []
        self.burnup = 0.0  # MWd/tU
        
    def calculate_k_effective(self, fuel_enrichment: float, moderator: str, 
                             temperature: float, burnup: float) -> float:
        """Calcule le facteur de multiplication effectif"""
        # Formule simplifiée
        k_inf = 1.3 * (fuel_enrichment / 3.5)  # k∞ approximatif
        
        # Corrections
        temp_coeff = -2e-5 * (temperature - 300)  # Coefficient température
        burnup_coeff = -1e-5 * burnup  # Épuisement combustible
        
        k_eff = k_inf * (1 + temp_coeff + burnup_coeff) * 0.95  # Fuites
        
        return max(0.5, min(1.5, k_eff))
    
    def calculate_six_factor_formula(self, params: Dict) -> Dict:
        """Formule des six facteurs: k∞ = ε·p·f·η"""
        epsilon = params.get('fast_fission_factor', 1.02)  # ε
        p = params.get('resonance_escape', 0.87)  # p
        f = params.get('thermal_utilization', 0.71)  # f
        eta = params.get('reproduction_factor', 2.07)  # η
        
        k_infinity = epsilon * p * f * eta
        
        # Facteurs géométriques
        L_squared = params.get('migration_area', 350)  # cm²
        B_squared = params.get('buckling', 8e-4)  # cm⁻²
        
        non_leakage = 1 / (1 + L_squared * B_squared)
        k_effective = k_infinity * non_leakage
        
        return {
            'k_infinity': k_infinity,
            'k_effective': k_effective,
            'epsilon': epsilon,
            'p': p,
            'f': f,
            'eta': eta,
            'non_leakage': non_leakage
        }
    
    def calculate_neutron_flux(self, power: float, volume: float, 
                              energy: str = "thermal") -> float:
        """Calcule le flux neutronique (n/cm²/s)"""
        if energy == "thermal":
            sigma_f = NuclearConstants.U235_THERMAL_FISSION * 1e-24  # cm²
        else:
            sigma_f = 2.0 * 1e-24  # Approximation rapide
        
        energy_per_fission = NuclearConstants.U235_FISSION_ENERGY * 1e6 * NuclearConstants.ELECTRON_VOLT  # J
        
        # Puissance = Flux × sigma_f × N × E_fission × Volume
        # Donc Flux = Puissance / (sigma_f × N × E_fission × Volume)
        
        N = 0.024e24  # Densité atomes fissiles (atomes/cm³)
        flux = (power * 1e6) / (sigma_f * N * energy_per_fission * volume * 1e6)
        
        return flux
    
    def diffusion_equation_solver(self, mesh_size: int, boundary_conditions: Dict) -> np.ndarray:
        """Résout l'équation de diffusion neutronique"""
        # Équation: -D∇²φ + Σₐφ = νΣ_f φ
        
        D = 1.0  # Coefficient de diffusion (cm)
        Sigma_a = 0.1  # Section efficace absorption (cm⁻¹)
        nu_Sigma_f = 0.12  # Production neutrons
        
        # Maillage 1D simplifié
        dx = 100.0 / mesh_size  # cm
        phi = np.ones(mesh_size)
        
        # Itérations
        for _ in range(100):
            phi_new = phi.copy()
            for i in range(1, mesh_size - 1):
                laplacian = (phi[i+1] - 2*phi[i] + phi[i-1]) / dx**2
                phi_new[i] = (nu_Sigma_f * phi[i]) / (Sigma_a - D * laplacian)
            
            phi_new[0] = boundary_conditions.get('left', 0)
            phi_new[-1] = boundary_conditions.get('right', 0)
            
            if np.max(np.abs(phi_new - phi)) < 1e-6:
                break
            
            phi = phi_new
        
        return phi / np.max(phi)  # Normalisation

# ==================== THERMODYNAMIQUE ====================

class ThermalHydraulics:
    """Thermohydraulique du réacteur"""
    
    def __init__(self):
        self.core_temperature = 300.0  # K
        self.coolant_temperature_inlet = 300.0
        self.coolant_temperature_outlet = 300.0
        self.pressure = 155.0  # bar (PWR)
        self.flow_rate = 0.0  # kg/s
        
    def calculate_heat_transfer(self, power: float, surface_area: float, 
                               coolant_temp: float, fuel_temp: float) -> Dict:
        """Calcule le transfert thermique"""
        # Coefficient de transfert thermique (W/m²K)
        h = 50000  # Typique pour eau sous pression
        
        # Flux thermique
        q_flux = power * 1e6 / surface_area  # W/m²
        
        # Température surface combustible
        T_surface = coolant_temp + q_flux / h
        
        # Température centre combustible (conduction)
        k_fuel = 3.0  # Conductivité UO₂ (W/mK)
        radius = 0.005  # m (rayon pastille)
        T_center = T_surface + q_flux * radius / (4 * k_fuel)
        
        return {
            'heat_flux': q_flux,
            'surface_temperature': T_surface,
            'center_temperature': T_center,
            'heat_transfer_coeff': h
        }
    
    def calculate_dnbr(self, heat_flux: float, pressure: float, 
                       quality: float, mass_flux: float) -> float:
        """Calcule le DNBR (Departure from Nucleate Boiling Ratio)"""
        # Corrélation W-3 simplifiée
        CHF_critical = 1e6 * (2.022 - 0.0004302 * pressure + 
                             0.1722 * np.exp(-18.177 * quality))
        
        DNBR = CHF_critical / heat_flux
        
        return max(0, DNBR)
    
    def coolant_properties(self, temperature: float, pressure: float) -> Dict:
        """Propriétés du fluide caloporteur"""
        # Pour l'eau
        if temperature < 273.15:
            phase = "solid"
            density = 917  # kg/m³
        elif temperature < 373.15 + 0.5 * pressure:  # Approximation
            phase = "liquid"
            density = 1000 - 0.2 * (temperature - 273.15)
        else:
            phase = "vapor"
            density = 0.6 * pressure / (8.314 * temperature / 18)
        
        cp = 4180 if phase == "liquid" else 2080  # J/kgK
        viscosity = 0.001 if phase == "liquid" else 1e-5  # Pa·s
        
        return {
            'phase': phase,
            'density': density,
            'specific_heat': cp,
            'viscosity': viscosity,
            'temperature': temperature,
            'pressure': pressure
        }

# ==================== COMBUSTIBLE ====================

class NuclearFuel:
    """Combustible nucléaire"""
    
    def __init__(self, fuel_type: FuelType, enrichment: float):
        self.fuel_type = fuel_type
        self.enrichment = enrichment  # % U-235 ou Pu-239
        
        # Propriétés physiques
        self.density = 10970  # kg/m³ (UO₂)
        self.thermal_conductivity = 3.0  # W/mK
        self.melting_point = 3120  # K
        
        # Inventaire isotopique
        self.inventory = {
            'U235': enrichment,
            'U238': 100 - enrichment if fuel_type == FuelType.UO2 else 0,
            'Pu239': 0,
            'Pu240': 0,
            'Pu241': 0,
            'FP': 0  # Produits de fission
        }
        
        # Burnup
        self.burnup = 0.0  # MWd/tU
        self.max_burnup = 60000  # MWd/tU
        
    def evolve_composition(self, power: float, time: float):
        """Évolution de la composition isotopique"""
        # Simplifié: épuisement U-235, production Pu
        
        burnup_increment = power * time / 1000  # MWd/tU approximatif
        self.burnup += burnup_increment
        
        # Taux de combustion
        burn_rate = burnup_increment / 50000
        
        # U-235 diminue
        self.inventory['U235'] *= (1 - 0.1 * burn_rate)
        
        # Pu-239 augmente (conversion U-238)
        self.inventory['Pu239'] += 0.05 * burn_rate * self.inventory['U238']
        self.inventory['U238'] *= (1 - 0.01 * burn_rate)
        
        # Produits de fission augmentent
        self.inventory['FP'] += 5 * burn_rate
    
    def calculate_reactivity_coefficients(self) -> Dict:
        """Calcule les coefficients de réactivité"""
        # Coefficient de température combustible (pcm/K)
        alpha_fuel = -2.5
        
        # Coefficient de température modérateur (pcm/K)
        alpha_moderator = -50 if self.fuel_type == FuelType.UO2 else -20
        
        # Coefficient de vide (pcm/%void)
        alpha_void = -100 if self.fuel_type == FuelType.UO2 else +50
        
        # Coefficient de puissance (pcm/% power)
        alpha_power = -10
        
        return {
            'fuel_temp_coeff': alpha_fuel,
            'moderator_temp_coeff': alpha_moderator,
            'void_coeff': alpha_void,
            'power_coeff': alpha_power
        }

# ==================== RÉACTEUR ====================

class NuclearReactor:
    """Réacteur nucléaire complet"""
    
    def __init__(self, reactor_id: str, name: str, reactor_type: ReactorType):
        self.id = reactor_id
        self.name = name
        self.type = reactor_type
        self.created_at = datetime.now().isoformat()
        
        # Spécifications
        self.thermal_power = 3000  # MWth
        self.electric_power = 1000  # MWe
        self.efficiency = 33.3  # %
        
        # Cœur
        self.core_height = 3.66  # m
        self.core_diameter = 3.37  # m
        self.core_volume = 0.0
        self.n_assemblies = 157
        self.n_rods_per_assembly = 264
        
        # Combustible
        self.fuel_type = FuelType.UO2
        self.fuel_enrichment = 4.5  # %
        self.fuel_mass = 80000  # kg
        self.fuel = None
        
        # Modérateur et caloporteur
        self.moderator = ModeratorType.LIGHT_WATER
        self.coolant = CoolantType.LIGHT_WATER
        
        # Paramètres thermohydrauliques
        self.coolant_inlet_temp = 293  # °C
        self.coolant_outlet_temp = 325  # °C
        self.pressure = 155  # bar
        self.flow_rate = 17500  # kg/s
        
        # Neutronique
        self.k_effective = 1.0
        self.neutron_flux = 0.0
        self.power_density = 100  # kW/L
        
        # Contrôle
        self.control_rod_worth = -10000  # pcm
        self.n_control_rods = 53
        self.rod_position = 0.0  # % insertion (0=sorti, 100=inséré)
        
        # Sécurité
        self.safety_systems = []
        self.scram_systems = []
        
        # État opérationnel
        self.status = "shutdown"  # shutdown, startup, operation, refueling
        self.power_level = 0.0  # % puissance nominale
        self.reactor_period = 0.0  # secondes
        self.operational_hours = 0.0
        self.capacity_factor = 0.0
        
        # Cycle combustible
        self.cycle_length = 18  # mois
        self.cycles_completed = 0
        self.burnup_average = 0.0  # MWd/tU
        
        # Enceinte de confinement
        self.containment = {
            'type': 'double_enceinte',
            'volume': 50000,  # m³
            'design_pressure': 5.0,  # bar
            'leak_rate': 0.1  # %/jour
        }
        
        # Économie
        self.construction_cost = 5000  # M€
        self.fuel_cost_per_year = 50  # M€
        self.operation_cost_per_year = 100  # M€
        self.decommissioning_cost = 1000  # M€
        
        # Production
        self.energy_produced = 0.0  # MWh
        self.co2_avoided = 0.0  # tonnes
        
        # Incidents et sûreté
        self.incidents = []
        self.ines_level = 0  # 0-7 (INES scale)
        self.scrams = 0
        
    def initialize_core(self):
        """Initialise le cœur du réacteur"""
        self.core_volume = np.pi * (self.core_diameter/2)**2 * self.core_height
        self.fuel = NuclearFuel(self.fuel_type, self.fuel_enrichment)
        
    def startup_sequence(self) -> Dict:
        """Séquence de démarrage du réacteur"""
        sequence = {
            'phase': [],
            'duration': [],
            'actions': []
        }
        
        # Phase 1: Approche sous-critique
        sequence['phase'].append('Approche sous-critique')
        sequence['duration'].append('2 heures')
        sequence['actions'].append('Retrait barres contrôle progressif, k_eff < 0.99')
        
        # Phase 2: Criticité
        sequence['phase'].append('Criticité')
        sequence['duration'].append('1 heure')
        sequence['actions'].append('k_eff = 1.000, stabilisation')
        
        # Phase 3: Montée en puissance
        sequence['phase'].append('Montée en puissance')
        sequence['duration'].append('24 heures')
        sequence['actions'].append('0% → 100% puissance thermique')
        
        # Phase 4: Couplage réseau
        sequence['phase'].append('Couplage réseau')
        sequence['duration'].append('2 heures')
        sequence['actions'].append('Synchronisation alternateur')
        
        self.status = "startup"
        
        return sequence
    
    def operate(self, duration_hours: float) -> Dict:
        """Opère le réacteur pendant une durée donnée"""
        if self.status != "operation":
            return {'error': 'Reactor not in operation mode'}
        
        result = {
            'duration': duration_hours,
            'energy_produced': 0.0,
            'burnup_increment': 0.0,
            'fuel_consumed': 0.0,
            'incidents': 0
        }
        
        # Production d'énergie
        energy = self.electric_power * duration_hours * (self.power_level / 100)
        result['energy_produced'] = energy
        self.energy_produced += energy
        
        # Burnup
        burnup_inc = (self.thermal_power * duration_hours * (self.power_level / 100) * 
                     24 / self.fuel_mass / 1000)  # MWd/tU
        result['burnup_increment'] = burnup_inc
        self.burnup_average += burnup_inc
        
        # Évolution combustible
        if self.fuel:
            self.fuel.evolve_composition(self.thermal_power, duration_hours / 24)
        
        # CO2 évité (vs charbon: ~1000 gCO2/kWh)
        co2_avoided = energy * 1000 * 1.0  # tonnes
        self.co2_avoided += co2_avoided
        
        # Heures de fonctionnement
        self.operational_hours += duration_hours
        
        return result
    
    def scram(self, reason: str):
        """Arrêt d'urgence du réacteur"""
        self.status = "shutdown"
        self.power_level = 0.0
        self.rod_position = 100.0  # Barres insérées
        self.scrams += 1
        
        incident = {
            'timestamp': datetime.now().isoformat(),
            'type': 'SCRAM',
            'reason': reason,
            'power_before': self.power_level
        }
        
        self.incidents.append(incident)
        
        return incident
    
    def calculate_reactivity_balance(self) -> Dict:
        """Calcule le bilan de réactivité"""
        # Réactivité totale disponible
        rho_total = 0.0
        
        # Contribution barres de contrôle
        rho_control = self.control_rod_worth * (self.rod_position / 100)
        
        # Empoisonnement Xénon-135
        rho_xenon = -2800 if self.power_level > 50 else -1000  # pcm
        
        # Empoisonnement Samarium-149
        rho_samarium = -700  # pcm
        
        # Température
        rho_temp = -50 * (self.coolant_outlet_temp - 293) / 100  # pcm
        
        # Burnup
        rho_burnup = -100 * (self.burnup_average / 10000)  # pcm
        
        rho_total = rho_control + rho_xenon + rho_samarium + rho_temp + rho_burnup
        
        return {
            'total': rho_total,
            'control_rods': rho_control,
            'xenon': rho_xenon,
            'samarium': rho_samarium,
            'temperature': rho_temp,
            'burnup': rho_burnup
        }

# ==================== CYCLE COMBUSTIBLE ====================

class FuelCycle:
    """Cycle du combustible nucléaire"""
    
    def __init__(self):
        self.stages = []
        
    def front_end(self, uranium_mass: float) -> Dict:
        """Amont du cycle (extraction → fabrication)"""
        stages = {
            'mining': {
                'input': f'{uranium_mass * 10} tonnes minerai',
                'output': f'{uranium_mass} tonnes U3O8',
                'cost': uranium_mass * 50  # $/kg U
            },
            'conversion': {
                'input': f'{uranium_mass} tonnes U3O8',
                'output': f'{uranium_mass * 0.85} tonnes UF6',
                'cost': uranium_mass * 10
            },
            'enrichment': {
                'input': f'{uranium_mass * 0.85} tonnes UF6 naturel',
                'output': f'{uranium_mass * 0.17} tonnes UF6 enrichi (4.5%)',
                'swu_required': uranium_mass * 120,  # SWU
                'cost': uranium_mass * 120
            },
            'fabrication': {
                'input': f'{uranium_mass * 0.17} tonnes UF6',
                'output': f'{uranium_mass * 0.15} tonnes assemblages UO2',
                'cost': uranium_mass * 250
            }
        }
        
        total_cost = sum(stage['cost'] for stage in stages.values())
        
        return {
            'stages': stages,
            'total_cost': total_cost,
            'fuel_produced': uranium_mass * 0.15
        }
    
    def back_end(self, spent_fuel_mass: float, reprocessing: bool = False) -> Dict:
        """Aval du cycle (combustible usé)"""
        if reprocessing:
            # Retraitement
            stages = {
                'cooling': {
                    'duration': '5 ans',
                    'location': 'Piscine réacteur'
                },
                'transport': {
                    'distance': '500 km',
                    'cost': spent_fuel_mass * 100
                },
                'reprocessing': {
                    'input': f'{spent_fuel_mass} tonnes combustible usé',
                    'output_pu': f'{spent_fuel_mass * 0.01} tonnes Pu',
                    'output_u': f'{spent_fuel_mass * 0.95} tonnes U',
                    'output_waste': f'{spent_fuel_mass * 0.04} tonnes déchets',
                    'cost': spent_fuel_mass * 1000
                },
                'mox_fabrication': {
                    'input': 'Pu + U appauvri',
                    'output': f'{spent_fuel_mass * 0.3} tonnes MOX',
                    'cost': spent_fuel_mass * 500
                }
            }
        else:
            # Stockage direct
            stages = {
                'cooling': {
                    'duration': '10 ans',
                    'location': 'Piscine réacteur'
                },
                'interim_storage': {
                    'duration': '50 ans',
                    'type': 'Entreposage sec',
                    'cost': spent_fuel_mass * 200
                },
                'final_disposal': {
                    'duration': '100,000 ans',
                    'type': 'Stockage géologique profond',
                    'depth': '500 m',
                    'cost': spent_fuel_mass * 1500
                }
            }
        
        total_cost = sum(stage.get('cost', 0) for stage in stages.values())
        
        return {
            'strategy': 'reprocessing' if reprocessing else 'direct_disposal',
            'stages': stages,
            'total_cost': total_cost
        }

# ==================== SÛRETÉ ====================

class SafetySystems:
    """Systèmes de sûreté"""
    
    def __init__(self):
        self.systems = []
        self.barriers = ['combustible', 'gaine', 'circuit_primaire', 'enceinte']
        
    def defense_in_depth(self) -> Dict:
        """Défense en profondeur (5 niveaux)"""
        levels = {
            'level_1': {
                'name': 'Prévention',
                'objective': 'Éviter incidents',
                'measures': ['Conception robuste', 'Qualité fabrication', 'Contrôle qualité']
            },
            'level_2': {
                'name': 'Surveillance et contrôle',
                'objective': 'Détecter anomalies',
                'measures': ['Instrumentation', 'Systèmes contrôle', 'Procédures']
            },
            'level_3': {
                'name': 'Systèmes de sauvegarde',
                'objective': 'Gérer incidents',
                'measures': ['SCRAM', 'ECCS', 'Refroidissement']
            },
            'level_4': {
                'name': 'Gestion accidents graves',
                'objective': 'Limiter rejets',
                'measures': ['Récupérateur corium', 'Filtration', 'Éventage']
            },
            'level_5': {
                'name': 'Atténuation conséquences',
                'objective': 'Protéger population',
                'measures': ['Plan urgence', 'Évacuation', 'Confinement']
            }
        }
        
        return levels
    
    def accident_analysis(self, accident_type: str) -> Dict:
        """Analyse d'accidents"""
        accidents = {
            'loca': {  # Loss of Coolant Accident
                'name': 'Perte de réfrigérant primaire',
                'probability': '1e-4 /réacteur-an',
                'consequences': 'Fusion cœur possible',
                'mitigation': ['ECCS', 'Accumulateurs', 'RIS']
            },
            'reactivity_insertion': {
                'name': 'Insertion réactivité',
                'probability': '1e-5 /réacteur-an',
                'consequences': 'Excursion puissance',
                'mitigation': ['SCRAM', 'Doppler', 'Bore']
            },
            'station_blackout': {
                'name': 'Perte alimentation électrique',
                'probability': '1e-6 /réacteur-an',
                'consequences': 'Perte refroidissement',
                'mitigation': ['Diesel secours', 'Batteries', 'Refroidissement passif']
            },
            'steam_line_break': {
                'name': 'Rupture ligne vapeur',
                'probability': '1e-3 /réacteur-an',
                'consequences': 'Refroidissement rapide',
                'mitigation': ['Isolation', 'Injection bore']
            }
        }
        
        return accidents.get(accident_type, {})

# ==================== RADIOPROTECTION ====================

class RadiationProtection:
    """Radioprotection et dosimétrie"""
    
    def __init__(self):
        self.dose_limits = {
            'public': 1.0,  # mSv/an
            'workers': 20.0,  # mSv/an
            'emergency': 100.0  # mSv
        }
        
    def calculate_dose(self, activity: float, distance: float, 
                      time: float, shielding: float = 1.0) -> float:
        """Calcule la dose de radiation (mSv)"""
        # Formule simplifiée
        # Dose ∝ Activité × temps / distance² × facteur blindage
        
        dose_rate = activity * 1e-3 / (distance ** 2)  # mSv/h
        dose = dose_rate * time * shielding
        
        return dose
    
    def shielding_calculation(self, material: str, thickness: float, 
                             energy: float) -> float:
        """Calcule l'atténuation du blindage"""
        # Coefficients d'atténuation (cm⁻¹)
        mu_values = {
            'lead': 1.2,
            'concrete': 0.2,
            'water': 0.08,
            'steel': 0.6
        }
        
        mu = mu_values.get(material.lower(), 0.1)
        attenuation = np.exp(-mu * thickness)
        
        return attenuation
    
    def alara_principle(self) -> Dict:
        """Principe ALARA (As Low As Reasonably Achievable)"""
        return {
            'time': 'Minimiser temps exposition',
            'distance': 'Maximiser distance à la source',
            'shielding': 'Utiliser blindage approprié',
            'planning': 'Planification interventions',
            'training': 'Formation personnel',
            'monitoring': 'Surveillance dosimétrique'
        }

# ==================== DÉCHETS RADIOACTIFS ====================

class RadioactiveWaste:
    """Gestion des déchets radioactifs"""
    
    def __init__(self):
        self.inventory = {
            'TFA': 0.0,  # Très Faible Activité
            'FA_VC': 0.0,  # Faible Activité Vie Courte
            'MA_VL': 0.0,  # Moyenne Activité Vie Longue
            'HA': 0.0  # Haute Activité
        }
        
    def classify_waste(self, activity: float, half_life: float) -> str:
        """Classe les déchets selon activité et demi-vie"""
        if activity < 100:  # Bq/g
            return 'TFA'
        elif activity < 1e6 and half_life < 31:  # ans
            return 'FA_VC'
        elif activity < 1e9:
            return 'MA_VL'
        else:
            return 'HA'
    
    def calculate_decay(self, initial_activity: float, half_life: float, 
                       time: float) -> float:
        """Calcule la décroissance radioactive"""
        # A(t) = A₀ × e^(-λt)
        # λ = ln(2) / T₁/₂
        
        decay_constant = np.log(2) / half_life
        activity = initial_activity * np.exp(-decay_constant * time)
        
        return activity
    
    def disposal_strategy(self, waste_type: str) -> Dict:
        """Stratégie de stockage"""
        strategies = {
            'TFA': {
                'method': 'Stockage surface',
                'duration': '30 ans surveillance',
                'location': 'Centre CSTFA',
                'cost': '100 €/m³'
            },
            'FA_VC': {
                'method': 'Stockage sub-surface',
                'duration': '300 ans confinement',
                'location': 'Centre CSA',
                'cost': '1,000 €/m³'
            },
            'MA_VL': {
                'method': 'Stockage géologique',
                'duration': '> 10,000 ans',
                'location': 'Cigéo',
                'depth': '500 m',
                'cost': '10,000 €/m³'
            },
            'HA': {
                'method': 'Stockage géologique profond',
                'duration': '> 100,000 ans',
                'location': 'Cigéo',
                'depth': '500 m',
                'vitrification': True,
                'cost': '1,000,000 €/m³'
            }
        }
        
        return strategies.get(waste_type, {})

# ==================== GESTIONNAIRE PRINCIPAL ====================

class NuclearReactorManager:
    """Gestionnaire de la plateforme de réacteurs nucléaires"""
    
    def __init__(self):
        self.reactors = {}
        self.fuel_cycles = {}
        self.waste_inventory = {}
        self.safety_reports = []
        self.simulations = []
        
    def create_reactor(self, name: str, reactor_type: str, config: Dict) -> str:
        """Crée un nouveau réacteur"""
        reactor_id = f"reactor_{len(self.reactors) + 1}"
        reactor = NuclearReactor(reactor_id, name, ReactorType(reactor_type))
        
        # Configuration
        reactor.thermal_power = config.get('thermal_power', 3000)
        reactor.electric_power = config.get('electric_power', 1000)
        reactor.efficiency = (reactor.electric_power / reactor.thermal_power) * 100
        
        reactor.fuel_enrichment = config.get('enrichment', 4.5)
        reactor.core_height = config.get('core_height', 3.66)
        reactor.core_diameter = config.get('core_diameter', 3.37)
        
        reactor.coolant_inlet_temp = config.get('inlet_temp', 293)
        reactor.coolant_outlet_temp = config.get('outlet_temp', 325)
        reactor.pressure = config.get('pressure', 155)
        
        reactor.construction_cost = config.get('construction_cost', 5000)
        
        reactor.initialize_core()
        
        self.reactors[reactor_id] = reactor
        
        return reactor_id
    
    def get_reactor(self, reactor_id: str) -> Optional[NuclearReactor]:
        """Récupère un réacteur"""
        return self.reactors.get(reactor_id)
    
    def simulate_reactor_operation(self, reactor_id: str, 
                                   duration_days: float) -> Dict:
        """Simule l'opération d'un réacteur"""
        reactor = self.get_reactor(reactor_id)
        if not reactor:
            return {'error': 'Reactor not found'}
        
        result = reactor.operate(duration_days * 24)
        
        simulation = {
            'sim_id': f"sim_{len(self.simulations) + 1}",
            'reactor_id': reactor_id,
            'duration_days': duration_days,
            'timestamp': datetime.now().isoformat(),
            'results': result
        }
        
        self.simulations.append(simulation)
        
        return simulation

# ==================== POINT D'ENTRÉE ====================

if __name__ == "__main__":
    manager = NuclearReactorManager()
    
    # Exemple: Créer un réacteur type EPR
    reactor_id = manager.create_reactor(
        "EPR Flamanville 3",
        "reacteur_eau_pressurisee",
        {
            'thermal_power': 4500,  # MWth
            'electric_power': 1650,  # MWe
            'enrichment': 5.0,  # %
            'core_height': 4.2,  # m
            'core_diameter': 3.76,  # m
            'inlet_temp': 295,  # °C
            'outlet_temp': 330,  # °C
            'pressure': 155,  # bar
            'construction_cost': 12400  # M€
        }
    )
    
    reactor = manager.get_reactor(reactor_id)
    
    print(json.dumps({
        'reactor': {
            'name': reactor.name,
            'type': reactor.type.value,
            'power_thermal': f"{reactor.thermal_power} MWth",
            'power_electric': f"{reactor.electric_power} MWe",
            'efficiency': f"{reactor.efficiency:.1f}%"
        }
    }, indent=2))