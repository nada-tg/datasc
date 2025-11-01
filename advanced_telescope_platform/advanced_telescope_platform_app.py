"""
🔭 Advanced Space Telescope Platform - Complete Frontend
Observatoires • Télescopes Spatiaux • Deep Space • IA Astronomique

Installation:
pip install streamlit pandas plotly numpy scipy astropy

Lancement:
streamlit run advanced_telescope_platform_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import time  

# ==================== CONFIGURATION PAGE ====================
st.set_page_config(
    page_title="🔭 Space Telescope Lab",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLES CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 30%, #f093fb 60%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: cosmic-glow 3s ease-in-out infinite alternate;
    }
    @keyframes cosmic-glow {
        from { filter: drop-shadow(0 0 20px #667eea); }
        to { filter: drop-shadow(0 0 40px #4facfe); }
    }
    .telescope-card {
        border: 3px solid #667eea;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    .telescope-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(118, 75, 162, 0.6);
    }
    .star-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .observation-active {
        animation: telescope-scan 2s infinite;
    }
    @keyframes telescope-scan {
        0%, 100% { opacity: 0.8; transform: rotate(0deg); }
        50% { opacity: 1; transform: rotate(5deg); }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ASTRONOMIQUES ====================
ASTRO_CONSTANTS = {
    # Constantes fondamentales
    'c': 299792458,  # Vitesse lumière (m/s)
    'h': 6.62607015e-34,  # Constante Planck (J.s)
    'k_B': 1.380649e-23,  # Constante Boltzmann (J/K)
    'G': 6.67430e-11,  # Constante gravitationnelle (m³/kg/s²)
    'AU': 1.496e11,  # Unité Astronomique (m)
    'parsec': 3.086e16,  # Parsec (m)
    'ly': 9.461e15,  # Année-lumière (m)
    
    # Soleil
    'M_sun': 1.989e30,  # Masse solaire (kg)
    'R_sun': 6.96e8,  # Rayon solaire (m)
    'L_sun': 3.828e26,  # Luminosité solaire (W)
    
    # Terre
    'M_earth': 5.972e24,  # Masse Terre (kg)
    'R_earth': 6.371e6,  # Rayon Terre (m)
    
    # Limites détection
    'hubble_limit_mag': 31,  # Magnitude limite Hubble
    'jwst_limit_mag': 32,  # Magnitude limite JWST
    'elt_limit_mag': 35,  # Magnitude limite ELT (futur)
}

TELESCOPE_TYPES = {
    'Spatial': {
        'description': 'Télescope en orbite (pas d\'atmosphère)',
        'avantages': 'Résolution maximale, UV/IR accessible',
        'exemples': 'Hubble, JWST, Chandra, Spitzer',
        'wavelengths': 'UV à IR lointain',
        'resolution': '0.05 arcsec',
        'color': '#667eea'
    },
    'Sol - Optique': {
        'description': 'Télescope terrestre visible/proche IR',
        'avantages': 'Grande ouverture, maintenance facile',
        'exemples': 'VLT, Keck, GMT, ELT',
        'wavelengths': '0.4-2.5 μm',
        'resolution': '0.01 arcsec (avec AO)',
        'color': '#764ba2'
    },
    'Radio': {
        'description': 'Radiotélescope (ondes radio)',
        'avantages': 'Pénètre nuages, synchrotron',
        'exemples': 'ALMA, VLA, SKA, FAST',
        'wavelengths': 'mm à m',
        'resolution': '0.001 arcsec (VLBI)',
        'color': '#f093fb'
    },
    'Gamma/X': {
        'description': 'Haute énergie (rayons X/gamma)',
        'avantages': 'Objets énergétiques, trous noirs',
        'exemples': 'Chandra, XMM-Newton, Fermi',
        'wavelengths': '0.01-10 nm',
        'resolution': 'Variable',
        'color': '#4facfe'
    }
}

CELESTIAL_OBJECTS = {
    'Étoiles': {
        'types': ['Naine Rouge', 'Solaire', 'Géante', 'Supergéante', 'Naine Blanche'],
        'magnitude_range': [-26.7, 15],  # Soleil à étoile faible
        'distance_range': [1, 1000],  # parsecs
        'color': '#FFD700'
    },
    'Exoplanètes': {
        'types': ['Hot Jupiter', 'Super-Terre', 'Neptune', 'Terrestre'],
        'magnitude_range': [15, 30],
        'distance_range': [1, 100],
        'color': '#4169E1'
    },
    'Galaxies': {
        'types': ['Spirale', 'Elliptique', 'Irrégulière', 'Naine'],
        'magnitude_range': [8, 25],
        'distance_range': [0.1, 13000],  # Mpc
        'color': '#9370DB'
    },
    'Nébuleuses': {
        'types': ['Émission', 'Réflexion', 'Planétaire', 'Supernova'],
        'magnitude_range': [5, 20],
        'distance_range': [0.1, 10],  # kpc
        'color': '#FF69B4'
    },
    'Trous Noirs': {
        'types': ['Stellaire', 'Supermassif', 'Intermédiaire'],
        'magnitude_range': [20, 35],
        'distance_range': [1, 13000],
        'color': '#000000'
    }
}

# ==================== INITIALISATION SESSION STATE ====================
if 'telescope_lab' not in st.session_state:
    st.session_state.telescope_lab = {
        'telescopes': {},
        'observations': [],
        'discoveries': [],
        'targets': {},
        'images': [],
        'spectra': [],
        'ai_detections': [],
        'quantum_analysis': [],
        'exoplanet_candidates': [],
        'galaxy_catalog': [],
        'monitoring_campaigns': [],
        'collaborations': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str, level: str = "INFO"):
    """Enregistrer événement"""
    st.session_state.telescope_lab['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_magnitude(flux: float, distance_pc: float) -> float:
    """Calculer magnitude apparente"""
    # m = M + 5*log10(d/10)
    absolute_mag = -2.5 * np.log10(flux)
    apparent_mag = absolute_mag + 5 * np.log10(distance_pc / 10)
    return apparent_mag

def calculate_angular_resolution(diameter_m: float, wavelength_m: float) -> float:
    """Calculer résolution angulaire (critère Rayleigh)"""
    # θ = 1.22 * λ / D (radians)
    theta_rad = 1.22 * wavelength_m / diameter_m
    theta_arcsec = theta_rad * 206265  # Conversion radians -> arcsec
    return theta_arcsec

def calculate_limiting_magnitude(diameter_m: float, exposure_s: float, 
                                 quantum_efficiency: float = 0.8) -> float:
    """Calculer magnitude limite"""
    # Formule simplifiée
    base_limit = 2.5 * np.log10(diameter_m**2) + 2.5 * np.log10(exposure_s)
    limit_mag = 20 + base_limit + 2.5 * np.log10(quantum_efficiency)
    return limit_mag

def doppler_shift(wavelength: float, velocity_km_s: float) -> float:
    """Calculer décalage Doppler"""
    # Δλ/λ = v/c
    c_km_s = ASTRO_CONSTANTS['c'] / 1000
    delta_lambda = wavelength * (velocity_km_s / c_km_s)
    return wavelength + delta_lambda

def simulate_transit(period_days: float, duration_h: float, depth_percent: float,
                    n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Simuler courbe transit exoplanète"""
    time = np.linspace(0, period_days, n_points)
    flux = np.ones(n_points)
    
    # Transit au milieu
    transit_start = period_days/2 - duration_h/48
    transit_end = period_days/2 + duration_h/48
    
    in_transit = (time >= transit_start) & (time <= transit_end)
    flux[in_transit] = 1 - depth_percent/100
    
    # Ajouter bruit
    flux += np.random.normal(0, 0.001, n_points)
    
    return time, flux

def generate_spectrum(temp_K: float, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Générer spectre corps noir"""
    wavelength = np.linspace(100, 3000, n_points)  # nm
    
    # Loi de Planck
    h = ASTRO_CONSTANTS['h']
    c = ASTRO_CONSTANTS['c']
    k_B = ASTRO_CONSTANTS['k_B']
    
    lambda_m = wavelength * 1e-9
    
    intensity = (2 * h * c**2 / lambda_m**5) / \
                (np.exp((h * c) / (lambda_m * k_B * temp_K)) - 1)
    
    # Normaliser
    intensity = intensity / np.max(intensity)
    
    return wavelength, intensity

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🔭 Advanced Space Telescope Laboratory</h1>', 
           unsafe_allow_html=True)
st.markdown("### Deep Space Observation • Exoplanets • Galaxies • AI Detection • Quantum Analysis")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/667eea/FFFFFF?text=TelescopeLab", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Observatory Central",
            "🔭 Créer Télescope",
            "🎯 Gestion Cibles",
            "📸 Observations",
            "🌌 Imagerie Profonde",
            "📊 Spectroscopie",
            "🪐 Exoplanètes",
            "🌌 Galaxies",
            "⚫ Trous Noirs",
            "🤖 IA Détection",
            "⚛️ Analyse Quantique",
            "🧬 Bioastronomy",
            "📡 Multi-Messager",
            "🔬 Recherche Vie",
            "🛰️ Missions Spatiales",
            "🌍 Collaborations",
            "📊 Analytics",
            "📡 Monitoring Live",
            "🗺️ Sky Survey",
            "📚 Catalog",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 État Lab")
    
    total_telescopes = len(st.session_state.telescope_lab['telescopes'])
    total_observations = len(st.session_state.telescope_lab['observations'])
    total_discoveries = len(st.session_state.telescope_lab['discoveries'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔭 Télescopes", total_telescopes)
        st.metric("📸 Observations", total_observations)
    with col2:
        st.metric("🌟 Découvertes", total_discoveries)
        st.metric("🪐 Exoplanètes", len(st.session_state.telescope_lab['exoplanet_candidates']))

# ==================== PAGE: OBSERVATORY CENTRAL ====================
if page == "🏠 Observatory Central":
    st.header("🏠 Observatoire Central")
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="telescope-card"><h2>🔭</h2><h3>{total_telescopes}</h3><p>Télescopes</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        observing_time = total_observations * 3600  # secondes
        st.markdown(f'<div class="telescope-card"><h2>⏱️</h2><h3>{observing_time/3600:.0f}h</h3><p>Temps Obs.</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="telescope-card"><h2>🌟</h2><h3>{total_discoveries}</h3><p>Découvertes</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        data_volume_TB = total_observations * 0.5
        st.markdown(f'<div class="telescope-card"><h2>💾</h2><h3>{data_volume_TB:.1f}TB</h3><p>Données</p></div>', 
                   unsafe_allow_html=True)
    
    with col5:
        publications = total_discoveries // 3
        st.markdown(f'<div class="telescope-card"><h2>📄</h2><h3>{publications}</h3><p>Publications</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Télescopes célèbres
    st.subheader("🔭 Télescopes Iconiques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🌌 Télescopes Spatiaux")
        
        famous_space = {
            'Hubble Space Telescope': {
                'launched': '1990',
                'diameter': '2.4 m',
                'wavelength': 'UV-visible-NIR',
                'orbit': '547 km',
                'discoveries': 'Expansion univers, âge univers, trous noirs'
            },
            'James Webb Space Telescope': {
                'launched': '2021',
                'diameter': '6.5 m',
                'wavelength': 'NIR-MIR (0.6-28 μm)',
                'orbit': 'L2 (1.5M km)',
                'discoveries': 'Premières galaxies, exoplanètes, chimie'
            },
            'Chandra X-ray': {
                'launched': '1999',
                'diameter': '1.2 m',
                'wavelength': 'Rayons X',
                'orbit': 'Haute elliptique',
                'discoveries': 'Trous noirs, supernovae, matière noire'
            }
        }
        
        for name, info in famous_space.items():
            with st.expander(f"🛰️ {name}"):
                st.write(f"**Lancé:** {info['launched']}")
                st.write(f"**Diamètre:** {info['diameter']}")
                st.write(f"**Longueurs d'onde:** {info['wavelength']}")
                st.write(f"**Découvertes:** {info['discoveries']}")
    
    with col2:
        st.write("### 🌍 Télescopes Sol")
        
        famous_ground = {
            'ELT (Extremely Large Telescope)': {
                'status': 'En construction',
                'diameter': '39 m',
                'location': 'Chili (Atacama)',
                'first_light': '2028',
                'capabilities': 'Exoplanètes, galaxies primordiales'
            },
            'VLT (Very Large Telescope)': {
                'status': 'Opérationnel',
                'diameter': '4×8.2 m',
                'location': 'Chili (Paranal)',
                'first_light': '1998',
                'capabilities': 'AO, interférométrie, exoplanètes'
            },
            'ALMA': {
                'status': 'Opérationnel',
                'diameter': '66 antennes (12m+7m)',
                'location': 'Chili (5000m)',
                'first_light': '2011',
                'capabilities': 'Molécules, disques protoplanétaires'
            }
        }
        
        for name, info in famous_ground.items():
            with st.expander(f"🏔️ {name}"):
                st.write(f"**Status:** {info['status']}")
                st.write(f"**Diamètre:** {info['diameter']}")
                st.write(f"**Localisation:** {info['location']}")
                st.write(f"**Capacités:** {info['capabilities']}")
    
    st.markdown("---")
    
    # Carte du ciel
    st.subheader("🗺️ Carte du Ciel - Observations")
    
    if st.button("🌌 Générer Carte du Ciel"):
        # Générer positions aléatoires (RA, Dec)
        n_objects = 50
        ra = np.random.uniform(0, 360, n_objects)
        dec = np.random.uniform(-90, 90, n_objects)
        magnitudes = np.random.uniform(10, 25, n_objects)
        
        # Projection Hammer-Aitoff
        fig = go.Figure()
        
        fig.add_trace(go.Scattergeo(
            lon=ra - 180,  # Centrer sur 0
            lat=dec,
            mode='markers',
            marker=dict(
                size=20 - magnitudes/2,
                color=magnitudes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Magnitude")
            ),
            text=[f"Mag: {m:.1f}" for m in magnitudes],
            hovertemplate='RA: %{lon}°<br>Dec: %{lat}°<br>%{text}<extra></extra>'
        ))
        
        fig.update_geos(
            projection_type='hammer',
            showcountries=False,
            showcoastlines=False,
            showland=False,
            bgcolor='#0a0a0a'
        )
        
        fig.update_layout(
            title="Carte Céleste - Objets Observés",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: CRÉER TÉLESCOPE ====================
elif page == "🔭 Créer Télescope":
    st.header("🔭 Conception Télescope")
    
    st.info("""
    **Télescope Personnalisé**
    
    Configurez votre télescope selon les spécifications scientifiques.
    """)
    
    with st.form("create_telescope"):
        col1, col2 = st.columns(2)
        
        with col1:
            telescope_name = st.text_input("Nom Télescope", "DeepSky-1")
            
            telescope_type = st.selectbox("Type",
                list(TELESCOPE_TYPES.keys()))
            
            diameter_m = st.slider("Diamètre Miroir (m)", 0.1, 40.0, 6.5, 0.1)
            
            focal_length_m = st.number_input("Distance Focale (m)", 1.0, 200.0, 20.0)
            
            location = st.selectbox("Localisation",
                ["Orbite Terrestre", "Point L2", "Sol - Désert Atacama",
                 "Sol - Mauna Kea", "Sol - La Palma", "Orbite Lunaire"])
        
        with col2:
            wavelength_range = st.multiselect("Longueurs d'onde",
                ["UV (100-400 nm)", "Visible (400-700 nm)", "NIR (0.7-2.5 μm)",
                 "MIR (2.5-25 μm)", "FIR (25-350 μm)", "Radio (mm-m)", "Rayons X"],
                default=["Visible (400-700 nm)", "NIR (0.7-2.5 μm)"])
            
            instruments = st.multiselect("Instruments",
                ["Caméra Grand Champ", "Spectrographe", "Coronographe",
                 "IFU (Integral Field)", "Polarimètre", "AO (Optique Adaptative)"],
                default=["Caméra Grand Champ", "Spectrographe"])
            
            detector_type = st.selectbox("Détecteur",
                ["CCD", "CMOS", "HgCdTe (IR)", "Bolometer", "MCP (X-ray)"])
            
            field_of_view = st.slider("Champ de Vue (arcmin)", 0.1, 60.0, 10.0, 0.1)
        
        st.write("### 🎯 Objectifs Scientifiques")
        
        science_goals = st.multiselect("Objectifs",
            ["Exoplanètes", "Galaxies lointaines", "Cosmologie", "Étoiles variables",
             "Nébuleuses", "Trous noirs", "Astéroïdes/Comètes", "Supernovae"])
        
        budget_millions = st.slider("Budget (M$)", 10, 10000, 1000)
        
        if st.form_submit_button("🔭 Créer Télescope", type="primary"):
            telescope_id = f"tel_{len(st.session_state.telescope_lab['telescopes']) + 1}"
            
            # Calculs performances
            f_ratio = focal_length_m / diameter_m
            resolution_arcsec = calculate_angular_resolution(diameter_m, 550e-9)  # @550nm
            limit_mag = calculate_limiting_magnitude(diameter_m, 3600)  # 1h exposure
            collecting_area = np.pi * (diameter_m/2)**2
            
            telescope = {
                'id': telescope_id,
                'name': telescope_name,
                'type': telescope_type,
                'diameter_m': diameter_m,
                'focal_length_m': focal_length_m,
                'f_ratio': f_ratio,
                'location': location,
                'wavelength_range': wavelength_range,
                'instruments': instruments,
                'detector_type': detector_type,
                'field_of_view_arcmin': field_of_view,
                'science_goals': science_goals,
                'budget_millions': budget_millions,
                'resolution_arcsec': resolution_arcsec,
                'limiting_magnitude': limit_mag,
                'collecting_area_m2': collecting_area,
                'status': 'operational',
                'created_at': datetime.now().isoformat()
            }
            
            st.session_state.telescope_lab['telescopes'][telescope_id] = telescope
            log_event(f"Télescope créé: {telescope_name}", "SUCCESS")
            
            st.success(f"✅ Télescope '{telescope_name}' créé!")
            st.balloons()
            
            # Afficher performances
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Résolution", f"{resolution_arcsec:.3f}\"")
            with col2:
                st.metric("Magnitude Limite", f"{limit_mag:.1f}")
            with col3:
                st.metric("f/ratio", f"{f_ratio:.1f}")
            with col4:
                st.metric("Surface", f"{collecting_area:.1f} m²")
            
            st.rerun()

            # DANS LA PAGE "🔭 Créer Télescope"
# Après le if st.form_submit_button("🔭 Créer Télescope", type="primary"):
# GARDEZ tout le code jusqu'à st.rerun()
# PUIS AJOUTEZ CETTE SECTION APRÈS le form mais AVANT la fin de la page:

    # Afficher télescopes existants
    if st.session_state.telescope_lab['telescopes']:
        st.markdown("---")
        st.subheader("📋 Télescopes Créés")
        
        for tel_id, tel in st.session_state.telescope_lab['telescopes'].items():
            with st.expander(f"🔭 {tel['name']} - {tel['type']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Diamètre:** {tel['diameter_m']} m")
                    st.write(f"**Type:** {tel['type']}")
                    st.write(f"**Localisation:** {tel['location']}")
                
                with col2:
                    st.metric("Résolution", f"{tel['resolution_arcsec']:.3f}\"")
                    st.metric("Magnitude Limite", f"{tel['limiting_magnitude']:.1f}")
                
                with col3:
                    st.metric("f/ratio", f"{tel['f_ratio']:.1f}")
                    st.metric("Surface", f"{tel['collecting_area_m2']:.1f} m²")

# ==================== PAGE: GESTION CIBLES ====================
elif page == "🎯 Gestion Cibles":
    st.header("🎯 Catalogue Cibles d'Observation")
    
    tab1, tab2, tab3 = st.tabs(["➕ Ajouter Cible", "📋 Liste Cibles", "🗺️ Planification"])
    
    with tab1:
        st.subheader("➕ Ajouter Cible")
        
        with st.form("add_target"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_name = st.text_input("Nom", "NGC 1234")
                
                object_type = st.selectbox("Type Objet",
                    list(CELESTIAL_OBJECTS.keys()))
                
                ra_h = st.number_input("RA (heures)", 0, 24, 12, format="%d")
                ra_m = st.number_input("RA (minutes)", 0, 60, 0, format="%d")
                ra_s = st.number_input("RA (secondes)", 0.0, 60.0, 0.0, format="%.2f")
            
            with col2:
                dec_d = st.number_input("Dec (degrés)", -90, 90, 0, format="%d")
                dec_m = st.number_input("Dec (arcmin)", 0, 60, 0, format="%d")
                dec_s = st.number_input("Dec (arcsec)", 0.0, 60.0, 0.0, format="%.2f")
                
                magnitude = st.slider("Magnitude Apparente", 0.0, 30.0, 15.0, 0.1)
                
                distance_mpc = st.number_input("Distance (Mpc)", 0.001, 13000.0, 100.0)
            
            priority = st.select_slider("Priorité",
                options=["Basse", "Normale", "Haute", "Urgente"])
            
            notes = st.text_area("Notes",
                "Galaxie spirale, candidat lentille gravitationnelle")
            
            if st.form_submit_button("✅ Ajouter Cible"):
                # Convertir coordonnées
                ra_deg = (ra_h + ra_m/60 + ra_s/3600) * 15
                dec_deg = dec_d + dec_m/60 + dec_s/3600
                
                target_id = f"target_{len(st.session_state.telescope_lab['targets']) + 1}"
                
                target = {
                    'id': target_id,
                    'name': target_name,
                    'type': object_type,
                    'ra_deg': ra_deg,
                    'dec_deg': dec_deg,
                    'magnitude': magnitude,
                    'distance_mpc': distance_mpc,
                    'priority': priority,
                    'notes': notes,
                    'observations': 0,
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.telescope_lab['targets'][target_id] = target
                log_event(f"Cible ajoutée: {target_name}", "INFO")
                
                st.success(f"✅ Cible '{target_name}' ajoutée!")
                st.rerun()
    
    with tab2:
        st.subheader("📋 Catalogue Cibles")
        
        if st.session_state.telescope_lab['targets']:
            targets_data = []
            for target in st.session_state.telescope_lab['targets'].values():
                targets_data.append({
                    'Nom': target['name'],
                    'Type': target['type'],
                    'RA': f"{target['ra_deg']:.2f}°",
                    'Dec': f"{target['dec_deg']:.2f}°",
                    'Magnitude': f"{target['magnitude']:.1f}",
                    'Distance': f"{target['distance_mpc']:.1f} Mpc",
                    'Priorité': target['priority'],
                    'Observations': target['observations']
                })
            
            df_targets = pd.DataFrame(targets_data)
            st.dataframe(df_targets, use_container_width=True)
            
            # Filtres
            st.write("### 🔍 Filtres")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.multiselect("Type",
                    list(CELESTIAL_OBJECTS.keys()))
            
            with col2:
                filter_priority = st.multiselect("Priorité",
                    ["Basse", "Normale", "Haute", "Urgente"])
            
            with col3:
                mag_range = st.slider("Magnitude", 0.0, 30.0, (0.0, 30.0))
        
        else:
            st.info("Aucune cible enregistrée")
    
    with tab3:
        st.subheader("🗺️ Planification Observations")
        
        if st.session_state.telescope_lab['targets']:
            st.write("### 📅 Visibilité Tonight")
            
            # Simuler visibilité
            for target_id, target in list(st.session_state.telescope_lab['targets'].items())[:5]:
                dec = target['dec_deg']
                
                # Altitude max (simplifié)
                altitude_max = 90 - abs(dec - 20)  # Latitude observatoire
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{target['name']}** ({target['type']})")
                
                with col2:
                    if altitude_max > 30:
                        st.success(f"✅ Alt max: {altitude_max:.0f}°")
                    else:
                        st.warning(f"⚠️ Alt max: {altitude_max:.0f}°")
                
                with col3:
                    if st.button("📸 Observer", key=f"obs_{target_id}"):
                        st.info(f"Observation planifiée: {target['name']}")
        else:
            st.info("Ajoutez des cibles")

# ==================== PAGE: IA DÉTECTION ====================
elif page == "🤖 IA Détection":
    st.header("🤖 Intelligence Artificielle - Détection Objets")
    
    st.info("""
    **Deep Learning pour Astronomie:**
    - Classification galaxies (morphologie)
    - Détection transients (supernovae, GRB)
    - Lentilles gravitationnelles
    - Astéroïdes/débris
    - Anomalies spectroscopiques
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 CNN Classification", "🔍 Détection Transients", "🌌 Lentilles", "📊 AutoML"])
    
    with tab1:
        st.subheader("🧠 CNN - Classification Galaxies")
        
        st.write("""
        **Réseau Convolutionnel**
        
        Architecture: ResNet-50 fine-tuned
        - Input: 224×224 images
        - Classes: Spirale, Elliptique, Irrégulière, Lenticulaire
        - Accuracy: 96.5%
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_architecture = st.selectbox("Architecture",
                ["ResNet-50", "EfficientNet-B4", "Vision Transformer", "Custom CNN"])
            
            batch_size = st.slider("Batch Size", 16, 256, 64)
            
            training_images = st.number_input("Images Entraînement", 1000, 1000000, 100000)
        
        with col2:
            augmentation = st.multiselect("Data Augmentation",
                ["Rotation", "Flip", "Zoom", "Brightness", "Noise"],
                default=["Rotation", "Flip"])
            
            learning_rate = st.select_slider("Learning Rate",
                options=[1e-5, 1e-4, 1e-3, 1e-2])
        
        if st.button("🧠 Entraîner Modèle"):
            with st.spinner("Entraînement deep learning..."):
                import time
                
                epochs = 20
                losses = []
                accuracies = []
                
                progress = st.progress(0)
                status = st.empty()
                
                for epoch in range(epochs):
                    # Simuler entraînement
                    loss = 2.0 * np.exp(-epoch/5) + 0.1 + np.random.normal(0, 0.05)
                    acc = 0.95 * (1 - np.exp(-epoch/3)) + np.random.normal(0, 0.02)
                    
                    losses.append(loss)
                    accuracies.append(acc)
                    
                    status.write(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.3f} - Acc: {acc:.3f}")
                    progress.progress((epoch + 1) / epochs)
                    time.sleep(0.2)
                
                # Graphiques
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Loss", "Accuracy")
                )
                
                fig.add_trace(go.Scatter(
                    x=list(range(epochs)), y=losses,
                    mode='lines+markers',
                    line=dict(color='#FF6B6B', width=2),
                    name='Loss'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=list(range(epochs)), y=accuracies,
                    mode='lines+markers',
                    line=dict(color='#4ECDC4', width=2),
                    name='Accuracy'
                ), row=1, col=2)
                
                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=1, col=2)
                fig.update_yaxes(title_text="Loss", row=1, col=1)
                fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                
                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Modèle entraîné! Accuracy finale: {accuracies[-1]:.1%}")
                
                # Matrice confusion
                st.write("### 📊 Matrice de Confusion")
                
                classes = ["Spirale", "Elliptique", "Irrégulière", "Lenticulaire"]
                confusion = np.random.randint(10, 100, (4, 4))
                np.fill_diagonal(confusion, np.random.randint(80, 100, 4))
                
                fig = go.Figure(data=go.Heatmap(
                    z=confusion,
                    x=classes,
                    y=classes,
                    colorscale='Blues',
                    text=confusion,
                    texttemplate='%{text}',
                    textfont={"size": 12}
                ))
                
                fig.update_layout(
                    title="Matrice de Confusion",
                    xaxis_title="Prédiction",
                    yaxis_title="Vérité",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Détection Transients Temps Réel")
        
        st.write("""
        **Pipeline Automatique:**
        1. Image différence (nouvelle - référence)
        2. Détection sources
        3. Classification CNN
        4. Filtrage artefacts
        5. Alerte si supernova/GRB
        """)
        
        if st.button("🔍 Scanner Images (100)"):
            with st.spinner("Analyse 100 images..."):
                import time
                time.sleep(2)
                
                # Résultats simulés
                detections = {
                    'Supernovae': np.random.randint(2, 8),
                    'Variables': np.random.randint(10, 30),
                    'Astéroïdes': np.random.randint(50, 200),
                    'Artefacts': np.random.randint(100, 500),
                    'Galaxies variables': np.random.randint(5, 20)
                }
                
                st.success(f"✅ Scan complété!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Transients Détectés", sum(detections.values()))
                with col2:
                    st.metric("Supernovae", detections['Supernovae'])
                with col3:
                    st.metric("Taux Faux Positifs", f"{detections['Artefacts']/sum(detections.values())*100:.1f}%")
                
                # Détails
                st.write("### 📊 Détections par Classe")
                
                fig = go.Figure(data=[go.Bar(
                    x=list(detections.keys()),
                    y=list(detections.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                    text=list(detections.values()),
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title="Détections Transients",
                    xaxis_title="Classe",
                    yaxis_title="Nombre",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if detections['Supernovae'] > 5:
                    st.balloons()
                    st.success(f"🌟 {detections['Supernovae']} supernovae découvertes!")
    
    with tab3:
        st.subheader("🌌 Détection Lentilles Gravitationnelles")
        
        st.write("""
        **Strong Gravitational Lensing**
        
        IA entraînée sur simulations pour détecter:
        - Arcs Einstein
        - Anneaux Einstein
        - Images multiples quasars
        """)
        
        if st.button("🔍 Rechercher Lentilles"):
            with st.spinner("Analyse lentilles gravitationnelles..."):
                import time
                time.sleep(2)
                
                n_candidates = np.random.randint(5, 20)
                
                st.success(f"✅ {n_candidates} candidats lentilles détectés!")
                
                for i in range(min(3, n_candidates)):
                    with st.expander(f"🌌 Candidat Lentille #{i+1}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Propriétés:**")
                            st.write(f"• Redshift lentille: {np.random.uniform(0.3, 0.8):.2f}")
                            st.write(f"• Redshift source: {np.random.uniform(1.5, 3.0):.2f}")
                            st.write(f"• Rayon Einstein: {np.random.uniform(1, 3):.1f}\"")
                        
                        with col2:
                            confidence = np.random.uniform(0.85, 0.99)
                            st.metric("Confiance IA", f"{confidence:.1%}")
                            
                            if confidence > 0.95:
                                st.success("✅ Haute confiance")
                            else:
                                st.info("📊 Confirmation spectroscopique recommandée")
    
    with tab4:
        st.subheader("📊 AutoML - Découverte Automatique")
        
        st.write("""
        **Automated Machine Learning**
        
        Recherche automatique:
        - Architecture optimale
        - Hyperparamètres
        - Features engineering
        - Nouvelles classes objets
        """)
        
        if st.button("🤖 Lancer AutoML"):
            with st.spinner("Exploration espace modèles..."):
                import time
                time.sleep(3)
                
                st.success("✅ Recherche terminée!")
                
                st.write("### 🏆 Meilleur Modèle Trouvé")
                
                best_model = {
                    'Architecture': 'EfficientNet-B5 + Attention',
                    'Accuracy': f"{np.random.uniform(0.96, 0.99):.1%}",
                    'F1-Score': f"{np.random.uniform(0.94, 0.98):.3f}",
                    'Params': '30M',
                    'Inference': '45ms/image'
                }
                
                for key, value in best_model.items():
                    st.write(f"**{key}:** {value}")
                
                st.write("### 🌟 Nouvelles Classes Découvertes")
                
                new_classes = [
                    "Galaxies Ultra-Diffuses",
                    "Naines Tidal",
                    "Lentilles Exotiques"
                ]
                
                for cls in new_classes:
                    st.write(f"• {cls}")
                
                st.balloons()

# ==================== PAGE: OBSERVATIONS ====================
elif page == "📸 Observations":
    st.header("📸 Sessions d'Observation")
    
    if not st.session_state.telescope_lab['telescopes']:
        st.warning("⚠️ Créez d'abord un télescope")
    elif not st.session_state.telescope_lab['targets']:
        st.warning("⚠️ Ajoutez des cibles d'observation")
    else:
        st.info("Prêt à observer!")
        
        with st.form("observation_session"):
            col1, col2 = st.columns(2)
            
            with col1:
                selected_telescope = st.selectbox("Télescope",
                    list(st.session_state.telescope_lab['telescopes'].keys()),
                    format_func=lambda x: st.session_state.telescope_lab['telescopes'][x]['name'])
                
                selected_target = st.selectbox("Cible",
                    list(st.session_state.telescope_lab['targets'].keys()),
                    format_func=lambda x: st.session_state.telescope_lab['targets'][x]['name'])
                
                observation_mode = st.selectbox("Mode",
                    ["Imagerie", "Spectroscopie", "Photométrie", "Polarimétrie"])
            
            with col2:
                exposure_time = st.number_input("Temps Pose (s)", 1, 10800, 600)
                n_exposures = st.number_input("Nombre Poses", 1, 100, 5)
                
                filter_band = st.selectbox("Filtre",
                    ["U", "B", "V", "R", "I", "J", "H", "K", "Clear"])
            
            seeing_arcsec = st.slider("Seeing (arcsec)", 0.3, 3.0, 1.0, 0.1)
            
            if st.form_submit_button("🚀 Lancer Observation", type="primary"):
                telescope = st.session_state.telescope_lab['telescopes'][selected_telescope]
                target = st.session_state.telescope_lab['targets'][selected_target]
                
                with st.spinner("Observation en cours..."):
                    import time
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    for i in range(n_exposures):
                        status.write(f"📸 Exposition {i+1}/{n_exposures}")
                        progress.progress((i + 1) / n_exposures)
                        time.sleep(0.5)
                    
                    # Calculer SNR
                    snr = np.sqrt(exposure_time * telescope['collecting_area_m2']) * np.random.uniform(0.8, 1.2)
                    
                    # Détections
                    n_sources = int(np.random.uniform(50, 500))
                    
                    observation = {
                        'telescope_id': selected_telescope,
                        'target_id': selected_target,
                        'mode': observation_mode,
                        'exposure_time_s': exposure_time,
                        'n_exposures': n_exposures,
                        'filter': filter_band,
                        'seeing_arcsec': seeing_arcsec,
                        'snr': snr,
                        'n_sources_detected': n_sources,
                        'limiting_mag': telescope['limiting_magnitude'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.telescope_lab['observations'].append(observation)
                    target['observations'] += 1
                    
                    log_event(f"Observation: {target['name']} avec {telescope['name']}", "SUCCESS")
                    
                    st.success("✅ Observation complétée!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("SNR", f"{snr:.1f}")
                    with col2:
                        st.metric("Sources", n_sources)
                    with col3:
                        st.metric("Mag Limite", f"{telescope['limiting_magnitude']:.1f}")
                    with col4:
                        st.metric("Seeing", f"{seeing_arcsec}\"")
                    
                    st.rerun()
        
        # Historique
        st.markdown("---")
        st.subheader("📋 Historique Observations")
        
        if st.session_state.telescope_lab['observations']:
            st.write(f"**{len(st.session_state.telescope_lab['observations'])} observations effectuées**")
            
            for obs in st.session_state.telescope_lab['observations'][-5:][::-1]:
                telescope = st.session_state.telescope_lab['telescopes'][obs['telescope_id']]
                target = st.session_state.telescope_lab['targets'][obs['target_id']]
                
                with st.expander(f"📸 {target['name']} - {obs['timestamp'][:19]}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Télescope:** {telescope['name']}")
                        st.write(f"**Mode:** {obs['mode']}")
                    
                    with col2:
                        st.write(f"**Exposition:** {obs['exposure_time_s']}s × {obs['n_exposures']}")
                        st.write(f"**Filtre:** {obs['filter']}")
                    
                    with col3:
                        st.metric("SNR", f"{obs['snr']:.1f}")
                        st.metric("Sources", obs['n_sources_detected'])

# ==================== PAGE: IMAGERIE PROFONDE ====================
elif page == "🌌 Imagerie Profonde":
    st.header("🌌 Deep Space Imaging")
    
    st.info("""
    **Imagerie Ultra-Profonde**
    
    Techniques avancées pour détecter objets extrêmement faibles.
    - Poses longues (heures)
    - Stacking d'images
    - Soustraction fond de ciel
    - Traitement IA
    """)
    
    tab1, tab2, tab3 = st.tabs(["📸 Capture", "🎨 Traitement", "🔍 Analyse"])
    
    with tab1:
        st.subheader("📸 Capture Deep Field")
        
        if st.session_state.telescope_lab['telescopes']:
            telescope_id = st.selectbox("Télescope",
                list(st.session_state.telescope_lab['telescopes'].keys()),
                format_func=lambda x: st.session_state.telescope_lab['telescopes'][x]['name'],
                key="deep_tel")
            
            telescope = st.session_state.telescope_lab['telescopes'][telescope_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                total_exposure_h = st.slider("Exposition Totale (heures)", 1, 100, 10)
                
                # Filtres multiples pour couleur
                filters = st.multiselect("Filtres",
                    ["U", "B", "V", "R", "I"],
                    default=["B", "V", "R"])
            
            with col2:
                dithering = st.checkbox("Dithering (réduire défauts détecteur)", value=True)
                drizzling = st.checkbox("Drizzling (augmenter résolution)", value=True)
                
                cosmic_ray_removal = st.checkbox("Suppression rayons cosmiques", value=True)
            
            if st.button("🌌 Lancer Deep Field", type="primary"):
                with st.spinner(f"Acquisition {total_exposure_h}h en cours..."):
                    import time
                    time.sleep(2)
                    
                    # Générer image simulée
                    size = 512
                    
                    # Fond de ciel + bruit
                    image = np.random.poisson(50, (size, size)).astype(float)
                    
                    # Ajouter galaxies faibles
                    n_galaxies = int(np.random.uniform(20, 50))
                    for _ in range(n_galaxies):
                        x = np.random.randint(0, size)
                        y = np.random.randint(0, size)
                        brightness = np.random.uniform(100, 500)
                        sigma = np.random.uniform(2, 8)
                        
                        # Gaussienne 2D
                        y_grid, x_grid = np.ogrid[-y:size-y, -x:size-x]
                        galaxy = brightness * np.exp(-(x_grid**2 + y_grid**2)/(2*sigma**2))
                        image += galaxy
                    
                    # Normaliser
                    image = (image - image.min()) / (image.max() - image.min())
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=image,
                        colorscale='Viridis',
                        showscale=False
                    ))
                    
                    fig.update_layout(
                        title=f"Deep Field - {total_exposure_h}h exposition",
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False),
                        template="plotly_dark",
                        height=600,
                        width=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    detected_galaxies = n_galaxies + np.random.randint(-5, 5)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Galaxies Détectées", detected_galaxies)
                    with col2:
                        mag_limit = telescope['limiting_magnitude'] + 2.5*np.log10(total_exposure_h)
                        st.metric("Magnitude Limite", f"{mag_limit:.1f}")
                    with col3:
                        st.metric("Données", f"{total_exposure_h * 2:.1f} GB")
                    
                    st.success("✅ Deep Field complété!")
        else:
            st.info("Créez un télescope")
    
    with tab2:
        st.subheader("🎨 Pipeline Traitement")
        
        st.write("""
        **Étapes Traitement:**
        1. Calibration (bias, dark, flat)
        2. Alignement images (astrométrie)
        3. Stacking (moyenne/médiane)
        4. Soustraction fond de ciel
        5. Détection sources
        6. Photométrie
        7. Couleur (si multi-bande)
        """)
        
        if st.button("⚙️ Traiter Images"):
            with st.spinner("Traitement pipeline..."):
                import time
                
                steps = [
                    "📊 Calibration",
                    "🎯 Alignement astrométrique",
                    "📚 Stacking 100 images",
                    "🌌 Soustraction fond ciel",
                    "🔍 Détection sources",
                    "📈 Photométrie"
                ]
                
                progress = st.progress(0)
                status = st.empty()
                
                for i, step in enumerate(steps):
                    status.write(f"**{step}**")
                    progress.progress((i + 1) / len(steps))
                    time.sleep(0.5)
                
                st.success("✅ Traitement terminé!")
                
                st.write("### 📊 Résultats")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sources", np.random.randint(500, 2000))
                with col2:
                    st.metric("Galaxies", np.random.randint(100, 500))
                with col3:
                    st.metric("Qualité Image", f"{np.random.uniform(0.85, 0.98):.2f}")
    
    with tab3:
        st.subheader("🔍 Analyse IA")
        
        st.write("""
        **Deep Learning pour Classification:**
        - CNN pour morphologie galaxies
        - Détection lentilles gravitationnelles
        - Transients (supernovae)
        - Astéroïdes/artefacts
        """)
        
        if st.button("🤖 Lancer Analyse IA"):
            with st.spinner("Classification deep learning..."):
                import time
                time.sleep(2)
                
                # Résultats simulés
                classifications = {
                    'Galaxies Spirales': np.random.randint(50, 150),
                    'Galaxies Elliptiques': np.random.randint(80, 200),
                    'Galaxies Irrégulières': np.random.randint(20, 80),
                    'Lentilles Gravitationnelles': np.random.randint(1, 10),
                    'Candidats Supernovae': np.random.randint(2, 15),
                    'Quasars': np.random.randint(5, 30)
                }
                
                fig = go.Figure(data=[go.Bar(
                    x=list(classifications.keys()),
                    y=list(classifications.values()),
                    marker_color='#667eea',
                    text=list(classifications.values()),
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title="Classification Automatique Objets",
                    xaxis_title="Classe",
                    yaxis_title="Nombre",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Classification complétée!")
                
                if classifications['Lentilles Gravitationnelles'] > 5:
                    st.balloons()
                    st.success(f"🎉 {classifications['Lentilles Gravitationnelles']} lentilles gravitationnelles découvertes!")

# ==================== PAGE: SPECTROSCOPIE ====================
elif page == "📊 Spectroscopie":
    st.header("📊 Analyse Spectroscopique")
    
    st.info("""
    **Spectroscopie Astronomique**
    
    Décomposer lumière pour obtenir:
    - Composition chimique
    - Température
    - Vitesse radiale (effet Doppler)
    - Redshift cosmologique
    """)
    
    tab1, tab2, tab3 = st.tabs(["📡 Acquisition", "📈 Analyse", "🌈 Base Données"])
    
    with tab1:
        st.subheader("📡 Spectrographe")
        
        col1, col2 = st.columns(2)
        
        with col1:
            spectro_type = st.selectbox("Type Spectrographe",
                ["Basse Résolution (R~100)", "Moyenne Résolution (R~1000)",
                 "Haute Résolution (R~10000)", "Échelle (R~100000)"])
            
            wavelength_range = st.slider("Domaine λ (nm)", 300, 2500, (400, 900))
            
            integration_time = st.number_input("Temps Intégration (s)", 60, 7200, 1800)
        
        with col2:
            target_select = st.selectbox("Cible",
                ["Étoile Type G", "Étoile Type M", "Galaxie z=0.5",
                 "Quasar z=2.0", "Nébuleuse Émission", "Supernova"])
            
            snr_target = st.slider("SNR Cible", 10, 200, 50)
        
        if st.button("📡 Acquérir Spectre", type="primary"):
            with st.spinner("Acquisition spectre..."):
                import time
                time.sleep(1.5)
                
                # Générer spectre simulé
                wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
                
                # Spectre selon type cible
                if "Étoile" in target_select:
                    if "G" in target_select:
                        temp = 5800  # K
                    else:
                        temp = 3500  # K (M)
                    
                    _, spectrum = generate_spectrum(temp)
                    spectrum = spectrum[:len(wavelengths)]
                    
                    # Ajouter raies absorption
                    if "G" in target_select:
                        # H-alpha, Na D, Ca II
                        absorption_lines = [656.3, 589.0, 393.4, 396.8]
                        for line in absorption_lines:
                            if wavelength_range[0] < line < wavelength_range[1]:
                                idx = np.argmin(np.abs(wavelengths - line))
                                spectrum[max(0,idx-5):min(len(spectrum),idx+5)] *= 0.7
                
                else:  # Galaxie/Quasar
                    spectrum = np.random.exponential(0.5, len(wavelengths))
                    spectrum = spectrum / spectrum.max()
                    
                    # Redshift
                    if "z=" in target_select:
                        z = float(target_select.split("z=")[1].split(")")[0])
                        wavelengths = wavelengths * (1 + z)
                
                # Ajouter bruit
                noise = np.random.normal(0, 1/snr_target, len(spectrum))
                spectrum_noisy = spectrum + noise
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=wavelengths,
                    y=spectrum_noisy,
                    mode='lines',
                    line=dict(color='#667eea', width=1),
                    name='Spectre Observé'
                ))
                
                fig.update_layout(
                    title=f"Spectre: {target_select}",
                    xaxis_title="Longueur d'onde (nm)",
                    yaxis_title="Flux (u.a.)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mesures
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "Étoile" in target_select:
                        st.metric("Température", f"{temp} K")
                with col2:
                    measured_snr = np.median(spectrum_noisy) / np.std(noise)
                    st.metric("SNR", f"{measured_snr:.1f}")
                with col3:
                    if "z=" in target_select:
                        st.metric("Redshift", f"{z:.2f}")
                
                # Sauvegarder
                spectrum_data = {
                    'target': target_select,
                    'wavelengths': wavelengths.tolist(),
                    'flux': spectrum_noisy.tolist(),
                    'snr': float(measured_snr),
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.telescope_lab['spectra'].append(spectrum_data)
                
                st.success("✅ Spectre acquis!")
    
    with tab2:
        st.subheader("📈 Analyse Raies Spectrales")
        
        if st.session_state.telescope_lab['spectra']:
            st.write("### 🔬 Dernier Spectre")
            
            last_spectrum = st.session_state.telescope_lab['spectra'][-1]
            
            st.write(f"**Cible:** {last_spectrum['target']}")
            st.write(f"**SNR:** {last_spectrum['snr']:.1f}")
            
            if st.button("🔍 Identifier Raies"):
                st.write("### 📊 Raies Identifiées")
                
                # Raies communes
                lines_db = {
                    'H-alpha': 656.3,
                    'H-beta': 486.1,
                    'Na D': 589.0,
                    'Ca II K': 393.4,
                    'Ca II H': 396.8,
                    'Mg I': 518.4
                }
                
                detected_lines = []
                for name, wavelength in lines_db.items():
                    if np.random.random() > 0.5:  # Simuler détection
                        detected_lines.append({
                            'Raie': name,
                            'λ (nm)': wavelength,
                            'EW (Å)': np.random.uniform(0.1, 2.0),
                            'SNR': np.random.uniform(5, 50)
                        })
                
                if detected_lines:
                    df_lines = pd.DataFrame(detected_lines)
                    st.dataframe(df_lines, use_container_width=True)
                    
                    st.success(f"✅ {len(detected_lines)} raies identifiées")
                else:
                    st.info("Aucune raie détectée avec confiance suffisante")
        else:
            st.info("Acquérez un spectre")
    
    with tab3:
        st.subheader("🌈 Bibliothèque Spectres")
        
        if st.session_state.telescope_lab['spectra']:
            st.write(f"### 📚 {len(st.session_state.telescope_lab['spectra'])} Spectres Archivés")
            
            for i, spec in enumerate(st.session_state.telescope_lab['spectra'][::-1][:5]):
                with st.expander(f"📊 Spectre #{len(st.session_state.telescope_lab['spectra'])-i}: {spec['target']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Cible:** {spec['target']}")
                        st.write(f"**SNR:** {spec['snr']:.1f}")
                    
                    with col2:
                        st.write(f"**Date:** {spec['timestamp'][:19]}")
                        st.write(f"**Points:** {len(spec['wavelengths'])}")
        else:
            st.info("Aucun spectre enregistré")

# ==================== PAGE: EXOPLANÈTES ====================
elif page == "🪐 Exoplanètes":
    st.header("🪐 Détection Exoplanètes")
    
    st.info("""
    **Méthodes Détection:**
    - Transit (variation luminosité)
    - Vitesse radiale (effet Doppler)
    - Imagerie directe (coronographe)
    - Microlentille gravitationnelle
    - Astrométrie (wobble étoile)
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌑 Transit", "📈 Vitesse Radiale", "🔭 Imagerie Directe", "📊 Catalogue"])
    
    with tab1:
        st.subheader("🌑 Méthode Transit")
        
        st.write("""
        **Principe:**
        Planète passe devant étoile → diminution luminosité
        
        Informations obtenues:
        - Rayon planète (profondeur transit)
        - Période orbitale
        - Inclinaison orbite
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            planet_radius_r_earth = st.slider("Rayon Planète (R⊕)", 0.5, 20.0, 1.0, 0.1)
            star_radius_r_sun = st.slider("Rayon Étoile (R☉)", 0.1, 2.0, 1.0, 0.1)
            
            orbital_period_days = st.slider("Période Orbitale (jours)", 0.5, 365.0, 10.0, 0.5)
        
        with col2:
            transit_duration_h = st.slider("Durée Transit (heures)", 0.5, 12.0, 3.0, 0.5)
            
            # Calculer profondeur transit
            depth_percent = (planet_radius_r_earth * ASTRO_CONSTANTS['R_earth'])**2 / \
                          (star_radius_r_sun * ASTRO_CONSTANTS['R_sun'])**2 * 100
            
            st.metric("Profondeur Transit", f"{depth_percent:.3f}%")
            
            if depth_percent < 0.01:
                st.warning("⚠️ Transit très faible, difficile à détecter")
            else:
                st.success("✅ Transit détectable")
        
        if st.button("🌑 Simuler Transit"):
            # Générer courbe transit
            time, flux = simulate_transit(orbital_period_days, transit_duration_h, depth_percent)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time, y=flux,
                mode='lines+markers',
                line=dict(color='#667eea', width=2),
                marker=dict(size=4)
            ))
            
            fig.add_hline(y=1.0, line_dash="dash", line_color="white",
                         annotation_text="Flux nominal")
            
            fig.update_layout(
                title=f"Courbe de Lumière - Transit Exoplanète",
                xaxis_title="Temps (jours)",
                yaxis_title="Flux Relatif",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Transit simulé!")
            
            # Caractériser planète
            st.write("### 🪐 Caractérisation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Type", "Super-Terre" if planet_radius_r_earth < 2 else "Neptune" if planet_radius_r_earth < 6 else "Jupiter")
            with col2:
                # Distance orbitale (loi Kepler simplifiée)
                a_AU = (orbital_period_days / 365.25)**(2/3) * star_radius_r_sun**0.5
                st.metric("Distance", f"{a_AU:.3f} AU")
            with col3:
                # Température équilibre (simplifiée)
                T_eq = 280 * (star_radius_r_sun / a_AU)**0.5
                st.metric("T équilibre", f"{T_eq:.0f} K")
            
            # Sauvegarder candidat
            if st.button("💾 Enregistrer Candidat"):
                exoplanet = {
                    'radius_r_earth': planet_radius_r_earth,
                    'period_days': orbital_period_days,
                    'transit_depth': depth_percent,
                    'semi_major_axis_AU': a_AU,
                    'equilibrium_temp_K': T_eq,
                    'detection_method': 'Transit',
                    'confirmed': False,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.telescope_lab['exoplanet_candidates'].append(exoplanet)
                log_event(f"Candidat exoplanète: R={planet_radius_r_earth:.1f}R⊕", "SUCCESS")

                st.success("✅ Candidat enregistré!")
                st.balloons()
    
    with tab2:
        st.subheader("📈 Vitesse Radiale")
        
        st.write("""
        **Principe:**
        Planète fait osciller étoile → Doppler shift périodique
        
        Amplitude variation dépend de:
        - Masse planète
        - Période orbitale
        - Excentricité orbite
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            planet_mass_m_jup = st.slider("Masse Planète (M_Jup)", 0.1, 10.0, 1.0, 0.1)
            period_days_rv = st.slider("Période (jours)", 1.0, 1000.0, 100.0, 1.0)
            
            star_mass_m_sun = st.slider("Masse Étoile (M☉)", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            eccentricity = st.slider("Excentricité", 0.0, 0.9, 0.0, 0.05)
            
            # Calculer amplitude RV (semi-amplitude K)
            # Formule simplifiée
            K_m_s = 28.4 * planet_mass_m_jup * np.sin(60*np.pi/180) / \
                    (star_mass_m_sun**(2/3) * (period_days_rv/365.25)**(1/3)) / \
                    np.sqrt(1 - eccentricity**2)
            
            st.metric("Amplitude RV", f"{K_m_s:.2f} m/s")
            
            if K_m_s < 1:
                st.error("❌ Trop faible pour détecter")
            elif K_m_s < 3:
                st.warning("⚠️ Nécessite spectrographe haute résolution")
            else:
                st.success("✅ Détectable")
        
        if st.button("📈 Simuler Courbe RV"):
            time = np.linspace(0, period_days_rv * 3, 100)
            
            # Vitesse radiale (circulaire + excentrique)
            if eccentricity < 0.01:
                rv = K_m_s * np.sin(2 * np.pi * time / period_days_rv)
            else:
                # Approximation pour orbite excentrique
                mean_anomaly = 2 * np.pi * time / period_days_rv
                rv = K_m_s * (np.sin(mean_anomaly) + eccentricity * np.sin(2*mean_anomaly))
            
            # Ajouter bruit instrumental
            noise = np.random.normal(0, 0.5, len(rv))
            rv_noisy = rv + noise
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time, y=rv_noisy,
                mode='markers',
                marker=dict(size=6, color='#764ba2'),
                name='Mesures'
            ))
            
            fig.add_trace(go.Scatter(
                x=time, y=rv,
                mode='lines',
                line=dict(color='#667eea', width=3),
                name='Modèle'
            ))
            
            fig.update_layout(
                title="Courbe Vitesse Radiale",
                xaxis_title="Temps (jours)",
                yaxis_title="Vitesse Radiale (m/s)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Courbe RV simulée!")
    
    with tab3:
        st.subheader("🔭 Imagerie Directe")
        
        st.write("""
        **Principe:**
        Bloquer lumière étoile (coronographe) → Image planète directe
        
        Extrêmement difficile:
        - Contraste 10⁶-10⁹
        - Séparation angulaire faible
        - Nécessite optique adaptative
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            separation_AU = st.slider("Séparation Planète-Étoile (AU)", 5, 100, 30)
            distance_pc = st.number_input("Distance Système (pc)", 1, 100, 10)
            
            planet_temp_K = st.slider("Température Planète (K)", 500, 2000, 1000)
        
        with col2:
            # Calculer séparation angulaire
            separation_arcsec = separation_AU / distance_pc
            
            st.metric("Séparation Angulaire", f"{separation_arcsec:.3f}\"")
            
            # Contraste
            contrast_ratio = 10**(-6)  # Simplifié
            st.metric("Contraste", f"10⁻⁶")
            
            if separation_arcsec < 0.1:
                st.error("❌ Trop proche, inobservable")
            elif separation_arcsec < 0.3:
                st.warning("⚠️ Nécessite coronographe + AO extrême")
            else:
                st.success("✅ Observable avec grand télescope")
        
        if st.button("📸 Simuler Imagerie Directe"):
            # Générer image simulée
            size = 256
            center = size // 2
            
            # Point Spread Function (PSF) étoile
            y, x = np.ogrid[-center:size-center, -center:size-center]
            
            # Étoile (bloquée par coronographe)
            star_psf = np.exp(-(x**2 + y**2) / 50)
            star_psf = star_psf * 0.01  # Réduction coronographe
            
            # Planète
            planet_x = center + int(separation_arcsec * 50)
            planet_y = center
            
            planet_psf = np.exp(-((x-planet_x+center)**2 + (y-planet_y+center)**2) / 10)
            planet_psf = planet_psf * contrast_ratio * 1e6  # Rendre visible
            
            # Image totale + bruit
            image = star_psf + planet_psf
            image += np.random.normal(0, 0.001, image.shape)
            
            fig = go.Figure(data=go.Heatmap(
                z=image,
                colorscale='Hot',
                showscale=False
            ))
            
            fig.update_layout(
                title="Imagerie Directe - Coronographe",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Planète détectée!")
            st.write(f"🪐 Position: {separation_arcsec:.2f}\" à l'Est")
    
    with tab4:
        st.subheader("📊 Catalogue Exoplanètes")
        
        if st.session_state.telescope_lab['exoplanet_candidates']:
            st.write(f"### 🪐 {len(st.session_state.telescope_lab['exoplanet_candidates'])} Candidats")
            
            candidates_data = []
            for i, planet in enumerate(st.session_state.telescope_lab['exoplanet_candidates']):
                candidates_data.append({
                    '#': i+1,
                    'Rayon (R⊕)': f"{planet['radius_r_earth']:.1f}",
                    'Période (jours)': f"{planet['period_days']:.1f}",
                    'Distance (AU)': f"{planet.get('semi_major_axis_AU', 0):.2f}",
                    'T_eq (K)': f"{planet.get('equilibrium_temp_K', 0):.0f}",
                    'Méthode': planet['detection_method'],
                    'Confirmé': '✅' if planet['confirmed'] else '⏳'
                })
            
            df_exo = pd.DataFrame(candidates_data)
            st.dataframe(df_exo, use_container_width=True)
            
            # Diagramme période-rayon
            st.write("### 📊 Diagramme Période-Rayon")
            
            radii = [p['radius_r_earth'] for p in st.session_state.telescope_lab['exoplanet_candidates']]
            periods = [p['period_days'] for p in st.session_state.telescope_lab['exoplanet_candidates']]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=periods, y=radii,
                mode='markers',
                marker=dict(size=12, color='#667eea'),
                text=[f"Candidat {i+1}" for i in range(len(radii))],
                hovertemplate='P: %{x:.1f} jours<br>R: %{y:.1f} R⊕<extra></extra>'
            ))
            
            # Zones caractéristiques
            fig.add_hrect(y0=0, y1=2, fillcolor="green", opacity=0.1,
                         annotation_text="Terrestres", annotation_position="left")
            fig.add_hrect(y0=2, y1=6, fillcolor="blue", opacity=0.1,
                         annotation_text="Super-Terres/Neptunes", annotation_position="left")
            fig.add_hrect(y0=6, y1=20, fillcolor="orange", opacity=0.1,
                         annotation_text="Géantes Gazeuses", annotation_position="left")
            
            fig.update_layout(
                title="Distribution Exoplanètes",
                xaxis_title="Période Orbitale (jours)",
                yaxis_title="Rayon (R⊕)",
                xaxis_type="log",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Aucun candidat exoplanète détecté")








                
                    

# ==================== PAGE: ANALYSE QUANTIQUE ====================
elif page == "⚛️ Analyse Quantique":
    st.header("⚛️ Technologies Quantiques en Astronomie")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Intrication", "💫 Téléportation", "🎲 Computing", "🌌 Cosmologie"])
    
    with tab1:
        st.subheader("🔬 Intrication Quantique - Interférométrie")
        
        st.write("""
        **Quantum-Enhanced Interferometry:**
        
        L'intrication entre télescopes permet de dépasser la **limite de diffraction classique**.
        
        **Avantages:**
        - Résolution angulaire sub-Rayleigh
        - Sensibilité √N → N (N télescopes)
        - Corrélations EPR longue distance
        - Cryptographie quantique sécurisée
        
        **État intriqué:** |Ψ⟩ = (|0⟩₁|1⟩₂ + |1⟩₁|0⟩₂)/√2
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_telescopes = st.slider("Nombre Télescopes Intriqués", 2, 10, 4)
            baseline_km = st.slider("Ligne de Base (km)", 100, 10000, 1000)
            wavelength_nm = st.number_input("Longueur d'onde (nm)", 100, 2000, 550)
        
        with col2:
            # Résolution classique (Rayleigh)
            wavelength_m = wavelength_nm * 1e-9
            resolution_classical_rad = wavelength_m / (baseline_km * 1000)
            resolution_classical_arcsec = resolution_classical_rad * 206265
            
            # Amélioration quantique
            enhancement_factor = np.sqrt(n_telescopes)  # Simplifié
            resolution_quantum_arcsec = resolution_classical_arcsec / enhancement_factor
            
            st.metric("Résolution Classique", f"{resolution_classical_arcsec:.6f}\"")
            st.metric("Résolution Quantique", f"{resolution_quantum_arcsec:.6f}\"")
            st.metric("Gain Quantique", f"{enhancement_factor:.2f}×")
            
            if enhancement_factor > 2:
                st.success("✅ Amélioration significative!")
        
        if st.button("🔬 Établir Intrication Télescopes", type="primary"):
            with st.spinner("Génération paires EPR et distribution..."):
                import time
                
                progress = st.progress(0)
                status = st.empty()
                
                steps = [
                    "Génération paires EPR",
                    "Distribution photons intriqués",
                    "Synchronisation horloges atomiques",
                    "Mesures conjointes",
                    "Vérification Bell inequality"
                ]
                
                for i, step in enumerate(steps):
                    status.write(f"**{step}...**")
                    progress.progress((i+1)/len(steps))
                    time.sleep(0.8)
                
                st.success("✅ État intriqué établi!")
                
                # Matrice densité
                size = min(8, 2**n_telescopes)
                
                # Créer état maximalement intriqué (simplifié)
                rho = np.zeros((size, size), dtype=complex)
                for i in range(size):
                    for j in range(size):
                        if i == j:
                            rho[i, j] = 1/size
                        else:
                            phase = np.random.uniform(0, 2*np.pi)
                            rho[i, j] = (1/size) * np.exp(1j * phase) * np.random.uniform(0, 0.3)
                
                # Hermitianiser
                rho = (rho + rho.conj().T) / 2
                rho = rho / np.trace(rho)
                
                # Visualiser
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Matrice Densité |ρ|", "Phase arg(ρ)")
                )
                
                fig.add_trace(go.Heatmap(
                    z=np.abs(rho),
                    colorscale='Viridis',
                    colorbar=dict(x=0.45, title="|ρ|")
                ), row=1, col=1)
                
                fig.add_trace(go.Heatmap(
                    z=np.angle(rho),
                    colorscale='HSV',
                    colorbar=dict(x=1.0, title="arg(ρ)")
                ), row=1, col=2)
                
                fig.update_layout(
                    title="État Quantique Intriqué",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Métriques quantiques
                st.write("### 📊 Métriques Intrication")
                
                # Pureté
                purity = np.real(np.trace(rho @ rho))
                
                # Entropie von Neumann
                eigenvalues = np.linalg.eigvalsh(rho)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                entropy_vn = -np.sum(eigenvalues * np.log2(eigenvalues))
                
                # Concurrence (approximation)
                concurrence = np.random.uniform(0.7, 0.95)  # Simulé pour état intriqué
                
                # Fidélité avec état cible
                fidelity = np.random.uniform(0.90, 0.98)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Pureté Tr(ρ²)", f"{purity:.4f}")
                    if purity > 0.9:
                        st.success("État pur")
                    else:
                        st.info("État mixte")
                
                with col2:
                    st.metric("Entropie vN", f"{entropy_vn:.3f} bits")
                
                with col3:
                    st.metric("Concurrence", f"{concurrence:.4f}")
                    if concurrence > 0.5:
                        st.success("✅ Fortement intriqué")
                
                with col4:
                    st.metric("Fidélité", f"{fidelity:.4f}")
                
                # Test Bell
                st.write("### 🔔 Violation Inégalité de Bell")
                
                # Paramètre CHSH
                S_bell = 2 * np.sqrt(2) * concurrence
                
                st.metric("Paramètre S (CHSH)", f"{S_bell:.3f}")
                
                if S_bell > 2:
                    st.success(f"✅ Violation Bell! S = {S_bell:.3f} > 2 (classique)")
                    st.balloons()
                    st.write("🎉 **Non-localité quantique confirmée!**")
                else:
                    st.warning("Pas de violation détectée")
                
                # Application observation
                st.write("### 🔭 Application: Super-résolution")
                
                st.info(f"""
                **Avec intrication quantique:**
                - Résolution: {resolution_quantum_arcsec:.6f}\"
                - Équivalent à télescope unique de {baseline_km * enhancement_factor:.0f} km
                - Permet d'imager exoplanètes directement!
                """)
    
    with tab2:
        st.subheader("💫 Téléportation Quantique de Données")
        
        st.write("""
        **Protocole Téléportation Quantique:**
        
        1. **Partage paire EPR** entre émetteur (Alice) et récepteur (Bob)
        2. Alice effectue **mesure de Bell** sur qubit à téléporter + son EPR
        3. **Communication classique** du résultat (2 bits)
        4. Bob applique **opération unitaire** selon résultat
        5. État téléporté reconstruit chez Bob!
        
        **Applications:**
        - Communication sécurisée entre observatoires
        - Distribution de clés quantiques (QKD)
        - Réseau quantique mondial
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_size_qubits = st.slider("Taille Données (qubits)", 1, 100, 10)
            distance_km = st.slider("Distance Alice-Bob (km)", 100, 100000, 10000)
        
        with col2:
            # Fidélité décroît avec distance (pertes)
            loss_db_per_km = 0.2  # Fibre optique
            total_loss_db = loss_db_per_km * distance_km / 1000
            transmission = 10**(-total_loss_db / 10)
            
            fidelity_teleport = 0.99 * transmission**(1/4)  # Simplifié
            
            st.metric("Transmission Photons", f"{transmission:.2%}")
            st.metric("Fidélité Téléportation", f"{fidelity_teleport:.4f}")
            
            if fidelity_teleport > 0.95:
                st.success("✅ Haute fidélité")
            elif fidelity_teleport > 0.85:
                st.warning("⚠️ Fidélité acceptable")
            else:
                st.error("❌ Répéteurs quantiques nécessaires")
        
        if st.button("💫 Téléporter Données Quantiques"):
            with st.spinner("Téléportation en cours..."):
                import time
                
                progress = st.progress(0)
                status_text = st.empty()
                
                steps = [
                    "🔗 Partage paires EPR",
                    "🔬 Mesures de Bell (Alice)",
                    "📡 Communication classique",
                    "⚛️ Opérations unitaires (Bob)",
                    "✅ Vérification état téléporté"
                ]
                
                for i, step in enumerate(steps):
                    status_text.write(f"**{step}**")
                    progress.progress((i+1)/len(steps))
                    time.sleep(0.7)
                
                st.success("✅ Téléportation complétée!")
                
                # Résultats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Qubits Téléportés", data_size_qubits)
                
                with col2:
                    st.metric("Fidélité Moyenne", f"{fidelity_teleport:.4f}")
                
                with col3:
                    # Latence communication classique
                    latency_ms = distance_km / 299.792  # Vitesse lumière
                    st.metric("Latence", f"{latency_ms:.2f} ms")
                
                # Comparaison classique vs quantique
                st.write("### 📊 Comparaison Transmission")
                
                comparison_data = {
                    'Méthode': ['Classique (copie)', 'Quantique (téléportation)'],
                    'Sécurité': ['❌ Peut être intercepté', '✅ Inviolable (no-cloning)'],
                    'Fidélité': ['~99.9%', f'{fidelity_teleport*100:.2f}%'],
                    'Latence': [f'{latency_ms:.2f} ms', f'{latency_ms:.2f} ms'],
                    'Bits Classiques': ['Tous', '2 bits/qubit']
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True)
                
                st.info("""
                🔐 **Avantage quantique:** 
                Impossible de cloner ou intercepter l'état quantique sans perturber la téléportation!
                """)
    
    with tab3:
        st.subheader("🎲 Quantum Computing - Simulations Astrophysiques")
        
        st.write("""
        **Applications Ordinateur Quantique:**
        
        1. **Simulation N-corps:** Évolution amas galaxies (O(N²) → O(log N))
        2. **Optimisation observations:** Ordonnancement télescopes (QAOA)
        3. **Machine Learning:** Classification galaxies (QML)
        4. **Chimie quantique:** Molécules interstellaires
        5. **Cryptographie:** Sécurisation données
        """)
        
        algorithm = st.selectbox("Algorithme Quantique",
            ["Grover (Recherche BD)", "Shor (Factorisation)", "VQE (Chimie Quantique)", 
             "QAOA (Optimisation)", "QML (Machine Learning)"])
        
        n_qubits = st.slider("Nombre de Qubits", 4, 100, 20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            noise_level = st.slider("Niveau Bruit (%)", 0.0, 10.0, 1.0, 0.1)
            st.info(f"Technologie actuelle: ~{n_qubits} qubits logiques")
        
        with col2:
            gate_fidelity = 1 - noise_level/100
            st.metric("Fidélité Portes", f"{gate_fidelity:.4f}")
            
            # Nombre portes avant décohérence
            T1_us = 100  # Temps relaxation
            T2_us = 50   # Temps déphasage
            gate_time_ns = 50
            
            max_gates = int(T2_us * 1000 / gate_time_ns)
            st.metric("Portes Max", max_gates)
        
        if st.button("🎲 Exécuter Algorithme Quantique", type="primary"):
            with st.spinner(f"Exécution {algorithm}..."):
                import time
                time.sleep(2.5)
                
                if "Grover" in algorithm:
                    # Recherche dans base de données
                    db_size = 2**n_qubits
                    classical_queries = db_size // 2
                    quantum_queries = int(np.pi/4 * np.sqrt(db_size))
                    speedup = classical_queries / quantum_queries
                    
                    st.success(f"✅ Élément trouvé!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Taille BD", f"{db_size:,}")
                    with col2:
                        st.metric("Requêtes Quantiques", quantum_queries)
                    with col3:
                        st.metric("Speedup", f"{speedup:.1f}×")
                    
                    st.info(f"💡 **Classique:** {classical_queries} requêtes vs **Quantique:** {quantum_queries}")
                
                elif "Shor" in algorithm:
                    # Factorisation
                    number_bits = n_qubits // 2
                    number_to_factor = 2**number_bits - 1
                    
                    # Temps classique (sous-exponentiel)
                    time_classical_years = np.exp(1.9 * number_bits**(1/3) * (np.log(number_bits))**(2/3)) / (3.15e7 * 1e9)
                    
                    # Temps quantique (polynomial)
                    time_quantum_s = number_bits**2 * gate_time_ns * 1e-9
                    
                    st.success(f"✅ Facteurs trouvés!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Nombre à factoriser", f"{number_to_factor:,}")
                        st.metric("Bits", number_bits)
                    
                    with col2:
                        st.metric("Temps Quantique", f"{time_quantum_s:.3f} s")
                        st.metric("Temps Classique", f"{time_classical_years:.2e} ans")
                    
                    st.balloons()
                    st.info("🔐 **Impact:** Casse RSA-2048 en quelques heures!")
                
                elif "VQE" in algorithm:
                    # Variational Quantum Eigensolver - Chimie
                    molecule = "H₂O (eau)" if np.random.random() > 0.5 else "CH₄ (méthane)"
                    
                    # Énergie fondamentale (simulée)
                    energy_hartree = np.random.uniform(-100, -50)
                    energy_kcal_mol = energy_hartree * 627.5
                    
                    # Convergence
                    iterations = []
                    energies = []
                    
                    E_target = energy_hartree
                    E_current = E_target + 10
                    
                    for i in range(50):
                        E_current = E_current - (E_current - E_target) * 0.15 + np.random.normal(0, 0.1)
                        iterations.append(i)
                        energies.append(E_current)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=iterations,
                        y=energies,
                        mode='lines+markers',
                        line=dict(color='#667eea', width=2),
                        name='Énergie'
                    ))
                    
                    fig.add_hline(y=E_target, line_dash="dash", line_color="green",
                                 annotation_text="Énergie exacte")
                    
                    fig.update_layout(
                        title=f"Convergence VQE - Molécule {molecule}",
                        xaxis_title="Itération",
                        yaxis_title="Énergie (Hartree)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Molécule", molecule)
                    with col2:
                        st.metric("Énergie Fond.", f"{energy_hartree:.2f} Ha")
                    with col3:
                        st.metric("Précision", "±0.01 Ha")
                    
                    st.success("✅ État fondamental calculé!")
                
                # Circuit quantique
                st.write("### 🔧 Architecture Circuit")
                
                circuit_depth = np.random.randint(20, 100)
                n_cnot = circuit_depth * n_qubits // 3
                n_single_qubit = circuit_depth * n_qubits
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Qubits", n_qubits)
                with col2:
                    st.metric("Profondeur", circuit_depth)
                with col3:
                    st.metric("CNOT Gates", n_cnot)
                with col4:
                    error_total = 1 - gate_fidelity**n_single_qubit
                    st.metric("Erreur Totale", f"{error_total*100:.2f}%")
                
                # Correction d'erreurs
                if error_total > 0.01:
                    st.warning("⚠️ Correction d'erreurs quantiques nécessaire")
                    
                    # Qubits physiques pour 1 qubit logique (code de surface)
                    physical_per_logical = int((error_total * 100)**2)
                    total_physical_qubits = n_qubits * physical_per_logical
                    
                    st.info(f"📊 Code de surface: {physical_per_logical} qubits physiques / qubit logique")
                    st.info(f"🔢 Total nécessaire: {total_physical_qubits} qubits physiques")
    
    with tab4:
        st.subheader("🌌 Cosmologie Quantique")
        
        st.write("""
        **Effets Quantiques en Cosmologie:**
        
        - **Fluctuations quantiques primordiales** → Structures à grande échelle
        - **Inflation quantique** → Horizon et platitude
        - **Intrication cosmologique** → Corrélations CMB
        - **Gravité quantique** → Singularité Big Bang
        - **Paradoxe information trous noirs**
        """)
        
        topic = st.selectbox("Sujet d'Analyse",
            ["Fluctuations Quantiques CMB", "Inflation Chaotique", "Intrication Cosmologique", 
             "Information Trous Noirs"])
        
        if topic == "Fluctuations Quantiques CMB":
            st.write("### 📡 Spectre Puissance Angulaire CMB")
            
            if st.button("🌌 Analyser Fluctuations Quantiques"):
                with st.spinner("Analyse fond diffus cosmologique..."):

                    import time
                    time.sleep(2)
                    
                    # Spectre puissance angulaire
                    l = np.logspace(1, 3.5, 200)
                    
                    # TT spectrum (température) - Forme théorique
                    C_l = 6000 * (l / 220)**(-1) * np.exp(-l/2000)
                    
                    # Pics acoustiques (Sakharov oscillations)
                    acoustic_peaks = [220, 540, 810, 1120, 1450]
                    for peak_l in acoustic_peaks:
                        C_l += 1500 * np.exp(-(l-peak_l)**2 / (50**2))
                    
                    # Queue Silk damping (diffusion photons)
                    damping_tail = np.exp(-((l - 1000)/500)**2)
                    C_l = C_l * damping_tail
                    
                    # Graphique
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=l,
                        y=l*(l+1)*C_l/(2*np.pi),
                        mode='lines',
                        line=dict(color='#667eea', width=3),
                        name='Données'
                    ))
                    
                    # Marquer pics
                    for i, peak in enumerate(acoustic_peaks[:3]):
                        fig.add_vline(x=peak, line_dash="dash", line_color="red",
                                     annotation_text=f"Pic {i+1}")
                    
                    fig.update_layout(
                        title="Spectre Puissance Angulaire CMB",
                        xaxis_title="Multipole l",
                        yaxis_title="l(l+1)C_l / 2π (μK²)",
                        template="plotly_dark",
                        height=500
                    )
                    
                    fig.update_xaxes(type="log")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Paramètres cosmologiques extraits
                    st.write("### 📊 Paramètres Cosmologiques")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("H₀", "67.4 ± 0.5 km/s/Mpc")
                        st.metric("Ω_m", "0.315 ± 0.007")
                        st.metric("Ω_b", "0.049 ± 0.001")
                    
                    with col2:
                        st.metric("Ω_Λ", "0.685 ± 0.007")
                        st.metric("τ (réionisation)", "0.054 ± 0.007")
                        st.metric("n_s", "0.965 ± 0.004")
                    
                    with col3:
                        st.metric("σ_8", "0.811 ± 0.006")
                        st.metric("Âge Univers", "13.80 ± 0.02 Gyr")
                        st.metric("z_reionisation", "7.7 ± 0.7")
                    
                    st.success("✅ Fluctuations quantiques primordiales confirmées!")
                    st.info("""
                    🎯 **Résultat:** Les anisotropies du CMB proviennent de fluctuations 
                    quantiques du champ inflatonique, amplifiées par l'inflation cosmique!
                    """)
        
        elif topic == "Inflation Chaotique":
            st.write("### 🌀 Modèle Inflation Chaotique")
            
            phi_initial = st.slider("Champ Inflaton Initial (M_Pl)", 0.1, 20.0, 15.0, 0.5)
            
            if st.button("🌀 Simuler Inflation"):
                with st.spinner("Résolution équations Friedmann..."):
                    import time
                    time.sleep(2)
                    
                    # Évolution champ inflaton
                    N_efolds = np.linspace(0, 60, 500)
                    
                    # Potentiel chaotique: V = (1/2) m² φ²
                    phi = phi_initial * np.exp(-N_efolds / 60)
                    
                    # Paramètre slow-roll
                    epsilon = (phi / phi_initial)**2 / 2
                    
                    # Facteur échelle
                    a = np.exp(N_efolds)
                    
                    # Graphiques
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Champ Inflaton φ", "Facteur Échelle a(t)", 
                                      "Paramètre Slow-Roll ε", "Spectre Puissance")
                    )
                    
                    # 1. Inflaton
                    fig.add_trace(go.Scatter(
                        x=N_efolds, y=phi,
                        mode='lines',
                        line=dict(color='#667eea', width=3)
                    ), row=1, col=1)
                    
                    fig.update_xaxes(title_text="N (e-folds)", row=1, col=1)
                    fig.update_yaxes(title_text="φ (M_Pl)", row=1, col=1)
                    
                    # 2. Facteur échelle
                    fig.add_trace(go.Scatter(
                        x=N_efolds, y=a,
                        mode='lines',
                        line=dict(color='#4ECDC4', width=3)
                    ), row=1, col=2)
                    
                    fig.update_xaxes(title_text="N (e-folds)", row=1, col=2)
                    fig.update_yaxes(title_text="a(t)", type="log", row=1, col=2)
                    
                    # 3. Slow-roll
                    fig.add_trace(go.Scatter(
                        x=N_efolds, y=epsilon,
                        mode='lines',
                        line=dict(color='#FF6B6B', width=3)
                    ), row=2, col=1)
                    
                    fig.add_hline(y=1, line_dash="dash", line_color="red",
                                 annotation_text="Fin inflation", row=2, col=1)
                    
                    fig.update_xaxes(title_text="N (e-folds)", row=2, col=1)
                    fig.update_yaxes(title_text="ε", type="log", row=2, col=1)
                    
                    # 4. Spectre puissance perturbations scalaires
                    k = np.logspace(-4, 0, 100)  # Modes k
                    
                    # Spectre quasi-invariant d'échelle
                    n_s = 0.965  # Indice spectral
                    A_s = 2.1e-9  # Amplitude
                    P_R = A_s * (k / 0.05)**(n_s - 1)
                    
                    fig.add_trace(go.Scatter(
                        x=k, y=P_R,
                        mode='lines',
                        line=dict(color='#FFEAA7', width=3)
                    ), row=2, col=2)
                    
                    fig.update_xaxes(title_text="k (Mpc⁻¹)", type="log", row=2, col=2)
                    fig.update_yaxes(title_text="P_R(k)", type="log", row=2, col=2)
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=800,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Résultats
                    N_total = N_efolds[-1]
                    expansion_factor = np.exp(N_total)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("N e-folds", f"{N_total:.0f}")
                        st.metric("Expansion", f"10^{int(np.log10(expansion_factor))}")
                    
                    with col2:
                        st.metric("Indice Spectral n_s", f"{n_s:.3f}")
                        st.metric("Rapport Tenseur/Scalaire r", "< 0.01")
                    
                    with col3:
                        duration_s = 1e-35  # Durée typique
                        st.metric("Durée", f"{duration_s:.2e} s")
                        st.metric("Énergie", "~10¹⁶ GeV")
                    
                    st.success("✅ Inflation résout problèmes horizon, platitude et monopôles!")
        
        elif topic == "Information Trous Noirs":
            st.write("### ⚫ Paradoxe de l'Information")
            
            st.write("""
            **Paradoxe de Hawking:**
            - TN émet radiation thermique (Hawking)
            - Information tombée semble perdue
            - Viole unitarité mécanique quantique!
            
            **Solutions proposées:**
            - Information encodée dans radiation
            - Correspondance AdS/CFT
            - Fuzzballs / Firewalls
            - Intrication entre TN et radiation
            """)
            
            if st.button("⚫ Analyser Paradoxe Information"):
                with st.spinner("Calcul entropie Bekenstein-Hawking..."):
                    import time
                    time.sleep(2)
                    
                    # Masse trou noir
                    M_bh_msun = st.slider("Masse TN (M☉)", 1.0, 100.0, 10.0)
                    
                    # Constantes
                    G = 6.67430e-11
                    c = 299792458
                    hbar = 1.054571817e-34
                    k_B = 1.380649e-23
                    M_sun = 1.989e30
                    
                    # Rayon Schwarzschild
                    Rs = 2 * G * M_bh_msun * M_sun / c**2
                    
                    # Entropie Bekenstein-Hawking
                    A_horizon = 4 * np.pi * Rs**2
                    S_BH = (k_B * c**3 * A_horizon) / (4 * G * hbar)
                    S_BH_dimensionless = S_BH / k_B
                    
                    # Température Hawking
                    T_H = hbar * c**3 / (8 * np.pi * G * M_bh_msun * M_sun * k_B)
                    
                    # Évaporation
                    t = np.linspace(0, 1, 100)
                    M_t = M_bh_msun * (1 - t)**(1/3)  # Masse décroissante
                    S_t = S_BH_dimensionless * (M_t / M_bh_msun)**2  # Entropie
                    
                    # Entropie radiation
                    S_rad = S_BH_dimensionless * (1 - (M_t / M_bh_msun)**2)
                    
                    # Entropie totale (devrait être conservée?)
                    S_total = S_t + S_rad
                    
                    # Page curve
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=t * 100,
                        y=S_t / S_BH_dimensionless,
                        mode='lines',
                        line=dict(color='black', width=3),
                        name='S_TN'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=t * 100,
                        y=S_rad / S_BH_dimensionless,
                        mode='lines',
                        line=dict(color='orange', width=3),
                        name='S_radiation'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=t * 100,
                        y=S_total / S_BH_dimensionless,
                        mode='lines',
                        line=dict(color='green', width=3, dash='dash'),
                        name='S_totale'
                    ))
                    
                    # Page time (milieu évaporation)
                    fig.add_vline(x=50, line_dash="dot", line_color="red",
                                 annotation_text="Page time")
                    
                    fig.update_layout(
                        title="Page Curve - Évolution Entropie",
                        xaxis_title="Temps Évaporation (%)",
                        yaxis_title="Entropie (S/S_BH)",
                        template="plotly_dark",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("S_BH", f"{S_BH_dimensionless:.2e} k_B")
                        st.metric("Aire Horizon", f"{A_horizon:.2e} m²")
                    
                    with col2:
                        st.metric("T_Hawking", f"{T_H:.2e} K")
                        
                        # Nombre bits d'information
                        n_bits = S_BH_dimensionless / np.log(2)
                        st.metric("Information", f"{n_bits:.2e} bits")
                    
                    with col3:
                        # Temps évaporation
                        t_evap = 2.1e67 * M_bh_msun**3
                        st.metric("τ_évaporation", f"{t_evap:.2e} ans")
                    
                    st.write("### 🔬 Résolution Paradoxe")
                    
                    st.info("""
                    **Page Curve:** L'entropie de la radiation commence à décroître après 
                    le "Page time" (~50% évaporation), suggérant que l'information est 
                    progressivement transférée du TN vers la radiation.
                    
                    **Mécanisme:** Intrication quantique entre l'intérieur du TN et 
                    la radiation émise préserverait l'unitarité.
                    """)
                    
                    st.success("✅ Information quantique préservée via intrication!")           

# ==================== PAGE: GALAXIES ====================
elif page == "🌌 Galaxies":
    st.header("🌌 Étude des Galaxies")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Classification", "🌀 Morphologie", "📈 Redshift", "🔍 Amas"])
    
    with tab1:
        st.subheader("📊 Classification Galaxies - Séquence de Hubble")
        
        st.write("""
        **Séquence de Hubble:**
        - **Elliptiques (E0-E7):** Pas de structure spirale, classées par aplatissement
        - **Lenticulaires (S0):** Disque sans bras spiraux
        - **Spirales (Sa, Sb, Sc):** Bras spiraux, bulbe central décroissant
        - **Spirales barrées (SBa, SBb, SBc):** Barre centrale
        - **Irrégulières (Irr):** Pas de structure régulière
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_galaxies = st.number_input("Nombre Galaxies à Générer", 10, 500, 50)
            redshift_max = st.slider("Redshift Maximum", 0.1, 5.0, 1.0)
        
        with col2:
            mass_range = st.slider("Plage Masse (log M☉)", 8.0, 13.0, (9.0, 12.0))
        
        if st.button("🌌 Générer Échantillon Galaxies", type="primary"):
            with st.spinner("Génération catalogue..."):
                import time
                time.sleep(1)
                
                # Types et propriétés
                galaxy_types = np.random.choice(['E0', 'E3', 'E7', 'S0', 'Sa', 'Sb', 'Sc', 'SBa', 'SBb', 'SBc', 'Irr'], n_galaxies)
                magnitudes = np.random.uniform(12, 22, n_galaxies)
                redshifts = np.random.exponential(redshift_max/3, n_galaxies)
                redshifts = np.clip(redshifts, 0.01, redshift_max)
                masses = 10**(np.random.uniform(mass_range[0], mass_range[1], n_galaxies))
                
                # Distance cosmologique (Mpc)
                H0 = 70  # km/s/Mpc
                c = 299792.458  # km/s
                distances = c * redshifts / H0
                
                galaxy_data = []
                for i in range(n_galaxies):
                    # Taux formation stellaire (SFR)
                    if galaxy_types[i].startswith('E'):
                        sfr = np.random.uniform(0, 1)  # Elliptiques: peu de formation
                    elif galaxy_types[i].startswith('S'):
                        sfr = np.random.uniform(1, 50)  # Spirales: formation active
                    else:
                        sfr = np.random.uniform(5, 100)  # Irrégulières: sursauts
                    
                    galaxy = {
                        'id': f'GAL_{i+1:04d}',
                        'type': galaxy_types[i],
                        'magnitude': magnitudes[i],
                        'redshift': redshifts[i],
                        'distance_Mpc': distances[i],
                        'mass_Msun': masses[i],
                        'sfr_Msun_per_year': sfr,
                        'metallicity': np.random.uniform(0.5, 2.0),  # Z/Z_sol
                        'detected': datetime.now().isoformat()
                    }
                    galaxy_data.append(galaxy)
                    st.session_state.telescope_lab['galaxy_catalog'].append(galaxy)
                
                df = pd.DataFrame(galaxy_data)
                
                # Afficher tableau
                display_df = df[['id', 'type', 'magnitude', 'redshift', 'distance_Mpc']].copy()
                display_df['distance_Mpc'] = display_df['distance_Mpc'].round(1)
                display_df['magnitude'] = display_df['magnitude'].round(2)
                display_df['redshift'] = display_df['redshift'].round(3)
                
                st.dataframe(display_df, use_container_width=True)
                
                st.success(f"✅ {n_galaxies} galaxies cataloguées!")
                log_event(f"{n_galaxies} galaxies ajoutées au catalogue", "SUCCESS")
                
                # Statistiques
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Elliptiques", sum(1 for g in galaxy_types if g.startswith('E')))
                with col2:
                    st.metric("Spirales", sum(1 for g in galaxy_types if g.startswith('S') and not g.startswith('SB')))
                with col3:
                    st.metric("Barrées", sum(1 for g in galaxy_types if g.startswith('SB')))
                with col4:
                    st.metric("Irrégulières", sum(1 for g in galaxy_types if g == 'Irr'))
    
    with tab2:
        st.subheader("🌀 Analyse Morphologique - Paramètres de Sérsic")
        
        st.write("""
        **Profil de Sérsic:** I(r) = I_e × exp(-b_n[(r/r_e)^(1/n) - 1])
        - n=0.5: Disque exponentiel
        - n=1: Profil exponentiel
        - n=4: Profil de Vaucouleurs (elliptiques)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            morphology_type = st.selectbox("Type Morphologique",
                ["Elliptique (E)", "Spirale (S)", "Spirale Barrée (SB)", "Irrégulière (Irr)", "Lenticulaire (S0)"])
            
            bulge_to_disk = st.slider("Rapport Bulbe/Disque", 0.0, 1.0, 0.3, 0.05)
            bar_strength = st.slider("Force Barre (si barrée)", 0.0, 1.0, 0.5, 0.05)
            sersic_index = st.slider("Indice Sérsic (n)", 0.5, 8.0, 4.0, 0.5)
        
        with col2:
            spiral_arms = st.slider("Nombre Bras Spiraux", 0, 8, 2)
            inclination = st.slider("Inclinaison (deg)", 0, 90, 45, 5)
            pitch_angle = st.slider("Angle Pitch Spirales (deg)", 5, 45, 15, 5)
            
            effective_radius = st.slider("Rayon Effectif (kpc)", 1.0, 50.0, 10.0, 1.0)
        
        if st.button("🌀 Simuler Morphologie Galaxie"):
            with st.spinner("Génération modèle morphologique..."):
                import time
                time.sleep(1)
                
                # Créer image morphologie simulée
                size = 512
                y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                
                # Profil de Sérsic pour le bulbe
                b_n = 2*sersic_index - 1/3  # Approximation
                r_e = effective_radius * 10  # pixels
                bulge = np.exp(-b_n * ((r/r_e)**(1/sersic_index) - 1))
                
                # Disque exponentiel
                scale_length = effective_radius * 15
                disk = np.exp(-r / scale_length) * (r > r_e/2)
                
                # Bras spiraux
                spiral = np.zeros_like(r)
                if spiral_arms > 0:
                    pitch_rad = pitch_angle * np.pi / 180
                    for i in range(spiral_arms):
                        angle_offset = 2 * np.pi * i / spiral_arms
                        # Spirale logarithmique
                        spiral_theta = np.log(r/20 + 1) / np.tan(pitch_rad) + angle_offset
                        spiral_pattern = np.cos(2 * (theta - spiral_theta))
                        spiral += np.maximum(0, spiral_pattern) * np.exp(-r/60)
                
                # Barre centrale
                bar = np.zeros_like(r)
                if bar_strength > 0:
                    bar_length = effective_radius * 20
                    bar_width = effective_radius * 5
                    bar_mask = (np.abs(x) < bar_length) & (np.abs(y) < bar_width)
                    bar[bar_mask] = bar_strength * np.exp(-r[bar_mask]/30)
                
                # Combiner composantes
                galaxy_image = bulge_to_disk * bulge + (1-bulge_to_disk) * disk + 0.5 * spiral + bar
                
                # Appliquer inclinaison (ellipse)
                inclination_factor = np.cos(inclination * np.pi / 180)
                y_stretched = y / (inclination_factor + 0.1)
                r_inclined = np.sqrt(x**2 + y_stretched**2)
                
                # Reappliquer profil avec inclinaison
                if inclination > 30:
                    galaxy_image = np.exp(-r_inclined / scale_length)
                
                # Ajouter bruit
                galaxy_image += np.random.normal(0, 0.01, galaxy_image.shape)
                galaxy_image = np.clip(galaxy_image, 0, None)
                
                # Normaliser
                galaxy_image = galaxy_image / galaxy_image.max()
                
                fig = go.Figure(data=go.Heatmap(
                    z=galaxy_image,
                    colorscale='Viridis',
                    showscale=False
                ))
                
                fig.update_layout(
                    title=f"Morphologie: {morphology_type} | n={sersic_index} | i={inclination}°",
                    xaxis=dict(showticklabels=False, showgrid=False),
                    yaxis=dict(showticklabels=False, showgrid=False),
                    template="plotly_dark",
                    height=600,
                    width=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Paramètres physiques
                st.write("### 📊 Paramètres Physiques")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rayon Effectif", f"{effective_radius:.1f} kpc")
                    st.metric("Indice Sérsic", f"{sersic_index:.1f}")
                
                with col2:
                    st.metric("B/D Ratio", f"{bulge_to_disk:.2f}")
                    st.metric("Inclinaison", f"{inclination}°")
                
                with col3:
                    concentration = 5  # R80/R20
                    st.metric("Concentration", f"{concentration:.1f}")
                    asymmetry = np.random.uniform(0.05, 0.3)
                    st.metric("Asymétrie", f"{asymmetry:.2f}")
                
                st.success("✅ Morphologie générée!")
    
    with tab3:
        st.subheader("📈 Analyse Redshift & Cosmologie")
        
        st.write("""
        **Relations Cosmologiques:**
        - **Loi de Hubble:** v = H₀ × d
        - **Redshift:** z = Δλ/λ = v/c
        - **Distance luminosité:** d_L = d_C × (1+z)
        - **Module de distance:** m - M = 5 log(d_L/10pc)
        """)
        
        if st.session_state.telescope_lab['galaxy_catalog']:
            st.write(f"### 📊 {len(st.session_state.telescope_lab['galaxy_catalog'])} Galaxies dans le Catalogue")
            
            redshifts = [g['redshift'] for g in st.session_state.telescope_lab['galaxy_catalog']]
            distances = [g['distance_Mpc'] for g in st.session_state.telescope_lab['galaxy_catalog']]
            magnitudes = [g['magnitude'] for g in st.session_state.telescope_lab['galaxy_catalog']]
            
            # Diagramme Hubble
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Diagramme de Hubble", "Distribution Redshift")
            )
            
            # Hubble diagram
            fig.add_trace(go.Scatter(
                x=distances,
                y=redshifts,
                mode='markers',
                marker=dict(size=8, color=magnitudes, colorscale='Plasma', showscale=True, colorbar=dict(x=0.45)),
                text=[f"z={z:.3f}<br>m={m:.1f}" for z, m in zip(redshifts, magnitudes)],
                hovertemplate='%{text}<br>Distance: %{x:.1f} Mpc<extra></extra>',
                showlegend=False
            ), row=1, col=1)
            
            # Ligne Hubble théorique
            d_fit = np.linspace(0, max(distances), 100)
            H0 = 70  # km/s/Mpc
            z_fit = H0 * d_fit / 299792
            
            fig.add_trace(go.Scatter(
                x=d_fit, y=z_fit,
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='H₀=70 km/s/Mpc',
                showlegend=True
            ), row=1, col=1)
            
            # Histogramme redshift
            fig.add_trace(go.Histogram(
                x=redshifts,
                nbinsx=30,
                marker_color='#667eea',
                showlegend=False
            ), row=1, col=2)
            
            fig.update_xaxes(title_text="Distance (Mpc)", row=1, col=1)
            fig.update_yaxes(title_text="Redshift z", row=1, col=1)
            fig.update_xaxes(title_text="Redshift z", row=1, col=2)
            fig.update_yaxes(title_text="Nombre", row=1, col=2)
            
            fig.update_layout(
                template="plotly_dark",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculer H0 observé
            if len(distances) > 5:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(distances, redshifts)
                H0_measured = slope * 299792
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("H₀ Mesuré", f"{H0_measured:.1f} km/s/Mpc")
                with col2:
                    st.metric("Corrélation R²", f"{r_value**2:.4f}")
                with col3:
                    st.metric("z moyen", f"{np.mean(redshifts):.3f}")
                with col4:
                    st.metric("z max", f"{np.max(redshifts):.3f}")
                
                # Âge de l'univers à différents z
                st.write("### ⏰ Âge de l'Univers")
                
                z_samples = [0, 0.5, 1.0, 2.0, 3.0]
                ages = []
                
                for z in z_samples:
                    # Formule simplifiée (univers plat)
                    Omega_m = 0.3
                    Omega_lambda = 0.7
                    H0_si = 70 * 1000 / (3.086e22)  # en s^-1
                    
                    # Temps lookback (approximation)
                    age_Gyr = 13.8 / (1 + z)**1.5  # Simplifié
                    ages.append(age_Gyr)
                
                df_age = pd.DataFrame({
                    'Redshift z': z_samples,
                    'Âge Univers (Gyr)': [f"{a:.2f}" for a in ages],
                    'Temps Lookback (Gyr)': [f"{13.8 - a:.2f}" for a in ages]
                })
                
                st.dataframe(df_age, use_container_width=True)
        else:
            st.info("Générez d'abord un catalogue de galaxies")
    
    with tab4:
        st.subheader("🔍 Détection Amas de Galaxies")
        
        st.write("""
        **Amas de Galaxies - Structures Cosmiques:**
        - **Masses:** 10¹⁴ - 10¹⁵ M☉
        - **Membres:** 50 - 1000 galaxies
        - **Gaz chaud:** T ~ 10⁷-10⁸ K (émission X)
        - **Matière noire:** ~85% de la masse
        - **Lentilles gravitationnelles:** Déformation d'images
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_method = st.selectbox("Méthode Détection",
                ["Clustering Spatial", "Émission X", "Effet Sunyaev-Zel'dovich", "Lentilles Faibles"])
            
            min_members = st.slider("Membres Minimum", 10, 100, 30)
        
        with col2:
            search_radius_mpc = st.slider("Rayon Recherche (Mpc)", 1.0, 20.0, 5.0, 0.5)
            mass_threshold = st.slider("Seuil Masse (10¹⁴ M☉)", 0.5, 10.0, 1.0, 0.5)
        
        if st.button("🔍 Rechercher Amas", type="primary"):
            with st.spinner("Analyse clustering spatial..."):
                import time
                time.sleep(2)
                
                n_clusters = np.random.randint(3, 10)
                
                st.success(f"✅ {n_clusters} amas de galaxies détectés!")
                
                # Détails des amas
                for i in range(n_clusters):
                    with st.expander(f"🌌 Amas #{i+1} - Abell {2000+i*100}"):
                        col1, col2, col3 = st.columns(3)
                        
                        n_members = np.random.randint(min_members, 500)
                        cluster_mass = np.random.uniform(mass_threshold, 10) * 1e14
                        cluster_z = np.random.uniform(0.1, 1.5)
                        
                        with col1:
                            st.metric("Galaxies Membres", n_members)
                            st.metric("Richesse (R)", np.random.randint(0, 3))
                        
                        with col2:
                            st.metric("Masse Totale", f"{cluster_mass:.2e} M☉")
                            st.metric("Masse M_200", f"{cluster_mass*0.8:.2e} M☉")
                        
                        with col3:
                            st.metric("Redshift", f"{cluster_z:.3f}")
                            st.metric("Distance", f"{cluster_z * 3000:.0f} Mpc")
                        
                        st.write("**Propriétés Dynamiques:**")
                        velocity_dispersion = np.random.randint(500, 1500)
                        st.write(f"• Dispersion vitesses: {velocity_dispersion} km/s")
                        
                        virial_radius = np.random.uniform(1.0, 3.0)
                        st.write(f"• Rayon virial: {virial_radius:.2f} Mpc")
                        
                        st.write("**Émission X:**")
                        x_luminosity = np.random.uniform(1e43, 1e45)
                        st.write(f"• Luminosité X: {x_luminosity:.2e} erg/s")
                        
                        gas_temp = np.random.uniform(5, 15)
                        st.write(f"• Température gaz: {gas_temp:.1f} keV (~{gas_temp*11.6e6:.1e} K)")
                        
                        # Visualisation carte amas
                        st.write("**Carte Amas:**")
                        
                        # Générer positions membres
                        r_members = np.random.exponential(virial_radius/3, n_members)
                        theta_members = np.random.uniform(0, 2*np.pi, n_members)
                        
                        x_members = r_members * np.cos(theta_members)
                        y_members = r_members * np.sin(theta_members)
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=x_members,
                            y=y_members,
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=np.random.uniform(15, 22, n_members),
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Magnitude")
                            ),
                            text=[f"Galaxy {j+1}" for j in range(n_members)],
                            hovertemplate='%{text}<extra></extra>'
                        ))
                        
                        # Cercle virial
                        theta_circle = np.linspace(0, 2*np.pi, 100)
                        fig.add_trace(go.Scatter(
                            x=virial_radius * np.cos(theta_circle),
                            y=virial_radius * np.sin(theta_circle),
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name='Rayon virial'
                        ))
                        
                        fig.update_layout(
                            title=f"Distribution Spatiale - Amas #{i+1}",
                            xaxis_title="ΔRA (Mpc)",
                            yaxis_title="ΔDec (Mpc)",
                            template="plotly_dark",
                            height=400,
                            showlegend=False
                        )
                        
                        fig.update_xaxes(scaleanchor="y", scaleratio=1)
                        fig.update_yaxes(scaleanchor="x", scaleratio=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                st.balloons()

# ==================== PAGE: TROUS NOIRS ====================
elif page == "⚫ Trous Noirs":
    st.header("⚫ Physique des Trous Noirs")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚫ Propriétés", "🌀 Disque Accrétion", "🔭 Détection", "📊 Catalogue"])
    
    with tab1:
        st.subheader("⚫ Calculs Relativité Générale")
        
        st.write("""
        **Types de Trous Noirs:**
        - **Stellaires:** 3-100 M☉ (effondrement étoiles massives)
        - **Intermédiaires:** 10²-10⁵ M☉ (amas stellaires)
        - **Supermassifs:** 10⁶-10¹⁰ M☉ (centres galactiques)
        - **Primordiaux:** Hypothétiques (Big Bang)
        
        **Métrique de Kerr** (trou noir en rotation):
        - **Paramètre de spin:** a = J/(Mc) avec 0 ≤ a ≤ M
        - **ISCO:** Rayon orbite stable interne
        - **Ergosphère:** Région où rotation obligatoire
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            bh_mass_msun = st.number_input("Masse Trou Noir (M☉)", 1.0, 1e10, 1e6, format="%.2e")
            spin_param = st.slider("Paramètre Spin (a/M)", 0.0, 0.998, 0.7, 0.001)
            
            st.info(f"""
            **Spin:**
            - 0 = Schwarzschild (non rotatif)
            - 0.998 = Rotation maximale (Kerr extrême)
            """)
        
        with col2:
            # Calculer rayon Schwarzschild
            G = 6.67430e-11
            c = 299792458
            M_sun_kg = 1.989e30
            
            rs_m = 2 * G * bh_mass_msun * M_sun_kg / c**2
            rs_km = rs_m / 1000
            
            st.metric("Rayon Schwarzschild (Rs)", 
                     f"{rs_km:.2f} km" if rs_km < 1e6 else f"{rs_km/1.496e8:.3f} AU")
            
            # ISCO (simplifié pour Kerr)
            if spin_param < 0.01:
                r_isco = 6  # En unités de M
            else:
                # Formule approx pour prograde
                Z1 = 1 + (1 - spin_param**2)**(1/3) * ((1+spin_param)**(1/3) + (1-spin_param)**(1/3))
                Z2 = np.sqrt(3*spin_param**2 + Z1**2)
                r_isco = 3 + Z2 - np.sqrt((3-Z1)*(3+Z1+2*Z2))
            
            r_isco_km = r_isco * rs_km / 2
            st.metric("ISCO (Orbite Stable)", f"{r_isco:.2f} Rs = {r_isco_km:.2f} km")
            
            # Température Hawking
            T_hawking = 6.17e-8 / bh_mass_msun  # Kelvin
            st.metric("Température Hawking", f"{T_hawking:.2e} K")
        
        if st.button("⚫ Analyse Complète Trou Noir", type="primary"):
            with st.spinner("Calculs relativité générale..."):
                import time
                time.sleep(1)
                
                st.write("### 📊 Propriétés Physiques Détaillées")
                
                # Densité moyenne
                volume = (4/3) * np.pi * rs_m**3
                density_kg_m3 = bh_mass_msun * M_sun_kg / volume
                
                # Accélération surface
                g_surface = G * bh_mass_msun * M_sun_kg / rs_m**2
                
                # Temps évaporation Hawking
                t_evap_s = 2.1e67 * (bh_mass_msun)**3
                t_evap_years = t_evap_s / (365.25 * 24 * 3600)
                
                # Luminosité Hawking
                hbar = 1.054571817e-34
                k_B = 1.380649e-23
                L_hawking = (hbar * c**6) / (15360 * np.pi * G**2 * (bh_mass_msun * M_sun_kg)**2)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Géométrie:**")
                    st.write(f"• Rs: {rs_km:.2f} km")
                    st.write(f"• ISCO: {r_isco:.2f} Rs")
                    st.write(f"• Rayon photon: {3*rs_km/2:.2f} km")
                    
                    ergosphere_r = rs_km * (1 + np.sqrt(1 - spin_param**2))
                    st.write(f"• Ergosphère: {ergosphere_r:.2f} km")
                
                with col2:
                    st.write("**Dynamique:**")
                    st.write(f"• Densité: {density_kg_m3:.2e} kg/m³")
                    st.write(f"• g surface: {g_surface:.2e} m/s²")
                    
                    # Vitesse orbite ISCO
                    v_isco = c / np.sqrt(2 * r_isco)
                    st.write(f"• v @ ISCO: {v_isco/c:.3f}c")
                    
                    # Fréquence orbitale
                    f_isco = c**3 / (2 * np.pi * G * bh_mass_msun * M_sun_kg * r_isco)
                    st.write(f"• f @ ISCO: {f_isco:.2f} Hz")
                
                with col3:
                    st.write("**Évaporation Hawking:**")
                    st.write(f"• T Hawking: {T_hawking:.2e} K")
                    st.write(f"• L Hawking: {L_hawking:.2e} W")
                    st.write(f"• τ évaporation: {t_evap_years:.2e} ans")
                    
                    if t_evap_years > 1e60:
                        st.info("⏰ Évaporation >> âge univers")
                
                # Graphique métrique
                st.write("### 📈 Géométrie Espace-Temps")
                
                r_range = np.linspace(rs_km, 10*rs_km, 1000)
                
                # Potentiel effectif (simplifié)
                V_eff = -G * bh_mass_msun * M_sun_kg / (r_range * 1000) + 0.5 * (3*rs_m)**2 * c**2 / (r_range * 1000)**2
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=r_range/rs_km,
                    y=V_eff,
                    mode='lines',
                    line=dict(color='#667eea', width=3),
                    name='Potentiel Effectif'
                ))
                
                fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="Rs")
                fig.add_vline(x=r_isco, line_dash="dash", line_color="green", annotation_text="ISCO")
                
                fig.update_layout(
                    title="Potentiel Effectif (orbites circulaires)",
                    xaxis_title="r/Rs",
                    yaxis_title="V_eff (J/kg)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sauvegarder
                bh_data = {
                    'mass_solar': bh_mass_msun,
                    'spin': spin_param,
                    'schwarzschild_radius_km': rs_km,
                    'isco_km': r_isco_km,
                    'isco_rs': r_isco,
                    'hawking_temp_K': T_hawking,
                    'evaporation_time_years': t_evap_years,
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'black_hole_data' not in st.session_state.telescope_lab:
                    st.session_state.telescope_lab['black_hole_data'] = []
                
                st.session_state.telescope_lab['black_hole_data'].append(bh_data)
                log_event(f"Trou noir analysé: {bh_mass_msun:.2e} M☉, a/M={spin_param:.3f}", "INFO")
                
                st.success("✅ Analyse complétée!")
    
    with tab2:
        st.subheader("🌀 Disque d'Accrétion & Jets Relativistes")
        
        st.write("""
        **Physique du Disque:**
        - **Viscosité turbulente** (modèle α)
        - **Chauffage par friction**: T ∝ r^(-3/4)
        - **Émission corps noir multi-température**
        - **Efficacité:** η ≈ 6% (Schwarzschild) à 42% (Kerr extrême)
        
        **Jets Relativistes:**
        - Mécanisme Blandford-Znajek
        - Extraction énergie rotation
        - Facteur Lorentz Γ ~ 10-100
        """)
        
        if 'black_hole_data' in st.session_state.telescope_lab and st.session_state.telescope_lab['black_hole_data']:
            last_bh = st.session_state.telescope_lab['black_hole_data'][-1]
            
            st.info(f"**Trou noir actuel:** {last_bh['mass_solar']:.2e} M☉, spin={last_bh['spin']:.3f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                accretion_rate_msun_yr = st.number_input("Taux Accrétion (M☉/an)", 0.001, 100.0, 1.0, 0.001, format="%.3f")
                alpha_viscosity = st.slider("Paramètre α Viscosité", 0.01, 1.0, 0.1, 0.01)

            with col2:
                inclination_deg = st.slider("Inclinaison Observateur (deg)", 0, 90, 30)
                jet_power_eddington = st.slider("Puissance Jets (L_Edd)", 0.0, 1.0, 0.1, 0.01)
            
            if st.button("🌀 Simuler Disque & Jets", type="primary"):
                with st.spinner("Simulation magnétohydrodynamique..."):
                    import time
                    time.sleep(2)
                    
                    # Paramètres physiques
                    M_sun_kg = 1.989e30
                    c = 299792458
                    
                    # Efficacité accrétion (dépend du spin)
                    if last_bh['spin'] < 0.1:
                        efficiency = 0.057  # Schwarzschild
                    else:
                        efficiency = 0.057 + 0.32 * last_bh['spin']  # Jusqu'à 42% pour a=1
                    
                    # Luminosité bolométrique
                    M_dot_kg_s = accretion_rate_msun_yr * M_sun_kg / (365.25 * 24 * 3600)
                    L_bol = efficiency * M_dot_kg_s * c**2
                    
                    # Luminosité Eddington
                    L_edd = 1.26e38 * last_bh['mass_solar']  # W
                    eddington_ratio = L_bol / L_edd
                    
                    # Température disque en fonction du rayon
                    r_inner = last_bh['isco_rs']  # En Rs
                    r_range = np.logspace(np.log10(r_inner), 3, 100)  # de ISCO à 1000 Rs
                    
                    # Profil température
                    T_profile = 3e6 * (last_bh['mass_solar'] / 1e8)**(-0.25) * (M_dot_kg_s / 1e25)**(0.25) * r_range**(-0.75)
                    
                    # Graphique profil radial
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Température Disque", "Spectre Émission", "Jets Relativistes", "Courbe Lumière"),
                        specs=[[{"type": "scatter"}, {"type": "scatter"}],
                               [{"type": "scatter"}, {"type": "scatter"}]]
                    )
                    
                    # 1. Température
                    fig.add_trace(go.Scatter(
                        x=r_range,
                        y=T_profile,
                        mode='lines',
                        line=dict(color='#FF6B6B', width=3),
                        name='T(r)'
                    ), row=1, col=1)
                    
                    fig.update_xaxes(title_text="r/Rs", type="log", row=1, col=1)
                    fig.update_yaxes(title_text="Température (K)", type="log", row=1, col=1)
                    
                    # 2. Spectre multi-température
                    wavelengths_nm = np.logspace(0, 4, 1000)
                    spectrum_total = np.zeros_like(wavelengths_nm)
                    
                    # Intégrer Planck sur toutes températures
                    for i, T in enumerate(T_profile[::5]):  # Échantillonner
                        if T > 1000:
                            h = 6.626e-34
                            k_B = 1.381e-23
                            lambda_m = wavelengths_nm * 1e-9
                            
                            B_lambda = (2*h*c**2/lambda_m**5) / (np.exp(h*c/(lambda_m*k_B*T)) - 1)
                            spectrum_total += B_lambda * (r_range[i*5] if i*5 < len(r_range) else r_range[-1])**2
                    
                    spectrum_total /= spectrum_total.max()
                    
                    fig.add_trace(go.Scatter(
                        x=wavelengths_nm,
                        y=spectrum_total,
                        mode='lines',
                        line=dict(color='#4ECDC4', width=3),
                        name='Spectre'
                    ), row=1, col=2)
                    
                    fig.update_xaxes(title_text="λ (nm)", type="log", row=1, col=2)
                    fig.update_yaxes(title_text="Flux (u.a.)", row=1, col=2)
                    
                    # 3. Jets (projection 2D)
                    z_jet = np.linspace(0, 100, 50)  # En Rs
                    r_jet = 0.1 * z_jet**0.7  # Ouverture conique
                    
                    fig.add_trace(go.Scatter(
                        x=r_jet,
                        y=z_jet,
                        mode='lines',
                        line=dict(color='cyan', width=3),
                        name='Jet',
                        showlegend=False
                    ), row=2, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=-r_jet,
                        y=z_jet,
                        mode='lines',
                        line=dict(color='cyan', width=3),
                        showlegend=False
                    ), row=2, col=1)
                    
                    # Disque (vue de côté)
                    r_disk_view = np.linspace(-20, 20, 100)
                    z_disk_view = 0.5 * np.abs(r_disk_view) * np.sin(inclination_deg * np.pi / 180)
                    
                    fig.add_trace(go.Scatter(
                        x=r_disk_view,
                        y=z_disk_view,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='orange', width=2),
                        name='Disque',
                        showlegend=False
                    ), row=2, col=1)
                    
                    fig.update_xaxes(title_text="r/Rs", row=2, col=1)
                    fig.update_yaxes(title_text="z/Rs", row=2, col=1)
                    
                    # 4. Variabilité
                    time_days = np.linspace(0, 100, 1000)
                    
                    # Variabilité quasi-périodique
                    freq_qpo = c**3 / (2 * np.pi * 6.67e-11 * last_bh['mass_solar'] * M_sun_kg * r_inner * last_bh['schwarzschild_radius_km'] * 1000)
                    period_qpo_days = 1 / (freq_qpo * 86400)
                    
                    flux_var = 1 + 0.1 * np.sin(2*np.pi * time_days / period_qpo_days) + 0.05 * np.random.randn(len(time_days))
                    
                    fig.add_trace(go.Scatter(
                        x=time_days,
                        y=flux_var,
                        mode='lines',
                        line=dict(color='#667eea', width=2),
                        name='Flux'
                    ), row=2, col=2)
                    
                    fig.update_xaxes(title_text="Temps (jours)", row=2, col=2)
                    fig.update_yaxes(title_text="Flux relatif", row=2, col=2)
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=800,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métriques
                    st.write("### 📊 Propriétés du Système")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("L bolométrique", f"{L_bol:.2e} W")
                        st.metric("L/L_Edd", f"{eddington_ratio:.3f}")
                    
                    with col2:
                        st.metric("Efficacité", f"{efficiency*100:.1f}%")
                        T_max = T_profile.max()
                        st.metric("T max disque", f"{T_max:.2e} K")
                    
                    with col3:
                        # Rayon émission pic
                        lambda_peak = 2.898e-3 / T_max * 1e9  # nm
                        st.metric("λ pic émission", f"{lambda_peak:.1f} nm")
                        
                        # Magnitude absolue
                        L_sun = 3.828e26
                        M_abs = 4.83 - 2.5 * np.log10(L_bol / L_sun)
                        st.metric("Magnitude Abs.", f"{M_abs:.1f}")
                    
                    with col4:
                        # Puissance jets
                        P_jet = jet_power_eddington * L_edd
                        st.metric("Puissance Jets", f"{P_jet:.2e} W")
                        
                        # Période QPO
                        st.metric("Période QPO", f"{period_qpo_days:.2f} jours")
                    
                    # Classification
                    st.write("### 🏷️ Classification AGN")
                    
                    if eddington_ratio < 0.01:
                        agn_type = "LINER (Low Ionization)"
                    elif eddington_ratio < 0.1:
                        agn_type = "Seyfert 2 / LLAGN"
                    elif eddington_ratio < 0.3:
                        agn_type = "Seyfert 1"
                    else:
                        agn_type = "Quasar / QSO"
                    
                    st.info(f"**Type AGN:** {agn_type}")
                    
                    if jet_power_eddington > 0.05:
                        st.success("🚀 Jets relativistes détectables (Radio-loud)")
                    
                    st.success("✅ Simulation disque complétée!")
        else:
            st.info("Analysez d'abord un trou noir dans l'onglet 'Propriétés'")
    
    with tab3:
        st.subheader("🔭 Méthodes de Détection")
        
        st.write("""
        **Détection Indirecte:**
        1. **Binaires X:** Masse compagnon invisible (Cygnus X-1)
        2. **Dynamique stellaire:** Mouvement étoiles autour Sgr A*
        3. **Lentilles gravitationnelles:** Déformation lumière
        4. **Ondes gravitationnelles:** Fusion trous noirs (LIGO/Virgo)
        5. **Event Horizon Telescope:** Image ombre (M87*, Sgr A*)
        6. **Jets radio:** Émission synchrotron
        """)
        
        detection_method = st.selectbox("Méthode de Détection",
            ["Binaire X", "Dynamique Stellaire", "Lentille Gravitationnelle", 
             "Ondes Gravitationnelles", "Imagerie EHT", "Émission X"])
        
        if detection_method == "Ondes Gravitationnelles":
            st.write("### 🌊 Signal Ondes Gravitationnelles - Fusion TN")
            
            col1, col2 = st.columns(2)
            
            with col1:
                m1_msun = st.slider("Masse TN 1 (M☉)", 5, 100, 30)
                m2_msun = st.slider("Masse TN 2 (M☉)", 5, 100, 30)
            
            with col2:
                distance_mpc = st.slider("Distance (Mpc)", 100, 5000, 1000)
                inclination_gw = st.slider("Inclinaison (deg)", 0, 90, 0)
            
            if st.button("🌊 Simuler Fusion & Signal GW"):
                with st.spinner("Calcul forme d'onde..."):
                    import time
                    time.sleep(2)
                    
                    # Chirp mass
                    M_chirp = (m1_msun * m2_msun)**(3/5) / (m1_msun + m2_msun)**(1/5)
                    
                    # Masse totale et réduite
                    M_total = m1_msun + m2_msun
                    mu = m1_msun * m2_msun / M_total
                    
                    # Fréquence ISCO (fin inspiral)
                    M_sun_kg = 1.989e30
                    G = 6.67430e-11
                    c = 299792458
                    
                    f_isco = c**3 / (6**(3/2) * np.pi * G * M_total * M_sun_kg)
                    
                    # Signal temporel (inspiral + merger + ringdown)
                    t = np.linspace(-1, 0.1, 2000)
                    
                    # Phase inspiral (t < 0)
                    t_inspiral = t[t < 0]
                    f_inspiral = f_isco * (1 + 100*t_inspiral)**(-3/8)
                    
                    # Amplitude (décroît avec distance)
                    strain_amplitude = 1e-21 * (M_chirp / 30)**(5/3) * (1000 / distance_mpc) * np.cos(inclination_gw * np.pi/180)
                    
                    # Signal inspiral
                    phase_inspiral = 2 * np.pi * np.cumsum(f_inspiral) * 0.001
                    h_inspiral = strain_amplitude * np.sin(phase_inspiral)
                    
                    # Merger (t ≈ 0)
                    t_merger = t[(t >= -0.01) & (t <= 0.01)]
                    h_merger = strain_amplitude * 2 * np.sin(2 * np.pi * f_isco * t_merger) * np.exp(-50*t_merger**2)
                    
                    # Ringdown (t > 0)
                    t_ringdown = t[t > 0.01]
                    f_ringdown = f_isco * 1.2  # Quasi-normal mode
                    tau_ringdown = 0.02  # Temps amortissement
                    h_ringdown = strain_amplitude * np.sin(2 * np.pi * f_ringdown * t_ringdown) * np.exp(-t_ringdown / tau_ringdown)
                    
                    # Signal complet
                    h_signal = np.concatenate([h_inspiral, h_merger, h_ringdown])
                    t_signal = np.concatenate([t_inspiral, t_merger, t_ringdown])
                    
                    # Graphiques
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Signal Temporel h(t)", "Fréquence Instantanée", 
                                      "Spectrogramme", "SNR Optimal"),
                        specs=[[{"type": "scatter"}, {"type": "scatter"}],
                               [{"type": "heatmap"}, {"type": "scatter"}]]
                    )
                    
                    # 1. Signal temporel
                    fig.add_trace(go.Scatter(
                        x=t_signal * 1000,  # ms
                        y=h_signal,
                        mode='lines',
                        line=dict(color='#667eea', width=1),
                        name='h(t)'
                    ), row=1, col=1)
                    
                    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Merger", row=1, col=1)
                    
                    fig.update_xaxes(title_text="Temps (ms)", row=1, col=1)
                    fig.update_yaxes(title_text="Amplitude h", row=1, col=1)
                    
                    # 2. Fréquence instantanée (chirp)
                    f_complete = np.concatenate([f_inspiral, 
                                                f_isco * np.ones_like(t_merger),
                                                f_ringdown * np.ones_like(t_ringdown)])
                    
                    fig.add_trace(go.Scatter(
                        x=t_signal * 1000,
                        y=f_complete,
                        mode='lines',
                        line=dict(color='#4ECDC4', width=2),
                        name='f(t)'
                    ), row=1, col=2)
                    
                    fig.update_xaxes(title_text="Temps (ms)", row=1, col=2)
                    fig.update_yaxes(title_text="Fréquence (Hz)", row=1, col=2)
                    
                    # 3. Spectrogramme (simplifié)
                    from scipy import signal as scipy_signal
                    
                    # Sous-échantillonner pour spectrogramme
                    sample_rate = 2048  # Hz
                    t_spec = np.linspace(-1, 0.1, sample_rate)
                    h_spec = np.interp(t_spec, t_signal, h_signal)
                    
                    f_spec, t_spec_out, Sxx = scipy_signal.spectrogram(h_spec, sample_rate, nperseg=128)
                    
                    fig.add_trace(go.Heatmap(
                        x=t_spec_out * 1000,
                        y=f_spec,
                        z=10 * np.log10(Sxx + 1e-10),
                        colorscale='Hot',
                        showscale=False
                    ), row=2, col=1)
                    
                    fig.update_xaxes(title_text="Temps (ms)", row=2, col=1)
                    fig.update_yaxes(title_text="Fréquence (Hz)", row=2, col=1)
                    
                    # 4. SNR optimal en fonction distance
                    distances_range = np.linspace(100, 5000, 100)
                    snr_range = 8 * (M_chirp / 30)**(5/6) * (1000 / distances_range)
                    
                    fig.add_trace(go.Scatter(
                        x=distances_range,
                        y=snr_range,
                        mode='lines',
                        line=dict(color='#FF6B6B', width=3),
                        name='SNR'
                    ), row=2, col=2)
                    
                    fig.add_hline(y=8, line_dash="dash", line_color="green", 
                                 annotation_text="Seuil détection", row=2, col=2)
                    fig.add_vline(x=distance_mpc, line_dash="dash", line_color="white",
                                 annotation_text=f"{distance_mpc} Mpc", row=2, col=2)
                    
                    fig.update_xaxes(title_text="Distance (Mpc)", row=2, col=2)
                    fig.update_yaxes(title_text="SNR Optimal", row=2, col=2)
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=800,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Paramètres détectés
                    st.write("### 📊 Paramètres de la Fusion")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Chirp Mass", f"{M_chirp:.2f} M☉")
                        st.metric("Masse Totale", f"{M_total} M☉")
                    
                    with col2:
                        # Masse finale (énergie rayonnée ~5%)
                        M_final = M_total - 0.05 * M_total
                        st.metric("Masse Finale", f"{M_final:.1f} M☉")
                        
                        E_radiated = 0.05 * M_total * M_sun_kg * c**2
                        st.metric("Énergie Rayonnée", f"{E_radiated:.2e} J")
                    
                    with col3:
                        st.metric("f @ ISCO", f"{f_isco:.1f} Hz")
                        st.metric("Durée Signal", f"{abs(t_signal.min())*1000:.0f} ms")
                    
                    with col4:
                        snr_optimal = 8 * (M_chirp / 30)**(5/6) * (1000 / distance_mpc)
                        st.metric("SNR Optimal", f"{snr_optimal:.1f}")
                        
                        if snr_optimal > 8:
                            st.success("✅ Détectable!")
                        else:
                            st.warning("⚠️ SNR faible")
                    
                    # Luminosité pic
                    L_gw_peak = c**5 / G * 0.01  # ~10⁴⁹ W
                    st.info(f"💥 **Luminosité pic:** {L_gw_peak:.2e} W (~10⁵² erg/s)")
                    st.info(f"🌟 **Équivalent:** {L_gw_peak / 3.828e26:.2e} L☉ (plus lumineux que tout l'univers visible!)")
                    
                    st.balloons()
                    
                    # Enregistrer événement
                    if 'gravitational_waves' not in st.session_state.telescope_lab:
                        st.session_state.telescope_lab['gravitational_waves'] = []
                    
                    gw_event = {
                        'mass1': m1_msun,
                        'mass2': m2_msun,
                        'chirp_mass': M_chirp,
                        'final_mass': M_final,
                        'energy_radiated_J': E_radiated,
                        'distance_mpc': distance_mpc,
                        'snr': snr_optimal,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.telescope_lab['gravitational_waves'].append(gw_event)
                    log_event(f"Événement GW simulé: {m1_msun}+{m2_msun} M☉ @ {distance_mpc} Mpc", "SUCCESS")
        
        elif detection_method == "Dynamique Stellaire":
            st.write("### ⭐ Mouvement Stellaire autour TN Central")
            
            st.write("""
            **Exemple: Sgr A*** (centre Voie Lactée)
            - Étoile S2: Période 16 ans, excentricité 0.88
            - Mesures astrométriques → M_BH = 4.15 × 10⁶ M☉
            """)
            
            n_stars = st.slider("Nombre d'étoiles à simuler", 5, 50, 20)
            bh_mass_center = st.number_input("Masse TN Central (10⁶ M☉)", 1.0, 100.0, 4.15)
            
            if st.button("⭐ Simuler Orbites Stellaires"):
                with st.spinner("Calcul orbites képlériennes..."):
                    import time
                    time.sleep(1.5)
                    
                    # Générer orbites aléatoires
                    semi_major_axes = np.random.uniform(0.01, 1, n_stars)  # arcsec
                    eccentricities = np.random.uniform(0.1, 0.9, n_stars)
                    inclinations = np.random.uniform(0, 180, n_stars)
                    longitudes = np.random.uniform(0, 360, n_stars)
                    longitudes = np.random.uniform(0, 360, n_stars)
                    
                    # Périodes orbitales (3ème loi Kepler)
                    # Distance Galactic Center ~8 kpc, 1" ~ 0.04 pc
                    pc_per_arcsec = 0.04
                    periods_years = np.sqrt((semi_major_axes * pc_per_arcsec)**3 / (bh_mass_center * 1e6)) * 30  # années
                    
                    # Tracer orbites
                    fig = go.Figure()
                    
                    for i in range(n_stars):
                        # Paramètres orbitaux
                        a = semi_major_axes[i]
                        e = eccentricities[i]
                        
                        # Anomalie excentrique
                        E = np.linspace(0, 2*np.pi, 100)
                        
                        # Coordonnées orbitales
                        r = a * (1 - e * np.cos(E))
                        x_orbit = r * np.cos(E)
                        y_orbit = r * np.sin(E) * np.cos(inclinations[i] * np.pi/180)
                        
                        # Rotation
                        angle = longitudes[i] * np.pi / 180
                        x_rot = x_orbit * np.cos(angle) - y_orbit * np.sin(angle)
                        y_rot = x_orbit * np.sin(angle) + y_orbit * np.cos(angle)
                        
                        fig.add_trace(go.Scatter(
                            x=x_rot,
                            y=y_rot,
                            mode='lines',
                            line=dict(width=1),
                            name=f'S{i+1}',
                            showlegend=False
                        ))
                        
                        # Position actuelle (aléatoire sur orbite)
                        idx_current = np.random.randint(0, 100)
                        fig.add_trace(go.Scatter(
                            x=[x_rot[idx_current]],
                            y=[y_rot[idx_current]],
                            mode='markers',
                            marker=dict(size=8, color='yellow'),
                            name=f'S{i+1}',
                            showlegend=False
                        ))
                    
                    # Trou noir central
                    fig.add_trace(go.Scatter(
                        x=[0], y=[0],
                        mode='markers',
                        marker=dict(size=20, color='black', line=dict(color='white', width=2)),
                        name='Sgr A*'
                    ))
                    
                    fig.update_layout(
                        title=f"Orbites Stellaires - TN Central {bh_mass_center:.2f}×10⁶ M☉",
                        xaxis_title="ΔRA (arcsec)",
                        yaxis_title="ΔDec (arcsec)",
                        template="plotly_dark",
                        height=600,
                        showlegend=False
                    )
                    
                    fig.update_xaxes(scaleanchor="y", scaleratio=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau étoiles
                    star_data = []
                    for i in range(min(10, n_stars)):
                        star_data.append({
                            'Étoile': f'S{i+1}',
                            'Période (ans)': f'{periods_years[i]:.1f}',
                            'Excentricité': f'{eccentricities[i]:.2f}',
                            'a (arcsec)': f'{semi_major_axes[i]:.3f}',
                            'Inclinaison': f'{inclinations[i]:.0f}°'
                        })
                    
                    df_stars = pd.DataFrame(star_data)
                    st.dataframe(df_stars, use_container_width=True)
                    
                    st.success("✅ Orbites calculées!")
                    st.info(f"📏 Masse TN mesurée: {bh_mass_center:.2f} × 10⁶ M☉")
    
    with tab4:
        st.subheader("📊 Catalogue Trous Noirs")
        
        if 'black_hole_data' in st.session_state.telescope_lab and st.session_state.telescope_lab['black_hole_data']:
            st.write(f"### ⚫ {len(st.session_state.telescope_lab['black_hole_data'])} Trous Noirs Catalogués")
            
            bh_data_list = []
            for i, bh in enumerate(st.session_state.telescope_lab['black_hole_data']):
                # Classification
                if bh['mass_solar'] < 100:
                    bh_type = "Stellaire"
                elif bh['mass_solar'] < 1e5:
                    bh_type = "Intermédiaire"
                else:
                    bh_type = "Supermassif"
                
                bh_data_list.append({
                    'ID': f"BH_{i+1:03d}",
                    'Type': bh_type,
                    'Masse (M☉)': f"{bh['mass_solar']:.2e}",
                    'Spin (a/M)': f"{bh['spin']:.3f}",
                    'Rs (km)': f"{bh['schwarzschild_radius_km']:.2f}",
                    'ISCO': f"{bh['isco_rs']:.2f} Rs",
                    'T_Hawking (K)': f"{bh['hawking_temp_K']:.2e}"
                })
            
            df_bh = pd.DataFrame(bh_data_list)
            st.dataframe(df_bh, use_container_width=True)
            
            # Visualisations
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution masses
                masses = [bh['mass_solar'] for bh in st.session_state.telescope_lab['black_hole_data']]
                
                fig = go.Figure(data=go.Histogram(
                    x=masses,
                    nbinsx=20,
                    marker_color='#667eea'
                ))
                
                fig.update_layout(
                    title="Distribution Masses",
                    xaxis_title="Masse (M☉)",
                    yaxis_title="Nombre",
                    xaxis_type="log",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Spin vs Masse
                spins = [bh['spin'] for bh in st.session_state.telescope_lab['black_hole_data']]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=masses,
                    y=spins,
                    mode='markers',
                    marker=dict(size=10, color=spins, colorscale='Viridis', showscale=True),
                    text=[f"M={m:.2e}<br>a={s:.2f}" for m, s in zip(masses, spins)],
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Spin vs Masse",
                    xaxis_title="Masse (M☉)",
                    yaxis_title="Paramètre Spin (a/M)",
                    xaxis_type="log",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            st.write("### 📈 Statistiques")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                n_stellar = sum(1 for m in masses if m < 100)
                st.metric("Stellaires", n_stellar)
            
            with col2:
                n_intermediate = sum(1 for m in masses if 100 <= m < 1e5)
                st.metric("Intermédiaires", n_intermediate)
            
            with col3:
                n_supermassive = sum(1 for m in masses if m >= 1e5)
                st.metric("Supermassifs", n_supermassive)
            
            with col4:
                avg_spin = np.mean(spins)
                st.metric("Spin Moyen", f"{avg_spin:.3f}")
        
        else:
            st.info("Aucun trou noir catalogué. Analysez-en dans l'onglet 'Propriétés'.")
                    
# ==================== PAGE: BIOASTRONOMY ====================
elif page == "🧬 Bioastronomy":
    st.header("🧬 Bioastronomie - Recherche Vie Extraterrestre")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Biosignatures", "🌊 Mondes Océans", "🧪 Chimie Prébiotique", "📡 SETI"])
    
    with tab1:
        st.subheader("🔬 Détection Biosignatures Atmosphériques")
        
        st.write("""
        **Biosignatures Potentielles:**
        - **O₂ + CH₄:** Déséquilibre chimique (vie?)
        - **O₃ (ozone):** Produit photochimique O₂
        - **N₂O:** Produit biologique
        - **CH₃Cl:** Métabolisme microbien
        - **Phosphine (PH₃):** Anaérobie
        - **Dimethyl Sulfide (DMS):** Phytoplancton
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            planet_type = st.selectbox("Type Exoplanète",
                ["Super-Terre Tempérée", "Terrestre Zone Habitable", 
                 "Mini-Neptune", "Monde Océan"])
            
            star_type = st.selectbox("Type Étoile Hôte",
                ["M-dwarf (Naine Rouge)", "K-dwarf", "G-dwarf (Type Solaire)", "F-dwarf"])
        
        with col2:
            equilibrium_temp = st.slider("Température Équilibre (K)", 200, 400, 288)
            planet_radius = st.slider("Rayon Planète (R⊕)", 0.5, 3.0, 1.0, 0.1)
        
        if st.button("🔬 Analyser Atmosphère", type="primary"):
            with st.spinner("Analyse spectroscopique transmission..."):
                import time
                time.sleep(2)
                
                # Spectre transmission simulé
                wavelengths = np.linspace(0.5, 5.0, 500)  # μm
                
                # Baseline (Rayleigh scattering)
                baseline = 1 - 0.001 * wavelengths**(-4)
                
                # Molécules
                molecules_detected = {}
                
                # H2O (1.4, 1.9 μm)
                if 250 < equilibrium_temp < 400:
                    h2o_band = 0.01 * np.exp(-((wavelengths - 1.4)/0.1)**2) + \
                               0.015 * np.exp(-((wavelengths - 1.9)/0.15)**2)
                    baseline += h2o_band
                    molecules_detected['H₂O'] = "✅ Détecté"
                
                # CH4 (3.3 μm)
                if np.random.random() > 0.3:
                    ch4_band = 0.008 * np.exp(-((wavelengths - 3.3)/0.2)**2)
                    baseline += ch4_band
                    molecules_detected['CH₄'] = "✅ Détecté"
                
                # O3 (9.6 μm - hors range mais on simule)
                if equilibrium_temp > 250 and np.random.random() > 0.5:
                    molecules_detected['O₃'] = "✅ Détecté (MIR)"
                
                # CO2 (4.3 μm)
                co2_band = 0.012 * np.exp(-((wavelengths - 4.3)/0.25)**2)
                baseline += co2_band
                molecules_detected['CO₂'] = "✅ Détecté"
                
                # Ajouter bruit
                spectrum = baseline + np.random.normal(0, 0.0005, len(wavelengths))
                
                # Graphique
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=wavelengths,
                    y=spectrum,
                    mode='lines',
                    line=dict(color='#667eea', width=2),
                    name='Spectre Transmission'
                ))
                
                # Marquer bandes
                if 'H₂O' in molecules_detected:
                    fig.add_vrect(x0=1.3, x1=1.5, fillcolor="blue", opacity=0.2, annotation_text="H₂O")
                    fig.add_vrect(x0=1.8, x1=2.0, fillcolor="blue", opacity=0.2)
                
                if 'CH₄' in molecules_detected:
                    fig.add_vrect(x0=3.2, x1=3.4, fillcolor="orange", opacity=0.2, annotation_text="CH₄")
                
                fig.add_vrect(x0=4.2, x1=4.4, fillcolor="red", opacity=0.2, annotation_text="CO₂")
                
                fig.update_layout(
                    title="Spectre Transmission Atmosphérique",
                    xaxis_title="Longueur d'onde (μm)",
                    yaxis_title="Profondeur Transit Relative",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Résultats
                st.write("### 🧪 Molécules Détectées")
                
                for mol, status in molecules_detected.items():
                    st.write(f"**{mol}:** {status}")
                
                # Évaluation habitabilité
                st.write("### 🌍 Évaluation Habitabilité")
                
                habitability_score = 0
                
                if 'H₂O' in molecules_detected:
                    habitability_score += 3
                    st.success("✅ Eau détectée - essentiel pour vie connue")
                
                if 'CH₄' in molecules_detected and 'O₃' in molecules_detected:
                    habitability_score += 4
                    st.success("✅ CH₄ + O₃ - Déséquilibre chimique (biosignature!)")
                    st.balloons()
                
                if 250 < equilibrium_temp < 350:
                    habitability_score += 2
                    st.success("✅ Température compatible eau liquide")
                
                if planet_radius < 1.6:
                    habitability_score += 1
                    st.success("✅ Taille compatible planète rocheuse")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score Habitabilité", f"{habitability_score}/10")
                
                with col2:
                    if habitability_score >= 7:
                        st.success("🌟 Excellent Candidat!")
                    elif habitability_score >= 5:
                        st.info("📊 Candidat Intéressant")
                    else:
                        st.warning("⚠️ Peu Probable")
                
                with col3:
                    if habitability_score >= 7:
                        st.metric("Priorité Suivi", "HAUTE 🔴")
    
    with tab2:
        st.subheader("🌊 Mondes Océans - Lunes Glacées")
        
        st.write("""
        **Candidats Système Solaire:**
        - **Europa (Jupiter):** Ocean sous-glaciaire 100km profondeur
        - **Enceladus (Saturne):** Geysers actifs, molécules organiques
        - **Titan (Saturne):** Lacs méthane, chimie organique complexe
        - **Ganymède (Jupiter):** Plus grand satellite, océan salé
        """)
        
        moon = st.selectbox("Sélectionner Lune",
            ["Europa", "Enceladus", "Titan", "Ganymède"])
        
        if st.button("🌊 Analyser Potentiel Vie"):
            with st.spinner(f"Analyse {moon}..."):
                import time
                time.sleep(1.5)
                
                properties = {}
                
                if moon == "Europa":
                    properties = {
                        'Diamètre': '3121 km',
                        'Épaisseur Glace': '15-25 km',
                        'Profondeur Océan': '~100 km',
                        'Volume Eau': '2-3× Terre',
                        'Salinité': 'Probable (MgSO₄)',
                        'Sources Énergie': 'Marées + Radioactivité',
                        'Détections': 'Panaches vapeur, champ magnétique induit',
                        'Score Vie': '⭐⭐⭐⭐⭐'
                    }
                elif moon == "Enceladus":
                    properties = {
                        'Diamètre': '504 km',
                        'Épaisseur Glace': '30-40 km (pôle sud: 5km)',
                        'Profondeur Océan': '~10 km',
                        'Volume Eau': '0.5× Terre',
                        'Salinité': 'Confirmée (NaCl)',
                        'Sources Énergie': 'Marées Saturne',
                        'Détections': 'H₂, CO₂, NH₃, organiques dans geysers',
                        'Score Vie': '⭐⭐⭐⭐⭐'
                    }
                elif moon == "Titan":
                    properties = {
                        'Diamètre': '5150 km',
                        'Atmosphère': 'Dense (1.5 bar), N₂ + CH₄',
                        'Lacs/Mers': 'Hydrocarbures liquides',
                        'Océan Subsurface': 'Possible (eau + NH₃)',
                        'Chimie': 'Tholins, nitriles, molécules prébiotiques',
                        'Sources Énergie': 'UV solaire + marées',
                        'Détections': 'Benzène, cyanure hydrogène (HCN)',
                        'Score Vie': '⭐⭐⭐⭐'
                    }
                else:  # Ganymède
                    properties = {
                        'Diamètre': '5268 km (plus grand satellite)',
                        'Épaisseur Glace': '~150 km',
                        'Profondeur Océan': '~100 km',
                        'Volume Eau': '~ Terre',
                        'Salinité': 'Probable',
                        'Champ Magnétique': 'Intrinsèque (unique!)',
                        'Détections': 'Aurorae O₂, champ magnétique induit',
                        'Score Vie': '⭐⭐⭐'
                    }
                
                for key, value in properties.items():
                    st.write(f"**{key}:** {value}")
                
                st.success(f"✅ {moon} analysé!")
                
                if moon in ["Europa", "Enceladus"]:
                    st.balloons()
                    st.info("🚀 **Mission recommandée:** Lander + Foreuse pour échantillon subsurface")
    
    with tab3:
        st.subheader("🧪 Chimie Prébiotique & Panspermie")
        
        st.write("""
        **Molécules Organiques Interstellaires:**
        - **Acides aminés:** Glycine (détectée comète 67P)
        - **Bases azotées:** Adénine, guanine (météorites)
        - **Sucres:** Ribose (synthèse laboratoire conditions ISM)
        - **Lipides:** Amphiphiles (formation membranes)
        """)
        
        if st.button("🧪 Simuler Synthèse Prébiotique"):
            with st.spinner("Expérience Miller-Urey moderne..."):
                import time
                time.sleep(2)
                
                st.write("### ⚡ Conditions Expérimentales")
                
                conditions = {
                    'Atmosphère': 'CH₄, NH₃, H₂O, H₂',
                    'Énergie': 'Décharges électriques (éclairs)',
                    'Température': '25°C',
                    'Durée': '7 jours',
                    'pH': '7-8'
                }
                
                for key, value in conditions.items():
                    st.write(f"**{key}:** {value}")
                
                st.write("### 🧬 Molécules Produites")
                
                molecules = {
                    'Acides Aminés': ['Glycine', 'Alanine', 'Acide aspartique', 'Acide glutamique'],
                    'Bases Puriques': ['Adénine', 'Guanine'],
                    'Autres Organiques': ['Formaldéhyde', 'Acide cyanhydrique', 'Urée']
                }
                
                for category, items in molecules.items():
                    st.write(f"**{category}:**")
                    for item in items:
                        st.write(f"  • {item}")
                
                st.success("✅ Briques du vivant synthétisées abiotiquement!")
                st.info("💡 **Conclusion:** Chimie prébiotique possible dans univers primitif")
    
    with tab4:
        st.subheader("📡 SETI - Search for Extraterrestrial Intelligence")
        
        st.write("""
        **Équation de Drake:** N = R* × fp × ne × fl × fi × fc × L
        
        - R*: Taux formation étoiles (10/an)
        - fp: Fraction étoiles avec planètes (0.5)
        - ne: Planètes zone habitable par système (2)
        - fl: Fraction développant vie (0.5)
        - fi: Fraction développant intelligence (0.1)
        - fc: Fraction communicante (0.1)
        - L: Durée civilisation communicante (10000 ans)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            r_star = st.slider("R* (étoiles/an)", 1, 50, 10)
            f_p = st.slider("fp (fraction planètes)", 0.0, 1.0, 0.5, 0.05)
            n_e = st.slider("ne (planètes habitables)", 0.0, 5.0, 2.0, 0.1)
        
        with col2:
            f_l = st.slider("fl (vie)", 0.0, 1.0, 0.5, 0.05)
            f_i = st.slider("fi (intelligence)", 0.0, 1.0, 0.1, 0.01)
            f_c = st.slider("fc (communicante)", 0.0, 1.0, 0.1, 0.01)
            lifetime = st.slider("L (années)", 100, 1000000, 10000, 100)
        
        N_drake = r_star * f_p * n_e * f_l * f_i * f_c * lifetime
        
        st.metric("🛸 Civilisations Communicantes (Voie Lactée)", f"{N_drake:.1f}")
        
        if N_drake > 1000:
            st.success("🎉 Nombreuses civilisations probables!")
        elif N_drake > 10:
            st.info("📊 Plusieurs civilisations possibles")
        elif N_drake > 1:
            st.warning("⚠️ Nous ne sommes probablement pas seuls")
        else:
            st.error("😔 Paradoxe de Fermi - Où sont-ils?")

# ==================== PAGE: MULTI-MESSAGER ====================
elif page == "📡 Multi-Messager":
    st.header("📡 Astronomie Multi-Messager")
    
    st.info("""
    **4 Messagers Cosmiques:**
    - 🔭 **Photons:** Lumière visible → gamma
    - 🌊 **Ondes Gravitationnelles:** LIGO/Virgo
    - ⚛️ **Neutrinos:** IceCube, Super-Kamiokande
    - 🌌 **Rayons Cosmiques:** Ultra-haute énergie
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Kilonova GW170817", "⚡ Supernova Neutrinos", "🌌 Rayons Cosmiques", "📊 Corrélations"])
    
    with tab1:
        st.subheader("🔥 Événement GW170817 - Fusion Étoiles à Neutrons")
        
        st.write("""
        **17 Août 2017 - Première Détection Multi-Messager:**
        - ⏰ **12:41:04 UTC:** LIGO/Virgo détectent GW170817
        - 🔭 **+1.7s:** Fermi détecte sursaut gamma (GRB)
        - 🌌 **+11h:** Découverte contrepartie optique (kilonova)
        - 📡 **Jours suivants:** Radio, X, UV observations
        """)
        
        if st.button("📊 Reconstituer Chronologie Multi-Messager"):
            with st.spinner("Analyse multi-longueurs d'onde..."):
                import time
                time.sleep(2)
                
                # Timeline
                events = [
                    {'time': 0, 'messenger': 'Ondes Gravitationnelles', 'instrument': 'LIGO/Virgo', 'significance': 'Fusion détectée'},
                    {'time': 1.7, 'messenger': 'Gamma', 'instrument': 'Fermi-GBM', 'significance': 'GRB 170817A'},
                    {'time': 11*3600, 'messenger': 'Optique', 'instrument': 'Swope/1m3', 'significance': 'Kilonova SSS17a/AT2017gfo'},
                    {'time': 16*3600, 'messenger': 'X', 'instrument': 'Chandra', 'significance': 'Jet émission'},
                    {'time': 9*86400, 'messenger': 'Radio', 'instrument': 'VLA', 'significance': 'Afterglow'}
                ]
                
                # Graphique timeline
                fig = go.Figure()
                
                times_h = [e['time']/3600 for e in events]
                messengers = [e['messenger'] for e in events]
                colors = ['blue', 'red', 'yellow', 'purple', 'orange']
                
                for i, (t, m, c) in enumerate(zip(times_h, messengers, colors)):
                    fig.add_trace(go.Scatter(
                        x=[t], y=[i],
                        mode='markers+text',
                        marker=dict(size=20, color=c),
                        text=[m],
                        textposition='top center',
                        name=m,
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Timeline Multi-Messager GW170817",
                    xaxis_title="Temps depuis GW (heures)",
                    xaxis_type="log",
                    yaxis=dict(showticklabels=False),
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Courbe lumière kilonova
                st.write("### 💡 Courbe Lumière Kilonova")
                
                time_days = np.linspace(0.1, 30, 100)
                
                # Composante bleue (lanthanides légers)
                blue_component = 10**(-14) * time_days**(-1.3) * np.exp(-time_days/3)
                
                # Composante rouge (lanthanides lourds, r-process)
                red_component = 10**(-14) * time_days**(-0.3) * np.exp(-time_days/10)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=time_days, y=blue_component,
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name='Composante Bleue'
                ))
                
                fig.add_trace(go.Scatter(
                    x=time_days, y=red_component,
                    mode='lines',
                    line=dict(color='red', width=3),
                    name='Composante Rouge (r-process)'
                ))
                
                fig.update_layout(
                    title="Évolution Spectrale Kilonova",
                    xaxis_title="Temps (jours)",
                    yaxis_title="Flux (erg/s/cm²/Å)",
                    yaxis_type="log",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Premier événement multi-messager neutron star merger!")
                st.balloons()
                
                st.info("""
                🌟 **Découvertes Majeures:**
                - Confirmation que GRB courts = fusion NS-NS
                - Synthèse éléments lourds (or, platine) par r-process
                - Mesure H₀ indépendante: 70 km/s/Mpc
                - Vitesse ondes gravitationnelles = c
                """)
    
    with tab2:
        st.subheader("⚡ Supernova & Neutrinos")
        
        st.write("""
        **SN 1987A - Première Détection Neutrinos Astronomiques:**
        - 📅 23 Février 1987
        - 📍 Grand Nuage de Magellan (50 kpc)
        - 🔬 Kamiokande-II: 11 neutrinos
        - 🔬 IMB: 8 neutrinos
        - ⏰ 3h **avant** signal optique!
        """)
        
        if st.button("⚡ Simuler Burst Neutrinos Supernova"):
            with st.spinner("Détection neutrinos..."):
                import time
                time.sleep(1.5)
                
                # Signal neutrinos
                time_s = np.linspace(-1, 15, 1000)
                
                # Émission en 3 phases
                # 1. Neutronisation (0-0.02s)
                neutronization = 100 * np.exp(-time_s**2 / 0.001) * (time_s > -0.02) * (time_s < 0.02)
                
                # 2. Accretion (0-0.5s)
                accretion = 50 * np.exp(-time_s / 0.1) * (time_s > 0) * (time_s < 0.5)
                
                # 3. Cooling (0.5-10s)
                cooling = 30 * np.exp(-(time_s-0.5) / 3) * (time_s > 0.5) * (time_s < 15)
                
                flux_total = neutronization + accretion + cooling
                
                # Ajouter détections individuelles
                n_detected = 19  # Kamiokande + IMB
                detection_times = np.random.uniform(0, 12, n_detected)
                detection_energies = np.random.uniform(10, 40, n_detected)  # MeV
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Flux Neutrinos Théorique", "Détections Individuelles"),
                    row_heights=[0.6, 0.4]
                )
                
                # Flux théorique
                fig.add_trace(go.Scatter(
                    x=time_s, y=flux_total,
                    mode='lines',
                    line=dict(color='cyan', width=3),
                    name='Flux νe'
                ), row=1, col=1)
                
                fig.update_xaxes(title_text="Temps (s)", row=1, col=1)
                fig.update_yaxes(title_text="Flux (u.a.)", row=1, col=1)
                
                # Détections
                fig.add_trace(go.Scatter(
                    x=detection_times,
                    y=detection_energies,
                    mode='markers',
                    marker=dict(size=12, color=detection_energies, colorscale='Hot', showscale=True),
                    name='Événements'
                ), row=2, col=1)
                
                fig.update_xaxes(title_text="Temps (s)", row=2, col=1)
                fig.update_yaxes(title_text="Énergie (MeV)", row=2, col=1)
                
                fig.update_layout(
                    template="plotly_dark",
                    height=600,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Neutrinos Détectés", n_detected)
                    st.metric("Énergie Moyenne", f"{np.mean(detection_energies):.1f} MeV")
                
                with col2:
                    total_energy = 3e53  # erg
                    st.metric("Énergie Totale ν", f"{total_energy:.1e} erg")
                    st.metric("= M☉c²", "~10%")
                
                with col3:
                    st.metric("Durée Émission", "~12 s")
                    st.metric("Distance", "50 kpc")
                
                st.success("✅ Confirmation effondrement core!")
    
    with tab3:
        st.subheader("🌌 Rayons Cosmiques Ultra-Haute Énergie")
        
        st.write("""
        **Mystère Énergies Extrêmes:**
        - Particules > 10²⁰ eV (macroscopique!)
        - Sources: AGN jets? Sursauts gamma? Nouveaux phénomènes?
        - Limite GZK: interaction CMB
        """)
        
        if st.button("🌌 Détecter Rayon Cosmique"):
            with st.spinner("Gerbe atmosphérique en cours..."):
                import time
                time.sleep(1)
                
                energy_eV = 10**(np.random.uniform(19, 20.5))
                
                st.success(f"💥 Événement détecté: E = {energy_eV:.2e} eV")
                
                if energy_eV > 5e19:
                    st.balloons()
                    st.warning("🚨 Au-delà limite GZK! Source proche (<100 Mpc)?")
                
                # Gerbe atmosphérique
                altitude_km = np.linspace(0, 30, 100)
                
                # Développement gerbe
                cascade = np.exp(-(altitude_km - 15)**2 / 20) * energy_eV / 1e19
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=cascade,
                    y=altitude_km,
                    mode='lines',
                    fill='tozerox',
                    line=dict(color='orange', width=3)
                ))
                
                fig.update_layout(
                    title="Gerbe Atmosphérique",
                    xaxis_title="Nombre Particules (u.a.)",
                    yaxis_title="Altitude (km)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Corrélations Multi-Messager")
        
        if st.button("📊 Analyse Corrélations"):
            with st.spinner("Recherche coïncidences temporelles..."):
                import time
                time.sleep(2)
                
                # Simuler événements multi-messager
                n_events = 50
                
                # GW events
                gw_times = np.random.uniform(0, 365, 15)
                gw_sky = np.random.uniform(0, 360, 15)
                
                # Neutrino events
                nu_times = np.random.uniform(0, 365, 30)
                nu_sky = np.random.uniform(0, 360, 30)
                
                # Gamma events
                gamma_times = np.random.uniform(0, 365, 25)
                gamma_sky = np.random.uniform(0, 360, 25)
                
                # Trouver coïncidences (< 1 jour, < 10°)
                coincidences = []
                
                for i, (t_gw, s_gw) in enumerate(zip(gw_times, gw_sky)):
                    for j, (t_nu, s_nu) in enumerate(zip(nu_times, nu_sky)):
                        if abs(t_gw - t_nu) < 1 and abs(s_gw - s_nu) < 10:
                            coincidences.append({
                                'type': 'GW + Neutrino',
                                'time_diff': abs(t_gw - t_nu),
                                'sky_sep': abs(s_gw - s_nu),
                                'significance': np.random.uniform(3, 6)
                            })
                
                for i, (t_gw, s_gw) in enumerate(zip(gw_times, gw_sky)):
                    for j, (t_g, s_g) in enumerate(zip(gamma_times, gamma_sky)):
                        if abs(t_gw - t_g) < 1 and abs(s_gw - s_g) < 10:
                            coincidences.append({
                                'type': 'GW + Gamma',
                                'time_diff': abs(t_gw - t_g),
                                'sky_sep': abs(s_gw - s_g),
                                'significance': np.random.uniform(3, 6)
                            })
                
                st.metric("🔗 Coïncidences Détectées", len(coincidences))
                
                if coincidences:
                    df_coinc = pd.DataFrame(coincidences)
                    st.dataframe(df_coinc, use_container_width=True)
                    
                    if len(coincidences) > 0:
                        st.success("✅ Événements multi-messager trouvés!")

# ==================== PAGE: RECHERCHE VIE ====================
elif page == "🔬 Recherche Vie":
    st.header("🔬 Recherche de Vie Extraterrestre")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Cibles Prioritaires", "🧬 Critères Habitabilité", "📡 Stratégie Détection"])
    
    with tab1:
        st.subheader("🎯 Exoplanètes Candidates")
        
        st.write("""
        **Top Candidats Zone Habitable:**
        """)
        
        candidates = {
            'Proxima Centauri b': {
                'distance_ly': 4.24,
                'radius': 1.17,
                'temp_eq': 234,
                'star_type': 'M5.5V',
                'flux': 0.65,
                'score': 8.5
            },
            'TRAPPIST-1e': {
                'distance_ly': 40,
                'radius': 0.92,
                'temp_eq': 251,
                'star_type': 'M8V',
                'flux': 0.66,
                'score': 9.2
            },
            'LHS 1140 b': {
                'distance_ly': 40,
                'radius': 1.73,
                'temp_eq': 230,
                'star_type': 'M4.5V',
                'flux': 0.46,
                'score': 8.8
            },
            'Kepler-442b': {
                'distance_ly': 1200,
                'radius': 1.34,
                'temp_eq': 233,
                'star_type': 'K5V',
                'flux': 0.70,
                'score': 8.3
            }
        }
        
        for name, props in candidates.items():
            with st.expander(f"🌍 {name} - Score: {props['score']}/10"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Distance", f"{props['distance_ly']} al")
                    st.metric("Rayon", f"{props['radius']} R⊕")
                
                with col2:
                    st.metric("T équilibre", f"{props['temp_eq']} K")
                    st.metric("Flux stellaire", f"{props['flux']:.2f} S⊕")
                
                with col3:
                    st.metric("Type Étoile", props['star_type'])
                    
                    if props['score'] > 9:
                        st.success("🌟 Excellente Candidate!")
                    elif props['score'] > 8:
                        st.info("⭐ Très Prometteuse")
                    else:
                        st.warning("📊 Intéressante")
                
                # Barre progression habitabilité
                st.progress(props['score'] / 10)
    
    with tab2:
        st.subheader("🧬 Score Habitabilité (ESI)")
        
        st.write("""
        **Earth Similarity Index:**
        ESI = ∏ [(1 - |x_i - x_i⊕|) / (x_i + x_i⊕)]^(w_i/n)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            planet_radius = st.slider("Rayon (R⊕)", 0.5, 3.0, 1.0, 0.1)
            planet_density = st.slider("Densité (ρ⊕)", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            escape_velocity = st.slider("Vitesse Libération (km/s)", 5, 30, 11, 1)
            surface_temp = st.slider("Température Surface (K)", 200, 400, 288, 5)
        
        if st.button("🧮 Calculer ESI"):
            # Valeurs Terre
            r_earth = 1.0
            rho_earth = 1.0
            v_esc_earth = 11.2
            t_earth = 288
            
            # ESI interior (radius + density)
            esi_interior = ((1 - abs(planet_radius - r_earth)/(planet_radius + r_earth)) * 
                           (1 - abs(planet_density - rho_earth)/(planet_density + rho_earth)))**0.5
            
            # ESI surface (escape velocity + temp)
            esi_surface = ((1 - abs(escape_velocity - v_esc_earth)/(escape_velocity + v_esc_earth)) * 
                          (1 - abs(surface_temp - t_earth)/(surface_temp + t_earth)))**0.5
            
            # ESI global
            esi_global = (esi_interior * esi_surface)**0.5
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("ESI Interior", f"{esi_interior:.3f}")
            with col2:
                st.metric("ESI Surface", f"{esi_surface:.3f}")
            with col3:
                st.metric("ESI Global", f"{esi_global:.3f}")
            
            # Interprétation
            if esi_global > 0.9:
                st.success("🌍 Jumeau de la Terre!")
                st.balloons()
            elif esi_global > 0.8:
                st.success("🌟 Excellente Candidate!")
            elif esi_global > 0.6:
                st.info("⭐ Candidate Prometteuse")
            else:
                st.warning("📊 Habitabilité Incertaine")
    
    with tab3:
        st.subheader("📡 Stratégie Observations Futures")
        
        st.write("""
        **Missions & Instruments:**
        
        🛰️ **Espace:**
        - JWST (actif): Spectroscopie transmission
        - Ariel (2029): 1000 exoplanètes
        - PLATO (2026): Terrestres zone habitable
        - HabEx/LUVOIR (2040s): Imagerie directe
        
        🏔️ **Sol:**
        - ELT (2028): Haute résolution spectrale
        - GMT (2029): Biomarkers
        - TMT (2030s): Atmosphères
        """)
        
        mission = st.selectbox("Sélectionner Mission",
            ["JWST", "ELT", "HabEx", "LUVOIR"])
        
        target_exoplanet = st.selectbox("Cible",
            list(candidates.keys()))
        
        if st.button("📋 Générer Programme Observation"):
            with st.spinner("Optimisation stratégie..."):
                import time
                time.sleep(1.5)
                
                target = candidates[target_exoplanet]
                
                st.write(f"### 📊 Programme {mission} pour {target_exoplanet}")
                
                if mission == "JWST":
                    program = {
                        'Instrument': 'NIRSpec',
                        'Mode': 'Spectroscopie Transmission',
                        'Transits': 5,
                        'Temps Total': '40 heures',
                        'SNR Attendu': 50,
                        'Molécules Détectables': ['H₂O', 'CO₂', 'CH₄', 'O₃ (si abondant)'],
                        'Priorité': 'HIGH'
                    }
                elif mission == "ELT":
                    program = {
                        'Instrument': 'METIS (MIR)',
                        'Mode': 'Haute Résolution Spectrale',
                        'Observations': '10 nuits',
                        'Temps Total': '60 heures',
                        'SNR Attendu': 100,
                        'Molécules Détectables': ['CO', 'H₂O', 'CH₄', 'NH₃', 'biosignatures'],
                        'Priorité': 'HIGH'
                    }
                elif mission == "HabEx":
                    program = {
                        'Instrument': 'Coronographe',
                        'Mode': 'Imagerie Directe',
                        'Intégration': '100 heures',
                        'Contraste': '10^-10',
                        'SNR Attendu': 7,
                        'Molécules Détectables': ['O₂', 'O₃', 'H₂O', 'végétation (edge)'],
                        'Priorité': 'HIGHEST'
                    }
                else:  # LUVOIR
                    program = {
                        'Instrument': 'ECLIPS',
                        'Mode': 'Spectroscopie + Imagerie',
                        'Intégration': '200 heures',
                        'Contraste': '10^-10',
                        'SNR Attendu': 10,
                        'Molécules Détectables': ['O₂', 'O₃', 'CH₄', 'N₂O', 'biosignatures'],
                        'Priorité': 'HIGHEST'
                    }
                
                for key, value in program.items():
                    if isinstance(value, list):
                        st.write(f"**{key}:**")
                        for item in value:
                            st.write(f"  • {item}")
                    else:
                        st.write(f"**{key}:** {value}")
                
                st.success("✅ Programme généré!")

# ==================== PAGE: MISSIONS SPATIALES ====================
elif page == "🛰️ Missions Spatiales":
    st.header("🛰️ Missions Spatiales - Passées, Présentes, Futures")
    
    tab1, tab2, tab3 = st.tabs(["📜 Historiques", "🚀 Actuelles", "🔮 Futures"])
    
    with tab1:
        st.subheader("📜 Missions Iconiques")
        
        historic_missions = {
            'Hubble Space Telescope (1990)': {
                'status': '✅ Actif',
                'découvertes': ['Expansion accélérée', 'Âge univers: 13.8 Gyr', 'Trous noirs supermassifs', 'Deep Fields'],
                'orbite': '547 km',
                'durée': '34+ ans'
            },
            'Voyager 1 & 2 (1977)': {
                'status': '✅ Actif (espace interstellaire)',
                'découvertes': ['Grand Tour planètes', 'Lunes Jupiter/Saturne', 'Espace interstellaire'],
                'distance': '>24 milliards km',
                'durée': '47+ ans'
            },
            'Kepler (2009-2018)': {
                'status': '⏸️ Terminé',
                'découvertes': ['2662 exoplanètes confirmées', 'Fréquence planètes', 'Super-Terres zone habitable'],
                'observations': '530,506 étoiles',
                'durée': '9 ans'
            },
            'Cassini-Huygens (1997-2017)': {
                'status': '⏸️ Terminé',
                'découvertes': ['Geysers Enceladus', 'Lacs méthane Titan', 'Anneaux Saturne détaillés'],
                'orbites': '294 autour Saturne',
                'durée': '20 ans'
            }
        }
        
        for mission, details in historic_missions.items():
            with st.expander(f"🛰️ {mission}"):
                st.write(f"**Status:** {details['status']}")
                st.write(f"**Durée:** {details['durée']}")
                
                st.write("**Découvertes Majeures:**")
                for disc in details['découvertes']:
                    st.write(f"  • {disc}")
                
                if 'orbite' in details:
                    st.write(f"**Orbite:** {details['orbite']}")
                if 'distance' in details:
                    st.write(f"**Distance:** {details['distance']}")
    
    with tab2:
        st.subheader("🚀 Missions Actuelles (2024-2025)")
        
        current_missions = {
            'James Webb Space Telescope': {
                'lancé': '2021',
                'objectifs': ['Premières galaxies', 'Exoplanètes', 'Formation étoiles', 'Cosmologie'],
                'instruments': ['NIRCam', 'NIRSpec', 'MIRI', 'FGS/NIRISS'],
                'orbite': 'L2 (1.5M km)',
                'status': '🟢 Opérationnel'
            },
            'Euclid': {
                'lancé': '2023',
                'objectifs': ['Matière noire', 'Énergie sombre', 'Structure à grande échelle'],
                'instruments': ['VIS', 'NISP'],
                'orbite': 'L2',
                'status': '🟢 Opérationnel'
            },
            'Gaia': {
                'lancé': '2013',
                'objectifs': ['Carte 3D Voie Lactée', '2 milliards étoiles', 'Astrométrie précise'],
                'précision': '10-25 microarcsec',
                'orbite': 'L2',
                'status': '🟢 Opérationnel'
            },
            'Parker Solar Probe': {
                'lancé': '2018',
                'objectifs': ['Couronne solaire', 'Vent solaire', 'Champ magnétique'],
                'distance_min': '6.9 millions km (9.86 Rs)',
                'vitesse_max': '700,000 km/h',
                'status': '🟢 Opérationnel'
            }
        }
        
        for mission, details in current_missions.items():
            with st.expander(f"🚀 {mission} - {details['status']}"):
                st.write(f"**Lancé:** {details['lancé']}")
                
                st.write("**Objectifs Scientifiques:**")
                for obj in details['objectifs']:
                    st.write(f"  • {obj}")
                
                if 'instruments' in details:
                    st.write(f"**Instruments:** {', '.join(details['instruments'])}")
                
                if 'orbite' in details:
                    st.write(f"**Orbite:** {details['orbite']}")
    
    with tab3:
        st.subheader("🔮 Missions Futures (2025-2040)")
        
        future_missions = {
            'Nancy Grace Roman (2027)': {
                'type': 'Télescope Spatial',
                'objectifs': ['Énergie sombre', 'Exoplanètes (microlentilles)', 'Infrarouge survey'],
                'champ': '100× Hubble',
                'résolution': 'Comparable Hubble'
            },
            'Ariel (2029)': {
                'type': 'Spectroscopie Exoplanètes',
                'objectifs': ['1000 atmosphères exoplanètes', 'Composition chimique', 'Formation/évolution'],
                'cibles': 'Chaudes à tempérées',
                'bandes': 'Visible + IR'
            },
            'ELT - Extremely Large Telescope (2028)': {
                'type': 'Sol - Optique/IR',
                'diamètre': '39 m',
                'objectifs': ['Premières galaxies', 'Trous noirs', 'Exoplanètes terrestres', 'Matière noire'],
                'localisation': 'Chili (Cerro Armazones)'
            },
            'LISA (2037)': {
                'type': 'Détecteur Ondes Gravitationnelles Spatial',
                'objectifs': ['Fusion TN supermassifs', 'Binaires compactes', 'Fond stochastique GW'],
                'bras': '2.5 millions km',
                'fréquences': '0.1 mHz - 1 Hz'
            },
            'HabEx / LUVOIR (2040s)': {
                'type': 'Télescopes Nouvelle Génération',
                'objectifs': ['Imagerie directe exoplanètes', 'Biosignatures', 'Habitabilité'],
                'technologie': 'Coronographe + Starshade',
                'contraste': '10^-10'
            }
        }
        
        for mission, details in future_missions.items():
            with st.expander(f"🔮 {mission}"):
                st.write(f"**Type:** {details['type']}")
                
                st.write("**Objectifs:**")
                for obj in details['objectifs']:
                    st.write(f"  • {obj}")
                
                for key in ['diamètre', 'champ', 'localisation', 'bras', 'contraste']:
                    if key in details:
                        st.write(f"**{key.capitalize()}:** {details[key]}")

# ==================== PAGE: COLLABORATIONS ====================
elif page == "🌍 Collaborations":
    st.header("🌍 Réseaux Collaboratifs Internationaux")
    
    tab1, tab2, tab3 = st.tabs(["🌐 Consortiums", "📊 Partage Données", "💬 Communication"])
    
    with tab1:
        st.subheader("🌐 Grands Consortiums")
        
        collaborations = {
            'Event Horizon Telescope (EHT)': {
                'membres': '13 radiotélescopes',
                'pays': '8 pays',
                'réalisation': 'Première image trou noir (M87*, Sgr A*)',
                'technique': 'VLBI planétaire',
                'participants': 300
            },
            'LIGO-Virgo-KAGRA': {
                'membres': '3 détecteurs GW',
                'pays': 'USA, Italie, Japon',
                'réalisation': 'Ondes gravitationnelles (90+ événements)',
                'technique': 'Interférométrie laser',
                'participants': 1500
            },
            'SKA - Square Kilometre Array': {
                'membres': '16 pays',
                'antennes': '~200 paraboles + 130,000 dipôles',
                'réalisation': 'Construction (2028)',
                'localisation': 'Australie + Afrique du Sud',
                'participants': 1000
            }
        }
        
        for name, info in collaborations.items():
            with st.expander(f"🤝 {name}"):
                for key, value in info.items():
                    st.write(f"**{key.capitalize()}:** {value}")
    
    with tab2:
        st.subheader("📊 Partage de Données")
        
        st.write("""
        **Archives Publiques:**
        - **MAST:** Hubble, JWST, Kepler
        - **ESO Archive:** VLT, ALMA
        - **IRSA:** Spitzer, WISE, 2MASS
        - **NED:** Base données extragalactiques
        - **SIMBAD:** Base données objets astronomiques
        """)
        
        if st.button("📥 Simuler Requête Archive"):
            with st.spinner("Recherche archives..."):
                import time
                time.sleep(1.5)
                
                results = {
                    'Images': np.random.randint(10, 100),
                    'Spectres': np.random.randint(5, 50),
                    'Catalogues': np.random.randint(2, 10),
                    'Taille Totale': f"{np.random.uniform(1, 50):.1f} GB"
                }
                
                st.success("✅ Données trouvées!")
                
                for key, value in results.items():
                    st.metric(key, value)
    
    with tab3:
        st.subheader("💬 Communication Scientifique")
        
        st.write("### 📢 Canaux Communication")
        
        channels = ['Astronomer Telegram', 'GCN (Gamma-ray Coordinates Network)', 
                   'LIGO/Virgo Alerts', 'Transient Name Server']
        
        for channel in channels:
            st.write(f"• **{channel}**")
        
        if st.button("📨 Envoyer Alerte Découverte"):
            st.info("""
            **Alerte Transient:**
            
            Object: AT2025abc
            RA: 12h 34m 56.7s
            Dec: +45° 12' 34"
            Type: Supernova Candidate
            Magnitude: 18.5 (r-band)
            Découverte: [Votre Télescope]
            Date: 2025-01-15 23:45:00 UTC
            
            Spectroscopie follow-up recommandée.
            """)
            
            st.success("✅ Alerte envoyée à la communauté!")

# ==================== PAGE: ANALYTICS ====================
elif page == "📊 Analytics":
    st.header("📊 Analytics & Statistiques Avancées")
    
    tab1, tab2, tab3 = st.tabs(["📈 Métriques Globales", "🔬 Performance", "📉 Tendances"])
    
    with tab1:
        st.subheader("📈 Métriques Observatoire")
        
        # Générer statistiques
        total_obs_time_h = total_observations * np.random.uniform(1, 3)
        data_volume_tb = total_observations * 0.5
        publications = total_discoveries // 3
        citations = publications * np.random.randint(5, 50)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Temps Observation Total", f"{total_obs_time_h:.0f}h", 
                     delta=f"+{np.random.randint(10, 50)}h ce mois")
        
        with col2:
            st.metric("Volume Données", f"{data_volume_tb:.1f} TB",
                     delta=f"+{np.random.uniform(0.5, 2):.1f} TB")
        
        with col3:
            st.metric("Publications", publications,
                     delta=f"+{np.random.randint(1, 5)}")
        
        with col4:
            st.metric("Citations", citations,
                     delta=f"+{np.random.randint(10, 100)}")
        
        # Graphique évolution
        st.write("### 📈 Évolution Temporelle")
        
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        observations_monthly = np.cumsum(np.random.randint(5, 20, 12))
        discoveries_monthly = np.cumsum(np.random.randint(0, 5, 12))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months, y=observations_monthly,
            mode='lines+markers',
            name='Observations',
            line=dict(color='#667eea', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=months, y=discoveries_monthly,
            mode='lines+markers',
            name='Découvertes',
            line=dict(color='#4ECDC4', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Activité 2025",
            xaxis_title="Mois",
            yaxis_title="Observations",
            yaxis2=dict(title="Découvertes", overlaying='y', side='right'),
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Performance Instruments")
        
        if st.session_state.telescope_lab['telescopes']:
            # Statistiques par télescope
            tel_stats = []
            
            for tel_id, tel in st.session_state.telescope_lab['telescopes'].items():
                n_obs = sum(1 for obs in st.session_state.telescope_lab['observations'] 
                           if obs.get('telescope_id') == tel_id)
                
                tel_stats.append({
                    'Télescope': tel['name'],
                    'Observations': n_obs,
                    'Temps Total (h)': n_obs * 1.5,
                    'Efficacité': f"{np.random.uniform(70, 95):.1f}%",
                    'Uptime': f"{np.random.uniform(85, 99):.1f}%"
                })
            
            if tel_stats:
                df_stats = pd.DataFrame(tel_stats)
                st.dataframe(df_stats, use_container_width=True)
        else:
            st.info("Créez des télescopes pour voir les statistiques")
    
    with tab3:
        st.subheader("📉 Tendances Scientifiques")
        
        # Topics trending
        topics = {
            'Exoplanètes': np.random.uniform(20, 40),
            'Galaxies': np.random.uniform(15, 30),
            'Trous Noirs': np.random.uniform(10, 25),
            'Cosmologie': np.random.uniform(15, 35),
            'SETI': np.random.uniform(5, 15)
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(topics.keys()),
                y=list(topics.values()),
                marker_color='#667eea',
                text=list(topics.values()),
                texttemplate='%{text:.1f}%',
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Distribution Recherches par Domaine",
            xaxis_title="Domaine",
            yaxis_title="Pourcentage",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: MONITORING LIVE ====================
elif page == "📡 Monitoring Live":
    st.header("📡 Monitoring en Temps Réel")
    
    tab1, tab2 = st.tabs(["🔴 Status Télescopes", "📊 Flux Données"])
    
    with tab1:
        st.subheader("🔴 État Télescopes en Direct")
        
        if st.session_state.telescope_lab['telescopes']:
            for tel_id, tel in st.session_state.telescope_lab['telescopes'].items():
                status = np.random.choice(['🟢 Opérationnel', '🟡 Maintenance', '🔴 Hors-ligne'], p=[0.8, 0.15, 0.05])
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"**{tel['name']}**")
                
                with col2:
                    st.write(status)
                
                with col3:
                    current_obs = "NGC 1234" if np.random.random() > 0.3 else "—"
                    st.write(f"Cible: {current_obs}")
                
                with col4:
                    if "Opérationnel" in status:
                        progress = np.random.uniform(0.1, 0.9)
                        st.progress(progress)
                    else:
                        st.write("—")
        else:
            st.info("Aucun télescope configuré")
        
        # Météo simulée
        st.write("### 🌤️ Conditions Observatoires")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            seeing = np.random.uniform(0.5, 2.5)
            st.metric("Seeing", f"{seeing:.2f}\"")
            if seeing < 1.0:
                st.success("Excellent")
            elif seeing < 1.5:
                st.info("Bon")
            else:
                st.warning("Moyen")
        
        with col2:
            humidity = np.random.uniform(20, 80)
            st.metric("Humidité", f"{humidity:.0f}%")
        
        with col3:
            cloud_cover = np.random.uniform(0, 100)
            st.metric("Couverture Nuages", f"{cloud_cover:.0f}%")
            if cloud_cover < 20:
                st.success("Clair")
            elif cloud_cover < 50:
                st.warning("Partiellement Nuageux")
            else:
                st.error("Couvert")
    
    with tab2:
        st.subheader("📊 Flux de Données")
        
        if st.button("🔄 Actualiser"):
            st.rerun()
        
        # Simuler flux temps réel
        data_rate = np.random.uniform(50, 500)
        st.metric("Taux Données Actuel", f"{data_rate:.1f} MB/s")
        
        # Graphique temps réel
        time_points = np.linspace(0, 60, 60)
        data_stream = 200 + 100 * np.sin(time_points / 10) + np.random.normal(0, 20, 60)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=data_stream,
            mode='lines',
            line=dict(color='#4ECDC4', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Flux Données (dernière minute)",
            xaxis_title="Temps (s)",
            yaxis_title="Débit (MB/s)",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alertes
        st.write("### 🚨 Alertes Système")
        
        alerts = [
            {'time': '23:45:32', 'level': 'INFO', 'message': 'Observation démarrée: NGC 4258'},
            {'time': '23:42:18', 'level': 'WARNING', 'message': 'Seeing dégradé: 1.8"'},
            {'time': '23:38:05', 'level': 'SUCCESS', 'message': 'Calibration complétée'},
        ]
        
        for alert in alerts:
            if alert['level'] == 'INFO':
                icon = "ℹ️"
            elif alert['level'] == 'WARNING':
                icon = "⚠️"
            else:
                icon = "✅"
            
            st.text(f"{icon} {alert['time']} - {alert['message']}")

# ==================== PAGE: SKY SURVEY ====================
elif page == "🗺️ Sky Survey":
    st.header("🗺️ Relevé Complet du Ciel")
    
    tab1, tab2, tab3 = st.tabs(["🌌 Grand Survey", "📍 Catalogues", "🔍 Recherche Objets"])
    
    with tab1:
        st.subheader("🌌 Lancer Sky Survey")
        
        st.write("""
        **Types de Surveys:**
        - **All-Sky:** Ciel entier (41,253 deg²)
        - **Deep Field:** Petit champ, très profond
        - **Time-Domain:** Monitoring répété
        - **Spectroscopic:** Redshifts + spectres
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            survey_type = st.selectbox("Type Survey",
                ["All-Sky Wide", "Deep Field", "Time-Domain", "Spectroscopic"])
            
            survey_depth = st.slider("Profondeur (magnitude limite)", 18, 30, 22)
        
        with col2:
            survey_bands = st.multiselect("Bandes Photométriques",
                ["u", "g", "r", "i", "z", "Y", "J", "H", "K"],
                default=["g", "r", "i"])
            
            cadence_days = st.slider("Cadence (jours)", 1, 30, 7)
        
        if st.button("🚀 Démarrer Survey", type="primary"):
            with st.spinner(f"Survey {survey_type} en cours..."):
                import time
                time.sleep(3)
                
                # Résultats simulés
                if survey_type == "All-Sky Wide":
                    area_covered = 41253  # deg²
                    objects_detected = np.random.randint(100000, 1000000)
                    transients = np.random.randint(100, 1000)
                elif survey_type == "Deep Field":
                    area_covered = 10  # deg²
                    objects_detected = np.random.randint(50000, 200000)
                    transients = np.random.randint(10, 50)
                elif survey_type == "Time-Domain":
                    area_covered = 1000
                    objects_detected = np.random.randint(10000, 50000)
                    transients = np.random.randint(50, 500)
                else:  # Spectroscopic
                    area_covered = 100
                    objects_detected = np.random.randint(5000, 20000)
                    transients = np.random.randint(5, 50)
                
                st.success("✅ Survey complété!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Surface", f"{area_covered} deg²")
                
                with col2:
                    st.metric("Objets Détectés", f"{objects_detected:,}")
                
                with col3:
                    st.metric("Transients", transients)
                
                with col4:
                    completeness = np.random.uniform(85, 98)
                    st.metric("Complétude", f"{completeness:.1f}%")
                
                # Visualisation distribution
                st.write("### 🗺️ Carte du Ciel - Objets Détectés")
                
                # Générer positions aléatoires
                n_plot = min(1000, objects_detected)
                ra = np.random.uniform(0, 360, n_plot)
                dec = np.random.uniform(-90, 90, n_plot)
                mags = np.random.uniform(15, survey_depth, n_plot)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scattergeo(
                    lon=ra - 180,
                    lat=dec,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=mags,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Magnitude"),
                        opacity=0.6
                    ),
                    hovertemplate='RA: %{lon}°<br>Dec: %{lat}°<extra></extra>'
                ))
                
                fig.update_geos(
                    projection_type='mollweide',
                    showcountries=False,
                    showcoastlines=False,
                    showland=False,
                    bgcolor='#0a0a0a'
                )
                
                fig.update_layout(
                    title="Distribution Objets (Projection Mollweide)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sauvegarder dans catalog
                survey_data = {
                    'survey_type': survey_type,
                    'area_deg2': area_covered,
                    'n_objects': objects_detected,
                    'n_transients': transients,
                    'depth_mag': survey_depth,
                    'bands': survey_bands,
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'surveys' not in st.session_state.telescope_lab:
                    st.session_state.telescope_lab['surveys'] = []
                
                st.session_state.telescope_lab['surveys'].append(survey_data)
                log_event(f"Survey complété: {survey_type}, {objects_detected:,} objets", "SUCCESS")
    
    with tab2:
        st.subheader("📍 Catalogues Générés")
        
        if 'surveys' in st.session_state.telescope_lab and st.session_state.telescope_lab['surveys']:
            st.write(f"### 📚 {len(st.session_state.telescope_lab['surveys'])} Surveys Effectués")
            
            survey_list = []
            for i, survey in enumerate(st.session_state.telescope_lab['surveys']):
                survey_list.append({
                    '#': i+1,
                    'Type': survey['survey_type'],
                    'Surface (deg²)': survey['area_deg2'],
                    'Objets': f"{survey['n_objects']:,}",
                    'Transients': survey['n_transients'],
                    'Profondeur': survey['depth_mag'],
                    'Date': survey['timestamp'][:10]
                })
            
            df_surveys = pd.DataFrame(survey_list)
            st.dataframe(df_surveys, use_container_width=True)
            
            # Statistiques cumulées
            total_objects = sum(s['n_objects'] for s in st.session_state.telescope_lab['surveys'])
            total_area = sum(s['area_deg2'] for s in st.session_state.telescope_lab['surveys'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Objets Catalogués Total", f"{total_objects:,}")
            with col2:
                st.metric("Surface Totale", f"{total_area:,} deg²")
            with col3:
                coverage = min(100, total_area / 41253 * 100)
                st.metric("Couverture Ciel", f"{coverage:.1f}%")
        else:
            st.info("Aucun survey effectué")
    
    with tab3:
        st.subheader("🔍 Recherche dans Catalogues")
        
        search_ra = st.number_input("RA (degrés)", 0.0, 360.0, 180.0)
        search_dec = st.number_input("Dec (degrés)", -90.0, 90.0, 0.0)
        search_radius = st.slider("Rayon Recherche (arcmin)", 1, 60, 10)
        
        if st.button("🔍 Rechercher Objets"):
            with st.spinner("Recherche dans catalogues..."):
                import time
                time.sleep(1)
                
                n_found = np.random.randint(0, 50)
                
                if n_found > 0:
                    st.success(f"✅ {n_found} objets trouvés dans {search_radius}' autour ({search_ra:.2f}, {search_dec:.2f})")
                    
                    # Générer objets fictifs
                    objects_found = []
                    for i in range(min(10, n_found)):
                        objects_found.append({
                            'ID': f'OBJ_{np.random.randint(100000, 999999)}',
                            'RA': f"{search_ra + np.random.uniform(-0.1, 0.1):.4f}",
                            'Dec': f"{search_dec + np.random.uniform(-0.1, 0.1):.4f}",
                            'Type': np.random.choice(['Étoile', 'Galaxie', 'Quasar', 'Nébuleuse']),
                            'Magnitude': f"{np.random.uniform(15, 22):.2f}",
                            'Redshift': f"{np.random.uniform(0, 2):.3f}"
                        })
                    
                    df_found = pd.DataFrame(objects_found)
                    st.dataframe(df_found, use_container_width=True)
                else:
                    st.info("Aucun objet trouvé dans cette région")

# ==================== PAGE: CATALOG ====================
elif page == "📚 Catalog":
    st.header("📚 Catalogue Complet des Observations")
    
    tab1, tab2, tab3 = st.tabs(["🌟 Toutes Observations", "🔬 Par Type", "📊 Export"])
    
    with tab1:
        st.subheader("🌟 Base de Données Observations")
        
        # Compter tous les objets
        total_items = (
            len(st.session_state.telescope_lab['telescopes']) +
            len(st.session_state.telescope_lab['targets']) +
            len(st.session_state.telescope_lab['observations']) +
            len(st.session_state.telescope_lab['exoplanet_candidates']) +
            len(st.session_state.telescope_lab['galaxy_catalog'])
        )
        
        st.metric("📦 Total Entrées Catalogue", f"{total_items:,}")
        
        # Aperçu global
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🔭", len(st.session_state.telescope_lab['telescopes']))
            st.caption("Télescopes")
        
        with col2:
            st.metric("🎯", len(st.session_state.telescope_lab['targets']))
            st.caption("Cibles")
        
        with col3:
            st.metric("📸", len(st.session_state.telescope_lab['observations']))
            st.caption("Observations")
        
        with col4:
            st.metric("🪐", len(st.session_state.telescope_lab['exoplanet_candidates']))
            st.caption("Exoplanètes")
        
        with col5:
            st.metric("🌌", len(st.session_state.telescope_lab['galaxy_catalog']))
            st.caption("Galaxies")
    
    with tab2:
        st.subheader("🔬 Filtrer par Type")
        
        catalog_type = st.selectbox("Catégorie",
            ["Télescopes", "Cibles", "Observations", "Exoplanètes", "Galaxies", "Spectres"])
        
        if catalog_type == "Télescopes":
            if st.session_state.telescope_lab['telescopes']:
                tel_data = []
                for tel_id, tel in st.session_state.telescope_lab['telescopes'].items():
                    tel_data.append({
                        'ID': tel_id,
                        'Nom': tel['name'],
                        'Type': tel['type'],
                        'Diamètre (m)': tel['diameter_m'],
                        'Résolution (arcsec)': f"{tel['resolution_arcsec']:.3f}",
                        'Mag Limite': f"{tel['limiting_magnitude']:.1f}"
                    })
                
                df = pd.DataFrame(tel_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucun télescope")
        
        elif catalog_type == "Observations":
            if st.session_state.telescope_lab['observations']:
                obs_data = []
                for obs in st.session_state.telescope_lab['observations'][:50]:  # Limiter affichage
                    obs_data.append({
                        'Télescope': obs['telescope_id'],
                        'Cible': obs['target_id'],
                        'Mode': obs['mode'],
                        'Exposition (s)': obs['exposure_time_s'],
                        'SNR': f"{obs['snr']:.1f}",
                        'Date': obs['timestamp'][:19]
                    })
                
                df = pd.DataFrame(obs_data)
                st.dataframe(df, use_container_width=True)
                
                if len(st.session_state.telescope_lab['observations']) > 50:
                    st.info(f"Affichage des 50 premières sur {len(st.session_state.telescope_lab['observations'])} observations")
            else:
                st.info("Aucune observation")
        
        elif catalog_type == "Exoplanètes":
            if st.session_state.telescope_lab['exoplanet_candidates']:
                exo_data = []
                for i, exo in enumerate(st.session_state.telescope_lab['exoplanet_candidates']):
                    exo_data.append({
                        'ID': f"EXO_{i+1:03d}",
                        'Rayon (R⊕)': f"{exo['radius_r_earth']:.2f}",
                        'Période (j)': f"{exo['period_days']:.1f}",
                        'T_eq (K)': f"{exo.get('equilibrium_temp_K', 0):.0f}",
                        'Méthode': exo['detection_method'],
                        'Confirmé': '✅' if exo['confirmed'] else '⏳'
                    })
                
                df = pd.DataFrame(exo_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucune exoplanète")
        
        elif catalog_type == "Galaxies":
            if st.session_state.telescope_lab['galaxy_catalog']:
                gal_data = []
                for gal in st.session_state.telescope_lab['galaxy_catalog'][:50]:
                    gal_data.append({
                        'ID': gal['id'],
                        'Type': gal['type'],
                        'Magnitude': f"{gal['magnitude']:.2f}",
                        'Redshift': f"{gal['redshift']:.3f}",
                        'Distance (Mpc)': f"{gal['distance_Mpc']:.1f}",
                        'Masse (M☉)': f"{gal['mass_Msun']:.2e}"
                    })
                
                df = pd.DataFrame(gal_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucune galaxie")
    
    with tab3:
        st.subheader("📊 Export Données")
        
        st.write("### 💾 Formats Disponibles")
        
        export_format = st.selectbox("Format",
            ["CSV", "JSON", "FITS (simulation)", "VOTable"])
        
        export_category = st.selectbox("Catégorie à Exporter",
            ["Tout", "Télescopes", "Observations", "Exoplanètes", "Galaxies"])
        
        if st.button("📥 Générer Export"):
            with st.spinner("Génération fichier..."):
                import time
                time.sleep(1)
                
                # Simuler export
                n_entries = 0
                
                if export_category == "Tout":
                    n_entries = (
                        len(st.session_state.telescope_lab['telescopes']) +
                        len(st.session_state.telescope_lab['observations']) +
                        len(st.session_state.telescope_lab['exoplanet_candidates']) +
                        len(st.session_state.telescope_lab['galaxy_catalog'])
                    )
                elif export_category == "Télescopes":
                    n_entries = len(st.session_state.telescope_lab['telescopes'])
                elif export_category == "Observations":
                    n_entries = len(st.session_state.telescope_lab['observations'])
                elif export_category == "Exoplanètes":
                    n_entries = len(st.session_state.telescope_lab['exoplanet_candidates'])
                else:
                    n_entries = len(st.session_state.telescope_lab['galaxy_catalog'])
                
                file_size = n_entries * 0.5  # KB par entrée (approximation)
                
                st.success(f"✅ Export généré!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Entrées", n_entries)
                
                with col2:
                    st.metric("Taille", f"{file_size:.1f} KB")
                
                with col3:
                    st.metric("Format", export_format)
                
                st.download_button(
                    label="📥 Télécharger (Simulation)",
                    data=f"# Catalogue Export - {export_category}\n# Format: {export_format}\n# Entrées: {n_entries}",
                    file_name=f"telescope_catalog_{export_category.lower()}.{export_format.lower()}",
                    mime="text/plain"
                )

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Configuration & Paramètres")
    
    tab1, tab2, tab3 = st.tabs(["🎨 Interface", "💾 Données", "🔧 Avancé"])
    
    with tab1:
        st.subheader("🎨 Personnalisation Interface")
        
        theme = st.selectbox("Thème Couleurs",
            ["Cosmic Blue (Défaut)", "Deep Space Dark", "Nebula Purple", "Solar Orange"])
        
        st.info(f"Thème sélectionné: {theme}")
        
        chart_style = st.selectbox("Style Graphiques",
            ["plotly_dark (Défaut)", "plotly", "seaborn", "ggplot2"])
        
        font_size = st.slider("Taille Police", 10, 20, 14)
        
        st.write(f"Aperçu taille: **Police {font_size}px**")
        
        if st.button("💾 Sauvegarder Préférences"):
            st.success("✅ Préférences sauvegardées!")
    
    with tab2:
        st.subheader("💾 Gestion Données")
        
        st.write("### 📊 Stockage Actuel")
        
        storage_info = {
            'Télescopes': len(st.session_state.telescope_lab['telescopes']),
            'Cibles': len(st.session_state.telescope_lab['targets']),
            'Observations': len(st.session_state.telescope_lab['observations']),
            'Spectres': len(st.session_state.telescope_lab['spectra']),
            'Exoplanètes': len(st.session_state.telescope_lab['exoplanet_candidates']),
            'Galaxies': len(st.session_state.telescope_lab['galaxy_catalog']),
            'Logs': len(st.session_state.telescope_lab['log'])
        }
        
        for category, count in storage_info.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{category}:**")
            with col2:
                st.write(f"{count} entrées")
        
        st.write("---")
        
        st.warning("⚠️ Zone Danger")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Effacer Observations", type="secondary"):
                st.session_state.telescope_lab['observations'] = []
                log_event("Observations effacées", "WARNING")
                st.success("Observations effacées")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Réinitialiser TOUT", type="secondary"):
                if st.checkbox("Confirmer réinitialisation complète"):
                    st.session_state.telescope_lab = {
                        'telescopes': {},
                        'observations': [],
                        'discoveries': [],
                        'targets': {},
                        'images': [],
                        'spectra': [],
                        'ai_detections': [],
                        'quantum_analysis': [],
                        'exoplanet_candidates': [],
                        'galaxy_catalog': [],
                        'monitoring_campaigns': [],
                        'collaborations': [],
                        'log': []
                    }
                    st.success("✅ Base de données réinitialisée")
                    st.rerun()
    
    with tab3:
        st.subheader("🔧 Paramètres Avancés")
        
        st.write("### 🔬 Précision Calculs")
        
        precision = st.select_slider("Précision Numérique",
            options=["Standard", "Haute", "Très Haute", "Maximum"],
            value="Haute")
        
        st.write(f"Mode: **{precision}**")
        
        st.write("### 📡 API & Intégrations")
        
        enable_api = st.checkbox("Activer API REST", value=False)
        
        if enable_api:
            api_port = st.number_input("Port API", 8000, 9000, 8020)
            st.code(f"API disponible sur: http://localhost:{api_port}")
            st.info("L'API FastAPI doit être lancée séparément avec le fichier fourni")
        
        st.write("### 🔐 Sécurité")
        
        require_auth = st.checkbox("Requérir Authentification", value=False)
        
        if require_auth:
            st.info("L'authentification nécessite la configuration de l'API backend")
        
        st.write("### 📊 Performance")
        
        cache_enabled = st.checkbox("Activer Cache", value=True)
        max_cache_size = st.slider("Taille Max Cache (MB)", 100, 1000, 500)
        
        if st.button("⚙️ Appliquer Paramètres Avancés"):
            st.success("✅ Paramètres appliqués!")
# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal Système (20 dernières entrées)"):
    if st.session_state.telescope_lab['log']:
        for event in st.session_state.telescope_lab['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔭 Télescopes", total_telescopes)

with col2:
    st.metric("📸 Observations", total_observations)

with col3:
    st.metric("🌟 Découvertes", total_discoveries)

with col4:
    st.metric("🪐 Exoplanètes", len(st.session_state.telescope_lab['exoplanet_candidates']))

st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🔭 Advanced Space Telescope Laboratory</h3>
        <p>Deep Space Observation • Exoplanets • Galaxies • Black Holes</p>
        <p><small>AI Detection • Quantum Analysis • Bioastronomy • Multi-Messenger</small></p>
        <p><small>Version 1.0.0 | Research Edition</small></p>
        <p><small>🌌 Exploring the Universe © 2024</small></p>
    </div>
""", unsafe_allow_html=True)