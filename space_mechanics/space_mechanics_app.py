"""
Interface Streamlit pour la Plateforme de Mécanique Spatiale
Système intégré pour créer, simuler et analyser
missions spatiales, orbites, satellites et trajectoires
streamlit run space_mechanics_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import numpy as np

# ==================== CONFIGURATION PAGE ====================
st.set_page_config(
    page_title="🚀 Plateforme Mécanique Spatiale",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLES CSS ====================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .satellite-card {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .orbit-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ====================
CONSTANTS = {
    'G': 6.67430e-11,
    'earth_mu': 3.986004418e14,
    'earth_radius': 6371000,
    'earth_mass': 5.972e24,
    'moon_distance': 384400000,
    'c': 299792458,
    'SUN_MASS': 1.98847e30,
    'EARTH_MASS': 5.9722e24,
    'MOON_MASS': 7.34767309e22,     # kg — masse de la Lune
    'G': 6.67430e-11,               # m^3 kg^-1 s^-2 — constante gravitationnelle
    'AU': 1.495978707e11,           # m — unité astronomique
    'C': 299792458,                 # m/s — vitesse de la lumière
    'EARTH_RADIUS': 6.371e6,        # m — rayon moyen de la Terre
    'SUN_RADIUS': 6.9634e8, 
}

# ==================== INITIALISATION SESSION STATE ====================
if 'space_system' not in st.session_state:
    st.session_state.space_system = {
        'satellites': {},
        'missions': {},
        'orbits': {},
        'maneuvers': [],
        'simulations': [],
        'ground_stations': {},
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================
def log_event(message: str):
    """Enregistre un événement"""
    st.session_state.space_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })

def get_orbit_badge(orbit_type: str) -> str:
    """Retourne un badge HTML pour le type d'orbite"""
    badges = {
        'LEO': '<span class="orbit-badge">🛰️ LEO</span>',
        'MEO': '<span class="orbit-badge">🛰️ MEO</span>',
        'GEO': '<span class="orbit-badge">🛰️ GEO</span>',
        'POLAR': '<span class="orbit-badge">🧭 Polaire</span>',
        'SSO': '<span class="orbit-badge">☀️ Héliosynchrone</span>',
    }
    return badges.get(orbit_type, '<span class="orbit-badge">🛰️</span>')

def create_satellite_mock(name, config):
    """Crée un satellite simulé"""
    sat_id = f"sat_{len(st.session_state.space_system['satellites']) + 1}"
    
    satellite = {
        'id': sat_id,
        'name': name,
        'created_at': datetime.now().isoformat(),
        'status': 'inactive',
        'masses': {
            'dry_mass': config.get('dry_mass', 1000),
            'propellant_mass': config.get('propellant_mass', 500),
            'payload_mass': config.get('payload_mass', 200),
            'total_mass': config.get('dry_mass', 1000) + config.get('propellant_mass', 500) + config.get('payload_mass', 200)
        },
        'dimensions': {
            'length': config.get('length', 2.0),
            'width': config.get('width', 2.0),
            'height': config.get('height', 3.0),
            'solar_panel_area': config.get('solar_area', 10.0)
        },
        'power': {
            'generation': config.get('power_gen', 5000),
            'battery_capacity': config.get('battery', 50000)
        },
        'propulsion': {
            'type': config.get('propulsion_type', 'chimique'),
            'isp': config.get('isp', 300),
            'thrust': config.get('thrust', 1000)
        },
        'orbit': config.get('orbit_id', None),
        'mission': {
            'type': config.get('mission_type', 'observation'),
            'lifetime_years': config.get('lifetime', 5),
            'operational_hours': 0.0
        },
        'mission': {
            'type': config.get('mission_type', 'observation'),
            'lifetime_years': config.get('lifetime', 5),
            'operational_hours': 0.0
        },
        'telemetry': {
            'altitude': 0.0,
            'velocity': 0.0,
            'latitude': 0.0,
            'longitude': 0.0,
            'battery_level': 100.0
        },
        'performance': {
            'data_transmitted': 0.0,  # GB
            'orbits_completed': 0,
            'maneuvers_executed': 0
        }
    }
    
    st.session_state.space_system['satellites'][sat_id] = satellite
    log_event(f"Satellite créé: {name}")
    return sat_id

def create_orbit_mock(name, orbital_elements):
    """Crée une orbite simulée"""
    orbit_id = f"orbit_{len(st.session_state.space_system['orbits']) + 1}"
    
    a = orbital_elements.get('semi_major_axis', 7000000)
    e = orbital_elements.get('eccentricity', 0.0)
    mu = CONSTANTS['earth_mu']
    
    # Calculs orbitaux
    period = 2 * np.pi * np.sqrt(a**3 / mu)
    periapsis = a * (1 - e)
    apoapsis = a * (1 + e)
    v_peri = np.sqrt(mu * (1 + e) / (a * (1 - e)))
    v_apo = np.sqrt(mu * (1 - e) / (a * (1 + e)))
    
    orbit = {
        'id': orbit_id,
        'name': name,
        'created_at': datetime.now().isoformat(),
        'elements': {
            'semi_major_axis': a,
            'eccentricity': e,
            'inclination': orbital_elements.get('inclination', 0.0),
            'raan': orbital_elements.get('raan', 0.0),
            'arg_periapsis': orbital_elements.get('arg_periapsis', 0.0),
            'true_anomaly': orbital_elements.get('true_anomaly', 0.0)
        },
        'parameters': {
            'period': period,
            'periapsis': periapsis,
            'apoapsis': apoapsis,
            'altitude_peri': periapsis - CONSTANTS['earth_radius'],
            'altitude_apo': apoapsis - CONSTANTS['earth_radius'],
            'velocity_peri': v_peri,
            'velocity_apo': v_apo
        },
        'type': orbital_elements.get('orbit_type', 'LEO')
    }
    
    st.session_state.space_system['orbits'][orbit_id] = orbit
    log_event(f"Orbite créée: {name}")
    return orbit_id

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🚀 Plateforme de Mécanique Spatiale</h1>', unsafe_allow_html=True)
st.markdown("### Système Intégré pour Missions Spatiales, Orbites et Trajectoires")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Space+Mechanics", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Tableau de Bord",
            "🛰️ Mes Satellites",
            "➕ Créer Satellite",
            "🌍 Orbites",
            "📐 Calculs Orbitaux",
            "🚀 Manœuvres",
            "⚡ Propulsion",
            "📡 Trajectoires",
            "🎯 Transferts",
            "🌙 Missions Lunaires",
            "🔴 Missions Mars",
            "📊 Simulations",
            "🗺️ Trace au Sol",
            "📡 Stations Sol",
            "🔭 Rendez-vous",
            "💫 Points Lagrange",
            "⏱️ Fenêtres Lancement",
            "📈 Analyses",
            "🌌 Espace Profond",
            "📚 Documentation"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_satellites = len(st.session_state.space_system['satellites'])
    active_satellites = sum(1 for s in st.session_state.space_system['satellites'].values() if s['status'] == 'active')
    total_missions = len(st.session_state.space_system['missions'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🛰️ Satellites", total_satellites)
        st.metric("🎯 Missions", total_missions)
    with col2:
        st.metric("✅ Actifs", active_satellites)
        total_orbits = len(st.session_state.space_system['orbits'])
        st.metric("🌍 Orbites", total_orbits)

# ==================== PAGE: TABLEAU DE BORD ====================
if page == "🏠 Tableau de Bord":
    st.header("📊 Tableau de Bord Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="satellite-card"><h2>🛰️</h2><h3>{total_satellites}</h3><p>Satellites</p></div>', unsafe_allow_html=True)
    
    with col2:
        total_orbits = len(st.session_state.space_system['orbits'])
        st.markdown(f'<div class="satellite-card"><h2>🌍</h2><h3>{total_orbits}</h3><p>Orbites</p></div>', unsafe_allow_html=True)
    
    with col3:
        total_maneuvers = len(st.session_state.space_system['maneuvers'])
        st.markdown(f'<div class="satellite-card"><h2>🚀</h2><h3>{total_maneuvers}</h3><p>Manœuvres</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="satellite-card"><h2>📡</h2><h3>{total_missions}</h3><p>Missions</p></div>', unsafe_allow_html=True)
    
    with col5:
        total_data = sum(s['performance']['data_transmitted'] for s in st.session_state.space_system['satellites'].values())
        st.markdown(f'<div class="satellite-card"><h2>💾</h2><h3>{total_data:.1f}</h3><p>TB Données</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Constantes fondamentales
    st.subheader("⚛️ Constantes Fondamentales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("G", "6.674×10⁻¹¹ m³/kg/s²")
        st.metric("Vitesse lumière", "299,792,458 m/s")
    
    with col2:
        st.metric("μ Terre", "3.986×10¹⁴ m³/s²")
        st.metric("Rayon Terre", "6,371 km")
    
    with col3:
        st.metric("Masse Terre", "5.972×10²⁴ kg")
        st.metric("Période rotation", "23h 56min 4s")
    
    with col4:
        st.metric("Unité Astronomique", "1.496×10⁸ km")
        st.metric("Distance Lune", "384,400 km")
    
    st.markdown("---")
    
    if st.session_state.space_system['satellites']:
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛰️ Satellites par Type Mission")
            
            mission_types = {}
            for sat in st.session_state.space_system['satellites'].values():
                m_type = sat['mission']['type']
                mission_types[m_type] = mission_types.get(m_type, 0) + 1
            
            fig = px.pie(values=list(mission_types.values()), 
                        names=list(mission_types.keys()),
                        title="Répartition par Mission")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌍 Orbites par Type")
            
            orbit_types = {}
            for orb in st.session_state.space_system['orbits'].values():
                o_type = orb['type']
                orbit_types[o_type] = orbit_types.get(o_type, 0) + 1
            
            if orbit_types:
                fig = px.bar(x=list(orbit_types.keys()), 
                           y=list(orbit_types.values()),
                           title="Distribution des Orbites")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Aucun satellite créé. Créez votre premier satellite!")

# ==================== PAGE: MES SATELLITES ====================
elif page == "🛰️ Mes Satellites":
    st.header("🛰️ Gestion des Satellites")
    
    if not st.session_state.space_system['satellites']:
        st.info("💡 Aucun satellite créé.")
    else:
        for sat_id, satellite in st.session_state.space_system['satellites'].items():
            st.markdown(f'<div class="satellite-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### 🛰️ {satellite['name']}")
                st.write(f"**Type Mission:** {satellite['mission']['type']}")
                status_icon = "🟢" if satellite['status'] == 'active' else "🔴"
                st.write(f"**Statut:** {status_icon} {satellite['status']}")
            
            with col2:
                st.metric("Masse Totale", f"{satellite['masses']['total_mass']:.0f} kg")
                st.metric("Masse Sèche", f"{satellite['masses']['dry_mass']:.0f} kg")
            
            with col3:
                st.metric("Puissance", f"{satellite['power']['generation']} W")
                st.metric("Batterie", f"{satellite['telemetry']['battery_level']:.0f}%")
            
            with col4:
                st.metric("Orbites", satellite['performance']['orbits_completed'])
                st.metric("Données", f"{satellite['performance']['data_transmitted']:.1f} GB")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Spécifications", "📡 Télémétrie", "🚀 Propulsion", "📊 Performance"])
                
                with tab1:
                    st.subheader("⚙️ Spécifications")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Masses:**")
                        st.write(f"• Sèche: {satellite['masses']['dry_mass']} kg")
                        st.write(f"• Propergol: {satellite['masses']['propellant_mass']} kg")
                        st.write(f"• Charge utile: {satellite['masses']['payload_mass']} kg")
                        st.write(f"• Totale: {satellite['masses']['total_mass']} kg")
                    
                    with col2:
                        st.write("**Dimensions:**")
                        st.write(f"• Longueur: {satellite['dimensions']['length']} m")
                        st.write(f"• Largeur: {satellite['dimensions']['width']} m")
                        st.write(f"• Hauteur: {satellite['dimensions']['height']} m")
                        st.write(f"• Panneaux solaires: {satellite['dimensions']['solar_panel_area']} m²")
                
                with tab2:
                    st.subheader("📡 Télémétrie")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Altitude", f"{satellite['telemetry']['altitude']/1000:.0f} km")
                        st.metric("Vitesse", f"{satellite['telemetry']['velocity']/1000:.2f} km/s")
                    
                    with col2:
                        st.metric("Latitude", f"{satellite['telemetry']['latitude']:.2f}°")
                        st.metric("Longitude", f"{satellite['telemetry']['longitude']:.2f}°")
                    
                    with col3:
                        st.metric("Batterie", f"{satellite['telemetry']['battery_level']:.1f}%")
                        st.progress(satellite['telemetry']['battery_level'] / 100)
                
                with tab3:
                    st.subheader("🚀 Système Propulsion")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {satellite['propulsion']['type']}")
                        st.metric("Isp", f"{satellite['propulsion']['isp']} s")
                    
                    with col2:
                        st.metric("Poussée", f"{satellite['propulsion']['thrust']} N")
                        
                        # Calcul delta-v
                        g0 = 9.80665
                        ve = satellite['propulsion']['isp'] * g0
                        m0 = satellite['masses']['total_mass']
                        mf = m0 - satellite['masses']['propellant_mass']
                        if mf > 0:
                            dv = ve * np.log(m0 / mf)
                            st.metric("Delta-v", f"{dv:.0f} m/s")
                
                with tab4:
                    st.subheader("📊 Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Heures Opération", f"{satellite['mission']['operational_hours']:.0f}h")
                    with col2:
                        st.metric("Orbites Complétées", satellite['performance']['orbits_completed'])
                    with col3:
                        st.metric("Manœuvres", satellite['performance']['maneuvers_executed'])
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button(f"▶️ {'Désactiver' if satellite['status'] == 'active' else 'Activer'}", key=f"toggle_{sat_id}"):
                        satellite['status'] = 'inactive' if satellite['status'] == 'active' else 'active'
                        log_event(f"{satellite['name']} {'désactivé' if satellite['status'] == 'inactive' else 'activé'}")
                        st.rerun()
                
                with col2:
                    if st.button(f"🚀 Manœuvre", key=f"maneuver_{sat_id}"):
                        st.info("Allez dans Manœuvres")
                
                with col3:
                    if st.button(f"📡 Télécommande", key=f"telecommand_{sat_id}"):
                        st.success("Télécommande envoyée")
                
                with col4:
                    if st.button(f"🗑️ Supprimer", key=f"del_{sat_id}"):
                        del st.session_state.space_system['satellites'][sat_id]
                        log_event(f"{satellite['name']} supprimé")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CRÉER SATELLITE ====================
elif page == "➕ Créer Satellite":
    st.header("➕ Créer un Nouveau Satellite")
    
    with st.form("create_satellite_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sat_name = st.text_input("📝 Nom du Satellite", placeholder="Ex: ObservationSat-1")
            
            mission_type = st.selectbox(
                "🎯 Type de Mission",
                ["observation", "communication", "navigation", "scientifique", 
                 "exploration", "militaire", "météo"]
            )
        
        with col2:
            orbit_type = st.selectbox(
                "🌍 Type d'Orbite",
                ["LEO", "MEO", "GEO", "Polaire", "Héliosynchrone", "Molniya", "Lunaire"]
            )
            
            lifetime = st.number_input("⏱️ Durée de Vie (années)", 1, 30, 5, 1)
        
        st.markdown("---")
        st.subheader("⚖️ Masses")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dry_mass = st.number_input("Masse Sèche (kg)", 100.0, 50000.0, 1000.0, 100.0)
        
        with col2:
            propellant_mass = st.number_input("Masse Propergol (kg)", 0.0, 20000.0, 500.0, 50.0)
        
        with col3:
            payload_mass = st.number_input("Masse Charge Utile (kg)", 10.0, 10000.0, 200.0, 10.0)
        
        total_mass = dry_mass + propellant_mass + payload_mass
        st.metric("Masse Totale", f"{total_mass:.0f} kg")
        
        st.markdown("---")
        st.subheader("📐 Dimensions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            length = st.number_input("Longueur (m)", 0.1, 50.0, 2.0, 0.1)
        with col2:
            width = st.number_input("Largeur (m)", 0.1, 50.0, 2.0, 0.1)
        with col3:
            height = st.number_input("Hauteur (m)", 0.1, 50.0, 3.0, 0.1)
        with col4:
            solar_area = st.number_input("Surface Panneaux (m²)", 1.0, 200.0, 10.0, 1.0)
        
        st.markdown("---")
        st.subheader("⚡ Énergie")
        
        col1, col2 = st.columns(2)
        
        with col1:
            power_gen = st.number_input("Puissance Générée (W)", 100, 50000, 5000, 100)
        
        with col2:
            battery = st.number_input("Capacité Batterie (Wh)", 1000, 500000, 50000, 1000)
        
        st.markdown("---")
        st.subheader("🚀 Propulsion")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            propulsion_type = st.selectbox("Type", 
                ["chimique", "electrique", "ionique", "effet_hall", "gaz_froid"])
        
        with col2:
            isp_dict = {
                "chimique": 300,
                "electrique": 3000,
                "ionique": 3500,
                "effet_hall": 1600,
                "gaz_froid": 70
            }
            isp = st.number_input("Isp (s)", 50, 5000, isp_dict[propulsion_type], 10)
        
        with col3:
            thrust = st.number_input("Poussée (N)", 0.001, 1000000.0, 1000.0, 1.0)
        
        # Calcul delta-v
        g0 = 9.80665
        ve = isp * g0
        if total_mass > dry_mass + payload_mass:
            dv = ve * np.log(total_mass / (dry_mass + payload_mass))
            st.metric("Delta-v Disponible", f"{dv:.0f} m/s")
        
        st.markdown("---")
        
        # Résumé
        st.subheader("📊 Résumé")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Masse", f"{total_mass:.0f} kg")
        with col2:
            st.metric("Puissance", f"{power_gen} W")
        with col3:
            st.metric("Delta-v", f"{dv:.0f} m/s" if 'dv' in locals() else "N/A")
        with col4:
            st.metric("Durée", f"{lifetime} ans")
        
        submitted = st.form_submit_button("🚀 Créer le Satellite", use_container_width=True, type="primary")
        
        if submitted:
            if not sat_name:
                st.error("⚠️ Veuillez donner un nom au satellite")
            else:
                with st.spinner("🔄 Création du satellite en cours..."):
                    config = {
                        'dry_mass': dry_mass,
                        'propellant_mass': propellant_mass,
                        'payload_mass': payload_mass,
                        'length': length,
                        'width': width,
                        'height': height,
                        'solar_area': solar_area,
                        'power_gen': power_gen,
                        'battery': battery,
                        'propulsion_type': propulsion_type,
                        'isp': isp,
                        'thrust': thrust,
                        'mission_type': mission_type,
                        'lifetime': lifetime
                    }
                    
                    sat_id = create_satellite_mock(sat_name, config)
                    
                    st.success(f"✅ Satellite '{sat_name}' créé avec succès!")
                    st.balloons()
                    
                    satellite = st.session_state.space_system['satellites'][sat_id]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("ID", sat_id)
                    with col2:
                        st.metric("Masse", f"{satellite['masses']['total_mass']:.0f} kg")
                    with col3:
                        st.metric("Puissance", f"{satellite['power']['generation']} W")
                    with col4:
                        st.metric("Type", mission_type)

# ==================== PAGE: ORBITES ====================
elif page == "🌍 Orbites":
    st.header("🌍 Gestion des Orbites")
    
    tab1, tab2, tab3 = st.tabs(["📊 Mes Orbites", "➕ Créer Orbite", "📚 Types d'Orbites"])
    
    with tab1:
        st.subheader("📊 Orbites Créées")
        
        if st.session_state.space_system['orbits']:
            for orbit_id, orbit in st.session_state.space_system['orbits'].items():
                with st.expander(f"🌍 {orbit['name']} - {orbit['type']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Demi-grand axe", f"{orbit['elements']['semi_major_axis']/1000:.0f} km")
                        st.metric("Excentricité", f"{orbit['elements']['eccentricity']:.4f}")
                    
                    with col2:
                        st.metric("Altitude Périgée", f"{orbit['parameters']['altitude_peri']/1000:.0f} km")
                        st.metric("Altitude Apogée", f"{orbit['parameters']['altitude_apo']/1000:.0f} km")
                    
                    with col3:
                        st.metric("Période", f"{orbit['parameters']['period']/60:.2f} min")
                        st.metric("Vitesse", f"{orbit['parameters']['velocity_peri']/1000:.2f} km/s")
        else:
            st.info("Aucune orbite créée")
    
    with tab2:
        st.subheader("➕ Créer une Nouvelle Orbite")
        
        with st.form("create_orbit_form"):
            orbit_name = st.text_input("Nom de l'Orbite", "Orbite LEO 500km")
            
            col1, col2 = st.columns(2)
            
            with col1:
                altitude = st.number_input("Altitude (km)", 200.0, 100000.0, 500.0, 10.0)
                eccentricity = st.slider("Excentricité", 0.0, 0.9, 0.0, 0.01)
            
            with col2:
                inclination = st.slider("Inclinaison (°)", 0.0, 180.0, 0.0, 1.0)
                orbit_type_sel = st.selectbox("Type", ["LEO", "MEO", "GEO", "POLAR", "SSO"])
            
            # Éléments avancés
            with st.expander("⚙️ Éléments Orbitaux Avancés"):
                raan = st.slider("RAAN (°)", 0.0, 360.0, 0.0, 1.0)
                arg_periapsis = st.slider("Argument Périapside (°)", 0.0, 360.0, 0.0, 1.0)
                true_anomaly = st.slider("Anomalie Vraie (°)", 0.0, 360.0, 0.0, 1.0)
            
            submitted_orbit = st.form_submit_button("🌍 Créer Orbite", type="primary")
            
            if submitted_orbit:
                # Calcul demi-grand axe
                r_earth = CONSTANTS['earth_radius']
                semi_major_axis = r_earth + altitude * 1000
                
                orbital_elements = {
                    'semi_major_axis': semi_major_axis,
                    'eccentricity': eccentricity,
                    'inclination': inclination,
                    'raan': raan if 'raan' in locals() else 0.0,
                    'arg_periapsis': arg_periapsis if 'arg_periapsis' in locals() else 0.0,
                    'true_anomaly': true_anomaly if 'true_anomaly' in locals() else 0.0,
                    'orbit_type': orbit_type_sel
                }
                
                orbit_id = create_orbit_mock(orbit_name, orbital_elements)
                
                st.success(f"✅ Orbite '{orbit_name}' créée!")
                
                orbit = st.session_state.space_system['orbits'][orbit_id]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Période", f"{orbit['parameters']['period']/60:.2f} min")
                with col2:
                    st.metric("Vitesse", f"{orbit['parameters']['velocity_peri']/1000:.2f} km/s")
                with col3:
                    st.metric("Type", orbit['type'])
    
    with tab3:
        st.subheader("📚 Types d'Orbites")
        
        orbit_types_info = {
            "LEO (Low Earth Orbit)": {
                "altitude": "200-2000 km",
                "période": "90-130 min",
                "utilisation": "Observation Terre, ISS, satellites reconnaissance",
                "avantages": "Faible latence, résolution élevée, faible coût lancement",
                "inconvénients": "Traînée atmosphérique, nécessite constellation pour couverture"
            },
            "MEO (Medium Earth Orbit)": {
                "altitude": "2,000-35,786 km",
                "période": "2-12 heures",
                "utilisation": "Navigation (GPS, Galileo, GLONASS)",
                "avantages": "Bon compromis couverture/latence",
                "inconvénients": "Ceintures Van Allen (radiations)"
            },
            "GEO (Geostationary)": {
                "altitude": "35,786 km",
                "période": "24 heures (synchrone)",
                "utilisation": "Communications, météo",
                "avantages": "Position fixe dans le ciel, couverture continue",
                "inconvénients": "Latence élevée (250ms), coût lancement"
            },
            "Polaire": {
                "altitude": "Variable",
                "inclinaison": "~90°",
                "utilisation": "Observation globale, reconnaissance",
                "avantages": "Couverture complète Terre",
                "inconvénients": "Pas de couverture continue point fixe"
            },
            "Héliosynchrone (SSO)": {
                "altitude": "600-800 km",
                "inclinaison": "~98°",
                "utilisation": "Observation Terre, météo",
                "avantages": "Éclairage solaire constant",
                "inconvénients": "Altitude et inclinaison contraintes"
            }
        }
        
        for orbit_name, orbit_info in orbit_types_info.items():
            with st.expander(f"🌍 {orbit_name}"):
                for key, value in orbit_info.items():
                    st.write(f"**{key.title()}:** {value}")

# ==================== PAGE: CALCULS ORBITAUX ====================
elif page == "📐 Calculs Orbitaux":
    st.header("📐 Calculs Orbitaux")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Vitesse Orbitale", "⏱️ Période", "⚡ Énergie", "🎯 Équation Vis-Viva"])
    
    with tab1:
        st.subheader("🌍 Calcul Vitesse Orbitale")
        
        st.latex(r"v = \sqrt{\frac{\mu}{r}}")
        
        with st.form("velocity_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                body = st.selectbox("Corps Central", ["Terre", "Lune", "Mars", "Soleil"])
                mu_dict = {
                    "Terre": 3.986004418e14,
                    "Lune": 4.9028e12,
                    "Mars": 4.282837e13,
                    "Soleil": 1.32712440018e20
                }
                mu = mu_dict[body]
                st.metric("μ", f"{mu:.3e} m³/s²")
            
            with col2:
                altitude_v = st.number_input("Altitude (km)", 100.0, 100000.0, 500.0, 10.0)
                
                r_dict = {
                    "Terre": 6371000,
                    "Lune": 1737400,
                    "Mars": 3389500,
                    "Soleil": 696000000
                }
                r_body = r_dict[body]
                r = r_body + altitude_v * 1000
            
            submitted_v = st.form_submit_button("🔬 Calculer")
            
            if submitted_v:
                v = np.sqrt(mu / r)
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vitesse", f"{v:.0f} m/s")
                with col2:
                    st.metric("Vitesse", f"{v/1000:.2f} km/s")
                with col3:
                    period = 2 * np.pi * r / v
                    st.metric("Période", f"{period/60:.2f} min")
    
    with tab2:
        st.subheader("⏱️ Calcul Période Orbitale")
        
        st.latex(r"T = 2\pi\sqrt{\frac{a^3}{\mu}}")
        
        with st.form("period_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                body_p = st.selectbox("Corps", ["Terre", "Lune", "Mars"], key="body_period")
                mu_p = mu_dict[body_p]
            
            with col2:
                altitude_p = st.number_input("Altitude (km)", 100.0, 100000.0, 500.0, 10.0, key="alt_period")
                r_p = r_dict[body_p] + altitude_p * 1000
            
            submitted_p = st.form_submit_button("🔬 Calculer")
            
            if submitted_p:
                T = 2 * np.pi * np.sqrt(r_p**3 / mu_p)
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Période", f"{T:.0f} s")
                with col2:
                    st.metric("Période", f"{T/60:.2f} min")
                with col3:
                    st.metric("Période", f"{T/3600:.2f} h")
    
    with tab3:
        st.subheader("⚡ Énergie Orbitale")
        
        st.latex(r"\varepsilon = -\frac{\mu}{2a}")
        
        with st.form("energy_calc"):
            altitude_e = st.number_input("Altitude (km)", 100.0, 100000.0, 500.0, 10.0, key="alt_energy")
            
            submitted_e = st.form_submit_button("🔬 Calculer")
            
            if submitted_e:
                mu_e = CONSTANTS['earth_mu']
                a = CONSTANTS['earth_radius'] + altitude_e * 1000
                
                energy_specific = -mu_e / (2 * a)
                
                st.success("✅ Calcul terminé!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Énergie spécifique", f"{energy_specific:.0f} J/kg")
                with col2:
                    st.metric("Énergie spécifique", f"{energy_specific/1e6:.2f} MJ/kg")
    
    with tab4:
        st.subheader("🎯 Équation Vis-Viva")
        
        st.latex(r"v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)")
        
        st.info("Calcule la vitesse en tout point d'une orbite elliptique")
        
        with st.form("vis_viva"):
            col1, col2 = st.columns(2)
            
            with col1:
                semi_major = st.number_input("Demi-grand axe (km)", 6571.0, 100000.0, 7000.0, 10.0)
                distance = st.number_input("Distance actuelle (km)", 6571.0, 100000.0, 6871.0, 10.0)
            
            with col2:
                mu_vv = CONSTANTS['earth_mu']
                st.metric("μ Terre", f"{mu_vv:.3e} m³/s²")
            
            submitted_vv = st.form_submit_button("🔬 Calculer")
            
            if submitted_vv:
                a_m = semi_major * 1000
                r_m = distance * 1000
                
                v_squared = mu_vv * (2/r_m - 1/a_m)
                v = np.sqrt(abs(v_squared))
                
                st.success("✅ Calcul terminé!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Vitesse", f"{v:.0f} m/s")
                with col2:
                    st.metric("Vitesse", f"{v/1000:.2f} km/s")

# ==================== PAGE: MANŒUVRES ====================
elif page == "🚀 Manœuvres":
    st.header("🚀 Manœuvres Orbitales")
    
    tab1, tab2, tab3 = st.tabs(["🔄 Hohmann", "📐 Changement Inclinaison", "🎯 Rendez-vous"])
    
    with tab1:
        st.subheader("🔄 Transfert de Hohmann")
        
        st.info("""
        **Transfert le plus économe en énergie** entre deux orbites circulaires coplanaires
        
        Nécessite 2 impulsions:
        - ΔV₁ au périgée de l'orbite de transfert
        - ΔV₂ à l'apogée de l'orbite de transfert
        """)
        
        with st.form("hohmann_transfer"):
            col1, col2 = st.columns(2)
            
            with col1:
                r1 = st.number_input("Rayon orbite initiale (km)", 6571.0, 100000.0, 6871.0, 10.0)
            
            with col2:
                r2 = st.number_input("Rayon orbite finale (km)", 6571.0, 200000.0, 42164.0, 10.0)
            
            submitted_hohmann = st.form_submit_button("🔬 Calculer Transfert")
            
            if submitted_hohmann:
                r1_m = r1 * 1000
                r2_m = r2 * 1000
                mu = CONSTANTS['earth_mu']
                
                # Vitesses circulaires
                v1 = np.sqrt(mu / r1_m)
                v2 = np.sqrt(mu / r2_m)
                
                # Orbite de transfert
                a_transfer = (r1_m + r2_m) / 2
                
                # Delta-v
                v_transfer_peri = np.sqrt(mu * (2/r1_m - 1/a_transfer))
                dv1 = v_transfer_peri - v1
                
                v_transfer_apo = np.sqrt(mu * (2/r2_m - 1/a_transfer))
                dv2 = v2 - v_transfer_apo
                
                total_dv = abs(dv1) + abs(dv2)
                
                # Temps de transfert
                transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("ΔV₁", f"{abs(dv1):.0f} m/s")
                with col2:
                    st.metric("ΔV₂", f"{abs(dv2):.0f} m/s")
                with col3:
                    st.metric("ΔV Total", f"{total_dv:.0f} m/s")
                
                st.metric("Temps de transfert", f"{transfer_time/3600:.2f} heures")
                
                # Graphique
                theta = np.linspace(0, 2*np.pi, 100)
                
                # Orbite 1
                x1 = r1 * np.cos(theta)
                y1 = r1 * np.sin(theta)
                
                # Orbite 2
                x2 = r2 * np.cos(theta)
                y2 = r2 * np.sin(theta)
                
                # Orbite transfert
                e_transfer = (r2_m - r1_m) / (r2_m + r1_m)
                a_trans_km = a_transfer / 1000
                
                theta_trans = np.linspace(0, np.pi, 100)
                r_trans = a_trans_km * (1 - e_transfer**2) / (1 + e_transfer * np.cos(theta_trans))
                x_trans = r_trans * np.cos(theta_trans)
                y_trans = r_trans * np.sin(theta_trans)
                
                fig = go.Figure()
                
                # Terre
                fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
                                        marker=dict(size=20, color='blue'),
                                        name='Terre'))
                
                fig.add_trace(go.Scatter(x=x1, y=y1, mode='lines',
                                        name='Orbite initiale', line=dict(color='green')))
                
                fig.add_trace(go.Scatter(x=x2, y=y2, mode='lines',
                                        name='Orbite finale', line=dict(color='red')))
                
                fig.add_trace(go.Scatter(x=x_trans, y=y_trans, mode='lines',
                                        name='Orbite transfert', line=dict(color='orange', dash='dash')))
                
                fig.update_layout(
                    title="Transfert de Hohmann",
                    xaxis_title="X (km)",
                    yaxis_title="Y (km)",
                    height=500,
                    showlegend=True
                )
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📐 Changement d'Inclinaison")
        
        st.latex(r"\Delta v = 2v\sin\left(\frac{\Delta i}{2}\right)")
        
        with st.form("inclination_change"):
            col1, col2 = st.columns(2)
            
            with col1:
                velocity = st.number_input("Vitesse orbitale (m/s)", 1000.0, 15000.0, 7500.0, 100.0)
            
            with col2:
                delta_i = st.slider("Changement inclinaison (°)", 0.0, 180.0, 10.0, 1.0)
            
            submitted_incl = st.form_submit_button("🔬 Calculer")
            
            if submitted_incl:
                delta_i_rad = delta_i * np.pi / 180
                dv = 2 * velocity * np.sin(delta_i_rad / 2)
                
                st.success("✅ Calcul terminé!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("ΔV requis", f"{dv:.0f} m/s")
                with col2:
                    st.metric("ΔV requis", f"{dv/1000:.2f} km/s")
                
                st.warning("⚠️ Le changement d'inclinaison est très coûteux en propergol!")
    
    with tab3:
        st.subheader("🎯 Manœuvre de Rendez-vous")
        
        st.info("""
        **Rendez-vous spatial:** Manœuvre pour rapprocher deux véhicules
        
        Phases:
        1. Phasage (ajustement période)
        2. Approche (transfert Hohmann)
        3. Proximité (manœuvres fines)
        4. Amarrage
        """)
        
        rendezvous_phases = [
            {"Phase": "1. Phasage", "Distance": "> 100 km", "ΔV typique": "10-50 m/s"},
            {"Phase": "2. Approche", "Distance": "100-10 km", "ΔV typique": "5-20 m/s"},
            {"Phase": "3. Proximité", "Distance": "10-0.1 km", "ΔV typique": "2-10 m/s"},
            {"Phase": "4. Amarrage", "Distance": "< 100 m", "ΔV typique": "1-5 m/s"}
        ]
        
        df_rdv = pd.DataFrame(rendezvous_phases)
        st.dataframe(df_rdv, use_container_width=True)

# ==================== PAGE: PROPULSION ====================
elif page == "⚡ Propulsion":
    st.header("⚡ Systèmes de Propulsion")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Types", "📊 Performances", "🔬 Équation Tsiolkovsky"])
    
    with tab1:
        st.subheader("🚀 Types de Propulsion")
        
        propulsion_types = {
            "Chimique": {
                "isp": "300-450 s",
                "poussée": "Très élevée (MN)",
                "exemples": "LOX/LH2, LOX/RP-1, hypergoliques",
                "usage": "Lancement, manœuvres importantes"
            },
            "Électrique": {
                "isp": "1500-3000 s",
                "poussée": "Faible (mN-N)",
                "exemples": "Propulseurs ioniques, Hall",
                "usage": "Missions longue durée, station-keeping"
            },
            "Ionique": {
                "isp": "3000-5000 s",
                "poussée": "Très faible (mN)",
                "exemples": "Xénon, Argon",
                "usage": "Missions interplanétaires (Deep Space 1)"
            },
            "Nucléaire": {
                "isp": "800-1000 s",
                "poussée": "Élevée (kN)",
                "exemples": "NTR (Nuclear Thermal)",
                "usage": "Missions martiennes, espace profond"
            },
            "Voile Solaire": {
                "isp": "Infini (pas de propergol)",
                "poussée": "Très faible (μN-mN)",
                "exemples": "IKAROS, LightSail",
                "usage": "Missions scientifiques, déorbitation"
            }
        }
        
        for prop_name, prop_info in propulsion_types.items():
            with st.expander(f"🚀 {prop_name}"):
                for key, value in prop_info.items():
                    st.write(f"**{key.title()}:** {value}")
    
    with tab2:
        st.subheader("📊 Comparaison Performances")
        
        comparison_data = [
            {"Type": "Chimique (LOX/LH2)", "Isp (s)": 450, "Poussée (N)": 1000000, "Puissance": "N/A"},
            {"Type": "Chimique (LOX/RP-1)", "Isp (s)": 300, "Poussée (N)": 800000, "Puissance": "N/A"},
            {"Type": "Ionique", "Isp (s)": 3500, "Poussée (N)": 0.09, "Puissance": "2.5 kW"},
            {"Type": "Hall Effect", "Isp (s)": 1600, "Poussée (N)": 0.08, "Puissance": "1.5 kW"},
            {"Type": "Nucléaire", "Isp (s)": 900, "Poussée (N)": 100000, "Puissance": "Réacteur"},
        ]
        
        df_prop = pd.DataFrame(comparison_data)
        st.dataframe(df_prop, use_container_width=True)
        
        # Graphique
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[d['Isp (s)'] for d in comparison_data],
            y=[d['Poussée (N)'] for d in comparison_data],
            mode='markers+text',
            text=[d['Type'] for d in comparison_data],
            textposition='top center',
            marker=dict(size=15, color='blue')
        ))
        
        fig.update_layout(
            title="Isp vs Poussée",
            xaxis_title="Isp (s)",
            yaxis_title="Poussée (N)",
            yaxis_type="log",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔬 Équation de Tsiolkovsky")
        
        st.latex(r"\Delta v = I_{sp} \cdot g_0 \cdot \ln\left(\frac{m_0}{m_f}\right)")
        
        st.info("Calcule le delta-v disponible en fonction de l'Isp et du rapport de masse")
        
        with st.form("tsiolkovsky"):
            col1, col2 = st.columns(2)
            
            with col1:
                isp_tsio = st.number_input("Isp (s)", 50.0, 5000.0, 300.0, 10.0)
                dry_mass_tsio = st.number_input("Masse sèche (kg)", 100.0, 100000.0, 1000.0, 100.0)
            
            with col2:
                propellant_tsio = st.number_input("Masse propergol (kg)", 10.0, 50000.0, 500.0, 10.0)
                g0 = 9.80665
                st.metric("g₀", f"{g0} m/s²")
            
            submitted_tsio = st.form_submit_button("🔬 Calculer Delta-v")
            
            if submitted_tsio:
                m0 = dry_mass_tsio + propellant_tsio
                mf = dry_mass_tsio
                
                ve = isp_tsio * g0
                dv = ve * np.log(m0 / mf)
                
                mass_ratio = m0 / mf
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Delta-v", f"{dv:.0f} m/s")
                with col2:
                    st.metric("Rapport masse", f"{mass_ratio:.2f}")
                with col3:
                    st.metric("V échappement", f"{ve:.0f} m/s")
                
                # Graphique delta-v vs propellant
                propellant_range = np.linspace(10, propellant_tsio * 2, 100)
                dv_range = ve * np.log((dry_mass_tsio + propellant_range) / dry_mass_tsio)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=propellant_range, y=dv_range,
                    mode='lines',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=[propellant_tsio], y=[dv],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name='Point actuel'
                ))
                
                fig.update_layout(
                    title="Delta-v vs Masse Propergol",
                    xaxis_title="Masse Propergol (kg)",
                    yaxis_title="Delta-v (m/s)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: SIMULATIONS ====================
elif page == "📊 Simulations":
    st.header("📊 Simulations Orbitales")
    
    tab1, tab2 = st.tabs(["🌍 Propagation Orbite", "🎯 Mission Complète"])
    
    with tab1:
        st.subheader("🌍 Propagation d'Orbite")
        
        with st.form("orbit_propagation"):
            col1, col2 = st.columns(2)
            
            with col1:
                altitude_sim = st.number_input("Altitude (km)", 200.0, 50000.0, 500.0, 10.0)
                eccentricity_sim = st.slider("Excentricité", 0.0, 0.5, 0.0, 0.01)
            
            with col2:
                duration_orbits = st.number_input("Nombre d'orbites", 1, 100, 5, 1)
            
            submitted_sim = st.form_submit_button("🚀 Lancer Simulation")
            
            if submitted_sim:
                with st.spinner("Simulation en cours..."):
                    # Paramètres orbitaux
                    r_earth = CONSTANTS['earth_radius']
                    mu = CONSTANTS['earth_mu']
                    a = r_earth + altitude_sim * 1000
                    e = eccentricity_sim
                    
                    # Période
                    T = 2 * np.pi * np.sqrt(a**3 / mu)
                    
                    # Simulation
                    t_range = np.linspace(0, T * duration_orbits, 1000)
                    
                    # Position (simplifiée - orbite dans le plan XY)
                    n = np.sqrt(mu / a**3)
                    M = n * t_range
                    
                    # Anomalie excentrique (approximation)
                    E = M
                    
                    # Position
                    r = a * (1 - e * np.cos(E))
                    x = r * np.cos(E) / 1000  # km
                    y = r * np.sin(E) * np.sqrt(1 - e**2) / 1000
                    
                    st.success("✅ Simulation terminée!")
                    
                    # Graphique 2D
                    fig = go.Figure()
                    
                    # Terre
                    theta_earth = np.linspace(0, 2*np.pi, 100)
                    x_earth = r_earth/1000 * np.cos(theta_earth)
                    y_earth = r_earth/1000 * np.sin(theta_earth)
                    
                    fig.add_trace(go.Scatter(
                        x=x_earth, y=y_earth,
                        mode='lines',
                        fill='toself',
                        name='Terre',
                        fillcolor='lightblue',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        name='Orbite',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Propagation Orbite - {duration_orbits} orbite(s)",
                        xaxis_title="X (km)",
                        yaxis_title="Y (km)",
                        height=600,
                        showlegend=True
                    )
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Période", f"{T/60:.2f} min")
                    with col2:
                        periapsis = a * (1 - e) - r_earth
                        st.metric("Périgée", f"{periapsis/1000:.0f} km")
                    with col3:
                        apoapsis = a * (1 + e) - r_earth
                        st.metric("Apogée", f"{apoapsis/1000:.0f} km")
    
    with tab2:
        st.subheader("🎯 Simulation Mission Complète")
        
        st.info("Simulation d'une mission depuis le lancement jusqu'au retour")
        
        mission_phases = [
            {"Phase": "1. Lancement", "Durée": "10 min", "Delta-v": "9,400 m/s", "Altitude": "0 → 200 km"},
            {"Phase": "2. Insertion LEO", "Durée": "5 min", "Delta-v": "100 m/s", "Altitude": "200 km"},
            {"Phase": "3. Opérations LEO", "Durée": "Variable", "Delta-v": "50 m/s/an", "Altitude": "200-500 km"},
            {"Phase": "4. Désorbitation", "Durée": "30 min", "Delta-v": "100 m/s", "Altitude": "→ 0 km"}
        ]
        
        df_mission = pd.DataFrame(mission_phases)
        st.dataframe(df_mission, use_container_width=True)
        
        total_dv = 9400 + 100 + 50 + 100
        st.metric("Delta-v Total Mission", f"{total_dv} m/s")

# ==================== PAGE: TRAJECTOIRES ====================
elif page == "📡 Trajectoires":
    st.header("📡 Trajectoires Spatiales")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Planification", "📊 Analyse", "🗺️ Visualisation"])
    
    with tab1:
        st.subheader("🎯 Planification de Trajectoire")
        
        st.write("### 🚀 Paramètres Mission")
        
        with st.form("trajectory_planning"):
            col1, col2 = st.columns(2)
            
            with col1:
                departure_body = st.selectbox("Départ", ["Terre", "Lune", "Mars", "Station spatiale"])
                arrival_body = st.selectbox("Arrivée", ["Lune", "Mars", "Astéroïde", "Jupiter"])
            
            with col2:
                launch_date = st.date_input("Date lancement", datetime.now())
                mission_duration = st.number_input("Durée max (jours)", 1, 1000, 180, 10)
            
            st.write("### ⚡ Contraintes")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_delta_v = st.number_input("Delta-v max (km/s)", 1.0, 20.0, 10.0, 0.5)
            with col2:
                max_acceleration = st.number_input("Accélération max (m/s²)", 0.001, 10.0, 0.1, 0.01)
            with col3:
                trajectory_type = st.selectbox("Type", ["Direct", "Gravity Assist", "Spirale"])
            
            submitted_traj = st.form_submit_button("🔬 Calculer Trajectoire")
            
            if submitted_traj:
                st.success("✅ Trajectoire calculée!")
                
                # Résultats simulés
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Delta-v total", f"{np.random.uniform(5, 12):.2f} km/s")
                with col2:
                    st.metric("Durée transit", f"{np.random.randint(120, 300)} jours")
                with col3:
                    st.metric("Consommation", f"{np.random.randint(500, 2000)} kg")
                with col4:
                    st.metric("Fenêtre", f"{np.random.randint(10, 30)} jours")
    
    with tab2:
        st.subheader("📊 Analyse de Trajectoire")
        
        st.write("### 📈 Profil Vitesse")
        
        # Simulation profil
        time_profile = np.linspace(0, 180, 500)
        velocity_profile = 11 + 3 * np.sin(time_profile / 30) + np.random.randn(500) * 0.2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_profile, y=velocity_profile,
            mode='lines',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Profil de Vitesse durant le Transit",
            xaxis_title="Temps (jours)",
            yaxis_title="Vitesse (km/s)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🗺️ Visualisation 3D")
        
        st.info("Visualisation interactive de la trajectoire en 3D")
        
        # Trajectoire simplifiée
        t = np.linspace(0, 2*np.pi, 100)
        x = np.cos(t) * 150e6  # km
        y = np.sin(t) * 150e6
        z = np.sin(t * 2) * 20e6
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='red', width=4)
        )])
        
        # Terre
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Terre'
        ))
        
        fig.update_layout(
            title="Trajectoire Interplanétaire",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: TRANSFERTS ====================
elif page == "🎯 Transferts":
    st.header("🎯 Transferts Interplanétaires")
    
    tab1, tab2, tab3 = st.tabs(["🌍→🌙 Terre-Lune", "🌍→🔴 Terre-Mars", "📊 Comparaison"])
    
    with tab1:
        st.subheader("🌍→🌙 Transfert Terre-Lune")
        
        st.info("""
        **Transfert Terre-Lune typique:**
        
        1. **Injection Trans-Lunaire (TLI):** ~3.1 km/s
        2. **Transit:** 3-5 jours
        3. **Insertion orbitale lunaire (LOI):** ~0.9 km/s
        4. **Delta-v total:** ~4 km/s
        """)
        
        with st.form("earth_moon_transfer"):
            col1, col2 = st.columns(2)
            
            with col1:
                parking_orbit = st.number_input("Orbite parking Terre (km)", 200.0, 2000.0, 300.0, 50.0)
                lunar_orbit = st.number_input("Orbite lunaire (km)", 100.0, 500.0, 100.0, 10.0)
            
            with col2:
                transfer_type_moon = st.selectbox("Type transfert", 
                    ["Direct", "Bi-elliptique", "Low Energy (WSB)"])
            
            submitted_moon = st.form_submit_button("🔬 Calculer")
            
            if submitted_moon:
                # Calculs simplifiés
                if transfer_type_moon == "Direct":
                    tli_dv = 3.1
                    loi_dv = 0.9
                    duration = 3.5
                elif transfer_type_moon == "Low Energy (WSB)":
                    tli_dv = 3.05
                    loi_dv = 0.4
                    duration = 30
                else:
                    tli_dv = 3.2
                    loi_dv = 0.95
                    duration = 5
                
                total_dv = tli_dv + loi_dv
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("TLI Delta-v", f"{tli_dv:.2f} km/s")
                with col2:
                    st.metric("LOI Delta-v", f"{loi_dv:.2f} km/s")
                with col3:
                    st.metric("Total", f"{total_dv:.2f} km/s")
                
                st.metric("Durée transit", f"{duration:.1f} jours")
    
    with tab2:
        st.subheader("🌍→🔴 Transfert Terre-Mars")
        
        st.info("""
        **Fenêtre de lancement Terre-Mars:** Tous les 26 mois
        
        **Transfert de Hohmann:**
        - Delta-v départ Terre: ~3.6 km/s
        - Durée transit: ~260 jours
        - Delta-v arrivée Mars: ~2.5 km/s
        """)
        
        st.write("### 📊 Fenêtres de Lancement")
        
        launch_windows = [
            {"Année": "2024", "Date optimale": "Nov 2024", "Delta-v (km/s)": "5.6", "Durée (jours)": "245"},
            {"Année": "2026", "Date optimale": "Déc 2026", "Delta-v (km/s)": "5.8", "Durée (jours)": "250"},
            {"Année": "2028", "Date optimale": "Jan 2029", "Delta-v (km/s)": "5.5", "Durée (jours)": "240"},
            {"Année": "2031", "Date optimale": "Fév 2031", "Delta-v (km/s)": "5.9", "Durée (jours)": "255"}
        ]
        
        df_windows = pd.DataFrame(launch_windows)
        st.dataframe(df_windows, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🗺️ Positions Planètes")
        
        # Graphique positions
        theta_earth = np.linspace(0, 2*np.pi, 100)
        r_earth = 1  # UA
        r_mars = 1.52  # UA
        
        # Position actuelle (simulée)
        earth_angle = 0
        mars_angle = 1.5
        
        fig = go.Figure()
        
        # Soleil
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=30, color='yellow'),
            name='Soleil'
        ))
        
        # Orbites
        fig.add_trace(go.Scatter(
            x=r_earth * np.cos(theta_earth),
            y=r_earth * np.sin(theta_earth),
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Orbite Terre'
        ))
        
        fig.add_trace(go.Scatter(
            x=r_mars * np.cos(theta_earth),
            y=r_mars * np.sin(theta_earth),
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Orbite Mars'
        ))
        
        # Planètes
        fig.add_trace(go.Scatter(
            x=[r_earth * np.cos(earth_angle)],
            y=[r_earth * np.sin(earth_angle)],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Terre'
        ))
        
        fig.add_trace(go.Scatter(
            x=[r_mars * np.cos(mars_angle)],
            y=[r_mars * np.sin(mars_angle)],
            mode='markers',
            marker=dict(size=12, color='red'),
            name='Mars'
        ))
        
        fig.update_layout(
            title="Configuration Terre-Mars",
            xaxis_title="X (UA)",
            yaxis_title="Y (UA)",
            height=500
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Comparaison Destinations")
        
        destinations_data = [
            {"Destination": "Lune", "Distance (km)": "384,400", "Delta-v (km/s)": "4.0", "Durée (jours)": "3-5"},
            {"Destination": "Mars", "Distance (km)": "225M", "Delta-v (km/s)": "6.0", "Durée (jours)": "240-270"},
            {"Destination": "Vénus", "Distance (km)": "108M", "Delta-v (km/s)": "5.5", "Durée (jours)": "120-150"},
            {"Destination": "Mercure", "Distance (km)": "91M", "Delta-v (km/s)": "13.0", "Durée (jours)": "100-150"},
            {"Destination": "Jupiter", "Distance (km)": "778M", "Delta-v (km/s)": "9.0", "Durée (jours)": "600-900"}
        ]
        
        df_dest = pd.DataFrame(destinations_data)
        st.dataframe(df_dest, use_container_width=True)

# ==================== PAGE: MISSIONS LUNAIRES ====================
elif page == "🌙 Missions Lunaires":
    st.header("🌙 Missions Lunaires")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Programme Artemis", "🏗️ Base Lunaire", "📊 Ressources"])
    
    with tab1:
        st.subheader("🚀 Programme Artemis")
        
        st.info("""
        **Programme Artemis - Retour sur la Lune**
        
        Objectif: Établir présence humaine durable sur la Lune
        """)
        
        artemis_missions = [
            {"Mission": "Artemis I", "Date": "2022", "Type": "Sans équipage", "Statut": "✅ Complétée"},
            {"Mission": "Artemis II", "Date": "2024", "Type": "Vol habité (survol)", "Statut": "🟡 Planifiée"},
            {"Mission": "Artemis III", "Date": "2025", "Type": "Alunissage pôle Sud", "Statut": "🟡 Planifiée"},
            {"Mission": "Artemis IV", "Date": "2027", "Type": "Gateway + alunissage", "Statut": "🔵 Future"}
        ]
        
        df_artemis = pd.DataFrame(artemis_missions)
        st.dataframe(df_artemis, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🛰️ Lunar Gateway")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Caractéristiques:**")
            st.write("• Orbite: NRHO (Near-Rectilinear Halo)")
            st.write("• Altitude: 1,500 - 70,000 km")
            st.write("• Modules: Habitation, Logistique, Airlock")
            st.write("• Puissance: 60 kW (panneaux solaires)")
        
        with col2:
            st.write("**Rôle:**")
            st.write("• Station spatiale lunaire")
            st.write("• Point de transit Terre-Lune")
            st.write("• Support missions surface")
            st.write("• Plateforme scientifique")
    
    with tab2:
        st.subheader("🏗️ Base Lunaire")
        
        st.write("### 🌙 Concept Base Permanente")
        
        base_elements = {
            "Modules Habitat": {
                "nombre": "4-6 modules",
                "capacité": "4-8 astronautes",
                "protection": "Régolithe (radiations)"
            },
            "Production Énergie": {
                "source": "Panneaux solaires + Nucléaire",
                "puissance": "100-200 kW",
                "stockage": "Batteries + Piles combustible"
            },
            "ISRU (Utilisation Ressources)": {
                "eau": "Extraction glace pôles",
                "oxygène": "Électrolyse eau",
                "propergol": "LOX/LH2 production locale"
            },
            "Systèmes Support Vie": {
                "air": "Recyclage O2/CO2",
                "eau": "Recyclage 95%+",
                "nourriture": "Serres hydroponiques"
            }
        }
        
        for element_name, element_info in base_elements.items():
            with st.expander(f"🏗️ {element_name}"):
                for key, value in element_info.items():
                    st.write(f"**{key.title()}:** {value}")
    
    with tab3:
        st.subheader("📊 Ressources Lunaires")
        
        st.write("### 💎 Ressources Disponibles")
        
        resources_data = [
            {"Ressource": "Eau (glace)", "Localisation": "Pôles (cratères ombre)", "Quantité": "~600M tonnes", "Usage": "Propergol, Vie"},
            {"Ressource": "Hélium-3", "Localisation": "Régolithe", "Quantité": "~1M tonnes", "Usage": "Fusion nucléaire"},
            {"Ressource": "Ilménite (FeTiO3)", "Localisation": "Mers lunaires", "Quantité": "Abondant", "Usage": "Oxygène, métaux"},
            {"Ressource": "Silicates", "Localisation": "Partout", "Quantité": "Très abondant", "Usage": "Construction, verre"},
            {"Ressource": "Aluminium", "Localisation": "Highlands", "Quantité": "10-15%", "Usage": "Structures"},
        ]
        
        df_resources = pd.DataFrame(resources_data)
        st.dataframe(df_resources, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### ⚡ Production ISRU")
        
        st.info("""
        **ISRU (In-Situ Resource Utilization):**
        
        Processus clé pour base lunaire durable:
        
        1. **Extraction eau:** Chauffage régolithe → vapeur
        2. **Électrolyse:** H₂O → H₂ + O₂
        3. **Production propergol:** LOX/LH2 pour retour Terre
        4. **Économie:** Réduction 90% masse lancée depuis Terre
        """)

# ==================== PAGE: MISSIONS MARS ====================
elif page == "🔴 Missions Mars":
    st.header("🔴 Missions Martiennes")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Architecture Mission", "🏗️ Colonisation", "🔬 Terraformation"])
    
    with tab1:
        st.subheader("🚀 Architecture Mission Mars")
        
        st.write("### 📊 Profil Mission Type")
        
        mission_profile = [
            {"Phase": "1. Lancement Terre", "Delta-v": "3.6 km/s", "Durée": "1 jour", "Date": "T+0"},
            {"Phase": "2. Transit Terre-Mars", "Delta-v": "0", "Durée": "6-9 mois", "Date": "T+1j"},
            {"Phase": "3. Capture Mars", "Delta-v": "2.5 km/s", "Durée": "1 jour", "Date": "T+240j"},
            {"Phase": "4. Descente/EDL", "Delta-v": "5.5 km/s", "Durée": "7 min", "Date": "T+241j"},
            {"Phase": "5. Surface Mars", "Delta-v": "0", "Durée": "18 mois", "Date": "T+241j"},
            {"Phase": "6. Ascension", "Delta-v": "5.5 km/s", "Durée": "1 jour", "Date": "T+780j"},
            {"Phase": "7. Transit Mars-Terre", "Delta-v": "2.5 km/s", "Durée": "6-9 mois", "Date": "T+781j"},
            {"Phase": "8. Rentrée Terre", "Delta-v": "3.6 km/s", "Durée": "1 jour", "Date": "T+1020j"}
        ]
        
        df_mission = pd.DataFrame(mission_profile)
        st.dataframe(df_mission, use_container_width=True)
        
        total_dv = 3.6 + 2.5 + 5.5 + 5.5 + 2.5 + 3.6
        st.metric("Delta-v Total", f"{total_dv:.1f} km/s")
        st.metric("Durée Mission", "~3 ans")
        
        st.markdown("---")
        
        st.write("### 🚀 Concepts Mission")
        
        concepts = {
            "Mars Direct (Zubrin)": {
                "description": "Mission directe avec ISRU",
                "équipage": "4-6 astronautes",
                "durée": "2.5 ans",
                "coût": "~50 Mrd$"
            },
            "SpaceX Starship": {
                "description": "Vaisseau réutilisable 100 tonnes",
                "équipage": "100+ passagers",
                "durée": "2 ans",
                "coût": "~10 Mrd$ (estimation)"
            },
            "NASA Moon to Mars": {
                "description": "Via Gateway, approche progressive",
                "équipage": "4 astronautes",
                "durée": "3 ans",
                "coût": "~200 Mrd$"
            }
        }
        
        for concept_name, concept_info in concepts.items():
            with st.expander(f"🚀 {concept_name}"):
                for key, value in concept_info.items():
                    st.write(f"**{key.title()}:** {value}")
    
    with tab2:
        st.subheader("🏗️ Colonisation de Mars")
        
        st.write("### 🌍 Base Martienne")
        
        st.info("""
        **Phases Colonisation:**
        
        1. **Phase 1 (2030s):** Avant-poste scientifique (4-6 personnes)
        2. **Phase 2 (2040s):** Base permanente (20-50 personnes)
        3. **Phase 3 (2050s+):** Village (500-1000 personnes)
        4. **Phase 4 (2100+):** Ville (10,000+ personnes)
        """)
        
        st.write("### 🏗️ Infrastructure Nécessaire")
        
        infrastructure = [
            {"Système": "Habitats", "Capacité": "100 personnes", "Modules": "20-30", "Protection": "Régolithe"},
            {"Système": "Énergie", "Puissance": "1-10 MW", "Source": "Nucléaire + Solaire", "Stockage": "Batteries"},
            {"Système": "ISRU", "Production": "Propergol, O2, H2O", "Rendement": "Tonnes/an", "Économie": "90% masse"},
            {"Système": "Serres", "Surface": "1000+ m²", "Production": "Nourriture", "Recyclage": "95%+"},
            {"Système": "Usine", "Fonction": "Fabrication pièces", "Technologie": "Impression 3D", "Matériaux": "Régolithe"}
        ]
        
        df_infra = pd.DataFrame(infrastructure)
        st.dataframe(df_infra, use_container_width=True)
    
    with tab3:
        st.subheader("🔬 Terraformation de Mars")
        
        st.warning("⚠️ Concept théorique à très long terme (siècles/millénaires)")
        
        st.write("### 🌍 Objectifs Terraformation")
        
        objectives = [
            "**Pression atmosphérique:** 0.006 bar → 0.6 bar (Terre: 1 bar)",
            "**Température:** -60°C → +15°C",
            "**Composition air:** CO2 → O2/N2",
            "**Eau liquide:** Fonte glace polaire"
        ]
        
        for obj in objectives:
            st.write(f"• {obj}")
        
        st.markdown("---")
        
        st.write("### 🔬 Méthodes Proposées")
        
        methods = {
            "Réchauffement": [
                "Miroirs orbitaux (réflecteurs solaires)",
                "Gaz à effet de serre (SF6, CHF3)",
                "Noircissement calottes polaires",
                "Impact astéroïdes volatiles"
            ],
            "Épaississement atmosphère": [
                "Dégazage CO2 du sol",
                "Sublimation glace polaire",
                "Import comètes glacées",
                "Activité volcanique artificielle"
            ],
            "Production O2": [
                "Photolyse H2O",
                "Cyanobactéries",
                "Plantes modifiées",
                "Électrolyse industrielle"
            ]
        }
        
        for method_cat, method_list in methods.items():
            with st.expander(f"🔬 {method_cat}"):
                for method in method_list:
                    st.write(f"• {method}")
        
        st.metric("Durée Estimée", "300-1000 ans")
        st.metric("Coût Estimé", "> 1000 Trillions $")

# ==================== PAGE: TRACE AU SOL ====================
elif page == "🗺️ Trace au Sol":
    st.header("🗺️ Trace au Sol (Ground Track)")
    
    st.info("""
    **Trace au sol:** Projection de l'orbite du satellite sur la surface terrestre
    
    La trace se déplace vers l'ouest en raison de la rotation de la Terre
    """)
    
    with st.form("ground_track"):
        col1, col2 = st.columns(2)
        
        with col1:
            altitude_gt = st.number_input("Altitude (km)", 200.0, 2000.0, 500.0, 10.0)
            inclination_gt = st.slider("Inclinaison (°)", 0.0, 180.0, 51.6, 1.0)
        
        with col2:
            duration_orbits_gt = st.number_input("Nombre d'orbites", 1, 20, 3, 1)
        
        submitted_gt = st.form_submit_button("🗺️ Générer Trace")
        
        if submitted_gt:
            # Calculs
            r_earth = CONSTANTS['earth_radius']
            mu = CONSTANTS['earth_mu']
            omega_earth = 7.2921159e-5  # rad/s
            
            r = r_earth + altitude_gt * 1000
            T = 2 * np.pi * np.sqrt(r**3 / mu)
            
            # Nombre de points
            n_points = 500
            t = np.linspace(0, T * duration_orbits_gt, n_points)
            
            # Position orbitale (simplifiée)
            theta = 2 * np.pi * t / T
            
            # Latitude (fonction de l'inclinaison)
            lat = np.degrees(np.arcsin(np.sin(np.radians(inclination_gt)) * np.sin(theta)))
            
            # Longitude (avec rotation Terre)
            lon = np.degrees(theta) - np.degrees(omega_earth * t)
            lon = (lon + 180) % 360 - 180
            
            # Graphique
            fig = go.Figure()
            
            fig.add_trace(go.Scattergeo(
                lon=lon,
                lat=lat,
                mode='lines+markers',
                line=dict(width=2, color='red'),
                marker=dict(size=4, color='red'),
                name='Trace au sol'
            ))
            
            fig.update_layout(
                title=f"Trace au Sol - {duration_orbits_gt} orbite(s)",
                geo=dict(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='lightgreen',
                    showocean=True,
                    oceancolor='lightblue',
                    showcountries=True
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Trace générée!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Période", f"{T/60:.2f} min")
            with col2:
                st.metric("Vitesse sol", f"{(2 * np.pi * r / T)/1000:.2f} km/s")
            with col3:
                st.metric("Couverture max", f"±{inclination_gt:.0f}° latitude")

# ==================== PAGE: STATIONS SOL ====================
elif page == "📡 Stations Sol":
    st.header("📡 Stations Sol (Ground Stations)")
    
    tab1, tab2, tab3 = st.tabs(["➕ Créer Station", "🗺️ Réseau Stations", "📊 Couverture"])
    
    with tab1:
        st.subheader("➕ Créer une Station Sol")
        
        with st.form("create_ground_station"):
            col1, col2 = st.columns(2)
            
            with col1:
                station_name = st.text_input("📝 Nom de la Station", "Station-Paris")
                location = st.text_input("📍 Localisation", "Paris, France")
                latitude = st.number_input("Latitude (°)", -90.0, 90.0, 48.8566, 0.0001)
                longitude = st.number_input("Longitude (°)", -180.0, 180.0, 2.3522, 0.0001)
            
            with col2:
                frequency = st.number_input("Fréquence (GHz)", 1.0, 50.0, 8.0, 0.1)
                antenna_diameter = st.number_input("Diamètre Antenne (m)", 1.0, 100.0, 15.0, 0.5)
                min_elevation = st.slider("Élévation Minimale (°)", 0, 45, 10, 1)
                max_data_rate = st.number_input("Débit Max (Mbps)", 1, 10000, 300, 10)
            
            bands = st.multiselect(
                "Bandes de Fréquence",
                ["S-band", "X-band", "Ka-band", "Ku-band", "C-band"],
                default=["S-band", "X-band"]
            )
            
            submitted_station = st.form_submit_button("📡 Créer Station", type="primary")
            
            if submitted_station:
                if not station_name:
                    st.error("⚠️ Veuillez donner un nom à la station")
                else:
                    station_id = f"gs_{len(st.session_state.space_system['ground_stations']) + 1}"
                    
                    st.session_state.space_system['ground_stations'][station_id] = {
                        'id': station_id,
                        'name': station_name,
                        'location': location,
                        'latitude': latitude,
                        'longitude': longitude,
                        'frequency': frequency,
                        'antenna_diameter': antenna_diameter,
                        'min_elevation': min_elevation,
                        'max_data_rate': max_data_rate,
                        'bands': bands,
                        'status': 'active',
                        'passes_today': 0,
                        'total_data': 0.0
                    }
                    
                    log_event(f"Station sol créée: {station_name}")
                    st.success(f"✅ Station '{station_name}' créée avec succès!")
                    st.balloons()
    
    with tab2:
        st.subheader("🗺️ Réseau de Stations Sol")
        
        if st.session_state.space_system['ground_stations']:
            for station_id, station in st.session_state.space_system['ground_stations'].items():
                with st.expander(f"📡 {station['name']} - {station['location']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Latitude", f"{station['latitude']:.2f}°")
                        st.metric("Longitude", f"{station['longitude']:.2f}°")
                    
                    with col2:
                        st.metric("Fréquence", f"{station['frequency']} GHz")
                        st.metric("Diamètre Antenne", f"{station['antenna_diameter']} m")
                    
                    with col3:
                        st.metric("Élévation Min", f"{station['min_elevation']}°")
                        st.metric("Débit Max", f"{station['max_data_rate']} Mbps")
                    
                    st.write(f"**Bandes:** {', '.join(station['bands'])}")
                    
                    if st.button(f"🗑️ Supprimer", key=f"del_station_{station_id}"):
                        del st.session_state.space_system['ground_stations'][station_id]
                        st.rerun()
        else:
            st.info("💡 Aucune station sol créée")
    
    with tab3:
        st.subheader("📊 Analyse de Couverture")
        
        if st.session_state.space_system['ground_stations']:
            # Carte des stations
            st.write("### 🗺️ Carte des Stations")
            
            stations_data = []
            for station in st.session_state.space_system['ground_stations'].values():
                stations_data.append({
                    'lat': station['latitude'],
                    'lon': station['longitude'],
                    'name': station['name']
                })
            
            df_stations = pd.DataFrame(stations_data)
            
            fig = go.Figure(data=go.Scattergeo(
                lon=df_stations['lon'],
                lat=df_stations['lat'],
                text=df_stations['name'],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='circle'),
                textposition='top center'
            ))
            
            fig.update_layout(
                title="Réseau de Stations Sol",
                geo=dict(
                    projection_type='natural earth',
                    showland=True,
                    landcolor='lightgreen',
                    showocean=True,
                    oceancolor='lightblue'
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            st.write("### 📊 Statistiques Réseau")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Stations Actives", len(st.session_state.space_system['ground_stations']))
            with col2:
                total_passes = sum(s['passes_today'] for s in st.session_state.space_system['ground_stations'].values())
                st.metric("Passages Aujourd'hui", total_passes)
            with col3:
                total_data = sum(s['total_data'] for s in st.session_state.space_system['ground_stations'].values())
                st.metric("Données Reçues", f"{total_data:.1f} GB")
            with col4:
                avg_elevation = sum(s['min_elevation'] for s in st.session_state.space_system['ground_stations'].values()) / len(st.session_state.space_system['ground_stations'])
                st.metric("Élévation Moy.", f"{avg_elevation:.1f}°")
        else:
            st.info("💡 Créez des stations sol pour voir l'analyse de couverture")

# ==================== PAGE: RENDEZ-VOUS ====================
elif page == "🔭 Rendez-vous":
    st.header("🔭 Rendez-vous Spatial")
    
    tab1, tab2, tab3 = st.tabs(["📐 Calcul Rendez-vous", "🎯 Approche Proximité", "📊 Historique"])
    
    with tab1:
        st.subheader("📐 Calcul de Rendez-vous")
        
        st.info("""
        **Rendez-vous spatial:** Manœuvre pour rapprocher deux véhicules en orbite
        
        **Phases principales:**
        1. **Phasage** - Ajustement de la période orbitale
        2. **Approche** - Réduction de la distance
        3. **Proximité** - Manœuvres fines (<1 km)
        4. **Capture/Amarrage** - Contact final
        """)
        
        with st.form("rendezvous_calc"):
            st.write("### 🛰️ Configuration Véhicules")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Chasseur (Véhicule Actif)**")
                chaser_altitude = st.number_input("Altitude Chasseur (km)", 200.0, 2000.0, 400.0, 10.0)
                chaser_phase = st.number_input("Phase Angle Chasseur (°)", 0.0, 360.0, 0.0, 1.0)
            
            with col2:
                st.write("**Cible (Véhicule Passif)**")
                target_altitude = st.number_input("Altitude Cible (km)", 200.0, 2000.0, 420.0, 10.0)
                target_phase = st.number_input("Phase Angle Cible (°)", 0.0, 360.0, 45.0, 1.0)
            
            st.write("### ⚙️ Paramètres Rendez-vous")
            
            col1, col2 = st.columns(2)
            
            with col1:
                approach_strategy = st.selectbox(
                    "Stratégie d'Approche",
                    ["Hohmann Direct", "Multi-impulsion", "Spirale Continue"]
                )
            
            with col2:
                safety_distance = st.number_input("Distance Sécurité (m)", 10.0, 1000.0, 100.0, 10.0)
            
            submitted_rdv = st.form_submit_button("🔬 Calculer Rendez-vous")
            
            if submitted_rdv:
                with st.spinner("Calcul en cours..."):
                    # Calculs
                    r_earth = CONSTANTS['earth_radius']
                    mu = CONSTANTS['earth_mu']
                    
                    r_chaser = r_earth + chaser_altitude * 1000
                    r_target = r_earth + target_altitude * 1000
                    
                    # Périodes
                    T_chaser = 2 * np.pi * np.sqrt(r_chaser**3 / mu)
                    T_target = 2 * np.pi * np.sqrt(r_target**3 / mu)
                    
                    # Vitesses
                    v_chaser = np.sqrt(mu / r_chaser)
                    v_target = np.sqrt(mu / r_target)
                    
                    # Phase angle difference
                    phase_diff = abs(target_phase - chaser_phase)
                    
                    # Calcul manœuvre phasage
                    if approach_strategy == "Hohmann Direct":
                        # Delta-v Hohmann
                        a_transfer = (r_chaser + r_target) / 2
                        v_transfer_peri = np.sqrt(mu * (2/r_chaser - 1/a_transfer))
                        dv1 = abs(v_transfer_peri - v_chaser)
                        
                        v_transfer_apo = np.sqrt(mu * (2/r_target - 1/a_transfer))
                        dv2 = abs(v_target - v_transfer_apo)
                        
                        total_dv = dv1 + dv2
                        transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)
                    else:
                        # Approximation multi-impulsion
                        total_dv = abs(v_target - v_chaser) * 1.5
                        transfer_time = abs(T_target - T_chaser) * 2
                    
                    # Temps de phasage
                    if T_chaser != T_target:
                        phasing_time = (phase_diff / 360) * abs(T_target * T_chaser / (T_target - T_chaser))
                    else:
                        phasing_time = 0
                    
                    st.success("✅ Calcul terminé!")
                    
                    # Résultats
                    st.write("### 📊 Résultats")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ΔV Total", f"{total_dv:.1f} m/s")
                    with col2:
                        st.metric("Temps Transfert", f"{transfer_time/3600:.2f} h")
                    with col3:
                        st.metric("Temps Phasage", f"{abs(phasing_time)/3600:.2f} h")
                    with col4:
                        total_time = transfer_time + abs(phasing_time)
                        st.metric("Durée Totale", f"{total_time/3600:.2f} h")
                    
                    # Détails par phase
                    st.write("### 📋 Détails par Phase")
                    
                    phases_data = [
                        {
                            "Phase": "1. Phasage",
                            "ΔV (m/s)": f"{total_dv * 0.15:.1f}",
                            "Durée": f"{abs(phasing_time)/3600:.1f} h",
                            "Distance": f"{abs(r_target - r_chaser)/1000:.1f} km"
                        },
                        {
                            "Phase": "2. Approche",
                            "ΔV (m/s)": f"{total_dv * 0.70:.1f}",
                            "Durée": f"{transfer_time/3600:.1f} h",
                            "Distance": "10 km → 1 km"
                        },
                        {
                            "Phase": "3. Proximité",
                            "ΔV (m/s)": f"{total_dv * 0.10:.1f}",
                            "Durée": "2-4 h",
                            "Distance": "1 km → 100 m"
                        },
                        {
                            "Phase": "4. Amarrage",
                            "ΔV (m/s)": f"{total_dv * 0.05:.1f}",
                            "Durée": "0.5-1 h",
                            "Distance": "100 m → 0 m"
                        }
                    ]
                    
                    df_phases = pd.DataFrame(phases_data)
                    st.dataframe(df_phases, use_container_width=True)
                    
                    # Visualisation trajectoire
                    st.write("### 🗺️ Visualisation Trajectoire")
                    
                    theta = np.linspace(0, 2*np.pi, 100)
                    
                    fig = go.Figure()
                    
                    # Terre
                    x_earth = r_earth/1000 * np.cos(theta)
                    y_earth = r_earth/1000 * np.sin(theta)
                    fig.add_trace(go.Scatter(
                        x=x_earth, y=y_earth,
                        mode='lines',
                        fill='toself',
                        name='Terre',
                        fillcolor='lightblue',
                        line=dict(color='blue')
                    ))
                    
                    # Orbite chasseur
                    x_chaser = r_chaser/1000 * np.cos(theta)
                    y_chaser = r_chaser/1000 * np.sin(theta)
                    fig.add_trace(go.Scatter(
                        x=x_chaser, y=y_chaser,
                        mode='lines',
                        name='Orbite Chasseur',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    # Orbite cible
                    x_target = r_target/1000 * np.cos(theta)
                    y_target = r_target/1000 * np.sin(theta)
                    fig.add_trace(go.Scatter(
                        x=x_target, y=y_target,
                        mode='lines',
                        name='Orbite Cible',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Position initiale chasseur
                    chaser_x = r_chaser/1000 * np.cos(chaser_phase * np.pi/180)
                    chaser_y = r_chaser/1000 * np.sin(chaser_phase * np.pi/180)
                    fig.add_trace(go.Scatter(
                        x=[chaser_x], y=[chaser_y],
                        mode='markers',
                        name='Chasseur',
                        marker=dict(size=15, color='green')
                    ))
                    
                    # Position cible
                    target_x = r_target/1000 * np.cos(target_phase * np.pi/180)
                    target_y = r_target/1000 * np.sin(target_phase * np.pi/180)
                    fig.add_trace(go.Scatter(
                        x=[target_x], y=[target_y],
                        mode='markers',
                        name='Cible',
                        marker=dict(size=15, color='red')
                    ))
                    
                    fig.update_layout(
                        title="Configuration Rendez-vous",
                        xaxis_title="X (km)",
                        yaxis_title="Y (km)",
                        height=600
                    )
                    fig.update_yaxes(scaleanchor="x", scaleratio=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 Opérations de Proximité")
        
        st.info("""
        **Opérations de proximité** (< 1 km):
        
        - Utilisation capteurs relatifs (LIDAR, radar, caméras)
        - Manœuvres fines et précises
        - Contrôle d'attitude strict
        - Communication continue
        """)
        
        st.write("### 📊 Profil Approche Finale")
        
        # Simulation approche
        distance_profile = np.array([1000, 500, 250, 100, 50, 20, 10, 5, 2, 1, 0])
        time_profile = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_profile, y=distance_profile,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Profil Distance durant Approche Finale",
            xaxis_title="Temps (minutes)",
            yaxis_title="Distance (mètres)",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Points de contrôle
        st.write("### 🎯 Points de Contrôle (Hold Points)")
        
        hold_points = [
            {"Distance": "1000 m", "Durée Hold": "30 min", "Vérifications": "Systèmes navigation, communications"},
            {"Distance": "250 m", "Durée Hold": "20 min", "Vérifications": "Capteurs proximité, alignement"},
            {"Distance": "100 m", "Durée Hold": "15 min", "Vérifications": "Systèmes amarrage, Go/No-Go"},
            {"Distance": "10 m", "Durée Hold": "10 min", "Vérifications": "Capture finale, contact imminent"}
        ]
        
        df_holds = pd.DataFrame(hold_points)
        st.dataframe(df_holds, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Historique Rendez-vous")
        
        st.write("### 🚀 Missions Historiques")
        
        historical_rdv = [
            {
                "Mission": "Gemini 6A/7 (1965)",
                "Véhicules": "2 capsules Gemini",
                "Distance Min": "30 cm",
                "Durée": "5h 18min",
                "Résultat": "✅ Succès"
            },
            {
                "Mission": "Apollo 11 (1969)",
                "Véhicules": "CSM/LM",
                "Distance Min": "Contact",
                "Durée": "~3h",
                "Résultat": "✅ Amarrage"
            },
            {
                "Mission": "Apollo-Soyuz (1975)",
                "Véhicules": "Apollo/Soyuz",
                "Distance Min": "Contact",
                "Durée": "~6h",
                "Résultat": "✅ Premier RDV international"
            },
            {
                "Mission": "STS-49/Intelsat VI (1992)",
                "Véhicules": "Shuttle/Satellite",
                "Distance Min": "Capture manuelle",
                "Durée": "8h 29min EVA",
                "Résultat": "✅ Sauvetage satellite"
            },
            {
                "Mission": "ATV-1/ISS (2008)",
                "Véhicules": "ATV Jules Verne/ISS",
                "Distance Min": "Contact",
                "Durée": "Automatique",
                "Résultat": "✅ Premier ATV"
            },
            {
                "Mission": "Dragon 2/ISS (2020)",
                "Véhicules": "Crew Dragon/ISS",
                "Distance Min": "Contact",
                "Durée": "Automatique",
                "Résultat": "✅ Vol habité commercial"
            }
        ]
        
        df_historical = pd.DataFrame(historical_rdv)
        st.dataframe(df_historical, use_container_width=True)

# ==================== PAGE: POINTS LAGRANGE ====================
elif page == "💫 Points Lagrange":
    st.header("💫 Points de Lagrange")
    
    tab1, tab2, tab3 = st.tabs(["📚 Théorie", "🧮 Calculs", "🛰️ Missions"])
    
    with tab1:
        st.subheader("📚 Théorie des Points de Lagrange")
        
        st.info("""
        **Points de Lagrange:** Positions d'équilibre gravitationnel dans un système à deux corps
        
        Joseph-Louis Lagrange (1772) a identifié 5 points d'équilibre dans le problème restreint des trois corps
        """)
        
        st.write("### 🎯 Les 5 Points de Lagrange")
        
        lagrange_points = {
            "L1 - Entre les deux corps": {
                "position": "Entre Terre et Soleil (ou Terre et Lune)",
                "stabilité": "❌ Instable (nécessite station-keeping)",
                "distance": "~1.5M km du Soleil (Terre-Soleil)",
                "usage": "Observation Soleil (SOHO, ACE)",
                "dv_annuel": "2-4 m/s"
            },
            "L2 - Derrière le corps secondaire": {
                "position": "Côté nuit du corps secondaire",
                "stabilité": "❌ Instable",
                "distance": "~1.5M km de Terre (Terre-Soleil)",
                "usage": "Télescopes spatiaux (JWST, Gaia)",
                "dv_annuel": "2-4 m/s"
            },
            "L3 - Opposé au corps secondaire": {
                "position": "De l'autre côté du corps primaire",
                "stabilité": "❌ Instable",
                "distance": "Orbite opposée",
                "usage": "Théorique (peu utilisé)",
                "dv_annuel": "~10 m/s"
            },
            "L4 - 60° en avance": {
                "position": "60° devant sur l'orbite",
                "stabilité": "✅ Stable (puits gravitationnel)",
                "distance": "Même orbite, 60° devant",
                "usage": "Astéroïdes troyens, missions futures",
                "dv_annuel": "< 1 m/s"
            },
            "L5 - 60° en retard": {
                "position": "60° derrière sur l'orbite",
                "stabilité": "✅ Stable",
                "distance": "Même orbite, 60° derrière",
                "usage": "Colonies spatiales (concept O'Neill)",
                "dv_annuel": "< 1 m/s"
            }
        }
        
        for point_name, point_info in lagrange_points.items():
            with st.expander(f"💫 {point_name}"):
                for key, value in point_info.items():
                    st.write(f"**{key.title()}:** {value}")
        
        # Visualisation
        st.write("### 🗺️ Configuration Points Lagrange (Terre-Soleil)")
        
        fig = go.Figure()
        
        # Soleil
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=40, color='yellow'),
            text=['☀️ Soleil'],
            textposition='bottom center',
            name='Soleil'
        ))
        
        # Terre
        earth_x, earth_y = 1, 0
        fig.add_trace(go.Scatter(
            x=[earth_x], y=[earth_y],
            mode='markers+text',
            marker=dict(size=20, color='blue'),
            text=['🌍 Terre'],
            textposition='bottom center',
            name='Terre'
        ))
        
        # L1
        fig.add_trace(go.Scatter(
            x=[0.99], y=[0],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=['L1'],
            textposition='top center',
            name='L1'
        ))
        
        # L2
        fig.add_trace(go.Scatter(
            x=[1.01], y=[0],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=['L2'],
            textposition='top center',
            name='L2'
        ))
        
        # L3
        fig.add_trace(go.Scatter(
            x=[-1], y=[0],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='diamond'),
            text=['L3'],
            textposition='top center',
            name='L3'
        ))
        
        # L4
        fig.add_trace(go.Scatter(
            x=[0.5], y=[0.866],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='diamond'),
            text=['L4'],
            textposition='top center',
            name='L4'
        ))
        
        # L5
        fig.add_trace(go.Scatter(
            x=[0.5], y=[-0.866],
            mode='markers+text',
            marker=dict(size=15, color='green', symbol='diamond'),
            text=['L5'],
            textposition='bottom center',
            name='L5'
        ))
        
        # Orbite Terre
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Orbite Terre',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Points de Lagrange Terre-Soleil",
            xaxis_title="Distance (UA)",
            yaxis_title="Distance (UA)",
            height=600,
            showlegend=True
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🧮 Calculs Points Lagrange")
        
        st.write("### 📐 Distance L1 (Approximation)")
        
        st.latex(r"r_{L1} \approx R \left(\frac{M_2}{3M_1}\right)^{1/3}")
        
        with st.form("lagrange_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                system = st.selectbox(
                    "Système",
                    ["Terre-Soleil", "Terre-Lune", "Mars-Soleil"]
                )
            
            with col2:
                point = st.selectbox(
                    "Point de Lagrange",
                    ["L1", "L2", "L3", "L4", "L5"]
                )
            
            submitted_lagr = st.form_submit_button("🔬 Calculer")
            
            if submitted_lagr:
                # Paramètres système
                if system == "Terre-Soleil":
                    M1 = CONSTANTS['SUN_MASS']
                    M2 = CONSTANTS['EARTH_MASS']
                    R = CONSTANTS['AU']
                    body1, body2 = "Soleil", "Terre"
                elif system == "Terre-Lune":
                    M1 = CONSTANTS['EARTH_MASS']
                    M2 = CONSTANTS['MOON_MASS']
                    R = CONSTANTS['MOON_DISTANCE']
                    body1, body2 = "Terre", "Lune"
                else:  # Mars-Soleil
                    M1 = CONSTANTS['SUN_MASS']
                    M2 = CONSTANTS['MARS_MASS']
                    R = 2.279e11
                    body1, body2 = "Soleil", "Mars"
                
                # Calcul position
                if point in ["L1", "L2"]:
                    # Approximation Hill
                    r_L = R * (M2 / (3 * M1))**(1/3)
                    
                    if point == "L1":
                        distance_from_body2 = r_L
                        description = f"Entre {body1} et {body2}"
                    else:  # L2
                        distance_from_body2 = r_L
                        description = f"Derrière {body2}"
                    
                    distance_from_body1 = R - r_L if point == "L1" else R + r_L
                    
                elif point == "L3":
                    r_L = R * (5 * M2 / (12 * M1))
                    distance_from_body1 = R + r_L
                    distance_from_body2 = 2 * R
                    description = f"Opposé à {body2}"
                    
                else:  # L4 ou L5
                    distance_from_body1 = R
                    distance_from_body2 = R
                    description = f"Triangle équilatéral (60° {'devant' if point == 'L4' else 'derrière'})"
                
                # Période orbitale
                mu = CONSTANTS['G'] * M1
                period = 2 * np.pi * np.sqrt(R**3 / mu)
                
                # Delta-v annuel station-keeping
                if point in ["L1", "L2"]:
                    dv_annual = 2.5  # m/s
                elif point == "L3":
                    dv_annual = 10.0
                else:  # L4, L5
                    dv_annual = 0.5
                
                st.success("✅ Calcul terminé!")
                
                # Résultats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Distance de " + body1, f"{distance_from_body1/1e6:.2f} M km")
                with col2:
                    st.metric("Distance de " + body2, f"{distance_from_body2/1e6:.2f} M km")
                with col3:
                    st.metric("Période", f"{period/(86400*365.25):.2f} ans")
                
                st.info(f"**Position:** {description}")
                st.metric("ΔV Station-keeping (annuel)", f"{dv_annual} m/s")
                
                # Calcul transfert Terre → Point Lagrange
                if system == "Terre-Soleil" or system == "Terre-Lune":
                    st.write("### 🚀 Transfert depuis Orbite Basse Terre")
                    
                    # Delta-v approximatif
                    if system == "Terre-Soleil":
                        dv_transfer = 3100  # m/s (approximation)
                        transfer_time = 100  # jours
                    else:  # Terre-Lune
                        dv_transfer = 3800  # m/s
                        transfer_time = 5  # jours
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("ΔV Transfert", f"{dv_transfer} m/s")
                    with col2:
                        st.metric("Temps Transfert", f"{transfer_time} jours")
    
    with tab3:
        st.subheader("🛰️ Missions aux Points Lagrange")
        
        st.write("### 🚀 Missions Actuelles et Passées")
        
        missions_lagrange = [
            {
                "Mission": "SOHO",
                "Point": "L1 (Terre-Soleil)",
                "Lancement": "1995",
                "Objectif": "Observation Soleil",
                "Statut": "✅ Actif",
                "Distance": "1.5M km de Terre"
            },
            {
                "Mission": "ACE",
                "Point": "L1 (Terre-Soleil)",
                "Lancement": "1997",
                "Objectif": "Météo spatiale",
                "Statut": "✅ Actif",
                "Distance": "1.5M km de Terre"
            },
            {
                "Mission": "JWST",
                "Point": "L2 (Terre-Soleil)",
                "Lancement": "2021",
                "Objectif": "Télescope infrarouge",
                "Statut": "✅ Actif",
                "Distance": "1.5M km de Terre"
            },
            {
                "Mission": "Gaia",
                "Point": "L2 (Terre-Soleil)",
                "Lancement": "2013",
                "Objectif": "Cartographie stellaire",
                "Statut": "✅ Actif",
                "Distance": "1.5M km de Terre"
            },
            {
                "Mission": "WMAP",
                "Point": "L2 (Terre-Soleil)",
                "Lancement": "2001",
                "Objectif": "Fond diffus cosmologique",
                "Statut": "✅ Terminé (2010)",
                "Distance": "1.5M km de Terre"
            },
            {
                "Mission": "Planck",
                "Point": "L2 (Terre-Soleil)",
                "Lancement": "2009",
                "Objectif": "Cosmologie",
                "Statut": "✅ Terminé (2013)",
                "Distance": "1.5M km de Terre"
            }
        ]
        
        df_missions = pd.DataFrame(missions_lagrange)
        st.dataframe(df_missions, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🔮 Missions Futures")
        
        future_missions = [
            {
                "Mission": "Lunar Gateway",
                "Point": "NRHO (quasi-L2 lunaire)",
                "Date Prévue": "2025-2028",
                "Objectif": "Station spatiale lunaire",
                "Agence": "NASA/ESA/JAXA"
            },
            {
                "Mission": "Nancy Grace Roman",
                "Point": "L2 (Terre-Soleil)",
                "Date Prévue": "2027",
                "Objectif": "Télescope spatial",
                "Agence": "NASA"
            },
            {
                "Mission": "PLATO",
                "Point": "L2 (Terre-Soleil)",
                "Date Prévue": "2026",
                "Objectif": "Exoplanètes",
                "Agence": "ESA"
            }
        ]
        
        df_future = pd.DataFrame(future_missions)
        st.dataframe(df_future, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 💡 Avantages Points Lagrange")
        
        advantages = {
            "L1": [
                "Observation continue du Soleil",
                "Alerte précoce météo spatiale",
                "Communication Terre permanente"
            ],
            "L2": [
                "Environnement thermique stable",
                "Vue dégagée espace profond",
                "Protection thermique Terre/Lune/Soleil",
                "Idéal pour télescopes infrarouges"
            ],
            "L4/L5": [
                "Stabilité naturelle (orbites troyennes)",
                "Station-keeping minimal",
                "Sites potentiels colonies spatiales"
            ]
        }
        
        for point, advs in advantages.items():
            with st.expander(f"💫 Avantages {point}"):
                for adv in advs:
                    st.write(f"✅ {adv}")

# ==================== PAGE: FENÊTRES LANCEMENT ====================
elif page == "⏱️ Fenêtres Lancement":
    st.header("⏱️ Fenêtres de Lancement")
    
    tab1, tab2, tab3 = st.tabs(["📅 Calculateur", "🌍→🔴 Interplanétaire", "📊 Optimisation"])
    
    with tab1:
        st.subheader("📅 Calculateur Fenêtres de Lancement")
        
        st.info("""
        **Fenêtre de lancement:** Période durant laquelle un lancement peut être effectué
        pour atteindre l'orbite ou la destination souhaitée.
        
        Contraintes principales:
        - Azimut de lancement (latitude site)
        - Inclinaison orbitale cible
        - Phase orbitale (rendez-vous)
        - Éclairage (panneaux solaires)
        - Alignement planétaire (interplanétaire)
        """)
        
        with st.form("launch_window_calc"):
            st.write("### 🚀 Site de Lancement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                launch_site = st.selectbox(
                    "Site",
                    [
                        "Cap Canaveral (28.5°N)",
                        "Baïkonour (45.6°N)",
                        "Kourou (5.2°N)",
                        "Vandenberg (34.4°N)",
                        "Jiuquan (40.6°N)",
                        "Personnalisé"
                    ]
                )
                
                if launch_site == "Personnalisé":
                    site_latitude = st.number_input("Latitude (°)", -90.0, 90.0, 28.5, 0.1)
                else:
                    lat_dict = {
                        "Cap Canaveral (28.5°N)": 28.5,
                        "Baïkonour (45.6°N)": 45.6,
                        "Kourou (5.2°N)": 5.2,
                        "Vandenberg (34.4°N)": 34.4,
                        "Jiuquan (40.6°N)": 40.6
                    }
                    site_latitude = lat_dict[launch_site]
            
            with col2:
                target_inclination = st.slider("Inclinaison Cible (°)", 0.0, 180.0, 51.6, 0.1)
                target_altitude = st.number_input("Altitude Cible (km)", 200.0, 2000.0, 400.0, 10.0)
            
            st.write("### 🎯 Type de Mission")
            
            mission_type_window = st.selectbox(
                "Type",
                ["Insertion Directe", "Rendez-vous (ISS)", "Orbite Polaire", "GTO"]
            )
            
            launch_date = st.date_input("Date de Lancement Souhaitée", datetime.now())
            
            submitted_window = st.form_submit_button("🔬 Calculer Fenêtres")
            
            if submitted_window:
                with st.spinner("Calcul en cours..."):
                    # Vérification contraintes physiques
                    min_inclination = abs(site_latitude)
                    
                    if target_inclination < min_inclination:
                        st.error(f"❌ Impossible: Inclinaison minimale = {min_inclination:.1f}° pour cette latitude")
                    else:
                        st.success("✅ Fenêtre calculée!")
                        
                        # Calcul azimut de lancement
                        i_rad = target_inclination * np.pi / 180
                        lat_rad = site_latitude * np.pi / 180
                        
                        # Azimut (direction de lancement)
                        if abs(site_latitude) < 90:
                            cos_az = np.cos(i_rad) / np.cos(lat_rad)
                            if abs(cos_az) <= 1:
                                azimuth = np.arccos(cos_az) * 180 / np.pi
                            else:
                                azimuth = 0 if target_inclination >= 90 else 90
                        else:
                            azimuth = 0
                        
                        # Vitesse rotation Terre à la latitude
                        earth_circum = 2 * np.pi * CONSTANTS['earth_radius']
                        v_earth = earth_circum * np.cos(lat_rad) / 86400
                        
                        # Vitesse orbitale cible
                        r = CONSTANTS['earth_radius'] + target_altitude * 1000
                        v_orbit = np.sqrt(CONSTANTS['earth_mu'] / r)
                        
                        # Bonus vitesse rotation
                        v_bonus = v_earth * np.sin(azimuth * np.pi / 180)
                        
                        # Delta-v économisé
                        dv_saved = abs(v_bonus)
                        
                        # Durée fenêtre (approximation)
                        if mission_type_window == "Rendez-vous (ISS)":
                            window_duration = 5  # minutes (instantanée)
                            windows_per_day = 2
                        elif mission_type_window == "Orbite Polaire":
                            window_duration = 60  # minutes
                            windows_per_day = 2
                        else:
                            window_duration = 120  # minutes
                            windows_per_day = 1
                        
                        # Résultats
                        st.write("### 📊 Résultats")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Azimut Lancement", f"{azimuth:.1f}°")
                        with col2:
                            st.metric("Durée Fenêtre", f"{window_duration} min")
                        with col3:
                            st.metric("Fenêtres/Jour", windows_per_day)
                        with col4:
                            st.metric("ΔV Économisé", f"{dv_saved:.0f} m/s")
                        
                        st.write("### 📅 Prochaines Fenêtres")
                        
                        # Génération fenêtres
                        windows_data = []
                        base_date = datetime.combine(launch_date, datetime.min.time())
                        
                        for day in range(7):
                            date = base_date + timedelta(days=day)
                            
                            for window_num in range(windows_per_day):
                                # Heure approximative
                                if mission_type_window == "Rendez-vous (ISS)":
                                    hour = 6 + window_num * 12 + np.random.randint(-1, 2)
                                else:
                                    hour = 10 + window_num * 12
                                
                                minute = np.random.randint(0, 60)
                                
                                window_time = date + timedelta(hours=hour, minutes=minute)
                                
                                windows_data.append({
                                    "Date": window_time.strftime("%Y-%m-%d"),
                                    "Heure": window_time.strftime("%H:%M UTC"),
                                    "Durée": f"{window_duration} min",
                                    "Azimut": f"{azimuth:.1f}°",
                                    "Type": "Primaire" if window_num == 0 else "Secondaire"
                                })
                        
                        df_windows = pd.DataFrame(windows_data)
                        st.dataframe(df_windows, use_container_width=True)
                        
                        # Visualisation azimut
                        st.write("### 🧭 Direction de Lancement")
                        
                        fig = go.Figure()
                        
                        # Cercle boussole
                        theta_compass = np.linspace(0, 2*np.pi, 100)
                        x_compass = np.cos(theta_compass)
                        y_compass = np.sin(theta_compass)
                        
                        fig.add_trace(go.Scatter(
                            x=x_compass, y=y_compass,
                            mode='lines',
                            line=dict(color='gray'),
                            showlegend=False
                        ))
                        
                        # Direction lancement
                        az_rad = azimuth * np.pi / 180
                        x_launch = np.sin(az_rad)
                        y_launch = np.cos(az_rad)
                        
                        fig.add_trace(go.Scatter(
                            x=[0, x_launch], y=[0, y_launch],
                            mode='lines+markers',
                            line=dict(color='red', width=4),
                            marker=dict(size=15),
                            name='Azimut Lancement'
                        ))
                        
                        # Points cardinaux
                        fig.add_annotation(x=0, y=1.15, text="N", showarrow=False, font=dict(size=16))
                        fig.add_annotation(x=1.15, y=0, text="E", showarrow=False, font=dict(size=16))
                        fig.add_annotation(x=0, y=-1.15, text="S", showarrow=False, font=dict(size=16))
                        fig.add_annotation(x=-1.15, y=0, text="O", showarrow=False, font=dict(size=16))
                        
                        fig.update_layout(
                            title=f"Azimut de Lancement: {azimuth:.1f}°",
                            xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False, showticklabels=False),
                            height=500
                        )
                        fig.update_yaxes(scaleanchor="x", scaleratio=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🌍→🔴 Fenêtres Interplanétaires")
        
        st.info("""
        **Fenêtres interplanétaires:** Périodes optimales pour les transferts entre planètes
        
        Dépendent de:
        - Position relative des planètes (phase angle)
        - Delta-v disponible
        - Durée de transit acceptable
        """)
        
        st.write("### 🚀 Fenêtres Terre → Mars")
        
        mars_windows = [
            {
                "Fenêtre": "2024",
                "Date Optimale": "Oct-Nov 2024",
                "Phase Angle": "44°",
                "Delta-v": "5.6 km/s",
                "Durée Transit": "245 jours",
                "Type": "Hohmann"
            },
            {
                "Fenêtre": "2026",
                "Date Optimale": "Nov-Déc 2026",
                "Phase Angle": "44°",
                "Delta-v": "5.8 km/s",
                "Durée Transit": "250 jours",
                "Type": "Hohmann"
            },
            {
                "Fenêtre": "2028",
                "Date Optimale": "Déc 2028-Jan 2029",
                "Phase Angle": "44°",
                "Delta-v": "5.5 km/s",
                "Durée Transit": "240 jours",
                "Type": "Hohmann"
            },
            {
                "Fenêtre": "2031",
                "Date Optimale": "Jan-Fév 2031",
                "Phase Angle": "44°",
                "Delta-v": "5.9 km/s",
                "Durée Transit": "255 jours",
                "Type": "Hohmann"
            },
            {
                "Fenêtre": "2033",
                "Date Optimale": "Fév-Mars 2033",
                "Phase Angle": "44°",
                "Delta-v": "5.7 km/s",
                "Durée Transit": "248 jours",
                "Type": "Hohmann"
            }
        ]
        
        df_mars = pd.DataFrame(mars_windows)
        st.dataframe(df_mars, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🌍→🌕 Fenêtres Terre → Vénus")
        
        venus_windows = [
            {
                "Fenêtre": "2025",
                "Date Optimale": "Juin 2025",
                "Delta-v": "5.3 km/s",
                "Durée Transit": "150 jours"
            },
            {
                "Fenêtre": "2026",
                "Date Optimale": "Déc 2026",
                "Delta-v": "5.5 km/s",
                "Durée Transit": "155 jours"
            },
            {
                "Fenêtre": "2028",
                "Date Optimale": "Juil 2028",
                "Delta-v": "5.2 km/s",
                "Durée Transit": "148 jours"
            }
        ]
        
        df_venus = pd.DataFrame(venus_windows)
        st.dataframe(df_venus, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Analyse Phase Angle")
        
        st.latex(r"\text{Phase Angle} = \arccos\left(\frac{r_1^2 + r_2^2 - d^2}{2r_1r_2}\right)")
        
        st.write("""
        **Phase angle optimal pour Mars:** ~44° (transfert Hohmann)
        
        **Synodique Terre-Mars:** 780 jours (~26 mois)
        """)
    
    with tab3:
        st.subheader("📊 Optimisation Fenêtres")
        
        st.write("### 🎯 Critères d'Optimisation")
        
        criteria = {
            "Delta-v Minimal": {
                "priorité": "Économie propergol",
                "impact": "Masse utile maximale",
                "contrainte": "Durée transit prolongée"
            },
            "Durée Minimale": {
                "priorité": "Temps de mission court",
                "impact": "Support vie réduit",
                "contrainte": "Delta-v élevé"
            },
            "Énergie C3 Minimale": {
                "priorité": "Capacité lanceur",
                "impact": "Masse satellisable maximale",
                "contrainte": "Flexibilité limitée"
            },
            "Fenêtre Large": {
                "priorité": "Flexibilité opérationnelle",
                "impact": "Contingence problèmes",
                "contrainte": "Performance sous-optimale"
            }
        }
        
        for criterion, details in criteria.items():
            with st.expander(f"🎯 {criterion}"):
                for key, value in details.items():
                    st.write(f"**{key.title()}:** {value}")
        
        st.markdown("---")
        
        st.write("### 📈 Graphique Porkchop Plot")
        
        st.info("""
        **Porkchop Plot:** Graphique montrant le delta-v requis en fonction
        des dates de départ et d'arrivée pour un transfert interplanétaire.
        
        Permet d'identifier visuellement les fenêtres optimales.
        """)
        
        # Simulation porkchop plot simplifié
        departure_days = np.linspace(0, 60, 50)
        arrival_days = np.linspace(180, 300, 50)
        
        X, Y = np.meshgrid(departure_days, arrival_days)
        
        # Delta-v simulé (formule simplifiée)
        Z = 5.5 + 0.5 * np.sin((X - 30) / 10) + 0.3 * np.sin((Y - 240) / 20) + 0.2 * np.random.randn(50, 50)
        
        fig = go.Figure(data=go.Contour(
            x=departure_days,
            y=arrival_days,
            z=Z,
            colorscale='Viridis',
            contours=dict(
                start=5.0,
                end=6.5,
                size=0.1
            ),
            colorbar=dict(title="Delta-v (km/s)")
        ))
        
        fig.update_layout(
            title="Porkchop Plot - Terre → Mars (Exemple)",
            xaxis_title="Jours après ouverture fenêtre (Départ)",
            yaxis_title="Durée transit (jours)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Lecture:** 
        - Zones bleues/vertes = Delta-v optimal
        - Zones jaunes/rouges = Delta-v élevé
        - Centre "en forme de côtelette de porc" = fenêtre optimale
        """)

# ==================== PAGE: ANALYSES ====================
elif page == "📈 Analyses":
    st.header("📈 Analyses et Statistiques")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'Ensemble", "🛰️ Flotte", "📡 Performance", "💰 Budget"])
    
    with tab1:
        st.subheader("📊 Vue d'Ensemble du Système")
        
        if not st.session_state.space_system['satellites']:
            st.info("💡 Créez des satellites pour voir les analyses")
        else:
            # Statistiques globales
            total_mass = sum(s['masses']['total_mass'] for s in st.session_state.space_system['satellites'].values())
            total_power = sum(s['power']['generation'] for s in st.session_state.space_system['satellites'].values())
            total_data = sum(s['performance']['data_transmitted'] for s in st.session_state.space_system['satellites'].values())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Masse Totale Flotte", f"{total_mass:,.0f} kg")
            with col2:
                st.metric("Puissance Totale", f"{total_power:,.0f} W")
            with col3:
                st.metric("Données Transmises", f"{total_data:.1f} TB")
            with col4:
                avg_lifetime = sum(s['mission']['lifetime_years'] for s in st.session_state.space_system['satellites'].values()) / len(st.session_state.space_system['satellites'])
                st.metric("Durée Vie Moyenne", f"{avg_lifetime:.1f} ans")
            
            st.markdown("---")
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Distribution Masses")
                
                masses = [s['masses']['total_mass'] for s in st.session_state.space_system['satellites'].values()]
                names = [s['name'] for s in st.session_state.space_system['satellites'].values()]
                
                fig = go.Figure(data=[go.Bar(
                    x=names,
                    y=masses,
                    marker_color='lightblue'
                )])
                
                fig.update_layout(
                    title="Masse par Satellite",
                    xaxis_title="Satellite",
                    yaxis_title="Masse (kg)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### ⚡ Distribution Puissance")
                
                power = [s['power']['generation'] for s in st.session_state.space_system['satellites'].values()]
                
                fig = go.Figure(data=[go.Bar(
                    x=names,
                    y=power,
                    marker_color='lightcoral'
                )])
                
                fig.update_layout(
                    title="Puissance par Satellite",
                    xaxis_title="Satellite",
                    yaxis_title="Puissance (W)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Répartition par type de mission
            st.write("### 🎯 Répartition par Type de Mission")
            
            mission_counts = {}
            for sat in st.session_state.space_system['satellites'].values():
                m_type = sat['mission']['type']
                mission_counts[m_type] = mission_counts.get(m_type, 0) + 1
            
            fig = go.Figure(data=[go.Pie(
                labels=list(mission_counts.keys()),
                values=list(mission_counts.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title="Types de Missions",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🛰️ Analyse de Flotte")
        
        if st.session_state.space_system['satellites']:
            # Tableau récapitulatif
            st.write("### 📋 Récapitulatif Flotte")
            
            fleet_data = []
            for sat in st.session_state.space_system['satellites'].values():
                fleet_data.append({
                    "Nom": sat['name'],
                    "Masse (kg)": f"{sat['masses']['total_mass']:.0f}",
                    "Puissance (W)": sat['power']['generation'],
                    "Type Mission": sat['mission']['type'],
                    "Statut": sat['status'],
                    "Durée Vie (ans)": sat['mission']['lifetime_years'],
                    "Données (GB)": f"{sat['performance']['data_transmitted']:.1f}"
                })
            
            df_fleet = pd.DataFrame(fleet_data)
            st.dataframe(df_fleet, use_container_width=True)
            
            # Analyse propulsion
            st.write("### 🚀 Systèmes de Propulsion")
            
            propulsion_types = {}
            for sat in st.session_state.space_system['satellites'].values():
                p_type = sat['propulsion']['type']
                propulsion_types[p_type] = propulsion_types.get(p_type, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=list(propulsion_types.keys()),
                    values=list(propulsion_types.values())
                )])
                
                fig.update_layout(
                    title="Types de Propulsion",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Isp moyen par type
                isp_by_type = {}
                count_by_type = {}
                
                for sat in st.session_state.space_system['satellites'].values():
                    p_type = sat['propulsion']['type']
                    isp = sat['propulsion']['isp']
                    
                    if p_type not in isp_by_type:
                        isp_by_type[p_type] = 0
                        count_by_type[p_type] = 0
                    
                    isp_by_type[p_type] += isp
                    count_by_type[p_type] += 1
                
                avg_isp = {k: v/count_by_type[k] for k, v in isp_by_type.items()}
                
                fig = go.Figure(data=[go.Bar(
                    x=list(avg_isp.keys()),
                    y=list(avg_isp.values()),
                    marker_color='lightgreen'
                )])
                
                fig.update_layout(
                    title="Isp Moyen par Type",
                    xaxis_title="Type Propulsion",
                    yaxis_title="Isp (s)",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline
            st.write("### 📅 Timeline Durée de Vie")
            
            fig = go.Figure()
            
            for i, sat in enumerate(st.session_state.space_system['satellites'].values()):
                created = datetime.fromisoformat(sat['created_at'])
                end_date = created + timedelta(days=sat['mission']['lifetime_years']*365.25)
                
                fig.add_trace(go.Scatter(
                    x=[created, end_date],
                    y=[sat['name'], sat['name']],
                    mode='lines+markers',
                    line=dict(width=10),
                    marker=dict(size=10),
                    name=sat['name']
                ))
            
            fig.update_layout(
                title="Durée de Vie des Satellites",
                xaxis_title="Date",
                yaxis_title="Satellite",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 Créez des satellites pour voir l'analyse de flotte")
    
    with tab3:
        st.subheader("📡 Analyse de Performance")
        
        if st.session_state.space_system['satellites']:
            # Performance par satellite
            st.write("### 🎯 Performance Opérationnelle")
            
            perf_data = []
            for sat in st.session_state.space_system['satellites'].values():
                perf_data.append({
                    "Satellite": sat['name'],
                    "Heures Opération": sat['mission']['operational_hours'],
                    "Orbites Complétées": sat['performance']['orbits_completed'],
                    "Manœuvres": sat['performance']['maneuvers_executed'],
                    "Données (GB)": sat['performance']['data_transmitted'],
                    "Batterie (%)": sat['telemetry']['battery_level']
                })
            
            df_perf = pd.DataFrame(perf_data)
            st.dataframe(df_perf, use_container_width=True)
            
            # Graphiques performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### ⏱️ Temps Opérationnel")
                
                names = [s['name'] for s in st.session_state.space_system['satellites'].values()]
                hours = [s['mission']['operational_hours'] for s in st.session_state.space_system['satellites'].values()]
                
                fig = go.Figure(data=[go.Bar(
                    x=names,
                    y=hours,
                    marker_color='lightblue'
                )])
                
                fig.update_layout(
                    xaxis_title="Satellite",
                    yaxis_title="Heures",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### 💾 Données Transmises")
                
                data_transmitted = [s['performance']['data_transmitted'] for s in st.session_state.space_system['satellites'].values()]
                
                fig = go.Figure(data=[go.Bar(
                    x=names,
                    y=data_transmitted,
                    marker_color='lightgreen'
                )])
                
                fig.update_layout(
                    xaxis_title="Satellite",
                    yaxis_title="Données (GB)",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # État batteries
            st.write("### 🔋 État des Batteries")
            
            battery_levels = [s['telemetry']['battery_level'] for s in st.session_state.space_system['satellites'].values()]
            
            fig = go.Figure()
            
            for i, (name, level) in enumerate(zip(names, battery_levels)):
                color = 'green' if level > 80 else 'orange' if level > 50 else 'red'
                
                fig.add_trace(go.Bar(
                    x=[name],
                    y=[level],
                    marker_color=color,
                    showlegend=False
                ))
            
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Optimal")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Alerte")
            fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Critique")
            
            fig.update_layout(
                title="Niveau Batterie par Satellite",
                xaxis_title="Satellite",
                yaxis_title="Niveau (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Efficacité
            st.write("### 📊 Efficacité Opérationnelle")
            
            efficiency_data = []
            for sat in st.session_state.space_system['satellites'].values():
                # Calcul efficacité (données/heure)
                if sat['mission']['operational_hours'] > 0:
                    efficiency = sat['performance']['data_transmitted'] / sat['mission']['operational_hours']
                else:
                    efficiency = 0
                
                efficiency_data.append({
                    "Satellite": sat['name'],
                    "Efficacité (GB/h)": f"{efficiency:.2f}",
                    "Utilisation (%)": f"{(sat['mission']['operational_hours'] / (sat['mission']['lifetime_years'] * 8760) * 100):.1f}" if sat['mission']['lifetime_years'] > 0 else "0"
                })
            
            df_efficiency = pd.DataFrame(efficiency_data)
            st.dataframe(df_efficiency, use_container_width=True)
        else:
            st.info("💡 Créez des satellites pour voir les performances")
    
    with tab4:
        st.subheader("💰 Analyse Budgétaire")
        
        if st.session_state.space_system['satellites']:
            st.write("### 💵 Budget Delta-v")
            
            # Calcul budget delta-v pour chaque satellite
            budget_data = []
            total_dv_available = 0
            
            for sat in st.session_state.space_system['satellites'].values():
                # Calcul delta-v disponible (Tsiolkovsky)
                g0 = 9.80665
                ve = sat['propulsion']['isp'] * g0
                m0 = sat['masses']['total_mass']
                mf = sat['masses']['dry_mass'] + sat['masses']['payload_mass']
                
                if mf > 0 and m0 > mf:
                    dv = ve * np.log(m0 / mf)
                else:
                    dv = 0
                
                total_dv_available += dv
                
                # Allocation typique
                allocation = {
                    'insertion': dv * 0.15,
                    'station_keeping': dv * 0.60,
                    'maneuvers': dv * 0.15,
                    'deorbit': dv * 0.10
                }
                
                budget_data.append({
                    "Satellite": sat['name'],
                    "ΔV Total (m/s)": f"{dv:.0f}",
                    "Insertion (m/s)": f"{allocation['insertion']:.0f}",
                    "Station-keeping (m/s)": f"{allocation['station_keeping']:.0f}",
                    "Manœuvres (m/s)": f"{allocation['maneuvers']:.0f}",
                    "Désorbitation (m/s)": f"{allocation['deorbit']:.0f}"
                })
            
            df_budget = pd.DataFrame(budget_data)
            st.dataframe(df_budget, use_container_width=True)
            
            st.metric("Delta-v Total Flotte", f"{total_dv_available:,.0f} m/s")
            
            # Graphique allocation
            st.write("### 📊 Allocation Budget Delta-v")
            
            allocation_labels = ['Insertion', 'Station-keeping', 'Manœuvres', 'Désorbitation']
            allocation_values = [15, 60, 15, 10]
            
            fig = go.Figure(data=[go.Pie(
                labels=allocation_labels,
                values=allocation_values,
                hole=0.3
            )])
            
            fig.update_layout(
                title="Répartition Typique Budget ΔV",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Budget propergol
            st.write("### ⛽ Budget Propergol")
            
            propellant_data = []
            total_propellant = 0
            
            for sat in st.session_state.space_system['satellites'].values():
                initial_prop = sat['masses']['propellant_mass']
                total_propellant += initial_prop
                
                # Consommation estimée (simplifiée)
                consumption_rate = initial_prop / (sat['mission']['lifetime_years'] * 365.25)
                
                propellant_data.append({
                    "Satellite": sat['name'],
                    "Propergol Initial (kg)": f"{initial_prop:.0f}",
                    "Consommation/jour (kg)": f"{consumption_rate:.2f}",
                    "Autonomie (jours)": f"{initial_prop/consumption_rate:.0f}" if consumption_rate > 0 else "N/A"
                })
            
            df_propellant = pd.DataFrame(propellant_data)
            st.dataframe(df_propellant, use_container_width=True)
            
            st.metric("Propergol Total Flotte", f"{total_propellant:,.0f} kg")
            
            # Coût estimatif
            st.write("### 💰 Estimation Coûts")
            
            st.info("""
            **Coûts moyens estimatifs:**
            - Lancement LEO: ~5,000 $/kg
            - Satellite LEO: ~50,000 $/kg
            - Propergol: ~500 $/kg
            - Opérations: ~2M $/an/satellite
            """)
            
            total_satellite_cost = total_mass * 50000 / 1000  # M$
            total_launch_cost = total_mass * 5000 / 1000  # M$
            total_propellant_cost = total_propellant * 500 / 1000  # k$
            total_ops_cost = len(st.session_state.space_system['satellites']) * 2 * avg_lifetime  # M$
            
            total_program_cost = total_satellite_cost + total_launch_cost + total_ops_cost
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Coût Satellites", f"${total_satellite_cost:.0f}M")
            with col2:
                st.metric("Coût Lancement", f"${total_launch_cost:.0f}M")
            with col3:
                st.metric("Coût Opérations", f"${total_ops_cost:.0f}M")
            with col4:
                st.metric("TOTAL Programme", f"${total_program_cost:.0f}M")
            
            # Graphique répartition coûts
            fig = go.Figure(data=[go.Pie(
                labels=['Satellites', 'Lancement', 'Propergol', 'Opérations'],
                values=[total_satellite_cost, total_launch_cost, total_propellant_cost/1000, total_ops_cost]
            )])
            
            fig.update_layout(
                title="Répartition Coûts Programme",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 Créez des satellites pour voir l'analyse budgétaire")

# ==================== PAGE: ESPACE PROFOND ====================
elif page == "🌌 Espace Profond":
    st.header("🌌 Missions Espace Profond")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Trajectoires", "🌍 Assistance Gravitationnelle", "🎯 Missions", "📊 Records"])
    
    with tab1:
        st.subheader("🚀 Trajectoires Espace Profond")
        
        st.info("""
        **Espace profond:** Région au-delà de l'orbite de la Lune (~380,000 km)
        
        Défis:
        - Delta-v élevé
        - Durées de mission longues (années)
        - Communications à grande distance
        - Autonomie nécessaire
        - Radiation interplanétaire
        """)
        
        st.write("### 🎯 Vitesses Caractéristiques")
        
        velocities_data = [
            {
                "Destination": "Échappement Terre",
                "C3 (km²/s²)": "0",
                "Vitesse (km/s)": "11.2",
                "Description": "Vitesse minimale pour quitter Terre"
            },
            {
                "Destination": "Lune",
                "C3 (km²/s²)": "-2",
                "Vitesse (km/s)": "10.9",
                "Description": "Capture lunaire"
            },
            {
                "Destination": "Mars",
                "C3 (km²/s²)": "10-20",
                "Vitesse (km/s)": "11.5-11.9",
                "Description": "Transfert Hohmann optimal"
            },
            {
                "Destination": "Vénus",
                "C3 (km²/s²)": "8-15",
                "Vitesse (km/s)": "11.4-11.7",
                "Description": "Transfert Hohmann"
            },
            {
                "Destination": "Jupiter",
                "C3 (km²/s²)": "80-90",
                "Vitesse (km/s)": "14.0-14.5",
                "Description": "Avec assistance grav."
            },
            {
                "Destination": "Sortie Système Solaire",
                "C3 (km²/s²)": "> 140",
                "Vitesse (km/s)": "> 16.7",
                "Description": "Vitesse d'échappement solaire"
            }
        ]
        
        df_velocities = pd.DataFrame(velocities_data)
        st.dataframe(df_velocities, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🧮 Calculateur C3")
        
        st.latex(r"C_3 = v_\infty^2 = v^2 - v_{esc}^2")
        
        with st.form("c3_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                departure_velocity = st.number_input("Vitesse Départ (km/s)", 11.0, 20.0, 11.5, 0.1)
            
            with col2:
                v_escape = 11.186  # km/s Terre
                st.metric("Vitesse Échappement Terre", f"{v_escape} km/s")
            
            submitted_c3 = st.form_submit_button("🔬 Calculer C3")
            
            if submitted_c3:
                v_inf = np.sqrt((departure_velocity**2) - (v_escape**2))
                c3 = v_inf**2
                
                st.success("✅ Calcul terminé!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("v∞", f"{v_inf:.2f} km/s")
                with col2:
                    st.metric("C3", f"{c3:.1f} km²/s²")
                
                # Destination possible
                if c3 < 5:
                    destination = "Lune / Orbites haute Terre"
                elif c3 < 20:
                    destination = "Mars / Vénus (optimal)"
                elif c3 < 50:
                    destination = "Mars / Vénus (rapide)"
                elif c3 < 100:
                    destination = "Ceinture astéroïdes / Jupiter (lent)"
                else:
                    destination = "Planètes extérieures / Espace interstellaire"
                
                st.info(f"**Destination possible avec C3 = {c3:.1f}:** {destination}")
    
    with tab2:
        st.subheader("🌍 Assistance Gravitationnelle (Gravity Assist)")
        
        st.info("""
        **Assistance gravitationnelle (Gravity Assist / Slingshot):**
        
        Utilisation de la gravité d'une planète pour:
        - Modifier trajectoire sans propulsion
        - Augmenter ou diminuer vitesse
        - Changer plan orbital
        
        **Avantages:**
        - Économie de propergol massive
        - Accès destinations lointaines
        - Durées mission réduites
        """)
        
        st.write("### 🎯 Principe")
        
        st.latex(r"\Delta v_{max} = 2v_{planet}")
        
        st.write("""
        Dans le référentiel de la planète:
        - Magnitude vitesse conservée
        - Direction changée (déviation)
        
        Dans le référentiel solaire:
        - Gain/perte vitesse possible
        - Dépend angle approche
        """)
        
        st.markdown("---")
        
        st.write("### 🪐 Planètes pour Assistance Gravitationnelle")
        
        planets_ga = [
            {
                "Planète": "Vénus",
                "Vitesse Orbitale (km/s)": "35.0",
                "Δv Max Gain (km/s)": "7.0",
                "Usage": "Missions internes (Mercure) ou Mars"
            },
            {
                "Planète": "Terre",
                "Vitesse Orbitale (km/s)": "29.8",
                "Δv Max Gain (km/s)": "6.0",
                "Usage": "Boost vers planètes extérieures"
            },
            {
                "Planète": "Mars",
                "Vitesse Orbitale (km/s)": "24.1",
                "Δv Max Gain (km/s)": "4.8",
                "Usage": "Ceinture astéroïdes"
            },
            {
                "Planète": "Jupiter",
                "Vitesse Orbitale (km/s)": "13.1",
                "Δv Max Gain (km/s)": "26.2",
                "Usage": "Grand boost vers espace profond"
            },
            {
                "Planète": "Saturne",
                "Vitesse Orbitale (km/s)": "9.7",
                "Δv Max Gain (km/s)": "19.4",
                "Usage": "Planètes extérieures"
            }
        ]
        
        df_planets_ga = pd.DataFrame(planets_ga)
        st.dataframe(df_planets_ga, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🚀 Missions Célèbres avec Gravity Assist")
        
        missions_ga = {
            "Voyager 2 (1977-1989)": {
                "route": "Terre → Jupiter → Saturne → Uranus → Neptune",
                "assists": "4 assistances gravitationnelles",
                "exploit": "Grand Tour des planètes extérieures",
                "vitesse_finale": "15.4 km/s (héliocentrique)"
            },
            "Galileo (1989-1995)": {
                "route": "Terre → Vénus → Terre → Terre → Jupiter",
                "assists": "VEEGA (Venus-Earth-Earth Gravity Assist)",
                "exploit": "Première sonde orbitale Jupiter",
                "gain": "~15 km/s delta-v économisé"
            },
            "Cassini (1997-2004)": {
                "route": "Terre → Vénus → Vénus → Terre → Jupiter → Saturne",
                "assists": "4 planètes, 6 assists",
                "exploit": "Mission Saturne 13 ans",
                "distance": "3.5 milliards km"
            },
            "Messenger (2004-2011)": {
                "route": "Terre → Terre → Vénus → Vénus → Mercure (x3)",
                "assists": "Freinage pour capture Mercure",
                "exploit": "Première orbite Mercure",
                "particularité": "Assists pour ralentir"
            },
            "New Horizons (2006-2015)": {
                "route": "Terre → Jupiter → Pluton",
                "assists": "Jupiter (+4 km/s)",
                "exploit": "Plus rapide vaisseau lancé",
                "vitesse": "58,000 km/h (record Terre)"
            },
            "Parker Solar Probe (2018-...)": {
                "route": "7 survols Vénus prévus",
                "assists": "Freinage progressif vers Soleil",
                "exploit": "Approche la plus proche Soleil",
                "particularité": "Assists pour perdre vitesse orbitale"
            }
        }
        
        for mission, details in missions_ga.items():
            with st.expander(f"🚀 {mission}"):
                for key, value in details.items():
                    st.write(f"**{key.title()}:** {value}")
        
        st.markdown("---")
        
        st.write("### 📊 Comparaison Direct vs Gravity Assist")
        
        comparison_ga = [
            {
                "Destination": "Jupiter",
                "Direct ΔV (km/s)": "9.0",
                "Avec GA ΔV (km/s)": "6.3",
                "Économie": "30%",
                "Durée Direct": "2.7 ans",
                "Durée GA": "6 ans"
            },
            {
                "Destination": "Saturne",
                "Direct ΔV (km/s)": "10.5",
                "Avec GA ΔV (km/s)": "7.0",
                "Économie": "33%",
                "Durée Direct": "6 ans",
                "Durée GA": "7-9 ans"
            },
            {
                "Destination": "Neptune",
                "Direct ΔV (km/s)": "13.0",
                "Avec GA ΔV (km/s)": "8.5",
                "Économie": "35%",
                "Durée Direct": "30 ans",
                "Durée GA": "12 ans"
            }
        ]
        
        df_comparison = pd.DataFrame(comparison_ga)
        st.dataframe(df_comparison, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Missions Espace Profond Notables")
        
        st.write("### 🏆 Missions Historiques")
        
        historic_missions = [
            {
                "Mission": "Voyager 1 & 2",
                "Lancement": "1977",
                "Destination": "Planètes extérieures",
                "Statut": "✅ Actives (espace interstellaire)",
                "Distance": "> 24 milliards km",
                "Records": "Objets humains les plus éloignés"
            },
            {
                "Mission": "Pioneer 10 & 11",
                "Lancement": "1972-1973",
                "Destination": "Jupiter, Saturne",
                "Statut": "📡 Contact perdu",
                "Distance": "> 20 milliards km",
                "Records": "Premières sondes Jupiter/Saturne"
            },
            {
                "Mission": "Cassini-Huygens",
                "Lancement": "1997",
                "Destination": "Saturne",
                "Statut": "✅ Terminé (2017)",
                "Durée": "13 ans orbite Saturne",
                "Records": "Atterrissage Titan (Huygens)"
            },
            {
                "Mission": "New Horizons",
                "Lancement": "2006",
                "Destination": "Pluton, Arrokoth",
                "Statut": "✅ Active",
                "Distance": "> 8 milliards km",
                "Records": "Premier survol Pluton (2015)"
            },
            {
                "Mission": "Juno",
                "Lancement": "2011",
                "Destination": "Jupiter",
                "Statut": "✅ Active",
                "Orbite": "Polaire Jupiter",
                "Records": "Étude structure interne Jupiter"
            },
            {
                "Mission": "Parker Solar Probe",
                "Lancement": "2018",
                "Destination": "Soleil",
                "Statut": "✅ Active",
                "Distance Min": "6.2 millions km du Soleil",
                "Records": "Objet le plus rapide (430,000 km/h)"
            }
        ]
        
        df_historic = pd.DataFrame(historic_missions)
        st.dataframe(df_historic, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🔮 Missions Futures")
        
        future_deep_space = [
            {
                "Mission": "Europa Clipper",
                "Lancement": "2024",
                "Destination": "Europe (lune Jupiter)",
                "Objectif": "Recherche conditions habitabilité",
                "Arrivée": "2030"
            },
            {
                "Mission": "Dragonfly",
                "Lancement": "2027",
                "Destination": "Titan (lune Saturne)",
                "Objectif": "Drone exploration surface",
                "Arrivée": "2034"
            },
            {
                "Mission": "Interstellar Probe",
                "Lancement": "~2030s",
                "Destination": "Héliosphère / Espace interstellaire",
                "Objectif": "Étude frontière système solaire",
                "Distance": "> 1000 UA"
            },
            {
                "Mission": "Uranus Orbiter",
                "Lancement": "~2030s",
                "Destination": "Uranus",
                "Objectif": "Première orbite Uranus",
                "Particularité": "Géante glace peu explorée"
            }
        ]
        
        df_future_ds = pd.DataFrame(future_deep_space)
        st.dataframe(df_future_ds, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Records Espace Profond")
        
        st.write("### 🏆 Records de Distance")
        
        distance_records = [
            {
                "Record": "Plus éloigné de Terre",
                "Objet": "Voyager 1",
                "Valeur": "~24 milliards km",
                "Date": "Actuel (2024)",
                "Vitesse": "17 km/s (héliocentrique)"
            },
            {
                "Record": "Signal le plus lointain",
                "Objet": "Voyager 1",
                "Valeur": "22h 30min lumière",
                "Date": "Actuel",
                "Temps AR": "~45 heures"
            },
            {
                "Record": "Mission plus longue",
                "Objet": "Voyager 2",
                "Valeur": "47 ans",
                "Date": "1977-actuel",
                "Statut": "Toujours active"
            },
            {
                "Record": "Plus rapide (Soleil)",
                "Objet": "Parker Solar Probe",
                "Valeur": "430,000 km/h",
                "Date": "2021",
                "Context": "Périhélie"
            },
            {
                "Record": "Plus rapide (Terre)",
                "Objet": "New Horizons",
                "Valeur": "58,000 km/h",
                "Date": "Lancement 2006",
                "Context": "Vitesse de fuite"
            }
        ]
        
        df_records = pd.DataFrame(distance_records)
        st.dataframe(df_records, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🌟 Records Techniques")
        
        technical_records = [
            {
                "Record": "Approche plus proche Soleil",
                "Mission": "Parker Solar Probe",
                "Valeur": "6.16 millions km",
                "Température": "~1,400°C",
                "Protection": "Bouclier thermique"
            },
            {
                "Record": "Puissance à distance",
                "Mission": "New Horizons (Pluton)",
                "Valeur": "200 W à 5 Md km",
                "Source": "RTG (Plutonium-238)",
                "Débit": "1-2 kbps"
            },
            {
                "Record": "Durée vol vers planète",
                "Mission": "Voyager 2 → Neptune",
                "Valeur": "12 ans",
                "Lancement": "1977",
                "Arrivée": "1989"
            },
            {
                "Record": "Plus longue orbite planète",
                "Mission": "Cassini (Saturne)",
                "Valeur": "13 ans",
                "Période": "2004-2017",
                "Orbites": "294 orbites Saturne"
            },
            {
                "Record": "Atterrissage plus éloigné",
                "Mission": "Huygens (Titan)",
                "Valeur": "1.4 Md km de Terre",
                "Date": "2005",
                "Durée": "2h 28min surface"
            }
        ]
        
        df_technical = pd.DataFrame(technical_records)
        st.dataframe(df_technical, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📡 Défis Communications")
        
        st.info("""
        **Délai signal (aller simple):**
        - Lune: 1.3 secondes
        - Mars: 3-22 minutes (selon position)
        - Jupiter: 35-52 minutes
        - Saturne: 68-84 minutes
        - Pluton: ~4.5 heures
        - Voyager 1: ~22 heures
        
        **Puissance signal:**
        À grande distance, signal reçu extrêmement faible (10⁻¹⁸ W).
        Nécessite antennes DSN (Deep Space Network) de 70m.
        """)
        
        # Visualisation distances
        st.write("### 🗺️ Distances Système Solaire")
        
        fig = go.Figure()
        
        # Échelle logarithmique pour visualiser
        bodies_distances = [
            ("Lune", 0.384, "gray"),
            ("Vénus", 108, "orange"),
            ("Mars", 228, "red"),
            ("Jupiter", 778, "brown"),
            ("Saturne", 1427, "gold"),
            ("Uranus", 2871, "lightblue"),
            ("Neptune", 4495, "blue"),
            ("Pluton", 5906, "purple"),
            ("Voyager 1", 24000, "darkred")
        ]
        
        for i, (body, dist, color) in enumerate(bodies_distances):
            fig.add_trace(go.Scatter(
                x=[dist],
                y=[i],
                mode='markers+text',
                marker=dict(size=15, color=color),
                text=[body],
                textposition='middle right',
                name=body
            ))
        
        fig.update_layout(
            title="Distances dans le Système Solaire (millions de km)",
            xaxis_title="Distance de Terre (millions km)",
            xaxis_type="log",
            yaxis=dict(showticklabels=False),
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: DOCUMENTATION ====================
elif page == "📚 Documentation":
    st.header("📚 Documentation Complète")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Guide Utilisateur",
        "🧮 Formules",
        "📊 Glossaire",
        "🔗 Ressources",
        "❓ FAQ"
    ])
    
    with tab1:
        st.subheader("📖 Guide d'Utilisation")
        
        st.write("### 🚀 Démarrage Rapide")
        
        st.markdown("""
        **1. Créer un Satellite**
        - Allez dans `➕ Créer Satellite`
        - Configurez les paramètres (masse, puissance, propulsion)
        - Cliquez sur `🚀 Créer le Satellite`
        
        **2. Définir une Orbite**
        - Allez dans `🌍 Orbites`
        - Choisissez type d'orbite (LEO, GEO, etc.)
        - Configurez éléments orbitaux
        - Créez l'orbite
        
        **3. Planifier des Manœuvres**
        - Allez dans `🚀 Manœuvres`
        - Sélectionnez type (Hohmann, changement inclinaison)
        - Calculez le delta-v nécessaire
        
        **4. Simuler la Mission**
        - Allez dans `📊 Simulations`
        - Lancez propagation orbite
        - Visualisez trajectoire
        
        **5. Analyser les Résultats**
        - Consultez `📈 Analyses`
        - Vérifiez performance, budget delta-v
        - Exportez données
        """)
        
        st.markdown("---")
        
        st.write("### 🎯 Fonctionnalités Principales")
        
        features = {
            "🛰️ Gestion Satellites": [
                "Création satellites personnalisés",
                "Configuration masse, puissance, propulsion",
                "Suivi télémétrie en temps réel",
                "Gestion statuts et performances"
            ],
            "🌍 Calculs Orbitaux": [
                "Éléments képlériens complets",
                "Périodes, vitesses, altitudes",
                "Trace au sol (ground track)",
                "Propagation orbite"
            ],
            "🚀 Manœuvres": [
                "Transfert Hohmann",
                "Changement inclinaison/plan",
                "Rendez-vous spatial",
                "Calcul delta-v"
            ],
            "📡 Missions": [
                "Missions LEO, GEO, lunaires, martiennes",
                "Trajectoires interplanétaires",
                "Assistance gravitationnelle",
                "Espace profond"
            ],
            "📊 Analyse": [
                "Performance flotte",
                "Budget delta-v et propergol",
                "Statistiques opérationnelles",
                "Visualisations graphiques"
            ]
        }
        
        for feature, items in features.items():
            with st.expander(f"{feature}"):
                for item in items:
                    st.write(f"✅ {item}")
        
        st.markdown("---")
        
        st.write("### 💡 Conseils et Bonnes Pratiques")
        
        st.info("""
        **Conception Satellite:**
        - Équilibrez masse vs performance
        - Prévoyez marge propergol (15-20%)
        - Dimensionnez puissance pour période éclipse
        - Choisissez Isp adapté à mission
        
        **Choix Orbite:**
        - LEO: Observation Terre, résolution élevée
        - MEO: Navigation (GPS)
        - GEO: Communications, météo
        - Polaire/SSO: Couverture globale
        
        **Optimisation Delta-v:**
        - Utilisez Hohmann pour transferts coplanaires
        - Changements inclinaison à apogée
        - Combinez manœuvres si possible
        - Considérez assists gravitationnels
        
        **Missions Longue Durée:**
        - Propulsion électrique pour station-keeping
        - RTG pour missions lointaines
        - Redondance systèmes critiques
        - Planification communications DSN
        """)
    
    with tab2:
        st.subheader("🧮 Formules et Équations")
        
        st.write("### 🌍 Mécanique Orbitale")
        
        with st.expander("📐 Lois de Kepler"):
            st.write("**1ère Loi (Orbites):**")
            st.write("Les orbites sont des ellipses avec le corps central à un foyer")
            
            st.write("**2ème Loi (Aires):**")
            st.write("Le rayon vecteur balaye des aires égales en temps égaux")
            
            st.write("**3ème Loi (Périodes):**")
            st.latex(r"T^2 = \frac{4\pi^2}{mu} a^3")
            st.write("où T = période, μ = paramètre gravitationnel, a = demi-grand axe")
        
        with st.expander("🎯 Vitesse Orbitale"):
            st.latex(r"v = \sqrt{\frac{\mu}{r}}")
            st.write("Vitesse pour orbite circulaire de rayon r")
            
            st.write("**Équation Vis-Viva (orbite elliptique):**")
            st.latex(r"v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)")
        
        with st.expander("⚡ Énergie Orbitale"):
            st.latex(r"\varepsilon = -\frac{\mu}{2a}")
            st.write("Énergie spécifique (par unité de masse)")
            
            st.write("**Énergie totale:**")
            st.latex(r"E = \frac{1}{2}mv^2 - \frac{GMm}{r}")
        
        with st.expander("🔄 Transfert de Hohmann"):
            st.latex(r"\Delta v_1 = \sqrt{\frac{\mu}{r_1}}\left(\sqrt{\frac{2r_2}{r_1+r_2}} - 1\right)")
            st.latex(r"\Delta v_2 = \sqrt{\frac{\mu}{r_2}}\left(1 - \sqrt{\frac{2r_1}{r_1+r_2}}\right)")
            st.write("Temps de transfert:")
            st.latex(r"t = \pi\sqrt{\frac{(r_1+r_2)^3}{8\mu}}")
        
        with st.expander("📐 Changement Inclinaison"):
            st.latex(r"\Delta v = 2v\sin\left(\frac{\Delta i}{2}\right)")
            st.write("Pour changement d'inclinaison pur")
            
            st.write("**Changement de plan:**")
            st.latex(r"\Delta\phi = \arccos(\cos\Delta i \cdot \cos\Delta\Omega)")
        
        st.markdown("---")
        
        st.write("### 🚀 Propulsion")
        
        with st.expander("🔬 Équation de Tsiolkovsky"):
            st.latex(r"\Delta v = I_{sp} \cdot g_0 \cdot \ln\left(\frac{m_0}{m_f}\right)")
            st.write("""
            - Isp = impulsion spécifique (s)
            - g₀ = 9.80665 m/s²
            - m₀ = masse initiale
            - mf = masse finale
            """)
            
            st.write("**Vitesse d'éjection:**")
            st.latex(r"v_e = I_{sp} \cdot g_0")
        
        with st.expander("⚡ Poussée et Accélération"):
            st.latex(r"F = \dot{m} \cdot v_e")
            st.write("F = poussée, ṁ = débit massique")
            
            st.latex(r"a = \frac{F}{m}")
            st.write("Accélération instantanée")
        
        st.markdown("---")
        
        st.write("### 🌌 Espace Profond")
        
        with st.expander("🎯 C3 (Énergie Caractéristique)"):
            st.latex(r"C_3 = v_\infty^2")
            st.write("v∞ = vitesse hyperbolique à l'infini")
            
            st.write("**Relation avec vitesse:**")
            st.latex(r"v = \sqrt{v_{esc}^2 + C_3}")
        
        with st.expander("🌍 Sphère d'Influence (SOI)"):
            st.latex(r"r_{SOI} = a\left(\frac{m}{M}\right)^{0.4}")
            st.write("""
            - a = demi-grand axe orbite planète
            - m = masse planète
            - M = masse corps central (Soleil)
            """)
        
        with st.expander("💫 Points de Lagrange"):
            st.write("**Distance L1 (approximation Hill):**")
            st.latex(r"r_{L1} \approx R\left(\frac{M_2}{3M_1}\right)^{1/3}")
            st.write("R = distance entre corps, M₁ = masse primaire, M₂ = masse secondaire")
    
    with tab3:
        st.subheader("📊 Glossaire")
        
        glossary = {
            "A": {
                "Apoapside": "Point le plus éloigné d'une orbite par rapport au corps central",
                "Apogée": "Apoapside pour orbite terrestre",
                "Aphélie": "Apoapside pour orbite solaire",
                "Anomalie Vraie": "Angle entre périapside et position actuelle sur orbite",
                "Argument du Périapside": "Angle entre nœud ascendant et périapside"
            },
            "C": {
                "C3": "Énergie caractéristique d'une trajectoire hyperbolique (v∞²)",
                "Circularisation": "Manœuvre pour rendre une orbite circulaire (e=0)"
            },
            "D": {
                "Delta-v (Δv)": "Changement de vitesse nécessaire pour une manœuvre",
                "DSN": "Deep Space Network - Réseau antennes NASA pour espace profond"
            },
            "E": {
                "Excentricité": "Mesure de l'aplatissement d'une ellipse (0=cercle, <1=ellipse)",
                "EDL": "Entry, Descent, Landing - Phase critique missions planétaires",
                "Éléments Képlériens": "6 paramètres décrivant orbite (a, e, i, Ω, ω, ν)"
            },
            "G": {
                "GEO": "Orbite Géostationnaire (35,786 km, période 24h)",
                "Gravity Assist": "Assistance gravitationnelle, effet de fronde",
                "Ground Track": "Trace au sol, projection orbite sur surface planète"
            },
            "H": {
                "Hohmann": "Transfert bi-impulsionnel le plus économe en énergie",
                "Halo Orbit": "Orbite 3D autour point de Lagrange"
            },
            "I": {
                "Inclinaison": "Angle entre plan orbital et plan équatorial",
                "Isp": "Impulsion Spécifique, mesure efficacité propulsion (secondes)",
                "ISRU": "In-Situ Resource Utilization, utilisation ressources locales"
            },
            "L": {
                "LEO": "Low Earth Orbit, orbite basse (200-2000 km)",
                "LOI": "Lunar Orbit Insertion, insertion en orbite lunaire",
                "Lagrange": "Points d'équilibre gravitationnel dans système à 2 corps"
            },
            "M": {
                "MEO": "Medium Earth Orbit, orbite moyenne (2,000-35,786 km)",
                "μ (Mu)": "Paramètre gravitationnel standard GM (m³/s²)"
            },
            "N": {
                "Nœud": "Intersection orbite avec plan référence",
                "NRHO": "Near-Rectilinear Halo Orbit (Gateway lunaire)"
            },
            "P": {
                "Périapside": "Point le plus proche d'une orbite",
                "Périgée": "Périapside pour orbite terrestre",
                "Périhélie": "Périapside pour orbite solaire",
                "Phasage": "Ajustement timing pour rendez-vous"
            },
            "R": {
                "RAAN": "Right Ascension of Ascending Node, longitude nœud ascendant",
                "RTG": "Radioisotope Thermoelectric Generator, générateur nucléaire"
            },
            "S": {
                "SOI": "Sphere of Influence, sphère d'influence gravitationnelle",
                "SSO": "Sun-Synchronous Orbit, orbite héliosynchrone",
                "Station-keeping": "Manœuvres maintien orbite nominale"
            },
            "T": {
                "TLI": "Trans-Lunar Injection, injection trans-lunaire",
                "Tsiolkovsky": "Équation fondamentale de l'astronautique (delta-v)"
            },
            "V": {
                "Vis-Viva": "Équation vitesse en fonction position sur orbite elliptique",
                "v∞": "Vitesse hyperbolique à l'infini"
            }
        }
        
        for letter, terms in glossary.items():
            with st.expander(f"📖 {letter}"):
                for term, definition in terms.items():
                    st.write(f"**{term}:** {definition}")
    
    with tab4:
        st.subheader("🔗 Ressources Externes")
        
        st.write("### 📚 Documentation Officielle")
        
        st.markdown("""
        **Agences Spatiales:**
        - [NASA](https://www.nasa.gov/) - Agence spatiale américaine
        - [ESA](https://www.esa.int/) - Agence spatiale européenne
        - [JAXA](https://global.jaxa.jp/) - Agence spatiale japonaise
        - [Roscosmos](https://www.roscosmos.ru/) - Agence spatiale russe
        - [CNES](https://cnes.fr/) - Centre National d'Études Spatiales (France)
        
        **Bases de Données:**
        - [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/) - Éphémérides haute précision
        - [Celestrak](https://celestrak.org/) - Éléments orbitaux satellites (TLE)
        - [Space-Track](https://www.space-track.org/) - Catalogue objets spatiaux
        - [N2YO](https://www.n2yo.com/) - Tracking satellites en temps réel
        
        **Outils en Ligne:**
        - [GMAT](https://software.nasa.gov/software/GSC-17177-1) - Logiciel NASA trajectoires
        - [Orekit](https://www.orekit.org/) - Bibliothèque mécanique spatiale (Java/Python)
        - [Poliastro](https://docs.poliastro.space/) - Python astrodynamique
        - [STK](https://www.ansys.com/products/missions/ansys-stk) - Systems Tool Kit (commercial)
        """)
        
        st.markdown("---")
        
        st.write("### 📖 Livres Recommandés")
        
        books = [
            {
                "Titre": "Fundamentals of Astrodynamics",
                "Auteurs": "Bate, Mueller, White",
                "Niveau": "Intermédiaire",
                "Description": "Référence classique, très pédagogique"
            },
            {
                "Titre": "Orbital Mechanics for Engineering Students",
                "Auteurs": "Howard Curtis",
                "Niveau": "Débutant-Intermédiaire",
                "Description": "Excellent pour débuter, nombreux exemples"
            },
            {
                "Titre": "Space Mission Analysis and Design (SMAD)",
                "Auteurs": "Wertz, Larson",
                "Niveau": "Tous niveaux",
                "Description": "Bible conception missions spatiales"
            },
            {
                "Titre": "Spacecraft Dynamics and Control",
                "Auteurs": "De Ruiter, Damaren, Forbes",
                "Niveau": "Avancé",
                "Description": "Dynamique et contrôle d'attitude"
            }
        ]
        
        df_books = pd.DataFrame(books)
        st.dataframe(df_books, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🎓 Cours en Ligne")
        
        st.markdown("""
        **MOOCs:**
        - Coursera: "Introduction to Aerospace Engineering"
        - edX: "Space Mission Design and Operations"
        - MIT OpenCourseWare: "Astrodynamics"
        
        **Chaînes YouTube:**
        - Scott Manley - Vulgarisation spatiale excellente
        - Everyday Astronaut - Missions et lanceurs
        - NASA - Contenus officiels
        """)
    
    with tab5:
        st.subheader("❓ Questions Fréquentes (FAQ)")
        
        faq = {
            "Quelle est la différence entre LEO, MEO et GEO?": """
            - **LEO (Low Earth Orbit):** 200-2000 km. Utilisé pour observation Terre, ISS. 
              Période ~90 minutes. Faible latence mais nécessite constellation.
            
            - **MEO (Medium Earth Orbit):** 2,000-35,786 km. GPS, Galileo. 
              Bon compromis couverture/latence. Période 2-12 heures.
            
            - **GEO (Geostationary):** 35,786 km exactement. Période = 24h (synchrone rotation Terre).
              Position fixe dans ciel. Communications, météo. Latence ~500ms aller-retour.
            """,
            
            "Pourquoi le changement d'inclinaison coûte-t-il si cher en delta-v?": """
            Le changement d'inclinaison nécessite de modifier le vecteur vitesse perpendiculairement 
            au mouvement. À 7-8 km/s en LEO, même un petit changement d'angle requiert un delta-v énorme:
            
            - 1° → ~120 m/s
            - 10° → ~1,200 m/s  
            - 28.5° → ~3,400 m/s (presque une nouvelle mise en orbite!)
            
            C'est pourquoi on lance selon l'inclinaison désirée depuis le début.
            """,
            
            "Qu'est-ce que l'Isp et pourquoi est-ce important?": """
            **Isp (Impulsion Spécifique)** mesure l'efficacité d'un moteur-fusée en secondes.
            
            Plus l'Isp est élevé, moins on consomme de propergol pour un delta-v donné:
            
            - Chimique: 300-450 s (poussée élevée, Isp moyen)
            - Électrique: 1500-3000 s (poussée faible, Isp excellent)
            - Ionique: 3000-5000 s (très faible poussée, Isp exceptionnel)
            
            Règle: Chimique pour manœuvres rapides, électrique pour longue durée.
            """,
            
            "Comment fonctionne l'assistance gravitationnelle?": """
            L'assistance gravitationnelle (gravity assist) utilise la gravité d'une planète pour:
            
            1. Modifier direction sans propulsion
            2. Gagner (ou perdre) vitesse par rapport au Soleil
            
            **Principe:** Dans référentiel planète, magnitude vitesse conservée mais direction change.
            Dans référentiel solaire, cela se traduit par gain/perte d'énergie.
            
            **Exemple:** Cassini a économisé ~10 km/s de delta-v grâce à 6 assists gravitationnels!
            """,
            
            "Quelle est la vitesse minimale pour échapper à la Terre?": """
            **Vitesse d'échappement Terre:** 11.186 km/s depuis la surface
            
            À 200 km d'altitude (LEO): ~10.9 km/s
            
            En réalité, fusées atteignent ~7.8 km/s pour mise en orbite LEO, puis boost ~3.1 km/s 
            pour échappement (TLI vers Lune par exemple).
            
            Pour échapper système solaire: ~42 km/s depuis surface Terre (16.7 km/s depuis orbite terrestre).
            """,
            
            "Pourquoi lance-t-on vers l'est?": """
            La Terre tourne vers l'est à ~465 m/s à l'équateur. 
            
            En lançant vers l'est, on bénéficie de ce "bonus" de vitesse gratuit:
            - À l'équateur: ~465 m/s économisés
            - À 45°N: ~329 m/s
            - Aux pôles: 0 m/s
            
            C'est pourquoi sites équatoriaux (Kourou, Cap Canaveral) sont prisés!
            
            Exception: Orbites polaires lancées vers le sud depuis Vandenberg.
            """,
            
            "Combien de temps faut-il pour aller sur Mars?": """
            **Transfert Hohmann optimal:** 6-9 mois (en moyenne 7 mois)
            
            Facteurs:
            - Position relative Terre-Mars (fenêtre tous les 26 mois)
            - Delta-v disponible (plus de delta-v = trajet plus rapide)
            - Type trajectoire (directe vs avec assist)
            
            Records:
            - Mariner 6/7: 156 jours (1969) - trajectoire rapide
            - Mariner 9: 168 jours (1971)
            - Missions récentes: 200-240 jours généralement
            
            Futur: Propulsion nucléaire/électrique pourrait réduire à 3-4 mois.
            """,
            
            "Qu'est-ce qu'un transfert de Hohmann?": """
            **Transfert de Hohmann:** Manœuvre la plus économe en énergie entre 2 orbites circulaires coplanaires.
            
            Principe:
            1. Impulsion au périgée de l'orbite initiale → ellipse de transfert
            2. Coast sur orbite transfert (demi-orbite)
            3. Impulsion à l'apogée → circularisation orbite finale
            
            Avantage: Delta-v minimal
            Inconvénient: Durée la plus longue
            
            Alternative: Bi-elliptique (plus économe si ratio r₂/r₁ > 11.94)
            """,
            
            "Peut-on respirer sur Mars sans combinaison?": """
            ❌ **NON - Absolument mortel!**
            
            Atmosphère Mars:
            - Pression: 0.6% de la Terre (~6 mbar) → Ébullition sang
            - Composition: 95% CO₂, presque pas d'O₂
            - Température: -60°C moyenne (-120°C à +20°C)
            - Radiation: Pas de magnétosphère protectrice
            
            Combinaison spatiale pressurisée obligatoire en permanence à l'extérieur.
            
            Terraformation (hypothétique) prendrait des siècles minimum.
            """
        }
        
        for question, answer in faq.items():
            with st.expander(f"❓ {question}"):
                st.write(answer)
        
        st.markdown("---")
        
        st.write("### 💬 Besoin d'aide supplémentaire?")
        
        st.info("""
        **Ressources communautaires:**
        - r/spaceflight (Reddit)
        - r/KerbalSpaceProgram (excellente communauté apprentissage orbital)
        - Space Stack Exchange
        - Forum NSF (NASASpaceFlight.com)
        """)
# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal des Événements (Dernières 10 entrées)"):
    if st.session_state.space_system['log']:
        for event in st.session_state.space_system['log'][-10:][::-1]:
            timestamp = event['timestamp'][:19]
            st.text(f"{timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")
    
    if st.button("🗑️ Effacer le Journal"):
        st.session_state.space_system['log'] = []
        st.rerun()

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🚀 Plateforme de Mécanique Spatiale</h3>
        <p>Système Intégré pour Missions et Orbites</p>
        <p><small>Version 1.0.0 | Mécanique Spatiale Complète</small></p>
        <p><small>🛰️ Satellites | 🌍 Orbites | 🚀 Manœuvres | 📊 Simulations</small></p>
        <p><small>Powered by Space Engineering © 2024</small></p>
    </div>
""", unsafe_allow_html=True)