"""
Interface Streamlit pour la Plateforme d'Accélérateur de Particules
Système complet pour créer, simuler, tester et analyser des accélérateurs
streamlit run accelerateur_particules_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import numpy as np
from scipy import constants

# ==================== CONFIGURATION PAGE ====================

st.set_page_config(
    page_title="⚛️ Plateforme Accélérateur de Particules",
    page_icon="⚛️",
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
    .accelerator-card {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .energy-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.3rem;
    }
    .low-energy {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
        color: white;
    }
    .medium-energy {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .high-energy {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .ultra-high-energy {
        background: linear-gradient(90deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .particle-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES PHYSIQUES ====================

PHYSICS_CONSTANTS = {
    'c': constants.c,
    'e': constants.e,
    'm_e': constants.m_e,
    'm_p': constants.m_p,
    'hbar': constants.hbar,
}

PARTICLE_DATA = {
    'electron': {'mass': 9.10938e-31, 'charge': -1.602176e-19, 'symbol': 'e⁻'},
    'positron': {'mass': 9.10938e-31, 'charge': 1.602176e-19, 'symbol': 'e⁺'},
    'proton': {'mass': 1.67262e-27, 'charge': 1.602176e-19, 'symbol': 'p'},
    'antiproton': {'mass': 1.67262e-27, 'charge': -1.602176e-19, 'symbol': 'p̄'},
    'muon': {'mass': 1.88353e-28, 'charge': -1.602176e-19, 'symbol': 'μ⁻'},
}

# ==================== INITIALISATION SESSION STATE ====================

if 'accelerator_system' not in st.session_state:
    st.session_state.accelerator_system = {
        'accelerators': {},
        'experiments': {},
        'simulations': [],
        'collisions': [],
        'detections': [],
        'analyses': [],
        'projects': {},
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str):
    """Enregistre un événement"""
    st.session_state.accelerator_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })

def get_energy_badge(energy_ev: float) -> str:
    """Retourne un badge HTML selon l'énergie"""
    if energy_ev < 1e6:  # < 1 MeV
        return '<span class="energy-badge low-energy">⚡ BASSE ÉNERGIE</span>'
    elif energy_ev < 1e9:  # < 1 GeV
        return '<span class="energy-badge medium-energy">⚡ MOYENNE ÉNERGIE</span>'
    elif energy_ev < 1e12:  # < 1 TeV
        return '<span class="energy-badge high-energy">⚡ HAUTE ÉNERGIE</span>'
    else:
        return '<span class="energy-badge ultra-high-energy">⚡ ULTRA-HAUTE ÉNERGIE</span>'

def format_energy(energy_ev: float) -> str:
    """Formate l'énergie avec l'unité appropriée"""
    if energy_ev < 1e3:
        return f"{energy_ev:.2f} eV"
    elif energy_ev < 1e6:
        return f"{energy_ev/1e3:.2f} keV"
    elif energy_ev < 1e9:
        return f"{energy_ev/1e6:.2f} MeV"
    elif energy_ev < 1e12:
        return f"{energy_ev/1e9:.2f} GeV"
    else:
        return f"{energy_ev/1e12:.2f} TeV"
    
    
import numpy as np
from scipy import constants

def calculate_relativistic_params(particle_type: str, energy_ev: float):
    """Calcule les paramètres relativistes avec protection contre les débordements numériques"""
    particle = PARTICLE_DATA[particle_type]
    
    # Conversion eV → Joules
    energy_joules = energy_ev * constants.e
    rest_energy = particle['mass'] * constants.c**2
    total_energy = energy_joules + rest_energy

    # --- Calcul stable de gamma ---
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        gamma = np.where(rest_energy > 0, total_energy / rest_energy, 1.0)
        gamma = np.nan_to_num(gamma, nan=1.0, posinf=1e12, neginf=1.0)

    # On borne gamma pour éviter des valeurs infinies (physiquement illogiques)
    gamma = np.clip(gamma, 1, 1e12)

    # --- Calcul stable de beta = sqrt(1 - 1/gamma²) ---
    try:
        value = 1 - 1 / gamma**2
    except OverflowError:
        value = 1.0  # approche limite relativiste

    value = np.clip(value, 0, 1)
    beta = np.sqrt(value)
    velocity = beta * constants.c

    # --- Quantité de mouvement relativiste ---
    momentum = gamma * particle['mass'] * velocity

    # --- Résultats ---
    return {
        'gamma': float(gamma),
        'beta': float(beta),
        'velocity': float(velocity),
        'momentum': float(momentum),
        'total_energy': float(total_energy),
        'rest_energy': float(rest_energy)
    }


def create_accelerator_mock(name, acc_type, config):
    """Crée un accélérateur simulé"""
    acc_id = f"acc_{len(st.session_state.accelerator_system['accelerators']) + 1}"
    
    accelerator = {
        'id': acc_id,
        'name': name,
        'type': acc_type,
        'created_at': datetime.now().isoformat(),
        'status': 'offline',
        'geometry': {
            'length': config.get('length', 1000),
            'radius': config.get('radius', 0),
            'circumference': config.get('circumference', 0)
        },
        'energy': {
            'min': config.get('energy_min', 1e6),
            'max': config.get('energy_max', 1e9),
            'final': config.get('energy_final', 1e9)
        },
        'components': {
            'rf_cavities': config.get('n_cavities', 10),
            'magnets': config.get('n_magnets', 50),
            'detectors': config.get('n_detectors', 5)
        },
        'beams': [],
        'performance': {
            'luminosity': config.get('luminosity', 1e34),
            'collision_rate': 0,
            'beam_current': 0,
            'efficiency': 0.85
        },
        'vacuum': {
            'pressure': 1e-10,  # Pascal
            'quality': 0.99
        },
        'costs': {
            'construction': config.get('construction_cost', 1e9),
            'operational': config.get('operational_cost', 1e8),
            'energy_consumption': config.get('energy_consumption', 100000)
        },
        'operational_hours': 0,
        'total_collisions': 0,
        'experiments_run': 0
    }
    
    # Calculer la circonférence pour les accélérateurs circulaires
    if acc_type in ['circulaire', 'synchrotron', 'collisionneur']:
        if accelerator['geometry']['radius'] > 0:
            accelerator['geometry']['circumference'] = 2 * np.pi * accelerator['geometry']['radius']
    
    st.session_state.accelerator_system['accelerators'][acc_id] = accelerator
    log_event(f"Accélérateur créé: {name} ({acc_type})")
    return acc_id

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">⚛️ Plateforme Accélérateur de Particules</h1>', unsafe_allow_html=True)
st.markdown("### Système Complet de Création, Simulation, Test et Analyse d'Accélérateurs de Particules")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Particle+Physics+Lab", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Tableau de Bord",
            "⚛️ Mes Accélérateurs",
            "➕ Créer Accélérateur",
            "🔬 Simulations",
            "💥 Collisions",
            "📡 Détecteurs",
            "🧪 Expériences",
            "📊 Analyses & Résultats",
            "📐 Physique des Particules",
            "🎯 Conception Optique",
            "🏭 Fabrication",
            "🔧 Tests & Calibration",
            "💰 Coûts & Budget",
            "📚 Bibliothèque",
            "🌌 Découvertes"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_acc = len(st.session_state.accelerator_system['accelerators'])
    active_acc = sum(1 for a in st.session_state.accelerator_system['accelerators'].values() if a['status'] == 'online')
    total_exp = len(st.session_state.accelerator_system['experiments'])
    total_sim = len(st.session_state.accelerator_system['simulations'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚛️ Accélérateurs", total_acc)
        st.metric("🧪 Expériences", total_exp)
    with col2:
        st.metric("✅ Actifs", active_acc)
        st.metric("🔬 Simulations", total_sim)

# ==================== PAGE: TABLEAU DE BORD ====================

if page == "🏠 Tableau de Bord":
    st.header("📊 Tableau de Bord Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="accelerator-card"><h2>⚛️</h2><h3>{total_acc}</h3><p>Accélérateurs</p></div>', unsafe_allow_html=True)
    
    with col2:
        total_collisions = sum(a['total_collisions'] for a in st.session_state.accelerator_system['accelerators'].values())
        st.markdown(f'<div class="accelerator-card"><h2>💥</h2><h3>{total_collisions:,}</h3><p>Collisions</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="accelerator-card"><h2>🧪</h2><h3>{total_exp}</h3><p>Expériences</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="accelerator-card"><h2>🔬</h2><h3>{total_sim}</h3><p>Simulations</p></div>', unsafe_allow_html=True)
    
    with col5:
        discoveries = len([d for d in st.session_state.accelerator_system.get('discoveries', [])])
        st.markdown(f'<div class="accelerator-card"><h2>🌟</h2><h3>{discoveries}</h3><p>Découvertes</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.accelerator_system['accelerators']:
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Types d'Accélérateurs")
            
            type_counts = {}
            for acc in st.session_state.accelerator_system['accelerators'].values():
                acc_type = acc['type'].replace('_', ' ').title()
                type_counts[acc_type] = type_counts.get(acc_type, 0) + 1
            
            fig = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                        color_discrete_sequence=px.colors.sequential.Purples_r)
            fig.update_layout(title="Répartition par Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Énergies Maximales")
            
            names = [a['name'][:15] for a in st.session_state.accelerator_system['accelerators'].values()]
            energies = [a['energy']['max']/1e9 for a in st.session_state.accelerator_system['accelerators'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=energies, marker_color='rgb(102, 126, 234)')
            ])
            fig.update_layout(title="Énergie Maximale (GeV)", yaxis_title="Énergie (GeV)", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Vue d'ensemble des accélérateurs actifs
        st.subheader("⚛️ Accélérateurs Actifs")
        
        active_accelerators = {k: v for k, v in st.session_state.accelerator_system['accelerators'].items() 
                              if v['status'] == 'online'}
        
        if active_accelerators:
            for acc_id, acc in active_accelerators.items():
                with st.expander(f"⚛️ {acc['name']} - {format_energy(acc['energy']['max'])}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Type", acc['type'].replace('_', ' ').title())
                        st.metric("Longueur", f"{acc['geometry']['length']:.0f} m")
                    
                    with col2:
                        st.metric("Énergie Max", format_energy(acc['energy']['max']))
                        st.metric("Luminosité", f"{acc['performance']['luminosity']:.2e} cm⁻²s⁻¹")
                    
                    with col3:
                        st.metric("Cavités RF", acc['components']['rf_cavities'])
                        st.metric("Aimants", acc['components']['magnets'])
                    
                    with col4:
                        st.metric("Collisions Totales", f"{acc['total_collisions']:,}")
                        st.metric("Expériences", acc['experiments_run'])
        else:
            st.info("Aucun accélérateur actif")
    else:
        st.info("💡 Aucun accélérateur créé. Créez votre premier accélérateur de particules!")

# ==================== PAGE: MES ACCÉLÉRATEURS ====================

elif page == "⚛️ Mes Accélérateurs":
    st.header("⚛️ Gestion des Accélérateurs")
    
    if not st.session_state.accelerator_system['accelerators']:
        st.info("💡 Aucun accélérateur créé.")
    else:
        for acc_id, acc in st.session_state.accelerator_system['accelerators'].items():
            st.markdown(f'<div class="accelerator-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### ⚛️ {acc['name']}")
                st.markdown(get_energy_badge(acc['energy']['max']), unsafe_allow_html=True)
                st.caption(f"Type: {acc['type'].replace('_', ' ').title()}")
            
            with col2:
                st.metric("Énergie Max", format_energy(acc['energy']['max']))
                st.metric("Longueur", f"{acc['geometry']['length']:.0f} m")
            
            with col3:
                st.metric("Luminosité", f"{acc['performance']['luminosity']:.2e}")
                st.metric("Efficacité", f"{acc['performance']['efficiency']:.0%}")
            
            with col4:
                status_icon = "🟢" if acc['status'] == 'online' else "🔴"
                st.write(f"**Statut:** {status_icon} {acc['status'].upper()}")
                st.write(f"**Heures Op.:** {acc['operational_hours']:.1f}h")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏗️ Géométrie", "⚡ Énergie", "🔧 Composants", "📊 Performance", "💰 Coûts"])
                
                with tab1:
                    st.subheader("🏗️ Configuration Géométrique")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Longueur Totale", f"{acc['geometry']['length']:.0f} m")
                    with col2:
                        if acc['geometry']['radius'] > 0:
                            st.metric("Rayon", f"{acc['geometry']['radius']:.0f} m")
                    with col3:
                        if acc['geometry']['circumference'] > 0:
                            st.metric("Circonférence", f"{acc['geometry']['circumference']:.0f} m")
                
                with tab2:
                    st.subheader("⚡ Caractéristiques Énergétiques")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Énergie Minimale", format_energy(acc['energy']['min']))
                    with col2:
                        st.metric("Énergie Maximale", format_energy(acc['energy']['max']))
                    with col3:
                        st.metric("Énergie Finale", format_energy(acc['energy']['final']))
                
                with tab3:
                    st.subheader("🔧 Composants Installés")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Cavités RF", acc['components']['rf_cavities'])
                    with col2:
                        st.metric("Aimants", acc['components']['magnets'])
                    with col3:
                        st.metric("Détecteurs", acc['components']['detectors'])
                
                with tab4:
                    st.subheader("📊 Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Luminosité", f"{acc['performance']['luminosity']:.2e} cm⁻²s⁻¹")
                    with col2:
                        st.metric("Taux Collision", f"{acc['performance']['collision_rate']:.2e} Hz")
                    with col3:
                        st.metric("Efficacité", f"{acc['performance']['efficiency']:.0%}")
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**🌡️ Système de Vide:**")
                        st.metric("Pression", f"{acc['vacuum']['pressure']:.2e} Pa")
                        st.metric("Qualité", f"{acc['vacuum']['quality']:.0%}")
                
                with tab5:
                    st.subheader("💰 Coûts et Économie")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Construction", f"${acc['costs']['construction']/1e9:.2f}B")
                    with col2:
                        st.metric("Opérationnel/an", f"${acc['costs']['operational']/1e6:.0f}M")
                    with col3:
                        st.metric("Énergie/an", f"{acc['costs']['energy_consumption']:,} MWh")
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button(f"▶️ {'Éteindre' if acc['status'] == 'online' else 'Activer'}", key=f"toggle_{acc_id}"):
                        acc['status'] = 'offline' if acc['status'] == 'online' else 'online'
                        log_event(f"{acc['name']} {'éteint' if acc['status'] == 'offline' else 'activé'}")
                        st.rerun()
                
                with col2:
                    if st.button(f"🔬 Simuler", key=f"sim_{acc_id}"):
                        st.info("Allez dans Simulations")
                
                with col3:
                    if st.button(f"💥 Collision", key=f"col_{acc_id}"):
                        st.info("Allez dans Collisions")
                
                with col4:
                    if st.button(f"📊 Analyser", key=f"analyze_{acc_id}"):
                        st.info("Allez dans Analyses")
                
                with col5:
                    if st.button(f"🗑️ Supprimer", key=f"del_{acc_id}"):
                        del st.session_state.accelerator_system['accelerators'][acc_id]
                        log_event(f"{acc['name']} supprimé")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CRÉER ACCÉLÉRATEUR ====================

elif page == "➕ Créer Accélérateur":
    st.header("➕ Créer un Nouvel Accélérateur")
    
    with st.form("create_accelerator_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            acc_name = st.text_input("📝 Nom de l'Accélérateur", placeholder="Ex: LHC-Mini")
            
            acc_type = st.selectbox(
                "🔬 Type d'Accélérateur",
                [
                    "lineaire",
                    "circulaire",
                    "synchrotron",
                    "cyclotron",
                    "collisionneur",
                    "anneau_stockage"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            primary_particle = st.selectbox(
                "⚛️ Particule Principale",
                list(PARTICLE_DATA.keys()),
                format_func=lambda x: f"{PARTICLE_DATA[x]['symbol']} {x.title()}"
            )
            
            application = st.selectbox(
                "🎯 Application Principale",
                ["Physique Fondamentale", "Recherche Médicale", "Science des Matériaux", 
                 "Production Isotopes", "Thérapie par Faisceau", "Recherche & Développement"]
            )
        
        st.markdown("---")
        st.subheader("🏗️ Configuration Géométrique")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if acc_type == "lineaire":
                length = st.number_input("Longueur (m)", 1.0, 100000.0, 1000.0, 100.0)
                radius = 0
            else:
                radius = st.number_input("Rayon (m)", 1.0, 10000.0, 100.0, 10.0)
                length = 2 * np.pi * radius
        
        with col2:
            if acc_type != "lineaire":
                st.metric("Circonférence Calculée", f"{length:.0f} m")
        
        with col3:
            tunnel_diameter = st.number_input("Diamètre Tunnel (m)", 1.0, 50.0, 3.0, 0.5)
        
        st.markdown("---")
        st.subheader("⚡ Configuration Énergétique")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            energy_unit = st.selectbox("Unité d'Énergie", ["eV", "keV", "MeV", "GeV", "TeV"])
            multiplier = {'eV': 1, 'keV': 1e3, 'MeV': 1e6, 'GeV': 1e9, 'TeV': 1e12}[energy_unit]
        
        with col2:
            energy_min_val = st.number_input(f"Énergie Minimale ({energy_unit})", 0.001, 1000000.0, 1.0)
            energy_min = energy_min_val * multiplier
        
        with col3:
            energy_max_val = st.number_input(f"Énergie Maximale ({energy_unit})", 0.001, 1000000.0, 10.0)
            energy_max = energy_max_val * multiplier
        
        # Affichage paramètres relativistes
        st.markdown("---")
        st.write("**📊 Paramètres Relativistes à Énergie Max:**")
        
        params = calculate_relativistic_params(primary_particle, energy_max)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Facteur γ (Lorentz)", f"{params['gamma']:.2f}")
        with col2:
            st.metric("β (v/c)", f"{params['beta']:.6f}")
        with col3:
            st.metric("Vitesse", f"{params['velocity']/constants.c:.6f}c")
        with col4:
            st.metric("Momentum", f"{params['momentum']:.2e} kg·m/s")
        
        st.markdown("---")
        st.subheader("🔧 Composants Principaux")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_cavities = st.number_input("Nombre de Cavités RF", 1, 10000, 100, 10)
            rf_frequency = st.number_input("Fréquence RF (MHz)", 1.0, 10000.0, 500.0, 10.0)
        
        with col2:
            n_magnets = st.number_input("Nombre d'Aimants", 1, 100000, 500, 50)
            magnet_field = st.number_input("Champ Magnétique (T)", 0.1, 20.0, 1.5, 0.1)
        
        with col3:
            n_detectors = st.number_input("Nombre de Détecteurs", 1, 100, 5, 1)
            vacuum_pressure = st.number_input("Pression Vide (Pa)", 1e-12, 1e-6, 1e-10, format="%.2e")
        
        st.markdown("---")
        st.subheader("📊 Performance Cible")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_luminosity = st.number_input("Luminosité Cible (cm⁻²s⁻¹)", 1e30, 1e38, 1e34, format="%.2e")
            target_efficiency = st.slider("Efficacité Cible", 0.5, 0.99, 0.85, 0.01)
        
        with col2:
            beam_current = st.number_input("Courant Faisceau (mA)", 0.001, 1000.0, 1.0)
            rep_rate = st.number_input("Taux de Répétition (Hz)", 1, 1000000, 100)
        
        st.markdown("---")
        st.subheader("💰 Estimation Budgétaire")
        
        # Calcul automatique des coûts
        if acc_type == "lineaire":
            base_cost = length * 1e6  # $1M par mètre
        elif acc_type == "collisionneur":
            base_cost = length * 50e6  # $50M par mètre pour collisionneur
        else:
            base_cost = length * 5e6  # $5M par mètre
        
        rf_cost = n_cavities * 500000
        magnet_cost = n_magnets * 100000
        detector_cost = n_detectors * 10000000
        infrastructure_cost = base_cost * 0.3
        
        total_construction = base_cost + rf_cost + magnet_cost + detector_cost + infrastructure_cost
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Construction Estimée", f"${total_construction/1e9:.2f}B")
        with col2:
            operational_cost = total_construction * 0.1
            st.metric("Opérationnel/an", f"${operational_cost/1e6:.0f}M")
        with col3:
            energy_consumption = n_cavities * 100 + n_magnets * 10  # MWh/an
            st.metric("Énergie/an", f"{energy_consumption:,} MWh")
        
        submitted = st.form_submit_button("🚀 Créer l'Accélérateur", use_container_width=True, type="primary")
        
        if submitted:
            if not acc_name:
                st.error("⚠️ Veuillez donner un nom à l'accélérateur")
            else:
                with st.spinner("🔄 Création de l'accélérateur en cours..."):
                    config = {
                        'length': length,
                        'radius': radius,
                        'circumference': length if acc_type != "lineaire" else 0,
                        'energy_min': energy_min,
                        'energy_max': energy_max,
                        'energy_final': energy_max,
                        'n_cavities': n_cavities,
                        'n_magnets': n_magnets,
                        'n_detectors': n_detectors,
                        'luminosity': target_luminosity,
                        'construction_cost': total_construction,
                        'operational_cost': operational_cost,
                        'energy_consumption': energy_consumption
                    }
                    
                    acc_id = create_accelerator_mock(acc_name, acc_type, config)
                    
                    st.success(f"✅ Accélérateur '{acc_name}' créé avec succès!")
                    st.balloons()
                    
                    acc = st.session_state.accelerator_system['accelerators'][acc_id]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Longueur", f"{acc['geometry']['length']:.0f} m")
                    with col2:
                        st.metric("Énergie Max", format_energy(acc['energy']['max']))
                    with col3:
                        st.metric("Composants", n_cavities + n_magnets + n_detectors)
                    with col4:
                        st.metric("Coût", f"${total_construction/1e9:.2f}B")
                    
                    st.code(f"ID: {acc_id}", language="text")

# ==================== PAGE: SIMULATIONS ====================

elif page == "🔬 Simulations":
    st.header("🔬 Simulations de Particules")
    
    if not st.session_state.accelerator_system['accelerators']:
        st.warning("⚠️ Aucun accélérateur disponible")
    else:
        tab1, tab2, tab3 = st.tabs(["🚀 Nouvelle Simulation", "📊 Résultats", "📜 Historique"])
        
        with tab1:
            st.subheader("🚀 Configurer une Simulation")
            
            acc_options = {a['id']: a['name'] for a in st.session_state.accelerator_system['accelerators'].values()}
            selected_acc = st.selectbox(
                "Sélectionner un accélérateur",
                options=list(acc_options.keys()),
                format_func=lambda x: acc_options[x]
            )
            
            acc = st.session_state.accelerator_system['accelerators'][selected_acc]
            
            st.write(f"### ⚛️ {acc['name']}")
            st.write(f"**Type:** {acc['type'].replace('_', ' ').title()}")
            st.write(f"**Longueur:** {acc['geometry']['length']:.0f} m")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                particle_type = st.selectbox(
                    "Type de Particule",
                    list(PARTICLE_DATA.keys()),
                    format_func=lambda x: f"{PARTICLE_DATA[x]['symbol']} {x.title()}"
                )
                
                n_particles = st.number_input("Nombre de Particules", 1, 1000000, 1000)
            
            with col2:
                initial_energy_unit = st.selectbox("Unité Énergie Initiale", ["eV", "keV", "MeV", "GeV"], index=2)
                multiplier = {'eV': 1, 'keV': 1e3, 'MeV': 1e6, 'GeV': 1e9}[initial_energy_unit]
                
                initial_energy_val = st.number_input(f"Énergie Initiale ({initial_energy_unit})", 0.001, 100000.0, 1.0)
                initial_energy = initial_energy_val * multiplier
            
            st.markdown("---")
            
            st.subheader("⚙️ Paramètres de Simulation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_steps = st.slider("Nombre d'Étapes", 10, 10000, 1000)
                time_step = st.number_input("Pas de Temps (ns)", 0.001, 1000.0, 1.0)
            
            with col2:
                beam_emittance = st.number_input("Émittance (m·rad)", 1e-12, 1e-6, 1e-9, format="%.2e")
                beam_spread = st.slider("Dispersion Énergie (ΔE/E)", 0.0001, 0.01, 0.001, format="%.4f")
            
            with col3:
                include_radiation = st.checkbox("Inclure Radiation Synchrotron", value=True)
                include_space_charge = st.checkbox("Inclure Charge d'Espace", value=False)
            
            st.markdown("---")
            
            if st.button("🚀 Lancer la Simulation", use_container_width=True, type="primary"):
                with st.spinner("🔄 Simulation en cours..."):
                    progress_bar = st.progress(0)
                    
                    particle_data = PARTICLE_DATA[particle_type]
                    
                    trajectory = {
                        'time': [],
                        'position': [],
                        'energy': [],
                        'velocity': [],
                        'gamma': [],
                        'momentum': []
                    }
                    
                    current_energy = initial_energy
                    
                    for step in range(n_steps):
                        progress_bar.progress((step + 1) / n_steps)
                        
                        params = calculate_relativistic_params(particle_type, current_energy)
                        
                        if acc['type'] == 'lineaire':
                            position = (step / n_steps) * acc['geometry']['length']
                        else:
                            position = (step / n_steps) * acc['geometry']['circumference']
                        
                        if acc['type'] == 'lineaire':
                            energy_gain = (acc['energy']['max'] - initial_energy) / n_steps
                        else:
                            energy_gain = acc['energy']['max'] / (n_steps * 10)  
                        
                        current_energy += energy_gain
                        current_energy = min(current_energy, acc['energy']['max'])
                        
                        if include_radiation and particle_type in ['electron', 'positron'] and acc['geometry']['radius'] > 0:
                            r_e = 2.8179e-15  

                            current_energy = min(current_energy, 1e15)  

                            try:
                                
                                energy_joules = current_energy * constants.e

                                log_power_loss = (
                                    np.log(2 * r_e * constants.c)
                                    - np.log(3)
                                    - 2 * np.log(acc['geometry']['radius'])
                                    + 4 * np.log(energy_joules)
                                    - 4 * np.log(particle_data['mass'] * constants.c**2)
                                )

                                power_loss = np.exp(np.clip(log_power_loss, -700, 700))

                                energy_loss = power_loss * time_step * 1e-9 / constants.e
                                current_energy = max(current_energy - energy_loss, 0)

                            except (OverflowError, FloatingPointError, ValueError):
                                current_energy = 0
                        
                        # Enregistrer
                        trajectory['time'].append(step * time_step)
                        trajectory['position'].append(position)
                        trajectory['energy'].append(current_energy)
                        trajectory['velocity'].append(params['velocity'])
                        trajectory['gamma'].append(params['gamma'])
                        trajectory['momentum'].append(params['momentum'])
                    
                    progress_bar.empty()
                    
                    # Sauvegarder la simulation
                    sim_id = f"sim_{len(st.session_state.accelerator_system['simulations']) + 1}"
                    
                    simulation = {
                        'simulation_id': sim_id,
                        'accelerator_id': selected_acc,
                        'accelerator_name': acc['name'],
                        'particle_type': particle_type,
                        'n_particles': n_particles,
                        'initial_energy': initial_energy,
                        'final_energy': trajectory['energy'][-1],
                        'trajectory': trajectory,
                        'parameters': {
                            'n_steps': n_steps,
                            'time_step': time_step,
                            'emittance': beam_emittance,
                            'energy_spread': beam_spread,
                            'radiation': include_radiation,
                            'space_charge': include_space_charge
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.accelerator_system['simulations'].append(simulation)
                    log_event(f"Simulation complétée: {acc['name']} - {particle_type}")
                    
                    st.success("✅ Simulation terminée!")
                    st.balloons()
                    
                    # Afficher les résultats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Énergie Initiale", format_energy(initial_energy))
                    with col2:
                        st.metric("Énergie Finale", format_energy(trajectory['energy'][-1]))
                    with col3:
                        st.metric("Gain Total", format_energy(trajectory['energy'][-1] - initial_energy))
                    with col4:
                        st.metric("Temps Total", f"{trajectory['time'][-1]:.2f} ns")
                    
                    # Graphiques
                    st.markdown("---")
                    st.subheader("📊 Résultats de la Simulation")
                    
                    # Énergie vs Position
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("Énergie vs Position", "Vitesse vs Temps", "Facteur γ vs Position", "Momentum vs Temps")
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=trajectory['position'], y=[e/1e9 for e in trajectory['energy']], 
                                  mode='lines', name='Énergie', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=trajectory['time'], y=[v/constants.c for v in trajectory['velocity']], 
                                  mode='lines', name='Vitesse', line=dict(color='green', width=2)),
                        row=1, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=trajectory['position'], y=trajectory['gamma'], 
                                  mode='lines', name='γ', line=dict(color='red', width=2)),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=trajectory['time'], y=trajectory['momentum'], 
                                  mode='lines', name='Momentum', line=dict(color='purple', width=2)),
                        row=2, col=2
                    )
                    
                    fig.update_xaxes(title_text="Position (m)", row=1, col=1)
                    fig.update_xaxes(title_text="Temps (ns)", row=1, col=2)
                    fig.update_xaxes(title_text="Position (m)", row=2, col=1)
                    fig.update_xaxes(title_text="Temps (ns)", row=2, col=2)
                    
                    fig.update_yaxes(title_text="Énergie (GeV)", row=1, col=1)
                    fig.update_yaxes(title_text="v/c", row=1, col=2)
                    fig.update_yaxes(title_text="γ (Lorentz)", row=2, col=1)
                    fig.update_yaxes(title_text="Momentum (kg·m/s)", row=2, col=2)
                    
                    fig.update_layout(height=700, showlegend=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Analyse des Résultats")
            
            if not st.session_state.accelerator_system['simulations']:
                st.info("Aucune simulation disponible")
            else:
                sim_options = {s['simulation_id']: f"{s['accelerator_name']} - {s['particle_type']}" 
                              for s in st.session_state.accelerator_system['simulations']}
                
                selected_sim = st.selectbox(
                    "Sélectionner une simulation",
                    options=list(sim_options.keys()),
                    format_func=lambda x: sim_options[x]
                )
                
                sim = next(s for s in st.session_state.accelerator_system['simulations'] if s['simulation_id'] == selected_sim)
                
                st.write(f"### 🔬 {sim['accelerator_name']}")
                st.write(f"**Particule:** {PARTICLE_DATA[sim['particle_type']]['symbol']} {sim['particle_type'].title()}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Nombre Particules", f"{sim['n_particles']:,}")
                with col2:
                    st.metric("Énergie Initiale", format_energy(sim['initial_energy']))
                with col3:
                    st.metric("Énergie Finale", format_energy(sim['final_energy']))
                with col4:
                    gain = sim['final_energy'] - sim['initial_energy']
                    st.metric("Gain Énergie", format_energy(gain))
                
                st.markdown("---")
                
                # Distribution d'énergie
                st.subheader("📈 Distribution d'Énergie")
                
                energies = [e/1e9 for e in sim['trajectory']['energy']]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=energies, nbinsx=50, marker_color='rgba(102, 126, 234, 0.7)'))
                fig.update_layout(
                    title="Distribution de l'Énergie des Particules",
                    xaxis_title="Énergie (GeV)",
                    yaxis_title="Nombre de Particules",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Phase space
                st.markdown("---")
                st.subheader("🌀 Espace des Phases")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sim['trajectory']['position'],
                    y=[e/1e9 for e in sim['trajectory']['energy']],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=sim['trajectory']['time'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Temps (ns)")
                    )
                ))
                fig.update_layout(
                    title="Espace des Phases (Position-Énergie)",
                    xaxis_title="Position (m)",
                    yaxis_title="Énergie (GeV)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📜 Historique des Simulations")
            
            if st.session_state.accelerator_system['simulations']:
                sim_data = []
                for sim in st.session_state.accelerator_system['simulations']:
                    sim_data.append({
                        'ID': sim['simulation_id'],
                        'Accélérateur': sim['accelerator_name'],
                        'Particule': sim['particle_type'],
                        'N Particules': f"{sim['n_particles']:,}",
                        'Énergie Initiale': format_energy(sim['initial_energy']),
                        'Énergie Finale': format_energy(sim['final_energy']),
                        'Date': sim['timestamp'][:10]
                    })
                
                df = pd.DataFrame(sim_data)
                st.dataframe(df, use_container_width=True)
                
                # Export
                if st.button("📥 Exporter les Simulations (CSV)"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Télécharger CSV",
                        data=csv,
                        file_name=f"simulations_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Aucune simulation disponible")

# ==================== PAGE: COLLISIONS ====================

elif page == "💥 Collisions":
    st.header("💥 Simulations de Collisions")
    
    if not st.session_state.accelerator_system['accelerators']:
        st.warning("⚠️ Aucun accélérateur disponible")
    else:
        tab1, tab2, tab3 = st.tabs(["🎯 Nouvelle Collision", "📊 Analyse", "📜 Historique"])
        
        with tab1:
            st.subheader("🎯 Configurer une Collision")
            
            acc_options = {a['id']: a['name'] for a in st.session_state.accelerator_system['accelerators'].values()}
            selected_acc = st.selectbox(
                "Sélectionner un accélérateur",
                options=list(acc_options.keys()),
                format_func=lambda x: acc_options[x],
                key="collision_acc"
            )
            
            acc = st.session_state.accelerator_system['accelerators'][selected_acc]
            
            st.write(f"### ⚛️ {acc['name']}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔴 Faisceau 1:**")
                beam1_particle = st.selectbox(
                    "Particule 1",
                    list(PARTICLE_DATA.keys()),
                    format_func=lambda x: f"{PARTICLE_DATA[x]['symbol']} {x.title()}",
                    key="beam1"
                )
                
                beam1_energy_unit = st.selectbox("Unité Énergie 1", ["MeV", "GeV", "TeV"], index=1, key="unit1")
                mult1 = {'MeV': 1e6, 'GeV': 1e9, 'TeV': 1e12}[beam1_energy_unit]
                
                beam1_energy_val = st.number_input(f"Énergie Faisceau 1 ({beam1_energy_unit})", 0.001, 100000.0, 7000.0, key="e1")
                beam1_energy = beam1_energy_val * mult1
                
                beam1_intensity = st.number_input("Intensité 1 (particules/bunch)", 1e8, 1e12, 1e11, format="%.2e", key="int1")
            
            with col2:
                st.write("**🔵 Faisceau 2:**")
                beam2_particle = st.selectbox(
                    "Particule 2",
                    list(PARTICLE_DATA.keys()),
                    format_func=lambda x: f"{PARTICLE_DATA[x]['symbol']} {x.title()}",
                    key="beam2"
                )
                
                beam2_energy_unit = st.selectbox("Unité Énergie 2", ["MeV", "GeV", "TeV"], index=1, key="unit2")
                mult2 = {'MeV': 1e6, 'GeV': 1e9, 'TeV': 1e12}[beam2_energy_unit]
                
                beam2_energy_val = st.number_input(f"Énergie Faisceau 2 ({beam2_energy_unit})", 0.001, 100000.0, 7000.0, key="e2")
                beam2_energy = beam2_energy_val * mult2
                
                beam2_intensity = st.number_input("Intensité 2 (particules/bunch)", 1e8, 1e12, 1e11, format="%.2e", key="int2")
            
            st.markdown("---")
            
            # Paramètres de collision
            st.subheader("⚙️ Paramètres de Collision")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_bunches = st.number_input("Nombre de Paquets", 1, 10000, 2808)
                bunch_spacing = st.number_input("Espacement Paquets (ns)", 1, 1000, 25)
            
            with col2:
                crossing_angle = st.slider("Angle de Croisement (mrad)", 0.0, 500.0, 285.0)
                beta_star = st.number_input("β* au Point d'Interaction (m)", 0.01, 10.0, 0.55)
            
            with col3:
                n_events = st.number_input("Nombre d'Événements à Simuler", 100, 1000000, 10000)
                collision_frequency = st.number_input("Fréquence Collision (kHz)", 1, 100, 40)
            
            # Calcul énergie centre de masse
            params1 = calculate_relativistic_params(beam1_particle, beam1_energy)
            params2 = calculate_relativistic_params(beam2_particle, beam2_energy)
            
            # Énergie dans le centre de masse (collision frontale)
            E_cm = np.sqrt(2 * beam1_energy * beam2_energy * (1 + params1['beta'] * params2['beta']))
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Énergie Centre de Masse", format_energy(E_cm))
            with col2:
                # Luminosité instantanée (formule simplifiée)
                lumi = (n_bunches * collision_frequency * 1e3 * beam1_intensity * beam2_intensity) / (4 * np.pi * beta_star**2)
                st.metric("Luminosité Instantanée", f"{lumi:.2e} cm⁻²s⁻¹")
            with col3:
                # Luminosité intégrée (sur 1 an)
                lumi_int = lumi * 3600 * 24 * 365 / 1e39  # fb⁻¹
                st.metric("Luminosité Intégrée/an", f"{lumi_int:.1f} fb⁻¹")
            
            st.markdown("---")
            
            if st.button("💥 Lancer la Simulation de Collision", use_container_width=True, type="primary"):
                with st.spinner("🔄 Simulation des collisions en cours..."):
                    progress_bar = st.progress(0)
                    
                    # Simulation des événements
                    events = {
                        'elastic': 0,
                        'inelastic': 0,
                        'diffractive': 0,
                        'hard_scattering': 0,
                        'particle_production': []
                    }
                    
                    detected_particles = []
                    energy_deposits = []
                    
                    for i in range(n_events):
                        progress_bar.progress((i + 1) / n_events)
                        
                        # Type d'événement
                        rand = np.random.random()
                        
                        if rand < 0.25:
                            events['elastic'] += 1
                            event_type = 'elastic'
                        elif rand < 0.65:
                            events['inelastic'] += 1
                            event_type = 'inelastic'
                        elif rand < 0.85:
                            events['diffractive'] += 1
                            event_type = 'diffractive'
                        else:
                            events['hard_scattering'] += 1
                            event_type = 'hard_scattering'
                        
                        # Production de particules
                        if event_type in ['inelastic', 'hard_scattering']:
                            n_produced = np.random.poisson(10 if event_type == 'inelastic' else 30)
                            events['particle_production'].append(n_produced)
                            
                            # Particules détectées
                            for _ in range(n_produced):
                                particle_energy = np.random.exponential(E_cm / 20)
                                theta = np.abs(np.random.normal(0, 0.5))  # Angle polaire
                                phi = np.random.uniform(0, 2*np.pi)  # Angle azimutal
                                
                                detected_particles.append({
                                    'energy': particle_energy,
                                    'theta': theta,
                                    'phi': phi,
                                    'pt': particle_energy * np.sin(theta),  # Momentum transverse
                                    'eta': -np.log(np.tan(theta/2))  # Pseudorapidité
                                })
                                
                                energy_deposits.append(particle_energy)
                    
                    progress_bar.empty()
                    
                    # Sauvegarder la collision
                    collision_id = f"col_{len(st.session_state.accelerator_system['collisions']) + 1}"
                    
                    collision = {
                        'collision_id': collision_id,
                        'accelerator_id': selected_acc,
                        'accelerator_name': acc['name'],
                        'beam1': {
                            'particle': beam1_particle,
                            'energy': beam1_energy,
                            'intensity': beam1_intensity
                        },
                        'beam2': {
                            'particle': beam2_particle,
                            'energy': beam2_energy,
                            'intensity': beam2_intensity
                        },
                        'parameters': {
                            'n_bunches': n_bunches,
                            'bunch_spacing': bunch_spacing,
                            'crossing_angle': crossing_angle,
                            'beta_star': beta_star,
                            'E_cm': E_cm,
                            'luminosity': lumi,
                            'luminosity_integrated': lumi_int
                        },
                        'events': events,
                        'n_events': n_events,
                        'detected_particles': detected_particles,
                        'energy_deposits': energy_deposits,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.accelerator_system['collisions'].append(collision)
                    
                    # Mettre à jour l'accélérateur
                    acc['total_collisions'] += n_events
                    
                    log_event(f"Collision simulée: {acc['name']} - {n_events} événements")
                    
                    st.success("✅ Simulation de collision terminée!")
                    st.balloons()
                    
                    # Résultats
                    st.markdown("---")
                    st.subheader("📊 Résultats de la Collision")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Événements Totaux", f"{n_events:,}")
                    with col2:
                        st.metric("Élastiques", f"{events['elastic']:,}")
                    with col3:
                        st.metric("Inélastiques", f"{events['inelastic']:,}")
                    with col4:
                        st.metric("Diffusion Dure", f"{events['hard_scattering']:,}")
                    
                    # Graphiques
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution types d'événements
                        labels = ['Élastique', 'Inélastique', 'Diffractif', 'Diffusion Dure']
                        values = [events['elastic'], events['inelastic'], events['diffractive'], events['hard_scattering']]
                        
                        fig = px.pie(values=values, names=labels, title="Distribution des Types d'Événements",
                                    color_discrete_sequence=px.colors.sequential.Purples_r)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Multiplicité
                        if events['particle_production']:
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=events['particle_production'], nbinsx=30,
                                                      marker_color='rgba(102, 126, 234, 0.7)'))
                            fig.update_layout(title="Multiplicité des Particules Produites",
                                            xaxis_title="Nombre de Particules",
                                            yaxis_title="Fréquence")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Dépôts d'énergie
                    if energy_deposits:
                        st.markdown("---")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=[e/1e9 for e in energy_deposits], nbinsx=50,
                                                  marker_color='rgba(102, 126, 234, 0.7)'))
                        fig.update_layout(title="Distribution des Dépôts d'Énergie",
                                        xaxis_title="Énergie (GeV)",
                                        yaxis_title="Nombre de Particules")
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Analyse Détaillée des Collisions")
            
            if not st.session_state.accelerator_system['collisions']:
                st.info("Aucune collision disponible")
            else:
                col_options = {c['collision_id']: f"{c['accelerator_name']} - {c['beam1']['particle']} vs {c['beam2']['particle']}" 
                              for c in st.session_state.accelerator_system['collisions']}
                
                selected_col = st.selectbox(
                    "Sélectionner une collision",
                    options=list(col_options.keys()),
                    format_func=lambda x: col_options[x]
                )
                
                col_data = next(c for c in st.session_state.accelerator_system['collisions'] if c['collision_id'] == selected_col)
                
                st.write(f"### 💥 Collision: {col_data['beam1']['particle']} + {col_data['beam2']['particle']}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Énergie CM", format_energy(col_data['parameters']['E_cm']))
                with col2:
                    st.metric("Luminosité", f"{col_data['parameters']['luminosity']:.2e} cm⁻²s⁻¹")
                with col3:
                    st.metric("Événements", f"{col_data['n_events']:,}")
                with col4:
                    total_particles = sum(col_data['events']['particle_production'])
                    st.metric("Particules Produites", f"{total_particles:,}")
                
                st.markdown("---")
                
                # Analyse des particules détectées
                if col_data['detected_particles']:
                    st.subheader("📡 Particules Détectées")
                    
                    particles_df = pd.DataFrame(col_data['detected_particles'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribution en énergie
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=particles_df['energy']/1e9, nbinsx=50,
                                                  marker_color='rgba(102, 126, 234, 0.7)'))
                        fig.update_layout(title="Distribution en Énergie",
                                        xaxis_title="Énergie (GeV)",
                                        yaxis_title="Nombre de Particules",
                                        xaxis_type="log")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Distribution en pT
                        fig = go.Figure()
                        fig.add_trace(go.Histogram(x=particles_df['pt']/1e9, nbinsx=50,
                                                  marker_color='rgba(118, 75, 162, 0.7)'))
                        fig.update_layout(title="Distribution en Momentum Transverse",
                                        xaxis_title="pT (GeV/c)",
                                        yaxis_title="Nombre de Particules",
                                        xaxis_type="log")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Carte η-φ
                    st.markdown("---")
                    st.subheader("🗺️ Carte Pseudorapidité-Azimut (η-φ)")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=particles_df['phi'],
                        y=particles_df['eta'],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=particles_df['energy']/1e9,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Énergie (GeV)")
                        )
                    ))
                    fig.update_layout(
                        title="Distribution des Particules (η-φ)",
                        xaxis_title="φ (rad)",
                        yaxis_title="η (pseudorapidité)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    st.markdown("---")
                    st.subheader("📊 Statistiques")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Énergie Moyenne", f"{particles_df['energy'].mean()/1e9:.2f} GeV")
                    with col2:
                        st.metric("pT Moyen", f"{particles_df['pt'].mean()/1e9:.2f} GeV/c")
                    with col3:
                        st.metric("η Moyen", f"{particles_df['eta'].mean():.2f}")
                    with col4:
                        st.metric("Particules Centrales (|η|<2.5)", 
                                 f"{len(particles_df[abs(particles_df['eta']) < 2.5]):,}")
        
        with tab3:
            st.subheader("📜 Historique des Collisions")
            
            if st.session_state.accelerator_system['collisions']:
                col_data = []
                for col in st.session_state.accelerator_system['collisions']:
                    col_data.append({
                        'ID': col['collision_id'],
                        'Accélérateur': col['accelerator_name'],
                        'Faisceau 1': f"{col['beam1']['particle']} @ {format_energy(col['beam1']['energy'])}",
                        'Faisceau 2': f"{col['beam2']['particle']} @ {format_energy(col['beam2']['energy'])}",
                        'E_cm': format_energy(col['parameters']['E_cm']),
                        'Événements': f"{col['n_events']:,}",
                        'Luminosité': f"{col['parameters']['luminosity']:.2e}",
                        'Date': col['timestamp'][:10]
                    })
                
                df = pd.DataFrame(col_data)
                st.dataframe(df, use_container_width=True)
                
                # Export
                if st.button("📥 Exporter les Collisions (CSV)"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Télécharger CSV",
                        data=csv,
                        file_name=f"collisions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Aucune collision disponible")

# ==================== PAGE: DÉTECTEURS ====================

elif page == "📡 Détecteurs":
    st.header("📡 Systèmes de Détection")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "📊 Performance", "📈 Données"])
    
    with tab1:
        st.subheader("🔧 Configuration des Détecteurs")
        
        detector_types = {
            "Calorimètre Électromagnétique": {
                "description": "Mesure l'énergie des électrons et photons",
                "resolution": "10%/√E ⊕ 1%",
                "coverage": "|η| < 3.0",
                "technology": "Cristaux PbWO₄"
            },
            "Calorimètre Hadronique": {
                "description": "Mesure l'énergie des hadrons",
                "resolution": "50%/√E ⊕ 3%",
                "coverage": "|η| < 5.0",
                "technology": "Scintillateur/Absorbeur"
            },
            "Trajectographe Silicium": {
                "description": "Reconstruction des traces de particules chargées",
                "resolution": "15 μm",
                "coverage": "|η| < 2.5",
                "technology": "Pixels et strips silicium"
            },
            "Chambres à Muons": {
                "description": "Détection et mesure des muons",
                "resolution": "100 μm",
                "coverage": "|η| < 2.4",
                "technology": "Chambres à dérive gazeuses"
            },
            "Détecteur Cherenkov": {
                "description": "Identification des particules par radiation Cherenkov",
                "resolution": "1 mrad",
                "coverage": "2.0 < η < 5.0",
                "technology": "Aérogel et gaz"
            }
        }
        
        for det_name, det_info in detector_types.items():
            with st.expander(f"📡 {det_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {det_info['description']}")
                    st.write(f"**Résolution:** {det_info['resolution']}")
                with col2:
                    st.write(f"**Couverture:** {det_info['coverage']}")
                    st.write(f"**Technologie:** {det_info['technology']}")
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    efficiency = st.slider(f"Efficacité", 0.5, 1.0, 0.95, 0.01, key=f"eff_{det_name}")
                with col2:
                    rate = st.number_input(f"Taux Max (kHz)", 1, 1000, 100, key=f"rate_{det_name}")
                with col3:
                    occupancy = st.slider(f"Occupation", 0.0, 1.0, 0.1, 0.01, key=f"occ_{det_name}")
                
                if st.button(f"💾 Sauvegarder Config", key=f"save_{det_name}"):
                    st.success(f"✅ Configuration {det_name} sauvegardée!")
    
    with tab2:
        st.subheader("📊 Performance des Détecteurs")
        
        # Simulation de données de performance
        st.write("### 📈 Résolution en Énergie")
        
        energies = np.logspace(0, 3, 50)  # 1 GeV à 1 TeV
        
        # Résolutions
        em_resolution = 0.10 / np.sqrt(energies) + 0.01
        had_resolution = 0.50 / np.sqrt(energies) + 0.03
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=energies, y=em_resolution*100, mode='lines',
                                name='EM Calorimètre', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=energies, y=had_resolution*100, mode='lines',
                                name='Hadron Calorimètre', line=dict(color='red', width=3)))
        
        fig.update_layout(
            title="Résolution en Énergie vs Énergie",
            xaxis_title="Énergie (GeV)",
            yaxis_title="Résolution σ(E)/E (%)",
            xaxis_type="log"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Efficacité vs η
        st.write("### 🎯 Efficacité vs Pseudorapidité")
        
        eta = np.linspace(-5, 5, 100)
        
        # Efficacités simulées
        tracker_eff = np.where(np.abs(eta) < 2.5, 0.95, 0)
        muon_eff = np.where(np.abs(eta) < 2.4, 0.90, 0)
        em_cal_eff = np.where(np.abs(eta) < 3.0, 0.98, 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eta, y=tracker_eff, mode='lines',
                                name='Trajectographe', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatter(x=eta, y=muon_eff, mode='lines',
                                name='Chambres à Muons', line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=eta, y=em_cal_eff, mode='lines',
                                name='Calorimètre EM', line=dict(color='purple', width=2)))
        
        fig.update_layout(
            title="Efficacité de Détection vs Pseudorapidité",
            xaxis_title="η (pseudorapidité)",
            yaxis_title="Efficacité",
            yaxis_range=[0, 1.1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Matrice de confusion
        st.write("### 🔍 Identification des Particules")
        
        particles = ['e±', 'γ', 'μ±', 'π±', 'K±', 'p']
        
        # Matrice de confusion simulée (identification)
        confusion = np.array([
            [0.95, 0.03, 0.00, 0.01, 0.01, 0.00],  # e±
            [0.02, 0.96, 0.00, 0.01, 0.01, 0.00],  # γ
            [0.00, 0.00, 0.98, 0.01, 0.01, 0.00],  # μ±
            [0.01, 0.01, 0.01, 0.90, 0.05, 0.02],  # π±
            [0.01, 0.01, 0.01, 0.05, 0.89, 0.03],  # K±
            [0.00, 0.00, 0.00, 0.02, 0.03, 0.95],  # p
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion,
            x=particles,
            y=particles,
            colorscale='Blues',
            text=confusion,
            texttemplate='%{text:.2f}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Matrice de Confusion - Identification des Particules",
            xaxis_title="Particule Identifiée",
            yaxis_title="Vraie Particule",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Acquisition de Données")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taux de Déclenchement", "100 kHz")
        with col2:
            st.metric("Taux d'Enregistrement", "10 kHz")
        with col3:
            st.metric("Taille Événement", "1.5 MB")
        with col4:
            st.metric("Débit Données", "15 GB/s")
        
        st.markdown("---")
        
        # Simulation du taux de déclenchement
        st.write("### ⚡ Taux de Déclenchement en Temps Réel")
        
        time = np.linspace(0, 60, 300)  # 1 minute
        trigger_rate = 100000 + np.random.normal(0, 5000, 300) + 20000*np.sin(time/10)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=trigger_rate, mode='lines',
                                line=dict(color='blue', width=2)))
        fig.add_hline(y=100000, line_dash="dash", line_color="green",
                     annotation_text="Taux Nominal")
        
        fig.update_layout(
            title="Taux de Déclenchement",
            xaxis_title="Temps (s)",
            yaxis_title="Taux (Hz)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Volume de données
        st.write("### 💾 Volume de Données Collectées")
        
        days = np.arange(1, 31)
        data_volume = np.cumsum(np.random.normal(15, 2, 30))  # TB par jour
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=days, y=data_volume, marker_color='rgba(102, 126, 234, 0.7)'))
        
        fig.update_layout(
            title="Volume de Données par Jour",
            xaxis_title="Jour du Mois",
            yaxis_title="Volume (TB)"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: EXPÉRIENCES ====================

elif page == "🧪 Expériences":
    st.header("🧪 Gestion des Expériences")
    
    tab1, tab2, tab3 = st.tabs(["➕ Nouvelle Expérience", "📊 Expériences Actives", "📜 Historique"])
    
    with tab1:
        st.subheader("➕ Créer une Nouvelle Expérience")
        
        with st.form("new_experiment"):
            col1, col2 = st.columns(2)
            
            with col1:
                exp_name = st.text_input("📝 Nom de l'Expérience", placeholder="Ex: Recherche Boson de Higgs")
                
                exp_type = st.selectbox(
                    "🎯 Type d'Expérience",
                    ["Physique Fondamentale", "Cible Fixe", "Spectroscopie", 
                     "Science des Matériaux", "Médical", "Irradiation"]
                )
                
                if st.session_state.accelerator_system['accelerators']:
                    acc_options = {a['id']: a['name'] for a in st.session_state.accelerator_system['accelerators'].values()}
                    selected_acc = st.selectbox("Accélérateur", list(acc_options.keys()),
                                               format_func=lambda x: acc_options[x])
                else:
                    st.warning("Aucun accélérateur disponible")
                    selected_acc = None
            
            with col2:
                principal_investigator = st.text_input("👨‍🔬 Chercheur Principal")
                institution = st.text_input("🏛️ Institution")
                start_date = st.date_input("📅 Date de Début")
                duration = st.number_input("⏱️ Durée (mois)", 1, 120, 12)
            
            st.markdown("---")
            
            objectives = st.text_area("🎯 Objectifs de l'Expérience", height=100,
                                     placeholder="Décrivez les objectifs scientifiques...")
            
            physics_goals = st.text_area("⚛️ Questions Physiques", height=100,
                                         placeholder="Quelles questions physiques cherchez-vous à résoudre?")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                beam_energy = st.number_input("Énergie Faisceau (GeV)", 0.1, 100000.0, 7000.0)
            with col2:
                target_luminosity = st.number_input("Luminosité Cible (fb⁻¹)", 0.1, 10000.0, 100.0)
            with col3:
                expected_events = st.number_input("Événements Attendus", 1000, int(1e9), 1000000)
            
            submitted = st.form_submit_button("🚀 Créer l'Expérience", use_container_width=True, type="primary")
            
            if submitted:
                if not exp_name or not principal_investigator or selected_acc is None:
                    st.error("⚠️ Veuillez remplir tous les champs obligatoires")
                else:
                    exp_id = f"exp_{len(st.session_state.accelerator_system['experiments']) + 1}"
                    
                    experiment = {
                        'experiment_id': exp_id,
                        'name': exp_name,
                        'type': exp_type,
                        'accelerator_id': selected_acc,
                        'principal_investigator': principal_investigator,
                        'institution': institution,
                        'start_date': start_date.isoformat(),
                        'duration': duration,
                        'objectives': objectives,
                        'physics_goals': physics_goals,
                        'parameters': {
                            'beam_energy': beam_energy,
                            'target_luminosity': target_luminosity,
                            'expected_events': expected_events
                        },
                        'status': 'planned',
                        'progress': 0.0,
                        'events_recorded': 0,
                        'data_volume': 0.0,
                        'results': {},
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.accelerator_system['experiments'][exp_id] = experiment
                    log_event(f"Expérience créée: {exp_name}")
                    
                    st.success(f"✅ Expérience '{exp_name}' créée avec succès!")
                    st.balloons()
                    
                    st.code(f"Experiment ID: {exp_id}", language="text")
    
    with tab2:
        st.subheader("📊 Expériences Actives")
        
        if not st.session_state.accelerator_system['experiments']:
            st.info("Aucune expérience créée")
        else:
            for exp_id, exp in st.session_state.accelerator_system['experiments'].items():
                status_colors = {
                    'planned': '🟡',
                    'running': '🟢',
                    'paused': '🟠',
                    'completed': '✅',
                    'analysing': '🔵'
                }
                
                status_icon = status_colors.get(exp['status'], '⚪')
                
                with st.expander(f"{status_icon} {exp['name']} - {exp['progress']:.0f}%"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Type:** {exp['type']}")
                        st.write(f"**Chercheur Principal:** {exp['principal_investigator']}")
                        st.write(f"**Institution:** {exp['institution']}")
                    
                    with col2:
                        st.metric("Progression", f"{exp['progress']:.0f}%")
                        st.progress(exp['progress'] / 100)
                        
                        st.metric("Événements Enregistrés", f"{exp['events_recorded']:,}")
                    
                    with col3:
                        st.metric("Luminosité Cible", f"{exp['parameters']['target_luminosity']:.1f} fb⁻¹")
                        st.metric("Volume de Données", f"{exp['data_volume']:.2f} TB")
                    
                    st.markdown("---")
                    
                    if exp['objectives']:
                        st.write("**🎯 Objectifs:**")
                        st.info(exp['objectives'][:200] + "...")
                    
                    st.markdown("---")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if exp['status'] == 'planned':
                            if st.button("▶️ Démarrer", key=f"start_{exp_id}"):
                                exp['status'] = 'running'
                                log_event(f"Expérience démarrée: {exp['name']}")
                                st.rerun()
                        elif exp['status'] == 'running':
                            if st.button("⏸️ Pause", key=f"pause_{exp_id}"):
                                exp['status'] = 'paused'
                                st.rerun()
                    
                    with col2:
                        new_progress = st.number_input("Progression", 0, 100, int(exp['progress']), key=f"prog_{exp_id}")
                        if st.button("💾 MAJ", key=f"update_{exp_id}"):
                            exp['progress'] = float(new_progress)
                            if new_progress >= 100:
                                exp['status'] = 'completed'
                            st.success("Mis à jour!")
                            st.rerun()
                    
                    with col3:
                        if st.button("📊 Analyser", key=f"analyze_{exp_id}"):
                            exp['status'] = 'analysing'
                            st.info("Analyse en cours...")
                    
                    with col4:
                        if st.button("📄 Rapport", key=f"report_{exp_id}"):
                            st.info("Génération du rapport...")
    
    with tab3:
        st.subheader("📜 Historique des Expériences")
        
        if st.session_state.accelerator_system['experiments']:
            exp_data = []
            for exp in st.session_state.accelerator_system['experiments'].values():
                exp_data.append({
                    'Nom': exp['name'],
                    'Type': exp['type'],
                    'Chercheur': exp['principal_investigator'],
                    'Institution': exp['institution'],
                    'Statut': exp['status'].upper(),
                    'Progression': f"{exp['progress']:.0f}%",
                    'Événements': f"{exp['events_recorded']:,}",
                    'Date': exp['start_date'][:10]
                })
            
            df = pd.DataFrame(exp_data)
            st.dataframe(df, use_container_width=True)
            
            # Export
            if st.button("📥 Exporter les Expériences (CSV)"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Télécharger CSV",
                    data=csv,
                    file_name=f"experiments_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("Aucune expérience disponible")

# ==================== PAGE: ANALYSES & RÉSULTATS ====================

elif page == "📊 Analyses & Résultats":
    st.header("📊 Analyses et Résultats Scientifiques")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analyse de Données", "🔬 Reconstruction", "📊 Statistiques", "📄 Publications"])
    
    with tab1:
        st.subheader("📈 Analyse des Données Expérimentales")
        
        if not st.session_state.accelerator_system['collisions']:
            st.info("Aucune donnée de collision disponible")
        else:
            # Sélection des données
            col_options = {c['collision_id']: f"{c['accelerator_name']} - {c['timestamp'][:10]}" 
                          for c in st.session_state.accelerator_system['collisions']}
            
            selected_cols = st.multiselect(
                "Sélectionner les données à analyser",
                options=list(col_options.keys()),
                format_func=lambda x: col_options[x],
                default=[list(col_options.keys())[0]] if col_options else []
            )
            
            if selected_cols:
                # Combiner les données
                all_particles = []
                total_events = 0
                
                for col_id in selected_cols:
                    col_data = next(c for c in st.session_state.accelerator_system['collisions'] if c['collision_id'] == col_id)
                    all_particles.extend(col_data['detected_particles'])
                    total_events += col_data['n_events']
                
                st.write(f"### 📊 Analyse de {len(selected_cols)} jeu(x) de données")
                st.write(f"**Événements Totaux:** {total_events:,}")
                st.write(f"**Particules Détectées:** {len(all_particles):,}")
                
                st.markdown("---")
                
                # Convertir en DataFrame
                df_particles = pd.DataFrame(all_particles)
                
                # Sélection des coupures
                st.subheader("✂️ Coupures Cinématiques")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    pt_min = st.slider("pT minimum (GeV/c)", 0.0, 100.0, 10.0, 1.0)
                    pt_max = st.slider("pT maximum (GeV/c)", 0.0, 1000.0, 500.0, 10.0)
                
                with col2:
                    eta_min = st.slider("η minimum", -5.0, 0.0, -2.5, 0.1)
                    eta_max = st.slider("η maximum", 0.0, 5.0, 2.5, 0.1)
                
                with col3:
                    e_min = st.slider("E minimum (GeV)", 0.0, 100.0, 5.0, 1.0)
                    e_max = st.slider("E maximum (GeV)", 0.0, 10000.0, 5000.0, 100.0)
                
                # Appliquer les coupures
                df_cut = df_particles[
                    (df_particles['pt']/1e9 >= pt_min) & 
                    (df_particles['pt']/1e9 <= pt_max) &
                    (df_particles['eta'] >= eta_min) & 
                    (df_particles['eta'] <= eta_max) &
                    (df_particles['energy']/1e9 >= e_min) & 
                    (df_particles['energy']/1e9 <= e_max)
                ]
                
                acceptance = len(df_cut) / len(df_particles) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Particules Avant Coupures", f"{len(df_particles):,}")
                with col2:
                    st.metric("Particules Après Coupures", f"{len(df_cut):,}")
                with col3:
                    st.metric("Acceptance", f"{acceptance:.1f}%")
                
                st.markdown("---")
                
                # Distributions
                st.subheader("📊 Distributions Cinématiques")
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Distribution en pT", "Distribution en η", 
                                   "Distribution en E", "Carte η vs φ")
                )
                
                # pT
                fig.add_trace(
                    go.Histogram(x=df_cut['pt']/1e9, nbinsx=50, name='pT',
                                marker_color='rgba(102, 126, 234, 0.7)'),
                    row=1, col=1
                )
                
                # η
                fig.add_trace(
                    go.Histogram(x=df_cut['eta'], nbinsx=50, name='η',
                                marker_color='rgba(118, 75, 162, 0.7)'),
                    row=1, col=2
                )
                
                # E
                fig.add_trace(
                    go.Histogram(x=df_cut['energy']/1e9, nbinsx=50, name='E',
                                marker_color='rgba(79, 172, 254, 0.7)'),
                    row=2, col=1
                )
                
                # η vs φ
                fig.add_trace(
                    go.Scatter(x=df_cut['phi'], y=df_cut['eta'], mode='markers',
                              marker=dict(size=2, color='rgba(102, 126, 234, 0.5)'),
                              name='Particules'),
                    row=2, col=2
                )
                
                fig.update_xaxes(title_text="pT (GeV/c)", row=1, col=1)
                fig.update_xaxes(title_text="η", row=1, col=2)
                fig.update_xaxes(title_text="E (GeV)", row=2, col=1)
                fig.update_xaxes(title_text="φ (rad)", row=2, col=2)
                
                fig.update_yaxes(title_text="Nombre", row=1, col=1)
                fig.update_yaxes(title_text="Nombre", row=1, col=2)
                fig.update_yaxes(title_text="Nombre", row=2, col=1)
                fig.update_yaxes(title_text="η", row=2, col=2)
                
                fig.update_layout(height=700, showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Reconstruction des Particules")
        
        st.write("### ⚛️ Reconstruction de Masse Invariante")
        
        st.write("""
        La masse invariante permet d'identifier des particules à partir de leurs produits de désintégration.
        Par exemple, le boson Z⁰ se désintègre en deux leptons : Z⁰ → l⁺l⁻
        """)
        
        # Simulation de reconstruction de masse
        if st.button("🔄 Simuler Reconstruction Z⁰ → e⁺e⁻"):
            with st.spinner("Reconstruction en cours..."):
                # Générer des événements avec un pic Z
                n_signal = 1000
                n_background = 5000
                
                # Signal (Z boson)
                z_mass = 91.2  # GeV/c²
                z_width = 2.5  # GeV/c²
                signal_mass = np.random.normal(z_mass, z_width, n_signal)
                
                # Background (continuum)
                background_mass = np.random.exponential(50, n_background) + 40
                
                # Combiner
                all_mass = np.concatenate([signal_mass, background_mass])
                
                # Histogramme
                fig = go.Figure()
                
                counts, bins = np.histogram(all_mass, bins=100, range=(40, 140))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                fig.add_trace(go.Scatter(
                    x=bin_centers,
                    y=counts,
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    name='Données'
                ))
                
                # Fit gaussien pour le signal
                from scipy.optimize import curve_fit
                
                def gauss_plus_exp(x, A, mu, sigma, B, tau):
                    return A * np.exp(-0.5*((x-mu)/sigma)**2) + B * np.exp(-x/tau)
                
                try:
                    mask = (bin_centers > 70) & (bin_centers < 110)
                    popt, _ = curve_fit(gauss_plus_exp, bin_centers[mask], counts[mask],
                                       p0=[1000, z_mass, z_width, 100, 20])
                    
                    fit_x = np.linspace(40, 140, 1000)
                    fit_y = gauss_plus_exp(fit_x, *popt)
                    
                    fig.add_trace(go.Scatter(
                        x=fit_x,
                        y=fit_y,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Fit'
                    ))
                    
                    st.success(f"✅ Reconstruction réussie!")
                    st.write(f"**Masse Reconstruite:** {popt[1]:.2f} ± {popt[2]:.2f} GeV/c²")
                    st.write(f"**Masse PDG (Z⁰):** 91.19 GeV/c²")
                    st.write(f"**Différence:** {abs(popt[1] - 91.19):.2f} GeV/c²")
                    
                except:
                    st.warning("Fit non convergent")
                
                fig.update_layout(
                    title="Distribution de Masse Invariante e⁺e⁻",
                    xaxis_title="M(e⁺e⁻) [GeV/c²]",
                    yaxis_title="Événements / 1 GeV",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Signification statistique
                signal_counts = np.sum((all_mass > 85) & (all_mass < 97))
                background_counts = np.sum(((all_mass > 70) & (all_mass < 85)) | ((all_mass > 97) & (all_mass < 110)))
                background_in_window = background_counts * (12/40)  # Normalisation
                
                significance = signal_counts / np.sqrt(signal_counts + background_in_window)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Signal", f"{signal_counts:.0f}")
                with col2:
                    st.metric("Fond", f"{background_in_window:.0f}")
                with col3:
                    st.metric("Signification (σ)", f"{significance:.1f}")
        
        st.markdown("---")
        
        # Reconstruction de jets
        st.write("### 🌪️ Reconstruction de Jets")
        
        st.info("""
        Les jets sont des gerbes de particules hadroniques issues de quarks ou gluons.
        Algorithmes courants: anti-kT, Cambridge-Aachen, kT
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            jet_algorithm = st.selectbox("Algorithme de Jet", ["anti-kT", "Cambridge-Aachen", "kT"])
            R_parameter = st.slider("Paramètre R", 0.2, 1.0, 0.4, 0.1)
        
        with col2:
            pt_min_jet = st.number_input("pT minimum du jet (GeV)", 10.0, 500.0, 30.0)
            eta_max_jet = st.slider("|η| maximum du jet", 1.0, 5.0, 2.5, 0.1)
        
        if st.button("🔄 Reconstruire Jets"):
            # Simulation simple
            n_jets = np.random.poisson(4)
            
            jets = []
            for i in range(n_jets):
                jet_pt = np.random.exponential(50) + pt_min_jet
                jet_eta = np.random.uniform(-eta_max_jet, eta_max_jet)
                jet_phi = np.random.uniform(0, 2*np.pi)
                jet_mass = np.random.normal(5, 2)
                
                jets.append({
                    'pt': jet_pt,
                    'eta': jet_eta,
                    'phi': jet_phi,
                    'mass': jet_mass
                })
            
            st.success(f"✅ {len(jets)} jet(s) reconstruit(s)")
            
            if jets:
                df_jets = pd.DataFrame(jets)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Jets Reconstruits:**")
                    st.dataframe(df_jets.round(2))
                
                with col2:
                    # Visualisation η-φ
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_jets['phi'],
                        y=df_jets['eta'],
                        mode='markers',
                        marker=dict(
                            size=df_jets['pt']/5,
                            color=df_jets['pt'],
                            colorscale='Hot',
                            showscale=True,
                            colorbar=dict(title="pT (GeV)")
                        ),
                        text=[f"Jet {i+1}<br>pT: {j['pt']:.1f} GeV" for i, j in enumerate(jets)],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="Distribution des Jets (η-φ)",
                        xaxis_title="φ (rad)",
                        yaxis_title="η",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Analyses Statistiques")
        
        st.write("### 📈 Tests d'Hypothèses")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Hypothèse Nulle (H₀):**")
            st.info("Pas de nouveau signal physique (Modèle Standard uniquement)")
            
            observed = st.number_input("Événements Observés", 0, 10000, 150)
            expected_bg = st.number_input("Fond Attendu", 0.0, 10000.0, 100.0, 1.0)
            bg_uncertainty = st.number_input("Incertitude Fond (%)", 0.0, 50.0, 10.0, 1.0)
        
        with col2:
            st.write("**Hypothèse Alternative (H₁):**")
            st.info("Présence d'un nouveau signal physique")
            
            expected_signal = st.number_input("Signal Attendu", 0.0, 1000.0, 50.0, 1.0)
            signal_uncertainty = st.number_input("Incertitude Signal (%)", 0.0, 50.0, 20.0, 1.0)
        
        if st.button("🔬 Calculer Signification", use_container_width=True):
            # Calcul simple de signification
            excess = observed - expected_bg
            error = np.sqrt(expected_bg + (bg_uncertainty/100 * expected_bg)**2)
            
            significance = excess / error if error > 0 else 0
            
            # p-value approximative
            from scipy.stats import norm
            p_value = 1 - norm.cdf(significance)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Excès", f"{excess:.1f}")
            with col2:
                st.metric("Erreur", f"{error:.1f}")
            with col3:
                st.metric("Signification", f"{significance:.2f}σ")
            with col4:
                st.metric("p-value", f"{p_value:.2e}")
            
            # Interprétation
            st.markdown("---")
            st.write("**📊 Interprétation:**")
            
            if significance < 3:
                st.info("⚪ Observation non significative (< 3σ)")
            elif significance < 5:
                st.warning("🟡 Évidence (3-5σ) - Nécessite confirmation")
            else:
                st.success("🟢 Découverte (> 5σ) - Forte évidence!")
            
            # Graphique
            x = np.linspace(-5, 10, 1000)
            
            # Distribution fond seul
            y_bg = norm.pdf(x, 0, 1)
            
            # Distribution signal + fond
            y_sb = norm.pdf(x, significance, 1)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x, y=y_bg, mode='lines',
                fill='tozeroy',
                name='Fond seul (H₀)',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=x, y=y_sb, mode='lines',
                fill='tozeroy',
                name='Signal + Fond (H₁)',
                line=dict(color='red', width=2),
                opacity=0.6
            ))
            
            fig.add_vline(x=significance, line_dash="dash", line_color="green",
                         annotation_text=f"{significance:.2f}σ")
            
            fig.update_layout(
                title="Distribution Test Statistique",
                xaxis_title="Valeur Test (σ)",
                yaxis_title="Probabilité",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Limites d'exclusion
        st.write("### 🚫 Limites d'Exclusion")
        
        st.info("""
        En absence de signal, on peut placer des limites supérieures sur la section efficace 
        ou sur des paramètres physiques (masse, couplage, etc.)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_level = st.slider("Niveau de Confiance (%)", 90, 99, 95)
            observed_limit = st.number_input("Limite Observée", 0.0, 100.0, 10.0, 0.1)
        
        with col2:
            expected_limit = st.number_input("Limite Attendue", 0.0, 100.0, 12.0, 0.1)
            one_sigma = st.number_input("Bande 1σ", 0.0, 50.0, 3.0, 0.1)
        
        # Graphique de limite
        mass_points = np.linspace(100, 1000, 50)
        
        # Limites simulées
        obs_limit_curve = observed_limit * (1 + 0.3*np.sin(mass_points/100))
        exp_limit_curve = expected_limit * (1 + 0.2*np.sin(mass_points/100))
        
        fig = go.Figure()
        
        # Bande ±2σ
        fig.add_trace(go.Scatter(
            x=np.concatenate([mass_points, mass_points[::-1]]),
            y=np.concatenate([exp_limit_curve + 2*one_sigma, (exp_limit_curve - 2*one_sigma)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 255, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±2σ'
        ))
        
        # Bande ±1σ
        fig.add_trace(go.Scatter(
            x=np.concatenate([mass_points, mass_points[::-1]]),
            y=np.concatenate([exp_limit_curve + one_sigma, (exp_limit_curve - one_sigma)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1σ'
        ))
        
        # Limite attendue
        fig.add_trace(go.Scatter(
            x=mass_points,
            y=exp_limit_curve,
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name='Limite Attendue'
        ))
        
        # Limite observée
        fig.add_trace(go.Scatter(
            x=mass_points,
            y=obs_limit_curve,
            mode='lines',
            line=dict(color='red', width=3),
            name='Limite Observée'
        ))
        
        fig.update_layout(
            title=f"Limites d'Exclusion à {confidence_level}% CL",
            xaxis_title="Masse (GeV/c²)",
            yaxis_title="Section Efficace (pb)",
            yaxis_type="log",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📄 Publications et Résultats")
        
        st.write("### 📚 Publications Scientifiques")
        
        # Publications simulées
        publications = [
            {
                "title": "Observation of a New Resonance at 750 GeV",
                "authors": "LHC Collaboration",
                "journal": "Physics Letters B",
                "year": 2024,
                "citations": 1250,
                "impact": "High",
                "status": "Published"
            },
            {
                "title": "Precision Measurement of the Higgs Boson Mass",
                "authors": "Atlas & CMS Collaborations",
                "journal": "Physical Review D",
                "year": 2024,
                "citations": 850,
                "impact": "Very High",
                "status": "Published"
            },
            {
                "title": "Search for Supersymmetry in Multi-Jet Events",
                "authors": "CMS Collaboration",
                "journal": "Journal of High Energy Physics",
                "year": 2024,
                "citations": 420,
                "impact": "Medium",
                "status": "Published"
            },
            {
                "title": "Dark Matter Candidates from Heavy Ion Collisions",
                "authors": "ALICE Collaboration",
                "journal": "Nature Physics",
                "year": 2024,
                "citations": 0,
                "impact": "Pending",
                "status": "Submitted"
            }
        ]
        
        for pub in publications:
            with st.expander(f"📄 {pub['title']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Auteurs:** {pub['authors']}")
                    st.write(f"**Journal:** {pub['journal']}")
                    st.write(f"**Année:** {pub['year']}")
                
                with col2:
                    status_color = "🟢" if pub['status'] == "Published" else "🟡"
                    st.write(f"**Statut:** {status_color} {pub['status']}")
                    st.metric("Citations", pub['citations'])
                    
                    impact_colors = {
                        "Very High": "🔴",
                        "High": "🟠",
                        "Medium": "🟡",
                        "Pending": "⚪"
                    }
                    st.write(f"**Impact:** {impact_colors.get(pub['impact'], '⚪')} {pub['impact']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Télécharger PDF", key=f"dl_{pub['title'][:20]}"):
                        st.info("Téléchargement simulé")
                
                with col2:
                    if st.button("🔗 DOI", key=f"doi_{pub['title'][:20]}"):
                        st.info("Lien DOI")
                
                with col3:
                    if st.button("📊 Données", key=f"data_{pub['title'][:20]}"):
                        st.info("Données ouvertes")
        
        st.markdown("---")
        
        # Statistiques publications
        st.write("### 📊 Statistiques des Publications")
        
        pub_df = pd.DataFrame(publications)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Publications par an
            pub_by_year = pub_df.groupby('year').size()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=pub_by_year.index,
                y=pub_by_year.values,
                marker_color='rgba(102, 126, 234, 0.7)'
            ))
            fig.update_layout(
                title="Publications par Année",
                xaxis_title="Année",
                yaxis_title="Nombre de Publications"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Citations
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[p['title'][:30] for p in publications],
                y=[p['citations'] for p in publications],
                marker_color='rgba(118, 75, 162, 0.7)'
            ))
            fig.update_layout(
                title="Citations par Publication",
                xaxis_title="Publication",
                yaxis_title="Citations",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: PHYSIQUE DES PARTICULES ====================

elif page == "📐 Physique des Particules":
    st.header("📐 Physique des Particules - Théorie et Calculs")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ Modèle Standard", "🔬 Cinématique", "📊 Sections Efficaces", "🌌 Au-delà du MS"])
    
    with tab1:
        st.subheader("⚛️ Le Modèle Standard des Particules")
        
        st.write("""
        Le Modèle Standard décrit les particules élémentaires et leurs interactions 
        via trois forces fondamentales: électromagnétique, faible et forte.
        """)
        
        # Tableau des particules
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🔴 Fermions (Matière)")
            
            st.write("**Quarks:**")
            quarks = pd.DataFrame({
                'Particule': ['Up (u)', 'Down (d)', 'Charm (c)', 'Strange (s)', 'Top (t)', 'Bottom (b)'],
                'Charge': ['+2/3', '-1/3', '+2/3', '-1/3', '+2/3', '-1/3'],
                'Masse': ['2.2 MeV', '4.7 MeV', '1.28 GeV', '96 MeV', '173 GeV', '4.18 GeV'],
                'Génération': [1, 1, 2, 2, 3, 3]
            })
            st.dataframe(quarks, use_container_width=True)
            
            st.write("**Leptons:**")
            leptons = pd.DataFrame({
                'Particule': ['Electron (e)', 'Neutrino e (νₑ)', 'Muon (μ)', 'Neutrino μ (νμ)', 'Tau (τ)', 'Neutrino τ (ντ)'],
                'Charge': ['-1', '0', '-1', '0', '-1', '0'],
                'Masse': ['0.511 MeV', '< 2 eV', '105.7 MeV', '< 0.19 MeV', '1.777 GeV', '< 18.2 MeV'],
                'Génération': [1, 1, 2, 2, 3, 3]
            })
            st.dataframe(leptons, use_container_width=True)
        
        with col2:
            st.write("### 🔵 Bosons (Forces)")
            
            bosons = pd.DataFrame({
                'Particule': ['Photon (γ)', 'Gluon (g)', 'W⁺/W⁻', 'Z⁰', 'Higgs (H)'],
                'Charge': ['0', '0', '±1', '0', '0'],
                'Masse': ['0', '0', '80.4 GeV', '91.2 GeV', '125.1 GeV'],
                'Force': ['EM', 'Forte', 'Faible', 'Faible', '-'],
                'Spin': ['1', '1', '1', '1', '0']
            })
            st.dataframe(bosons, use_container_width=True)
        
        st.markdown("---")
        
        # Visualisation interactive
        st.write("### 🎨 Visualisation Interactive du Modèle Standard")
        
        particle_category = st.selectbox(
            "Catégorie",
            ["Tous", "Quarks", "Leptons", "Bosons de Jauge", "Boson de Higgs"]
        )
        # Données pour visualisation
        particles_viz = {
            'name': ['u', 'd', 'c', 's', 't', 'b', 'e', 'νe', 'μ', 'νμ', 'τ', 'ντ', 'γ', 'g', 'W', 'Z', 'H'],
            'type': ['quark']*6 + ['lepton']*6 + ['boson']*5,
            'mass': [2.2, 4.7, 1280, 96, 173000, 4180, 0.511, 0.000002, 105.7, 0.00019, 1777, 0.0182, 0, 0, 80400, 91200, 125100],
            'charge': [2/3, -1/3, 2/3, -1/3, 2/3, -1/3, -1, 0, -1, 0, -1, 0, 0, 0, 1, 0, 0],
            'generation': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0]
        }
        
        df_particles = pd.DataFrame(particles_viz)
        
        # Filtrer selon catégorie
        if particle_category == "Quarks":
            df_filtered = df_particles[df_particles['type'] == 'quark']
        elif particle_category == "Leptons":
            df_filtered = df_particles[df_particles['type'] == 'lepton']
        elif particle_category == "Bosons de Jauge":
            df_filtered = df_particles[(df_particles['type'] == 'boson') & (df_particles['name'] != 'H')]
        elif particle_category == "Boson de Higgs":
            df_filtered = df_particles[df_particles['name'] == 'H']
        else:
            df_filtered = df_particles
        
        # Graphique interactif
        fig = go.Figure()
        
        for ptype in df_filtered['type'].unique():
            df_type = df_filtered[df_filtered['type'] == ptype]
            
            fig.add_trace(go.Scatter(
                x=df_type['generation'],
                y=np.log10(df_type['mass'] + 0.001),  # +0.001 pour éviter log(0)
                mode='markers+text',
                marker=dict(
                    size=20 + np.abs(df_type['charge']) * 20,
                    color=['red' if ptype == 'quark' else 'blue' if ptype == 'lepton' else 'green'][0]
                ),
                text=df_type['name'],
                textposition='top center',
                name=ptype.capitalize(),
                hovertemplate='<b>%{text}</b><br>Masse: %{customdata[0]:.3f} MeV<br>Charge: %{customdata[1]}<extra></extra>',
                customdata=df_type[['mass', 'charge']].values
            ))
        
        fig.update_layout(
            title="Particules du Modèle Standard",
            xaxis_title="Génération",
            yaxis_title="log₁₀(Masse) [MeV]",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Cinématique Relativiste")
        
        st.write("### ⚡ Calculateur de Paramètres Relativistes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            particle_calc = st.selectbox(
                "Particule",
                list(PARTICLE_DATA.keys()),
                format_func=lambda x: f"{PARTICLE_DATA[x]['symbol']} {x.title()}"
            )
            
            energy_input = st.number_input("Énergie Cinétique (GeV)", 0.001, 100000.0, 1000.0)
            energy_joules = energy_input * 1e9 * constants.e
        
        with col2:
            particle_info = PARTICLE_DATA[particle_calc]
            
            st.write(f"**Particule:** {particle_info['symbol']}")
            st.write(f"**Masse:** {particle_info['mass']:.2e} kg")
            st.write(f"**Charge:** {particle_info['charge']:.2e} C")
        
        # Calculs
        rest_energy = particle_info['mass'] * constants.c**2
        total_energy = energy_joules + rest_energy
        
        gamma = total_energy / rest_energy
        beta = np.sqrt(1 - 1/gamma**2)
        velocity = beta * constants.c
        momentum = gamma * particle_info['mass'] * velocity
        
        # Énergie en unités pratiques
        rest_energy_mev = rest_energy / (constants.e * 1e6)
        total_energy_gev = total_energy / (constants.e * 1e9)
        
        st.markdown("---")
        
        st.write("### 📊 Résultats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("γ (Lorentz)", f"{gamma:.2f}")
            st.metric("β (v/c)", f"{beta:.6f}")
        
        with col2:
            st.metric("Vitesse", f"{velocity/constants.c:.6f} c")
            st.metric("", f"{velocity:.2e} m/s")
        
        with col3:
            st.metric("Momentum", f"{momentum:.2e} kg·m/s")
            st.metric("", f"{momentum/(constants.e*1e9):.2f} GeV/c")
        
        with col4:
            st.metric("Énergie Totale", f"{total_energy_gev:.2f} GeV")
            st.metric("Énergie Repos", f"{rest_energy_mev:.2f} MeV")
        
        # Formules
        st.markdown("---")
        st.write("### 📐 Formules Utilisées")
        
        st.latex(r"E_{totale} = \gamma m c^2")
        st.latex(r"\gamma = \frac{1}{\sqrt{1 - \beta^2}} = \frac{E_{totale}}{m c^2}")
        st.latex(r"\beta = \frac{v}{c} = \sqrt{1 - \frac{1}{\gamma^2}}")
        st.latex(r"p = \gamma m v")
        
        # Graphique β vs γ
        st.markdown("---")
        st.write("### 📈 Relation β-γ")
        
        gamma_range = np.logspace(0, 4, 1000)
        beta_range = np.sqrt(1 - 1/gamma_range**2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gamma_range, y=beta_range, mode='lines',
                                line=dict(color='blue', width=3)))
        
        # Point actuel
        fig.add_trace(go.Scatter(x=[gamma], y=[beta], mode='markers',
                                marker=dict(size=15, color='red'),
                                name='Point Actuel'))
        
        fig.update_layout(
            title="Relation entre β et γ",
            xaxis_title="γ (Facteur de Lorentz)",
            yaxis_title="β = v/c",
            xaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Longueur de contraction
        st.write("### 📏 Effets Relativistes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Contraction des Longueurs:**")
            L0 = st.number_input("Longueur au Repos (m)", 0.001, 1000.0, 1.0)
            L = L0 / gamma
            st.success(f"Longueur Contractée: {L:.6f} m")
            st.latex(r"L = \frac{L_0}{\gamma}")
        
        with col2:
            st.write("**Dilatation du Temps:**")
            t0 = st.number_input("Temps Propre (μs)", 0.001, 1000.0, 1.0)
            t = t0 * gamma
            st.success(f"Temps Dilaté: {t:.6f} μs")
            st.latex(r"t = \gamma t_0")
    
    with tab3:
        st.subheader("📊 Sections Efficaces")
        
        st.write("""
        La section efficace (σ) quantifie la probabilité d'interaction entre particules.
        Elle s'exprime généralement en barns (1 barn = 10⁻²⁴ cm²).
        """)
        
        # Calculateur de section efficace
        st.write("### 🎯 Calculateur de Section Efficace")
        
        col1, col2 = st.columns(2)
        
        with col1:
            process_type = st.selectbox(
                "Type de Processus",
                ["Diffusion Élastique", "Diffusion Inélastique", "Production de Paires", 
                 "Annihilation", "Bremsstrahlung", "Compton"]
            )
            
            beam_energy_sigma = st.number_input("Énergie Faisceau (GeV)", 0.1, 100000.0, 1000.0, key="sigma_e")
        
        with col2:
            target_particle = st.selectbox("Particule Cible", ["Proton", "Électron", "Noyau"])
            
            st.write("**Paramètres:**")
            st.write(f"Énergie CM: √s = {np.sqrt(2 * beam_energy_sigma * 0.938):.2f} GeV")  # Approximation
        
        # Calcul de section efficace (formules simplifiées)
        alpha = 1/137  # Constante de structure fine
        
        if process_type == "Diffusion Élastique":
            # Formule de Rutherford modifiée
            sigma = 0.1 / (beam_energy_sigma**2)  # mb
        elif process_type == "Production de Paires":
            sigma = 0.01 * np.log(beam_energy_sigma) if beam_energy_sigma > 1 else 0
        else:
            sigma = 1.0 / beam_energy_sigma
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Section Efficace", f"{sigma:.4f} mb")
        with col2:
            st.metric("", f"{sigma*1e-27:.2e} cm²")
        with col3:
            st.metric("", f"{sigma*1000:.2f} μb")
        
        # Graphique σ vs Énergie
        st.markdown("---")
        st.write("### 📈 Section Efficace vs Énergie")
        
        energies = np.logspace(-1, 5, 200)
        
        if process_type == "Diffusion Élastique":
            sigmas = 0.1 / (energies**2)
        elif process_type == "Production de Paires":
            sigmas = 0.01 * np.log(energies + 1)
        else:
            sigmas = 1.0 / energies
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=energies, y=sigmas, mode='lines',
                                line=dict(color='blue', width=3)))
        
        fig.add_vline(x=beam_energy_sigma, line_dash="dash", line_color="red",
                     annotation_text=f"E = {beam_energy_sigma:.0f} GeV")
        
        fig.update_layout(
            title=f"Section Efficace: {process_type}",
            xaxis_title="Énergie (GeV)",
            yaxis_title="σ (mb)",
            xaxis_type="log",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Luminosité et taux d'événements
        st.write("### 📊 Calcul du Taux d'Événements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            luminosity = st.number_input("Luminosité (cm⁻²s⁻¹)", 1e30, 1e38, 1e34, format="%.2e")
            sigma_process = st.number_input("Section Efficace (pb)", 0.001, 1e6, 100.0)
        
        with col2:
            # Conversion pb → cm²
            sigma_cm2 = sigma_process * 1e-36
            
            # Taux d'événements
            rate = luminosity * sigma_cm2  # événements/s
            rate_per_day = rate * 86400
            
            st.metric("Taux d'Événements", f"{rate:.2e} Hz")
            st.metric("Événements/Jour", f"{rate_per_day:.2e}")
            st.metric("Événements/An", f"{rate_per_day * 365:.2e}")
        
        st.info(f"""
        **Formule:** R = L × σ
        
        Avec une luminosité de {luminosity:.2e} cm⁻²s⁻¹ et une section efficace de {sigma_process} pb,
        on attend environ **{rate:.1f} événements par seconde**.
        """)
    
    with tab4:
        st.subheader("🌌 Au-delà du Modèle Standard")
        
        st.write("""
        Le Modèle Standard, bien que très réussi, ne peut pas expliquer plusieurs phénomènes:
        - La matière noire
        - L'asymétrie matière-antimatière
        - La masse des neutrinos
        - La gravité quantique
        """)
        
        # Théories BSM
        st.write("### 🔭 Théories Beyond Standard Model (BSM)")
        
        bsm_theories = {
            "Supersymétrie (SUSY)": {
                "description": "Chaque particule du MS a un superpartenaire",
                "predictions": ["Neutralinos (matière noire)", "Squarks et sleptons", "Unification des forces"],
                "status": "Non observée (limite > 1 TeV)",
                "search": "Jets + énergie transverse manquante"
            },
            "Dimensions Extra": {
                "description": "L'espace-temps a plus de 4 dimensions",
                "predictions": ["Tours de Kaluza-Klein", "Gravitons", "Mini trous noirs"],
                "status": "Non observée (limite > 5 TeV)",
                "search": "Résonances dans les paires de jets/leptons"
            },
            "Compositeness": {
                "description": "Les quarks et leptons ne sont pas élémentaires",
                "predictions": ["Preons", "Leptoquarks", "Nouvelles interactions"],
                "status": "Non observée (limite > 10 TeV)",
                "search": "Excès dans les distributions angulaires"
            },
            "Matière Noire": {
                "description": "Nouvelle particule stable et non-chargée",
                "predictions": ["WIMPs", "Axions", "Particules de Majorana"],
                "status": "Évidence cosmologique mais non détectée directement",
                "search": "Énergie transverse manquante, diffusion directe"
            }
        }
        
        for theory, info in bsm_theories.items():
            with st.expander(f"🌟 {theory}"):
                st.write(f"**Description:** {info['description']}")
                
                st.write("**Prédictions:**")
                for pred in info['predictions']:
                    st.write(f"• {pred}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Statut:** {info['status']}")
                with col2:
                    st.info(f"**Recherche:** {info['search']}")
        
        st.markdown("---")
        
        # Calculateur de masse SUSY
        st.write("### 🎯 Calculateur de Masse SUSY")
        
        st.write("Estimation de la masse des superpartenaires dans un modèle SUSY simplifié")
        
        col1, col2 = st.columns(2)
        
        with col1:
            m_gluino = st.slider("Masse Gluino (GeV)", 500, 3000, 1500, 50)
            m_squark = st.slider("Masse Squark (GeV)", 500, 3000, 1200, 50)
        
        with col2:
            m_neutralino = m_gluino * 0.3  # Relation simplifiée
            m_chargino = m_gluino * 0.4
            
            st.metric("Neutralino (LSP)", f"{m_neutralino:.0f} GeV")
            st.metric("Chargino", f"{m_chargino:.0f} GeV")
        
        # Graphique de spectre de masse
        particles_susy = ['Gluino', 'Squark', 'Chargino', 'Neutralino']
        masses_susy = [m_gluino, m_squark, m_chargino, m_neutralino]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=particles_susy,
            y=masses_susy,
            marker_color=['red', 'blue', 'green', 'orange'],
            text=[f"{m:.0f} GeV" for m in masses_susy],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Spectre de Masse SUSY Simplifié",
            xaxis_title="Particule",
            yaxis_title="Masse (GeV)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Canaux de désintégration
        st.markdown("---")
        st.write("### 🔀 Canaux de Désintégration")
        
        decay_channel = st.selectbox(
            "Sélectionner un canal",
            ["g̃ → qq̃", "q̃ → qχ̃₁⁰", "χ̃₁± → W±χ̃₁⁰", "H → ZZ → 4l"]
        )
        
        if decay_channel == "g̃ → qq̃":
            st.write("**Gluino → quark + squark**")
            st.latex(r"\tilde{g} \rightarrow q + \tilde{q}")
            st.info(f"Masse Mère: {m_gluino} GeV → Masse Fille: {m_squark} GeV")
            
        elif decay_channel == "q̃ → qχ̃₁⁰":
            st.write("**Squark → quark + neutralino**")
            st.latex(r"\tilde{q} \rightarrow q + \tilde{\chi}_1^0")
            st.info(f"Masse Mère: {m_squark} GeV → Masse Fille: {m_neutralino} GeV")
            
        elif decay_channel == "χ̃₁± → W±χ̃₁⁰":
            st.write("**Chargino → W boson + neutralino**")
            st.latex(r"\tilde{\chi}_1^{\pm} \rightarrow W^{\pm} + \tilde{\chi}_1^0")
            st.info(f"Masse Mère: {m_chargino} GeV → Masses Filles: 80.4 GeV (W), {m_neutralino} GeV (χ̃₁⁰)")

# Continuer avec les autres pages...

# ==================== PAGE: CONCEPTION OPTIQUE ====================

elif page == "🎯 Conception Optique":
    st.header("🎯 Conception Optique du Faisceau")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📐 Optique Linéaire", "🌀 Fonctions de Twiss", "🔧 Éléments Focalisants", "📊 Chromaticité"])
    
    with tab1:
        st.subheader("📐 Optique Linéaire du Faisceau")
        
        st.write("""
        L'optique du faisceau décrit comment les particules se propagent dans l'accélérateur.
        Les paramètres de Twiss (α, β, γ) caractérisent l'enveloppe du faisceau.
        """)
        
        # Paramètres initiaux
        st.write("### ⚙️ Paramètres Initiaux du Faisceau")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            beta_x_init = st.number_input("βₓ initial (m)", 0.1, 100.0, 10.0, 0.5)
            alpha_x_init = st.number_input("αₓ initial", -5.0, 5.0, 0.0, 0.1)
        
        with col2:
            beta_y_init = st.number_input("βᵧ initial (m)", 0.1, 100.0, 10.0, 0.5)
            alpha_y_init = st.number_input("αᵧ initial", -5.0, 5.0, 0.0, 0.1)
        
        with col3:
            emittance_x = st.number_input("εₓ (mm·mrad)", 0.001, 10.0, 1.0, 0.1)
            emittance_y = st.number_input("εᵧ (mm·mrad)", 0.001, 10.0, 1.0, 0.1)
        
        # Conversion en unités SI
        emittance_x_si = emittance_x * 1e-6  # m·rad
        emittance_y_si = emittance_y * 1e-6
        
        # Calcul gamma
        gamma_x = (1 + alpha_x_init**2) / beta_x_init
        gamma_y = (1 + alpha_y_init**2) / beta_y_init
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Plan Horizontal (x):**")
            st.metric("γₓ (m⁻¹)", f"{gamma_x:.4f}")
            beam_size_x = np.sqrt(emittance_x_si * beta_x_init) * 1e3  # mm
            st.metric("Taille Faisceau σₓ", f"{beam_size_x:.3f} mm")
            
        with col2:
            st.write("**Plan Vertical (y):**")
            st.metric("γᵧ (m⁻¹)", f"{gamma_y:.4f}")
            beam_size_y = np.sqrt(emittance_y_si * beta_y_init) * 1e3  # mm
            st.metric("Taille Faisceau σᵧ", f"{beam_size_y:.3f} mm")
        
        # Formules
        st.markdown("---")
        st.write("### 📐 Relations Fondamentales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.latex(r"\gamma\beta - \alpha^2 = 1")
            st.latex(r"\sigma = \sqrt{\epsilon \beta}")
            st.latex(r"\sigma' = \sqrt{\epsilon \gamma}")
        
        with col2:
            st.latex(r"\epsilon = \gamma x^2 + 2\alpha x x' + \beta x'^2")
            st.latex(r"A = \epsilon \pi \quad \text{(Aire ellipse)}")
        
        # Ellipse dans l'espace des phases
        st.markdown("---")
        st.write("### 🌀 Ellipse dans l'Espace des Phases")
        
        # Génération de l'ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Transformation de l'ellipse canonique
        x_ellipse = np.sqrt(emittance_x_si * beta_x_init) * np.cos(theta) * 1e3  # mm
        xp_ellipse = np.sqrt(emittance_x_si / beta_x_init) * (-alpha_x_init * np.cos(theta) + np.sin(theta)) * 1e3  # mrad
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_ellipse,
            y=xp_ellipse,
            mode='lines',
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.3)',
            line=dict(color='blue', width=3),
            name='Enveloppe'
        ))
        
        # Particules dans l'ellipse
        n_particles = 50
        theta_particles = np.random.uniform(0, 2*np.pi, n_particles)
        r_particles = np.random.uniform(0, 1, n_particles)
        
        x_particles = np.sqrt(r_particles * emittance_x_si * beta_x_init) * np.cos(theta_particles) * 1e3
        xp_particles = np.sqrt(r_particles * emittance_x_si / beta_x_init) * (-alpha_x_init * np.cos(theta_particles) + np.sin(theta_particles)) * 1e3
        
        fig.add_trace(go.Scatter(
            x=x_particles,
            y=xp_particles,
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Particules'
        ))
        
        fig.update_layout(
            title="Espace des Phases (x-x')",
            xaxis_title="x (mm)",
            yaxis_title="x' (mrad)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Paramètres de l'ellipse:**
        - Aire: A = π ε = {np.pi * emittance_x_si * 1e6:.3f} π mm·mrad
        - Demi-axe horizontal: {beam_size_x:.3f} mm
        - Demi-axe vertical: {np.sqrt(emittance_x_si * gamma_x) * 1e3:.3f} mrad
        """)
    
    with tab2:
        st.subheader("🌀 Fonctions de Twiss le Long de l'Accélérateur")
        
        if not st.session_state.accelerator_system['accelerators']:
            st.warning("⚠️ Aucun accélérateur disponible")
        else:
            acc_options = {a['id']: a['name'] for a in st.session_state.accelerator_system['accelerators'].values()}
            selected_acc = st.selectbox(
                "Sélectionner un accélérateur",
                options=list(acc_options.keys()),
                format_func=lambda x: acc_options[x],
                key="twiss_acc"
            )
            
            acc = st.session_state.accelerator_system['accelerators'][selected_acc]
            
            st.write(f"### ⚛️ {acc['name']}")
            
            # Configuration
            col1, col2 = st.columns(2)
            
            with col1:
                n_cells = st.slider("Nombre de Cellules FODO", 1, 100, 20)
                cell_length = st.number_input("Longueur Cellule (m)", 1.0, 100.0, 10.0)
            
            with col2:
                focal_length = st.number_input("Longueur Focale (m)", 1.0, 50.0, 5.0)
                phase_advance = st.slider("Avance de Phase (degrés)", 10, 180, 90)
            
            # Simulation des fonctions de Twiss
            s = np.linspace(0, n_cells * cell_length, 1000)
            
            # Fonctions β périodiques (simplifiées)
            k = 2 * np.pi / cell_length
            beta_max = focal_length
            beta_min = focal_length / 2
            
            beta_x = beta_min + (beta_max - beta_min) * (1 + np.cos(k * s)) / 2
            beta_y = beta_max - (beta_max - beta_min) * (1 + np.cos(k * s)) / 2
            
            # Dispersion
            dispersion = 0.5 * np.sin(k * s / 2)
            
            # Graphiques
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Fonctions β", "Fonction de Dispersion"),
                vertical_spacing=0.12
            )
            
            fig.add_trace(
                go.Scatter(x=s, y=beta_x, mode='lines', name='βₓ',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=s, y=beta_y, mode='lines', name='βᵧ',
                          line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=s, y=dispersion, mode='lines', name='Dₓ',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Position s (m)", row=2, col=1)
            fig.update_yaxes(title_text="β (m)", row=1, col=1)
            fig.update_yaxes(title_text="D (m)", row=2, col=1)
            
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("βₓ max", f"{beta_x.max():.2f} m")
            with col2:
                st.metric("βₓ min", f"{beta_x.min():.2f} m")
            with col3:
                st.metric("βᵧ max", f"{beta_y.max():.2f} m")
            with col4:
                st.metric("βᵧ min", f"{beta_y.min():.2f} m")
    
    with tab3:
        st.subheader("🔧 Éléments Focalisants")
        
        st.write("### 🧲 Types d'Éléments Magnétiques")
        
        magnet_types = {
            "Quadrupôle": {
                "fonction": "Focalisation dans un plan, défocalisation dans l'autre",
                "gradient": "B = G × x (linéaire)",
                "application": "Lattice FODO, Doublets, Triplets",
                "force": "F = q v B = q v G x"
            },
            "Sextupôle": {
                "fonction": "Correction de la chromaticité",
                "gradient": "B = S × x²",
                "application": "Correction chromatique, aberrations",
                "force": "Dépend de x²"
            },
            "Octupôle": {
                "fonction": "Correction des aberrations d'ordre supérieur",
                "gradient": "B = O × x³",
                "application": "Tune spread, stabilité",
                "force": "Dépend de x³"
            },
            "Solénoïde": {
                "fonction": "Focalisation azimutale",
                "gradient": "B = B₀ (uniforme)",
                "application": "Injection, faisceaux à basse énergie",
                "force": "Force de Lorentz circulaire"
            }
        }
        
        for mag_name, mag_info in magnet_types.items():
            with st.expander(f"🧲 {mag_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Fonction:** {mag_info['fonction']}")
                    st.write(f"**Gradient:** {mag_info['gradient']}")
                
                with col2:
                    st.write(f"**Application:** {mag_info['application']}")
                    st.write(f"**Force:** {mag_info['force']}")
        
        st.markdown("---")
        
        # Calculateur de quadrupôle
        st.write("### 🧮 Calculateur de Quadrupôle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quad_gradient = st.number_input("Gradient (T/m)", 0.1, 100.0, 10.0)
            quad_length = st.number_input("Longueur (m)", 0.1, 5.0, 0.5)
            aperture = st.number_input("Ouverture (cm)", 1.0, 20.0, 5.0)
        
        with col2:
            particle_energy = st.number_input("Énergie Particule (GeV)", 0.1, 10000.0, 1.0, key="quad_e")
            particle_quad = st.selectbox("Type Particule", ["electron", "proton"], key="quad_p")
        
        # Calcul
        particle_mass = PARTICLE_DATA[particle_quad]['mass']
        particle_charge = abs(PARTICLE_DATA[particle_quad]['charge'])
        
        energy_joules = particle_energy * 1e9 * constants.e
        momentum = np.sqrt(energy_joules**2 - (particle_mass * constants.c**2)**2) / constants.c
        
        # Rigidité magnétique
        B_rho = momentum / particle_charge
        
        # Force focalisante
        k = quad_gradient / B_rho  # m⁻²
        
        # Longueur focale
        focal_length_quad = 1 / (k * quad_length)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rigidité Magnétique", f"{B_rho:.2f} T·m")
        with col2:
            st.metric("Force Focalisante k", f"{k:.4f} m⁻²")
        with col3:
            st.metric("Longueur Focale", f"{abs(focal_length_quad):.2f} m")
        
        # Visualisation du champ
        st.markdown("---")
        st.write("### 🗺️ Carte du Champ Magnétique")
        
        x = np.linspace(-aperture/200, aperture/200, 100)
        y = np.linspace(-aperture/200, aperture/200, 100)
        X, Y = np.meshgrid(x, y)
        
        # Champ quadrupolaire: Bx = G*y, By = G*x
        Bx = quad_gradient * Y
        By = quad_gradient * X
        B_magnitude = np.sqrt(Bx**2 + By**2)
        
        fig = go.Figure(data=go.Contour(
            z=B_magnitude,
            x=x * 100,  # cm
            y=y * 100,
            colorscale='Viridis',
            colorbar=dict(title="B (T)")
        ))
        
        # Lignes de champ
        fig.add_trace(go.Scatter(
            x=X[::5, ::5].flatten() * 100,
            y=Y[::5, ::5].flatten() * 100,
            mode='markers',
            marker=dict(size=2, color='white'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Intensité du Champ Magnétique (Quadrupôle)",
            xaxis_title="x (cm)",
            yaxis_title="y (cm)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Chromaticité et Corrections")
        
        st.write("""
        La chromaticité naturelle provient de la dépendance en énergie de la focalisation.
        Elle doit être corrigée pour assurer la stabilité du faisceau.
        """)
        
        # Calcul de chromaticité
        st.write("### 🎯 Chromaticité Naturelle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tune_x = st.slider("Tune Horizontal Qₓ", 0.1, 20.0, 6.3, 0.1)
            tune_y = st.slider("Tune Vertical Qᵧ", 0.1, 20.0, 6.2, 0.1)
        
        with col2:
            # Chromaticité naturelle (simplifiée)
            xi_x_natural = -tune_x / (2 * np.pi)
            xi_y_natural = -tune_y / (2 * np.pi)
            
            st.metric("ξₓ naturelle", f"{xi_x_natural:.2f}")
            st.metric("ξᵧ naturelle", f"{xi_y_natural:.2f}")
        
        st.info("""
        **Chromaticité naturelle:**
        - ξₓ,ᵧ = ΔQₓ,ᵧ / (Δp/p)
        - Généralement négative et proportionnelle au tune
        """)
        
        st.markdown("---")
        
        # Correction avec sextupôles
        st.write("### 🔧 Correction par Sextupôles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_sextupoles = st.number_input("Nombre de Sextupôles", 2, 100, 20)
            sextupole_strength = st.slider("Force Sextupôle (m⁻³)", -100.0, 100.0, 10.0)
        
        with col2:
            target_xi_x = st.slider("ξₓ cible", -5.0, 5.0, 0.0, 0.1)
            target_xi_y = st.slider("ξᵧ cible", -5.0, 5.0, 0.0, 0.1)
        
        # Correction calculée
        xi_x_corrected = xi_x_natural + sextupole_strength * 0.1
        xi_y_corrected = xi_y_natural + sextupole_strength * 0.08
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("ξₓ après correction", f"{xi_x_corrected:.2f}",
                     delta=f"{xi_x_corrected - xi_x_natural:+.2f}")
        with col2:
            st.metric("ξᵧ après correction", f"{xi_y_corrected:.2f}",
                     delta=f"{xi_y_corrected - xi_y_natural:+.2f}")
        
        # Graphique tune vs momentum
        st.markdown("---")
        st.write("### 📈 Tune vs Déviation en Momentum")
        
        dp_p = np.linspace(-0.01, 0.01, 100)  # Δp/p
        
        Q_x_natural = tune_x + xi_x_natural * dp_p
        Q_y_natural = tune_y + xi_y_natural * dp_p
        
        Q_x_corrected = tune_x + xi_x_corrected * dp_p
        Q_y_corrected = tune_y + xi_y_corrected * dp_p
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Qₓ vs Δp/p", "Qᵧ vs Δp/p"))
        
        # Qx
        fig.add_trace(
            go.Scatter(x=dp_p*100, y=Q_x_natural, mode='lines',
                      name='Naturel', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=dp_p*100, y=Q_x_corrected, mode='lines',
                      name='Corrigé', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Qy
        fig.add_trace(
            go.Scatter(x=dp_p*100, y=Q_y_natural, mode='lines',
                      name='Naturel', line=dict(color='red', dash='dash'), showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=dp_p*100, y=Q_y_corrected, mode='lines',
                      name='Corrigé', line=dict(color='blue', width=3), showlegend=False),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Δp/p (%)", row=1, col=1)
        fig.update_xaxes(title_text="Δp/p (%)", row=1, col=2)
        fig.update_yaxes(title_text="Qₓ", row=1, col=1)
        fig.update_yaxes(title_text="Qᵧ", row=1, col=2)
        
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: FABRICATION ====================

elif page == "🏭 Fabrication":
    st.header("🏭 Fabrication et Construction")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Plan de Fabrication", "🏗️ Construction", "🔧 Assemblage", "📊 Suivi"])
    
    with tab1:
        st.subheader("📋 Plan de Fabrication Détaillé")
        
        if not st.session_state.accelerator_system['accelerators']:
            st.warning("⚠️ Aucun accélérateur disponible")
        else:
            acc_options = {a['id']: a['name'] for a in st.session_state.accelerator_system['accelerators'].values()}
            selected_acc = st.selectbox(
                "Sélectionner un accélérateur à fabriquer",
                options=list(acc_options.keys()),
                format_func=lambda x: acc_options[x],
                key="fab_acc"
            )
            
            acc = st.session_state.accelerator_system['accelerators'][selected_acc]
            
            st.write(f"### ⚛️ {acc['name']}")
            
            # Phases de fabrication
            fabrication_phases = [
                {
                    'phase': 1,
                    'name': 'Études et Conception Détaillée',
                    'duration': 24,  # mois
                    'cost': acc['costs']['construction'] * 0.10,
                    'tasks': [
                        'Design détaillé de tous les composants',
                        'Simulations complètes',
                        'Optimisation du lattice',
                        'Études de sécurité',
                        'Approbations réglementaires'
                    ]
                },
                {
                    'phase': 2,
                    'name': 'Génie Civil et Infrastructure',
                    'duration': 36,
                    'cost': acc['costs']['construction'] * 0.25,
                    'tasks': [
                        'Excavation du tunnel',
                        'Construction des bâtiments techniques',
                        'Infrastructure électrique (réseau HT)',
                        'Système de refroidissement',
                        'Systèmes de sécurité'
                    ]
                },
                {
                    'phase': 3,
                    'name': 'Fabrication des Aimants',
                    'duration': 30,
                    'cost': acc['costs']['construction'] * 0.20,
                    'tasks': [
                        f"Fabrication {acc['components']['magnets']} aimants dipolaires",
                        'Fabrication aimants quadrupolaires',
                        'Aimants correcteurs (sextupôles, octupôles)',
                        'Tests cryogéniques individuels',
                        'Mesures de champ magnétique'
                    ]
                },
                {
                    'phase': 4,
                    'name': 'Système Radiofréquence',
                    'duration': 24,
                    'cost': acc['costs']['construction'] * 0.15,
                    'tasks': [
                        f"Fabrication {acc['components']['rf_cavities']} cavités RF",
                        'Klystrons et amplificateurs',
                        'Systèmes de contrôle RF',
                        'Tests en puissance',
                        'Conditionnement des cavités'
                    ]
                },
                {
                    'phase': 5,
                    'name': 'Système de Vide',
                    'duration': 18,
                    'cost': acc['costs']['construction'] * 0.08,
                    'tasks': [
                        'Fabrication des chambres à vide',
                        'Pompes à vide (turbo, ioniques)',
                        'Vannes et instruments',
                        'Tests d\'étanchéité',
                        'Système de dégazage (bake-out)'
                    ]
                },
                {
                    'phase': 6,
                    'name': 'Détecteurs et Instrumentation',
                    'duration': 30,
                    'cost': acc['costs']['construction'] * 0.12,
                    'tasks': [
                        f"Fabrication {acc['components']['detectors']} détecteurs principaux",
                        'Électronique de lecture',
                        'Système d\'acquisition de données (DAQ)',
                        'Trigger et filtrage',
                        'Calibration complète'
                    ]
                },
                {
                    'phase': 7,
                    'name': 'Installation et Assemblage',
                    'duration': 24,
                    'cost': acc['costs']['construction'] * 0.05,
                    'tasks': [
                        'Installation des aimants dans le tunnel',
                        'Installation cavités RF',
                        'Raccordement cryogénique',
                        'Câblage et connexions',
                        'Alignement précis (± 0.1 mm)'
                    ]
                },
                {
                    'phase': 8,
                    'name': 'Commissioning et Tests',
                    'duration': 18,
                    'cost': acc['costs']['construction'] * 0.05,
                    'tasks': [
                        'Tests du système de vide',
                        'Tests cryogéniques',
                        'Premiers faisceaux (pilote)',
                        'Optimisation du faisceau',
                        'Validation complète'
                    ]
                }
            ]
            
            # Affichage des phases
            for phase in fabrication_phases:
                with st.expander(f"📌 Phase {phase['phase']}: {phase['name']} ({phase['duration']} mois)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Tâches Principales:**")
                        for task in phase['tasks']:
                            st.write(f"✓ {task}")
                    
                    with col2:
                        st.metric("Durée", f"{phase['duration']} mois")
                        st.metric("Coût", f"${phase['cost']/1e9:.2f}B")
                        st.metric("% Total", f"{phase['cost']/acc['costs']['construction']*100:.0f}%")
            
            # Résumé
            st.markdown("---")
            st.write("### 📊 Résumé Global")
            
            total_duration = sum(p['duration'] for p in fabrication_phases)
            total_cost = sum(p['cost'] for p in fabrication_phases)
            
            # Note: durée parallélisée
            critical_path_duration = max(36, 30, 24)  # Chemin critique
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Durée Séquentielle", f"{total_duration} mois")
            with col2:
                st.metric("Durée Optimisée", f"{critical_path_duration + 42} mois")
            with col3:
                st.metric("Coût Total", f"${total_cost/1e9:.2f}B")
            with col4:
                st.metric("Personnel", f"{int(total_cost/1e9 * 100)} personnes")
            
            # Diagramme de Gantt
            st.markdown("---")
            st.write("### 📅 Diagramme de Gantt")
            
            fig = go.Figure()
            
            current_start = 0
            for phase in fabrication_phases:
                fig.add_trace(go.Bar(
                    y=[phase['name']],
                    x=[phase['duration']],
                    orientation='h',
                    name=f"Phase {phase['phase']}",
                    text=f"{phase['duration']}m",
                    textposition='inside',
                    hovertemplate=f"<b>{phase['name']}</b><br>Durée: {phase['duration']} mois<br>Coût: ${phase['cost']/1e9:.2f}B<extra></extra>"
                ))
            
            fig.update_layout(
                title="Planning de Fabrication",
                xaxis_title="Durée (mois)",
                yaxis_title="Phase",
                height=600,
                barmode='stack',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🏗️ Construction - Génie Civil")
        
        st.write("### 🏗️ Infrastructure Principale")
        
        construction_elements = {
            "Tunnel": {
                "description": "Tunnel circulaire ou linéaire pour l'accélérateur",
                "specs": f"Longueur: {acc['geometry']['length']:.0f} m, Diamètre: 3-5 m",
                "challenges": "Excavation, stabilité, drainage, ventilation",
                "duration": "24-36 mois",
                "cost": acc['costs']['construction'] * 0.15
            },
            "Puits d'Accès": {
                "description": "Puits verticaux pour accéder au tunnel",
                "specs": "8-12 puits de 50-100m de profondeur",
                "challenges": "Forage, étanchéité, ascenseurs",
                "duration": "12-18 mois",
                "cost": acc['costs']['construction'] * 0.03
            },
            "Bâtiments de Surface": {
                "description": "Centres de contrôle, alimentation électrique, etc.",
                "specs": "10,000-50,000 m² de surface",
                "challenges": "Intégration, sismique, sécurité",
                "duration": "18-24 mois",
                "cost": acc['costs']['construction'] * 0.05
            },
            "Alimentation Électrique": {
                "description": "Sous-stations électriques haute tension",
                "specs": f"{acc['costs']['energy_consumption']} MWh/an",
                "challenges": "Puissance crête, stabilité, redondance",
                "duration": "12-18 mois",
                "cost": acc['costs']['construction'] * 0.04
            },
            "Refroidissement": {
                "description": "Tours de refroidissement et distribution",
                "specs": "50-200 MW de puissance thermique",
                "challenges": "Capacité, efficacité, environnement",
                "duration": "12-15 mois",
                "cost": acc['costs']['construction'] * 0.03
            }
        }
        
        for elem_name, elem_info in construction_elements.items():
            with st.expander(f"🏗️ {elem_name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {elem_info['description']}")
                    st.write(f"**Spécifications:** {elem_info['specs']}")
                    st.write(f"**Défis:** {elem_info['challenges']}")
                
                with col2:
                    st.metric("Durée", elem_info['duration'])
                    st.metric("Coût", f"${elem_info['cost']/1e6:.0f}M")
        
        st.markdown("---")
        
        # Coupe transversale du tunnel
        st.write("### 🔍 Coupe Transversale du Tunnel")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dessin simplifié du tunnel
            fig = go.Figure()
            
            # Tunnel principal
            theta = np.linspace(0, 2*np.pi, 100)
            tunnel_x = 2.5 * np.cos(theta)
            tunnel_y = 2.5 * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=tunnel_x, y=tunnel_y,
                fill='toself',
                fillcolor='rgba(200, 200, 200, 0.3)',
                line=dict(color='gray', width=3),
                name='Tunnel'
            ))
            
            # Tube à vide
            vide_x = 0.05 * np.cos(theta)
            vide_y = 0.05 * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=vide_x, y=vide_y,
                fill='toself',
                fillcolor='rgba(100, 100, 255, 0.3)',
                line=dict(color='blue', width=2),
                name='Chambre à Vide'
            ))
            
            # Aimants (rectangles simplifiés)
            fig.add_shape(type="rect", x0=-2, y0=-0.3, x1=-1.5, y1=0.3,
                         fillcolor='red', opacity=0.5, line=dict(color='red'))
            fig.add_shape(type="rect", x0=1.5, y0=-0.3, x1=2, y1=0.3,
                         fillcolor='red', opacity=0.5, line=dict(color='red'))
            
            fig.update_layout(
                title="Coupe Transversale du Tunnel",
                xaxis=dict(scaleanchor="y", scaleratio=1, range=[-3, 3]),
                yaxis=dict(range=[-3, 3]),
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Éléments:**")
            st.info("🔵 Chambre à Vide\n(Ø 10 cm)")
            st.info("🔴 Aimants Dipôles\n(0.5 x 0.6 m)")
            st.info("⚪ Espace Technique\n(câbles, cryogénie)")
            st.info("🟤 Tunnel\n(Ø 5 m)")
    
    with tab3:
        st.subheader("🔧 Assemblage des Composants")
        
        st.write("### ⚙️ Procédures d'Assemblage")
        
        assembly_procedures = [
            {
                "step": 1,
                "name": "Préparation du Site",
                "description": "Nettoyage classe 10000, contrôle température/humidité",
                "duration": "1 semaine",
                "personnel": 10,
                "risk": "Faible"
            },
            {
                "step": 2,
                "name": "Installation Rails de Support",
                "description": "Installation et alignement des rails (précision ±0.1mm)",
                "duration": "2 semaines",
                "personnel": 15,
                "risk": "Moyen"
            },
            {
                "step": 3,
                "name": "Transport et Positionnement Aimants",
                "description": "Transport des aimants (jusqu'à 15 tonnes) et positionnement",
                "duration": "4 semaines",
                "personnel": 25,
                "risk": "Élevé"
            },
            {
                "step": 4,
                "name": "Connexions Cryogéniques",
                "description": "Raccordement du système de refroidissement",
                "duration": "3 semaines",
                "personnel": 20,
                "risk": "Moyen"
            },
            {
                "step": 5,
                "name": "Installation Chambres à Vide",
                "description": "Installation et connexion des sections de vide",
                "duration": "3 semaines",
                "personnel": 18,
                "risk": "Moyen"
            },
            {
                "step": 6,
                "name": "Câblage Électrique",
                "description": "Câblage puissance et instrumentation",
                "duration": "4 semaines",
                "personnel": 30,
                "risk": "Faible"
            },
            {
                "step": 7,
                "name": "Installation Cavités RF",
                "description": "Installation et alignement des cavités RF",
                "duration": "2 semaines",
                "personnel": 12,
                "risk": "Moyen"
            },
            {
                "step": 8,
                "name": "Alignement Précis",
                "description": "Alignement laser de tous les éléments (±0.1mm)",
                "duration": "3 semaines",
                "personnel": 8,
                "risk": "Élevé"
            },
            {
                "step": 9,
                "name": "Tests Système par Système",
                "description": "Tests individuels de chaque sous-système",
                "duration": "4 semaines",
                "personnel": 25,
                "risk": "Moyen"
            },
            {
                "step": 10,
                "name": "Tests Intégrés",
                "description": "Tests du système complet",
                "duration": "2 semaines",
                "personnel": 30,
                "risk": "Faible"
            }
        ]
        
        for proc in assembly_procedures:
            risk_colors = {"Faible": "🟢", "Moyen": "🟡", "Élevé": "🔴"}
            risk_icon = risk_colors.get(proc['risk'], "⚪")
            
            with st.expander(f"Étape {proc['step']}: {proc['name']} {risk_icon}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**Description:**")
                    st.write(proc['description'])
                
                with col2:
                    st.metric("Durée", proc['duration'])
                with col3:
                    st.metric("Personnel", proc['personnel'])
                with col4:
                    st.metric("Risque", proc['risk'])
        
        st.markdown("---")
        
        # Checklist d'assemblage
        st.write("### ✅ Checklist d'Assemblage")
        
        checklist_items = [
            "Vérification de l'alignement des rails",
            "Test de charge des ponts roulants",
            "Inspection visuelle de tous les aimants",
            "Tests électriques des bobines",
            "Vérification des connexions cryogéniques",
            "Tests d'étanchéité du système de vide",
            "Mesure de la résistance des câbles",
            "Calibration des instruments de mesure",
            "Test des interlocks de sécurité",
            "Documentation photographique complète"
        ]
        
        completed = 0
        for i, item in enumerate(checklist_items):
            if st.checkbox(item, key=f"check_assembly_{i}"):
                completed += 1
        
        progress = completed / len(checklist_items)
        st.progress(progress)
        st.write(f"**Progression:** {completed}/{len(checklist_items)} ({progress*100:.0f}%)")
        
        if progress == 1.0:
            st.success("🎉 Assemblage terminé! Prêt pour les tests.")
            st.balloons()
    
    with tab4:
        st.subheader("📊 Suivi de Fabrication")
        
        st.write("### 📈 Tableau de Bord Production")
        
        # Métriques de production simulées
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            magnets_produced = int(acc['components']['magnets'] * 0.75)
            st.metric("Aimants Produits", 
                     f"{magnets_produced}/{acc['components']['magnets']}",
                     delta=f"{(magnets_produced/acc['components']['magnets']*100):.0f}%")
        
        with col2:
            rf_produced = int(acc['components']['rf_cavities'] * 0.60)
            st.metric("Cavités RF Produites", 
                     f"{rf_produced}/{acc['components']['rf_cavities']}",
                     delta=f"{(rf_produced/acc['components']['rf_cavities']*100):.0f}%")
        
        with col3:
            detectors_produced = int(acc['components']['detectors'] * 0.40)
            st.metric("Détecteurs Produits", 
                     f"{detectors_produced}/{acc['components']['detectors']}",
                     delta=f"{(detectors_produced/acc['components']['detectors']*100):.0f}%")
        
        with col4:
            overall_progress = (magnets_produced/acc['components']['magnets'] * 0.5 + 
                               rf_produced/acc['components']['rf_cavities'] * 0.3 +
                               detectors_produced/acc['components']['detectors'] * 0.2)
            st.metric("Progression Globale", f"{overall_progress*100:.0f}%")
        
        st.markdown("---")
        
        # Graphique de progression
        st.write("### 📊 Progression par Composant")
        
        components = ['Aimants Dipôles', 'Aimants Quadrupôles', 'Cavités RF', 
                     'Chambres à Vide', 'Détecteurs', 'Électronique']
        progress_values = [75, 80, 60, 85, 40, 70]
        target_values = [100] * len(components)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=components,
            x=progress_values,
            orientation='h',
            name='Actuel',
            marker_color='rgba(102, 126, 234, 0.7)',
            text=[f"{v}%" for v in progress_values],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            y=components,
            x=[100-v for v in progress_values],
            orientation='h',
            name='Restant',
            marker_color='rgba(200, 200, 200, 0.3)',
            showlegend=False
        ))
        
        fig.update_layout(
            title="État d'Avancement de la Production",
            xaxis_title="Progression (%)",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Planning vs Réalisé
        st.write("### 📅 Planning vs Réalisé")
        
        months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        planned = [5, 10, 15, 22, 30, 38, 46, 54, 62, 70, 80, 90]
        actual = [4, 9, 16, 24, 32, 40, 48, 55, 63, 72, 0, 0]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=planned,
            mode='lines+markers',
            name='Planifié',
            line=dict(color='blue', dash='dash', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=months[:10],
            y=actual[:10],
            mode='lines+markers',
            name='Réalisé',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="Courbe en S - Progression du Projet",
            xaxis_title="Mois",
            yaxis_title="Progression (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicateurs de qualité
        st.markdown("---")
        st.write("### 🎯 Indicateurs de Qualité")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taux de Défauts", "0.5%", delta="-0.2%", delta_color="inverse")
        with col2:
            st.metric("Taux de Rejet", "1.2%", delta="-0.5%", delta_color="inverse")
        with col3:
            st.metric("Tests Réussis", "98.5%", delta="+1.2%")
        with col4:
            st.metric("Conformité Spec", "99.2%", delta="+0.3%")

# ==================== PAGE: TESTS & CALIBRATION ====================

elif page == "🔧 Tests & Calibration":
    st.header("🔧 Tests et Calibration")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧪 Tests Composants", "⚡ Tests Système", "📐 Calibration", "📊 Résultats"])
    
    with tab1:
        st.subheader("🧪 Tests des Composants")
        
        st.write("### 🧲 Tests des Aimants")
        
        with st.expander("Test 1: Mesure du Champ Magnétique"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Protocole:**")
                st.write("1. Positionner la sonde Hall au centre de l'aimant")
                st.write("2. Alimenter l'aimant au courant nominal")
                st.write("3. Mesurer le champ en 3D (grille 10x10x10 cm)")
                st.write("4. Analyser l'uniformité et les harmoniques")
                
                if st.button("🚀 Lancer Test Champ", key="test_field"):
                    with st.spinner("Test en cours..."):
                        # Simulation
                        nominal_field = 8.33  # Tesla
                        measured_field = nominal_field * (1 + np.random.normal(0, 0.001))
                        uniformity = 99.95 + np.random.random() * 0.05
                        
                        st.success("✅ Test terminé!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Champ Nominal", f"{nominal_field:.3f} T")
                        with col2:
                            st.metric("Champ Mesuré", f"{measured_field:.3f} T")
                        with col3:
                            deviation = abs(measured_field - nominal_field) / nominal_field * 100
                            st.metric("Écart", f"{deviation:.4f}%")
                        
                        st.metric("Uniformité", f"{uniformity:.2f}%")
                        
                        if deviation < 0.01 and uniformity > 99.9:
                            st.success("✅ ACCEPTÉ - Aimant conforme aux spécifications")
                        else:
                            st.warning("⚠️ À REVOIR - Écart hors tolérance")
            
            with col2:
                # Carte de champ simulée
                x = np.linspace(-0.05, 0.05, 50)
                y = np.linspace(-0.05, 0.05, 50)
                X, Y = np.meshgrid(x, y)
                
                # Champ dipolaire avec petites imperfections
                B = 8.33 * (1 + 0.001*np.sin(10*X) * np.cos(10*Y))
                
                fig = go.Figure(data=go.Contour(
                    z=B,
                    x=x*100,
                    y=y*100,
                    colorscale='Viridis',
                    colorbar=dict(title="B (T)")
                ))
                
                fig.update_layout(
                    title="Carte de Champ Magnétique",
                    xaxis_title="x (cm)",
                    yaxis_title="y (cm)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Test 2: Test Cryogénique"):
            st.write("**Test de refroidissement à 4.2K**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_temp = st.number_input("Température Cible (K)", 1.8, 300.0, 4.2, 0.1)
                cooling_rate = st.slider("Taux de Refroidissement (K/min)", 0.1, 5.0, 1.0)
                
                if st.button("🚀 Démarrer Refroidissement", key="test_cryo"):
                    # Simulation de refroidissement
                    time = np.linspace(0, 300, 100)  # 5 minutes
                    temp = 300 * np.exp(-time/60) + target_temp
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=time,
                        y=temp,
                        mode='lines',
                        line=dict(color='blue', width=3)
                    ))
                    fig.add_hline(y=target_temp, line_dash="dash", line_color="red",
                                 annotation_text=f"Cible: {target_temp}K")
                    
                    fig.update_layout(
                        title="Courbe de Refroidissement",
                        xaxis_title="Temps (s)",
                        yaxis_title="Température (K)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Température de {target_temp}K atteinte en {time[-1]/60:.1f} min")
            
            with col2:
                st.write("**Critères d'Acceptation:**")
                st.info("✓ Température < 4.5K")
                st.info("✓ Stabilité ±0.1K")
                st.info("✓ Gradient < 0.5K/m")
                st.info("✓ Temps refroidissement < 6h")
                st.info("✓ Pas de quench durant le test")
        
        st.markdown("---")
        
        st.write("### 📡 Tests des Cavités RF")
        
        with st.expander("Test 3: Gradient Accélérateur"):
            col1, col2 = st.columns(2)
            
            with col1:
                rf_freq = st.number_input("Fréquence RF (MHz)", 100, 3000, 500)
                target_voltage = st.number_input("Tension Cible (MV)", 1, 50, 10)
                
                if st.button("🚀 Test RF", key="test_rf"):
                    power_levels = np.linspace(0, 100, 50)
                    voltage_achieved = target_voltage * power_levels / 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=power_levels,
                        y=voltage_achieved,
                        mode='lines+markers',
                        line=dict(color='purple', width=3)
                    ))
                    
                    fig.update_layout(
                        title="Voltage vs Puissance RF",
                        xaxis_title="Puissance (%)",
                        yaxis_title="Voltage (MV)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Gradient de {target_voltage} MV atteint")
            
            with col2:
                st.write("**Mesures:**")
                Q_factor = 1e10 * (1 + np.random.normal(0, 0.05))
                shunt_impedance = 100 * (1 + np.random.normal(0, 0.1))
                
                st.metric("Facteur Q", f"{Q_factor:.2e}")
                st.metric("Impédance Shunt", f"{shunt_impedance:.1f} MΩ")
                st.metric("Fréquence Résonance", f"{rf_freq:.2f} MHz")
    
    with tab2:
        st.subheader("⚡ Tests du Système Complet")
        
        st.write("### 🔄 Test d'Intégration")
        
        test_sequence = [
            {"name": "Vide", "status": "passed", "value": "1.2e-10 Pa", "spec": "< 1e-9 Pa"},
            {"name": "Refroidissement", "status": "passed", "value": "4.18 K", "spec": "< 4.5 K"},
            {"name": "Champs Magnétiques", "status": "passed", "value": "8.331 T", "spec": "8.33 ± 0.01 T"},
            {"name": "RF", "status": "passed", "value": "9.98 MV/m", "spec": "10 ± 0.1 MV/m"},
            {"name": "Alignement", "status": "warning", "value": "0.12 mm", "spec": "< 0.1 mm"},
            {"name": "Instrumentation", "status": "passed", "value": "OK", "spec": "OK"},
            {"name": "Interlocks", "status": "passed", "value": "OK", "spec": "OK"},
            {"name": "DAQ", "status": "passed", "value": "100 kHz", "spec": "> 10 kHz"}
        ]
        
        for test in test_sequence:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{test['name']}**")
            with col2:
                st.write(test['value'])
            with col3:
                st.write(f"Spec: {test['spec']}")
            with col4:
                if test['status'] == 'passed':
                    st.success("✅")
                elif test['status'] == 'warning':
                    st.warning("⚠️")
                else:
                    st.error("❌")
        
        st.markdown("---")
        
        # Test du premier faisceau
        st.write("### 🎯 Test du Premier Faisceau (Pilote)")
        
        st.info("""
        **Objectifs:**
        - Injecter un faisceau à basse intensité (10⁸ particules)
        - Vérifier que le faisceau fait au moins 1 tour complet
        - Mesurer la position et le profil du faisceau
        - Ajuster les paramètres optiques
        """)
        
        if st.button("🚀 Lancer Test Faisceau Pilote", use_container_width=True, type="primary"):
            with st.spinner("Injection et circulation du faisceau..."):
                progress_bar = st.progress(0)
                
                steps = [
                    "Injection du faisceau",
                    "Capture RF",
                    "Ramping magnétique",
                    "Premier tour complet",
                    "Stabilisation",
                    "Mesures BPM",
                    "Profil du faisceau",
                    "Analyse des pertes"
                ]
                
                for i, step in enumerate(steps):
                    progress_bar.progress((i + 1) / len(steps))
                    st.write(f"✓ {step}")
                
                progress_bar.empty()
                
                st.success("✅ Premier faisceau circulant avec succès!")
                st.balloons()
                
                # Résultats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Tours Complets", "1000+")
                with col2:
                    st.metric("Intensité", "8.5e8 p")
                with col3:
                    st.metric("Lifetime", "12.5 h")
                with col4:
                    st.metric("Émittance", "2.1 μm")
                
                # Profil du faisceau
                st.markdown("---")
                st.write("**📊 Profil du Faisceau au BPM-1:**")
                
                x = np.linspace(-5, 5, 100)
                profile = np.exp(-x**2 / (2 * 1.5**2))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x,
                    y=profile,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title="Profil Transverse du Faisceau",
                    xaxis_title="Position (mm)",
                    yaxis_title="Intensité (u.a.)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📐 Calibration des Systèmes")
        
        st.write("### 🎯 Calibration des BPM (Beam Position Monitors)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bpm_number = st.number_input("Numéro BPM", 1, 1000, 1)
            calibration_method = st.selectbox("Méthode", ["Référence Optique", "Faisceau à Offset", "Gradient de Quadrupôle"])
        
        with col2:
            n_measurements = st.slider("Nombre de Mesures", 10, 1000, 100)
            
            if st.button("🔧 Calibrer BPM"):
                # Simulation
                measurements = np.random.normal(0, 0.05, n_measurements)  # mm
                
                mean = np.mean(measurements)
                std = np.std(measurements)
                
                st.success(f"✅ BPM-{bpm_number} calibré")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Offset Moyen", f"{mean:.3f} mm")
                with col2:
                    st.metric("Résolution σ", f"{std:.3f} mm")
                with col3:
                    precision = std / np.sqrt(n_measurements)
                    st.metric("Précision", f"{precision:.4f} mm")
                
                # Histogramme
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=measurements,
                    nbinsx=30,
                    marker_color='rgba(102, 126, 234, 0.7)'
                ))
                
                fig.update_layout(
                    title=f"Distribution des Mesures BPM-{bpm_number}",
                    xaxis_title="Position (mm)",
                    yaxis_title="Fréquence",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Calibration des aimants
        st.write("### 🧲 Calibration des Aimants")
        
        with st.expander("Calibration Dipôle"):
            col1, col2 = st.columns(2)
            
            with col1:
                dipole_id = st.text_input("ID Dipôle", "D001")
                current_range = st.slider("Plage Courant (A)", 0, 10000, (0, 6000))
                n_points = st.slider("Points de Mesure", 5, 50, 10)
            
            with col2:
                if st.button("📊 Générer Courbe Calibration"):
                    currents = np.linspace(current_range[0], current_range[1], n_points)
                    
                    # B = k * I (relation linéaire idéale)
                    k = 0.00139  # T/A (exemple)
                    fields = k * currents * (1 + np.random.normal(0, 0.001, n_points))
                    
                    # Fit linéaire
                    from scipy.stats import linregress
                    slope, intercept, r_value, p_value, std_err = linregress(currents, fields)
                    
                    fig = go.Figure()
                    
                    # Points de mesure
                    fig.add_trace(go.Scatter(
                        x=currents,
                        y=fields,
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        name='Mesures'
                    ))
                    
                    # Fit
                    fit_line = slope * currents + intercept
                    fig.add_trace(go.Scatter(
                        x=currents,
                        y=fit_line,
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Fit Linéaire'
                    ))
                    
                    fig.update_layout(
                        title=f"Calibration {dipole_id}: B vs I",
                        xaxis_title="Courant (A)",
                        yaxis_title="Champ Magnétique (T)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("**Résultats:**")
                    st.write(f"- Pente: {slope:.6f} T/A")
                    st.write(f"- Intercept: {intercept:.6f} T")
                    st.write(f"- R²: {r_value**2:.6f}")
                    st.write(f"- Erreur std: {std_err:.6f} T/A")
        
        st.markdown("---")
        
        # Table de calibration
        st.write("### 📋 Table de Calibration Générale")
        
        calibration_data = {
            'Composant': ['BPM-001', 'BPM-002', 'Dipôle-D001', 'Quadrupôle-Q001', 'Cavité-RF-01'],
            'Type': ['BPM', 'BPM', 'Dipôle', 'Quadrupôle', 'Cavité RF'],
            'Paramètre': ['Position', 'Position', 'Champ B', 'Gradient', 'Voltage'],
            'Valeur Calibrée': ['0.025 mm', '0.031 mm', '8.331 T @ 6000A', '23.5 T/m', '10.1 MV'],
            'Incertitude': ['±0.005 mm', '±0.007 mm', '±0.001 T', '±0.1 T/m', '±0.05 MV'],
            'Date': ['2024-10-01', '2024-10-01', '2024-09-28', '2024-09-30', '2024-10-02'],
            'Statut': ['✅ Valide', '✅ Valide', '✅ Valide', '✅ Valide', '✅ Valide']
        }
        
        df_calib = pd.DataFrame(calibration_data)
        st.dataframe(df_calib, use_container_width=True)
        
        if st.button("📥 Exporter Table de Calibration"):
            csv = df_calib.to_csv(index=False)
            st.download_button(
                label="⬇️ Télécharger CSV",
                data=csv,
                file_name=f"calibration_table_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with tab4:
        st.subheader("📊 Résultats des Tests")
        
        st.write("### 📈 Résumé Global")
        
        # Statistiques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tests Réalisés", "247")
        with col2:
            st.metric("Tests Réussis", "241", delta="97.6%")
        with col3:
            st.metric("Tests Échoués", "3", delta="-1.2%", delta_color="inverse")
        with col4:
            st.metric("Tests en Attente", "3")
        
        st.markdown("---")
        
        # Graphique par catégorie
        st.write("### 📊 Tests par Catégorie")
        
        categories = ['Aimants', 'Cavités RF', 'Vide', 'Cryogénie', 'Instrumentation', 'Contrôle']
        passed = [98, 45, 20, 15, 38, 25]
        failed = [2, 0, 0, 1, 0, 0]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Réussis',
            x=categories,
            y=passed,
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name='Échoués',
            x=categories,
            y=failed,
            marker_color='red'
        ))
        
        fig.update_layout(
            title="Tests par Catégorie",
            xaxis_title="Catégorie",
            yaxis_title="Nombre de Tests",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Liste des tests échoués
        st.write("### ⚠️ Tests Nécessitant une Attention")
        
        failed_tests = [
            {
                "ID": "TEST-MAG-157",
                "Composant": "Quadrupôle Q-234",
                "Problème": "Alignement hors tolérance (0.15mm > 0.1mm)",
                "Action": "Réalignement nécessaire",
                "Priorité": "Haute"
            },
            {
                "ID": "TEST-CRY-045",
                "Composant": "Ligne cryogénique Section 3",
                "Problème": "Fuite détectée (taux 10⁻⁸ mbar·L/s)",
                "Action": "Resserrer connexions",
                "Priorité": "Critique"
            },
            {
                "ID": "TEST-INS-089",
                "Composant": "BPM-456",
                "Problème": "Signal bruité (SNR < 40dB)",
                "Action": "Vérifier câblage",
                "Priorité": "Moyenne"
            }
        ]
        
        for test in failed_tests:
            priority_colors = {"Critique": "🔴", "Haute": "🟠", "Moyenne": "🟡"}
            priority_icon = priority_colors.get(test['Priorité'], "⚪")
            
            with st.expander(f"{priority_icon} {test['ID']}: {test['Composant']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Problème:** {test['Problème']}")
                    st.write(f"**Action Requise:** {test['Action']}")
                
                with col2:
                    st.metric("Priorité", test['Priorité'])
                    
                    if st.button("✅ Marquer Résolu", key=f"resolve_{test['ID']}"):
                        st.success("Test marqué comme résolu!")
        
        st.markdown("---")
        
        # Rapport final
        st.write("### 📄 Génération du Rapport Final")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            report_type = st.selectbox(
                "Type de Rapport",
                ["Rapport Complet", "Résumé Exécutif", "Rapport Technique Détaillé", "Non-Conformités Uniquement"]
            )
            
            include_sections = st.multiselect(
                "Sections à Inclure",
                ["Tests Aimants", "Tests RF", "Tests Vide", "Tests Cryogénie", 
                 "Calibrations", "Problèmes Identifiés", "Recommandations"],
                default=["Tests Aimants", "Tests RF", "Problèmes Identifiés"]
            )
        
        with col2:
            if st.button("📄 Générer Rapport", use_container_width=True, type="primary"):
                st.success("✅ Rapport généré!")
                
                # Contenu du rapport (simplifié)
                report_content = f"""
# RAPPORT DE TESTS ET CALIBRATION
Date: {datetime.now().strftime('%Y-%m-%d')}
Accélérateur: {acc['name'] if 'acc' in locals() else 'N/A'}

## RÉSUMÉ EXÉCUTIF
- Tests réalisés: 247
- Tests réussis: 241 (97.6%)
- Tests échoués: 3 (1.2%)
- Composants calibrés: 156

## CONCLUSION
Le système est prêt pour le commissioning avec 3 actions correctives mineures.
                """
                
                st.download_button(
                    label="⬇️ Télécharger Rapport (PDF simulé)",
                    data=report_content,
                    file_name=f"test_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

# ==================== PAGE: COÛTS & BUDGET ====================

elif page == "💰 Coûts & Budget":
    st.header("💰 Gestion des Coûts et Budget")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💵 Budget Global", "📊 Suivi Dépenses", "📈 Prévisions", "💳 Financement"])
    
    with tab1:
        st.subheader("💵 Budget Global du Projet")
        
        if not st.session_state.accelerator_system['accelerators']:
            st.warning("⚠️ Aucun accélérateur disponible")
        else:
            acc_options = {a['id']: a['name'] for a in st.session_state.accelerator_system['accelerators'].values()}
            selected_acc = st.selectbox(
                "Sélectionner un projet",
                options=list(acc_options.keys()),
                format_func=lambda x: acc_options[x],
                key="budget_acc"
            )
            
            acc = st.session_state.accelerator_system['accelerators'][selected_acc]
            
            st.write(f"### 💰 {acc['name']}")
            
            # Budget par catégorie
            total_budget = acc['costs']['construction']
            
            budget_breakdown = {
                'Infrastructure & Génie Civil': total_budget * 0.25,
                'Aimants Supraconducteurs': total_budget * 0.20,
                'Système Radiofréquence': total_budget * 0.15,
                'Système de Vide': total_budget * 0.08,
                'Cryogénie': total_budget * 0.10,
                'Détecteurs': total_budget * 0.12,
                'Électronique & DAQ': total_budget * 0.05,
                'Installation & Commissioning': total_budget * 0.05
            }
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Budget Total", f"${total_budget/1e9:.2f}B")
            with col2:
                operational = acc['costs']['operational']
                st.metric("Opérationnel/an", f"${operational/1e6:.0f}M")
            with col3:
                lifetime_cost = total_budget + operational * 20  # 20 ans
                st.metric("Coût sur 20 ans", f"${lifetime_cost/1e9:.2f}B")
            with col4:
                st.metric("Énergie/an", f"${acc['costs']['energy_consumption'] * 100:,.0f}")
            
            st.markdown("---")
            
            # Graphique circulaire
            st.write("### 📊 Répartition du Budget")
            
            fig = px.pie(
                values=list(budget_breakdown.values()),
                names=list(budget_breakdown.keys()),
                title="Budget par Catégorie",
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Tableau détaillé
            st.write("### 📋 Détail Budgétaire")
            
            budget_data = []
            for category, amount in budget_breakdown.items():
                budget_data.append({
                    'Catégorie': category,
                    'Budget (M$)': f"{amount/1e6:.2f}",
                    'Pourcentage': f"{amount/total_budget*100:.1f}%",
                    'Dépensé': f"{amount * 0.65/1e6:.2f}",  # 65% dépensé
                    'Restant': f"{amount * 0.35/1e6:.2f}"
                })
            
            df_budget = pd.DataFrame(budget_data)
            st.dataframe(df_budget, use_container_width=True)
            
            st.markdown("---")
            
            # Comparaison internationale
            st.write("### 🌍 Comparaison avec Projets Similaires")
            
            comparison_data = {
                'Projet': [acc['name'], 'LHC (CERN)', 'ILC (Projet)', 'LCLS-II (SLAC)', 'FAIR (GSI)'],
                'Type': [acc['type'], 'Collisionneur', 'Linéaire', 'Linéaire', 'Synchrotron'],
                'Longueur (km)': [acc['geometry']['length']/1000, 27, 31, 3.2, 1.1],
                'Énergie (TeV)': [acc['energy']['max']/1e12, 14, 0.5, 0.008, 0.002],
                'Coût (G$)': [total_budget/1e9, 4.75, 20, 1.0, 1.4],
                'Année': [2024, 2008, 'N/A', 2020, 2018]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Graphique comparatif
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_comparison['Projet'],
                y=df_comparison['Coût (G$)'],
                marker_color=['red' if p == acc['name'] else 'lightblue' for p in df_comparison['Projet']],
                text=df_comparison['Coût (G$)'],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Comparaison des Coûts",
                xaxis_title="Projet",
                yaxis_title="Coût (Milliards $)",
                height=400,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Suivi des Dépenses")
        
        st.write("### 💸 Dépenses Cumulées")
        
        # Simulation de dépenses sur 5 ans
        months = pd.date_range(start='2020-01', end='2024-10', freq='M')
        
        # Dépenses cumulées (courbe en S)
        t = np.linspace(0, 1, len(months))
        cumulative_spent = total_budget * (1 / (1 + np.exp(-10*(t-0.5))))
        
        # Budget planifié
        cumulative_budget = total_budget * t
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_budget/1e9,
            mode='lines',
            name='Budget Planifié',
            line=dict(color='blue', dash='dash', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=months,
            y=cumulative_spent/1e9,
            mode='lines',
            name='Dépenses Réelles',
            line=dict(color='red', width=3),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title="Dépenses Cumulées vs Budget",
            xaxis_title="Date",
            yaxis_title="Montant (Milliards $)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicateurs
        col1, col2, col3, col4 = st.columns(4)
        
        current_spent = cumulative_spent[-1]
        current_budget = cumulative_budget[-1]
        variance = current_spent - current_budget
        
        with col1:
            st.metric("Dépensé à Date", f"${current_spent/1e9:.2f}B")
        with col2:
            st.metric("Budget à Date", f"${current_budget/1e9:.2f}B")
        with col3:
            st.metric("Variance", f"${variance/1e9:.2f}B", 
                     delta=f"{variance/current_budget*100:.1f}%",
                     delta_color="inverse")
        with col4:
            completion = (current_spent / total_budget) * 100
            st.metric("Avancement", f"{completion:.1f}%")
        
        st.markdown("---")
        
        # Dépenses mensuelles
        st.write("### 📅 Dépenses Mensuelles")
        
        monthly_spent = np.diff(cumulative_spent, prepend=0) / 1e6  # Millions
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=months,
            y=monthly_spent,
            marker_color='rgba(102, 126, 234, 0.7)',
            name='Dépenses Mensuelles'
        ))
        
        # Moyenne mobile
        window = 3
        moving_avg = np.convolve(monthly_spent, np.ones(window)/window, mode='same')
        
        fig.add_trace(go.Scatter(
            x=months,
            y=moving_avg,
            mode='lines',
            name='Moyenne Mobile (3 mois)',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Dépenses Mensuelles",
            xaxis_title="Date",
            yaxis_title="Dépenses (Millions $)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top dépenses
        st.write("### 🔝 Top 10 des Dépenses")
        
        top_expenses = [
            {"Description": "Aimants dipolaires supraconducteurs (Lot 1)", "Montant": 450, "Fournisseur": "CERN", "Date": "2022-06"},
            {"Description": "Cavités RF 500 MHz (50 unités)", "Montant": 380, "Fournisseur": "Jefferson Lab", "Date": "2023-03"},
            {"Description": "Excavation tunnel principal", "Montant": 320, "Fournisseur": "BTP International", "Date": "2021-09"},
            {"Description": "Système cryogénique complet", "Montant": 280, "Fournisseur": "Air Liquide", "Date": "2022-11"},
            {"Description": "Détecteurs silicium (10,000 m²)", "Montant": 250, "Fournisseur": "Hamamatsu", "Date": "2023-08"},
            {"Description": "Quadrupôles supraconducteurs (200 unités)", "Montant": 220, "Fournisseur": "CERN", "Date": "2022-12"},
            {"Description": "Infrastructure électrique HT", "Montant": 180, "Fournisseur": "Schneider Electric", "Date": "2021-12"},
            {"Description": "Système de vide ultra-poussé", "Montant": 150, "Fournisseur": "VAT", "Date": "2023-02"},
            {"Description": "Électronique DAQ et trigger", "Montant": 140, "Fournisseur": "CAEN", "Date": "2023-06"},
            {"Description": "Klystrons 10 MW (20 unités)", "Montant": 130, "Fournisseur": "Thales", "Date": "2023-04"}
        ]
        
        df_expenses = pd.DataFrame(top_expenses)
        st.dataframe(df_expenses, use_container_width=True)
        
        total_top10 = sum(exp['Montant'] for exp in top_expenses)
        st.info(f"**Total Top 10:** ${total_top10}M ({total_top10/current_spent*1e9*100:.1f}% du total dépensé)")
    
    with tab3:
        st.subheader("📈 Prévisions et Projections")
        
        st.write("### 🔮 Projection des Coûts Futurs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            years_ahead = st.slider("Années à Projeter", 1, 10, 5)
            inflation_rate = st.slider("Taux d'Inflation Annuel (%)", 0.0, 10.0, 2.5, 0.5)
        
        with col2:
            contingency = st.slider("Contingence (%)", 0, 30, 10)
            escalation = st.slider("Escalade des Coûts (%)", 0, 20, 5)
        
        # Calcul des projections
        future_years = np.arange(2024, 2024 + years_ahead + 1)
        
        # Coût opérationnel avec inflation
        operational_projected = []
        for i, year in enumerate(future_years):
            cost = operational * (1 + inflation_rate/100)**i
            operational_projected.append(cost)
        
        # Coûts de maintenance (augmente avec l'âge)
        maintenance_projected = []
        for i, year in enumerate(future_years):
            maintenance = operational * 0.3 * (1 + i*0.05)  # Augmente de 5% par an
            maintenance_projected.append(maintenance)
        
        # Upgrades planifiés
        upgrades = [0, 0, 0, 200e6, 0, 0]  # Upgrade majeur en année 3
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=future_years,
            y=[o/1e6 for o in operational_projected],
            name='Opérationnel',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=future_years,
            y=[m/1e6 for m in maintenance_projected],
            name='Maintenance',
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            x=future_years,
            y=[u/1e6 for u in upgrades],
            name='Upgrades',
            marker_color='green'
        ))
        
        fig.update_layout(
            title="Projection des Coûts Annuels",
            xaxis_title="Année",
            yaxis_title="Coût (Millions $)",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Résumé financier
        st.write("### 💰 Résumé Financier Projeté")
        
        total_operational = sum(operational_projected)
        total_maintenance = sum(maintenance_projected)
        total_upgrades = sum(upgrades)
        grand_total = total_operational + total_maintenance + total_upgrades
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Coûts Opérationnels", f"${total_operational/1e9:.2f}B")
        with col2:
            st.metric("Maintenance", f"${total_maintenance/1e9:.2f}B")
        with col3:
            st.metric("Upgrades", f"${total_upgrades/1e6:.0f}M")
        with col4:
            st.metric("Total Projeté", f"${grand_total/1e9:.2f}B")
        
        st.markdown("---")
        
        # Analyse de sensibilité
        st.write("### 📊 Analyse de Sensibilité")
        
        st.info("""
        Cette analyse montre comment les variations de paramètres clés affectent le coût total.
        """)
        
        params = ['Inflation', 'Coût Énergie', 'Personnel', 'Maintenance', 'Contingence']
        low_impact = [grand_total * 0.95 / 1e9 for _ in params]
        base_impact = [grand_total / 1e9 for _ in params]
        high_impact = [grand_total * 1.15 / 1e9, grand_total * 1.20 / 1e9, 
                      grand_total * 1.10 / 1e9, grand_total * 1.25 / 1e9, grand_total * 1.12 / 1e9]
        
        fig = go.Figure()
        
        for i, param in enumerate(params):
            fig.add_trace(go.Scatter(
                x=[low_impact[i], base_impact[i], high_impact[i]],
                y=[param, param, param],
                mode='lines+markers',
                name=param,
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Analyse de Sensibilité (Scénarios -10%, Base, +15-25%)",
            xaxis_title="Coût Total (Milliards $)",
            yaxis_title="Paramètre",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("💳 Sources de Financement")
        
        st.write("### 💵 Structure de Financement")
        
        # Sources de financement
        funding_sources = {
            'Gouvernement National': total_budget * 0.40,
            'Organisations Internationales': total_budget * 0.25,
            'Partenariats Publics-Privés': total_budget * 0.15,
            'Universités et Instituts': total_budget * 0.10,
            'Fondations Privées': total_budget * 0.05,
            'Autres Sources': total_budget * 0.05
        }
        
        fig = px.pie(
            values=list(funding_sources.values()),
            names=list(funding_sources.keys()),
            title="Sources de Financement",
            color_discrete_sequence=px.colors.sequential.Greens_r,
            hole=0.4
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Tableau des contributeurs
        st.write("### 🌍 Principaux Contributeurs")
        
        contributors = [
            {"Pays/Organisation": "États-Unis (DoE)", "Contribution (M$)": total_budget * 0.20 / 1e6, "Pourcentage": "20%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "Union Européenne", "Contribution (M$)": total_budget * 0.15 / 1e6, "Pourcentage": "15%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "Japon (KEK)", "Contribution (M$)": total_budget * 0.10 / 1e6, "Pourcentage": "10%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "Chine (IHEP)", "Contribution (M$)": total_budget * 0.08 / 1e6, "Pourcentage": "8%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "CERN", "Contribution (M$)": total_budget * 0.12 / 1e6, "Pourcentage": "12%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "Russie (JINR)", "Contribution (M$)": total_budget * 0.05 / 1e6, "Pourcentage": "5%", "Statut": "🟡 En négociation"},
            {"Pays/Organisation": "Inde (BARC)", "Contribution (M$)": total_budget * 0.04 / 1e6, "Pourcentage": "4%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "Canada (TRIUMF)", "Contribution (M$)": total_budget * 0.03 / 1e6, "Pourcentage": "3%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "Australie", "Contribution (M$)": total_budget * 0.02 / 1e6, "Pourcentage": "2%", "Statut": "✅ Engagé"},
            {"Pays/Organisation": "Autres", "Contribution (M$)": total_budget * 0.21 / 1e6, "Pourcentage": "21%", "Statut": "🟡 En discussion"}
        ]
        
        df_contributors = pd.DataFrame(contributors)
        st.dataframe(df_contributors, use_container_width=True)
        
        st.markdown("---")
        
        # Calendrier de financement
        st.write("### 📅 Calendrier de Financement")
        
        funding_years = ['2020', '2021', '2022', '2023', '2024', '2025', '2026']
        funding_received = [500, 800, 1200, 1500, 1800, 0, 0]  # Millions
        funding_planned = [500, 800, 1200, 1500, 1800, 2000, 1500]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=funding_years,
            y=funding_received,
            name='Fonds Reçus',
            marker_color='green',
            text=[f"${v}M" if v > 0 else "" for v in funding_received],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=funding_years,
            y=[p - r for p, r in zip(funding_planned, funding_received)],
            name='Fonds Planifiés',
            marker_color='lightgreen',
            text=[f"${funding_planned[i]}M" for i in range(len(funding_years))],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Financement Annuel (Planifié vs Reçu)",
            xaxis_title="Année",
            yaxis_title="Montant (Millions $)",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Retour sur investissement
        st.write("### 📈 Retour sur Investissement (ROI)")
        
        st.info("""
        **Bénéfices Estimés:**
        - Avancées scientifiques majeures
        - Formation de 5,000+ scientifiques et ingénieurs
        - Développement technologique (retombées industrielles)
        - Création de 10,000 emplois directs et indirects
        - Attraction de talents internationaux
        - Publications scientifiques: ~5,000 articles
        - Brevets technologiques: ~200
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Emplois Créés", "10,000+")
            st.metric("Scientifiques Formés", "5,000+")
        
        with col2:
            st.metric("Publications Attendues", "5,000+")
            st.metric("Brevets Prévus", "200+")
        
        with col3:
            economic_impact = total_budget * 1.5  # Multiplicateur économique
            st.metric("Impact Économique", f"${economic_impact/1e9:.2f}B")
            st.metric("Ratio Bénéfice/Coût", "1.5x")

# ==================== PAGE: BIBLIOTHÈQUE ====================

elif page == "📚 Bibliothèque":
    st.header("📚 Bibliothèque de Ressources")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Documentation", "🔬 Physique", "📐 Formules", "🎓 Formation"])
    
    with tab1:
        st.subheader("📖 Documentation Technique")
        
        doc_categories = {
            "📘 Manuels Techniques": [
                {"title": "Design and Construction of Particle Accelerators", "pages": 850, "type": "Manuel", "year": 2023},
                {"title": "Superconducting Magnets for Accelerators", "pages": 420, "type": "Guide", "year": 2022},
                {"title": "RF Cavity Design and Optimization", "pages": 320, "type": "Tutorial", "year": 2023},
                {"title": "Vacuum Systems for Accelerators", "pages": 280, "type": "Manuel", "year": 2021},
                {"title": "Beam Dynamics and Optics", "pages": 650, "type": "Textbook", "year": 2022}
            ],
            "📗 Standards et Normes": [
                {"title": "IEEE Standards for Particle Accelerators", "pages": 200, "type": "Standard", "year": 2023},
                {"title": "Safety Standards for High Energy Physics", "pages": 180, "type": "Standard", "year": 2022},
                {"title": "Quality Assurance Guidelines", "pages": 150, "type": "Guide", "year": 2023},
                {"title": "Radiation Protection Standards", "pages": 220, "type": "Standard", "year": 2021}
            ],
            "📙 Publications Scientifiques": [
                {"title": "Recent Advances in Accelerator Physics", "pages": 45, "type": "Article", "year": 2024},
                {"title": "Novel Acceleration Techniques", "pages": 32, "type": "Review", "year": 2024},
                {"title": "Beam-Beam Interactions at High Luminosity", "pages": 28, "type": "Article", "year": 2023},
                {"title": "Machine Learning for Beam Optimization", "pages": 38, "type": "Research", "year": 2024}
            ],
            "📕 Rapports de Projet": [
                {"title": "LHC Conceptual Design Report", "pages": 430, "type": "CDR", "year": 1995},
                {"title": "ILC Technical Design Report", "pages": 1200, "type": "TDR", "year": 2013},
                {"title": "FAIR Baseline Technical Report", "pages": 680, "type": "BTR", "year": 2006},
                {"title": "ESS Technical Design Report", "pages": 520, "type": "TDR", "year": 2013}
            ]
        }
        
        for category, docs in doc_categories.items():
            with st.expander(f"{category} ({len(docs)} documents)"):
                for doc in docs:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{doc['title']}**")
                    with col2:
                        st.caption(f"{doc['type']}")
                    with col3:
                        st.caption(f"{doc['pages']} pages")
                    with col4:
                        if st.button("📥", key=f"dl_{doc['title'][:20]}"):
                            st.info("Téléchargement simulé")
        
        st.markdown("---")
        
        # Recherche dans la bibliothèque
        st.write("### 🔍 Recherche dans la Bibliothèque")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("Rechercher un document", placeholder="Ex: beam dynamics")
        
        with col2:
            doc_type_filter = st.selectbox("Type", ["Tous", "Manuel", "Article", "Standard", "Guide"])
        
        if search_query:
            st.success(f"🔍 Recherche pour: '{search_query}'")
            st.info("Résultats de recherche simulés...")
    
    with tab2:
        st.subheader("🔬 Concepts de Physique des Accélérateurs")
        
        physics_topics = {
            "Dynamique du Faisceau": {
                "description": "Mouvement des particules dans les champs EM",
                "concepts": [
                    "Équation de Hill",
                    "Oscillations betatron",
                    "Tune et chromaticité",
                    "Acceptance et émittance",
                    "Espace des phases"
                ],
                "formulas": [
                    r"x'' + K(s)x = 0",
                    r"\nu = \frac{1}{2\pi}\oint \frac{ds}{\beta(s)}",
                    r"\xi = -\frac{1}{4\pi}\oint \beta(s)K(s)ds"
                ]
            },
            "Synchrotron Radiation": {
                "description": "Rayonnement émis par les particules accélérées",
                "concepts": [
                    "Puissance rayonnée",
                    "Spectre du rayonnement",
                    "Amortissement radiatif",
                    "Excitation quantique",
                    "Temps d'amortissement"
                ],
                "formulas": [
                    r"P = \frac{C_\gamma}{2\pi}\frac{E^4}{\rho^2}",
                    r"\lambda_c = \frac{4\pi\rho}{3\gamma^3}",
                    r"\tau_x = \frac{2E\rho}{J_x C_\gamma E^3}"
                ]
            },
            "Interactions Faisceau-Faisceau": {
                "description": "Effets lors de collisions de faisceaux",
                "concepts": [
                    "Tune shift",
                    "Hourglass effect",
                    "Luminosité",
                    "Crossing angle",
                    "Crab cavities"
                ],
                "formulas": [
                    r"\Delta\nu = \frac{N r_p}{4\pi\epsilon\gamma}",
                    r"\mathcal{L} = \frac{n N_1 N_2 f}{4\pi\sigma_x\sigma_y}",
                    r"R = \sigma_z/\beta^*"
                ]
            },
            "Instabilités": {
                "description": "Instabilités collectives du faisceau",
                "concepts": [
                    "Instabilité head-tail",
                    "Instabilité multi-bunch",
                    "Landau damping",
                    "Feedback systems",
                    "Impedance budget"
                ],
                "formulas": [
                    r"\omega_s = \omega_0\sqrt{\frac{\eta h eV_{RF}}{2\pi E}}",
                    r"Z_\parallel^{eff} = \frac{Im(Z_\parallel)}{\omega}",
                    r"\tau^{-1} = -\frac{I e^2 \eta}{2E \omega_s m c^2} Im(Z_\parallel)"
                ]
            }
        }
        
        for topic, info in physics_topics.items():
            with st.expander(f"📚 {topic}"):
                st.write(f"**Description:** {info['description']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Concepts Clés:**")
                    for concept in info['concepts']:
                        st.write(f"• {concept}")
                
                with col2:
                    st.write("**Formules Principales:**")
                    for formula in info['formulas']:
                        st.latex(formula)
        
        st.markdown("---")
        
        # Glossaire
        st.write("### 📖 Glossaire")
        
        glossary = {
            "Émittance": "Mesure de la qualité du faisceau, produit de la taille et de la divergence",
            "Tune": "Nombre d'oscillations par tour dans un accélérateur circulaire",
            "Chromaticité": "Variation du tune avec l'énergie (Δν/(Δp/p))",
            "β-fonction": "Fonction de Twiss décrivant l'enveloppe du faisceau",
            "Luminosité": "Mesure du taux de collision dans un collisionneur (cm⁻²s⁻¹)",
            "Dispersion": "Séparation spatiale des particules selon leur énergie",
            "Rigidité magnétique": "Rapport p/q, lien entre momentum et champ de courbure",
            "Acceptance": "Émittance maximale que le système peut accepter",
            "Quench": "Perte de supraconductivité dans un aimant",
            "RF bucket": "Région de stabilité en phase dans le système RF"
        }
        
        for term, definition in glossary.items():
            with st.expander(f"📌 {term}"):
                st.write(definition)
    
    with tab3:
        st.subheader("📐 Formulaire de Calculs")
        
        st.write("### 🧮 Calculateurs Rapides")
        
        calc_type = st.selectbox(
            "Sélectionner un calculateur",
            ["Rigidité Magnétique", "Énergie Synchrotron", "Luminosité", 
             "Temps de Damping", "Section Efficace", "Taux d'Événements"]
        )
        
        if calc_type == "Rigidité Magnétique":
            st.write("**Calcul de la Rigidité Magnétique Bρ**")
            
            st.latex(r"B\rho = \frac{p}{q} = \frac{\sqrt{E^2 - (mc^2)^2}}{qc}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                particle_calc = st.selectbox("Particule", list(PARTICLE_DATA.keys()), key="rigidity_p")
                energy_calc = st.number_input("Énergie (GeV)", 0.1, 10000.0, 100.0, key="rigidity_e")
            
            with col2:
                if st.button("🔢 Calculer"):
                    particle_info = PARTICLE_DATA[particle_calc]
                    
                    energy_j = energy_calc * 1e9 * constants.e
                    rest_energy = particle_info['mass'] * constants.c**2
                    momentum = np.sqrt(energy_j**2 - rest_energy**2) / constants.c
                    
                    B_rho = momentum / abs(particle_info['charge'])
                    
                    st.success(f"**Bρ = {B_rho:.3f} T·m**")
                    
                    # Exemples de rayons de courbure
                    fields = [0.5, 1.0, 2.0, 5.0, 8.33]
                    radii = [B_rho / B for B in fields]
                    
                    st.write("**Rayon de courbure pour différents champs:**")
                    for B, R in zip(fields, radii):
                        st.write(f"• B = {B} T → R = {R:.2f} m")
        
        elif calc_type == "Luminosité":
            st.write("**Calcul de la Luminosité**")
            
            st.latex(r"\mathcal{L} = \frac{n_b N_1 N_2 f_{rev}}{4\pi\sigma_x\sigma_y} F")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_bunches_lumi = st.number_input("Nombre de paquets", 1, 10000, 2808)
                N1 = st.number_input("Particules/paquet (faisceau 1)", 1e8, 1e12, 1.15e11, format="%.2e")
                N2 = st.number_input("Particules/paquet (faisceau 2)", 1e8, 1e12, 1.15e11, format="%.2e")
            
            with col2:
                f_rev = st.number_input("Fréquence révolution (kHz)", 1.0, 100.0, 11.245)
                sigma_x = st.number_input("σₓ (μm)", 1.0, 1000.0, 16.7)
                sigma_y = st.number_input("σᵧ (μm)", 1.0, 1000.0, 16.7)
            
            if st.button("🔢 Calculer Luminosité"):
                F = 0.836  # Facteur géométrique
                
                lumi = (n_bunches_lumi * N1 * N2 * f_rev * 1e3 * F) / (4 * np.pi * (sigma_x*1e-6) * (sigma_y*1e-6))
                
                st.success(f"**ℒ = {lumi:.2e} cm⁻²s⁻¹**")
                
                # Luminosité intégrée
                lumi_int_sec = lumi * 1e-4  # barn⁻¹s⁻¹
                lumi_int_day = lumi_int_sec * 86400 / 1e15  # fb⁻¹/jour
                
                st.write(f"**Luminosité intégrée:** {lumi_int_day:.2f} fb⁻¹/jour")
        
        elif calc_type == "Taux d'Événements":
            st.write("**Calcul du Taux d'Événements**")
            
            st.latex(r"R = \mathcal{L} \times \sigma")
            
            col1, col2 = st.columns(2)
            
            with col1:
                lumi_input = st.number_input("Luminosité (cm⁻²s⁻¹)", 1e30, 1e38, 1e34, format="%.2e")
                sigma_input = st.number_input("Section Efficace (pb)", 0.001, 1e6, 100.0)
            
            with col2:
                if st.button("🔢 Calculer Taux"):
                    # Conversion pb → cm²
                    sigma_cm2 = sigma_input * 1e-36
                    
                    rate = lumi_input * sigma_cm2  # Hz
                    rate_per_day = rate * 86400
                    rate_per_year = rate_per_day * 365
                    
                    st.success(f"**Taux: {rate:.2e} Hz**")
                    st.write(f"**Par jour:** {rate_per_day:.2e} événements")
                    st.write(f"**Par an:** {rate_per_year:.2e} événements")
        
        st.markdown("---")
        
        # Table de conversion
        st.write("### 🔄 Table de Conversion d'Unités")
        
        conversions = {
            "Énergie": {
                "1 eV": "1.602176634×10⁻¹⁹ J",
                "1 MeV": "1.602176634×10⁻¹³ J",
                "1 GeV": "1.602176634×10⁻¹⁰ J",
                "1 TeV": "1.602176634×10⁻⁷ J"
            },
            "Section Efficace": {
                "1 barn": "10⁻²⁴ cm²",
                "1 millibarn (mb)": "10⁻²⁷ cm²",
                "1 microbarn (μb)": "10⁻³⁰ cm²",
                "1 nanobarn (nb)": "10⁻³³ cm²",
                "1 picobarn (pb)": "10⁻³⁶ cm²",
                "1 femtobarn (fb)": "10⁻³⁹ cm²"
            },
            "Luminosité Intégrée": {
                "1 fb⁻¹": "10³⁹ cm⁻²",
                "1 pb⁻¹": "10³⁶ cm⁻²",
                "1 nb⁻¹": "10³³ cm⁻²"
            }
        }
        
        for cat, conv in conversions.items():
            with st.expander(f"📊 {cat}"):
                for unit, value in conv.items():
                    st.write(f"**{unit}** = {value}")
    
    with tab4:
        st.subheader("🎓 Ressources de Formation")
        
        st.write("### 📺 Cours et Tutoriels")
        
        courses = [
            {
                "title": "Introduction to Particle Accelerators",
                "level": "Débutant",
                "duration": "10 heures",
                "instructor": "CERN",
                "topics": ["Bases de la physique", "Types d'accélérateurs", "Applications"],
                "format": "Vidéo + Exercices"
            },
            {
                "title": "Beam Dynamics and Optics",
                "level": "Intermédiaire",
                "duration": "20 heures",
                "instructor": "US Particle Accelerator School",
                "topics": ["Équations de Hill", "Matrices de transfert", "Lattice design"],
                "format": "Cours magistral + TP"
            },
            {
                "title": "Superconducting RF Technology",
                "level": "Avancé",
                "duration": "15 heures",
                "instructor": "Jefferson Lab",
                "topics": ["Cavités supraconductrices", "Q factor", "Multipacting"],
                "format": "Laboratoire + Séminaires"
            },
            {
                "title": "Machine Learning for Accelerators",
                "level": "Avancé",
                "duration": "12 heures",
                "instructor": "SLAC",
                "topics": ["Optimisation ML", "Contrôle par IA", "Prédiction"],
                "format": "Coding + Projets"
            }
        ]
        
        for course in courses:
            with st.expander(f"🎓 {course['title']} ({course['level']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Instructeur:** {course['instructor']}")
                    st.write(f"**Durée:** {course['duration']}")
                    st.write(f"**Format:** {course['format']}")
                    
                    st.write("**Sujets Couverts:**")
                    for topic in course['topics']:
                        st.write(f"• {topic}")
                
                with col2:
                    level_colors = {"Débutant": "🟢", "Intermédiaire": "🟡", "Avancé": "🔴"}
                    st.metric("Niveau", f"{level_colors[course['level']]} {course['level']}")
                    
                    if st.button("📚 S'inscrire", key=f"enroll_{course['title'][:15]}"):
                        st.success("Inscription simulée!")
        
        st.markdown("---")
        
        # Conférences et écoles
        st.write("### 🌍 Conférences et Écoles")
        
        events = [
            {"name": "International Particle Accelerator Conference (IPAC)", "date": "Mai 2025", "location": "Venise, Italie"},
            {"name": "US Particle Accelerator School", "date": "Juin 2025", "location": "Chicago, USA"},
            {"name": "CERN Accelerator School", "date": "Septembre 2025", "location": "Genève, Suisse"},
            {"name": "Asian Accelerator School", "date": "Octobre 2025", "location": "Tokyo, Japon"}
        ]
        
        for event in events:
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{event['name']}**")
            with col2:
                st.write(event['date'])
            with col3:
                st.write(event['location'])
            with col4:
                if st.button("ℹ️", key=f"info_{event['name'][:10]}"):
                    st.info("Plus d'informations...")

# ==================== PAGE: DÉCOUVERTES ====================

elif page == "🌌 Découvertes":
    st.header("🌌 Découvertes Scientifiques")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Découvertes Majeures", "📊 Données Récentes", "🔬 Analyses en Cours", "🌟 Prix et Reconnaissances"])
    
    with tab1:
        st.subheader("🏆 Découvertes Majeures en Physique des Particules")
        
        discoveries = [
            {
                "title": "Boson de Higgs",
                "year": 2012,
                "accelerator": "LHC (CERN)",
                "significance": "Découverte majeure - Prix Nobel 2013",
                "description": "Confirmation du mécanisme de Higgs donnant la masse aux particules",
                "mass": "125.1 GeV/c²",
                "significance_sigma": "5.0σ (découverte)",
                "experiments": ["ATLAS", "CMS"],
                "impact": "⭐⭐⭐⭐⭐"
            },
            {
                "title": "Quark Top",
                "year": 1995,
                "accelerator": "Tevatron (Fermilab)",
                "significance": "Découverte majeure",
                "description": "Découverte du quark le plus lourd",
                "mass": "173.1 GeV/c²",
                "significance_sigma": ">5σ",
                "experiments": ["CDF", "D0"],
                "impact": "⭐⭐⭐⭐⭐"
            },
            {
                "title": "Boson W et Z",
                "year": 1983,
                "accelerator": "SPS (CERN)",
                "significance": "Découverte majeure - Prix Nobel 1984",
                "description": "Bosons médiateurs de la force faible",
                "mass": "W: 80.4 GeV/c², Z: 91.2 GeV/c²",
                "significance_sigma": ">5σ",
                "experiments": ["UA1", "UA2"],
                "impact": "⭐⭐⭐⭐⭐"
            },
            {
                "title": "Neutrinos Atmosphériques (Oscillation)",
                "year": 1998,
                "accelerator": "Super-Kamiokande",
                "significance": "Découverte majeure - Prix Nobel 2015",
                "description": "Preuve que les neutrinos ont une masse",
                "mass": "Δm² ~ 10⁻³ eV²",
                "significance_sigma": ">5σ",
                "experiments": ["Super-Kamiokande"],
                "impact": "⭐⭐⭐⭐⭐"
            },
            {
                "title": "Violation CP dans les Mésons B",
                "year": 2001,
                "accelerator": "PEP-II, KEKB",
                "significance": "Découverte majeure - Prix Nobel 2008",
                "description": "Asymétrie matière-antimatière dans les mésons B",
                "mass": "N/A",
                "significance_sigma": ">5σ",
                "experiments": ["BaBar", "Belle"],
                "impact": "⭐⭐⭐⭐⭐"
            }
        ]
        
        for disc in discoveries:
            with st.expander(f"🏆 {disc['title']} ({disc['year']}) {disc['impact']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {disc['description']}")
                    st.write(f"**Accélérateur:** {disc['accelerator']}")
                    st.write(f"**Expériences:** {', '.join(disc['experiments'])}")
                    st.write(f"**Masse:** {disc['mass']}")
                
                with col2:
                    st.metric("Année", disc['year'])
                    st.metric("Signification", disc['significance_sigma'])
                    st.write(f"**Impact:** {disc['impact']}")
                    
                    if "Nobel" in disc['significance']:
                        st.success("🏅 Prix Nobel!")
        
        st.markdown("---")
        
        # Timeline
        st.write("### 📅 Timeline des Découvertes")
        
        fig = go.Figure()
        
        years = [d['year'] for d in discoveries]
        titles = [d['title'] for d in discoveries]
        
        fig.add_trace(go.Scatter(
            x=years,
            y=[i for i in range(len(discoveries))],
            mode='markers+text',
            marker=dict(size=20, color='gold', line=dict(color='darkgoldenrod', width=2)),
            text=titles,
            textposition='top center',
            textfont=dict(size=10),
            hovertemplate='<b>%{text}</b><br>Année: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Timeline des Découvertes Majeures",
            xaxis_title="Année",
            yaxis_title="",
            yaxis=dict(showticklabels=False),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Résultats Expérimentaux Récents")
        
        st.write("### 🔬 Dernières Mesures")
        
        recent_results = [
            {
                "measurement": "Masse du Boson de Higgs",
                "value": "125.25 ± 0.17 GeV/c²",
                "experiment": "ATLAS+CMS combiné",
                "date": "2024-09",
                "precision": "0.14%",
                "comparison": "PDG 2023: 125.10 ± 0.14 GeV/c²"
            },
            {
                "measurement": "Moment Magnétique Anomal du Muon",
                "value": "(g-2)/2 = 0.00116592061 ± 0.00000000041",
                "experiment": "Muon g-2 (Fermilab)",
                "date": "2024-08",
                "precision": "0.035 ppm",
                "comparison": "Déviation 4.2σ du MS"
            },
            {
                "measurement": "Masse du Quark Top",
                "value": "172.52 ± 0.32 GeV/c²",
                "experiment": "LHC combiné",
                "date": "2024-07",
                "precision": "0.19%",
                "comparison": "PDG 2023: 172.76 ± 0.30 GeV/c²"
            },
            {
                "measurement": "Angle de Mélange θ₁₃ (Neutrinos)",
                "value": "sin²(2θ₁₃) = 0.0841 ± 0.0027",
                "experiment": "Daya Bay",
                "date": "2024-06",
                "precision": "3.2%",
                "comparison": "Confirmation des oscillations"
            }
        ]
        
        for result in recent_results:
            with st.expander(f"📊 {result['measurement']} ({result['date']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Valeur Mesurée:** {result['value']}")
                    st.write(f"**Expérience:** {result['experiment']}")
                    st.write(f"**Comparaison:** {result['comparison']}")
                
                with col2:
                    st.metric("Date", result['date'])
                    st.metric("Précision", result['precision'])
                    
                    if st.button("📄 Voir Publication", key=f"pub_{result['measurement'][:15]}"):
                        st.info("Lien vers la publication...")
        
        st.markdown("---")
        
        # Graphique de précision
        st.write("### 📈 Évolution de la Précision")
        
        years_precision = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
        higgs_precision = [5.0, 0.5, 0.3, 0.25, 0.20, 0.18, 0.15, 0.14]  # % incertitude
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years_precision,
            y=higgs_precision,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=10),
            name='Masse Higgs'
        ))
        
        fig.update_layout(
            title="Amélioration de la Précision - Masse du Higgs",
            xaxis_title="Année",
            yaxis_title="Incertitude (%)",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔬 Analyses en Cours")
        
        st.write("### 🔄 Recherches Actuelles")
        
        ongoing_searches = [
            {
                "search": "Recherche de Supersymétrie",
                "status": "En cours",
                "progress": 75,
                "target": "Particules SUSY (squarks, gluinos)",
                "current_limit": "Gluino > 2.3 TeV",
                "expected_result": "2025 Q1",
                "data_analyzed": "180 fb⁻¹"
            },
            {
                "search": "Matière Noire - WIMPs",
                "status": "En cours",
                "progress": 60,
                "target": "Particules massives faiblement interactives",
                "current_limit": "σ < 10⁻⁴⁶ cm²",
                "expected_result": "2025 Q2",
                "data_analyzed": "220 fb⁻¹"
            },
            {
                "search": "Dimensions Supplémentaires",
                "status": "En cours",
                "progress": 85,
                "target": "Gravitons de Kaluza-Klein",
                "current_limit": "M_D > 11 TeV",
                "expected_result": "2024 Q4",
                "data_analyzed": "300 fb⁻¹"
            },
            {
                "search": "Leptoquarks",
                "status": "En cours",
                "progress": 50,
                "target": "Leptoquarks scalaires et vectoriels",
                "current_limit": "M_LQ > 1.8 TeV",
                "expected_result": "2025 Q3",
                "data_analyzed": "140 fb⁻¹"
            },
            {
                "search": "Violation de Saveur Leptonique",
                "status": "En cours",
                "progress": 40,
                "target": "Désintégrations H → μτ",
                "current_limit": "BR < 0.15%",
                "expected_result": "2025 Q4",
                "data_analyzed": "100 fb⁻¹"
            }
        ]
        
        for search in ongoing_searches:
            with st.expander(f"🔬 {search['search']} ({search['progress']}%)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Cible:** {search['target']}")
                    st.write(f"**Limite Actuelle:** {search['current_limit']}")
                    st.write(f"**Données Analysées:** {search['data_analyzed']}")
                    
                    st.progress(search['progress'] / 100)
                
                with col2:
                    status_color = "🟢" if search['progress'] > 70 else "🟡"
                    st.write(f"**Statut:** {status_color} {search['status']}")
                    st.metric("Résultat Attendu", search['expected_result'])
                
                # Graphique de sensibilité
                if st.button("📊 Voir Sensibilité", key=f"sens_{search['search'][:15]}"):
                    # Simulation de courbe de sensibilité
                    luminosity = np.linspace(0, 400, 50)
                    sensitivity = search['progress'] * (1 - np.exp(-luminosity/100))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=luminosity,
                        y=sensitivity,
                        mode='lines',
                        line=dict(color='green', width=3)
                    ))
                    
                    fig.update_layout(
                        title=f"Sensibilité vs Luminosité - {search['search']}",
                        xaxis_title="Luminosité Intégrée (fb⁻¹)",
                        yaxis_title="Sensibilité (%)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Roadmap
        st.write("### 🗺️ Roadmap de Recherche")
        
        roadmap_items = [
            {"quarter": "2024 Q4", "milestone": "Finalisation analyse SUSY", "status": "✅ En cours"},
            {"quarter": "2025 Q1", "milestone": "Publication résultats dimensions extra", "status": "🟡 Planifié"},
            {"quarter": "2025 Q2", "milestone": "Nouvelles limites matière noire", "status": "🟡 Planifié"},
            {"quarter": "2025 Q3", "milestone": "Résultats leptoquarks", "status": "⚪ À venir"},
            {"quarter": "2025 Q4", "milestone": "Bilan complet Run 3", "status": "⚪ À venir"}
        ]
        
        for item in roadmap_items:
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.write(f"**{item['quarter']}**")
            with col2:
                st.write(item['milestone'])
            with col3:
                st.write(item['status'])
    
    with tab4:
        st.subheader("🌟 Prix et Reconnaissances")
        
        st.write("### 🏅 Prix Nobel de Physique Liés aux Accélérateurs")
        
        nobel_prizes = [
            {
                "year": 2013,
                "laureates": "François Englert, Peter Higgs",
                "discovery": "Mécanisme de Higgs",
                "accelerator": "LHC (CERN)",
                "citation": "Pour la découverte théorique d'un mécanisme contribuant à notre compréhension de l'origine de la masse"
            },
            {
                "year": 2008,
                "laureates": "Yoichiro Nambu, Makoto Kobayashi, Toshihide Maskawa",
                "discovery": "Brisure de symétrie, Violation CP",
                "accelerator": "KEK, Belle",
                "citation": "Pour la découverte de l'origine de la brisure de symétrie"
            },
            {
                "year": 2004,
                "laureates": "David Gross, David Politzer, Frank Wilczek",
                "discovery": "Liberté asymptotique (QCD)",
                "accelerator": "SLAC",
                "citation": "Pour la découverte de la liberté asymptotique dans la théorie des interactions fortes"
            },
            {
                "year": 1984,
                "laureates": "Carlo Rubbia, Simon van der Meer",
                "discovery": "Bosons W et Z",
                "accelerator": "SPS (CERN)",
                "citation": "Pour leurs contributions décisives au projet qui a mené à la découverte des bosons W et Z"
            },
            {
                "year": 1976,
                "laureates": "Burton Richter, Samuel Ting",
                "discovery": "Particule J/ψ",
                "accelerator": "SLAC, BNL",
                "citation": "Pour leur travail pionnier dans la découverte d'une particule élémentaire lourde"
            }
        ]
        
        for prize in nobel_prizes:
            with st.expander(f"🏅 {prize['year']}: {prize['discovery']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Lauréats:** {prize['laureates']}")
                    st.write(f"**Découverte:** {prize['discovery']}")
                    st.write(f"**Accélérateur:** {prize['accelerator']}")
                    st.write(f"**Citation:** *{prize['citation']}*")
                
                with col2:
                    st.image("https://via.placeholder.com/150x150/gold/white?text=Nobel", width=150)
        
        st.markdown("---")
        
        # Autres distinctions
        st.write("### 🏆 Autres Distinctions Majeures")
        
        other_awards = [
            {"award": "Breakthrough Prize in Fundamental Physics", "year": 2013, "recipients": "ATLAS & CMS Collaborations"},
            {"award": "Wolf Prize in Physics", "year": 2004, "recipients": "Robert Brout, François Englert, Peter Higgs"},
            {"award": "European Physical Society Prize", "year": 2017, "recipients": "ATLAS & CMS Higgs Discoverers"},
            {"award": "Dirac Medal", "year": 2010, "recipients": "Guido Altarelli, Yuri Dokshitzer, Lev Lipatov"}
        ]
        
        for award in other_awards:
            col1, col2, col3 = st.columns([3, 1, 2])
            
            with col1:
                st.write(f"**{award['award']}**")
            with col2:
                st.write(award['year'])
            with col3:
                st.write(award['recipients'])
        
        st.markdown("---")
        
        # Impact scientifique
        st.write("### 📊 Impact Scientifique")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Publications Totales", "50,000+")
        with col2:
            st.metric("Citations", "2,000,000+")
        with col3:
            st.metric("Prix Nobel", "10+")
        with col4:
            st.metric("Collaborations", "150+")
        
        # Graphique citations par an
        years_impact = np.arange(2010, 2025)
        citations_per_year = np.array([50000, 60000, 80000, 150000, 180000, 160000, 140000, 
                                       120000, 110000, 105000, 100000, 98000, 95000, 93000, 90000])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years_impact,
            y=citations_per_year,
            marker_color='rgba(102, 126, 234, 0.7)',
            text=citations_per_year,
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Citations Annuelles des Publications",
            xaxis_title="Année",
            yaxis_title="Nombre de Citations",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Hall of Fame
        st.write("### 🌟 Hall of Fame - Physiciens Notables")
        
        physicists = [
            {"name": "Peter Higgs", "contribution": "Mécanisme de Higgs", "years": "1929-2024"},
            {"name": "François Englert", "contribution": "Boson de Higgs", "years": "1932-"},
            {"name": "Carlo Rubbia", "contribution": "Bosons W et Z", "years": "1934-"},
            {"name": "Leon Lederman", "contribution": "Neutrino muonique", "years": "1922-2018"},
            {"name": "Murray Gell-Mann", "contribution": "Quarks", "years": "1929-2019"},
            {"name": "Richard Feynman", "contribution": "QED", "years": "1918-1988"}
        ]
        
        for physicist in physicists:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"⭐ **{physicist['name']}**")
            with col2:
                st.write(physicist['contribution'])
            with col3:
                st.write(physicist['years'])

# ==================== FOOTER FINAL ====================

st.markdown("---")

with st.expander("📜 Journal des Événements (Dernières 10 entrées)"):
    if st.session_state.accelerator_system['log']:
        for event in st.session_state.accelerator_system['log'][-10:][::-1]:
            timestamp = event['timestamp'][:19]
            st.text(f"{timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")
    
    if st.button("🗑️ Effacer le Journal", key="clear_log_final"):
        st.session_state.accelerator_system['log'] = []
        st.rerun()

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>⚛️ Plateforme Accélérateur de Particules</h3>
        <p>Système Complet de Création, Simulation, Test et Analyse</p>
        <p><small>Version 1.0.0 | Physique des Hautes Énergies</small></p>
        <p><small>⚛️ Conception | 🏭 Fabrication | 🔧 Tests | 💥 Collisions</small></p>
        <p><small>📊 Analyses | 🌌 Découvertes | 💰 Budget | 📚 Formation</small></p>
        <p><small>Powered by Quantum Physics & Artificial Intelligence</small></p>
        <p><small>© 2024 - Tous droits réservés</small></p>
    </div>
""", unsafe_allow_html=True)