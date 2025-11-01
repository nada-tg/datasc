"""
⚛️ Nuclear Fusion Laboratory Platform - Complete Frontend
Réacteurs • Plasma • Tokamaks • Confinement Magnétique • Fusion Control

Installation:
pip install streamlit pandas plotly numpy scipy

Lancement:
streamlit run fusion_nuclear_lab_app.py
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

# ==================== CONFIGURATION PAGE ====================
st.set_page_config(
    page_title="⚛️ Nuclear Fusion Lab",
    page_icon="⚛️",
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
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 30%, #FDC830 60%, #F37335 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: fusion-glow 2s ease-in-out infinite alternate;
    }
    @keyframes fusion-glow {
        from { filter: drop-shadow(0 0 20px #FF6B35); }
        to { filter: drop-shadow(0 0 40px #F37335); }
    }
    .fusion-card {
        border: 3px solid #FF6B35;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, rgba(243, 115, 53, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.4);
        transition: all 0.3s;
    }
    .fusion-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(247, 147, 30, 0.6);
    }
    .plasma-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4);
    }
    .reactor-active {
        animation: reactor-pulse 1s infinite;
    }
    @keyframes reactor-pulse {
        0%, 100% { opacity: 0.8; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES PHYSIQUES ====================
PHYSICS_CONSTANTS = {
    # Constantes fondamentales
    'c': 299792458,  # Vitesse lumière (m/s)
    'k_B': 1.380649e-23,  # Constante Boltzmann (J/K)
    'e': 1.602176634e-19,  # Charge électron (C)
    'epsilon_0': 8.8541878128e-12,  # Permittivité vide (F/m)
    'mu_0': 1.25663706212e-6,  # Perméabilité vide (H/m)
    
    # Masses atomiques (kg)
    'mass_deuterium': 3.344e-27,
    'mass_tritium': 5.008e-27,
    'mass_helium': 6.646e-27,
    'mass_neutron': 1.675e-27,
    'mass_proton': 1.673e-27,
    
    # Énergies réactions fusion (MeV)
    'energy_DT': 17.6,  # D + T → He + n
    'energy_DD': 3.27,  # D + D → T + p
    'energy_DHe3': 18.3,  # D + He3 → He4 + p
    
    # Paramètres plasma
    'ion_temperature_keV': 15,  # Température ions (keV)
    'electron_temperature_keV': 10,  # Température électrons (keV)
    'density_m3': 1e20,  # Densité plasma (m^-3)
    'confinement_time_s': 3,  # Temps confinement (s)
    
    # Critère Lawson (pour ignition)
    'lawson_criterion': 3e21,  # n*τ*T (m^-3·s·keV)
    
    # Champs magnétiques
    'toroidal_field_T': 5.3,  # Champ toroïdal (Tesla)
    'poloidal_field_T': 0.5,  # Champ poloïdal (Tesla)
    
    # Géométrie tokamak
    'major_radius_m': 6.2,  # Rayon majeur (m)
    'minor_radius_m': 2.0,  # Rayon mineur (m)
    'aspect_ratio': 3.1,  # A = R/a
    'plasma_current_MA': 15,  # Courant plasma (MA)
}

REACTOR_TYPES = {
    'Tokamak': {
        'description': 'Confinement magnétique toroïdal',
        'confinement': 'Magnétique',
        'geometry': 'Toroïdale',
        'examples': 'ITER, JET, SPARC',
        'q_factor': 3.0,
        'beta_limit': 0.025,
        'color': '#FF6B35'
    },
    'Stellarator': {
        'description': 'Confinement magnétique avec bobines hélicoïdales',
        'confinement': 'Magnétique',
        'geometry': 'Hélicoïdale',
        'examples': 'Wendelstein 7-X, LHD',
        'q_factor': 2.5,
        'beta_limit': 0.05,
        'color': '#F7931E'
    },
    'Laser ICF': {
        'description': 'Fusion par confinement inertiel laser',
        'confinement': 'Inertiel',
        'geometry': 'Sphérique',
        'examples': 'NIF, LMJ',
        'q_factor': None,
        'beta_limit': None,
        'color': '#FDC830'
    },
    'Z-Pinch': {
        'description': 'Confinement par compression magnétique',
        'confinement': 'Magnétique pulsé',
        'geometry': 'Cylindrique',
        'examples': 'Z Machine',
        'q_factor': None,
        'beta_limit': 0.1,
        'color': '#F37335'
    }
}

FUSION_REACTIONS = {
    'D-T': {
        'formula': 'D + T → He-4 + n',
        'energy_MeV': 17.6,
        'cross_section_peak_keV': 64,
        'reactivity_peak': 1.24e-24,
        'products': ['Helium-4 (3.5 MeV)', 'Neutron (14.1 MeV)'],
        'probability': 'Très élevée'
    },
    'D-D': {
        'formula': 'D + D → T + p (50%) ou D + D → He-3 + n (50%)',
        'energy_MeV': 3.27,
        'cross_section_peak_keV': 1500,
        'reactivity_peak': 9.4e-28,
        'products': ['Tritium + Proton', 'Helium-3 + Neutron'],
        'probability': 'Moyenne'
    },
    'D-He3': {
        'formula': 'D + He-3 → He-4 + p',
        'energy_MeV': 18.3,
        'cross_section_peak_keV': 200,
        'reactivity_peak': 5.7e-25,
        'products': ['Helium-4 (3.6 MeV)', 'Proton (14.7 MeV)'],
        'probability': 'Élevée (mais He-3 rare)'
    },
    'p-B11': {
        'formula': 'p + B-11 → 3 He-4',
        'energy_MeV': 8.7,
        'cross_section_peak_keV': 600,
        'reactivity_peak': 1.5e-27,
        'products': ['3 × Helium-4 (aneutronique)'],
        'probability': 'Faible (haute température)'
    }
}

# ==================== INITIALISATION SESSION STATE ====================
if 'fusion_lab' not in st.session_state:
    st.session_state.fusion_lab = {
        'reactors': {},
        'plasma_shots': [],
        'experiments': [],
        'diagnostics': [],
        'heating_systems': {},
        'magnets': {},
        'fuel_inventory': {
            'deuterium_kg': 1000,
            'tritium_g': 500,
            'helium3_g': 10
        },
        'safety_systems': {},
        'simulations': [],
        'maintenance_log': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str, level: str = "INFO"):
    """Enregistrer événement"""
    st.session_state.fusion_lab['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_fusion_power(n: float, T: float, reaction: str = 'D-T') -> float:
    """Calculer puissance fusion (W/m³)"""
    # Réactivité <σv> (m³/s)
    T_keV = T / 1000  # Conversion eV → keV
    
    if reaction == 'D-T':
        # Formule simplifiée pour D-T
        if T_keV < 1:
            reactivity = 1e-30
        else:
            reactivity = 1.1e-24 * (T_keV**2) / (1 + (T_keV/25)**3)
    elif reaction == 'D-D':
        reactivity = 2.33e-14 * (T_keV**(-2/3)) * np.exp(-18.76 * T_keV**(-1/3))
    else:
        reactivity = 1e-24  # Approximation
    
    # Puissance fusion: P = n² <σv> E / 4
    energy_per_reaction = FUSION_REACTIONS[reaction]['energy_MeV'] * 1.602e-13  # MeV → J
    power_density = 0.25 * n**2 * reactivity * energy_per_reaction
    
    return power_density

def calculate_triple_product(n: float, T: float, tau: float) -> float:
    """Calculer produit triple de Lawson (m^-3·s·keV)"""
    T_keV = T / 1000
    return n * tau * T_keV

def calculate_beta(n: float, T: float, B: float) -> float:
    """Calculer paramètre beta (pression plasma / pression magnétique)"""
    # Pression plasma: p = n*k_B*T
    p_plasma = n * PHYSICS_CONSTANTS['k_B'] * T
    
    # Pression magnétique: p_B = B²/(2*μ₀)
    p_magnetic = B**2 / (2 * PHYSICS_CONSTANTS['mu_0'])
    
    return p_plasma / p_magnetic

def calculate_q_factor(P_fusion: float, P_heating: float) -> float:
    """Calculer facteur Q (gain fusion)"""
    if P_heating == 0:
        return 0
    return P_fusion / P_heating

def simulate_plasma_evolution(duration_s: float, n0: float, T0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simuler évolution temporelle du plasma"""
    dt = 0.01
    steps = int(duration_s / dt)
    
    time = np.linspace(0, duration_s, steps)
    n = np.zeros(steps)
    T = np.zeros(steps)
    
    n[0] = n0
    T[0] = T0
    
    # Paramètres évolution
    tau_E = PHYSICS_CONSTANTS['confinement_time_s']
    tau_particle = tau_E * 2
    
    for i in range(1, steps):
        # Pertes particules
        dn_dt = -n[i-1] / tau_particle
        
        # Pertes énergie + chauffage fusion
        P_fusion = calculate_fusion_power(n[i-1], T[i-1])
        P_loss = n[i-1] * PHYSICS_CONSTANTS['k_B'] * T[i-1] / tau_E
        dT_dt = (P_fusion - P_loss) / (1.5 * n[i-1] * PHYSICS_CONSTANTS['k_B'])
        
        n[i] = n[i-1] + dn_dt * dt
        T[i] = max(0, T[i-1] + dT_dt * dt)
    
    return time, n, T

def calculate_neutron_flux(P_fusion: float, volume: float) -> float:
    """Calculer flux neutronique (neutrons/m²/s)"""
    # Pour D-T: 80% énergie dans neutrons (14.1 MeV)
    E_neutron = 14.1 * 1.602e-13  # J
    n_neutrons_per_second = (0.8 * P_fusion) / E_neutron
    
    # Surface réacteur (approximation sphérique)
    radius = (3 * volume / (4 * np.pi))**(1/3)
    surface = 4 * np.pi * radius**2
    
    return n_neutrons_per_second / surface

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">⚛️ Nuclear Fusion Laboratory</h1>', 
           unsafe_allow_html=True)
st.markdown("### Plasma Physics • Tokamaks • Magnetic Confinement • Fusion Energy")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/FF6B35/FFFFFF?text=FusionLab", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Lab Central",
            "⚛️ Créer Réacteur",
            "🔥 Plasma Control",
            "🧲 Champs Magnétiques",
            "🔋 Systèmes Chauffage",
            "💥 Tir Plasma",
            "📊 Diagnostics",
            "⚡ Fusion Reactions",
            "🎯 Confinement",
            "🔬 Expériences",
            "💻 Simulations",
            "📈 Performance",
            "🛡️ Sécurité",
            "⚙️ Maintenance",
            "📊 Analytics",
            "📡 Monitoring Live",
            "🌍 ITER Database",
            "📚 Physics Library",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 État Lab")
    
    total_reactors = len(st.session_state.fusion_lab['reactors'])
    total_shots = len(st.session_state.fusion_lab['plasma_shots'])
    total_experiments = len(st.session_state.fusion_lab['experiments'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚛️ Réacteurs", total_reactors)
        st.metric("💥 Tirs", total_shots)
    with col2:
        st.metric("🔬 Expériences", total_experiments)
        st.metric("📊 Diagnostics", len(st.session_state.fusion_lab['diagnostics']))

# ==================== PAGE: LAB CENTRAL ====================
if page == "🏠 Lab Central":
    st.header("🏠 Laboratoire Fusion Central")
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="fusion-card"><h2>⚛️</h2><h3>{total_reactors}</h3><p>Réacteurs</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        avg_Q = 0.65 if total_reactors > 0 else 0
        st.markdown(f'<div class="fusion-card"><h2>Q</h2><h3>{avg_Q:.2f}</h3><p>Gain Moyen</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="fusion-card"><h2>💥</h2><h3>{total_shots}</h3><p>Tirs Plasma</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        total_energy_MJ = total_shots * np.random.uniform(50, 200)
        st.markdown(f'<div class="fusion-card"><h2>⚡</h2><h3>{total_energy_MJ:.0f}</h3><p>Énergie (MJ)</p></div>', 
                   unsafe_allow_html=True)
    
    with col5:
        uptime = 94.5 if total_reactors > 0 else 0
        st.markdown(f'<div class="fusion-card"><h2>✓</h2><h3>{uptime:.1f}%</h3><p>Disponibilité</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Types de réacteurs
    st.subheader("⚛️ Types de Réacteurs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔬 Technologies Disponibles")
        
        for rtype, info in REACTOR_TYPES.items():
            with st.expander(f"⚛️ {rtype}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Confinement:** {info['confinement']}")
                st.write(f"**Géométrie:** {info['geometry']}")
                st.write(f"**Exemples:** {info['examples']}")
                if info['q_factor']:
                    st.write(f"**Facteur q:** {info['q_factor']}")
                if info['beta_limit']:
                    st.write(f"**Limite β:** {info['beta_limit']}")
    
    with col2:
        st.write("### 📊 Répartition")
        
        fig = go.Figure(data=[go.Pie(
            labels=list(REACTOR_TYPES.keys()),
            values=[1, 1, 1, 1],  # Equal distribution for display
            marker=dict(colors=[info['color'] for info in REACTOR_TYPES.values()]),
            hole=0.4
        )])
        
        fig.update_layout(
            title="Technologies Fusion",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Critère Lawson
    st.subheader("📊 Critère de Lawson (Ignition)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        **Critère de Lawson pour ignition plasma:**
        
        n · τ · T ≥ 3×10²¹ m⁻³·s·keV
        
        Où:
        - **n**: Densité plasma (m⁻³)
        - **τ**: Temps confinement énergie (s)
        - **T**: Température ions (keV)
        """)
        
        # Simuler progression vers ignition
        densities = np.logspace(19, 21, 50)
        temps_conf = np.linspace(0.1, 5, 50)
        
        fig = go.Figure()
        
        for T in [5, 10, 15, 20]:
            triple_products = []
            for n, tau in zip(densities, temps_conf):
                tp = calculate_triple_product(n, T*1000, tau)
                triple_products.append(tp)
            
            fig.add_trace(go.Scatter(
                x=list(range(50)),
                y=triple_products,
                mode='lines',
                name=f'T = {T} keV',
                line=dict(width=3)
            ))
        
        fig.add_hline(y=PHYSICS_CONSTANTS['lawson_criterion'],
                     line_dash="dash", line_color="white",
                     annotation_text="Seuil Ignition")
        
        fig.update_layout(
            title="Évolution Produit Triple de Lawson",
            xaxis_title="Progression",
            yaxis_title="n·τ·T (m⁻³·s·keV)",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("### 🎯 Objectifs")
        
        st.metric("Seuil Ignition", "3×10²¹")
        st.metric("T Optimal", "15-20 keV")
        st.metric("τ Requis", "3-5 s")
        st.metric("n Typique", "10²⁰ m⁻³")
        
        st.write("### 🏆 Records")
        st.write("**JET (1997):** Q = 0.67")
        st.write("**NIF (2022):** Q = 1.5 (ICF)")
        st.write("**ITER (proj.):** Q = 10")

# ==================== PAGE: CRÉER RÉACTEUR ====================
elif page == "⚛️ Créer Réacteur":
    st.header("⚛️ Créer Réacteur Fusion")
    
    st.info("""
    **Conception Réacteur**
    
    Configurez votre réacteur de fusion selon les paramètres plasma et géométriques.
    """)
    
    with st.form("create_reactor"):
        col1, col2 = st.columns(2)
        
        with col1:
            reactor_name = st.text_input("Nom Réacteur", "FUSION-R1")
            
            reactor_type = st.selectbox("Type Réacteur",
                list(REACTOR_TYPES.keys()))
            
            fuel_type = st.selectbox("Combustible",
                ["D-T", "D-D", "D-He3", "p-B11"])
            
            major_radius = st.slider("Rayon Majeur R (m)", 1.0, 15.0, 6.2, 0.1)
            minor_radius = st.slider("Rayon Mineur a (m)", 0.5, 5.0, 2.0, 0.1)
        
        with col2:
            toroidal_field = st.slider("Champ Toroïdal B_T (T)", 1.0, 10.0, 5.3, 0.1)
            plasma_current = st.slider("Courant Plasma I_p (MA)", 5.0, 25.0, 15.0, 0.5)
            
            target_density = st.number_input("Densité Cible (×10²⁰ m⁻³)", 
                0.5, 5.0, 1.0, 0.1) * 1e20
            
            target_temperature = st.slider("Température Cible (keV)", 5, 30, 15)
            
            conf_time = st.slider("Temps Confinement τ_E (s)", 0.5, 10.0, 3.0, 0.1)
        
        st.write("### 🔧 Systèmes Auxiliaires")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            heating_power = st.number_input("Puissance Chauffage (MW)", 10, 200, 50)
            heating_methods = st.multiselect("Méthodes Chauffage",
                ["NBI (Neutral Beam)", "ICRH (Radio-fréquence)", "ECRH (Micro-ondes)"],
                default=["NBI (Neutral Beam)", "ICRH (Radio-fréquence)"])
        
        with col2:
            first_wall_material = st.selectbox("Matériau Première Paroi",
                ["Beryllium", "Tungsten", "Carbon", "Liquid Lithium"])
            
            divertor_type = st.selectbox("Type Divertor",
                ["Single-null", "Double-null", "Super-X"])
        
        with col3:
            blanket_type = st.selectbox("Couverture Tritium",
                ["Lithium Lead", "Ceramic Breeder", "Liquid Lithium"])
            
            vacuum_vessel = st.checkbox("Enceinte Vide", value=True)
        
        if st.form_submit_button("⚛️ Créer Réacteur", type="primary"):
            reactor_id = f"reactor_{len(st.session_state.fusion_lab['reactors']) + 1}"
            
            # Calculs caractéristiques
            aspect_ratio = major_radius / minor_radius
            volume = 2 * np.pi**2 * major_radius * minor_radius**2
            
            # Calcul Q factor estimé
            P_fusion_est = calculate_fusion_power(target_density, target_temperature*1000, fuel_type) * volume
            Q_factor_est = calculate_q_factor(P_fusion_est, heating_power * 1e6)
            
            # Triple produit
            triple_product = calculate_triple_product(target_density, target_temperature*1000, conf_time)
            
            # Beta
            beta = calculate_beta(target_density, target_temperature*1000*PHYSICS_CONSTANTS['e'], toroidal_field)
            
            reactor = {
                'id': reactor_id,
                'name': reactor_name,
                'type': reactor_type,
                'fuel_type': fuel_type,
                'major_radius_m': major_radius,
                'minor_radius_m': minor_radius,
                'aspect_ratio': aspect_ratio,
                'toroidal_field_T': toroidal_field,
                'plasma_current_MA': plasma_current,
                'target_density_m3': target_density,
                'target_temperature_keV': target_temperature,
                'confinement_time_s': conf_time,
                'volume_m3': volume,
                'heating_power_MW': heating_power,
                'heating_methods': heating_methods,
                'first_wall_material': first_wall_material,
                'divertor_type': divertor_type,
                'blanket_type': blanket_type,
                'Q_factor_est': Q_factor_est,
                'triple_product': triple_product,
                'beta': beta,
                'status': 'offline',
                'created_at': datetime.now().isoformat()
            }
            
            st.session_state.fusion_lab['reactors'][reactor_id] = reactor
            log_event(f"Réacteur créé: {reactor_name}", "SUCCESS")
            
            st.success(f"✅ Réacteur '{reactor_name}' créé!")
            st.balloons()
            
            # Afficher caractéristiques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Volume", f"{volume:.1f} m³")
            with col2:
                st.metric("Q Factor (est.)", f"{Q_factor_est:.2f}")
            with col3:
                st.metric("Produit Triple", f"{triple_product:.2e}")
            with col4:
                ignition = "✅ OUI" if triple_product >= PHYSICS_CONSTANTS['lawson_criterion'] else "❌ NON"
                st.metric("Ignition?", ignition)
            
            st.rerun()

# ==================== PAGE: PLASMA CONTROL ====================
elif page == "🔥 Plasma Control":
    st.header("🔥 Contrôle Plasma")
    
    if not st.session_state.fusion_lab['reactors']:
        st.warning("⚠️ Aucun réacteur créé")
    else:
        selected_reactor = st.selectbox("Réacteur",
            list(st.session_state.fusion_lab['reactors'].keys()),
            format_func=lambda x: st.session_state.fusion_lab['reactors'][x]['name'])
        
        reactor = st.session_state.fusion_lab['reactors'][selected_reactor]
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 État Plasma", "🎛️ Contrôle", "📈 Évolution", "🔥 Ignition"])
        
        with tab1:
            st.subheader("📊 État Actuel du Plasma")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("### 🌡️ Température")
                T_ion = reactor['target_temperature_keV']
                T_electron = T_ion * 0.8
                
                st.metric("Ions", f"{T_ion} keV", f"{T_ion*11.6} MK")
                st.metric("Électrons", f"{T_electron:.1f} keV")
                
                if T_ion >= 10:
                    st.success("✅ Température fusion atteinte")
                else:
                    st.warning("⚠️ Chauffage requis")
            
            with col2:
                st.write("### 📦 Densité")
                n = reactor['target_density_m3']
                
                st.metric("Plasma", f"{n:.2e} m⁻³")
                st.metric("Pression", f"{n*PHYSICS_CONSTANTS['k_B']*T_ion*1000:.2e} Pa")
                
                # Rapport Greenwald
                I_p = reactor['plasma_current_MA']
                a = reactor['minor_radius_m']
                n_greenwald = I_p * 1e6 / (np.pi * a**2)
                ratio = n / n_greenwald
                
                st.metric("n/n_G", f"{ratio:.2f}")
                
                if ratio < 0.8:
                    st.success("✅ Sous limite Greenwald")
                else:
                    st.error("❌ Risque disruption")
            
            with col3:
                st.write("### ⚡ Puissance")
                
                P_fusion = calculate_fusion_power(n, T_ion*1000, reactor['fuel_type'])
                P_total = P_fusion * reactor['volume_m3'] / 1e6  # MW
                
                st.metric("Fusion", f"{P_total:.1f} MW")
                st.metric("Chauffage", f"{reactor['heating_power_MW']} MW")
                
                Q = reactor['Q_factor_est']
                st.metric("Q Factor", f"{Q:.2f}")
                
                if Q > 1:
                    st.success("🎉 Gain net!")
                elif Q > 0.5:
                    st.info("📈 Proche breakeven")
                else:
                    st.warning("⚠️ Chauffage nécessaire")
            
            # Profils radiaux
            st.write("### 📊 Profils Radiaux")
            
            r = np.linspace(0, reactor['minor_radius_m'], 100)
            r_norm = r / reactor['minor_radius_m']
            
            # Profil température (parabolique)
            T_profile = T_ion * (1 - r_norm**2)**2
            
            # Profil densité (peaked)
            n_profile = n * (1 - r_norm**2)**1.5
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Température", "Densité")
            )
            
            fig.add_trace(go.Scatter(
                x=r, y=T_profile,
                mode='lines',
                line=dict(color='#FF6B35', width=3),
                name='T_ion'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=r, y=n_profile,
                mode='lines',
                line=dict(color='#F7931E', width=3),
                name='n_e'
            ), row=1, col=2)
            
            fig.update_xaxes(title_text="Rayon (m)", row=1, col=1)
            fig.update_xaxes(title_text="Rayon (m)", row=1, col=2)
            fig.update_yaxes(title_text="T (keV)", row=1, col=1)
            fig.update_yaxes(title_text="n (m⁻³)", row=1, col=2)
            
            fig.update_layout(
                template="plotly_dark",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🎛️ Contrôle Temps Réel")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🔋 Chauffage")
                
                nbi_power = st.slider("NBI Power (MW)", 0, 50, 20)
                icrh_power = st.slider("ICRH Power (MW)", 0, 30, 15)
                ecrh_power = st.slider("ECRH Power (MW)", 0, 20, 10)
                
                total_heating = nbi_power + icrh_power + ecrh_power
                st.metric("Puissance Totale", f"{total_heating} MW")
                
                if st.button("🔥 Appliquer Chauffage", type="primary"):
                    st.success(f"✅ Chauffage appliqué: {total_heating} MW")
                    log_event(f"Chauffage: {total_heating} MW sur {reactor['name']}", "INFO")
            
            with col2:
                st.write("### ⚙️ Confinement")
                
                plasma_current_control = st.slider("Courant Plasma (MA)", 
                    5.0, reactor['plasma_current_MA']*1.5, reactor['plasma_current_MA'])
                
                q95 = st.slider("q₉₅ (facteur sécurité)", 2.0, 5.0, 3.0, 0.1)
                
                beta_N = st.slider("β_N (beta normalisé)", 1.0, 4.0, 2.5, 0.1)
                
                if st.button("⚙️ Ajuster Confinement"):
                    st.success("✅ Paramètres confinement ajustés")
                    reactor['plasma_current_MA'] = plasma_current_control
                    log_event(f"Confinement ajusté: I_p={plasma_current_control} MA", "INFO")
            
            st.write("### 🎯 Contrôle Position")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                delta_R = st.slider("Δ Position Radiale (cm)", -10, 10, 0)
            
            with col2:
                delta_Z = st.slider("Δ Position Verticale (cm)", -10, 10, 0)
            
            with col3:
                triangularity = st.slider("Triangularité δ", 0.0, 0.5, 0.3, 0.05)
            
            if st.button("🎯 Appliquer Position"):
                st.success("✅ Position plasma ajustée")
                st.info(f"ΔR = {delta_R} cm, ΔZ = {delta_Z} cm")
        
        with tab3:
            st.subheader("📈 Évolution Temporelle")
            
            duration = st.slider("Durée Simulation (s)", 1, 30, 10)
            
            if st.button("▶️ Simuler Évolution", type="primary"):
                with st.spinner("Simulation en cours..."):
                    import time
                    time.sleep(1)
                    
                    n0 = reactor['target_density_m3']
                    T0 = reactor['target_temperature_keV'] * 1000
                    
                    t, n_t, T_t = simulate_plasma_evolution(duration, n0, T0)
                    
                    # Calculer puissance fusion
                    P_fusion_t = [calculate_fusion_power(n, T, reactor['fuel_type']) * reactor['volume_m3'] / 1e6 
                                  for n, T in zip(n_t, T_t)]
                    
                    # Graphiques
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=("Densité", "Température", "Puissance Fusion")
                    )
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=n_t,
                        mode='lines',
                        line=dict(color='#FF6B35', width=2),
                        name='n'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=T_t/1000,
                        mode='lines',
                        line=dict(color='#F7931E', width=2),
                        name='T'
                    ), row=2, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=P_fusion_t,
                        mode='lines',
                        line=dict(color='#FDC830', width=2),
                        name='P_fusion'
                    ), row=3, col=1)
                    
                    fig.update_xaxes(title_text="Temps (s)", row=3, col=1)
                    fig.update_yaxes(title_text="n (m⁻³)", row=1, col=1)
                    fig.update_yaxes(title_text="T (keV)", row=2, col=1)
                    fig.update_yaxes(title_text="P (MW)", row=3, col=1)
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=800,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Densité Finale", f"{n_t[-1]:.2e} m⁻³")
                    with col2:
                        st.metric("Température Finale", f"{T_t[-1]/1000:.1f} keV")
                    with col3:
                        st.metric("Puissance Max", f"{max(P_fusion_t):.1f} MW")
        
        with tab4:
            st.subheader("🔥 Chemin vers Ignition")
            
            st.write("""
            **Conditions Ignition:**
            
            Le plasma atteint l'ignition quand le chauffage par fusion alpha maintient 
            la température sans chauffage externe.
            
            Critères:
            - Q → ∞ (théorique) ou Q > 5 (pratique)
            - n·τ·T > 3×10²¹ m⁻³·s·keV
            - β < β_limite
            - MHD stable
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 État Actuel")
                
                current_triple = reactor['triple_product']
                progress = (current_triple / PHYSICS_CONSTANTS['lawson_criterion']) * 100
                
                st.progress(min(progress/100, 1.0))
                st.write(f"**Progression:** {progress:.1f}%")
                
                st.metric("Produit Triple Actuel", f"{current_triple:.2e}")
                st.metric("Seuil Ignition", f"{PHYSICS_CONSTANTS['lawson_criterion']:.2e}")
                
                deficit = PHYSICS_CONSTANTS['lawson_criterion'] - current_triple
                if deficit > 0:
                    st.warning(f"⚠️ Déficit: {deficit:.2e}")
                else:
                    st.success("🎉 IGNITION ATTEINTE!")
            
            with col2:
                st.write("### 🎯 Scénarios")
                
                scenarios = {
                    "Augmenter T (+5 keV)": {
                        'n': reactor['target_density_m3'],
                        'T': (reactor['target_temperature_keV'] + 5) * 1000,
                        'tau': reactor['confinement_time_s']
                    },
                    "Augmenter n (+50%)": {
                        'n': reactor['target_density_m3'] * 1.5,
                        'T': reactor['target_temperature_keV'] * 1000,
                        'tau': reactor['confinement_time_s']
                    },
                    "Augmenter τ (+2s)": {
                        'n': reactor['target_density_m3'],
                        'T': reactor['target_temperature_keV'] * 1000,
                        'tau': reactor['confinement_time_s'] + 2
                    },
                    "Optimisation Totale": {
                        'n': reactor['target_density_m3'] * 1.3,
                        'T': (reactor['target_temperature_keV'] + 3) * 1000,
                        'tau': reactor['confinement_time_s'] + 1.5
                    }
                }
                
                for scenario_name, params in scenarios.items():
                    tp = calculate_triple_product(params['n'], params['T'], params['tau'])
                    ratio = tp / PHYSICS_CONSTANTS['lawson_criterion']
                    
                    with st.expander(f"📋 {scenario_name}"):
                        st.write(f"**Produit Triple:** {tp:.2e}")
                        st.write(f"**Ratio:** {ratio:.2f}")
                        
                        if ratio >= 1.0:
                            st.success("✅ Ignition atteinte!")
                        else:
                            st.info(f"📊 {ratio*100:.0f}% du seuil")

# ==================== PAGE: CHAMPS MAGNÉTIQUES ====================
elif page == "🧲 Champs Magnétiques":
    st.header("🧲 Champs Magnétiques & Confinement")
    
    st.info("""
    **Confinement Magnétique**
    
    Les champs magnétiques confinent le plasma chaud en forçant les particules 
    chargées à suivre des lignes de champ hélicoïdales.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🧲 Configuration", "📊 Topologie", "⚙️ Bobines"])
    
    with tab1:
        st.subheader("🧲 Configuration Champs")
        
        if st.session_state.fusion_lab['reactors']:
            selected_reactor = st.selectbox("Réacteur",
                list(st.session_state.fusion_lab['reactors'].keys()),
                format_func=lambda x: st.session_state.fusion_lab['reactors'][x]['name'],
                key="mag_reactor")
            
            reactor = st.session_state.fusion_lab['reactors'][selected_reactor]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🔵 Champ Toroïdal B_T")
                
                B_T = reactor['toroidal_field_T']
                st.metric("Intensité", f"{B_T} T")
                
                # Énergie magnétique
                R = reactor['major_radius_m']
                a = reactor['minor_radius_m']
                volume = 2 * np.pi**2 * R * a**2
                
                E_mag = (B_T**2 / (2 * PHYSICS_CONSTANTS['mu_0'])) * volume / 1e6  # MJ
                st.metric("Énergie Stockée", f"{E_mag:.0f} MJ")
                
                # Visualisation champ toroïdal
                theta = np.linspace(0, 2*np.pi, 100)
                phi = np.linspace(0, 2*np.pi, 50)
                
                fig = go.Figure()
                
                for p in np.linspace(0, 2*np.pi, 8):
                    x = (R + a*np.cos(theta)) * np.cos(p)
                    y = (R + a*np.cos(theta)) * np.sin(p)
                    z = a * np.sin(theta)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        line=dict(color='#FF6B35', width=3),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Lignes de Champ Toroïdal",
                    template="plotly_dark",
                    height=400,
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        zaxis_title="Z (m)"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### 🟢 Champ Poloïdal B_P")
                
                I_p = reactor['plasma_current_MA']
                B_P = (PHYSICS_CONSTANTS['mu_0'] * I_p * 1e6) / (2 * np.pi * a)
                
                st.metric("Intensité", f"{B_P:.2f} T")
                st.metric("Courant Plasma", f"{I_p} MA")
                
                # Facteur de sécurité q
                q = (a * B_T) / (R * B_P)
                st.metric("Facteur q", f"{q:.2f}")
                
                if q > 2:
                    st.success("✅ Stable MHD (q > 2)")
                else:
                    st.error("❌ Instable! Augmenter I_p")
                
                # Profil q
                r_norm = np.linspace(0, 1, 100)
                q_profile = q * (1 + 1.5 * r_norm**2)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=r_norm, y=q_profile,
                    mode='lines',
                    line=dict(color='#F7931E', width=3),
                    fill='tozeroy'
                ))
                
                fig.add_hline(y=1, line_dash="dash", line_color="red",
                             annotation_text="q=1 (sawteeth)")
                fig.add_hline(y=2, line_dash="dash", line_color="yellow",
                             annotation_text="q=2 (modes m=2)")
                
                fig.update_layout(
                    title="Profil Facteur de Sécurité q(r)",
                    xaxis_title="r/a",
                    yaxis_title="q",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Créez un réacteur")
    
    with tab2:
        st.subheader("📊 Topologie Magnétique")
        
        st.write("""
        **Surfaces Magnétiques**
        
        Le plasma est organisé en surfaces magnétiques fermées concentriques.
        Les lignes de champ s'enroulent autour du tore en formant des surfaces.
        """)
        
        # Visualisation surfaces magnétiques
        n_surfaces = st.slider("Nombre surfaces", 3, 20, 10)
        
        if st.button("🎨 Visualiser Topologie"):
            R = 6.2
            a = 2.0
            
            fig = go.Figure()
            
            for i, r_surface in enumerate(np.linspace(0.2*a, 0.9*a, n_surfaces)):
                theta = np.linspace(0, 2*np.pi, 200)
                
                # Sections poloïdales
                for phi in np.linspace(0, 2*np.pi, 12):
                    x = (R + r_surface*np.cos(theta)) * np.cos(phi)
                    y = (R + r_surface*np.cos(theta)) * np.sin(phi)
                    z = r_surface * np.sin(theta)
                    
                    color_intensity = i / n_surfaces
                    color = f'rgb({int(255*color_intensity)}, {int(107*(1-color_intensity))}, {int(53*(1-color_intensity))})'
                    
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False,
                        opacity=0.7
                    ))
            
            fig.update_layout(
                title="Surfaces Magnétiques Emboîtées",
                template="plotly_dark",
                height=600,
                scene=dict(
                    xaxis_title="X (m)",
                    yaxis_title="Y (m)",
                    zaxis_title="Z (m)",
                    aspectmode='data'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚙️ Système de Bobines")
        
        st.write("""
        **Bobines Magnétiques**
        
        - **TF Coils**: Bobines toroïdales (champ principal)
        - **PF Coils**: Bobines poloïdales (forme plasma)
        - **CS**: Solénoïde central (courant plasma)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🔵 Bobines Toroïdales (TF)")
            
            n_tf_coils = st.slider("Nombre bobines TF", 8, 24, 18)
            tf_current = st.slider("Courant TF (kA)", 50, 100, 68)
            
            st.write(f"**Configuration:** {n_tf_coils} bobines")
            st.write(f"**Courant total:** {n_tf_coils * tf_current / 1000:.1f} MA")
            
            # Force Lorentz
            if st.session_state.fusion_lab['reactors']:
                B_T = list(st.session_state.fusion_lab['reactors'].values())[0]['toroidal_field_T']
                force_per_coil = B_T * tf_current * 1000 * 10  # N (approximation)
                st.metric("Force/bobine", f"{force_per_coil/1e6:.1f} MN")
        
        with col2:
            st.write("### 🟢 Bobines Poloïdales (PF)")
            
            n_pf_coils = st.slider("Nombre bobines PF", 4, 12, 6)
            pf_max_current = st.slider("Courant max PF (kA)", 10, 50, 25)
            
            st.write(f"**Configuration:** {n_pf_coils} bobines")
            st.write("**Fonction:** Contrôle forme/position")
            
            st.write("### 🎯 Contrôle Position")
            feedback_gain = st.slider("Gain feedback", 0.1, 5.0, 1.0, 0.1)
            
            if st.button("🎛️ Activer Contrôle"):
                st.success("✅ Système contrôle position activé")


# ==================== PAGE: TIR PLASMA ====================
elif page == "💥 Tir Plasma":
    st.header("💥 Tir Plasma & Décharges")
    
    if not st.session_state.fusion_lab['reactors']:
        st.warning("⚠️ Créez un réacteur")
    else:
        selected_reactor = st.selectbox("Réacteur",
            list(st.session_state.fusion_lab['reactors'].keys()),
            format_func=lambda x: st.session_state.fusion_lab['reactors'][x]['name'],
            key="shot_reactor")
        
        reactor = st.session_state.fusion_lab['reactors'][selected_reactor]
        
        st.info(f"""
        **Réacteur: {reactor['name']}**
        
        Préparez et lancez une décharge plasma complète.
        """)
        
        with st.form("plasma_shot"):
            st.write("### ⚙️ Paramètres Tir")
            
            col1, col2 = st.columns(2)
            
            with col1:
                shot_duration = st.slider("Durée Décharge (s)", 1, 300, 10)
                ramp_up_time = st.slider("Temps Montée (s)", 0.5, 5.0, 2.0, 0.5)
                flat_top_time = shot_duration - ramp_up_time - 2
                
                heating_scenario = st.selectbox("Scénario Chauffage",
                    ["Progressif", "Impulsionnel", "Constant", "Optimisé"])
            
            with col2:
                target_Q = st.slider("Q Factor Cible", 0.1, 2.0, 0.65, 0.05)
                fueling_rate = st.slider("Taux Injection (Pa·m³/s)", 10, 200, 50)
                
                safety_checks = st.checkbox("Vérifications Sécurité", value=True)
            
            if st.form_submit_button("🚀 LANCER TIR", type="primary"):
                if safety_checks:
                    with st.spinner("Préparation tir..."):
                        import time
                        
                        # Séquence démarrage
                        progress = st.progress(0)
                        status = st.empty()
                        
                        stages = [
                            ("🔋 Charge condensateurs", 0.2),
                            ("🧲 Activation champs magnétiques", 0.4),
                            ("💨 Injection combustible", 0.6),
                            ("⚡ Breakdown plasma", 0.7),
                            ("🔥 Montée courant", 0.85),
                            ("🎯 Chauffage & contrôle", 1.0)
                        ]
                        
                        for stage_name, stage_progress in stages:
                            status.write(f"**{stage_name}**")
                            progress.progress(stage_progress)
                            time.sleep(0.5)
                        
                        status.success("✅ Plasma établi!")
                        time.sleep(0.5)
                        
                        # Simulation décharge
                        shot_id = f"shot_{len(st.session_state.fusion_lab['plasma_shots']) + 1}"
                        
                        # Calculs
                        n = reactor['target_density_m3']
                        T = reactor['target_temperature_keV'] * 1000
                        P_fusion = calculate_fusion_power(n, T, reactor['fuel_type']) * reactor['volume_m3']
                        
                        # Données tir
                        shot_data = {
                            'id': shot_id,
                            'reactor_id': selected_reactor,
                            'duration_s': shot_duration,
                            'ramp_up_s': ramp_up_time,
                            'flat_top_s': flat_top_time,
                            'heating_scenario': heating_scenario,
                            'target_Q': target_Q,
                            'achieved_Q': target_Q + np.random.normal(0, 0.05),
                            'max_power_MW': P_fusion / 1e6,
                            'total_energy_MJ': P_fusion * flat_top_time / 1e6,
                            'max_neutron_rate': calculate_neutron_flux(P_fusion, reactor['volume_m3']),
                            'disruption': np.random.random() > 0.95,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.session_state.fusion_lab['plasma_shots'].append(shot_data)
                        log_event(f"Tir plasma: {shot_id} (Q={shot_data['achieved_Q']:.2f})", "SUCCESS")
                        
                        st.balloons()
                        
                        # Résultats
                        st.write("### 📊 Résultats Tir")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Q Atteint", f"{shot_data['achieved_Q']:.3f}")
                        with col2:
                            st.metric("Puissance Max", f"{shot_data['max_power_MW']:.1f} MW")
                        with col3:
                            st.metric("Énergie Totale", f"{shot_data['total_energy_MJ']:.1f} MJ")
                        with col4:
                            if shot_data['disruption']:
                                st.error("❌ Disruption")
                            else:
                                st.success("✅ Succès")
                        
                        # Graphique temporel
                        t = np.linspace(0, shot_duration, 1000)
                        
                        # Courant plasma
                        I_p = np.zeros_like(t)
                        I_p[t < ramp_up_time] = reactor['plasma_current_MA'] * (t[t < ramp_up_time] / ramp_up_time)
                        I_p[(t >= ramp_up_time) & (t < ramp_up_time + flat_top_time)] = reactor['plasma_current_MA']
                        I_p[t >= ramp_up_time + flat_top_time] = reactor['plasma_current_MA'] * (1 - (t[t >= ramp_up_time + flat_top_time] - ramp_up_time - flat_top_time) / 2)
                        
                        # Puissance fusion
                        P_fus = np.zeros_like(t)
                        P_fus[(t >= ramp_up_time) & (t < ramp_up_time + flat_top_time)] = shot_data['max_power_MW']
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Courant Plasma", "Puissance Fusion")
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=t, y=I_p,
                            mode='lines',
                            line=dict(color='#FF6B35', width=3),
                            fill='tozeroy'
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=t, y=P_fus,
                            mode='lines',
                            line=dict(color='#FDC830', width=3),
                            fill='tozeroy'
                        ), row=2, col=1)
                        
                        fig.update_xaxes(title_text="Temps (s)", row=2, col=1)
                        fig.update_yaxes(title_text="I_p (MA)", row=1, col=1)
                        fig.update_yaxes(title_text="P_fusion (MW)", row=2, col=1)
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=600,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.rerun()
                else:
                    st.error("❌ Vérifications sécurité requises!")
        
        # Historique tirs
        st.markdown("---")
        st.subheader("📋 Historique Tirs")
        
        reactor_shots = [s for s in st.session_state.fusion_lab['plasma_shots'] 
                        if s['reactor_id'] == selected_reactor]
        
        if reactor_shots:
            st.write(f"**{len(reactor_shots)} tirs effectués**")
            
            for shot in reactor_shots[-5:][::-1]:
                with st.expander(f"💥 {shot['id']} - {shot['timestamp'][:19]}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Q Factor", f"{shot['achieved_Q']:.3f}")
                        st.metric("Durée", f"{shot['duration_s']} s")
                    
                    with col2:
                        st.metric("Puissance Max", f"{shot['max_power_MW']:.1f} MW")
                        st.metric("Énergie", f"{shot['total_energy_MJ']:.1f} MJ")
                    
                    with col3:
                        st.metric("Neutrons", f"{shot['max_neutron_rate']:.2e} n/m²/s")
                        if shot['disruption']:
                            st.error("❌ Disruption")
                        else:
                            st.success("✅ Nominal")
        else:
            st.info("Aucun tir effectué")

# ==================== PAGE: DIAGNOSTICS ====================
elif page == "📊 Diagnostics":
    st.header("📊 Diagnostics Plasma")
    
    st.info("""
    **Systèmes Diagnostiques**
    
    Mesure des paramètres plasma en temps réel.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔬 Actifs", "📈 Mesures", "🎯 Thomson Scattering"])
    
    with tab1:
        st.subheader("🔬 Diagnostics Disponibles")
        
        diagnostics = {
            'Thomson Scattering': {
                'mesure': 'T_e, n_e (profils)',
                'principe': 'Diffusion laser sur électrons',
                'résolution': '~10 cm spatial, 10 ms temporel',
                'gamme': '0.1-50 keV'
            },
            'Interferometry': {
                'mesure': 'n_e (ligne intégrée)',
                'principe': 'Déphasage onde EM',
                'résolution': '~1 ms temporel',
                'gamme': '10¹⁸-10²¹ m⁻³'
            },
            'ECE': {
                'mesure': 'T_e (profil radial)',
                'principe': 'Émission cyclotron électrons',
                'résolution': '~1 cm spatial, <1 μs temporel',
                'gamme': '0.1-100 keV'
            },
            'Spectroscopy': {
                'mesure': 'T_ion, v_rot, Z_eff, impuretés',
                'principe': 'Spectres raies atomiques',
                'résolution': 'Variable',
                'gamme': 'All'
            },
            'Bolometry': {
                'mesure': 'P_rad (puissance rayonnée)',
                'principe': 'Détection radiation totale',
                'résolution': 'Tomographie',
                'gamme': '0.1-1000 MW/m³'
            },
            'Neutron Detectors': {
                'mesure': 'Taux neutrons, T_ion',
                'principe': 'Détection neutrons 14.1 MeV',
                'résolution': '~10 ms',
                'gamme': '10¹⁴-10²⁰ n/s'
            },
            'Magnetics': {
                'mesure': 'I_p, β, MHD',
                'principe': 'Bobines magnétiques',
                'résolution': '<1 ms',
                'gamme': 'All'
            },
            'Soft X-ray': {
                'mesure': 'Sawteeth, MHD',
                'principe': 'Rayonnement X mou',
                'résolution': '~1 μs',
                'gamme': '0.1-20 keV'
            }
        }
        
        selected_diags = st.multiselect("Activer Diagnostics",
            list(diagnostics.keys()),
            default=list(diagnostics.keys())[:4])
        
        for diag in selected_diags:
            info = diagnostics[diag]
            with st.expander(f"📊 {diag}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Mesure:** {info['mesure']}")
                    st.write(f"**Principe:** {info['principe']}")
                
                with col2:
                    st.write(f"**Résolution:** {info['résolution']}")
                    st.write(f"**Gamme:** {info['gamme']}")
    
    with tab2:
        st.subheader("📈 Mesures Temps Réel")
        
        if st.button("📊 Acquérir Données", type="primary"):
            with st.spinner("Acquisition..."):
                import time
                time.sleep(1)
                
                # Simuler mesures
                r = np.linspace(0, 2, 50)
                
                # Thomson Scattering
                T_e = 15 * (1 - (r/2)**2)**2
                n_e = 1e20 * (1 - (r/2)**2)**1.5
                
                # ECE
                T_e_ece = T_e + np.random.normal(0, 0.5, len(T_e))
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Température Électrons", "Densité Électrons")
                )
                
                fig.add_trace(go.Scatter(
                    x=r, y=T_e,
                    mode='markers+lines',
                    name='Thomson',
                    marker=dict(color='#FF6B35', size=8)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=r, y=T_e_ece,
                    mode='lines',
                    name='ECE',
                    line=dict(color='#F7931E', dash='dash')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=r, y=n_e,
                    mode='markers+lines',
                    marker=dict(color='#FDC830', size=8),
                    showlegend=False
                ), row=1, col=2)
                
                fig.update_xaxes(title_text="Rayon (m)", row=1, col=1)
                fig.update_xaxes(title_text="Rayon (m)", row=1, col=2)
                fig.update_yaxes(title_text="T_e (keV)", row=1, col=1)
                fig.update_yaxes(title_text="n_e (m⁻³)", row=1, col=2)
                
                fig.update_layout(
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sauvegarder
                diag_data = {
                    'type': 'Thomson + ECE',
                    'T_e_profile': T_e.tolist(),
                    'n_e_profile': n_e.tolist(),
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.fusion_lab['diagnostics'].append(diag_data)
                log_event("Diagnostics acquis", "INFO")
                
                st.success("✅ Données acquises!")
    
    with tab3:
        st.subheader("🎯 Thomson Scattering - Détail")
        
        st.write("""
        **Diffusion Thomson**
        
        Laser haute puissance (Nd:YAG, 1064 nm) diffusé par électrons plasma.
        
        Mesure simultanée T_e et n_e via forme spectrale.
        """)
        
        # Simuler spectre Thomson
        wavelengths = np.linspace(1060, 1068, 1000)
        lambda_0 = 1064
        
        T_e_sim = st.slider("Température e⁻ (keV)", 1, 30, 10)
        
        # Largeur Doppler
        delta_lambda = lambda_0 * np.sqrt(2 * T_e_sim * 1000 * PHYSICS_CONSTANTS['e'] / 
                                          (PHYSICS_CONSTANTS['mass_proton'] * PHYSICS_CONSTANTS['c']**2)) * 1e9
        
        spectrum = np.exp(-(wavelengths - lambda_0)**2 / (2 * delta_lambda**2))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=wavelengths, y=spectrum,
            mode='lines',
            line=dict(color='#FF6B35', width=3),
            fill='tozeroy'
        ))
        
        fig.add_vline(x=lambda_0, line_dash="dash", line_color="white",
                     annotation_text="λ₀")
        
        fig.update_layout(
            title=f"Spectre Thomson (T_e = {T_e_sim} keV)",
            xaxis_title="Longueur d'onde (nm)",
            yaxis_title="Intensité (u.a.)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Largeur Spectrale Δλ", f"{delta_lambda:.3f} nm")

# ==================== PAGE: FUSION REACTIONS ====================
elif page == "⚡ Fusion Reactions":
    st.header("⚡ Réactions de Fusion")
    
    tab1, tab2, tab3 = st.tabs(["⚛️ Réactions", "📊 Sections Efficaces", "💥 Énergétique"])
    
    with tab1:
        st.subheader("⚛️ Réactions Disponibles")
        
        for reaction, info in FUSION_REACTIONS.items():
            with st.expander(f"⚡ {reaction}: {info['formula']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Énergie:** {info['energy_MeV']} MeV")
                    st.write(f"**Pic section:** {info['cross_section_peak_keV']} keV")
                    st.write(f"**Probabilité:** {info['probability']}")
                
                with col2:
                    st.write("**Produits:**")
                    for product in info['products']:
                        st.write(f"  • {product}")
    
    with tab2:
        st.subheader("📊 Sections Efficaces")
        
        st.write("""
        Section efficace σ(E) : probabilité interaction vs énergie particule.
        """)
        
        # Calculer sections efficaces
        E = np.logspace(0, 3, 1000)  # keV
        
        # Formules paramétriques (Bosch-Hale)
        def sigma_DT(E_keV):
            B_G = 34.382
            A = np.array([6.927e4, 7.454e8, 2.050e6, 5.2002e4, 0])
            return (A[0] + E_keV*(A[1] + E_keV*(A[2] + E_keV*A[3]))) / \
                   (1 + E_keV*(A[4] + E_keV*(B_G/np.sqrt(E_keV))))
        
        sigma_DT_vals = np.array([sigma_DT(e) for e in E]) * 1e-31  # m²
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=E, y=sigma_DT_vals * 1e28,  # barn
            mode='lines',
            line=dict(color='#FF6B35', width=3),
            name='D-T'
        ))
        
        fig.add_vline(x=64, line_dash="dash", line_color="yellow",
                     annotation_text="Pic D-T (64 keV)")
        
        fig.update_layout(
            title="Section Efficace D-T",
            xaxis_title="Énergie Centre Masse (keV)",
            yaxis_title="σ (barn = 10⁻²⁸ m²)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 Maximum à ~64 keV → température optimale ~15-20 keV")
    
    with tab3:
        st.subheader("💥 Bilan Énergétique")
        
        reaction_select = st.selectbox("Réaction", list(FUSION_REACTIONS.keys()))
        
        info = FUSION_REACTIONS[reaction_select]
        
        st.write(f"### {info['formula']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Énergie Libérée:**")
            st.metric("Total", f"{info['energy_MeV']} MeV")
            
            # Détail produits
            if reaction_select == 'D-T':
                st.write("• Helium-4: 3.5 MeV (20%)")
                st.write("• Neutron: 14.1 MeV (80%)")
            elif reaction_select == 'D-He3':
                st.write("• Helium-4: 3.6 MeV (20%)")
                st.write("• Proton: 14.7 MeV (80%)")
        
        with col2:
            st.write("**Conversion:**")
            
            E_joules = info['energy_MeV'] * 1.602e-13
            st.metric("Joules", f"{E_joules:.2e} J")
            
            # Nombre réactions pour 1 MJ
            n_reactions_MJ = 1e6 / E_joules
            st.metric("Pour 1 MJ", f"{n_reactions_MJ:.2e} réactions")
            
            # Masse combustible
            if reaction_select == 'D-T':
                mass_kg = (PHYSICS_CONSTANTS['mass_deuterium'] + 
                          PHYSICS_CONSTANTS['mass_tritium']) * n_reactions_MJ
                st.metric("Combustible requis", f"{mass_kg*1e6:.2f} mg")

# ==================== PAGE: SIMULATIONS ====================
elif page == "💻 Simulations":
    st.header("💻 Simulations Numériques")
    
    tab1, tab2, tab3 = st.tabs(["🔥 Transport", "🌊 MHD", "⚛️ Particules"])
    
    with tab1:
        st.subheader("🔥 Transport Énergie")
        
        st.write("""
        **Modèles Transport:**
        
        - Diffusion thermique
        - Turbulence edge
        - Transport néoclassique
        """)
        
        if st.session_state.fusion_lab['reactors']:
            sim_reactor = st.selectbox("Réacteur",
                list(st.session_state.fusion_lab['reactors'].keys()),
                format_func=lambda x: st.session_state.fusion_lab['reactors'][x]['name'],
                key="sim_reactor")
            
            chi = st.slider("Diffusivité thermique χ (m²/s)", 0.1, 5.0, 1.0, 0.1)
            
            if st.button("▶️ Simuler Transport", type="primary"):
                with st.spinner("Simulation..."):
                    import time
                    time.sleep(2)
                    
                    reactor = st.session_state.fusion_lab['reactors'][sim_reactor]
                    
                    # Grille radiale
                    r = np.linspace(0, reactor['minor_radius_m'], 100)
                    
                    # Équation diffusion 1D
                    # ∂T/∂t = (1/r)∂/∂r(r·χ·∂T/∂r) + S
                    
                    T_init = reactor['target_temperature_keV'] * (1 - (r/reactor['minor_radius_m'])**2)**2
                    
                    # Simulation simplifiée
                    T_final = T_init * 0.8  # Pertes transport
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=r, y=T_init,
                        mode='lines',
                        name='t=0',
                        line=dict(color='#FF6B35', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=r, y=T_final,
                        mode='lines',
                        name='t=10s',
                        line=dict(color='#F7931E', width=3, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Évolution Profil Température",
                        xaxis_title="Rayon (m)",
                        yaxis_title="T (keV)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Simulation terminée")
        else:
            st.info("Créez un réacteur")
    
    with tab2:
        st.subheader("🌊 Instabilités MHD")
        
        st.write("""
        **Modes MHD: MagnetoHydroDynamics **
        - Modes m/n (kink, tearing)
        - Ballooning
        - ELMs (Edge Localized Modes)
        Les modes MHD décrivent les **instabilités magnétohydrodynamiques** qui apparaissent dans le plasma confiné.  
        Elles peuvent affecter la stabilité, le confinement et parfois provoquer une décharge du plasma.

        **Principaux types de modes :**
        - **Modes m/n (kink, tearing)** :  
        Instabilités hélicoïdales caractérisées par les nombres poloidaux `m` et toroïdaux `n`.  
        - *Kink mode* : déformation globale de la colonne de plasma.  
        - *Tearing mode* : formation d’îlots magnétiques et reconnexion des lignes de champ.
        - **Ballooning modes** :  
        Instabilités localisées sur le bord du plasma dues à un fort gradient de pression.  
        Souvent liées à la limite de performance de confinement.
        - **ELMs (Edge Localized Modes)** :  
        Instabilités périodiques au bord du plasma qui expulsent de la chaleur et des particules.  
        Leur contrôle est essentiel pour protéger les parois du tokamak.

        **Diagnostic associé :**
        - Détection via signaux Mirnov coils et spectrogrammes temps-fréquence.  
        - Analyse des harmoniques `m/n` pour identifier le type de mode.  
        - Simulation via équations MHD (code M3D, JOREK, etc.)

        **But du contrôle :**
        - Stabiliser les modes par rétroaction magnétique active.  
        - Optimiser le profil de courant et la pression pour réduire la croissance des instabilités.

        """)
        
        if st.button("🌊 Simuler MHD"):
            st.write("Simulation instabilités MHD en cours...")
            
            # Modes disponibles
            modes = {
                'm=1, n=1': {'growth_rate': 0.05, 'frequency': 10},
                'm=2, n=1': {'growth_rate': -0.02, 'frequency': 15},
                'm=3, n=2': {'growth_rate': 0.01, 'frequency': 20}
            }
            
            for mode, params in modes.items():
                status = "🔴 Instable" if params['growth_rate'] > 0 else "🟢 Stable"
                st.write(f"**{mode}**: {status} (γ={params['growth_rate']:.3f}, f={params['frequency']} kHz)")
    
    with tab3:
        st.subheader("⚛️ Simulation Particules")
        
        st.write("""
        **Monte-Carlo / PIC**
        
        Suivi trajectoires particules individuelles dans champs E et B.
        """)
        
        n_particles = st.slider("Nombre particules", 100, 10000, 1000)
        
        if st.button("⚛️ Simuler Trajectoires"):
            st.info(f"Simulation {n_particles} particules...")

# ==================== PAGE: SYSTÈMES CHAUFFAGE ====================
elif page == "🔋 Systèmes Chauffage":
    st.header("🔋 Systèmes de Chauffage Plasma")
    
    st.info("""
    **Méthodes Chauffage:**
    - **NBI** (Neutral Beam Injection): Faisceaux neutres énergétiques
    - **ICRH** (Ion Cyclotron): Résonance radiofréquence ions
    - **ECRH** (Electron Cyclotron): Résonance micro-ondes électrons
    - **LHCD** (Lower Hybrid): Génération courant
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔵 NBI", "📡 ICRH/ECRH", "⚡ Efficacité"])
    
    with tab1:
        st.subheader("🔵 Neutral Beam Injection (NBI)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            beam_energy = st.slider("Énergie faisceau (keV)", 50, 1000, 500)
            beam_power = st.slider("Puissance (MW)", 10, 50, 33)
            n_beamlines = st.slider("Nombre lignes", 1, 4, 2)
        
        with col2:
            species = st.selectbox("Espèce", ["D⁰", "H⁰", "He⁰"])
            pulse_length = st.slider("Durée pulse (s)", 1, 100, 10)
            
            efficiency = 0.35
            st.metric("Efficacité", f"{efficiency*100:.0f}%")
            st.metric("Puissance Effective", f"{beam_power * efficiency:.1f} MW")
        
        if st.button("🔵 Activer NBI"):
            st.success(f"✅ NBI activé: {beam_power} MW @ {beam_energy} keV")
            log_event(f"NBI: {beam_power}MW", "INFO")
    
    with tab2:
        st.subheader("📡 Chauffage Radiofréquence")
        
        rf_method = st.radio("Méthode", ["ICRH", "ECRH", "LHCD"], horizontal=True)
        
        if rf_method == "ICRH":
            st.write("**Ion Cyclotron Resonance Heating**")
            frequency = st.slider("Fréquence (MHz)", 20, 80, 50)
            power = st.slider("Puissance (MW)", 5, 30, 20)
            
            st.write(f"🎯 Résonance ions à {frequency} MHz")
        
        elif rf_method == "ECRH":
            st.write("**Electron Cyclotron Resonance Heating**")
            frequency = st.slider("Fréquence (GHz)", 100, 170, 140)
            power = st.slider("Puissance (MW)", 5, 20, 10)
            
            st.write(f"🎯 Résonance électrons à {frequency} GHz")
        
        if st.button(f"📡 Activer {rf_method}"):
            st.success(f"✅ {rf_method} activé")
    
    with tab3:
        st.subheader("⚡ Efficacité Chauffage")
        
        methods_comparison = {
            'Méthode': ['NBI', 'ICRH', 'ECRH', 'Ohmique', 'Alpha (fusion)'],
            'Efficacité (%)': [35, 50, 60, 20, 20],
            'Puissance Max (MW)': [50, 30, 20, 10, 'Variable'],
            'Localisation': ['Core', 'Réglable', 'Précise', 'Uniforme', 'Core']
        }
        
        df = pd.DataFrame(methods_comparison)
        st.dataframe(df, use_container_width=True)
                 
# ==================== PAGE: ANALYTICS AVANCÉ ====================
elif page == "📊 Analytics":
    st.header("📊 Analytics & Intelligence des Données")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔍 Deep Analytics", "🎯 Prédictif", "📊 Big Data"])
    
    with tab1:
        st.subheader("📈 Dashboard Temps Réel")
        
        # Métriques globales
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Réacteurs Actifs", 
                     len([r for r in st.session_state.fusion_lab['reactors'].values() if r.get('status') == 'online']),
                     f"+{np.random.randint(0, 3)}")
        
        with col2:
            total_shots = len(st.session_state.fusion_lab['plasma_shots'])
            st.metric("Tirs Aujourd'hui", total_shots, f"+{np.random.randint(5, 15)}")
        
        with col3:
            if st.session_state.fusion_lab['plasma_shots']:
                avg_Q = np.mean([s.get('achieved_Q', 0) for s in st.session_state.fusion_lab['plasma_shots']])
                st.metric("Q Moyen", f"{avg_Q:.2f}", f"+{np.random.uniform(0.01, 0.05):.2f}")
            else:
                st.metric("Q Moyen", "0.00")
        
        with col4:
            if st.session_state.fusion_lab['plasma_shots']:
                total_energy = sum([s.get('total_energy_MJ', 0) for s in st.session_state.fusion_lab['plasma_shots']])
                st.metric("Énergie Produite", f"{total_energy:.1f} MJ", "↑")
            else:
                st.metric("Énergie Produite", "0.0 MJ")
        
        with col5:
            availability = 94.5 + np.random.uniform(-2, 2)
            st.metric("Disponibilité", f"{availability:.1f}%", 
                     f"{np.random.uniform(-1, 1):+.1f}%")
        
        # Graphiques temps réel
        st.write("### 📊 Métriques en Temps Réel")
        
        time_points = pd.date_range(end=datetime.now(), periods=50, freq='1min')
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Q Factor", "Puissance Fusion", "Densité Plasma", "Température")
        )
        
        # Q factor
        Q_data = 0.6 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.normal(0, 0.02, 50)
        fig.add_trace(go.Scatter(
            x=time_points, y=Q_data,
            mode='lines',
            line=dict(color='#FF6B35', width=2),
            fill='tozeroy'
        ), row=1, col=1)
        
        # Puissance
        P_data = 150 + 30 * np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.normal(0, 5, 50)
        fig.add_trace(go.Scatter(
            x=time_points, y=P_data,
            mode='lines',
            line=dict(color='#FDC830', width=2),
            fill='tozeroy'
        ), row=1, col=2)
        
        # Densité
        n_data = 1e20 * (1 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 50))) + np.random.normal(0, 1e18, 50)
        fig.add_trace(go.Scatter(
            x=time_points, y=n_data,
            mode='lines',
            line=dict(color='#F7931E', width=2),
            fill='tozeroy'
        ), row=2, col=1)
        
        # Température
        T_data = 15 + 2 * np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.normal(0, 0.3, 50)
        fig.add_trace(go.Scatter(
            x=time_points, y=T_data,
            mode='lines',
            line=dict(color='#F37335', width=2),
            fill='tozeroy'
        ), row=2, col=2)
        
        fig.update_yaxes(title_text="Q", row=1, col=1)
        fig.update_yaxes(title_text="P (MW)", row=1, col=2)
        fig.update_yaxes(title_text="n (m⁻³)", row=2, col=1)
        fig.update_yaxes(title_text="T (keV)", row=2, col=2)
        
        fig.update_layout(
            template="plotly_dark",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Deep Analytics")
        
        st.write("### 🔬 Analyse Corrélations")
        
        if st.button("🔍 Analyser Corrélations"):
            # Générer matrice corrélation
            params = ['I_p', 'B_T', 'n_e', 'T_e', 'P_heat', 'Q', 'β', 'τ_E']
            n_params = len(params)
            
            corr_matrix = np.random.rand(n_params, n_params)
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            np.fill_diagonal(corr_matrix, 1)
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=params,
                y=params,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Corrélation")
            ))
            
            fig.update_layout(
                title="Matrice Corrélations Paramètres",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### 💡 Insights")
            st.success("✅ Forte corrélation: Q ↔ τ_E (0.87)")
            st.info("📊 Corrélation modérée: β ↔ P_heat (0.54)")
            st.warning("⚠️ Anti-corrélation: n_e ↔ T_e (-0.32)")
    
    with tab3:
        st.subheader("🎯 Analytics Prédictif")
        
        st.write("### 🔮 Prédiction Prochains Tirs")
        
        prediction_model = st.selectbox("Modèle",
            ["LSTM", "Prophet", "ARIMA", "Random Forest"])
        
        if st.button("🔮 Générer Prédictions"):
            # Prédictions 10 prochains tirs
            n_future = 10
            
            Q_predicted = np.linspace(0.65, 0.85, n_future) + np.random.normal(0, 0.03, n_future)
            
            fig = go.Figure()
            
            # Historique
            if st.session_state.fusion_lab['plasma_shots']:
                Q_history = [s.get('achieved_Q', 0) for s in st.session_state.fusion_lab['plasma_shots'][-20:]]
                fig.add_trace(go.Scatter(
                    x=list(range(-len(Q_history), 0)),
                    y=Q_history,
                    mode='lines+markers',
                    name='Historique',
                    line=dict(color='#FF6B35', width=2)
                ))
            
            # Prédictions
            fig.add_trace(go.Scatter(
                x=list(range(0, n_future)),
                y=Q_predicted,
                mode='lines+markers',
                name='Prédiction',
                line=dict(color='#FDC830', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f"Prédictions Q Factor ({prediction_model})",
                xaxis_title="Tir Relatif",
                yaxis_title="Q Factor",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Prédiction: Q moyen futur = {np.mean(Q_predicted):.2f}")
    
    with tab4:
        st.subheader("📊 Big Data Pipeline")
        
        st.write("""
        **Infrastructure Data:**
        - Ingestion: 10+ TB/jour
        - Processing: Apache Spark
        - Storage: Data Lake (S3)
        - Analytics: Databricks
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📥 Sources Données")
            
            data_sources = {
                'Source': ['Diagnostics', 'Caméras', 'Magnétiques', 'Neutrons', 'Simulations'],
                'Fréquence': ['1 MHz', '1 kHz', '10 kHz', '1 kHz', 'Variable'],
                'Volume/jour': ['5 TB', '2 TB', '1 TB', '0.5 TB', '2 TB']
            }
            
            df_sources = pd.DataFrame(data_sources)
            st.dataframe(df_sources, use_container_width=True)
        
        with col2:
            st.write("### 🔄 Pipeline")
            
            pipeline = st.checkbox("Activer Pipeline Temps Réel")
            
            if pipeline:
                st.success("✅ Pipeline actif")
                
                processing_rate = np.random.uniform(8, 12)
                st.metric("Taux Traitement", f"{processing_rate:.1f} TB/h")
                
                latency = np.random.uniform(50, 150)
                st.metric("Latence", f"{latency:.0f} ms")

# ==================== PAGE: ITER DATABASE ====================
elif page == "🌍 ITER Database":
    st.header("🌍 Base de Données ITER & Tokamaks Mondiaux")
    
    st.info("""
    **Tokamaks Majeurs:**
    - ITER (International)
    - JET (UK)
    - TFTR (USA, décommissionné)
    - JT-60SA (Japon)
    - EAST (Chine)
    - KSTAR (Corée)
    - DIII-D (USA)
    """)
    
    tab1, tab2, tab3 = st.tabs(["🌐 Tokamaks", "📊 Comparaisons", "🏆 Records"])
    
    with tab1:
        st.subheader("🌐 Tokamaks Mondiaux")
        
        tokamaks_data = {
            'Nom': ['ITER', 'JET', 'JT-60SA', 'EAST', 'KSTAR', 'DIII-D', 'SPARC'],
            'Pays': ['🌍 Int.', '🇬🇧 UK', '🇯🇵 Japon', '🇨🇳 Chine', '🇰🇷 Corée', '🇺🇸 USA', '🇺🇸 USA'],
            'R (m)': [6.2, 3.0, 3.0, 1.9, 1.8, 1.67, 1.85],
            'a (m)': [2.0, 1.0, 1.18, 0.45, 0.5, 0.67, 0.57],
            'B_T (T)': [5.3, 3.8, 2.25, 3.5, 3.5, 2.2, 12.2],
            'I_p (MA)': [15, 4.8, 5.5, 1.0, 2.0, 2.0, 8.7],
            'Q (max)': [10, 0.67, '—', '—', '—', '—', 2],
            'Status': ['Construction', 'Opérationnel', 'Opérationnel', 'Opérationnel', 
                      'Opérationnel', 'Opérationnel', 'En construction']
        }
        
        df_tokamaks = pd.DataFrame(tokamaks_data)
        st.dataframe(df_tokamaks, use_container_width=True)
        
        # Carte interactive
        st.write("### 🗺️ Localisation Mondiale")
        
        locations = {
            'ITER': [43.7, 5.8],
            'JET': [51.7, -1.2],
            'JT-60SA': [36.3, 140.5],
            'EAST': [31.9, 117.3],
            'KSTAR': [36.1, 128.3],
            'DIII-D': [32.9, -117.2]
        }
        
        map_data = pd.DataFrame([
            {'name': name, 'lat': coords[0], 'lon': coords[1]}
            for name, coords in locations.items()
        ])
        
        fig = go.Figure(go.Scattergeo(
            lon=map_data['lon'],
            lat=map_data['lat'],
            text=map_data['name'],
            mode='markers+text',
            marker=dict(size=15, color='#FF6B35'),
            textposition='top center'
        ))
        
        fig.update_layout(
            title="Tokamaks Majeurs - Distribution Mondiale",
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(50, 50, 50)',
                coastlinecolor='rgb(100, 100, 100)'
            ),
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Comparaisons Techniques")
        
        comparison_param = st.selectbox("Paramètre",
            ["Rayon majeur R", "Champ toroïdal B_T", "Courant plasma I_p", "Q factor"])
        
        param_map = {
            "Rayon majeur R": 'R (m)',
            "Champ toroïdal B_T": 'B_T (T)',
            "Courant plasma I_p": 'I_p (MA)',
            "Q factor": 'Q (max)'
        }
        
        selected_col = param_map[comparison_param]
        
        # Exclure valeurs non numériques
        df_plot = df_tokamaks[df_tokamaks[selected_col] != '—'].copy()
        df_plot[selected_col] = pd.to_numeric(df_plot[selected_col])
        
        fig = go.Figure(data=[go.Bar(
            x=df_plot['Nom'],
            y=df_plot[selected_col],
            marker_color='#FF6B35',
            text=df_plot[selected_col],
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f"Comparaison {comparison_param}",
            xaxis_title="Tokamak",
            yaxis_title=selected_col,
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🏆 Records Historiques")
        
        records = [
            {
                'Record': 'Q Factor le plus élevé',
                'Valeur': 'Q = 0.67',
                'Tokamak': 'JET',
                'Date': '1997',
                'Conditions': '24 MW NBI, D-T'
            },
            {
                'Record': 'Puissance fusion',
                'Valeur': '16.1 MW',
                'Tokamak': 'JET',
                'Date': '1997',
                'Conditions': 'Pic 4 secondes'
            },
            {
                'Record': 'Durée pulse la plus longue',
                'Valeur': '1056 secondes',
                'Tokamak': 'EAST',
                'Date': '2022',
                'Conditions': 'T_e > 100 MK'
            },
            {
                'Record': 'Température la plus élevée',
                'Valeur': '510 MK (44 keV)',
                'Tokamak': 'JET',
                'Date': '2021',
                'Conditions': 'Record monde'
            },
            {
                'Record': 'β_N le plus élevé',
                'Valeur': 'β_N = 3.8',
                'Tokamak': 'DIII-D',
                'Date': '2019',
                'Conditions': 'Régime avancé'
            }
        ]
        
        for record in records:
            with st.expander(f"🏆 {record['Record']}: {record['Valeur']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Tokamak:** {record['Tokamak']}")
                    st.write(f"**Date:** {record['Date']}")
                
                with col2:
                    st.write(f"**Conditions:** {record['Conditions']}")

# ==================== PAGE: PHYSICS LIBRARY ====================
elif page == "📚 Physics Library":
    st.header("📚 Bibliothèque Physique Fusion")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Concepts", "🧮 Formules", "📊 Calculateurs", "🎓 Tutoriels"])
    
    with tab1:
        st.subheader("📖 Concepts Fondamentaux")
        
        concepts = {
            "⚛️ Fusion Thermonucléaire": """
            Réaction nucléaire où deux noyaux légers fusionnent pour former un noyau plus lourd,
            libérant de l'énergie selon E=mc².
            
            **Conditions:**
            - T > 10 keV (~100 millions K)
            - n > 10²⁰ m⁻³
            - τ_E > 1 seconde
            """,
            
            "🧲 Confinement Magnétique": """
            Utilisation champs magnétiques pour confiner plasma chaud.
            Force Lorentz: F = q(v × B) maintient particules sur orbites circulaires.
            
            **Tokamak:** Configuration toroïdale avec champs B_T et B_P
            """,
            
            "📊 Critère Lawson": """
            Condition pour ignition plasma:
            
            n · τ_E · T ≥ 3×10²¹ m⁻³·s·keV
            
            Établi par John Lawson (1957)
            """,
            
            "⚡ Q Factor": """
            Gain énergétique fusion:
            
            Q = P_fusion / P_heating
            
            - Q < 1: Pas de gain
            - Q = 1: Breakeven
            - Q > 1: Gain net
            - Q → ∞: Ignition
            """
        }
        
        for concept, description in concepts.items():
            with st.expander(concept):
                st.markdown(description)
    
    with tab2:
        st.subheader("🧮 Formules Essentielles")
        
        st.write("### ⚛️ Puissance Fusion D-T")
        st.latex(r"P_{fus} = \frac{1}{4} n^2 \langle\sigma v\rangle E_{fusion}")
        st.write("Où:")
        st.write("- n: densité (m⁻³)")
        st.write("- ⟨σv⟩: réactivité (m³/s)")
        st.write("- E_fusion: 17.6 MeV pour D-T")
        
        st.write("### 🧲 Pression Plasma (Beta)")
        st.latex(r"\beta = \frac{p_{plasma}}{p_{magnetic}} = \frac{nk_BT}{B^2/(2\mu_0)}")
        
        st.write("### 🔄 Temps Confinement Énergie")
        st.latex(r"\tau_E = \frac{W_{plasma}}{P_{loss}}")
        
        st.write("### ⚙️ Facteur de Sécurité q")
        st.latex(r"q = \frac{aB_T}{RB_P}")
    
    with tab3:
        st.subheader("📊 Calculateurs Interactifs")
        
        calc_type = st.selectbox("Calculateur",
            ["Puissance Fusion", "Critère Lawson", "Beta", "Réactivité", "Triple Produit"])
        
        if calc_type == "Puissance Fusion":
            st.write("### ⚛️ Calcul Puissance Fusion")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_calc = st.number_input("Densité n (m⁻³)", 1e19, 1e21, 1e20, format="%.2e")
                T_calc = st.slider("Température T (keV)", 1, 50, 15)
                V_calc = st.number_input("Volume (m³)", 1, 10000, 1000)
            
            with col2:
                reaction_calc = st.selectbox("Réaction", ["D-T", "D-D", "D-He3"])
            
            if st.button("🧮 Calculer"):
                P_calc = calculate_fusion_power(n_calc, T_calc*1000, reaction_calc) * V_calc
                
                st.success(f"✅ Puissance Fusion: {P_calc/1e6:.2f} MW")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("P_fusion", f"{P_calc/1e6:.1f} MW")
                with col2:
                    st.metric("P_density", f"{P_calc/V_calc/1e6:.2f} MW/m³")
                with col3:
                    neutrons = calculate_neutron_flux(P_calc, V_calc)
                    st.metric("Neutron flux", f"{neutrons:.2e} n/m²/s")
    
    with tab4:
        st.subheader("🎓 Tutoriels")
        
        tutorials = [
            {
                'title': '🎯 Introduction Fusion Nucléaire',
                'duration': '30 min',
                'level': 'Débutant',
                'topics': ['Principes base', 'Réactions fusion', 'Applications']
            },
            {
                'title': '🧲 Confinement Magnétique',
                'duration': '45 min',
                'level': 'Intermédiaire',
                'topics': ['Tokamaks', 'Champs magnétiques', 'MHD']
            },
            {
                'title': '⚡ Physique Plasma Avancée',
                'duration': '60 min',
                'level': 'Avancé',
                'topics': ['Transport', 'Instabilités', 'Turbulence']
            },
            {
                'title': '🤖 IA pour Contrôle Plasma',
                'duration': '90 min',
                'level': 'Expert',
                'topics': ['Deep Learning', 'RL', 'Prédiction']
            }
        ]
        
        for tuto in tutorials:
            with st.expander(f"{tuto['title']} ({tuto['level']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Durée:** {tuto['duration']}")
                    st.write(f"**Niveau:** {tuto['level']}")
                
                with col2:
                    st.write("**Sujets:**")
                    for topic in tuto['topics']:
                        st.write(f"  • {topic}")
                
                if st.button(f"▶️ Démarrer", key=f"tuto_{tuto['title']}"):
                    st.info("📚 Tutoriel en cours de chargement...")

# ==================== PAGE: MAINTENANCE ====================
elif page == "⚙️ Maintenance":
    st.header("⚙️ Maintenance & Opérations")
    
    st.info("""
    **Maintenance Préventive:**
    - Inspection première paroi
    - Vérification bobines magnétiques
    - Calibration diagnostics
    - Remplacement composants usés
    """)
    
    tab1, tab2, tab3 = st.tabs(["📋 Planning", "🔧 Interventions", "📊 Historique"])
    
    with tab1:
        st.subheader("📋 Planning Maintenance")
        
        # Calendrier maintenance
        maintenance_schedule = {
            'Composant': ['Première Paroi', 'Bobines TF', 'Diagnostics Thomson', 
                         'Système Vide', 'NBI', 'Divertor'],
            'Prochaine Maintenance': ['2024-12-15', '2025-01-20', '2024-11-30',
                                      '2024-12-01', '2024-11-25', '2025-02-10'],
            'Fréquence': ['6 mois', '12 mois', '3 mois', '3 mois', '3 mois', '12 mois'],
            'Criticité': ['Haute', 'Haute', 'Moyenne', 'Haute', 'Moyenne', 'Haute'],
            'Durée (jours)': [14, 21, 2, 5, 3, 30]
        }
        
        df_maintenance = pd.DataFrame(maintenance_schedule)
        st.dataframe(df_maintenance, use_container_width=True)
        
        # Diagramme Gantt simplifié
        st.write("### 📅 Timeline Maintenance")
        
        fig = go.Figure()
        
        for i, row in df_maintenance.iterrows():
            start_date = pd.to_datetime(row['Prochaine Maintenance'])
            end_date = start_date + pd.Timedelta(days=row['Durée (jours)'])
            
            color = '#FF0000' if row['Criticité'] == 'Haute' else '#FFA500' if row['Criticité'] == 'Moyenne' else '#00FF00'
            
            fig.add_trace(go.Scatter(
                x=[start_date, end_date],
                y=[i, i],
                mode='lines',
                line=dict(color=color, width=20),
                name=row['Composant'],
                hovertemplate=f"{row['Composant']}<br>%{{x}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Planning Maintenance 6 Prochains Mois",
            xaxis_title="Date",
            yaxis=dict(
                ticktext=df_maintenance['Composant'],
                tickvals=list(range(len(df_maintenance)))
            ),
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔧 Planifier Intervention")
        
        with st.form("schedule_maintenance"):
            col1, col2 = st.columns(2)
            
            with col1:
                component = st.selectbox("Composant",
                    ["Première Paroi", "Bobines Magnétiques", "Diagnostics",
                     "Système Chauffage", "Divertor", "Autre"])
                
                maintenance_type = st.selectbox("Type",
                    ["Préventive", "Corrective", "Améliorative", "Inspection"])
                
                scheduled_date = st.date_input("Date Planifiée")
            
            with col2:
                duration_days = st.number_input("Durée (jours)", 1, 60, 7)
                
                priority = st.select_slider("Priorité",
                    options=["Basse", "Normale", "Haute", "Urgente"])
                
                shutdown_required = st.checkbox("Arrêt réacteur requis", value=True)
            
            description = st.text_area("Description Intervention",
                "Inspection et remplacement composants usés")
            
            if st.form_submit_button("📝 Planifier", type="primary"):
                maintenance_record = {
                    'component': component,
                    'type': maintenance_type,
                    'date': scheduled_date.isoformat(),
                    'duration': duration_days,
                    'priority': priority,
                    'description': description
                }
                
                st.session_state.fusion_lab['maintenance_log'].append(maintenance_record)
                log_event(f"Maintenance planifiée: {component}", "INFO")
                
                st.success(f"✅ Maintenance planifiée: {component} le {scheduled_date}")
    
    with tab3:
        st.subheader("📊 Historique Maintenance")
        
        if st.session_state.fusion_lab['maintenance_log']:
            st.write(f"### 📋 {len(st.session_state.fusion_lab['maintenance_log'])} Interventions Enregistrées")
            
            for i, record in enumerate(st.session_state.fusion_lab['maintenance_log'][::-1][:10]):
                with st.expander(f"🔧 {record['component']} - {record['date']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Type:** {record['type']}")
                        st.write(f"**Date:** {record['date']}")
                    
                    with col2:
                        st.write(f"**Durée:** {record['duration']} jours")
                        st.write(f"**Priorité:** {record['priority']}")
                    
                    with col3:
                        status = np.random.choice(['✅ Complété', '🔄 En cours', '📅 Planifié'])
                        st.write(f"**Status:** {status}")
                    
                    st.write(f"**Description:** {record['description']}")
        else:
            st.info("Aucune maintenance enregistrée")

# ==================== PAGE: MONITORING LIVE ====================
elif page == "📡 Monitoring Live":
    st.header("📡 Monitoring Temps Réel")
    
    st.info("Actualisation automatique toutes les 5 secondes")
    
    # Auto-refresh
    if st.button("🔄 Actualiser"):
        st.rerun()
    
    # Statut global
    st.write("### 🎛️ État Système Global")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        status = "🟢 Opérationnel"
        st.metric("Status", status)
    
    with col2:
        cpu_usage = np.random.uniform(40, 70)
        st.metric("CPU", f"{cpu_usage:.0f}%")
    
    with col3:
        memory_usage = np.random.uniform(60, 80)
        st.metric("RAM", f"{memory_usage:.0f}%")
    
    with col4:
        network = np.random.uniform(100, 500)
        st.metric("Réseau", f"{network:.0f} Mbps")
    
    with col5:
        temp = np.random.uniform(35, 45)
        st.metric("Temp", f"{temp:.1f}°C")
    
    # Métriques réacteurs
    if st.session_state.fusion_lab['reactors']:
        st.write("### ⚛️ Réacteurs")
        
        for reactor_id, reactor in st.session_state.fusion_lab['reactors'].items():
            with st.expander(f"🔥 {reactor['name']} - {reactor.get('status', 'offline').upper()}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    plasma_status = "🟢 Stable" if np.random.random() > 0.2 else "🟡 Fluctuant"
                    st.write(f"**Plasma:** {plasma_status}")
                    
                    T_live = reactor['target_temperature_keV'] * (1 + np.random.uniform(-0.1, 0.1))
                    st.metric("T", f"{T_live:.1f} keV")
                
                with col2:
                    n_live = reactor['target_density_m3'] * (1 + np.random.uniform(-0.05, 0.05))
                    st.metric("n", f"{n_live:.2e} m⁻³")
                    
                    I_p_live = reactor['plasma_current_MA'] * (1 + np.random.uniform(-0.02, 0.02))
                    st.metric("I_p", f"{I_p_live:.1f} MA")
                
                with col3:
                    P_heat_live = reactor['heating_power_MW'] * (1 + np.random.uniform(-0.1, 0.1))
                    st.metric("P_heat", f"{P_heat_live:.1f} MW")
                    
                    Q_live = reactor['Q_factor_est'] * (1 + np.random.uniform(-0.1, 0.1))
                    st.metric("Q", f"{Q_live:.2f}")
                
                with col4:
                    disruption_risk = np.random.uniform(0, 1)
                    if disruption_risk > 0.7:
                        st.error(f"⚠️ Risque: {disruption_risk:.0%}")
                    else:
                        st.success(f"✅ Risque: {disruption_risk:.0%}")
    
    # Graphique live
    st.write("### 📈 Signaux Temps Réel")
    
    # Simuler données temps réel
    time_window = 60  # secondes
    t = np.linspace(-time_window, 0, 100)
    
    signal1 = 10 + 2*np.sin(2*np.pi*t/10) + np.random.normal(0, 0.3, 100)
    signal2 = 5 + np.sin(2*np.pi*t/5) + np.random.normal(0, 0.2, 100)
    
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Signal 1", "Signal 2"))
    
    fig.add_trace(go.Scatter(
        x=t, y=signal1,
        mode='lines',
        line=dict(color='#FF6B35', width=2),
        name='Signal 1'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=t, y=signal2,
        mode='lines',
        line=dict(color='#FDC830', width=2),
        name='Signal 2'
    ), row=2, col=1)
    
    fig.update_xaxes(title_text="Temps (s)", row=2, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Configuration Laboratoire")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 Général", "💾 Données", "🔔 Notifications", "🔄 Reset"])
    
    with tab1:
        st.subheader("🔧 Paramètres Généraux")
        
        with st.form("settings_general"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🌡️ Physique")
                default_fuel = st.selectbox("Combustible par défaut", ["D-T", "D-D", "D-He3"])
                default_temp = st.slider("Température par défaut (keV)", 5, 30, 15)
                default_density = st.number_input("Densité par défaut (m⁻³)", 1e19, 1e21, 1e20, format="%.2e")
            
            with col2:
                st.write("### 🖥️ Interface")
                theme = st.selectbox("Thème", ["Dark", "Light"], index=0)
                language = st.selectbox("Langue", ["English", "Français", "日本語", "中文"])
                refresh_rate = st.slider("Taux rafraîchissement (s)", 1, 30, 5)
            
            st.write("### 🔬 Unités")
            col1, col2 = st.columns(2)
            
            with col1:
                temp_unit = st.radio("Température", ["keV", "MK", "K"], horizontal=True)
                density_unit = st.radio("Densité", ["m⁻³", "cm⁻³"], horizontal=True)
            
            with col2:
                energy_unit = st.radio("Énergie", ["MeV", "J", "eV"], horizontal=True)
                power_unit = st.radio("Puissance", ["MW", "W", "GW"], horizontal=True)
            
            if st.form_submit_button("💾 Sauvegarder", type="primary"):
                st.success("✅ Paramètres sauvegardés!")
                log_event("Paramètres mis à jour", "INFO")
    
    with tab2:
        st.subheader("💾 Gestion Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📊 Statistiques")
            
            total_data = {
                'Réacteurs': len(st.session_state.fusion_lab['reactors']),
                'Tirs Plasma': len(st.session_state.fusion_lab['plasma_shots']),
                'Expériences': len(st.session_state.fusion_lab['experiments']),
                'Diagnostics': len(st.session_state.fusion_lab['diagnostics']),
                'Logs': len(st.session_state.fusion_lab['log'])
            }
            
            for key, value in total_data.items():
                st.metric(key, value)
        
        with col2:
            st.write("### 💾 Export/Import")
            
            export_format = st.selectbox("Format", ["JSON", "CSV", "HDF5", "Pickle"])
            
            if st.button("📥 Exporter Tout"):
                data_export = {
                    'reactors': len(st.session_state.fusion_lab['reactors']),
                    'shots': len(st.session_state.fusion_lab['plasma_shots']),
                    'export_date': datetime.now().isoformat()
                }
                
                st.success("✅ Données exportées!")
                st.json(data_export)
            
            st.write("---")
            
            uploaded_file = st.file_uploader("📤 Importer Données", type=['json'])
            
            if uploaded_file and st.button("📤 Importer"):
                st.success("✅ Données importées!")
    
    with tab3:
        st.subheader("🔔 Notifications & Alertes")
        
        st.write("### ⚙️ Configuration Alertes")
        
        with st.form("notifications"):
            col1, col2 = st.columns(2)
            
            with col1:
                email_alerts = st.checkbox("Alertes Email", value=True)
                email_address = st.text_input("Email", "physicist@lab.com")
                
                sms_alerts = st.checkbox("Alertes SMS")
                phone_number = st.text_input("Téléphone", "+33...")
            
            with col2:
                st.write("**Événements:**")
                
                alert_disruption = st.checkbox("Disruptions", value=True)
                alert_Q_threshold = st.checkbox("Q > Seuil", value=True)
                alert_maintenance = st.checkbox("Maintenance", value=True)
                alert_safety = st.checkbox("Sécurité", value=True)
            
            st.write("### 🎯 Seuils")
            
            col1, col2 = st.columns(2)
            
            with col1:
                Q_threshold = st.slider("Q Factor minimum", 0.1, 2.0, 0.5)
                disruption_prob = st.slider("Probabilité disruption (%)", 0, 100, 70)
            
            with col2:
                viability_min = st.slider("Viabilité minimum (%)", 50, 95, 85)
                temp_max = st.slider("Température max (keV)", 10, 50, 30)
            
            if st.form_submit_button("💾 Sauvegarder Alertes"):
                st.success("✅ Configuration alertes sauvegardée!")
    
    with tab4:
        st.subheader("🔄 Réinitialisation")
        
        st.error("### ⚠️ DANGER ZONE")
        st.warning("Les actions suivantes sont irréversibles!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Effacer Tirs Plasma"):
                if st.checkbox("Confirmer effacement tirs", key="conf_shots"):
                    st.session_state.fusion_lab['plasma_shots'] = []
                    st.success("✅ Tirs effacés")
                    log_event("Tirs plasma effacés", "WARNING")
            
            if st.button("🗑️ Effacer Diagnostics"):
                if st.checkbox("Confirmer effacement diagnostics", key="conf_diag"):
                    st.session_state.fusion_lab['diagnostics'] = []
                    st.success("✅ Diagnostics effacés")
        
        with col2:
            if st.button("🗑️ Effacer Expériences"):
                if st.checkbox("Confirmer effacement expériences", key="conf_exp"):
                    st.session_state.fusion_lab['experiments'] = []
                    st.success("✅ Expériences effacées")
            
            if st.button("🗑️ Effacer Logs"):
                if st.checkbox("Confirmer effacement logs", key="conf_logs"):
                    st.session_state.fusion_lab['log'] = []
                    st.success("✅ Logs effacés")
        
        st.markdown("---")
        
        st.error("### 🔴 RÉINITIALISATION TOTALE")
        
        if st.button("💥 TOUT RÉINITIALISER"):
            reset_confirm = st.text_input("Tapez 'RESET FUSION' pour confirmer")
            
            if reset_confirm == "RESET FUSION":
                st.session_state.fusion_lab = {
                    'reactors': {},
                    'plasma_shots': [],
                    'experiments': [],
                    'diagnostics': [],
                    'heating_systems': {},
                    'magnets': {},
                    'fuel_inventory': {
                        'deuterium_kg': 1000,
                        'tritium_g': 500,
                        'helium3_g': 10
                    },
                    'safety_systems': {},
                    'simulations': [],
                    'maintenance_log': [],
                    'log': []
                }
                
                st.success("✅ Laboratoire complètement réinitialisé!")
                st.balloons()
                log_event("RÉINITIALISATION TOTALE", "CRITICAL")
                st.rerun()

# ==================== PAGE: CONFINEMENT ====================
elif page == "🎯 Confinement":
    st.header("🎯 Confinement & Transport")
    
    tab1, tab2, tab3 = st.tabs(["📊 Scaling Laws", "🌀 Turbulence", "🔒 Barrières"])
    
    with tab1:
        st.subheader("📊 Lois d'Échelle Confinement")
        
        st.write("""
        **IPB98(y,2) Scaling Law (ITER):**
        
        τ_E = 0.0562 × I_p^0.93 × B_T^0.15 × n^0.41 × P^-0.69 × M^0.19 × R^1.97 × κ^0.78 × ε^0.58
        """)
        
        if st.session_state.fusion_lab['reactors']:
            selected_reactor = st.selectbox("Réacteur",
                list(st.session_state.fusion_lab['reactors'].keys()),
                format_func=lambda x: st.session_state.fusion_lab['reactors'][x]['name'],
                key="conf_reactor")
            
            reactor = st.session_state.fusion_lab['reactors'][selected_reactor]
            
            # Calcul τ_E selon IPB98
            I_p = reactor['plasma_current_MA']
            B_T = reactor['toroidal_field_T']
            n = reactor['target_density_m3'] / 1e19
            P = reactor['heating_power_MW']
            R = reactor['major_radius_m']
            epsilon = reactor['minor_radius_m'] / R
            
            tau_E_IPB = 0.0562 * (I_p**0.93) * (B_T**0.15) * (n**0.41) * (P**-0.69) * R**1.97 * epsilon**0.58
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("τ_E Cible", f"{reactor['confinement_time_s']:.2f} s")
            with col2:
                st.metric("τ_E IPB98", f"{tau_E_IPB:.2f} s")
            with col3:
                H_factor = reactor['confinement_time_s'] / tau_E_IPB
                st.metric("H Factor", f"{H_factor:.2f}")
                
                if H_factor > 1:
                    st.success("✅ Meilleur que scaling")
                else:
                    st.warning("⚠️ Sous scaling")
        else:
            st.info("Créez un réacteur")
    
    with tab2:
        st.subheader("🌀 Turbulence & Transport")
        
        st.write("""
        **Transport Turbulent**
        
        Modes ITG (Ion Temperature Gradient), TEM (Trapped Electron Mode)
        dominent le transport dans le coeur du plasma.
        """)
        
        turbulence_level = st.slider("Niveau Turbulence", 0.0, 1.0, 0.3, 0.1)
        
        chi_turb = 1.0 + 10 * turbulence_level
        st.metric("Diffusivité χ (m²/s)", f"{chi_turb:.1f}")
        
        if turbulence_level > 0.5:
            st.error("❌ Turbulence élevée → Pertes importantes")
        else:
            st.success("✅ Turbulence contrôlée")
    
    with tab3:
        st.subheader("🔒 Barrières Transport (H-mode)")
        
        st.write("""
        **H-mode (High Confinement)**
        
        Formation barrière edge → Réduction transport → Amélioration confinement
        """)
        
        transition_power = st.slider("Puissance Chauffage (MW)", 0, 100, 30)
        
        P_threshold = 2.0  # MW (simplifié)
        
        if transition_power > P_threshold:
            st.success("✅ H-mode atteint!")
            st.write("**Caractéristiques:**")
            st.write("• Pédestal pression à l'edge")
            st.write("• Confinement amélioré (H > 1)")
            st.write("• ELMs possibles")
        else:
            st.info("📊 L-mode (Low Confinement)")
            st.write(f"Augmenter puissance de {P_threshold - transition_power:.1f} MW pour H-mode")

# ==================== PAGE: IA AVANCÉE ====================
elif page == "🤖 IA & Machine Learning":
    st.header("🤖 Intelligence Artificielle pour Fusion")
    
    st.info("""
    **IA pour Contrôle Plasma:**
    - Prédiction disruptions (Deep Learning)
    - Optimisation temps réel (RL)
    - Reconstruction paramètres
    - Découverte régimes confinement
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Prédiction Disruptions", "🎮 RL Control", "🔮 Prédictif", "🔬 AutoML"])
    
    with tab1:
        st.subheader("🧠 Prédiction Disruptions (Deep Learning)")
        
        st.write("""
        **Réseau Neural Convolutionnel**
        
        Prédire disruptions 30-100ms avant occurrence → Temps pour mitigation
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("Architecture", 
                ["CNN-LSTM", "Transformer", "ResNet", "Attention"])
            
            training_shots = st.number_input("Tirs Entraînement", 1000, 100000, 10000)
            
            if st.button("🧠 Entraîner Modèle"):
                with st.spinner("Entraînement..."):
                    import time
                    progress = st.progress(0)
                    
                    for i in range(100):
                        time.sleep(0.03)
                        progress.progress(i + 1)
                    
                    accuracy = np.random.uniform(0.85, 0.95)
                    st.success(f"✅ Modèle entraîné: Accuracy = {accuracy:.1%}")
                    
                    st.write("### 📊 Métriques")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.1%}")
                    with col2:
                        st.metric("Precision", f"{accuracy*0.97:.1%}")
                    with col3:
                        st.metric("Recall", f"{accuracy*0.93:.1%}")
        
        with col2:
            st.write("### 🎯 Prédiction Temps Réel")
            
            if st.button("🔮 Prédire Disruption"):
                # Simuler prédiction
                disruption_prob = np.random.uniform(0, 1)
                time_to_disruption = np.random.uniform(30, 100)  # ms
                
                if disruption_prob > 0.7:
                    st.error(f"⚠️ ALERTE: Disruption probable ({disruption_prob:.0%})")
                    st.write(f"⏰ Temps restant: ~{time_to_disruption:.0f} ms")
                    st.write("🛡️ Activation mitigation recommandée")
                elif disruption_prob > 0.4:
                    st.warning(f"⚠️ Risque modéré: {disruption_prob:.0%}")
                else:
                    st.success(f"✅ Plasma stable: {disruption_prob:.0%}")
    
    with tab2:
        st.subheader("🎮 Reinforcement Learning Control")
        
        st.write("""
        **Contrôle Optimal par RL**
        
        Agent RL apprend politique optimale pour:
        - Maintenir H-mode
        - Maximiser Q factor
        - Éviter disruptions
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            rl_algorithm = st.selectbox("Algorithme",
                ["PPO", "SAC", "TD3", "DQN", "A3C"])
            
            objective = st.selectbox("Objectif",
                ["Maximiser Q", "Maintenir H-mode", "Prolonger durée", "Multi-objectif"])
            
            episodes = st.number_input("Épisodes", 100, 10000, 1000)
        
        with col2:
            st.write("### 🎯 État & Actions")
            
            st.write("**État (Observation):**")
            st.write("• T_e, n_e profils")
            st.write("• I_p, β, q profils")
            st.write("• P_heating, P_radiation")
            
            st.write("**Actions:**")
            st.write("• Puissance NBI")
            st.write("• Puissance RF")
            st.write("• Position plasma")
        
        if st.button("🚀 Entraîner Agent RL"):
            with st.spinner("Entraînement RL..."):
                import time
                
                rewards = []
                for episode in range(0, episodes, episodes//20):
                    time.sleep(0.1)
                    reward = -100 + episode * 0.15 + np.random.normal(0, 10)
                    rewards.append(reward)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(0, episodes, episodes//20)),
                    y=rewards,
                    mode='lines+markers',
                    line=dict(color='#FF6B35', width=2)
                ))
                
                fig.update_layout(
                    title="Courbe Apprentissage RL",
                    xaxis_title="Épisode",
                    yaxis_title="Reward Cumulé",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Agent RL entraîné!")
    
    with tab3:
        st.subheader("🔮 Modèles Prédictifs")
        
        st.write("""
        **Prédiction Paramètres Plasma**
        
        Prédire évolution paramètres (T, n, β) pour optimisation scénarios
        """)
        
        prediction_horizon = st.slider("Horizon prédiction (s)", 1, 30, 10)
        
        if st.button("🔮 Générer Prédictions"):
            t_future = np.linspace(0, prediction_horizon, 100)
            
            # Prédictions simulées
            T_pred = 15 * np.exp(-t_future/20) + 10
            n_pred = 1e20 * (1 - 0.1 * t_future/prediction_horizon)
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Température", "Densité"))
            
            fig.add_trace(go.Scatter(
                x=t_future, y=T_pred,
                mode='lines',
                line=dict(color='#FF6B35', width=3),
                name='Prédiction T'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=t_future, y=n_pred,
                mode='lines',
                line=dict(color='#FDC830', width=3),
                name='Prédiction n'
            ), row=2, col=1)
            
            fig.update_xaxes(title_text="Temps (s)", row=2, col=1)
            fig.update_yaxes(title_text="T (keV)", row=1, col=1)
            fig.update_yaxes(title_text="n (m⁻³)", row=2, col=1)
            
            fig.update_layout(
                template="plotly_dark",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔬 AutoML - Découverte Automatique")
        
        st.write("""
        **AutoML pour Fusion**
        
        Recherche automatique de:
        - Nouveaux régimes confinement
        - Stratégies optimales
        - Corrélations cachées
        """)
        
        if st.button("🔬 Lancer AutoML"):
            with st.spinner("Exploration espace paramètres..."):
                import time
                time.sleep(2)
                
                discoveries = [
                    {
                        'regime': 'Enhanced D-alpha (EDA)',
                        'characteristics': 'ELM-free, high density',
                        'Q_improvement': '+15%'
                    },
                    {
                        'regime': 'Quiescent H-mode (QH)',
                        'characteristics': 'Sans ELMs, edge harmonic oscillations',
                        'Q_improvement': '+20%'
                    },
                    {
                        'regime': 'Super H-mode',
                        'characteristics': 'Très haut confinement, β élevé',
                        'Q_improvement': '+35%'
                    }
                ]
                
                st.success("✅ 3 nouveaux régimes découverts!")
                
                for disc in discoveries:
                    with st.expander(f"🌟 {disc['regime']}"):
                        st.write(f"**Caractéristiques:** {disc['characteristics']}")
                        st.write(f"**Amélioration Q:** {disc['Q_improvement']}")

# ==================== PAGE: COMPUTING QUANTIQUE ====================
elif page == "⚛️ Quantum Computing":
    st.header("⚛️ Calcul Quantique pour Fusion")
    
    st.info("""
    **Applications Quantiques:**
    - Simulation plasma quantique
    - Optimisation scénarios (QAOA)
    - Machine Learning quantique
    - Cryptographie post-quantique
    """)
    
    tab1, tab2, tab3 = st.tabs(["🌀 Simulation Quantique", "🎯 QAOA", "🧠 QML"])
    
    with tab1:
        st.subheader("🌀 Simulation Quantique Plasma")
        
        st.write("""
        **Algorithme VQE (Variational Quantum Eigensolver)**
        
        Calculer états fondamentaux systèmes quantiques
        → Simulation interactions plasma à l'échelle quantique
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_qubits = st.slider("Nombre Qubits", 4, 20, 8)
            circuit_depth = st.slider("Profondeur Circuit", 2, 10, 4)
            
            backend = st.selectbox("Backend Quantique",
                ["IBM Quantum", "Ionq", "Rigetti", "Simulateur"])
        
        with col2:
            st.write("### 🎛️ Paramètres")
            
            hamiltonian = st.selectbox("Hamiltonien",
                ["Ising", "Heisenberg", "Hubbard"])
            
            optimizer = st.selectbox("Optimiseur",
                ["COBYLA", "SPSA", "ADAM"])
        
        if st.button("🌀 Lancer Simulation Quantique"):
            with st.spinner("Exécution sur processeur quantique..."):
                import time
                
                # Simuler exécution quantique
                progress = st.progress(0)
                status = st.empty()
                
                for i in range(100):
                    time.sleep(0.02)
                    progress.progress(i + 1)
                    if i % 20 == 0:
                        status.write(f"⚛️ Itération {i//20 + 1}/5")
                
                energy = -1.5 + np.random.normal(0, 0.1)
                
                st.success(f"✅ Simulation complétée!")
                st.metric("Énergie Fondamentale", f"{energy:.4f} a.u.")
                
                # Visualiser état quantique
                st.write("### 📊 État Quantique Final")
                
                amplitudes = np.random.rand(2**min(n_qubits, 4))
                amplitudes = amplitudes / np.linalg.norm(amplitudes)
                
                fig = go.Figure(data=[go.Bar(
                    x=[f"|{format(i, f'0{min(n_qubits, 4)}b')}⟩" for i in range(len(amplitudes))],
                    y=amplitudes**2,
                    marker_color='#FF6B35'
                )])
                
                fig.update_layout(
                    title="Distribution Probabilité États Quantiques",
                    xaxis_title="État",
                    yaxis_title="Probabilité",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 QAOA - Optimisation Quantique")
        
        st.write("""
        **Quantum Approximate Optimization Algorithm**
        
        Optimiser scénarios fusion:
        - Trajectoires plasma
        - Séquences chauffage
        - Contrôle feedback
        """)
        
        problem_size = st.slider("Taille Problème", 4, 16, 8)
        p_layers = st.slider("Couches QAOA (p)", 1, 5, 2)
        
        if st.button("🎯 Optimiser avec QAOA"):
            with st.spinner("Optimisation quantique..."):
                import time
                time.sleep(2)
                
                # Solution simulée
                best_solution = np.random.randint(0, 2**problem_size)
                best_cost = -np.random.uniform(80, 100)
                
                st.success("✅ Solution optimale trouvée!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Solution", format(best_solution, f'0{problem_size}b'))
                    st.metric("Coût", f"{best_cost:.2f}")
                
                with col2:
                    st.write("**Interprétation:**")
                    st.write("Séquence optimale activation systèmes chauffage")
                    st.write(f"Gain Q factor estimé: +{abs(best_cost)/10:.1f}%")
    
    with tab3:
        st.subheader("🧠 Quantum Machine Learning")
        
        st.write("""
        **QML - Classification Quantique**
        
        Classifier régimes plasma avec circuits quantiques variationnels
        """)
        
        n_features = st.slider("Features", 2, 8, 4)
        n_layers = st.slider("Couches Quantiques", 1, 6, 3)
        
        if st.button("🧠 Entraîner Modèle Quantique"):
            with st.spinner("Entraînement QML..."):
                import time
                
                losses = []
                for epoch in range(20):
                    time.sleep(0.1)
                    loss = 0.5 * np.exp(-epoch/5) + 0.05 + np.random.normal(0, 0.02)
                    losses.append(loss)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(20)),
                    y=losses,
                    mode='lines+markers',
                    line=dict(color='#FF6B35', width=2),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Convergence Modèle Quantique",
                    xaxis_title="Époque",
                    yaxis_title="Loss",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                accuracy_quantum = 0.92
                accuracy_classical = 0.88
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy Quantique", f"{accuracy_quantum:.1%}")
                with col2:
                    st.metric("Gain vs Classique", f"+{(accuracy_quantum-accuracy_classical)*100:.1f}%")

# ==================== PAGE: BIOCOMPUTING ====================
elif page == "🧬 Biocomputing":
    st.header("🧬 Biocomputing & Systèmes Hybrides")
    
    st.info("""
    **Biocomputing pour Fusion:**
    - Organoïdes neuronaux comme processeurs
    - Optimisation bio-inspirée
    - Contrôle adaptatif biologique
    - Hybridation silicium-biologique
    """)
    
    tab1, tab2, tab3 = st.tabs(["🧠 Organoïdes Neuronaux", "🔄 Algorithmes Bio", "🔬 Hybride"])
    
    with tab1:
        st.subheader("🧠 Organoïdes Neuronaux pour Contrôle")
        
        st.write("""
        **Wetware Computing**
        
        Utiliser organoïdes cérébraux comme substrat calcul pour:
        - Reconnaissance patterns plasma
        - Contrôle temps réel
        - Apprentissage adaptatif naturel
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            organoid_size = st.selectbox("Taille Organoïde",
                ["Mini (0.5mm, 100k neurones)",
                 "Standard (2mm, 1M neurones)",
                 "Large (5mm, 10M neurones)"])
            
            culture_duration = st.slider("Maturation (jours)", 30, 180, 90)
            
            interface_type = st.selectbox("Interface",
                ["MEA 64 électrodes", "MEA 256 électrodes", "MEA 1024 électrodes"])
        
        with col2:
            st.write("### 🎯 Configuration")
            
            input_channels = 16
            output_channels = 8
            
            st.metric("Canaux Input", input_channels)
            st.metric("Canaux Output", output_channels)
            st.metric("Neurones", "1M")
            
            if st.button("🧬 Connecter Organoïde"):
                st.success("✅ Organoïde connecté au réacteur!")
                st.info("🔄 Calibration interface en cours...")
        
        st.write("### 📊 Performance Biocomputing")
        
        comparison_data = {
            'Système': ['Organoïde', 'GPU (A100)', 'CPU (i9)', 'FPGA'],
            'Puissance (W)': [0.1, 400, 125, 50],
            'Latence (ms)': [50, 10, 20, 5],
            'Adaptatif': ['✅', '❌', '❌', '❌'],
            'Efficacité (TOPS/W)': [1000, 0.5, 0.08, 2]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.subheader("🔄 Algorithmes Bio-Inspirés")
        
        st.write("""
        **Optimisation Évolutionnaire**
        
        Appliquer principes biologiques à l'optimisation:
        - Algorithmes génétiques
        - Essaims particulaires
        - Colonies fourmis
        - Immunité artificielle
        """)
        
        algo_type = st.selectbox("Algorithme",
            ["Génétique", "Essaim Particulaires (PSO)", "Colonies Fourmis (ACO)", "Évolution Différentielle"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            population_size = st.slider("Taille Population", 10, 200, 50)
            generations = st.slider("Générations", 10, 500, 100)
        
        with col2:
            if algo_type == "Génétique":
                mutation_rate = st.slider("Taux Mutation", 0.01, 0.5, 0.1)
                crossover_rate = st.slider("Taux Croisement", 0.5, 1.0, 0.8)
            elif algo_type == "Essaim Particulaires (PSO)":
                inertia = st.slider("Inertie w", 0.1, 1.0, 0.7)
                cognitive = st.slider("Paramètre cognitif c1", 0.5, 3.0, 2.0)
        
        if st.button("🔄 Optimiser avec Bio-Algo"):
            with st.spinner(f"Optimisation {algo_type}..."):
                import time
                
                fitness_history = []
                for gen in range(0, generations, generations//20):
                    time.sleep(0.1)
                    # Convergence simulée
                    fitness = 100 - 95 * (1 - np.exp(-gen/30))
                    fitness_history.append(fitness)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(0, generations, generations//20)),
                    y=fitness_history,
                    mode='lines+markers',
                    line=dict(color='#FF6B35', width=3),
                    marker=dict(size=8),
                    fill='tozeroy'
                ))
                
                fig.update_layout(
                    title=f"Convergence {algo_type}",
                    xaxis_title="Génération",
                    yaxis_title="Fitness (Q factor)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Optimum trouvé: Q = {fitness_history[-1]:.2f}")
    
    with tab3:
        st.subheader("🔬 Systèmes Hybrides Bio-Électroniques")
        
        st.write("""
        **Architecture Hybride**
        
        Combiner:
        - Organoïdes neuronaux (apprentissage adaptatif)
        - IA classique (vitesse, précision)
        - Computing quantique (optimisation)
        """)
        
        st.write("### 🏗️ Architecture Proposée")
        
        architecture = """
        ```
        ┌─────────────────────────────────────────┐
        │         RÉACTEUR FUSION                 │
        │    (Plasma, Diagnostics, Actuators)    │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────▼──────────────────────────┐
        │    INTERFACE TEMPS RÉEL (FPGA)          │
        │    • Acquisition haute fréquence        │
        │    • Pré-traitement signaux             │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────▼──────────────────────────┐
        │    COUCHE ORGANOÏDE NEURONAL            │
        │    • Pattern recognition                │
        │    • Contrôle adaptatif                 │
        │    • 1M neurones, MEA 1024              │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────▼──────────────────────────┐
        │    COUCHE IA CLASSIQUE                  │
        │    • Deep Learning (disruption)         │
        │    • RL Agent (optimisation)            │
        │    • GPU Cluster                        │
        └──────────────┬──────────────────────────┘
                       │
        ┌──────────────▼──────────────────────────┐
        │    COUCHE QUANTIQUE                     │
        │    • QAOA (optimisation globale)        │
        │    • VQE (simulations)                  │
        │    • IBM Quantum / Ionq                 │
        └─────────────────────────────────────────┘
        ```
        """
        
        st.code(architecture, language="text")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🧠 Organoïde**")
            st.write("• Latence: 50ms")
            st.write("• Adaptatif: ✅")
            st.write("• Puissance: 0.1W")
        
        with col2:
            st.write("**🤖 IA Classique**")
            st.write("• Latence: 10ms")
            st.write("• Précision: ✅")
            st.write("• Puissance: 400W")
        
        with col3:
            st.write("**⚛️ Quantique**")
            st.write("• Latence: 100ms")
            st.write("• Optimisation: ✅")
            st.write("• Qubits: 20+")
        
        if st.button("🚀 Activer Architecture Hybride"):
            st.success("✅ Système hybride activé!")
            st.balloons()
            
            st.write("### 📊 Performance Hybride")
            
            metrics_hybrid = {
                'Métrique': ['Q factor', 'Uptime', 'Disruptions évitées', 'Efficacité énergétique'],
                'Standard': [0.67, '85%', '60%', '100%'],
                'Hybride': [0.95, '98%', '95%', '250%']
            }
            
            df_hybrid = pd.DataFrame(metrics_hybrid)
            st.dataframe(df_hybrid, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal Système (20 dernières entrées)"):
    if st.session_state.fusion_lab['log']:
        for event in st.session_state.fusion_lab['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            if level == "INFO":
                icon = "ℹ️"
            elif level == "SUCCESS":
                icon = "✅"
            elif level == "WARNING":
                icon = "⚠️"
            elif level == "CRITICAL":
                icon = "🔴"
            else:
                icon = "❌"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")

st.markdown("---")

# Statistiques finales
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_reactors = len(st.session_state.fusion_lab['reactors'])
    st.metric("⚛️ Réacteurs", total_reactors)

with col2:
    total_shots = len(st.session_state.fusion_lab['plasma_shots'])
    st.metric("💥 Tirs", total_shots)

with col3:
    if st.session_state.fusion_lab['plasma_shots']:
        total_energy = sum([s.get('total_energy_MJ', 0) for s in st.session_state.fusion_lab['plasma_shots']])
        st.metric("⚡ Énergie Totale", f"{total_energy:.1f} MJ")
    else:
        st.metric("⚡ Énergie Totale", "0.0 MJ")

with col4:
    total_experiments = len(st.session_state.fusion_lab['experiments'])
    st.metric("🔬 Expériences", total_experiments)

st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>⚛️ Nuclear Fusion Laboratory Platform</h3>
        <p>Plasma Physics • Tokamaks • Magnetic Confinement • Fusion Energy</p>
        <p><small>Biocomputing • Quantum Computing • AI-Powered • Hybrid Systems</small></p>
        <p><small>Version 2.0.0 | Advanced Edition</small></p>
        <p><small>⚛️ Harnessing the Power of the Stars © 2024</small></p>
    </div>
""", unsafe_allow_html=True)