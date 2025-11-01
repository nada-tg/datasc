"""
Plateforme Avancée Recherche Énergétique
Energy Research Platform - IA • Quantique • Bio-Computing
Fusion • Fission • Renouvelables • Stockage • Optimisation

Installation:
pip install streamlit pandas plotly numpy scipy scikit-learn

Lancement:
streamlit run energy_platform_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# ==================== CONFIGURATION PAGE ====================
st.set_page_config(
    page_title="⚡ Energy Research Platform",
    page_icon="⚡",
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
        background: linear-gradient(90deg, #FFD700 0%, #FF8C00 30%, #FF4500 60%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: energy-pulse 2s ease-in-out infinite alternate;
    }
    @keyframes energy-pulse {
        from { filter: drop-shadow(0 0 10px #FFD700); }
        to { filter: drop-shadow(0 0 30px #FF4500); }
    }
    .energy-card {
        border: 3px solid #FFD700;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 140, 0, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(255, 215, 0, 0.4);
        transition: all 0.3s;
    }
    .energy-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(255, 140, 0, 0.6);
    }
    .tech-badge-energy {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
        background: linear-gradient(90deg, #FF8C00 0%, #FF4500 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.4);
    }
    .power-meter {
        animation: power-flow 3s infinite;
    }
    @keyframes power-flow {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ÉNERGÉTIQUES ====================
ENERGY_CONSTANTS = {
    'c': 299792458,  # m/s
    'h': 6.62607015e-34,  # J⋅s
    'e': 1.602176634e-19,  # Coulomb
    'NA': 6.02214076e23,  # mol⁻¹
    'k_B': 1.380649e-23,  # J/K
    'electron_mass': 9.10938356e-31,  # kg
    'proton_mass': 1.6726219e-27,  # kg
    'fusion_energy_d_t': 17.6,  # MeV (Deutérium-Tritium)
    'fission_energy_u235': 200,  # MeV
}

ENERGY_SOURCES = {
    'Fusion Nucléaire': {'potential': 'Illimité', 'efficiency': 0.4, 'emissions': 'Zéro'},
    'Fission Avancée': {'potential': 'Élevé', 'efficiency': 0.33, 'emissions': 'Faible'},
    'Solaire': {'potential': 'Très Élevé', 'efficiency': 0.22, 'emissions': 'Zéro'},
    'Éolien': {'potential': 'Élevé', 'efficiency': 0.45, 'emissions': 'Zéro'},
    'Hydrogène': {'potential': 'Élevé', 'efficiency': 0.60, 'emissions': 'Zéro'},
    'Géothermique': {'potential': 'Moyen', 'efficiency': 0.15, 'emissions': 'Très Faible'},
}

STORAGE_TECHNOLOGIES = {
    'Batteries Li-ion': {'density_wh_kg': 250, 'cycles': 3000, 'efficiency': 0.95},
    'Batteries Solid-State': {'density_wh_kg': 500, 'cycles': 10000, 'efficiency': 0.98},
    'Supercondensateurs': {'density_wh_kg': 15, 'cycles': 1000000, 'efficiency': 0.99},
    'Hydrogène': {'density_wh_kg': 33000, 'cycles': 50000, 'efficiency': 0.60},
    'Volants Inertie': {'density_wh_kg': 100, 'cycles': 100000, 'efficiency': 0.90},
}

# ==================== INITIALISATION SESSION STATE ====================
if 'energy_lab' not in st.session_state:
    st.session_state.energy_lab = {
        'reactors': {},
        'power_plants': {},
        'storage_systems': {},
        'smart_grids': {},
        'ai_models': {},
        'quantum_simulations': [],
        'bio_batteries': {},
        'fusion_experiments': [],
        'production_data': [],
        'consumption_data': [],
        'optimizations': [],
        'materials': {},
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================
def log_event(message: str, level: str = "INFO"):
    """Enregistrer événement"""
    st.session_state.energy_lab['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_fusion_energy(fuel_mass_kg: float, fuel_type: str = "D-T") -> float:
    """Calculer énergie fusion nucléaire"""
    # Énergie par réaction (MeV)
    energy_per_reaction = {
        'D-T': 17.6,  # Deutérium-Tritium
        'D-D': 3.27,  # Deutérium-Deutérium
        'D-He3': 18.3,  # Deutérium-Hélium-3
        'p-B11': 8.7   # Proton-Bore-11
    }
    
    energy_mev = energy_per_reaction.get(fuel_type, 17.6)
    
    # Nombre de réactions
    avogadro = 6.022e23
    molar_mass = 5  # g/mol approximatif
    n_moles = (fuel_mass_kg * 1000) / molar_mass
    n_reactions = n_moles * avogadro / 2  # 2 noyaux par réaction
    
    # Énergie totale (Joules)
    energy_j = n_reactions * energy_mev * 1.602e-13
    
    # Convertir en GWh
    energy_gwh = energy_j / 3.6e12
    
    return energy_gwh

def calculate_fission_energy(fuel_mass_kg: float) -> float:
    """Calculer énergie fission U-235"""
    # ~200 MeV par fission
    # 1 kg U-235 ≈ 24,000 MWh
    energy_mwh = fuel_mass_kg * 24000
    return energy_mwh / 1000  # GWh

def simulate_solar_production(capacity_mw: float, hours: int, location: str = "Optimal") -> List[float]:
    """Simuler production solaire"""
    # Facteurs selon localisation
    sun_factors = {
        'Optimal': 1.0,
        'Desert': 0.95,
        'Tropical': 0.85,
        'Temperate': 0.75,
        'Northern': 0.60
    }
    
    factor = sun_factors.get(location, 0.75)
    
    production = []
    for hour in range(hours):
        # Cycle jour/nuit (sinusoïde)
        hour_of_day = hour % 24
        
        if 6 <= hour_of_day <= 18:  # Jour
            sun_intensity = np.sin((hour_of_day - 6) * np.pi / 12) * factor
            noise = np.random.normal(0, 0.05)
            power = capacity_mw * sun_intensity * (1 + noise)
        else:  # Nuit
            power = 0
        
        production.append(max(0, power))
    
    return production

def ai_optimize_grid(supply: List[float], demand: List[float]) -> Dict:
    """Optimiser réseau avec IA"""
    supply_arr = np.array(supply)
    demand_arr = np.array(demand)
    
    # Balance énergétique
    balance = supply_arr - demand_arr
    
    # Stockage nécessaire
    storage_needed = np.maximum(-balance, 0)
    storage_available = np.maximum(balance, 0)
    
    # Optimisation
    total_deficit = np.sum(storage_needed)
    total_surplus = np.sum(storage_available)
    
    efficiency = 1 - (total_deficit / np.sum(demand_arr))
    
    return {
        'balance': balance.tolist(),
        'storage_needed': storage_needed.tolist(),
        'storage_available': storage_available.tolist(),
        'efficiency': efficiency,
        'deficit_total': total_deficit,
        'surplus_total': total_surplus
    }

def quantum_optimize_reactor(temperature_k: float, pressure_atm: float, fuel_density: float) -> Dict:
    """Optimiser réacteur avec computing quantique"""
    # Simulation optimisation quantique
    # En réalité utiliserait algorithme VQE ou QAOA
    
    # Score performance basé sur paramètres
    temp_score = 1 / (1 + abs(temperature_k - 150e6) / 50e6)
    pressure_score = pressure_atm / 10
    density_score = fuel_density / 1e20
    
    performance = (temp_score + pressure_score + density_score) / 3
    
    # Paramètres optimaux
    optimal_temp = 150e6 + np.random.normal(0, 10e6)
    optimal_pressure = 5 + np.random.normal(0, 0.5)
    optimal_density = 1e20 + np.random.normal(0, 1e19)
    
    gain = np.random.uniform(1.1, 1.5)
    
    return {
        'performance_score': performance,
        'optimal_temperature': optimal_temp,
        'optimal_pressure': optimal_pressure,
        'optimal_density': optimal_density,
        'energy_gain_factor': gain,
        'quantum_advantage': f"{gain:.2f}x"
    }

def bio_generate_electricity(bio_fuel_kg: float, efficiency: float = 0.40) -> float:
    """Générer électricité par bio-computing"""
    # Énergie combustion biomasse ~15-20 MJ/kg
    energy_mj = bio_fuel_kg * 17.5
    energy_kwh = (energy_mj / 3.6) * efficiency
    return energy_kwh

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">⚡ Energy Research Platform</h1>', unsafe_allow_html=True)
st.markdown("### Recherche Énergétique Avancée • IA • Quantique • Bio-Computing • Fusion • Renouvelables")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/FFD700/000000?text=Energy+Lab", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Centre Contrôle",
            "⚛️ Fusion Nucléaire",
            "🔬 Fission Avancée",
            "☀️ Solaire Intelligent",
            "💨 Éolien Optimisé",
            "💧 Hydrogène",
            "🔋 Stockage Énergie",
            "🌐 Smart Grid",
            "🤖 IA Optimisation",
            "⚛️ Computing Quantique",
            "🧬 Bio-Batteries",
            "🔬 Matériaux Avancés",
            "📊 Production",
            "📈 Consommation",
            "⚡ Distribution",
            "🌍 Impact Carbone",
            "💰 Économie Énergie",
            "🔮 Prédictions",
            "📊 Analytics",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_reactors = len(st.session_state.energy_lab['reactors'])
    total_plants = len(st.session_state.energy_lab['power_plants'])
    total_storage = len(st.session_state.energy_lab['storage_systems'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚛️ Réacteurs", total_reactors)
        st.metric("🏭 Centrales", total_plants)
    with col2:
        st.metric("🔋 Stockage", total_storage)
        st.metric("🌐 Grids", len(st.session_state.energy_lab['smart_grids']))

# ==================== PAGE: CENTRE CONTRÔLE ====================
if page == "🏠 Centre Contrôle":
    st.header("🏠 Centre de Contrôle Énergétique")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="energy-card"><h2>⚛️</h2><h3>{total_reactors}</h3><p>Réacteurs</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        total_capacity_gw = total_plants * 1.2  # Simulation
        st.markdown(f'<div class="energy-card"><h2>⚡</h2><h3>{total_capacity_gw:.1f}</h3><p>GW Capacité</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="energy-card"><h2>🔋</h2><h3>{total_storage}</h3><p>Stockage</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        efficiency = 87.5
        st.markdown(f'<div class="energy-card"><h2>📊</h2><h3>{efficiency}%</h3><p>Efficacité</p></div>', 
                   unsafe_allow_html=True)
    
    with col5:
        emissions = 0.15
        st.markdown(f'<div class="energy-card"><h2>🌍</h2><h3>{emissions}</h3><p>CO₂ (Mt)</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mix énergétique
    st.subheader("⚡ Mix Énergétique Global")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        energy_mix = {
            'Source': ['Fusion', 'Fission', 'Solaire', 'Éolien', 'Hydrogène', 'Géothermie', 'Autres'],
            'Production_GWh': [1250, 2100, 1800, 1500, 900, 600, 350],
            'Part': [15, 25, 21, 18, 11, 7, 3]
        }
        
        df_mix = pd.DataFrame(energy_mix)
        
        fig = go.Figure(data=[go.Pie(
            labels=df_mix['Source'],
            values=df_mix['Production_GWh'],
            hole=.4,
            marker=dict(colors=['#FFD700', '#FF8C00', '#FFA500', '#00CED1', '#32CD32', '#8B4513', '#808080'])
        )])
        
        fig.update_layout(
            title="Production Énergétique par Source",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("### 🎯 Objectifs 2030")
        
        objectives = {
            "Renouvelables": "80%",
            "Émissions CO₂": "-70%",
            "Efficacité": "+25%",
            "Stockage": "500 GWh"
        }
        
        for obj, target in objectives.items():
            st.metric(obj, target)
        
        st.write("\n### 🌟 Innovations")
        st.write("✅ Fusion commerciale")
        st.write("✅ Batteries solid-state")
        st.write("✅ Hydrogène vert")
        st.write("✅ Smart grids IA")
    
    st.markdown("---")
    
    # Production temps réel
    st.subheader("⚡ Production en Temps Réel (24h)")
    
    hours = list(range(24))
    production = [50 + 30*np.sin((h-6)*np.pi/12) + np.random.uniform(-5, 5) if 6 <= h <= 20 else 30 + np.random.uniform(-3, 3) for h in hours]
    demand = [45 + 25*np.sin((h-8)*np.pi/10) + np.random.uniform(-3, 3) for h in hours]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=production,
        mode='lines+markers',
        name='Production',
        line=dict(color='#FFD700', width=3),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=demand,
        mode='lines+markers',
        name='Demande',
        line=dict(color='#FF4500', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="Production vs Demande",
        xaxis_title="Heure",
        yaxis_title="Puissance (GW)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Technologies avancées
    st.subheader("🚀 Technologies Intégrées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 🤖 Intelligence Artificielle")
        st.write("✅ Prédiction demande temps réel")
        st.write("✅ Optimisation smart grids")
        st.write("✅ Maintenance prédictive")
        st.write("✅ Gestion stockage intelligente")
        st.write("✅ Trading énergie automatisé")
    
    with col2:
        st.write("### ⚛️ Computing Quantique")
        st.write("✅ Optimisation réacteurs fusion")
        st.write("✅ Simulation matériaux")
        st.write("✅ Design catalyseurs H₂")
        st.write("✅ Prévisions météo énergie")
        st.write("✅ Cryptographie réseau")
    
    with col3:
        st.write("### 🧬 Bio-Computing")
        st.write("✅ Batteries organiques")
        st.write("✅ Biocarburants avancés")
        st.write("✅ Capture CO₂ biologique")
        st.write("✅ Production hydrogène enzymes")
        st.write("✅ Stockage ADN données")

# ==================== PAGE: FUSION NUCLÉAIRE ====================
elif page == "⚛️ Fusion Nucléaire":
    st.header("⚛️ Recherche Fusion Nucléaire")
    
    st.info("""
    **Fusion Thermonucléaire Contrôlée**
    
    La fusion nucléaire reproduit le processus énergétique des étoiles.
    Énergie quasi-illimitée, propre, sans déchets radioactifs longue durée.
    
    **Réactions principales:**
    - D + T → He-4 + n + 17.6 MeV (Deutérium-Tritium)
    - D + D → He-3 + n + 3.27 MeV
    - D + He-3 → He-4 + p + 18.3 MeV
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Réacteurs", "➕ Créer Réacteur", "📊 Expériences", "📈 Résultats"])
    
    with tab1:
        st.subheader("🔬 Réacteurs Fusion Actifs")
        
        if not st.session_state.energy_lab['reactors']:
            st.info("Aucun réacteur créé. Créez votre premier réacteur!")
            
            if st.button("➕ Créer Premier Réacteur", type="primary"):
                st.info("Accédez à l'onglet 'Créer Réacteur'")
        else:
            for reactor_id, reactor in st.session_state.energy_lab['reactors'].items():
                with st.expander(f"⚛️ {reactor['name']} ({reactor['type']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Paramètres")
                        st.write(f"**Type:** {reactor['type']}")
                        st.write(f"**Combustible:** {reactor['fuel']}")
                        st.write(f"**Température:** {reactor['temperature_k']/1e6:.0f} M°K")
                        st.write(f"**Pression:** {reactor['pressure_atm']:.1f} atm")
                        
                        status_icon = "🟢" if reactor['status'] == 'active' else "🔴"
                        st.write(f"**Statut:** {status_icon} {reactor['status']}")
                    
                    with col2:
                        st.write("### ⚡ Performance")
                        st.metric("Q Factor", f"{reactor.get('q_factor', 0):.2f}")
                        st.metric("Gain Énergie", f"{reactor.get('energy_gain', 0):.1f}x")
                        st.metric("Puissance", f"{reactor.get('power_output_mw', 0):.0f} MW")
                        st.metric("Temps Confinement", f"{reactor.get('confinement_time_s', 0):.3f} s")
                    
                    with col3:
                        st.write("### 🎯 Objectifs")
                        st.write("**Q > 10:** " + ("✅" if reactor.get('q_factor', 0) > 10 else "❌"))
                        st.write("**Ignition:** " + ("✅" if reactor.get('ignition', False) else "❌"))
                        st.write("**Commercial:** " + ("✅" if reactor.get('commercial', False) else "❌"))
                    
                    st.markdown("---")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("🚀 Lancer Pulse", key=f"pulse_{reactor_id}"):
                            st.success("Pulse fusion lancé!")
                    
                    with col2:
                        if st.button("⚙️ Optimiser", key=f"opt_{reactor_id}"):
                            st.info("Optimisation quantique...")
                    
                    with col3:
                        if st.button("📊 Diagnostics", key=f"diag_{reactor_id}"):
                            st.info("Diagnostics plasma...")
                    
                    with col4:
                        if st.button("🗑️ Supprimer", key=f"del_{reactor_id}"):
                            del st.session_state.energy_lab['reactors'][reactor_id]
                            log_event(f"Réacteur supprimé: {reactor['name']}", "WARNING")
                            st.rerun()
    
    with tab2:
        st.subheader("➕ Créer Nouveau Réacteur Fusion")
        
        with st.form("create_fusion_reactor"):
            st.write("### 🎨 Configuration Réacteur")
            
            col1, col2 = st.columns(2)
            
            with col1:
                reactor_name = st.text_input("Nom Réacteur", "TOKAMAK-01")
                
                reactor_type = st.selectbox("Type Confinement",
                    ["Tokamak", "Stellarator", "Inertiel (Laser)", "Z-Pinch", "Field-Reversed"])
                
                fuel_type = st.selectbox("Combustible",
                    ["D-T (Deutérium-Tritium)", "D-D (Deutérium-Deutérium)", 
                     "D-He3 (Deutérium-Hélium3)", "p-B11 (Proton-Bore11)"])
            
            with col2:
                temperature_mk = st.number_input("Température Plasma (M°K)", 50, 500, 150, 10)
                
                pressure_atm = st.slider("Pression (atm)", 1.0, 20.0, 5.0, 0.5)
                
                magnetic_field_t = st.slider("Champ Magnétique (Tesla)", 1, 20, 5, 1)
            
            st.write("### ⚙️ Paramètres Avancés")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                plasma_current_ma = st.number_input("Courant Plasma (MA)", 1, 20, 15, 1)
                fuel_mass_kg = st.number_input("Masse Combustible (kg)", 0.001, 1.0, 0.1, 0.001)
            
            with col2:
                confinement_time_s = st.number_input("Temps Confinement (s)", 0.1, 10.0, 1.0, 0.1)
                heating_power_mw = st.number_input("Puissance Chauffage (MW)", 10, 200, 50, 10)
            
            with col3:
                ai_control = st.checkbox("🤖 Contrôle IA", value=True)
                quantum_opt = st.checkbox("⚛️ Optimisation Quantique", value=True)
            
            if st.form_submit_button("⚛️ Créer Réacteur Fusion", type="primary"):
                if not reactor_name:
                    st.error("⚠️ Veuillez donner un nom")
                else:
                    reactor_id = f"reactor_{len(st.session_state.energy_lab['reactors']) + 1}"
                    
                    # Calculer Q factor (gain énergie)
                    # Q = Puissance Fusion / Puissance Chauffage
                    # Simplifié pour démo
                    fusion_power = heating_power_mw * np.random.uniform(5, 15)
                    q_factor = fusion_power / heating_power_mw
                    
                    reactor = {
                        'id': reactor_id,
                        'name': reactor_name,
                        'type': reactor_type.split()[0],
                        'fuel': fuel_type.split()[0],
                        'temperature_k': temperature_mk * 1e6,
                        'pressure_atm': pressure_atm,
                        'magnetic_field_t': magnetic_field_t,
                        'plasma_current_ma': plasma_current_ma,
                        'fuel_mass_kg': fuel_mass_kg,
                        'confinement_time_s': confinement_time_s,
                        'heating_power_mw': heating_power_mw,
                        'fusion_power_mw': fusion_power,
                        'q_factor': q_factor,
                        'energy_gain': q_factor,
                        'ignition': q_factor > 10,
                        'commercial': q_factor > 20,
                        'ai_control': ai_control,
                        'quantum_opt': quantum_opt,
                        'status': 'active',
                        'created_at': datetime.now().isoformat(),
                        'power_output_mw': fusion_power - heating_power_mw
                    }
                    
                    st.session_state.energy_lab['reactors'][reactor_id] = reactor
                    log_event(f"Réacteur fusion créé: {reactor_name}", "SUCCESS")
                    
                    with st.spinner("Initialisation réacteur..."):
                        import time
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)
                        progress_bar.empty()
                    
                    st.success(f"✅ Réacteur '{reactor_name}' créé!")
                    st.balloons()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Q Factor", f"{q_factor:.2f}")
                    with col2:
                        st.metric("Température", f"{temperature_mk} M°K")
                    with col3:
                        st.metric("Puissance Nette", f"{reactor['power_output_mw']:.0f} MW")
                    with col4:
                        ignition_status = "🎉 OUI" if reactor['ignition'] else "❌ Non"
                        st.metric("Ignition", ignition_status)
                    
                    if q_factor > 10:
                        st.success("🎉 IGNITION ATTEINTE! Q > 10")
                    
                    if quantum_opt:
                        st.info("⚛️ Optimisation quantique disponible")
                    
                    st.rerun()
    
    with tab3:
        st.subheader("📊 Expériences Fusion")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### 🔬 Lancer Pulse Fusion")
            
            if not st.session_state.energy_lab['reactors']:
                st.warning("⚠️ Créez d'abord un réacteur")
            else:
                selected_reactor = st.selectbox("Sélectionner Réacteur",
                    list(st.session_state.energy_lab['reactors'].keys()),
                    format_func=lambda x: st.session_state.energy_lab['reactors'][x]['name'])
                
                reactor = st.session_state.energy_lab['reactors'][selected_reactor]
                
                pulse_duration_ms = st.slider("Durée Pulse (ms)", 10, 10000, 1000, 10)
                
                if st.button("🚀 Lancer Pulse Fusion", type="primary", use_container_width=True):
                    with st.spinner(f"Pulse fusion {pulse_duration_ms}ms..."):
                        import time
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Chauffage plasma...")
                        time.sleep(0.5)
                        progress_bar.progress(0.3)
                        
                        status_text.text("Confinement magnétique...")
                        time.sleep(0.5)
                        progress_bar.progress(0.6)
                        
                        status_text.text("Fusion en cours...")
                        time.sleep(0.5)
                        progress_bar.progress(0.9)
                        
                        # Calculer énergie produite
                        energy_gwh = calculate_fusion_energy(reactor['fuel_mass_kg'], reactor['fuel'])
                        
                        status_text.text("Collecte données...")
                        time.sleep(0.3)
                        progress_bar.progress(1.0)
                        
                        time.sleep(0.2)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Sauvegarder expérience
                        experiment = {
                            'timestamp': datetime.now().isoformat(),
                            'reactor_id': selected_reactor,
                            'pulse_duration_ms': pulse_duration_ms,
                            'energy_produced_gwh': energy_gwh * (pulse_duration_ms / 3600000),
                            'q_factor': reactor['q_factor'],
                            'temperature_k': reactor['temperature_k'],
                            'success': True
                        }
                        
                        st.session_state.energy_lab['fusion_experiments'].append(experiment)
                    
                    st.success(f"✅ Pulse réussi!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        energy_mwh = energy_gwh * 1000 * (pulse_duration_ms / 3600000)
                        st.metric("Énergie Produite", f"{energy_mwh:.2f} MWh")
                    with col2:
                        st.metric("Q Factor", f"{reactor['q_factor']:.2f}")
                    with col3:
                        st.metric("Température", f"{reactor['temperature_k']/1e6:.0f} M°K")
                    
                    log_event(f"Pulse fusion: {energy_mwh:.2f} MWh", "SUCCESS")
        
        with col2:
            st.write("### 📊 Expériences Récentes")
            
            if st.session_state.energy_lab['fusion_experiments']:
                for exp in st.session_state.energy_lab['fusion_experiments'][-5:][::-1]:
                    st.write(f"⚛️ {exp['timestamp'][:19]}")
                    st.write(f"Énergie: {exp['energy_produced_gwh']*1000:.2f} MWh")
                    st.write("---")
            else:
                st.info("Aucune expérience")
    
    with tab4:
        st.subheader("📈 Analyse Résultats")
        
        if len(st.session_state.energy_lab['fusion_experiments']) > 0:
            # Graphique évolution Q factor
            experiments = st.session_state.energy_lab['fusion_experiments']
            
            q_factors = [e['q_factor'] for e in experiments]
            energies = [e['energy_produced_gwh'] * 1000 for e in experiments]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Scatter(
                    y=q_factors,
                    mode='lines+markers',
                    line=dict(color='#FFD700', width=3),
                    marker=dict(size=10)
                )])
                
                fig.add_hline(y=1, line_dash="dash", line_color="red", annotation_text="Breakeven")
                fig.add_hline(y=10, line_dash="dash", line_color="green", annotation_text="Ignition")
                
                fig.update_layout(
                    title="Évolution Q Factor",
                    xaxis_title="Expérience #",
                    yaxis_title="Q Factor",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[go.Bar(
                    y=energies,
                    marker_color='#FF8C00'
                )])
                
                fig.update_layout(
                    title="Énergie Produite par Expérience",
                    xaxis_title="Expérience #",
                    yaxis_title="Énergie (MWh)",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            st.write("### 📊 Statistiques Globales")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Expériences", len(experiments))
            with col2:
                st.metric("Q Moyen", f"{np.mean(q_factors):.2f}")
            with col3:
                st.metric("Q Max", f"{np.max(q_factors):.2f}")
            with col4:
                st.metric("Énergie Totale", f"{np.sum(energies):.1f} MWh")
        else:
            st.info("Lancez des expériences pour voir les résultats")

# ==================== PAGE: STOCKAGE ÉNERGIE ====================
elif page == "🔋 Stockage Énergie":
    st.header("🔋 Technologies Stockage Énergie")
    
    st.info("""
    **Systèmes de Stockage Avancés**
    
    Le stockage est crucial pour l'intégration des énergies renouvelables.
    Technologies: batteries, hydrogène, volants d'inertie, air comprimé, etc.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔋 Technologies", "➕ Créer Système", "📊 Performance"])
    
    with tab1:
        st.subheader("🔋 Technologies Disponibles")
        
        for tech, specs in STORAGE_TECHNOLOGIES.items():
            with st.expander(f"🔋 {tech}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Densité Énergie", f"{specs['density_wh_kg']} Wh/kg")
                with col2:
                    st.metric("Cycles de Vie", f"{specs['cycles']:,}")
                with col3:
                    st.metric("Efficacité", f"{specs['efficiency']*100:.0f}%")
                
                # Calculer coût énergétique
                cost_per_kwh = np.random.uniform(100, 500)
                st.write(f"**Coût estimé:** ${cost_per_kwh:.0f}/kWh")
                
                if st.button(f"📊 Voir Détails", key=f"details_{tech}"):
                    st.info(f"Détails techniques pour {tech}")
    
    with tab2:
        st.subheader("➕ Créer Système Stockage")
        
        with st.form("create_storage"):
            col1, col2 = st.columns(2)
            
            with col1:
                storage_name = st.text_input("Nom Système", "Battery Farm 01")
                
                technology = st.selectbox("Technologie",
                    list(STORAGE_TECHNOLOGIES.keys()))
                
                capacity_mwh = st.number_input("Capacité (MWh)", 1, 10000, 100, 10)
            
            with col2:
                power_mw = st.number_input("Puissance (MW)", 1, 1000, 50, 10)
                
                location = st.text_input("Localisation", "Grid Node A")
                
                ai_managed = st.checkbox("🤖 Gestion IA", value=True)
            
            if st.form_submit_button("🔋 Créer Système", type="primary"):
                storage_id = f"storage_{len(st.session_state.energy_lab['storage_systems']) + 1}"
                
                specs = STORAGE_TECHNOLOGIES[technology]
                
                storage_system = {
                    'id': storage_id,
                    'name': storage_name,
                    'technology': technology,
                    'capacity_mwh': capacity_mwh,
                    'power_mw': power_mw,
                    'location': location,
                    'ai_managed': ai_managed,
                    'specs': specs,
                    'current_charge': capacity_mwh * 0.5,  # 50% initial
                    'cycles_used': 0,
                    'efficiency': specs['efficiency'],
                    'status': 'operational',
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.energy_lab['storage_systems'][storage_id] = storage_system
                log_event(f"Système stockage créé: {storage_name}", "SUCCESS")
                
                st.success(f"✅ Système '{storage_name}' créé!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Capacité", f"{capacity_mwh} MWh")
                with col2:
                    st.metric("Puissance", f"{power_mw} MW")
                with col3:
                    duration_h = capacity_mwh / power_mw
                    st.metric("Durée", f"{duration_h:.1f} h")
                
                st.rerun()
    
    with tab3:
        st.subheader("📊 Performance Stockage")
        
        if st.session_state.energy_lab['storage_systems']:
            for storage_id, storage in st.session_state.energy_lab['storage_systems'].items():
                with st.expander(f"🔋 {storage['name']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        charge_pct = (storage['current_charge'] / storage['capacity_mwh']) * 100
                        st.metric("Charge", f"{charge_pct:.0f}%")
                        st.progress(charge_pct / 100)
                    
                    with col2:
                        st.metric("Cycles", f"{storage['cycles_used']:,}")
                        remaining = storage['specs']['cycles'] - storage['cycles_used']
                        st.write(f"Restants: {remaining:,}")
                    
                    with col3:
                        st.metric("Efficacité", f"{storage['efficiency']*100:.0f}%")
                        st.metric("Statut", storage['status'])
                    
                    # Actions
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("⚡ Charger", key=f"charge_{storage_id}"):
                            storage['current_charge'] = min(
                                storage['current_charge'] + 10,
                                storage['capacity_mwh']
                            )
                            st.success("Charge +10 MWh")
                            st.rerun()
                    
                    with col2:
                        if st.button("🔋 Décharger", key=f"discharge_{storage_id}"):
                            storage['current_charge'] = max(
                                storage['current_charge'] - 10,
                                0
                            )
                            storage['cycles_used'] += 1
                            st.info("Décharge -10 MWh")
                            st.rerun()
                    
                    with col3:
                        if st.button("🗑️ Supprimer", key=f"del_storage_{storage_id}"):
                            del st.session_state.energy_lab['storage_systems'][storage_id]
                            st.rerun()
        else:
            st.info("Aucun système de stockage créé")

# ==================== PAGE: SMART GRID ====================
elif page == "🌐 Smart Grid":
    st.header("🌐 Réseaux Intelligents (Smart Grids)")
    
    st.info("""
    **Smart Grid avec IA**
    
    Réseaux électriques intelligents optimisés par IA pour:
    - Équilibrage offre/demande en temps réel
    - Intégration énergies renouvelables
    - Gestion stockage distribué
    - Réduction pertes transmission
    """)
    
    tab1, tab2, tab3 = st.tabs(["🌐 Vue Réseau", "🤖 Optimisation IA", "📊 Monitoring"])
    
    with tab1:
        st.subheader("🌐 Topologie Réseau")
        
        st.write("### 🗺️ Carte Réseau Énergétique")
        
        # Simulation réseau
        n_nodes = 50
        np.random.seed(42)
        
        nodes_data = {
            'x': np.random.uniform(0, 100, n_nodes),
            'y': np.random.uniform(0, 100, n_nodes),
            'type': np.random.choice(['Production', 'Consommation', 'Stockage'], n_nodes, p=[0.3, 0.5, 0.2]),
            'power': np.random.uniform(10, 100, n_nodes)
        }
        
        color_map = {'Production': '#00FF00', 'Consommation': '#FF0000', 'Stockage': '#FFD700'}
        colors = [color_map[t] for t in nodes_data['type']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=nodes_data['x'],
            y=nodes_data['y'],
            mode='markers',
            marker=dict(
                size=nodes_data['power'] / 5,
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=[f"{t}<br>{p:.0f} MW" for t, p in zip(nodes_data['type'], nodes_data['power'])],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Réseau Smart Grid (50 nœuds)",
            xaxis_title="X",
            yaxis_title="Y",
            template="plotly_dark",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            production_nodes = sum(1 for t in nodes_data['type'] if t == 'Production')
            st.metric("Nœuds Production", production_nodes)
        
        with col2:
            consumption_nodes = sum(1 for t in nodes_data['type'] if t == 'Consommation')
            st.metric("Nœuds Consommation", consumption_nodes)
        
        with col3:
            storage_nodes = sum(1 for t in nodes_data['type'] if t == 'Stockage')
            st.metric("Nœuds Stockage", storage_nodes)
    
    with tab2:
        st.subheader("🤖 Optimisation IA du Réseau")
        
        st.write("### ⚡ Équilibrage Offre/Demande")
        
        # Générer données
        hours = 24
        supply = simulate_solar_production(1000, hours, "Optimal")
        demand = [800 + 200*np.sin((h-8)*np.pi/10) + np.random.uniform(-30, 30) for h in range(hours)]
        
        if st.button("🤖 Optimiser avec IA", type="primary", use_container_width=True):
            with st.spinner("Optimisation IA en cours..."):
                import time
                time.sleep(2)
                
                # Optimiser
                optimization = ai_optimize_grid(supply, demand)
                
                st.success("✅ Optimisation terminée!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Efficacité", f"{optimization['efficiency']*100:.1f}%")
                with col2:
                    st.metric("Surplus", f"{optimization['surplus_total']:.0f} MWh")
                with col3:
                    st.metric("Déficit", f"{optimization['deficit_total']:.0f} MWh")
                
                # Graphique
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=supply,
                    name='Production',
                    line=dict(color='#00FF00', width=3)
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=demand,
                    name='Demande',
                    line=dict(color='#FF0000', width=3, dash='dash')
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(24)),
                    y=optimization['balance'],
                    name='Balance',
                    fill='tozeroy',
                    line=dict(color='#FFD700', width=2)
                ))
                
                fig.update_layout(
                    title="Optimisation IA - Balance Énergétique",
                    xaxis_title="Heure",
                    yaxis_title="Puissance (MW)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                log_event(f"Optimisation IA: {optimization['efficiency']*100:.1f}% efficacité", "SUCCESS")
    
    with tab3:
        st.subheader("📊 Monitoring Temps Réel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### ⚡ Flux Énergétiques")
            
            # Simulation flux
            current_production = np.random.uniform(800, 1200)
            current_demand = np.random.uniform(700, 1100)
            current_storage = np.random.uniform(-200, 200)
            
            st.metric("Production", f"{current_production:.0f} MW", f"{np.random.uniform(-50, 50):.0f} MW")
            st.metric("Demande", f"{current_demand:.0f} MW", f"{np.random.uniform(-30, 30):.0f} MW")
            st.metric("Stockage", f"{current_storage:+.0f} MW")
            
            balance = current_production - current_demand
            
            if balance > 0:
                st.success(f"✅ Surplus: {balance:.0f} MW")
            else:
                st.warning(f"⚠️ Déficit: {abs(balance):.0f} MW")
        
        with col2:
            st.write("### 📊 Qualité Réseau")
            
            frequency_hz = 50 + np.random.normal(0, 0.05)
            voltage_kv = 400 + np.random.normal(0, 5)
            power_factor = 0.95 + np.random.normal(0, 0.02)
            
            st.metric("Fréquence", f"{frequency_hz:.3f} Hz")
            
            if 49.9 <= frequency_hz <= 50.1:
                st.success("✅ Normale")
            else:
                st.warning("⚠️ Hors limites")
            
            st.metric("Tension", f"{voltage_kv:.1f} kV")
            st.metric("Facteur Puissance", f"{power_factor:.3f}")

# ==================== PAGE: SOLAIRE INTELLIGENT ====================
elif page == "☀️ Solaire Intelligent":
    st.header("☀️ Énergie Solaire Intelligente")
    
    st.info("""
    **Photovoltaïque Avancé + IA**
    
    - Panneaux haute efficacité (>25%)
    - Tracking solaire optimisé par IA
    - Prédiction production météo
    - Intégration stockage intelligent
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["☀️ Installations", "📊 Production", "🤖 IA Prédiction", "⚙️ Optimisation"])
    
    with tab1:
        st.subheader("☀️ Créer Installation Solaire")
        
        with st.form("create_solar"):
            col1, col2 = st.columns(2)
            
            with col1:
                solar_name = st.text_input("Nom Installation", "Solar Farm Alpha")
                
                capacity_mw = st.number_input("Capacité (MW)", 1, 10000, 100, 10)
                
                location = st.selectbox("Localisation",
                    ["Optimal", "Desert", "Tropical", "Temperate", "Northern"])
                
                panel_type = st.selectbox("Type Panneaux",
                    ["Silicium Monocristallin", "Silicium Polycristallin", 
                     "Pérovskite", "Tandem", "Organique"])
            
            with col2:
                efficiency = st.slider("Efficacité (%)", 15, 35, 22, 1)
                
                tracking = st.selectbox("Tracking",
                    ["Fixe", "1-Axe", "2-Axes", "IA Optimisé"])
                
                storage_mwh = st.number_input("Stockage Intégré (MWh)", 0, 5000, 500, 50)
                
                ai_prediction = st.checkbox("🤖 Prédiction IA", value=True)
            
            if st.form_submit_button("☀️ Créer Installation", type="primary"):
                plant_id = f"solar_{len(st.session_state.energy_lab['power_plants']) + 1}"
                
                solar_plant = {
                    'id': plant_id,
                    'name': solar_name,
                    'type': 'Solar',
                    'capacity_mw': capacity_mw,
                    'location': location,
                    'panel_type': panel_type,
                    'efficiency': efficiency / 100,
                    'tracking': tracking,
                    'storage_mwh': storage_mwh,
                    'ai_prediction': ai_prediction,
                    'status': 'operational',
                    'production_history': [],
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.energy_lab['power_plants'][plant_id] = solar_plant
                log_event(f"Installation solaire créée: {solar_name}", "SUCCESS")
                
                st.success(f"✅ Installation '{solar_name}' créée!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Capacité", f"{capacity_mw} MW")
                with col2:
                    st.metric("Efficacité", f"{efficiency}%")
                with col3:
                    annual_gwh = capacity_mw * 24 * 365 * (efficiency/100) * 0.25  # Facteur capacité
                    st.metric("Production Annuelle", f"{annual_gwh:.0f} GWh")
                
                st.rerun()
    
    with tab2:
        st.subheader("📊 Production Solaire")
        
        if st.session_state.energy_lab['power_plants']:
            solar_plants = {k: v for k, v in st.session_state.energy_lab['power_plants'].items() 
                          if v['type'] == 'Solar'}
            
            if solar_plants:
                selected_plant = st.selectbox("Sélectionner Installation",
                    list(solar_plants.keys()),
                    format_func=lambda x: solar_plants[x]['name'])
                
                plant = solar_plants[selected_plant]
                
                simulation_hours = st.slider("Simuler Production (heures)", 24, 720, 168)
                
                if st.button("📊 Simuler Production", type="primary"):
                    with st.spinner(f"Simulation {simulation_hours}h..."):
                        import time
                        time.sleep(1)
                        
                        # Simuler production
                        production = simulate_solar_production(
                            plant['capacity_mw'],
                            simulation_hours,
                            plant['location']
                        )
                        
                        plant['production_history'] = production
                    
                    st.success("✅ Simulation terminée!")
                    
                    # Statistiques
                    total_mwh = sum(production)
                    avg_mw = np.mean(production)
                    peak_mw = max(production)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Production Totale", f"{total_mwh:.0f} MWh")
                    with col2:
                        st.metric("Moyenne", f"{avg_mw:.1f} MW")
                    with col3:
                        st.metric("Pic", f"{peak_mw:.1f} MW")
                    
                    # Graphique
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        y=production[:min(168, len(production))],  # 7 jours max
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#FFD700', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Production Solaire (7 premiers jours)",
                        xaxis_title="Heure",
                        yaxis_title="Puissance (MW)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune installation solaire créée")
        else:
            st.info("Créez d'abord une installation")
    
    with tab3:
        st.subheader("🤖 IA Prédiction Production")
        
        st.write("### 🌤️ Prédiction Météo → Production")
        
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_days = st.slider("Horizon Prédiction (jours)", 1, 7, 3)
            
            # Simulation données météo
            weather_conditions = st.selectbox("Conditions Prévues",
                ["Ensoleillé", "Partiellement Nuageux", "Nuageux", "Pluie"])
        
        with col2:
            st.write("**Modèle IA:**")
            st.write("• LSTM Neural Network")
            st.write("• Training: 5 ans données")
            st.write("• Précision: 94.3%")
            st.write("• Update: Temps réel")
        
        if st.button("🤖 Prédire Production", type="primary"):
            with st.spinner("Prédiction IA..."):
                import time
                time.sleep(2)
                
                # Facteur météo
                weather_factors = {
                    'Ensoleillé': 1.0,
                    'Partiellement Nuageux': 0.7,
                    'Nuageux': 0.4,
                    'Pluie': 0.2
                }
                
                factor = weather_factors[weather_conditions]
                
                # Générer prédictions
                predictions = []
                confidence = []
                
                for day in range(forecast_days):
                    base_production = 80 * factor  # MW
                    noise = np.random.uniform(-5, 5)
                    predictions.append(base_production + noise)
                    confidence.append(95 - day * 3)  # Confiance diminue avec horizon
                
                st.success("✅ Prédiction terminée!")
                
                # Afficher résultats
                for i, (pred, conf) in enumerate(zip(predictions, confidence)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Jour {i+1}**")
                    with col2:
                        st.metric("Production", f"{pred:.1f} MW")
                    with col3:
                        st.metric("Confiance", f"{conf:.0f}%")
    
    with tab4:
        st.subheader("⚙️ Optimisation Performance")
        
        st.write("### 🎯 Paramètres Optimisables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Angle Panneaux:**")
            current_angle = st.slider("Inclinaison (°)", 0, 90, 35, 1)
            optimal_angle = 30 + np.random.uniform(-5, 5)
            
            if abs(current_angle - optimal_angle) < 5:
                st.success(f"✅ Optimal (~{optimal_angle:.0f}°)")
            else:
                st.warning(f"⚠️ Suggéré: {optimal_angle:.0f}°")
            
            st.write("**Nettoyage:**")
            cleaning_frequency = st.selectbox("Fréquence",
                ["Quotidien", "Hebdomadaire", "Mensuel"])
            
            loss_dust = {'Quotidien': 1, 'Hebdomadaire': 3, 'Mensuel': 7}[cleaning_frequency]
            st.write(f"Perte poussière: ~{loss_dust}%")
        
        with col2:
            st.write("**Refroidissement:**")
            cooling = st.checkbox("Système Refroidissement Actif", value=False)
            
            if cooling:
                st.info("Gain efficacité: +2-3%")
            
            st.write("**Optimisation IA:**")
            
            if st.button("⚛️ Optimiser avec Quantique"):
                with st.spinner("Optimisation quantique..."):
                    import time
                    time.sleep(2)
                    
                    gain = np.random.uniform(8, 15)
                    st.success(f"✅ Gain performance: +{gain:.1f}%")

# ==================== PAGE: ÉOLIEN OPTIMISÉ ====================
elif page == "💨 Éolien Optimisé":
    st.header("💨 Énergie Éolienne Optimisée")
    
    st.info("""
    **Éoliennes Intelligentes**
    
    - Turbines offshore/onshore
    - Contrôle pitch par IA
    - Prédiction vent machine learning
    - Maintenance prédictive
    """)
    
    tab1, tab2, tab3 = st.tabs(["💨 Parcs Éoliens", "📊 Production", "🤖 IA Contrôle"])
    
    with tab1:
        st.subheader("💨 Créer Parc Éolien")
        
        with st.form("create_wind_farm"):
            col1, col2 = st.columns(2)
            
            with col1:
                wind_name = st.text_input("Nom Parc", "Wind Farm Offshore")
                
                n_turbines = st.number_input("Nombre Turbines", 1, 500, 50, 1)
                
                turbine_capacity_mw = st.selectbox("Capacité/Turbine",
                    [2, 3, 5, 8, 10, 12, 15])
                
                location_type = st.selectbox("Type",
                    ["Offshore", "Onshore", "Montagne"])
            
            with col2:
                hub_height_m = st.slider("Hauteur Mât (m)", 80, 200, 120, 10)
                
                rotor_diameter_m = st.slider("Diamètre Rotor (m)", 80, 240, 150, 10)
                
                avg_wind_speed = st.slider("Vent Moyen (m/s)", 5, 15, 9, 1)
                
                ai_control = st.checkbox("🤖 Contrôle IA", value=True)
            
            if st.form_submit_button("💨 Créer Parc", type="primary"):
                farm_id = f"wind_{len(st.session_state.energy_lab['power_plants']) + 1}"
                
                total_capacity = n_turbines * turbine_capacity_mw
                
                # Calculer facteur capacité
                if location_type == "Offshore":
                    capacity_factor = 0.45
                elif location_type == "Montagne":
                    capacity_factor = 0.35
                else:
                    capacity_factor = 0.30
                
                wind_farm = {
                    'id': farm_id,
                    'name': wind_name,
                    'type': 'Wind',
                    'location_type': location_type,
                    'n_turbines': n_turbines,
                    'turbine_capacity_mw': turbine_capacity_mw,
                    'total_capacity_mw': total_capacity,
                    'hub_height_m': hub_height_m,
                    'rotor_diameter_m': rotor_diameter_m,
                    'avg_wind_speed': avg_wind_speed,
                    'capacity_factor': capacity_factor,
                    'ai_control': ai_control,
                    'status': 'operational',
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.energy_lab['power_plants'][farm_id] = wind_farm
                log_event(f"Parc éolien créé: {wind_name}", "SUCCESS")
                
                st.success(f"✅ Parc '{wind_name}' créé!")
                st.balloons()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Turbines", n_turbines)
                with col2:
                    st.metric("Capacité", f"{total_capacity} MW")
                with col3:
                    annual_gwh = total_capacity * 24 * 365 * capacity_factor / 1000
                    st.metric("Production/an", f"{annual_gwh:.0f} GWh")
                with col4:
                    st.metric("Facteur", f"{capacity_factor*100:.0f}%")
                
                st.rerun()
    
    with tab2:
        st.subheader("📊 Production Éolienne")
        
        wind_farms = {k: v for k, v in st.session_state.energy_lab['power_plants'].items() 
                      if v.get('type') == 'Wind'}
        
        if wind_farms:
            selected_farm = st.selectbox("Sélectionner Parc",
                list(wind_farms.keys()),
                format_func=lambda x: wind_farms[x]['name'])
            
            farm = wind_farms[selected_farm]
            
            st.write("### 📊 Données Temps Réel")
            
            # Simulation production
            current_wind = farm['avg_wind_speed'] + np.random.uniform(-2, 2)
            
            # Courbe puissance (simplifiée)
            if current_wind < 3:
                power_pct = 0
            elif current_wind < 12:
                power_pct = ((current_wind - 3) / 9) ** 3
            elif current_wind < 25:
                power_pct = 1.0
            else:
                power_pct = 0  # Arrêt sécurité
            
            current_power = farm['total_capacity_mw'] * power_pct
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vent Actuel", f"{current_wind:.1f} m/s")
            with col2:
                st.metric("Production", f"{current_power:.1f} MW")
            with col3:
                st.metric("Taux Charge", f"{power_pct*100:.0f}%")
            
            # Graphique courbe puissance
            st.write("### 📈 Courbe de Puissance")
            
            wind_speeds = np.linspace(0, 30, 100)
            power_curve = []
            
            for v in wind_speeds:
                if v < 3:
                    p = 0
                elif v < 12:
                    p = ((v - 3) / 9) ** 3
                elif v < 25:
                    p = 1.0
                else:
                    p = 0
                power_curve.append(p * farm['turbine_capacity_mw'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=wind_speeds,
                y=power_curve,
                mode='lines',
                line=dict(color='#00CED1', width=3),
                fill='tozeroy'
            ))
            
            fig.add_vline(x=current_wind, line_dash="dash", line_color="red",
                         annotation_text="Vent actuel")
            
            fig.update_layout(
                title="Courbe de Puissance Turbine",
                xaxis_title="Vitesse Vent (m/s)",
                yaxis_title="Puissance (MW)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Créez d'abord un parc éolien")
    
    with tab3:
        st.subheader("🤖 Contrôle IA Turbines")
        
        st.write("### 🎯 Optimisations Temps Réel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Angle Pitch:**")
            st.write("Ajustement pales pour maximiser capture")
            
            pitch_angle = st.slider("Angle (°)", 0, 90, 15, 1)
            
            st.write("**Yaw Control:**")
            st.write("Orientation nacelle vers vent")
            
            yaw_angle = st.slider("Orientation (°)", 0, 360, 180, 1)
        
        with col2:
            st.write("**Wake Effect:**")
            st.write("Gestion sillages entre turbines")
            
            wake_optimization = st.checkbox("Optimisation Sillages IA", value=True)
            
            if wake_optimization:
                st.success("Gain: +5-10% production parc")
            
            st.write("**Maintenance Prédictive:**")
            
            health_score = np.random.uniform(85, 98)
            st.metric("Score Santé", f"{health_score:.1f}%")
            
            if health_score < 90:
                st.warning("⚠️ Maintenance recommandée")

# ==================== PAGE: HYDROGÈNE ====================
elif page == "💧 Hydrogène":
    st.header("💧 Économie Hydrogène")
    
    st.info("""
    **Hydrogène Vert - Vecteur Énergétique**
    
    - Production par électrolyse (renouvelables)
    - Stockage longue durée
    - Piles à combustible
    - Applications transport & industrie
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Électrolyse", "🔋 Piles Combustible", "💾 Stockage", "📊 Économie"])
    
    with tab1:
        st.subheader("⚡ Production Hydrogène par Électrolyse")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### 💧 Calculateur Production")
            
            power_input_mw = st.slider("Puissance Électrique (MW)", 1, 1000, 100, 10)
            
            electrolyzer_type = st.selectbox("Type Électrolyseur",
                ["Alcalin", "PEM (Membrane)", "SOEC (Haute Température)", "AEM"])
            
            efficiencies = {
                'Alcalin': 0.65,
                'PEM (Membrane)': 0.70,
                'SOEC (Haute Température)': 0.85,
                'AEM': 0.75
            }
            
            efficiency = efficiencies[electrolyzer_type]
            
            # Calculer production H2
            # 1 kg H2 ≈ 33.3 kWh (PCI) / 39.4 kWh (PCS)
            # Avec efficacité: kWh élec → kg H2
            
            hours_operation = st.slider("Heures Fonctionnement", 1, 8760, 4000)
            
            if st.button("💧 Calculer Production", type="primary"):
                energy_input_mwh = power_input_mw * hours_operation
                energy_input_kwh = energy_input_mwh * 1000
                
                # Production H2 (kg)
                h2_production_kg = (energy_input_kwh * efficiency) / 50  # ~50 kWh/kg H2
                h2_production_tonnes = h2_production_kg / 1000
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Production H₂", f"{h2_production_tonnes:.1f} tonnes")
                with col2:
                    st.metric("Énergie Stockée", f"{h2_production_kg * 33.3 / 1000:.1f} MWh")
                with col3:
                    st.metric("Efficacité", f"{efficiency*100:.0f}%")
                
                # Coût
                electricity_cost = 50  # $/MWh
                total_cost = energy_input_mwh * electricity_cost
                cost_per_kg = total_cost / h2_production_kg
                
                st.write(f"**Coût Production:** ${total_cost:,.0f}")
                st.write(f"**Coût/kg H₂:** ${cost_per_kg:.2f}")
        
        with col2:
            st.write("### 🔬 Technologies")
            
            st.write("**Alcalin:**")
            st.write("• Mature")
            st.write("• Coût bas")
            st.write("• 65% efficient")
            
            st.write("\n**PEM:**")
            st.write("• Flexible")
            st.write("• Démarrage rapide")
            st.write("• 70% efficient")
            
            st.write("\n**SOEC:**")
            st.write("• Haute température")
            st.write("• Très efficient")
            st.write("• 85% efficient")
    
    with tab2:
        st.subheader("🔋 Piles à Combustible")
        
        st.write("### ⚡ Génération Électricité depuis H₂")
        
        col1, col2 = st.columns(2)
        
        with col1:
            h2_input_kg = st.number_input("Hydrogène Disponible (kg)", 1.0, 10000.0, 100.0, 1.0)
            
            fuel_cell_type = st.selectbox("Type Pile",
                ["PEMFC (Basse T)", "SOFC (Haute T)", "MCFC", "PAFC"])
            
            fc_efficiencies = {
                'PEMFC (Basse T)': 0.60,
                'SOFC (Haute T)': 0.65,
                'MCFC': 0.55,
                'PAFC': 0.45
            }
            
            fc_efficiency = fc_efficiencies[fuel_cell_type]
        
        with col2:
            # Calculer électricité
            energy_available_kwh = h2_input_kg * 33.3  # PCI H2
            electricity_kwh = energy_available_kwh * fc_efficiency
            
            st.metric("Énergie H₂", f"{energy_available_kwh:.0f} kWh")
            st.metric("Électricité Produite", f"{electricity_kwh:.0f} kWh")
            st.metric("Efficacité Pile", f"{fc_efficiency*100:.0f}%")
            
            # Efficacité round-trip
            roundtrip = efficiency * fc_efficiency
            st.metric("Efficacité Round-Trip", f"{roundtrip*100:.0f}%")
    
    with tab3:
        st.subheader("💾 Stockage Hydrogène")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🗜️ Méthodes Stockage")
            
            storage_method = st.selectbox("Méthode",
                ["Comprimé 700 bar", "Liquide (-253°C)", "Hydrures Métalliques", 
                 "LOHC (Liquide Organique)", "Ammoniaque"])
            
            storage_densities = {
                'Comprimé 700 bar': 42,  # kg/m³
                'Liquide (-253°C)': 71,
                'Hydrures Métalliques': 100,
                'LOHC (Liquide Organique)': 50,
                'Ammoniaque': 108
            }
            
            density = storage_densities[storage_method]
            
            st.metric("Densité", f"{density} kg/m³")
            
            volume_needed_m3 = h2_input_kg / density
            st.metric("Volume Nécessaire", f"{volume_needed_m3:.1f} m³")
        
        with col2:
            st.write("### 📊 Comparaison")
            
            comparison = pd.DataFrame({
                'Méthode': list(storage_densities.keys()),
                'Densité (kg/m³)': list(storage_densities.values()),
                'Énergie (MWh/m³)': [d * 33.3 / 1000 for d in storage_densities.values()]
            })
            
            st.dataframe(comparison, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Économie Hydrogène")
        
        st.write("### 💰 Analyse Coûts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            production_cost = cost_per_kg if 'cost_per_kg' in locals() else 4.0
            
            st.metric("Coût Production", f"${production_cost:.2f}/kg")
            
            storage_cost = 1.5
            transport_cost = 0.5
            total_cost = production_cost + storage_cost + transport_cost
            
            st.metric("Coût Total", f"${total_cost:.2f}/kg")
            
            # Comparaison énergies
            h2_cost_per_mwh = (total_cost / 33.3) * 1000
            st.metric("Équivalent", f"${h2_cost_per_mwh:.0f}/MWh")
        
        with col2:
            st.write("**Objectifs 2030:**")
            st.write("• Production: $2/kg")
            st.write("• Électrolyseurs: <$500/kW")
            st.write("• Piles: <$50/kW")
            
            st.write("\n**Applications:**")
            st.write("✅ Transport lourd")
            st.write("✅ Industrie (acier, chimie)")
            st.write("✅ Stockage saisonnier")
            st.write("✅ Aviation/Maritime")

# ==================== PAGE: IA OPTIMISATION ====================
elif page == "🤖 IA Optimisation":
    st.header("🤖 Intelligence Artificielle - Optimisation Énergétique")
    
    st.info("""
    **IA pour l'Énergie**
    
    - Prédiction demande/production
    - Optimisation réseaux
    - Maintenance prédictive
    - Trading énergétique automatisé
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Modèles IA", "📈 Prédictions", "⚡ Optimisation", "🔧 Maintenance"])
    
    with tab1:
        st.subheader("🧠 Modèles IA Disponibles")
        
        models = {
            "LSTM Demand Forecasting": {
                "Type": "Réseau Neuronal Récurrent",
                "Usage": "Prédiction demande 24-72h",
                "Précision": "96.3%",
                "Training": "5 ans données",
                "Update": "Quotidien"
            },

            "CNN Production Forecast": {
                "Type": "Réseau Convolutif",
                "Usage": "Prédiction production solaire/éolien",
                "Précision": "94.8%",
                "Training": "Images satellite + météo",
                "Update": "Temps réel"
            },
            "Reinforcement Learning Grid": {
                "Type": "Deep Q-Learning",
                "Usage": "Optimisation smart grid",
                "Précision": "98.1%",
                "Training": "Simulation 1M scénarios",
                "Update": "Continu"
            },
            "Transformer Energy Trading": {
                "Type": "Attention Mechanism",
                "Usage": "Trading énergétique",
                "Précision": "92.5%",
                "Training": "10 ans marchés",
                "Update": "Temps réel"
            },
            "GAN Anomaly Detection": {
                "Type": "Generative Adversarial",
                "Usage": "Détection anomalies équipements",
                "Précision": "99.2%",
                "Training": "Données capteurs",
                "Update": "Streaming"
            }
        }
        

        for model_name, specs in models.items():
            with st.expander(f"🤖 {model_name}"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {specs['Type']}")
                    st.write(f"**Usage:** {specs['Usage']}")
                    st.write(f"**Précision:** {specs['Précision']}")
                
                with col2:
                    st.write(f"**Training:** {specs['Training']}")
                    st.write(f"**Update:** {specs['Update']}")
                
                if st.button(f"🚀 Déployer {model_name}", key=f"deploy_{model_name}"):
                    st.success(f"✅ Modèle {model_name} déployé!")
                    log_event(f"Modèle IA déployé: {model_name}", "SUCCESS")
    
    with tab2:
        st.subheader("📈 Prédictions IA")
        
        st.write("### 🔮 Prédiction Demande Énergétique")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            horizon = st.selectbox("Horizon Prédiction",
                ["24 heures", "48 heures", "7 jours", "30 jours"])
            
            features = st.multiselect("Variables Prédictives",
                ["Température", "Météo", "Jour Semaine", "Vacances", "Événements", "Historique"],
                default=["Température", "Historique"])
            
            if st.button("🔮 Prédire Demande", type="primary"):
                with st.spinner("Prédiction en cours..."):
                    import time
                    time.sleep(2)
                    
                    # Générer prédictions
                    hours_map = {"24 heures": 24, "48 heures": 48, "7 jours": 168, "30 jours": 720}
                    n_hours = hours_map[horizon]
                    
                    # Simulation prédiction
                    actual = []
                    predicted = []
                    confidence_low = []
                    confidence_high = []
                    
                    for h in range(min(n_hours, 168)):  # Max 7 jours affichage
                        base = 800 + 200 * np.sin((h % 24 - 12) * np.pi / 12)
                        noise = np.random.normal(0, 20)
                        
                        act = base + noise
                        pred = base + np.random.normal(0, 10)
                        
                        actual.append(act)
                        predicted.append(pred)
                        confidence_low.append(pred - 30)
                        confidence_high.append(pred + 30)
                    
                    st.success("✅ Prédiction terminée!")
                    
                    # Métriques
                    mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
                    rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted))**2))
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("MAE", f"{mae:.1f} MW")
                    with col2:
                        st.metric("RMSE", f"{rmse:.1f} MW")
                    with col3:
                        st.metric("R²", "0.963")
                    
                    # Graphique
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        y=actual,
                        mode='lines+markers',
                        name='Réel',
                        line=dict(color='#FF4500', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        y=predicted,
                        mode='lines+markers',
                        name='Prédit',
                        line=dict(color='#00FF00', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        y=confidence_high,
                        mode='lines',
                        name='Intervalle Confiance',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        y=confidence_low,
                        mode='lines',
                        fill='tonexty',
                        name='IC 95%',
                        line=dict(width=0),
                        fillcolor='rgba(0, 255, 0, 0.2)'
                    ))
                    
                    fig.update_layout(
                        title="Prédiction vs Réel",
                        xaxis_title="Heure",
                        yaxis_title="Demande (MW)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### 📊 Performance")
            
            st.metric("Précision", "96.3%")
            st.metric("Latence", "< 100ms")
            st.metric("Fiabilité", "99.9%")
            
            st.write("\n### 🎯 Amélioration")
            st.write(f"vs Baseline: +23%")
            st.write(f"vs Ancien: +8%")
    
    with tab3:
        st.subheader("⚡ Optimisation Réseau IA")
        
        st.write("### 🎯 Optimisation Multi-Objectifs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Objectifs:**")
            
            obj_cost = st.checkbox("💰 Minimiser Coûts", value=True)
            obj_emissions = st.checkbox("🌍 Minimiser Émissions", value=True)
            obj_reliability = st.checkbox("⚡ Maximiser Fiabilité", value=True)
            obj_renewable = st.checkbox("♻️ Maximiser Renouvelables", value=False)
            
            optimization_method = st.selectbox("Méthode",
                ["Reinforcement Learning", "Genetic Algorithm", "Particle Swarm", "Gradient Descent"])
        
        with col2:
            st.write("**Contraintes:**")
            
            max_load = st.slider("Charge Max (%)", 50, 100, 85)
            min_reserve = st.slider("Réserve Min (%)", 5, 30, 15)
            max_renewable_var = st.slider("Variabilité Renouv. Max (%)", 10, 50, 30)
        
        if st.button("⚡ Optimiser Réseau", type="primary", use_container_width=True):
            with st.spinner("Optimisation IA en cours..."):
                import time
                
                progress = st.progress(0)
                status = st.empty()
                
                for i in range(100):
                    time.sleep(0.03)
                    progress.progress(i + 1)
                    
                    if i < 20:
                        status.text("Analyse état réseau...")
                    elif i < 40:
                        status.text("Calcul solutions optimales...")
                    elif i < 70:
                        status.text("Évaluation contraintes...")
                    else:
                        status.text("Finalisation...")
                
                progress.empty()
                status.empty()
                
                st.success("✅ Optimisation terminée!")
                
                # Résultats
                st.write("### 📊 Résultats Optimisation")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cost_reduction = np.random.uniform(15, 25)
                    st.metric("Réduction Coûts", f"{cost_reduction:.1f}%", f"-${cost_reduction*100:.0f}k")
                
                with col2:
                    emission_reduction = np.random.uniform(20, 35)
                    st.metric("Réduction CO₂", f"{emission_reduction:.1f}%", f"-{emission_reduction*10:.0f} tonnes")
                
                with col3:
                    efficiency_gain = np.random.uniform(8, 15)
                    st.metric("Gain Efficacité", f"{efficiency_gain:.1f}%", f"+{efficiency_gain:.1f}%")
                
                with col4:
                    renewable_increase = np.random.uniform(10, 20)
                    st.metric("↑ Renouvelables", f"{renewable_increase:.1f}%", f"+{renewable_increase:.1f}%")
                
                # Actions recommandées
                st.write("### 🎯 Actions Recommandées")
                
                actions = [
                    "🔄 Redistribuer charge vers centrales efficientes",
                    "🔋 Augmenter stockage batteries 15%",
                    "☀️ Prioriser production solaire heures pic",
                    "💨 Activer éoliennes offshore supplémentaires",
                    "⚡ Réduire pertes transmission nœuds critiques"
                ]
                
                for action in actions:
                    st.write(f"• {action}")
    
    with tab4:
        st.subheader("🔧 Maintenance Prédictive")
        
        st.write("### 🔍 Détection Anomalies & Prédiction Pannes")
        
        # Simulation équipements
        n_equipment = 50
        equipment_data = []
        
        for i in range(n_equipment):
            health = np.random.uniform(60, 100)
            risk = "Faible" if health > 85 else "Moyen" if health > 70 else "Élevé"
            
            equipment_data.append({
                'ID': f"EQ-{i+1:03d}",
                'Type': np.random.choice(['Turbine', 'Transformateur', 'Disjoncteur', 'Générateur']),
                'Santé': health,
                'Risque': risk,
                'Maintenance': np.random.randint(30, 365)
            })
        
        df_equipment = pd.DataFrame(equipment_data)
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.multiselect("Niveau Risque",
                ["Faible", "Moyen", "Élevé"],
                default=["Élevé"])
        
        with col2:
            type_filter = st.multiselect("Type Équipement",
                df_equipment['Type'].unique(),
                default=df_equipment['Type'].unique())
        
        with col3:
            health_threshold = st.slider("Santé < ", 0, 100, 80)
        
        # Filtrer données
        df_filtered = df_equipment[
            (df_equipment['Risque'].isin(risk_filter)) &
            (df_equipment['Type'].isin(type_filter)) &
            (df_equipment['Santé'] < health_threshold)
        ]
        
        st.write(f"### 📊 {len(df_filtered)} Équipements Nécessitant Attention")
        
        # Afficher équipements critiques
        for _, eq in df_filtered.iterrows():
            risk_color = "#FF0000" if eq['Risque'] == "Élevé" else "#FFA500" if eq['Risque'] == "Moyen" else "#00FF00"
            
            with st.expander(f"⚠️ {eq['ID']} - {eq['Type']} (Risque: {eq['Risque']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score Santé", f"{eq['Santé']:.0f}%")
                    st.progress(eq['Santé'] / 100)
                
                with col2:
                    st.metric("Maintenance dans", f"{eq['Maintenance']} jours")
                    st.write(f"**Risque:** {eq['Risque']}")
                
                with col3:
                    if eq['Risque'] == "Élevé":
                        st.error("🚨 Action Urgente")
                        if st.button(f"📅 Planifier", key=f"plan_{eq['ID']}"):
                            st.success("Maintenance planifiée!")
                    elif eq['Risque'] == "Moyen":
                        st.warning("⚠️ Surveiller")
                    else:
                        st.success("✅ Normal")

# ==================== PAGE: MATÉRIAUX AVANCÉS ====================
elif page == "🔬 Matériaux Avancés":
    st.header("🔬 Recherche Matériaux Énergétiques")
    
    st.info("""
    **Nouveaux Matériaux pour l'Énergie**
    
    - Superconducteurs haute température
    - Pérovskites photovoltaïques
    - Matériaux stockage hydrogène
    - Catalyseurs avancés
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Bibliothèque", "➕ Nouveau Matériau", "🧪 Tests", "📊 Performances"])
    
    with tab1:
        st.subheader("🔬 Bibliothèque Matériaux")
        
        material_library = {
            "YBa₂Cu₃O₇ (YBCO)": {
                "type": "Superconducteur",
                "tc_k": 92,
                "application": "Câbles haute puissance",
                "trl": 7
            },
            "CH₃NH₃PbI₃": {
                "type": "Pérovskite PV",
                "efficiency": 0.25,
                "application": "Cellules solaires",
                "trl": 6
            },
            "MOF-5": {
                "type": "Metal-Organic Framework",
                "h2_capacity": 7.1,  # wt%
                "application": "Stockage H₂",
                "trl": 4
            },
            "Pt-Ru Nanoparticules": {
                "type": "Catalyseur",
                "efficiency": 0.85,
                "application": "Piles combustible",
                "trl": 8
            },
            "Graphène": {
                "type": "2D Matériau",
                "conductivity": 10000,  # S/m
                "application": "Supercondensateurs",
                "trl": 5
            },
            "LiFePO₄": {
                "type": "Cathode Batterie",
                "energy_density": 170,  # Wh/kg
                "application": "Batteries Li-ion",
                "trl": 9
            }
        }
        
        for material, specs in material_library.items():
            with st.expander(f"🔬 {material}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {specs['type']}")
                    st.write(f"**Application:** {specs['application']}")
                    st.write(f"**TRL:** {specs['trl']}/9")
                    
                    # Barre TRL
                    st.progress(specs['trl'] / 9)
                
                with col2:
                    # Propriétés spécifiques
                    for key, value in specs.items():
                        if key not in ['type', 'application', 'trl']:
                            unit = ""
                            if 'tc_k' in key:
                                unit = " K"
                            elif 'efficiency' in key:
                                value = value * 100
                                unit = "%"
                            elif 'capacity' in key:
                                unit = " wt%"
                            elif 'conductivity' in key:
                                unit = " S/m"
                            elif 'energy_density' in key:
                                unit = " Wh/kg"
                            
                            st.metric(key.replace('_', ' ').title(), f"{value}{unit}")
                
                if st.button(f"📊 Analyser {material}", key=f"analyze_{material}"):
                    st.info(f"Analyse détaillée de {material}")
    
    with tab2:
        st.subheader("➕ Découvrir Nouveau Matériau")
        
        st.write("### 🤖 Génération IA + Simulation Quantique")
        
        with st.form("discover_material"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_application = st.selectbox("Application Cible",
                    ["Superconducteur", "Photovoltaïque", "Stockage H₂", 
                     "Catalyseur", "Batterie", "Thermoélectrique"])
                
                target_property = st.text_input("Propriété Cible", 
                    "Haute efficacité, faible coût")
                
                base_elements = st.multiselect("Éléments Base",
                    ["H", "Li", "C", "N", "O", "Na", "Mg", "Al", "Si", "S", 
                     "K", "Ca", "Ti", "Fe", "Ni", "Cu", "Zn", "Pt", "Pb"],
                    default=["Li", "O"])
            
            with col2:
                max_cost = st.slider("Coût Max ($/kg)", 1, 1000, 100, 10)
                
                toxicity_limit = st.selectbox("Toxicité Max",
                    ["Nulle", "Très Faible", "Faible", "Modérée"])
                
                use_quantum = st.checkbox("🔬 Simulation Quantique", value=True)
                use_ai = st.checkbox("🤖 Génération IA", value=True)
            
            if st.form_submit_button("🔬 Découvrir Matériau", type="primary"):
                with st.spinner("Recherche nouveau matériau..."):
                    import time
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.text("Génération candidats IA...")
                    time.sleep(1)
                    progress.progress(0.25)
                    
                    status.text("Simulation structures quantique...")
                    time.sleep(1.5)
                    progress.progress(0.50)
                    
                    status.text("Calcul propriétés DFT...")
                    time.sleep(1.5)
                    progress.progress(0.75)
                    
                    status.text("Optimisation composition...")
                    time.sleep(1)
                    progress.progress(1.0)
                    
                    time.sleep(0.5)
                    progress.empty()
                    status.empty()
                    
                    # Générer matériau fictif
                    elements_str = "".join(base_elements[:3])
                    formula = f"{elements_str}{np.random.randint(2, 6)}O{np.random.randint(2, 8)}"
                    
                    st.success(f"✅ Nouveau matériau découvert: **{formula}**")
                    st.balloons()
                    
                    # Propriétés prédites
                    st.write("### 🎯 Propriétés Prédites")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        efficiency = np.random.uniform(0.75, 0.95)
                        st.metric("Efficacité", f"{efficiency*100:.1f}%")
                    
                    with col2:
                        stability = np.random.uniform(0.80, 0.99)
                        st.metric("Stabilité", f"{stability*100:.1f}%")
                    
                    with col3:
                        cost_predicted = np.random.uniform(10, max_cost)
                        st.metric("Coût Estimé", f"${cost_predicted:.0f}/kg")
                    
                    # Sauvegarder
                    material_id = f"mat_{len(st.session_state.energy_lab['materials']) + 1}"
                    
                    new_material = {
                        'id': material_id,
                        'formula': formula,
                        'application': target_application,
                        'efficiency': efficiency,
                        'stability': stability,
                        'cost': cost_predicted,
                        'toxicity': toxicity_limit,
                        'elements': base_elements,
                        'discovered_at': datetime.now().isoformat(),
                        'trl': 1
                    }
                    
                    st.session_state.energy_lab['materials'][material_id] = new_material
                    log_event(f"Nouveau matériau: {formula}", "SUCCESS")
                    
                    st.info("Matériau ajouté à la bibliothèque (TRL 1)")
    
    with tab3:
        st.subheader("🧪 Tests Expérimentaux")
        
        if st.session_state.energy_lab['materials']:
            selected_material = st.selectbox("Sélectionner Matériau",
                list(st.session_state.energy_lab['materials'].keys()),
                format_func=lambda x: st.session_state.energy_lab['materials'][x]['formula'])
            
            material = st.session_state.energy_lab['materials'][selected_material]
            
            st.write(f"### 🔬 Tests pour {material['formula']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                test_type = st.selectbox("Type Test",
                    ["Caractérisation XRD", "Spectroscopie", "Test Performance", 
                     "Stabilité Thermique", "Cyclage"])
                
                test_conditions = st.text_area("Conditions Test",
                    "Température: 25°C\nPression: 1 atm\nDurée: 24h")
            
            with col2:
                st.write("**État Actuel:**")
                st.write(f"TRL: {material['trl']}/9")
                st.write(f"Efficacité: {material['efficiency']*100:.1f}%")
                st.write(f"Stabilité: {material['stability']*100:.1f}%")
            
            if st.button("🧪 Lancer Test", type="primary"):
                with st.spinner(f"Test {test_type} en cours..."):
                    import time
                    time.sleep(3)
                    
                    # Résultats test
                    success = np.random.choice([True, False], p=[0.8, 0.2])
                    
                    if success:
                        st.success("✅ Test réussi!")
                        
                        # Améliorer TRL
                        if material['trl'] < 9:
                            material['trl'] += 1
                            st.info(f"TRL augmenté: {material['trl']}/9")
                        
                        # Affiner propriétés
                        improvement = np.random.uniform(0.95, 1.05)
                        material['efficiency'] *= improvement
                        material['stability'] *= improvement
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Nouvelle Efficacité", 
                                    f"{material['efficiency']*100:.1f}%",
                                    f"+{(improvement-1)*100:.1f}%")
                        
                        with col2:
                            st.metric("Nouvelle Stabilité",
                                    f"{material['stability']*100:.1f}%",
                                    f"+{(improvement-1)*100:.1f}%")
                        
                        log_event(f"Test réussi: {material['formula']}", "SUCCESS")
                    else:
                        st.error("❌ Test échoué - Optimisation nécessaire")
                        log_event(f"Test échoué: {material['formula']}", "WARNING")
        else:
            st.info("Découvrez d'abord un nouveau matériau")
    
    with tab4:
        st.subheader("📊 Comparaison Performances")
        
        if st.session_state.energy_lab['materials']:
            # Créer dataframe
            materials_data = []
            
            for mat_id, mat in st.session_state.energy_lab['materials'].items():
                materials_data.append({
                    'Formule': mat['formula'],
                    'Application': mat['application'],
                    'Efficacité (%)': mat['efficiency'] * 100,
                    'Stabilité (%)': mat['stability'] * 100,
                    'Coût ($/kg)': mat['cost'],
                    'TRL': mat['trl']
                })
            
            df_materials = pd.DataFrame(materials_data)
            
            st.dataframe(df_materials, use_container_width=True)
            
            # Graphique radar
            if len(materials_data) > 0:
                categories = ['Efficacité', 'Stabilité', 'TRL', 'Coût (inv)']
                
                fig = go.Figure()
                
                for mat_data in materials_data[:5]:  # Max 5 matériaux
                    values = [
                        mat_data['Efficacité (%)'],
                        mat_data['Stabilité (%)'],
                        mat_data['TRL'] * 11.11,  # Normaliser sur 100
                        100 - (mat_data['Coût ($/kg)'] / 10)  # Inverser coût
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        name=mat_data['Formule']
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Comparaison Multi-Critères",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun matériau dans la bibliothèque")

# ==================== PAGES RESTANTES (Production, Consommation, etc.) ====================

elif page == "📊 Production":
    st.header("📊 Analyse Production Énergétique")
    
    # Production par source
    st.subheader("⚡ Production par Source (Temps Réel)")
    
    # Données simulées
    sources = ['Fusion', 'Fission', 'Solaire', 'Éolien', 'Hydrogène', 'Géothermie']
    production_current = [
        np.random.uniform(800, 1500),  # Fusion
        np.random.uniform(1800, 2400),  # Fission
        np.random.uniform(500, 2200),  # Solaire (dépend heure)
        np.random.uniform(1000, 2000),  # Éolien
        np.random.uniform(600, 1200),  # Hydrogène
        np.random.uniform(400, 800)  # Géothermie
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=[go.Bar(
            x=sources,
            y=production_current,
            marker_color=['#FFD700', '#FF8C00', '#FFA500', '#00CED1', '#32CD32', '#8B4513']
        )])
        
        fig.update_layout(
            title="Production Actuelle par Source",
            xaxis_title="Source",
            yaxis_title="Puissance (MW)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("### 📈 Statistiques")
        
        total_prod = sum(production_current)
        st.metric("Production Totale", f"{total_prod:.0f} MW")
        
        renewable_pct = (sum(production_current[2:]) / total_prod) * 100
        st.metric("Part Renouvelable", f"{renewable_pct:.1f}%")
        
        st.metric("Peak Aujourd'hui", f"{total_prod * 1.2:.0f} MW")

elif page == "📈 Consommation":
    st.header("📈 Analyse Consommation")
    
    st.subheader("📊 Profil Consommation 24h")
    
    # Simulation consommation
    hours = list(range(24))
    consumption = [700 + 300*np.sin((h-6)*np.pi/12) + np.random.uniform(-30, 30) for h in hours]
    
    # Décomposition par secteur
    residential = [c * 0.35 for c in consumption]
    industrial = [c * 0.40 for c in consumption]
    commercial = [c * 0.20 for c in consumption]
    transport = [c * 0.05 for c in consumption]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=hours, y=residential, name='Résidentiel', 
                            stackgroup='one', fillcolor='#FFD700'))
    fig.add_trace(go.Scatter(x=hours, y=industrial, name='Industriel',
                            stackgroup='one', fillcolor='#FF8C00'))
    fig.add_trace(go.Scatter(x=hours, y=commercial, name='Commercial',
                            stackgroup='one', fillcolor='#FFA500'))
    fig.add_trace(go.Scatter(x=hours, y=transport, name='Transport',
                            stackgroup='one', fillcolor='#00CED1'))
    
    fig.update_layout(
        title="Consommation par Secteur",
        xaxis_title="Heure",
        yaxis_title="Puissance (MW)",
        template="plotly_dark",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "⚡ Distribution":
    st.header("⚡ Réseau Distribution")
    
    st.subheader("🗺️ État Réseau Transmission")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Lignes Actives", "1,247")
        st.metric("Transformateurs", "8,543")
    
    with col2:
        st.metric("Pertes Transmission", "4.2%")
        st.metric("Charge Moyenne", "73%")
    
    with col3:
        st.metric("Incidents/24h", "3")
        st.metric("Fiabilité", "99.97%")
    
    # Carte flux énergétiques
    st.write("### 🌐 Flux Énergétiques")
    
    # Simulation flux entre régions
    regions = ['Nord', 'Sud', 'Est', 'Ouest', 'Centre']
    
    # Matrice flux
    flow_data = np.random.randint(-200, 300, (5, 5))
    np.fill_diagonal(flow_data, 0)
    
    fig = go.Figure(data=go.Heatmap(
        z=flow_data,
        x=regions,
        y=regions,
        colorscale='RdYlGn',
        zmid=0
    ))
    
    fig.update_layout(
        title="Flux Inter-Régions (MW)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "🌍 Impact Carbone":
    st.header("🌍 Impact Environnemental")
    
    st.subheader("📊 Émissions CO₂")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        emissions_today = np.random.uniform(50, 150)
        st.metric("Aujourd'hui", f"{emissions_today:.0f} tonnes CO₂")
    
    with col2:
        emissions_month = emissions_today * 30
        st.metric("Ce Mois", f"{emissions_month/1000:.1f} kt CO₂")
    
    with col3:
        reduction_pct = np.random.uniform(15, 30)
        st.metric("Réduction vs 2020", f"-{reduction_pct:.0f}%", f"-{reduction_pct:.0f}%")
    
    with col4:
        target_2030 = 70
        st.metric("Objectif 2030", f"-{target_2030}%")
    
    # Évolution émissions
    st.write("### 📈 Évolution Émissions CO₂")
    
    years = list(range(2020, 2031))
    emissions_history = [1000 * (0.85 ** (year - 2020)) for year in years]
    target_line = [1000 * (1 - target_2030/100 * (year-2020)/10) for year in years]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=emissions_history,
        mode='lines+markers',
        name='Émissions Réelles',
        line=dict(color='#FF4500', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=years,
        y=target_line,
        mode='lines',
        name='Trajectoire Objectif',
        line=dict(color='#00FF00', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Trajectoire Décarbonation",
        xaxis_title="Année",
        yaxis_title="Émissions (kt CO₂/an)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Impact positif
    st.write("### 🌱 Actions Positives")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**CO₂ Évité:**")
        co2_avoided = np.random.uniform(500, 1000)
        st.metric("", f"{co2_avoided:.0f} tonnes/mois")
        st.success("Équivalent: 100k voitures retirées")
    
    with col2:
        st.write("**Capture Bio:**")
        co2_captured = np.random.uniform(50, 150)
        st.metric("", f"{co2_captured:.0f} tonnes/mois")
        st.info("Via micro-algues & biochar")
    
    with col3:
        st.write("**Compensation:**")
        co2_offset = np.random.uniform(20, 80)
        st.metric("", f"{co2_offset:.0f} tonnes/mois")
        st.info("Crédits carbone investis")

elif page == "💰 Économie Énergie":
    st.header("💰 Économie & Marchés Énergétiques")
    
    st.subheader("📊 Prix Énergie Temps Réel")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        price_spot = np.random.uniform(40, 80)
        st.metric("Prix Spot", f"${price_spot:.2f}/MWh", f"{np.random.uniform(-5, 5):.1f}%")
    
    with col2:
        price_day_ahead = np.random.uniform(45, 85)
        st.metric("Day-Ahead", f"${price_day_ahead:.2f}/MWh")
    
    with col3:
        volume_traded = np.random.uniform(5000, 15000)
        st.metric("Volume Échangé", f"{volume_traded:.0f} MWh")
    
    with col4:
        revenue_day = volume_traded * price_spot / 1000
        st.metric("Revenus Jour", f"${revenue_day:.0f}k")
    
    # Graphique prix 24h
    st.write("### 📈 Prix Spot 24h")
    
    hours = list(range(24))
    prices = [40 + 30*np.sin((h-6)*np.pi/12) + np.random.uniform(-5, 5) for h in hours]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=prices,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#FFD700', width=3)
    ))
    
    # Zones prix
    fig.add_hrect(y0=0, y1=40, fillcolor="green", opacity=0.1, annotation_text="Prix Bas")
    fig.add_hrect(y0=40, y1=60, fillcolor="yellow", opacity=0.1, annotation_text="Prix Normal")
    fig.add_hrect(y0=60, y1=100, fillcolor="red", opacity=0.1, annotation_text="Prix Élevé")
    
    fig.update_layout(
        title="Évolution Prix Spot",
        xaxis_title="Heure",
        yaxis_title="Prix ($/MWh)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trading automatisé
    st.write("### 🤖 Trading IA Automatisé")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Activer Trading IA", type="primary", use_container_width=True):
            with st.spinner("Analyse marché & exécution trades..."):
                import time
                time.sleep(2)
                
                # Simulation trades
                n_trades = np.random.randint(10, 30)
                profit = np.random.uniform(5000, 20000)
                
                st.success(f"✅ {n_trades} trades exécutés!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Trades", n_trades)
                with col2:
                    st.metric("Profit Net", f"${profit:.0f}")
                with col3:
                    roi = np.random.uniform(2, 8)
                    st.metric("ROI", f"{roi:.1f}%")
    
    with col2:
        st.write("**Stratégies:**")
        st.write("✅ Arbitrage prix")
        st.write("✅ Peak shaving")
        st.write("✅ Load shifting")
        st.write("✅ Reserve trading")

elif page == "🔮 Prédictions":
    st.header("🔮 Prédictions & Scénarios Futurs")
    
    st.subheader("📊 Prédictions Mix Énergétique 2030-2050")
    
    # Données prédictives
    years = [2025, 2030, 2035, 2040, 2045, 2050]
    
    predictions = {
        'Fusion': [5, 15, 30, 45, 60, 70],
        'Fission': [25, 20, 15, 10, 8, 5],
        'Solaire': [20, 30, 35, 38, 40, 42],
        'Éolien': [18, 25, 28, 30, 32, 33],
        'Hydrogène': [10, 20, 25, 28, 30, 32],
        'Autres': [22, 10, 7, 4, 3, 3]
    }
    
    fig = go.Figure()
    
    for source, values in predictions.items():
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name=source,
            stackgroup='one'
        ))
    
    fig.update_layout(
        title="Évolution Mix Énergétique (% Production)",
        xaxis_title="Année",
        yaxis_title="Part Production (%)",
        template="plotly_dark",
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scénarios
    st.write("### 🎯 Scénarios 2050")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**🚀 Optimiste**")
        st.write("• Fusion: 70%")
        st.write("• Renouvelables: 25%")
        st.write("• Émissions: -95%")
        st.write("• Coût: -60%")
        st.success("Prob: 35%")
    
    with col2:
        st.write("**📊 Modéré**")
        st.write("• Fusion: 45%")
        st.write("• Renouvelables: 40%")
        st.write("• Émissions: -80%")
        st.write("• Coût: -40%")
        st.info("Prob: 50%")
    
    with col3:
        st.write("**⚠️ Conservateur**")
        st.write("• Fusion: 20%")
        st.write("• Renouvelables: 50%")
        st.write("• Émissions: -60%")
        st.write("• Coût: -20%")
        st.warning("Prob: 15%")
    
    # Prédiction demande
    st.write("### 📈 Prédiction Demande Globale")
    
    demand_growth = [100, 115, 130, 145, 158, 170]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=demand_growth,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color='#FF4500', width=3)
    ))
    
    fig.update_layout(
        title="Croissance Demande Énergétique (Index 2025=100)",
        xaxis_title="Année",
        yaxis_title="Index Demande",
        template="plotly_dark",
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Analytics":
    st.header("📊 Analytics & KPIs Avancés")
    
    st.subheader("🎯 Tableau de Bord KPIs")
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        capacity_factor = np.random.uniform(75, 90)
        st.metric("Facteur Capacité", f"{capacity_factor:.1f}%", 
                 f"+{np.random.uniform(0, 3):.1f}%")
    
    with col2:
        availability = np.random.uniform(95, 99)
        st.metric("Disponibilité", f"{availability:.1f}%")
    
    with col3:
        efficiency = np.random.uniform(85, 92)
        st.metric("Efficacité Globale", f"{efficiency:.1f}%",
                 f"+{np.random.uniform(0, 2):.1f}%")
    
    with col4:
        lcoe = np.random.uniform(30, 60)
        st.metric("LCOE", f"${lcoe:.0f}/MWh",
                 f"-{np.random.uniform(1, 5):.1f}%")
    
    with col5:
        reliability = np.random.uniform(99.5, 99.9)
        st.metric("Fiabilité", f"{reliability:.2f}%")
    
    # Analyse comparative
    st.write("### 📊 Analyse Comparative Sources")
    
    comparison_data = {
        'Source': ['Fusion', 'Fission', 'Solaire', 'Éolien', 'Hydrogène', 'Géothermie'],
        'LCOE ($/MWh)': [45, 55, 40, 38, 60, 50],
        'Facteur Capacité (%)': [85, 90, 25, 35, 60, 75],
        'Émissions (gCO₂/kWh)': [0, 12, 45, 11, 0, 38],
        'Durée Vie (ans)': [40, 60, 30, 25, 30, 50]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Graphique radar
    fig = go.Figure()
    
    categories = ['LCOE', 'Capacité', 'Émissions (inv)', 'Durée Vie']
    
    for idx, source in enumerate(df_comparison['Source']):
        # Normaliser valeurs
        lcoe_norm = 100 - (df_comparison.loc[idx, 'LCOE ($/MWh)'] / 60 * 100)
        capacity_norm = df_comparison.loc[idx, 'Facteur Capacité (%)']
        emissions_norm = 100 - (df_comparison.loc[idx, 'Émissions (gCO₂/kWh)'] / 50 * 100)
        life_norm = df_comparison.loc[idx, 'Durée Vie (ans)'] / 60 * 100
        
        values = [lcoe_norm, capacity_norm, emissions_norm, life_norm]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            name=source
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Comparaison Multi-Critères Sources Énergétiques",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.write("### 📋 Données Détaillées")
    st.dataframe(df_comparison, use_container_width=True)
    
    # Métriques avancées
    st.write("### 📈 Métriques Avancées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Financier:**")
        st.metric("ROI Moyen", "12.5%")
        st.metric("Payback Period", "7.2 ans")
        st.metric("NPV", "$2.4M")
        st.metric("IRR", "14.8%")
    
    with col2:
        st.write("**Opérationnel:**")
        st.metric("MTBF", "8,760 heures")
        st.metric("MTTR", "4.2 heures")
        st.metric("OEE", "87.3%")
        st.metric("Downtime", "2.1%")

elif page == "⚙️ Paramètres":
    st.header("⚙️ Paramètres Plateforme")
    
    st.subheader("🔧 Configuration Générale")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Système", "🔐 Sécurité", "🌐 API", "📊 Export"])
    
    with tab1:
        st.write("### ⚙️ Paramètres Système")
        
        col1, col2 = st.columns(2)
        
        with col1:
            update_frequency = st.selectbox("Fréquence Mise à Jour",
                ["Temps Réel", "1 minute", "5 minutes", "15 minutes"])
            
            data_retention = st.slider("Rétention Données (jours)", 30, 365, 90)
            
            enable_notifications = st.checkbox("Notifications Actives", value=True)
            
            enable_autosave = st.checkbox("Sauvegarde Auto", value=True)
        
        with col2:
            theme = st.selectbox("Thème Interface",
                ["Dark (Défaut)", "Light", "Auto"])
            
            language = st.selectbox("Langue",
                ["Français", "English", "Español", "Deutsch"])
            
            timezone = st.selectbox("Fuseau Horaire",
                ["UTC", "Europe/Paris", "America/New_York", "Asia/Tokyo"])
        
        if st.button("💾 Sauvegarder Paramètres", type="primary"):
            st.success("✅ Paramètres sauvegardés!")
            log_event("Paramètres système mis à jour", "INFO")
    
    with tab2:
        st.write("### 🔐 Sécurité & Accès")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Authentification:**")
            
            two_factor = st.checkbox("2FA Activée", value=True)
            session_timeout = st.slider("Timeout Session (min)", 5, 120, 30)
            
            st.write("\n**Permissions:**")
            access_level = st.selectbox("Niveau Accès",
                ["Admin", "Opérateur", "Analyste", "Lecture Seule"])
        
        with col2:
            st.write("**Logs & Audit:**")
            
            enable_audit = st.checkbox("Audit Trail", value=True)
            log_level = st.selectbox("Niveau Logs",
                ["DEBUG", "INFO", "WARNING", "ERROR"])
            
            st.write("\n**Sauvegardes:**")
            backup_frequency = st.selectbox("Fréquence Backup",
                ["Horaire", "Quotidien", "Hebdomadaire"])
    
    with tab3:
        st.write("### 🌐 Configuration API")
        
        st.write("**Endpoints API:**")
        
        api_base_url = "https://api.energy-platform.com/v1"
        
        st.code(f"""
# Base URL
{api_base_url}

# Endpoints
GET  /reactors              # Liste réacteurs
POST /reactors              # Créer réacteur
GET  /reactors/{{id}}        # Détails réacteur
PUT  /reactors/{{id}}        # Mettre à jour
DELETE /reactors/{{id}}      # Supprimer

GET  /production            # Données production
GET  /consumption           # Données consommation
GET  /storage               # État stockage
POST /optimize              # Lancer optimisation

GET  /analytics/kpis        # KPIs
GET  /analytics/predictions # Prédictions
        """)
        
        st.write("**Clés API:**")
        
        api_key = "sk_live_" + "x" * 40
        st.text_input("API Key", api_key, type="password")
        
        if st.button("🔄 Régénérer Clé API"):
            st.warning("⚠️ Confirmation requise")
            st.info("La clé actuelle sera révoquée")
    
    with tab4:
        st.write("### 📊 Export Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Format Export:**")
            
            export_format = st.selectbox("Format",
                ["CSV", "JSON", "Excel", "Parquet", "HDF5"])
            
            export_data = st.multiselect("Données à Exporter",
                ["Réacteurs", "Production", "Consommation", "Stockage", 
                 "Optimisations", "Analytics", "Logs"],
                default=["Production", "Consommation"])
            
            date_range = st.date_input("Plage Dates",
                value=(datetime.now() - timedelta(days=30), datetime.now()))
        
        with col2:
            st.write("**Options:**")
            
            compress = st.checkbox("Compression", value=True)
            include_metadata = st.checkbox("Inclure Métadonnées", value=True)
            anonymize = st.checkbox("Anonymiser Données Sensibles", value=False)
        
        if st.button("📥 Exporter Données", type="primary", use_container_width=True):
            with st.spinner("Export en cours..."):
                import time
                
                progress = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.02)
                    progress.progress(i + 1)
                
                progress.empty()
                
                st.success("✅ Export terminé!")
                
                # Simuler fichier
                filename = f"energy_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
                file_size = np.random.uniform(5, 50)
                
                st.info(f"📁 Fichier: {filename} ({file_size:.1f} MB)")
                
                st.download_button(
                    label="⬇️ Télécharger",
                    data="# Données exportées\n# Format: " + export_format,
                    file_name=filename,
                    mime="application/octet-stream"
                )
                
                log_event(f"Export données: {filename}", "INFO") 

# ==================== PAGE: COMPUTING QUANTIQUE ====================
elif page == "⚛️ Computing Quantique":
    st.header("⚛️ Optimisation Quantique pour l'Énergie")
    
    st.info("""
    **Quantum Computing Applications**
    
    - Optimisation réacteurs fusion
    - Simulation matériaux avancés
    - Design catalyseurs hydrogène
    - Optimisation portfolios énergétiques
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ Simulateur", "🔬 Optimisation", "📊 Résultats", "🎯 Applications"])
    
    with tab1:
        st.subheader("⚛️ Simulateur Quantique")
        
        st.write("### 🎛️ Configuration Circuit Quantique")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            n_qubits = st.slider("Nombre Qubits", 2, 50, 10, 1)
            
            circuit_depth = st.slider("Profondeur Circuit", 1, 100, 20, 1)
            
            algorithm = st.selectbox("Algorithme",
                ["VQE (Variational Quantum Eigensolver)",
                 "QAOA (Quantum Approx. Optimization)",
                 "Quantum Annealing",
                 "Grover Search",
                 "Shor Factorization"])
            
            backend = st.selectbox("Backend",
                ["Simulateur Local", "IBM Quantum", "Google Quantum", "IonQ", "Rigetti"])
        
        with col2:
            st.write("### 📊 Capacités")
            
            max_states = 2 ** n_qubits
            st.metric("États Possibles", f"{max_states:,}")
            
            if max_states > 1e9:
                st.success("🚀 Avantage Quantique")
            
            complexity = "O(2^n)" if algorithm in ["VQE", "QAOA"] else "O(√N)"
            st.write(f"**Complexité:** {complexity}")
        
        if st.button("⚛️ Exécuter Circuit Quantique", type="primary", use_container_width=True):
            with st.spinner(f"Exécution quantique sur {n_qubits} qubits..."):
                import time
                
                progress = st.progress(0)
                status = st.empty()
                
                for i in range(100):
                    time.sleep(0.05)
                    progress.progress(i + 1)
                    
                    if i < 30:
                        status.text("Initialisation qubits...")
                    elif i < 60:
                        status.text("Application portes quantiques...")
                    elif i < 90:
                        status.text("Mesure états...")
                    else:
                        status.text("Analyse résultats...")
                
                progress.empty()
                status.empty()
                
                st.success("✅ Circuit exécuté avec succès!")
                
                # Résultats simulés
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fidelity = np.random.uniform(0.92, 0.99)
                    st.metric("Fidélité", f"{fidelity:.4f}")
                
                with col2:
                    shots = 1000
                    st.metric("Shots", f"{shots:,}")
                
                with col3:
                    runtime_ms = n_qubits * circuit_depth * np.random.uniform(0.1, 0.5)
                    st.metric("Temps Exec", f"{runtime_ms:.1f} ms")
                
                # Visualisation états
                st.write("### 📊 Distribution États Quantiques")
                
                n_states = min(2**n_qubits, 16)
                states = [format(i, f'0{n_qubits}b')[::-1] for i in range(n_states)]
                probabilities = np.random.dirichlet(np.ones(n_states))
                
                fig = go.Figure(data=[go.Bar(
                    x=states,
                    y=probabilities,
                    marker_color='#FFD700'
                )])
                
                fig.update_layout(
                    title="États Quantiques Mesurés",
                    xaxis_title="État |ψ⟩",
                    yaxis_title="Probabilité",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Optimisation Quantique Réacteur Fusion")
        
        st.write("### ⚛️ Optimisation Paramètres Plasma")
        
        if not st.session_state.energy_lab['reactors']:
            st.warning("⚠️ Créez d'abord un réacteur fusion")
        else:
            selected_reactor = st.selectbox("Sélectionner Réacteur",
                list(st.session_state.energy_lab['reactors'].keys()),
                format_func=lambda x: st.session_state.energy_lab['reactors'][x]['name'])
            
            reactor = st.session_state.energy_lab['reactors'][selected_reactor]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Paramètres Actuels:**")
                st.write(f"• Température: {reactor['temperature_k']/1e6:.0f} M°K")
                st.write(f"• Pression: {reactor['pressure_atm']:.1f} atm")
                st.write(f"• Q Factor: {reactor.get('q_factor', 0):.2f}")
            
            with col2:
                st.write("**Objectifs:**")
                
                target_q = st.number_input("Q Factor Cible", 1.0, 50.0, 15.0, 0.5)
                max_iterations = st.slider("Itérations Max", 10, 1000, 100)
            
            if st.button("⚛️ Optimiser avec Quantique", type="primary", use_container_width=True):
                with st.spinner("Optimisation quantique en cours..."):
                    import time
                    time.sleep(3)
                    
                    # Simuler optimisation quantique
                    optimization = quantum_optimize_reactor(
                        reactor['temperature_k'],
                        reactor['pressure_atm'],
                        reactor.get('fuel_mass_kg', 0.1) * 1e20
                    )
                    
                    st.success("✅ Optimisation quantique terminée!")
                    
                    st.write("### 🎯 Paramètres Optimaux")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Température", 
                                f"{optimization['optimal_temperature']/1e6:.0f} M°K",
                                f"{(optimization['optimal_temperature'] - reactor['temperature_k'])/1e6:.0f} M°K")
                    
                    with col2:
                        st.metric("Pression", 
                                f"{optimization['optimal_pressure']:.2f} atm",
                                f"{optimization['optimal_pressure'] - reactor['pressure_atm']:.2f} atm")
                    
                    with col3:
                        st.metric("Gain Énergie", 
                                optimization['quantum_advantage'],
                                f"+{(optimization['energy_gain_factor']-1)*100:.0f}%")
                    
                    st.success(f"🚀 Avantage quantique: {optimization['quantum_advantage']}")
                    
                    # Sauvegarder optimisation
                    st.session_state.energy_lab['quantum_simulations'].append({
                        'timestamp': datetime.now().isoformat(),
                        'reactor_id': selected_reactor,
                        'optimization': optimization
                    })
                    
                    log_event(f"Optimisation quantique: {optimization['quantum_advantage']}", "SUCCESS")
                    
                    if st.button("✅ Appliquer Paramètres"):
                        reactor['temperature_k'] = optimization['optimal_temperature']
                        reactor['pressure_atm'] = optimization['optimal_pressure']
                        reactor['q_factor'] *= optimization['energy_gain_factor']
                        
                        st.success("Paramètres appliqués au réacteur!")
                        st.rerun()
    
    with tab3:
        st.subheader("📊 Résultats Optimisations Quantiques")
        
        if st.session_state.energy_lab['quantum_simulations']:
            st.write(f"### 📈 {len(st.session_state.energy_lab['quantum_simulations'])} Optimisations Réalisées")
            
            gains = [sim['optimization']['energy_gain_factor'] 
                    for sim in st.session_state.energy_lab['quantum_simulations']]
            
            fig = go.Figure(data=[go.Scatter(
                y=gains,
                mode='lines+markers',
                line=dict(color='#FFD700', width=3),
                marker=dict(size=10)
            )])
            
            fig.add_hline(y=1, line_dash="dash", line_color="white", 
                         annotation_text="Baseline")
            
            fig.update_layout(
                title="Gains Énergétiques - Optimisations Quantiques",
                xaxis_title="Simulation #",
                yaxis_title="Facteur Gain",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Gain Moyen", f"{np.mean(gains):.2f}x")
            with col2:
                st.metric("Gain Maximum", f"{np.max(gains):.2f}x")
            with col3:
                improvement = (np.mean(gains) - 1) * 100
                st.metric("Amélioration Moy.", f"{improvement:.1f}%")
        else:
            st.info("Aucune optimisation quantique réalisée")
    
    with tab4:
        st.subheader("🎯 Applications Quantum Computing")
        
        applications = {
            "Optimisation Réacteurs Fusion": {
                "description": "Trouver paramètres optimaux plasma",
                "gain": "15-40% amélioration Q factor",
                "algorithme": "VQE + QAOA",
                "qubits": "20-50"
            },
            "Simulation Matériaux": {
                "description": "Prédire propriétés nouveaux matériaux",
                "gain": "100x plus rapide que classique",
                "algorithme": "Quantum Phase Estimation",
                "qubits": "30-100"
            },
            "Design Catalyseurs H₂": {
                "description": "Optimiser catalyseurs électrolyse",
                "gain": "5-10% efficacité supplémentaire",
                "algorithme": "VQE",
                "qubits": "15-40"
            },
            "Optimisation Portfolios": {
                "description": "Mix énergétique optimal",
                "gain": "Réduction coûts 20-30%",
                "algorithme": "QAOA",
                "qubits": "10-30"
            },
            "Prévision Météo": {
                "description": "Prédiction production renouvelables",
                "gain": "Précision +15%",
                "algorithme": "Quantum Machine Learning",
                "qubits": "20-50"
            }
        }
        
        for app_name, specs in applications.items():
            with st.expander(f"⚛️ {app_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:**")
                    st.write(specs['description'])
                    
                    st.write(f"\n**Gain:**")
                    st.write(specs['gain'])
                
                with col2:
                    st.write(f"**Algorithme:**")
                    st.write(specs['algorithme'])
                    
                    st.write(f"\n**Qubits Nécessaires:**")
                    st.write(specs['qubits'])

# ==================== PAGE: BIO-BATTERIES ====================
elif page == "🧬 Bio-Batteries":
    st.header("🧬 Bio-Computing & Batteries Biologiques")
    
    st.info("""
    **Bio-Computing pour l'Énergie**
    
    - Batteries organiques biodégradables
    - Biocarburants 3ème génération
    - Capture CO₂ par micro-algues
    - Production H₂ par enzymes
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔋 Bio-Batteries", "🌱 Biocarburants", "🌿 Capture CO₂", "💧 H₂ Enzymatique"])
    
    with tab1:
        st.subheader("🔋 Batteries Organiques")
        
        st.write("### 🧬 Créer Bio-Batterie")
        
        with st.form("create_biobattery"):
            col1, col2 = st.columns(2)
            
            with col1:
                battery_name = st.text_input("Nom Bio-Batterie", "BioCell-01")
                
                organic_material = st.selectbox("Matériau Organique",
                    ["Quinone", "TEMPO", "Lignine", "Cellulose", "Chitosan"])
                
                capacity_kwh = st.number_input("Capacité (kWh)", 1, 1000, 100, 10)
                
                voltage_v = st.slider("Tension (V)", 1.0, 5.0, 3.3, 0.1)
            
            with col2:
                electrolyte = st.selectbox("Électrolyte",
                    ["Aqueux", "Gel Polymère", "Ionique Liquide"])
                
                cycles_life = st.number_input("Cycles Vie", 100, 10000, 2000, 100)
                
                biodegradable = st.checkbox("100% Biodégradable", value=True)
                
                toxicity = st.selectbox("Toxicité",
                    ["Nulle", "Très Faible", "Faible"])
            
            if st.form_submit_button("🧬 Créer Bio-Batterie", type="primary"):
                battery_id = f"biobat_{len(st.session_state.energy_lab['bio_batteries']) + 1}"
                
                efficiency = np.random.uniform(0.85, 0.95)
                
                bio_battery = {
                    'id': battery_id,
                    'name': battery_name,
                    'material': organic_material,
                    'capacity_kwh': capacity_kwh,
                    'voltage_v': voltage_v,
                    'electrolyte': electrolyte,
                    'cycles_life': cycles_life,
                    'biodegradable': biodegradable,
                    'toxicity': toxicity,
                    'efficiency': efficiency,
                    'current_charge': capacity_kwh * 0.8,
                    'cycles_used': 0,
                    'status': 'operational',
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.energy_lab['bio_batteries'][battery_id] = bio_battery
                log_event(f"Bio-batterie créée: {battery_name}", "SUCCESS")
                
                st.success(f"✅ Bio-Batterie '{battery_name}' créée!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Capacité", f"{capacity_kwh} kWh")
                with col2:
                    st.metric("Efficacité", f"{efficiency*100:.1f}%")
                with col3:
                    st.metric("Cycles Vie", f"{cycles_life:,}")
                
                if biodegradable:
                    st.success("🌱 100% Biodégradable")
                
                st.rerun()
        
        # Afficher bio-batteries existantes
        if st.session_state.energy_lab['bio_batteries']:
            st.write("### 🔋 Bio-Batteries Actives")
            
            for bat_id, battery in st.session_state.energy_lab['bio_batteries'].items():
                with st.expander(f"🧬 {battery['name']} ({battery['material']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        charge_pct = (battery['current_charge'] / battery['capacity_kwh']) * 100
                        st.metric("Charge", f"{charge_pct:.0f}%")
                        st.progress(charge_pct / 100)
                    
                    with col2:
                        st.metric("Cycles", f"{battery['cycles_used']:,}")
                        remaining = battery['cycles_life'] - battery['cycles_used']
                        st.write(f"Restants: {remaining:,}")
                    
                    with col3:
                        st.metric("Efficacité", f"{battery['efficiency']*100:.1f}%")
                        st.write(f"**Toxicité:** {battery['toxicity']}")
                    
                    if battery['biodegradable']:
                        st.success("🌱 Biodégradable")
    
    with tab2:
        st.subheader("🌱 Biocarburants Avancés")
        
        st.write("### 🧬 Production Biocarburants 3G/4G")
        
        col1, col2 = st.columns(2)
        
        with col1:
            biofuel_type = st.selectbox("Type Biocarburant",
                ["Micro-algues (3G)", "Cyanobactéries", "Synthèse Enzymatique (4G)", 
                 "E-fuel (CO₂ + H₂)", "Bio-méthane"])
            
            biomass_kg = st.number_input("Biomasse (kg)", 1.0, 10000.0, 1000.0, 10.0)
            
            conversion_efficiency = st.slider("Efficacité Conversion (%)", 20, 80, 60, 5)
        
        with col2:
            st.write("**Caractéristiques:**")
            
            yields = {
                "Micro-algues (3G)": 50,  # L/tonne
                "Cyanobactéries": 60,
                "Synthèse Enzymatique (4G)": 70,
                "E-fuel (CO₂ + H₂)": 40,
                "Bio-méthane": 55
            }
            
            fuel_yield = yields[biofuel_type]
            st.metric("Rendement", f"{fuel_yield} L/tonne")
            
            co2_capture = biomass_kg * 1.8  # kg CO2 capturé
            st.metric("CO₂ Capturé", f"{co2_capture:.0f} kg")
        
        if st.button("🌱 Produire Biocarburant", type="primary"):
            with st.spinner("Production en cours..."):
                import time
                time.sleep(2)
                
                # Calculs production
                fuel_liters = (biomass_kg / 1000) * fuel_yield * (conversion_efficiency / 100)
                energy_kwh = bio_generate_electricity(biomass_kg, conversion_efficiency / 100)
                
                st.success("✅ Production terminée!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Biocarburant", f"{fuel_liters:.0f} L")
                with col2:
                    st.metric("Énergie Équiv.", f"{energy_kwh:.0f} kWh")
                with col3:
                    st.metric("CO₂ Net", f"-{co2_capture:.0f} kg")
                
                log_event(f"Biocarburant produit: {fuel_liters:.0f} L", "SUCCESS")
    
    with tab3:
        st.subheader("🌿 Capture CO₂ Biologique")
        
        st.write("### 🦠 Systèmes Bio-Capture")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            capture_method = st.selectbox("Méthode",
                ["Photo-bioréacteur Micro-algues", "Cyanobactéries", 
                 "Arbres Artificiels Enzymes", "Biochar"])
            
            reactor_volume_m3 = st.slider("Volume Réacteur (m³)", 1, 1000, 100, 10)
            
            co2_concentration = st.slider("Concentration CO₂ (%)", 0.1, 20.0, 5.0, 0.1)
            
            light_intensity = st.slider("Intensité Lumineuse (µmol/m²/s)", 0, 2000, 800, 100)
        
        with col2:
            st.write("**Performance:**")
            
            # Calcul capture théorique
            capture_rate = reactor_volume_m3 * 1.5 * (light_intensity / 1000)  # kg CO2/jour
            
            st.metric("Capture/Jour", f"{capture_rate:.1f} kg CO₂")
            st.metric("Capture/An", f"{capture_rate * 365 / 1000:.1f} tonnes CO₂")
            
            biomass_growth = capture_rate * 0.5  # kg biomasse/jour
            st.metric("Biomasse/Jour", f"{biomass_growth:.1f} kg")
        
        if st.button("🌿 Simuler Capture 30 jours", type="primary"):
            with st.spinner("Simulation bio-capture..."):
                import time
                time.sleep(2)
                
                # Simulation 30 jours
                days = 30
                daily_capture = []
                cumulative = 0
                
                for day in range(days):
                    # Variation jour/nuit et conditions
                    factor = np.random.uniform(0.8, 1.2)
                    daily = capture_rate * factor
                    cumulative += daily
                    daily_capture.append(cumulative)
                
                st.success("✅ Simulation terminée!")
                
                # Graphique
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(1, days+1)),
                    y=daily_capture,
                    mode='lines+markers',
                    fill='tozeroy',
                    line=dict(color='#00FF00', width=3)
                ))
                
                fig.update_layout(
                    title="Capture CO₂ Cumulative (30 jours)",
                    xaxis_title="Jour",
                    yaxis_title="CO₂ Capturé (kg)",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Capturé", f"{cumulative:.0f} kg CO₂")
                with col2:
                    st.metric("Moyenne/Jour", f"{cumulative/days:.1f} kg")
                with col3:
                    biomass_total = cumulative * 0.5
                    st.metric("Biomasse Produite", f"{biomass_total:.0f} kg")
    
    with tab4:
        st.subheader("💧 Production H₂ Enzymatique")
        
        st.write("### 🧬 Hydrogénases & Photo-production")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enzyme_system = st.selectbox("Système Enzymatique",
                ["Hydrogénase [FeFe]", "Hydrogénase [NiFe]", 
                 "Cyanobactéries Modifiées", "E. coli Engineered"])
            
            substrate = st.selectbox("Substrat",
                ["Glucose", "Acétate", "Lumière + H₂O", "Déchets Organiques"])
            
            reactor_l = st.number_input("Volume Réacteur (L)", 1, 10000, 1000, 100)
            
            temperature_c = st.slider("Température (°C)", 20, 60, 37, 1)
        
        with col2:
            st.write("**Paramètres Production:**")
            
            # Calcul production H2
            base_rate = 50  # mL H2/L/h
            temp_factor = 1 + ((temperature_c - 37) / 100)
            
            h2_rate_ml_h = reactor_l * base_rate * temp_factor
            h2_rate_l_day = (h2_rate_ml_h * 24) / 1000
            
            st.metric("Production", f"{h2_rate_ml_h:.0f} mL/h")
            st.metric("Production/Jour", f"{h2_rate_l_day:.1f} L H₂")
            
            # Masse H2 (1L H2 = 0.09 g à STP)
            h2_g_day = h2_rate_l_day * 0.09
            st.metric("Masse H₂/Jour", f"{h2_g_day:.2f} g")
        
        if st.button("🧬 Lancer Production Enzymatique", type="primary"):
            with st.spinner("Production enzymatique H₂..."):
                import time
                time.sleep(2)
                
                # Simulation 24h
                hours = 24
                production = []
                
                for h in range(hours):
                    # Variation activité enzymatique
                    activity = np.sin(h * np.pi / 12) * 0.3 + 0.7  # Cycle circadien
                    h2_ml = h2_rate_ml_h * activity * np.random.uniform(0.9, 1.1)
                    production.append(h2_ml)
                
                st.success("✅ Production 24h terminée!")
                
                total_h2_l = sum(production) / 1000
                total_h2_g = total_h2_l * 0.09
                energy_kwh = total_h2_g * 33.3 / 1000
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("H₂ Produit", f"{total_h2_l:.2f} L")
                with col2:
                    st.metric("Masse H₂", f"{total_h2_g:.3f} g")
                with col3:
                    st.metric("Énergie Équiv.", f"{energy_kwh:.3f} kWh")
                
                # Graphique
                fig = go.Figure(data=[go.Scatter(
                    x=list(range(24)),
                    y=production,
                    mode='lines+markers',
                    fill='tozeroy',
                    line=dict(color='#00CED1', width=2)
                )])
                
                fig.update_layout(
                    title="Production H₂ Enzymatique (24h)",
                    xaxis_title="Heure",
                    yaxis_title="Production (mL/h)",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: FISSION AVANCÉE ====================
elif page == "🔬 Fission Avancée":
    st.header("🔬 Réacteurs Fission Nucléaire Avancés")
    
    st.info("""
    **Fission Nucléaire 4ème Génération**
    
    Réacteurs avancés avec sécurité passive, combustibles innovants et gestion optimisée des déchets.
    
    **Technologies:**
    - Réacteurs à neutrons rapides (SFR)
    - Réacteurs à sels fondus (MSR)
    - Réacteurs à haute température (VHTR)
    - SMR (Small Modular Reactors)
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ Réacteurs", "➕ Créer Réacteur", "⚡ Production", "📊 Performance"])
    
    with tab1:
        st.subheader("⚛️ Réacteurs Fission Actifs")
        
        # Filtrer réacteurs fission dans power_plants
        fission_reactors = {k: v for k, v in st.session_state.energy_lab['power_plants'].items() 
                           if v.get('type') == 'Fission'}
        
        if not fission_reactors:
            st.info("Aucun réacteur fission créé. Créez votre premier réacteur!")
            
            if st.button("➕ Créer Premier Réacteur Fission", type="primary"):
                st.info("Accédez à l'onglet 'Créer Réacteur'")
        else:
            for reactor_id, reactor in fission_reactors.items():
                with st.expander(f"⚛️ {reactor['name']} ({reactor['reactor_type']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Spécifications")
                        st.write(f"**Type:** {reactor['reactor_type']}")
                        st.write(f"**Combustible:** {reactor['fuel_type']}")
                        st.write(f"**Capacité:** {reactor['capacity_mw']} MW")
                        st.write(f"**Température:** {reactor.get('coolant_temp_c', 0)} °C")
                        
                        status_icon = "🟢" if reactor['status'] == 'operational' else "🔴"
                        st.write(f"**Statut:** {status_icon} {reactor['status']}")
                    
                    with col2:
                        st.write("### ⚡ Performance")
                        st.metric("Efficacité", f"{reactor.get('efficiency', 0)*100:.1f}%")
                        st.metric("Facteur Charge", f"{reactor.get('capacity_factor', 0)*100:.1f}%")
                        st.metric("Puissance Actuelle", f"{reactor.get('current_power_mw', 0):.0f} MW")
                        st.metric("Burnup", f"{reactor.get('burnup_mwd_kg', 0):.0f} MWd/kg")
                    
                    with col3:
                        st.write("### 🎯 Sécurité")
                        st.write("**Sécurité Passive:** " + ("✅" if reactor.get('passive_safety', False) else "❌"))
                        st.write("**Contrôle IA:** " + ("✅" if reactor.get('ai_control', False) else "❌"))
                        st.write("**Cycles Combustible:** " + f"{reactor.get('fuel_cycles', 0)}")
                        
                        temp_marge = reactor.get('coolant_temp_c', 0) / reactor.get('max_temp_c', 1000)
                        if temp_marge < 0.7:
                            st.success("✅ Température Normale")
                        else:
                            st.warning("⚠️ Température Élevée")
                    
                    st.markdown("---")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("⚡ Augmenter Puissance", key=f"increase_{reactor_id}"):
                            current = reactor.get('current_power_mw', reactor['capacity_mw'] * 0.9)
                            reactor['current_power_mw'] = min(current + 50, reactor['capacity_mw'])
                            st.success("Puissance augmentée!")
                            st.rerun()
                    
                    with col2:
                        if st.button("🔽 Réduire Puissance", key=f"decrease_{reactor_id}"):
                            current = reactor.get('current_power_mw', reactor['capacity_mw'] * 0.9)
                            reactor['current_power_mw'] = max(current - 50, 0)
                            st.info("Puissance réduite!")
                            st.rerun()
                    
                    with col3:
                        if st.button("🛑 Arrêt d'Urgence", key=f"scram_{reactor_id}"):
                            reactor['current_power_mw'] = 0
                            reactor['status'] = 'shutdown'
                            st.error("SCRAM activé!")
                            st.rerun()
                    
                    with col4:
                        if st.button("🗑️ Supprimer", key=f"del_{reactor_id}"):
                            del st.session_state.energy_lab['power_plants'][reactor_id]
                            log_event(f"Réacteur fission supprimé: {reactor['name']}", "WARNING")
                            st.rerun()
    
    with tab2:
        st.subheader("➕ Créer Nouveau Réacteur Fission")
        
        with st.form("create_fission_reactor"):
            st.write("### 🎨 Configuration Réacteur")
            
            col1, col2 = st.columns(2)
            
            with col1:
                reactor_name = st.text_input("Nom Réacteur", "Fission-Gen4-01")
                
                reactor_type = st.selectbox("Type Réacteur",
                    ["SFR (Sodium Fast Reactor)", 
                     "MSR (Molten Salt Reactor)", 
                     "VHTR (Very High Temp Reactor)",
                     "SMR (Small Modular Reactor)",
                     "BWR (Boiling Water Reactor)",
                     "PWR (Pressurized Water Reactor)"])
                
                fuel_type = st.selectbox("Combustible",
                    ["UO₂ Enrichi (3-5%)", 
                     "MOX (U-Pu)", 
                     "Thorium",
                     "U-233",
                     "Combustible Métallique"])
                
                capacity_mw = st.number_input("Capacité Thermique (MW)", 100, 5000, 1000, 100)
            
            with col2:
                coolant = st.selectbox("Fluide Caloporteur",
                    ["Eau Légère", "Eau Lourde", "Sodium Liquide", "Sels Fondus", "Hélium", "Plomb"])
                
                coolant_temp_c = st.number_input("Température Caloporteur (°C)", 250, 900, 330, 10)
                
                pressure_bar = st.slider("Pression Primaire (bar)", 1, 200, 155, 5)
                
                enrichment = st.slider("Enrichissement (%)", 0.7, 20.0, 4.5, 0.1)
            
            st.write("### ⚙️ Paramètres Avancés")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fuel_mass_kg = st.number_input("Masse Combustible (kg)", 1000, 200000, 50000, 1000)
                burnup_target = st.number_input("Burnup Cible (MWd/kg)", 30, 200, 50, 5)
            
            with col2:
                passive_safety = st.checkbox("🛡️ Sécurité Passive", value=True)
                ai_control = st.checkbox("🤖 Contrôle IA", value=True)
            
            with col3:
                waste_recycling = st.checkbox("♻️ Recyclage Déchets", value=False)
                breeding = st.checkbox("⚛️ Mode Surgénérateur", value=False)
            
            if st.form_submit_button("⚛️ Créer Réacteur Fission", type="primary"):
                if not reactor_name:
                    st.error("⚠️ Veuillez donner un nom")
                else:
                    reactor_id = f"fission_{len(st.session_state.energy_lab['power_plants']) + 1}"
                    
                    # Calculer efficacité selon type
                    efficiency_map = {
                        "SFR": 0.42,
                        "MSR": 0.45,
                        "VHTR": 0.48,
                        "SMR": 0.33,
                        "BWR": 0.33,
                        "PWR": 0.33
                    }
                    
                    reactor_type_short = reactor_type.split()[0]
                    efficiency = efficiency_map.get(reactor_type_short, 0.33)
                    
                    # Calculer capacité électrique
                    capacity_electric_mw = capacity_mw * efficiency
                    
                    # Facteur de charge
                    capacity_factor = 0.90 if passive_safety else 0.85
                    
                    fission_reactor = {
                        'id': reactor_id,
                        'name': reactor_name,
                        'type': 'Fission',
                        'reactor_type': reactor_type_short,
                        'fuel_type': fuel_type,
                        'capacity_mw': capacity_mw,
                        'capacity_electric_mw': capacity_electric_mw,
                        'coolant': coolant,
                        'coolant_temp_c': coolant_temp_c,
                        'max_temp_c': coolant_temp_c * 1.5,
                        'pressure_bar': pressure_bar,
                        'enrichment_pct': enrichment,
                        'fuel_mass_kg': fuel_mass_kg,
                        'burnup_target': burnup_target,
                        'burnup_mwd_kg': 0,
                        'efficiency': efficiency,
                        'capacity_factor': capacity_factor,
                        'passive_safety': passive_safety,
                        'ai_control': ai_control,
                        'waste_recycling': waste_recycling,
                        'breeding': breeding,
                        'current_power_mw': capacity_electric_mw * 0.95,
                        'fuel_cycles': 0,
                        'status': 'operational',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.energy_lab['power_plants'][reactor_id] = fission_reactor
                    log_event(f"Réacteur fission créé: {reactor_name}", "SUCCESS")
                    
                    with st.spinner("Initialisation réacteur..."):
                        import time
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)
                            progress_bar.progress(i + 1)
                        progress_bar.empty()
                    
                    st.success(f"✅ Réacteur '{reactor_name}' créé!")
                    st.balloons()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Capacité", f"{capacity_electric_mw:.0f} MWe")
                    with col2:
                        st.metric("Efficacité", f"{efficiency*100:.1f}%")
                    with col3:
                        annual_gwh = capacity_electric_mw * 24 * 365 * capacity_factor / 1000
                        st.metric("Production/an", f"{annual_gwh:.0f} GWh")
                    with col4:
                        st.metric("Facteur Charge", f"{capacity_factor*100:.0f}%")
                    
                    if passive_safety:
                        st.success("🛡️ Sécurité passive active")
                    
                    if breeding:
                        st.info("⚛️ Mode surgénérateur: Production Pu-239")
                    
                    st.rerun()
    
    with tab3:
        st.subheader("⚡ Production Énergétique")
        
        fission_reactors = {k: v for k, v in st.session_state.energy_lab['power_plants'].items() 
                           if v.get('type') == 'Fission'}
        
        if fission_reactors:
            selected_reactor = st.selectbox("Sélectionner Réacteur",
                list(fission_reactors.keys()),
                format_func=lambda x: fission_reactors[x]['name'])
            
            reactor = fission_reactors[selected_reactor]
            
            st.write("### ⚡ Production Temps Réel")
            
            # Données actuelles
            current_power = reactor.get('current_power_mw', reactor['capacity_electric_mw'] * 0.95)
            max_power = reactor['capacity_electric_mw']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Puissance", f"{current_power:.0f} MW")
                # st.progress(current_power / max_power)
                if max_power > 0:
                    progress_value = current_power / max_power
                    progress_value = min(progress_value, 1.0)  # jamais > 1
                    st.progress(progress_value)
                else:
                    st.progress(0)

            
            with col2:
                daily_mwh = current_power * 24
                st.metric("Production/Jour", f"{daily_mwh:.0f} MWh")
            
            with col3:
                temp = reactor.get('coolant_temp_c', 330)
                st.metric("Température", f"{temp} °C")
            
            with col4:
                burnup = reactor.get('burnup_mwd_kg', 0)
                st.metric("Burnup", f"{burnup:.1f} MWd/kg")
            
            # Simulation production 30 jours
            st.write("### 📊 Historique Production (30 jours)")
            
            if st.button("📊 Simuler 30 jours", type="primary"):
                with st.spinner("Simulation production..."):
                    import time
                    time.sleep(1)
                    
                    days = 30
                    production_daily = []
                    
                    for day in range(days):
                        # Variations aléatoires (maintenance, etc.)
                        if np.random.random() < 0.05:  # 5% chance maintenance
                            daily_prod = current_power * 24 * 0.3
                        else:
                            daily_prod = current_power * 24 * np.random.uniform(0.92, 0.98)
                        
                        production_daily.append(daily_prod)
                    
                    st.success("✅ Simulation terminée!")
                    
                    # Graphique
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=list(range(1, days+1)),
                        y=production_daily,
                        marker_color='#FF8C00'
                    ))
                    
                    fig.update_layout(
                        title="Production Quotidienne (30 jours)",
                        xaxis_title="Jour",
                        yaxis_title="Énergie (MWh)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    total_mwh = sum(production_daily)
                    avg_daily = np.mean(production_daily)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Production Totale", f"{total_mwh:.0f} MWh")
                    with col2:
                        st.metric("Moyenne/Jour", f"{avg_daily:.0f} MWh")
                    with col3:
                        availability = (avg_daily / (current_power * 24)) * 100
                        st.metric("Disponibilité", f"{availability:.1f}%")
        else:
            st.info("Créez d'abord un réacteur fission")
    
    with tab4:
        st.subheader("📊 Performance & Gestion Combustible")
        
        fission_reactors = {k: v for k, v in st.session_state.energy_lab['power_plants'].items() 
                           if v.get('type') == 'Fission'}
        
        if fission_reactors:
            selected_reactor = st.selectbox("Réacteur",
                list(fission_reactors.keys()),
                format_func=lambda x: fission_reactors[x]['name'],
                key="perf_reactor")
            
            reactor = fission_reactors[selected_reactor]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("### ⚛️ Cycle Combustible")
                
                burnup_current = reactor.get('burnup_mwd_kg', 0)
                burnup_target = reactor.get('burnup_target', 50)
                
                burnup_pct = (burnup_current / burnup_target) * 100
                
                st.progress(burnup_pct / 100)
                st.write(f"**Burnup:** {burnup_current:.1f} / {burnup_target} MWd/kg ({burnup_pct:.1f}%)")
                
                fuel_remaining_pct = 100 - burnup_pct
                st.metric("Combustible Restant", f"{fuel_remaining_pct:.1f}%")
                
                if burnup_pct > 90:
                    st.error("🚨 Recharge combustible nécessaire!")
                elif burnup_pct > 70:
                    st.warning("⚠️ Planifier recharge prochaine")
                else:
                    st.success("✅ Combustible OK")
                
                # Actions
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Recharger Combustible", key="refuel"):
                        reactor['burnup_mwd_kg'] = 0
                        reactor['fuel_cycles'] = reactor.get('fuel_cycles', 0) + 1
                        st.success("Combustible rechargé!")
                        log_event(f"Recharge combustible: {reactor['name']}", "INFO")
                        st.rerun()
                
                with col2:
                    if st.button("⚡ Simuler Burnup +10", key="burnup"):
                        reactor['burnup_mwd_kg'] = min(
                            reactor.get('burnup_mwd_kg', 0) + 10,
                            burnup_target
                        )
                        st.info("Burnup augmenté")
                        st.rerun()
            
            with col2:
                st.write("### 📊 Métriques")
                
                st.metric("Cycles Combustible", reactor.get('fuel_cycles', 0))
                
                efficiency = reactor.get('efficiency', 0.33)
                st.metric("Efficacité Thermique", f"{efficiency*100:.1f}%")
                
                capacity_factor = reactor.get('capacity_factor', 0.90)
                st.metric("Facteur Charge", f"{capacity_factor*100:.1f}%")
                
                st.write("\n### 🎯 Sécurité")
                
                if reactor.get('passive_safety'):
                    st.success("🛡️ Sécurité Passive")
                else:
                    st.info("⚙️ Sécurité Active")
                
                if reactor.get('ai_control'):
                    st.success("🤖 Contrôle IA")
            
            # Gestion déchets
            st.write("### ♻️ Gestion Déchets Radioactifs")
            
            fuel_used_kg = reactor.get('fuel_mass_kg', 50000) * (burnup_pct / 100)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                waste_high_level = fuel_used_kg * 0.03  # 3% déchets haute activité
                st.metric("Déchets Haute Activité", f"{waste_high_level:.0f} kg")
            
            with col2:
                waste_medium = fuel_used_kg * 0.07  # 7% moyenne activité
                st.metric("Déchets Moyenne Act.", f"{waste_medium:.0f} kg")
            
            with col3:
                if reactor.get('waste_recycling'):
                    recycling_rate = 95
                    st.metric("Taux Recyclage", f"{recycling_rate}%")
                    st.success("♻️ Recyclage actif")
                else:
                    st.info("♻️ Pas de recyclage")
            
            # Recommandations
            st.write("### 💡 Recommandations")
            
            recommendations = []
            
            if burnup_pct > 85:
                recommendations.append("🔴 URGENT: Planifier arrêt pour recharge")
            elif burnup_pct > 70:
                recommendations.append("🟡 Préparer recharge combustible")
            
            temp = reactor.get('coolant_temp_c', 330)
            max_temp = reactor.get('max_temp_c', 500)
            if temp / max_temp > 0.8:
                recommendations.append("🟡 Surveiller température caloporteur")
            
            if not reactor.get('ai_control'):
                recommendations.append("💡 Activer contrôle IA pour optimisation")
            
            if not reactor.get('waste_recycling'):
                recommendations.append("💡 Considérer recyclage déchets")
            
            if recommendations:
                for rec in recommendations:
                    st.write(f"• {rec}")
            else:
                st.success("✅ Aucune action requise - Fonctionnement optimal")
        else:
            st.info("Créez d'abord un réacteur fission")

# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal Système (20 dernières entrées)"):
    if st.session_state.energy_lab['log']:
        for event in st.session_state.energy_lab['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️" if level == "WARNING" else "❌"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>⚡ Energy Research Platform</h3>
        <p>Recherche Énergétique Avancée • IA • Quantique • Bio-Computing</p>
        <p><small>Fusion • Fission • Renouvelables • Stockage • Smart Grids</small></p>
        <p><small>Version 1.0.0 | Énergie du Futur</small></p>
        <p><small>⚡ Powering Tomorrow © 2024</small></p>
    </div>
""", unsafe_allow_html=True)