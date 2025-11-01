"""
Interface Streamlit pour la Plateforme de Réacteurs Nucléaires
Système intégré pour créer, développer, simuler et analyser
des réacteurs nucléaires et systèmes énergétiques
streamlit run nuclear_reactor_app.py
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
    page_title="☢️ Plateforme Réacteurs Nucléaires",
    page_icon="☢️",
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
        background: linear-gradient(90deg, #00b4d8 0%, #0077b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .reactor-card {
        border: 3px solid #0077b6;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.1) 0%, rgba(0, 119, 182, 0.1) 100%);
        box-shadow: 0 4px 12px rgba(0, 119, 182, 0.3);
    }
    .metric-box {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .danger-badge {
        background: linear-gradient(90deg, #ef233c 0%, #d90429 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .safe-badge {
        background: linear-gradient(90deg, #06ffa5 0%, #00d9ff 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .warning-badge {
        background: linear-gradient(90deg, #ffa500 0%, #ff8c00 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ====================
CONSTANTS = {
    'avogadro': 6.022e23,
    'u235_fission_energy': 200,  # MeV
    'neutron_mass': 1.008664916,  # uma
    'u235_thermal_xs': 585,  # barns
}

# ==================== INITIALISATION SESSION STATE ====================
if 'nuclear_system' not in st.session_state:
    st.session_state.nuclear_system = {
        'reactors': {},
        'fuel_cycles': {},
        'waste_inventory': {},
        'simulations': [],
        'incidents': [],
        'maintenance': [],
        'inspections': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================
def log_event(message: str):
    """Enregistre un événement"""
    st.session_state.nuclear_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })

def get_status_badge(status: str) -> str:
    """Retourne un badge HTML pour le statut"""
    badges = {
        'shutdown': '<span class="safe-badge">🔵 Arrêté</span>',
        'startup': '<span class="warning-badge">🟡 Démarrage</span>',
        'operation': '<span class="safe-badge">🟢 En Opération</span>',
        'refueling': '<span class="warning-badge">🟠 Rechargement</span>',
        'scram': '<span class="danger-badge">🔴 SCRAM</span>',
        'maintenance': '<span class="warning-badge">🔧 Maintenance</span>'
    }
    return badges.get(status, '<span>❓</span>')

def create_reactor_mock(name, reactor_type, config):
    """Crée un réacteur simulé"""
    reactor_id = f"reactor_{len(st.session_state.nuclear_system['reactors']) + 1}"
    
    reactor = {
        'id': reactor_id,
        'name': name,
        'type': reactor_type,
        'created_at': datetime.now().isoformat(),
        'status': 'shutdown',
        'specifications': {
            'thermal_power': config.get('thermal_power', 3000),
            'electric_power': config.get('electric_power', 1000),
            'efficiency': (config.get('electric_power', 1000) / config.get('thermal_power', 3000)) * 100,
            'core_height': config.get('core_height', 3.66),
            'core_diameter': config.get('core_diameter', 3.37),
            'core_volume': 0.0
        },
        'fuel': {
            'type': config.get('fuel_type', 'UO2'),
            'enrichment': config.get('enrichment', 4.5),
            'mass': config.get('fuel_mass', 80000),
            'burnup': 0.0,
            'max_burnup': 60000
        },
        'thermal': {
            'inlet_temp': config.get('inlet_temp', 293),
            'outlet_temp': config.get('outlet_temp', 325),
            'pressure': config.get('pressure', 155),
            'flow_rate': config.get('flow_rate', 17500)
        },
        'neutronics': {
            'k_effective': 1.0,
            'neutron_flux': 0.0,
            'power_density': 100,
            'control_rod_position': 0.0
        },
        'operations': {
            'power_level': 0.0,
            'operational_hours': 0.0,
            'capacity_factor': 0.0,
            'cycles_completed': 0,
            'energy_produced': 0.0,
            'co2_avoided': 0.0
        },
        'safety': {
            'scrams': 0,
            'incidents': [],
            'ines_level': 0,
            'last_inspection': None
        },
        'economics': {
            'construction_cost': config.get('construction_cost', 5000),
            'fuel_cost_year': config.get('fuel_cost', 50),
            'operation_cost_year': config.get('operation_cost', 100),
            'decommissioning_cost': config.get('decommissioning', 1000)
        }
    }
    
    # Calcul volume cœur
    reactor['specifications']['core_volume'] = (np.pi * (reactor['specifications']['core_diameter']/2)**2 * 
                                                 reactor['specifications']['core_height'])
    
    st.session_state.nuclear_system['reactors'][reactor_id] = reactor
    log_event(f"Réacteur créé: {name} ({reactor_type})")
    return reactor_id

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">☢️ Plateforme de Réacteurs Nucléaires</h1>', unsafe_allow_html=True)
st.markdown("### Système Intégré pour Conception, Simulation et Analyse de Réacteurs Nucléaires")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/0077b6/ffffff?text=Nuclear+Engineering", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Tableau de Bord",
            "⚛️ Mes Réacteurs",
            "➕ Créer Réacteur",
            "🔬 Neutronique",
            "🌡️ Thermohydraulique",
            "⚡ Production Énergie",
            "🔋 Combustible",
            "♻️ Cycle Combustible",
            "🛡️ Systèmes Sûreté",
            "☢️ Radioprotection",
            "🗑️ Déchets Radioactifs",
            "📊 Simulations",
            "📈 Analyses",
            "🚨 Incidents & SCRAM",
            "🔧 Maintenance",
            "📋 Inspections",
            "💰 Économie",
            "🌍 Impact Environnemental",
            "📚 Réglementation",
            "🎓 Formation",
            "📖 Documentation"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques Globales")
    
    total_reactors = len(st.session_state.nuclear_system['reactors'])
    active_reactors = sum(1 for r in st.session_state.nuclear_system['reactors'].values() if r['status'] == 'operation')
    total_incidents = len(st.session_state.nuclear_system['incidents'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚛️ Réacteurs", total_reactors)
        st.metric("🚨 Incidents", total_incidents)
    with col2:
        st.metric("✅ Actifs", active_reactors)
        total_energy = sum(r['operations']['energy_produced'] for r in st.session_state.nuclear_system['reactors'].values())
        st.metric("⚡ TWh", f"{total_energy/1e6:.1f}")

# ==================== PAGE: TABLEAU DE BORD ====================
if page == "🏠 Tableau de Bord":
    st.header("📊 Tableau de Bord Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="reactor-card"><h2>⚛️</h2><h3>{total_reactors}</h3><p>Réacteurs</p></div>', unsafe_allow_html=True)
    
    with col2:
        total_power = sum(r['specifications']['electric_power'] for r in st.session_state.nuclear_system['reactors'].values())
        st.markdown(f'<div class="reactor-card"><h2>⚡</h2><h3>{total_power}</h3><p>MWe Installés</p></div>', unsafe_allow_html=True)
    
    with col3:
        total_energy = sum(r['operations']['energy_produced'] for r in st.session_state.nuclear_system['reactors'].values())
        st.markdown(f'<div class="reactor-card"><h2>🔋</h2><h3>{total_energy/1e6:.1f}</h3><p>TWh Produits</p></div>', unsafe_allow_html=True)
    
    with col4:
        total_co2 = sum(r['operations']['co2_avoided'] for r in st.session_state.nuclear_system['reactors'].values())
        st.markdown(f'<div class="reactor-card"><h2>🌱</h2><h3>{total_co2/1e6:.1f}M</h3><p>t CO₂ Évités</p></div>', unsafe_allow_html=True)
    
    with col5:
        avg_capacity = np.mean([r['operations']['capacity_factor'] for r in st.session_state.nuclear_system['reactors'].values()]) if total_reactors > 0 else 0
        st.markdown(f'<div class="reactor-card"><h2>📈</h2><h3>{avg_capacity:.1f}%</h3><p>Facteur Charge</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Constantes nucléaires
    st.subheader("⚛️ Constantes Fondamentales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre d'Avogadro", "6.022×10²³ mol⁻¹")
        st.metric("Masse neutron", "1.0087 uma")
    
    with col2:
        st.metric("Énergie fission U-235", "200 MeV")
        st.metric("ν (U-235)", "2.43 n/fission")
    
    with col3:
        st.metric("σ fission U-235", "585 barns")
        st.metric("σ fission Pu-239", "750 barns")
    
    with col4:
        st.metric("Énergie/fission", "3.2×10⁻¹¹ J")
        st.metric("Fissions/MWj", "~10²¹")
    
    st.markdown("---")
    
    if st.session_state.nuclear_system['reactors']:
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Puissance par Réacteur")
            
            names = [r['name'][:25] for r in st.session_state.nuclear_system['reactors'].values()]
            powers = [r['specifications']['electric_power'] for r in st.session_state.nuclear_system['reactors'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=powers, marker_color='rgb(0, 119, 182)',
                      text=[f"{p} MWe" for p in powers],
                      textposition='outside')
            ])
            fig.update_layout(title="Puissance Électrique", yaxis_title="MWe", xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔋 Production Énergétique")
            
            names = [r['name'][:25] for r in st.session_state.nuclear_system['reactors'].values()]
            energies = [r['operations']['energy_produced']/1e3 for r in st.session_state.nuclear_system['reactors'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=energies, marker_color='rgb(0, 180, 216)',
                      text=[f"{e:.1f} GWh" for e in energies],
                      textposition='outside')
            ])
            fig.update_layout(title="Énergie Produite", yaxis_title="GWh", xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Aucun réacteur créé. Créez votre premier réacteur nucléaire!")

# ==================== PAGE: MES RÉACTEURS ====================
elif page == "⚛️ Mes Réacteurs":
    st.header("⚛️ Gestion des Réacteurs Nucléaires")
    
    if not st.session_state.nuclear_system['reactors']:
        st.info("💡 Aucun réacteur créé.")
    else:
        for reactor_id, reactor in st.session_state.nuclear_system['reactors'].items():
            st.markdown(f'<div class="reactor-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### ☢️ {reactor['name']}")
                st.write(f"**Type:** {reactor['type'].replace('_', ' ').title()}")
                st.markdown(get_status_badge(reactor['status']), unsafe_allow_html=True)
            
            with col2:
                st.metric("Puissance Th.", f"{reactor['specifications']['thermal_power']} MWth")
                st.metric("Puissance Él.", f"{reactor['specifications']['electric_power']} MWe")
            
            with col3:
                st.metric("Rendement", f"{reactor['specifications']['efficiency']:.1f}%")
                st.metric("Niveau Puissance", f"{reactor['operations']['power_level']:.0f}%")
            
            with col4:
                st.metric("k_eff", f"{reactor['neutronics']['k_effective']:.4f}")
                st.metric("Burnup", f"{reactor['fuel']['burnup']:.0f} MWd/tU")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⚙️ Spécifications", "🔋 Combustible", "🌡️ Thermique", "⚛️ Neutronique", "📊 Opérations", "💰 Économie"])
                
                with tab1:
                    st.subheader("⚙️ Spécifications Techniques")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Puissance Thermique", f"{reactor['specifications']['thermal_power']} MWth")
                    with col2:
                        st.metric("Puissance Électrique", f"{reactor['specifications']['electric_power']} MWe")
                    with col3:
                        st.metric("Hauteur Cœur", f"{reactor['specifications']['core_height']:.2f} m")
                    with col4:
                        st.metric("Diamètre Cœur", f"{reactor['specifications']['core_diameter']:.2f} m")
                    
                    st.metric("Volume Cœur", f"{reactor['specifications']['core_volume']:.2f} m³")
                
                with tab2:
                    st.subheader("🔋 Combustible Nucléaire")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {reactor['fuel']['type']}")
                        st.metric("Enrichissement", f"{reactor['fuel']['enrichment']:.2f}%")
                        st.metric("Masse", f"{reactor['fuel']['mass']:,} kg")
                    
                    with col2:
                        st.metric("Burnup Actuel", f"{reactor['fuel']['burnup']:.0f} MWd/tU")
                        st.metric("Burnup Max", f"{reactor['fuel']['max_burnup']:,} MWd/tU")
                        
                        progress = reactor['fuel']['burnup'] / reactor['fuel']['max_burnup']
                        st.progress(progress)
                        st.write(f"Épuisement: {progress*100:.1f}%")
                
                with tab3:
                    st.subheader("🌡️ Thermohydraulique")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Température Entrée", f"{reactor['thermal']['inlet_temp']} °C")
                        st.metric("Température Sortie", f"{reactor['thermal']['outlet_temp']} °C")
                    
                    with col2:
                        st.metric("ΔT", f"{reactor['thermal']['outlet_temp'] - reactor['thermal']['inlet_temp']} °C")
                        st.metric("Pression Primaire", f"{reactor['thermal']['pressure']} bar")
                    
                    with col3:
                        st.metric("Débit", f"{reactor['thermal']['flow_rate']:,} kg/s")
                        st.metric("Puissance Extraite", f"{reactor['specifications']['thermal_power']} MWth")
                
                with tab4:
                    st.subheader("⚛️ Neutronique")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("k_effectif", f"{reactor['neutronics']['k_effective']:.5f}")
                        status_k = "✅ Critique" if abs(reactor['neutronics']['k_effective'] - 1.0) < 0.01 else "⚠️ Non-critique"
                        st.write(status_k)
                    
                    with col2:
                        st.metric("Flux Neutronique", f"{reactor['neutronics']['neutron_flux']:.2e} n/cm²/s")
                        st.metric("Densité Puissance", f"{reactor['neutronics']['power_density']} kW/L")
                    
                    with col3:
                        st.metric("Position Barres", f"{reactor['neutronics']['control_rod_position']:.1f}%")
                        st.progress(reactor['neutronics']['control_rod_position'] / 100)
                
                with tab5:
                    st.subheader("📊 Statistiques Opérationnelles")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Heures Opération", f"{reactor['operations']['operational_hours']:,.0f}h")
                        st.metric("Facteur Charge", f"{reactor['operations']['capacity_factor']:.1f}%")
                    
                    with col2:
                        st.metric("Cycles Complétés", reactor['operations']['cycles_completed'])
                        st.metric("Énergie Produite", f"{reactor['operations']['energy_produced']/1e3:.1f} GWh")
                    
                    with col3:
                        st.metric("CO₂ Évité", f"{reactor['operations']['co2_avoided']/1e3:.0f} kt")
                        st.metric("Niveau Puissance", f"{reactor['operations']['power_level']:.0f}%")
                
                with tab6:
                    st.subheader("💰 Analyse Économique")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Construction", f"€{reactor['economics']['construction_cost']:.0f}M")
                    with col2:
                        st.metric("Combustible/an", f"€{reactor['economics']['fuel_cost_year']:.0f}M")
                    with col3:
                        st.metric("Opération/an", f"€{reactor['economics']['operation_cost_year']:.0f}M")
                    
                    st.metric("Démantèlement", f"€{reactor['economics']['decommissioning_cost']:.0f}M")
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    if st.button(f"▶️ {'Arrêter' if reactor['status'] == 'operation' else 'Démarrer'}", key=f"toggle_{reactor_id}"):
                        if reactor['status'] == 'operation':
                            reactor['status'] = 'shutdown'
                            reactor['operations']['power_level'] = 0.0
                            log_event(f"{reactor['name']} arrêté")
                        else:
                            reactor['status'] = 'operation'  # ✅ CORRECTION: mettre directement 'operation'
                            reactor['operations']['power_level'] = 100.0  # ✅ Mettre à 100%
                            log_event(f"{reactor['name']} démarré")
                        st.rerun()
                
                # with col1:
                #     if st.button(f"▶️ {'Arrêter' if reactor['status'] == 'operation' else 'Démarrer'}", key=f"toggle_{reactor_id}"):
                #         if reactor['status'] == 'operation':
                #             reactor['status'] = 'shutdown'
                #             reactor['operations']['power_level'] = 0.0
                #         else:
                #             reactor['status'] = 'startup'
                #         log_event(f"{reactor['name']} {'arrêté' if reactor['status'] == 'shutdown' else 'démarrage'}")
                #         st.rerun()
                
                with col2:
                    if st.button(f"⚡ Opération", key=f"operate_{reactor_id}"):
                        st.info("Allez dans Production Énergie")
                
                with col3:
                    if st.button(f"🚨 SCRAM", key=f"scram_{reactor_id}"):
                        reactor['status'] = 'scram'
                        reactor['operations']['power_level'] = 0.0
                        reactor['safety']['scrams'] += 1
                        log_event(f"SCRAM déclenché: {reactor['name']}")
                        st.warning("⚠️ SCRAM activé!")
                        st.rerun()
                
                with col4:
                    if st.button(f"🔧 Maintenance", key=f"maint_{reactor_id}"):
                        reactor['status'] = 'maintenance'
                        st.info("Mode maintenance")
                
                with col5:
                    if st.button(f"🗑️ Supprimer", key=f"del_{reactor_id}"):
                        del st.session_state.nuclear_system['reactors'][reactor_id]
                        log_event(f"{reactor['name']} supprimé")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CRÉER RÉACTEUR ====================
elif page == "➕ Créer Réacteur":
    st.header("➕ Créer un Nouveau Réacteur Nucléaire")
    
    with st.form("create_reactor_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            reactor_name = st.text_input("📝 Nom du Réacteur", placeholder="Ex: EPR Flamanville")
            
            reactor_type = st.selectbox(
                "⚛️ Type de Réacteur",
                [
                    "reacteur_eau_pressurisee",  # PWR/REP
                    "reacteur_eau_bouillante",  # BWR/REB
                    "reacteur_eau_lourde",  # PHWR/CANDU
                    "reacteur_graphite_gaz",  # GCR
                    "reacteur_rapide_sodium",  # LMFBR
                    "reacteur_sels_fondus",  # MSR
                    "reacteur_haute_temperature",  # HTR
                    "reacteur_fusion",  # Fusion
                    "petit_reacteur_modulaire",  # SMR
                    "generation_4"  # Gen IV
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            application = st.selectbox(
                "🎯 Application Principale",
                ["Production Électricité", "Cogénération", "Recherche", 
                 "Production Isotopes", "Dessalement", "Propulsion Navale"]
            )
            
            generation = st.selectbox(
                "🕐 Génération",
                ["Gen II", "Gen III", "Gen III+", "Gen IV", "Fusion"]
            )
        
        st.markdown("---")
        st.subheader("⚡ Spécifications Puissance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            thermal_power = st.number_input("Puissance Thermique (MWth)", 10, 10000, 3000, 10)
        
        with col2:
            efficiency = st.slider("Rendement (%)", 20.0, 45.0, 33.0, 0.5)
            electric_power = int(thermal_power * efficiency / 100)
            st.metric("Puissance Électrique", f"{electric_power} MWe")
        
        with col3:
            power_density = st.number_input("Densité Puissance (kW/L)", 50, 200, 100, 5)
        
        st.markdown("---")
        st.subheader("🔬 Géométrie du Cœur")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            core_height = st.number_input("Hauteur Cœur (m)", 1.0, 10.0, 3.66, 0.1)
        
        with col2:
            core_diameter = st.number_input("Diamètre Cœur (m)", 1.0, 10.0, 3.37, 0.1)
        
        with col3:
            core_volume = np.pi * (core_diameter/2)**2 * core_height
            st.metric("Volume Cœur", f"{core_volume:.2f} m³")
        
        st.markdown("---")
        st.subheader("🔋 Combustible")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fuel_type = st.selectbox(
                "Type de Combustible",
                ["UO2", "MOX", "uranium_metallique", "thorium", "plutonium", "sel_fondu"],
                format_func=lambda x: x.replace('_', ' ').upper()
            )
            
            enrichment = st.slider("Enrichissement U-235 (%)", 0.7, 20.0, 4.5, 0.1)
        
        with col2:
            fuel_mass = st.number_input("Masse Combustible (kg)", 1000, 500000, 80000, 1000)
            
            max_burnup = st.number_input("Burnup Maximum (MWd/tU)", 10000, 100000, 60000, 1000)
        
        st.markdown("---")
        st.subheader("🌡️ Thermohydraulique")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inlet_temp = st.number_input("Température Entrée (°C)", 100, 500, 293, 1)
        
        with col2:
            outlet_temp = st.number_input("Température Sortie (°C)", 200, 600, 325, 1)
        
        with col3:
            delta_t = outlet_temp - inlet_temp
            st.metric("ΔT", f"{delta_t} °C")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pressure = st.number_input("Pression Primaire (bar)", 1, 200, 155, 1)
        
        with col2:
            flow_rate = st.number_input("Débit Caloporteur (kg/s)", 1000, 50000, 17500, 100)
        
        st.markdown("---")
        st.subheader("🛡️ Systèmes de Sûreté")
        
        n_safety_systems = st.number_input("Nombre Systèmes Sûreté", 3, 10, 5, 1)
        
        safety_systems = []
        for i in range(n_safety_systems):
            col1, col2 = st.columns(2)
            with col1:
                sys_name = st.text_input(f"Système {i+1}", f"Safety System {i+1}", key=f"safety_{i}")
            with col2:
                sys_type = st.selectbox(f"Type {i+1}", 
                    ["SCRAM", "ECCS", "Confinement", "Refroidissement Passif", "Soupapes"],
                    key=f"safety_type_{i}")
            
            if sys_name:
                safety_systems.append({'name': sys_name, 'type': sys_type})
        
        st.markdown("---")
        st.subheader("💰 Économie")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            construction_cost = st.number_input("Coût Construction (M€)", 100, 50000, 5000, 100)
        with col2:
            fuel_cost = st.number_input("Coût Combustible/an (M€)", 10, 500, 50, 5)
        with col3:
            operation_cost = st.number_input("Coût Opération/an (M€)", 10, 1000, 100, 10)
        with col4:
            decommissioning = st.number_input("Démantèlement (M€)", 100, 5000, 1000, 50)
        
        st.markdown("---")
        
        # Résumé
        st.subheader("📊 Résumé")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Puissance Th.", f"{thermal_power} MWth")
        with col2:
            st.metric("Puissance Él.", f"{electric_power} MWe")
        with col3:
            st.metric("Rendement", f"{efficiency:.1f}%")
        with col4:
            st.metric("Coût Total", f"€{construction_cost}M")
        
        submitted = st.form_submit_button("🚀 Créer le Réacteur", use_container_width=True, type="primary")
        
        if submitted:
            if not reactor_name:
                st.error("⚠️ Veuillez donner un nom au réacteur")
            else:
                with st.spinner("🔄 Création du réacteur en cours..."):
                    config = {
                        'thermal_power': thermal_power,
                        'electric_power': electric_power,
                        'core_height': core_height,
                        'core_diameter': core_diameter,
                        'fuel_type': fuel_type,
                        'enrichment': enrichment,
                        'fuel_mass': fuel_mass,
                        'inlet_temp': inlet_temp,
                        'outlet_temp': outlet_temp,
                        'pressure': pressure,
                        'flow_rate': flow_rate,
                        'construction_cost': construction_cost,
                        'fuel_cost': fuel_cost,
                        'operation_cost': operation_cost,
                        'decommissioning': decommissioning,
                        'safety_systems': safety_systems
                    }
                    
                    reactor_id = create_reactor_mock(reactor_name, reactor_type, config)
                    
                    st.success(f"✅ Réacteur '{reactor_name}' créé avec succès!")
                    st.balloons()
                    
                    reactor = st.session_state.nuclear_system['reactors'][reactor_id]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Puissance", f"{reactor['specifications']['electric_power']} MWe")
                    with col2:
                        st.metric("Rendement", f"{reactor['specifications']['efficiency']:.1f}%")
                    with col3:
                        st.metric("Volume Cœur", f"{reactor['specifications']['core_volume']:.2f} m³")
                    with col4:
                        st.metric("Systèmes Sûreté", len(safety_systems))

# ==================== PAGE: NEUTRONIQUE ====================
elif page == "🔬 Neutronique":
    st.header("🔬 Physique Neutronique")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ k-effectif", "📊 Flux Neutrons", "🎯 Sections Efficaces", "📈 Équations"])
    
    with tab1:
        st.subheader("⚛️ Calcul du Facteur de Multiplication")
        
        st.write("### 🧮 Formule des Six Facteurs")
        
        st.latex(r"k_{\infty} = \varepsilon \cdot p \cdot f \cdot \eta")
        st.latex(r"k_{eff} = \frac{k_{\infty}}{1 + L^2 B^2}")
        
        with st.form("k_effective_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Facteurs de Production:**")
                epsilon = st.number_input("ε (fission rapide)", 1.0, 1.1, 1.02, 0.01)
                eta = st.number_input("η (reproduction)", 1.5, 2.5, 2.07, 0.01)
            
            with col2:
                st.write("**Facteurs d'Absorption:**")
                p = st.number_input("p (échappement résonance)", 0.7, 1.0, 0.87, 0.01)
                f = st.number_input("f (utilisation thermique)", 0.5, 1.0, 0.71, 0.01)
            
            st.write("**Facteurs Géométriques:**")
            col1, col2 = st.columns(2)
            
            with col1:
                L_squared = st.number_input("L² (aire migration, cm²)", 100, 1000, 350, 10)
            with col2:
                B_squared = st.number_input("B² (laplacien géom., cm⁻²)", 1e-5, 1e-2, 8e-4, 1e-5, format="%.2e")
            
            if st.form_submit_button("🔬 Calculer k_eff"):
                k_infinity = epsilon * p * f * eta
                non_leakage = 1 / (1 + L_squared * B_squared)
                k_effective = k_infinity * non_leakage
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("k∞", f"{k_infinity:.5f}")
                with col2:
                    st.metric("Facteur fuite", f"{non_leakage:.5f}")
                with col3:
                    st.metric("k_eff", f"{k_effective:.5f}")
                
                # État du réacteur
                if abs(k_effective - 1.0) < 0.001:
                    st.success("✅ Réacteur CRITIQUE (k_eff ≈ 1.000)")
                elif k_effective > 1.0:
                    st.warning(f"⚠️ Réacteur SURCRITIQUE (k_eff = {k_effective:.5f})")
                else:
                    st.info(f"ℹ️ Réacteur SOUS-CRITIQUE (k_eff = {k_effective:.5f})")
        
        st.markdown("---")
        
        st.write("### 📊 Réactivité")
        
        st.latex(r"\rho = \frac{k_{eff} - 1}{k_{eff}} = \frac{\Delta k}{k}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            k_eff_input = st.number_input("k_effectif", 0.9, 1.1, 1.0, 0.001)
            
            reactivity = (k_eff_input - 1.0) / k_eff_input
            reactivity_pcm = reactivity * 1e5  # en pcm
            
            st.metric("Réactivité ρ", f"{reactivity:.6f}")
            st.metric("Réactivité", f"{reactivity_pcm:.0f} pcm")
        
        with col2:
            st.write("**Échelles de réactivité:**")
            st.write("• 1 $ (dollar) = β_eff ≈ 650 pcm")
            st.write("• 1 ¢ (cent) = β_eff/100 ≈ 6.5 pcm")
            st.write("• pcm = 10⁻⁵ Δk/k")
            
            if abs(reactivity_pcm) < 10:
                st.success("✅ Réactivité négligeable")
            elif reactivity_pcm > 650:
                st.error("⚠️ DANGER: Réactivité > 1$")
    
    with tab2:
        st.subheader("📊 Flux Neutronique")
        
        st.write("### 🌊 Distribution du Flux")
        
        col1, col2 = st.columns(2)
        
        with col1:
            reactor_power = st.number_input("Puissance Thermique (MWth)", 100, 5000, 3000, 100)
            core_volume_flux = st.number_input("Volume Cœur (m³)", 10, 500, 30, 5)
        
        with col2:
            # Calcul flux moyen
            energy_per_fission = 200 * 1.6e-13  # J
            fissions_per_second = (reactor_power * 1e6) / energy_per_fission
            
            # Approximation
            sigma_f = 585e-24  # cm²
            N_fuel = 0.024e24  # at/cm³
            
            flux_average = fissions_per_second / (sigma_f * N_fuel * core_volume_flux * 1e6)
            
            st.metric("Flux Moyen", f"{flux_average:.2e} n/cm²/s")
            st.metric("Fissions/s", f"{fissions_per_second:.2e}")
        
        st.markdown("---")
        
        # Distribution spatiale
        st.write("### 📈 Distribution Spatiale (1D)")
        
        # Simulation flux simplifié (cosinus)
        # z = np.linspace(0, core_height, 200)
        core_height_flux = 3.66  # valeur par défaut
        if st.session_state.nuclear_system['reactors']:
            # Prendre le premier réacteur pour la démo
            first_reactor = list(st.session_state.nuclear_system['reactors'].values())[0]
            core_height_flux = first_reactor['specifications']['core_height']

        z = np.linspace(0, core_height_flux, 200)
        H = core_height_flux
        B = np.pi / H
        phi_z = np.cos(B * (z - H/2))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=z, y=phi_z,
            mode='lines',
            line=dict(color='blue', width=3),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Distribution Axiale du Flux (fondamental)",
            xaxis_title="Position Axiale (m)",
            yaxis_title="Flux Normalisé φ(z)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Sections Efficaces")
        
        st.write("### 📊 Sections Efficaces Microscopiques")
        
        cross_sections = [
            {"Isotope": "U-235", "σ_fission (thermique)": "585 b", "σ_capture": "99 b", "σ_total": "684 b"},
            {"Isotope": "U-238", "σ_fission (thermique)": "~0 b", "σ_capture": "2.7 b", "σ_total": "8.3 b"},
            {"Isotope": "Pu-239", "σ_fission (thermique)": "750 b", "σ_capture": "271 b", "σ_total": "1021 b"},
            {"Isotope": "Pu-240", "σ_fission (thermique)": "0.06 b", "σ_capture": "290 b", "σ_total": "290 b"},
            {"Isotope": "H-1", "σ_scattering": "20 b", "σ_capture": "0.33 b", "σ_total": "20.3 b"},
            {"Isotope": "B-10", "σ_capture": "3840 b", "σ_total": "3840 b", "Usage": "Absorbant"}
        ]
        
        df_xs = pd.DataFrame(cross_sections)
        st.dataframe(df_xs, use_container_width=True)
        
        st.info("💡 1 barn = 10⁻²⁴ cm²")
        
        st.markdown("---")
        
        st.write("### 🎯 Calcul Section Efficace Macroscopique")
        
        st.latex(r"\Sigma = N \cdot \sigma")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sigma_micro = st.number_input("σ microscopique (barns)", 1, 10000, 585, 1)
            density = st.number_input("Densité atomique (×10²⁴ at/cm³)", 0.001, 0.1, 0.024, 0.001)
        
        with col2:
            sigma_macro = sigma_micro * 1e-24 * density * 1e24
            st.metric("Σ macroscopique", f"{sigma_macro:.4f} cm⁻¹")
            
            mfp = 1 / sigma_macro if sigma_macro > 0 else 0
            st.metric("Libre parcours moyen", f"{mfp:.2f} cm")
    
    with tab4:
        st.subheader("📈 Équations de Transport")
        
        st.write("### ⚛️ Équation de Diffusion")
        
        st.latex(r"-D\nabla^2\phi(\vec{r}) + \Sigma_a\phi(\vec{r}) = \nu\Sigma_f\phi(\vec{r})")
        
        st.write("""
        Où:
        - **D** : Coefficient de diffusion
        - **Σₐ** : Section efficace macroscopique d'absorption
        - **Σ_f** : Section efficace macroscopique de fission
        - **ν** : Nombre de neutrons par fission
        - **φ** : Flux neutronique
        """)
        
        st.markdown("---")
        
        st.write("### 🌊 Équation de Transport de Boltzmann")
        
        st.latex(r"\Omega \cdot \nabla\psi + \Sigma_t\psi = \int\Sigma_s\psi' d\Omega' + S")
        
        st.markdown("---")
        
        st.write("### ⏱️ Équation Cinétique Ponctuelle")
        
        st.latex(r"\frac{dn}{dt} = \frac{\rho - \beta}{\Lambda}n + \sum_{i}\lambda_i C_i")
        
        st.write("**Précurseurs retardés:**")
        
        precursors = [
            {"Groupe": "1", "β_i": "0.000215", "λ_i (s⁻¹)": "0.0127", "T_1/2": "55 s"},
            {"Groupe": "2", "β_i": "0.001424", "λ_i (s⁻¹)": "0.0317", "T_1/2": "22 s"},
            {"Groupe": "3", "β_i": "0.001274", "λ_i (s⁻¹)": "0.115", "T_1/2": "6 s"},
            {"Groupe": "4", "β_i": "0.002568", "λ_i (s⁻¹)": "0.311", "T_1/2": "2.2 s"},
            {"Groupe": "5", "β_i": "0.000748", "λ_i (s⁻¹)": "1.40", "T_1/2": "0.5 s"},
            {"Groupe": "6", "β_i": "0.000273", "λ_i (s⁻¹)": "3.87", "T_1/2": "0.18 s"}
        ]
        
        df_prec = pd.DataFrame(precursors)
        st.dataframe(df_prec, use_container_width=True)
        
        beta_total = sum(float(p['β_i']) for p in precursors)
        st.metric("β_total (U-235)", f"{beta_total:.5f} = {beta_total*1e5:.0f} pcm")

# ==================== PAGE: THERMOHYDRAULIQUE ====================
elif page == "🌡️ Thermohydraulique":
    st.header("🌡️ Thermohydraulique du Réacteur")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Transfert Thermique", "💧 Écoulement", "⚠️ DNBR", "🌡️ Températures"])
    
    with tab1:
        st.subheader("🔥 Transfert de Chaleur")
        
        st.write("### 🔬 Calcul Flux Thermique")
        
        with st.form("heat_transfer"):
            col1, col2 = st.columns(2)
            
            with col1:
                power_linear = st.number_input("Puissance Linéaire (kW/m)", 1, 50, 20, 1)
                # rod_diameter = st.number_input("Diamètre Crayon (mm)", 5, 15, 9.5, 0.1)
                rod_diameter = st.number_input("Diamètre Crayon (mm)", 5.0, 15.0, 9.5, 0.1)
            
            with col2:
                coolant_temp = st.number_input("T caloporteur (°C)", 200, 350, 300, 5)
                h_coeff = st.number_input("h (W/m²K)", 10000, 100000, 50000, 1000)
            
                if st.form_submit_button("🔬 Calculer"):
                    # Surface externe
                    surface = np.pi * (rod_diameter/1000) * 1  # m² par mètre
                    
                    # Flux thermique
                    q_flux = (power_linear * 1000) / surface  # W/m²
                    
                    # Température surface gaine
                    T_surface = coolant_temp + (q_flux / h_coeff)
                    
                    # Température centre combustible (conductivité UO2 ~ 3 W/mK)
                    k_fuel = 3.0
                    r_pellet = (rod_diameter * 0.8) / 2000  # m (80% du diamètre)
                    T_center = T_surface + (q_flux * r_pellet) / (4 * k_fuel)
                    
                    st.success("✅ Calcul terminé!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Flux Thermique", f"{q_flux/1e6:.2f} MW/m²")
                    with col2:
                        st.metric("T surface", f"{T_surface:.1f} °C")
                    with col3:
                        st.metric("T centre", f"{T_center:.1f} °C")
                    
                    # Alerte température
                    if T_center > 2800:
                        st.error("⚠️ DANGER: Température > limite UO₂ !")
                    elif T_surface > 350:
                        st.warning("⚠️ Température surface élevée")
                    else:
                        st.success("✅ Températures dans les limites")
            
            st.markdown("---")
            
            st.write("### 📊 Profil de Température Radial")
            
            # Simulation profil température
            r = np.linspace(0, 5, 100)  # mm
            T_clad = 320  # °C
            T_fuel_center = 1200  # °C
            
            # Profil parabolique dans le combustible
            T_profile = np.where(r < 4, 
                                T_fuel_center - (T_fuel_center - T_clad) * (r/4)**2,
                                T_clad)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=r, y=T_profile,
                mode='lines',
                line=dict(color='red', width=3),
                fill='tozeroy'
            ))
            
            fig.add_vline(x=4, line_dash="dash", annotation_text="Gaine")
            
            fig.update_layout(
                title="Profil de Température Radial",
                xaxis_title="Rayon (mm)",
                yaxis_title="Température (°C)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("💧 Écoulement du Caloporteur")
            
            st.write("### 🌊 Paramètres Hydrauliques")
            
            col1, col2 = st.columns(2)
            
            with col1:
                flow_velocity = st.slider("Vitesse écoulement (m/s)", 1.0, 10.0, 5.0, 0.1)
                hydraulic_diameter = st.number_input("Diamètre hydraulique (mm)", 5.0, 20.0, 11.7, 0.1)
            
            with col2:
                # Propriétés eau à 300°C, 155 bar
                density = 720  # kg/m³
                viscosity = 9e-5  # Pa·s
                
                # Nombre de Reynolds
                Re = (density * flow_velocity * (hydraulic_diameter/1000)) / viscosity
                
                st.metric("Densité", f"{density} kg/m³")
                st.metric("Reynolds", f"{Re:.0f}")
                
                if Re < 2300:
                    regime = "Laminaire"
                elif Re < 4000:
                    regime = "Transitoire"
                else:
                    regime = "Turbulent"
                
                st.write(f"**Régime:** {regime}")
            
            st.markdown("---")
            
            st.write("### 📊 Pertes de Charge")
            
            col1, col2 = st.columns(2)
            
            with col1:
                length = st.number_input("Longueur canal (m)", 1.0, 10.0, 3.66, 0.1)
                roughness = st.number_input("Rugosité (μm)", 0.1, 100.0, 10.0, 0.1)
            
            with col2:
                # Coefficient de frottement (Colebrook simplifié)
                if Re > 4000:
                    f = 0.316 / (Re ** 0.25)  # Blasius
                else:
                    f = 64 / Re
                
                # Perte de charge
                dp = f * (length / (hydraulic_diameter/1000)) * (density * flow_velocity**2 / 2)
                
                st.metric("Coeff. frottement", f"{f:.4f}")
                st.metric("Perte de charge", f"{dp/1e5:.2f} bar")
        
        with tab3:
            st.subheader("⚠️ DNBR - Departure from Nucleate Boiling Ratio")
            
            st.info("""
            **DNBR** = Flux Thermique Critique / Flux Thermique Réel
            
            - DNBR > 1.3 : Sûr (critère de conception)
            - DNBR < 1.3 : Risque ébullition en film
            - DNBR < 1.0 : DANGER - Crise d'ébullition
            """)
            
            with st.form("dnbr_calc"):
                col1, col2 = st.columns(2)
                
                with col1:
                    q_actual = st.number_input("Flux thermique réel (MW/m²)", 0.1, 3.0, 0.8, 0.1)
                    pressure_dnbr = st.slider("Pression (bar)", 50, 200, 155, 5)
                
                with col2:
                    mass_flux = st.number_input("Flux massique (kg/m²s)", 1000, 5000, 3000, 100)
                    quality = st.slider("Titre vapeur", 0.0, 1.0, 0.0, 0.01)
                
                if st.form_submit_button("🔬 Calculer DNBR"):
                    # Corrélation W-3 simplifiée
                    CHF = (2.022 - 0.0004302 * pressure_dnbr) * (1 - 0.1 * quality)
                    
                    DNBR = CHF / q_actual
                    
                    st.success("✅ Calcul DNBR terminé!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("CHF", f"{CHF:.2f} MW/m²")
                    with col2:
                        st.metric("DNBR", f"{DNBR:.2f}")
                    with col3:
                        if DNBR >= 1.3:
                            st.success("✅ DNBR OK")
                        else:
                            st.error("⚠️ DNBR < 1.3")
                    
                    # Graphique marge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=DNBR,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [None, 3]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 1.3], 'color': "red"},
                                {'range': [1.3, 2], 'color': "yellow"},
                                {'range': [2, 3], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 1.3
                            }
                        }
                    ))
                    
                    fig.update_layout(title="DNBR Margin", height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🌡️ Températures Opérationnelles")
            
            st.write("### 📊 Limites de Température")
            
            temp_limits = [
                {"Composant": "Combustible UO₂ (centre)", "T max": "2800°C", "T fusion": "3120°C"},
                {"Composant": "Gaine Zircaloy", "T max": "1200°C", "T fusion": "1850°C"},
                {"Composant": "Caloporteur (sortie)", "T max": "350°C", "T sat": "345°C @ 155 bar"},
                {"Composant": "Structures internes", "T max": "400°C", "Matériau": "Acier inox"},
            ]
            
            df_temps = pd.DataFrame(temp_limits)
            st.dataframe(df_temps, use_container_width=True)
            
            st.markdown("---")
            
            st.write("### 📈 Évolution Températures en Régime")
            
            # Simulation montée en puissance
            time = np.linspace(0, 24, 100)  # heures
            power_ramp = np.minimum(time / 20 * 100, 100)  # % puissance
            
            T_fuel = 600 + 6 * power_ramp
            T_clad = 300 + 0.5 * power_ramp
            T_coolant_out = 293 + 0.32 * power_ramp
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=time, y=T_fuel, mode='lines', name='Combustible (centre)',
                                    line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=time, y=T_clad, mode='lines', name='Gaine',
                                    line=dict(color='orange', width=3)))
            fig.add_trace(go.Scatter(x=time, y=T_coolant_out, mode='lines', name='Caloporteur (sortie)',
                                    line=dict(color='blue', width=3)))
            
            fig.update_layout(
                title="Évolution Températures - Montée en Puissance",
                xaxis_title="Temps (heures)",
                yaxis_title="Température (°C)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: IMPACT ENVIRONNEMENTAL ====================
elif page == "🌍 Impact Environnemental":
    st.header("🌍 Impact Environnemental")
    
    tab1, tab2, tab3 = st.tabs(["🌱 Émissions CO₂", "💧 Eau", "🌡️ Climat"])
    
    with tab1:
        st.subheader("🌱 Bilan Carbone")
        
        st.write("### 📊 Émissions sur Cycle de Vie")
        
        emissions_data = [
            {"Source": "Charbon", "gCO₂/kWh": "820-1000", "Couleur": "gray"},
            {"Source": "Gaz CCGT", "gCO₂/kWh": "410-490", "Couleur": "orange"},
            {"Source": "Solaire PV", "gCO₂/kWh": "40-50", "Couleur": "yellow"},
            {"Source": "Éolien", "gCO₂/kWh": "10-15", "Couleur": "green"},
            {"Source": "Nucléaire", "gCO₂/kWh": "6-12", "Couleur": "blue"},
            {"Source": "Hydraulique", "gCO₂/kWh": "4-10", "Couleur": "cyan"}
        ]
        
        df_emissions = pd.DataFrame(emissions_data)
        st.dataframe(df_emissions, use_container_width=True)
        
        st.markdown("---")
        
        # Comparaison visuelle
        sources = [e['Source'] for e in emissions_data]
        emissions_mid = [910, 450, 45, 12.5, 9, 7]
        colors = [e['Couleur'] for e in emissions_data]
        
        fig = go.Figure(data=[
            go.Bar(x=sources, y=emissions_mid, marker_color=colors,
                  text=[f"{v} g" for v in emissions_mid],
                  textposition='outside')
        ])
        
        fig.update_layout(
            title="Émissions CO₂ par Source (gCO₂/kWh)",
            yaxis_title="gCO₂/kWh",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🌍 CO₂ Évité par le Nucléaire")
        
        if st.session_state.nuclear_system['reactors']:
            total_energy = sum(r['operations']['energy_produced'] for r in st.session_state.nuclear_system['reactors'].values())
            total_co2 = sum(r['operations']['co2_avoided'] for r in st.session_state.nuclear_system['reactors'].values())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Énergie Produite", f"{total_energy/1e6:.2f} TWh")
            with col2:
                st.metric("CO₂ Évité", f"{total_co2/1e6:.2f} Mt")
            with col3:
                cars_equivalent = (total_co2 / 1e6) / 4.6 * 1e6  # 4.6 t/voiture/an
                st.metric("Équivalent Voitures", f"{cars_equivalent:,.0f}")
    
    with tab2:
        st.subheader("💧 Consommation d'Eau")
        
        st.write("### 🌊 Prélèvements et Rejets")
        
        water_data = {
            "Circuit ouvert (rivière/mer)": {
                "prélèvement": "~50 m³/MWh",
                "consommation": "~1 m³/MWh",
                "rejet": "~49 m³/MWh (+10°C)"
            },
            "Circuit fermé (tours aéro)": {
                "prélèvement": "~2 m³/MWh",
                "consommation": "~2 m³/MWh",
                "rejet": "Évaporation"
            }
        }
        
        for system, data in water_data.items():
            with st.expander(f"💧 {system}"):
                for key, value in data.items():
                    st.write(f"**{key.title()}:** {value}")
        
        st.markdown("---")
        
        st.write("### 📊 Comparaison Sources Énergie")
        
        water_consumption = {
            'Nucléaire (circuit fermé)': 2.0,
            'Charbon': 2.0,
            'Gaz': 0.8,
            'Solaire thermique': 3.0,
            'Éolien': 0.01,
            'Solaire PV': 0.03
        }
        
        fig = go.Figure(data=[
            go.Bar(x=list(water_consumption.keys()),
                  y=list(water_consumption.values()),
                  marker_color='lightblue',
                  text=[f"{v:.2f}" for v in water_consumption.values()],
                  textposition='outside')
        ])
        
        fig.update_layout(
            title="Consommation Eau (m³/MWh)",
            yaxis_title="m³/MWh",
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🌡️ Impact Climatique")
        
        st.write("### 🌍 Contribution à l'Atténuation")
        
        st.info("""
        **Rôle du Nucléaire:**
        
        ✅ Source bas-carbone (6-12 gCO₂/kWh)
        ✅ Production stable (baseload)
        ✅ Densité énergétique élevée
        ✅ Emprise au sol faible
        ✅ Compatible avec EnR
        """)
        
        # Scénarios GIEC
        st.write("### 📊 Scénarios GIEC 1.5°C")
        
        scenarios = [
            {"Scénario": "P1 - EnR dominantes", "Part Nucléaire 2050": "3-7%"},
            {"Scénario": "P2 - Mix équilibré", "Part Nucléaire 2050": "8-15%"},
            {"Scénario": "P3 - Diversifié", "Part Nucléaire 2050": "15-25%"},
            {"Scénario": "P4 - Nucléaire renforcé", "Part Nucléaire 2050": "25-35%"}
        ]
        
        df_scenarios = pd.DataFrame(scenarios)
        st.dataframe(df_scenarios, use_container_width=True)
        
        st.write("""
        💡 **Note GIEC:** Tous les scénarios 1.5°C incluent le nucléaire dans le mix énergétique
        """)

# ==================== PAGE: RÉGLEMENTATION ====================
elif page == "📚 Réglementation":
    st.header("📚 Réglementation Nucléaire")
    
    tab1, tab2, tab3 = st.tabs(["🏛️ Autorités", "📜 Textes", "🔍 Autorisations"])
    
    with tab1:
        st.subheader("🏛️ Autorités de Sûreté")
        
        authorities = {
            "🇫🇷 France - ASN": {
                "nom": "Autorité de Sûreté Nucléaire",
                "rôle": "Contrôle sûreté et radioprotection",
                "appui": "IRSN (Institut Radioprotection)",
                "indépendance": "AAI depuis 2006"
            },
            "🇺🇸 USA - NRC": {
                "nom": "Nuclear Regulatory Commission",
                "rôle": "Réglementation et contrôle",
                "appui": "Laboratoires DOE",
                "indépendance": "Agence fédérale indépendante"
            },
            "🌍 AIEA": {
                "nom": "Agence Internationale Énergie Atomique",
                "rôle": "Normes internationales, coopération",
                "appui": "169 États membres",
                "indépendance": "Organisation ONU"
            },
            "🇪🇺 ENSREG": {
                "nom": "European Nuclear Safety Regulators Group",
                "rôle": "Harmonisation Europe",
                "appui": "Autorités nationales",
                "indépendance": "Groupe UE"
            }
        }
        
        for auth_name, auth_info in authorities.items():
            with st.expander(f"🏛️ {auth_name}"):
                for key, value in auth_info.items():
                    st.write(f"**{key.title()}:** {value}")
    
    with tab2:
        st.subheader("📜 Textes Réglementaires")
        
        st.write("### 🇫🇷 France")
        
        french_texts = [
            "📕 Code de l'Environnement (Livre V)",
            "📕 Code de la Santé Publique (radioprotection)",
            "📄 Loi TSN (Transparence Sûreté Nucléaire) 2006",
            "📄 Arrêté ministériel INB",
            "📄 Décisions ASN",
            "📄 Guides ASN",
            "🇪🇺 Directive européenne 2009/71 (sûreté)",
            "🇪🇺 Directive 2013/59 (radioprotection)"
        ]
        
        for text in french_texts:
            st.write(text)
        
        st.markdown("---")
        
        st.write("### 🌍 Conventions Internationales")
        
        conventions = [
            "Convention Sûreté Nucléaire (1994)",
            "Convention Gestion Déchets (1997)",
            "Convention Notification Rapide (1986)",
            "Convention Assistance (1986)",
            "Convention Responsabilité Civile (Paris, Vienne)"
        ]
        
        for conv in conventions:
            st.write(f"• {conv}")
    
    with tab3:
        st.subheader("🔍 Procédures d'Autorisation")
        
        st.write("### 📋 Étapes Autorisation INB")
        
        authorization_steps = [
            {"Étape": "1. Demande Autorisation Création (DAC)", "Durée": "~5 ans", "Contenu": "Dossier sûreté préliminaire"},
            {"Étape": "2. Enquête Publique", "Durée": "2 mois", "Contenu": "Consultation citoyens"},
            {"Étape": "3. Avis ASN", "Durée": "6-12 mois", "Contenu": "Instruction technique"},
            {"Étape": "4. Décret Autorisation Création", "Durée": "Variable", "Contenu": "Décision Gouvernement"},
            {"Étape": "5. Construction", "Durée": "5-10 ans", "Contenu": "Suivant autorisation"},
            {"Étape": "6. Demande Autorisation Mise en Service", "Durée": "~2 ans", "Contenu": "Dossier complet"},
            {"Étape": "7. Essais", "Durée": "1-2 ans", "Contenu": "Démonstration sûreté"},
            {"Étape": "8. Autorisation Mise en Service", "Durée": "Variable", "Contenu": "Décision ASN"}
        ]
        
        df_auth = pd.DataFrame(authorization_steps)
        st.dataframe(df_auth, use_container_width=True)

# ==================== PAGE: FORMATION ====================
elif page == "🎓 Formation":
    st.header("🎓 Formation et Éducation")
    
    tab1, tab2 = st.tabs(["📚 Cursus", "🏫 Établissements"])
    
    with tab1:
        st.subheader("📚 Parcours de Formation")
        
        st.write("### 🎓 Formations Diplômantes")
        
        formations = {
            "Niveau Bac+5 - Ingénieur": {
                "écoles": "INSTN, Mines, Centrale, INSA",
                "spécialités": "Génie atomique, Neutronique, Thermohydraulique",
                "durée": "3 ans post-prépa"
            },
            "Master - Nucléaire": {
                "universités": "Paris-Saclay, Grenoble, Nantes",
                "spécialités": "Physique nucléaire, Radioprotection, Démantèlement",
                "durée": "2 ans"
            },
            "Doctorat": {
                "laboratoires": "CEA, CNRS, Universités",
                "domaines": "Recherche fondamentale et appliquée",
                "durée": "3 ans"
            },
            "Formation Continue": {
                "organismes": "INSTN, CNAM",
                "publics": "Professionnels en activité",
                "formats": "Stages, certificats, VAE"
            }
        }
        
        for form_name, form_info in formations.items():
            with st.expander(f"🎓 {form_name}"):
                for key, value in form_info.items():
                    st.write(f"**{key.title()}:** {value}")
    
    with tab2:
        st.subheader("🏫 Établissements")
        
        st.write("### 🇫🇷 France")
        
        establishments = [
            "🏫 INSTN (Institut National Sciences Techniques Nucléaires) - CEA",
            "🏫 École des Mines",
            "🏫 Centrale Paris/Lyon",
            "🏫 INSA Lyon",
            "🏫 Université Paris-Saclay",
            "🏫 Grenoble INP - Phelma",
            "🏫 IMT Atlantique"
        ]
        
        for estab in establishments:
            st.write(estab)


# ==================== PAGE: DOCUMENTATION ====================
elif page == "📖 Documentation":
    st.header("📖 Documentation Technique")
    
    tab1, tab2 = st.tabs(["📚 Ressources", "🔗 Liens"])
    
    with tab1:
        st.subheader("📚 Ressources Documentaires")
        
        resources = {
            "📕 Normes et Standards": [
                "AIEA Safety Standards",
                "IEEE Nuclear Standards",
                "ASME Boiler & Pressure Vessel Code",
                "RCC-M (Règles Conception Construction)",
                "Guides ASN"
            ],
            "📘 Ouvrages de Référence": [
                "Lamarsh - Nuclear Reactor Theory",
                "Duderstadt & Hamilton - Nuclear Reactor Analysis",
                "Todreas & Kazimi - Nuclear Systems",
                "Glasstone & Sesonske - Nuclear Reactor Engineering"
            ],
            "📄 Revues Scientifiques": [
                "Nuclear Engineering and Design",
                "Annals of Nuclear Energy",
                "Nuclear Technology",
                "Progress in Nuclear Energy"
            ]
        }
        
        for cat, items in resources.items():
            with st.expander(cat):
                for item in items:
                    st.write(f"• {item}")
    
    with tab2:
        st.subheader("🔗 Liens Utiles")
        
        links = [
            "🌐 AIEA - www.iaea.org",
            "🇫🇷 ASN - www.asn.fr",
            "🇫🇷 IRSN - www.irsn.fr",
            "🇫🇷 CEA - www.cea.fr",
            "🇺🇸 NRC - www.nrc.gov",
            "🌍 World Nuclear Association - world-nuclear.org",
            "📊 NEA-OCDE - www.oecd-nea.org"
        ]
        
        for link in links:
            st.write(link)

# ==================== PAGE: ANALYSES ====================
elif page == "📈 Analyses":
    st.header("📈 Analyses de Données")
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔍 Tendances", "📉 Benchmarking"])
    
    with tab1:
        st.subheader("📊 Analyse de Performance")
        
        if st.session_state.nuclear_system['reactors']:
            # KPIs principaux
            st.write("### 🎯 Indicateurs Clés de Performance")
            
            kpis = []
            for reactor in st.session_state.nuclear_system['reactors'].values():
                kpis.append({
                    'Réacteur': reactor['name'][:25],
                    'Facteur Charge (%)': f"{reactor['operations']['capacity_factor']:.1f}",
                    'Disponibilité (%)': f"{np.random.uniform(85, 95):.1f}",
                    'INES': reactor['safety']['ines_level'],
                    'Scrams': reactor['safety']['scrams'],
                    'MWh/kg U': f"{reactor['operations']['energy_produced']/reactor['fuel']['mass']:.2f}"
                })
            
            df_kpis = pd.DataFrame(kpis)
            st.dataframe(df_kpis, use_container_width=True)
            
            st.markdown("---")
            
            # Analyse comparative
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Facteur de Charge")
                
                names = [r['name'][:20] for r in st.session_state.nuclear_system['reactors'].values()]
                cf_values = [r['operations']['capacity_factor'] for r in st.session_state.nuclear_system['reactors'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=names, y=cf_values, marker_color='lightblue',
                          text=[f"{v:.1f}%" for v in cf_values],
                          textposition='outside')
                ])
                
                fig.add_hline(y=90, line_dash="dash", line_color="green",
                             annotation_text="Objectif 90%")
                
                fig.update_layout(
                    yaxis_title="Facteur Charge (%)",
                    xaxis_tickangle=-45,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### 🔥 Burnup Combustible")
                
                burnup_values = [r['fuel']['burnup'] for r in st.session_state.nuclear_system['reactors'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=names, y=burnup_values, marker_color='orange',
                          text=[f"{v:.0f}" for v in burnup_values],
                          textposition='outside')
                ])
                
                fig.update_layout(
                    yaxis_title="Burnup (MWd/tU)",
                    xaxis_tickangle=-45,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun réacteur disponible")
    
    with tab2:
        st.subheader("🔍 Analyse de Tendances")
        
        st.write("### 📈 Tendances Secteur Nucléaire")
        
        # Données mondiales simulées
        years = np.arange(2000, 2025)
        
        # Capacité installée mondiale
        capacity = 350 + (years - 2000) * 2.5 + np.random.randn(len(years)) * 10
        
        # Production électrique
        production = 2500 + (years - 2000) * 30 + np.random.randn(len(years)) * 50
        
        # Part dans mix énergétique
        nuclear_share = 16 - (years - 2000) * 0.15 + np.random.randn(len(years)) * 0.5
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Capacité Installée", "Production Électrique", 
                          "Part dans Mix Électrique", "Réacteurs par Type")
        )
        
        fig.add_trace(go.Scatter(x=years, y=capacity, mode='lines+markers',
                                name='Capacité (GWe)', line=dict(width=3)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=years, y=production, mode='lines+markers',
                                name='Production (TWh)', line=dict(width=3)), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=years, y=nuclear_share, mode='lines+markers',
                                name='Part (%)', line=dict(width=3)), row=2, col=1)
        
        # Répartition par type
        reactor_types = ['PWR', 'BWR', 'PHWR', 'GCR', 'LMFBR', 'Autres']
        counts = [300, 80, 50, 15, 5, 10]
        
        fig.add_trace(go.Bar(x=reactor_types, y=counts, name='Nombre',
                            marker_color='lightgreen'), row=2, col=2)
        
        fig.update_xaxes(title_text="Année")
        fig.update_layout(height=700, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📉 Benchmarking International")
        
        st.write("### 🌍 Comparaison Pays")
        
        country_data = [
            {"Pays": "🇺🇸 USA", "Réacteurs": 93, "Capacité (GWe)": 95, "Part (%)": 19, "Facteur Charge": 92},
            {"Pays": "🇫🇷 France", "Réacteurs": 56, "Capacité (GWe)": 61, "Part (%)": 70, "Facteur Charge": 71},
            {"Pays": "🇨🇳 Chine", "Réacteurs": 55, "Capacité (GWe)": 53, "Part (%)": 5, "Facteur Charge": 91},
            {"Pays": "🇯🇵 Japon", "Réacteurs": 33, "Capacité (GWe)": 32, "Part (%)": 7, "Facteur Charge": 45},
            {"Pays": "🇷🇺 Russie", "Réacteurs": 38, "Capacité (GWe)": 29, "Part (%)": 20, "Facteur Charge": 82},
            {"Pays": "🇰🇷 Corée", "Réacteurs": 26, "Capacité (GWe)": 25, "Part (%)": 29, "Facteur Charge": 88},
            {"Pays": "🇨🇦 Canada", "Réacteurs": 19, "Capacité (GWe)": 13, "Part (%)": 15, "Facteur Charge": 82},
            {"Pays": "🇬🇧 UK", "Réacteurs": 9, "Capacité (GWe)": 6, "Part (%)": 16, "Facteur Charge": 68}
        ]
        
        df_countries = pd.DataFrame(country_data)
        st.dataframe(df_countries, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_countries, x='Pays', y='Réacteurs',
                        title="Nombre de Réacteurs",
                        color='Réacteurs', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_countries, x='Pays', y='Part (%)',
                        title="Part Nucléaire dans Mix Électrique",
                        color='Part (%)', color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: INCIDENTS & SCRAM ====================
elif page == "🚨 Incidents & SCRAM":
    st.header("🚨 Incidents et Arrêts d'Urgence")
    
    tab1, tab2, tab3 = st.tabs(["📋 Historique", "⚠️ Échelle INES", "📊 Analyse"])
    
    with tab1:
        st.subheader("📋 Historique des Incidents")
        
        if st.session_state.nuclear_system['incidents']:
            for incident in st.session_state.nuclear_system['incidents'][-10:][::-1]:
                with st.expander(f"🚨 {incident['type']} - {incident['timestamp'][:10]}"):
                    st.write(f"**Type:** {incident['type']}")
                    st.write(f"**Date:** {incident['timestamp']}")
                    st.write(f"**Description:** {incident.get('description', 'N/A')}")
                    st.write(f"**Niveau INES:** {incident.get('ines_level', 0)}")
        else:
            st.success("✅ Aucun incident enregistré")
        
        st.markdown("---")
        
        # Ajouter incident test
        if st.button("➕ Ajouter Incident Test"):
            incident = {
                'timestamp': datetime.now().isoformat(),
                'type': 'Test SCRAM',
                'description': 'Test procédure arrêt urgence',
                'ines_level': 0
            }
            st.session_state.nuclear_system['incidents'].append(incident)
            st.rerun()
    
    with tab2:
        st.subheader("⚠️ Échelle INES")
        
        st.info("""
        **INES: International Nuclear Event Scale**
        
        Échelle de classification des événements nucléaires (0-7)
        """)
        
        ines_levels = [
            {"Niveau": "0", "Classification": "Écart", "Impact": "Aucun", "Exemple": "Événement sans importance sûreté"},
            {"Niveau": "1", "Classification": "Anomalie", "Impact": "Aucun", "Exemple": "Dépassement limites opérationnelles"},
            {"Niveau": "2", "Classification": "Incident", "Impact": "Aucun", "Exemple": "Défaillance équipements sûreté"},
            {"Niveau": "3", "Classification": "Incident grave", "Impact": "Aucun/Mineur", "Exemple": "Contamination, exposition"},
            {"Niveau": "4", "Classification": "Accident local", "Impact": "Local", "Exemple": "Saint-Laurent (1980)"},
            {"Niveau": "5", "Classification": "Accident étendu", "Impact": "Étendu", "Exemple": "Three Mile Island (1979)"},
            {"Niveau": "6", "Classification": "Accident grave", "Impact": "Important", "Exemple": "Kychtym (1957)"},
            {"Niveau": "7", "Classification": "Accident majeur", "Impact": "Majeur", "Exemple": "Tchernobyl (1986), Fukushima (2011)"}
        ]
        
        df_ines = pd.DataFrame(ines_levels)
        st.dataframe(df_ines, use_container_width=True)
        
        st.markdown("---")
        
        # Visualisation
        levels = [0, 1, 2, 3, 4, 5, 6, 7]
        colors = ['lightgreen', 'lightgreen', 'yellow', 'yellow', 'orange', 'orange', 'red', 'darkred']
        
        fig = go.Figure(data=[
            go.Bar(x=levels, y=[1]*8, marker_color=colors,
                  text=['Écart', 'Anomalie', 'Incident', 'Incident grave',
                        'Accident local', 'Accident étendu', 'Accident grave', 'Accident majeur'],
                  textposition='inside')
        ])
        
        fig.update_layout(
            title="Échelle INES",
            xaxis_title="Niveau",
            yaxis_visible=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Analyse Statistique")
        
        st.write("### 📈 Fréquence des Événements")
        
        # Statistiques mondiales
        event_stats = [
            {"Type": "Niveau 0-1", "Fréquence": "~1000 / an", "Impact": "Négligeable"},
            {"Type": "Niveau 2", "Fréquence": "~50 / an", "Impact": "Mineur"},
            {"Type": "Niveau 3", "Fréquence": "~5 / an", "Impact": "Faible"},
            {"Type": "Niveau 4+", "Fréquence": "< 1 / 10 ans", "Impact": "Significatif"}
        ]
        
        df_stats = pd.DataFrame(event_stats)
        st.dataframe(df_stats, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🎯 Taux SCRAM")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Cibles Industrie:**")
            st.write("• Automatiques: < 1 / an / réacteur")
            st.write("• Manuels: < 0.5 / an / réacteur")
            st.write("• Total: < 1.5 / an / réacteur")
        
        with col2:
            st.write("**Performance Mondiale (2023):**")
            st.write("• Moyenne: 0.8 SCRAM / réacteur / an")
            st.write("• Meilleurs: 0.1 / an (Corée)")
            st.write("• Amélioration continue")


# ==================== PAGE: MAINTENANCE ====================
elif page == "🔧 Maintenance":
    st.header("🔧 Maintenance et Inspections")
    
    tab1, tab2 = st.tabs(["📅 Planning", "🔍 Activités"])
    
    with tab1:
        st.subheader("📅 Planning de Maintenance")
        
        st.write("### 🗓️ Types de Maintenance")
        
        maintenance_types = {
            "Maintenance Préventive": {
                "fréquence": "Quotidienne à mensuelle",
                "activités": ["Rondes", "Contrôles", "Lubrification", "Ajustements"],
                "arrêt": "Non"
            },
            "Arrêt pour Rechargement": {
                "fréquence": "12-24 mois",
                "activités": ["Rechargement 1/3 ou 1/4 cœur", "Inspections", "Maintenance"],
                "arrêt": "Oui (4-8 semaines)"
            },
            "Visite Partielle (VP)": {
                "fréquence": "Tous les 4-6 ans",
                "activités": ["Inspections réglementaires", "Essais périodiques", "Modifications"],
                "arrêt": "Oui (6-10 semaines)"
            },
            "Visite Complète (VC)": {
                "fréquence": "Tous les 10 ans",
                "activités": ["Réexamen sûreté", "Inspections approfondies", "Remplacements"],
                "arrêt": "Oui (12-20 semaines)"
            }
        }
        
        for maint_name, maint_info in maintenance_types.items():
            with st.expander(f"🔧 {maint_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Fréquence:** {maint_info['fréquence']}")
                    st.write(f"**Arrêt:** {maint_info['arrêt']}")
                
                with col2:
                    st.write("**Activités:**")
                    for act in maint_info['activités']:
                        st.write(f"• {act}")
    
    with tab2:
        st.subheader("🔍 Activités de Maintenance")
        
        st.write("### 📋 Checklist Arrêt de Tranche")
        
        checklist = [
            {"Phase": "Préparation", "Activité": "Planification détaillée", "Durée": "Semaines avant", "✓": True},
            {"Phase": "Préparation", "Activité": "Commande pièces/combustible", "Durée": "Mois avant", "✓": True},
            {"Phase": "Descente puissance", "Activité": "Réduction progressive", "Durée": "24-48h", "✓": False},
            {"Phase": "Arrêt", "Activité": "SCRAM et refroidissement", "Durée": "1 semaine", "✓": False},
            {"Phase": "Ouverture cuve", "Activité": "Retrait couvercle", "Durée": "3-5 jours", "✓": False},
            {"Phase": "Rechargement", "Activité": "Manutention assemblages", "Durée": "1-2 semaines", "✓": False},
            {"Phase": "Maintenance", "Activité": "Inspections/réparations", "Durée": "2-4 semaines", "✓": False},
            {"Phase": "Fermeture", "Activité": "Remontage", "Durée": "1 semaine", "✓": False},
            {"Phase": "Essais", "Activité": "Tests redémarrage", "Durée": "1 semaine", "✓": False},
            {"Phase": "Montée puissance", "Activité": "Criticité → 100%", "Durée": "2-3 jours", "✓": False}
        ]
        
        df_checklist = pd.DataFrame(checklist)
        st.dataframe(df_checklist, use_container_width=True)

# ==================== PAGE: INSPECTIONS ====================
elif page == "📋 Inspections":
    st.header("📋 Inspections Réglementaires")
    
    st.write("### 🔍 Contrôles Réglementaires")
    
    inspections = [
        {"Type": "Essais Périodiques", "Fréquence": "Mensuelle/Trimestrielle", "Autorité": "Exploitant"},
        {"Type": "Inspections ASN", "Fréquence": "~20-30 / an / site", "Autorité": "ASN"},
        {"Type": "Réexamen Sûreté", "Fréquence": "Tous les 10 ans", "Autorité": "ASN + IRSN"},
        {"Type": "Contrôles Indépendants", "Fréquence": "Selon programme", "Autorité": "Organismes agréés"}
    ]
    
    df_insp = pd.DataFrame(inspections)
    st.dataframe(df_insp, use_container_width=True)

# ==================== PAGE: ÉCONOMIE ====================
elif page == "💰 Économie":
    st.header("💰 Aspects Économiques")
    
    tab1, tab2, tab3 = st.tabs(["💵 Coûts", "📊 LCOE", "📈 Rentabilité"])
    
    with tab1:
        st.subheader("💵 Structure des Coûts")
        
        st.write("### 🏗️ Répartition Coûts (EPR type)")
        
        cost_breakdown = {
            "Investissement initial": 12000,  # M€
            "Intérêts intercalaires": 2000,
            "Exploitation (60 ans)": 6000,
            "Combustible (60 ans)": 3000,
            "Démantèlement": 1000,
            "Gestion déchets": 500
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=list(cost_breakdown.values()),
                        names=list(cost_breakdown.keys()),
                        title="Répartition Coûts Totaux")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            total_cost = sum(cost_breakdown.values())
            st.metric("Coût Total", f"€{total_cost:,}M")
            st.metric("Dont CAPEX", f"€{cost_breakdown['Investissement initial']:,}M")
            st.metric("Dont OPEX", f"€{cost_breakdown['Exploitation (60 ans)']:,}M")
    
    with tab2:
        st.subheader("📊 LCOE (Levelized Cost of Energy)")
        
        st.info("""
        **LCOE:** Coût actualisé de l'énergie sur la durée de vie
        
        LCOE = (CAPEX + ∑ OPEX actualisé) / ∑ Production actualisée
        """)
        
        with st.form("lcoe_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                capex = st.number_input("CAPEX (M€)", 1000, 20000, 12000, 100)
                power = st.number_input("Puissance (MWe)", 100, 2000, 1650, 50)
                lifetime = st.number_input("Durée vie (ans)", 40, 80, 60, 5)
            
            with col2:
                opex_annual = st.number_input("OPEX annuel (M€)", 50, 500, 100, 10)
                fuel_annual = st.number_input("Combustible annuel (M€)", 20, 200, 50, 5)
                capacity_factor_lcoe = st.slider("Facteur charge (%)", 50, 95, 85, 1)
                discount_rate = st.slider("Taux actualisation (%)", 3.0, 10.0, 5.0, 0.5)
            
            if st.form_submit_button("🔬 Calculer LCOE"):
                # Production annuelle
                annual_production = power * 8760 * (capacity_factor_lcoe / 100) / 1000  # TWh
                
                # Calcul actualisé
                discount_factor = (1 + discount_rate/100)
                
                total_capex = capex
                total_opex = 0
                total_production = 0
                
                for year in range(1, lifetime + 1):
                    opex_year = (opex_annual + fuel_annual) / (discount_factor ** year)
                    prod_year = annual_production / (discount_factor ** year)
                    
                    total_opex += opex_year
                    total_production += prod_year
                
                # LCOE
                lcoe = (total_capex + total_opex) / total_production  # M€/TWh = €/MWh
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("LCOE", f"{lcoe:.1f} €/MWh")
                with col2:
                    st.metric("Production totale", f"{total_production:.1f} TWh")
                with col3:
                    st.metric("Coût total actualisé", f"€{total_capex + total_opex:,.0f}M")
                
                # Comparaison sources
                st.markdown("---")
                st.write("### 📊 Comparaison LCOE par Source")
                
                sources = ['Nucléaire', 'Éolien terrestre', 'Solaire PV', 'Gaz CCGT', 'Charbon']
                lcoe_values = [lcoe, 50, 45, 80, 90]
                
                fig = go.Figure(data=[
                    go.Bar(x=sources, y=lcoe_values,
                          marker_color=['blue', 'green', 'yellow', 'orange', 'gray'],
                          text=[f"{v:.0f} €/MWh" for v in lcoe_values],
                          textposition='outside')
                ])
                
                fig.update_layout(
                    title="LCOE Comparatif",
                    yaxis_title="LCOE (€/MWh)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Analyse de Rentabilité")
        
        st.write("### 💰 Flux de Trésorerie")
        
        # Simulation flux
        years_flow = np.arange(0, 61)
        
        # Construction: années 0-7
        construction_flow = np.where(years_flow < 7, -12000/7, 0)
        
        # Exploitation: années 7-60
        revenue = np.where(years_flow >= 7, 1650 * 8760 * 0.85 * 60 / 1000, 0)  # M€
        opex_flow = np.where(years_flow >= 7, -150, 0)
        
        net_flow = construction_flow + revenue + opex_flow
        cumulative_flow = np.cumsum(net_flow)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Flux Annuels", "Flux Cumulés")
        )
        
        fig.add_trace(go.Bar(x=years_flow, y=construction_flow, name='CAPEX',
                            marker_color='red'), row=1, col=1)
        fig.add_trace(go.Bar(x=years_flow, y=revenue, name='Revenus',
                            marker_color='green'), row=1, col=1)
        fig.add_trace(go.Bar(x=years_flow, y=opex_flow, name='OPEX',
                            marker_color='orange'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=years_flow, y=cumulative_flow, name='Cumulé',
                                line=dict(color='blue', width=3)), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", row=2, col=1)
        
        fig.update_xaxes(title_text="Année")
        fig.update_yaxes(title_text="Flux (M€)")
        fig.update_layout(height=700, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Retour sur investissement
        breakeven_year = np.where(cumulative_flow > 0)[0]
        if len(breakeven_year) > 0:
            st.success(f"✅ Retour sur investissement: Année {breakeven_year[0]}")
        else:
            st.warning("⚠️ Pas de retour sur investissement sur la période")

# ==================== PAGE: COMBUSTIBLE ====================
elif page == "🔋 Combustible":
    st.header("🔋 Combustible Nucléaire")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ Types", "📊 Composition", "🔥 Burnup", "📈 Évolution"])
    
    with tab1:
        st.subheader("⚛️ Types de Combustible")
        
        fuel_types = {
            "UO₂ (Dioxyde d'Uranium)": {
                "composition": "UO₂",
                "enrichissement": "3-5% U-235",
                "usage": "REP, REB, CANDU",
                "avantages": ["Stable", "Technologie mature", "Disponible"],
                "inconvénients": ["Enrichissement nécessaire", "Burnup limité"]
            },
            "MOX (Mixed Oxide)": {
                "composition": "(U,Pu)O₂",
                "enrichissement": "5-10% Pu fissile",
                "usage": "REP",
                "avantages": ["Recyclage Pu", "Valorisation"],
                "inconvénients": ["Plus cher", "Neutrons retardés"]
            },
            "Uranium Métallique": {
                "composition": "U métal",
                "enrichissement": "Variable",
                "usage": "Réacteurs rapides, recherche",
                "avantages": ["Haute densité", "Conductivité"],
                "inconvénients": ["Gonflement", "Corrosion"]
            },
            "Thorium": {
                "composition": "ThO₂",
                "enrichissement": "Fertile (Th-232)",
                "usage": "Réacteurs Gen IV",
                "avantages": ["Abondant", "Moins déchets", "U-233"],
                "inconvénients": ["Pas de fission directe", "Technologie"]
            }
        }
        
        for fuel_name, fuel_info in fuel_types.items():
            with st.expander(f"🔋 {fuel_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Composition:** {fuel_info['composition']}")
                    st.write(f"**Enrichissement:** {fuel_info['enrichissement']}")
                    st.write(f"**Usage:** {fuel_info['usage']}")
                
                with col2:
                    st.write("**Avantages:**")
                    for av in fuel_info['avantages']:
                        st.write(f"✓ {av}")
                    
                    st.write("**Inconvénients:**")
                    for inc in fuel_info['inconvénients']:
                        st.write(f"✗ {inc}")
    
    with tab2:
        st.subheader("📊 Composition Isotopique")
        
        st.write("### ⚛️ Combustible Neuf vs Usé")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Combustible Neuf (UO₂ 4.5%)**")
            
            fresh_fuel = {
                'U-235': 4.5,
                'U-238': 95.5,
                'Pu-239': 0.0,
                'Produits Fission': 0.0
            }
            
            fig = px.pie(values=list(fresh_fuel.values()), names=list(fresh_fuel.keys()),
                        title="Composition Neuf")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Combustible Usé (45 GWd/tU)**")
            
            spent_fuel = {
                'U-235': 0.8,
                'U-238': 93.4,
                'Pu total': 1.0,
                'Actinides mineurs': 0.1,
                'Produits Fission': 4.7
            }
            
            fig = px.pie(values=list(spent_fuel.values()), names=list(spent_fuel.keys()),
                        title="Composition Usé")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔥 Burnup du Combustible")
        
        st.info("""
        **Burnup (Taux de Combustion):** Mesure de l'énergie extraite du combustible
        
        - Unité: MWd/tU (Mégawatt-jour par tonne d'Uranium)
        - REP typique: 45,000 - 60,000 MWd/tU
        - Limite: Dégradation matériaux, gonflement, relâchement gaz fission
        """)
        
        st.write("### 📊 Calcul Burnup")
        
        with st.form("burnup_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                power_thermal_bu = st.number_input("Puissance thermique (MWth)", 100, 5000, 3000, 100)
                fuel_mass_bu = st.number_input("Masse combustible (tU)", 10, 200, 80, 5)
            
            with col2:
                operation_days = st.number_input("Durée opération (jours)", 1, 2000, 540, 10)
                capacity_factor_bu = st.slider("Facteur charge (%)", 50, 100, 90, 1)
            
            if st.form_submit_button("🔬 Calculer Burnup"):
                # Burnup = (Puissance × Temps × Facteur) / Masse
                burnup = (power_thermal_bu * operation_days * capacity_factor_bu / 100) / fuel_mass_bu
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Burnup", f"{burnup:,.0f} MWd/tU")
                with col2:
                    st.metric("Énergie totale", f"{power_thermal_bu * operation_days * capacity_factor_bu / 100:,.0f} MWd")
                with col3:
                    pct_burnup = (burnup / 60000) * 100
                    st.metric("% Burnup max", f"{pct_burnup:.1f}%")
                
                # Graphique progression
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=burnup,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Burnup (MWd/tU)"},
                    delta={'reference': 60000},
                    gauge={
                        'axis': {'range': [None, 70000]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 45000], 'color': "lightgreen"},
                            {'range': [45000, 60000], 'color': "yellow"},
                            {'range': [60000, 70000], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60000
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📈 Évolution Isotopique")
        
        st.write("### 📊 Évolution des Isotopes avec le Burnup")
        
        # Simulation évolution
        burnup_values = np.linspace(0, 60000, 100)
        
        # Fractions isotopiques approximatives
        u235_frac = 4.5 * np.exp(-burnup_values / 50000)
        pu239_frac = 0.7 * (1 - np.exp(-burnup_values / 30000))
        fp_frac = 5 * (1 - np.exp(-burnup_values / 40000))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=burnup_values, y=u235_frac, mode='lines',
                                name='U-235', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=burnup_values, y=pu239_frac, mode='lines',
                                name='Pu-239', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=burnup_values, y=fp_frac, mode='lines',
                                name='Prod. Fission', line=dict(color='green', width=3)))
        
        fig.update_layout(
            title="Évolution Composition Isotopique",
            xaxis_title="Burnup (MWd/tU)",
            yaxis_title="Fraction (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE: PRODUCTION ÉNERGIE ====================
elif page == "⚡ Production Énergie":
    st.header("⚡ Production d'Énergie")
    
    tab1, tab2, tab3 = st.tabs(["🔋 Opération", "📊 Performance", "📈 Historique"])
    
    with tab1:
        st.subheader("🔋 Opération du Réacteur")
        
        if not st.session_state.nuclear_system['reactors']:
            st.warning("Aucun réacteur disponible")
        else:
            reactor_ids = list(st.session_state.nuclear_system['reactors'].keys())
            selected_reactor = st.selectbox(
                "Sélectionner Réacteur",
                reactor_ids,
                format_func=lambda x: st.session_state.nuclear_system['reactors'][x]['name']
            )
            
            reactor = st.session_state.nuclear_system['reactors'][selected_reactor]
            
            st.write(f"### ☢️ {reactor['name']}")
            st.markdown(get_status_badge(reactor['status']), unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Puissance Th.", f"{reactor['specifications']['thermal_power']} MWth")
            with col2:
                st.metric("Puissance Él.", f"{reactor['specifications']['electric_power']} MWe")
            with col3:
                st.metric("k_eff", f"{reactor['neutronics']['k_effective']:.4f}")
            with col4:
                st.metric("Burnup", f"{reactor['fuel']['burnup']:.0f} MWd/tU")
            
            st.markdown("---")
            
            with st.form("operate_reactor"):
                col1, col2 = st.columns(2)
                
                with col1:
                    target_power = st.slider("Niveau Puissance Cible (%)", 0, 100, 
                                            int(reactor['operations']['power_level']))
                    duration_days = st.number_input("Durée Opération (jours)", 1, 365, 30, 1)
                
                with col2:
                    rod_adjustment = st.slider("Ajustement Barres (%)", -50, 50, 0, 1)
                    xenon_mode = st.checkbox("Mode compensation Xénon", value=True)
                
                if st.form_submit_button("▶️ Lancer Production", type="primary"):
                    if reactor['status'] not in ['operation', 'startup']:
                        st.warning("⚠️ Réacteur doit être en opération")
                    else:
                        with st.spinner("⚡ Production en cours..."):
                            progress_bar = st.progress(0)
                            
                            # Simulation
                            reactor['operations']['power_level'] = target_power
                            reactor['neutronics']['control_rod_position'] += rod_adjustment
                            reactor['neutronics']['control_rod_position'] = np.clip(
                                reactor['neutronics']['control_rod_position'], 0, 100)
                            
                            # Production énergie
                            energy_produced = (reactor['specifications']['electric_power'] * 
                                             target_power / 100 * duration_days * 24)  # MWh
                            
                            reactor['operations']['energy_produced'] += energy_produced
                            reactor['operations']['operational_hours'] += duration_days * 24
                            
                            # Burnup
                            burnup_increment = (reactor['specifications']['thermal_power'] * 
                                              target_power / 100 * duration_days / 
                                              reactor['fuel']['mass'] * 1000)
                            reactor['fuel']['burnup'] += burnup_increment
                            
                            # CO2 évité
                            co2_avoided = energy_produced * 1.0  # tonnes (vs charbon)
                            reactor['operations']['co2_avoided'] += co2_avoided
                            
                            # Facteur de charge
                            reactor['operations']['capacity_factor'] = (
                                reactor['operations']['energy_produced'] / 
                                (reactor['specifications']['electric_power'] * 
                                 reactor['operations']['operational_hours'])
                            ) * 100 if reactor['operations']['operational_hours'] > 0 else 0
                            
                            for i in range(100):
                                progress_bar.progress(i + 1)
                            
                            progress_bar.empty()
                            
                            st.success(f"✅ Production terminée!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Énergie Produite", f"{energy_produced/1e3:.1f} GWh")
                            with col2:
                                st.metric("Burnup Δ", f"{burnup_increment:.1f} MWd/tU")
                            with col3:
                                st.metric("CO₂ Évité", f"{co2_avoided/1e3:.1f} kt")
                            with col4:
                                st.metric("Facteur Charge", f"{reactor['operations']['capacity_factor']:.1f}%")
                            
                            log_event(f"Production: {reactor['name']} - {energy_produced/1e3:.1f} GWh")
                            
                            # Rechargement nécessaire?
                            if reactor['fuel']['burnup'] > reactor['fuel']['max_burnup'] * 0.9:
                                st.warning("⚠️ Rechargement combustible bientôt nécessaire!")
    
    with tab2:
        st.subheader("📊 Performance Énergétique")
        
        if st.session_state.nuclear_system['reactors']:
            # Tableau performance
            perf_data = []
            for r in st.session_state.nuclear_system['reactors'].values():
                perf_data.append({
                    'Réacteur': r['name'][:30],
                    'Puissance (MWe)': r['specifications']['electric_power'],
                    'Facteur Charge (%)': f"{r['operations']['capacity_factor']:.1f}",
                    'Énergie (GWh)': f"{r['operations']['energy_produced']/1e3:.1f}",
                    'CO₂ Évité (kt)': f"{r['operations']['co2_avoided']/1e3:.1f}",
                    'Heures': f"{r['operations']['operational_hours']:,.0f}"
                })
            
            df_perf = pd.DataFrame(perf_data)
            st.dataframe(df_perf, use_container_width=True)
        else:
            st.info("Aucun réacteur")
    
    with tab3:
        st.subheader("📈 Historique Production")
        
        if st.session_state.nuclear_system['reactors']:
            # Simulation historique
            months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                     'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
            
            production_monthly = np.random.uniform(80, 95, 12)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=months, y=production_monthly,
                marker_color='lightblue',
                text=[f"{p:.1f}%" for p in production_monthly],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Facteur de Charge Mensuel",
                yaxis_title="Facteur de Charge (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée")

# ==================== PAGE: CYCLE COMBUSTIBLE (COMPLÈTE) ====================
elif page == "♻️ Cycle Combustible":
    st.header("♻️ Cycle du Combustible Nucléaire")
    
    tab1, tab2, tab3 = st.tabs(["🔄 Cycle Complet", "⚙️ Amont", "🗑️ Aval"])
    
    with tab1:
        st.subheader("🔄 Cycle du Combustible")
        
        st.write("### 📊 Vue d'Ensemble")
        
        st.info("""
        **Deux Stratégies Principales:**
        
        1. **Cycle Ouvert** (Once-Through):
           - Extraction → Conversion → Enrichissement → Fabrication
           - Utilisation en réacteur
           - Stockage direct déchets
        
        2. **Cycle Fermé** (Recyclage):
           - Même amont
           - Utilisation → Retraitement
           - Recyclage Pu en MOX
           - Stockage déchets ultimes
        """)
        
        # Diagramme flux
        st.write("### 🔁 Flux Matières (pour 1 GWe·an)")
        
        cycle_data = [
            {"Étape": "1. Extraction", "Quantité": "174 tonnes U naturel (minerai)"},
            {"Étape": "2. Conversion", "Quantité": "200 tonnes UF6 naturel"},
            {"Étape": "3. Enrichissement", "Quantité": "30 tonnes UF6 enrichi (4%)"},
            {"Étape": "4. Fabrication", "Quantité": "27 tonnes combustible UO2"},
            {"Étape": "5. Réacteur", "Quantité": "27 tonnes chargées/an"},
            {"Étape": "6. Déchargement", "Quantité": "27 tonnes usées/an"},
            {"Étape": "7. Retraitement*", "Quantité": "25.5 t U + 0.27 t Pu récupérés"},
            {"Étape": "8. Déchets finaux", "Quantité": "1.3 tonnes (HA-VL)"}
        ]
        
        df_cycle = pd.DataFrame(cycle_data)
        st.dataframe(df_cycle, use_container_width=True)

    with tab2:
        st.subheader("⚡ Amont")
        
        st.write("### 📊 Types de Transitoires")
        
        transient_types = st.selectbox(
            "Sélectionner Transitoire",
            ["Montée en puissance", "Insertion réactivité", "Variation débit", 
             "Variation température", "Retrait barre contrôle"]
        )
        
        if st.button("🚀 Simuler Transitoire"):
            time_transient = np.linspace(0, 100, 500)
            
            if transient_types == "Montée en puissance":
                power = 20 + 80 * (1 - np.exp(-time_transient / 30))
                temp_fuel = 600 + 600 * (1 - np.exp(-time_transient / 35))
                temp_coolant = 293 + 32 * (1 - np.exp(-time_transient / 30))
            
            elif transient_types == "Insertion réactivité":
                # Insertion +100 pcm à t=10s
                rho = np.where(time_transient < 10, 0, 100)
                power = np.where(time_transient < 10, 100, 100 * np.exp(0.05 * (time_transient - 10)))
                temp_fuel = 1200 + 200 * np.where(time_transient < 10, 0, (time_transient - 10) / 50)
                temp_coolant = 325 + 10 * np.where(time_transient < 10, 0, (time_transient - 10) / 50)
            
            else:
                power = 100 + 5 * np.sin(time_transient / 10)
                temp_fuel = 1200 + 50 * np.sin(time_transient / 10)
                temp_coolant = 325 + 2 * np.sin(time_transient / 10)
            
            # Graphiques
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Puissance", "Temp. Combustible", "Temp. Caloporteur", "Vue d'ensemble")
            )
            
            fig.add_trace(go.Scatter(x=time_transient, y=power, name="Puissance (%)",
                                    line=dict(color='green', width=3)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=time_transient, y=temp_fuel, name="T fuel (°C)",
                                    line=dict(color='red', width=3)), row=1, col=2)
            
            fig.add_trace(go.Scatter(x=time_transient, y=temp_coolant, name="T coolant (°C)",
                                    line=dict(color='blue', width=3)), row=2, col=1)
            
            # Vue ensemble
            fig.add_trace(go.Scatter(x=time_transient, y=power/100, name="Puissance (norm.)",
                                    line=dict(color='green')), row=2, col=2)
            fig.add_trace(go.Scatter(x=time_transient, y=temp_fuel/1500, name="T fuel (norm.)",
                                    line=dict(color='red')), row=2, col=2)
            
            fig.update_xaxes(title_text="Temps (s)")
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔥 Aval")
        
        st.write("### 🚨 Simulation LOCA (Loss of Coolant Accident)")
        
        accident_severity = st.selectbox(
            "Sévérité",
            ["Petite brèche", "Brèche moyenne", "Grosse brèche", "Rupture guillotine"]
        )
        
        if st.button("⚠️ Simuler Accident"):
            time_accident = np.linspace(0, 300, 1000)
            
            # Paramètres selon sévérité
            severity_params = {
                "Petite brèche": {"rate": 0.05, "eccs_time": 30},
                "Brèche moyenne": {"rate": 0.15, "eccs_time": 15},
                "Grosse brèche": {"rate": 0.35, "eccs_time": 5},
                "Rupture guillotine": {"rate": 0.6, "eccs_time": 2}
            }
            
            params = severity_params[accident_severity]
            
            # Pression primaire
            pressure = 155 * np.exp(-params['rate'] * time_accident / 100)
            pressure = np.maximum(pressure, 10)
            
            # Niveau eau cœur
            level = 100 * np.exp(-params['rate'] * time_accident / 80)
            # ECCS injection
            eccs_injection = np.where(time_accident > params['eccs_time'],
                                     100 * (1 - np.exp(-(time_accident - params['eccs_time']) / 50)),
                                     0)
            level = np.minimum(level + eccs_injection, 100)
            
            # Température combustible
            temp_fuel_acc = 1200 + 800 * np.exp(-level / 50) * (1 - np.exp(-time_accident / 30))
            temp_fuel_acc = np.where(time_accident > params['eccs_time'] + 50,
                                     temp_fuel_acc * np.exp(-(time_accident - params['eccs_time'] - 50) / 100),
                                     temp_fuel_acc)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Pression Primaire", "Niveau Eau Cœur", "Temp. Combustible", "ECCS Injection")
            )
            
            fig.add_trace(go.Scatter(x=time_accident, y=pressure, name="Pression (bar)",
                                    line=dict(color='blue', width=3)), row=1, col=1)
            fig.add_hline(y=40, line_dash="dash", line_color="red", row=1, col=1,
                         annotation_text="Seuil accumulateurs")
            
            fig.add_trace(go.Scatter(x=time_accident, y=level, name="Niveau (%)",
                                    line=dict(color='cyan', width=3)), row=1, col=2)
            fig.add_hline(y=100, line_dash="dash", row=1, col=2)
            
            fig.add_trace(go.Scatter(x=time_accident, y=temp_fuel_acc, name="T fuel (°C)",
                                    line=dict(color='red', width=3)), row=2, col=1)
            fig.add_hline(y=1200, line_dash="dash", line_color="orange", row=2, col=1,
                         annotation_text="T nominal")
            fig.add_hline(y=2800, line_dash="dash", line_color="red", row=2, col=1,
                         annotation_text="Limite UO2")
            
            fig.add_trace(go.Scatter(x=time_accident, y=eccs_injection, name="Injection ECCS",
                                    line=dict(color='green', width=3), fill='tozeroy'), row=2, col=2)
            
            fig.update_xaxes(title_text="Temps (s)")
            fig.update_layout(height=700, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Évaluation
            max_temp = np.max(temp_fuel_acc)
            if max_temp < 1200:
                st.success("✅ Température maintenue - Pas de dommage combustible")
            elif max_temp < 2800:
                st.warning("⚠️ Température élevée - Surveillance requise")
            else:
                st.error("🚨 DANGER - Risque fusion combustible!")
    

# ==================== PAGE: SYSTÈMES SÛRETÉ ====================
elif page == "🛡️ Systèmes Sûreté":
    st.header("🛡️ Systèmes de Sûreté")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Défense Profondeur", "🚨 SCRAM", "❄️ Refroidissement", "📊 Barrières"])
    
    with tab1:
        st.subheader("🛡️ Défense en Profondeur")
        
        st.write("### 📊 Les 5 Niveaux")
        
        levels = [
            {
                "Niveau": "1 - Prévention",
                "Objectif": "Éviter incidents",
                "Mesures": "Conception robuste, Qualité fabrication, Contrôles",
                "Exemple": "Redondance systèmes, Marges conception"
            },
            {
                "Niveau": "2 - Surveillance",
                "Objectif": "Détecter anomalies",
                "Mesures": "Instrumentation, Alarmes, Procédures",
                "Exemple": "1000+ capteurs, Salle de contrôle"
            },
            {
                "Niveau": "3 - Systèmes sauvegarde",
                "Objectif": "Maîtriser incidents",
                "Mesures": "SCRAM, ECCS, Alimentation secours",
                "Exemple": "Insertion barres < 2s, Diesels"
            },
            {
                "Niveau": "4 - Accidents graves",
                "Objectif": "Limiter rejets",
                "Mesures": "Récupérateur corium, Filtres, Enceinte",
                "Exemple": "Core catcher, Recombinaison H₂"
            },
            {
                "Niveau": "5 - Conséquences",
                "Objectif": "Protéger population",
                "Mesures": "PPI, Évacuation, Distribution iode",
                "Exemple": "Plans 5-10-20 km"
            }
        ]
        
        df_defense = pd.DataFrame(levels)
        st.dataframe(df_defense, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🔒 Concept des 3 Barrières")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**1️⃣ Gaine Combustible**")
            st.write("• Matériau: Zircaloy")
            st.write("• Épaisseur: 0.6 mm")
            st.write("• Fonction: Confinement PF")
            st.write("• Intégrité: 99.9%")
        
        with col2:
            st.markdown("**2️⃣ Circuit Primaire**")
            st.write("• Matériau: Acier inox")
            st.write("• Épaisseur: 20 cm")
            st.write("• Pression: 155 bar")
            st.write("• Fonction: 2ème barrière")
        
        with col3:
            st.markdown("**3️⃣ Enceinte Confinement**")
            st.write("• Matériau: Béton + liner")
            st.write("• Épaisseur: 1.2 m")
            st.write("• Résistance: 5 bar")
            st.write("• Fonction: Confinement ultime")
    
    with tab2:
        st.subheader("🚨 Système d'Arrêt d'Urgence (SCRAM)")
        
        st.info("""
        **SCRAM (Safety Control Rod Axe Man):**
        Insertion rapide des barres de contrôle pour arrêt d'urgence
        
        **Objectif:** Rendre réacteur sous-critique en < 2 secondes
        """)
        
        st.write("### ⚡ Déclencheurs SCRAM")
        
        scram_triggers = [
            {"Paramètre": "Puissance thermique", "Seuil": "> 118% Pnom", "Temps": "< 0.5 s"},
            {"Paramètre": "Niveau eau pressuriseur", "Seuil": "Bas/Haut", "Temps": "< 1 s"},
            {"Paramètre": "Pression primaire", "Seuil": "< 130 ou > 165 bar", "Temps": "< 1 s"},
            {"Paramètre": "Température sortie cœur", "Seuil": "> 350°C", "Temps": "< 1 s"},
            {"Paramètre": "Flux neutronique", "Seuil": "Croissance rapide", "Temps": "< 0.1 s"},
            {"Paramètre": "Séisme", "Seuil": "> 0.1 g", "Temps": "Immédiat"},
            {"Paramètre": "Manuel", "Seuil": "Opérateur", "Temps": "< 0.5 s"}
        ]
        
        df_scram = pd.DataFrame(scram_triggers)
        st.dataframe(df_scram, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Simulation Insertion Barres")
        
        time_scram = np.linspace(0, 5, 100)
        rod_position = 100 * (1 - np.exp(-time_scram / 0.5))
        k_eff_scram = 1.0 - 0.3 * (rod_position / 100)
        power_scram = 100 * np.exp(-time_scram / 0.8)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Position Barres & k_eff", "Puissance Réacteur")
        )
        
        fig.add_trace(go.Scatter(x=time_scram, y=rod_position, name="Position Barres (%)",
                                line=dict(color='blue', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time_scram, y=k_eff_scram, name="k_eff",
                                line=dict(color='red', width=3)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=time_scram, y=power_scram, name="Puissance (%)",
                                line=dict(color='green', width=3)), row=2, col=1)
        
        fig.update_xaxes(title_text="Temps (s)")
        fig.update_yaxes(title_text="Position/k_eff", row=1, col=1)
        fig.update_yaxes(title_text="Puissance (%)", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("❄️ Refroidissement d'Urgence (ECCS)")
        
        st.write("### 💧 Système ECCS (Emergency Core Cooling System)")
        
        eccs_systems = {
            "RIS (Injection Sécurité)": {
                "fonction": "Injection eau borée haute pression",
                "capacité": "3 pompes × 150 m³/h",
                "pression": "165 bar",
                "activation": "Pression < 130 bar"
            },
            "Accumulateurs": {
                "fonction": "Injection passive azote pressurisé",
                "capacité": "4 × 30 m³",
                "pression": "45 bar",
                "activation": "Pression < 40 bar"
            },
            "RRA (Recirculation)": {
                "fonction": "Recirculation eau puisard",
                "capacité": "2 pompes × 1000 m³/h",
                "pression": "10 bar",
                "activation": "Long terme"
            },
            "Aspersion Enceinte": {
                "fonction": "Refroidissement enceinte",
                "capacité": "2 pompes × 900 m³/h",
                "fonction2": "Condensation vapeur"
            }
        }
        
        for sys_name, sys_info in eccs_systems.items():
            with st.expander(f"💧 {sys_name}"):
                for key, value in sys_info.items():
                    st.write(f"**{key.title()}:** {value}")
        
        st.markdown("---")
        
        st.write("### 📊 Séquence LOCA (Loss of Coolant Accident)")
        
        loca_sequence = [
            {"Temps": "t = 0s", "Événement": "Rupture tuyauterie", "Action": "Détection pression"},
            {"Temps": "t < 1s", "Événement": "SCRAM automatique", "Action": "Insertion barres"},
            {"Temps": "t < 10s", "Événement": "Injection RIS", "Action": "3 pompes démarrent"},
            {"Temps": "t < 30s", "Événement": "Injection accumulateurs", "Action": "Décharge passive"},
            {"Temps": "t < 300s", "Événement": "Basculement RRA", "Action": "Recirculation puisard"},
            {"Temps": "Long terme", "Événement": "Refroidissement", "Action": "Maintien < 100°C"}
        ]
        
        df_loca = pd.DataFrame(loca_sequence)
        st.dataframe(df_loca, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Intégrité des Barrières")
        
        st.write("### 🔒 État des Barrières")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Barrière 1: Gaine**")
            integrity_1 = 99.9
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=integrity_1,
                title={'text': "Intégrité (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 95], 'color': "red"},
                        {'range': [95, 99], 'color': "yellow"},
                        {'range': [99, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key="temperature_chart")
        
        with col2:
            st.markdown("**Barrière 2: Primaire**")
            integrity_2 = 100.0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=integrity_2,
                title={'text': "Intégrité (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 95], 'color': "red"},
                        {'range': [95, 99], 'color': "yellow"},
                        {'range': [99, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key="pressure_chart")
        
        with col3:
            st.markdown("**Barrière 3: Enceinte**")
            integrity_3 = 100.0
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=integrity_3,
                title={'text': "Intégrité (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 95], 'color': "red"},
                        {'range': [95, 99], 'color': "yellow"},
                        {'range': [99, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True, key="flux_chart")

# ==================== PAGE: RADIOPROTECTION ====================
elif page == "☢️ Radioprotection":
    st.header("☢️ Radioprotection et Dosimétrie")
    
    tab1, tab2, tab3 = st.tabs(["📏 Dosimétrie", "🛡️ Blindage", "⚠️ Limites"])
    
    with tab1:
        st.subheader("📏 Calcul de Doses")
        
        st.write("### 🔬 Unités Radiologiques")
        
        units_info = [
            {"Grandeur": "Activité", "Unité SI": "Becquerel (Bq)", "Ancienne": "Curie (Ci)", "Conversion": "1 Ci = 3.7×10¹⁰ Bq"},
            {"Grandeur": "Dose absorbée", "Unité SI": "Gray (Gy)", "Ancienne": "rad", "Conversion": "1 Gy = 100 rad"},
            {"Grandeur": "Dose équivalente", "Unité SI": "Sievert (Sv)", "Ancienne": "rem", "Conversion": "1 Sv = 100 rem"},
        ]
        
        df_units = pd.DataFrame(units_info)
        st.dataframe(df_units, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Calculateur de Dose")
        
        with st.form("dose_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                activity = st.number_input("Activité source (MBq)", 1.0, 100000.0, 1000.0, 10.0)
                distance = st.number_input("Distance (m)", 0.1, 100.0, 1.0, 0.1)
                exposure_time = st.number_input("Temps exposition (heures)", 0.1, 100.0, 1.0, 0.1)
            
            with col2:
                shielding_present = st.checkbox("Blindage présent", value=False)
                
                if shielding_present:
                    shield_material = st.selectbox("Matériau blindage", ["Plomb", "Béton", "Eau", "Acier"])
                    shield_thickness = st.number_input("Épaisseur (cm)", 1.0, 100.0, 10.0, 1.0)
            
            submitted = st.form_submit_button("🔬 Calculer Dose")
            
            if submitted:
                mu_values = {"Plomb": 1.2, "Béton": 0.2, "Eau": 0.08, "Acier": 0.6}
                
                if shielding_present:
                    mu = mu_values[shield_material]
                    attenuation = np.exp(-mu * shield_thickness)
                else:
                    attenuation = 1.0
                
                dose_rate = (activity * 0.01) / (distance ** 2) * attenuation
                total_dose = dose_rate * exposure_time
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Débit de dose", f"{dose_rate:.3f} mSv/h")
                with col2:
                    st.metric("Dose totale", f"{total_dose:.3f} mSv")
                with col3:
                    if shielding_present:
                        st.metric("Atténuation", f"{attenuation:.4f}")
                
                if total_dose < 1:
                    st.success("✅ Dose faible - acceptable")
                elif total_dose < 20:
                    st.warning("⚠️ Dose modérée - surveillance requise")
                else:
                    st.error("🚨 Dose élevée - DANGER!")
    
    with tab2:
        st.subheader("🛡️ Calcul de Blindage")
        
        st.write("### 📊 Épaisseur Nécessaire")
        
        with st.form("shielding_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                initial_dose = st.number_input("Débit dose initial (mSv/h)", 1.0, 10000.0, 100.0, 10.0)
                target_dose = st.number_input("Débit dose cible (mSv/h)", 0.001, 10.0, 0.1, 0.01)
            
            with col2:
                shield_mat = st.selectbox("Matériau", ["Plomb", "Béton", "Acier", "Eau"])
                mu_dict = {"Plomb": 1.2, "Béton": 0.2, "Acier": 0.6, "Eau": 0.08}
                mu = mu_dict[shield_mat]
                st.metric("Coeff. atténuation", f"{mu} cm⁻¹")
            
            submitted2 = st.form_submit_button("🔬 Calculer Épaisseur")
            
            if submitted2:
                thickness = np.log(initial_dose / target_dose) / mu
                hvl = np.log(2) / mu
                
                st.success("✅ Calcul terminé!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Épaisseur requise", f"{thickness:.1f} cm")
                with col2:
                    st.metric("HVL (½)", f"{hvl:.2f} cm")
    
    with tab3:
        st.subheader("⚠️ Limites Réglementaires")
        
        st.write("### 📊 Limites de Dose")
        
        dose_limits = [
            {"Catégorie": "Public", "Dose annuelle": "1 mSv/an"},
            {"Catégorie": "Travailleurs", "Dose annuelle": "20 mSv/an"},
            {"Catégorie": "Femmes enceintes", "Dose (grossesse)": "1 mSv"},
        ]
        
        df_limits = pd.DataFrame(dose_limits)
        st.dataframe(df_limits, use_container_width=True)

# ==================== PAGE: DÉCHETS RADIOACTIFS ====================
elif page == "🗑️ Déchets Radioactifs":
    st.header("🗑️ Gestion des Déchets Radioactifs")
    
    tab1, tab2, tab3 = st.tabs(["📊 Classification", "📉 Décroissance", "🗄️ Stockage"])
    
    with tab1:
        st.subheader("📊 Classification des Déchets")
        
        st.write("### 🔢 Catégories Françaises")
        
        waste_categories = {
            "TFA (Très Faible Activité)": {
                "activité": "< 100 Bq/g",
                "volume": "28% du total",
                "stockage": "Centre CSTFA (Morvilliers)"
            },
            "FA-VC (Faible Activité Vie Courte)": {
                "activité": "< 1 MBq/g, T½ < 31 ans",
                "volume": "68% du total",
                "stockage": "Centre CSA (Soulaines)"
            },
            "MA-VL (Moyenne Activité Vie Longue)": {
                "activité": "1 MBq/g - 1 GBq/g",
                "volume": "3% du total",
                "stockage": "Cigéo (projet)"
            },
            "HA (Haute Activité)": {
                "activité": "> 1 GBq/g",
                "volume": "0.2% du total",
                "stockage": "Cigéo (projet)"
            }
        }
        
        for cat_name, cat_info in waste_categories.items():
            with st.expander(f"🗑️ {cat_name}"):
                for key, value in cat_info.items():
                    st.write(f"**{key.title()}:** {value}")
        
        st.markdown("---")
        
        volumes = [28, 68, 3, 0.2]
        categories = ["TFA", "FA-VC", "MA-VL", "HA"]
        
        fig = px.pie(values=volumes, names=categories, title="Répartition Volume (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📉 Décroissance Radioactive")
        
        st.write("### ⚛️ Loi de Décroissance")
        
        st.latex(r"A(t) = A_0 \cdot e^{-\lambda t}")
        
        with st.form("decay_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                isotope = st.selectbox("Isotope", ["Cs-137", "Sr-90", "I-131", "Pu-239"])
                half_lives = {"Cs-137": 30.17, "Sr-90": 28.8, "I-131": 0.022, "Pu-239": 24110}
                half_life = half_lives[isotope]
                st.metric("Demi-vie", f"{half_life:.2f} ans")
            
            with col2:
                initial_activity = st.number_input("Activité initiale (TBq)", 0.1, 10000.0, 100.0, 0.1)
                decay_time = st.number_input("Temps écoulé (années)", 0.0, 1000.0, 100.0, 10.0)
            
            submitted3 = st.form_submit_button("🔬 Calculer")
            
            if submitted3:
                lambda_decay = np.log(2) / half_life
                final_activity = initial_activity * np.exp(-lambda_decay * decay_time)
                
                st.success("✅ Calcul terminé!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Activité finale", f"{final_activity:.2f} TBq")
                with col2:
                    percent_remaining = (final_activity / initial_activity) * 100
                    st.metric("% restant", f"{percent_remaining:.4f}%")
    
    with tab3:
        st.subheader("🗄️ Solutions de Stockage")
        
        st.write("### 🏗️ Stockage Géologique Profond")
        
        st.info("""
        **Projet Cigéo (France)**
        • Localisation: Bure (Meuse/Haute-Marne)
        • Profondeur: 500 m
        • Capacité: 80,000 m³ (HA + MA-VL)
        • Coût: 25-35 Mrd€
        """)
        
        storage_strategy = [
            {"Phase": "Refroidissement piscine", "Durée": "5-10 ans"},
            {"Phase": "Entreposage sec", "Durée": "50-100 ans"},
            {"Phase": "Stockage géologique", "Durée": ">100,000 ans"}
        ]
        
        df_storage = pd.DataFrame(storage_strategy)
        st.dataframe(df_storage, use_container_width=True)

# ==================== PAGE: SIMULATIONS ====================
elif page == "📊 Simulations":
    st.header("📊 Simulations Avancées")
    
    tab1, tab2 = st.tabs(["🔬 Monte Carlo", "⚡ Transitoires"])
    
    with tab1:
        st.subheader("🔬 Simulations Monte Carlo")
        
        with st.form("monte_carlo_sim"):
            col1, col2 = st.columns(2)
            
            with col1:
                n_particles = st.number_input("Nombre neutrons", 1000, 1000000, 10000, 1000)
                n_generations = st.number_input("Générations", 10, 1000, 100, 10)
            
            with col2:
                geometry = st.selectbox("Géométrie", ["Cylindre", "Sphère"])
                material = st.selectbox("Matériau", ["UO2 4.5%", "MOX"])
            
            submitted4 = st.form_submit_button("🚀 Lancer Simulation")
            
            if submitted4:
                with st.spinner("Simulation en cours..."):
                    progress = st.progress(0)
                    
                    k_eff_values = []
                    for gen in range(n_generations):
                        progress.progress((gen + 1) / n_generations)
                        k_eff = 1.0 + np.random.randn() * 0.01 * np.exp(-gen/50)
                        k_eff_values.append(k_eff)
                    
                    progress.empty()
                    
                    k_eff_final = np.mean(k_eff_values[-20:])
                    std_dev = np.std(k_eff_values[-20:])
                    
                    st.success("✅ Simulation terminée!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("k_eff moyen", f"{k_eff_final:.5f}")
                    with col2:
                        st.metric("Écart-type", f"{std_dev:.5f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(n_generations)),
                        y=k_eff_values,
                        mode='lines',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Convergence k_effectif",
                        xaxis_title="Génération",
                        yaxis_title="k_eff",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                    progress.empty()

                    std_dev = np.std(k_eff_values[-20:])
                    
                    st.success("✅ Simulation terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("k_eff moyen", f"{k_eff_final:.5f}")
                    with col2:
                        st.metric("Écart-type", f"{std_dev:.5f}")
                    with col3:
                        st.metric("Incertitude", f"{std_dev*2:.5f} (2σ)")
                    
                    # Graphique convergence
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(n_generations)),
                        y=k_eff_values,
                        mode='lines+markers',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.add_hline(y=k_eff_final, line_dash="dash",
                                 annotation_text=f"k_eff = {k_eff_final:.5f}")
                    
                    fig.update_layout(
                        title="Convergence k_effectif",
                        xaxis_title="Génération",
                        yaxis_title="k_eff",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="chart_deu")
    
    with tab2:
        st.subheader("⚡ Transitoires Opérationnels")
        
        transient_types = st.selectbox(
            "Sélectionner Transitoire",
            ["Montée en puissance", "Insertion réactivité", "Variation débit"]
        )
        
        if st.button("🚀 Simuler Transitoire"):
            time_transient = np.linspace(0, 100, 500)
            
            if transient_types == "Montée en puissance":
                power = 20 + 80 * (1 - np.exp(-time_transient / 30))
                temp_fuel = 600 + 600 * (1 - np.exp(-time_transient / 35))
            else:
                power = 100 + 5 * np.sin(time_transient / 10)
                temp_fuel = 1200 + 50 * np.sin(time_transient / 10)
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Puissance", "Température"))
            
            fig.add_trace(go.Scatter(x=time_transient, y=power, name="Puissance (%)",
                                    line=dict(color='green', width=3)), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=time_transient, y=temp_fuel, name="T fuel (°C)",
                                    line=dict(color='red', width=3)), row=1, col=2)
            
            fig.update_xaxes(title_text="Temps (s)")
            fig.update_layout(height=400, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal des Événements (Dernières 10 entrées)"):
    if st.session_state.nuclear_system['log']:
        for event in st.session_state.nuclear_system['log'][-10:][::-1]:
            timestamp = event['timestamp'][:19]
            st.text(f"{timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")
    
    if st.button("🗑️ Effacer le Journal", key="clear_log_nuclear"):
        st.session_state.nuclear_system['log'] = []
        st.rerun()

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>☢️ Plateforme de Réacteurs Nucléaires</h3>
        <p>Système Intégré pour Conception et Analyse de Réacteurs</p>
        <p><small>Version 1.0.0 | Génie Nucléaire Complet</small></p>
        <p><small>⚛️ Neutronique | 🌡️ Thermique | 🔋 Combustible | 🛡️ Sûreté | ♻️ Cycle</small></p>
        <p><small>☢️ Radioprotection | 🗑️ Déchets | 💰 Économie | 🌍 Environnement</small></p>
        <p><small>Powered by Nuclear Engineering © 2024</small></p>
    </div>
""", unsafe_allow_html=True)