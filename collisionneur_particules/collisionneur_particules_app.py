
"""
Interface Streamlit pour la Plateforme de Physique des Particules
Système intégré pour créer, développer, simuler et analyser
des collisionneurs de particules et expériences de physique des hautes énergies
streamlit run collisionneur_particules_app.py
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
    page_title="⚛️ Plateforme Physique des Particules",
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
    .collider-card {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .particle-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .lepton {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .quark {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .boson {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    .hadron {
        background: linear-gradient(90deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .metric-box {
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
CONSTANTS = {
    'c': 299792458,  # m/s
    'h': 6.62607015e-34,  # J·s
    'electron_mass': 0.511,  # MeV/c²
    'proton_mass': 938.272,  # MeV/c²
    'Z_mass': 91.1876,  # GeV/c²
    'W_mass': 80.379,  # GeV/c²
    'Higgs_mass': 125.10,  # GeV/c²
    'top_mass': 173.0,  # GeV/c²
}

# ==================== INITIALISATION SESSION STATE ====================
if 'particle_system' not in st.session_state:
    st.session_state.particle_system = {
        'colliders': {},
        'experiments': {},
        'simulations': [],
        'analyses': {},
        'datasets': {},
        'detectors': {},
        'beams': {},
        'results': [],
        'discoveries': [],
        'publications': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================
def log_event(message: str):
    """Enregistre un événement"""
    st.session_state.particle_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })

def get_particle_badge(particle_type: str) -> str:
    """Retourne un badge HTML pour un type de particule"""
    badges = {
        'electron': '<span class="particle-badge lepton">e⁻ Électron</span>',
        'positron': '<span class="particle-badge lepton">e⁺ Positron</span>',
        'muon': '<span class="particle-badge lepton">μ⁻ Muon</span>',
        'proton': '<span class="particle-badge hadron">p Proton</span>',
        'quark_top': '<span class="particle-badge quark">t Top</span>',
        'quark_bottom': '<span class="particle-badge quark">b Bottom</span>',
        'w_boson': '<span class="particle-badge boson">W± Boson W</span>',
        'z_boson': '<span class="particle-badge boson">Z⁰ Boson Z</span>',
        'higgs': '<span class="particle-badge boson">H Higgs</span>',
        'photon': '<span class="particle-badge boson">γ Photon</span>',
    }
    return badges.get(particle_type, '<span class="particle-badge">?</span>')

def create_collider_mock(name, collider_type, config):
    """Crée un collisionneur simulé"""
    collider_id = f"collider_{len(st.session_state.particle_system['colliders']) + 1}"
    
    collider = {
        'id': collider_id,
        'name': name,
        'type': collider_type,
        'created_at': datetime.now().isoformat(),
        'status': 'offline',
        'specifications': {
            'circumference': config.get('circumference', 27.0),
            'beam_energy': config.get('beam_energy', 7000),
            'center_mass_energy': config.get('beam_energy', 7000) * 2,
            'tunnel_depth': config.get('tunnel_depth', 100),
        },
        'performance': {
            'luminosity': config.get('luminosity', 1e34),
            'peak_luminosity': config.get('peak_luminosity', 2e34),
            'integrated_luminosity': 0.0,
            'collision_rate': config.get('collision_rate', 40e6),
            'uptime': 0.0
        },
        'beams': {
            'particle_type_1': config.get('particle_1', 'proton'),
            'particle_type_2': config.get('particle_2', 'proton'),
            'bunches_per_beam': config.get('bunches', 2808),
            'particles_per_bunch': config.get('particles_bunch', 1.15e11),
            'bunch_spacing': config.get('bunch_spacing', 25.0)
        },
        'infrastructure': {
            'power_consumption': config.get('power', 200),
            'cooling_capacity': config.get('cooling', 150),
            'cryogenic_capacity': config.get('cryo', 50),
            'dipole_magnets': config.get('dipoles', 1232),
            'quadrupole_magnets': config.get('quadrupoles', 392),
            'rf_cavities': config.get('rf_cavities', 400)
        },
        'detectors': config.get('detectors', []),
        'experiments': [],
        'operations': {
            'hours': 0.0,
            'collisions_delivered': 0,
            'data_recorded': 0.0,
            'efficiency': 0.0
        },
        'costs': {
            'construction': config.get('construction_cost', 5000),
            'annual_operation': config.get('operation_cost', 500),
            'upgrade_budget': config.get('upgrade_budget', 1000)
        },
        'physics': {
            'discoveries': [],
            'publications': 0,
            'citations': 0
        }
    }
    
    st.session_state.particle_system['colliders'][collider_id] = collider
    log_event(f"Collisionneur créé: {name} ({collider_type})")
    return collider_id

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">⚛️ Plateforme de Physique des Particules</h1>', unsafe_allow_html=True)
st.markdown("### Système Intégré pour Collisionneurs, Expériences et Analyses en Physique des Hautes Énergies")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Particle+Physics+Lab", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Tableau de Bord",
            "⚛️ Mes Collisionneurs",
            "➕ Créer Collisionneur",
            "🔬 Détecteurs",
            "📡 Faisceaux & Injection",
            "🧲 Magnets & RF",
            "💫 Simulations Monte Carlo",
            "🎯 Collisions & Luminosité",
            "📊 Acquisition de Données",
            "🔍 Reconstruction d'Événements",
            "📈 Analyses Physiques",
            "🏆 Découvertes",
            "📚 Modèle Standard",
            "🌌 Physique BSM",
            "⚡ Sections Efficaces",
            "🎲 Générateurs d'Événements",
            "🔧 Calibration",
            "💰 Coûts & Budget",
            "📑 Publications",
            "🌟 Applications",
            "🎓 Formation",
            "🔬 Laboratoires"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_colliders = len(st.session_state.particle_system['colliders'])
    active_colliders = sum(1 for c in st.session_state.particle_system['colliders'].values() if c['status'] == 'online')
    total_experiments = len(st.session_state.particle_system['experiments'])
    total_discoveries = len(st.session_state.particle_system['discoveries'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚛️ Collisionneurs", total_colliders)
        st.metric("🔬 Expériences", total_experiments)
    with col2:
        st.metric("✅ Actifs", active_colliders)
        st.metric("🏆 Découvertes", total_discoveries)

# ==================== PAGE: TABLEAU DE BORD ====================
if page == "🏠 Tableau de Bord":
    st.header("📊 Tableau de Bord Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="collider-card"><h2>⚛️</h2><h3>{total_colliders}</h3><p>Collisionneurs</p></div>', unsafe_allow_html=True)
    
    with col2:
        total_lumi = sum(c['performance']['integrated_luminosity'] for c in st.session_state.particle_system['colliders'].values())
        st.markdown(f'<div class="collider-card"><h2>💫</h2><h3>{total_lumi:.1f}</h3><p>fb⁻¹ Livrés</p></div>', unsafe_allow_html=True)
    
    with col3:
        total_events = sum(c['operations']['collisions_delivered'] for c in st.session_state.particle_system['colliders'].values())
        st.markdown(f'<div class="collider-card"><h2>🎯</h2><h3>{total_events/1e9:.1f}B</h3><p>Collisions</p></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="collider-card"><h2>🏆</h2><h3>{total_discoveries}</h3><p>Découvertes</p></div>', unsafe_allow_html=True)
    
    with col5:
        total_pubs = sum(c['physics']['publications'] for c in st.session_state.particle_system['colliders'].values())
        st.markdown(f'<div class="collider-card"><h2>📄</h2><h3>{total_pubs}</h3><p>Publications</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Constantes physiques
    st.subheader("📐 Constantes Fondamentales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vitesse de la lumière", "2.998×10⁸ m/s")
        st.metric("Constante de Planck", "6.626×10⁻³⁴ J·s")
    
    with col2:
        st.metric("Masse électron", "0.511 MeV/c²")
        st.metric("Masse proton", "938.3 MeV/c²")
    
    with col3:
        st.metric("Boson Z", "91.19 GeV/c²")
        st.metric("Boson W", "80.38 GeV/c²")
    
    with col4:
        st.metric("Boson de Higgs", "125.1 GeV/c²")
        st.metric("Quark Top", "173.0 GeV/c²")
    
    st.markdown("---")
    
    if st.session_state.particle_system['colliders']:
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚡ Énergie par Collisionneur")
            
            names = [c['name'][:20] for c in st.session_state.particle_system['colliders'].values()]
            energies = [c['specifications']['center_mass_energy']/1000 for c in st.session_state.particle_system['colliders'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=energies, marker_color='rgb(102, 126, 234)',
                      text=[f"{e:.1f} TeV" for e in energies],
                      textposition='outside')
            ])
            fig.update_layout(title="Énergie Centre de Masse", yaxis_title="TeV", xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💫 Luminosité Intégrée")
            
            names = [c['name'][:20] for c in st.session_state.particle_system['colliders'].values()]
            lumis = [c['performance']['integrated_luminosity'] for c in st.session_state.particle_system['colliders'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=lumis, marker_color='rgb(118, 75, 162)',
                      text=[f"{l:.1f} fb⁻¹" for l in lumis],
                      textposition='outside')
            ])
            fig.update_layout(title="Luminosité Livrée", yaxis_title="fb⁻¹", xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Timeline des découvertes
        if st.session_state.particle_system['discoveries']:
            st.subheader("🏆 Timeline des Découvertes")
            
            discoveries_df = pd.DataFrame(st.session_state.particle_system['discoveries'])
            st.dataframe(discoveries_df, use_container_width=True)
    else:
        st.info("💡 Aucun collisionneur créé. Créez votre premier collisionneur!")

# ==================== PAGE: MES COLLISIONNEURS ====================
elif page == "⚛️ Mes Collisionneurs":
    st.header("⚛️ Gestion des Collisionneurs")
    
    if not st.session_state.particle_system['colliders']:
        st.info("💡 Aucun collisionneur créé.")
    else:
        for collider_id, collider in st.session_state.particle_system['colliders'].items():
            st.markdown(f'<div class="collider-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### ⚛️ {collider['name']}")
                st.write(f"**Type:** {collider['type'].replace('_', ' ').title()}")
                
                # Badges particules
                p1 = collider['beams']['particle_type_1']
                p2 = collider['beams']['particle_type_2']
                st.markdown(get_particle_badge(p1) + " ⚔️ " + get_particle_badge(p2), unsafe_allow_html=True)
            
            with col2:
                st.metric("Énergie CM", f"{collider['specifications']['center_mass_energy']/1000:.1f} TeV")
                st.metric("Circonférence", f"{collider['specifications']['circumference']:.1f} km")
            
            with col3:
                st.metric("Luminosité", f"{collider['performance']['luminosity']:.2e} cm⁻²s⁻¹")
                st.metric("∫L dt", f"{collider['performance']['integrated_luminosity']:.1f} fb⁻¹")
            
            with col4:
                status_icon = "🟢" if collider['status'] == 'online' else "🔴"
                st.write(f"**Statut:** {status_icon} {collider['status'].upper()}")
                st.metric("Uptime", f"{collider['performance']['uptime']:.0f}%")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚙️ Spécifications", "📡 Faisceaux", "🧲 Infrastructure", "📊 Opérations", "💰 Coûts"])
                
                with tab1:
                    st.subheader("⚙️ Spécifications Techniques")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Circonférence", f"{collider['specifications']['circumference']:.1f} km")
                    with col2:
                        st.metric("Énergie Faisceau", f"{collider['specifications']['beam_energy']:.0f} GeV")
                    with col3:
                        st.metric("Énergie CM", f"{collider['specifications']['center_mass_energy']/1000:.1f} TeV")
                    with col4:
                        st.metric("Profondeur Tunnel", f"{collider['specifications']['tunnel_depth']:.0f} m")
                
                with tab2:
                    st.subheader("📡 Configuration des Faisceaux")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Faisceau 1:**")
                        st.write(f"Particule: {collider['beams']['particle_type_1']}")
                        st.metric("Paquets", f"{collider['beams']['bunches_per_beam']:,}")
                        st.metric("Particules/paquet", f"{collider['beams']['particles_per_bunch']:.2e}")
                    
                    with col2:
                        st.write("**Faisceau 2:**")
                        st.write(f"Particule: {collider['beams']['particle_type_2']}")
                        st.metric("Espacement", f"{collider['beams']['bunch_spacing']:.1f} ns")
                        st.metric("Fréquence collision", f"{collider['performance']['collision_rate']/1e6:.0f} MHz")
                
                with tab3:
                    st.subheader("🧲 Infrastructure")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Consommation", f"{collider['infrastructure']['power_consumption']:.0f} MW")
                        st.metric("Refroidissement", f"{collider['infrastructure']['cooling_capacity']:.0f} MW")
                    
                    with col2:
                        st.metric("Aimants Dipôles", collider['infrastructure']['dipole_magnets'])
                        st.metric("Quadrupôles", collider['infrastructure']['quadrupole_magnets'])
                    
                    with col3:
                        st.metric("Cavités RF", collider['infrastructure']['rf_cavities'])
                        st.metric("Cryogénie", f"{collider['infrastructure']['cryogenic_capacity']:.0f} kW")
                
                with tab4:
                    st.subheader("📊 Statistiques Opérationnelles")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Heures Opération", f"{collider['operations']['hours']:.0f}h")
                        st.metric("Efficacité", f"{collider['operations']['efficiency']:.1f}%")
                    
                    with col2:
                        st.metric("Collisions Livrées", f"{collider['operations']['collisions_delivered']/1e9:.2f}B")
                        st.metric("Données Enregistrées", f"{collider['operations']['data_recorded']:.1f} PB")
                    
                    with col3:
                        st.metric("Publications", collider['physics']['publications'])
                        st.metric("Citations", collider['physics']['citations'])
                
                with tab5:
                    st.subheader("💰 Analyse Financière")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Construction", f"€{collider['costs']['construction']:.0f}M")
                    with col2:
                        st.metric("Opération Annuelle", f"€{collider['costs']['annual_operation']:.0f}M")
                    with col3:
                        st.metric("Budget Upgrades", f"€{collider['costs']['upgrade_budget']:.0f}M")
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button(f"▶️ {'Éteindre' if collider['status'] == 'online' else 'Activer'}", key=f"toggle_{collider_id}"):
                        collider['status'] = 'offline' if collider['status'] == 'online' else 'online'
                        log_event(f"{collider['name']} {'éteint' if collider['status'] == 'offline' else 'activé'}")
                        st.rerun()
                
                with col2:
                    if st.button(f"💫 Collision Run", key=f"run_{collider_id}"):
                        st.info("Allez dans Collisions & Luminosité")
                
                with col3:
                    if st.button(f"📊 Analyser", key=f"analyze_{collider_id}"):
                        st.info("Allez dans Analyses Physiques")
                
                with col4:
                    if st.button(f"🔧 Maintenance", key=f"maint_{collider_id}"):
                        st.warning("Mode maintenance activé")
                
                with col5:
                    if st.button(f"🗑️ Supprimer", key=f"del_{collider_id}"):
                        del st.session_state.particle_system['colliders'][collider_id]
                        log_event(f"{collider['name']} supprimé")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CRÉER COLLISIONNEUR ====================
elif page == "➕ Créer Collisionneur":
    st.header("➕ Créer un Nouveau Collisionneur")
    
    with st.form("create_collider_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            collider_name = st.text_input("📝 Nom du Collisionneur", placeholder="Ex: Future Circular Collider")
            
            collider_type = st.selectbox(
                "⚛️ Type de Collisionneur",
                [
                    "circulaire",
                    "lineaire",
                    "plasma",
                    "muon",
                    "electron_positron",
                    "proton_proton",
                    "ion_lourd",
                    "electron_proton",
                    "photon_photon"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            application = st.selectbox(
                "🎯 Objectif Principal",
                ["Découvertes", "Physique de Précision", "Recherche BSM", 
                 "Physique du Higgs", "Physique du Top", "QCD", "Électrofaible"]
            )
            
            era = st.selectbox(
                "🕐 Génération",
                ["Actuelle", "Haute Luminosité", "Future", "Post-LHC", "Ultime"]
            )
        
        st.markdown("---")
        st.subheader("📐 Spécifications Physiques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            circumference = st.number_input("Circonférence (km)", 0.1, 1000.0, 27.0, 0.1)
            tunnel_depth = st.number_input("Profondeur Tunnel (m)", 10, 500, 100, 10)
        
        with col2:
            beam_energy = st.number_input("Énergie Faisceau (GeV)", 1, 100000, 7000, 100)
            cm_energy = beam_energy * 2
            st.metric("Énergie CM", f"{cm_energy/1000:.1f} TeV")
        
        with col3:
            luminosity_target = st.number_input("Luminosité Cible (×10³⁴)", 0.1, 100.0, 1.0, 0.1)
            st.metric("Luminosité", f"{luminosity_target:.1f}×10³⁴ cm⁻²s⁻¹")
        
        st.markdown("---")
        st.subheader("📡 Configuration des Faisceaux")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Faisceau 1:**")
            particle_1 = st.selectbox(
                "Type de Particule 1",
                ["electron", "positron", "proton", "antiproton", "muon", "ion_lourd"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            bunches_per_beam = st.number_input("Paquets par Faisceau", 1, 10000, 2808, 1)
        
        with col2:
            st.write("**Faisceau 2:**")
            particle_2 = st.selectbox(
                "Type de Particule 2",
                ["electron", "positron", "proton", "antiproton", "muon", "ion_lourd"],
                format_func=lambda x: x.replace('_', ' ').title(),
                index=2
            )
            
            particles_per_bunch = st.number_input("Particules/Paquet (×10¹¹)", 0.1, 10.0, 1.15, 0.01)
        
        bunch_spacing = st.slider("Espacement des Paquets (ns)", 1, 100, 25, 1)
        
        st.markdown("---")
        st.subheader("🧲 Infrastructure")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dipole_magnets = st.number_input("Aimants Dipôles", 100, 10000, 1232, 10)
            quadrupole_magnets = st.number_input("Quadrupôles", 50, 5000, 392, 10)
        
        with col2:
            rf_cavities = st.number_input("Cavités RF", 10, 2000, 400, 10)
            magnetic_field = st.number_input("Champ Magnétique (T)", 1.0, 20.0, 8.3, 0.1)
        
        with col3:
            power_consumption = st.number_input("Consommation (MW)", 10, 1000, 200, 10)
            cooling_capacity = st.number_input("Refroidissement (MW)", 10, 500, 150, 10)
        
        st.markdown("---")
        st.subheader("🔬 Détecteurs")
        
        n_detectors = st.number_input("Nombre de Détecteurs", 1, 10, 4, 1)
        
        detectors = []
        for i in range(n_detectors):
            col1, col2 = st.columns(2)
            with col1:
                det_name = st.text_input(f"Nom Détecteur {i+1}", f"Detector_{i+1}", key=f"det_name_{i}")
            with col2:
                det_type = st.selectbox(
                    f"Type {i+1}",
                    ["Général", "Précision", "Heavy Ion", "Forward"],
                    key=f"det_type_{i}"
                )
            
            if det_name:
                detectors.append({'name': det_name, 'type': det_type})
        
        st.markdown("---")
        st.subheader("💰 Budget et Coûts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            construction_cost = st.number_input("Coût Construction (M€)", 100, 50000, 5000, 100)
        with col2:
            operation_cost = st.number_input("Coût Opération Annuel (M€)", 10, 5000, 500, 10)
        with col3:
            upgrade_budget = st.number_input("Budget Upgrades (M€)", 100, 10000, 1000, 100)
        
        st.markdown("---")
        
        # Résumé
        st.subheader("📊 Résumé")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Énergie CM", f"{cm_energy/1000:.1f} TeV")
        with col2:
            st.metric("Luminosité", f"{luminosity_target:.1f}×10³⁴")
        with col3:
            st.metric("Circonférence", f"{circumference:.1f} km")
        with col4:
            st.metric("Coût Total", f"€{construction_cost:.0f}M")
        
        submitted = st.form_submit_button("🚀 Créer le Collisionneur", use_container_width=True, type="primary")
        
        if submitted:
            if not collider_name:
                st.error("⚠️ Veuillez donner un nom au collisionneur")
            else:
                with st.spinner("🔄 Création du collisionneur en cours..."):
                    config = {
                        'circumference': circumference,
                        'beam_energy': beam_energy,
                        'tunnel_depth': tunnel_depth,
                        'luminosity': luminosity_target * 1e34,
                        'peak_luminosity': luminosity_target * 2e34,
                        'collision_rate': 40e6,
                        'particle_1': particle_1,
                        'particle_2': particle_2,
                        'bunches': bunches_per_beam,
                        'particles_bunch': particles_per_bunch * 1e11,
                        'bunch_spacing': bunch_spacing,
                        'dipoles': dipole_magnets,
                        'quadrupoles': quadrupole_magnets,
                        'rf_cavities': rf_cavities,
                        'power': power_consumption,
                        'cooling': cooling_capacity,
                        'cryo': 50,
                        'detectors': detectors,
                        'construction_cost': construction_cost,
                        'operation_cost': operation_cost,
                        'upgrade_budget': upgrade_budget
                    }
                    
                    collider_id = create_collider_mock(collider_name, collider_type, config)
                    
                    st.success(f"✅ Collisionneur '{collider_name}' créé avec succès!")
                    st.balloons()
                    
                    collider = st.session_state.particle_system['colliders'][collider_id]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Énergie CM", f"{collider['specifications']['center_mass_energy']/1000:.1f} TeV")
                    with col2:
                        st.metric("Luminosité", f"{collider['performance']['luminosity']:.2e}")
                    with col3:
                        st.metric("Circonférence", f"{collider['specifications']['circumference']:.1f} km")
                    with col4:
                        st.metric("Détecteurs", len(detectors))
                    
                    st.code(f"ID: {collider_id}", language="text")

# ==================== PAGE: DÉTECTEURS ====================
elif page == "🔬 Détecteurs":
    st.header("🔬 Systèmes de Détection")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📡 Types", "🔧 Configuration", "📊 Performance", "💾 DAQ"])
    
    with tab1:
        st.subheader("📡 Types de Détecteurs")
        
        detector_types = {
            "Trajectographe (Tracker)": {
                "description": "Mesure la trajectoire des particules chargées",
                "technologies": ["Silicium pixels", "Silicium strips", "Micro-strips"],
                "resolution": "10-100 μm",
                "couverture": "|η| < 2.5",
                "applications": ["Reconstruction vertex", "Mesure moment"]
            },
            "Calorimètre Électromagnétique": {
                "description": "Mesure l'énergie des électrons et photons",
                "technologies": ["Cristaux scintillants", "Lead/Tungsten-Argon liquide"],
                "resolution": "σ/E = 10%/√E ⊕ 0.7%",
                "couverture": "|η| < 3.0",
                "applications": ["Électrons", "Photons", "Higgs→γγ"]
            },
            "Calorimètre Hadronique": {
                "description": "Mesure l'énergie des hadrons",
                "technologies": ["Fer-Scintillateur", "Cuivre-Argon liquide"],
                "resolution": "σ/E = 50%/√E ⊕ 3%",
                "couverture": "|η| < 5.0",
                "applications": ["Jets", "Énergie manquante", "Quarks"]
            },
            "Chambres à Muons": {
                "description": "Détection et mesure des muons",
                "technologies": ["RPC", "CSC", "MDT"],
                "resolution": "100 μm - 1 mm",
                "couverture": "|η| < 2.4",
                "applications": ["Identification muons", "Trigger", "Z→μμ"]
            },
            "Détecteur de Vertex": {
                "description": "Haute résolution près du point d'interaction",
                "technologies": ["Pixels 3D", "MAPS", "Diamond"],
                "resolution": "< 10 μm",
                "couverture": "|η| < 2.5",
                "applications": ["Quarks b/c", "Temps de vie", "Vertex secondaires"]
            },
            "Cherenkov": {
                "description": "Identification de particules par effet Cherenkov",
                "technologies": ["RICH", "TRD"],
                "resolution": "Identification π/K/p",
                "couverture": "Variable",
                "applications": ["PID", "Séparation particules"]
            }
        }
        
        for det_name, det_info in detector_types.items():
            with st.expander(f"🔬 {det_name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {det_info['description']}")
                    st.write(f"**Résolution:** {det_info['resolution']}")
                    st.write(f"**Couverture:** {det_info['couverture']}")
                    
                    st.write("\n**Technologies:**")
                    for tech in det_info['technologies']:
                        st.write(f"• {tech}")
                
                with col2:
                    st.write("**Applications:**")
                    for app in det_info['applications']:
                        st.write(f"✓ {app}")
    
    with tab2:
        st.subheader("🔧 Configurer un Détecteur")
        
        with st.form("detector_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                det_name = st.text_input("Nom du Détecteur", "ATLAS-like")
                det_type = st.selectbox("Type", ["Général", "Précision", "Heavy Ion", "Forward"])
            
            with col2:
                acceptance = st.slider("Acceptance géométrique", 0.0, 1.0, 0.95, 0.01)
                n_layers = st.number_input("Nombre de Couches", 1, 20, 6, 1)
            
            st.write("### 📏 Géométrie")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                inner_radius = st.number_input("Rayon Interne (m)", 0.0, 10.0, 0.3, 0.1)
            with col2:
                outer_radius = st.number_input("Rayon Externe (m)", 0.1, 20.0, 5.0, 0.1)
            with col3:
                length = st.number_input("Longueur (m)", 0.1, 50.0, 10.0, 0.1)
            
            st.write("### 🎯 Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                res_energy = st.number_input("Résolution Énergie (%)", 0.1, 50.0, 10.0, 0.1)
            with col2:
                res_position = st.number_input("Résolution Position (μm)", 1.0, 1000.0, 100.0, 1.0)
            with col3:
                res_time = st.number_input("Résolution Temps (ps)", 10.0, 1000.0, 100.0, 10.0)
            
            st.write("### 🔌 Électronique")
            
            col1, col2 = st.columns(2)
            
            with col1:
                channels = st.number_input("Nombre de Canaux", 1000, 100000000, 100000000, 1000)
            with col2:
                readout_rate = st.number_input("Taux Lecture (MHz)", 1, 1000, 40, 1)
            
            submitted = st.form_submit_button("💾 Sauvegarder Configuration")
            
            if submitted:
                detector = {
                    'name': det_name,
                    'type': det_type,
                    'geometry': {
                        'inner_radius': inner_radius,
                        'outer_radius': outer_radius,
                        'length': length
                    },
                    'performance': {
                        'resolution_energy': res_energy,
                        'resolution_position': res_position,
                        'resolution_time': res_time,
                        'acceptance': acceptance
                    },
                    'electronics': {
                        'channels': channels,
                        'readout_rate': readout_rate
                    },
                    'layers': n_layers
                }
                
                st.session_state.particle_system['detectors'][det_name] = detector
                st.success(f"✅ Détecteur '{det_name}' configuré!")
                log_event(f"Détecteur créé: {det_name}")
    
    with tab3:
        st.subheader("📊 Performance des Détecteurs")
        
        if st.session_state.particle_system['detectors']:
            detector_names = list(st.session_state.particle_system['detectors'].keys())
            selected_det = st.selectbox("Sélectionner Détecteur", detector_names)
            
            detector = st.session_state.particle_system['detectors'][selected_det]
            
            st.write(f"### 🔬 {selected_det}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Résolution E", f"{detector['performance']['resolution_energy']:.1f}%")
            with col2:
                st.metric("Résolution x", f"{detector['performance']['resolution_position']:.0f} μm")
            with col3:
                st.metric("Résolution t", f"{detector['performance']['resolution_time']:.0f} ps")
            with col4:
                st.metric("Acceptance", f"{detector['performance']['acceptance']:.0%}")
            
            st.markdown("---")
            
            # Fonction de résolution
            st.write("### 📈 Fonction de Résolution en Énergie")
            
            energy = np.logspace(0, 3, 100)  # 1 GeV à 1 TeV
            
            # σ/E = a/√E ⊕ b
            a = detector['performance']['resolution_energy']
            b = 0.7
            resolution = np.sqrt((a / np.sqrt(energy))**2 + b**2)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=energy, y=resolution,
                mode='lines',
                line=dict(color='blue', width=3)
            ))
            
            fig.update_layout(
                title="Résolution Relative vs Énergie",
                xaxis_title="Énergie (GeV)",
                yaxis_title="σ/E (%)",
                xaxis_type="log",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun détecteur configuré")
    
    with tab4:
        st.subheader("💾 Data Acquisition (DAQ)")
        
        st.write("### 🔄 Système d'Acquisition")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Taux Collision", "40 MHz")
            st.metric("Taux Trigger L1", "100 kHz")
        
        with col2:
            st.metric("Taux Trigger HLT", "1 kHz")
            st.metric("Taux Enregistrement", "1 kHz")
        
        with col3:
            st.metric("Taille Événement", "1.5 MB")
            st.metric("Flux Données", "1.5 GB/s")
        
        st.markdown("---")
        
        st.write("### 📊 Pipeline de Traitement")
        
        pipeline_stages = [
            {"stage": "L1 Trigger", "rate_in": "40 MHz", "rate_out": "100 kHz", "latency": "2.5 μs", "rejection": "400x"},
            {"stage": "HLT", "rate_in": "100 kHz", "rate_out": "1 kHz", "latency": "200 ms", "rejection": "100x"},
            {"stage": "Reconstruction", "rate_in": "1 kHz", "rate_out": "1 kHz", "latency": "~24h", "rejection": "1x"},
            {"stage": "Analyse", "rate_in": "1 kHz", "rate_out": "Variable", "latency": "Semaines", "rejection": "Variable"}
        ]
        
        df = pd.DataFrame(pipeline_stages)
        st.dataframe(df, use_container_width=True)
        
        # Graphique flux
        stages = [s['stage'] for s in pipeline_stages]
        rates_in = [40e6, 100e3, 1e3, 1e3]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stages, y=rates_in,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=12)
        ))
        
        fig.update_layout(
            title="Flux de Données à Travers le DAQ",
            yaxis_title="Taux (Hz)",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: SIMULATIONS MONTE CARLO ====================
elif page == "💫 Simulations Monte Carlo":
    st.header("💫 Générateurs et Simulations Monte Carlo")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎲 Générateurs", "🔬 Événements", "📊 Distributions", "🎯 Validation"])
    
    with tab1:
        st.subheader("🎲 Générateurs d'Événements")
        
        generators = {
            "PYTHIA": {
                "description": "Générateur généraliste pour collisions hadroniques",
                "processes": ["QCD", "Électrofaible", "Higgs", "BSM"],
                "features": ["Hadronisation", "Parton Shower", "MPI"],
                "version": "8.3"
            },
            "HERWIG": {
                "description": "Générateur avec parton shower angulaire",
                "processes": ["QCD", "Électrofaible", "Higgs"],
                "features": ["Cluster hadronisation", "Angular ordering"],
                "version": "7.2"
            },
            "MadGraph": {
                "description": "Calculs matrice exacte multi-jambes",
                "processes": ["NLO", "Processus complexes", "BSM"],
                "features": ["Automation", "NLO", "Interface UFO"],
                "version": "3.5"
            },
            "SHERPA": {
                "description": "Multi-purpose event generator",
                "processes": ["ME+PS matching", "NLO", "NNLO"],
                "features": ["Dipole shower", "Multi-jet merging"],
                "version": "2.2"
            },
            "POWHEG": {
                "description": "Générateur NLO+PS",
                "processes": ["NLO matching", "Tous processus SM"],
                "features": ["Positive weights", "Unitarité"],
                "version": "V2"
            }
        }
        
        for gen_name, gen_info in generators.items():
            with st.expander(f"🎲 {gen_name} v{gen_info['version']}"):
                st.write(f"**Description:** {gen_info['description']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Processus:**")
                    for proc in gen_info['processes']:
                        st.write(f"• {proc}")
                
                with col2:
                    st.write("**Fonctionnalités:**")
                    for feat in gen_info['features']:
                        st.write(f"✓ {feat}")
    
    with tab2:
        st.subheader("🔬 Génération d'Événements")
        
        with st.form("generate_events"):
            col1, col2 = st.columns(2)
            
            with col1:
                process = st.selectbox(
                    "Processus Physique",
                    ["Higgs → γγ", "Higgs → ZZ* → 4l", "tt̄ production", 
                     "Z → l⁺l⁻", "W → lν", "Diboson (WW, ZZ, WZ)",
                     "QCD dijets", "SUSY", "Z' → ll", "Graviton → ll"]
                )
                
                generator = st.selectbox("Générateur", ["PYTHIA", "HERWIG", "MadGraph", "SHERPA"])
            
            with col2:
                energy_cm = st.number_input("Énergie CM (TeV)", 1.0, 100.0, 13.0, 0.1)
                n_events = st.number_input("Nombre d'Événements", 100, 10000000, 100000, 100)
            
            st.write("### ⚙️ Paramètres")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pdf_set = st.selectbox("PDF Set", ["NNPDF3.1", "CT18", "MMHT2014"])
                alpha_s = st.number_input("αs(MZ)", 0.10, 0.13, 0.118, 0.001)
            
            with col2:
                parton_shower = st.checkbox("Parton Shower", value=True)
                hadronization = st.checkbox("Hadronisation", value=True)
                underlying_event = st.checkbox("Underlying Event", value=True)
            
            submitted = st.form_submit_button("🚀 Générer Événements", type="primary")
            
            if submitted:
                with st.spinner(f"Génération de {n_events:,} événements..."):
                    progress_bar = st.progress(0)
                    
                    simulation = {
                        'sim_id': f"sim_{len(st.session_state.particle_system['simulations']) + 1}",
                        'process': process,
                        'generator': generator,
                        'energy': energy_cm,
                        'n_events': n_events,
                        'timestamp': datetime.now().isoformat(),
                        'cross_section': 0.0,
                        'events': []
                    }
                    
                    # Calcul section efficace (simplifié)
                    cross_sections = {
                        "Higgs → γγ": 50 * 0.00227,
                        "Higgs → ZZ* → 4l": 50 * 0.000124,
                        "tt̄ production": 830,
                        "Z → l⁺l⁻": 6000,
                        "W → lν": 20000,
                        "Diboson (WW, ZZ, WZ)": 120,
                        "QCD dijets": 50000,
                    }
                    
                    simulation['cross_section'] = cross_sections.get(process, 100.0)
                    
                    # Génération simplifiée
                    for i in range(min(n_events, 1000)):  # Limiter pour performance
                        progress_bar.progress((i + 1) / min(n_events, 1000))
                        
                        event = {
                            'event_id': i,
                            'weight': 1.0,
                            'particles': []
                        }
                        
                        # Génération particules selon le processus
                        if "Higgs" in process:
                            if "γγ" in process:
                                event['particles'] = [
                                    {'type': 'photon', 'pt': np.random.exponential(40), 'eta': np.random.uniform(-2.5, 2.5)},
                                    {'type': 'photon', 'pt': np.random.exponential(30), 'eta': np.random.uniform(-2.5, 2.5)}
                                ]
                            elif "4l" in process:
                                event['particles'] = [
                                    {'type': 'muon', 'pt': np.random.exponential(25), 'eta': np.random.uniform(-2.4, 2.4)},
                                    {'type': 'muon', 'pt': np.random.exponential(20), 'eta': np.random.uniform(-2.4, 2.4)},
                                    {'type': 'muon', 'pt': np.random.exponential(15), 'eta': np.random.uniform(-2.4, 2.4)},
                                    {'type': 'muon', 'pt': np.random.exponential(10), 'eta': np.random.uniform(-2.4, 2.4)}
                                ]
                        
                        simulation['events'].append(event)
                    
                    progress_bar.empty()
                    
                    st.session_state.particle_system['simulations'].append(simulation)
                    
                    st.success(f"✅ {n_events:,} événements générés!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Processus", process[:20])
                    with col2:
                        st.metric("σ", f"{simulation['cross_section']:.2f} pb")
                    with col3:
                        st.metric("Événements", f"{n_events:,}")
                    with col4:
                        st.metric("Générateur", generator)
                    
                    log_event(f"Simulation MC: {process} - {n_events:,} événements")
    
    with tab3:
        st.subheader("📊 Distributions Cinématiques")
        
        if st.session_state.particle_system['simulations']:
            sim_ids = [s['sim_id'] for s in st.session_state.particle_system['simulations']]
            selected_sim = st.selectbox("Sélectionner Simulation", sim_ids,
                                       format_func=lambda x: next(s['process'] for s in st.session_state.particle_system['simulations'] if s['sim_id'] == x))
            
            simulation = next(s for s in st.session_state.particle_system['simulations'] if s['sim_id'] == selected_sim)
            
            st.write(f"### {simulation['process']}")
            st.write(f"**Section Efficace:** {simulation['cross_section']:.3f} pb")
            st.write(f"**Événements:** {simulation['n_events']:,}")
            
            if simulation['events']:
                # Extraction des pT
                all_pt = []
                all_eta = []
                
                for event in simulation['events']:
                    for particle in event['particles']:
                        all_pt.append(particle['pt'])
                        all_eta.append(particle['eta'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution pT
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=all_pt,
                        nbinsx=50,
                        marker_color='blue',
                        name='pT'
                    ))
                    
                    fig.update_layout(
                        title="Distribution de pT",
                        xaxis_title="pT (GeV)",
                        yaxis_title="Événements",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Distribution η
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=all_eta,
                        nbinsx=50,
                        marker_color='green',
                        name='η'
                    ))
                    
                    fig.update_layout(
                        title="Distribution de η (Pseudorapidité)",
                        xaxis_title="η",
                        yaxis_title="Événements",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Distribution 2D
                st.markdown("---")
                
                fig = go.Figure(data=go.Histogram2d(
                    x=all_eta,
                    y=all_pt,
                    colorscale='Viridis',
                    nbinsx=30,
                    nbinsy=30
                ))
                
                fig.update_layout(
                    title="Distribution 2D: η vs pT",
                    xaxis_title="η",
                    yaxis_title="pT (GeV)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune simulation disponible")
    
    with tab4:
        st.subheader("🎯 Validation Monte Carlo")
        
        st.write("### ✓ Checks de Validation")
        
        validation_checks = [
            {"check": "Conservation énergie-impulsion", "status": "✅ PASS", "tolerance": "< 0.1%"},
            {"check": "Unitarité sections efficaces", "status": "✅ PASS", "tolerance": "< 1%"},
            {"check": "Limites infrarouges", "status": "✅ PASS", "tolerance": "Analytique"},
            {"check": "Limites collinéaires", "status": "✅ PASS", "tolerance": "Analytique"},
            {"check": "Cohérence NLO", "status": "⚠️ WARNING", "tolerance": "< 5%"},
            {"check": "Accord avec données", "status": "✅ PASS", "tolerance": "χ²/ndf < 2"}
        ]
        
        df = pd.DataFrame(validation_checks)
        st.dataframe(df, use_container_width=True)

# ==================== PAGE: MODÈLE STANDARD ====================
elif page == "📚 Modèle Standard":
    st.header("📚 Modèle Standard de la Physique des Particules")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ Particules", "🔗 Interactions", "📐 Paramètres", "🧮 Calculs"])
    
    with tab1:
        st.subheader("⚛️ Table des Particules du Modèle Standard")
        
        st.write("### Fermions (Spin 1/2)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Quarks:**")
            
            quarks = [
                {"Gen": "1", "Nom": "up (u)", "Masse": "2.2 MeV", "Charge": "+2/3", "Couleur": "RGB"},
                {"Gen": "1", "Nom": "down (d)", "Masse": "4.7 MeV", "Charge": "-1/3", "Couleur": "RGB"},
                {"Gen": "2", "Nom": "charm (c)", "Masse": "1.28 GeV", "Charge": "+2/3", "Couleur": "RGB"},
                {"Gen": "2", "Nom": "strange (s)", "Masse": "96 MeV", "Charge": "-1/3", "Couleur": "RGB"},
                {"Gen": "3", "Nom": "top (t)", "Masse": "173.0 GeV", "Charge": "+2/3", "Couleur": "RGB"},
                {"Gen": "3", "Nom": "bottom (b)", "Masse": "4.18 GeV", "Charge": "-1/3", "Couleur": "RGB"},
            ]
            
            df_quarks = pd.DataFrame(quarks)
            st.dataframe(df_quarks, use_container_width=True)
        
        with col2:
            st.write("**Leptons:**")
            
            leptons = [
                {"Gen": "1", "Nom": "électron (e⁻)", "Masse": "0.511 MeV", "Charge": "-1"},
                {"Gen": "1", "Nom": "neutrino e (νₑ)", "Masse": "< 1 eV", "Charge": "0"},
                {"Gen": "2", "Nom": "muon (μ⁻)", "Masse": "105.7 MeV", "Charge": "-1"},
                {"Gen": "2", "Nom": "neutrino μ (νμ)", "Masse": "< 0.19 MeV", "Charge": "0"},
                {"Gen": "3", "Nom": "tau (τ⁻)", "Masse": "1.777 GeV", "Charge": "-1"},
                {"Gen": "3", "Nom": "neutrino τ (ντ)", "Masse": "< 18.2 MeV", "Charge": "0"},
            ]
            
            df_leptons = pd.DataFrame(leptons)
            st.dataframe(df_leptons, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### Bosons de Jauge (Spin 1)")
        
        bosons = [
            {"Nom": "Photon (γ)", "Masse": "0", "Charge": "0", "Interaction": "Électromagnétique"},
            {"Nom": "Gluon (g)", "Masse": "0", "Charge": "0", "Interaction": "Forte", "Note": "8 types"},
            {"Nom": "W⁺", "Masse": "80.379 GeV", "Charge": "+1", "Interaction": "Faible"},
            {"Nom": "W⁻", "Masse": "80.379 GeV", "Charge": "-1", "Interaction": "Faible"},
            {"Nom": "Z⁰", "Masse": "91.1876 GeV", "Charge": "0", "Interaction": "Faible"},
        ]
        
        df_bosons = pd.DataFrame(bosons)
        st.dataframe(df_bosons, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### Boson Scalaire (Spin 0)")
        
        higgs = [
            {"Nom": "Higgs (H⁰)", "Masse": "125.10 GeV", "Charge": "0", "Rôle": "Brisure symétrie électrofaible"}
        ]
        
        df_higgs = pd.DataFrame(higgs)
        st.dataframe(df_higgs, use_container_width=True)
    
    with tab2:
        st.subheader("🔗 Forces Fondamentales")
        
        forces = {
            "Force Forte (QCD)": {
                "médiateur": "8 Gluons",
                "portée": "~ 1 fm (10⁻¹⁵ m)",
                "couplage": "αₛ(MZ) = 0.1181",
                "particules": "Quarks, Gluons",
                "propriétés": ["Confinement", "Liberté asymptotique", "Charge de couleur"]
            },
            "Force Électromagnétique": {
                "médiateur": "Photon (γ)",
                "portée": "Infinie",
                "couplage": "α = 1/137.036",
                "particules": "Particules chargées",
                "propriétés": ["Longue portée", "QED", "Renormalisable"]
            },
            "Force Faible": {
                "médiateur": "W±, Z⁰",
                "portée": "~ 10⁻¹⁸ m",
                "couplage": "GF = 1.166×10⁻⁵ GeV⁻²",
                "particules": "Tous les fermions",
                "propriétés": ["Violation CP", "Changement de saveur", "Masse des bosons"]
            },
            "Gravitation": {
                "médiateur": "Graviton (hypothétique)",
                "portée": "Infinie",
                "couplage": "G = 6.674×10⁻¹¹ m³kg⁻¹s⁻²",
                "particules": "Toute masse-énergie",
                "propriétés": ["Non renormalisable", "Très faible", "Non unifiée"]
            }
        }
        
        for force_name, force_info in forces.items():
            with st.expander(f"⚡ {force_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Médiateur:** {force_info['médiateur']}")
                    st.write(f"**Portée:** {force_info['portée']}")
                    st.write(f"**Couplage:** {force_info['couplage']}")
                
                with col2:
                    st.write(f"**Particules affectées:** {force_info['particules']}")
                    st.write("\n**Propriétés:**")
                    for prop in force_info['propriétés']:
                        st.write(f"• {prop}")
        
        st.markdown("---")
        
        # Graphique intensité des forces
        st.write("### 📊 Intensité Relative des Forces")
        
        forces_names = ["Forte", "EM", "Faible", "Gravité"]
        forces_strength = [1, 1/137, 1e-6, 1e-39]
        
        fig = go.Figure(data=[
            go.Bar(x=forces_names, y=forces_strength,
                  marker_color=['red', 'blue', 'green', 'purple'])
        ])
        
        fig.update_layout(
            title="Intensité Relative (échelle log)",
            yaxis_type="log",
            yaxis_title="Intensité relative",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📐 Paramètres Libres du Modèle Standard")
        
        st.write("### 🔢 19 Paramètres Fondamentaux")
        
        st.info("Le Modèle Standard contient 19 paramètres libres qui doivent être mesurés expérimentalement")
        
        parameters = {
            "Masses des Quarks (6)": [
                "mᵤ = 2.2 MeV", "m_d = 4.7 MeV", "m_c = 1.28 GeV",
                "m_s = 96 MeV", "m_t = 173.0 GeV", "m_b = 4.18 GeV"
            ],
            "Masses des Leptons Chargés (3)": [
                "m_e = 0.511 MeV", "m_μ = 105.7 MeV", "m_τ = 1.777 GeV"
            ],
            "Matrice CKM (4)": [
                "θ₁₂ = 13.04°", "θ₂₃ = 2.38°", "θ₁₃ = 0.201°", "δ_CP = 1.20 rad"
            ],
            "Constantes de Couplage (3)": [
                "g₁ (U(1)Y)", "g₂ (SU(2)L)", "g₃ (SU(3)C)"
            ],
            "Paramètres de Higgs (2)": [
                "m_H = 125.10 GeV", "v = 246.22 GeV (VEV)"
            ],
            "Angle θ_QCD (1)": [
                "θ_QCD < 10⁻¹⁰ (problème de la CP forte)"
            ]
        }
        
        for category, params in parameters.items():
            with st.expander(f"📊 {category}"):
                for param in params:
                    st.write(f"• {param}")
        
        st.markdown("---")
        
        st.write("### 🎯 Précision des Mesures")
        
        precision_data = [
            {"Paramètre": "Masse W", "Valeur": "80.379 GeV", "Précision": "0.012 GeV", "Relative": "0.015%"},
            {"Paramètre": "Masse Z", "Valeur": "91.1876 GeV", "Précision": "0.0021 GeV", "Relative": "0.0023%"},
            {"Paramètre": "Masse Top", "Valeur": "173.0 GeV", "Précision": "0.4 GeV", "Relative": "0.23%"},
            {"Paramètre": "Masse Higgs", "Valeur": "125.10 GeV", "Précision": "0.14 GeV", "Relative": "0.11%"},
            {"Paramètre": "αₛ(MZ)", "Valeur": "0.1181", "Précision": "0.0011", "Relative": "0.9%"},
            {"Paramètre": "sin²θW", "Valeur": "0.23122", "Précision": "0.00004", "Relative": "0.017%"},
        ]
        
        df_precision = pd.DataFrame(precision_data)
        st.dataframe(df_precision, use_container_width=True)
    
    with tab4:
        st.subheader("🧮 Calculateurs Physique")
        
        st.write("### ⚡ Calculateur d'Énergie Relativiste")
        
        col1, col2 = st.columns(2)
        
        with col1:
            particle_calc = st.selectbox(
                "Particule",
                ["Electron", "Proton", "Higgs", "Top quark", "W boson"],
                key="calc_particle"
            )
            
            masses_calc = {
                "Electron": 0.000511,
                "Proton": 0.938272,
                "Higgs": 125.10,
                "Top quark": 173.0,
                "W boson": 80.379
            }
            
            mass_calc = masses_calc[particle_calc]
            
            momentum_calc = st.number_input("Impulsion (GeV/c)", 0.0, 10000.0, 100.0, 1.0)
        
        with col2:
            # Calculs
            energy_calc = np.sqrt(momentum_calc**2 + mass_calc**2)
            gamma_calc = energy_calc / mass_calc if mass_calc > 0 else 1
            beta_calc = momentum_calc / energy_calc if energy_calc > 0 else 0
            velocity_calc = beta_calc * 299792458
            
            st.metric("Énergie", f"{energy_calc:.4f} GeV")
            st.metric("γ (gamma)", f"{gamma_calc:.2f}")
            st.metric("β (beta)", f"{beta_calc:.6f}")
            st.metric("Vitesse", f"{velocity_calc:.0f} m/s")
        
        st.markdown("---")
        
        st.write("### 🎯 Calculateur de Section Efficace")
        
        col1, col2 = st.columns(2)
        
        with col1:
            process_calc = st.selectbox(
                "Processus",
                ["pp → H", "pp → tt̄", "pp → ZZ", "pp → WW", "e⁺e⁻ → Z"]
            )
            
            energy_cm_calc = st.number_input("√s (GeV)", 100, 100000, 13000, 100)
        
        with col2:
            # Sections efficaces approximatives
            if process_calc == "pp → H":
                sigma = 50 * (energy_cm_calc / 13000)**0.3
                unit = "pb"
            elif process_calc == "pp → tt̄":
                sigma = 830 * (energy_cm_calc / 13000)**0.3
                unit = "pb"
            elif process_calc == "pp → ZZ":
                sigma = 16 * (energy_cm_calc / 13000)**0.3
                unit = "pb"
            elif process_calc == "pp → WW":
                sigma = 120 * (energy_cm_calc / 13000)**0.3
                unit = "pb"
            else:
                sigma = 41490 * (91.1876 / energy_cm_calc)**2
                unit = "nb"
            
            st.metric("Section Efficace", f"{sigma:.2f} {unit}")
            
            if st.button("📊 Voir Dépendance en Énergie"):
                energies_range = np.linspace(energy_cm_calc * 0.5, energy_cm_calc * 2, 50)
                
                if "pp" in process_calc:
                    sigmas_range = sigma * (energies_range / energy_cm_calc)**0.3
                else:
                    sigmas_range = sigma * (energy_cm_calc / energies_range)**2
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=energies_range, y=sigmas_range,
                    mode='lines',
                    line=dict(color='blue', width=3)
                ))
                
                fig.update_layout(
                    title=f"σ({process_calc}) vs √s",
                    xaxis_title="√s (GeV)",
                    yaxis_title=f"σ ({unit})",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: PHYSIQUE BSM ====================
elif page == "🌌 Physique BSM":
    st.header("🌌 Physique au-delà du Modèle Standard (BSM)")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Théories", "🎯 Recherches", "🌌 Matière Noire"])
    
    with tab1:
        st.subheader("🔍 Théories BSM")
        
        bsm_theories = {
            "Supersymétrie (SUSY)": {
                "description": "Symétrie fermions ↔ bosons",
                "motivation": ["Hiérarchie", "Unification", "Matière noire"],
                "prédictions": ["Sparticules", "Neutralino", "Stop, Gluino"],
                "signatures": ["Jets + MET", "Multi-leptons", "Photons + MET"],
                "status": "Non observée (limites > 2 TeV)"
            },
            "Dimensions Supplémentaires": {
                "description": "Dimensions spatiales supplémentaires compactifiées",
                "motivation": ["Hiérarchie", "Gravité", "Unification"],
                "prédictions": ["Kaluza-Klein", "Mini trous noirs", "Gravitons"],
                "signatures": ["Résonances", "MET", "Dijets"],
                "status": "Limites > 5-10 TeV"
            },
            "Compositeness": {
                "description": "Quarks et leptons sont composites",
                "motivation": ["Hiérarchie", "Nombre de générations"],
                "prédictions": ["Particules excitées", "Leptoquarks", "Contact"],
                "signatures": ["Résonances", "Déviation angulaire"],
                "status": "Limites > 5 TeV"
            },
            "Technicouleur": {
                "description": "EWSB par nouvelle interaction forte",
                "motivation": ["Alternative au Higgs élémentaire"],
                "prédictions": ["Technimésons", "PNGB"],
                "signatures": ["Résonances", "tt̄"],
                "status": "Défavorisée (Higgs découvert)"
            },
            "Grand Unification (GUT)": {
                "description": "Unification des 3 forces",
                "motivation": ["Élégance théorique", "Proton decay"],
                "prédictions": ["Désintégration proton", "Monopôles"],
                "signatures": ["p → e⁺π⁰"],
                "status": "τ_p > 10³⁴ ans"
            },
            "Leptoquarks": {
                "description": "Particules liant quarks et leptons",
                "motivation": ["Unification quark-lepton", "Anomalies saveur"],
                "prédictions": ["Résonances", "LQ → qℓ"],
                "signatures": ["e+jets", "μ+jets"],
                "status": "Limites > 1-2 TeV"
            }
        }
        
        for theory_name, theory_info in bsm_theories.items():
            with st.expander(f"🌌 {theory_name}"):
                st.write(f"**Description:** {theory_info['description']}")
                st.write(f"**Statut:** {theory_info['status']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("\n**Motivations:**")
                    for mot in theory_info['motivation']:
                        st.write(f"• {mot}")
                    
                    st.write("\n**Prédictions:**")
                    for pred in theory_info['prédictions']:
                        st.write(f"• {pred}")
                
                with col2:
                    st.write("\n**Signatures Expérimentales:**")
                    for sig in theory_info['signatures']:
                        st.write(f"• {sig}")
    
    with tab2:
        st.subheader("🎯 Recherches BSM Actives")
        
        st.write("### 🔍 Stratégies de Recherche")
        
        search_strategies = [
            {
                "Type": "Recherche Directe",
                "Cible": "Nouvelles particules",
                "Méthode": "Résonances, bosses",
                "Exemples": "Z', W', Leptoquarks, SUSY",
                "Sensibilité": "Masse < ~7 TeV"
            },
            {
                "Type": "Recherche Indirecte",
                "Cible": "Déviations du MS",
                "Méthode": "Mesures précision",
                "Exemples": "AFB, σ(tt̄), Higgs couplings",
                "Sensibilité": "Λ > 10-100 TeV"
            },
            {
                "Type": "Rare Decays",
                "Cible": "Processus interdits/rares",
                "Méthode": "Branching ratios",
                "Exemples": "B → μμ, μ → eγ",
                "Sensibilité": "Très haute"
            },
            {
                "Type": "Asymétries",
                "Cible": "Violation CP, asymétries",
                "Méthode": "Différences particule/antiparticule",
                "Exemples": "CP violation, AFB",
                "Sensibilité": "Subtile"
            }
        ]
        
        df_strategies = pd.DataFrame(search_strategies)
        st.dataframe(df_strategies, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Limites d'Exclusion BSM")
        
        # Graphique limites
        particles_bsm = ["Z' SSM", "W'", "Gluino", "Stop", "Leptoquark", "q*"]
        mass_limits = [6000, 6500, 2300, 1200, 1800, 7000]
        
        fig = go.Figure(data=[
            go.Bar(x=particles_bsm, y=mass_limits,
                  marker_color='lightcoral',
                  text=[f"{m/1000:.1f} TeV" for m in mass_limits],
                  textposition='outside')
        ])
        
        fig.update_layout(
            title="Limites de Masse à 95% CL (Exemple)",
            xaxis_title="Particule BSM",
            yaxis_title="Limite de masse (GeV)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🌌 Matière Noire")
        
        st.write("### 🔭 Evidence Astrophysique")
        
        evidence = [
            "Courbes de rotation des galaxies",
            "Lentille gravitationnelle",
            "Structure à grande échelle de l'Univers",
            "CMB (Planck)",
            "Amas de galaxies (Bullet Cluster)"
        ]
        
        for ev in evidence:
            st.write(f"✓ {ev}")
        
        st.markdown("---")
        
        st.write("### 🎯 Candidats Matière Noire")
        
        dm_candidates = {
            "WIMP (Weakly Interacting Massive Particle)": {
                "masse": "GeV - TeV",
                "interaction": "Faible",
                "candidat": "Neutralino (SUSY), KK photon",
                "détection": "Directe, Indirecte, Collisionneurs"
            },
            "Axion": {
                "masse": "μeV - meV",
                "interaction": "Très faible",
                "candidat": "Pseudoscalaire",
                "détection": "Cavités résonantes, Hélioscopes"
            },
            "Gravitino": {
                "masse": "Variable",
                "interaction": "Gravitationnelle",
                "candidat": "SUSY (Superpartenaire du graviton)",
                "détection": "Cosmologique"
            },
            "Neutrinos Stériles": {
                "masse": "keV - GeV",
                "interaction": "Mélange avec neutrinos actifs",
                "candidat": "Neutrino droit",
                "détection": "Rayons X, Oscillations"
            }
        }
        
        for dm_name, dm_info in dm_candidates.items():
            with st.expander(f"🌑 {dm_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Masse:** {dm_info['masse']}")
                    st.write(f"**Interaction:** {dm_info['interaction']}")
                
                with col2:
                    st.write(f"**Candidat:** {dm_info['candidat']}")
                    st.write(f"**Détection:** {dm_info['détection']}")
        
        st.markdown("---")
        
        st.write("### 🔬 Recherches au Collisionneur")
        
        st.info("**Signatures:** MET (Énergie Transverse Manquante) + jets/leptons/photons")
        
        dm_searches = [
            "Monojets + MET",
            "Mono-photon + MET",
            "Mono-Z/W + MET",
            "tt̄ + MET",
            "Invisibles Higgs decays"
        ]
        
        for search in dm_searches:
            st.write(f"• {search}")

# ==================== FOOTER & AUTRES PAGES ====================
# elif page in ["📡 Faisceaux & Injection", "🧲 Magnets & RF", "📊 Acquisition de Données", 
#               "🔍 Reconstruction d'Événements", "⚡ Sections Efficaces", "🎲 Générateurs d'Événements",
#               "🔧 Calibration", "💰 Coûts & Budget", "📑 Publications", "🌟 Applications",
#               "🎓 Formation", "🔬 Laboratoires"]:
    
#     st.header(f"{page}")
#     st.info(f"Page {page} - En développement. Structure similaire aux autres pages avec contenu spécialisé.")
    
#     if "Coûts" in page:
#         st.write("### 💰 Analyse Budgétaire")
#         if st.session_state.particle_system['colliders']:
#             total_construction = sum(c['costs']['construction'] for c in st.session_state.particle_system['colliders'].values())
#             total_operation = sum(c['costs']['annual_operation'] for c in st.session_state.particle_system['colliders'].values())
            
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 st.metric("Construction Totale", f"€{total_construction:.0f}M")
#             with col2:
#                 st.metric("Opération Annuelle", f"€{total_operation:.0f}M")
#             with col3:
#                 st.metric("Par Découverte", f"€{total_construction/max(1, len(st.session_state.particle_system['discoveries'])):.0f}M")

# ==================== PAGE: COLLISIONS & LUMINOSITÉ ====================
elif page == "🎯 Collisions & Luminosité":
    st.header("🎯 Gestion des Collisions et Luminosité")
    
    tab1, tab2, tab3 = st.tabs(["💫 Run de Collisions", "📊 Luminosité", "📈 Performance"])
    
    with tab1:
        st.subheader("💫 Lancer un Run de Collisions")
        
        if not st.session_state.particle_system['colliders']:
            st.warning("Aucun collisionneur disponible")
        else:
            collider_ids = list(st.session_state.particle_system['colliders'].keys())
            selected_collider = st.selectbox(
                "Sélectionner Collisionneur",
                collider_ids,
                format_func=lambda x: st.session_state.particle_system['colliders'][x]['name']
            )
            
            collider = st.session_state.particle_system['colliders'][selected_collider]
            
            st.write(f"### ⚛️ {collider['name']}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Énergie CM", f"{collider['specifications']['center_mass_energy']/1000:.1f} TeV")
            with col2:
                st.metric("Luminosité", f"{collider['performance']['luminosity']:.2e} cm⁻²s⁻¹")
            with col3:
                st.metric("Status", collider['status'].upper())
            
            st.markdown("---")
            
            with st.form("collision_run"):
                col1, col2 = st.columns(2)
                
                with col1:
                    run_duration = st.number_input("Durée du Run (heures)", 1, 168, 24, 1)
                    target_lumi = st.number_input("Luminosité Cible (fb⁻¹)", 0.1, 100.0, 10.0, 0.1)
                
                with col2:
                    fill_scheme = st.selectbox("Schéma de Remplissage", 
                                              ["Standard", "High Intensity", "Special Physics"])
                    beta_star = st.number_input("β* (cm)", 10, 200, 55, 5)
                
                submitted = st.form_submit_button("🚀 Démarrer Run", type="primary")
                
                if submitted:
                    with st.spinner("💫 Run en cours..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Simulation du run
                        n_steps = 100
                        lumi_delivered = 0.0
                        
                        for step in range(n_steps):
                            progress_bar.progress((step + 1) / n_steps)
                            status_text.text(f"Heure {step * run_duration / n_steps:.1f}/{run_duration}")
                            
                            # Luminosité instantanée qui décroît
                            lumi_inst = collider['performance']['luminosity'] * np.exp(-step / 50)
                            
                            # Luminosité intégrée
                            lumi_delivered += lumi_inst * run_duration / n_steps * 3600 * 1e-39  # fb⁻¹
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Mise à jour
                        collider['performance']['integrated_luminosity'] += lumi_delivered
                        collider['operations']['hours'] += run_duration
                        collider['operations']['collisions_delivered'] += int(lumi_delivered * 1e15 * 50)  # approximatif
                        
                        st.success(f"✅ Run complété!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Durée", f"{run_duration}h")
                        with col2:
                            st.metric("Lumi Livrée", f"{lumi_delivered:.2f} fb⁻¹")
                        with col3:
                            st.metric("Lumi Totale", f"{collider['performance']['integrated_luminosity']:.2f} fb⁻¹")
                        with col4:
                            st.metric("Efficacité", f"{np.random.uniform(85, 95):.1f}%")

# ==================== PAGE: PUBLICATIONS ====================
elif page == "📑 Publications":
    st.header("📑 Publications Scientifiques")
    
    tab1, tab2, tab3 = st.tabs(["📚 Bibliothèque", "✍️ Nouvelle Publication", "📊 Statistiques"])
    
    with tab1:
        st.subheader("📚 Bibliothèque de Publications")
        
        if 'publications' not in st.session_state.particle_system:
            st.session_state.particle_system['publications'] = []
        
        if st.session_state.particle_system['publications']:
            st.write(f"### 📄 {len(st.session_state.particle_system['publications'])} Publications")
            
            for pub in st.session_state.particle_system['publications']:
                with st.expander(f"📄 {pub['title']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Auteurs:** {pub['authors']}")
                        st.write(f"**Journal:** {pub['journal']}")
                        st.write(f"**Date:** {pub['date']}")
                        st.write(f"**Résumé:** {pub['abstract']}")
                    
                    with col2:
                        st.metric("Citations", pub.get('citations', 0))
                        st.metric("Impact Factor", pub.get('impact_factor', 'N/A'))
                        
                        if st.button("🔗 Lien", key=f"link_{pub['title'][:20]}"):
                            st.info(f"arXiv: {pub.get('arxiv', 'N/A')}")
        else:
            st.info("Aucune publication enregistrée")
    
    with tab2:
        st.subheader("✍️ Enregistrer une Publication")
        
        with st.form("new_publication"):
            title = st.text_input("Titre", "Observation of a new particle in the search for...")
            
            col1, col2 = st.columns(2)
            
            with col1:
                authors = st.text_area("Auteurs", "ATLAS Collaboration")
                journal = st.selectbox("Journal", 
                    ["Physical Review Letters", "Physical Review D", "JHEP", 
                     "Physics Letters B", "European Physical Journal C", "Nature", "Science"])
            
            with col2:
                date_pub = st.date_input("Date Publication", datetime.now())
                arxiv_id = st.text_input("arXiv ID", "2024.12345")
                impact_factor = st.number_input("Impact Factor", 0.0, 100.0, 5.0, 0.1)
            
            abstract = st.text_area("Résumé", 
                "This paper presents the observation of...", 
                height=150)
            
            keywords = st.text_input("Mots-clés (séparés par virgules)", 
                "Higgs boson, LHC, ATLAS")
            
            submitted_pub = st.form_submit_button("📤 Publier", type="primary")
            
            if submitted_pub:
                publication = {
                    'pub_id': f"pub_{len(st.session_state.particle_system['publications']) + 1}",
                    'title': title,
                    'authors': authors,
                    'journal': journal,
                    'date': date_pub.isoformat(),
                    'arxiv': arxiv_id,
                    'impact_factor': impact_factor,
                    'abstract': abstract,
                    'keywords': [k.strip() for k in keywords.split(',')],
                    'citations': 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.particle_system['publications'].append(publication)
                
                st.success("✅ Publication enregistrée!")
                st.balloons()
                
                log_event(f"Publication: {title[:50]}")
    
    with tab3:
        st.subheader("📊 Statistiques de Publication")
        
        if st.session_state.particle_system['publications']:
            n_pubs = len(st.session_state.particle_system['publications'])
            total_citations = sum(p.get('citations', 0) for p in st.session_state.particle_system['publications'])
            avg_impact = np.mean([p.get('impact_factor', 0) for p in st.session_state.particle_system['publications']])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Publications Totales", n_pubs)
            with col2:
                st.metric("Citations Totales", total_citations)
            with col3:
                st.metric("Impact Factor Moyen", f"{avg_impact:.2f}")
            
            st.markdown("---")
            
            # Publications par journal
            st.write("### 📊 Répartition par Journal")
            
            journal_counts = {}
            for pub in st.session_state.particle_system['publications']:
                journal = pub['journal']
                journal_counts[journal] = journal_counts.get(journal, 0) + 1
            
            if journal_counts:
                fig = px.bar(x=list(journal_counts.keys()), y=list(journal_counts.values()),
                           labels={'x': 'Journal', 'y': 'Nombre de Publications'},
                           title="Publications par Journal")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune statistique disponible")

# ==================== PAGE: APPLICATIONS ====================
elif page == "🌟 Applications":
    st.header("🌟 Applications et Retombées")
    
    tab1, tab2, tab3 = st.tabs(["🏥 Médical", "💻 Technologie", "🎓 Éducation"])
    
    with tab1:
        st.subheader("🏥 Applications Médicales")
        
        medical_apps = {
            "Hadronthérapie": {
                "description": "Traitement du cancer par faisceaux de protons/ions",
                "origine": "Accélérateurs de particules",
                "avantages": ["Précision millimétrique", "Moins d'effets secondaires", "Tumeurs profondes"],
                "marché": "~10 milliards €/an",
                "centres": "100+ dans le monde"
            },
            "PET Scan": {
                "description": "Tomographie par émission de positrons",
                "origine": "Détection antimatière",
                "avantages": ["Imagerie fonctionnelle", "Diagnostic précoce", "Oncologie"],
                "marché": "~5 milliards €/an",
                "centres": "5000+ scanners"
            },
            "Radioisotopes Médicaux": {
                "description": "Production isotopes pour diagnostic/thérapie",
                "origine": "Cyclotrons, réacteurs",
                "avantages": ["Médecine nucléaire", "Traceurs", "Thérapie ciblée"],
                "marché": "~8 milliards €/an",
                "centres": "Milliers d'hôpitaux"
            },
            "Détecteurs Médicaux": {
                "description": "Capteurs haute précision pour imagerie",
                "origine": "Détecteurs particules",
                "avantages": ["Haute résolution", "Faible dose", "Temps réel"],
                "marché": "~3 milliards €/an",
                "centres": "Mondial"
            }
        }
        
        for app_name, app_info in medical_apps.items():
            with st.expander(f"🏥 {app_name}"):
                st.write(f"**Description:** {app_info['description']}")
                st.write(f"**Origine:** {app_info['origine']}")
                st.write(f"**Marché:** {app_info['marché']}")
                st.write(f"**Déploiement:** {app_info['centres']}")
                
                st.write("\n**Avantages:**")
                for adv in app_info['avantages']:
                    st.write(f"✓ {adv}")
    
    with tab2:
        st.subheader("💻 Retombées Technologiques")
        
        tech_spinoffs = {
            "World Wide Web": {
                "inventeur": "Tim Berners-Lee (CERN, 1989)",
                "application": "Internet moderne",
                "impact": "Révolution communication mondiale",
                "valeur": "Trillions €"
            },
            "GRID Computing": {
                "inventeur": "CERN + Partenaires",
                "application": "Cloud computing, Big Data",
                "impact": "Infrastructure calcul distribué",
                "valeur": "Centaines milliards €"
            },
            "Supraconducteurs": {
                "inventeur": "Développement accélérateurs",
                "application": "IRM, Maglev, électronique",
                "impact": "Médical, transport, énergie",
                "valeur": "Dizaines milliards €"
            },
            "Détecteurs Silicium": {
                "inventeur": "Physique particules",
                "application": "Caméras, smartphones, auto",
                "impact": "Imagerie numérique",
                "valeur": "Centaines milliards €"
            },
            "Traitement d'Images": {
                "inventeur": "Analyse données HEP",
                "application": "IA, reconnaissance, médical",
                "impact": "Machine Learning",
                "valeur": "Marché en croissance"
            },
            "Cryogénie": {
                "inventeur": "Systèmes He liquide",
                "application": "Industriel, spatial, médical",
                "impact": "Technologies extrêmes",
                "valeur": "Milliards €"
            }
        }
        
        for tech_name, tech_info in tech_spinoffs.items():
            with st.expander(f"💻 {tech_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Inventeur:** {tech_info['inventeur']}")
                    st.write(f"**Application:** {tech_info['application']}")
                
                with col2:
                    st.write(f"**Impact:** {tech_info['impact']}")
                    st.write(f"**Valeur:** {tech_info['valeur']}")
        
        st.markdown("---")
        
        st.info("""
        💡 **Le saviez-vous?**
        
        Le World Wide Web a été inventé au CERN en 1989 pour faciliter le partage 
        d'informations entre physiciens. Aujourd'hui, il génère des trillions d'euros 
        d'activité économique mondiale!
        """)
    
    with tab3:
        st.subheader("🎓 Impact Éducatif et Formation")
        
        st.write("### 📚 Formation")
        
        education_stats = [
            {"Niveau": "Doctorants", "Nombre/an": "~1000", "Domaines": "Physique, Ingénierie, Computing"},
            {"Niveau": "Post-docs", "Nombre/an": "~500", "Domaines": "Recherche fondamentale"},
            {"Niveau": "Ingénieurs", "Nombre/an": "~300", "Domaines": "Technique, R&D"},
            {"Niveau": "Étudiants visiteurs", "Nombre/an": "~5000", "Domaines": "Tous niveaux"},
        ]
        
        df_education = pd.DataFrame(education_stats)
        st.dataframe(df_education, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🌍 Outreach et Sensibilisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Visiteurs CERN:**")
            st.metric("Par an", "~150,000")
            st.write("**Expositions:**")
            st.metric("Visiteurs totaux", "~5 millions")
        
        with col2:
            st.write("**Ressources en ligne:**")
            st.metric("Cours/Vidéos", "1000+")
            st.write("**Collaborations écoles:**")
            st.metric("Écoles partenaires", "5000+")
        
        st.markdown("---")
        
        st.write("### 📖 Matériel Pédagogique")
        
        resources = [
            "🎥 Vidéos éducatives sur YouTube",
            "📱 Applications mobiles interactives",
            "🎮 Jeux éducatifs sur la physique",
            "📚 MOOCs sur la physique des particules",
            "🔬 Kits expérimentaux pour écoles",
            "🌐 Visites virtuelles du LHC"
        ]
        
        for resource in resources:
            st.write(resource)

# ==================== PAGE: FORMATION ====================
elif page == "🎓 Formation":
    st.header("🎓 Formation et Enseignement")
    
    tab1, tab2, tab3 = st.tabs(["📚 Cours", "🏫 Écoles", "🎯 Tutoriels"])
    
    with tab1:
        st.subheader("📚 Bibliothèque de Cours")
        
        courses = {
            "Introduction à la Physique des Particules": {
                "niveau": "Débutant",
                "durée": "20 heures",
                "sujets": ["Modèle Standard", "Particules élémentaires", "Forces fondamentales"],
                "prérequis": "Physique de base"
            },
            "Théorie Quantique des Champs": {
                "niveau": "Avancé",
                "durée": "60 heures",
                "sujets": ["QED", "QCD", "Théorie électrofaible", "Renormalisation"],
                "prérequis": "Mécanique quantique, Relativité"
            },
            "Physique Expérimentale HEP": {
                "niveau": "Intermédiaire",
                "durée": "40 heures",
                "sujets": ["Détecteurs", "Accélérateurs", "Analyse données", "Statistiques"],
                "prérequis": "Physique particules de base"
            },
            "Phénoménologie du Modèle Standard": {
                "niveau": "Avancé",
                "durée": "50 heures",
                "sujets": ["Sections efficaces", "Décroissances", "Corrections radiatives"],
                "prérequis": "QFT, Modèle Standard"
            },
            "Physique BSM": {
                "niveau": "Expert",
                "durée": "40 heures",
                "sujets": ["SUSY", "Dimensions extra", "Matière noire", "Unification"],
                "prérequis": "Phénoménologie MS"
            }
        }
        
        for course_name, course_info in courses.items():
            with st.expander(f"📖 {course_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Niveau:** {course_info['niveau']}")
                    st.write(f"**Durée:** {course_info['durée']}")
                    st.write(f"**Prérequis:** {course_info['prérequis']}")
                
                with col2:
                    st.write("**Sujets couverts:**")
                    for sujet in course_info['sujets']:
                        st.write(f"• {sujet}")
                
                if st.button("📥 S'inscrire", key=f"enroll_{course_name}"):
                    st.success(f"✅ Inscrit à '{course_name}'")
    
    with tab2:
        st.subheader("🏫 Écoles d'Été et Workshops")
        
        schools = [
            {
                "nom": "CERN Summer Student Programme",
                "dates": "Juin-Août",
                "durée": "8-13 semaines",
                "participants": "~300",
                "niveau": "Étudiants licence/master"
            },
            {
                "nom": "CERN School of Computing",
                "dates": "Septembre",
                "durée": "2 semaines",
                "participants": "~100",
                "niveau": "Doctorants, post-docs"
            },
            {
                "nom": "European School of High-Energy Physics",
                "dates": "Juin",
                "durée": "2 semaines",
                "participants": "~120",
                "niveau": "Doctorants"
            },
            {
                "nom": "Latin American School (CLASHEP)",
                "dates": "Mars",
                "durée": "2 semaines",
                "participants": "~80",
                "niveau": "Doctorants"
            }
        ]
        
        for school in schools:
            with st.expander(f"🏫 {school['nom']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Dates:** {school['dates']}")
                    st.write(f"**Durée:** {school['durée']}")
                
                with col2:
                    st.write(f"**Participants:** {school['participants']}")
                    st.write(f"**Niveau:** {school['niveau']}")
    
    with tab3:
        st.subheader("🎯 Tutoriels Pratiques")
        
        tutorials = [
            "Installation et configuration ROOT",
            "Analyse de données avec Python/ROOT",
            "Introduction à RooFit",
            "Machine Learning pour HEP",
            "Générateurs Monte Carlo (PYTHIA)",
            "Visualisation événements",
            "Statistiques pour physiciens",
            "Grid Computing"
        ]
        
        selected_tutorial = st.selectbox("Sélectionner un tutoriel", tutorials)
        
        st.write(f"### 📖 {selected_tutorial}")
        
        st.info("Tutoriel interactif disponible avec exemples de code et exercices")
        
        if st.button("▶️ Commencer le Tutoriel"):
            st.success("Tutoriel démarré!")

# ==================== PAGE: LABORATOIRES ====================
elif page == "🔬 Laboratoires":
    st.header("🔬 Laboratoires et Infrastructures")
    
    tab1, tab2, tab3 = st.tabs(["🌍 Centres Mondiaux", "🤝 Collaborations", "📡 Installations"])
    
    with tab1:
        st.subheader("🌍 Grands Centres de Recherche")
        
        labs = {
            "CERN": {
                "nom_complet": "Organisation Européenne pour la Recherche Nucléaire",
                "localisation": "Genève, Suisse/France",
                "fondation": "1954",
                "membres": "23 États membres",
                "installations": ["LHC", "SPS", "PS", "ISOLDE"],
                "personnel": "~3000 + 17000 visiteurs"
            },
            "Fermilab": {
                "nom_complet": "Fermi National Accelerator Laboratory",
                "localisation": "Illinois, USA",
                "fondation": "1967",
                "membres": "DOE USA",
                "installations": ["Tevatron (arrêté)", "NOvA", "Muon g-2"],
                "personnel": "~1800"
            },
            "SLAC": {
                "nom_complet": "Stanford Linear Accelerator Center",
                "localisation": "Californie, USA",
                "fondation": "1962",
                "membres": "Stanford University, DOE",
                "installations": ["LCLS", "FACET", "PEP-II (arrêté)"],
                "personnel": "~1600"
            },
            "DESY": {
                "nom_complet": "Deutsches Elektronen-Synchrotron",
                "localisation": "Hambourg, Allemagne",
                "fondation": "1959",
                "membres": "Allemagne",
                "installations": ["PETRA", "FLASH", "European XFEL"],
                "personnel": "~2300"
            },
            "KEK": {
                "nom_complet": "High Energy Accelerator Research Organization",
                "localisation": "Tsukuba, Japon",
                "fondation": "1997",
                "membres": "Japon",
                "installations": ["SuperKEKB", "J-PARC"],
                "personnel": "~700"
            }
        }
        
        for lab_name, lab_info in labs.items():
            with st.expander(f"🔬 {lab_name} - {lab_info['nom_complet']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Localisation:** {lab_info['localisation']}")
                    st.write(f"**Fondation:** {lab_info['fondation']}")
                    st.write(f"**Membres:** {lab_info['membres']}")
                
                with col2:
                    st.write(f"**Personnel:** {lab_info['personnel']}")
                    st.write("\n**Installations principales:**")
                    for install in lab_info['installations']:
                        st.write(f"• {install}")
    
    with tab2:
        st.subheader("🤝 Grandes Collaborations")
        
        collaborations = [
            {
                "nom": "ATLAS",
                "type": "Détecteur LHC",
                "membres": "~3000 physiciens",
                "institutions": "183 institutions, 42 pays",
                "objectifs": "Physique Higgs, recherche BSM, précision MS"
            },
            {
                "nom": "CMS",
                "type": "Détecteur LHC",
                "membres": "~4000 physiciens",
                "institutions": "230 institutions, 50 pays",
                "objectifs": "Higgs, Top, BSM, QCD"
            },
            {
                "nom": "ALICE",
                "type": "Détecteur LHC (ions lourds)",
                "membres": "~1800 physiciens",
                "institutions": "175 institutions, 41 pays",
                "objectifs": "Plasma quark-gluon, QCD"
            },
            {
                "nom": "LHCb",
                "type": "Détecteur LHC (saveur)",
                "membres": "~1400 physiciens",
                "institutions": "82 institutions, 18 pays",
                "objectifs": "Physique B, violation CP, saveur"
            }
        ]
        
        for collab in collaborations:
            with st.expander(f"🤝 {collab['nom']} ({collab['type']})"):
                st.write(f"**Membres:** {collab['membres']}")
                st.write(f"**Institutions:** {collab['institutions']}")
                st.write(f"**Objectifs scientifiques:** {collab['objectifs']}")
    
    with tab3:
        st.subheader("📡 Installations Majeures")
        
        facilities = {
            "Collisionneurs en Opération": [
                {"Nom": "LHC (CERN)", "Type": "pp", "Énergie": "13.6 TeV", "Lumi": "2×10³⁴"},
                {"Nom": "SuperKEKB (KEK)", "Type": "e⁺e⁻", "Énergie": "10.58 GeV", "Lumi": "4×10³⁵"},
                {"Nom": "RHIC (BNL)", "Type": "Ion lourd", "Énergie": "510 GeV", "Lumi": "Variable"},
            ],
            "Projets Futurs": [
                {"Nom": "HL-LHC", "Type": "pp (upgrade)", "Énergie": "14 TeV", "Lumi": "7.5×10³⁴", "Démarrage": "2029"},
                {"Nom": "FCC-ee", "Type": "e⁺e⁻", "Énergie": "91-365 GeV", "Lumi": "Variable", "Démarrage": "2045?"},
                {"Nom": "ILC", "Type": "e⁺e⁻ linéaire", "Énergie": "250-500 GeV", "Lumi": "1.8×10³⁴", "Démarrage": "TBD"},
                {"Nom": "CLIC", "Type": "e⁺e⁻ linéaire", "Énergie": "380-3000 GeV", "Lumi": "Variable", "Démarrage": "TBD"},
            ]
        }
        
        for category, machines in facilities.items():
            st.write(f"### {category}")
            df_facilities = pd.DataFrame(machines)
            st.dataframe(df_facilities, use_container_width=True)
            st.markdown("---")

# ==================== PAGE: GÉNÉRATEURS D'ÉVÉNEMENTS ====================
elif page == "🎲 Générateurs d'Événements":
    st.header("🎲 Générateurs d'Événements Monte Carlo")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "▶️ Production", "✅ Validation"])
    
    with tab1:
        st.subheader("🔧 Configuration des Générateurs")
        
        st.write("### 🎯 Sélection du Générateur")
        
        generators_config = {
            "PYTHIA 8": {
                "type": "Parton Shower + Hadronisation",
                "processes": ["Tous processus 2→2", "Décroissances"],
                "tunes": ["Monash", "4C", "A14"],
                "pdf": ["NNPDF2.3", "CTEQ6L1"],
                "version": "8.310"
            },
            "HERWIG 7": {
                "type": "Parton Shower angulaire",
                "processes": ["QCD", "EW", "Higgs"],
                "tunes": ["Default", "LHC-UE7"],
                "pdf": ["MMHT2014", "CT14"],
                "version": "7.2.3"
            },
            "MadGraph5": {
                "type": "Matrix Element (LO/NLO)",
                "processes": ["Processus multi-jambes", "BSM"],
                "matching": ["MLM", "FxFx", "UNLOPS"],
                "pdf": ["NNPDF3.1", "CT18"],
                "version": "3.5.0"
            },
            "POWHEG": {
                "type": "NLO + PS matching",
                "processes": ["SM NLO", "Higgs", "Top"],
                "matching": ["Automatique"],
                "pdf": ["NNPDF3.1"],
                "version": "V2"
            },
            "SHERPA": {
                "type": "Multi-purpose ME+PS",
                "processes": ["LO/NLO", "Multi-jet merging"],
                "matching": ["CKKW", "MEPS@NLO"],
                "pdf": ["NNPDF3.0"],
                "version": "2.2.15"
            }
        }
        
        selected_gen = st.selectbox("Générateur Principal", list(generators_config.keys()))
        
        gen_info = generators_config[selected_gen]
        
        with st.expander(f"ℹ️ Détails {selected_gen}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {gen_info['type']}")
                st.write(f"**Version:** {gen_info['version']}")
                
                st.write("\n**Processus disponibles:**")
                for proc in gen_info['processes']:
                    st.write(f"• {proc}")
            
            with col2:
                if 'tunes' in gen_info:
                    tune_selected = st.selectbox("Tune", gen_info['tunes'])
                
                if 'pdf' in gen_info:
                    pdf_selected = st.selectbox("PDF Set", gen_info['pdf'])
                
                if 'matching' in gen_info:
                    st.write("\n**Matching/Merging:**")
                    for match in gen_info['matching']:
                        st.write(f"• {match}")
        
        st.markdown("---")
        
        st.write("### ⚙️ Paramètres de Génération")
        
        with st.form("generator_params"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_events_gen = st.number_input("Nombre d'événements", 100, 10000000, 100000, 1000)
                random_seed = st.number_input("Seed aléatoire", 0, 999999, 12345)
            
            with col2:
                energy_gen = st.number_input("√s (GeV)", 1000, 100000, 13000, 100)
                beam_type = st.selectbox("Type de faisceau", ["pp", "pp̄", "e⁺e⁻", "ep"])
            
            with col3:
                alpha_s_gen = st.number_input("αₛ(MZ)", 0.10, 0.13, 0.118, 0.001)
                shower_on = st.checkbox("Parton Shower", value=True)
                hadron_on = st.checkbox("Hadronisation", value=True)
            
            submitted_gen = st.form_submit_button("💾 Sauvegarder Configuration")
            
            if submitted_gen:
                st.success("✅ Configuration sauvegardée!")
    
    with tab2:
        st.subheader("▶️ Production d'Événements")
        
        st.write("### 🚀 Lancer la Production")
        
        with st.form("run_generation"):
            process_gen = st.selectbox(
                "Processus à Générer",
                ["gg → H → γγ", "gg → H → ZZ* → 4l", "qq̄ → tt̄",
                 "qq̄' → W → lν", "qq̄ → Z → l⁺l⁻", "gg → ZZ",
                 "qq̄ → WW", "pp → jj (QCD)", "pp → SUSY"]
            )
            
            n_events_prod = st.number_input("Événements à produire", 1000, 10000000, 100000, 1000)
            
            col1, col2 = st.columns(2)
            
            with col1:
                filter_cuts = st.checkbox("Appliquer filtres", value=False)
                if filter_cuts:
                    pt_min = st.number_input("pT min (GeV)", 0, 500, 20, 5)
                    eta_max = st.number_input("|η| max", 0.0, 5.0, 2.5, 0.1)
            
            with col2:
                output_format = st.selectbox("Format sortie", ["HepMC", "LHE", "ROOT", "HEPEVT"])
                n_jobs = st.number_input("Jobs parallèles", 1, 1000, 10, 1)
            
            run_gen = st.form_submit_button("🚀 Lancer Production", type="primary")
            
            if run_gen:
                with st.spinner(f"Production de {n_events_prod:,} événements..."):
                    progress_bar = st.progress(0)
                    
                    # Simulation production
                    n_steps = 100
                    for i in range(n_steps):
                        progress_bar.progress((i + 1) / n_steps)
                    
                    progress_bar.empty()
                    
                    st.success(f"✅ Production terminée: {n_events_prod:,} événements générés")
                    
                    # Statistiques
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Événements", f"{n_events_prod:,}")
                    with col2:
                        st.metric("Temps CPU", f"{n_events_prod/1000:.1f} h")
                    with col3:
                        filter_eff = 0.85 if filter_cuts else 1.0
                        st.metric("Efficacité", f"{filter_eff:.1%}")
                    with col4:
                        size_mb = n_events_prod * 0.5 / 1000  # MB
                        st.metric("Taille", f"{size_mb:.1f} GB")
                    
                    log_event(f"Production MC: {process_gen} - {n_events_prod:,} événements")
        
        st.markdown("---")
        
        st.write("### 📊 Historique de Production")
        
        if st.session_state.particle_system['simulations']:
            production_history = []
            for sim in st.session_state.particle_system['simulations']:
                production_history.append({
                    'ID': sim['sim_id'],
                    'Processus': sim['process'],
                    'Générateur': sim['generator'],
                    'Événements': f"{sim['n_events']:,}",
                    'Date': sim['timestamp'][:10]
                })
            
            df_prod = pd.DataFrame(production_history)
            st.dataframe(df_prod, use_container_width=True)
        else:
            st.info("Aucune production enregistrée")
    
    with tab3:
        st.subheader("✅ Validation des Événements")
        
        st.write("### 🔍 Checks de Validation")
        
        validation_checks = [
            {"Check": "Conservation 4-impulsion", "Status": "✅ PASS", "Tolérance": "< 0.1%"},
            {"Check": "Unitarité", "Status": "✅ PASS", "Tolérance": "< 1%"},
            {"Check": "Limites IR/Collinéaire", "Status": "✅ PASS", "Tolérance": "Analytique"},
            {"Check": "Normalisation section efficace", "Status": "✅ PASS", "Tolérance": "< 1%"},
            {"Check": "Distributions physiques", "Status": "✅ PASS", "Tolérance": "Visuel"},
            {"Check": "Pas de poids négatifs", "Status": "⚠️ WARNING", "Tolérance": "< 5% négatifs"},
        ]
        
        df_validation = pd.DataFrame(validation_checks)
        st.dataframe(df_validation, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Distributions de Contrôle")
        
        if st.session_state.particle_system['simulations']:
            # Sélection simulation
            sim_ids = [s['sim_id'] for s in st.session_state.particle_system['simulations']]
            selected_sim_val = st.selectbox("Simulation à valider", sim_ids,
                format_func=lambda x: next(s['process'] for s in st.session_state.particle_system['simulations'] if s['sim_id'] == x))
            
            simulation = next(s for s in st.session_state.particle_system['simulations'] if s['sim_id'] == selected_sim_val)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution masse invariante
                masses = np.random.normal(125, 2, 1000)
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=masses, nbinsx=50, marker_color='blue'))
                
                fig.update_layout(
                    title="Distribution Masse Invariante",
                    xaxis_title="m (GeV)",
                    yaxis_title="Événements",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution poids
                weights = np.concatenate([np.random.normal(1, 0.1, 950), 
                                        np.random.uniform(-0.5, 0, 50)])
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=weights, nbinsx=50, marker_color='green'))
                
                fig.update_layout(
                    title="Distribution des Poids",
                    xaxis_title="Poids",
                    yaxis_title="Événements",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune simulation disponible pour validation")

# ==================== PAGE: CALIBRATION ====================
elif page == "🔧 Calibration":
    st.header("🔧 Calibration des Détecteurs")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Procédures", "📊 Monitoring", "✅ Validation"])
    
    with tab1:
        st.subheader("🎯 Procédures de Calibration")
        
        st.write("### 🔬 Types de Calibration")
        
        calibration_types = {
            "Énergie EM": {
                "Méthode": "Z → e⁺e⁻, E/p",
                "Précision": "< 0.5%",
                "Fréquence": "Quotidienne",
                "Outils": "Électrons, Photons"
            },
            "Échelle Jets": {
                "Méthode": "γ+jet, Z+jet balance",
                "Précision": "1-3%",
                "Fréquence": "Hebdomadaire",
                "Outils": "Pythia, Herwig"
            },
            "Énergie HAD": {
                "Méthode": "Single particle response",
                "Précision": "3-5%",
                "Fréquence": "Par run",
                "Outils": "Test beam, pions"
            },
            "Moment Muons": {
                "Méthode": "J/ψ, Z → μ⁺μ⁻",
                "Précision": "< 0.1%",
                "Fréquence": "Quotidienne",
                "Outils": "Résonances"
            },
            "MET": {
                "Méthode": "Balance pT, Z → νν",
                "Précision": "2-5%",
                "Fréquence": "Par période",
                "Outils": "Simulation"
            }
        }
        
        for calib_name, calib_info in calibration_types.items():
            with st.expander(f"🔧 {calib_name}"):
                for key, value in calib_info.items():
                    st.write(f"**{key}:** {value}")
        
        st.markdown("---")
        
        st.write("### 📐 Calibration Échelle Énergie EM")
        
        with st.form("em_calibration"):
            st.write("**Utilisation des Z → e⁺e⁻**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_zee_events = st.number_input("Événements Z→ee", 1000, 1000000, 100000)
                barrel_endcap = st.selectbox("Région", ["Barrel", "Endcap", "Les deux"])
            
            with col2:
                target_mass = st.number_input("Masse Z cible (GeV)", 90.0, 92.0, 91.1876, 0.0001)
                max_deviation = st.slider("Déviation max (%)", 0.1, 5.0, 0.5, 0.1)
            
            if st.form_submit_button("🔧 Lancer Calibration"):
                with st.spinner("Calibration en cours..."):
                    progress = st.progress(0)
                    for i in range(100):
                        progress.progress(i + 1)
                    
                    # Résultats simulés
                    scale_factor = 1.0 + np.random.uniform(-0.002, 0.002)
                    uncertainty = np.random.uniform(0.001, 0.005)
                    
                    st.success("✅ Calibration terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Scale Factor", f"{scale_factor:.5f}")
                    with col2:
                        st.metric("Incertitude", f"±{uncertainty:.4f}")
                    with col3:
                        st.metric("χ²/ndf", f"{np.random.uniform(0.9, 1.1):.2f}")
                    
                    log_event(f"Calibration EM: SF={scale_factor:.5f}")
    
    with tab2:
        st.subheader("📊 Monitoring des Calibrations")
        
        st.write("### 📈 Évolution Temporelle")
        
        # Simulation évolution calibration
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        scale_factors = 1.0 + np.random.randn(30) * 0.001
        scale_factors = scale_factors.cumsum() * 0.0001 + 1.0
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=scale_factors,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                     annotation_text="Nominal")
        
        fig.update_layout(
            title="Évolution du Scale Factor EM",
            xaxis_title="Date",
            yaxis_title="Scale Factor",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Stabilité par Run")
        
        run_numbers = [f"Run {300000+i}" for i in range(20)]
        stability = np.random.uniform(0.9995, 1.0005, 20)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=run_numbers, y=stability,
            mode='markers',
            marker=dict(size=10, color='green')
        ))
        
        fig.add_hrect(y0=0.999, y1=1.001, fillcolor="lightgreen", opacity=0.2,
                     annotation_text="Tolérance", annotation_position="top right")
        
        fig.update_layout(
            title="Stabilité Calibration par Run",
            xaxis_title="Run",
            yaxis_title="Facteur Normalisé",
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("✅ Validation des Calibrations")
        
        st.write("### 🎯 Tests de Validation")
        
        validation_tests = [
            {"Test": "Masse Z peak", "Valeur Mesurée": "91.188 GeV", "Référence": "91.1876 GeV", "Status": "✅"},
            {"Test": "Largeur Z", "Valeur Mesurée": "2.495 GeV", "Référence": "2.4952 GeV", "Status": "✅"},
            {"Test": "Masse J/ψ", "Valeur Mesurée": "3.097 GeV", "Référence": "3.0969 GeV", "Status": "✅"},
            {"Test": "E/p électrons", "Valeur Mesurée": "1.002", "Référence": "1.000", "Status": "✅"},
            {"Test": "η symétrie", "Valeur Mesurée": "< 0.5%", "Référence": "< 1%", "Status": "✅"},
        ]
        
        df_val = pd.DataFrame(validation_tests)
        st.dataframe(df_val, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Distribution Masse Z")
        
        # Simulation pic Z
        mass_range = np.linspace(70, 110, 200)
        signal = 10000 * np.exp(-0.5 * ((mass_range - 91.19) / 2.5)**2)
        background = 100 * np.exp(-(mass_range - 70) / 15)
        data = signal + background + np.random.randn(200) * 30
        
        fig = go.Figure()
        
        # Données
        fig.add_trace(go.Scatter(
            x=mass_range, y=data,
            mode='markers',
            marker=dict(color='black', size=4),
            name='Data'
        ))
        
        # Fit
        fig.add_trace(go.Scatter(
            x=mass_range, y=signal + background,
            mode='lines',
            line=dict(color='red', width=2),
            name='Fit'
        ))
        
        fig.update_layout(
            title="Masse Invariante e⁺e⁻ (Calibration)",
            xaxis_title="m_ee (GeV)",
            yaxis_title="Événements / 0.2 GeV",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Calibration validée: m_Z = 91.188 ± 0.002 GeV")

# ==================== PAGE: COÛTS & BUDGET ====================
elif page == "💰 Coûts & Budget":
    st.header("💰 Analyse des Coûts et Budget")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💵 Construction", "🔄 Opération", "📊 ROI", "📈 Projections"])
    
    with tab1:
        st.subheader("💵 Coûts de Construction")
        
        st.write("### 🏗️ Décomposition des Coûts")
        
        if st.session_state.particle_system['colliders']:
            total_construction = sum(c['costs']['construction'] for c in st.session_state.particle_system['colliders'].values())
            
            st.metric("Coût Total Construction", f"€{total_construction:,.0f}M")
            
            st.markdown("---")
            
            # Répartition typique
            construction_breakdown = {
                "Génie Civil & Tunnel": 25,
                "Aimants Supraconducteurs": 30,
                "Système Cryogénique": 10,
                "Cavités RF": 5,
                "Système de Vide": 5,
                "Détecteurs": 15,
                "Infrastructure & Services": 10
            }
            
            fig = px.pie(values=list(construction_breakdown.values()),
                        names=list(construction_breakdown.keys()),
                        title="Répartition des Coûts de Construction (%)",
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            st.write("### 📊 Détails par Système")
            
            systems_cost = []
            for system, percentage in construction_breakdown.items():
                cost = total_construction * percentage / 100
                systems_cost.append({
                    'Système': system,
                    'Coût (M€)': f"{cost:.1f}",
                    'Pourcentage': f"{percentage}%"
                })
            
            df_systems = pd.DataFrame(systems_cost)
            st.dataframe(df_systems, use_container_width=True)
        else:
            st.info("Aucun collisionneur créé")
    
    with tab2:
        st.subheader("🔄 Coûts d'Opération")
        
        if st.session_state.particle_system['colliders']:
            st.write("### 💸 Coûts Annuels")
            
            total_operation = sum(c['costs']['annual_operation'] for c in st.session_state.particle_system['colliders'].values())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Opération Annuelle", f"€{total_operation:.0f}M")
            with col2:
                st.metric("Par Jour", f"€{total_operation*1000/365:.0f}k")
            with col3:
                st.metric("Par Heure", f"€{total_operation*1000/8760:.0f}k")
            
            st.markdown("---")
            
            # Répartition opération
            operation_breakdown = {
                "Électricité": 45,
                "Personnel": 35,
                "Maintenance": 10,
                "Cryogénie (He)": 5,
                "Computing": 5
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(values=list(operation_breakdown.values()),
                            names=list(operation_breakdown.keys()),
                            title="Répartition Coûts Opérationnels (%)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                operation_details = []
                for item, percentage in operation_breakdown.items():
                    cost = total_operation * percentage / 100
                    operation_details.append({
                        'Poste': item,
                        'Coût Annual (M€)': f"{cost:.1f}",
                        '%': f"{percentage}%"
                    })
                
                df_operation = pd.DataFrame(operation_details)
                st.dataframe(df_operation, use_container_width=True)
            
            st.markdown("---")
            
            st.write("### ⚡ Détail Consommation Électrique")
            
            power_cost_data = [
                {"Système": "Aimants", "Puissance (MW)": "120", "€/an (M)": "52.6"},
                {"Système": "Cryogénie", "Puissance (MW)": "30", "€/an (M)": "13.1"},
                {"Système": "RF", "Puissance (MW)": "10", "€/an (M)": "4.4"},
                {"Système": "Détecteurs", "Puissance (MW)": "20", "€/an (M)": "8.8"},
                {"Système": "Infrastructure", "Puissance (MW)": "20", "€/an (M)": "8.8"},
            ]
            
            df_power = pd.DataFrame(power_cost_data)
            st.dataframe(df_power, use_container_width=True)
            
            st.info("💡 Prix électricité: ~0.05 €/kWh (moyenne industrielle)")
        else:
            st.info("Aucun collisionneur créé")
    
    with tab3:
        st.subheader("📊 Retour sur Investissement")
        
        st.write("### 🎯 Bénéfices Scientifiques et Économiques")
        
        if st.session_state.particle_system['colliders']:
            total_investment = sum(c['costs']['construction'] + c['costs']['annual_operation'] * 10 
                                  for c in st.session_state.particle_system['colliders'].values())
            
            st.metric("Investissement 10 ans", f"€{total_investment:,.0f}M")
            
            st.markdown("---")
            
            st.write("### 💡 Impacts")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📚 Impact Scientifique:**")
                st.write("• Publications scientifiques: ~2000/an")
                st.write("• Citations: ~100,000/an")
                st.write("• Découvertes majeures: 1-5")
                st.write("• Prix Nobel potentiels: 1-3")
                st.write("• Formation jeunes chercheurs: ~1000/an")
            
            with col2:
                st.write("**💼 Impact Économique:**")
                st.write("• Emplois directs: ~3,000")
                st.write("• Emplois indirects: ~10,000")
                st.write("• Retombées technologiques: ~500M€")
                st.write("• Brevets: ~50/an")
                st.write("• Spin-offs: ~10 entreprises")
            
            st.markdown("---")
            
            st.write("### 🌐 Retombées Technologiques")
            
            spinoffs = [
                {"Domaine": "Médical", "Technologie": "PET scan, hadronthérapie", "Marché": "10 Mrd €/an"},
                {"Domaine": "Computing", "Technologie": "GRID, Cloud, Big Data", "Marché": "5 Mrd €/an"},
                {"Domaine": "Instrumentation", "Technologie": "Détecteurs, électronique", "Marché": "2 Mrd €/an"},
                {"Domaine": "Supraconducteurs", "Technologie": "Aimants, câbles", "Marché": "3 Mrd €/an"},
                {"Domaine": "Cryogénie", "Technologie": "Systèmes He", "Marché": "1 Mrd €/an"},
            ]
            
            df_spinoffs = pd.DataFrame(spinoffs)
            st.dataframe(df_spinoffs, use_container_width=True)
            
            st.success("💡 Multiplicateur économique estimé: 1€ investi → 3-5€ de retombées")
        else:
            st.info("Aucun collisionneur créé")
    
    with tab4:
        st.subheader("📈 Projections Budgétaires")
        
        st.write("### 📊 Évolution Budget sur 20 ans")
        
        years = np.arange(2025, 2046)
        
        # Phase construction (5 ans)
        construction_phase = np.linspace(0, 5000, 5)
        construction_phase = np.concatenate([construction_phase, np.zeros(15)])
        
        # Phase opération
        operation_phase = np.concatenate([np.zeros(5), np.full(15, 500)])
        
        # Upgrades
        upgrades_phase = np.zeros(20)
        upgrades_phase[[8, 15]] = [1000, 1500]
        
        # Total
        total_budget = construction_phase + operation_phase + upgrades_phase
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years, y=construction_phase,
            name='Construction',
            marker_color='blue'
        ))
        
        fig.add_trace(go.Bar(
            x=years, y=operation_phase,
            name='Opération',
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            x=years, y=upgrades_phase,
            name='Upgrades',
            marker_color='orange'
        ))
        
        fig.update_layout(
            title="Projection Budget 2025-2045",
            xaxis_title="Année",
            yaxis_title="Budget (M€)",
            barmode='stack',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Construction", f"€{construction_phase.sum():,.0f}M")
        with col2:
            st.metric("Total Opération", f"€{operation_phase.sum():,.0f}M")
        with col3:
            st.metric("Total 20 ans", f"€{total_budget.sum():,.0f}M")

# ==================== PAGE: RECONSTRUCTION D'ÉVÉNEMENTS ====================
elif page == "🔍 Reconstruction d'Événements":
    st.header("🔍 Reconstruction d'Événements")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Traces", "⚡ Calorimétrie", "🔗 Objets Physiques", "👁️ Visualisation"])
    
    with tab1:
        st.subheader("🎯 Reconstruction de Traces")
        
        st.write("### 🧭 Algorithmes de Tracking")
        
        st.info("""
        **Objectif:** Reconstruire trajectoires de particules chargées
        
        **Challenges:**
        - Haute multiplicité (~1000 traces/événement)
        - Bruit de fond
        - Interactions matériau
        - Efficacité et pureté
        """)
        
        tracking_algos = [
            {"Algorithme": "Kalman Filter", "Efficacité": "95%", "Fake Rate": "5%", "CPU": "Moyen"},
            {"Algorithme": "Cellular Automaton", "Efficacité": "93%", "Fake Rate": "3%", "CPU": "Rapide"},
            {"Algorithme": "Hough Transform", "Efficacité": "90%", "Fake Rate": "8%", "CPU": "Lent"},
            {"Algorithme": "Neural Network", "Efficacité": "96%", "Fake Rate": "4%", "CPU": "Variable"},
        ]
        
        df_tracking = pd.DataFrame(tracking_algos)
        st.dataframe(df_tracking, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Performance du Tracking")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Efficacité", "95%")
        with col2:
            st.metric("Résolution pT", "1-2%")
        with col3:
            st.metric("Résolution d₀", "10-20 μm")
        with col4:
            st.metric("Traces/Evt", "~1000")
        
        st.markdown("---")
        
        # Visualisation trace hélicoïdale
        st.write("### 🌀 Trajectoire dans le Champ Magnétique")
        
        # Simulation hélice
        t = np.linspace(0, 4*np.pi, 1000)
        pT = 50  # GeV
        B = 3.8  # Tesla
        R = pT / (0.3 * B) * 1000  # rayon en mm
        
        x = R * np.cos(t)
        y = R * np.sin(t)
        z = t * 50  # pitch
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='blue', width=4)
        )])
        
        fig.update_layout(
            title=f"Trajectoire Hélicoïdale (pT = {pT} GeV)",
            scene=dict(
                xaxis_title="x (mm)",
                yaxis_title="y (mm)",
                zaxis_title="z (mm)"
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("⚡ Reconstruction Calorimétrique")
        
        st.write("### 🔥 Clustering")
        
        clustering_algos = {
            "Topological": {
                "Description": "Clusters topologiques basés cellules voisines",
                "Seuils": "Signal > 4σ, Voisin > 2σ",
                "Efficacité": "Haute",
                "Usage": "Électrons, photons"
            },
            "Sliding Window": {
                "Description": "Fenêtre glissante taille fixe",
                "Seuils": "Grille 3×3, 5×5, 7×7",
                "Efficacité": "Bonne",
                "Usage": "Électrons, photons"
            },
            "Particle Flow": {
                "Description": "Combinaison tracker + calo",
                "Seuils": "Variable",
                "Efficacité": "Optimale",
                "Usage": "Jets, MET"
            }
        }
        
        for algo_name, algo_info in clustering_algos.items():
            with st.expander(f"🔥 {algo_name}"):
                for key, value in algo_info.items():
                    st.write(f"**{key}:** {value}")
        
        st.markdown("---")
        
        st.write("### 📊 Résolutions Énergétiques")
        
        # Formule résolution
        st.latex(r"\frac{\sigma_E}{E} = \frac{a}{\sqrt{E}} \oplus b \oplus \frac{c}{E}")
        
        resolutions = [
            {"Calorimètre": "EM Barrel", "a (stochastique)": "10%", "b (constant)": "0.7%", "c (bruit)": "0"},
            {"Calorimètre": "EM Endcap", "a (stochastique)": "12%", "b (constant)": "0.8%", "c (bruit)": "0"},
            {"Calorimètre": "HAD Barrel", "a (stochastique)": "50%", "b (constant)": "3%", "c (bruit)": "0"},
            {"Calorimètre": "HAD Endcap", "a (stochastique)": "55%", "b (constant)": "4%", "c (bruit)": "0"},
        ]
        
        df_reso = pd.DataFrame(resolutions)
        st.dataframe(df_reso, use_container_width=True)
        
        st.markdown("---")
        
        # Graphique résolution vs énergie
        E = np.logspace(0, 3, 100)  # 1 GeV à 1 TeV
        
        sigma_em = np.sqrt((10/np.sqrt(E))**2 + 0.7**2)
        sigma_had = np.sqrt((50/np.sqrt(E))**2 + 3**2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=E, y=sigma_em,
            mode='lines',
            name='EM Calo',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=E, y=sigma_had,
            mode='lines',
            name='HAD Calo',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Résolution Énergétique vs E",
            xaxis_title="Énergie (GeV)",
            yaxis_title="σ/E (%)",
            xaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔗 Reconstruction d'Objets Physiques")
        
        st.write("### 🎯 Électrons et Photons")
        
        electron_criteria = {
            "Isolation": "R < 0.3, ΣpT < 0.1×pT",
            "Shower Shape": "Variables η×φ, E/p",
            "Track Match": "ΔR(track, cluster) < 0.05",
            "Conversion Veto": "Pas de vertex γ→e⁺e⁻",
            "Efficacité": "~80% (Tight), ~95% (Loose)"
        }
        
        for crit, value in electron_criteria.items():
            st.write(f"**{crit}:** {value}")
        
        st.markdown("---")
        
        st.write("### 🔵 Muons")
        
        muon_types = [
            {"Type": "Standalone", "Détecteur": "Chambres muons seules", "Résolution pT": "15-40%"},
            {"Type": "Global", "Détecteur": "Tracker + Muon chambers", "Résolution pT": "1-5%"},
            {"Type": "Tracker", "Détecteur": "Tracker seul", "Résolution pT": "1-2%"},
        ]
        
        df_muons = pd.DataFrame(muon_types)
        st.dataframe(df_muons, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### ✈️ Jets")
        
        st.info("""
        **Algorithmes de Jets:**
        - **anti-kT** (R=0.4, R=0.8) - Standard
        - **Cambridge-Aachen** - Jets larges
        - **kT** - Recherche théorique
        
        **Corrections:**
        - JES (Jet Energy Scale)
        - JER (Jet Energy Resolution)
        - Pile-up subtraction
        - b-tagging
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Algo Standard", "anti-kT R=0.4")
        with col2:
            st.metric("JES Incertitude", "1-5%")
        with col3:
            st.metric("b-tag Eff", "70-85%")
        
        st.markdown("---")
        
        st.write("### 💨 Énergie Transverse Manquante (MET)")
        
        st.latex(r"\vec{E}_T^{miss} = -\sum_{i} \vec{p}_T^i")
        
        met_types = [
            {"Type": "Calo MET", "Source": "Calorimètres", "Résolution": "~5-10 GeV"},
            {"Type": "Track MET", "Source": "Traces", "Résolution": "~3-5 GeV"},
            {"Type": "PF MET", "Source": "Particle Flow", "Résolution": "~2-4 GeV"},
        ]
        
        df_met = pd.DataFrame(met_types)
        st.dataframe(df_met, use_container_width=True)
    
    with tab4:
        st.subheader("👁️ Visualisation d'Événements")
        
        st.write("### 🎨 Event Display")
        
        # Simulation événement simple
        st.info("**Événement Simulé: H → γγ**")
        
        # Vue transverse (η-φ)
        st.write("#### Vue Transverse (η-φ)")
        
        # Deux photons
        photon1_eta = 0.5
        photon1_phi = 1.2
        photon1_et = 60
        
        photon2_eta = -0.8
        photon2_phi = -2.5
        photon2_et = 45
        
        # Bruit de fond
        np.random.seed(42)
        n_particles = 50
        bg_eta = np.random.uniform(-2.5, 2.5, n_particles)
        bg_phi = np.random.uniform(-np.pi, np.pi, n_particles)
        bg_et = np.random.exponential(5, n_particles)
        
        fig = go.Figure()
        
        # Background
        fig.add_trace(go.Scatter(
            x=bg_eta, y=bg_phi,
            mode='markers',
            marker=dict(size=bg_et, color='lightgray', opacity=0.5),
            name='Background'
        ))
        
        # Photons
        fig.add_trace(go.Scatter(
            x=[photon1_eta], y=[photon1_phi],
            mode='markers',
            marker=dict(size=photon1_et, color='yellow', symbol='star',
                       line=dict(color='orange', width=2)),
            name=f'Photon 1 (ET={photon1_et} GeV)'
        ))
        
        fig.add_trace(go.Scatter(
            x=[photon2_eta], y=[photon2_phi],
            mode='markers',
            marker=dict(size=photon2_et, color='yellow', symbol='star',
                       line=dict(color='orange', width=2)),
            name=f'Photon 2 (ET={photon2_et} GeV)'
        ))
        
        fig.update_layout(
            title="Event Display: H → γγ Candidate",
            xaxis_title="η (pseudorapidity)",
            yaxis_title="φ (azimuth)",
            xaxis=dict(range=[-3, 3]),
            yaxis=dict(range=[-np.pi, np.pi]),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Informations événement
        st.write("### 📊 Propriétés de l'Événement")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Run Number", "123456")
            st.metric("Event Number", "789012345")
        
        with col2:
            st.metric("N Photons", "2")
            st.metric("N Jets", "0")
        
        with col3:
            m_inv = np.sqrt(2 * photon1_et * photon2_et * 
                           (np.cosh(photon1_eta - photon2_eta) - 
                            np.cos(photon1_phi - photon2_phi)))
            st.metric("m_γγ", f"{m_inv:.1f} GeV")
        
        with col4:
            st.metric("MET", "5.2 GeV")
            st.metric("Vertex", "23 PV")

# ==================== PAGE: SECTIONS EFFICACES ====================
elif page == "⚡ Sections Efficaces":
    st.header("⚡ Sections Efficaces et Prédictions Théoriques")
    
    tab1, tab2, tab3 = st.tabs(["📊 Calculs", "📈 Mesures", "🔍 Comparaisons"])
    
    with tab1:
        st.subheader("📊 Calcul de Sections Efficaces")
        
        st.write("### 🧮 Formules Fondamentales")
        
        st.latex(r"\sigma = \int d\sigma = \int \frac{1}{2s} |{\cal M}|^2 d\Phi")
        
        st.write("""
        Où:
        - **σ** : Section efficace totale
        - **s** : Énergie dans le centre de masse au carré
        - **|ℳ|²** : Élément de matrice au carré
        - **dΦ** : Espace de phase
        """)
        
        st.markdown("---")
        
        st.write("### 📐 Calculateur Interactif")
        
        col1, col2 = st.columns(2)
        
        with col1:
            process_sigma = st.selectbox(
                "Processus",
                ["pp → H (ggF)", "pp → tt̄", "pp → Z", "pp → W", 
                 "pp → ZZ", "pp → WW", "pp → γγ (prompt)"],
                key="sigma_process"
            )
            
            sqrt_s = st.number_input("√s (GeV)", 1000, 100000, 13000, 100)
        
        with col2:
            order_qcd = st.selectbox("Ordre QCD", ["LO", "NLO", "NNLO", "N³LO"])
            pdf_set_sigma = st.selectbox("PDF Set", ["NNPDF3.1", "CT18", "MMHT2014"])
        
        if st.button("🔬 Calculer Section Efficace"):
            # Calculs simplifiés
            cross_sections_base = {
                "pp → H (ggF)": 50.0,
                "pp → tt̄": 830.0,
                "pp → Z": 60000.0,
                "pp → W": 200000.0,
                "pp → ZZ": 16.0,
                "pp → WW": 120.0,
                "pp → γγ (prompt)": 140.0
            }
            
            k_factors = {"LO": 1.0, "NLO": 1.3, "NNLO": 1.05, "N³LO": 1.02}
            
            sigma_lo = cross_sections_base[process_sigma]
            k_factor_total = np.prod([k_factors[o] for o in ["NLO", "NNLO", "N³LO"][:list(k_factors.keys()).index(order_qcd)+1]])
            
            sigma_final = sigma_lo * k_factor_total * (sqrt_s / 13000)**0.3
            
            uncertainty_scale = 5  # %
            uncertainty_pdf = 3  # %
            uncertainty_alpha_s = 2  # %
            uncertainty_total = np.sqrt(uncertainty_scale**2 + uncertainty_pdf**2 + uncertainty_alpha_s**2)
            
            st.success(f"✅ Calcul terminé!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("σ (LO)", f"{sigma_lo:.2f} pb")
            with col2:
                st.metric("K-factor", f"{k_factor_total:.2f}")
            with col3:
                st.metric("σ ("+order_qcd+")", f"{sigma_final:.2f} pb")
            
            st.write(f"**Incertitude totale:** ±{uncertainty_total:.1f}%")
            st.write(f"**Échelle:** ±{uncertainty_scale}%")
            st.write(f"**PDF:** ±{uncertainty_pdf}%")
            st.write(f"**αₛ:** ±{uncertainty_alpha_s}%")
        
        st.markdown("---")
        
        st.write("### 📊 K-factors")
        
        st.info("""
        **K-factor** = σ(ordre supérieur) / σ(LO)
        
        Mesure l'importance des corrections radiatives
        """)
        
        k_factor_data = [
            {"Processus": "gg → H", "K(NLO)": "2.0", "K(NNLO)": "1.3"},
            {"Processus": "qq̄ → W/Z", "K(NLO)": "1.3", "K(NNLO)": "1.1"},
            {"Processus": "gg → tt̄", "K(NLO)": "1.5", "K(NNLO)": "1.1"},
        ]
        
        df_k = pd.DataFrame(k_factor_data)
        st.dataframe(df_k, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Mesures Expérimentales")
        
        st.write("### 🎯 Sections Efficaces Mesurées")
        
        measured_xs = [
            {"Processus": "pp → Z → ll", "σ (pb)": "1981 ± 25", "√s": "13 TeV", "Expérience": "ATLAS"},
            {"Processus": "pp → W → lν", "σ (pb)": "20450 ± 260", "√s": "13 TeV", "Expérience": "ATLAS"},
            {"Processus": "pp → tt̄", "σ (pb)": "830 ± 40", "√s": "13 TeV", "Expérience": "CMS"},
            {"Processus": "pp → H", "σ (pb)": "55.6 ± 2.5", "√s": "13 TeV", "Expérience": "ATLAS+CMS"},
            {"Processus": "pp → ZZ", "σ (pb)": "17.2 ± 0.9", "√s": "13 TeV", "Expérience": "ATLAS"},
            {"Processus": "pp → WW", "σ (pb)": "118.7 ± 6.0", "√s": "13 TeV", "Expérience": "CMS"},
        ]
        
        df_measured = pd.DataFrame(measured_xs)
        st.dataframe(df_measured, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Précision des Mesures")
        
        # Graphique précision
        processes = [row['Processus'] for row in measured_xs]
        precision = [1.3, 1.3, 4.8, 4.5, 5.2, 5.1]  # %
        
        fig = go.Figure(data=[
            go.Bar(x=processes, y=precision,
                  marker_color='lightblue',
                  text=[f"{p:.1f}%" for p in precision],
                  textposition='outside')
        ])
        
        fig.update_layout(
            title="Précision des Mesures de Section Efficace",
            xaxis_title="Processus",
            yaxis_title="Incertitude (%)",
            xaxis_tickangle=-45,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Comparaison Théorie vs Expérience")
        
        st.write("### 📊 Accord Théorie-Expérience")
        
        # Données de comparaison
        comparison_data = [
            {"Processus": "Z → ll", "Théorie": "2000", "Mesure": "1981 ± 25", "Ratio": "0.990 ± 0.013"},
            {"Processus": "W → lν", "Théorie": "20500", "Mesure": "20450 ± 260", "Ratio": "0.998 ± 0.013"},
            {"Processus": "tt̄", "Théorie": "832", "Mesure": "830 ± 40", "Ratio": "0.998 ± 0.048"},
            {"Processus": "H (ggF)", "Théorie": "54.7", "Mesure": "55.6 ± 2.5", "Ratio": "1.016 ± 0.046"},
            {"Processus": "ZZ", "Théorie": "17.0", "Mesure": "17.2 ± 0.9", "Ratio": "1.012 ± 0.053"},
        ]
        
        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True)
        
        st.markdown("---")
        
        # Graphique ratio
        processes_comp = [row['Processus'] for row in comparison_data]
        ratios = [0.990, 0.998, 0.998, 1.016, 1.012]
        errors = [0.013, 0.013, 0.048, 0.046, 0.053]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=processes_comp,
            y=ratios,
            error_y=dict(type='data', array=errors, visible=True),
            mode='markers',
            marker=dict(size=12, color='blue'),
            name='Data/Theory'
        ))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="Accord parfait")
        
        fig.update_layout(
            title="Ratio Mesure/Théorie",
            xaxis_title="Processus",
            yaxis_title="σ_mesure / σ_théorie",
            yaxis_range=[0.9, 1.1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ Excellent accord entre théorie et expérience!")
       
# ==================== PAGE: FAISCEAUX & INJECTION ====================
elif page == "📡 Faisceaux & Injection":
    st.header("📡 Faisceaux et Système d'Injection")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Configuration Faisceaux", "💉 Injection", "🔄 Chaîne d'Accélération", "📊 Qualité Faisceau"])
    
    with tab1:
        st.subheader("🎯 Configuration des Faisceaux")
        
        with st.form("beam_config"):
            st.write("### Paramètres du Faisceau")
            
            col1, col2 = st.columns(2)
            
            with col1:
                particle_type = st.selectbox("Type de Particule", 
                    ["Proton", "Antiproton", "Électron", "Positron", "Ions Lourds (Pb)", "Muon"])
                beam_energy = st.number_input("Énergie (GeV)", 1, 100000, 7000, 100)
                
                n_bunches = st.number_input("Nombre de Paquets", 1, 10000, 2808, 1)
                bunch_intensity = st.number_input("Intensité/Paquet (×10¹¹)", 0.1, 10.0, 1.15, 0.01)
            
            with col2:
                bunch_spacing = st.slider("Espacement (ns)", 5, 200, 25, 5)
                # bunch_length = st.number_input("Longueur Paquet (cm)", 1, 100, 7.5, 0.1)
                bunch_length = st.number_input("Longueur Paquet (cm)", 1.0, 100.0, 7.5, 0.1)
                
                emittance_x = st.number_input("Émittance εₓ (μm·rad)", 0.1, 10.0, 3.5, 0.1)
                emittance_y = st.number_input("Émittance εᵧ (μm·rad)", 0.1, 10.0, 3.5, 0.1)
            
            st.write("### Optique du Faisceau")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                beta_x = st.number_input("βₓ* (cm)", 10, 500, 55, 5)
            with col2:
                beta_y = st.number_input("βᵧ* (cm)", 10, 500, 55, 5)
            with col3:
                crossing_angle = st.number_input("Angle Croisement (μrad)", 0, 500, 285, 5)
            
            submitted = st.form_submit_button("💾 Sauvegarder Configuration")
            
            if submitted:
                beam_config = {
                    'particle': particle_type,
                    'energy': beam_energy,
                    'n_bunches': n_bunches,
                    'intensity': bunch_intensity * 1e11,
                    'spacing': bunch_spacing,
                    'length': bunch_length,
                    'emittance': {'x': emittance_x, 'y': emittance_y},
                    'beta_star': {'x': beta_x, 'y': beta_y},
                    'crossing_angle': crossing_angle
                }
                
                st.session_state.particle_system['beams'][f"beam_{len(st.session_state.particle_system['beams'])+1}"] = beam_config
                
                st.success("✅ Configuration faisceau sauvegardée!")
                
                # Calcul luminosité géométrique
                N = bunch_intensity * 1e11
                n_b = n_bunches
                f_rev = 11245  # Hz pour 27 km
                sigma_x = np.sqrt(emittance_x * beta_x * 1e-4) * 1e-2  # en m
                sigma_y = np.sqrt(emittance_y * beta_y * 1e-4) * 1e-2
                
                lumi = (N * N * n_b * f_rev) / (4 * np.pi * sigma_x * sigma_y)
                
                st.metric("Luminosité Estimée", f"{lumi:.2e} cm⁻²s⁻¹")
                
                log_event(f"Configuration faisceau: {particle_type} @ {beam_energy} GeV")
    
    with tab2:
        st.subheader("💉 Système d'Injection")
        
        st.write("### 🔗 Chaîne d'Injection")
        
        injection_chain = [
            {"Étape": "Source", "Énergie": "100 keV", "Système": "Source ions/électrons", "Durée": "continu"},
            {"Étape": "RFQ", "Énergie": "3 MeV", "Système": "Quadrupôle RF", "Durée": "μs"},
            {"Étape": "Linac", "Énergie": "50 MeV", "Système": "Accélérateur linéaire", "Durée": "ms"},
            {"Étape": "Booster", "Énergie": "1.4 GeV", "Système": "Synchrotron", "Durée": "1.2 s"},
            {"Étape": "PS", "Énergie": "25 GeV", "Système": "Proton Synchrotron", "Durée": "3.6 s"},
            {"Étape": "SPS", "Énergie": "450 GeV", "Système": "Super PS", "Durée": "10 s"},
            {"Étape": "Collisionneur", "Énergie": "7000 GeV", "Système": "Ring principal", "Durée": "20 min"}
        ]
        
        df_injection = pd.DataFrame(injection_chain)
        st.dataframe(df_injection, use_container_width=True)
        
        st.markdown("---")
        
        # Graphique énergie vs temps
        times_cumul = np.cumsum([0, 1e-6, 1e-3, 1.2, 3.6, 10, 1200])
        energies = [0.0001, 0.003, 0.05, 1.4, 25, 450, 7000]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times_cumul, y=energies,
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title="Énergie du Faisceau dans la Chaîne d'Injection",
            xaxis_title="Temps Cumulé (s)",
            yaxis_title="Énergie (GeV)",
            xaxis_type="log",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🎯 Contrôle d'Injection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Injecter Faisceau 1", use_container_width=True):
                with st.spinner("Injection en cours..."):
                    progress = st.progress(0)
                    for i in range(100):
                        progress.progress(i + 1)
                    st.success("✅ Faisceau 1 injecté!")
        
        with col2:
            if st.button("🚀 Injecter Faisceau 2", use_container_width=True):
                with st.spinner("Injection en cours..."):
                    progress = st.progress(0)
                    for i in range(100):
                        progress.progress(i + 1)
                    st.success("✅ Faisceau 2 injecté!")
        
        with col3:
            if st.button("⚡ Injection Simultanée", use_container_width=True, type="primary"):
                with st.spinner("Injection des deux faisceaux..."):
                    progress = st.progress(0)
                    for i in range(100):
                        progress.progress(i + 1)
                    st.success("✅ Les deux faisceaux injectés!")
    
    with tab3:
        st.subheader("🔄 Chaîne d'Accélération Complète")
        
        st.write("### 🏗️ Architecture du Complexe")
        
        # Diagramme simplifié
        st.info("""
        **Chaîne Type LHC:**
        
        1. **Source** → Ions H⁻ ou électrons
        2. **RFQ** (Radio Frequency Quadrupole) → 3 MeV
        3. **Linac2** → 50 MeV
        4. **PSB** (Proton Synchrotron Booster) → 1.4 GeV
        5. **PS** (Proton Synchrotron) → 25 GeV
        6. **SPS** (Super Proton Synchrotron) → 450 GeV
        7. **LHC** (Large Hadron Collider) → 7000 GeV
        
        **Temps total de remplissage:** ~20 minutes
        """)
        
        st.markdown("---")
        
        st.write("### ⚙️ Paramètres par Étage")
        
        stages_params = [
            {"Accélérateur": "PSB", "Circonférence": "157 m", "Aimants": "100 dipôles", "RF": "400 MHz", "Cycle": "1.2 s"},
            {"Accélérateur": "PS", "Circonférence": "628 m", "Aimants": "277 dipôles", "RF": "10-200 MHz", "Cycle": "3.6 s"},
            {"Accélérateur": "SPS", "Circonférence": "6.9 km", "Aimants": "744 dipôles", "RF": "200 MHz", "Cycle": "10 s"},
            {"Accélérateur": "LHC", "Circonférence": "27 km", "Aimants": "1232 dipôles", "RF": "400 MHz", "Cycle": "20 min"}
        ]
        
        df_stages = pd.DataFrame(stages_params)
        st.dataframe(df_stages, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Qualité et Diagnostic du Faisceau")
        
        st.write("### 🔍 Mesures de Qualité")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Intensité", f"{np.random.uniform(1.1, 1.2):.2f}×10¹¹ p/bunch")
            st.metric("Émittance εₓ", f"{np.random.uniform(3.0, 4.0):.2f} μm·rad")
        
        with col2:
            st.metric("Longueur Paquet", f"{np.random.uniform(7.0, 8.0):.1f} cm")
            st.metric("ΔE/E", f"{np.random.uniform(0.01, 0.02):.3f}%")
        
        with col3:
            st.metric("Pertes", f"{np.random.uniform(0.1, 0.5):.2f}%")
            st.metric("Temps de Vie", f"{np.random.uniform(20, 30):.0f} heures")
        
        st.markdown("---")
        
        st.write("### 📈 Profils Transverses")
        
        # Simulation profils
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)
        
        # Distribution gaussienne 2D
        Z = np.exp(-(X**2 + Y**2) / 2)
        
        fig = go.Figure(data=go.Contour(
            z=Z, x=x, y=y,
            colorscale='Hot',
            contours=dict(coloring='heatmap')
        ))
        
        fig.update_layout(
            title="Profil Transverse du Faisceau",
            xaxis_title="x (σ)",
            yaxis_title="y (σ)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🎯 Instruments de Diagnostic")
        
        diagnostics = [
            {"Instrument": "BPM (Beam Position Monitor)", "Mesure": "Position", "Résolution": "1 μm", "Nombre": "1000+"},
            {"Instrument": "Wire Scanner", "Mesure": "Profil transverse", "Résolution": "10 μm", "Nombre": "100"},
            {"Instrument": "Synchrotron Light Monitor", "Mesure": "Profil longitudinal", "Résolution": "10 ps", "Nombre": "10"},
            {"Instrument": "BCT (Beam Current Transformer)", "Mesure": "Intensité", "Résolution": "0.1%", "Nombre": "50"},
            {"Instrument": "Schottky Monitor", "Mesure": "Tune, chromaticité", "Résolution": "0.001", "Nombre": "4"}
        ]
        
        df_diag = pd.DataFrame(diagnostics)
        st.dataframe(df_diag, use_container_width=True)

# ==================== PAGE: MAGNETS & RF ====================
elif page == "🧲 Magnets & RF":
    st.header("🧲 Aimants et Système RF")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧲 Aimants", "📡 Cavités RF", "❄️ Cryogénie", "⚡ Alimentation"])
    
    with tab1:
        st.subheader("🧲 Système d'Aimants")
        
        st.write("### 🔩 Types d'Aimants")
        
        magnet_types = {
            "Dipôles": {
                "fonction": "Courbure du faisceau",
                "champ": "8.33 T",
                "longueur": "14.3 m",
                "nombre": "1232",
                "courant": "11,850 A",
                "température": "1.9 K"
            },
            "Quadrupôles": {
                "fonction": "Focalisation du faisceau",
                "gradient": "223 T/m",
                "longueur": "3.1 m",
                "nombre": "392",
                "courant": "Variable",
                "température": "1.9 K"
            },
            "Sextupôles": {
                "fonction": "Correction chromaticité",
                "gradient": "1500 T/m²",
                "longueur": "0.5-1 m",
                "nombre": "~2000",
                "courant": "Variable",
                "température": "1.9 K ou 4.5 K"
            },
            "Octupôles": {
                "fonction": "Correction non-linéaire",
                "ordre": "3",
                "longueur": "0.5 m",
                "nombre": "~300",
                "courant": "Variable",
                "température": "4.5 K"
            },
            "Correcteurs": {
                "fonction": "Corrections d'orbite",
                "champ": "Variable",
                "longueur": "0.5-1 m",
                "nombre": "~1000",
                "courant": "Variable",
                "température": "4.5 K"
            }
        }
        
        for mag_name, mag_info in magnet_types.items():
            with st.expander(f"🧲 {mag_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    for key in list(mag_info.keys())[:3]:
                        st.write(f"**{key.title()}:** {mag_info[key]}")
                
                with col2:
                    for key in list(mag_info.keys())[3:]:
                        st.write(f"**{key.title()}:** {mag_info[key]}")
        
        st.markdown("---")
        
        st.write("### 📊 Distribution des Aimants")
        
        mag_names = ["Dipôles", "Quadrupôles", "Sextupôles", "Octupôles", "Correcteurs"]
        mag_counts = [1232, 392, 2000, 300, 1000]
        
        fig = px.pie(values=mag_counts, names=mag_names, 
                     title="Répartition des Aimants",
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🎯 Performance des Dipôles")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Champ Nominal", "8.33 T")
        with col2:
            st.metric("Courant", "11,850 A")
        with col3:
            st.metric("Énergie Stockée", "7 MJ/aimant")
        with col4:
            st.metric("Homogénéité", "< 10⁻⁴")
    
    with tab2:
        st.subheader("📡 Système de Radiofréquence")
        
        st.write("### 🔊 Cavités RF")
        
        rf_specs = {
            "Type": "Supraconductrice",
            "Fréquence": "400.789 MHz",
            "Tension": "2 MV par cavité",
            "Nombre cavités": "8 par faisceau (16 total)",
            "Puissance": "300 kW par cavité",
            "Température": "4.5 K",
            "Facteur Q": "> 10⁵"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for key in list(rf_specs.keys())[:4]:
                st.metric(key, rf_specs[key])
        
        with col2:
            for key in list(rf_specs.keys())[4:]:
                st.metric(key, rf_specs[key])
        
        st.markdown("---")
        
        st.write("### ⚡ Gain d'Énergie")
        
        st.info("""
        **Principe:** Les cavités RF accélèrent les particules chargées par champs électriques oscillants.
        
        **Gain par tour:** ~480 keV
        **Temps d'accélération:** 450 GeV → 7 TeV en ~20 minutes
        **Nombre de tours:** ~10 millions
        """)
        
        # Simulation rampe d'énergie
        n_turns = np.linspace(0, 1e7, 1000)
        energy_ramp = 450 + (7000 - 450) * (n_turns / 1e7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=n_turns/1e6, y=energy_ramp,
            mode='lines',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Rampe d'Énergie du Faisceau",
            xaxis_title="Nombre de Tours (×10⁶)",
            yaxis_title="Énergie (GeV)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🎛️ Contrôle RF")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            voltage = st.slider("Tension (MV)", 0.5, 3.0, 2.0, 0.1)
            st.metric("Gain/Tour", f"{voltage * 8:.0f} keV")
        
        with col2:
            phase = st.slider("Phase (°)", -180, 180, 0, 5)
            st.metric("Efficacité", f"{np.cos(np.radians(phase))*100:.1f}%")
        
        with col3:
            frequency_offset = st.number_input("Δf (Hz)", -1000, 1000, 0, 10)
            st.metric("Fréquence", f"{400.789 + frequency_offset/1e6:.6f} MHz")
    
    with tab3:
        st.subheader("❄️ Système Cryogénique")
        
        st.write("### 🧊 Températures de Fonctionnement")
        
        cryo_temps = [
            {"Système": "Dipôles & Quadrupôles", "Température": "1.9 K", "Hélium": "Superfluide", "Puissance": "40 kW"},
            {"Système": "Autres aimants supraconducteurs", "Température": "4.5 K", "Hélium": "Liquide", "Puissance": "20 kW"},
            {"Système": "Écrans thermiques", "Température": "60-80 K", "Hélium": "Gazeux", "Puissance": "10 kW"},
            {"Système": "Cavités RF", "Température": "4.5 K", "Hélium": "Liquide", "Puissance": "5 kW"}
        ]
        
        df_cryo = pd.DataFrame(cryo_temps)
        st.dataframe(df_cryo, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🏭 Stations Cryogéniques")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Capacité Totale", "144 kW @ 4.5K")
        with col2:
            st.metric("He Liquide", "~130 tonnes")
        with col3:
            st.metric("Stations", "8 principales")
        with col4:
            st.metric("Temps Refroidissement", "~2 semaines")
        
        st.markdown("---")
        
        # Diagramme température
        sections = ["Dipôles", "Quadrupôles", "RF", "Autres", "Écrans"]
        temperatures = [1.9, 1.9, 4.5, 4.5, 70]
        
        fig = go.Figure(data=[
            go.Bar(x=sections, y=temperatures,
                  marker_color=['darkblue', 'darkblue', 'blue', 'blue', 'lightblue'],
                  text=[f"{t:.1f} K" for t in temperatures],
                  textposition='outside')
        ])
        
        fig.update_layout(
            title="Températures de Fonctionnement",
            yaxis_title="Température (K)",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 L'hélium superfluide à 1.9 K permet des champs magnétiques plus élevés")
    
    with tab4:
        st.subheader("⚡ Système d'Alimentation Électrique")
        
        st.write("### 🔌 Consommation Énergétique")
        
        power_systems = [
            {"Système": "Aimants", "Puissance": "120 MW", "Pourcentage": "60%"},
            {"Système": "Cryogénie", "Puissance": "30 MW", "Pourcentage": "15%"},
            {"Système": "RF", "Puissance": "10 MW", "Pourcentage": "5%"},
            {"Système": "Détecteurs & DAQ", "Puissance": "20 MW", "Pourcentage": "10%"},
            {"Système": "Infrastructure", "Puissance": "20 MW", "Pourcentage": "10%"}
        ]
        
        df_power = pd.DataFrame(power_systems)
        st.dataframe(df_power, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(df_power, values='Puissance', names='Système',
                        title="Répartition de la Consommation")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Consommation Totale", "200 MW")
            st.metric("Coût Électricité/An", "~100 M€")
            st.metric("Équivalent", "~200,000 foyers")
            
            st.info("💡 Pic de consommation pendant l'accélération")
        
        st.markdown("---")
        
        st.write("### 🔋 Convertisseurs de Puissance")
        
        st.write("""
        **Types de convertisseurs:**
        - **PC (Power Converter):** Alimentation aimants dipôles (11,850 A)
        - **QPS (Quench Protection System):** Protection contre quench
        - **UPS:** Alimentation sans interruption
        - **Onduleurs:** Conversion AC/DC haute précision
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Convertisseurs", "~1,700")
        with col2:
            st.metric("Stabilité Courant", "< 10 ppm")
        with col3:
            st.metric("Temps Réponse", "< 100 μs")

# ==================== PAGE: ACQUISITION DE DONNÉES ====================
elif page == "📊 Acquisition de Données":
    st.header("📊 Système d'Acquisition de Données (DAQ)")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Architecture", "⚡ Trigger", "💾 Stockage", "🔄 Traitement"])
    
    with tab1:
        st.subheader("🎯 Architecture du DAQ")
        
        st.write("### 📡 Pipeline de Données")
        
        daq_pipeline = [
            {"Étape": "Collision", "Taux": "40 MHz", "Données": "-", "Latence": "0"},
            {"Étape": "Front-End", "Taux": "40 MHz", "Données": "~1 MB/evt", "Latence": "2.5 μs"},
            {"Étape": "L1 Trigger", "Taux": "100 kHz", "Données": "~1 MB/evt", "Latence": "< 4 μs"},
            {"Étape": "DAQ Readout", "Taux": "100 kHz", "Données": "~1 MB/evt", "Latence": "100 ms"},
            {"Étape": "HLT", "Taux": "1 kHz", "Données": "~1.5 MB/evt", "Latence": "200 ms"},
            {"Étape": "Storage", "Taux": "1 kHz", "Données": "~1.5 GB/s", "Latence": "~1 s"},
        ]
        
        df_daq = pd.DataFrame(daq_pipeline)
        st.dataframe(df_daq, use_container_width=True)
        
        st.markdown("---")
        
        # Graphique taux
        stages = [row['Étape'] for row in daq_pipeline]
        rates = [40e6, 40e6, 100e3, 100e3, 1e3, 1e3]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stages, y=rates,
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=12)
        ))
        
        fig.update_layout(
            title="Flux de Données à Travers le DAQ",
            yaxis_title="Taux (Hz)",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🏗️ Infrastructure")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Canaux Lecture", "100M+")
        with col2:
            st.metric("Serveurs HLT", "~30,000")
        with col3:
            st.metric("Réseau", "100-400 Gb/s")
        with col4:
            st.metric("Stockage Tape", "PB/an")
    
    with tab2:
        st.subheader("⚡ Système de Trigger")
        
        st.write("### 🎯 Trigger Niveau 1 (L1)")
        
        st.info("""
        **Objectif:** Réduire 40 MHz → 100 kHz en < 4 μs
        
        **Hardware:** FPGA, ASIC personnalisés
        
        **Critères:**
                             
        - Muons haute-pT (pT > 20 GeV)
        - Électrons/photons (ET > 30 GeV)
        - Jets (ET > 100 GeV)
        - Énergie transverse manquante (MET > 50 GeV)
        - Tau leptons
        """)
        
        l1_objects = {
            "Muons": {"Seuil": "20 GeV", "Taux": "~20 kHz", "Efficacité": "95%"},
            "e/γ": {"Seuil": "30 GeV", "Taux": "~30 kHz", "Efficacité": "90%"},
            "Jets": {"Seuil": "100 GeV", "Taux": "~30 kHz", "Efficacité": "98%"},
            "MET": {"Seuil": "50 GeV", "Taux": "~20 kHz", "Efficacité": "85%"},
        }
        
        for obj, specs in l1_objects.items():
            with st.expander(f"🎯 {obj}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Seuil pT/ET", specs["Seuil"])
                with col2:
                    st.metric("Taux Typique", specs["Taux"])
                with col3:
                    st.metric("Efficacité", specs["Efficacité"])
        
        st.markdown("---")
        
        st.write("### 🖥️ High Level Trigger (HLT)")
        
        st.info("""
        **Objectif:** Réduire 100 kHz → 1 kHz en ~200 ms
        
        **Infrastructure:** ~30,000 CPU cores
        
        **Algorithmes:** Reconstruction quasi-complète
        - Tracking précis
        - Identification particules
        - Isolation
        - Vertex reconstruction
        - B-tagging
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Input Rate", "100 kHz")
        with col2:
            st.metric("Output Rate", "1 kHz")
        with col3:
            st.metric("Réjection", "100:1")
        
        st.markdown("---")
        
        st.write("### 📊 Menu de Trigger")
        
        trigger_menu = [
            {"Trigger": "SingleMuon", "Seuil": "pT > 24 GeV", "Prescale": "1", "Taux": "250 Hz"},
            {"Trigger": "DiMuon", "Seuil": "pT > 17, 8 GeV", "Prescale": "1", "Taux": "100 Hz"},
            {"Trigger": "SingleElectron", "Seuil": "ET > 32 GeV", "Prescale": "1", "Taux": "200 Hz"},
            {"Trigger": "DiPhoton", "Seuil": "ET > 30, 18 GeV", "Prescale": "1", "Taux": "80 Hz"},
            {"Trigger": "MET", "Seuil": "MET > 120 GeV", "Prescale": "1", "Taux": "150 Hz"},
            {"Trigger": "HT", "Seuil": "HT > 900 GeV", "Prescale": "1", "Taux": "120 Hz"},
        ]
        
        df_triggers = pd.DataFrame(trigger_menu)
        st.dataframe(df_triggers, use_container_width=True)
    
    with tab3:
        st.subheader("💾 Système de Stockage")
        
        st.write("### 📦 Volumes de Données")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taux Enregistrement", "1 kHz")
        with col2:
            st.metric("Taille Événement", "1.5 MB")
        with col3:
            st.metric("Flux Données", "1.5 GB/s")
        with col4:
            st.metric("Volume Annuel", "~50 PB")
        
        st.markdown("---")
        
        st.write("### 🗄️ Hiérarchie de Stockage")
        
        storage_tiers = [
            {"Tier": "T0 (CERN)", "Rôle": "Reconstruction initiale", "Capacité": "~100 PB", "Bande": "100 GB/s"},
            {"Tier": "T1 (7 centres)", "Rôle": "Re-reconstruction, archivage", "Capacité": "~50 PB chacun", "Bande": "50 GB/s"},
            {"Tier": "T2 (~150 sites)", "Rôle": "Analyse utilisateur", "Capacité": "~10 PB chacun", "Bande": "10 GB/s"},
            {"Tier": "T3 (Local)", "Rôle": "Analyse locale", "Capacité": "Variable", "Bande": "Variable"},
        ]
        
        df_storage = pd.DataFrame(storage_tiers)
        st.dataframe(df_storage, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📊 Distribution Géographique")
        
        # Carte conceptuelle
        st.info("""
        **WLCG (Worldwide LHC Computing Grid):**
        
        - **170+ sites** dans 42 pays
        - **~1.4 million** cœurs CPU
        - **~1.5 exabyte** de stockage
        - **Réseau:** LHCONE, GÉANT
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            regions = ["Europe", "Amérique", "Asie", "Autres"]
            capacities = [60, 25, 12, 3]
            
            fig = px.pie(values=capacities, names=regions,
                        title="Capacité Computing par Région (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("Sites Totaux", "170+")
            st.metric("CPU Cores", "1.4M")
            st.metric("Stockage Total", "1.5 EB")
            st.metric("Réseau Backbone", "100 Gb/s")
    
    with tab4:
        st.subheader("🔄 Traitement et Reconstruction")
        
        st.write("### ⚙️ Pipeline de Reconstruction")
        
        reco_steps = [
            {"Étape": "1. Hit Reconstruction", "Description": "Signaux détecteurs → hits", "CPU": "~5%"},
            {"Étape": "2. Track Finding", "Description": "Hits → traces particules", "CPU": "~30%"},
            {"Étape": "3. Track Fitting", "Description": "Paramètres traces", "CPU": "~20%"},
            {"Étape": "4. Vertex Reconstruction", "Description": "Vertex primaire/secondaires", "CPU": "~10%"},
            {"Étape": "5. Calorimeter Clustering", "Description": "Dépôts énergie → clusters", "CPU": "~15%"},
            {"Étape": "6. Particle ID", "Description": "Identification e/γ/μ/τ", "CPU": "~10%"},
            {"Étape": "7. Jet Reconstruction", "Description": "Algorithmes anti-kT", "CPU": "~5%"},
            {"Étape": "8. MET Calculation", "Description": "Énergie manquante", "CPU": "~5%"},
        ]
        
        df_reco = pd.DataFrame(reco_steps)
        st.dataframe(df_reco, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 💻 Ressources de Calcul")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Temps/Événement", "~10 s")
            st.metric("Débit T0", "~500 kHz·s")
        
        with col2:
            st.metric("Re-reco/An", "2-3 passes")
            st.metric("CPU Total", "~500k HS06")
        
        with col3:
            st.metric("Efficacité Grid", "~80%")
            st.metric("Coût/Événement", "~0.01 €")
        
        st.markdown("---")
        
        st.write("### 📊 Format des Données")
        
        data_formats = {
            "RAW": {
                "Description": "Données brutes des détecteurs",
                "Taille": "~1.5 MB/evt",
                "Usage": "Reconstruction",
                "Retention": "Archive permanente"
            },
            "AOD": {
                "Description": "Analysis Object Data",
                "Taille": "~500 kB/evt",
                "Usage": "Analyses physique",
                "Retention": "Plusieurs années"
            },
            "MINIAOD": {
                "Description": "Version compacte AOD",
                "Taille": "~50 kB/evt",
                "Usage": "Analyses utilisateur",
                "Retention": "Permanente"
            },
            "NANOAOD": {
                "Description": "Format minimal",
                "Taille": "~2 kB/evt",
                "Usage": "Analyses rapides",
                "Retention": "Permanente"
            }
        }
        
        for fmt_name, fmt_info in data_formats.items():
            with st.expander(f"📄 {fmt_name}"):
                for key, value in fmt_info.items():
                    st.write(f"**{key}:** {value}")

# ==================== FOOTER ====================

st.markdown("---")

with st.expander("📜 Journal des Événements (Dernières 10 entrées)"):
    if st.session_state.particle_system['log']:
        for event in st.session_state.particle_system['log'][-10:][::-1]:
            timestamp = event['timestamp'][:19]
            st.text(f"{timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")
    
    if st.button("🗑️ Effacer le Journal", key="clear_log_particle"):
        st.session_state.particle_system['log'] = []
        st.rerun()

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>⚛️ Plateforme de Physique des Particules</h3>
        <p>Système Intégré pour Collisionneurs et Expériences HEP</p>
        <p><small>Version 1.0.0 | Tous Domaines de la Physique des Hautes Énergies</small></p>
        <p><small>⚛️ Collisionneurs | 🔬 Détecteurs | 💫 Simulations | 📊 Analyses | 🏆 Découvertes</small></p>
        <p><small>🧲 Accélérateurs | 📡 DAQ | 🎲 Monte Carlo | 📈 Physique | 🌌 BSM</small></p>
        <p><small>Powered by Particle Physics Research © 2024</small></p>
    </div>
""", unsafe_allow_html=True)