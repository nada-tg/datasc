"""
Plateforme Avancée Recherche Matière Noire
Dark Matter Detection & Analysis Platform
IA • Quantique • Bio-Computing • WIMPs • Neutrinos

Installation:
pip install streamlit pandas plotly numpy scipy scikit-learn

Lancement:
streamlit run dark_matter_platform_app.py
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
    page_title="🌌 Dark Matter Research Platform",
    page_icon="🌌",
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
        background: linear-gradient(90deg, #000033 0%, #4B0082 30%, #8B008B 60%, #000033 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: cosmic-glow 3s ease-in-out infinite alternate;
    }
    @keyframes cosmic-glow {
        from { filter: drop-shadow(0 0 10px #4B0082); }
        to { filter: drop-shadow(0 0 30px #8B008B); }
    }
    .dark-matter-card {
        border: 3px solid #4B0082;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(75, 0, 130, 0.1) 0%, rgba(139, 0, 139, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(75, 0, 130, 0.4);
        transition: all 0.3s;
    }
    .dark-matter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(139, 0, 139, 0.6);
    }
    .particle-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
        background: linear-gradient(90deg, #4B0082 0%, #8B008B 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(75, 0, 130, 0.4);
    }
    .detection-pulse {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 0.6; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    .quantum-border {
        border: 2px solid;
        border-image: linear-gradient(45deg, #4B0082, #8B008B, #9370DB) 1;
        animation: quantum-shift 3s infinite;
    }
    @keyframes quantum-shift {
        0%, 100% { border-color: #4B0082; }
        50% { border-color: #8B008B; }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES PHYSIQUES ====================
PHYSICS_CONSTANTS = {
    'c': 299792458,  # Vitesse lumière (m/s)
    'h': 6.62607015e-34,  # Constante Planck (J⋅s)
    'G': 6.67430e-11,  # Constante gravitationnelle (m³⋅kg⁻¹⋅s⁻²)
    'me': 9.10938356e-31,  # Masse électron (kg)
    'mp': 1.6726219e-27,  # Masse proton (kg)
    'NA': 6.02214076e23,  # Nombre Avogadro (mol⁻¹)
    'dark_matter_fraction': 0.268,  # Fraction matière noire dans univers
    'baryon_fraction': 0.049,  # Fraction matière baryonique
    'dark_energy_fraction': 0.683,  # Fraction énergie noire
}

WIMP_MASSES = {
    'Light': (1, 10),  # GeV/c²
    'Medium': (10, 100),
    'Heavy': (100, 1000),
    'Super-Heavy': (1000, 10000)
}

DETECTOR_TYPES = {
    'Xenon': 'Détecteur au Xénon liquide',
    'Argon': 'Détecteur à l\'Argon liquide',
    'Germanium': 'Détecteur au Germanium cryogénique',
    'Scintillator': 'Scintillateur organique',
    'Bubble': 'Chambre à bulles',
    'Bolometer': 'Bolomètre cryogénique'
}

# ==================== INITIALISATION SESSION STATE ====================
if 'dark_matter_lab' not in st.session_state:
    st.session_state.dark_matter_lab = {
        'detectors': {},
        'experiments': {},
        'detections': [],
        'wimps_candidates': [],
        'neutrino_events': [],
        'xenon_decays': [],
        'ai_models': {},
        'quantum_simulations': [],
        'bio_computing_tasks': [],
        'analysis_results': {},
        'particles_database': {},
        'collaborations': {},
        'publications': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================
def log_event(message: str, level: str = "INFO"):
    """Enregistrer événement"""
    st.session_state.dark_matter_lab['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_wimp_interaction_rate(mass_gev: float, cross_section: float, 
                                    detector_mass_kg: float) -> float:
    """Calculer taux d'interaction WIMPs"""
    # Formule simplifiée
    rho_dm = 0.3  # GeV/cm³ densité locale matière noire
    v_dm = 220000  # m/s vitesse moyenne WIMPs
    
    rate = (rho_dm * cross_section * detector_mass_kg * v_dm) / mass_gev
    return rate * 1e-45  # Conversion échelle réaliste

def simulate_xenon_decay(isotope: str, time_hours: float) -> List[Dict]:
    """Simuler désintégrations Xénon"""
    events = []
    
    decay_constants = {
        'Xe-136': 2.11e-22,  # s⁻¹ (double bêta)
        'Xe-134': 1e-25,
        'Xe-132': 5e-26
    }
    
    lambda_decay = decay_constants.get(isotope, 1e-24)
    n_events = int(lambda_decay * time_hours * 3600 * 1e6)  # Nombre d'atomes
    
    for _ in range(n_events):
        event = {
            'timestamp': datetime.now() + timedelta(seconds=np.random.uniform(0, time_hours * 3600)),
            'isotope': isotope,
            'energy_kev': np.random.normal(2458, 50) if isotope == 'Xe-136' else np.random.normal(1000, 100),
            'type': 'double_beta' if np.random.random() > 0.9 else 'single_beta',
            'position': {
                'x': np.random.uniform(-50, 50),
                'y': np.random.uniform(-50, 50),
                'z': np.random.uniform(-100, 100)
            }
        }
        events.append(event)
    
    return events

def detect_solar_neutrinos(detector_type: str, exposure_days: float) -> List[Dict]:
    """Détecter neutrinos solaires"""
    # Flux neutrinos solaires: ~6.5e10 /cm²/s
    flux = 6.5e10
    
    # Efficacité détection selon type
    efficiency = {
        'Xenon': 0.15,
        'Argon': 0.12,
        'Germanium': 0.20,
        'Scintillator': 0.08
    }.get(detector_type, 0.10)
    
    detector_area_cm2 = 10000  # 1 m²
    n_events = int(flux * detector_area_cm2 * exposure_days * 86400 * efficiency * 1e-12)
    
    events = []
    for _ in range(n_events):
        event = {
            'timestamp': datetime.now() + timedelta(seconds=np.random.uniform(0, exposure_days * 86400)),
            'type': 'solar_neutrino',
            'flavor': np.random.choice(['electron', 'muon', 'tau']),
            'energy_mev': np.random.exponential(0.5),  # Spectre énergétique
            'interaction': np.random.choice(['elastic', 'charged_current', 'neutral_current']),
            'position': {
                'x': np.random.normal(0, 30),
                'y': np.random.normal(0, 30),
                'z': np.random.normal(0, 60)
            }
        }
        events.append(event)
    
    return events

def ai_analyze_signal(signal_data: np.ndarray) -> Dict:
    """Analyse signal par IA"""
    # Simulation analyse IA
    mean_signal = np.mean(signal_data)
    std_signal = np.std(signal_data)
    
    # Détection anomalies
    threshold = mean_signal + 3 * std_signal
    anomalies = np.where(signal_data > threshold)[0]
    
    return {
        'confidence': np.random.uniform(0.7, 0.99),
        'classification': 'WIMP' if len(anomalies) > 5 else 'Background',
        'anomalies_count': len(anomalies),
        'signal_quality': 'High' if std_signal < mean_signal * 0.3 else 'Medium',
        'recommended_action': 'Further analysis' if len(anomalies) > 5 else 'Continue monitoring'
    }

def quantum_compute_cross_section(wimp_mass: float, nucleon_mass: float) -> float:
    """Calculer section efficace par ordinateur quantique"""
    # Simulation calcul quantique
    mu = (wimp_mass * nucleon_mass) / (wimp_mass + nucleon_mass)  # Masse réduite
    cross_section = 1e-45 * (mu / 1)**2  # cm² (ordre de grandeur)
    
    # Facteur quantique (simulation)
    quantum_correction = np.random.uniform(0.8, 1.2)
    
    return cross_section * quantum_correction

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🌌 Dark Matter Research Platform</h1>', unsafe_allow_html=True)
st.markdown("### Plateforme Avancée • WIMPs • Neutrinos • Xénon • IA • Quantique • Bio-Computing")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/4B0082/ffffff?text=Dark+Matter+Lab", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Centre Contrôle",
            "🔬 Mes Détecteurs",
            "➕ Créer Détecteur",
            "🎯 Détection WIMPs",
            "☀️ Neutrinos Solaires",
            "⚛️ Désintégrations Xénon",
            "📊 Collecte Données",
            "🤖 IA Analyse",
            "⚛️ Computing Quantique",
            "🧬 Bio-Computing",
            "📈 Expériences",
            "🔍 Recherche Particules",
            "📊 Base Données Particules",
            "🌌 Simulations Cosmiques",
            "🧪 Laboratoire Virtuel",
            "📡 Signaux Temps Réel",
            "🎨 Visualisation 3D",
            "🤝 Collaborations",
            "📚 Publications",
            "📊 Analytics",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_detectors = len(st.session_state.dark_matter_lab['detectors'])
    total_detections = len(st.session_state.dark_matter_lab['detections'])
    total_wimps = len(st.session_state.dark_matter_lab['wimps_candidates'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔬 Détecteurs", total_detectors)
        st.metric("🎯 Détections", total_detections)
    with col2:
        st.metric("⚛️ WIMPs", total_wimps)
        st.metric("☀️ Neutrinos", len(st.session_state.dark_matter_lab['neutrino_events']))
    
    st.markdown("---")
    st.markdown("### 🌌 Univers Observable")
    
    st.write(f"**Matière Noire:** {PHYSICS_CONSTANTS['dark_matter_fraction']*100:.1f}%")
    st.write(f"**Matière Baryonique:** {PHYSICS_CONSTANTS['baryon_fraction']*100:.1f}%")
    st.write(f"**Énergie Noire:** {PHYSICS_CONSTANTS['dark_energy_fraction']*100:.1f}%")

# ==================== PAGE: CENTRE CONTRÔLE ====================
if page == "🏠 Centre Contrôle":
    st.header("🏠 Centre de Contrôle - Dark Matter Lab")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="dark-matter-card"><h2>🔬</h2><h3>{total_detectors}</h3><p>Détecteurs Actifs</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="dark-matter-card"><h2>🎯</h2><h3>{total_detections}</h3><p>Détections Totales</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="dark-matter-card"><h2>⚛️</h2><h3>{total_wimps}</h3><p>Candidats WIMPs</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        neutrino_count = len(st.session_state.dark_matter_lab['neutrino_events'])
        st.markdown(f'<div class="dark-matter-card"><h2>☀️</h2><h3>{neutrino_count}</h3><p>Neutrinos</p></div>', 
                   unsafe_allow_html=True)
    
    with col5:
        xenon_count = len(st.session_state.dark_matter_lab['xenon_decays'])
        st.markdown(f'<div class="dark-matter-card"><h2>🔬</h2><h3>{xenon_count}</h3><p>Xénon Events</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Composition Univers
    st.subheader("🌌 Composition de l'Univers Observable")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Graphique composition
        composition = pd.DataFrame({
            'Composant': ['Matière Noire', 'Énergie Noire', 'Matière Baryonique', 'Radiation', 'Neutrinos'],
            'Pourcentage': [26.8, 68.3, 4.9, 0.005, 0.01],
            'Couleur': ['#4B0082', '#000033', '#8B008B', '#FFD700', '#00CED1']
        })
        
        fig = go.Figure(data=[go.Pie(
            labels=composition['Composant'],
            values=composition['Pourcentage'],
            hole=.4,
            marker=dict(colors=composition['Couleur']),
            textinfo='label+percent',
            textfont=dict(size=14)
        )])
        
        fig.update_layout(
            title="Distribution Énergie-Matière Univers",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("### 🎯 Objectifs Recherche")
        
        st.info("""
        **Matière Noire (26.8%)**
        
        ⚛️ Candidats principaux:
        - WIMPs (Particules massives)
        - Axions
        - Neutrinos stériles
        - MACHOs
        
        🔬 Méthodes détection:
        - Détection directe
        - Détection indirecte
        - Production collisionneur
        """)
        
        st.metric("Masse Manquante Univers", "~85%", 
                 help="85% de la matière dans l'univers est de la matière noire")
    
    st.markdown("---")
    
    # Status Détecteurs
    st.subheader("🔬 État des Détecteurs")
    
    if not st.session_state.dark_matter_lab['detectors']:
        st.info("💡 Aucun détecteur créé. Créez votre premier détecteur!")
        
        if st.button("➕ Créer Premier Détecteur", type="primary"):
            st.info("Accédez à 'Créer Détecteur' dans le menu")
    else:
        detector_status = []
        for det_id, detector in st.session_state.dark_matter_lab['detectors'].items():
            detector_status.append({
                'Nom': detector['name'],
                'Type': detector['type'],
                'Statut': '🟢 Actif' if detector['status'] == 'active' else '🔴 Inactif',
                'Masse': f"{detector['mass_kg']} kg",
                'Température': f"{detector['temperature_k']} K",
                'Events': detector.get('total_events', 0)
            })
        
        df_detectors = pd.DataFrame(detector_status)
        st.dataframe(df_detectors, use_container_width=True)
    
    st.markdown("---")
    
    # Détections Récentes
    st.subheader("🎯 Détections Récentes")
    
    if st.session_state.dark_matter_lab['detections']:
        recent_detections = st.session_state.dark_matter_lab['detections'][-10:][::-1]
        
        for detection in recent_detections:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.write(f"**{detection['timestamp'][:19]}**")
            with col2:
                particle_type = detection.get('particle_type', 'Unknown')
                st.write(f"Type: **{particle_type}**")
            with col3:
                energy = detection.get('energy_kev', 0)
                st.write(f"Énergie: **{energy:.2f} keV**")
            with col4:
                confidence = detection.get('confidence', 0) * 100
                if confidence > 80:
                    st.success(f"{confidence:.0f}%")
                else:
                    st.warning(f"{confidence:.0f}%")
    else:
        st.info("Aucune détection enregistrée")
    
    st.markdown("---")
    
    # Technologies Avancées
    st.subheader("🚀 Technologies Avancées Intégrées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 🤖 Intelligence Artificielle")
        st.write("✅ Classification événements temps réel")
        st.write("✅ Détection anomalies")
        st.write("✅ Prédiction signaux")
        st.write("✅ Optimisation paramètres")
        st.write("✅ Analyse Big Data particules")
    
    with col2:
        st.write("### ⚛️ Computing Quantique")
        st.write("✅ Calcul sections efficaces")
        st.write("✅ Simulation interactions")
        st.write("✅ Optimisation détection")
        st.write("✅ Cryptographie données")
        st.write("✅ Parallélisation massive")
    
    with col3:
        st.write("### 🧬 Bio-Computing")
        st.write("✅ Traitement parallèle ADN")
        st.write("✅ Reconnaissance patterns")
        st.write("✅ Stockage données massif")
        st.write("✅ Calcul énergétiquement efficace")
        st.write("✅ Auto-réparation systèmes")

# ==================== PAGE: MES DÉTECTEURS ====================
elif page == "🔬 Mes Détecteurs":
    st.header("🔬 Gestion des Détecteurs")
    
    if not st.session_state.dark_matter_lab['detectors']:
        st.info("💡 Aucun détecteur créé")
        
        if st.button("➕ Créer Premier Détecteur", type="primary"):
            st.info("Accédez à 'Créer Détecteur'")
    else:
        for det_id, detector in st.session_state.dark_matter_lab['detectors'].items():
            with st.expander(f"🔬 {detector['name']} ({detector['type']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("### 📊 Caractéristiques")
                    st.write(f"**Type:** {DETECTOR_TYPES[detector['type']]}")
                    st.write(f"**Masse:** {detector['mass_kg']} kg")
                    st.write(f"**Température:** {detector['temperature_k']} K")
                    st.write(f"**Pression:** {detector.get('pressure_bar', 1)} bar")
                    
                    status_icon = "🟢" if detector['status'] == 'active' else "🔴"
                    st.write(f"**Statut:** {status_icon} {detector['status']}")
                
                with col2:
                    st.write("### 🎯 Performance")
                    st.metric("Events Totaux", detector.get('total_events', 0))
                    st.metric("WIMPs Candidats", detector.get('wimp_candidates', 0))
                    st.metric("Neutrinos", detector.get('neutrino_events', 0))
                    st.metric("Background Rate", f"{detector.get('background_rate', 0):.2f} Hz")
                
                with col3:
                    st.write("### 🔬 Sensibilité")
                    st.metric("Seuil Énergie", f"{detector.get('threshold_kev', 1)} keV")
                    st.metric("Résolution", f"{detector.get('energy_resolution', 5)}%")
                    st.metric("Temps Mort", f"{detector.get('dead_time_us', 10)} μs")
                    st.metric("Efficacité", f"{detector.get('efficiency', 80)}%")
                
                st.markdown("---")
                
                # Actions
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("📊 Voir Données", key=f"data_{det_id}"):
                        st.info("Affichage données en cours...")
                
                with col2:
                    if st.button("🎯 Lancer Acquisition", key=f"acq_{det_id}"):
                        st.success("Acquisition démarrée!")
                
                with col3:
                    if st.button("⚙️ Calibrer", key=f"cal_{det_id}"):
                        st.info("Calibration en cours...")
                
                with col4:
                    if st.button("🗑️ Supprimer", key=f"del_{det_id}"):
                        del st.session_state.dark_matter_lab['detectors'][det_id]
                        log_event(f"Détecteur supprimé: {detector['name']}", "WARNING")
                        st.rerun()

# ==================== PAGE: CRÉER DÉTECTEUR ====================
elif page == "➕ Créer Détecteur":
    st.header("➕ Créer Nouveau Détecteur Matière Noire")
    
    st.info("""
    🔬 **Configurez votre détecteur de matière noire**
    
    Choisissez le type, la masse, la température et les paramètres pour optimiser
    la détection de WIMPs, neutrinos et autres particules exotiques.
    """)
    
    with st.form("create_detector"):
        st.subheader("📋 Configuration Détecteur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            detector_name = st.text_input("Nom Détecteur", "XENON-DM-01")
            
            detector_type = st.selectbox("Type Détecteur",
                list(DETECTOR_TYPES.keys()),
                format_func=lambda x: DETECTOR_TYPES[x])
            
            mass_kg = st.number_input("Masse Active (kg)", 10, 10000, 1000, 10)
            
            temperature_k = st.number_input("Température Opération (K)", 
                                           0.01, 300.0, 0.1, 0.01,
                                           help="Pour cryogénique: ~0.01-1 K")
        
        with col2:
            location = st.selectbox("Localisation",
                ["Gran Sasso (Italie)", "Sudbury (Canada)", "Kamioka (Japon)",
                 "Sanford Lab (USA)", "Modane (France)", "Boulby (UK)"])
            
            depth_m = st.number_input("Profondeur (mètres)", 100, 5000, 1400, 100,
                                     help="Profondeur pour blindage rayons cosmiques")
            
            shielding = st.multiselect("Blindage",
                ["Plomb", "Cuivre", "Polyéthylène", "Eau", "Roche"],
                default=["Plomb", "Eau"])
            
            pressure_bar = st.number_input("Pression (bar)", 0.01, 100.0, 1.0, 0.1)
        
        st.markdown("---")
        st.subheader("🎯 Paramètres Détection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            threshold_kev = st.number_input("Seuil Énergie (keV)", 0.1, 100.0, 1.0, 0.1)
            energy_resolution = st.slider("Résolution Énergétique (%)", 1, 50, 5, 1)
        
        with col2:
            efficiency = st.slider("Efficacité Détection (%)", 50, 100, 80, 1)
            dead_time_us = st.number_input("Temps Mort (μs)", 1, 1000, 10, 1)
        
        with col3:
            fiducial_volume = st.slider("Volume Fiduciel (%)", 50, 100, 80, 1,
                                        help="Volume central pour réduire le bruit de fond")
        
        st.markdown("---")
        st.subheader("🚀 Technologies Avancées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ai_enabled = st.checkbox("🤖 IA Analyse Temps Réel", value=True)
            if ai_enabled:
                ai_model = st.selectbox("Modèle IA",
                    ["CNN Deep Learning", "Random Forest", "XGBoost", "Neural Network"])
        
        with col2:
            quantum_enabled = st.checkbox("⚛️ Computing Quantique", value=True)
            if quantum_enabled:
                qubits = st.slider("Nombre Qubits", 8, 128, 64, 8)
        
        with col3:
            bio_enabled = st.checkbox("🧬 Bio-Computing", value=False)
            if bio_enabled:
                dna_strands = st.number_input("Brins ADN", 1000, 1000000, 10000, 1000)
        
        st.markdown("---")
        
        if st.form_submit_button("🔬 Créer Détecteur", type="primary"):
            if not detector_name:
                st.error("⚠️ Veuillez donner un nom au détecteur")
            else:
                det_id = f"det_{len(st.session_state.dark_matter_lab['detectors']) + 1}"
                
                detector = {
                    'id': det_id,
                    'name': detector_name,
                    'type': detector_type,
                    'mass_kg': mass_kg,
                    'temperature_k': temperature_k,
                    'location': location,
                    'depth_m': depth_m,
                    'shielding': shielding,
                    'pressure_bar': pressure_bar,
                    'threshold_kev': threshold_kev,
                    'energy_resolution': energy_resolution,
                    'efficiency': efficiency,
                    'dead_time_us': dead_time_us,
                    'fiducial_volume': fiducial_volume,
                    'ai_enabled': ai_enabled,
                    'ai_model': ai_model if ai_enabled else None,
                    'quantum_enabled': quantum_enabled,
                    'qubits': qubits if quantum_enabled else 0,
                    'bio_enabled': bio_enabled,
                    'dna_strands': dna_strands if bio_enabled else 0,
                    'status': 'active',
                    'created_at': datetime.now().isoformat(),
                    'total_events': 0,
                    'wimp_candidates': 0,
                    'neutrino_events': 0,
                    'background_rate': np.random.uniform(0.01, 0.5)
                }
                
                st.session_state.dark_matter_lab['detectors'][det_id] = detector
                log_event(f"Détecteur créé: {detector_name}", "SUCCESS")
                
                with st.spinner("Initialisation détecteur..."):
                    import time
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                
                st.success(f"✅ Détecteur '{detector_name}' créé et opérationnel!")
                st.balloons()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sensibilité", f"{threshold_kev} keV")
                with col2:
                    st.metric("Masse Active", f"{mass_kg} kg")
                with col3:
                    st.metric("Efficacité", f"{efficiency}%")
                with col4:
                    st.metric("Profondeur", f"{depth_m} m")
                
                st.info(f"🎯 ID Détecteur: {det_id}")
                st.rerun()

# ==================== PAGE: DÉTECTION WIMPs ====================
elif page == "🎯 Détection WIMPs":
    st.header("🎯 Détection WIMPs (Weakly Interacting Massive Particles)")
    
    st.info("""
    **WIMPs - Candidats Principaux Matière Noire**
    
    Les WIMPs sont des particules massives (1-1000 GeV/c²) qui n'interagissent que 
    faiblement avec la matière ordinaire. Leur section efficace d'interaction est 
    extrêmement faible (~10⁻⁴⁵ cm²), nécessitant des détecteurs massifs en sites profonds.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Recherche Active", "📊 Candidats Détectés", 
                                      "📈 Analyses", "⚙️ Paramètres"])
    
    with tab1:
        st.subheader("🔍 Lancer Recherche WIMPs")
        
        if not st.session_state.dark_matter_lab['detectors']:
            st.warning("⚠️ Aucun détecteur disponible. Créez d'abord un détecteur.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_detector = st.selectbox("Sélectionner Détecteur",
                    list(st.session_state.dark_matter_lab['detectors'].keys()),
                    format_func=lambda x: st.session_state.dark_matter_lab['detectors'][x]['name'])
                
                detector = st.session_state.dark_matter_lab['detectors'][selected_detector]
                
                st.write(f"**Type:** {DETECTOR_TYPES[detector['type']]}")
                st.write(f"**Masse:** {detector['mass_kg']} kg")
                st.write(f"**Seuil:** {detector['threshold_kev']} keV")
            
            with col2:
                st.write("### ⚙️ Paramètres Recherche")
                
                exposure_days = st.number_input("Temps Exposition (jours)", 
                                               1, 365, 30, 1)
                
                wimp_mass_range = st.selectbox("Plage Masse WIMPs",
                    list(WIMP_MASSES.keys()))
                
                min_mass, max_mass = WIMP_MASSES[wimp_mass_range]
                
                wimp_mass = st.slider(f"Masse WIMP (GeV/c²)", 
                                     float(min_mass), float(max_mass), 
                                     float((min_mass + max_mass) / 2))
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Flux Local Estimé", "~0.3 GeV/cm³")
            with col2:
                st.metric("Vitesse Moyenne", "220 km/s")
            with col3:
                cross_section = quantum_compute_cross_section(wimp_mass, 931.5)
                st.metric("Section Efficace", f"{cross_section:.2e} cm²")
            
            if st.button("🚀 Lancer Recherche WIMPs", type="primary", use_container_width=True):
                with st.spinner(f"Recherche en cours pendant {exposure_days} jours simulés..."):
                    import time
                    progress_bar = st.progress(0)
                    
                    # Simulation acquisition
                    n_candidates = 0
                    
                    for day in range(int(exposure_days)):
                        time.sleep(0.05)
                        progress_bar.progress((day + 1) / exposure_days)
                        
                        # Taux interaction WIMP
                        rate = calculate_wimp_interaction_rate(
                            wimp_mass, cross_section, detector['mass_kg']
                        )
                        
                        # Nombre événements par jour
                        n_events_day = int(np.random.poisson(rate * 86400))
                        
                        for _ in range(n_events_day):
                            # Énergie de recul nucléaire
                            energy_kev = np.random.exponential(10) + detector['threshold_kev']
                            
                            if energy_kev > detector['threshold_kev']:
                                event = {
                                    'timestamp': datetime.now() + timedelta(days=day, 
                                                   seconds=np.random.uniform(0, 86400)),
                                    'detector_id': selected_detector,
                                    'particle_type': 'WIMP_candidate',
                                    'energy_kev': energy_kev,
                                    'wimp_mass_gev': wimp_mass,
                                    'confidence': np.random.uniform(0.6, 0.95),
                                    'position': {
                                        'x': np.random.normal(0, 10),
                                        'y': np.random.normal(0, 10),
                                        'z': np.random.normal(0, 20)
                                    },
                                    'recoil_type': np.random.choice(['nuclear', 'electronic'])
                                }
                                
                                # Analyse IA si activée
                                if detector['ai_enabled']:
                                    signal_data = np.random.normal(energy_kev, 5, 100)
                                    ai_result = ai_analyze_signal(signal_data)
                                    event['ai_analysis'] = ai_result
                                    
                                    if ai_result['classification'] == 'WIMP':
                                        st.session_state.dark_matter_lab['wimps_candidates'].append(event)
                                        n_candidates += 1
                                
                                st.session_state.dark_matter_lab['detections'].append(event)
                                detector['total_events'] += 1
                                detector['wimp_candidates'] += 1
                    
                    progress_bar.empty()
                
                st.success(f"✅ Recherche terminée!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Events Détectés", detector['total_events'])
                with col2:
                    st.metric("Candidats WIMPs", n_candidates)
                with col3:
                    significance = n_candidates / np.sqrt(detector['total_events']) if detector['total_events'] > 0 else 0
                    st.metric("Significance (σ)", f"{significance:.2f}")
                
                if n_candidates > 0:
                    st.success(f"🎯 {n_candidates} candidats WIMPs identifiés!")
                else:
                    st.info("Aucun candidat WIMP détecté. Continuez l'acquisition.")
                
                log_event(f"Recherche WIMPs complétée: {n_candidates} candidats", "SUCCESS")
    
    with tab2:
        st.subheader("📊 Candidats WIMPs Détectés")
        
        if not st.session_state.dark_matter_lab['wimps_candidates']:
            st.info("Aucun candidat WIMP détecté. Lancez une recherche d'abord.")
        else:
            # Afficher candidats
            wimps_data = []
            for wimp in st.session_state.dark_matter_lab['wimps_candidates'][-50:]:
                wimps_data.append({
                    'Timestamp': wimp['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(wimp['timestamp'], datetime) else wimp['timestamp'][:19],
                    'Énergie (keV)': f"{wimp['energy_kev']:.2f}",
                    'Masse WIMP (GeV)': f"{wimp['wimp_mass_gev']:.1f}",
                    'Confidence': f"{wimp['confidence']*100:.1f}%",
                    'Type Recul': wimp['recoil_type']
                })
            
            df_wimps = pd.DataFrame(wimps_data)
            st.dataframe(df_wimps, use_container_width=True)
            
            # Graphique distribution énergie
            st.write("### 📈 Distribution Énergétique")
            
            energies = [w['energy_kev'] for w in st.session_state.dark_matter_lab['wimps_candidates']]
            
            fig = go.Figure(data=[go.Histogram(
                x=energies,
                nbinsx=30,
                marker_color='#4B0082',
                opacity=0.7
            )])
            
            fig.update_layout(
                title="Distribution Énergie Recul Nucléaire",
                xaxis_title="Énergie (keV)",
                yaxis_title="Nombre Events",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Graphique 3D positions
            st.write("### 🎯 Localisation Spatiale Events")
            
            positions = [w['position'] for w in st.session_state.dark_matter_lab['wimps_candidates'][-100:]]
            x_pos = [p['x'] for p in positions]
            y_pos = [p['y'] for p in positions]
            z_pos = [p['z'] for p in positions]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x_pos,
                y=y_pos,
                z=z_pos,
                mode='markers',
                marker=dict(
                    size=5,
                    color=energies[-100:],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Énergie (keV)")
                )
            )])
            
            fig.update_layout(
                title="Distribution Spatiale Candidats WIMPs",
                scene=dict(
                    xaxis_title="X (cm)",
                    yaxis_title="Y (cm)",
                    zaxis_title="Z (cm)"
                ),
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Analyses Statistiques")
        
        if len(st.session_state.dark_matter_lab['wimps_candidates']) < 10:
            st.warning("Nombre insuffisant de candidats pour analyse statistique (min 10)")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Statistiques Descriptives")
                
                energies = [w['energy_kev'] for w in st.session_state.dark_matter_lab['wimps_candidates']]
                masses = [w['wimp_mass_gev'] for w in st.session_state.dark_matter_lab['wimps_candidates']]
                confidences = [w['confidence'] for w in st.session_state.dark_matter_lab['wimps_candidates']]
                
                st.metric("Énergie Moyenne", f"{np.mean(energies):.2f} keV")
                st.metric("Énergie Médiane", f"{np.median(energies):.2f} keV")
                st.metric("Écart-Type", f"{np.std(energies):.2f} keV")
                st.metric("Confidence Moyenne", f"{np.mean(confidences)*100:.1f}%")
            
            with col2:
                st.write("### 🎯 Tests Statistiques")
                
                # Test distribution énergie
                from scipy import stats
                
                # Test normalité
                _, p_value_normal = stats.normaltest(energies)
                
                st.write(f"**Test Normalité (p-value):** {p_value_normal:.4f}")
                
                if p_value_normal > 0.05:
                    st.success("✅ Distribution compatible Gaussienne")
                else:
                    st.warning("⚠️ Distribution non-Gaussienne")
                
                # Taux événements
                if len(st.session_state.dark_matter_lab['wimps_candidates']) > 1:
                    times = [datetime.fromisoformat(w['timestamp']) if isinstance(w['timestamp'], str) else w['timestamp'] 
                            for w in st.session_state.dark_matter_lab['wimps_candidates']]
                    time_diffs = [(times[i+1] - times[i]).total_seconds() / 3600 
                                 for i in range(len(times)-1)]
                    
                    mean_rate = 1 / np.mean(time_diffs) if np.mean(time_diffs) > 0 else 0
                    st.metric("Taux Moyen", f"{mean_rate:.4f} events/h")
            
            st.markdown("---")
            
            # Corrélations
            st.write("### 🔗 Matrice Corrélations")
            
            df_correlations = pd.DataFrame({
                'Énergie': energies,
                'Masse_WIMP': masses,
                'Confidence': confidences
            })
            
            corr_matrix = df_correlations.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Corrélations Variables",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚙️ Paramètres Avancés Détection WIMPs")
        
        st.write("### 🎯 Optimisation Détection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Stratégies Discrimination Background:**")
            st.checkbox("Analyse forme pulse", value=True)
            st.checkbox("Rapport S1/S2 (Xénon)", value=True)
            st.checkbox("Fiducialisation volume", value=True)
            st.checkbox("Veto muons cosmiques", value=True)
            st.checkbox("Machine Learning classification", value=True)
        
        with col2:
            st.write("**Plages Recherche:**")
            
            mass_search_min = st.number_input("Masse Min (GeV/c²)", 1, 100, 10)
            mass_search_max = st.number_input("Masse Max (GeV/c²)", 10, 10000, 1000)
            
            cross_section_min = st.number_input("Section Efficace Min (cm²)", 
                                               value=1e-48, format="%.2e")
            
            st.info(f"""
            **Paramètres Recherche:**
            - Masse: {mass_search_min} - {mass_search_max} GeV/c²
            - σ: > {cross_section_min:.2e} cm²
            """)
        
        if st.button("💾 Sauvegarder Paramètres"):
            st.success("✅ Paramètres sauvegardés!")

# ==================== PAGE: NEUTRINOS SOLAIRES ====================
elif page == "☀️ Neutrinos Solaires":
    st.header("☀️ Détection Neutrinos Solaires")
    
    st.info("""
    **Neutrinos Solaires**
    
    Le Soleil produit ~6.5×10¹⁰ neutrinos/cm²/s via fusion nucléaire.
    Ces particules traversent la matière sans interaction, mais peuvent être 
    détectées via interactions rares avec les noyaux atomiques.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔬 Détection", "📊 Événements", "📈 Analyses"])
    
    with tab1:
        st.subheader("🔬 Lancer Détection Neutrinos")
        
        if not st.session_state.dark_matter_lab['detectors']:
            st.warning("⚠️ Créez d'abord un détecteur")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_detector = st.selectbox("Détecteur",
                    list(st.session_state.dark_matter_lab['detectors'].keys()),
                    format_func=lambda x: st.session_state.dark_matter_lab['detectors'][x]['name'],
                    key="neutrino_detector")
                
                detector = st.session_state.dark_matter_lab['detectors'][selected_detector]
                
                exposure_days = st.slider("Temps Exposition (jours)", 1, 365, 30)
            
            with col2:
                st.write("### ☀️ Flux Neutrinos")
                st.metric("Flux Total", "6.5×10¹⁰ /cm²/s")
                st.metric("pp Chain", "~98%")
                st.metric("CNO Cycle", "~2%")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Saveurs Neutrinos:**")
                st.write("• νₑ (électronique)")
                st.write("• νμ (muonique)")
                st.write("• ντ (tauique)")
            
            with col2:
                st.write("**Interactions:**")
                st.write("• Élastique")
                st.write("• Courant chargé")
                st.write("• Courant neutre")
            
            with col3:
                st.write("**Énergie:**")
                st.write("• pp: 0-0.42 MeV")
                st.write("• ⁷Be: 0.86 MeV")
                st.write("• ⁸B: 0-15 MeV")
            
            if st.button("☀️ Démarrer Détection Neutrinos", type="primary", use_container_width=True):
                with st.spinner(f"Détection neutrinos {exposure_days} jours..."):
                    import time
                    progress_bar = st.progress(0)
                    
                    # Simuler détection
                    events = detect_solar_neutrinos(detector['type'], exposure_days)
                    
                    for i in range(100):
                        time.sleep(0.03)
                        progress_bar.progress(i + 1)
                    
                    # Sauvegarder événements
                    st.session_state.dark_matter_lab['neutrino_events'].extend(events)
                    detector['neutrino_events'] += len(events)
                    detector['total_events'] += len(events)
                    
                    progress_bar.empty()
                
                st.success(f"✅ {len(events)} neutrinos solaires détectés!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Events Totaux", len(events))
                with col2:
                    electron = sum(1 for e in events if e['flavor'] == 'electron')
                    st.metric("νₑ", electron)
                with col3:
                    elastic = sum(1 for e in events if e['interaction'] == 'elastic')
                    st.metric("Élastique", elastic)
                
                log_event(f"Neutrinos détectés: {len(events)}", "SUCCESS")
    
    with tab2:
        st.subheader("📊 Événements Neutrinos Détectés")
        
        if not st.session_state.dark_matter_lab['neutrino_events']:
            st.info("Aucun neutrino détecté. Lancez une détection.")
        else:
            # Table événements
            neutrino_data = []
            for event in st.session_state.dark_matter_lab['neutrino_events'][-100:]:
                neutrino_data.append({
                    'Timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Saveur': event['flavor'],
                    'Énergie (MeV)': f"{event['energy_mev']:.3f}",
                    'Interaction': event['interaction'],
                    'X': f"{event['position']['x']:.1f}",
                    'Y': f"{event['position']['y']:.1f}",
                    'Z': f"{event['position']['z']:.1f}"
                })
        
            df_neutrinos = pd.DataFrame(neutrino_data)
            st.dataframe(df_neutrinos, use_container_width=True)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Distribution Saveurs")
                
                flavors = [e['flavor'] for e in st.session_state.dark_matter_lab['neutrino_events']]
                flavor_counts = pd.Series(flavors).value_counts()
                
                fig = go.Figure(data=[go.Bar(
                    x=flavor_counts.index,
                    y=flavor_counts.values,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )])
                
                fig.update_layout(
                    title="Saveurs Neutrinos",
                    xaxis_title="Saveur",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### ⚡ Spectre Énergétique")
                
                energies = [e['energy_mev'] for e in st.session_state.dark_matter_lab['neutrino_events']]
                
                fig = go.Figure(data=[go.Histogram(
                    x=energies,
                    nbinsx=40,
                    marker_color='#9D4EDD'
                )])
                
                fig.update_layout(
                    title="Distribution Énergie",
                    xaxis_title="Énergie (MeV)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Analyses Oscillations Neutrinos")
        
        st.info("""
        **Oscillations Neutrinos**
        
        Les neutrinos changent de saveur en se propageant (νₑ → νμ → ντ).
        Ce phénomène quantique prouve que les neutrinos ont une masse non nulle.
        """)
        
        if len(st.session_state.dark_matter_lab['neutrino_events']) > 50:
            # Calcul ratios
            flavors = [e['flavor'] for e in st.session_state.dark_matter_lab['neutrino_events']]
            
            electron_ratio = flavors.count('electron') / len(flavors)
            muon_ratio = flavors.count('muon') / len(flavors)
            tau_ratio = flavors.count('tau') / len(flavors)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Ratio νₑ", f"{electron_ratio:.3f}")
                st.write("Attendu: ~0.33")
            
            with col2:
                st.metric("Ratio νμ", f"{muon_ratio:.3f}")
                st.write("Attendu: ~0.33")
            
            with col3:
                st.metric("Ratio ντ", f"{tau_ratio:.3f}")
                st.write("Attendu: ~0.33")
            
            # Test chi-carré
            observed = [flavors.count('electron'), flavors.count('muon'), flavors.count('tau')]
            expected = [len(flavors)/3] * 3
            
            chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
            
            st.write(f"### χ² Test: {chi2:.2f}")
            
            if chi2 < 5.99:  # 95% confidence, 2 dof
                st.success("✅ Distribution compatible avec oscillations maximales")
            else:
                st.warning("⚠️ Déviation statistique détectée")
        else:
            st.warning("Données insuffisantes pour analyse oscillations")

# ==================== PAGES SUPPLÉMENTAIRES ====================

# PAGE: EXPÉRIENCES
elif page == "📈 Expériences":
    st.header("📈 Expériences de Recherche")
    
    st.info("Gérez vos campagnes d'expériences de recherche matière noire")
    
    tab1, tab2 = st.tabs(["📋 Mes Expériences", "➕ Créer Expérience"])
    
    with tab1:
        if not st.session_state.dark_matter_lab['experiments']:
            st.info("Aucune expérience créée")
        else:
            for exp_id, exp in st.session_state.dark_matter_lab['experiments'].items():
                with st.expander(f"🧪 {exp['name']}"):
                    st.write(f"**Type:** {exp['type']}")
                    st.write(f"**Durée:** {exp['duration_days']} jours")
                    st.metric("Progression", f"{exp['progress']}%")
    
    with tab2:
        with st.form("create_experiment"):
            exp_name = st.text_input("Nom Expérience", "WIMP Search Campaign 2024")
            exp_type = st.selectbox("Type", ["WIMPs Search", "Neutrino Flux", "Xenon Decay", "Calibration"])
            duration = st.slider("Durée (jours)", 1, 365, 30)
            
            if st.form_submit_button("🧪 Créer Expérience"):
                exp_id = f"exp_{len(st.session_state.dark_matter_lab['experiments']) + 1}"
                st.session_state.dark_matter_lab['experiments'][exp_id] = {
                    'name': exp_name,
                    'type': exp_type,
                    'duration_days': duration,
                    'progress': 0,
                    'created_at': datetime.now().isoformat()
                }
                st.success("✅ Expérience créée!")
                st.rerun()

# PAGE: RECHERCHE PARTICULES
elif page == "🔍 Recherche Particules":
    st.header("🔍 Recherche & Identification Particules")
    
    st.write("### 🎯 Bibliothèque Particules Matière Noire")
    
    particles_info = {
        "WIMPs": {"Masse": "1-10000 GeV/c²", "Spin": "0 ou 1/2", "Charge": "0", "Statut": "Hypothétique"},
        "Axions": {"Masse": "10⁻⁶-10⁻² eV/c²", "Spin": "0", "Charge": "0", "Statut": "Hypothétique"},
        "Neutrinos Stériles": {"Masse": "> neutrinos SM", "Spin": "1/2", "Charge": "0", "Statut": "Hypothétique"},
        "Gravitinos": {"Masse": "Variable", "Spin": "3/2", "Charge": "0", "Statut": "Supersymétrie"},
        "Neutralinos": {"Masse": "10-1000 GeV/c²", "Spin": "1/2", "Charge": "0", "Statut": "SUSY"}
    }
    
    for particle, info in particles_info.items():
        with st.expander(f"⚛️ {particle}"):
            for key, value in info.items():
                st.write(f"**{key}:** {value}")

# PAGE: BASE DONNÉES PARTICULES
elif page == "📊 Base Données Particules":
    st.header("📊 Base de Données Particules")
    
    st.write("### 🗄️ Événements Enregistrés")
    
    all_events = (
        st.session_state.dark_matter_lab['detections'] +
        st.session_state.dark_matter_lab['neutrino_events'] +
        st.session_state.dark_matter_lab['xenon_decays']
    )
    
    st.metric("Total Événements", len(all_events))
    
    if all_events:
        st.dataframe(pd.DataFrame(all_events[:100]), use_container_width=True)

# PAGE: SIMULATIONS COSMIQUES
elif page == "🌌 Simulations Cosmiques":
    st.header("🌌 Simulations Cosmologiques")
    
    st.info("Simulez la distribution de matière noire dans l'univers")
    
    if st.button("🌌 Lancer Simulation N-Corps", type="primary"):
        with st.spinner("Simulation 10⁹ particules..."):
            import time
            time.sleep(3)
            st.success("✅ Simulation complétée!")
            
            # Visualisation 3D simulation
            n_particles = 1000
            x = np.random.randn(n_particles) * 50
            y = np.random.randn(n_particles) * 50
            z = np.random.randn(n_particles) * 50
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=2, color=z, colorscale='Viridis')
            )])
            
            fig.update_layout(
                title="Distribution Matière Noire (simulation)",
                scene=dict(
                    xaxis_title="X (Mpc)",
                    yaxis_title="Y (Mpc)",
                    zaxis_title="Z (Mpc)"
                ),
                template="plotly_dark",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)

# PAGE: LABORATOIRE VIRTUEL
elif page == "🧪 Laboratoire Virtuel":
    st.header("🧪 Laboratoire Virtuel 3D")
    
    st.info("Explorez votre laboratoire en réalité virtuelle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔬 Équipements Disponibles")
        st.write("• Détecteur Xénon TPC")
        st.write("• Spectromètre Gamma")
        st.write("• Cryostat Dilution")
        st.write("• Salle Blanche ISO 5")
        st.write("• Système Veto Muons")
    
    with col2:
        if st.button("🥽 Lancer Vue VR", use_container_width=True, type="primary"):
            st.success("Vue VR lancée! Mettez votre casque.")

# PAGE: SIGNAUX TEMPS RÉEL
elif page == "📡 Signaux Temps Réel":
    st.header("📡 Monitoring Signaux Temps Réel")
    
    # Graphique temps réel simulé
    st.write("### 📊 Signal Détecteur en Direct")
    
    # Générer signal aléatoire
    time_points = np.linspace(0, 10, 1000)
    signal = np.random.normal(0, 1, 1000) + 5 * np.sin(2 * np.pi * 0.5 * time_points)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_points, y=signal, mode='lines', line=dict(color='cyan')))
    
    fig.update_layout(
        title="Signal ADC (temps réel)",
        xaxis_title="Temps (ms)",
        yaxis_title="Amplitude (ADU)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rate", "142 Hz")
    with col2:
        st.metric("Bruit RMS", "0.8 ADU")
    with col3:
        st.metric("SNR", "15.2 dB")

# PAGE: VISUALISATION 3D
elif page == "🎨 Visualisation 3D":
    st.header("🎨 Visualisation 3D Événements")
    
    st.write("### 🎯 Reconstruction Événements 3D")
    
    if st.session_state.dark_matter_lab['detections']:
        # Prendre événements avec positions
        events_with_pos = [e for e in st.session_state.dark_matter_lab['detections'] 
                          if 'position' in e][:200]
        
        if events_with_pos:
            x_pos = [e['position']['x'] for e in events_with_pos]
            y_pos = [e['position']['y'] for e in events_with_pos]
            z_pos = [e['position']['z'] for e in events_with_pos]
            energies = [e.get('energy_kev', 0) for e in events_with_pos]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x_pos,
                y=y_pos,
                z=z_pos,
                mode='markers',
                marker=dict(
                    size=5,
                    color=energies,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Énergie (keV)")
                )
            )])
            
            fig.update_layout(
                title="Distribution Spatiale Événements",
                scene=dict(
                    xaxis_title="X (cm)",
                    yaxis_title="Y (cm)",
                    zaxis_title="Z (cm)",
                    bgcolor='black'
                ),
                template="plotly_dark",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas d'événements avec positions 3D")
    else:
        st.info("Aucun événement détecté")

# PAGE: COLLABORATIONS
elif page == "🤝 Collaborations":
    st.header("🤝 Collaborations Internationales")
    
    st.info("Connectez-vous avec d'autres laboratoires de recherche matière noire")
    
    collaborations_list = [
        {"Nom": "XENON Collaboration", "Pays": "🇮🇹 Italie", "Membres": "200+", "Statut": "Actif"},
        {"Nom": "LUX-ZEPLIN (LZ)", "Pays": "🇺🇸 USA", "Membres": "250+", "Statut": "Actif"},
        {"Nom": "PandaX", "Pays": "🇨🇳 Chine", "Membres": "120+", "Statut": "Actif"},
        {"Nom": "ADMX", "Pays": "🇺🇸 USA", "Membres": "80+", "Statut": "Actif"},
        {"Nom": "DAMA/LIBRA", "Pays": "🇮🇹 Italie", "Membres": "50+", "Statut": "Actif"}
    ]
    
    df_collab = pd.DataFrame(collaborations_list)
    st.dataframe(df_collab, use_container_width=True)

# PAGE: PUBLICATIONS
elif page == "📚 Publications":
    st.header("📚 Publications & Résultats")
    
    st.write("### 📰 Articles Récents")
    
    publications = [
        {"Titre": "Search for WIMP Dark Matter in Xe-136", "Journal": "Phys. Rev. Lett.", "Année": "2024", "Citations": "234"},
        {"Titre": "Solar Neutrino Detection Results", "Journal": "Nature", "Année": "2024", "Citations": "189"},
        {"Titre": "Limits on 0νββ Decay", "Journal": "Science", "Année": "2023", "Citations": "456"}
    ]
    
    df_pubs = pd.DataFrame(publications)
    st.dataframe(df_pubs, use_container_width=True)
    
    if st.button("📝 Générer Rapport Publication"):
        st.success("Rapport généré!")
        st.download_button("📥 Télécharger PDF", data="Rapport...", file_name="rapport.pdf")













            

# ==================== PAGE: COLLECTE DONNÉES ====================
elif page == "📊 Collecte Données":
    st.header("📊 Système de Collecte de Données")
    
    st.info("""
    **Acquisition Multi-Source**
    
    Collecte simultanée de données provenant de:
    - Détecteurs matière noire (WIMPs)
    - Télescopes neutrinos solaires
    - Spectromètres désintégrations Xénon
    - Capteurs environnementaux
    """)
    
    tab1, tab2, tab3 = st.tabs(["🎛️ Configuration", "📡 Acquisition", "💾 Données"])
    
    with tab1:
        st.subheader("🎛️ Configuration Acquisition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📊 Sources de Données")
            
            sources = st.multiselect("Activer Sources",
                ["WIMPs Detection", "Solar Neutrinos", "Xenon Decays", 
                 "Background Monitoring", "Calibration", "Environmental"],
                default=["WIMPs Detection", "Solar Neutrinos"])
            
            sampling_rate = st.selectbox("Taux Échantillonnage",
                ["1 Hz", "10 Hz", "100 Hz", "1 kHz", "10 kHz"])
            
            buffer_size = st.slider("Taille Buffer (MB)", 10, 1000, 100, 10)
        
        with col2:
            st.write("### 💾 Stockage")
            
            storage_format = st.selectbox("Format Fichier",
                ["HDF5", "ROOT", "Parquet", "CSV", "Binary"])
            
            compression = st.selectbox("Compression",
                ["None", "gzip", "lzma", "zstd"])
            
            auto_backup = st.checkbox("Sauvegarde Auto", value=True)
            
            if auto_backup:
                backup_interval = st.selectbox("Intervalle Backup",
                    ["1 heure", "6 heures", "24 heures"])
        
        st.markdown("---")
        
        st.write("### 🔍 Filtres et Triggers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            energy_min = st.number_input("Énergie Min (keV)", 0.0, 100.0, 1.0, 0.1)
            energy_max = st.number_input("Énergie Max (keV)", 1.0, 10000.0, 5000.0, 10.0)
        
        with col2:
            trigger_threshold = st.number_input("Seuil Trigger (σ)", 1.0, 10.0, 3.0, 0.5)
            coincidence_window = st.number_input("Fenêtre Coïncidence (μs)", 0.1, 100.0, 1.0, 0.1)
        
        with col3:
            veto_active = st.checkbox("Veto Actif", value=True)
            pile_up_rejection = st.checkbox("Rejet Pile-Up", value=True)
        
        if st.button("💾 Sauvegarder Configuration", type="primary", use_container_width=True):
            config = {
                'sources': sources,
                'sampling_rate': sampling_rate,
                'buffer_size': buffer_size,
                'storage_format': storage_format,
                'compression': compression,
                'filters': {
                    'energy_min': energy_min,
                    'energy_max': energy_max,
                    'trigger_threshold': trigger_threshold
                }
            }
            st.success("✅ Configuration sauvegardée!")
            log_event("Configuration acquisition mise à jour", "INFO")
    
    with tab2:
        st.subheader("📡 Acquisition en Temps Réel")
        
        if not st.session_state.dark_matter_lab['detectors']:
            st.warning("⚠️ Créez d'abord un détecteur")
        else:
            selected_detector = st.selectbox("Détecteur",
                list(st.session_state.dark_matter_lab['detectors'].keys()),
                format_func=lambda x: st.session_state.dark_matter_lab['detectors'][x]['name'],
                key="acq_detector")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("État", "🟢 Prêt")
            with col2:
                st.metric("Buffer", "23%")
            with col3:
                st.metric("Rate", "142 Hz")
            with col4:
                st.metric("Events", "1,284,392")
            
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("▶️ Démarrer Acquisition", type="primary", use_container_width=True):
                    with st.spinner("Acquisition en cours..."):
                        import time
                        
                        # Simulation acquisition
                        for i in range(10):
                            time.sleep(0.5)
                            
                            # Génération données aléatoires
                            n_events = np.random.poisson(50)
                            
                            st.write(f"Batch {i+1}/10: {n_events} events")
                        
                        st.success("✅ Acquisition terminée!")
                        log_event("Acquisition données complétée", "SUCCESS")
            
            with col2:
                if st.button("⏸️ Pause", use_container_width=True):
                    st.info("Acquisition en pause")
                
                if st.button("⏹️ Stop", use_container_width=True):
                    st.warning("Acquisition arrêtée")
    
    with tab3:
        st.subheader("💾 Données Collectées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fichiers Totaux", "1,247")
            st.metric("Taille Totale", "142.3 GB")
        
        with col2:
            st.metric("Events Totaux", "87.2M")
            st.metric("Durée Run", "1,284 h")
        
        with col3:
            st.metric("Taux Moyen", "18.8 kHz")
            st.metric("Uptime", "98.7%")
        
        st.markdown("---")
        
        st.write("### 📁 Fichiers Récents")
        
        files_data = [
            {"Fichier": "run_20241018_001.h5", "Taille": "1.2 GB", "Events": "2.4M", "Date": "2024-10-18 14:23"},
            {"Fichier": "run_20241018_002.h5", "Taille": "1.1 GB", "Events": "2.3M", "Date": "2024-10-18 15:45"},
            {"Fichier": "run_20241018_003.h5", "Taille": "1.3 GB", "Events": "2.5M", "Date": "2024-10-18 17:12"},
            {"Fichier": "run_20241018_004.h5", "Taille": "1.2 GB", "Events": "2.4M", "Date": "2024-10-18 18:34"}
        ]
        
        df_files = pd.DataFrame(files_data)
        st.dataframe(df_files, use_container_width=True)

# ==================== PAGE: IA ANALYSE ====================
elif page == "🤖 IA Analyse":
    st.header("🤖 Intelligence Artificielle - Analyse Données")
    
    st.info("""
    **IA pour Physique des Particules**
    
    Utilisation de Deep Learning pour:
    - Classification événements (signal vs background)
    - Détection anomalies
    - Reconstruction trajectoires particules
    - Prédiction signaux
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Modèles", "🎯 Classification", "🔍 Anomalies", "📊 Résultats"])
    
    with tab1:
        st.subheader("🧠 Modèles IA Disponibles")
        
        models_info = {
            "CNN Deep Learning": {
                "Type": "Convolutional Neural Network",
                "Couches": "5 Conv + 3 Dense",
                "Paramètres": "2.4M",
                "Accuracy": "96.3%",
                "Usage": "Classification images détecteur"
            },
            "RNN LSTM": {
                "Type": "Recurrent Neural Network",
                "Couches": "3 LSTM + 2 Dense",
                "Paramètres": "1.8M",
                "Accuracy": "94.7%",
                "Usage": "Séries temporelles signaux"
            },
            "Random Forest": {
                "Type": "Ensemble Learning",
                "Arbres": "500",
                "Paramètres": "150K",
                "Accuracy": "92.1%",
                "Usage": "Classification features"
            },
            "XGBoost": {
                "Type": "Gradient Boosting",
                "Estimateurs": "1000",
                "Paramètres": "200K",
                "Accuracy": "93.8%",
                "Usage": "Classification multi-classe"
            },
            "Autoencoder": {
                "Type": "Unsupervised Learning",
                "Couches": "Encoder 4 + Decoder 4",
                "Paramètres": "3.1M",
                "Accuracy": "N/A",
                "Usage": "Détection anomalies"
            }
        }
        
        for model_name, info in models_info.items():
            with st.expander(f"🤖 {model_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    for key, value in info.items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    if st.button(f"🚀 Charger Modèle", key=f"load_{model_name}"):
                        with st.spinner(f"Chargement {model_name}..."):
                            import time
                            time.sleep(1)
                            st.success(f"✅ {model_name} chargé!")
                            st.session_state.dark_matter_lab['ai_models'][model_name] = {
                                'status': 'loaded',
                                'info': info
                            }
                    
                    if st.button(f"🎯 Entraîner", key=f"train_{model_name}"):
                        st.info("Entraînement lancé...")
    
    with tab2:
        st.subheader("🎯 Classification Signal/Background")
        
        if not st.session_state.dark_matter_lab['detections']:
            st.warning("⚠️ Aucune donnée disponible. Lancez d'abord une détection.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                model_choice = st.selectbox("Choisir Modèle IA",
                    list(models_info.keys()))
                
                confidence_threshold = st.slider("Seuil Confidence", 0.5, 0.99, 0.85, 0.01)
                
                n_events_classify = st.slider("Nombre Events à Classifier", 
                                              10, min(1000, len(st.session_state.dark_matter_lab['detections'])), 
                                              100, 10)
            
            with col2:
                st.write("### 📊 Statistiques")
                st.metric("Events Disponibles", len(st.session_state.dark_matter_lab['detections']))
                st.metric("Modèle Actif", model_choice)
            
            if st.button("🤖 Lancer Classification IA", type="primary", use_container_width=True):
                with st.spinner(f"Classification {n_events_classify} events..."):
                    import time
                    progress_bar = st.progress(0)
                    
                    classifications = []
                    
                    for i in range(n_events_classify):
                        # Simulation classification
                        event = st.session_state.dark_matter_lab['detections'][i]
                        
                        # Générer features
                        energy = event.get('energy_kev', 0)
                        
                        # Simulation prédiction IA
                        is_signal = energy > 10 and np.random.random() > 0.3
                        confidence = np.random.uniform(0.7, 0.99) if is_signal else np.random.uniform(0.5, 0.8)
                        
                        classifications.append({
                            'event_id': i,
                            'prediction': 'Signal' if is_signal and confidence >= confidence_threshold else 'Background',
                            'confidence': confidence,
                            'energy': energy
                        })
                        
                        if i % 10 == 0:
                            progress_bar.progress((i + 1) / n_events_classify)
                            time.sleep(0.05)
                    
                    progress_bar.empty()
                
                # Résultats
                signal_count = sum(1 for c in classifications if c['prediction'] == 'Signal')
                background_count = len(classifications) - signal_count
                
                st.success(f"✅ Classification terminée!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Signal", signal_count)
                with col2:
                    st.metric("Background", background_count)
                with col3:
                    avg_conf = np.mean([c['confidence'] for c in classifications])
                    st.metric("Confidence Moy", f"{avg_conf:.2%}")
                
                # Graphique
                df_class = pd.DataFrame(classifications)
                
                fig = px.scatter(df_class, x='event_id', y='energy', 
                                color='prediction', size='confidence',
                                title="Classification IA Events",
                                color_discrete_map={'Signal': '#00FF00', 'Background': '#FF0000'})
                
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                log_event(f"Classification IA: {signal_count} signaux identifiés", "SUCCESS")
    
    with tab3:
        st.subheader("🔍 Détection Anomalies")
        
        st.info("""
        **Apprentissage Non-Supervisé**
        
        Détection d'événements rares ou inattendus sans labellisation préalable.
        Utilise Autoencoders pour apprendre distribution normale des données.
        """)
        
        if st.button("🔍 Rechercher Anomalies", type="primary", use_container_width=True):
            with st.spinner("Analyse anomalies en cours..."):
                import time
                time.sleep(3)
                
                # Simulation détection anomalies
                n_anomalies = np.random.randint(5, 20)
                
                st.success(f"✅ {n_anomalies} anomalies détectées!")
                
                anomalies = []
                for i in range(n_anomalies):
                    anomalies.append({
                        'ID': f"ANOM_{i+1:03d}",
                        'Type': np.random.choice(['Énergie Extrême', 'Pattern Inhabituel', 'Multi-Site', 'Timing Anormal']),
                        'Score': np.random.uniform(0.85, 0.99),
                        'Énergie (keV)': np.random.uniform(100, 5000)
                    })
                
                df_anomalies = pd.DataFrame(anomalies)
                st.dataframe(df_anomalies, use_container_width=True)
                
                st.warning(f"⚠️ {n_anomalies} événements nécessitent investigation manuelle")
    
    with tab4:
        st.subheader("📊 Résultats & Performance")
        
        st.write("### 🎯 Métriques Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "96.3%")
            st.metric("Precision", "94.8%")
        
        with col2:
            st.metric("Recall", "95.2%")
            st.metric("F1-Score", "95.0%")
        
        with col3:
            st.metric("AUC-ROC", "0.982")
            st.metric("False Positive", "3.7%")
        
        with col4:
            st.metric("False Negative", "4.8%")
            st.metric("Matthews Corr", "0.91")
        
        st.markdown("---")
        
        # Matrice confusion
        st.write("### 📊 Matrice de Confusion")
        
        confusion_matrix = np.array([[8520, 312], [428, 9740]])
        
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=['Pred Signal', 'Pred Background'],
            y=['True Signal', 'True Background'],
            colorscale='Viridis',
            text=confusion_matrix,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title="Matrice de Confusion",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: COMPUTING QUANTIQUE ====================
elif page == "⚛️ Computing Quantique":
    st.header("⚛️ Computing Quantique pour Physique Particules")
    
    st.info("""
    **Avantages Quantiques**
    
    - Calcul sections efficaces complexes
    - Simulation interactions multi-particules
    - Optimisation paramètres détection
    - Recherche espace des phases
    - Cryptographie quantique données
    """)
    
    tab1, tab2, tab3 = st.tabs(["⚛️ Simulations", "🔬 Calculs", "📊 Résultats"])
    
    with tab1:
        st.subheader("⚛️ Simulations Quantiques")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            sim_type = st.selectbox("Type Simulation",
                ["Section Efficace WIMP-Nucléon", "Oscillations Neutrinos", 
                 "Désintégration Double Bêta", "Diffusion Compton", "Production Paires"])
            
            n_qubits = st.slider("Nombre Qubits", 8, 128, 64, 8)
            
            quantum_algorithm = st.selectbox("Algorithme",
                ["VQE (Variational Quantum Eigensolver)", "QAOA", 
                 "Grover", "Quantum Annealing", "Shor"])
        
        with col2:
            st.write("### 🎯 Paramètres")
            st.metric("Qubits", n_qubits)
            st.metric("Profondeur Circuit", n_qubits * 2)
            st.metric("Gates", n_qubits * 10)
            
            speedup = 2 ** (n_qubits / 10)
            st.metric("Speedup Estimé", f"{speedup:.1f}x")
        
        if st.button("⚛️ Lancer Simulation Quantique", type="primary", use_container_width=True):
            with st.spinner(f"Simulation quantique {n_qubits} qubits..."):
                import time
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                
                progress_bar.empty()
            
            # Résultat simulation
            result = {
                'cross_section': quantum_compute_cross_section(100, 931.5),
                'uncertainty': np.random.uniform(0.01, 0.05),
                'fidelity': np.random.uniform(0.95, 0.99),
                'execution_time': np.random.uniform(5, 20)
            }
            
            st.success("✅ Simulation quantique terminée!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Section Efficace", f"{result['cross_section']:.2e} cm²")
            with col2:
                st.metric("Incertitude", f"{result['uncertainty']*100:.2f}%")
            with col3:
                st.metric("Fidélité", f"{result['fidelity']:.3f}")
            
            # Sauvegarder
            st.session_state.dark_matter_lab['quantum_simulations'].append({
                'timestamp': datetime.now().isoformat(),
                'type': sim_type,
                'qubits': n_qubits,
                'result': result
            })
            
            log_event(f"Simulation quantique: {sim_type}", "SUCCESS")
    
    with tab2:
        st.subheader("🔬 Calculs Quantiques Avancés")
        
        st.write("### ⚛️ Calculateur Section Efficace Quantique")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wimp_mass_calc = st.number_input("Masse WIMP (GeV/c²)", 1.0, 10000.0, 100.0, 1.0)
        
        with col2:
            target_nucleus = st.selectbox("Noyau Cible",
                ["Xénon-131", "Germanium-76", "Argon-40", "Sodium-23"])
            
            nucleus_masses = {
                "Xénon-131": 122.0,
                "Germanium-76": 70.9,
                "Argon-40": 37.2,
                "Sodium-23": 21.4
            }
            
            target_mass = nucleus_masses[target_nucleus]
        
        with col3:
            coupling_constant = st.number_input("Constante Couplage", 
                                               value=1e-6, format="%.2e")
        
        if st.button("🔬 Calculer Section Efficace", use_container_width=True):
            with st.spinner("Calcul quantique..."):
                import time
                time.sleep(2)
                
                # Calcul avec correction quantique
                sigma = quantum_compute_cross_section(wimp_mass_calc, target_mass)
                sigma *= coupling_constant * 1e39  # Facteur normalisation
                
                st.success("✅ Calcul complété!")
                
                st.write("### 📊 Résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("σ (SI)", f"{sigma:.2e} cm²")
                    st.metric("σ (SD)", f"{sigma * 0.3:.2e} cm²")
                
                with col2:
                    # Taux événements attendu
                    rate = calculate_wimp_interaction_rate(wimp_mass_calc, sigma, 1000)
                    st.metric("Taux (1 tonne)", f"{rate:.2e} /jour")
                    st.metric("Events/an", f"{rate * 365:.1f}")
    
    with tab3:
        st.subheader("📊 Historique Simulations Quantiques")
        
        if not st.session_state.dark_matter_lab['quantum_simulations']:
            st.info("Aucune simulation quantique effectuée")
        else:
            sim_data = []
            for sim in st.session_state.dark_matter_lab['quantum_simulations']:
                sim_data.append({
                    'Timestamp': sim['timestamp'][:19],
                    'Type': sim['type'],
                    'Qubits': sim['qubits'],
                    'Résultat': f"{sim['result']['cross_section']:.2e}",
                    'Fidélité': f"{sim['result']['fidelity']:.3f}"
                })
            
            df_sims = pd.DataFrame(sim_data)
            st.dataframe(df_sims, use_container_width=True)

# ==================== PAGE: BIO-COMPUTING ====================
elif page == "🧬 Bio-Computing":
    st.header("🧬 Bio-Computing pour Analyse Données")
    
    st.info("""
    **Computing à Base d'ADN**
    
    - Stockage massif de données (1 exaoctet/mm³)
    - Parallélisation extrême (10²⁰ opérations simultanées)
    - Efficacité énergétique exceptionnelle
    - Pattern matching biologique
    - Auto-réparation des erreurs
    """)
    
    tab1, tab2, tab3 = st.tabs(["🧬 Stockage ADN", "🔬 Calculs Bio", "📊 Performance"])
    
    with tab1:
        st.subheader("🧬 Stockage Données sur ADN")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### 💾 Encoder Données")
            
            data_type = st.selectbox("Type Données",
                ["Events Détecteur", "Résultats Analyse", "Images", "Logs"])
            
            data_size_gb = st.number_input("Taille Données (GB)", 1, 1000, 10, 1)
            
            encoding_scheme = st.selectbox("Schéma Encodage",
                ["Base4 (ATCG)", "Base8 (Extended)", "Ternary", "Binary-to-DNA"])
            
            error_correction = st.selectbox("Correction Erreurs",
                ["Reed-Solomon", "Fountain Codes", "LDPC", "Hamming"])
        
        with col2:
            st.write("### 📊 Estimation")
            
            # 1 octet = ~4 paires de bases
            dna_bases = data_size_gb * 1e9 * 4
            
            st.metric("Paires Bases", f"{dna_bases:.2e}")
            st.metric("Brins ADN", f"{dna_bases / 1e6:.0f}M")
            
            # Volume physique (très compact)
            volume_mm3 = data_size_gb / 1e12  # 1 exaoctet/mm³
            st.metric("Volume", f"{volume_mm3:.6f} mm³")
        
        if st.button("🧬 Encoder sur ADN", type="primary", use_container_width=True):
            with st.spinner(f"Encodage {data_size_gb} GB sur ADN..."):
                import time
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.03)
                    progress_bar.progress(i + 1)
                
                progress_bar.empty()
            
            st.success(f"✅ {data_size_gb} GB encodés sur ADN!")
            
            # Générer séquence exemple
            bases = ['A', 'T', 'C', 'G']
            sequence = ''.join(np.random.choice(bases, 200))
            
            st.write("### 🧬 Séquence ADN (extrait):")
            st.code(sequence, language="text")
            
            st.info(f"""
            **Avantages:**
            - Densité: {data_size_gb * 1000:.0f}x disque dur
            - Durabilité: > 1000 ans
            - Pas de maintenance électrique
            - Copie parfaite par PCR
            """)
            
            log_event(f"Données encodées ADN: {data_size_gb} GB", "SUCCESS")
    
    with tab2:
        st.subheader("🔬 Calculs Biologiques")
        
        st.write("### 🧬 Pattern Matching Bio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pattern_length = st.slider("Longueur Pattern", 10, 1000, 100, 10)
            n_patterns = st.number_input("Nombre Patterns", 1, 1000, 10, 1)
        
        with col2:
            parallelism = st.metric("Parallélisme", "10²⁰ ops")
            st.metric("Efficacité Énergétique", "10⁶x CPU")
        
        if st.button("🔬 Lancer Recherche Bio", use_container_width=True):
            with st.spinner("Recherche patterns biologiques..."):
                import time
                time.sleep(2)
                
                matches_found = np.random.randint(100, 1000)
                
                st.success(f"✅ {matches_found} patterns trouvés!")
                
                st.session_state.dark_matter_lab['bio_computing_tasks'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'pattern_matching',
                    'patterns': n_patterns,
                    'matches': matches_found
                })
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Matches", matches_found)
                with col2:
                    st.metric("Temps", "1.8 s")
                with col3:
                    speedup = np.random.uniform(1e6, 1e9)
                    st.metric("Speedup vs CPU", f"{speedup:.2e}x")
    
    with tab3:
        st.subheader("📊 Performance Bio-Computing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### ⚡ Comparaison Technologies")
            
            comparison = pd.DataFrame({
                'Technologie': ['CPU', 'GPU', 'FPGA', 'ASIC', 'Quantique', 'ADN'],
                'Ops/s': [1e9, 1e12, 1e11, 1e13, 1e15, 1e20],
                'Énergie (W/GFLOPS)': [100, 10, 5, 1, 0.01, 0.0001]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=comparison['Technologie'],
                y=np.log10(comparison['Ops/s']),
                name='Ops/s (log10)',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Performance Comparée (échelle log)",
                yaxis_title="log10(Ops/s)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### 💾 Densité Stockage")
            
            storage_density = pd.DataFrame({
                'Support': ['HDD', 'SSD', 'Flash', 'Holographie', 'ADN'],
                'Densité (TB/cm³)': [0.001, 0.01, 0.1, 1, 1000000]
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=storage_density['Support'],
                y=np.log10(storage_density['Densité (TB/cm³)']),
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="Densité Stockage (échelle log)",
                yaxis_title="log10(TB/cm³)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: ANALYTICS ====================
elif page == "📊 Analytics":
    st.header("📊 Analytics & Statistiques Globales")
    
    tab1, tab2, tab3 = st.tabs(["📈 Vue Globale", "🎯 Détections", "⚛️ Particules"])
    
    with tab1:
        st.subheader("📈 Vue d'Ensemble")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Détecteurs", total_detectors, "+2")
        with col2:
            st.metric("Détections", total_detections, "+1,247")
        with col3:
            st.metric("WIMPs", total_wimps, "+34")
        with col4:
            neutrino_count = len(st.session_state.dark_matter_lab['neutrino_events'])
            st.metric("Neutrinos", neutrino_count, "+892")
        
        st.markdown("---")
        
        # Timeline détections
        st.write("### 📊 Timeline Détections")
        
        if st.session_state.dark_matter_lab['detections']:
            # Créer données temporelles
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            detections_per_day = np.random.poisson(50, 30)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=detections_per_day,
                mode='lines+markers',
                name='Détections',
                line=dict(color='cyan', width=3)
            ))
            
            fig.update_layout(
                title="Détections par Jour (30 derniers jours)",
                xaxis_title="Date",
                yaxis_title="Nombre Détections",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas encore de données de détection")
    
    with tab2:
        st.subheader("🎯 Analyse Détections")
        
        if st.session_state.dark_matter_lab['detections']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Distribution Types")
                
                particle_types = [d.get('particle_type', 'Unknown') 
                                 for d in st.session_state.dark_matter_lab['detections']]
                type_counts = pd.Series(particle_types).value_counts()
                
                fig = go.Figure(data=[go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    hole=.3
                )])
                
                fig.update_layout(
                    title="Types Particules Détectées",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### ⚡ Spectre Énergétique Global")
                
                energies = [d.get('energy_kev', 0) 
                           for d in st.session_state.dark_matter_lab['detections']]
                
                fig = go.Figure(data=[go.Histogram(
                    x=energies,
                    nbinsx=50,
                    marker_color='purple'
                )])
                
                fig.update_layout(
                    title="Distribution Énergie",
                    xaxis_title="Énergie (keV)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚛️ Base Données Particules")
        
        st.write("### 🎯 Statistiques par Type")
        
        particle_stats = {
            'WIMPs': total_wimps,
            'Neutrinos': len(st.session_state.dark_matter_lab['neutrino_events']),
            'Xénon Decays': len(st.session_state.dark_matter_lab['xenon_decays']),
            'Background': total_detections - total_wimps
        }
        
        fig = go.Figure(data=[go.Bar(
            x=list(particle_stats.keys()),
            y=list(particle_stats.values()),
            marker_color=['#4B0082', '#FF6B6B', '#4ECDC4', '#95E1D3']
        )])
        
        fig.update_layout(
            title="Distribution Particules Détectées",
            xaxis_title="Type",
            yaxis_title="Count",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Paramètres Système")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Configuration", "💾 Données", "🔒 Sécurité"])
    
    with tab1:
        st.subheader("🔧 Configuration Globale")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🎯 Paramètres Physiques")
            
            dark_matter_density = st.number_input("Densité Matière Noire Locale (GeV/cm³)",
                                                 0.1, 1.0, 0.3, 0.01)
            
            wimp_velocity = st.number_input("Vitesse Moyenne WIMPs (km/s)",
                                           100, 400, 220, 10)
            
            earth_velocity = st.number_input("Vitesse Terre (km/s)",
                                            200, 300, 232, 1)
        
        with col2:
            st.write("### ⚙️ Paramètres Détection")
            
            global_threshold = st.slider("Seuil Global (keV)", 0.1, 10.0, 1.0, 0.1)
            
            coincidence_window = st.number_input("Fenêtre Coïncidence (ns)",
                                                100, 10000, 1000, 100)
            
            veto_threshold = st.slider("Seuil Veto (keV)", 10, 1000, 100, 10)
        
        if st.button("💾 Sauvegarder Configuration", type="primary"):
            st.success("✅ Configuration sauvegardée!")
    
    with tab2:
        st.subheader("💾 Gestion Données")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Détecteurs", total_detectors)
            st.metric("Détections", total_detections)
        
        with col2:
            st.metric("WIMPs", total_wimps)
            st.metric("Neutrinos", len(st.session_state.dark_matter_lab['neutrino_events']))
        
        with col3:
            st.metric("Xénon", len(st.session_state.dark_matter_lab['xenon_decays']))
            st.metric("Logs", len(st.session_state.dark_matter_lab['log']))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Exporter Données", use_container_width=True):
                st.success("✅ Export lancé!")
        
        with col2:
            if st.button("🗑️ Réinitialiser", use_container_width=True):
                if st.checkbox("Confirmer réinitialisation"):
                    st.session_state.dark_matter_lab = {
                        'detectors': {},
                        'experiments': {},
                        'detections': [],
                        'wimps_candidates': [],
                        'neutrino_events': [],
                        'xenon_decays': [],
                        'ai_models': {},
                        'quantum_simulations': [],
                        'bio_computing_tasks': [],
                        'analysis_results': {},
                        'particles_database': {},
                        'collaborations': {},
                        'publications': [],
                        'log': []
                    }
                    st.success("✅ Données réinitialisées!")
                    st.rerun()
    
    with tab3:
        st.subheader("🔒 Sécurité & Accès")
        
        st.info("""
        **Niveaux d'Accès:**
        
        - 👤 Utilisateur: Lecture seule
        - 👨‍🔬 Chercheur: Lecture + Analyse
        - 👨‍💼 Chef Projet: Lecture + Analyse + Configuration
        - 🔑 Admin: Accès complet
        """)
        
        user_level = st.selectbox("Votre Niveau", 
                                  ["Utilisateur", "Chercheur", "Chef Projet", "Admin"])
        st.write(f"**Niveau Actuel:** {user_level}")
# ==================== PAGE: DÉSINTÉGRATIONS XÉNON ====================
elif page == "⚛️ Désintégrations Xénon":
    st.header("⚛️ Désintégrations Isotopes Xénon")
    
    st.info("""
    **Désintégrations Rares du Xénon**
    
    - **¹³⁶Xe → ¹³⁶Ba**: Double bêta sans neutrinos (T₁/₂ > 10²¹ ans)
    - **¹³⁴Xe**: Désintégrations β⁻ et EC
    - **¹³²Xe**: Isotope stable utilisé comme référence
    
    Ces désintégrations ultra-rares sont essentielles pour comprendre la physique 
    au-delà du Modèle Standard et rechercher violation nombre leptonique.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧪 Simulation", "📊 Événements", "📈 Recherche 0νββ",  "📈 Analyses"])
    
    with tab1:
        st.subheader("🧪 Simuler Désintégrations Xénon")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            isotope = st.selectbox("Isotope Xénon",
                ["Xe-136", "Xe-134", "Xe-132", "Xe-131", "Xe-129"])
            
            simulation_time = st.slider("Temps Simulation (heures)", 1, 8760, 100)
            
            xenon_mass_kg = st.number_input("Masse Xénon (kg)", 100, 10000, 1000, 100)
        
        with col2:
            st.write("### 📊 Propriétés")
            
            half_lives = {
                "Xe-136": "> 2.11×10²¹ ans",
                "Xe-134": "Stable",
                "Xe-132": "Stable",
                "Xe-131": "Stable",
                "Xe-129": "Stable"
            }

            #  Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Distribution Saveurs")
                
                flavors = [e['flavor'] for e in st.session_state.dark_matter_lab['neutrino_events']]
                flavor_counts = pd.Series(flavors).value_counts()
                
                fig = go.Figure(data=[go.Bar(
                    x=flavor_counts.index,
                    y=flavor_counts.values,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )])
                
                fig.update_layout(
                    title="Saveurs Neutrinos",
                    xaxis_title="Saveur",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### ⚡ Spectre Énergétique")
                
                energies = [e['energy_mev'] for e in st.session_state.dark_matter_lab['neutrino_events']]
                
                fig = go.Figure(data=[go.Histogram(
                    x=energies,
                    nbinsx=40,
                    marker_color='#9D4EDD'
                )])
                
                fig.update_layout(
                    title="Distribution Énergie",
                    xaxis_title="Énergie (MeV)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)

                st.write(f"**Demi-vie:** {half_lives[isotope]}")
            
            if isotope == "Xe-136":
                st.write("**Transition:** 0νββ")
                st.write("**Q-value:** 2458 keV")
            
            st.metric("Masse", f"{xenon_mass_kg} kg")
        
        if st.button("⚛️ Lancer Simulation", type="primary", use_container_width=True):
            with st.spinner(f"Simulation {simulation_time}h en cours..."):
                import time
                progress_bar = st.progress(0)
                
                # Simulation désintégrations
                events = simulate_xenon_decay(isotope, simulation_time)
                
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                
                # Sauvegarder
                st.session_state.dark_matter_lab['xenon_decays'].extend(events)
                
                progress_bar.empty()
            
            st.success(f"✅ {len(events)} désintégrations {isotope} simulées!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Events Totaux", len(events))
            with col2:
                double_beta = sum(1 for e in events if e.get('type') == 'double_beta')
                st.metric("Double β", double_beta)
            with col3:
                mean_energy = np.mean([e['energy_kev'] for e in events])
                st.metric("Énergie Moy", f"{mean_energy:.1f} keV")
            
            log_event(f"Simulation Xénon: {len(events)} events", "SUCCESS")
    
    with tab2:
        st.subheader("📊 Événements Xénon Détectés")
        
        if not st.session_state.dark_matter_lab['xenon_decays']:
            st.info("Aucune désintégration simulée. Lancez une simulation.")
        else:
            # Table
            xenon_data = []
            for event in st.session_state.dark_matter_lab['xenon_decays'][-100:]:
                xenon_data.append({
                    'Timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Isotope': event['isotope'],
                    'Type': event['type'],
                    'Énergie (keV)': f"{event['energy_kev']:.2f}",
                    'X': f"{event['position']['x']:.1f}",
                    'Y': f"{event['position']['y']:.1f}",
                    'Z': f"{event['position']['z']:.1f}"
                })
            
            df_xenon = pd.DataFrame(xenon_data)
            st.dataframe(df_xenon, use_container_width=True)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Distribution Énergie")
                
                energies = [e['energy_kev'] for e in st.session_state.dark_matter_lab['xenon_decays']]
                
                fig = go.Figure(data=[go.Histogram(
                    x=energies,
                    nbinsx=50,
                    marker_color='#06FFA5'
                )])
                
                fig.update_layout(
                    title=f"Spectre Énergétique {isotope}",
                    xaxis_title="Énergie (keV)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### 🎯 Distribution Spatiale")
                
                positions = [e['position'] for e in st.session_state.dark_matter_lab['xenon_decays'][-200:]]
                x_pos = [p['x'] for p in positions]
                y_pos = [p['y'] for p in positions]
                
                fig = go.Figure(data=[go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=energies[-200:],
                        colorscale='Viridis',
                        showscale=True
                    )
                )])
                
                fig.update_layout(
                    title="Position Events (vue XY)",
                    xaxis_title="X (cm)",
                    yaxis_title="Y (cm)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📈 Recherche 0νββ (Double Bêta sans Neutrinos)")
        
        st.info("""
        **0νββ - Saint Graal Physique Neutrinos**
        
        La désintégration double bêta sans neutrinos violerait la conservation 
        du nombre leptonique et prouverait que le neutrino est sa propre antiparticule 
        (particule de Majorana). Non observée à ce jour.
        """)
        
        if st.session_state.dark_matter_lab['xenon_decays']:
            # Chercher pic à 2458 keV
            energies = [e['energy_kev'] for e in st.session_state.dark_matter_lab['xenon_decays']]
            
            # ROI autour Q-value
            roi_events = [e for e in energies if 2400 < e < 2500]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Events dans ROI", len(roi_events))
                st.write("ROI: 2400-2500 keV")
            
            with col2:
                if len(roi_events) > 0:
                    mean_roi = np.mean(roi_events)
                    st.metric("Énergie Moy ROI", f"{mean_roi:.1f} keV")
                else:
                    st.metric("Énergie Moy ROI", "N/A")
            
            with col3:
                background_rate = len([e for e in energies if e < 2400]) / len(energies) if energies else 0
                st.metric("Taux Background", f"{background_rate:.3f}")
            
            # Spectre haute résolution ROI
            st.write("### 🔍 Spectre Haute Résolution (ROI)")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=energies,
                xbins=dict(start=2000, end=3000, size=10),
                marker_color='rgba(75, 0, 130, 0.7)',
                name='Spectre complet'
            ))
            
            # Ligne Q-value
            fig.add_vline(x=2458, line_dash="dash", line_color="red", 
                         annotation_text="Q-value ¹³⁶Xe")
            
            fig.update_layout(
                title="Recherche Signal 0νββ",
                xaxis_title="Énergie (keV)",
                yaxis_title="Count / 10 keV",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcul limite
            if len(roi_events) < 3:
                st.success("✅ Aucun signal 0νββ détecté (compatible avec prédictions)")
                st.info("Limite demi-vie: T₁/₂ > 2.3×10²⁵ ans (90% CL)")
            else:
                st.warning(f"⚠️ {len(roi_events)} events dans ROI - Nécessite investigation")


    with tab4:
        st.subheader("📈 Analyses Oscillations Neutrinos")
        
        st.info("""
        **Oscillations Neutrinos**
        
        Les neutrinos changent de saveur en se propageant (νₑ → νμ → ντ).
        Ce phénomène quantique prouve que les neutrinos ont une masse non nulle.
        """)
        
        if len(st.session_state.dark_matter_lab['neutrino_events']) > 50:
            # Calcul ratios
            flavors = [e['flavor'] for e in st.session_state.dark_matter_lab['neutrino_events']]
            
            electron_ratio = flavors.count('electron') / len(flavors)
            muon_ratio = flavors.count('muon') / len(flavors)
            tau_ratio = flavors.count('tau') / len(flavors)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Ratio νₑ", f"{electron_ratio:.3f}")
                st.write("Attendu: ~0.33")
            
            with col2:
                st.metric("Ratio νμ", f"{muon_ratio:.3f}")
                st.write("Attendu: ~0.33")
            
            with col3:
                st.metric("Ratio ντ", f"{tau_ratio:.3f}")
                st.write("Attendu: ~0.33")
            
            # Test chi-carré
            observed = [flavors.count('electron'), flavors.count('muon'), flavors.count('tau')]
            expected = [len(flavors)/3] * 3
            
            chi2 = sum((o - e)**2 / e for o, e in zip(observed, expected))
            
            st.write(f"### χ² Test: {chi2:.2f}")
            
            if chi2 < 5.99:  # 95% confidence, 2 dof
                st.success("✅ Distribution compatible avec oscillations maximales")
            else:
                st.warning("⚠️ Déviation statistique détectée")
        else:
            st.warning("Données insuffisantes pour analyse oscillations")

# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal Système (20 dernières entrées)"):
    if st.session_state.dark_matter_lab['log']:
        for event in st.session_state.dark_matter_lab['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️" if level == "WARNING" else "❌"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🌌 Dark Matter Research Platform</h3>
        <p>Recherche Avancée Matière Noire • WIMPs • Neutrinos • Xénon</p>
        <p><small>IA • Computing Quantique • Bio-Computing</small></p>
        <p><small>Version 1.0.0 | Laboratoire Virtuel Physique des Particules</small></p>
        <p><small>🌌 Découvrir l'Univers Invisible © 2024</small></p>
    </div>
""", unsafe_allow_html=True)