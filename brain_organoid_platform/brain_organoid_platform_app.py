"""
🧠 Brain Organoid Computing Platform - Biocomputing Research
Organoïdes Cérébraux • Neurones Humains • Biocomputing • Neuroplasticité

Installation:
pip install streamlit pandas plotly numpy scipy networkx

Lancement:
streamlit run brain_organoid_platform_app.py
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
    page_title="🧠 Brain Organoid Computing",
    page_icon="🧠",
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
        background: linear-gradient(90deg, #FF6B9D 0%, #C06C84 30%, #6C5B7B 60%, #355C7D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: neural-pulse 2s ease-in-out infinite alternate;
    }
    @keyframes neural-pulse {
        from { filter: drop-shadow(0 0 15px #FF6B9D); }
        to { filter: drop-shadow(0 0 35px #355C7D); }
    }
    .neural-card {
        border: 3px solid #FF6B9D;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.1) 0%, rgba(53, 92, 125, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(255, 107, 157, 0.4);
        transition: all 0.3s;
    }
    .neural-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(192, 108, 132, 0.6);
    }
    .neuron-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
        background: linear-gradient(90deg, #FF6B9D 0%, #C06C84 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 107, 157, 0.4);
    }
    .synapse-active {
        animation: synapse-fire 0.8s infinite;
    }
    @keyframes synapse-fire {
        0%, 100% { opacity: 0.7; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES BIOLOGIQUES ====================
BIO_CONSTANTS = {
    'neuron_diameter_um': 20,  # Diamètre neurone (μm)
    'synapse_density': 10000,  # Synapses par neurone
    'action_potential_mv': 100,  # Potentiel d'action (mV)
    'resting_potential_mv': -70,  # Potentiel repos (mV)
    'firing_threshold_mv': -55,  # Seuil déclenchement (mV)
    'refractory_period_ms': 2,  # Période réfractaire (ms)
    'synaptic_delay_ms': 0.5,  # Délai synaptique (ms)
    'glucose_consumption_umol': 5.5,  # Glucose (μmol/min/100g)
    'oxygen_consumption_ml': 3.5,  # O2 (ml/min/100g)
    'neuron_growth_rate': 0.1,  # Taux croissance/jour
    'max_organoid_size_mm': 5,  # Taille max organoïde
}

NEURON_TYPES = {
    'Pyramidal': {
        'description': 'Neurones excitateurs principaux',
        'percentage': 80,
        'neurotransmitter': 'Glutamate',
        'firing_rate': '1-20 Hz',
        'color': '#FF6B9D'
    },
    'Interneuron': {
        'description': 'Neurones inhibiteurs (GABA)',
        'percentage': 15,
        'neurotransmitter': 'GABA',
        'firing_rate': '10-100 Hz',
        'color': '#C06C84'
    },
    'Dopaminergic': {
        'description': 'Neurones dopaminergiques',
        'percentage': 3,
        'neurotransmitter': 'Dopamine',
        'firing_rate': '1-10 Hz',
        'color': '#6C5B7B'
    },
    'Serotonergic': {
        'description': 'Neurones sérotoninergiques',
        'percentage': 2,
        'neurotransmitter': 'Serotonin',
        'firing_rate': '1-5 Hz',
        'color': '#355C7D'
    }
}

# ==================== INITIALISATION SESSION STATE ====================
if 'organoid_lab' not in st.session_state:
    st.session_state.organoid_lab = {
        'organoids': {},
        'neurons': {},
        'synapses': {},
        'neural_networks': {},
        'experiments': [],
        'recordings': [],
        'stimulations': [],
        'training_sessions': [],
        'computations': [],
        'culture_media': {},
        'growth_factors': {},
        'pharmacology': [],
        'electrophysiology': [],
        'imaging_sessions': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str, level: str = "INFO"):
    """Enregistrer événement"""
    st.session_state.organoid_lab['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_neuron_count(organoid_size_mm: float) -> int:
    """Calculer nombre de neurones selon taille"""
    # Densité ~100,000 neurones/mm³
    volume = (4/3) * np.pi * (organoid_size_mm/2)**3
    return int(volume * 100000)

def simulate_action_potential(duration_ms: float = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Simuler potentiel d'action"""
    t = np.linspace(0, duration_ms, 1000)
    
    # Modèle Hodgkin-Huxley simplifié
    depolarization = t < 1
    repolarization = (t >= 1) & (t < 3)
    hyperpolarization = (t >= 3) & (t < 4)
    
    V = np.zeros_like(t)
    V[depolarization] = -70 + 170 * (t[depolarization] / 1)
    V[repolarization] = 100 - 150 * ((t[repolarization] - 1) / 2)
    V[hyperpolarization] = -50 - 30 * ((t[hyperpolarization] - 3) / 1)
    V[t >= 4] = -70
    
    return t, V

def calculate_synaptic_strength(pre_activity: float, post_activity: float, 
                               stdp_window_ms: float = 20) -> float:
    """Calculer force synaptique (STDP - Spike-Timing Dependent Plasticity)"""
    # STDP: renforcement si pré avant post, affaiblissement sinon
    delta_t = pre_activity - post_activity
    
    if abs(delta_t) < stdp_window_ms:
        if delta_t > 0:  # Pré avant post → LTP (potentialisation)
            strength = 0.1 * np.exp(-abs(delta_t) / 10)
        else:  # Post avant pré → LTD (dépression)
            strength = -0.1 * np.exp(-abs(delta_t) / 10)
    else:
        strength = 0
    
    return strength

def calculate_metabolic_rate(n_neurons: int, firing_rate_hz: float) -> Dict:
    """Calculer taux métabolique"""
    # Consommation basale
    glucose_base = BIO_CONSTANTS['glucose_consumption_umol'] * (n_neurons / 1e6)
    oxygen_base = BIO_CONSTANTS['oxygen_consumption_ml'] * (n_neurons / 1e6)
    
    # Augmentation avec activité
    activity_factor = 1 + (firing_rate_hz / 10)
    
    return {
        'glucose_umol_min': glucose_base * activity_factor,
        'oxygen_ml_min': oxygen_base * activity_factor,
        'atp_production': glucose_base * activity_factor * 38,  # 38 ATP par glucose
        'heat_production_mw': n_neurons * firing_rate_hz * 0.01  # mW
    }

def simulate_network_activity(n_neurons: int, connectivity: float, 
                              duration_s: float = 1) -> np.ndarray:
    """Simuler activité réseau neuronal"""
    dt = 0.001  # 1 ms
    steps = int(duration_s / dt)
    
    # Matrice connectivité
    connections = np.random.random((n_neurons, n_neurons)) < connectivity
    np.fill_diagonal(connections, 0)
    
    # État neurones
    activity = np.zeros((n_neurons, steps))
    activity[:, 0] = np.random.random(n_neurons) > 0.9  # Activité initiale
    
    # Simulation
    for t in range(1, steps):
        # Input synaptique
        synaptic_input = connections @ activity[:, t-1]
        
        # Probabilité firing
        prob_fire = 1 / (1 + np.exp(-(synaptic_input - 2)))
        
        # Firing
        activity[:, t] = np.random.random(n_neurons) < prob_fire
    
    return activity

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🧠 Brain Organoid Computing Platform</h1>', 
           unsafe_allow_html=True)
st.markdown("### Biocomputing • Organoïdes Cérébraux • Neurones Humains • Intelligence Biologique")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/FF6B9D/FFFFFF?text=NeuroLab", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Lab Neuronal",
            "🧬 Créer Organoïde",
            "🔬 Culture & Croissance",
            "⚡ Neurones",
            "🔗 Synapses & Connexions",
            "🌐 Réseaux Neuronaux",
            "📊 Électrophysiologie",
            "🎯 Stimulation",
            "🧠 Apprentissage",
            "💻 Biocomputing",
            "🔬 Expériences",
            "📈 Enregistrements",
            "🧪 Pharmacologie",
            "🔬 Imagerie",
            "📊 Analytics",
            "📡 Monitoring Live",
            "⚖️ Bioéthique",
            "👥 Collaboration",
            "📄 Publications",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 État Lab")
    
    total_organoids = len(st.session_state.organoid_lab['organoids'])
    total_neurons = sum(o.get('neuron_count', 0) 
                       for o in st.session_state.organoid_lab['organoids'].values())
    total_experiments = len(st.session_state.organoid_lab['experiments'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧠 Organoïdes", total_organoids)
        st.metric("⚡ Neurones", f"{total_neurons:,}")
    with col2:
        st.metric("🔬 Expériences", total_experiments)
        st.metric("📈 Recordings", len(st.session_state.organoid_lab['recordings']))

# ==================== PAGE: LAB NEURONAL ====================
if page == "🏠 Lab Neuronal":
    st.header("🏠 Laboratoire Neuronal Central")
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="neural-card"><h2>🧠</h2><h3>{total_organoids}</h3><p>Organoïdes</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        avg_size = np.mean([o.get('size_mm', 0) 
                           for o in st.session_state.organoid_lab['organoids'].values()]) if total_organoids > 0 else 0
        st.markdown(f'<div class="neural-card"><h2>📏</h2><h3>{avg_size:.2f}</h3><p>Taille Moy (mm)</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="neural-card"><h2>⚡</h2><h3>{total_neurons:,}</h3><p>Neurones</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        total_synapses = total_neurons * BIO_CONSTANTS['synapse_density']
        st.markdown(f'<div class="neural-card"><h2>🔗</h2><h3>{total_synapses/1e9:.2f}B</h3><p>Synapses</p></div>', 
                   unsafe_allow_html=True)
    
    with col5:
        viability = np.random.uniform(85, 98) if total_organoids > 0 else 0
        st.markdown(f'<div class="neural-card"><h2>✓</h2><h3>{viability:.1f}%</h3><p>Viabilité</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Types de neurones
    st.subheader("⚛️ Types de Neurones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔬 Distribution Types Neuronaux")
        
        for ntype, info in NEURON_TYPES.items():
            with st.expander(f"⚡ {ntype} ({info['percentage']}%)"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Neurotransmetteur:** {info['neurotransmitter']}")
                st.write(f"**Fréquence:** {info['firing_rate']}")
                st.markdown(f"**Couleur:** <span style='color:{info['color']}'>●●●</span>", 
                           unsafe_allow_html=True)
    
    with col2:
        st.write("### 📊 Répartition")
        
        fig = go.Figure(data=[go.Pie(
            labels=list(NEURON_TYPES.keys()),
            values=[info['percentage'] for info in NEURON_TYPES.values()],
            marker=dict(colors=[info['color'] for info in NEURON_TYPES.values()]),
            hole=0.4
        )])
        
        fig.update_layout(
            title="Distribution Types Neuronaux",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Visualisation potentiel d'action
    st.subheader("⚡ Potentiel d'Action")
    
    if st.button("🔬 Simuler Potentiel d'Action", type="primary"):
        t, V = simulate_action_potential()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=t, y=V,
            mode='lines',
            line=dict(color='#FF6B9D', width=3),
            fill='tozeroy'
        ))
        
        fig.add_hline(y=-70, line_dash="dash", line_color="white",
                     annotation_text="Repos (-70 mV)")
        fig.add_hline(y=-55, line_dash="dash", line_color="yellow",
                     annotation_text="Seuil (-55 mV)")
        
        fig.update_layout(
            title="Potentiel d'Action Neuronal",
            xaxis_title="Temps (ms)",
            yaxis_title="Potentiel (mV)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Repos", "-70 mV")
        with col2:
            st.metric("Pic", "+40 mV")
        with col3:
            st.metric("Durée", "~2 ms")
        with col4:
            st.metric("Amplitude", "110 mV")
    
    st.markdown("---")
    
    # Expériences récentes
    st.subheader("🔬 Expériences Récentes")
    
    if st.session_state.organoid_lab['experiments']:
        for exp in st.session_state.organoid_lab['experiments'][-5:][::-1]:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"🧪 **{exp['name']}**")
                st.write(f"{exp['timestamp'][:19]}")
            
            with col2:
                st.write(f"**Type:** {exp.get('type', 'N/A')}")
            
            with col3:
                status = exp.get('status', 'pending')
                if status == 'completed':
                    st.success("✅ Complété")
                elif status == 'running':
                    st.info("🔄 En cours")
                else:
                    st.warning("⏳ Planifié")
    else:
        st.info("Aucune expérience réalisée")

# ==================== PAGE: CRÉER ORGANOÏDE ====================
elif page == "🧬 Créer Organoïde":
    st.header("🧬 Créer Organoïde Cérébral")
    
    st.info("""
    **Organoïde Cérébral**
    
    Structure 3D de tissu neural dérivé de cellules souches pluripotentes.
    
    **Protocole:**
    1. Cellules souches (iPSC/ESC)
    2. Différenciation neurale
    3. Auto-organisation 3D
    4. Maturation (~3-6 mois)
    """)
    
    with st.form("create_organoid"):
        col1, col2 = st.columns(2)
        
        with col1:
            organoid_name = st.text_input("Nom Organoïde", "NeuroOrg-001")
            
            cell_source = st.selectbox("Source Cellules",
                ["iPSC (Induced Pluripotent)", "ESC (Embryonic)", 
                 "Direct Reprogramming", "Patient-Derived"])
            
            brain_region = st.selectbox("Région Cérébrale",
                ["Cortex", "Hippocampus", "Cerebellum", "Midbrain", 
                 "Whole Brain", "Hypothalamus"])
            
            initial_cells = st.number_input("Cellules Initiales", 
                1000, 100000, 10000, 1000)
        
        with col2:
            culture_duration_days = st.slider("Durée Culture (jours)", 
                30, 365, 90)
            
            growth_factors = st.multiselect("Facteurs Croissance",
                ["EGF", "FGF2", "BDNF", "NGF", "Retinoic Acid", "Shh"],
                default=["EGF", "FGF2"])
            
            oxygen_level = st.slider("Niveau O₂ (%)", 5, 21, 20)
            
            rotation_speed = st.slider("Vitesse Rotation (rpm)", 
                0, 100, 40)
        
        advanced = st.checkbox("Paramètres Avancés")
        
        if advanced:
            col1, col2 = st.columns(2)
            
            with col1:
                glucose_concentration = st.slider("Glucose (mM)", 5.0, 25.0, 17.5, 0.5)
                serum_percentage = st.slider("Sérum (%)", 0, 20, 10)
            
            with col2:
                antibiotics = st.checkbox("Antibiotiques", value=True)
                antioxidants = st.checkbox("Antioxydants", value=True)
        
        if st.form_submit_button("🧬 Créer Organoïde", type="primary"):
            organoid_id = f"org_{len(st.session_state.organoid_lab['organoids']) + 1}"
            
            # Calculs initiaux
            expected_size = min(culture_duration_days * 0.01, BIO_CONSTANTS['max_organoid_size_mm'])
            neuron_count = calculate_neuron_count(expected_size)
            
            organoid = {
                'id': organoid_id,
                'name': organoid_name,
                'cell_source': cell_source,
                'brain_region': brain_region,
                'initial_cells': initial_cells,
                'culture_duration_days': culture_duration_days,
                'growth_factors': growth_factors,
                'size_mm': expected_size,
                'neuron_count': neuron_count,
                'viability': np.random.uniform(85, 98),
                'maturation_stage': 'Early' if culture_duration_days < 60 else 'Intermediate' if culture_duration_days < 120 else 'Mature',
                'oxygen_level': oxygen_level,
                'rotation_speed': rotation_speed,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            
            st.session_state.organoid_lab['organoids'][organoid_id] = organoid
            log_event(f"Organoïde créé: {organoid_name}", "SUCCESS")
            
            st.success(f"✅ Organoïde '{organoid_name}' créé!")
            st.balloons()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Taille", f"{expected_size:.2f} mm")
            with col2:
                st.metric("Neurones", f"{neuron_count:,}")
            with col3:
                st.metric("Viabilité", f"{organoid['viability']:.1f}%")
            with col4:
                st.metric("Maturation", organoid['maturation_stage'])
            
            st.rerun()

# ==================== PAGE: CULTURE & CROISSANCE ====================
elif page == "🔬 Culture & Croissance":
    st.header("🔬 Culture & Suivi Croissance")
    
    if not st.session_state.organoid_lab['organoids']:
        st.warning("⚠️ Aucun organoïde créé. Créez d'abord un organoïde!")
    else:
        selected_organoid = st.selectbox("Sélectionner Organoïde",
            list(st.session_state.organoid_lab['organoids'].keys()),
            format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'])
        
        organoid = st.session_state.organoid_lab['organoids'][selected_organoid]
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Croissance", "🧪 Milieu Culture", "💊 Facteurs", "📊 Métabolisme"])
        
        with tab1:
            st.subheader("📈 Courbe de Croissance")
            
            # Simuler croissance
            days = np.linspace(0, organoid['culture_duration_days'], 100)
            
            # Croissance sigmoïde
            K = BIO_CONSTANTS['max_organoid_size_mm']  # Capacité max
            r = BIO_CONSTANTS['neuron_growth_rate']
            size = K / (1 + np.exp(-r * (days - 60)))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=days, y=size,
                mode='lines',
                line=dict(color='#FF6B9D', width=3),
                fill='tozeroy',
                name='Taille'
            ))
            
            fig.add_vline(x=organoid['culture_duration_days'], 
                         line_dash="dash", line_color="white",
                         annotation_text="Actuel")
            
            fig.update_layout(
                title="Courbe de Croissance Organoïde",
                xaxis_title="Jours en Culture",
                yaxis_title="Taille (mm)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Métriques croissance
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Taille Actuelle", f"{organoid['size_mm']:.2f} mm")
            with col2:
                st.metric("Neurones", f"{organoid['neuron_count']:,}")
            with col3:
                growth_rate = organoid['size_mm'] / organoid['culture_duration_days'] * 100
                st.metric("Taux Croissance", f"{growth_rate:.1f} μm/jour")
            with col4:
                st.metric("Maturation", organoid['maturation_stage'])
        
        with tab2:
            st.subheader("🧪 Milieu de Culture")
            
            st.write("### 📋 Composition Actuelle")
            
            media_composition = {
                'Composant': ['DMEM/F12', 'B27 Supplement', 'N2 Supplement', 
                            'Glutamine', 'Glucose', 'Sérum', 'Antibiotiques'],
                'Concentration': ['Base', '2%', '1%', '2 mM', '17.5 mM', 
                                '10%', '1x'],
                'Fonction': ['Base nutritive', 'Neurones', 'Prolifération',
                           'Synthèse protéines', 'Énergie', 'Facteurs croissance',
                           'Protection']
            }
            
            df_media = pd.DataFrame(media_composition)
            st.dataframe(df_media, use_container_width=True)
            
            if st.button("🔄 Changer Milieu"):
                st.success("✅ Milieu changé!")
                organoid['last_updated'] = datetime.now().isoformat()
                log_event(f"Milieu changé: {organoid['name']}", "INFO")
        
        with tab3:
            st.subheader("💊 Facteurs de Croissance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Actifs:**")
                for factor in organoid.get('growth_factors', []):
                    st.write(f"✅ {factor}")
            
            with col2:
                st.write("**Disponibles:**")
                all_factors = ["EGF", "FGF2", "BDNF", "NGF", "Retinoic Acid", "Shh"]
                for factor in all_factors:
                    if factor not in organoid.get('growth_factors', []):
                        if st.button(f"➕ {factor}", key=f"add_{factor}"):
                            organoid.setdefault('growth_factors', []).append(factor)
                            st.success(f"Ajouté: {factor}")
                            st.rerun()
        
        with tab4:
            st.subheader("📊 Métabolisme")
            
            # Calculer métabolisme
            firing_rate = 5.0  # Hz moyen
            metabolism = calculate_metabolic_rate(organoid['neuron_count'], firing_rate)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🔋 Consommation")
                st.metric("Glucose", f"{metabolism['glucose_umol_min']:.2f} μmol/min")
                st.metric("Oxygène", f"{metabolism['oxygen_ml_min']:.3f} ml/min")
                st.metric("Production ATP", f"{metabolism['atp_production']:.2e}")
            
            with col2:
                st.write("### 🔥 Production")
                st.metric("Chaleur", f"{metabolism['heat_production_mw']:.2f} mW")
                st.metric("CO₂", f"{metabolism['oxygen_ml_min']:.3f} ml/min")
                st.metric("Lactate", f"{metabolism['glucose_umol_min']*0.1:.2f} μmol/min")
            
            # Visualisation métabolisme temps réel
            if st.button("📊 Monitorer Métabolisme", type="primary"):
                time_hours = np.linspace(0, 24, 100)
                
                # Variation circadienne simulée
                glucose_consumption = metabolism['glucose_umol_min'] * (1 + 0.2 * np.sin(2*np.pi*time_hours/24))
                oxygen_consumption = metabolism['oxygen_ml_min'] * (1 + 0.2 * np.sin(2*np.pi*time_hours/24))
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Consommation Glucose", "Consommation O₂")
                )
                
                fig.add_trace(go.Scatter(
                    x=time_hours, y=glucose_consumption,
                    mode='lines',
                    line=dict(color='#FF6B9D', width=2),
                    name='Glucose'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=time_hours, y=oxygen_consumption,
                    mode='lines',
                    line=dict(color='#355C7D', width=2),
                    name='O₂'
                ), row=2, col=1)
                
                fig.update_xaxes(title_text="Temps (heures)", row=2, col=1)
                fig.update_yaxes(title_text="Glucose (μmol/min)", row=1, col=1)
                fig.update_yaxes(title_text="O₂ (ml/min)", row=2, col=1)
                
                fig.update_layout(
                    title="Métabolisme sur 24h",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: NEURONES ====================
elif page == "⚡ Neurones":
    st.header("⚡ Neurones & Activité")
    
    if not st.session_state.organoid_lab['organoids']:
        st.warning("⚠️ Créez d'abord un organoïde")
    else:
        selected_organoid = st.selectbox("Organoïde",
            list(st.session_state.organoid_lab['organoids'].keys()),
            format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
            key="neuron_org")
        
        organoid = st.session_state.organoid_lab['organoids'][selected_organoid]
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Population", "⚡ Activité", "🎯 Types", "🔬 Propriétés"])
        
        with tab1:
            st.subheader("📊 Population Neuronale")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Neurones", f"{organoid['neuron_count']:,}")
                st.metric("Densité", f"{organoid['neuron_count']/(organoid['size_mm']**3):.0f}/mm³")
            
            with col2:
                # Distribution types
                for ntype, info in NEURON_TYPES.items():
                    count = int(organoid['neuron_count'] * info['percentage'] / 100)
                    st.write(f"**{ntype}:** {count:,}")
            
            with col3:
                viability = organoid['viability']
                alive = int(organoid['neuron_count'] * viability / 100)
                dead = organoid['neuron_count'] - alive
                
                st.metric("Vivants", f"{alive:,}")
                st.metric("Morts", f"{dead:,}")
                
                if viability > 90:
                    st.success(f"✅ {viability:.1f}%")
                elif viability > 80:
                    st.warning(f"⚠️ {viability:.1f}%")
                else:
                    st.error(f"❌ {viability:.1f}%")
            
            # Graphique distribution
            fig = go.Figure(data=[go.Bar(
                x=list(NEURON_TYPES.keys()),
                y=[int(organoid['neuron_count'] * info['percentage'] / 100) 
                   for info in NEURON_TYPES.values()],
                marker_color=[info['color'] for info in NEURON_TYPES.values()],
                text=[f"{info['percentage']}%" for info in NEURON_TYPES.values()],
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Distribution Types Neuronaux",
                xaxis_title="Type",
                yaxis_title="Nombre",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("⚡ Activité Neuronale")
            
            recording_duration = st.slider("Durée Enregistrement (s)", 0.1, 10.0, 1.0, 0.1)
            
            if st.button("📊 Enregistrer Activité", type="primary"):
                with st.spinner("Enregistrement en cours..."):
                    # Simuler activité
                    n_neurons_sample = min(100, organoid['neuron_count'])
                    activity = simulate_network_activity(
                        n_neurons_sample, 
                        connectivity=0.1, 
                        duration_s=recording_duration
                    )
                    
                    # Raster plot
                    fig = go.Figure()
                    
                    # Créer raster plot
                    for neuron_idx in range(n_neurons_sample):
                        spike_times = np.where(activity[neuron_idx, :] > 0.5)[0] * 0.001
                        if len(spike_times) > 0:
                            fig.add_trace(go.Scatter(
                                x=spike_times,
                                y=[neuron_idx] * len(spike_times),
                                mode='markers',
                                marker=dict(size=2, color='#FF6B9D'),
                                showlegend=False,
                                hovertemplate=f'Neurone {neuron_idx}<br>Temps: %{{x:.3f}}s<extra></extra>'
                            ))
                    
                    fig.update_layout(
                        title=f"Raster Plot - {n_neurons_sample} Neurones",
                        xaxis_title="Temps (s)",
                        yaxis_title="Neurone #",
                        template="plotly_dark",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    total_spikes = np.sum(activity > 0.5)
                    firing_rate = total_spikes / (n_neurons_sample * recording_duration)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Spikes", f"{total_spikes:,}")
                    with col2:
                        st.metric("Taux Moyen", f"{firing_rate:.2f} Hz")
                    with col3:
                        active_neurons = np.sum(np.any(activity > 0.5, axis=1))
                        st.metric("Neurones Actifs", f"{active_neurons}/{n_neurons_sample}")
                    
                    # Sauvegarder
                    recording = {
                        'organoid_id': selected_organoid,
                        'duration_s': recording_duration,
                        'n_neurons': n_neurons_sample,
                        'total_spikes': int(total_spikes),
                        'firing_rate': float(firing_rate),
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.organoid_lab['recordings'].append(recording)
                    log_event(f"Enregistrement: {organoid['name']}", "SUCCESS")
        
        with tab3:
            st.subheader("🎯 Types Neuronaux Détaillés")
            
            for ntype, info in NEURON_TYPES.items():
                with st.expander(f"⚡ {ntype}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Description:** {info['description']}")
                        st.write(f"**Neurotransmetteur:** {info['neurotransmitter']}")
                        st.write(f"**Fréquence tir:** {info['firing_rate']}")
                        st.write(f"**Pourcentage:** {info['percentage']}%")
                    
                    with col2:
                        count = int(organoid['neuron_count'] * info['percentage'] / 100)
                        st.metric("Nombre", f"{count:,}")
                        
                        # Graphique mini
                        t, V = simulate_action_potential()
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=t, y=V,
                            mode='lines',
                            line=dict(color=info['color'], width=2)
                        ))
                        
                        fig.update_layout(
                            title="Potentiel Action",
                            xaxis_title="ms",
                            yaxis_title="mV",
                            template="plotly_dark",
                            height=200,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🔬 Propriétés Électrophysiologiques")
            
            st.write("### ⚡ Paramètres Moyens")
            
            properties = {
                'Propriété': ['Potentiel repos', 'Seuil déclenchement', 'Amplitude PA', 
                            'Durée PA', 'Période réfractaire', 'Capacité membrane',
                            'Résistance membrane', 'Constante temps'],
                'Valeur': ['-70 mV', '-55 mV', '110 mV', '2 ms', '2 ms', 
                          '100 pF', '100 MΩ', '10 ms'],
                'Variation': ['±5', '±3', '±10', '±0.5', '±0.5',
                            '±20', '±30', '±3']
            }
            
            df_props = pd.DataFrame(properties)
            st.dataframe(df_props, use_container_width=True)
            
            # Courbe I-V
            st.write("### 📈 Courbe I-V (Courant-Voltage)")
            
            V_range = np.linspace(-100, 50, 100)
            
            # Conductances voltage-dépendantes (simplifié)
            g_Na = 120 * (1 / (1 + np.exp(-(V_range + 40)/10)))  # Sodium
            g_K = 36 * (1 / (1 + np.exp(-(V_range + 50)/20)))    # Potassium
            g_leak = 0.3  # Fuite
            
            I_total = g_Na * (V_range - 50) + g_K * (V_range + 77) + g_leak * (V_range + 54)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=V_range, y=I_total,
                mode='lines',
                line=dict(color='#FF6B9D', width=3),
                name='I total'
            ))
            
            fig.add_vline(x=-70, line_dash="dash", line_color="white",
                         annotation_text="Repos")
            fig.add_vline(x=-55, line_dash="dash", line_color="yellow",
                         annotation_text="Seuil")
            
            fig.update_layout(
                title="Courbe Courant-Voltage",
                xaxis_title="Voltage (mV)",
                yaxis_title="Courant (pA)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: SYNAPSES & CONNEXIONS ====================
elif page == "🔗 Synapses & Connexions":
    st.header("🔗 Synapses & Connexions Neuronales")
    
    st.info("""
    **Synapses**
    
    Jonctions spécialisées permettant communication entre neurones.
    
    **Types:**
    - Chimiques (majoritaires) : neurotransmetteurs
    - Électriques (gap junctions) : ions directs
    
    **Plasticité:** LTP/LTD (Long-Term Potentiation/Depression)
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Anatomie", "⚡ Transmission", "🧠 Plasticité", "📊 Analyse"])
    
    with tab1:
        st.subheader("🔬 Anatomie Synaptique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📋 Composants")
            
            components = {
                'Présynaptique': ['Vésicules', 'Canaux Ca²⁺', 'Protéines SNARE'],
                'Fente': ['Largeur ~20nm', 'Matrice extracellulaire'],
                'Postsynaptique': ['Récepteurs', 'Densité post-synaptique', 'Canaux ioniques']
            }
            
            for region, items in components.items():
                st.write(f"**{region}:**")
                for item in items:
                    st.write(f"  • {item}")
        
        with col2:
            st.write("### 🔢 Statistiques")
            
            if st.session_state.organoid_lab['organoids']:
                total_neurons = sum(o['neuron_count'] 
                    for o in st.session_state.organoid_lab['organoids'].values())
                
                total_synapses = total_neurons * BIO_CONSTANTS['synapse_density']
                
                st.metric("Synapses Totales", f"{total_synapses/1e9:.2f} milliards")
                st.metric("Par Neurone", f"{BIO_CONSTANTS['synapse_density']:,}")
                st.metric("Densité", f"{total_synapses/(total_neurons*20e-6):.0f}/μm²")
            else:
                st.info("Créez un organoïde")
    
    with tab2:
        st.subheader("⚡ Transmission Synaptique")
        
        st.write("### 🔄 Processus")
        
        steps = [
            "1️⃣ Potentiel action arrive au terminal",
            "2️⃣ Canaux Ca²⁺ voltage-dépendants s'ouvrent",
            "3️⃣ Influx Ca²⁺ déclenche exocytose vésicules",
            "4️⃣ Neurotransmetteurs relâchés dans fente",
            "5️⃣ Liaison aux récepteurs postsynaptiques",
            "6️⃣ Ouverture canaux ioniques",
            "7️⃣ Potentiel postsynaptique (EPSP/IPSP)",
            "8️⃣ Recapture/dégradation neurotransmetteurs"
        ]
        
        for step in steps:
            st.write(step)
        
        st.write("### ⏱️ Chronologie")
        
        time_ms = np.linspace(0, 10, 1000)
        
        # Simulation EPSP
        tau_rise = 0.5
        tau_decay = 3
        epsp = (np.exp(-time_ms/tau_decay) - np.exp(-time_ms/tau_rise))
        epsp = epsp / epsp.max() * 5  # Normaliser à ~5 mV
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_ms, y=epsp,
            mode='lines',
            line=dict(color='#FF6B9D', width=3),
            fill='tozeroy',
            name='EPSP'
        ))
        
        fig.add_vline(x=BIO_CONSTANTS['synaptic_delay_ms'], 
                     line_dash="dash", line_color="yellow",
                     annotation_text="Délai synaptique")
        
        fig.update_layout(
            title="Potentiel Postsynaptique Excitateur (EPSP)",
            xaxis_title="Temps (ms)",
            yaxis_title="Amplitude (mV)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Délai Synaptique", f"{BIO_CONSTANTS['synaptic_delay_ms']} ms")
            st.metric("Amplitude EPSP", "~5 mV")
        
        with col2:
            st.metric("Temps montée", f"{tau_rise} ms")
            st.metric("Temps décroissance", f"{tau_decay} ms")
    
    with tab3:
        st.subheader("🧠 Plasticité Synaptique")
        
        st.write("""
        **STDP (Spike-Timing Dependent Plasticity)**
        
        La force synaptique dépend du timing relatif des spikes pré/post:
        - Pré avant Post (Δt > 0) → **LTP** (renforcement)
        - Post avant Pré (Δt < 0) → **LTD** (affaiblissement)
        """)
        
        # Courbe STDP
        delta_t = np.linspace(-50, 50, 200)
        
        # Fenêtre STDP
        tau_plus = 20
        tau_minus = 20
        A_plus = 0.1
        A_minus = -0.1
        
        stdp = np.where(delta_t > 0,
                       A_plus * np.exp(-delta_t / tau_plus),
                       A_minus * np.exp(delta_t / tau_minus))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=delta_t, y=stdp,
            mode='lines',
            line=dict(color='#FF6B9D', width=3),
            fill='tozeroy'
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        fig.add_vline(x=0, line_dash="dash", line_color="yellow")
        
        # Annotations
        fig.add_annotation(x=20, y=0.05, text="LTP<br>(renforcement)",
                          showarrow=True, arrowhead=2)
        fig.add_annotation(x=-20, y=-0.05, text="LTD<br>(affaiblissement)",
                          showarrow=True, arrowhead=2)
        
        fig.update_layout(
            title="Courbe STDP (Spike-Timing Dependent Plasticity)",
            xaxis_title="Δt = t_pré - t_post (ms)",
            yaxis_title="Δw (changement force synaptique)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Simulation interactive
        st.write("### 🎯 Simulateur STDP")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pre_spike_time = st.slider("Temps spike pré (ms)", 0.0, 100.0, 40.0, 1.0)
        
        with col2:
            post_spike_time = st.slider("Temps spike post (ms)", 0.0, 100.0, 45.0, 1.0)
        
        delta = pre_spike_time - post_spike_time
        strength_change = calculate_synaptic_strength(pre_spike_time, post_spike_time)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Δt", f"{delta:.1f} ms")
        with col2:
            st.metric("Δw", f"{strength_change:.4f}")
        with col3:
            if strength_change > 0:
                st.success("✅ LTP (renforcement)")
            elif strength_change < 0:
                st.error("❌ LTD (affaiblissement)")
            else:
                st.info("➖ Pas de changement")
    
    with tab4:
        st.subheader("📊 Analyse Réseau Synaptique")
        
        if st.session_state.organoid_lab['organoids']:
            selected_organoid = st.selectbox("Organoïde",
                list(st.session_state.organoid_lab['organoids'].keys()),
                format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                key="syn_org")
            
            organoid = st.session_state.organoid_lab['organoids'][selected_organoid]
            
            n_sample = st.slider("Neurones à analyser", 10, 500, 100)
            connectivity = st.slider("Probabilité connexion", 0.01, 0.5, 0.1, 0.01)
            
            if st.button("🔬 Analyser Connectivité", type="primary"):
                with st.spinner("Analyse en cours..."):
                    import time
                    time.sleep(1)
                    
                    # Générer matrice connectivité
                    conn_matrix = (np.random.random((n_sample, n_sample)) < connectivity).astype(int)
                    np.fill_diagonal(conn_matrix, 0)
                    
                    # Heatmap connectivité
                    fig = go.Figure(data=go.Heatmap(
                        z=conn_matrix,
                        colorscale=[[0, '#1a1a2e'], [1, '#FF6B9D']],
                        showscale=False
                    ))
                    
                    fig.update_layout(
                        title=f"Matrice de Connectivité ({n_sample}x{n_sample})",
                        xaxis_title="Neurone Post",
                        yaxis_title="Neurone Pré",
                        template="plotly_dark",
                        height=500,
                        width=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    total_connections = np.sum(conn_matrix)
                    possible_connections = n_sample * (n_sample - 1)
                    actual_connectivity = total_connections / possible_connections
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Connexions", f"{total_connections:,}")
                    with col2:
                        st.metric("Connectivité", f"{actual_connectivity:.1%}")
                    with col3:
                        avg_in = np.mean(np.sum(conn_matrix, axis=0))
                        st.metric("Entrées moy", f"{avg_in:.1f}")
                    with col4:
                        avg_out = np.mean(np.sum(conn_matrix, axis=1))
                        st.metric("Sorties moy", f"{avg_out:.1f}")

        else:
            st.info("Créez un organoïde")
    
    with tab2:
        st.subheader("🌊 Oscillations Neuronales")
        
        st.write("""
        **Rythmes Cérébraux**
        
        Oscillations synchrones reflétant activité coordonnée.
        """)
        
        # Définir bandes fréquence
        freq_bands = {
            'Delta (δ)': {'range': '0.5-4 Hz', 'state': 'Sommeil profond', 'color': '#355C7D'},
            'Theta (θ)': {'range': '4-8 Hz', 'state': 'Sommeil léger, méditation', 'color': '#6C5B7B'},
            'Alpha (α)': {'range': '8-13 Hz', 'state': 'Repos éveillé, yeux fermés', 'color': '#C06C84'},
            'Beta (β)': {'range': '13-30 Hz', 'state': 'Éveil actif, concentration', 'color': '#FF6B9D'},
            'Gamma (γ)': {'range': '30-100 Hz', 'state': 'Attention, conscience', 'color': '#FF1493'}
        }
        
        for band, info in freq_bands.items():
            with st.expander(f"🌊 {band}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Fréquence:** {info['range']}")
                    st.write(f"**État:** {info['state']}")
                
                with col2:
                    # Générer signal
                    t = np.linspace(0, 2, 1000)
                    freq = float(info['range'].split('-')[0])
                    signal = np.sin(2 * np.pi * freq * t)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=t, y=signal,
                        mode='lines',
                        line=dict(color=info['color'], width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Oscillation {band}",
                        xaxis_title="Temps (s)",
                        yaxis_title="Amplitude",
                        template="plotly_dark",
                        height=200,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 🎵 Simuler Oscillations Complexes")
        
        if st.button("🌊 Générer Signal Multi-Bande", type="primary"):
            t = np.linspace(0, 5, 5000)
            
            # Combiner plusieurs bandes
            signal = (0.5 * np.sin(2*np.pi*2*t) +     # Delta
                     0.3 * np.sin(2*np.pi*6*t) +     # Theta
                     0.4 * np.sin(2*np.pi*10*t) +    # Alpha
                     0.2 * np.sin(2*np.pi*20*t) +    # Beta
                     0.1 * np.sin(2*np.pi*40*t))     # Gamma
            
            # Ajouter bruit
            signal += np.random.normal(0, 0.1, len(t))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=t, y=signal,
                mode='lines',
                line=dict(color='#FF6B9D', width=1),
                name='Signal EEG'
            ))
            
            fig.update_layout(
                title="Signal Multi-Bande (EEG Simulé)",
                xaxis_title="Temps (s)",
                yaxis_title="Amplitude (μV)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse spectrale (FFT)
            from scipy import signal as sp_signal
            
            freqs, psd = sp_signal.welch(signal, fs=1000, nperseg=1024)
            
            fig_psd = go.Figure()
            
            fig_psd.add_trace(go.Scatter(
                x=freqs, y=psd,
                mode='lines',
                line=dict(color='#C06C84', width=2),
                fill='tozeroy'
            ))
            
            # Marquer bandes
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
            colors = ['#355C7D', '#6C5B7B', '#C06C84', '#FF6B9D', '#FF1493']
            
            for (low, high), color in zip(bands, colors):
                fig_psd.add_vrect(x0=low, x1=high, fillcolor=color, opacity=0.2, line_width=0)
            
            fig_psd.update_layout(
                title="Densité Spectrale de Puissance",
                xaxis_title="Fréquence (Hz)",
                yaxis_title="PSD",
                xaxis_range=[0, 50],
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_psd, use_container_width=True)
    
    with tab3:
        st.subheader("🔥 Propagation d'Activité")
        
        st.write("""
        **Ondes d'Activité**
        
        Propagation spatiale de l'activité neuronale à travers le réseau.
        """)
        
        n_neurons_line = st.slider("Neurones (ligne)", 20, 100, 50, key="prop_neurons")
        propagation_speed = st.slider("Vitesse propagation (neurones/ms)", 1, 20, 5)
        
        if st.button("🔥 Simuler Propagation", type="primary"):
            duration_ms = 100
            time_steps = 1000
            
            # Matrice activité
            activity_matrix = np.zeros((n_neurons_line, time_steps))
            
            # Initier activité au centre
            start_neuron = n_neurons_line // 2
            activity_matrix[start_neuron, 0] = 1
            
            # Propager
            for t in range(1, time_steps):
                for n in range(n_neurons_line):
                    if activity_matrix[n, t-1] > 0.1:
                        # Propager aux voisins
                        if n > 0:
                            activity_matrix[n-1, t] = max(activity_matrix[n-1, t], 
                                                         activity_matrix[n, t-1] * 0.9)
                        if n < n_neurons_line - 1:
                            activity_matrix[n+1, t] = max(activity_matrix[n+1, t], 
                                                         activity_matrix[n, t-1] * 0.9)
                
                # Décroissance
                activity_matrix[:, t] *= 0.95
            
            # Visualiser
            fig = go.Figure(data=go.Heatmap(
                z=activity_matrix,
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Activité")
            ))
            
            fig.update_layout(
                title="Propagation d'Onde d'Activité",
                xaxis_title="Temps (ms)",
                yaxis_title="Neurone #",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Onde propagée à travers le réseau!")
    
    with tab4:
        st.subheader("🧠 Patterns d'Activité")
        
        st.write("""
        **Détection de Patterns**
        
        Identification de motifs récurrents dans l'activité neuronale.
        """)
        
        pattern_type = st.selectbox("Type Pattern",
            ["Synchronisation", "Avalanches", "Up/Down States", "Bursts"])
        
        if pattern_type == "Synchronisation":
            st.write("""
            **Synchronisation Neuronale**
            
            Coordination temporelle de l'activité de populations neuronales.
            
            Mesure: Coefficient de corrélation entre neurones
            """)
            
            if st.button("📊 Analyser Synchronisation", type="primary"):
                n_neurons = 10
                duration_s = 2
                
                # Générer activité avec synchronisation variable
                t = np.linspace(0, duration_s, 2000)
                base_frequency = 10  # Hz
                
                signals = []
                for i in range(n_neurons):
                    # Ajouter variabilité phase
                    phase = np.random.uniform(0, 0.5)
                    signal = np.sin(2*np.pi*base_frequency*t + phase)
                    signal += np.random.normal(0, 0.3, len(t))
                    signals.append(signal)
                
                signals = np.array(signals)
                
                # Calculer matrice corrélation
                corr_matrix = np.corrcoef(signals)
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Corrélation")
                ))
                
                fig.update_layout(
                    title="Matrice de Synchronisation",
                    xaxis_title="Neurone",
                    yaxis_title="Neurone",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                avg_sync = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Synchronisation Moyenne", f"{avg_sync:.3f}")
                
                with col2:
                    if avg_sync > 0.7:
                        st.success("✅ Haute synchronisation")
                    elif avg_sync > 0.4:
                        st.info("➖ Synchronisation modérée")
                    else:
                        st.warning("⚠️ Faible synchronisation")
        
        elif pattern_type == "Avalanches":
            st.write("""
            **Avalanches Neuronales**
            
            Cascades d'activité se propageant à travers le réseau.
            
            Distribution taille avalanches suit loi puissance (criticalité).
            """)
            
            if st.button("⚡ Détecter Avalanches", type="primary"):
                # Simuler tailles avalanches (loi puissance)
                alpha = 1.5
                avalanche_sizes = np.random.pareto(alpha, 1000) + 1
                
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=avalanche_sizes,
                    nbinsx=50,
                    marker_color='#FF6B9D',
                    name='Avalanches'
                ))
                
                fig.update_layout(
                    title="Distribution Taille Avalanches",
                    xaxis_title="Taille (nombre neurones)",
                    yaxis_title="Fréquence",
                    xaxis_type="log",
                    yaxis_type="log",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("📊 Distribution suit loi puissance → Réseau critique!")

# ==================== PAGE: ÉLECTROPHYSIOLOGIE ====================
elif page == "📊 Électrophysiologie":
    st.header("📊 Électrophysiologie & Enregistrements")
    
    st.info("""
    **Électrophysiologie**
    
    Étude des propriétés électriques des cellules et tissus biologiques.
    
    **Techniques:**
    - Patch-Clamp (cellule unique)
    - Multi-Electrode Array (MEA)
    - EEG (surface)
    - Calcium Imaging
    """)
    
    tab1, tab2, tab3 = st.tabs(["🔬 Patch-Clamp", "📡 MEA", "📊 Analyse"])
    
    with tab1:
        st.subheader("🔬 Patch-Clamp")
        
        st.write("""
        **Technique de référence**
        
        Enregistrement courants ioniques à travers membrane cellulaire.
        
        **Modes:**
        - Voltage-Clamp: contrôle voltage, mesure courant
        - Current-Clamp: contrôle courant, mesure voltage
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            recording_mode = st.selectbox("Mode",
                ["Current-Clamp", "Voltage-Clamp"])
            
            if recording_mode == "Current-Clamp":
                current_injection = st.slider("Injection courant (pA)", -100, 500, 0, 10)
            else:
                holding_voltage = st.slider("Voltage maintien (mV)", -100, 50, -70, 5)
        
        with col2:
            recording_duration = st.slider("Durée (ms)", 10, 1000, 100)
            sampling_rate = st.selectbox("Fréquence échantillonnage",
                ["10 kHz", "20 kHz", "50 kHz"], index=1)
        
        if st.button("🔬 Démarrer Enregistrement", type="primary"):
            with st.spinner("Enregistrement..."):
                import time
                time.sleep(1)
                
                t = np.linspace(0, recording_duration, int(recording_duration * 20))
                
                if recording_mode == "Current-Clamp":
                    # Simuler réponse voltage
                    V = np.ones_like(t) * -70
                    
                    if current_injection > 150:  # Au-dessus seuil
                        # Générer trains de PA
                        n_spikes = int(current_injection / 50)
                        for i in range(n_spikes):
                            spike_time = 20 + i * 30
                            spike_idx = np.where((t >= spike_time) & (t < spike_time + 2))[0]
                            if len(spike_idx) > 0:
                                V[spike_idx] = -70 + 110 * np.sin(np.pi * np.arange(len(spike_idx)) / len(spike_idx))
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=V,
                        mode='lines',
                        line=dict(color='#FF6B9D', width=2),
                        name='Voltage'
                    ))
                    
                    fig.update_layout(
                        title=f"Current-Clamp: {current_injection} pA",
                        xaxis_title="Temps (ms)",
                        yaxis_title="Voltage (mV)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    n_spikes = int(np.sum(V > 0) / 20)
                    st.metric("Spikes détectés", n_spikes)
                
                else:  # Voltage-Clamp
                    # Simuler courants
                    I_Na = np.zeros_like(t)
                    I_K = np.zeros_like(t)
                    
                    if holding_voltage > -55:
                        # Activation canaux Na+
                        I_Na = -100 * np.exp(-(t-10)**2/50)
                        # Activation canaux K+
                        I_K = 50 * (1 - np.exp(-(t-15)/20))
                    
                    I_total = I_Na + I_K
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=I_Na,
                        mode='lines',
                        line=dict(color='#FF6B9D', width=2),
                        name='I_Na'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=I_K,
                        mode='lines',
                        line=dict(color='#355C7D', width=2),
                        name='I_K'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=I_total,
                        mode='lines',
                        line=dict(color='white', width=2, dash='dash'),
                        name='I_total'
                    ))
                    
                    fig.update_layout(
                        title=f"Voltage-Clamp: {holding_voltage} mV",
                        xaxis_title="Temps (ms)",
                        yaxis_title="Courant (pA)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📡 Multi-Electrode Array (MEA)")
        
        st.write("""
        **MEA - Enregistrement Multi-Sites**
        
        Array d'électrodes permettant enregistrer simultanément dizaines/centaines de neurones.
        
        **Configuration:** Grille 8x8, 16x16, ou configurations personnalisées
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            array_config = st.selectbox("Configuration MEA",
                ["8x8 (64 électrodes)", "16x16 (256 électrodes)", "32x32 (1024 électrodes)"])
            
            n_electrodes = int(array_config.split('(')[1].split()[0])
        
        with col2:
            recording_time = st.slider("Durée enregistrement (s)", 1, 60, 10)
        
        if st.button("📡 Enregistrer MEA", type="primary"):
            with st.spinner("Enregistrement MEA..."):
                import time
                time.sleep(2)
                
                # Simuler activité sur grille
                grid_size = int(np.sqrt(n_electrodes))
                activity = np.random.poisson(5, (grid_size, grid_size))
                
                fig = go.Figure(data=go.Heatmap(
                    z=activity,
                    colorscale='Hot',
                    colorbar=dict(title="Spikes/s")
                ))
                
                fig.update_layout(
                    title=f"Activité MEA - {array_config}",
                    xaxis_title="Électrode X",
                    yaxis_title="Électrode Y",
                    template="plotly_dark",
                    height=500,
                    width=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                total_spikes = np.sum(activity) * recording_time
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Spikes", f"{total_spikes:,}")
                with col2:
                    active_electrodes = np.sum(activity > 0)
                    st.metric("Électrodes Actives", f"{active_electrodes}/{n_electrodes}")
                with col3:
                    avg_rate = total_spikes / (n_electrodes * recording_time)
                    st.metric("Taux Moyen", f"{avg_rate:.1f} Hz")
                
                # Sauvegarder
                recording = {
                    'type': 'MEA',
                    'n_electrodes': n_electrodes,
                    'duration_s': recording_time,
                    'total_spikes': int(total_spikes),
                    'timestamp': datetime.now().isoformat()
                }
                st.session_state.organoid_lab['electrophysiology'].append(recording)
                log_event("Enregistrement MEA effectué", "SUCCESS")
    
    with tab3:
        st.subheader("📊 Analyse Enregistrements")
        
        if st.session_state.organoid_lab['electrophysiology']:
            st.write(f"### 📋 {len(st.session_state.organoid_lab['electrophysiology'])} Enregistrements")
            
            for i, rec in enumerate(st.session_state.organoid_lab['electrophysiology'][::-1][:10]):
                with st.expander(f"📊 Enregistrement {len(st.session_state.organoid_lab['electrophysiology'])-i}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Type:** {rec['type']}")
                        st.write(f"**Durée:** {rec['duration_s']} s")
                    
                    with col2:
                        st.write(f"**Électrodes:** {rec['n_electrodes']}")
                        st.write(f"**Spikes:** {rec['total_spikes']:,}")
                    
                    with col3:
                        st.write(f"**Date:** {rec['timestamp'][:19]}")
        else:
            st.info("Aucun enregistrement disponible")

# ==================== PAGE: RÉSEAUX NEURONAUX ====================
elif page == "🌐 Réseaux Neuronaux":
    st.header("🌐 Réseaux Neuronaux & Dynamiques")
    
    st.info("""
    **Réseau Neuronal Biologique**
    
    Organisation complexe de neurones interconnectés.
    
    **Propriétés émergentes:**
    - Oscillations synchrones
    - Ondes propagation
    - Patterns activité
    - Mémoire distribuée
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Topologie", "🌊 Oscillations", "🔥 Propagation", "🧠 Patterns"])
    
    with tab1:
        st.subheader("🌐 Topologie Réseau")
        
        if st.session_state.organoid_lab['organoids']:
            selected_organoid = st.selectbox("Organoïde",
                list(st.session_state.organoid_lab['organoids'].keys()),
                format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                key="net_org")
            
            organoid = st.session_state.organoid_lab['organoids'][selected_organoid]
            
            col1, col2 = st.columns(2)
            
            with col1:
                topology_type = st.selectbox("Type Topologie",
                    ["Random (Erdős-Rényi)", "Small-World (Watts-Strogatz)",
                     "Scale-Free (Barabási-Albert)", "Modular"])
                
                n_neurons_viz = st.slider("Neurones (visualisation)", 20, 200, 50)
            
            with col2:
                if topology_type == "Random (Erdős-Rényi)":
                    prob_connection = st.slider("Probabilité connexion", 0.01, 0.5, 0.1)
                elif topology_type == "Small-World (Watts-Strogatz)":
                    k_neighbors = st.slider("Voisins initiaux", 2, 20, 4)
                    rewiring_prob = st.slider("Probabilité rewiring", 0.0, 1.0, 0.1)
            
            if st.button("🎨 Visualiser Réseau", type="primary"):
                with st.spinner("Génération réseau..."):
                    import time
                    time.sleep(1)
                    
                    # Générer positions neurones
                    angles = np.linspace(0, 2*np.pi, n_neurons_viz, endpoint=False)
                    x_pos = np.cos(angles)
                    y_pos = np.sin(angles)
                    
                    # Générer connexions
                    if topology_type == "Random (Erdős-Rényi)":
                        connections = np.random.random((n_neurons_viz, n_neurons_viz)) < prob_connection
                    elif topology_type == "Small-World (Watts-Strogatz)":
                        # Simplification: connexions aux k voisins
                        connections = np.zeros((n_neurons_viz, n_neurons_viz), dtype=bool)
                        for i in range(n_neurons_viz):
                            for j in range(1, k_neighbors//2 + 1):
                                connections[i, (i+j) % n_neurons_viz] = True
                                connections[i, (i-j) % n_neurons_viz] = True
                    
                    np.fill_diagonal(connections, False)
                    
                    fig = go.Figure()
                    
                    # Dessiner connexions
                    for i in range(n_neurons_viz):
                        for j in range(n_neurons_viz):
                            if connections[i, j]:
                                fig.add_trace(go.Scatter(
                                    x=[x_pos[i], x_pos[j]],
                                    y=[y_pos[i], y_pos[j]],
                                    mode='lines',
                                    line=dict(color='rgba(255,107,157,0.2)', width=0.5),
                                    showlegend=False,
                                    hoverinfo='none'
                                ))
                    
                    # Dessiner neurones
                    fig.add_trace(go.Scatter(
                        x=x_pos,
                        y=y_pos,
                        mode='markers',
                        marker=dict(size=10, color='#FF6B9D', line=dict(color='white', width=1)),
                        showlegend=False,
                        hovertemplate='Neurone %{pointNumber}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"Réseau Neuronal - {topology_type}",
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        template="plotly_dark",
                        height=600,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistiques
                    n_connections = np.sum(connections)
                    density = n_connections / (n_neurons_viz * (n_neurons_viz - 1))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Connexions", f"{n_connections:,}")
                    with col2:
                        st.metric("Densité", f"{density:.3f}")
                    with col3:
                        avg_degree = n_connections / n_neurons_viz
                        st.metric("Degré moyen", f"{avg_degree:.1f}")

        else:
            st.info("Créez un organoïde")

    # ==================== COMPLÉTER PAGE: RÉSEAUX NEURONAUX - OSCILLATIONS ====================
# À insérer dans with tab2: (Oscillations)

    with tab2:
        st.subheader("🌊 Oscillations Neuronales")
        
        st.write("""
        **Rythmes Cérébraux**
        
        Oscillations synchrones reflétant activité coordonnée.
        """)
        
        # Définir bandes fréquence
        freq_bands = {
            'Delta (δ)': {'range': '0.5-4 Hz', 'state': 'Sommeil profond', 'color': '#355C7D'},
            'Theta (θ)': {'range': '4-8 Hz', 'state': 'Sommeil léger, méditation', 'color': '#6C5B7B'},
            'Alpha (α)': {'range': '8-13 Hz', 'state': 'Repos éveillé, yeux fermés', 'color': '#C06C84'},
            'Beta (β)': {'range': '13-30 Hz', 'state': 'Éveil actif, concentration', 'color': '#FF6B9D'},
            'Gamma (γ)': {'range': '30-100 Hz', 'state': 'Attention, conscience', 'color': '#FF1493'}
        }
        
        for band, info in freq_bands.items():
            with st.expander(f"🌊 {band}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Fréquence:** {info['range']}")
                    st.write(f"**État:** {info['state']}")
                
                with col2:
                    # Générer signal
                    t = np.linspace(0, 2, 1000)
                    freq = float(info['range'].split('-')[0])
                    signal = np.sin(2 * np.pi * freq * t)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=t, y=signal,
                        mode='lines',
                        line=dict(color=info['color'], width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"Oscillation {band}",
                        xaxis_title="Temps (s)",
                        yaxis_title="Amplitude",
                        template="plotly_dark",
                        height=200,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 🎵 Simuler Oscillations Complexes")
        
        if st.button("🌊 Générer Signal Multi-Bande", type="primary"):
            t = np.linspace(0, 5, 5000)
            
            # Combiner plusieurs bandes
            signal = (0.5 * np.sin(2*np.pi*2*t) +     # Delta
                     0.3 * np.sin(2*np.pi*6*t) +     # Theta
                     0.4 * np.sin(2*np.pi*10*t) +    # Alpha
                     0.2 * np.sin(2*np.pi*20*t) +    # Beta
                     0.1 * np.sin(2*np.pi*40*t))     # Gamma
            
            # Ajouter bruit
            signal += np.random.normal(0, 0.1, len(t))
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=t, y=signal,
                mode='lines',
                line=dict(color='#FF6B9D', width=1),
                name='Signal EEG'
            ))
            
            fig.update_layout(
                title="Signal Multi-Bande (EEG Simulé)",
                xaxis_title="Temps (s)",
                yaxis_title="Amplitude (μV)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse spectrale (FFT)
            from scipy import signal as sp_signal
            
            freqs, psd = sp_signal.welch(signal, fs=1000, nperseg=1024)
            
            fig_psd = go.Figure()
            
            fig_psd.add_trace(go.Scatter(
                x=freqs, y=psd,
                mode='lines',
                line=dict(color='#C06C84', width=2),
                fill='tozeroy'
            ))
            
            # Marquer bandes
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
            colors = ['#355C7D', '#6C5B7B', '#C06C84', '#FF6B9D', '#FF1493']
            
            for (low, high), color in zip(bands, colors):
                fig_psd.add_vrect(x0=low, x1=high, fillcolor=color, opacity=0.2, line_width=0)
            
            fig_psd.update_layout(
                title="Densité Spectrale de Puissance",
                xaxis_title="Fréquence (Hz)",
                yaxis_title="PSD",
                xaxis_range=[0, 50],
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_psd, use_container_width=True)
            
            st.success("✅ Analyse spectrale complétée!")
            
            # Cohérence entre signaux
            st.write("### 🔗 Cohérence entre Régions")
            
            if st.checkbox("Calculer cohérence"):
                # Simuler 2 signaux avec cohérence
                signal2 = 0.7 * signal + 0.3 * np.random.normal(0, 0.1, len(signal))
                
                freqs_coh, coherence = sp_signal.coherence(signal, signal2, fs=1000, nperseg=1024)
                
                fig_coh = go.Figure()
                
                fig_coh.add_trace(go.Scatter(
                    x=freqs_coh, y=coherence,
                    mode='lines',
                    line=dict(color='#FF6B9D', width=3),
                    fill='tozeroy'
                ))
                
                fig_coh.add_hline(y=0.5, line_dash="dash", line_color="white",
                                 annotation_text="Seuil significatif")
                
                fig_coh.update_layout(
                    title="Cohérence entre Deux Régions",
                    xaxis_title="Fréquence (Hz)",
                    yaxis_title="Cohérence",
                    xaxis_range=[0, 50],
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig_coh, use_container_width=True)
                
                # Bandes avec haute cohérence
                high_coherence_bands = []
                for (low, high), name in zip(bands, ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
                    band_coherence = np.mean(coherence[(freqs_coh >= low) & (freqs_coh <= high)])
                    if band_coherence > 0.5:
                        high_coherence_bands.append(f"{name}: {band_coherence:.2f}")
                
                if high_coherence_bands:
                    st.success("🔗 Bandes avec haute cohérence:")
                    for band in high_coherence_bands:
                        st.write(f"  • {band}")

# ==================== COMPLÉTER PAGE: RÉSEAUX NEURONAUX - PROPAGATION ====================
# À insérer dans with tab3: (Propagation)

    with tab3:
        st.subheader("🔥 Propagation d'Activité")
        
        st.write("""
        **Ondes d'Activité**
        
        Propagation spatiale de l'activité neuronale à travers le réseau.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_neurons_line = st.slider("Neurones (ligne)", 20, 100, 50, key="prop_neurons")
            propagation_speed = st.slider("Vitesse propagation (neurones/ms)", 1, 20, 5)
        
        with col2:
            wave_type = st.selectbox("Type d'Onde",
                ["Travelling Wave", "Spiral Wave", "Plane Wave", "Circular Wave"])
            
            initiation_point = st.selectbox("Point Initiation",
                ["Centre", "Bord Gauche", "Bord Droit", "Multiple Points"])
        
        if st.button("🔥 Simuler Propagation", type="primary"):
            duration_ms = 100
            time_steps = 1000
            
            # Matrice activité
            activity_matrix = np.zeros((n_neurons_line, time_steps))
            
            # Initier activité selon point
            if initiation_point == "Centre":
                start_neuron = n_neurons_line // 2
                activity_matrix[start_neuron, 0] = 1
            elif initiation_point == "Bord Gauche":
                activity_matrix[0, 0] = 1
            elif initiation_point == "Bord Droit":
                activity_matrix[-1, 0] = 1
            else:  # Multiple Points
                for i in range(0, n_neurons_line, n_neurons_line//4):
                    activity_matrix[i, 0] = 1
            
            # Propager
            for t in range(1, time_steps):
                for n in range(n_neurons_line):
                    if activity_matrix[n, t-1] > 0.1:
                        # Propager aux voisins
                        if n > 0:
                            activity_matrix[n-1, t] = max(activity_matrix[n-1, t], 
                                                         activity_matrix[n, t-1] * 0.9)
                        if n < n_neurons_line - 1:
                            activity_matrix[n+1, t] = max(activity_matrix[n+1, t], 
                                                         activity_matrix[n, t-1] * 0.9)
                
                # Décroissance
                activity_matrix[:, t] *= 0.95
            
            # Visualiser
            fig = go.Figure(data=go.Heatmap(
                z=activity_matrix,
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Activité")
            ))
            
            fig.update_layout(
                title="Propagation d'Onde d'Activité",
                xaxis_title="Temps (ms)",
                yaxis_title="Neurone #",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Onde propagée à travers le réseau!")
            
            # Analyse propagation
            st.write("### 📊 Analyse Propagation")
            
            col1, col2, col3 = st.columns(3)
            
            # Vitesse propagation
            peak_times = []
            for n in range(n_neurons_line):
                peak_time = np.argmax(activity_matrix[n, :])
                if activity_matrix[n, peak_time] > 0.3:
                    peak_times.append((n, peak_time))
            
            if len(peak_times) > 1:
                speed = (peak_times[-1][0] - peak_times[0][0]) / (peak_times[-1][1] - peak_times[0][1]) * 1000
                
                with col1:
                    st.metric("Vitesse", f"{speed:.1f} neurones/s")
                
                with col2:
                    duration = peak_times[-1][1] - peak_times[0][1]
                    st.metric("Durée", f"{duration} ms")
                
                with col3:
                    neurons_activated = np.sum(np.max(activity_matrix, axis=1) > 0.3)
                    st.metric("Neurones Activés", f"{neurons_activated}/{n_neurons_line}")
            
            # Graphique vitesse propagation
            if len(peak_times) > 2:
                neurons_pos = [pt[0] for pt in peak_times]
                times = [pt[1] for pt in peak_times]
                
                fig_speed = go.Figure()
                
                fig_speed.add_trace(go.Scatter(
                    x=times, y=neurons_pos,
                    mode='markers+lines',
                    marker=dict(size=8, color='#FF6B9D'),
                    line=dict(color='#C06C84', width=2)
                ))
                
                fig_speed.update_layout(
                    title="Vitesse de Propagation",
                    xaxis_title="Temps (ms)",
                    yaxis_title="Position Neurone",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig_speed, use_container_width=True)
        
        # Propagation 2D
        st.write("### 🌊 Propagation 2D")
        
        if st.button("🌀 Simuler Propagation 2D", key="prop2d"):
            grid_size = 50
            time_steps_2d = 100
            
            # Grille activité
            grid = np.zeros((grid_size, grid_size, time_steps_2d))
            
            # Initiation au centre
            center = grid_size // 2
            grid[center, center, 0] = 1
            
            # Propager en cercles
            for t in range(1, time_steps_2d):
                for i in range(1, grid_size-1):
                    for j in range(1, grid_size-1):
                        if grid[i, j, t-1] > 0.1:
                            # Propager aux 4 voisins
                            grid[i-1, j, t] = max(grid[i-1, j, t], grid[i, j, t-1] * 0.85)
                            grid[i+1, j, t] = max(grid[i+1, j, t], grid[i, j, t-1] * 0.85)
                            grid[i, j-1, t] = max(grid[i, j-1, t], grid[i, j, t-1] * 0.85)
                            grid[i, j+1, t] = max(grid[i, j+1, t], grid[i, j, t-1] * 0.85)
                
                # Décroissance
                grid[:, :, t] *= 0.92
            
            # Animation frames
            frames = []
            for t in range(0, time_steps_2d, 5):
                frames.append(go.Frame(
                    data=[go.Heatmap(z=grid[:, :, t], colorscale='Hot')],
                    name=str(t)
                ))
            
            fig_2d = go.Figure(
                data=[go.Heatmap(z=grid[:, :, 0], colorscale='Hot', showscale=True)],
                frames=frames
            )
            
            fig_2d.update_layout(
                title="Propagation 2D (Animation)",
                xaxis_title="X",
                yaxis_title="Y",
                template="plotly_dark",
                height=500,
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {'label': '▶ Play', 'method': 'animate',
                         'args': [None, {'frame': {'duration': 50}}]},
                        {'label': '⏸ Pause', 'method': 'animate',
                         'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                    ]
                }]
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
            
            st.success("🌊 Propagation circulaire 2D simulée!")

# ==================== COMPLÉTER PAGE: RÉSEAUX NEURONAUX - PATTERNS ====================
# À insérer dans with tab4: (Patterns)

    with tab4:
        st.subheader("🧠 Patterns d'Activité")
        
        st.write("""
        **Détection de Patterns**
        
        Identification de motifs récurrents dans l'activité neuronale.
        """)
        
        pattern_type = st.selectbox("Type Pattern",
            ["Synchronisation", "Avalanches", "Up/Down States", "Bursts", "Replay"])
        
        if pattern_type == "Synchronisation":
            st.write("""
            **Synchronisation Neuronale**
            
            Coordination temporelle de l'activité de populations neuronales.
            
            Mesure: Coefficient de corrélation entre neurones
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_neurons_sync = st.slider("Nombre neurones", 5, 50, 10, key="sync_n")
                sync_strength = st.slider("Force synchronisation", 0.0, 1.0, 0.7, 0.1)
            
            with col2:
                duration_s = st.slider("Durée (s)", 1, 10, 2)
                base_freq = st.slider("Fréquence base (Hz)", 5, 50, 10)
            
            if st.button("📊 Analyser Synchronisation", type="primary"):
                # Générer activité avec synchronisation variable
                t = np.linspace(0, duration_s, duration_s * 1000)
                
                signals = []
                for i in range(n_neurons_sync):
                    # Phase aléatoire selon force sync
                    phase = np.random.uniform(0, (1-sync_strength) * np.pi)
                    signal = np.sin(2*np.pi*base_freq*t + phase)
                    signal += np.random.normal(0, 0.3, len(t))
                    signals.append(signal)
                
                signals = np.array(signals)
                
                # Calculer matrice corrélation
                corr_matrix = np.corrcoef(signals)
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Corrélation")
                ))
                
                fig.update_layout(
                    title="Matrice de Synchronisation",
                    xaxis_title="Neurone",
                    yaxis_title="Neurone",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                avg_sync = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Synchronisation Moyenne", f"{avg_sync:.3f}")
                
                with col2:
                    if avg_sync > 0.7:
                        st.success("✅ Haute synchronisation")
                    elif avg_sync > 0.4:
                        st.info("➖ Synchronisation modérée")
                    else:
                        st.warning("⚠️ Faible synchronisation")
                
                with col3:
                    pairs_synced = np.sum(corr_matrix[np.triu_indices_from(corr_matrix, k=1)] > 0.5)
                    total_pairs = (n_neurons_sync * (n_neurons_sync - 1)) // 2
                    st.metric("Paires Synchronisées", f"{pairs_synced}/{total_pairs}")
                
                # Traces temporelles
                st.write("### 📈 Traces Temporelles (5 neurones)")
                
                fig_traces = go.Figure()
                
                for i in range(min(5, n_neurons_sync)):
                    fig_traces.add_trace(go.Scatter(
                        x=t[:500], y=signals[i, :500] + i*3,
                        mode='lines',
                        name=f'Neurone {i+1}',
                        line=dict(width=1.5)
                    ))
                
                fig_traces.update_layout(
                    title="Activité Synchronisée",
                    xaxis_title="Temps (s)",
                    yaxis_title="Neurone",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig_traces, use_container_width=True)
        
        elif pattern_type == "Avalanches":
            st.write("""
            **Avalanches Neuronales**
            
            Cascades d'activité se propageant à travers le réseau.
            
            Distribution taille avalanches suit loi puissance (criticalité).
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_avalanches = st.slider("Nombre avalanches", 100, 10000, 1000)
                alpha_param = st.slider("Exposant α", 1.0, 3.0, 1.5, 0.1)
            
            with col2:
                threshold = st.slider("Seuil détection", 1, 10, 3)
            
            if st.button("⚡ Détecter Avalanches", type="primary"):
                # Simuler tailles avalanches (loi puissance)
                avalanche_sizes = np.random.pareto(alpha_param, n_avalanches) + 1
                
                # Histogramme
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=avalanche_sizes,
                    nbinsx=50,
                    marker_color='#FF6B9D',
                    name='Avalanches'
                ))
                
                fig.update_layout(
                    title="Distribution Taille Avalanches",
                    xaxis_title="Taille (nombre neurones)",
                    yaxis_title="Fréquence",
                    xaxis_type="log",
                    yaxis_type="log",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse
                large_avalanches = np.sum(avalanche_sizes > 10)
                max_size = np.max(avalanche_sizes)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Avalanches Totales", n_avalanches)
                with col2:
                    st.metric("Grandes (>10 neurones)", large_avalanches)
                with col3:
                    st.metric("Taille Max", f"{max_size:.0f}")
                
                st.info("📊 Distribution suit loi puissance → Réseau critique!")
                
                # Série temporelle avalanches
                st.write("### ⏱️ Série Temporelle")
                
                time = np.arange(min(100, n_avalanches))
                sizes_sample = avalanche_sizes[:len(time)]
                
                fig_time = go.Figure()
                
                fig_time.add_trace(go.Scatter(
                    x=time, y=sizes_sample,
                    mode='lines+markers',
                    marker=dict(size=6, color='#FF6B9D'),
                    line=dict(color='#C06C84', width=1)
                ))
                
                fig_time.add_hline(y=10, line_dash="dash", line_color="yellow",
                                  annotation_text="Seuil grande avalanche")
                
                fig_time.update_layout(
                    title="Avalanches dans le Temps",
                    xaxis_title="Temps",
                    yaxis_title="Taille Avalanche",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig_time, use_container_width=True)
        
        elif pattern_type == "Up/Down States":
            st.write("""
            **États Up/Down**
            
            Alternance entre états haute activité (Up) et basse activité (Down).
            
            Observé dans sommeil profond et anesthésie.
            """)
            
            if st.button("🔄 Simuler Up/Down States", type="primary"):
                duration_s = 30
                t = np.linspace(0, duration_s, duration_s * 100)
                
                # Générer états Up/Down
                state = np.zeros_like(t)
                current_state = 0  # 0=Down, 1=Up
                
                i = 0
                while i < len(t):
                    if current_state == 0:  # Down state
                        duration = int(np.random.uniform(50, 200))
                        state[i:min(i+duration, len(t))] = np.random.uniform(0, 2, min(duration, len(t)-i))
                        current_state = 1
                    else:  # Up state
                        duration = int(np.random.uniform(100, 400))
                        state[i:min(i+duration, len(t))] = np.random.uniform(8, 12, min(duration, len(t)-i))
                        current_state = 0
                    
                    i += duration
                
                # Ajouter bruit
                state += np.random.normal(0, 0.5, len(t))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=t, y=state,
                    mode='lines',
                    line=dict(color='#FF6B9D', width=2),
                    fill='tozeroy'
                ))
                
                fig.add_hline(y=5, line_dash="dash", line_color="white",
                             annotation_text="Seuil Up/Down")
                
                fig.update_layout(
                    title="États Up/Down States",
                    xaxis_title="Temps (s)",
                    yaxis_title="Activité Population",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Détecter transitions
                up_states = state > 5
                transitions_down_to_up = np.where(np.diff(up_states.astype(int)) == 1)[0]
                transitions_up_to_down = np.where(np.diff(up_states.astype(int)) == -1)[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Transitions Up→Down", len(transitions_up_to_down))
                with col2:
                    st.metric("Transitions Down→Up", len(transitions_down_to_up))
                with col3:
                    avg_up_duration = np.mean(np.diff(transitions_down_to_up)) / 100
                    st.metric("Durée Up Moyenne", f"{avg_up_duration:.1f}s")
        
        elif pattern_type == "Bursts":
            st.write("""
            **Bursts Neuronaux**
            
            Décharges rapides et intenses de potentiels d'action.
            
            Caractérisé par: fréquence élevée instantanée, durée brève.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_neurons_burst = st.slider("Nombre neurones", 5, 20, 10, key="burst_n")
                burst_freq = st.slider("Fréquence bursts (Hz)", 0.1, 5.0, 1.0, 0.1)
            
            with col2:
                spikes_per_burst = st.slider("Spikes par burst", 3, 20, 5)
                burst_duration_ms = st.slider("Durée burst (ms)", 10, 200, 50)
            
            if st.button("💥 Générer Bursts", type="primary"):
                duration_s = 10
                t = np.linspace(0, duration_s, duration_s * 1000)
                
                # Générer bursts
                burst_times = np.arange(0, duration_s, 1/burst_freq)
                
                # Raster plot
                fig = go.Figure()
                
                for neuron_idx in range(n_neurons_burst):
                    spike_times = []
                    
                    for burst_t in burst_times:
                        # Spikes dans burst
                        for spike in range(spikes_per_burst):
                            spike_time = burst_t + spike * (burst_duration_ms/1000/spikes_per_burst)
                            if spike_time < duration_s:
                                spike_times.append(spike_time)
                    
                    fig.add_trace(go.Scatter(
                        x=spike_times,
                        y=[neuron_idx] * len(spike_times),
                        mode='markers',
                        marker=dict(size=4, color='#FF6B9D', symbol='line-ns'),
                        showlegend=False,
                        hovertemplate=f'Neurone {neuron_idx}<br>Temps: %{{x:.3f}}s<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"Raster Plot - Bursts ({n_neurons_burst} neurones)",
                    xaxis_title="Temps (s)",
                    yaxis_title="Neurone #",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques bursts
                total_bursts = len(burst_times) * n_neurons_burst
                total_spikes = total_bursts * spikes_per_burst
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Bursts Totaux", total_bursts)
                with col2:
                    st.metric("Spikes Totaux", total_spikes)
                with col3:
                    avg_isi = burst_duration_ms / spikes_per_burst
                    st.metric("ISI Intra-burst", f"{avg_isi:.1f} ms")
                
                # Histogramme intervalles inter-spikes
                st.write("### 📊 Distribution ISI")
                
                # Simuler ISI
                isi_intra = np.random.normal(avg_isi, 5, total_spikes//2)
                isi_inter = np.random.normal(1000/burst_freq, 100, total_bursts)
                
                fig_isi = go.Figure()
                
                fig_isi.add_trace(go.Histogram(
                    x=isi_intra,
                    name='Intra-burst',
                    marker_color='#FF6B9D',
                    opacity=0.7
                ))
                
                fig_isi.add_trace(go.Histogram(
                    x=isi_inter,
                    name='Inter-burst',
                    marker_color='#355C7D',
                    opacity=0.7
                ))
                
                fig_isi.update_layout(
                    title="Distribution Intervalles Inter-Spikes",
                    xaxis_title="ISI (ms)",
                    yaxis_title="Fréquence",
                    barmode='overlay',
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig_isi, use_container_width=True)
        
        elif pattern_type == "Replay":
            st.write("""
            **Replay de Séquences**
            
            Réactivation spontanée de patterns d'activité précédemment expérimentés.
            
            Observé pendant sommeil et repos éveillé.
            """)
            
            if st.button("🔄 Simuler Replay", type="primary"):
                n_neurons = 20
                sequence_length = 10
                
                # Séquence originale
                original_sequence = np.random.permutation(n_neurons)[:sequence_length]
                
                # Replay avec bruit
                replay_sequence = original_sequence.copy()
                # Ajouter quelques erreurs
                n_errors = np.random.randint(0, 3)
                if n_errors > 0:
                    error_positions = np.random.choice(sequence_length, n_errors, replace=False)
                    replay_sequence[error_positions] = np.random.randint(0, n_neurons, n_errors)
                
                # Visualiser
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Séquence Originale:**")
                    
                    fig_orig = go.Figure()
                    
                    for i, neuron_id in enumerate(original_sequence):
                        fig_orig.add_trace(go.Scatter(
                            x=[i], y=[neuron_id],
                            mode='markers+text',
                            marker=dict(size=20, color='#FF6B9D'),
                            text=[f'{neuron_id}'],
                            textposition='middle center',
                            showlegend=False
                        ))
                    
                    fig_orig.update_layout(
                        title="Séquence Apprise",
                        xaxis_title="Position Temporelle",
                        yaxis_title="Neurone ID",
                        template="plotly_dark",
                        height=300
                    )
                    
                    st.plotly_chart(fig_orig, use_container_width=True)
                
                with col2:
                    st.write("**Replay:**")
                    
                    fig_replay = go.Figure()
                    
                    for i, neuron_id in enumerate(replay_sequence):
                        color = '#FF6B9D' if neuron_id == original_sequence[i] else '#FF1493'
                        
                        fig_replay.add_trace(go.Scatter(
                            x=[i], y=[neuron_id],
                            mode='markers+text',
                            marker=dict(size=20, color=color),
                            text=[f'{neuron_id}'],
                            textposition='middle center',
                            showlegend=False
                        ))
                    
                    fig_replay.update_layout(
                        title="Replay Spontané",
                        xaxis_title="Position Temporelle",
                        yaxis_title="Neurone ID",
                        template="plotly_dark",
                        height=300
                    )
                    
                    st.plotly_chart(fig_replay, use_container_width=True)
                
                # Similarité
                similarity = np.sum(original_sequence == replay_sequence) / sequence_length
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Similarité", f"{similarity*100:.0f}%")
                with col2:
                    st.metric("Erreurs", n_errors)
                with col3:
                    if similarity > 0.8:
                        st.success("✅ Replay fidèle")
                    elif similarity > 0.5:
                        st.info("➖ Replay partiel")
                    else:
                        st.warning("⚠️ Replay dégradé")

# ==================== AMÉLIORER PAGE: EXPÉRIENCES ====================
# Remplacer la page Expériences existante par cette version améliorée

elif page == "🔬 Expériences":
    st.header("🔬 Expériences & Protocoles Avancés")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Designer", "🔬 Bibliothèque", "📊 Historique", "🤖 IA Assistant", "📈 Analyse"])
    
    with tab1:
        st.subheader("📋 Designer d'Expérience")
        
        with st.form("design_experiment"):
            col1, col2 = st.columns(2)
            
            with col1:
                exp_name = st.text_input("Nom Expérience", "EXP-001")
                
                exp_type = st.selectbox("Type",
                    ["Électrophysiologie", "Pharmacologie", "Stimulation",
                     "Apprentissage", "Imagerie", "Biocomputing", "Plasticité Synaptique",
                     "Connectivité", "Métabolisme", "Stress Test"])
                
                duration_min = st.number_input("Durée (min)", 1, 480, 60)
                n_trials = st.number_input("Essais", 1, 1000, 10)
            
            with col2:
                if st.session_state.organoid_lab['organoids']:
                    organoid_id = st.selectbox("Organoïde",
                        list(st.session_state.organoid_lab['organoids'].keys()),
                        format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'])
                else:
                    st.warning("Créez un organoïde")
                    organoid_id = None
                
                priority = st.select_slider("Priorité",
                    options=["Basse", "Normale", "Haute", "Urgente"])
                
                automated = st.checkbox("Automatisation", value=True)
            
            hypothesis = st.text_area("Hypothèse",
                "L'organoïde montrera une augmentation d'activité après stimulation répétée")
            
            # Protocole structuré
            st.write("### 📝 Protocole (étapes)")
            
            protocol_steps = []
            n_steps = st.number_input("Nombre d'étapes", 1, 10, 3, key="n_steps")
            
            for i in range(n_steps):
                with st.expander(f"Étape {i+1}"):
                    step_col1, step_col2 = st.columns(2)
                    
                    with step_col1:
                        step_name = st.text_input(f"Nom", f"Étape {i+1}", key=f"step_name_{i}")
                        step_duration = st.number_input(f"Durée (min)", 1, 120, 10, key=f"step_dur_{i}")
                    
                    with step_col2:
                        step_action = st.selectbox(f"Action",
                            ["Baseline", "Stimulation", "Recording", "Application", "Lavage", "Mesure"],
                            key=f"step_action_{i}")
                        step_params = st.text_input(f"Paramètres", "", key=f"step_params_{i}")
                    
                    protocol_steps.append({
                        'name': step_name,
                        'duration': step_duration,
                        'action': step_action,
                        'params': step_params
                    })
            
            # Contrôles et mesures
            st.write("### 🎯 Contrôles & Mesures")
            
            controls_col1, controls_col2 = st.columns(2)
            
            with controls_col1:
                control_group = st.checkbox("Groupe contrôle")
                blind = st.checkbox("Aveugle")
                randomized = st.checkbox("Randomisé")
            
            with controls_col2:
                measures = st.multiselect("Mesures",
                    ["Électrophysiologie", "Imagerie", "Viabilité", "Métabolisme",
                     "Expression génétique", "Morphologie"])
            
            if st.form_submit_button("🚀 Créer & Lancer Expérience", type="primary"):
                if organoid_id:
                    experiment = {
                        'name': exp_name,
                        'type': exp_type,
                        'organoid_id': organoid_id,
                        'duration_min': duration_min,
                        'n_trials': n_trials,
                        'priority': priority,
                        'automated': automated,
                        'hypothesis': hypothesis,
                        'protocol_steps': protocol_steps,
                        'controls': {
                            'control_group': control_group,
                            'blind': blind,
                            'randomized': randomized
                        },
                        'measures': measures,
                        'status': 'running' if automated else 'planned',
                        'timestamp': datetime.now().isoformat(),
                        'progress': 0
                    }
                    
                    st.session_state.organoid_lab['experiments'].append(experiment)
                    log_event(f"Expérience créée: {exp_name}", "SUCCESS")
                    
                    st.success(f"✅ Expérience '{exp_name}' créée!")
                    
                    if automated:
                        with st.spinner("Lancement automatique..."):
                            import time
                            progress_bar = st.progress(0)
                            
                            for i in range(100):
                                time.sleep(0.02)
                                progress_bar.progress(i + 1)
                            
                            experiment['status'] = 'completed'
                            experiment['progress'] = 100
                            
                            st.balloons()
                            st.success("🎉 Expérience terminée!")
                    
                    st.rerun()
                else:
                    st.error("Sélectionnez un organoïde")
    
    with tab2:
        st.subheader("🔬 Bibliothèque Protocoles")
        
        st.write("### 📚 Protocoles Standards")
        
        standard_protocols = {
            "Test LTP (Long-Term Potentiation)": {
                "description": "Induire potentialisation à long terme par stimulation haute fréquence",
                "duration": "120 min",
                "steps": ["Baseline 20min", "HFS 100Hz 1s", "Recording 90min"],
                "difficulty": "Intermédiaire"
            },
            "Pharmacologie: Dose-Réponse": {
                "description": "Tester effet dose-dépendant d'un composé",
                "duration": "180 min",
                "steps": ["Baseline", "Doses croissantes", "Lavage", "Récupération"],
                "difficulty": "Facile"
            },
            "Apprentissage Pavlovien": {
                "description": "Conditionnement classique avec organoïde",
                "duration": "240 min",
                "steps": ["Habituation", "CS+US pairings", "Test", "Extinction"],
                "difficulty": "Avancé"
            },
            "Privation Oxygène": {
                "description": "Étudier résistance à l'hypoxie",
                "duration": "90 min",
                "steps": ["Baseline", "Hypoxie graduelle", "Réoxygénation", "Récupération"],
                "difficulty": "Avancé"
            },
            "Calcium Imaging Time-Lapse": {
                "description": "Imagerie calcium sur période prolongée",
                "duration": "360 min",
                "steps": ["Setup", "Acquisition continue", "Analyse temps réel"],
                "difficulty": "Intermédiaire"
            }
        }
        
        for protocol_name, info in standard_protocols.items():
            with st.expander(f"📋 {protocol_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Durée:** {info['duration']}")
                    st.write(f"**Difficulté:** {info['difficulty']}")
                
                with col2:
                    st.write("**Étapes:**")
                    for step in info['steps']:
                        st.write(f"  • {step}")
                    
                    if st.button(f"📥 Charger Protocole", key=f"load_{protocol_name}"):
                        st.success(f"✅ Protocole '{protocol_name}' chargé!")
                        st.info("Retournez à l'onglet Designer pour modifier et lancer")
    
    with tab3:
        st.subheader("📊 Historique Expériences")
        
        if st.session_state.organoid_lab['experiments']:
            # Filtres
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.multiselect("Type",
                    list(set([exp['type'] for exp in st.session_state.organoid_lab['experiments']])))
            
            with col2:
                filter_status = st.multiselect("Status",
                    ["planned", "running", "completed", "failed"])
            
            with col3:
                sort_by = st.selectbox("Trier par",
                    ["Date (récent)", "Date (ancien)", "Durée", "Priorité"])
            
            # Afficher expériences
            for i, exp in enumerate(st.session_state.organoid_lab['experiments'][::-1]):
                # Appliquer filtres
                if filter_type and exp['type'] not in filter_type:
                    continue
                if filter_status and exp.get('status', 'planned') not in filter_status:
                    continue
                
                with st.expander(f"🔬 {exp['name']} - {exp['type']} ({exp.get('status', 'planned').upper()})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📋 Info")
                        st.write(f"**Type:** {exp['type']}")
                        st.write(f"**Durée:** {exp['duration_min']} min")
                        st.write(f"**Essais:** {exp['n_trials']}")
                        
                        status = exp.get('status', 'planned')
                        if status == 'completed':
                            st.success("✅ Complété")
                        elif status == 'running':
                            st.info("🔄 En cours")
                            progress = exp.get('progress', 0)
                            st.progress(progress / 100)
                        elif status == 'failed':
                            st.error("❌ Échoué")
                        else:
                            st.warning("⏳ Planifié")
                    
                    with col2:
                        st.write("### 🎯 Hypothèse")
                        st.write(exp['hypothesis'])
                        
                        if 'measures' in exp and exp['measures']:
                            st.write("**Mesures:**")
                            for measure in exp['measures']:
                                st.write(f"  • {measure}")
                    
                    with col3:
                        st.write("### 📅 Dates")
                        st.write(f"**Créé:** {exp['timestamp'][:19]}")
                        
                        if 'executed_at' in exp and exp['executed_at']:
                            st.write(f"**Exécuté:** {exp['executed_at'][:19]}")
                        
                        st.write(f"**Priorité:** {exp.get('priority', 'Normale')}")
                    
                    # Protocole
                    if 'protocol_steps' in exp and exp['protocol_steps']:
                        st.write("### 📝 Protocole")
                        
                        for i, step in enumerate(exp['protocol_steps']):
                            st.write(f"**{i+1}. {step['name']}** ({step['duration']} min) - {step['action']}")
                            if step['params']:
                                st.write(f"   Params: {step['params']}")
                    
                    # Actions
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if exp.get('status') == 'planned':
                            if st.button("▶️ Lancer", key=f"start_{i}"):
                                exp['status'] = 'running'
                                st.rerun()
                    
                    with col2:
                        if st.button("📊 Analyser", key=f"analyze_{i}"):
                            st.info("Ouverture analyse...")
                    
                    with col3:
                        if st.button("📥 Exporter", key=f"export_{i}"):
                            st.json(exp)
        else:
            st.info("Aucune expérience enregistrée")
    
    with tab4:
        st.subheader("🤖 Assistant IA Expériences")
        
        st.write("""
        **Assistant Intelligent**
        
        L'IA vous aide à concevoir, optimiser et analyser vos expériences.
        """)
        
        ai_task = st.selectbox("Que voulez-vous faire?",
            ["Suggérer expérience", "Optimiser protocole", "Prédire résultats",
             "Analyser données", "Détecter anomalies", "Recommander contrôles"])
        
        if ai_task == "Suggérer expérience":
            st.write("### 💡 Suggestions Basées sur Votre Lab")
            
            if st.button("🤖 Générer Suggestions", type="primary"):
                suggestions = [
                    {
                        "title": "Test Plasticité Synaptique",
                        "rationale": "Vos organoïdes sont matures (>90 jours), idéal pour LTP/LTD",
                        "confidence": 0.85,
                        "estimated_duration": "180 min"
                    },
                    {
                        "title": "Screening Pharmacologique",
                        "rationale": "Haute viabilité (>95%), parfait pour tester composés",
                        "confidence": 0.78,
                        "estimated_duration": "240 min"
                    },
                    {
                        "title": "Cartographie Connectivité",
                        "rationale": "Nombre élevé de neurones, réseau complexe à explorer",
                        "confidence": 0.72,
                        "estimated_duration": "360 min"
                    }
                ]
                
                for i, sug in enumerate(suggestions):
                    with st.container():
                        st.markdown(f"### {i+1}. {sug['title']}")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Rationale:** {sug['rationale']}")
                            st.write(f"**Durée estimée:** {sug['estimated_duration']}")
                        
                        with col2:
                            st.metric("Confiance IA", f"{sug['confidence']*100:.0f}%")
                            
                            if st.button("➕ Créer", key=f"create_sug_{i}"):
                                st.success(f"✅ Expérience '{sug['title']}' ajoutée au designer!")
                        
                        st.markdown("---")
        
        elif ai_task == "Prédire résultats":
            st.write("### 🔮 Prédiction Résultats")
            
            if st.session_state.organoid_lab['organoids']:
                selected_organoid = st.selectbox("Organoïde",
                    list(st.session_state.organoid_lab['organoids'].keys()),
                    format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                    key="pred_org")
                
                exp_to_predict = st.selectbox("Type expérience",
                    ["Stimulation électrique", "Application Glutamate", "LTP", "Apprentissage"])
                
                if st.button("🔮 Prédire", type="primary"):
                    with st.spinner("IA calcule prédictions..."):
                        import time
                        time.sleep(2)
                        
                        # Prédictions simulées
                        predictions = {
                            "success_probability": np.random.uniform(0.7, 0.95),
                            "expected_response": f"+{np.random.uniform(20, 60):.0f}%",
                            "confidence_interval": "(±15%)",
                            "optimal_parameters": {
                                "amplitude": f"{np.random.uniform(50, 150):.0f} μA",
                                "duration": f"{np.random.uniform(100, 500):.0f} ms",
                                "frequency": f"{np.random.uniform(10, 50):.0f} Hz"
                            }
                        }
                        
                        st.success("✅ Prédictions générées!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Probabilité Succès", 
                                     f"{predictions['success_probability']*100:.0f}%")
                            st.metric("Réponse Attendue", 
                                     predictions['expected_response'])
                        
                        with col2:
                            st.write("**Paramètres Optimaux:**")
                            for param, value in predictions['optimal_parameters'].items():
                                st.write(f"• {param}: {value}")
                        
                        st.info(f"💡 Intervalle confiance: {predictions['confidence_interval']}")
            else:
                st.info("Créez un organoïde pour prédire")
        
        elif ai_task == "Détecter anomalies":
            st.write("### 🔍 Détection Anomalies")
            
            if st.button("🔍 Scanner Lab", type="primary"):
                anomalies = [
                    {
                        "type": "warning",
                        "message": "Viabilité organoïde ORG-002 en baisse (87%)",
                        "recommendation": "Vérifier milieu culture, considérer changement"
                    },
                    {
                        "type": "info",
                        "message": "Activité neuronale inhabituelle détectée (burst rate +40%)",
                        "recommendation": "Possiblement normal, surveiller 24h"
                    }
                ]
                
                if anomalies:
                    st.warning(f"⚠️ {len(anomalies)} anomalie(s) détectée(s)")
                    
                    for i, anom in enumerate(anomalies):
                        icon = "⚠️" if anom['type'] == "warning" else "ℹ️"
                        
                        with st.expander(f"{icon} Anomalie {i+1}"):
                            st.write(f"**Message:** {anom['message']}")
                            st.write(f"**Recommandation:** {anom['recommendation']}")
                else:
                    st.success("✅ Aucune anomalie détectée")
    
    with tab5:
        st.subheader("📈 Analyse Avancée")
        
        if len(st.session_state.organoid_lab['experiments']) > 0:
            st.write("### 📊 Vue d'Ensemble")
            
            # Statistiques globales
            completed = sum(1 for exp in st.session_state.organoid_lab['experiments'] 
                           if exp.get('status') == 'completed')
            running = sum(1 for exp in st.session_state.organoid_lab['experiments'] 
                         if exp.get('status') == 'running')
            planned = sum(1 for exp in st.session_state.organoid_lab['experiments'] 
                         if exp.get('status') == 'planned')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total", len(st.session_state.organoid_lab['experiments']))
            with col2:
                st.metric("Complétées", completed)
            with col3:
                st.metric("En Cours", running)
            with col4:
                st.metric("Planifiées", planned)
            
            # Graphique types expériences
            exp_types = {}
            for exp in st.session_state.organoid_lab['experiments']:
                exp_type = exp['type']
                exp_types[exp_type] = exp_types.get(exp_type, 0) + 1
            
            fig = go.Figure(data=[go.Bar(
                x=list(exp_types.keys()),
                y=list(exp_types.values()),
                marker_color='#FF6B9D',
                text=list(exp_types.values()),
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Répartition Types Expériences",
                xaxis_title="Type",
                yaxis_title="Nombre",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Timeline expériences
            st.write("### ⏱️ Timeline Expériences")
            
            timeline_data = []
            for exp in st.session_state.organoid_lab['experiments']:
                timeline_data.append({
                    'Nom': exp['name'],
                    'Type': exp['type'],
                    'Date': exp['timestamp'][:10],
                    'Durée (min)': exp['duration_min'],
                    'Status': exp.get('status', 'planned')
                })
            
            df_timeline = pd.DataFrame(timeline_data)
            st.dataframe(df_timeline, use_container_width=True)
        else:
            st.info("Aucune expérience à analyser")

# ==================== FONCTIONNALITÉS ADDITIONNELLES - NOUVELLES PAGES ====================

# Ajouter ces nouvelles pages dans la sidebar navigation

# PAGE: Collaboration
elif page == "👥 Collaboration":
    st.header("👥 Collaboration & Partage")
    
    st.info("""
    **Plateforme Collaborative**
    
    Partagez vos organoïdes, expériences et résultats avec d'autres chercheurs.
    """)
    
    tab1, tab2, tab3 = st.tabs(["🌐 Partager", "📥 Importer", "👥 Équipe"])
    
    with tab1:
        st.subheader("🌐 Partager Ressources")
        
        resource_type = st.selectbox("Type Ressource",
            ["Organoïde", "Expérience", "Protocole", "Dataset"])
        
        if resource_type == "Organoïde":
            if st.session_state.organoid_lab['organoids']:
                org_to_share = st.selectbox("Sélectionner Organoïde",
                    list(st.session_state.organoid_lab['organoids'].keys()),
                    format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'])
                
                sharing_level = st.radio("Niveau Partage",
                    ["Public", "Équipe", "Privé (lien)"], horizontal=True)
                
                if st.button("🌐 Générer Lien Partage", type="primary"):
                    share_link = f"https://organoid-platform.com/share/{org_to_share}"
                    
                    st.success("✅ Lien généré!")
                    st.code(share_link)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📋 Copier Lien"):
                            st.info("Lien copié dans presse-papiers!")
                    
                    with col2:
                        if st.button("📧 Envoyer Email"):
                            st.info("Email envoyé!")
            else:
                st.info("Créez un organoïde pour partager")
    
    with tab2:
        st.subheader("📥 Importer Ressources")
        
        import_method = st.radio("Méthode Import",
            ["Lien", "Fichier", "Base de données"], horizontal=True)
        
        if import_method == "Lien":
            share_link = st.text_input("Lien Partage", "https://...")
            
            if st.button("📥 Importer", type="primary"):
                st.success("✅ Ressource importée!")
        
        elif import_method == "Fichier":
            uploaded_file = st.file_uploader("Choisir Fichier", type=['json', 'csv', 'h5'])
            
            if uploaded_file and st.button("📥 Importer"):
                st.success("✅ Fichier importé!")
    
    with tab3:
        st.subheader("👥 Gestion Équipe")
        
        st.write("### 👤 Membres Équipe")
        
        team_members = [
            {"name": "Dr. Alice Smith", "role": "Principal Investigator", "access": "Admin"},
            {"name": "Dr. Bob Johnson", "role": "Post-Doc", "access": "Editor"},
            {"name": "Jane Doe", "role": "PhD Student", "access": "Viewer"}
        ]
        
        for member in team_members:
            with st.expander(f"👤 {member['name']} - {member['role']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Rôle:** {member['role']}")
                with col2:
                    st.write(f"**Accès:** {member['access']}")
                with col3:
                    if st.button("🗑️ Retirer", key=f"remove_{member['name']}"):
                        st.warning("Membre retiré")
        
        st.write("### ➕ Inviter Membre")
        
        with st.form("invite_member"):
            col1, col2 = st.columns(2)
            
            with col1:
                email = st.text_input("Email")
                role = st.selectbox("Rôle", ["Viewer", "Editor", "Admin"])
            
            with col2:
                message = st.text_area("Message Invitation", "Rejoignez notre équipe!")
            
            if st.form_submit_button("📧 Envoyer Invitation"):
                st.success(f"✅ Invitation envoyée à {email}")

# PAGE: Bioéthique
elif page == "⚖️ Bioéthique":
    st.header("⚖️ Considérations Bioéthiques")
    
    st.info("""
    **Bioéthique des Organoïdes Cérébraux**
    
    Questions éthiques importantes sur la conscience, la sensibilité et l'utilisation.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Conscience", "📜 Régulations", "✅ Checklist", "📚 Ressources"])
    
    with tab1:
        st.subheader("🧠 Conscience & Sensibilité")
        
        st.write("""
        **Questions Clés:**
        
        1. **Les organoïdes peuvent-ils développer une forme de conscience?**
        2. **Faut-il limiter la taille/complexité des organoïdes?**
        3. **Comment détecter des signes de sensibilité?**
        4. **Quelles protections éthiques sont nécessaires?**
        """)
        
        st.write("### 📊 Évaluation Éthique de Vos Organoïdes")
        
        if st.session_state.organoid_lab['organoids']:
            for org_id, org in st.session_state.organoid_lab['organoids'].items():
                with st.expander(f"🧠 {org['name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Paramètres:**")
                        st.write(f"• Taille: {org['size_mm']} mm")
                        st.write(f"• Neurones: {org['neuron_count']:,}")
                        st.write(f"• Âge: {org['culture_duration_days']} jours")
                        st.write(f"• Région: {org['brain_region']}")
                    
                    with col2:
                        st.write("**Évaluation Éthique:**")
                        
                        # Calcul score risque
                        risk_score = 0
                        
                        if org['size_mm'] > 4:
                            risk_score += 2
                            st.warning("⚠️ Taille importante")
                        
                        if org['neuron_count'] > 1000000:
                            risk_score += 2
                            st.warning("⚠️ Haute complexité neuronale")
                        
                        if org['culture_duration_days'] > 180:
                            risk_score += 1
                            st.warning("⚠️ Maturation avancée")
                        
                        if org['brain_region'] == "Whole Brain":
                            risk_score += 2
                            st.warning("⚠️ Cerveau complet")
                        
                        if risk_score == 0:
                            st.success("✅ Risque éthique faible")
                        elif risk_score <= 3:
                            st.info("➖ Risque éthique modéré")
                        else:
                            st.error("❌ Risque éthique élevé - Révision nécessaire")
                        
                        st.metric("Score Risque", f"{risk_score}/7")
        else:
            st.info("Aucun organoïde à évaluer")
    
    with tab2:
        st.subheader("📜 Cadre Réglementaire")
        
        st.write("""
        **Régulations Internationales**
        """)
        
        regulations = {
            "États-Unis": {
                "organisme": "NIH, FDA",
                "restrictions": "Limite 280 jours culture, pas d'implantation animaux",
                "statut": "Guidelines 2021"
            },
            "Europe": {
                "organisme": "EMA, Comités d'éthique nationaux",
                "restrictions": "Évaluation cas par cas, consentement éclairé requis",
                "statut": "Cadre en développement"
            },
            "Japon": {
                "organisme": "MEXT",
                "restrictions": "Autorisation implantation animaux sous conditions",
                "statut": "Loi 2019"
            },
            "Chine": {
                "organisme": "MOST",
                "restrictions": "Encadrement strict recherche cellules souches",
                "statut": "Régulations 2020"
            }
        }
        
        for country, info in regulations.items():
            with st.expander(f"🌍 {country}"):
                st.write(f"**Organisme:** {info['organisme']}")
                st.write(f"**Restrictions:** {info['restrictions']}")
                st.write(f"**Statut:** {info['statut']}")
    
    with tab3:
        st.subheader("✅ Checklist Éthique")
        
        st.write("### 📋 Vérifications Obligatoires")
        
        checklist_items = [
            "Consentement éclairé obtenu pour cellules souches",
            "Approbation comité d'éthique institutionnel",
            "Protocole destruction organoïdes établi",
            "Limite durée culture respectée (< 280 jours)",
            "Pas d'implantation dans cerveaux animaux",
            "Surveillance signes activité organisée",
            "Documentation complète procédures",
            "Formation équipe aux enjeux éthiques",
            "Plan gestion découvertes inattendues",
            "Transparence et communication publique"
        ]
        
        completed = 0
        for i, item in enumerate(checklist_items):
            checked = st.checkbox(item, key=f"ethics_check_{i}")
            if checked:
                completed += 1
        
        progress = completed / len(checklist_items)
        st.progress(progress)
        
        if progress == 1.0:
            st.success("✅ Conformité éthique complète!")
        elif progress >= 0.7:
            st.warning(f"⚠️ Conformité partielle ({completed}/{len(checklist_items)})")
        else:
            st.error(f"❌ Conformité insuffisante ({completed}/{len(checklist_items)})")
    
    with tab4:
        st.subheader("📚 Ressources & Littérature")
        
        st.write("""
        **Publications Clés:**
        
        1. **Sawai et al. (2019)** - "Ethical considerations for human brain organoid research"
        2. **Hyun et al. (2020)** - "Human organoid ethics: NIH guidelines"
        3. **Lavazza & Massimini (2018)** - "Cerebral organoids: consciousness questions"
        4. **Farahany et al. (2018)** - "Neurorights framework"
        
        **Organisations:**
        - International Society for Stem Cell Research (ISSCR)
        - Nuffield Council on Bioethics
        - Presidential Commission for Bioethics
        """)
        
        if st.button("📚 Accéder Bibliothèque Complète"):
            st.info("Ouverture bibliothèque éthique...")

# PAGE: Publications
elif page == "📄 Publications":
    st.header("📄 Publications & Rapports")
    
    st.info("""
    **Génération Automatique Publications**
    
    Créez rapports, présentations et manuscrits à partir de vos données.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📝 Générer", "📊 Templates", "📤 Exporter"])
    
    with tab1:
        st.subheader("📝 Générateur Publication")
        
        pub_type = st.selectbox("Type Publication",
            ["Article Scientifique", "Rapport Technique", "Présentation",
             "Poster Conférence", "Thesis Chapter", "Grant Proposal"])
        
        if pub_type == "Article Scientifique":
            with st.form("generate_article"):
                title = st.text_input("Titre", "Novel Insights from Brain Organoid Computing")
                
                authors = st.text_area("Auteurs (un par ligne)",
                    "Dr. Jane Smith\nDr. John Doe\nAlice Johnson")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    journal = st.selectbox("Journal Cible",
                        ["Nature", "Science", "Cell", "Neuron", "PNAS", "eLife"])
                    
                    sections = st.multiselect("Sections Inclure",
                        ["Abstract", "Introduction", "Methods", "Results", 
                         "Discussion", "Conclusion", "References"],
                        default=["Abstract", "Methods", "Results"])
                
                with col2:
                    include_figures = st.checkbox("Inclure Figures", value=True)
                    include_stats = st.checkbox("Inclure Statistiques", value=True)
                    include_code = st.checkbox("Inclure Code", value=False)
                
                if st.form_submit_button("📝 Générer Article", type="primary"):
                    with st.spinner("Génération article..."):
                        import time
                        time.sleep(2)
                        
                        st.success("✅ Article généré!")
                        
                        # Afficher aperçu
                        st.markdown("---")
                        st.markdown(f"# {title}")
                        st.markdown(f"**Auteurs:** {authors.replace(chr(10), ', ')}")
                        st.markdown(f"**Journal:** {journal}")
                        st.markdown("---")
                        
                        if "Abstract" in sections:
                            st.markdown("## Abstract")
                            st.write("""
                            Brain organoids represent a powerful platform for studying neural 
                            development and disease. Here we present a comprehensive analysis 
                            of computational capabilities in human brain organoids...
                            """)
                        
                        if "Methods" in sections:
                            st.markdown("## Methods")
                            st.write("""
                            **Organoid Culture:** Organoids were generated from iPSCs following 
                            established protocols (Lancaster et al., 2013)...
                            """)
                        
                        if include_figures and "Results" in sections:
                            st.markdown("## Results")
                            
                            # Figure exemple
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=['Control', 'Treated'],
                                y=[100, 145],
                                error_y=dict(array=[10, 15]),
                                marker_color='#FF6B9D'
                            ))
                            fig.update_layout(
                                title="Figure 1: Response to Stimulation",
                                yaxis_title="Activity (%)",
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("📥 Télécharger PDF"):
                                st.info("PDF généré!")
                        
                        with col2:
                            if st.button("📝 Télécharger LaTeX"):
                                st.info("LaTeX généré!")
                        
                        with col3:
                            if st.button("📄 Télécharger Word"):
                                st.info("Word généré!")
    
    with tab2:
        st.subheader("📊 Templates")
        
        templates = {
            "Nature Article": {
                "length": "5000 words",
                "figures": "6 max",
                "format": "LaTeX preferred"
            },
            "Grant Proposal NIH": {
                "length": "12 pages",
                "sections": "Specific Aims, Research Strategy, Bibliography",
                "format": "PDF"
            },
            "Conference Poster": {
                "size": "A0 (841×1189mm)",
                "orientation": "Portrait/Landscape",
                "format": "PowerPoint, PDF"
            }
        }
        
        for template_name, info in templates.items():
            with st.expander(f"📄 {template_name}"):
                for key, value in info.items():
                    st.write(f"**{key}:** {value}")
                
                if st.button(f"📥 Utiliser Template", key=f"use_{template_name}"):
                    st.success(f"✅ Template '{template_name}' chargé!")
    
    with tab3:
        st.subheader("📤 Export & Partage")
        
        export_format = st.selectbox("Format Export",
            ["PDF", "LaTeX", "Word (.docx)", "Markdown", "HTML", "PowerPoint"])
        
        include_data = st.checkbox("Inclure Données Brutes")
        include_code = st.checkbox("Inclure Code Analyses")
        
        if st.button("📤 Exporter", type="primary"):
            st.success(f"✅ Export {export_format} généré!")
            st.download_button(
                label="⬇️ Télécharger",
                data="Export content here...",
                file_name=f"publication.{export_format.lower()}",
                mime="application/octet-stream"
            )

# PAGE: Monitoring Temps Réel
elif page == "📡 Monitoring Live":
    st.header("📡 Monitoring Temps Réel")
    
    st.info("""
    **Surveillance Continue**
    
    Monitoring en temps réel de vos organoïdes et expériences.
    """)
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=True)
    
    if auto_refresh:
        st.markdown("""
        <meta http-equiv="refresh" content="5">
        """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "⚠️ Alertes", "📈 Trends"])
    
    with tab1:
        st.subheader("📊 Dashboard Live")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Métriques temps réel simulées
        with col1:
            viability_live = np.random.uniform(88, 98)
            delta_viability = np.random.uniform(-2, 2)
            st.metric("Viabilité Moyenne", f"{viability_live:.1f}%", 
                     f"{delta_viability:+.1f}%")
        
        with col2:
            activity_live = np.random.uniform(3, 8)
            delta_activity = np.random.uniform(-1, 1)
            st.metric("Activité Moyenne", f"{activity_live:.1f} Hz",
                     f"{delta_activity:+.1f}")
        
        with col3:
            temp_live = np.random.uniform(36.5, 37.5)
            st.metric("Température", f"{temp_live:.1f}°C")
        
        with col4:
            o2_live = np.random.uniform(19, 21)
            st.metric("O₂", f"{o2_live:.1f}%")
        
        st.markdown("---")
        
        # Graphique temps réel
        st.write("### 📈 Activité Temps Réel (dernières 60s)")
        
        # Générer données
        time_points = list(range(60))
        activity_data = [np.random.uniform(3, 8) for _ in time_points]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=activity_data,
            mode='lines',
            line=dict(color='#FF6B9D', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Activité Neuronale Live",
            xaxis_title="Temps (s)",
            yaxis_title="Firing Rate (Hz)",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Status organoïdes
        st.write("### 🧠 Status Organoïdes")
        
        if st.session_state.organoid_lab['organoids']:
            for org_id, org in st.session_state.organoid_lab['organoids'].items():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"**{org['name']}**")
                
                with col2:
                    viability = org['viability']
                    if viability > 90:
                        st.success(f"✅ {viability:.1f}%")
                    elif viability > 80:
                        st.warning(f"⚠️ {viability:.1f}%")
                    else:
                        st.error(f"❌ {viability:.1f}%")
                
                with col3:
                    activity = np.random.uniform(3, 8)
                    st.write(f"⚡ {activity:.1f} Hz")
                
                with col4:
                    status_icon = "🟢" if np.random.random() > 0.2 else "🟡"
                    st.write(f"{status_icon} Online")
    
    with tab2:
        st.subheader("⚠️ Alertes Système")
        
        # Alertes simulées
        alerts = [
            {
                "level": "warning",
                "message": "Viabilité ORG-002 < 85%",
                "time": "Il y a 2 min",
                "action": "Vérifier milieu"
            },
            {
                "level": "info",
                "message": "Expérience EXP-005 terminée",
                "time": "Il y a 15 min",
                "action": "Analyser résultats"
            }
        ]
        
        for alert in alerts:
            if alert['level'] == "warning":
                st.warning(f"⚠️ **{alert['message']}** - {alert['time']}")
            elif alert['level'] == "error":
                st.error(f"❌ **{alert['message']}** - {alert['time']}")
            else:
                st.info(f"ℹ️ **{alert['message']}** - {alert['time']}")
            
            st.write(f"   ➡️ {alert['action']}")
            st.markdown("---")
        
        # Configuration alertes
        st.write("### ⚙️ Configuration Alertes")
        
        with st.form("alert_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                viability_threshold = st.slider("Seuil viabilité (%)", 50, 95, 85)
                activity_threshold = st.slider("Seuil activité (Hz)", 0, 20, 2)
            
            with col2:
                email_alerts = st.checkbox("Alertes Email", value=True)
                sms_alerts = st.checkbox("Alertes SMS")
                sound_alerts = st.checkbox("Alertes Sonores", value=True)
            
            if st.form_submit_button("💾 Sauvegarder Configuration"):
                st.success("✅ Configuration alertes sauvegardée!")
    
    with tab3:
        st.subheader("📈 Trends Long Terme")
        
        # Générer données tendance
        days = list(range(30))
        viability_trend = [95 - i*0.1 + np.random.normal(0, 1) for i in days]
        activity_trend = [5 + np.sin(i/5) + np.random.normal(0, 0.5) for i in days]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Viabilité (30 jours)", "Activité (30 jours)")
        )
        
        fig.add_trace(go.Scatter(
            x=days, y=viability_trend,
            mode='lines+markers',
            line=dict(color='#FF6B9D', width=2),
            name='Viabilité'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=days, y=activity_trend,
            mode='lines+markers',
            line=dict(color='#355C7D', width=2),
            name='Activité'
        ), row=2, col=1)
        
        fig.update_xaxes(title_text="Jours", row=2, col=1)
        fig.update_yaxes(title_text="Viabilité (%)", row=1, col=1)
        fig.update_yaxes(title_text="Activité (Hz)", row=2, col=1)
        
        fig.update_layout(
            template="plotly_dark",
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Ajouter "📡 Monitoring Live", "⚖️ Bioéthique", "👥 Collaboration", "📄 Publications" à la navigation sidebar
# ==================== PAGE: STIMULATION ====================
elif page == "🎯 Stimulation":
    st.header("🎯 Stimulation Neuronale")
    
    st.info("""
    **Stimulation**
    
    Techniques pour activer/moduler l'activité neuronale.
    
    **Méthodes:**
    - Électrique (courant direct)
    - Optogénétique (lumière)
    - Chimique (neurotransmetteurs)
    - Magnétique (TMS)
    """)
    
    tab1, tab2, tab3 = st.tabs(["⚡ Électrique", "💡 Optogénétique", "💊 Chimique"])
    
    with tab1:
        st.subheader("⚡ Stimulation Électrique")
        
        if st.session_state.organoid_lab['organoids']:
            selected_organoid = st.selectbox("Organoïde",
                list(st.session_state.organoid_lab['organoids'].keys()),
                format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                key="stim_org")
            
            col1, col2 = st.columns(2)
            
            with col1:
                stim_pattern = st.selectbox("Pattern",
                    ["Pulse unique", "Train pulses", "Rampe", "Burst", "Continu"])
                
                amplitude_ua = st.slider("Amplitude (μA)", 1, 100, 10)
                
                if stim_pattern != "Pulse unique":
                    frequency_hz = st.slider("Fréquence (Hz)", 1, 100, 10)
            
            with col2:
                duration_ms = st.slider("Durée (ms)", 1, 1000, 100)
                
                pulse_width_ms = st.slider("Largeur pulse (ms)", 0.1, 10.0, 1.0, 0.1)
            
            if st.button("⚡ Stimuler", type="primary"):
                with st.spinner("Stimulation en cours..."):
                    import time
                    time.sleep(1)
                    
                    t = np.linspace(0, duration_ms, int(duration_ms * 10))
                    
                    if stim_pattern == "Pulse unique":
                        stim_signal = np.zeros_like(t)
                        stim_signal[(t > 10) & (t < 10 + pulse_width_ms)] = amplitude_ua
                    
                    elif stim_pattern == "Train pulses":
                        stim_signal = np.zeros_like(t)
                        period = 1000 / frequency_hz
                        for i in range(int(duration_ms / period)):
                            start_t = i * period
                            stim_signal[(t > start_t) & (t < start_t + pulse_width_ms)] = amplitude_ua
                    
                    elif stim_pattern == "Rampe":
                        stim_signal = amplitude_ua * t / duration_ms
                    
                    else:  # Continu
                        stim_signal = np.ones_like(t) * amplitude_ua
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=t, y=stim_signal,
                        mode='lines',
                        line=dict(color='#FF6B9D', width=2),
                        fill='tozeroy'
                    ))
                    
                    fig.update_layout(
                        title=f"Signal Stimulation - {stim_pattern}",
                        xaxis_title="Temps (ms)",
                        yaxis_title="Courant (μA)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Stimulation appliquée!")
                    
                    # Sauvegarder
                    stimulation = {
                        'organoid_id': selected_organoid,
                        'type': 'Électrique',
                        'pattern': stim_pattern,
                        'amplitude_ua': amplitude_ua,
                        'duration_ms': duration_ms,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.organoid_lab['stimulations'].append(stimulation)
                    log_event(f"Stimulation électrique: {stim_pattern}", "SUCCESS")
        else:
            st.info("Créez un organoïde")
    
    with tab2:
        st.subheader("💡 Optogénétique")
        
        st.write("""
        **Optogénétique**
        
        Contrôle activité neuronale par lumière + protéines photosensibles.
        
        **Opsines courantes:**
        - Channelrhodopsin-2 (ChR2): Activation (lumière bleue)
        - Halorhodopsin (NpHR): Inhibition (lumière jaune)
        - Archaerhodopsin (Arch): Inhibition (lumière verte)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            opsin = st.selectbox("Opsine",
                ["ChR2 (Activation)", "NpHR (Inhibition)", "Arch (Inhibition)"])
            
            wavelength = st.slider("Longueur d'onde (nm)", 450, 650, 470 if "ChR2" in opsin else 590)
        
        with col2:
            light_intensity = st.slider("Intensité (mW/mm²)", 0.1, 10.0, 1.0, 0.1)
            
            pulse_duration = st.slider("Durée pulse (ms)", 1, 100, 10)
        
        if st.button("💡 Photostimuler", type="primary"):
            st.success(f"✅ Photostimulation {opsin} appliquée!")
            st.info(f"💡 Lumière {wavelength}nm, {light_intensity} mW/mm², {pulse_duration}ms")
            
            if "ChR2" in opsin:
                st.write("**Effet:** Dépolarisation → Augmentation activité")
            else:
                st.write("**Effet:** Hyperpolarisation → Inhibition activité")
    
    with tab3:
        st.subheader("💊 Stimulation Chimique")
        
        st.write("""
        **Pharmacologie**
        
        Modulation activité par neurotransmetteurs et drogues.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            compound = st.selectbox("Composé",
                ["Glutamate (excitateur)", "GABA (inhibiteur)", 
                 "Dopamine", "Sérotonine", "Acétylcholine",
                 "Bicuculline (bloque GABA)", "TTX (bloque Na+)",
                 "APV (bloque NMDA)"])
            
            concentration_um = st.slider("Concentration (μM)", 0.1, 1000.0, 10.0, 0.1)
        
        with col2:
            application_method = st.selectbox("Méthode",
                ["Bath application", "Perfusion", "Microinjection", "Puff"])
            
            wash_duration = st.slider("Durée lavage (min)", 0, 30, 5)
        
        if st.button("💊 Appliquer Composé", type="primary"):
            st.success(f"✅ {compound} appliqué à {concentration_um} μM")
            
            # Effet prédit
            if "excitateur" in compound or compound == "Glutamate":
                st.info("📈 Effet attendu: Augmentation activité neuronale")
            elif "inhibiteur" in compound or "GABA" in compound:
                st.info("📉 Effet attendu: Diminution activité neuronale")
            elif "bloque" in compound:
                st.warning("🚫 Effet attendu: Blocage canaux/récepteurs")
            
            # Sauvegarder
            pharmacology = {
                'compound': compound,
                'concentration_um': concentration_um,
                'method': application_method,
                'timestamp': datetime.now().isoformat()
            }
            st.session_state.organoid_lab['pharmacology'].append(pharmacology)
            log_event(f"Application pharmacologique: {compound}", "SUCCESS")

# ==================== PAGE: APPRENTISSAGE ====================
elif page == "🧠 Apprentissage":
    st.header("🧠 Apprentissage & Plasticité")
    
    st.info("""
    **Apprentissage Neuronal**
    
    Capacité du réseau à modifier ses connexions en réponse à l'expérience.
    
    **Mécanismes:**
    - Plasticité synaptique (LTP/LTD)
    - Neurogénèse
    - Pruning synaptique
    - Renforcement sélectif
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Protocoles", "🎯 Entraînement", "📊 Performance", "🧠 Mémoire"])
    
    with tab1:
        st.subheader("📚 Protocoles d'Apprentissage")
        
        protocol = st.selectbox("Protocole",
            ["Conditionnement Pavlovien", "Apprentissage Hebbien",
             "Renforcement (Reward-based)", "Pattern Recognition",
             "Sequence Learning"])
        
        if protocol == "Conditionnement Pavlovien":
            st.write("""
            **Conditionnement Classique**
            
            Association stimulus neutre + stimulus inconditionnel.
            
            **Phases:**
            1. Habituation
            2. Acquisition (CS + US)
            3. Consolidation
            4. Test (CS seul)
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_trials = st.slider("Nombre essais", 10, 200, 50)
                isi_ms = st.slider("Intervalle inter-stimuli (ms)", 100, 1000, 250)
            
            with col2:
                cs_duration = st.slider("Durée CS (ms)", 100, 1000, 500)
                us_duration = st.slider("Durée US (ms)", 50, 500, 100)
            
            if st.button("🧪 Lancer Conditionnement", type="primary"):
                with st.spinner("Entraînement en cours..."):
                    import time
                    time.sleep(2)
                    
                    # Simuler courbe apprentissage
                    trials = np.arange(1, n_trials + 1)
                    
                    # Réponse conditionnée (sigmoïde)
                    response = 100 / (1 + np.exp(-(trials - 25) / 10))
                    response += np.random.normal(0, 5, n_trials)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=trials, y=response,
                        mode='lines+markers',
                        line=dict(color='#FF6B9D', width=3),
                        marker=dict(size=6),
                        name='Réponse CR'
                    ))
                    
                    fig.add_hline(y=50, line_dash="dash", line_color="white",
                                 annotation_text="50% critère")
                    
                    fig.update_layout(
                        title="Courbe d'Acquisition - Conditionnement",
                        xaxis_title="Essai #",
                        yaxis_title="Réponse Conditionnée (%)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Critère atteint
                    criterion_trial = np.where(response > 50)[0]
                    if len(criterion_trial) > 0:
                        st.success(f"✅ Critère atteint à l'essai {criterion_trial[0] + 1}")
                    else:
                        st.warning("⚠️ Critère non atteint")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Essais totaux", n_trials)
                    with col2:
                        st.metric("Réponse finale", f"{response[-1]:.1f}%")
                    with col3:
                        learning_rate = (response[-1] - response[0]) / n_trials
                        st.metric("Taux apprentissage", f"{learning_rate:.2f}%/essai")
        
        elif protocol == "Pattern Recognition":
            st.write("""
            **Reconnaissance de Patterns**
            
            Entraîner réseau à reconnaître patterns spatiaux.
            """)
            
            if st.button("🎯 Entraîner Recognition", type="primary"):
                # Générer patterns
                n_patterns = 5
                pattern_size = 8
                
                patterns = []
                for i in range(n_patterns):
                    pattern = np.random.randint(0, 2, (pattern_size, pattern_size))
                    patterns.append(pattern)
                
                # Afficher patterns
                fig = make_subplots(
                    rows=1, cols=n_patterns,
                    subplot_titles=[f"Pattern {i+1}" for i in range(n_patterns)]
                )
                
                for i, pattern in enumerate(patterns):
                    fig.add_trace(go.Heatmap(
                        z=pattern,
                        colorscale='Greys',
                        showscale=False
                    ), row=1, col=i+1)
                
                fig.update_layout(
                    title="Patterns à Apprendre",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ {n_patterns} patterns présentés au réseau")
                st.info("🧠 Réseau encodant patterns via plasticité synaptique...")
    
    with tab2:
        st.subheader("🎯 Session d'Entraînement")
        
        if st.session_state.organoid_lab['organoids']:
            selected_organoid = st.selectbox("Organoïde",
                list(st.session_state.organoid_lab['organoids'].keys()),
                format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                key="train_org")
            
            with st.form("training_session"):
                task_type = st.selectbox("Tâche",
                    ["Classification", "Séquence temporelle", "Association",
                     "Prédiction", "Mémoire de travail"])
                
                n_epochs = st.slider("Époques", 10, 1000, 100)
                
                learning_rule = st.selectbox("Règle apprentissage",
                    ["STDP", "Hebbien", "BCM", "Reward-modulated"])
                
                if st.form_submit_button("🚀 Lancer Entraînement", type="primary"):
                    with st.spinner(f"Entraînement {n_epochs} époques..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        import time
                        
                        # Simuler entraînement
                        accuracies = []
                        losses = []
                        
                        for epoch in range(n_epochs):
                            # Simuler métrique
                            accuracy = 100 * (1 - np.exp(-epoch/50)) + np.random.normal(0, 2)
                            loss = 100 * np.exp(-epoch/50) + np.random.normal(0, 5)
                            
                            accuracies.append(accuracy)
                            losses.append(loss)
                            
                            if epoch % 10 == 0:
                                progress_bar.progress((epoch + 1) / n_epochs)
                                status_text.text(f"Époque {epoch+1}/{n_epochs} - Accuracy: {accuracy:.1f}%")
                                time.sleep(0.05)
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Entraînement terminé!")
                        
                        # Graphiques
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Accuracy", "Loss")
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(n_epochs)), y=accuracies,
                            mode='lines',
                            line=dict(color='#FF6B9D', width=2),
                            name='Accuracy'
                        ), row=1, col=1)
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(n_epochs)), y=losses,
                            mode='lines',
                            line=dict(color='#355C7D', width=2),
                            name='Loss'
                        ), row=1, col=2)
                        
                        fig.update_xaxes(title_text="Époque", row=1, col=1)
                        fig.update_xaxes(title_text="Époque", row=1, col=2)
                        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
                        fig.update_yaxes(title_text="Loss", row=1, col=2)
                        
                        fig.update_layout(
                            title=f"Entraînement - {task_type}",
                            template="plotly_dark",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Métriques finales
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy finale", f"{accuracies[-1]:.1f}%")
                        with col2:
                            st.metric("Loss finale", f"{losses[-1]:.2f}")
                        with col3:
                            improvement = accuracies[-1] - accuracies[0]
                            st.metric("Amélioration", f"+{improvement:.1f}%")
                        
                        # Sauvegarder
                        training = {
                            'organoid_id': selected_organoid,
                            'task': task_type,
                            'n_epochs': n_epochs,
                            'final_accuracy': accuracies[-1],
                            'final_loss': losses[-1],
                            'learning_rule': learning_rule,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.organoid_lab['training_sessions'].append(training)
                        log_event(f"Entraînement complété: {task_type}", "SUCCESS")
        else:
            st.info("Créez un organoïde")
    
    with tab3:
        st.subheader("📊 Performance & Évaluation")
        
        if st.session_state.organoid_lab['training_sessions']:
            st.write(f"### 📋 {len(st.session_state.organoid_lab['training_sessions'])} Sessions d'Entraînement")
            
            # Tableau récapitulatif
            sessions_data = []
            for session in st.session_state.organoid_lab['training_sessions']:
                sessions_data.append({
                    'Tâche': session['task'],
                    'Époques': session['n_epochs'],
                    'Accuracy': f"{session['final_accuracy']:.1f}%",
                    'Loss': f"{session['final_loss']:.2f}",
                    'Date': session['timestamp'][:19]
                })
            
            df_sessions = pd.DataFrame(sessions_data)
            st.dataframe(df_sessions, use_container_width=True)
            
            # Comparaison performances
            st.write("### 📈 Comparaison Tâches")
            
            tasks = [s['task'] for s in st.session_state.organoid_lab['training_sessions']]
            accuracies = [s['final_accuracy'] for s in st.session_state.organoid_lab['training_sessions']]
            
            fig = go.Figure(data=[go.Bar(
                x=tasks,
                y=accuracies,
                marker_color='#FF6B9D',
                text=[f"{a:.1f}%" for a in accuracies],
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Performance par Tâche",
                xaxis_title="Tâche",
                yaxis_title="Accuracy (%)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune session d'entraînement")
    
    with tab4:
        st.subheader("🧠 Tests de Mémoire")
        
        st.write("""
        **Capacité Mémoire**
        
        - Mémoire de travail (court terme)
        - Consolidation (long terme)
        - Rappel (retrieval)
        """)
        
        memory_test = st.selectbox("Type Test",
            ["Capacité mémoire de travail", "Rétention long terme",
             "Rappel après interférence", "Reconnaissance vs Rappel"])
        
        if memory_test == "Capacité mémoire de travail":
            st.write("""
            **Test Empan Mnésique**
            
            Nombre d'items mémorisables simultanément.
            """)
            
            if st.button("🧪 Tester Capacité", type="primary"):
                # Simuler test empan
                span_sizes = np.arange(1, 10)
                correct_rates = 100 * np.exp(-(span_sizes - 2)**2 / 8)
                correct_rates = np.clip(correct_rates, 0, 100)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=span_sizes, y=correct_rates,
                    mode='lines+markers',
                    line=dict(color='#FF6B9D', width=3),
                    marker=dict(size=10)
                ))
                
                fig.add_hline(y=50, line_dash="dash", line_color="white",
                             annotation_text="Seuil 50%")
                
                fig.update_layout(
                    title="Courbe Empan Mnésique",
                    xaxis_title="Nombre d'Items",
                    yaxis_title="% Correct",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Empan = dernier point > 50%
                span = np.where(correct_rates > 50)[0]
                if len(span) > 0:
                    working_memory_span = span[-1] + 1
                    st.success(f"✅ Empan mémoire de travail: {working_memory_span} items")
                else:
                    st.warning("Empan < 1 item")

# ==================== PAGE: BIOCOMPUTING ====================
elif page == "💻 Biocomputing":
    st.header("💻 Biocomputing & Calcul Neuronal")
    
    st.info("""
    **Biocomputing**
    
    Utiliser organoïdes cérébraux comme substrat de calcul.
    
    **Avantages:**
    - Efficacité énergétique extrême (~20W pour cerveau humain)
    - Parallélisme massif
    - Apprentissage adaptatif naturel
    - Traitement analogique
    
    **Défis:**
    - Interface I/O
    - Reproductibilité
    - Éthique
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💻 Tâches", "⚡ Performance", "🔌 Interface", "📊 Benchmarks"])
    
    with tab1:
        st.subheader("💻 Tâches de Calcul")
        
        if st.session_state.organoid_lab['organoids']:
            selected_organoid = st.selectbox("Organoïde",
                list(st.session_state.organoid_lab['organoids'].keys()),
                format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                key="comp_org")
            
            organoid = st.session_state.organoid_lab['organoids'][selected_organoid]
            
            task = st.selectbox("Tâche de Calcul",
                ["Classification Images", "Prédiction Séries Temporelles",
                 "Fonction XOR", "Pattern Matching", "Traitement Signal"])
            
            if task == "Fonction XOR":
                st.write("""
                **Fonction XOR**
                
                Problème classique non-linéaire.
                
                | A | B | XOR |
                |---|---|-----|
                | 0 | 0 |  0  |
                | 0 | 1 |  1  |
                | 1 | 0 |  1  |
                | 1 | 1 |  0  |
                """)
                
                if st.button("🧮 Calculer XOR", type="primary"):
                    with st.spinner("Calcul neuronal..."):
                        import time
                        time.sleep(1.5)
                        
                        # Simuler calcul
                        inputs = [(0,0), (0,1), (1,0), (1,1)]
                        expected = [0, 1, 1, 0]
                        
                        # Résultats avec bruit
                        results = []
                        for inp, exp in zip(inputs, expected):
                            result = exp + np.random.normal(0, 0.1)
                            result = np.clip(result, 0, 1)
                            results.append(result)
                        
                        # Afficher résultats
                        results_data = {
                            'A': [inp[0] for inp in inputs],
                            'B': [inp[1] for inp in inputs],
                            'Attendu': expected,
                            'Calculé': [f"{r:.3f}" for r in results],
                            'Correct': ['✅' if abs(r - e) < 0.2 else '❌' 
                                      for r, e in zip(results, expected)]
                        }
                        
                        df_xor = pd.DataFrame(results_data)
                        st.dataframe(df_xor, use_container_width=True)
                        
                        accuracy = sum(abs(r - e) < 0.2 for r, e in zip(results, expected)) / len(expected)
                        
                        st.metric("Accuracy", f"{accuracy*100:.0f}%")
                        
                        if accuracy == 1.0:
                            st.success("🎉 XOR parfaitement résolu!")
                            st.balloons()
                        
                        # Sauvegarder
                        computation = {
                            'organoid_id': selected_organoid,
                            'task': task,
                            'accuracy': accuracy * 100,
                            'timestamp': datetime.now().isoformat()
                        }
                        st.session_state.organoid_lab['computations'].append(computation)
                        log_event(f"Calcul XOR: {accuracy*100:.0f}%", "SUCCESS")
            
            elif task == "Classification Images":
                st.write("### 🖼️ Classification Images")
                
                n_classes = st.slider("Nombre classes", 2, 10, 3)
                image_size = st.selectbox("Taille image", ["28x28", "64x64", "128x128"])
                
                if st.button("🖼️ Classifier", type="primary"):
                    with st.spinner("Classification..."):
                        import time
                        time.sleep(2)
                        
                        # Simuler classification
                        accuracy = np.random.uniform(70, 95)
                        
                        st.success(f"✅ Classification: {accuracy:.1f}% accuracy")
                        
                        # Matrice confusion
                        confusion = np.random.randint(0, 100, (n_classes, n_classes))
                        np.fill_diagonal(confusion, np.random.randint(80, 100, n_classes))
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=confusion,
                            x=[f"Classe {i}" for i in range(n_classes)],
                            y=[f"Classe {i}" for i in range(n_classes)],
                            colorscale='Blues',
                            text=confusion,
                            texttemplate="%{text}",
                            textfont={"size": 12},
                            colorbar=dict(title="Count")
                        ))
                        
                        fig.update_layout(
                            title="Matrice de Confusion",
                            xaxis_title="Prédiction",
                            yaxis_title="Vérité",
                            template="plotly_dark",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Créez un organoïde")
    
    with tab2:
        st.subheader("⚡ Performance & Efficacité")
        
        st.write("### 🔋 Comparaison Énergétique")
        
        # Comparaison avec systèmes artificiels
        systems = {
            'Système': ['Organoïde (ce lab)', 'Cerveau Humain', 'GPU (NVIDIA A100)', 
                       'CPU (Intel i9)', 'Supercalculateur'],
            'Puissance (W)': [0.1, 20, 400, 125, 20000000],
            'FLOPS': [1e12, 1e16, 1.9e14, 1e12, 1e18],
            'FLOPS/W': [1e13, 5e14, 4.75e11, 8e9, 5e10]
        }
        
        df_systems = pd.DataFrame(systems)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_systems['Système'],
            y=df_systems['FLOPS/W'],
            marker_color=['#FF6B9D', '#C06C84', '#6C5B7B', '#355C7D', '#1a1a2e'],
            text=[f"{v:.2e}" for v in df_systems['FLOPS/W']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Efficacité Énergétique (FLOPS/Watt)",
            xaxis_title="Système",
            yaxis_title="FLOPS/W",
            yaxis_type="log",
            template="plotly_dark",
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("🧠 Systèmes biologiques: efficacité énergétique supérieure de plusieurs ordres de grandeur!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Avantages Biocomputing:**")
            st.write("✅ Efficacité énergétique extrême")
            st.write("✅ Parallélisme massif naturel")
            st.write("✅ Apprentissage adaptatif")
            st.write("✅ Traitement analogique")
        
        with col2:
            st.write("**Limitations:**")
            st.write("⚠️ Vitesse calcul (ms vs ns)")
            st.write("⚠️ Précision limitée")
            st.write("⚠️ Interface complexe")
            st.write("⚠️ Reproductibilité")
    
    with tab3:
        st.subheader("🔌 Interface Input/Output")
        
        st.write("""
        **Défi Interface**
        
        Convertir données numériques ↔ signaux neuronaux.
        
        **Approches:**
        - MEA (Multi-Electrode Array)
        - Optogénétique
        - Stimulation chimique
        - Calcium imaging pour output
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📥 INPUT")
            
            input_method = st.selectbox("Méthode Input",
                ["Stimulation électrique", "Optogénétique", "Chimique"])
            
            if input_method == "Stimulation électrique":
                st.write("**Encodage:**")
                st.write("• Fréquence → Amplitude signal")
                st.write("• Position électrode → Feature spatiale")
                st.write("• Timing → Feature temporelle")
        
        with col2:
            st.write("### 📤 OUTPUT")
            
            output_method = st.selectbox("Méthode Output",
                ["MEA recording", "Calcium imaging", "Voltage imaging"])
            
            if output_method == "MEA recording":
                st.write("**Décodage:**")
                st.write("• Taux de firing → Valeur numérique")
                st.write("• Population coding")
                st.write("• Temporal patterns")
        
        st.write("### 🔄 Exemple Pipeline")
        
        if st.button("🔄 Tester Pipeline I/O", type="primary"):
            st.write("**1. INPUT:** Image 28x28 pixels")
            st.write("   → Conversion en patterns stimulation")
            st.write("   → 784 électrodes (1 par pixel)")
            
            st.write("**2. PROCESSING:** Réseau neuronal traite")
            st.write("   → Propagation activité")
            st.write("   → Computation distribuée")
            
            st.write("**3. OUTPUT:** Enregistrement activité")
            st.write("   → Population 10 neurones (classes)")
            st.write("   → Décodage: neurone le plus actif = classe")
            
            st.success("✅ Pipeline fonctionnel!")
    
    with tab4:
        st.subheader("📊 Benchmarks")
        
        if st.session_state.organoid_lab['computations']:
            st.write(f"### 📋 {len(st.session_state.organoid_lab['computations'])} Calculs Effectués")
            
            comp_data = []
            for comp in st.session_state.organoid_lab['computations']:
                comp_data.append({
                    'Tâche': comp['task'],
                    'Accuracy': f"{comp['accuracy']:.1f}%",
                    'Date': comp['timestamp'][:19]
                })
            
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.info("Effectuez des calculs pour voir les benchmarks")

# ==================== PAGE: EXPÉRIENCES ====================
elif page == "🔬 Expériences":
    st.header("🔬 Expériences & Protocoles")
    
    tab1, tab2 = st.tabs(["📋 Designer", "📊 Historique"])
    
    with tab1:
        st.subheader("📋 Designer d'Expérience")
        
        with st.form("design_experiment"):
            exp_name = st.text_input("Nom Expérience", "EXP-001")
            
            exp_type = st.selectbox("Type",
                ["Électrophysiologie", "Pharmacologie", "Stimulation",
                 "Apprentissage", "Imagerie", "Biocomputing"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                duration_min = st.number_input("Durée (min)", 1, 480, 60)
                n_trials = st.number_input("Essais", 1, 1000, 10)
            
            with col2:
                if st.session_state.organoid_lab['organoids']:
                    organoid_id = st.selectbox("Organoïde",
                        list(st.session_state.organoid_lab['organoids'].keys()),
                        format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'])
                else:
                    st.warning("Créez un organoïde")
                    organoid_id = None
            
            hypothesis = st.text_area("Hypothèse",
                "L'organoïde montrera une augmentation d'activité après stimulation répétée")
            
            protocol_steps = st.text_area("Protocole (étapes)",
                "1. Baseline (10 min)\n2. Stimulation (30 min)\n3. Recording post (20 min)")
            
            if st.form_submit_button("🚀 Lancer Expérience", type="primary"):
                if organoid_id:
                    experiment = {
                        'name': exp_name,
                        'type': exp_type,
                        'organoid_id': organoid_id,
                        'duration_min': duration_min,
                        'n_trials': n_trials,
                        'hypothesis': hypothesis,
                        'protocol': protocol_steps,
                        'status': 'completed',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.organoid_lab['experiments'].append(experiment)
                    log_event(f"Expérience créée: {exp_name}", "SUCCESS")
                    
                    st.success(f"✅ Expérience '{exp_name}' lancée!")
                    st.rerun()
                else:
                    st.error("Sélectionnez un organoïde")
    
    with tab2:
        st.subheader("📊 Historique Expériences")
        
        if st.session_state.organoid_lab['experiments']:
            for i, exp in enumerate(st.session_state.organoid_lab['experiments'][::-1]):
                with st.expander(f"🔬 {exp['name']} - {exp['type']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📋 Info")
                        st.write(f"**Type:** {exp['type']}")
                        st.write(f"**Durée:** {exp['duration_min']} min")
                        st.write(f"**Essais:** {exp['n_trials']}")
                    
                    with col2:
                        st.write("### 🎯 Hypothèse")
                        st.write(exp['hypothesis'])
                    
                    with col3:
                        st.write("### 📅 Date")
                        st.write(exp['timestamp'][:19])
                        
                        status = exp.get('status', 'pending')
                        if status == 'completed':
                            st.success("✅ Complété")
                        else:
                            st.info("🔄 En cours")
                    
                    st.write("### 📝 Protocole")
                    st.text(exp['protocol'])
        else:
            st.info("Aucune expérience enregistrée")

# ==================== PAGE: ENREGISTREMENTS ====================
elif page == "📈 Enregistrements":
    st.header("📈 Enregistrements & Données")
    
    st.info("""
    **Base de Données Enregistrements**
    
    Archive de toutes les mesures et enregistrements effectués.
    """)
    
    tab1, tab2 = st.tabs(["📊 Tous", "🔍 Filtrer"])
    
    with tab1:
        st.subheader("📊 Tous les Enregistrements")
        
        total_recordings = len(st.session_state.organoid_lab['recordings'])
        
        if total_recordings > 0:
            st.write(f"### 📋 {total_recordings} Enregistrements")
            
            # Convertir en DataFrame
            recordings_data = []
            for rec in st.session_state.organoid_lab['recordings']:
                recordings_data.append({
                    'Durée (s)': rec.get('duration_s', 0),
                    'Neurones': rec.get('n_neurons', 0),
                    'Spikes': rec.get('total_spikes', 0),
                    'Taux (Hz)': rec.get('firing_rate', 0),
                    'Date': rec['timestamp'][:19]
                })
            
            df_rec = pd.DataFrame(recordings_data)
            
            st.dataframe(df_rec, use_container_width=True)
            
            # Statistiques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_spikes = df_rec['Spikes'].sum()
                st.metric("Total Spikes", f"{total_spikes:,}")
            
            with col2:
                avg_rate = df_rec['Taux (Hz)'].mean()
                st.metric("Taux Moyen", f"{avg_rate:.2f} Hz")
            
            with col3:
                total_duration = df_rec['Durée (s)'].sum()
                st.metric("Durée Totale", f"{total_duration:.1f} s")
            
            with col4:
                st.metric("Enregistrements", total_recordings)
            
            # Graphique évolution
            st.write("### 📈 Évolution Activité")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_rec.index,
                y=df_rec['Taux (Hz)'],
                mode='lines+markers',
                line=dict(color='#FF6B9D', width=2),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Taux de Firing au Fil du Temps",
                xaxis_title="Enregistrement #",
                yaxis_title="Taux Firing (Hz)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun enregistrement")
    
    with tab2:
        st.subheader("🔍 Filtrer Enregistrements")
        
        if st.session_state.organoid_lab['recordings']:
            col1, col2 = st.columns(2)
            
            with col1:
                min_rate = st.slider("Taux minimum (Hz)", 0.0, 50.0, 0.0)
            
            with col2:
                min_duration = st.slider("Durée minimum (s)", 0.0, 10.0, 0.0)
            
            # Filtrer
            filtered = []
            for rec in st.session_state.organoid_lab['recordings']:
                if (rec.get('firing_rate', 0) >= min_rate and 
                    rec.get('duration_s', 0) >= min_duration):
                    filtered.append(rec)
            
            st.write(f"### 📊 {len(filtered)} Enregistrements Filtrés")
            
            if filtered:
                filtered_data = []
                for rec in filtered:
                    filtered_data.append({
                        'Durée (s)': rec.get('duration_s', 0),
                        'Taux (Hz)': rec.get('firing_rate', 0),
                        'Spikes': rec.get('total_spikes', 0),
                        'Date': rec['timestamp'][:19]
                    })
                
                df_filtered = pd.DataFrame(filtered_data)
                st.dataframe(df_filtered, use_container_width=True)
            else:
                st.info("Aucun enregistrement ne correspond aux critères")
        else:
            st.info("Aucun enregistrement disponible")

# ==================== PAGE: PHARMACOLOGIE ====================
elif page == "🧪 Pharmacologie":
    st.header("🧪 Pharmacologie & Modulateurs")
    
    st.info("""
    **Pharmacologie**
    
    Étude effets de composés chimiques sur activité neuronale.
    
    **Catégories:**
    - Neurotransmetteurs
    - Antagonistes
    - Modulateurs
    - Bloqueurs canaux
    """)
    
    tab1, tab2, tab3 = st.tabs(["💊 Bibliothèque", "🧪 Appliquer", "📊 Historique"])
    
    with tab1:
        st.subheader("💊 Bibliothèque Pharmacologique")
        
        compounds = {
            'Glutamate': {
                'type': 'Neurotransmetteur excitateur',
                'target': 'Récepteurs AMPA/NMDA',
                'effect': 'Augmentation activité',
                'concentration': '1-100 μM'
            },
            'GABA': {
                'type': 'Neurotransmetteur inhibiteur',
                'target': 'Récepteurs GABA_A/B',
                'effect': 'Diminution activité',
                'concentration': '1-1000 μM'
            },
            'Bicuculline': {
                'type': 'Antagoniste GABA_A',
                'target': 'Récepteurs GABA_A',
                'effect': 'Désinhibition → hyperactivité',
                'concentration': '10-50 μM'
            },
            'APV (D-AP5)': {
                'type': 'Antagoniste NMDA',
                'target': 'Récepteurs NMDA',
                'effect': 'Blocage plasticité synaptique',
                'concentration': '25-100 μM'
            },
            'TTX': {
                'type': 'Neurotoxine',
                'target': 'Canaux Na+ voltage-dépendants',
                'effect': 'Blocage potentiels action',
                'concentration': '0.5-2 μM'
            },
            'BDNF': {
                'type': 'Facteur neurotrophique',
                'target': 'Récepteurs TrkB',
                'effect': 'Promotion survie/croissance',
                'concentration': '10-100 ng/ml'
            }
        }
        
        for compound, info in compounds.items():
            with st.expander(f"💊 {compound}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {info['type']}")
                    st.write(f"**Cible:** {info['target']}")
                
                with col2:
                    st.write(f"**Effet:** {info['effect']}")
                    st.write(f"**Concentration:** {info['concentration']}")
    
    with tab2:
        st.subheader("🧪 Application Composé")
        
        if st.session_state.organoid_lab['organoids']:
            selected_organoid = st.selectbox("Organoïde",
                list(st.session_state.organoid_lab['organoids'].keys()),
                format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                key="pharm_org")
            
            compound = st.selectbox("Composé",
                list(compounds.keys()))
            
            concentration = st.number_input(
                f"Concentration (μM)", 
                0.1, 1000.0, 10.0, 0.1
            )
            
            application_duration = st.slider("Durée application (min)", 1, 60, 10)
            
            if st.button("💊 Appliquer", type="primary"):
                with st.spinner(f"Application {compound}..."):
                    import time
                    time.sleep(2)
                    
                    st.success(f"✅ {compound} appliqué à {concentration} μM")
                    
                    # Simuler effet
                    info = compounds[compound]
                    
                    if 'excitateur' in info['type'] or 'Glutamate' in compound:
                        st.info("📈 Effet: Augmentation activité neuronale observée")
                        activity_change = "+40%"
                    elif 'inhibiteur' in info['type'] or 'GABA' == compound:
                        st.info("📉 Effet: Diminution activité neuronale observée")
                        activity_change = "-35%"
                    elif 'TTX' in compound:
                        st.warning("🚫 Effet: Blocage complet potentiels action")
                        activity_change = "-95%"
                    else:
                        st.info("🔄 Effet: Modulation activité")
                        activity_change = "+15%"
                    
                    # Graphique avant/après
                    time_points = ['Baseline', 'Application', 'Washout']
                    
                    if activity_change.startswith('+'):
                        values = [100, 100 + float(activity_change[1:-1]), 105]
                    else:
                        values = [100, 100 + float(activity_change[:-1]), 95]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=time_points,
                        y=values,
                        mode='lines+markers',
                        line=dict(color='#FF6B9D', width=3),
                        marker=dict(size=12)
                    ))
                    
                    fig.update_layout(
                        title=f"Effet {compound}",
                        xaxis_title="Phase",
                        yaxis_title="Activité Relative (%)",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Sauvegarder
                    pharmacology = {
                        'organoid_id': selected_organoid,
                        'compound': compound,
                        'concentration_um': concentration,
                        'duration_min': application_duration,
                        'effect': activity_change,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.organoid_lab['pharmacology'].append(pharmacology)
                    log_event(f"Pharmacologie: {compound}", "SUCCESS")
        else:
            st.info("Créez un organoïde")
    
    with tab3:
        st.subheader("📊 Historique Pharmacologique")
        
        if st.session_state.organoid_lab['pharmacology']:
            st.write(f"### 💊 {len(st.session_state.organoid_lab['pharmacology'])} Applications")
            
            pharm_data = []
            for pharm in st.session_state.organoid_lab['pharmacology']:
                
                pharm_data.append({
                    'Composé': pharm['compound'],
                    'Concentration (μM)': pharm['concentration_um'],
                    'Durée (min)': pharm.get('duration_min'),
                    'Date': pharm['timestamp'][:19]
                })
            
            df_pharm = pd.DataFrame(pharm_data)
            st.dataframe(df_pharm, use_container_width=True)
            
            # Graphique fréquence composés
            compound_counts = df_pharm['Composé'].value_counts()
            
            fig = go.Figure(data=[go.Bar(
                x=compound_counts.index,
                y=compound_counts.values,
                marker_color='#FF6B9D'
            )])
            
            fig.update_layout(
                title="Fréquence d'Utilisation des Composés",
                xaxis_title="Composé",
                yaxis_title="Nombre d'applications",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune application pharmacologique enregistrée")

# ==================== PAGE: IMAGERIE ====================
elif page == "🔬 Imagerie":
    st.header("🔬 Imagerie Neuronale")
    
    st.info("""
    **Techniques d'Imagerie**
    
    Visualisation structure et activité neuronale.
    
    **Méthodes:**
    - Calcium Imaging (activité)
    - Voltage Imaging (potentiels)
    - Immunofluorescence (marqueurs)
    - Microscopie confocale (3D)
    """)
    
    tab1, tab2, tab3 = st.tabs(["📸 Calcium Imaging", "🔬 Immunofluorescence", "📊 Galerie"])
    
    with tab1:
        st.subheader("📸 Calcium Imaging")
        
        st.write("""
        **Imaging Calcium**
        
        Indicateurs fluorescents (GCaMP, Cal-520) pour visualiser activité neuronale.
        
        Ca²⁺ intracellulaire ↑ pendant potentiel action → Fluorescence ↑
        """)
        
        if st.session_state.organoid_lab['organoids']:
            selected_organoid = st.selectbox("Organoïde",
                list(st.session_state.organoid_lab['organoids'].keys()),
                format_func=lambda x: st.session_state.organoid_lab['organoids'][x]['name'],
                key="img_org")
            
            col1, col2 = st.columns(2)
            
            with col1:
                indicator = st.selectbox("Indicateur",
                    ["GCaMP6f", "GCaMP7", "Cal-520", "Fluo-4"])
                
                frame_rate = st.selectbox("Frame Rate",
                    ["10 Hz", "30 Hz", "100 Hz", "200 Hz"])
            
            with col2:
                recording_duration = st.slider("Durée (s)", 1, 60, 10)
                
                roi_count = st.slider("Nombre ROI", 10, 200, 50)
            
            if st.button("📸 Enregistrer Calcium", type="primary"):
                with st.spinner("Acquisition images..."):
                    import time
                    time.sleep(2)
                    
                    # Simuler données calcium
                    t = np.linspace(0, recording_duration, recording_duration * 10)
                    
                    # Générer traces calcium pour quelques neurones
                    n_neurons_show = min(10, roi_count)
                    
                    fig = go.Figure()
                    
                    for i in range(n_neurons_show):
                        # Activité spontanée avec spikes calcium
                        baseline = 100 + i * 20
                        activity = baseline + np.random.normal(0, 5, len(t))
                        
                        # Ajouter spikes calcium
                        n_spikes = np.random.randint(2, 8)
                        spike_times = np.random.uniform(0, recording_duration, n_spikes)
                        
                        for spike_t in spike_times:
                            spike_idx = np.abs(t - spike_t).argmin()
                            spike_profile = 50 * np.exp(-(t - spike_t)**2 / 0.5)
                            activity += spike_profile
                        
                        fig.add_trace(go.Scatter(
                            x=t, y=activity,
                            mode='lines',
                            name=f'Neurone {i+1}',
                            line=dict(width=1.5)
                        ))
                    
                    fig.update_layout(
                        title=f"Traces Calcium - {n_neurons_show} Neurones",
                        xaxis_title="Temps (s)",
                        yaxis_title="ΔF/F (%)",
                        template="plotly_dark",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Enregistrement terminé: {roi_count} ROI, {recording_duration}s")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ROI Actifs", f"{int(roi_count * 0.7)}/{roi_count}")
                    with col2:
                        st.metric("Transients Détectés", np.random.randint(100, 500))
                    with col3:
                        st.metric("Taux Moyen", f"{np.random.uniform(2, 8):.1f} Hz")
                    
                    # Sauvegarder
                    imaging = {
                        'type': 'Calcium',
                        'organoid_id': selected_organoid,
                        'indicator': indicator,
                        'duration_s': recording_duration,
                        'roi_count': roi_count,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.organoid_lab['imaging_sessions'].append(imaging)
                    log_event(f"Calcium imaging: {roi_count} ROI", "SUCCESS")
        else:
            st.info("Créez un organoïde")
    
    with tab2:
        st.subheader("🔬 Immunofluorescence")
        
        st.write("""
        **Marquage Immunofluorescent**
        
        Anticorps fluorescents pour identifier types cellulaires et structures.
        """)
        
        markers = st.multiselect("Marqueurs",
            ["MAP2 (neurones)", "GFAP (astrocytes)", "NeuN (noyaux neuronaux)",
             "Synapsin (synapses)", "DAPI (noyaux)", "TuJ1 (neurones immatures)"],
            default=["MAP2 (neurones)", "DAPI (noyaux)"])
        
        if st.button("🔬 Imager Marqueurs", type="primary"):
            if markers:
                st.success(f"✅ Imagerie {len(markers)} marqueurs")
                
                # Simuler image
                st.write("### 🖼️ Image Confocale Simulée")
                
                # Créer pseudo-image
                img_size = 256
                img = np.zeros((img_size, img_size, 3))
                
                # Ajouter "neurones" (points rouges)
                if "MAP2 (neurones)" in markers:
                    n_neurons = 50
                    for _ in range(n_neurons):
                        x, y = np.random.randint(0, img_size, 2)
                        img[max(0,x-3):min(img_size,x+3), max(0,y-3):min(img_size,y+3), 0] = 1
                
                # Ajouter noyaux (points bleus)
                if "DAPI (noyaux)" in markers:
                    n_nuclei = 100
                    for _ in range(n_nuclei):
                        x, y = np.random.randint(0, img_size, 2)
                        img[max(0,x-2):min(img_size,x+2), max(0,y-2):min(img_size,y+2), 2] = 1
                
                fig = go.Figure(data=go.Image(z=img))
                
                fig.update_layout(
                    title="Image Immunofluorescence (Simulée)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Marqueurs visibles:**")
                for marker in markers:
                    st.write(f"  • {marker}")
            else:
                st.warning("Sélectionnez au moins un marqueur")
    
    with tab3:
        st.subheader("📊 Galerie Images")
        
        if st.session_state.organoid_lab['imaging_sessions']:
            st.write(f"### 📸 {len(st.session_state.organoid_lab['imaging_sessions'])} Sessions d'Imagerie")
            
            for i, session in enumerate(st.session_state.organoid_lab['imaging_sessions'][::-1][:10]):
                with st.expander(f"📸 Session {len(st.session_state.organoid_lab['imaging_sessions'])-i} - {session['type']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {session['type']}")
                        if session['type'] == 'Calcium':
                            st.write(f"**Indicateur:** {session.get('indicator', 'N/A')}")
                            st.write(f"**ROI:** {session.get('roi_count', 0)}")
                    
                    with col2:
                        st.write(f"**Durée:** {session.get('duration_s', 0)} s")
                        st.write(f"**Date:** {session['timestamp'][:19]}")
        else:
            st.info("Aucune session d'imagerie enregistrée")

# ==================== PAGE: ANALYTICS ====================
elif page == "📊 Analytics":
    st.header("📊 Analytics & Tableaux de Bord")
    
    tab1, tab2, tab3 = st.tabs(["📈 Vue d'Ensemble", "🔬 Comparaisons", "📊 Statistiques"])
    
    with tab1:
        st.subheader("📈 Vue d'Ensemble du Lab")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Organoïdes", total_organoids)
            st.metric("Neurones Totaux", f"{total_neurons:,}")
        
        with col2:
            st.metric("Expériences", total_experiments)
            st.metric("Enregistrements", len(st.session_state.organoid_lab['recordings']))
        
        with col3:
            st.metric("Stimulations", len(st.session_state.organoid_lab['stimulations']))
            st.metric("Applications Pharm", len(st.session_state.organoid_lab['pharmacology']))
        
        with col4:
            st.metric("Sessions Imaging", len(st.session_state.organoid_lab['imaging_sessions']))
            st.metric("Entraînements", len(st.session_state.organoid_lab['training_sessions']))
        
        # Graphique activité
        st.write("### 📊 Répartition Activités")
        
        activities = {
            'Expériences': total_experiments,
            'Enregistrements': len(st.session_state.organoid_lab['recordings']),
            'Stimulations': len(st.session_state.organoid_lab['stimulations']),
            'Pharmacologie': len(st.session_state.organoid_lab['pharmacology']),
            'Imaging': len(st.session_state.organoid_lab['imaging_sessions']),
            'Entraînements': len(st.session_state.organoid_lab['training_sessions'])
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(activities.keys()),
            values=list(activities.values()),
            hole=0.4,
            marker=dict(colors=['#FF6B9D', '#C06C84', '#6C5B7B', '#355C7D', '#FF1493', '#9D50FF'])
        )])
        
        fig.update_layout(
            title="Distribution des Activités de Recherche",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Comparaisons Organoïdes")
        
        if len(st.session_state.organoid_lab['organoids']) > 1:
            # Comparer organoïdes
            comparison_data = []
            for org_id, org in st.session_state.organoid_lab['organoids'].items():
                comparison_data.append({
                    'Nom': org['name'],
                    'Taille (mm)': org['size_mm'],
                    'Neurones': org['neuron_count'],
                    'Viabilité (%)': org['viability'],
                    'Âge (jours)': org['culture_duration_days']
                })
            
            df_comp = pd.DataFrame(comparison_data)
            
            # Graphique comparaison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Taille", "Nombre Neurones", "Viabilité", "Âge")
            )
            
            fig.add_trace(go.Bar(x=df_comp['Nom'], y=df_comp['Taille (mm)'],
                                marker_color='#FF6B9D'), row=1, col=1)
            fig.add_trace(go.Bar(x=df_comp['Nom'], y=df_comp['Neurones'],
                                marker_color='#C06C84'), row=1, col=2)
            fig.add_trace(go.Bar(x=df_comp['Nom'], y=df_comp['Viabilité (%)'],
                                marker_color='#6C5B7B'), row=2, col=1)
            fig.add_trace(go.Bar(x=df_comp['Nom'], y=df_comp['Âge (jours)'],
                                marker_color='#355C7D'), row=2, col=2)
            
            fig.update_layout(
                title="Comparaison Organoïdes",
                template="plotly_dark",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.info("Créez au moins 2 organoïdes pour comparer")
    
    with tab3:
        st.subheader("📊 Statistiques Détaillées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🧠 Neurones")
            if total_neurons > 0:
                st.metric("Total", f"{total_neurons:,}")
                st.metric("Synapses", f"{(total_neurons * 10000)/1e9:.2f}B")
                st.metric("Moy/Organoïde", f"{total_neurons/total_organoids:,.0f}" if total_organoids > 0 else "0")
        
        with col2:
            st.write("### 📊 Activité")
            total_activities = sum(activities.values())
            st.metric("Total Activités", total_activities)
            
            if total_activities > 0:
                most_frequent = max(activities, key=activities.get)
                st.metric("Plus Fréquent", most_frequent)

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Configuration Laboratoire")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Général", "💾 Données", "🔄 Reset"])
    
    with tab1:
        st.subheader("🔧 Paramètres Généraux")
        
        with st.form("settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🧪 Culture")
                default_temp = st.number_input("Température par défaut (°C)", 35, 39, 37)
                default_o2 = st.slider("O₂ par défaut (%)", 5, 21, 20)
                auto_media_change = st.checkbox("Changement milieu automatique", value=True)
            
            with col2:
                st.write("### 📊 Enregistrements")
                default_sampling = st.selectbox("Fréquence échantillonnage",
                    ["10 kHz", "20 kHz", "50 kHz"], index=1)
                
                auto_save = st.checkbox("Sauvegarde automatique", value=True)
                save_interval = st.slider("Intervalle sauvegarde (min)", 1, 60, 15)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🖥️ Interface")
                theme = st.selectbox("Thème", ["Dark", "Light", "Auto"], index=0)
                show_animations = st.checkbox("Animations", value=True)
            
            with col2:
                st.write("### 🔔 Notifications")
                notify_experiments = st.checkbox("Fin expériences", value=True)
                notify_viability = st.checkbox("Alerte viabilité < 80%", value=True)
            
            if st.form_submit_button("💾 Sauvegarder Paramètres", type="primary"):
                st.success("✅ Paramètres sauvegardés!")
                log_event("Paramètres mis à jour", "INFO")
    
    with tab2:
        st.subheader("💾 Gestion des Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📥 Export")
            
            export_format = st.selectbox("Format",
                ["JSON", "CSV", "Excel", "HDF5"])
            
            if st.button("📥 Exporter Tout", type="primary"):
                # Préparer données export
                export_data = {
                    'organoids': len(st.session_state.organoid_lab['organoids']),
                    'total_neurons': total_neurons,
                    'experiments': len(st.session_state.organoid_lab['experiments']),
                    'recordings': len(st.session_state.organoid_lab['recordings']),
                    'export_date': datetime.now().isoformat()
                }
                
                st.success("✅ Données exportées!")
                st.json(export_data)
                
                log_event(f"Export données: {export_format}", "SUCCESS")
        
        with col2:
            st.write("### 📊 Statistiques Stockage")
            
            # Calculer taille approximative
            total_items = (len(st.session_state.organoid_lab['organoids']) +
                          len(st.session_state.organoid_lab['experiments']) +
                          len(st.session_state.organoid_lab['recordings']) +
                          len(st.session_state.organoid_lab['stimulations']) +
                          len(st.session_state.organoid_lab['pharmacology']) +
                          len(st.session_state.organoid_lab['imaging_sessions']) +
                          len(st.session_state.organoid_lab['training_sessions']))
            
            st.metric("Objets Total", total_items)
            st.metric("Événements Log", len(st.session_state.organoid_lab['log']))
            
            estimated_size = total_items * 2  # KB approximatif
            st.metric("Taille Estimée", f"{estimated_size} KB")
    
    with tab3:
        st.subheader("🔄 Réinitialisation")
        
        st.warning("⚠️ **Actions Irréversibles!**")
        st.write("Les données supprimées ne peuvent pas être récupérées.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Supprimer Organoïdes", key="del_org"):
                if st.checkbox("Confirmer suppression organoïdes", key="conf_org"):
                    st.session_state.organoid_lab['organoids'] = {}
                    st.success("✅ Organoïdes supprimés")
                    log_event("Organoïdes supprimés", "WARNING")
                    st.rerun()
            
            if st.button("🗑️ Supprimer Expériences", key="del_exp"):
                if st.checkbox("Confirmer suppression expériences", key="conf_exp"):
                    st.session_state.organoid_lab['experiments'] = []
                    st.success("✅ Expériences supprimées")
                    log_event("Expériences supprimées", "WARNING")
                    st.rerun()
            
            if st.button("🗑️ Supprimer Enregistrements", key="del_rec"):
                if st.checkbox("Confirmer suppression enregistrements", key="conf_rec"):
                    st.session_state.organoid_lab['recordings'] = []
                    st.session_state.organoid_lab['electrophysiology'] = []
                    st.success("✅ Enregistrements supprimés")
                    log_event("Enregistrements supprimés", "WARNING")
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Supprimer Pharmacologie", key="del_pharm"):
                if st.checkbox("Confirmer suppression pharmacologie", key="conf_pharm"):
                    st.session_state.organoid_lab['pharmacology'] = []
                    st.success("✅ Données pharmacologie supprimées")
                    log_event("Pharmacologie supprimée", "WARNING")
                    st.rerun()
            
            if st.button("🗑️ Supprimer Imagerie", key="del_img"):
                if st.checkbox("Confirmer suppression imagerie", key="conf_img"):
                    st.session_state.organoid_lab['imaging_sessions'] = []
                    st.success("✅ Sessions imagerie supprimées")
                    log_event("Imagerie supprimée", "WARNING")
                    st.rerun()
            
            if st.button("🗑️ Effacer Log", key="del_log"):
                if st.checkbox("Confirmer effacement log", key="conf_log"):
                    st.session_state.organoid_lab['log'] = []
                    st.success("✅ Log effacé")
                    st.rerun()
        
        st.markdown("---")
        
        st.error("### ⚠️ DANGER ZONE")
        
        if st.button("🔴 RÉINITIALISER TOUT LE LABORATOIRE", key="reset_all"):
            confirm_text = st.text_input("Tapez 'RESET' pour confirmer", key="reset_confirm")
            
            if confirm_text == "RESET":
                st.session_state.organoid_lab = {
                    'organoids': {},
                    'neurons': {},
                    'synapses': {},
                    'neural_networks': {},
                    'experiments': [],
                    'recordings': [],
                    'stimulations': [],
                    'training_sessions': [],
                    'computations': [],
                    'culture_media': {},
                    'growth_factors': {},
                    'pharmacology': [],
                    'electrophysiology': [],
                    'imaging_sessions': [],
                    'log': []
                }
                
                st.success("✅ Laboratoire réinitialisé!")
                st.balloons()
                log_event("Réinitialisation complète du laboratoire", "WARNING")
                st.rerun()
            elif confirm_text and confirm_text != "RESET":
                st.error("❌ Texte incorrect. Réinitialisation annulée.")

# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal Laboratoire (20 dernières entrées)"):
    if st.session_state.organoid_lab['log']:
        for event in st.session_state.organoid_lab['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️" if level == "WARNING" else "❌"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🧠 Brain Organoid Computing Platform</h3>
        <p>Biocomputing • Organoïdes Cérébraux • Intelligence Biologique</p>
        <p><small>Neurosciences • Biotechnologie • Computing Neuronal</small></p>
        <p><small>Version 1.0.0 | Research Edition</small></p>
        <p><small>🧠 Exploring the Future of Biological Computing © 2024</small></p>
    </div>
""", unsafe_allow_html=True)
