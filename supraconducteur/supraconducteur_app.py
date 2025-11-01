"""
Interface Streamlit pour la Plateforme Supraconducteur-Magnétique-IA
Système complet pour créer, développer, fabriquer, tester et déployer
des supraconducteurs, systèmes magnétiques, lévitation et amplificateurs
streamlit run supraconducteur_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import numpy as np

# ==================== CONFIGURATION PAGE ====================

st.set_page_config(
    page_title="🧲 Plateforme Supraconducteur-Magnétique-IA",
    page_icon="🧲",
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
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .system-card {
        border: 3px solid #4facfe;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.3);
    }
    .temp-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.3rem;
    }
    .ultra-cold {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
    }
    .cryogenic {
        background: linear-gradient(90deg, #0093E9 0%, #80D0C7 100%);
        color: white;
    }
    .room-temp {
        background: linear-gradient(90deg, #FBAB7E 0%, #F7CE68 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION SESSION STATE ====================

if 'superconductor_system' not in st.session_state:
    st.session_state.superconductor_system = {
        'systems': {},
        'fabrications': [],
        'tests': [],
        'deployments': {},
        'projects': {},
        'experiments': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def create_system_mock(name, system_type, config):
    """Crée un système supraconducteur simulé"""
    system_id = f"sys_{len(st.session_state.superconductor_system['systems']) + 1}"
    
    # Paramètres critiques selon le matériau
    material_params = {
        'ybco': {'Tc': 92, 'Jc': 1e10, 'Bc': 100},
        'bscco': {'Tc': 110, 'Jc': 5e9, 'Bc': 50},
        'nbti': {'Tc': 9.2, 'Jc': 3e9, 'Bc': 15},
        'nb3sn': {'Tc': 18.3, 'Jc': 5e9, 'Bc': 30},
        'mgdb2': {'Tc': 39, 'Jc': 1e10, 'Bc': 40}
    }
    
    material = config.get('material', 'ybco')
    params = material_params.get(material, material_params['ybco'])
    
    system = {
        'id': system_id,
        'name': name,
        'type': system_type,
        'created_at': datetime.now().isoformat(),
        'status': 'offline',
        'health': 1.0,
        'material': material,
        'critical_temperature': params['Tc'],
        'critical_current': params['Jc'],
        'critical_field': params['Bc'],
        'cooling': {
            'system': config.get('cooling_system', 'azote_liquide'),
            'temperature': config.get('temperature', 77.0),
            'efficiency': np.random.random() * 0.3 + 0.7
        },
        'magnetic_properties': {
            'field_strength': np.random.random() * 20,
            'field_uniformity': np.random.random() * 0.1 + 0.9,
            'field_stability': np.random.random() * 0.1 + 0.9
        },
        'performance': {
            'efficiency': config.get('efficiency', 0.85),
            'reliability': 0.95,
            'stability': 0.9
        },
        'operational_hours': 0.0
    }
    
    # Système de lévitation
    if system_type in ['levitation_magnetique', 'supraconducteur_hybride']:
        system['levitation'] = {
            'type': config.get('levitation_type', 'meissner'),
            'load_capacity': config.get('load_capacity', 100.0),
            'levitation_height': 0.0,
            'stability': 0.95
        }
    
    # Système amplificateur
    if system_type in ['amplificateur', 'supraconducteur_hybride']:
        system['amplifier'] = {
            'type': config.get('amplifier_type', 'puissance'),
            'gain': config.get('gain', 40.0),
            'bandwidth': config.get('bandwidth', 1e9),
            'noise_figure': np.random.random() * 2 + 1
        }
    
    # Système quantique
    if system_type in ['supraconducteur_quantique', 'supraconducteur_ia']:
        system['quantum'] = {
            'qubits': config.get('qubits', 100),
            'coherence_time': np.random.random() * 100 + 50,
            'gate_fidelity': 0.99
        }
    
    # Système biologique
    if system_type in ['supraconducteur_biologique', 'supraconducteur_ia']:
        system['biological'] = {
            'bio_interface': True,
            'biocompatibility': np.random.random() * 0.2 + 0.8,
            'self_healing': np.random.random() * 0.3 + 0.6
        }
    
    # Système IA
    if system_type == 'supraconducteur_ia':
        system['ai'] = {
            'enabled': True,
            'intelligence_level': config.get('ai_level', 0.7),
            'autonomous_control': config.get('autonomous', False)
        }
    
    st.session_state.superconductor_system['systems'][system_id] = system
    log_event(f"Système créé: {name} ({system_type})")
    return system_id

def log_event(message: str):
    """Enregistre un événement"""
    st.session_state.superconductor_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })

def get_temp_badge(temp: float) -> str:
    """Retourne un badge HTML pour la température"""
    if temp < 20:
        return '<span class="temp-badge ultra-cold">❄️ ULTRA-FROID</span>'
    elif temp < 100:
        return '<span class="temp-badge cryogenic">🧊 CRYOGÉNIQUE</span>'
    else:
        return '<span class="temp-badge room-temp">🌡️ TEMPÉRATURE AMBIANTE</span>'

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">🧲 Plateforme Supraconducteur-Magnétique-IA</h1>', unsafe_allow_html=True)
st.markdown("### Système complet pour supraconducteurs, lévitation magnétique et amplificateurs avec IA-Quantique-Biologique")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/4facfe/ffffff?text=SuperMag+Lab", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Tableau de Bord",
            "🧲 Mes Systèmes",
            "➕ Créer Système",
            "🏭 Fabrication",
            "🔧 Configuration",
            "🧪 Tests & Certification",
            "📊 Propriétés Magnétiques",
            "🚁 Lévitation Magnétique",
            "📡 Amplificateurs",
            "🚀 Déploiement",
            "📁 Projets",
            "❄️ Cryogénie",
            "📚 Bibliothèque",
            "⚙️ Maintenance"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_systems = len(st.session_state.superconductor_system['systems'])
    active_systems = sum(1 for s in st.session_state.superconductor_system['systems'].values() if s['status'] == 'online')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🧲 Systèmes", total_systems)
        st.metric("📁 Projets", len(st.session_state.superconductor_system['projects']))
    with col2:
        st.metric("✅ Actifs", active_systems)
        st.metric("🧪 Tests", len(st.session_state.superconductor_system['tests']))

# ==================== PAGE: TABLEAU DE BORD ====================

if page == "🏠 Tableau de Bord":
    st.header("📊 Tableau de Bord Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="system-card"><h2>🧲</h2><h3>{}</h3><p>Systèmes Totaux</p></div>'.format(total_systems), unsafe_allow_html=True)
    
    with col2:
        if st.session_state.superconductor_system['systems']:
            avg_tc = np.mean([s['critical_temperature'] for s in st.session_state.superconductor_system['systems'].values()])
            st.markdown('<div class="system-card"><h2>❄️</h2><h3>{:.1f}K</h3><p>Tc Moyen</p></div>'.format(avg_tc), unsafe_allow_html=True)
        else:
            st.markdown('<div class="system-card"><h2>❄️</h2><h3>N/A</h3><p>Tc Moyen</p></div>', unsafe_allow_html=True)
    
    with col3:
        levitation_systems = sum(1 for s in st.session_state.superconductor_system['systems'].values() if 'levitation' in s)
        st.markdown('<div class="system-card"><h2>🚁</h2><h3>{}</h3><p>Lévitation</p></div>'.format(levitation_systems), unsafe_allow_html=True)
    
    with col4:
        amplifier_systems = sum(1 for s in st.session_state.superconductor_system['systems'].values() if 'amplifier' in s)
        st.markdown('<div class="system-card"><h2>📡</h2><h3>{}</h3><p>Amplificateurs</p></div>'.format(amplifier_systems), unsafe_allow_html=True)
    
    with col5:
        if st.session_state.superconductor_system['systems']:
            avg_field = np.mean([s['magnetic_properties']['field_strength'] for s in st.session_state.superconductor_system['systems'].values()])
            st.markdown('<div class="system-card"><h2>🧲</h2><h3>{:.1f}T</h3><p>Champ Moyen</p></div>'.format(avg_field), unsafe_allow_html=True)
        else:
            st.markdown('<div class="system-card"><h2>🧲</h2><h3>0T</h3><p>Champ Moyen</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques
    if st.session_state.superconductor_system['systems']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribution par Type")
            
            type_counts = {}
            for s in st.session_state.superconductor_system['systems'].values():
                s_type = s['type'].replace('_', ' ').title()
                type_counts[s_type] = type_counts.get(s_type, 0) + 1
            
            fig = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(title="Répartition des Systèmes")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("❄️ Températures Critiques")
            
            names = [s['name'][:15] for s in st.session_state.superconductor_system['systems'].values()]
            temps = [s['critical_temperature'] for s in st.session_state.superconductor_system['systems'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=temps, marker_color='rgb(79, 172, 254)')
            ])
            fig.update_layout(title="Tc par Système", yaxis_title="Température (K)", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Aucun système créé. Créez votre premier supraconducteur!")

# ==================== PAGE: MES SYSTÈMES ====================

elif page == "🧲 Mes Systèmes":
    st.header("🧲 Gestion des Systèmes")
    
    if not st.session_state.superconductor_system['systems']:
        st.info("💡 Aucun système créé. Créez votre premier système!")
    else:
        for sys_id, sys in st.session_state.superconductor_system['systems'].items():
            st.markdown(f'<div class="system-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### 🧲 {sys['name']}")
                st.markdown(get_temp_badge(sys['critical_temperature']), unsafe_allow_html=True)
                st.caption(f"Type: {sys['type'].replace('_', ' ').title()}")
            
            with col2:
                st.metric("Tc", f"{sys['critical_temperature']:.1f} K")
                st.metric("Jc", f"{sys['critical_current']:.2e} A/m²")
            
            with col3:
                st.metric("Bc", f"{sys['critical_field']:.1f} T")
                st.metric("Efficacité", f"{sys['performance']['efficiency']:.0%}")
            
            with col4:
                status_icon = "🟢" if sys['status'] == 'online' else "🔴"
                st.write(f"**Statut:** {status_icon} {sys['status'].upper()}")
                st.write(f"**Santé:** {sys['health']:.0%}")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4 = st.tabs(["🧲 Magnétique", "❄️ Refroidissement", "🚁 Lévitation", "📡 Amplificateur"])
                
                with tab1:
                    st.subheader("Propriétés Magnétiques")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Intensité Champ", f"{sys['magnetic_properties']['field_strength']:.2f} T")
                    with col2:
                        st.metric("Uniformité", f"{sys['magnetic_properties']['field_uniformity']:.0%}")
                    with col3:
                        st.metric("Stabilité", f"{sys['magnetic_properties']['field_stability']:.0%}")
                
                with tab2:
                    st.subheader("Système de Refroidissement")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Système:** {sys['cooling']['system'].replace('_', ' ').title()}")
                        st.write(f"**Température:** {sys['cooling']['temperature']:.1f} K")
                    with col2:
                        st.write(f"**Efficacité:** {sys['cooling']['efficiency']:.0%}")
                        st.progress(sys['cooling']['efficiency'])
                
                with tab3:
                    if 'levitation' in sys:
                        st.subheader("Système de Lévitation")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Capacité Charge", f"{sys['levitation']['load_capacity']:.0f} kg")
                        with col2:
                            st.metric("Hauteur", f"{sys['levitation']['levitation_height']:.1f} mm")
                        with col3:
                            st.metric("Stabilité", f"{sys['levitation']['stability']:.0%}")
                    else:
                        st.info("Pas de système de lévitation")
                
                with tab4:
                    if 'amplifier' in sys:
                        st.subheader("Système Amplificateur")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Gain", f"{sys['amplifier']['gain']:.1f} dB")
                        with col2:
                            st.metric("Bande Passante", f"{sys['amplifier']['bandwidth']/1e9:.2f} GHz")
                        with col3:
                            st.metric("Figure de Bruit", f"{sys['amplifier']['noise_figure']:.2f} dB")
                    else:
                        st.info("Pas d'amplificateur")
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button(f"▶️ {'Éteindre' if sys['status'] == 'online' else 'Activer'}", key=f"toggle_{sys_id}"):
                        sys['status'] = 'offline' if sys['status'] == 'online' else 'online'
                        log_event(f"{sys['name']} {'éteint' if sys['status'] == 'offline' else 'activé'}")
                        st.rerun()
                
                with col2:
                    if st.button(f"🔧 Optimiser", key=f"opt_{sys_id}"):
                        sys['performance']['efficiency'] = min(0.99, sys['performance']['efficiency'] * 1.05)
                        sys['magnetic_properties']['field_uniformity'] = min(0.99, sys['magnetic_properties']['field_uniformity'] * 1.02)
                        st.success("Optimisation appliquée!")
                        st.rerun()
                
                with col3:
                    if st.button(f"🧪 Tester", key=f"test_{sys_id}"):
                        st.info("Allez dans Tests & Certification")
                
                with col4:
                    if st.button(f"🔬 Diagnostiquer", key=f"diag_{sys_id}"):
                        if sys['health'] < 0.95:
                            st.warning(f"⚠️ Santé: {sys['health']:.0%}")
                        else:
                            st.success("✅ Système en bon état")
                
                with col5:
                    if st.button(f"🗑️ Supprimer", key=f"del_{sys_id}"):
                        del st.session_state.superconductor_system['systems'][sys_id]
                        log_event(f"{sys['name']} supprimé")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CRÉER SYSTÈME ====================

elif page == "➕ Créer Système":
    st.header("➕ Créer un Nouveau Système")
    
    with st.form("create_system_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            system_name = st.text_input("📝 Nom du Système", placeholder="Ex: SuperMag-Alpha-1")
            system_type = st.selectbox(
                "🧬 Type de Système",
                [
                    "supraconducteur",
                    "systeme_magnetique",
                    "levitation_magnetique",
                    "amplificateur",
                    "supraconducteur_hybride",
                    "supraconducteur_quantique",
                    "supraconducteur_biologique",
                    "supraconducteur_ia"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            material = st.selectbox(
                "🔬 Matériau Supraconducteur",
                ["ybco", "bscco", "nbti", "nb3sn", "mgdb2"],
                format_func=lambda x: x.upper()
            )
            
            target_efficiency = st.slider("⚙️ Efficacité Cible", 0.7, 0.99, 0.85, 0.01)
        
        st.markdown("---")
        st.subheader("❄️ Système de Refroidissement")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cooling_system = st.selectbox(
                "Système de Refroidissement",
                ["azote_liquide", "helium_liquide", "cryorefroidisseur", "refrigerateur_dilution"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            target_temperature = st.number_input("Température Opérationnelle (K)", 4.0, 300.0, 77.0, 1.0)
        
        st.markdown("---")
        
        # Configuration spécifique selon le type
        if system_type in ['levitation_magnetique', 'supraconducteur_hybride']:
            st.subheader("🚁 Configuration Lévitation")
            col1, col2 = st.columns(2)
            
            with col1:
                levitation_type = st.selectbox("Type Lévitation", ["meissner", "ancrage_flux", "verrouillage_quantique"])
                load_capacity = st.number_input("Capacité de Charge (kg)", 1.0, 10000.0, 100.0)
            
            with col2:
                st.info(f"Type: {levitation_type.replace('_', ' ').title()}")
        
        if system_type in ['amplificateur', 'supraconducteur_hybride']:
            st.subheader("📡 Configuration Amplificateur")
            col1, col2 = st.columns(2)
            
            with col1:
                amplifier_type = st.selectbox("Type Amplificateur", ["puissance", "tension", "courant", "signal"])
                gain = st.number_input("Gain (dB)", 10.0, 100.0, 40.0)
            
            with col2:
                bandwidth = st.number_input("Bande Passante (GHz)", 0.1, 100.0, 1.0) * 1e9
        
        if system_type in ['supraconducteur_quantique', 'supraconducteur_ia']:
            st.subheader("⚛️ Configuration Quantique")
            qubits = st.number_input("Nombre de Qubits", 10, 1000, 100)
        
        if system_type in ['supraconducteur_biologique', 'supraconducteur_ia']:
            st.subheader("🧬 Configuration Biologique")
            st.checkbox("Interface Biologique", value=True)
        
        if system_type == 'supraconducteur_ia':
            st.subheader("🤖 Configuration IA")
            col1, col2 = st.columns(2)
            with col1:
                ai_level = st.slider("Niveau Intelligence IA", 0.0, 1.0, 0.7, 0.1)
            with col2:
                autonomous = st.checkbox("Contrôle Autonome")
        
        submitted = st.form_submit_button("🚀 Créer le Système", use_container_width=True, type="primary")
        
        if submitted:
            if not system_name:
                st.error("⚠️ Veuillez donner un nom au système")
            else:
                with st.spinner("🔄 Création du système en cours..."):
                    config = {
                        'material': material,
                        'cooling_system': cooling_system,
                        'temperature': target_temperature,
                        'efficiency': target_efficiency
                    }
                    
                    if system_type in ['levitation_magnetique', 'supraconducteur_hybride']:
                        config['levitation_type'] = levitation_type
                        config['load_capacity'] = load_capacity
                    
                    if system_type in ['amplificateur', 'supraconducteur_hybride']:
                        config['amplifier_type'] = amplifier_type
                        config['gain'] = gain
                        config['bandwidth'] = bandwidth
                    
                    if system_type in ['supraconducteur_quantique', 'supraconducteur_ia']:
                        config['qubits'] = qubits
                    
                    if system_type == 'supraconducteur_ia':
                        config['ai_level'] = ai_level
                        config['autonomous'] = autonomous
                    
                    sys_id = create_system_mock(system_name, system_type, config)
                    
                    st.success(f"✅ Système '{system_name}' créé avec succès!")
                    st.balloons()
                    
                    sys = st.session_state.superconductor_system['systems'][sys_id]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tc", f"{sys['critical_temperature']:.1f} K")
                    with col2:
                        st.metric("Jc", f"{sys['critical_current']:.2e} A/m²")
                    with col3:
                        st.metric("Bc", f"{sys['critical_field']:.1f} T")
                    with col4:
                        st.metric("Efficacité", f"{sys['performance']['efficiency']:.0%}")
                    
                    st.code(f"ID: {sys_id}", language="text")

# ==================== PAGE: FABRICATION ====================

elif page == "🏭 Fabrication":
    st.header("🏭 Chaîne de Fabrication")
    
    if not st.session_state.superconductor_system['systems']:
        st.warning("⚠️ Aucun système disponible pour fabrication")
    else:
        tab1, tab2, tab3 = st.tabs(["🏗️ Nouvelle Fabrication", "📊 En Cours", "📜 Historique"])
        
        with tab1:
            st.subheader("🏗️ Planifier une Fabrication")
            
            sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
            selected_sys = st.selectbox(
                "Sélectionner le système à fabriquer",
                options=list(sys_options.keys()),
                format_func=lambda x: sys_options[x]
            )
            
            sys = st.session_state.superconductor_system['systems'][selected_sys]
            
            st.write(f"### 🧲 {sys['name']}")
            st.write(f"**Matériau:** {sys['material'].upper()}")
            st.write(f"**Tc:** {sys['critical_temperature']:.1f} K")
            
            st.markdown("---")
            st.subheader("📋 Phases de Fabrication")
            
            phases = [
                {
                    'phase': 1,
                    'name': 'Préparation Matériaux',
                    'duration': 30,
                    'cost': 500000,
                    'steps': ['Purification', 'Synthèse', 'Caractérisation', 'Tests pureté']
                },
                {
                    'phase': 2,
                    'name': 'Fabrication Supraconducteur',
                    'duration': 45,
                    'cost': 1000000,
                    'steps': ['Dépôt couches minces', 'Traitement thermique', 'Structuration']
                },
                {
                    'phase': 3,
                    'name': 'Système Magnétique',
                    'duration': 30,
                    'cost': 800000,
                    'steps': ['Conception bobines', 'Assemblage', 'Tests champ', 'Calibration']
                },
                {
                    'phase': 4,
                    'name': 'Refroidissement',
                    'duration': 20,
                    'cost': 600000,
                    'steps': ['Installation cryostat', 'Tests refroidissement', 'Optimisation']
                },
                {
                    'phase': 5,
                    'name': 'Tests Finaux',
                    'duration': 25,
                    'cost': 300000,
                    'steps': ['Tests performance', 'Tests sécurité', 'Certification']
                }
            ]
            
            for phase in phases:
                with st.expander(f"Phase {phase['phase']}: {phase['name']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Durée:** {phase['duration']} jours")
                        st.write(f"**Coût:** ${phase['cost']:,}")
                    with col2:
                        st.write("**Étapes:**")
                        for step in phase['steps']:
                            st.write(f"• {step}")
            
            total_duration = sum(p['duration'] for p in phases)
            total_cost = sum(p['cost'] for p in phases)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Durée Totale", f"{total_duration} jours ({total_duration/30:.1f} mois)")
            with col2:
                st.metric("Coût Total", f"${total_cost:,}")
            
            if st.button("🚀 Lancer la Fabrication", use_container_width=True, type="primary"):
                fab_id = f"fab_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                fabrication = {
                    'fabrication_id': fab_id,
                    'system_id': selected_sys,
                    'system_name': sys['name'],
                    'phases': phases,
                    'current_phase': 0,
                    'progress': 0.0,
                    'status': 'in_progress',
                    'start_date': datetime.now().isoformat(),
                    'estimated_completion': (datetime.now() + timedelta(days=total_duration)).isoformat()
                }
                
                st.session_state.superconductor_system['fabrications'].append(fabrication)
                log_event(f"Fabrication démarrée: {sys['name']}")
                
                st.success("✅ Fabrication démarrée!")
                st.balloons()
        
        with tab2:
            st.subheader("📊 Fabrications en Cours")
            
            in_progress = [f for f in st.session_state.superconductor_system['fabrications'] if f['status'] == 'in_progress']
            
            if not in_progress:
                st.info("Aucune fabrication en cours")
            else:
                for fab in in_progress:
                    with st.expander(f"🏭 {fab['system_name']} - {fab['progress']:.0f}%"):
                        st.progress(fab['progress'] / 100)
                        
                        st.write(f"**Démarrage:** {fab['start_date'][:10]}")
                        st.write(f"**Fin estimée:** {fab['estimated_completion'][:10]}")
                        
                        if st.button(f"⏩ Avancer Phase", key=f"adv_{fab['fabrication_id']}"):
                            if fab['current_phase'] < len(fab['phases']):
                                fab['current_phase'] += 1
                                fab['progress'] = (fab['current_phase'] / len(fab['phases'])) * 100
                                
                                if fab['current_phase'] >= len(fab['phases']):
                                    fab['status'] = 'completed'
                                    st.success("🎉 Fabrication terminée!")
                                
                                st.rerun()
        
        with tab3:
            st.subheader("📜 Historique")
            
            if st.session_state.superconductor_system['fabrications']:
                fab_data = []
                for fab in st.session_state.superconductor_system['fabrications']:
                    fab_data.append({
                        'Système': fab['system_name'],
                        'Démarrage': fab['start_date'][:10],
                        'Statut': fab['status'].upper(),
                        'Progression': f"{fab['progress']:.0f}%"
                    })
                
                df = pd.DataFrame(fab_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucun historique")

# ==================== PAGE: TESTS & CERTIFICATION ====================

elif page == "🧪 Tests & Certification":
    st.header("🧪 Tests et Certification")
    
    if not st.session_state.superconductor_system['systems']:
        st.warning("⚠️ Aucun système disponible")
    else:
        sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
        selected_sys = st.selectbox(
            "Sélectionner un système",
            options=list(sys_options.keys()),
            format_func=lambda x: sys_options[x]
        )
        
        sys = st.session_state.superconductor_system['systems'][selected_sys]
        
        tab1, tab2, tab3 = st.tabs(["🧪 Tests Standards", "🔬 Tests Avancés", "📊 Historique"])
        
        with tab1:
            st.subheader("🧪 Suite de Tests Standards")
            
            tests = {
                "Test Température Critique": "Mesure précise de Tc",
                "Test Courant Critique": "Mesure de Jc",
                "Test Champ Critique": "Mesure de Bc",
                "Test Stabilité Magnétique": "Uniformité et stabilité du champ",
                "Test Efficacité": "Performance globale"
            }
            
            for test_name, description in tests.items():
                st.write(f"**{test_name}:** {description}")
            
            if st.button("🚀 Lancer Tous les Tests", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                
                results = {}
                
                # Test Tc
                progress_bar.progress(0.2)
                measured_tc = sys['critical_temperature'] * (0.98 + np.random.random() * 0.04)
                results['Tc'] = {
                    'expected': sys['critical_temperature'],
                    'measured': measured_tc,
                    'passed': abs(measured_tc - sys['critical_temperature']) < sys['critical_temperature'] * 0.05
                }
                
                # Test Jc
                progress_bar.progress(0.4)
                measured_jc = sys['critical_current'] * (0.95 + np.random.random() * 0.1)
                results['Jc'] = {
                    'expected': sys['critical_current'],
                    'measured': measured_jc,
                    'passed': abs(measured_jc - sys['critical_current']) < sys['critical_current'] * 0.1
                }
                
                # Test Bc
                progress_bar.progress(0.6)
                measured_bc = sys['critical_field'] * (0.98 + np.random.random() * 0.04)
                results['Bc'] = {
                    'expected': sys['critical_field'],
                    'measured': measured_bc,
                    'passed': abs(measured_bc - sys['critical_field']) < sys['critical_field'] * 0.05
                }
                
                # Test stabilité
                progress_bar.progress(0.8)
                stability = sys['magnetic_properties']['field_stability']
                results['Stability'] = {
                    'value': stability,
                    'passed': stability > 0.9
                }
                
                # Test efficacité
                progress_bar.progress(1.0)
                efficiency = sys['performance']['efficiency']
                results['Efficiency'] = {
                    'value': efficiency,
                    'passed': efficiency > 0.8
                }
                
                progress_bar.empty()
                
                st.success("✅ Tests terminés!")
                
                # Résultats
                passed = sum(1 for r in results.values() if r.get('passed', False))
                total = len(results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tests Réussis", f"{passed}/{total}")
                with col2:
                    st.metric("Score", f"{(passed/total)*100:.0f}%")
                with col3:
                    grade = 'A' if passed/total >= 0.9 else 'B' if passed/total >= 0.8 else 'C'
                    st.metric("Note", grade)
                
                # Détails
                st.markdown("---")
                for test_name, result in results.items():
                    status = "✅" if result.get('passed', False) else "❌"
                    st.write(f"{status} **{test_name}**")
                    if 'measured' in result:
                        st.write(f"  Attendu: {result['expected']:.2e}, Mesuré: {result['measured']:.2e}")
                    elif 'value' in result:
                        st.write(f"  Valeur: {result['value']:.2%}")
                
                # Sauvegarder
                test_record = {
                    'system_id': selected_sys,
                    'system_name': sys['name'],
                    'timestamp': datetime.now().isoformat(),
                    'results': results,
                    'score': passed/total,
                    'grade': grade
                }
                
                st.session_state.superconductor_system['tests'].append(test_record)
                log_event(f"Tests complétés: {sys['name']} - Note {grade}")
        
        with tab2:
            st.subheader("🔬 Tests Avancés")
            
            advanced_tests = st.multiselect(
                "Sélectionner tests avancés",
                [
                    "Test Lévitation (si applicable)",
                    "Test Amplificateur (si applicable)",
                    "Test Quantique (si applicable)",
                    "Test Bio-interface (si applicable)",
                    "Test Longue Durée",
                    "Test Cyclage Thermique",
                    "Test Vibrations"
                ]
            )
            
            if advanced_tests and st.button("🚀 Lancer Tests Avancés"):
                st.success(f"✅ {len(advanced_tests)} test(s) lancé(s)!")
                
                for test in advanced_tests:
                    st.write(f"• {test}: En cours...")
        
        with tab3:
            st.subheader("📊 Historique des Tests")
            
            if st.session_state.superconductor_system['tests']:
                test_data = []
                for test in st.session_state.superconductor_system['tests']:
                    test_data.append({
                        'Système': test['system_name'],
                        'Date': test['timestamp'][:10],
                        'Score': f"{test['score']:.0%}",
                        'Note': test['grade']
                    })
                
                df = pd.DataFrame(test_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Aucun historique")

# ==================== PAGE: LÉVITATION MAGNÉTIQUE ====================

elif page == "🚁 Lévitation Magnétique":
    st.header("🚁 Systèmes de Lévitation Magnétique")
    
    levitation_systems = {k: v for k, v in st.session_state.superconductor_system['systems'].items() if 'levitation' in v}
    
    if not levitation_systems:
        st.info("💡 Aucun système de lévitation. Créez un système avec lévitation magnétique!")
    else:
        sys_options = {s['id']: s['name'] for k, s in levitation_systems.items()}
        selected_sys = st.selectbox(
            "Sélectionner un système de lévitation",
            options=list(sys_options.keys()),
            format_func=lambda x: sys_options[x]
        )
        
        sys = st.session_state.superconductor_system['systems'][selected_sys]
        lev = sys['levitation']
        
        st.markdown(f"### 🚁 {sys['name']}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Type", lev['type'].replace('_', ' ').title())
        with col2:
            st.metric("Capacité", f"{lev['load_capacity']:.0f} kg")
        with col3:
            st.metric("Hauteur", f"{lev['levitation_height']:.1f} mm")
        with col4:
            st.metric("Stabilité", f"{lev['stability']:.0%}")
        
        st.markdown("---")
        
        # Contrôle de lévitation
        st.subheader("🎮 Contrôle de Lévitation")
        
        load = st.slider("Charge à Léviter (kg)", 0.0, lev['load_capacity'], 0.0, 1.0)
        
        if st.button("🚀 Activer Lévitation", use_container_width=True):
            if load > lev['load_capacity']:
                st.error(f"❌ Charge trop élevée! Maximum: {lev['load_capacity']} kg")
            else:
                # Calcul hauteur
                height = (lev['load_capacity'] - load) * 0.1
                lev['levitation_height'] = height
                
                # Calcul stabilité
                stability = 1.0 - (load / lev['load_capacity'])
                lev['stability'] = stability
                
                st.success(f"✅ Lévitation activée!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Hauteur Lévitation", f"{height:.2f} mm")
                with col2:
                    st.metric("Stabilité", f"{stability:.0%}")
                with col3:
                    energy = load * 9.81 * height * 0.001
                    st.metric("Énergie", f"{energy:.2f} J")
                
                # Visualisation
                st.markdown("---")
                
                fig = go.Figure()
                
                # Objet en lévitation
                fig.add_trace(go.Scatter(
                    x=[0, 1, 1, 0, 0],
                    y=[height, height, height+10, height+10, height],
                    fill='toself',
                    fillcolor='rgba(79, 172, 254, 0.5)',
                    line=dict(color='rgb(79, 172, 254)'),
                    name='Objet'
                ))
                
                # Base supraconductrice
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color='blue', width=5),
                    name='Supraconducteur'
                ))
                
                fig.update_layout(
                    title="Visualisation Lévitation",
                    xaxis_title="Position (m)",
                    yaxis_title="Hauteur (mm)",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: AMPLIFICATEURS ====================

elif page == "📡 Amplificateurs":
    st.header("📡 Systèmes Amplificateurs")
    
    amplifier_systems = {k: v for k, v in st.session_state.superconductor_system['systems'].items() if 'amplifier' in v}
    
    if not amplifier_systems:
        st.info("💡 Aucun amplificateur. Créez un système amplificateur!")
    else:
        sys_options = {s['id']: s['name'] for k, s in amplifier_systems.items()}
        selected_sys = st.selectbox(
            "Sélectionner un amplificateur",
            options=list(sys_options.keys()),
            format_func=lambda x: sys_options[x]
        )
        
        sys = st.session_state.superconductor_system['systems'][selected_sys]
        amp = sys['amplifier']
        
        st.markdown(f"### 📡 {sys['name']}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Type", amp['type'].replace('_', ' ').title())
        with col2:
            st.metric("Gain", f"{amp['gain']:.1f} dB")
        with col3:
            st.metric("Bande Passante", f"{amp['bandwidth']/1e9:.2f} GHz")
        with col4:
            st.metric("Figure de Bruit", f"{amp['noise_figure']:.2f} dB")
        
        st.markdown("---")
        
        # Simulation d'amplification
        st.subheader("🎛️ Simulation d'Amplification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            input_signal = st.number_input("Signal d'Entrée (V)", 0.001, 10.0, 0.1, 0.001, format="%.3f")
            frequency = st.number_input("Fréquence (GHz)", 0.1, amp['bandwidth']/1e9, 1.0)
        
        with col2:
            st.write("**Paramètres Amplificateur:**")
            st.write(f"Gain Linéaire: {10**(amp['gain']/20):.2f}x")
            st.write(f"Bande Passante: {amp['bandwidth']/1e9:.2f} GHz")
        
        if st.button("📊 Amplifier Signal", use_container_width=True):
            if frequency * 1e9 > amp['bandwidth']:
                st.error(f"❌ Fréquence hors bande passante!")
            else:
                # Calcul
                gain_linear = 10 ** (amp['gain'] / 20)
                output_signal = input_signal * gain_linear
                noise = 10 ** (amp['noise_figure'] / 10) * 1e-9
                snr = 20 * np.log10(output_signal / noise)
                
                st.success("✅ Amplification effectuée!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Signal Entrée", f"{input_signal:.3f} V")
                with col2:
                    st.metric("Signal Sortie", f"{output_signal:.3f} V")
                with col3:
                    st.metric("SNR", f"{snr:.1f} dB")
                
                # Graphique
                st.markdown("---")
                
                x = np.linspace(0, 10, 1000)
                input_wave = input_signal * np.sin(2 * np.pi * frequency * x)
                output_wave = output_signal * np.sin(2 * np.pi * frequency * x)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=input_wave, mode='lines', name='Entrée'))
                fig.add_trace(go.Scatter(x=x, y=output_wave, mode='lines', name='Sortie'))
                fig.update_layout(
                    title="Signaux d'Entrée et de Sortie",
                    xaxis_title="Temps (ns)",
                    yaxis_title="Amplitude (V)"
                )
                st.plotly_chart(fig, use_container_width=True)


# ==================== PAGE: CONFIGURATION ====================

elif page == "🔧 Configuration":
    st.header("🔧 Configuration du Système")
    
    if not st.session_state.superconductor_system['systems']:
        st.warning("⚠️ Aucun système disponible pour configuration")
    else:
        sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
        selected_sys = st.selectbox(
            "Sélectionner un système",
            options=list(sys_options.keys()),
            format_func=lambda x: sys_options[x]
        )
        
        sys = st.session_state.superconductor_system['systems'][selected_sys]
        
        st.markdown(f"### 🧲 {sys['name']}")
        st.markdown(f"**Type:** {sys['type'].replace('_', ' ').title()}")
        
        tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Paramètres Généraux", "🧲 Magnétique", "❄️ Refroidissement", "🤖 Systèmes Avancés"])
        
        with tab1:
            st.subheader("⚙️ Paramètres Généraux")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input("Nom du Système", value=sys['name'])
                
                new_material = st.selectbox(
                    "Matériau",
                    ["ybco", "bscco", "nbti", "nb3sn", "mgdb2"],
                    index=["ybco", "bscco", "nbti", "nb3sn", "mgdb2"].index(sys['material']),
                    format_func=lambda x: x.upper()
                )
                
                new_purity = st.slider("Pureté Matériau", 0.90, 0.9999, 0.999, 0.0001, format="%.4f")
            
            with col2:
                target_efficiency = st.slider(
                    "Efficacité Cible",
                    0.70, 0.99,
                    sys['performance']['efficiency'],
                    0.01
                )
                
                maintenance_interval = st.number_input(
                    "Intervalle Maintenance (heures)",
                    100, 10000, 1000, 100
                )
            
            if st.button("💾 Sauvegarder Configuration Générale", use_container_width=True):
                sys['name'] = new_name
                sys['material'] = new_material
                sys['performance']['efficiency'] = target_efficiency
                
                st.success("✅ Configuration sauvegardée!")
                log_event(f"Configuration mise à jour: {new_name}")
                st.rerun()
        
        with tab2:
            st.subheader("🧲 Configuration Magnétique")
            
            col1, col2 = st.columns(2)
            
            with col1:
                field_target = st.number_input(
                    "Intensité Champ Cible (Tesla)",
                    0.1, 200.0,
                    sys['magnetic_properties']['field_strength'],
                    0.1
                )
                
                uniformity_target = st.slider(
                    "Uniformité du Champ",
                    0.85, 0.99,
                    sys['magnetic_properties']['field_uniformity'],
                    0.01
                )
            
            with col2:
                stability_target = st.slider(
                    "Stabilité du Champ",
                    0.85, 0.99,
                    sys['magnetic_properties']['field_stability'],
                    0.01
                )
                
                flux_density = st.number_input("Densité de Flux (T)", 0.0, 10.0, 0.5, 0.1)
            
            st.markdown("---")
            
            st.write("**Configuration Avancée:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                coercivity = st.number_input("Coercivité (A/m)", 0, 100000, 10000, 1000)
            with col2:
                remanence = st.number_input("Rémanence (T)", 0.0, 5.0, 1.0, 0.1)
            with col3:
                hysteresis = st.checkbox("Compensation Hystérésis", value=True)
            
            if st.button("💾 Sauvegarder Configuration Magnétique", use_container_width=True):
                sys['magnetic_properties']['field_strength'] = field_target
                sys['magnetic_properties']['field_uniformity'] = uniformity_target
                sys['magnetic_properties']['field_stability'] = stability_target
                sys['magnetic_properties']['flux_density'] = flux_density
                sys['magnetic_properties']['coercivity'] = coercivity
                sys['magnetic_properties']['remanence'] = remanence
                
                st.success("✅ Configuration magnétique sauvegardée!")
                log_event(f"Paramètres magnétiques mis à jour: {sys['name']}")
                st.rerun()
        
        with tab3:
            st.subheader("❄️ Configuration Refroidissement")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cooling_system = st.selectbox(
                    "Système de Refroidissement",
                    ["azote_liquide", "helium_liquide", "cryorefroidisseur", "refrigerateur_dilution", "tube_pulsation"],
                    index=["azote_liquide", "helium_liquide", "cryorefroidisseur", "refrigerateur_dilution", "tube_pulsation"].index(sys['cooling']['system']),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                target_temp = st.number_input(
                    "Température Cible (K)",
                    1.0, 300.0,
                    sys['cooling']['temperature'],
                    0.1
                )
                
                cooling_power = st.number_input("Puissance Refroidissement (W)", 1, 10000, 1000, 100)
            
            with col2:
                st.write("**Températures de Référence:**")
                st.info(f"🧊 Azote Liquide: 77 K (-196°C)")
                st.info(f"❄️ Hélium Liquide: 4.2 K (-269°C)")
                st.info(f"🔬 Réfrigérateur Dilution: 0.01 K")
                
                st.write(f"**Tc du Système:** {sys['critical_temperature']:.1f} K")
                if target_temp > sys['critical_temperature'] * 0.9:
                    st.warning("⚠️ Température proche de Tc!")
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                safety_margin = st.slider("Marge de Sécurité (%)", 5, 30, 10, 5)
            with col2:
                auto_regulation = st.checkbox("Régulation Automatique", value=True)
            with col3:
                emergency_mode = st.checkbox("Mode Urgence Activé", value=False)
            
            if st.button("💾 Sauvegarder Configuration Refroidissement", use_container_width=True):
                sys['cooling']['system'] = cooling_system
                sys['cooling']['temperature'] = target_temp
                sys['cooling']['cooling_power'] = cooling_power
                
                st.success("✅ Configuration refroidissement sauvegardée!")
                log_event(f"Système de refroidissement mis à jour: {sys['name']}")
                st.rerun()
        
        with tab4:
            st.subheader("🤖 Systèmes Avancés")
            
            # Configuration Quantique
            if 'quantum' in sys:
                st.write("### ⚛️ Système Quantique")
                col1, col2 = st.columns(2)
                
                with col1:
                    qubits = st.number_input("Nombre de Qubits", 10, 10000, sys['quantum']['qubits'], 10)
                    coherence_time = st.number_input("Temps de Cohérence (μs)", 10, 1000, int(sys['quantum']['coherence_time']), 10)
                
                with col2:
                    gate_fidelity = st.slider("Fidélité des Portes", 0.90, 0.9999, sys['quantum']['gate_fidelity'], 0.0001, format="%.4f")
                    quantum_volume = st.number_input("Volume Quantique", 1, 1000000, 1000, 100)
                
                if st.button("💾 Sauvegarder Config Quantique"):
                    sys['quantum']['qubits'] = qubits
                    sys['quantum']['coherence_time'] = coherence_time
                    sys['quantum']['gate_fidelity'] = gate_fidelity
                    st.success("✅ Configuration quantique sauvegardée!")
                
                st.markdown("---")
            
            # Configuration Biologique
            if 'biological' in sys:
                st.write("### 🧬 Interface Biologique")
                col1, col2 = st.columns(2)
                
                with col1:
                    bio_interface = st.checkbox("Interface Bio Activée", value=sys['biological']['bio_interface'])
                    biocompatibility = st.slider("Biocompatibilité", 0.5, 1.0, sys['biological']['biocompatibility'], 0.01)
                
                with col2:
                    self_healing = st.slider("Auto-Réparation", 0.0, 1.0, sys['biological']['self_healing'], 0.01)
                    adaptive_response = st.slider("Réponse Adaptative", 0.0, 1.0, 0.7, 0.01)
                
                if st.button("💾 Sauvegarder Config Biologique"):
                    sys['biological']['bio_interface'] = bio_interface
                    sys['biological']['biocompatibility'] = biocompatibility
                    sys['biological']['self_healing'] = self_healing
                    sys['biological']['adaptive_response'] = adaptive_response
                    st.success("✅ Configuration biologique sauvegardée!")
                
                st.markdown("---")
            
            # Configuration IA
            if 'ai' in sys:
                st.write("### 🤖 Intelligence Artificielle")
                col1, col2 = st.columns(2)
                
                with col1:
                    ai_enabled = st.checkbox("IA Activée", value=sys['ai']['enabled'])
                    intelligence_level = st.slider("Niveau d'Intelligence", 0.0, 1.0, sys['ai']['intelligence_level'], 0.05)
                    autonomous_control = st.checkbox("Contrôle Autonome", value=sys['ai']['autonomous_control'])
                
                with col2:
                    predictive_maintenance = st.checkbox("Maintenance Prédictive", value=True)
                    self_optimization = st.checkbox("Auto-Optimisation", value=False)
                    learning_rate = st.slider("Taux d'Apprentissage", 0.001, 0.1, 0.01, 0.001)
                
                if st.button("💾 Sauvegarder Config IA"):
                    sys['ai']['enabled'] = ai_enabled
                    sys['ai']['intelligence_level'] = intelligence_level
                    sys['ai']['autonomous_control'] = autonomous_control
                    st.success("✅ Configuration IA sauvegardée!")

# ==================== PAGE: PROPRIÉTÉS MAGNÉTIQUES ====================

elif page == "📊 Propriétés Magnétiques":
    st.header("📊 Analyse des Propriétés Magnétiques")
    
    if not st.session_state.superconductor_system['systems']:
        st.warning("⚠️ Aucun système disponible")
    else:
        sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
        selected_sys = st.selectbox(
            "Sélectionner un système",
            options=list(sys_options.keys()),
            format_func=lambda x: sys_options[x]
        )
        
        sys = st.session_state.superconductor_system['systems'][selected_sys]
        mag = sys['magnetic_properties']
        
        st.markdown(f"### 🧲 {sys['name']}")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Intensité Champ", f"{mag['field_strength']:.2f} T", 
                     delta=f"+{np.random.random()*0.5:.2f} T")
        with col2:
            st.metric("Uniformité", f"{mag['field_uniformity']:.1%}",
                     delta=f"+{np.random.random()*2:.1f}%")
        with col3:
            st.metric("Stabilité", f"{mag['field_stability']:.1%}",
                     delta=f"+{np.random.random()*1.5:.1f}%")
        with col4:
            st.metric("Flux Density", f"{mag.get('flux_density', 0):.2f} T")
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Cartographie", "📊 Analyse Temporelle", "🔬 Hystérésis", "⚡ Calculs"])
        
        with tab1:
            st.subheader("📈 Cartographie du Champ Magnétique")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Génération de la carte de champ
                x = np.linspace(-10, 10, 100)
                y = np.linspace(-10, 10, 100)
                X, Y = np.meshgrid(x, y)
                
                # Simulation d'un champ magnétique
                R = np.sqrt(X**2 + Y**2)
                B = mag['field_strength'] * np.exp(-R/5) * (1 + 0.1*np.sin(3*np.arctan2(Y, X)))
                
                fig = go.Figure(data=go.Contour(
                    z=B,
                    x=x,
                    y=y,
                    colorscale='Viridis',
                    colorbar=dict(title="Intensité (T)")
                ))
                
                fig.update_layout(
                    title="Cartographie 2D du Champ Magnétique",
                    xaxis_title="Position X (cm)",
                    yaxis_title="Position Y (cm)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Statistiques:**")
                st.metric("Champ Maximum", f"{B.max():.2f} T")
                st.metric("Champ Minimum", f"{B.min():.2f} T")
                st.metric("Champ Moyen", f"{B.mean():.2f} T")
                st.metric("Écart-Type", f"{B.std():.3f} T")
                
                st.write("**Zones d'Intérêt:**")
                st.info(f"🔴 Zone Centrale: {mag['field_strength']:.2f} T")
                st.info(f"🟡 Zone Moyenne: {mag['field_strength']*0.7:.2f} T")
                st.info(f"🟢 Zone Périphérique: {mag['field_strength']*0.3:.2f} T")
        
        with tab2:
            st.subheader("📊 Évolution Temporelle du Champ")
            
            # Simulation de données temporelles
            time = np.linspace(0, 100, 1000)
            field = mag['field_strength'] * (1 + 0.02*np.sin(0.1*time) + 0.01*np.random.randn(1000))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time,
                y=field,
                mode='lines',
                name='Champ Magnétique',
                line=dict(color='rgb(79, 172, 254)', width=2)
            ))
            
            # Ligne de référence
            fig.add_hline(y=mag['field_strength'], line_dash="dash", 
                         annotation_text="Valeur Nominale", line_color="green")
            
            fig.update_layout(
                title="Stabilité Temporelle du Champ Magnétique",
                xaxis_title="Temps (s)",
                yaxis_title="Intensité (T)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dérive Maximum", f"{(field.max()-mag['field_strength'])/mag['field_strength']*100:.2f}%")
            with col2:
                st.metric("Stabilité RMS", f"{((field.std()/mag['field_strength'])*100):.3f}%")
            with col3:
                st.metric("Temps de Stabilisation", "12.5 s")
        
        with tab3:
            st.subheader("🔬 Courbe d'Hystérésis")
            
            # Génération courbe d'hystérésis
            H = np.linspace(-10, 10, 200)
            
            # Modèle simplifié d'hystérésis
            Br = mag.get('remanence', 1.0)
            Hc = mag.get('coercivity', 5000) / 1000  # Conversion en kA/m
            
            # Branches montante et descendante
            B_up = Br * np.tanh(H/Hc) + 0.1*H
            B_down = -Br * np.tanh((H+2*Hc)/Hc) + 0.1*H
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=H,
                y=B_up,
                mode='lines',
                name='Montée',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=H,
                y=B_down,
                mode='lines',
                name='Descente',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title="Courbe d'Hystérésis Magnétique",
                xaxis_title="Champ H (kA/m)",
                yaxis_title="Induction B (T)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rémanence (Br)", f"{Br:.2f} T")
            with col2:
                st.metric("Coercivité (Hc)", f"{Hc:.1f} kA/m")
            with col3:
                area = np.trapz(B_up - B_down, H)
                st.metric("Aire Hystérésis", f"{abs(area):.2f} J/m³")
        
        with tab4:
            st.subheader("⚡ Calculs Magnétiques")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📐 Loi d'Ampère")
                current = st.number_input("Courant (A)", 1.0, 10000.0, 1000.0, 100.0)
                radius = st.number_input("Rayon bobine (m)", 0.01, 1.0, 0.1, 0.01)
                
                if st.button("Calculer Champ"):
                    mu_0 = 4 * np.pi * 1e-7
                    B_calculated = (mu_0 * current) / (2 * np.pi * radius)
                    
                    st.success(f"🧲 Champ Magnétique: **{B_calculated:.4f} T**")
                    st.info(f"En Gauss: {B_calculated*10000:.2f} G")
            
            with col2:
                st.write("### ⚡ Énergie Magnétique")
                volume = st.number_input("Volume (m³)", 0.001, 1.0, 0.01, 0.001)
                
                if st.button("Calculer Énergie"):
                    mu_0 = 4 * np.pi * 1e-7
                    energy = (mag['field_strength']**2 * volume) / (2 * mu_0)
                    
                    st.success(f"⚡ Énergie Stockée: **{energy:.2f} J**")
                    st.info(f"Puissance (1s): {energy:.2f} W")
            
            st.markdown("---")
            
            st.write("### 🔬 Force de Lorentz")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                charge = st.number_input("Charge (C)", 1e-19, 1.0, 1.6e-19, format="%.2e")
            with col2:
                velocity = st.number_input("Vitesse (m/s)", 1.0, 1e8, 1e6, format="%.2e")
            with col3:
                angle = st.slider("Angle (degrés)", 0, 90, 90)
            
            if st.button("Calculer Force de Lorentz", use_container_width=True):
                F = charge * velocity * mag['field_strength'] * np.sin(np.radians(angle))
                
                st.success(f"💪 Force: **{F:.6e} N**")
                
                # Rayon de courbure
                mass = 9.11e-31  # électron
                radius_curv = (mass * velocity) / (charge * mag['field_strength'])
                st.info(f"📐 Rayon de courbure (électron): {radius_curv:.6e} m")

# ==================== PAGE: DÉPLOIEMENT ====================

elif page == "🚀 Déploiement":
    st.header("🚀 Déploiement et Exploitation")
    
    if not st.session_state.superconductor_system['systems']:
        st.warning("⚠️ Aucun système disponible")
    else:
        tab1, tab2, tab3 = st.tabs(["🚀 Nouveau Déploiement", "📍 Déploiements Actifs", "📊 Monitoring"])
        
        with tab1:
            st.subheader("🚀 Planifier un Déploiement")
            
            sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
            selected_sys = st.selectbox(
                "Sélectionner le système à déployer",
                options=list(sys_options.keys()),
                format_func=lambda x: sys_options[x]
            )
            
            sys = st.session_state.superconductor_system['systems'][selected_sys]
            
            st.markdown(f'<div class="system-card">', unsafe_allow_html=True)
            st.write(f"### 🧲 {sys['name']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tc", f"{sys['critical_temperature']:.1f} K")
            with col2:
                st.metric("Efficacité", f"{sys['performance']['efficiency']:.0%}")
            with col3:
                st.metric("Santé", f"{sys['health']:.0%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                location = st.text_input("📍 Localisation", placeholder="Ex: Lab A - Salle 101")
                
                application = st.selectbox(
                    "🎯 Application",
                    [
                        "Recherche Fondamentale",
                        "Transport Lévitation Magnétique",
                        "IRM Médicale",
                        "Accélérateur de Particules",
                        "Fusion Nucléaire",
                        "Informatique Quantique",
                        "Transmission d'Énergie",
                        "Stockage d'Énergie",
                        "Télécommunications",
                        "Autre"
                    ]
                )
                
                environment = st.selectbox(
                    "🌍 Environnement",
                    ["Laboratoire", "Industriel", "Médical", "Spatial", "Sous-marin", "Militaire"]
                )
            
            with col2:
                deployment_date = st.date_input("📅 Date de Déploiement", datetime.now())
                
                operational_mode = st.selectbox(
                    "⚙️ Mode Opérationnel",
                    ["Continu 24/7", "Intermittent", "Sur Demande", "Test/Validation"]
                )
                
                security_level = st.selectbox(
                    "🔒 Niveau de Sécurité",
                    ["Standard", "Élevé", "Très Élevé", "Critique"]
                )
            
            st.markdown("---")
            
            st.subheader("👥 Équipe et Responsables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                project_manager = st.text_input("Chef de Projet", placeholder="Nom du responsable")
                technical_lead = st.text_input("Responsable Technique", placeholder="Nom du technicien")
            
            with col2:
                safety_officer = st.text_input("Responsable Sécurité", placeholder="Nom du responsable")
                team_size = st.number_input("Taille de l'Équipe", 1, 50, 5)
            
            st.markdown("---")
            
            st.subheader("📋 Configuration du Déploiement")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                power_supply = st.selectbox("Alimentation", ["Réseau Standard", "Réseau Stabilisé", "UPS", "Générateur Secours"])
            with col2:
                monitoring = st.multiselect(
                    "Monitoring",
                    ["Température", "Champ Magnétique", "Courant", "Pression", "Vibrations"],
                    default=["Température", "Champ Magnétique"]
                )
            with col3:
                maintenance_plan = st.selectbox("Plan Maintenance", ["Standard", "Intensif", "Prédictif"])
            
            st.markdown("---")
            
            notes = st.text_area("📝 Notes et Remarques", placeholder="Ajoutez des notes sur le déploiement...")
            
            if st.button("🚀 Déployer le Système", use_container_width=True, type="primary"):
                if not location or not project_manager:
                    st.error("⚠️ Veuillez remplir tous les champs obligatoires")
                else:
                    deploy_id = f"deploy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    deployment = {
                        'deployment_id': deploy_id,
                        'system_id': selected_sys,
                        'system_name': sys['name'],
                        'location': location,
                        'application': application,
                        'environment': environment,
                        'deployment_date': deployment_date.isoformat(),
                        'operational_mode': operational_mode,
                        'security_level': security_level,
                        'team': {
                            'project_manager': project_manager,
                            'technical_lead': technical_lead,
                            'safety_officer': safety_officer,
                            'team_size': team_size
                        },
                        'configuration': {
                            'power_supply': power_supply,
                            'monitoring': monitoring,
                            'maintenance_plan': maintenance_plan
                        },
                        'notes': notes,
                        'status': 'operational',
                        'uptime': 0.0,
                        'incidents': []
                    }
                    
                    st.session_state.superconductor_system['deployments'][deploy_id] = deployment
                    sys['status'] = 'online'
                    
                    log_event(f"Déploiement: {sys['name']} → {location}")
                    
                    st.success(f"✅ Système déployé avec succès!")
                    st.balloons()
                    
                    st.code(f"Deployment ID: {deploy_id}", language="text")
        
        with tab2:
            st.subheader("📍 Déploiements Actifs")
            
            if not st.session_state.superconductor_system['deployments']:
                st.info("Aucun déploiement actif")
            else:
                for deploy_id, deploy in st.session_state.superconductor_system['deployments'].items():
                    with st.expander(f"📍 {deploy['system_name']} - {deploy['location']}"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Application:** {deploy['application']}")
                            st.write(f"**Environnement:** {deploy['environment']}")
                            st.write(f"**Mode:** {deploy['operational_mode']}")
                            st.write(f"**Sécurité:** {deploy['security_level']}")
                        
                        with col2:
                            st.metric("Uptime", f"{deploy['uptime']:.1f}h")
                            status_icon = "🟢" if deploy['status'] == 'operational' else "🔴"
                            st.write(f"**Statut:** {status_icon} {deploy['status'].upper()}")
                        
                        with col3:
                            st.metric("Incidents", len(deploy['incidents']))
                            st.write(f"**Déployé:** {deploy['deployment_date'][:10]}")
                        
                        st.markdown("---")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**👥 Équipe:**")
                            st.write(f"• Chef: {deploy['team']['project_manager']}")
                            st.write(f"• Technique: {deploy['team']['technical_lead']}")
                        
                        with col2:
                            st.write("**⚙️ Configuration:**")
                            st.write(f"• Alimentation: {deploy['configuration']['power_supply']}")
                            st.write(f"• Maintenance: {deploy['configuration']['maintenance_plan']}")
                        
                        with col3:
                            st.write("**📊 Monitoring:**")
                            for m in deploy['configuration']['monitoring']:
                                st.write(f"• {m}")
                        
                        if deploy['notes']:
                            st.info(f"📝 {deploy['notes']}")
                        
                        st.markdown("---")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if st.button("⏸️ Suspendre", key=f"pause_{deploy_id}"):
                                deploy['status'] = 'suspended'
                                st.rerun()
                        
                        with col2:
                            if st.button("🔧 Maintenance", key=f"maint_{deploy_id}"):
                                deploy['status'] = 'maintenance'
                                st.info("Mode maintenance activé")
                        
                        with col3:
                            if st.button("⚠️ Incident", key=f"incident_{deploy_id}"):
                                incident = {
                                    'timestamp': datetime.now().isoformat(),
                                    'type': 'manual',
                                    'description': 'Incident signalé manuellement'
                                }
                                deploy['incidents'].append(incident)
                                st.warning("Incident enregistré!")
                        
                        with col4:
                            if st.button("🗑️ Arrêter", key=f"stop_{deploy_id}"):
                                deploy['status'] = 'stopped'
                                sys = st.session_state.superconductor_system['systems'][deploy['system_id']]
                                sys['status'] = 'offline'
                                st.rerun()
        
        with tab3:
            st.subheader("📊 Monitoring en Temps Réel")
            
            if not st.session_state.superconductor_system['deployments']:
                st.info("Aucun déploiement à monitorer")
            else:
                deploy_options = {d['deployment_id']: f"{d['system_name']} - {d['location']}" 
                                 for d in st.session_state.superconductor_system['deployments'].values()}
                
                selected_deploy = st.selectbox(
                    "Sélectionner un déploiement",
                    options=list(deploy_options.keys()),
                    format_func=lambda x: deploy_options[x]
                )
                
                deploy = st.session_state.superconductor_system['deployments'][selected_deploy]
                sys = st.session_state.superconductor_system['systems'][deploy['system_id']]
                
                # Métriques temps réel
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    temp = sys['cooling']['temperature']
                    st.metric("Température", f"{temp:.1f} K", delta=f"{np.random.randn()*0.5:.2f} K")
                
                with col2:
                    field = sys['magnetic_properties']['field_strength']
                    st.metric("Champ", f"{field:.2f} T", delta=f"{np.random.randn()*0.1:.2f} T")
                
                with col3:
                    current = np.random.random() * 1000
                    st.metric("Courant", f"{current:.1f} A", delta=f"{np.random.randn()*10:.1f} A")
                
                with col4:
                    power = np.random.random() * 5000
                    st.metric("Puissance", f"{power:.0f} W", delta=f"{np.random.randn()*100:.0f} W")
                
                with col5:
                    efficiency = sys['performance']['efficiency']
                    st.metric("Efficacité", f"{efficiency:.0%}", delta=f"{np.random.randn()*1:.1f}%")
                
                st.markdown("---")
                
                # Graphiques temps réel
                col1, col2 = st.columns(2)
                
                with col1:
                    # Température
                    time = np.linspace(0, 60, 100)
                    temp_data = temp * (1 + 0.02*np.sin(0.5*time) + 0.01*np.random.randn(100))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=time, y=temp_data, mode='lines', 
                                            line=dict(color='blue', width=2)))
                    fig.add_hline(y=sys['critical_temperature'], line_dash="dash", 
                                 annotation_text="Tc", line_color="red")
                    fig.update_layout(title="Température (dernière heure)", 
                                     xaxis_title="Temps (min)", yaxis_title="T (K)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Champ magnétique
                    field_data = field * (1 + 0.01*np.sin(0.3*time) + 0.005*np.random.randn(100))
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=time, y=field_data, mode='lines',
                                            line=dict(color='green', width=2)))
                    fig.update_layout(title="Champ Magnétique (dernière heure)",
                                     xaxis_title="Temps (min)", yaxis_title="B (T)", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Alertes et événements
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("⚠️ Alertes Actives")
                    
                    if temp > sys['critical_temperature'] * 0.9:
                        st.error("🔥 Température proche de Tc!")
                    
                    if sys['health'] < 0.95:
                        st.warning(f"⚠️ Santé système: {sys['health']:.0%}")
                    
                    if sys['magnetic_properties']['field_stability'] < 0.9:
                        st.warning("⚠️ Stabilité magnétique faible")
                    
                    if not any([temp > sys['critical_temperature'] * 0.9, 
                               sys['health'] < 0.95,
                               sys['magnetic_properties']['field_stability'] < 0.9]):
                        st.success("✅ Aucune alerte")
                
                with col2:
                    st.subheader("📜 Événements Récents")
                    
                    events = [
                        {"time": "14:32", "event": "✅ Système démarré"},
                        {"time": "14:35", "event": "📊 Calibration complétée"},
                        {"time": "14:40", "event": "🔄 Optimisation auto"},
                        {"time": "14:45", "event": "📈 Performance stable"},
                    ]
                    
                    for evt in events[-5:]:
                        st.text(f"{evt['time']} - {evt['event']}")

# ==================== PAGE: PROJETS ====================

elif page == "📁 Projets":
    st.header("📁 Gestion de Projets")
    
    tab1, tab2, tab3 = st.tabs(["➕ Nouveau Projet", "📊 Projets Actifs", "📜 Archive"])
    
    with tab1:
        st.subheader("➕ Créer un Nouveau Projet")
        
        with st.form("new_project_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                project_name = st.text_input("📝 Nom du Projet", placeholder="Ex: MagLev Transport System")
                project_type = st.selectbox(
                    "🎯 Type de Projet",
                    ["Recherche & Développement", "Production", "Prototype", "Amélioration", "Maintenance"]
                )
                priority = st.selectbox("⚡ Priorité", ["Basse", "Moyenne", "Haute", "Critique"])
            
            with col2:
                start_date = st.date_input("📅 Date de Début")
                end_date = st.date_input("📅 Date de Fin Prévue")
                budget = st.number_input("💰 Budget ($)", 0, 100000000, 1000000, 100000)
            
            st.markdown("---")
            
            description = st.text_area("📋 Description du Projet", height=100,
                                      placeholder="Décrivez les objectifs et la portée du projet...")
            
            objectives = st.text_area("🎯 Objectifs Principaux", height=100,
                                     placeholder="Listez les objectifs clés du projet...")
            
            st.markdown("---")
            
            st.subheader("👥 Équipe du Projet")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                project_lead = st.text_input("Chef de Projet")
            with col2:
                technical_lead = st.text_input("Lead Technique")
            with col3:
                team_members = st.number_input("Membres d'Équipe", 1, 100, 5)
            
            st.markdown("---")
            
            st.subheader("🧲 Systèmes Associés")
            
            if st.session_state.superconductor_system['systems']:
                sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
                associated_systems = st.multiselect(
                    "Sélectionner les systèmes",
                    options=list(sys_options.keys()),
                    format_func=lambda x: sys_options[x]
                )
            else:
                st.info("Aucun système disponible")
                associated_systems = []
            
            st.markdown("---")
            
            milestones = st.text_area("🎯 Jalons du Projet", height=100,
                                     placeholder="Ex:\n- Phase 1: Conception (2 mois)\n- Phase 2: Prototypage (3 mois)")
            
            submitted = st.form_submit_button("🚀 Créer le Projet", use_container_width=True, type="primary")
            
            if submitted:
                if not project_name or not project_lead:
                    st.error("⚠️ Veuillez remplir les champs obligatoires")
                else:
                    project_id = f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    project = {
                        'project_id': project_id,
                        'name': project_name,
                        'type': project_type,
                        'priority': priority,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'budget': budget,
                        'description': description,
                        'objectives': objectives,
                        'team': {
                            'project_lead': project_lead,
                            'technical_lead': technical_lead,
                            'team_members': team_members
                        },
                        'associated_systems': associated_systems,
                        'milestones': milestones,
                        'status': 'active',
                        'progress': 0.0,
                        'spent_budget': 0.0,
                        'tasks': [],
                        'documents': [],
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.superconductor_system['projects'][project_id] = project
                    log_event(f"Projet créé: {project_name}")
                    
                    st.success("✅ Projet créé avec succès!")
                    st.balloons()
                    
                    st.code(f"Project ID: {project_id}", language="text")
    
    with tab2:
        st.subheader("📊 Projets Actifs")
        
        if not st.session_state.superconductor_system['projects']:
            st.info("Aucun projet actif. Créez votre premier projet!")
        else:
            # Filtres
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.multiselect("Type", ["Recherche & Développement", "Production", "Prototype", "Amélioration", "Maintenance"])
            with col2:
                filter_priority = st.multiselect("Priorité", ["Basse", "Moyenne", "Haute", "Critique"])
            with col3:
                filter_status = st.multiselect("Statut", ["active", "on_hold", "completed", "cancelled"])
            
            st.markdown("---")
            
            for proj_id, proj in st.session_state.superconductor_system['projects'].items():
                # Appliquer les filtres
                if filter_type and proj['type'] not in filter_type:
                    continue
                if filter_priority and proj['priority'] not in filter_priority:
                    continue
                if filter_status and proj['status'] not in filter_status:
                    continue
                
                # Déterminer la couleur selon la priorité
                priority_colors = {
                    'Critique': '🔴',
                    'Haute': '🟠',
                    'Moyenne': '🟡',
                    'Basse': '🟢'
                }
                
                priority_icon = priority_colors.get(proj['priority'], '⚪')
                
                with st.expander(f"{priority_icon} {proj['name']} - {proj['progress']:.0f}%"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**Type:** {proj['type']}")
                        st.write(f"**Priorité:** {proj['priority']}")
                        st.write(f"**Chef de Projet:** {proj['team']['project_lead']}")
                        st.write(f"**Description:** {proj['description'][:100]}...")
                    
                    with col2:
                        st.metric("Progression", f"{proj['progress']:.0f}%")
                        st.progress(proj['progress'] / 100)
                        
                        days_left = (datetime.fromisoformat(proj['end_date']) - datetime.now()).days
                        st.metric("Jours Restants", days_left)
                    
                    with col3:
                        budget_spent_pct = (proj['spent_budget'] / proj['budget'] * 100) if proj['budget'] > 0 else 0
                        st.metric("Budget", f"${proj['budget']:,.0f}")
                        st.metric("Dépensé", f"{budget_spent_pct:.0f}%")
                        st.progress(min(budget_spent_pct / 100, 1.0))
                    
                    st.markdown("---")
                    
                    if proj['objectives']:
                        st.write("**🎯 Objectifs:**")
                        st.info(proj['objectives'][:200] + "...")
                    
                    if proj['associated_systems']:
                        st.write("**🧲 Systèmes Associés:**")
                        for sys_id in proj['associated_systems']:
                            if sys_id in st.session_state.superconductor_system['systems']:
                                sys = st.session_state.superconductor_system['systems'][sys_id]
                                st.write(f"• {sys['name']}")
                    
                    st.markdown("---")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        if st.button("➕ Ajouter Tâche", key=f"task_{proj_id}"):
                            st.info("Fonctionnalité à venir")
                    
                    with col2:
                        if st.button("📄 Documents", key=f"docs_{proj_id}"):
                            st.info(f"{len(proj['documents'])} document(s)")
                    
                    with col3:
                        new_progress = st.number_input("Progression", 0, 100, int(proj['progress']), key=f"prog_{proj_id}")
                        if st.button("💾 MAJ", key=f"update_{proj_id}"):
                            proj['progress'] = float(new_progress)
                            st.success("Mis à jour!")
                    
                    with col4:
                        if st.button("⏸️ Pause", key=f"pause_{proj_id}"):
                            proj['status'] = 'on_hold'
                            st.rerun()
                    
                    with col5:
                        if st.button("✅ Terminer", key=f"complete_{proj_id}"):
                            proj['status'] = 'completed'
                            proj['progress'] = 100.0
                            st.rerun()
    
    with tab3:
        st.subheader("📜 Projets Archivés")
        
        archived = {k: v for k, v in st.session_state.superconductor_system['projects'].items() 
                   if v['status'] in ['completed', 'cancelled']}
        
        if not archived:
            st.info("Aucun projet archivé")
        else:
            proj_data = []
            for proj in archived.values():
                proj_data.append({
                    'Nom': proj['name'],
                    'Type': proj['type'],
                    'Statut': proj['status'].upper(),
                    'Progression': f"{proj['progress']:.0f}%",
                    'Budget': f"${proj['budget']:,.0f}",
                    'Début': proj['start_date'][:10],
                    'Fin': proj['end_date'][:10]
                })
            
            df = pd.DataFrame(proj_data)
            st.dataframe(df, use_container_width=True)

# ==================== PAGE: CRYOGÉNIE ====================

elif page == "❄️ Cryogénie":
    st.header("❄️ Systèmes Cryogéniques")
    
    if not st.session_state.superconductor_system['systems']:
        st.warning("⚠️ Aucun système disponible")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🌡️ Monitoring", "❄️ Systèmes", "📊 Performance", "🔧 Maintenance"])
        
        with tab1:
            st.subheader("🌡️ Monitoring Cryogénique")
            
            sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
            selected_sys = st.selectbox(
                "Sélectionner un système",
                options=list(sys_options.keys()),
                format_func=lambda x: sys_options[x]
            )
            
            sys = st.session_state.superconductor_system['systems'][selected_sys]
            cooling = sys['cooling']
            
            st.markdown(f"### ❄️ {sys['name']}")
            
            # Métriques principales
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Température", f"{cooling['temperature']:.2f} K",
                         delta=f"{np.random.randn()*0.1:.2f} K")
            
            with col2:
                st.metric("Tc", f"{sys['critical_temperature']:.1f} K")
            
            with col3:
                margin = (sys['critical_temperature'] - cooling['temperature']) / sys['critical_temperature'] * 100
                st.metric("Marge", f"{margin:.1f}%",
                         delta=f"{np.random.randn()*2:.1f}%")
            
            with col4:
                st.metric("Puissance", f"{cooling.get('cooling_power', 1000):.0f} W")
            
            with col5:
                st.metric("Efficacité", f"{cooling['efficiency']:.0%}")
            
            st.markdown("---")
            
            # Graphique température
            time = np.linspace(0, 24, 288)  # 24h par pas de 5 min
            temp_baseline = cooling['temperature']
            temp_variation = 0.5 * np.sin(2 * np.pi * time / 24) + 0.2 * np.random.randn(288)
            temp_data = temp_baseline + temp_variation
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time,
                y=temp_data,
                mode='lines',
                name='Température Réelle',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_hline(y=sys['critical_temperature'], line_dash="dash",
                         annotation_text="Température Critique", line_color="red")
            
            fig.add_hline(y=temp_baseline, line_dash="dot",
                         annotation_text="Cible", line_color="green")
            
            fig.update_layout(
                title="Évolution Température (24h)",
                xaxis_title="Temps (heures)",
                yaxis_title="Température (K)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Statistiques")
                
                st.metric("Température Moyenne", f"{temp_data.mean():.2f} K")
                st.metric("Écart-Type", f"{temp_data.std():.3f} K")
                st.metric("Température Min", f"{temp_data.min():.2f} K")
                st.metric("Température Max", f"{temp_data.max():.2f} K")
            
            with col2:
                st.subheader("⚠️ Alertes")
                
                if cooling['temperature'] > sys['critical_temperature'] * 0.95:
                    st.error("🔥 CRITIQUE: Température > 95% Tc!")
                elif cooling['temperature'] > sys['critical_temperature'] * 0.9:
                    st.warning("⚠️ Température > 90% Tc")
                else:
                    st.success("✅ Température dans la normale")
                
                if cooling['efficiency'] < 0.7:
                    st.warning("⚠️ Efficacité faible")
                else:
                    st.success(f"✅ Efficacité: {cooling['efficiency']:.0%}")
        
        with tab2:
            st.subheader("❄️ Systèmes de Refroidissement")
            
            # Vue d'ensemble
            cooling_systems = {}
            for sys in st.session_state.superconductor_system['systems'].values():
                sys_type = sys['cooling']['system']
                if sys_type not in cooling_systems:
                    cooling_systems[sys_type] = []
                cooling_systems[sys_type].append(sys)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Répartition par Type")
                
                labels = [k.replace('_', ' ').title() for k in cooling_systems.keys()]
                values = [len(v) for v in cooling_systems.values()]
                
                ice_colors = ['#e0f7fa', '#b2ebf2', '#80deea', '#4dd0e1', '#26c6da', '#00bcd4']
                fig = px.pie(values=values, names=labels,
                            color_discrete_sequence=ice_colors)
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### 🌡️ Plages de Température")
                
                temp_info = {
                    "Azote Liquide": "77 K (-196°C)",
                    "Hélium Liquide": "4.2 K (-269°C)",
                    "Cryorefroidisseur": "10-80 K",
                    "Réfrigérateur Dilution": "0.01-1 K",
                    "Tube Pulsation": "20-80 K"
                }
                
                for name, temp in temp_info.items():
                    st.info(f"**{name}:** {temp}")
            
            st.markdown("---")
            
            # Détails par système
            for cool_type, systems in cooling_systems.items():
                with st.expander(f"❄️ {cool_type.replace('_', ' ').title()} ({len(systems)} système(s))"):
                    for sys in systems:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**{sys['name']}**")
                        with col2:
                            st.metric("T", f"{sys['cooling']['temperature']:.1f} K")
                        with col3:
                            st.metric("Tc", f"{sys['critical_temperature']:.1f} K")
                        with col4:
                            st.metric("η", f"{sys['cooling']['efficiency']:.0%}")
        
        with tab3:
            st.subheader("📊 Performance Cryogénique")
            
            if st.session_state.superconductor_system['systems']:
                # Comparaison des efficacités
                systems_data = []
                for sys in st.session_state.superconductor_system['systems'].values():
                    systems_data.append({
                        'Système': sys['name'][:20],
                        'Type Refroidissement': sys['cooling']['system'].replace('_', ' ').title(),
                        'Température': sys['cooling']['temperature'],
                        'Efficacité': sys['cooling']['efficiency'],
                        'Marge': (sys['critical_temperature'] - sys['cooling']['temperature']) / sys['critical_temperature']
                    })
                
                df = pd.DataFrame(systems_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(df, x='Système', y='Efficacité',
                                color='Type Refroidissement',
                                title="Efficacité par Système")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(df, x='Température', y='Marge',
                                    size='Efficacité', color='Type Refroidissement',
                                    title="Température vs Marge de Sécurité",
                                    hover_data=['Système'])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Tableau détaillé
                st.dataframe(df, use_container_width=True)
                
                # Statistiques globales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Efficacité Moyenne", f"{df['Efficacité'].mean():.0%}")
                with col2:
                    st.metric("Température Moyenne", f"{df['Température'].mean():.1f} K")
                with col3:
                    st.metric("Marge Moyenne", f"{df['Marge'].mean():.0%}")
                with col4:
                    st.metric("Meilleure Efficacité", f"{df['Efficacité'].max():.0%}")
            else:
                st.info("Aucune donnée disponible")
        
        with tab4:
            st.subheader("🔧 Maintenance Cryogénique")
            
            sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
            selected_sys = st.selectbox(
                "Système à maintenir",
                options=list(sys_options.keys()),
                format_func=lambda x: sys_options[x],
                key="maint_sys"
            )
            
            sys = st.session_state.superconductor_system['systems'][selected_sys]
            
            st.markdown(f"### 🔧 {sys['name']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🧊 Recharge Cryogène:**")
                
                if sys['cooling']['system'] == 'azote_liquide':
                    level = np.random.random() * 100
                    st.metric("Niveau Azote", f"{level:.0f}%")
                    st.progress(level / 100)
                    
                    if level < 20:
                        st.error("🔴 Niveau critique! Recharge nécessaire")
                    elif level < 50:
                        st.warning("🟡 Niveau bas")
                    else:
                        st.success("🟢 Niveau OK")
                    
                    if st.button("➕ Recharger Azote"):
                        st.success("✅ Recharge azote liquide planifiée")
                
                elif sys['cooling']['system'] == 'helium_liquide':
                    level = np.random.random() * 100
                    st.metric("Niveau Hélium", f"{level:.0f}%")
                    st.progress(level / 100)
                    
                    if level < 15:
                        st.error("🔴 Niveau critique! Recharge urgente")
                    elif level < 40:
                        st.warning("🟡 Niveau bas")
                    else:
                        st.success("🟢 Niveau OK")
                    
                    if st.button("➕ Recharger Hélium"):
                        st.success("✅ Recharge hélium liquide planifiée")
            
            with col2:
                st.write("**⚙️ Maintenance Préventive:**")
                
                last_maintenance = datetime.now() - timedelta(days=np.random.randint(1, 90))
                next_maintenance = last_maintenance + timedelta(days=90)
                days_until = (next_maintenance - datetime.now()).days
                
                st.write(f"**Dernière maintenance:** {last_maintenance.strftime('%Y-%m-%d')}")
                st.write(f"**Prochaine maintenance:** {next_maintenance.strftime('%Y-%m-%d')}")
                st.metric("Jours restants", days_until)
                
                if days_until < 7:
                    st.warning("⚠️ Maintenance prochaine!")
                elif days_until < 0:
                    st.error("🔴 Maintenance en retard!")
                else:
                    st.success("✅ Planning OK")
                
                if st.button("🔧 Programmer Maintenance"):
                    st.success("✅ Maintenance programmée")
            
            st.markdown("---")
            
            st.subheader("📋 Checklist de Maintenance")
            
            checklist_items = [
                "Vérifier niveau cryogène",
                "Inspecter isolation thermique",
                "Contrôler capteurs température",
                "Tester vannes de sécurité",
                "Vérifier pompes circulation",
                "Nettoyer échangeurs thermiques",
                "Calibrer régulation température",
                "Vérifier alarmes",
                "Test système secours",
                "Documenter relevés"
            ]
            
            completed = 0
            for item in checklist_items:
                if st.checkbox(item, key=f"check_{selected_sys}_{item}"):
                    completed += 1
            
            st.progress(completed / len(checklist_items))
            st.write(f"**Progression:** {completed}/{len(checklist_items)} ({completed/len(checklist_items)*100:.0f}%)")
            
            if completed == len(checklist_items):
                if st.button("✅ Valider Maintenance Complète", use_container_width=True, type="primary"):
                    st.success("🎉 Maintenance complétée et validée!")
                    st.balloons()
                    log_event(f"Maintenance complétée: {sys['name']}")

# ==================== PAGE: BIBLIOTHÈQUE ====================

elif page == "📚 Bibliothèque":
    st.header("📚 Bibliothèque de Ressources")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Documentation", "🔬 Matériaux", "📐 Formules", "📊 Données"])
    
    with tab1:
        st.subheader("📖 Documentation Technique")
        
        docs_categories = {
            "🧲 Supraconductivité": [
                {"title": "Introduction aux Supraconducteurs", "type": "Guide", "pages": 45},
                {"title": "Supraconducteurs de Type I et II", "type": "Article", "pages": 12},
                {"title": "Effet Meissner et Applications", "type": "Tutorial", "pages": 23},
                {"title": "Température Critique - Théorie BCS", "type": "Recherche", "pages": 67},
                {"title": "Supraconducteurs Haute Température", "type": "Guide", "pages": 89}
            ],
            "🧊 Cryogénie": [
                {"title": "Systèmes de Refroidissement", "type": "Manuel", "pages": 120},
                {"title": "Azote Liquide - Guide Pratique", "type": "Guide", "pages": 34},
                {"title": "Hélium Liquide et Ultra-Basse Température", "type": "Tutorial", "pages": 56},
                {"title": "Cryorefroidisseurs Modernes", "type": "Article", "pages": 28},
                {"title": "Sécurité en Cryogénie", "type": "Manuel", "pages": 42}
            ],
            "🧲 Magnétisme": [
                {"title": "Champs Magnétiques Intenses", "type": "Recherche", "pages": 78},
                {"title": "Bobines Supraconductrices", "type": "Guide", "pages": 54},
                {"title": "Lévitation Magnétique", "type": "Tutorial", "pages": 39},
                {"title": "Blindage Magnétique", "type": "Article", "pages": 21},
                {"title": "Mesure et Caractérisation", "type": "Manuel", "pages": 67}
            ],
            "⚡ Applications": [
                {"title": "IRM et Applications Médicales", "type": "Guide", "pages": 92},
                {"title": "Transport MagLev", "type": "Article", "pages": 45},
                {"title": "Accélérateurs de Particules", "type": "Recherche", "pages": 134},
                {"title": "Fusion Nucléaire - Tokamaks", "type": "Guide", "pages": 156},
                {"title": "Stockage d'Énergie Supraconducteur", "type": "Tutorial", "pages": 48}
            ]
        }
        
        for category, docs in docs_categories.items():
            with st.expander(f"{category} ({len(docs)} documents)"):
                for doc in docs:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{doc['title']}**")
                    with col2:
                        st.caption(doc['type'])
                    with col3:
                        st.caption(f"{doc['pages']} pages")
                    with col4:
                        if st.button("📥", key=f"dl_{doc['title']}"):
                            st.info("Téléchargement simulé")
    
    with tab2:
        st.subheader("🔬 Base de Données Matériaux")
        
        materials_data = {
            "YBCO (YBa₂Cu₃O₇)": {
                "famille": "Cuprate",
                "tc": 92,
                "jc": "1×10¹⁰ A/m²",
                "bc": "100-250 T",
                "type": "Type II",
                "couleur": "Noir",
                "densite": "6.3 g/cm³",
                "applications": ["IRM", "Câbles", "SMES"],
                "cout": "Élevé",
                "disponibilite": "Commerciale"
            },
            "BSCCO (Bi₂Sr₂CaCu₂O₈)": {
                "famille": "Cuprate",
                "tc": 110,
                "jc": "5×10⁹ A/m²",
                "bc": "50-150 T",
                "type": "Type II",
                "couleur": "Gris-noir",
                "densite": "6.2 g/cm³",
                "applications": ["Câbles HT", "Bobines"],
                "cout": "Très élevé",
                "disponibilite": "Limitée"
            },
            "NbTi (Niobium-Titane)": {
                "famille": "Alliage métallique",
                "tc": 9.2,
                "jc": "3×10⁹ A/m²",
                "bc": "12-15 T",
                "type": "Type II",
                "couleur": "Métallique",
                "densite": "6.5 g/cm³",
                "applications": ["IRM", "Accélérateurs"],
                "cout": "Modéré",
                "disponibilite": "Excellente"
            },
            "Nb₃Sn (Niobium-Étain)": {
                "famille": "Intermétallique",
                "tc": 18.3,
                "jc": "5×10⁹ A/m²",
                "bc": "24-30 T",
                "type": "Type II",
                "couleur": "Gris",
                "densite": "8.9 g/cm³",
                "applications": ["Fusion", "Aimants intenses"],
                "cout": "Élevé",
                "disponibilite": "Bonne"
            },
            "MgB₂ (Diborure de Magnésium)": {
                "famille": "Intermétallique",
                "tc": 39,
                "jc": "1×10¹⁰ A/m²",
                "bc": "30-40 T",
                "type": "Type II",
                "couleur": "Gris",
                "densite": "2.6 g/cm³",
                "applications": ["IRM portable", "Câbles"],
                "cout": "Faible",
                "disponibilite": "Excellente"
            }
        }
        
        selected_material = st.selectbox("Sélectionner un matériau", list(materials_data.keys()))
        
        mat = materials_data[selected_material]
        
        st.markdown(f"### 🔬 {selected_material}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Propriétés Physiques:**")
            st.info(f"**Famille:** {mat['famille']}")
            st.info(f"**Type:** {mat['type']}")
            st.info(f"**Densité:** {mat['densite']}")
            st.info(f"**Couleur:** {mat['couleur']}")
        
        with col2:
            st.write("**Propriétés Supraconductrices:**")
            st.success(f"**Tc:** {mat['tc']} K")
            st.success(f"**Jc:** {mat['jc']}")
            st.success(f"**Bc:** {mat['bc']}")
        
        with col3:
            st.write("**Informations Pratiques:**")
            st.info(f"**Coût:** {mat['cout']}")
            st.info(f"**Disponibilité:** {mat['disponibilite']}")
            st.write("**Applications:**")
            for app in mat['applications']:
                st.write(f"• {app}")
        
        st.markdown("---")
        
        # Comparaison
        st.subheader("📊 Comparaison des Matériaux")
        
        comparison_data = []
        for name, data in materials_data.items():
            comparison_data.append({
                'Matériau': name,
                'Tc (K)': data['tc'],
                'Type': data['type'],
                'Coût': data['cout'],
                'Disponibilité': data['disponibilite']
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Graphique Tc
        fig = px.bar(df, x='Matériau', y='Tc (K)', 
                     title="Température Critique par Matériau",
                     color='Tc (K)',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📐 Formules et Calculs")
        
        formula_categories = {
            "🧲 Électromagnétisme": {
                "Loi d'Ampère": {
                    "formule": "B = (μ₀ × I) / (2π × r)",
                    "description": "Champ magnétique créé par un fil",
                    "variables": {"B": "Champ magnétique (T)", "I": "Courant (A)", "r": "Distance (m)", "μ₀": "4π×10⁻⁷ H/m"}
                },
                "Force de Lorentz": {
                    "formule": "F = q × v × B × sin(θ)",
                    "description": "Force sur une charge en mouvement",
                    "variables": {"F": "Force (N)", "q": "Charge (C)", "v": "Vitesse (m/s)", "B": "Champ (T)", "θ": "Angle"}
                },
                "Énergie Magnétique": {
                    "formule": "U = (B² × V) / (2μ₀)",
                    "description": "Énergie stockée dans le champ",
                    "variables": {"U": "Énergie (J)", "B": "Champ (T)", "V": "Volume (m³)"}
                },
                "Inductance": {
                    "formule": "L = (μ₀ × N² × A) / l",
                    "description": "Inductance d'un solénoïde",
                    "variables": {"L": "Inductance (H)", "N": "Spires", "A": "Section (m²)", "l": "Longueur (m)"}
                }
            },
            "❄️ Supraconductivité": {
                "Longueur de Cohérence": {
                    "formule": "ξ = ℏvF / (π × Δ)",
                    "description": "Distance de cohérence des paires de Cooper",
                    "variables": {"ξ": "Longueur (m)", "ℏ": "h/2π", "vF": "Vitesse Fermi", "Δ": "Gap énergétique"}
                },
                "Profondeur de Pénétration": {
                    "formule": "λ = √(m / (μ₀ × n × q²))",
                    "description": "Profondeur de pénétration de London",
                    "variables": {"λ": "Profondeur (m)", "m": "Masse", "n": "Densité porteurs", "q": "Charge"}
                },
                "Densité de Courant Critique": {
                    "formule": "Jc = Jc0 × (1 - T/Tc)^n",
                    "description": "Dépendance en température",
                    "variables": {"Jc": "Densité critique", "T": "Température", "Tc": "Temp. critique", "n": "Exposant"}
                },
                "Champ Critique": {
                    "formule": "Bc(T) = Bc0 × [1 - (T/Tc)²]",
                    "description": "Champ critique en fonction de T",
                    "variables": {"Bc": "Champ critique", "T": "Température", "Tc": "Temp. critique"}
                }
            },
            "🧊 Thermodynamique": {
                "Loi de Carnot": {
                    "formule": "η = 1 - Tc/Th",
                    "description": "Efficacité maximale thermodynamique",
                    "variables": {"η": "Efficacité", "Tc": "Temp. froide", "Th": "Temp. chaude"}
                },
                "Transfert Thermique": {
                    "formule": "Q = k × A × ΔT / d",
                    "description": "Flux de chaleur par conduction",
                    "variables": {"Q": "Flux (W)", "k": "Conductivité", "A": "Surface", "ΔT": "Diff. temp.", "d": "Épaisseur"}
                },
                "Capacité Calorifique": {
                    "formule": "Q = m × c × ΔT",
                    "description": "Énergie pour chauffer/refroidir",
                    "variables": {"Q": "Énergie (J)", "m": "Masse (kg)", "c": "Capacité (J/kg·K)", "ΔT": "Variation T"}
                },
                "Temps de Refroidissement": {
                    "formule": "t = (m × c × ΔT) / P",
                    "description": "Temps pour atteindre température",
                    "variables": {"t": "Temps (s)", "m": "Masse", "c": "Capacité", "ΔT": "Variation", "P": "Puissance"}
                }
            }
        }
        
        for category, formulas in formula_categories.items():
            with st.expander(f"{category} ({len(formulas)} formules)"):
                for name, data in formulas.items():
                    st.markdown(f"#### {name}")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.code(data['formule'], language='text')
                        st.caption(data['description'])
                    
                    with col2:
                        st.write("**Variables:**")
                        for var, desc in data['variables'].items():
                            st.write(f"• **{var}:** {desc}")
                    
                    st.markdown("---")
    
    with tab4:
        st.subheader("📊 Données de Référence")
        
        ref_categories = {
            "🌡️ Températures Caractéristiques": {
                "Zéro Absolu": "0 K = -273.15°C",
                "Hélium Liquide": "4.2 K = -268.95°C",
                "Hydrogène Liquide": "20 K = -253°C",
                "Néon Liquide": "27 K = -246°C",
                "Azote Liquide": "77 K = -196°C",
                "Oxygène Liquide": "90 K = -183°C",
                "Température Ambiante": "293 K = 20°C",
                "Eau Bouillante": "373 K = 100°C"
            },
            "🧲 Champs Magnétiques Typiques": {
                "Champ Terrestre": "~50 μT",
                "Aimant Réfrigérateur": "~5 mT",
                "IRM Médical (faible)": "0.5-1.5 T",
                "IRM Médical (fort)": "3-7 T",
                "IRM Recherche": "7-11.7 T",
                "Bobine Supraconductrice": "10-30 T",
                "Record Laboratoire": ">100 T (pulsé)",
                "Étoile à Neutrons": "10⁸-10¹¹ T"
            },
            "⚡ Densités de Courant": {
                "Cuivre (normal)": "~10⁶ A/m²",
                "Aluminium (normal)": "~10⁶ A/m²",
                "NbTi (4.2K)": "~10⁹ A/m²",
                "Nb₃Sn (4.2K)": "~10⁹ A/m²",
                "YBCO (77K)": "~10¹⁰ A/m²",
                "BSCCO (77K)": "~10⁹ A/m²",
                "MgB₂ (20K)": "~10¹⁰ A/m²"
            },
            "💰 Coûts Indicatifs": {
                "Azote Liquide": "~0.5-1 $/L",
                "Hélium Liquide": "~10-30 $/L",
                "Fil NbTi": "~50-100 $/kg",
                "Fil YBCO": "~500-2000 $/m",
                "Cryostat Simple": "~10k-50k $",
                "Système IRM": "~1-3 M$",
                "Tokamak Recherche": "~100-500 M$"
            }
        }
        
        for category, data in ref_categories.items():
            with st.expander(f"{category}"):
                for name, value in data.items():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{name}**")
                    with col2:
                        st.info(value)
        
        st.markdown("---")
        
        # Tableau de conversion
        st.subheader("🔄 Conversions Utiles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🌡️ Température:**")
            kelvin = st.number_input("Kelvin", 0.0, 1000.0, 77.0, 1.0)
            celsius = kelvin - 273.15
            fahrenheit = celsius * 9/5 + 32
            
            st.success(f"**Celsius:** {celsius:.2f}°C")
            st.success(f"**Fahrenheit:** {fahrenheit:.2f}°F")
        
        with col2:
            st.write("**🧲 Champ Magnétique:**")
            tesla = st.number_input("Tesla", 0.0, 100.0, 1.0, 0.1)
            gauss = tesla * 10000
            oersted = tesla * 795.77
            
            st.success(f"**Gauss:** {gauss:.0f} G")
            st.success(f"**Oersted:** {oersted:.1f} Oe")

# ==================== PAGE: MAINTENANCE ====================

elif page == "⚙️ Maintenance":
    st.header("⚙️ Gestion de la Maintenance")
    
    if not st.session_state.superconductor_system['systems']:
        st.warning("⚠️ Aucun système disponible")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Vue d'Ensemble", "🔧 Préventive", "🚨 Corrective", "📊 Historique"])
        
        with tab1:
            st.subheader("📋 Vue d'Ensemble de la Maintenance")
            
            # Statistiques globales
            total_systems = len(st.session_state.superconductor_system['systems'])
            systems_needing_maint = sum(1 for s in st.session_state.superconductor_system['systems'].values() 
                                       if s['health'] < 0.95 or s['operational_hours'] > 1000)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Systèmes Totaux", total_systems)
            with col2:
                st.metric("Nécessitent Maintenance", systems_needing_maint)
            with col3:
                avg_health = np.mean([s['health'] for s in st.session_state.superconductor_system['systems'].values()])
                st.metric("Santé Moyenne", f"{avg_health:.0%}")
            with col4:
                critical = sum(1 for s in st.session_state.superconductor_system['systems'].values() if s['health'] < 0.85)
                st.metric("États Critiques", critical)
            
            st.markdown("---")
            
            # Liste des systèmes
            st.subheader("🔍 État des Systèmes")
            
            for sys_id, sys in st.session_state.superconductor_system['systems'].items():
                health_color = "🟢" if sys['health'] >= 0.95 else "🟡" if sys['health'] >= 0.85 else "🔴"
                
                with st.expander(f"{health_color} {sys['name']} - Santé: {sys['health']:.0%}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Santé Globale", f"{sys['health']:.0%}")
                        st.progress(sys['health'])
                    
                    with col2:
                        st.metric("Heures Opération", f"{sys['operational_hours']:.1f}h")
                        next_maint = max(0, 1000 - sys['operational_hours'])
                        st.metric("Prochaine Maint.", f"{next_maint:.0f}h")
                    
                    with col3:
                        st.metric("Efficacité", f"{sys['performance']['efficiency']:.0%}")
                        st.metric("Fiabilité", f"{sys['performance']['reliability']:.0%}")
                    
                    st.markdown("---")
                    
                    # Diagnostics
                    st.write("**🔬 Diagnostics:**")
                    
                    issues = []
                    
                    if sys['cooling']['temperature'] > sys['critical_temperature'] * 0.9:
                        issues.append("⚠️ Température proche de Tc")
                    
                    if sys['magnetic_properties']['field_stability'] < 0.9:
                        issues.append("⚠️ Stabilité magnétique faible")
                    
                    if sys['health'] < 0.9:
                        issues.append("⚠️ Santé système dégradée")
                    
                    if sys['operational_hours'] > 1000:
                        issues.append("🔧 Maintenance programmée nécessaire")
                    
                    if issues:
                        for issue in issues:
                            st.warning(issue)
                    else:
                        st.success("✅ Aucun problème détecté")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🔧 Maintenance", key=f"maint_btn_{sys_id}"):
                            sys['health'] = min(1.0, sys['health'] + 0.1)
                            sys['operational_hours'] = 0
                            st.success("✅ Maintenance effectuée!")
                            st.rerun()
                    
                    with col2:
                        if st.button("🔬 Diagnostic Complet", key=f"diag_btn_{sys_id}"):
                            st.info("Diagnostic lancé...")
                    
                    with col3:
                        if st.button("📊 Rapport", key=f"report_btn_{sys_id}"):
                            st.info("Génération du rapport...")
        
        with tab2:
            st.subheader("🔧 Maintenance Préventive")
            
            st.write("### 📅 Planning de Maintenance")
            
            # Créer un planning
            maintenance_schedule = []
            
            for sys in st.session_state.superconductor_system['systems'].values():
                hours_until_maint = max(0, 1000 - sys['operational_hours'])
                days_until_maint = int(hours_until_maint / 24)
                
                maintenance_schedule.append({
                    'Système': sys['name'],
                    'Santé': f"{sys['health']:.0%}",
                    'Heures Op.': f"{sys['operational_hours']:.0f}h",
                    'Prochaine Maint.': f"{days_until_maint}j",
                    'Priorité': 'Haute' if days_until_maint < 7 else 'Moyenne' if days_until_maint < 30 else 'Basse'
                })
            
            df = pd.DataFrame(maintenance_schedule)
            df = df.sort_values('Prochaine Maint.')
            
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            
            st.write("### 📋 Checklist Maintenance Préventive")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔧 Maintenance Niveau 1 (Mensuelle):**")
                tasks_l1 = [
                    "Inspection visuelle générale",
                    "Vérification niveaux cryogènes",
                    "Contrôle capteurs température",
                    "Test alarmes sécurité",
                    "Nettoyage extérieur",
                    "Vérification connexions électriques"
                ]
                
                for task in tasks_l1:
                    st.checkbox(task, key=f"l1_{task}")
            
            with col2:
                st.write("**🔧 Maintenance Niveau 2 (Trimestrielle):**")
                tasks_l2 = [
                    "Calibration capteurs",
                    "Test systèmes refroidissement",
                    "Inspection isolation thermique",
                    "Vérification bobines magnétiques",
                    "Test systèmes secours",
                    "Mise à jour logiciels"
                ]
                
                for task in tasks_l2:
                    st.checkbox(task, key=f"l2_{task}")
            
            st.markdown("---")
            
            st.write("**🔧 Maintenance Niveau 3 (Annuelle):**")
            tasks_l3 = [
                "Révision complète système cryogénique",
                "Test charge maximale",
                "Recalibration complète",
                "Remplacement préventif composants",
                "Audit sécurité complet",
                "Certification annuelle"
            ]
            
            for task in tasks_l3:
                st.checkbox(task, key=f"l3_{task}")
        
        with tab3:
            st.subheader("🚨 Maintenance Corrective")
            
            st.write("### 🆘 Interventions d'Urgence")
            
            # Incidents simulés
            incidents = [
                {
                    "id": "INC001",
                    "système": "SuperMag-Alpha",
                    "gravité": "Critique",
                    "type": "Surchauffe cryostat",
                    "statut": "Résolu",
                    "date": "2025-10-10"
                },
                {
                    "id": "INC002",
                    "système": "MagLev-Beta",
                    "gravité": "Moyenne",
                    "type": "Instabilité champ magnétique",
                    "statut": "En cours",
                    "date": "2025-10-12"
                },
                {
                    "id": "INC003",
                    "système": "Quantum-Gamma",
                    "gravité": "Faible",
                    "type": "Dérive température",
                    "statut": "Nouveau",
                    "date": "2025-10-13"
                }
            ]
            
            for incident in incidents:
                severity_color = "🔴" if incident['gravité'] == 'Critique' else "🟡" if incident['gravité'] == 'Moyenne' else "🟢"
                status_icon = "✅" if incident['statut'] == 'Résolu' else "⏳" if incident['statut'] == 'En cours' else "🆕"
                
                with st.expander(f"{severity_color} {incident['id']} - {incident['système']} - {incident['type']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**ID:** {incident['id']}")
                        st.write(f"**Système:** {incident['système']}")
                    
                    with col2:
                        st.write(f"**Type:** {incident['type']}")
                        st.write(f"**Gravité:** {incident['gravité']}")
                    
                    with col3:
                        st.write(f"**Statut:** {status_icon} {incident['statut']}")
                        st.write(f"**Date:** {incident['date']}")
                    
                    with col4:
                        if incident['statut'] != 'Résolu':
                            if st.button("🔧 Intervenir", key=f"fix_{incident['id']}"):
                                st.success("Intervention lancée!")
                    
                    st.markdown("---")
                    
                    st.write("**📝 Description:**")
                    st.info("Détection d'une anomalie nécessitant une intervention rapide.")
                    
                    st.write("**🔧 Actions Correctives:**")
                    if incident['statut'] == 'Résolu':
                        st.success("✅ Problème résolu - Système opérationnel")
                    else:
                        st.warning("⏳ Intervention en cours")
                    
                    if st.text_area("Ajouter un commentaire", key=f"comment_{incident['id']}"):
                        if st.button("💾 Sauvegarder", key=f"save_{incident['id']}"):
                            st.success("Commentaire sauvegardé!")
            
            st.markdown("---")
            
            st.write("### ➕ Signaler un Nouveau Problème")
            
            with st.form("new_incident"):
                col1, col2 = st.columns(2)
                
                with col1:
                    sys_options = {s['id']: s['name'] for s in st.session_state.superconductor_system['systems'].values()}
                    incident_system = st.selectbox("Système Concerné", options=list(sys_options.keys()),
                                                   format_func=lambda x: sys_options[x])
                    
                    incident_type = st.selectbox("Type de Problème", 
                                                ["Surchauffe", "Fuite Cryogène", "Instabilité Magnétique",
                                                 "Panne Électrique", "Défaillance Capteur", "Autre"])
                
                with col2:
                    severity = st.selectbox("Gravité", ["Faible", "Moyenne", "Haute", "Critique"])
                    priority = st.selectbox("Priorité", ["P4 - Routine", "P3 - Normal", "P2 - Urgent", "P1 - Critique"])
                
                description = st.text_area("Description Détaillée du Problème")
                
                immediate_action = st.text_area("Actions Immédiates Prises")
                
                submitted = st.form_submit_button("🚨 Signaler l'Incident", use_container_width=True, type="primary")
                
                if submitted:
                    incident_id = f"INC{len(incidents)+1:03d}"
                    st.success(f"✅ Incident {incident_id} créé et assigné!")
                    log_event(f"Incident signalé: {incident_id} - {sys_options[incident_system]}")
        
        with tab4:
            st.subheader("📊 Historique de Maintenance")
            
            # Données d'historique simulées
            history_data = []
            
            for i in range(15):
                days_ago = np.random.randint(1, 90)
                date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
                
                system_name = np.random.choice(list(st.session_state.superconductor_system['systems'].values()))['name'] if st.session_state.superconductor_system['systems'] else "Système Test"
                
                maint_type = np.random.choice(['Préventive', 'Corrective', 'Prédictive'])
                status = np.random.choice(['Complété', 'Complété', 'Complété', 'En cours'])
                
                history_data.append({
                    'Date': date,
                    'Système': system_name,
                    'Type': maint_type,
                    'Durée': f"{np.random.randint(1, 8)}h",
                    'Technicien': f"Tech-{np.random.randint(1, 5)}",
                    'Statut': status,
                    'Coût': f"${np.random.randint(500, 5000):,}"
                })
            
            df = pd.DataFrame(history_data)
            df = df.sort_values('Date', ascending=False)
            
            # Filtres
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.multiselect("Type", ["Préventive", "Corrective", "Prédictive"], key="hist_type")
            with col2:
                filter_status = st.multiselect("Statut", ["Complété", "En cours"], key="hist_status")
            with col3:
                date_range = st.slider("Derniers jours", 7, 90, 30)
            
            # Appliquer filtres
            if filter_type:
                df = df[df['Type'].isin(filter_type)]
            if filter_status:
                df = df[df['Statut'].isin(filter_status)]
            
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            
            # Statistiques
            st.write("### 📈 Statistiques de Maintenance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Interventions", len(df))
            
            with col2:
                preventive_pct = len(df[df['Type'] == 'Préventive']) / len(df) * 100 if len(df) > 0 else 0
                st.metric("Préventive", f"{preventive_pct:.0f}%")
            
            with col3:
                completed_pct = len(df[df['Statut'] == 'Complété']) / len(df) * 100 if len(df) > 0 else 0
                st.metric("Taux Complétion", f"{completed_pct:.0f}%")
            
            with col4:
                avg_duration = df['Durée'].str.replace('h', '').astype(float).mean()
                st.metric("Durée Moyenne", f"{avg_duration:.1f}h")
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                type_counts = df['Type'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                            title="Répartition par Type de Maintenance",
                            color_discrete_sequence=px.colors.sequential.Blues_r)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Tendance temporelle
                df_temp = df.copy()
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                df_temp = df_temp.sort_values('Date')
                df_temp['Mois'] = df_temp['Date'].dt.to_period('M').astype(str)
                monthly_counts = df_temp.groupby('Mois').size().reset_index(name='Interventions')
                
                fig = px.line(monthly_counts, x='Mois', y='Interventions',
                             title="Tendance des Interventions",
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Export
            if st.button("📥 Exporter l'Historique (CSV)", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Télécharger CSV",
                    data=csv,
                    file_name=f"maintenance_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# ==================== LOG D'ÉVÉNEMENTS ====================

st.markdown("---")

with st.expander("📜 Journal des Événements (Dernières 10 entrées)"):
    if st.session_state.superconductor_system['log']:
        for event in st.session_state.superconductor_system['log'][-10:][::-1]:
            timestamp = event['timestamp'][:19]
            st.text(f"{timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")
    
    if st.button("🗑️ Effacer le Journal"):
        st.session_state.superconductor_system['log'] = []
        st.rerun()


# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🧲 Plateforme Supraconducteur-Magnétique-IA</h3>
        <p>Système Complet de Création, Fabrication et Déploiement</p>
        <p><small>Version 1.0.0 | Architecture IA-Quantique-Biologique</small></p>
        <p><small>🧲 Supraconducteurs | 🚁 Lévitation Magnétique | 📡 Amplificateurs</small></p>
        <p><small>⚛️ Quantum | 🧬 Biological | 🤖 AI Integration</small></p>
    </div>
""", unsafe_allow_html=True)