"""
Interface Streamlit pour la Plateforme Robotique Complète
Système intégré pour créer, développer, fabriquer, tester et déployer
tous types de robots avec IA, Quantique et Systèmes Biologiques
streamlit run robotique_app.py
"""

import hashlib
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
    page_title="🤖 Plateforme Robotique Complète",
    page_icon="🤖",
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
    .robot-card {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .type-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.3rem;
    }
    .humanoid {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .industrial {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .mobile {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .aerial {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    .medical {
        background: linear-gradient(90deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .component-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .ai-badge {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .quantum-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .bio-badge {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION SESSION STATE ====================

if 'robotics_system' not in st.session_state:
    st.session_state.robotics_system = {
        'robots': {},
        'simulations': [],
        'projects': {},
        'experiments': [],
        'manufacturing': [],
        'tests': [],
        'deployments': {},
        'training_data': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str):
    """Enregistre un événement"""
    st.session_state.robotics_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })

def get_type_badge(robot_type: str) -> str:
    """Retourne un badge HTML selon le type de robot"""
    badges = {
        'humanoide': '<span class="type-badge humanoid">🦾 HUMANOÏDE</span>',
        'industriel': '<span class="type-badge industrial">🏭 INDUSTRIEL</span>',
        'mobile': '<span class="type-badge mobile">🚗 MOBILE</span>',
        'aerien': '<span class="type-badge aerial">🚁 AÉRIEN</span>',
        'medical': '<span class="type-badge medical">⚕️ MÉDICAL</span>',
        'agricole': '<span class="type-badge mobile">🌾 AGRICOLE</span>',
    }
    return badges.get(robot_type, '<span class="type-badge">🤖 ROBOT</span>')

def create_robot_mock(name, robot_type, config):
    """Crée un robot simulé"""
    robot_id = f"robot_{len(st.session_state.robotics_system['robots']) + 1}"
    
    robot = {
        'id': robot_id,
        'name': name,
        'type': robot_type,
        'created_at': datetime.now().isoformat(),
        'status': 'offline',
        'health': 1.0,
        'specifications': {
            'dimensions': config.get('dimensions', [500, 500, 500]),
            'weight': config.get('weight', 10.0),
            'payload': config.get('payload', 5.0),
            'dof': config.get('dof', 6)
        },
        'components': {
            'actuators': config.get('n_actuators', 6),
            'sensors': config.get('n_sensors', 5),
            'controllers': 1
        },
        'power': {
            'source': config.get('power_source', 'batterie'),
            'capacity': config.get('battery_capacity', 1000.0),
            'charge': 100.0,
            'consumption': config.get('power_consumption', 100.0),
            'autonomy': config.get('battery_capacity', 1000.0) / config.get('power_consumption', 100.0)
        },
        'performance': {
            'max_speed': config.get('max_speed', 1.0),
            'precision': config.get('precision', 0.1),
            'repeatability': 0.05,
            'reach': config.get('reach', 1000.0)
        },
        'intelligence': {
            'ai_enabled': config.get('ai_enabled', False),
            'ai_type': config.get('ai_type', 'deep_learning'),
            'level': config.get('intelligence', 0.5),
            'autonomy': config.get('autonomy', 0.5),
            'learning': config.get('learning', False)
        },
        'advanced_systems': {
            'quantum': config.get('quantum_enabled', False),
            'n_qubits': config.get('n_qubits', 0),
            'biological': config.get('bio_enabled', False),
            'bio_type': config.get('bio_type', '')
        },
        'operations': {
            'hours': 0.0,
            'missions': 0,
            'success_rate': 100.0
        },
        'costs': {
            'development': config.get('dev_cost', 100000),
            'manufacturing': config.get('mfg_cost', 30000),
            'operational_per_hour': 10.0
        }
    }
    
    st.session_state.robotics_system['robots'][robot_id] = robot
    log_event(f"Robot créé: {name} ({robot_type})")
    return robot_id

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">🤖 Plateforme Robotique Complète - IA Quantique Biologique</h1>', unsafe_allow_html=True)
st.markdown("### Système Intégré pour Créer, Développer, Fabriquer et Déployer Tous Types de Robots")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=RoboTech+Lab", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Tableau de Bord",
            "🤖 Mes Robots",
            "➕ Créer Robot",
            "🧠 Intelligence Artificielle",
            "⚛️ Système Quantique",
            "🧬 Systèmes Biologiques",
            "🔧 Composants & Actionneurs",
            "📡 Capteurs & Perception",
            "🎮 Contrôle & Commande",
            "🔬 Simulations",
            "🧪 Expériences & Tests",
            "🏭 Fabrication",
            "⚙️ Assemblage",
            "📊 Analyses & Résultats",
            "🚀 Déploiement",
            "📁 Projets",
            "🎓 Formation & IA",
            "💰 Coûts & ROI",
            "📚 Bibliothèque",
            "🌟 Applications"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_robots = len(st.session_state.robotics_system['robots'])
    active_robots = sum(1 for r in st.session_state.robotics_system['robots'].values() if r['status'] == 'online')
    total_projects = len(st.session_state.robotics_system['projects'])
    total_sims = len(st.session_state.robotics_system['simulations'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🤖 Robots", total_robots)
        st.metric("📁 Projets", total_projects)
    with col2:
        st.metric("✅ Actifs", active_robots)
        st.metric("🔬 Simulations", total_sims)

# ==================== PAGE: TABLEAU DE BORD ====================

if page == "🏠 Tableau de Bord":
    st.header("📊 Tableau de Bord Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="robot-card"><h2>🤖</h2><h3>{total_robots}</h3><p>Robots Totaux</p></div>', unsafe_allow_html=True)
    
    with col2:
        ai_robots = sum(1 for r in st.session_state.robotics_system['robots'].values() if r['intelligence']['ai_enabled'])
        st.markdown(f'<div class="robot-card"><h2>🧠</h2><h3>{ai_robots}</h3><p>Avec IA</p></div>', unsafe_allow_html=True)
    
    with col3:
        quantum_robots = sum(1 for r in st.session_state.robotics_system['robots'].values() if r['advanced_systems']['quantum'])
        st.markdown(f'<div class="robot-card"><h2>⚛️</h2><h3>{quantum_robots}</h3><p>Quantiques</p></div>', unsafe_allow_html=True)
    
    with col4:
        bio_robots = sum(1 for r in st.session_state.robotics_system['robots'].values() if r['advanced_systems']['biological'])
        st.markdown(f'<div class="robot-card"><h2>🧬</h2><h3>{bio_robots}</h3><p>Biologiques</p></div>', unsafe_allow_html=True)
    
    with col5:
        total_missions = sum(r['operations']['missions'] for r in st.session_state.robotics_system['robots'].values())
        st.markdown(f'<div class="robot-card"><h2>🎯</h2><h3>{total_missions}</h3><p>Missions</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.robotics_system['robots']:
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Répartition par Type")
            
            type_counts = {}
            for robot in st.session_state.robotics_system['robots'].values():
                r_type = robot['type'].replace('_', ' ').title()
                type_counts[r_type] = type_counts.get(r_type, 0) + 1
            
            fig = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                        color_discrete_sequence=px.colors.sequential.Purples_r)
            fig.update_layout(title="Types de Robots")
            st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{robot}")
        
        with col2:
            st.subheader("🧠 Niveaux d'Intelligence")
            
            names = [r['name'][:15] for r in st.session_state.robotics_system['robots'].values()]
            intelligence = [r['intelligence']['level'] * 100 for r in st.session_state.robotics_system['robots'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=intelligence, marker_color='rgb(102, 126, 234)')
            ])
            fig.update_layout(title="Niveau d'Intelligence (%)", yaxis_title="Intelligence", xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, key='niveau')
        
        st.markdown("---")
        
        # Robots actifs
        st.subheader("🤖 Robots Actifs")
        
        active = {k: v for k, v in st.session_state.robotics_system['robots'].items() if v['status'] == 'online'}
        
        if active:
            for robot_id, robot in active.items():
                with st.expander(f"🤖 {robot['name']} - {robot['type'].replace('_', ' ').title()}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Type", robot['type'].replace('_', ' ').title())
                        st.metric("Poids", f"{robot['specifications']['weight']:.1f} kg")
                    
                    with col2:
                        st.metric("DoF", robot['specifications']['dof'])
                        st.metric("Charge Batterie", f"{robot['power']['charge']:.0f}%")
                    
                    with col3:
                        st.metric("Intelligence", f"{robot['intelligence']['level']:.0%}")
                        st.metric("Missions", robot['operations']['missions'])
                    
                    with col4:
                        st.metric("Santé", f"{robot['health']:.0%}")
                        st.metric("Taux Succès", f"{robot['operations']['success_rate']:.0f}%")
        else:
            st.info("Aucun robot actif")
    else:
        st.info("💡 Aucun robot créé. Créez votre premier robot!")

# ==================== PAGE: MES ROBOTS ====================

elif page == "🤖 Mes Robots":
    st.header("🤖 Gestion des Robots")
    
    if not st.session_state.robotics_system['robots']:
        st.info("💡 Aucun robot créé.")
    else:
        for robot_id, robot in st.session_state.robotics_system['robots'].items():
            st.markdown(f'<div class="robot-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### 🤖 {robot['name']}")
                st.markdown(get_type_badge(robot['type']), unsafe_allow_html=True)
                
                # Badges avancés
                if robot['intelligence']['ai_enabled']:
                    st.markdown('<span class="ai-badge">🧠 IA</span>', unsafe_allow_html=True)
                if robot['advanced_systems']['quantum']:
                    st.markdown('<span class="quantum-badge">⚛️ QUANTIQUE</span>', unsafe_allow_html=True)
                if robot['advanced_systems']['biological']:
                    st.markdown('<span class="bio-badge">🧬 BIO</span>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Poids", f"{robot['specifications']['weight']:.1f} kg")
                st.metric("DoF", robot['specifications']['dof'])
            
            with col3:
                st.metric("Intelligence", f"{robot['intelligence']['level']:.0%}")
                st.metric("Autonomie", f"{robot['power']['autonomy']:.1f}h")
            
            with col4:
                status_icon = "🟢" if robot['status'] == 'online' else "🔴"
                st.write(f"**Statut:** {status_icon} {robot['status'].upper()}")
                st.write(f"**Santé:** {robot['health']:.0%}")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["⚙️ Spécifications", "🔋 Énergie", "🧠 Intelligence", "🔧 Composants", "📊 Opérations"])
                
                with tab1:
                    st.subheader("⚙️ Spécifications Techniques")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Longueur", f"{robot['specifications']['dimensions'][0]} mm")
                    with col2:
                        st.metric("Largeur", f"{robot['specifications']['dimensions'][1]} mm")
                    with col3:
                        st.metric("Hauteur", f"{robot['specifications']['dimensions'][2]} mm")
                    with col4:
                        st.metric("Charge Utile", f"{robot['specifications']['payload']:.1f} kg")
                    
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Vitesse Max", f"{robot['performance']['max_speed']:.2f} m/s")
                    with col2:
                        st.metric("Précision", f"{robot['performance']['precision']:.2f} mm")
                    with col3:
                        st.metric("Répétabilité", f"{robot['performance']['repeatability']:.3f} mm")
                
                with tab2:
                    st.subheader("🔋 Système d'Alimentation")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Source:** {robot['power']['source'].title()}")
                        st.metric("Capacité", f"{robot['power']['capacity']:.0f} Wh")
                        st.metric("Charge Actuelle", f"{robot['power']['charge']:.0f}%")
                        st.progress(robot['power']['charge'] / 100)
                    
                    with col2:
                        st.metric("Consommation", f"{robot['power']['consumption']:.0f} W")
                        st.metric("Autonomie", f"{robot['power']['autonomy']:.1f}h")
                        
                        # Graphique de décharge
                        time = np.linspace(0, robot['power']['autonomy'], 100)
                        charge = 100 * (1 - time / robot['power']['autonomy'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=time, y=charge, mode='lines',
                                                line=dict(color='green', width=3)))
                        fig.update_layout(title="Courbe de Décharge", xaxis_title="Temps (h)",
                                        yaxis_title="Charge (%)", height=250)
                        st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{robot_id}")
                
                with tab3:
                    st.subheader("🧠 Systèmes d'Intelligence")
                    
                    if robot['intelligence']['ai_enabled']:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type IA:** {robot['intelligence']['ai_type'].replace('_', ' ').title()}")
                            st.metric("Niveau Intelligence", f"{robot['intelligence']['level']:.0%}")
                            st.metric("Niveau Autonomie", f"{robot['intelligence']['autonomy']:.0%}")
                        
                        with col2:
                            st.write(f"**Apprentissage:** {'✅ Activé' if robot['intelligence']['learning'] else '❌ Désactivé'}")
                            
                            # Capacités
                            st.write("**Capacités:**")
                            capacities = {
                                'Perception': 0.85,
                                'Décision': 0.78,
                                'Apprentissage': 0.92,
                                'Adaptation': 0.88,
                                'Raisonnement': 0.75
                            }
                            
                            for cap, val in capacities.items():
                                st.write(f"• {cap}: {val:.0%}")
                    else:
                        st.info("IA non activée sur ce robot")
                    
                    st.markdown("---")
                    
                    # Systèmes avancés
                    if robot['advanced_systems']['quantum']:
                        st.write("### ⚛️ Processeur Quantique")
                        st.success(f"✅ QPU avec {robot['advanced_systems']['n_qubits']} qubits")
                        st.write("• Optimisation quantique")
                        st.write("• Capteurs quantiques")
                        st.write("• Cryptographie quantique")
                    
                    if robot['advanced_systems']['biological']:
                        st.write("### 🧬 Système Biologique")
                        st.success(f"✅ Interface {robot['advanced_systems']['bio_type']}")
                        st.write("• Auto-réparation")
                        st.write("• Adaptation biologique")
                        st.write("• Capteurs biologiques")
                
                with tab4:
                    st.subheader("🔧 Composants Installés")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Actionneurs", robot['components']['actuators'])
                    with col2:
                        st.metric("Capteurs", robot['components']['sensors'])
                    with col3:
                        st.metric("Contrôleurs", robot['components']['controllers'])
                
                with tab5:
                    st.subheader("📊 Opérations")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Heures Opération", f"{robot['operations']['hours']:.1f}h")
                    with col2:
                        st.metric("Missions Complétées", robot['operations']['missions'])
                    with col3:
                        st.metric("Taux de Succès", f"{robot['operations']['success_rate']:.0f}%")
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button(f"▶️ {'Éteindre' if robot['status'] == 'online' else 'Activer'}", key=f"toggle_{robot_id}"):
                        robot['status'] = 'offline' if robot['status'] == 'online' else 'online'
                        log_event(f"{robot['name']} {'éteint' if robot['status'] == 'offline' else 'activé'}")
                        st.rerun()
                
                with col2:
                    if st.button(f"🔬 Simuler", key=f"sim_{robot_id}"):
                        st.info("Allez dans Simulations")
                
                with col3:
                    if st.button(f"🧪 Tester", key=f"test_{robot_id}"):
                        st.info("Allez dans Tests")
                
                with col4:
                    if st.button(f"🔧 Diagnostiquer", key=f"diag_{robot_id}"):
                        if robot['health'] < 0.95:
                            st.warning(f"⚠️ Santé: {robot['health']:.0%}")
                        else:
                            st.success("✅ Robot en bon état")
                
                with col5:
                    if st.button(f"🗑️ Supprimer", key=f"del_{robot_id}"):
                        del st.session_state.robotics_system['robots'][robot_id]
                        log_event(f"{robot['name']} supprimé")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CRÉER ROBOT ====================

elif page == "➕ Créer Robot":
    st.header("➕ Créer un Nouveau Robot")
    
    with st.form("create_robot_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            robot_name = st.text_input("📝 Nom du Robot", placeholder="Ex: Atlas-Pro-X1")
            
            robot_type = st.selectbox(
                "🤖 Type de Robot",
                [
                    "humanoide",
                    "industriel",
                    "mobile",
                    "aerien",
                    "aquatique",
                    "medical",
                    "agricole",
                    "spatial",
                    "nano",
                    "essaim",
                    "mou",
                    "bio_hybride",
                    "exosquelette",
                    "prothese",
                    "compagnon"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            application = st.selectbox(
                "🎯 Application Principale",
                ["Industrie", "Médical", "Recherche", "Service", "Agriculture", 
                 "Exploration", "Militaire", "Domestique", "Education"]
            )
            
            environment = st.multiselect(
                "🌍 Environnement d'Utilisation",
                ["Intérieur", "Extérieur", "Sous-marin", "Aérien", "Spatial", "Extrême"]
            )
        
        st.markdown("---")
        st.subheader("📐 Spécifications Physiques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            length = st.number_input("Longueur (mm)", 100, 10000, 1000, 50)
            width = st.number_input("Largeur (mm)", 100, 10000, 600, 50)
            height = st.number_input("Hauteur (mm)", 100, 10000, 1500, 50)
        
        with col2:
            weight = st.number_input("Poids (kg)", 0.1, 10000.0, 50.0, 1.0)
            payload = st.number_input("Charge Utile (kg)", 0.0, 5000.0, 20.0, 1.0)
        
        with col3:
            dof = st.number_input("Degrés de Liberté", 1, 100, 12, 1)
            reach = st.number_input("Portée (mm)", 0, 5000, 1000, 50)
        
        st.markdown("---")
        st.subheader("🔧 Actionneurs et Mobilité")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            actuator_type = st.selectbox(
                "Type d'Actionneur",
                ["Moteur Électrique", "Servo", "Pas à Pas", "Hydraulique", 
                 "Pneumatique", "Mémoire de Forme", "Muscle Artificiel"]
            )
            n_actuators = st.number_input("Nombre d'Actionneurs", 1, 100, 12, 1)
        
        with col2:
            max_speed = st.number_input("Vitesse Max (m/s)", 0.01, 50.0, 1.5, 0.1)
            max_torque = st.number_input("Couple Max (Nm)", 1.0, 1000.0, 100.0, 10.0)
        
        with col3:
            precision = st.number_input("Précision (mm)", 0.001, 10.0, 0.1, 0.01)
            locomotion = st.selectbox(
                "Type de Locomotion",
                ["Roues", "Jambes", "Chenilles", "Flottant", "Volant", "Hybride"]
            )
        
        st.markdown("---")
        st.subheader("📡 Capteurs et Perception")
        
        sensors_config = st.multiselect(
            "Capteurs à Installer",
            ["Caméra RGB", "Caméra Profondeur", "LiDAR", "Radar", "Ultrason", 
             "IMU", "GPS", "Force/Couple", "Tactile", "Température", "Chimique", "Biologique"],
            default=["Caméra RGB", "LiDAR", "IMU"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            vision_resolution = st.selectbox("Résolution Caméra", ["640x480", "1920x1080", "3840x2160"])
            lidar_range = st.slider("Portée LiDAR (m)", 1, 200, 50, 1)
        
        with col2:
            sensor_frequency = st.slider("Fréquence Capteurs (Hz)", 10, 1000, 100, 10)
            sensor_redundancy = st.checkbox("Redondance des Capteurs", value=True)
        
        st.markdown("---")
        st.subheader("🧠 Intelligence Artificielle")
        
        ai_enabled = st.checkbox("Activer l'Intelligence Artificielle", value=True)
        
        if ai_enabled:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ai_type = st.selectbox(
                    "Type d'IA",
                    ["Deep Learning", "Reinforcement Learning", "Swarm Intelligence", 
                     "Evolutionary", "Neuromorphic", "Quantum ML", "Hybrid AI"]
                )
            
            with col2:
                intelligence_level = st.slider("Niveau d'Intelligence", 0.0, 1.0, 0.7, 0.05)
                autonomy_level = st.slider("Niveau d'Autonomie", 0.0, 1.0, 0.6, 0.05)
            
            with col3:
                learning_enabled = st.checkbox("Apprentissage Continu", value=True)
                transfer_learning = st.checkbox("Transfer Learning", value=True)
                
            # Architecture réseau
            st.write("**Architecture du Réseau de Neurones:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_layers = st.number_input("Nombre de Couches", 3, 100, 10, 1)
            with col2:
                n_neurons = st.number_input("Neurones par Couche", 10, 10000, 256, 10)
            with col3:
                activation = st.selectbox("Fonction d'Activation", ["ReLU", "Tanh", "Sigmoid", "Leaky ReLU"])
        
        st.markdown("---")
        st.subheader("⚛️ Système Quantique")
        
        quantum_enabled = st.checkbox("Intégrer Processeur Quantique", value=False)
        
        if quantum_enabled:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_qubits = st.slider("Nombre de Qubits", 2, 100, 20, 1)
                coherence_time = st.number_input("Temps de Cohérence (μs)", 10, 1000, 100, 10)
            
            with col2:
                gate_fidelity = st.slider("Fidélité des Portes", 0.90, 0.9999, 0.99, 0.0001)
                quantum_apps = st.multiselect(
                    "Applications Quantiques",
                    ["Optimisation", "Sensing", "Communication", "Machine Learning"],
                    default=["Optimisation", "Sensing"]
                )
            
            with col3:
                quantum_volume = 2 ** min(n_qubits, 10)
                st.metric("Volume Quantique", quantum_volume)
                st.info("Le volume quantique mesure la complexité des circuits quantiques exécutables")
        
        st.markdown("---")
        st.subheader("🧬 Système Biologique")
        
        bio_enabled = st.checkbox("Intégrer Systèmes Biologiques", value=False)
        
        if bio_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                bio_type = st.selectbox(
                    "Type de Système Biologique",
                    ["Interface Neuronale", "Tissus Biologiques", "Capteurs Bio", 
                     "Muscles Biologiques", "Système Immunitaire Artificiel"]
                )
                
                bio_capabilities = st.multiselect(
                    "Capacités Biologiques",
                    ["Auto-réparation", "Adaptation", "Sensing Biologique", "Production d'Énergie"],
                    default=["Auto-réparation", "Adaptation"]
                )
            
            with col2:
                biocompatibility = st.slider("Biocompatibilité", 0.0, 1.0, 0.95, 0.01)
                cell_count = st.number_input("Nombre de Cellules (x10⁶)", 0, 1000, 100)
                
                st.info("""
                **Avantages des Systèmes Biologiques:**
                - Auto-réparation
                - Adaptation à l'environnement
                - Efficacité énergétique
                - Capteurs ultra-sensibles
                """)
        
        st.markdown("---")
        st.subheader("🔋 Système d'Alimentation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            power_source = st.selectbox(
                "Source d'Énergie",
                ["Batterie", "Solaire", "Pile à Combustible", "Nucléaire", 
                 "Biocarburant", "Sans Fil", "Hybride", "Quantique"]
            )
            
            battery_capacity = st.number_input("Capacité Batterie (Wh)", 10, 100000, 2000, 100)
        
        with col2:
            power_consumption = st.number_input("Consommation (W)", 1, 10000, 200, 10)
            charging_time = st.number_input("Temps de Charge (h)", 0.5, 24.0, 2.0, 0.5)
        
        with col3:
            autonomy = battery_capacity / power_consumption if power_consumption > 0 else 0
            st.metric("Autonomie Calculée", f"{autonomy:.1f}h")
            
            fast_charging = st.checkbox("Charge Rapide", value=True)
            wireless_charging = st.checkbox("Charge Sans Fil", value=False)
        
        st.markdown("---")
        st.subheader("🛡️ Sécurité et Redondance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            safety_features = st.multiselect(
                "Fonctionnalités de Sécurité",
                ["Arrêt d'Urgence", "Évitement de Collision", "Détection de Chute", 
                 "Limiteur de Force", "Surveillance Santé", "Mode Sécurisé"],
                default=["Arrêt d'Urgence", "Évitement de Collision"]
            )
        
        with col2:
            redundancy_level = st.slider("Niveau de Redondance", 0, 3, 1)
            fail_safe = st.checkbox("Mode Fail-Safe", value=True)
            
            st.info(f"""
            **Niveau de Redondance: {redundancy_level}**
            - 0: Aucune redondance
            - 1: Capteurs redondants
            - 2: Actionneurs + Capteurs redondants
            - 3: Système complet redondant
            """)
        
        st.markdown("---")
        st.subheader("💰 Estimation des Coûts")
        
        # Calcul automatique des coûts
        base_cost = 50000
        
        # Coûts des composants
        actuator_cost = n_actuators * 2000
        sensor_cost = len(sensors_config) * 1500
        ai_cost = 30000 if ai_enabled else 0
        quantum_cost = 200000 if quantum_enabled else 0
        bio_cost = 100000 if bio_enabled else 0
        
        dev_cost = base_cost + actuator_cost + sensor_cost + ai_cost + quantum_cost + bio_cost
        mfg_cost = dev_cost * 0.3
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Coût Développement", f"${dev_cost:,}")
        with col2:
            st.metric("Coût Fabrication", f"${mfg_cost:,}")
        with col3:
            st.metric("Coût Total", f"${(dev_cost + mfg_cost):,}")
        
        submitted = st.form_submit_button("🚀 Créer le Robot", use_container_width=True, type="primary")
        
        if submitted:
            if not robot_name:
                st.error("⚠️ Veuillez donner un nom au robot")
            else:
                with st.spinner("🔄 Création du robot en cours..."):
                    config = {
                        'dimensions': [length, width, height],
                        'weight': weight,
                        'payload': payload,
                        'dof': dof,
                        'reach': reach,
                        'n_actuators': n_actuators,
                        'n_sensors': len(sensors_config),
                        'max_speed': max_speed,
                        'precision': precision,
                        'power_source': power_source.lower(),
                        'battery_capacity': battery_capacity,
                        'power_consumption': power_consumption,
                        'ai_enabled': ai_enabled,
                        'ai_type': ai_type.lower().replace(' ', '_') if ai_enabled else '',
                        'intelligence': intelligence_level if ai_enabled else 0.0,
                        'autonomy': autonomy_level if ai_enabled else 0.0,
                        'learning': learning_enabled if ai_enabled else False,
                        'quantum_enabled': quantum_enabled,
                        'n_qubits': n_qubits if quantum_enabled else 0,
                        'bio_enabled': bio_enabled,
                        'bio_type': bio_type.lower().replace(' ', '_') if bio_enabled else '',
                        'dev_cost': dev_cost,
                        'mfg_cost': mfg_cost
                    }
                    
                    robot_id = create_robot_mock(robot_name, robot_type, config)
                    
                    st.success(f"✅ Robot '{robot_name}' créé avec succès!")
                    st.balloons()
                    
                    robot = st.session_state.robotics_system['robots'][robot_id]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Poids", f"{robot['specifications']['weight']:.1f} kg")
                    with col2:
                        st.metric("DoF", robot['specifications']['dof'])
                    with col3:
                        st.metric("Intelligence", f"{robot['intelligence']['level']:.0%}")
                    with col4:
                        st.metric("Autonomie", f"{robot['power']['autonomy']:.1f}h")
                    
                    st.code(f"ID: {robot_id}", language="text")

# ==================== PAGE: INTELLIGENCE ARTIFICIELLE ====================

elif page == "🧠 Intelligence Artificielle":
    st.header("🧠 Systèmes d'Intelligence Artificielle")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎓 Entraînement", "🔮 Inférence", "📊 Performance", "🧬 Architectures"])
    
    with tab1:
        st.subheader("🎓 Entraînement des Modèles IA")
        
        if not st.session_state.robotics_system['robots']:
            st.warning("⚠️ Aucun robot disponible")
        else:
            # Sélection du robot
            ai_robots = {k: v for k, v in st.session_state.robotics_system['robots'].items() 
                        if v['intelligence']['ai_enabled']}
            
            if not ai_robots:
                st.info("Aucun robot avec IA disponible. Créez un robot avec IA activée.")
            else:
                robot_options = {r['id']: r['name'] for r in ai_robots.values()}
                selected_robot = st.selectbox(
                    "Sélectionner un Robot",
                    options=list(robot_options.keys()),
                    format_func=lambda x: robot_options[x]
                )
                
                robot = st.session_state.robotics_system['robots'][selected_robot]
                
                st.write(f"### 🤖 {robot['name']}")
                st.write(f"**Type IA:** {robot['intelligence']['ai_type'].replace('_', ' ').title()}")
                
                st.markdown("---")
                
                # Configuration de l'entraînement
                st.write("### ⚙️ Configuration de l'Entraînement")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    task_type = st.selectbox(
                        "Type de Tâche",
                        ["Classification", "Détection d'Objets", "Segmentation", 
                         "Navigation", "Manipulation", "Interaction Humaine"]
                    )
                    
                    dataset_size = st.number_input("Taille Dataset", 100, 1000000, 10000)
                
                with col2:
                    epochs = st.slider("Nombre d'Époques", 10, 1000, 100, 10)
                    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128, 256], index=2)
                
                with col3:
                    learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, 0.0001, format="%.4f")
                    optimizer = st.selectbox("Optimiseur", ["Adam", "SGD", "RMSprop", "AdaGrad"])
                
                # Augmentation de données
                st.markdown("---")
                st.write("**🔄 Augmentation de Données:**")
                
                augmentation = st.multiselect(
                    "Techniques d'Augmentation",
                    ["Rotation", "Flip", "Zoom", "Brightness", "Noise", "Elastic Transform"],
                    default=["Rotation", "Flip"]
                )
                
                if st.button("🚀 Lancer l'Entraînement", use_container_width=True, type="primary"):
                    with st.spinner("🔄 Entraînement en cours..."):
                        progress_bar = st.progress(0)
                        
                        # Simulation d'entraînement
                        training_history = {
                            'epoch': [],
                            'loss': [],
                            'accuracy': [],
                            'val_loss': [],
                            'val_accuracy': []
                        }
                        
                        for epoch in range(epochs):
                            progress_bar.progress((epoch + 1) / epochs)
                            
                            # Simulation des métriques
                            loss = 2.0 * np.exp(-epoch / 20) + 0.1 + np.random.random() * 0.05
                            accuracy = 1.0 - 0.5 * np.exp(-epoch / 15) + np.random.random() * 0.02
                            val_loss = loss * 1.1
                            val_accuracy = accuracy * 0.98
                            
                            training_history['epoch'].append(epoch)
                            training_history['loss'].append(loss)
                            training_history['accuracy'].append(accuracy)
                            training_history['val_loss'].append(val_loss)
                            training_history['val_accuracy'].append(val_accuracy)
                        
                        progress_bar.empty()
                        
                        st.success("✅ Entraînement terminé!")
                        
                        # Métriques finales
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy Finale", f"{training_history['accuracy'][-1]:.2%}")
                        with col2:
                            st.metric("Val Accuracy", f"{training_history['val_accuracy'][-1]:.2%}")
                        with col3:
                            st.metric("Loss Finale", f"{training_history['loss'][-1]:.4f}")
                        with col4:
                            st.metric("Temps Entraînement", f"{epochs * 2.5:.1f}s")
                        
                        # Graphiques
                        st.markdown("---")
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Loss", "Accuracy")
                        )
                        
                        # Loss
                        fig.add_trace(
                            go.Scatter(x=training_history['epoch'], y=training_history['loss'],
                                      mode='lines', name='Train Loss', line=dict(color='blue')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=training_history['epoch'], y=training_history['val_loss'],
                                      mode='lines', name='Val Loss', line=dict(color='red', dash='dash')),
                            row=1, col=1
                        )
                        
                        # Accuracy
                        fig.add_trace(
                            go.Scatter(x=training_history['epoch'], y=training_history['accuracy'],
                                      mode='lines', name='Train Acc', line=dict(color='green')),
                            row=1, col=2
                        )
                        fig.add_trace(
                            go.Scatter(x=training_history['epoch'], y=training_history['val_accuracy'],
                                      mode='lines', name='Val Acc', line=dict(color='orange', dash='dash')),
                            row=1, col=2
                        )
                        
                        fig.update_xaxes(title_text="Epoch", row=1, col=1)
                        fig.update_xaxes(title_text="Epoch", row=1, col=2)
                        fig.update_yaxes(title_text="Loss", row=1, col=1)
                        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
                        
                        fig.update_layout(height=400, showlegend=True)
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{epoch}")
                        
                        # Sauvegarder dans le training_data
                        st.session_state.robotics_system['training_data'].append({
                            'robot_id': selected_robot,
                            'task': task_type,
                            'epochs': epochs,
                            'accuracy': training_history['accuracy'][-1],
                            'history': training_history,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        log_event(f"Entraînement IA complété: {robot['name']} - {task_type}")
    
    with tab2:
        st.subheader("🔮 Inférence et Prédiction")
        
        if not st.session_state.robotics_system['training_data']:
            st.info("Aucun modèle entraîné disponible. Entraînez d'abord un modèle.")
        else:
            # Sélection du modèle
            model_options = {i: f"{data['robot_id'][:15]} - {data['task']} ({data['accuracy']:.1%})" 
                           for i, data in enumerate(st.session_state.robotics_system['training_data'])}
            
            selected_model = st.selectbox(
                "Sélectionner un Modèle Entraîné",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x]
            )
            
            model_data = st.session_state.robotics_system['training_data'][selected_model]
            
            st.write(f"### 🎯 Tâche: {model_data['task']}")
            st.write(f"**Accuracy:** {model_data['accuracy']:.2%}")
            
            st.markdown("---")
            
            # Input pour inférence
            st.write("### 📥 Données d'Entrée")
            
            if model_data['task'] == "Classification":
                uploaded_file = st.file_uploader("Charger une image", type=['jpg', 'png'])
                
                if uploaded_file:
                    st.image(uploaded_file, width=300)
                    
                    if st.button("🔮 Prédire"):
                        with st.spinner("Prédiction en cours..."):
                            # Simulation
                            classes = ["Objet A", "Objet B", "Objet C", "Objet D"]
                            confidences = np.random.dirichlet(np.ones(4)) * 100
                            
                            st.success("✅ Prédiction terminée!")
                            
                            # Résultat
                            predicted_class = classes[np.argmax(confidences)]
                            confidence = confidences[np.argmax(confidences)]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Classe Prédite", predicted_class)
                                st.metric("Confiance", f"{confidence:.1f}%")
                            
                            with col2:
                                # Graphique confidences
                                fig = go.Figure(data=[
                                    go.Bar(x=classes, y=confidences, marker_color='rgba(102, 126, 234, 0.7)')
                                ])
                                fig.update_layout(title="Confidences par Classe", 
                                                yaxis_title="Confiance (%)", height=300)
                                st.plotly_chart(fig, use_container_width=True)
            
            elif model_data['task'] == "Navigation":
                st.write("**Environnement de Navigation:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    start_x = st.number_input("Position X Départ", -10.0, 10.0, 0.0)
                    start_y = st.number_input("Position Y Départ", -10.0, 10.0, 0.0)
                
                with col2:
                    goal_x = st.number_input("Position X Objectif", -10.0, 10.0, 5.0)
                    goal_y = st.number_input("Position Y Objectif", -10.0, 10.0, 5.0)
                
                if st.button("🔮 Planifier Trajectoire"):
                    # Simulation de planification
                    t = np.linspace(0, 1, 50)
                    path_x = start_x + (goal_x - start_x) * t + np.random.random(50) * 0.2
                    path_y = start_y + (goal_y - start_y) * t + np.random.random(50) * 0.2
                    
                    fig = go.Figure()
                    
                    # Trajectoire
                    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines+markers',
                                            line=dict(color='blue', width=2),
                                            name='Trajectoire'))
                    
                    # Départ et objectif
                    fig.add_trace(go.Scatter(x=[start_x], y=[start_y], mode='markers',
                                            marker=dict(size=15, color='green'),
                                            name='Départ'))
                    fig.add_trace(go.Scatter(x=[goal_x], y=[goal_y], mode='markers',
                                            marker=dict(size=15, color='red'),
                                            name='Objectif'))
                    
                    fig.update_layout(title="Trajectoire Planifiée",
                                    xaxis_title="X (m)", yaxis_title="Y (m)",
                                    height=500)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"✅ Trajectoire planifiée: {len(path_x)} points")
    
    with tab3:
        st.subheader("📊 Performance des Modèles")
        
        if st.session_state.robotics_system['training_data']:
            # Comparaison des modèles
            st.write("### 📈 Comparaison des Modèles")
            
            models_df = pd.DataFrame([
                {
                    'Robot': data['robot_id'][:20],
                    'Tâche': data['task'],
                    'Époques': data['epochs'],
                    'Accuracy': f"{data['accuracy']:.2%}",
                    'Date': data['timestamp'][:10]
                }
                for data in st.session_state.robotics_system['training_data']
            ])
            
            st.dataframe(models_df, use_container_width=True)
            
            st.markdown("---")
            
            # Graphique accuracy
            accuracies = [data['accuracy'] * 100 for data in st.session_state.robotics_system['training_data']]
            tasks = [data['task'] for data in st.session_state.robotics_system['training_data']]
            
            fig = go.Figure(data=[
                go.Bar(x=tasks, y=accuracies, marker_color='rgba(102, 126, 234, 0.7)',
                      text=[f"{a:.1f}%" for a in accuracies], textposition='outside')
            ])
            
            fig.update_layout(title="Accuracy par Modèle",
                            xaxis_title="Tâche", yaxis_title="Accuracy (%)",
                            height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée d'entraînement disponible")
    
    with tab4:
        st.subheader("🧬 Architectures de Réseaux")
        
        st.write("### 🏗️ Architectures Pré-configurées")
        
        architectures = {
            "CNN - ConvNet": {
                "description": "Réseau convolutif pour vision par ordinateur",
                "layers": ["Conv2D(32)", "MaxPool", "Conv2D(64)", "MaxPool", "Dense(128)", "Output"],
                "parameters": "~2.5M",
                "use_case": "Classification d'images, détection d'objets",
                "accuracy": "95-98%"
            },
            "ResNet-50": {
                "description": "Architecture résiduelle profonde",
                "layers": ["Conv", "ResBlock x16", "AvgPool", "Dense"],
                "parameters": "~25M",
                "use_case": "Vision complexe, segmentation",
                "accuracy": "96-99%"
            },
            "LSTM - RNN": {
                "description": "Réseau récurrent pour séquences temporelles",
                "layers": ["LSTM(256)", "LSTM(128)", "Dense(64)", "Output"],
                "parameters": "~1.2M",
                "use_case": "Navigation, contrôle trajectoire",
                "accuracy": "90-95%"
            },
            "Transformer": {
                "description": "Architecture attention pour traitement de séquences",
                "layers": ["MultiHeadAttention", "FeedForward", "LayerNorm x12"],
                "parameters": "~110M",
                "use_case": "Compréhension langage, planification",
                "accuracy": "92-97%"
            },
            "GAN - Generative": {
                "description": "Réseau génératif adversaire",
                "layers": ["Generator", "Discriminator"],
                "parameters": "~5M",
                "use_case": "Génération données, simulation",
                "accuracy": "N/A"
            },
            "DQN - Reinforcement": {
                "description": "Deep Q-Network pour apprentissage par renforcement",
                "layers": ["Conv2D x3", "Dense(512)", "Q-values"],
                "parameters": "~3M",
                "use_case": "Contrôle autonome, jeux",
                "accuracy": "Reward-based"
            }
        }
        
        for arch_name, arch_info in architectures.items():
            with st.expander(f"🏗️ {arch_name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {arch_info['description']}")
                    st.write(f"**Cas d'Usage:** {arch_info['use_case']}")
                    st.write("**Couches:**")
                    for layer in arch_info['layers']:
                        st.write(f"  • {layer}")
                
                with col2:
                    st.metric("Paramètres", arch_info['parameters'])
                    st.metric("Accuracy Typique", arch_info['accuracy'])
                    
                    if st.button(f"📥 Utiliser", key=f"use_{arch_name}"):
                        st.success(f"Architecture {arch_name} sélectionnée!")
        
        st.markdown("---")
        
        # Créer architecture personnalisée
        st.write("### 🎨 Créer Architecture Personnalisée")
        
        col1, col2 = st.columns(2)
        
        with col1:
            arch_name_custom = st.text_input("Nom Architecture", "CustomNet-1")
            n_layers_custom = st.number_input("Nombre de Couches", 3, 50, 10)
        
        with col2:
            layer_type = st.selectbox("Type de Couche Principale", 
                                     ["Dense", "Conv2D", "LSTM", "Attention"])
            activation_func = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"])
        
        if st.button("🏗️ Créer Architecture"):
            st.success(f"✅ Architecture '{arch_name_custom}' créée avec {n_layers_custom} couches!")
            
            # Visualisation simplifiée
            layers_viz = []
            for i in range(n_layers_custom):
                size = 256 // (2 ** (i // 3))
                layers_viz.append(f"{layer_type}({size})")
            
            st.write("**Structure:**")
            st.code("\n".join(layers_viz))

# ==================== PAGE: SYSTÈME QUANTIQUE ====================

elif page == "⚛️ Système Quantique":
    st.header("⚛️ Processeurs Quantiques pour Robotique")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Configuration", "🔬 Circuits", "📊 Résultats", "🚀 Applications"])
    
    with tab1:
        st.subheader("🎯 Configuration du Processeur Quantique")
        
        quantum_robots = {k: v for k, v in st.session_state.robotics_system['robots'].items() 
                         if v['advanced_systems']['quantum']}
        
        if not quantum_robots:
            st.info("Aucun robot avec processeur quantique. Créez un robot avec système quantique activé.")
        else:
            robot_options = {r['id']: r['name'] for r in quantum_robots.values()}
            selected_robot = st.selectbox(
                "Sélectionner un Robot Quantique",
                options=list(robot_options.keys()),
                format_func=lambda x: robot_options[x]
            )
            
            robot = st.session_state.robotics_system['robots'][selected_robot]
            n_qubits = robot['advanced_systems']['n_qubits']
            
            st.write(f"### ⚛️ {robot['name']}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Nombre de Qubits", n_qubits)
            with col2:
                st.metric("Volume Quantique", 2 ** min(n_qubits, 10))
            with col3:
                st.metric("Fidélité des Portes", "99.0%")
            with col4:
                st.metric("Temps Cohérence", "100 µs")
            
            st.markdown("---")
            
            # Applications quantiques
            st.write("### 🎯 Applications Quantiques Disponibles")
            
            quantum_apps = {
                "Optimisation Quantique": {
                    "description": "Optimisation de trajectoires et planification",
                    "algorithm": "QAOA (Quantum Approximate Optimization Algorithm)",
                    "speedup": "Quadratique",
                    "qubits_required": 10
                },
                "Sensing Quantique": {
                    "description": "Capteurs ultra-précis basés sur effets quantiques",
                    "algorithm": "Ramsey Interferometry",
                    "speedup": "Exponentiel (sensibilité)",
                    "qubits_required": 5
                },
                "Machine Learning Quantique": {
                    "description": "Apprentissage quantique pour reconnaissance",
                    "algorithm": "Quantum Neural Networks",
                    "speedup": "Exponentiel (certains cas)",
                    "qubits_required": 15
                },
                "Communication Quantique": {
                    "description": "Communication sécurisée par cryptographie quantique",
                    "algorithm": "QKD (Quantum Key Distribution)",
                    "speedup": "Sécurité absolue",
                    "qubits_required": 2
                }
            }
            
            for app_name, app_info in quantum_apps.items():
                with st.expander(f"⚛️ {app_name}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {app_info['description']}")
                        st.write(f"**Algorithme:** {app_info['algorithm']}")
                    
                    with col2:
                        st.metric("Accélération", app_info['speedup'])
                        st.metric("Qubits Requis", app_info['qubits_required'])
                        
                        if n_qubits >= app_info['qubits_required']:
                            st.success("✅ Compatible")
                        else:
                            st.error(f"❌ Requiert {app_info['qubits_required']} qubits")
    
    with tab2:
        st.subheader("🔬 Circuits Quantiques")
        
        st.write("### 🎨 Créer un Circuit Quantique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            circuit_type = st.selectbox(
                "Type de Circuit",
                ["Superposition", "Entanglement", "Optimization", "Sensing", "Custom"]
            )
            
            n_qubits_circuit = st.slider("Nombre de Qubits", 2, 20, 5)
        
        with col2:
            depth = st.slider("Profondeur du Circuit", 1, 100, 10)
            shots = st.number_input("Nombre de Mesures", 100, 10000, 1000)
        
        if st.button("🔬 Générer Circuit"):
            st.success(f"✅ Circuit quantique généré: {n_qubits_circuit} qubits, profondeur {depth}")
            
            # Visualisation simplifiée du circuit
            st.write("**Structure du Circuit:**")
            
            circuit_gates = []
            for d in range(min(depth, 10)):
                gate_type = np.random.choice(['H', 'X', 'Y', 'Z', 'CNOT', 'RX', 'RY'])
                qubit = np.random.randint(0, n_qubits_circuit)
                circuit_gates.append(f"Layer {d}: {gate_type} on qubit {qubit}")
            
            st.code("\n".join(circuit_gates))
            
            st.markdown("---")
            
            # Simulation d'exécution
            if st.button("▶️ Exécuter Circuit"):
                with st.spinner("Exécution sur QPU..."):
                    # Simulation de résultats
                    states = [f"|{i:0{n_qubits_circuit}b}⟩" for i in range(2**min(n_qubits_circuit, 4))]
                    probabilities = np.random.dirichlet(np.ones(len(states))) * 100
                    
                    st.success(f"✅ Circuit exécuté: {shots} mesures")
                    
                    # Résultats
                    fig = go.Figure(data=[
                        go.Bar(x=states, y=probabilities, marker_color='rgba(102, 126, 234, 0.7)')
                    ])
                    
                    fig.update_layout(
                        title="Distribution des États Quantiques",
                        xaxis_title="État",
                        yaxis_title="Probabilité (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{d}")
    
    with tab3:
        st.subheader("📊 Résultats Quantiques")
        
        st.write("### 📈 Comparaison Classique vs Quantique")
        
        # Simulation de benchmark
        problems = ["Optimisation Route", "Classification Image", "Recherche Base", "Cryptographie"]
        classical_time = [100, 50, 200, 300]  # ms
        quantum_time = [10, 45, 20, 1]  # ms
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=problems,
            y=classical_time,
            name='Classique',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=problems,
            y=quantum_time,
            name='Quantique',
            marker_color='purple'
        ))
        
        fig.update_layout(
            title="Temps d'Exécution: Classique vs Quantique",
            xaxis_title="Problème",
            yaxis_title="Temps (ms)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Métriques quantiques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accélération Moyenne", "8.5x")
        with col2:
            st.metric("Circuits Exécutés", "1,247")
        with col3:
            st.metric("Fidélité Moyenne", "98.7%")
        with col4:
            st.metric("Efficacité Énergétique", "+45%")
    
    with tab4:
        st.subheader("🚀 Applications Robotiques")
        
        st.write("### 🎯 Cas d'Usage Quantiques en Robotique")
        
        use_cases = [
            {
                "name": "Navigation Quantique Optimisée",
                "description": "Planification de trajectoire optimale en temps réel",
                "benefit": "Réduction 70% du temps de calcul",
                "implementation": "QAOA + A* hybride",
                "status": "Production"
            },
            {
                "name": "Vision Quantique",
                "description": "Reconnaissance d'objets avec réseaux quantiques",
                "benefit": "Amélioration 15% de l'accuracy",
                "implementation": "Quantum Convolutional NN",
                "status": "Beta"
            },
            {
                "name": "Contrôle Quantique Prédictif",
                "description": "Contrôle adaptatif utilisant prédiction quantique",
                "benefit": "Stabilité +30%",
                "implementation": "Quantum MPC",
                "status": "Recherche"
            },
            {
                "name": "Swarm Intelligence Quantique",
                "description": "Coordination d'essaims via enchevêtrement",
                "benefit": "Synchronisation parfaite",
                "implementation": "Quantum Entanglement Protocol",
                "status": "Concept"
            }
        ]
        
        for uc in use_cases:
            status_colors = {
                "Production": "🟢",
                "Beta": "🟡",
                "Recherche": "🔵",
                "Concept": "⚪"
            }
            
            with st.expander(f"{status_colors[uc['status']]} {uc['name']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {uc['description']}")
                    st.write(f"**Bénéfice:** {uc['benefit']}")
                    st.write(f"**Implémentation:** {uc['implementation']}")
                
                with col2:
                    st.metric("Statut", uc['status'])
                    
                    if uc['status'] in ["Production", "Beta"]:
                        if st.button("🚀 Déployer", key=f"deploy_{uc['name']}"):
                            st.success("Application déployée!")

# ==================== PAGE: SYSTÈMES BIOLOGIQUES ====================

elif page == "🧬 Systèmes Biologiques":
    st.header("🧬 Systèmes Biologiques Intégrés")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Bio-Interfaces", "🧫 Tissus Biologiques", "📊 Monitoring", "⚕️ Applications"])
    
    with tab1:
        st.subheader("🔬 Interfaces Bio-Électroniques")
        
        bio_robots = {k: v for k, v in st.session_state.robotics_system['robots'].items() 
                     if v['advanced_systems']['biological']}
        
        if not bio_robots:
            st.info("Aucun robot avec système biologique. Créez un robot bio-hybride.")
        else:
            robot_options = {r['id']: r['name'] for r in bio_robots.values()}
            selected_robot = st.selectbox(
                "Sélectionner un Robot Bio-Hybride",
                options=list(robot_options.keys()),
                format_func=lambda x: robot_options[x]
            )
            
            robot = st.session_state.robotics_system['robots'][selected_robot]
            
            st.write(f"### 🧬 {robot['name']}")
            st.write(f"**Type Bio:** {robot['advanced_systems']['bio_type'].replace('_', ' ').title()}")
            
            st.markdown("---")
            
            # Propriétés biologiques
            st.write("### 🧫 Propriétés Biologiques")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                viability = 95 + np.random.random() * 5
                st.metric("Viabilité Cellulaire", f"{viability:.1f}%")
            
            with col2:
                biocompatibility = 0.95
                st.metric("Biocompatibilité", f"{biocompatibility:.0%}")
            
            with col3:
                self_healing = 0.8
                st.metric("Auto-Réparation", f"{self_healing:.0%}")
            
            with col4:
                adaptation = 0.9
                st.metric("Adaptation", f"{adaptation:.0%}")
            
            st.markdown("---")
            
            # Types de bio-interfaces
            st.write("### 🔌 Types de Bio-Interfaces")
            
            interfaces = {
                "Interface Neuronale": {
                    "description": "Connexion directe aux systèmes nerveux",
                    "channels": 1024,
                    "resolution": "< 1 µV",
                    "bandwidth": "10 kHz",
                    "applications": ["Contrôle neuronal", "Feedback sensoriel", "BCI"]
                },
                "Muscles Biologiques": {
                    "description": "Actionneurs musculaires vivants",
                    "force": "20 N/cm²",
                    "efficiency": "40%",
                    "response_time": "100 ms",
                    "applications": ["Manipulation douce", "Mouvement naturel"]
                },
                "Capteurs Biologiques": {
                    "description": "Cellules sensorielles pour détection",
                    "sensitivity": "Moléculaire",
                    "selectivity": "Haute",
                    "dynamic_range": "10⁶",
                    "applications": ["Détection chimique", "Olfaction", "Goût"]
                },
                "Peau Artificielle": {
                    "description": "Tissu cutané avec sensation tactile",
                    "tactile_points": 10000,
                    "pressure_range": "1-1000 kPa",
                    "temperature_range": "0-50°C",
                    "applications": ["Manipulation précise", "Interaction humaine"]
                }
            }
            
            for interface_name, interface_info in interfaces.items():
                with st.expander(f"🔌 {interface_name}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {interface_info['description']}")
                        st.write("**Spécifications:**")
                        for key, value in interface_info.items():
                            if key not in ['description', 'applications']:
                                st.write(f"  • {key.replace('_', ' ').title()}: {value}")
                    
                    with col2:
                        st.write("**Applications:**")
                        for app in interface_info['applications']:
                            st.write(f"• {app}")
    
    with tab2:
        st.subheader("🧫 Culture et Maintenance des Tissus")
        
        st.write("### 🌡️ Paramètres Environnementaux")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temperature = st.slider("Température (°C)", 20.0, 40.0, 37.0, 0.1)
            ph_level = st.slider("pH", 6.0, 8.0, 7.4, 0.1)
        
        with col2:
            nutrient_level = st.slider("Niveau Nutriments", 0, 100, 80)
            oxygen_level = st.slider("Oxygénation (%)", 0, 100, 95)
        
        with col3:
            co2_level = st.slider("CO₂ (%)", 0, 10, 5)
            humidity = st.slider("Humidité (%)", 0, 100, 95)
        
        # Vérification des paramètres
        st.markdown("---")
        
        status_ok = (36.5 <= temperature <= 37.5 and 
                    7.2 <= ph_level <= 7.6 and 
                    nutrient_level > 60 and 
                    oxygen_level > 90)
        
        if status_ok:
            st.success("✅ Tous les paramètres sont dans la plage optimale")
        else:
            st.warning("⚠️ Certains paramètres nécessitent un ajustement")
        
        # Graphique évolution
        st.markdown("---")
        st.write("### 📈 Évolution de la Culture Cellulaire")
        
        days = np.arange(0, 30)
        cell_density = 1000 * np.exp(days * 0.1) + np.random.random(30) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days,
            y=cell_density,
            mode='lines+markers',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)'
        ))
        
        fig.update_layout(
            title="Densité Cellulaire au Fil du Temps",
            xaxis_title="Jours",
            yaxis_title="Cellules/mL",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Monitoring en Temps Réel")
        
        st.write("### 🔬 Indicateurs Biologiques")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            viability_rt = 95 + np.random.random() * 5
            st.metric("Viabilité", f"{viability_rt:.1f}%", delta=f"{np.random.randn():.1f}%")
        
        with col2:
            metabolic_rate = 80 + np.random.random() * 20
            st.metric("Taux Métabolique", f"{metabolic_rate:.0f}%", delta=f"{np.random.randn()*2:.1f}%")
        
        with col3:
            growth_rate = 0.05 + np.random.random() * 0.05
            st.metric("Taux Croissance", f"{growth_rate:.3f} /h")
        
        with col4:
            waste_level = np.random.random() * 20
            st.metric("Niveau Déchets", f"{waste_level:.1f}%")
        
        st.markdown("---")
        
        # Signaux bio-électriques
        st.write("### 📡 Signaux Bio-Électriques")
        
        time = np.linspace(0, 10, 1000)
        
        # ECG simulé
        ecg_signal = np.sin(2 * np.pi * 1.2 * time) + 0.3 * np.sin(2 * np.pi * 8 * time)
        
        # EMG simulé
        emg_signal = np.random.randn(1000) * 0.5 + np.sin(2 * np.pi * 2 * time)
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Signal ECG", "Signal EMG"))
        
        fig.add_trace(
            go.Scatter(x=time, y=ecg_signal, mode='lines', line=dict(color='red', width=1)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=time, y=emg_signal, mode='lines', line=dict(color='blue', width=1)),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Temps (s)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (mV)", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚕️ Applications Bio-Robotiques")
        
        st.write("### 🎯 Domaines d'Application")
        
        applications = [
            {
                "domain": "Prothèses Bioniques",
                "description": "Membres artificiels avec sensation tactile",
                "benefits": ["Feedback sensoriel naturel", "Contrôle intuitif", "Auto-réparation"],
                "maturity": "Clinique",
                "icon": "🦾"
            },
            {
                "domain": "Organes Artificiels",
                "description": "Organes bio-hybrides fonctionnels",
                "benefits": ["Biocompatibilité élevée", "Fonctions biologiques", "Longévité"],
                "maturity": "Recherche",
                "icon": "❤️"
            },
            {
                "domain": "Robots Médicaux",
                "description": "Robots chirurgicaux avec tissus vivants",
                "benefits": ["Manipulation délicate", "Cicatrisation rapide", "Pas de rejet"],
                "maturity": "Prototype",
                "icon": "⚕️"
            },
            {
                "domain": "Bio-Capteurs Implantables",
                "description": "Capteurs biologiques pour monitoring continu",
                "benefits": ["Détection moléculaire", "Intégration corporelle", "Longue durée"],
                "maturity": "Production",
                "icon": "🔬"
            },
            {
                "domain": "Exosquelettes Biologiques",
                "description": "Augmentation physique via muscles artificiels",
                "benefits": ["Mouvement naturel", "Endurance accrue", "Léger"],
                "maturity": "Beta",
                "icon": "🦿"
            }
        ]
        
        for app in applications:
            with st.expander(f"{app['icon']} {app['domain']} ({app['maturity']})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {app['description']}")
                    st.write("**Bénéfices:**")
                    for benefit in app['benefits']:
                        st.write(f"  • {benefit}")
                
                with col2:
                    st.metric("Maturité", app['maturity'])
                    
                    if st.button("📚 En Savoir Plus", key=f"learn_{app['domain']}"):
                        st.info("Documentation technique disponible")

# ==================== PAGE: COMPOSANTS & ACTIONNEURS ====================

elif page == "🔧 Composants & Actionneurs":
    st.header("🔧 Composants et Actionneurs Robotiques")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔩 Catalogue", "⚙️ Spécifications", "📊 Performance", "🛒 Sélection"])
    
    with tab1:
        st.subheader("🔩 Catalogue de Composants")
        
        # Catégories d'actionneurs
        actuator_categories = {
            "Moteurs Électriques": {
                "DC Brushless": {
                    "couple": "0.1-100 Nm",
                    "vitesse": "1000-10000 rpm",
                    "efficacite": "85-95%",
                    "cout": "$50-500",
                    "applications": ["Roues", "Propulsion", "Manipulation"]
                },
                "Servo-Moteur": {
                    "couple": "0.5-50 Nm",
                    "vitesse": "60-300 rpm",
                    "efficacite": "75-85%",
                    "cout": "$20-300",
                    "applications": ["Articulations", "Positionnement précis"]
                },
                "Moteur Pas-à-Pas": {
                    "couple": "0.2-20 Nm",
                    "vitesse": "100-1000 rpm",
                    "efficacite": "70-80%",
                    "cout": "$15-200",
                    "applications": ["Positionnement", "Imprimantes 3D"]
                }
            },
            "Actionneurs Hydrauliques": {
                "Vérin Simple": {
                    "force": "100-50000 N",
                    "vitesse": "10-500 mm/s",
                    "efficacite": "80-90%",
                    "cout": "$100-2000",
                    "applications": ["Levage lourd", "Construction"]
                },
                "Vérin Rotatif": {
                    "couple": "100-10000 Nm",
                    "vitesse": "10-180 rpm",
                    "efficacite": "85-92%",
                    "cout": "$200-5000",
                    "applications": ["Rotation puissante", "Excavation"]
                }
            },
            "Actionneurs Pneumatiques": {
                "Vérin Pneumatique": {
                    "force": "10-5000 N",
                    "vitesse": "100-2000 mm/s",
                    "efficacite": "20-40%",
                    "cout": "$30-500",
                    "applications": ["Pick & place", "Assemblage"]
                },
                "Muscles Pneumatiques": {
                    "force": "100-3000 N",
                    "vitesse": "Variable",
                    "efficacite": "25-45%",
                    "cout": "$50-800",
                    "applications": ["Soft robotics", "Réhabilitation"]
                }
            },
            "Actionneurs Avancés": {
                "Alliage à Mémoire Forme": {
                    "force": "10-500 N",
                    "vitesse": "1-50 mm/s",
                    "efficacite": "5-10%",
                    "cout": "$100-1500",
                    "applications": ["Micro-robots", "Bio-médical"]
                },
                "Muscle Artificiel": {
                    "force": "50-2000 N",
                    "vitesse": "10-200 mm/s",
                    "efficacite": "30-50%",
                    "cout": "$200-3000",
                    "applications": ["Humanoïdes", "Prothèses"]
                },
                "Piézoélectrique": {
                    "force": "0.1-100 N",
                    "vitesse": "0.001-10 mm/s",
                    "efficacite": "60-80%",
                    "cout": "$50-1000",
                    "applications": ["Nano-positionnement", "Précision"]
                }
            }
        }
        
        for category, actuators in actuator_categories.items():
            st.write(f"### {category}")
            
            for act_name, specs in actuators.items():
                with st.expander(f"⚙️ {act_name}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        for key, value in specs.items():
                            if key != 'applications':
                                st.write(f"**{key.title()}:** {value}")
                    
                    with col2:
                        st.write("**Applications:**")
                        for app in specs['applications']:
                            st.write(f"• {app}")
                        
                        if st.button("🛒 Ajouter au Panier", key=f"cart_{act_name}"):
                            st.success(f"{act_name} ajouté!")
    
    with tab2:
        st.subheader("⚙️ Spécifications Détaillées")
        
        # Comparateur d'actionneurs
        st.write("### 🔍 Comparateur d'Actionneurs")
        
        actuator_type = st.selectbox(
            "Type d'Actionneur",
            ["Moteur DC Brushless", "Servo-Moteur", "Vérin Hydraulique", 
             "Muscle Artificiel", "Moteur Pas-à-Pas"]
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            power_req = st.slider("Puissance Requise (W)", 1, 5000, 100)
            torque_req = st.slider("Couple Requis (Nm)", 0.1, 1000.0, 10.0)
        
        with col2:
            speed_req = st.slider("Vitesse Requise (rpm)", 10, 10000, 1000)
            efficiency_min = st.slider("Efficacité Min (%)", 50, 95, 80)
        
        with col3:
            budget_max = st.number_input("Budget Max ($)", 10, 10000, 500)
            weight_max = st.number_input("Poids Max (kg)", 0.1, 100.0, 5.0)
        
        if st.button("🔍 Rechercher Actionneurs Compatibles"):
            st.success("✅ 12 actionneurs trouvés correspondant aux critères")
            
            # Résultats simulés
            results = [
                {"model": "BLM-3000", "torque": 15.0, "speed": 3000, "eff": 92, "price": 450, "weight": 3.2},
                {"model": "SM-500", "torque": 8.5, "speed": 5000, "eff": 88, "price": 280, "weight": 2.1},
                {"model": "BLDC-H", "torque": 20.0, "speed": 2500, "eff": 94, "price": 490, "weight": 4.5},
            ]
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Tests de Performance")
        
        st.write("### ⚡ Courbes Caractéristiques")
        
        # Courbe couple-vitesse
        speed = np.linspace(0, 5000, 100)
        torque = 20 * (1 - speed / 5000) + np.random.random(100) * 0.5
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=speed,
            y=torque,
            mode='lines',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'
        ))
        
        fig.update_layout(
            title="Courbe Couple-Vitesse",
            xaxis_title="Vitesse (rpm)",
            yaxis_title="Couple (Nm)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Efficacité vs charge
        load = np.linspace(0, 100, 50)
        efficiency = 50 + 40 * np.exp(-((load - 60)**2) / 500)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=load,
            y=efficiency,
            mode='lines',
            line=dict(color='green', width=3)
        ))
        
        fig.update_layout(
            title="Efficacité vs Charge",
            xaxis_title="Charge (%)",
            yaxis_title="Efficacité (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🛒 Assistant de Sélection")
        
        st.write("### 🎯 Configuration Personnalisée")
        
        with st.form("actuator_selection"):
            st.write("**Répondez aux questions pour obtenir une recommandation:**")
            
            q1 = st.radio("Type de robot?", 
                         ["Humanoïde", "Bras manipulateur", "Mobile", "Drone"])
            
            q2 = st.radio("Charge utile?", 
                         ["< 1 kg", "1-5 kg", "5-20 kg", "> 20 kg"])
            
            q3 = st.radio("Vitesse requise?", 
                         ["Lente (précision)", "Moyenne", "Rapide"])
            
            q4 = st.radio("Environnement?", 
                         ["Intérieur propre", "Extérieur", "Industriel", "Extrême"])
            
            q5 = st.radio("Budget?", 
                         ["< $100/actuator", "$100-500", "$500-2000", "> $2000"])
            
            submitted = st.form_submit_button("💡 Obtenir Recommandation")
            
            if submitted:
                st.success("✅ Analyse terminée!")
                
                st.write("### 🎯 Recommandations:")
                
                recommendations = [
                    {
                        "rank": 1,
                        "type": "Servo-Moteur Haute Performance",
                        "model": "SM-HD-500",
                        "score": 95,
                        "pros": ["Précision excellente", "Bon rapport qualité/prix", "Compact"],
                        "cons": ["Puissance limitée"]
                    },
                    {
                        "rank": 2,
                        "type": "Moteur DC Brushless",
                        "model": "BLDC-3K",
                        "score": 88,
                        "pros": ["Puissant", "Efficace", "Fiable"],
                        "cons": ["Plus cher", "Nécessite contrôleur"]
                    }
                ]
                
                for rec in recommendations:
                    with st.expander(f"#{rec['rank']} - {rec['type']} (Score: {rec['score']}/100)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Modèle:** {rec['model']}")
                            st.write("**Avantages:**")
                            for pro in rec['pros']:
                                st.write(f"✅ {pro}")
                        
                        with col2:
                            st.write("**Inconvénients:**")
                            for con in rec['cons']:
                                st.write(f"⚠️ {con}")
                            
                            st.form_submit_button(f"🛒 Commander", key=f"order_{rec['model']}")

# ==================== PAGE: CAPTEURS & PERCEPTION ====================

elif page == "📡 Capteurs & Perception":
    st.header("📡 Systèmes de Capteurs et Perception")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Vision", "🔊 Audio", "🤚 Tactile", "🧭 Navigation"])
    
    with tab1:
        st.subheader("📷 Systèmes de Vision")
        
        st.write("### 🎥 Types de Caméras")
        
        camera_types = {
            "Caméra RGB": {
                "resolution": ["640x480", "1920x1080", "3840x2160"],
                "fps": [30, 60, 120, 240],
                "field_of_view": "60-180°",
                "applications": ["Navigation", "Reconnaissance objets", "Interaction"],
                "cost": "$20-500"
            },
            "Caméra Profondeur": {
                "resolution": ["320x240", "640x480", "1280x720"],
                "range": "0.5-10m",
                "technology": ["Stereo", "ToF", "Structured Light"],
                "applications": ["Cartographie 3D", "Évitement obstacles", "Manipulation"],
                "cost": "$100-1500"
            },
            "Caméra Thermique": {
                "resolution": ["80x60", "160x120", "640x480"],
                "temperature_range": "-20 à 500°C",
                "accuracy": "±2°C",
                "applications": ["Vision nocturne", "Détection personnes", "Inspection"],
                "cost": "$200-5000"
            },
            "Caméra Hyperspectrale": {
                "bands": "100-300 bandes",
                "wavelength": "400-2500 nm",
                "resolution": "Variable",
                "applications": ["Agriculture", "Inspection qualité", "Recherche"],
                "cost": "$5000-50000"
            }
        }
        
        for cam_name, specs in camera_types.items():
            with st.expander(f"📷 {cam_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    for key, value in specs.items():
                        if key != 'applications':
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                with col2:
                    st.write("**Applications:**")
                    for app in specs['applications']:
                        st.write(f"• {app}")
        
        st.markdown("---")
        
        # Test de vision
        st.write("### 🧪 Test de Vision")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader("Charger une image test", type=['jpg', 'png'])
            
            if uploaded_image:
                st.image(uploaded_image, width=400)
        
        with col2:
            if uploaded_image:
                vision_task = st.selectbox(
                    "Tâche de Vision",
                    ["Détection d'Objets", "Segmentation", "Classification", 
                     "Estimation de Profondeur", "Tracking"]
                )
                
                if st.button("🔍 Analyser Image"):
                    with st.spinner("Analyse en cours..."):
                        st.success("✅ Analyse terminée!")
                        
                        st.write("**Résultats:**")
                        if vision_task == "Détection d'Objets":
                            st.write("• Personne (conf: 98%)")
                            st.write("• Chaise (conf: 95%)")
                            st.write("• Table (conf: 92%)")
                        elif vision_task == "Segmentation":
                            st.write("• 3 objets segmentés")
                            st.write("• Précision: 94%")
    
    with tab2:
        st.subheader("🔊 Perception Audio")
        
        st.write("### 🎤 Microphones et Traitement Audio")
        
        audio_systems = {
            "Microphone Directionnel": {
                "pattern": "Cardioïde, Supercardioïde",
                "frequency": "20 Hz - 20 kHz",
                "snr": "> 70 dB",
                "applications": ["Commande vocale", "Source localization"]
            },
            "Array de Microphones": {
                "channels": "4-64",
                "beamforming": "Adaptatif",
                "range": "1-10m",
                "applications": ["Réduction bruit", "Localisation 3D"]
            },
            "Microphone Ultrason": {
                "frequency": "20-200 kHz",
                "range": "0.1-10m",
                "resolution": "1 mm",
                "applications": ["Détection obstacles", "Communication"]
            }
        }
        
        for audio_name, specs in audio_systems.items():
            with st.expander(f"🎤 {audio_name}"):
                for key, value in specs.items():
                    if key != 'applications':
                        st.write(f"**{key.title()}:** {value}")
                
                st.write("**Applications:**")
                for app in specs['applications']:
                    st.write(f"• {app}")
        
        st.markdown("---")
        
        # Visualisation audio
        st.write("### 📊 Signal Audio")
        
        t = np.linspace(0, 1, 1000)
        audio_signal = np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=audio_signal, mode='lines',
                                line=dict(color='blue', width=1)))
        
        fig.update_layout(
            title="Forme d'Onde Audio",
            xaxis_title="Temps (s)",
            yaxis_title="Amplitude",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{audio_name}")
    
    with tab3:
        st.subheader("🤚 Capteurs Tactiles")
        
        st.write("### ✋ Technologies Tactiles")
        
        tactile_sensors = {
            "Résistif": {
                "resolution": "1-5 mm",
                "pressure_range": "0.1-1000 kPa",
                "response_time": "< 10 ms",
                "durability": "1M cycles",
                "cost": "$10-100"
            },
            "Capacitif": {
                "resolution": "0.5-3 mm",
                "pressure_range": "0.01-100 kPa",
                "response_time": "< 5 ms",
                "durability": "10M cycles",
                "cost": "$20-200"
            },
            "Piézoélectrique": {
                "resolution": "0.1-1 mm",
                "pressure_range": "0.001-10 kPa",
                "response_time": "< 1 ms",
                "durability": "100M cycles",
                "cost": "$50-500"
            },
            "Optique": {
                "resolution": "0.01-0.5 mm",
                "pressure_range": "0.001-100 kPa",
                "response_time": "< 1 ms",
                "durability": "Illimité",
                "cost": "$100-1000"
            }
        }
        
        for sensor_name, specs in tactile_sensors.items():
            with st.expander(f"✋ Capteur {sensor_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    for key, value in list(specs.items())[:3]:
                        st.metric(key.replace('_', ' ').title(), value)
                
                with col2:
                    for key, value in list(specs.items())[3:]:
                        st.metric(key.replace('_', ' ').title(), value)
        
        st.markdown("---")
        
        # Carte de pression
        st.write("### 🗺️ Carte de Pression Tactile")
        
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        
        # Simulation de pression tactile
        Z = np.exp(-((X-5)**2 + (Y-5)**2) / 5) * 100
        
        fig = go.Figure(data=go.Contour(
            z=Z,
            x=x,
            y=y,
            colorscale='Hot',
            colorbar=dict(title="Pression (kPa)")
        ))
        
        fig.update_layout(
            title="Distribution de Pression",
            xaxis_title="X (cm)",
            yaxis_title="Y (cm)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{sensor_name}")
    
    with tab4:
        st.subheader("🧭 Systèmes de Navigation")
        
        st.write("### 🛰️ Capteurs de Localisation")
        
        nav_systems = {
            "GPS/GNSS": {
                "precision": "1-5m (standard), 1-10cm (RTK)",
                "update_rate": "1-10 Hz",
                "environment": "Extérieur uniquement",
                "cost": "$50-2000"
            },
            "IMU": {
                "precision": "Drift: 1-10°/h",
                "update_rate": "100-1000 Hz",
                "sensors": "Accéléromètre, Gyroscope, Magnétomètre",
                "cost": "$20-500"
            },
            "LiDAR": {
                "precision": "±2cm",
                "range": "0.1-200m",
                "scan_rate": "5-20 Hz",
                "cost": "$1000-75000"
            },
            "Odométrie Visuelle": {
                "precision": "< 1% drift",
                "update_rate": "10-30 Hz",
                "environment": "Texturé",
                "cost": "$100-1000"
            },
            "UWB": {
                "precision": "10-30 cm",
                "range": "10-100m",
                "update_rate": "10-100 Hz",
                "cost": "$50-300"
            }
        }
        
        for nav_name, specs in nav_systems.items():
            with st.expander(f"🧭 {nav_name}"):
                for key, value in specs.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        st.markdown("---")
        
        # Fusion de capteurs
        st.write("### 🔀 Fusion Multi-Capteurs (Kalman Filter)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Capteurs Actifs:**")
            sensors_active = st.multiselect(
                "Sélectionner capteurs",
                ["GPS", "IMU", "Odométrie", "LiDAR"],
                default=["GPS", "IMU"]
            )
            
            fusion_rate = st.slider("Taux de Fusion (Hz)", 1, 100, 20)
        
        with col2:
            if len(sensors_active) > 1:
                st.success(f"✅ {len(sensors_active)} capteurs fusionnés")
                
                # Amélioration précision
                base_precision = 5.0  # mètres
                improvement = 1 / np.sqrt(len(sensors_active))
                final_precision = base_precision * improvement
                
                st.metric("Précision Fusionnée", f"{final_precision:.2f} m")
                st.metric("Amélioration", f"{(1-improvement)*100:.0f}%")
            else:
                st.warning("Sélectionnez au moins 2 capteurs")

# ==================== PAGE: COÛTS & ROI ====================

elif page == "💰 Coûts & ROI":
    st.header("💰 Analyse des Coûts et Retour sur Investissement")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💵 Coûts", "📊 ROI", "📈 Prévisions", "💡 Optimisation"])
    
    with tab1:
        st.subheader("💵 Structure des Coûts")
        
        if st.session_state.robotics_system['robots']:
            st.write("### 💰 Coûts par Robot")
            
            # Tableau des coûts
            cost_data = []
            total_dev = 0
            total_mfg = 0
            total_op = 0
            
            for robot_id, robot in st.session_state.robotics_system['robots'].items():
                dev_cost = robot['costs']['development']
                mfg_cost = robot['costs']['manufacturing']
                op_cost = robot['costs']['operational_per_hour'] * robot['operations']['hours']
                
                total_dev += dev_cost
                total_mfg += mfg_cost
                total_op += op_cost
                
                cost_data.append({
                    'Robot': robot['name'][:20],
                    'Développement': f"${dev_cost:,.0f}",
                    'Fabrication': f"${mfg_cost:,.0f}",
                    'Opérationnel': f"${op_cost:,.0f}",
                    'Total': f"${dev_cost + mfg_cost + op_cost:,.0f}"
                })
            
            df = pd.DataFrame(cost_data)
            st.dataframe(df, use_container_width=True)
            
            # Totaux
            st.markdown("---")
            st.write("### 📊 Coûts Totaux")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Développement", f"${total_dev:,.0f}")
            with col2:
                st.metric("Fabrication", f"${total_mfg:,.0f}")
            with col3:
                st.metric("Opérationnel", f"${total_op:,.0f}")
            with col4:
                st.metric("TOTAL", f"${total_dev + total_mfg + total_op:,.0f}")
            
            # Graphique répartition
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Répartition des Coûts")
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Développement', 'Fabrication', 'Opérationnel'],
                    values=[total_dev, total_mfg, total_op],
                    hole=0.4
                )])
                
                fig.update_layout(title="Répartition Globale", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### 📈 Coûts par Robot")
                
                robot_names = [r['name'][:15] for r in st.session_state.robotics_system['robots'].values()]
                total_costs = [r['costs']['development'] + r['costs']['manufacturing'] + 
                             r['costs']['operational_per_hour'] * r['operations']['hours']
                             for r in st.session_state.robotics_system['robots'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=robot_names, y=total_costs, marker_color='lightcoral')
                ])
                
                fig.update_layout(
                    title="Coût Total par Robot",
                    xaxis_tickangle=-45,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun robot créé")
    
    with tab2:
        st.subheader("📊 Retour sur Investissement (ROI)")
        
        st.write("### 💡 Calculateur de ROI")
        
        with st.form("roi_calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Coûts:**")
                investment = st.number_input("Investissement Initial ($)", 0, 10000000, 500000, 10000)
                operational_cost = st.number_input("Coût Opérationnel Annuel ($)", 0, 1000000, 50000, 1000)
                maintenance_cost = st.number_input("Coût Maintenance Annuel ($)", 0, 500000, 20000, 1000)
            
            with col2:
                st.write("**Bénéfices:**")
                revenue = st.number_input("Revenus Annuels ($)", 0, 10000000, 200000, 10000)
                cost_savings = st.number_input("Économies Annuelles ($)", 0, 5000000, 100000, 5000)
                years = st.slider("Période d'Analyse (années)", 1, 10, 5)
            
            submitted = st.form_submit_button("📊 Calculer ROI", type="primary")
            
            if submitted:
                # Calculs
                total_cost = investment + (operational_cost + maintenance_cost) * years
                total_benefit = (revenue + cost_savings) * years
                net_benefit = total_benefit - total_cost
                roi = (net_benefit / investment) * 100 if investment > 0 else 0
                payback_period = investment / (revenue + cost_savings - operational_cost - maintenance_cost) if (revenue + cost_savings - operational_cost - maintenance_cost) > 0 else float('inf')
                
                st.markdown("---")
                st.write("### 📈 Résultats")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ROI", f"{roi:.1f}%", delta="Positif" if roi > 0 else "Négatif")
                with col2:
                    st.metric("Bénéfice Net", f"${net_benefit:,.0f}")
                with col3:
                    st.metric("Période Retour", f"{payback_period:.1f} ans" if payback_period != float('inf') else "∞")
                with col4:
                    irr = ((total_benefit / total_cost) ** (1/years) - 1) * 100 if total_cost > 0 else 0
                    st.metric("TRI", f"{irr:.1f}%")
                
                # Graphique flux de trésorerie
                st.markdown("---")
                st.write("### 💰 Flux de Trésorerie Cumulé")
                
                cashflow = [-investment]
                for year in range(1, years + 1):
                    annual_cashflow = revenue + cost_savings - operational_cost - maintenance_cost
                    cashflow.append(cashflow[-1] + annual_cashflow)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(years + 1)),
                    y=cashflow,
                    mode='lines+markers',
                    line=dict(color='green', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.2)'
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title="Évolution du Flux de Trésorerie",
                    xaxis_title="Année",
                    yaxis_title="Flux Cumulé ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{year}")
                
                # Interprétation
                st.markdown("---")
                st.write("### 💡 Interprétation")
                
                if roi > 50:
                    st.success(f"✅ Excellent ROI ({roi:.1f}%). Investissement très rentable!")
                elif roi > 20:
                    st.success(f"✅ Bon ROI ({roi:.1f}%). Investissement rentable.")
                elif roi > 0:
                    st.info(f"ℹ️ ROI positif ({roi:.1f}%). Rentabilité modérée.")
                else:
                    st.warning(f"⚠️ ROI négatif ({roi:.1f}%). Revoir le modèle économique.")
    
    with tab3:
        st.subheader("📈 Prévisions Financières")
        
        st.write("### 🔮 Projection sur 10 ans")
        
        # Simulation de prévisions
        years = np.arange(1, 11)
        
        # Scénarios
        scenario = st.selectbox("Scénario", ["Conservateur", "Réaliste", "Optimiste"])
        
        growth_rates = {
            "Conservateur": 0.05,
            "Réaliste": 0.10,
            "Optimiste": 0.20
        }
        
        growth_rate = growth_rates[scenario]
        
        initial_revenue = 200000
        revenues = [initial_revenue * (1 + growth_rate) ** year for year in years]
        costs = [100000 * (1 + 0.03) ** year for year in years]
        profits = [r - c for r, c in zip(revenues, costs)]
        
        # Graphique
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years, y=revenues,
            mode='lines+markers',
            name='Revenus',
            line=dict(color='green', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=costs,
            mode='lines+markers',
            name='Coûts',
            line=dict(color='red', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years, y=profits,
            mode='lines+markers',
            name='Profits',
            line=dict(color='blue', width=3),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=f"Prévisions Financières - Scénario {scenario}",
            xaxis_title="Année",
            yaxis_title="Montant ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métriques finales
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Revenus Année 10", f"${revenues[-1]:,.0f}")
        with col2:
            st.metric("Profit Année 10", f"${profits[-1]:,.0f}")
        with col3:
            total_profit = sum(profits)
            st.metric("Profit Total 10 ans", f"${total_profit:,.0f}")
        with col4:
            avg_growth = ((revenues[-1] / revenues[0]) ** (1/10) - 1) * 100
            st.metric("TCAM", f"{avg_growth:.1f}%")
    
    with tab4:
        st.subheader("💡 Optimisation des Coûts")
        
        st.write("### 🎯 Opportunités d'Optimisation")
        
        opportunities = [
            {
                "category": "Énergie",
                "opportunity": "Optimisation de la consommation énergétique",
                "current_cost": 50000,
                "potential_saving": 15000,
                "saving_pct": 30,
                "implementation": "Moyen",
                "timeframe": "3 mois"
            },
            {
                "category": "Maintenance",
                "opportunity": "Maintenance prédictive vs réactive",
                "current_cost": 80000,
                "potential_saving": 32000,
                "saving_pct": 40,
                "implementation": "Élevé",
                "timeframe": "6 mois"
            },
            {
                "category": "Production",
                "opportunity": "Automatisation de l'assemblage",
                "current_cost": 120000,
                "potential_saving": 36000,
                "saving_pct": 30,
                "implementation": "Très Élevé",
                "timeframe": "12 mois"
            },
            {
                "category": "Composants",
                "opportunity": "Achat en volume - Réduction coûts",
                "current_cost": 200000,
                "potential_saving": 30000,
                "saving_pct": 15,
                "implementation": "Faible",
                "timeframe": "1 mois"
            }
        ]
        
        total_potential = sum(o['potential_saving'] for o in opportunities)
        
        st.metric("💰 Économies Potentielles Totales", f"${total_potential:,.0f}")
        
        st.markdown("---")
        
        for opp in opportunities:
            with st.expander(f"💡 {opp['opportunity']} - Économie: ${opp['potential_saving']:,.0f}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Catégorie:** {opp['category']}")
                    st.write(f"**Coût Actuel:** ${opp['current_cost']:,.0f}/an")
                    st.write(f"**Économie Potentielle:** ${opp['potential_saving']:,.0f}/an ({opp['saving_pct']}%)")
                    
                    st.progress(opp['saving_pct'] / 100)
                
                with col2:
                    st.write(f"**Complexité:** {opp['implementation']}")
                    st.write(f"**Délai:** {opp['timeframe']}")
                    
                    if st.button("🚀 Lancer", key=f"launch_{opp['opportunity'][:10]}"):
                        st.success("Initiative lancée!")
        
        st.markdown("---")
        
        # Priorisation
        st.write("### 📊 Matrice de Priorisation")
        
        # Graphique impact vs effort
        impact_scores = [o['saving_pct'] for o in opportunities]
        effort_scores = [{'Faible': 1, 'Moyen': 2, 'Élevé': 3, 'Très Élevé': 4}[o['implementation']] for o in opportunities]
        labels = [o['opportunity'][:30] for o in opportunities]
        
        fig = go.Figure(data=go.Scatter(
            x=effort_scores,
            y=impact_scores,
            mode='markers+text',
            text=labels,
            textposition='top center',
            marker=dict(
                size=[o['potential_saving'] / 1000 for o in opportunities],
                color=impact_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Impact (%)")
            )
        ))
        
        fig.update_layout(
            title="Impact vs Effort (taille = économies)",
            xaxis_title="Effort d'Implémentation",
            yaxis_title="Impact (% d'économies)",
            height=500,
            xaxis=dict(tickvals=[1, 2, 3, 4], ticktext=['Faible', 'Moyen', 'Élevé', 'Très Élevé'])
        )
        
        fig.add_shape(type="line", x0=0, y0=25, x1=5, y1=25, line=dict(color="red", dash="dash"))
        fig.add_shape(type="line", x0=2.5, y0=0, x1=2.5, y1=50, line=dict(color="red", dash="dash"))
        
        st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{opp}")
        
        st.info("💡 Priorisez les initiatives dans le quadrant supérieur gauche (Impact élevé, Effort faible)")

# ==================== PAGE: BIBLIOTHÈQUE ====================

elif page == "📚 Bibliothèque":
    st.header("📚 Bibliothèque de Ressources")
    
    tab1, tab2, tab3 = st.tabs(["📖 Documentation", "🔧 Templates", "🌐 Communauté"])
    
    with tab1:
        st.subheader("📖 Documentation Technique")
        
        doc_categories = {
            "🤖 Robotique Générale": [
                "Guide de Conception de Robots",
                "Principes de Mécanique Robotique",
                "Cinématique et Dynamique",
                "Actionneurs et Capteurs"
            ],
            "💻 Programmation": [
                "Python pour la Robotique",
                "ROS (Robot Operating System)",
                "C++ Avancé pour Robots",
                "Frameworks et Librairies"
            ],
            "🧠 Intelligence Artificielle": [
                "Machine Learning pour Robots",
                "Deep Learning Appliqué",
                "Reinforcement Learning",
                "Computer Vision"
            ],
            "⚛️ Technologies Avancées": [
                "Robotique Quantique",
                "Systèmes Bio-Hybrides",
                "Capteurs Avancés",
                "Fusion Multi-Capteurs"
            ],
            "🔧 Maintenance": [
                "Guide de Maintenance Préventive",
                "Diagnostic et Réparation",
                "Calibration des Systèmes",
                "Sécurité et Normes"
            ]
        }
        
        for category, docs in doc_categories.items():
            with st.expander(category):
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"📄 {doc}")
                    
                    with col2:
                        if st.button("📥 Télécharger", key=f"download_{doc}"):
                            st.success("Document téléchargé!")
    
    with tab2:
        st.subheader("🔧 Templates et Modèles")
        
        templates = {
            "Configurations Robot": [
                {"name": "Robot Mobile Basique", "type": "YAML", "desc": "Configuration pour robot mobile à roues"},
                {"name": "Bras Manipulateur 6-DoF", "type": "YAML", "desc": "Configuration bras robotique 6 axes"},
                {"name": "Drone Quadricoptère", "type": "YAML", "desc": "Configuration drone avec caméra"},
            ],
            "Code Source": [
                {"name": "Navigation Autonome", "type": "Python", "desc": "Algorithme de navigation avec évitement"},
                {"name": "Contrôleur PID", "type": "Python", "desc": "Implémentation contrôleur PID générique"},
                {"name": "Vision Processing", "type": "Python", "desc": "Pipeline de traitement d'images"},
            ],
            "Rapports": [
                {"name": "Rapport de Test", "type": "Markdown", "desc": "Template rapport de tests"},
                {"name": "Documentation Projet", "type": "Markdown", "desc": "Structure documentation projet"},
                {"name": "Analyse Performance", "type": "Excel", "desc": "Tableau analyse performances"},
            ]
        }
        
        for cat, items in templates.items():
            st.write(f"### {cat}")
            
            for template in items:
                with st.expander(f"📋 {template['name']} ({template['type']})"):
                    st.write(f"**Description:** {template['desc']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📥 Télécharger", key=f"dl_temp_{template['name']}"):
                            st.success("Template téléchargé!")
                    
                    with col2:
                        if st.button("👁️ Aperçu", key=f"prev_temp_{template['name']}"):
                            st.code("# Template code preview\n# ...", language="python")
    
    with tab3:
        st.subheader("🌐 Communauté et Support")
        
        st.write("### 💬 Forums de Discussion")
        
        forums = [
            {"name": "Questions Générales", "posts": 1247, "members": 856},
            {"name": "Aide Technique", "posts": 892, "members": 654},
            {"name": "Projets Partagés", "posts": 445, "members": 423},
            {"name": "Annonces", "posts": 123, "members": 1200}
        ]
        
        for forum in forums:
            with st.expander(f"💬 {forum['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Messages", forum['posts'])
                with col2:
                    st.metric("Membres", forum['members'])
                
                if st.button("🔗 Accéder", key=f"forum_{forum['name']}"):
                    st.info("Ouverture du forum...")
        
        st.markdown("---")
        
        st.write("### 🎓 Experts et Mentors")
        
        experts = [
            {"name": "Dr. Sarah Chen", "specialty": "IA & ML", "rating": 4.9, "sessions": 234},
            {"name": "Prof. Marc Dubois", "specialty": "Robotique Mobile", "rating": 4.8, "sessions": 189},
            {"name": "Ing. Lisa Wang", "specialty": "Vision par Ordinateur", "rating": 4.9, "sessions": 156}
        ]
        
        for expert in experts:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"**{expert['name']}**")
                st.write(f"Spécialité: {expert['specialty']}")
            
            with col2:
                st.write(f"⭐ {expert['rating']}/5")
                st.write(f"{expert['sessions']} sessions")
            
            with col3:
                if st.button("📅 Réserver", key=f"book_{expert['name']}"):
                    st.success("Session réservée!")

# ==================== PAGE: APPLICATIONS ====================

elif page == "🌟 Applications":
    st.header("🌟 Applications et Cas d'Usage")
    
    tab1, tab2, tab3 = st.tabs(["🏭 Industrie", "⚕️ Santé", "🌾 Autres Secteurs"])
    
    with tab1:
        st.subheader("🏭 Applications Industrielles")
        
        industrial_apps = {
            "Automatisation d'Entrepôt": {
                "description": "Robots mobiles autonomes pour picking et transport",
                "benefits": ["Productivité +40%", "Coûts -30%", "Erreurs -95%"],
                "robots_required": 5,
                "roi": "12-18 mois",
                "complexity": "Moyenne"
            },
            "Assemblage Automatisé": {
                "description": "Bras robotiques collaboratifs pour assemblage",
                "benefits": ["Vitesse +60%", "Qualité +35%", "Flexibilité ++"],
                "robots_required": 3,
                "roi": "18-24 mois",
                "complexity": "Élevée"
            },
            "Inspection Qualité": {
                "description": "Robots avec vision pour contrôle qualité",
                "benefits": ["Détection défauts 99.9%", "24/7", "Traçabilité complète"],
                "robots_required": 2,
                "roi": "9-15 mois",
                "complexity": "Moyenne"
            },
            "Maintenance Prédictive": {
                "description": "Robots d'inspection autonomes",
                "benefits": ["Pannes -50%", "Downtime -40%", "Coûts maintenance -25%"],
                "robots_required": 4,
                "roi": "15-20 mois",
                "complexity": "Élevée"
            }
        }
        
        for app_name, app_info in industrial_apps.items():
            with st.expander(f"🏭 {app_name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {app_info['description']}")
                    
                    st.write("\n**Bénéfices:**")
                    for benefit in app_info['benefits']:
                        st.write(f"✅ {benefit}")
                
                with col2:
                    st.metric("Robots Requis", app_info['robots_required'])
                    st.metric("ROI", app_info['roi'])
                    st.write(f"**Complexité:** {app_info['complexity']}")
                    
                    if st.button("🚀 Démarrer Projet", key=f"start_{app_name}"):
                        st.success("Projet initialisé!")
    
    with tab2:
        st.subheader("⚕️ Applications Médicales")
        
        medical_apps = {
            "Chirurgie Assistée": {
                "description": "Robots chirurgicaux de précision",
                "benefits": ["Précision submillimétrique", "Récupération rapide", "Cicatrices minimales"],
                "robots_required": 1,
                "roi": "24-36 mois",
                "complexity": "Très Élevée"
            },
            "Livraison Hospitalière": {
                "description": "Robots autonomes pour transport interne",
                "benefits": ["Efficacité +50%", "Personnel libéré", "Traçabilité"],
                "robots_required": 8,
                "roi": "18-24 mois",
                "complexity": "Moyenne"
            },
            "Réhabilitation": {
                "description": "Exosquelettes et robots de rééducation",
                "benefits": ["Récupération +40%", "Motivation patient", "Suivi précis"],
                "robots_required": 3,
                "roi": "20-30 mois",
                "complexity": "Élevée"
            },
            "Désinfection Autonome": {
                "description": "Robots UV-C pour désinfection",
                "benefits": ["Élimination 99.99% pathogènes", "24/7", "Sécurité"],
                "robots_required": 5,
                "roi": "12-18 mois",
                "complexity": "Faible"
            }
        }
        
        for app_name, app_info in medical_apps.items():
            with st.expander(f"⚕️ {app_name}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {app_info['description']}")
                    
                    st.write("\n**Bénéfices:**")
                    for benefit in app_info['benefits']:
                        st.write(f"✅ {benefit}")
                
                with col2:
                    st.metric("Robots Requis", app_info['robots_required'])
                    st.metric("ROI", app_info['roi'])
                    st.write(f"**Complexité:** {app_info['complexity']}")
                    
                    if st.button("🚀 Démarrer Projet", key=f"start_med_{app_name}"):
                        st.success("Projet initialisé!")
    
    with tab3:
        st.subheader("🌾 Autres Secteurs")
        
        other_sectors = {
            "Agriculture": {
                "applications": ["Récolte automatisée", "Pulvérisation précise", "Surveillance cultures"],
                "icon": "🌾"
            },
            "Logistique": {
                "applications": ["Tri automatique", "Chargement/Déchargement", "Suivi inventaire"],
                "icon": "📦"
            },
            "Construction": {
                "applications": ["Impression 3D bâtiments", "Inspection sites", "Assemblage modulaire"],
                "icon": "🏗️"
            },
            "Exploration": {
                "applications": ["Exploration sous-marine", "Missions spatiales", "Zones dangereuses"],
                "icon": "🚀"
            },
            "Service": {
                "applications": ["Hôtellerie", "Nettoyage", "Livraison dernier kilomètre"],
                "icon": "🤝"
            },
            "Éducation": {
                "applications": ["Enseignement STEM", "Assistants pédagogiques", "Recherche"],
                "icon": "🎓"
            }
        }
        
        for sector, info in other_sectors.items():
            with st.expander(f"{info['icon']} {sector}"):
                st.write("**Applications Clés:**")
                for app in info['applications']:
                    st.write(f"• {app}")
                if st.button(f"💡 Explorer {sector}", key=f"explore_{sector}"):
                    st.info(f"Documentation {sector} disponible")
        
        st.markdown("---")
        
        # Galerie de projets
        st.write("### 🎨 Galerie de Projets")
        
        col1, col2, col3 = st.columns(3)
        
        
        projects_gallery = [
            {"name": "Robot Agricole Autonome", "sector": "Agriculture", "status": "Production"},
            {"name": "Drone Livraison Urbaine", "sector": "Logistique", "status": "Pilote"},
            {"name": "Exosquelette Médical", "sector": "Santé", "status": "Recherche"},
            {"name": "Robot Sous-Marin", "sector": "Exploration", "status": "Production"},
            {"name": "Assistant Domestique", "sector": "Service", "status": "Beta"},
            {"name": "Bras Industriel Collaboratif", "sector": "Industrie", "status": "Production"}
        ]
        
        for i, project in enumerate(projects_gallery):
            col = [col1, col2, col3][i % 3]
            
            with col:
                st.markdown(f"""
                <div class="robot-card">
                    <h4>{project['name']}</h4>
                    <p><strong>Secteur:</strong> {project['sector']}</p>
                    <p><strong>Statut:</strong> {project['status']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("👁️ Voir Détails", key=f"view_project_{i}"):
                    st.info(f"Détails du projet: {project['name']}")

                    

# ==================== PAGE: CONTRÔLE & COMMANDE ====================

elif page == "🎮 Contrôle & Commande":
    st.header("🎮 Systèmes de Contrôle et Commande")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎛️ Contrôleurs", "📈 Réglage PID", "🤖 Commande Robot", "📊 Monitoring"])
    
    with tab1:
        st.subheader("🎛️ Types de Contrôleurs")
        
        controllers = {
            "PID": {
                "description": "Contrôleur Proportionnel-Intégral-Dérivé",
                "parameters": ["Kp", "Ki", "Kd"],
                "advantages": ["Simple", "Robuste", "Bien connu"],
                "applications": ["Position", "Vitesse", "Température"],
                "performance": "Bonne pour systèmes linéaires"
            },
            "Fuzzy Logic": {
                "description": "Contrôle par logique floue",
                "parameters": ["Règles floues", "Fonctions d'appartenance"],
                "advantages": ["Gère l'incertitude", "Pas de modèle requis"],
                "applications": ["Systèmes complexes", "Non-linéaire"],
                "performance": "Excellente robustesse"
            },
            "MPC": {
                "description": "Model Predictive Control",
                "parameters": ["Horizon de prédiction", "Modèle dynamique"],
                "advantages": ["Prédictif", "Gère contraintes", "Optimal"],
                "applications": ["Multi-variables", "Trajectoires"],
                "performance": "Optimal avec calcul intensif"
            },
            "Adaptive": {
                "description": "Contrôle adaptatif",
                "parameters": ["Loi d'adaptation", "Modèle de référence"],
                "advantages": ["S'adapte", "Gère changements"],
                "applications": ["Environnement variable", "Incertitudes"],
                "performance": "Excellente adaptation"
            },
            "Neural": {
                "description": "Contrôle par réseau de neurones",
                "parameters": ["Architecture réseau", "Training data"],
                "advantages": ["Apprend", "Non-linéaire"],
                "applications": ["Systèmes complexes", "Pattern-based"],
                "performance": "Haute performance après training"
            }
        }

        for ctrl_name, ctrl_info in controllers.items():
            with st.expander(f"🎛️ {ctrl_name} - {ctrl_info['description']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Paramètres:**")
                    for param in ctrl_info['parameters']:
                        st.write(f"• {param}")
                    
                    st.write("\n**Avantages:**")
                    for adv in ctrl_info['advantages']:
                        st.write(f"✅ {adv}")
                
                with col2:
                    st.write("**Applications:**")
                    for app in ctrl_info['applications']:
                        st.write(f"• {app}")
                    
                    st.info(f"**Performance:** {ctrl_info['performance']}")
                        
    with tab2:
        st.subheader("🧪 Tests Unitaires des Composants")
        
        if not st.session_state.robotics_system['robots']:
            st.info("Aucun robot disponible pour les tests")
        else:
            robot_ids = list(st.session_state.robotics_system['robots'].keys())
            selected_robot = st.selectbox(
                "Sélectionner Robot à Tester",
                robot_ids,
                format_func=lambda x: st.session_state.robotics_system['robots'][x]['name']
            )
            
            robot = st.session_state.robotics_system['robots'][selected_robot]
            
            st.write(f"### 🤖 {robot['name']}")
            
            # Tests disponibles
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### ⚙️ Tests Actionneurs")
                
                if st.button("🔧 Test Tous Actionneurs"):
                    with st.spinner("Test en cours..."):
                        test_result = {
                            'test_id': f"test_{len(st.session_state.robotics_system.get('tests', []))}",
                            'robot_id': selected_robot,
                            'type': 'Actionneurs',
                            'timestamp': datetime.now().isoformat(),
                            'results': []
                        }
                        
                        progress = st.progress(0)
                        n_actuators = robot['components']['actuators']
                        
                        for i in range(n_actuators):
                            progress.progress((i + 1) / n_actuators)
                            
                            result = {
                                'actuator': f"Actuator {i+1}",
                                'status': 'PASS' if np.random.random() > 0.1 else 'FAIL',
                                'torque': np.random.uniform(80, 100),
                                'speed': np.random.uniform(90, 100),
                                'temperature': np.random.uniform(25, 45)
                            }
                            test_result['results'].append(result)
                        
                        progress.empty()
                        
                        if 'tests' not in st.session_state.robotics_system:
                            st.session_state.robotics_system['tests'] = []
                        
                        st.session_state.robotics_system['tests'].append(test_result)
                        
                        passed = sum(1 for r in test_result['results'] if r['status'] == 'PASS')
                        total = len(test_result['results'])
                        
                        if passed == total:
                            st.success(f"✅ Tous les tests réussis ({passed}/{total})")
                        else:
                            st.warning(f"⚠️ {passed}/{total} tests réussis")
                        
                        # Afficher résultats
                        for result in test_result['results']:
                            status_icon = "✅" if result['status'] == 'PASS' else "❌"
                            st.write(f"{status_icon} {result['actuator']}: {result['status']}")
            
            with col2:
                st.write("#### 📡 Tests Capteurs")
                
                if st.button("🔍 Test Tous Capteurs"):
                    with st.spinner("Test en cours..."):
                        n_sensors = robot['components']['sensors']
                        
                        st.write("**Résultats:**")
                        for i in range(n_sensors):
                            status = "✅ PASS" if np.random.random() > 0.05 else "❌ FAIL"
                            accuracy = np.random.uniform(90, 100)
                            st.write(f"Capteur {i+1}: {status} (Précision: {accuracy:.1f}%)")
            
            st.markdown("---")
            
            # Tests avancés
            st.write("### 🧪 Tests Avancés")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔋 Test Batterie"):
                    st.info("Test de capacité batterie lancé")
                    
                    # Courbe de décharge
                    time = np.linspace(0, robot['power']['autonomy'], 100)
                    charge = 100 * (1 - time / robot['power']['autonomy'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=time, y=charge,
                        mode='lines',
                        line=dict(color='green', width=3),
                        fill='tozeroy'
                    ))
                    
                    fig.update_layout(
                        title="Courbe de Décharge Batterie",
                        xaxis_title="Temps (h)",
                        yaxis_title="Charge (%)",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if st.button("🧠 Test IA"):
                    if robot['intelligence']['ai_enabled']:
                        st.success("✅ Système IA fonctionnel")
                        st.metric("Accuracy", f"{np.random.uniform(85, 99):.1f}%")
                        st.metric("Temps Inférence", f"{np.random.uniform(10, 50):.1f} ms")
                    else:
                        st.warning("IA non activée")
            
            with col3:
                if st.button("🔗 Test Communication"):
                    st.success("✅ Communication établie")
                    st.metric("Latence", f"{np.random.uniform(5, 20):.0f} ms")
                    st.metric("Débit", f"{np.random.uniform(10, 100):.0f} Mbps")
    
    with tab3:
        st.subheader("📊 Résultats des Tests")
        
        if 'tests' not in st.session_state.robotics_system or not st.session_state.robotics_system['tests']:
            st.info("Aucun test effectué")
        else:
            st.write(f"### 📋 {len(st.session_state.robotics_system['tests'])} Tests Effectués")
            
            # Tableau des tests
            test_summary = []
            for test in st.session_state.robotics_system['tests']:
                robot_name = st.session_state.robotics_system['robots'][test['robot_id']]['name']
                passed = sum(1 for r in test['results'] if r['status'] == 'PASS')
                total = len(test['results'])
                
                test_summary.append({
                    'Test ID': test['test_id'],
                    'Robot': robot_name[:20],
                    'Type': test['type'],
                    'Réussis': f"{passed}/{total}",
                    'Taux': f"{passed/total*100:.0f}%",
                    'Date': test['timestamp'][:19]
                })
            
            df = pd.DataFrame(test_summary)
            st.dataframe(df, use_container_width=True)
            
            # Statistiques globales
            st.markdown("---")
            st.write("### 📈 Statistiques Globales")
            
            total_tests = sum(len(t['results']) for t in st.session_state.robotics_system['tests'])
            total_passed = sum(sum(1 for r in t['results'] if r['status'] == 'PASS') 
                             for t in st.session_state.robotics_system['tests'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tests Totaux", total_tests)
            with col2:
                st.metric("Réussis", total_passed)
            with col3:
                st.metric("Échoués", total_tests - total_passed)
            with col4:
                st.metric("Taux de Réussite", f"{total_passed/total_tests*100:.1f}%")
    
    with tab4:
        st.subheader("📈 Benchmarks de Performance")
        
        st.write("### 🏆 Comparaison des Robots")
        
        if len(st.session_state.robotics_system['robots']) < 2:
            st.info("Créez au moins 2 robots pour comparer les performances")
        else:
            # Sélection robots à comparer
            robot_ids = list(st.session_state.robotics_system['robots'].keys())
            selected_robots = st.multiselect(
                "Sélectionner Robots à Comparer",
                robot_ids,
                format_func=lambda x: st.session_state.robotics_system['robots'][x]['name'],
                default=robot_ids[:min(3, len(robot_ids))]
            )
            
            if len(selected_robots) >= 2:
                # Métriques à comparer
                metrics = ['Vitesse Max', 'Précision', 'Autonomie', 'Intelligence', 'Charge Utile']
                
                # Préparation données
                comparison_data = []
                for robot_id in selected_robots:
                    robot = st.session_state.robotics_system['robots'][robot_id]
                    comparison_data.append({
                        'Robot': robot['name'][:15],
                        'Vitesse Max': robot['performance']['max_speed'],
                        'Précision': robot['performance']['precision'],
                        'Autonomie': robot['power']['autonomy'],
                        'Intelligence': robot['intelligence']['level'] * 100,
                        'Charge Utile': robot['specifications']['payload']
                    })
                
                df_compare = pd.DataFrame(comparison_data)
                
                # Graphique radar
                fig = go.Figure()
                
                for _, row in df_compare.iterrows():
                    # Normalisation
                    values = [
                        row['Vitesse Max'] / df_compare['Vitesse Max'].max() * 100,
                        100 - row['Précision'] * 10,  # Inverse car moins = mieux
                        row['Autonomie'] / df_compare['Autonomie'].max() * 100,
                        row['Intelligence'],
                        row['Charge Utile'] / df_compare['Charge Utile'].max() * 100
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values + [values[0]],  # Fermer le polygone
                        theta=metrics + [metrics[0]],
                        name=row['Robot'],
                        fill='toself'
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="Comparaison Multi-critères",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau détaillé
                st.markdown("---")
                st.write("### 📋 Données Détaillées")
                st.dataframe(df_compare, use_container_width=True)
                
                # Classement
                st.markdown("---")
                st.write("### 🏆 Classements")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**🏃 Plus Rapide:**")
                    fastest = df_compare.nlargest(3, 'Vitesse Max')
                    for i, row in fastest.iterrows():
                        st.write(f"{i+1}. {row['Robot']}: {row['Vitesse Max']:.2f} m/s")
                
                with col2:
                    st.write("**🎯 Plus Précis:**")
                    most_precise = df_compare.nsmallest(3, 'Précision')
                    for i, row in most_precise.iterrows():
                        st.write(f"{i+1}. {row['Robot']}: {row['Précision']:.3f} mm")

# ==================== PAGE: FABRICATION ====================

elif page == "🏭 Fabrication":
    st.header("🏭 Système de Fabrication")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Commandes", "🔧 Production", "📊 Suivi", "💰 Coûts"])
    
    with tab1:
        st.subheader("📦 Nouvelle Commande de Fabrication")
        
        if not st.session_state.robotics_system['robots']:
            st.info("Créez d'abord un robot pour lancer une fabrication")
        else:
            with st.form("manufacturing_order"):
                st.write("### 🤖 Sélection du Robot")
                
                robot_ids = list(st.session_state.robotics_system['robots'].keys())
                selected_robot = st.selectbox(
                    "Robot à Fabriquer",
                    robot_ids,
                    format_func=lambda x: st.session_state.robotics_system['robots'][x]['name']
                )
                
                robot = st.session_state.robotics_system['robots'][selected_robot]
                
                st.write(f"**Type:** {robot['type'].replace('_', ' ').title()}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    quantity = st.number_input("Quantité", 1, 1000, 1)
                
                with col2:
                    priority = st.selectbox("Priorité", ["Normale", "Élevée", "Urgente"])
                
                with col3:
                    quality = st.selectbox("Niveau Qualité", ["Standard", "Premium", "Prototype"])
                
                st.markdown("---")
                st.write("### 🏭 Paramètres de Production")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    manufacturing_process = st.multiselect(
                        "Procédés de Fabrication",
                        ["Impression 3D", "Usinage CNC", "Moulage", "Assemblage Manuel", 
                         "Assemblage Automatisé", "Soudage", "Peinture"],
                        default=["Impression 3D", "Assemblage Automatisé"]
                    )
                    
                    materials = st.multiselect(
                        "Matériaux",
                        ["Aluminium", "Acier", "Plastique ABS", "Fibre de Carbone", 
                         "Titane", "Composites"],
                        default=["Aluminium", "Plastique ABS"]
                    )
                
                with col2:
                    testing_level = st.select_slider(
                        "Niveau de Tests",
                        options=["Minimal", "Standard", "Approfondi", "Exhaustif"],
                        value="Standard"
                    )
                    
                    delivery_date = st.date_input(
                        "Date de Livraison Souhaitée",
                        value=datetime.now() + timedelta(days=30)
                    )
                
                st.markdown("---")
                st.write("### 💰 Estimation des Coûts")
                
                # Calcul coûts
                unit_cost = robot['costs']['manufacturing']
                material_cost = unit_cost * 0.4
                labor_cost = unit_cost * 0.3
                overhead_cost = unit_cost * 0.3
                
                quality_multiplier = {"Standard": 1.0, "Premium": 1.5, "Prototype": 2.0}[quality]
                total_cost = unit_cost * quantity * quality_multiplier
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Coût Unitaire", f"${unit_cost:,.0f}")
                with col2:
                    st.metric("Matériaux", f"${material_cost * quantity:,.0f}")
                with col3:
                    st.metric("Main d'Œuvre", f"${labor_cost * quantity:,.0f}")
                with col4:
                    st.metric("TOTAL", f"${total_cost:,.0f}")
                
                submitted = st.form_submit_button("🚀 Lancer Fabrication", type="primary")
                
                if submitted:
                    order = {
                        'order_id': f"MFG_{len(st.session_state.robotics_system['manufacturing']) + 1:04d}",
                        'robot_id': selected_robot,
                        'robot_name': robot['name'],
                        'quantity': quantity,
                        'priority': priority,
                        'quality': quality,
                        'processes': manufacturing_process,
                        'materials': materials,
                        'testing_level': testing_level,
                        'delivery_date': delivery_date.isoformat(),
                        'status': 'En Attente',
                        'progress': 0,
                        'cost': total_cost,
                        'order_date': datetime.now().isoformat(),
                        'estimated_duration': 30,  # jours
                        'completed_units': 0
                    }
                    
                    st.session_state.robotics_system['manufacturing'].append(order)
                    
                    st.success(f"✅ Commande {order['order_id']} créée avec succès!")
                    st.balloons()
                    
                    log_event(f"Commande fabrication: {quantity}x {robot['name']}")
    
    with tab2:
        st.subheader("🔧 Production en Cours")
        
        if not st.session_state.robotics_system['manufacturing']:
            st.info("Aucune commande de fabrication en cours")
        else:
            active_orders = [o for o in st.session_state.robotics_system['manufacturing'] 
                           if o['status'] != 'Terminée']
            
            if not active_orders:
                st.info("Aucune production active")
            else:
                for order in active_orders:
                    with st.expander(f"📦 {order['order_id']} - {order['robot_name']} (x{order['quantity']})", 
                                   expanded=True):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Statut:** {order['status']}")
                            st.write(f"**Priorité:** {order['priority']}")
                            st.write(f"**Qualité:** {order['quality']}")
                            
                            # Barre de progression
                            st.progress(order['progress'] / 100)
                            st.write(f"Progression: {order['progress']:.0f}%")
                        
                        with col2:
                            st.metric("Complétés", f"{order['completed_units']}/{order['quantity']}")
                            st.metric("Jours Restants", 
                                    max(0, order['estimated_duration'] - int(order['progress'] / 100 * order['estimated_duration'])))
                        
                        with col3:
                            st.metric("Coût Total", f"${order['cost']:,.0f}")
                            
                            # Actions
                            if st.button("▶️ Avancer Production", key=f"prod_{order['order_id']}"):
                                order['progress'] = min(100, order['progress'] + 10)
                                order['completed_units'] = int(order['quantity'] * order['progress'] / 100)
                                
                                if order['progress'] >= 100:
                                    order['status'] = 'Terminée'
                                    st.success("✅ Production terminée!")
                                elif order['progress'] < 30:
                                    order['status'] = 'Fabrication'
                                elif order['progress'] < 70:
                                    order['status'] = 'Assemblage'
                                else:
                                    order['status'] = 'Tests'
                                
                                st.rerun()
                        
                        # Détails processus
                        st.write("**Processus:**")
                        process_steps = {
                            0: "⏳ En attente",
                            20: "🔧 Fabrication pièces",
                            40: "🔩 Assemblage structure",
                            60: "⚡ Installation électronique",
                            80: "🧪 Tests et calibration",
                            100: "✅ Contrôle qualité final"
                        }
                        
                        current_step = max([k for k in process_steps.keys() if k <= order['progress']])
                        st.info(f"Étape actuelle: {process_steps[current_step]}")
    
    with tab3:
        st.subheader("📊 Suivi de Production")
        
        if st.session_state.robotics_system['manufacturing']:
            st.write("### 📋 Toutes les Commandes")
            
            # Tableau récapitulatif
            orders_data = []
            for order in st.session_state.robotics_system['manufacturing']:
            # for i, (order, robot) in enumerate(st.session_state.robotics_system['robots'].items()):
                orders_data.append({
                    'N° Commande': order['order_id'],
                    'Robot': order['robot_name'][:20],
                    'Quantité': order['quantity'],
                    'Complétés': order['completed_units'],
                    'Statut': order['status'],
                    'Progression': f"{order['progress']:.0f}%",
                    'Priorité': order['priority'],
                    'Coût': f"${order['cost']:,.0f}",
                    'Date Commande': order['order_date'][:10]
                })
            
            df = pd.DataFrame(orders_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            
            # Statistiques
            st.write("### 📈 Statistiques de Production")
            
            total_orders = len(st.session_state.robotics_system['manufacturing'])
            completed_orders = sum(1 for o in st.session_state.robotics_system['manufacturing'] 
                                 if o['status'] == 'Terminée')
            total_units = sum(o['quantity'] for o in st.session_state.robotics_system['manufacturing'])
            completed_units = sum(o['completed_units'] for o in st.session_state.robotics_system['manufacturing'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Commandes Totales", total_orders)
            with col2:
                st.metric("Commandes Terminées", completed_orders)
            with col3:
                st.metric("Unités Totales", total_units)
            with col4:
                st.metric("Unités Produites", completed_units)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                # Répartition par statut
                status_counts = {}
                # for order in st.session_state.robotics_system['manufacturing']:
                for i, (robot_id, robot) in enumerate(st.session_state.robotics_system['robots'].items()):
                    status = order['status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                fig = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="Répartition par Statut"
                )
                order_hash = hashlib.md5(json.dumps(robot, sort_keys=True).encode()).hexdigest()
                unique_key = f"robot_plot_{order_hash}_{i}"
                
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
            
            with col2:
                # Production par robot
                robot_counts = {}
                # for order in st.session_state.robotics_system['manufacturing']:
                for i, (robot_id, robot) in enumerate(st.session_state.robotics_system['robots'].items()):
                    robot = order['robot_name']
                    robot_counts[robot] = robot_counts.get(robot, 0) + order['quantity']
                
                fig = go.Figure(data=[
                    go.Bar(x=list(robot_counts.keys()), y=list(robot_counts.values()),
                          marker_color='rgb(102, 126, 234)')
                ])
                fig.update_layout(title="Production par Type de Robot",
                                xaxis_title="Robot", yaxis_title="Quantité")
                order_hash = hashlib.md5(json.dumps(robot, sort_keys=True).encode()).hexdigest()
                uniquer_key = f"robot_plot_{order_hash}_{i}"
                
                st.plotly_chart(fig, use_container_width=True, key=uniquer_key)
        else:
            st.info("Aucune donnée de production")
    
    with tab4:
        st.subheader("💰 Analyse des Coûts")
        
        if st.session_state.robotics_system['manufacturing']:
            # Coûts totaux
            total_cost = sum(o['cost'] for o in st.session_state.robotics_system['manufacturing'])
            completed_cost = sum(o['cost'] for o in st.session_state.robotics_system['manufacturing'] 
                               if o['status'] == 'Terminée')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Coût Total Engagé", f"${total_cost:,.0f}")
            with col2:
                st.metric("Coût Complété", f"${completed_cost:,.0f}")
            with col3:
                st.metric("En Production", f"${total_cost - completed_cost:,.0f}")
            
            st.markdown("---")
            
            # Répartition des coûts
            st.write("### 📊 Répartition des Coûts")
            
            cost_breakdown = {
                'Matériaux': 40,
                'Main d\'Œuvre': 30,
                'Équipement': 15,
                'Tests': 10,
                'Overhead': 5
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(cost_breakdown.keys()),
                values=list(cost_breakdown.values()),
                hole=0.4
            )])
            
            fig.update_layout(
                title="Distribution des Coûts (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Coût par commande
            st.markdown("---")
            st.write("### 💵 Coûts par Commande")
            
            orders_sorted = sorted(st.session_state.robotics_system['manufacturing'],
                                 key=lambda x: x['cost'], reverse=True)
            
            order_names = [o['order_id'] for o in orders_sorted[:10]]
            order_costs = [o['cost'] for o in orders_sorted[:10]]
            
            fig = go.Figure(data=[
                go.Bar(x=order_names, y=order_costs,
                      marker_color='lightblue',
                      text=[f"${c:,.0f}" for c in order_costs],
                      textposition='outside')
            ])
            
            fig.update_layout(
                title="Top 10 Commandes par Coût",
                xaxis_title="N° Commande",
                yaxis_title="Coût ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée de coûts")

# ==================== PAGE: ASSEMBLAGE ====================

elif page == "⚙️ Assemblage":
    st.header("⚙️ Chaîne d'Assemblage")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Instructions", "📦 Inventaire", "🤖 Assemblage Guidé"])
    
    with tab1:
        st.subheader("🔧 Instructions d'Assemblage")
        
        if not st.session_state.robotics_system['robots']:
            st.info("Créez un robot pour voir les instructions")
        else:
            robot_ids = list(st.session_state.robotics_system['robots'].keys())
            selected = st.selectbox(
                "Sélectionner Robot",
                robot_ids,
                format_func=lambda x: st.session_state.robotics_system['robots'][x]['name']
            )
            
            robot = st.session_state.robotics_system['robots'][selected]
            
            st.write(f"### 📋 Instructions pour {robot['name']}")
            
            # Étapes d'assemblage
            assembly_steps = [
                {
                    "step": 1,
                    "title": "Préparation de la Structure",
                    "description": "Assemblage du châssis principal et montage des supports",
                    "parts": ["Châssis principal", "4x Supports moteur", "Vis M6x20 (x24)"],
                    "tools": ["Clé Allen 4mm", "Tournevis cruciforme"],
                    "duration": "30 min",
                    "difficulty": "Facile"
                },
                {
                    "step": 2,
                    "title": "Installation des Actionneurs",
                    "description": "Montage des moteurs et servos sur la structure",
                    "parts": [f"{robot['components']['actuators']}x Actionneurs", 
                             "Câbles d'alimentation", "Connecteurs"],
                    "tools": ["Clé dynamométrique", "Pince"],
                    "duration": "45 min",
                    "difficulty": "Moyen"
                },
                {
                    "step": 3,
                    "title": "Câblage Électrique",
                    "description": "Connexion de tous les composants électriques",
                    "parts": ["Contrôleur principal", "Fils électriques", "Gaine thermorétractable"],
                    "tools": ["Fer à souder", "Multimètre", "Pince à dénuder"],
                    "duration": "60 min",
                    "difficulty": "Difficile"
                },
                {
                    "step": 4,
                    "title": "Installation des Capteurs",
                    "description": "Montage et calibration des capteurs",
                    "parts": [f"{robot['components']['sensors']}x Capteurs", "Supports capteurs", "Câbles données"],
                    "tools": ["Tournevis de précision", "Logiciel calibration"],
                    "duration": "40 min",
                    "difficulty": "Moyen"
                },
                {
                    "step": 5,
                    "title": "Système d'Alimentation",
                    "description": "Installation batterie et gestion d'énergie",
                    "parts": ["Batterie", "BMS", "Câbles alimentation", "Connecteurs XT60"],
                    "tools": ["Multimètre", "Testeur batterie"],
                    "duration": "25 min",
                    "difficulty": "Facile"
                },
                {
                    "step": 6,
                    "title": "Tests et Calibration",
                    "description": "Vérification fonctionnelle et calibration finale",
                    "parts": ["Logiciel de test", "Checklist qualité"],
                    "tools": ["Ordinateur", "Câble USB", "Multimètre"],
                    "duration": "90 min",
                    "difficulty": "Moyen"
                }
            ]
            
            if robot['intelligence']['ai_enabled']:
                assembly_steps.append({
                    "step": 7,
                    "title": "Installation Système IA",
                    "description": "Configuration du système d'intelligence artificielle",
                    "parts": ["Module IA", "GPU", "Refroidissement"],
                    "tools": ["Logiciel configuration", "Pâte thermique"],
                    "duration": "45 min",
                    "difficulty": "Difficile"
                })
            
            if robot['advanced_systems']['quantum']:
                assembly_steps.append({
                    "step": 8,
                    "title": "Intégration Processeur Quantique",
                    "description": "Installation et isolation du QPU",
                    "parts": ["QPU", "Système cryogénique", "Blindage magnétique"],
                    "tools": ["Équipement spécialisé", "Chambre propre"],
                    "duration": "120 min",
                    "difficulty": "Expert"
                })
            
            # Affichage des étapes
            for step_info in assembly_steps:
                with st.expander(f"Étape {step_info['step']}: {step_info['title']}", expanded=False):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {step_info['description']}")
                        
                        st.write("\n**Pièces Requises:**")
                        for part in step_info['parts']:
                            st.write(f"• {part}")
                        
                        st.write("\n**Outils Nécessaires:**")
                        for tool in step_info['tools']:
                            st.write(f"🔧 {tool}")
                    
                    with col2:
                        difficulty_colors = {
                            "Facile": "🟢",
                            "Moyen": "🟡",
                            "Difficile": "🟠",
                            "Expert": "🔴"
                        }
                        
                        st.metric("Durée Estimée", step_info['duration'])
                        st.write(f"**Difficulté:** {difficulty_colors[step_info['difficulty']]} {step_info['difficulty']}")
                        
                        if st.button("✅ Marquer comme Complétée", key=f"complete_step_{step_info['step']}"):
                            st.success(f"Étape {step_info['step']} complétée!")
            
            # Temps total
            st.markdown("---")
            total_time = sum([30, 45, 60, 40, 25, 90])
            if robot['intelligence']['ai_enabled']:
                total_time += 45
            if robot['advanced_systems']['quantum']:
                total_time += 120
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temps Total", f"{total_time} min")
            with col2:
                st.metric("Étapes", len(assembly_steps))
            with col3:
                st.metric("Personnel Requis", "2-3 personnes")
    
    with tab2:
        st.subheader("📦 Inventaire des Composants")
        
        st.write("### 🔍 Composants en Stock")
        
        # Inventaire simulé
        inventory = {
            "Actionneurs": {
                "Moteur DC Brushless": {"stock": 45, "min": 20, "prix": 250},
                "Servo-Moteur": {"stock": 78, "min": 30, "prix": 85},
                "Moteur Pas-à-Pas": {"stock": 12, "min": 15, "prix": 45},
            },
            "Capteurs": {
                "Caméra RGB": {"stock": 23, "min": 10, "prix": 120},
                "LiDAR": {"stock": 8, "min": 5, "prix": 1200},
                "IMU": {"stock": 56, "min": 20, "prix": 65},
                "Capteur Force": {"stock": 15, "min": 10, "prix": 180},
            },
            "Électronique": {
                "Contrôleur Principal": {"stock": 18, "min": 10, "prix": 350},
                "Raspberry Pi 4": {"stock": 32, "min": 15, "prix": 75},
                "Arduino Mega": {"stock": 41, "min": 20, "prix": 45},
            },
            "Alimentation": {
                "Batterie Li-Po 5000mAh": {"stock": 25, "min": 15, "prix": 120},
                "BMS": {"stock": 20, "min": 10, "prix": 45},
                "Chargeur": {"stock": 15, "min": 10, "prix": 65},
            },
            "Mécanique": {
                "Châssis Aluminium": {"stock": 14, "min": 5, "prix": 280},
                "Support Moteur": {"stock": 67, "min": 30, "prix": 15},
                "Vis/Écrous (lot)": {"stock": 150, "min": 50, "prix": 8},
            }
        }
        
        for category, items in inventory.items():
            st.write(f"#### {category}")
            
            items_data = []
            for item_name, item_info in items.items():
                status = "✅" if item_info['stock'] >= item_info['min'] else "⚠️"
                items_data.append({
                    'Statut': status,
                    'Composant': item_name,
                    'Stock': item_info['stock'],
                    'Min Requis': item_info['min'],
                    'Prix Unit.': f"${item_info['prix']}",
                    'Valeur Stock': f"${item_info['stock'] * item_info['prix']}"
                })
            
            df = pd.DataFrame(items_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
        
        # Alertes stock
        st.write("### ⚠️ Alertes de Stock")
        
        low_stock = []
        for category, items in inventory.items():
            for item_name, item_info in items.items():
                if item_info['stock'] < item_info['min']:
                    low_stock.append(f"{item_name} ({category}): {item_info['stock']} unités")
        
        if low_stock:
            for alert in low_stock:
                st.warning(f"⚠️ Stock faible: {alert}")
        else:
            st.success("✅ Tous les stocks sont au niveau optimal")
    
    with tab3:
        st.subheader("🤖 Assemblage Guidé Interactif")
        
        st.write("### 🎯 Mode Pas-à-Pas")
        
        if 'assembly_progress' not in st.session_state:
            st.session_state.assembly_progress = {
                'current_step': 0,
                'steps_completed': [],
                'start_time': None
            }
        
        total_steps = 6
        current_step = st.session_state.assembly_progress['current_step']
        
        # Barre de progression
        st.progress(current_step / total_steps)
        st.write(f"**Progression:** {current_step}/{total_steps} étapes complétées")
        
        if current_step >= total_steps:
            st.success("🎉 Assemblage Terminé!")
            st.balloons()
            
            if st.button("🔄 Recommencer"):
                st.session_state.assembly_progress = {
                    'current_step': 0,
                    'steps_completed': [],
                    'start_time': None
                }
                st.rerun()
        else:
            st.markdown("---")
            
            step_details = assembly_steps[current_step]
            
            st.write(f"## Étape {step_details['step']}: {step_details['title']}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"### 📝 Instructions")
                st.info(step_details['description'])
                
                st.write("**Pièces Nécessaires:**")
                for part in step_details['parts']:
                    st.checkbox(part, key=f"part_{current_step}_{part}")
                
                st.write("\n**Outils:**")
                for tool in step_details['tools']:
                    st.write(f"🔧 {tool}")
            
            with col2:
                st.write("### ⏱️ Informations")
                st.metric("Durée", step_details['duration'])
                st.metric("Difficulté", step_details['difficulty'])
                
                st.write("\n### ✅ Validation")
                
                # Checklist
                checks = [
                    "Toutes les pièces sont présentes",
                    "Les outils sont prêts",
                    "Instructions comprises"
                ]
                
                all_checked = True
                for check in checks:
                    if not st.checkbox(check, key=f"check_{current_step}_{check}"):
                        all_checked = False
                
                if all_checked:
                    if st.button("➡️ Étape Suivante", type="primary", use_container_width=True):
                        st.session_state.assembly_progress['current_step'] += 1
                        st.session_state.assembly_progress['steps_completed'].append(step_details['step'])
                        st.rerun()
                else:
                    st.button("➡️ Étape Suivante", type="primary", use_container_width=True, disabled=True)
            
            st.markdown("---")
            
            # Aide vidéo/image
            st.write("### 📹 Aide Visuelle")
            st.info("💡 Vidéo d'instruction disponible (placeholder)")

# ==================== PAGE: ANALYSES & RÉSULTATS ====================

elif page == "📊 Analyses & Résultats":
    st.header("📊 Analyses et Résultats")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Analytics", "🔬 Données", "📑 Rapports", "💡 Insights"])
    
    with tab1:
        st.subheader("📈 Tableau de Bord Analytique")
        
        if not st.session_state.robotics_system['robots']:
            st.info("Aucune donnée à analyser")
        else:
            # KPIs globaux
            st.write("### 🎯 Indicateurs Clés de Performance")
            
            total_robots = len(st.session_state.robotics_system['robots'])
            total_missions = sum(r['operations']['missions'] for r in st.session_state.robotics_system['robots'].values())
            avg_success = np.mean([r['operations']['success_rate'] for r in st.session_state.robotics_system['robots'].values()])
            total_hours = sum(r['operations']['hours'] for r in st.session_state.robotics_system['robots'].values())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Robots Totaux", total_robots, delta="+2 ce mois")
            with col2:
                st.metric("Missions Complétées", total_missions)
            with col3:
                st.metric("Taux Succès Moyen", f"{avg_success:.1f}%", delta="+3.2%")
            with col4:
                st.metric("Heures Opération", f"{total_hours:.0f}h")
            
            st.markdown("---")
            
            # Graphiques d'analyse
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 Performance par Robot")
                
                robot_names = [r['name'][:15] for r in st.session_state.robotics_system['robots'].values()]
                success_rates = [r['operations']['success_rate'] for r in st.session_state.robotics_system['robots'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=robot_names, y=success_rates,
                          marker_color='lightblue',
                          text=[f"{s:.1f}%" for s in success_rates],
                          textposition='outside')
                ])
                
                fig.update_layout(
                    title="Taux de Succès par Robot",
                    yaxis_title="Taux de Succès (%)",
                    xaxis_tickangle=-45,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### ⚡ Efficacité Énergétique")
                
                robot_names = [r['name'][:15] for r in st.session_state.robotics_system['robots'].values()]
                efficiency = [r['power']['autonomy'] / r['power']['consumption'] * 100 
                            for r in st.session_state.robotics_system['robots'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=robot_names, y=efficiency,
                          marker_color='lightgreen')
                ])
                
                fig.update_layout(
                    title="Efficacité Énergétique",
                    yaxis_title="Score d'Efficacité",
                    xaxis_tickangle=-45,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Analyse temporelle
            st.write("### 📅 Analyse Temporelle")
            
            # Simulation de données temporelles
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            missions_per_day = np.random.poisson(5, 30)
            success_per_day = 85 + np.random.randn(30) * 5
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Missions Quotidiennes", "Taux de Succès Quotidien")
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=missions_per_day, mode='lines+markers',
                          name='Missions', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=dates, y=success_per_day, mode='lines+markers',
                          name='Succès %', line=dict(color='green', width=2)),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Nombre", row=1, col=1)
            fig.update_yaxes(title_text="Succès (%)", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Exploration des Données")
        
        if st.session_state.robotics_system['robots']:
            st.write("### 📋 Données Brutes")
            
            # Export format
            export_format = st.selectbox("Format d'Export", ["DataFrame", "JSON", "CSV"])
            
            # Préparation données
            data_export = []
            for robot_id, robot in st.session_state.robotics_system['robots'].items():
                data_export.append({
                    'ID': robot_id,
                    'Nom': robot['name'],
                    'Type': robot['type'],
                    'Poids (kg)': robot['specifications']['weight'],
                    'DoF': robot['specifications']['dof'],
                    'Vitesse Max (m/s)': robot['performance']['max_speed'],
                    'Autonomie (h)': robot['power']['autonomy'],
                    'Intelligence': robot['intelligence']['level'],
                    'Missions': robot['operations']['missions'],
                    'Taux Succès (%)': robot['operations']['success_rate'],
                    'Heures Op': robot['operations']['hours'],
                    'Santé': robot['health']
                })
            
            df_export = pd.DataFrame(data_export)
            
            if export_format == "DataFrame":
                st.dataframe(df_export, use_container_width=True)
            elif export_format == "JSON":
                st.json(data_export)
            else:
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name="robots_data.csv",
                    mime="text/csv"
                )
                st.dataframe(df_export, use_container_width=True)
            
            st.markdown("---")
            
            # Statistiques descriptives
            st.write("### 📊 Statistiques Descriptives")
            
            numeric_cols = df_export.select_dtypes(include=[np.number]).columns
            stats_df = df_export[numeric_cols].describe()
            
            st.dataframe(stats_df, use_container_width=True)
            
            st.markdown("---")
            
            # Corrélations
            st.write("### 🔗 Matrice de Corrélation")
            
            corr_matrix = df_export[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(
                title="Corrélations entre Variables",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée disponible")
    
    with tab3:
        st.subheader("📑 Génération de Rapports")
        
        st.write("### 📝 Créer un Rapport Personnalisé")
        
        with st.form("report_generation"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_type = st.selectbox(
                    "Type de Rapport",
                    ["Performance Globale", "Analyse Coûts", "Maintenance", 
                     "Production", "Tests & Qualité", "Rapport Complet"]
                )
                
                period = st.selectbox(
                    "Période",
                    ["Dernière Semaine", "Dernier Mois", "Dernier Trimestre", 
                     "Dernière Année", "Personnalisée"]
                )
            
            with col2:
                format_rapport = st.multiselect(
                    "Inclure dans le Rapport",
                    ["Graphiques", "Tableaux", "Statistiques", "Recommandations", "Photos"],
                    default=["Graphiques", "Tableaux", "Statistiques"]
                )
                
                export_format = st.radio("Format d'Export", ["PDF", "HTML", "Markdown"])
            
            submitted = st.form_submit_button("📄 Générer Rapport", type="primary")
            
        if submitted:
            with st.spinner("Génération du rapport en cours..."):
                st.success("✅ Rapport généré avec succès!")
                            
                # Aperçu du rapport
                st.write("### 📄 Aperçu du Rapport")
                            
                st.markdown(f"""## Rapport: {report_type}
                            
                **Période:** {period}  
                **Date de Génération:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            
                ## Résumé Exécutif
                            
                Ce rapport présente une analyse détaillée de {report_type.lower()} pour la période sélectionnée.
                            
                ### Faits Marquants
                - {len(st.session_state.robotics_system['robots'])} robots actifs
                - {sum(r['operations']['missions'] for r in st.session_state.robotics_system['robots'].values())} missions complétées
                - Taux de succès moyen: {np.mean([r['operations']['success_rate'] for r in st.session_state.robotics_system['robots'].values()]):.1f}%
                            
                ### Recommandations
                1. Continuer le monitoring des performances
                2. Planifier maintenance préventive
                3. Optimiser la consommation énergétique
                """)            
                            
                st.download_button(
                    label=f"📥 Télécharger Rapport ({export_format})",
                    data="Contenu du rapport (placeholder)",
                    file_name=f"rapport_{report_type.lower().replace(' ', '_')}.{export_format.lower()}",
                    mime="application/octet-stream"
                )
    
    with tab4:
        st.subheader("💡 Insights et Recommandations")
        
        st.write("### 🤖 Analyse Intelligente")
        
        if st.session_state.robotics_system['robots']:
            # Insights automatiques
            insights = []
            
            # Analyse performance
            avg_success = np.mean([r['operations']['success_rate'] for r in st.session_state.robotics_system['robots'].values()])
            if avg_success > 90:
                insights.append({
                    'type': 'success',
                    'title': 'Excellente Performance',
                    'message': f'Le taux de succès moyen ({avg_success:.1f}%) est excellent. Continuez vos bonnes pratiques!'
                })
            elif avg_success < 80:
                insights.append({
                    'type': 'warning',
                    'title': 'Performance à Améliorer',
                    'message': f'Le taux de succès ({avg_success:.1f}%) pourrait être amélioré. Analysez les causes d\'échec.'
                })
            
            # Analyse énergie
            low_autonomy = [r for r in st.session_state.robotics_system['robots'].values() if r['power']['autonomy'] < 2]
            if low_autonomy:
                insights.append({
                    'type': 'warning',
                    'title': 'Autonomie Limitée',
                    'message': f'{len(low_autonomy)} robot(s) ont une autonomie < 2h. Considérez des batteries plus grandes.'
                })
            
            # Analyse technologie
            ai_robots = sum(1 for r in st.session_state.robotics_system['robots'].values() if r['intelligence']['ai_enabled'])
            if ai_robots < len(st.session_state.robotics_system['robots']) * 0.5:
                insights.append({
                    'type': 'info',
                    'title': 'Potentiel IA',
                    'message': f'Seulement {ai_robots} robots avec IA. Envisagez d\'intégrer l\'IA pour améliorer l\'intelligence.'
                })
            
            # Affichage insights
            for insight in insights:
                if insight['type'] == 'success':
                    st.success(f"**{insight['title']}:** {insight['message']}")
                elif insight['type'] == 'warning':
                    st.warning(f"**{insight['title']}:** {insight['message']}")
                else:
                    st.info(f"**{insight['title']}:** {insight['message']}")
            
            st.markdown("---")
            
            # Recommandations
            st.write("### 🎯 Recommandations Stratégiques")
            
            recommendations = [
                {
                    'priority': 'Haute',
                    'category': 'Performance',
                    'recommendation': 'Implémenter un système de maintenance prédictive',
                    'impact': 'Réduction de 30% des pannes',
                    'effort': 'Moyen'
                },
                {
                    'priority': 'Moyenne',
                    'category': 'Technologie',
                    'recommendation': 'Upgrade vers capteurs LiDAR nouvelle génération',
                    'impact': 'Amélioration précision navigation de 25%',
                    'effort': 'Élevé'
                },
                {
                    'priority': 'Haute',
                    'category': 'Énergie',
                    'recommendation': 'Optimiser algorithmes de gestion d\'énergie',
                    'impact': 'Augmentation autonomie de 15%',
                    'effort': 'Faible'
                },
                {
                    'priority': 'Basse',
                    'category': 'Formation',
                    'recommendation': 'Formation équipe sur nouvelles fonctionnalités IA',
                    'impact': 'Meilleure utilisation des capacités',
                    'effort': 'Faible'
                }
            ]
            
            for rec in recommendations:
                priority_color = {'Haute': '🔴', 'Moyenne': '🟡', 'Basse': '🟢'}
                
                with st.expander(f"{priority_color[rec['priority']]} [{rec['priority']}] {rec['recommendation']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Catégorie:** {rec['category']}")
                        st.write(f"**Impact Attendu:** {rec['impact']}")
                    
                    with col2:
                        st.write(f"**Effort:** {rec['effort']}")
                        if st.button("✅ Implémenter", key=f"rec_{rec['recommendation'][:20]}"):
                            st.success("Recommandation ajoutée au plan d'action")
        else:
            st.info("Créez des robots pour obtenir des insights")

# ==================== PAGE: DÉPLOIEMENT ====================

elif page == "🚀 Déploiement":
    st.header("🚀 Déploiement et Mise en Production")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📍 Sites", "🚀 Déployer", "🌐 Gestion Flotte", "📡 Monitoring"])
    
    with tab1:
        st.subheader("📍 Sites de Déploiement")
        
        st.write("### 🗺️ Gestion des Sites")
        
        if 'deployment_sites' not in st.session_state:
            st.session_state.deployment_sites = [
                {
                    'site_id': 'SITE_001',
                    'name': 'Entrepôt Paris Nord',
                    'location': 'Paris, France',
                    'type': 'Industriel',
                    'size': '5000 m²',
                    'robots_deployed': 0,
                    'capacity': 20,
                    'status': 'Actif'
                },
                {
                    'site_id': 'SITE_002',
                    'name': 'Hôpital Central Lyon',
                    'location': 'Lyon, France',
                    'type': 'Médical',
                    'size': '1200 m²',
                    'robots_deployed': 0,
                    'capacity': 8,
                    'status': 'Actif'
                }
            ]
        
        # Affichage sites
        for site in st.session_state.deployment_sites:
            with st.expander(f"📍 {site['name']} ({site['site_id']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Localisation:** {site['location']}")
                    st.write(f"**Type:** {site['type']}")
                    st.write(f"**Taille:** {site['size']}")
                
                with col2:
                    st.metric("Robots Déployés", f"{site['robots_deployed']}/{site['capacity']}")
                    st.progress(site['robots_deployed'] / site['capacity'])
                
                with col3:
                    status_color = {"Actif": "🟢", "Maintenance": "🟡", "Inactif": "🔴"}
                    st.write(f"**Statut:** {status_color[site['status']]} {site['status']}")
                    
                    if st.button("👁️ Voir Détails", key=f"view_site_{site['site_id']}"):
                        st.info(f"Détails complets du site {site['name']}")
        
        # Ajouter nouveau site
        st.markdown("---")
        st.write("### ➕ Ajouter Nouveau Site")
        
        with st.form("new_site"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_site_name = st.text_input("Nom du Site")
                new_site_location = st.text_input("Localisation")
                new_site_type = st.selectbox("Type", ["Industriel", "Médical", "Commercial", "Recherche"])
            
            with col2:
                new_site_size = st.text_input("Taille", "1000 m²")
                new_site_capacity = st.number_input("Capacité (robots)", 1, 100, 10)
            
            if st.form_submit_button("➕ Créer Site"):
                new_site = {
                    'site_id': f"SITE_{len(st.session_state.deployment_sites) + 1:03d}",
                    'name': new_site_name,
                    'location': new_site_location,
                    'type': new_site_type,
                    'size': new_site_size,
                    'robots_deployed': 0,
                    'capacity': new_site_capacity,
                    'status': 'Actif'
                }
                st.session_state.deployment_sites.append(new_site)
                st.success(f"✅ Site '{new_site_name}' créé!")
                log_event(f"Nouveau site créé: {new_site_name}")
    
    with tab2:
        st.subheader("🚀 Déployer un Robot")
        
        if not st.session_state.robotics_system['robots']:
            st.warning("Aucun robot disponible pour déploiement")
        elif not st.session_state.deployment_sites:
            st.warning("Aucun site de déploiement configuré")
        else:
            with st.form("deploy_robot"):
                st.write("### 🤖 Configuration du Déploiement")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    robot_ids = list(st.session_state.robotics_system['robots'].keys())
                    selected_robot = st.selectbox(
                        "Sélectionner Robot",
                        robot_ids,
                        format_func=lambda x: st.session_state.robotics_system['robots'][x]['name']
                    )
                    
                    site_ids = [s['site_id'] for s in st.session_state.deployment_sites]
                    selected_site = st.selectbox(
                        "Site de Déploiement",
                        site_ids,
                        format_func=lambda x: next(s['name'] for s in st.session_state.deployment_sites if s['site_id'] == x)
                    )
                
                with col2:
                    deployment_mode = st.selectbox(
                        "Mode de Déploiement",
                        ["Production Complète", "Test Pilote", "Démonstration", "Maintenance"]
                    )
                    
                    deployment_date = st.date_input(
                        "Date de Déploiement",
                        value=datetime.now()
                    )
                
                st.markdown("---")
                st.write("### ⚙️ Configuration Opérationnelle")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    work_schedule = st.selectbox(
                        "Horaire de Travail",
                        ["24/7", "Jour (8h-18h)", "Nuit (18h-8h)", "Personnalisé"]
                    )
                    
                    auto_recharge = st.checkbox("Recharge Automatique", value=True)
                
                with col2:
                    maintenance_plan = st.selectbox(
                        "Plan de Maintenance",
                        ["Hebdomadaire", "Bi-mensuel", "Mensuel", "Sur Demande"]
                    )
                    
                    remote_monitoring = st.checkbox("Monitoring à Distance", value=True)
                
                st.markdown("---")
                st.write("### 📋 Checklist Pré-Déploiement")
                
                checklist_items = [
                    "Robot testé et calibré",
                    "Site préparé et sécurisé",
                    "Personnel formé",
                    "Système de monitoring configuré",
                    "Plan d'urgence établi",
                    "Documentation complète"
                ]
                
                all_checked = True
                for item in checklist_items:
                    if not st.checkbox(item, key=f"checklist_{item}"):
                        all_checked = False
                
                submitted = st.form_submit_button("🚀 Déployer", type="primary", disabled=not all_checked)
                
                if submitted:
                    robot = st.session_state.robotics_system['robots'][selected_robot]
                    site = next(s for s in st.session_state.deployment_sites if s['site_id'] == selected_site)
                    
                    if site['robots_deployed'] >= site['capacity']:
                        st.error(f"❌ Capacité du site atteinte ({site['capacity']} robots)")
                    else:
                        deployment = {
                            'deployment_id': f"DEP_{len(st.session_state.robotics_system.get('deployments', {})) + 1:04d}",
                            'robot_id': selected_robot,
                            'robot_name': robot['name'],
                            'site_id': selected_site,
                            'site_name': site['name'],
                            'mode': deployment_mode,
                            'schedule': work_schedule,
                            'deployment_date': deployment_date.isoformat(),
                            'status': 'Actif',
                            'uptime': 0.0,
                            'missions_completed': 0
                        }
                        
                        if 'deployments' not in st.session_state.robotics_system:
                            st.session_state.robotics_system['deployments'] = {}
                        
                        st.session_state.robotics_system['deployments'][deployment['deployment_id']] = deployment
                        
                        # Mettre à jour le site
                        site['robots_deployed'] += 1
                        
                        # Mettre à jour le robot
                        robot['status'] = 'online'
                        
                        st.success(f"✅ Robot '{robot['name']}' déployé sur {site['name']}!")
                        st.balloons()
                        
                        log_event(f"Déploiement: {robot['name']} -> {site['name']}")
    
    with tab3:
        st.subheader("🌐 Gestion de Flotte")
        
        if 'deployments' not in st.session_state.robotics_system or not st.session_state.robotics_system['deployments']:
            st.info("Aucun robot déployé")
        else:
            st.write(f"### 🤖 {len(st.session_state.robotics_system['deployments'])} Robots Déployés")
            
            # Vue d'ensemble
            deployments_data = []
            for dep_id, dep in st.session_state.robotics_system['deployments'].items():
                deployments_data.append({
                    'ID': dep['deployment_id'],
                    'Robot': dep['robot_name'][:20],
                    'Site': dep['site_name'][:20],
                    'Mode': dep['mode'],
                    'Statut': dep['status'],
                    'Uptime': f"{dep['uptime']:.1f}h",
                    'Missions': dep['missions_completed'],
                    'Date': dep['deployment_date'][:10]
                })
            
            df = pd.DataFrame(deployments_data)
            st.dataframe(df, use_container_width=True)
            
            st.markdown("---")
            
            # Contrôle de flotte
            st.write("### 🎮 Contrôle de Flotte")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("▶️ Démarrer Tous", use_container_width=True):
                    for dep in st.session_state.robotics_system['deployments'].values():
                        dep['status'] = 'Actif'
                    st.success("✅ Tous les robots démarrés")
                    st.rerun()
            
            with col2:
                if st.button("⏸️ Pause Tous", use_container_width=True):
                    for dep in st.session_state.robotics_system['deployments'].values():
                        dep['status'] = 'Pause'
                    st.warning("⏸️ Tous les robots en pause")
                    st.rerun()
            
            with col3:
                if st.button("🛑 Arrêt d'Urgence", use_container_width=True, type="primary"):
                    for dep in st.session_state.robotics_system['deployments'].values():
                        dep['status'] = 'Arrêt Urgence'
                    st.error("🛑 ARRÊT D'URGENCE ACTIVÉ")
                    st.rerun()
            
            st.markdown("---")
            
            # Détails par déploiement
            st.write("### 📊 Détails des Déploiements")
            
            for dep_id, dep in st.session_state.robotics_system['deployments'].items():
                with st.expander(f"🤖 {dep['robot_name']} @ {dep['site_name']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Statut", dep['status'])
                        st.metric("Mode", dep['mode'])
                    
                    with col2:
                        st.metric("Uptime", f"{dep['uptime']:.1f}h")
                        st.metric("Missions", dep['missions_completed'])
                    
                    with col3:
                        robot = st.session_state.robotics_system['robots'][dep['robot_id']]
                        st.metric("Santé", f"{robot['health']:.0%}")
                        st.metric("Batterie", f"{robot['power']['charge']:.0f}%")
                    
                    # Actions
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📊 Statistiques", key=f"stats_{dep_id}"):
                            st.info("Statistiques détaillées")
                    
                    with col2:
                        if st.button("🔧 Maintenance", key=f"maint_{dep_id}"):
                            st.info("Mode maintenance activé")
                    
                    with col3:
                        if st.button("🔙 Rappeler", key=f"recall_{dep_id}"):
                            dep['status'] = 'Rappelé'
                            site = next(s for s in st.session_state.deployment_sites if s['site_id'] == dep['site_id'])
                            site['robots_deployed'] -= 1
                            st.success("Robot rappelé")
    
    with tab4:
        st.subheader("📡 Monitoring en Temps Réel")
        
        if 'deployments' not in st.session_state.robotics_system or not st.session_state.robotics_system['deployments']:
            st.info("Aucun robot à monitorer")
        else:
            # Métriques globales
            st.write("### 📊 Métriques Globales de Flotte")
            
            active_robots = sum(1 for d in st.session_state.robotics_system['deployments'].values() if d['status'] == 'Actif')
            total_deployed = len(st.session_state.robotics_system['deployments'])
            total_missions = sum(d['missions_completed'] for d in st.session_state.robotics_system['deployments'].values())
            avg_uptime = np.mean([d['uptime'] for d in st.session_state.robotics_system['deployments'].values()])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Robots Actifs", f"{active_robots}/{total_deployed}")
            with col2:
                st.metric("Missions Totales", total_missions)
            with col3:
                st.metric("Uptime Moyen", f"{avg_uptime:.1f}h")
            with col4:
                availability = active_robots / total_deployed * 100 if total_deployed > 0 else 0
                st.metric("Disponibilité", f"{availability:.0f}%")
            
            st.markdown("---")
            
            # Carte de statut
            st.write("### 🗺️ Carte de Statut")
            
            status_counts = {}
            for dep in st.session_state.robotics_system['deployments'].values():
                status = dep['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            fig = go.Figure(data=[
                go.Pie(labels=list(status_counts.keys()), 
                      values=list(status_counts.values()),
                      hole=0.3)
            ])
            
            fig.update_layout(title="Répartition des Statuts", height=400)
            st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{dep}")
            
            st.markdown("---")
            
            # Alertes
            st.write("### ⚠️ Alertes et Notifications")
            
            alerts = []
            for dep in st.session_state.robotics_system['deployments'].values():
                robot = st.session_state.robotics_system['robots'][dep['robot_id']]
                
                if robot['power']['charge'] < 20:
                    alerts.append(f"🔋 {dep['robot_name']}: Batterie faible ({robot['power']['charge']:.0f}%)")
                
                if robot['health'] < 0.8:
                    alerts.append(f"⚠️ {dep['robot_name']}: Santé dégradée ({robot['health']:.0%})")
                
                if dep['status'] == 'Arrêt Urgence':
                    alerts.append(f"🛑 {dep['robot_name']}: Arrêt d'urgence actif")
            
            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ Aucune alerte active")

# ==================== PAGE: PROJETS ====================

elif page == "📁 Projets":
    st.header("📁 Gestion de Projets Robotiques")
    
    tab1, tab2, tab3 = st.tabs(["➕ Nouveau Projet", "📋 Mes Projets", "📊 Suivi"])
    
    with tab1:
        st.subheader("➕ Créer Nouveau Projet")
        
        with st.form("new_project"):
            st.write("### 📝 Informations Générales")
            
            col1, col2 = st.columns(2)
            
            with col1:
                project_name = st.text_input("Nom du Projet", placeholder="Ex: Automatisation Entrepôt 2024")
                project_description = st.text_area("Description", placeholder="Objectifs et portée du projet...")
            
            with col2:
                project_status = st.selectbox("Statut Initial", ["Planification", "En Cours", "En Pause"])
                project_budget = st.number_input("Budget ($)", 0, 10000000, 100000, 10000)
            
            st.markdown("---")
            st.write("### 🤖 Robots du Projet")
            
            if st.session_state.robotics_system['robots']:
                robot_ids = list(st.session_state.robotics_system['robots'].keys())
                selected_robots = st.multiselect(
                    "Sélectionner Robots",
                    robot_ids,
                    format_func=lambda x: st.session_state.robotics_system['robots'][x]['name']
                )
            else:
                st.info("Créez des robots pour les ajouter au projet")
                selected_robots = []
            
            st.markdown("---")
            st.write("### 👥 Équipe")
            
            col1, col2 = st.columns(2)
            
            with col1:
                team_size = st.number_input("Taille de l'Équipe", 1, 100, 5)
                project_manager = st.text_input("Chef de Projet", placeholder="Nom du responsable")
            
            with col2:
                start_date = st.date_input("Date de Début", value=datetime.now())
                end_date = st.date_input("Date de Fin Prévue", value=datetime.now() + timedelta(days=180))
            
            st.markdown("---")
            st.write("### 🎯 Jalons du Projet")
            
            n_milestones = st.number_input("Nombre de Jalons", 1, 20, 3)
            
            milestones = []
            for i in range(n_milestones):
                col1, col2 = st.columns(2)
                with col1:
                    milestone_name = st.text_input(f"Jalon {i+1}", key=f"milestone_name_{i}", placeholder="Ex: Prototype fonctionnel")
                with col2:
                    milestone_date = st.date_input(f"Date Jalon {i+1}", key=f"milestone_date_{i}")
                
                if milestone_name:
                    milestones.append({
                        'name': milestone_name,
                        'date': milestone_date.isoformat(),
                        'completed': False
                    })
            
            submitted = st.form_submit_button("📁 Créer Projet", type="primary")
            
            if submitted:
                if not project_name:
                    st.error("Le nom du projet est requis")
                else:
                    project_id = f"PROJ_{len(st.session_state.robotics_system['projects']) + 1:04d}"
                    
                    project = {
                        'project_id': project_id,
                        'name': project_name,
                        'description': project_description,
                        'status': project_status,
                        'start_date': start_date.isoformat(),
                        'end_date': end_date.isoformat(),
                        'progress': 0.0,
                        'budget': project_budget,
                        'team_size': team_size,
                        'project_manager': project_manager,
                        'robots': selected_robots,
                        'milestones': milestones,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.robotics_system['projects'][project_id] = project
                    
                    st.success(f"✅ Projet '{project_name}' créé avec succès!")
                    st.balloons()
                    
                    log_event(f"Nouveau projet créé: {project_name}")
                    
    with tab2:
        st.subheader("📋 Projets en Cours")
        
        if not st.session_state.robotics_system['projects']:
            st.info("Aucun projet créé")
        else:
            for proj_id, project in st.session_state.robotics_system['projects'].items():
                with st.expander(f"📁 {project['name']} ({proj_id})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {project['description']}")
                        st.write(f"**Statut:** {project['status']}")
                        st.write(f"**Date Début:** {project['start_date'][:10]}")
                        
                        st.progress(project['progress'] / 100)
                        st.write(f"Progression: {project['progress']:.0f}%")
                    
                    with col2:
                        st.metric("Budget", f"${project['budget']:,.0f}")
                        st.metric("Équipe", f"{project['team_size']} personnes")
                        st.metric("Robots", len(project['robots']))
                    
                    # Milestones
                    if project.get('milestones'):
                        st.write("**🎯 Jalons:**")
                        for milestone in project['milestones']:
                            status_icon = "✅" if milestone.get('completed', False) else "⏳"
                            st.write(f"{status_icon} {milestone['name']} - {milestone['date']}")
    
    
    with tab3:
        st.subheader("📊 Suivi des Projets")
        
        if st.session_state.robotics_system['projects']:
            # Vue d'ensemble
            st.write("### 📈 Vue d'Ensemble")
            
            total_projects = len(st.session_state.robotics_system['projects'])
            active_projects = sum(1 for p in st.session_state.robotics_system['projects'].values() if p['status'] == 'En Cours')
            total_budget = sum(p['budget'] for p in st.session_state.robotics_system['projects'].values())
            avg_progress = np.mean([p['progress'] for p in st.session_state.robotics_system['projects'].values()])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Projets Totaux", total_projects)
            with col2:
                st.metric("Projets Actifs", active_projects)
            with col3:
                st.metric("Budget Total", f"${total_budget:,.0f}")
            with col4:
                st.metric("Progression Moy.", f"{avg_progress:.0f}%")
            
            st.markdown("---")
            
            # Graphique progression
            st.write("### 📊 Progression des Projets")
            
            project_names = [p['name'][:20] for p in st.session_state.robotics_system['projects'].values()]
            project_progress = [p['progress'] for p in st.session_state.robotics_system['projects'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=project_names, y=project_progress,
                      marker_color='lightblue',
                      text=[f"{p:.0f}%" for p in project_progress],
                      textposition='outside')
            ])
            
            fig.update_layout(
                title="Progression par Projet",
                yaxis_title="Progression (%)",
                xaxis_tickangle=-45,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Timeline
            st.write("### 📅 Timeline des Projets")
            
            fig = go.Figure()
            
            for i, (proj_id, project) in enumerate(st.session_state.robotics_system['projects'].items()):
                fig.add_trace(go.Scatter(
                    x=[project['start_date'], project['end_date']],
                    y=[project['name'], project['name']],
                    mode='lines+markers',
                    name=project['name'],
                    line=dict(width=10),
                    marker=dict(size=10)
                ))
            
            fig.update_layout(
                title="Timeline des Projets",
                xaxis_title="Date",
                yaxis_title="Projet",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"robot_plot_{i}")
        else:
            st.info("Aucun projet à suivre")

# ==================== PAGE: FORMATION & IA ====================

elif page == "🎓 Formation & IA":
    st.header("🎓 Formation et Apprentissage IA")
    
    tab1, tab2, tab3 = st.tabs(["📚 Ressources", "🎯 Tutoriels", "🏆 Certifications"])
    
    with tab1:
        st.subheader("📚 Bibliothèque de Ressources")
        
        resources = {
            "Débutant": [
                {"title": "Introduction à la Robotique", "type": "Cours", "duration": "4h", "rating": 4.8},
                {"title": "Premiers Pas avec ROS", "type": "Tutoriel", "duration": "2h", "rating": 4.6},
                {"title": "Programmation Python pour Robots", "type": "Cours", "duration": "6h", "rating": 4.9},
            ],
            "Intermédiaire": [
                {"title": "Navigation Autonome", "type": "Cours", "duration": "8h", "rating": 4.7},
                {"title": "Vision par Ordinateur Appliquée", "type": "Tutoriel", "duration": "5h", "rating": 4.8},
                {"title": "Contrôle Avancé PID", "type": "Cours", "duration": "3h", "rating": 4.5},
            ],
            "Avancé": [
                {"title": "Deep Reinforcement Learning", "type": "Cours", "duration": "12h", "rating": 4.9},
                {"title": "SLAM et Cartographie", "type": "Tutoriel", "duration": "7h", "rating": 4.7},
                {"title": "Robotique Quantique", "type": "Cours", "duration": "10h", "rating": 4.8},
            ]
        }
        
        for level, items in resources.items():
            st.write(f"### {level}")
            
            for resource in items:
                with st.expander(f"📖 {resource['title']} ({resource['type']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Type:** {resource['type']}")
                        st.write(f"**Durée:** {resource['duration']}")
                        st.write(f"**Note:** {'⭐' * int(resource['rating'])} ({resource['rating']}/5)")
                    
                    with col2:
                        if st.button("▶️ Commencer", key=f"start_{resource['title']}"):
                            st.success("Cours démarré!")
                        
                        if st.button("💾 Sauvegarder", key=f"save_{resource['title']}"):
                            st.info("Ajouté aux favoris")
    
    with tab2:
        st.subheader("🎯 Tutoriels Interactifs")
        
        tutorials = [
            {
                "title": "Créer Votre Premier Robot Mobile",
                "description": "Apprenez à concevoir et programmer un robot mobile autonome",
                "steps": 8,
                "difficulty": "Débutant",
                "time": "3h"
            },
            {
                "title": "Intégration de l'IA dans un Robot",
                "description": "Ajoutez des capacités d'intelligence artificielle à votre robot",
                "steps": 12,
                "difficulty": "Intermédiaire",
                "time": "5h"
            },
            {
                "title": "Robotique Collaborative Multi-Agents",
                "description": "Coordination de plusieurs robots travaillant ensemble",
                "steps": 15,
                "difficulty": "Avancé",
                "time": "8h"
            }
        ]
        
        for tutorial in tutorials:
            with st.expander(f"🎓 {tutorial['title']}"):
                st.write(f"**Description:** {tutorial['description']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Étapes", tutorial['steps'])
                with col2:
                    st.metric("Difficulté", tutorial['difficulty'])
                with col3:
                    st.metric("Durée", tutorial['time'])
                
                progress = st.slider(f"Votre progression", 0, 100, 0, key=f"tuto_{tutorial['title']}")
                
                if st.button(f"▶️ Continuer le Tutoriel", key=f"continue_{tutorial['title']}"):
                    st.info(f"Reprise à l'étape {int(progress / 100 * tutorial['steps'])}/{tutorial['steps']}")
    
    with tab3:
        st.subheader("🏆 Certifications")
        
        certifications = [
            {
                "name": "Certified Robot Developer",
                "level": "Foundation",
                "requirements": ["5 robots créés", "10 simulations", "Score > 80%"],
                "status": "Disponible"
            },
            {
                "name": "AI Robotics Specialist",
                "level": "Professional",
                "requirements": ["3 robots avec IA", "ML training complété", "Projet déployé"],
                "status": "En Cours"
            },
            {
                "name": "Quantum Robotics Expert",
                "level": "Expert",
                "requirements": ["Robot quantique", "Publication recherche", "Examen final"],
                "status": "Verrouillé"
            }
        ]
        
        for cert in certifications:
            status_colors = {"Disponible": "🟢", "En Cours": "🟡", "Verrouillé": "🔴"}
            
            with st.expander(f"{status_colors[cert['status']]} {cert['name']} ({cert['level']})"):
                st.write(f"**Niveau:** {cert['level']}")
                # Fin du code des pages manquantes
                st.info(f"**Statut:** {cert['status']}")
                
                st.write("\n**Prérequis:**")
                for req in cert['requirements']:
                    st.write(f"• {req}")
                
                if cert['status'] == "Disponible":
                    if st.button(f"🎯 Passer l'Examen", key=f"exam_{cert['name']}"):
                        st.success("Inscription à l'examen réussie!")
                elif cert['status'] == "En Cours":
                    progress = np.random.randint(30, 70)
                    st.progress(progress / 100)
                    st.write(f"Progression: {progress}%")

# ==================== PAGE: SIMULATIONS ====================

elif page == "🔬 Simulations":
    st.header("🔬 Environnement de Simulation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌍 Environnements", "▶️ Lancer Simulation", "📊 Résultats", "🎥 Visualisation"])
    
    with tab1:
        st.subheader("🌍 Environnements de Simulation")
        
        environments = {
            "Intérieur - Bureau": {
                "description": "Bureau standard avec meubles et obstacles",
                "size": "10x10x3 m",
                "obstacles": ["Tables", "Chaises", "Murs"],
                "complexity": "Moyenne",
                "physics": "Standard"
            },
            "Industriel - Usine": {
                "description": "Environnement industriel avec machines",
                "size": "50x30x10 m",
                "obstacles": ["Machines", "Convoyeurs", "Robots"],
                "complexity": "Élevée",
                "physics": "Avancée + Collisions"
            },
            "Extérieur - Urbain": {
                "description": "Rue urbaine avec trottoirs et obstacles",
                "size": "100x100x20 m",
                "obstacles": ["Bâtiments", "Véhicules", "Piétons"],
                "complexity": "Très élevée",
                "physics": "Standard + Météo"
            },
            "Terrain Accidenté": {
                "description": "Terrain naturel avec dénivelés",
                "size": "50x50x20 m",
                "obstacles": ["Rochers", "Pentes", "Végétation"],
                "complexity": "Élevée",
                "physics": "Déformable"
            },
            "Sous-marin": {
                "description": "Environnement aquatique",
                "size": "20x20x10 m",
                "obstacles": ["Récifs", "Courants"],
                "complexity": "Moyenne",
                "physics": "Fluides"
            },
            "Spatial": {
                "description": "Orbite terrestre basse",
                "size": "Illimité",
                "obstacles": ["Débris", "Satellites"],
                "complexity": "Très élevée",
                "physics": "Microgravité"
            }
        }
        
        for env_name, env_info in environments.items():
            with st.expander(f"🌍 {env_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {env_info['description']}")
                    st.write(f"**Taille:** {env_info['size']}")
                    st.write(f"**Complexité:** {env_info['complexity']}")
                
                with col2:
                    st.write("**Obstacles:**")
                    for obs in env_info['obstacles']:
                        st.write(f"• {obs}")
                    st.write(f"**Physique:** {env_info['physics']}")
                
                if st.button(f"✅ Sélectionner", key=f"env_{env_name}"):
                    st.session_state.selected_env = env_name
                    st.success(f"Environnement '{env_name}' sélectionné")
    
    with tab2:
        st.subheader("▶️ Configurer et Lancer Simulation")
        
        if not st.session_state.robotics_system['robots']:
            st.warning("Aucun robot disponible pour la simulation")
        else:
            with st.form("simulation_config"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### 🤖 Robot")
                    robot_ids = list(st.session_state.robotics_system['robots'].keys())
                    selected_robot = st.selectbox(
                        "Sélectionner Robot",
                        robot_ids,
                        format_func=lambda x: st.session_state.robotics_system['robots'][x]['name']
                    )
                    
                    st.write("### 🌍 Environnement")
                    env_selected = st.selectbox(
                        "Environnement",
                        list(environments.keys())
                    )
                    
                    st.write("### ⏱️ Durée")
                    duration = st.number_input("Durée (secondes)", 1, 3600, 60)
                    timestep = st.number_input("Pas de temps (ms)", 1, 100, 10)
                
                with col2:
                    st.write("### 🎯 Mission")
                    mission_type = st.selectbox(
                        "Type de Mission",
                        ["Navigation Point à Point", "Exploration", "Manipulation",
                         "Suivi de Trajectoire", "Évitement d'Obstacles", "Libre"]
                    )
                    
                    if mission_type == "Navigation Point à Point":
                        start_x = st.number_input("Départ X", -50.0, 50.0, 0.0)
                        start_y = st.number_input("Départ Y", -50.0, 50.0, 0.0)
                        goal_x = st.number_input("Objectif X", -50.0, 50.0, 10.0)
                        goal_y = st.number_input("Objectif Y", -50.0, 50.0, 10.0)
                    
                    st.write("### ⚙️ Paramètres Physiques")
                    gravity = st.checkbox("Gravité", value=True)
                    friction = st.slider("Friction", 0.0, 1.0, 0.5)
                    wind = st.checkbox("Vent/Perturbations", value=False)
                
                submit_sim = st.form_submit_button("🚀 Lancer Simulation", type="primary")
                
                if submit_sim:
                    robot = st.session_state.robotics_system['robots'][selected_robot]
                    
                    with st.spinner("🔄 Simulation en cours..."):
                        progress_bar = st.progress(0)
                        
                        # Simulation
                        n_steps = int(duration / (timestep / 1000))
                        
                        sim_result = {
                            'sim_id': f"sim_{len(st.session_state.robotics_system['simulations']) + 1}",
                            'robot_id': selected_robot,
                            'robot_name': robot['name'],
                            'environment': env_selected,
                            'mission': mission_type,
                            'duration': duration,
                            'timestamp': datetime.now().isoformat(),
                            'trajectory': [],
                            'velocities': [],
                            'energy': [],
                            'collisions': 0,
                            'success': False,
                            'completion': 0.0
                        }
                        
                        # Génération trajectoire
                        for step in range(min(n_steps, 1000)):
                            progress_bar.progress(step / min(n_steps, 1000))
                            
                            if mission_type == "Navigation Point à Point":
                                t = step / min(n_steps, 1000)
                                x = start_x + (goal_x - start_x) * t
                                y = start_y + (goal_y - start_y) * t
                                z = 0.5
                            else:
                                x = 10 * np.sin(step * 0.01)
                                y = 10 * np.cos(step * 0.01)
                                z = 0.5
                            
                            sim_result['trajectory'].append([x, y, z])
                            
                            if step > 0:
                                prev = sim_result['trajectory'][-2]
                                vel = np.sqrt((x-prev[0])**2 + (y-prev[1])**2 + (z-prev[2])**2)
                                sim_result['velocities'].append(vel)
                                sim_result['energy'].append(vel * robot['power']['consumption'] * 0.001)
                        
                        progress_bar.empty()
                        
                        # Résultats
                        sim_result['success'] = True
                        sim_result['completion'] = 100.0
                        sim_result['collisions'] = np.random.randint(0, 3)
                        
                        st.session_state.robotics_system['simulations'].append(sim_result)
                        
                        st.success("✅ Simulation terminée avec succès!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Durée", f"{duration}s")
                        with col2:
                            st.metric("Collisions", sim_result['collisions'])
                        with col3:
                            total_energy = sum(sim_result['energy'])
                            st.metric("Énergie", f"{total_energy:.2f} Wh")
                        with col4:
                            st.metric("Complétion", "100%")
                        
                        log_event(f"Simulation terminée: {robot['name']} dans {env_selected}")
    
    with tab3:
        st.subheader("📊 Résultats des Simulations")
        
        if not st.session_state.robotics_system['simulations']:
            st.info("Aucune simulation effectuée")
        else:
            st.write(f"### 📋 {len(st.session_state.robotics_system['simulations'])} Simulations Effectuées")
            
            # Tableau récapitulatif
            sim_df = pd.DataFrame([
                {
                    'ID': sim['sim_id'],
                    'Robot': sim['robot_name'][:20],
                    'Environnement': sim['environment'][:20],
                    'Mission': sim['mission'][:20],
                    'Durée (s)': sim['duration'],
                    'Succès': '✅' if sim['success'] else '❌',
                    'Collisions': sim['collisions'],
                    'Date': sim['timestamp'][:10]
                }
                for sim in st.session_state.robotics_system['simulations']
            ])
            
            st.dataframe(sim_df, use_container_width=True)
            
            st.markdown("---")
            
            # Sélection simulation pour détails
            sim_ids = [s['sim_id'] for s in st.session_state.robotics_system['simulations']]
            selected_sim_id = st.selectbox("Voir détails de la simulation", sim_ids)
            
            sim = next(s for s in st.session_state.robotics_system['simulations'] if s['sim_id'] == selected_sim_id)
            
            st.write(f"### 📊 {sim['sim_id']} - {sim['robot_name']}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Environnement", sim['environment'])
            with col2:
                st.metric("Mission", sim['mission'])
            with col3:
                st.metric("Durée", f"{sim['duration']}s")
            with col4:
                st.metric("Collisions", sim['collisions'])
            
            # Graphiques
            if sim['trajectory']:
                trajectory = np.array(sim['trajectory'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trajectoire 2D
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=trajectory[:, 0],
                        y=trajectory[:, 1],
                        mode='lines+markers',
                        name='Trajectoire',
                        line=dict(color='blue', width=2),
                        marker=dict(size=3)
                    ))
                    
                    fig.update_layout(
                        title="Trajectoire (Vue du Dessus)",
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Vitesse
                    if sim['velocities']:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=sim['velocities'],
                            mode='lines',
                            name='Vitesse',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Vitesse au Cours du Temps",
                            xaxis_title="Pas de temps",
                            yaxis_title="Vitesse (m/s)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🎥 Visualisation 3D")
        
        if st.session_state.robotics_system['simulations']:
            sim_ids = [s['sim_id'] for s in st.session_state.robotics_system['simulations']]
            selected_sim_viz = st.selectbox("Sélectionner simulation à visualiser", sim_ids, key="viz_sim")
            
            sim = next(s for s in st.session_state.robotics_system['simulations'] if s['sim_id'] == selected_sim_viz)
            
            if sim['trajectory']:
                trajectory = np.array(sim['trajectory'])
                
                # Visualisation 3D
                fig = go.Figure(data=[go.Scatter3d(
                    x=trajectory[:, 0],
                    y=trajectory[:, 1],
                    z=trajectory[:, 2],
                    mode='lines+markers',
                    marker=dict(
                        size=3,
                        color=np.arange(len(trajectory)),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Temps")
                    ),
                    line=dict(color='blue', width=3)
                )])
                
                fig.update_layout(
                    title="Trajectoire 3D du Robot",
                    scene=dict(
                        xaxis_title="X (m)",
                        yaxis_title="Y (m)",
                        zaxis_title="Z (m)",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune simulation à visualiser")

# ==================== PAGE: EXPÉRIENCES & TESTS ====================

elif page == "🧪 Expériences & Tests":
    st.header("🧪 Expériences et Tests")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Protocoles", "🧪 Tests Unitaires", "📊 Résultats", "📈 Benchmarks"])
    
    with tab1:
        st.subheader("🔬 Protocoles d'Expérimentation")
        
        test_protocols = {
            "Test de Charge": {
                "objective": "Vérifier capacité de charge du robot",
                "steps": [
                    "Placer poids de 0 à max sur robot",
                    "Mesurer vitesse et précision",
                    "Enregistrer consommation énergie",
                    "Vérifier stabilité"
                ],
                "metrics": ["Vitesse", "Précision", "Stabilité", "Énergie"],
                "duration": "2 heures"
            },
            "Test d'Endurance": {
                "objective": "Évaluer fonctionnement prolongé",
                "steps": [
                    "Cycle de travail répétitif",
                    "Monitoring température",
                    "Surveillance usure",
                    "Test jusqu'à 24h"
                ],
                "metrics": ["Température", "Usure", "Défaillances", "Performance"],
                "duration": "24 heures"
            },
            "Test de Précision": {
                "objective": "Mesurer précision positionnelle",
                "steps": [
                    "Positionnement sur grille",
                    "Répétition 100 fois",
                    "Mesure écarts",
                    "Calcul statistiques"
                ],
                "metrics": ["Précision", "Répétabilité", "Écart-type"],
                "duration": "1 heure"
            },
            "Test Environnemental": {
                "objective": "Vérifier résistance conditions extrêmes",
                "steps": [
                    "Test température (-20°C à 60°C)",
                    "Test humidité (0-100%)",
                    "Test poussière/eau (IP rating)",
                    "Test vibrations"
                ],
                "metrics": ["Fonctionnement", "Étanchéité", "Résistance"],
                "duration": "8 heures"
            }
        }
        
        for protocol_name, protocol_info in test_protocols.items():
            with st.expander(f"🔬 {protocol_name}"):
                st.write(f"**Objectif:** {protocol_info['objective']}")
                
                st.write("\n**Étapes:**")
                for i, step in enumerate(protocol_info['steps'], 1):
                    st.write(f"{i}. {step}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Métriques:**")
                    for metric in protocol_info['metrics']:
                        st.write(f"• {metric}")
                
                with col2:
                    st.metric("Durée", protocol_info['duration'])
                    if st.button(f"🚀 Lancer Test", key=f"launch_{protocol_name}"):
                        st.session_state.current_test = protocol_name
                        st.success(f"Test '{protocol_name}' démarré")

    with tab2:
        st.subheader("📈 Réglage de Contrôleur PID")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### ⚙️ Paramètres PID")
            kp = st.slider("Kp (Proportionnel)", 0.0, 10.0, 1.0, 0.1)
            ki = st.slider("Ki (Intégral)", 0.0, 5.0, 0.1, 0.01)
            kd = st.slider("Kd (Dérivé)", 0.0, 2.0, 0.01, 0.001)
            
            setpoint = st.number_input("Consigne", 0.0, 100.0, 50.0, 1.0)
            
            st.write("### 🎯 Critères de Performance")
            st.write(f"**Temps de montée:** ~{2/(kp+0.1):.2f}s")
            st.write(f"**Dépassement:** ~{100*kd/(kp+0.1):.1f}%")
            st.write(f"**Erreur statique:** ~{1/(ki+0.01):.2f}%")
        
        with col2:
            st.write("### 📊 Réponse du Système")
            
            # Simulation PID
            t = np.linspace(0, 10, 500)
            
            # Réponse simplifiée
            wn = np.sqrt(kp)
            zeta = kd / (2 * np.sqrt(kp)) if kp > 0 else 0
            
            if zeta < 1 and kp > 0:
                wd = wn * np.sqrt(1 - zeta**2)
                response = setpoint * (1 - np.exp(-zeta * wn * t) * 
                          (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t)))
            else:
                response = setpoint * (1 - np.exp(-kp * t))
            
            # Effet intégral
            response = response + ki * (setpoint - response) * 0.1
            
            fig = go.Figure()
            
            # Consigne
            fig.add_trace(go.Scatter(
                x=t, y=[setpoint]*len(t),
                mode='lines',
                name='Consigne',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            # Réponse
            fig.add_trace(go.Scatter(
                x=t, y=response,
                mode='lines',
                name='Réponse',
                line=dict(color='blue', width=3)
            ))
            
            fig.update_layout(
                title="Réponse Indicielle",
                xaxis_title="Temps (s)",
                yaxis_title="Sortie",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Méthode de réglage
            st.write("### 🔧 Méthodes de Réglage Suggérées")
            
            method = st.selectbox(
                "Méthode",
                ["Ziegler-Nichols", "Cohen-Coon", "Manual Tuning", "Auto-Tune"]
            )
            
            if st.button("🎯 Appliquer Méthode"):
                st.success(f"✅ Paramètres calculés selon {method}")
                if method == "Ziegler-Nichols":
                    st.write("Kp = 1.2, Ki = 0.15, Kd = 0.05")
    
    with tab3:
        st.subheader("🤖 Commande de Robot")
        
        if not st.session_state.robotics_system['robots']:
            st.info("Aucun robot disponible")
        else:
            robot_ids = list(st.session_state.robotics_system['robots'].keys())
            selected = st.selectbox(
                "Sélectionner Robot",
                robot_ids,
                format_func=lambda x: st.session_state.robotics_system['robots'][x]['name']
            )
            
            robot = st.session_state.robotics_system['robots'][selected]
            
            st.write(f"### 🤖 {robot['name']}")
            
            # Interface de commande
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### 🎮 Commande Manuelle")
                
                control_mode = st.radio(
                    "Mode de Contrôle",
                    ["Position", "Vitesse", "Couple/Force"]
                )
                
                n_joints = robot['specifications']['dof']
                
                st.write(f"**Articulations ({n_joints} DoF):**")
                
                joint_commands = []
                for i in range(min(n_joints, 6)):
                    val = st.slider(
                        f"Joint {i+1}",
                        -180.0, 180.0, 0.0, 1.0,
                        key=f"joint_{selected}_{i}"
                    )
                    joint_commands.append(val)
                
                if st.button("▶️ Envoyer Commande", type="primary"):
                    st.success("✅ Commande envoyée au robot")
                    log_event(f"Commande envoyée à {robot['name']}")
            
            with col2:
                st.write("#### 📊 État Actuel")
                
                # État simulé
                for i in range(min(n_joints, 6)):
                    current_pos = np.random.uniform(-90, 90)
                    st.metric(
                        f"Joint {i+1}",
                        f"{current_pos:.1f}°",
                        delta=f"{np.random.uniform(-5, 5):.1f}°"
                    )
                
                st.write("#### 🔋 Système")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Température", f"{25 + np.random.random()*10:.1f}°C")
                with col_b:
                    st.metric("Courant", f"{np.random.uniform(1, 5):.2f} A")
    
    with tab4:
        st.subheader("📊 Monitoring en Temps Réel")
        
        st.write("### 📈 Signaux de Contrôle")
        
        # Génération de données temps réel simulées
        if 'control_data' not in st.session_state:
            st.session_state.control_data = {
                'time': [],
                'setpoint': [],
                'output': [],
                'error': []
            }
        
        # Simulation temps réel
        t_current = len(st.session_state.control_data['time'])
        setpoint_val = 50 + 20 * np.sin(t_current * 0.1)
        output_val = setpoint_val + np.random.randn() * 2
        error_val = setpoint_val - output_val
        
        st.session_state.control_data['time'].append(t_current)
        st.session_state.control_data['setpoint'].append(setpoint_val)
        st.session_state.control_data['output'].append(output_val)
        st.session_state.control_data['error'].append(error_val)
        
        # Limiter historique
        if len(st.session_state.control_data['time']) > 100:
            for key in st.session_state.control_data:
                st.session_state.control_data[key] = st.session_state.control_data[key][-100:]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Consigne vs Sortie", "Erreur")
        )
        
        # Graphique 1
        fig.add_trace(
            go.Scatter(
                x=st.session_state.control_data['time'],
                y=st.session_state.control_data['setpoint'],
                name='Consigne',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=st.session_state.control_data['time'],
                y=st.session_state.control_data['output'],
                name='Sortie',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Graphique 2
        fig.add_trace(
            go.Scatter(
                x=st.session_state.control_data['time'],
                y=st.session_state.control_data['error'],
                name='Erreur',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("🔄 Rafraîchir"):
            st.rerun()
                
# ==================== FOOTER ====================

st.markdown("---")

with st.expander("📜 Journal des Événements (Dernières 10 entrées)"):
    if st.session_state.robotics_system['log']:
        for event in st.session_state.robotics_system['log'][-10:][::-1]:
            timestamp = event['timestamp'][:19]
            st.text(f"{timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")
    
    if st.button("🗑️ Effacer le Journal", key="clear_log_main"):
        st.session_state.robotics_system['log'] = []
        st.rerun()
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🤖 Plateforme Robotique Complète - IA Quantique Biologique</h3>
        <p>Système Intégré de Création, Développement et Déploiement</p>
        <p><small>Version 1.0.0 | Tous Domaines de la Robotique</small></p>
        <p><small>🦾 Humanoïdes | 🏭 Industriels | 🚁 Aériens | 🌊 Aquatiques | ⚕️ Médicaux</small></p>
        <p><small>🧠 IA Avancée | ⚛️ Quantique | 🧬 Biologique | 🔧 Fabrication</small></p>
        <p><small>Powered by Advanced Robotics & AI © 2024</small></p>
    </div>
""", unsafe_allow_html=True)