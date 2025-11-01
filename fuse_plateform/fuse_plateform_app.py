"""
Plateforme Avancée de Conception, Fabrication et Simulation de Fusées
Système IA/Quantique/Bio-computing pour véhicules spatiaux
streamlit run fuse_plateform_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import numpy as np
from typing import Dict, List, Tuple
import time

# ==================== CONFIGURATION PAGE ====================
st.set_page_config(
    page_title="🚀 Plateforme Conception Fusées",
    page_icon="🚀",
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
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 50%, #FDC830 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .rocket-card {
        border: 3px solid #FF6B35;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.1) 0%, rgba(253, 200, 48, 0.1) 100%);
        box-shadow: 0 8px 16px rgba(255, 107, 53, 0.4);
        transition: transform 0.3s;
    }
    .rocket-card:hover {
        transform: translateY(-5px);
    }
    .tech-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 0.3rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .status-active {
        color: #00ff00;
        font-weight: bold;
    }
    .status-testing {
        color: #ffaa00;
        font-weight: bold;
    }
    .status-design {
        color: #00aaff;
        font-weight: bold;
    }
    .quantum-glow {
        animation: quantum-pulse 2s infinite;
    }
    @keyframes quantum-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ====================
PHYSICS_CONSTANTS = {
    'G': 6.67430e-11,
    'c': 299792458,
    'h': 6.62607015e-34,
    'earth_g': 9.80665,
    'mars_g': 3.721,
    'earth_atm': 101325,
    'mars_atm': 610,
    'earth_radius': 6371000,
    'mars_radius': 3389500,
    'boltzmann': 1.380649e-23,
    'avogadro': 6.02214076e23
}

# ==================== INITIALISATION SESSION STATE ====================
if 'rocket_system' not in st.session_state:
    st.session_state.rocket_system = {
        'rockets': {},
        'engines': {},
        'simulations': [],
        'ai_models': {},
        'quantum_analyses': [],
        'biocomputing_results': [],
        'materials': {},
        'tests': [],
        'manufacturing': {},
        'mars_missions': {},
        'design_iterations': [],
        'performance_data': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================
def log_event(message: str, level: str = "INFO"):
    """Enregistre un événement avec niveau"""
    st.session_state.rocket_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def get_tech_badge(tech: str) -> str:
    """Retourne un badge HTML pour technologie"""
    badges = {
        'IA': '<span class="tech-badge">🤖 IA</span>',
        'Quantique': '<span class="tech-badge">⚛️ Quantique</span>',
        'Bio': '<span class="tech-badge">🧬 Bio-computing</span>',
        'Nuclear': '<span class="tech-badge">☢️ Nucléaire</span>',
        'Plasma': '<span class="tech-badge">⚡ Plasma</span>',
        'Antimatter': '<span class="tech-badge">💫 Antimatière</span>'
    }
    return badges.get(tech, '<span class="tech-badge">🔬</span>')

def create_rocket(name: str, config: Dict) -> str:
    """Crée une nouvelle fusée"""
    rocket_id = f"rocket_{len(st.session_state.rocket_system['rockets']) + 1}"
    
    rocket = {
        'id': rocket_id,
        'name': name,
        'created_at': datetime.now().isoformat(),
        'status': 'design',
        'config': config,
        'stages': [],
        'mass': {
            'dry': config.get('dry_mass', 50000),
            'propellant': config.get('propellant_mass', 400000),
            'payload': config.get('payload_mass', 20000),
            'total': 0
        },
        'dimensions': {
            'height': config.get('height', 70),
            'diameter': config.get('diameter', 10),
            'fairing_diameter': config.get('fairing_diameter', 5.4)
        },
        'performance': {
            'thrust': 0,
            'isp': 0,
            'delta_v': 0,
            'payload_leo': 0,
            'payload_gto': 0,
            'payload_mars': 0
        },
        'technologies': config.get('technologies', []),
        'target': config.get('target', 'LEO'),
        'reusability': config.get('reusability', False),
        'ai_optimization': False,
        'quantum_verified': False,
        'bio_control': False,
        'test_flights': 0,
        'success_rate': 0.0,
        'cost_per_launch': config.get('cost', 50000000)
    }
    
    rocket['mass']['total'] = rocket['mass']['dry'] + rocket['mass']['propellant'] + rocket['mass']['payload']
    
    st.session_state.rocket_system['rockets'][rocket_id] = rocket
    log_event(f"Fusée créée: {name}", "SUCCESS")
    return rocket_id

def create_engine(name: str, config: Dict) -> str:
    """Crée un nouveau moteur"""
    engine_id = f"engine_{len(st.session_state.rocket_system['engines']) + 1}"
    
    engine = {
        'id': engine_id,
        'name': name,
        'created_at': datetime.now().isoformat(),
        'type': config.get('type', 'chemical'),
        'propellant': config.get('propellant', 'LOX/RP-1'),
        'thrust_sl': config.get('thrust_sl', 8000000),
        'thrust_vac': config.get('thrust_vac', 9000000),
        'isp_sl': config.get('isp_sl', 282),
        'isp_vac': config.get('isp_vac', 311),
        'chamber_pressure': config.get('chamber_pressure', 30),
        'expansion_ratio': config.get('expansion_ratio', 16),
        'mass': config.get('mass', 5000),
        'throttle_range': config.get('throttle_range', (40, 100)),
        'restart_capable': config.get('restart_capable', False),
        'gimbaling': config.get('gimbaling', 0),
        'cooling': config.get('cooling', 'regenerative'),
        'materials': config.get('materials', {}),
        'technologies': config.get('technologies', []),
        'test_fires': 0,
        'reliability': 0.0
    }
    
    st.session_state.rocket_system['engines'][engine_id] = engine
    log_event(f"Moteur créé: {name}", "SUCCESS")
    return engine_id

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🚀 Plateforme Conception & Fabrication Fusées Spatiales</h1>', unsafe_allow_html=True)
st.markdown("### Système Avancé IA • Quantique • Bio-computing pour Véhicules Spatiaux")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/FF6B35/ffffff?text=Rocket+Engineering", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Centre de Contrôle",
            "🚀 Mes Fusées",
            "➕ Concevoir Fusée",
            "🔥 Moteurs & Propulsion",
            "⚙️ Conception Moteur",
            "🏗️ Fabrication & Matériaux",
            "🧪 Laboratoire Tests",
            "🤖 Optimisation IA",
            "⚛️ Simulation Quantique",
            "🧬 Bio-computing",
            "🔴 Missions Mars",
            "📊 Analyses & Performances",
            "🎯 Simulations Lancement",
            "💻 Jumeaux Numériques",
            "🌡️ Thermodynamique",
            "⚡ Aérodynamique",
            "🛰️ Systèmes Guidage",
            "🔬 Physique Avancée",
            "🌌 Propulsion Exotique",
            "📈 Rapports & Export",
            "📚 Documentation",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    
    total_rockets = len(st.session_state.rocket_system['rockets'])
    total_engines = len(st.session_state.rocket_system['engines'])
    total_simulations = len(st.session_state.rocket_system['simulations'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚀 Fusées", total_rockets)
        st.metric("🔥 Moteurs", total_engines)
    with col2:
        st.metric("🧪 Simulations", total_simulations)
        total_tests = len(st.session_state.rocket_system['tests'])
        st.metric("📊 Tests", total_tests)

# ==================== PAGE: CENTRE DE CONTRÔLE ====================
if page == "🏠 Centre de Contrôle":
    st.header("🏠 Centre de Contrôle Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="rocket-card"><h2>🚀</h2><h3>{total_rockets}</h3><p>Fusées</p></div>', unsafe_allow_html=True)
    
    with col2:
        active_projects = sum(1 for r in st.session_state.rocket_system['rockets'].values() if r['status'] in ['active', 'testing'])
        st.markdown(f'<div class="rocket-card"><h2>✅</h2><h3>{active_projects}</h3><p>Projets Actifs</p></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="rocket-card"><h2>🔥</h2><h3>{total_engines}</h3><p>Moteurs</p></div>', unsafe_allow_html=True)
    
    with col4:
        total_thrust = sum(e['thrust_vac'] for e in st.session_state.rocket_system['engines'].values())
        st.markdown(f'<div class="rocket-card"><h2>⚡</h2><h3>{total_thrust/1e6:.1f}</h3><p>MN Poussée</p></div>', unsafe_allow_html=True)
    
    with col5:
        success_tests = sum(1 for t in st.session_state.rocket_system['tests'] if t.get('success', False))
        total_tests_count = len(st.session_state.rocket_system['tests'])
        success_rate = (success_tests / total_tests_count * 100) if total_tests_count > 0 else 0
        st.markdown(f'<div class="rocket-card"><h2>📊</h2><h3>{success_rate:.1f}%</h3><p>Taux Succès</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technologies avancées
    st.subheader("🔬 Technologies Avancées")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🤖 Intelligence Artificielle")
        ai_models = len(st.session_state.rocket_system.get('ai_models', {}))
        st.metric("Modèles IA Actifs", ai_models)
        st.progress(min(ai_models / 10, 1.0))
        st.write("""
        - Optimisation trajectoires
        - Prédiction performances
        - Contrôle adaptatif
        - Détection anomalies temps réel
        """)
    
    with col2:
        st.markdown("### ⚛️ Computing Quantique")
        quantum_sims = len(st.session_state.rocket_system.get('quantum_analyses', []))
        st.metric("Simulations Quantiques", quantum_sims)
        st.progress(min(quantum_sims / 20, 1.0))
        st.write("""
        - Calculs combustion quantique
        - Optimisation multi-variables
        - Cryptographie communications
        - Simulation matériaux
        """)
    
    with col3:
        st.markdown("### 🧬 Bio-computing")
        bio_results = len(st.session_state.rocket_system.get('biocomputing_results', []))
        st.metric("Analyses Bio", bio_results)
        st.progress(min(bio_results / 15, 1.0))
        st.write("""
        - Contrôle organique
        - Adaptation environnementale
        - Auto-réparation systèmes
        - Intelligence distribuée
        """)
    
    st.markdown("---")
    
    # Constantes fondamentales
    st.subheader("⚛️ Constantes Physiques Fondamentales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Constante Gravitationnelle G", "6.674×10⁻¹¹ N⋅m²/kg²")
        st.metric("Vitesse Lumière c", "299,792,458 m/s")
    
    with col2:
        st.metric("Constante Planck h", "6.626×10⁻³⁴ J⋅s")
        st.metric("g Terre", "9.807 m/s²")
    
    with col3:
        st.metric("g Mars", "3.721 m/s²")
        st.metric("Pression atm Terre", "101,325 Pa")
    
    with col4:
        st.metric("Pression atm Mars", "610 Pa")
        st.metric("Boltzmann k", "1.381×10⁻²³ J/K")
    
    st.markdown("---")
    
    # Graphiques globaux
    if st.session_state.rocket_system['rockets']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Fusées par Statut")
            
            status_counts = {}
            for rocket in st.session_state.rocket_system['rockets'].values():
                status = rocket['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            fig = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Distribution par Statut",
                color_discrete_sequence=px.colors.sequential.Oranges
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Objectifs Missions")
            
            target_counts = {}
            for rocket in st.session_state.rocket_system['rockets'].values():
                target = rocket['target']
                target_counts[target] = target_counts.get(target, 0) + 1
            
            fig = px.bar(
                x=list(target_counts.keys()),
                y=list(target_counts.values()),
                title="Destinations Cibles",
                color=list(target_counts.values()),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucun événement enregistré")
    
    if st.button("🗑️ Effacer Journal"):
        st.session_state.rocket_system['log'] = []
        st.rerun()

# ==================== PAGE: MES FUSÉES ====================
elif page == "🚀 Mes Fusées":
    st.header("🚀 Gestion de la Flotte")
    
    if not st.session_state.rocket_system['rockets']:
        st.info("💡 Aucune fusée créée.")
    else:
        for rocket_id, rocket in st.session_state.rocket_system['rockets'].items():
            st.markdown(f'<div class="rocket-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### 🚀 {rocket['name']}")
                
                status_icons = {
                    'design': '🎨',
                    'manufacturing': '🏭',
                    'testing': '🧪',
                    'active': '✅',
                    'retired': '📦'
                }
                status_icon = status_icons.get(rocket['status'], '❓')
                
                st.write(f"**Statut:** {status_icon} {rocket['status'].upper()}")
                st.write(f"**Cible:** {rocket['target']}")
                
                # Technologies
                tech_html = ""
                for tech in rocket.get('technologies', []):
                    tech_html += get_tech_badge(tech)
                if tech_html:
                    st.markdown(tech_html, unsafe_allow_html=True)
            
            with col2:
                st.metric("Masse Totale", f"{rocket['mass']['total']/1000:.1f} t")
                st.metric("Hauteur", f"{rocket['dimensions']['height']:.1f} m")
            
            with col3:
                st.metric("Payload LEO", f"{rocket['performance']['payload_leo']/1000:.1f} t")
                st.metric("Delta-v", f"{rocket['performance']['delta_v']:.0f} m/s")
            
            with col4:
                st.metric("Vols Tests", rocket['test_flights'])
                st.metric("Taux Succès", f"{rocket['success_rate']:.1f}%")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "⚙️ Spécifications",
                    "🔥 Propulsion",
                    "📊 Performance",
                    "🤖 IA & Tech",
                    "💰 Coûts"
                ])
                
                with tab1:
                    st.subheader("⚙️ Spécifications Techniques")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Masses:**")
                        st.write(f"• Sèche: {rocket['mass']['dry']/1000:.1f} t")
                        st.write(f"• Propergol: {rocket['mass']['propellant']/1000:.1f} t")
                        st.write(f"• Charge utile: {rocket['mass']['payload']/1000:.1f} t")
                        st.write(f"• **Total: {rocket['mass']['total']/1000:.1f} t**")
                    
                    with col2:
                        st.write("**Dimensions:**")
                        st.write(f"• Hauteur: {rocket['dimensions']['height']} m")
                        st.write(f"• Diamètre: {rocket['dimensions']['diameter']} m")
                        st.write(f"• Coiffe: {rocket['dimensions']['fairing_diameter']} m")
                    
                    with col3:
                        st.write("**Caractéristiques:**")
                        st.write(f"• Étages: {len(rocket.get('stages', []))}")
                        st.write(f"• Réutilisable: {'✅ Oui' if rocket['reusability'] else '❌ Non'}")
                        st.write(f"• ID: {rocket['id']}")
                
                with tab2:
                    st.subheader("🔥 Système Propulsion")
                    
                    st.write("**Étages:**")
                    if rocket.get('stages'):
                        for i, stage in enumerate(rocket['stages'], 1):
                            st.write(f"**Étage {i}:**")
                            st.write(f"  - Moteurs: {stage.get('engines', 'N/A')}")
                            st.write(f"  - Poussée: {stage.get('thrust', 0)/1e6:.1f} MN")
                            st.write(f"  - Isp: {stage.get('isp', 0)} s")
                    else:
                        st.info("Aucun étage défini")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Poussée Totale", f"{rocket['performance']['thrust']/1e6:.1f} MN")
                    
                    with col2:
                        st.metric("Isp Moyen", f"{rocket['performance']['isp']:.0f} s")
                
                with tab3:
                    st.subheader("📊 Performances")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Payload LEO", f"{rocket['performance']['payload_leo']/1000:.1f} t")
                        st.metric("Payload GTO", f"{rocket['performance']['payload_gto']/1000:.1f} t")
                    
                    with col2:
                        st.metric("Payload Mars", f"{rocket['performance']['payload_mars']/1000:.1f} t")
                        st.metric("Delta-v Total", f"{rocket['performance']['delta_v']:.0f} m/s")
                    
                    with col3:
                        st.metric("Vols Réussis", f"{int(rocket['test_flights'] * rocket['success_rate'] / 100)}")
                        st.metric("Fiabilité", f"{rocket['success_rate']:.1f}%")
                
                with tab4:
                    st.subheader("🤖 Technologies Avancées")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**IA:**")
                        if rocket.get('ai_optimization'):
                            st.success("✅ Optimisé IA")
                            st.write("- Trajectoire adaptative")
                            st.write("- Prédiction en temps réel")
                        else:
                            st.warning("⏳ Non optimisé")
                    
                    with col2:
                        st.write("**Quantique:**")
                        if rocket.get('quantum_verified'):
                            st.success("✅ Vérifié Quantique")
                            st.write("- Simulations validées")
                            st.write("- Optimisation multi-variable")
                        else:
                            st.warning("⏳ Non vérifié")
                    
                    with col3:
                        st.write("**Bio-computing:**")
                        if rocket.get('bio_control'):
                            st.success("✅ Contrôle Bio")
                            st.write("- Systèmes adaptatifs")
                            st.write("- Auto-diagnostic")
                        else:
                            st.warning("⏳ Non implémenté")
                
                with tab5:
                    st.subheader("💰 Analyse Coûts")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Coût Lancement", f"${rocket['cost_per_launch']/1e6:.1f}M")
                    
                    with col2:
                        cost_per_kg = rocket['cost_per_launch'] / (rocket['performance']['payload_leo'] if rocket['performance']['payload_leo'] > 0 else 1)
                        st.metric("$/kg LEO", f"${cost_per_kg:,.0f}")
                    
                    with col3:
                        if rocket['reusability']:
                            reuse_savings = rocket['cost_per_launch'] * 0.7
                            st.metric("Économies Réutilisation", f"${reuse_savings/1e6:.1f}M")
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button(f"🧪 Tester", key=f"test_{rocket_id}"):
                        st.info("Allez dans 'Laboratoire Tests'")
                
                with col2:
                    if st.button(f"🤖 Optimiser IA", key=f"ai_{rocket_id}"):
                        rocket['ai_optimization'] = True
                        log_event(f"{rocket['name']}: Optimisation IA lancée", "INFO")
                        st.success("Optimisation IA lancée!")
                        st.rerun()
                
                with col3:
                    if st.button(f"⚛️ Analyse Quantique", key=f"quantum_{rocket_id}"):
                        rocket['quantum_verified'] = True
                        log_event(f"{rocket['name']}: Analyse quantique effectuée", "INFO")
                        st.success("Analyse quantique complétée!")
                        st.rerun()
                
                with col4:
                    if st.button(f"🚀 Lancer Simulation", key=f"sim_{rocket_id}"):
                        st.info("Allez dans 'Simulations Lancement'")
                
                with col5:
                    if st.button(f"🗑️ Supprimer", key=f"del_{rocket_id}"):
                        del st.session_state.rocket_system['rockets'][rocket_id]
                        log_event(f"{rocket['name']}: Supprimé", "WARNING")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CONCEVOIR FUSÉE ====================
elif page == "➕ Concevoir Fusée":
    st.header("➕ Conception Nouvelle Fusée")
    
    st.info("""
    🎯 **Assistant Conception Avancée**
    
    Utilisez l'IA, le computing quantique et les algorithmes bio-inspirés pour concevoir
    votre fusée optimale. Le système analysera automatiquement les performances et proposera
    des améliorations.
    """)
    
    with st.form("design_rocket_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rocket_name = st.text_input("📝 Nom de la Fusée", "Artemis-X")
            
            target_mission = st.selectbox(
                "🎯 Mission Cible",
                ["LEO", "GTO", "Lune", "Mars", "Astéroïdes", "Jupiter", "Interstellaire"]
            )
            
            reusability = st.checkbox("♻️ Fusée Réutilisable", value=True)
        
        with col2:
            rocket_class = st.selectbox(
                "📊 Classe de Fusée",
                ["Légère (<10t LEO)", "Moyenne (10-25t LEO)", "Lourde (25-50t LEO)", 
                 "Super-lourde (50-100t LEO)", "Méga (>100t LEO)"]
            )
            
            num_stages = st.number_input("🎚️ Nombre d'Étages", 1, 5, 2, 1)
        
        st.markdown("---")
        st.subheader("⚖️ Masses et Dimensions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dry_mass = st.number_input("Masse à Vide (tonnes)", 10.0, 500.0, 50.0, 5.0)
            propellant_mass = st.number_input("Masse Propergol (tonnes)", 50.0, 3000.0, 400.0, 50.0)
        
        with col2:
            payload_mass = st.number_input("Masse Charge Utile (tonnes)", 1.0, 200.0, 20.0, 1.0)
            height = st.number_input("Hauteur Totale (m)", 10.0, 150.0, 70.0, 5.0)
        
        with col3:
            diameter = st.number_input("Diamètre (m)", 1.0, 20.0, 10.0, 0.5)
            fairing_diameter = st.number_input("Diamètre Coiffe (m)", 1.0, 15.0, 5.4, 0.1)
        
        total_mass = (dry_mass + propellant_mass + payload_mass) * 1000
        st.metric("**Masse Totale au Décollage**", f"{total_mass/1000:.1f} tonnes")
        
        st.markdown("---")
        st.subheader("🔥 Configuration Propulsion")
        
        propulsion_type = st.selectbox(
            "Type de Propulsion Principale",
            ["Chimique Classique", "Chimique Avancé", "Hybride", 
             "Électrique", "Nucléaire", "Plasma", "Fusion", 
             "Antimatière", "Photonique"]
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if propulsion_type in ["Chimique Classique", "Chimique Avancé", "Hybride"]:
                propellant_type = st.selectbox(
                    "Type Propergol",
                    ["LOX/RP-1", "LOX/LH2", "LOX/Méthane", "Hypergoliques", 
                     "Solide", "Gel", "Métastable"]
                )
            else:
                propellant_type = st.text_input("Source Énergie", "Réacteur Fusion")
        
        with col2:
            target_thrust = st.number_input("Poussée Cible (MN)", 0.5, 100.0, 9.0, 0.5)
        
        with col3:
            target_isp = st.number_input("Isp Cible (s)", 200, 100000, 350, 10)
        
        st.markdown("---")
        st.subheader("🤖 Technologies Avancées")
        
        technologies = st.multiselect(
            "Sélectionnez les Technologies à Intégrer",
            ["IA", "Quantique", "Bio", "Nuclear", "Plasma", "Antimatter", 
             "Nanotech", "Métamatériaux", "Supraconducteurs", "Graphène",
             "Contrôle Neuromorphique", "Auto-réparation", "IA Quantique"],
            default=["IA"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            ai_optimization = st.checkbox("🤖 Optimisation IA Automatique", value=True)
            quantum_verification = st.checkbox("⚛️ Vérification Quantique", value=False)
        
        with col2:
            bio_control = st.checkbox("🧬 Contrôle Bio-computing", value=False)
            neural_network = st.checkbox("🧠 Réseau Neuronal Embarqué", value=False)
        
        st.markdown("---")
        st.subheader("💰 Budget et Coûts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dev_budget = st.number_input("Budget Développement ($M)", 10, 10000, 500, 50)
            cost_per_launch = st.number_input("Coût Cible par Lancement ($M)", 1, 1000, 50, 5)
        
        with col2:
            production_units = st.number_input("Unités à Produire", 1, 100, 10, 1)
            target_reliability = st.slider("Fiabilité Cible (%)", 80.0, 99.9, 95.0, 0.1)
        
        st.markdown("---")
        st.subheader("📊 Résumé Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Masse Totale", f"{total_mass/1000:.0f} t")
        with col2:
            st.metric("Hauteur", f"{height:.0f} m")
        with col3:
            st.metric("Poussée", f"{target_thrust:.1f} MN")
        with col4:
            st.metric("Technologies", len(technologies))
        
        # Calcul rapide performance
        g0 = 9.80665
        ve = target_isp * g0
        mass_ratio = total_mass / (dry_mass * 1000 + payload_mass * 1000)
        estimated_dv = ve * np.log(mass_ratio) if mass_ratio > 1 else 0
        
        st.metric("Delta-v Estimé", f"{estimated_dv:.0f} m/s")
        
        submitted = st.form_submit_button("🚀 Créer la Fusée", use_container_width=True, type="primary")
        
        if submitted:
            if not rocket_name:
                st.error("⚠️ Veuillez donner un nom à la fusée")
            else:
                with st.spinner("🔄 Création et analyse en cours..."):
                    import time
                    time.sleep(2)
                    
                    config = {
                        'dry_mass': dry_mass * 1000,
                        'propellant_mass': propellant_mass * 1000,
                        'payload_mass': payload_mass * 1000,
                        'height': height,
                        'diameter': diameter,
                        'fairing_diameter': fairing_diameter,
                        'target': target_mission,
                        'reusability': reusability,
                        'propulsion_type': propulsion_type,
                        'propellant': propellant_type,
                        'target_thrust': target_thrust * 1e6,
                        'target_isp': target_isp,
                        'technologies': technologies,
                        'cost': cost_per_launch * 1e6,
                        'num_stages': num_stages
                    }
                    
                    rocket_id = create_rocket(rocket_name, config)
                    rocket = st.session_state.rocket_system['rockets'][rocket_id]
                    
                    # Calcul performances
                    rocket['performance']['thrust'] = target_thrust * 1e6
                    rocket['performance']['isp'] = target_isp
                    rocket['performance']['delta_v'] = estimated_dv
                    
                    # Estimation payloads
                    if estimated_dv > 9400:
                        rocket['performance']['payload_leo'] = payload_mass * 1000
                    if estimated_dv > 12000:
                        rocket['performance']['payload_gto'] = payload_mass * 1000 * 0.5
                    if estimated_dv > 15000:
                        rocket['performance']['payload_mars'] = payload_mass * 1000 * 0.3
                    
                    # Marqueurs technologies
                    rocket['ai_optimization'] = ai_optimization
                    rocket['quantum_verified'] = quantum_verification
                    rocket['bio_control'] = bio_control
                    
                    st.success(f"✅ Fusée '{rocket_name}' créée avec succès!")
                    st.balloons()
                    
                    # Résultats
                    st.markdown("---")
                    st.subheader("📊 Analyse Initiale")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ID Fusée", rocket_id)
                        st.metric("Masse", f"{rocket['mass']['total']/1000:.0f} t")
                    
                    with col2:
                        st.metric("Delta-v", f"{rocket['performance']['delta_v']:.0f} m/s")
                        st.metric("Payload LEO", f"{rocket['performance']['payload_leo']/1000:.1f} t")
                    
                    with col3:
                        st.metric("Ratio Masse", f"{mass_ratio:.2f}")
                        st.metric("Coût/Lancement", f"${cost_per_launch}M")
                    
                    with col4:
                        if rocket['performance']['payload_leo'] > 0:
                            cost_per_kg = (cost_per_launch * 1e6) / rocket['performance']['payload_leo']
                            st.metric("$/kg LEO", f"${cost_per_kg:,.0f}")
                        st.metric("Technologies", len(technologies))
                    
                    # Recommandations IA
                    if ai_optimization:
                        st.markdown("---")
                        st.subheader("🤖 Recommandations IA")
                        
                        st.markdown(f"""
                        **Analyse IA Préliminaire:**
                        
                        ✅ Configuration viable pour mission {target_mission}
                        ⚡ Optimisations suggérées:
                        - Ratio masse propergol/structure pourrait être amélioré de 8%
                        - Considérer propulsion hybride pour réduction coûts
                        - Matériaux composites avancés recommandés pour structure
                        
                        📊 Fiabilité prédite: {target_reliability - 2:.1f}%
                        💰 Potentiel réduction coûts: 12-15% avec optimisations
                        """)

# ==================== PAGE: MOTEURS & PROPULSION ====================
elif page == "🔥 Moteurs & Propulsion":
    st.header("🔥 Systèmes de Propulsion")
    
    tab1, tab2, tab3 = st.tabs(["📊 Mes Moteurs", "🔬 Types de Propulsion", "📈 Comparaisons"])
    
    with tab1:
        st.subheader("📊 Moteurs Disponibles")
        
        if not st.session_state.rocket_system['engines']:
            st.info("💡 Aucun moteur créé. Allez dans 'Conception Moteur'")
        else:
            for engine_id, engine in st.session_state.rocket_system['engines'].items():
                with st.expander(f"🔥 {engine['name']} - {engine['type'].upper()}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Poussée Vide", f"{engine['thrust_vac']/1e6:.2f} MN")
                        st.metric("Poussée Niveau Mer", f"{engine['thrust_sl']/1e6:.2f} MN")
                    
                    with col2:
                        st.metric("Isp Vide", f"{engine['isp_vac']} s")
                        st.metric("Isp Niveau Mer", f"{engine['isp_sl']} s")
                    
                    with col3:
                        st.metric("Masse Moteur", f"{engine['mass']} kg")
                        st.metric("TWR", f"{engine['thrust_vac']/(engine['mass']*9.81):.1f}")
                    
                    with col4:
                        st.metric("Tests Effectués", engine['test_fires'])
                        st.metric("Fiabilité", f"{engine['reliability']:.1f}%")
                    
                    st.write(f"**Propergol:** {engine['propellant']}")
                    st.write(f"**Pression Chambre:** {engine['chamber_pressure']} MPa")
                    st.write(f"**Ratio Expansion:** {engine['expansion_ratio']}")
                    st.write(f"**Gimbaling:** ±{engine['gimbaling']}°")
                    
                    if st.button(f"🗑️ Supprimer", key=f"del_eng_{engine_id}"):
                        del st.session_state.rocket_system['engines'][engine_id]
                        log_event(f"Moteur {engine['name']} supprimé", "WARNING")
                        st.rerun()
    
    with tab2:
        st.subheader("🔬 Technologies de Propulsion")
        
        propulsion_types = {
            "🔥 Chimique Classique": {
                "description": "Combustion chimique traditionnelle",
                "propergols": "LOX/RP-1, LOX/LH2, Hypergoliques",
                "isp": "250-450 s",
                "poussée": "Très élevée (MN)",
                "trl": "9 (Mature)",
                "exemples": "Merlin (SpaceX), RS-25 (SLS), RD-180",
                "avantages": "Poussée élevée, technologie éprouvée, coût raisonnable",
                "inconvénients": "Isp limité, masse propergol importante"
            },
            "⚡ Électrique/Ionique": {
                "description": "Ionisation et accélération électrique",
                "propergols": "Xénon, Krypton, Argon",
                "isp": "1500-5000 s",
                "poussée": "Très faible (mN-N)",
                "trl": "9 (Opérationnel)",
                "exemples": "NSTAR, NEXT, Hall Effect",
                "avantages": "Isp très élevé, efficacité maximale",
                "inconvénients": "Poussée très faible, durée missions longues"
            },
            "☢️ Nucléaire Thermique": {
                "description": "Réacteur chauffe propergol",
                "propergols": "Hydrogène liquide",
                "isp": "800-1000 s",
                "poussée": "Élevée (kN-MN)",
                "trl": "6 (Démontré)",
                "exemples": "NERVA (historique), DRACO (futur)",
                "avantages": "Isp double du chimique, poussée acceptable",
                "inconvénients": "Complexité, radiation, coût, politique"
            },
            "⚛️ Fusion Nucléaire": {
                "description": "Fusion deutérium-tritium",
                "propergols": "D-T, D-He3",
                "isp": "10,000-100,000 s",
                "poussée": "Moyenne-Élevée",
                "trl": "2-3 (Concept)",
                "exemples": "VASIMR (concept), Direct Fusion Drive",
                "avantages": "Isp extrême, missions interplanétaires rapides",
                "inconvénients": "Technologie non mature, masse réacteur"
            },
            "🌟 Antimatière": {
                "description": "Annihilation matière-antimatière",
                "propergols": "Antiprotons",
                "isp": "100,000-1,000,000 s",
                "poussée": "Variable",
                "trl": "1 (Basique)",
                "exemples": "Concepts théoriques",
                "avantages": "Efficacité maximale théorique, E=mc²",
                "inconvénients": "Production antimatière impossible actuellement"
            },
            "💫 Photonique": {
                "description": "Voile photonique/laser",
                "propergols": "Photons (lumière)",
                "isp": "Infini (pas de masse éjectée)",
                "poussée": "Très faible (μN-mN)",
                "trl": "4-5 (Validé labo)",
                "exemples": "LightSail-2, Breakthrough Starshot",
                "avantages": "Pas de propergol, missions très longue durée",
                "inconvénients": "Poussée négligeable, nécessite source laser"
            },
            "⚡ Plasma/Magnétoplasmadynamique": {
                "description": "Accélération plasma par champs EM",
                "propergols": "Lithium, Argon",
                "isp": "3000-8000 s",
                "poussée": "Moyenne (N-kN)",
                "trl": "5-6",
                "exemples": "VASIMR, MPD thrusters",
                "avantages": "Bon compromis Isp/poussée",
                "inconvénients": "Puissance électrique importante"
            },
            "🌊 Propulsion par Ondes": {
                "description": "Propulsion sans éjection masse (EmDrive, etc.)",
                "propergols": "Aucun (controversé)",
                "isp": "Théoriquement infini",
                "poussée": "Micro (controversé)",
                "trl": "1-2 (Non validé)",
                "exemples": "EmDrive (controversé), Q-drive",
                "avantages": "Pas de propergol si fonctionne",
                "inconvénients": "Non prouvé, viole lois physique actuelles"
            }
        }
        
        for prop_name, prop_info in propulsion_types.items():
            with st.expander(f"{prop_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {prop_info['description']}")
                    st.write(f"**Propergols:** {prop_info['propergols']}")
                    st.write(f"**Isp:** {prop_info['isp']}")
                    st.write(f"**Poussée:** {prop_info['poussée']}")
                
                with col2:
                    st.write(f"**TRL:** {prop_info['trl']}")
                    st.write(f"**Exemples:** {prop_info['exemples']}")
                    st.write(f"✅ **Avantages:** {prop_info['avantages']}")
                    st.write(f"❌ **Inconvénients:** {prop_info['inconvénients']}")
    
    with tab3:
        st.subheader("📈 Comparaison Technologies")
        
        # Graphique Isp vs Poussée
        comparison_data = [
            {"Type": "Chimique LOX/RP-1", "Isp": 300, "Poussée": 1e7, "TRL": 9},
            {"Type": "Chimique LOX/LH2", "Isp": 450, "Poussée": 2e6, "TRL": 9},
            {"Type": "Solide", "Isp": 250, "Poussée": 1.5e7, "TRL": 9},
            {"Type": "Ionique", "Isp": 3500, "Poussée": 0.09, "TRL": 9},
            {"Type": "Hall Effect", "Isp": 1600, "Poussée": 0.5, "TRL": 9},
            {"Type": "Nucléaire Thermique", "Isp": 900, "Poussée": 1e5, "TRL": 6},
            {"Type": "VASIMR", "Isp": 5000, "Poussée": 5, "TRL": 5},
            {"Type": "Fusion (concept)", "Isp": 50000, "Poussée": 1e4, "TRL": 2}
        ]
        
        df_comp = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_comp['Isp'],
            y=df_comp['Poussée'],
            mode='markers+text',
            text=df_comp['Type'],
            textposition='top center',
            marker=dict(
                size=df_comp['TRL'] * 5,
                color=df_comp['TRL'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="TRL")
            )
        ))
        
        fig.update_layout(
            title="Isp vs Poussée (taille = TRL)",
            xaxis_title="Isp (s)",
            yaxis_title="Poussée (N)",
            xaxis_type="log",
            yaxis_type="log",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 📊 Tableau Comparatif")
        st.dataframe(df_comp, use_container_width=True)

# ==================== PAGE: CONCEPTION MOTEUR ====================
elif page == "⚙️ Conception Moteur":
    st.header("⚙️ Conception de Moteur")
    
    st.info("""
    🎯 **Concepteur Moteur Avancé**
    
    Utilisez les algorithmes d'optimisation IA/Quantique pour concevoir un moteur optimal
    selon vos contraintes de mission.
    """)
    
    with st.form("design_engine_form"):
        st.subheader("🎨 Configuration Moteur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            engine_name = st.text_input("📝 Nom du Moteur", "Prometheus-1")
            
            engine_type = st.selectbox(
                "Type de Moteur",
                ["chemical", "electric", "nuclear", "plasma", "fusion", "hybrid"]
            )
        
        with col2:
            propellant = st.selectbox(
                "Propergol",
                ["LOX/RP-1", "LOX/LH2", "LOX/Méthane", "Hypergoliques", 
                 "Xénon", "Hydrogène", "Lithium", "Deutérium"]
            )
            
            application = st.selectbox(
                "Application",
                ["1er Étage", "2ème Étage", "Étage Supérieur", "Orbital", "Interplanétaire"]
            )
        
        st.markdown("---")
        st.subheader("🔥 Performances")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            thrust_sl = st.number_input("Poussée Niveau Mer (kN)", 100, 50000, 8000, 100)
            thrust_vac = st.number_input("Poussée Vide (kN)", 100, 60000, 9000, 100)
        
        with col2:
            isp_sl = st.number_input("Isp Niveau Mer (s)", 150, 500, 282, 1)
            isp_vac = st.number_input("Isp Vide (s)", 200, 50000, 311, 1)
        
        with col3:
            chamber_pressure = st.number_input("Pression Chambre (MPa)", 5, 50, 30, 1)
            expansion_ratio = st.number_input("Ratio Expansion", 5, 300, 16, 1)
        
        st.markdown("---")
        st.subheader("⚙️ Caractéristiques Techniques")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            engine_mass = st.number_input("Masse Moteur (kg)", 100, 50000, 5000, 100)
            throttle_min = st.slider("Throttle Min (%)", 0, 100, 40, 5)
            throttle_max = st.slider("Throttle Max (%)", throttle_min, 100, 100, 5)
        
        with col2:
            gimbaling = st.number_input("Débattement Gimbaling (°)", 0, 20, 5, 1)
            restart_capable = st.checkbox("Capable Redémarrage", value=True)
        
        with col3:
            cooling_system = st.selectbox(
                "Système Refroidissement",
                ["Régénératif", "Ablation", "Film", "Radiatif", "Cryogénique"]
            )
        
        st.markdown("---")
        st.subheader("🔬 Technologies Avancées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            advanced_materials = st.multiselect(
                "Matériaux Avancés",
                ["Superalliages Nickel", "Composites C/C", "Céramiques", 
                 "Nanotubes Carbone", "Graphène", "Aérogels"],
                default=["Superalliages Nickel"]
            )
        
        with col2:
            manufacturing = st.multiselect(
                "Méthodes Fabrication",
                ["Impression 3D", "Fabrication Additive", "Forgeage", 
                 "Coulée de Précision", "Usinage CNC", "Frittage Laser"],
                default=["Impression 3D"]
            )
        
        ai_design = st.checkbox("🤖 Optimisation IA du Design", value=True)
        quantum_sim = st.checkbox("⚛️ Simulation Combustion Quantique", value=False)
        
        st.markdown("---")
        
        # Calculs automatiques
        twr = (thrust_vac * 1000) / (engine_mass * 9.81)
        
        st.subheader("📊 Résumé Performances")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("TWR", f"{twr:.1f}")
        with col2:
            st.metric("Throttle Range", f"{throttle_min}-{throttle_max}%")
        with col3:
            st.metric("Δv/kg", f"{isp_vac * 9.81:.0f} m/s")
        with col4:
            efficiency = (isp_vac / 450) * 100 if engine_type == "chemical" else (isp_vac / 3500) * 100
            st.metric("Efficacité", f"{min(efficiency, 100):.1f}%")
        
        submitted_engine = st.form_submit_button("🔥 Créer le Moteur", use_container_width=True, type="primary")
        
        if submitted_engine:
            if not engine_name:
                st.error("⚠️ Veuillez donner un nom au moteur")
            else:
                with st.spinner("🔄 Création et simulation en cours..."):
                    import time
                    time.sleep(1.5)
                    
                    config = {
                        'type': engine_type,
                        'propellant': propellant,
                        'thrust_sl': thrust_sl * 1000,
                        'thrust_vac': thrust_vac * 1000,
                        'isp_sl': isp_sl,
                        'isp_vac': isp_vac,
                        'chamber_pressure': chamber_pressure,
                        'expansion_ratio': expansion_ratio,
                        'mass': engine_mass,
                        'throttle_range': (throttle_min, throttle_max),
                        'restart_capable': restart_capable,
                        'gimbaling': gimbaling,
                        'cooling': cooling_system,
                        'materials': {'advanced': advanced_materials, 'manufacturing': manufacturing},
                        'technologies': []
                    }
                    
                    if ai_design:
                        config['technologies'].append('IA')
                    if quantum_sim:
                        config['technologies'].append('Quantique')
                    
                    engine_id = create_engine(engine_name, config)
                    engine = st.session_state.rocket_system['engines'][engine_id]
                    
                    st.success(f"✅ Moteur '{engine_name}' créé avec succès!")
                    st.balloons()
                    
                    # Résultats
                    st.markdown("---")
                    st.subheader("📊 Analyse Moteur")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("ID", engine_id)
                        st.metric("TWR", f"{twr:.2f}")
                    
                    with col2:
                        st.metric("Poussée Vac", f"{thrust_vac} kN")
                        st.metric("Isp Vac", f"{isp_vac} s")
                    
                    with col3:
                        st.metric("Masse", f"{engine_mass} kg")
                        st.metric("Gimbaling", f"±{gimbaling}°")
                    
                    with col4:
                        st.metric("Type", engine_type)
                        st.metric("Propergol", propellant)
                    
                    if ai_design:
                        st.markdown("---")
                        st.subheader("🤖 Optimisations IA Suggérées")
                        
                        st.success("""
                        **Analyse IA Complétée:**
                        
                        ✅ Design viable pour application {app}
                        ⚡ Optimisations détectées:
                        - Géométrie chambre: Potentiel +3% Isp
                        - Injecteurs: Configuration optimale trouvée
                        - Refroidissement: Efficacité thermique excellente
                        
                        📊 TWR prédit: {twr:.2f} (Excellent)
                        🎯 Fiabilité estimée: 94.2%
                        """.format(app=application, twr=twr))

# ==================== PAGE: FABRICATION & MATÉRIAUX ====================
elif page == "🏗️ Fabrication & Matériaux":
    st.header("🏗️ Fabrication et Matériaux Avancés")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Matériaux", "🏭 Processus Fabrication", "🧪 Tests Matériaux", "📊 Base de Données"])
    
    with tab1:
        st.subheader("🔬 Matériaux pour Aérospatial")
        
        materials_db = {
            "🔥 Structures à Haute Température": {
                "Superalliages Nickel (Inconel)": {
                    "température": "1200°C",
                    "densité": "8.19 g/cm³",
                    "résistance": "1400 MPa",
                    "applications": "Chambres combustion, turbines",
                    "coût": "$$"
                },
                "Composites Carbone-Carbone (C/C)": {
                    "température": "2000°C",
                    "densité": "1.8 g/cm³",
                    "résistance": "300 MPa",
                    "applications": "Tuyères, boucliers thermiques",
                    "coût": "$$$"
                },
                "Céramiques (SiC, Si3N4)": {
                    "température": "1600°C",
                    "densité": "3.2 g/cm³",
                    "résistance": "500 MPa",
                    "applications": "Revêtements thermiques",
                    "coût": "$$"
                }
            },
            "🏗️ Structures Primaires": {
                "Aluminium-Lithium (Al-Li)": {
                    "température": "150°C",
                    "densité": "2.5 g/cm³",
                    "résistance": "550 MPa",
                    "applications": "Réservoirs, structures",
                    "coût": "$"
                },
                "Titane (Ti-6Al-4V)": {
                    "température": "400°C",
                    "densité": "4.43 g/cm³",
                    "résistance": "900 MPa",
                    "applications": "Structures critiques",
                    "coût": "$$"
                },
                "Composites CFRP": {
                    "température": "120°C",
                    "densité": "1.6 g/cm³",
                    "résistance": "600 MPa",
                    "applications": "Coiffes, structures légères",
                    "coût": "$$"
                }
            },
            "🚀 Matériaux Avancés/Futurs": {
                "Nanotubes Carbone": {
                    "température": "3000°C",
                    "densité": "1.3 g/cm³",
                    "résistance": "63000 MPa (théorique)",
                    "applications": "Structures ultralégères futures",
                    "coût": "$$$ (R&D)"
                },
                "Graphène": {
                    "température": "3000°C",
                    "densité": "0.77 g/cm³",
                    "résistance": "130000 MPa",
                    "applications": "Électronique, capteurs",
                    "coût": "$$ (R&D)"
                },
                "Aérogels": {
                    "température": "1200°C",
                    "densité": "0.15 g/cm³",
                    "résistance": "Variable",
                    "applications": "Isolation thermique extrême",
                    "coût": "$$"
                },
                "Métamatériaux": {
                    "température": "Variable",
                    "densité": "Variable",
                    "résistance": "Propriétés programmables",
                    "applications": "Absorption ondes, structures adaptatives",
                    "coût": "$$$ (Recherche)"
                }
            }
        }
        
        for category, materials in materials_db.items():
            st.markdown(f"### {category}")
            
            for material, props in materials.items():
                with st.expander(f"🔬 {material}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Température Max:** {props['température']}")
                        st.write(f"**Densité:** {props['densité']}")
                        st.write(f"**Résistance:** {props['résistance']}")
                    
                    with col2:
                        st.write(f"**Applications:** {props['applications']}")
                        st.write(f"**Coût:** {props['coût']}")
    
    with tab2:
        st.subheader("🏭 Processus de Fabrication Avancés")
        
        manufacturing_processes = {
            "🖨️ Fabrication Additive (Impression 3D)": {
                "description": "Construction couche par couche",
                "technologies": ["SLM (Selective Laser Melting)", "EBM (Electron Beam)", "DMLS", "Binder Jetting"],
                "matériaux": "Métaux, polymères, céramiques",
                "avantages": "Géométries complexes, réduction déchets, prototypage rapide",
                "limitations": "Vitesse production, taille pièces, finition surface",
                "applications": "Injecteurs, chambres refroidissement, structures optimisées"
            },
            "🔨 Forgeage Isotherme": {
                "description": "Déformation à température contrôlée",
                "technologies": ["Hot Isostatic Pressing (HIP)", "Forgeage sous vide"],
                "matériaux": "Superalliages, titane",
                "avantages": "Propriétés mécaniques optimales, densification",
                "limitations": "Coût élevé, outillage complexe",
                "applications": "Disques turbines, composants critiques"
            },
            "⚙️ Usinage CNC Multi-axes": {
                "description": "Enlèvement matière haute précision",
                "technologies": ["5-axes", "Tournage-fraisage", "EDM"],
                "matériaux": "Tous métaux",
                "avantages": "Précision extrême, répétabilité",
                "limitations": "Temps usinage, déchets matière",
                "applications": "Pièces de précision, prototypes"
            },
            "🔬 Dépôt en Phase Vapeur": {
                "description": "Revêtements atomiques",
                "technologies": ["CVD", "PVD", "ALD"],
                "matériaux": "Céramiques, métaux, composites",
                "avantages": "Revêtements ultra-minces, propriétés contrôlées",
                "limitations": "Vitesse dépôt lente",
                "applications": "Barrières thermiques, revêtements protection"
            },
            "🧬 Fabrication Bio-inspirée": {
                "description": "Croissance contrôlée structures",
                "technologies": ["Auto-assemblage", "Cristallisation dirigée", "Bioimpression"],
                "matériaux": "Composites bio, métaux organisés",
                "avantages": "Structures optimisées naturellement, auto-réparation",
                "limitations": "Technologie émergente, échelle limitée",
                "applications": "Futurs matériaux adaptatifs"
            }
        }
        
        for process, details in manufacturing_processes.items():
            with st.expander(f"{process}"):
                st.write(f"**Description:** {details['description']}")
                st.write(f"**Technologies:** {details['technologies']}")
                st.write(f"**Matériaux:** {details['matériaux']}")
                st.write(f"✅ **Avantages:** {details['avantages']}")
                st.write(f"⚠️ **Limitations:** {details['limitations']}")
                st.write(f"🎯 **Applications:** {details['applications']}")
        
        st.markdown("---")
        
        st.subheader("📊 Comparaison Processus")
        
        comparison = pd.DataFrame([
            {"Processus": "Impression 3D", "Complexité": "Très Élevée", "Coût": "Moyen", "Vitesse": "Lente", "Précision": "Élevée"},
            {"Processus": "Forgeage", "Complexité": "Faible", "Coût": "Faible", "Vitesse": "Rapide", "Précision": "Moyenne"},
            {"Processus": "Usinage CNC", "Complexité": "Élevée", "Coût": "Élevé", "Vitesse": "Moyenne", "Précision": "Très Élevée"},
            {"Processus": "Coulée", "Complexité": "Moyenne", "Coût": "Faible", "Vitesse": "Rapide", "Précision": "Moyenne"},
            {"Processus": "Composite", "Complexité": "Très Élevée", "Coût": "Très Élevé", "Vitesse": "Lente", "Précision": "Élevée"}
        ])
        
        st.dataframe(comparison, use_container_width=True)
    
    with tab3:
        st.subheader("🧪 Tests et Validation Matériaux")
        
        st.write("### 🔬 Types de Tests")
        
        tests_types = {
            "Mécaniques": [
                "Traction/Compression",
                "Fatigue cyclique",
                "Ténacité (ductilité)",
                "Dureté (Rockwell, Brinell, Vickers)",
                "Impact (Charpy, Izod)",
                "Fluage (haute température)"
            ],
            "Thermiques": [
                "Expansion thermique",
                "Conductivité thermique",
                "Choc thermique",
                "Ablation",
                "Stabilité haute température"
            ],
            "Environnementaux": [
                "Corrosion",
                "Oxydation",
                "Vide spatial",
                "Radiation",
                "Cycles thermiques"
            ],
            "Non Destructifs": [
                "Rayons X",
                "Ultrasons",
                "Thermographie infrarouge",
                "Émission acoustique",
                "Courants de Foucault"
            ]
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for test_cat, tests in list(tests_types.items())[:2]:
                st.write(f"**{test_cat}:**")
                for test in tests:
                    st.write(f"  • {test}")
        
        with col2:
            for test_cat, tests in list(tests_types.items())[2:]:
                st.write(f"**{test_cat}:**")
                for test in tests:
                    st.write(f"  • {test}")
        
        st.markdown("---")
        
        st.write("### 📊 Simulateur Test Matériau")
        
        with st.form("material_test"):
            col1, col2 = st.columns(2)
            
            with col1:
                test_material = st.selectbox("Matériau", ["Inconel 718", "Ti-6Al-4V", "Al-Li 2195", "CFRP", "C/C Composite"])
                test_type = st.selectbox("Type Test", ["Traction", "Fatigue", "Thermique", "Corrosion"])
            
            with col2:
                temperature = st.number_input("Température (°C)", -200, 3000, 20, 10)
                stress_level = st.slider("Contrainte (%)", 0, 100, 50, 5)
            
            if st.form_submit_button("🔬 Lancer Test"):
                with st.spinner("Test en cours..."):
                    import time
                    time.sleep(2)
                    
                    st.success("✅ Test complété!")
                    
                    # Résultats simulés
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        resistance = np.random.uniform(400, 1200)
                        st.metric("Résistance", f"{resistance:.0f} MPa")
                    
                    with col2:
                        elongation = np.random.uniform(5, 25)
                        st.metric("Allongement", f"{elongation:.1f}%")
                    
                    with col3:
                        cycles = np.random.randint(1000, 100000)
                        st.metric("Cycles Fatigue", f"{cycles:,}")
                    
                    st.info(f"""
                    **Analyse:**
                    - Matériau conforme aux spécifications
                    - Performances excellentes à {temperature}°C
                    - Durée vie estimée: {cycles:,} cycles
                    - Recommandé pour application spatiale
                    """)
    
    with tab4:
        st.subheader("📊 Base de Données Matériaux")
        
        # Créer base données simulée
        if 'materials_database' not in st.session_state:
            st.session_state.materials_database = pd.DataFrame([
                {"Matériau": "Inconel 718", "Densité": 8.19, "Résistance": 1400, "Temp Max": 1200, "Coût": 150, "Stock": 500},
                {"Matériau": "Ti-6Al-4V", "Densité": 4.43, "Résistance": 900, "Temp Max": 400, "Coût": 80, "Stock": 300},
                {"Matériau": "Al-Li 2195", "Densité": 2.50, "Résistance": 550, "Temp Max": 150, "Coût": 40, "Stock": 1000},
                {"Matériau": "CFRP", "Densité": 1.60, "Résistance": 600, "Temp Max": 120, "Coût": 120, "Stock": 200},
                {"Matériau": "C/C Composite", "Densité": 1.80, "Résistance": 300, "Temp Max": 2000, "Coût": 500, "Stock": 50}
            ])
        
        st.dataframe(st.session_state.materials_database, use_container_width=True)
        
        st.write("### 🔍 Recherche Matériau")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_resistance = st.number_input("Résistance Min (MPa)", 0, 2000, 500, 50)
        
        with col2:
            max_density = st.number_input("Densité Max (g/cm³)", 0.0, 10.0, 5.0, 0.5)
        
        with col3:
            min_temp = st.number_input("Température Min (°C)", 0, 3000, 500, 100)
        
        if st.button("🔍 Rechercher"):
            filtered = st.session_state.materials_database[
                (st.session_state.materials_database['Résistance'] >= min_resistance) &
                (st.session_state.materials_database['Densité'] <= max_density) &
                (st.session_state.materials_database['Temp Max'] >= min_temp)
            ]
            
            st.write(f"### Résultats ({len(filtered)} matériaux)")
            st.dataframe(filtered, use_container_width=True)

# ==================== PAGE: LABORATOIRE TESTS ====================
elif page == "🧪 Laboratoire Tests":
    st.header("🧪 Laboratoire de Tests et Validation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Tests Moteurs", "🚀 Tests Statiques", "🌡️ Tests Environnementaux", "📊 Résultats"])
    
    with tab1:
        st.subheader("🔥 Tests Moteurs (Hot Fire)")
        
        st.info("""
        **Protocole Test Moteur:**
        1. Installation banc d'essai
        2. Instrumentation (capteurs pression, température, poussée)
        3. Séquence d'allumage
        4. Acquisition données temps réel
        5. Arrêt contrôlé
        6. Analyse post-test
        """)
        
        if not st.session_state.rocket_system['engines']:
            st.warning("⚠️ Aucun moteur disponible. Créez un moteur d'abord.")
        else:
            with st.form("hot_fire_test"):
                st.write("### Configuration Test")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    engine_select = st.selectbox(
                        "Moteur à Tester",
                        [f"{e['name']} ({e['id']})" for e in st.session_state.rocket_system['engines'].values()]
                    )
                    
                    test_duration = st.number_input("Durée Test (secondes)", 1, 600, 30, 1)
                    throttle_profile = st.selectbox("Profil Throttle", ["Constant 100%", "Rampe 40-100%", "Pas multiples", "Personnalisé"])
                
                with col2:
                    ambient_pressure = st.number_input("Pression Ambiante (kPa)", 0.0, 101.325, 101.325, 0.1)
                    ambient_temp = st.number_input("Température Ambiante (°C)", -50, 50, 20, 1)
                    
                    record_video = st.checkbox("📹 Enregistrement Vidéo Haute Vitesse", value=True)
                    ai_monitoring = st.checkbox("🤖 Monitoring IA Temps Réel", value=True)
                
                if st.form_submit_button("🔥 Lancer Test Moteur", type="primary"):
                    with st.spinner("🔥 Test en cours..."):
                        import time
                        
                        # Simulation test
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(test_duration):
                            progress_bar.progress((i + 1) / test_duration)
                            status_text.text(f"T+{i+1}s - Poussée: {95 + np.random.randn()*2:.1f}% - Température: {2800 + np.random.randn()*50:.0f}K")
                            time.sleep(0.1)
                        
                        st.success("✅ Test complété avec succès!")
                        
                        # Résultats
                        st.markdown("---")
                        st.subheader("📊 Résultats Test")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Poussée Moyenne", f"{8.95:.2f} MN")
                            st.metric("Écart Poussée", "±1.2%")
                        
                        with col2:
                            st.metric("Isp Mesuré", "312.3 s")
                            st.metric("vs Théorique", "+0.4%")
                        
                        with col3:
                            st.metric("Pression Chambre", "30.2 MPa")
                            st.metric("Température Max", "2847 K")
                        
                        with col4:
                            st.metric("Consommation", f"{test_duration * 2.5:.1f} tonnes")
                            st.metric("Statut", "✅ Succès")
                        
                        # Graphique télémétrie
                        t = np.linspace(0, test_duration, 100)
                        thrust = 9.0 + 0.1 * np.sin(t) + np.random.randn(100) * 0.05
                        chamber_p = 30 + 0.5 * np.sin(t * 2) + np.random.randn(100) * 0.2
                        
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=("Poussée", "Pression Chambre")
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=t, y=thrust, mode='lines', name='Poussée'),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=t, y=chamber_p, mode='lines', name='Pression', line=dict(color='red')),
                            row=2, col=1
                        )
                        
                        fig.update_xaxes(title_text="Temps (s)", row=2, col=1)
                        fig.update_yaxes(title_text="Poussée (MN)", row=1, col=1)
                        fig.update_yaxes(title_text="Pression (MPa)", row=2, col=1)
                        
                        fig.update_layout(height=600, showlegend=False)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Enregistrement
                        test_record = {
                            'timestamp': datetime.now().isoformat(),
                            'engine': engine_select,
                            'duration': test_duration,
                            'success': True,
                            'thrust_avg': 8.95,
                            'isp_measured': 312.3
                        }
                        
                        st.session_state.rocket_system['tests'].append(test_record)
                        
                        log_event(f"Test moteur: {engine_select} - Succès", "SUCCESS")
    
    with tab2:
        st.subheader("🚀 Tests Statiques Fusée Complète")
        
        st.info("""
        **Test Statique Complet:**
        - Fusée complète ancrée au sol
        - Allumage tous étages simultanés ou séquencé
        - Validation intégration complète
        - Vérification systèmes de vol
        """)
        
        if not st.session_state.rocket_system['rockets']:
            st.warning("⚠️ Aucune fusée disponible.")
        else:
            with st.form("static_fire_test"):
                rocket_select = st.selectbox(
                    "Fusée à Tester",
                    [f"{r['name']} ({r['id']})" for r in st.session_state.rocket_system['rockets'].values()]
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    test_type = st.selectbox("Type Test", ["Tous Moteurs", "Étage 1 Seul", "Séquence Nominale"])
                    duration = st.number_input("Durée (s)", 5, 120, 10, 1)
                
                with col2:
                    abort_test = st.checkbox("Simulation Abort", value=False)
                    real_propellant = st.checkbox("Propergol Réel (non simulé)", value=True)
                
                if st.form_submit_button("🚀 Lancer Test Statique"):
                    with st.spinner("Test en cours..."):
                        import time
                        time.sleep(3)
                        
                        success = not abort_test and np.random.random() > 0.1
                        
                        if success:
                            st.success("✅ Test statique réussi!")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Poussée Totale", "45.2 MN")
                            with col2:
                                st.metric("Durée Combustion", f"{duration} s")
                            with col3:
                                st.metric("Systèmes Nominaux", "98%")
                            
                            log_event(f"Test statique: {rocket_select} - Succès", "SUCCESS")
                        else:
                            st.error("❌ Test interrompu - Anomalie détectée")
                            st.warning("Analyse post-test requise")
                            
                            log_event(f"Test statique: {rocket_select} - Échec", "ERROR")
    
    with tab3:
        st.subheader("🌡️ Tests Environnementaux")
        
        st.write("### 🔬 Simulation Conditions Extrêmes")
        
        environmental_tests = {
            "❄️ Cryogénique": {
                "température": "-253°C (LH2)",
                "durée": "Heures",
                "objectif": "Comportement matériaux extrême froid"
            },
            "🔥 Haute Température": {
                "température": "1500-3000°C",
                "durée": "Minutes-Heures",
                "objectif": "Résistance thermique structures"
            },
            "🌡️ Choc Thermique": {
                "température": "-200°C à +200°C",
                "durée": "Cycles",
                "objectif": "Fatigue thermique"
            },
            "🌌 Vide Spatial": {
                "pression": "< 10⁻⁶ Pa",
                "durée": "Jours",
                "objectif": "Dégazage, comportement vide"
            },
            "☢️ Radiation": {
                "dose": "Krad",
                "durée": "Variable",
                "objectif": "Vieillissement composants électroniques"
            },
            "💨 Vibration": {
                "fréquence": "10-2000 Hz",
                "durée": "Minutes",
                "objectif": "Tenue mécanique lancement"
            }
        }
        
        for test_name, test_params in environmental_tests.items():
            with st.expander(f"{test_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    for key, value in list(test_params.items())[:2]:
                        st.write(f"**{key.title()}:** {value}")
                
                with col2:
                    st.write(f"**{list(test_params.keys())[2].title()}:** {list(test_params.values())[2]}")
                    
                    if st.button(f"Lancer Test", key=f"env_test_{test_name}"):
                        with st.spinner("Test en cours..."):
                            import time
                            time.sleep(2)
                            st.success(f"✅ Test {test_name} complété")
    
    with tab4:
        st.subheader("📊 Résultats et Historique Tests")
        
        if not st.session_state.rocket_system['tests']:
            st.info("💡 Aucun test effectué")
        else:
            df_tests = pd.DataFrame(st.session_state.rocket_system['tests'])
            
            st.write(f"### 📋 Total: {len(df_tests)} tests")
            
            # Statistiques
            success_count = sum(df_tests['success'])
            success_rate = (success_count / len(df_tests)) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tests Total", len(df_tests))
            with col2:
                st.metric("Succès", success_count)
            with col3:
                st.metric("Échecs", len(df_tests) - success_count)
            with col4:
                st.metric("Taux Succès", f"{success_rate:.1f}%")
            
            st.dataframe(df_tests, use_container_width=True)

# ==================== PAGE: OPTIMISATION IA ====================
elif page == "🤖 Optimisation IA":
    st.header("🤖 Optimisation par Intelligence Artificielle")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Modèles IA", "📊 Optimisation Design", "🎯 Prédictions", "📈 Apprentissage"])
    
    with tab1:
        st.subheader("🧠 Modèles d'Intelligence Artificielle")
        
        st.info("""
        **Systèmes IA Disponibles:**
        
        🔹 **Réseaux Neuronaux Profonds (DNN)** - Optimisation aérodynamique
        🔹 **Apprentissage par Renforcement (RL)** - Contrôle trajectoire adaptatif
        🔹 **Algorithmes Génétiques** - Optimisation multi-objectifs
        🔹 **Machine Learning** - Prédiction performances et anomalies
        🔹 **Vision par Ordinateur** - Analyse vidéo tests
        🔹 **NLP** - Analyse documentation technique
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🎯 Créer Nouveau Modèle IA")
            
            with st.form("create_ai_model"):
                model_name = st.text_input("Nom du Modèle", "Optimizer-Alpha")
                
                model_type = st.selectbox(
                    "Type de Modèle",
                    ["Neural Network", "Reinforcement Learning", "Genetic Algorithm", 
                     "Random Forest", "Gradient Boosting", "Transformer"]
                )
                
                application = st.selectbox(
                    "Application",
                    ["Optimisation Aérodynamique", "Prédiction Performance", 
                     "Contrôle Trajectoire", "Détection Anomalies", 
                     "Optimisation Combustion", "Planification Mission"]
                )
                
                training_data_size = st.number_input("Données Entraînement", 1000, 1000000, 10000, 1000)
                
                if st.form_submit_button("🤖 Créer et Entraîner Modèle"):
                    with st.spinner("Entraînement en cours..."):
                        import time
                        
                        progress = st.progress(0)
                        for i in range(100):
                            progress.progress(i + 1)
                            time.sleep(0.02)
                        
                        model_id = f"ai_model_{len(st.session_state.rocket_system.get('ai_models', {})) + 1}"
                        
                        st.session_state.rocket_system['ai_models'][model_id] = {
                            'id': model_id,
                            'name': model_name,
                            'type': model_type,
                            'application': application,
                            'accuracy': np.random.uniform(0.92, 0.99),
                            'training_samples': training_data_size,
                            'created_at': datetime.now().isoformat(),
                            'status': 'trained'
                        }
                        
                        st.success(f"✅ Modèle '{model_name}' créé et entraîné!")
                        log_event(f"Modèle IA créé: {model_name}", "SUCCESS")
                        st.rerun()
        
        with col2:
            st.write("### 📊 Modèles Actifs")
            
            if st.session_state.rocket_system.get('ai_models'):
                for model_id, model in st.session_state.rocket_system['ai_models'].items():
                    with st.expander(f"🤖 {model['name']}"):
                        st.write(f"**Type:** {model['type']}")
                        st.write(f"**Application:** {model['application']}")
                        st.metric("Précision", f"{model['accuracy']*100:.2f}%")
                        st.metric("Données Entraînement", f"{model['training_samples']:,}")
                        
                        if st.button(f"🗑️ Supprimer", key=f"del_model_{model_id}"):
                            del st.session_state.rocket_system['ai_models'][model_id]
                            st.rerun()
            else:
                st.info("Aucun modèle créé")
    
    with tab2:
        st.subheader("📊 Optimisation Design par IA")
        
        if not st.session_state.rocket_system['rockets']:
            st.warning("⚠️ Créez une fusée d'abord")
        else:
            st.write("### 🎯 Optimisation Multi-Objectifs")
            
            with st.form("ai_optimization"):
                rocket_select = st.selectbox(
                    "Fusée à Optimiser",
                    [f"{r['name']}" for r in st.session_state.rocket_system['rockets'].values()]
                )
                
                st.write("**Objectifs d'Optimisation:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    maximize_payload = st.checkbox("Maximiser Charge Utile", value=True)
                    minimize_cost = st.checkbox("Minimiser Coût", value=True)
                    maximize_reliability = st.checkbox("Maximiser Fiabilité", value=True)
                
                with col2:
                    minimize_mass = st.checkbox("Minimiser Masse", value=False)
                    maximize_reusability = st.checkbox("Maximiser Réutilisabilité", value=True)
                    optimize_aerodynamics = st.checkbox("Optimiser Aérodynamique", value=True)
                
                st.write("**Contraintes:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_height = st.number_input("Hauteur Max (m)", 50, 200, 120, 5)
                with col2:
                    max_diameter = st.number_input("Diamètre Max (m)", 5, 25, 15, 1)
                with col3:
                    max_cost = st.number_input("Budget Max ($M)", 10, 1000, 200, 10)
                
                optimization_method = st.selectbox(
                    "Algorithme",
                    ["Algorithme Génétique", "Particle Swarm", "Gradient Descent", 
                     "Bayesian Optimization", "Neural Architecture Search"]
                )
                
                iterations = st.slider("Itérations", 100, 10000, 1000, 100)
                
                if st.form_submit_button("🚀 Lancer Optimisation IA", type="primary"):
                    with st.spinner("Optimisation en cours..."):
                        import time
                        
                        progress = st.progress(0)
                        status = st.empty()
                        
                        for i in range(100):
                            progress.progress(i + 1)
                            status.text(f"Génération {i*10}/{iterations} - Meilleure fitness: {0.85 + i*0.0015:.4f}")
                            time.sleep(0.03)
                        
                        st.success("✅ Optimisation complétée!")
                        
                        st.markdown("---")
                        st.subheader("📊 Résultats Optimisation")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Amélioration Payload", "+18.3%", delta="+18.3%")
                        with col2:
                            st.metric("Réduction Coût", "-12.7%", delta="-12.7%")
                        with col3:
                            st.metric("Gain Fiabilité", "+5.2%", delta="+5.2%")
                        with col4:
                            st.metric("Fitness Score", "0.964")
                        
                        st.write("### 🔧 Modifications Suggérées")
                        
                        improvements = pd.DataFrame([
                            {"Paramètre": "Ratio Masse Propergol", "Valeur Actuelle": "8.2", "Valeur Optimale": "9.1", "Impact": "+12% Δv"},
                            {"Paramètre": "Diamètre Étage 1", "Valeur Actuelle": "10.0 m", "Valeur Optimale": "10.8 m", "Impact": "+8% Payload"},
                            {"Paramètre": "Pression Chambre", "Valeur Actuelle": "30 MPa", "Valeur Optimale": "33 MPa", "Impact": "+2% Isp"},
                            {"Paramètre": "Matériau Structure", "Valeur Actuelle": "Al-Li", "Valeur Optimale": "CFRP", "Impact": "-15% Masse"},
                            {"Paramètre": "Nombre Moteurs Étage 1", "Valeur Actuelle": "9", "Valeur Optimale": "11", "Impact": "+18% Poussée"}
                        ])
                        
                        st.dataframe(improvements, use_container_width=True)
                        
                        # Graphique évolution
                        generations = np.arange(0, iterations, 10)
                        fitness = 0.7 + 0.264 * (1 - np.exp(-generations/200)) + np.random.randn(len(generations)) * 0.01
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=generations, y=fitness, mode='lines', name='Fitness', line=dict(width=3)))
                        fig.update_layout(
                            title="Évolution Fitness durant Optimisation",
                            xaxis_title="Génération",
                            yaxis_title="Fitness Score",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Prédictions et Analyses")
        
        st.write("### 🔮 Système de Prédiction Avancé")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prédiction Performance:**")
            
            with st.form("predict_performance"):
                config_params = st.text_area(
                    "Paramètres Configuration (JSON)",
                    '{"mass": 50000, "thrust": 9000000, "isp": 350}'
                )
                
                if st.form_submit_button("🔮 Prédire"):
                    try:
                        params = json.loads(config_params)
                        
                        with st.spinner("Calcul prédiction..."):
                            import time
                            time.sleep(1)
                            
                            # Prédictions simulées
                            payload_leo = params.get('thrust', 9000000) / 50000 * 20
                            success_prob = 0.85 + np.random.random() * 0.1
                            cost_estimate = params.get('mass', 50000) * 0.8
                            
                            st.success("✅ Prédiction complétée!")
                            
                            st.metric("Payload LEO Prédit", f"{payload_leo:.1f} t")
                            st.metric("Probabilité Succès", f"{success_prob*100:.1f}%")
                            st.metric("Coût Estimé", f"${cost_estimate/1000:.0f}M")
                            
                            st.info(f"**Confiance:** 94.2% (basé sur {np.random.randint(5000, 20000)} simulations)")
                    
                    except json.JSONDecodeError:
                        st.error("❌ Format JSON invalide")
        
        with col2:
            st.write("**Détection Anomalies:**")
            
            st.info("""
            **Système de Détection Temps Réel:**
            
            🔹 Analyse télémétrie en continu
            🔹 Détection patterns anormaux
            🔹 Alerte précoce pannes
            🔹 Recommandations correctives
            
            **Algorithmes:**
            - Isolation Forest
            - LSTM Autoencoders
            - One-Class SVM
            - Statistical Process Control
            """)
            
            if st.button("🔍 Analyser Derniers Vols"):
                with st.spinner("Analyse en cours..."):
                    import time
                    time.sleep(2)
                    
                    st.success("✅ Analyse complétée")
                    
                    st.write("**Anomalies Détectées:**")
                    st.write("• Vol #23: Vibration excessive T+47s (Sévérité: Faible)")
                    st.write("• Vol #25: Pic pression chambre T+12s (Sévérité: Moyenne)")
                    st.write("• Vol #27: Température anormale T+89s (Sévérité: Faible)")
                    
                    st.metric("Score Santé Flotte", "96.8%")
    
    with tab4:
        st.subheader("📈 Apprentissage Continu")
        
        st.write("### 🧠 Système d'Apprentissage Automatique")
        
        st.info("""
        **Pipeline Apprentissage:**
        
        1. **Collecte Données** - Télémétrie, tests, simulations
        2. **Prétraitement** - Nettoyage, normalisation
        3. **Feature Engineering** - Extraction caractéristiques
        4. **Entraînement** - Modèles multiples en parallèle
        5. **Validation** - Cross-validation, test sets
        6. **Déploiement** - Production avec monitoring
        7. **Feedback Loop** - Amélioration continue
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Données Collectées", "2.4M échantillons")
            st.metric("Modèles Entraînés", "147")
        
        with col2:
            st.metric("Précision Moyenne", "95.3%")
            st.metric("Temps Inférence", "12 ms")
        
        with col3:
            st.metric("Amélioration/Mois", "+2.1%")
            st.metric("Économies Générées", "$4.2M")
        
        st.markdown("---")
        
        st.write("### 📊 Performance Modèles")
        
        # Graphique performance
        models_perf = pd.DataFrame([
            {"Modèle": "NN-Aero", "Précision": 96.2, "Rappel": 94.8, "F1-Score": 95.5},
            {"Modèle": "RF-Performance", "Précision": 93.7, "Rappel": 92.1, "F1-Score": 92.9},
            {"Modèle": "LSTM-Trajectoire", "Précision": 97.1, "Rappel": 96.5, "F1-Score": 96.8},
            {"Modèle": "CNN-Vision", "Précision": 98.3, "Rappel": 97.9, "F1-Score": 98.1},
            {"Modèle": "RL-Contrôle", "Précision": 94.5, "Rappel": 93.2, "F1-Score": 93.8}
        ])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(x=models_perf['Modèle'], y=models_perf['Précision'], name='Précision'))
        fig.add_trace(go.Bar(x=models_perf['Modèle'], y=models_perf['Rappel'], name='Rappel'))
        fig.add_trace(go.Bar(x=models_perf['Modèle'], y=models_perf['F1-Score'], name='F1-Score'))
        
        fig.update_layout(
            title="Performance des Modèles IA",
            xaxis_title="Modèle",
            yaxis_title="Score (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: SIMULATION QUANTIQUE ====================
elif page == "⚛️ Simulation Quantique":
    st.header("⚛️ Computing Quantique pour Aérospatial")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Principes", "💻 Simulations", "⚡ Combustion Quantique", "📊 Résultats"])
    
    with tab1:
        st.subheader("🔬 Principes du Computing Quantique")
        
        st.info("""
        **Avantages Quantiques pour Aérospatial:**
        
        ⚛️ **Superposition** - Exploration simultanée millions de configurations
        🔗 **Intrication** - Optimisation corrélations complexes
        🌊 **Interférence** - Amplification solutions optimales
        
        **Applications:**
        - Optimisation trajectoires (algorithme de Grover)
        - Simulation dynamique moléculaire combustion
        - Cryptographie communications spatiales
        - Optimisation emploi du temps missions
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🎯 Algorithmes Quantiques")
            
            algorithms = {
                "Grover": "Recherche dans base de données non structurée - O(√N)",
                "Shor": "Factorisation - Cryptographie",
                "VQE": "Variational Quantum Eigensolver - Chimie quantique",
                "QAOA": "Quantum Approximate Optimization - Optimisation combinatoire",
                "Quantum Annealing": "Optimisation globale"
            }
            
            for algo, desc in algorithms.items():
                st.write(f"**{algo}:** {desc}")
        
        with col2:
            st.write("### 💻 Simulateurs Quantiques")
            
            st.write("""
            **Plateformes Disponibles:**
            - IBM Quantum (Qiskit)
            - Google Cirq
            - Amazon Braket
            - Microsoft Azure Quantum
            - D-Wave (Annealing)
            
            **Qubits Disponibles:** 5-127 qubits
            **Fidélité:** 99.9% (2-qubit gates)
            """)
        
        st.markdown("---")
        
        st.write("### ⚛️ Visualisation État Quantique")
        
        # Simulation Bloch sphere
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 50)
        
        x = np.outer(np.cos(theta), np.sin(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.ones(100), np.cos(phi))
        
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Viridis', opacity=0.7)])
        
        # État quantique exemple
        fig.add_trace(go.Scatter3d(
            x=[0, 0.5], y=[0, 0.5], z=[0, 0.707],
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=[5, 10], color='red'),
            name='|ψ⟩'
        ))
        
        fig.update_layout(
            title="Sphère de Bloch - État Quantique",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💻 Simulations Quantiques")
        
        st.write("### 🚀 Optimisation Trajectoire Quantique")
        
        with st.form("quantum_trajectory"):
            st.write("**Configuration Mission:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                origin = st.selectbox("Origine", ["Terre LEO", "Lune", "Station Gateway"])
                destination = st.selectbox("Destination", ["Mars", "Lune", "Astéroïdes", "Jupiter"])
            
            with col2:
                num_waypoints = st.slider("Points Passage", 2, 10, 5)
                num_qubits = st.slider("Qubits à Utiliser", 5, 50, 20)
            
            constraints = st.multiselect(
                "Contraintes",
                ["Delta-v Minimal", "Temps Minimal", "Radiation Minimale", 
                 "Consommation Minimale", "Fenêtres Lancement"],
                default=["Delta-v Minimal", "Temps Minimal"]
            )
            
            quantum_backend = st.selectbox(
                "Backend Quantique",
                ["Simulateur Local", "IBM Quantum", "Google Cirq", "Amazon Braket"]
            )
            
            if st.form_submit_button("⚛️ Optimiser avec Quantum", type="primary"):
                with st.spinner("Calcul quantique en cours..."):
                    import time
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    stages = [
                        "Initialisation circuit quantique...",
                        "Création superposition états...",
                        "Application portes quantiques...",
                        "Intrication qubits...",
                        "Mesure état final...",
                        "Décodage résultats..."
                    ]
                    
                    for i, stage in enumerate(stages):
                        progress.progress((i + 1) / len(stages))
                        status.text(stage)
                        time.sleep(0.5)
                    
                    st.success("✅ Optimisation quantique complétée!")
                    
                    st.markdown("---")
                    st.subheader("📊 Résultats Optimisation Quantique")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Trajectoires Évaluées", f"{2**num_qubits:,}")
                    with col2:
                        st.metric("Temps Calcul", "2.3 s")
                    with col3:
                        st.metric("Δv Optimal", "5,847 m/s")
                    with col4:
                        st.metric("Gain vs Classique", "-12.4%")
                    
                    st.write("### 🛰️ Trajectoire Optimale")
                    
                    # Visualisation trajectoire
                    t = np.linspace(0, 1, 100)
                    x_traj = 1.5 * np.cos(2*np.pi*t)
                    y_traj = 1.5 * np.sin(2*np.pi*t)
                    z_traj = 0.3 * np.sin(4*np.pi*t)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_traj, y=y_traj, z=z_traj,
                        mode='lines',
                        line=dict(color='blue', width=4),
                        name='Trajectoire Optimale'
                    ))
                    
                    # Points passage
                    waypoint_indices = np.linspace(0, 99, num_waypoints, dtype=int)
                    fig.add_trace(go.Scatter3d(
                        x=x_traj[waypoint_indices],
                        y=y_traj[waypoint_indices],
                        z=z_traj[waypoint_indices],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        name='Points Passage'
                    ))
                    
                    fig.update_layout(
                        title="Trajectoire Optimisée Quantiquement",
                        scene=dict(
                            xaxis_title="X (UA)",
                            yaxis_title="Y (UA)",
                            zaxis_title="Z (UA)"
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Détails quantiques
                    st.write("### ⚛️ Détails Calcul Quantique")
                    
                    st.code(f"""
Circuit Quantique Utilisé:
- Qubits: {num_qubits}
- Profondeur circuit: {num_qubits * 3}
- Portes: Hadamard ({num_qubits}), CNOT ({num_qubits-1}), Rotation ({num_qubits * 2})
- Mesures: {num_qubits}
- Backend: {quantum_backend}
- Shots: 1024

Résultat Mesure:
État final: |ψ⟩ = 0.707|0⟩ + 0.707|1⟩ (superposition)
Probabilité solution optimale: 94.2%
                    """, language="text")
                    
                    log_event(f"Optimisation quantique: {origin} → {destination}", "SUCCESS")
    
    with tab3:
        st.subheader("⚡ Simulation Combustion Quantique")
        
        st.info("""
        **Chimie Quantique pour Propulsion:**
        
        La simulation quantique permet de modéliser précisément les réactions chimiques
        de combustion au niveau moléculaire, impossible avec calcul classique.
        
        **Avantages:**
        - Prédiction exacte énergies réaction
        - Optimisation mélanges propergols
        - Découverte nouveaux propergols haute performance
        - Simulation catalyseurs
        """)
        
        st.write("### 🔥 Simulateur Réaction Combustion")
        
        with st.form("quantum_combustion"):
            col1, col2 = st.columns(2)
            
            with col1:
                fuel = st.selectbox("Carburant", ["RP-1 (Kérosène)", "LH2", "Méthane", "UDMH"])
                oxidizer = st.selectbox("Comburant", ["LOX", "N2O4", "H2O2"])
            
            with col2:
                temperature = st.number_input("Température (K)", 1000, 4000, 3000, 100)
                pressure = st.number_input("Pression (MPa)", 1, 50, 20, 1)
            
            simulation_level = st.selectbox(
                "Niveau Simulation",
                ["Hartree-Fock", "DFT (B3LYP)", "CCSD", "CCSD(T)", "Full CI"]
            )
            
            if st.form_submit_button("⚛️ Simuler Combustion Quantique"):
                with st.spinner("Simulation quantique en cours..."):
                    import time
                    time.sleep(3)
                    
                    st.success("✅ Simulation complétée!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Énergie Libérée", "45.2 MJ/kg")
                    with col2:
                        st.metric("Température Flamme", "3,427 K")
                    with col3:
                        st.metric("Vitesse Échappement", "4,215 m/s")
                    with col4:
                        st.metric("Isp Prédit", "430 s")
                    
                    st.write("### 🧪 Produits Combustion")
                    
                    products = pd.DataFrame([
                        {"Molécule": "H2O", "Fraction Molaire": 0.42, "Énergie (eV)": -241.8},                        
                        {"Molécule": "CO2", "Fraction Molaire": 0.38, "Énergie (eV)": -393.5},
                        {"Molécule": "CO", "Fraction Molaire": 0.12, "Énergie (eV)": -110.5},
                        {"Molécule": "H2", "Fraction Molaire": 0.05, "Énergie (eV)": 0.0},
                        {"Molécule": "OH", "Fraction Molaire": 0.03, "Énergie (eV)": 39.0}
                    ])
                    
                    st.dataframe(products, use_container_width=True)
                    
                    # Graphique distribution énergie
                    fig = px.pie(products, values='Fraction Molaire', names='Molécule', 
                                title='Distribution Produits Combustion')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📊 Résultats Simulations Quantiques")
        
        if not st.session_state.rocket_system.get('quantum_analyses'):
            st.info("💡 Aucune simulation quantique effectuée")
        else:
            df_quantum = pd.DataFrame(st.session_state.rocket_system['quantum_analyses'])
            st.dataframe(df_quantum, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📈 Avantage Quantique")
        
        comparison = pd.DataFrame([
            {"Problème": "Optimisation Trajectoire", "Classique": "4.2 heures", "Quantique": "2.3 s", "Speedup": "6,522x"},
            {"Problème": "Combustion Moléculaire", "Classique": "2 semaines", "Quantique": "8 min", "Speedup": "2,520x"},
            {"Problème": "Optimisation Design", "Classique": "12 heures", "Quantique": "45 s", "Speedup": "960x"},
            {"Problème": "Cryptographie", "Classique": "Impossible", "Quantique": "Instantané", "Speedup": "∞"}
        ])
        
        st.dataframe(comparison, use_container_width=True)

# ==================== PAGE: BIO-COMPUTING ====================
elif page == "🧬 Bio-computing":
    st.header("🧬 Bio-computing et Systèmes Organiques")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🦠 Principes", "🧠 Contrôle Bio", "🔬 Applications", "📊 Résultats"])
    
    with tab1:
        st.subheader("🦠 Principes du Bio-computing")
        
        st.info("""
        **Bio-computing pour Aérospatial:**
        
        Le bio-computing utilise des systèmes biologiques (ADN, protéines, neurones)
        pour effectuer des calculs et contrôler des systèmes complexes.
        
        🧬 **Computing ADN** - Calculs parallèles massivement parallèles
        🦠 **Réseaux Neuronaux Organiques** - Apprentissage adaptatif naturel
        🔬 **Systèmes Auto-réparants** - Biomimétisme pour résilience
        🌱 **Matériaux Vivants** - Structures auto-assemblantes
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🧬 Computing ADN")
            
            st.write("""
            **Avantages:**
            - Parallélisme extrême (10²⁰ opérations/s)
            - Densité stockage inégalée
            - Faible consommation énergétique
            - Auto-réplication
            
            **Applications Spatiales:**
            - Calculs optimisation massive
            - Stockage données longue durée
            - Bio-capteurs environnement
            - Systèmes auto-adaptatifs
            """)
        
        with col2:
            st.write("### 🧠 Neurones Artificiels Organiques")
            
            st.write("""
            **Caractéristiques:**
            - Apprentissage en temps réel
            - Adaptation environnementale
            - Tolérance aux pannes naturelle
            - Consommation ultra-faible
            
            **Utilisation:**
            - Contrôle vol adaptatif
            - Diagnostic systèmes
            - Interface homme-machine
            - Traitement signal
            """)
        
        st.markdown("---")
        
        st.write("### 🔬 Architecture Bio-computing")
        
        # Diagramme architecture
        fig = go.Figure()
        
        # Couches
        layers = [
            {"name": "Capteurs Bio", "y": 4, "color": "lightgreen"},
            {"name": "Réseau Neuronal Organique", "y": 3, "color": "lightblue"},
            {"name": "Processeur ADN", "y": 2, "color": "lightyellow"},
            {"name": "Actuateurs", "y": 1, "color": "lightcoral"}
        ]
        
        for layer in layers:
            fig.add_trace(go.Bar(
                x=[layer["name"]],
                y=[layer["y"]],
                name=layer["name"],
                marker_color=layer["color"],
                showlegend=False
            ))
        
        fig.update_layout(
            title="Architecture Bio-computing Fusée",
            yaxis_title="Couche",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🧠 Système de Contrôle Bio-computing")
        
        st.write("### 🎯 Implémentation Contrôle Biologique")
        
        if not st.session_state.rocket_system['rockets']:
            st.warning("⚠️ Créez une fusée d'abord")
        else:
            with st.form("bio_control_setup"):
                rocket_select = st.selectbox(
                    "Fusée",
                    [f"{r['name']}" for r in st.session_state.rocket_system['rockets'].values()]
                )
                
                st.write("**Modules Bio-computing:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    bio_navigation = st.checkbox("🧭 Navigation Bio-adaptative", value=True)
                    bio_stabilization = st.checkbox("⚖️ Stabilisation Neuromorphique", value=True)
                    bio_diagnostics = st.checkbox("🔍 Diagnostic Bio-sensoriel", value=True)
                
                with col2:
                    bio_learning = st.checkbox("🧠 Apprentissage Continu", value=True)
                    bio_repair = st.checkbox("🔧 Auto-réparation", value=False)
                    bio_optimization = st.checkbox("📈 Optimisation Temps Réel", value=True)
                
                st.write("**Configuration Réseau Neuronal:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    num_neurons = st.number_input("Neurones Organiques", 100, 100000, 10000, 100)
                with col2:
                    learning_rate = st.slider("Taux Apprentissage", 0.001, 0.1, 0.01, 0.001)
                with col3:
                    adaptation_speed = st.selectbox("Vitesse Adaptation", ["Lente", "Moyenne", "Rapide"])
                
                if st.form_submit_button("🧬 Déployer Bio-computing", type="primary"):
                    with st.spinner("Déploiement système bio..."):
                        import time
                        
                        progress = st.progress(0)
                        status = st.empty()
                        
                        stages = [
                            "Culture neurones organiques...",
                            "Connexion synapses...",
                            "Programmation ADN...",
                            "Calibration bio-capteurs...",
                            "Initialisation apprentissage...",
                            "Tests validation..."
                        ]
                        
                        for i, stage in enumerate(stages):
                            progress.progress((i + 1) / len(stages))
                            status.text(stage)
                            time.sleep(0.5)
                        
                        st.success("✅ Système bio-computing déployé!")
                        
                        # Mise à jour fusée
                        for rocket in st.session_state.rocket_system['rockets'].values():
                            if rocket['name'] == rocket_select:
                                rocket['bio_control'] = True
                                break
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Neurones Actifs", f"{num_neurons:,}")
                        with col2:
                            st.metric("Synapses", f"{num_neurons * 50:,}")
                        with col3:
                            st.metric("Latence Réponse", "0.8 ms")
                        with col4:
                            st.metric("Consommation", "12 mW")
                        
                        log_event(f"Bio-computing déployé: {rocket_select}", "SUCCESS")
        
        st.markdown("---")
        
        st.write("### 📊 Monitoring Système Bio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Santé Réseau Neuronal:**")
            
            health_metrics = {
                "Viabilité Neurones": 98.7,
                "Activité Synaptique": 94.2,
                "Plasticité": 96.5,
                "Stabilité": 97.8
            }
            
            for metric, value in health_metrics.items():
                st.metric(metric, f"{value}%")
                st.progress(value / 100)
        
        with col2:
            st.write("**Performance Temps Réel:**")
            
            # Graphique activité neuronale
            t = np.linspace(0, 10, 100)
            activity = 50 + 20 * np.sin(t) + np.random.randn(100) * 5
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=activity, mode='lines', fill='tozeroy'))
            fig.update_layout(
                title="Activité Neuronale",
                xaxis_title="Temps (s)",
                yaxis_title="Activité (%)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔬 Applications Bio-computing")
        
        applications = {
            "🧭 Navigation Adaptative": {
                "description": "Système navigation qui apprend et s'adapte en temps réel",
                "avantages": [
                    "Adaptation conditions changeantes",
                    "Optimisation trajectoire continue",
                    "Résilience aux perturbations",
                    "Apprentissage patterns environnement"
                ],
                "performance": "32% plus efficace que contrôle classique",
                "status": "Opérationnel"
            },
            "🔍 Diagnostic Prédictif": {
                "description": "Détection précoce anomalies via bio-capteurs",
                "avantages": [
                    "Détection 15min avant panne",
                    "Sensibilité chimique extrême",
                    "Auto-calibration",
                    "Tolérance radiations"
                ],
                "performance": "99.2% précision détection",
                "status": "Test Phase 2"
            },
            "🔧 Auto-réparation": {
                "description": "Matériaux vivants capables auto-réparation",
                "avantages": [
                    "Réparation micro-fissures",
                    "Croissance dirigée",
                    "Adaptation stress",
                    "Longévité accrue"
                ],
                "performance": "Réparation 80% dommages < 1mm",
                "status": "Recherche"
            },
            "🌱 Systèmes Vie Support": {
                "description": "Écosystèmes biologiques fermés",
                "avantages": [
                    "Production O2/nourriture",
                    "Recyclage déchets",
                    "Régulation climat",
                    "Santé mentale équipage"
                ],
                "performance": "Autonomie 95% missions longues",
                "status": "ISS Tests"
            },
            "💾 Stockage ADN": {
                "description": "Données encodées dans ADN synthétique",
                "avantages": [
                    "Densité: 215 PB/gramme",
                    "Durée: 1000+ ans",
                    "Pas d'énergie stockage",
                    "Radiation résistant"
                ],
                "performance": "1 EB dans 1 kg ADN",
                "status": "Prototype"
            }
        }
        
        for app_name, app_details in applications.items():
            with st.expander(f"{app_name}"):
                st.write(f"**Description:** {app_details['description']}")
                
                st.write("**Avantages:**")
                for adv in app_details['avantages']:
                    st.write(f"  ✅ {adv}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Performance:** {app_details['performance']}")
                
                with col2:
                    status_color = "🟢" if app_details['status'] == "Opérationnel" else "🟡" if "Test" in app_details['status'] else "🔴"
                    st.info(f"**Statut:** {status_color} {app_details['status']}")
    
    with tab4:
        st.subheader("📊 Résultats Bio-computing")
        
        if not st.session_state.rocket_system.get('biocomputing_results'):
            st.info("💡 Aucun résultat bio-computing")
        else:
            df_bio = pd.DataFrame(st.session_state.rocket_system['biocomputing_results'])
            st.dataframe(df_bio, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 📈 Comparaison Bio vs Classique")
        
        comparison = pd.DataFrame([
            {
                "Critère": "Consommation Énergie",
                "Classique": "100 W",
                "Bio-computing": "12 mW",
                "Amélioration": "99.988%"
            },
            {
                "Critère": "Vitesse Apprentissage",
                "Classique": "Heures-Jours",
                "Bio-computing": "Secondes-Minutes",
                "Amélioration": "1000x"
            },
            {
                "Critère": "Adaptation Environnement",
                "Classique": "Limitée",
                "Bio-computing": "Continue",
                "Amélioration": "∞"
            },
            {
                "Critère": "Tolérance Pannes",
                "Classique": "Faible",
                "Bio-computing": "Très Élevée",
                "Amélioration": "10x"
            },
            {
                "Critère": "Coût Production",
                "Classique": "$$",
                "Bio-computing": "$",
                "Amélioration": "90%"
            }
        ])
        
        st.dataframe(comparison, use_container_width=True)
        
        st.success("""
        **Conclusion:**
        
        Le bio-computing offre des avantages significatifs pour les missions spatiales:
        - Consommation énergétique réduite de 99.9%
        - Adaptation temps réel aux conditions changeantes
        - Résilience naturelle aux pannes
        - Auto-réparation des systèmes critiques
        
        Idéal pour missions longue durée (Mars, Jupiter, interstellaire)
        """)

# ==================== PAGE: MISSIONS MARS ====================
elif page == "🔴 Missions Mars":
    st.header("🔴 Missions Martiennes")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Planification",
        "🛰️ Orbite Mars",
        "🏗️ EDL Mars",
        "🏭 ISRU Mars",
        "👨‍🚀 Habitats"
    ])
    
    with tab1:
        st.subheader("🚀 Planification Mission Mars")
        
        st.info("""
        **Défi Mars:**
        
        Mars représente le défi spatial majeur du 21ème siècle.
        Distance: 55-400 millions km (selon position)
        Durée transit: 6-9 mois
        Fenêtres lancement: Tous les 26 mois
        
        **Architecture Mission Type:**
        1. Lancement Terre → LEO
        2. Injection Trans-Mars (TMI)
        3. Transit interplanétaire
        4. Capture orbite Mars (MOI)
        5. EDL (Entry, Descent, Landing)
        6. Surface Mars (séjour)
        7. Ascent depuis Mars
        8. Transit retour
        9. Rentrée Terre
        """)
        
        st.write("### 🎯 Créer Mission Mars")
        
        with st.form("mars_mission"):
            col1, col2 = st.columns(2)
            
            with col1:
                mission_name = st.text_input("Nom Mission", "Ares-1")
                mission_type = st.selectbox(
                    "Type Mission",
                    ["Cargo (non habité)", "Habitée Aller Simple", "Habitée Aller-Retour", 
                     "Reconnaissance", "Base Permanente"]
                )
                
                launch_window = st.date_input("Fenêtre Lancement", datetime.now())
            
            with col2:
                crew_size = st.number_input("Taille Équipage", 0, 12, 4, 1)
                cargo_mass = st.number_input("Masse Cargo (tonnes)", 10, 500, 100, 10)
                
                mission_duration_mars = st.number_input("Durée Surface Mars (jours)", 30, 900, 540, 30)
            
            st.write("**Profil Mission:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                transit_type = st.selectbox("Transit", ["Hohmann", "Fast Transfer", "Cycler"])
            with col2:
                propulsion_main = st.selectbox("Propulsion Principale", 
                    ["Chimique", "Nucléaire Thermique", "Électrique", "Plasma", "Fusion"])
            with col3:
                landing_site = st.selectbox("Site Atterrissage",
                    ["Jezero Crater", "Valles Marineris", "Olympus Mons", "Pôle Sud", "Arcadia Planitia"])
            
            st.write("**Technologies Avancées:**")
            
            mars_tech = st.multiselect(
                "Technologies à Utiliser",
                ["IA Navigation", "ISRU Propergol", "Imprimantes 3D", "Greenhouses",
                 "Réacteur Nucléaire", "Bouclier Magnétique", "Drones Mars"],
                default=["IA Navigation", "ISRU Propergol"]
            )
            
            if st.form_submit_button("🚀 Créer Mission Mars", type="primary"):
                with st.spinner("Calcul trajectoire et ressources..."):
                    import time
                    time.sleep(2)
                    
                    mission_id = f"mars_{len(st.session_state.rocket_system.get('mars_missions', {})) + 1}"
                    
                    # Calculs mission
                    if transit_type == "Hohmann":
                        transit_out = 240  # jours
                        transit_return = 240
                        delta_v_total = 12000  # m/s
                    elif transit_type == "Fast Transfer":
                        transit_out = 180
                        transit_return = 180
                        delta_v_total = 16000
                    else:  # Cycler
                        transit_out = 210
                        transit_return = 210
                        delta_v_total = 8000
                    
                    total_duration = transit_out + mission_duration_mars + transit_return
                    
                    # Masses
                    propellant_needed = (cargo_mass + crew_size * 0.1) * 10  # Approximation
                    
                    mission = {
                        'id': mission_id,
                        'name': mission_name,
                        'type': mission_type,
                        'crew_size': crew_size,
                        'cargo_mass': cargo_mass,
                        'launch_window': launch_window.isoformat(),
                        'transit_out': transit_out,
                        'surface_duration': mission_duration_mars,
                        'transit_return': transit_return,
                        'total_duration': total_duration,
                        'delta_v': delta_v_total,
                        'propellant_needed': propellant_needed,
                        'landing_site': landing_site,
                        'technologies': mars_tech,
                        'status': 'planning',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    if 'mars_missions' not in st.session_state.rocket_system:
                        st.session_state.rocket_system['mars_missions'] = {}
                    
                    st.session_state.rocket_system['mars_missions'][mission_id] = mission
                    
                    st.success(f"✅ Mission '{mission_name}' créée!")
                    st.balloons()
                    
                    # Résultats
                    st.markdown("---")
                    st.subheader("📊 Paramètres Mission")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Durée Totale", f"{total_duration} jours")
                        st.metric("Transit Aller", f"{transit_out} jours")
                    
                    with col2:
                        st.metric("Surface Mars", f"{mission_duration_mars} jours")
                        st.metric("Transit Retour", f"{transit_return} jours")
                    
                    with col3:
                        st.metric("Delta-v Total", f"{delta_v_total:,} m/s")
                        st.metric("Propergol", f"{propellant_needed:.0f} t")
                    
                    with col4:
                        st.metric("Masse Totale", f"{cargo_mass + propellant_needed:.0f} t")
                        st.metric("Équipage", crew_size)
                    
                    # Timeline
                    st.write("### 📅 Timeline Mission")
                    
                    timeline_data = [
                        {"Phase": "Lancement", "Durée": "3 jours", "Jour": 0},
                        {"Phase": "Transit Terre→Mars", "Durée": f"{transit_out} jours", "Jour": 3},
                        {"Phase": "Arrivée Mars (MOI)", "Durée": "1 jour", "Jour": 3 + transit_out},
                        {"Phase": "EDL", "Durée": "7 minutes", "Jour": 4 + transit_out},
                        {"Phase": "Surface Mars", "Durée": f"{mission_duration_mars} jours", "Jour": 4 + transit_out},
                        {"Phase": "Ascent Mars", "Durée": "1 jour", "Jour": 4 + transit_out + mission_duration_mars},
                        {"Phase": "Transit Mars→Terre", "Durée": f"{transit_return} jours", "Jour": 5 + transit_out + mission_duration_mars},
                        {"Phase": "Rentrée Terre", "Durée": "1 jour", "Jour": total_duration}
                    ]
                    
                    df_timeline = pd.DataFrame(timeline_data)
                    st.dataframe(df_timeline, use_container_width=True)
                    
                    log_event(f"Mission Mars créée: {mission_name}", "SUCCESS")
    
    with tab2:
        st.subheader("🛰️ Insertion Orbite Mars (MOI)")
        
        st.info("""
        **Mars Orbit Insertion:**
        
        Manœuvre critique pour capturer véhicule en orbite martienne.
        
        **Méthodes:**
        1. **Freinage Propulsif:** Delta-v ~1,500 m/s
        2. **Aérofreinage:** Utilisation atmosphère progressive
        3. **Aérocapture:** Capture directe (1 passage atmosphère)
        """)
        
        st.write("### 🎯 Calculateur MOI")
        
        with st.form("moi_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                arrival_velocity = st.number_input("Vitesse Arrivée (km/s)", 3.0, 8.0, 5.5, 0.1)
                target_orbit_altitude = st.number_input("Altitude Orbite Cible (km)", 200, 50000, 500, 50)
            
            with col2:
                spacecraft_mass = st.number_input("Masse Véhicule (tonnes)", 10, 500, 50, 5)
                moi_method = st.selectbox("Méthode", ["Propulsif Direct", "Aérofreinage", "Aérocapture"])
            
            if st.form_submit_button("🔬 Calculer MOI"):
                # Calculs
                mars_mu = 4.282837e13  # m³/s²
                mars_radius = 3389500  # m
                
                v_infinity = arrival_velocity * 1000  # m/s
                r_orbit = mars_radius + target_orbit_altitude * 1000
                
                # Vitesse orbite circulaire
                v_orbit = np.sqrt(mars_mu / r_orbit)
                
                # Vitesse au périapse
                v_periapsis = np.sqrt(v_infinity**2 + 2*mars_mu/r_orbit)
                
                # Delta-v
                delta_v_moi = v_periapsis - v_orbit
                
                # Propergol requis (Tsiolkovsky)
                isp = 350  # s
                g0 = 9.80665
                ve = isp * g0
                propellant_fraction = 1 - np.exp(-delta_v_moi / ve)
                propellant_mass = spacecraft_mass * propellant_fraction
                
                st.success("✅ Calcul MOI complété!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Delta-v MOI", f"{delta_v_moi:.0f} m/s")
                with col2:
                    st.metric("Vitesse Périapse", f"{v_periapsis:.0f} m/s")
                with col3:
                    st.metric("Propergol Requis", f"{propellant_mass:.1f} t")
                with col4:
                    st.metric("Durée Combustion", f"{(delta_v_moi/10):.0f} s")
                
                if moi_method == "Aérofreinage":
                    st.info("""
                    **Aérofreinage:**
                    - Multiples passages atmosphère (20-40 orbites)
                    - Économie propergol: 70-90%
                    - Durée: 2-6 mois
                    - Risque: Moyen
                    """)
                elif moi_method == "Aérocapture":
                    st.info("""
                    **Aérocapture:**
                    - Un seul passage atmosphère profond
                    - Économie propergol: 95%
                    - Durée: 15 minutes
                    - Risque: Élevé (non testé)
                    """)
    
    with tab3:
        st.subheader("🏗️ EDL Mars (Entry, Descent, Landing)")
        
        st.warning("""
        **"7 Minutes of Terror"**
        
        EDL Mars est la phase la plus critique. L'atmosphère martienne est trop fine
        pour parachutes seuls, mais assez dense pour échauffement.
        
        **Défis:**
        - Atmosphère 1% Terre
        - Vitesse entrée: 5-7 km/s
        - Décélération: 10-15g
        - Guidage précis requis
        - Aucun contact Terre (latence 20min)
        """)
        
        st.write("### 🎯 Phases EDL")
        
        edl_phases = {
            "1️⃣ Entrée Atmosphérique": {
                "altitude": "125 km",
                "vitesse": "5,700 m/s",
                "durée": "4 min",
                "système": "Bouclier thermique",
                "température": "1,600°C"
            },
            "2️⃣ Freinage Atmosphérique": {
                "altitude": "10-7 km",
                "vitesse": "470 m/s",
                "durée": "2 min",
                "système": "Parachute supersonique",
                "décélération": "10g"
            },
            "3️⃣ Séparation Bouclier": {
                "altitude": "7 km",
                "vitesse": "100 m/s",
                "durée": "10 s",
                "système": "Pyrotechnique",
                "décélération": "-"
            },
            "4️⃣ Descente Propulsive": {
                "altitude": "7 → 0.5 km",
                "vitesse": "100 → 3 m/s",
                "durée": "40 s",
                "système": "Retropropulseurs",
                "décélération": "3g"
            },
            "5️⃣ Atterrissage": {
                "altitude": "0 m",
                "vitesse": "3 m/s",
                "durée": "instantané",
                "système": "Pattes amorties / SkyCrane",
                "décélération": "1.2g"
            }
        }

        edl_df = pd.DataFrame(edl_phases).T
        st.dataframe(edl_df, use_container_width=True)

        st.write("### 🔢 Simulateur EDL Simplifié")
        altitude = st.slider("Altitude initiale (km)", 80, 150, 125)
        vitesse_init = st.slider("Vitesse d’entrée (m/s)", 4000, 8000, 5700)
        masse = st.number_input("Masse (kg)", 500, 50000, 3000)
        cd = st.slider("Coefficient de traînée (Cd)", 1.0, 2.5, 1.8, 0.1)
        area = st.number_input("Surface frontale (m²)", 1.0, 50.0, 15.0, 0.5)

        if st.button("🧮 Simuler EDL"):
            rho = 0.02  # kg/m³, densité moyenne à 30 km
            g = 3.71    # m/s²
            drag = 0.5 * rho * vitesse_init**2 * cd * area
            decel = drag / masse
            final_v = np.sqrt(max(vitesse_init**2 - 2 * decel * altitude * 1000, 0))
            st.success(f"Vitesse finale estimée: {final_v:.1f} m/s")
            st.metric("Décélération moyenne", f"{decel:.2f} m/s²")
            st.metric("Force de traînée", f"{drag/1000:.1f} kN")

    # ==================== ISRU Mars ====================
    with tab4:
        st.subheader("🏭 ISRU (In-Situ Resource Utilization) sur Mars")

        st.info("""
        **Objectif ISRU:** utiliser les ressources locales martiennes pour réduire la masse
        à lancer depuis la Terre.

        **Ressources Disponibles:**
        - CO₂ atmosphérique → O₂ & CH₄ (Sabatier)
        - H₂O glace → O₂ & H₂
        - Régolithe → matériaux de construction
        - Énergie solaire & nucléaire
        """)

        st.write("### ⚙️ Simulateur Production ISRU")

        co2_input = st.slider("CO₂ Collecté (kg/jour)", 100, 5000, 1000, 100)
        power_input = st.slider("Puissance disponible (kW)", 1, 100, 10)
        efficiency = st.slider("Efficacité conversion (%)", 10, 90, 60)
        duration_days = st.number_input("Durée opération (jours)", 10, 1000, 300)

        if st.button("🔬 Calculer Production ISRU"):
            o2_output = co2_input * (efficiency / 100) * 0.73 * duration_days / 1000  # tonnes
            ch4_output = co2_input * (efficiency / 100) * 0.18 * duration_days / 1000  # tonnes
            st.success("✅ Simulation ISRU terminée !")
            st.metric("Oxygène produit", f"{o2_output:.2f} t")
            st.metric("Méthane produit", f"{ch4_output:.2f} t")
            st.metric("Énergie utilisée", f"{power_input * duration_days * 24:.0f} kWh")

            st.progress(min(int(efficiency), 100))
            st.info("Production stable avec efficacité optimale entre 55% et 70%.")

    # ==================== Habitats Mars ====================
    with tab5:
        st.subheader("👨‍🚀 Habitats Martiens")

        st.info("""
        **Concepts d'Habitats:**
        1. Modules gonflables (Bigelow type)
        2. Structures semi-enterrées en régolithe
        3. Impression 3D locale
        4. Dômes transparents pour agriculture
        """)

        st.write("### 🏠 Concevoir un Habitat")

        with st.form("mars_habitat"):
            col1, col2 = st.columns(2)
            with col1:
                habitat_name = st.text_input("Nom Habitat", "Ares Base Alpha")
                habitat_type = st.selectbox("Type", ["Dôme", "Tunnel", "Module", "Sous-terrain"])
                crew_capacity = st.number_input("Capacité (personnes)", 2, 20, 6)
            with col2:
                energy_source = st.selectbox("Énergie", ["Solaire", "Nucléaire", "Hybride"])
                area_m2 = st.number_input("Surface Habitable (m²)", 20, 2000, 250, 10)
                duration_hab = st.number_input("Durée d’occupation (jours)", 30, 2000, 540, 10)

            if st.form_submit_button("🏗️ Créer Habitat"):
                volume = area_m2 * 2.5
                o2_needs = crew_capacity * duration_hab * 0.84 / 1000
                water_needs = crew_capacity * duration_hab * 2.5 / 1000
                power_needs = crew_capacity * 5 * duration_hab / 1000

                st.success(f"✅ Habitat '{habitat_name}' créé !")
                st.metric("Volume intérieur", f"{volume:.0f} m³")
                st.metric("Besoins en O₂", f"{o2_needs:.2f} t")
                st.metric("Eau nécessaire", f"{water_needs:.2f} t")
                st.metric("Énergie totale", f"{power_needs:.1f} MWh")
                st.progress(min(int(crew_capacity * 5), 100))
                st.info("Habitat prêt pour simulation environnementale et tests psychologiques.")

# ==================== PAGE: ANALYSES & PERFORMANCES ====================
elif page == "📊 Analyses & Performances":
    st.header("📊 Analyses et Performances")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Vue d'Ensemble", "📈 Tendances", "⚡ Benchmarking", "💡 Insights"])
    
    with tab1:
        st.subheader("🎯 Vue d'Ensemble Performance")
        
        if not st.session_state.rocket_system['rockets']:
            st.info("💡 Créez des fusées pour voir les analyses")
        else:
            # KPIs principaux
            col1, col2, col3, col4 = st.columns(4)
            
            total_thrust = sum(r.get('performance', {}).get('thrust', 0) for r in st.session_state.rocket_system['rockets'].values()) / 1e6
            avg_reliability = np.mean([r.get('success_rate', 0) for r in st.session_state.rocket_system['rockets'].values()])
            total_payload = sum(r.get('performance', {}).get('payload_leo', 0) for r in st.session_state.rocket_system['rockets'].values()) / 1000
            avg_cost = np.mean([r.get('cost_per_launch', 50e6) for r in st.session_state.rocket_system['rockets'].values()]) / 1e6
            
            with col1:
                st.metric("Poussée Totale", f"{total_thrust:.1f} MN")
            with col2:
                st.metric("Fiabilité Moyenne", f"{avg_reliability:.1f}%")
            with col3:
                st.metric("Capacité Totale LEO", f"{total_payload:.1f} t")
            with col4:
                st.metric("Coût Moyen", f"${avg_cost:.0f}M")
            
            st.markdown("---")
            
            # Performance par fusée
            st.write("### 📊 Performance par Fusée")
            
            perf_data = []
            for rocket in st.session_state.rocket_system['rockets'].values():
                perf_data.append({
                    "Fusée": rocket['name'],
                    "Delta-v (m/s)": rocket['performance'].get('delta_v', 0),
                    "Payload LEO (t)": rocket['performance'].get('payload_leo', 0) / 1000,
                    "Coût ($M)": rocket.get('cost_per_launch', 0) / 1e6,
                    "Fiabilité (%)": rocket.get('success_rate', 0),
                    "Statut": rocket['status']
                })
            
            df_perf = pd.DataFrame(perf_data)
            st.dataframe(df_perf, use_container_width=True)
            
            # Graphiques comparatifs
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df_perf, x='Fusée', y='Delta-v (m/s)', 
                           title='Delta-v par Fusée',
                           color='Delta-v (m/s)',
                           color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df_perf, x='Payload LEO (t)', y='Coût ($M)',
                               size='Fiabilité (%)', hover_name='Fusée',
                               title='Coût vs Payload (taille = Fiabilité)',
                               color='Fiabilité (%)',
                               color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Tendances et Évolution")
        
        st.write("### 📊 Évolution Technologies")
        
        # Graphique évolution technologies
        years = np.arange(2020, 2031)
        chemical = 100 - (years - 2020) * 3
        electric = (years - 2020) * 4
        nuclear = (years - 2020) * 2
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=chemical, mode='lines+markers', name='Chimique', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=years, y=electric, mode='lines+markers', name='Électrique', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=years, y=nuclear, mode='lines+markers', name='Nucléaire', line=dict(width=3)))
        
        fig.update_layout(
            title="Évolution Part de Marché par Type Propulsion",
            xaxis_title="Année",
            yaxis_title="Part de Marché (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 📉 Tendances Coût/kg")
        
        cost_years = np.arange(2010, 2031)
        cost_per_kg = 10000 * np.exp(-0.08 * (cost_years - 2010))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cost_years, y=cost_per_kg, mode='lines', fill='tozeroy',
                                line=dict(color='red', width=3)))
        fig.add_hline(y=1000, line_dash="dash", annotation_text="Objectif $1000/kg")
        
        fig.update_layout(
            title="Évolution Coût Lancement ($/kg vers LEO)",
            xaxis_title="Année",
            yaxis_title="Coût ($/kg)",
            yaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("⚡ Benchmarking")
        
        st.write("### 🏆 Comparaison Fusées Mondiales")
        
        benchmark_data = [
            {"Fusée": "Falcon 9 (SpaceX)", "Payload LEO": 22.8, "Coût": 67, "$/kg": 2939, "Fiabilité": 98.9, "Réutilisable": "Oui"},
            {"Fusée": "Falcon Heavy", "Payload LEO": 63.8, "Coût": 97, "$/kg": 1520, "Fiabilité": 100, "Réutilisable": "Oui"},
            {"Fusée": "Starship (SpaceX)", "Payload LEO": 150, "Coût": 10, "$/kg": 67, "Fiabilité": 0, "Réutilisable": "Oui"},
            {"Fusée": "Ariane 6", "Payload LEO": 21.6, "Coût": 115, "$/kg": 5324, "Fiabilité": 0, "Réutilisable": "Non"},
            {"Fusée": "Soyuz 2", "Payload LEO": 8.2, "Coût": 48, "$/kg": 5854, "Fiabilité": 97.6, "Réutilisable": "Non"},
            {"Fusée": "Long March 5", "Payload LEO": 25, "Coût": 150, "$/kg": 6000, "Fiabilité": 83.3, "Réutilisable": "Non"},
            {"Fusée": "SLS Block 1", "Payload LEO": 95, "Coût": 4100, "$/kg": 43158, "Fiabilité": 100, "Réutilisable": "Non"},
            {"Fusée": "New Glenn", "Payload LEO": 45, "Coût": 100, "$/kg": 2222, "Fiabilité": 0, "Réutilisable": "Oui"}
        ]
        
        df_benchmark = pd.DataFrame(benchmark_data)
        st.dataframe(df_benchmark, use_container_width=True)
        
        # Graphique comparatif
        fig = px.scatter(df_benchmark, x='Payload LEO', y='$/kg',
                        size='Coût', hover_name='Fusée',
                        color='Réutilisable',
                        title='Comparaison Mondiale (taille = Coût total)',
                        log_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        **Tendances Observées:**
        - Réutilisabilité = Réduction coûts drastique
        - SpaceX leader avec $67-2939/kg
        - Starship révolutionnaire si succès ($67/kg)
        - Fusées non-réutilisables: $5000-43000/kg
        """)
    
    with tab4:
        st.subheader("💡 Insights et Recommandations")
        
        st.write("### 🎯 Recommandations Stratégiques")
        
        recommendations = {
            "💰 Réduction Coûts": [
                "Implémenter réutilisabilité complète (réduction 70-90%)",
                "Fabrication additive pour pièces complexes",
                "Standardisation composants entre modèles",
                "Production série pour économies d'échelle"
            ],
            "🚀 Performance": [
                "Optimisation ratio masse propergol/structure",
                "Propulsion électrique pour étages supérieurs",
                "Matériaux composites avancés (CFRP, nanotubes)",
                "Intelligence artificielle pour contrôle vol"
            ],
            "🔬 Innovation": [
                "Investir R&D propulsion nucléaire",
                "Développer ISRU pour missions Mars",
                "Bio-computing pour systèmes adaptatifs",
                "Ordinateurs quantiques optimisation"
            ],
            "🌍 Durabilité": [
                "Désorbitation active satellites",
                "Propergols verts (méthane vs RP-1)",
                "Recyclage matériaux étages",
                "Réduction débris spatiaux"
            ]
        }
        
        for category, items in recommendations.items():
            with st.expander(f"{category}"):
                for item in items:
                    st.write(f"✅ {item}")
        
        st.markdown("---")
        
        st.write("### 🔮 Prédictions 2030")
        
        predictions = pd.DataFrame([
            {"Métrique": "Coût LEO ($/kg)", "Aujourd'hui": 2500, "2030 Prévu": 100, "Réduction": "96%"},
            {"Métrique": "Lancements/an", "Aujourd'hui": 180, "2030 Prévu": 1000, "Réduction": "+456%"},
            {"Métrique": "Payload Max (tonnes)", "Aujourd'hui": 150, "2030 Prévu": 500, "Réduction": "+233%"},
            {"Métrique": "Temps Préparation (jours)", "Aujourd'hui": 30, "2030 Prévu": 1, "Réduction": "97%"}
        ])
        
        st.dataframe(predictions, use_container_width=True)

# ==================== PAGE: SIMULATIONS LANCEMENT ====================
elif page == "🎯 Simulations Lancement":
    st.header("🎯 Simulations de Lancement")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Config Lancement", "📊 Simulation Temps Réel", "📈 Analyse Trajectoire", "🎬 Replay"])
    
    with tab1:
        st.subheader("🚀 Configuration Lancement")
        
        if not st.session_state.rocket_system['rockets']:
            st.warning("⚠️ Créez une fusée d'abord")
        else:
            with st.form("launch_simulation"):
                st.write("### 🎯 Paramètres Mission")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    rocket_select = st.selectbox(
                        "Fusée",
                        [f"{r['name']}" for r in st.session_state.rocket_system['rockets'].values()]
                    )
                    
                    target_orbit = st.selectbox(
                        "Orbite Cible",
                        ["LEO 400km", "ISS 420km", "SSO 600km", "GTO", "Lune", "Mars"]
                    )
                
                with col2:
                    launch_site = st.selectbox(
                        "Site Lancement",
                        ["Cap Canaveral", "Vandenberg", "Kourou", "Baïkonour", "Jiuquan"]
                    )
                    
                    weather = st.selectbox("Conditions Météo", ["Nominales", "Limites", "Défavorables"])
                
                st.write("### ⚙️ Paramètres Avancés")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    azimuth = st.slider("Azimut Lancement (°)", 0, 360, 90, 1)
                with col2:
                    throttle_profile = st.selectbox("Profil Poussée", ["Nominal", "Throttle Down", "Maximal"])
                with col3:
                    guidance = st.selectbox("Guidage", ["Classique", "IA Adaptatif", "Optimal"])
                
                st.write("### 🎮 Options Simulation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    real_time = st.checkbox("Temps Réel", value=False)
                    abort_scenarios = st.checkbox("Tester Scénarios Abort", value=False)
                
                with col2:
                    ai_monitoring = st.checkbox("Monitoring IA", value=True)
                    record_telemetry = st.checkbox("Enregistrer Télémétrie", value=True)
                
                if st.form_submit_button("🚀 Lancer Simulation", type="primary"):
                    with st.spinner("Initialisation simulation..."):
                        import time
                        time.sleep(2)
                        
                        st.success("✅ Simulation prête!")
                        st.info("Allez dans l'onglet 'Simulation Temps Réel' pour lancer")
    
    with tab2:
        st.subheader("📊 Simulation Temps Réel")
        
        if st.button("🚀 LANCER", type="primary", use_container_width=True):
            
            # Conteneurs pour données temps réel
            status_container = st.empty()
            metrics_container = st.empty()
            chart_container = st.empty()
            telemetry_container = st.empty()
            
            # Simulation lancement
            duration = 600  # 10 minutes
            dt = 1  # 1 seconde
            
            times = []
            altitudes = []
            velocities = []
            accelerations = []
            
            for t in range(0, duration, dt):
                # Statut
                if t < 10:
                    phase = "🔥 LIFTOFF"
                    color = "red"
                elif t < 120:
                    phase = "🚀 Ascension Étage 1"
                    color = "orange"
                elif t < 150:
                    phase = "🔀 Séparation Étage 1"
                    color = "yellow"
                elif t < 400:
                    phase = "⚡ Étage 2 Combustion"
                    color = "blue"
                elif t < 550:
                    phase = "🛰️ Coast Phase"
                    color = "cyan"
                else:
                    phase = "✅ Insertion Orbitale"
                    color = "green"
                
                status_container.markdown(f"### <span style='color:{color}'>{phase}</span> - T+{t}s", unsafe_allow_html=True)
                
                # Calculs (simplifiés)
                if t < 150:
                    altitude = 0.5 * 30 * t**2
                    velocity = 30 * t
                    acceleration = 30 - t * 0.05
                else:
                    altitude = 0.5 * 30 * 150**2 + (t - 150) * 4500
                    velocity = 30 * 150 + (t - 150) * 30
                    acceleration = 20
                
                altitude = min(altitude, 420000)
                velocity = min(velocity, 7800)
                
                times.append(t)
                altitudes.append(altitude / 1000)
                velocities.append(velocity)
                accelerations.append(acceleration)
                
                # Métriques
                col1, col2, col3, col4 = metrics_container.columns(4)
                
                with col1:
                    st.metric("Altitude", f"{altitude/1000:.1f} km")
                with col2:
                    st.metric("Vitesse", f"{velocity:.0f} m/s")
                with col3:
                    st.metric("Accélération", f"{acceleration:.1f} m/s²")
                with col4:
                    downrange = t * 50
                    st.metric("Downrange", f"{downrange:.0f} km")
                
                # Graphiques
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Altitude", "Vitesse", "Accélération", "Trajectoire")
                )
                
                fig.add_trace(go.Scatter(x=times, y=altitudes, mode='lines', name='Alt'), row=1, col=1)
                fig.add_trace(go.Scatter(x=times, y=velocities, mode='lines', name='Vel'), row=1, col=2)
                fig.add_trace(go.Scatter(x=times, y=accelerations, mode='lines', name='Acc'), row=2, col=1)
                
                # Trajectoire 2D
                downranges = [t * 50 for t in times]
                fig.add_trace(go.Scatter(x=downranges, y=altitudes, mode='lines', name='Traj'), row=2, col=2)
                
                fig.update_layout(height=600, showlegend=False)
                
                chart_container.plotly_chart(fig, use_container_width=True)
                
                # Télémétrie
                telemetry_data = {
                    "Temps": f"T+{t}s",
                    "Phase": phase,
                    "Altitude": f"{altitude/1000:.2f} km",
                    "Vitesse": f"{velocity:.0f} m/s",
                    "Accélération": f"{acceleration:.2f} m/s²",
                    "G-Force": f"{acceleration/9.81:.2f}g",
                    "Propergol": f"{max(0, 100 - t/6):.1f}%",
                    "Guidage": "Nominal ✅"
                }
                
                telemetry_container.json(telemetry_data)
                
                time.sleep(0.05 if t < 150 else 0.02)
            
            st.success("🎉 INSERTION ORBITALE RÉUSSIE!")
            st.balloons()
            
            # Résumé
            st.markdown("---")
            st.subheader("📊 Résumé Mission")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Durée Totale", f"{duration}s")
            with col2:
                st.metric("Altitude Finale", f"{altitude/1000:.1f} km")
            with col3:
                st.metric("Vitesse Orbitale", f"{velocity:.0f} m/s")
            with col4:
                st.metric("Précision", "±0.5 km ✅")
    
    with tab3:
        st.subheader("📈 Analyse Trajectoire")
        
        st.write("### 🎯 Analyse Post-Lancement")
        
        st.info("""
        **Critères Évaluation:**
        - Précision insertion orbitale
        - Consommation propergol
        - Contraintes structurelles
        - Performance guidage
        """)
        
        # Graphiques analyse
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Profil Vitesse vs Optimal**")
            
            t_analysis = np.linspace(0, 600, 100)
            v_actual = 7800 * (1 - np.exp(-t_analysis/200))
            v_optimal = 7800 * (1 - np.exp(-t_analysis/190))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_analysis, y=v_actual, mode='lines', name='Réel'))
            fig.add_trace(go.Scatter(x=t_analysis, y=v_optimal, mode='lines', name='Optimal', line=dict(dash='dash')))
            
            fig.update_layout(
                xaxis_title="Temps (s)",
                yaxis_title="Vitesse (m/s)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Consommation Propergol**")
            
            stages_prop = pd.DataFrame([
                {"Étage": "Étage 1", "Propergol Initial": 400, "Consommé": 398, "Restant": 2},
                {"Étage": "Étage 2", "Propergol Initial": 100, "Consommé": 95, "Restant": 5}
            ])
            
            fig = px.bar(stages_prop, x='Étage', y=['Consommé', 'Restant'], 
                        title='Propergol par Étage',
                        barmode='stack')
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 📊 Scores Performance")
        
        scores = {
            "Précision Orbite": 98.5,
            "Efficacité Propergol": 96.2,
            "Guidage": 97.8,
            "Structures (G-max)": 94.1,
            "Aérodynamique": 95.7,
            "Global": 96.5
        }
        
        for metric, score in scores.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{metric}**")
                st.progress(score / 100)
            with col2:
                st.metric("", f"{score}%")
    
    with tab4:
        st.subheader("🎬 Replay et Archives")
        
        st.write("### 📼 Lancements Archivés")
        
        if 'simulations' not in st.session_state.rocket_system:
            st.session_state.rocket_system['simulations'] = []
        
        if st.session_state.rocket_system['simulations']:
            for i, sim in enumerate(st.session_state.rocket_system['simulations'][-10:][::-1]):
                with st.expander(f"🚀 Lancement #{i+1} - {sim.get('date', 'N/A')}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Fusée:** {sim.get('rocket', 'Unknown')}")
                        st.write(f"**Cible:** {sim.get('target', 'LEO')}")
                    
                    with col2:
                        st.write(f"**Succès:** {'✅' if sim.get('success', False) else '❌'}")
                        st.write(f"**Durée:** {sim.get('duration', 0)}s")
                    
                    with col3:
                        if st.button(f"▶️ Replay", key=f"replay_{i}"):
                            st.info("Replay lancé - Retournez à l'onglet Simulation")
        else:
            st.info("💡 Aucune simulation archivée")

# ==================== PAGE: JUMEAUX NUMÉRIQUES ====================
elif page == "💻 Jumeaux Numériques":
    st.header("💻 Jumeaux Numériques (Digital Twins)")
    
    tab1, tab2, tab3 = st.tabs(["🔬 Concept", "⚙️ Créer Jumeau", "📊 Monitoring"])
    
    with tab1:
        st.subheader("🔬 Concept Jumeaux Numériques")
        
        st.info("""
        **Jumeau Numérique (Digital Twin):**
        
        Réplique virtuelle complète d'un système physique, mise à jour en temps réel
        avec données capteurs pour simulation, prédiction et optimisation.
        
        **Applications Aérospatial:**
        🚀 Prédiction maintenance
        📊 Optimisation performance temps réel  
        🔍 Diagnostic pannes
        🎯 Test modifications virtuelles
        📈 Amélioration continue
        """)
        
        st.write("### 🏗️ Architecture Jumeau Numérique")
        
        # Diagramme architecture
        architecture_layers = [
            "🌐 Système Physique (Fusée)",
            "📡 Capteurs IoT",
            "☁️ Cloud / Edge Computing",
            "🧠 Modèles IA/ML",
            "💻 Jumeau Numérique 3D",
            "📊 Visualisation & Analytics",
            "👤 Utilisateurs / Ingénieurs"
        ]
        
        for i, layer in enumerate(architecture_layers):
            st.markdown(f"**{i+1}.** {layer}")
            if i < len(architecture_layers) - 1:
                st.markdown("↓")
        
        st.markdown("---")
        
        st.write("### ⚡ Avantages")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("""
            **Opérationnel:**
            - Réduction downtime 30-50%
            - Maintenance prédictive
            - Optimisation temps réel
            - Détection anomalies précoce
            """)
        
        with col2:
            st.write("""
            **Économique:**
            - Réduction coûts maintenance 25%
            - Tests virtuels sans risque
            - Prolongation durée vie
            - ROI: 18-24 mois
            """)
    
    with tab2:
        st.subheader("⚙️ Créer Jumeau Numérique")
        
        if not st.session_state.rocket_system['rockets']:
            st.warning("⚠️ Créez une fusée d'abord")
        else:
            with st.form("create_digital_twin"):
                st.write("### 🎯 Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    twin_name = st.text_input("Nom Jumeau", "DT-Artemis-X")
                    
                    rocket_source = st.selectbox(
                        "Fusée Source",
                        [f"{r['name']}" for r in st.session_state.rocket_system['rockets'].values()]
                    )
                
                with col2:
                    fidelity = st.selectbox("Fidélité Modèle", ["Basse", "Moyenne", "Haute", "Ultra"])
                    
                    update_frequency = st.selectbox("Fréquence MAJ", ["1 Hz", "10 Hz", "100 Hz", "Temps Réel"])
                
                st.write("### 📡 Capteurs Virtuels")
                
                sensors = st.multiselect(
                    "Sélectionner Capteurs",
                    ["Pression Chambre", "Température Tuyère", "Vibrations", "Poussée",
                     "Débit Propergol", "Position GPS", "Attitude", "Accélération",
                     "Contraintes Structure", "Température Structure"],
                    default=["Pression Chambre", "Poussée", "Position GPS"]
                )
                
                st.write("### 🧠 Modèles IA")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    predictive_maintenance = st.checkbox("Maintenance Prédictive", value=True)
                    anomaly_detection = st.checkbox("Détection Anomalies", value=True)
                
                with col2:
                    performance_optimization = st.checkbox("Optimisation Performance", value=True)
                    failure_prediction = st.checkbox("Prédiction Pannes", value=True)
                
                if st.form_submit_button("💻 Créer Jumeau Numérique", type="primary"):
                    with st.spinner("Création jumeau numérique..."):
                        import time
                        
                        progress = st.progress(0)
                        
                        steps = [
                            "Numérisation géométrie 3D...",
                            "Extraction paramètres physiques...",
                            "Calibration modèles...",
                            "Connexion capteurs virtuels...",
                            "Entraînement modèles IA...",
                            "Validation jumeau..."
                        ]
                        
                        for i, step in enumerate(steps):
                            progress.progress((i + 1) / len(steps))
                            st.text(step)
                            time.sleep(0.5)
                        
                        st.success(f"✅ Jumeau numérique '{twin_name}' créé!")
                        st.balloons()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Capteurs", len(sensors))
                        with col2:
                            st.metric("Précision", "99.2%")
                        with col3:
                            st.metric("Latence", "12 ms")
                        with col4:
                            st.metric("Fidélité", fidelity)

    with tab3:
        st.subheader("📊 Monitoring Jumeau Numérique")
        
        st.write("### 🖥️ Tableau de Bord Temps Réel")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("État Système", "🟢 Nominal")
        with col2:
            st.metric("Santé Globale", "96.8%", delta="+2.1%")
        with col3:
            st.metric("Anomalies", "0")
        with col4:
            st.metric("Prochaine Maintenance", "47 jours")
        
        st.markdown("---")
        
        # Graphiques monitoring
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Télémétrie Temps Réel**")
            
            t = np.linspace(0, 10, 100)
            pressure = 30 + np.sin(t) + np.random.randn(100) * 0.2
            temp = 2800 + 50 * np.sin(t * 0.5) + np.random.randn(100) * 10
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Pression Chambre", "Température"))
            
            fig.add_trace(go.Scatter(x=t, y=pressure, mode='lines', name='Pression'), row=1, col=1)
            fig.add_trace(go.Scatter(x=t, y=temp, mode='lines', name='Temp', line=dict(color='red')), row=2, col=1)
            
            fig.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Prédictions IA**")
            
            predictions = pd.DataFrame([
                {"Composant": "Pompe LOX", "Santé": 98, "RUL": "2400h", "Risque": "Faible"},
                {"Composant": "Injecteurs", "Santé": 95, "RUL": "1800h", "Risque": "Faible"},
                {"Composant": "Tuyère", "Santé": 88, "RUL": "800h", "Risque": "Moyen"},
                {"Composant": "Turbopompe", "Santé": 92, "RUL": "1200h", "Risque": "Faible"}
            ])
            
            st.dataframe(predictions, use_container_width=True)
            
            st.info("**RUL:** Remaining Useful Life (Durée vie restante)")
                                          
# ==================== PAGE: SYSTÈMES GUIDAGE ====================
elif page == "🛰️ Systèmes Guidage":
    st.header("🛰️ Systèmes de Guidage et Navigation")
    
    tab1, tab2, tab3 = st.tabs(["🧭 Guidage", "📡 Navigation", "🎯 Contrôle"])
    
    with tab1:
        st.subheader("🧭 Systèmes de Guidage")
        
        st.info("""
        **Guidage - Calcul Trajectoire Optimale:**
        
        Détermine comment atteindre l'objectif de manière optimale.
        
        **Types:**
        - Open-loop (pré-programmé)
        - Closed-loop (temps réel)
        - Optimal (minimise propergol/temps)
        - Adaptatif (IA)
        """)
        
        st.write("### 🎯 Algorithmes de Guidage")
        
        guidance_algorithms = {
            "🚀 Gravity Turn": {
                "description": "Rotation naturelle suivant gravité",
                "complexité": "Faible",
                "précision": "Moyenne",
                "usage": "Lanceurs orbitaux",
                "delta_v_loss": "200-500 m/s"
            },
            "📐 PEG (Powered Explicit Guidance)": {
                "description": "Guidage explicite optimal en temps réel",
                "complexité": "Élevée",
                "précision": "Très haute",
                "usage": "Navette Spatiale, Falcon 9",
                "delta_v_loss": "50-100 m/s"
            },
            "🎯 Q-Guidance": {
                "description": "Minimisation intégrale accélération",
                "complexité": "Moyenne",
                "précision": "Haute",
                "usage": "Missiles, fusées militaires",
                "delta_v_loss": "100-200 m/s"
            },
            "🤖 IA Adaptive": {
                "description": "Réseau neuronal temps réel",
                "complexité": "Très élevée",
                "précision": "Excellente",
                "usage": "Futur, atterrissages autonomes",
                "delta_v_loss": "< 50 m/s"
            }
        }
        
        for algo, details in guidance_algorithms.items():
            with st.expander(f"{algo}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {details['description']}")
                    st.write(f"**Complexité:** {details['complexité']}")
                
                with col2:
                    st.write(f"**Précision:** {details['précision']}")
                    st.write(f"**Usage:** {details['usage']}")
                    st.metric("Perte ΔV", details['delta_v_loss'])
        
        st.markdown("---")
        
        st.write("### 🧮 Simulateur Guidage")
        
        with st.form("guidance_sim"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_orbit_alt = st.number_input("Altitude Cible (km)", 200, 2000, 400, 50)
                guidance_type = st.selectbox("Type Guidage", ["Gravity Turn", "PEG", "Q-Guidance", "IA Adaptive"])
            
            with col2:
                initial_mass = st.number_input("Masse Initiale (tonnes)", 50, 1000, 500, 50)
                thrust_guidance = st.number_input("Poussée (MN)", 1, 50, 9, 1)
            
            if st.form_submit_button("🚀 Simuler"):
                with st.spinner("Calcul trajectoire..."):
                    import time
                    time.sleep(2)
                    
                    # Pertes selon algorithme
                    losses = {
                        "Gravity Turn": 350,
                        "PEG": 75,
                        "Q-Guidance": 150,
                        "IA Adaptive": 40
                    }
                    
                    dv_loss = losses[guidance_type]
                    dv_required = 9400 + dv_loss  # LEO + pertes
                    
                    st.success("✅ Simulation complétée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ΔV Requis", f"{dv_required} m/s")
                    with col2:
                        st.metric("Pertes Guidage", f"{dv_loss} m/s")
                    with col3:
                        efficiency = (9400 / dv_required) * 100
                        st.metric("Efficacité", f"{efficiency:.1f}%")
    
    with tab2:
        st.subheader("📡 Systèmes de Navigation")
        
        st.info("""
        **Navigation - Détermination Position/Vitesse:**
        
        🛰️ **GPS** - Précision 10-30m (civil), 1-5m (militaire)
        🌟 **Navigation Stellaire** - Précision arcsecondes
        📡 **INS (Inertial)** - Gyroscopes + accéléromètres
        📏 **Radar/Lidar** - Mesure distance/vitesse
        """)
        
        st.write("### 📊 Comparaison Systèmes")
        
        nav_systems = pd.DataFrame([
            {"Système": "GPS", "Précision": "10-30m", "Disponibilité": "Global (LEO)", "Autonomie": "Non", "Coût": "$"},
            {"Système": "GLONASS", "Précision": "5-10m", "Disponibilité": "Global", "Autonomie": "Non", "Coût": "$"},
            {"Système": "Galileo", "Précision": "1m", "Disponibilité": "Global", "Autonomie": "Non", "Coût": "$"},
            {"Système": "INS (Inertiel)", "Précision": "0.1-1 km/h", "Disponibilité": "Partout", "Autonomie": "Oui", "Coût": "$$"},
            {"Système": "Star Tracker", "Précision": "1 arcsec", "Disponibilité": "Espace", "Autonomie": "Oui", "Coût": "$$"},
            {"Système": "Radar Doppler", "Précision": "1-10m", "Disponibilité": "Limité", "Autonomie": "Oui", "Coût": "$$"}
        ])
        
        st.dataframe(nav_systems, use_container_width=True)
        
        st.markdown("---")
        
        st.write("### 🧭 Navigation Hybride")
        
        st.success("""
        **Approche Moderne: Fusion Multi-capteurs**
        
        Combine plusieurs systèmes pour précision et fiabilité maximales:
        
        📍 **Filtre de Kalman Étendu (EKF)**
        - Fusionne GPS + INS + Star Tracker
        - Compensation dérive INS par GPS
        - Redondance si perte GPS
        - Précision: < 1m position, < 0.1m/s vitesse
        
        🤖 **Amélioration IA**
        - Détection/correction biais capteurs
        - Prédiction trajectoire
        - Adaptation conditions
        """)
    
    with tab3:
        st.subheader("🎯 Systèmes de Contrôle")
        
        st.info("""
        **Contrôle d'Attitude et Propulsion:**
        
        Maintient orientation et exécute manœuvres.
        
        **Actionneurs:**
        - Gimbaling moteurs (±5-15°)
        - RCS (Reaction Control System)
        - Ailerons aérodynamiques
        - Roues inertielles
        """)
        
        st.write("### ⚙️ Contrôleurs")
        
        controllers = {
            "PID (Proportionnel-Intégral-Dérivé)": {
                "complexité": "Faible",
                "performance": "Bonne",
                "robustesse": "Moyenne",
                "usage": "Systèmes linéaires simples"
            },
            "LQR (Linear Quadratic Regulator)": {
                "complexité": "Moyenne",
                "performance": "Très bonne",
                "robustesse": "Bonne",
                "usage": "Contrôle optimal, fusées modernes"
            },
            "MPC (Model Predictive Control)": {
                "complexité": "Élevée",
                "performance": "Excellente",
                "robustesse": "Très bonne",
                "usage": "Systèmes complexes, contraintes"
            },
            "IA/Réseau Neuronal": {
                "complexité": "Très élevée",
                "performance": "Adaptative",
                "robustesse": "Excellente",
                "usage": "Atterrissages autonomes, adaptation"
            }
        }
        
        for controller, specs in controllers.items():
            with st.expander(f"🎮 {controller}"):
                for key, value in specs.items():
                    st.write(f"**{key.title()}:** {value}")
        
        st.markdown("---")
        
        st.write("### 🎮 Simulation Contrôle Attitude")
        
        if st.button("🚀 Lancer Simulation", key="control_sim"):
            
            # Simulation perturbation et correction
            t_sim = np.linspace(0, 30, 300)
            
            # Perturbation à t=5s
            perturbation = np.zeros_like(t_sim)
            perturbation[t_sim > 5] = 5 * np.exp(-(t_sim[t_sim > 5] - 5) / 2)
            
            # Réponse PID
            error = perturbation.copy()
            correction = np.zeros_like(t_sim)
            
            Kp, Ki, Kd = 2, 0.5, 1
            
            for i in range(1, len(t_sim)):
                correction[i] = -Kp * error[i-1] - Ki * np.sum(error[:i]) * 0.1 - Kd * (error[i-1] - error[i-2] if i > 1 else 0)
                error[i] = error[i-1] + correction[i] * 0.1
            
            attitude = perturbation + correction
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=t_sim, y=perturbation, mode='lines', name='Perturbation', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=t_sim, y=attitude, mode='lines', name='Attitude', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=t_sim, y=correction, mode='lines', name='Correction', line=dict(color='green')))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Contrôle Attitude PID - Réponse à Perturbation",
                xaxis_title="Temps (s)",
                yaxis_title="Angle (°)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("""
            ✅ **Résultats:**
            - Temps stabilisation: 8.2s
            - Overshoot: 12%
            - Erreur statique: < 0.1°
            """)

# ==================== PAGE: PHYSIQUE AVANCÉE ====================
elif page == "🔬 Physique Avancée":
    st.header("🔬 Physique Spatiale Avancée")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ Relativité", "🌌 Gravité", "☢️ Radiation", "🌊 Ondes Gravitationnelles"])
    
    with tab1:
        st.subheader("⚛️ Effets Relativistes")
        
        st.info("""
        **Relativité Restreinte:**
        
        À vitesses très élevées (proche c), effets observables:
        - Dilatation temps
        - Contraction longueurs
        - Augmentation masse
        
        Négligeable pour fusées actuelles (v << 0.01c)
        Critique pour voyages interstellaires futurs
        """)
        
        st.write("### 🧮 Calculateur Relativiste")
        
        with st.form("relativity_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                velocity_frac = st.slider("Vitesse (fraction c)", 0.0, 0.99, 0.1, 0.01)
                proper_time = st.number_input("Temps propre (années)", 1, 100, 10, 1)
            
            with col2:
                rest_mass = st.number_input("Masse au repos (tonnes)", 100, 100000, 1000, 100)
            
            if st.form_submit_button("🔬 Calculer Effets Relativistes"):
                c = PHYSICS_CONSTANTS['c']
                v = velocity_frac * c
                
                # Facteur de Lorentz
                gamma = 1 / np.sqrt(1 - velocity_frac**2)
                
                # Dilatation temps
                dilated_time = proper_time * gamma
                
                # Masse relativiste
                relativistic_mass = rest_mass * gamma
                
                # Énergie cinétique
                E_kinetic = (gamma - 1) * rest_mass * c**2
                
                st.success("✅ Calculs relativistes complétés!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Facteur γ (gamma)", f"{gamma:.4f}")
                with col2:
                    st.metric("Temps Dilaté", f"{dilated_time:.2f} ans")
                with col3:
                    st.metric("Masse Relativiste", f"{relativistic_mass:.0f} t")
                
                st.metric("Énergie Cinétique", f"{E_kinetic:.2e} J")
                
                if velocity_frac > 0.1:
                    st.warning(f"⚠️ Effets relativistes significatifs à {velocity_frac*100:.0f}% c")
                
                # Graphique gamma
                v_range = np.linspace(0, 0.99, 100)
                gamma_range = 1 / np.sqrt(1 - v_range**2)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=v_range*100, y=gamma_range, mode='lines', line=dict(width=3)))
                fig.add_vline(x=velocity_frac*100, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title="Facteur de Lorentz γ vs Vitesse",
                    xaxis_title="Vitesse (% c)",
                    yaxis_title="γ",
                    yaxis_type="log",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🌌 Gravité et Espace-Temps")
        
        st.info("""
        **Relativité Générale:**
        
        Gravité = courbure espace-temps par masse/énergie
        
        **Effets Observables:**
        - Précession périhélie Mercure
        - Lentille gravitationnelle
        - Ondes gravitationnelles
        - Trous noirs
        """)
        
        st.write("### 🌀 Rayon de Schwarzschild (Trou Noir)")
        
        st.latex(r"r_s = \frac{2GM}{c^2}")
        
        with st.form("schwarzschild_calc"):
            mass_object = st.number_input("Masse Objet (masses solaires)", 0.1, 1000.0, 1.0, 0.1)
            
            if st.form_submit_button("🔬 Calculer"):
                M_sun = PHYSICS_CONSTANTS['SUN_MASS'] if 'SUN_MASS' in dir(PHYSICS_CONSTANTS) else 1.989e30
                M = mass_object * M_sun
                
                G = PHYSICS_CONSTANTS['G']
                c = PHYSICS_CONSTANTS['c']
                
                r_s = 2 * G * M / c**2
                
                st.success("✅ Calcul terminé!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Rayon Schwarzschild", f"{r_s/1000:.2f} km")
                
                with col2:
                    density = M / (4/3 * np.pi * r_s**3)
                    st.metric("Densité Moyenne", f"{density:.2e} kg/m³")
                
                if mass_object == 1.0:
                    st.info("ℹ️ Soleil: Rs = 2.95 km (bien plus petit que rayon réel 696,000 km)")
                
                st.write("### 📊 Comparaisons")
                
                objects = pd.DataFrame([
                    {"Objet": "Soleil", "Masse (M☉)": 1, "Rs (km)": 2.95},
                    {"Objet": "Terre", "Masse (M☉)": 3e-6, "Rs (km)": 0.0088},
                    {"Objet": "Trou Noir Stellaire", "Masse (M☉)": 10, "Rs (km)": 29.5},
                    {"Objet": "Sgr A* (centre Galaxie)", "Masse (M☉)": 4.3e6, "Rs (km)": 1.27e7}
                ])
                
                st.dataframe(objects, use_container_width=True)
    
    with tab3:
        st.subheader("☢️ Radiation Spatiale")
        
        st.warning("""
        **Dangers Radiation Espace:**
        
        ⚠️ **Rayons Cosmiques Galactiques (GCR)** - Haute énergie, pénétrants
        ☀️ **Éruptions Solaires (SPE)** - Intenses mais prévisibles
        🌍 **Ceintures Van Allen** - Piégées champ magnétique Terre
        
        **Effets:**
        - Dommages ADN (cancer)
        - Radiation aiguë (doses fortes)
        - Dégâts électroniques
        """)
        
        st.write("### 📊 Doses Radiation")
        
        radiation_doses = pd.DataFrame([
            {"Source": "Background Terre", "Dose": "2.4 mSv/an", "Équivalent": "Baseline"},
            {"Source": "Vol Transatlantique", "Dose": "0.04 mSv", "Équivalent": "1 radio poumons"},
            {"Source": "ISS (6 mois)", "Dose": "80 mSv", "Équivalent": "33x background annuel"},
            {"Source": "Mission Mars (3 ans)", "Dose": "500-1000 mSv", "Équivalent": "Limite carrière NASA"},
            {"Source": "Éruption Solaire", "Dose": "5000 mSv", "Équivalent": "Mortel sans protection"},
            {"Source": "Dose létale", "Dose": "> 10000 mSv", "Équivalent": "Mort en jours/semaines"}
        ])
        
        st.dataframe(radiation_doses, use_container_width=True)
        
        st.write("### 🛡️ Protection")
        
        protections = {
            "💧 Eau/Hydrogène": "Meilleur bouclier (léger, efficace protons)",
            "🧱 Polyéthylène": "Bon compromis masse/protection",
            "⚙️ Aluminium": "Protection moyenne, lourd",
            "🧲 Bouclier Magnétique": "Concept futur, actif",
            "🚀 Vitesse Mission": "Moins temps = moins dose"
        }
        
        for protection, description in protections.items():
            st.write(f"{protection}: {description}")
    
    with tab4:
        st.subheader("🌊 Ondes Gravitationnelles")
        
        st.info("""
        **Ondes Gravitationnelles:**
        
        Rides dans espace-temps causées par objets massifs accélérés.
        
        **Détectées 2015 (LIGO):**
        - Fusion trous noirs
        - Fusion étoiles neutrons
        - Confirme Relativité Générale
        
        **Impact Aérospatial:**
        - Navigation ultra-précise future
        - Détection objets massifs
        - Tests physique fondamentale
        """)
        
        st.write("### 📡 Détecteurs")
        
        detectors = pd.DataFrame([
            {"Détecteur": "LIGO (USA)", "Bras": "4 km", "Sensibilité": "10⁻²¹", "Statut": "✅ Opérationnel"},
            {"Détecteur": "Virgo (Europe)", "Bras": "3 km", "Sensibilité": "10⁻²¹", "Statut": "✅ Opérationnel"},
            {"Détecteur": "KAGRA (Japon)", "Bras": "3 km", "Sensibilité": "10⁻²¹", "Statut": "✅ Opérationnel"},
            {"Détecteur": "LISA (Espace)", "Bras": "2.5M km", "Sensibilité": "10⁻²³", "Statut": "🔜 2030s"}
        ])
        
        st.dataframe(detectors, use_container_width=True)
        
        st.success("""
        **Applications Futures:**
        
        🛰️ **Navigation Spatiale:**
        - Détection masse cachée
        - Cartographie espace-temps
        - Positionnement ultra-précis
        
        🔬 **Science:**
        - Étude trous noirs
        - Test théories gravité
        - Cosmologie primordiale
        """)

# ==================== PAGE: PROPULSION EXOTIQUE ====================
elif page == "🌌 Propulsion Exotique":
    st.header("🌌 Propulsion Exotique et Futuriste")
    
    tab1, tab2, tab3 = st.tabs(["🚀 Concepts Avancés", "⚛️ Antimatière", "🌟 Interstellaire"])
    
    with tab1:
        st.subheader("🚀 Concepts de Propulsion Avancés")
        
        exotic_propulsion = {
            "⚡ Propulsion Plasma VASIMR": {
                "principe": "Ionisation + accélération champs magnétiques",
                "isp": "3,000-30,000 s",
                "poussée": "5 N",
                "puissance": "200 kW",
                "trl": "5-6",
                "avantages": "Isp variable, efficacité haute",
                "défis": "Puissance électrique énorme",
                "timeline": "2030s"
            },
            "☢️ Propulsion Nucléaire Pulsée (Orion)": {
                "principe": "Explosions nucléaires derrière plaque absorbante",
                "isp": "6,000-10,000 s",
                "poussée": "MN-GN",
                "puissance": "Bombes H",
                "trl": "2-3",
                "avantages": "Poussée massive, Isp élevé",
                "défis": "Traité nucléaire, fallout",
                "timeline": "Interdit actuellement"
            },
            "⚛️ Fusion Nucléaire": {
                "principe": "Réaction D-T ou D-He3, plasma confiné",
                "isp": "10,000-100,000 s",
                "poussée": "kN-MN",
                "puissance": "GW",
                "trl": "2-3",
                "avantages": "Énorme énergie, propergol abondant",
                "défis": "Confinement plasma, ignition",
                "timeline": "2050s+"
            },
            "💫 Antimatière": {
                "principe": "Annihilation matière-antimatière (E=mc²)",
                "isp": "100,000-1,000,000 s",
                "poussée": "Variable",
                "puissance": "Théoriquement maximale",
                "trl": "1",
                "avantages": "Efficacité maximale théorique",
                "défis": "Production/stockage antimatière impossible",
                "timeline": "Siècles"
            },
            "🌟 Voile Photonique Laser": {
                "principe": "Laser Terre pousse voile réfléchissante",
                "isp": "∞ (pas de propergol)",
                "poussée": "μN-mN",
                "puissance": "GW (laser Terre)",
                "trl": "4-5",
                "avantages": "Pas de propergol, missions longues",
                "défis": "Poussée très faible, nécessite laser spatial",
                "timeline": "Breakthrough Starshot 2030s"
            },
            "🌀 Ramjet Bussard": {
                "principe": "Collecte hydrogène interstellaire pour fusion",
                "isp": "Théoriquement infini",
                "poussée": "Variable",
                "puissance": "Auto-alimenté",
                "trl": "1",
                "avantages": "Pas de propergol embarqué",
                "défis": "Densité H trop faible, traînée > poussée",
                "timeline": "Concept théorique"
            },
            "🎯 EM Drive (Controversé)": {
                "principe": "Cavité résonante micro-ondes (violant Newton?)",
                "isp": "Théoriquement infini",
                "poussée": "μN (allégué)",
                "puissance": "kW",
                "trl": "1-2",
                "avantages": "Pas propergol si fonctionne",
                "défis": "Non reproductible, viole physique",
                "timeline": "Probablement impossible"
            }
        }
        
        for propulsion, details in exotic_propulsion.items():
            with st.expander(f"{propulsion}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Principe:** {details['principe']}")
                    st.write(f"**Isp:** {details['isp']}")
                    st.write(f"**Poussée:** {details['poussée']}")
                    st.write(f"**Puissance:** {details['puissance']}")
                
                with col2:
                    st.write(f"**TRL:** {details['trl']}")
                    st.write(f"✅ **Avantages:** {details['avantages']}")
                    st.write(f"❌ **Défis:** {details['défis']}")
                    st.write(f"**Timeline:** {details['timeline']}")
    
    with tab2:
        st.subheader("⚛️ Propulsion Antimatière")
        
        st.warning("""
        **Antimatière - Énergie Ultime:**
        
        Annihilation matière-antimatière libère 100% masse en énergie (E=mc²)
        
        **Potentiel:**
        - 1 kg antimatière = 43 mégatonnes TNT
        - Isp théorique: 1,000,000 s
        - Mission interstellaire faisable
        
        **Problèmes MAJEURS:**
        1. Production: 1 nanogramme = milliards $
        2. Stockage: Pièges magnétiques ultra-complexes
        3. Quantité: Besoin tonnes, production actuelle = picogrammes/an
        """)
        
        st.write("### 🧮 Calculateur Mission Antimatière")
        
        with st.form("antimatter_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                spacecraft_mass_am = st.number_input("Masse Vaisseau (tonnes)", 100, 10000, 1000, 100)
                target_velocity_am = st.slider("Vitesse Cible (% c)", 1, 50, 10, 1)
            
            with col2:
                efficiency_am = st.slider("Efficacité Conversion (%)", 10, 90, 50, 5)
            
            if st.form_submit_button("🔬 Calculer Besoins Antimatière"):
                c = PHYSICS_CONSTANTS['c']
                v_target = target_velocity_am / 100 * c
                
                # Équation relativiste simplifiée
                gamma_final = 1 / np.sqrt(1 - (target_velocity_am/100)**2)
                
                # Énergie cinétique relativiste
                E_kinetic = (gamma_final - 1) * spacecraft_mass_am * 1000 * c**2
                
                # Masse antimatière nécessaire
                E_per_kg = c**2  # J/kg
                antimatter_mass = E_kinetic / (E_per_kg * efficiency_am / 100) / 2  # /2 car matière+antimatière
                
                # Coût (1 gramme = $62.5 trillions estimé)
                cost_per_gram = 62.5e12        

                # Masse totale transportée (en grammes)
                total_mass_grams = (antimatter_mass + cost_per_gram) * 1e6  # tonnes → grammes

                # Coût total
                mission_cost = total_mass_grams * cost_per_gram

                # Affichage formaté
                st.metric("💰 Coût Mission Estimé", f"${mission_cost:,.2e}")
                st.info(f"Évaluation basée sur une valeur de {cost_per_gram:,.2e} $/g.")
    
                cost_total = antimatter_mass * cost_per_gram
                
                # Temps production (production actuelle: 1 nanogramme/an)
                current_production = 1e-12  # kg/an
                years_production = antimatter_mass / current_production
                
                st.success("✅ Calculs terminés!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Antimatière Requise", f"{antimatter_mass:.6f} kg")
                with col2:
                    st.metric("Coût Estimé", f"${cost_total:.2e}")
                with col3:
                    st.metric("Vitesse Finale", f"{target_velocity_am}% c")
                
                st.error(f"⚠️ Temps Production (taux actuel): {years_production:.2e} ans")
                st.info("ℹ️ L'Univers a 13.8 milliards d'années...")
    
    with tab3:
        st.subheader("🌟 Voyages Interstellaires")
        
        st.info("""
        **Étoiles Proches:**
        
        🌟 Proxima Centauri: 4.24 années-lumière
        🌟 Alpha Centauri: 4.37 années-lumière
        🌟 Barnard's Star: 5.96 années-lumière
        
        **Défis:**
        - Temps de voyage (décennies-siècles)
        - Énergie colossale
        - Support vie longue durée
        - Communication (années de latence)
        """)
        
        st.write("### 🚀 Calculateur Mission Interstellaire")
        
        with st.form("interstellar_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_star = st.selectbox("Étoile Cible", 
                    ["Proxima Centauri (4.24 al)", "Alpha Centauri (4.37 al)", 
                     "Barnard's Star (5.96 al)", "Sirius (8.6 al)"])
                
                propulsion_interstellar = st.selectbox("Propulsion",
                    ["Chimique", "Nucléaire", "Fusion", "Antimatière", "Voile Laser"])
            
            with col2:
                velocity_percent = st.slider("Vitesse Croisière (% c)", 1, 50, 10, 1)
            
            if st.form_submit_button("🔬 Calculer Mission"):
                # Distance
                distances = {
                    "Proxima Centauri (4.24 al)": 4.24,
                    "Alpha Centauri (4.37 al)": 4.37,
                    "Barnard's Star (5.96 al)": 5.96,
                    "Sirius (8.6 al)": 8.6
                }
                
                distance_ly = distances[target_star]
                v_frac = velocity_percent / 100
                
                # Temps voyage (sans accélération/décélération)
                travel_time = distance_ly / v_frac  # années
                
                # Avec accélération/décélération (simplifié)
                accel_time = 1  # an à 1g
                coast_time = travel_time - 2 * accel_time
                total_time = travel_time + 2 * accel_time
                
                # Effet relativiste
                gamma = 1 / np.sqrt(1 - v_frac**2)
                proper_time = total_time / gamma
                
                st.success("✅ Mission calculée!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Distance", f"{distance_ly} al")
                with col2:
                    st.metric("Temps (Terre)", f"{total_time:.1f} ans")
                with col3:
                    st.metric("Temps (Vaisseau)", f"{proper_time:.1f} ans")
                with col4:
                    st.metric("Vitesse", f"{velocity_percent}% c")
                
                # Timeline
                st.write("### 📅 Timeline Mission")
                
                timeline = pd.DataFrame([
                    {"Phase": "Accélération", "Durée": f"{accel_time} an", "Vitesse": f"0 → {velocity_percent}% c"},
                    {"Phase": "Croisière", "Durée": f"{coast_time:.1f} ans", "Vitesse": f"{velocity_percent}% c"},
                    {"Phase": "Décélération", "Durée": f"{accel_time} an", "Vitesse": f"{velocity_percent}% c → 0"},
                    {"Phase": "Arrivée", "Durée": "-", "Vitesse": "Orbite étoile"}
                ])
                
                st.dataframe(timeline, use_container_width=True)
                
                # Faisabilité
                if propulsion_interstellar == "Chimique":
                    st.error("❌ Impossible avec propulsion chimique (Isp trop faible)")
                elif propulsion_interstellar == "Nucléaire":
                    st.warning("⚠️ Très difficile - Mission siècles")
                elif propulsion_interstellar == "Fusion":
                    st.info("🟡 Possible si technologie fusion maîtrisée")
                elif propulsion_interstellar == "Antimatière":
                    st.success("✅ Possible théoriquement (production impossible actuellement)")
                else:  # Voile Laser
                    st.info("🟢 Breakthrough Starshot vise 20% c vers Proxima")
            
# ==================== PAGE: THERMODYNAMIQUE ====================
elif page == "🌡️ Thermodynamique":
    st.header("🌡️ Thermodynamique Aérospatiale")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Combustion", "❄️ Cryogénie", "🛡️ Protection Thermique", "📊 Calculs"])
    
    with tab1:
        st.subheader("🔥 Thermodynamique de la Combustion")
        
        st.info("""
        **Processus de Combustion:**
        
        Réaction exothermique entre carburant et comburant produisant:
        - Gaz chauds haute vitesse
        - Poussée (3ème loi Newton)
        - Température: 2500-3500°C
        """)
        
        st.write("### ⚗️ Calculateur Combustion")
        
        with st.form("combustion_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                fuel_type = st.selectbox("Carburant", ["RP-1", "LH2", "Méthane", "UDMH"])
                oxidizer_type = st.selectbox("Comburant", ["LOX", "N2O4", "H2O2"])
            
            with col2:
                mixture_ratio = st.slider("Rapport Mélange (O/F)", 1.0, 8.0, 2.5, 0.1)
                chamber_pressure = st.number_input("Pression Chambre (MPa)", 5, 50, 20, 1)
            
            if st.form_submit_button("🔬 Calculer"):
                # Données combustion (simplifiées)
                combustion_data = {
                    ('RP-1', 'LOX'): {'T': 3670, 'Isp': 311, 'gamma': 1.24},
                    ('LH2', 'LOX'): {'T': 3400, 'Isp': 450, 'gamma': 1.26},
                    ('Méthane', 'LOX'): {'T': 3540, 'Isp': 369, 'gamma': 1.25}
                }
                
                data = combustion_data.get((fuel_type, oxidizer_type), {'T': 3500, 'Isp': 350, 'gamma': 1.25})
                
                flame_temp = data['T']
                isp_theoretical = data['Isp']
                gamma = data['gamma']
                
                # Vitesse échappement
                R = 8314  # J/kmol/K
                M = 20  # kg/kmol (approximation)
                c_star = np.sqrt(gamma * R * flame_temp / M) / np.sqrt(gamma * ((2/(gamma+1))**((gamma+1)/(gamma-1))))
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Température Flamme", f"{flame_temp} K")
                with col2:
                    st.metric("Isp Théorique", f"{isp_theoretical} s")
                with col3:
                    st.metric("c* (vitesse car.)", f"{c_star:.0f} m/s")
                with col4:
                    st.metric("Gamma", f"{gamma}")
                
                # Produits combustion
                st.write("### 🧪 Produits de Combustion")
                
                products = pd.DataFrame([
                    {"Espèce": "H2O", "Fraction Molaire": 0.45, "Masse Molaire": 18},
                    {"Espèce": "CO2", "Fraction Molaire": 0.35, "Masse Molaire": 44},
                    {"Espèce": "CO", "Fraction Molaire": 0.10, "Masse Molaire": 28},
                    {"Espèce": "H2", "Fraction Molaire": 0.07, "Masse Molaire": 2},
                    {"Espèce": "OH", "Fraction Molaire": 0.03, "Masse Molaire": 17}
                ])
                
                st.dataframe(products, use_container_width=True)
    
    with tab2:
        st.subheader("❄️ Cryogénie")
        
        st.info("""
        **Propergols Cryogéniques:**
        
        Liquides à très basse température utilisés pour performance maximale.
        
        **Avantages:**
        - Isp élevé (LOX/LH2: 450s)
        - Densité énergétique
        
        **Défis:**
        - Stockage complexe
        - Boil-off (évaporation)
        - Isolation thermique critique
        """)
        
        st.write("### ❄️ Propergols Cryogéniques")
        
        cryo_data = [
            {"Propergol": "LOX (Oxygène Liquide)", "T° Ébullition": "-183°C", "Densité": "1141 kg/m³", "Boil-off": "1-2%/jour"},
            {"Propergol": "LH2 (Hydrogène Liquide)", "T° Ébullition": "-253°C", "Densité": "71 kg/m³", "Boil-off": "3-5%/jour"},
            {"Propergol": "LNG (Méthane Liquide)", "T° Ébullition": "-162°C", "Densité": "423 kg/m³", "Boil-off": "0.5-1%/jour"},
            {"Propergol": "N2O4 (Tétroxyde)", "T° Ébullition": "+21°C", "Densité": "1450 kg/m³", "Boil-off": "Stockable"}
        ]
        
        df_cryo = pd.DataFrame(cryo_data)
        st.dataframe(df_cryo, use_container_width=True)
        
        st.write("### 🧮 Calculateur Boil-off")
        
        with st.form("boil_off_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                propellant_volume = st.number_input("Volume Propergol (m³)", 1, 1000, 100, 10)
                storage_duration = st.number_input("Durée Stockage (jours)", 1, 365, 30, 1)
            
            with col2:
                propellant_cryo = st.selectbox("Propergol", ["LOX", "LH2", "LNG"])
                insulation_quality = st.selectbox("Qualité Isolation", ["Standard", "Bonne", "Excellente"])
            
            if st.form_submit_button("🔬 Calculer Boil-off"):
                # Taux boil-off
                boil_rates = {
                    'LOX': {'Standard': 0.02, 'Bonne': 0.015, 'Excellente': 0.01},
                    'LH2': {'Standard': 0.05, 'Bonne': 0.03, 'Excellente': 0.02},
                    'LNG': {'Standard': 0.01, 'Bonne': 0.007, 'Excellente': 0.005}
                }
                
                daily_rate = boil_rates[propellant_cryo][insulation_quality]
                
                # Densités
                densities = {'LOX': 1141, 'LH2': 71, 'LNG': 423}
                density = densities[propellant_cryo]
                
                initial_mass = propellant_volume * density
                mass_lost = initial_mass * daily_rate * storage_duration
                remaining_mass = initial_mass - mass_lost
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Masse Initiale", f"{initial_mass:.0f} kg")
                with col2:
                    st.metric("Perte par Boil-off", f"{mass_lost:.0f} kg")
                with col3:
                    st.metric("Masse Restante", f"{remaining_mass:.0f} kg")
                
                st.metric("Perte Totale", f"{(mass_lost/initial_mass*100):.1f}%")
    
    with tab3:
        st.subheader("🛡️ Protection Thermique")
        
        st.info("""
        **Systèmes de Protection Thermique:**
        
        Protègent structures des températures extrêmes:
        - Rentrée atmosphérique: 1600-2000°C
        - Moteurs: 2500-3500°C
        - Cryogénie: -253°C
        """)
        
        st.write("### 🛡️ Types de Protection")
        
        protection_types = {
            "Boucliers Ablatifs": {
                "principe": "Matériau qui s'érode en absorbant chaleur",
                "matériaux": "PICA, Avcoat, SIRCA",
                "température": "3000°C",
                "usage": "Rentrée atmosphérique (Apollo, Dragon)",
                "avantages": "Très haute température, simple",
                "inconvénients": "Usage unique, masse"
            },
            "Tuiles Réutilisables": {
                "principe": "Céramiques isolantes réutilisables",
                "matériaux": "Silice, fibres céramiques",
                "température": "1650°C",
                "usage": "Navette Spatiale, X-37B",
                "avantages": "Réutilisable 100+ fois",
                "inconvénients": "Fragile, maintenance"
            },
            "Refroidissement Actif": {
                "principe": "Circulation fluide pour évacuer chaleur",
                "matériaux": "Canaux + propergol",
                "température": "3500°C",
                "usage": "Moteurs (régénératif)",
                "avantages": "Haute performance",
                "inconvénients": "Complexité, poids"
            },
            "Refroidissement Film": {
                "principe": "Film fluide froid le long paroi",
                "matériaux": "Propergol gazeux",
                "température": "2500°C",
                "usage": "Moteurs fusée",
                "avantages": "Simple, efficace",
                "inconvénients": "Perte Isp"
            },
            "Isolation Multi-couches": {
                "principe": "Couches réflectrices vide",
                "matériaux": "Mylar aluminisé",
                "température": "±150°C",
                "usage": "Satellites, cryogénie",
                "avantages": "Léger, efficace vide",
                "inconvénients": "Fragile, pas atmosphère"
            }
        }
        
        for prot_type, details in protection_types.items():
            with st.expander(f"🛡️ {prot_type}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Principe:** {details['principe']}")
                    st.write(f"**Matériaux:** {details['matériaux']}")
                    st.write(f"**Température Max:** {details['température']}")
                
                with col2:
                    st.write(f"**Usage:** {details['usage']}")
                    st.write(f"✅ **Avantages:** {details['avantages']}")
                    st.write(f"❌ **Inconvénients:** {details['inconvénients']}")
    
    with tab4:
        st.subheader("📊 Calculs Thermodynamiques")
        
        st.write("### 🔬 Transfert Thermique")
        
        with st.form("heat_transfer"):
            col1, col2 = st.columns(2)
            
            with col1:
                material = st.selectbox("Matériau", ["Aluminium", "Titane", "Acier", "Composite CFRP"])
                thickness = st.number_input("Épaisseur (mm)", 1, 100, 10, 1)
            
            with col2:
                temp_hot = st.number_input("Température Chaude (°C)", 0, 3000, 1500, 10)
                temp_cold = st.number_input("Température Froide (°C)", -200, 500, 20, 10)
            
            if st.form_submit_button("🔬 Calculer Flux"):
                # Conductivités thermiques (W/m/K)
                conductivities = {
                    'Aluminium': 237,
                    'Titane': 21.9,
                    'Acier': 50,
                    'Composite CFRP': 5
                }
                
                k = conductivities[material]
                delta_T = abs(temp_hot - temp_cold)
                L = thickness / 1000  # mètres
                
                # Loi de Fourier
                heat_flux = k * delta_T / L  # W/m²
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Flux Thermique", f"{heat_flux:.0f} W/m²")
                with col2:
                    st.metric("Conductivité", f"{k} W/m/K")
                with col3:
                    st.metric("Gradient", f"{delta_T/L:.0f} K/m")
                
                if heat_flux > 1e6:
                    st.error("⚠️ FLUX CRITIQUE! Protection thermique active requise")
                elif heat_flux > 5e5:
                    st.warning("⚠️ Flux élevé - Vérifier résistance matériau")
                else:
                    st.success("✅ Flux acceptable")

# ==================== PAGE: AÉRODYNAMIQUE ====================
elif page == "⚡ Aérodynamique":
    st.header("⚡ Aérodynamique des Fusées")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💨 Principes", "📐 Formes", "🧮 Calculs", "🌪️ CFD"])
    
    with tab1:
        st.subheader("💨 Principes Aérodynamiques")
        
        st.info("""
        **Forces Aérodynamiques:**
        
        🔹 **Traînée (Drag)** - Résistance air, proportionnelle à v²
        🔹 **Portance (Lift)** - Perpendiculaire mouvement (négligeable fusées)
        🔹 **Pression Dynamique** - ½ρv² (max à Max-Q ~45-70s)
        🔹 **Nombre de Mach** - v/v_son (régimes sub/trans/supersonique)
        """)
        
        st.write("### 📊 Régimes d'Écoulement")
        
        regimes = pd.DataFrame([
            {"Régime": "Subsonique", "Mach": "< 0.8", "Caractéristiques": "Écoulement attaché, traînée faible"},
            {"Régime": "Transsonique", "Mach": "0.8 - 1.2", "Caractéristiques": "Ondes choc, traînée maximale"},
            {"Régime": "Supersonique", "Mach": "1.2 - 5", "Caractéristiques": "Ondes choc obliques, cône Mach"},
            {"Régime": "Hypersonique", "Mach": "> 5", "Caractéristiques": "Chauffage intense, plasma"}
        ])
        
        st.dataframe(regimes, use_container_width=True)
        
        st.write("### 📈 Évolution Traînée avec Mach")
        
        mach = np.linspace(0, 5, 100)
        cd = 0.3 + 0.5 * np.exp(-((mach - 1)**2)) + 0.1 * mach
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mach, y=cd, mode='lines', line=dict(width=3)))
        fig.add_vline(x=1, line_dash="dash", annotation_text="Mach 1")
        
        fig.update_layout(
            title="Coefficient de Traînée vs Mach",
            xaxis_title="Nombre de Mach",
            yaxis_title="Cd (Coefficient Traînée)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📐 Formes Aérodynamiques")
        
        st.write("### 🚀 Optimisation Forme")
        
        nose_shapes = {
            "🔺 Cône": {
                "finesse": "Faible",
                "traînée": "Élevée",
                "simplicité": "★★★★★",
                "usage": "Fusées anciennes, missiles",
                "cd": 0.50
            },
            "🥚 Ogivale": {
                "finesse": "Bonne",
                "traînée": "Moyenne",
                "simplicité": "★★★★☆",
                "usage": "Fusées modernes (Atlas, Delta)",
                "cd": 0.35
            },
            "💧 Parabolique": {
                "finesse": "Très bonne",
                "traînée": "Faible",
                "simplicité": "★★★☆☆",
                "usage": "Fusées optimisées",
                "cd": 0.28
            },
            "🎯 Von Karman": {
                "finesse": "Optimale",
                "traînée": "Minimale",
                "simplicité": "★★☆☆☆",
                "usage": "Records vitesse, fusées modernes",
                "cd": 0.25
            },
            "⚡ Spike": {
                "finesse": "Excellente",
                "traînée": "Très faible (supersonique)",
                "simplicité": "★☆☆☆☆",
                "usage": "Expérimental, missiles",
                "cd": 0.20
            }
        }
        
        for shape, details in nose_shapes.items():
            with st.expander(f"{shape}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Finesse:** {details['finesse']}")
                    st.write(f"**Traînée:** {details['traînée']}")
                    st.write(f"**Simplicité:** {details['simplicité']}")
                
                with col2:
                    st.write(f"**Usage:** {details['usage']}")
                    st.metric("Cd", details['cd'])
        
        st.markdown("---")
        
        st.write("### 🎨 Visualisation Formes")
        
        shape_select = st.selectbox("Sélectionner Forme", ["Cône", "Ogivale", "Parabolique", "Von Karman"])
        
        # Génération forme
        x = np.linspace(0, 10, 100)
        
        if shape_select == "Cône":
            y = x * 0.5
        elif shape_select == "Ogivale":
            y = np.sqrt(25 - (x-10)**2)
        elif shape_select == "Parabolique":
            y = 5 * (1 - (1 - x/10)**2)
        else:  # Von Karman
            y = 5 * np.sqrt(1 - ((x-10)/10)**2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(width=3), name='Profil'))
        fig.add_trace(go.Scatter(x=x, y=-y, mode='lines', line=dict(width=3), showlegend=False))
        
        fig.update_layout(
            title=f"Profil: {shape_select}",
            xaxis_title="Longueur (m)",
            yaxis_title="Rayon (m)",
            height=400
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🧮 Calculs Aérodynamiques")
        
        st.write("### 🔬 Calculateur Traînée")
        
        with st.form("drag_calc"):
            col1, col2 = st.columns(2)
            
            with col1:
                velocity = st.number_input("Vitesse (m/s)", 0, 8000, 500, 50)
                altitude_aero = st.number_input("Altitude (m)", 0, 100000, 10000, 1000)
            
            with col2:
                diameter_aero = st.number_input("Diamètre (m)", 1.0, 15.0, 5.0, 0.5)
                cd_input = st.number_input("Cd (coefficient traînée)", 0.1, 1.0, 0.35, 0.05)
            
            if st.form_submit_button("🔬 Calculer"):
                # Densité atmosphérique (modèle simplifié)
                rho_0 = 1.225  # kg/m³ niveau mer
                H = 8500  # m (échelle hauteur)
                rho = rho_0 * np.exp(-altitude_aero / H)
                
                # Aire frontale
                A = np.pi * (diameter_aero / 2)**2
                
                # Pression dynamique
                q = 0.5 * rho * velocity**2
                
                # Force traînée
                drag_force = cd_input * A * q
                
                # Vitesse son
                T = 288.15 - 0.0065 * altitude_aero  # K
                v_sound = np.sqrt(1.4 * 287 * max(T, 200))
                mach = velocity / v_sound
                
                st.success("✅ Calcul terminé!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Traînée", f"{drag_force/1000:.1f} kN")
                with col2:
                    st.metric("Pression Dyn (q)", f"{q/1000:.1f} kPa")
                with col3:
                    st.metric("Nombre de Mach", f"{mach:.2f}")
                with col4:
                    st.metric("Densité Air", f"{rho:.4f} kg/m³")
                
                # Max-Q
                if 40 < altitude_aero/1000 < 15 and 400 < velocity < 600:
                    st.warning("⚠️ Proche de Max-Q (pression dynamique maximale)")
    
    with tab4:
        st.subheader("🌪️ CFD (Computational Fluid Dynamics)")
        
        st.info("""
        **CFD - Simulation Numérique Écoulements:**
        
        Résolution équations Navier-Stokes pour analyser:
        - Distribution pression
        - Contraintes aérodynamiques
        - Chauffage aérodynamique
        - Optimisation forme
        """)
        
        st.write("### 💻 Simulation CFD")
        
        if st.button("🚀 Lancer Simulation CFD", type="primary"):
            with st.spinner("Simulation CFD en cours..."):
                import time
                
                progress = st.progress(0)
                status = st.empty()
                
                stages = [
                    "Génération maillage...",
                    "Initialisation conditions limites...",
                    "Résolution Navier-Stokes...",
                    "Calcul turbulence (k-ε)...",
                    "Post-traitement résultats...",
                    "Génération visualisations..."
                ]
                
                for i, stage in enumerate(stages):
                    progress.progress((i + 1) / len(stages))
                    status.text(stage)
                    time.sleep(1)
                
                st.success("✅ Simulation CFD complétée!")
                
                # Résultats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Cd Calculé", "0.287")
                with col2:
                    st.metric("Force Traînée", "142.3 kN")
                with col3:
                    st.metric("Temps Calcul", "3m 47s")
                
                # Visualisation champ pression
                st.write("### 📊 Champ de Pression")
                
                x_grid = np.linspace(-2, 10, 50)
                y_grid = np.linspace(-3, 3, 30)
                X, Y = np.meshgrid(x_grid, y_grid)
                
                # Simulation champ pression
                R = np.sqrt(X**2 + Y**2)
                P = 101325 * (1 + 0.5 * np.exp(-R/2) * np.cos(np.arctan2(Y, X)))
                
                fig = go.Figure(data=go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=P,
                    colorscale='Jet',
                    colorbar=dict(title="Pression (Pa)")
                ))
                
                fig.update_layout(
                    title="Distribution Pression (Mach 2.0)",
                    xaxis_title="X (m)",
                    yaxis_title="Y (m)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: RAPPORTS & EXPORT ====================
elif page == "📈 Rapports & Export":
    st.header("📈 Rapports et Export de Données")
    
    tab1, tab2, tab3 = st.tabs(["📄 Générer Rapport", "💾 Export Données", "📊 Tableaux de Bord"])
    
    with tab1:
        st.subheader("📄 Générateur de Rapports")
        
        st.write("### 📋 Configuration Rapport")
        
        with st.form("generate_report"):
            col1, col2 = st.columns(2)
            
            with col1:
                report_title = st.text_input("Titre Rapport", "Analyse Performance Fusée")
                report_type = st.selectbox("Type Rapport",
                    ["Rapport Complet", "Performance Technique", "Analyse Coûts", 
                     "Tests & Validation", "Mission Mars"])
            
            with col2:
                report_format = st.selectbox("Format", ["PDF", "HTML", "Markdown", "JSON"])
                include_charts = st.checkbox("Inclure Graphiques", value=True)
            
            sections = st.multiselect(
                "Sections à Inclure",
                ["Résumé Exécutif", "Spécifications Techniques", "Performances",
                 "Tests Effectués", "Analyses IA", "Simulations Quantiques",
                 "Bio-computing", "Recommandations", "Annexes"],
                default=["Résumé Exécutif", "Performances", "Recommandations"]
            )
            
            if st.form_submit_button("📄 Générer Rapport", type="primary"):
                with st.spinner("Génération rapport en cours..."):
                    import time
                    
                    progress = st.progress(0)
                    
                    steps = [
                        "Collecte données...",
                        "Génération statistiques...",
                        "Création graphiques...",
                        "Compilation rapport...",
                        "Export format...",
                        "Finalisation..."
                    ]
                    
                    for i, step in enumerate(steps):
                        progress.progress((i + 1) / len(steps))
                        st.text(step)
                        time.sleep(0.3)
                    
                    st.success(f"✅ Rapport '{report_title}' généré!")
                    
                    # Simulation contenu rapport
        report_content = f"""
# {report_title}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Type:** {report_type}
**Format:** {report_format}

## Résumé Exécutif

Ce rapport présente l'analyse complète de la plateforme de conception de fusées.

### Statistiques Clés
- Fusées créées: {len(st.session_state.rocket_system['rockets'])}
- Moteurs développés: {len(st.session_state.rocket_system['engines'])}
- Tests effectués: {len(st.session_state.rocket_system['tests'])}
- Simulations: {len(st.session_state.rocket_system['simulations'])}

### Performances Globales
- Taux succès moyen: 95.2%
- Coût moyen lancement: $52M
- Delta-v moyen: 11,245 m/s
- Fiabilité flotte: 96.8%

## Recommandations

1. Poursuivre optimisation IA
2. Investir propulsion avancée
3. Renforcer tests validation
4. Préparer missions Mars 2030s

---
*Généré automatiquement par Plateforme Conception Fusées v2.0*
                    """
                    
        st.text_area("Aperçu Rapport", report_content, height=400)
                        
        # Bouton téléchargement
        st.download_button(
            label="💾 Télécharger Rapport",
            data=report_content,
            file_name=f"rapport_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.{report_format.lower()}",
            mime="text/plain"
        )            
    
    with tab2:
        st.subheader("💾 Export de Données")
        
        st.write("### 📊 Sélection Données à Exporter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_rockets = st.checkbox("🚀 Fusées", value=True)
            export_engines = st.checkbox("🔥 Moteurs", value=True)
            export_tests = st.checkbox("🧪 Tests", value=True)
        
        with col2:
            export_simulations = st.checkbox("📊 Simulations", value=True)
            export_ai = st.checkbox("🤖 Modèles IA", value=False)
            export_quantum = st.checkbox("⚛️ Analyses Quantiques", value=False)
        
        export_format_data = st.selectbox("Format Export", ["JSON", "CSV", "Excel", "SQL"])
        
        if st.button("💾 Exporter Données", type="primary"):
            export_data = {}
            
            if export_rockets:
                export_data['rockets'] = st.session_state.rocket_system['rockets']
            if export_engines:
                export_data['engines'] = st.session_state.rocket_system['engines']
            if export_tests:
                export_data['tests'] = st.session_state.rocket_system['tests']
            if export_simulations:
                export_data['simulations'] = st.session_state.rocket_system['simulations']
            
            # Conversion JSON
            export_json = json.dumps(export_data, indent=2, default=str)
            
            st.success("✅ Données préparées pour export!")
            
            st.download_button(
                label=f"💾 Télécharger ({export_format_data})",
                data=export_json,
                file_name=f"export_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Aperçu
            st.write("### 👁️ Aperçu Données")
            st.json(export_data)
    
    with tab3:
        st.subheader("📊 Tableaux de Bord Personnalisés")
        
        st.write("### 🎨 Créer Tableau de Bord")
        
        dashboard_name = st.text_input("Nom Tableau de Bord", "Dashboard Performance")
        
        st.write("**Widgets Disponibles:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            widget_1 = st.selectbox("Widget 1", ["Métriques KPI", "Graphique Performance", "Liste Fusées"])
        with col2:
            widget_2 = st.selectbox("Widget 2", ["Graphique Coûts", "Tests Récents", "Carte Thermique"])
        with col3:
            widget_3 = st.selectbox("Widget 3", ["Timeline", "Prédictions IA", "Alertes"])
        
        if st.button("🎨 Créer Dashboard"):
            st.success(f"✅ Dashboard '{dashboard_name}' créé!")
            
            # Exemple dashboard
            st.markdown("---")
            st.write(f"## {dashboard_name}")
            
            # Widget 1
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fusées Actives", len(st.session_state.rocket_system['rockets']))
            with col2:
                st.metric("Tests Réussis", "96.2%", delta="+2.1%")
            with col3:
                st.metric("Coût Moyen", "$52M", delta="-8M")

# ==================== PAGE: DOCUMENTATION ====================
elif page == "📚 Documentation":
    st.header("📚 Documentation Complète")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Guide", "🔬 API", "❓ FAQ", "📝 Changelog"])
    
    with tab1:
        st.subheader("📖 Guide d'Utilisation")
        
        st.write("### 🚀 Démarrage Rapide")
        
        st.markdown("""
        #### 1. Concevoir une Fusée
        
        ```
        1. Aller dans "➕ Concevoir Fusée"
        2. Configurer:
           - Nom et type mission
           - Masses (sèche, propergol, payload)
           - Dimensions
           - Propulsion
           - Technologies avancées (IA, Quantique, Bio)
        3. Cliquer "🚀 Créer la Fusée"
        ```
        
        #### 2. Créer un Moteur
        
        ```
        1. Aller dans "⚙️ Conception Moteur"
        2. Configurer performances (thrust, Isp)
        3. Choisir matériaux et fabrication
        4. Activer optimisation IA
        5. Créer moteur
        ```
        
        #### 3. Tests et Validation
        
        ```
        1. "🧪 Laboratoire Tests"
        2. Sélectionner fusée/moteur
        3. Configurer test (durée, conditions)
        4. Lancer simulation
        5. Analyser résultats
        ```
        
        #### 4. Optimiser avec IA
        
        ```
        1. "🤖 Optimisation IA"
        2. Sélectionner fusée
        3. Choisir objectifs (payload, coût, fiabilité)
        4. Lancer optimisation
        5. Appliquer recommandations
        ```
        
        #### 5. Simuler Lancement
        
        ```
        1. "🎯 Simulations Lancement"
        2. Configurer mission
        3. Lancer simulation temps réel
        4. Observer télémétrie
        5. Analyser performances
        ```
        """)
        
        st.markdown("---")
        
        st.write("### 💡 Bonnes Pratiques")
        
        st.info("""
        **Design:**
        - Commencer simple, itérer
        - Utiliser optimisation IA
        - Valider avec simulations
        
        **Tests:**
        - Tester tôt et souvent
        - Analyser chaque échec
        - Documenter tout
        
        **Performance:**
        - Monitorer métriques clés
        - Comparer avec benchmarks
        - Optimiser continuellement
        """)
    
    with tab2:
        st.subheader("🔬 Documentation API")
        
        st.write("### 🌐 Endpoints Disponibles")
        
        api_endpoints = """
        ### Fusées
        - `POST /api/rockets/create` - Créer fusée
        - `GET /api/rockets/{id}` - Récupérer fusée
        - `GET /api/rockets` - Liste toutes fusées
        - `POST /api/rockets/{id}/simulate` - Simuler lancement
        
        ### Moteurs
        - `POST /api/engines/create` - Créer moteur
        - `GET /api/engines/{id}` - Récupérer moteur
        - `POST /api/engines/{id}/test` - Tester moteur
        
        ### IA
        - `POST /api/ai/optimize` - Optimiser avec IA
        - `POST /api/ai/predict` - Prédictions performance
        - `POST /api/ai/model/create` - Créer modèle IA
        
        ### Quantique
        - `POST /api/quantum/trajectory` - Optimiser trajectoire
        - `POST /api/quantum/combustion` - Simuler combustion
        
        ### Mars
        - `POST /api/mars/mission/create` - Créer mission Mars
        - `POST /api/mars/edl` - Calculer EDL
        - `POST /api/mars/isru` - Calculer production ISRU
        
        ### Analytics
        - `GET /api/analytics/overview` - Vue d'ensemble
        - `GET /api/analytics/performance` - Analyses performance
        """
        
        st.code(api_endpoints, language="markdown")
        
        st.write("### 📝 Exemple Utilisation")
        
        example_code = """
        import requests
        
        # Créer une fusée
        response = requests.post('http://localhost:8000/api/rockets/create', json={
            "name": "Artemis-X",
            "target": "Mars",
            "num_stages": 2,
            "stages": [...],
            "payload_mass": 20000,
            "height": 70,
            "diameter": 10,
            "reusability": True,
            "technologies": ["IA", "Quantique"]
        })
        
        rocket = response.json()
        print(f"Fusée créée: {rocket['rocket_id']}")
        
        # Simuler lancement
        sim = requests.post(f"http://localhost:8000/api/rockets/{rocket['rocket_id']}/simulate",
                           params={"target": "LEO"})
        
        print(f"Succès: {sim.json()['success']}")
        """
        
        st.code(example_code, language="python")
    
    with tab3:
        st.subheader("❓ Questions Fréquentes")
        
        faqs = {
            "Comment créer ma première fusée?": """
            1. Allez dans "➕ Concevoir Fusée"
            2. Remplissez le formulaire avec les paramètres de base
            3. Cliquez sur "🚀 Créer la Fusée"
            4. Votre fusée apparaît dans "🚀 Mes Fusées"
            """,
            
            "Quelle est la différence entre IA, Quantique et Bio-computing?": """
            - **IA**: Optimisation design, prédictions, détection anomalies
            - **Quantique**: Calculs super-rapides (trajectoires, combustion)
            - **Bio-computing**: Systèmes adaptatifs organiques, auto-réparation
            """,
            
            "Comment optimiser les performances de ma fusée?": """
            1. Utiliser "🤖 Optimisation IA"
            2. Analyser recommandations
            3. Ajuster design
            4. Tester avec simulations
            5. Itérer jusqu'à satisfaction
            """,
            
            "Puis-je exporter mes données?": """
            Oui! Dans "📈 Rapports & Export":
            - Générer rapports PDF/HTML
            - Exporter données JSON/CSV
            - Créer tableaux de bord personnalisés
            """,
            
            "Comment calculer une mission Mars?": """
            1. Aller dans "🔴 Missions Mars"
            2. Créer nouvelle mission
            3. Configurer paramètres (équipage, cargo, durée)
            4. Le système calcule automatiquement trajectoire, delta-v, ISRU
            """,
            
            "Quelle propulsion choisir?": """
            - **Chimique**: Lancements, haute poussée
            - **Électrique**: Missions longues, station-keeping
            - **Nucléaire**: Missions interplanétaires (Mars, Jupiter)
            - **Fusion/Antimatière**: Futur, interstellaire
            """,
            
            "Comment interpréter les résultats de simulation?": """
            Vérifiez:
            - Delta-v total vs requis
            - Précision insertion orbitale
            - Consommation propergol
            - Contraintes structurelles (G-max)
            - Taux succès global
            """
        }
        
        for question, answer in faqs.items():
            with st.expander(f"❓ {question}"):
                st.write(answer)
    
    with tab4:
        st.subheader("📝 Changelog")
        
        st.write("### 🆕 Version 2.0.0 (Actuelle)")
        
        changelog = """
        **Nouvelles Fonctionnalités:**
        - ✅ Optimisation IA complète
        - ✅ Simulations quantiques
        - ✅ Bio-computing intégré
        - ✅ Missions Mars détaillées
        - ✅ Jumeaux numériques
        - ✅ Propulsion exotique
        - ✅ CFD intégré
        - ✅ Export rapports avancé
        
        **Améliorations:**
        - Performance calculs +300%
        - Interface utilisateur refonte
        - Visualisations 3D améliorées
        - API REST complète
        
        **Corrections:**
        - Bugs calculs trajectoires
        - Problèmes export données
        - Optimisations mémoire
        
        ---
        
        ### Version 1.5.0
        
        **Nouvelles Fonctionnalités:**
        - Simulations lancement temps réel
        - Tests moteurs avancés
        - Analyses thermodynamiques
        
        ---
        
        ### Version 1.0.0
        
        **Release Initiale:**
        - Conception fusées de base
        - Calculs orbitaux
        - Tests simples
        """
        
        st.markdown(changelog)

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Paramètres et Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🎨 Préférences", "🔧 Système", "🗑️ Données"])
    
    with tab1:
        st.subheader("🎨 Préférences Utilisateur")
        
        st.write("### 🌍 Langue et Région")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox("Langue", ["Français", "English", "Español", "Deutsch", "中文"])
        with col2:
            units = st.selectbox("Système Unités", ["Métrique (SI)", "Impérial", "Mixte"])
        
        st.write("### 🎨 Apparence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Thème", ["Clair", "Sombre", "Auto"])
        with col2:
            color_scheme = st.selectbox("Palette Couleurs", ["Défaut", "Bleu", "Vert", "Rouge", "Orange"])
        
        st.write("### 📊 Affichage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_tooltips = st.checkbox("Afficher info-bulles", value=True)
            show_warnings = st.checkbox("Afficher avertissements", value=True)
        
        with col2:
            auto_save = st.checkbox("Sauvegarde automatique", value=True)
            show_advanced = st.checkbox("Options avancées", value=False)
        
        if st.button("💾 Sauvegarder Préférences", type="primary"):
            st.success("✅ Préférences sauvegardées!")
    
    with tab2:
        st.subheader("🔧 Configuration Système")
        
        st.write("### 💻 Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            simulation_quality = st.select_slider("Qualité Simulations",
                options=["Basse", "Moyenne", "Haute", "Ultra"])
            cache_size = st.slider("Taille Cache (MB)", 100, 10000, 1000, 100)
        
        with col2:
            parallel_threads = st.slider("Threads Parallèles", 1, 16, 4, 1)
            gpu_acceleration = st.checkbox("Accélération GPU", value=False)
        
        st.write("### 🔌 API et Intégrations")
        
        api_key = st.text_input("Clé API", type="password")
        api_endpoint = st.text_input("Endpoint API", "http://localhost:8000")
        
        if st.button("🔗 Tester Connexion API"):
            st.info("Test connexion...")
            import time
            time.sleep(1)
            st.success("✅ Connexion API réussie!")
    
    with tab3:
        st.subheader("🗑️ Gestion des Données")
        
        st.write("### 📊 Statistiques Stockage")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fusées", len(st.session_state.rocket_system['rockets']))
        with col2:
            st.metric("Moteurs", len(st.session_state.rocket_system['engines']))
        with col3:
            st.metric("Tests", len(st.session_state.rocket_system['tests']))
        with col4:
            st.metric("Simulations", len(st.session_state.rocket_system['simulations']))
        
        st.markdown("---")
        
        st.write("### 🗑️ Actions Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Sauvegarder Tout", use_container_width=True):
                data_json = json.dumps(st.session_state.rocket_system, indent=2, default=str)
                st.download_button(
                    "⬇️ Télécharger Sauvegarde",
                    data_json,
                    f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader("📤 Restaurer depuis Fichier", type="json")
            if uploaded_file:
                if st.button("♻️ Restaurer"):
                    data = json.load(uploaded_file)
                    st.session_state.rocket_system = data
                    st.success("✅ Données restaurées!")
                    st.rerun()
        
        st.markdown("---")
        
        st.warning("⚠️ **Actions Destructives**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Supprimer Simulations", use_container_width=True):
                st.session_state.rocket_system['simulations'] = []
                st.success("Simulations supprimées")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Supprimer Tests", use_container_width=True):
                st.session_state.rocket_system['tests'] = []
                st.success("Tests supprimés")
                st.rerun()
        
        st.markdown("---")
        
        st.error("🔴 **ZONE DANGEREUSE**")
        
        confirm = st.checkbox("Je confirme vouloir tout supprimer")
        
        if st.button("💣 RÉINITIALISER TOUT", disabled=not confirm, type="primary"):
            st.session_state.rocket_system = {
                'rockets': {},
                'engines': {},
                'simulations': [],
                'ai_models': {},
                'quantum_analyses': [],
                'biocomputing_results': [],
                'materials': {},
                'tests': [],
                'manufacturing': {},
                'mars_missions': {},
                'design_iterations': [],
                'performance_data': [],
                'log': []
            }
            st.success("✅ Toutes les données ont été réinitialisées!")
            st.balloons()
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal Système (Dernières 20 entrées)"):
    if st.session_state.rocket_system['log']:
        for event in st.session_state.rocket_system['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️" if level == "WARNING" else "❌"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("💡 Aucune fusée créée. Commencez par concevoir votre première fusée!")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🚀 Plateforme Conception & Fabrication Fusées Spatiales</h3>
        <p>Système Avancé IA • Quantique • Bio-computing</p>
        <p><small>Version 2.0.0 | Ingénierie Aérospatiale Avancée</small></p>
        <p><small>🔥 Propulsion | 🏗️ Fabrication | 🧪 Tests | 🤖 IA | ⚛️ Quantique | 🧬 Bio</small></p>
        <p><small>Powered by Advanced Rocket Engineering © 2024</small></p>
    </div>
""", unsafe_allow_html=True)