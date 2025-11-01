"""Interface Streamlit pour la Plateforme Supercalculateur Quantique-Biologique
Système complet pour créer, développer, fabriquer, tester et déployer des supercalculateurs
streamlit run supercalculateur_app.py
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
    page_title="⚡ Plateforme Supercalculateur Quantique-Biologique",
    page_icon="⚡",
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
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .supercomputer-card {
        border: 3px solid #f093fb;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(240, 147, 251, 0.1) 0%, rgba(245, 87, 108, 0.1) 100%);
        box-shadow: 0 4px 12px rgba(240, 147, 251, 0.3);
    }
    .performance-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        margin: 0.3rem;
    }
    .yottaflops {
        background: linear-gradient(90deg, #ff0080 0%, #ff8c00 100%);
        color: white;
    }
    .zettaflops {
        background: linear-gradient(90deg, #ffd700 0%, #ff6347 100%);
        color: white;
    }
    .exaflops {
        background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
        color: white;
    }
    .petaflops {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .fabrication-step {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #f093fb;
    }
    .fabrication-step.active {
        border-left-color: #ffc107;
        background: #fff3cd;
    }
    .fabrication-step.completed {
        border-left-color: #28a745;
        background: #d4edda;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION SESSION STATE ====================

if 'supercomputer_system' not in st.session_state:
    st.session_state.supercomputer_system = {
        'supercomputers': {},
        'fabrications': [],
        'benchmarks': [],
        'deployments': {},
        'projects': {},
        'jobs': [],
        'maintenance_logs': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def create_supercomputer_mock(name, sc_type, config):
    """Crée un supercalculateur simulé"""
    sc_id = f"sc_{len(st.session_state.supercomputer_system['supercomputers']) + 1}"
    
    st.session_state.supercomputer_system['supercomputers'][sc_id] = {
        'id': sc_id,
        'name': name,
        'type': sc_type,
        'created_at': datetime.now().isoformat(),
        'status': 'offline',
        'health': 1.0,
        'nodes': config.get('nodes', 1000),
        'cores_per_node': config.get('cores_per_node', 128),
        'total_cores': config.get('nodes', 1000) * config.get('cores_per_node', 128),
        'memory_total': config.get('nodes', 1000) * 512,  # GB
        'peak_performance': 0.0,
        'sustained_performance': 0.0,
        'performance_class': 'petaflops',
        'efficiency': config.get('efficiency', 0.85),
        'quantum_system': {
            'qubits': config.get('qubits', 10000),
            'coherence_time': np.random.random() * 1000,
            'gate_fidelity': 0.999,
            'quantum_volume': config.get('qubits', 10000) * 100
        } if sc_type in ['supercalculateur_quantique', 'supercalculateur_hybride', 'supercalculateur_conscient'] else None,
        'biological_system': {
            'neurons': config.get('neurons', 100000000000),
            'synapses': config.get('neurons', 100000000000) * 10,
            'plasticity': 0.9,
            'bio_efficiency': 0.95
        } if sc_type in ['supercalculateur_biologique', 'supercalculateur_hybride', 'ordinateur_adn', 'supercalculateur_conscient'] else None,
        'network': {
            'interconnect': 'infiniband',
            'bandwidth': 200,
            'latency': 1.0
        },
        'storage': {
            'capacity': 100000,  # TB
            'type': 'nvme',
            'read_speed': 50,
            'write_speed': 40
        },
        'cooling': {
            'system': config.get('cooling', 'refroidissement_liquide'),
            'temperature': 20,
            'pue': 1.1
        },
        'power': {
            'consumption': config.get('power_mw', 10.0),
            'renewable_percentage': config.get('renewable', 0.5),
            'efficiency': 0.85
        },
        'consciousness': {
            'level': config.get('consciousness_level', 0.0),
            'enabled': config.get('enable_consciousness', False)
        } if config.get('enable_consciousness') else None,
        'benchmarks': {
            'linpack': 0.0,
            'hpcg': 0.0,
            'green500': 0.0,
            'quantum': 0.0
        },
        'jobs_completed': 0,
        'uptime': 0.0,
        'utilization': 0.0
    }
    
    # Calculer la performance
    sc = st.session_state.supercomputer_system['supercomputers'][sc_id]
    base_flops = sc['total_cores'] * 2.5e9 * 4  # GHz * threads
    
    if sc['quantum_system']:
        base_flops += sc['quantum_system']['qubits'] * 1e12
    
    if sc['biological_system']:
        base_flops += sc['biological_system']['neurons'] * 1000
    
    sc['peak_performance'] = base_flops
    sc['sustained_performance'] = base_flops * sc['efficiency']
    
    # Classe de performance
    if base_flops >= 1e24:
        sc['performance_class'] = 'yottaflops'
    elif base_flops >= 1e21:
        sc['performance_class'] = 'zettaflops'
    elif base_flops >= 1e18:
        sc['performance_class'] = 'exaflops'
    else:
        sc['performance_class'] = 'petaflops'
    
    log_event(f"Supercalculateur créé: {name} ({sc_type})")
    return sc_id

def log_event(message: str):
    """Enregistre un événement"""
    st.session_state.supercomputer_system['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })

def get_performance_badge(perf_class: str) -> str:
    """Retourne un badge HTML pour la classe de performance"""
    badges = {
        'yottaflops': '<span class="performance-badge yottaflops">🚀 YOTTAFLOPS</span>',
        'zettaflops': '<span class="performance-badge zettaflops">⚡ ZETTAFLOPS</span>',
        'exaflops': '<span class="performance-badge exaflops">💫 EXAFLOPS</span>',
        'petaflops': '<span class="performance-badge petaflops">⭐ PETAFLOPS</span>'
    }
    return badges.get(perf_class, badges['petaflops'])

def format_flops(flops: float) -> str:
    """Formate les FLOPS en unité lisible"""
    if flops >= 1e24:
        return f"{flops/1e24:.2f} YFLOPS"
    elif flops >= 1e21:
        return f"{flops/1e21:.2f} ZFLOPS"
    elif flops >= 1e18:
        return f"{flops/1e18:.2f} EFLOPS"
    elif flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPS"
    else:
        return f"{flops/1e12:.2f} TFLOPS"

# ==================== HEADER ====================

st.markdown('<h1 class="main-header">⚡ Plateforme Supercalculateur Quantique-Biologique</h1>', unsafe_allow_html=True)
st.markdown("### Système complet de création, développement, fabrication et déploiement de supercalculateurs avancés")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/f093fb/ffffff?text=Supercomputer+Hub", use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Tableau de Bord",
            "⚡ Mes Supercalculateurs",
            "➕ Créer Supercalculateur",
            "🏭 Fabrication",
            "🔧 Configuration & Optimisation",
            "📊 Benchmarking & TOP500",
            "💼 Gestion des Jobs",
            "🚀 Déploiement",
            "📁 Projets",
            "🌍 Efficacité Énergétique",
            "🔬 Recherche & Applications",
            "📚 Bibliothèque Technique",
            "⚙️ Maintenance & Monitoring"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques Globales")
    
    total_sc = len(st.session_state.supercomputer_system['supercomputers'])
    active_sc = sum(1 for sc in st.session_state.supercomputer_system['supercomputers'].values() if sc['status'] == 'online')
    total_jobs = len(st.session_state.supercomputer_system['jobs'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚡ Supercalculateurs", total_sc)
        st.metric("📁 Projets", len(st.session_state.supercomputer_system['projects']))
    with col2:
        st.metric("✅ Actifs", active_sc)
        st.metric("💼 Jobs", total_jobs)
    
    # Performance totale
    if st.session_state.supercomputer_system['supercomputers']:
        total_flops = sum(sc['peak_performance'] for sc in st.session_state.supercomputer_system['supercomputers'].values())
        st.markdown(f"### 🚀 Performance Totale")
        st.write(format_flops(total_flops))

# ==================== PAGE: TABLEAU DE BORD ====================

if page == "🏠 Tableau de Bord":
    st.header("📊 Tableau de Bord Principal")
    
    # Métriques principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="supercomputer-card"><h2>⚡</h2><h3>{}</h3><p>Supercalculateurs</p></div>'.format(total_sc), unsafe_allow_html=True)
    
    with col2:
        if st.session_state.supercomputer_system['supercomputers']:
            avg_efficiency = np.mean([sc['efficiency'] for sc in st.session_state.supercomputer_system['supercomputers'].values()])
            st.markdown('<div class="supercomputer-card"><h2>⚙️</h2><h3>{:.0%}</h3><p>Efficacité Moyenne</p></div>'.format(avg_efficiency), unsafe_allow_html=True)
        else:
            st.markdown('<div class="supercomputer-card"><h2>⚙️</h2><h3>N/A</h3><p>Efficacité Moyenne</p></div>', unsafe_allow_html=True)
    
    with col3:
        if st.session_state.supercomputer_system['supercomputers']:
            total_cores = sum(sc['total_cores'] for sc in st.session_state.supercomputer_system['supercomputers'].values())
            st.markdown('<div class="supercomputer-card"><h2>🔲</h2><h3>{:,}</h3><p>Cœurs Totaux</p></div>'.format(total_cores), unsafe_allow_html=True)
        else:
            st.markdown('<div class="supercomputer-card"><h2>🔲</h2><h3>0</h3><p>Cœurs Totaux</p></div>', unsafe_allow_html=True)
    
    with col4:
        completed_jobs = sum(sc['jobs_completed'] for sc in st.session_state.supercomputer_system['supercomputers'].values())
        st.markdown('<div class="supercomputer-card"><h2>✅</h2><h3>{}</h3><p>Jobs Complétés</p></div>'.format(completed_jobs), unsafe_allow_html=True)
    
    with col5:
        if st.session_state.supercomputer_system['supercomputers']:
            total_power = sum(sc['power']['consumption'] for sc in st.session_state.supercomputer_system['supercomputers'].values())
            st.markdown('<div class="supercomputer-card"><h2>⚡</h2><h3>{:.1f} MW</h3><p>Consommation</p></div>'.format(total_power), unsafe_allow_html=True)
        else:
            st.markdown('<div class="supercomputer-card"><h2>⚡</h2><h3>0 MW</h3><p>Consommation</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques
    if st.session_state.supercomputer_system['supercomputers']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribution par Type")
            
            type_counts = {}
            for sc in st.session_state.supercomputer_system['supercomputers'].values():
                sc_type = sc['type'].replace('_', ' ').title()
                type_counts[sc_type] = type_counts.get(sc_type, 0) + 1
            
            fig = px.pie(values=list(type_counts.values()), names=list(type_counts.keys()),
                        color_discrete_sequence=px.colors.sequential.Reds_r)
            fig.update_layout(title="Répartition des Supercalculateurs")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🚀 Performances (PFLOPS)")
            
            names = [sc['name'][:20] for sc in st.session_state.supercomputer_system['supercomputers'].values()]
            performances = [sc['peak_performance'] / 1e15 for sc in st.session_state.supercomputer_system['supercomputers'].values()]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=performances, marker_color='rgb(240, 147, 251)')
            ])
            fig.update_layout(title="Performance de Crête", yaxis_title="PFLOPS")
            st.plotly_chart(fig, use_container_width=True)
        
        # Classement TOP500 simulé
        st.markdown("---")
        st.subheader("🏆 Classement Performance")
        
        ranking_data = []
        for sc in sorted(st.session_state.supercomputer_system['supercomputers'].values(), 
                        key=lambda x: x['peak_performance'], reverse=True):
            ranking_data.append({
                'Rang': len(ranking_data) + 1,
                'Nom': sc['name'],
                'Performance Crête': format_flops(sc['peak_performance']),
                'Performance Soutenue': format_flops(sc['sustained_performance']),
                'Efficacité': f"{sc['efficiency']:.0%}",
                'Cœurs': f"{sc['total_cores']:,}",
                'Classe': sc['performance_class'].replace('_', ' ').upper()
            })
        
        df_ranking = pd.DataFrame(ranking_data)
        st.dataframe(df_ranking, use_container_width=True)
    else:
        st.info("💡 Aucun supercalculateur créé. Commencez par créer votre premier système!")

# ==================== PAGE: MES SUPERCALCULATEURS ====================

elif page == "⚡ Mes Supercalculateurs":
    st.header("⚡ Gestion des Supercalculateurs")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.info("💡 Aucun supercalculateur créé. Créez votre premier système pour commencer!")
    else:
        for sc_id, sc in st.session_state.supercomputer_system['supercomputers'].items():
            st.markdown(f'<div class="supercomputer-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"### ⚡ {sc['name']}")
                st.markdown(get_performance_badge(sc['performance_class']), unsafe_allow_html=True)
                st.caption(f"Type: {sc['type'].replace('_', ' ').title()}")
            
            with col2:
                st.metric("Performance Crête", format_flops(sc['peak_performance']))
                st.metric("Performance Soutenue", format_flops(sc['sustained_performance']))
            
            with col3:
                st.metric("Efficacité", f"{sc['efficiency']:.0%}")
                st.metric("Santé", f"{sc['health']:.0%}")
            
            with col4:
                status_icon = "🟢" if sc['status'] == 'online' else "🔴"
                st.write(f"**Statut:** {status_icon} {sc['status'].upper()}")
                st.write(f"**Jobs:** {sc['jobs_completed']}")
            
            with st.expander("📋 Détails Complets", expanded=False):
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏗️ Architecture", "⚛️ Quantique", "🧬 Biologique", "🌐 Réseau", "⚡ Énergie"])
                
                with tab1:
                    st.subheader("Architecture Matérielle")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Nœuds:** {sc['nodes']:,}")
                        st.write(f"**Cœurs/Nœud:** {sc['cores_per_node']}")
                        st.write(f"**Total Cœurs:** {sc['total_cores']:,}")
                    
                    with col2:
                        st.write(f"**Mémoire Totale:** {sc['memory_total']:,} GB")
                        st.write(f"**Stockage:** {sc['storage']['capacity']:,} TB")
                        st.write(f"**Type Stockage:** {sc['storage']['type'].upper()}")
                    
                    with col3:
                        st.write(f"**Vitesse Lecture:** {sc['storage']['read_speed']} GB/s")
                        st.write(f"**Vitesse Écriture:** {sc['storage']['write_speed']} GB/s")
                
                with tab2:
                    if sc['quantum_system']:
                        st.subheader("Système Quantique")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Qubits", f"{sc['quantum_system']['qubits']:,}")
                        with col2:
                            st.metric("Volume Quantique", f"{sc['quantum_system']['quantum_volume']:,}")
                        with col3:
                            st.metric("Fidélité", f"{sc['quantum_system']['gate_fidelity']:.3f}")
                        
                        st.progress(sc['quantum_system']['gate_fidelity'], text=f"Qualité Quantique: {sc['quantum_system']['gate_fidelity']:.1%}")
                    else:
                        st.info("Pas de système quantique")
                
                with tab3:
                    if sc['biological_system']:
                        st.subheader("Système Biologique")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Neurones", f"{sc['biological_system']['neurons']:,}")
                        with col2:
                            st.metric("Synapses", f"{sc['biological_system']['synapses']:,}")
                        with col3:
                            st.metric("Plasticité", f"{sc['biological_system']['plasticity']:.0%}")
                        
                        st.progress(sc['biological_system']['bio_efficiency'], text=f"Efficacité Bio: {sc['biological_system']['bio_efficiency']:.0%}")
                    else:
                        st.info("Pas de système biologique")
                
                with tab4:
                    st.subheader("Réseau et Interconnexion")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Interconnexion:** {sc['network']['interconnect'].upper()}")
                        st.write(f"**Bande Passante:** {sc['network']['bandwidth']} Gbps")
                    
                    with col2:
                        st.write(f"**Latence:** {sc['network']['latency']} μs")
                
                with tab5:
                    st.subheader("Énergie et Refroidissement")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Consommation", f"{sc['power']['consumption']:.1f} MW")
                        st.metric("Renouvelable", f"{sc['power']['renewable_percentage']:.0%}")
                        st.metric("Efficacité", f"{sc['power']['efficiency']:.0%}")
                    
                    with col2:
                        st.write(f"**Refroidissement:** {sc['cooling']['system'].replace('_', ' ').title()}")
                        st.write(f"**Température:** {sc['cooling']['temperature']}°C")
                        st.write(f"**PUE:** {sc['cooling']['pue']:.2f}")
                
                # Actions
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    if st.button(f"▶️ {'Éteindre' if sc['status'] == 'online' else 'Démarrer'}", key=f"toggle_{sc_id}"):
                        sc['status'] = 'offline' if sc['status'] == 'online' else 'online'
                        log_event(f"{sc['name']} {'éteint' if sc['status'] == 'offline' else 'démarré'}")
                        st.rerun()
                
                with col2:
                    if st.button(f"🔧 Optimiser", key=f"opt_{sc_id}"):
                        sc['efficiency'] = min(0.98, sc['efficiency'] * 1.05)
                        st.success("Optimisation appliquée!")
                        st.rerun()
                
                with col3:
                    if st.button(f"📊 Benchmark", key=f"bench_{sc_id}"):
                        st.info("Allez dans l'onglet Benchmarking")
                
                with col4:
                    if st.button(f"🔬 Diagnostiquer", key=f"diag_{sc_id}"):
                        issues = []
                        if sc['cooling']['temperature'] > 30:
                            issues.append("⚠️ Température élevée")
                        if sc['health'] < 0.95:
                            issues.append(f"⚠️ Santé: {sc['health']:.0%}")
                        
                        if issues:
                            for issue in issues:
                                st.warning(issue)
                        else:
                            st.success("✅ Système en bon état")
                
                with col5:
                    if st.button(f"🗑️ Supprimer", key=f"del_{sc_id}"):
                        del st.session_state.supercomputer_system['supercomputers'][sc_id]
                        log_event(f"{sc['name']} supprimé")
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: CRÉER SUPERCALCULATEUR ====================

elif page == "➕ Créer Supercalculateur":
    st.header("➕ Créer un Nouveau Supercalculateur")
    
    with st.form("create_supercomputer_form"):
        st.subheader("🎨 Configuration de Base")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sc_name = st.text_input("📝 Nom du Supercalculateur", placeholder="Ex: Titan-Quantum-X")
            sc_type = st.selectbox(
                "🧬 Type de Supercalculateur",
                [
                    "supercalculateur_quantique",
                    "supercalculateur_biologique",
                    "supercalculateur_hybride",
                    "exascale",
                    "zettascale",
                    "neuromorphique",
                    "photonique",
                    "ordinateur_adn",
                    "supercalculateur_conscient"
                ],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            target_performance = st.selectbox(
                "🎯 Performance Cible",
                ["Petaflops", "Exaflops", "Zettaflops", "Yottaflops"]
            )
            efficiency_target = st.slider("⚙️ Efficacité Cible", 0.7, 0.98, 0.85, 0.01)
        
        st.markdown("---")
        st.subheader("🏗️ Architecture Matérielle")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nodes = st.number_input("Nombre de Nœuds", 100, 1000000, 10000)
            cores_per_node = st.number_input("Cœurs par Nœud", 32, 512, 128)
        
        with col2:
            memory_per_node = st.number_input("Mémoire/Nœud (GB)", 128, 2048, 512)
            storage_total = st.number_input("Stockage Total (TB)", 1000, 1000000, 100000)
        
        with col3:
            interconnect = st.selectbox("Interconnexion", ["InfiniBand", "Ethernet 100G", "Omni-Path", "Photonique"])
            storage_type = st.selectbox("Type Stockage", ["HDD", "SSD", "NVMe", "Optane", "ADN"])
        
        st.markdown("---")
        st.subheader("⚛️ Composants Quantiques")
        
        if sc_type in ['supercalculateur_quantique', 'supercalculateur_hybride', 'supercalculateur_conscient']:
            col1, col2 = st.columns(2)
            with col1:
                qubits = st.number_input("Nombre de Qubits", 100, 100000, 10000)
            with col2:
                quantum_topology = st.selectbox("Topologie Quantique", ["All-to-all", "Linear", "Grid", "Star"])
        else:
            qubits = 0
            st.info("Type non-quantique sélectionné")
        
        st.markdown("---")
        st.subheader("🧬 Composants Biologiques")
        
        if sc_type in ['supercalculateur_biologique', 'supercalculateur_hybride', 'ordinateur_adn', 'supercalculateur_conscient']:
            col1, col2 = st.columns(2)
            with col1:
                neurons = st.number_input("Neurones (Milliards)", 1, 1000, 100) * 1000000000
            with col2:
                neural_substrate = st.selectbox("Substrat Neuronal", ["Organoïdes", "Cultures 3D", "ADN", "Hybride"])
        else:
            neurons = 0
            st.info("Type non-biologique sélectionné")
        
        st.markdown("---")
        st.subheader("❄️ Refroidissement et Énergie")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cooling_system = st.selectbox(
                "Système de Refroidissement",
                ["Air", "Liquide", "Immersion", "Cryogénique", "Quantique", "Bio-thermique", "Hybride"]
            )
        
        with col2:
            power_mw = st.number_input("Consommation (MW)", 1.0, 100.0, 10.0, 0.5)
            renewable = st.slider("% Énergie Renouvelable", 0, 100, 50) / 100
        
        with col3:
            pue_target = st.number_input("PUE Cible", 1.05, 2.0, 1.1, 0.05)
        
        st.markdown("---")
        st.subheader("🧠 Intelligence et Conscience")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_consciousness = st.checkbox("🧠 Activer la Conscience")
            if enable_consciousness:
                consciousness_level = st.slider("Niveau de Conscience", 0.0, 1.0, 0.7, 0.1)
        
        with col2:
            enable_agi = st.checkbox("🤖 Intégrer AGI")
            enable_self_optimization = st.checkbox("⚡ Auto-optimisation")
        
        submitted = st.form_submit_button("🚀 Créer le Supercalculateur", use_container_width=True, type="primary")
        
        if submitted:
            if not sc_name:
                st.error("⚠️ Veuillez donner un nom au supercalculateur")
            else:
                with st.spinner("🔄 Création du supercalculateur en cours..."):
                    config = {
                        'nodes': nodes,
                        'cores_per_node': cores_per_node,
                        'memory_per_node': memory_per_node,
                        'storage': storage_total,
                        'efficiency': efficiency_target,
                        'qubits': qubits,
                        'neurons': neurons,
                        'cooling': f"refroidissement_{cooling_system.lower()}",
                        'power_mw': power_mw,
                        'renewable': renewable,
                        'enable_consciousness': enable_consciousness,
                        'consciousness_level': consciousness_level if enable_consciousness else 0.0
                    }
                    
                    sc_id = create_supercomputer_mock(sc_name, sc_type, config)
                    
                    st.success(f"✅ Supercalculateur '{sc_name}' créé avec succès!")
                    st.balloons()
                    
                    sc = st.session_state.supercomputer_system['supercomputers'][sc_id]
                    
                    # Afficher les spécifications
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Performance Crête", format_flops(sc['peak_performance']))
                    with col2:
                        st.metric("Total Cœurs", f"{sc['total_cores']:,}")
                    with col3:
                        st.metric("Classe", sc['performance_class'].upper())
                    with col4:
                        st.metric("Efficacité", f"{sc['efficiency']:.0%}")
                    
                    st.code(f"ID: {sc_id}", language="text")

# ==================== PAGE: EFFICACITÉ ÉNERGÉTIQUE ====================

elif page == "🌍 Efficacité Énergétique":
    st.header("🌍 Centre d'Efficacité Énergétique")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.info("Aucun supercalculateur disponible")
    else:
        tab1, tab2, tab3 = st.tabs(["⚡ Consommation", "♻️ Énergies Renouvelables", "📊 Optimisation"])
        
        with tab1:
            st.subheader("⚡ Analyse de la Consommation")
            
            # Métriques globales
            total_power = sum(sc['power']['consumption'] for sc in st.session_state.supercomputer_system['supercomputers'].values())
            avg_pue = np.mean([sc['cooling']['pue'] for sc in st.session_state.supercomputer_system['supercomputers'].values()])
            avg_renewable = np.mean([sc['power']['renewable_percentage'] for sc in st.session_state.supercomputer_system['supercomputers'].values()])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Consommation Totale", f"{total_power:.1f} MW")
            
            with col2:
                st.metric("PUE Moyen", f"{avg_pue:.2f}")
            
            with col3:
                st.metric("Renouvelable Moyen", f"{avg_renewable:.0%}")
            
            with col4:
                carbon_footprint = total_power * 8760 * (1 - avg_renewable) * 0.5  # tonnes CO2/an
                st.metric("Empreinte Carbone", f"{carbon_footprint:,.0f} t CO2/an")
            
            # Graphiques
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                names = [sc['name'][:15] for sc in st.session_state.supercomputer_system['supercomputers'].values()]
                power_consumption = [sc['power']['consumption'] for sc in st.session_state.supercomputer_system['supercomputers'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=names, y=power_consumption, marker_color='rgb(240, 147, 251)')
                ])
                fig.update_layout(title="Consommation par Supercalculateur (MW)", yaxis_title="MW")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                pue_values = [sc['cooling']['pue'] for sc in st.session_state.supercomputer_system['supercomputers'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=names, y=pue_values, marker_color='rgb(245, 87, 108)')
                ])
                fig.update_layout(title="PUE par Système", yaxis_title="PUE")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("♻️ Transition Énergies Renouvelables")
            
            # for sc in st.session_state.supercomputer_system['supercomputers'].values():
            for i, sc in enumerate(st.session_state.supercomputer_system['supercomputers'].values()):
                with st.expander(f"⚡ {sc['name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Consommation", f"{sc['power']['consumption']:.1f} MW")
                        st.metric("Part Renouvelable", f"{sc['power']['renewable_percentage']:.0%}")
                        
                        renewable_mw = sc['power']['consumption'] * sc['power']['renewable_percentage']
                        fossil_mw = sc['power']['consumption'] * (1 - sc['power']['renewable_percentage'])
                        
                        st.write(f"**Renouvelable:** {renewable_mw:.2f} MW")
                        st.write(f"**Fossile:** {fossil_mw:.2f} MW")
                    
                    with col2:
                        new_renewable = st.slider(
                            "Nouvelle part renouvelable",
                            0.0, 1.0, sc['power']['renewable_percentage'],
                            key=f"renewable_{sc['id']}"
                        )
                        
                        if st.button(f"✅ Appliquer", key=f"apply_renewable_{sc['id']}"):
                            sc['power']['renewable_percentage'] = new_renewable
                            st.success("Mise à jour appliquée!")
                            st.rerun()
                    
                    # Graphique évolution CO2
                    years = list(range(2025, 2031))
                    co2_current = [sc['power']['consumption'] * 8760 * (1 - sc['power']['renewable_percentage']) * 0.5 for _ in years]
                    co2_improved = [sc['power']['consumption'] * 8760 * (1 - min(1.0, sc['power']['renewable_percentage'] + 0.1 * i)) * 0.5 for i in range(len(years))]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years, y=co2_current, mode='lines', name='Actuel', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=years, y=co2_improved, mode='lines', name='Objectif +10%/an', line=dict(color='green')))
                    fig.update_layout(title="Projection Empreinte Carbone", xaxis_title="Année", yaxis_title="Tonnes CO2")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{i}")
        
        with tab3:
            st.subheader("📊 Optimisation Énergétique")
            
            st.write("### Actions d'Optimisation Disponibles")
            
            optimizations = {
                "Améliorer PUE": "Optimiser le système de refroidissement (-5% PUE)",
                "Augmenter Renouvelable": "Installer panneaux solaires (+10% renouvelable)",
                "Optimiser Charges": "Équilibrage dynamique des charges (-3% consommation)",
                "Refroidissement Libre": "Utiliser air extérieur quand possible (-8% refroidissement)",
                "Gestion Dynamique": "Éteindre nœuds inutilisés (-5% consommation)"
            }
            
            selected_optimizations = st.multiselect("Sélectionner optimisations", list(optimizations.keys()))
            
            if selected_optimizations:
                for opt in selected_optimizations:
                    st.info(f"**{opt}:** {optimizations[opt]}")
                
                if st.button("⚡ Appliquer Optimisations", use_container_width=True, type="primary"):
                    for sc in st.session_state.supercomputer_system['supercomputers'].values():
                        if "Améliorer PUE" in selected_optimizations:
                            sc['cooling']['pue'] = max(1.05, sc['cooling']['pue'] * 0.95)
                        
                        if "Augmenter Renouvelable" in selected_optimizations:
                            sc['power']['renewable_percentage'] = min(1.0, sc['power']['renewable_percentage'] + 0.1)
                        
                        if "Optimiser Charges" in selected_optimizations or "Gestion Dynamique" in selected_optimizations:
                            sc['power']['consumption'] *= 0.95
                        
                        sc['power']['efficiency'] = min(0.98, sc['power']['efficiency'] * 1.02)
                    
                    st.success(f"✅ {len(selected_optimizations)} optimisation(s) appliquée(s)!")
                    st.balloons()
                    st.rerun()

# ==================== PAGE: RECHERCHE & APPLICATIONS ====================

elif page == "🔬 Recherche & Applications":
    st.header("🔬 Recherche et Applications Avancées")
    
    tab1, tab2, tab3 = st.tabs(["🧪 Domaines de Recherche", "🎯 Projets Actifs", "📈 Impact"])
    
    with tab1:
        st.subheader("🧪 Domaines de Recherche Supportés")
        
        domains = {
            "Recherche Scientifique": {
                "description": "Simulations physiques, chimiques et biologiques avancées",
                "applications": ["Physique des particules", "Chimie quantique", "Biologie structurale"],
                "performance_required": "Exaflops+"
            },
            "Modélisation Climat": {
                "description": "Prédiction et simulation du changement climatique",
                "applications": ["Modèles atmosphériques", "Simulation océanique", "Prévisions long terme"],
                "performance_required": "Multi-Exaflops"
            },
            "Découverte Médicaments": {
                "description": "Screening virtuel et conception de molécules",
                "applications": ["Docking moléculaire", "Prédiction structure protéines", "Optimisation composés"],
                "performance_required": "Exaflops"
            },
            "Intelligence Artificielle": {
                "description": "Entraînement de modèles IA et deep learning",
                "applications": ["LLMs", "Vision par ordinateur", "AGI"],
                "performance_required": "Zettaflops+"
            },
            "Simulation Quantique": {
                "description": "Simulation de systèmes quantiques complexes",
                "applications": ["Cryptographie quantique", "Matériaux quantiques", "Chimie quantique"],
                "performance_required": "Quantum Supremacy"
            },
            "Génomique": {
                "description": "Séquençage et analyse génomique massive",
                "applications": ["Séquençage ADN", "Médecine personnalisée", "Évolution"],
                "performance_required": "Petaflops-Exaflops"
            },
            "Astrophysique": {
                "description": "Simulation univers et phénomènes cosmiques",
                "applications": ["Formation galaxies", "Trous noirs", "Matière noire"],
                "performance_required": "Exaflops+"
            }
        }
        
        for domain_name, info in domains.items():
            with st.expander(f"🔬 {domain_name}", expanded=False):
                st.write(f"**Description:** {info['description']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Applications:**")
                    for app in info['applications']:
                        st.write(f"• {app}")
                
                with col2:
                    st.write(f"**Performance requise:** {info['performance_required']}")
    
    with tab2:
        st.subheader("🎯 Projets de Recherche Actifs")
        
        if st.session_state.supercomputer_system['projects']:
            for project in st.session_state.supercomputer_system['projects'].values():
                st.write(f"**{project['name']}:** {project.get('description', 'Pas de description')}")
        else:
            st.info("Aucun projet de recherche actif")
        
        # Créer un nouveau projet de recherche
        st.markdown("---")
        st.write("### ➕ Nouveau Projet de Recherche")
        
        with st.form("research_project_form"):
            project_name = st.text_input("Nom du Projet")
            domain = st.selectbox("Domaine", list(domains.keys()))
            description = st.text_area("Description")
            duration = st.number_input("Durée (mois)", 1, 120, 12)
            
            if st.form_submit_button("🚀 Créer Projet"):
                if project_name:
                    project_id = f"proj_{len(st.session_state.supercomputer_system['projects']) + 1}"
                    
                    st.session_state.supercomputer_system['projects'][project_id] = {
                        'id': project_id,
                        'name': project_name,
                        'domain': domain,
                        'description': description,
                        'duration': duration,
                        'created_at': datetime.now().isoformat(),
                        'status': 'active'
                    }
                    
                    st.success(f"✅ Projet '{project_name}' créé!")
                    log_event(f"Projet de recherche créé: {project_name}")
                    st.rerun()
    
    with tab3:
        st.subheader("📈 Impact et Découvertes")
        
        if st.session_state.supercomputer_system['supercomputers']:
            total_compute = sum(sc['jobs_completed'] * sc['peak_performance'] for sc in st.session_state.supercomputer_system['supercomputers'].values())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Heures de Calcul", f"{total_compute/1e18:.0f} Exaflop-heures")
            
            with col2:
                publications = np.random.randint(10, 100)
                st.metric("Publications Scientifiques", publications)
            
            with col3:
                discoveries = np.random.randint(5, 50)
                st.metric("Découvertes Majeures", discoveries)

# ==================== PAGE: BIBLIOTHÈQUE ====================

elif page == "📚 Bibliothèque Technique":
    st.header("📚 Bibliothèque Technique des Supercalculateurs")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Types", "🔧 Technologies", "❄️ Refroidissement", "📊 Métriques"])
    
    with tab1:
        st.subheader("🏗️ Types de Supercalculateurs")
        
        types_info = {
            "Supercalculateur Quantique": {
                "description": "Utilise qubits pour calculs exponentiellement plus rapides",
                "avantages": ["Suprématie quantique", "Simulation moléculaire", "Cryptographie"],
                "défis": ["Température cryogénique", "Correction d'erreurs", "Cohérence limitée"]
            },
            "Supercalculateur Biologique": {
                "description": "Basé sur substrats neuronaux et processus biologiques",
                "avantages": ["Efficacité énergétique extrême", "Apprentissage naturel", "Parallélisme massif"],
                "défis": ["Vitesse limitée", "Interface électronique", "Maintenance biologique"]
            },
            "Hybride Quantique-Biologique": {
                "description": "Combine puissance quantique et efficacité biologique",
                "avantages": ["Meilleure performance globale", "Polyvalence", "Efficacité optimale"],
                "défis": ["Complexité extrême", "Intégration difficile", "Coût élevé"]
            },
            "Exascale": {
                "description": "Classe exaflops (10^18 FLOPS)",
                "avantages": ["Performances massives", "Technologies matures", "Large écosystème"],
                "défis": ["Consommation énergétique", "Coût", "Refroidissement"]
            }
        }
        
        for type_name, info in types_info.items():
            with st.expander(f"⚡ {type_name}"):
                st.write(f"**{info['description']}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**✅ Avantages:**")
                    for adv in info['avantages']:
                        st.write(f"• {adv}")
                
                with col2:
                    st.write("**⚠️ Défis:**")
                    for defi in info['défis']:
                        st.write(f"• {defi}")
    
    with tab2:
        st.subheader("🔧 Technologies Clés")
        
        technologies = {
            "Processeurs": ["Intel Xeon", "AMD EPYC", "NVIDIA Grace", "ARM Neoverse", "IBM POWER"],
            "Accélérateurs": ["NVIDIA GPU", "AMD Instinct", "Intel Xe", "Google TPU", "Cerebras WSE"],
            "Interconnexion": ["InfiniBand HDR", "Ethernet 400G", "Intel Omni-Path", "Slingshot"],
            "Stockage": ["Lustre", "GPFS", "BeeGFS", "Ceph", "All-Flash Arrays"],
            "Refroidissement": ["Liquid Direct-to-Chip", "Immersion", "Rear Door Heat Exchanger"]
        }
        
        for tech_category, tech_list in technologies.items():
            st.write(f"**{tech_category}:**")
            for tech in tech_list:
                st.write(f"• {tech}")
            st.markdown("---")
    
    with tab3:
        st.subheader("❄️ Systèmes de Refroidissement")
        
        cooling_systems = {
            "Refroidissement Air": "Traditionnel, PUE ~1.5-2.0",
            "Refroidissement Liquide": "Direct-to-chip, PUE ~1.2-1.3",
            "Immersion": "Composants immergés, PUE ~1.05-1.1",
            "Cryogénique": "Pour systèmes quantiques, <4K",
            "Free Cooling": "Air extérieur, très efficace",
            "Hybrid": "Combinaison de méthodes"
        }
        
        for system, description in cooling_systems.items():
            st.write(f"**{system}:** {description}")
    
    with tab4:
        st.subheader("📊 Métriques de Performance")
        
        metrics = {
            "FLOPS": "Floating Point Operations Per Second - Performance brute",
            "Rpeak": "Performance théorique maximale",
            "Rmax": "Performance maximale mesurée (Linpack)",
            "PUE": "Power Usage Effectiveness - Efficacité datacenter",
            "FLOPS/Watt": "Efficacité énergétique",
            "Quantum Volume": "Capacité quantique effective",
            "MTBF": "Mean Time Between Failures - Fiabilité",
            "TCO": "Total Cost of Ownership - Coût total"
        }
        
        for metric, description in metrics.items():
            st.write(f"**{metric}:** {description}")

# ==================== PAGE: MAINTENANCE ====================

elif page == "⚙️ Maintenance & Monitoring":
    st.header("⚙️ Centre de Maintenance et Monitoring")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.info("Aucun système à monitorer")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Monitoring Temps Réel", "🔧 Maintenance", "📜 Logs"])
        
        with tab1:
            st.subheader("📊 Monitoring en Temps Réel")
            
            # Métriques système
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_health = np.mean([sc['health'] for sc in st.session_state.supercomputer_system['supercomputers'].values()])
                st.metric("Santé Moyenne", f"{avg_health:.0%}")
            
            with col2:
                online_count = sum(1 for sc in st.session_state.supercomputer_system['supercomputers'].values() if sc['status'] == 'online')
                st.metric("Systèmes en Ligne", f"{online_count}/{total_sc}")
            
            with col3:
                avg_utilization = np.mean([sc.get('utilization', 0) for sc in st.session_state.supercomputer_system['supercomputers'].values()])
                st.metric("Utilisation Moyenne", f"{avg_utilization:.0%}")
            
            with col4:
                issues = sum(1 for sc in st.session_state.supercomputer_system['supercomputers'].values() if sc['health'] < 0.95)
                st.metric("Alertes", issues)
            
            # Graphique de santé
            st.markdown("---")
            
            time_series = list(range(60))
            health_data = [90 + np.random.randint(-5, 5) for _ in time_series]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_series,
                y=health_data,
                mode='lines',
                line=dict(color='rgb(240, 147, 251)', width=2),
                fill='tozeroy'
            ))
            fig.update_layout(
                title="Santé Système (dernière heure)",
                xaxis_title="Minutes",
                yaxis_title="Santé (%)",
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("🔧 Opérations de Maintenance")
            
            for sc in st.session_state.supercomputer_system['supercomputers'].values():
                with st.expander(f"⚡ {sc['name']} - Santé: {sc['health']:.0%}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**État des Composants:**")
                        components = {
                            'Processeurs': np.random.random(),
                            'Mémoire': np.random.random(),
                            'Stockage': np.random.random(),
                            'Réseau': np.random.random(),
                            'Refroidissement': np.random.random()
                        }
                        
                        for comp, health in components.items():
                            color = '🟢' if health > 0.95 else '🟡' if health > 0.8 else '🔴'
                            st.write(f"{color} {comp}: {health:.0%}")
                    
                    with col2:
                        st.write("**Actions Disponibles:**")
                        
                        if st.button(f"🔄 Redémarrer", key=f"restart_{sc['id']}"):
                            sc['health'] = min(1.0, sc['health'] + 0.05)
                            st.success("Redémarrage effectué")
                            st.rerun()
                        
                        if st.button(f"🧹 Nettoyage", key=f"clean_{sc['id']}"):
                            sc['health'] = min(1.0, sc['health'] + 0.02)
                            st.success("Nettoyage effectué")
                        
                        if st.button(f"⚡ Optimiser", key=f"optimize_{sc['id']}"):
                            sc['efficiency'] = min(0.98, sc['efficiency'] * 1.02)
                            st.success("Optimisation appliquée")
                            st.rerun()
        
        with tab3:
            st.subheader("📜 Journal des Événements")
            
            if st.session_state.supercomputer_system['log']:
                for log_entry in reversed(st.session_state.supercomputer_system['log'][-20:]):
                    st.text(f"{log_entry['timestamp'][:19]} - {log_entry['message']}")
            else:
                st.info("Aucun événement enregistré")

# ==================== PAGE: FABRICATION ====================

elif page == "🏭 Fabrication":
    st.header("🏭 Chaîne de Fabrication")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.warning("⚠️ Aucun supercalculateur disponible pour fabrication")
    else:
        tab1, tab2, tab3 = st.tabs(["🏗️ Nouvelle Fabrication", "📊 Fabrications en Cours", "📜 Historique"])
        
        with tab1:
            st.subheader("🏗️ Planifier une Nouvelle Fabrication")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner le supercalculateur à fabriquer",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x]
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            st.markdown(f"### ⚡ {sc['name']}")
            st.write(f"**Type:** {sc['type'].replace('_', ' ').title()}")
            st.write(f"**Performance:** {format_flops(sc['peak_performance'])}")
            
            st.markdown("---")
            
            # Plan de fabrication
            st.subheader("📋 Plan de Fabrication")
            
            phases = []
            
            # Phase 1: Infrastructure
            phases.append({
                'phase': 1,
                'name': 'Infrastructure et Datacenter',
                'duration': 180,
                'cost': 50000000,
                'steps': [
                    'Construction du bâtiment',
                    'Installation électrique',
                    'Système de refroidissement',
                    'Réseau de communication',
                    'Systèmes de sécurité'
                ]
            })
            
            # Phase 2: Composants classiques
            phases.append({
                'phase': 2,
                'name': 'Installation Composants Classiques',
                'duration': 90,
                'cost': 100000000,
                'steps': [
                    f"Installation de {sc['nodes']:,} nœuds",
                    'Configuration processeurs',
                    'Installation mémoire',
                    'Déploiement stockage',
                    'Interconnexion réseau'
                ]
            })
            
            # Phase 3: Quantique
            if sc['quantum_system']:
                phases.append({
                    'phase': 3,
                    'name': 'Intégration Système Quantique',
                    'duration': 120,
                    'cost': 200000000,
                    'steps': [
                        f"Fabrication de {sc['quantum_system']['qubits']:,} qubits",
                        'Système cryogénique',
                        'Isolation magnétique',
                        'Calibration quantique',
                        'Tests de cohérence'
                    ]
                })
            
            # Phase 4: Biologique
            if sc['biological_system']:
                phases.append({
                    'phase': 4,
                    'name': 'Intégration Système Biologique',
                    'duration': 150,
                    'cost': 150000000,
                    'steps': [
                        'Culture substrats neuronaux',
                        'Assemblage organoïdes',
                        'Interfaces bio-électroniques',
                        'Systèmes de maintien vital',
                        'Tests biocompatibilité'
                    ]
                })
            
            # Phase 5: Intégration
            phases.append({
                'phase': len(phases) + 1,
                'name': 'Intégration et Tests',
                'duration': 60,
                'cost': 20000000,
                'steps': [
                    'Intégration systèmes',
                    'Tests de performance',
                    'Benchmarks standards',
                    'Optimisation',
                    'Certification'
                ]
            })
            
            # Affichage des phases
            for phase in phases:
                with st.expander(f"Phase {phase['phase']}: {phase['name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Durée:** {phase['duration']} jours ({phase['duration']/30:.1f} mois)")
                        st.write(f"**Coût:** ${phase['cost']:,}")
                    
                    with col2:
                        st.write("**Étapes:**")
                        for step in phase['steps']:
                            st.write(f"• {step}")
            
            # Totaux
            total_duration = sum(p['duration'] for p in phases)
            total_cost = sum(p['cost'] for p in phases)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Durée Totale", f"{total_duration} jours")
                st.caption(f"{total_duration/30:.1f} mois")
            
            with col2:
                st.metric("Coût Total", f"${total_cost:,}")
                st.caption(f"{total_cost/1e9:.2f} milliards USD")
            
            with col3:
                st.metric("Phases", len(phases))
            
            # Lancer la fabrication
            location = st.text_input("📍 Localisation du Datacenter", "Silicon Valley, USA")
            contractor = st.text_input("🏢 Entreprise de Construction", "Advanced Tech Builders")
            
            if st.button("🚀 Lancer la Fabrication", use_container_width=True, type="primary"):
                fabrication_id = f"fab_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                fabrication = {
                    'fabrication_id': fabrication_id,
                    'supercomputer_id': selected_sc_id,
                    'supercomputer_name': sc['name'],
                    'phases': phases,
                    'current_phase': 0,
                    'progress': 0.0,
                    'status': 'in_progress',
                    'location': location,
                    'contractor': contractor,
                    'start_date': datetime.now().isoformat(),
                    'estimated_completion': (datetime.now() + timedelta(days=total_duration)).isoformat(),
                    'total_cost': total_cost,
                    'total_duration': total_duration
                }
                
                st.session_state.supercomputer_system['fabrications'].append(fabrication)
                log_event(f"Fabrication démarrée: {sc['name']}")
                
                st.success(f"✅ Fabrication démarrée!")
                st.balloons()
                st.code(f"Fabrication ID: {fabrication_id}")
        
        with tab2:
            st.subheader("📊 Fabrications en Cours")
            
            in_progress = [f for f in st.session_state.supercomputer_system['fabrications'] if f['status'] == 'in_progress']
            
            if not in_progress:
                st.info("Aucune fabrication en cours")
            else:
                for fab in in_progress:
                    with st.expander(f"🏭 {fab['supercomputer_name']} - {fab['progress']:.0f}%", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**ID:** {fab['fabrication_id']}")
                            st.write(f"**Localisation:** {fab['location']}")
                        
                        with col2:
                            st.write(f"**Démarrage:** {fab['start_date'][:10]}")
                            st.write(f"**Fin estimée:** {fab['estimated_completion'][:10]}")
                        
                        with col3:
                            st.write(f"**Coût total:** ${fab['total_cost']:,}")
                            st.write(f"**Durée:** {fab['total_duration']} jours")
                        
                        # Barre de progression globale
                        st.progress(fab['progress'] / 100, text=f"Progression globale: {fab['progress']:.0f}%")
                        
                        # Phases
                        st.markdown("---")
                        st.write("**Phases:**")
                        
                        for phase in fab['phases']:
                            phase_num = phase['phase']
                            if phase_num < fab['current_phase']:
                                status_class = "completed"
                                status_icon = "✅"
                            elif phase_num == fab['current_phase']:
                                status_class = "active"
                                status_icon = "⏳"
                            else:
                                status_class = ""
                                status_icon = "⏸️"
                            
                            st.markdown(f'<div class="fabrication-step {status_class}">{status_icon} Phase {phase_num}: {phase["name"]}</div>', unsafe_allow_html=True)
                        
                        # Actions
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button(f"⏩ Avancer Phase", key=f"advance_{fab['fabrication_id']}"):
                                if fab['current_phase'] < len(fab['phases']):
                                    fab['current_phase'] += 1
                                    fab['progress'] = (fab['current_phase'] / len(fab['phases'])) * 100
                                    
                                    if fab['current_phase'] >= len(fab['phases']):
                                        fab['status'] = 'completed'
                                        fab['progress'] = 100.0
                                        st.success("🎉 Fabrication terminée!")
                                        st.balloons()
                                    
                                    st.rerun()
                        
                        with col2:
                            if st.button(f"⏸️ Pause", key=f"pause_{fab['fabrication_id']}"):
                                fab['status'] = 'paused'
                                st.rerun()
                        
                        with col3:
                            if st.button(f"❌ Annuler", key=f"cancel_{fab['fabrication_id']}"):
                                fab['status'] = 'cancelled'
                                st.rerun()
        
        with tab3:
            st.subheader("📜 Historique des Fabrications")
            
            if st.session_state.supercomputer_system['fabrications']:
                history_data = []
                for fab in st.session_state.supercomputer_system['fabrications']:
                    history_data.append({
                        'Supercalculateur': fab['supercomputer_name'],
                        'Localisation': fab['location'],
                        'Démarrage': fab['start_date'][:10],
                        'Statut': fab['status'].upper(),
                        'Progression': f"{fab['progress']:.0f}%",
                        'Coût': f"${fab['total_cost']:,}"
                    })
                
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True)
            else:
                st.info("Aucun historique")

# ==================== PAGE: BENCHMARKING ====================

elif page == "📊 Benchmarking & TOP500":
    st.header("📊 Suite de Benchmarking & Classement TOP500")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.warning("⚠️ Aucun supercalculateur disponible")
    else:
        tab1, tab2, tab3 = st.tabs(["🏆 TOP500", "🌱 GREEN500", "⚛️ Benchmarks Spécialisés"])
        
        with tab1:
            st.subheader("🏆 Benchmark TOP500 (Linpack)")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner un supercalculateur",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x]
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            if st.button("🚀 Lancer Benchmark Linpack", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status = st.empty()
                
                for i in range(100):
                    progress_bar.progress((i + 1) / 100)
                    status.text(f"Exécution Linpack: {i + 1}%")
                
                # Calcul du score
                linpack_score = sc['sustained_performance'] / 1e15  # PFLOPS
                sc['benchmarks']['linpack'] = linpack_score
                
                # Rang estimé
                rank = max(1, int(500 / (linpack_score / 10 + 1)))
                
                progress_bar.empty()
                status.empty()
                
                st.success("✅ Benchmark Linpack terminé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Score Linpack", f"{linpack_score:.2f} PFLOPS")
                
                with col2:
                    st.metric("Rang Estimé TOP500", f"#{rank}")
                
                with col3:
                    st.metric("Performance Soutenue", format_flops(sc['sustained_performance']))
                
                # Graphique de comparaison
                st.markdown("---")
                
                # Simulation TOP10
                top10_names = [f"#{i+1} System" for i in range(10)]
                top10_scores = [linpack_score * (10 - i) / 5 for i in range(10)]
                if rank <= len(top10_scores):
                    top10_scores[rank-1] = linpack_score
                else:
                    top10_scores.append(linpack_score)

                # top10_scores[rank-1] = linpack_score if rank <= 10 else top10_scores[9]
                
                fig = go.Figure(data=[
                    go.Bar(x=top10_names, y=top10_scores, marker_color=['red' if i == rank-1 else 'lightblue' for i in range(10)])
                ])
                fig.update_layout(title="Comparaison TOP10", yaxis_title="Performance (PFLOPS)")
                st.plotly_chart(fig, use_container_width=True)
                
                log_event(f"Benchmark Linpack: {sc['name']} - {linpack_score:.2f} PFLOPS")
        
        with tab2:
            st.subheader("🌱 GREEN500 - Efficacité Énergétique")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner un supercalculateur",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x],
                key="green500_select"
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            if st.button("🌱 Calculer Efficacité GREEN500", use_container_width=True):
                # FLOPS par Watt
                flops_per_watt = sc['peak_performance'] / (sc['power']['consumption'] * 1e6)
                sc['benchmarks']['green500'] = flops_per_watt
                
                st.success("✅ Calcul d'efficacité terminé!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("FLOPS/Watt", f"{flops_per_watt/1e9:.2f} GFLOPS/W")
                
                with col2:
                    st.metric("Consommation", f"{sc['power']['consumption']:.1f} MW")
                
                with col3:
                    st.metric("PUE", f"{sc['cooling']['pue']:.2f}")
                
                with col4:
                    rating = 'A+' if flops_per_watt > 50e9 else 'A' if flops_per_watt > 25e9 else 'B' if flops_per_watt > 15e9 else 'C'
                    st.metric("Note Efficacité", rating)
                
                # Graphique d'efficacité
                st.markdown("---")
                
                labels = ['Performance\n(PFLOPS)', 'Efficacité\n(GFLOPS/W)', 'Renouvelable\n(%)', 'PUE\nInverse']
                values = [
                    sc['peak_performance'] / 1e18 * 100,  # Normalized
                    flops_per_watt / 1e11 * 100,
                    sc['power']['renewable_percentage'] * 100,
                    (2 - sc['cooling']['pue']) * 100
                ]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself'
                ))
                fig.update_layout(title="Profil Efficacité Énergétique")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("⚛️ Benchmarks Spécialisés")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner un supercalculateur",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x],
                key="special_bench_select"
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            benchmark_suite = st.multiselect(
                "Sélectionner les benchmarks",
                ["HPCG", "Graph500", "MLPerf", "Quantum Volume", "Bio-Efficiency", "IO500"]
            )
            
            if benchmark_suite and st.button("🚀 Lancer Suite de Benchmarks"):
                progress_bar = st.progress(0)
                
                results = {}
                
                for i, bench in enumerate(benchmark_suite):
                    progress_bar.progress((i + 1) / len(benchmark_suite))
                    
                    if bench == "HPCG":
                        score = sc['sustained_performance'] / 1e15 * 0.3
                        sc['benchmarks']['hpcg'] = score
                        results[bench] = f"{score:.2f} PFLOPS"
                    
                    elif bench == "Graph500":
                        score = sc['total_cores'] * 1000
                        results[bench] = f"{score:,} MTEPS"
                    
                    elif bench == "MLPerf":
                        score = sc['peak_performance'] / 1e15 * 2
                        results[bench] = f"{score:.2f} Images/sec"
                    
                    elif bench == "Quantum Volume" and sc['quantum_system']:
                        score = sc['quantum_system']['quantum_volume']
                        sc['benchmarks']['quantum'] = score
                        results[bench] = f"{score:,}"
                    
                    elif bench == "Bio-Efficiency" and sc['biological_system']:
                        score = sc['biological_system']['bio_efficiency'] * 100
                        results[bench] = f"{score:.0f}%"
                    
                    else:
                        results[bench] = "N/A"
                
                progress_bar.empty()
                
                st.success("✅ Suite de benchmarks terminée!")
                
                # Affichage résultats
                for bench, result in results.items():
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**{bench}:**")
                    with col2:
                        st.write(result)

# ==================== PAGE: GESTION DES JOBS ====================

elif page == "💼 Gestion des Jobs":
    st.header("💼 Système de Gestion des Jobs")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.warning("⚠️ Aucun supercalculateur disponible")
    else:
        tab1, tab2 = st.tabs(["➕ Soumettre Job", "📊 Jobs en Cours"])
        
        with tab1:
            st.subheader("➕ Soumettre un Nouveau Job")
            
            with st.form("submit_job_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    job_name = st.text_input("Nom du Job")
                    
                    sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values() if sc['status'] == 'online'}
                    if not sc_options:
                        st.error("Aucun supercalculateur en ligne")
                        selected_sc = None
                    else:
                        selected_sc = st.selectbox(
                            "Supercalculateur Cible",
                            options=list(sc_options.keys()),
                            format_func=lambda x: sc_options[x]
                        )
                
                with col2:
                    application = st.selectbox(
                        "Type d'Application",
                        [
                            "Recherche Scientifique",
                            "Modélisation Climat",
                            "Découverte Médicaments",
                            "Simulation Quantique",
                            "Entraînement IA",
                            "Génomique",
                            "Astrophysique",
                            "Dynamique Moléculaire"
                        ]
                    )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    nodes_requested = st.number_input("Nœuds Demandés", 1, 10000, 100)
                
                with col2:
                    walltime = st.number_input("Temps Max (heures)", 1, 720, 24)
                
                with col3:
                    priority = st.select_slider("Priorité", ["Basse", "Normale", "Haute", "Urgente"])
                
                job_description = st.text_area("Description du Job")
                
                submitted = st.form_submit_button("🚀 Soumettre le Job")
                
                if submitted and selected_sc:
                    job_id = f"job_{len(st.session_state.supercomputer_system['jobs']) + 1}"
                    
                    job = {
                        'job_id': job_id,
                        'name': job_name,
                        'supercomputer_id': selected_sc,
                        'application': application,
                        'nodes_requested': nodes_requested,
                        'walltime': walltime,
                        'priority': priority,
                        'description': job_description,
                        'status': 'queued',
                        'progress': 0.0,
                        'submit_time': datetime.now().isoformat(),
                        'start_time': None,
                        'end_time': None
                    }
                    
                    st.session_state.supercomputer_system['jobs'].append(job)
                    log_event(f"Job soumis: {job_name}")
                    
                    st.success(f"✅ Job soumis avec succès!")
                    st.code(f"Job ID: {job_id}")
        
        with tab2:
            st.subheader("📊 Jobs en Cours et En Attente")
            
            if not st.session_state.supercomputer_system['jobs']:
                st.info("Aucun job soumis")
            else:
                for job in st.session_state.supercomputer_system['jobs']:
                    status_color = {
                        'queued': '🟡',
                        'running': '🟢',
                        'completed': '✅',
                        'failed': '❌'
                    }.get(job['status'], '❓')
                    
                    with st.expander(f"{status_color} {job['name']} - {job['status'].upper()}"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write(f"**Job ID:** {job['job_id']}")
                            st.write(f"**Application:** {job['application']}")
                        
                        with col2:
                            st.write(f"**Nœuds:** {job['nodes_requested']}")
                            st.write(f"**Temps Max:** {job['walltime']}h")
                        
                        with col3:
                            st.write(f"**Priorité:** {job['priority']}")
                            st.write(f"**Soumis:** {job['submit_time'][:19]}")
                        
                        if job['status'] == 'running':
                            st.progress(job['progress'] / 100, text=f"Progression: {job['progress']:.0f}%")
                        
                        # Actions
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if job['status'] == 'queued' and st.button(f"▶️ Démarrer", key=f"start_{job['job_id']}"):
                                job['status'] = 'running'
                                job['start_time'] = datetime.now().isoformat()
                                st.rerun()
                        
                        with col2:
                            if job['status'] == 'running' and st.button(f"✅ Terminer", key=f"complete_{job['job_id']}"):
                                job['status'] = 'completed'
                                job['end_time'] = datetime.now().isoformat()
                                job['progress'] = 100.0
                                
                                # Mettre à jour le supercalculateur
                                sc = st.session_state.supercomputer_system['supercomputers'][job['supercomputer_id']]
                                sc['jobs_completed'] += 1
                                
                                st.rerun()
                        
                        with col3:
                            if st.button(f"❌ Annuler", key=f"cancel_{job['job_id']}"):
                                job['status'] = 'failed'
                                st.rerun()


# ==================== PAGE: CONFIGURATION & OPTIMISATION ====================

elif page == "🔧 Configuration & Optimisation":
    st.header("🔧 Configuration et Optimisation Avancée")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.warning("⚠️ Aucun supercalculateur disponible")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Configuration Matérielle", "🚀 Optimisation Performance", "🔄 Reconfiguration", "📊 Tuning Avancé"])
        
        with tab1:
            st.subheader("⚙️ Configuration Matérielle")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner un supercalculateur",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x]
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            st.markdown(f"### ⚡ Configuration Actuelle - {sc['name']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Ressources de Calcul")
                
                new_nodes = st.number_input("Nœuds", 100, 1000000, sc['nodes'], key="config_nodes")
                new_cores_per_node = st.number_input("Cœurs/Nœud", 32, 512, sc['cores_per_node'], key="config_cores")
                new_memory = st.number_input("Mémoire Total (TB)", 100, 100000, sc['memory_total']//1000) * 1000
                
                if st.button("💾 Appliquer Config Ressources"):
                    sc['nodes'] = new_nodes
                    sc['cores_per_node'] = new_cores_per_node
                    sc['total_cores'] = new_nodes * new_cores_per_node
                    sc['memory_total'] = new_memory
                    
                    # Recalculer performance
                    base_flops = sc['total_cores'] * 2.5e9 * 4
                    if sc['quantum_system']:
                        base_flops += sc['quantum_system']['qubits'] * 1e12
                    if sc['biological_system']:
                        base_flops += sc['biological_system']['neurons'] * 1000
                    
                    sc['peak_performance'] = base_flops
                    sc['sustained_performance'] = base_flops * sc['efficiency']
                    
                    st.success("✅ Configuration ressources mise à jour!")
                    log_event(f"Configuration ressources modifiée: {sc['name']}")
                    st.rerun()
            
            with col2:
                st.subheader("🌐 Configuration Réseau")
                
                new_interconnect = st.selectbox(
                    "Interconnexion",
                    ["infiniband", "ethernet_100g", "omni_path", "photonique"],
                    index=["infiniband", "ethernet_100g", "omni_path", "photonique"].index(sc['network']['interconnect']) if sc['network']['interconnect'] in ["infiniband", "ethernet_100g", "omni_path", "photonique"] else 0
                )
                
                new_bandwidth = st.slider("Bande Passante (Gbps)", 100, 800, sc['network']['bandwidth'])
                new_latency = st.slider("Latence (μs)", 0.5, 10.0, sc['network']['latency'], 0.5)
                
                if st.button("💾 Appliquer Config Réseau"):
                    sc['network']['interconnect'] = new_interconnect
                    sc['network']['bandwidth'] = new_bandwidth
                    sc['network']['latency'] = new_latency
                    
                    st.success("✅ Configuration réseau mise à jour!")
                    log_event(f"Configuration réseau modifiée: {sc['name']}")
                    st.rerun()
            
            # Stockage
            st.markdown("---")
            st.subheader("💾 Configuration Stockage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_storage_capacity = st.number_input("Capacité (TB)", 1000, 10000000, sc['storage']['capacity'])
            
            with col2:
                new_storage_type = st.selectbox("Type", ["hdd", "ssd", "nvme", "optane"], 
                                               index=["hdd", "ssd", "nvme", "optane"].index(sc['storage']['type']) if sc['storage']['type'] in ["hdd", "ssd", "nvme", "optane"] else 2)
            
            with col3:
                new_read_speed = st.number_input("Vitesse Lecture (GB/s)", 1, 200, sc['storage']['read_speed'])
            
            if st.button("💾 Appliquer Config Stockage"):
                sc['storage']['capacity'] = new_storage_capacity
                sc['storage']['type'] = new_storage_type
                sc['storage']['read_speed'] = new_read_speed
                sc['storage']['write_speed'] = int(new_read_speed * 0.8)
                
                st.success("✅ Configuration stockage mise à jour!")
                st.rerun()
        
        with tab2:
            st.subheader("🚀 Optimisation des Performances")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner un supercalculateur",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x],
                key="opt_select"
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            st.write(f"### ⚡ {sc['name']}")
            st.write(f"**Efficacité Actuelle:** {sc['efficiency']:.0%}")
            st.write(f"**Performance Soutenue:** {format_flops(sc['sustained_performance'])}")
            
            st.markdown("---")
            
            optimization_strategies = {
                "🔧 Optimisation Compilateur": {
                    "description": "Recompiler avec optimisations avancées (-O3, vectorisation)",
                    "efficiency_gain": 0.03,
                    "time": "2 heures",
                    "risk": "Faible"
                },
                "⚡ Équilibrage Charges": {
                    "description": "Optimiser distribution des tâches entre nœuds",
                    "efficiency_gain": 0.05,
                    "time": "1 heure",
                    "risk": "Faible"
                },
                "🌐 Optimisation Réseau": {
                    "description": "Tuning TCP/IP et topologie réseau",
                    "efficiency_gain": 0.04,
                    "time": "3 heures",
                    "risk": "Moyen"
                },
                "💾 Optimisation I/O": {
                    "description": "Améliorer buffers et cache",
                    "efficiency_gain": 0.06,
                    "time": "2 heures",
                    "risk": "Faible"
                },
                "🧠 Optimisation Mémoire": {
                    "description": "Tuning allocation mémoire et NUMA",
                    "efficiency_gain": 0.04,
                    "time": "2 heures",
                    "risk": "Moyen"
                },
                "⚛️ Optimisation Quantique": {
                    "description": "Améliorer cohérence et fidélité qubits",
                    "efficiency_gain": 0.08,
                    "time": "8 heures",
                    "risk": "Élevé",
                    "requires": "quantum_system"
                },
                "🧬 Optimisation Biologique": {
                    "description": "Améliorer plasticité et efficacité neuronale",
                    "efficiency_gain": 0.07,
                    "time": "12 heures",
                    "risk": "Élevé",
                    "requires": "biological_system"
                }
            }
            
            selected_optimizations = []
            
            for opt_name, opt_info in optimization_strategies.items():
                # Vérifier si l'optimisation est applicable
                if 'requires' in opt_info:
                    if opt_info['requires'] == 'quantum_system' and not sc['quantum_system']:
                        continue
                    if opt_info['requires'] == 'biological_system' and not sc['biological_system']:
                        continue
                
                with st.expander(f"{opt_name} - Gain: +{opt_info['efficiency_gain']:.0%}"):
                    st.write(f"**Description:** {opt_info['description']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Gain:** +{opt_info['efficiency_gain']:.0%}")
                    with col2:
                        st.write(f"**Durée:** {opt_info['time']}")
                    with col3:
                        st.write(f"**Risque:** {opt_info['risk']}")
                    
                    if st.checkbox(f"Sélectionner {opt_name}", key=f"opt_{opt_name}"):
                        selected_optimizations.append((opt_name, opt_info))
            
            if selected_optimizations:
                total_gain = sum(opt[1]['efficiency_gain'] for opt in selected_optimizations)
                total_time = sum(float(opt[1]['time'].split()[0]) for opt in selected_optimizations)
                
                st.markdown("---")
                st.write(f"### 📊 Résumé des Optimisations")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gain Total", f"+{total_gain:.0%}")
                with col2:
                    st.metric("Durée Totale", f"{total_time:.0f}h")
                with col3:
                    new_efficiency = min(0.99, sc['efficiency'] + total_gain)
                    st.metric("Nouvelle Efficacité", f"{new_efficiency:.0%}")
                
                if st.button("🚀 Appliquer Optimisations", use_container_width=True, type="primary"):
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    for i, (opt_name, opt_info) in enumerate(selected_optimizations):
                        progress = (i + 1) / len(selected_optimizations)
                        progress_bar.progress(progress)
                        status.text(f"Application: {opt_name}...")
                        
                        sc['efficiency'] = min(0.99, sc['efficiency'] + opt_info['efficiency_gain'])
                    
                    # Recalculer performance soutenue
                    sc['sustained_performance'] = sc['peak_performance'] * sc['efficiency']
                    
                    progress_bar.empty()
                    status.empty()
                    
                    st.success(f"✅ {len(selected_optimizations)} optimisation(s) appliquée(s)!")
                    st.balloons()
                    
                    log_event(f"Optimisations appliquées sur {sc['name']}: {len(selected_optimizations)} optimisations")
                    st.rerun()
        
        with tab3:
            st.subheader("🔄 Reconfiguration Dynamique")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner un supercalculateur",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x],
                key="reconfig_select"
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            st.write("### 🔄 Modes de Reconfiguration")
            
            reconfig_modes = {
                "Mode Performance": {
                    "description": "Maximise la performance au détriment de la consommation",
                    "efficiency_change": 0.02,
                    "power_change": 1.15,
                    "icon": "🚀"
                },
                "Mode Efficacité": {
                    "description": "Optimise l'efficacité énergétique",
                    "efficiency_change": 0.05,
                    "power_change": 0.85,
                    "icon": "🌱"
                },
                "Mode Équilibré": {
                    "description": "Balance performance et efficacité",
                    "efficiency_change": 0.0,
                    "power_change": 1.0,
                    "icon": "⚖️"
                },
                "Mode Économie": {
                    "description": "Réduit drastiquement la consommation",
                    "efficiency_change": -0.05,
                    "power_change": 0.7,
                    "icon": "💡"
                }
            }
            
            for mode_name, mode_info in reconfig_modes.items():
                with st.expander(f"{mode_info['icon']} {mode_name}"):
                    st.write(f"**{mode_info['description']}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Impact Efficacité:** {mode_info['efficiency_change']:+.0%}")
                    with col2:
                        st.write(f"**Impact Consommation:** {mode_info['power_change']:.0%}x")
                    
                    if st.button(f"Appliquer {mode_name}", key=f"mode_{mode_name}"):
                        old_efficiency = sc['efficiency']
                        old_power = sc['power']['consumption']
                        
                        sc['efficiency'] = max(0.5, min(0.99, sc['efficiency'] + mode_info['efficiency_change']))
                        sc['power']['consumption'] = sc['power']['consumption'] * mode_info['power_change']
                        sc['sustained_performance'] = sc['peak_performance'] * sc['efficiency']
                        
                        st.success(f"✅ {mode_name} appliqué!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Efficacité", f"{sc['efficiency']:.0%}", f"{(sc['efficiency'] - old_efficiency):.0%}")
                        with col2:
                            st.metric("Consommation", f"{sc['power']['consumption']:.1f} MW", f"{(sc['power']['consumption'] - old_power):.1f} MW")
                        
                        log_event(f"Mode {mode_name} appliqué sur {sc['name']}")
                        st.rerun()
            
            # Overclocking
            st.markdown("---")
            st.subheader("⚡ Overclocking (Risqué)")
            
            st.warning("⚠️ L'overclocking peut améliorer les performances mais augmente la consommation et réduit la durée de vie")
            
            overclock_level = st.slider("Niveau d'Overclocking", 0.0, 0.2, 0.0, 0.05)
            
            if overclock_level > 0:
                performance_gain = overclock_level * sc['peak_performance']
                power_increase = overclock_level * sc['power']['consumption'] * 2
                health_decrease = overclock_level * 0.5
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Gain Performance", format_flops(performance_gain))
                with col2:
                    st.metric("⚡ Consommation +", f"{power_increase:.1f} MW")
                with col3:
                    st.metric("⚠️ Santé -", f"{health_decrease:.0%}")
                
                if st.button("⚡ Appliquer Overclocking", type="secondary"):
                    sc['peak_performance'] *= (1 + overclock_level)
                    sc['sustained_performance'] = sc['peak_performance'] * sc['efficiency']
                    sc['power']['consumption'] += power_increase
                    sc['health'] = max(0.5, sc['health'] - health_decrease)
                    
                    st.warning(f"⚠️ Overclocking appliqué! Santé réduite à {sc['health']:.0%}")
                    log_event(f"Overclocking {overclock_level:.0%} appliqué sur {sc['name']}")
                    st.rerun()
        
        with tab4:
            st.subheader("📊 Tuning Avancé des Paramètres")
            
            sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values()}
            selected_sc_id = st.selectbox(
                "Sélectionner un supercalculateur",
                options=list(sc_options.keys()),
                format_func=lambda x: sc_options[x],
                key="tuning_select"
            )
            
            sc = st.session_state.supercomputer_system['supercomputers'][selected_sc_id]
            
            st.write("### 🎛️ Paramètres Système")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("❄️ Refroidissement")
                
                new_cooling_system = st.selectbox(
                    "Système",
                    ["refroidissement_air", "refroidissement_liquide", "refroidissement_immersion", 
                     "cryogenique", "refroidissement_quantique", "bio_thermique", "refroidissement_hybride"],
                    index=0 if sc['cooling']['system'] not in ["refroidissement_air", "refroidissement_liquide", "refroidissement_immersion"] else 
                          ["refroidissement_air", "refroidissement_liquide", "refroidissement_immersion"].index(sc['cooling']['system']),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                
                target_temp = st.slider("Température Cible (°C)", 10, 35, sc['cooling']['temperature'])
                target_pue = st.slider("PUE Cible", 1.05, 2.0, sc['cooling']['pue'], 0.05)
                
                if st.button("💾 Appliquer Config Refroidissement"):
                    sc['cooling']['system'] = new_cooling_system
                    sc['cooling']['temperature'] = target_temp
                    sc['cooling']['pue'] = target_pue
                    
                    st.success("✅ Configuration refroidissement mise à jour!")
                    st.rerun()
            
            with col2:
                st.subheader("⚛️ Paramètres Quantiques")
                
                if sc['quantum_system']:
                    new_qubits = st.number_input("Qubits", 100, 100000, sc['quantum_system']['qubits'], key="tune_qubits")
                    target_fidelity = st.slider("Fidélité Cible", 0.99, 0.9999, sc['quantum_system']['gate_fidelity'], 0.0001, format="%.4f")
                    
                    if st.button("💾 Appliquer Config Quantique"):
                        sc['quantum_system']['qubits'] = new_qubits
                        sc['quantum_system']['gate_fidelity'] = target_fidelity
                        sc['quantum_system']['quantum_volume'] = new_qubits * 100
                        
                        # Recalculer performance
                        base_flops = sc['total_cores'] * 2.5e9 * 4
                        base_flops += new_qubits * 1e12
                        if sc['biological_system']:
                            base_flops += sc['biological_system']['neurons'] * 1000
                        
                        sc['peak_performance'] = base_flops
                        sc['sustained_performance'] = base_flops * sc['efficiency']
                        
                        st.success("✅ Configuration quantique mise à jour!")
                        st.rerun()
                else:
                    st.info("Pas de système quantique disponible")
            
            # Paramètres biologiques
            if sc['biological_system']:
                st.markdown("---")
                st.subheader("🧬 Paramètres Biologiques")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    new_neurons = st.number_input("Neurones (Milliards)", 1, 1000, 
                                                 sc['biological_system']['neurons']//1000000000) * 1000000000
                
                with col2:
                    target_plasticity = st.slider("Plasticité", 0.5, 1.0, sc['biological_system']['plasticity'], 0.05)
                
                if st.button("💾 Appliquer Config Biologique"):
                    sc['biological_system']['neurons'] = new_neurons
                    sc['biological_system']['synapses'] = new_neurons * 10
                    sc['biological_system']['plasticity'] = target_plasticity
                    
                    # Recalculer performance
                    base_flops = sc['total_cores'] * 2.5e9 * 4
                    if sc['quantum_system']:
                        base_flops += sc['quantum_system']['qubits'] * 1e12
                    base_flops += new_neurons * 1000
                    
                    sc['peak_performance'] = base_flops
                    sc['sustained_performance'] = base_flops * sc['efficiency']
                    
                    st.success("✅ Configuration biologique mise à jour!")
                    st.rerun()

# ==================== PAGE: DÉPLOIEMENT ====================

elif page == "🚀 Déploiement":
    st.header("🚀 Déploiement et Mise en Service")
    
    if not st.session_state.supercomputer_system['supercomputers']:
        st.warning("⚠️ Aucun supercalculateur disponible")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Nouveau Déploiement", "📊 Déploiements Actifs", "🌍 Datacenters", "📜 Historique"])
        
        with tab1:
            st.subheader("🚀 Planifier un Nouveau Déploiement")
            
            with st.form("deployment_form"):
                sc_options = {sc['id']: sc['name'] for sc in st.session_state.supercomputer_system['supercomputers'].values() if sc['status'] != 'online'}
                
                if not sc_options:
                    st.error("❌ Aucun supercalculateur disponible (tous déjà en ligne)")
                    selected_sc = None
                else:
                    selected_sc = st.selectbox(
                        "Sélectionner le supercalculateur",
                        options=list(sc_options.keys()),
                        format_func=lambda x: sc_options[x]
                    )
                
                st.markdown("---")
                st.subheader("📍 Localisation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    datacenter_name = st.text_input("Nom du Datacenter", placeholder="Ex: Silicon Valley DC-1")
                    country = st.selectbox("Pays", [
                        "États-Unis", "Chine", "Japon", "Allemagne", "France", 
                        "Royaume-Uni", "Suisse", "Singapour", "Corée du Sud", "Canada"
                    ])
                
                with col2:
                    city = st.text_input("Ville", placeholder="Ex: San Francisco")
                    datacenter_tier = st.selectbox("Tier Datacenter", ["Tier 1", "Tier 2", "Tier 3", "Tier 4"])
                
                st.markdown("---")
                st.subheader("🎯 Configuration du Déploiement")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    deployment_type = st.selectbox(
                        "Type de Déploiement",
                        ["Production", "Recherche", "Test", "Développement", "Hybride"]
                    )
                    
                    access_policy = st.selectbox(
                        "Politique d'Accès",
                        ["Public", "Académique", "Commercial", "Gouvernemental", "Mixte"]
                    )
                
                with col2:
                    operational_mode = st.selectbox(
                        "Mode Opérationnel",
                        ["24/7", "Business Hours", "On-Demand", "Scheduled"]
                    )
                    
                    redundancy_level = st.selectbox(
                        "Niveau de Redondance",
                        ["Aucune", "N+1", "N+2", "2N", "2N+1"]
                    )
                
                st.markdown("---")
                st.subheader("👥 Utilisateurs et Applications")
                
                primary_users = st.multiselect(
                    "Utilisateurs Principaux",
                    ["Universités", "Laboratoires Recherche", "Entreprises Tech", 
                     "Instituts Gouvernementaux", "Startups", "Organisations Internationales"]
                )
                
                applications = st.multiselect(
                    "Applications Cibles",
                    ["Recherche Scientifique", "Modélisation Climat", "Découverte Médicaments",
                     "Simulation Quantique", "Entraînement IA", "Cryptographie", "Génomique",
                     "Astrophysique", "Modélisation Financière", "Dynamique Moléculaire"]
                )
                
                st.markdown("---")
                st.subheader("📅 Planning")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    deployment_date = st.date_input("Date de Déploiement", value=datetime.now() + timedelta(days=30))
                
                with col2:
                    commissioning_duration = st.number_input("Durée Mise en Service (jours)", 1, 90, 14)
                
                with col3:
                    warranty_period = st.number_input("Période Garantie (ans)", 1, 10, 3)
                
                notes = st.text_area("Notes et Observations", placeholder="Informations supplémentaires sur le déploiement...")
                
                submitted = st.form_submit_button("🚀 Créer le Déploiement", use_container_width=True, type="primary")
                
                if submitted and selected_sc:
                    deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    
                    sc = st.session_state.supercomputer_system['supercomputers'][selected_sc]
                    
                    deployment = {
                        'deployment_id': deployment_id,
                        'supercomputer_id': selected_sc,
                        'supercomputer_name': sc['name'],
                        'datacenter': {
                            'name': datacenter_name,
                            'country': country,
                            'city': city,
                            'tier': datacenter_tier
                        },
                        'config': {
                            'type': deployment_type,
                            'access_policy': access_policy,
                            'operational_mode': operational_mode,
                            'redundancy_level': redundancy_level
                        },
                        'users': primary_users,
                        'applications': applications,
                        'planning': {
                            'deployment_date': deployment_date.isoformat(),
                            'commissioning_duration': commissioning_duration,
                            'warranty_period': warranty_period
                        },
                        'notes': notes,
                        'status': 'planned',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.supercomputer_system['deployments'][deployment_id] = deployment
                    
                    st.success(f"✅ Déploiement planifié avec succès!")
                    st.balloons()
                    
                    st.code(f"Deployment ID: {deployment_id}")
                    
                    log_event(f"Déploiement planifié: {sc['name']} à {datacenter_name}")
        
        with tab2:
            st.subheader("📊 Déploiements Actifs")
            
            active_deployments = {k: v for k, v in st.session_state.supercomputer_system['deployments'].items() 
                                 if v['status'] in ['planned', 'commissioning', 'operational']}
            
            if not active_deployments:
                st.info("Aucun déploiement actif")
            else:
                for deploy_id, deploy in active_deployments.items():
                    status_colors = {
                        'planned': '🔵',
                        'commissioning': '🟡',
                        'operational': '🟢',
                        'maintenance': '🟠',
                        'decommissioned': '⚫'
                    }
                    
                    status_icon = status_colors.get(deploy['status'], '❓')
                    
                    with st.expander(f"{status_icon} {deploy['supercomputer_name']} - {deploy['datacenter']['name']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Deployment ID:** {deploy_id}")
                            st.write(f"**Statut:** {deploy['status'].upper()}")
                            st.write(f"**Type:** {deploy['config']['type']}")
                        
                        with col2:
                            st.write(f"**Localisation:** {deploy['datacenter']['city']}, {deploy['datacenter']['country']}")
                            st.write(f"**Tier:** {deploy['datacenter']['tier']}")
                            st.write(f"**Mode:** {deploy['config']['operational_mode']}")
                        
                        with col3:
                            st.write(f"**Date Déploiement:** {deploy['planning']['deployment_date'][:10]}")
                            st.write(f"**Redondance:** {deploy['config']['redundancy_level']}")
                            st.write(f"**Garantie:** {deploy['planning']['warranty_period']} ans")
                        
                        # Détails supplémentaires
                        st.markdown("---")
                        
                        if deploy['users']:
                            st.write("**👥 Utilisateurs:**")
                            st.write(", ".join(deploy['users']))
                        
                        if deploy['applications']:
                            st.write("**🎯 Applications:**")
                            for app in deploy['applications']:
                                st.write(f"• {app}")
                        
                        # Actions de gestion
                        st.markdown("---")
                        st.write("**Actions:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if deploy['status'] == 'planned' and st.button("🚀 Démarrer Mise en Service", key=f"start_{deploy_id}"):
                                deploy['status'] = 'commissioning'
                                log_event(f"Mise en service démarrée: {deploy['supercomputer_name']}")
                                st.rerun()
                        
                        with col2:
                            if deploy['status'] == 'commissioning' and st.button("✅ Mise en Production", key=f"prod_{deploy_id}"):
                                deploy['status'] = 'operational'
                                sc = st.session_state.supercomputer_system['supercomputers'][deploy['supercomputer_id']]
                                sc['status'] = 'online'
                                log_event(f"Déploiement opérationnel: {deploy['supercomputer_name']}")
                                st.rerun()
                        
                        with col3:
                            if deploy['status'] == 'operational' and st.button("🔧 Maintenance", key=f"maint_{deploy_id}"):
                                deploy['status'] = 'maintenance'
                                log_event(f"Maintenance démarrée: {deploy['supercomputer_name']}")
                                st.rerun()
                        
                        with col4:
                            if st.button("🛑 Décommissionner", key=f"decom_{deploy_id}"):
                                deploy['status'] = 'decommissioned'
                                sc = st.session_state.supercomputer_system['supercomputers'][deploy['supercomputer_id']]
                                sc['status'] = 'offline'
                                log_event(f"Décommissionné: {deploy['supercomputer_name']}")
                                st.rerun()
                        
                        # Métriques opérationnelles (si opérationnel)
                        if deploy['status'] == 'operational':
                            st.markdown("---")
                            st.subheader("📊 Métriques Opérationnelles")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                uptime = np.random.uniform(95, 99.99)
                                st.metric("Uptime", f"{uptime:.2f}%")
                            
                            with col2:
                                utilization = np.random.uniform(60, 95)
                                st.metric("Utilisation", f"{utilization:.0f}%")
                            
                            with col3:
                                jobs_running = np.random.randint(50, 500)
                                st.metric("Jobs Actifs", jobs_running)
                            
                            with col4:
                                users_active = np.random.randint(10, 200)
                                st.metric("Utilisateurs", users_active)
        
        with tab3:
            st.subheader("🌍 Carte des Datacenters")
            
            st.write("### 📍 Localisation Géographique des Déploiements")
            
            if st.session_state.supercomputer_system['deployments']:
                # Compter les déploiements par pays
                country_counts = {}
                for deploy in st.session_state.supercomputer_system['deployments'].values():
                    country = deploy['datacenter']['country']
                    country_counts[country] = country_counts.get(country, 0) + 1
                
                # Créer un graphique
                fig = px.bar(
                    x=list(country_counts.keys()),
                    y=list(country_counts.values()),
                    labels={'x': 'Pays', 'y': 'Nombre de Déploiements'},
                    title="Distribution Géographique des Déploiements"
                )
                fig.update_traces(marker_color='rgb(240, 147, 251)')
                st.plotly_chart(fig, use_container_width=True)
                
                # Liste détaillée
                st.markdown("---")
                st.write("### 🏢 Liste des Datacenters")
                
                datacenter_data = []
                for deploy in st.session_state.supercomputer_system['deployments'].values():
                    datacenter_data.append({
                        'Datacenter': deploy['datacenter']['name'],
                        'Pays': deploy['datacenter']['country'],
                        'Ville': deploy['datacenter']['city'],
                        'Tier': deploy['datacenter']['tier'],
                        'Supercalculateur': deploy['supercomputer_name'],
                        'Statut': deploy['status'].upper()
                    })
                
                df_datacenters = pd.DataFrame(datacenter_data)
                st.dataframe(df_datacenters, use_container_width=True)
                
                # Statistiques par Tier
                st.markdown("---")
                st.write("### 📊 Répartition par Tier")
                
                tier_counts = {}
                for deploy in st.session_state.supercomputer_system['deployments'].values():
                    tier = deploy['datacenter']['tier']
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                
                col1, col2, col3, col4 = st.columns(4)
                
                for i, (tier, count) in enumerate(sorted(tier_counts.items())):
                    with [col1, col2, col3, col4][i % 4]:
                        st.metric(tier, count)
            
            else:
                st.info("Aucun datacenter déployé")
        
        with tab4:
            st.subheader("📜 Historique des Déploiements")
            
            if st.session_state.supercomputer_system['deployments']:
                history_data = []
                
                for deploy in st.session_state.supercomputer_system['deployments'].values():
                    history_data.append({
                        'Date': deploy['created_at'][:10],
                        'Supercalculateur': deploy['supercomputer_name'],
                        'Datacenter': deploy['datacenter']['name'],
                        'Localisation': f"{deploy['datacenter']['city']}, {deploy['datacenter']['country']}",
                        'Type': deploy['config']['type'],
                        'Statut': deploy['status'].upper(),
                        'Applications': len(deploy['applications'])
                    })
                
                df_history = pd.DataFrame(history_data)
                df_history = df_history.sort_values('Date', ascending=False)
                
                st.dataframe(df_history, use_container_width=True)
                
                # Timeline des déploiements
                st.markdown("---")
                st.write("### 📅 Timeline des Déploiements")
                
                dates = [datetime.fromisoformat(d['created_at']) for d in st.session_state.supercomputer_system['deployments'].values()]
                names = [d['supercomputer_name'] for d in st.session_state.supercomputer_system['deployments'].values()]
                
                if dates:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=list(range(len(dates))),
                        mode='markers+text',
                        marker=dict(size=15, color='rgb(240, 147, 251)'),
                        text=names,
                        textposition="top center"
                    ))
                    
                    fig.update_layout(
                        title="Chronologie des Déploiements",
                        xaxis_title="Date",
                        yaxis_title="Numéro de Déploiement",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export
                st.markdown("---")
                if st.button("📥 Exporter Historique"):
                    json_data = json.dumps(st.session_state.supercomputer_system['deployments'], indent=2, ensure_ascii=False, default=str)
                    st.download_button(
                        "💾 Télécharger JSON",
                        data=json_data,
                        file_name=f"deployments_history_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            else:
                st.info("Aucun historique de déploiement")

# ==================== PAGE: PROJETS ====================

elif page == "📁 Projets":
    st.header("📁 Gestion de Projets de Calcul Intensif")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Mes Projets", "➕ Nouveau Projet", "📊 Suivi Avancement", "🏆 Projets Terminés"])
    
    with tab1:
        st.subheader("📋 Projets Actifs")
        
        if not st.session_state.supercomputer_system['projects']:
            st.info("💡 Aucun projet créé. Créez votre premier projet de calcul intensif!")
        else:
            for project_id, project in st.session_state.supercomputer_system['projects'].items():
                if project['status'] != 'completed':
                    with st.expander(f"📁 {project['name']} - {project['status'].upper()}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Type:** {project.get('type', 'Non défini')}")
                            st.write(f"**Domaine:** {project.get('domain', 'Non défini')}")
                            st.write(f"**Créé le:** {project['created_at'][:10]}")
                        
                        with col2:
                            progress = project.get('progress', 0)
                            st.metric("Progression", f"{progress}%")
                            st.progress(progress / 100)
                        
                        with col3:
                            st.write(f"**Durée:** {project.get('duration', 0)} mois")
                            st.write(f"**Budget:** ${project.get('budget', 0):,}")
                            st.write(f"**Équipe:** {project.get('team_size', 0)} personnes")
                        
                        if project.get('description'):
                            st.markdown("---")
                            st.write("**Description:**")
                            st.write(project['description'])
                        
                        # Ressources allouées
                        if project.get('allocated_supercomputers'):
                            st.markdown("---")
                            st.write("**💻 Supercalculateurs Alloués:**")
                            for sc_id in project['allocated_supercomputers']:
                                if sc_id in st.session_state.supercomputer_system['supercomputers']:
                                    sc = st.session_state.supercomputer_system['supercomputers'][sc_id]
                                    st.write(f"• {sc['name']} - {format_flops(sc['peak_performance'])}")
                        
                        # Milestones
                        if project.get('milestones'):
                            st.markdown("---")
                            st.write("**🎯 Jalons du Projet:**")
                            
                            for i, milestone in enumerate(project['milestones'], 1):
                                milestone_status = milestone.get('status', 'pending')
                                icon = '✅' if milestone_status == 'completed' else '⏳' if milestone_status == 'in_progress' else '⏸️'
                                
                                st.write(f"{icon} **Jalon {i}:** {milestone['name']} - {milestone_status.upper()}")
                                if milestone.get('deadline'):
                                    st.caption(f"Échéance: {milestone['deadline']}")
                        
                        # Résultats et publications
                        if project.get('results'):
                            st.markdown("---")
                            st.write("**📊 Résultats:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Simulations", project['results'].get('simulations', 0))
                            with col2:
                                st.metric("Publications", project['results'].get('publications', 0))
                            with col3:
                                st.metric("Découvertes", project['results'].get('discoveries', 0))
                        
                        # Actions
                        st.markdown("---")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if st.button("📈 Mettre à Jour", key=f"update_{project_id}"):
                                st.session_state[f'edit_project_{project_id}'] = True
                        
                        with col2:
                            if st.button("✅ Marquer Terminé", key=f"complete_{project_id}"):
                                project['status'] = 'completed'
                                project['progress'] = 100
                                project['end_date'] = datetime.now().isoformat()
                                log_event(f"Projet terminé: {project['name']}")
                                st.success("Projet marqué comme terminé!")
                                st.rerun()
                        
                        with col3:
                            if st.button("⏸️ Suspendre", key=f"pause_{project_id}"):
                                project['status'] = 'paused'
                                log_event(f"Projet suspendu: {project['name']}")
                                st.rerun()
                        
                        with col4:
                            if st.button("🗑️ Supprimer", key=f"delete_{project_id}"):
                                del st.session_state.supercomputer_system['projects'][project_id]
                                log_event(f"Projet supprimé: {project['name']}")
                                st.rerun()
                        
                        # Formulaire de mise à jour
                        if st.session_state.get(f'edit_project_{project_id}'):
                            st.markdown("---")
                            st.write("### ✏️ Mettre à Jour le Projet")
                            
                            new_progress = st.slider("Progression", 0, 100, project.get('progress', 0), key=f"prog_{project_id}")
                            new_status = st.selectbox("Statut", ['active', 'paused', 'delayed'], key=f"stat_{project_id}")
                            
                            if st.button("💾 Sauvegarder", key=f"save_{project_id}"):
                                project['progress'] = new_progress
                                project['status'] = new_status
                                st.session_state[f'edit_project_{project_id}'] = False
                                st.success("Projet mis à jour!")
                                st.rerun()
    
    with tab2:
        st.subheader("➕ Créer un Nouveau Projet")
        
        with st.form("create_project_form"):
            st.write("### 📝 Informations Générales")
            
            col1, col2 = st.columns(2)
            
            with col1:
                project_name = st.text_input("Nom du Projet", placeholder="Ex: Simulation Climat 2025")
                
                project_type = st.selectbox(
                    "Type de Projet",
                    [
                        "Recherche Fondamentale",
                        "Recherche Appliquée",
                        "Développement Industriel",
                        "Simulation Scientifique",
                        "Entraînement IA",
                        "Analyse de Données",
                        "Prototype"
                    ]
                )
            
            with col2:
                project_domain = st.selectbox(
                    "Domaine d'Application",
                    [
                        "Recherche Scientifique",
                        "Modélisation Climat",
                        "Découverte Médicaments",
                        "Simulation Quantique",
                        "Intelligence Artificielle",
                        "Génomique",
                        "Astrophysique",
                        "Finance",
                        "Ingénierie",
                        "Autre"
                    ]
                )
                
                priority = st.select_slider("Priorité", ["Basse", "Normale", "Haute", "Critique"])
            
            project_description = st.text_area(
                "Description du Projet",
                placeholder="Décrivez les objectifs, la méthodologie et les résultats attendus...",
                height=120
            )
            
            st.markdown("---")
            st.write("### 💼 Ressources et Budget")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                duration = st.number_input("Durée (mois)", 1, 120, 12)
                team_size = st.number_input("Taille de l'Équipe", 1, 100, 5)
            
            with col2:
                budget = st.number_input("Budget Total ($)", 10000, 100000000, 100000, step=10000)
                compute_hours = st.number_input("Heures de Calcul Estimées", 100, 10000000, 10000)
            
            with col3:
                deadline = st.date_input("Date Limite", value=datetime.now() + timedelta(days=365))
            
            # Allocation de supercalculateurs
            st.markdown("---")
            st.write("### 💻 Allocation de Ressources")
            
            if st.session_state.supercomputer_system['supercomputers']:
                available_sc = {
                    sc['id']: f"{sc['name']} - {format_flops(sc['peak_performance'])}"
                    for sc in st.session_state.supercomputer_system['supercomputers'].values()
                }
                
                allocated_sc = st.multiselect(
                    "Supercalculateurs à Allouer",
                    options=list(available_sc.keys()),
                    format_func=lambda x: available_sc[x]
                )
            else:
                allocated_sc = []
                st.info("Aucun supercalculateur disponible")
            
            st.markdown("---")
            st.write("### 🎯 Jalons du Projet")
            
            num_milestones = st.number_input("Nombre de Jalons", 1, 20, 3)
            
            milestones = []
            for i in range(num_milestones):
                with st.expander(f"Jalon {i+1}", expanded=i < 2):
                    milestone_name = st.text_input(f"Nom du Jalon {i+1}", f"Jalon {i+1}", key=f"milestone_name_{i}")
                    milestone_desc = st.text_area(f"Description", key=f"milestone_desc_{i}")
                    milestone_deadline = st.date_input(f"Échéance", value=datetime.now() + timedelta(days=30*(i+1)), key=f"milestone_date_{i}")
                    
                    milestones.append({
                        'name': milestone_name,
                        'description': milestone_desc,
                        'deadline': milestone_deadline.isoformat(),
                        'status': 'pending'
                    })
            
            st.markdown("---")
            st.write("### 👥 Collaborateurs et Partenaires")
            
            col1, col2 = st.columns(2)
            
            with col1:
                principal_investigator = st.text_input("Chercheur Principal")
                institution = st.text_input("Institution", placeholder="Université, Laboratoire...")
            
            with col2:
                partners = st.text_area("Partenaires", placeholder="Liste des organisations partenaires (une par ligne)")
                funding_source = st.text_input("Source de Financement")
            
            submitted = st.form_submit_button("🚀 Créer le Projet", use_container_width=True, type="primary")
            
            if submitted:
                if not project_name:
                    st.error("⚠️ Veuillez donner un nom au projet")
                else:
                    project_id = f"proj_{len(st.session_state.supercomputer_system['projects']) + 1}"
                    
                    new_project = {
                        'id': project_id,
                        'name': project_name,
                        'type': project_type,
                        'domain': project_domain,
                        'priority': priority,
                        'description': project_description,
                        'duration': duration,
                        'team_size': team_size,
                        'budget': budget,
                        'compute_hours': compute_hours,
                        'deadline': deadline.isoformat(),
                        'allocated_supercomputers': allocated_sc,
                        'milestones': milestones,
                        'principal_investigator': principal_investigator,
                        'institution': institution,
                        'partners': partners.split('\n') if partners else [],
                        'funding_source': funding_source,
                        'status': 'active',
                        'progress': 0,
                        'created_at': datetime.now().isoformat(),
                        'results': {
                            'simulations': 0,
                            'publications': 0,
                            'discoveries': 0
                        }
                    }
                    
                    st.session_state.supercomputer_system['projects'][project_id] = new_project
                    
                    st.success(f"✅ Projet '{project_name}' créé avec succès!")
                    st.balloons()
                    
                    st.code(f"Project ID: {project_id}")
                    
                    log_event(f"Nouveau projet créé: {project_name}")
                    
                    # Afficher résumé
                    st.markdown("---")
                    st.write("### 📋 Résumé du Projet")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Durée:** {duration} mois")
                        st.write(f"**Équipe:** {team_size} personnes")
                    
                    with col2:
                        st.write(f"**Budget:** ${budget:,}")
                        st.write(f"**Heures Calcul:** {compute_hours:,}h")
                    
                    with col3:
                        st.write(f"**Jalons:** {len(milestones)}")
                        st.write(f"**Supercalculateurs:** {len(allocated_sc)}")
    
    with tab3:
        st.subheader("📊 Suivi de l'Avancement Global")
        
        if st.session_state.supercomputer_system['projects']:
            # Statistiques globales
            col1, col2, col3, col4 = st.columns(4)
            
            total_projects = len(st.session_state.supercomputer_system['projects'])
            active_projects = sum(1 for p in st.session_state.supercomputer_system['projects'].values() if p['status'] == 'active')
            completed_projects = sum(1 for p in st.session_state.supercomputer_system['projects'].values() if p['status'] == 'completed')
            avg_progress = np.mean([p.get('progress', 0) for p in st.session_state.supercomputer_system['projects'].values()])
            
            with col1:
                st.metric("Total Projets", total_projects)
            
            with col2:
                st.metric("Projets Actifs", active_projects)
            
            with col3:
                st.metric("Projets Terminés", completed_projects)
            
            with col4:
                st.metric("Progression Moyenne", f"{avg_progress:.0f}%")
            
            # Graphiques
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Progression par projet
                project_names = [p['name'][:20] for p in st.session_state.supercomputer_system['projects'].values()]
                project_progress = [p.get('progress', 0) for p in st.session_state.supercomputer_system['projects'].values()]
                
                fig = go.Figure(data=[
                    go.Bar(x=project_names, y=project_progress, marker_color='rgb(240, 147, 251)')
                ])
                fig.update_layout(title="Progression des Projets", yaxis_title="Progression (%)", xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Répartition par domaine
                domain_counts = {}
                for p in st.session_state.supercomputer_system['projects'].values():
                    domain = p.get('domain', 'Non défini')
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                fig = px.pie(
                    values=list(domain_counts.values()),
                    names=list(domain_counts.keys()),
                    title="Répartition par Domaine"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Budget et ressources
            st.markdown("---")
            st.write("### 💰 Ressources Allouées")
            
            col1, col2, col3 = st.columns(3)
            
            total_budget = sum(p.get('budget', 0) for p in st.session_state.supercomputer_system['projects'].values())
            total_compute_hours = sum(p.get('compute_hours', 0) for p in st.session_state.supercomputer_system['projects'].values())
            total_team = sum(p.get('team_size', 0) for p in st.session_state.supercomputer_system['projects'].values())
            
            with col1:
                st.metric("Budget Total", f"${total_budget:,}")
            
            with col2:
                st.metric("Heures de Calcul", f"{total_compute_hours:,}h")
            
            with col3:
                st.metric("Personnel Total", f"{total_team} personnes")
        
        else:
            st.info("Aucun projet à suivre")
    
    with tab4:
        st.subheader("🏆 Projets Terminés et Résultats")
        
        completed_projects = {k: v for k, v in st.session_state.supercomputer_system['projects'].items() if v['status'] == 'completed'}
        
        if not completed_projects:
            st.info("Aucun projet terminé pour le moment")
        else:
            for project_id, project in completed_projects.items():
                with st.expander(f"🏆 {project['name']}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Type:** {project.get('type', 'Non défini')}")
                        st.write(f"**Domaine:** {project.get('domain', 'Non défini')}")
                        st.write(f"**Durée:** {project.get('duration', 0)} mois")
                    
                    with col2:
                        st.write(f"**Date Fin:** {project.get('end_date', 'N/A')[:10]}")
                        st.write(f"**Budget:** ${project.get('budget', 0):,}")
                        st.write(f"**Équipe:** {project.get('team_size', 0)} personnes")
                    
                    with col3:
                        if project.get('results'):
                            st.metric("Simulations", project['results'].get('simulations', 0))
                            st.metric("Publications", project['results'].get('publications', 0))
                            st.metric("Découvertes", project['results'].get('discoveries', 0))
                    
                    if project.get('description'):
                        st.markdown("---")
                        st.write("**Description:**")
                        st.write(project['description'])
                    
                    # Impact
                    st.markdown("---")
                    st.write("**📊 Impact et Résultats:**")
                    
                    impact_score = np.random.randint(50, 100)
                    st.progress(impact_score / 100, text=f"Score d'Impact: {impact_score}/100")
                    
                    # Export rapport
                    if st.button(f"📥 Télécharger Rapport", key=f"report_{project_id}"):
                        report = {
                            'project': project,
                            'generated_at': datetime.now().isoformat()
                        }
                        
                        json_data = json.dumps(report, indent=2, ensure_ascii=False, default=str)
                        st.download_button(
                            "💾 Télécharger JSON",
                            data=json_data,
                            file_name=f"project_report_{project['name']}_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            key=f"download_{project_id}"
                        )
            
            # Statistiques globales
            st.markdown("---")
            st.write("### 📈 Statistiques Globales des Projets Terminés")
            
            total_publications = sum(p.get('results', {}).get('publications', 0) for p in completed_projects.values())
            total_discoveries = sum(p.get('results', {}).get('discoveries', 0) for p in completed_projects.values())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Projets Terminés", len(completed_projects))
            
            with col2:
                st.metric("Publications Totales", total_publications)
            
            with col3:
                st.metric("Découvertes Totales", total_discoveries)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>⚡ Plateforme Supercalculateur Quantique-Biologique</h3>
        <p>Système Complet de Création, Fabrication et Déploiement de Supercalculateurs Avancés</p>
        <p><small>Version 1.0.0 | Architecture Hybride Quantique-Biologique</small></p>
        <p><small>⚛️ Quantum Computing | 🧬 Biological Computing | 💫 Exascale to Yottascale</small></p>
        <p><small>🌍 Green Computing | 📊 TOP500 Ready | 🔬 Research Excellence</small></p>
    </div>
""", unsafe_allow_html=True)