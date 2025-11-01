"""
Frontend Streamlit - Moteur IA et Quantique d'Optimisation V2.0
Interface complète et professionnelle
streamlit run optimisation_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import numpy as np

# Configuration
st.set_page_config(
    page_title="Quantum Performance Optimization V2",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS avancé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem;
        animation: gradient 3s ease infinite;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# État de session
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8035"
if 'selected_system' not in st.session_state:
    st.session_state.selected_system = None
if 'selected_strategy' not in st.session_state:
    st.session_state.selected_strategy = None
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False

if 'strategies_cache' not in st.session_state:
    st.session_state.strategies_cache = []
if 'systems_cache' not in st.session_state:
    st.session_state.systems_cache = []
# ==================== FONCTIONS UTILITAIRES ====================
def api_request(endpoint, method="GET", data=None, show_error=True):
    """Effectue une requête API avec gestion d'erreurs"""
    try:
        url = f"{st.session_state.api_url}{endpoint}"
        
        # ✅ Ajouter headers pour éviter le cache
        headers = {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache'
        }
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        
        # ✅ Vérifier le status
        response.raise_for_status()
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            if show_error:
                st.error(f"❌ Erreur API ({response.status_code}): {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        if show_error:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez qu'elle est démarrée.")
        return None
    except requests.exceptions.Timeout:
        if show_error:
            st.error("❌ Timeout - L'API met trop de temps à répondre")
        return None
    except Exception as e:
        if show_error:
            st.error(f"❌ Erreur: {str(e)}")
        return None

def create_gauge_chart(value, title, max_value=100, color='blue'):
    """Crée un graphique de jauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': max_value * 0.7},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': "lightgray"},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_line_chart(data, x_col, y_col, title):
    """Crée un graphique linéaire"""
    fig = px.line(data, x=x_col, y=y_col, title=title)
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown('<h2 style="text-align: center;">⚡ Quantum Engine V2</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Menu principal
    menu = st.radio(
        "🧭 Navigation",
        [
            "🏠 Tableau de Bord",
            "🎯 Stratégies",
            "💻 Systèmes",
            "📊 Benchmarking",
            "🤖 Optimisations IA",
            "⚛️ Optimisations Quantiques",
            "⚙️ Ordonnancement",
            "📈 Profilage",
            "⚖️ Load Balancing",
            "💾 Cache Intelligent",
            "🗜️ Compression",
            "⚡ Énergie",
            "🔮 Maintenance Prédictive",
            "🎛️ Auto-Tuning",
            "📑 Rapports",
            "📊 Analytics",
            "🔧 Administration"
        ]
    )
    
    st.markdown("---")
    
    # Statut API
    health = api_request("/health", show_error=False)
    if health:
        st.success("✅ API Connectée")
        st.info(f"**Version:** {health.get('version', 'N/A')}")
        
        if 'components' in health:
            for comp, status in health['components'].items():
                if status == 'operational':
                    st.text(f"✓ {comp}")
    else:
        st.error("❌ API Déconnectée")
        st.warning("Démarrez l'API avec:\n`uvicorn quantum_performance_api_v2:app --reload`")
    
    st.markdown("---")
    
    # Statistiques en temps réel
    stats = api_request("/api/stats", show_error=False)
    if stats and 'database_stats' in stats:
        db = stats['database_stats']
        st.metric("📊 Stratégies", db['strategies']['total'], 
                 delta=db['strategies'].get('active', 0))
        st.metric("💻 Systèmes", db['systems']['total'],
                 delta=db['systems'].get('online', 0))
        st.metric("🚀 Optimisations", db['optimizations']['total'])
        st.metric("🤖 Modèles IA", db['ai_models']['total'])

# ==================== TABLEAU DE BORD ====================

if menu == "🏠 Tableau de Bord":
    st.markdown('<h1 class="main-header">🏠 Tableau de Bord Global</h1>', unsafe_allow_html=True)
    
    # Rafraîchissement automatique
    col1, col2 = st.columns([4, 1])
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
    
    if auto_refresh:
        time.sleep(2)
        st.rerun()
    
    analytics = api_request("/api/analytics/global")
    
    if analytics:
        # Métriques principales avec style
        st.markdown("### 📊 Métriques Principales")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Stratégies</h3>
                <h1>{analytics['total_strategies']}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💻 Systèmes</h3>
                <h1>{analytics['total_systems']}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Benchmarks</h3>
                <h1>{analytics['total_benchmarks']}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🚀 Optimisations</h3>
                <h1>{analytics['total_optimizations']}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performances
        st.markdown("### 🎯 Performances Globales")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = create_gauge_chart(
                analytics['average_performance_improvement'],
                "Amélioration Performances (%)",
                max_value=100,
                color='#667eea'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tendances temporelles
        st.markdown("### 📅 Tendances Temporelles")
        
        # Simuler des données historiques
        days = 30
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        trends_data = pd.DataFrame({
            'Date': dates,
            'Optimisations': np.cumsum(np.random.randint(1, 5, days)),
            'Performance': np.cumsum(np.random.uniform(0.5, 2, days)),
            'Économies': np.cumsum(np.random.uniform(0.3, 1.5, days))
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trends_data['Date'], y=trends_data['Optimisations'],
                                name='Optimisations', mode='lines+markers', line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=trends_data['Date'], y=trends_data['Performance'],
                                name='Performance (x10)', mode='lines+markers', line=dict(color='#38ef7d', width=3)))
        fig.add_trace(go.Scatter(x=trends_data['Date'], y=trends_data['Économies'],
                                name='Économies (x10)', mode='lines+markers', line=dict(color='#f093fb', width=3)))
        
        fig.update_layout(
            title="Évolution sur 30 jours",
            xaxis_title="Date",
            yaxis_title="Valeur",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des systèmes
        systems = api_request("/api/system/list", show_error=False) or []
        if systems:
            st.markdown("### 💻 Distribution des Systèmes")
            
            system_types = {}
            for sys in systems:
                sys_type = sys.get('type', 'unknown')
                system_types[sys_type] = system_types.get(sys_type, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Pie(
                    labels=list(system_types.keys()),
                    values=list(system_types.values()),
                    hole=.4,
                    marker=dict(colors=['#667eea', '#764ba2', '#f093fb'])
                )])
                fig.update_layout(title="Répartition par Type", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Statut des systèmes
                status_online = sum(1 for s in systems if s.get('status') in ['online', 'calibrated', 'ready'])
                status_data = pd.DataFrame({
                    'Statut': ['Online', 'Offline'],
                    'Nombre': [status_online, len(systems) - status_online]
                })
                
                fig = go.Figure(data=[go.Bar(
                    x=status_data['Statut'],
                    y=status_data['Nombre'],
                    marker_color=['#38ef7d', '#f5576c']
                )])
                fig.update_layout(title="Statut des Systèmes", height=400)
                st.plotly_chart(fig, use_container_width=True)

# ==================== ADMINISTRATION ====================

elif menu == "🔧 Administration":
    st.title("🔧 Administration du Système")
    
    st.warning("⚠️ Zone d'administration - Utilisez ces fonctions avec précaution")
    
    tab1, tab2, tab3 = st.tabs(["🗑️ Nettoyage", "📊 Statistiques", "💾 Export/Import"])
    
    with tab1:
        st.markdown("### 🗑️ Nettoyage de la Base de Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Nettoyage Sélectif")
            if st.button("🧹 Nettoyer les Anciennes Données", use_container_width=True):
                with st.spinner("Nettoyage en cours..."):
                    result = api_request("/api/maintenance/cleanup", "POST")
                
                if result:
                    st.success("✅ Nettoyage effectué!")
                    st.json(result)
        
        with col2:
            st.markdown("#### Réinitialisation Complète")
            st.error("🚨 Cette action est IRRÉVERSIBLE!")
            
            if st.checkbox("Je confirme vouloir tout supprimer"):
                if st.button("💣 RÉINITIALISER TOUT", use_container_width=True, type="secondary"):
                    with st.spinner("Réinitialisation..."):
                        result = api_request("/api/maintenance/reset", "POST")
                    
                    if result:
                        st.success("✅ Base de données réinitialisée!")
                        st.json(result)
                        time.sleep(2)
                        st.rerun()
    
    with tab2:
        st.markdown("### 📊 Statistiques Détaillées")
        
        if st.button("🔄 Rafraîchir les Statistiques", use_container_width=True):
            st.rerun()
        
        stats = api_request("/api/stats")
        
        if stats:
            db_stats = stats.get('database_stats', {})
            
            # Stratégies
            st.markdown("#### 🎯 Stratégies")
            strat_stats = db_stats.get('strategies', {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", strat_stats.get('total', 0))
            with col2:
                st.metric("Actives", strat_stats.get('active', 0))
            
            # Systèmes
            st.markdown("#### 💻 Systèmes")
            sys_stats = db_stats.get('systems', {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", sys_stats.get('total', 0))
            with col2:
                st.metric("En Ligne", sys_stats.get('online', 0))
            
            by_type = sys_stats.get('by_type', {})
            if by_type:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Binary", by_type.get('binary', 0))
                with col2:
                    st.metric("Quantum", by_type.get('quantum', 0))
                with col3:
                    st.metric("Hybrid", by_type.get('hybrid', 0))
            
            # Autres statistiques
            st.markdown("#### 📈 Autres Métriques")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Benchmarks", db_stats.get('benchmarks', {}).get('total', 0))
            with col2:
                st.metric("Optimisations", db_stats.get('optimizations', {}).get('total', 0))
            with col3:
                st.metric("Modèles IA", db_stats.get('ai_models', {}).get('total', 0))
            
            # Performances moyennes
            perf_stats = stats.get('performance_metrics', {})
            if perf_stats:
                st.markdown("#### 🎯 Performances")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Optimisations", perf_stats.get('total_optimizations', 0))
                with col2:
                    st.metric("Amélioration Moyenne", f"{perf_stats.get('avg_improvement', 0):.1f}%")
    
    with tab3:
        st.markdown("### 💾 Export / Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 Export")
            if st.button("📥 Exporter Toutes les Données", use_container_width=True):
                with st.spinner("Export en cours..."):
                    result = api_request("/api/export/all")
                
                if result:
                    st.success("✅ Export terminé!")
                    
                    # Bouton de téléchargement
                    export_json = json.dumps(result, indent=2)
                    st.download_button(
                        label="💾 Télécharger l'Export",
                        data=export_json,
                        file_name=f"quantum_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    # Statistiques de l'export
                    counts = result.get('counts', {})
                    st.info(f"""
                    **Export contient:**
                    - {counts.get('strategies', 0)} stratégies
                    - {counts.get('systems', 0)} systèmes
                    - {counts.get('benchmarks', 0)} benchmarks
                    - {counts.get('optimizations', 0)} optimisations
                    - {counts.get('ai_models', 0)} modèles IA
                    - {counts.get('profiles', 0)} profils
                    """)
        
        with col2:
            st.markdown("#### 📥 Import")
            uploaded_file = st.file_uploader("Choisir un fichier JSON", type=['json'])
            
            if uploaded_file is not None:
                try:
                    import_data = json.load(uploaded_file)
                    st.success("✅ Fichier chargé!")
                    
                    st.info(f"""
                    **Données détectées:**
                    - Version: {import_data.get('version', 'N/A')}
                    - Export ID: {import_data.get('export_id', 'N/A')[:12]}...
                    - Date: {import_data.get('exported_at', 'N/A')[:10]}
                    """)
                    
                    if st.button("📥 Importer les Données", use_container_width=True):
                        st.warning("⚠️ Fonctionnalité d'import à implémenter côté API")
                
                except json.JSONDecodeError:
                    st.error("❌ Fichier JSON invalide")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")


# ==================== STRATÉGIES ====================

elif menu == "🎯 Stratégies":
    st.title("🎯 Stratégies d'Optimisation")
    
    tab1, tab2, tab3 = st.tabs(["➕ Créer", "📋 Gérer", "🎬 Appliquer"])
    
    with tab1:
        st.markdown("### Créer une Nouvelle Stratégie")
        with st.form("strategy_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("📝 Nom de la stratégie", placeholder="Ma Stratégie Performante")
                target_system = st.selectbox("🎯 Système Cible", 
                    ["binary", "quantum", "hybrid"],
                    help="Type de système à optimiser")
                
                optimization_targets = st.multiselect(
                    "🎯 Cibles d'Optimisation",
                    ["cpu", "memory", "io", "network", "energy", "qubits", "latency", "throughput"],
                    default=["cpu", "memory"]
                )
            
            with col2:
                ai_algorithms = st.multiselect(
                    "🤖 Algorithmes IA",
                    ["reinforcement_learning", "genetic_algorithm", "neural_network", 
                     "swarm_intelligence", "deep_learning", "ensemble"],
                    default=["reinforcement_learning"]
                )
                
                quantum_algorithms = st.multiselect(
                    "⚛️ Algorithmes Quantiques",
                    ["quantum_annealing", "qaoa", "vqe", "grover", "quantum_ml", "shor"],
                    default=["qaoa"]
                )
            
            description = st.text_area("📄 Description", 
                placeholder="Décrivez votre stratégie d'optimisation...",
                height=100)
            
            col1, col2, col3 = st.columns(3)
            with col2:
                submitted = st.form_submit_button("🚀 Créer la Stratégie", use_container_width=True)
            
            if submitted:
                if len(name) >= 3 and len(description) >= 10:
                    data = {
                        'name': name,
                        'description': description,
                        'target_system': target_system,
                        'optimization_targets': optimization_targets,
                        'ai_algorithms': ai_algorithms,
                        'quantum_algorithms': quantum_algorithms
                    }
                    
                    with st.spinner("Création en cours..."):
                        result = api_request("/api/strategy/create", "POST", data)
                    
                    if result:
                        st.success("✅ Stratégie créée avec succès!")
                        st.balloons()
                        with st.expander("📋 Détails de la stratégie créée"):
                            st.json(result)
                        # ✅ CORRECTION: Attendre un peu et recharger
                        time.sleep(0.5)  # Petit délai pour que l'API persiste
                        st.rerun()  # Recharger APRÈS la création
                else:
                    st.error("❌ Le nom doit contenir au moins 3 caractères et la description 10 caractères")
    
    with tab2:
        st.markdown("### Gérer les Stratégies Existantes")
        # strategies = api_request("/api/strategy/list", show_error=False) or []
        strategies = api_request("/api/strategy/list", show_error=False)
        if strategies:
            st.session_state.strategies_cache = strategies
        else:
            strategies = st.session_state.strategies_cache
        
        if strategies:
            # Filtres
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_type = st.selectbox("Filtrer par type", 
                    ["Tous", "binary", "quantum", "hybrid"])
            with col2:
                sort_by = st.selectbox("Trier par", 
                    ["Nom", "Date création", "Performance"])
            
            for idx, s in enumerate(strategies):
                if filter_type != "Tous" and s['target_system'] != filter_type:
                    continue
                
                with st.expander(f"🎯 {s['name']} ({s['target_system'].upper()})", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**📋 ID:** `{s['strategy_id'][:12]}...`")
                        st.write(f"**🎯 Système:** {s['target_system']}")
                        st.write(f"**📊 Status:** {s['status']}")
                        st.write(f"**📅 Créée le:** {s.get('created_at', 'N/A')[:10]}")
                        st.write(f"**🔢 Applications:** {s.get('applications_count', 0)}")
                    
                    with col2:
                        st.write(f"**🎯 Cibles:** {', '.join(s['optimization_targets'])}")
                        st.write(f"**🤖 IA:** {', '.join(s['ai_algorithms'])}")
                        st.write(f"**⚛️ Quantum:** {', '.join(s['quantum_algorithms'])}")
                        
                        if s.get('performance_improvement', 0) > 0:
                            st.metric("Performance", f"+{s['performance_improvement']:.1f}%")
                    
                    st.markdown("**📄 Description:**")
                    st.info(s['description'])
                    
                    # Actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📊 Analytics", key=f"analytics_{idx}"):
                            result = api_request(f"/api/analytics/strategy/{s['strategy_id']}")
                            if result:
                                st.json(result)
                    
                    with col2:
                        if st.button("🎬 Appliquer", key=f"apply_{idx}"):
                            st.session_state.selected_strategy = s['strategy_id']
                            st.info("Allez dans l'onglet 'Appliquer' pour sélectionner un système")
                    
                    with col3:
                        if st.button("🗑️ Supprimer", key=f"del_{idx}", type="secondary"):
                            if st.session_state.get(f'confirm_del_{idx}', False):
                                api_request(f"/api/strategy/{s['strategy_id']}", "DELETE")
                                st.success("Stratégie supprimée")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.session_state[f'confirm_del_{idx}'] = True
                                st.warning("Cliquez à nouveau pour confirmer")
        else:
            st.info("📭 Aucune stratégie créée. Créez-en une dans l'onglet 'Créer'!")
    
    with tab3:
        st.markdown("### Appliquer une Stratégie")
        # ✅ Forcer le rechargement des systèmes
        if st.button("🔄 Actualiser la Liste", key="refresh_systems_monitor"):
            st.rerun()
        
        systems = api_request("/api/system/list", show_error=False) or []
        
        if systems:
            system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
            selected = st.selectbox("Sélectionnez un système", 
                options=list(system_names.keys()),
                format_func=lambda x: system_names[x],
                key="monitor_system_select")
            
        strategies = api_request("/api/strategy/list", show_error=False) or []
        systems = api_request("/api/system/list", show_error=False) or []
        
        if strategies and systems:
            col1, col2 = st.columns(2)
            
            with col1:
                strategy_names = {s['strategy_id']: s['name'] for s in strategies}
                selected_strategy = st.selectbox(
                    "🎯 Sélectionnez une stratégie",
                    options=list(strategy_names.keys()),
                    format_func=lambda x: strategy_names[x]
                )
            
            with col2:
                system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
                selected_system = st.selectbox(
                    "💻 Sélectionnez un système",
                    options=list(system_names.keys()),
                    format_func=lambda x: system_names[x]
                )
            
            if st.button("🚀 Appliquer la Stratégie", use_container_width=True, type="primary"):
                with st.spinner("Application en cours..."):
                    result = api_request(
                        f"/api/strategy/{selected_strategy}/apply",
                        "POST",
                        {'target_system_id': selected_system}
                    )
                
                if result:
                    st.success("✅ Stratégie appliquée avec succès!")
                    st.balloons()
                    
                    # Affichage des résultats
                    st.markdown("### 📊 Résultats de l'Application")
                    
                    improvements = result.get('improvements', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CPU", f"+{improvements.get('cpu_improvement', 0):.1f}%")
                        st.metric("Memory", f"+{improvements.get('memory_improvement', 0):.1f}%")
                    with col2:
                        st.metric("I/O", f"+{improvements.get('io_improvement', 0):.1f}%")
                        st.metric("Énergie", f"-{improvements.get('energy_savings', 0):.1f}%")
                    with col3:
                        st.metric("Response Time", f"-{improvements.get('response_time_reduction', 0):.1f}%")
                        st.metric("Throughput", f"+{improvements.get('throughput_increase', 0):.1f}%")
                    
                    st.markdown("---")
                    st.metric("🎯 Gain Global de Performance", 
                             f"+{result.get('overall_performance_gain', 0):.1f}%",
                             delta=f"+{result.get('overall_performance_gain', 0):.1f}%")
                    
                    # Graphique radar des améliorations
                    categories = list(improvements.keys())
                    values = list(improvements.values())
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        line=dict(color='#667eea', width=2)
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, max(values) * 1.2])),
                        showlegend=False,
                        title="Répartition des Améliorations",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            if not strategies:
                st.warning("⚠️ Créez d'abord des stratégies")
            if not systems:
                st.warning("⚠️ Créez d'abord des systèmes")

# ==================== SYSTÈMES ====================

elif menu == "💻 Systèmes":
    st.title("💻 Gestion des Systèmes")
    
    tab1, tab2, tab3 = st.tabs(["➕ Créer", "📋 Gérer", "📊 Surveiller"])
    
    with tab1:
        st.markdown("### Créer un Nouveau Système")
        
        system_type = st.selectbox(
            "🔧 Type de Système",
            ["binary", "quantum", "hybrid"],
            help="Binary: Système classique | Quantum: Système quantique | Hybrid: Combinaison"
        )
        
        with st.form("system_form"):
            name = st.text_input("📝 Nom du système", placeholder="Mon Serveur Production")
            
            if system_type == "binary":
                st.markdown("#### ⚙️ Spécifications Classiques")
                col1, col2 = st.columns(2)
                with col1:
                    cpu_cores = st.number_input("CPU Cores", min_value=1, value=16)
                    cpu_freq = st.number_input("CPU Fréquence (GHz)", min_value=0.1, value=3.5, step=0.1)
                    memory_gb = st.number_input("Mémoire (GB)", min_value=1, value=64)
                with col2:
                    storage_gb = st.number_input("Stockage (GB)", min_value=1, value=1000)
                    gpu_count = st.number_input("Nombre de GPU", min_value=0, value=2)
                    network_bw = st.number_input("Bande Passante (Gbps)", min_value=1, value=10)
                
                specs = {
                    'cpu_cores': cpu_cores,
                    'cpu_frequency': cpu_freq,
                    'memory_gb': memory_gb,
                    'storage_gb': storage_gb,
                    'gpu_count': gpu_count,
                    'network_bandwidth': network_bw
                }
            
            elif system_type == "quantum":
                st.markdown("#### ⚛️ Spécifications Quantiques")
                col1, col2 = st.columns(2)
                with col1:
                    qubits = st.number_input("Nombre de Qubits", min_value=1, value=50)
                    qubit_type = st.selectbox("Type de Qubits", 
                        ["superconducting", "ion_trap", "photonic", "topological"])
                with col2:
                    connectivity = st.selectbox("Connectivité", 
                        ["all_to_all", "nearest_neighbor", "ring", "custom"])
                
                specs = {
                    'qubits': qubits,
                    'qubit_type': qubit_type,
                    'connectivity': connectivity
                }
            
            else:  # hybrid
                st.markdown("#### 🔄 Spécifications Hybrides")
                col1, col2 = st.columns(2)
                with col1:
                    cpu_cores = st.number_input("CPU Cores", min_value=1, value=32)
                    memory_gb = st.number_input("Mémoire (GB)", min_value=1, value=128)
                with col2:
                    qubits = st.number_input("Nombre de Qubits", min_value=1, value=20)
                
                specs = {
                    'cpu_cores': cpu_cores,
                    'memory_gb': memory_gb,
                    'qubits': qubits
                }
            
            submitted = st.form_submit_button("🚀 Créer le Système", use_container_width=True)
            
            if submitted and name:
                data = {
                    'system_type': system_type,
                    'name': name,
                    **specs
                }
                
                with st.spinner("Création du système..."):
                    result = api_request("/api/system/create", "POST", data)
                
                if result:
                    st.success("✅ Système créé avec succès!")
                    st.balloons()
                    with st.expander("📋 Détails du système créé"):
                        st.json(result)
                    # ✅ CORRECTION: Attendre et recharger
                    time.sleep(0.5)
                    st.rerun()
    
    with tab2:
        st.markdown("### Gérer les Systèmes Existants")
        
        systems = api_request("/api/system/list", show_error=False) or []
        
        if systems:
            # Filtres
            col1, col2 = st.columns([1, 3])
            with col1:
                filter_type = st.selectbox("Type", ["Tous", "binary", "quantum", "hybrid"])
            
            filtered_systems = [s for s in systems if filter_type == "Tous" or s['type'] == filter_type]
            
            for idx, sys in enumerate(filtered_systems):
                icon = {"binary": "💻", "quantum": "⚛️", "hybrid": "🔄"}[sys['type']]
                
                with st.expander(f"{icon} {sys['name']} ({sys['type'].upper()})", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**🆔 ID:** `{sys['system_id'][:12]}...`")
                        st.write(f"**📊 Status:** {sys['status']}")
                        st.write(f"**📅 Créé le:** {sys.get('created_at', 'N/A')[:10]}")
                    
                    with col2:
                        specs = sys.get('specifications', {})
                        st.write("**⚙️ Spécifications:**")
                        for key, value in list(specs.items())[:5]:
                            st.write(f"• {key}: {value}")
                    
                    # Actions
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("📊 Analytics", key=f"sys_analytics_{idx}"):
                            result = api_request(f"/api/analytics/system/{sys['system_id']}")
                            if result:
                                st.json(result)
                    with col2:
                        if st.button("👁️ Surveiller", key=f"monitor_{idx}"):
                            st.session_state.selected_system = sys['system_id']
                            st.info("Allez dans l'onglet 'Surveiller'")
                    with col3:
                        if st.button("📈 Profiler", key=f"profile_{idx}"):
                            with st.spinner("Création du profil..."):
                                result = api_request("/api/profile/create", "POST", 
                                    {'system_id': sys['system_id'], 'duration_seconds': 30})
                            if result:
                                st.success("Profil créé!")
                    with col4:
                        if st.button("🗑️ Supprimer", key=f"sys_del_{idx}"):
                            api_request(f"/api/system/{sys['system_id']}", "DELETE")
                            st.success("Système supprimé")
                            time.sleep(1)
                            st.rerun()
        else:
            st.info("📭 Aucun système créé")
    
    with tab3:
        st.markdown("### 📊 Surveillance en Temps Réel")
        
        systems = api_request("/api/system/list", show_error=False) or []
        
        if systems:
            system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
            selected = st.selectbox("Sélectionnez un système", 
                options=list(system_names.keys()),
                format_func=lambda x: system_names[x])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                duration = st.slider("Durée (secondes)", 10, 300, 60)
            with col2:
                auto_refresh = st.checkbox("🔄 Rafraîchir auto", value=False)
            
            if st.button("🚀 Démarrer la Surveillance", use_container_width=True) or auto_refresh:
                with st.spinner("Collecte des données..."):
                    result = api_request(f"/api/system/{selected}/monitor", "POST",
                        {'duration_seconds': duration})
                
                if result:
                    st.success("✅ Données collectées!")
                    
                    timeline = result.get('timeline', [])
                    if timeline:
                        df = pd.DataFrame(timeline)
                        
                        # Déterminer les colonnes selon le type
                        system = next(s for s in systems if s['system_id'] == selected)
                        
                        if system['type'] == 'binary':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df['second'], y=df['cpu_usage'],
                                name='CPU Usage (%)', line=dict(color='#667eea', width=2)))
                            fig.add_trace(go.Scatter(x=df['second'], y=df['memory_usage'],
                                name='Memory Usage (%)', line=dict(color='#38ef7d', width=2)))
                            fig.add_trace(go.Scatter(x=df['second'], y=df['power_consumption_w']/5,
                                name='Power (W/5)', line=dict(color='#f5576c', width=2)))
                            
                        elif system['type'] == 'quantum':
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df['second'], y=df['qubit_utilization'],
                                name='Qubit Utilization (%)', line=dict(color='#667eea', width=2)))
                            fig.add_trace(go.Scatter(x=df['second'], y=df['fidelity']*100,
                                name='Fidelity (%)', line=dict(color='#38ef7d', width=2)))
                            
                        else:  # hybrid
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df['second'], y=df['classical_usage'],
                                name='Classical Usage (%)', line=dict(color='#667eea', width=2)))
                            fig.add_trace(go.Scatter(x=df['second'], y=df['quantum_usage'],
                                name='Quantum Usage (%)', line=dict(color='#f093fb', width=2)))
                            fig.add_trace(go.Scatter(x=df['second'], y=df['hybrid_efficiency']*100,
                                name='Hybrid Efficiency (%)', line=dict(color='#38ef7d', width=2)))
                        
                        fig.update_layout(
                            title=f"Surveillance de {system['name']}",
                            xaxis_title="Temps (secondes)",
                            yaxis_title="Valeur",
                            height=500,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Résumé
                        st.markdown("### 📊 Résumé")
                        summary = result.get('summary', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Utilisation Moyenne", f"{summary.get('avg_utilization', 0):.1f}%")
                        with col2:
                            st.metric("Pic d'Utilisation", f"{summary.get('peak_utilization', 0):.1f}%")
                        with col3:
                            st.metric("Score Efficacité", f"{summary.get('efficiency_score', 0):.1f}")
                
                if auto_refresh:
                    time.sleep(5)
                    st.rerun()

# ==================== BENCHMARKING ====================

elif menu == "📊 Benchmarking":
    st.title("📊 Benchmarking de Performance")
    
    tab1, tab2, tab3, tab4 = st.tabs(["💻 CPU", "💾 Memory", "💿 I/O", "⚛️ Quantum"])
    
    with tab1:
        st.markdown("### Benchmark CPU")
        
        with st.form("cpu_benchmark"):
            col1, col2 = st.columns(2)
            with col1:
                num_threads = st.slider("Nombre de Threads", 1, 64, 8)
            with col2:
                duration = st.slider("Durée (secondes)", 10, 300, 60)
            
            if st.form_submit_button("🚀 Lancer le Benchmark CPU", use_container_width=True):
                with st.spinner("Benchmark en cours..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(duration / 100)
                        progress_bar.progress(i + 1)
                    
                    result = api_request("/api/benchmark/cpu", "POST", {
                        'num_threads': num_threads,
                        'duration_seconds': duration
                    })
                
                if result:
                    st.success("✅ Benchmark CPU terminé!")
                    
                    results = result.get('results', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Score Single-Core", f"{results.get('single_core_score', 0):.0f}")
                        st.metric("Score Multi-Core", f"{results.get('multi_core_score', 0):.0f}")
                    with col2:
                        st.metric("Integer Perf", f"{results.get('integer_performance', 0):.0f}")
                        st.metric("Float Perf", f"{results.get('floating_point_performance', 0):.0f}")
                    with col3:
                        st.metric("Memory BW", f"{results.get('memory_bandwidth_gbps', 0):.1f} GB/s")
                        st.metric("Cache Perf", f"{results.get('cache_performance', 0):.2%}")
                    
                    # Graphique radar
                    categories = ['Single-Core', 'Multi-Core', 'Integer', 'Float', 'Memory BW']
                    values = [
                        results.get('single_core_score', 0) / 30,
                        results.get('multi_core_score', 0) / 250,
                        results.get('integer_performance', 0) / 150,
                        results.get('floating_point_performance', 0) / 120,
                        results.get('memory_bandwidth_gbps', 0)
                    ]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        line=dict(color='#667eea', width=3)
                    ))
                    fig.update_layout(title="Profil de Performance CPU", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.metric("🎯 Score Global", f"{result.get('overall_score', 0):.0f}")
    
    with tab2:
        st.markdown("### Benchmark Memory")
        
        with st.form("memory_benchmark"):
            test_size = st.slider("Taille du Test (MB)", 128, 8192, 1024)
            
            if st.form_submit_button("🚀 Lancer le Benchmark Memory", use_container_width=True):
                with st.spinner("Benchmark en cours..."):
                    result = api_request("/api/benchmark/memory", "POST", {
                        'test_size_mb': test_size
                    })
                
                if result:
                    st.success("✅ Benchmark Memory terminé!")
                    
                    results = result.get('results', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sequential Read", f"{results.get('sequential_read_mbps', 0):.0f} MB/s")
                        st.metric("Sequential Write", f"{results.get('sequential_write_mbps', 0):.0f} MB/s")
                        st.metric("Random Read", f"{results.get('random_read_mbps', 0):.0f} MB/s")
                    with col2:
                        st.metric("Random Write", f"{results.get('random_write_mbps', 0):.0f} MB/s")
                        st.metric("Latency", f"{results.get('latency_ns', 0):.1f} ns")
                        st.metric("Bandwidth Efficiency", f"{results.get('bandwidth_efficiency', 0):.2%}")
                    
                    # Graphique en barres
                    fig = go.Figure(data=[
                        go.Bar(name='Read', x=['Sequential', 'Random'], 
                               y=[results.get('sequential_read_mbps', 0), results.get('random_read_mbps', 0)],
                               marker_color='#667eea'),
                        go.Bar(name='Write', x=['Sequential', 'Random'],
                               y=[results.get('sequential_write_mbps', 0), results.get('random_write_mbps', 0)],
                               marker_color='#38ef7d')
                    ])
                    fig.update_layout(title="Performance Memory (MB/s)", barmode='group', height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Benchmark I/O")
        
        with st.form("io_benchmark"):
            file_size = st.slider("Taille du Fichier (MB)", 100, 10000, 1000)
            
            if st.form_submit_button("🚀 Lancer le Benchmark I/O", use_container_width=True):
                with st.spinner("Benchmark en cours..."):
                    result = api_request("/api/benchmark/io", "POST", {
                        'file_size_mb': file_size
                    })
                
                if result:
                    st.success("✅ Benchmark I/O terminé!")
                    
                    results = result.get('results', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sequential Read", f"{results.get('sequential_read_mbps', 0):.0f} MB/s")
                        st.metric("Sequential Write", f"{results.get('sequential_write_mbps', 0):.0f} MB/s")
                        st.metric("Random Read IOPS", f"{results.get('random_read_iops', 0):.0f}")
                    with col2:
                        st.metric("Random Write IOPS", f"{results.get('random_write_iops', 0):.0f}")
                        st.metric("Access Latency", f"{results.get('access_latency_us', 0):.1f} µs")
                        st.metric("Queue Depth Optimal", f"{results.get('queue_depth_optimal', 0)}")
    
    with tab4:
        st.markdown("### Benchmark Quantum")
        
        with st.form("quantum_benchmark"):
            col1, col2 = st.columns(2)
            with col1:
                num_qubits = st.slider("Nombre de Qubits", 5, 100, 20)
            with col2:
                circuit_depth = st.slider("Profondeur du Circuit", 10, 200, 50)
            
            if st.form_submit_button("🚀 Lancer le Benchmark Quantum", use_container_width=True):
                with st.spinner("Benchmark quantique en cours..."):
                    result = api_request("/api/benchmark/quantum", "POST", {
                        'num_qubits': num_qubits,
                        'circuit_depth': circuit_depth
                    })
                
                if result:
                    st.success("✅ Benchmark Quantum terminé!")
                    
                    results = result.get('results', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Gate Fidelity", f"{results.get('gate_fidelity', 0):.4f}")
                        st.metric("Coherence Time", f"{results.get('coherence_time_us', 0):.1f} µs")
                    with col2:
                        st.metric("Gate Time", f"{results.get('gate_time_ns', 0):.1f} ns")
                        st.metric("Readout Fidelity", f"{results.get('readout_fidelity', 0):.4f}")
                    with col3:
                        st.metric("Crosstalk Suppression", f"{results.get('crosstalk_suppression_db', 0):.1f} dB")
                        st.metric("Quantum Volume", f"{results.get('quantum_volume', 0)}")
                    
                    st.metric("⚛️ Avantage Quantique Estimé", 
                             f"{result.get('quantum_advantage_estimate', 0):.1f}x",
                             delta=f"+{result.get('quantum_advantage_estimate', 0):.1f}x")

# ==================== OPTIMISATIONS IA ====================

elif menu == "🤖 Optimisations IA":
    st.title("🤖 Optimisations par Intelligence Artificielle")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 RL Scheduler", "🧬 Genetic", "🧠 Neural", "🐝 Swarm"])
    
    with tab1:
        st.markdown("### Reinforcement Learning - Ordonnancement")
        
        with st.form("rl_form"):
            col1, col2 = st.columns(2)
            with col1:
                num_tasks = st.number_input("Nombre de Tâches", 10, 10000, 100)
            with col2:
                resources = st.number_input("Nombre de Ressources", 1, 100, 10)
            
            if st.form_submit_button("🚀 Optimiser avec RL", use_container_width=True):
                with st.spinner("Apprentissage en cours..."):
                    result = api_request("/api/optimize/reinforcement-learning", "POST", {
                        'num_tasks': num_tasks,
                        'resources': resources
                    })
                
                if result:
                    st.success("✅ Optimisation RL terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Agent Type", result.get('agent_type', 'N/A'))
                        st.metric("Episodes", result.get('training_episodes', 0))
                    with col2:
                        st.metric("Reward Final", f"{result.get('final_reward', 0):.2f}")
                        st.metric("Utilisation Ressources", f"{result.get('resource_utilization', 0):.2%}")
                    with col3:
                        st.metric("Réduction Makespan", f"{result.get('makespan_reduction', 0):.1f}%")
                    
                    # Ordonnancement
                    schedule = result.get('schedule', [])
                    if schedule:
                        df = pd.DataFrame(schedule)
                        st.dataframe(df, use_container_width=True)
    
    with tab2:
        st.markdown("### Algorithme Génétique")
        
        with st.form("genetic_form"):
            col1, col2 = st.columns(2)
            with col1:
                population = st.slider("Taille Population", 10, 500, 100)
            with col2:
                generations = st.slider("Générations", 10, 500, 100)
            
            if st.form_submit_button("🚀 Optimiser avec AG", use_container_width=True):
                with st.spinner("Évolution en cours..."):
                    result = api_request("/api/optimize/genetic-algorithm", "POST", {
                        'population_size': population,
                        'generations': generations
                    })
                
                if result:
                    st.success("✅ Optimisation Génétique terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Best Fitness", f"{result.get('best_fitness', 0):.2f}")
                    with col2:
                        st.metric("Convergence à Gen", result.get('convergence_generation', 0))
                    with col3:
                        st.metric("Diversité", f"{result.get('diversity_maintained', 0):.2%}")
                    
                    # Courbe de fitness
                    fitness_history = result.get('fitness_history', [])
                    if fitness_history:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=fitness_history,
                            mode='lines+markers',
                            name='Fitness',
                            line=dict(color='#667eea', width=3)
                        ))
                        fig.update_layout(
                            title="Évolution du Fitness",
                            xaxis_title="Génération",
                            yaxis_title="Fitness",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Réseau de Neurones - Prédicteur")
        
        with st.form("neural_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_features = st.number_input("Features d'Entrée", 5, 100, 20)
            with col2:
                hidden_layers = st.slider("Couches Cachées", 1, 10, 3)
            
            if st.form_submit_button("🚀 Créer le Modèle", use_container_width=True):
                with st.spinner("Entraînement du réseau..."):
                    result = api_request("/api/optimize/neural-predictor", "POST", {
                        'input_features': input_features,
                        'hidden_layers': hidden_layers
                    })
                
                if result:
                    st.success("✅ Modèle créé!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Paramètres Totaux", result.get('total_parameters', 0))
                        st.metric("Architecture", result.get('architecture', 'N/A'))
                    with col2:
                        st.metric("Précision", f"{result.get('prediction_accuracy', 0):.2%}")
                        st.metric("Temps Inférence", f"{result.get('inference_time_ms', 0):.2f} ms")
                    with col3:
                        st.metric("Epochs", result.get('training_epochs', 0))
                        st.metric("Loss", f"{result.get('loss', 0):.4f}")
    
    with tab4:
        st.markdown("### Intelligence en Essaim (PSO)")
        
        with st.form("swarm_form"):
            col1, col2 = st.columns(2)
            with col1:
                swarm_size = st.slider("Taille de l'Essaim", 10, 200, 50)
            with col2:
                dimensions = st.slider("Dimensions", 2, 50, 10)
            
            if st.form_submit_button("🚀 Optimiser avec PSO", use_container_width=True):
                with st.spinner("Optimisation par essaim..."):
                    result = api_request("/api/optimize/swarm-intelligence", "POST", {
                        'swarm_size': swarm_size,
                        'dimensions': dimensions
                    })
                
                if result:
                    st.success("✅ Optimisation PSO terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Meilleure Valeur", f"{result.get('best_value', 0):.4f}")
                    with col2:
                        st.metric("Convergence", f"Iter {result.get('convergence_iteration', 0)}")
                    with col3:
                        st.metric("Amélioration", f"+{result.get('improvement_over_random', 0):.1f}%")

# ==================== OPTIMISATIONS QUANTIQUES ====================

elif menu == "⚛️ Optimisations Quantiques":
    st.title("⚛️ Optimisations Quantiques Avancées")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧊 Annealing", "🔄 QAOA", "⚡ VQE", "🔍 Grover", "🤖 Quantum ML"])
    
    with tab1:
        st.markdown("### Quantum Annealing")
        st.info("Optimisation par recuit quantique pour problèmes combinatoires")
        
        with st.form("annealing_form"):
            col1, col2 = st.columns(2)
            with col1:
                problem_size = st.slider("Taille du Problème", 10, 1000, 100)
            with col2:
                constraint_type = st.selectbox("Type de Contraintes",
                    ["linear", "quadratic", "mixed"])
            
            if st.form_submit_button("🚀 Optimiser", use_container_width=True):
                with st.spinner("Recuit quantique en cours..."):
                    result = api_request("/api/optimize/quantum-annealing", "POST", {
                        'problem_size': problem_size,
                        'constraints': {'type': constraint_type}
                    })
                
                if result:
                    st.success("✅ Optimisation terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Énergie Optimale", f"{result.get('optimal_energy', 0):.4f}")
                        st.metric("Qualité Solution", f"{result.get('solution_quality', 0):.2%}")
                    with col2:
                        st.metric("Accélération Quantique", f"{result.get('quantum_speedup', 0):.2f}x")
                        st.metric("Iterations", result.get('iterations', 0))
                    with col3:
                        st.metric("Temps d'Exécution", f"{result.get('execution_time_ms', 0):.2f} ms")
                    
                    # Vecteur solution (premiers éléments)
                    solution = result.get('solution_vector', [])
                    if solution:
                        fig = go.Figure(data=go.Bar(
                            x=list(range(len(solution))),
                            y=solution,
                            marker_color='#667eea'
                        ))
                        fig.update_layout(title="Vecteur Solution (10 premiers éléments)", height=300)
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### QAOA - Quantum Approximate Optimization")
        
        with st.form("qaoa_form"):
            col1, col2 = st.columns(2)
            with col1:
                qubits = st.slider("Nombre de Qubits", 4, 50, 10)
            with col2:
                layers = st.slider("Nombre de Couches", 1, 10, 3)
            
            if st.form_submit_button("🚀 Optimiser avec QAOA", use_container_width=True):
                with st.spinner("Exécution QAOA..."):
                    result = api_request("/api/optimize/qaoa", "POST", {
                        'qubits': qubits,
                        'layers': layers
                    })
                
                if result:
                    st.success("✅ QAOA terminé!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Ratio d'Approximation", f"{result.get('approximation_ratio', 0):.4f}")
                        st.metric("Profondeur Circuit", result.get('circuit_depth', 0))
                    with col2:
                        st.metric("Nombre de Gates", result.get('gate_count', 0))
                        st.metric("Probabilité Succès", f"{result.get('success_probability', 0):.2%}")
                    with col3:
                        st.write("**Paramètres Optimaux:**")
                        params = result.get('optimal_parameters', {})
                        st.write(f"β layers: {len(params.get('beta', []))}")
                        st.write(f"γ layers: {len(params.get('gamma', []))}")
    
    with tab3:
        st.markdown("### VQE - Variational Quantum Eigensolver")
        
        with st.form("vqe_form"):
            col1, col2 = st.columns(2)
            with col1:
                molecules = st.number_input("Nombre de Molécules", 1, 20, 5)
            with col2:
                basis_set = st.selectbox("Ensemble de Base",
                    ["sto-3g", "6-31g", "cc-pvdz", "cc-pvtz"])
            
            if st.form_submit_button("🚀 Calculer avec VQE", use_container_width=True):
                with st.spinner("Calcul VQE en cours..."):
                    result = api_request("/api/optimize/vqe", "POST", {
                        'molecules': molecules,
                        'basis_set': basis_set
                    })
                
                if result:
                    st.success("✅ Calcul VQE terminé!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Énergie État Fondamental", f"{result.get('ground_state_energy', 0):.4f}")
                        st.metric("Iterations Convergence", result.get('convergence_iterations', 0))
                    with col2:
                        st.metric("Précision", f"{result.get('accuracy', 0):.5f}")
                        st.metric("Économies Énergie", f"{result.get('energy_savings_potential', 0):.1f}%")
                    with col3:
                        st.metric("Avantage Quantique", f"{result.get('quantum_advantage', 0):.1f}x")
    
    with tab4:
        st.markdown("### Grover - Recherche Quantique")
        
        with st.form("grover_form"):
            database_size = st.number_input("Taille Base de Données", 100, 10000000, 1000000,
                help="Nombre d'éléments dans la base de données")
            
            if st.form_submit_button("🚀 Recherche Grover", use_container_width=True):
                with st.spinner("Recherche quantique..."):
                    result = api_request("/api/optimize/grover", "POST", {
                        'database_size': database_size
                    })
                
                if result:
                    st.success("✅ Recherche terminée!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Requêtes Classiques", result.get('classical_queries_needed', 0))
                        st.metric("Requêtes Quantiques", result.get('quantum_queries_needed', 0))
                        st.metric("Probabilité Succès", f"{result.get('success_probability', 0):.4f}")
                    with col2:
                        st.metric("Accélération", f"{result.get('speedup', 0):.2f}x")
                        st.metric("Appels Oracle", result.get('oracle_calls', 0))
                        st.metric("Iterations Optimales", result.get('optimal_iterations', 0))
                    
                    # Comparaison visuelle
                    fig = go.Figure(data=[
                        go.Bar(name='Classique', x=['Requêtes'], y=[result.get('classical_queries_needed', 0)],
                               marker_color='#f5576c'),
                        go.Bar(name='Quantique', x=['Requêtes'], y=[result.get('quantum_queries_needed', 0)],
                               marker_color='#667eea')
                    ])
                    fig.update_layout(title="Comparaison Classique vs Quantique", height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### Quantum Machine Learning")
        
        with st.form("qml_form"):
            col1, col2 = st.columns(2)
            with col1:
                dataset_size = st.number_input("Taille Dataset", 100, 100000, 10000)
            with col2:
                features = st.number_input("Nombre de Features", 5, 200, 50)
            
            if st.form_submit_button("🚀 Optimiser avec QML", use_container_width=True):
                with st.spinner("Entraînement quantique..."):
                    result = api_request("/api/optimize/quantum-ml", "POST", {
                        'dataset_size': dataset_size,
                        'features': features
                    })
                
                if result:
                    st.success("✅ Entraînement QML terminé!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accélération Training", f"{result.get('training_speedup', 0):.2f}x")
                        st.metric("Précision Modèle", f"{result.get('model_accuracy', 0):.2%}")
                    with col2:
                        st.metric("Qualité Optimisation", f"{result.get('parameter_optimization_quality', 0):.2%}")
                        st.metric("Avantage Kernel", f"{result.get('quantum_kernel_advantage', 0):.2f}x")
                    with col3:
                        st.metric("Dimensionalité Feature Space", result.get('feature_space_dimensionality', 0))

# ==================== ORDONNANCEMENT ====================

elif menu == "⚙️ Ordonnancement":
    st.title("⚙️ Ordonnancement Intelligent de Tâches")
    
    tab1, tab2 = st.tabs(["➕ Créer Scheduler", "📋 Ordonnancer"])
    
    with tab1:
        st.markdown("### Créer un Ordonnanceur")
        
        with st.form("scheduler_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Nom", placeholder="Mon Ordonnanceur")
                algorithm = st.selectbox("Algorithme",
                    ["round_robin", "priority", "fair_share", "shortest_job_first", "earliest_deadline_first"])
                priority_levels = st.slider("Niveaux de Priorité", 3, 10, 5)
            with col2:
                ai_enhanced = st.checkbox("✨ IA Enhanced", value=True)
                quantum_enhanced = st.checkbox("⚛️ Quantum Enhanced", value=False)
            
            if st.form_submit_button("🚀 Créer l'Ordonnanceur", use_container_width=True):
                result = api_request("/api/scheduler/create", "POST", {
                    'name': name,
                    'algorithm': algorithm,
                    'ai_enhanced': ai_enhanced,
                    'quantum_enhanced': quantum_enhanced,
                    'priority_levels': priority_levels
                })
                
                if result:
                    st.success("✅ Ordonnanceur créé!")
                    st.json(result)
    
    with tab2:
        st.markdown("### Ordonnancer des Tâches")
        
        # Créer des tâches exemple
        st.markdown("#### Configuration des Tâches")
        num_tasks = st.number_input("Nombre de tâches", 1, 100, 10)
        
        tasks = []
        with st.expander("⚙️ Configuration Détaillée des Tâches"):
            for i in range(min(num_tasks, 10)):
                col1, col2, col3 = st.columns(3)
                with col1:
                    task_id = st.text_input(f"ID Tâche {i+1}", f"task_{i}", key=f"id_{i}")
                with col2:
                    duration = st.number_input(f"Durée {i+1}", 1, 100, 10, key=f"dur_{i}")
                with col3:
                    priority = st.slider(f"Priorité {i+1}", 1, 10, 5, key=f"pri_{i}")
                
                tasks.append({'id': task_id, 'duration': duration, 'priority': priority})
        
        # Auto-générer les tâches restantes
        for i in range(len(tasks), num_tasks):
            tasks.append({
                'id': f'task_{i}',
                'duration': np.random.randint(5, 30),
                'priority': np.random.randint(1, 10)
            })
        
        if st.button("🚀 Ordonnancer les Tâches", use_container_width=True):
            # Créer un scheduler temporaire
            scheduler_result = api_request("/api/scheduler/create", "POST", {
                'name': 'temp_scheduler',
                'algorithm': 'priority',
                'ai_enhanced': True
            })
            
            if scheduler_result:
                scheduler_id = scheduler_result['scheduler_id']
                
                with st.spinner("Ordonnancement en cours..."):
                    result = api_request(f"/api/scheduler/{scheduler_id}/schedule", "POST", {
                        'tasks': tasks
                    })
                
                if result:
                    st.success("✅ Ordonnancement terminé!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nombre de Tâches", result.get('num_tasks', 0))
                    with col2:
                        st.metric("Makespan Total", f"{result.get('total_makespan', 0):.2f}s")
                    with col3:
                        st.metric("Temps d'Attente Moyen", f"{result.get('average_wait_time', 0):.2f}s")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Utilisation Ressources", f"{result.get('resource_utilization', 0):.2%}")
                    with col2:
                        st.metric("Efficacité Ordonnancement", f"{result.get('scheduling_efficiency', 0):.2%}")
                    
                    # Diagramme de Gantt
                    scheduled = result.get('scheduled_tasks', [])
                    if scheduled:
                        df = pd.DataFrame(scheduled)
                        
                        fig = go.Figure()
                        for idx, task in df.iterrows():
                            fig.add_trace(go.Bar(
                                name=task['task_id'],
                                x=[task['duration']],
                                y=[task['task_id']],
                                orientation='h',
                                base=task['start_time'],
                                marker=dict(color=f'rgb({np.random.randint(100, 255)}, {np.random.randint(100, 255)}, {np.random.randint(100, 255)})')
                            ))
                        
                        fig.update_layout(
                            title="Diagramme de Gantt - Ordonnancement",
                            xaxis_title="Temps (s)",
                            yaxis_title="Tâches",
                            height=400,
                            showlegend=False,
                            barmode='overlay'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tableau détaillé
                        st.markdown("### 📋 Détails de l'Ordonnancement")
                        st.dataframe(df, use_container_width=True)

# ==================== PROFILAGE ====================

elif menu == "📈 Profilage":
    st.title("📈 Profilage de Performance")
    
    tab1, tab2 = st.tabs(["🔍 Créer Profil", "📊 Analyser"])
    
    with tab1:
        st.markdown("### Créer un Profil de Performance")
        
        systems = api_request("/api/system/list", show_error=False) or []
        
        if systems:
            system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
            selected = st.selectbox("Sélectionnez un système",
                options=list(system_names.keys()),
                format_func=lambda x: system_names[x])
            
            duration = st.slider("Durée du profilage (secondes)", 10, 300, 60)
            
            if st.button("🚀 Créer le Profil", use_container_width=True, type="primary"):
                with st.spinner("Profilage en cours..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(duration / 100)
                        progress.progress(i + 1)
                    
                    result = api_request("/api/profile/create", "POST", {
                        'system_id': selected,
                        'duration_seconds': duration
                    })
                
                if result:
                    st.success("✅ Profil créé!")
                    st.session_state.current_profile = result['profile_id']
                    
                    # Affichage des profils
                    st.markdown("### 📊 Résultats du Profilage")
                    
                    cpu = result.get('cpu_profile', {})
                    memory = result.get('memory_profile', {})
                    io = result.get('io_profile', {})
                    energy = result.get('energy_profile', {})
                    
                    # CPU Profile
                    st.markdown("#### 💻 Profil CPU")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Usage Moyen", f"{cpu.get('average_usage', 0):.1f}%")
                    with col2:
                        st.metric("Pic d'Usage", f"{cpu.get('peak_usage', 0):.1f}%")
                    with col3:
                        st.metric("Temps Idle", f"{cpu.get('idle_time_percentage', 0):.1f}%")
                    with col4:
                        st.metric("Cache Miss Rate", f"{cpu.get('cache_miss_rate', 0):.2%}")
                    
                    # Memory Profile
                    st.markdown("#### 💾 Profil Mémoire")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Usage Moyen", f"{memory.get('average_usage', 0):.1f}%")
                    with col2:
                        st.metric("Pic d'Usage", f"{memory.get('peak_usage', 0):.1f}%")
                    with col3:
                        st.metric("Bandwidth Utilisé", f"{memory.get('memory_bandwidth_utilized', 0):.2%}")
                    
                    # I/O Profile
                    st.markdown("#### 💿 Profil I/O")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Lectures/s", f"{io.get('read_operations_per_sec', 0):.0f}")
                    with col2:
                        st.metric("Écritures/s", f"{io.get('write_operations_per_sec', 0):.0f}")
                    with col3:
                        st.metric("Latence Moy.", f"{io.get('average_latency_ms', 0):.2f} ms")
                    
                    # Energy Profile
                    st.markdown("#### ⚡ Profil Énergétique")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Puissance Moy.", f"{energy.get('average_power_w', 0):.1f} W")
                    with col2:
                        st.metric("Pic de Puissance", f"{energy.get('peak_power_w', 0):.1f} W")
                    with col3:
                        st.metric("Score Efficacité", f"{energy.get('energy_efficiency_score', 0):.2f}")
                    
                    # Goulots d'étranglement
                    st.markdown("#### 🚨 Goulots d'Étranglement Détectés")
                    bottlenecks = result.get('bottlenecks_detected', [])
                    if bottlenecks:
                        for bn in bottlenecks:
                            severity_color = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                            st.warning(f"{severity_color.get(bn['severity'], '⚪')} **{bn['component'].upper()}** - "
                                     f"Sévérité: {bn['severity']} - Impact: {bn['impact']:.1f}%")
                    
                    # Recommandations
                    st.markdown("#### 💡 Recommandations d'Optimisation")
                    recommendations = result.get('optimization_recommendations', [])
                    for rec in recommendations:
                        st.info(f"✓ {rec}")
        else:
            st.warning("⚠️ Créez d'abord un système")
    
    with tab2:
        st.markdown("### Analyser un Profil")
        
        if 'current_profile' in st.session_state:
            if st.button("🔍 Analyser le Profil Actuel", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    result = api_request(f"/api/profile/{st.session_state.current_profile}/analyze", "POST")
                
                if result:
                    st.success("✅ Analyse terminée!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Score Santé Global", f"{result.get('overall_health_score', 0):.1f}/100")
                    with col2:
                        st.metric("Note Performance", result.get('performance_rating', 'N/A'))
                    
                    # Potentiel d'optimisation
                    st.markdown("### 🎯 Potentiel d'Optimisation")
                    potential = result.get('optimization_potential', {})
                    
                    categories = list(potential.keys())
                    values = list(potential.values())
                    
                    fig = go.Figure(data=[go.Bar(
                        x=categories,
                        y=values,
                        marker_color=['#667eea', '#38ef7d', '#f093fb', '#f5576c']
                    )])
                    fig.update_layout(
                        title="Potentiel d'Amélioration par Composant (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Améliorations prédites
                    st.markdown("### 🚀 Améliorations Prédites")
                    predicted = result.get('predicted_improvements', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🤖 IA", f"+{predicted.get('with_ai_optimization', 0):.1f}%")
                    with col2:
                        st.metric("⚛️ Quantum", f"+{predicted.get('with_quantum_optimization', 0):.1f}%")
                    with col3:
                        st.metric("🔄 Hybrid", f"+{predicted.get('with_hybrid_optimization', 0):.1f}%")
                    
                    # Analyse coût-bénéfice
                    st.markdown("### 💰 Analyse Coût-Bénéfice")
                    cost_benefit = result.get('cost_benefit_analysis', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Coût:** {cost_benefit.get('implementation_cost', 'N/A')}")
                    with col2:
                        st.metric("ROI Attendu", f"{cost_benefit.get('expected_roi', 0):.0f}%")
                    with col3:
                        st.metric("Période de Retour", f"{cost_benefit.get('payback_period_months', 0)} mois")
        else:
            st.info("📭 Créez d'abord un profil dans l'onglet 'Créer Profil'")

# ==================== LOAD BALANCING ====================

elif menu == "⚖️ Load Balancing":
    st.title("⚖️ Équilibrage de Charge Intelligent")
    
    tab1, tab2 = st.tabs(["➕ Créer LB", "📊 Distribuer"])
    
    with tab1:
        st.markdown("### Créer un Load Balancer")
        
        with st.form("lb_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Nom", placeholder="Mon Load Balancer")
                algorithm = st.selectbox("Algorithme",
                    ["round_robin", "weighted_round_robin", "least_connections", 
                     "ip_hash", "random", "ai_predictive"])
            with col2:
                ai_enabled = st.checkbox("✨ IA Enabled", value=True)
                quantum_enabled = st.checkbox("⚛️ Quantum Enabled", value=False)
                health_check = st.number_input("Intervalle Health Check (s)", 10, 300, 30)
            
            if st.form_submit_button("🚀 Créer le Load Balancer", use_container_width=True):
                result = api_request("/api/loadbalancer/create", "POST", {
                    'name': name,
                    'algorithm': algorithm,
                    'ai_enabled': ai_enabled,
                    'quantum_enabled': quantum_enabled,
                    'health_check_interval': health_check
                })
                
                if result:
                    st.success("✅ Load Balancer créé!")
                    st.json(result)
                    st.session_state.current_lb = result['lb_id']
    
    with tab2:
        st.markdown("### Distribuer la Charge")
        
        if 'current_lb' in st.session_state or True:
            # Créer un LB temporaire si nécessaire
            lb_id = st.session_state.get('current_lb')
            if not lb_id:
                temp_lb = api_request("/api/loadbalancer/create", "POST", {
                    'name': 'temp_lb',
                    'algorithm': 'weighted_round_robin',
                    'ai_enabled': True
                }, show_error=False)
                if temp_lb:
                    lb_id = temp_lb['lb_id']
            
            if lb_id:
                requests_num = st.number_input("Nombre de Requêtes", 100, 1000000, 10000)
                
                if st.button("🚀 Distribuer la Charge", use_container_width=True, type="primary"):
                    with st.spinner("Distribution en cours..."):
                        result = api_request(f"/api/loadbalancer/{lb_id}/distribute", "POST", {
                            'requests': requests_num
                        })
                    
                    if result:
                        st.success("✅ Charge distribuée!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Requêtes", result.get('total_requests', 0))
                        with col2:
                            st.metric("Nombre de Nœuds", result.get('num_nodes', 0))
                        with col3:
                            st.metric("Score Balance", f"{result.get('balance_score', 0):.2%}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Temps Réponse Moy.", f"{result.get('overall_response_time_ms', 0):.2f} ms")
                        with col2:
                            st.metric("Throughput", f"{result.get('throughput_requests_per_sec', 0):.0f} req/s")
                        
                        # Distribution par nœud
                        distribution = result.get('distribution', [])
                        if distribution:
                            df = pd.DataFrame(distribution)
                            
                            # Graphique de distribution
                            fig = go.Figure(data=[
                                go.Bar(name='Requêtes Allouées',
                                      x=df['node_id'],
                                      y=df['requests_allocated'],
                                      marker_color='#667eea'),
                                go.Bar(name='Utilisation (%)',
                                      x=df['node_id'],
                                      y=df['utilization']*1000,
                                      marker_color='#38ef7d')
                            ])
                            fig.update_layout(
                                title="Distribution de la Charge par Nœud",
                                barmode='group',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Tableau détaillé
                            st.markdown("### 📋 Détails par Nœud")
                            st.dataframe(df, use_container_width=True)

# ==================== CACHE INTELLIGENT ====================

elif menu == "💾 Cache Intelligent":
    st.title("💾 Système de Cache Intelligent")
    
    tab1, tab2 = st.tabs(["➕ Créer Cache", "🚀 Optimiser"])
    
    with tab1:
        st.markdown("### Créer un Cache Intelligent")
        
        with st.form("cache_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Nom", placeholder="Mon Cache")
                size_gb = st.number_input("Taille (GB)", 1, 512, 32)
                levels = st.slider("Niveaux de Cache", 1, 5, 3)
            with col2:
                eviction = st.selectbox("Politique d'Éviction",
                    ["lru", "lfu", "fifo", "random", "ai_predictive", "quantum_optimized"])
                ai_enabled = st.checkbox("✨ IA Enabled", value=True)
                quantum_enabled = st.checkbox("⚛️ Quantum Enabled", value=False)
            
            if st.form_submit_button("🚀 Créer le Cache", use_container_width=True):
                result = api_request("/api/cache/create", "POST", {
                    'name': name,
                    'size_gb': size_gb,
                    'eviction_policy': eviction,
                    'levels': levels,
                    'ai_enabled': ai_enabled,
                    'quantum_enabled': quantum_enabled
                })
                
                if result:
                    st.success("✅ Cache créé!")
                    st.json(result)
                    st.session_state.current_cache = result['cache_id']
    
    with tab2:
        st.markdown("### Optimiser le Cache")
        
        if 'current_cache' in st.session_state or True:
            cache_id = st.session_state.get('current_cache')
            if not cache_id:
                # Créer un cache temporaire
                temp_cache = api_request("/api/cache/create", "POST", {
                    'name': 'temp_cache',
                    'size_gb': 32,
                    'eviction_policy': 'ai_predictive',
                    'ai_enabled': True
                }, show_error=False)
                if temp_cache:
                    cache_id = temp_cache['cache_id']
            
            if cache_id:
                if st.button("🚀 Optimiser le Cache", use_container_width=True, type="primary"):
                    with st.spinner("Optimisation en cours..."):
                        result = api_request(f"/api/cache/{cache_id}/optimize", "POST")
                    
                    if result:
                        st.success("✅ Cache optimisé!")
                        
                        # Avant/Après
                        st.markdown("### 📊 Résultats de l'Optimisation")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Avant Optimisation")
                            before = result.get('before_optimization', {})
                            st.metric("Hit Rate", f"{before.get('hit_rate', 0):.2%}")
                            st.metric("Miss Rate", f"{before.get('miss_rate', 0):.2%}")
                            st.metric("Latence Moy.", f"{before.get('average_latency_us', 0):.1f} µs")
                        
                        with col2:
                            st.markdown("#### Après Optimisation")
                            after = result.get('after_optimization', {})
                            st.metric("Hit Rate", f"{after.get('hit_rate', 0):.2%}",
                                    delta=f"+{(after.get('hit_rate', 0) - before.get('hit_rate', 0))*100:.1f}%")
                            st.metric("Miss Rate", f"{after.get('miss_rate', 0):.2%}",
                                    delta=f"-{(before.get('miss_rate', 0) - after.get('miss_rate', 0))*100:.1f}%")
                            st.metric("Latence Moy.", f"{after.get('average_latency_us', 0):.1f} µs",
                                    delta=f"-{before.get('average_latency_us', 0) - after.get('average_latency_us', 0):.1f} µs")
                        
                        # Graphique de comparaison
                        fig = go.Figure(data=[
                            go.Bar(name='Avant', x=['Hit Rate', 'Latence'],
                                  y=[before.get('hit_rate', 0)*100, before.get('average_latency_us', 0)],
                                  marker_color='#f5576c'),
                            go.Bar(name='Après', x=['Hit Rate', 'Latence'],
                                  y=[after.get('hit_rate', 0)*100, after.get('average_latency_us', 0)],
                                  marker_color='#38ef7d')
                        ])
                        fig.update_layout(title="Comparaison Avant/Après", barmode='group', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Améliorations
                        st.markdown("### 🎯 Améliorations")
                        improvements = result.get('improvements', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("↑ Hit Rate", f"+{improvements.get('hit_rate_improvement', 0):.1f}%")
                        with col2:
                            st.metric("↓ Latence", f"-{improvements.get('latency_reduction', 0):.1f}%")
                        with col3:
                            st.metric("↑ Throughput", f"+{improvements.get('throughput_increase', 0):.1f}%")
                        
                        # Techniques appliquées
                        st.markdown("### 🔧 Techniques Appliquées")
                        techniques = result.get('optimization_techniques_applied', [])
                        for tech in techniques:
                            st.success(f"✓ {tech}")

# ==================== COMPRESSION ====================

elif menu == "🗜️ Compression":
    st.title("🗜️ Compression et Déduplication")
    
    tab1, tab2 = st.tabs(["🔍 Analyser", "🚀 Appliquer"])
    
    with tab1:
        st.markdown("### Analyser le Potentiel de Compression")
        
        with st.form("compression_analysis"):
            col1, col2 = st.columns(2)
            with col1:
                data_size = st.number_input("Taille des Données (GB)", 1, 10000, 100)
            with col2:
                data_type = st.selectbox("Type de Données",
                    ["text", "binary", "media", "mixed", "database", "logs"])
            
            if st.form_submit_button("🔍 Analyser", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    result = api_request("/api/compression/analyze", "POST", {
                        'data_size_gb': data_size,
                        'data_type': data_type
                    })
                
                if result:
                    st.success("✅ Analyse terminée!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Taille Originale", f"{result.get('data_size_gb', 0):.1f} GB")
                        st.metric("Type de Données", result.get('data_type', 'N/A'))
                    with col2:
                        st.metric("Économies d'Espace", f"{result.get('space_savings_gb', 0):.1f} GB")
                        st.metric("Pourcentage", f"{result.get('space_savings_percentage', 0):.1f}%")
                    
                    # Comparaison des algorithmes
                    st.markdown("### 📊 Comparaison des Algorithmes")
                    algorithms = result.get('compression_algorithms_tested', {})
                    
                    algo_names = list(algorithms.keys())
                    ratios = [algo['ratio'] for algo in algorithms.values()]
                    speeds = [algo['speed_mbps'] for algo in algorithms.values()]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure(data=[go.Bar(
                            x=algo_names,
                            y=ratios,
                            marker_color=['#667eea', '#38ef7d', '#f093fb', '#f5576c']
                        )])
                        fig.update_layout(title="Ratio de Compression", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure(data=[go.Bar(
                            x=algo_names,
                            y=speeds,
                            marker_color=['#667eea', '#38ef7d', '#f093fb', '#f5576c']
                        )])
                        fig.update_layout(title="Vitesse (MB/s)", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommandation
                    st.success(f"💡 **Algorithme Recommandé:** {result.get('recommended_algorithm', 'N/A')}")
                    st.info(f"🔄 **Potentiel de Déduplication:** {result.get('deduplication_potential', 0):.1f}%")
    
    with tab2:
        st.markdown("### Appliquer la Compression")
        
        with st.form("compression_apply"):
            col1, col2 = st.columns(2)
            with col1:
                data_size = st.number_input("Taille des Données (GB)", 1, 10000, 100)
                algorithm = st.selectbox("Algorithme",
                    ["gzip", "lz4", "zstd", "bzip2", "quantum_compression"])
            
            if st.form_submit_button("🚀 Appliquer la Compression", use_container_width=True):
                with st.spinner("Compression en cours..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.05)
                        progress.progress(i + 1)
                    
                    result = api_request("/api/compression/apply", "POST", {
                        'algorithm': algorithm,
                        'data_size_gb': data_size
                    })
                
                if result:
                    st.success("✅ Compression appliquée!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Taille Originale", f"{result.get('original_size_gb', 0):.2f} GB")
                        st.metric("Taille Compressée", f"{result.get('compressed_size_gb', 0):.2f} GB")
                    with col2:
                        st.metric("Ratio Compression", f"{result.get('compression_ratio', 0):.2f}x")
                        st.metric("Espace Économisé", f"{result.get('space_saved_gb', 0):.2f} GB")
                    with col3:
                        st.metric("Temps Compression", f"{result.get('compression_time_seconds', 0):.2f}s")
                        st.metric("Throughput", f"{result.get('throughput_mbps', 0):.1f} MB/s")
                    
                    # Graphique circulaire
                    fig = go.Figure(data=[go.Pie(
                        labels=['Compressé', 'Économisé'],
                        values=[result.get('compressed_size_gb', 0), result.get('space_saved_gb', 0)],
                        hole=.3,
                        marker=dict(colors=['#667eea', '#38ef7d'])
                    )])
                    fig.update_layout(title="Distribution de l'Espace", height=400)
                    st.plotly_chart(fig, use_container_width=True)

# ==================== ÉNERGIE ====================

elif menu == "⚡ Énergie":
    st.title("⚡ Optimisation Énergétique")
    
    tab1, tab2 = st.tabs(["🔍 Analyser", "🚀 Optimiser"])
    
    with tab1:
        st.markdown("### Analyser la Consommation Énergétique")
        
        systems = api_request("/api/system/list", show_error=False) or []
        
        if systems:
            system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
            selected = st.selectbox("Sélectionnez un système",
                options=list(system_names.keys()),
                format_func=lambda x: system_names[x])
            
            if st.button("🔍 Analyser la Consommation", use_container_width=True):
                with st.spinner("Analyse énergétique..."):
                    result = api_request("/api/energy/analyze", "POST", {
                        'system_id': selected
                    })
                
                if result:
                    st.success("✅ Analyse terminée!")
                    
                    # Consommation actuelle
                    st.markdown("### 📊 Consommation Actuelle")
                    current = result.get('current_consumption', {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Puissance Moy.", f"{current.get('average_power_w', 0):.1f} W")
                    with col2:
                        st.metric("Pic Puissance", f"{current.get('peak_power_w', 0):.1f} W")
                    with col3:
                        st.metric("Puissance Idle", f"{current.get('idle_power_w', 0):.1f} W")
                    with col4:
                        st.metric("Énergie Quotidienne", f"{current.get('daily_energy_kwh', 0):.2f} kWh")
                    
                    # Métriques d'efficacité
                    st.markdown("### 🎯 Métriques d'Efficacité")
                    efficiency = result.get('efficiency_metrics', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("PUE", f"{efficiency.get('pue', 0):.2f}")
                    with col2:
                        st.metric("Énergie par Opération", f"{efficiency.get('energy_per_operation', 0):.4f} J")
                    with col3:
                        st.metric("Empreinte Carbone", f"{efficiency.get('carbon_footprint_kg_co2_per_day', 0):.2f} kg CO₂/jour")
                    
                    # Potentiel d'optimisation
                    st.markdown("### 🚀 Potentiel d'Optimisation")
                    potential = result.get('optimization_potential', {})
                    
                    categories = list(potential.keys())
                    values = list(potential.values())
                    
                    fig = go.Figure(data=[go.Bar(
                        x=categories,
                        y=values,
                        marker_color=['#667eea', '#38ef7d', '#f093fb', '#f5576c']
                    )])
                    fig.update_layout(
                        title="Économies d'Énergie Potentielles (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommandations
                    st.markdown("### 💡 Recommandations")
                    recommendations = result.get('recommendations', [])
                    for rec in recommendations:
                        st.info(f"✓ {rec}")
                    
                    st.metric("💰 Économies Annuelles Estimées", 
                             f"${result.get('estimated_annual_savings_usd', 0):,.0f}")
        else:
            st.warning("⚠️ Créez d'abord un système")
    
    with tab2:
        st.markdown("### Optimiser la Consommation")
        
        systems = api_request("/api/system/list", show_error=False) or []
        
        if systems:
            system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
            selected = st.selectbox("Système à optimiser",
                options=list(system_names.keys()),
                format_func=lambda x: system_names[x])
            
            optimization_level = st.select_slider(
                "Niveau d'Optimisation",
                options=['conservative', 'balanced', 'aggressive'],
                value='balanced',
                help="Conservative: Faible impact sur les performances | Balanced: Équilibre | Aggressive: Économies maximales"
            )
            
            if st.button("🚀 Optimiser l'Énergie", use_container_width=True, type="primary"):
                with st.spinner("Optimisation en cours..."):
                    result = api_request("/api/energy/optimize", "POST", {
                        'system_id': selected,
                        'level': optimization_level
                    })
                
                if result:
                    st.success("✅ Optimisation appliquée!")
                    
                    # Avant/Après
                    st.markdown("### 📊 Résultats")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ⬅️ Avant")
                        before = result.get('before_optimization', {})
                        st.metric("Puissance Moy.", f"{before.get('average_power_w', 0):.1f} W")
                        st.metric("Énergie/Jour", f"{before.get('daily_energy_kwh', 0):.2f} kWh")
                        st.metric("PUE", f"{before.get('pue', 0):.2f}")
                    
                    with col2:
                        st.markdown("#### ➡️ Après")
                        after = result.get('after_optimization', {})
                        st.metric("Puissance Moy.", f"{after.get('average_power_w', 0):.1f} W",
                                delta=f"-{before.get('average_power_w', 0) - after.get('average_power_w', 0):.1f} W")
                        st.metric("Énergie/Jour", f"{after.get('daily_energy_kwh', 0):.2f} kWh",
                                delta=f"-{before.get('daily_energy_kwh', 0) - after.get('daily_energy_kwh', 0):.2f} kWh")
                        st.metric("PUE", f"{after.get('pue', 0):.2f}",
                                delta=f"-{before.get('pue', 0) - after.get('pue', 0):.2f}")
                    
                    # Métriques globales
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💚 Économies d'Énergie", 
                                f"{result.get('energy_savings_percentage', 0):.1f}%")
                    with col2:
                        st.metric("⚠️ Impact Performance", 
                                f"{result.get('performance_impact_percentage', 0):.1f}%")
                    
                    # Techniques appliquées
                    st.markdown("### 🔧 Techniques Appliquées")
                    techniques = result.get('techniques_applied', [])
                    for tech in techniques:
                        st.success(f"✓ {tech}")
        else:
            st.warning("⚠️ Créez d'abord un système")

# ==================== MAINTENANCE PRÉDICTIVE ====================

elif menu == "🔮 Maintenance Prédictive":
    st.title("🔮 Maintenance Prédictive par IA")
    
    systems = api_request("/api/system/list", show_error=False) or []
    
    if systems:
        system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
        selected = st.selectbox("Sélectionnez un système",
            options=list(system_names.keys()),
            format_func=lambda x: system_names[x])
        
        if st.button("🔮 Prédire les Besoins", use_container_width=True, type="primary"):
            with st.spinner("Analyse prédictive en cours..."):
                result = api_request("/api/maintenance/predict", "POST", {
                    'system_id': selected
                })
            
            if result:
                st.success("✅ Prédiction terminée!")
                
                # Score de santé
                health_score = result.get('health_score', 0)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=health_score,
                        title={'text': "Score de Santé Global", 'font': {'size': 24}},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#38ef7d" if health_score > 80 else "#f5576c"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "gray"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Santé des composants
                st.markdown("### 🔧 Santé des Composants")
                components = result.get('component_health', {})
                
                for comp_name, comp_data in components.items():
                    with st.expander(f"💻 {comp_name.upper()}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            health_pct = comp_data.get('health_percentage', 0)
                            st.metric("Santé", f"{health_pct:.1f}%")
                        with col2:
                            st.metric("Défaillance dans", f"{comp_data.get('predicted_failure_days', 0)} jours")
                        with col3:
                            st.info(f"📋 {comp_data.get('recommendation', 'N/A')}")
                
                # Anomalies détectées
                st.markdown("### 🚨 Anomalies Détectées")
                anomalies = result.get('anomalies_detected', [])
                if anomalies:
                    for anom in anomalies:
                        severity_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                        st.warning(f"{severity_emoji.get(anom['severity'], '⚪')} **{anom['component'].upper()}** - "
                                 f"Sévérité: {anom['severity']} - {anom['description']}")
                else:
                    st.success("✅ Aucune anomalie détectée")
                
                # Planning de maintenance
                st.markdown("### 📅 Planning de Maintenance Recommandé")
                schedule = result.get('maintenance_schedule', [])
                if schedule:
                    df_schedule = pd.DataFrame(schedule)
                    
                    # Tri par priorité
                    priority_order = {'high': 0, 'medium': 1, 'low': 2}
                    df_schedule['priority_num'] = df_schedule['priority'].map(priority_order)
                    df_schedule = df_schedule.sort_values('priority_num')
                    
                    for _, task in df_schedule.iterrows():
                        priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                        st.info(f"{priority_color.get(task['priority'], '⚪')} **{task['task']}** - "
                               f"Priorité: {task['priority']} - Dans {task['due_in_days']} jours")
                
                st.metric("🎯 Confiance IA", f"{result.get('ai_confidence', 0):.1%}")
    else:
        st.warning("⚠️ Créez d'abord un système")

# ==================== AUTO-TUNING ====================

elif menu == "🎛️ Auto-Tuning":
    st.title("🎛️ Auto-Tuning Intelligent")
    
    tab1, tab2 = st.tabs(["🚀 Activer", "📊 Résultats"])
    
    with tab1:
        st.markdown("### Activer l'Auto-Tuning")
        
        systems = api_request("/api/system/list", show_error=False) or []
        
        if systems:
            system_names = {s['system_id']: f"{s['name']} ({s['type']})" for s in systems}
            selected = st.selectbox("Sélectionnez un système",
                options=list(system_names.keys()),
                format_func=lambda x: system_names[x])
            
            with st.form("autotune_form"):
                col1, col2 = st.columns(2)
                with col1:
                    targets = st.multiselect("Cibles d'Optimisation",
                        ['cpu', 'memory', 'io', 'network', 'energy'],
                        default=['cpu', 'memory', 'io'])
                    aggressiveness = st.select_slider("Agressivité",
                        options=['conservative', 'balanced', 'aggressive'],
                        value='balanced')
                with col2:
                    learning_rate = st.slider("Taux d'Apprentissage", 0.001, 0.1, 0.01, step=0.001,
                        format="%.3f")
                    interval = st.number_input("Intervalle d'Adaptation (s)", 30, 600, 60)
                
                if st.form_submit_button("🚀 Activer l'Auto-Tuning", use_container_width=True):
                    result = api_request("/api/autotune/enable", "POST", {
                        'system_id': selected,
                        'targets': targets,
                        'aggressiveness': aggressiveness,
                        'learning_rate': learning_rate,
                        'interval': interval
                    })
                    
                    if result:
                        st.success("✅ Auto-Tuning activé!")
                        st.session_state.current_autotune = result['autotune_id']
                        st.json(result)
        else:
            st.warning("⚠️ Créez d'abord un système")
    
    with tab2:
        st.markdown("### Résultats de l'Auto-Tuning")
        
        if 'current_autotune' in st.session_state:
            if st.button("📊 Obtenir les Résultats", use_container_width=True):
                with st.spinner("Récupération des résultats..."):
                    result = api_request(f"/api/autotune/{st.session_state.current_autotune}/results", "POST")
                
                if result:
                    st.success("✅ Résultats obtenus!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Durée Exécution", f"{result.get('runtime_hours', 0):.1f}h")
                    with col2:
                        st.metric("Ajustements", result.get('adjustments_made', 0))
                    with col3:
                        st.metric("Score Stabilité", f"{result.get('stability_score', 0):.2%}")
                    
                    # Améliorations
                    st.markdown("### 🎯 Améliorations de Performance")
                    improvements = result.get('performance_improvements', {})
                    
                    categories = list(improvements.keys())
                    values = list(improvements.values())
                    
                    fig = go.Figure(data=[go.Bar(
                        x=categories,
                        y=values,
                        marker_color=['#667eea', '#38ef7d', '#f093fb', '#f5576c']
                    )])
                    fig.update_layout(
                        title="Améliorations par Catégorie (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Paramètres ajustés
                    st.markdown("### 🔧 Paramètres Ajustés")
                    params = result.get('parameters_tuned', [])
                    if params:
                        for param in params:
                            col1, col2, col3 = st.columns([2, 2, 2])
                            with col1:
                                st.write(f"**{param['parameter']}**")
                            with col2:
                                st.write(f"Ancien: `{param['old_value']}`")
                            with col3:
                                st.write(f"Nouveau: `{param['new_value']}`")
                    
                    st.info(f"💡 Recommandation: {result.get('recommendation', 'N/A')}")
        else:
            st.info("📭 Activez d'abord l'auto-tuning dans l'onglet 'Activer'")

# ==================== RAPPORTS ====================

elif menu == "📑 Rapports":
    st.title("📑 Rapports et Analytics")
    
    if st.button("📊 Générer Rapport Complet", use_container_width=True, type="primary"):
        with st.spinner("Génération du rapport..."):
            result = api_request("/api/report/comprehensive")
        
        if result:
            st.success("✅ Rapport généré!")
            
            # Executive Summary
            st.markdown("## 📋 Résumé Exécutif")
            summary = result.get('executive_summary', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Systèmes Gérés", summary.get('total_systems_managed', 0))
            with col2:
                st.metric("Stratégies Déployées", summary.get('total_strategies_deployed', 0))
            with col3:
                st.metric("Amélioration Moy.", f"{summary.get('average_performance_improvement', 0):.1f}%")
            with col4:
                st.metric("Économies Énergie", f"{summary.get('average_energy_savings', 0):.1f}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avantage Quantique", f"{summary.get('quantum_advantage_realized', 0):.1f}x")
            with col2:
                st.metric("Gain Efficacité Global", f"{summary.get('overall_efficiency_gain', 0):.1f}%")
            
            # Accomplissements clés
            st.markdown("## 🏆 Accomplissements Clés")
            achievements = result.get('key_achievements', [])
            for ach in achievements:
                st.success(f"✓ {ach}")
            
            # Vue d'ensemble de la santé
            st.markdown("## 💚 Vue d'Ensemble de la Santé des Systèmes")
            health = result.get('system_health_overview', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Systèmes Sains", health.get('healthy_systems', 0))
            with col2:
                st.metric("Nécessitent Attention", health.get('systems_needing_attention', 0))
            with col3:
                st.metric("Problèmes Critiques", health.get('critical_issues', 0))
            with col4:
                st.metric("Score Santé Moy.", f"{health.get('average_health_score', 0):.1f}/100")
            
            # Graphique de santé
            fig = go.Figure(data=[go.Pie(
                labels=['Sains', 'Attention Requise', 'Critiques'],
                values=[
                    health.get('healthy_systems', 0),
                    health.get('systems_needing_attention', 0),
                    health.get('critical_issues', 0)
                ],
                marker=dict(colors=['#38ef7d', '#f5a623', '#f5576c'])
            )])
            fig.update_layout(title="Distribution de la Santé des Systèmes", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations
            st.markdown("## 💡 Recommandations Stratégiques")
            recommendations = result.get('recommendations', [])
            for idx, rec in enumerate(recommendations, 1):
                st.info(f"{idx}. {rec}")
            
            # Projections futures
            st.markdown("## 🔮 Projections Futures")
            projections = result.get('future_projections', {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Amélioration Prochaine Q", 
                         f"+{projections.get('expected_improvement_next_quarter', 0):.1f}%")
            with col2:
                st.metric("Économies Annuelles Projetées",
                         f"${projections.get('projected_energy_savings_annual_usd', 0):,.0f}")
            with col3:
                st.metric("Projection ROI",
                         f"{projections.get('roi_projection_percentage', 0):.0f}%")
            
            # Export
            st.markdown("---")
            if st.button("💾 Exporter le Rapport (JSON)", use_container_width=True):
                st.download_button(
                    label="📥 Télécharger",
                    data=json.dumps(result, indent=2),
                    file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# ==================== ANALYTICS ====================

elif menu == "📊 Analytics":
    st.title("📊 Analytics Avancés")
    
    analytics = api_request("/api/analytics/global")
    
    if analytics:
        # Métriques en temps réel
        st.markdown("### ⚡ Métriques en Temps Réel")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Stratégies", analytics['total_strategies'])
        with col2:
            st.metric("Systèmes", analytics['total_systems'])
        with col3:
            st.metric("Benchmarks", analytics['total_benchmarks'])
        with col4:
            st.metric("Optimisations", analytics['total_optimizations'])
        with col5:
            st.metric("Modèles IA", analytics['total_ai_models'])
        
        # Comparaison des performances
        st.markdown("### 📈 Comparaison des Performances")
        
        comparison_data = pd.DataFrame({
            'Méthode': ['Baseline', 'IA', 'Quantum', 'Hybrid'],
            'Performance': [100, 100 + analytics['average_performance_improvement'] * 0.6,
                          100 + analytics['quantum_advantage_average'] * 2,
                          100 + analytics['average_performance_improvement']],
            'Énergie': [100, 100 - analytics['average_energy_savings'] * 0.5,
                       100 - analytics['average_energy_savings'] * 0.7,
                       100 - analytics['average_energy_savings']]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Performance', x=comparison_data['Méthode'], 
                            y=comparison_data['Performance'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Énergie', x=comparison_data['Méthode'],
                            y=comparison_data['Énergie'], marker_color='#38ef7d'))
        fig.update_layout(
            title="Comparaison Baseline vs Optimisations",
            barmode='group',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
                        

# ==================== FOOTER ====================

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h4 style="color: #667eea;">⚡ Quantum Performance Engine V2.0</h4>
    <p style="font-size: 0.9rem;">Optimisation IA & Quantique</p>
    <p style="font-size: 0.8rem; color: gray;">© 2024 - Tous droits réservés</p>
    
    <div style="margin-top: 1rem;">
        <p style="font-size: 0.8rem;">
            🚀 Powered by FastAPI & Streamlit<br>
            🤖 IA + ⚛️ Quantum = 🔥 Performance
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Info bulle d'aide
with st.sidebar.expander("❓ Aide Rapide"):
    st.markdown("""
    **🎯 Stratégies**: Créez et appliquez des stratégies d'optimisation
    
    **💻 Systèmes**: Gérez vos systèmes (Binary/Quantum/Hybrid)
    
    **📊 Benchmarking**: Testez les performances
    
    **🤖 IA**: Utilisez l'apprentissage automatique
    
    **⚛️ Quantum**: Exploitez la puissance quantique
    
    **📈 Profilage**: Analysez en profondeur
    
    **⚡ Énergie**: Optimisez la consommation
    
    **🔮 Maintenance**: Prédisez les pannes
    
    **📑 Rapports**: Générez des rapports complets
    """)

# Notes de version
with st.sidebar.expander("📝 Notes de Version"):
    st.markdown("""
    **Version 2.0.0** - Dernière version
    
    ✨ **Nouvelles fonctionnalités:**
    - Auto-tuning intelligent
    - Maintenance prédictive IA
    - Cache quantum-optimisé
    - Load balancing avancé
    - Compression intelligente
    - Analytics en temps réel
    - Rapports complets
    - Interface moderne et réactive
    
    🔧 **Améliorations:**
    - Performance UI optimisée
    - Meilleure gestion d'erreurs
    - Graphiques interactifs
    - Export/Import de données
    
    🐛 **Corrections:**
    - Stabilité générale améliorée
    - Meilleure compatibilité API
    """)

# Raccourcis clavier (info)
with st.sidebar.expander("⌨️ Raccourcis"):
    st.markdown("""
    **Navigation:**
    - `R` : Rafraîchir la page
    - `Ctrl + K` : Recherche rapide
    - `Ctrl + /` : Afficher les raccourcis
    
    **Actions:**
    - `Ctrl + S` : Sauvegarder (si applicable)
    - `Ctrl + E` : Export
    - `Esc` : Fermer les dialogues
    """)

# Mode debug (caché)
if st.sidebar.checkbox("🔧 Mode Debug", value=False):
    st.sidebar.json({
        'session_state_keys': list(st.session_state.keys()),
        'selected_system': st.session_state.get('selected_system'),
        'selected_strategy': st.session_state.get('selected_strategy'),
        'api_url': st.session_state.api_url
    })
    width=True
        
    with col2:
            fig = create_gauge_chart(
                analytics['average_energy_savings'],
                "Économies Énergie (%)",
                max_value=100,
                color='#38ef7d'
            )
            st.plotly_chart(fig, use_container_width=True)
        
    with col3:
            fig = create_gauge_chart(
                analytics['quantum_advantage_average'],
                "Avantage Quantique (x)",
                max_value=100,
                color='#f093fb'
            )
            st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
        
        # Graphiques de tendances
    st.markdown("### 📈 Tendances de Performance")
        
        # Simulation de données historiques
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    perf_data = pd.DataFrame({
            'Date': dates,
            'Performance': np.random.uniform(20, 60, 30).cumsum() / 10,
            'Énergie': np.random.uniform(10, 40, 30).cumsum() / 10,
            'Quantum': np.random.uniform(15, 50, 30).cumsum() / 10
        })
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=perf_data['Date'], y=perf_data['Performance'],
                                name='Performance', line=dict(color='#667eea', width=3)))
    fig.add_trace(go.Scatter(x=perf_data['Date'], y=perf_data['Énergie'],
                                name='Énergie', line=dict(color='#38ef7d', width=3)))
    fig.add_trace(go.Scatter(x=perf_data['Date'], y=perf_data['Quantum'],
                                name='Quantum', line=dict(color='#f093fb', width=3)))
        
    fig.update_layout(
            title="Évolution des Améliorations (30 jours)",
            xaxis_title="Date",
            yaxis_title="Amélioration (%)",
            height=400,
            hovermode='x unified'
        )
    st.plotly_chart(fig, use_container_width=True)
        
        # Statut des systèmes
    st.markdown("### 💻 Statut des Systèmes")
    systems = api_request("/api/system/list", show_error=False) or []
        
    if systems:
        system_types = {}
        for sys in systems:
            sys_type = sys.get('type', 'unknown')
            system_types[sys_type] = system_types.get(sys_type, 0) + 1
            
        fig = go.Figure(data=[go.Pie(
            labels=list(system_types.keys()),
            values=list(system_types.values()),
            hole=.3,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb'])
        )])
        fig.update_layout(title="Répartition des Systèmes", height=400)
        st.plotly_chart(fig, use_container_width=True)
