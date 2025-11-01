"""
platform_tester_frontend.py - Interface Streamlit pour Platform Testing System

Installation:
pip install streamlit requests plotly pandas

Lancement:
streamlit run plateforme_test_app.py
"""

import streamlit as st
import requests
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8037"

# Configuration de la page
st.set_page_config(
    page_title="Platform Testing & Analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Fonctions API
def test_platform_url(url, name, category, num_agents, duration):
    """Lance un test par URL"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/test/url",
            json={
                "platform_url": url,
                "platform_name": name,
                "category": category,
                "num_agents": num_agents,
                "test_duration_minutes": duration
            }
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_platform_features(name, category, features, num_agents, description=None):
    """Lance un test par fonctionnalités"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/test/features",
            json={
                "platform_name": name,
                "category": category,
                "features": features,
                "description": description,
                "num_agents": num_agents
            }
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_test_results(test_id):
    """Récupère les résultats d'un test"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/test/{test_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_lifecycle(platform_id):
    """Récupère le cycle de vie"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/lifecycle/{platform_id}")
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def create_promotion(platform_id, audience, budget):
    """Crée une promotion"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/promotion/create",
            json={
                "platform_id": platform_id,
                "target_audience": audience,
                "budget": budget
            }
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# Interface principale
def main():
    # Header
    st.markdown('<div class="main-header">🧪 Platform Testing & Analysis System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=AI+Testing", use_container_width=True)
        st.title("Navigation")
        page = st.radio(
            "Sélectionnez une section:",
            ["🏠 Accueil", "🔗 Test par URL", "⚙️ Test par Fonctionnalités", 
             "📊 Résultats", "🔄 Cycle de Vie", "📢 Promotions"]
        )
        
        st.markdown("---")
        st.info("**API Status**")
        try:
            health = requests.get(f"{API_BASE_URL}/health").json()
            st.success(f"✅ {health['status'].upper()}")
            st.metric("Tests actifs", health['tests'])
            st.metric("Plateformes", health['platforms'])
        except:
            st.error("❌ API non accessible")
    
    # Pages
    if page == "🏠 Accueil":
        show_home()
    elif page == "🔗 Test par URL":
        show_url_test()
    elif page == "⚙️ Test par Fonctionnalités":
        show_feature_test()
    elif page == "📊 Résultats":
        show_results()
    elif page == "🔄 Cycle de Vie":
        show_lifecycle()
    elif page == "📢 Promotions":
        show_promotions()

def show_home():
    """Page d'accueil"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖</h2>
            <h3>Agents IA</h3>
            <p>5 types d'agents spécialisés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>📈</h2>
            <h3>Analyse Marché</h3>
            <p>Insights automatiques</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🔄</h2>
            <h3>Cycle de Vie</h3>
            <p>Gestion complète</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("🎯 Fonctionnalités")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Tests Automatisés")
        st.write("""
        - **Test par URL**: Analysez n'importe quelle plateforme web
        - **Test par Fonctionnalités**: Évaluez des features spécifiques
        - **Multi-agents**: Jusqu'à 50 agents simultanés
        - **5 spécialités**: UX, Performance, Sécurité, Fonctionnel, SEO
        """)
        
        st.subheader("📊 Analyse de Marché")
        st.write("""
        - Taille du marché
        - Taux de croissance
        - Niveau de concurrence
        - Opportunités et menaces
        - Stratégies de pricing
        """)
    
    with col2:
        st.subheader("🔄 Gestion du Cycle de Vie")
        st.write("""
        - Suivi des phases (Idéation → Optimisation)
        - Jalons et métriques
        - Plan d'évolution
        - KPIs détaillés
        """)
        
        st.subheader("📢 Système de Promotion")
        st.write("""
        - Création de campagnes
        - Ciblage d'audience
        - Gestion de budget
        - Suivi des conversions
        """)
    
    st.markdown("---")
    st.info("💡 **Astuce**: Commencez par un test URL pour une analyse complète de votre plateforme!")

def show_url_test():
    """Page de test par URL"""
    st.header("🔗 Test de Plateforme par URL")
    
    st.markdown('<div class="info-box">Testez n\'importe quelle plateforme web avec nos agents IA spécialisés</div>', unsafe_allow_html=True)
    
    with st.form("url_test_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            platform_url = st.text_input(
                "URL de la plateforme*",
                placeholder="https://exemple.com",
                help="URL complète de la plateforme à tester"
            )
            
            platform_name = st.text_input(
                "Nom de la plateforme*",
                placeholder="Ma Super Plateforme",
                help="Nom descriptif de votre plateforme"
            )
            
            category = st.selectbox(
                "Catégorie*",
                ["ecommerce", "social", "education", "finance", "saas", "marketplace", "gaming", "other"],
                help="Catégorie de votre plateforme"
            )
        
        with col2:
            num_agents = st.slider(
                "Nombre d'agents IA",
                min_value=1,
                max_value=50,
                value=5,
                help="Plus d'agents = analyse plus complète"
            )
            
            duration = st.slider(
                "Durée du test (minutes)",
                min_value=5,
                max_value=120,
                value=30,
                help="Durée simulée du test"
            )
            
            st.markdown("**Agents utilisés:**")
            agents_info = {
                "🎨 UX Tester": "Navigation, Design, Accessibilité",
                "⚡ Performance": "Vitesse, Optimisation, Cache",
                "🔒 Sécurité": "HTTPS, Headers, Protections",
                "⚙️ Fonctionnel": "Formulaires, Recherche, Paiement",
                "🔍 SEO": "Meta tags, Structure, Sitemap"
            }
            for agent, desc in list(agents_info.items())[:num_agents]:
                st.caption(f"{agent}: {desc}")
        
        submitted = st.form_submit_button("🚀 Lancer le Test", use_container_width=True)
        
        if submitted:
            if not platform_url or not platform_name:
                st.error("⚠️ Veuillez remplir tous les champs obligatoires")
            else:
                with st.spinner("🔄 Lancement du test..."):
                    result = test_platform_url(platform_url, platform_name, category, num_agents, duration)
                    
                    if result.get("success"):
                        st.markdown(f'<div class="success-box">✅ Test lancé avec succès!<br>Test ID: <b>{result["test_id"]}</b></div>', unsafe_allow_html=True)
                        st.session_state['last_test_id'] = result['test_id']
                        
                        # Simulation de progression
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            if i < 20:
                                status_text.text("🔍 Initialisation des agents...")
                            elif i < 60:
                                status_text.text(f"🤖 Tests en cours... ({num_agents} agents actifs)")
                            elif i < 90:
                                status_text.text("📊 Analyse des résultats...")
                            else:
                                status_text.text("✅ Finalisation...")
                            time.sleep(0.05)
                        
                        st.success("✨ Test terminé! Consultez les résultats dans l'onglet 📊 Résultats")
                        st.balloons()
                    else:
                        st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")

def show_feature_test():
    """Page de test par fonctionnalités"""
    st.header("⚙️ Test de Plateforme par Fonctionnalités")
    
    st.markdown('<div class="info-box">Évaluez des fonctionnalités spécifiques de votre plateforme</div>', unsafe_allow_html=True)
    
    with st.form("feature_test_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            platform_name = st.text_input(
                "Nom de la plateforme*",
                placeholder="Ma Plateforme"
            )
            
            category = st.selectbox(
                "Catégorie*",
                ["ecommerce", "social", "education", "finance", "saas", "marketplace", "gaming", "other"]
            )
            
            description = st.text_area(
                "Description (optionnelle)",
                placeholder="Description de votre plateforme...",
                height=100
            )
        
        with col2:
            st.subheader("Fonctionnalités à tester")
            
            # Exemples pré-définis
            feature_templates = {
                "E-commerce": ["Panier d'achat", "Paiement sécurisé", "Gestion de stock", "Suivi commandes"],
                "Social": ["Messagerie", "Fil d'actualité", "Notifications", "Profils utilisateurs"],
                "SaaS": ["Dashboard", "Analytics", "API", "Intégrations"],
                "Custom": []
            }
            
            template = st.selectbox("Template de fonctionnalités", list(feature_templates.keys()))
            
            if template != "Custom":
                features_input = st.multiselect(
                    "Sélectionnez les fonctionnalités",
                    feature_templates[template],
                    default=feature_templates[template][:2]
                )
            else:
                features_text = st.text_area(
                    "Entrez les fonctionnalités (une par ligne)",
                    placeholder="Fonction 1\nFonction 2\nFonction 3",
                    height=150
                )
                features_input = [f.strip() for f in features_text.split('\n') if f.strip()]
            
            num_agents = st.slider("Nombre d'agents", 1, 50, 5)
        
        submitted = st.form_submit_button("🚀 Lancer le Test", use_container_width=True)
        
        if submitted:
            if not platform_name or not features_input:
                st.error("⚠️ Veuillez remplir tous les champs et sélectionner des fonctionnalités")
            else:
                with st.spinner("🔄 Lancement du test..."):
                    result = test_platform_features(platform_name, category, features_input, num_agents, description)
                    
                    if result.get("success"):
                        st.markdown(f'<div class="success-box">✅ Test lancé!<br>Test ID: <b>{result["test_id"]}</b></div>', unsafe_allow_html=True)
                        st.session_state['last_test_id'] = result['test_id']
                        
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            time.sleep(0.03)
                        
                        st.success("✨ Test terminé!")
                        st.balloons()
                    else:
                        st.error(f"❌ Erreur: {result.get('error')}")

def show_results():
    """Page des résultats"""
    st.header("📊 Résultats des Tests")
    
    # Récupération du test ID
    test_id = st.text_input(
        "ID du Test",
        value=st.session_state.get('last_test_id', ''),
        placeholder="Entrez l'ID du test"
    )
    
    if st.button("🔍 Charger les Résultats", use_container_width=True):
        if test_id:
            with st.spinner("📥 Chargement des résultats..."):
                data = get_test_results(test_id)
                
                if "error" in data:
                    st.error(f"❌ {data['error']}")
                elif data.get("status") == "pending":
                    st.warning("⏳ Test en attente...")
                elif data.get("status") == "running":
                    st.info("🔄 Test en cours d'exécution...")
                elif data.get("status") == "completed":
                    show_detailed_results(data)
                else:
                    st.info("ℹ️ Aucun résultat disponible")
        else:
            st.warning("⚠️ Veuillez entrer un ID de test")

def show_detailed_results(data):
    """Affiche les résultats détaillés"""
    results = data.get("results", {})
    
    # En-tête
    st.success(f"✅ Test terminé pour **{data['platform_name']}**")
    
    # Score global
    col1, col2, col3, col4 = st.columns(4)
    
    final_score = results.get("final_score", 0)
    grade = results.get("grade", "N/A")
    
    with col1:
        st.metric("Score Final", f"{final_score}/30", delta=None)
    with col2:
        st.metric("Note", grade)
    with col3:
        st.metric("Agents", data.get("num_agents", 0))
    with col4:
        st.metric("Catégorie", data.get("category", "N/A").upper())
    
    st.markdown("---")
    
    # Graphique de score
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=final_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Score Global"},
        delta={'reference': 25},
        gauge={
            'axis': {'range': [None, 30]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 15], 'color': "lightgray"},
                {'range': [15, 24], 'color': "gray"},
                {'range': [24, 30], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 27
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    # Résultats des agents
    st.subheader("🤖 Résultats par Agent")
    
    agent_results = results.get("agent_results", [])
    
    if agent_results:
        for idx, agent in enumerate(agent_results):
            with st.expander(f"Agent {idx+1}: {agent['agent_type']} - Score: {agent.get('score', 0):.1f}/10"):
                
                # Tests effectués
                st.write("**Tests effectués:**")
                tests = agent.get("tests_performed", [])
                if tests:
                    df_tests = pd.DataFrame(tests)
                    st.dataframe(df_tests, use_container_width=True)
                
                # Problèmes détectés
                issues = agent.get("issues_found", [])
                if issues:
                    st.write("**⚠️ Problèmes détectés:**")
                    for issue in issues:
                        severity_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                        st.write(f"{severity_color.get(issue['severity'], '⚪')} **{issue['issue']}** - Impact: {issue['impact']}")
                
                # Recommandations
                recs = agent.get("recommendations", [])
                if recs:
                    st.write("**💡 Recommandations:**")
                    for rec in recs:
                        st.write(f"- {rec}")
    
    st.markdown("---")
    
    # Analyse de marché
    st.subheader("📈 Analyse de Marché")
    
    market = results.get("market_analysis", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Taille du Marché", f"${market.get('market_size_billions', 0)}B")
        st.metric("Croissance", f"{market.get('growth_rate_percent', 0)}%")
        st.metric("Concurrence", market.get('competition_level', 'N/A'))
        
        # Opportunités
        st.write("**🎯 Opportunités:**")
        for opp in market.get('opportunities', []):
            st.write(f"✅ {opp}")
    
    with col2:
        # Pricing
        pricing = market.get('pricing_strategy', {})
        st.write("**💰 Stratégie de Prix:**")
        st.write(f"Modèle: {pricing.get('model', 'N/A')}")
        
        tiers = pricing.get('tiers', [])
        if tiers:
            df_pricing = pd.DataFrame(tiers)
            st.dataframe(df_pricing, use_container_width=True)
        
        # Canaux marketing
        st.write("**📢 Canaux Marketing:**")
        for channel in market.get('marketing_channels', []):
            st.write(f"- {channel}")
    
    # Bouton d'action
    if data.get('platform_id'):
        st.markdown("---")
        if st.button("🔄 Voir le Cycle de Vie", use_container_width=True):
            st.session_state['active_platform_id'] = data['platform_id']
            st.rerun()

def show_lifecycle():
    """Page du cycle de vie"""
    st.header("🔄 Gestion du Cycle de Vie")
    
    platform_id = st.text_input(
        "ID de la Plateforme",
        value=st.session_state.get('active_platform_id', ''),
        placeholder="Entrez l'ID de la plateforme"
    )
    
    if st.button("📥 Charger le Cycle", use_container_width=True):
        if platform_id:
            data = get_lifecycle(platform_id)
            
            if "error" not in data:
                st.success(f"✅ Cycle de vie chargé pour la plateforme {platform_id}")
                
                # Phase actuelle
                # st.subheader(f"📍 Phase actuelle: {data['current_phase']}")
                st.subheader(f"📍 Phase actuelle: {data.get('current_phase', 'Non définie')}")
                
                # Progression
                phases = ["Idéation", "MVP", "Lancement", "Croissance", "Maturité", "Optimisation"]
                current_idx = data.get('phase_index', 0)
                progress = (current_idx / len(phases)) * 100
                
                st.progress(progress / 100)
                
                col1, col2, col3 = st.columns(3)
                
                # Métriques
                metrics = data.get('metrics', {})
                with col1:
                    st.metric("Utilisateurs", metrics.get('users', 0))
                with col2:
                    st.metric("Revenu", f"${metrics.get('revenue', 0)}")
                with col3:
                    st.metric("Engagement", f"{metrics.get('engagement', 0)}%")
                
                st.markdown("---")
                
                # Jalons
                st.subheader("🎯 Jalons")
                milestones = data.get('milestones', [])
                for milestone in milestones:
                    status_icon = "✅" if milestone['status'] == "completed" else "⏳"
                    st.write(f"{status_icon} **{milestone['milestone']}** ({milestone['phase']}) - Cible: {milestone['target_date'][:10]}")
                
                st.markdown("---")
                
                # Plan d'évolution
                st.subheader("📅 Plan d'Évolution")
                evolution = data.get('evolution_plan', [])
                for plan in evolution:
                    with st.expander(f"Mois {plan['month']}: {plan['focus']}"):
                        st.write("**Actions:**")
                        for action in plan['actions']:
                            st.write(f"- {action}")
                        st.write("**KPIs:**")
                        st.json(plan['kpis'])
            else:
                st.error(f"❌ {data['error']}")

def show_promotions():
    """Page des promotions"""
    st.header("📢 Système de Promotion")
    
    st.markdown('<div class="info-box">Créez des campagnes de promotion pour votre plateforme</div>', unsafe_allow_html=True)
    
    with st.form("promotion_form"):
        platform_id = st.text_input("ID de la Plateforme*", placeholder="platform-id-123")
        
        col1, col2 = st.columns(2)
        
        with col1:
            audience_options = [
                "Développeurs",
                "Entrepreneurs",
                "PME",
                "Grandes Entreprises",
                "Étudiants",
                "Freelances",
                "Startups"
            ]
            target_audience = st.multiselect("Audience Cible*", audience_options)
            
        with col2:
            budget = st.number_input("Budget ($)", min_value=0.0, value=1000.0, step=100.0)
        
        submitted = st.form_submit_button("🚀 Créer la Promotion", use_container_width=True)
        
        if submitted:
            if not platform_id or not target_audience:
                st.error("⚠️ Veuillez remplir tous les champs obligatoires")
            else:
                result = create_promotion(platform_id, target_audience, budget)
                
                if result.get("success"):
                    promo = result['promotion']
                    st.markdown(f'<div class="success-box">✅ Promotion créée!<br>Promo ID: <b>{promo["promo_id"]}</b></div>', unsafe_allow_html=True)
                    
                    st.json(promo)
                    st.balloons()
                else:
                    st.error(f"❌ Erreur: {result.get('error')}")
    
    st.markdown("---")
    st.info("💡 Les promotions sont diffusées automatiquement dans la communauté selon votre audience cible et votre budget")

if __name__ == "__main__":
    # Initialisation session state
    if 'last_test_id' not in st.session_state:
        st.session_state['last_test_id'] = ''
    if 'active_platform_id' not in st.session_state:
        st.session_state['active_platform_id'] = ''
    
    main()