"""
av_quantum_frontend.py - Interface Streamlit pour AV Quantum Engine

Installation:
pip install streamlit requests plotly pandas numpy

Lancement:
streamlit run autonomous_vehicle_app.py
"""

import streamlit as st
import requests
import time
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Configuration
API_BASE_URL = "http://localhost:8010"

st.set_page_config(
    page_title="AV Quantum Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Futuriste
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;700&display=swap');
    
    .av-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .module-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .stage-card {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
    }
    
    .quantum-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions API
def api_call(method, endpoint, **kwargs):
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.markdown('<div class="av-header">🚗 AV QUANTUM ENGINE</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=AV+Quantum", use_container_width=True)
        st.title("🎯 Navigation")
        
        page = st.radio("Navigation", [
            "🏠 Dashboard",
            "🚗 Nouveau Module AV",
            "💼 Workplace",
            "📊 Génération Données",
            "🧠 Entraînement IA",
            "🧪 Tests & Validation",
            "🛒 Marketplace",
            "⚛️ Projets Quantiques",
            "📚 Apprentissage"
        ])
        
        st.markdown("---")
        try:
            health = requests.get(f"{API_BASE_URL}/health").json()
            st.success("✅ API Active")
            st.metric("Modules", health.get('modules', 0))
            st.metric("Datasets", health.get('datasets', 0))
            st.metric("Modèles", health.get('models', 0))
        except:
            st.error("❌ API Offline")
    
    # Routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🚗 Nouveau Module AV":
        show_create_module()
    elif page == "💼 Workplace":
        show_workplace()
    elif page == "📊 Génération Données":
        show_data_generation()
    elif page == "🧠 Entraînement IA":
        show_model_training()
    elif page == "🧪 Tests & Validation":
        show_testing()
    elif page == "🛒 Marketplace":
        show_marketplace()
    elif page == "⚛️ Projets Quantiques":
        show_quantum_projects()
    elif page == "📚 Apprentissage":
        show_learning()

def show_dashboard():
    st.header("🏠 Tableau de Bord")
    
    stats = api_call("GET", "/api/v1/stats/overview")
    
    if "error" not in stats:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="module-card">
                <h3>🚗 Modules AV</h3>
                <h1>{stats.get('modules', {}).get('total', 0)}</h1>
                <p>{stats.get('modules', {}).get('completed', 0)} complétés</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="module-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>📊 Datasets</h3>
                <h1>{stats.get('datasets', {}).get('total', 0)}</h1>
                <p>{stats.get('datasets', {}).get('total_size_gb', 0):.1f} GB</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="module-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>🧠 Modèles IA</h3>
                <h1>{stats.get('models', {}).get('total', 0)}</h1>
                <p>{stats.get('models', {}).get('trained', 0)} entraînés</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="module-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>🛒 Marketplace</h3>
                <h1>{stats.get('marketplace', {}).get('total_listings', 0)}</h1>
                <p>{stats.get('marketplace', {}).get('published', 0)} publiés</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Types de modules
        st.subheader("📊 Répartition des Modules")
        
        modules_by_type = stats.get('modules', {}).get('by_type', {})
        if modules_by_type:
            df = pd.DataFrame([
                {"Type": k.replace("_", " ").title(), "Count": v}
                for k, v in modules_by_type.items()
            ])
            
            fig = px.bar(df, x='Type', y='Count', color='Count',
                        color_continuous_scale='Blues')
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

def show_create_module():
    st.header("🚗 Créer un Nouveau Module AV")
    
    # Templates
    templates = api_call("GET", "/api/v1/templates/module-types")
    
    if "templates" in templates:
        st.subheader("📋 Templates Disponibles")
        
        for template in templates['templates']:
            with st.expander(f"🔧 {template['name']}"):
                st.write(f"**Description:** {template['description']}")
                st.write(f"**Complexité:** {template['complexity']}")
                st.write(f"**Durée estimée:** {template['estimated_duration_days']} jours")
                
                if st.button(f"Utiliser ce template", key=template['type']):
                    st.session_state['selected_template'] = template['type']
    
    st.markdown("---")
    st.subheader("🆕 Création")
    
    with st.form("create_module_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom du Module*", placeholder="Mon Module AV")
            
            module_type = st.selectbox("Type*", [
                "perception", "localization", "planning", "control",
                "prediction", "decision_making", "sensor_fusion",
                "mapping", "v2x_communication", "safety_monitoring"
            ], format_func=lambda x: x.replace("_", " ").title())
            
            description = st.text_area("Description*", height=100)
        
        with col2:
            use_quantum = st.checkbox("Utiliser IA Quantique", value=True)
            target_accuracy = st.slider("Précision Cible", 0.80, 1.0, 0.95, 0.01)
            
            if use_quantum:
                st.markdown('<span class="quantum-badge">⚛️ QUANTUM ACCELERATED</span>', unsafe_allow_html=True)
                st.info("Speedup estimé: **256x** vs classique")
        
        if st.form_submit_button("🚀 Créer le Module", use_container_width=True):
            if name and description:
                result = api_call("POST", "/api/v1/module/create", json={
                    "module_name": name,
                    "module_type": module_type,
                    "description": description,
                    "use_quantum": use_quantum,
                    "target_accuracy": target_accuracy
                })
                
                if result.get("success"):
                    module = result['module']
                    st.success(f"✅ Module créé: {module['module_id']}")
                    st.session_state['active_module_id'] = module['module_id']
                    st.json(module)
                    st.balloons()

def show_workplace():
    st.header("💼 Workplace de Développement")
    
    module_id = st.text_input("Module ID", value=st.session_state.get('active_module_id', ''))
    
    if st.button("📂 Ouvrir le Workplace"):
        if not module_id:
            st.warning("⚠️ Veuillez entrer un Module ID")
            return
        
        workplace = api_call("GET", f"/api/v1/workplace/{module_id}")
        
        if "error" in workplace:
            st.error(f"❌ {workplace['error']}")
            return
        
        if workplace.get('completed'):
            st.success("✅ Tous les stages sont complétés!")
            return
        
        # Header
        # st.success(f"📁 {workplace['module_name']}")
        st.success(f"📁 {workplace.get('module_name', 'Nom inconnu')}")
        
        # Progress
        progress = workplace['progress']
        st.progress(progress / 100)
        st.write(f"**Progression:** {progress}% - Stage {workplace['stage_number']}/{workplace['total_stages']}")
        
        st.markdown("---")
        
        # Stage actuel
        current_stage = workplace['current_stage']
        
        st.markdown(f"""
        <div class="module-card">
            <h2>🎯 Stage Actuel: {current_stage.get('name', 'N/A')}</h2>
            <p>{current_stage.get('description', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tâches
        st.subheader("📋 Tâches à Réaliser")
        
        for i, task in enumerate(current_stage.get('tasks', []), 1):
            st.markdown(f"""
            <div class="stage-card">
                <strong>{i}.</strong> {task}
            </div>
            """, unsafe_allow_html=True)
        
        # Outils
        st.subheader("🛠️ Outils Nécessaires")
        
        tools = current_stage.get('tools', [])
        cols = st.columns(len(tools) if tools else 1)
        
        for i, tool in enumerate(tools):
            with cols[i]:
                st.info(f"🔧 {tool}")
        
        # Actions
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Valider ce Stage", use_container_width=True):
                result = api_call("POST", f"/api/v1/module/{module_id}/stage/{workplace['stage_number']-1}/complete")
                
                if result.get("success"):
                    st.success("✅ Stage validé!")
                    st.rerun()
        
        with col2:
            if st.button("📊 Voir les Détails", use_container_width=True):
                module = api_call("GET", f"/api/v1/module/{module_id}")
                st.json(module)

def show_data_generation():
    st.header("📊 Génération de Données")
    
    st.info("💡 Générez des données quantiques ou classiques pour l'entraînement")
    
    with st.form("data_gen_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            module_id = st.text_input("Module ID*", value=st.session_state.get('active_module_id', ''))
            
            data_types = st.multiselect("Types de Capteurs*", [
                "lidar", "camera", "radar", "gps", "imu", "ultrasonic", "v2x"
            ], default=["camera", "lidar"])
            
            scenario = st.selectbox("Scénario*", [
                "urban", "highway", "rural", "adverse_weather", "night", "mixed"
            ])
        
        with col2:
            quantity_gb = st.slider("Quantité (GB)", 0.1, 1000.0, 10.0)
            use_quantum = st.checkbox("Génération Quantique", value=True)
            
            if use_quantum:
                st.markdown('<span class="quantum-badge">⚛️ QUANTUM</span>', unsafe_allow_html=True)
                st.info(f"""
                **Avantages:**
                - Diversité: +40%
                - Réalisme: +30%
                - Vitesse: 10x
                """)
        
        if st.form_submit_button("🚀 Générer les Données", use_container_width=True):
            if module_id and data_types:
                result = api_call("POST", "/api/v1/data/generate", json={
                    "module_id": module_id,
                    "data_types": data_types,
                    "scenario": scenario,
                    "quantity_gb": quantity_gb,
                    "use_quantum_generation": use_quantum
                })
                
                if result.get("success"):
                    st.success(f"✅ Génération lancée: {result['dataset_id']}")
                    st.session_state['last_dataset_id'] = result['dataset_id']
                    
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        time.sleep(0.02)
                    
                    st.balloons()
    
    # Visualiser un dataset
    st.markdown("---")
    st.subheader("📂 Datasets Existants")
    
    dataset_id = st.text_input("Dataset ID", value=st.session_state.get('last_dataset_id', ''))
    
    if st.button("📊 Charger"):
        if dataset_id:
            dataset = api_call("GET", f"/api/v1/data/{dataset_id}")
            
            if "error" not in dataset:
                st.success(f"✅ Dataset: {dataset.get('scenario', 'N/A').upper()}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Taille", f"{dataset.get('size_gb', 0):.1f} GB")
                with col2:
                    st.metric("Échantillons", f"{dataset.get('total_samples', 0):,}")
                with col3:
                    st.metric("Qualité", f"{dataset.get('quality_metrics', {}).get('realism_score', 0):.1%}")
                
                st.json(dataset)

def show_model_training():
    st.header("🧠 Entraînement de Modèles IA")
    
    tab1, tab2 = st.tabs(["🆕 Nouvel Entraînement", "📊 Résultats"])
    
    with tab1:
        with st.form("training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                module_id = st.text_input("Module ID*")
                dataset_id = st.text_input("Dataset ID*")
                
                model_type = st.selectbox("Type de Modèle*", [
                    "classical_cnn", "quantum_cnn",
                    "classical_rnn", "quantum_rnn",
                    "transformer", "quantum_transformer",
                    "hybrid_quantum_classical"
                ], format_func=lambda x: x.replace("_", " ").upper())
            
            with col2:
                epochs = st.slider("Époques", 1, 10000, 100)
                use_quantum = st.checkbox("Accélération Quantique", value=True)
                
                if use_quantum:
                    num_qubits = st.slider("Qubits", 4, 64, 16)
                    speedup = 2 ** (num_qubits / 4)
                    st.info(f"Speedup: **{speedup:.0f}x**")
                else:
                    num_qubits = 4
            
            if st.form_submit_button("🚀 Lancer l'Entraînement", use_container_width=True):
                if module_id and dataset_id:
                    result = api_call("POST", "/api/v1/model/train", json={
                        "module_id": module_id,
                        "model_type": model_type,
                        "dataset_id": dataset_id,
                        "epochs": epochs,
                        "use_quantum_acceleration": use_quantum,
                        "num_qubits": num_qubits
                    })
                    
                    if result.get("success"):
                        st.success(f"✅ Entraînement lancé: {result['model_id']}")
                        st.session_state['last_model_id'] = result['model_id']
    
    with tab2:
        model_id = st.text_input("Model ID", value=st.session_state.get('last_model_id', ''))
        
        if st.button("📊 Charger les Résultats"):
            if not model_id:
                st.warning("⚠️ Veuillez entrer un Model ID")
                return
            
            model = api_call("GET", f"/api/v1/model/{model_id}")
            
            if "error" in model:
                st.error(f"❌ {model['error']}")
                return
            
            st.success(f"✅ Modèle: {model.get('model_type', 'N/A').upper()}")
            
            # Métriques finales
            final_metrics = model.get('final_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{final_metrics.get('accuracy', 0):.2%}")
            with col2:
                st.metric("Precision", f"{final_metrics.get('precision', 0):.2%}")
            with col3:
                st.metric("Recall", f"{final_metrics.get('recall', 0):.2%}")
            with col4:
                st.metric("F1-Score", f"{final_metrics.get('f1_score', 0):.2%}")
            
            # Performance
            perf = model.get('performance', {})
            if perf.get('quantum_advantage'):
                st.success(f"⚡ Speedup Quantique: **{perf.get('speedup_factor', 1):.1f}x**")
            
            # Historique
            if 'training_history' in model and model['training_history']:
                st.subheader("📈 Historique d'Entraînement")
                
                df = pd.DataFrame(model['training_history'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['epoch'], y=df['accuracy'], name='Accuracy', line=dict(color='#4facfe', width=3)))
                fig.add_trace(go.Scatter(x=df['epoch'], y=df['f1_score'], name='F1-Score', line=dict(color='#43e97b', width=3)))
                fig.update_layout(template="plotly_dark", title="Métriques de Performance")
                st.plotly_chart(fig, use_container_width=True)

def show_testing():
    st.header("🧪 Tests & Validation")
    
    tab1, tab2 = st.tabs(["🚀 Lancer Tests", "📊 Résultats"])
    
    with tab1:
        with st.form("test_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                module_id = st.text_input("Module ID*")
                
                test_scenarios = st.multiselect("Scénarios de Test*", [
                    "Urban Navigation",
                    "Highway Merging",
                    "Emergency Braking",
                    "Adverse Weather",
                    "Night Driving",
                    "Pedestrian Detection",
                    "Traffic Light Recognition",
                    "Lane Keeping"
                ], default=["Urban Navigation", "Emergency Braking"])
            
            with col2:
                safety_critical = st.checkbox("Test Safety-Critical", value=True)
                real_time_ms = st.slider("Contrainte Temps Réel (ms)", 1, 1000, 100)
                
                st.info(f"""
                **Standards:**
                - ISO 26262 (Safety)
                - ASIL-D Compliant
                - Real-time: < {real_time_ms}ms
                """)
            
            if st.form_submit_button("🧪 Lancer les Tests", use_container_width=True):
                if module_id and test_scenarios:
                    result = api_call("POST", "/api/v1/test/run", json={
                        "module_id": module_id,
                        "test_scenarios": test_scenarios,
                        "safety_critical": safety_critical,
                        "real_time_constraints_ms": real_time_ms
                    })
                    
                    if result.get("success"):
                        st.success(f"✅ Tests lancés: {result['test_id']}")
                        st.session_state['last_test_id'] = result['test_id']
                        
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                            time.sleep(0.03)
    
    with tab2:
        test_id = st.text_input("Test ID", value=st.session_state.get('last_test_id', ''))
        
        if st.button("📊 Charger Résultats"):
            if not test_id:
                st.warning("⚠️ Veuillez entrer un Test ID")
                return
            
            test = api_call("GET", f"/api/v1/test/{test_id}")
            
            if "error" in test:
                st.error(f"❌ {test['error']}")
                return
            
            st.success("✅ Tests Complétés")
            
            # Métriques globales
            overall = test.get('overall_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Success Rate", f"{overall.get('success_rate', 0):.1%}")
            with col2:
                st.metric("Latence Moy", f"{overall.get('average_latency_ms', 0):.1f}ms")
            with col3:
                st.metric("Safety Score", f"{overall.get('safety_score', 0):.1%}")
            with col4:
                compliant = "✅" if overall.get('real_time_compliant') else "❌"
                st.metric("Temps Réel", compliant)
            
            # Résultats par scénario
            st.subheader("📊 Résultats par Scénario")
            
            scenario_results = test.get('scenario_results', [])
            if scenario_results:
                df = pd.DataFrame(scenario_results)
                
                fig = px.bar(df, x='scenario', y='success_rate', 
                           color='success_rate', color_continuous_scale='RdYlGn')
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
                # Tableau détaillé
                st.dataframe(df, use_container_width=True)

def show_marketplace():
    st.header("🛒 Marketplace de Modules")
    
    tab1, tab2 = st.tabs(["🔍 Explorer", "📤 Publier"])
    
    with tab1:
        st.subheader("🔍 Filtres")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            module_type_filter = st.selectbox("Type", ["Tous"] + [
                "perception", "localization", "planning", "control"
            ])
        
        with col2:
            min_accuracy = st.slider("Précision Min", 0.0, 1.0, 0.8)
        
        with col3:
            max_price = st.number_input("Prix Max ($)", 0.0, 10000.0, 1000.0)
        
        if st.button("🔍 Rechercher"):
            params = {}
            if module_type_filter != "Tous":
                params['module_type'] = module_type_filter
            params['min_accuracy'] = min_accuracy
            params['max_price'] = max_price
            
            listings = api_call("GET", "/api/v1/marketplace/listings", params=params)
            
            if "error" not in listings:
                st.write(f"**{listings.get('total', 0)} modules trouvés**")
                
                for listing in listings.get('listings', []):
                    with st.expander(f"🔧 {listing.get('module_name', 'N/A')} - ${listing.get('price', 0):.2f}"):
                        st.write(f"**Type:** {listing.get('module_type', 'N/A')}")
                        st.write(f"**Description:** {listing.get('description', 'N/A')}")
                        st.write(f"**Licence:** {listing.get('license_type', 'N/A')}")
                        
                        metrics = listing.get('metrics', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                        with col2:
                            st.metric("Latence", f"{metrics.get('latency_ms', 0):.1f}ms")
                        with col3:
                            st.metric("Safety", f"{metrics.get('safety_score', 0):.1%}")
                        
                        if st.button("💳 Acheter", key=listing['listing_id']):
                            st.success("✅ Module ajouté à votre bibliothèque!")
    
    with tab2:
        st.subheader("📤 Publier un Module")
        
        with st.form("submit_form"):
            module_id = st.text_input("Module ID*")
            price = st.number_input("Prix ($)", 0.0, 10000.0, 99.0)
            license_type = st.selectbox("Type de Licence", 
                ["open_source", "commercial", "academic"])
            doc_url = st.text_input("URL Documentation (optionnel)")
            
            if st.form_submit_button("📤 Soumettre"):
                if module_id:
                    result = api_call("POST", "/api/v1/marketplace/submit", json={
                        "module_id": module_id,
                        "price": price,
                        "license_type": license_type,
                        "documentation_url": doc_url if doc_url else None
                    })
                    
                    if result.get("success"):
                        st.success("✅ Module soumis pour review!")
                        st.json(result['listing'])

def show_quantum_projects():
    st.header("⚛️ Projets Quantiques")
    
    st.info("💡 Développez n'importe quel projet avec l'informatique quantique")
    
    with st.form("quantum_project_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom du Projet*")
            
            project_type = st.selectbox("Type de Projet*", [
                "ai_model", "ai_agent", "mobile_app", "web_app",
                "cloud_service", "iot_device", "quantum_algorithm"
            ], format_func=lambda x: x.replace("_", " ").title())
            
            description = st.text_area("Description*")
        
        with col2:
            num_qubits = st.slider("Nombre de Qubits", 2, 128, 16)
            use_entanglement = st.checkbox("Utiliser l'Intrication", value=True)
            
            st.info(f"""
            **Puissance:**
            - États: **{2**num_qubits:,}**
            - Speedup: **{2**(num_qubits/4):.0f}x**
            """)
        
        if st.form_submit_button("🚀 Créer le Projet", use_container_width=True):
            if name and description:
                result = api_call("POST", "/api/v1/quantum-project/create", json={
                    "project_name": name,
                    "project_type": project_type,
                    "description": description,
                    "num_qubits": num_qubits,
                    "use_entanglement": use_entanglement
                })
                
                if result.get("success"):
                    st.success(f"✅ Projet créé: {result['project']['project_id']}")
                    st.json(result['project'])

def show_learning():
    st.header("📚 Plateforme d'Apprentissage")
    
    courses = api_call("GET", "/api/v1/learning/courses")
    
    if "courses" in courses:
        st.subheader("🎓 Cours Disponibles")
        
        for course in courses['courses']:
            with st.expander(f"📖 {course['title']} - {course['total_hours']}h"):
                st.write(f"**Modules:** {course['num_modules']}")
                
                if st.button("🚀 Commencer", key=course['course_id']):
                    st.session_state['active_course'] = course['course_id']
                    
                    # Charger détails
                    details = api_call("GET", f"/api/v1/learning/course/{course['course_id']}")
                    
                    if "modules" in details:
                        for module in details['modules']:
                            st.markdown(f"### {module['title']}")
                            st.write(f"**Durée:** {module['duration_hours']}h")
                            
                            for i, lesson in enumerate(module['lessons'], 1):
                                st.write(f"{i}. {lesson}")

if __name__ == "__main__":
    if 'active_module_id' not in st.session_state:
        st.session_state['active_module_id'] = ''
    if 'last_dataset_id' not in st.session_state:
        st.session_state['last_dataset_id'] = ''
    if 'last_model_id' not in st.session_state:
        st.session_state['last_model_id'] = ''
    if 'last_test_id' not in st.session_state:
        st.session_state['last_test_id'] = ''
    
    main()