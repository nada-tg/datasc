"""
entanglement_frontend.py - Interface Streamlit Complète pour Quantum Entanglement Engine

Installation:
pip install streamlit requests plotly pandas numpy

Lancement:
streamlit run intrication_quantique_app.py
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
API_BASE_URL = "http://localhost:8030"

# Configuration de la page
st.set_page_config(
    page_title="Quantum Entanglement Engine",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Ultra-Futuriste
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .entangle-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #ff006e, #8338ec, #3a86ff, #06ffa5);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .quantum-box {
        background: linear-gradient(135deg, rgba(138, 43, 226, 0.1), rgba(0, 191, 255, 0.1));
        border: 2px solid rgba(138, 43, 226, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(138, 43, 226, 0.3);
    }
    
    .entangled-card {
        background: linear-gradient(135deg, #ff006e 0%, #8338ec 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 40px rgba(255, 0, 110, 0.4);
        margin: 1rem 0;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stats-mega {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #06ffa5, #3a86ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions API
def api_call(method, endpoint, **kwargs):
    """Fonction générique pour appels API"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Interface principale
def main():
    st.markdown('<div class="entangle-header">🔗 QUANTUM ENTANGLEMENT ENGINE</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/8338ec/ffffff?text=Entanglement", use_container_width=True)
        st.title("🌌 Navigation")
        
        page = st.radio("Navigation", [
            "🏠 Dashboard",
            "🔗 Créer Intrication", 
            "📊 Analyser Phases",
            "🧠 IA Intriquée",
            "☁️ Cloud Quantique",
            "📡 Téléportation",
            "🌐 Réseaux Quantiques",
            "🔄 Cycle de Vie",
            "📚 Protocoles"
        ])
        
        st.markdown("---")
        st.markdown("### ⚡ API Status")
        
        try:
            health = requests.get(f"{API_BASE_URL}/health").json()
            st.success("✅ Connected")
            st.metric("Intrications", health.get('entanglements', 0))
            st.metric("Modèles IA", health.get('ai_models', 0))
            st.metric("Cloud", health.get('cloud_services', 0))
        except:
            st.error("❌ API Offline")
    
    # Routing des pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔗 Créer Intrication":
        show_create_entanglement()
    elif page == "📊 Analyser Phases":
        show_analyze_phases()
    elif page == "🧠 IA Intriquée":
        show_entangled_ai()
    elif page == "☁️ Cloud Quantique":
        show_quantum_cloud()
    elif page == "📡 Téléportation":
        show_teleportation()
    elif page == "🌐 Réseaux Quantiques":
        show_quantum_networks()
    elif page == "🔄 Cycle de Vie":
        show_lifecycle()
    elif page == "📚 Protocoles":
        show_protocols()

def show_dashboard():
    """Dashboard principal"""
    st.header("🌌 Tableau de Bord Quantique")
    
    stats = api_call("GET", "/api/v1/stats/global")
    
    if "error" not in stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="entangled-card">
                <h3 style="text-align:center;">🔗 Intrications</h3>
                <p class="stats-mega">{stats.get("entanglements", {}).get("total", 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="entangled-card" style="background: linear-gradient(135deg, #3a86ff 0%, #06ffa5 100%);">
                <h3 style="text-align:center;">🧠 Modèles IA</h3>
                <p class="stats-mega">{stats.get("ai_models", {}).get("total", 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="entangled-card" style="background: linear-gradient(135deg, #06ffa5 0%, #8338ec 100%);">
                <h3 style="text-align:center;">☁️ Services</h3>
                <p class="stats-mega">{stats.get("cloud_services", {}).get("total", 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="entangled-card" style="background: linear-gradient(135deg, #ff006e 0%, #3a86ff 100%);">
                <h3 style="text-align:center;">🌐 Réseaux</h3>
                <p class="stats-mega">{stats.get("networks", {}).get("total", 0)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Avantages
        st.subheader("✨ Puissance de l'Intrication")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="quantum-box">
                <h3>⚡ Vitesse Instantanée</h3>
                <p>Corrélation plus rapide que la lumière</p>
                <p><strong>Avantage: ∞</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="quantum-box">
                <h3>🔒 Sécurité Absolue</h3>
                <p>Inviolable théoriquement</p>
                <p><strong>QKD garanti</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="quantum-box">
                <h3>💪 Puissance Exponentielle</h3>
                <p>N qubits = 2^N états</p>
                <p><strong>Parallélisme massif</strong></p>
            </div>
            """, unsafe_allow_html=True)

def show_create_entanglement():
    """Création d'intrication"""
    st.header("🔗 Créer une Intrication Quantique")
    
    with st.form("entanglement_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom*", placeholder="Mon Intrication")
            entanglement_type = st.selectbox("Type*", 
                ["bell_state", "ghz_state", "w_state", "cluster_state"])
            num_qubits = st.slider("Qubits", 2, 100, 8)
        
        with col2:
            fidelity = st.slider("Fidélité", 0.5, 1.0, 0.95, 0.01)
            monitoring = st.checkbox("Monitoring", value=True)
            st.info(f"États: **{2**num_qubits:,}**")
        
        if st.form_submit_button("🚀 Lancer", use_container_width=True):
            if name:
                result = api_call("POST", "/api/v1/entanglement/create", json={
                    "name": name,
                    "entanglement_type": entanglement_type,
                    "num_qubits": num_qubits,
                    "fidelity_target": fidelity,
                    "enable_monitoring": monitoring
                })
                
                if result.get("success"):
                    st.success(f"✅ ID: {result['entanglement_id']}")
                    st.session_state['last_ent_id'] = result['entanglement_id']
                    
                    progress = st.progress(0)
                    for i in range(100):
                        progress.progress(i + 1)
                        time.sleep(0.02)
                    st.balloons()

def show_analyze_phases():
    """Analyse des phases"""
    st.header("📊 Analyser les Phases")
    
    ent_id = st.text_input("ID", value=st.session_state.get('last_ent_id', ''))
    
    if st.button("🔍 Analyser"):
        if not ent_id:
            st.warning("⚠️ Veuillez entrer un ID d'intrication")
            return
            
        data = api_call("GET", f"/api/v1/entanglement/{ent_id}")
        
        if "error" in data:
            st.error(f"❌ Erreur: {data['error']}")
            return
            
        st.success(f"✅ {data.get('name', 'Intrication')}")
        
        metrics = data.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Fidélité", f"{metrics.get('final_fidelity', 0):.2%}")
        with col2:
            st.metric("Force", f"{metrics.get('entanglement_strength', 0):.2%}")
        with col3:
            st.metric("Cohérence", f"{metrics.get('coherence_time_us', 0):.0f}µs")
        with col4:
            st.metric("Bell", "✅" if metrics.get('bell_inequality_violation') else "❌")
        
        st.markdown("---")
        st.subheader("🔄 Phases")
        
        for i, phase in enumerate(data.get("phases", [])):
            with st.expander(f"Phase {i+1}: {phase.get('phase', 'N/A')}"):
                st.json(phase)

def show_entangled_ai():
    """Modèles IA"""
    st.header("🧠 IA avec Intrication")
    
    tab1, tab2 = st.tabs(["Entraîner", "Résultats"])
    
    with tab1:
        with st.form("ai_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nom*")
                arch = st.selectbox("Architecture", 
                    ["quantum_cnn", "quantum_rnn", "quantum_transformer", "quantum_gan"])
                data_id = st.text_input("Data ID*")
            
            with col2:
                qubits = st.slider("Qubits", 4, 64, 16)
                epochs = st.slider("Époques", 1, 1000, 100)
                speedup = 2 ** (qubits / 4)
                st.info(f"Speedup: **{speedup:.0f}x**")
            
            if st.form_submit_button("🚀 Entraîner"):
                if name and data_id:
                    result = api_call("POST", "/api/v1/ai/train-entangled", json={
                        "model_name": name,
                        "architecture": arch,
                        "num_entangled_qubits": qubits,
                        "training_data_id": data_id,
                        "epochs": epochs,
                        "use_entanglement_acceleration": True
                    })
                    
                    if result.get("success"):
                        st.success(f"✅ Model ID: {result.get('model_id', 'N/A')}")
                        st.session_state['last_model_id'] = result.get('model_id', '')
                    else:
                        st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                else:
                    st.warning("⚠️ Veuillez remplir tous les champs obligatoires")
    
    with tab2:
        model_id = st.text_input("Model ID", value=st.session_state.get('last_model_id', ''))
        
        if st.button("Charger"):
            if not model_id:
                st.warning("⚠️ Veuillez entrer un Model ID")
                return
                
            model = api_call("GET", f"/api/v1/ai/model/{model_id}")
            
            if "error" in model:
                st.error(f"❌ Erreur: {model['error']}")
                return
                
            st.success(f"✅ {model.get('model_name', 'Modèle IA')}")
            
            perf = model.get("performance", {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Speedup", f"{perf.get('speedup_factor', 1):.1f}x")
            with col2:
                st.metric("Temps Q", f"{perf.get('quantum_time_actual_s', 0):.1f}s")
            with col3:
                st.metric("Économie", f"{perf.get('time_saved_percent', 0):.1f}%")
            
            if 'training_history' in model and model['training_history']:
                df = pd.DataFrame(model['training_history'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['epoch'], y=df['accuracy'], name='Accuracy', line=dict(color='#06ffa5')))
                fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'], name='Loss', line=dict(color='#ff006e')))
                fig.update_layout(template="plotly_dark", title="Historique d'Entraînement")
                st.plotly_chart(fig, use_container_width=True)

def show_quantum_cloud():
    """Cloud quantique"""
    st.header("☁️ Cloud Quantique")
    
    tab1, tab2 = st.tabs(["Créer", "Services"])
    
    with tab1:
        with st.form("cloud_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nom*")
                stype = st.selectbox("Type", 
                    ["quantum_compute", "quantum_storage", "quantum_ai_training"])
                qubits = st.slider("Qubits", 2, 256, 32)
            
            with col2:
                ent = st.checkbox("Intrication", value=True)
                scale = st.checkbox("Auto-scaling", value=True)
                pairs = st.slider("Max Paires", 100, 10000, 1000)
            
            if st.form_submit_button("Créer"):
                if name:
                    result = api_call("POST", "/api/v1/cloud/create-service", json={
                        "service_name": name,
                        "service_type": stype,
                        "num_qubits": qubits,
                        "entanglement_enabled": ent,
                        "auto_scaling": scale,
                        "max_entangled_pairs": pairs
                    })
                    
                    if result.get("success"):
                        st.success(f"✅ Service créé!")
                        if 'service' in result:
                            st.json(result['service'])
                    else:
                        st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
                else:
                    st.warning("⚠️ Veuillez entrer un nom")
    
    with tab2:
        services = api_call("GET", "/api/v1/cloud/services")
        if "error" in services:
            st.error(f"❌ Erreur: {services['error']}")
        elif "services" in services:
            st.write(f"**{services.get('total', 0)} services**")
            for svc in services['services']:
                with st.expander(f"☁️ {svc.get('service_name', 'Service')}"):
                    st.json(svc)
        else:
            st.info("Aucun service disponible")

def show_teleportation():
    """Téléportation"""
    st.header("📡 Téléportation Quantique")
    
    st.markdown("""
    <div class="quantum-box">
        <h3>🌟 Téléportation d'État Quantique</h3>
        <p>Transférez instantanément l'état d'un qubit via intrication!</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("teleport_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            source = st.text_input("Qubit Source*", "qubit_0")
            dest = st.text_input("Destination*", "node_remote")
        
        with col2:
            pair_id = st.text_input("Paire Intriquée*", "entangled_pair_1")
            message = st.text_input("Message", "Hello Quantum World!")
        
        if st.form_submit_button("📡 Téléporter"):
            if source and dest and pair_id:
                result = api_call("POST", "/api/v1/teleportation/teleport", json={
                    "source_qubit_id": source,
                    "destination_id": dest,
                    "entangled_pair_id": pair_id,
                    "message": message
                })
                
                if result.get("success"):
                    st.markdown('<div class="teleport-effect">', unsafe_allow_html=True)
                    st.success("✅ Téléportation réussie!")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    teleport = result.get('teleportation', {})
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Fidélité", f"{teleport.get('teleportation_fidelity', 0):.2%}")
                    with col2:
                        st.metric("Temps", f"{teleport.get('total_time_us', 0):.1f}µs")
                    with col3:
                        st.metric("Succès", "✅" if teleport.get('success') else "❌")
                    
                    st.json(teleport)
                else:
                    st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
            else:
                st.warning("⚠️ Veuillez remplir tous les champs")

def show_quantum_networks():
    """Réseaux quantiques"""
    st.header("🌐 Réseaux Quantiques")
    
    with st.form("network_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom*", "Mon Réseau Quantique")
            nodes = st.slider("Nœuds", 2, 100, 10)
        
        with col2:
            topology = st.selectbox("Topologie", ["star", "mesh", "ring", "tree"])
            distrib = st.selectbox("Distribution", 
                ["centralized", "distributed", "hierarchical"])
        
        if st.form_submit_button("🌐 Créer Réseau"):
            if name:
                result = api_call("POST", "/api/v1/network/create", json={
                    "network_name": name,
                    "num_nodes": nodes,
                    "topology": topology,
                    "entanglement_distribution": distrib
                })
                
                if result.get("success"):
                    st.success("✅ Réseau créé!")
                    network = result.get('network', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Nœuds", network.get('num_nodes', 0))
                    with col2:
                        st.metric("Connexions", len(network.get('connections', [])))
                    with col3:
                        st.metric("Paires", network.get('total_entangled_pairs', 0))
                    
                    st.json(network)
                else:
                    st.error(f"❌ Erreur: {result.get('error', 'Erreur inconnue')}")
            else:
                st.warning("⚠️ Veuillez entrer un nom")

def show_lifecycle():
    """Cycle de vie"""
    st.header("🔄 Cycle de Vie")
    
    lifecycle_id = st.text_input("Lifecycle ID")
    
    if st.button("Charger"):
        if not lifecycle_id:
            st.warning("⚠️ Veuillez entrer un Lifecycle ID")
            return
            
        lc = api_call("GET", f"/api/v1/lifecycle/{lifecycle_id}")
        
        if "error" in lc:
            st.error(f"❌ Erreur: {lc['error']}")
            return
            
        st.success(f"✅ {lc.get('resource_type', 'Resource')}")
        
        st.metric("Stage Actuel", lc.get('current_stage', 'N/A'))
        st.metric("Score Santé", f"{lc.get('quantum_health_score', 0):.2%}")
        
        st.subheader("Jalons")
        for milestone in lc.get('milestones', []):
            status = "✅" if milestone.get('status') == 'completed' else "⏳"
            st.write(f"{status} **{milestone.get('milestone', 'N/A')}** - {milestone.get('stage', 'N/A')}")

def show_protocols():
    """Protocoles quantiques"""
    st.header("📚 Protocoles Quantiques")
    
    protocols = api_call("GET", "/api/v1/protocols/list")
    
    if "error" in protocols:
        st.error(f"❌ Erreur: {protocols['error']}")
        return
    
    if "protocols" in protocols and protocols['protocols']:
        st.write(f"**{protocols.get('total_protocols', 0)} protocoles disponibles**")
        
        for proto in protocols['protocols']:
            with st.expander(f"🔬 {proto.get('protocol', 'Protocole')}"):
                st.write(f"**Description:** {proto.get('description', 'N/A')}")
                st.write(f"**Applications:** {', '.join(proto.get('applications', []))}")
                st.write(f"**Requis:** {', '.join(proto.get('requirements', []))}")
                
                if 'fidelity' in proto:
                    st.write(f"**Fidélité:** {proto['fidelity']}")
                if 'security' in proto:
                    st.write(f"**Sécurité:** {proto['security']}")
                if 'capacity' in proto:
                    st.write(f"**Capacité:** {proto['capacity']}")
                if 'overhead' in proto:
                    st.write(f"**Overhead:** {proto['overhead']}")
                if 'scalability' in proto:
                    st.write(f"**Scalabilité:** {proto['scalability']}")
    else:
        st.info("Aucun protocole disponible")

if __name__ == "__main__":
    # Initialisation session state
    if 'last_ent_id' not in st.session_state:
        st.session_state['last_ent_id'] = ''
    if 'last_model_id' not in st.session_state:
        st.session_state['last_model_id'] = ''
    
    main()