""""
Frontend Streamlit - Moteur IA et Quantique de Bio-Computing
Interface utilisateur complète pour ordinateurs biologiques et quantiques
streamlit run ai_quantique_biocomputing_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Bio-Quantum Computing Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de l'API
API_URL = "http://localhost:8007"

# Initialisation du state
if 'computers' not in st.session_state:
    st.session_state.computers = []
if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'bio_ai_models' not in st.session_state:
    st.session_state.bio_ai_models = []
if 'simulations' not in st.session_state:
    st.session_state.simulations = []
if 'current_sim_id' not in st.session_state:
    st.session_state.current_sim_id = ''
if 'current_exp_id' not in st.session_state:
    st.session_state.current_exp_id = ''
if 'current_network_id' not in st.session_state:
    st.session_state.current_network_id = ''

# ==================== FONCTIONS UTILITAIRES ====================

def api_request(endpoint, method="GET", data=None):
    """Effectue une requête API"""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion: {e}")
        return None

# ==================== SIDEBAR NAVIGATION ====================

st.sidebar.title("🧬 Bio-Quantum Engine")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Tableau de Bord",
     "💻 Ordinateurs Bio-Quantiques",
     "🤖 Agents IA Bio-Quantiques",
     "🧠 Modèles d'IA Biologiques",
     "⚛️ Calculs Bio-Quantiques",
     "🧪 Simulations Avancées",
     "🔬 Expériences",
     "🧬 Biologie Synthétique",
     "🌐 Réseaux Bio-Quantiques",
     "🧬 Évolution & Optimisation",
     "📊 Analytics & Rapports"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Version:** 1.0.0\n**Status:** Opérationnel ✅")

# ==================== TABLEAU DE BORD ====================

if menu == "🏠 Tableau de Bord":
    st.title("🏠 Tableau de Bord Bio-Quantum Computing")
    st.markdown("### Vue d'ensemble de l'écosystème bio-quantique")
    
    # Récupérer les analytics globaux
    analytics = api_request("/api/analytics/global")
    
    if analytics:
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Ordinateurs Bio-Quantiques", analytics['total_bio_computers'])
        with col2:
            st.metric("Modèles IA Biologiques", analytics['total_bio_ai_models'])
        with col3:
            st.metric("Agents Intelligents", analytics['total_agents'])
        with col4:
            st.metric("Simulations", analytics['total_simulations'])
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Efficacité Biologique", f"{analytics['biological_efficiency_avg']:.2%}")
        with col2:
            st.metric("Avantage Quantique", f"{analytics['quantum_advantage_avg']:.1f}x")
        with col3:
            st.metric("Synergie Bio-Quantique", f"{analytics['bio_quantum_synergy_avg']:.2f}x")
        
        st.markdown("---")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Efficacité Biologique")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=analytics['biological_efficiency_avg'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Efficacité (%)"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "lightgreen"},
                        {'range': [90, 100], 'color': "lime"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚛️ Avantage Quantique")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=analytics['quantum_advantage_avg'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Speedup (x)"},
                gauge={
                    'axis': {'range': [None, 1000]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgray"},
                        {'range': [100, 500], 'color': "lightblue"},
                        {'range': [500, 1000], 'color': "cyan"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # État du système
        st.subheader("🎯 État des Technologies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Réseaux Bio-Quantiques", analytics['total_networks'])
            st.metric("Expériences", analytics['total_experiments'])
        with col2:
            st.metric("Évolutions", analytics['total_evolutions'])
            st.info("**Status:** " + analytics['system_status'].upper())

# ==================== ORDINATEURS BIO-QUANTIQUES ====================

elif menu == "💻 Ordinateurs Bio-Quantiques":
    st.title("💻 Ordinateurs Biologiques et Quantiques")
    
    tab1, tab2, tab3 = st.tabs(["➕ Créer", "📋 Ordinateurs Existants", "⚡ Exécuter des Tâches"])
    
    with tab1:
        st.subheader("Créer un Nouvel Ordinateur")
        
        with st.form("computer_form"):
            computer_type = st.selectbox(
                "Type d'Ordinateur",
                ["Biologique", "Quantique", "Bio-Quantique", "ADN", "Hybride Complet"]
            )
            
            name = st.text_input("Nom de l'Ordinateur")
            
            if computer_type == "Biologique":
                st.markdown("**Configuration Biologique**")
                neurons = st.slider("Nombre de Neurones", 10000, 1000000, 100000)
                synapses = st.slider("Nombre de Synapses", neurons*10, neurons*100, neurons*50)
                organoids = st.slider("Nombre d'Organoïdes", 1, 20, 5)
                substrate = st.selectbox("Substrat", ["neural_tissue", "stem_cells", "neurons_3d"])
                
                config = {
                    'computer_type': 'biological',
                    'name': name,
                    'neurons': neurons,
                    'synapses': synapses,
                    'organoids': organoids,
                    'substrate': substrate
                }
            
            elif computer_type == "Quantique":
                st.markdown("**Configuration Quantique**")
                qubits = st.slider("Nombre de Qubits", 10, 1000, 100)
                qubit_type = st.selectbox("Type de Qubit", ["superconducting", "trapped_ion", "photonic", "topological"])
                connectivity = st.selectbox("Connectivité", ["all_to_all", "linear", "grid"])
                
                config = {
                    'computer_type': 'quantum',
                    'name': name,
                    'qubits': qubits,
                    'qubit_type': qubit_type,
                    'connectivity': connectivity
                }
            
            elif computer_type == "Bio-Quantique":
                st.markdown("**Configuration Bio-Quantique Hybride**")
                bio_qubits = st.slider("Bio-Qubits", 10, 500, 50)
                quantum_qubits = st.slider("Qubits Quantiques", 10, 500, 50)
                
                config = {
                    'computer_type': 'bio_quantum',
                    'name': name,
                    'bio_qubits': bio_qubits,
                    'quantum_qubits': quantum_qubits
                }
            
            elif computer_type == "ADN":
                st.markdown("**Configuration ADN Computing**")
                dna_strands = st.slider("Brins d'ADN", 100000, 10000000, 1000000)
                sequence_length = st.slider("Longueur de Séquence", 100, 10000, 1000)
                volume = st.slider("Volume de Réaction (ml)", 0.1, 10.0, 1.0)
                
                config = {
                    'computer_type': 'dna_computer',
                    'name': name,
                    'dna_strands': dna_strands,
                    'sequence_length': sequence_length,
                    'volume': volume
                }
            
            else:  # Hybride Complet
                st.markdown("**Configuration Hybride Complète**")
                cores = st.slider("Cœurs Classiques", 16, 256, 128)
                qubits = st.slider("Qubits", 10, 200, 50)
                neurons = st.slider("Neurones Biologiques", 10000, 500000, 50000)
                
                config = {
                    'computer_type': 'full_hybrid',
                    'name': name,
                    'cores': cores,
                    'qubits': qubits,
                    'neurons': neurons
                }
            
            submitted = st.form_submit_button("🚀 Créer l'Ordinateur")
            
            if submitted and name:
                result = api_request("/api/computer/create", method="POST", data=config)
                
                if result:
                    st.success(f"✅ Ordinateur '{name}' créé avec succès!")
                    st.json(result)
    
    with tab2:
        st.subheader("Ordinateurs Existants")
        computers = api_request("/api/computer/list")
        
        if computers:
            for computer in computers:
                with st.expander(f"💻 {computer['name']} ({computer['type'].upper()}) - {computer.get('status', 'active')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {computer['type']}")
                        st.write(f"**ID:** {computer['computer_id'][:8]}")
                        
                        st.write("**Spécifications:**")
                        specs = computer.get('specifications', {})
                        for key, value in specs.items():
                            st.write(f"- **{key}:** {value}")
                    
                    with col2:
                        if 'computational_power' in computer:
                            st.write("**Puissance Computationnelle:**")
                            power = computer['computational_power']
                            for key, value in power.items():
                                st.write(f"- **{key}:** {value}")
                        
                        if st.button(f"📊 Analyser", key=f"analyze_{computer['computer_id']}"):
                            analytics = api_request(f"/api/analytics/computer/{computer['computer_id']}")
                            if analytics:
                                st.write(f"**Performance Moyenne:** {analytics['average_performance']:.2%}")
                                st.write(f"**Efficacité Énergétique:** {analytics['energy_efficiency']:.2%}")
                                if analytics.get('biological_health'):
                                    st.write(f"**Santé Biologique:** {analytics['biological_health']:.2%}")
        else:
            st.info("Aucun ordinateur créé pour le moment.")
    
    with tab3:
        st.subheader("Exécuter des Tâches")
        
        computers = api_request("/api/computer/list") or []
        
        if computers:
            computer_names = {f"{c['name']} ({c['type']})": c['computer_id'] for c in computers}
            selected_computer = st.selectbox("Sélectionner un ordinateur", list(computer_names.keys()))
            
            task_type = st.selectbox("Type de Tâche", 
                ["computation", "optimization", "simulation", "learning", "pattern_recognition"])
            
            if st.button("⚡ Exécuter"):
                with st.spinner("🔄 Exécution en cours..."):
                    computer_id = computer_names[selected_computer]
                    
                    result = api_request(
                        f"/api/computer/{computer_id}/execute",
                        method="POST",
                        data={'type': task_type}
                    )
                    
                    if result:
                        st.success("✅ Tâche exécutée!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Temps d'Exécution", f"{result['execution_time_ms']:.2f} ms")
                            st.metric("Qualité du Résultat", f"{result['result_quality']:.2%}")
                        with col2:
                            st.metric("Énergie Consommée", f"{result['energy_consumed_uj']:.3f} µJ")
                            st.metric("Opérations Parallèles", f"{result['parallel_operations']:,}")
                        with col3:
                            if result['quantum_advantage'] > 1:
                                st.metric("Avantage Quantique", f"{result['quantum_advantage']:.1f}x")
                            if result.get('biological_efficiency'):
                                st.metric("Efficacité Biologique", f"{result['biological_efficiency']:.2%}")
        else:
            st.info("Créez d'abord un ordinateur.")

# ==================== AGENTS IA BIO-QUANTIQUES ====================

elif menu == "🤖 Agents IA Bio-Quantiques":
    st.title("🤖 Agents IA Bio-Quantiques")
    
    tab1, tab2, tab3, tab4 = st.tabs(["➕ Créer", "📋 Agents", "🎓 Entraîner", "🧬 Faire Évoluer"])
    
    with tab1:
        st.subheader("Créer un Agent Bio-Quantique")
        
        with st.form("agent_form"):
            name = st.text_input("Nom de l'Agent")
            
            agent_type = st.selectbox("Type d'Agent", 
                ["autonomous", "collaborative", "learning", "adaptive"])
            
            intelligence_level = st.selectbox("Niveau d'Intelligence",
                ["basic", "intermediate", "advanced", "superintelligent"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                neurons = st.slider("Neurones Biologiques", 1000, 100000, 10000)
                qubits = st.slider("Qubits Quantiques", 5, 100, 20)
            
            with col2:
                processors = st.slider("Processeurs Classiques", 1, 16, 4)
                layers = st.slider("Couches Hybrides", 2, 20, 5)
            
            bio_quantum_ratio = st.slider("Ratio Bio/Quantique", 0.0, 1.0, 0.5, 0.1)
            consciousness = st.checkbox("Simulation de Conscience")
            
            submitted = st.form_submit_button("🚀 Créer l'Agent")
            
            if submitted and name:
                config = {
                    'name': name,
                    'agent_type': agent_type,
                    'intelligence_level': intelligence_level,
                    'neurons': neurons,
                    'qubits': qubits,
                    'processors': processors,
                    'layers': layers,
                    'bio_quantum_ratio': bio_quantum_ratio,
                    'consciousness': consciousness
                }
                
                result = api_request("/api/agent/create", method="POST", data=config)
                
                if result:
                    st.success(f"✅ Agent '{name}' créé avec succès!")
                    st.json(result)
    
    with tab2:
        st.subheader("Agents Existants")
        agents = api_request("/api/agent/list")
        
        if agents:
            for agent in agents:
                with st.expander(f"🤖 {agent['name']} - {agent['intelligence_level'].upper()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Type:** {agent['agent_type']}")
                        st.write(f"**Niveau Intelligence:** {agent['intelligence_level']}")
                        st.write(f"**Ratio Bio/Quantique:** {agent['bio_quantum_ratio']:.2f}")
                        
                        st.write("**Architecture:**")
                        for key, value in agent['architecture'].items():
                            st.write(f"- {key}: {value}")
                    
                    with col2:
                        st.write("**Capacités:**")
                        for cap, enabled in agent['capabilities'].items():
                            status = "✅" if enabled else "❌"
                            st.write(f"{status} {cap}")
                        
                        st.write("**Performance:**")
                        perf = agent['performance']
                        st.write(f"- Vitesse Décision: {perf['decision_speed_ms']:.2f} ms")
                        st.write(f"- Précision: {perf['accuracy']:.2%}")
                        st.write(f"- Efficacité: {perf['energy_efficiency']:.2%}")
        else:
            st.info("Aucun agent créé pour le moment.")
    
    with tab3:
        st.subheader("Entraîner un Agent")
        
        agents = api_request("/api/agent/list") or []
        
        if agents:
            agent_names = {a['name']: a['agent_id'] for a in agents}
            selected_agent = st.selectbox("Sélectionner un agent", list(agent_names.keys()))
            
            with st.form("training_form"):
                episodes = st.slider("Nombre d'Épisodes", 100, 10000, 1000)
                environment = st.selectbox("Environnement", 
                    ["simulated", "real_world", "hybrid", "adversarial"])
                
                submitted = st.form_submit_button("🎓 Entraîner")
                
                if submitted:
                    with st.spinner("🤖 Entraînement en cours..."):
                        agent_id = agent_names[selected_agent]
                        
                        result = api_request(
                            f"/api/agent/{agent_id}/train",
                            method="POST",
                            data={'episodes': episodes, 'environment': environment}
                        )
                        
                        if result:
                            st.success("✅ Entraînement terminé!")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Épisodes", result['training_episodes'])
                                st.metric("Récompense Finale", f"{result['final_reward']:.1f}")
                            with col2:
                                st.metric("Adaptations Biologiques", result['biological_adaptations'])
                                st.metric("Optimisations Quantiques", result['quantum_optimizations'])
                            with col3:
                                st.metric("Amélioration", f"+{result['performance_improvement']:.1f}%")
                                st.metric("Temps", f"{result['training_time_hours']:.2f}h")
                            
                            # Courbe d'apprentissage
                            if result.get('reward_progression'):
                                st.subheader("Courbe d'Apprentissage")
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    y=result['reward_progression'],
                                    mode='lines+markers',
                                    name='Récompense',
                                    line=dict(color='blue')
                                ))
                                
                                fig.update_layout(
                                    xaxis_title="Échantillon d'Épisodes",
                                    yaxis_title="Récompense",
                                    hovermode='x'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Créez d'abord un agent.")
    
    with tab4:
        st.subheader("Évolution Biologique")
        
        agents = api_request("/api/agent/list") or []
        
        if agents:
            agent_names = {a['name']: a['agent_id'] for a in agents}
            selected_agent = st.selectbox("Sélectionner un agent", list(agent_names.keys()), key="evolve_agent")
            
            generations = st.slider("Nombre de Générations", 10, 5000, 100)
            
            if st.button("🧬 Lancer l'Évolution"):
                with st.spinner("🧬 Évolution en cours..."):
                    agent_id = agent_names[selected_agent]
                    
                    result = api_request(
                        f"/api/agent/{agent_id}/evolve",
                        method="POST",
                        data={'generations': generations}
                    )
                    
                    if result:
                        st.success("✅ Évolution terminée!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Générations", result['generations'])
                            st.metric("Mutations Totales", result['total_mutations'])
                        with col2:
                            st.metric("Amélioration Fitness", f"+{result['fitness_improvement_percent']:.1f}%")
                            st.metric("Score d'Adaptation", f"{result['adaptation_score']:.2%}")
                        with col3:
                            st.metric("Taux de Survie", f"{result['survival_rate']:.2%}")
                            st.metric("Années Simulées", f"{result['evolution_time_simulated_years']:.1f}")
                        
                        if result.get('emergent_capabilities'):
                            st.markdown("---")
                            st.subheader("🌟 Capacités Émergentes")
                            for cap in result['emergent_capabilities']:
                                st.write(f"✨ {cap}")
        else:
            st.info("Créez d'abord un agent.")

# ==================== MODÈLES D'IA BIOLOGIQUES ====================

elif menu == "🧠 Modèles d'IA Biologiques":
    st.title("🧠 Modèles d'Intelligence Artificielle Biologique")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Organoïdes Neuraux", "🧬 Réseaux ADN", "🔬 IA Protéique", "🦠 IA Microbienne"])
    
    with tab1:
        st.subheader("Créer un Modèle Organoïde Neural")
        
        with st.form("organoid_form"):
            neurons = st.slider("Nombre de Neurones", 10000, 1000000, 100000)
            architecture = st.selectbox("Architecture", ["cortical", "hippocampal", "cerebellar", "custom"])
            
            submitted = st.form_submit_button("🧠 Créer")
            
            if submitted:
                with st.spinner("🧠 Création en cours..."):
                    result = api_request(
                        "/api/bioai/neural-organoid/create",
                        method="POST",
                        data={'neurons': neurons, 'architecture': architecture}
                    )
                    
                    if result:
                        st.success("✅ Organoïde neural créé!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**ID:** {result['model_id'][:8]}")
                            st.write(f"**Neurones:** {result['neurons']:,}")
                            st.write(f"**Couches:** {result['layers']}")
                            st.write(f"**Densité Synaptique:** {result['synaptic_density']:,}")
                        
                        with col2:
                            st.write(f"**Plasticité:** {result['biological_plasticity']:.2%}")
                            st.write(f"**Efficacité:** {result['energy_efficiency']:.2%}")
                            st.write(f"**Auto-Réparation:** {'✅' if result['self_repair'] else '❌'}")
                            st.write(f"**Niveau Conscience:** {result['consciousness_level']:.2f}")
    
    with tab2:
        st.subheader("Créer un Réseau de Neurones ADN")
        
        with st.form("dna_nn_form"):
            sequence_length = st.slider("Longueur de Séquence", 1000, 100000, 10000)
            layers = st.slider("Nombre de Couches", 2, 20, 5)
            
            submitted = st.form_submit_button("🧬 Créer")
            
            if submitted:
                with st.spinner("🧬 Création en cours..."):
                    result = api_request(
                        "/api/bioai/dna-network/create",
                        method="POST",
                        data={'sequence_length': sequence_length, 'layers': layers}
                    )
                    
                    if result:
                        st.success("✅ Réseau ADN créé!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Poids Totaux", f"{result['total_weights']:,}")
                            st.metric("Bases Totales", f"{result['total_bases']:,}")
                            st.metric("Densité Stockage", f"{result['storage_density_tb_per_gram']} TB/g")
                        
                        with col2:
                            st.metric("Temps Inférence", f"{result['inference_time_ms']:.2f} ms")
                            st.metric("Précision", f"{result['accuracy']:.2%}")
                            st.metric("Énergie/Inférence", f"{result['energy_per_inference_fj']:.3f} fJ")
    
    with tab3:
        st.subheader("Créer une IA Protéique")
        
        with st.form("protein_ai_form"):
            protein_input = st.text_area("Types de Protéines (un par ligne)", 
                "enzyme_a\nenzyme_b\nreceptor_c")
            interactions = st.slider("Nombre d'Interactions", 10, 1000, 100)
            
            submitted = st.form_submit_button("🔬 Créer")
            
            if submitted:
                protein_types = [p.strip() for p in protein_input.split('\n') if p.strip()]
                
                with st.spinner("🔬 Création en cours..."):
                    result = api_request(
                        "/api/bioai/protein-ai/create",
                        method="POST",
                        data={'protein_types': protein_types, 'interactions': interactions}
                    )
                    
                    if result:
                        st.success("✅ IA protéique créée!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Types de Protéines:** {result['num_protein_types']}")
                            st.write(f"**Complexité Réseau:** {result['network_complexity']:,}")
                            st.write(f"**États Computationnels:** {result['computational_states']:,}")
                        
                        with col2:
                            st.write(f"**Cohérence Quantique:** {'✅' if result['quantum_coherence'] else '❌'}")
                            st.write(f"**Auto-Assemblage:** {'✅' if result['self_assembly'] else '❌'}")
                            st.write(f"**Spécificité:** {result['specificity']:.2%}")
    
    with tab4:
        st.subheader("Créer une IA Microbienne Collective")
        
        with st.form("microbial_ai_form"):
            species_input = st.text_area("Espèces Microbiennes (une par ligne)",
                "e_coli\nb_subtilis\ns_cerevisiae")
            colony_size = st.slider("Taille de la Colonie", 10000, 10000000, 1000000)
            
            submitted = st.form_submit_button("🦠 Créer")
            
            if submitted:
                species = [s.strip() for s in species_input.split('\n') if s.strip()]
                
                with st.spinner("🦠 Création en cours..."):
                    result = api_request(
                        "/api/bioai/microbial-ai/create",
                        method="POST",
                        data={'species': species, 'colony_size': colony_size}
                    )
                    
                    if result:
                        st.success("✅ IA microbienne créée!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Taille Colonie", f"{result['colony_size']:,}")
                            st.metric("Intelligence Collective", f"{result['collective_intelligence_score']:.2%}")
                            st.metric("Molécules Signal", result['signaling_molecules'])
                        
                        with col2:
                            st.metric("Générations Optimisation", result['generations_for_optimization'])
                            st.metric("Résilience", f"{result['resilience']:.2%}")
                            st.metric("Taux d'Adaptation", f"{result['adaptation_rate']:.2%}")

# ==================== CALCULS BIO-QUANTIQUES ====================

elif menu == "⚛️ Calculs Bio-Quantiques":
    st.title("⚛️ Calculs Bio-Quantiques Avancés")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧬 ADN", "🧠 Organoïde", "🔬 Protéines", "🔲 Automates", "⚛️ Hybride", "🌿 Photosynthèse"
    ])
    
    with tab1:
        st.subheader("Calcul par ADN")
        
        with st.form("dna_compute_form"):
            sequence_length = st.slider("Longueur Séquence", 100, 10000, 1000)
            operations = st.multiselect("Opérations", 
                ["sort", "search", "match", "encode", "decode", "parallel_compute"])
            
            submitted = st.form_submit_button("🧬 Calculer")
            
            if submitted and operations:
                with st.spinner("🧬 Calcul ADN en cours..."):
                    result = api_request(
                        "/api/compute/dna",
                        method="POST",
                        data={'sequence_length': sequence_length, 'operations': operations}
                    )
                    
                    if result:
                        st.success("✅ Calcul terminé!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Opérations Parallèles", f"{result['parallel_operations']:,}")
                            st.metric("Temps Calcul", f"{result['computation_time_ms']:.2f} ms")
                        with col2:
                            st.metric("Densité Encodage", f"{result['encoding_density_gb_per_gram']:,} GB/g")
                            st.metric("Efficacité Énergétique", f"{result['energy_efficiency']:.2%}")
                        with col3:
                            st.metric("Taux d'Erreur", f"{result['error_rate']:.3%}")
                            st.metric("Speedup vs Silicium", f"{result['speedup_vs_silicon']:.0f}x")
                        
                        st.markdown("---")
                        st.write(f"**Échantillon Séquence:** `{result['dna_sequence_sample']}`")
                        
                        if result.get('results'):
                            st.subheader("Résultats des Opérations")
                            results_df = pd.DataFrame(result['results'])
                            st.dataframe(results_df, use_container_width=True)
    
    with tab2:
        st.subheader("Calcul par Organoïde Neural")
        
        with st.form("organoid_compute_form"):
            neurons = st.slider("Neurones", 10000, 500000, 50000)
            synapses = st.slider("Synapses", neurons*3, neurons*10, neurons*5)
            learning_task = st.selectbox("Tâche d'Apprentissage",
                ["pattern_recognition", "classification", "prediction", "optimization"])
            
            submitted = st.form_submit_button("🧠 Calculer")
            
            if submitted:
                with st.spinner("🧠 Calcul organoïde en cours..."):
                    result = api_request(
                        "/api/compute/neural-organoid",
                        method="POST",
                        data={'neurons': neurons, 'synapses': synapses, 'task': learning_task}
                    )
                    
                    if result:
                        st.success("✅ Calcul terminé!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Neurones", f"{result['neurons']:,}")
                            st.metric("Synapses", f"{result['synapses']:,}")
                        with col2:
                            st.metric("Score Plasticité", f"{result['plasticity_score']:.2%}")
                            st.metric("Précision Tâche", f"{result['task_accuracy']:.2%}")
                        with col3:
                            st.metric("Temps Adaptation", f"{result['adaptation_time_hours']:.1f}h")
                            st.metric("Efficacité Bio", f"{result['biological_efficiency']:.2%}")
    
    with tab3:
        st.subheader("Calcul par Repliement de Protéines")
        
        with st.form("protein_compute_form"):
            protein_length = st.slider("Longueur Protéine", 50, 500, 100)
            target_structure = st.selectbox("Structure Cible",
                ["alpha_helix", "beta_sheet", "random_coil", "complex_fold"])
            
            submitted = st.form_submit_button("🔬 Calculer")
            
            if submitted:
                with st.spinner("🔬 Repliement en cours..."):
                    result = api_request(
                        "/api/compute/protein-folding",
                        method="POST",
                        data={'protein_length': protein_length, 'target_structure': target_structure}
                    )
                    
                    if result:
                        st.success("✅ Repliement terminé!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Espace Conformationnel", f"{result['conformational_space_size']:.2e}")
                            st.metric("Temps Repliement", f"{result['folding_time_ms']:.2f} ms")
                        with col2:
                            st.metric("Précision", f"{result['folding_accuracy']:.2%}")
                            st.metric("Tunneling Quantique", result['quantum_tunneling_events'])
                        with col3:
                            st.metric("Puissance Calcul", f"{result['computational_power_tflops']:.0f} TFLOPS")
                            st.metric("Efficacité vs Silicium", f"{result['energy_efficiency_vs_silicon']:.0f}x")
    
    with tab4:
        st.subheader("Calcul par Automates Cellulaires")
        
        with st.form("cellular_compute_form"):
            grid_size = st.slider("Taille Grille", 50, 500, 100)
            iterations = st.slider("Itérations", 100, 10000, 1000)
            
            submitted = st.form_submit_button("🔲 Calculer")
            
            if submitted:
                with st.spinner("🔲 Simulation en cours..."):
                    result = api_request(
                        "/api/compute/cellular-automata",
                        method="POST",
                        data={'grid_size': grid_size, 'rules': {}, 'iterations': iterations}
                    )
                    
                    if result:
                        st.success("✅ Simulation terminée!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Cellules Totales", f"{result['total_cells']:,}")
                            st.metric("États Explorés", f"{result['states_explored']:,}")
                        with col2:
                            st.metric("Motifs Émergents", result['emergent_patterns'])
                            st.metric("Cycles Réplication", result['self_replication_cycles'])
                        with col3:
                            st.metric("Auto-Organisation", f"{result['self_organization_score']:.2%}")
                            st.metric("Niveau Complexité", result['complexity_level'])
    
    with tab5:
        st.subheader("Calcul Hybride Bio-Quantique")
        
        with st.form("hybrid_compute_form"):
            bio_qubits = st.slider("Bio-Qubits", 10, 200, 30)
            quantum_qubits = st.slider("Qubits Quantiques", 10, 200, 30)
            task = st.selectbox("Tâche",
                ["optimization", "search", "simulation", "machine_learning", "cryptography"])
            
            submitted = st.form_submit_button("⚛️ Calculer")
            
            if submitted:
                with st.spinner("⚛️ Calcul hybride en cours..."):
                    result = api_request(
                        "/api/compute/bio-quantum-hybrid",
                        method="POST",
                        data={'bio_qubits': bio_qubits, 'quantum_qubits': quantum_qubits, 'task': task}
                    )
                    
                    if result:
                        st.success("✅ Calcul terminé!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Qubits Totaux", result['total_qubits'])
                            st.metric("Intrication Bio-Q", f"{result['bio_quantum_entanglement']:.2%}")
                        with col2:
                            st.metric("Facteur Synergie", f"{result['synergy_factor']:.2f}x")
                            st.metric("Avantage Quantique", f"{result['quantum_advantage']:.0f}x")
                        with col3:
                            st.metric("Temps Cohérence", f"{result['coherence_time_ms']:.1f} ms")
                            st.metric("Stabilité Bio", f"{result['biological_stability']:.2%}")
                        
                        # Graphique de contribution
                        fig = go.Figure(data=[
                            go.Pie(
                                labels=['Biologique', 'Quantique'],
                                values=[result['biological_contribution'], result['quantum_contribution']],
                                hole=.3
                            )
                        ])
                        fig.update_layout(title="Contribution Bio vs Quantique")
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("Calcul Photosynthèse Quantique")
        
        with st.form("photosynthesis_compute_form"):
            photosystems = st.slider("Nombre de Photosystèmes", 5, 100, 10)
            light_intensity = st.slider("Intensité Lumineuse (%)", 10, 100, 80)
            
            submitted = st.form_submit_button("🌿 Calculer")
            
            if submitted:
                with st.spinner("🌿 Calcul photosynthétique en cours..."):
                    result = api_request(
                        "/api/compute/quantum-photosynthesis",
                        method="POST",
                        data={'photosystems': photosystems, 'light_intensity': light_intensity}
                    )
                    
                    if result:
                        st.success("✅ Calcul terminé!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Photosystèmes", result['photosystems'])
                            st.metric("Efficacité Quantique", f"{result['quantum_efficiency']:.2%}")
                        with col2:
                            st.metric("Chemins Parallèles", f"{result['parallel_pathways']:,}")
                            st.metric("Conversion Énergie", f"{result['energy_conversion_efficiency']:.2%}")
                        with col3:
                            st.metric("Cohérence Quantique", f"{result['quantum_coherence_time_ps']:.0f} ps")
                            st.metric("Résistance Bruit", f"{result['noise_resistance']:.2%}")

# ==================== SIMULATIONS AVANCÉES ====================

elif menu == "🧪 Simulations Avancées":
    st.title("🧪 Simulations Bio-Quantiques Avancées")
    
    tab1, tab2 = st.tabs(["➕ Créer", "▶️ Exécuter"])
    
    with tab1:
        st.subheader("Créer une Simulation")
        
        with st.form("simulation_form"):
            name = st.text_input("Nom de la Simulation")
            sim_type = st.selectbox("Type", ["biological", "quantum", "hybrid"])
            duration = st.slider("Durée (secondes)", 10, 300, 60)
            
            st.markdown("**Environnement**")
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider("Température (°C)", 0, 50, 37)
                ph = st.slider("pH", 0.0, 14.0, 7.4, 0.1)
            
            with col2:
                pressure = st.slider("Pression (atm)", 0.1, 10.0, 1.0, 0.1)
                quantum_noise = st.slider("Bruit Quantique", 0.0, 1.0, 0.01, 0.01)
            
            submitted = st.form_submit_button("🧪 Créer")
            
            if submitted and name:
                config = {
                    'name': name,
                    'type': sim_type,
                    'duration': duration,
                    'temperature': temperature,
                    'ph': ph,
                    'pressure': pressure,
                    'quantum_noise': quantum_noise
                }
                
                result = api_request("/api/simulation/create", method="POST", data=config)
                
                if result:
                    st.success(f"✅ Simulation '{name}' créée!")
                    st.session_state.current_sim_id = result['simulation_id']
                    st.json(result)
    
    with tab2:
        st.subheader("Exécuter une Simulation")
        
        sim_id = st.text_input("ID de la Simulation", 
            value=st.session_state.get('current_sim_id', ''))
        
        if st.button("▶️ Exécuter") and sim_id:
            with st.spinner("🔄 Simulation en cours..."):
                result = api_request(
                    f"/api/simulation/{sim_id}/run",
                    method="POST"
                )
                
                if result:
                    st.success("✅ Simulation terminée!")
                    
                    summary = result['summary']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Taux de Succès", f"{summary['success_rate']:.2%}")
                    with col2:
                        st.metric("Score Stabilité", f"{summary['stability_score']:.2%}")
                    with col3:
                        st.metric("Index Performance", f"{summary['performance_index']:.2%}")
                    with col4:
                        st.metric("Efficacité Énergie", f"{summary['energy_efficiency']:.2%}")
                    
                    st.markdown("---")
                    
                    # Timeline
                    st.subheader("Évolution Temporelle")
                    timeline_df = pd.DataFrame(result['timeline'])
                    
                    fig = go.Figure()
                    
                    for col in timeline_df.columns:
                        if col != 'second':
                            fig.add_trace(go.Scatter(
                                x=timeline_df['second'],
                                y=timeline_df[col],
                                name=col,
                                mode='lines'
                            ))
                    
                    fig.update_layout(
                        xaxis_title="Temps (s)",
                        yaxis_title="Valeur",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("💡 Insights")
                    for insight in result['insights']:
                        st.write(f"✨ {insight}")

# ==================== EXPÉRIENCES ====================

elif menu == "🔬 Expériences":
    st.title("🔬 Expériences Bio-Quantiques")
    
    tab1, tab2 = st.tabs(["➕ Créer", "🔬 Exécuter"])
    
    with tab1:
        st.subheader("Créer une Expérience")
        
        with st.form("experiment_form"):
            name = st.text_input("Nom de l'Expérience")
            hypothesis = st.text_area("Hypothèse")
            exp_type = st.selectbox("Type", 
                ["computational", "biological", "quantum", "hybrid"])
            
            st.markdown("**Variables**")
            independent_vars = st.text_area("Variables Indépendantes (une par ligne)")
            dependent_vars = st.text_area("Variables Dépendantes (une par ligne)")
            
            methodology = st.selectbox("Méthodologie",
                ["comparative", "longitudinal", "cross_sectional", "experimental"])
            
            sample_size = st.slider("Taille Échantillon", 10, 1000, 100)
            
            submitted = st.form_submit_button("🔬 Créer")
            
            if submitted and name:
                config = {
                    'name': name,
                    'hypothesis': hypothesis,
                    'type': exp_type,
                    'independent_vars': [v.strip() for v in independent_vars.split('\n') if v.strip()],
                    'dependent_vars': [v.strip() for v in dependent_vars.split('\n') if v.strip()],
                    'methodology': methodology,
                    'sample_size': sample_size
                }
                
                result = api_request("/api/experiment/create", method="POST", data=config)
                
                if result:
                    st.success(f"✅ Expérience '{name}' créée!")
                    st.session_state.current_exp_id = result['experiment_id']
                    st.json(result)
    
    with tab2:
        st.subheader("Exécuter une Expérience")
        
        exp_id = st.text_input("ID de l'Expérience",
            value=st.session_state.get('current_exp_id', ''))
        
        if st.button("🔬 Exécuter") and exp_id:
            with st.spinner("🔬 Expérience en cours..."):
                result = api_request(
                    f"/api/experiment/{exp_id}/run",
                    method="POST"
                )
                
                if result:
                    st.success("✅ Expérience terminée!")
                    
                    stats = result['statistics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Moyenne", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Écart-Type", f"{stats['std_dev']:.2f}")
                    with col3:
                        st.metric("Taux Succès", f"{stats['success_rate']:.2%}")
                    with col4:
                        st.metric("p-value", f"{result['p_value']:.4f}")
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Hypothèse Supportée", 
                            "✅ OUI" if result['hypothesis_supported'] else "❌ NON")
                    with col2:
                        st.metric("Niveau Confiance", f"{result['confidence_level']:.2%}")
                    
                    st.markdown("---")
                    
                    # Distribution des mesures
                    st.subheader("Distribution des Résultats")
                    
                    results_df = pd.DataFrame(result['results'])
                    
                    fig = px.histogram(results_df, x='measurement', nbins=30,
                                      title="Distribution des Mesures")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("📝 Conclusions")
                    for conclusion in result['conclusions']:
                        st.write(f"✅ {conclusion}")

# ==================== BIOLOGIE SYNTHÉTIQUE ====================

elif menu == "🧬 Biologie Synthétique":
    st.title("🧬 Biologie Synthétique")
    
    tab1, tab2 = st.tabs(["🧬 Circuits Génétiques", "🦠 Organismes Synthétiques"])
    
    with tab1:
        st.subheader("Créer un Circuit Génétique")
        
        with st.form("genetic_circuit_form"):
            name = st.text_input("Nom du Circuit")
            
            st.markdown("**Composants**")
            promoters = st.text_area("Promoteurs (un par ligne)", "pLac\npTet")
            genes = st.text_area("Gènes (un par ligne)", "gfp\nrfp")
            terminators = st.text_area("Terminateurs (un par ligne)", "T1\nT2")
            
            logic_function = st.selectbox("Fonction Logique", 
                ["AND", "OR", "NOT", "NOR", "NAND", "XOR"])
            
            host = st.selectbox("Organisme Hôte",
                ["e_coli", "b_subtilis", "s_cerevisiae", "mammalian_cell"])
            
            submitted = st.form_submit_button("🧬 Créer")
            
            if submitted and name:
                config = {
                    'name': name,
                    'promoters': [p.strip() for p in promoters.split('\n') if p.strip()],
                    'genes': [g.strip() for g in genes.split('\n') if g.strip()],
                    'terminators': [t.strip() for t in terminators.split('\n') if t.strip()],
                    'logic': logic_function,
                    'host': host
                }
                
                result = api_request("/api/synbio/circuit/create", method="POST", data=config)
                
                if result:
                    st.success(f"✅ Circuit '{name}' créé!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Fonction Logique:** {result['logic_function']}")
                        st.write(f"**Hôte:** {result['host_organism']}")
                    
                    with col2:
                        st.metric("Niveau Expression", f"{result['expression_level']:.2%}")
                        st.metric("Score Stabilité", f"{result['stability_score']:.2%}")
    
    with tab2:
        st.subheader("Concevoir un Organisme Synthétique")
        
        with st.form("synthetic_organism_form"):
            name = st.text_input("Nom de l'Organisme")
            base_organism = st.selectbox("Organisme de Base",
                ["minimal_cell", "e_coli", "yeast", "custom"])
            
            st.markdown("**Modifications Génétiques**")
            genes_added = st.text_area("Gènes Ajoutés (un par ligne)")
            genes_removed = st.text_area("Gènes Retirés (un par ligne)")
            pathways = st.text_area("Voies Métaboliques Modifiées (une par ligne)")
            
            capabilities = st.multiselect("Capacités",
                ["computation", "biosensing", "bioproduction", "bioremediation", 
                 "data_storage", "energy_production"])
            
            submitted = st.form_submit_button("🦠 Concevoir")
            
            if submitted and name:
                config = {
                    'name': name,
                    'base': base_organism,
                    'genes_added': [g.strip() for g in genes_added.split('\n') if g.strip()],
                    'genes_removed': [g.strip() for g in genes_removed.split('\n') if g.strip()],
                    'pathways': [p.strip() for p in pathways.split('\n') if p.strip()],
                    'capabilities': capabilities
                }
                
                result = api_request("/api/synbio/organism/design", method="POST", data=config)
                
                if result:
                    st.success(f"✅ Organisme '{name}' conçu!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Taille Génome", f"{result['genome_size_mb']:.2f} MB")
                        st.metric("Taux Croissance", f"{result['growth_rate_doublings_per_hour']:.2f} /h")
                    
                    with col2:
                        st.metric("Efficacité Métabolique", f"{result['metabolic_efficiency']:.2%}")
                        st.write(f"**Capacités:** {', '.join(result['capabilities'])}")

# ==================== RÉSEAUX BIO-QUANTIQUES ====================

elif menu == "🌐 Réseaux Bio-Quantiques":
    st.title("🌐 Réseaux Bio-Quantiques Distribués")
    
    tab1, tab2 = st.tabs(["➕ Créer", "📊 Distribuer Tâche"])
    
    with tab1:
        st.subheader("Créer un Réseau")
        
        with st.form("network_form"):
            name = st.text_input("Nom du Réseau")
            topology = st.selectbox("Topologie", ["mesh", "star", "ring", "tree", "fully_connected"])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bio_nodes = st.slider("Nœuds Biologiques", 1, 50, 5)
            with col2:
                quantum_nodes = st.slider("Nœuds Quantiques", 1, 50, 3)
            with col3:
                hybrid_nodes = st.slider("Nœuds Hybrides", 0, 20, 2)
            
            submitted = st.form_submit_button("🌐 Créer")
            
            if submitted and name:
                total_nodes = bio_nodes + quantum_nodes + hybrid_nodes
                
                config = {
                    'name': name,
                    'topology': topology,
                    'nodes': total_nodes,
                    'bio_nodes': bio_nodes,
                    'quantum_nodes': quantum_nodes,
                    'hybrid_nodes': hybrid_nodes
                }
                
                result = api_request("/api/network/create", method="POST", data=config)
                
                if result:
                    st.success(f"✅ Réseau '{name}' créé!")
                    st.session_state.current_network_id = result['network_id']
                    st.json(result)
    
    with tab2:
        st.subheader("Distribuer une Tâche sur le Réseau")
        
        network_id = st.text_input("ID du Réseau",
            value=st.session_state.get('current_network_id', ''))
        
        task_type = st.selectbox("Type de Tâche",
            ["computation", "optimization", "simulation", "learning", "data_processing"])
        
        complexity = st.selectbox("Complexité", ["low", "medium", "high", "extreme"])
        
        if st.button("📊 Distribuer") and network_id:
            with st.spinner("📡 Distribution en cours..."):
                result = api_request(
                    f"/api/network/{network_id}/distribute-task",
                    method="POST",
                    data={'type': task_type, 'complexity': complexity}
                )
                
                if result:
                    st.success("✅ Tâche distribuée et terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Nœuds Utilisés", result['total_nodes'])
                        st.metric("Qualité Agrégée", f"{result['aggregated_quality']:.2%}")
                    with col2:
                        st.metric("Temps Total", f"{result['total_time_ms']:.2f} ms")
                        st.metric("Énergie Totale", f"{result['total_energy_uj']:.3f} µJ")
                    with col3:
                        st.metric("Efficacité Réseau", f"{result['network_efficiency']:.2%}")
                        st.metric("Avantage Distribué", f"{result['distributed_advantage']:.1f}x")
                    
                    st.markdown("---")
                    
                    st.subheader("Résultats par Nœud")
                    nodes_df = pd.DataFrame(result['node_results'])
                    st.dataframe(nodes_df, use_container_width=True)

# ==================== ÉVOLUTION & OPTIMISATION ====================

elif menu == "🧬 Évolution & Optimisation":
    st.title("🧬 Évolution et Optimisation Biologique")
    
    st.subheader("Optimisation Évolutive")
    
    with st.form("evolution_form"):
        generations = st.slider("Nombre de Générations", 10, 5000, 100)
        population_size = st.slider("Taille Population", 10, 500, 50)
        mutation_rate = st.slider("Taux de Mutation", 0.001, 0.1, 0.01, 0.001)
        
        submitted = st.form_submit_button("🧬 Lancer l'Évolution")
        
        if submitted:
            with st.spinner("🧬 Évolution en cours..."):
                result = api_request(
                    "/api/evolution/run",
                    method="POST",
                    data={
                        'generations': generations,
                        'population_size': population_size,
                        'mutation_rate': mutation_rate
                    }
                )
                
                if result:
                    st.success("✅ Évolution terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Générations", result['generations'])
                        st.metric("Taille Population", result['population_size'])
                    with col2:
                        st.metric("Mutations Totales", result['total_mutations'])
                        st.metric("Adaptations Réussies", result['successful_adaptations'])
                    with col3:
                        st.metric("Génération Convergence", result['convergence_generation'])
                        st.metric("Diversité", f"{result['diversity_maintained']:.2%}")
                    
                    st.markdown("---")
                    
                    # Courbe d'évolution de fitness
                    st.subheader("Évolution de la Fitness")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=result['fitness_history'],
                        mode='lines+markers',
                        name='Fitness',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Génération",
                        yaxis_title="Fitness",
                        hovermode='x'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    if result.get('best_individual'):
                        st.subheader("🏆 Meilleur Individu")
                        best = result['best_individual']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Génération:** {best['generation']}")
                            st.write(f"**Fitness:** {best['fitness']:.2f}")
                        
                        with col2:
                            st.write(f"**Génome (échantillon):** `{best['genome'][:30]}...`")
                    
                    st.markdown("---")
                    
                    st.subheader("🌟 Propriétés Émergentes")
                    for prop in result['emergent_properties']:
                        st.write(f"✨ {prop}")

# ==================== ANALYTICS & RAPPORTS ====================

elif menu == "📊 Analytics & Rapports":
    st.title("📊 Analytics et Rapports Bio-Quantiques")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Analytics Globales", 
        "💻 Analyse Ordinateur",
        "🤖 Analyse Agent",
        "📄 Rapport Complet"
    ])
    
    with tab1:
        st.subheader("Analytics Globales")
        
        analytics = api_request("/api/analytics/global")
        
        if analytics:
            # Vue d'ensemble
            st.markdown("### Vue d'Ensemble du Système")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Ordinateurs Bio-Q", analytics['total_bio_computers'])
                st.metric("Modèles IA Bio", analytics['total_bio_ai_models'])
            with col2:
                st.metric("Agents", analytics['total_agents'])
                st.metric("Simulations", analytics['total_simulations'])
            with col3:
                st.metric("Expériences", analytics['total_experiments'])
                st.metric("Réseaux", analytics['total_networks'])
            with col4:
                st.metric("Évolutions", analytics['total_evolutions'])
                st.info("**Status:** " + analytics['system_status'].upper())
            
            st.markdown("---")
            
            # Métriques clés
            st.markdown("### Métriques Clés de Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Efficacité Biologique", f"{analytics['biological_efficiency_avg']:.2%}")
            with col2:
                st.metric("Avantage Quantique", f"{analytics['quantum_advantage_avg']:.1f}x")
            with col3:
                st.metric("Synergie Bio-Quantique", f"{analytics['bio_quantum_synergy_avg']:.2f}x")
            
            st.markdown("---")
            
            # Graphiques comparatifs
            st.subheader("Comparaison des Technologies")
            
            technologies = ['ADN', 'Organoïdes', 'Protéines', 'Quantique', 'Hybride']
            performance = [np.random.uniform(80, 95) for _ in technologies]
            
            fig = go.Figure(data=[
                go.Bar(x=technologies, y=performance, marker_color='lightgreen')
            ])
            
            fig.update_layout(
                title="Performance par Technologie",
                yaxis_title="Score de Performance",
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Analyse d'un Ordinateur")
        
        computers = api_request("/api/computer/list") or []
        
        if computers:
            computer_names = {f"{c['name']} ({c['type']})": c['computer_id'] for c in computers}
            selected_computer = st.selectbox("Sélectionner un ordinateur", list(computer_names.keys()))
            
            if st.button("📊 Analyser"):
                computer_id = computer_names[selected_computer]
                analytics = api_request(f"/api/analytics/computer/{computer_id}")
                
                if analytics:
                    st.success("✅ Analyse terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Temps de Fonctionnement", f"{analytics['uptime_hours']:.1f}h")
                        st.metric("Tâches Complétées", f"{analytics['tasks_completed']:,}")
                    with col2:
                        st.metric("Performance Moyenne", f"{analytics['average_performance']:.2%}")
                        st.metric("Efficacité Énergétique", f"{analytics['energy_efficiency']:.2%}")
                    with col3:
                        st.metric("Score Fiabilité", f"{analytics['reliability_score']:.2%}")
                        
                        if analytics.get('biological_health'):
                            st.metric("Santé Biologique", f"{analytics['biological_health']:.2%}")
                        
                        if analytics.get('quantum_fidelity'):
                            st.metric("Fidélité Quantique", f"{analytics['quantum_fidelity']:.4f}")
        else:
            st.info("Créez d'abord un ordinateur.")
    
    with tab3:
        st.subheader("Analyse d'un Agent")
        
        agents = api_request("/api/agent/list") or []
        
        if agents:
            agent_names = {a['name']: a['agent_id'] for a in agents}
            selected_agent = st.selectbox("Sélectionner un agent", list(agent_names.keys()))
            
            if st.button("📊 Analyser", key="analyze_agent"):
                agent_id = agent_names[selected_agent]
                analytics = api_request(f"/api/analytics/agent/{agent_id}")
                
                if analytics:
                    st.success("✅ Analyse terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Niveau Intelligence", analytics['intelligence_level'].upper())
                        st.metric("Décisions Prises", f"{analytics['decisions_made']:,}")
                    with col2:
                        st.metric("Épisodes Apprentissage", f"{analytics['learning_episodes']:,}")
                        st.metric("Événements Adaptation", f"{analytics['adaptation_events']:,}")
                    with col3:
                        st.metric("Score Performance", f"{analytics['performance_score']:.2%}")
                        st.metric("Niveau Autonomie", f"{analytics['autonomy_level']:.2%}")
                    
                    st.markdown("---")
                    
                    # Graphique radar
                    st.subheader("Profil de Compétences")
                    
                    categories = ['Intelligence', 'Autonomie', 'Performance', 'Adaptation', 'Efficacité']
                    values = [
                        analytics['performance_score'] * 100,
                        analytics['autonomy_level'] * 100,
                        analytics['performance_score'] * 100,
                        np.random.uniform(80, 95),
                        np.random.uniform(85, 98)
                    ]
                    
                    fig = go.Figure(data=go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself'
                    ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Créez d'abord un agent.")
    
    with tab4:
        st.subheader("Rapport Complet du Système")
        
        if st.button("📄 Générer le Rapport Complet"):
            with st.spinner("📝 Génération du rapport..."):
                report = api_request("/api/report/comprehensive")
                
                if report:
                    st.success("✅ Rapport généré!")
                    
                    # Résumé exécutif
                    st.markdown("### 📋 Résumé Exécutif")
                    
                    summary = report['executive_summary']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Ordinateurs Bio-Q", summary['total_bio_computers'])
                        st.metric("Modèles IA", summary['total_ai_models'])
                    with col2:
                        st.metric("Agents", summary['total_agents'])
                        st.metric("Efficacité Bio", f"{summary['biological_efficiency']:.2%}")
                    with col3:
                        st.metric("Avantage Quantique", f"{summary['quantum_advantage']:.1f}x")
                        st.metric("Synergie Bio-Q", f"{summary['bio_quantum_synergy']:.2f}x")
                    
                    st.markdown("---")
                    
                    # Réalisations clés
                    st.markdown("### 🏆 Réalisations Clés")
                    for achievement in report['key_achievements']:
                        st.write(f"✅ {achievement}")
                    
                    st.markdown("---")
                    
                    # État des technologies
                    st.markdown("### 🔬 État des Technologies")
                    
                    tech_status = report['technology_status']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        for tech, status in list(tech_status.items())[:3]:
                            st.write(f"**{tech}:** {status.upper()}")
                    
                    with col2:
                        for tech, status in list(tech_status.items())[3:]:
                            st.write(f"**{tech}:** {status.upper()}")
                    
                    st.markdown("---")
                    
                    # Insights de recherche
                    st.markdown("### 💡 Insights de Recherche")
                    for insight in report['research_insights']:
                        st.write(f"🔬 {insight}")
                    
                    st.markdown("---")
                    
                    # Projections futures
                    st.markdown("### 🔮 Projections Futures")
                    
                    projections = report['future_projections']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Potentiel Scalabilité:** {projections['scalability_potential']}")
                        st.write(f"**Émergence Conscience:** {projections['consciousness_emergence']}")
                    
                    with col2:
                        st.write(f"**Singularité Techno. (ETA):** {projections['technological_singularity_eta_years']} ans")
                        st.write(f"**Maturité Marché:** {projections['market_readiness']}")
                    
                    st.markdown("---")
                    
                    # Bouton de téléchargement
                    report_json = json.dumps(report, indent=2)
                    st.download_button(
                        label="📥 Télécharger le Rapport Complet (JSON)",
                        data=report_json,
                        file_name=f"rapport_bioquantum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

# ==================== FOOTER ====================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🧬 Bio-Quantum Computing Engine v1.0.0</p>
        <p>Ordinateurs Biologiques • Quantiques • Hybrides</p>
        <p>Intelligence Artificielle Biologique & Quantique</p>
        <p>© 2025 - Tous droits réservés</p>
    </div>
    """,
    unsafe_allow_html=True
)