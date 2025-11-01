"""
🛡️ Ultra Conservation Technologies Platform - Frontend Streamlit
Préservation • Restauration • Archivage • Monitoring • Protection

Lancement:
streamlit run ultra_conservation_platform_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="🛡️ Ultra Conservation",
    page_icon="🛡️",
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
        background: linear-gradient(90deg, #2E7D32 0%, #43A047 30%, #66BB6A 60%, #81C784 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .conservation-card {
        border: 3px solid #2E7D32;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.1) 0%, rgba(129, 199, 132, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(46, 125, 50, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if 'conservation_lab' not in st.session_state:
    st.session_state.conservation_lab = {
        'artifacts': {},
        'analyses': [],
        'treatments': [],
        'monitoring': [],
        'plans': [],
        'archives': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str, level: str = "INFO"):
    st.session_state.conservation_lab['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_risk_score(artifact: Dict) -> float:
    base = 0.2
    material_risk = {'Organique': 0.3, 'Inorganique': 0.1, 'Composite': 0.2, 'Synthétique': 0.15}
    risk = base + material_risk.get(artifact.get('material_type', 'Composite'), 0.2)
    return min(1.0, max(0.0, risk + np.random.uniform(-0.05, 0.15)))

def assess_state(risk: float) -> str:
    if risk < 0.2: return "Excellent"
    elif risk < 0.4: return "Bon"
    elif risk < 0.6: return "Moyen"
    elif risk < 0.8: return "Mauvais"
    else: return "Critique"

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🛡️ Ultra Conservation Technologies</h1>', unsafe_allow_html=True)
st.markdown("### Préservation • Restauration • Archivage • Monitoring • Protection Patrimoine")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/2E7D32/FFFFFF?text=Conservation+Lab", use_container_width=True)
    st.markdown("---")
    
    page = st.radio("🎯 Navigation", [
        "🏠 Dashboard",
        "📦 Enregistrer Artefact",
        "🔬 Analyse Matériaux",
        "🌡️ Monitoring Climat",
        "💊 Traitements",
        "📋 Plans Préservation",
        "🗂️ Numérisation",
        "📊 Dégradation",
        "📈 Statistiques",
        "⚙️ Paramètres"
    ])
    
    st.markdown("---")
    st.markdown("### 📊 État Lab")
    
    total_artifacts = len(st.session_state.conservation_lab['artifacts'])
    at_risk = sum(1 for a in st.session_state.conservation_lab['artifacts'].values() if a.get('risk_score', 0) > 0.6)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📦 Artefacts", total_artifacts)
        st.metric("🔬 Analyses", len(st.session_state.conservation_lab['analyses']))
    with col2:
        st.metric("⚠️ À Risque", at_risk)
        st.metric("💊 Traitements", len(st.session_state.conservation_lab['treatments']))

# ==================== PAGE: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.header("🏠 Dashboard Central")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="conservation-card"><h2>📦</h2><h3>{total_artifacts}</h3><p>Artefacts</p></div>', unsafe_allow_html=True)
    with col2:
        excellent = sum(1 for a in st.session_state.conservation_lab['artifacts'].values() if a.get('conservation_state') == 'Excellent')
        st.markdown(f'<div class="conservation-card"><h2>✅</h2><h3>{excellent}</h3><p>Excellent État</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="conservation-card"><h2>⚠️</h2><h3>{at_risk}</h3><p>À Risque</p></div>', unsafe_allow_html=True)
    with col4:
        treatments = len(st.session_state.conservation_lab['treatments'])
        st.markdown(f'<div class="conservation-card"><h2>💊</h2><h3>{treatments}</h3><p>Traitements</p></div>', unsafe_allow_html=True)
    with col5:
        archives = len(st.session_state.conservation_lab['archives'])
        st.markdown(f'<div class="conservation-card"><h2>🗂️</h2><h3>{archives}</h3><p>Archives Num.</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.conservation_lab['artifacts']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 État de Conservation")
            states = {}
            for art in st.session_state.conservation_lab['artifacts'].values():
                state = art.get('conservation_state', 'Non évalué')
                states[state] = states.get(state, 0) + 1
            
            fig = go.Figure(data=[go.Pie(labels=list(states.keys()), values=list(states.values()),
                                         hole=0.4, marker_colors=['#2E7D32', '#43A047', '#FDD835', '#FB8C00', '#E53935'])])
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Scores de Risque")
            risks = [a.get('risk_score', 0) for a in st.session_state.conservation_lab['artifacts'].values()]
            if risks:
                fig = go.Figure(data=[go.Histogram(x=risks, nbinsx=10, marker_color='#2E7D32')])
                fig.update_layout(title="Distribution Risques", xaxis_title="Score Risque", yaxis_title="Nombre",
                                template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: ENREGISTRER ARTEFACT ====================
elif page == "📦 Enregistrer Artefact":
    st.header("📦 Enregistrement Nouvel Artefact")
    
    with st.form("register_artifact"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom Artefact", "La Joconde")
            artifact_type = st.selectbox("Type", ["Peinture", "Sculpture", "Manuscrit", "Textile", 
                                                  "Céramique", "Métal", "Bois", "Pierre", "Numérique"])
            material_type = st.selectbox("Type Matériau", ["Organique", "Inorganique", "Composite", "Synthétique"])
            creation_date = st.text_input("Date Création", "1503-1519")
            origin = st.text_input("Origine", "Italie, Renaissance")
        
        with col2:
            height = st.number_input("Hauteur (cm)", 1.0, 1000.0, 77.0)
            width = st.number_input("Largeur (cm)", 1.0, 1000.0, 53.0)
            depth = st.number_input("Profondeur (cm)", 0.1, 100.0, 2.0)
            weight = st.number_input("Poids (kg)", 0.01, 10000.0, 1.0)
            location = st.text_input("Localisation", "Musée du Louvre")
            value = st.number_input("Valeur Estimée (€)", 0.0, 1000000000.0, 1000000.0)
        
        if st.form_submit_button("📦 Enregistrer", type="primary"):
            artifact_id = f"art_{len(st.session_state.conservation_lab['artifacts']) + 1}"
            
            artifact = {
                'id': artifact_id,
                'name': name,
                'artifact_type': artifact_type,
                'material_type': material_type,
                'creation_date': creation_date,
                'origin': origin,
                'dimensions': {'height': height, 'width': width, 'depth': depth},
                'weight_kg': weight,
                'current_location': location,
                'estimated_value_eur': value,
                'risk_score': calculate_risk_score({'material_type': material_type}),
                'created_at': datetime.now().isoformat()
            }
            
            artifact['conservation_state'] = assess_state(artifact['risk_score'])
            artifact['requires_intervention'] = artifact['risk_score'] > 0.6
            
            st.session_state.conservation_lab['artifacts'][artifact_id] = artifact
            log_event(f"Artefact enregistré: {name}", "SUCCESS")
            
            st.success(f"✅ Artefact '{name}' enregistré!")
            st.balloons()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("État", artifact['conservation_state'])
            with col2:
                st.metric("Risque", f"{artifact['risk_score']:.2f}")
            with col3:
                st.metric("Intervention?", "OUI" if artifact['requires_intervention'] else "NON")
            with col4:
                st.metric("ID", artifact_id)

# ==================== PAGE: ANALYSE MATÉRIAUX ====================
elif page == "🔬 Analyse Matériaux":
    st.header("🔬 Analyse des Matériaux")
    
    if not st.session_state.conservation_lab['artifacts']:
        st.warning("⚠️ Enregistrez d'abord un artefact")
    else:
        artifact_id = st.selectbox("Artefact", list(st.session_state.conservation_lab['artifacts'].keys()),
                                   format_func=lambda x: st.session_state.conservation_lab['artifacts'][x]['name'])
        
        st.write("### 🧪 Composition Matériau")
        
        col1, col2 = st.columns(2)
        
        with col1:
            porosity = st.slider("Porosité", 0.0, 1.0, 0.3, 0.01)
            moisture = st.slider("Humidité (%)", 0.0, 100.0, 12.0)
            ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
        
        with col2:
            integrity = st.slider("Intégrité Structurelle (%)", 0.0, 100.0, 85.0)
            
            st.write("**Composition (%)**")
            comp1 = st.number_input("Cellulose", 0.0, 100.0, 60.0)
            comp2 = st.number_input("Lignine", 0.0, 100.0, 25.0)
            comp3 = st.number_input("Autres", 0.0, 100.0, 15.0)
        
        if st.button("🔬 Lancer Analyse", type="primary"):
            with st.spinner("Analyse en cours..."):
                import time
                time.sleep(2)
                
                composition = {'Cellulose': comp1, 'Lignine': comp2, 'Autres': comp3}
                
                # Indicateurs dégradation
                degradation = {}
                if moisture > 65:
                    degradation['biodeterioration'] = np.random.uniform(0.4, 0.8)
                if ph < 5.5 or ph > 8.5:
                    degradation['acidification'] = np.random.uniform(0.3, 0.7)
                if porosity > 0.5:
                    degradation['structural_weakness'] = np.random.uniform(0.2, 0.6)
                
                degradation['discoloration'] = np.random.uniform(0.1, 0.5)
                
                # Recommandations
                recommendations = []
                if degradation.get('biodeterioration', 0) > 0.5:
                    recommendations.extend(["🌡️ Contrôler humidité (40-55%)", "🧪 Traitement biocide préventif"])
                if degradation.get('acidification', 0) > 0.5:
                    recommendations.extend(["⚗️ Désacidification", "📦 Stockage alcalin"])
                if not recommendations:
                    recommendations.append("✅ Maintenir conditions actuelles")
                
                urgency = "URGENT" if max(degradation.values(), default=0) > 0.7 else \
                         "High" if max(degradation.values(), default=0) > 0.5 else "Medium"
                
                analysis = {
                    'analysis_id': f"analysis_{len(st.session_state.conservation_lab['analyses']) + 1}",
                    'artifact_id': artifact_id,
                    'composition': composition,
                    'degradation': degradation,
                    'recommendations': recommendations,
                    'urgency': urgency,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.conservation_lab['analyses'].append(analysis)
                log_event(f"Analyse matériau: {artifact_id}", "INFO")
                
                st.success("✅ Analyse complétée!")
                
                # Résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### 📊 Indicateurs Dégradation")
                    if degradation:
                        fig = go.Figure(data=[go.Bar(x=list(degradation.keys()), y=list(degradation.values()),
                                                    marker_color='#E53935')])
                        fig.update_layout(yaxis_title="Score", template="plotly_dark", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("### 💡 Recommandations")
                    for rec in recommendations:
                        st.write(f"- {rec}")
                    
                    if urgency == "URGENT":
                        st.error(f"⚠️ Urgence: {urgency}")
                    else:
                        st.info(f"📊 Urgence: {urgency}")

# ==================== PAGE: MONITORING CLIMAT ====================
elif page == "🌡️ Monitoring Climat":
    st.header("🌡️ Monitoring Conditions Climatiques")
    
    if not st.session_state.conservation_lab['artifacts']:
        st.warning("⚠️ Sélectionnez d'abord un artefact")
    else:
        artifact_id = st.selectbox("Artefact à Monitorer",
                                   list(st.session_state.conservation_lab['artifacts'].keys()),
                                   format_func=lambda x: st.session_state.conservation_lab['artifacts'][x]['name'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            temp = st.slider("Température (°C)", -20.0, 40.0, 20.0, 0.1)
            humidity = st.slider("Humidité (%)", 0.0, 100.0, 50.0, 1.0)
        
        with col2:
            light = st.slider("Luminosité (lux)", 0.0, 1000.0, 100.0, 10.0)
            uv = st.slider("UV Index", 0.0, 15.0, 0.3, 0.1)
        
        if st.button("📊 Enregistrer Mesure", type="primary"):
            # Vérifier seuils
            alerts = []
            
            if not (18 <= temp <= 22):
                alerts.append(f"🌡️ Température hors plage: {temp}°C (optimal: 18-22°C)")
            if not (40 <= humidity <= 55):
                alerts.append(f"💧 Humidité hors plage: {humidity}% (optimal: 40-55%)")
            if light > 150:
                alerts.append(f"☀️ Lumière excessive: {light} lux (max: 150 lux)")
            if uv > 0.5:
                alerts.append(f"🔆 UV trop élevé: {uv} (max: 0.5)")
            
            monitoring_data = {
                'artifact_id': artifact_id,
                'temperature_c': temp,
                'humidity_percent': humidity,
                'light_lux': light,
                'uv_index': uv,
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
            st.session_state.conservation_lab['monitoring'].append(monitoring_data)
            log_event(f"Monitoring: {artifact_id}", "INFO" if not alerts else "WARNING")
            
            if alerts:
                st.error("⚠️ ALERTES DÉTECTÉES!")
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ Conditions optimales")
            
            # Graphique historique
            if len(st.session_state.conservation_lab['monitoring']) > 1:
                st.write("### 📈 Historique (dernières 24h)")
                
                recent = st.session_state.conservation_lab['monitoring'][-20:]
                temps = [m['temperature_c'] for m in recent]
                hums = [m['humidity_percent'] for m in recent]
                
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Température", "Humidité"))
                
                fig.add_trace(go.Scatter(y=temps, mode='lines+markers', line=dict(color='#E53935', width=2),
                                        name='Température'), row=1, col=1)
                fig.add_hrect(y0=18, y1=22, fillcolor="green", opacity=0.2, row=1, col=1)
                
                fig.add_trace(go.Scatter(y=hums, mode='lines+markers', line=dict(color='#1E88E5', width=2),
                                        name='Humidité'), row=2, col=1)
                fig.add_hrect(y0=40, y1=55, fillcolor="green", opacity=0.2, row=2, col=1)
                
                fig.update_layout(template="plotly_dark", height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: TRAITEMENTS ====================
elif page == "💊 Traitements":
    st.header("💊 Traitements de Conservation")
    
    if not st.session_state.conservation_lab['artifacts']:
        st.warning("⚠️ Enregistrez d'abord un artefact")
    else:
        tab1, tab2 = st.tabs(["➕ Appliquer Traitement", "📋 Historique"])
        
        with tab1:
            artifact_id = st.selectbox("Artefact", list(st.session_state.conservation_lab['artifacts'].keys()),
                                      format_func=lambda x: st.session_state.conservation_lab['artifacts'][x]['name'])
            
            treatment_type = st.selectbox("Type Traitement",
                ["Nettoyage", "Consolidation", "Stabilisation", "Restauration", "Numérisation", "Encapsulation"])
            
            description = st.text_area("Description Traitement",
                "Nettoyage doux de la surface avec solution non-abrasive")
            
            col1, col2 = st.columns(2)
            with col1:
                products = st.text_input("Produits Utilisés", "Solution eau déminéralisée + savon neutre")
                duration = st.number_input("Durée (heures)", 0.5, 100.0, 2.0, 0.5)
            with col2:
                cost = st.number_input("Coût (€)", 0.0, 100000.0, 500.0, 50.0)
            
            if st.button("💊 Appliquer Traitement", type="primary"):
                with st.spinner("Traitement en cours..."):
                    import time
                    time.sleep(2)
                    
                    artifact = st.session_state.conservation_lab['artifacts'][artifact_id]
                    before_state = artifact.get('conservation_state', 'Moyen')
                    before_risk = artifact.get('risk_score', 0.5)
                    
                    # Amélioration
                    improvements = {
                        'Nettoyage': 0.15, 'Consolidation': 0.25, 'Stabilisation': 0.30,
                        'Restauration': 0.40, 'Numérisation': 0.0, 'Encapsulation': 0.20
                    }
                    
                    improvement = improvements.get(treatment_type, 0.15) + np.random.uniform(-0.05, 0.1)
                    new_risk = max(0.0, before_risk - improvement)
                    after_state = assess_state(new_risk)
                    
                    # Mise à jour
                    artifact['conservation_state'] = after_state
                    artifact['risk_score'] = new_risk
                    artifact['last_inspection'] = datetime.now().isoformat()
                    
                    treatment = {
                        'treatment_id': f"treat_{len(st.session_state.conservation_lab['treatments']) + 1}",
                        'artifact_id': artifact_id,
                        'type': treatment_type,
                        'description': description,
                        'products': products,
                        'duration_hours': duration,
                        'cost_eur': cost,
                        'before_state': before_state,
                        'after_state': after_state,
                        'improvement_percent': improvement * 100,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.conservation_lab['treatments'].append(treatment)
                    log_event(f"Traitement appliqué: {treatment_type} sur {artifact_id}", "SUCCESS")
                    
                    st.success("✅ Traitement complété!")
                    st.balloons()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("État Avant", before_state)
                    with col2:
                        st.metric("État Après", after_state, delta="Amélioration")
                    with col3:
                        st.metric("Amélioration", f"{improvement*100:.1f}%")
        
        with tab2:
            st.subheader("📋 Historique Traitements")
            
            if st.session_state.conservation_lab['treatments']:
                for treat in st.session_state.conservation_lab['treatments'][-10:][::-1]:
                    artifact_name = st.session_state.conservation_lab['artifacts'][treat['artifact_id']]['name']
                    
                    with st.expander(f"💊 {treat['type']} - {artifact_name} ({treat['timestamp'][:10]})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Description:** {treat['description']}")
                            st.write(f"**Produits:** {treat['products']}")
                            st.write(f"**Durée:** {treat['duration_hours']}h")
                        with col2:
                            st.write(f"**Coût:** {treat['cost_eur']}€")
                            st.write(f"**Avant:** {treat['before_state']}")
                            st.write(f"**Après:** {treat['after_state']}")
                            st.metric("Amélioration", f"{treat['improvement_percent']:.1f}%")
            else:
                st.info("Aucun traitement effectué")

# ==================== PAGE: PLANS PRÉSERVATION ====================
elif page == "📋 Plans Préservation":
    st.header("📋 Plans de Préservation")
    
    if not st.session_state.conservation_lab['artifacts']:
        st.warning("⚠️ Enregistrez d'abord un artefact")
    else:
        tab1, tab2 = st.tabs(["➕ Créer Plan", "📋 Plans Actifs"])
        
        with tab1:
            artifact_id = st.selectbox("Artefact", list(st.session_state.conservation_lab['artifacts'].keys()),
                                      format_func=lambda x: st.session_state.conservation_lab['artifacts'][x]['name'])
            
            st.write("### 🎯 Objectifs")
            target_temp = st.slider("Température Cible (°C)", 15.0, 25.0, 20.0)
            target_humidity = st.slider("Humidité Cible (%)", 30.0, 70.0, 50.0)
            target_light = st.slider("Lumière Max (lux)", 0.0, 200.0, 100.0)
            
            treatments_planned = st.multiselect("Traitements Planifiés",
                ["Nettoyage", "Consolidation", "Stabilisation", "Restauration", "Encapsulation"],
                default=["Nettoyage", "Consolidation"])
            
            col1, col2 = st.columns(2)
            with col1:
                timeline = st.number_input("Durée Totale (mois)", 1, 60, 6)
            with col2:
                budget = st.number_input("Budget (€)", 100.0, 1000000.0, 5000.0)
            
            priority = st.select_slider("Priorité", ["Low", "Medium", "High", "Critical"])
            
            if st.button("📋 Créer Plan", type="primary"):
                plan = {
                    'plan_id': f"plan_{len(st.session_state.conservation_lab['plans']) + 1}",
                    'artifact_id': artifact_id,
                    'targets': {'temp': target_temp, 'humidity': target_humidity, 'light': target_light},
                    'treatments': treatments_planned,
                    'timeline_months': timeline,
                    'budget_eur': budget,
                    'priority': priority,
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.conservation_lab['plans'].append(plan)
                log_event(f"Plan créé: {artifact_id}", "SUCCESS")
                
                st.success("✅ Plan de préservation créé!")
                
                st.write("### 📊 Résumé Plan")
                st.write(f"**Artefact:** {st.session_state.conservation_lab['artifacts'][artifact_id]['name']}")
                st.write(f"**Durée:** {timeline} mois")
                st.write(f"**Budget:** {budget}€")
                st.write(f"**Priorité:** {priority}")
                st.write(f"**Traitements:** {', '.join(treatments_planned)}")
        
        with tab2:
            st.subheader("📋 Plans Actifs")
            
            if st.session_state.conservation_lab['plans']:
                for plan in st.session_state.conservation_lab['plans']:
                    artifact_name = st.session_state.conservation_lab['artifacts'][plan['artifact_id']]['name']
                    
                    with st.expander(f"📋 {artifact_name} - Priorité {plan['priority']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Durée:** {plan['timeline_months']} mois")
                            st.write(f"**Budget:** {plan['budget_eur']}€")
                            st.write(f"**Traitements:** {', '.join(plan['treatments'])}")
                        with col2:
                            st.write("**Objectifs:**")
                            st.write(f"• Température: {plan['targets']['temp']}°C")
                            st.write(f"• Humidité: {plan['targets']['humidity']}%")
                            st.write(f"• Lumière: {plan['targets']['light']} lux")
            else:
                st.info("Aucun plan actif")

# ==================== PAGE: NUMÉRISATION ====================
elif page == "🗂️ Numérisation":
    st.header("🗂️ Numérisation et Archivage")
    
    if not st.session_state.conservation_lab['artifacts']:
        st.warning("⚠️ Enregistrez d'abord un artefact")
    else:
        artifact_id = st.selectbox("Artefact à Numériser",
                                   list(st.session_state.conservation_lab['artifacts'].keys()),
                                   format_func=lambda x: st.session_state.conservation_lab['artifacts'][x]['name'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            resolution = st.select_slider("Résolution (DPI)", [300, 600, 1200, 2400, 4800], value=600)
            color_depth = st.selectbox("Profondeur Couleur (bits)", [24, 48, 96], index=1)
        
        with col2:
            format_type = st.selectbox("Format", ["TIFF", "PNG", "JPEG2000", "RAW"])
            compression = st.selectbox("Compression", ["None", "Lossless", "Low Loss"])
        
        if st.button("📸 Numériser", type="primary"):
            with st.spinner("Numérisation en cours..."):
                import time
                time.sleep(3)
                
                artifact = st.session_state.conservation_lab['artifacts'][artifact_id]
                dims = artifact.get('dimensions', {'height': 30, 'width': 20})
                
                # Calcul taille
                width_px = int(dims['width'] * resolution / 2.54)
                height_px = int(dims['height'] * resolution / 2.54)
                file_size_mb = (width_px * height_px * color_depth / 8) / (1024 * 1024)
                
                archive = {
                    'archive_id': f"archive_{len(st.session_state.conservation_lab['archives']) + 1}",
                    'artifact_id': artifact_id,
                    'resolution_dpi': resolution,
                    'dimensions_px': {'width': width_px, 'height': height_px},
                    'color_depth': color_depth,
                    'format': format_type,
                    'file_size_mb': file_size_mb,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.conservation_lab['archives'].append(archive)
                log_event(f"Numérisation: {artifact_id}", "SUCCESS")
                
                st.success("✅ Numérisation complétée!")
                st.balloons()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Résolution", f"{resolution} DPI")
                with col2:
                    st.metric("Dimensions", f"{width_px}×{height_px} px")
                with col3:
                    st.metric("Taille", f"{file_size_mb:.1f} MB")
                with col4:
                    st.metric("Format", format_type)
                
                st.info("""
                📦 **Stockage:**
                - ✅ Serveur Principal (RAID 6)
                - ✅ Cloud (Chiffré)
                - ✅ Backup Offline (Bande LTO)
                """)

# ==================== PAGE: DÉGRADATION ====================
elif page == "📊 Dégradation":
    st.header("📊 Analyse de Dégradation")
    
    if not st.session_state.conservation_lab['artifacts']:
        st.warning("⚠️ Enregistrez d'abord un artefact")
    else:
        artifact_id = st.selectbox("Artefact", list(st.session_state.conservation_lab['artifacts'].keys()),
                                   format_func=lambda x: st.session_state.conservation_lab['artifacts'][x]['name'])
        
        degradation_type = st.selectbox("Type de Dégradation",
            ["Physique (usure, fissures)", "Chimique (oxydation, acidification)",
             "Biologique (moisissures, insectes)", "Environnemental (lumière, humidité)",
             "Mécanique (chocs, vibrations)"])
        
        affected_area = st.slider("Zone Affectée (%)", 0.0, 100.0, 15.0)
        progression = st.selectbox("Taux Progression", ["Slow", "Moderate", "Fast"])
        
        if st.button("📊 Analyser Dégradation", type="primary"):
            with st.spinner("Analyse en cours..."):
                import time
                time.sleep(2)
                
                artifact = st.session_state.conservation_lab['artifacts'][artifact_id]
                
                # Calculs
                severity = (affected_area / 100 * 0.7) + (artifact.get('risk_score', 0.5) * 0.3)
                
                rates = {'Slow': 0.05, 'Moderate': 0.15, 'Fast': 0.30}
                degradation_rate = rates[progression]
                
                lifespan = (0.9 - artifact.get('risk_score', 0.5)) / degradation_rate if degradation_rate > 0 else 1000
                
                if severity > 0.8 or lifespan < 5:
                    urgency = "CRITIQUE - Action immédiate"
                elif severity > 0.6 or lifespan < 15:
                    urgency = "HAUTE - Action sous 6 mois"
                elif severity > 0.4 or lifespan < 30:
                    urgency = "MOYENNE - Action sous 2 ans"
                else:
                    urgency = "BASSE - Monitoring routinier"
                
                strategies = []
                if "Physique" in degradation_type:
                    strategies.extend(["Consolidation structurelle", "Encadrement protecteur"])
                elif "Chimique" in degradation_type:
                    strategies.extend(["Neutralisation pH", "Atmosphère contrôlée"])
                elif "Biologique" in degradation_type:
                    strategies.extend(["Traitement biocide", "Contrôle humidité"])
                elif "Environnemental" in degradation_type:
                    strategies.extend(["Filtres UV", "Contrôle climatique"])
                
                st.success("✅ Analyse complétée!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sévérité", f"{severity:.2f}", delta="Critique" if severity > 0.8 else "Acceptable")
                with col2:
                    st.metric("Durée Vie Estimée", f"{lifespan:.0f} ans")
                with col3:
                    st.metric("Urgence", urgency.split('-')[0])
                
                st.write("### 💡 Stratégies de Mitigation")
                for strategy in strategies:
                    st.write(f"✅ {strategy}")
                
                if severity > 0.7:
                    st.error("⚠️ ATTENTION: Dégradation sévère détectée!")

# ==================== PAGE: STATISTIQUES ====================
elif page == "📈 Statistiques":
    st.header("📈 Statistiques et Rapports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📦 Total Artefacts", total_artifacts)
        st.metric("💊 Traitements", len(st.session_state.conservation_lab['treatments']))
    
    with col2:
        st.metric("🔬 Analyses", len(st.session_state.conservation_lab['analyses']))
        st.metric("📋 Plans Actifs", len(st.session_state.conservation_lab['plans']))
    
    with col3:
        st.metric("🗂️ Archives Num.", len(st.session_state.conservation_lab['archives']))
        st.metric("⚠️ À Risque", at_risk)
    
    if st.session_state.conservation_lab['artifacts']:
        st.markdown("---")
        
        # Types d'artefacts
        st.subheader("📊 Distribution Types")
        
        types = {}
        for art in st.session_state.conservation_lab['artifacts'].values():
            t = art.get('artifact_type', 'Autre')
            types[t] = types.get(t, 0) + 1
        
        fig = go.Figure(data=[go.Bar(x=list(types.keys()), y=list(types.values()),
                                     marker_color='#2E7D32')])
        fig.update_layout(title="Types d'Artefacts", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Coûts
        if st.session_state.conservation_lab['treatments']:
            st.markdown("---")
            st.subheader("💰 Coûts Préservation")
            
            total_cost = sum(t.get('cost_eur', 0) for t in st.session_state.conservation_lab['treatments'])
            avg_cost = total_cost / len(st.session_state.conservation_lab['treatments'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Coût Total", f"{total_cost:.2f}€")
            with col2:
                st.metric("Coût Moyen/Traitement", f"{avg_cost:.2f}€")

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Configuration")
    
    st.subheader("💾 Données")
    
    if st.button("🗑️ Réinitialiser Tout"):
        if st.checkbox("Confirmer réinitialisation"):
            st.session_state.conservation_lab = {
                'artifacts': {}, 'analyses': [], 'treatments': [],
                'monitoring': [], 'plans': [], 'archives': [], 'log': []
            }
            st.success("✅ Plateforme réinitialisée")
            st.rerun()

# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal (20 dernières entrées)"):
    if st.session_state.conservation_lab['log']:
        for event in st.session_state.conservation_lab['log'][-20:][::-1]:
            icon = "✅" if event['level'] == "SUCCESS" else "⚠️" if event['level'] == "WARNING" else "ℹ️"
            st.text(f"{icon} {event['timestamp'][:19]} - {event['message']}")
    else:
        st.info("Aucun événement")

st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🛡️ Ultra Conservation Technologies Platform</h3>
        <p>Préservation • Restauration • Archivage • Monitoring</p>
        <p><small>Protection du Patrimoine pour les Générations Futures</small></p>
        <p><small>Version 1.0.0 © 2025</small></p>
    </div>
""", unsafe_allow_html=True)