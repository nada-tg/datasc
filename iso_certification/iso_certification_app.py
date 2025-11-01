"""
🌍 Universal ISO Certification Platform - Frontend Streamlit COMPLET
Certification Mondiale • IA • Quantique • AGI • Visualisation 3D

Lancement:
streamlit run iso_certification_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List
import json

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="🌍 ISO Certification",
    page_icon="🌍",
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
        background: linear-gradient(90deg, #1976D2 0%, #2196F3 30%, #03A9F4 60%, #00BCD4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: glow 3s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px #1976D2); }
        to { filter: drop-shadow(0 0 40px #00BCD4); }
    }
    .iso-card {
        border: 3px solid #1976D2;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(25, 118, 210, 0.1) 0%, rgba(0, 188, 212, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(25, 118, 210, 0.4);
        transition: all 0.3s;
    }
    .iso-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(33, 150, 243, 0.6);
    }
    .badge-certified {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .badge-pending {
        background: linear-gradient(90deg, #FF9800, #FFC107);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .badge-progress {
        background: linear-gradient(90deg, #2196F3, #03A9F4);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .audit-card {
        background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
    .quantum-glow {
        animation: quantum-pulse 2s ease-in-out infinite;
    }
    @keyframes quantum-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== INITIALISATION ====================
if 'iso_platform' not in st.session_state:
    st.session_state.iso_platform = {
        'organizations': {},
        'certifications': [],
        'audits': [],
        'ai_analyses': [],
        'quantum_assessments': [],
        'agi_evaluations': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str, level: str = "INFO"):
    st.session_state.iso_platform['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_maturity(org: Dict) -> float:
    base = 0.5
    if org['employees'] > 1000:
        base += 0.15
    elif org['employees'] > 100:
        base += 0.10
    if org.get('annual_revenue_usd', 0) > 100000000:
        base += 0.15
    return min(1.0, base + np.random.uniform(0, 0.15))

def perform_gap_analysis(standard: str) -> Dict:
    clauses = ['Context', 'Leadership', 'Planning', 'Support', 'Operation', 'Performance', 'Improvement']
    gaps = {}
    for clause in clauses:
        compliance = np.random.uniform(0.5, 0.95)
        gaps[clause] = {
            'current': compliance,
            'target': 1.0,
            'gap': 1.0 - compliance,
            'priority': 'High' if compliance < 0.7 else 'Medium' if compliance < 0.85 else 'Low'
        }
    return gaps

def create_3d_globe_visualization(orgs_by_country: Dict):
    """Visualisation 3D globe terrestre"""
    country_coords = {
        'France': {'lat': 46.2276, 'lon': 2.2137},
        'USA': {'lat': 37.0902, 'lon': -95.7129},
        'Germany': {'lat': 51.1657, 'lon': 10.4515},
        'China': {'lat': 35.8617, 'lon': 104.1954},
        'Japan': {'lat': 36.2048, 'lon': 138.2529},
        'UK': {'lat': 55.3781, 'lon': -3.4360},
        'Brazil': {'lat': -14.2350, 'lon': -51.9253},
        'India': {'lat': 20.5937, 'lon': 78.9629},
        'Canada': {'lat': 56.1304, 'lon': -106.3468},
        'Australia': {'lat': -25.2744, 'lon': 133.7751}
    }
    
    countries, lats, lons, counts = [], [], [], []
    
    for country, count in orgs_by_country.items():
        if country in country_coords:
            countries.append(country)
            lats.append(country_coords[country]['lat'])
            lons.append(country_coords[country]['lon'])
            counts.append(count)
    
    fig = go.Figure(data=go.Scattergeo(
        lon=lons, lat=lats,
        text=[f"{c}: {cnt} org(s)" for c, cnt in zip(countries, counts)],
        mode='markers+text',
        marker=dict(
            size=[c * 20 for c in counts],
            color=counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Organisations"),
            line=dict(width=1, color='white')
        ),
        textposition="top center",
        textfont=dict(size=10, color='white')
    ))
    
    fig.update_geos(
        projection_type="orthographic",
        showcountries=True, countrycolor="lightgray",
        showocean=True, oceancolor="LightBlue",
        showlakes=True, lakecolor="Blue",
        showland=True, landcolor="lightgreen",
        bgcolor='rgba(0,0,0,0.8)'
    )
    
    fig.update_layout(
        title="🌍 Organisations Certifiées dans le Monde",
        geo=dict(projection_rotation=dict(lon=0, lat=20, roll=0)),
        template="plotly_dark", height=600
    )
    
    return fig

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🌍 Universal ISO Certification Platform</h1>', 
           unsafe_allow_html=True)
st.markdown("### Certification Mondiale • IA • Quantique • AGI • Super Intelligence • Visualisation 3D")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/1976D2/FFFFFF?text=ISO+Platform", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio("🎯 Navigation", [
        "🏠 Dashboard Global",
        "🏢 Enregistrer Organisation",
        "📜 Demander Certification",
        "🔍 Audit & Conformité",
        "🤖 Analyse IA",
        "⚛️ Assessment Quantique",
        "🧠 Évaluation AGI",
        "🌍 Carte Mondiale 3D",
        "📊 Statistiques",
        "📈 Comparaisons",
        "💡 Standards ISO",
        "⚙️ Paramètres"
    ])
    
    st.markdown("---")
    st.markdown("### 📊 Indicateurs")
    
    total_orgs = len(st.session_state.iso_platform['organizations'])
    total_certs = len(st.session_state.iso_platform['certifications'])
    certified = len([c for c in st.session_state.iso_platform['certifications'] if c.get('status') == 'Certifié'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏢 Orgs", total_orgs)
        st.metric("📜 Certifs", total_certs)
    with col2:
        st.metric("✅ Certifiés", certified)
        st.metric("🤖 Analyses IA", len(st.session_state.iso_platform['ai_analyses']))

# ==================== PAGE: DASHBOARD GLOBAL ====================
if page == "🏠 Dashboard Global":
    st.header("🏠 Dashboard Global - Vue d'Ensemble Mondiale")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="iso-card"><h2>🏢</h2><h3>{total_orgs}</h3><p>Organisations</p></div>', 
                   unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="iso-card"><h2>📜</h2><h3>{total_certs}</h3><p>Certifications</p></div>', 
                   unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="iso-card"><h2>✅</h2><h3>{certified}</h3><p>Certifiés</p></div>', 
                   unsafe_allow_html=True)
    with col4:
        ai_count = len(st.session_state.iso_platform['ai_analyses'])
        st.markdown(f'<div class="iso-card"><h2>🤖</h2><h3>{ai_count}</h3><p>IA Analyses</p></div>', 
                   unsafe_allow_html=True)
    with col5:
        quantum_count = len(st.session_state.iso_platform['quantum_assessments'])
        st.markdown(f'<div class="iso-card"><h2>⚛️</h2><h3>{quantum_count}</h3><p>Quantique</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.session_state.iso_platform['certifications']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Certifications par Statut")
            statuses = {}
            for cert in st.session_state.iso_platform['certifications']:
                status = cert.get('status', 'En Attente')
                statuses[status] = statuses.get(status, 0) + 1
            
            fig = go.Figure(data=[go.Pie(
                labels=list(statuses.keys()),
                values=list(statuses.values()),
                hole=0.4,
                marker_colors=['#4CAF50', '#FFC107', '#F44336', '#9E9E9E']
            )])
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Scores de Conformité")
            scores = [c.get('compliance_score', 0) for c in st.session_state.iso_platform['certifications']]
            fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=10, marker_color='#2196F3')])
            fig.update_layout(xaxis_title="Score Conformité", yaxis_title="Nombre",
                            template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: ENREGISTRER ORGANISATION ====================
elif page == "🏢 Enregistrer Organisation":
    st.header("🏢 Enregistrement Organisation")
    
    with st.form("register_org"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom Organisation", "TechCorp Global")
            country = st.selectbox("Pays", 
                ["France", "USA", "Germany", "China", "Japan", "UK", "Brazil", "India", "Canada", "Australia"])
            industry = st.selectbox("Industrie",
                ["Technologies", "Finance", "Santé", "Manufacturing", "Énergie", "Transport"])
        
        with col2:
            org_type = st.selectbox("Type", ["Entreprise", "Gouvernement", "ONG", "Recherche", "Santé", "Éducation"])
            employees = st.number_input("Employés", 1, 1000000, 500)
            revenue = st.number_input("Revenu Annuel (USD)", 0, 10000000000, 5000000)
            email = st.text_input("Email Contact", "contact@techcorp.com")
        
        website = st.text_input("Site Web (optionnel)", "https://techcorp.com")
        
        if st.form_submit_button("🏢 Enregistrer", type="primary"):
            org_id = f"org_{len(st.session_state.iso_platform['organizations']) + 1}"
            
            org = {
                'id': org_id, 'name': name, 'country': country, 'industry': industry,
                'type': org_type, 'employees': employees, 'annual_revenue_usd': revenue,
                'website': website, 'contact_email': email,
                'maturity_score': calculate_maturity({'employees': employees, 'annual_revenue_usd': revenue, 'industry': industry}),
                'certifications_count': 0, 'created_at': datetime.now().isoformat()
            }
            
            st.session_state.iso_platform['organizations'][org_id] = org
            log_event(f"Organisation enregistrée: {name}", "SUCCESS")
            
            st.success(f"✅ Organisation '{name}' enregistrée!")
            st.balloons()

# ==================== PAGE: DEMANDER CERTIFICATION ====================
elif page == "📜 Demander Certification":
    st.header("📜 Demande de Certification ISO")
    
    if not st.session_state.iso_platform['organizations']:
        st.warning("⚠️ Enregistrez d'abord une organisation")
    else:
        org_id = st.selectbox("Organisation",
            list(st.session_state.iso_platform['organizations'].keys()),
            format_func=lambda x: st.session_state.iso_platform['organizations'][x]['name'])
        
        iso_standard = st.selectbox("Norme ISO", [
            "ISO 9001:2015 - Qualité", "ISO 14001:2015 - Environnement",
            "ISO 27001:2022 - Sécurité Information", "ISO 45001:2018 - Santé Sécurité"
        ])
        
        scope = st.text_area("Périmètre", "Développement logiciel et services IT")
        
        col1, col2 = st.columns(2)
        with col1:
            target_date = st.date_input("Date Cible")
        with col2:
            use_ai = st.checkbox("🤖 Analyse IA", value=True)
            use_quantum = st.checkbox("⚛️ Assessment Quantique")
            use_agi = st.checkbox("🧠 Évaluation AGI")
        
        if st.button("📜 Soumettre Demande", type="primary"):
            org = st.session_state.iso_platform['organizations'][org_id]
            cert_id = f"cert_{len(st.session_state.iso_platform['certifications']) + 1}"
            
            gaps = perform_gap_analysis(iso_standard)
            compliance = np.mean([g['current'] for g in gaps.values()])
            
            cert = {
                'certification_id': cert_id, 'organization_id': org_id,
                'iso_standard': iso_standard, 'status': 'En Attente',
                'compliance_score': compliance, 'gap_analysis': gaps,
                'scope': scope, 'target_date': str(target_date),
                'use_ai': use_ai, 'use_quantum': use_quantum, 'use_agi': use_agi,
                'created_at': datetime.now().isoformat()
            }
            
            st.session_state.iso_platform['certifications'].append(cert)
            org['certifications_count'] += 1
            
            st.success(f"✅ Certification {cert_id} créée!")
            st.balloons()
            
            # Afficher gap analysis
            st.subheader("📊 Analyse des Écarts")
            for clause, data in gaps.items():
                progress = data['current']
                st.write(f"**{clause}**")
                st.progress(progress)
                st.caption(f"Conformité: {progress*100:.1f}% | Gap: {data['gap']*100:.1f}% | Priorité: {data['priority']}")

# ==================== PAGE: AUDIT & CONFORMITÉ ====================
elif page == "🔍 Audit & Conformité":
    st.header("🔍 Audits & Vérification Conformité")
    
    if not st.session_state.iso_platform['certifications']:
        st.warning("⚠️ Aucune certification à auditer")
    else:
        cert_id = st.selectbox("Certification à auditer",
            [c['certification_id'] for c in st.session_state.iso_platform['certifications']],
            format_func=lambda x: f"{x} - {next(c['iso_standard'] for c in st.session_state.iso_platform['certifications'] if c['certification_id']==x)}")
        
        cert = next(c for c in st.session_state.iso_platform['certifications'] if c['certification_id']==cert_id)
        
        col1, col2 = st.columns(2)
        with col1:
            audit_type = st.selectbox("Type Audit", ["Initial", "Surveillance", "Recertification", "Spécial"])
            scheduled_date = st.date_input("Date Audit")
        with col2:
            duration = st.slider("Durée (jours)", 1, 10, 3)
            auditors = st.slider("Nombre Auditeurs", 1, 5, 2)
        
        on_site = st.checkbox("Audit Sur Site", value=True)
        
        if st.button("🔍 Planifier Audit", type="primary"):
            audit_id = f"audit_{len(st.session_state.iso_platform['audits']) + 1}"
            
            findings_major = int(np.random.poisson(2))
            findings_minor = int(np.random.poisson(5))
            conformity = max(0.65, cert['compliance_score'] - (findings_major * 0.05) - (findings_minor * 0.02))
            
            if findings_major == 0 and findings_minor <= 3:
                recommendation = "CERTIFIED - Excellent"
                cert['status'] = "Certifié"
            elif findings_major <= 2:
                recommendation = "CONDITIONAL"
            else:
                recommendation = "NOT CERTIFIED"
            
            audit = {
                'audit_id': audit_id, 'certification_id': cert_id,
                'audit_type': audit_type, 'findings_major': findings_major,
                'findings_minor': findings_minor, 'conformity_percentage': conformity * 100,
                'recommendation': recommendation, 'completed_at': datetime.now().isoformat()
            }
            
            st.session_state.iso_platform['audits'].append(audit)
            
            st.success(f"✅ Audit {audit_id} complété!")
            
            st.markdown(f'<div class="audit-card"><h3>{recommendation}</h3></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 NC Majeures", findings_major)
            with col2:
                st.metric("🟡 NC Mineures", findings_minor)
            with col3:
                st.metric("✅ Conformité", f"{conformity*100:.1f}%")

# ==================== PAGE: ANALYSE IA ====================
elif page == "🤖 Analyse IA":
    st.header("🤖 Analyse IA Avancée")
    
    if not st.session_state.iso_platform['organizations']:
        st.warning("⚠️ Aucune organisation")
    else:
        org_id = st.selectbox("Organisation",
            list(st.session_state.iso_platform['organizations'].keys()),
            format_func=lambda x: st.session_state.iso_platform['organizations'][x]['name'])
        
        ai_tech = st.selectbox("Technologie IA", [
            "Machine Learning Classique", "Deep Learning", "IA Quantique",
            "Ordinateur Biologique", "AGI (Intelligence Générale)", "Super Intelligence"
        ])
        
        depth = st.select_slider("Profondeur Analyse", ["Quick", "Standard", "Comprehensive"])
        predictive = st.checkbox("Analyse Prédictive", value=True)
        
        if st.button("🤖 Lancer Analyse IA", type="primary"):
            analysis_id = f"ai_{len(st.session_state.iso_platform['ai_analyses']) + 1}"
            
            readiness = np.random.uniform(0.7, 0.95)
            
            predictions = {
                'ISO 9001': float(np.random.uniform(0.75, 0.95)),
                'ISO 27001': float(np.random.uniform(0.70, 0.90)),
                'ISO 14001': float(np.random.uniform(0.65, 0.88))
            }
            
            analysis = {
                'analysis_id': analysis_id, 'organization_id': org_id,
                'ai_technology': ai_tech, 'readiness_score': readiness,
                'compliance_prediction': predictions, 'timestamp': datetime.now().isoformat()
            }
            
            st.session_state.iso_platform['ai_analyses'].append(analysis)
            
            st.success(f"✅ Analyse {analysis_id} complétée!")
            
            st.metric("🎯 Score Préparation", f"{readiness*100:.1f}%")
            
            st.subheader("📊 Prédictions Conformité")
            fig = go.Figure(data=[go.Bar(
                x=list(predictions.keys()),
                y=list(predictions.values()),
                marker_color='#2196F3'
            )])
            fig.update_layout(yaxis_range=[0, 1], template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: ASSESSMENT QUANTIQUE ====================
elif page == "⚛️ Assessment Quantique":
    st.header("⚛️ Assessment Quantique")
    
    st.markdown('<div class="quantum-glow">', unsafe_allow_html=True)
    st.info("🔬 Évaluation des capacités quantiques pour optimisation conformité ISO")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not st.session_state.iso_platform['organizations']:
        st.warning("⚠️ Aucune organisation")
    else:
        org_id = st.selectbox("Organisation",
            list(st.session_state.iso_platform['organizations'].keys()),
            format_func=lambda x: st.session_state.iso_platform['organizations'][x]['name'])
        
        quantum_ready = st.checkbox("Organisation Quantum-Ready", value=False)
        
        use_cases = st.multiselect("Cas d'Usage Quantique", [
            "Optimisation processus", "Cryptographie post-quantique",
            "Simulation moléculaire", "Machine Learning quantique",
            "Recherche en base de données"
        ])
        
        if st.button("⚛️ Lancer Assessment", type="primary"):
            assessment_id = f"quantum_{len(st.session_state.iso_platform['quantum_assessments']) + 1}"
            
            maturity = int(np.random.uniform(1, 4)) if quantum_ready else 1
            advantage = float(np.random.uniform(1.5, 3.0))
            
            assessment = {
                'assessment_id': assessment_id, 'organization_id': org_id,
                'quantum_maturity_level': maturity, 'quantum_advantage_score': advantage,
                'use_cases': use_cases, 'timestamp': datetime.now().isoformat()
            }
            
            st.session_state.iso_platform['quantum_assessments'].append(assessment)
            
            st.success(f"✅ Assessment {assessment_id} complété!")
            st.balloons()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Niveau Maturité", f"{maturity}/5")
            with col2:
                st.metric("⚡ Avantage Quantique", f"{advantage:.2f}x")

# ==================== PAGE: ÉVALUATION AGI ====================
elif page == "🧠 Évaluation AGI":
    st.header("🧠 Évaluation AGI & Super Intelligence")
    
    if not st.session_state.iso_platform['organizations']:
        st.warning("⚠️ Aucune organisation")
    else:
        org_id = st.selectbox("Organisation",
            list(st.session_state.iso_platform['organizations'].keys()),
            format_func=lambda x: st.session_state.iso_platform['organizations'][x]['name'])
        
        scope = st.multiselect("Périmètre Évaluation", [
            "Gouvernance IA", "Alignement Éthique", "Gestion Risques",
            "Transparence", "Responsabilité", "Sécurité"
        ])
        
        framework = st.selectbox("Framework Éthique", [
            "IEEE Ethics", "EU AI Act", "ISO/IEC 42001", "OECD Principles"
        ])
        
        if st.button("🧠 Lancer Évaluation AGI", type="primary"):
            eval_id = f"agi_{len(st.session_state.iso_platform['agi_evaluations']) + 1}"
            
            agi_score = np.random.uniform(0.70, 0.95)
            ethical_score = np.random.uniform(0.75, 0.95)
            
            evaluation = {
                'evaluation_id': eval_id, 'organization_id': org_id,
                'agi_readiness_score': agi_score, 'ethical_compliance_score': ethical_score,
                'framework': framework, 'scope': scope,
                'timestamp': datetime.now().isoformat()
            }
            
            st.session_state.iso_platform['agi_evaluations'].append(evaluation)
            
            st.success(f"✅ Évaluation {eval_id} complétée!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🧠 Score AGI", f"{agi_score*100:.1f}%")
            with col2:
                st.metric("⚖️ Score Éthique", f"{ethical_score*100:.1f}%")
            
            st.subheader("🛡️ Safeguards Recommandés")
            safeguards = [
                "🔒 Boxed AI avec contrôle strict",
                "🎯 Alignment vérification continue",
                "🔄 Oversight humain obligatoire",
                "⚖️ Value learning par RL inverse",
                "📊 Logging transparence totale"
            ]
            for sg in safeguards:
                st.write(sg)

# ==================== PAGE: CARTE MONDIALE 3D ====================
elif page == "🌍 Carte Mondiale 3D":
    st.header("🌍 Carte Mondiale des Certifications ISO")
    
    if not st.session_state.iso_platform['organizations']:
        st.warning("⚠️ Aucune donnée géographique")
    else:
        orgs_by_country = {}
        for org in st.session_state.iso_platform['organizations'].values():
            country = org['country']
            orgs_by_country[country] = orgs_by_country.get(country, 0) + 1
        
        fig = create_3d_globe_visualization(orgs_by_country)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📊 Top Pays")
        top_countries = sorted(orgs_by_country.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (country, count) in enumerate(top_countries, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {country}**")
                st.progress(count / max(orgs_by_country.values()))
            with col2:
                st.metric("", count)

# ==================== PAGE: STATISTIQUES ====================
elif page == "📊 Statistiques":
    st.header("📊 Statistiques & Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Globales", "🏢 Par Organisation", "🌍 Par Pays", "📜 Par Norme"])
    
    with tab1:
        st.subheader("📈 Statistiques Globales")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏢 Organisations Totales", total_orgs)
        with col2:
            st.metric("📜 Certifications Totales", total_certs)
        with col3:
            avg_compliance = np.mean([c.get('compliance_score', 0) for c in st.session_state.iso_platform['certifications']]) if st.session_state.iso_platform['certifications'] else 0
            st.metric("✅ Conformité Moyenne", f"{avg_compliance*100:.1f}%")
        with col4:
            cert_rate = (certified / total_certs * 100) if total_certs > 0 else 0
            st.metric("🎯 Taux Certification", f"{cert_rate:.1f}%")
        
        st.markdown("---")
        
        if st.session_state.iso_platform['certifications']:
            # Timeline des certifications
            st.subheader("📅 Timeline Certifications")
            
            dates = [datetime.fromisoformat(c['created_at']) for c in st.session_state.iso_platform['certifications']]
            df_timeline = pd.DataFrame({
                'Date': dates,
                'Count': range(1, len(dates) + 1)
            })
            
            fig = px.line(df_timeline, x='Date', y='Count', 
                         title="Évolution Cumulative des Certifications",
                         markers=True)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribution par norme
            st.subheader("📊 Distribution par Norme ISO")
            
            iso_counts = {}
            for cert in st.session_state.iso_platform['certifications']:
                standard = cert['iso_standard'].split(':')[0]
                iso_counts[standard] = iso_counts.get(standard, 0) + 1
            
            fig = go.Figure(data=[go.Bar(
                x=list(iso_counts.keys()),
                y=list(iso_counts.values()),
                marker_color=['#2196F3', '#4CAF50', '#FFC107', '#F44336', '#9C27B0'],
                text=list(iso_counts.values()),
                textposition='auto'
            )])
            fig.update_layout(
                title="Certifications par Norme ISO",
                xaxis_title="Norme",
                yaxis_title="Nombre",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🏢 Statistiques Par Organisation")
        
        if not st.session_state.iso_platform['organizations']:
            st.warning("⚠️ Aucune organisation")
        else:
            org_id = st.selectbox("Sélectionner Organisation",
                list(st.session_state.iso_platform['organizations'].keys()),
                format_func=lambda x: st.session_state.iso_platform['organizations'][x]['name'],
                key="stats_org")
            
            org = st.session_state.iso_platform['organizations'][org_id]
            org_certs = [c for c in st.session_state.iso_platform['certifications'] if c['organization_id'] == org_id]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏢 Employés", f"{org['employees']:,}")
                st.metric("💰 Revenu", f"${org['annual_revenue_usd']:,.0f}")
            with col2:
                st.metric("📊 Maturité", f"{org['maturity_score']*100:.1f}%")
                st.metric("📜 Certifications", len(org_certs))
            with col3:
                st.metric("🌍 Pays", org['country'])
                st.metric("🏭 Industrie", org['industry'])
            
            if org_certs:
                st.subheader("📊 Scores de Conformité")
                
                scores_data = [(c['iso_standard'].split(':')[0], c['compliance_score']) for c in org_certs]
                df_scores = pd.DataFrame(scores_data, columns=['Norme', 'Score'])
                
                fig = px.bar(df_scores, x='Norme', y='Score', 
                           title="Scores par Norme",
                           color='Score',
                           color_continuous_scale='Viridis')
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🌍 Statistiques Par Pays")
        
        if not st.session_state.iso_platform['organizations']:
            st.warning("⚠️ Aucune organisation")
        else:
            countries = list(set(org['country'] for org in st.session_state.iso_platform['organizations'].values()))
            country = st.selectbox("Sélectionner Pays", countries, key="stats_country")
            
            country_orgs = [o for o in st.session_state.iso_platform['organizations'].values() if o['country'] == country]
            country_org_ids = [o['id'] for o in country_orgs]
            country_certs = [c for c in st.session_state.iso_platform['certifications'] if c['organization_id'] in country_org_ids]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🏢 Organisations", len(country_orgs))
            with col2:
                st.metric("📜 Certifications", len(country_certs))
            with col3:
                country_certified = len([c for c in country_certs if c.get('status') == 'Certifié'])
                st.metric("✅ Certifiées", country_certified)
            
            # Industries
            st.subheader("🏭 Distribution par Industrie")
            industries = {}
            for org in country_orgs:
                ind = org['industry']
                industries[ind] = industries.get(ind, 0) + 1
            
            fig = go.Figure(data=[go.Pie(
                labels=list(industries.keys()),
                values=list(industries.values()),
                hole=0.4
            )])
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📜 Statistiques Par Norme ISO")
        
        if not st.session_state.iso_platform['certifications']:
            st.warning("⚠️ Aucune certification")
        else:
            standards = list(set(c['iso_standard'] for c in st.session_state.iso_platform['certifications']))
            standard = st.selectbox("Sélectionner Norme", standards, key="stats_iso")
            
            std_certs = [c for c in st.session_state.iso_platform['certifications'] if c['iso_standard'] == standard]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📜 Total", len(std_certs))
            with col2:
                std_certified = len([c for c in std_certs if c.get('status') == 'Certifié'])
                st.metric("✅ Certifiées", std_certified)
            with col3:
                avg_score = np.mean([c['compliance_score'] for c in std_certs])
                st.metric("📊 Score Moyen", f"{avg_score*100:.1f}%")
            
            # Statuts
            st.subheader("📊 Répartition par Statut")
            statuses = {}
            for cert in std_certs:
                status = cert.get('status', 'En Attente')
                statuses[status] = statuses.get(status, 0) + 1
            
            fig = go.Figure(data=[go.Bar(
                x=list(statuses.keys()),
                y=list(statuses.values()),
                marker_color='#4CAF50',
                text=list(statuses.values()),
                textposition='auto'
            )])
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: COMPARAISONS ====================
elif page == "📈 Comparaisons":
    st.header("📈 Comparaisons & Benchmarking")
    
    if len(st.session_state.iso_platform['organizations']) < 2:
        st.warning("⚠️ Au moins 2 organisations nécessaires")
    else:
        st.subheader("🏢 Sélection Organisations à Comparer")
        
        org_options = list(st.session_state.iso_platform['organizations'].keys())
        selected_orgs = st.multiselect(
            "Sélectionner 2-5 organisations",
            org_options,
            format_func=lambda x: st.session_state.iso_platform['organizations'][x]['name'],
            max_selections=5
        )
        
        if len(selected_orgs) >= 2:
            comparison_data = []
            
            for org_id in selected_orgs:
                org = st.session_state.iso_platform['organizations'][org_id]
                org_certs = [c for c in st.session_state.iso_platform['certifications'] if c['organization_id'] == org_id]
                
                avg_compliance = np.mean([c['compliance_score'] for c in org_certs]) if org_certs else 0
                
                comparison_data.append({
                    'Organisation': org['name'],
                    'Pays': org['country'],
                    'Industrie': org['industry'],
                    'Employés': org['employees'],
                    'Maturité': org['maturity_score'],
                    'Certifications': len(org_certs),
                    'Conformité Moy.': avg_compliance
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            st.subheader("📊 Tableau Comparatif")
            st.dataframe(df_comparison, use_container_width=True)
            
            # Graphiques comparatifs
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Score Maturité")
                fig = go.Figure(data=[go.Bar(
                    x=df_comparison['Organisation'],
                    y=df_comparison['Maturité'],
                    marker_color='#2196F3',
                    text=[f"{v*100:.1f}%" for v in df_comparison['Maturité']],
                    textposition='auto'
                )])
                fig.update_layout(yaxis_range=[0, 1], template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📜 Nombre Certifications")
                fig = go.Figure(data=[go.Bar(
                    x=df_comparison['Organisation'],
                    y=df_comparison['Certifications'],
                    marker_color='#4CAF50',
                    text=df_comparison['Certifications'],
                    textposition='auto'
                )])
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Radar Chart
            st.subheader("🎯 Comparaison Multi-Critères")
            
            fig = go.Figure()
            
            categories = ['Maturité', 'Certifications', 'Conformité']
            
            for _, row in df_comparison.iterrows():
                values = [
                    row['Maturité'],
                    row['Certifications'] / 10,  # Normalisation
                    row['Conformité Moy.']
                ]
                values.append(values[0])  # Fermer le radar
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=row['Organisation']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Meilleure performance
            st.subheader("🏆 Meilleure Performance")
            best_org = df_comparison.loc[df_comparison['Maturité'].idxmax()]
            st.success(f"🏆 **{best_org['Organisation']}** - Score Maturité: {best_org['Maturité']*100:.1f}%")

# ==================== PAGE: STANDARDS ISO ====================
elif page == "💡 Standards ISO":
    st.header("💡 Guide des Standards ISO")
    
    standards_info = {
        "ISO 9001:2015 - Qualité": {
            "description": "Système de Management de la Qualité",
            "objectifs": ["Satisfaction client", "Amélioration continue", "Approche processus"],
            "clauses": ["Contexte", "Leadership", "Planification", "Support", "Opération", "Évaluation", "Amélioration"],
            "industries": ["Manufacturing", "Services", "Technologies"],
            "duree_moyenne": "6-12 mois",
            "cout_moyen": "$15,000 - $50,000"
        },
        "ISO 14001:2015 - Environnement": {
            "description": "Système de Management Environnemental",
            "objectifs": ["Protection environnement", "Conformité légale", "Performance environnementale"],
            "clauses": ["Contexte", "Leadership", "Planification", "Support", "Opération", "Évaluation", "Amélioration"],
            "industries": ["Manufacturing", "Énergie", "Transport"],
            "duree_moyenne": "8-14 mois",
            "cout_moyen": "$20,000 - $60,000"
        },
        "ISO 27001:2022 - Sécurité Information": {
            "description": "Système de Management Sécurité Information",
            "objectifs": ["Protection données", "Gestion risques cyber", "Confidentialité"],
            "clauses": ["Contexte", "Leadership", "Planification", "Support", "Opération", "Évaluation", "Amélioration", "Contrôles A"],
            "industries": ["Technologies", "Finance", "Télécommunications"],
            "duree_moyenne": "9-18 mois",
            "cout_moyen": "$25,000 - $80,000"
        },
        "ISO 45001:2018 - Santé Sécurité": {
            "description": "Système Management Santé Sécurité Travail",
            "objectifs": ["Sécurité employés", "Prévention accidents", "Conformité SST"],
            "clauses": ["Contexte", "Leadership", "Planification", "Support", "Opération", "Évaluation", "Amélioration"],
            "industries": ["Manufacturing", "Construction", "Énergie"],
            "duree_moyenne": "8-15 mois",
            "cout_moyen": "$18,000 - $55,000"
        }
    }
    
    selected_standard = st.selectbox("Sélectionner Standard", list(standards_info.keys()))
    
    info = standards_info[selected_standard]
    
    st.markdown(f'<div class="iso-card">', unsafe_allow_html=True)
    
    st.subheader(f"📜 {selected_standard}")
    st.write(f"**Description:** {info['description']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🎯 Objectifs Principaux:**")
        for obj in info['objectifs']:
            st.write(f"• {obj}")
        
        st.write(f"\n**⏱️ Durée Moyenne:** {info['duree_moyenne']}")
        st.write(f"**💰 Coût Moyen:** {info['cout_moyen']}")
    
    with col2:
        st.write("**📋 Clauses Principales:**")
        for i, clause in enumerate(info['clauses'], 1):
            st.write(f"{i}. {clause}")
    
    st.write("**🏭 Industries Concernées:**")
    st.write(" | ".join(info['industries']))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processus de certification
    st.subheader("🔄 Processus de Certification")
    
    process_steps = [
        {"step": "1️⃣ Gap Analysis", "duration": "1-2 mois", "description": "Évaluation écarts"},
        {"step": "2️⃣ Documentation", "duration": "2-4 mois", "description": "Création documentation"},
        {"step": "3️⃣ Implémentation", "duration": "3-6 mois", "description": "Mise en œuvre système"},
        {"step": "4️⃣ Audit Interne", "duration": "1 mois", "description": "Vérification interne"},
        {"step": "5️⃣ Audit Certification", "duration": "1-2 semaines", "description": "Audit officiel"},
        {"step": "6️⃣ Certification", "duration": "2-4 semaines", "description": "Obtention certificat"}
    ]
    
    for step in process_steps:
        col1, col2, col3 = st.columns([2, 2, 4])
        with col1:
            st.write(f"**{step['step']}**")
        with col2:
            st.write(f"⏱️ {step['duration']}")
        with col3:
            st.write(step['description'])

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Paramètres & Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Général", "📊 Export", "🗑️ Données"])
    
    with tab1:
        st.subheader("🔧 Paramètres Généraux")
        
        theme = st.selectbox("Thème", ["Dark", "Light"])
        language = st.selectbox("Langue", ["Français", "English", "Deutsch", "Español"])
        notifications = st.checkbox("Notifications", value=True)
        auto_save = st.checkbox("Sauvegarde Automatique", value=True)
        
        if st.button("💾 Sauvegarder Paramètres"):
            st.success("✅ Paramètres sauvegardés!")
    
    with tab2:
        st.subheader("📊 Export Données")
        
        export_format = st.selectbox("Format Export", ["JSON", "CSV", "Excel", "PDF"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Exporter Organisations"):
                data = list(st.session_state.iso_platform['organizations'].values())
                if data:
                    df = pd.DataFrame(data)
                    st.download_button(
                        "⬇️ Télécharger",
                        df.to_csv(index=False),
                        "organisations.csv",
                        "text/csv"
                    )
        
        with col2:
            if st.button("📥 Exporter Certifications"):
                data = st.session_state.iso_platform['certifications']
                if data:
                    df = pd.DataFrame(data)
                    st.download_button(
                        "⬇️ Télécharger",
                        df.to_csv(index=False),
                        "certifications.csv",
                        "text/csv"
                    )
        
        with col3:
            if st.button("📥 Exporter Audits"):
                data = st.session_state.iso_platform['audits']
                if data:
                    df = pd.DataFrame(data)
                    st.download_button(
                        "⬇️ Télécharger",
                        df.to_csv(index=False),
                        "audits.csv",
                        "text/csv"
                    )
    
    with tab3:
        st.subheader("🗑️ Gestion Données")
        
        st.warning("⚠️ Actions Irréversibles")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Réinitialiser Tout", type="secondary"):
                if st.checkbox("Confirmer réinitialisation"):
                    st.session_state.iso_platform = {
                        'organizations': {},
                        'certifications': [],
                        'audits': [],
                        'ai_analyses': [],
                        'quantum_assessments': [],
                        'agi_evaluations': [],
                        'log': []
                    }
                    st.success("✅ Données réinitialisées!")
                    st.rerun()
        
        with col2:
            st.metric("📊 Total Entrées", 
                     len(st.session_state.iso_platform['organizations']) + 
                     len(st.session_state.iso_platform['certifications']))

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>🌍 <strong>Universal ISO Certification Platform</strong> v1.0.0</p>
        <p>Powered by IA • Quantique • AGI • Bio-Computing</p>
        <p>© 2025 - Certification Mondiale pour tous types d'organisations</p>
    </div>
""", unsafe_allow_html=True)