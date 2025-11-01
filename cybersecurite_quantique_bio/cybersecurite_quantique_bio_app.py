"""
Interface Streamlit pour Moteur IA de Cybersécurité V2.0
Interface utilisateur avancée pour la protection multi-domaines
Architecture Robuste et Complète
streamlit run cybersecurite_quantique_bio_app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Moteur IA Cybersécurité V2",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS avancé
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .threat-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-weight: bold;
        margin: 0.25rem;
    }
    
    .critical-badge { background: linear-gradient(135deg, #f5576c, #f093fb); color: white; }
    .high-badge { background: linear-gradient(135deg, #fa709a, #fee140); color: white; }
    .medium-badge { background: linear-gradient(135deg, #fccb90, #d57eeb); color: white; }
    .low-badge { background: linear-gradient(135deg, #a8edea, #fed6e3); color: #333; }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 5px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation du session state
if 'security_engine' not in st.session_state:
    st.session_state.security_engine = "initialized"
    st.session_state.tools = []
    st.session_state.strategies = []
    st.session_state.simulations = []
    st.session_state.alerts = []
    st.session_state.security_score = 85.0
    st.session_state.threats_blocked = 1247

# En-tête principal
st.markdown('<h1 class="main-header">🛡️ Moteur IA de Cybersécurité Multi-Domaines V2.0</h1>', unsafe_allow_html=True)
st.markdown("---")

# Barre latérale
with st.sidebar:
    st.image("https://via.placeholder.com/200x120/667eea/FFFFFF?text=CyberSec+AI", use_container_width=True)
    
    st.markdown("### 🎯 Navigation Principale")
    
    page = st.radio(
        "Sélectionner une section:",
        [
            "🏠 Tableau de Bord Sécurité",
            "📦 Catalogue de Ressources",
            "🛡️ Ressources Classiques",
            "🤖 Ressources IA/ML",
            "⚛️ Ressources Quantiques",
            "🧬 Ressources Biologiques",
            "🔧 Créer un Outil",
            "📋 Créer une Stratégie",
            "✅ Gestion des Étapes",
            "🧪 Simulations d'Attaques",
            "🌐 Environnements Virtuels",
            "🔍 Intelligence des Menaces",
            "📊 Analytics Sécurité",
            "🎨 Workspace",
            "⚙️ Configurations"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Statut en Temps Réel")
    st.metric("🛡️ Score de Sécurité", f"{st.session_state.security_score:.1f}%", "+2.3%")
    st.metric("🚫 Menaces Bloquées", f"{st.session_state.threats_blocked:,}", "+47")
    st.metric("⚠️ Alertes Actives", len(st.session_state.alerts))
    
    st.markdown("---")
    st.markdown("### 🔔 Alertes Récentes")
    if st.session_state.alerts:
        for alert in st.session_state.alerts[-3:]:
            st.warning(f"⚠️ {alert}")
    else:
        st.success("✅ Aucune alerte")
    
    st.markdown("---")
    st.markdown(f"**⏰ {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}**")

# ==================== PAGE: TABLEAU DE BORD SÉCURITÉ ====================
if page == "🏠 Tableau de Bord Sécurité":
    st.header("🛡️ Tableau de Bord de Sécurité Exécutif")
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1.2rem;">🛡️ Protection</h3>
            <p style="font-size: 2rem; margin: 0.5rem 0;">98.7%</p>
            <small>Taux global</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1.2rem;">🔧 Outils</h3>
            <p style="font-size: 2rem; margin: 0.5rem 0;">""" + str(len(st.session_state.tools)) + """</p>
            <small>Déployés</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1.2rem;">📋 Stratégies</h3>
            <p style="font-size: 2rem; margin: 0.5rem 0;">""" + str(len(st.session_state.strategies)) + """</p>
            <small>Actives</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1.2rem;">🚫 Bloquées</h3>
            <p style="font-size: 2rem; margin: 0.5rem 0;">1,247</p>
            <small>Menaces/24h</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1.2rem;">⚡ Réponse</h3>
            <p style="font-size: 2rem; margin: 0.5rem 0;">12ms</p>
            <small>Temps moyen</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques de monitoring
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Couverture par Type de Menace")
        
        threat_data = {
            'Menace': ['Malware', 'APT', 'IA Adv.', 'Ransomware', 'Phishing', 'DDoS', 'Quantum', 'Bio'],
            'Couverture': [96, 92, 88, 94, 97, 93, 85, 82],
            'Incidents': [247, 12, 8, 34, 156, 45, 2, 3]
        }
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(name='Couverture (%)', x=threat_data['Menace'], y=threat_data['Couverture'],
                   marker_color='#667eea', text=threat_data['Couverture'], textposition='auto'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(name='Incidents', x=threat_data['Menace'], y=threat_data['Incidents'],
                      mode='lines+markers', marker_color='#f5576c', line=dict(width=3)),
            secondary_y=True
        )
        
        fig.update_layout(height=400, legend=dict(orientation="h", y=1.1))
        fig.update_yaxes(title_text="Couverture (%)", secondary_y=False)
        fig.update_yaxes(title_text="Incidents", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💻 Systèmes Protégés")
        
        systems_data = {
            'Système': ['Classiques', 'Serveurs', 'IA/ML', 'Quantique', 'Bio', 'IoT', 'Cloud'],
            'Nombre': [1250, 340, 85, 12, 8, 2400, 156]
        }
        
        fig = go.Figure(data=[
            go.Pie(
                labels=systems_data['Système'],
                values=systems_data['Nombre'],
                hole=0.4,
                marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b']
            )
        ])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Activité en temps réel
    st.subheader("📡 Activité de Sécurité en Temps Réel")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Dernières Défenses Réussies</h4>
            <p>🚫 Ransomware bloqué - 14:32:15</p>
            <p>🚫 Attaque DDoS neutralisée - 14:28:03</p>
            <p>🚫 Phishing détecté et bloqué - 14:25:47</p>
            <p>🚫 Tentative d'intrusion arrêtée - 14:22:11</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Alertes Actives</h4>
            <p>🔍 Comportement suspect détecté - Serveur #42</p>
            <p>🔍 Tentatives de connexion multiples - Admin</p>
            <p>🔍 Trafic anormal - Réseau interne</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>📊 Statistiques du Jour</h4>
            <p>✓ 1,247 menaces bloquées</p>
            <p>✓ 12,458 événements analysés</p>
            <p>✓ 99.2% de disponibilité</p>
            <p>✓ 0 incidents critiques</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Timeline d'activité
    st.subheader("📈 Tendance des Menaces (7 derniers jours)")
    
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    threats_trend = [890, 1050, 975, 1120, 1340, 1180, 1247]
    blocked_trend = [862, 1015, 945, 1085, 1295, 1142, 1210]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=threats_trend, name='Menaces Détectées',
        mode='lines+markers', line=dict(color='#f5576c', width=3),
        fill='tonexty'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=blocked_trend, name='Menaces Bloquées',
        mode='lines+markers', line=dict(color='#00f2fe', width=3),
        fill='tozeroy'
    ))
    fig.update_layout(height=300, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
                       
# ==================== PAGE: CATALOGUE DE RESSOURCES ====================
elif page == "📦 Catalogue de Ressources":
    st.header("📦 Catalogue Complet des Ressources de Sécurité")
    
    st.markdown("""
    <div class="info-box">
        <h3>🛡️ Arsenal de Sécurité Disponible</h3>
        <p>Explorez notre catalogue complet de ressources de cybersécurité:</p>
        <ul>
            <li><strong>🛡️ Classiques:</strong> 5 systèmes (IDS/IPS, SIEM, Firewall, EDR, Sandbox)</li>
            <li><strong>🤖 IA/ML:</strong> 5 systèmes (Anomaly Detection, Threat Prediction, Adversarial Defense)</li>
            <li><strong>⚛️ Quantiques:</strong> 5 systèmes (QKD, PQC, QRNG, Quantum Auth)</li>
            <li><strong>🧬 Biologiques:</strong> 5 systèmes (DNA Firewall, Bio Detection, Error Correction)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filtres
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category_filter = st.multiselect(
            "Catégorie:",
            ["Classique", "IA/ML", "Quantique", "Biologique"],
            default=["Classique", "IA/ML"]
        )
    
    with col2:
        effectiveness_min = st.slider("Efficacité min (%):", 0, 100, 80)
    
    with col3:
        threat_filter = st.selectbox(
            "Type de menace:",
            ["Toutes", "Malware", "APT", "Ransomware", "Phishing", "DDoS"]
        )
    
    with col4:
        sort_by = st.selectbox("Trier par:", ["Nom", "Efficacité", "Coût"])
    
    st.markdown("---")
    
    # Vue comparative
    st.subheader("📊 Comparaison des Capacités")
    
    comparison_data = {
        'Catégorie': ['Classique', 'IA/ML', 'Quantique', 'Biologique'],
        'Détection': [92, 96, 88, 94],
        'Prévention': [94, 89, 95, 91],
        'Réponse': [90, 93, 85, 87],
        'Coût': [60, 75, 85, 70]
    }
    
    fig = go.Figure()
    
    for metric in ['Détection', 'Prévention', 'Réponse']:
        fig.add_trace(go.Scatterpolar(
            r=comparison_data[metric],
            theta=comparison_data['Catégorie'],
            fill='toself',
            name=metric
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    st.subheader("📋 Résumé des Ressources")
    
    summary_df = pd.DataFrame({
        'Catégorie': ['Classique', 'IA/ML', 'Quantique', 'Biologique'],
        'Ressources': [5, 5, 5, 5],
        'Efficacité Moyenne': ['94%', '95%', '91%', '93%'],
        'Temps de Réponse': ['<1ms', '5-20ms', 'N/A', '0.001-1s'],
        'Coût Relatif': ['Moyen', 'Élevé', 'Très Élevé', 'Élevé'],
        'Maturité': ['Mature', 'Émergent', 'Expérimental', 'Recherche']
    })
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==================== PAGE: RESSOURCES CLASSIQUES ====================
elif page == "🛡️ Ressources Classiques":
    st.header("🛡️ Ressources de Sécurité Classiques")
    
    st.markdown("""
    <div class="info-box">
        <h3>🔒 Solutions de Sécurité Éprouvées</h3>
        <p>5 systèmes de sécurité classiques de niveau entreprise</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    classic_resources = [
        {
            "name": "IDS/IPS Nouvelle Génération",
            "type": "Détection/Prévention",
            "throughput": "100 Gbps",
            "signatures": "500,000",
            "effectiveness": 94.0,
            "false_positive": 2.0,
            "response_time": "0.8 ms",
            "status": "Disponible",
            "applications": ["Deep Packet Inspection", "Behavior Analysis", "Real-time Blocking"]
        },
        {
            "name": "SIEM Enterprise Pro",
            "type": "Gestion Événements",
            "throughput": "1M events/s",
            "signatures": "N/A",
            "effectiveness": 92.0,
            "false_positive": 3.0,
            "response_time": "100 ms",
            "status": "Disponible",
            "applications": ["Log Aggregation", "Correlation", "Threat Hunting", "Forensics"]
        },
        {
            "name": "Next-Gen Firewall",
            "type": "Pare-feu",
            "throughput": "80 Gbps",
            "signatures": "N/A",
            "effectiveness": 96.0,
            "false_positive": 1.0,
            "response_time": "0.5 ms",
            "status": "Disponible",
            "applications": ["Application Control", "SSL Inspection", "IPS", "URL Filtering"]
        },
        {
            "name": "EDR Advanced Protection",
            "type": "Endpoint",
            "throughput": "100K agents",
            "signatures": "N/A",
            "effectiveness": 95.0,
            "false_positive": 2.0,
            "response_time": "10 ms",
            "status": "Disponible",
            "applications": ["Behavioral Analysis", "Memory Protection", "Ransomware Blocking"]
        },
        {
            "name": "Sandbox Analyse Dynamique",
            "type": "Analyse Malware",
            "throughput": "100 VMs",
            "signatures": "N/A",
            "effectiveness": 93.0,
            "false_positive": 4.0,
            "response_time": "5000 ms",
            "status": "Disponible",
            "applications": ["Malware Detonation", "Behavior Monitoring", "API Tracking"]
        }
    ]
    
    for i, resource in enumerate(classic_resources):
        with st.expander(f"🛡️ {resource['name']} - {resource['status']}", expanded=(i==0)):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Spécifications**")
                st.write(f"🔹 **Type:** {resource['type']}")
                st.write(f"🔹 **Débit:** {resource['throughput']}")
                if resource['signatures'] != "N/A":
                    st.write(f"🔹 **Signatures:** {resource['signatures']}")
                st.write(f"🔹 **Temps de Réponse:** {resource['response_time']}")
            
            with col2:
                st.markdown("**📈 Performance**")
                st.metric("Efficacité", f"{resource['effectiveness']}%")
                st.metric("Faux Positifs", f"{resource['false_positive']}%")
                
                # Jauge d'efficacité
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=resource['effectiveness'],
                    title={'text': "Efficacité"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "lightblue"},
                            {'range': [90, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                st.markdown("**🎯 Applications**")
                for app in resource['applications']:
                    st.write(f"• {app}")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button(f"🚀 Déployer", key=f"deploy_classic_{i}"):
                    st.success(f"✅ {resource['name']} déployé!")
            with col_b:
                if st.button(f"📊 Tester", key=f"test_classic_{i}"):
                    st.info("Tests en cours...")
            with col_c:
                if st.button(f"⭐ Favoris", key=f"fav_classic_{i}"):
                    st.success("Ajouté aux favoris!")

# ==================== PAGE: CRÉER UN OUTIL ====================
elif page == "🔧 Créer un Outil":
    st.header("🔧 Créateur d'Outils de Sécurité")
    
    st.markdown("""
    <div class="info-box">
        <h3>🛠️ Créez Votre Outil de Sécurité Personnalisé</h3>
        <p>Combinez différentes ressources pour créer un outil sur mesure.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("tool_creation_form"):
        st.subheader("1️⃣ Informations de Base")
        
        col1, col2 = st.columns(2)
        with col1:
            tool_name = st.text_input("Nom de l'outil*", placeholder="Ex: Détecteur Multi-Menaces IA")
        with col2:
            tool_category = st.selectbox(
                "Catégorie*",
                ["DETECTION", "PREVENTION", "RESPONSE", "ENCRYPTION", "MONITORING", 
                 "AUTHENTICATION", "FIREWALL", "INTRUSION_DETECTION", "ANOMALY_DETECTION"]
            )
        
        tool_description = st.text_area(
            "Description",
            placeholder="Décrivez l'objectif et le fonctionnement de votre outil...",
            height=100
        )
        
        st.markdown("---")
        st.subheader("2️⃣ Sélection des Ressources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ressources Classiques:**")
            res_ids = st.checkbox("IDS/IPS Nouvelle Génération")
            res_siem = st.checkbox("SIEM Enterprise Pro")
            res_ngfw = st.checkbox("Next-Gen Firewall")
            res_edr = st.checkbox("EDR Advanced Protection")
            
            st.markdown("**Ressources IA:**")
            res_ai_anomaly = st.checkbox("Détecteur d'Anomalies IA")
            res_ai_threat = st.checkbox("Prédicteur de Menaces IA")
            res_ai_adv = st.checkbox("Défense Adversariale IA")
        
        with col2:
            st.markdown("**Ressources Quantiques:**")
            res_qkd = st.checkbox("Système QKD")
            res_pqc = st.checkbox("Cryptographie Post-Quantique")
            res_qrng = st.checkbox("Générateur Quantique Aléatoire")
            
            st.markdown("**Ressources Biologiques:**")
            res_bio_fw = st.checkbox("Pare-feu ADN Biologique")
            res_bio_detect = st.checkbox("Détecteur Contamination Bio")
            res_bio_auth = st.checkbox("Authentification Biomoléculaire")
        
        st.markdown("---")
        st.subheader("3️⃣ Menaces Ciblées")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            threat_malware = st.checkbox("Malware Classique", value=True)
            threat_apt = st.checkbox("APT")
            threat_ai_adv = st.checkbox("IA Adversariale")
            threat_ai_poison = st.checkbox("Empoisonnement IA")
        with col2:
            threat_quantum = st.checkbox("Cryptanalyse Quantique")
            threat_bio = st.checkbox("Contamination Bio")
            threat_zero = st.checkbox("Zero-Day")
            threat_ransom = st.checkbox("Ransomware", value=True)
        with col3:
            threat_phishing = st.checkbox("Phishing")
            threat_ddos = st.checkbox("DDoS")
        
        st.markdown("---")
        st.subheader("4️⃣ Systèmes Compatibles")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sys_classic = st.checkbox("Ordinateur Classique", value=True)
            sys_server = st.checkbox("Serveur")
        with col2:
            sys_ai = st.checkbox("Système IA")
            sys_quantum = st.checkbox("Quantique")
        with col3:
            sys_bio = st.checkbox("Biologique")
            sys_iot = st.checkbox("IoT")
        with col4:
            sys_cloud = st.checkbox("Cloud")
            sys_hybrid = st.checkbox("Hybride")
        
        st.markdown("---")
        st.subheader("5️⃣ Paramètres de Déploiement")
        
        col1, col2 = st.columns(2)
        with col1:
            complexity = st.selectbox("Complexité de Déploiement:", ["Low", "Medium", "High"])
        with col2:
            cost = st.number_input("Budget Estimé ($)", 1000, 1000000, 10000, 1000)
        
        st.markdown("---")
        
        submitted = st.form_submit_button("🚀 Créer l'Outil", use_container_width=True)
        
        if submitted:
            if not tool_name:
                st.error("❌ Veuillez donner un nom à l'outil")
            else:
                # Collecter les ressources sélectionnées
                selected_resources = []
                if res_ids: selected_resources.append("ids_nextgen")
                if res_siem: selected_resources.append("siem_enterprise")
                if res_ngfw: selected_resources.append("firewall_ngfw")
                if res_edr: selected_resources.append("edr_advanced")
                if res_ai_anomaly: selected_resources.append("ai_anomaly_detector")
                if res_ai_threat: selected_resources.append("ai_threat_predictor")
                if res_ai_adv: selected_resources.append("ai_adversarial_defense")
                if res_qkd: selected_resources.append("qkd_system")
                if res_pqc: selected_resources.append("pqc_lattice")
                if res_qrng: selected_resources.append("qrng")
                if res_bio_fw: selected_resources.append("bio_dna_firewall")
                if res_bio_detect: selected_resources.append("bio_contamination_detector")
                if res_bio_auth: selected_resources.append("bio_auth_system")
                
                # Collecter les menaces
                selected_threats = []
                if threat_malware: selected_threats.append("Malware")
                if threat_apt: selected_threats.append("APT")
                if threat_ai_adv: selected_threats.append("IA Adversariale")
                if threat_quantum: selected_threats.append("Quantum")
                if threat_ransom: selected_threats.append("Ransomware")
                if threat_phishing: selected_threats.append("Phishing")
                if threat_ddos: selected_threats.append("DDoS")
                
                # Collecter les systèmes
                selected_systems = []
                if sys_classic: selected_systems.append("Classique")
                if sys_server: selected_systems.append("Serveur")
                if sys_ai: selected_systems.append("IA")
                if sys_quantum: selected_systems.append("Quantique")
                if sys_bio: selected_systems.append("Bio")
                if sys_iot: selected_systems.append("IoT")
                if sys_cloud: selected_systems.append("Cloud")
                
                if not selected_resources:
                    st.error("❌ Veuillez sélectionner au moins une ressource")
                elif not selected_threats:
                    st.error("❌ Veuillez sélectionner au moins une menace cible")
                elif not selected_systems:
                    st.error("❌ Veuillez sélectionner au moins un système compatible")
                else:
                    # Créer l'outil
                    new_tool = {
                        "id": f"tool_{len(st.session_state.tools) + 1}",
                        "name": tool_name,
                        "category": tool_category,
                        "description": tool_description,
                        "resources": selected_resources,
                        "threats": selected_threats,
                        "systems": selected_systems,
                        "complexity": complexity,
                        "cost": cost,
                        "status": "Créé",
                        "created_at": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "performance": {
                            "effectiveness": round(85 + len(selected_resources) * 2, 1),
                            "detection_rate": round(82 + len(selected_resources) * 1.8, 1),
                            "false_positive": round(max(1, 5 - len(selected_resources) * 0.3), 2),
                            "response_time": round(50 - len(selected_resources) * 2, 1)
                        }
                    }
                    
                    st.session_state.tools.append(new_tool)
                    
                    st.success(f"✅ Outil '{tool_name}' créé avec succès!")
                    st.balloons()
                    
                    # Afficher les métriques
                    st.markdown("### 📊 Métriques de Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Efficacité", f"{new_tool['performance']['effectiveness']}%")
                    with col2:
                        st.metric("Détection", f"{new_tool['performance']['detection_rate']}%")
                    with col3:
                        st.metric("Faux Positifs", f"{new_tool['performance']['false_positive']}%")
                    with col4:
                        st.metric("Temps Réponse", f"{new_tool['performance']['response_time']}ms")
    
    # Liste des outils créés
    if st.session_state.tools:
        st.markdown("---")
        st.subheader("🗂️ Outils Créés")
        
        for tool in st.session_state.tools:
            with st.expander(f"🔧 {tool['name']} - {tool['status']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** {tool['id']}")
                    st.write(f"**Catégorie:** {tool['category']}")
                    st.write(f"**Créé le:** {tool['created_at']}")
                    st.write(f"**Complexité:** {tool['complexity']}")
                    st.write(f"**Coût:** ${tool['cost']:,}")
                with col2:
                    st.write(f"**Ressources ({len(tool['resources'])}):**")
                    for r in tool['resources'][:3]:
                        st.write(f"  • {r}")
                    if len(tool['resources']) > 3:
                        st.write(f"  ... et {len(tool['resources'])-3} autres")
                    st.write(f"**Menaces ({len(tool['threats'])}):** {', '.join(tool['threats'][:3])}")
                    st.write(f"**Systèmes ({len(tool['systems'])}):** {', '.join(tool['systems'][:3])}")

# ==================== PAGE: SIMULATIONS D'ATTAQUES ====================
elif page == "🧪 Simulations d'Attaques":
    st.header("🧪 Simulateur d'Attaques et Défenses")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎮 Tests de Sécurité Avancés</h3>
        <p>Testez vos défenses contre différents types d'attaques dans un environnement contrôlé.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🚀 Nouvelle Simulation", "📊 Résultats"])
    
    with tab1:
        st.subheader("Configuration de la Simulation")
        
        with st.form("simulation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                sim_name = st.text_input("Nom de la simulation*", placeholder="Ex: Test Protection Ransomware")
                
                target_system = st.selectbox(
                    "Système cible*",
                    ["CLASSIC_COMPUTER", "CLASSIC_SERVER", "AI_SYSTEM", "QUANTUM_COMPUTER", 
                     "BIOLOGICAL_COMPUTER", "IOT_DEVICES", "CLOUD_INFRASTRUCTURE"]
                )
                
                attack_intensity = st.selectbox(
                    "Intensité de l'attaque",
                    ["Low", "Medium", "High", "Critical"]
                )
            
            with col2:
                duration = st.slider("Durée de la simulation (secondes)", 10, 300, 60)
                
                st.markdown("**Scénarios d'attaque:**")
                attack_scenarios = st.multiselect(
                    "Sélectionner les types d'attaques",
                    ["CLASSIC_MALWARE", "ADVANCED_PERSISTENT", "AI_ADVERSARIAL", "RANSOMWARE",
                     "PHISHING", "DDoS", "QUANTUM_CRYPTANALYSIS", "BIO_CONTAMINATION"],
                    default=["CLASSIC_MALWARE", "RANSOMWARE"]
                )
            
            st.markdown("---")
            st.markdown("**Outils de défense à tester:**")
            
            if not st.session_state.tools:
                st.warning("Aucun outil créé. Créez d'abord des outils à tester.")
                defense_tools = []
            else:
                defense_tools = []
                col1, col2, col3 = st.columns(3)
                for i, tool in enumerate(st.session_state.tools):
                    with [col1, col2, col3][i % 3]:
                        if st.checkbox(f"{tool['name']}", key=f"sim_tool_{tool['id']}"):
                            defense_tools.append(tool['id'])
            
            st.markdown("---")
            
            submitted = st.form_submit_button("🎯 Lancer la Simulation", use_container_width=True)
            
            if submitted:
                if not sim_name:
                    st.error("❌ Veuillez donner un nom à la simulation")
                elif len(attack_scenarios) == 0:
                    st.error("❌ Veuillez sélectionner au moins un scénario d'attaque")
                elif len(defense_tools) == 0:
                    st.error("❌ Veuillez sélectionner au moins un outil de défense")
                else:
                    # Simulation du processus
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    import time
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 20:
                            status_text.text(f"Initialisation de l'environnement... {i}%")
                        elif i < 50:
                            status_text.text(f"Lancement des attaques... {i}%")
                        elif i < 80:
                            status_text.text(f"Évaluation des défenses... {i}%")
                        else:
                            status_text.text(f"Compilation des résultats... {i}%")
                        time.sleep(duration / 200)
                    
                    status_text.text("✅ Simulation terminée!")
                    
                    # Générer les résultats
                    num_attacks = int(100 * len(attack_scenarios) * (duration / 60))
                    defense_effectiveness = 0.75 + (len(defense_tools) * 0.05)
                    
                    attacks_detected = int(num_attacks * defense_effectiveness * np.random.uniform(0.9, 1.1))
                    attacks_blocked = int(attacks_detected * defense_effectiveness * np.random.uniform(0.85, 0.95))
                    attacks_successful = num_attacks - attacks_blocked
                    false_positives = int(num_attacks * 0.02)
                    
                    sim_result = {
                        "id": f"sim_{len(st.session_state.simulations) + 1}",
                        "name": sim_name,
                        "target_system": target_system,
                        "attack_scenarios": attack_scenarios,
                        "defense_tools": defense_tools,
                        "intensity": attack_intensity,
                        "duration": duration,
                        "timestamp": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "results": {
                            "total_attacks": num_attacks,
                            "detected": attacks_detected,
                            "blocked": attacks_blocked,
                            "successful": attacks_successful,
                            "false_positives": false_positives,
                            "detection_rate": round(attacks_detected / num_attacks * 100, 2),
                            "blocking_rate": round(attacks_blocked / num_attacks * 100, 2),
                            "success_rate": round(attacks_blocked / num_attacks * 100, 2)
                        },
                        "system_impact": {
                            "availability": round(100 - (attacks_successful / num_attacks * 50), 1),
                            "integrity": round(100 - (attacks_successful / num_attacks * 30), 1),
                            "confidentiality": round(100 - (attacks_successful / num_attacks * 40), 1)
                        }
                    }
                    
                    st.session_state.simulations.append(sim_result)
                    st.session_state.threats_blocked += attacks_blocked
                    
                    st.success("✅ Simulation complétée avec succès!")
                    st.balloons()
                    
                    # Afficher les résultats
                    st.markdown("### 📊 Résultats de la Simulation")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Attaques Totales", f"{num_attacks:,}")
                    with col2:
                        st.metric("Détectées", f"{attacks_detected:,}", 
                                 f"{sim_result['results']['detection_rate']:.1f}%")
                    with col3:
                        st.metric("Bloquées", f"{attacks_blocked:,}",
                                 f"{sim_result['results']['blocking_rate']:.1f}%")
                    with col4:
                        st.metric("Réussies", f"{attacks_successful:,}",
                                 delta_color="inverse")
                    
                    # Graphique d'impact système
                    st.markdown("### 🎯 Impact sur le Système")
                    
                    impact = sim_result['system_impact']
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Disponibilité', 'Intégrité', 'Confidentialité'],
                            y=[impact['availability'], impact['integrity'], impact['confidentiality']],
                            marker_color=['#00f2fe', '#667eea', '#f093fb'],
                            text=[f"{impact['availability']:.1f}%", f"{impact['integrity']:.1f}%", 
                                  f"{impact['confidentiality']:.1f}%"],
                            textposition='auto'
                        )
                    ])
                    fig.update_layout(
                        title="Préservation des Propriétés de Sécurité",
                        yaxis_title="Score (%)",
                        yaxis_range=[0, 100],
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Historique des Simulations")
        
        if not st.session_state.simulations:
            st.info("Aucune simulation n'a encore été effectuée.")
        else:
            for sim in reversed(st.session_state.simulations):
                with st.expander(f"🧪 {sim['name']} - {sim['timestamp']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**ID:** {sim['id']}")
                        st.write(f"**Système:** {sim['target_system']}")
                        st.write(f"**Intensité:** {sim['intensity']}")
                        st.write(f"**Durée:** {sim['duration']}s")
                        st.write(f"**Scénarios:** {len(sim['attack_scenarios'])}")
                    
                    with col2:
                        st.metric("Taux de Détection", f"{sim['results']['detection_rate']}%")
                        st.metric("Taux de Blocage", f"{sim['results']['blocking_rate']}%")
                        st.metric("Taux de Succès Défense", f"{sim['results']['success_rate']}%")
                    
                    # Graphique comparatif
                    fig = go.Figure(data=[
                        go.Bar(name='Attaques', x=['Total', 'Détectées', 'Bloquées', 'Réussies'],
                              y=[sim['results']['total_attacks'], sim['results']['detected'],
                                 sim['results']['blocked'], sim['results']['successful']],
                              marker_color=['#667eea', '#00f2fe', '#00c853', '#f5576c'])
                    ])
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: ANALYTICS SÉCURITÉ ====================
elif page == "📊 Analytics Sécurité":
    st.header("📊 Analytics et Tableaux de Bord")
    
    st.markdown("""
    <div class="info-box">
        <h3>📈 Vue d'Ensemble Complète</h3>
        <p>Analyses détaillées de votre posture de sécurité.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # KPIs globaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Outils Déployés", len(st.session_state.tools), "+2")
    with col2:
        st.metric("Stratégies Actives", len(st.session_state.strategies), "+1")
    with col3:
        st.metric("Simulations", len(st.session_state.simulations), "+3")
    with col4:
        st.metric("Score Sécurité", f"{st.session_state.security_score:.1f}%", "+2.3%")
    with col5:
        st.metric("Menaces Bloquées", f"{st.session_state.threats_blocked:,}", "+47")
    
    st.markdown("---")
    
    # Graphiques analytiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution des Outils par Catégorie")
        
        if st.session_state.tools:
            categories = {}
            for tool in st.session_state.tools:
                cat = tool['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            fig = go.Figure(data=[
                go.Pie(labels=list(categories.keys()), values=list(categories.values()),
                      hole=0.4, marker_colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'])
            ])
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun outil créé")
    
    with col2:
        st.subheader("📈 Performance des Simulations")
        
        if st.session_state.simulations:
            sim_names = [s['name'][:20] for s in st.session_state.simulations]
            success_rates = [s['results']['success_rate'] for s in st.session_state.simulations]
            
            fig = go.Figure(data=[
                go.Bar(x=sim_names, y=success_rates, marker_color='#00f2fe',
                      text=[f"{sr:.1f}%" for sr in success_rates], textposition='auto')
            ])
            fig.update_layout(height=350, yaxis_title="Taux de Succès (%)", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune simulation effectuée")
    
    st.markdown("---")
    
    # Tableaux de données
    tab1, tab2, tab3 = st.tabs(["🔧 Outils", "📋 Stratégies", "🧪 Simulations"])
    
    with tab1:
        if st.session_state.tools:
            tools_df = pd.DataFrame([
                {
                    "Nom": t['name'],
                    "Catégorie": t['category'],
                    "Efficacité": f"{t['performance']['effectiveness']}%",
                    "Détection": f"{t['performance']['detection_rate']}%",
                    "Faux Positifs": f"{t['performance']['false_positive']}%",
                    "Coût": f"${t['cost']:,}"
                }
                for t in st.session_state.tools
            ])
            st.dataframe(tools_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucun outil créé")
    
    with tab2:
        if st.session_state.strategies:
            strategies_df = pd.DataFrame([
                {
                    "Nom": s['name'],
                    "Priorité": s['priority'],
                    "Outils": len(s['tools']),
                    "Systèmes": len(s['target_systems']),
                    "Menaces": len(s['threat_coverage']),
                    "Progression": f"{s['current_step']}/6",
                    "Budget": f"${s['budget']:,}"
                }
                for s in st.session_state.strategies
            ])
            st.dataframe(strategies_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune stratégie créée")
    
    with tab3:
        if st.session_state.simulations:
            simulations_df = pd.DataFrame([
                {
                    "Nom": s['name'],
                    "Système": s['target_system'],
                    "Attaques": s['results']['total_attacks'],
                    "Bloquées": s['results']['blocked'],
                    "Taux Succès": f"{s['results']['success_rate']}%",
                    "Date": s['timestamp']
                }
                for s in st.session_state.simulations
            ])
            st.dataframe(simulations_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune simulation effectuée")

# ==================== PAGE: WORKSPACE ====================
elif page == "🎨 Workspace":
    st.header("🎨 Workspace Personnel")
    
    tab1, tab2, tab3 = st.tabs(["📝 Notes", "📁 Projets", "📥 Export/Import"])
    
    with tab1:
        st.subheader("📝 Notes de Sécurité")
        
        if 'notes' not in st.session_state:
            st.session_state.notes = []
        
        with st.form("note_form"):
            note_title = st.text_input("Titre")
            note_content = st.text_area("Contenu", height=150)
            note_tags = st.text_input("Tags (séparés par virgules)")
            
            if st.form_submit_button("💾 Enregistrer"):
                if note_title and note_content:
                    st.session_state.notes.append({
                        "id": len(st.session_state.notes) + 1,
                        "title": note_title,
                        "content": note_content,
                        "tags": [t.strip() for t in note_tags.split(',') if t.strip()],
                        "created_at": datetime.now().strftime('%d/%m/%Y %H:%M')
                    })
                    st.success("✅ Note enregistrée!")
                    st.rerun()
        
        if st.session_state.notes:
            st.markdown("---")
            for note in reversed(st.session_state.notes):
                with st.expander(f"📝 {note['title']} - {note['created_at']}"):
                    st.write(note['content'])
                    if note['tags']:
                        st.markdown("**Tags:** " + ", ".join([f"`{tag}`" for tag in note['tags']]))
    
    with tab2:
        st.subheader("📁 Vue d'Ensemble du Projet")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔧 Outils")
            st.metric("Total", len(st.session_state.tools))
        with col2:
            st.markdown("### 📋 Stratégies")
            st.metric("Total", len(st.session_state.strategies))
        with col3:
            st.markdown("### 🧪 Simulations")
            st.metric("Total", len(st.session_state.simulations))
    
    with tab3:
        st.subheader("📥 Export et Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Exporter")
            
            export_data = {
                'tools': st.session_state.tools,
                'strategies': st.session_state.strategies,
                'simulations': st.session_state.simulations,
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'version': '2.0'
                }
            }
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="💾 Télécharger",
                data=json_str,
                file_name=f"cybersecurity_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            st.markdown("### 📥 Importer")
            
            uploaded_file = st.file_uploader("Fichier JSON", type=['json'])
            
            if uploaded_file is not None:
                try:
                    import_data = json.load(uploaded_file)
                    
                    st.write("**Aperçu:**")
                    st.write(f"• Outils: {len(import_data.get('tools', []))}")
                    st.write(f"• Stratégies: {len(import_data.get('strategies', []))}")
                    st.write(f"• Simulations: {len(import_data.get('simulations', []))}")
                    
                    if st.button("✅ Importer", use_container_width=True):
                        st.session_state.tools.extend(import_data.get('tools', []))
                        st.session_state.strategies.extend(import_data.get('strategies', []))
                        st.session_state.simulations.extend(import_data.get('simulations', []))
                        st.success("✅ Données importées!")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")

# ==================== PAGE: CONFIGURATIONS ====================
elif page == "⚙️ Configurations":
    st.header("⚙️ Configurations du Système")
    
    tab1, tab2, tab3 = st.tabs(["🔔 Notifications", "📊 Performance", "🔒 Sécurité"])
    
    with tab1:
        st.subheader("🔔 Paramètres de Notification")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_alerts = st.checkbox("Activer les alertes", value=True)
            alert_level = st.select_slider(
                "Niveau minimum:",
                options=["Info", "Avertissement", "Erreur", "Critique"],
                value="Avertissement"
            )
            
            alert_channels = st.multiselect(
                "Canaux de notification:",
                ["Email", "SMS", "Slack", "Dashboard"],
                default=["Dashboard"]
            )
        
        with col2:
            notification_types = st.multiselect(
                "Types d'alertes:",
                ["Attaques détectées", "Menaces bloquées", "Outils déployés", 
                 "Stratégies actives", "Erreurs système"],
                default=["Attaques détectées", "Menaces bloquées"]
            )
            
            frequency = st.selectbox(
                "Fréquence des rapports:",
                ["Temps réel", "Horaire", "Quotidien", "Hebdomadaire"]
            )
        
        if st.button("💾 Sauvegarder les Notifications", use_container_width=True):
            st.success("✅ Paramètres enregistrés!")
    
    with tab2:
        st.subheader("📊 Optimisation des Performances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_cpu = st.slider("CPU max (%)", 0, 100, 80)
            max_memory = st.slider("Mémoire max (%)", 0, 100, 75)
            
            scan_frequency = st.selectbox(
                "Fréquence de scan:",
                ["Continu", "Toutes les 5 min", "Toutes les 15 min", "Horaire"]
            )
        
        with col2:
            auto_update = st.checkbox("Mises à jour automatiques", value=True)
            auto_response = st.checkbox("Réponse automatique aux incidents", value=True)
            
            log_retention = st.number_input("Rétention des logs (jours)", 30, 365, 90)
        
        if st.button("💾 Appliquer les Paramètres", use_container_width=True):
            st.success("✅ Configuration appliquée!")
    
    with tab3:
        st.subheader("🔒 Paramètres de Sécurité")
        
        col1, col2 = st.columns(2)
        
        with col1:
            require_auth = st.checkbox("Authentification requise", value=True)
            
            if require_auth:
                auth_method = st.selectbox(
                    "Méthode:",
                    ["Mot de passe", "2FA", "Biométrique", "Token"]
                )
                
                session_timeout = st.number_input("Timeout (minutes)", 5, 480, 60)
            
            encryption = st.selectbox(
                "Niveau de chiffrement:",
                ["Standard (AES-128)", "Fort (AES-256)", "Quantique"]
            )
        
        with col2:
            audit_logging = st.checkbox("Logs d'audit", value=True)
            
            if audit_logging:
                audit_level = st.selectbox(
                    "Niveau d'audit:",
                    ["Minimal", "Standard", "Complet", "Forensique"]
                )
            
            backup_frequency = st.selectbox(
                "Fréquence de sauvegarde:",
                ["Horaire", "Quotidien", "Hebdomadaire"]
            )
            
            compliance = st.multiselect(
                "Conformité:",
                ["GDPR", "ISO 27001", "NIST", "SOC 2", "HIPAA"],
                default=["ISO 27001"]
            )
        
        if st.button("💾 Sauvegarder la Sécurité", use_container_width=True):
            st.success("✅ Paramètres de sécurité enregistrés!")

# ==================== PAGES ADDITIONNELLES (stubs) ====================
elif page == "🤖 Ressources IA/ML":
    st.header("🤖 Ressources de Sécurité IA/ML")
    
    st.markdown("""
    <div class="info-box">
        <h3>🧠 Intelligence Artificielle pour la Cybersécurité</h3>
        <p>5 systèmes IA avancés pour la détection et la prévention des menaces</p>
    </div>
    """, unsafe_allow_html=True)
    
    ai_resources = [
        {"name": "Détecteur d'Anomalies IA", "accuracy": 96, "type": "Autoencoder + Isolation Forest"},
        {"name": "Prédicteur de Menaces IA", "accuracy": 94, "type": "Transformer + LSTM"},
        {"name": "Défense Adversariale IA", "accuracy": 92, "type": "Adversarial Training"},
        {"name": "Classificateur Malware ML", "accuracy": 98, "type": "Random Forest + DL"},
        {"name": "Détecteur Phishing NLP", "accuracy": 97, "type": "BERT + CNN"}
    ]
    
    for resource in ai_resources:
        with st.expander(f"🤖 {resource['name']} - {resource['accuracy']}% de précision"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Type:** {resource['type']}")
                st.write(f"**Précision:** {resource['accuracy']}%")
            with col2:
                st.metric("Score IA", resource['accuracy'])

elif page == "⚛️ Ressources Quantiques":
    st.header("⚛️ Ressources de Sécurité Quantique")
    
    st.markdown("""
    <div class="info-box">
        <h3>🔐 Cryptographie Quantique et Post-Quantique</h3>
        <p>5 systèmes quantiques pour une sécurité du futur</p>
    </div>
    """, unsafe_allow_html=True)
    
    quantum_resources = [
        {"name": "Système QKD", "security": "Information-Theoretic", "rate": "10 Mbps"},
        {"name": "Cryptographie Post-Quantique", "security": "NIST Level 5", "rate": "1000 keys/s"},
        {"name": "QRNG", "security": "True Randomness", "rate": "100 Mbps"},
        {"name": "Authentification Quantique", "security": "Unconditional", "rate": "1 Mbps"},
        {"name": "Stéganographie Quantique", "security": "Undetectable", "rate": "N/A"}
    ]
    
    for resource in quantum_resources:
        with st.expander(f"⚛️ {resource['name']} - {resource['security']}"):
            st.write(f"**Niveau de sécurité:** {resource['security']}")
            st.write(f"**Taux:** {resource['rate']}")

elif page == "🧬 Ressources Biologiques":
    st.header("🧬 Ressources de Sécurité Biologique")
    
    st.markdown("""
    <div class="info-box">
        <h3>🔬 Biocomputing et Sécurité Moléculaire</h3>
        <p>5 systèmes biologiques pour protéger les ordinateurs biomoléculaires</p>
    </div>
    """, unsafe_allow_html=True)
    
    bio_resources = [
        {"name": "Pare-feu ADN", "sensitivity": 99.9, "type": "Enzyme-Based"},
        {"name": "Détecteur Contamination", "sensitivity": 99.99, "type": "Biosensor Array"},
        {"name": "Correction d'Erreurs Bio", "sensitivity": 99.95, "type": "DNA Proofreading"},
        {"name": "Authentification Biomoléculaire", "sensitivity": 99.98, "type": "DNA Barcode"},
        {"name": "Détecteur d'Intrusion Bio", "sensitivity": 99.7, "type": "Molecular Sensor"}
    ]
    
    for resource in bio_resources:
        with st.expander(f"🧬 {resource['name']} - {resource['sensitivity']}% de sensibilité"):
            st.write(f"**Type:** {resource['type']}")
            st.write(f"**Sensibilité:** {resource['sensitivity']}%")

elif page == "✅ Gestion des Étapes":
    st.header("✅ Gestion des Étapes de Stratégies")
    
    if not st.session_state.strategies:
        st.warning("⚠️ Aucune stratégie créée. Créez d'abord une stratégie.")
    else:
        strategy_names = [s['name'] for s in st.session_state.strategies]
        selected_strategy_name = st.selectbox("Choisir une stratégie:", strategy_names)
        
        selected_strategy = next(s for s in st.session_state.strategies if s['name'] == selected_strategy_name)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Statut", selected_strategy['status'])
        with col2:
            st.metric("Étape Actuelle", f"{selected_strategy['current_step']}/6")
        with col3:
            completed = sum([1 for step in selected_strategy['steps'] if step['validated']])
            st.metric("Complétées", f"{completed}/6")
        
        st.markdown("---")
        st.subheader("📋 Étapes de la Stratégie")
        
        for i, step in enumerate(selected_strategy['steps']):
            col1, col2, col3 = st.columns([3, 2, 2])
            
            with col1:
                if step['validated']:
                    st.success(f"✅ Étape {step['num']}: {step['name']}")
                elif step['num'] == selected_strategy['current_step']:
                    st.info(f"▶️ Étape {step['num']}: {step['name']} (En cours)")
                else:
                    st.write(f"⏸️ Étape {step['num']}: {step['name']} (En attente)")
            
            with col2:
                st.write(f"**Statut:** {step['status']}")
            
            with col3:
                if not step['validated'] and step['num'] == selected_strategy['current_step']:
                    if st.button(f"Valider", key=f"validate_{selected_strategy['id']}_{step['num']}"):
                        step['validated'] = True
                        step['status'] = "Complétée"
                        
                        if step['num'] < 6:
                            selected_strategy['current_step'] += 1
                            selected_strategy['steps'][step['num']]['status'] = "En cours"
                        else:
                            selected_strategy['status'] = "Déployée"
                        
                        st.success(f"✅ Étape {step['num']} validée!")
                        st.rerun()
            
            st.markdown("---")
        
        # Graphique de progression
        progress_value = (completed / 6) * 100
        st.progress(progress_value / 100)
        st.write(f"**{progress_value:.0f}% complété**")

elif page == "🌐 Environnements Virtuels":
    st.header("🌐 Environnements Virtuels de Test")
    
    st.markdown("""
    <div class="info-box">
        <h3>🖥️ Laboratoires de Sécurité Virtuels</h3>
        <p>4 environnements isolés pour tester vos défenses en toute sécurité</p>
    </div>
    """, unsafe_allow_html=True)
    
    environments = [
        {
            "name": "Réseau d'Entreprise Virtuel",
            "type": "Classique",
            "components": ["Firewall", "Serveurs", "Workstations", "Databases"],
            "vulnerabilities": 25,
            "security_score": 75
        },
        {
            "name": "Laboratoire IA Sécurisé",
            "type": "IA",
            "components": ["ML Models", "Training Data", "Inference Engines"],
            "vulnerabilities": 15,
            "security_score": 82
        },
        {
            "name": "Installation Quantique Simulée",
            "type": "Quantique",
            "components": ["Quantum Processors", "Control Systems", "Cryogenics"],
            "vulnerabilities": 8,
            "security_score": 90
        },
        {
            "name": "Laboratoire Biocomputing",
            "type": "Biologique",
            "components": ["DNA Storage", "Enzymatic Processors", "Biosensors"],
            "vulnerabilities": 12,
            "security_score": 85
        }
    ]
    
    for env in environments:
        with st.expander(f"🌐 {env['name']} - Score: {env['security_score']}%"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {env['type']}")
                st.write(f"**Composants:** {len(env['components'])}")
                st.write(f"**Vulnérabilités:** {env['vulnerabilities']}")
            
            with col2:
                st.metric("Score de Sécurité", f"{env['security_score']}%")
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=env['security_score'],
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#667eea"}},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ))
                fig.update_layout(height=200)
                st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(f"🚀 Démarrer", key=f"start_{env['name']}"):
                    st.success("Environnement démarré!")
            with col_b:
                if st.button(f"📊 Analyser", key=f"analyze_{env['name']}"):
                    st.info("Analyse en cours...")

elif page == "🔍 Intelligence des Menaces":
    st.header("🔍 Intelligence des Menaces")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Threat Intelligence</h3>
        <p>Restez informé des dernières menaces et vulnérabilités</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    threats = [
        {
            "name": "Ransomware",
            "severity": "Critique",
            "prevalence": 85,
            "targets": ["Ordinateurs", "Serveurs", "Cloud"],
            "trend": "↗️ En hausse"
        },
        {
            "name": "Attaques IA Adversariales",
            "severity": "Élevé",
            "prevalence": 65,
            "targets": ["Systèmes IA"],
            "trend": "↗️ En hausse"
        },
        {
            "name": "Cryptanalyse Quantique",
            "severity": "Futur Critique",
            "prevalence": 10,
            "targets": ["Tous systèmes"],
            "trend": "→ Stable"
        }
    ]
    
    for threat in threats:
        severity_class = {
            "Critique": "critical-badge",
            "Élevé": "high-badge",
            "Futur Critique": "high-badge"
        }
        
        with st.expander(f"⚠️ {threat['name']} - {threat['severity']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<span class='threat-badge {severity_class[threat['severity']]}'>{threat['severity']}</span>", 
                          unsafe_allow_html=True)
                st.write(f"**Prévalence:** {threat['prevalence']}%")
                st.write(f"**Tendance:** {threat['trend']}")
            
            with col2:
                st.write(f"**Systèmes ciblés:**")
                for target in threat['targets']:
                    st.write(f"• {target}")
            
            st.progress(threat['prevalence'] / 100)

# ==================== PAGE: CRÉER UNE STRATÉGIE ====================
elif page == "📋 Créer une Stratégie":
    st.header("📋 Créateur de Stratégies de Sécurité")
    
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Créez Votre Stratégie de Défense</h3>
        <p>Combinez plusieurs outils pour une protection multicouche complète.</p>
    </div>
    """, unsafe_allow_html=True)
   
    if not st.session_state.tools:
        st.warning("⚠️ Vous devez d'abord créer des outils avant de pouvoir créer une stratégie.")
        st.info("👉 Rendez-vous dans 'Créer un Outil' pour commencer.")
    else:
        with st.form("strategy_creation_form"):
            st.subheader("1️⃣ Informations de Base")
            
            col1, col2 = st.columns(2)
            with col1:
                strategy_name = st.text_input("Nom de la stratégie*", placeholder="Ex: Protection Entreprise 360°")
            with col2:
                strategy_priority = st.selectbox("Priorité", ["Low", "Medium", "High", "Critical"])
            
            strategy_description = st.text_area(
                "Description détaillée*",
                placeholder="Décrivez la stratégie, ses objectifs et sa portée...",
                height=100
            )
            
            st.markdown("---")
            st.subheader("2️⃣ Systèmes à Protéger")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sys_classic = st.checkbox("Ordinateurs Classiques", value=True, key="strat_sys_classic")
                sys_server = st.checkbox("Serveurs", value=True, key="strat_sys_server")
            with col2:
                sys_ai = st.checkbox("Systèmes IA", key="strat_sys_ai")
                sys_quantum = st.checkbox("Ordinateurs Quantiques", key="strat_sys_quantum")
            with col3:
                sys_bio = st.checkbox("Ordinateurs Biologiques", key="strat_sys_bio")
                sys_iot = st.checkbox("Appareils IoT", key="strat_sys_iot")
            with col4:
                sys_cloud = st.checkbox("Infrastructure Cloud", key="strat_sys_cloud")
                sys_hybrid = st.checkbox("Systèmes Hybrides", key="strat_sys_hybrid")
            
            st.markdown("---")
            st.subheader("3️⃣ Couverture des Menaces")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                threat_malware = st.checkbox("Malware Classique", value=True, key="strat_threat_malware")
                threat_apt = st.checkbox("APT", value=True, key="strat_threat_apt")
                threat_ransomware = st.checkbox("Ransomware", value=True, key="strat_threat_ransom")
            with col2:
                threat_ai_adv = st.checkbox("Attaques IA Adversariales", key="strat_threat_ai_adv")
                threat_ai_poison = st.checkbox("Empoisonnement IA", key="strat_threat_ai_poison")
                threat_phishing = st.checkbox("Phishing", key="strat_threat_phishing")
            with col3:
                threat_quantum = st.checkbox("Cryptanalyse Quantique", key="strat_threat_quantum")
                threat_bio = st.checkbox("Contamination Bio", key="strat_threat_bio")
                threat_ddos = st.checkbox("DDoS", key="strat_threat_ddos")
            
            st.markdown("---")
            st.subheader("4️⃣ Sélection des Outils")
            
            st.info("💡 Sélectionnez les outils à inclure dans votre stratégie de défense")
            
            selected_tools = []
            col1, col2, col3 = st.columns(3)
            for i, tool in enumerate(st.session_state.tools):
                with [col1, col2, col3][i % 3]:
                    if st.checkbox(f"{tool['name']}", key=f"strat_tool_{tool['id']}"):
                        selected_tools.append(tool)
            
            st.markdown("---")
            st.subheader("5️⃣ Budget et Timeline")
            
            col1, col2 = st.columns(2)
            with col1:
                budget = st.number_input("Budget Total ($)", 10000, 10000000, 100000, 10000)
            with col2:
                timeline = st.number_input("Timeline (jours)", 30, 365, 90, 10)
            
            st.markdown("---")
            
            submitted = st.form_submit_button("🚀 Créer la Stratégie", use_container_width=True)
            
            if submitted:
                if not strategy_name or not strategy_description:
                    st.error("❌ Veuillez remplir tous les champs obligatoires")
                elif len(selected_tools) == 0:
                    st.error("❌ Veuillez sélectionner au moins un outil")
                else:
                    # Collecter les systèmes
                    target_systems = []
                    if sys_classic: target_systems.append("Ordinateurs Classiques")
                    if sys_server: target_systems.append("Serveurs")
                    if sys_ai: target_systems.append("Systèmes IA")
                    if sys_quantum: target_systems.append("Ordinateurs Quantiques")
                    if sys_bio: target_systems.append("Ordinateurs Biologiques")
                    if sys_iot: target_systems.append("Appareils IoT")
                    if sys_cloud: target_systems.append("Infrastructure Cloud")
                    if sys_hybrid: target_systems.append("Systèmes Hybrides")
                    
                    # Collecter les menaces
                    threat_coverage = []
                    if threat_malware: threat_coverage.append("Malware Classique")
                    if threat_apt: threat_coverage.append("APT")
                    if threat_ransomware: threat_coverage.append("Ransomware")
                    if threat_ai_adv: threat_coverage.append("Attaques IA Adversariales")
                    if threat_ai_poison: threat_coverage.append("Empoisonnement IA")
                    if threat_phishing: threat_coverage.append("Phishing")
                    if threat_quantum: threat_coverage.append("Cryptanalyse Quantique")
                    if threat_bio: threat_coverage.append("Contamination Bio")
                    if threat_ddos: threat_coverage.append("DDoS")
                    
                    # Créer la stratégie
                    new_strategy = {
                        "id": f"strat_{len(st.session_state.strategies) + 1}",
                        "name": strategy_name,
                        "description": strategy_description,
                        "priority": strategy_priority,
                        "target_systems": target_systems,
                        "threat_coverage": threat_coverage,
                        "tools": [tool['id'] for tool in selected_tools],
                        "tool_names": [tool['name'] for tool in selected_tools],
                        "budget": budget,
                        "timeline": timeline,
                        "steps": [
                            {"num": 1, "name": "Évaluation", "status": "En attente", "validated": False},
                            {"num": 2, "name": "Planification", "status": "En attente", "validated": False},
                            {"num": 3, "name": "Déploiement", "status": "En attente", "validated": False},
                            {"num": 4, "name": "Tests", "status": "En attente", "validated": False},
                            {"num": 5, "name": "Monitoring", "status": "En attente", "validated": False},
                            {"num": 6, "name": "Optimisation", "status": "En attente", "validated": False}
                        ],
                        "current_step": 1,
                        "status": "Créée",
                        "created_at": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                        "risk_level": "Moyen",
                        "effectiveness": {
                            threat: round(85 + np.random.uniform(-5, 10), 1)
                            for threat in threat_coverage
                        }
                    }
                    
                    st.session_state.strategies.append(new_strategy)
                    
                    st.success(f"✅ Stratégie '{strategy_name}' créée avec succès!")
                    st.balloons()
                    
                    # Afficher le résumé
                    st.markdown("### 📊 Résumé de la Stratégie")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Systèmes", len(target_systems))
                    with col2:
                        st.metric("Menaces", len(threat_coverage))
                    with col3:
                        st.metric("Outils", len(selected_tools))
                    with col4:
                        st.metric("Risque", new_strategy['risk_level'])
                    
                    # Graphique radar de l'efficacité
                    if new_strategy['effectiveness']:
                        fig = go.Figure(data=go.Scatterpolar(
                            r=list(new_strategy['effectiveness'].values()),
                            theta=list(new_strategy['effectiveness'].keys()),
                            fill='toself',
                            marker_color='#667eea'
                        ))
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=False,
                            height=400,
                            title="Efficacité par Type de Menace"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Liste des stratégies créées
        if st.session_state.strategies:
            st.markdown("---")
            st.subheader("🗂️ Stratégies Créées")
            
            for strategy in st.session_state.strategies:
                with st.expander(f"📋 {strategy['name']} - {strategy['status']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**ID:** {strategy['id']}")
                        st.write(f"**Priorité:** {strategy['priority']}")
                        st.write(f"**Créée le:** {strategy['created_at']}")
                        st.write(f"**Étape actuelle:** {strategy['current_step']}/6")
                        st.write(f"**Budget:** ${strategy['budget']:,}")
                    with col2:
                        st.write(f"**Systèmes ({len(strategy['target_systems'])}):**")
                        for sys in strategy['target_systems'][:3]:
                            st.write(f"  • {sys}")
                        st.write(f"**Menaces ({len(strategy['threat_coverage'])}):**")
                        for threat in strategy['threat_coverage'][:3]:
                            st.write(f"  • {threat}")
                        st.write(f"**Outils:** {len(strategy['tools'])}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 1rem; margin-top: 2rem;">
    <h3 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🛡️ Moteur IA de Cybersécurité Multi-Domaines V2.0
    </h3>
    <p style="margin: 1rem 0;">
        <strong>Protection Complète:</strong> Classique | IA/ML | Quantique | Biologique
    </p>
    <p style="margin: 0.5rem 0;">
        <strong>Ressources:</strong> 20 systèmes de sécurité avancés | 6 étapes de déploiement
    </p>
    <p style="margin: 0.5rem 0;">
        <strong>Capacités:</strong> Détection | Prévention | Réponse | Monitoring 24/7
    </p>
    <p style="margin: 1rem 0; font-size: 0.9rem; color: #888;">
        Version 2.0.0 | © 2025 | Architecture Robuste et Complète
    </p>
    <p style="margin: 0;">
        <span style="display: inline-block; margin: 0 0.5rem;">🛡️ Classique</span>
        <span style="display: inline-block; margin: 0 0.5rem;">🤖 IA/ML</span>
        <span style="display: inline-block; margin: 0 0.5rem;">⚛️ Quantique</span>
        <span style="display: inline-block; margin: 0 0.5rem;">🧬 Biologique</span>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔗 Liens Rapides")
    st.markdown("• [Documentation](https://docs.cybersec-ai.io)")
    st.markdown("• [API Reference](https://api.cybersec-ai.io)")
    st.markdown("• [Support](https://support.cybersec-ai.io)")
    st.markdown("• [Threat Intel Feed](https://intel.cybersec-ai.io)")
    
    st.markdown("---")
    st.caption("Propulsé par 🛡️ CyberSec AI V2.0")
    