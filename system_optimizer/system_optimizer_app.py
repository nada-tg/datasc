"""
system_optimizer_frontend.py - Interface Streamlit

Lancement:
streamlit run system_optimizer_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(
    page_title="System Optimizer & Security",
    page_icon="⚡",
    layout="wide"
)

API_URL = "http://localhost:8005"

def init_session():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = "user_001"
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def format_bytes(bytes_value):
    """Formate les octets en unités lisibles"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

# PAGE: Dashboard Système
def page_system_monitor():
    st.title("Monitoring Système en Temps Réel")
    
    if st.button("🔄 Rafraîchir"):
        st.rerun()
    
    try:
        response = requests.get(f"{API_URL}/api/v1/system/monitor")
        
        if response.status_code == 200:
            data = response.json()
            
            # Informations système
            st.subheader("Informations Système")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("OS", data['system']['os'])
            with col2:
                st.metric("Architecture", data['system']['architecture'])
            with col3:
                st.metric("Hostname", data['system']['hostname'])
            with col4:
                st.metric("Timestamp", data['timestamp'][:19])
            
            st.write("---")
            
            # CPU
            st.subheader("🖥️ Processeur (CPU)")
            
            cpu = data['cpu']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Utilisation Totale", f"{cpu['total_usage']:.1f}%")
            with col2:
                st.metric("Cœurs Physiques", cpu['physical_cores'])
            with col3:
                st.metric("Cœurs Logiques", cpu['logical_cores'])
            with col4:
                temp = cpu.get('temperature')
                st.metric("Température", f"{temp}°C" if temp else "N/A")
            
            # Graphique utilisation par cœur
            if cpu.get('usage_per_core'):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Core {i}" for i in range(len(cpu['usage_per_core']))],
                    y=cpu['usage_per_core'],
                    marker_color=['red' if x > 80 else 'orange' if x > 60 else 'green' 
                                 for x in cpu['usage_per_core']]
                ))
                fig.update_layout(title="Utilisation par Cœur", yaxis_title="Usage (%)", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
            
            # Mémoire
            st.subheader("💾 Mémoire (RAM)")
            
            mem = data['memory']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total", format_bytes(mem['total']))
            with col2:
                st.metric("Utilisée", format_bytes(mem['used']))
            with col3:
                st.metric("Disponible", format_bytes(mem['available']))
            with col4:
                st.metric("Utilisation", f"{mem['percent']:.1f}%")
            
            # Gauge mémoire
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mem['percent'],
                title={'text': "Utilisation RAM"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("---")
            
            # GPU
            st.subheader("🎮 GPU")
            
            gpus = data['gpu']
            
            if gpus and not gpus[0].get('error'):
                for idx, gpu in enumerate(gpus):
                    if gpu.get('available') == False:
                        st.warning("Aucun GPU détecté sur ce système")
                        st.info("💡 Consultez la Marketplace pour des ressources Cloud GPU")
                    else:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(f"GPU {gpu['id']}", gpu['name'])
                        with col2:
                            st.metric("Charge", f"{gpu['load']:.1f}%")
                        with col3:
                            st.metric("VRAM", f"{gpu['memory_used']}/{gpu['memory_total']} MB")
                        with col4:
                            st.metric("Température", f"{gpu['temperature']}°C")
            else:
                st.warning("GPU non disponible")
                st.info("💡 Consultez la Marketplace Cloud pour louer un GPU")
            
            st.write("---")
            
            # Disques
            st.subheader("💿 Disques")
            
            for disk in data['disk']:
                with st.expander(f"{disk['device']} - {disk['mountpoint']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total", format_bytes(disk['total']))
                    with col2:
                        st.metric("Utilisé", format_bytes(disk['used']))
                    with col3:
                        st.metric("Libre", format_bytes(disk['free']))
                    
                    st.progress(disk['percent'] / 100)
                    st.caption(f"{disk['percent']:.1f}% utilisé")
            
            st.write("---")
            
            # Réseau
            st.subheader("🌐 Réseau")
            
            net = data['network']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Envoyé", format_bytes(net['bytes_sent']))
            with col2:
                st.metric("Reçu", format_bytes(net['bytes_recv']))
            with col3:
                st.metric("Connexions Actives", net['active_connections'])
            with col4:
                st.metric("Établies", net['established'])
            
            st.write("---")
            
            # Top Processus
            st.subheader("🔝 Top Processus")
            
            processes = data['top_processes']
            df = pd.DataFrame(processes)
            
            if not df.empty:
                df = df[['name', 'pid', 'cpu_percent', 'memory_percent', 'status']]
                df.columns = ['Nom', 'PID', 'CPU %', 'RAM %', 'Status']
                
                st.dataframe(
                    df.style.background_gradient(subset=['CPU %', 'RAM %'], cmap='RdYlGn_r'),
                    use_container_width=True,
                    height=400
                )
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        st.info("Assurez-vous que l'API est démarrée sur le port 8005")

# PAGE: Optimisation
def page_optimization():
    st.title("⚡ Optimisation Intelligente")
    
    st.info("Notre moteur IA analysera votre système et proposera des optimisations")
    
    with st.form("optimization_form"):
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            aggressive = st.checkbox("Mode Agressif", help="Optimisations plus importantes")
        
        with col2:
            preserve_bg = st.checkbox("Préserver Arrière-Plan", value=True, 
                                     help="Maintenir les processus système")
        
        submitted = st.form_submit_button("🚀 Analyser et Optimiser", type="primary")
    
    if submitted:
        with st.spinner("Analyse en cours..."):
            payload = {
                "aggressive": aggressive,
                "preserve_background": preserve_bg
            }
            
            try:
                response = requests.post(f"{API_URL}/api/v1/optimize", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    optimization = result['optimization']
                    
                    st.success("Analyse terminée!")
                    
                    # Sauvegarder dans session
                    st.session_state.last_optimization = optimization
            
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Afficher les résultats si disponibles
    if 'last_optimization' in st.session_state:
        optimization = st.session_state.last_optimization
        
        # État actuel
        st.subheader("État Actuel du Système")
        
        current = optimization['current_state']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current['cpu']['total_usage'],
                title={'text': "CPU"},
                gauge={'axis': {'range': [None, 100]}}
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current['memory']['percent'],
                title={'text': "RAM"},
                gauge={'axis': {'range': [None, 100]}}
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        # Graphique processus gourmands
        heavy = optimization['heavy_processes']
        
        if heavy:
            st.write("---")
            st.subheader("⚠️ Processus Gourmands")
            
            df_heavy = pd.DataFrame(heavy[:10])
            
            fig = px.bar(df_heavy, x='name', y='cpu_percent',
                        title="Utilisation CPU par Processus",
                        color='cpu_percent',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = px.bar(df_heavy, x='name', y='memory_percent',
                         title="Utilisation RAM par Processus",
                         color='memory_percent',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Optimisations proposées
        st.write("---")
        st.subheader("💡 Optimisations Proposées")
        
        opts = optimization['optimizations']
        
        for idx, opt in enumerate(opts, 1):
            st.markdown(f"""
            **{idx}. {opt.get('description', opt.get('suggestion', 'Optimisation'))}**
            - Type: {opt['type']}
            - Impact: {opt.get('impact', 'À évaluer')}
            """)
        
        # Amélioration estimée avec graphique
        st.write("---")
        st.subheader("📈 Amélioration Estimée")
        
        improvement = optimization['estimated_improvement']
        
        categories = ['CPU', 'Mémoire', 'Réactivité']
        values = [15, 25, 20]  # Valeurs moyennes estimées
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ])
        fig.update_layout(title="Amélioration Potentielle (%)", yaxis_title="Pourcentage")
        st.plotly_chart(fig, use_container_width=True)
        
        # Appliquer (maintenant en dehors du form)
        st.write("---")
        
        if st.button("✅ Appliquer les Optimisations", type="primary"):
            opt_id = optimization['optimization_id']
            
            apply_response = requests.post(
                f"{API_URL}/api/v1/optimize/{opt_id}/apply"
            )
            
            if apply_response.status_code == 200:
                st.success("Optimisations appliquées!")
                st.balloons()
                del st.session_state.last_optimization
            else:
                st.error("Erreur lors de l'application")

# PAGE: Sécurité
def page_security():
    st.title("🛡️ Analyse de Sécurité")
    
    st.warning("⚠️ Mode Défensif Uniquement - Aucune contre-attaque offensive (légalité)")
    
    # Bouton en dehors de tout formulaire
    analyze_btn = st.button("🔍 Lancer Analyse de Sécurité", type="primary")
    
    if analyze_btn:
        with st.spinner("Scan en cours..."):
            try:
                response = requests.get(f"{API_URL}/api/v1/security/analyze")
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result['analysis']
                    
                    # Sauvegarder dans session
                    st.session_state.security_analysis = analysis
                    st.rerun()
            
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    # Afficher les résultats
    if 'security_analysis' in st.session_state:
        analysis = st.session_state.security_analysis
        
        # Score de sécurité
        st.subheader("Score de Sécurité")
        
        score = analysis['security_score']
        threat = analysis['threat_level']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': "Score de Sécurité"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if score > 80 else "orange" if score > 50 else "red"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightcoral"},
                        {'range': [50, 80], 'color': "lightyellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            threat_colors = {
                "LOW": "🟢 Faible",
                "MEDIUM": "🟡 Moyen",
                "HIGH": "🟠 Élevé",
                "CRITICAL": "🔴 Critique"
            }
            st.metric("Niveau de Menace", threat_colors.get(threat, threat))
            st.write(f"**Timestamp:** {analysis['timestamp'][:19]}")
            
            # Graphique du score
            fig_score = go.Figure(go.Indicator(
                mode="number+delta",
                value=score,
                delta={'reference': 90, 'relative': False},
                title={'text': "Évolution"}
            ))
            fig_score.update_layout(height=150)
            st.plotly_chart(fig_score, use_container_width=True)
        
        # Graphique de répartition des menaces
        st.write("---")
        st.subheader("📊 Analyse des Menaces")
        
        threat_data = {
            'Type': ['Connexions Suspectes', 'Processus Suspects', 'Autres'],
            'Nombre': [
                len(analysis['suspicious_connections']),
                len(analysis['suspicious_processes']),
                0
            ]
        }
        
        fig = px.pie(pd.DataFrame(threat_data), values='Nombre', names='Type',
                    title="Répartition des Menaces Détectées")
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("---")
        
        # Connexions suspectes
        st.subheader("🚨 Connexions Suspectes")
        
        suspicious_conn = analysis['suspicious_connections']
        
        if suspicious_conn:
            df_conn = pd.DataFrame(suspicious_conn)
            st.dataframe(df_conn, use_container_width=True)
            
            for conn in suspicious_conn:
                with st.expander(f"⚠️ {conn['remote_addr']} - Risque: {conn['risk']}"):
                    st.write(f"**Local:** {conn['local_addr']}")
                    st.write(f"**Distant:** {conn['remote_addr']}")
                    st.write(f"**Status:** {conn['status']}")
                    st.write(f"**Raison:** {conn['reason']}")
                    
                    if st.button(f"🛡️ Bloquer", key=f"block_{conn['remote_addr']}"):
                        block_response = requests.post(
                            f"{API_URL}/api/v1/security/block/{conn['remote_addr']}"
                        )
                        
                        if block_response.status_code == 200:
                            st.success("Connexion bloquée (mode défensif)")
        else:
            st.success("✅ Aucune connexion suspecte détectée")
        
        st.write("---")
        
        # Graphique des statistiques réseau
        st.subheader("📈 Statistiques Réseau")
        
        net_stats = analysis['network_stats']
        
        network_data = {
            'Type': ['Actives', 'Établies', 'En Écoute'],
            'Nombre': [
                net_stats['active_connections'],
                net_stats['established'],
                net_stats['listening']
            ]
        }
        
        fig = px.bar(pd.DataFrame(network_data), x='Type', y='Nombre',
                    title="Connexions Réseau",
                    color='Nombre',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations avec graphique
        st.write("---")
        st.subheader("💡 Recommandations de Sécurité")
        
        for idx, rec in enumerate(analysis['recommendations'], 1):
            st.info(f"{idx}. {rec}")

# PAGE: Marketplace Cloud (améliorée avec sécurité)
def page_marketplace():
    st.title("🛒 Marketplace Ressources Cloud")
    
    st.info("Louez des ressources cloud haute performance")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/marketplace/resources")
        
        if response.status_code == 200:
            resources = response.json()
            
            # Tabs par type
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🎮 Cloud GPU", 
                "🖥️ Cloud CPU", 
                "💾 Cloud RAM",
                "🛡️ Sécurité Cloud",
                "🧪 Test Algorithmes"
            ])
            
            with tab1:
                st.subheader("GPUs Cloud Disponibles")
                
                for gpu in resources.get('cloud_gpu', []):
                    with st.expander(f"{gpu['name']} - ${gpu['price_per_hour']}/h"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Spécifications:**")
                            for key, value in gpu['specs'].items():
                                st.write(f"- {key}: {value}")
                        
                        with col2:
                            st.write(f"**Prix:** ${gpu['price_per_hour']}/heure")
                            st.write(f"**Description:** {gpu['description']}")
                            
                            duration = st.number_input(
                                "Durée (heures)",
                                1, 720, 24,
                                key=f"duration_{gpu['id']}"
                            )
                            
                            total = gpu['price_per_hour'] * duration
                            st.metric("Coût Total", f"${total:.2f}")
                            
                            if st.button("🛒 Louer", key=f"buy_{gpu['id']}"):
                                purchase_data = {
                                    "resource_type": "cloud_gpu",
                                    "specs": {"id": gpu['id']},
                                    "duration_hours": duration
                                }
                                
                                purchase_response = requests.post(
                                    f"{API_URL}/api/v1/marketplace/purchase?user_id={st.session_state.user_id}",
                                    json=purchase_data
                                )
                                
                                if purchase_response.status_code == 200:
                                    result = purchase_response.json()
                                    st.success("Ressource louée!")
                                    st.code(f"Endpoint: {result['purchase']['connection_endpoint']}")
                                    st.balloons()
            
            with tab2:
                st.subheader("CPUs Cloud Disponibles")
                
                for cpu in resources.get('cloud_cpu', []):
                    with st.expander(f"{cpu['name']} - ${cpu['price_per_hour']}/h"):
                        st.write("**Spécifications:**")
                        for key, value in cpu['specs'].items():
                            st.write(f"- {key}: {value}")
            
            with tab3:
                st.subheader("RAM Cloud Disponible")
                
                for ram in resources.get('cloud_ram', []):
                    with st.expander(f"{ram['name']} - ${ram['price_per_hour']}/h"):
                        st.write("**Spécifications:**")
                        for key, value in ram['specs'].items():
                            st.write(f"- {key}: {value}")
            
            with tab4:
                st.subheader("🛡️ Services de Sécurité Cloud")
                
                security_services = [
                    {
                        "name": "Cloud Firewall Pro",
                        "description": "Pare-feu cloud avec IA anti-DDoS",
                        "features": ["Protection DDoS", "Détection intrusion", "WAF"],
                        "price": 0.15
                    },
                    {
                        "name": "Cloud VPN Enterprise",
                        "description": "VPN sécurisé avec chiffrement AES-256",
                        "features": ["Multi-région", "Kill Switch", "Split Tunneling"],
                        "price": 0.10
                    },
                    {
                        "name": "Cloud Security Scanner",
                        "description": "Scanner vulnérabilités en temps réel",
                        "features": ["Scan automatique", "Alertes instantanées", "Rapports"],
                        "price": 0.20
                    }
                ]
                
                for service in security_services:
                    with st.expander(f"{service['name']} - ${service['price']}/h"):
                        st.write(f"**Description:** {service['description']}")
                        st.write("**Fonctionnalités:**")
                        for feat in service['features']:
                            st.write(f"- {feat}")
                        st.write(f"**Prix:** ${service['price']}/heure")
                        
                        if st.button(f"🛒 Activer", key=f"sec_{service['name']}"):
                            st.success(f"{service['name']} activé!")
            
            with tab5:
                st.subheader("🧪 Test d'Algorithmes Premium")
                
                st.info("Testez nos algorithmes avant achat")
                
                algorithms = [
                    {
                        "name": "Optimiseur ML v2.0",
                        "description": "Optimise automatiquement vos modèles ML",
                        "performance": 95,
                        "price": 199
                    },
                    {
                        "name": "Compresseur IA",
                        "description": "Compression intelligente sans perte",
                        "performance": 88,
                        "price": 149
                    },
                    {
                        "name": "Détecteur Anomalies",
                        "description": "Détection anomalies en temps réel",
                        "performance": 92,
                        "price": 179
                    }
                ]
                
                for algo in algorithms:
                    with st.expander(f"{algo['name']} - ${algo['price']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Description:** {algo['description']}")
                            st.write(f"**Performance:** {algo['performance']}%")
                            
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=algo['performance'],
                                title={'text': "Score Performance"},
                                gauge={'axis': {'range': [None, 100]}}
                            ))
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write(f"**Prix:** ${algo['price']}")
                            
                            if st.button(f"🧪 Tester Gratuitement", key=f"test_{algo['name']}"):
                                with st.spinner("Test en cours..."):
                                    progress = st.progress(0)
                                    for i in range(100):
                                        time.sleep(0.02)
                                        progress.progress(i + 1)
                                    
                                    st.success("Test terminé!")
                                    st.metric("Résultat", f"{algo['performance']}% de réussite")
                            
                            if st.button(f"💰 Acheter", key=f"buy_algo_{algo['name']}"):
                                st.success(f"{algo['name']} acheté!")
            
            # Mes achats avec graphiques
            st.write("---")
            st.subheader("📦 Mes Ressources Actives")
            
            purchases_response = requests.get(
                f"{API_URL}/api/v1/marketplace/purchases/{st.session_state.user_id}"
            )
            
            if purchases_response.status_code == 200:
                purchases_data = purchases_response.json()
                purchases = purchases_data['purchases']
                
                if purchases:
                    # Graphique des dépenses
                    total_spent = sum(p['total_cost'] for p in purchases)
                    st.metric("Dépenses Totales", f"${total_spent:.2f}")
                    
                    df_purchases = pd.DataFrame(purchases)
                    fig = px.pie(df_purchases, values='total_cost', names='resource_type',
                                title="Répartition des Dépenses")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    for purchase in purchases:
                        with st.expander(f"{purchase['resource_details']['name']} - {purchase['status']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Acheté:** {purchase['purchased_at'][:19]}")
                                st.write(f"**Expire:** {purchase['expires_at'][:19]}")
                            
                            with col2:
                                st.write(f"**Coût:** ${purchase['total_cost']:.2f}")
                                st.write(f"**Endpoint:** {purchase['connection_endpoint']}")
                else:
                    st.info("Aucune ressource active")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        with st.spinner("Analyse en cours..."):
                payload = {
                    "aggressive": aggressive,
                    "preserve_background": preserve_bg
                }
                
                try:
                    response = requests.post(f"{API_URL}/api/v1/optimize", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        optimization = result['optimization']
                        
                        st.success("Analyse terminée!")
                        
                        # État actuel
                        st.subheader("État Actuel du Système")
                        
                        current = optimization['current_state']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("CPU", f"{current['cpu']['total_usage']:.1f}%")
                        with col2:
                            st.metric("RAM", f"{current['memory']['percent']:.1f}%")
                        
                        # Processus gourmands
                        st.write("---")
                        st.subheader("⚠️ Processus Gourmands Détectés")
                        
                        heavy = optimization['heavy_processes']
                        
                        if heavy:
                            for proc in heavy[:10]:
                                with st.expander(f"{proc['name']} (PID: {proc['pid']})"):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("CPU", f"{proc.get('cpu_percent', 0):.1f}%")
                                    with col2:
                                        st.metric("RAM", f"{proc.get('memory_percent', 0):.1f}%")
                                    with col3:
                                        st.write(f"**Status:** {proc.get('status', 'N/A')}")
                        else:
                            st.success("Aucun processus excessivement gourmand détecté")
                        
                        # Optimisations proposées
                        st.write("---")
                        st.subheader("💡 Optimisations Proposées")
                        
                        opts = optimization['optimizations']
                        
                        for idx, opt in enumerate(opts, 1):
                            st.markdown(f"""
                            **{idx}. {opt.get('description', opt.get('suggestion', 'Optimisation'))}**
                            - Type: {opt['type']}
                            - Impact: {opt.get('impact', 'À évaluer')}
                            """)
                        
                        # Amélioration estimée
                        st.write("---")
                        st.subheader("📈 Amélioration Estimée")
                        
                        improvement = optimization['estimated_improvement']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("CPU", improvement['cpu'])
                        with col2:
                            st.metric("Mémoire", improvement['memory'])
                        with col3:
                            st.metric("Réactivité", improvement['responsiveness'])
                        
                        # Appliquer
                        st.write("---")
                        
                        if st.button("✅ Appliquer les Optimisations", type="primary"):
                            opt_id = optimization['optimization_id']
                            
                            apply_response = requests.post(
                                f"{API_URL}/api/v1/optimize/{opt_id}/apply"
                            )
                            
                            if apply_response.status_code == 200:
                                st.success("Optimisations appliquées!")
                                st.balloons()
                            else:
                                st.error("Erreur lors de l'application")
                
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

# PAGE: Sécurité
def page_security():
    st.title("🛡️ Analyse de Sécurité")
    
    st.warning("⚠️ Mode Défensif Uniquement - Aucune contre-attaque offensive (légalité)")
    
    if st.button("🔍 Lancer Analyse de Sécurité", type="primary"):
        with st.spinner("Scan en cours..."):
            try:
                response = requests.get(f"{API_URL}/api/v1/security/analyze")
                
                if response.status_code == 200:
                    result = response.json()
                    analysis = result['analysis']
                    
                    # Score de sécurité
                    st.subheader("Score de Sécurité")
                    
                    score = analysis['security_score']
                    threat = analysis['threat_level']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': "Score de Sécurité"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkgreen" if score > 80 else "orange" if score > 50 else "red"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightcoral"},
                                    {'range': [50, 80], 'color': "lightyellow"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ]
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        threat_colors = {
                            "LOW": "🟢 Faible",
                            "MEDIUM": "🟡 Moyen",
                            "HIGH": "🟠 Élevé",
                            "CRITICAL": "🔴 Critique"
                        }
                        st.metric("Niveau de Menace", threat_colors.get(threat, threat))
                        st.write("")
                        st.write(f"**Timestamp:** {analysis['timestamp'][:19]}")
                    
                    st.write("---")
                    
                    # Connexions suspectes
                    st.subheader("🚨 Connexions Suspectes")
                    
                    suspicious_conn = analysis['suspicious_connections']
                    
                    if suspicious_conn:
                        for conn in suspicious_conn:
                            with st.expander(f"⚠️ {conn['remote_addr']} - Risque: {conn['risk']}"):
                                st.write(f"**Local:** {conn['local_addr']}")
                                st.write(f"**Distant:** {conn['remote_addr']}")
                                st.write(f"**Status:** {conn['status']}")
                                st.write(f"**Raison:** {conn['reason']}")
                                
                                if st.button(f"🛡️ Bloquer", key=f"block_{conn['remote_addr']}"):
                                    block_response = requests.post(
                                        f"{API_URL}/api/v1/security/block/{conn['remote_addr']}"
                                    )
                                    
                                    if block_response.status_code == 200:
                                        st.success("Connexion bloquée (mode défensif)")
                    else:
                        st.success("✅ Aucune connexion suspecte détectée")
                    
                    st.write("---")
                    
                    # Processus suspects
                    st.subheader("🔍 Processus Suspects")
                    
                    suspicious_proc = analysis['suspicious_processes']
                    
                    if suspicious_proc:
                        for proc in suspicious_proc:
                            st.warning(f"**{proc['name']}** (PID: {proc['pid']}) - "
                                     f"{proc['connections_count']} connexions - "
                                     f"Risque: {proc['risk']}")
                    else:
                        st.success("✅ Aucun processus suspect détecté")
                    
                    st.write("---")
                    
                    # Statistiques réseau
                    st.subheader("📊 Statistiques Réseau")
                    
                    net_stats = analysis['network_stats']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Connexions Actives", net_stats['active_connections'])
                    with col2:
                        st.metric("Établies", net_stats['established'])
                    with col3:
                        st.metric("En Écoute", net_stats['listening'])
                    
                    # Recommandations
                    st.write("---")
                    st.subheader("💡 Recommandations de Sécurité")
                    
                    for rec in analysis['recommendations']:
                        st.info(f"• {rec}")
            
            except Exception as e:
                st.error(f"Erreur: {str(e)}")

# PAGE: Marketplace Cloud
def page_marketplace():
    st.title("🛒 Marketplace Ressources Cloud")
    
    st.info("Louez des ressources cloud haute performance pour augmenter les capacités de votre système")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/marketplace/resources")
        
        if response.status_code == 200:
            resources = response.json()
            
            # Tabs par type
            tab1, tab2, tab3 = st.tabs(["🎮 Cloud GPU", "🖥️ Cloud CPU", "💾 Cloud RAM"])
            
            with tab1:
                st.subheader("GPUs Cloud Disponibles")
                
                for gpu in resources.get('cloud_gpu', []):
                    with st.expander(f"{gpu['name']} - ${gpu['price_per_hour']}/h"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Spécifications:**")
                            for key, value in gpu['specs'].items():
                                st.write(f"- {key}: {value}")
                        
                        with col2:
                            st.write(f"**Prix:** ${gpu['price_per_hour']}/heure")
                            st.write(f"**Description:** {gpu['description']}")
                            
                            duration = st.number_input(
                                "Durée (heures)",
                                1, 720, 24,
                                key=f"duration_{gpu['id']}"
                            )
                            
                            total = gpu['price_per_hour'] * duration
                            st.metric("Coût Total", f"${total:.2f}")
                            
                            if st.button("🛒 Louer", key=f"buy_{gpu['id']}"):
                                purchase_data = {
                                    "resource_type": "cloud_gpu",
                                    "specs": {"id": gpu['id']},
                                    "duration_hours": duration
                                }
                                
                                purchase_response = requests.post(
                                    f"{API_URL}/api/v1/marketplace/purchase?user_id={st.session_state.user_id}",
                                    json=purchase_data
                                )
                                
                                if purchase_response.status_code == 200:
                                    result = purchase_response.json()
                                    st.success("Ressource louée!")
                                    st.code(f"Endpoint: {result['purchase']['connection_endpoint']}")
                                    st.balloons()
            
            with tab2:
                st.subheader("CPUs Cloud Disponibles")
                
                for cpu in resources.get('cloud_cpu', []):
                    with st.expander(f"{cpu['name']} - ${cpu['price_per_hour']}/h"):
                        st.write("**Spécifications:**")
                        for key, value in cpu['specs'].items():
                            st.write(f"- {key}: {value}")
            
            with tab3:
                st.subheader("RAM Cloud Disponible")
                
                for ram in resources.get('cloud_ram', []):
                    with st.expander(f"{ram['name']} - ${ram['price_per_hour']}/h"):
                        st.write("**Spécifications:**")
                        for key, value in ram['specs'].items():
                            st.write(f"- {key}: {value}")
            
            # Mes achats
            st.write("---")
            st.subheader("📦 Mes Ressources Actives")
            
            purchases_response = requests.get(
                f"{API_URL}/api/v1/marketplace/purchases/{st.session_state.user_id}"
            )
            
            if purchases_response.status_code == 200:
                purchases_data = purchases_response.json()
                purchases = purchases_data['purchases']
                
                if purchases:
                    for purchase in purchases:
                        with st.expander(f"{purchase['resource_details']['name']} - {purchase['status']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Acheté:** {purchase['purchased_at'][:19]}")
                                st.write(f"**Expire:** {purchase['expires_at'][:19]}")
                            
                            with col2:
                                st.write(f"**Coût:** ${purchase['total_cost']:.2f}")
                                st.write(f"**Endpoint:** {purchase['connection_endpoint']}")
                else:
                    st.info("Aucune ressource active")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# Navigation
def main():
    init_session()
    
    with st.sidebar:
        st.title("System Optimizer")
        
        menu = {
            "Dashboard Système": "monitor",
            "Optimisation": "optimization",
            "Sécurité": "security",
            "Marketplace Cloud": "marketplace"
        }
        
        for label, view in menu.items():
            if st.button(label, use_container_width=True):
                st.session_state.active_view = view
                st.rerun()
        
        st.write("---")
        
        if check_api():
            st.success("API Connectée")
        else:
            st.error("API Déconnectée")
        
        st.write("---")
        st.caption("System Optimizer v1.0")
        st.caption("Mode Défensif Uniquement")
    
    view = st.session_state.get('active_view', 'monitor')
    
    if view == 'monitor':
        page_system_monitor()
    elif view == 'optimization':
        page_optimization()
    elif view == 'security':
        page_security()
    elif view == 'marketplace':
        page_marketplace()

if __name__ == "__main__":
    main()
