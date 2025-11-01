"""
🏢 Datacenter Management Platform - Advanced Edition
Gestion Complète • IA • Monitoring • Automation • Security

Installation:
pip install streamlit pandas plotly numpy scipy scikit-learn networkx requests

Lancement:
streamlit run datacenter_platform_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# ==================== CONFIGURATION PAGE ====================
st.set_page_config(
    page_title="🏢 Datacenter Management Platform",
    page_icon="🏢",
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
        background: linear-gradient(90deg, #00D9FF 0%, #0080FF 30%, #0040FF 60%, #00D9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: glow-pulse 3s ease-in-out infinite alternate;
    }
    @keyframes glow-pulse {
        from { filter: drop-shadow(0 0 10px #00D9FF); }
        to { filter: drop-shadow(0 0 30px #0040FF); }
    }
    .dc-card {
        border: 3px solid #00D9FF;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 64, 255, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.4);
        transition: all 0.3s;
    }
    .dc-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 128, 255, 0.6);
    }
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
    }
    .status-online {
        background: linear-gradient(90deg, #00FF00 0%, #00CC00 100%);
        color: white;
    }
    .status-warning {
        background: linear-gradient(90deg, #FFA500 0%, #FF8C00 100%);
        color: white;
    }
    .status-critical {
        background: linear-gradient(90deg, #FF0000 0%, #CC0000 100%);
        color: white;
    }
    .metric-card {
        background: rgba(0, 217, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 5px solid #00D9FF;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES ====================
DATACENTER_TIERS = {
    'Tier 1': {'uptime': 99.671, 'power': 'N', 'cooling': 'N', 'downtime_hours': 28.8},
    'Tier 2': {'uptime': 99.741, 'power': 'N+1', 'cooling': 'N+1', 'downtime_hours': 22.0},
    'Tier 3': {'uptime': 99.982, 'power': 'N+1', 'cooling': 'N+1', 'downtime_hours': 1.6},
    'Tier 4': {'uptime': 99.995, 'power': '2N', 'cooling': '2N+1', 'downtime_hours': 0.4}
}

SERVER_TYPES = {
    'Compute': {'power_w': 300, 'cpu_cores': 64, 'ram_gb': 256, 'storage_tb': 2},
    'Storage': {'power_w': 200, 'cpu_cores': 16, 'ram_gb': 64, 'storage_tb': 100},
    'GPU': {'power_w': 800, 'cpu_cores': 32, 'ram_gb': 512, 'storage_tb': 4},
    'Database': {'power_w': 400, 'cpu_cores': 48, 'ram_gb': 512, 'storage_tb': 20},
    'Network': {'power_w': 150, 'cpu_cores': 8, 'ram_gb': 32, 'storage_tb': 1}
}

COOLING_TYPES = {
    'Air Cooling': {'efficiency_pue': 1.8, 'cost_per_kw': 100, 'maintenance': 'Low'},
    'Liquid Cooling': {'efficiency_pue': 1.3, 'cost_per_kw': 300, 'maintenance': 'Medium'},
    'Immersion Cooling': {'efficiency_pue': 1.05, 'cost_per_kw': 500, 'maintenance': 'High'},
    'Free Cooling': {'efficiency_pue': 1.2, 'cost_per_kw': 150, 'maintenance': 'Low'}
}

SECURITY_LEVELS = ['Public', 'Confidential', 'Secret', 'Top Secret']

# ==================== INITIALISATION SESSION STATE ====================
if 'datacenter' not in st.session_state:
    st.session_state.datacenter = {
        'racks': {},
        'servers': {},
        'storage_systems': {},
        'network_devices': {},
        'cooling_systems': {},
        'power_systems': {},
        'vm_instances': {},
        'containers': {},
        'security_zones': {},
        'incidents': [],
        'maintenance_schedules': [],
        'backups': [],
        'monitoring_data': [],
        'alerts': [],
        'capacity_planning': {},
        'cost_tracking': [],
        'automation_jobs': [],
        'ai_predictions': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================
def log_event(message: str, level: str = "INFO"):
    """Enregistrer événement"""
    st.session_state.datacenter['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def calculate_pue(it_power_kw: float, total_power_kw: float) -> float:
    """Calculer Power Usage Effectiveness"""
    if it_power_kw == 0:
        return 0
    return total_power_kw / it_power_kw

def calculate_dcie(it_power_kw: float, total_power_kw: float) -> float:
    """Calculer Data Center Infrastructure Efficiency"""
    if total_power_kw == 0:
        return 0
    return (it_power_kw / total_power_kw) * 100

def generate_server_metrics(server_type: str, load_pct: float) -> Dict:
    """Générer métriques serveur"""
    base_power = SERVER_TYPES[server_type]['power_w']
    
    return {
        'cpu_usage': load_pct,
        'memory_usage': load_pct * np.random.uniform(0.8, 1.2),
        'disk_usage': np.random.uniform(30, 80),
        'network_in_mbps': load_pct * 100 * np.random.uniform(0.5, 1.5),
        'network_out_mbps': load_pct * 80 * np.random.uniform(0.5, 1.5),
        'power_consumption_w': base_power * (0.3 + 0.7 * load_pct / 100),
        'temperature_c': 20 + load_pct * 0.5 + np.random.uniform(-2, 2)
    }

def predict_failure_probability(age_days: int, usage_pct: float, temp_c: float) -> float:
    """Prédire probabilité de panne (IA simplifiée)"""
    age_factor = min(age_days / 1825, 1.0)  # 5 ans = 100%
    usage_factor = usage_pct / 100
    temp_factor = max(0, (temp_c - 20) / 60)  # Température idéale 20°C
    
    probability = (age_factor * 0.4 + usage_factor * 0.3 + temp_factor * 0.3) * 100
    return min(probability, 100)

def simulate_network_traffic(duration_hours: int) -> List[float]:
    """Simuler trafic réseau"""
    traffic = []
    for hour in range(duration_hours):
        # Pattern journalier
        hour_of_day = hour % 24
        if 9 <= hour_of_day <= 18:  # Heures bureau
            base = 80
        elif 18 <= hour_of_day <= 23:  # Soirée
            base = 60
        else:  # Nuit
            base = 20
        
        # Ajouter variation
        traffic.append(base + np.random.uniform(-10, 10))
    
    return traffic

def optimize_vm_placement(vms: List[Dict], servers: List[Dict]) -> Dict:
    """Optimiser placement VMs (algorithme bin packing simplifié)"""
    # Trier VMs par taille décroissante
    sorted_vms = sorted(vms, key=lambda x: x['cpu_cores'], reverse=True)
    
    placement = {i: [] for i in range(len(servers))}
    server_usage = {i: {'cpu': 0, 'ram': 0} for i in range(len(servers))}
    
    for vm in sorted_vms:
        # Trouver serveur avec assez de ressources
        placed = False
        for i, server in enumerate(servers):
            available_cpu = server['cpu_cores'] - server_usage[i]['cpu']
            available_ram = server['ram_gb'] - server_usage[i]['ram']
            
            if available_cpu >= vm['cpu_cores'] and available_ram >= vm['ram_gb']:
                placement[i].append(vm['name'])
                server_usage[i]['cpu'] += vm['cpu_cores']
                server_usage[i]['ram'] += vm['ram_gb']
                placed = True
                break
        
        if not placed:
            return {'status': 'error', 'message': 'Insufficient resources'}
    
    return {
        'status': 'success',
        'placement': placement,
        'server_usage': server_usage,
        'utilization': sum(u['cpu'] for u in server_usage.values()) / sum(s['cpu_cores'] for s in servers) * 100
    }

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">🏢 Datacenter Management Platform</h1>', unsafe_allow_html=True)
st.markdown("### Infrastructure • Monitoring • Automation • AI • Security • Analytics")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/00D9FF/FFFFFF?text=DataCenter+Pro", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Dashboard",
            "🖥️ Infrastructure",
            "🔌 Racks Management",
            "💻 Servers",
            "💾 Storage",
            "🌐 Network",
            "❄️ Cooling",
            "⚡ Power",
            "☁️ Virtualization",
            "🐳 Containers",
            "📊 Monitoring",
            "🤖 AI Operations",
            "🔐 Security",
            "🚨 Incidents",
            "🔧 Maintenance",
            "💰 Cost Management",
            "📈 Capacity Planning",
            "🔄 Backup & DR",
            "⚙️ Automation",
            "📉 Analytics",
            "⚙️ Settings"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    total_racks = len(st.session_state.datacenter['racks'])
    total_servers = len(st.session_state.datacenter['servers'])
    total_vms = len(st.session_state.datacenter['vm_instances'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔌 Racks", total_racks)
        st.metric("💻 Servers", total_servers)
    with col2:
        st.metric("☁️ VMs", total_vms)
        st.metric("🚨 Alerts", len(st.session_state.datacenter['alerts']))
    
    # Status global
    if total_servers > 0:
        uptime = np.random.uniform(99.9, 99.99)
        st.metric("⏱️ Uptime", f"{uptime:.3f}%")
    
    st.markdown("---")
    st.markdown("### 🔔 Recent Alerts")
    if st.session_state.datacenter['alerts']:
        for alert in st.session_state.datacenter['alerts'][-3:]:
            severity_icon = "🔴" if alert.get('severity') == 'critical' else "🟡" if alert.get('severity') == 'warning' else "🔵"
            st.write(f"{severity_icon} {alert.get('message', 'Alert')[:30]}...")
    else:
        st.info("No active alerts")

# ==================== PAGE: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.header("🏠 Datacenter Control Center")
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_power_kw = total_servers * 0.3 if total_servers else 0
        st.markdown(f'<div class="dc-card"><h2>⚡</h2><h3>{total_power_kw:.1f} kW</h3><p>Power Draw</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        pue = np.random.uniform(1.2, 1.5)
        st.markdown(f'<div class="dc-card"><h2>📊</h2><h3>{pue:.2f}</h3><p>PUE</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        avg_temp = np.random.uniform(22, 26)
        st.markdown(f'<div class="dc-card"><h2>🌡️</h2><h3>{avg_temp:.1f}°C</h3><p>Avg Temp</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        capacity_used = (total_servers / max(total_racks * 42, 1)) * 100 if total_racks else 0
        st.markdown(f'<div class="dc-card"><h2>📈</h2><h3>{capacity_used:.0f}%</h3><p>Capacity</p></div>', 
                   unsafe_allow_html=True)
    
    with col5:
        uptime = np.random.uniform(99.95, 99.99)
        st.markdown(f'<div class="dc-card"><h2>⏱️</h2><h3>{uptime:.2f}%</h3><p>Uptime</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphiques principaux
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Power Consumption (24h)")
        
        hours = list(range(24))
        power_consumption = [
            50 + 30 * np.sin((h - 6) * np.pi / 12) + np.random.uniform(-5, 5) 
            for h in hours
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=power_consumption,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#00D9FF', width=3),
            name='Power (kW)'
        ))
        
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌡️ Temperature Distribution")
        
        zones = ['Cold Aisle', 'Hot Aisle', 'Rack Top', 'Rack Bottom', 'CRAC Unit']
        temperatures = [22, 35, 28, 24, 18]
        
        fig = go.Figure(data=[go.Bar(
            x=zones,
            y=temperatures,
            marker_color=['#00D9FF', '#FF6B6B', '#FFA500', '#00FF00', '#0080FF']
        )])
        
        fig.update_layout(
            xaxis_title="Zone",
            yaxis_title="Temperature (°C)",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Resource Utilization
    st.subheader("📊 Resource Utilization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_util = np.random.uniform(60, 85)
        st.metric("CPU", f"{cpu_util:.1f}%")
        st.progress(cpu_util / 100)
    
    with col2:
        mem_util = np.random.uniform(55, 80)
        st.metric("Memory", f"{mem_util:.1f}%")
        st.progress(mem_util / 100)
    
    with col3:
        storage_util = np.random.uniform(40, 70)
        st.metric("Storage", f"{storage_util:.1f}%")
        st.progress(storage_util / 100)
    
    with col4:
        network_util = np.random.uniform(30, 60)
        st.metric("Network", f"{network_util:.1f}%")
        st.progress(network_util / 100)
    
    st.markdown("---")
    
    # Infrastructure Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🏗️ Infrastructure Map")
        
        # Simulation carte datacenter
        if total_racks > 0:
            st.info("Interactive datacenter map would be displayed here with 3D visualization")
            st.write("📍 Click on racks for details")
            st.write("🔍 Zoom and pan to navigate")
            st.write("🎨 Color-coded by temperature/utilization")
        else:
            st.warning("No infrastructure configured yet")
    
    with col2:
        st.subheader("🎯 Today's Summary")
        
        st.write("### ✅ Status: OPERATIONAL")
        
        st.write("**Active Resources:**")
        st.write(f"• Racks: {total_racks}")
        st.write(f"• Servers: {total_servers}")
        st.write(f"• VMs: {total_vms}")
        st.write(f"• Containers: {len(st.session_state.datacenter['containers'])}")
        
        st.write("\n**Today's Incidents:**")
        incidents_today = [i for i in st.session_state.datacenter['incidents'] 
                          if i.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))]
        st.write(f"• Total: {len(incidents_today)}")
        st.write(f"• Critical: 0")
        st.write(f"• Warning: {len(incidents_today)}")
        
        st.write("\n**Maintenance:**")
        st.write(f"• Scheduled: {len(st.session_state.datacenter['maintenance_schedules'])}")
        st.write(f"• In Progress: 0")

# ==================== PAGE: INFRASTRUCTURE ====================
elif page == "🖥️ Infrastructure":
    st.header("🖥️ Infrastructure Overview")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "➕ Add Infrastructure", "🗺️ Topology", "📈 Growth"])
    
    with tab1:
        st.subheader("📊 Infrastructure Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Centers", "1")  # Single DC for now
            st.metric("Racks", total_racks)
            st.metric("U Space Total", total_racks * 42)
        
        with col2:
            st.metric("Servers", total_servers)
            st.metric("Network Devices", len(st.session_state.datacenter['network_devices']))
            st.metric("Storage Arrays", len(st.session_state.datacenter['storage_systems']))
        
        with col3:
            total_cpu_cores = sum(SERVER_TYPES[s.get('type', 'Compute')]['cpu_cores'] 
                                 for s in st.session_state.datacenter['servers'].values())
            st.metric("Total CPU Cores", f"{total_cpu_cores:,}")
            
            total_ram_gb = sum(SERVER_TYPES[s.get('type', 'Compute')]['ram_gb'] 
                              for s in st.session_state.datacenter['servers'].values())
            st.metric("Total RAM", f"{total_ram_gb:,} GB")
        
        with col4:
            total_storage_tb = sum(SERVER_TYPES[s.get('type', 'Compute')]['storage_tb'] 
                                  for s in st.session_state.datacenter['servers'].values())
            st.metric("Total Storage", f"{total_storage_tb:.1f} TB")
            st.metric("Power Capacity", f"{total_racks * 10} kW")
        
        # Tier Classification
        st.write("### 🏆 Datacenter Tier Classification")
        
        tier = st.selectbox("Current Tier", list(DATACENTER_TIERS.keys()), index=2)
        
        tier_info = DATACENTER_TIERS[tier]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Uptime", f"{tier_info['uptime']}%")
        with col2:
            st.metric("Power", tier_info['power'])
        with col3:
            st.metric("Cooling", tier_info['cooling'])
        with col4:
            st.metric("Downtime/Year", f"{tier_info['downtime_hours']}h")
    
    with tab2:
        st.subheader("➕ Add Infrastructure Component")
        
        component_type = st.selectbox("Component Type",
            ["Rack", "Server", "Network Device", "Storage Array", "Cooling Unit", "Power Unit"])
        
        if component_type == "Rack":
            with st.form("add_rack"):
                col1, col2 = st.columns(2)
                
                with col1:
                    rack_name = st.text_input("Rack Name", "RACK-001")
                    location = st.text_input("Location", "Row A, Position 1")
                    u_capacity = st.number_input("U Capacity", 1, 52, 42)
                
                with col2:
                    power_capacity_kw = st.number_input("Power Capacity (kW)", 1, 30, 10)
                    cooling_type = st.selectbox("Cooling", list(COOLING_TYPES.keys()))
                    security_zone = st.selectbox("Security Zone", SECURITY_LEVELS)
                
                if st.form_submit_button("➕ Add Rack", type="primary"):
                    rack_id = f"rack_{len(st.session_state.datacenter['racks']) + 1}"
                    
                    rack = {
                        'id': rack_id,
                        'name': rack_name,
                        'location': location,
                        'u_capacity': u_capacity,
                        'u_used': 0,
                        'power_capacity_kw': power_capacity_kw,
                        'power_used_kw': 0,
                        'cooling_type': cooling_type,
                        'security_zone': security_zone,
                        'status': 'active',
                        'temperature_c': 22.0,
                        'servers': [],
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.datacenter['racks'][rack_id] = rack
                    log_event(f"Rack added: {rack_name}", "SUCCESS")
                    
                    st.success(f"✅ Rack '{rack_name}' added successfully!")
                    st.balloons()
                    st.rerun()
    
    with tab3:
        st.subheader("🗺️ Network Topology")
        
        if total_racks > 0:
            # Créer graphique réseau simplifié
            st.info("Network topology visualization with interactive graph")
            
            # Simuler topologie
            st.write("### Core Network")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Core Switches:**")
                st.write("• Core-SW-01 (Active)")
                st.write("• Core-SW-02 (Standby)")
            
            with col2:
                st.write("**Distribution:**")
                st.write("• Dist-SW-01")
                st.write("• Dist-SW-02")
            
            with col3:
                st.write("**Access:**")
                st.write(f"• {total_racks} ToR Switches")
                st.write(f"• {total_servers} Connected Servers")
        else:
            st.warning("No infrastructure to display")
    
    with tab4:
        st.subheader("📈 Infrastructure Growth Projection")
        
        years = list(range(2024, 2029))
        racks_projection = [total_racks * (1.15 ** i) for i in range(5)]
        servers_projection = [total_servers * (1.20 ** i) for i in range(5)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=racks_projection,
            mode='lines+markers',
            name='Racks',
            line=dict(color='#00D9FF', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=years,
            y=servers_projection,
            mode='lines+markers',
            name='Servers',
            line=dict(color='#0080FF', width=3)
        ))
        
        fig.update_layout(
            title="5-Year Growth Forecast",
            xaxis_title="Year",
            yaxis_title="Count",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: RACKS MANAGEMENT ====================
elif page == "🔌 Racks Management":
    st.header("🔌 Racks Management")
    
    if not st.session_state.datacenter['racks']:
        st.info("No racks configured. Add your first rack in the Infrastructure page.")
    else:
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location_filter = st.multiselect("Location",
                list(set(r['location'] for r in st.session_state.datacenter['racks'].values())))
        
        with col2:
            status_filter = st.multiselect("Status",
                ["active", "maintenance", "offline"],
                default=["active"])
        
        with col3:
            utilization_threshold = st.slider("Show racks with utilization >", 0, 100, 0)
        
        # Afficher racks
        for rack_id, rack in st.session_state.datacenter['racks'].items():
            # Appliquer filtres
            if location_filter and rack['location'] not in location_filter:
                continue
            if status_filter and rack['status'] not in status_filter:
                continue
            
            utilization_pct = (rack['u_used'] / rack['u_capacity']) * 100
            if utilization_pct < utilization_threshold:
                continue
            
            with st.expander(f"🔌 {rack['name']} - {rack['location']} ({utilization_pct:.0f}% used)"):
                col1, col2, col3, col4 = st.columns(4)
                # df_vms = pd.DataFrame([
                #     st.session_state.datacenter['servers'][srv_id]
                #     for srv_id in rack['servers']
                #     if srv_id in st.session_state.datacenter['servers']
                # ])

            # Vérifier la présence de serveurs
            servers_data = []

            if 'servers' in rack and isinstance(rack['servers'], list):
                for srv_id in rack['servers']:
                    server = st.session_state.datacenter.get('servers', {}).get(srv_id)
                    if server and isinstance(server, dict):
                        # Ne garder que les champs essentiels avec valeurs par défaut
                        safe_server = {
                            'id': srv_id,
                            'name': server.get('name', 'Unknown'),
                            'CPU Usage (%)': float(server.get('cpu_usage', 0)),
                            'CPU Cores': int(server.get('cpu_cores', 0)),
                            'RAM (GB)': float(server.get('ram_gb', 0)),
                            'u_position': int(server.get('u_position', 1)),
                            'u_size': int(server.get('u_size', 1)),
                        }
                        servers_data.append(safe_server)

            # Construire le DataFrame
            if servers_data:
                df_vms = pd.DataFrame(servers_data)
            else:
                # DataFrame vide mais avec colonnes attendues (évite KeyError plus tard)
                df_vms = pd.DataFrame(columns=[
                    'id', 'name', 'CPU Usage (%)', 'CPU Cores', 'RAM (GB)', 'u_position', 'u_size'
                ])

                with col1:
                    st.write("### 📊 Capacity")
                    st.metric("U Used", f"{rack['u_used']}/{rack['u_capacity']}")
                    st.progress(utilization_pct / 100)
                    
                    power_pct = (rack['power_used_kw'] / rack['power_capacity_kw']) * 100
                    st.metric("Power", f"{rack['power_used_kw']:.1f}/{rack['power_capacity_kw']} kW")
                    st.progress(power_pct / 100)
                
                with col2:
                    st.write("### 🌡️ Environment")
                    st.metric("Temperature", f"{rack['temperature_c']:.1f}°C")
                    st.metric("Cooling", rack['cooling_type'])

                with col3:
                    st.metric("Total vRAM", f"{df_vms['RAM (GB)'].sum():.0f} GB")
                with col4:
                    st.metric("Avg CPU Usage", f"{df_vms['CPU Usage (%)'].mean():.1f}%")

                # Statistiques
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total VMs", len(df_vms))
                with col2:
                    st.metric("Total vCPUs", df_vms['CPU Cores'].sum())
                with col3:
                    st.metric("Cooling", rack['cooling_type'])   
                    humidity = np.random.uniform(40, 60)
                    st.metric("Humidity", f"{humidity:.1f}%")
                    
                with col3:
                    st.write("### 🔐 Security")
                    st.write(f"**Zone:** {rack['security_zone']}")
                    st.write(f"**Status:** {rack['status']}")
                    st.write(f"**Servers:** {len(rack['servers'])}")
                    
                with col4:
                    st.write("### ⚙️ Actions")
                        
                    if st.button("📊 View Details", key=f"details_{rack_id}"):
                        st.info(f"Detailed view for {rack['name']}")
                        
                    if st.button("🔧 Maintenance", key=f"maint_{rack_id}"):
                        rack['status'] = 'maintenance'
                        st.warning("Rack set to maintenance mode")
                        st.rerun()
                        
                    if st.button("🗑️ Remove", key=f"del_{rack_id}"):
                        if len(rack['servers']) == 0:
                            del st.session_state.datacenter['racks'][rack_id]
                            st.success("Rack removed")
                            st.rerun()
                        else:
                            st.error("Cannot remove rack with servers")
                    
                    # Visualisation rack
                    st.write("### 🖼️ Rack Layout")
                    
                    # Créer visualisation U
                    rack_slots = ["Empty"] * rack['u_capacity']
                    for server_id in rack['servers']:
                        if server_id in st.session_state.datacenter['servers']:
                            server = st.session_state.datacenter['servers'][server_id]
                            start_u = server.get('u_position', 1) - 1
                            size_u = server.get('u_size', 1)
                            for i in range(start_u, min(start_u + size_u, rack['u_capacity'])):
                                rack_slots[i] = server['name']
                    
                    # Afficher en grille
                    cols_per_row = 6
                    for i in range(0, len(rack_slots), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(rack_slots):
                                u_num = i + j + 1
                                status = rack_slots[i + j]
                                color = "#00FF00" if status != "Empty" else "#333333"
                                col.markdown(f"<div style='background:{color};padding:5px;text-align:center;border-radius:5px;margin:2px;'>U{u_num}</div>", unsafe_allow_html=True)

                                html = "<div style='display:grid;grid-template-columns:repeat(6,1fr);gap:4px;'>"
                                for i, status in enumerate(rack_slots):
                                    color = "#00FF00" if status != "Empty" else "#333333"
                                    html += f"<div style='background:{color};padding:5px;border-radius:5px;text-align:center;'>U{i+1}</div>"
                                html += "</div>"
                                st.markdown(html, unsafe_allow_html=True)

        else:
            st.info("No VMs to analyze")

# ==================== PAGE: CONTAINERS ====================
elif page == "🐳 Containers":
    st.header("🐳 Container Orchestration")
    
    tab1, tab2, tab3 = st.tabs(["📦 Containers", "🎯 Kubernetes", "📊 Monitoring"])
    
    with tab1:
        st.subheader("📦 Container Management")
        
        # Ajouter container
        with st.expander("➕ Deploy Container"):
            with st.form("deploy_container"):
                col1, col2 = st.columns(2)
                
                with col1:
                    container_name = st.text_input("Container Name", "nginx-web")
                    image = st.text_input("Image", "nginx:latest")
                    replicas = st.number_input("Replicas", 1, 100, 3)
                
                with col2:
                    cpu_limit = st.number_input("CPU Limit (cores)", 0.1, 16.0, 1.0, 0.1)
                    memory_limit = st.number_input("Memory Limit (GB)", 0.1, 64.0, 2.0, 0.1)
                    port = st.number_input("Port", 1, 65535, 80)
                
                if st.form_submit_button("🚀 Deploy"):
                    container_id = f"container_{len(st.session_state.datacenter['containers']) + 1}"
                    
                    container = {
                        'id': container_id,
                        'name': container_name,
                        'image': image,
                        'replicas': replicas,
                        'cpu_limit': cpu_limit,
                        'memory_limit': memory_limit,
                        'port': port,
                        'status': 'running',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.datacenter['containers'][container_id] = container
                    log_event(f"Container deployed: {container_name}", "SUCCESS")
                    
                    st.success(f"✅ Container '{container_name}' deployed!")
                    st.rerun()
        
        # Liste containers
        if st.session_state.datacenter['containers']:
            for container_id, container in st.session_state.datacenter['containers'].items():
                with st.expander(f"🐳 {container['name']} ({container['image']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Configuration")
                        st.write(f"**Replicas:** {container['replicas']}")
                        st.write(f"**CPU:** {container['cpu_limit']} cores")
                        st.write(f"**Memory:** {container['memory_limit']} GB")
                        st.write(f"**Port:** {container['port']}")
                    
                    with col2:
                        st.write("### 📈 Metrics")
                        cpu_usage = np.random.uniform(10, 60)
                        mem_usage = np.random.uniform(20, 70)
                        st.metric("CPU", f"{cpu_usage:.1f}%")
                        st.metric("Memory", f"{mem_usage:.1f}%")
                        st.metric("Network", f"{np.random.uniform(100, 1000):.0f} MB/s")
                    
                    with col3:
                        st.write("### ⚙️ Actions")
                        
                        if st.button("🔄 Scale", key=f"scale_{container_id}"):
                            new_replicas = st.number_input("New Replicas", 1, 100, container['replicas'])
                        
                        if st.button("🔄 Restart", key=f"restart_cont_{container_id}"):
                            st.info("Container restarting...")
                        
                        if st.button("🗑️ Delete", key=f"del_cont_{container_id}"):
                            del st.session_state.datacenter['containers'][container_id]
                            st.rerun()
        else:
            st.info("No containers deployed")
    
    with tab2:
        st.subheader("🎯 Kubernetes Cluster")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nodes", np.random.randint(3, 10))
            st.metric("Pods", np.random.randint(50, 200))
        
        with col2:
            st.metric("Services", np.random.randint(10, 50))
            st.metric("Deployments", np.random.randint(20, 100))
        
        with col3:
            st.metric("CPU Usage", f"{np.random.uniform(40, 70):.1f}%")
            st.metric("Memory Usage", f"{np.random.uniform(50, 80):.1f}%")
        
        st.write("### 🔧 Cluster Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("➕ Add Node"):
                st.success("Node added to cluster!")
            
            if st.button("🔄 Rolling Update"):
                st.info("Performing rolling update...")
        
        with col2:
            if st.button("🔍 Health Check"):
                st.success("✅ Cluster healthy!")
            
            if st.button("📊 View Logs"):
                st.info("Displaying cluster logs...")
    
    with tab3:
        st.subheader("📊 Container Monitoring")
        
        if st.session_state.datacenter['containers']:
            # Métriques agrégées
            total_replicas = sum(c['replicas'] for c in st.session_state.datacenter['containers'].values())
            
            st.metric("Total Container Instances", total_replicas)
            
            # Graphique utilisation
            container_names = [c['name'] for c in st.session_state.datacenter['containers'].values()]
            cpu_usage = [np.random.uniform(10, 60) for _ in container_names]
            
            fig = go.Figure(data=[go.Bar(
                x=container_names,
                y=cpu_usage,
                marker_color='#00D9FF'
            )])
            
            fig.update_layout(
                title="CPU Usage by Container",
                xaxis_title="Container",
                yaxis_title="CPU (%)",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No containers to monitor")

# ==================== PAGE: SECURITY ====================
elif page == "🔐 Security":
    st.header("🔐 Security Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Overview", "👥 Access Control", "🔍 Audit", "🚨 Threats"])
    
    with tab1:
        st.subheader("🛡️ Security Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            security_score = np.random.uniform(85, 95)
            st.metric("Security Score", f"{security_score:.1f}/100")
        
        with col2:
            st.metric("Active Users", np.random.randint(50, 200))
        
        with col3:
            st.metric("Failed Logins (24h)", np.random.randint(5, 20))
        
        with col4:
            st.metric("Vulnerabilities", np.random.randint(0, 5))
        
        # Zones de sécurité
        st.write("### 🔐 Security Zones")
        
        zones = []
        for level in SECURITY_LEVELS:
            racks_in_zone = sum(1 for r in st.session_state.datacenter['racks'].values() 
                               if r['security_zone'] == level)
            zones.append({
                'Level': level,
                'Racks': racks_in_zone,
                'Servers': np.random.randint(0, racks_in_zone * 10) if racks_in_zone > 0 else 0,
                'Access': '2FA + Biometric' if level in ['Secret', 'Top Secret'] else '2FA'
            })
        
        df_zones = pd.DataFrame(zones)
        st.dataframe(df_zones, use_container_width=True)
        
        # Compliance
        st.write("### ✅ Compliance Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Standards:**")
            st.write("✅ ISO 27001")
            st.write("✅ SOC 2 Type II")
            st.write("✅ PCI DSS")
            st.write("✅ HIPAA")
        
        with col2:
            st.write("**Last Audits:**")
            st.write("• ISO 27001: 2024-09-15")
            st.write("• SOC 2: 2024-08-20")
            st.write("• PCI DSS: 2024-10-01")
            st.write("• HIPAA: 2024-09-30")
    
    with tab2:
        st.subheader("👥 Access Control")
        
        st.write("### 🔑 User Management")
        
        # Ajouter utilisateur
        with st.expander("➕ Add User"):
            with st.form("add_user"):
                col1, col2 = st.columns(2)
                
                with col1:
                    username = st.text_input("Username")
                    email = st.text_input("Email")
                    role = st.selectbox("Role", ["Admin", "Operator", "Viewer", "Auditor"])
                
                with col2:
                    security_clearance = st.selectbox("Security Clearance", SECURITY_LEVELS)
                    mfa_enabled = st.checkbox("Enable 2FA", value=True)
                    expiry_days = st.number_input("Access Expiry (days)", 1, 365, 90)
                
                if st.form_submit_button("➕ Add User"):
                    st.success(f"User '{username}' added!")
                    log_event(f"User added: {username}", "INFO")
        
        # Permissions actives
        st.write("### 🔐 Active Permissions")
        
        permissions_data = [
            {"User": "admin", "Role": "Admin", "Last Login": "2 min ago", "2FA": "✅"},
            {"User": "operator1", "Role": "Operator", "Last Login": "1 hour ago", "2FA": "✅"},
            {"User": "viewer1", "Role": "Viewer", "Last Login": "3 hours ago", "2FA": "❌"},
        ]
        
        df_permissions = pd.DataFrame(permissions_data)
        st.dataframe(df_permissions, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Security Audit Trail")
        
        # Filtres audit
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_range = st.date_input("Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now()))
        
        with col2:
            event_type = st.multiselect("Event Type",
                ["Login", "Logout", "Access Denied", "Configuration Change", "Data Access"])
        
        with col3:
            user_filter = st.text_input("User Filter")
        
        # Logs audit
        st.write("### 📋 Audit Logs")
        
        audit_logs = [
            {"Time": "10:23:45", "User": "admin", "Action": "Login", "IP": "10.0.0.100", "Status": "Success"},
            {"Time": "10:15:32", "User": "operator1", "Action": "Server Reboot", "IP": "10.0.0.101", "Status": "Success"},
            {"Time": "09:45:12", "User": "unknown", "Action": "Login", "IP": "192.168.1.50", "Status": "Failed"},
            {"Time": "09:30:00", "User": "admin", "Action": "Config Change", "IP": "10.0.0.100", "Status": "Success"},
        ]
        
        df_audit = pd.DataFrame(audit_logs)
        st.dataframe(df_audit, use_container_width=True)
        
        if st.button("📥 Export Audit Log"):
            st.success("Audit log exported!")
    
    with tab4:
        st.subheader("🚨 Threat Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🔍 Active Threats")
            
            threats = [
                {"Type": "Brute Force", "Severity": "High", "Source": "192.168.1.50", "Status": "Blocked"},
                {"Type": "Port Scan", "Severity": "Medium", "Source": "10.50.30.20", "Status": "Monitoring"},
                {"Type": "DDoS Attempt", "Severity": "Critical", "Source": "Multiple", "Status": "Mitigated"},
            ]
            
            for threat in threats:
                severity_color = "🔴" if threat['Severity'] == "Critical" else "🟡" if threat['Severity'] == "High" else "🔵"
                st.write(f"{severity_color} **{threat['Type']}** from {threat['Source']} - {threat['Status']}")
        
        with col2:
            st.write("### 📊 Threat Statistics (7 days)")
            
            st.metric("Total Threats", np.random.randint(50, 200))
            st.metric("Blocked", np.random.randint(45, 195))
            st.metric("False Positives", np.random.randint(0, 10))
        
        # Threat map
        st.write("### 🗺️ Threat Origin Map")
        st.info("Geographic threat visualization would be displayed here")

# ==================== PAGE: COST MANAGEMENT ====================
elif page == "💰 Cost Management":
    st.header("💰 Cost Management & Billing")
    
    tab1, tab2, tab3 = st.tabs(["💵 Overview", "📊 Analysis", "🎯 Optimization"])
    
    with tab1:
        st.subheader("💵 Cost Overview")
        
        # Coûts mensuels
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_cost = np.random.uniform(50000, 150000)
            st.metric("This Month", f"${monthly_cost:,.0f}")
        
        with col2:
            last_month = np.random.uniform(45000, 140000)
            change = ((monthly_cost - last_month) / last_month) * 100
            st.metric("Last Month", f"${last_month:,.0f}", f"{change:+.1f}%")
        
        with col3:
            projected = monthly_cost * 1.05
            st.metric("Projected", f"${projected:,.0f}")
        
        with col4:
            budget = 120000
            st.metric("Budget", f"${budget:,.0f}")
            if monthly_cost > budget:
                st.warning("⚠️ Over budget!")
        
        # Breakdown par catégorie
        st.write("### 💸 Cost Breakdown")
        
        cost_categories = {
            'Compute': 40000,
            'Storage': 25000,
            'Network': 15000,
            'Power': 30000,
            'Cooling': 20000,
            'Licenses': 10000,
            'Support': 8000
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(cost_categories.keys()),
            values=list(cost_categories.values()),
            hole=.4,
            marker=dict(colors=['#00D9FF', '#0080FF', '#0040FF', '#FF6B6B', '#FFA500', '#00FF00', '#FFD700'])
        )])
        
        fig.update_layout(
            title="Cost Distribution",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tendance coûts
        st.write("### 📈 Cost Trend (12 months)")
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        costs = [monthly_cost * (1 + np.random.uniform(-0.1, 0.1)) for _ in months]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=months,
            y=costs,
            mode='lines+markers',
            name='Actual Cost',
            line=dict(color='#00D9FF', width=3)
        ))
        
        fig.add_hline(y=budget, line_dash="dash", line_color="red", annotation_text="Budget")
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Cost ($)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Cost Analysis")
        
        # Coût par serveur
        if st.session_state.datacenter['servers']:
            st.write("### 💻 Cost by Server")
            
            server_costs = []
            for server in st.session_state.datacenter['servers'].values():
                specs = SERVER_TYPES[server['type']]
                monthly_cost = (specs['power_w'] * 0.12 * 730 / 1000) + 100  # Power + fixed
                server_costs.append({
                    'Server': server['name'],
                    'Type': server['type'],
                    'Monthly Cost': monthly_cost,
                    'Annual Cost': monthly_cost * 12
                })
            
            df_costs = pd.DataFrame(server_costs)
            
            fig = px.bar(df_costs, x='Server', y='Monthly Cost', color='Type',
                        title="Monthly Cost by Server",
                        template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Top coûteux
            st.write("### 💰 Top 5 Most Expensive")
            top_costs = df_costs.nlargest(5, 'Monthly Cost')[['Server', 'Monthly Cost', 'Annual Cost']]
            st.dataframe(top_costs, use_container_width=True)
        else:
            st.info("No servers to analyze")
    
    with tab3:
        st.subheader("🎯 Cost Optimization")
        
        st.write("### 💡 Optimization Opportunities")
        
        opportunities = [
            {
                "opportunity": "🔋 Power Efficiency",
                "savings": "$15,000/year",
                "effort": "Low",
                "description": "Optimize server utilization and consolidate workloads"
            },
            {
                "opportunity": "☁️ Cloud Burst",
                "savings": "$25,000/year",
                "effort": "Medium",
                "description": "Move non-critical workloads to cloud during off-peak"
            },
            {
                "opportunity": "💾 Storage Tiering",
                "savings": "$10,000/year",
                "effort": "Low",
                "description": "Move cold data to cheaper storage tiers"
            },
            {
                "opportunity": "🔄 Right-sizing VMs",
                "savings": "$8,000/year",
                "effort": "Low",
                "description": "Resize over-provisioned VMs"
            }
        ]
        
        for opp in opportunities:
            with st.expander(f"{opp['opportunity']} - Save {opp['savings']}"):
                st.write(f"**Description:** {opp['description']}")
                st.write(f"**Effort:** {opp['effort']}")
                
                if st.button("✅ Implement", key=f"impl_{opp['opportunity']}"):
                    st.success("Optimization implemented!")
                    log_event(f"Cost optimization: {opp['opportunity']}", "SUCCESS")

# ==================== PAGE: CAPACITY PLANNING ====================
elif page == "📈 Capacity Planning":
    st.header("📈 Capacity Planning")
    
    tab1, tab2, tab3 = st.tabs(["📊 Current Capacity", "🔮 Forecasting", "🎯 Recommendations"])
    
    with tab1:
        st.subheader("📊 Current Capacity Utilization")
        
        # Capacités actuelles
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rack_util = (total_servers * 2 / max(total_racks * 42, 1)) * 100 if total_racks > 0 else 0
            st.metric("Rack Space", f"{rack_util:.1f}%")
            st.progress(rack_util / 100)
        
        with col2:
            cpu_util = np.random.uniform(60, 80)
            st.metric("CPU", f"{cpu_util:.1f}%")
            st.progress(cpu_util / 100)
        
        with col3:
            mem_util = np.random.uniform(55, 75)
            st.metric("Memory", f"{mem_util:.1f}%")
            st.progress(mem_util / 100)
        
        with col4:
            storage_util = np.random.uniform(45, 70)
            st.metric("Storage", f"{storage_util:.1f}%")
            st.progress(storage_util / 100)
        
        # Graphique tendance
        st.write("### 📈 Capacity Trend (6 months)")
        
        months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        cpu_trend = [50, 55, 60, 65, 70, cpu_util]
        mem_trend = [45, 50, 52, 58, 62, mem_util]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=months, y=cpu_trend,
                                mode='lines+markers', name='CPU',
                                line=dict(color='#00D9FF', width=3)))
        
        fig.add_trace(go.Scatter(x=months, y=mem_trend,
                                mode='lines+markers', name='Memory',
                                line=dict(color='#FF6B6B', width=3)))
        
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                     annotation_text="Warning Threshold")
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Utilization (%)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Capacity Forecasting")
        
        forecast_period = st.selectbox("Forecast Period",
            ["3 months", "6 months", "1 year", "2 years"])
        
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner("AI analyzing growth patterns..."):
                import time
                time.sleep(2)
                
                periods = {"3 months": 3, "6 months": 6, "1 year": 12, "2 years": 24}[forecast_period]
                
                current_servers = total_servers
                forecast = [current_servers * (1 + 0.05 * i) for i in range(periods + 1)]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(periods + 1)),
                    y=forecast,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#00D9FF', width=3)
                ))
                
                # Zone confiance
                upper_bound = [f * 1.1 for f in forecast]
                lower_bound = [f * 0.9 for f in forecast]
                
                fig.add_trace(go.Scatter(
                    x=list(range(periods + 1)),
                    y=upper_bound,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(periods + 1)),
                    y=lower_bound,
                    mode='lines',
                    fill='tonexty',
                    name='Confidence Interval',
                    line=dict(width=0),
                    fillcolor='rgba(0, 217, 255, 0.2)'
                ))
                
                fig.update_layout(
                    title="Server Count Forecast",
                    xaxis_title="Months from Now",
                    yaxis_title="Number of Servers",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"📊 Forecast: You'll need ~{int(forecast[-1])} servers in {forecast_period}")
                
                # Recommandations
                st.write("### 📋 Procurement Recommendations")
                
                servers_needed = int(forecast[-1] - current_servers)
                racks_needed = int(servers_needed / 20)  # ~20 servers per rack
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Additional Servers", servers_needed)
                with col2:
                    st.metric("Additional Racks", racks_needed)
                with col3:
                    st.metric("Power Needed", f"{servers_needed * 0.3:.0f} kW")
    
    with tab3:
        st.subheader("🎯 Capacity Recommendations")
        
        st.write("### 💡 Action Items")
        
        recommendations = [
            {
                "priority": "🔴 High",
                "action": "Order 2 additional racks",
                "reason": "Current utilization at 85%",
                "timeline": "1 month"
            },
            {
                "priority": "🟡 Medium",
                "action": "Upgrade network to 100Gbps",
                "reason": "Network saturation during peak",
                "timeline": "3 months"
            },
            {
                "priority": "🟢 Low",
                "action": "Expand storage by 100TB",
                "reason": "Storage will reach 80% in 6 months",
                "timeline": "6 months"
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"{rec['priority']} - {rec['action']}"):
                st.write(f"**Reason:** {rec['reason']}")
                st.write(f"**Timeline:** {rec['timeline']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Approve", key=f"approve_{rec['action']}"):
                        st.success("Approved!")
                
                with col2:
                    if st.button("⏭️ Defer", key=f"defer_{rec['action']}"):
                        st.info("Deferred")
        
        # Graphiques temps réel
        st.write("### 📊 Time Series Data (Last Hour)")
        
        time_points = list(range(60))
        
        col1, col2 = st.columns(2)
        
        with col1:
            cpu_history = [metrics_current['cpu'] + np.random.uniform(-10, 10) for _ in time_points]
            mem_history = [metrics_current['memory'] + np.random.uniform(-8, 8) for _ in time_points]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=time_points, y=cpu_history,
                                    mode='lines', name='CPU',
                                    line=dict(color='#00D9FF', width=2)))
            
            fig.add_trace(go.Scatter(x=time_points, y=mem_history,
                                    mode='lines', name='Memory',
                                    line=dict(color='#FF6B6B', width=2)))
            
            fig.update_layout(
                title="Compute Resources",
                xaxis_title="Minutes Ago",
                yaxis_title="Utilization (%)",
                template="plotly_dark",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            network_in = [metrics_current['network'] + np.random.uniform(-5, 5) for _ in time_points]
            network_out = [metrics_current['network'] * 0.8 + np.random.uniform(-4, 4) for _ in time_points]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=time_points, y=network_in,
                                    mode='lines', name='Inbound',
                                    line=dict(color='#00FF00', width=2)))
            
            fig.add_trace(go.Scatter(x=time_points, y=network_out,
                                    mode='lines', name='Outbound',
                                    line=dict(color='#FFA500', width=2)))
            
            fig.update_layout(
                title="Network Traffic",
                xaxis_title="Minutes Ago",
                yaxis_title="Traffic (Gbps)",
                template="plotly_dark",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.write("### 🗺️ Infrastructure Heatmap")
        
        if total_racks > 0:
            # Créer heatmap température
            heatmap_data = np.random.uniform(20, 35, (5, max(total_racks, 5)))
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                colorscale='RdYlBu_r',
                text=np.round(heatmap_data, 1),
                texttemplate='%{text}°C',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Temperature Heatmap by Zone",
                xaxis_title="Rack",
                yaxis_title="Row",
                template="plotly_dark",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Configure racks to see heatmap")
    
    with tab2:
        st.subheader("🎯 Detailed Metrics")
        
        metric_category = st.selectbox("Category",
            ["Compute", "Storage", "Network", "Power", "Cooling", "Environmental"])
        
        if metric_category == "Compute":
            # Métriques compute détaillées
            servers_list = list(st.session_state.datacenter['servers'].values())
            
            if servers_list:
                df_compute = pd.DataFrame([
                    {
                        'Server': s['name'],
                        'CPU (%)': np.random.uniform(30, 90),
                        'Memory (%)': np.random.uniform(40, 85),
                        'Load Avg': np.random.uniform(1, 8),
                        'Processes': np.random.randint(100, 500),
                        'Uptime (days)': np.random.randint(1, 365)
                    }
                    for s in servers_list[:10]  # Limit à 10
                ])
                
                st.dataframe(df_compute, use_container_width=True)
                
                # Graphique distribution CPU
                fig = px.histogram(df_compute, x='CPU (%)', nbins=20,
                                 title="CPU Usage Distribution",
                                 template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No servers to monitor")
        
        elif metric_category == "Network":
            st.write("### 🌐 Network Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Bandwidth In", f"{np.random.uniform(10, 50):.1f} Gbps")
                st.metric("Bandwidth Out", f"{np.random.uniform(8, 40):.1f} Gbps")
            
            with col2:
                st.metric("Packets In", f"{np.random.randint(100000, 500000):,} pps")
                st.metric("Packets Out", f"{np.random.randint(80000, 400000):,} pps")
            
            with col3:
                st.metric("Latency", f"{np.random.uniform(0.5, 2.0):.2f} ms")
                st.metric("Packet Loss", f"{np.random.uniform(0, 0.1):.3f}%")
        
        elif metric_category == "Power":
            st.write("### ⚡ Power Metrics")
            
            # Graphique distribution power
            power_zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Cooling']
            power_consumption = [
                np.random.uniform(80, 120),
                np.random.uniform(70, 110),
                np.random.uniform(60, 100),
                np.random.uniform(75, 115),
                np.random.uniform(30, 50)
            ]
            
            fig = go.Figure(data=[go.Bar(
                x=power_zones,
                y=power_consumption,
                marker_color=['#00D9FF', '#0080FF', '#0040FF', '#6B00FF', '#FF6B6B']
            )])
            
            fig.update_layout(
                title="Power Consumption by Zone",
                xaxis_title="Zone",
                yaxis_title="Power (kW)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # PUE
            it_power = sum(power_consumption[:-1])
            total_power = sum(power_consumption)
            pue = calculate_pue(it_power, total_power)
            dcie = calculate_dcie(it_power, total_power)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("IT Power", f"{it_power:.1f} kW")
            with col2:
                st.metric("PUE", f"{pue:.2f}")
            with col3:
                st.metric("DCiE", f"{dcie:.1f}%")
    
    with tab3:
        st.subheader("📊 Custom Monitoring Views")
        
        st.write("### 🎨 Create Custom Dashboard")
        
        with st.form("custom_dashboard"):
            dashboard_name = st.text_input("Dashboard Name", "My Dashboard")
            
            st.write("**Select Metrics:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                show_cpu = st.checkbox("CPU Usage", value=True)
                show_memory = st.checkbox("Memory Usage", value=True)
                show_disk = st.checkbox("Disk I/O", value=True)
            
            with col2:
                show_network = st.checkbox("Network Traffic", value=True)
                show_power = st.checkbox("Power Consumption", value=False)
                show_temp = st.checkbox("Temperature", value=True)
            
            with col3:
                show_alerts = st.checkbox("Active Alerts", value=True)
                show_top_servers = st.checkbox("Top Servers", value=False)
                show_capacity = st.checkbox("Capacity Planning", value=False)
            
            refresh_rate = st.selectbox("Refresh Rate", ["5s", "10s", "30s", "1m", "5m", "Manual"])
            
            if st.form_submit_button("💾 Save Dashboard"):
                st.success(f"Dashboard '{dashboard_name}' saved!")
                log_event(f"Custom dashboard created: {dashboard_name}", "INFO")

# ==================== PAGE: AI OPERATIONS ====================
elif page == "🤖 AI Operations":
    st.header("🤖 AI-Powered Operations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictions", "🎯 Optimization", "🔧 Auto-Remediation", "📊 Insights"])
    
    with tab1:
        st.subheader("🔮 AI Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📈 Capacity Forecast")
            
            prediction_horizon = st.selectbox("Forecast Horizon",
                ["1 week", "1 month", "3 months", "6 months", "1 year"])
            
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("AI analyzing patterns..."):
                    import time
                    time.sleep(2)
                    
                    # Générer prédictions
                    days = {"1 week": 7, "1 month": 30, "3 months": 90, "6 months": 180, "1 year": 365}[prediction_horizon]
                    
                    current_servers = total_servers
                    predictions = [current_servers * (1 + 0.02 * i) for i in range(days)]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=list(range(days)),
                        y=predictions,
                        mode='lines',
                        name='Predicted Servers',
                        line=dict(color='#00D9FF', width=3)
                    ))
                    
                    fig.update_layout(
                        title="Server Count Forecast",
                        xaxis_title="Days",
                        yaxis_title="Number of Servers",
                        template="plotly_dark",
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"📊 Forecast: You'll need {int(predictions[-1])} servers in {prediction_horizon}")
        
        with col2:
            st.write("### 🚨 Failure Prediction")
            
            if st.session_state.datacenter['servers']:
                st.write("**High Risk Servers (Next 30 days):**")
                
                for server_id, server in list(st.session_state.datacenter['servers'].items())[:5]:
                    age_days = (datetime.now() - datetime.fromisoformat(server['created_at'])).days
                    metrics = generate_server_metrics(server['type'], np.random.uniform(60, 90))
                    
                    failure_prob = predict_failure_probability(
                        age_days,
                        metrics['cpu_usage'],
                        metrics['temperature_c']
                    )
                    
                    if failure_prob > 30:
                        st.warning(f"⚠️ {server['name']}: {failure_prob:.1f}% failure risk")
                        
                        if st.button(f"🔧 Schedule Maintenance", key=f"maint_pred_{server_id}"):
                            st.success("Maintenance scheduled")
                            log_event(f"Predictive maintenance: {server['name']}", "INFO")
            else:
                st.info("No servers to analyze")
        
        # Anomaly Detection
        st.write("### 🔍 Anomaly Detection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Anomalies Detected (24h)", np.random.randint(0, 10))
        
        with col2:
            st.metric("False Positives", np.random.randint(0, 3))
        
        with col3:
            st.metric("Accuracy", f"{np.random.uniform(92, 98):.1f}%")
    
    with tab2:
        st.subheader("🎯 AI Optimization")
        
        st.write("### ⚡ Workload Optimization")
        
        optimization_type = st.selectbox("Optimization Type",
            ["VM Placement", "Power Efficiency", "Cooling Optimization", "Network Routing"])
        
        if optimization_type == "VM Placement":
            st.write("**Current VM Distribution:**")
            
            if st.session_state.datacenter['vm_instances']:
                vm_count = len(st.session_state.datacenter['vm_instances'])
                server_count = max(total_servers, 1)
                
                st.metric("VMs per Server (avg)", f"{vm_count / server_count:.1f}")
                st.metric("Fragmentation", f"{np.random.uniform(20, 40):.1f}%")
                
                if st.button("🤖 Optimize Placement", type="primary"):
                    with st.spinner("AI optimizing VM placement..."):
                        import time
                        time.sleep(2)
                        
                        st.success("✅ Optimization completed!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Servers Saved", np.random.randint(3, 8))
                        with col2:
                            st.metric("Cost Reduction", f"${np.random.randint(5000, 15000):,}/month")
                        with col3:
                            st.metric("Efficiency Gain", f"{np.random.uniform(15, 30):.1f}%")
                        
                        log_event("VM placement optimized", "SUCCESS")
            else:
                st.info("No VMs to optimize")
        
        elif optimization_type == "Power Efficiency":
            st.write("**Power Optimization Opportunities:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current PUE", f"{np.random.uniform(1.4, 1.6):.2f}")
                st.metric("Potential PUE", f"{np.random.uniform(1.2, 1.3):.2f}")
            
            with col2:
                st.metric("Power Savings", f"{np.random.uniform(50, 100):.0f} kW")
                st.metric("Cost Savings", f"${np.random.randint(20000, 50000):,}/year")
            
            if st.button("⚡ Apply Optimization", type="primary"):
                st.success("Power optimization applied!")
                log_event("Power optimization executed", "SUCCESS")
    
    with tab3:
        st.subheader("🔧 Auto-Remediation")
        
        st.write("### 🤖 Automated Issue Resolution")
        
        # Configuration auto-remediation
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Enable Auto-Remediation:**")
            
            auto_restart = st.checkbox("Auto-restart failed services", value=True)
            auto_scale = st.checkbox("Auto-scale on high load", value=True)
            auto_migrate = st.checkbox("Auto-migrate VMs on failure", value=False)
            auto_clear_logs = st.checkbox("Auto-clear full disks", value=True)
        
        with col2:
            st.write("**Remediation History (24h):**")
            
            st.metric("Actions Taken", np.random.randint(10, 30))
            st.metric("Success Rate", f"{np.random.uniform(90, 98):.1f}%")
            st.metric("Time Saved", f"{np.random.randint(2, 8)} hours")
        
        # Recent actions
        st.write("### 📋 Recent Auto-Remediation Actions")
        
        actions = [
            {"time": "5 min ago", "action": "Restarted hung service", "status": "success"},
            {"time": "23 min ago", "action": "Cleared /var/log on SRV-042", "status": "success"},
            {"time": "1 hour ago", "action": "Scaled up VM pool", "status": "success"},
            {"time": "2 hours ago", "action": "Migrated VM from failed host", "status": "success"}
        ]
        
        for action in actions:
            status_icon = "✅" if action['status'] == 'success' else "❌"
            st.write(f"{status_icon} **{action['time']}**: {action['action']}")
    
    with tab4:
        st.subheader("📊 AI Insights & Recommendations")
        
        st.write("### 💡 Smart Recommendations")
        
        recommendations = [
            {
                "title": "🔋 Power Efficiency",
                "description": "Consolidate workloads on 8 servers during off-peak hours",
                "impact": "Save $25k/year",
                "effort": "Low"
            },
            {
                "title": "❄️ Cooling Optimization",
                "description": "Adjust CRAC setpoints based on predicted load",
                "impact": "Reduce cooling by 15%",
                "effort": "Medium"
            },
            {
                "title": "🔧 Preventive Maintenance",
                "description": "Schedule maintenance for 5 high-risk servers",
                "impact": "Prevent 3-5 failures",
                "effort": "Medium"
            },
            {
                "title": "☁️ Cloud Migration",
                "description": "Migrate 20% of workloads to cloud during peak",
                "impact": "Save $40k/year",
                "effort": "High"
            }
        ]
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{rec['title']}"):
                st.write(f"**Description:** {rec['description']}")
                st.write(f"**Impact:** {rec['impact']}")
                st.write(f"**Effort:** {rec['effort']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Accept", key=f"accept_{i}"):
                        st.success("Recommendation accepted!")
                
                with col2:
                    if st.button("❌ Dismiss", key=f"dismiss_rec_{i}"):
                        st.info("Recommendation dismissed")

# ==================== PAGE: VIRTUALIZATION ====================
elif page == "☁️ Virtualization":
    st.header("☁️ Virtualization Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🖥️ Virtual Machines", "➕ Create VM", "📊 Performance", "🔄 Migration"])
    
    with tab1:
        if not st.session_state.datacenter['vm_instances']:
            st.info("No VMs deployed. Create your first VM!")
        else:
            # Filtres VMs
            col1, col2, col3 = st.columns(3)
            
            with col1:
                os_filter = st.multiselect("OS", 
                    list(set(vm.get('os', 'Linux') for vm in st.session_state.datacenter['vm_instances'].values())))
            
            with col2:
                status_filter = st.multiselect("Status",
                    ["running", "stopped", "suspended"], default=["running"])
            
            with col3:
                search_vm = st.text_input("🔍 Search VM", "")
            
            # Liste VMs
            for vm_id, vm in st.session_state.datacenter['vm_instances'].items():
                if status_filter and vm['status'] not in status_filter:
                    continue
                if search_vm and search_vm.lower() not in vm['name'].lower():
                    continue
                
                with st.expander(f"☁️ {vm['name']} ({vm['status']})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write("### 🖥️ Configuration")
                        st.write(f"**CPU:** {vm['cpu_cores']} cores")
                        st.write(f"**RAM:** {vm['ram_gb']} GB")
                        st.write(f"**Disk:** {vm['disk_gb']} GB")
                        st.write(f"**OS:** {vm['os']}")
                    
                    with col2:
                        st.write("### 📊 Metrics")
                        cpu_usage = np.random.uniform(20, 80)
                        mem_usage = np.random.uniform(30, 75)
                        st.metric("CPU", f"{cpu_usage:.1f}%")
                        st.metric("Memory", f"{mem_usage:.1f}%")
                    
                    with col3:
                        st.write("### 🌐 Network")
                        st.write(f"**IP:** {vm.get('ip_address', 'DHCP')}")
                        st.write(f"**VLAN:** {vm.get('vlan', 'Default')}")
                        st.write(f"**Uptime:** {np.random.randint(1, 100)} days")
                    
                    with col4:
                        st.write("### ⚙️ Actions")
                        
                        if vm['status'] == 'running':
                            if st.button("⏸️ Stop", key=f"stop_vm_{vm_id}"):
                                vm['status'] = 'stopped'
                                st.rerun()
                            
                            if st.button("🔄 Restart", key=f"restart_vm_{vm_id}"):
                                st.info("VM restarting...")
                        else:
                            if st.button("▶️ Start", key=f"start_vm_{vm_id}"):
                                vm['status'] = 'running'
                                st.rerun()
                        
                        if st.button("📸 Snapshot", key=f"snap_vm_{vm_id}"):
                            st.success("Snapshot created!")
    
    with tab2:
        st.subheader("➕ Create Virtual Machine")
        
        with st.form("create_vm"):
            col1, col2 = st.columns(2)
            
            with col1:
                vm_name = st.text_input("VM Name", "VM-WebServer-01")
                
                vm_template = st.selectbox("Template",
                    ["Ubuntu 22.04", "CentOS 8", "Windows Server 2022", "Debian 12", "Custom"])
                
                cpu_cores = st.slider("CPU Cores", 1, 64, 4)
                ram_gb = st.slider("RAM (GB)", 1, 512, 16)
            
            with col2:
                disk_gb = st.slider("Disk (GB)", 10, 2000, 100)
                
                network_type = st.selectbox("Network",
                    ["Bridged", "NAT", "Internal", "Host-only"])
                
                vlan = st.text_input("VLAN", "100")
                
                autostart = st.checkbox("Auto-start on boot", value=True)
            
            if st.form_submit_button("🚀 Create VM", type="primary"):
                vm_id = f"vm_{len(st.session_state.datacenter['vm_instances']) + 1}"
                
                vm = {
                    'id': vm_id,
                    'name': vm_name,
                    'os': vm_template,
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'disk_gb': disk_gb,
                    'network_type': network_type,
                    'vlan': vlan,
                    'ip_address': f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                    'status': 'running',
                    'autostart': autostart,
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.datacenter['vm_instances'][vm_id] = vm
                log_event(f"VM created: {vm_name}", "SUCCESS")
                
                st.success(f"✅ VM '{vm_name}' created!")
                st.balloons()
                st.rerun()
    
    with tab3:
        st.subheader("📊 VM Performance Analytics")
        
        if st.session_state.datacenter['vm_instances']:
            # Agrégation métriques VMs
            vm_data = []
            for vm in st.session_state.datacenter['vm_instances'].values():
                vm_data.append({
                    'Name': vm['name'],
                    'CPU Cores': vm['cpu_cores'],
                    'RAM (GB)': vm['ram_gb'],
                    'Disk (GB)': vm['disk_gb'],
                    'CPU Usage (%)': np.random.uniform(20, 80),
                    'Memory Usage (%)': np.random.uniform(30, 75)
                })
            
            df_vms = pd.DataFrame(vm_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(df_vms, x='CPU Cores', y='RAM (GB)',
                               size='Disk (GB)', hover_data=['Name'],
                               title="VM Resource Allocation",
                               template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df_vms, x='Name', y='CPU Usage (%)',
                           title="CPU Usage by VM",
                           template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
    with tab4:
        st.subheader("🔄 VM Migration")
        
        st.write("### 🚀 Live Migration")
        
        if st.session_state.datacenter['vm_instances'] and total_servers > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                vm_to_migrate = st.selectbox("Select VM",
                    list(st.session_state.datacenter['vm_instances'].keys()),
                    format_func=lambda x: st.session_state.datacenter['vm_instances'][x]['name'])
                
                current_host = st.text_input("Current Host", "HOST-01", disabled=True)
            
            with col2:
                target_host = st.selectbox("Target Host",
                    [f"HOST-{i:02d}" for i in range(2, min(total_servers + 1, 10))])
                
                migration_type = st.selectbox("Migration Type",
                    ["Live Migration", "Cold Migration", "Storage vMotion"])
            
            if st.button("🚀 Start Migration", type="primary"):
                with st.spinner("Migrating VM..."):
                    import time
                    progress = st.progress(0)
                    
                    for i in range(100):
                        time.sleep(0.05)
                        progress.progress(i + 1)
                    
                    progress.empty()
                    st.success(f"✅ VM migrated to {target_host}!")
                    log_event(f"VM migration: {st.session_state.datacenter['vm_instances'][vm_to_migrate]['name']} -> {target_host}", "SUCCESS")
        else:
            st.info("Need multiple servers for migration")
# ==================== PAGE: SERVERS ====================
elif page == "💻 Servers":
    st.header("💻 Server Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Server List", "➕ Add Server", "📊 Performance", "🔧 Bulk Actions"])
    
    with tab1:
        if not st.session_state.datacenter['servers']:
            st.info("No servers configured. Add your first server!")
        else:
            # Filtres
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                type_filter = st.multiselect("Type", list(SERVER_TYPES.keys()))
            
            with col2:
                status_filter = st.multiselect("Status", 
                    ["online", "offline", "maintenance"], default=["online"])
            
            with col3:
                rack_filter = st.multiselect("Rack",
                    [r['name'] for r in st.session_state.datacenter['racks'].values()])
            
            with col4:
                search = st.text_input("🔍 Search", "")
            
            # Liste serveurs
            for server_id, server in st.session_state.datacenter['servers'].items():
                # Appliquer filtres
                if type_filter and server['type'] not in type_filter:
                    continue
                if status_filter and server['status'] not in status_filter:
                    continue
                if search and search.lower() not in server['name'].lower():
                    continue
                
                # Générer métriques
                metrics = generate_server_metrics(server['type'], np.random.uniform(40, 90))
                
                with st.expander(f"💻 {server['name']} ({server['type']}) - {server['status']}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write("### 🖥️ Specs")
                        specs = SERVER_TYPES[server['type']]
                        st.write(f"**CPU:** {specs['cpu_cores']} cores")
                        st.write(f"**RAM:** {specs['ram_gb']} GB")
                        st.write(f"**Storage:** {specs['storage_tb']} TB")
                        st.write(f"**Power:** {specs['power_w']} W")
                    
                    with col2:
                        st.write("### 📊 Utilization")
                        st.metric("CPU", f"{metrics['cpu_usage']:.1f}%")
                        st.metric("Memory", f"{metrics['memory_usage']:.1f}%")
                        st.metric("Disk", f"{metrics['disk_usage']:.1f}%")
                    
                    with col3:
                        st.write("### 🌡️ Monitoring")
                        st.metric("Temperature", f"{metrics['temperature_c']:.1f}°C")
                        st.metric("Power Draw", f"{metrics['power_consumption_w']:.0f} W")
                        
                        # Prédiction panne
                        age_days = (datetime.now() - datetime.fromisoformat(server['created_at'])).days
                        failure_prob = predict_failure_probability(
                            age_days, 
                            metrics['cpu_usage'], 
                            metrics['temperature_c']
                        )
                        st.metric("Failure Risk", f"{failure_prob:.1f}%")
                    
                    with col4:
                        st.write("### ⚙️ Actions")
                        
                        if server['status'] == 'online':
                            if st.button("🔄 Reboot", key=f"reboot_{server_id}"):
                                st.info("Rebooting server...")
                                log_event(f"Server rebooted: {server['name']}", "INFO")
                            
                            if st.button("⏸️ Shutdown", key=f"shutdown_{server_id}"):
                                server['status'] = 'offline'
                                st.warning("Server shutdown")
                                st.rerun()
                        else:
                            if st.button("▶️ Power On", key=f"poweron_{server_id}"):
                                server['status'] = 'online'
                                st.success("Server powered on")
                                st.rerun()
                        
                        if st.button("🗑️ Decommission", key=f"decom_{server_id}"):
                            del st.session_state.datacenter['servers'][server_id]
                            st.success("Server decommissioned")
                            st.rerun()
                    
                    # Graphique métriques temps réel
                    st.write("### 📈 Real-Time Metrics (Last Hour)")
                    
                    time_points = list(range(60))
                    cpu_history = [metrics['cpu_usage'] + np.random.uniform(-10, 10) for _ in time_points]
                    mem_history = [metrics['memory_usage'] + np.random.uniform(-5, 5) for _ in time_points]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=time_points, y=cpu_history,
                        mode='lines', name='CPU', line=dict(color='#00D9FF', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=time_points, y=mem_history,
                        mode='lines', name='Memory', line=dict(color='#FF6B6B', width=2)
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Minutes Ago",
                        yaxis_title="Utilization (%)",
                        template="plotly_dark",
                        height=250
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("➕ Add New Server")
        
        with st.form("add_server"):
            col1, col2 = st.columns(2)
            
            with col1:
                server_name = st.text_input("Server Name", "SRV-001")
                server_type = st.selectbox("Server Type", list(SERVER_TYPES.keys()))
                
                if st.session_state.datacenter['racks']:
                    rack_id = st.selectbox("Rack",
                        list(st.session_state.datacenter['racks'].keys()),
                        format_func=lambda x: st.session_state.datacenter['racks'][x]['name'])
                else:
                    st.error("Please create a rack first")
                    rack_id = None
                
                u_position = st.number_input("U Position", 1, 42, 1)
                u_size = st.number_input("U Size", 1, 4, 1)
            
            with col2:
                os_type = st.selectbox("Operating System",
                    ["Ubuntu 22.04", "CentOS 8", "Windows Server 2022", "RHEL 9", "VMware ESXi 8"])
                
                ip_address = st.text_input("IP Address", "10.0.0.1")
                
                management_ip = st.text_input("Management IP", "10.0.1.1")
                
                warranty_years = st.number_input("Warranty (years)", 1, 5, 3)
            
            if st.form_submit_button("➕ Add Server", type="primary"):
                if rack_id is None:
                    st.error("Please select a rack")
                else:
                    server_id = f"server_{len(st.session_state.datacenter['servers']) + 1}"
                    
                    server = {
                        'id': server_id,
                        'name': server_name,
                        'type': server_type,
                        'rack_id': rack_id,
                        'u_position': u_position,
                        'u_size': u_size,
                        'os': os_type,
                        'ip_address': ip_address,
                        'management_ip': management_ip,
                        'warranty_years': warranty_years,
                        'status': 'online',
                        'created_at': datetime.now().isoformat(),
                        'last_boot': datetime.now().isoformat()
                    }
                    
                    # Mettre à jour rack
                    rack = st.session_state.datacenter['racks'][rack_id]
                    if rack['u_used'] + u_size <= rack['u_capacity']:
                        rack['u_used'] += u_size
                        rack['servers'].append(server_id)
                        rack['power_used_kw'] += SERVER_TYPES[server_type]['power_w'] / 1000
                        
                        st.session_state.datacenter['servers'][server_id] = server
                        log_event(f"Server added: {server_name}", "SUCCESS")
                        
                        st.success(f"✅ Server '{server_name}' added successfully!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Not enough U space in rack")
    
    with tab3:
        st.subheader("📊 Server Performance Analytics")
        
        if st.session_state.datacenter['servers']:
            # Agrégation métriques
            all_metrics = []
            for server in st.session_state.datacenter['servers'].values():
                metrics = generate_server_metrics(server['type'], np.random.uniform(40, 90))
                metrics['name'] = server['name']
                metrics['type'] = server['type']
                all_metrics.append(metrics)
            
            df_metrics = pd.DataFrame(all_metrics)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df_metrics, y='cpu_usage', x='type',
                           title="CPU Usage by Server Type",
                           template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df_metrics, x='cpu_usage', y='temperature_c',
                               size='power_consumption_w', color='type',
                               title="CPU vs Temperature",
                               template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            
            # Top/Bottom performers
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🔥 Highest CPU Usage")
                top_cpu = df_metrics.nlargest(5, 'cpu_usage')[['name', 'cpu_usage']]
                st.dataframe(top_cpu, use_container_width=True)
            
            with col2:
                st.write("### 🌡️ Highest Temperature")
                top_temp = df_metrics.nlargest(5, 'temperature_c')[['name', 'temperature_c']]
                st.dataframe(top_temp, use_container_width=True)
        else:
            st.info("No servers to analyze")
    
    with tab4:
        st.subheader("🔧 Bulk Operations")
        
        if st.session_state.datacenter['servers']:
            operation = st.selectbox("Operation",
                ["Power On All", "Power Off All", "Reboot All", "Update OS", "Security Patch"])
            
            if operation == "Update OS":
                os_version = st.text_input("New OS Version", "Ubuntu 24.04")
            
            selected_servers = st.multiselect("Select Servers",
                list(st.session_state.datacenter['servers'].keys()),
                format_func=lambda x: st.session_state.datacenter['servers'][x]['name'])
            
            if st.button(f"🚀 Execute {operation}", type="primary"):
                if selected_servers:
                    with st.spinner(f"Executing {operation} on {len(selected_servers)} servers..."):
                        import time
                        progress = st.progress(0)
                        
                        for i, server_id in enumerate(selected_servers):
                            time.sleep(0.1)
                            progress.progress((i + 1) / len(selected_servers))
                        
                        progress.empty()
                        st.success(f"✅ {operation} completed on {len(selected_servers)} servers!")
                        log_event(f"Bulk operation: {operation} on {len(selected_servers)} servers", "SUCCESS")
                else:
                    st.warning("Please select at least one server")
        else:
            st.info("No servers available")

# ==================== PAGE: STORAGE ====================
elif page == "💾 Storage":
    st.header("💾 Storage Management")
    
    tab1, tab2, tab3 = st.tabs(["📊 Storage Overview", "➕ Add Storage", "📈 Analytics"])
    
    with tab1:
        if not st.session_state.datacenter['storage_systems']:
            st.info("No storage systems configured")
        else:
            for storage_id, storage in st.session_state.datacenter['storage_systems'].items():
                with st.expander(f"💾 {storage['name']} ({storage['type']})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write("### 📊 Capacity")
                        used_pct = (storage['used_tb'] / storage['capacity_tb']) * 100
                        st.metric("Used", f"{storage['used_tb']:.1f}/{storage['capacity_tb']} TB")
                        st.progress(used_pct / 100)
                        st.metric("Available", f"{storage['capacity_tb'] - storage['used_tb']:.1f} TB")
                    
                    with col2:
                        st.write("### ⚡ Performance")
                        st.metric("IOPS", f"{storage['iops']:,}")
                        st.metric("Throughput", f"{storage['throughput_gbps']} GB/s")
                        st.metric("Latency", f"{storage['latency_ms']:.2f} ms")
                    
                    with col3:
                        st.write("### 🔐 Data Protection")
                        st.write(f"**RAID:** {storage['raid_level']}")
                        st.write(f"**Replication:** {storage['replication']}")
                        st.write(f"**Snapshots:** {storage.get('snapshots', 0)}")
                    
                    with col4:
                        st.write("### ⚙️ Status")
                        st.write(f"**Status:** {storage['status']}")
                        st.write(f"**Health:** {storage.get('health', 100)}%")
                        
                        if st.button("📊 Details", key=f"storage_details_{storage_id}"):
                            st.info("Storage details")
    
    with tab2:
        st.subheader("➕ Add Storage System")
        
        with st.form("add_storage"):
            col1, col2 = st.columns(2)
            
            with col1:
                storage_name = st.text_input("Storage Name", "SAN-001")
                storage_type = st.selectbox("Type",
                    ["SAN", "NAS", "Object Storage", "Block Storage", "File Storage"])
                capacity_tb = st.number_input("Capacity (TB)", 1, 1000, 100)
                raid_level = st.selectbox("RAID Level",
                    ["RAID 0", "RAID 1", "RAID 5", "RAID 6", "RAID 10"])
            
            with col2:
                iops = st.number_input("IOPS", 1000, 1000000, 50000, 1000)
                throughput_gbps = st.number_input("Throughput (GB/s)", 1, 100, 10)
                replication = st.selectbox("Replication",
                    ["None", "Sync", "Async", "3-Way"])
                encryption = st.checkbox("Encryption at Rest", value=True)
            
            if st.form_submit_button("➕ Add Storage", type="primary"):
                storage_id = f"storage_{len(st.session_state.datacenter['storage_systems']) + 1}"
                
                storage = {
                    'id': storage_id,
                    'name': storage_name,
                    'type': storage_type,
                    'capacity_tb': capacity_tb,
                    'used_tb': 0,
                    'raid_level': raid_level,
                    'iops': iops,
                    'throughput_gbps': throughput_gbps,
                    'latency_ms': np.random.uniform(0.5, 2.0),
                    'replication': replication,
                    'encryption': encryption,
                    'status': 'online',
                    'health': 100,
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.datacenter['storage_systems'][storage_id] = storage
                log_event(f"Storage system added: {storage_name}", "SUCCESS")
                
                st.success(f"✅ Storage '{storage_name}' added!")
                st.balloons()
                st.rerun()
    
    with tab3:
        st.subheader("📈 Storage Analytics")
        
        if st.session_state.datacenter['storage_systems']:
            # Calculs agrégés
            total_capacity = sum(s['capacity_tb'] for s in st.session_state.datacenter['storage_systems'].values())
            total_used = sum(s['used_tb'] for s in st.session_state.datacenter['storage_systems'].values())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Capacity", f"{total_capacity:.1f} TB")
            with col2:
                st.metric("Total Used", f"{total_used:.1f} TB")
            with col3:
                st.metric("Available", f"{total_capacity - total_used:.1f} TB")
            with col4:
                usage_pct = (total_used / total_capacity * 100) if total_capacity > 0 else 0
                st.metric("Usage", f"{usage_pct:.1f}%")
            
            # Graphique utilisation
            storage_data = []
            for storage in st.session_state.datacenter['storage_systems'].values():
                storage_data.append({
                    'name': storage['name'],
                    'used': storage['used_tb'],
                    'free': storage['capacity_tb'] - storage['used_tb']
                })
            
            df_storage = pd.DataFrame(storage_data)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(name='Used', x=df_storage['name'], y=df_storage['used'],
                               marker_color='#FF6B6B'))
            fig.add_trace(go.Bar(name='Free', x=df_storage['name'], y=df_storage['free'],
                               marker_color='#00D9FF'))
            
            fig.update_layout(
                barmode='stack',
                title="Storage Utilization by System",
                xaxis_title="Storage System",
                yaxis_title="Capacity (TB)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No storage systems to analyze")

# ==================== PAGE: NETWORK ====================
elif page == "🌐 Network":
    st.header("🌐 Network Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔌 Devices", "📊 Traffic", "🔒 Security", "🗺️ Topology"])
    
    with tab1:
        st.subheader("🔌 Network Devices")
        
        # Ajouter device réseau
        with st.expander("➕ Add Network Device"):
            with st.form("add_network_device"):
                col1, col2 = st.columns(2)
                
                with col1:
                    device_name = st.text_input("Device Name", "CORE-SW-01")
                    device_type = st.selectbox("Type",
                        ["Core Switch", "Distribution Switch", "Access Switch", 
                         "Router", "Firewall", "Load Balancer"])
                    ports = st.number_input("Ports", 1, 96, 48)
                
                with col2:
                    speed_gbps = st.selectbox("Port Speed", [1, 10, 25, 40, 100])
                    management_ip = st.text_input("Management IP", "10.0.0.254")
                    redundancy = st.selectbox("Redundancy", ["None", "Active-Standby", "Active-Active"])
                
                if st.form_submit_button("➕ Add Device"):
                    device_id = f"net_{len(st.session_state.datacenter['network_devices']) + 1}"
                    
                    device = {
                        'id': device_id,
                        'name': device_name,
                        'type': device_type,
                        'ports': ports,
                        'ports_used': 0,
                        'speed_gbps': speed_gbps,
                        'management_ip': management_ip,
                        'redundancy': redundancy,
                        'status': 'online',
                        'cpu_usage': 0,
                        'memory_usage': 0,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.datacenter['network_devices'][device_id] = device
                    log_event(f"Network device added: {device_name}", "SUCCESS")
                    
                    st.success("✅ Device added!")
                    st.rerun()
        
        # Liste devices
        if st.session_state.datacenter['network_devices']:
            for device_id, device in st.session_state.datacenter['network_devices'].items():
                with st.expander(f"🔌 {device['name']} ({device['type']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Port Usage")
                        port_usage_pct = (device['ports_used'] / device['ports']) * 100
                        st.metric("Ports", f"{device['ports_used']}/{device['ports']}")
                        st.progress(port_usage_pct / 100)
                    
                    with col2:
                        st.write("### ⚡ Performance")
                        device['cpu_usage'] = np.random.uniform(10, 60)
                        device['memory_usage'] = np.random.uniform(20, 70)
                        st.metric("CPU", f"{device['cpu_usage']:.1f}%")
                        st.metric("Memory", f"{device['memory_usage']:.1f}%")
                    
                    with col3:
                        st.write("### 🔧 Config")
                        st.write(f"**Speed:** {device['speed_gbps']} Gbps")
                        st.write(f"**Redundancy:** {device['redundancy']}")
                        st.write(f"**Status:** {device['status']}")
        else:
            st.info("No network devices configured")
    
    with tab2:
        st.subheader("📊 Network Traffic")
        
        # Simuler trafic
        hours = 24
        traffic = simulate_network_traffic(hours)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(hours)),
            y=traffic,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#00D9FF', width=3),
            name='Traffic (Gbps)'
        ))
        
        fig.update_layout(
            title="Network Traffic (24h)",
            xaxis_title="Hour",
            yaxis_title="Traffic (Gbps)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques trafic
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current", f"{traffic[-1]:.1f} Gbps")
        with col2:
            st.metric("Average", f"{np.mean(traffic):.1f} Gbps")
        with col3:
            st.metric("Peak", f"{max(traffic):.1f} Gbps")
        with col4:
            st.metric("Min", f"{min(traffic):.1f} Gbps")
    
    with tab3:
        st.subheader("🔒 Network Security")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🛡️ Security Status")
            
            st.metric("Firewall Rules", "1,247")
            st.metric("Active Sessions", "12,543")
            st.metric("Blocked Attacks", "156 (24h)")
            st.metric("DDoS Protection", "✅ Active")
        
        with col2:
            st.write("### 🚨 Recent Security Events")
            
            events = [
                {"time": "2 min ago", "event": "Port scan detected", "severity": "warning"},
                {"time": "15 min ago", "event": "Brute force attempt blocked", "severity": "critical"},
                {"time": "1 hour ago", "event": "Suspicious traffic pattern", "severity": "info"}
            ]
            
            for event in events:
                severity_icon = "🔴" if event['severity'] == 'critical' else "🟡" if event['severity'] == 'warning' else "🔵"
                st.write(f"{severity_icon} **{event['time']}**: {event['event']}")
    
    with tab4:
        st.subheader("🗺️ Network Topology")
        
        st.info("Interactive network topology map would be displayed here")
        st.write("Features:")
        st.write("• Real-time device status")
        st.write("• Link utilization visualization")
        st.write("• Redundancy paths")
        st.write("• Failure simulation")

# ==================== PAGE: MONITORING ====================
elif page == "📊 Monitoring":
    st.header("📊 Real-Time Monitoring")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboards", "🎯 Metrics", "📊 Custom Views", "🔔 Alerting"])
    
    with tab1:
        st.subheader("📈 Monitoring Dashboards")
        
        # Refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            dashboard = st.selectbox("Dashboard",
                ["Overview", "Infrastructure", "Applications", "Security", "Custom"])
        
        with col2:
            refresh_interval = st.selectbox("Refresh", ["5s", "10s", "30s", "1m", "5m"])
        
        with col3:
            if st.button("🔄 Refresh Now"):
                st.rerun()
        
        # Métriques temps réel
        st.write("### ⚡ Real-Time Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        metrics_current = {
            'cpu': np.random.uniform(40, 80),
            'memory': np.random.uniform(50, 85),
            'disk': np.random.uniform(30, 70),
            'network': np.random.uniform(20, 60),
            'power': np.random.uniform(50, 90),
            'temp': np.random.uniform(22, 28)
        }
                                
        with col1:
            st.metric("CPU", f"{metrics_current['cpu']:.1f}%", 
                     f"{np.random.uniform(-5, 5):.1f}%")
        
        with col2:
            st.metric("Memory", f"{metrics_current['memory']:.1f}%",
                     f"{np.random.uniform(-3, 3):.1f}%")
        
        with col3:
            st.metric("Disk I/O", f"{metrics_current['disk']:.1f}%",
                     f"{np.random.uniform(-2, 2):.1f}%")
        
        with col4:
            st.metric("Network", f"{metrics_current['network']:.1f} Gbps",
                     f"{np.random.uniform(-5, 5):.1f} Gbps")
        
        with col5:
            st.metric("Power", f"{metrics_current['power']:.1f} kW",
                     f"{np.random.uniform(-2, 2):.1f} kW")
        
        with col6:
            st.metric("Temp", f"{metrics_current['temp']:.1f}°C",
                     f"{np.random.uniform(-0.5, 0.5):.1f}°C")

    with tab4:
        st.subheader("🔔 Alert Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### 🚨 Active Alerts")
            
            if st.session_state.datacenter['alerts']:
                for i, alert in enumerate(st.session_state.datacenter['alerts']):
                    severity = alert.get('severity', 'info')
                    
                    if severity == 'critical':
                        icon = "🔴"
                        color = "#FF0000"
                    elif severity == 'warning':
                        icon = "🟡"
                        color = "#FFA500"
                    else:
                        icon = "🔵"
                        color = "#00D9FF"
                    
                    with st.expander(f"{icon} {alert.get('message', 'Alert')} - {alert.get('timestamp', '')}"):
                        st.write(f"**Source:** {alert.get('source', 'Unknown')}")
                        st.write(f"**Severity:** {severity}")
                        st.write(f"**Details:** {alert.get('details', 'No details')}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("✅ Acknowledge", key=f"ack_{i}"):
                                st.info("Alert acknowledged")
                        
                        with col2:
                            if st.button("🗑️ Dismiss", key=f"dismiss_{i}"):
                                st.session_state.datacenter['alerts'].remove(alert)
                                st.rerun()
            else:
                st.success("✅ No active alerts")
        
        with col2:
            st.write("### ⚙️ Alert Rules")
            
            with st.form("add_alert_rule"):
                rule_name = st.text_input("Rule Name", "High CPU")
                
                metric = st.selectbox("Metric",
                    ["CPU Usage", "Memory Usage", "Disk Usage", "Temperature", "Power"])
                
                condition = st.selectbox("Condition",
                    [">", "<", ">=", "<=", "=="])
                
                threshold = st.number_input("Threshold", 0, 100, 80)
                
                severity = st.selectbox("Severity",
                    ["info", "warning", "critical"])
                
                if st.form_submit_button("➕ Add Rule"):
                    st.success("Alert rule added!")
                    log_event(f"Alert rule created: {rule_name}", "INFO")
# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 System Event Log (Last 20 Events)"):
    if st.session_state.datacenter['log']:
        for event in st.session_state.datacenter['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️" if level == "WARNING" else "❌"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("No events logged")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>🏢 Datacenter Management Platform</h3>
        <p>Advanced Infrastructure Management • AI-Powered • Real-Time Monitoring</p>
        <p><small>Infrastructure • Virtualization • Security • Automation • Analytics</small></p>
        <p><small>Version 1.0.0 | Enterprise Edition</small></p>
        <p><small>🚀 Powered by AI & Advanced Analytics © 2024</small></p>
    </div>
""", unsafe_allow_html=True)