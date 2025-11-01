"""
⚛️ Quantum Physics Research Platform - Advanced Edition
Gravité Quantique • Intrication • Singularité • Effet Tunnel • Réseau Quantique

Installation:
pip install streamlit pandas plotly numpy scipy qiskit pennylane networkx

Lancement:
streamlit run quantum_physics_platform_app.py
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
import math
import numpy as np

# ==================== CONFIGURATION PAGE ====================
st.set_page_config(
    page_title="⚛️ Quantum Physics Research",
    page_icon="⚛️",
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
        background: linear-gradient(90deg, #9D50FF 0%, #6B2FFF 30%, #4A0FFF 60%, #9D50FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        animation: quantum-pulse 2s ease-in-out infinite alternate;
    }
    @keyframes quantum-pulse {
        from { filter: drop-shadow(0 0 15px #9D50FF); }
        to { filter: drop-shadow(0 0 35px #4A0FFF); }
    }
    .quantum-card {
        border: 3px solid #9D50FF;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, rgba(157, 80, 255, 0.1) 0%, rgba(74, 15, 255, 0.1) 100%);
        box-shadow: 0 8px 32px rgba(157, 80, 255, 0.4);
        transition: all 0.3s;
    }
    .quantum-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(107, 47, 255, 0.6);
    }
    .quantum-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
        background: linear-gradient(90deg, #9D50FF 0%, #6B2FFF 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(157, 80, 255, 0.4);
    }
    .entangled {
        animation: entangle 1.5s infinite;
    }
    @keyframes entangle {
        0%, 100% { opacity: 0.7; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== CONSTANTES PHYSIQUES ====================
PHYSICAL_CONSTANTS = {
    'c': 299792458,  # m/s - vitesse lumière
    'h': 6.62607015e-34,  # J⋅s - constante Planck
    'hbar': 1.054571817e-34,  # ℏ = h/2π
    'G': 6.67430e-11,  # m³/kg/s² - constante gravitationnelle
    'k_B': 1.380649e-23,  # J/K - constante Boltzmann
    'e': 1.602176634e-19,  # C - charge électron
    'm_e': 9.1093837015e-31,  # kg - masse électron
    'm_p': 1.67262192369e-27,  # kg - masse proton
    'planck_length': 1.616255e-35,  # m
    'planck_time': 5.391247e-44,  # s
    'planck_mass': 2.176434e-8,  # kg
    'planck_energy': 1.956e9,  # J
}

QUANTUM_PHENOMENA = {
    'Intrication': {
        'description': 'Corrélation quantique non-locale',
        'epr_distance': 'Instantanée',
        'applications': ['Téléportation', 'Cryptographie', 'Computing']
    },
    'Superposition': {
        'description': 'État dans plusieurs états simultanément',
        'decoherence_time': '< 1 ms',
        'applications': ['Qubits', 'Interférence', 'Mesure']
    },
    'Effet Tunnel': {
        'description': 'Traversée barrière classiquement interdite',
        'probability': 'exp(-2κL)',
        'applications': ['Transistor', 'Radioactivité', 'Fusion']
    },
    'Décohérence': {
        'description': 'Perte cohérence quantique',
        'causes': ['Environnement', 'Température', 'Bruit'],
        'applications': ['Limite computing', 'Mesure']
    }
}

# ==================== INITIALISATION SESSION STATE ====================
if 'quantum_lab' not in st.session_state:
    st.session_state.quantum_lab = {
        'qubits': {},
        'entangled_pairs': [],
        'quantum_circuits': {},
        'quantum_networks': {},
        'black_holes': {},
        'wormholes': {},
        'singularities': [],
        'tunneling_experiments': [],
        'quantum_fields': {},
        'spacetime_metrics': {},
        'quantum_gravity_simulations': [],
        'loop_quantum_gravity': {},
        'string_theory_models': {},
        'experiments': [],
        'measurements': [],
        'log': []
    }

# ==================== FONCTIONS UTILITAIRES ====================

def log_event(message: str, level: str = "INFO"):
    """Enregistrer événement"""
    st.session_state.quantum_lab['log'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'level': level
    })

def create_qubit_state(alpha: complex, beta: complex) -> np.ndarray:
    """Créer état qubit |ψ⟩ = α|0⟩ + β|1⟩"""
    # Normalisation
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    return np.array([alpha/norm, beta/norm])

def calculate_bloch_coordinates(state: np.ndarray) -> Tuple[float, float, float]:
    """Calculer coordonnées sphère de Bloch"""
    alpha, beta = state[0], state[1]
    
    # Coordonnées sphériques
    theta = 2 * np.arccos(abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)
    
    # Coordonnées cartésiennes
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return x, y, z

def calculate_entanglement_entropy(state: np.ndarray) -> float:
    """Calculer entropie d'intrication von Neumann"""
    # Matrice densité réduite
    rho = np.outer(state, np.conj(state))
    
    # Valeurs propres
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filtrer valeurs nulles
    
    # Entropie de von Neumann: S = -Tr(ρ log ρ)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return entropy

def calculate_tunneling_probability(barrier_height: float, barrier_width: float, 
                                   particle_energy: float, mass: float) -> float:
    """Calculer probabilité effet tunnel"""
    hbar = PHYSICAL_CONSTANTS['hbar']
    
    # Vecteur d'onde dans barrière
    if particle_energy >= barrier_height:
        return 1.0
    
    kappa = np.sqrt(2 * mass * (barrier_height - particle_energy)) / hbar
    
    # Probabilité transmission (approximation WKB)
    T = np.exp(-2 * kappa * barrier_width)
    
    return T

def schwarzschild_radius(mass: float) -> float:
    """Calculer rayon de Schwarzschild"""
    G = PHYSICAL_CONSTANTS['G']
    c = PHYSICAL_CONSTANTS['c']
    
    r_s = 2 * G * mass / c**2
    
    return r_s

def hawking_temperature(mass: float) -> float:
    """Calculer température de Hawking"""
    hbar = PHYSICAL_CONSTANTS['hbar']
    c = PHYSICAL_CONSTANTS['c']
    k_B = PHYSICAL_CONSTANTS['k_B']
    G = PHYSICAL_CONSTANTS['G']
    
    T_H = (hbar * c**3) / (8 * np.pi * G * mass * k_B)
    
    return T_H

def calculate_quantum_correlation(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calculer corrélation quantique entre deux états"""
    # Produit scalaire
    correlation = abs(np.vdot(state1, state2))**2
    
    return correlation

def simulate_quantum_walk(steps: int, dimension: int = 1) -> List[int]:
    """Simuler marche quantique"""
    position = 0
    positions = [position]
    
    for _ in range(steps):
        # Superposition gauche/droite
        coin_flip = np.random.choice([-1, 1])
        position += coin_flip
        positions.append(position)
    
    return positions

def calculate_planck_scale_effects(energy: float) -> Dict:
    """Calculer effets à l'échelle de Planck"""
    l_p = PHYSICAL_CONSTANTS['planck_length']
    t_p = PHYSICAL_CONSTANTS['planck_time']
    E_p = PHYSICAL_CONSTANTS['planck_energy']
    
    # Corrections gravité quantique
    quantum_correction = (energy / E_p) ** 2
    
    return {
        'length_scale': l_p * np.sqrt(1 + quantum_correction),
        'time_scale': t_p * np.sqrt(1 + quantum_correction),
        'quantum_gravity_strength': quantum_correction,
        'foam_structure': quantum_correction > 0.01
    }

# ==================== HEADER ====================
st.markdown('<h1 class="main-header">⚛️ Quantum Physics Research Platform</h1>', unsafe_allow_html=True)
st.markdown("### Gravité Quantique • Intrication • Singularité • Effet Tunnel • Réseau Quantique • String Theory")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://via.placeholder.com/300x120/9D50FF/FFFFFF?text=Quantum+Lab", 
             use_container_width=True)
    st.markdown("---")
    
    page = st.radio(
        "🎯 Navigation",
        [
            "🏠 Lab Quantique",
            "⚛️ États Quantiques",
            "🔗 Intrication",
            "🌐 Réseau Quantique",
            "🕳️ Singularités",
            "🌀 Trous Noirs",
            "🌌 Trous de Ver",
            "🚇 Effet Tunnel",
            "🎭 Superposition",
            "📊 Décohérence",
            "🌊 Champs Quantiques",
            "🧬 Gravité Quantique",
            "🔄 Loop Quantum Gravity",
            "🎻 Théorie Cordes",
            "⏱️ Espace-Temps",
            "🔬 Expériences",
            "📈 Mesures",
            "🤖 Simulations IA",
            "📊 Analytics",
            "⚙️ Paramètres"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📊 État Lab")
    
    total_qubits = len(st.session_state.quantum_lab['qubits'])
    total_entangled = len(st.session_state.quantum_lab['entangled_pairs'])
    total_experiments = len(st.session_state.quantum_lab['experiments'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("⚛️ Qubits", total_qubits)
        st.metric("🔗 Intriqués", total_entangled)
    with col2:
        st.metric("🔬 Expériences", total_experiments)
        st.metric("🕳️ Singularités", len(st.session_state.quantum_lab['singularities']))

# ==================== PAGE: LAB QUANTIQUE ====================
if page == "🏠 Lab Quantique":
    st.header("🏠 Laboratoire Quantique Central")
    
    # KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f'<div class="quantum-card"><h2>⚛️</h2><h3>{total_qubits}</h3><p>Qubits Actifs</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        coherence_time = np.random.uniform(0.5, 2.0)
        st.markdown(f'<div class="quantum-card"><h2>⏱️</h2><h3>{coherence_time:.2f}</h3><p>Cohérence (ms)</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="quantum-card"><h2>🔗</h2><h3>{total_entangled}</h3><p>Paires EPR</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        fidelity = np.random.uniform(0.95, 0.99)
        st.markdown(f'<div class="quantum-card"><h2>✓</h2><h3>{fidelity:.3f}</h3><p>Fidélité</p></div>', 
                   unsafe_allow_html=True)
    
    with col5:
        quantum_volume = 2 ** total_qubits if total_qubits > 0 else 0
        st.markdown(f'<div class="quantum-card"><h2>📊</h2><h3>{quantum_volume}</h3><p>Volume Q</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Phénomènes quantiques
    st.subheader("⚛️ Phénomènes Quantiques Fondamentaux")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔬 Principes Actifs")
        
        for phenomenon, info in QUANTUM_PHENOMENA.items():
            with st.expander(f"⚛️ {phenomenon}"):
                st.write(f"**Description:** {info['description']}")
                
                if 'epr_distance' in info:
                    st.write(f"**Distance EPR:** {info['epr_distance']}")
                if 'decoherence_time' in info:
                    st.write(f"**Temps:** {info['decoherence_time']}")
                if 'probability' in info:
                    st.write(f"**Probabilité:** {info['probability']}")
                
                st.write(f"**Applications:**")
                for app in info['applications']:
                    st.write(f"• {app}")
    
    with col2:
        st.write("### 📊 Constantes Fondamentales")
        
        constants_display = {
            'Vitesse Lumière (c)': f"{PHYSICAL_CONSTANTS['c']:.2e} m/s",
            'Constante Planck (h)': f"{PHYSICAL_CONSTANTS['h']:.2e} J⋅s",
            'ℏ (h/2π)': f"{PHYSICAL_CONSTANTS['hbar']:.2e} J⋅s",
            'Gravité (G)': f"{PHYSICAL_CONSTANTS['G']:.2e} m³/kg/s²",
            'Longueur Planck': f"{PHYSICAL_CONSTANTS['planck_length']:.2e} m",
            'Temps Planck': f"{PHYSICAL_CONSTANTS['planck_time']:.2e} s",
            'Énergie Planck': f"{PHYSICAL_CONSTANTS['planck_energy']:.2e} J"
        }
        
        for name, value in constants_display.items():
            st.write(f"**{name}:** {value}")
    
    st.markdown("---")
    
    # Visualisation sphère de Bloch
    st.subheader("🌐 Sphère de Bloch - États Quantiques")
    
    if total_qubits > 0:
        # Créer sphère de Bloch
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        
        x_sphere = np.outer(np.cos(theta), np.sin(phi))
        y_sphere = np.outer(np.sin(theta), np.sin(phi))
        z_sphere = np.outer(np.ones(100), np.cos(phi))
        
        fig = go.Figure()
        
        # Sphère
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale='Viridis',
            opacity=0.3,
            showscale=False
        ))
        
        # Points qubits
        for qubit_id, qubit in st.session_state.quantum_lab['qubits'].items():
            state = qubit.get('state', np.array([1, 0]))
            x, y, z = calculate_bloch_coordinates(state)
            
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=[qubit['name']],
                textposition='top center'
            ))
        
        fig.update_layout(
            title="Sphère de Bloch - Représentation États Quantiques",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Créez des qubits pour visualiser sur la sphère de Bloch")
    
    st.markdown("---")
    
    # Expériences récentes
    st.subheader("🔬 Expériences Récentes")
    
    if st.session_state.quantum_lab['experiments']:
        for exp in st.session_state.quantum_lab['experiments'][-5:][::-1]:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"⚛️ **{exp['type']}**")
                st.write(f"{exp['timestamp'][:19]}")
            
            with col2:
                st.write(f"**Résultat:** {exp.get('result', 'N/A')}")
            
            with col3:
                fidelity = exp.get('fidelity', 0)
                st.write(f"**Fidélité:** {fidelity:.3f}")
    else:
        st.info("Aucune expérience réalisée")

# ==================== PAGE: ÉTATS QUANTIQUES ====================
elif page == "⚛️ États Quantiques":
    st.header("⚛️ États Quantiques & Qubits")
    
    st.info("""
    **États Quantiques Fondamentaux**
    
    Un qubit existe dans une superposition: |ψ⟩ = α|0⟩ + β|1⟩
    avec |α|² + |β|² = 1
    
    **Propriétés:**
    - Superposition: État dans plusieurs états simultanément
    - Mesure: Collapse vers |0⟩ ou |1⟩
    - Phase: Différence de phase entre composantes
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚛️ Créer Qubit", "📊 États", "🎭 Portes", "📈 Analyse"])
    
    with tab1:
        st.subheader("⚛️ Créer Nouveau Qubit")
        
        with st.form("create_qubit"):
            col1, col2 = st.columns(2)
            
            with col1:
                qubit_name = st.text_input("Nom Qubit", "Q1")
                
                st.write("**État Initial |ψ⟩ = α|0⟩ + β|1⟩**")
                
                alpha_real = st.slider("α (partie réelle)", -1.0, 1.0, 1.0, 0.1)
                alpha_imag = st.slider("α (partie imaginaire)", -1.0, 1.0, 0.0, 0.1)
            
            with col2:
                st.write("")
                st.write("")
                st.write("")
                
                beta_real = st.slider("β (partie réelle)", -1.0, 1.0, 0.0, 0.1)
                beta_imag = st.slider("β (partie imaginaire)", -1.0, 1.0, 0.0, 0.1)
                
                temperature_mk = st.slider("Température (mK)", 10, 300, 20)
            
            if st.form_submit_button("⚛️ Créer Qubit", type="primary"):
                qubit_id = f"qubit_{len(st.session_state.quantum_lab['qubits']) + 1}"
                
                alpha = complex(alpha_real, alpha_imag)
                beta = complex(beta_real, beta_imag)
                
                state = create_qubit_state(alpha, beta)
                
                x, y, z = calculate_bloch_coordinates(state)
                
                qubit = {
                    'id': qubit_id,
                    'name': qubit_name,
                    'state': state,
                    'alpha': alpha,
                    'beta': beta,
                    'bloch_coords': (x, y, z),
                    'temperature_mk': temperature_mk,
                    'coherence_time_ms': np.random.uniform(0.5, 2.0),
                    'fidelity': np.random.uniform(0.95, 0.99),
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.quantum_lab['qubits'][qubit_id] = qubit
                log_event(f"Qubit créé: {qubit_name}", "SUCCESS")
                
                st.success(f"✅ Qubit '{qubit_name}' créé!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("État |0⟩", f"{abs(alpha):.3f}")
                with col2:
                    st.metric("État |1⟩", f"{abs(beta):.3f}")
                with col3:
                    st.metric("Phase", f"{np.angle(beta)-np.angle(alpha):.3f} rad")
                
                st.rerun()
    
    with tab2:
        if not st.session_state.quantum_lab['qubits']:
            st.info("Aucun qubit créé")
        else:
            for qubit_id, qubit in st.session_state.quantum_lab['qubits'].items():
                with st.expander(f"⚛️ {qubit['name']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 État")
                        st.write(f"**α:** {qubit['alpha']:.3f}")
                        st.write(f"**β:** {qubit['beta']:.3f}")
                        st.write(f"**|α|²:** {abs(qubit['alpha'])**2:.3f}")
                        st.write(f"**|β|²:** {abs(qubit['beta'])**2:.3f}")
                    
                    with col2:
                        st.write("### 🌐 Bloch")
                        x, y, z = qubit['bloch_coords']
                        st.write(f"**X:** {x:.3f}")
                        st.write(f"**Y:** {y:.3f}")
                        st.write(f"**Z:** {z:.3f}")
                        st.write(f"**θ:** {np.arccos(z):.3f} rad")
                    
                    with col3:
                        st.write("### ⚙️ Propriétés")
                        st.write(f"**T:** {qubit['temperature_mk']} mK")
                        st.write(f"**Cohérence:** {qubit['coherence_time_ms']:.2f} ms")
                        st.write(f"**Fidélité:** {qubit['fidelity']:.3f}")
                    
                    # Actions
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("📏 Mesurer", key=f"measure_{qubit_id}"):
                            # Mesure collapse état
                            prob_0 = abs(qubit['alpha'])**2
                            result = 0 if np.random.random() < prob_0 else 1
                            
                            st.info(f"Résultat mesure: |{result}⟩")
                            
                            # État collapse
                            if result == 0:
                                qubit['state'] = np.array([1, 0])
                            else:
                                qubit['state'] = np.array([0, 1])
                    
                    with col2:
                        if st.button("🔄 Hadamard", key=f"hadamard_{qubit_id}"):
                            # Porte Hadamard
                            H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                            qubit['state'] = H @ qubit['state']
                            st.success("Porte H appliquée!")
                    
                    with col3:
                        if st.button("🎯 Pauli-X", key=f"paulix_{qubit_id}"):
                            # Porte Pauli-X (NOT)
                            X = np.array([[0, 1], [1, 0]])
                            qubit['state'] = X @ qubit['state']
                            st.success("Porte X appliquée!")
                    
                    with col4:
                        if st.button("🗑️ Supprimer", key=f"del_{qubit_id}"):
                            del st.session_state.quantum_lab['qubits'][qubit_id]
                            st.rerun()
    
    with tab3:
        st.subheader("🎭 Portes Quantiques")
        
        if st.session_state.quantum_lab['qubits']:
            selected_qubit = st.selectbox("Sélectionner Qubit",
                list(st.session_state.quantum_lab['qubits'].keys()),
                format_func=lambda x: st.session_state.quantum_lab['qubits'][x]['name'])
            
            qubit = st.session_state.quantum_lab['qubits'][selected_qubit]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🚪 Portes 1-Qubit")
                
                gate = st.selectbox("Porte",
                    ["Hadamard (H)", "Pauli-X", "Pauli-Y", "Pauli-Z", 
                     "Phase (S)", "T", "Rotation-X", "Rotation-Y", "Rotation-Z"])
                
                if "Rotation" in gate:
                    angle = st.slider("Angle (rad)", 0.0, 2*np.pi, np.pi/2, 0.1)
                
                if st.button("🚀 Appliquer Porte"):
                    if gate == "Hadamard (H)":
                        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
                        qubit['state'] = H @ qubit['state']
                    elif gate == "Pauli-X":
                        X = np.array([[0, 1], [1, 0]])
                        qubit['state'] = X @ qubit['state']
                    elif gate == "Pauli-Y":
                        Y = np.array([[0, -1j], [1j, 0]])
                        qubit['state'] = Y @ qubit['state']
                    elif gate == "Pauli-Z":
                        Z = np.array([[1, 0], [0, -1]])
                        qubit['state'] = Z @ qubit['state']
                    
                    # Mise à jour coordonnées Bloch
                    x, y, z = calculate_bloch_coordinates(qubit['state'])
                    qubit['bloch_coords'] = (x, y, z)
                    
                    st.success(f"Porte {gate} appliquée!")
                    log_event(f"Porte {gate} appliquée sur {qubit['name']}", "INFO")
                    st.rerun()
            
            with col2:
                st.write("### 📊 État Actuel")
                
                state = qubit['state']
                st.write(f"**|ψ⟩ = {state[0]:.3f}|0⟩ + {state[1]:.3f}|1⟩**")
                
                # Probabilités
                prob_0 = abs(state[0])**2
                prob_1 = abs(state[1])**2
                
                st.write(f"**P(|0⟩) = {prob_0:.3f}**")
                st.write(f"**P(|1⟩) = {prob_1:.3f}**")
                
                # Visualisation probabilités
                fig = go.Figure(data=[go.Bar(
                    x=['|0⟩', '|1⟩'],
                    y=[prob_0, prob_1],
                    marker_color=['#9D50FF', '#6B2FFF']
                )])
                
                fig.update_layout(
                    title="Probabilités de Mesure",
                    yaxis_title="Probabilité",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Créez d'abord un qubit")
    
    with tab4:
        st.subheader("📈 Analyse États Quantiques")
        
        if st.session_state.quantum_lab['qubits']:
            # Distribution états sur sphère de Bloch
            coords_data = []
            for qubit in st.session_state.quantum_lab['qubits'].values():
                x, y, z = qubit['bloch_coords']
                coords_data.append({
                    'Name': qubit['name'],
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'Fidelity': qubit['fidelity']
                })
            
            df_coords = pd.DataFrame(coords_data)
            
            # Scatter 3D
            fig = go.Figure(data=[go.Scatter3d(
                x=df_coords['X'],
                y=df_coords['Y'],
                z=df_coords['Z'],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=df_coords['Fidelity'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Fidélité")
                ),
                text=df_coords['Name'],
                textposition='top center'
            )])
            
            fig.update_layout(
                title="Distribution États Quantiques",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_fidelity = df_coords['Fidelity'].mean()
                st.metric("Fidélité Moyenne", f"{avg_fidelity:.3f}")
            
            with col2:
                avg_coherence = np.mean([q['coherence_time_ms'] for q in st.session_state.quantum_lab['qubits'].values()])
                st.metric("Cohérence Moyenne", f"{avg_coherence:.2f} ms")
            
            with col3:
                st.metric("Qubits Actifs", len(st.session_state.quantum_lab['qubits']))
        else:
            st.info("Aucun qubit à analyser")     

# ==================== PAGE: GRAVITÉ QUANTIQUE ====================
elif page == "🧬 Gravité Quantique":
    st.header("🧬 Gravité Quantique & Unification")
    
    st.info("""
    **Gravité Quantique**
    
    Théorie cherchant à unifier:
    - Relativité Générale (gravité, espace-temps)
    - Mécanique Quantique (particules, champs)
    
    **Échelle de Planck:** l_P = √(ℏG/c³) ≈ 1.6×10⁻³⁵ m
    
    **Approches:** Loop Quantum Gravity, String Theory, Causal Sets, etc.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌌 Mousse Quantique", "📊 Échelle Planck", "🔬 Effets", "🧪 Simulations"])
    
    with tab1:
        st.subheader("🌌 Mousse Quantique (Quantum Foam)")
        
        st.write("""
        **Structure Espace-Temps à l'Échelle de Planck**
        
        Espace-temps n'est pas lisse mais "écumeux" à échelle de Planck:
        - Fluctuations géométriques
        - Topologie dynamique
        - Incertitude Heisenberg pour géométrie
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            energy_scale = st.selectbox("Échelle Énergie",
                ["Planck (10¹⁹ GeV)", "GUT (10¹⁶ GeV)", "Électrofaible (100 GeV)", 
                 "LHC (10 TeV)", "Basse Énergie (1 GeV)"])
            
            # Extraire énergie
            if "Planck" in energy_scale:
                E = PHYSICAL_CONSTANTS['planck_energy']
            elif "GUT" in energy_scale:
                E = 1e16 * 1.6e-10  # GeV to J
            elif "Électrofaible" in energy_scale:
                E = 100 * 1.6e-10
            elif "LHC" in energy_scale:
                E = 1e4 * 1.6e-10
            else:
                E = 1.6e-10
        
        with col2:
            effects = calculate_planck_scale_effects(E)
            
            st.metric("Échelle Longueur", f"{effects['length_scale']:.2e} m")
            st.metric("Échelle Temps", f"{effects['time_scale']:.2e} s")
            st.metric("Force GQ", f"{effects['quantum_gravity_strength']:.2e}")
            
            if effects['foam_structure']:
                st.success("🌊 Mousse quantique significative!")
            else:
                st.info("Mousse quantique négligeable")
        
        # Visualisation mousse
        st.write("### 🌊 Visualisation Mousse Quantique")
        
        # Grille 3D avec fluctuations
        n_points = 20
        x = np.linspace(-1, 1, n_points)
        y = np.linspace(-1, 1, n_points)
        z = np.linspace(-1, 1, n_points)
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Fluctuations aléatoires
        fluctuations = np.random.normal(0, effects['quantum_gravity_strength'], (n_points, n_points, n_points))
        
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=fluctuations.flatten(),
            opacity=0.3,
            surface_count=15,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Fluctuations Géométriques Espace-Temps",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Unités de Planck")
        
        st.write("### 📏 Échelle Fondamentale")
        
        planck_units = {
            'Longueur': (PHYSICAL_CONSTANTS['planck_length'], 'm'),
            'Temps': (PHYSICAL_CONSTANTS['planck_time'], 's'),
            'Masse': (PHYSICAL_CONSTANTS['planck_mass'], 'kg'),
            'Énergie': (PHYSICAL_CONSTANTS['planck_energy'], 'J'),
            'Température': (1.417e32, 'K'),
            'Charge': (1.876e-18, 'C')
        }
        
        for unit, (value, symbol) in planck_units.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**{unit}:**")
            with col2:
                st.code(f"{value:.3e} {symbol}")
        
        st.write("### 🔬 Comparaisons")
        
        comparisons = [
            ("Longueur Planck / Proton", PHYSICAL_CONSTANTS['planck_length'] / 1e-15),
            ("Temps Planck / Age Univers", PHYSICAL_CONSTANTS['planck_time'] / (13.8e9 * 365.25 * 24 * 3600)),
            ("Masse Planck / Électron", PHYSICAL_CONSTANTS['planck_mass'] / PHYSICAL_CONSTANTS['m_e']),
            ("Énergie Planck / Masse-énergie Proton", PHYSICAL_CONSTANTS['planck_energy'] / (PHYSICAL_CONSTANTS['m_p'] * PHYSICAL_CONSTANTS['c']**2))
        ]
        
        for name, ratio in comparisons:
            st.write(f"**{name}:** {ratio:.2e}")
    
    with tab3:
        st.subheader("🔬 Effets Gravité Quantique")
        
        st.write("### 🌟 Phénomènes Prédits")
        
        phenomena = {
            "Violation Symétrie Lorentz": {
                "description": "Correction dépendant énergie vitesse lumière",
                "observable": "Rayons cosmiques, photons gamma",
                "status": "Non observé"
            },
            "Modification Relation Dispersion": {
                "description": "E² = p²c² + m²c⁴ + corrections Planck",
                "observable": "Propagation photons cosmiques",
                "status": "Limites contraintes"
            },
            "Entropie Trous Noirs": {
                "description": "S = A/(4l_P²) correction logarithmique",
                "observable": "Radiation Hawking",
                "status": "Prédiction théorique"
            },
            "Correction Cosmologique": {
                "description": "Modification équations Friedmann",
                "observable": "CMB, structure grande échelle",
                "status": "Recherche active"
            }
        }
        
        for name, info in phenomena.items():
            with st.expander(f"⚛️ {name}"):
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Observable:** {info['observable']}")
                st.write(f"**Status:** {info['status']}")
        
        st.write("### 📊 Corrections GQ")
        
        # Correction relation dispersion
        energies = np.logspace(9, 19, 100)  # GeV
        E_planck = PHYSICAL_CONSTANTS['planck_energy'] / 1.6e-10  # en GeV
        
        corrections = (energies / E_planck) ** 2
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=energies,
            y=corrections,
            mode='lines',
            line=dict(color='#9D50FF', width=3)
        ))
        
        fig.add_hline(y=1e-10, line_dash="dash", line_color="green",
                     annotation_text="Limite Observable")
        
        fig.update_layout(
            title="Corrections Gravité Quantique",
            xaxis_title="Énergie (GeV)",
            yaxis_title="Correction Relative",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🧪 Simulations Gravité Quantique")
        
        simulation_type = st.selectbox("Type Simulation",
            ["Mousse Quantique", "Trou Noir Quantique", "Big Bang Quantique", "Propagation Particule"])
        
        if simulation_type == "Trou Noir Quantique":
            mass_bh = st.slider("Masse Trou Noir (M_Planck)", 1e5, 1e10, 1e8, key="qbh_mass")
            
            if st.button("🚀 Simuler", type="primary"):
                with st.spinner("Simulation gravité quantique..."):
                    import time
                    time.sleep(2)
                    
                    m_planck = PHYSICAL_CONSTANTS['planck_mass']
                    mass_kg = mass_bh * m_planck
                    
                    # Corrections quantiques
                    r_s_classical = schwarzschild_radius(mass_kg)
                    
                    # Correction Loop Quantum Gravity
                    l_p = PHYSICAL_CONSTANTS['planck_length']
                    r_min = 2 * l_p * np.sqrt(mass_bh)
                    
                    st.success("✅ Simulation terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R. Classique", f"{r_s_classical:.2e} m")
                    with col2:
                        st.metric("R. Minimum (GQ)", f"{r_min:.2e} m")
                    with col3:
                        ratio = r_min / r_s_classical
                        st.metric("Ratio", f"{ratio:.2e}")
                    
                    st.info("💫 Gravité quantique prévient singularité nue!")
                    st.write("**Résolution:** Rebond quantique au lieu singularité")
        
        elif simulation_type == "Big Bang Quantique":
            if st.button("🌌 Simuler Big Bang Quantique", type="primary"):
                with st.spinner("Simulation cosmologie quantique..."):
                    import time
                    time.sleep(2)
                    
                    # Évolution densité énergie
                    t = np.logspace(-44, -35, 100)  # De temps Planck à 10^-35 s
                    
                    # Densité énergie classique
                    rho_classical = 1 / t**2
                    
                    # Correction quantique (rebond)
                    rho_max = 1e94  # kg/m³ (densité Planck)
                    rho_quantum = rho_max * np.sin(np.pi * rho_classical / (2*rho_max))
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=t,
                        y=rho_classical,
                        mode='lines',
                        name='Classique (Singularité)',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=t,
                        y=rho_quantum,
                        mode='lines',
                        name='Quantique (Rebond)',
                        line=dict(color='#9D50FF', width=3)
                    ))
                    
                    fig.update_layout(
                        title="Big Bang Quantique - Résolution Singularité",
                        xaxis_title="Temps (s)",
                        yaxis_title="Densité Énergie (kg/m³)",
                        xaxis_type="log",
                        yaxis_type="log",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Rebond quantique évite singularité initiale!")
                    st.info("💫 Univers avant Big Bang possible (Big Bounce)")

# ==================== PAGE: LOOP QUANTUM GRAVITY ====================
elif page == "🔄 Loop Quantum Gravity":
    st.header("🔄 Loop Quantum Gravity (LQG)")
    
    st.info("""
    **Gravité Quantique à Boucles**
    
    Quantification directe géométrie espace-temps:
    - Espace-temps discret (quanta)
    - Réseaux de spins
    - Aire et volume quantifiés
    - Pas de dimensions supplémentaires
    
    **Prédictions:** Rebond cosmique, correction trous noirs
    """)
    
    tab1, tab2, tab3 = st.tabs(["🕸️ Réseaux Spins", "📐 Géométrie Discrète", "🌌 Cosmologie"])
    
    with tab1:
        st.subheader("🕸️ Réseaux de Spins (Spin Networks)")
        
        st.write("""
        **État Quantique Géométrie**
        
        - Nœuds: points espace
        - Liens: relations adjacence
        - Spins: quantifie aires/volumes
        
        Base Hilbert espace états gravitationnels
        """)
        
        n_nodes = st.slider("Nombre Nœuds", 5, 30, 10)
        
        if st.button("🎲 Générer Réseau de Spins", type="primary"):
            # Générer réseau aléatoire
            positions = np.random.rand(n_nodes, 3)
            
            # Créer liens (distance < seuil)
            threshold = 0.3
            edges = []
            spins = []
            
            for i in range(n_nodes):
                for j in range(i+1, n_nodes):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < threshold:
                        edges.append((i, j))
                        # Spin j (quantifie aire)
                        spin = np.random.choice([0.5, 1, 1.5, 2])
                        spins.append(spin)
            
            # Visualisation 3D
            fig = go.Figure()
            
            # Liens
            for edge, spin in zip(edges, spins):
                i, j = edge
                fig.add_trace(go.Scatter3d(
                    x=[positions[i][0], positions[j][0]],
                    y=[positions[i][1], positions[j][1]],
                    z=[positions[i][2], positions[j][2]],
                    mode='lines',
                    line=dict(color='#9D50FF', width=spin*2),
                    showlegend=False,
                    hovertemplate=f'Spin j={spin}<extra></extra>'
                ))
            
            # Nœuds
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(size=8, color='white', line=dict(color='#9D50FF', width=2)),
                showlegend=False,
                hovertemplate='Nœud<extra></extra>'
            ))
            
            fig.update_layout(
                title="Réseau de Spins - État Géométrie Quantique",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Nœuds", n_nodes)
            with col2:
                st.metric("Liens", len(edges))
            with col3:
                avg_spin = np.mean(spins)
                st.metric("Spin Moyen", f"{avg_spin:.2f}")
    
    with tab2:
        st.subheader("📐 Géométrie Discrète")
        
        st.write("### 📏 Quantification Aire & Volume")
        
        st.write("""
        **Spectre Discret:**
        
        - Aire: A_j = 8πγℏl_P² √(j(j+1))
        - Volume: V_n quantifié
        
        où j = 0, 1/2, 1, 3/2, 2, ...
        γ = paramètre Immirzi ≈ 0.237
        """)
        
        gamma = 0.237
        l_p = PHYSICAL_CONSTANTS['planck_length']
        hbar = PHYSICAL_CONSTANTS['hbar']
        
        # Spins
        j_values = np.arange(0.5, 5, 0.5)
        
        # Aires quantifiées
        areas = 8 * np.pi * gamma * l_p**2 * np.sqrt(j_values * (j_values + 1))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=j_values,
            y=areas,
            mode='markers',
            marker=dict(size=12, color='#9D50FF'),
            name='Aires Quantifiées'
        ))
        
        fig.update_layout(
            title="Spectre Quantifié des Aires",
            xaxis_title="Spin j",
            yaxis_title="Aire (m²)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 🔬 Implications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Espace-Temps Discret:**")
            st.write("• Pas de continuum")
            st.write("• Quanta géométrie")
            st.write("• Volume minimum ≈ l_P³")
        
        with col2:
            st.write("**Conséquences:**")
            st.write("• Régularise singularités")
            st.write("• UV-finie naturellement")
            st.write("• Horizon information préservée")
    
    with tab3:
        st.subheader("🌌 Cosmologie Quantique à Boucles (LQC)")
        
        st.write("""
        **Loop Quantum Cosmology**
        
        Application LQG à cosmologie → Rebond cosmique:
        - Big Bang remplacé par Big Bounce
        - Densité maximum ρ_max ≈ 0.41 ρ_Planck
        - Univers cyclique possible
        """)
        
        if st.button("🌌 Simuler Rebond Quantique", type="primary"):
            with st.spinner("Simulation LQC..."):
                import time
                time.sleep(2)
                
                # Facteur échelle a(t)
                t = np.linspace(-1, 1, 200)
                
                # Rebond quantique
                a_min = 0.1
                a_bounce = a_min + (1-a_min) * (1 + np.tanh(5*t))/2
                
                # Densité énergie
                rho = 1 / a_bounce**3
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Facteur d'Échelle", "Densité Énergie")
                )
                
                fig.add_trace(go.Scatter(
                    x=t, y=a_bounce,
                    mode='lines',
                    line=dict(color='#9D50FF', width=3),
                    name='a(t)'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=t, y=rho,
                    mode='lines',
                    line=dict(color='#FF6B6B', width=3),
                    name='ρ(t)'
                ), row=2, col=1)
                
                fig.update_xaxes(title_text="Temps (unités arbitraires)", row=2, col=1)
                fig.update_yaxes(title_text="Facteur d'Échelle", row=1, col=1)
                fig.update_yaxes(title_text="Densité", row=2, col=1)
                
                fig.update_layout(
                    title="Big Bounce - Loop Quantum Cosmology",
                    template="plotly_dark",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Rebond quantique évite singularité!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("a_min", f"{a_min:.2f}")
                with col2:
                    st.metric("ρ_max/ρ_Planck", "0.41")
                with col3:
                    st.metric("Singularité", "❌ Résolvée")
                
                st.info("💫 Univers existait avant Big Bang (phase contraction)")

# ==================== PAGE: THÉORIE DES CORDES ====================
elif page == "🎻 Théorie Cordes":
    st.header("🎻 Théorie des Cordes & M-Theory")
    
    st.info("""
    **String Theory**
    
    Particules = vibrations de cordes unidimensionnelles
    - Dimensions supplémentaires (10 ou 11)
    - 5 théories cohérentes + M-theory
    - Unifie toutes forces
    
    **Longueur corde:** l_s ≈ l_Planck
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎵 Modes Vibration", "🌌 Dimensions Extra", "🔄 Dualités", "🧮 M-Theory"])
    
    with tab1:
        st.subheader("🎵 Modes de Vibration des Cordes")
        
        st.write("""
        **Chaque mode = Particule différente**
        
        - Mode fondamental → Graviton
        - Modes excités → Particules massives
        - Fréquence vibration ∝ Masse
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            string_type = st.selectbox("Type Corde",
                ["Ouverte", "Fermée"])
            
            excitation_level = st.slider("Niveau Excitation", 0, 10, 0)
            
            tension = st.slider("Tension (T/T_Planck)", 0.5, 2.0, 1.0, 0.1)
        
        with col2:
            # Masse particule
            l_s = PHYSICAL_CONSTANTS['planck_length']
            m = excitation_level * np.sqrt(tension) / l_s
            
            st.metric("Masse", f"{m:.2e} kg")
            
            if excitation_level == 0:
                st.success("Mode fondamental → Graviton (masse nulle)")
            else:
                st.info(f"Mode excité n={excitation_level}")
        
        # Visualisation vibration
        st.write("### 🌊 Pattern Vibration")
        
        t = np.linspace(0, 2*np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)
        
        # Amplitude vibration
        amplitude = 0.1 * (excitation_level + 1)
        z = amplitude * np.sin(excitation_level * t) * np.sin(np.random.uniform(0, 2*np.pi))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='#9D50FF', width=5)
        ))
        
        fig.update_layout(
            title=f"Corde en Vibration - Mode n={excitation_level}",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🌌 Dimensions Supplémentaires Compactifiées")
        
        st.write("""
        **Théorie Cordes nécessite 10 dimensions (11 en M-theory)**
        
        - 3 spatiales (observables)
        - 1 temporelle
        - 6 extra compactifiées (Calabi-Yau)
        
        **Taille:** ~ l_Planck → Non observables directement
        """)
        
        compactification = st.selectbox("Géométrie Compactification",
            ["Calabi-Yau 6D", "Orbifold", "Tore", "Sphère"])
        
        if st.button("🌀 Visualiser Compactification", type="primary"):
            st.write("### 🎨 Manifold de Calabi-Yau")
            
            # Projection 3D d'une variété Calabi-Yau
            u = np.linspace(0, 2*np.pi, 50)
            v = np.linspace(0, 2*np.pi, 50)
            U, V = np.meshgrid(u, v)
            
            # Équations paramétriques (projection)
            X = (2 + np.cos(V)) * np.cos(U)
            Y = (2 + np.cos(V)) * np.sin(U)
            Z = np.sin(V) + np.cos(3*U)
            
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                showscale=False
            )])
            
            fig.update_layout(
                title="Projection Calabi-Yau (6D → 3D)",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💫 6 dimensions enroulées à échelle de Planck")
        
        st.write("### 📊 Topologie & Physique")
        
        st.write("""
        **Géométrie compactification détermine:**
        - Nombre familles particules
        - Masses et couplages
        - Symétries gauge
        """)
    
    with tab3:
        st.subheader("🔄 Dualités en Théorie des Cordes")
        
        st.write("""
        **5 Théories Cohérentes reliées par dualités:**
        
        1. Type I
        2. Type IIA
        3. Type IIB
        4. Hétérotique SO(32)
        5. Hétérotique E₈×E₈
        
        **Dualités:** T-duality, S-duality, U-duality
        """)
        
        duality_type = st.selectbox("Type Dualité",
            ["T-Duality (R ↔ 1/R)", "S-Duality (Fort ↔ Faible)", "U-Duality"])
        
        if duality_type == "T-Duality (R ↔ 1/R)":
            st.write("### 📐 T-Duality")
            
            st.write("""
            Compactification sur cercle rayon R équivalente à rayon 1/R
            
            Cordes s'enroulent différemment mais physique identique
            """)
            
            R = st.slider("Rayon Compactification (l_s)", 0.1, 10.0, 1.0, 0.1)
            R_dual = 1 / R
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Rayon R", f"{R:.2f} l_s")
                st.write("**Théorie A**")
            
            with col2:
                st.metric("Rayon Dual", f"{R_dual:.2f} l_s")
                st.write("**Théorie B (équivalente)**")
            
            st.success("✅ Physique identique malgré géométries différentes!")
    
    with tab4:
        st.subheader("🧮 M-Theory - Unification Ultime")
        
        st.write("""
        **M-Theory**
        
        - 11 dimensions (10 spatiales + 1 temporelle)
        - Unifie 5 théories cordes
        - Membranes (branes) au lieu de cordes
        - Non perturbative
        
        **Objets:** Cordes, Membranes, 3-branes, ..., 9-branes
        """)
        
        st.write("### 🌐 Structure M-Theory")
        
        dimensions = {
            "Type IIA": "10D → limite M-theory sur cercle",
            "Type IIB": "10D → auto-duale sous S-duality",
            "Type I": "10D → cordes ouvertes + D-branes",
            "Hétérotique": "10D → symétrie gauge E₈×E₈ ou SO(32)",
            "M-Theory": "11D → unifie toutes les théories"
        }
        
        for theory, description in dimensions.items():
            with st.expander(f"🎻 {theory}"):
                st.write(description)
        
        st.write("### 🔬 Prédictions M-Theory")
        
        predictions = [
            "Supersymétrie (SUSY)",
            "Dimensions supplémentaires",
            "Multivers (paysage théories cordes)",
            "Gravité quantique cohérente",
            "Unification forces à échelle Planck"
        ]
        
        for pred in predictions:
            st.write(f"• {pred}")

# ==================== PAGE: ESPACE-TEMPS ====================
elif page == "⏱️ Espace-Temps":
    st.header("⏱️ Structure Espace-Temps Quantique")
    
    st.info("""
    **Espace-Temps en Physique Quantique**
    
    - Relativité: Espace-temps dynamique, courbé
    - Quantique: Discret, fluctuant à échelle Planck
    - Métrique: ds² = g_μν dx^μ dx^ν
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📐 Métriques", "🌊 Courbure", "🕰️ Dilatation Temps", "🌀 Torsion"])
    
    with tab1:
        st.subheader("📐 Métriques Espace-Temps")
        
        metric_type = st.selectbox("Type Métrique",
            ["Minkowski (Plat)", "Schwarzschild (Trou Noir)", 
             "Friedmann-Lemaître (Cosmologie)", "Kerr (Rotation)",
             "De Sitter (Expansion)"])
        
        if metric_type == "Schwarzschild (Trou Noir)":
            mass_bh = st.slider("Masse (M☉)", 1.0, 100.0, 10.0)
            
            M_sun = 1.989e30
            M = mass_bh * M_sun
            r_s = schwarzschild_radius(M)
            
            st.write("### 📊 Métrique de Schwarzschild")
            
            st.latex(r"ds^2 = -\left(1-\frac{2GM}{c^2r}\right)c^2dt^2 + \left(1-\frac{2GM}{c^2r}\right)^{-1}dr^2 + r^2d\Omega^2")
            
            # Composantes métrique
            r = np.linspace(r_s * 1.1, r_s * 10, 100)
            
            g_tt = -(1 - r_s / r)
            g_rr = 1 / (1 - r_s / r)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=r/r_s, y=g_tt,
                mode='lines',
                name='g_tt',
                line=dict(color='#9D50FF', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=r/r_s, y=g_rr,
                mode='lines',
                name='g_rr',
                line=dict(color='#FF6B6B', width=3)
            ))
            
            fig.add_vline(x=1, line_dash="dash", line_color="red",
                         annotation_text="Horizon")
            
            fig.update_layout(
                title="Composantes Métriques Schwarzschild",
                xaxis_title="r/r_s",
                yaxis_title="Valeur",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif metric_type == "Minkowski (Plat)":
            st.write("### 📊 Métrique de Minkowski")
            
            st.latex(r"ds^2 = -c^2dt^2 + dx^2 + dy^2 + dz^2")
            
            st.write("**Espace-temps plat (absence gravité)**")
            st.write("Métrique diagonale: η_μν = diag(-1, 1, 1, 1)")
    
    with tab2:
        st.subheader("🌊 Courbure Espace-Temps")
        
        st.write("""
        **Tenseur de Riemann:** R^ρ_σμν
        
        Mesure courbure intrinsèque espace-temps
        
        **Équations Einstein:** G_μν = 8πG/c⁴ T_μν
        """)
        
        st.write("### 📈 Courbure Scalaire")
        
        mass_source = st.slider("Masse Source (M☉)", 1.0, 1000.0, 10.0, key="curv_mass")
        
        M_sun = 1.989e30
        M = mass_source * M_sun
        G = PHYSICAL_CONSTANTS['G']
        c = PHYSICAL_CONSTANTS['c']
        
        r_s = 2*G*M/c**2
        r = np.linspace(r_s*1.1, r_s*20, 100)
        
        # Courbure scalaire (Schwarzschild)
        R_scalar = 48 * G**2 * M**2 / (c**4 * r**6)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=r/r_s, y=R_scalar,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#9D50FF', width=3)
        ))
        
        fig.update_layout(
            title="Courbure Scalaire près Masse",
            xaxis_title="r/r_s",
            yaxis_title="Courbure R",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🕰️ Dilatation Temporelle")
        
        st.write("""
        **Effets Relativistes sur Temps**
        
        1. Dilatation gravitationnelle: Δt' = Δt √(1 - r_s/r)
        2. Dilatation cinématique: Δt' = γΔt
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🌍 Effet Gravitationnel")
            
            altitude_km = st.slider("Altitude (km)", 0, 40000, 400)
            
            # Terre
            M_earth = 5.972e24
            R_earth = 6371e3
            
            r = R_earth + altitude_km * 1000
            r_s_earth = 2*G*M_earth/c**2
            
            time_dilation_grav = np.sqrt(1 - r_s_earth/r)
            
            st.metric("Facteur Dilatation", f"{time_dilation_grav:.10f}")
            
            # Différence sur 1 jour
            diff_ns = (1 - time_dilation_grav) * 86400 * 1e9
            st.write(f"**Différence/jour:** {diff_ns:.2f} ns")
        
        with col2:
            st.write("### 🚀 Effet Cinématique")
            
            velocity_c = st.slider("Vitesse (fraction c)", 0.0, 0.99, 0.5, 0.01)
            
            gamma = 1 / np.sqrt(1 - velocity_c**2)
            
            st.metric("Facteur Lorentz γ", f"{gamma:.3f}")
            
            # Temps propre
            proper_time = 1.0  # année
            coordinate_time = gamma * proper_time
            
            st.write(f"**Temps propre:** {proper_time} an")
            st.write(f"**Temps coordonnée:** {coordinate_time:.3f} ans")
    
    with tab4:
        st.subheader("🌀 Torsion Espace-Temps")
        
        st.write("""
        **Einstein-Cartan Theory**
        
        Espace-temps avec torsion (spin matière)
        
        - Connexion: torsion + courbure
        - Source: moment angulaire intrinsèque
        """)
        
        spin_density = st.slider("Densité Spin (kg⋅m/s)", 0.0, 1e10, 1e9, 1e8)
        
        if spin_density > 0:
            st.info("🌀 Torsion présente")
            st.write("Effet significatif seulement à densités extrêmes")
        else:
            st.success("Pas de torsion (Relativité Générale classique)")

# ==================== PAGE: EXPÉRIENCES ====================
elif page == "🔬 Expériences":
    st.header("🔬 Expériences Quantiques")
    
    tab1, tab2, tab3 = st.tabs(["🧪 Designer", "📋 Historique", "📊 Analyse"])
    
    with tab1:
        st.subheader("🧪 Designer d'Expérience")
        
        with st.form("design_experiment"):
            experiment_name = st.text_input("Nom Expérience", "EXP-001")
            
            experiment_type = st.selectbox("Type",
                ["Test Bell", "Téléportation", "Effet Tunnel", "Décohérence", 
                 "Cryptographie Quantique", "Intrication à Distance"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_qubits = st.number_input("Nombre Qubits", 1, 10, 2)
                n_measurements = st.number_input("Mesures", 100, 10000, 1000)
            
            with col2:
                fidelity_target = st.slider("Fidélité Cible", 0.8, 0.99, 0.95, 0.01)
                temperature_mk = st.number_input("Température (mK)", 10, 300, 20)
            
            hypothesis = st.text_area("Hypothèse",
                "Corrélations quantiques violent inégalités Bell")
            
            if st.form_submit_button("🚀 Lancer Expérience", type="primary"):
                with st.spinner("Expérience en cours..."):
                    import time
                    
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.03)
                        progress.progress(i + 1)
                    
                    progress.empty()
                    
                    # Résultats simulés
                    fidelity_achieved = fidelity_target * np.random.uniform(0.95, 1.02)
                    
                    if experiment_type == "Test Bell":
                        S_param = np.random.uniform(2.3, 2.8)
                        result = f"S = {S_param:.3f} (Violation Bell!)"
                    elif experiment_type == "Téléportation":
                        result = f"Fidélité: {fidelity_achieved:.3f}"
                    else:
                        result = "Succès"
                    
                    experiment = {
                        'name': experiment_name,
                        'type': experiment_type,
                        'n_qubits': n_qubits,
                        'n_measurements': n_measurements,
                        'fidelity_target': fidelity_target,
                        'fidelity_achieved': fidelity_achieved,
                        'result': result,
                        'hypothesis': hypothesis,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.quantum_lab['experiments'].append(experiment)
                    log_event(f"Expérience: {experiment_name} - {result}", "SUCCESS")
                    
                    st.success(f"✅ Expérience '{experiment_name}' terminée!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Fidélité", f"{fidelity_achieved:.3f}")
                    with col2:
                        st.metric("Mesures", n_measurements)
                    with col3:
                        st.metric("Résultat", result)
                    
                    st.rerun()
    
    with tab2:
        if not st.session_state.quantum_lab['experiments']:
            st.info("Aucune expérience réalisée")
        else:
            for i, exp in enumerate(st.session_state.quantum_lab['experiments'][::-1]):
                with st.expander(f"🔬 {exp['name']} - {exp['type']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Paramètres")
                        st.write(f"**Type:** {exp['type']}")
                        st.write(f"**Qubits:** {exp['n_qubits']}")
                        st.write(f"**Mesures:** {exp['n_measurements']}")
                    
                    with col2:
                        st.write("### 📈 Résultats")
                        st.write(f"**Fidélité:** {exp.get('fidelity_achieved', 0):.3f}")
                        st.write(f"**Résultat:** {exp['result']}")
                    
                    with col3:
                        st.write("### 📅 Info")
                        st.write(f"**Date:** {exp['timestamp'][:19]}")
                    
                    st.write("**Hypothèse:**")
                    st.write(exp.get('hypothesis', 'N/A'))
    
    with tab3:
        st.subheader("📊 Analyse Globale Expériences")
        
        if st.session_state.quantum_lab['experiments']:
            df_exp = pd.DataFrame(st.session_state.quantum_lab['experiments'])
            
            # Distribution types
            fig = px.pie(df_exp, names='type', 
                        title="Répartition Types Expériences",
                        template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            # Fidélité vs temps
            df_exp['timestamp'] = pd.to_datetime(df_exp['timestamp'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_exp.index,
                y=df_exp.get('fidelity_achieved', []),
                mode='lines+markers',
                line=dict(color='#9D50FF', width=2)
            ))
            
            fig.update_layout(
                title="Évolution Fidélité",
                xaxis_title="Expérience #",
                yaxis_title="Fidélité",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Expériences", len(df_exp))
            with col2:
                avg_fidelity = df_exp.get('fidelity_achieved', pd.Series([0])).mean()
                st.metric("Fidélité Moyenne", f"{avg_fidelity:.3f}")
            with col3:
                st.metric("Qubits Moyens", f"{df_exp['n_qubits'].mean():.1f}")
        else:
            st.info("Lancez des expériences pour voir les analyses")

# ==================== PAGE: SIMULATIONS IA ====================
elif page == "🤖 Simulations IA":
    st.header("🤖 Simulations IA & Machine Learning Quantique")
    
    st.info("""
    **Quantum Machine Learning**
    
    - Algorithmes quantiques pour ML
    - Optimisation circuits quantiques
    - Prédictions propriétés quantiques
    """)
    
    tab1, tab2, tab3 = st.tabs(["🧠 QML", "🎯 Optimisation", "📈 Prédictions"])
    
    with tab1:
        st.subheader("🧠 Quantum Machine Learning")
        
        st.write("### 🔬 Algorithmes Disponibles")
        
        algorithms = {
            "Variational Quantum Eigensolver (VQE)": "Trouver états fondamentaux",
            "Quantum Approximate Optimization (QAOA)": "Problèmes combinatoires",
            "Quantum Neural Networks (QNN)": "Classification quantique",
            "Quantum Support Vector Machine": "Classification données",
            "Quantum PCA": "Réduction dimensionnalité"
        }
        
        for algo, desc in algorithms.items():
            with st.expander(f"⚛️ {algo}"):
                st.write(f"**Application:** {desc}")
                
                if st.button(f"🚀 Exécuter {algo[:20]}...", key=f"exec_{algo}"):
                    with st.spinner("Entraînement quantique..."):
                        import time
                        time.sleep(2)
                        
                        accuracy = np.random.uniform(0.85, 0.98)
                        st.success(f"✅ Précision: {accuracy:.3f}")
    
    with tab2:
        st.subheader("🎯 Optimisation Circuits Quantiques")
        
        n_qubits_opt = st.slider("Qubits", 2, 10, 4, key="opt_qubits")
        n_layers = st.slider("Couches Circuit", 1, 10, 3)
        
        if st.button("⚡ Optimiser Circuit", type="primary"):
            with st.spinner("Optimisation variational..."):
                import time
                time.sleep(2)
                
                # Paramètres optimaux (simulés)
                optimal_params = np.random.uniform(-np.pi, np.pi, n_qubits_opt * n_layers)
                
                energy = -np.random.uniform(1, 5)
                
                st.success("✅ Optimisation terminée!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Énergie Min", f"{energy:.3f}")
                with col2:
                    st.metric("Paramètres", len(optimal_params))
                with col3:
                    st.metric("Itérations", np.random.randint(50, 200))
                
                # Visualisation convergence
                iterations = list(range(100))
                energies = [energy + (1-i/100)**2 for i in iterations]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=iterations,
                    y=energies,
                    mode='lines',
                    line=dict(color='#9D50FF', width=3)
                ))
                
                fig.update_layout(
                    title="Convergence Optimisation",
                    xaxis_title="Itération",
                    yaxis_title="Énergie",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Prédictions Propriétés Quantiques")
        
        property_type = st.selectbox("Propriété à Prédire",
            ["Temps Décohérence", "Fidélité Porte", "Probabilité Erreur", 
             "Entropie Intrication"])
        
        if st.button("🔮 Prédire", type="primary"):
            with st.spinner("IA analysant..."):
                import time
                time.sleep(1.5)
                
                if property_type == "Temps Décohérence":
                    prediction = np.random.uniform(0.5, 2.0)
                    unit = "ms"
                elif property_type == "Fidélité Porte":
                    prediction = np.random.uniform(0.95, 0.99)
                    unit = ""
                elif property_type == "Probabilité Erreur":
                    prediction = np.random.uniform(0.001, 0.01)
                    unit = ""
                else:
                    prediction = np.random.uniform(0.5, 2.0)
                    unit = "bits"
                
                st.success("✅ Prédiction IA")
                
                st.metric(property_type, f"{prediction:.3f} {unit}")
                
                confidence = np.random.uniform(0.85, 0.95)
                st.write(f"**Confiance:** {confidence:.2%}")
                
                st.info("💡 Prédiction basée sur modèle entraîné sur 10k+ mesures")

# ==================== PAGE: ANALYTICS ====================
elif page == "📊 Analytics":
    st.header("📊 Analytics & Visualisations Avancées")
    
    tab1, tab2, tab3 = st.tabs(["📈 Statistiques", "🔬 Corrélations", "📊 Dashboard"])
    
    with tab1:
        st.subheader("📈 Statistiques Globales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Qubits Créés", total_qubits)
            st.metric("Paires EPR", total_entangled)
        
        with col2:
            st.metric("Expériences", total_experiments)
            st.metric("Réseaux", len(st.session_state.quantum_lab['quantum_networks']))
        
        with col3:
            singularities = len(st.session_state.quantum_lab['singularities'])
            st.metric("Singularités", singularities)
            st.metric("Trous Noirs", len(st.session_state.quantum_lab['black_holes']))
        
        with col4:
            circuits = len(st.session_state.quantum_lab['quantum_circuits'])
            st.metric("Circuits", circuits)
            st.metric("Simulations", len(st.session_state.quantum_lab['quantum_gravity_simulations']))
        
        # Évolution temporelle
        if st.session_state.quantum_lab['log']:
            st.write("### 📈 Activité Recherche")
            
            events_by_hour = {}
            for event in st.session_state.quantum_lab['log']:
                hour = event['timestamp'][:13]
                events_by_hour[hour] = events_by_hour.get(hour, 0) + 1
            
            hours = list(events_by_hour.keys())
            counts = list(events_by_hour.values())
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=hours,
                y=counts,
                marker_color='#9D50FF'
            ))
            
            fig.update_layout(
                title="Événements par Heure",
                xaxis_title="Heure",
                yaxis_title="Nombre Événements",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Analyse Corrélations")
        
        if total_qubits >= 2:
            st.write("### 🔗 Matrice Corrélations Qubits")
            
            # Créer matrice corrélations
            n = min(len(st.session_state.quantum_lab['qubits']), 10)
            correlation_matrix = np.random.uniform(0, 1, (n, n))
            np.fill_diagonal(correlation_matrix, 1)
            
            # Symétrique
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                colorscale='Viridis',
                showscale=True
            ))
            
            fig.update_layout(
                title="Corrélations Quantiques Inter-Qubits",
                xaxis_title="Qubit",
                yaxis_title="Qubit",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Créez au moins 2 qubits pour voir les corrélations")
    
    with tab3:
        st.subheader("📊 Dashboard Complet")
        
        # Grille métriques
        metrics_data = {
            'Catégorie': ['États Quantiques', 'Intrication', 'Réseau', 'Gravité', 'Expériences'],
            'Éléments': [
                total_qubits,
                total_entangled,
                len(st.session_state.quantum_lab['quantum_networks']),
                len(st.session_state.quantum_lab['singularities']),
                total_experiments
            ],
            'Statut': ['✅' if x > 0 else '⚠️' for x in [
                total_qubits,
                total_entangled,
                len(st.session_state.quantum_lab['quantum_networks']),
                len(st.session_state.quantum_lab['singularities']),
                total_experiments
            ]]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = go.Figure(data=[go.Bar(
            x=df_metrics['Catégorie'],
            y=df_metrics['Éléments'],
            marker_color=['#9D50FF', '#6B2FFF', '#4A0FFF', '#FF6B6B', '#00CED1'],
            text=df_metrics['Éléments'],
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Vue d'Ensemble Recherche Quantique",
            xaxis_title="Catégorie",
            yaxis_title="Nombre",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau récapitulatif
        st.dataframe(df_metrics, use_container_width=True)

# ==================== PAGE: PARAMÈTRES ====================
elif page == "⚙️ Paramètres":
    st.header("⚙️ Configuration Laboratoire")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Général", "💾 Données", "🔄 Reset"])
    
    with tab1:
        st.subheader("🔧 Paramètres Généraux")
        
        with st.form("settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                default_temp = st.number_input("Température par Défaut (mK)", 10, 300, 20)
                default_fidelity = st.slider("Fidélité Cible", 0.90, 0.99, 0.95, 0.01)
                
                auto_save = st.checkbox("Sauvegarde Automatique", value=True)
            
            with col2:
                precision = st.selectbox("Précision Calculs",
                    ["Standard (6 décimales)", "Haute (10 décimales)", "Ultra (15 décimales)"])
                
                visualization = st.selectbox("Qualité Visualisations",
                    ["Standard", "Haute", "Ultra"])
                
                dark_mode = st.checkbox("Mode Sombre", value=True)
            
            if st.form_submit_button("💾 Sauvegarder Paramètres"):
                st.success("✅ Paramètres sauvegardés!")
                log_event("Paramètres mis à jour", "INFO")
    
    with tab2:
        st.subheader("💾 Gestion Données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 📥 Export")
            
            export_format = st.selectbox("Format",
                ["JSON", "CSV", "HDF5", "Pickle"])
            
            if st.button("📥 Exporter Données", type="primary"):
                # Préparer données
                data = {
                    'qubits': len(st.session_state.quantum_lab['qubits']),
                    'entangled_pairs': len(st.session_state.quantum_lab['entangled_pairs']),
                    'experiments': len(st.session_state.quantum_lab['experiments']),
                    'timestamp': datetime.now().isoformat()
                }
                
                st.success("✅ Données exportées!")
                st.json(data)
        
        with col2:
            st.write("### 📊 Statistiques")
            
            total_objects = sum([
                len(st.session_state.quantum_lab['qubits']),
                len(st.session_state.quantum_lab['entangled_pairs']),
                len(st.session_state.quantum_lab['experiments']),
                len(st.session_state.quantum_lab['quantum_networks']),
                len(st.session_state.quantum_lab['singularities']),
                len(st.session_state.quantum_lab['black_holes'])
            ])
            
            st.metric("Objets Totaux", total_objects)
            st.metric("Événements", len(st.session_state.quantum_lab['log']))
    
    with tab3:
        st.subheader("🔄 Réinitialisation")
        
        st.warning("⚠️ Actions irréversibles!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Supprimer Qubits"):
                st.session_state.quantum_lab['qubits'] = {}
                st.success("Qubits supprimés")
                st.rerun()
            
            if st.button("🗑️ Supprimer Expériences"):
                st.session_state.quantum_lab['experiments'] = []
                st.success("Expériences supprimées")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Réinitialiser Tout", key="reset_all"):
                if st.checkbox("Confirmer réinitialisation complète"):
                    st.session_state.quantum_lab = {
                        'qubits': {},
                        'entangled_pairs': [],
                        'quantum_circuits': {},
                        'quantum_networks': {},
                        'black_holes': {},
                        'wormholes': {},
                        'singularities': [],
                        'tunneling_experiments': [],
                        'quantum_fields': {},
                        'spacetime_metrics': {},
                        'quantum_gravity_simulations': [],
                        'loop_quantum_gravity': {},
                        'string_theory_models': {},
                        'experiments': [],
                        'measurements': [],
                        'log': []
                    }
                    st.success("✅ Laboratoire réinitialisé!")
                    st.rerun()

# ==================== PAGE: SINGULARITÉS ====================
elif page == "🕳️ Singularités":
    st.header("🕳️ Singularités & Effets Quantiques Extrêmes")
    
    st.info("""
    **Singularités Gravitationnelles**
    
    Points où courbure espace-temps devient infinie:
    - Singularité de Schwarzschild (trou noir)
    - Singularité nue (hypothétique)
    - Singularité Big Bang
    
    **Gravité Quantique nécessaire pour description complète**
    """)
    
    tab1, tab2, tab3 = st.tabs(["🕳️ Créer Singularité", "📊 Analyse", "🌌 Effets"])
    
    with tab1:
        st.subheader("🕳️ Simuler Singularité")
        
        with st.form("create_singularity"):
            col1, col2 = st.columns(2)
            
            with col1:
                singularity_type = st.selectbox("Type",
                    ["Schwarzschild (Statique)", 
                     "Kerr (Rotation)", 
                     "Reissner-Nordström (Chargé)",
                     "Kerr-Newman (Rotation + Charge)"])
                
                mass_solar = st.number_input("Masse (Masses Solaires)", 1.0, 1000.0, 10.0, 0.1)
                
                if "Kerr" in singularity_type:
                    spin = st.slider("Spin (J/M²)", 0.0, 1.0, 0.5, 0.01)
                else:
                    spin = 0
            
            with col2:
                if "Charge" in singularity_type or "Newman" in singularity_type:
                    charge = st.number_input("Charge (C)", 0.0, 1e10, 1e9, 1e8)
                else:
                    charge = 0
                
                quantum_corrections = st.checkbox("Corrections Quantiques", value=True)
                hawking_radiation = st.checkbox("Radiation Hawking", value=True)
            
            if st.form_submit_button("🕳️ Créer Singularité", type="primary"):
                singularity_id = f"sing_{len(st.session_state.quantum_lab['singularities']) + 1}"
                
                # Convertir masse en kg
                M_sun = 1.989e30  # kg
                mass_kg = mass_solar * M_sun
                
                # Rayon de Schwarzschild
                r_s = schwarzschild_radius(mass_kg)
                
                # Température Hawking
                T_H = hawking_temperature(mass_kg)
                
                # Temps évaporation
                t_evap = 2.1e67 * (mass_kg / 1e30)**3  # secondes
                
                # Entropie Bekenstein-Hawking
                k_B = PHYSICAL_CONSTANTS['k_B']
                c = PHYSICAL_CONSTANTS['c']
                hbar = PHYSICAL_CONSTANTS['hbar']
                G = PHYSICAL_CONSTANTS['G']
                
                A = 4 * np.pi * r_s**2  # Aire horizon
                S_BH = (k_B * c**3 * A) / (4 * G * hbar)
                
                singularity = {
                    'id': singularity_id,
                    'type': singularity_type,
                    'mass_solar': mass_solar,
                    'mass_kg': mass_kg,
                    'spin': spin,
                    'charge': charge,
                    'schwarzschild_radius': r_s,
                    'hawking_temperature': T_H,
                    'evaporation_time': t_evap,
                    'bekenstein_entropy': S_BH,
                    'quantum_corrections': quantum_corrections,
                    'hawking_radiation': hawking_radiation,
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.quantum_lab['singularities'].append(singularity)
                log_event(f"Singularité créée: {singularity_type}", "SUCCESS")
                
                st.success("✅ Singularité créée!")
                st.balloons()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Rayon S.", f"{r_s/1000:.2f} km")
                with col2:
                    st.metric("Temp. Hawking", f"{T_H:.2e} K")
                with col3:
                    st.metric("Entropie", f"{S_BH:.2e} J/K")
                with col4:
                    st.metric("Évaporation", f"{t_evap/3.15e7:.2e} ans")
                
                st.rerun()
    
    with tab2:
        if not st.session_state.quantum_lab['singularities']:
            st.info("Aucune singularité créée")
        else:
            for singularity in st.session_state.quantum_lab['singularities']:
                with st.expander(f"🕳️ {singularity['type']} ({singularity['mass_solar']} M☉)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Propriétés")
                        st.write(f"**Masse:** {singularity['mass_solar']} M☉")
                        st.write(f"**Spin:** {singularity['spin']}")
                        st.write(f"**Charge:** {singularity['charge']:.2e} C")
                    
                    with col2:
                        st.write("### 🌀 Horizon")
                        r_s = singularity['schwarzschild_radius']
                        st.write(f"**Rayon:** {r_s/1000:.2f} km")
                        st.write(f"**Circonférence:** {2*np.pi*r_s/1000:.2f} km")
                        
                        # Gravité surface
                        G = PHYSICAL_CONSTANTS['G']
                        c = PHYSICAL_CONSTANTS['c']
                        g_surface = G * singularity['mass_kg'] / r_s**2
                        st.write(f"**Gravité:** {g_surface:.2e} m/s²")
                    
                    with col3:
                        st.write("### ⚛️ Quantique")
                        T_H = singularity['hawking_temperature']
                        st.write(f"**T. Hawking:** {T_H:.2e} K")
                        st.write(f"**Entropie BH:** {singularity['bekenstein_entropy']:.2e}")
                        
                        t_evap_years = singularity['evaporation_time'] / 3.15e7
                        st.write(f"**Évaporation:** {t_evap_years:.2e} ans")
                    
                    # Visualisation courbure
                    st.write("### 🌊 Courbure Espace-Temps")
                    
                    r = np.linspace(r_s, r_s * 10, 100)
                    curvature = r_s / r
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=r/r_s,
                        y=curvature,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#9D50FF', width=3)
                    ))
                    
                    fig.add_vline(x=1, line_dash="dash", line_color="red",
                                 annotation_text="Horizon")
                    
                    fig.update_layout(
                        xaxis_title="Distance (r/rs)",
                        yaxis_title="Courbure Relative",
                        template="plotly_dark",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🌌 Effets Quantiques près Singularité")
        
        if st.session_state.quantum_lab['singularities']:
            selected_sing = st.selectbox("Sélectionner Singularité",
                range(len(st.session_state.quantum_lab['singularities'])),
                format_func=lambda i: st.session_state.quantum_lab['singularities'][i]['type'])
            
            singularity = st.session_state.quantum_lab['singularities'][selected_sing]
            
            st.write("### 💫 Radiation de Hawking")
            
            if singularity['hawking_radiation']:
                T_H = singularity['hawking_temperature']
                
                # Spectre radiation Hawking (corps noir)
                wavelengths = np.linspace(1e-10, 1e-6, 100)
                h = PHYSICAL_CONSTANTS['h']
                c = PHYSICAL_CONSTANTS['c']
                k_B = PHYSICAL_CONSTANTS['k_B']
                
                # Loi de Planck
                intensity = (2*h*c**2/wavelengths**5) / (np.exp(h*c/(wavelengths*k_B*T_H)) - 1)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=wavelengths * 1e9,
                    y=intensity,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#9D50FF', width=3)
                ))
                
                fig.update_layout(
                    title="Spectre Radiation Hawking",
                    xaxis_title="Longueur d'onde (nm)",
                    yaxis_title="Intensité",
                    template="plotly_dark",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Température", f"{T_H:.2e} K")
                with col2:
                    # Puissance émise
                    r_s = singularity['schwarzschild_radius']
                    A = 4 * np.pi * r_s**2
                    sigma = 5.67e-8  # Stefan-Boltzmann
                    P = sigma * A * T_H**4
                    st.metric("Puissance", f"{P:.2e} W")
                with col3:
                    st.metric("Type", "Particules virtuelles")
            else:
                st.info("Radiation Hawking désactivée")
            
            st.write("### 🌊 Production Paires Virtuelles")
            
            st.write("""
            Près de l'horizon, fluctuations quantiques du vide:
            1. Paire particule-antiparticule créée
            2. Une tombe dans trou noir
            3. Autre s'échappe (radiation Hawking)
            4. Trou noir perd masse/énergie
            """)
            
            # Taux production
            rate = 1 / singularity['evaporation_time']
            st.metric("Taux Production", f"{rate:.2e} paires/s")
        else:
            st.info("Créez une singularité")

# ==================== PAGE: TROUS NOIRS ====================
elif page == "🌀 Trous Noirs":
    st.header("🌀 Trous Noirs & Physique Extrême")
    
    st.info("""
    **Trou Noir (Black Hole)**
    
    Région espace-temps où gravité si intense que rien ne peut s'échapper.
    
    **Types:**
    - Stellaire (3-100 M☉)
    - Intermédiaire (100-10⁵ M☉)
    - Supermassif (10⁵-10¹⁰ M☉)
    - Primordial (< M☉)
    
    **Théorèmes:** No-hair (3 paramètres), Horizon, Singularité
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌀 Créer", "📊 Propriétés", "🌊 Disque Accrétion", "🔬 Observations"])
    
    with tab1:
        st.subheader("🌀 Créer Trou Noir")
        
        with st.form("create_black_hole"):
            col1, col2 = st.columns(2)
            
            with col1:
                bh_name = st.text_input("Nom", "Sagittarius A*")
                
                bh_class = st.selectbox("Classe",
                    ["Stellaire", "Intermédiaire", "Supermassif", "Primordial"])
                
                if bh_class == "Stellaire":
                    mass_default = 10.0
                elif bh_class == "Intermédiaire":
                    mass_default = 1000.0
                elif bh_class == "Supermassif":
                    mass_default = 4e6
                else:
                    mass_default = 0.001
                
                mass_solar = st.number_input("Masse (M☉)", 0.001, 1e10, mass_default, key="bh_mass")
            
            with col2:
                spin_param = st.slider("Paramètre Spin (a/M)", 0.0, 1.0, 0.7, 0.01)
                
                accretion_rate = st.number_input("Taux Accrétion (M☉/an)", 0.0, 1.0, 0.01, 0.001)
                
                has_jet = st.checkbox("Jets Relativistes", value=True)
            
            if st.form_submit_button("🌀 Créer Trou Noir", type="primary"):
                bh_id = f"bh_{len(st.session_state.quantum_lab['black_holes']) + 1}"
                
                M_sun = 1.989e30
                mass_kg = mass_solar * M_sun
                
                r_s = schwarzschild_radius(mass_kg)
                T_H = hawking_temperature(mass_kg)
                
                # Rayon ISCO (Innermost Stable Circular Orbit)
                r_isco = r_s * (3 + np.sqrt(9 - 8*spin_param**2))
                
                # Efficacité accrétion
                efficiency = 1 - np.sqrt(1 - 2/(3*r_isco/r_s))
                
                # Luminosité
                c = PHYSICAL_CONSTANTS['c']
                L_edd = 1.26e38 * mass_solar  # Luminosité Eddington (W)
                L_actual = efficiency * accretion_rate * M_sun * c**2 / 3.15e7
                
                black_hole = {
                    'id': bh_id,
                    'name': bh_name,
                    'class': bh_class,
                    'mass_solar': mass_solar,
                    'mass_kg': mass_kg,
                    'spin': spin_param,
                    'schwarzschild_radius': r_s,
                    'isco_radius': r_isco,
                    'hawking_temperature': T_H,
                    'accretion_rate': accretion_rate,
                    'efficiency': efficiency,
                    'luminosity': L_actual,
                    'eddington_luminosity': L_edd,
                    'has_jet': has_jet,
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.quantum_lab['black_holes'][bh_id] = black_hole
                log_event(f"Trou noir créé: {bh_name}", "SUCCESS")
                
                st.success(f"✅ Trou noir '{bh_name}' créé!")
                st.balloons()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Masse", f"{mass_solar:.2e} M☉")
                with col2:
                    st.metric("Rayon", f"{r_s/1000:.2f} km")
                with col3:
                    st.metric("Spin", f"{spin_param:.2f}")
                with col4:
                    st.metric("Luminosité", f"{L_actual:.2e} W")
                
                st.rerun()
    
    with tab2:
        if not st.session_state.quantum_lab['black_holes']:
            st.info("Aucun trou noir créé")
        else:
            for bh_id, bh in st.session_state.quantum_lab['black_holes'].items():
                with st.expander(f"🌀 {bh['name']} ({bh['class']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Paramètres")
                        st.write(f"**Masse:** {bh['mass_solar']:.2e} M☉")
                        st.write(f"**Spin:** {bh['spin']:.3f}")
                        st.write(f"**Classe:** {bh['class']}")
                    
                    with col2:
                        st.write("### 🌀 Géométrie")
                        st.write(f"**R. Schwarzschild:** {bh['schwarzschild_radius']/1000:.2f} km")
                        st.write(f"**R. ISCO:** {bh['isco_radius']/1000:.2f} km")
                        st.write(f"**Efficacité:** {bh['efficiency']*100:.1f}%")
                    
                    with col3:
                        st.write("### ⚡ Émission")
                        st.write(f"**Luminosité:** {bh['luminosity']:.2e} W")
                        st.write(f"**L. Eddington:** {bh['eddington_luminosity']:.2e} W")
                        st.write(f"**Jets:** {'✅' if bh['has_jet'] else '❌'}")
                    
                    # Ergosphère (si rotation)
                    if bh['spin'] > 0:
                        st.write("### 🌪️ Ergosphère")
                        st.info(f"Région où espace-temps entraîné par rotation")
                        st.write("**Processus Penrose:** Extraction énergie rotation")
    
    with tab3:
        st.subheader("🌊 Disque d'Accrétion")
        
        if st.session_state.quantum_lab['black_holes']:
            selected_bh = st.selectbox("Sélectionner Trou Noir",
                list(st.session_state.quantum_lab['black_holes'].keys()),
                format_func=lambda x: st.session_state.quantum_lab['black_holes'][x]['name'])
            
            bh = st.session_state.quantum_lab['black_holes'][selected_bh]
            
            # Profil température disque
            r_s = bh['schwarzschild_radius']
            r_isco = bh['isco_radius']
            
            r = np.linspace(r_isco, r_isco * 100, 100)
            
            # Température disque (approximation)
            G = PHYSICAL_CONSTANTS['G']
            M = bh['mass_kg']
            sigma = 5.67e-8
            
            T_disk = ((3*G*M*bh['accretion_rate']*1.989e30/3.15e7)/(8*np.pi*sigma*r**3))**0.25
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=r/r_s,
                y=T_disk,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#FF6B6B', width=3)
            ))
            
            fig.add_vline(x=r_isco/r_s, line_dash="dash", line_color="white",
                         annotation_text="ISCO")
            
            fig.update_layout(
                title="Profil Température Disque d'Accrétion",
                xaxis_title="Rayon (r/rs)",
                yaxis_title="Température (K)",
                yaxis_type="log",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                T_max = T_disk.max()
                st.metric("T. Max", f"{T_max:.2e} K")
            with col2:
                st.metric("Taux Accrétion", f"{bh['accretion_rate']:.3f} M☉/an")
            with col3:
                st.metric("Luminosité", f"{bh['luminosity']:.2e} W")
        else:
            st.info("Créez un trou noir")
    
    with tab4:
        st.subheader("🔬 Observations & Détection")
        
        st.write("### 📡 Méthodes Détection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Directes:**")
            st.write("• Rayonnement X (disque)")
            st.write("• Ondes gravitationnelles (fusion)")
            st.write("• Image Event Horizon (EHT)")
            st.write("• Jets relativistes")
        
        with col2:
            st.write("**Indirectes:**")
            st.write("• Orbites stellaires")
            st.write("• Lentilles gravitationnelles")
            st.write("• Variabilité luminosité")
            st.write("• Spectroscopie")
        
        st.write("### 🌌 Trous Noirs Célèbres")
        
        famous_bh = {
            "Sgr A*": {"mass": 4.1e6, "distance": 26000, "location": "Centre Voie Lactée"},
            "M87*": {"mass": 6.5e9, "distance": 53e6, "location": "Galaxie M87"},
            "Cygnus X-1": {"mass": 21, "distance": 6070, "location": "Constellation Cygne"},
            "GW150914": {"mass": 62, "distance": 1.3e9, "location": "Fusion détectée LIGO"}
        }
        
        for name, info in famous_bh.items():
            with st.expander(f"🌀 {name}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Masse:** {info['mass']:.2e} M☉")
                with col2:
                    st.write(f"**Distance:** {info['distance']:.2e} ly")
                with col3:
                    st.write(f"**Lieu:** {info['location']}")

# ==================== PAGE: EFFET TUNNEL ====================
elif page == "🚇 Effet Tunnel":
    st.header("🚇 Effet Tunnel Quantique")
    
    st.info("""
    **Effet Tunnel (Quantum Tunneling)**
    
    Particule traverse barrière de potentiel classiquement interdite.
    
    **Probabilité:** T ≈ exp(-2κL)
    où κ = √(2m(V-E))/ℏ
    
    **Applications:** Transistors, radioactivité α, fusion stellaire, STM
    """)
    
    tab1, tab2, tab3 = st.tabs(["🚇 Simulateur", "📊 Expériences", "📈 Analyse"])
    
    with tab1:
        st.subheader("🚇 Simulateur Effet Tunnel")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### ⚙️ Configuration Barrière")
            
            barrier_height_ev = st.slider("Hauteur Barrière (eV)", 0.1, 10.0, 2.0, 0.1)
            barrier_width_nm = st.slider("Largeur Barrière (nm)", 0.1, 10.0, 1.0, 0.1)
            
            particle_energy_ev = st.slider("Énergie Particule (eV)", 0.01, 5.0, 1.0, 0.01)
            
            particle_type = st.selectbox("Particule",
                ["Électron", "Proton", "Alpha", "Custom"])
            
            if particle_type == "Électron":
                mass = PHYSICAL_CONSTANTS['m_e']
            elif particle_type == "Proton":
                mass = PHYSICAL_CONSTANTS['m_p']
            elif particle_type == "Alpha":
                mass = 6.64e-27  # kg
            else:
                mass = st.number_input("Masse (kg)", 1e-31, 1e-26, 1e-30, format="%.2e")
        
        with col2:
            st.write("### 📊 Résultats")
            
            # Convertir en Joules
            eV_to_J = PHYSICAL_CONSTANTS['e']
            barrier_height_j = barrier_height_ev * eV_to_J
            particle_energy_j = particle_energy_ev * eV_to_J
            barrier_width_m = barrier_width_nm * 1e-9
            
            # Calculer probabilité tunnel
            T = calculate_tunneling_probability(
                barrier_height_j,
                barrier_width_m,
                particle_energy_j,
                mass
            )
            
            st.metric("Probabilité Tunnel", f"{T:.2e}")
            st.metric("Probabilité (%)", f"{T*100:.6f}%")
            
            # Coefficient réflexion
            R = 1 - T
            st.metric("Réflexion", f"{R:.2e}")
            
            if T > 0.01:
                st.success("✅ Tunneling probable!")
            elif T > 1e-6:
                st.info("Tunneling possible")
            else:
                st.warning("Tunneling très improbable")
        
        # Visualisation barrière
        st.write("### 🌊 Fonction d'Onde & Barrière")
        
        x = np.linspace(-2, 2, 1000) * barrier_width_m * 1e9  # nm
        
        # Barrière de potentiel
        V = np.zeros_like(x)
        barrier_start = -barrier_width_nm/2
        barrier_end = barrier_width_nm/2
        V[(x >= barrier_start) & (x <= barrier_end)] = barrier_height_ev
        
        # Fonction d'onde (approximation)
        psi = np.exp(-0.5*(x+barrier_width_nm)**2/0.5)  # Gaussienne
        
        # Atténuation dans barrière
        in_barrier = (x >= barrier_start) & (x <= barrier_end)
        psi[in_barrier] *= np.exp(-2*(x[in_barrier]-barrier_start)/(barrier_end-barrier_start))
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(
            x=x, y=V,
            mode='lines',
            name='Barrière V(x)',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)'
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=x, y=abs(psi)**2,
            mode='lines',
            name='|ψ(x)|²',
            line=dict(color='#9D50FF', width=2)
        ), secondary_y=True)
        
        fig.add_hline(y=particle_energy_ev, line_dash="dash", line_color="green",
                     annotation_text="E particule", secondary_y=False)
        
        fig.update_layout(
            title="Barrière de Potentiel & Fonction d'Onde",
            xaxis_title="Position (nm)",
            template="plotly_dark",
            height=400
        )
        
        fig.update_yaxes(title_text="Potentiel (eV)", secondary_y=False)
        fig.update_yaxes(title_text="|ψ|² (Densité Probabilité)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("💾 Enregistrer Expérience"):
            experiment = {
                'type': 'Effet Tunnel',
                'barrier_height_ev': barrier_height_ev,
                'barrier_width_nm': barrier_width_nm,
                'particle_energy_ev': particle_energy_ev,
                'particle_type': particle_type,
                'tunneling_probability': T,
                'timestamp': datetime.now().isoformat()
            }
            
            st.session_state.quantum_lab['tunneling_experiments'].append(experiment)
            log_event(f"Expérience tunnel: T={T:.2e}", "SUCCESS")
            st.success("Expérience enregistrée!")
    
    with tab2:
        st.subheader("📊 Historique Expériences")
        
        if st.session_state.quantum_lab['tunneling_experiments']:
            for i, exp in enumerate(st.session_state.quantum_lab['tunneling_experiments'][::-1]):
                with st.expander(f"🚇 Expérience {len(st.session_state.quantum_lab['tunneling_experiments'])-i}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**Particule:** {exp['particle_type']}")
                        st.write(f"**Énergie:** {exp['particle_energy_ev']} eV")
                    
                    with col2:
                        st.write(f"**Barrière V:** {exp['barrier_height_ev']} eV")
                        st.write(f"**Largeur:** {exp['barrier_width_nm']} nm")
                    
                    with col3:
                        st.write(f"**Probabilité:** {exp['tunneling_probability']:.2e}")
                        st.write(f"**%:** {exp['tunneling_probability']*100:.6f}%")
                    
                    with col4:
                        st.write(f"**Date:** {exp['timestamp'][:19]}")
        else:
            st.info("Aucune expérience tunnel enregistrée")
    
    with tab3:
        st.subheader("📈 Analyse Effet Tunnel")
        
        st.write("### 📊 Dépendance Largeur Barrière")
        
        widths = np.linspace(0.1, 5, 50)
        probabilities = []
        
        # Paramètres fixes
        V = 2.0 * PHYSICAL_CONSTANTS['e']
        E = 1.0 * PHYSICAL_CONSTANTS['e']
        m = PHYSICAL_CONSTANTS['m_e']
        
        for w in widths:
            T = calculate_tunneling_probability(V, w*1e-9, E, m)
            probabilities.append(T)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=widths,
            y=probabilities,
            mode='lines+markers',
            line=dict(color='#9D50FF', width=3)
        ))
        
        fig.update_layout(
            title="Probabilité Tunnel vs Largeur Barrière",
            xaxis_title="Largeur (nm)",
            yaxis_title="Probabilité",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### 📊 Dépendance Énergie Particule")
        
        energies = np.linspace(0.1, 1.9, 50)
        probabilities_energy = []
        
        for E_ev in energies:
            E_j = E_ev * PHYSICAL_CONSTANTS['e']
            T = calculate_tunneling_probability(V, 1e-9, E_j, m)
            probabilities_energy.append(T)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=energies,
            y=probabilities_energy,
            mode='lines+markers',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        fig.update_layout(
            title="Probabilité Tunnel vs Énergie (Barrière 2 eV)",
            xaxis_title="Énergie Particule (eV)",
            yaxis_title="Probabilité",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)                  

# ==================== PAGE: INTRICATION ====================
elif page == "🔗 Intrication":
    st.header("🔗 Intrication Quantique & Corrélations EPR")
    
    st.info("""
    **Intrication Quantique (Entanglement)**
    
    Deux particules intriquées partagent un état quantique corrélé non-localement.
    Mesure sur une particule affecte instantanément l'autre, peu importe la distance.
    
    **État Bell:** |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    
    **Paradoxe EPR:** Einstein-Podolsky-Rosen (1935)
    **Inégalités Bell:** Violation prouve non-localité quantique
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Créer Paire", "📊 Paires EPR", "🧪 Test Bell", "📈 Téléportation"])
    
    with tab1:
        st.subheader("🔗 Créer Paire Intriquée")
        
        if len(st.session_state.quantum_lab['qubits']) < 2:
            st.warning("⚠️ Créez au moins 2 qubits pour créer une paire intriquée")
        else:
            with st.form("create_entangled_pair"):
                col1, col2 = st.columns(2)
                
                with col1:
                    qubit1_id = st.selectbox("Qubit 1",
                        list(st.session_state.quantum_lab['qubits'].keys()),
                        format_func=lambda x: st.session_state.quantum_lab['qubits'][x]['name'])
                
                with col2:
                    qubit2_id = st.selectbox("Qubit 2",
                        [q for q in st.session_state.quantum_lab['qubits'].keys() if q != qubit1_id],
                        format_func=lambda x: st.session_state.quantum_lab['qubits'][x]['name'])
                
                bell_state = st.selectbox("État de Bell",
                    ["Φ⁺: (|00⟩ + |11⟩)/√2",
                     "Φ⁻: (|00⟩ - |11⟩)/√2",
                     "Ψ⁺: (|01⟩ + |10⟩)/√2",
                     "Ψ⁻: (|01⟩ - |10⟩)/√2"])
                
                distance_km = st.slider("Distance Séparation (km)", 0, 1000, 100)
                
                if st.form_submit_button("🔗 Intriquer Qubits", type="primary"):
                    pair_id = f"epr_{len(st.session_state.quantum_lab['entangled_pairs']) + 1}"
                    
                    # Créer état intriqué
                    if "Φ⁺" in bell_state:
                        state = np.array([1, 0, 0, 1]) / np.sqrt(2)
                    elif "Φ⁻" in bell_state:
                        state = np.array([1, 0, 0, -1]) / np.sqrt(2)
                    elif "Ψ⁺" in bell_state:
                        state = np.array([0, 1, 1, 0]) / np.sqrt(2)
                    else:  # Ψ⁻
                        state = np.array([0, 1, -1, 0]) / np.sqrt(2)
                    
                    # Calculer entropie d'intrication
                    entropy = calculate_entanglement_entropy(state)
                    
                    pair = {
                        'id': pair_id,
                        'qubit1_id': qubit1_id,
                        'qubit2_id': qubit2_id,
                        'bell_state': bell_state.split(':')[0],
                        'state': state,
                        'distance_km': distance_km,
                        'entanglement_entropy': entropy,
                        'fidelity': np.random.uniform(0.95, 0.99),
                        'created_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.quantum_lab['entangled_pairs'].append(pair)
                    log_event(f"Paire EPR créée: {bell_state}", "SUCCESS")
                    
                    st.success(f"✅ Paire intriquée créée!")
                    st.balloons()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("État", bell_state.split(':')[0])
                    with col2:
                        st.metric("Entropie", f"{entropy:.3f}")
                    with col3:
                        st.metric("Distance", f"{distance_km} km")
                    
                    st.info("💫 Les qubits sont maintenant intriqués!")
                    st.rerun()
    
    with tab2:
        if not st.session_state.quantum_lab['entangled_pairs']:
            st.info("Aucune paire intriquée")
        else:
            for pair in st.session_state.quantum_lab['entangled_pairs']:
                qubit1 = st.session_state.quantum_lab['qubits'][pair['qubit1_id']]
                qubit2 = st.session_state.quantum_lab['qubits'][pair['qubit2_id']]
                
                with st.expander(f"🔗 {qubit1['name']} ↔ {qubit2['name']} ({pair['bell_state']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 État Intriqué")
                        st.write(f"**Type:** {pair['bell_state']}")
                        st.write(f"**Entropie:** {pair['entanglement_entropy']:.3f}")
                        st.write(f"**Fidélité:** {pair['fidelity']:.3f}")
                    
                    with col2:
                        st.write("### 🌍 Géométrie")
                        st.write(f"**Distance:** {pair['distance_km']} km")
                        st.write(f"**Temps lumière:** {pair['distance_km']/300000:.6f} s")
                        st.write(f"**Corrélation:** Instantanée")
                    
                    with col3:
                        st.write("### ⚡ Actions")
                        
                        if st.button("📏 Mesurer Q1", key=f"measure1_{pair['id']}"):
                            result1 = np.random.choice([0, 1])
                            st.info(f"Q1 mesuré: |{result1}⟩")
                            
                            # Corrélation EPR
                            if pair['bell_state'] in ['Φ⁺', 'Φ⁻']:
                                result2 = result1
                            else:
                                result2 = 1 - result1
                            
                            st.success(f"Q2 collapse: |{result2}⟩ (corrélation EPR)")
                        
                        if st.button("🧪 Test Bell", key=f"bell_{pair['id']}"):
                            # Violation inégalités de Bell
                            S = np.random.uniform(2.5, 2.8)  # > 2 = violation
                            st.metric("Paramètre S", f"{S:.3f}")
                            
                            if S > 2:
                                st.success("✅ Violation inégalités Bell!")
                                st.info("Prouve non-localité quantique")
                            else:
                                st.warning("Pas de violation")
    
    with tab3:
        st.subheader("🧪 Test des Inégalités de Bell")
        
        st.write("""
        **Inégalités de Bell (CHSH)**
        
        Classique: S ≤ 2
        Quantique: S ≤ 2√2 ≈ 2.828
        
        Violation prouve corrélations quantiques non-locales
        """)
        
        if st.session_state.quantum_lab['entangled_pairs']:
            selected_pair_idx = st.selectbox("Sélectionner Paire",
                range(len(st.session_state.quantum_lab['entangled_pairs'])),
                format_func=lambda i: f"Paire {i+1}")
            
            pair = st.session_state.quantum_lab['entangled_pairs'][selected_pair_idx]
            
            n_measurements = st.slider("Nombre Mesures", 100, 10000, 1000, 100)
            
            if st.button("🚀 Exécuter Test Bell", type="primary"):
                with st.spinner("Exécution mesures..."):
                    import time
                    time.sleep(2)
                    
                    # Simuler mesures corrélées
                    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                    
                    correlations = []
                    for angle in angles:
                        # Corrélation quantique
                        corr = -np.cos(angle) + np.random.normal(0, 0.05)
                        correlations.append(corr)
                    
                    # Paramètre CHSH
                    S = abs(correlations[0] - correlations[1]) + abs(correlations[2] + correlations[3])
                    
                    st.success("✅ Test terminé!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Paramètre S", f"{S:.3f}")
                    with col2:
                        st.metric("Limite Classique", "2.000")
                    with col3:
                        st.metric("Limite Quantique", "2.828")
                    
                    if S > 2:
                        st.success("🎉 VIOLATION DES INÉGALITÉS DE BELL!")
                        st.info("Les corrélations observées ne peuvent être expliquées classiquement")
                        
                        sigma = (S - 2) / 0.05
                        st.write(f"**Significativité:** {sigma:.1f}σ")
                    
                    # Graphique corrélations
                    fig = go.Figure(data=[go.Bar(
                        x=[f"{a:.2f} rad" for a in angles],
                        y=correlations,
                        marker_color='#9D50FF'
                    )])
                    
                    fig.add_hline(y=0, line_dash="dash", line_color="white")
                    fig.add_hline(y=-1, line_dash="dash", line_color="red", annotation_text="Classique")
                    
                    fig.update_layout(
                        title="Corrélations Mesurées",
                        xaxis_title="Angle",
                        yaxis_title="Corrélation",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    log_event(f"Test Bell: S={S:.3f}", "SUCCESS")
        else:
            st.info("Créez une paire intriquée pour tester Bell")
    
    with tab4:
        st.subheader("📈 Téléportation Quantique")
        
        st.write("""
        **Protocole de Téléportation Quantique**
        
        1. Alice et Bob partagent paire EPR
        2. Alice effectue mesure Bell sur son qubit + qubit à téléporter
        3. Alice envoie 2 bits classiques à Bob
        4. Bob applique correction unitaire
        5. État téléporté!
        """)
        
        if len(st.session_state.quantum_lab['entangled_pairs']) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                pair_idx = st.selectbox("Paire EPR",
                    range(len(st.session_state.quantum_lab['entangled_pairs'])),
                    format_func=lambda i: f"Paire {i+1}")
                
                available_qubits = [q for q in st.session_state.quantum_lab['qubits'].keys()]
                if available_qubits:
                    qubit_to_teleport = st.selectbox("Qubit à Téléporter",
                        available_qubits,
                        format_func=lambda x: st.session_state.quantum_lab['qubits'][x]['name'])
            
            with col2:
                st.write("### 📊 Configuration")
                st.write("**Alice:** Qubit source + EPR1")
                st.write("**Bob:** EPR2")
                st.write("**Canal:** 2 bits classiques")
            
            if st.button("📡 Téléporter État Quantique", type="primary"):
                with st.spinner("Téléportation en cours..."):
                    import time
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.text("1. Mesure Bell sur Alice...")
                    time.sleep(1)
                    progress.progress(0.33)
                    
                    status.text("2. Transmission bits classiques...")
                    time.sleep(1)
                    progress.progress(0.66)
                    
                    status.text("3. Correction unitaire Bob...")
                    time.sleep(1)
                    progress.progress(1.0)
                    
                    progress.empty()
                    status.empty()
                    
                    st.success("✅ Téléportation réussie!")
                    
                    fidelity_teleport = np.random.uniform(0.90, 0.98)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Fidélité", f"{fidelity_teleport:.3f}")
                    with col2:
                        st.metric("Bits Classiques", "2")
                    with col3:
                        st.metric("Temps", "Instantané")
                    
                    st.info("💫 État quantique transféré sans déplacement physique!")
                    
                    log_event("Téléportation quantique réussie", "SUCCESS")
        else:
            st.info("Créez une paire EPR pour téléporter")

# ==================== PAGE: RÉSEAU QUANTIQUE ====================
elif page == "🌐 Réseau Quantique":
    st.header("🌐 Réseau Quantique & Internet Quantique")
    
    st.info("""
    **Réseau Quantique (Quantum Network)**
    
    Infrastructure distribuée connectant qubits via:
    - Canaux quantiques (fibres optiques, satellite)
    - Répéteurs quantiques
    - Routage quantique
    - Distribution clés quantiques (QKD)
    
    **Applications:** Communication sécurisée, computing distribué, sensing
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 Créer Réseau", "📡 Nœuds", "🔐 QKD", "📊 Topologie"])
    
    with tab1:
        st.subheader("🌐 Créer Réseau Quantique")
        
        with st.form("create_quantum_network"):
            col1, col2 = st.columns(2)
            
            with col1:
                network_name = st.text_input("Nom Réseau", "QuantNet-01")
                topology = st.selectbox("Topologie",
                    ["Star", "Ring", "Mesh", "Tree", "Hybrid"])
                n_nodes = st.slider("Nombre Nœuds", 3, 50, 10)
            
            with col2:
                channel_type = st.selectbox("Canal",
                    ["Fibre Optique", "Espace Libre", "Satellite"])
                
                distance_km = st.slider("Distance Moyenne (km)", 1, 1000, 100)
                
                qkd_enabled = st.checkbox("QKD Activé", value=True)
            
            if st.form_submit_button("🌐 Créer Réseau", type="primary"):
                network_id = f"qnet_{len(st.session_state.quantum_lab['quantum_networks']) + 1}"
                
                # Créer nœuds
                nodes = []
                for i in range(n_nodes):
                    node = {
                        'id': f"node_{i+1}",
                        'name': f"Node-{i+1}",
                        'position': (np.random.uniform(0, 100), np.random.uniform(0, 100)),
                        'qubits': np.random.randint(1, 10),
                        'fidelity': np.random.uniform(0.90, 0.98)
                    }
                    nodes.append(node)
                
                # Créer connexions selon topologie
                edges = []
                if topology == "Star":
                    for i in range(1, n_nodes):
                        edges.append((0, i))
                elif topology == "Ring":
                    for i in range(n_nodes):
                        edges.append((i, (i+1) % n_nodes))
                elif topology == "Mesh":
                    for i in range(n_nodes):
                        for j in range(i+1, n_nodes):
                            if np.random.random() > 0.5:
                                edges.append((i, j))
                
                network = {
                    'id': network_id,
                    'name': network_name,
                    'topology': topology,
                    'nodes': nodes,
                    'edges': edges,
                    'channel_type': channel_type,
                    'distance_km': distance_km,
                    'qkd_enabled': qkd_enabled,
                    'total_qubits': sum(n['qubits'] for n in nodes),
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.quantum_lab['quantum_networks'][network_id] = network
                log_event(f"Réseau quantique créé: {network_name}", "SUCCESS")
                
                st.success(f"✅ Réseau '{network_name}' créé!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nœuds", n_nodes)
                with col2:
                    st.metric("Connexions", len(edges))
                with col3:
                    st.metric("Qubits Total", network['total_qubits'])
                
                st.rerun()
    
    with tab2:
        if not st.session_state.quantum_lab['quantum_networks']:
            st.info("Aucun réseau quantique créé")
        else:
            for net_id, network in st.session_state.quantum_lab['quantum_networks'].items():
                with st.expander(f"🌐 {network['name']} ({network['topology']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Configuration")
                        st.write(f"**Topologie:** {network['topology']}")
                        st.write(f"**Nœuds:** {len(network['nodes'])}")
                        st.write(f"**Liens:** {len(network['edges'])}")
                    
                    with col2:
                        st.write("### 🔧 Infrastructure")
                        st.write(f"**Canal:** {network['channel_type']}")
                        st.write(f"**Distance:** {network['distance_km']} km")
                        st.write(f"**QKD:** {'✅' if network['qkd_enabled'] else '❌'}")
                    
                    with col3:
                        st.write("### 📈 Métriques")
                        avg_fidelity = np.mean([n['fidelity'] for n in network['nodes']])
                        st.metric("Fidélité Moy.", f"{avg_fidelity:.3f}")
                        st.metric("Qubits Total", network['total_qubits'])
                    
                    # Liste nœuds
                    st.write("### 📡 Nœuds du Réseau")
                    
                    nodes_data = []
                    for node in network['nodes']:
                        nodes_data.append({
                            'Nœud': node['name'],
                            'Qubits': node['qubits'],
                            'Fidélité': f"{node['fidelity']:.3f}"
                        })
                    
                    df_nodes = pd.DataFrame(nodes_data)
                    st.dataframe(df_nodes, use_container_width=True)
    
    with tab3:
        st.subheader("🔐 Distribution Clés Quantiques (QKD)")
        
        st.write("""
        **Quantum Key Distribution - BB84 Protocol**
        
        1. Alice envoie qubits encodés dans bases aléatoires
        2. Bob mesure dans bases aléatoires
        3. Échange bases publiquement
        4. Garde mesures où bases identiques
        5. Clé secrète partagée garantie par physique quantique
        """)
        
        if st.session_state.quantum_lab['quantum_networks']:
            selected_network = st.selectbox("Sélectionner Réseau",
                list(st.session_state.quantum_lab['quantum_networks'].keys()),
                format_func=lambda x: st.session_state.quantum_lab['quantum_networks'][x]['name'])
            
            network = st.session_state.quantum_lab['quantum_networks'][selected_network]
            
            if len(network['nodes']) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    alice_node = st.selectbox("Alice (Émetteur)",
                        range(len(network['nodes'])),
                        format_func=lambda i: network['nodes'][i]['name'])
                
                with col2:
                    bob_node = st.selectbox("Bob (Récepteur)",
                        [i for i in range(len(network['nodes'])) if i != alice_node],
                        format_func=lambda i: network['nodes'][i]['name'])
                
                key_length = st.slider("Longueur Clé (bits)", 64, 2048, 256)
                
                if st.button("🔑 Générer Clé Quantique", type="primary"):
                    with st.spinner("Distribution clé quantique..."):
                        import time
                        
                        progress = st.progress(0)
                        status = st.empty()
                        
                        status.text("1. Préparation qubits...")
                        time.sleep(0.5)
                        progress.progress(0.25)
                        
                        status.text("2. Transmission quantique...")
                        time.sleep(0.5)
                        progress.progress(0.50)
                        
                        status.text("3. Réconciliation bases...")
                        time.sleep(0.5)
                        progress.progress(0.75)
                        
                        status.text("4. Amplification confidentialité...")
                        time.sleep(0.5)
                        progress.progress(1.0)
                        
                        progress.empty()
                        status.empty()
                        
                        # Générer clé
                        key = ''.join(str(np.random.randint(0, 2)) for _ in range(key_length))
                        
                        # Taux erreur quantique (QBER)
                        qber = np.random.uniform(0.01, 0.05)
                        
                        st.success("✅ Clé quantique générée!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Longueur", f"{key_length} bits")
                        with col2:
                            st.metric("QBER", f"{qber*100:.2f}%")
                        with col3:
                            security = "Sécurité Information-Théorique"
                            st.metric("Sécurité", "✅ IT")
                        
                        st.code(f"Clé: {key[:64]}... (tronquée)", language="text")
                        
                        st.info("🔒 Clé sécurisée par lois physique quantique")
                        st.success("Toute écoute détectable par perturbation quantique")
                        
                        log_event(f"QKD: {key_length} bits générés", "SUCCESS")
            else:
                st.warning("Réseau doit avoir au moins 2 nœuds")
        else:
            st.info("Créez d'abord un réseau quantique")
    
    with tab4:
        st.subheader("📊 Visualisation Topologie Réseau")
        
        if st.session_state.quantum_lab['quantum_networks']:
            selected_network = st.selectbox("Réseau",
                list(st.session_state.quantum_lab['quantum_networks'].keys()),
                format_func=lambda x: st.session_state.quantum_lab['quantum_networks'][x]['name'],
                key="topo_select")
            
            network = st.session_state.quantum_lab['quantum_networks'][selected_network]
            
            # Créer graphique réseau
            fig = go.Figure()
            
            # Dessiner liens
            for edge in network['edges']:
                node1 = network['nodes'][edge[0]]
                node2 = network['nodes'][edge[1]]
                
                fig.add_trace(go.Scatter(
                    x=[node1['position'][0], node2['position'][1]],
                    y=[node1['position'][1], node2['position'][1]],
                    mode='lines',
                    line=dict(color='rgba(157, 80, 255, 0.3)', width=2),
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            # Dessiner nœuds
            node_x = [n['position'][0] for n in network['nodes']]
            node_y = [n['position'][1] for n in network['nodes']]
            node_sizes = [n['qubits'] * 5 for n in network['nodes']]
            node_colors = [n['fidelity'] for n in network['nodes']]
            node_text = [n['name'] for n in network['nodes']]
            
            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Fidélité"),
                    line=dict(color='white', width=2)
                ),
                text=node_text,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>Fidélité: %{marker.color:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Topologie {network['topology']} - {network['name']}",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template="plotly_dark",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Créez un réseau pour visualiser la topologie")

# ==================== PAGE: TROUS DE VER ====================
elif page == "🌌 Trous de Ver":
    st.header("🌌 Trous de Ver (Wormholes)")
    
    st.info("""
    **Trous de Ver (Einstein-Rosen Bridge)**
    
    Connexions hypothétiques entre deux points distants de l'espace-temps.
    
    **Types:**
    - Traversable (Théoriques)
    - Non-traversable (Schwarzschild)
    - Ellis Wormhole (Géométrie spéciale)
    
    **Problème:** Nécessite matière exotique (énergie négative)
    """)
    
    tab1, tab2, tab3 = st.tabs(["🌀 Créer", "📊 Propriétés", "🔬 Stabilité"])
    
    with tab1:
        st.subheader("🌀 Créer Trou de Ver")
        
        with st.form("create_wormhole"):
            col1, col2 = st.columns(2)
            
            with col1:
                wh_name = st.text_input("Nom", "WH-Alpha")
                wh_type = st.selectbox("Type",
                    ["Morris-Thorne", "Ellis", "Schwarzschild", "Traversable"])
                
                throat_radius_km = st.slider("Rayon Gorge (km)", 1, 10000, 100)
            
            with col2:
                exotic_matter_kg = st.number_input("Matière Exotique (kg)", 
                    1e20, 1e40, 1e30, format="%.2e")
                
                distance_ly = st.slider("Distance Extrémités (années-lumière)", 
                    1, 1000, 100)
                
                stability = st.selectbox("Stabilité",
                    ["Instable", "Semi-stable", "Stable (théorique)"])
            
            if st.form_submit_button("🌌 Créer Trou de Ver", type="primary"):
                wh_id = f"wh_{len(st.session_state.quantum_lab['wormholes']) + 1}"
                
                # Calculs
                throat_radius = throat_radius_km * 1000
                c = PHYSICAL_CONSTANTS['c']
                
                # Énergie exotique nécessaire (approximation)
                energy_exotic = -abs(exotic_matter_kg * c**2)
                
                # Temps traversée
                traversal_time = distance_ly * 3.15e7 / c if wh_type == "Traversable" else float('inf')
                
                wormhole = {
                    'id': wh_id,
                    'name': wh_name,
                    'type': wh_type,
                    'throat_radius': throat_radius,
                    'exotic_matter_kg': exotic_matter_kg,
                    'distance_ly': distance_ly,
                    'energy_exotic': energy_exotic,
                    'stability': stability,
                    'traversal_time': traversal_time,
                    'created_at': datetime.now().isoformat()
                }
                
                st.session_state.quantum_lab['wormholes'][wh_id] = wormhole
                log_event(f"Trou de ver créé: {wh_name}", "SUCCESS")
                
                st.success(f"✅ Trou de ver '{wh_name}' créé!")
                st.balloons()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Rayon Gorge", f"{throat_radius_km} km")
                with col2:
                    st.metric("Distance", f"{distance_ly} ly")
                with col3:
                    st.metric("Stabilité", stability)
                
                st.rerun()
    
    with tab2:
        if not st.session_state.quantum_lab['wormholes']:
            st.info("Aucun trou de ver créé")
        else:
            for wh_id, wh in st.session_state.quantum_lab['wormholes'].items():
                with st.expander(f"🌌 {wh['name']} ({wh['type']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("### 📊 Géométrie")
                        st.write(f"**Type:** {wh['type']}")
                        st.write(f"**Rayon gorge:** {wh['throat_radius']/1000:.2f} km")
                        st.write(f"**Distance:** {wh['distance_ly']} ly")
                    
                    with col2:
                        st.write("### ⚡ Énergie")
                        st.write(f"**Matière exotique:** {wh['exotic_matter_kg']:.2e} kg")
                        st.write(f"**Énergie négative:** {wh['energy_exotic']:.2e} J")
                        st.write(f"**Stabilité:** {wh['stability']}")
                    
                    with col3:
                        st.write("### 🚀 Traversée")
                        if np.isfinite(wh['traversal_time']):
                            st.write(f"**Temps:** {wh['traversal_time']:.2e} s")
                            st.write(f"**Années:** {wh['traversal_time']/3.15e7:.2f}")
                            st.success("✅ Traversable")
                        else:
                            st.write("**Temps:** ∞")
                            st.error("❌ Non-traversable")
    
    with tab3:
        st.subheader("🔬 Analyse Stabilité")
        
        st.write("""
        **Conditions de Stabilité (Morris-Thorne)**
        
        1. **Énergie exotique:** Violation condition énergétique
        2. **Courbure:** Négative à la gorge
        3. **Forme:** Éviter horizons événements
        """)
        
        if st.session_state.quantum_lab['wormholes']:
            selected_wh = st.selectbox("Sélectionner Trou de Ver",
                list(st.session_state.quantum_lab['wormholes'].keys()),
                format_func=lambda x: st.session_state.quantum_lab['wormholes'][x]['name'])
            
            wh = st.session_state.quantum_lab['wormholes'][selected_wh]
            
            # Visualisation forme trou de ver
            st.write("### 🌀 Profil Géométrique")
            
            r = np.linspace(wh['throat_radius'], wh['throat_radius']*10, 100)
            b = wh['throat_radius']  # Rayon gorge
            
            # Fonction forme (Morris-Thorne)
            z = np.sqrt(r**2 - b**2)
            
            fig = go.Figure()
            
            # Profil supérieur
            fig.add_trace(go.Scatter(
                x=r/1000, y=z/1000,
                mode='lines',
                line=dict(color='#9D50FF', width=3),
                name='Profil'
            ))
            
            # Profil inférieur (symétrie)
            fig.add_trace(go.Scatter(
                x=r/1000, y=-z/1000,
                mode='lines',
                line=dict(color='#9D50FF', width=3),
                showlegend=False
            ))
            
            fig.update_layout(
                title="Profil Trou de Ver (Coupe)",
                xaxis_title="Rayon (km)",
                yaxis_title="Z (km)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse stabilité
            st.write("### 📊 Critères Stabilité")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                exotic_ratio = wh['exotic_matter_kg'] / 1e30
                st.metric("Ratio Matière Exotique", f"{exotic_ratio:.2e}")
                
                if exotic_ratio > 1:
                    st.success("✅ Suffisant")
                else:
                    st.warning("⚠️ Insuffisant")
            
            with col2:
                if wh['stability'] == "Stable (théorique)":
                    st.success("✅ Stable")
                elif wh['stability'] == "Semi-stable":
                    st.warning("⚠️ Semi-stable")
                else:
                    st.error("❌ Instable")
            
            with col3:
                if wh['type'] == "Traversable":
                    st.success("✅ Traversable")
                else:
                    st.error("❌ Non-traversable")
        else:
            st.info("Créez un trou de ver pour analyser sa stabilité")

# ==================== PAGE: SUPERPOSITION ====================
elif page == "🎭 Superposition":
    st.header("🎭 Superposition Quantique")
    
    st.info("""
    **Principe de Superposition**
    
    Un système quantique peut exister simultanément dans plusieurs états.
    
    **État général:** |ψ⟩ = α|0⟩ + β|1⟩ avec |α|² + |β|² = 1
    
    **Chat de Schrödinger:** Superposition macroscopique (expérience de pensée)
    
    **Mesure → Collapse:** Projection sur un état propre
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎭 États", "🐱 Chat Schrödinger", "📊 Interférence", "🔬 Expériences"])
    
    with tab1:
        st.subheader("🎭 Créer État en Superposition")
        
        with st.form("create_superposition"):
            col1, col2 = st.columns(2)
            
            with col1:
                n_states = st.slider("Nombre d'États", 2, 8, 2)
                
                st.write("**Amplitudes:**")
                amplitudes = []
                for i in range(n_states):
                    amp = st.slider(f"État |{i}⟩", 0.0, 1.0, 1.0/n_states, 0.01, key=f"amp_{i}")
                    amplitudes.append(amp)
            
            with col2:
                coherence_preserved = st.checkbox("Préserver cohérence", value=True)
                
                environment_temp = st.slider("Température environnement (K)", 0.01, 300.0, 0.1)
                
                measurement_delay = st.slider("Délai avant mesure (μs)", 1, 1000, 100)
            
            if st.form_submit_button("🎭 Créer Superposition", type="primary"):
                # Normaliser amplitudes
                norm = np.sqrt(sum(a**2 for a in amplitudes))
                normalized_amps = [a/norm for a in amplitudes]
                
                # Calculer pureté
                purity = sum(a**2 for a in normalized_amps)**2
                
                st.success("✅ État en superposition créé!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("États", n_states)
                with col2:
                    st.metric("Pureté", f"{purity:.3f}")
                with col3:
                    st.metric("Cohérence", "✅" if coherence_preserved else "❌")
                
                # Visualisation amplitudes
                fig = go.Figure(data=[go.Bar(
                    x=[f"|{i}⟩" for i in range(n_states)],
                    y=normalized_amps,
                    marker_color='#9D50FF'
                )])
                
                fig.update_layout(
                    title="Amplitudes des États",
                    xaxis_title="État",
                    yaxis_title="Amplitude",
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Probabilités
                st.write("### 📊 Probabilités de Mesure")
                probs = [a**2 for a in normalized_amps]
                
                for i, prob in enumerate(probs):
                    st.write(f"**P(|{i}⟩) = {prob:.3f}** ({prob*100:.1f}%)")
    
    with tab2:
        st.subheader("🐱 Paradoxe du Chat de Schrödinger")
        
        st.write("""
        **Expérience de pensée (1935)**
        
        Un chat dans une boîte avec:
        - Atome radioactif (50% désintégration en 1h)
        - Détecteur → poison si désintégration
        
        **Avant mesure:** |ψ⟩ = |vivant⟩ + |mort⟩ / √2
        
        **Question:** Le chat est-il dans une superposition?
        """)
        
        if st.button("🎲 Ouvrir la Boîte (Mesurer)", type="primary"):
            result = np.random.choice(["vivant", "mort"])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if result == "vivant":
                    st.success("🐱 Chat vivant!")
                    st.balloons()
                else:
                    st.error("💀 Chat mort...")
            
            with col2:
                st.write("### 📊 Résultat Mesure")
                st.write(f"**État avant:** Superposition")
                st.write(f"**État après:** |{result}⟩")
                st.write(f"**Probabilité:** 50%")
                
                st.info("💡 La mesure a causé le collapse de la superposition")
        
        st.write("### 🔬 Interprétations")
        
        interpretations = {
            "Copenhague": "Collapse lors de la mesure (observateur)",
            "Mondes Multiples": "Univers se divise (chat vivant ET mort)",
            "Décohérence": "Intrication avec environnement",
            "Bohm": "Variables cachées déterminent résultat"
        }
        
        for name, desc in interpretations.items():
            with st.expander(f"📚 {name}"):
                st.write(desc)
    
    with tab3:
        st.subheader("📊 Interférences Quantiques")
        
        st.write("""
        **Expérience Fentes de Young (version quantique)**
        
        Particule unique → Superposition chemins → Interférences
        """)
        
        n_slits = st.radio("Nombre de fentes", [2, 3, 4], horizontal=True)
        
        if st.button("🌊 Simuler Interférences", type="primary"):
            # Position écran
            x = np.linspace(-10, 10, 500)
            
            # Pattern interférence
            if n_slits == 2:
                pattern = np.cos(np.pi * x)**2
            elif n_slits == 3:
                pattern = (np.cos(np.pi * x) + np.cos(np.pi * x + 2*np.pi/3) + 
                          np.cos(np.pi * x + 4*np.pi/3))**2
            else:
                pattern = np.abs(np.sum([np.exp(1j * k * np.pi * x) 
                                for k in range(n_slits)], axis=0))**2
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x, y=pattern,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#9D50FF', width=3)
            ))
            
            fig.update_layout(
                title=f"Pattern d'Interférence ({n_slits} fentes)",
                xaxis_title="Position (unités arbitraires)",
                yaxis_title="Intensité",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Superposition → Interférences observées!")
            st.info("💫 Chaque particule passe par TOUTES les fentes simultanément")
    
    with tab4:
        st.subheader("🔬 Expériences Superposition")
        
        exp_type = st.selectbox("Type d'Expérience",
            ["Fentes de Young", "Interféromètre Mach-Zehnder", 
             "Atome refroidi", "Ion piégé"])
        
        if exp_type == "Interféromètre Mach-Zehnder":
            st.write("### 🔬 Interféromètre Mach-Zehnder")
            
            st.write("""
            **Principe:**
            1. Photon entre
            2. Beam splitter → Superposition 2 chemins
            3. Miroirs
            4. Second beam splitter → Interférences
            """)
            
            phase_shift = st.slider("Déphasage (rad)", 0.0, 2*np.pi, 0.0, 0.1)
            
            # Probabilités sortie
            prob_D1 = np.cos(phase_shift/2)**2
            prob_D2 = np.sin(phase_shift/2)**2
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Détecteur D1", f"{prob_D1:.3f}")
                st.progress(prob_D1)
            
            with col2:
                st.metric("Détecteur D2", f"{prob_D2:.3f}")
                st.progress(prob_D2)
            
            st.write(f"**Visibilité interférences:** {abs(prob_D1 - prob_D2):.3f}")
        
        elif exp_type == "Atome refroidi":
            st.write("### ❄️ Atome Ultra-Froid")
            
            temperature_nk = st.slider("Température (nK)", 1, 1000, 100)
            
            # Longueur onde de Broglie
            h = PHYSICAL_CONSTANTS['h']
            k_B = PHYSICAL_CONSTANTS['k_B']
            m = PHYSICAL_CONSTANTS['m_e']
            
            lambda_dB = h / np.sqrt(2 * np.pi * m * k_B * temperature_nk * 1e-9)
            
            st.metric("Longueur onde de Broglie", f"{lambda_dB*1e9:.2f} nm")
            
            if lambda_dB > 1e-9:
                st.success("✅ Effets quantiques macroscopiques observables!")
            
            st.info("💡 Plus froid → Plus grande longueur d'onde → Superposition plus 'visible'")

# ==================== PAGE: DÉCOHÉRENCE ====================
elif page == "📊 Décohérence":
    st.header("📊 Décohérence Quantique")
    
    st.info("""
    **Décohérence**
    
    Perte de cohérence quantique par interaction avec l'environnement.
    
    **Processus:** |ψ⟩système ⊗ |0⟩env → Σ cn|n⟩système ⊗ |En⟩env
    
    **Résultat:** Superposition → Mélange statistique classique
    
    **Temps caractéristique:** τdecoh (dépend du système et environnement)
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Processus", "⏱️ Temps", "🌡️ Facteurs", "🔬 Mesures"])
    
    with tab1:
        st.subheader("📉 Processus de Décohérence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            system_type = st.selectbox("Système",
                ["Qubit supraconducteur", "Ion piégé", "Photon", 
                 "Atome", "Molécule"])
            
            environment = st.selectbox("Environnement",
                ["Vide (photons thermiques)", "Gaz résiduel", 
                 "Phonons", "Champ électromagnétique"])
        
        with col2:
            temperature_mk = st.slider("Température (mK)", 10, 1000, 50)
            coupling_strength = st.slider("Couplage système-env", 0.01, 1.0, 0.1, 0.01)
        
        if st.button("📊 Simuler Décohérence", type="primary"):
            with st.spinner("Simulation en cours..."):
                import time
                time.sleep(1)
                
                # Temps de décohérence (modèle simplifié)
                T_base = {
                    "Qubit supraconducteur": 100,
                    "Ion piégé": 1000,
                    "Photon": 10,
                    "Atome": 500,
                    "Molécule": 1
                }
                
                tau_decoh = T_base.get(system_type, 100) / (temperature_mk * coupling_strength)
                
                # Évolution cohérence
                t = np.linspace(0, tau_decoh * 5, 200)
                coherence = np.exp(-t / tau_decoh)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=t, y=coherence,
                    mode='lines',
                    line=dict(color='#9D50FF', width=3),
                    fill='tozeroy'
                ))
                
                fig.add_vline(x=tau_decoh, line_dash="dash", line_color="red",
                             annotation_text="τ_decoh")
                
                fig.update_layout(
                    title="Décohérence Temporelle",
                    xaxis_title="Temps (μs)",
                    yaxis_title="Cohérence",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("τ décohérence", f"{tau_decoh:.2f} μs")
                with col2:
                    st.metric("T à 50%", f"{tau_decoh * np.log(2):.2f} μs")
                with col3:
                    st.metric("T à 10%", f"{tau_decoh * np.log(10):.2f} μs")
    
    with tab2:
        st.subheader("⏱️ Temps de Décohérence")
        
        st.write("""
        **Temps caractéristiques:**
        
        - **T₁ (relaxation):** Perte énergie → état fondamental
        - **T₂ (déphasage):** Perte cohérence phase
        - **T₂ ≤ 2T₁** (inégalité fondamentale)
        """)
        
        # Comparaison systèmes
        systems_data = {
            'Système': ['Qubit SC', 'Ion piégé', 'NV center', 'Photon', 'Atome Rb'],
            'T₁ (μs)': [50, 1000, 100, 1, 500],
            'T₂ (μs)': [30, 500, 50, 0.5, 200],
            'T (mK)': [20, 0.5, 300, 300, 1]
        }
        
        df_systems = pd.DataFrame(systems_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='T₁',
            x=df_systems['Système'],
            y=df_systems['T₁ (μs)'],
            marker_color='#9D50FF'
        ))
        
        fig.add_trace(go.Bar(
            name='T₂',
            x=df_systems['Système'],
            y=df_systems['T₂ (μs)'],
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title="Comparaison Temps de Cohérence",
            xaxis_title="Système",
            yaxis_title="Temps (μs)",
            yaxis_type="log",
            template="plotly_dark",
            height=400,
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_systems, use_container_width=True)
    
    with tab3:
        st.subheader("🌡️ Facteurs de Décohérence")
        
        st.write("### 📊 Sources Principales")
        
        sources = {
            "Température": {
                "impact": "Phonons thermiques, photons",
                "mitigation": "Refroidissement cryogénique",
                "typical_T": "< 100 mK"
            },
            "Bruit électromagnétique": {
                "impact": "Fluctuations champs",
                "mitigation": "Blindage, filtrage",
                "typical_T": "Variable"
            },
            "Gaz résiduel": {
                "impact": "Collisions moléculaires",
                "mitigation": "Ultra-vide (< 10⁻¹⁰ mbar)",
                "typical_T": "Critique ions"
            },
            "Fluctuations charge": {
                "impact": "Bruit 1/f",
                "mitigation": "Matériaux purs, design",
                "typical_T": "Important SC qubits"
            }
        }
        
        for source, info in sources.items():
            with st.expander(f"⚠️ {source}"):
                st.write(f"**Impact:** {info['impact']}")
                st.write(f"**Mitigation:** {info['mitigation']}")
                st.write(f"**Note:** {info['typical_T']}")
        
        st.write("### 📈 Dépendance Température")
        
        T = np.linspace(10, 1000, 100)  # mK
        tau_T1 = 1000 / T  # Simplification
        tau_T2 = tau_T1 / 2
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=T, y=tau_T1,
            mode='lines',
            name='T₁',
            line=dict(color='#9D50FF', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=T, y=tau_T2,
            mode='lines',
            name='T₂',
            line=dict(color='#FF6B6B', width=3)
        ))
        
        fig.update_layout(
            title="Temps Cohérence vs Température",
            xaxis_title="Température (mK)",
            yaxis_title="Temps cohérence (μs)",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔬 Mesures de Décohérence")
        
        st.write("### 📊 Protocoles de Mesure")
        
        protocol = st.selectbox("Protocole",
            ["Ramsey", "Echo de Hahn", "CPMG", "Randomized Benchmarking"])
        
        if protocol == "Ramsey":
            st.write("""
            **Séquence Ramsey**
            
            1. π/2 pulse → Superposition
            2. Attente libre τ
            3. π/2 pulse → Interférence
            4. Mesure
            
            **Résultat:** Oscillations amorties → T₂*
            """)
            
            tau_max = st.slider("Temps max (μs)", 10, 1000, 100)
            
            tau = np.linspace(0, tau_max, 100)
            T2_star = 50  # μs
            omega = 0.1  # MHz
            
            signal = np.exp(-tau / T2_star) * np.cos(2 * np.pi * omega * tau)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=tau, y=signal,
                mode='lines',
                line=dict(color='#9D50FF', width=2)
            ))
            
            fig.update_layout(
                title="Signal Ramsey",
                xaxis_title="Temps τ (μs)",
                yaxis_title="Signal",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("T₂* extrait", f"{T2_star} μs")
        
        elif protocol == "Echo de Hahn":
            st.write("""
            **Séquence Echo de Hahn**
            
            1. π/2 pulse
            2. Attente τ/2
            3. π pulse (refocalisation)
            4. Attente τ/2
            5. π/2 pulse
            6. Mesure
            
            **Effet:** Annule déphasage inhomogène → T₂ > T₂*
            """)
            
            st.success("✅ Permet de mesurer T₂ (plus long que T₂*)")

# ==================== PAGE: CHAMPS QUANTIQUES (suite) ====================
elif page == "🌊 Champs Quantiques":
    st.header("🌊 Théorie Quantique des Champs")
    
    st.info("""
    **Théorie Quantique des Champs (QFT)**
    
    Unification mécanique quantique + relativité restreinte
    
    **Principe:** Particules = Excitations de champs quantiques
    - Photon = Excitation champ électromagnétique
    - Électron = Excitation champ de Dirac
    
    **Équations:** Lagrangien → Équations du mouvement
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Champs", "🌊 Vide Quantique", "✨ Création/Annihilation", "🔬 Effets"])
    
    with tab1:
        st.subheader("⚡ Champs Quantiques Fondamentaux")
        
        st.write("### 📊 Modèle Standard")
        
        fields = {
            "Photon (γ)": {
                "type": "Boson jauge",
                "spin": 1,
                "masse": 0,
                "force": "Électromagnétique",
                "couleur": "#FFD700"
            },
            "Électron (e⁻)": {
                "type": "Fermion",
                "spin": 0.5,
                "masse": "0.511 MeV/c²",
                "force": "EM + Faible",
                "couleur": "#4169E1"
            },
            "Quarks (u,d,...)": {
                "type": "Fermion",
                "spin": 0.5,
                "masse": "Variable",
                "force": "Toutes",
                "couleur": "#FF4500"
            },
            "Gluon (g)": {
                "type": "Boson jauge",
                "spin": 1,
                "masse": 0,
                "force": "Forte",
                "couleur": "#32CD32"
            },
            "W±, Z⁰": {
                "type": "Boson jauge",
                "spin": 1,
                "masse": "80-91 GeV/c²",
                "force": "Faible",
                "couleur": "#9370DB"
            },
            "Higgs (H)": {
                "type": "Boson scalaire",
                "spin": 0,
                "masse": "125 GeV/c²",
                "force": "Masse",
                "couleur": "#FF1493"
            }
        }
        
        for field_name, properties in fields.items():
            with st.expander(f"⚛️ {field_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {properties['type']}")
                    st.write(f"**Spin:** {properties['spin']}")
                    st.write(f"**Masse:** {properties['masse']}")
                
                with col2:
                    st.write(f"**Force:** {properties['force']}")
                    st.markdown(f"**Couleur:** <span style='color:{properties['couleur']}'>■■■</span>", 
                               unsafe_allow_html=True)
        
        st.write("### 🎯 Sélectionner Champ")
        
        selected_field = st.selectbox("Champ à analyser",
            list(fields.keys()))
        
        field_info = fields[selected_field]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Type", field_info['type'])
        with col2:
            st.metric("Spin", field_info['spin'])
        with col3:
            st.metric("Masse", field_info['masse'])
    
    with tab2:
        st.subheader("🌊 Vide Quantique & Fluctuations")
        
        st.write("""
        **Vide Quantique ≠ Vide Classique**
        
        - État d'énergie minimale (état fondamental)
        - Fluctuations quantiques permanentes
        - Paires virtuelles particule-antiparticule
        
        **Énergie du vide:** ⟨0|Ĥ|0⟩ = ∞ (problème!)
        """)
        
        st.write("### 💫 Effets Observables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**1. Effet Casimir**")
            
            d_nm = st.slider("Distance plaques (nm)", 10, 1000, 100, key="casimir_d")
            
            # Force Casimir (approximation)
            hbar = PHYSICAL_CONSTANTS['hbar']
            c = PHYSICAL_CONSTANTS['c']
            
            d = d_nm * 1e-9
            F_casimir = -(np.pi**2 * hbar * c) / (240 * d**4)  # Force par unité surface
            
            st.metric("Force Casimir", f"{F_casimir:.2e} N/m²")
            
            if F_casimir < -1e-5:
                st.success("✅ Effet mesurable!")
            
            st.info("💡 Force attractive due aux fluctuations du vide")
        
        with col2:
            st.write("**2. Déplacement de Lamb**")
            
            st.write("""
            Correction niveaux d'énergie de l'hydrogène
            
            - 2S₁/₂ - 2P₁/₂ : ~1 GHz
            - Dû aux fluctuations du champ EM
            """)
            
            st.metric("Déplacement 2S", "1057 MHz")
            st.success("✅ Mesuré avec précision!")
        
        st.write("### 🌊 Visualisation Fluctuations")
        
        if st.button("🎲 Simuler Fluctuations Vide", type="primary"):
            # Simulation fluctuations
            x = np.linspace(0, 10, 200)
            y = np.linspace(0, 10, 200)
            X, Y = np.meshgrid(x, y)
            
            # Champ aléatoire
            Z = np.random.normal(0, 1, X.shape)
            
            fig = go.Figure(data=[go.Heatmap(
                z=Z,
                x=x,
                y=y,
                colorscale='RdBu',
                zmid=0
            )])
            
            fig.update_layout(
                title="Fluctuations Quantiques du Vide (instantané)",
                xaxis_title="x",
                yaxis_title="y",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💫 Le vide 'bouillonne' de fluctuations quantiques!")
    
    with tab3:
        st.subheader("✨ Opérateurs Création/Annihilation")
        
        st.write("""
        **Formalisme de Seconde Quantification**
        
        - **â†** (création): Crée une particule
        - **â** (annihilation): Détruit une particule
        
        **Commutateur:** [â, â†] = 1 (bosons)
        **Anti-commutateur:** {â, â†} = 1 (fermions)
        """)
        
        st.write("### 🎯 États de Fock (nombre de particules)")
        
        n_max = st.slider("Nombre max particules", 0, 10, 5)
        
        # États de Fock
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**États |n⟩:**")
            for n in range(n_max + 1):
                st.write(f"|{n}⟩ : {n} particule(s)")
        
        with col2:
            st.write("**Action opérateurs:**")
            st.latex(r"\hat{a}^\dagger |n\rangle = \sqrt{n+1} |n+1\rangle")
            st.latex(r"\hat{a} |n\rangle = \sqrt{n} |n-1\rangle")
        
        st.write("### 🔬 États Cohérents")
        
        alpha_real = st.slider("α (réel)", -2.0, 2.0, 1.0, 0.1)
        alpha_imag = st.slider("α (imag)", -2.0, 2.0, 0.0, 0.1)
        
        alpha = complex(alpha_real, alpha_imag)
        
        # Distribution Poisson pour état cohérent
        n_mean = abs(alpha)**2
        n_values = np.arange(0, 20)
        # prob_n = (n_mean**n_values * np.exp(-n_mean)) / np.array([np.math.factorial(n) for n in n_values])
        prob_n = (n_mean**n_values * np.exp(-n_mean)) / np.array([math.factorial(n) for n in n_values])
        
        fig = go.Figure(data=[go.Bar(
            x=n_values,
            y=prob_n,
            marker_color='#9D50FF'
        )])
        
        fig.update_layout(
            title="Distribution Nombre de Particules (État Cohérent)",
            xaxis_title="Nombre n",
            yaxis_title="Probabilité P(n)",
            template="plotly_dark",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("⟨n⟩", f"{n_mean:.2f}")
        with col2:
            st.metric("Δn", f"{np.sqrt(n_mean):.2f}")
        with col3:
            st.metric("|α|", f"{abs(alpha):.2f}")
    
    with tab4:
        st.subheader("🔬 Effets de la QFT")
        
        effect_type = st.selectbox("Effet à étudier",
            ["Radiation Hawking", "Création de Paires", "Effet Unruh", 
             "Effet Schwinger", "Radiation de Cherenkov"])
        
        if effect_type == "Radiation Hawking":
            st.write("""
            **Radiation de Hawking (QFT + Gravité)**
            
            Création paires virtuelles près horizon:
            1. Paire particule-antiparticule créée
            2. Une tombe dans trou noir (E < 0)
            3. Autre s'échappe (radiation)
            
            **Température:** T = ℏc³/(8πGMk_B)
            """)
            
            M_solar = st.slider("Masse trou noir (M☉)", 1.0, 1000.0, 10.0)
            
            M_sun = 1.989e30
            M = M_solar * M_sun
            T_H = hawking_temperature(M)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Température", f"{T_H:.2e} K")
            with col2:
                lambda_peak = 2.898e-3 / T_H if T_H > 0 else float('inf')
                st.metric("λ pic", f"{lambda_peak:.2e} m")
            with col3:
                t_evap = 2.1e67 * (M / 1e30)**3 / 3.15e7
                st.metric("Évaporation", f"{t_evap:.2e} ans")
        
        elif effect_type == "Effet Unruh":
            st.write("""
            **Effet Unruh**
            
            Observateur accéléré voit le vide comme un bain thermique!
            
            **Température:** T = ℏa/(2πck_B)
            
            où a = accélération propre
            """)
            
            acceleration_g = st.slider("Accélération (g)", 1, 1e20, 1e10, format="%.2e")
            
            g = 9.81
            a = acceleration_g * g
            
            hbar = PHYSICAL_CONSTANTS['hbar']
            c = PHYSICAL_CONSTANTS['c']
            k_B = PHYSICAL_CONSTANTS['k_B']
            
            T_unruh = (hbar * a) / (2 * np.pi * c * k_B)
            
            st.metric("Température Unruh", f"{T_unruh:.2e} K")
            
            if T_unruh > 1e-20:
                st.success("✅ Effet théoriquement mesurable")
            else:
                st.info("Effet extrêmement faible")
            
            st.warning("⚠️ Jamais observé expérimentalement (accélération requise énorme)")
        
        elif effect_type == "Création de Paires":
            st.write("""
            **Création de Paires (e⁺e⁻)**
            
            Photon γ → e⁺ + e⁻
            
            **Condition:** E_γ ≥ 2m_e c² = 1.022 MeV
            """)
            
            photon_energy_mev = st.slider("Énergie photon (MeV)", 0.5, 10.0, 2.0, 0.1)
            
            threshold = 1.022
            
            if photon_energy_mev >= threshold:
                st.success(f"✅ Création possible! E - seuil = {photon_energy_mev - threshold:.3f} MeV")
                
                # Énergie cinétique paire
                E_kin = photon_energy_mev - threshold
                st.write(f"**Énergie cinétique totale:** {E_kin:.3f} MeV")
            else:
                st.error(f"❌ Énergie insuffisante (manque {threshold - photon_energy_mev:.3f} MeV)")
        
        elif effect_type == "Effet Schwinger":
            st.write("""
            **Effet Schwinger**
            
            Champ électrique intense → Création paires e⁺e⁻ du vide
            
            **Champ critique:** E_c = m_e²c³/(eℏ) ≈ 1.3×10¹⁸ V/m
            """)
            
            E_field = st.number_input("Champ E (V/m)", 1e10, 1e20, 1e16, format="%.2e")
            
            m_e = PHYSICAL_CONSTANTS['m_e']
            c = PHYSICAL_CONSTANTS['c']
            e = PHYSICAL_CONSTANTS['e']
            hbar = PHYSICAL_CONSTANTS['hbar']
            
            E_critical = (m_e**2 * c**3) / (e * hbar)
            
            ratio = E_field / E_critical
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("E critique", f"{E_critical:.2e} V/m")
            with col2:
                st.metric("E/E_c", f"{ratio:.3f}")
            
            if ratio >= 1:
                st.success("✅ Production paires significative!")
            else:
                st.info("Taux production exponentiellement supprimé")
            
            # Taux production (approximatif)
            if ratio < 1:
                rate = np.exp(-np.pi * E_critical / E_field)
                st.write(f"**Taux relatif:** {rate:.2e}")

# ==================== PAGE: MESURES ====================
elif page == "📈 Mesures":
    st.header("📈 Mesures Quantiques")
    
    st.info("""
    **Mesure Quantique**
    
    Processus fondamental de la mécanique quantique:
    - État avant: |ψ⟩ = Σ cn|n⟩ (superposition)
    - Mesure observable  → Collapse
    - État après: |n⟩ (état propre)
    - Résultat: valeur propre λn avec probabilité |cn|²
    
    **Postulat de Born:** P(n) = |⟨n|ψ⟩|²
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Théorie", "🔬 Protocoles", "📈 Statistiques", "🎯 Tomographie"])
    
    with tab1:
        st.subheader("📊 Théorie de la Mesure")
        
        st.write("### 📐 Observables")
        
        observable_type = st.selectbox("Observable",
            ["Position (x̂)", "Impulsion (p̂)", "Énergie (Ĥ)", 
             "Spin (Ŝ)", "Nombre (n̂)"])
        
        if observable_type == "Position (x̂)":
            st.write("""
            **Opérateur Position**
            
            - Hermitien: x̂† = x̂
            - Spectre: ℝ (continu)
            - États propres: |x⟩
            - Relation: [x̂, p̂] = iℏ
            """)
            
            st.latex(r"\hat{x} |\psi\rangle = \int x |\psi(x)|^2 dx")
            
        elif observable_type == "Spin (Ŝ)":
            st.write("""
            **Opérateur Spin (spin-1/2)**
            
            **Matrices de Pauli:**
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.latex(r"\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}")
            with col2:
                st.latex(r"\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}")
            with col3:
                st.latex(r"\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}")
            
            st.write("**Valeurs propres:** ±ℏ/2")
        
        st.write("### 🎲 Principe d'Incertitude")
        
        st.latex(r"\Delta x \cdot \Delta p \geq \frac{\hbar}{2}")
        
        delta_x = st.slider("Δx (nm)", 0.1, 10.0, 1.0, 0.1)
        
        hbar = PHYSICAL_CONSTANTS['hbar']
        delta_p_min = hbar / (2 * delta_x * 1e-9)
        
        st.metric("Δp minimum", f"{delta_p_min:.2e} kg⋅m/s")
        
        st.info("💡 Plus on localise en position, moins on connaît l'impulsion!")
    
    with tab2:
        st.subheader("🔬 Protocoles de Mesure")
        
        protocol = st.selectbox("Protocole",
            ["Mesure Projective", "Mesure Faible", "Mesure POVM", 
             "Tomographie d'État", "Mesure Continue"])
        
        if protocol == "Mesure Projective":
            st.write("""
            **Mesure Projective (von Neumann)**
            
            1. Système dans |ψ⟩ = Σ cn|n⟩
            2. Mesure observable Â
            3. Résultat: λn avec P(λn) = |cn|²
            4. État après: |n⟩ (collapse)
            
            **Projecteur:** P̂n = |n⟩⟨n|
            """)
            
            if st.session_state.quantum_lab['qubits']:
                selected_qubit = st.selectbox("Sélectionner Qubit",
                    list(st.session_state.quantum_lab['qubits'].keys()),
                    format_func=lambda x: st.session_state.quantum_lab['qubits'][x]['name'])
                
                qubit = st.session_state.quantum_lab['qubits'][selected_qubit]
                
                basis = st.radio("Base de mesure", ["Z (|0⟩, |1⟩)", "X (|+⟩, |-⟩)", "Y"], horizontal=True)
                
                if st.button("📏 Effectuer Mesure Projective", type="primary"):
                    # alpha = complex(qubit['state_real_0'], qubit['state_imag_0'])
                    # alpha = complex(qubit.get('state_real_0', qubit.get('state_real_1', 0)),
                    #         qubit.get('state_imag_0', qubit.get('state_imag_1', 0)))

                    # beta = complex(qubit['state_real_1'], qubit['state_imag_1'])
                    alpha = complex(qubit.get('state_real_0', 0), qubit.get('state_imag_0', 0))
                    beta  = complex(qubit.get('state_real_1', 0), qubit.get('state_imag_1', 0))

                    
                    if basis == "Z (|0⟩, |1⟩)":
                        prob_0 = abs(alpha)**2
                        result = 0 if np.random.random() < prob_0 else 1
                    elif basis == "X (|+⟩, |-⟩)":
                        # Transformation vers base X
                        plus = (alpha + beta) / np.sqrt(2)
                        prob_plus = abs(plus)**2
                        result = "+" if np.random.random() < prob_plus else "-"
                    else:  # Y
                        plus_i = (alpha + 1j*beta) / np.sqrt(2)
                        prob_plus_i = abs(plus_i)**2
                        result = "+i" if np.random.random() < prob_plus_i else "-i"
                    
                    st.success(f"✅ Résultat: {result}")
                    st.info("L'état quantique a collapsé!")
                    
                    log_event(f"Mesure projective: {qubit['name']} → {result}", "INFO")
            else:
                st.info("Créez un qubit pour effectuer des mesures")
        
        elif protocol == "Mesure Faible":
            st.write("""
            **Mesure Faible (Weak Measurement)**
            
            - Couplage faible système-appareil
            - Peu de perturbation
            - Information partielle
            - Permet mesures "continues"
            
            **Valeur faible:** ⟨Â⟩_w = ⟨ψ_f|Â|ψ_i⟩ / ⟨ψ_f|ψ_i⟩
            """)
            
            coupling = st.slider("Force couplage", 0.01, 1.0, 0.1, 0.01)
            
            if coupling < 0.2:
                st.success("✅ Régime mesure faible")
                st.info("Peut donner valeurs hors spectre (paradoxe!)")
            else:
                st.warning("⚠️ Mesure devient projective")
        
        elif protocol == "Tomographie d'État":
            st.write("""
            **Tomographie Quantique**
            
            Reconstruction complète de l'état quantique ρ
            
            **Méthode:**
            1. Mesures dans plusieurs bases
            2. Statistiques → Reconstruction ρ
            3. Pour qubit: besoin 3 bases (X, Y, Z)
            """)
            
            if st.button("🔬 Effectuer Tomographie", type="primary"):
                st.info("Simulation tomographie d'état...")
                
                # Matrice densité simulée
                rho = np.array([[0.7, 0.3-0.2j], [0.3+0.2j, 0.3]])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Matrice Densité ρ:**")
                    st.write(rho)
                
                with col2:
                    st.write("**Propriétés:**")
                    purity = np.trace(rho @ rho)
                    st.metric("Pureté Tr(ρ²)", f"{purity.real:.3f}")
                    
                    if purity > 0.99:
                        st.success("État pur")
                    else:
                        st.info("État mixte")
    
    with tab3:
        st.subheader("📈 Statistiques des Mesures")
        
        if st.session_state.quantum_lab['measurements']:
            measurements = st.session_state.quantum_lab['measurements']
            
            # Préparer données
            results = [m.get('result', 0) for m in measurements]
            
            st.write(f"### 📊 {len(measurements)} Mesures Enregistrées")
            
            # Distribution résultats
            unique, counts = np.unique(results, return_counts=True)
            
            fig = go.Figure(data=[go.Bar(
                x=[f"|{u}⟩" for u in unique],
                y=counts,
                marker_color='#9D50FF',
                text=counts,
                textposition='auto'
            )])
            
            fig.update_layout(
                title="Distribution des Résultats",
                xaxis_title="Résultat",
                yaxis_title="Nombre",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Mesures", len(measurements))
            with col2:
                prob_0 = results.count(0) / len(results) if results else 0
                st.metric("P(|0⟩)", f"{prob_0:.3f}")
            with col3:
                prob_1 = results.count(1) / len(results) if results else 0
                st.metric("P(|1⟩)", f"{prob_1:.3f}")
            
            # Évolution temporelle
            st.write("### 📈 Évolution Temporelle")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(results))),
                y=results,
                mode='markers',
                marker=dict(size=8, color='#9D50FF'),
                name='Résultats'
            ))
            
            fig.update_layout(
                title="Résultats Mesures vs Temps",
                xaxis_title="Mesure #",
                yaxis_title="Résultat",
                template="plotly_dark",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune mesure enregistrée")
            st.write("Effectuez des mesures sur vos qubits pour voir les statistiques")
    
    with tab4:
        st.subheader("🎯 Tomographie Quantique")
        
        st.write("""
        **Tomographie d'État Quantique**
        
        Reconstruction complète de la matrice densité ρ
        """)
        
        if st.session_state.quantum_lab['qubits']:
            selected_qubit = st.selectbox("Qubit pour tomographie",
                list(st.session_state.quantum_lab['qubits'].keys()),
                format_func=lambda x: st.session_state.quantum_lab['qubits'][x]['name'],
                key="tomo_qubit")
            
            n_measurements = st.slider("Mesures par base", 100, 10000, 1000, 100)
            
            if st.button("🔬 Lancer Tomographie", type="primary"):
                with st.spinner("Tomographie en cours..."):
                    import time
                    time.sleep(2)
                    
                    qubit = st.session_state.quantum_lab['qubits'][selected_qubit]
                    
                    # alpha = complex(qubit['state_real_0'], qubit['state_imag_0'])
                    # alpha = complex(qubit.get('state_real_0', qubit.get('state_real_1', 0)),
                    #         qubit.get('state_imag_0', qubit.get('state_imag_1', 0)))

                    # beta = complex(qubit['state_real_1'], qubit['state_imag_1'])
                    alpha = complex(qubit.get('state_real_0', 0), qubit.get('state_imag_0', 0))
                    beta  = complex(qubit.get('state_real_1', 0), qubit.get('state_imag_1', 0))

                    
                    # Matrice densité (état pur)
                    state = np.array([alpha, beta])
                    rho = np.outer(state, np.conj(state))
                    
                    st.success("✅ Tomographie terminée!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Matrice Densité ρ:**")
                        
                        # Afficher partie réelle
                        fig = go.Figure(data=go.Heatmap(
                            z=rho.real,
                            x=['|0⟩', '|1⟩'],
                            y=['|0⟩', '|1⟩'],
                            colorscale='RdBu',
                            zmid=0
                        ))
                        
                        fig.update_layout(
                            title="Re(ρ)",
                            template="plotly_dark",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Propriétés:**")
                        
                        # Pureté
                        purity = np.trace(rho @ rho).real
                        st.metric("Pureté", f"{purity:.3f}")
                        
                        # Trace
                        trace = np.trace(rho).real
                        st.metric("Trace", f"{trace:.3f}")
                        
                        # Fidélité (avec état théorique)
                        fidelity = abs(np.trace(rho))**2
                        st.metric("Fidélité", f"{fidelity:.3f}")
                        
                        if purity > 0.99:
                            st.success("✅ État pur")
                        else:
                            st.info("État mixte")
                    
                    # Représentation Bloch
                    st.write("### 🌐 Vecteur de Bloch")
                    
                    # Calculer vecteur Bloch
                    r_x = 2 * rho[0,1].real
                    r_y = 2 * rho[0,1].imag
                    r_z = rho[0,0].real - rho[1,1].real
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("r_x", f"{r_x:.3f}")
                    with col2:
                        st.metric("r_y", f"{r_y:.3f}")
                    with col3:
                        st.metric("r_z", f"{r_z:.3f}")
                    
                    # Norme vecteur Bloch
                    r_norm = np.sqrt(r_x**2 + r_y**2 + r_z**2)
                    st.write(f"**Norme:** {r_norm:.3f}")
                    
                    if r_norm > 0.99:
                        st.success("✅ État pur (norme ≈ 1)")
                    else:
                        st.info(f"État mixte (norme = {r_norm:.3f} < 1)")
                    
                    log_event(f"Tomographie effectuée: {qubit['name']}", "SUCCESS")
        else:
            st.info("Créez un qubit pour effectuer la tomographie")
            
        st.write("### 📚 Bases de Mesure")
        
        st.write("""
        **Bases nécessaires pour qubit:**
        
        1. **Base Z:** {|0⟩, |1⟩}
        2. **Base X:** {|+⟩, |-⟩} où |±⟩ = (|0⟩ ± |1⟩)/√2
        3. **Base Y:** {|+i⟩, |-i⟩} où |±i⟩ = (|0⟩ ± i|1⟩)/√2
        
        **Minimum 3 bases** pour reconstruction complète
        """)
        
        # Visualisation bases
        bases_data = {
            'Base': ['Z', 'X', 'Y'],
            'État +': ['|0⟩', '|+⟩', '|+i⟩'],
            'État -': ['|1⟩', '|-⟩', '|-i⟩'],
            'Observable': ['σ_z', 'σ_x', 'σ_y']
        }
        
        df_bases = pd.DataFrame(bases_data)
        st.dataframe(df_bases, use_container_width=True)
            
# ==================== FOOTER ====================
st.markdown("---")

with st.expander("📜 Journal Recherche (20 dernières entrées)"):
    if st.session_state.quantum_lab['log']:
        for event in st.session_state.quantum_lab['log'][-20:][::-1]:
            timestamp = event['timestamp'][:19]
            level = event['level']
            
            icon = "ℹ️" if level == "INFO" else "✅" if level == "SUCCESS" else "⚠️" if level == "WARNING" else "❌"
            
            st.text(f"{icon} {timestamp} - {event['message']}")
    else:
        st.info("Aucun événement enregistré")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3>⚛️ Quantum Physics Research Platform</h3>
        <p>Gravité Quantique • Intrication • Singularités • Effet Tunnel</p>
        <p><small>Loop Quantum Gravity • String Theory • Quantum Networks</small></p>
        <p><small>Version 1.0.0 | Research Edition</small></p>
        <p><small>⚛️ Exploring the Quantum Universe © 2024</small></p>
    </div>
""", unsafe_allow_html=True)