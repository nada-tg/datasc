# personal_data_dashboard.py - Interface Streamlit pour la plateforme de données personnelles

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import uuid
import math

# Configuration de la page    streamlit run data_platform_app.py
st.set_page_config(
    page_title="Personal Data Intelligence Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration API
PERSONAL_DATA_API_URL = "http://localhost:8022"
AUTOSCI_DASHBOARD_URL = "http://localhost:8501"

# Style CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .privacy-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .consent-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .data-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# État de session
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Fonctions utilitaires
def call_personal_data_api(endpoint, method="GET", data=None):
    """Appel API pour la plateforme de données personnelles"""
    url = f"{PERSONAL_DATA_API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur {response.status_code}: {response.text}"
    except Exception as e:
        return None, str(e)

def record_consent(user_id, data_type, consent_given):
    """Enregistre un consentement utilisateur"""
    consent_data = {
        "user_id": user_id,
        "data_type": data_type,
        "consent_given": consent_given,
        "timestamp": datetime.now().isoformat()
    }
    return call_personal_data_api("/consent/record", "POST", consent_data)

def show_home_page_corrected():
    """Page d'accueil avec navigation corrigée"""
    st.title("Bienvenue sur votre Plateforme de Données Personnelles")
    
    st.markdown("""
    <div class="privacy-card">
        <h3>Votre vie privée, votre contrôle</h3>
        <p>Cette plateforme vous permet de collecter, analyser et exploiter vos propres données 
        en gardant un contrôle total sur leur utilisation. Toutes les opérations nécessitent 
        votre consentement explicite.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques rapides
    col1, col2, col3, col4 = st.columns(4)
    
    # Récupérer les données utilisateur
    files_data, _ = call_personal_data_api(f"/data/files/{st.session_state.user_id}")
    analyses_data, _ = call_personal_data_api(f"/data/analyses/{st.session_state.user_id}")
    studies_data, _ = call_personal_data_api(f"/data/studies/{st.session_state.user_id}")
    
    with col1:
        file_count = len(files_data.get("files", [])) if files_data else 0
        st.metric("Fichiers Collectés", file_count)
    
    with col2:
        analysis_count = len(analyses_data.get("analyses", [])) if analyses_data else 0
        st.metric("Analyses Effectuées", analysis_count)
    
    with col3:
        study_count = len(studies_data.get("studies", [])) if studies_data else 0
        st.metric("Études Complètes", study_count)
    
    with col4:
        # TODO: Calculer les revenus des ventes
        st.metric("Revenus Générés", "$0.00")
    
    st.divider()
    
    # Actions rapides avec navigation fonctionnelle
    st.subheader("Actions Rapides")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Nouvelle Collecte", use_container_width=True, type="primary"):
            st.session_state.current_section = "Collecte de Données"
            st.rerun()
    
    with col2:
        if st.button("Voir Mes Fichiers", use_container_width=True):
            st.session_state.current_section = "Mes Fichiers"
            st.rerun()
    
    with col3:
        if st.button("Marketplace", use_container_width=True):
            st.session_state.current_section = "Marketplace"
            st.rerun()
    
    # Dernières activités
    st.divider()
    st.subheader("Activité Récente")
    
    # Afficher les dernières analyses/études
    recent_activities = []
    
    if analyses_data and analyses_data.get("analyses"):
        for analysis in analyses_data["analyses"][:3]:
            recent_activities.append({
                "type": "Analyse",
                "name": analysis["file_name"],
                "date": analysis["created_at"],
                "status": analysis["status"]
            })
    
    if studies_data and studies_data.get("studies"):
        for study in studies_data["studies"][:3]:
            recent_activities.append({
                "type": "Étude",
                "name": study["file_name"],
                "date": study["created_at"],
                "status": study["status"]
            })
    
    if recent_activities:
        # Trier par date
        recent_activities.sort(key=lambda x: x["date"], reverse=True)
        
        for activity in recent_activities[:5]:
            status_color = {"completed": "🟢", "in_progress": "🟡", "failed": "🔴"}.get(activity["status"], "⚪")
            st.write(f"{status_color} **{activity['type']}** - {activity['name']} - {activity['date']}")
    else:
        st.info("Aucune activité récente. Commencez par collecter des données!")

# 3. MARKETPLACE AMÉLIORÉE

def show_marketplace_page_improved():
    """Page du marketplace améliorée"""
    st.title("Marketplace de Données et Modèles IA")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Acheter Données", "Acheter Modèles", "Mes Ventes", "Mes Achats"])
    
    with tab1:
        st.subheader("Données Disponibles à l'Achat")
        
        # Filtres avancés
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            price_filter = st.selectbox("Prix:", ["Tous", "< $10", "$10-50", "$50-100", "> $100"])
        with col2:
            type_filter = st.selectbox("Type:", ["Tous", "Système", "Réseau", "Fichiers", "Analysé"])
        with col3:
            license_filter = st.selectbox("Licence:", ["Toutes", "Usage unique", "Commercial", "Open"])
        with col4:
            sort_by = st.selectbox("Trier par:", ["Plus récent", "Prix croissant", "Prix décroissant", "Popularité"])
        
        # Récupérer les offres disponibles
        sales_data, error = call_personal_data_api("/marketplace/sales")
        
        if error:
            st.error(f"Erreur lors du chargement: {error}")
        else:
            sales = sales_data.get("available_sales", [])
            
            if not sales:
                st.info("Aucune offre de données disponible actuellement.")
                st.write("Soyez le premier à vendre vos données analysées!")
            else:
                # Affichage amélioré des offres
                for sale in sales:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            st.markdown(f"### {sale.get('title', sale['description'][:50]+'...')}")
                            st.write(f"**Type:** {sale['data_type']}")
                            st.write(f"**Taille:** {sale['size_bytes']} bytes")
                            st.write(f"**Publié:** {sale['created_at']}")
                        
                        with col2:
                            st.write(f"**Description:**")
                            st.write(sale['description'])
                            
                            # Tags
                            tags = ["Anonymisé", "Qualité vérifiée", "Support inclus"]
                            for tag in tags:
                                st.badge(tag, type="secondary")
                        
                        with col3:
                            st.markdown(f"### ${sale['price']}")
                            if st.button(f"Acheter", key=f"buy_{sale['sale_id']}", type="primary"):
                                show_purchase_form(sale)
                            
                            st.button("Aperçu", key=f"preview_{sale['sale_id']}")
                        
                        st.divider()
    
    with tab2:
        st.subheader("Modèles IA Disponibles")
        
        # Simuler des modèles disponibles
        sample_models = [
            {
                "id": "model_001",
                "name": "Prédicteur de Performance Système",
                "description": "Modèle entraîné sur 10000+ échantillons de données système",
                "accuracy": 0.95,
                "price": 299.99,
                "type": "Classification",
                "features": ["CPU Usage", "Memory", "Network", "Disk I/O"]
            },
            {
                "id": "model_002", 
                "name": "Analyseur de Trafic Réseau",
                "description": "Détection d'anomalies dans le trafic réseau",
                "accuracy": 0.91,
                "price": 450.00,
                "type": "Détection d'anomalies",
                "features": ["Packets/sec", "Bytes transferrés", "Connexions actives"]
            }
        ]
        
        for model in sample_models:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"### {model['name']}")
                    st.write(f"**Type:** {model['type']}")
                    st.write(f"**Précision:** {model['accuracy']:.1%}")
                    st.write(f"**Features:** {', '.join(model['features'][:2])}...")
                
                with col2:
                    st.write(model['description'])
                    st.progress(model['accuracy'])
                
                with col3:
                    st.markdown(f"### ${model['price']}")
                    st.button("Acheter Modèle", key=f"buy_model_{model['id']}", type="primary")
                    st.button("Démo", key=f"demo_{model['id']}")
                
                st.divider()
    
    with tab3:
        st.subheader("Mes Offres de Vente")
        
        # TODO: Récupérer les ventes de l'utilisateur
        st.info("Fonctionnalité en cours de développement")
        st.write("Utilisez les boutons 'Vendre' sur vos analyses terminées pour créer des offres.")
        
        # Statistiques de vente simulées
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Offres Actives", 0)
        with col2:
            st.metric("Ventes Réalisées", 0)
        with col3:
            st.metric("Revenus Totaux", "$0.00")
        with col4:
            st.metric("Note Moyenne", "N/A")
    
    with tab4:
        st.subheader("Mes Achats")
        st.info("Fonctionnalité en cours de développement")
        st.write("Vos achats de données et modèles apparaîtront ici.")

def show_purchase_form(sale_item):
    """Affiche le formulaire d'achat"""
    with st.form(f"purchase_form_{sale_item['sale_id']}"):
        st.subheader(f"Achat: {sale_item.get('title', 'Données')}")
        
        st.write(f"**Prix:** ${sale_item['price']}")
        st.write(f"**Description:** {sale_item['description']}")
        
        # Informations de paiement simulées
        st.write("**Informations de paiement**")
        email = st.text_input("Email:")
        payment_method = st.selectbox("Méthode de paiement:", ["Carte de crédit", "PayPal", "Crypto"])
        
        agree_terms = st.checkbox("J'accepte les conditions d'utilisation")
        
        if st.form_submit_button("Confirmer l'Achat", type="primary", disabled=not agree_terms):
            if agree_terms:
                st.success("Achat simulé réussi !")
                st.info("Fonctionnalité de paiement réel à implémenter")
            else:
                st.error("Veuillez accepter les conditions d'utilisation")


# Interface principale
def main():
    st.markdown('<h1 class="main-header">Personal Data Intelligence Platform</h1>', unsafe_allow_html=True)
    
    # Navigation corrigée
    if st.session_state.current_page == 'autosci':
        redirect_to_autosci()
        return
    
    # Sidebar avec navigation fonctionnelle
    with st.sidebar:
        st.header("Navigation")
        
        # Bouton vers AutoSciML
        if st.button("AutoSciML Dashboard", use_container_width=True, type="primary"):
            st.session_state.current_page = 'autosci'
            st.rerun()
        
        st.divider()
        
        # Menu de navigation avec gestion d'état
        page = st.selectbox(
            "Sélectionnez une section:",
            ["Accueil", "Collecte de Données", "Mes Fichiers", "Analyses", "Études", "Marketplace", "Consentements"],
            key="page_selector"
        )
        
        # Mettre à jour l'état de session
        st.session_state.current_section = page
        
        st.divider()
        
        # Informations utilisateur
        st.subheader("Mon Compte")
        st.write(f"ID Utilisateur: `{st.session_state.user_id[:8]}...`")
        
        # Statut de la plateforme
        health, error = call_personal_data_api("/health")
        if health:
            st.success("Plateforme en ligne")
        else:
            st.error("Plateforme hors ligne")
    
    # Navigation vers les pages avec gestion des boutons
    if page == "Accueil":
        show_home_page_corrected()
    elif page == "Collecte de Données":
        show_data_collection_page()
    elif page == "Mes Fichiers":
        show_files_page()
    elif page == "Analyses":
        show_analyses_page()
    elif page == "Études":
        show_studies_page()
    elif page == "Marketplace":
        show_marketplace_page_improved()
    elif page == "Consentements":
        show_consent_page()


def redirect_to_autosci():
    """Page de redirection vers AutoSciML"""
    st.title("Redirection vers AutoSciML Dashboard")
    
    st.info("Vous allez être redirigé vers le dashboard AutoSciML...")
    
    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0;">
        <a href="{AUTOSCI_DASHBOARD_URL}" target="_blank">
            <button style="
                background-color: #007bff;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 18px;
                text-decoration: none;
            ">
                Ouvrir AutoSciML Dashboard
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Retour à Personal Data Platform", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()
    
    with col2:
        st.button("Actualiser", use_container_width=True)

def show_home_page():
    """Page d'accueil"""
    st.title("Bienvenue sur votre Plateforme de Données Personnelles")
    
    st.markdown("""
    <div class="privacy-card">
        <h3> Votre vie privée, votre contrôle</h3>
        <p>Cette plateforme vous permet de collecter, analyser et exploiter vos propres données 
        en gardant un contrôle total sur leur utilisation. Toutes les opérations nécessitent 
        votre consentement explicite.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques rapides
    col1, col2, col3, col4 = st.columns(4)
    
    # Récupérer les données utilisateur
    files_data, _ = call_personal_data_api(f"/data/files/{st.session_state.user_id}")
    analyses_data, _ = call_personal_data_api(f"/data/analyses/{st.session_state.user_id}")
    studies_data, _ = call_personal_data_api(f"/data/studies/{st.session_state.user_id}")
    
    with col1:
        file_count = len(files_data.get("files", [])) if files_data else 0
        st.metric("Fichiers Collectés", file_count)
    
    with col2:
        analysis_count = len(analyses_data.get("analyses", [])) if analyses_data else 0
        st.metric("Analyses Effectuées", analysis_count)
    
    with col3:
        study_count = len(studies_data.get("studies", [])) if studies_data else 0
        st.metric("Études Complètes", study_count)
    
    with col4:
        # TODO: Calculer les revenus des ventes
        st.metric("Revenus Générés", "$0.00")
    
    st.divider()
    
    # Actions rapides
    st.subheader("Actions Rapides")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Nouvelle Collecte", use_container_width=True, type="primary"):
            st.session_state.show_collection_form = True
            st.rerun()
    
    with col2:
        if st.button("Voir Mes Fichiers", use_container_width=True):
            # Navigation programmée vers la page fichiers
            pass
    
    with col3:
        if st.button("Marketplace", use_container_width=True):
            # Navigation programmée vers le marketplace
            pass

# Version corrigée de show_data_collection_page()
# Ajoutez cette fonction au début de show_data_collection_page() pour diagnostiquer
def show_data_collection_page_with_debug():
    """Version avec diagnostic complet"""
    st.title("Collecte de Données Personnelles")
    
    # Test de connectivité API
    st.subheader("Diagnostic du Système")
    with st.expander("Test de Connectivité", expanded=False):
        if st.button("Tester l'API"):
            test_api_connection()

def show_data_collection_page():
    """Page de collecte de données - Version corrigée"""
    st.title("Collecte de Données Personnelles")
    
    # Avertissement de confidentialité
    st.markdown("""
    <div class="consent-box">
        <h4> Informations Importantes</h4>
        <ul>
            <li>Seules les métadonnées système non-sensibles sont collectées</li>
            <li>Aucun contenu privé (mots de passe, messages, etc.) n'est collecté</li>
            <li>Toutes les données restent sur votre appareil</li>
            <li>Vous gardez un contrôle total sur l'utilisation de vos données</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Formulaire de collecte
    # Consentements requis
    st.divider()
    st.subheader("Consentements Requis")
        
    consent_collection = st.checkbox(
            "Je consens à la collecte des données sélectionnées selon les paramètres choisis", 
            value=False
        )
    consent_processing = st.checkbox(
            "Je consens au traitement automatisé de mes données pour l'analyse", 
            value=False
        )
    consent_storage = st.checkbox(
            "Je consens au stockage sécurisé de mes données sur cet appareil", 
            value=False
        )
    with st.form("data_collection_form"):
        st.subheader("Configuration de la Collecte")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Types de données à collecter:**")
            collect_network = st.checkbox("Données réseau (connexions actives, statistiques)", value=True)
            collect_system = st.checkbox("Données système (utilisation CPU/RAM, plateforme)", value=True)
            collect_files = st.checkbox("Métadonnées de fichiers (taille, type, dates)", value=False)
        
        with col2:
            st.write("**Paramètres de confidentialité:**")
            privacy_level = st.selectbox("Niveau de confidentialité:", ["Élevé", "Moyen", "Basique"])
            duration = st.slider("Durée de collecte (heures):", 1, 24, 1)
            include_sensitive = st.checkbox("Inclure des données système détaillées", value=False)
        
        
        
        # Vérifications des conditions
        any_data_selected = any([collect_network, collect_system, collect_files])
        all_consents_given = all([consent_collection, consent_processing, consent_storage])
        
        # Afficher l'état des conditions
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if any_data_selected:
                st.success(" Types de données sélectionnés")
            else:
                st.error(" Aucun type de données sélectionné")
        
        with col2:
            if all_consents_given:
                st.success(" Tous les consentements accordés")
            else:
                st.error(" Consentements requis manquants")
        
        # Messages d'aide
        if not any_data_selected:
            st.warning(" Veuillez sélectionner au moins un type de données à collecter")
        
        if not all_consents_given:
            st.warning(" Tous les consentements sont requis pour démarrer la collecte")
        
        # Debug info (optionnel - à supprimer en production)
        with st.expander("Debug Info", expanded=False):
            st.write(f"Data selected: {any_data_selected}")
            st.write(f"Network: {collect_network}")
            st.write(f"System: {collect_system}")
            st.write(f"Files: {collect_files}")
            st.write(f"All consents: {all_consents_given}")
            st.write(f"Consent 1: {consent_collection}")
            st.write(f"Consent 2: {consent_processing}")
            st.write(f"Consent 3: {consent_storage}")
        
        # Bouton avec logique corrigée
        button_enabled = any_data_selected and all_consents_given
        
        # Message explicatif si bouton désactivé
        if not button_enabled:
            st.info("Le bouton sera activé une fois que vous aurez sélectionné des données et accordé tous les consentements.")
        
        # Form submit button (dans le form)
        submitted = st.form_submit_button(" Démarrer la Collecte", use_container_width=True)

        # Bloquer l'exécution si conditions non remplies
        # if submitted and not button_enabled:
        #     st.error("Impossible de démarrer : conditions non remplies")
        #     submitted = False  # Empêche la suite du code de s'exécuter

        # submitted = st.form_submit_button(
        #     "🚀 Démarrer la Collecte", 
        #     type="primary" if button_enabled else "secondary",
        #     disabled=not button_enabled,
        #     use_container_width=True
        # )
        
        # Traitement lors de la soumission
        if submitted:
            if not button_enabled:
                st.error("Impossible de démarrer : conditions non remplies")
                return
            
            # Préparer la configuration
            collection_types = []
            if collect_network:
                collection_types.append("network")
            if collect_system:
                collection_types.append("system")
            if collect_files:
                collection_types.append("files")
            
            config = {
                "user_id": st.session_state.user_id,
                "collection_types": collection_types,
                "consent_timestamp": datetime.now().isoformat(),
                "duration_hours": duration,
                "privacy_level": privacy_level.lower(),
                "include_sensitive": include_sensitive
            }
            
            # Afficher la configuration pour debug
            st.write("Configuration de collecte:")
            st.json(config)
            
            # Enregistrer les consentements
            for data_type in collection_types:
                consent_result, consent_error = record_consent(st.session_state.user_id, data_type, True)
                if consent_error:
                    st.warning(f"Erreur consentement {data_type}: {consent_error}")
            
            # Démarrer la collecte
            with st.spinner("Démarrage de la collecte..."):
                result, error = call_personal_data_api("/data/collect", "POST", config)
            
            if result:
                st.success(f" Collecte démarrée avec succès !")
                st.info(f"ID de collecte: {result['collection_id']}")
                st.info("La collecte s'exécute en arrière-plan. Vous pouvez suivre le progrès dans la section 'Mes Fichiers'.")
                
                # Optionnel: Redirection automatique
                # if st.button("Voir mes fichiers"):
                #     st.rerun()
            else:
                st.error(f" Erreur lors du démarrage de la collecte:")
                st.error(f"Détails: {error}")
                st.info("Vérifiez que l'API Personal Data Platform est accessible sur le port 8003")

    if submitted:
        st.info("Collecte démarrée !")
    if st.button("Voir mes fichiers"):
        st.session_state.current_page = "Mes Fichiers"
        st.rerun()
# Fonction de test pour vérifier la connectivité
def test_api_connection():
    """Teste la connexion à l'API"""
    health, error = call_personal_data_api("/health")
    if health:
        st.success(" API Personal Data Platform connectée")
        return True
    else:
        st.error(f" API non accessible: {error}")
        st.info("Démarrez l'API avec: uvicorn personal_data_platform:app --port 8003")
        return False

def show_file_details(file_info):
    """Affiche les détails complets d’un fichier"""
    st.subheader(f"Détails du fichier : {file_info['file_name']}")
    st.write(f"- ID: {file_info['file_id']}")
    st.write(f"- Type de données: {file_info['data_type']}")
    st.write(f"- Source: {file_info['source_type']}")
    st.write(f"- Taille: {file_info['size_bytes']} bytes")
    st.write(f"- Créé le: {file_info['created_at']}")
    if file_info.get("metadata"):
        st.write("**Métadonnées**")
        for k, v in file_info["metadata"].items():
            st.write(f"- {k}: {v}")

def show_files_page():
    """Page des fichiers collectés"""
    st.title("Mes Fichiers Collectés")
    
    # Récupérer les fichiers
    files_data, error = call_personal_data_api(f"/data/files/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur lors du chargement: {error}")
        return
    
    files = files_data.get("files", [])
    
    if not files:
        st.info("Aucun fichier collecté. Démarrez une collecte de données pour commencer.")
        return
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_type_filter = st.selectbox("Filtrer par type:", ["Tous"] + list(set([f["data_type"] for f in files])))
    
    with col2:
        source_filter = st.selectbox("Filtrer par source:", ["Toutes"] + list(set([f["source_type"] for f in files])))
    
    with col3:
        sort_by = st.selectbox("Trier par:", ["Date (récent)", "Date (ancien)", "Taille", "Nom"])
    
    # Filtrage et tri
    filtered_files = files
    if data_type_filter != "Tous":
        filtered_files = [f for f in filtered_files if f["data_type"] == data_type_filter]
    if source_filter != "Toutes":
        filtered_files = [f for f in filtered_files if f["source_type"] == source_filter]
    
    # Tri
    if sort_by == "Date (récent)":
        filtered_files.sort(key=lambda x: x["created_at"], reverse=True)
    elif sort_by == "Date (ancien)":
        filtered_files.sort(key=lambda x: x["created_at"])
    elif sort_by == "Taille":
        filtered_files.sort(key=lambda x: x["size_bytes"], reverse=True)
    elif sort_by == "Nom":
        filtered_files.sort(key=lambda x: x["file_name"])
    
    st.write(f"Affichage de {len(filtered_files)} fichier(s)")
    
    # Liste des fichiers
    for file_info in filtered_files:
        with st.expander(f"{file_info['file_name']} ({file_info['data_type']})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Informations du fichier:**")
                st.write(f"- ID: `{file_info['file_id'][:16]}...`")
                st.write(f"- Type: {file_info['data_type']}")
                st.write(f"- Source: {file_info['source_type']}")
                st.write(f"- Taille: {file_info['size_bytes']} bytes")
                st.write(f"- Créé: {file_info['created_at']}")
            
            with col2:
                st.write("**Métadonnées:**")
                if file_info.get("metadata"):
                    metadata = file_info["metadata"]
                    for key, value in metadata.items():
                        st.write(f"- {key}: {value}")
            
            # Boutons d'action
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Analyser", key=f"analyze_{file_info['file_id']}", use_container_width=True):
                    start_analysis(file_info['file_id'], "descriptive")
            
            with col2:
                if st.button("Étudier", key=f"study_{file_info['file_id']}", use_container_width=True):
                    start_study(file_info['file_id'])
            
            with col3:
                if st.button("Visualiser", key=f"view_{file_info['file_id']}", use_container_width=True):
                    show_file_details(file_info)

def start_analysis(file_id, analysis_type):
    """Démarre une analyse de données"""
    analysis_config = {
        "file_id": file_id,
        "analysis_type": analysis_type,
        "user_id": st.session_state.user_id
    }
    
    result, error = call_personal_data_api("/data/analyze", "POST", analysis_config)
    
    if result:
        st.success(f"Analyse démarrée ! ID: {result['analysis_id']}")
        st.info("L'analyse s'exécute en arrière-plan. Consultez la section 'Analyses' pour voir les résultats.")
    else:
        st.error(f"Erreur lors du démarrage de l'analyse: {error}")

def start_study(file_id):
    """Démarre une étude data science"""
    study_config = {
        "file_id": file_id,
        "target_variable": None,  # Auto-détection
        "problem_type": "auto",
        "user_id": st.session_state.user_id
    }
    
    result, error = call_personal_data_api("/data/study", "POST", study_config)
    
    if result:
        st.success(f"Étude démarrée ! ID: {result['study_id']}")
        st.info("L'étude s'exécute en arrière-plan. Consultez la section 'Études' pour voir les résultats.")
    else:
        st.error(f"Erreur lors du démarrage de l'étude: {error}")



# Fonctions manquantes pour personal_data_dashboard.py

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

# Variables globales de navigation
if 'current_section' not in st.session_state:
    st.session_state.current_section = 'Accueil'

# 1. FONCTIONS MANQUANTES

def show_train_form(file_id):
    """Affiche le formulaire d'entraînement d'IA"""
    st.subheader("Entraîner un Modèle IA")
    
    with st.form(f"train_form_{file_id}"):
        st.write("Configurez l'entraînement de votre modèle IA personnalisé")
        
        model_type = st.selectbox(
            "Type de modèle:",
            ["Classification Automatique", "Régression Automatique", "Analyse Exploratoire", "Clustering"]
        )
        
        target_column = st.text_input(
            "Colonne cible (optionnel):",
            placeholder="Nom de la variable à prédire"
        )
        
        model_name = st.text_input(
            "Nom du modèle:",
            value=f"MonModele_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        advanced_options = st.expander("Options avancées")
        with advanced_options:
            test_size = st.slider("Taille du jeu de test:", 0.1, 0.5, 0.2)
            cross_validation = st.checkbox("Validation croisée", value=True)
            feature_selection = st.checkbox("Sélection automatique des features", value=True)
        
        if st.form_submit_button("Lancer l'Entraînement", type="primary"):
            study_config = {
                "file_id": file_id,
                "target_variable": target_column if target_column else None,
                "problem_type": model_type.lower().split()[0],  # classification, regression, etc.
                "user_id": st.session_state.user_id,
                "model_name": model_name,
                "config": {
                    "test_size": test_size,
                    "cross_validation": cross_validation,
                    "feature_selection": feature_selection
                }
            }
            
            result, error = call_personal_data_api("/data/study", "POST", study_config)
            
            if result:
                st.success(f"Entraînement démarré ! ID: {result['study_id']}")
                st.info("L'entraînement s'exécute en arrière-plan. Consultez la section 'Études' pour voir les résultats.")
                st.balloons()
            else:
                st.error(f"Erreur lors du lancement: {error}")

def show_sell_form(file_id):
    """Affiche le formulaire de vente de données analysées"""
    st.subheader("Vendre vos Données Analysées")
    
    with st.form(f"sell_data_form_{file_id}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Informations de vente**")
            price = st.number_input("Prix ($):", min_value=1.0, max_value=10000.0, value=25.0, step=5.0)
            title = st.text_input("Titre de l'offre:", placeholder="ex: Données d'analyse système optimisées")
            description = st.text_area(
                "Description détaillée:", 
                placeholder="Décrivez vos données: type, qualité, utilité potentielle...",
                height=100
            )
        
        with col2:
            st.write("**Paramètres de confidentialité**")
            anonymization = st.selectbox("Niveau d'anonymisation:", ["Élevé", "Moyen", "Basique"])
            license_type = st.selectbox("Type de licence:", [
                "Usage unique", "Usage multiple (5x)", "Usage commercial", "Licence ouverte"
            ])
            data_format = st.selectbox("Format de livraison:", ["CSV", "JSON", "Excel", "Tous formats"])
        
        # Prévisualisation du prix selon la licence
        multiplier = {"Usage unique": 1, "Usage multiple (5x)": 0.8, "Usage commercial": 1.5, "Licence ouverte": 0.6}
        final_price = price * multiplier.get(license_type, 1)
        st.info(f"Prix final: ${final_price:.2f}")
        
        # Options additionnelles
        st.write("**Options additionnelles**")
        include_analysis = st.checkbox("Inclure les résultats d'analyse", value=True)
        include_visualizations = st.checkbox("Inclure les visualisations", value=True)
        support_included = st.checkbox("Support technique inclus (30 jours)", value=False)
        
        if st.form_submit_button("Créer l'Offre de Vente", type="primary"):
            offer_data = {
                "file_id": file_id,
                "price": final_price,
                "title": title,
                "description": description,
                "anonymization_level": anonymization.lower(),
                "license_type": license_type.lower(),
                "data_format": data_format.lower(),
                "user_id": st.session_state.user_id,
                "options": {
                    "include_analysis": include_analysis,
                    "include_visualizations": include_visualizations,
                    "support_included": support_included
                }
            }
            
            result, error = call_personal_data_api("/data/sell", "POST", offer_data)
            
            if result:
                st.success("Offre créée avec succès !")
                st.info(f"ID de vente: {result['sale_id']}")
                st.info("Votre offre apparaîtra dans le marketplace sous 24h après vérification.")
            else:
                st.error(f"Erreur lors de la création: {error}")

def show_sell_model_form(study_id):
    """Affiche le formulaire de vente de modèle entraîné"""
    st.subheader("Vendre votre Modèle IA")
    
    with st.form(f"sell_model_form_{study_id}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Informations du modèle**")
            model_price = st.number_input("Prix ($):", min_value=10.0, max_value=50000.0, value=100.0, step=10.0)
            model_title = st.text_input("Nom du modèle:", placeholder="ex: Modèle de Prédiction Personnalisé")
            model_description = st.text_area(
                "Description du modèle:",
                placeholder="Performance, cas d'usage, données d'entraînement...",
                height=100
            )
        
        with col2:
            st.write("**Licence et distribution**")
            model_license = st.selectbox("Type de licence:", [
                "Usage personnel", "Usage commercial", "Licence académique", "Open source"
            ])
            deployment_support = st.checkbox("Support de déploiement inclus", value=True)
            source_code_included = st.checkbox("Code source inclus", value=False)
            training_data_included = st.checkbox("Données d'entraînement incluses", value=False)
        
        st.write("**Garanties et support**")
        performance_guarantee = st.checkbox("Garantie de performance", value=True)
        support_duration = st.selectbox("Durée du support:", ["30 jours", "90 jours", "1 an", "Aucun support"])
        
        if st.form_submit_button("Mettre le Modèle en Vente", type="primary"):
            model_offer = {
                "study_id": study_id,
                "price": model_price,
                "title": model_title,
                "description": model_description,
                "license_type": model_license,
                "user_id": st.session_state.user_id,
                "options": {
                    "deployment_support": deployment_support,
                    "source_code_included": source_code_included,
                    "training_data_included": training_data_included,
                    "performance_guarantee": performance_guarantee,
                    "support_duration": support_duration
                }
            }
            
            # API call pour vendre le modèle
            result, error = call_personal_data_api("/models/sell", "POST", model_offer)
            
            if result:
                st.success("Modèle mis en vente avec succès !")
                st.balloons()
            else:
                st.error(f"Erreur: {error}")

def deploy_model(study_id):
    """Déploie un modèle étudié sur la plateforme"""
    st.subheader(f"Déploiement du Modèle {study_id[:8]}...")
    
    with st.spinner("Déploiement en cours..."):
        # Simuler le déploiement
        import time
        time.sleep(2)
        
        deployment_config = {
            "study_id": study_id,
            "user_id": st.session_state.user_id,
            "deployment_type": "cloud",
            "auto_scaling": True
        }
        
        # API call pour déployer
        result, error = call_personal_data_api("/models/deploy", "POST", deployment_config)
        
        if result:
            st.success("Modèle déployé avec succès !")
            st.info(f"URL du modèle: {result.get('model_url', 'URL non disponible')}")
            st.info("Votre modèle est maintenant accessible via API.")
            
            # Afficher les informations de déploiement
            with st.expander("Informations de déploiement"):
                st.json(result)
        else:
            st.error(f"Erreur de déploiement: {error}")
            st.info("Le déploiement sera disponible dans une version future.")

def download_model(study_id):
    """Télécharge un modèle étudié"""
    st.subheader(f"Téléchargement du Modèle {study_id[:8]}...")
    
    with st.spinner("Préparation du téléchargement..."):
        # Simuler la préparation
        import time
        time.sleep(1)
        
        download_config = {
            "study_id": study_id,
            "user_id": st.session_state.user_id,
            "format": "zip",
            "include_data": True
        }
        
        # API call pour télécharger
        result, error = call_personal_data_api("/models/download", "POST", download_config)
        
        if result:
            st.success("Modèle préparé pour téléchargement !")
            
            # Instructions de téléchargement
            st.info("Instructions d'utilisation du modèle téléchargé:")
            st.code("""
# Python - Utilisation du modèle téléchargé
import pickle
import pandas as pd

# Charger le modèle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger les données de test
data = pd.read_csv('test_data.csv')

# Faire des prédictions
predictions = model.predict(data)
print(predictions)
            """)
            
            # Bouton de téléchargement simulé
            st.download_button(
                label="Télécharger le Modèle (ZIP)",
                data="Modèle simulé - contenu du fichier ZIP",  # En réalité, ce serait le contenu du modèle
                file_name=f"model_{study_id[:8]}.zip",
                mime="application/zip"
            )
        else:
            st.error(f"Erreur de téléchargement: {error}")

def show_analyses_page():
    """Page des analyses effectuées"""
    st.title("Mes Analyses de Données")
    
    # Récupérer les analyses
    analyses_data, error = call_personal_data_api(f"/data/analyses/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur lors du chargement: {error}")
        return
    
    analyses = analyses_data.get("analyses", [])
    
    if not analyses:
        st.info("Aucune analyse effectuée. Utilisez le bouton 'Analyser' sur vos fichiers collectés.")
        return
    
    # Afficher les analyses
    for analysis in analyses:
        status_color = {"completed": "🟢", "in_progress": "🟡", "failed": "🔴"}.get(analysis["status"], "⚪")
        
        with st.expander(f"{status_color} Analyse {analysis['file_name']} - {analysis['analysis_type']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Informations:**")
                st.write(f"- ID: `{analysis['analysis_id'][:16]}...`")
                st.write(f"- Type: {analysis['analysis_type']}")
                st.write(f"- Statut: {analysis['status']}")
                st.write(f"- Créée: {analysis['created_at']}")
            
            if analysis["status"] == "completed" and analysis.get("results"):
                st.divider()
                st.subheader("Résultats de l'Analyse")
                
                # Afficher les résultats
                results = analysis["results"]
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key == "summary_stats" and value:
                            st.write("**Statistiques Descriptives:**")
                            df_stats = pd.DataFrame(value)
                            st.dataframe(df_stats, use_container_width=True)
                        elif key == "correlation_analysis" and value:
                            st.write("**Analyse de Corrélation:**")
                            st.json(value)
                        else:
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # Visualisations
                if analysis.get("visualizations"):
                    st.subheader("Visualisations")
                    visualizations = analysis["visualizations"]
                    
                    for viz_name, viz_data in visualizations.items():
                        try:
                            fig = go.Figure(json.loads(viz_data))
                            st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.write(f"Erreur d'affichage pour {viz_name}")
                
                # Actions sur les résultats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Vendre Données", key=f"sell_{analysis['analysis_id']}", use_container_width=True):
                        show_sell_form(analysis['file_id'])
                
                with col2:
                    if st.button("Entraîner IA", key=f"train_{analysis['analysis_id']}", use_container_width=True):
                        show_train_form(analysis['file_id'])
                
                with col3:
                    if st.button("Faire Don", key=f"donate_{analysis['analysis_id']}", use_container_width=True):
                        show_donation_form(analysis['file_id'])

def sanitize_floats(obj):
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None  # ou 0, selon ton besoin
        else:
            return obj
    else:
        return obj
    

def show_studies_page():
    """Page des études data science"""
    st.title("Mes Études Data Science")
    
    # Récupérer les études
    studies_data, error = call_personal_data_api(f"/data/studies/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur lors du chargement: {error}")
        return
    
    studies = studies_data.get("studies", [])
    
    if not studies:
        st.info("Aucune étude effectuée. Utilisez le bouton 'Étudier' sur vos fichiers collectés.")
        return
    
    # Afficher les études
    for study in studies:
        status_color = {"completed": "🟢", "in_progress": "🟡", "failed": "🔴"}.get(study["status"], "⚪")
        
        with st.expander(f"{status_color} Étude {study['file_name']} - {study['problem_type']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Informations:**")
                st.write(f"- ID: `{study['study_id'][:16]}...`")
                st.write(f"- Type de problème: {study['problem_type']}")
                st.write(f"- Statut: {study['status']}")
                st.write(f"- Créée: {study['created_at']}")
            
            if study["status"] == "completed" and study.get("model_performance"):
                st.divider()
                st.subheader("Résultats de l'Étude")
                
                # Performance du modèle
                performance = study["model_performance"]
                
                if performance.get("problem_type") == "classification":
                    st.metric("Accuracy", f"{performance.get('accuracy', 0):.4f}")
                    
                    if performance.get("classification_report"):
                        st.write("**Rapport de Classification:**")
                        report_df = pd.DataFrame(performance["classification_report"]).transpose()
                        st.dataframe(report_df, use_container_width=True)
                
                elif performance.get("problem_type") == "regression":
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("RMSE", f"{performance.get('rmse', 0):.4f}")
                    with col2:
                        st.metric("R² Score", f"{performance.get('r2_score', 0):.4f}")
                
                # Importance des features
                if performance.get("feature_importance"):
                    st.write("**Importance des Variables:**")
                    importance_df = pd.DataFrame(list(performance["feature_importance"].items()), 
                                               columns=["Feature", "Importance"])
                    importance_df = importance_df.sort_values("Importance", ascending=False)
                    
                    fig = px.bar(importance_df.head(10), x="Importance", y="Feature", 
                               orientation="h", title="Top 10 Features Importantes")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Vendre Modèle", key=f"sell_model_{study['study_id']}", use_container_width=True):
                        show_sell_model_form(study['study_id'])
                
                with col2:
                    if st.button("Déployer", key=f"deploy_{study['study_id']}", use_container_width=True):
                        deploy_model(study['study_id'])
                
                with col3:
                    if st.button("Télécharger", key=f"download_{study['study_id']}", use_container_width=True):
                        download_model(study['study_id'])

def show_marketplace_page():
    """Page du marketplace"""
    st.title("Marketplace de Données")
    
    tab1, tab2 = st.tabs(["Acheter des Données", "Mes Ventes"])
    
    with tab1:
        # Récupérer les offres disponibles
        sales_data, error = call_personal_data_api("/marketplace/sales")
        
        if error:
            st.error(f"Erreur lors du chargement: {error}")
            return
        
        sales = sales_data.get("available_sales", [])
        
        if not sales:
            st.info("Aucune offre de données disponible actuellement.")
            return
        
        for sale in sales:
            with st.container():
                st.markdown(f"""
                <div class="data-card">
                    <h4>{sale['description']}</h4>
                    <p><strong>Type:</strong> {sale['data_type']}</p>
                    <p><strong>Taille:</strong> {sale['size_bytes']} bytes</p>
                    <p><strong>Prix:</strong> ${sale['price']}</p>
                    <p><strong>Publié:</strong> {sale['created_at']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Acheter ${sale['price']}", key=f"buy_{sale['sale_id']}"):
                    st.info("Fonctionnalité d'achat à implémenter")
    
    with tab2:
        st.subheader("Mes Offres de Vente")
        st.info("Utilisez les boutons 'Vendre' sur vos analyses terminées pour créer des offres.")

def show_consent_page():
    """Page de gestion des consentements"""
    st.title("Gestion de mes Consentements")
    
    # Récupérer les consentements
    consents_data, error = call_personal_data_api(f"/consent/{st.session_state.user_id}")
    
    if error:
        st.error(f"Erreur lors du chargement: {error}")
        return
    
    consents = consents_data.get("consents", [])
    
    if not consents:
        st.info("Aucun consentement enregistré.")
        return
    
    # Afficher les consentements
    st.subheader("Historique des Consentements")
    
    consent_df = pd.DataFrame(consents)
    consent_df["status"] = consent_df["consent_given"].apply(lambda x: "✅ Accordé" if x else "❌ Refusé")
    
    st.dataframe(
        consent_df[["data_type", "status", "timestamp", "expires_at"]].rename(columns={
            "data_type": "Type de Données",
            "status": "Statut",
            "timestamp": "Date",
            "expires_at": "Expire le"
        }),
        use_container_width=True
    )
    
    # Révoquer des consentements
    st.divider()
    st.subheader("Révoquer un Consentement")
    
    active_consents = [c for c in consents if c["consent_given"] and not c["revoked_at"]]
    
    if active_consents:
        consent_to_revoke = st.selectbox(
            "Sélectionnez le consentement à révoquer:",
            options=[c["data_type"] for c in active_consents]
        )
        
        if st.button("Révoquer ce Consentement", type="secondary"):
            st.warning(f"Consentement pour '{consent_to_revoke}' révoqué (fonctionnalité à implémenter)")
    else:
        st.info("Aucun consentement actif à révoquer.")

# Fonctions utilitaires pour les actions
def show_sell_form(file_id):
    """Affiche le formulaire de vente"""
    with st.form(f"sell_form_{file_id}"):
        st.subheader("Vendre vos Données")
        
        price = st.number_input("Prix ($):", min_value=1.0, value=10.0, step=1.0)
        description = st.text_area("Description:", placeholder="Décrivez vos données...")
        anonymization = st.selectbox("Niveau d'anonymisation:", ["Élevé", "Moyen", "Basique"])
        license_type = st.selectbox("Type de licence:", ["Usage unique", "Usage multiple", "Commercial"])
        
        if st.form_submit_button("Créer l'Offre"):
            offer_data = {
                "file_id": file_id,
                "price": price,
                "description": description,
                "anonymization_level": anonymization.lower(),
                "license_type": license_type.lower(),
                "user_id": st.session_state.user_id
            }
            
            result, error = call_personal_data_api("/data/sell", "POST", offer_data)
            
            if result:
                st.success("Offre créée avec succès !")
            else:
                st.error(f"Erreur: {error}")

def show_donation_form(file_id):
    """Affiche le formulaire de don"""
    with st.form(f"donate_form_{file_id}"):
        st.subheader("Faire un Don de Données")
        
        organization = st.text_input("Organisation bénéficiaire:", placeholder="ex: Recherche médicale, ONG...")
        purpose = st.text_area("Objectif du don:", placeholder="À quoi vos données vont-elles servir?")
        anonymization = st.selectbox("Niveau d'anonymisation:", ["Élevé", "Moyen", "Basique"])
        
        if st.form_submit_button("Faire le Don"):
            donation_data = {
                "file_id": file_id,
                "recipient_organization": organization,
                "purpose": purpose,
                "anonymization_level": anonymization.lower(),
                "user_id": st.session_state.user_id
            }
            
            result, error = call_personal_data_api("/data/donate", "POST", donation_data)
            
            if result:
                st.success("Don enregistré avec succès !")
            else:
                st.error(f"Erreur: {error}")

if __name__ == "__main__":
    main()

