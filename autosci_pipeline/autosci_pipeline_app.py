"""
AutoSciML Dashboard - Interface Streamlit
Interface de contrôle et visualisation des modèles ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
import zipfile
import io
import pickle
import joblib
from datetime import datetime, timedelta
import time

# Configuration Streamlit
st.set_page_config(
    page_title="AutoSciML Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration API
API_BASE_URL = "http://localhost:8011"
DEPLOYMENT_PLATFORM_URL = "http://localhost:8062"
MLFLOW_URL = "http://127.0.0.1:5000"

# État de session - CORRECTION: Initialisation complète
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 5
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Fonctions utilitaires - CORRECTION: Ajout de timeout
def call_api(endpoint: str, method: str = "GET", data: dict = None, timeout: int = 10):
    """Appel générique à AutoSciML avec gestion d'erreurs améliorée"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return None, f"Méthode {method} non supportée"
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Erreur {response.status_code}: {response.text}"
    except requests.exceptions.ConnectionError:
        return None, "Impossible de se connecter à AutoSciML. Vérifiez qu'elle est lancée."
    except requests.exceptions.Timeout:
        return None, "Timeout de connexion à AutoSciML"
    except Exception as e:
        return None, str(e)

def format_timestamp(timestamp_str):
    """Formate un timestamp pour l'affichage"""
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        return dt.strftime("%d/%m/%Y %H:%M:%S")
    except:
        return timestamp_str

def get_metrics_dataframe():
    """Récupère et formate les métriques en DataFrame"""
    metrics, error = call_api("/metrics")
    if error:
        return None
    
    if not metrics or not metrics.get("metrics"):
        return None
    
    rows = []
    for run in metrics["metrics"]:
        timestamp = run.get("timestamp", "")
        for framework, values in run.get("metrics", {}).items():
            if isinstance(values, dict) and "error" not in values:
                rows.append({
                    "Timestamp": format_timestamp(timestamp),
                    "Framework": framework,
                    "Accuracy": values.get("accuracy", 0),
                    "AUC": values.get("auc", 0),
                    "Domaine": run["config"].get("domaine", ""),
                    "Spécialité": run["config"].get("specialite", "")
                })
    
    if rows:
        return pd.DataFrame(rows)
    return None

# CORRECTION: Fonctions de gestion des modèles améliorées
def download_model(run_id):
    """Télécharge un modèle MLflow et le sauvegarde localement"""
    try:
        with st.spinner("Téléchargement en cours..."):
            response = requests.get(f"{API_BASE_URL}/models/{run_id}/download", timeout=60)
            
            if response.status_code == 200:
                download_dir = "downloaded_models"
                os.makedirs(download_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_{run_id[:8]}_{timestamp}.zip"
                filepath = os.path.join(download_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                st.success(f"Modèle téléchargé avec succès: {filename}")
                st.info(f"Emplacement: {filepath}")
                
                with st.expander("Instructions d'utilisation locale"):
                    st.code(f"""
# Python - Chargement du modèle
import zipfile
import joblib
import pickle
import os

# Extraire le modèle
with zipfile.ZipFile('{filepath}', 'r') as zip_ref:
    zip_ref.extractall('extracted_model/')

# Lister les fichiers extraits
model_files = os.listdir('extracted_model/')
print("Fichiers disponibles:", model_files)

# Charger le modèle (adapter selon le type)
try:
    # Pour scikit-learn/joblib
    model = joblib.load('extracted_model/model.pkl')
except:
    try:
        # Pour pickle
        with open('extracted_model/model.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        print("Format de modèle non reconnu")

# Utiliser le modèle
# predictions = model.predict(your_data)
                    """)
            else:
                st.error(f"Erreur lors du téléchargement: {response.status_code}")
                if response.status_code == 404:
                    st.warning("Le modèle n'existe pas ou n'est pas accessible")
                    
    except requests.exceptions.Timeout:
        st.error("Timeout lors du téléchargement. Le modèle est peut-être volumineux.")
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

def deploy_model(run_id, model_info):
    """Déploie un modèle sur la plateforme"""
    try:
        with st.spinner("Déploiement en cours..."):
            deployment_data = {
                "run_id": run_id,
                "model_name": f"Model_{run_id[:8]}",
                "framework": model_info.get('tags', {}).get('framework', 'unknown'),
                "model_type": model_info.get('tags', {}).get('model_type', 'unknown'),
                "metrics": model_info.get('metrics', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{DEPLOYMENT_PLATFORM_URL}/deploy", 
                json=deployment_data,
                timeout=30
            )
            
            if response.status_code == 200:
                deployment_info = response.json()
                st.success("Modèle déployé avec succès!")
                st.info(f"URL du modèle: {deployment_info.get('model_url', 'N/A')}")
                st.rerun()  # Actualiser pour afficher le bouton "Utiliser"
            else:
                st.error(f"Erreur lors du déploiement: {response.status_code}")
                st.error(f"Détails: {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter à la plateforme de déploiement")
    except Exception as e:
        st.error(f"Erreur de déploiement: {str(e)}")

def deploy_model_with_checks(run_id, model_info):
    """Déploie un modèle avec vérifications préalables"""
    # Vérifier les prérequis
    can_deploy, message = check_model_deployment_requirements(run_id)
    
    if not can_deploy:
        st.error(f"Impossible de déployer: {message}")
        return
    
    try:
        with st.spinner("Déploiement en cours..."):
            deployment_data = {
                "run_id": run_id,
                "model_name": f"Model_{run_id[:8]}",
                "framework": model_info.get('tags', {}).get('framework', 'unknown'),
                "model_type": model_info.get('tags', {}).get('model_type', 'unknown'),
                "metrics": model_info.get('metrics', {}),
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(
                f"{DEPLOYMENT_PLATFORM_URL}/deploy", 
                json=deployment_data,
                timeout=30
            )
            
            if response.status_code == 200:
                deployment_info = response.json()
                st.success("Modèle déployé avec succès!")
                st.info(f"URL du modèle: {deployment_info.get('model_url', 'N/A')}")
                
                # Vérifier le déploiement
                time.sleep(2)
                status = check_deployment_status(run_id)
                if status.get('is_deployed'):
                    st.success("Déploiement vérifié - Le modèle est prêt à utiliser")
                else:
                    st.warning("Déploiement en cours - Patientez quelques secondes")
                
                st.rerun()  # Actualiser pour afficher le bouton "Utiliser"
            else:
                st.error(f"Erreur lors du déploiement: {response.status_code}")
                st.error(f"Détails: {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter à la plateforme de déploiement")
        st.info("Vérifiez que le service deployment-platform est lancé sur le port 8002")
    except Exception as e:
        st.error(f"Erreur de déploiement: {str(e)}")
        
def check_deployment_status(run_id):
    """Vérifie si un modèle est déployé"""
    try:
        response = requests.get(f"{DEPLOYMENT_PLATFORM_URL}/status/{run_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"is_deployed": False}
    except:
        return {"is_deployed": False}

def check_model_deployment_requirements(run_id):
    """Vérifie que le modèle peut être déployé"""
    try:
        # Vérifier que le modèle existe dans MLflow
        mlflow_check = requests.get(f"{API_BASE_URL}/models/{run_id}", timeout=5)
        if mlflow_check.status_code != 200:
            return False, "Modèle non trouvé dans AutoSciML"
        
        # Vérifier que l'API de déploiement est accessible
        deploy_check = requests.get(f"{DEPLOYMENT_PLATFORM_URL}/health", timeout=5)
        if deploy_check.status_code != 200:
            return False, "Service de déploiement non accessible"
        
        return True, "Prêt pour déploiement"
        
    except Exception as e:
        return False, f"Erreur de vérification: {str(e)}"

def make_prediction(model_id, data):
    """Effectue une prédiction avec le modèle déployé"""
    try:
        # Vérifier d'abord si le modèle existe
        check_response = requests.get(f"{DEPLOYMENT_PLATFORM_URL}/model/{model_id}", timeout=10)
        if check_response.status_code != 200:
            return {"error": f"Modèle {model_id} non trouvé sur la plateforme de déploiement"}
        
        with st.spinner("Prédiction en cours..."):
            response = requests.post(
                f"{DEPLOYMENT_PLATFORM_URL}/predict/{model_id}",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 500:
                return {"error": f"Erreur interne du serveur - Le modèle {model_id} ne peut pas être chargé. Vérifiez qu'il est correctement déployé."}
            elif response.status_code == 422:
                return {"error": f"Format de données incorrect. Détails: {response.text}"}
            else:
                return {"error": f"Erreur de prédiction: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Erreur de connexion: {str(e)}"}


def display_prediction_result(result):
    """Affiche les résultats de prédiction avec gestion d'erreurs améliorée"""
    if "error" in result:
        st.error("Erreur lors de la prédiction")
        error_msg = result["error"]
        
        # Messages d'erreur spécifiques
        if "500" in error_msg:
            st.error("Le modèle ne peut pas être chargé. Possible causes:")
            st.write("- Le modèle n'est pas correctement déployé")
            st.write("- Le fichier de modèle est corrompu")
            st.write("- Problème de compatibilité des versions")
            st.info("Solution: Redéployez le modèle depuis l'onglet Modèles")
            
        elif "422" in error_msg:
            st.error("Format de données incorrect. Vérifiez:")
            st.write("- Le format des données d'entrée")
            st.write("- Les noms des colonnes/features")
            st.write("- Les types de données (numériques vs text)")
            
        elif "404" in error_msg:
            st.error("Modèle non trouvé sur la plateforme de déploiement")
            st.info("Solution: Redéployez le modèle depuis l'onglet Modèles")
            
        st.error(f"Détails techniques: {error_msg}")
        
    else:
        st.success("Prédiction effectuée avec succès!")
        
        if "prediction" in result:
            st.write("**Résultat:**")
            st.json(result["prediction"])
        
        if "confidence" in result:
            st.write("**Confiance:**")
            confidence_val = float(result["confidence"])
            st.progress(confidence_val)
            st.write(f"Niveau de confiance: {confidence_val:.2%}")
        
        if "probabilities" in result:
            st.write("**Probabilités:**")
            st.json(result["probabilities"])


def process_payment_simulation(model_id, purchase_type, price, email):
    """Simulation du traitement de paiement"""
    try:
        payment_data = {
            "model_id": model_id,
            "purchase_type": purchase_type,
            "price": price,
            "email": email,
            "timestamp": datetime.now().isoformat()
        }
        
        # Simuler la réponse d'un processeur de paiement
        return {
            "success": True,
            "transaction_id": f"txn_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "receipt_url": f"http://receipts.example.com/receipt123"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# CORRECTION: Interface d'utilisation d'un modèle déployé améliorée
def render_model_interface():
    """Interface d'utilisation d'un modèle déployé - Version corrigée"""
    if not st.session_state.current_model:
        st.error("Aucun modèle sélectionné")
        if st.button("← Retour aux modèles"):
            st.session_state.page = 'dashboard'
            st.rerun()
        return
    
    model_id = st.session_state.current_model
    st.title(f"Interface Modèle {model_id[:8]}...")
    
    try:
        response = requests.get(f"{DEPLOYMENT_PLATFORM_URL}/model/{model_id}", timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            
            # Affichage des informations du modèle
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Framework", model_info.get('framework', 'N/A'))
            with col2:
                st.metric("Type", model_info.get('model_type', 'N/A'))
            with col3:
                status_text = "Déployé" if model_info.get('status') == 'active' else "Inactif"
                st.metric("Status", status_text)
            
            st.divider()
            
            # Interface d'utilisation du modèle
            st.subheader("Utiliser le modèle")
            
            input_method = st.radio("Méthode d'entrée:", ["Texte", "Fichier CSV", "JSON"])
            
            if input_method == "Texte":
                user_input = st.text_area("Entrez vos données:", height=100, 
                                        help="Entrez les données que vous souhaitez analyser")
                if st.button("Prédire", type="primary"):
                    if user_input.strip():
                        # CORRECTION: Format correct pour l'API
                        prediction_result = make_prediction(model_id, {"input": user_input})
                        display_prediction_result(prediction_result)
                    else:
                        st.warning("Veuillez entrer des données")
            
            elif input_method == "Fichier CSV":
                uploaded_file = st.file_uploader("Choisir un fichier CSV", type=['csv'])
                if uploaded_file:
                    st.info(f"Fichier sélectionné: {uploaded_file.name}")
                    
                    # Aperçu du fichier
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.write("Aperçu des données:")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        # Options de traitement
                        col1, col2 = st.columns(2)
                        with col1:
                            process_method = st.selectbox(
                                "Comment traiter le fichier:",
                                ["Première ligne", "Toutes les lignes", "Résumé statistique"]
                            )
                        
                        with col2:
                            if len(df.columns) > 1:
                                target_column = st.selectbox(
                                    "Colonne à prédire (optionnel):",
                                    ["Aucune"] + list(df.columns)
                                )
                            else:
                                target_column = "Aucune"
                        
                        if st.button("Prédire", type="primary"):
                            # CORRECTION: Format correct pour l'API de prédiction
                            if process_method == "Première ligne":
                                # Utiliser la première ligne comme input
                                first_row = df.iloc[0].to_dict()
                                # Retirer la colonne target si spécifiée
                                if target_column != "Aucune" and target_column in first_row:
                                    del first_row[target_column]
                                
                                prediction_data = {"input": first_row}
                            
                            elif process_method == "Toutes les lignes":
                                # Traiter toutes les lignes
                                data_dict = df.to_dict('records')
                                # Limiter à 100 lignes pour éviter les timeouts
                                if len(data_dict) > 100:
                                    data_dict = data_dict[:100]
                                    st.warning("Limite à 100 lignes pour la prédiction")
                                
                                prediction_data = {"input": data_dict}
                            
                            else:  # Résumé statistique
                                # Utiliser les statistiques comme input
                                numeric_cols = df.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) > 0:
                                    stats = df[numeric_cols].describe().loc['mean'].to_dict()
                                    prediction_data = {"input": stats}
                                else:
                                    st.error("Aucune colonne numérique trouvée")
                                    return
                            
                            prediction_result = make_prediction(model_id, prediction_data)
                            display_prediction_result(prediction_result)
                            
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture du fichier: {e}")
            
            else:  # JSON
                json_input = st.text_area("Entrez les données JSON:", height=100,
                                        help="Format JSON valide requis. Exemple: {\"feature1\": 1.2, \"feature2\": 3.4}")
                
                # Exemple de format JSON
                st.info("Exemple de format JSON valide:")
                st.code('{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}')
                
                if st.button("Prédire", type="primary"):
                    if json_input.strip():
                        try:
                            data = json.loads(json_input)
                            # CORRECTION: S'assurer que les données sont dans le bon format
                            prediction_data = {"input": data}
                            prediction_result = make_prediction(model_id, prediction_data)
                            display_prediction_result(prediction_result)
                        except json.JSONDecodeError as e:
                            st.error(f"Format JSON invalide: {e}")
                    else:
                        st.warning("Veuillez entrer des données JSON")
            
            st.divider()
            
            # Options de monétisation (inchangées mais avec corrections mineures)
            st.subheader("Options d'accès")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Achat unique**")
                st.write("Téléchargez le modèle pour utilisation locale")
                price_buy = model_info.get('pricing', {}).get('buy_price', 99.99)
                st.write(f"Prix: **${price_buy}**")
                
                if st.button("Acheter maintenant", key="buy_model"):
                    # Formulaire de paiement dans un expander
                    with st.expander("Formulaire de paiement", expanded=True):
                        with st.form("payment_form_buy"):
                            st.write(f"**Type:** Achat unique")
                            st.write(f"**Prix:** ${price_buy}")
                            
                            email = st.text_input("Email*")
                            card_number = st.text_input("Numéro de carte*", type="password", 
                                                      placeholder="1234 5678 9012 3456")
                            col_exp, col_cvv = st.columns(2)
                            with col_exp:
                                exp_date = st.text_input("MM/YY*", placeholder="12/25")
                            with col_cvv:
                                cvv = st.text_input("CVV*", type="password", placeholder="123")
                            
                            if st.form_submit_button("Finaliser le paiement", type="primary"):
                                if all([email, card_number, exp_date, cvv]):
                                    payment_result = process_payment_simulation(model_id, "buy", price_buy, email)
                                    
                                    if payment_result["success"]:
                                        st.success("Paiement réussi!")
                                        st.success("Vous pouvez maintenant télécharger le modèle!")
                                        if st.button("Télécharger maintenant", type="primary"):
                                            download_model(model_id)
                                    else:
                                        st.error(f"Erreur de paiement: {payment_result['error']}")
                                else:
                                    st.error("Veuillez remplir tous les champs obligatoires")
            
            with col2:
                st.info("**Abonnement mensuel**")
                st.write("Utilisez le modèle en ligne sans limite")
                price_subscription = model_info.get('pricing', {}).get('subscription_price', 29.99)
                st.write(f"Prix: **${price_subscription}/mois**")
                
                if st.button("S'abonner", key="subscribe_model"):
                    # Formulaire d'abonnement similaire
                    with st.expander("Formulaire d'abonnement", expanded=True):
                        with st.form("payment_form_subscription"):
                            st.write(f"**Type:** Abonnement mensuel")
                            st.write(f"**Prix:** ${price_subscription}/mois")
                            
                            email = st.text_input("Email*")
                            card_number = st.text_input("Numéro de carte*", type="password")
                            col_exp, col_cvv = st.columns(2)
                            with col_exp:
                                exp_date = st.text_input("MM/YY*")
                            with col_cvv:
                                cvv = st.text_input("CVV*", type="password")
                            
                            if st.form_submit_button("Finaliser l'abonnement", type="primary"):
                                if all([email, card_number, exp_date, cvv]):
                                    payment_result = process_payment_simulation(model_id, "subscription", price_subscription, email)
                                    
                                    if payment_result["success"]:
                                        st.success("Paiement réussi!")
                                        st.success("Abonnement activé! Vous pouvez utiliser le modèle en ligne.")
                                        st.balloons()  # Animation de célébration
                                    else:
                                        st.error(f"Erreur de paiement: {payment_result['error']}")
                                else:
                                    st.error("Veuillez remplir tous les champs obligatoires")
            
        else:
            st.error("Modèle non trouvé ou non déployé")
            if response.status_code == 404:
                st.warning("Le modèle n'existe pas dans la plateforme de déploiement")
            
    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter à la plateforme de déploiement")
        st.info("Vérifiez que le service deployment-platform est lancé sur le port 8002")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
    
    # Bouton retour toujours visible
    st.divider()
    if st.button("← Retour aux modèles", type="secondary"):
        st.session_state.page = 'dashboard'
        st.session_state.current_model = None
        st.rerun()

# Interface principale - CORRECTION: Navigation améliorée
def main():
    # Header
    st.markdown('<h1 class="main-header">AutoSciML Dashboard</h1>', unsafe_allow_html=True)

    # Navigation conditionnelle
    if st.session_state.page == 'model_interface':
        render_model_interface()
        return

    # Sidebar pour la page principale
    with st.sidebar:
        st.header("Configuration")
        
        # État de l'API
        st.subheader("État de AutoSciML")
        status, error = call_api("/status")
        if error:
            st.error(error)
        else:
            if status:
                st.success("AutoSciML connectée")
                if status.get("status") == "running":
                    st.info("AutoSciML en cours...")
                elif status.get("status") == "completed":
                    st.success("Dernière exécution réussie")
                elif status.get("status") == "error":
                    st.error("Erreur lors de la dernière exécution")
        
        st.divider()
        
        # Configuration du pipeline
        st.subheader("Configuration AutoSciML")
        
        config, error = call_api("/config")
        if not error and config:
            with st.form("config_form"):
                domaine = st.text_input("Domaine", value=config.get("domaine", "demo"))
                specialite = st.text_input("Spécialité", value=config.get("specialite", "general"))
                freq_minutes = st.number_input(
                    "Fréquence (minutes)",
                    min_value=1,
                    max_value=1440,
                    value=config.get("freq_minutes", 60)
                )
                n_samples = st.number_input(
                    "Nombre d'échantillons",
                    min_value=100,
                    max_value=100000,
                    value=config.get("n_samples", 1000),
                    step=100
                )
                test_size = st.slider(
                    "Taille du test set",
                    min_value=0.1,
                    max_value=0.5,
                    value=config.get("test_size", 0.2),
                    step=0.05
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Sauvegarder", width="stretch"):
                        new_config = {
                            "domaine": domaine,
                            "specialite": specialite,
                            "freq_minutes": freq_minutes,
                            "n_samples": n_samples,
                            "test_size": test_size
                        }
                        result, error = call_api("/config", "POST", new_config)
                        if error:
                            st.error(error)
                        else:
                            st.success("Configuration mise à jour!")
                            st.rerun()
                
                with col2:
                    if st.form_submit_button("Lancer maintenant", width="stretch"):
                        result, error = call_api("/run", "POST")
                        if error:
                            st.error(error)
                        else:
                            st.success("AutoSciML lancé!")
        
        st.divider()
        
        # Auto-refresh
        st.subheader("Actualisation")
        st.session_state.auto_refresh = st.toggle("Auto-refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider(
                "Intervalle (secondes)",
                min_value=2,
                max_value=60,
                value=5
            )
            st.info(f"Actualisation toutes les {st.session_state.refresh_interval}s")
        
        if st.button("Actualiser maintenant", width="stretch"):
            st.rerun()

    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Vue d'ensemble",
        "Métriques", 
        "Modèles",
        "Données",
        "MLflow"
    ])

    # Tab 3: Modèles - CORRECTION: Interface améliorée
    with tab3:
        st.header("Gestion des Modèles")
        
        # Indicateur de statut des services
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Status des services:")
        with col2:
            # Test rapide de connectivité
            api_status = "🟢" if call_api("/models")[1] is None else "🔴"
            deploy_status = "🟢" if requests.get(f"{DEPLOYMENT_PLATFORM_URL}/models", timeout=2).status_code == 200 else "🔴"
            st.write(f"AutoSciML: {api_status} | Deploy: {deploy_status}")
        
        models_data, error = call_api("/models")
        
        if error:
            st.error(f"Erreur AutoSciML: {error}")
            st.info("Vérifiez que AutoSciML AutoSci est lancée sur le port 8000")
        # elif models_data and models_data.get("models"):
        # models = models_data["models"]
        elif models_data:
            models = models_data.get("models", models_data)
            
            # Statistiques améliorées
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Modèles", len(models))
            with col2:
                frameworks = set(m.get("tags", {}).get("framework", "unknown") for m in models)
                st.metric("Frameworks", len(frameworks))
            with col3:
                successful = sum(1 for m in models if m.get("status") == "FINISHED")
                st.metric("Réussis", successful)
            with col4:
                # Compter les modèles déployés
                deployed_count = 0
                for model in models[:5]:  # Test sur les premiers pour éviter timeout
                    if check_deployment_status(model['run_id']).get('is_deployed'):
                        deployed_count += 1
                st.metric("Déployés", deployed_count)
            
            st.divider()
            
            # Filtres
            col1, col2 = st.columns(2)
            with col1:
                framework_filter = st.selectbox(
                    "Filtrer par framework:",
                    options=["Tous"] + list(frameworks)
                )
            with col2:
                status_filter = st.selectbox(
                    "Filtrer par status:",
                    options=["Tous", "FINISHED", "FAILED", "RUNNING"]
                )
            
            # Filtrage des modèles
            filtered_models = models
            if framework_filter != "Tous":
                filtered_models = [m for m in filtered_models 
                                 if m.get("tags", {}).get("framework") == framework_filter]
            if status_filter != "Tous":
                filtered_models = [m for m in filtered_models 
                                 if m.get("status") == status_filter]
            
            st.write(f"Affichage de {len(filtered_models)} modèle(s)")
            
            # Liste des modèles avec interface améliorée
            for i, model in enumerate(filtered_models[:10]):
                with st.expander(
                    f"🤖 Modèle {model['run_id'][:8]}... - {model.get('tags', {}).get('framework', 'N/A')}",
                    expanded=(i == 0)  # Premier modèle ouvert par défaut
                ):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Informations**")
                        st.write(f"- Run ID: `{model['run_id']}`")
                        st.write(f"- Framework: {model.get('tags', {}).get('framework', 'N/A')}")
                        st.write(f"- Type: {model.get('tags', {}).get('model_type', 'N/A')}")
                        st.write(f"- Status: {model.get('status', 'N/A')}")
                    
                    with col2:
                        st.write("**Métriques**")
                        metrics = model.get('metrics', {})
                        if metrics:
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    st.write(f"- {key}: {value:.4f}")
                        else:
                            st.write("Aucune métrique disponible")
                    
                    # Vérification du statut de déploiement
                    deployment_status = check_deployment_status(model['run_id'])
                    is_deployed = deployment_status.get('is_deployed', False)
                    
                    # Boutons d'action avec statut visuel
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("Détails", key=f"details_{model['run_id']}", width="stretch"):
                            st.info(f"Ouvrez MLflow UI: {MLFLOW_URL}")
                    
                    with col2:
                        if st.button("Télécharger", key=f"download_{model['run_id']}", width="stretch", type="secondary"):
                            download_model(model['run_id'])
                    
                    with col3:
                        deploy_label = "Redéployer" if is_deployed else "Déployer"
                        if st.button(deploy_label, key=f"deploy_{model['run_id']}", width="stretch", type="primary"):
                            deploy_model(model['run_id'], model)

                    with col4:
                        if is_deployed:
                            if st.button("Utiliser", key=f"use_{model['run_id']}", width="stretch", type="primary"):
                                st.session_state.current_model = model['run_id']
                                st.session_state.page = 'model_interface'
                                st.rerun()
                        else:
                            st.button(
                                "Non déployé",
                                disabled=True,
                                width="stretch",
                                key=f"non_deploye_{model['run_id']}"
                            )
                    
                    # Indicateur de statut
                    if is_deployed:
                        st.success("Modèle déployé et prêt à utiliser")
                    else:
                        st.info("Modèle non déployé - cliquez sur 'Déployer' pour l'activer")
        else:
            st.info("Aucun modèle disponible")
            st.write("Pour voir des modèles ici:")
            st.write("1. Assurez-vous que AutoSci est lancée")
            st.write("2. Lancez l'entraînement depuis la sidebar")
            st.write("3. Attendez que les modèles soient créés")

    # Autres tabs (Vue d'ensemble, Métriques, Données, MLflow)
    with tab1:
        st.header("Vue d'ensemble")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_df = get_metrics_dataframe()
        
        if metrics_df is not None and not metrics_df.empty:
            with col1:
                best_accuracy = metrics_df["Accuracy"].max()
                st.metric(
                    "Meilleure Accuracy",
                    f"{best_accuracy:.3f}",
                    delta=f"+{best_accuracy - metrics_df['Accuracy'].mean():.3f}"
                )
            
            with col2:
                best_auc = metrics_df["AUC"].max()
                st.metric(
                    "Meilleur AUC", 
                    f"{best_auc:.3f}",
                    delta=f"+{best_auc - metrics_df['AUC'].mean():.3f}"
                )
            
            with col3:
                total_runs = len(metrics_df["Timestamp"].unique())
                st.metric("Nombre d'exécutions", total_runs)
            
            with col4:
                total_models = len(metrics_df)
                st.metric("Modèles entraînés", total_models)
                
            st.divider()
            
            # Graphiques de performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Évolution de l'Accuracy")
                fig = px.line(
                    metrics_df,
                    x="Timestamp",
                    y="Accuracy",
                    color="Framework",
                    markers=True,
                    title="Accuracy par Framework"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Distribution AUC")
                fig = px.box(
                    metrics_df,
                    x="Framework",
                    y="AUC",
                    color="Framework",
                    title="Distribution AUC par Framework"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune métrique disponible. Lancez AutoSciML pour commencer.")

    with tab2:
        st.header("Analyse détaillée des Métriques")
        
        if metrics_df is not None and not metrics_df.empty:
            # Filtres
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_frameworks = st.multiselect(
                    "Frameworks",
                    options=metrics_df["Framework"].unique().tolist(),
                    default=metrics_df["Framework"].unique().tolist()
                )
            
            with col2:
                selected_domaine = st.selectbox(
                    "Domaine",
                    options=["Tous"] + metrics_df["Domaine"].unique().tolist(),
                    index=0
                )
            
            with col3:
                metric_type = st.selectbox(
                    "Métrique",
                    options=["Accuracy", "AUC", "Les deux"],
                    index=2
                )
            
            # Filtrage
            filtered_df = metrics_df[metrics_df["Framework"].isin(selected_frameworks)]
            if selected_domaine != "Tous":
                filtered_df = filtered_df[filtered_df["Domaine"] == selected_domaine]
            
            # Graphique temporel
            if metric_type == "Accuracy":
                fig = px.line(filtered_df, x="Timestamp", y="Accuracy", color="Framework", markers=True)
            elif metric_type == "AUC":
                fig = px.line(filtered_df, x="Timestamp", y="AUC", color="Framework", markers=True)
            else:
                # Les deux métriques
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Accuracy", "AUC"), shared_xaxes=True)
                
                for framework in selected_frameworks:
                    framework_data = filtered_df[filtered_df["Framework"] == framework]
                    fig.add_trace(
                        go.Scatter(x=framework_data["Timestamp"], y=framework_data["Accuracy"],
                                 mode='lines+markers', name=framework, legendgroup=framework),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=framework_data["Timestamp"], y=framework_data["AUC"],
                                 mode='lines+markers', name=framework, showlegend=False, legendgroup=framework),
                        row=2, col=1
                    )
                
                fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau détaillé
            st.subheader("Données détaillées")
            st.dataframe(filtered_df.sort_values("Timestamp", ascending=False), width="stretch")
            
            # Export
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

                #         # Comparaison des frameworks
            st.subheader(" Comparaison des Frameworks")
            
            comparison_df = metrics_df.groupby("Framework").agg({
                "Accuracy": ["mean", "std", "max"],
                "AUC": ["mean", "std", "max"]
            }).round(4)
            
            comparison_df.columns = [
                "Accuracy Moyenne", "Accuracy Std", "Accuracy Max",
                "AUC Moyen", "AUC Std", "AUC Max"
            ]
            
            st.dataframe(
                comparison_df.style.highlight_max(axis=0, color='lightgreen'),
                width="stretch"
            )

        else:
            st.info("Aucune métrique à afficher")

        

    with tab4:
        st.header("Gestion des Données")
        
        data_dir = "donnees_propres"
        if os.path.exists(data_dir):
            files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')], reverse=True)
            
            if files:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Fichiers disponibles")
                    selected_file = st.selectbox("Sélectionner un fichier", options=files[:20])
                
                with col2:
                    st.subheader("Statistiques")
                    st.metric("Nombre de fichiers", len(files))
                    total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in files) / (1024 * 1024)
                    st.metric("Taille totale", f"{total_size:.2f} MB")
                
                if selected_file:
                    st.divider()
                    file_path = os.path.join(data_dir, selected_file)
                    
                    try:
                        df = pd.read_csv(file_path)
                        st.subheader(f"Aperçu: {selected_file}")
                        
                        # Info sur le dataset
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Lignes", df.shape[0])
                        with col2:
                            st.metric("Colonnes", df.shape[1])
                        with col3:
                            st.metric("Valeurs manquantes", df.isnull().sum().sum())
                        with col4:
                            size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            st.metric("Taille", f"{size_mb:.2f} MB")
                        
                        # Aperçu des données
                        st.dataframe(df.head(100), width="stretch")
                        
                        # Téléchargement
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Télécharger {selected_file}",
                            data=csv,
                            file_name=selected_file,
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la lecture du fichier: {e}")
            else:
                st.info("Aucun fichier de données disponible")
        else:
            st.warning(f"Le dossier {data_dir} n'existe pas")

    with tab5:
        st.header("Interface MLflow")
        
        st.info(f"MLflow UI disponible à: {MLFLOW_URL}")
        
        st.markdown(f"""
        <div class="info-box">
            <h3>Accès rapide MLflow</h3>
            <p>Cliquez sur le bouton ci-dessous pour ouvrir MLflow dans un nouvel onglet</p>
            <a href="{MLFLOW_URL}" target="_blank">
                <button style="
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                ">
                    Ouvrir MLflow UI
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

    # Auto-refresh conditionnel
    if st.session_state.auto_refresh and st.session_state.page == 'dashboard':
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
