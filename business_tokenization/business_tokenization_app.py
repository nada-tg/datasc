"""
business_tokenization_frontend.py - Interface Streamlit

Lancement:
streamlit run business_tokenization_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Business Tokenization Platform",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8014"

# CSS Simple
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def init_session():
    if 'current_company' not in st.session_state:
        st.session_state.current_company = None
    if 'investor_id' not in st.session_state:
        st.session_state.investor_id = "investor_001"

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# PAGE: Dashboard
def page_dashboard():
    st.title("Plateforme de Tokenisation d'Entreprises")
    st.write("Créez, valorisez et tokenisez votre entreprise")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/statistics/platform")
        if response.status_code == 200:
            stats = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entreprises", stats['total_companies'])
            with col2:
                st.metric("Tokens Émis", stats['total_tokens'])
            with col3:
                st.metric("Capitalisation", f"${stats['total_market_cap']:,.0f}")
            with col4:
                st.metric("Investisseurs", stats['total_investors'])
            
            st.write("---")
            
            # Graphique par industrie
            if stats['by_industry']:
                st.subheader("Entreprises par Industrie")
                df = pd.DataFrame(list(stats['by_industry'].items()), columns=['Industrie', 'Nombre'])
                fig = px.bar(df, x='Industrie', y='Nombre')
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
    
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Créer une Entreprise", type="primary"):
            st.session_state.active_view = 'create_company'
            st.rerun()
    
    with col2:
        if st.button("Marketplace de Tokens"):
            st.session_state.active_view = 'marketplace'
            st.rerun()

# PAGE: Créer Entreprise
def page_create_company():
    st.title("Créer et Valoriser votre Entreprise")
    
    with st.form("company_form"):
        st.subheader("Informations Générales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Nom de l'Entreprise *")
            company_type = st.selectbox("Type", ["startup", "sme", "corporation", "nonprofit"])
            status = st.selectbox("Statut", ["new", "existing"])
            industry = st.selectbox("Industrie", [
                "technology", "finance", "healthcare", "retail",
                "manufacturing", "services", "energy", "real_estate"
            ])
        
        with col2:
            founded_year = st.number_input("Année de Fondation", 1900, 2025, 2020) if status == "existing" else None
            description = st.text_area("Description")
        
        st.write("---")
        st.subheader("Informations Financières (USD)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            annual_revenue = st.number_input("Revenus Annuels", 0.0, step=10000.0)
        with col2:
            annual_profit = st.number_input("Bénéfices Annuels", step=10000.0)
        with col3:
            total_assets = st.number_input("Actifs Totaux", 0.0, step=10000.0)
        with col4:
            total_liabilities = st.number_input("Passifs Totaux", 0.0, step=10000.0)
        
        st.write("---")
        st.subheader("Équipe et Organisation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            employees_count = st.number_input("Nombre d'Employés", 0, step=1)
        with col2:
            founders_count = st.number_input("Nombre de Fondateurs", 1, step=1)
        with col3:
            management_experience = st.number_input("Expérience Moyenne (années)", 0, step=1)
        
        st.write("---")
        st.subheader("Marché et Clients")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            market_size = st.number_input("Taille du Marché (USD)", 0.0, step=1000000.0)
        with col2:
            market_share = st.number_input("Part de Marché (%)", 0.0, 100.0, step=0.1)
        with col3:
            growth_rate = st.number_input("Taux de Croissance (%)", -100.0, 1000.0, step=1.0)
        with col4:
            customers_count = st.number_input("Nombre de Clients", 0, step=1)
        
        st.write("---")
        st.subheader("Produits et Innovation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            products_count = st.number_input("Nombre de Produits", 0, step=1)
        with col2:
            retention_rate = st.number_input("Taux de Rétention (%)", 0.0, 100.0, step=0.1)
        with col3:
            patents_count = st.number_input("Nombre de Brevets", 0, step=1)
        with col4:
            rd_investment = st.number_input("Investissement R&D (USD)", 0.0, step=10000.0)
        
        tech_score = st.slider("Score Technologique", 0, 100, 50)
        
        st.write("---")
        st.subheader("Tokenisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            shares_to_tokenize = st.number_input("Nombre d'Actions à Tokeniser", 1, step=1000, value=1000000)
        with col2:
            st.info("Le prix par action sera calculé automatiquement par notre IA")
        
        submitted = st.form_submit_button("Créer et Valoriser", type="primary")
        
        if submitted:
            if not name or not description:
                st.error("Veuillez remplir tous les champs obligatoires")
            else:
                with st.spinner("Analyse et valorisation en cours..."):
                    payload = {
                        "name": name,
                        "type": company_type,
                        "status": status,
                        "industry": industry,
                        "founded_year": founded_year,
                        "description": description,
                        "annual_revenue": annual_revenue,
                        "annual_profit": annual_profit,
                        "total_assets": total_assets,
                        "total_liabilities": total_liabilities,
                        "employees_count": employees_count,
                        "founders_count": founders_count,
                        "management_experience": management_experience,
                        "market_size": market_size,
                        "market_share": market_share,
                        "growth_rate": growth_rate,
                        "products_count": products_count,
                        "customers_count": customers_count,
                        "retention_rate": retention_rate,
                        "patents_count": patents_count,
                        "rd_investment": rd_investment,
                        "tech_score": tech_score,
                        "shares_to_tokenize": shares_to_tokenize
                    }
                    
                    try:
                        response = requests.post(f"{API_URL}/api/v1/companies/create", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Entreprise créée et valorisée avec succès!")
                            
                            st.session_state.current_company = result['company_id']
                            st.session_state.active_view = 'company_details'
                            st.rerun()
                        else:
                            st.error(f"Erreur: {response.text}")
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")

# PAGE: Détails Entreprise
def page_company_details():
    company_id = st.session_state.current_company
    
    if not company_id:
        st.warning("Aucune entreprise sélectionnée")
        return
    
    try:
        response = requests.get(f"{API_URL}/api/v1/companies/{company_id}")
        
        if response.status_code == 200:
            data = response.json()
            company = data['company']
            valuation = data['valuation']
            predictions = data['predictions']
            events = data['events']
            evolution = data['evolution']
            
            st.title(f"{company['name']}")
            st.caption(f"{company['industry'].upper()} • {company['type'].upper()}")
            
            # Métriques principales
            st.subheader("Valorisation")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Valorisation Totale", f"${valuation['total_valuation']:,.2f}")
            with col2:
                st.metric("Prix par Action", f"${valuation['share_price']:.2f}")
            with col3:
                st.metric("Score de Confiance", f"{valuation['confidence_score']}%")
            with col4:
                st.metric("Niveau de Risque", valuation['risk_level'])
            
            st.write("---")
            
            # Évolution du prix
            st.subheader("Évolution du Prix")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prix Initial", f"${evolution['initial_price']:.2f}")
            with col2:
                st.metric("Prix Actuel", f"${evolution['current_price']:.2f}")
            with col3:
                change = evolution['change_percentage']
                st.metric("Variation", f"{change:+.2f}%", delta=f"{change:.2f}%")
            
            st.write("---")
            
            # Méthodes de valorisation
            st.subheader("Détail des Valorisations")
            
            methods_df = pd.DataFrame([
                {"Méthode": "DCF", "Valorisation": valuation['valuation_per_method']['dcf']},
                {"Méthode": "Actifs", "Valorisation": valuation['valuation_per_method']['asset_based']},
                {"Méthode": "Multiple CA", "Valorisation": valuation['valuation_per_method']['revenue_multiple']}
            ])
            
            fig = px.bar(methods_df, x='Méthode', y='Valorisation', title="Valorisation par Méthode")
            st.plotly_chart(fig, use_container_width=True)
            
            # Métriques financières
            st.write("---")
            st.subheader("Métriques Financières")
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = valuation['financial_metrics']
            
            with col1:
                st.metric("Marge Bénéficiaire", f"{metrics['profit_margin']:.2f}%")
            with col2:
                st.metric("ROE", f"{metrics['roe']:.2f}%")
            with col3:
                st.metric("Rotation Actifs", f"{metrics['asset_turnover']:.2f}")
            with col4:
                st.metric("Dette/Capitaux", f"{metrics['debt_to_equity']:.2f}")
            
            # Prédictions
            st.write("---")
            st.subheader("Prédictions d'Événements")
            
            if predictions:
                for pred in predictions:
                    with st.expander(f"{pred['title']} - Probabilité: {pred['probability']}%"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type:** {pred['event_type']}")
                            st.write(f"**Date estimée:** {pred['expected_date']}")
                            st.write(f"**Impact:** {pred['impact']}")
                        
                        with col2:
                            st.progress(pred['probability'] / 100)
                            st.write(pred['description'])
            else:
                st.info("Aucune prédiction disponible")
            
            # Événements historiques
            st.write("---")
            st.subheader("Événements Majeurs")
            
            if events:
                events_df = pd.DataFrame(events)
                st.dataframe(events_df, use_container_width=True)
            else:
                st.info("Aucun événement enregistré")
            
            # Ajouter un événement
            with st.expander("Ajouter un Événement"):
                with st.form("event_form"):
                    event_type = st.selectbox("Type", ["funding", "product_launch", "acquisition", "partnership", "expansion", "crisis"])
                    event_title = st.text_input("Titre")
                    event_desc = st.text_area("Description")
                    impact = st.slider("Score d'Impact", -100, 100, 0)
                    event_date = st.date_input("Date")
                    
                    if st.form_submit_button("Ajouter"):
                        event_payload = {
                            "company_id": company_id,
                            "event_type": event_type,
                            "title": event_title,
                            "description": event_desc,
                            "impact_score": impact,
                            "date": event_date.isoformat()
                        }
                        
                        try:
                            resp = requests.post(f"{API_URL}/api/v1/events/create", json=event_payload)
                            if resp.status_code == 200:
                                st.success("Événement ajouté!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Marketplace
def page_marketplace():
    st.title("Marketplace de Tokens")
    
    tab1, tab2 = st.tabs(["Tokens Disponibles", "Actions"])
    
    with tab1:
        try:
            response = requests.get(f"{API_URL}/api/v1/marketplace/tokens")
            
            if response.status_code == 200:
                data = response.json()
                tokens = data['tokens']
                
                if not tokens:
                    st.info("Aucun token disponible")
                    return
                
                # Filtres
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    industries = list(set(t['company_industry'] for t in tokens))
                    industry_filter = st.selectbox("Industrie", ["Toutes"] + industries)
                
                with col2:
                    risk_filter = st.selectbox("Risque", ["Tous", "LOW", "MEDIUM", "HIGH"])
                
                with col3:
                    growth_filter = st.selectbox("Potentiel", ["Tous", "LOW", "MEDIUM", "HIGH"])
                
                # Appliquer filtres
                filtered = tokens
                if industry_filter != "Toutes":
                    filtered = [t for t in filtered if t['company_industry'] == industry_filter]
                if risk_filter != "Tous":
                    filtered = [t for t in filtered if t['risk_level'] == risk_filter]
                if growth_filter != "Tous":
                    filtered = [t for t in filtered if t['growth_potential'] == growth_filter]
                
                st.write(f"**{len(filtered)} token(s) trouvé(s)**")
                st.write("---")
                
                # Afficher tokens
                for token in filtered:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                        
                        with col1:
                            st.write(f"**{token['company_name']}**")
                            st.caption(f"{token['company_industry']}")
                        
                        with col2:
                            st.metric("Prix", f"${token['price_per_token']:.2f}")
                        
                        with col3:
                            st.write(f"Disponible: {token['available_supply']:,}")
                            st.write(f"Risque: {token['risk_level']}")
                        
                        with col4:
                            if st.button("Acheter", key=f"buy_{token['token_id']}"):
                                st.session_state.selected_token = token['token_id']
                                st.session_state.show_purchase = True
                        
                        st.write("---")
                
                # Modal d'achat
                if st.session_state.get('show_purchase', False):
                    token_id = st.session_state.selected_token
                    token = next(t for t in tokens if t['token_id'] == token_id)
                    
                    st.subheader(f"Acheter {token['company_name']}")
                    
                    quantity = st.number_input("Quantité", 1, token['available_supply'], 1)
                    total = quantity * token['price_per_token']
                    
                    st.write(f"**Total:** ${total:,.2f}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Confirmer l'Achat", type="primary"):
                            purchase_payload = {
                                "investor_id": st.session_state.investor_id,
                                "token_id": token_id,
                                "quantity": quantity,
                                "price_per_token": token['price_per_token']
                            }
                            
                            try:
                                resp = requests.post(f"{API_URL}/api/v1/tokens/purchase", json=purchase_payload)
                                if resp.status_code == 200:
                                    st.success("Achat réussi!")
                                    st.session_state.show_purchase = False
                                    st.rerun()
                                else:
                                    st.error("Erreur lors de l'achat")
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
                    
                    with col2:
                        if st.button("Annuler"):
                            st.session_state.show_purchase = False
                            st.rerun()
        
        except Exception as e:
            st.error(f"Erreur: {str(e)}")

# PAGE: Portefeuille
def page_portfolio():
    st.title("Mon Portefeuille")
    
    investor_id = st.session_state.investor_id
    
    try:
        response = requests.get(f"{API_URL}/api/v1/portfolio/{investor_id}")
        
        if response.status_code == 200:
            data = response.json()
            holdings = data['holdings']
            total_value = data['total_value']
            
            # Métriques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Valeur Totale", f"${total_value:,.2f}")
            with col2:
                st.metric("Positions", len(holdings))
            with col3:
                total_pl = sum(h['profit_loss'] for h in holdings)
                st.metric("P&L Total", f"${total_pl:,.2f}", delta=f"{total_pl:,.2f}")
            
            st.write("---")
            
            if not holdings:
                st.info("Votre portefeuille est vide")
                return
            
            # Tableau des positions
            st.subheader("Mes Positions")
            
            df = pd.DataFrame(holdings)
            df = df[['company_name', 'quantity', 'purchase_price', 'current_price', 'current_value', 'profit_loss', 'profit_loss_percentage']]
            df.columns = ['Entreprise', 'Quantité', 'Prix Achat', 'Prix Actuel', 'Valeur', 'P&L', 'P&L %']
            
            st.dataframe(df, use_container_width=True)
            
            # Graphique de répartition
            st.write("---")
            st.subheader("Répartition du Portefeuille")
            
            fig = px.pie(holdings, values='current_value', names='company_name', title="Par Entreprise")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Statistiques
def page_statistics():
    st.title("Statistiques et Analyses")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/statistics/platform")
        
        if response.status_code == 200:
            stats = response.json()
            
            # Métriques globales
            st.subheader("Vue d'Ensemble")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Entreprises", stats['total_companies'])
            with col2:
                st.metric("Tokens", stats['total_tokens'])
            with col3:
                st.metric("Transactions", stats['total_transactions'])
            with col4:
                st.metric("Investisseurs", stats['total_investors'])
            with col5:
                st.metric("Market Cap", f"${stats['total_market_cap']:,.0f}")
            
            st.write("---")
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Par Industrie")
                if stats['by_industry']:
                    df = pd.DataFrame(list(stats['by_industry'].items()), columns=['Industrie', 'Nombre'])
                    fig = px.pie(df, values='Nombre', names='Industrie')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Volume 24h")
                st.metric("Volume Total", f"${stats['total_volume_24h']:,.2f}")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Paramètres Avancés
def page_advanced_settings():
    st.title("Paramètres Avancés")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Général", "Valorisation", "Sécurité", "API"])
    
    with tab1:
        st.subheader("Paramètres Généraux")
        
        investor_id = st.text_input("ID Investisseur", value=st.session_state.investor_id)
        if st.button("Mettre à jour ID"):
            st.session_state.investor_id = investor_id
            st.success("ID mis à jour")
        
        st.write("---")
        
        currency = st.selectbox("Devise", ["USD", "EUR", "GBP"])
        language = st.selectbox("Langue", ["Français", "English"])
        timezone = st.selectbox("Fuseau Horaire", ["UTC", "EST", "PST"])
    
    with tab2:
        st.subheader("Paramètres de Valorisation IA")
        
        st.write("Multiplicateurs par Type d'Entreprise")
        startup_mult = st.slider("Startup", 1.0, 15.0, 8.0)
        sme_mult = st.slider("PME", 1.0, 10.0, 5.0)
        corp_mult = st.slider("Corporation", 1.0, 8.0, 3.0)
        
        st.write("---")
        
        st.write("Pondérations des Méthodes")
        dcf_weight = st.slider("DCF", 0.0, 1.0, 0.5)
        asset_weight = st.slider("Actifs", 0.0, 1.0, 0.2)
        revenue_weight = st.slider("Revenus", 0.0, 1.0, 0.3)
        
        total = dcf_weight + asset_weight + revenue_weight
        if abs(total - 1.0) > 0.01:
            st.warning(f"Total: {total:.2f} (doit être 1.0)")
    
    with tab3:
        st.subheader("Sécurité")
        
        two_factor = st.checkbox("Authentification à deux facteurs")
        email_notifications = st.checkbox("Notifications par email", value=True)
        transaction_alerts = st.checkbox("Alertes de transactions", value=True)
        
        st.write("---")
        
        st.write("Limites de Transaction")
        max_transaction = st.number_input("Transaction Max (USD)", 0.0, step=1000.0, value=100000.0)
        daily_limit = st.number_input("Limite Quotidienne (USD)", 0.0, step=1000.0, value=500000.0)
    
    with tab4:
        st.subheader("Configuration API")
        
        api_url = st.text_input("URL API", value=API_URL)
        api_key = st.text_input("Clé API", type="password")
        timeout = st.number_input("Timeout (secondes)", 1, 60, 30)
        
        if st.button("Tester Connexion"):
            try:
                resp = requests.get(f"{api_url}/health", timeout=timeout)
                if resp.status_code == 200:
                    st.success("Connexion réussie!")
                    st.json(resp.json())
                else:
                    st.error("Erreur de connexion")
            except Exception as e:
                st.error(f"Erreur: {str(e)}")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Sauvegarder les Paramètres", type="primary"):
            st.success("Paramètres sauvegardés")
    
    with col2:
        if st.button("Réinitialiser"):
            st.warning("Paramètres réinitialisés")

# Navigation
def main():
    init_session()
    
    with st.sidebar:
        st.title("Navigation")
        
        menu = {
            "Dashboard": "dashboard",
            "Créer Entreprise": "create_company",
            "Détails Entreprise": "company_details",
            "Marketplace": "marketplace",
            "Mon Portefeuille": "portfolio",
            "Statistiques": "statistics",
            "Paramètres Avancés": "settings"
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
        st.caption("Business Tokenization Platform v1.0")
    
    # Afficher la page active
    view = st.session_state.get('active_view', 'dashboard')
    
    if view == 'dashboard':
        page_dashboard()
    elif view == 'create_company':
        page_create_company()
    elif view == 'company_details':
        page_company_details()
    elif view == 'marketplace':
        page_marketplace()
    elif view == 'portfolio':
        page_portfolio()
    elif view == 'statistics':
        page_statistics()
    elif view == 'settings':
        page_advanced_settings()

if __name__ == "__main__":
    main()