"""
business_tokenization_api.py - API pour Plateforme de Tokenisation d'Entreprises

Installation:
pip install fastapi uvicorn pydantic sqlalchemy numpy pandas scikit-learn

Lancement:
uvicorn business_tokenization_api:app --host 0.0.0.0 --port 8014 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import numpy as np
import json
from collections import defaultdict

app = FastAPI(
    title="Business Tokenization Platform API",
    description="Plateforme de tokenisation et valorisation d'entreprises par IA",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bases de données
COMPANIES_DB = {}
VALUATIONS_DB = {}
TOKENS_DB = {}
SHARES_DB = {}
PORTFOLIOS_DB = {}
EVENTS_DB = {}
PREDICTIONS_DB = {}
TRANSACTIONS_DB = {}

# Enums
class CompanyType(str, Enum):
    STARTUP = "startup"
    SME = "sme"
    CORPORATION = "corporation"
    NONPROFIT = "nonprofit"

class CompanyStatus(str, Enum):
    NEW = "new"
    EXISTING = "existing"

class Industry(str, Enum):
    TECH = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    ENERGY = "energy"
    REAL_ESTATE = "real_estate"

class EventType(str, Enum):
    FUNDING = "funding"
    PRODUCT_LAUNCH = "product_launch"
    ACQUISITION = "acquisition"
    PARTNERSHIP = "partnership"
    EXPANSION = "expansion"
    CRISIS = "crisis"

# Modèles Pydantic
class CompanyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    type: CompanyType
    status: CompanyStatus
    industry: Industry
    founded_year: Optional[int] = None
    description: str
    
    # Informations financières
    annual_revenue: float = Field(ge=0)
    annual_profit: float
    total_assets: float = Field(ge=0)
    total_liabilities: float = Field(ge=0)
    
    # Équipe
    employees_count: int = Field(ge=0)
    founders_count: int = Field(ge=1)
    management_experience: int = Field(ge=0, description="Années d'expérience moyenne")
    
    # Marché
    market_size: float = Field(ge=0)
    market_share: float = Field(ge=0, le=100)
    growth_rate: float = Field(ge=-100, le=1000)
    
    # Produits/Services
    products_count: int = Field(ge=0)
    customers_count: int = Field(ge=0)
    retention_rate: float = Field(ge=0, le=100)
    
    # Innovation
    patents_count: int = Field(ge=0)
    rd_investment: float = Field(ge=0)
    tech_score: float = Field(ge=0, le=100, description="Score technologique")
    
    # Tokenisation
    shares_to_tokenize: int = Field(gt=0)
    share_price_suggestion: Optional[float] = None

class TokenPurchase(BaseModel):
    investor_id: str
    token_id: str
    quantity: int = Field(gt=0)
    price_per_token: float = Field(gt=0)

class EventCreate(BaseModel):
    company_id: str
    event_type: EventType
    title: str
    description: str
    impact_score: float = Field(ge=-100, le=100)
    date: str

# Moteur IA de Valorisation
class AIValuationEngine:
    
    @staticmethod
    def calculate_company_valuation(company_data: Dict) -> Dict:
        """Calcule la valorisation complète d'une entreprise"""
        
        # Extraction des données
        revenue = company_data['annual_revenue']
        profit = company_data['annual_profit']
        assets = company_data['total_assets']
        liabilities = company_data['total_liabilities']
        employees = company_data['employees_count']
        growth_rate = company_data['growth_rate']
        market_share = company_data['market_share']
        
        # Calcul des métriques financières
        net_assets = assets - liabilities
        profit_margin = (profit / revenue * 100) if revenue > 0 else 0
        
        # Multiplicateurs selon le type et l'industrie
        type_multipliers = {
            'startup': 8,
            'sme': 5,
            'corporation': 3,
            'nonprofit': 2
        }
        
        industry_multipliers = {
            'technology': 1.5,
            'finance': 1.2,
            'healthcare': 1.3,
            'retail': 0.9,
            'manufacturing': 1.0,
            'services': 1.1,
            'energy': 1.2,
            'real_estate': 1.1
        }
        
        base_multiplier = type_multipliers.get(company_data['type'], 5)
        industry_mult = industry_multipliers.get(company_data['industry'], 1.0)
        
        # Valorisation par méthode DCF simplifiée
        dcf_value = revenue * base_multiplier * industry_mult
        
        # Ajustements basés sur la croissance
        growth_adjustment = 1 + (growth_rate / 100) * 0.5
        dcf_value *= growth_adjustment
        
        # Ajustements basés sur la rentabilité
        if profit_margin > 20:
            dcf_value *= 1.2
        elif profit_margin < 0:
            dcf_value *= 0.8
        
        # Valorisation par actifs nets
        asset_value = net_assets * 1.2
        
        # Valorisation par multiple de revenus
        revenue_multiple = revenue * 3 * industry_mult
        
        # Valorisation finale (moyenne pondérée)
        final_valuation = (dcf_value * 0.5 + asset_value * 0.2 + revenue_multiple * 0.3)
        
        # Score de confiance
        confidence_score = AIValuationEngine._calculate_confidence(company_data)
        
        # Calcul du prix par action
        total_shares = company_data.get('shares_to_tokenize', 1000000)
        share_price = final_valuation / total_shares
        
        return {
            "total_valuation": round(final_valuation, 2),
            "valuation_per_method": {
                "dcf": round(dcf_value, 2),
                "asset_based": round(asset_value, 2),
                "revenue_multiple": round(revenue_multiple, 2)
            },
            "share_price": round(share_price, 2),
            "total_shares": total_shares,
            "confidence_score": round(confidence_score, 2),
            "financial_metrics": {
                "profit_margin": round(profit_margin, 2),
                "roe": round((profit / net_assets * 100) if net_assets > 0 else 0, 2),
                "asset_turnover": round(revenue / assets if assets > 0 else 0, 2),
                "debt_to_equity": round((liabilities / net_assets) if net_assets > 0 else 0, 2)
            },
            "risk_level": AIValuationEngine._assess_risk(company_data),
            "growth_potential": AIValuationEngine._assess_growth_potential(company_data)
        }
    
    @staticmethod
    def _calculate_confidence(data: Dict) -> float:
        """Calcule le score de confiance de la valorisation"""
        score = 50
        
        if data['status'] == 'existing':
            score += 20
        
        # years_active = datetime.now().year - data.get('founded_year', datetime.now().year)
        try:
            founded_year = int(data.get("founded_year")) if data.get("founded_year") else datetime.now().year
        except ValueError:
            founded_year = datetime.now().year

        years_active = datetime.now().year - founded_year

        score += min(years_active * 2, 15)
        
        if data['annual_profit'] > 0:
            score += 10
        
        if data['customers_count'] > 1000:
            score += 5
        
        return min(score, 95)
    
    @staticmethod
    def _assess_risk(data: Dict) -> str:
        """Évalue le niveau de risque"""
        risk_score = 0
        
        if data['annual_profit'] < 0:
            risk_score += 30
        
        debt_ratio = data['total_liabilities'] / data['total_assets'] if data['total_assets'] > 0 else 1
        if debt_ratio > 0.7:
            risk_score += 25
        
        if data['employees_count'] < 10:
            risk_score += 15
        
        if data['market_share'] < 5:
            risk_score += 10
        
        if risk_score > 50:
            return "HIGH"
        elif risk_score > 25:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def _assess_growth_potential(data: Dict) -> str:
        """Évalue le potentiel de croissance"""
        potential_score = 0
        
        if data['growth_rate'] > 50:
            potential_score += 40
        elif data['growth_rate'] > 20:
            potential_score += 25
        
        if data['rd_investment'] > data['annual_revenue'] * 0.1:
            potential_score += 20
        
        if data['tech_score'] > 70:
            potential_score += 20
        
        if data['market_share'] < 20:
            potential_score += 10
        
        if potential_score > 60:
            return "HIGH"
        elif potential_score > 30:
            return "MEDIUM"
        else:
            return "LOW"
    
    @staticmethod
    def predict_future_events(company_id: str, company_data: Dict, timeframe_months: int = 12) -> List[Dict]:
        """Prédit les événements futurs pour une entreprise"""
        
        predictions = []
        
        # Prédiction de financement
        if company_data['growth_rate'] > 30 and company_data['annual_profit'] < 0:
            predictions.append({
                "event_type": "funding",
                "title": "Levée de fonds probable",
                "probability": 75,
                "expected_date": (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d"),
                "impact": "positive",
                "description": "Forte probabilité de levée de fonds basée sur la croissance"
            })
        
        # Prédiction de lancement produit
        if company_data['rd_investment'] > company_data['annual_revenue'] * 0.15:
            predictions.append({
                "event_type": "product_launch",
                "title": "Lancement de nouveau produit",
                "probability": 65,
                "expected_date": (datetime.now() + timedelta(days=120)).strftime("%Y-%m-%d"),
                "impact": "positive",
                "description": "Fort investissement R&D indique un lancement prochain"
            })
        
        # Prédiction d'expansion
        if company_data['market_share'] < 10 and company_data['growth_rate'] > 40:
            predictions.append({
                "event_type": "expansion",
                "title": "Expansion géographique",
                "probability": 55,
                "expected_date": (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d"),
                "impact": "positive",
                "description": "Croissance rapide et faible part de marché favorisent l'expansion"
            })
        
        # Prédiction de crise
        debt_ratio = company_data['total_liabilities'] / company_data['total_assets'] if company_data['total_assets'] > 0 else 0
        if debt_ratio > 0.8 and company_data['annual_profit'] < 0:
            predictions.append({
                "event_type": "crisis",
                "title": "Risque de difficultés financières",
                "probability": 45,
                "expected_date": (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
                "impact": "negative",
                "description": "Fort endettement et pertes nécessitent vigilance"
            })
        
        return predictions

# Routes API
@app.post("/api/v1/companies/create")
async def create_company(company: CompanyCreate):
    """Créer une entreprise et lancer la valorisation"""
    
    company_id = str(uuid.uuid4())
    
    company_data = company.dict()
    company_data['id'] = company_id
    company_data['created_at'] = datetime.now().isoformat()
    company_data['status_verification'] = 'pending'
    
    # Valorisation IA
    valuation = AIValuationEngine.calculate_company_valuation(company_data)
    
    # Création des tokens
    token_id = str(uuid.uuid4())
    token_data = {
        "token_id": token_id,
        "company_id": company_id,
        "total_supply": company.shares_to_tokenize,
        "available_supply": company.shares_to_tokenize,
        "price_per_token": valuation['share_price'],
        "created_at": datetime.now().isoformat(),
        "market_cap": valuation['total_valuation']
    }
    
    # Création des actions
    share_id = str(uuid.uuid4())
    share_data = {
        "share_id": share_id,
        "company_id": company_id,
        "total_shares": company.shares_to_tokenize,
        "available_shares": company.shares_to_tokenize,
        "price_per_share": valuation['share_price'],
        "created_at": datetime.now().isoformat()
    }
    
    # Prédictions d'événements
    predictions = AIValuationEngine.predict_future_events(company_id, company_data)
    
    # Sauvegarder
    COMPANIES_DB[company_id] = company_data
    VALUATIONS_DB[company_id] = valuation
    TOKENS_DB[token_id] = token_data
    SHARES_DB[share_id] = share_data
    PREDICTIONS_DB[company_id] = predictions
    
    return {
        "success": True,
        "company_id": company_id,
        "company": company_data,
        "valuation": valuation,
        "tokens": token_data,
        "shares": share_data,
        "predictions": predictions
    }

@app.get("/api/v1/companies/{company_id}")
async def get_company(company_id: str):
    """Récupérer les détails d'une entreprise"""
    
    if company_id not in COMPANIES_DB:
        raise HTTPException(status_code=404, detail="Entreprise non trouvée")
    
    company = COMPANIES_DB[company_id]
    valuation = VALUATIONS_DB.get(company_id, {})
    predictions = PREDICTIONS_DB.get(company_id, [])
    events = [e for e in EVENTS_DB.values() if e['company_id'] == company_id]
    
    # Calculer l'évolution
    token = next((t for t in TOKENS_DB.values() if t['company_id'] == company_id), None)
    
    evolution = {
        "initial_price": valuation.get('share_price', 0),
        "current_price": token['price_per_token'] if token else 0,
        "change_percentage": 0
    }
    
    if token and valuation.get('share_price'):
        change = ((token['price_per_token'] - valuation['share_price']) / valuation['share_price']) * 100
        evolution['change_percentage'] = round(change, 2)
    
    return {
        "company": company,
        "valuation": valuation,
        "predictions": predictions,
        "events": events,
        "evolution": evolution
    }

@app.get("/api/v1/marketplace/tokens")
async def list_tokens():
    """Liste tous les tokens disponibles"""
    
    tokens = list(TOKENS_DB.values())
    
    enriched_tokens = []
    for token in tokens:
        company = COMPANIES_DB.get(token['company_id'], {})
        valuation = VALUATIONS_DB.get(token['company_id'], {})
        
        enriched_tokens.append({
            **token,
            "company_name": company.get('name', 'Unknown'),
            "company_industry": company.get('industry', 'Unknown'),
            "risk_level": valuation.get('risk_level', 'UNKNOWN'),
            "growth_potential": valuation.get('growth_potential', 'UNKNOWN')
        })
    
    return {
        "total": len(enriched_tokens),
        "tokens": enriched_tokens
    }

@app.post("/api/v1/tokens/purchase")
async def purchase_tokens(purchase: TokenPurchase):
    """Acheter des tokens"""
    
    if purchase.token_id not in TOKENS_DB:
        raise HTTPException(status_code=404, detail="Token non trouvé")
    
    token = TOKENS_DB[purchase.token_id]
    
    if token['available_supply'] < purchase.quantity:
        raise HTTPException(status_code=400, detail="Quantité insuffisante")
    
    # Mise à jour du token
    token['available_supply'] -= purchase.quantity
    
    # Enregistrer la transaction
    transaction_id = str(uuid.uuid4())
    transaction = {
        "transaction_id": transaction_id,
        "investor_id": purchase.investor_id,
        "token_id": purchase.token_id,
        "quantity": purchase.quantity,
        "price_per_token": purchase.price_per_token,
        "total_amount": purchase.quantity * purchase.price_per_token,
        "timestamp": datetime.now().isoformat(),
        "type": "purchase"
    }
    
    TRANSACTIONS_DB[transaction_id] = transaction
    
    # Mettre à jour le portefeuille
    if purchase.investor_id not in PORTFOLIOS_DB:
        PORTFOLIOS_DB[purchase.investor_id] = []
    
    PORTFOLIOS_DB[purchase.investor_id].append({
        "token_id": purchase.token_id,
        "quantity": purchase.quantity,
        "purchase_price": purchase.price_per_token,
        "purchase_date": datetime.now().isoformat()
    })
    
    return {
        "success": True,
        "transaction": transaction,
        "remaining_supply": token['available_supply']
    }

@app.get("/api/v1/portfolio/{investor_id}")
async def get_portfolio(investor_id: str):
    """Récupérer le portefeuille d'un investisseur"""
    
    if investor_id not in PORTFOLIOS_DB:
        return {"investor_id": investor_id, "holdings": [], "total_value": 0}
    
    holdings = PORTFOLIOS_DB[investor_id]
    
    enriched_holdings = []
    total_value = 0
    
    for holding in holdings:
        token = TOKENS_DB.get(holding['token_id'], {})
        company = COMPANIES_DB.get(token.get('company_id'), {})
        
        current_price = token.get('price_per_token', holding['purchase_price'])
        current_value = holding['quantity'] * current_price
        profit_loss = (current_price - holding['purchase_price']) * holding['quantity']
        profit_loss_pct = ((current_price - holding['purchase_price']) / holding['purchase_price']) * 100 if holding['purchase_price'] > 0 else 0
        
        enriched_holdings.append({
            **holding,
            "company_name": company.get('name', 'Unknown'),
            "current_price": current_price,
            "current_value": round(current_value, 2),
            "profit_loss": round(profit_loss, 2),
            "profit_loss_percentage": round(profit_loss_pct, 2)
        })
        
        total_value += current_value
    
    return {
        "investor_id": investor_id,
        "holdings": enriched_holdings,
        "total_value": round(total_value, 2),
        "holdings_count": len(holdings)
    }

@app.post("/api/v1/events/create")
async def create_event(event: EventCreate):
    """Créer un événement pour une entreprise"""
    
    if event.company_id not in COMPANIES_DB:
        raise HTTPException(status_code=404, detail="Entreprise non trouvée")
    
    event_id = str(uuid.uuid4())
    event_data = event.dict()
    event_data['event_id'] = event_id
    event_data['created_at'] = datetime.now().isoformat()
    
    EVENTS_DB[event_id] = event_data
    
    # Mettre à jour le prix du token basé sur l'impact
    tokens = [t for t in TOKENS_DB.values() if t['company_id'] == event.company_id]
    for token in tokens:
        impact_multiplier = 1 + (event.impact_score / 100) * 0.1
        token['price_per_token'] *= impact_multiplier
        token['price_per_token'] = round(token['price_per_token'], 2)
    
    return {
        "success": True,
        "event": event_data
    }

@app.get("/api/v1/statistics/platform")
async def get_platform_statistics():
    """Statistiques globales de la plateforme"""
    
    total_companies = len(COMPANIES_DB)
    total_tokens = len(TOKENS_DB)
    total_transactions = len(TRANSACTIONS_DB)
    total_investors = len(PORTFOLIOS_DB)
    
    total_market_cap = sum(t.get('market_cap', 0) for t in TOKENS_DB.values())
    total_volume = sum(t.get('total_amount', 0) for t in TRANSACTIONS_DB.values())
    
    # Par industrie
    by_industry = {}
    for company in COMPANIES_DB.values():
        industry = company.get('industry', 'unknown')
        by_industry[industry] = by_industry.get(industry, 0) + 1
    
    return {
        "total_companies": total_companies,
        "total_tokens": total_tokens,
        "total_transactions": total_transactions,
        "total_investors": total_investors,
        "total_market_cap": round(total_market_cap, 2),
        "total_volume_24h": round(total_volume, 2),
        "by_industry": by_industry
    }

@app.get("/")
async def root():
    return {
        "message": "Business Tokenization Platform API",
        "version": "1.0.0",
        "endpoints": {
            "companies": "/api/v1/companies",
            "marketplace": "/api/v1/marketplace/tokens",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "companies": len(COMPANIES_DB),
        "tokens": len(TOKENS_DB)
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("BUSINESS TOKENIZATION PLATFORM API")
    print("=" * 70)
    print("\nAPI démarrée sur http://localhost:8002")
    print("Documentation: http://localhost:8002/docs")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8014)