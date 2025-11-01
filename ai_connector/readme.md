# 🔗 AI Connector Platform

Plateforme complète de connexion, benchmarking et orchestration de multiples modèles d'IA.

## 🎯 Vue d'Ensemble

Cette plateforme permet de :
- **Connecter** plusieurs modèles d'IA (ChatGPT, Claude, Llama, etc.)
- **Synthétiser** leurs réponses en une réponse optimale
- **Benchmarker** leurs performances avec des tests standardisés
- **Orchestrer** des architectures complexes de traitement
- **Analyser** les performances comparatives

## 🚀 Fonctionnalités Principales

### 1. Connexion Multi-Modèles

Connectez 2 ou plusieurs modèles d'IA :
- ChatGPT (GPT-3.5, GPT-4)
- Claude (Anthropic)
- Llama 2/3
- Gemini (Google)
- Mistral AI
- PaLM
- Modèles personnalisés

### 2. Types de Connexion

**Parallel** - Tous les modèles répondent simultanément
```
Query → [Model A, Model B, Model C] → Synthesis
```

**Sequential** - Chaîne de traitement
```
Query → Model A → Model B → Model C → Result
```

**Voting** - Vote démocratique
```
Query → [Models] → Vote → Winner Response
```

**Hierarchical** - Structure hiérarchique
```
Query → Expert Model → [Specialist Models] → Final
```

### 3. Stratégies de Synthèse

**Best Response**
- Sélectionne la meilleure réponse selon des critères
- Évalue : longueur, confiance, temps de réponse
- Score pondéré

**Consensus**
- Construit un consensus entre toutes les réponses
- Extrait les points communs
- Génère une réponse unifiée

**Fusion**
- Fusionne toutes les réponses
- Présente chaque perspective
- Conclusion synthétique

**Voting**
- Vote majoritaire
- Sélectionne la réponse la plus fréquente
- Basé sur similarité sémantique

### 4. Système de Benchmark

#### Types de Tests

- **Reasoning** - Raisonnement logique
- **Coding** - Génération de code
- **Math** - Calculs mathématiques
- **Creative** - Créativité
- **Factual** - Connaissances factuelles
- **Multilingual** - Capacités multilingues
- **Comprehensive** - Tests complets

#### Métriques Calculées

- Score moyen
- Médiane
- Écart-type
- Min/Max
- Percentiles (25%, 75%)
- Taux de réussite
- Temps de réponse

### 5. Architecture Visuelle

Créez des architectures de test personnalisées :
- Définissez des nœuds (input, processor, output)
- Connectez-les visuellement
- Placez des modèles sur chaque nœud
- Exécutez des workflows complexes

## 📊 Utilisation

### Enregistrer un Modèle

```python
POST /api/v1/models/register
{
    "name": "Mon GPT-4",
    "model_type": "chatgpt",
    "api_key": "sk-...",
    "model_version": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000
}
```

### Créer une Connexion

```python
POST /api/v1/connections/create
{
    "name": "Trio Expert",
    "model_ids": ["id1", "id2", "id3"],
    "connection_type": "parallel",
    "synthesis_strategy": "best_response"
}
```

### Exécuter une Requête

```python
POST /api/v1/query
{
    "connection_id": "conn_123",
    "query": "Expliquez la théorie de la relativité"
}
```

Réponse :
```json
{
    "synthesis": {
        "synthesized_response": "...",
        "source_model": "best_model",
        "confidence": 92.5,
        "synthesis_method": "best_response"
    },
    "individual_responses": [...]
}
```

### Lancer un Benchmark

```python
POST /api/v1/benchmark/create
{
    "name": "Test Reasoning",
    "model_ids": ["id1", "id2"],
    "benchmark_type": "reasoning",
    "test_cases": [
        {"name": "Test 1", "difficulty": "medium"},
        {"name": "Test 2", "difficulty": "hard"}
    ]
}
```

## 🔧 Installation

### Backend API

```bash
# Installer les dépendances
pip install fastapi uvicorn pydantic openai anthropic requests

# Lancer l'API
uvicorn ai_connector_api:app --host 0.0.0.0 --port 8003 --reload
```

### Frontend Streamlit

```bash
# Installer Streamlit
pip install streamlit plotly pandas

# Lancer l'interface
streamlit run ai_connector_frontend.py
```

## 📈 Analyses et Statistiques

### Par Modèle

- Nombre de requêtes traitées
- Temps de réponse moyen
- Score de confiance moyen
- Taux d'utilisation
- Coût par requête

### Par Connexion

- Nombre d'utilisations
- Performance de synthèse
- Modèles les plus sollicités
- Stratégie la plus efficace

### Benchmarks

- Classement global
- Évolution dans le temps
- Comparaison par catégorie
- Points forts/faibles

## 🎨 Architectures Avancées

### Exemple: Pipeline de Traitement

```
Input → Preprocessor Model
      → Analyzer Model A
      → Analyzer Model B  
      → Synthesizer Model
      → Output
```

### Exemple: Validation Croisée

```
Query → [Model 1, Model 2, Model 3]
      → Validator Model
      → Confidence Check
      → Final Response
```

## 🔒 Sécurité

- Clés API chiffrées
- Rate limiting
- Validation des entrées
- Logs d'audit
- Isolation des modèles

## ⚙️ Configuration

### Paramètres Globaux

```python
{
    "default_temperature": 0.7,
    "default_max_tokens": 2000,
    "timeout": 30,
    "cache_enabled": true,
    "parallel_requests": true
}
```

### Paramètres par Modèle

```python
{
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

## 📊 Cas d'Usage

### 1. Question Complexe

Utilisez plusieurs modèles pour obtenir différentes perspectives sur une question difficile.

### 2. Validation de Réponse

Demandez à plusieurs modèles de vérifier une réponse et construisez un consensus.

### 3. Spécialisation

Dirigez les questions techniques vers des modèles spécialisés et combinez les résultats.

### 4. Benchmarking Continu

Testez régulièrement vos modèles pour suivre leur évolution.

### 5. Optimisation de Coûts

Utilisez un modèle rapide pour le tri initial, puis des modèles puissants pour les cas complexes.

## 🚦 Bonnes Pratiques

1. **Choisir les bons modèles** - Sélectionnez des modèles complémentaires
2. **Optimiser les prompts** - Utilisez des prompts clairs et spécifiques
3. **Surveiller les coûts** - Trackez l'utilisation des APIs
4. **Tester régulièrement** - Benchmarkez pour maintenir la qualité
5. **Analyser les résultats** - Utilisez les métriques pour améliorer

## 📝 Historique

Toutes les requêtes sont sauvegardées avec :
- Query originale
- Réponses individuelles
- Synthèse finale
- Métriques de performance
- Timestamp

## 🔮 Roadmap

- [ ] Support de plus de modèles (Cohere, AI21, etc.)
- [ ] Éditeur visuel d'architecture drag & drop
- [ ] Benchmarks standardisés (MMLU, HumanEval, etc.)
- [ ] Fine-tuning collaboratif
- [ ] API webhooks pour événements
- [ ] Dashboard temps réel
- [ ] Export des résultats (PDF, Excel)
- [ ] Intégration CI/CD

## 🤝 Contribution

Les contributions sont les bienvenues !

## 📄 Licence

MIT License

## 📞 Support

- Documentation: `/docs`
- API Health: `GET /health`
- Issues: GitHub Issues

---

Développé pour faciliter l'orchestration de modèles d'IA multiples.


Récapitulatif des Plateformes Créées
1. AI Development Platform (Ports 8001 & 8501)

Développement de projets IA (modèles, agents, apps)
Workspace avec éditeur de code multi-langages
Entraînement de modèles IA
Statistiques utilisateur et plateforme
Déploiement automatisé

2. Business Tokenization Platform (Ports 8002 & 8502)

Valorisation d'entreprises par IA
Conversion en tokens négociables
Marketplace d'actifs tokenisés
Portefeuille électronique
Prédictions d'événements
Gestion d'entreprises

3. AI Connector Platform (Ports 8003 & 8503)

Connexion de multiples modèles IA
Synthèse intelligente de réponses
Benchmarking automatisé
Architecture visuelle de tests
Historique et analytics