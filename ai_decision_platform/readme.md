# 🤖 AI Decision Intelligence Platform

## Vue d'Ensemble

Plateforme complète et interactive pour comprendre **comment l'IA fonctionne**, **comment elle prend ses décisions**, et **comment résoudre les problèmes de biais et hallucinations**.

### 🎯 Objectifs

- ✅ **Comprendre l'architecture** des modèles IA (Transformers, CNN, RNN, etc.)
- ✅ **Analyser les décisions** et le processus de raisonnement
- ✅ **Détecter et mitiger les biais** (démographiques, algorithmiques, historiques)
- ✅ **Identifier les hallucinations** et fausses informations générées
- ✅ **Expliquer les prédictions** (SHAP, LIME, Attention)
- ✅ **Appliquer des techniques de mitigation** pour IA responsable

---

## 📦 Installation

### Prérequis

- Python 3.8+
- pip

### Étape 1: Cloner ou créer le projet

```bash
mkdir ai_decision_platform
cd ai_decision_platform
```

### Étape 2: Installer les dépendances

```bash
pip install fastapi uvicorn streamlit pandas plotly numpy scikit-learn networkx scipy
pip install python-jose[cryptography] passlib[bcrypt] python-multipart pydantic
```

### Étape 3: Créer les fichiers

Créez 3 fichiers principaux:

1. **ai_decision_api.py** - Backend FastAPI (fourni dans l'artifact)
2. **ai_decision_app.py** - Frontend Streamlit (combinez les 3 parties fournies)
3. **requirements.txt** - Dépendances

---

## 🚀 Lancement

### Démarrer le Backend (Optionnel)

```bash
uvicorn ai_decision_api:app --reload --host 0.0.0.0 --port 8030
```

Documentation API: http://localhost:8030/docs

### Démarrer le Frontend

```bash
streamlit run ai_decision_app.py
```

Interface: http://localhost:8501

---

## 📚 Fonctionnalités Principales

### 1. 🧠 Architecture IA

**Comprendre les différentes architectures:**

- **Transformers** (GPT, BERT)
  - Multi-head attention
  - Positional encoding
  - Feed-forward networks
  - Calcul paramètres et complexité

- **CNN** (Vision)
  - Convolutions
  - Pooling
  - Architecture complète

- **RNN/LSTM** (Séquences)
- **Arbres de Décision & Random Forests**

**Visualisations:**
- Architecture layer-by-layer
- Distribution paramètres
- Code PyTorch généré automatiquement

### 2. 🤖 Création de Modèles

**Configurer votre modèle personnalisé:**
- Type d'architecture
- Nombre de paramètres (millions)
- Nombre de couches
- Taille cachée
- Têtes d'attention
- Fenêtre de contexte

**Métriques calculées automatiquement:**
- Complexité
- Temps d'inférence estimé
- Utilisation mémoire
- FLOPs

### 3. 💭 Prise de Décision

**Générer des prédictions et analyser:**

- Entrée de texte/question
- Paramètres (temperature, top_p, max_tokens)
- Sortie générée avec confiance
- **Processus de raisonnement étape par étape**
- Poids d'attention visualisés
- Temps de traitement

### 4. ⚖️ Détection de Biais

**Types de biais détectés:**
- Biais de sélection
- Biais de confirmation
- Biais d'échantillonnage
- Biais algorithmique
- Biais historique
- Biais démographique

**Métriques de Fairness:**
- **Demographic Parity:** Taux prédiction égal entre groupes
- **Equal Opportunity:** Taux vrais positifs égal
- **Equalized Odds:** TPR + FPR égaux
- **Disparate Impact:** Ratio min/max ≥ 0.8
- **Calibration:** Probabilités calibrées

**Visualisations:**
- Scores de biais
- Métriques fairness par groupe
- Suggestions de mitigation automatiques

### 5. 👁️ Détection Hallucinations

**Identifier les fausses informations:**

- **Types d'hallucinations:**
  - Factuelles (faits inventés)
  - Logiques (incohérences)
  - Contextuelles (hors sujet)
  - Temporelles (anachronismes)

**Analyses:**
- Détection segments problématiques
- Score de risque par phrase
- Indicateurs (surconfiance, vague, nombres suspects)
- Fact-checking simulé
- Suggestions de correction

**Stratégies de prévention:**
- Retrieval-Augmented Generation (RAG)
- Temperature tuning
- Constrained decoding
- Fact-checking temps réel
- Confidence thres

Pages Complétées
1. 📊 Métriques Fairness (4 tabs)

Calculateur de métriques avec visualisations
Benchmarks par industrie
Définition d'objectifs personnalisés
Dashboard temps réel

2. 🔬 Analyse Profonde (4 tabs)

Dissection architecture modèle
Feature analysis (SHAP, PDP, interactions)
Gradient flow analysis
Tests adversariaux

3. 📚 Knowledge Base (4 tabs)

Documentation complète (7 sections majeures)
5 tutoriels pratiques avec code
FAQ détaillée (25+ questions)
Ressources externes et papers

4. 🎓 Entraînement (4 tabs)

Configuration training avec monitoring temps réel
Historique et comparaison runs
Hyperparameter optimization (Grid Search, Bayesian)
Gestion checkpoints

5. 🧪 Laboratoire Tests (3 tabs)

A/B Testing avec analyse statistique
Stress testing avec métriques
Security testing (injections, adversarial, extraction)

6. 📈 Performance (3 tabs)

Guide optimisations (Quantization, Distillation, etc.)
Benchmarks comparatifs
Profiling (cProfile, PyTorch Profiler)

7. 🌐 Comparaisons (2 tabs)

Comparaison multi-modèles avec visualisations
Leaderboards publics (GLUE, etc.)

🎯 Fonctionnalités Clés Ajoutées
✅ Graphiques interactifs Plotly
✅ Simulations réalistes
✅ Code examples pratiques
✅ Recommandations intelligentes
✅ Métriques détaillées
✅ Documentation complète
✅ Tutoriels step-by-step