# 🤖 AI Training Platform

Une plateforme complète d'entraînement d'intelligence artificielle avec API robuste et interface utilisateur moderne.

## 🚀 Fonctionnalités

### 🔧 API d'Entraînement (Port 8006)
- **Multi-framework** : scikit-learn, PyTorch, TensorFlow, XGBoost
- **Types de tâches** : Classification, Régression, Clustering
- **Monitoring temps réel** : WebSocket, métriques système
- **Gestion de modèles** : Checkpoints, déploiement, API de prédiction
- **Datasets partagés** : Upload, gestion, prévisualisation

### 🎨 Interface Streamlit (Port 8007) 
- **Dashboard interactif** avec métriques temps réel
- **Configurateur de modèles** avec hyperparamètres dynamiques
- **Monitoring live** avec graphiques Plotly
- **Gestionnaire de datasets** avec prévisualisation
- **Testeur de modèles déployés**
- **Visualisations avancées** de performance

## 📦 Installation

### 1. Cloner ou créer les fichiers
Créez les fichiers suivants dans un dossier :
- `ai_training_platform.py` (API FastAPI)
- `ai_training_dashboard.py` (Interface Streamlit)
- `start_ai_platform.py` (Script de démarrage)
- `requirements.txt` (Dépendances)

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Démarrer la plateforme
```bash
python start_ai_platform.py
```

## 🎯 Utilisation

### Accès aux services
- **Interface utilisateur** : http://localhost:8007
- **API d'entraînement** : http://localhost:8006  
- **Documentation API** : http://localhost:8006/docs

### Workflow typique

1. **Datasets** : Ajoutez vos données ou utilisez les datasets par défaut
2. **Nouveau Modèle** : Configurez algorithme, hyperparamètres, dataset
3. **Entraînement** : Lancez et suivez le progrès en temps réel
4. **Monitoring** : Visualisez métriques, logs, performance
5. **Déploiement** : Déployez votre modèle comme API de prédiction
6. **Test** : Testez vos modèles directement dans l'interface

## 🔧 Configuration

### Datasets par défaut
Au démarrage, 3 datasets sont automatiquement créés :
- **Iris Dataset** (classification) - 150 échantillons
- **House Prices** (régression) - 1000 échantillons  
- **Customer Segmentation** (clustering) - 500 échantillons

### Algorithmes supportés

#### Scikit-learn
- Random Forest, Logistic Regression, SVM
- Neural Networks, Gradient Boosting

#### PyTorch
- Réseaux de neurones personnalisables
- Support GPU automatique

#### TensorFlow/Keras
- Architectures denses et convolutionnelles
- Callbacks personnalisés

#### XGBoost
- Classification et régression
- Validation croisée intégrée

## 📊 Monitoring

### Temps réel
- **Métriques d'entraînement** : Loss, accuracy, learning rate
- **Ressources système** : CPU, RAM, GPU
- **Logs live** : Progression détaillée
- **WebSocket** : Mises à jour automatiques

### Visualisations
- **Courbes d'apprentissage** interactives
- **Matrices de corrélation**
- **Comparaisons de modèles**
- **Performance système**

## 🚀 Déploiement

### API de prédiction
Déployez vos modèles entraînés comme APIs REST :

```python
# Exemple d'utilisation d'un modèle déployé
import requests

response = requests.post(
    "http://localhost:8006/api/predict/MODEL_ID",
    json={"features": [1.2, 3.4, 5.6, 7.8]}
)
prediction = response.json()
```

### Endpoints principaux
- `POST /training/start` - Lancer un entraînement
- `GET /training/jobs/{user_id}` - Liste des jobs
- `GET /training/job/{job_id}/status` - Statut temps réel
- `POST /models/deploy` - Déployer un modèle
- `POST /api/predict/{model_id}` - Prédiction

## 🛠 Architecture technique

### Backend
- **FastAPI** avec WebSocket pour temps réel
- **SQLite** pour persistance des métadonnées
- **Threading** pour jobs asynchrones
- **Monitoring système** avec psutil/GPUtil

### Frontend  
- **Streamlit** avec CSS personnalisé
- **Plotly** pour visualisations interactives
- **WebSocket client** pour mises à jour live
- **Design responsive** moderne

## 🔍 Dépannage

### API ne démarre pas
```bash
# Vérifier les dépendances
pip install -r requirements.txt

# Démarrage manuel
python -m uvicorn ai_training_platform:app --host 0.0.0.0 --port 8006
```

### Interface Streamlit inaccessible
```bash
# Démarrage manuel
streamlit run ai_training_dashboard.py --server.port 8007
```

### Erreurs de mémoire
- Réduisez la taille des datasets
- Diminuez batch_size dans les hyperparamètres
- Utilisez des modèles moins complexes

### WebSocket ne fonctionne pas
- Vérifiez que l'API est démarrée
- Désactivez les bloqueurs de popup
- Utilisez le mode debug dans l'interface

## 📈 Performance

### Optimisations
- **Datasets** : Stockage optimisé avec pandas
- **Modèles** : Checkpoints automatiques
- **Mémoire** : Libération automatique des ressources
- **Cache** : Mise en cache des prédictions

### Limites
- **SQLite** : Pour usage développement/test
- **Threading** : Pas de distribution multi-machines
- **Stockage** : Fichiers locaux uniquement

## 🔮 Extensions futures

- Support PostgreSQL/MySQL
- Distribution avec Celery/Redis  
- AutoML et recherche d'hyperparamètres
- Pipelines MLOps complets
- Intégration cloud (AWS, GCP, Azure)
- Monitoring avancé avec Prometheus
- A/B testing de modèles

## 📞 Support

Pour toute question ou problème :
1. Vérifiez les logs dans la console
2. Consultez la documentation API : http://localhost:8006/docs
3. Utilisez le mode debug dans l'interface Streamlit

---

**Version** : 1.0.0  
**Dernière mise à jour** : 2024

Créé avec ❤️ pour simplifier l'entraînement d'IA














<!-- API AI Training Platform (Port 8006)
Fonctionnalités principales :

Moteur d'entraînement multi-framework : scikit-learn, PyTorch, TensorFlow, XGBoost
Support de tâches diverses : classification, régression, clustering
Monitoring temps réel : WebSocket, métriques système, logs streaming
Gestion complète des modèles : checkpoints, déploiement, API de prédiction
Datasets partagés : upload, gestion, prévisualisation
Architecture asynchrone : jobs en arrière-plan, suivi de progression

Endpoints clés :

/training/start - Lancer un entraînement
/training/job/{job_id}/status - Statut en temps réel
/models/deploy - Déployer un modèle
/api/predict/{model_id} - Utiliser un modèle déployé

📊 Interface Streamlit Avancée
Pages principales :

Dashboard - Vue d'ensemble avec métriques temps réel
Nouveau Modèle - Configurateur complet avec hyperparamètres
Mes Modèles - Gestion, monitoring, déploiement
Monitoring - Suivi temps réel avec graphiques interactifs
Datasets - Upload, gestion, prévisualisation
Modèles Déployés - Test, monitoring d'usage, endpoints API

Fonctionnalités avancées :

Interface responsive avec design moderne
Monitoring système : CPU, RAM, GPU en temps réel
WebSocket pour mises à jour live
Visualisations Plotly : métriques, loss curves, performance
Testeur de modèles intégré
Configuration d'hyperparamètres dynamique

🚀 Architecture Technique
Backend :

FastAPI avec WebSocket
SQLite pour persistance
Threading pour jobs asynchrones
Monitoring système avec psutil/GPUtil
Support de tous les frameworks ML populaires

Frontend :

Streamlit avec CSS personnalisé
Connexions WebSocket temps réel
Graphiques interactifs Plotly
Interface modulaire et extensible

🔧 Utilisation

Lancer l'API : python ai_training_platform.py (port 8006)
Lancer l'interface : streamlit run ai_training_dashboard.py
Créer un modèle : Choisir dataset → algorithme → hyperparamètres → lancer
Suivre l'entraînement : Monitoring temps réel avec métriques et logs
Déployer : Un clic pour créer une API de prédiction
Utiliser : Testeur intégré ou appels API directs

Votre plateforme est maintenant complète et prête pour l'entraînement d'IA professionnel avec toutes les fonctionnalités demandées ! -->