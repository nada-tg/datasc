# 🔧 Guide de Configuration des APIs Réelles

## Vue d'Ensemble

La plateforme supporte maintenant de **vraies connexions** aux modèles d'IA :
- ✅ **ChatGPT** (OpenAI GPT-4)
- ✅ **Claude** (Anthropic Claude 3.5 Sonnet)
- ✅ **Llama** (Meta Llama 3 via Together AI)
- ✅ **DeepSeek** (DeepSeek Chat)
- 🔄 **Mistral** (À venir)

## 📦 Installation des Dépendances

```bash
pip install openai anthropic together litellm
```

## 🔑 Configuration des Clés API

### Option 1: Variables d'Environnement (Recommandé)

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-openai-key-here"
$env:ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
$env:TOGETHER_API_KEY="your-together-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
export TOGETHER_API_KEY="your-together-key-here"
```

### Option 2: Fichier .env

Créez un fichier `.env` à la racine :

```env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
TOGETHER_API_KEY=your-together-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
```

Puis installez python-dotenv :
```bash
pip install python-dotenv
```

Et ajoutez au début de votre API :
```python
from dotenv import load_dotenv
load_dotenv()
```

### Option 3: Configuration via API

Utilisez l'endpoint de configuration :

```python
import requests

response = requests.post("http://localhost:8004/api/v1/models/configure", json={
    "openai": "sk-your-key",
    "anthropic": "sk-ant-your-key",
    "together": "your-key"
})
```

## 🎯 Obtenir les Clés API

### OpenAI (ChatGPT)

1. Créez un compte sur [platform.openai.com](https://platform.openai.com)
2. Allez dans **API Keys**
3. Cliquez sur **Create new secret key**
4. Copiez votre clé (commence par `sk-`)
5. **Coût**: ~$0.03 par 1K tokens (GPT-4)

### Anthropic (Claude)

1. Créez un compte sur [console.anthropic.com](https://console.anthropic.com)
2. Allez dans **API Keys**
3. Créez une nouvelle clé
4. Copiez votre clé (commence par `sk-ant-`)
5. **Coût**: ~$0.015 par 1K tokens (Claude 3.5 Sonnet)

### Together AI (Llama)

1. Inscrivez-vous sur [together.ai](https://together.ai)
2. Allez dans **Settings** → **API Keys**
3. Créez une nouvelle clé
4. **Coût**: ~$0.002 par 1K tokens (Llama 3 70B)

### DeepSeek

1. Créez un compte sur [platform.deepseek.com](https://platform.deepseek.com)
2. Générez une clé API
3. **Coût**: ~$0.001 par 1K tokens

## 🧪 Tester les Connexions

### Via l'API

```python
import requests

# Vérifier le statut
response = requests.get("http://localhost:8004/api/v1/models/status")
print(response.json())

# Résultat attendu:
# {
#   "openai": {"available": true, "configured": true, "status": "ready"},
#   "anthropic": {"available": true, "configured": true, "status": "ready"},
#   ...
# }
```

### Test Manuel

```python
import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[{"role": "user", "content": "Test"}]
)

print(response.choices[0].message.content)
```

## 🚀 Utilisation

### 1. Conversation avec Modèles Réels

```python
# Créer une conversation
response = requests.post("http://localhost:8004/api/v1/conversation/start", json={
    "query": "Expliquez-moi la théorie de la relativité",
    "execution_mode": "model",
    "auto_assign_models": True
})

request_id = response.json()["request_id"]

# Vérifier le statut
status = requests.get(f"http://localhost:8004/api/v1/conversation/{request_id}")
print(status.json())
```

### 2. Agents avec Exécution Réelle

```python
# Créer une entreprise
company = requests.post("http://localhost:8004/api/v1/company/create", json={
    "name": "AI Consulting Inc",
    "industry": "Technology",
    "description": "Cabinet de conseil en IA",
    "ceo_name": "John Doe"
})

company_id = company.json()["company"]["company_id"]

# Recruter un agent
agent = requests.post(f"http://localhost:8004/api/v1/company/{company_id}/recruit", json={
    "name": "Alice",
    "role": "researcher",
    "specialization": "Machine Learning",
    "skills": ["Python", "TensorFlow", "Research"],
    "experience_level": 8
})

agent_id = agent.json()["agent"]["agent_id"]

# Assigner une tâche (exécution réelle!)
task = requests.post("http://localhost:8004/api/v1/company/assign-task", json={
    "company_id": company_id,
    "agent_id": agent_id,
    "task_description": "Faire une analyse complète des dernières avancées en IA générative",
    "priority": "high",
    "start_date": "2025-01-01T09:00:00",
    "end_date": "2025-01-05T17:00:00",
    "responsibility_level": 90,
    "deliverables": ["Rapport détaillé", "Présentation", "Recommandations"]
})

# Récupérer le résultat
task_id = task.json()["task"]["task_id"]
result = requests.get(f"http://localhost:8004/api/v1/task/{task_id}/result")
print(result.json())
```

## 💡 Mode Simulation vs Mode Réel

### Mode Simulation (Sans clés API)

Si aucune clé n'est configurée, la plateforme fonctionne en **mode simulation** :
- Réponses générées localement
- Pas de coûts
- Idéal pour tester l'architecture
- Mention `"simulation": true` dans les réponses

### Mode Réel (Avec clés API)

Avec les clés configurées :
- Appels aux vraies APIs
- Réponses de qualité production
- Coûts selon l'utilisation
- `"real_api_call": true` dans les réponses

## 📊 Monitoring des Coûts

### Suivre l'Utilisation

```python
# Dans chaque réponse, consultez:
{
    "tokens_used": 450,
    "model_used": "gpt-4-turbo-preview",
    "real_api_call": true
}

# Calculez le coût:
# GPT-4: $0.03 per 1K tokens
# Coût = (450 / 1000) * 0.03 = $0.0135
```

### Dashboard de Coûts (À implémenter)

```python
@app.get("/api/v1/costs/summary")
async def get_costs():
    return {
        "total_tokens": 125000,
        "estimated_cost": 3.75,
        "by_model": {
            "gpt-4": {"tokens": 50000, "cost": 1.50},
            "claude": {"tokens": 75000, "cost": 1.125}
        }
    }
```

## ⚠️ Bonnes Pratiques

### 1. Sécurité des Clés

- ❌ Ne commitez JAMAIS les clés dans Git
- ✅ Utilisez `.gitignore` pour `.env`
- ✅ Utilisez des variables d'environnement
- ✅ Rotation régulière des clés

### 2. Gestion des Coûts

- Définissez des limites de tokens
- Surveillez l'utilisation quotidienne
- Utilisez le mode simulation pour les tests
- Implémentez du caching pour les requêtes répétées

### 3. Gestion des Erreurs

```python
try:
    response = await RealModelExecutor.call_chatgpt(prompt)
    if "error" in response:
        # Basculer sur un modèle de secours
        response = await RealModelExecutor.call_claude(prompt)
except Exception as e:
    # Mode fallback
    response = RealModelExecutor._fallback_response("ChatGPT", prompt)
```

## 🔍 Debugging

### Vérifier les logs

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dans vos fonctions:
logger.debug(f"Calling {model} with prompt: {prompt[:100]}")
```

### Tester individuellement

```python
# Test ChatGPT
result = await RealModelExecutor.call_chatgpt("Test")
print(result)

# Test Claude
result = await RealModelExecutor.call_claude("Test")
print(result)
```

## 📚 Ressources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Anthropic Documentation](https://docs.anthropic.com)
- [Together AI Documentation](https://docs.together.ai)
- [DeepSeek Documentation](https://platform.deepseek.com/docs)

## 🆘 Support

En cas de problème :
1. Vérifiez que les clés sont correctement configurées
2. Consultez le statut : `GET /api/v1/models/status`
3. Vérifiez les logs de l'API
4. Testez les clés directement avec les SDKs

---

**Note**: Les coûts indiqués sont approximatifs. Consultez les sites officiels pour les tarifs actuels.



Ce qui a été amélioré
1. Appels API Réels

✅ ChatGPT (GPT-4) via OpenAI
✅ Claude 3.5 Sonnet via Anthropic
✅ Llama 3 70B via Together AI
✅ DeepSeek Chat
✅ Mode fallback si API non configurée

2. Agents Intelligents Fonctionnels

Les agents utilisent de vrais modèles IA pour accomplir les tâches
Sélection automatique du meilleur modèle selon le rôle
Analyse de qualité du travail produit
Rapports détaillés avec métriques

3. Exécution Séquentielle Réelle

Chaque étape appelle vraiment l'API correspondante
Le contexte est transmis d'une étape à l'autre
Synthèse finale basée sur les vraies réponses

🚀 Utilisation Rapide
Démarrer l'API
bash# Configurer les clés
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Lancer
uvicorn conversation_director_api:app --port 8004 --reload
Tester
bash# Vérifier les APIs disponibles
curl http://localhost:8004/api/v1/models/status

# Lancer une conversation
curl -X POST http://localhost:8004/api/v1/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"query": "Expliquez la blockchain", "execution_mode": "model