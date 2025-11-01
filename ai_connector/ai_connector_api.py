"""
ai_connector_api.py - API pour Plateforme de Connexion d'IA

Installation:
pip install fastapi uvicorn pydantic openai anthropic requests

Lancement:
uvicorn ai_connector_api:app --host 0.0.0.0 --port 8003 --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import json
import asyncio
import time
import numpy as np

app = FastAPI(
    title="AI Connector Platform API",
    description="Plateforme de connexion et benchmarking d'IA multiples",
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
MODELS_DB = {}
CONNECTIONS_DB = {}
QUERIES_DB = {}
BENCHMARKS_DB = {}
ARCHITECTURES_DB = {}
HISTORY_DB = {}

# Enums
class ModelType(str, Enum):
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    PALM = "palm"
    CUSTOM = "custom"

class BenchmarkType(str, Enum):
    REASONING = "reasoning"
    CODING = "coding"
    MATH = "math"
    CREATIVE = "creative"
    FACTUAL = "factual"
    MULTILINGUAL = "multilingual"
    COMPREHENSIVE = "comprehensive"

class ConnectionType(str, Enum):
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    VOTING = "voting"
    HIERARCHICAL = "hierarchical"

# Modèles Pydantic
class AIModelConfig(BaseModel):
    name: str
    model_type: ModelType
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    model_version: str = "latest"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2000, gt=0)
    custom_params: Dict[str, Any] = {}

class ConnectionCreate(BaseModel):
    name: str
    model_ids: List[str] = Field(min_items=2)
    connection_type: ConnectionType
    synthesis_strategy: str = "best_response"
    description: Optional[str] = None

class QueryRequest(BaseModel):
    connection_id: str
    query: str
    context: Optional[str] = None
    expected_response_type: Optional[str] = None

class BenchmarkCreate(BaseModel):
    name: str
    model_ids: List[str] = Field(min_items=1)
    benchmark_type: BenchmarkType
    test_cases: List[Dict[str, Any]]
    custom_criteria: Optional[Dict[str, float]] = None

class ArchitectureCreate(BaseModel):
    name: str
    description: str
    nodes: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]

# Moteur de Synthèse
class SynthesisEngine:
    
    @staticmethod
    def synthesize_responses(responses: List[Dict], strategy: str, query: str) -> Dict:
        """Synthétise plusieurs réponses en une réponse optimale"""
        
        if strategy == "best_response":
            return SynthesisEngine._select_best_response(responses)
        elif strategy == "consensus":
            return SynthesisEngine._build_consensus(responses)
        elif strategy == "fusion":
            return SynthesisEngine._fuse_responses(responses, query)
        elif strategy == "voting":
            return SynthesisEngine._voting_synthesis(responses)
        else:
            return responses[0] if responses else {}
    
    @staticmethod
    def _select_best_response(responses: List[Dict]) -> Dict:
        """Sélectionne la meilleure réponse basée sur plusieurs critères"""
        
        scored_responses = []
        
        for resp in responses:
            score = 0
            
            # Longueur appropriée
            length = len(resp.get('response', ''))
            if 100 < length < 2000:
                score += 20
            
            # Confiance du modèle
            score += resp.get('confidence', 50)
            
            # Temps de réponse (plus rapide = mieux)
            response_time = resp.get('response_time', 1.0)
            score += max(0, 30 - response_time)
            
            scored_responses.append({
                'response': resp,
                'score': score
            })
        
        best = max(scored_responses, key=lambda x: x['score'])
        
        return {
            'synthesized_response': best['response']['response'],
            'source_model': best['response']['model_name'],
            'confidence': best['score'],
            'synthesis_method': 'best_response',
            'all_responses': responses
        }
    
    @staticmethod
    def _build_consensus(responses: List[Dict]) -> Dict:
        """Construit un consensus à partir de toutes les réponses"""
        
        # Extraire les points communs
        all_texts = [r.get('response', '') for r in responses]
        
        # Simulation de consensus (en production, utiliser NLP)
        consensus = f"Consensus de {len(responses)} modèles: "
        consensus += " | ".join([f"{r['model_name']}: {r['response'][:100]}..." for r in responses])
        
        avg_confidence = sum(r.get('confidence', 50) for r in responses) / len(responses)
        
        return {
            'synthesized_response': consensus,
            'source_model': 'consensus',
            'confidence': avg_confidence,
            'synthesis_method': 'consensus',
            'all_responses': responses
        }
    
    @staticmethod
    def _fuse_responses(responses: List[Dict], query: str) -> Dict:
        """Fusionne les réponses en une réponse cohérente"""
        
        fused = f"Synthèse de {len(responses)} analyses:\n\n"
        
        for i, resp in enumerate(responses, 1):
            fused += f"{i}. **{resp['model_name']}**: {resp['response']}\n\n"
        
        fused += f"\n**Conclusion**: Basé sur l'analyse combinée de {len(responses)} modèles d'IA."
        
        return {
            'synthesized_response': fused,
            'source_model': 'fusion',
            'confidence': 85,
            'synthesis_method': 'fusion',
            'all_responses': responses
        }
    
    @staticmethod
    def _voting_synthesis(responses: List[Dict]) -> Dict:
        """Synthèse par vote majoritaire"""
        
        # Simuler un vote (en production, analyser la similarité sémantique)
        votes = {}
        
        for resp in responses:
            key = resp['response'][:50]  # Simplification
            votes[key] = votes.get(key, []) + [resp]
        
        winner = max(votes.items(), key=lambda x: len(x[1]))
        
        return {
            'synthesized_response': winner[1][0]['response'],
            'source_model': 'voting',
            'confidence': (len(winner[1]) / len(responses)) * 100,
            'synthesis_method': 'voting',
            'votes': len(winner[1]),
            'all_responses': responses
        }

# Simulateur de Modèles IA
class AIModelSimulator:
    
    @staticmethod
    async def call_model(model_config: Dict, query: str) -> Dict:
        """Simule un appel à un modèle IA"""
        
        await asyncio.sleep(np.random.uniform(0.5, 2.0))  # Simuler latence
        
        model_type = model_config['model_type']
        
        # Réponses simulées par type de modèle
        responses = {
            'chatgpt': f"ChatGPT: En analysant votre question '{query}', je dirais que...",
            'claude': f"Claude: Après réflexion sur '{query}', voici mon analyse...",
            'llama': f"Llama: Concernant '{query}', mon évaluation est...",
            'gemini': f"Gemini: Pour répondre à '{query}', je propose...",
            'mistral': f"Mistral: En examinant '{query}', il apparaît que...",
        }
        
        response_text = responses.get(model_type, f"Réponse de {model_config['name']}")
        
        return {
            'model_id': model_config['id'],
            'model_name': model_config['name'],
            'model_type': model_type,
            'response': response_text,
            'confidence': np.random.uniform(70, 95),
            'response_time': np.random.uniform(0.5, 2.0),
            'tokens_used': np.random.randint(100, 500),
            'timestamp': datetime.now().isoformat()
        }

# Moteur de Benchmark
class BenchmarkEngine:
    
    @staticmethod
    async def run_benchmark(model_ids: List[str], benchmark_type: str, test_cases: List[Dict]) -> Dict:
        """Execute un benchmark complet"""
        
        results = {
            'benchmark_id': str(uuid.uuid4()),
            'benchmark_type': benchmark_type,
            'started_at': datetime.now().isoformat(),
            'model_results': {}
        }
        
        for model_id in model_ids:
            if model_id not in MODELS_DB:
                continue
            
            model = MODELS_DB[model_id]
            model_result = await BenchmarkEngine._test_model(model, test_cases, benchmark_type)
            results['model_results'][model_id] = model_result
        
        # Calcul des scores globaux
        results['rankings'] = BenchmarkEngine._calculate_rankings(results['model_results'])
        results['completed_at'] = datetime.now().isoformat()
        
        return results
    
    @staticmethod
    async def _test_model(model: Dict, test_cases: List[Dict], benchmark_type: str) -> Dict:
        """Teste un modèle sur tous les cas de test"""
        
        scores = []
        responses = []
        
        for test_case in test_cases:
            # Simuler l'exécution du test
            await asyncio.sleep(0.5)
            
            score = BenchmarkEngine._evaluate_response(
                test_case,
                benchmark_type,
                model['model_type']
            )
            
            scores.append(score)
            responses.append({
                'test_case': test_case.get('name', 'Test'),
                'score': score,
                'passed': score >= 70
            })
        
        return {
            'model_name': model['name'],
            'model_type': model['model_type'],
            'total_tests': len(test_cases),
            'passed': sum(1 for r in responses if r['passed']),
            'failed': sum(1 for r in responses if not r['passed']),
            'average_score': sum(scores) / len(scores) if scores else 0,
            'details': responses,
            'metrics': BenchmarkEngine._calculate_metrics(scores, benchmark_type)
        }
    
    @staticmethod
    def _evaluate_response(test_case: Dict, benchmark_type: str, model_type: str) -> float:
        """Évalue une réponse"""
        
        # Scores de base par type de modèle
        base_scores = {
            'chatgpt': {'reasoning': 85, 'coding': 90, 'math': 80, 'creative': 88},
            'claude': {'reasoning': 90, 'coding': 85, 'math': 82, 'creative': 92},
            'llama': {'reasoning': 75, 'coding': 80, 'math': 70, 'creative': 78},
            'gemini': {'reasoning': 82, 'coding': 88, 'math': 85, 'creative': 80},
            'mistral': {'reasoning': 80, 'coding': 83, 'math': 78, 'creative': 75}
        }
        
        base = base_scores.get(model_type, {}).get(benchmark_type, 75)
        variation = np.random.uniform(-10, 10)
        
        return max(0, min(100, base + variation))
    
    @staticmethod
    def _calculate_metrics(scores: List[float], benchmark_type: str) -> Dict:
        """Calcule les métriques détaillées"""
        
        return {
            'mean': round(np.mean(scores), 2),
            'median': round(np.median(scores), 2),
            'std': round(np.std(scores), 2),
            'min': round(min(scores), 2),
            'max': round(max(scores), 2),
            'percentile_25': round(np.percentile(scores, 25), 2),
            'percentile_75': round(np.percentile(scores, 75), 2)
        }
    
    @staticmethod
    def _calculate_rankings(model_results: Dict) -> List[Dict]:
        """Calcule le classement des modèles"""
        
        rankings = []
        
        for model_id, result in model_results.items():
            rankings.append({
                'model_id': model_id,
                'model_name': result['model_name'],
                'average_score': result['average_score'],
                'passed_tests': result['passed'],
                'total_tests': result['total_tests']
            })
        
        rankings.sort(key=lambda x: x['average_score'], reverse=True)
        
        for i, rank in enumerate(rankings, 1):
            rank['rank'] = i
        
        return rankings

# Routes API
@app.post("/api/v1/models/register")
async def register_model(model: AIModelConfig):
    """Enregistrer un modèle IA"""
    
    model_id = str(uuid.uuid4())
    
    model_data = model.dict()
    model_data['id'] = model_id
    model_data['registered_at'] = datetime.now().isoformat()
    model_data['status'] = 'active'
    
    MODELS_DB[model_id] = model_data
    
    return {
        "success": True,
        "model_id": model_id,
        "model": model_data
    }

@app.get("/api/v1/models")
async def list_models():
    """Liste tous les modèles"""
    
    return {
        "total": len(MODELS_DB),
        "models": list(MODELS_DB.values())
    }

@app.post("/api/v1/connections/create")
async def create_connection(connection: ConnectionCreate):
    """Créer une connexion entre modèles"""
    
    # Vérifier que tous les modèles existent
    for model_id in connection.model_ids:
        if model_id not in MODELS_DB:
            raise HTTPException(status_code=404, detail=f"Modèle {model_id} non trouvé")
    
    connection_id = str(uuid.uuid4())
    
    connection_data = connection.dict()
    connection_data['id'] = connection_id
    connection_data['created_at'] = datetime.now().isoformat()
    connection_data['queries_count'] = 0
    
    CONNECTIONS_DB[connection_id] = connection_data
    
    return {
        "success": True,
        "connection_id": connection_id,
        "connection": connection_data
    }

@app.post("/api/v1/query")
async def execute_query(query_request: QueryRequest):
    """Exécuter une requête sur une connexion"""
    
    if query_request.connection_id not in CONNECTIONS_DB:
        raise HTTPException(status_code=404, detail="Connexion non trouvée")
    
    connection = CONNECTIONS_DB[query_request.connection_id]
    query_id = str(uuid.uuid4())
    
    # Collecter les réponses de tous les modèles
    responses = []
    
    for model_id in connection['model_ids']:
        if model_id in MODELS_DB:
            model = MODELS_DB[model_id]
            response = await AIModelSimulator.call_model(model, query_request.query)
            responses.append(response)
    
    # Synthétiser les réponses
    synthesis = SynthesisEngine.synthesize_responses(
        responses,
        connection['synthesis_strategy'],
        query_request.query
    )
    
    # Sauvegarder
    query_data = {
        'query_id': query_id,
        'connection_id': query_request.connection_id,
        'query': query_request.query,
        'responses': responses,
        'synthesis': synthesis,
        'timestamp': datetime.now().isoformat()
    }
    
    QUERIES_DB[query_id] = query_data
    connection['queries_count'] += 1
    
    # Historique
    if connection_id not in HISTORY_DB:
        HISTORY_DB[connection['id']] = []
    HISTORY_DB[connection['id']].append(query_data)
    
    return {
        "success": True,
        "query_id": query_id,
        "synthesis": synthesis,
        "individual_responses": responses
    }

@app.post("/api/v1/benchmark/create")
async def create_benchmark(benchmark: BenchmarkCreate, background_tasks: BackgroundTasks):
    """Créer et exécuter un benchmark"""
    
    benchmark_id = str(uuid.uuid4())
    
    # Créer le benchmark
    benchmark_data = {
        'benchmark_id': benchmark_id,
        'name': benchmark.name,
        'model_ids': benchmark.model_ids,
        'benchmark_type': benchmark.benchmark_type,
        'test_cases': benchmark.test_cases,
        'status': 'pending',
        'created_at': datetime.now().isoformat()
    }
    
    BENCHMARKS_DB[benchmark_id] = benchmark_data
    
    # Exécuter en arrière-plan
    async def run():
        benchmark_data['status'] = 'running'
        results = await BenchmarkEngine.run_benchmark(
            benchmark.model_ids,
            benchmark.benchmark_type,
            benchmark.test_cases
        )
        benchmark_data['results'] = results
        benchmark_data['status'] = 'completed'
    
    background_tasks.add_task(run)
    
    return {
        "success": True,
        "benchmark_id": benchmark_id,
        "status": "pending"
    }

@app.get("/api/v1/benchmark/{benchmark_id}")
async def get_benchmark(benchmark_id: str):
    """Récupérer les résultats d'un benchmark"""
    
    if benchmark_id not in BENCHMARKS_DB:
        raise HTTPException(status_code=404, detail="Benchmark non trouvé")
    
    return BENCHMARKS_DB[benchmark_id]

@app.post("/api/v1/architecture/create")
async def create_architecture(architecture: ArchitectureCreate):
    """Créer une architecture de test"""
    
    arch_id = str(uuid.uuid4())
    
    arch_data = architecture.dict()
    arch_data['id'] = arch_id
    arch_data['created_at'] = datetime.now().isoformat()
    
    ARCHITECTURES_DB[arch_id] = arch_data
    
    return {
        "success": True,
        "architecture_id": arch_id,
        "architecture": arch_data
    }

@app.get("/api/v1/history/{connection_id}")
async def get_history(connection_id: str, limit: int = 50):
    """Récupérer l'historique d'une connexion"""
    
    if connection_id not in HISTORY_DB:
        return {"connection_id": connection_id, "history": []}
    
    history = HISTORY_DB[connection_id][-limit:]
    
    return {
        "connection_id": connection_id,
        "total": len(HISTORY_DB[connection_id]),
        "history": history
    }

@app.get("/api/v1/statistics")
async def get_statistics():
    """Statistiques globales"""
    
    total_queries = len(QUERIES_DB)
    total_benchmarks = len(BENCHMARKS_DB)
    
    return {
        "total_models": len(MODELS_DB),
        "total_connections": len(CONNECTIONS_DB),
        "total_queries": total_queries,
        "total_benchmarks": total_benchmarks,
        "models_by_type": {
            model_type: len([m for m in MODELS_DB.values() if m['model_type'] == model_type])
            for model_type in ModelType
        }
    }

@app.get("/")
async def root():
    return {
        "message": "AI Connector Platform API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/api/v1/models",
            "connections": "/api/v1/connections",
            "benchmarks": "/api/v1/benchmark",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": len(MODELS_DB),
        "connections": len(CONNECTIONS_DB)
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("AI CONNECTOR PLATFORM API")
    print("=" * 70)
    print("\nAPI démarrée sur http://localhost:8003")
    print("Documentation: http://localhost:8003/docs")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8003)