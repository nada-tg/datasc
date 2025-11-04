"""
conversation_director_api.py - API de Direction de Conversations IA avec intégrations réelles

Installation:
pip install fastapi uvicorn pydantic openai anthropic together litellm schedule

Lancement:
uvicorn conversation_director_api:app --host 0.0.0.0 --port 8017 --reload
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import json
import os
from dotenv import load_dotenv
load_dotenv()
# Imports pour les APIs réelles
try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except:
    ANTHROPIC_AVAILABLE = False

try:
    from together import Together
    TOGETHER_AVAILABLE = True
except:
    TOGETHER_AVAILABLE = False

app = FastAPI(
    title="AI Conversation Director Platform",
    description="Gestion intelligente avec connexions réelles aux APIs",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des APIs (à définir via variables d'environnement)
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", "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"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "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"),
    "together": os.getenv("TOGETHER_API_KEY", "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"),
}

# Bases de données
REQUESTS_DB = {}
STEPS_DB = {}
MODELS_DB = {}
AGENTS_DB = {}
COMPANIES_DB = {}
TASKS_DB = {}
SCHEDULES_DB = {}
RESULTS_DB = {}

# Enums (identiques)
class ExecutionMode(str, Enum):
    MODEL = "model"
    AGENT = "agent"

class ModelType(str, Enum):
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GEMINI = "gemini"

class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    DEVELOPER = "developer"
    MANAGER = "manager"
    SPECIALIST = "specialist"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

# Modèles Pydantic (identiques)
class ConversationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    execution_mode: ExecutionMode
    context: Optional[str] = None
    auto_assign_models: bool = True

class StepConfig(BaseModel):
    step_id: str
    model_type: Optional[ModelType] = None
    model_version: Optional[str] = None
    agent_id: Optional[str] = None

class CompanyCreate(BaseModel):
    name: str
    industry: str
    description: str
    ceo_name: str

class AgentCreate(BaseModel):
    name: str
    role: AgentRole
    specialization: str
    skills: List[str]
    experience_level: int = Field(ge=1, le=10)

class TaskAssignment(BaseModel):
    company_id: str
    agent_id: str
    task_description: str
    priority: TaskPriority
    start_date: str
    end_date: str
    responsibility_level: int = Field(ge=1, le=100)
    deliverables: List[str]

# Gestionnaire d'APIs réelles
class RealModelExecutor:
    """Exécute les requêtes sur de vrais modèles IA"""
    
    @staticmethod
    async def call_chatgpt(prompt: str, context: str = "") -> Dict:
        """Appelle ChatGPT"""
        if not API_KEYS.get("openai") or not OPENAI_AVAILABLE:
            return RealModelExecutor._fallback_response("ChatGPT", prompt)
        
        try:
            client = openai.OpenAI(api_key=API_KEYS["openai"])
            
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": "gpt-4-turbo-preview",
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            return {"error": str(e), "response": f"Erreur ChatGPT: {str(e)}"}
    
    @staticmethod
    async def call_claude(prompt: str, context: str = "") -> Dict:
        """Appelle Claude"""
        if not API_KEYS.get("anthropic") or not ANTHROPIC_AVAILABLE:
            return RealModelExecutor._fallback_response("Claude", prompt)
        
        try:
            client = anthropic.Anthropic(api_key=API_KEYS["anthropic"])
            
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.7,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            return {
                "response": message.content[0].text,
                "model": "claude-3-5-sonnet",
                "tokens_used": message.usage.input_tokens + message.usage.output_tokens,
                "stop_reason": message.stop_reason
            }
        except Exception as e:
            return {"error": str(e), "response": f"Erreur Claude: {str(e)}"}
    
    @staticmethod
    async def call_llama(prompt: str, context: str = "") -> Dict:
        """Appelle Llama via Together AI"""
        if not API_KEYS.get("together") or not TOGETHER_AVAILABLE:
            return RealModelExecutor._fallback_response("Llama", prompt)
        
        try:
            client = Together(api_key=API_KEYS["together"])
            
            full_prompt = f"{context}\n\n{prompt}" if context else prompt
            
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_tokens=1500
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": "llama-3-70b",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            return {"error": str(e), "response": f"Erreur Llama: {str(e)}"}
    
    @staticmethod
    async def call_deepseek(prompt: str, context: str = "") -> Dict:
        """Appelle DeepSeek"""
        # DeepSeek utilise une API compatible OpenAI
        if not API_KEYS.get("deepseek"):
            return RealModelExecutor._fallback_response("DeepSeek", prompt)
        
        try:
            client = openai.OpenAI(
                api_key=API_KEYS.get("deepseek"),
                base_url="https://api.deepseek.com"
            )
            
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return {
                "response": response.choices[0].message.content,
                "model": "deepseek-chat",
                "tokens_used": response.usage.total_tokens,
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            return {"error": str(e), "response": f"Erreur DeepSeek: {str(e)}"}
    
    @staticmethod
    async def call_mistral(prompt: str, context: str = "") -> Dict:
        """Appelle Mistral"""
        return RealModelExecutor._fallback_response("Mistral", prompt)
    
    @staticmethod
    def _fallback_response(model_name: str, prompt: str) -> Dict:
        """Réponse de secours si l'API n'est pas disponible"""
        return {
            "response": f"[{model_name} - Mode Simulation] Analyse de: '{prompt[:50]}...' - "
                       f"Pour activer l'API réelle, configurez la clé API dans les variables d'environnement.",
            "model": f"{model_name.lower()}-simulated",
            "tokens_used": 150,
            "simulation": True
        }

# Système d'Agents Intelligents
class IntelligentAgent:
    """Agent IA capable d'exécuter des tâches réelles"""
    
    def __init__(self, agent_data: Dict):
        self.agent_id = agent_data['agent_id']
        self.name = agent_data['name']
        self.role = agent_data['role']
        self.specialization = agent_data['specialization']
        self.skills = agent_data['skills']
        self.experience = agent_data['experience_level']
    
    async def execute_task(self, task: Dict) -> Dict:
        """Exécute une tâche en utilisant un vrai modèle IA"""
        
        # Construire le prompt selon le rôle
        system_context = self._build_system_context()
        task_prompt = self._build_task_prompt(task)
        
        # Choisir le modèle optimal selon le rôle
        model_choice = self._select_optimal_model()
        
        # Exécuter avec le vrai modèle
        if model_choice == "claude":
            result = await RealModelExecutor.call_claude(task_prompt, system_context)
        elif model_choice == "chatgpt":
            result = await RealModelExecutor.call_chatgpt(task_prompt, system_context)
        else:
            result = await RealModelExecutor.call_llama(task_prompt, system_context)
        
        # Analyser le résultat
        analysis = self._analyze_result(result, task)
        
        return {
            "task_id": task['task_id'],
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "result": result.get('response', 'Aucune réponse'),
            "model_used": result.get('model', 'unknown'),
            "tokens_used": result.get('tokens_used', 0),
            "analysis": analysis,
            "quality_score": self._calculate_quality_score(result),
            "completed_at": datetime.now().isoformat()
        }
    
    def _build_system_context(self) -> str:
        """Construit le contexte système pour l'agent"""
        return f"""Tu es {self.name}, un agent IA spécialisé en {self.specialization}.
Ton rôle est: {self.role}.
Tes compétences: {', '.join(self.skills)}.
Niveau d'expérience: {self.experience}/10.

Tu dois accomplir les tâches qui te sont assignées avec professionnalisme et expertise."""
    
    def _build_task_prompt(self, task: Dict) -> str:
        """Construit le prompt pour la tâche"""
        return f"""Tâche à accomplir: {task['task_description']}

Priorité: {task['priority']}
Niveau de responsabilité: {task['responsibility_level']}%

Livrables attendus:
{chr(10).join(f"- {d}" for d in task['deliverables'])}

Merci de fournir un travail détaillé et de qualité."""
    
    def _select_optimal_model(self) -> str:
        """Sélectionne le modèle optimal selon le rôle"""
        role_models = {
            "researcher": "claude",  # Claude excellent pour la recherche
            "analyst": "chatgpt",    # GPT-4 bon pour l'analyse
            "writer": "claude",      # Claude excellent en écriture
            "developer": "chatgpt",  # GPT-4 bon en code
            "manager": "chatgpt",    # GPT-4 pour la gestion
            "specialist": "claude"   # Claude pour expertise
        }
        return role_models.get(self.role, "chatgpt")
    
    def _analyze_result(self, result: Dict, task: Dict) -> Dict:
        """Analyse le résultat produit"""
        response_length = len(result.get('response', ''))
        
        return {
            "completeness": "high" if response_length > 500 else "medium" if response_length > 200 else "low",
            "meets_requirements": len(task['deliverables']) > 0,
            "technical_depth": self.experience * 10,
            "recommendations": ["Approfondir l'analyse", "Ajouter des exemples"] if response_length < 300 else []
        }
    
    def _calculate_quality_score(self, result: Dict) -> int:
        """Calcule un score de qualité"""
        base_score = 70
        
        if not result.get('error'):
            base_score += 15
        
        if result.get('tokens_used', 0) > 500:
            base_score += 10
        
        if not result.get('simulation'):
            base_score += 5
        
        return min(base_score, 100)

# Moteur de Décomposition (identique)
class QueryDecomposer:
    
    @staticmethod
    def decompose_query(query: str, mode: ExecutionMode) -> List[Dict]:
        words = query.split()
        complexity = len(words)
        
        if complexity < 10:
            num_steps = 2
        elif complexity < 30:
            num_steps = 3
        elif complexity < 50:
            num_steps = 4
        else:
            num_steps = 5
        
        steps = []
        
        if mode == ExecutionMode.MODEL:
            steps = QueryDecomposer._decompose_for_models(query, num_steps)
        else:
            steps = QueryDecomposer._decompose_for_agents(query, num_steps)
        
        return steps
    
    @staticmethod
    def _decompose_for_models(query: str, num_steps: int) -> List[Dict]:
        model_rotation = [
            ModelType.CLAUDE,
            ModelType.CHATGPT,
            ModelType.LLAMA,
            ModelType.DEEPSEEK,
            ModelType.MISTRAL
        ]
        
        base_steps = [
            {
                "order": 1,
                "name": "Analyse et Compréhension",
                "description": "Analyser la requête et identifier les concepts clés",
                "type": "analysis"
            },
            {
                "order": 2,
                "name": "Recherche d'Informations",
                "description": "Collecter les informations pertinentes",
                "type": "research"
            },
            {
                "order": 3,
                "name": "Traitement et Structuration",
                "description": "Organiser et structurer les données",
                "type": "processing"
            },
            {
                "order": 4,
                "name": "Génération de Réponse",
                "description": "Créer une réponse cohérente",
                "type": "generation"
            },
            {
                "order": 5,
                "name": "Validation et Amélioration",
                "description": "Vérifier et optimiser la réponse",
                "type": "validation"
            }
        ]
        
        steps = []
        for i in range(num_steps):
            step = base_steps[i].copy()
            step['step_id'] = str(uuid.uuid4())
            step['query_context'] = query
            step['assigned_model'] = model_rotation[i % len(model_rotation)]
            step['model_version'] = 'latest'
            step['status'] = 'pending'
            step['result'] = None
            steps.append(step)
        
        return steps
    
    @staticmethod
    def _decompose_for_agents(query: str, num_steps: int) -> List[Dict]:
        base_steps = [
            {
                "order": 1,
                "name": "Recherche Initiale",
                "description": "Rechercher les informations de base",
                "required_role": AgentRole.RESEARCHER
            },
            {
                "order": 2,
                "name": "Analyse Approfondie",
                "description": "Analyser les données collectées",
                "required_role": AgentRole.ANALYST
            },
            {
                "order": 3,
                "name": "Développement de Solution",
                "description": "Développer la solution",
                "required_role": AgentRole.DEVELOPER
            },
            {
                "order": 4,
                "name": "Rédaction du Rapport",
                "description": "Documenter les résultats",
                "required_role": AgentRole.WRITER
            },
            {
                "order": 5,
                "name": "Validation Experte",
                "description": "Validation finale par un expert",
                "required_role": AgentRole.SPECIALIST
            }
        ]
        
        steps = []
        for i in range(num_steps):
            step = base_steps[i].copy()
            step['step_id'] = str(uuid.uuid4())
            step['query_context'] = query
            step['status'] = 'pending'
            step['result'] = None
            steps.append(step)
        
        return steps

# Moteur d'Exécution Amélioré
class ConversationExecutor:
    
    @staticmethod
    async def execute_conversation(request_id: str, steps: List[Dict], mode: ExecutionMode) -> Dict:
        """Exécute la conversation avec de vrais appels API"""
        
        results = []
        accumulated_context = ""
        
        for i, step in enumerate(steps):
            step['status'] = 'running'
            step['started_at'] = datetime.now().isoformat()
            
            if mode == ExecutionMode.MODEL:
                result = await ConversationExecutor._execute_real_model_step(step, accumulated_context)
            else:
                result = await ConversationExecutor._execute_real_agent_step(step, accumulated_context)
            
            step['result'] = result
            step['status'] = 'completed'
            step['completed_at'] = datetime.now().isoformat()
            
            results.append(result)
            
            # Accumuler le contexte pour l'étape suivante
            if 'response' in result:
                accumulated_context += f"\nÉtape {i+1} ({step['name']}): {result['response'][:200]}..."
        
        synthesis = ConversationExecutor._synthesize_results(results, steps)
        
        return {
            "request_id": request_id,
            "steps_results": results,
            "synthesis": synthesis,
            "completed_at": datetime.now().isoformat()
        }
    
    @staticmethod
    async def _execute_real_model_step(step: Dict, context: str) -> Dict:
        """Exécute une étape avec un vrai modèle"""
        
        model = step['assigned_model']
        prompt = f"Étape: {step['name']}\nDescription: {step['description']}\nRequête: {step['query_context']}"
        
        if model == ModelType.CLAUDE:
            result = await RealModelExecutor.call_claude(prompt, context)
        elif model == ModelType.CHATGPT:
            result = await RealModelExecutor.call_chatgpt(prompt, context)
        elif model == ModelType.LLAMA:
            result = await RealModelExecutor.call_llama(prompt, context)
        elif model == ModelType.DEEPSEEK:
            result = await RealModelExecutor.call_deepseek(prompt, context)
        else:
            result = await RealModelExecutor.call_mistral(prompt, context)
        
        return {
            'step_id': step['step_id'],
            'step_name': step['name'],
            'model_used': model,
            'response': result.get('response', ''),
            'confidence': 90 if not result.get('simulation') else 70,
            'tokens_used': result.get('tokens_used', 0),
            'execution_time': 2.5,
            'real_api_call': not result.get('simulation', False)
        }
    
    @staticmethod
    async def _execute_real_agent_step(step: Dict, context: str) -> Dict:
        """Exécute une étape avec un agent réel"""
        
        # Trouver un agent disponible avec le bon rôle
        required_role = step.get('required_role')
        available_agents = [
            agent for agent in AGENTS_DB.values()
            if agent['role'] == required_role and agent['status'] == 'available'
        ]
        
        if not available_agents:
            return {
                'step_id': step['step_id'],
                'step_name': step['name'],
                'agent_role': required_role,
                'error': 'Aucun agent disponible pour ce rôle',
                'work_completed': f"Simulation: {step['description']}",
                'quality_score': 60
            }
        
        agent_data = available_agents[0]
        agent = IntelligentAgent(agent_data)
        
        # Créer une tâche temporaire
        task = {
            'task_id': str(uuid.uuid4()),
            'task_description': f"{step['name']}: {step['description']}\nContexte: {context}",
            'priority': 'high',
            'responsibility_level': 80,
            'deliverables': [step['description']]
        }
        
        result = await agent.execute_task(task)
        
        return {
            'step_id': step['step_id'],
            'step_name': step['name'],
            'agent_id': agent.agent_id,
            'agent_name': agent.name,
            'agent_role': agent.role,
            'work_completed': result['result'],
            'quality_score': result['quality_score'],
            'model_used': result.get('model_used'),
            'tokens_used': result.get('tokens_used', 0),
            'real_agent_execution': True
        }
    
    @staticmethod
    def _synthesize_results(results: List[Dict], steps: List[Dict]) -> Dict:
        synthesis_text = "Synthèse Finale Complète:\n\n"
        
        for i, result in enumerate(results, 1):
            step_name = result.get('step_name', f'Étape {i}')
            if 'response' in result:
                synthesis_text += f"{i}. {step_name}:\n{result['response']}\n\n"
            else:
                synthesis_text += f"{i}. {step_name}:\n{result.get('work_completed', 'Travail terminé')}\n\n"
        
        synthesis_text += "\nConclusion: Analyse séquentielle complète avec appels API réels."
        
        total_tokens = sum(r.get('tokens_used', 0) for r in results)
        real_calls = sum(1 for r in results if r.get('real_api_call') or r.get('real_agent_execution'))
        
        return {
            'synthesis': synthesis_text,
            'total_steps': len(results),
            'overall_confidence': sum(r.get('confidence', r.get('quality_score', 80)) for r in results) / len(results),
            'total_tokens_used': total_tokens,
            'real_api_calls': real_calls,
            'key_findings': [f"Finding {i+1}" for i in range(3)]
        }

# Routes (identiques mais utilisant les nouvelles classes)
# ... [Le reste des routes reste identique] ...
    
    @staticmethod
    def decompose_query(query: str, mode: ExecutionMode) -> List[Dict]:
        """Décompose une requête en étapes"""
        
        # Analyse de la complexité
        words = query.split()
        complexity = len(words)
        
        if complexity < 10:
            num_steps = 2
        elif complexity < 30:
            num_steps = 3
        elif complexity < 50:
            num_steps = 4
        else:
            num_steps = 5
        
        steps = []
        
        if mode == ExecutionMode.MODEL:
            steps = QueryDecomposer._decompose_for_models(query, num_steps)
        else:
            steps = QueryDecomposer._decompose_for_agents(query, num_steps)
        
        return steps
    
    @staticmethod
    def _decompose_for_models(query: str, num_steps: int) -> List[Dict]:
        """Décomposition pour modèles"""
        
        model_rotation = [
            ModelType.CLAUDE,
            ModelType.CHATGPT,
            ModelType.LLAMA,
            ModelType.DEEPSEEK,
            ModelType.MISTRAL
        ]
        
        base_steps = [
            {
                "order": 1,
                "name": "Analyse et Compréhension",
                "description": "Analyser la requête et identifier les concepts clés",
                "type": "analysis"
            },
            {
                "order": 2,
                "name": "Recherche d'Informations",
                "description": "Collecter les informations pertinentes",
                "type": "research"
            },
            {
                "order": 3,
                "name": "Traitement et Structuration",
                "description": "Organiser et structurer les données",
                "type": "processing"
            },
            {
                "order": 4,
                "name": "Génération de Réponse",
                "description": "Créer une réponse cohérente",
                "type": "generation"
            },
            {
                "order": 5,
                "name": "Validation et Amélioration",
                "description": "Vérifier et optimiser la réponse",
                "type": "validation"
            }
        ]
        
        steps = []
        for i in range(num_steps):
            step = base_steps[i].copy()
            step['step_id'] = str(uuid.uuid4())
            step['query_context'] = query
            step['assigned_model'] = model_rotation[i % len(model_rotation)]
            step['model_version'] = 'latest'
            step['status'] = 'pending'
            step['result'] = None
            steps.append(step)
        
        return steps
    
    @staticmethod
    def _decompose_for_agents(query: str, num_steps: int) -> List[Dict]:
        """Décomposition pour agents"""
        
        agent_roles = [
            AgentRole.RESEARCHER,
            AgentRole.ANALYST,
            AgentRole.WRITER,
            AgentRole.DEVELOPER,
            AgentRole.SPECIALIST
        ]
        
        base_steps = [
            {
                "order": 1,
                "name": "Recherche Initiale",
                "description": "Rechercher les informations de base",
                "required_role": AgentRole.RESEARCHER
            },
            {
                "order": 2,
                "name": "Analyse Approfondie",
                "description": "Analyser les données collectées",
                "required_role": AgentRole.ANALYST
            },
            {
                "order": 3,
                "name": "Développement de Solution",
                "description": "Développer la solution",
                "required_role": AgentRole.DEVELOPER
            },
            {
                "order": 4,
                "name": "Rédaction du Rapport",
                "description": "Documenter les résultats",
                "required_role": AgentRole.WRITER
            },
            {
                "order": 5,
                "name": "Validation Experte",
                "description": "Validation finale par un expert",
                "required_role": AgentRole.SPECIALIST
            }
        ]
        
        steps = []
        for i in range(num_steps):
            step = base_steps[i].copy()
            step['step_id'] = str(uuid.uuid4())
            step['query_context'] = query
            step['status'] = 'pending'
            step['result'] = None
            steps.append(step)
        
        return steps

# Moteur d'Exécution
class ConversationExecutor:
    
    @staticmethod
    async def execute_conversation(request_id: str, steps: List[Dict], mode: ExecutionMode) -> Dict:
        """Execute la conversation étape par étape"""
        
        results = []
        
        for step in steps:
            step['status'] = 'running'
            step['started_at'] = datetime.now().isoformat()
            
            if mode == ExecutionMode.MODEL:
                result = await ConversationExecutor._execute_model_step(step)
            else:
                result = await ConversationExecutor._execute_agent_step(step)
            
            step['result'] = result
            step['status'] = 'completed'
            step['completed_at'] = datetime.now().isoformat()
            
            results.append(result)
            
            # Simuler le délai entre étapes
            await asyncio.sleep(1)
        
        # Synthèse finale
        synthesis = ConversationExecutor._synthesize_results(results, steps)
        
        return {
            "request_id": request_id,
            "steps_results": results,
            "synthesis": synthesis,
            "completed_at": datetime.now().isoformat()
        }
    
    @staticmethod
    async def _execute_model_step(step: Dict) -> Dict:
        """Exécute une étape avec un modèle"""
        
        await asyncio.sleep(2)  # Simuler l'appel API
        
        model = step['assigned_model']
        
        responses = {
            'claude': f"Claude analyse: Pour l'étape '{step['name']}', voici mon analyse détaillée...",
            'chatgpt': f"ChatGPT répond: Concernant '{step['name']}', j'ai trouvé que...",
            'llama': f"Llama traite: Pour '{step['name']}', après analyse...",
            'deepseek': f"DeepSeek explore: Sur '{step['name']}', mes recherches indiquent...",
            'mistral': f"Mistral évalue: Relativement à '{step['name']}', il apparaît que..."
        }
        
        return {
            'step_id': step['step_id'],
            'step_name': step['name'],
            'model_used': model,
            'response': responses.get(model, "Réponse générée"),
            'confidence': 85 + (hash(step['step_id']) % 15),
            'tokens_used': 150 + (hash(step['step_id']) % 100),
            'execution_time': 1.5 + (hash(step['step_id']) % 100) / 100
        }
    
    @staticmethod
    async def _execute_agent_step(step: Dict) -> Dict:
        """Exécute une étape avec un agent"""
        
        await asyncio.sleep(2)
        
        role = step.get('required_role', 'specialist')
        
        return {
            'step_id': step['step_id'],
            'step_name': step['name'],
            'agent_role': role,
            'work_completed': f"L'agent {role} a complété: {step['description']}",
            'deliverables': [f"Livrable {i+1}" for i in range(2)],
            'quality_score': 80 + (hash(step['step_id']) % 20),
            'time_spent': 30 + (hash(step['step_id']) % 30)
        }
    
    @staticmethod
    def _synthesize_results(results: List[Dict], steps: List[Dict]) -> Dict:
        """Synthétise tous les résultats"""
        
        synthesis_text = "Synthèse Finale:\n\n"
        
        for i, result in enumerate(results, 1):
            step_name = result.get('step_name', f'Étape {i}')
            if 'response' in result:
                synthesis_text += f"{i}. {step_name}: {result['response'][:100]}...\n\n"
            else:
                synthesis_text += f"{i}. {step_name}: {result['work_completed']}\n\n"
        
        synthesis_text += "\nConclusion: Basé sur l'analyse séquentielle de toutes les étapes, "
        synthesis_text += "la réponse optimale a été construite en combinant les expertises de chaque étape."
        
        return {
            'synthesis': synthesis_text,
            'total_steps': len(results),
            'overall_confidence': sum(r.get('confidence', r.get('quality_score', 80)) for r in results) / len(results),
            'key_findings': [f"Finding {i+1}" for i in range(3)]
        }

# Moteur de Gestion d'Entreprise
class CompanyManager:
    
    @staticmethod
    def create_company(company_data: Dict) -> Dict:
        """Créer une entreprise"""
        
        company_id = str(uuid.uuid4())
        
        company = {
            'company_id': company_id,
            **company_data,
            'created_at': datetime.now().isoformat(),
            'agents': [],
            'active_tasks': 0,
            'completed_tasks': 0,
            'performance_score': 0
        }
        
        COMPANIES_DB[company_id] = company
        
        return company
    
    @staticmethod
    def recruit_agent(company_id: str, agent_data: Dict) -> Dict:
        """Recruter un agent pour l'entreprise"""
        
        if company_id not in COMPANIES_DB:
            raise HTTPException(status_code=404, detail="Entreprise non trouvée")
        
        agent_id = str(uuid.uuid4())
        
        agent = {
            'agent_id': agent_id,
            'company_id': company_id,
            **agent_data,
            'hired_at': datetime.now().isoformat(),
            'tasks_completed': 0,
            'performance_rating': 0,
            'status': 'available'
        }
        
        AGENTS_DB[agent_id] = agent
        COMPANIES_DB[company_id]['agents'].append(agent_id)
        
        return agent
    
    @staticmethod
    async def assign_task(task_data: Dict) -> Dict:
        """Assigner une tâche à un agent"""
        
        company_id = task_data['company_id']
        agent_id = task_data['agent_id']
        
        if company_id not in COMPANIES_DB:
            raise HTTPException(status_code=404, detail="Entreprise non trouvée")
        
        if agent_id not in AGENTS_DB:
            raise HTTPException(status_code=404, detail="Agent non trouvé")
        
        task_id = str(uuid.uuid4())
        
        task = {
            'task_id': task_id,
            **task_data,
            'status': 'assigned',
            'assigned_at': datetime.now().isoformat(),
            'progress': 0
        }
        
        TASKS_DB[task_id] = task
        COMPANIES_DB[company_id]['active_tasks'] += 1
        AGENTS_DB[agent_id]['status'] = 'busy'
        
        # Créer le calendrier
        schedule_id = str(uuid.uuid4())
        schedule = {
            'schedule_id': schedule_id,
            'task_id': task_id,
            'agent_id': agent_id,
            'start_date': task_data['start_date'],
            'end_date': task_data['end_date'],
            'milestones': CompanyManager._generate_milestones(
                task_data['start_date'],
                task_data['end_date']
            )
        }
        
        SCHEDULES_DB[schedule_id] = schedule
        
        return task
    
    @staticmethod
    def _generate_milestones(start_date: str, end_date: str) -> List[Dict]:
        """Générer des jalons"""
        
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        duration = (end - start).days
        
        milestones = []
        for i in range(1, 5):
            milestone_date = start + timedelta(days=(duration * i) // 4)
            milestones.append({
                'milestone': i,
                'date': milestone_date.isoformat(),
                'description': f"Jalon {i}",
                'status': 'pending'
            })
        
        return milestones

# Routes API
@app.post("/api/v1/conversation/start")
async def start_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """Démarrer une conversation dirigée"""
    
    request_id = str(uuid.uuid4())
    
    # Décomposer la requête
    steps = QueryDecomposer.decompose_query(request.query, request.execution_mode)
    
    conversation_data = {
        'request_id': request_id,
        'query': request.query,
        'execution_mode': request.execution_mode,
        'steps': steps,
        'status': 'pending',
        'created_at': datetime.now().isoformat()
    }
    
    REQUESTS_DB[request_id] = conversation_data
    STEPS_DB[request_id] = steps
    
    # Exécuter en arrière-plan
    async def execute():
        conversation_data['status'] = 'running'
        result = await ConversationExecutor.execute_conversation(
            request_id,
            steps,
            request.execution_mode
        )
        conversation_data['result'] = result
        conversation_data['status'] = 'completed'
        RESULTS_DB[request_id] = result
    
    background_tasks.add_task(execute)
    
    return {
        "success": True,
        "request_id": request_id,
        "steps": steps,
        "message": "Conversation démarrée"
    }

@app.put("/api/v1/conversation/{request_id}/step/{step_id}")
async def update_step_config(request_id: str, step_id: str, config: StepConfig):
    """Modifier la configuration d'une étape"""
    
    if request_id not in STEPS_DB:
        raise HTTPException(status_code=404, detail="Requête non trouvée")
    
    steps = STEPS_DB[request_id]
    step = next((s for s in steps if s['step_id'] == step_id), None)
    
    if not step:
        raise HTTPException(status_code=404, detail="Étape non trouvée")
    
    if config.model_type:
        step['assigned_model'] = config.model_type
    if config.model_version:
        step['model_version'] = config.model_version
    if config.agent_id:
        step['agent_id'] = config.agent_id
    
    return {"success": True, "step": step}

@app.get("/api/v1/conversation/{request_id}")
async def get_conversation_status(request_id: str):
    """Récupérer le statut d'une conversation"""
    
    if request_id not in REQUESTS_DB:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    conversation = REQUESTS_DB[request_id]
    steps = STEPS_DB.get(request_id, [])
    result = RESULTS_DB.get(request_id)
    
    return {
        "conversation": conversation,
        "steps": steps,
        "result": result
    }

@app.post("/api/v1/company/create")
async def create_company(company: CompanyCreate):
    """Créer une entreprise"""
    
    company_data = CompanyManager.create_company(company.dict())
    
    return {
        "success": True,
        "company": company_data
    }

@app.post("/api/v1/company/{company_id}/recruit")
async def recruit_agent(company_id: str, agent: AgentCreate):
    """Recruter un agent"""
    
    agent_data = CompanyManager.recruit_agent(company_id, agent.dict())
    
    return {
        "success": True,
        "agent": agent_data
    }

@app.post("/api/v1/company/assign-task")
async def assign_task(task: TaskAssignment):
    """Assigner une tâche"""
    
    task_data = await CompanyManager.assign_task(task.dict())
    
    return {
        "success": True,
        "task": task_data
    }

@app.get("/api/v1/company/{company_id}")
async def get_company(company_id: str):
    """Récupérer les détails d'une entreprise"""
    
    if company_id not in COMPANIES_DB:
        raise HTTPException(status_code=404, detail="Entreprise non trouvée")
    
    company = COMPANIES_DB[company_id]
    agents = [AGENTS_DB[aid] for aid in company['agents'] if aid in AGENTS_DB]
    tasks = [t for t in TASKS_DB.values() if t['company_id'] == company_id]
    
    return {
        "company": company,
        "agents": agents,
        "tasks": tasks
    }

@app.get("/api/v1/agent/{agent_id}/calendar")
async def get_agent_calendar(agent_id: str):
    """Récupérer le calendrier d'un agent"""
    
    if agent_id not in AGENTS_DB:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    schedules = [s for s in SCHEDULES_DB.values() if s['agent_id'] == agent_id]
    
    return {
        "agent_id": agent_id,
        "schedules": schedules
    }

# Gestionnaire d'Entreprise Amélioré
class CompanyManager:
    
    @staticmethod
    def create_company(company_data: Dict) -> Dict:
        company_id = str(uuid.uuid4())
        
        company = {
            'company_id': company_id,
            **company_data,
            'created_at': datetime.now().isoformat(),
            'agents': [],
            'active_tasks': 0,
            'completed_tasks': 0,
            'performance_score': 0
        }
        
        COMPANIES_DB[company_id]['active_tasks'] += 1
        AGENTS_DB[agent_id]['status'] = 'busy'
        
        # Créer le calendrier
        schedule_id = str(uuid.uuid4())
        schedule = {
            'schedule_id': schedule_id,
            'task_id': task_id,
            'agent_id': agent_id,
            'start_date': task_data['start_date'],
            'end_date': task_data['end_date'],
            'milestones': CompanyManager._generate_milestones(
                task_data['start_date'],
                task_data['end_date']
            )
        }
        
        SCHEDULES_DB[schedule_id] = schedule
        
        # Exécuter la tâche immédiatement avec l'agent
        agent_data = AGENTS_DB[agent_id]
        agent = IntelligentAgent(agent_data)
        
        task['status'] = 'in_progress'
        # result = await agent.execute_task(task)
        import asyncio

        async def main():
            task = "Écris un résumé sur l'IA générative."
            result = await agent.execute_task(task)
            print(result)

        asyncio.run(main())

        
        # Sauvegarder le résultat
        RESULTS_DB[task_id] = result
        task['result'] = result
        task['status'] = 'completed'
        task['completed_at'] = datetime.now().isoformat()
        task['progress'] = 100
        
        # Mettre à jour les statistiques
        COMPANIES_DB[company_id]['active_tasks'] -= 1
        COMPANIES_DB[company_id]['completed_tasks'] += 1
        AGENTS_DB[agent_id]['status'] = 'available'
        AGENTS_DB[agent_id]['tasks_completed'] += 1
        AGENTS_DB[agent_id]['performance_rating'] = result['quality_score'] / 20
        
        return task
    
    @staticmethod
    def _generate_milestones(start_date: str, end_date: str) -> List[Dict]:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        duration = (end - start).days
        
        milestones = []
        for i in range(1, 5):
            milestone_date = start + timedelta(days=(duration * i) // 4)
            milestones.append({
                'milestone': i,
                'date': milestone_date.isoformat(),
                'description': f"Jalon {i}",
                'status': 'pending'
            })
        
        return milestones

# Routes API
@app.post("/api/v1/conversation/start")
async def start_conversation(request: ConversationRequest, background_tasks: BackgroundTasks):
    """Démarrer une conversation dirigée"""
    
    request_id = str(uuid.uuid4())
    
    steps = QueryDecomposer.decompose_query(request.query, request.execution_mode)
    
    conversation_data = {
        'request_id': request_id,
        'query': request.query,
        'execution_mode': request.execution_mode,
        'steps': steps,
        'status': 'pending',
        'created_at': datetime.now().isoformat()
    }
    
    REQUESTS_DB[request_id] = conversation_data
    STEPS_DB[request_id] = steps
    
    async def execute():
        conversation_data['status'] = 'running'
        result = await ConversationExecutor.execute_conversation(
            request_id,
            steps,
            request.execution_mode
        )
        conversation_data['result'] = result
        conversation_data['status'] = 'completed'
        RESULTS_DB[request_id] = result
    
    background_tasks.add_task(execute)
    
    return {
        "success": True,
        "request_id": request_id,
        "steps": steps,
        "message": "Conversation démarrée avec appels API réels"
    }

@app.put("/api/v1/conversation/{request_id}/step/{step_id}")
async def update_step_config(request_id: str, step_id: str, config: StepConfig):
    """Modifier la configuration d'une étape"""
    
    if request_id not in STEPS_DB:
        raise HTTPException(status_code=404, detail="Requête non trouvée")
    
    steps = STEPS_DB[request_id]
    step = next((s for s in steps if s['step_id'] == step_id), None)
    
    if not step:
        raise HTTPException(status_code=404, detail="Étape non trouvée")
    
    if config.model_type:
        step['assigned_model'] = config.model_type
    if config.model_version:
        step['model_version'] = config.model_version
    if config.agent_id:
        step['agent_id'] = config.agent_id
    
    return {"success": True, "step": step}

@app.get("/api/v1/conversation/{request_id}")
async def get_conversation_status(request_id: str):
    """Récupérer le statut d'une conversation"""
    
    if request_id not in REQUESTS_DB:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")
    
    conversation = REQUESTS_DB[request_id]
    steps = STEPS_DB.get(request_id, [])
    result = RESULTS_DB.get(request_id)
    
    return {
        "conversation": conversation,
        "steps": steps,
        "result": result
    }

@app.post("/api/v1/company/create")
async def create_company(company: CompanyCreate):
    """Créer une entreprise"""
    
    company_data = CompanyManager.create_company(company.dict())
    
    return {
        "success": True,
        "company": company_data
    }

@app.post("/api/v1/company/{company_id}/recruit")
async def recruit_agent(company_id: str, agent: AgentCreate):
    """Recruter un agent"""
    
    agent_data = CompanyManager.recruit_agent(company_id, agent.dict())
    
    return {
        "success": True,
        "agent": agent_data
    }

@app.post("/api/v1/company/assign-task")
async def assign_task(task: TaskAssignment):
    """Assigner une tâche à un agent (exécution réelle)"""
    
    task_data = await CompanyManager.assign_and_execute_task(task.dict())
    
    return {
        "success": True,
        "task": task_data,
        "message": "Tâche exécutée avec succès par l'agent IA"
    }

@app.get("/api/v1/company/{company_id}")
async def get_company(company_id: str):
    """Récupérer les détails d'une entreprise"""
    
    if company_id not in COMPANIES_DB:
        raise HTTPException(status_code=404, detail="Entreprise non trouvée")
    
    company = COMPANIES_DB[company_id]
    agents = [AGENTS_DB[aid] for aid in company['agents'] if aid in AGENTS_DB]
    tasks = [t for t in TASKS_DB.values() if t['company_id'] == company_id]
    
    return {
        "company": company,
        "agents": agents,
        "tasks": tasks
    }

@app.get("/api/v1/task/{task_id}/result")
async def get_task_result(task_id: str):
    """Récupérer le résultat détaillé d'une tâche"""
    
    if task_id not in RESULTS_DB:
        raise HTTPException(status_code=404, detail="Résultat non trouvé")
    
    result = RESULTS_DB[task_id]
    task = TASKS_DB.get(task_id)
    
    return {
        "task": task,
        "result": result,
        "detailed_analysis": {
            "execution_quality": result.get('quality_score', 0),
            "model_used": result.get('model_used', 'N/A'),
            "tokens_consumed": result.get('tokens_used', 0),
            "agent_performance": result.get('analysis', {})
        }
    }

@app.get("/api/v1/agent/{agent_id}/calendar")
async def get_agent_calendar(agent_id: str):
    """Récupérer le calendrier d'un agent"""
    
    if agent_id not in AGENTS_DB:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    schedules = [s for s in SCHEDULES_DB.values() if s['agent_id'] == agent_id]
    
    return {
        "agent_id": agent_id,
        "schedules": schedules
    }

@app.get("/api/v1/models/status")
async def get_models_status():
    """Vérifier le statut des connexions aux APIs"""
    
    return {
        "openai": {
            "available": OPENAI_AVAILABLE,
            "configured": bool(API_KEYS.get("openai")),
            "status": "ready" if (OPENAI_AVAILABLE and API_KEYS.get("openai")) else "not_configured"
        },
        "anthropic": {
            "available": ANTHROPIC_AVAILABLE,
            "configured": bool(API_KEYS.get("anthropic")),
            "status": "ready" if (ANTHROPIC_AVAILABLE and API_KEYS.get("anthropic")) else "not_configured"
        },
        "together": {
            "available": TOGETHER_AVAILABLE,
            "configured": bool(API_KEYS.get("together")),
            "status": "ready" if (TOGETHER_AVAILABLE and API_KEYS.get("together")) else "not_configured"
        },
        "message": "Configurez les clés API via variables d'environnement pour activer les appels réels"
    }

@app.post("/api/v1/models/configure")
async def configure_api_keys(keys: Dict[str, str]):
    """Configurer les clés API"""
    
    for provider, key in keys.items():
        if provider in API_KEYS:
            API_KEYS[provider] = key
    
    return {
        "success": True,
        "message": "Clés API configurées",
        "configured_providers": [k for k, v in API_KEYS.items() if v]
    }
    
@staticmethod
def recruit_agent(company_id: str, agent_data: Dict) -> Dict:
        if company_id not in COMPANIES_DB:
            raise HTTPException(status_code=404, detail="Entreprise non trouvée")
        
        agent_id = str(uuid.uuid4())
        
        agent = {
            'agent_id': agent_id,
            'company_id': company_id,
            **agent_data,
            'hired_at': datetime.now().isoformat(),
            'tasks_completed': 0,
            'performance_rating': 0,
            'status': 'available'
        }
        
        AGENTS_DB[agent_id] = agent
        COMPANIES_DB[company_id]['agents'].append(agent_id)
        
        return agent
    
@staticmethod
async def assign_and_execute_task(task_data: Dict) -> Dict:
        """Assigner et exécuter une tâche réelle"""
        
        company_id = task_data['company_id']
        agent_id = task_data['agent_id']
        
        if company_id not in COMPANIES_DB:
            raise HTTPException(status_code=404, detail="Entreprise non trouvée")
        
        if agent_id not in AGENTS_DB:
            raise HTTPException(status_code=404, detail="Agent non trouvé")
        
        task_id = str(uuid.uuid4())
        
        task = {
            'task_id': task_id,
            **task_data,
            'status': 'assigned',
            'assigned_at': datetime.now().isoformat(),
            'progress': 0
        }
        
        TASKS_DB[task_id] = task
        COMPANIES_DB
        [company_id] = company
        return company

@app.get("/")
async def root():
    return {
        "message": "AI Conversation Director Platform v2.0",
        "version": "2.0.0",
        "features": [
            "Real API calls to ChatGPT, Claude, Llama",
            "Intelligent AI agents with real task execution",
            "Sequential conversation management",
            "Company management with agent recruitment"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "conversations": len(REQUESTS_DB),
        "companies": len(COMPANIES_DB),
        "agents": len(AGENTS_DB),
        "api_integrations": {
            "openai": OPENAI_AVAILABLE,
            "anthropic": ANTHROPIC_AVAILABLE,
            "together": TOGETHER_AVAILABLE
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("AI CONVERSATION DIRECTOR PLATFORM API v2.0")
    print("=" * 70)
    print("\nFonctionnalités:")
    print("  ✓ Appels API réels (ChatGPT, Claude, Llama)")
    print("  ✓ Agents IA intelligents")
    print("  ✓ Exécution réelle des tâches")
    print("\nConfiguration:")
    print("  Définissez les variables d'environnement:")
    print("    OPENAI_API_KEY=your_key")
    print("    ANTHROPIC_API_KEY=your_key")
    print("    TOGETHER_API_KEY=your_key")
    print("\nAPI démarrée sur http://localhost:8004")
    print("Documentation: http://localhost:8004/docs")
    print("\n" + "=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8017)
 