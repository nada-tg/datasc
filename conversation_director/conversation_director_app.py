"""
conversation_director_frontend.py - Interface Streamlit

Lancement:
streamlit run conversation_director_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="AI Conversation Director",
    page_icon="🎯",
    layout="wide"
)

API_URL = "http://localhost:8017"

def init_session():
    if 'current_request' not in st.session_state:
        st.session_state.current_request = None
    if 'current_company' not in st.session_state:
        st.session_state.current_company = None

def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# PAGE: Dashboard
def page_dashboard():
    st.title("AI Conversation Director")
    st.write("Gestion intelligente de conversations IA multi-étapes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Nouvelle Conversation", type="primary", use_container_width=True):
            st.session_state.active_view = 'new_conversation'
            st.rerun()
    
    with col2:
        if st.button("Gérer mon Entreprise", use_container_width=True):
            st.session_state.active_view = 'company'
            st.rerun()
    
    with col3:
        if st.button("Voir Historique", use_container_width=True):
            st.session_state.active_view = 'history'
            st.rerun()

# PAGE: Nouvelle Conversation
def page_new_conversation():
    st.title("Nouvelle Conversation")
    
    execution_mode = st.radio(
        "Mode d'Exécution",
        ["model", "agent"],
        format_func=lambda x: "Modèles IA" if x == "model" else "Agents IA",
        horizontal=True
    )
    
    query = st.text_area(
        "Votre Requête",
        height=200,
        placeholder="Entrez une requête complexe qui sera décomposée en étapes..."
    )
    
    context = st.text_area("Contexte (optionnel)", height=100)
    
    auto_assign = st.checkbox("Assignment automatique des modèles/agents", value=True)
    
    if st.button("Décomposer et Lancer", type="primary"):
        if query:
            with st.spinner("Décomposition en cours..."):
                payload = {
                    "query": query,
                    "execution_mode": execution_mode,
                    "context": context if context else None,
                    "auto_assign_models": auto_assign
                }
                
                try:
                    response = requests.post(f"{API_URL}/api/v1/conversation/start", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Requête décomposée!")
                        
                        st.session_state.current_request = result['request_id']
                        st.session_state.active_view = 'conversation_steps'
                        st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
        else:
            st.warning("Veuillez entrer une requête")

# PAGE: Étapes de Conversation
def page_conversation_steps():
    request_id = st.session_state.current_request
    
    if not request_id:
        st.warning("Aucune conversation active")
        return
    
    try:
        response = requests.get(f"{API_URL}/api/v1/conversation/{request_id}")
        
        if response.status_code == 200:
            data = response.json()
            conversation = data['conversation']
            steps = data['steps']
            result = data.get('result')
            
            st.title("Étapes de la Conversation")
            st.write(f"**Requête:** {conversation['query']}")
            st.write(f"**Mode:** {conversation['execution_mode']}")
            st.write(f"**Status:** {conversation['status']}")
            
            st.write("---")
            
            # Afficher les étapes
            st.subheader("Étapes Décomposées")
            
            for step in steps:
                with st.expander(f"Étape {step['order']}: {step['name']}", expanded=True):
                    st.write(f"**Description:** {step['description']}")
                    st.write(f"**Type:** {step.get('type', 'N/A')}")
                    st.write(f"**Status:** {step['status']}")
                    
                    # Configuration du modèle/agent
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if conversation['execution_mode'] == 'model':
                            current_model = step.get('assigned_model', 'claude')
                            new_model = st.selectbox(
                                "Modèle",
                                ["chatgpt", "claude", "llama", "deepseek", "mistral", "gemini"],
                                index=["chatgpt", "claude", "llama", "deepseek", "mistral", "gemini"].index(current_model),
                                key=f"model_{step['step_id']}"
                            )
                            
                            if new_model != current_model and step['status'] == 'pending':
                                if st.button(f"Changer pour {new_model}", key=f"change_{step['step_id']}"):
                                    config_payload = {
                                        "step_id": step['step_id'],
                                        "model_type": new_model,
                                        "model_version": "latest"
                                    }
                                    
                                    try:
                                        requests.put(
                                            f"{API_URL}/api/v1/conversation/{request_id}/step/{step['step_id']}",
                                            json=config_payload
                                        )
                                        st.success("Modèle modifié!")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Erreur: {str(e)}")
                    
                    with col2:
                        version = st.text_input("Version", "latest", key=f"version_{step['step_id']}")
                    
                    # Résultat final
            if result:
                st.write("---")
                st.subheader("Résultat Final")
                
                synthesis = result['synthesis']
                
                st.success(synthesis['synthesis'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Étapes Complétées", synthesis['total_steps'])
                with col2:
                    st.metric("Confiance Globale", f"{synthesis['overall_confidence']:.1f}%")
                with col3:
                    st.write("**Découvertes Clés:**")
                    for finding in synthesis['key_findings']:
                        st.write(f"- {finding}")
            
            elif conversation['status'] == 'running':
                st.info("Conversation en cours d'exécution...")
                if st.button("Rafraîchir"):
                    st.rerun()
            
            elif conversation['status'] == 'pending':
                if st.button("Lancer l'Exécution", type="primary"):
                    st.info("Exécution lancée!")
                    time.sleep(2)
                    st.rerun()
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Gestion Entreprise
def page_company():
    st.title("Gestion d'Entreprise")
    
    company_id = st.session_state.current_company
    
    if not company_id:
        st.info("Aucune entreprise créée")
        
        with st.form("create_company_form"):
            st.subheader("Créer votre Entreprise")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Nom de l'Entreprise")
                industry = st.selectbox("Industrie", [
                    "Technology", "Finance", "Healthcare", "Education", 
                    "Manufacturing", "Services", "Retail"
                ])
            
            with col2:
                ceo_name = st.text_input("Nom du CEO")
                description = st.text_area("Description")
            
            submitted = st.form_submit_button("Créer Entreprise", type="primary")
            
            if submitted:
                if name and ceo_name:
                    payload = {
                        "name": name,
                        "industry": industry,
                        "description": description,
                        "ceo_name": ceo_name
                    }
                    
                    try:
                        response = requests.post(f"{API_URL}/api/v1/company/create", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Entreprise créée!")
                            st.session_state.current_company = result['company']['company_id']
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
                else:
                    st.error("Nom et CEO requis")
        
        return
    
    # Afficher l'entreprise
    try:
        response = requests.get(f"{API_URL}/api/v1/company/{company_id}")
        
        if response.status_code == 200:
            data = response.json()
            company = data['company']
            agents = data['agents']
            tasks = data['tasks']
            
            st.write(f"## {company['name']}")
            st.write(f"**Industrie:** {company['industry']}")
            st.write(f"**CEO:** {company['ceo_name']}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Agents", len(agents))
            with col2:
                st.metric("Tâches Actives", company['active_tasks'])
            with col3:
                st.metric("Tâches Complétées", company['completed_tasks'])
            with col4:
                st.metric("Performance", f"{company['performance_score']}/100")
            
            st.write("---")
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["Agents", "Tâches", "Recruter"])
            
            with tab1:
                st.subheader("Mes Agents")
                
                if not agents:
                    st.info("Aucun agent recruté")
                else:
                    for agent in agents:
                        with st.expander(f"{agent['name']} - {agent['role']}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**Spécialisation:** {agent['specialization']}")
                                st.write(f"**Status:** {agent['status']}")
                            
                            with col2:
                                st.write(f"**Niveau:** {agent['experience_level']}/10")
                                st.write(f"**Tâches:** {agent['tasks_completed']}")
                            
                            with col3:
                                st.write(f"**Rating:** {agent['performance_rating']}/5")
                                
                                if st.button("Voir Calendrier", key=f"cal_{agent['agent_id']}"):
                                    st.session_state.current_agent = agent['agent_id']
                                    st.session_state.active_view = 'agent_calendar'
                                    st.rerun()
            
            with tab2:
                st.subheader("Tâches")
                
                if not tasks:
                    st.info("Aucune tâche assignée")
                else:
                    for task in tasks:
                        with st.expander(f"{task['task_description'][:50]}... - {task['priority']}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Agent:** {task['agent_id']}")
                                st.write(f"**Priorité:** {task['priority']}")
                                st.write(f"**Responsabilité:** {task['responsibility_level']}%")
                            
                            with col2:
                                st.write(f"**Début:** {task['start_date'][:10]}")
                                st.write(f"**Fin:** {task['end_date'][:10]}")
                                st.write(f"**Status:** {task['status']}")
                            
                            st.progress(task.get('progress', 0) / 100)
            
            with tab3:
                st.subheader("Recruter un Agent")
                
                with st.form("recruit_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        agent_name = st.text_input("Nom de l'Agent")
                        role = st.selectbox("Rôle", [
                            "researcher", "analyst", "writer", 
                            "developer", "manager", "specialist"
                        ])
                        specialization = st.text_input("Spécialisation")
                    
                    with col2:
                        experience = st.slider("Niveau d'Expérience", 1, 10, 5)
                        skills = st.text_input("Compétences (séparées par virgule)")
                    
                    if st.form_submit_button("Recruter", type="primary"):
                        if agent_name and specialization:
                            payload = {
                                "name": agent_name,
                                "role": role,
                                "specialization": specialization,
                                "skills": [s.strip() for s in skills.split(',') if s.strip()],
                                "experience_level": experience
                            }
                            
                            try:
                                resp = requests.post(
                                    f"{API_URL}/api/v1/company/{company_id}/recruit",
                                    json=payload
                                )
                                
                                if resp.status_code == 200:
                                    st.success("Agent recruté!")
                                    time.sleep(1)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
            
            # Assigner une tâche
            st.write("---")
            st.subheader("Assigner une Tâche")
            
            if agents:
                with st.form("assign_task_form"):
                    task_desc = st.text_area("Description de la Tâche")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        agent_options = {a['agent_id']: a['name'] for a in agents}
                        selected_agent = st.selectbox(
                            "Agent",
                            options=list(agent_options.keys()),
                            format_func=lambda x: agent_options[x]
                        )
                    
                    with col2:
                        priority = st.selectbox("Priorité", ["low", "medium", "high", "urgent"])
                    
                    with col3:
                        responsibility = st.slider("Niveau de Responsabilité (%)", 1, 100, 50)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_date = st.date_input("Date de Début")
                    
                    with col2:
                        end_date = st.date_input("Date de Fin")
                    
                    deliverables = st.text_area("Livrables (un par ligne)")
                    
                    if st.form_submit_button("Assigner Tâche", type="primary"):
                        if task_desc and selected_agent:
                            payload = {
                                "company_id": company_id,
                                "agent_id": selected_agent,
                                "task_description": task_desc,
                                "priority": priority,
                                "start_date": start_date.isoformat(),
                                "end_date": end_date.isoformat(),
                                "responsibility_level": responsibility,
                                "deliverables": [d.strip() for d in deliverables.split('\n') if d.strip()]
                            }
                            
                            try:
                                resp = requests.post(
                                    f"{API_URL}/api/v1/company/assign-task",
                                    json=payload
                                )
                                
                                if resp.status_code == 200:
                                    st.success("Tâche assignée!")
                                    time.sleep(1)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Erreur: {str(e)}")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# PAGE: Calendrier Agent
def page_agent_calendar():
    agent_id = st.session_state.get('current_agent')
    
    if not agent_id:
        st.warning("Aucun agent sélectionné")
        return
    
    st.title("Calendrier de l'Agent")
    
    try:
        response = requests.get(f"{API_URL}/api/v1/agent/{agent_id}/calendar")
        
        if response.status_code == 200:
            data = response.json()
            schedules = data['schedules']
            
            if not schedules:
                st.info("Aucun planning")
                return
            
            for schedule in schedules:
                st.subheader(f"Tâche: {schedule['task_id'][:8]}...")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Début:** {schedule['start_date'][:10]}")
                    st.write(f"**Fin:** {schedule['end_date'][:10]}")
                
                with col2:
                    st.write("**Jalons:**")
                    for milestone in schedule['milestones']:
                        status_icon = "✅" if milestone['status'] == 'completed' else "⏳"
                        st.write(f"{status_icon} Jalon {milestone['milestone']} - {milestone['date'][:10]}")
                
                st.write("---")
    
    except Exception as e:
        st.error(f"Erreur: {str(e)}")

# Navigation
def main():
    init_session()
    
    with st.sidebar:
        st.title("AI Director")
        
        menu = {
            "Dashboard": "dashboard",
            "Nouvelle Conversation": "new_conversation",
            "Étapes Conversation": "conversation_steps",
            "Mon Entreprise": "company",
            "Calendrier Agent": "agent_calendar"
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
        
        st.caption("AI Conversation Director v1.0")
    
    view = st.session_state.get('active_view', 'dashboard')
    
    if view == 'dashboard':
        page_dashboard()
    elif view == 'new_conversation':
        page_new_conversation()
    elif view == 'conversation_steps':
        page_conversation_steps()
    elif view == 'company':
        page_company()
    elif view == 'agent_calendar':
        page_agent_calendar()

if __name__ == "__main__":
    main()