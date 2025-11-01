#!/bin/bash
# ============================================================
# SCRIPT DE DÉMARRAGE AUTOMATIQUE - AI DEVELOPMENT PLATFORM
# ============================================================
# Fichier: start_platform.sh
# Usage: ./start_platform.sh [mode]
# Modes: dev, prod, docker

set -e

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
cat << "EOF"
    ___    ____   ____                 ____  __      __  ____                 
   /   |  /  _/  / __ \___  _   __   / __ \/ /___ _/ /_/ __/___  _________ ___ 
  / /| |  / /   / / / / _ \| | / /  / /_/ / / __ `/ __/ /_/ __ \/ ___/ __ `__ \
 / ___ |_/ /   / /_/ /  __/| |/ /  / ____/ / /_/ / /_/ __/ /_/ / /  / / / / / /
/_/  |_/___/  /_____/\___/ |___/  /_/   /_/\__,_/\__/_/  \____/_/  /_/ /_/ /_/ 
                                                                                 
EOF
echo -e "${NC}"
echo -e "${GREEN}🚀 AI Development Platform - Démarrage${NC}"
echo ""

# Fonction de vérification des prérequis
check_requirements() {
    echo -e "${YELLOW}📋 Vérification des prérequis...${NC}"
    
    # Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 n'est pas installé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python $(python3 --version)${NC}"
    
    # pip
    if ! command -v pip3 &> /dev/null; then
        echo -e "${RED}❌ pip n'est pas installé${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ pip installé${NC}"
    
    echo ""
}

# Fonction d'installation des dépendances
install_dependencies() {
    echo -e "${YELLOW}📦 Installation des dépendances...${NC}"
    
    # Créer un environnement virtuel si nécessaire
    if [ ! -d "venv" ]; then
        echo -e "${BLUE}Création de l'environnement virtuel...${NC}"
        python3 -m venv venv
    fi
    
    # Activer l'environnement virtuel
    source venv/bin/activate
    
    # Installer les dépendances
    echo -e "${BLUE}Installation via pip...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo -e "${GREEN}✓ Dépendances installées${NC}"
    echo ""
}

# Mode Développement
start_dev_mode() {
    echo -e "${YELLOW}🔧 Mode Développement${NC}"
    echo ""
    
    check_requirements
    install_dependencies
    
    # Activer l'environnement virtuel
    source venv/bin/activate
    
    # Lancer l'API en arrière-plan
    echo -e "${BLUE}Démarrage de l'API Backend...${NC}"
    cd backend
    uvicorn ai_development_platform_api:app --host 0.0.0.0 --port 8001 --reload &
    API_PID=$!
    cd ..
    
    sleep 3
    
    # Vérifier que l'API est démarrée
    if curl -s http://localhost:8001/health > /dev/null; then
        echo -e "${GREEN}✓ API démarrée sur http://localhost:8001${NC}"
        echo -e "${GREEN}  📖 Documentation: http://localhost:8001/docs${NC}"
    else
        echo -e "${RED}❌ Erreur de démarrage de l'API${NC}"
        exit 1
    fi
    
    echo ""
    
    # Lancer Streamlit
    echo -e "${BLUE}Démarrage du Frontend Streamlit...${NC}"
    cd frontend
    streamlit run ai_development_platform_frontend.py --server.port=8501 &
    STREAMLIT_PID=$!
    cd ..
    
    sleep 3
    
    echo -e "${GREEN}✓ Frontend démarré sur http://localhost:8501${NC}"
    echo ""
    
    # Afficher les informations
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           🎉 Plateforme Démarrée avec Succès!         ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📍 URLs d'accès:${NC}"
    echo -e "   🌐 Frontend:      ${GREEN}http://localhost:8501${NC}"
    echo -e "   🔧 API Backend:   ${GREEN}http://localhost:8001${NC}"
    echo -e "   📖 API Docs:      ${GREEN}http://localhost:8001/docs${NC}"
    echo ""
    echo -e "${YELLOW}💡 Appuyez sur Ctrl+C pour arrêter tous les services${NC}"
    echo ""
    
    # Attendre l'interruption
    trap "echo -e '\n${YELLOW}Arrêt des services...${NC}'; kill $API_PID $STREAMLIT_PID 2>/dev/null; echo -e '${GREEN}✓ Services arrêtés${NC}'; exit 0" INT TERM
    
    wait
}

# Mode Production
start_prod_mode() {
    echo -e "${YELLOW}🚀 Mode Production${NC}"
    echo ""
    
    check_requirements
    
    # Variables d'environnement
    if [ ! -f ".env" ]; then
        echo -e "${RED}❌ Fichier .env manquant${NC}"
        echo -e "${YELLOW}Création d'un fichier .env d'exemple...${NC}"
        cat > .env << EOF
DATABASE_URL=postgresql://user:password@localhost:5432/aidevdb
REDIS_URL=redis://localhost:6379/0
OPENAI_API_KEY=your_openai_key_here
AWS_ACCESS_KEY=your_aws_key_here
AWS_SECRET_KEY=your_aws_secret_here
EOF
        echo -e "${GREEN}✓ Fichier .env créé. Veuillez le configurer.${NC}"
        exit 1
    fi
    
    source .env
    
    # Lancer avec gunicorn pour l'API
    echo -e "${BLUE}Démarrage de l'API (production)...${NC}"
    cd backend
    gunicorn ai_development_platform_api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001 --daemon
    cd ..
    
    echo -e "${GREEN}✓ API en production sur port 8001${NC}"
    
    # Lancer Streamlit
    echo -e "${BLUE}Démarrage du Frontend...${NC}"
    cd frontend
    streamlit run ai_development_platform_frontend.py --server.port=8501 --server.headless=true &
    cd ..
    
    echo -e "${GREEN}✓ Frontend démarré${NC}"
    echo ""
    echo -e "${GREEN}🎉 Plateforme en production!${NC}"
}

# Mode Docker
start_docker_mode() {
    echo -e "${YELLOW}🐳 Mode Docker${NC}"
    echo ""
    
    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker n'est pas installé${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose n'est pas installé${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Construction des images Docker...${NC}"
    docker-compose build
    
    echo -e "${BLUE}Démarrage des conteneurs...${NC}"
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}✓ Conteneurs démarrés${NC}"
    echo ""
    
    # Attendre que les services soient prêts
    echo -e "${YELLOW}Attente du démarrage complet...${NC}"
    sleep 10
    
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║        🎉 Plateforme Docker Démarrée avec Succès!     ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}📍 Services disponibles:${NC}"
    echo -e "   🌐 Frontend:      ${GREEN}http://localhost:8501${NC}"
    echo -e "   🔧 API:           ${GREEN}http://localhost:8001${NC}"
    echo -e "   📊 Grafana:       ${GREEN}http://localhost:3000${NC} (admin/admin123)"
    echo -e "   🌸 Flower:        ${GREEN}http://localhost:5555${NC}"
    echo -e "   📈 MLflow:        ${GREEN}http://localhost:5000${NC}"
    echo ""
    echo -e "${YELLOW}💡 Commandes utiles:${NC}"
    echo -e "   • Voir les logs:      ${BLUE}docker-compose logs -f${NC}"
    echo -e "   • Arrêter:            ${BLUE}docker-compose down${NC}"
    echo -e "   • Redémarrer:         ${BLUE}docker-compose restart${NC}"
    echo ""
}

# Fonction d'arrêt
stop_platform() {
    echo -e "${YELLOW}🛑 Arrêt de la plateforme...${NC}"
    
    # Tuer les processus
    pkill -f "uvicorn ai_development_platform_api"
    pkill -f "streamlit run ai_development_platform_frontend"
    
    # Si Docker
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
    fi
    
    echo -e "${GREEN}✓ Plateforme arrêtée${NC}"
}

# Fonction de status
check_status() {
    echo -e "${BLUE}📊 Status de la plateforme${NC}"
    echo ""
    
    # API
    if curl -s http://localhost:8001/health > /dev/null; then
        echo -e "${GREEN}✓ API: Running${NC} (http://localhost:8001)"
    else
        echo -e "${RED}✗ API: Stopped${NC}"
    fi
    
    # Frontend
    if curl -s http://localhost:8501 > /dev/null; then
        echo -e "${GREEN}✓ Frontend: Running${NC} (http://localhost:8501)"
    else
        echo -e "${RED}✗ Frontend: Stopped${NC}"
    fi
    
    echo ""
}

# Menu principal
show_menu() {
    echo -e "${BLUE}Choisissez un mode de démarrage:${NC}"
    echo "  1) Mode Développement (local, reload automatique)"
    echo "  2) Mode Production (optimisé, sans reload)"
    echo "  3) Mode Docker (conteneurs, complet)"
    echo "  4) Vérifier le status"
    echo "  5) Arrêter la plateforme"
    echo "  6) Quitter"
    echo ""
    read -p "Votre choix [1-6]: " choice
    
    case $choice in
        1) start_dev_mode ;;
        2) start_prod_mode ;;
        3) start_docker_mode ;;
        4) check_status ;;
        5) stop_platform ;;
        6) echo -e "${GREEN}Au revoir!${NC}"; exit 0 ;;
        *) echo -e "${RED}Choix invalide${NC}"; show_menu ;;
    esac
}

# Point d'entrée
if [ $# -eq 0 ]; then
    show_menu
else
    case $1 in
        dev) start_dev_mode ;;
        prod) start_prod_mode ;;
        docker) start_docker_mode ;;
        status) check_status ;;
        stop) stop_platform ;;
        *) echo -e "${RED}Usage: $0 [dev|prod|docker|status|stop]${NC}"; exit 1 ;;
    esac
fi

# ============================================================
# DOCKERFILE - BACKEND
# ============================================================
# Fichier: backend/Dockerfile

# FROM python:3.11-slim
# 
# WORKDIR /app
# 
# # Installation des dépendances système
# RUN apt-get update && apt-get install -y \
#     gcc \
#     postgresql-client \
#     && rm -rf /var/lib/apt/lists/*
# 
# # Copier requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# 
# # Copier l'application
# COPY . .
# 
# EXPOSE 8001
# 
# CMD ["uvicorn", "ai_development_platform_api:app", "--host", "0.0.0.0", "--port", "8001"]

# ============================================================
# DOCKERFILE - FRONTEND
# ============================================================
# Fichier: frontend/Dockerfile

# FROM python:3.11-slim
# 
# WORKDIR /app
# 
# # Copier requirements
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# 
# # Copier l'application
# COPY . .
# 
# EXPOSE 8501
# 
# CMD ["streamlit", "run", "ai_development_platform_frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]