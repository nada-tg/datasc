.PHONY: install start stop restart logs clean help

install:
	@echo "🔧 Installation d'AutoSci Pipeline..."
	powershell -Command "Get-ChildItem -Filter '*.ps1' | ForEach-Object { $_.Attributes = 'Normal' }"
	powershell -File start_all_services.ps1

start:
	@echo "🚀 Démarrage des services..."
	powershell -File start_all_services.ps1

stop:
	@echo "🛑 Arrêt des services..."
	powershell -File stop_all_services.ps1

restart:
	@echo "🔄 Redémarrage des services..."
	powershell -File stop_all_services.ps1
	timeout /t 5
	powershell -File start_all_services.ps1

logs:
	@echo "📋 Consultation des logs..."
	powershell -File view_logs.ps1

clean:
	@echo "🧹 Nettoyage..."
	docker-compose down -v || exit 0
	docker system prune -f
	del /Q logs\*.pid
	if exist temp rmdir /S /Q temp

help:
	@echo "AutoSci Pipeline - Commandes disponibles:"
	@echo "  make install  - Installation complète"
	@echo "  make start    - Démarrer les services"
	@echo "  make stop     - Arrêter les services"
	@echo "  make restart  - Redémarrer les services"
	@echo "  make logs     - Voir les logs"
	@echo "  make clean    - Nettoyer le système"





# .PHONY: install start stop restart logs health backup clean help

# install:
# 	@echo "🔧 Installation d'AutoSci Pipeline..."
# 	chmod +x *.sh
# 	./start_all_services.sh

# start:
# 	@echo "🚀 Démarrage des services..."
# 	./start_all_services.sh

# stop:
# 	@echo "🛑 Arrêt des services..."
# 	./stop_all_services.sh

# restart:
# 	@echo "🔄 Redémarrage des services..."
# 	./stop_all_services.sh && sleep 5 && ./start_all_services.sh

# logs:
# 	@echo "📋 Consultation des logs..."
# 	./view_logs.sh

# clean:
# 	@echo "🧹 Nettoyage..."
# 	docker-compose down -v 2>/dev/null || true
# 	docker system prune -f
# 	rm -rf logs/*.pid temp/

# help:
# 	@echo "AutoSci Pipeline - Commandes disponibles:"
# 	@echo "  make install  - Installation complète"
# 	@echo "  make start    - Démarrer les services"
# 	@echo "  make stop     - Arrêter les services"
# 	@echo "  make restart  - Redémarrer les services"
# 	@echo "  make logs     - Voir les logs"
# 	@echo "  make clean    - Nettoyer le système"