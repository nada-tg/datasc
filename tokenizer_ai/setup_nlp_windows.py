import sys
import subprocess
import platform
import os
import urllib.request

def run_cmd(cmd):
    """Exécute une commande shell et affiche la sortie"""
    print(f"\n➡️  Exécution : {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def download_file(url, dest):
    print(f"➡️ Téléchargement : {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"✅ Fichier téléchargé : {dest}")

def install_whl(file_path):
    run_cmd(f"pip install {file_path}")

def install_packages():
    print("\n=== Installation des bibliothèques principales ===")
    packages = "sentencepiece spacy polyglot langdetect textstat nltk fasttext"
    run_cmd(f"pip install -U {packages}")

def download_nltk_resources():
    import nltk
    print("\n=== Téléchargement des ressources NLTK ===")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

def download_spacy_models():
    print("\n=== Téléchargement des modèles spaCy ===")
    run_cmd("python -m spacy download fr_core_news_sm")
    run_cmd("python -m spacy download en_core_web_sm")

def download_polyglot_resources():
    print("\n=== Téléchargement des ressources Polyglot ===")
    # Français
    run_cmd("polyglot download embeddings2.fr")
    run_cmd("polyglot download ner2.fr")
    run_cmd("polyglot download sentiment2.fr")
    # Anglais
    run_cmd("polyglot download embeddings2.en")
    run_cmd("polyglot download ner2.en")
    run_cmd("polyglot download sentiment2.en")

def install_pyicu_pycld2():
    print("\n=== Installation pyicu et pycld2 (Windows) ===")
    py_version = f"{sys.version_info.major}{sys.version_info.minor}"
    arch = platform.architecture()[0]
    wheel_arch = "win_amd64" if arch == "64bit" else "win32"

    # URLs des wheels précompilés (Gohlke)
    pyicu_url = f"https://download.lfd.uci.edu/pythonlibs/w4tscw6l/pyicu-2.12-cp{py_version}-cp{py_version}-{wheel_arch}.whl"
    pycld2_url = f"https://download.lfd.uci.edu/pythonlibs/w4tscw6l/pycld2-0.41-cp{py_version}-cp{py_version}-{wheel_arch}.whl"

    # Dossiers temporaires pour télécharger
    temp_dir = os.path.join(os.getcwd(), "temp_wheels")
    os.makedirs(temp_dir, exist_ok=True)
    pyicu_file = os.path.join(temp_dir, os.path.basename(pyicu_url))
    pycld2_file = os.path.join(temp_dir, os.path.basename(pycld2_url))

    # Téléchargement
    download_file(pyicu_url, pyicu_file)
    download_file(pycld2_url, pycld2_file)

    # Installation
    install_whl(pyicu_file)
    install_whl(pycld2_file)

if __name__ == "__main__":
    print("\n=== Début de l'installation NLP complète sur Windows ===")
    install_pyicu_pycld2()
    install_packages()
    download_nltk_resources()
    download_spacy_models()
    download_polyglot_resources()
    print("\n✅ Installation NLP complète terminée !")
