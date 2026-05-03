# Experience RAG Top-K

Ce projet teste l'impact du parametre **Top-K** dans un pipeline de Retrieval-Augmented Generation (RAG).

L'objectif est de comparer plusieurs valeurs de `K` pour observer leur effet sur :

- la qualite des reponses generees ;
- les sources recuperees ;
- la latence moyenne ;
- le compromis entre precision et temps de reponse.

## Structure du projet

```text
data/
scripts/
results/
data_collector.py
main.py
requirements.txt
```

## Donnees

Le dossier `data/` contient des documents texte sur l'intelligence artificielle, le RAG, les LLM, les bases vectorielles, les transformers et le prompt engineering.

Ces donnees servent de base documentaire pour la recherche semantique.

Le fichier `data_collector.py` permet de reconstruire ou enrichir ce dossier a partir de pages Wikipedia et d'articles web.

## Fonctionnement

Le script principal `main.py` :

1. charge les fichiers `.txt` du dossier `data/` ;
2. decoupe les documents en chunks ;
3. genere des embeddings avec `sentence-transformers` ;
4. stocke les chunks dans une base vectorielle ChromaDB ;
5. teste plusieurs valeurs de Top-K : `1`, `3`, `5`, `10` ;
6. envoie le contexte recupere a un modele Groq ;
7. sauvegarde les resultats dans `results/`.

## Installation

Installer les dependances :

```powershell
pip install -r requirements.txt
```

Si vous utilisez l'environnement Conda `safran` :

```powershell
& 'C:\Users\lenovo\anaconda3\envs\safran\python.exe' -m pip install -r requirements.txt
```

## Cle API Groq

Le projet attend une variable d'environnement appelee `GROQ_API_KEY`.

Vous pouvez la definir dans PowerShell :

```powershell
$env:GROQ_API_KEY="votre_cle_groq"
```

Le projet supporte aussi un fichier local `.env` :

```text
GROQ_API_KEY=votre_cle_groq
```

Le fichier `.env` est ignore par Git afin d'eviter de publier une cle API.

## Lancer l'experience

```powershell
python main.py
```

Avec l'environnement `safran` :

```powershell
& 'C:\Users\lenovo\anaconda3\envs\safran\python.exe' main.py
```

Les resultats sont sauvegardes dans :

```text
results/rag_results.csv
results/latency_vs_top_k.png
results/source_frequency.png
```

## Visualiser le premier resultat

Avant d'analyser finement l'effet du Top-K, un script se concentre sur le resultat de base avec `K=1` :

```powershell
python scripts\visualize_first_result.py
```

Il genere :

```text
results/baseline_k1_summary.csv
results/baseline_k1_latency_by_question.png
results/baseline_k1_answer_coverage.png
results/baseline_k1_source_frequency.png
```

## Interpretation attendue

Une valeur faible de `K`, comme `K=1`, donne souvent une latence plus faible mais peut manquer de contexte.

Une valeur plus elevee, comme `K=10`, donne plus de contexte au modele, mais augmente :

- la taille du prompt ;
- le risque de bruit documentaire ;
- la latence ;
- la probabilite d'atteindre les limites de l'API.

L'objectif final est de trouver un bon compromis entre qualite de reponse et cout/latence.

## Modele utilise

Le modele utilise par defaut est :

```text
llama-3.3-70b-versatile
```

Il peut etre remplace via la variable :

```powershell
$env:GROQ_MODEL="nom_du_modele"
```

