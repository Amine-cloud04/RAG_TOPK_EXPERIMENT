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
5. teste plusieurs valeurs de Top-K : `1`, `2`, `3`, `5`, `7`, `10`, `15`, `20` ;
6. envoie le contexte recupere a un modele Groq ;
7. mesure la latence, la memoire, la qualite et la precision du retrieval ;
8. sauvegarde les resultats dans `results/`.

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
results/rag_results_evaluated.csv
results/topk_summary.csv
results/latency_vs_top_k.png
results/source_frequency.png
results/quality_vs_top_k.png
results/quality_latency_tradeoff.png
results/retrieval_precision_recall_vs_top_k.png
results/memory_vs_top_k.png
results/rapport_experimentation_topk.txt
```

Le rapport final `results/rapport_experimentation_topk.txt` conclut automatiquement
sur l'impact d'un K faible et d'un K eleve, le compromis qualite/performance et la
valeur de Top-K la plus pertinente.

## Analyser des resultats existants

Si `results/rag_results.csv` existe deja, vous pouvez regenerer uniquement l'analyse
et le rapport sans relancer les appels au modele :

```powershell
python analyze_topk_results.py
```

Ce script produit :

```text
results/topk_summary.csv
results/rag_results_evaluated.csv
results/quality_vs_top_k.png
results/quality_latency_tradeoff.png
results/retrieval_precision_recall_vs_top_k.png
results/memory_vs_top_k.png
results/rapport_experimentation_topk.txt
```

## Evaluation avancee

Le projet mesure maintenant :

- **Qualite** : score automatique, BLEU, ROUGE-L, similarite semantique,
  answer relevance et faithfulness proxy ;
- **Performance** : latence totale, latence retrieval, latence generation,
  taille du contexte et memoire utilisee ;
- **Precision du retrieval** : verification de la source attendue,
  precision@K et recall@K.

Un jeu de questions avec reponses de reference et sources attendues est defini dans :

```text
evaluation_dataset.py
```

## Evaluation optionnelle avec RAGAS

Apres avoir lance `main.py` ou `analyze_topk_results.py`, vous pouvez calculer des
scores RAGAS officiels si votre environnement est configure pour l'evaluation LLM :

```powershell
python evaluate_with_ragas.py
```

Ce script produit :

```text
results/ragas_scores.csv
```

RAGAS permet notamment d'evaluer la faithfulness, l'answer relevancy, la context
precision et la context recall. Si l'environnement RAGAS ou les cles API necessaires
ne sont pas disponibles, les metriques locales du projet restent utilisables.

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

Le projet calcule un score final qui combine :

- la qualite moyenne des reponses ;
- le taux de questions repondues ;
- la precision du retrieval ;
- la latence moyenne.

La meilleure valeur de Top-K est celle qui obtient le meilleur score global, pas
necessairement celle qui est seulement la plus rapide ou seulement la plus complete.

## Configuration recommandee pour l'experience finale

Pour un projet academique faisable avec Groq et un ordinateur local sans GPU, la
configuration recommandee est :

```text
Documents: 15
Chunks: environ 1100
Questions evaluees: 30
Valeurs de K: 1, 2, 3, 5, 7, 10, 15, 20
Delai entre appels Groq: 2 secondes
```

Le fichier `evaluation_dataset.py` contient une banque de 50 questions. Par defaut,
`main.py` utilise les 30 premieres pour eviter de depasser trop vite les limites de
l'API.

Vous pouvez modifier ce comportement avec :

```powershell
$env:EVALUATION_QUESTION_LIMIT="50"
$env:GROQ_CALL_DELAY_SECONDS="2"
python main.py
```

## Ameliorations avancees

### Test t de Student

L'analyse genere automatiquement un test t apparie entre le Top-K recommande et
les autres valeurs de K :

```text
results/student_t_tests.csv
```

Ce test compare le score final par question et permet de verifier si l'ecart entre
deux valeurs de K est statistiquement significatif sur l'echantillon.

### Evaluation humaine

Pour ajouter une evaluation humaine sur 20 questions :

```powershell
python scripts\generate_human_eval_template.py
```

Remplir ensuite dans `results/human_evaluation_template.csv` :

```text
Human Relevance Score
Human Faithfulness Score
Human Clarity Score
```

avec des notes de 1 a 5, puis lancer :

```powershell
python scripts\summarize_human_eval.py
```

Le resume est sauvegarde dans :

```text
results/human_evaluation_summary.csv
```

### Tester deux tailles de chunks

Pour preparer deux experiences avec deux tailles de chunks :

```powershell
python scripts\chunk_size_plan.py
```

Le script genere `results/chunk_size_plan.csv` avec les commandes a lancer. Chaque
experience utilise un dossier de resultats separe, par exemple `results_chunk_500`.

### Deuxieme corpus

Le code supporte un deuxieme corpus sans modification. Placer les fichiers `.txt`
dans un autre dossier, puis lancer :

```powershell
$env:DATA_FOLDER="data_corpus2"
$env:RESULTS_FOLDER="results_corpus2"
python main.py
```

### Deuxieme LLM avec Gemini

Le code supporte aussi Gemini comme second modele :

```powershell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="votre_cle_gemini"
$env:GEMINI_MODEL="gemini-1.5-flash"
$env:RESULTS_FOLDER="results_gemini"
python main.py
```

Pour revenir a Groq :

```powershell
$env:LLM_PROVIDER="groq"
python main.py
```

## Modele utilise

Le modele utilise par defaut est :

```text
llama-3.3-70b-versatile
```

Il peut etre remplace via la variable :

```powershell
$env:GROQ_MODEL="nom_du_modele"
```

