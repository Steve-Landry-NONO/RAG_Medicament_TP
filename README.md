# TP RAG Medicaments

Assistant d'information sur les medicaments construit dans le cadre du TP "Construire un RAG avec Python et Groq".

Projet realise par :

- Steve Landry KOUOKAM NONO
- Ludovic TUEKAM

## 1. Presentation du projet

Ce projet implemente un systeme RAG, pour Retrieval-Augmented Generation, applique aux notices de medicaments issues de la base ANSM/CIS_RCP.

L'objectif est de permettre a un utilisateur de poser des questions en langage naturel sur des medicaments courants, par exemple :

- Quelle est la posologie du Doliprane adulte ?
- Quels sont les effets indesirables de l'ibuprofene ?
- Y a-t-il des interactions avec l'ibuprofene ?
- Quelles sont les contre-indications de l'aspirine ?

Le systeme recherche d'abord les passages les plus pertinents dans une base vectorielle FAISS, puis transmet ces passages a un modele de langage via l'API Groq afin de generer une reponse structuree, sourcee et prudente.

Ce projet est volontairement realise sans LangChain ni LlamaIndex, afin de comprendre et maitriser chaque brique technique du pipeline RAG.

## 2. Objectifs pedagogiques

Ce TP vise a comprendre et implementer les etapes principales d'un RAG :

1. Charger une base de connaissances reelle.
2. Nettoyer les donnees.
3. Transformer les documents en textes exploitables.
4. Decouper les textes en chunks coherents.
5. Calculer des embeddings avec sentence-transformers.
6. Creer et sauvegarder un index FAISS.
7. Rechercher les chunks les plus pertinents pour une question.
8. Generer une reponse avec Groq en s'appuyant uniquement sur le contexte recupere.
9. Citer les sources utilisees.
10. Refuser de repondre lorsque l'information n'est pas presente dans la base.

## 3. Sujet choisi

Nous avons choisi le Sujet B : Assistant Medicaments.

Ce choix est motive par plusieurs raisons :

- Les donnees sont issues d'une source officielle et structuree.
- Les notices contiennent plusieurs sections medicales utiles : indications, posologie, contre-indications, effets indesirables, interactions, mises en garde.
- Le sujet impose des contraintes interessantes autour de la fiabilite, des sources et de la prudence medicale.
- Le cas d'usage est adapte a une demonstration claire du fonctionnement d'un RAG.

## 4. Avertissement medical

Ce projet est un outil pedagogique.

Il ne fournit pas de diagnostic medical et ne remplace pas l'avis d'un medecin, d'un pharmacien ou d'un professionnel de sante.

Chaque reponse generee par le systeme doit inclure la phrase suivante :

```text
Ces informations ne remplacent pas l'avis d'un professionnel de sante.
```

## 5. Architecture du projet

```text
RAG_Medicaments/
├── app.py
├── indexation.py
├── rag.py
├── context.txt
├── requirements.txt
├── README.md
├── AUDIT.md
├── CHANGELOG.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── CIS_RCP_export.xlsx
├── storage/
│   ├── index.faiss
│   └── chunks.json
└── env/
```

## 6. Role des fichiers principaux

### indexation.py

Ce script construit la base vectorielle.

Il realise les etapes suivantes :

1. Chargement du fichier Excel `data/CIS_RCP_export.xlsx`.
2. Filtrage des medicaments cibles pour le TP.
3. Nettoyage des sections medicales.
4. Decoupage des textes en chunks.
5. Ajout de metadonnees : medicament, section, code CIS, source.
6. Calcul des embeddings avec le modele `paraphrase-multilingual-mpnet-base-v2`.
7. Creation d'un index FAISS avec `IndexFlatIP`.
8. Sauvegarde de l'index dans `storage/index.faiss`.
9. Sauvegarde des chunks et metadonnees dans `storage/chunks.json`.

### rag.py

Ce script permet d'interroger le RAG en ligne de commande.

Il realise les etapes suivantes :

1. Chargement de l'index FAISS.
2. Chargement des chunks et metadonnees.
3. Chargement du modele d'embedding.
4. Encodage de la question utilisateur.
5. Recherche des chunks les plus proches dans FAISS.
6. Construction du contexte.
7. Chargement du prompt systeme depuis `context.txt`.
8. Appel au modele Groq.
9. Affichage de la reponse et des sources.

### app.py

Ce fichier fournit une interface web Streamlit.

Il permet de poser des questions dans une interface de type chat et d'afficher :

- la reponse generee ;
- les sources utilisees ;
- le medicament concerne ;
- la section de la notice ;
- le code CIS ;
- le score de similarite.

### context.txt

Ce fichier contient le prompt systeme utilise par le modele de langage.

Il a ete externalise afin de pouvoir modifier le comportement de l'assistant sans modifier le code Python.

### AUDIT.md

Ce fichier contient l'audit du projet, avec les points forts, les limites identifiees et les ameliorations appliquees.

### CHANGELOG.md

Ce fichier resume les principales modifications realisees pendant l'amelioration du projet.

## 7. Donnees utilisees

Le fichier utilise est :

```text
data/CIS_RCP_export.xlsx
```

Il contient des informations issues des resumes des caracteristiques des produits, notamment :

- code CIS ;
- denomination du medicament ;
- composition ;
- forme pharmaceutique ;
- indications ;
- posologie ;
- contre-indications ;
- mises en garde ;
- interactions ;
- grossesse et allaitement ;
- effets indesirables ;
- surdosage ;
- conditions de prescription.

Le fichier source n'est pas destine a etre versionne sur GitHub afin d'eviter d'ajouter un fichier volumineux au depot.

En revanche, les fichiers generes dans `storage/` peuvent etre versionnes pour permettre le deploiement de l'interface Streamlit sans reindexer les donnees.

## 8. Medicaments cibles

Pour conserver un corpus raisonnable et coherent avec le sujet du TP, le script filtre une liste de medicaments courants, parmi lesquels :

- Doliprane ;
- Dafalgan ;
- Efferalgan ;
- Ibuprofene ;
- Advil ;
- Nurofen ;
- Aspirine ;
- Aspegic ;
- Amoxicilline ;
- Augmentin ;
- Smecta ;
- Imodium ;
- Ventoline ;
- Becotide ;
- Omeprazole ;
- Inexium ;
- Metformine ;
- Glucophage.

## 9. Choix techniques

### Modele d'embedding

Le modele utilise est :

```text
paraphrase-multilingual-mpnet-base-v2
```

Ce choix est adapte car les donnees et les questions sont en francais. Le modele est multilingue et produit des embeddings de dimension 768.

### Index FAISS

L'index utilise est :

```python
faiss.IndexFlatIP(dimension)
```

Les embeddings sont normalises avec `normalize_embeddings=True`.

Ainsi, le produit scalaire utilise par `IndexFlatIP` se comporte comme une similarite cosinus. Plus le score est eleve, plus le chunk est considere comme pertinent.

### Chunking

Le chunking est effectue par section medicale.

Au lieu d'indexer une notice complete en un seul bloc, chaque section est traitee separement :

- posologie ;
- indications ;
- contre-indications ;
- interactions ;
- effets indesirables ;
- mises en garde ;
- etc.

Cette approche permet au systeme de retrouver directement la section la plus pertinente selon la question.

Apres audit, le chunking a ete ameliore : le contenu medical est d'abord decoupe, puis le prefixe contenant le medicament, le code CIS et la section est ajoute a chaque chunk. Cela permet a chaque morceau de rester correctement contextualise.

### Prompt systeme

Le prompt systeme est stocke dans :

```text
context.txt
```

Il impose au modele :

- de repondre uniquement a partir du contexte fourni ;
- de ne pas inventer ;
- de citer les sources ;
- de preciser le medicament et la section ;
- d'ajouter l'avertissement medical obligatoire ;
- de refuser si l'information n'est pas presente dans la base.

### Garde-fou de confiance

Un seuil minimal de similarite est utilise :

```python
SCORE_MIN_ACCEPTABLE = 0.25
```

Si le meilleur score est inferieur a ce seuil, le systeme refuse de repondre sans appeler Groq. Cette decision limite les hallucinations et rend le comportement plus deterministe.

## 10. Installation locale

### 10.1. Cloner le projet

```bash
git clone https://github.com/Steve-Landry-NONO/rag-medicaments.git
cd rag-medicaments
```

### 10.2. Creer un environnement virtuel

```bash
python -m venv env
```

Activation sous Linux ou Mac :

```bash
source env/bin/activate
```

Activation sous Windows :

```bash
env\\Scripts\\activate
```

### 10.3. Installer les dependances

```bash
pip install -r requirements.txt
```

### 10.4. Configurer la cle Groq

Creer un fichier `.env` a la racine du projet :

```env
GROQ_API_KEY=votre_cle_groq
```

Le fichier `.env` ne doit jamais etre envoye sur GitHub.

## 11. Utilisation

### 11.1. Generer l'index FAISS

Avant la premiere utilisation, lancer :

```bash
python indexation.py
```

Cette commande genere :

```text
storage/index.faiss
storage/chunks.json
```

### 11.2. Lancer le RAG en ligne de commande

```bash
python rag.py
```

Exemple de question :

```text
Quels sont les effets indesirables de l'ibuprofene ?
```

Pour quitter :

```text
q
```

### 11.3. Mode debug

Le mode debug permet de voir les chunks recuperes sans appeler Groq :

```text
/debug Quels sont les effets indesirables de l'ibuprofene ?
```

Ce mode est utile pour verifier la qualite de la recherche vectorielle.

### 11.4. Lancer l'interface Streamlit

```bash
python -m streamlit run app.py
```

L'application est ensuite accessible localement a l'adresse :

```text
http://localhost:8501
```

## 12. Deploiement Streamlit Cloud

Pour deployer l'interface :

1. Creer un depot GitHub.
2. Pousser les fichiers du projet.
3. Aller sur Streamlit Community Cloud.
4. Connecter le depot GitHub.
5. Choisir `app.py` comme fichier principal.
6. Ajouter la cle Groq dans les secrets Streamlit :

```toml
GROQ_API_KEY = "votre_cle_groq"
```

7. Lancer le deploiement.

Pour que l'application fonctionne sans reindexation sur Streamlit Cloud, les fichiers suivants doivent etre presents dans le depot :

```text
storage/index.faiss
storage/chunks.json
```

Le fichier `.env` et l'environnement virtuel `env/` ne doivent pas etre versionnes.

## 13. Exemples de questions

Quelques exemples de questions testees ou prevues :

```text
Quelle est la posologie du Doliprane adulte ?
```

```text
Quels sont les effets indesirables de l'ibuprofene ?
```

```text
Y a-t-il des interactions avec l'ibuprofene ?
```

```text
Quelles sont les contre-indications de l'aspirine ?
```

```text
Quels sont les effets secondaires du medicament Xyzimaginaire ?
```

La derniere question sert a tester le comportement hors perimetre. Le systeme doit refuser ou signaler que l'information n'est pas presente dans sa base.

## 14. Reponses aux questions de reflexion du sujet

### Q1. Les notices sont longues et denses. Quelle strategie de chunking adopter ?

Nous avons choisi un chunking par section medicale. Chaque section d'une notice est traitee separement, puis decoupee si elle est trop longue.

Cette approche est plus pertinente qu'un decoupage arbitraire sur toute la notice, car les questions portent souvent sur une section precise : posologie, effets indesirables, interactions ou contre-indications.

### Q2. Peut-on exploiter la structure des notices ?

Oui. Le fichier contient deja des colonnes structurees correspondant aux principales sections medicales.

Nous utilisons ces colonnes comme metadonnees et comme unites de decoupage. Cela permet de retrouver plus facilement une information specifique et de citer la section utilisee dans la reponse.

### Q3. Comment distinguer les chunks relatifs aux effets secondaires de ceux relatifs a la posologie ?

Chaque chunk contient des metadonnees :

```json
{
  "medicament": "IBUPROFENE SANDOZ 200 mg, comprime enrobe",
  "section": "effets_indesirables",
  "code_cis": "60129665",
  "source": "CIS_RCP_export.xlsx"
}
```

La section est egalement ajoutee dans le texte du chunk afin d'ameliorer la recherche semantique.

### Q4. Comment gerer une question sur deux medicaments ?

Le systeme effectue une recherche vectorielle globale dans l'ensemble des chunks. Si la question mentionne deux medicaments, la recherche peut recuperer des chunks correspondant aux deux medicaments.

Le modele Groq recoit ensuite ces chunks dans le contexte et peut produire une synthese comparative, a condition que les informations soient presentes dans la base.

### Q5. Comment formuler le prompt pour etre informatif et prudent ?

Le prompt systeme impose plusieurs regles :

- ne repondre qu'a partir du contexte ;
- ne pas inventer ;
- signaler l'absence d'information ;
- citer les sources ;
- inclure l'avertissement medical obligatoire ;
- conseiller de consulter un professionnel de sante en cas de doute.

## 15. Limites du projet

Le projet presente plusieurs limites :

- Le corpus est limite a une selection de medicaments courants.
- Les reponses dependent de la qualite des chunks recuperes.
- Le systeme ne comprend pas reellement la situation medicale personnelle de l'utilisateur.
- Il ne remplace pas une consultation medicale.
- Le seuil de similarite peut parfois refuser une reponse meme si une information partielle existe dans la base.
- Le modele Groq peut reformuler de maniere imparfaite le contenu fourni.

## 16. Ameliorations possibles

Plusieurs ameliorations peuvent etre envisagees :

- Ajouter un mode comparaison entre deux medicaments.
- Ajouter une recherche hybride combinant recherche vectorielle et recherche par mots-cles.
- Ajouter un filtre explicite par medicament.
- Ajouter un historique de conversation.
- Ameliorer la detection des questions hors perimetre.
- Ajouter des tests unitaires sur le chunking et la recherche.
- Ajouter une page d'administration pour visualiser les chunks indexes.
- Utiliser une base vectorielle plus avancee si le corpus devient plus volumineux.

## 17. Conformite avec les contraintes du TP

| Contrainte | Statut |
|---|---|
| Pas de LangChain | Respecte |
| Pas de LlamaIndex | Respecte |
| FAISS persistant | Respecte |
| Embeddings avec sentence-transformers | Respecte |
| Utilisation de Groq | Respecte |
| Reponses avec sources | Respecte |
| Refus si information absente | Respecte |
| Avertissement medical obligatoire | Respecte |
| Interface bonus | Realisee avec Streamlit |

## 18. Commandes utiles

Installation :

```bash
pip install -r requirements.txt
```

Indexation :

```bash
python indexation.py
```

Mode terminal :

```bash
python rag.py
```

Interface web :

```bash
python -m streamlit run app.py
```

Verification des fichiers generes :

```bash
ls storage
```

## 19. Auteurs

Projet realise par :

- Steve Landry KOUOKAM NONO
- Ludovic TUEKAM
