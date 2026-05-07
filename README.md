# TP RAG Medicaments

Assistant RAG d'information sur les medicaments, realise dans le cadre du TP **Construire un RAG avec Python et Groq**.

Projet realise par :

- Steve Landry KOUOKAM NONO
- Ludovic TUEKAM

## Demonstration en ligne

Application Streamlit deployee :

https://ragmedicamenttp-nqsq37jtueuzjmkt7jajzjkg.streamlit.app

## Presentation

Ce projet implemente un systeme RAG, pour **Retrieval-Augmented Generation**, applique aux notices de medicaments issues de la base ANSM/CIS_RCP.

L'utilisateur peut poser une question en langage naturel, par exemple :

- Quelle est la posologie du Doliprane adulte ?
- Quels sont les effets indesirables de l'ibuprofene ?
- Y a-t-il des interactions avec l'ibuprofene ?
- Quelles sont les contre-indications de l'aspirine ?

Le systeme recherche les passages les plus pertinents dans une base vectorielle FAISS, puis utilise un modele Groq pour generer une reponse claire, sourcee et prudente.

## Sujet choisi

Nous avons choisi le sujet **Assistant Medicaments**.

Ce sujet est interessant car les notices sont longues, structurees et sensibles. Il oblige donc a reflechir a la qualite du chunking, a la citation des sources et a la prudence des reponses.

Chaque reponse doit rappeler :

```text
Ces informations ne remplacent pas l'avis d'un professionnel de sante.
```

## Architecture

```text
RAG_Medicament_TP/
├── app.py                  # Interface Streamlit
├── indexation.py           # Creation de la base vectorielle FAISS
├── rag.py                  # Moteur RAG en ligne de commande
├── context.txt             # Prompt systeme du modele
├── requirements.txt        # Dependances Python
├── storage/
│   ├── index.faiss         # Index vectoriel persistant
│   └── chunks.json         # Chunks et metadonnees
└── README.md
```

## Fonctionnement du pipeline

Le projet est separe en deux grandes phases.

### 1. Indexation

Le script `indexation.py` :

1. charge les donnees medicaments ;
2. nettoie les textes ;
3. filtre une liste de medicaments courants ;
4. decoupe les notices par sections medicales ;
5. cree des chunks coherents ;
6. calcule les embeddings avec `sentence-transformers` ;
7. cree un index FAISS ;
8. sauvegarde `storage/index.faiss` et `storage/chunks.json`.

### 2. Question-reponse

Le script `rag.py` :

1. charge l'index FAISS ;
2. encode la question utilisateur ;
3. recherche les chunks les plus proches ;
4. construit un contexte avec les sources ;
5. interroge Groq ;
6. affiche une reponse avec les sources utilisees.

## Choix techniques

| Element | Choix |
|---|---|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` |
| Base vectorielle | FAISS |
| Similarite | Cosinus via `IndexFlatIP` avec embeddings normalises |
| LLM | Groq |
| Interface bonus | Streamlit |
| Frameworks interdits | LangChain et LlamaIndex non utilises |

## Metadonnees des chunks

Chaque chunk conserve des metadonnees utiles pour citer les sources :

```json
{
  "medicament": "IBUPROFENE SANDOZ 200 mg",
  "section": "effets_indesirables",
  "code_cis": "60129665",
  "source": "CIS_RCP_export.xlsx"
}
```

Cela permet d'afficher le medicament, la section de notice et le code CIS dans les reponses.

## Installation locale

Cloner le projet :

```bash
git clone https://github.com/Steve-Landry-NONO/RAG_Medicament_TP.git
cd RAG_Medicament_TP
```

Creer un environnement virtuel :

```bash
python -m venv env
source env/bin/activate
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

Creer un fichier `.env` :

```env
GROQ_API_KEY=votre_cle_groq
```

## Utilisation

Lancer l'interface web :

```bash
python -m streamlit run app.py
```

Lancer le mode terminal :

```bash
python rag.py
```

Relancer l'indexation si necessaire :

```bash
python indexation.py
```

## Exemples de questions

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
Quels sont les effets secondaires du medicament Xyzimaginaire ?
```

La derniere question sert a tester le comportement hors perimetre : le systeme doit refuser ou signaler que l'information n'est pas presente dans sa base.

## Limites

Ce projet reste un TP pedagogique.

Ses principales limites sont :

- corpus limite a une selection de medicaments ;
- reponses dependantes de la qualite des chunks recuperes ;
- pas de diagnostic medical ;
- pas de prise en compte du profil patient ;
- seuil de similarite pouvant parfois refuser une reponse partiellement pertinente.

## Contraintes respectees

| Contrainte du TP | Statut |
|---|---|
| Pas de LangChain | Respecte |
| Pas de LlamaIndex | Respecte |
| FAISS persistant | Respecte |
| Reponses basees sur le contexte | Respecte |
| Sources citees | Respecte |
| Refus si information absente | Respecte |
| Avertissement medical | Respecte |
| Bonus interface | Streamlit |
