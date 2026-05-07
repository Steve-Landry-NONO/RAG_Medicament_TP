# -*- coding: utf-8 -*-
"""
rag.py

Script d'interrogation pour le TP RAG - Assistant Médicaments.

Objectif :
- Charger l'index FAISS créé par indexation.py
- Charger les chunks + métadonnées
- Recevoir une question utilisateur
- Rechercher les chunks les plus pertinents
- Construire un prompt contrôlé (chargé depuis context.txt)
- Appeler Groq pour générer une réponse prudente avec sources

Contraintes TP :
- Pas de LangChain
- Pas de LlamaIndex
- Réponse basée uniquement sur le contexte récupéré
- Mention médicale obligatoire
- Refus si l'information n'est pas dans la base
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from groq import (
    Groq,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
    AuthenticationError,
    APIStatusError,
)
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Configuration générale
# ============================================================

STORAGE_DIR = Path("storage")
INDEX_PATH = STORAGE_DIR / "index.faiss"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"

# Prompt système externalisé : modifiable sans toucher au code Python.
CONTEXT_PATH = Path("context.txt")

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Modèle Groq :
# - llama-3.1-8b-instant : rapide, suffisant pour le TP
# - llama3-70b-8192 : meilleur, mais plus lent
GROQ_MODEL = "llama-3.1-8b-instant"

# Nombre de chunks à récupérer.
TOP_K = 5

# Seuil de confiance minimal.
# Avec IndexFlatIP + embeddings normalisés, le score est une similarité cosinus.
# En dessous de ce seuil, le refus est immédiat côté Python (Option A : refus strict)
# pour éviter toute hallucination du LLM sur un contexte non pertinent.
SCORE_MIN_ACCEPTABLE = 0.25

AVERTISSEMENT_MEDICAL = (
    "Ces informations ne remplacent pas l'avis d'un professionnel de santé."
)


# ============================================================
# 2. Chargement de la base vectorielle
# ============================================================

def charger_index_et_chunks() -> tuple[faiss.Index, list[dict]]:
    """
    Charge l'index FAISS et les chunks sauvegardés par indexation.py.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {INDEX_PATH}\n"
            "Lance d'abord : python indexation.py"
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier chunks introuvable : {CHUNKS_PATH}\n"
            "Lance d'abord : python indexation.py"
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)

    if index.ntotal != len(chunks_avec_meta):
        raise ValueError(
            "Incohérence entre l'index FAISS et chunks.json : "
            f"{index.ntotal} vecteurs FAISS pour {len(chunks_avec_meta)} chunks."
        )

    return index, chunks_avec_meta


def charger_modele_embedding() -> SentenceTransformer:
    """
    Charge le même modèle d'embedding que celui utilisé dans indexation.py.
    """
    print(f"[INFO] Chargement du modèle d'embedding : {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def charger_client_groq() -> Groq:
    """
    Charge la clé Groq depuis le fichier .env puis initialise le client.
    """
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Clé GROQ_API_KEY introuvable.\n"
            "Crée un fichier .env à la racine du projet avec :\n"
            "GROQ_API_KEY=ta_cle_api_groq"
        )

    return Groq(api_key=api_key)


# ============================================================
# 3. Chargement du prompt système
# ============================================================

def charger_prompt_systeme() -> str:
    """
    Charge le prompt système depuis context.txt.

    Le fichier peut contenir le placeholder {AVERTISSEMENT_MEDICAL},
    remplacé dynamiquement via str.format(). Ce mécanisme a été retenu
    car il est lisible dans context.txt par un non-développeur et ne
    nécessite aucune dépendance supplémentaire.

    Erreurs gérées :
    - Fichier absent → FileNotFoundError explicite
    - Fichier vide   → ValueError explicite
    - Placeholder inconnu → ValueError explicite (évite un KeyError silencieux)
    """
    if not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de prompt introuvable : {CONTEXT_PATH}\n"
            "Crée un fichier context.txt à la racine du projet."
        )

    contenu = CONTEXT_PATH.read_text(encoding="utf-8").strip()

    if not contenu:
        raise ValueError(
            f"Le fichier {CONTEXT_PATH} est vide. "
            "Il doit contenir le prompt système de l'assistant."
        )

    try:
        # {AVERTISSEMENT_MEDICAL} est le seul placeholder supporté.
        return contenu.format(AVERTISSEMENT_MEDICAL=AVERTISSEMENT_MEDICAL)
    except KeyError as e:
        raise ValueError(
            f"Placeholder inconnu dans context.txt : {e}\n"
            "Seul {{AVERTISSEMENT_MEDICAL}} est supporté comme placeholder."
        ) from e


# ============================================================
# 4. Recherche vectorielle
# ============================================================

def rechercher(
    question: str,
    modele: SentenceTransformer,
    index: faiss.Index,
    chunks_avec_meta: list[dict],
    k: int = TOP_K,
) -> list[dict]:
    """
    Recherche les k chunks les plus pertinents pour une question.

    Comme l'indexation a été faite avec des embeddings normalisés et IndexFlatIP,
    les scores FAISS correspondent à une similarité cosinus approximative.
    Plus le score est élevé, plus le chunk est pertinent.
    """
    if not question.strip():
        return []

    vecteur_question = modele.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    vecteur_question = np.asarray(vecteur_question, dtype=np.float32)

    scores, indices = index.search(vecteur_question, k)

    resultats = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks_avec_meta[int(idx)].copy()
        chunk["score"] = float(score)
        chunk["rang_faiss"] = int(idx)
        resultats.append(chunk)

    return resultats


# ============================================================
# 5. Construction du prompt utilisateur
# ============================================================

def formatter_source(chunk: dict, numero: int) -> str:
    """
    Formate les métadonnées d'un chunk pour les présenter au LLM.
    """
    meta = chunk.get("metadata", {})

    medicament = meta.get("medicament", "Médicament inconnu")
    section = meta.get("section", "section inconnue")
    code_cis = meta.get("code_cis", "code CIS inconnu")
    source = meta.get("source", "source inconnue")
    score = chunk.get("score", 0.0)

    return (
        f"[SOURCE {numero}]\n"
        f"Médicament : {medicament}\n"
        f"Section : {section}\n"
        f"Code CIS : {code_cis}\n"
        f"Fichier source : {source}\n"
        f"Score de similarité : {score:.3f}\n"
        f"Contenu :\n{chunk.get('contenu', '')}"
    )


def construire_contexte(chunks_pertinents: list[dict]) -> str:
    """
    Assemble les chunks récupérés en un contexte structuré pour le LLM.
    Chaque source est délimitée par une ligne de tirets.
    """
    blocs = [formatter_source(chunk, i) for i, chunk in enumerate(chunks_pertinents, start=1)]

    # join() place le séparateur ENTRE chaque bloc, pas avant le premier.
    separateur = "\n\n" + "-" * 80 + "\n\n"
    return separateur.join(blocs)


def construire_prompt_utilisateur(question: str, chunks_pertinents: list[dict]) -> str:
    """
    Construit le message utilisateur envoyé au LLM.
    """
    contexte = construire_contexte(chunks_pertinents)

    return f"""
QUESTION UTILISATEUR :
{question}

CONTEXTE DISPONIBLE :
{contexte}

Consigne :
Réponds à la question uniquement avec les informations du contexte.
Si le contexte ne permet pas de répondre, dis-le explicitement.
N'oublie pas les sources et l'avertissement médical obligatoire.
""".strip()


# ============================================================
# 6. Génération avec Groq
# ============================================================

def generer_reponse(
    question: str,
    chunks_pertinents: list[dict],
    client: Groq,
    prompt_systeme: str,
) -> str:
    """
    Génère la réponse finale avec Groq à partir des chunks récupérés.

    Deux garde-fous côté Python avant d'appeler Groq :
    1. Aucun chunk trouvé → refus immédiat
    2. Score trop faible  → refus immédiat (Option A : refus strict)
       Évite que le LLM hallucine sur des chunks non pertinents.
    """
    # Garde-fou 1 : aucun chunk récupéré.
    if not chunks_pertinents:
        return (
            "Je ne trouve pas cette information dans ma base de connaissances.\n\n"
            f"{AVERTISSEMENT_MEDICAL}"
        )

    meilleur_score = chunks_pertinents[0].get("score", 0.0)

    # Garde-fou 2 : score trop faible → refus strict, sans appel Groq.
    if meilleur_score < SCORE_MIN_ACCEPTABLE:
        return (
            f"Je ne trouve pas d'information suffisamment pertinente dans ma base "
            f"pour répondre à cette question "
            f"(score max = {meilleur_score:.3f}, seuil = {SCORE_MIN_ACCEPTABLE}).\n\n"
            f"{AVERTISSEMENT_MEDICAL}"
        )

    prompt_utilisateur = construire_prompt_utilisateur(question, chunks_pertinents)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": prompt_systeme},
                {"role": "user", "content": prompt_utilisateur},
            ],
            temperature=0.2,
            max_tokens=900,  # calé sur la taille max d'un chunk (~900 chars ≈ 250 tokens)
        )
    except RateLimitError:
        raise RuntimeError(
            "Quota Groq dépassé. Attends quelques secondes et réessaie."
        )
    except APIConnectionError:
        raise RuntimeError(
            "Impossible de joindre l'API Groq. Vérifie ta connexion internet."
        )
    except BadRequestError as e:
        raise RuntimeError(
            f"Requête rejetée par Groq (prompt trop long ou invalide) : {e}"
        )
    except AuthenticationError:
        raise RuntimeError(
            "Clé API Groq invalide. Vérifie GROQ_API_KEY dans ton fichier .env."
        )
    except APIStatusError as e:
        raise RuntimeError(
            f"Erreur Groq inattendue (statut {e.status_code}) : {e.message}"
        )

    texte = response.choices[0].message.content.strip()

    # Sécurité : si le modèle oublie l'avertissement, on l'ajoute.
    if AVERTISSEMENT_MEDICAL not in texte:
        texte += f"\n\n{AVERTISSEMENT_MEDICAL}"

    return texte


# ============================================================
# 7. Affichage des sources
# ============================================================

def afficher_sources(chunks_pertinents: list[dict]) -> None:
    """
    Affiche les sources récupérées, utile pour debug et transparence.
    """
    print("\n" + "=" * 70)
    print("SOURCES RÉCUPÉRÉES")
    print("=" * 70)

    for i, chunk in enumerate(chunks_pertinents, start=1):
        meta = chunk.get("metadata", {})
        print(
            f"{i}. {meta.get('medicament', 'Médicament inconnu')} | "
            f"section={meta.get('section', 'inconnue')} | "
            f"code_cis={meta.get('code_cis', 'inconnu')} | "
            f"score={chunk.get('score', 0.0):.3f}"
        )


def afficher_apercu_chunks(chunks_pertinents: list[dict], longueur: int = 280) -> None:
    """
    Affiche un aperçu des chunks retrouvés pour tester la recherche
    sans dépendre uniquement du LLM.
    """
    print("\n" + "=" * 70)
    print("APERÇU DES CHUNKS")
    print("=" * 70)

    for i, chunk in enumerate(chunks_pertinents, start=1):
        contenu = chunk.get("contenu", "").replace("\n", " ")
        apercu = contenu[:longueur] + ("..." if len(contenu) > longueur else "")
        print(f"\n[{i}] score={chunk.get('score', 0.0):.3f}")
        print(apercu)


# ============================================================
# 8. Boucle interactive
# ============================================================

def main() -> None:
    """
    Interface en ligne de commande.
    """
    print("=" * 70)
    print("RAG MÉDICAMENTS - QUESTION/RÉPONSE")
    print("=" * 70)

    # Chargement unique du prompt système au démarrage (lecture de context.txt).
    # Toute erreur ici (fichier absent, vide, placeholder invalide) est fatale.
    print(f"[INFO] Chargement du prompt système depuis : {CONTEXT_PATH}")
    prompt_systeme = charger_prompt_systeme()

    print("[INFO] Chargement de la base de connaissances...")
    index, chunks_avec_meta = charger_index_et_chunks()

    print(f"[INFO] Index FAISS chargé : {index.ntotal} vecteurs")
    print(f"[INFO] Chunks chargés : {len(chunks_avec_meta)}")

    modele = charger_modele_embedding()
    client = charger_client_groq()

    print("\n[OK] Système RAG prêt.")
    print("Tape 'quit', 'exit' ou 'q' pour quitter.")
    print("Tape '/debug ta question' pour voir les chunks récupérés sans appeler Groq.\n")

    while True:
        question = input("Votre question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        mode_debug = False
        if question.startswith("/debug "):
            mode_debug = True
            question = question.removeprefix("/debug ").strip()

        try:
            chunks_pertinents = rechercher(
                question=question,
                modele=modele,
                index=index,
                chunks_avec_meta=chunks_avec_meta,
                k=TOP_K,
            )

            if mode_debug:
                afficher_sources(chunks_pertinents)
                afficher_apercu_chunks(chunks_pertinents)
                continue

            reponse = generer_reponse(
                question=question,
                chunks_pertinents=chunks_pertinents,
                client=client,
                prompt_systeme=prompt_systeme,
            )

            print("\n" + "=" * 70)
            print("RÉPONSE")
            print("=" * 70)
            print(reponse)

            afficher_sources(chunks_pertinents)
            print()

        except Exception as e:
            print("\n[ERREUR]")
            print(str(e))
            print()


if __name__ == "__main__":
    main()
